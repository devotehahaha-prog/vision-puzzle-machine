"""Seeded visual end-to-end runs; all outcomes, including failures, are retained."""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：固定种子生成多组 4 片布局，批量跑视觉+仿真，失败也写入 summary.json。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约36行  layouts：抖动位置/角度，保证不出纸、不重叠
#   - 约72行  main：逐案运行并统计通过数
# =============================================================================
# =============================================================================
# 【分区】文件说明
# 功能：用固定随机种子生成多组四片布局，逐案跑视觉+仿真，失败也写入 summary.json。
# 可修改：默认 count=10、seed=20260916、输出目录 output_sim/qa_layouts。
# 看情况改：位移 ±8mm、转角 ±20°、膨胀核 25×25 控制碎片间距；太密会检测粘连，太疏不像赛场。
# 不要改：必须 4 片才 yield；不要丢掉失败案例，本脚本就是用来留失败证据的。
# =============================================================================

import argparse
import contextlib
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np

import run_sim


# =============================================================================
# 【分区】随机合法布局生成
# 功能：在默认场景上抖动位置/角度，检查不出纸、互不重叠后产出 4 片规格。
# 可修改：count、seed、uniform(-8,8)、角度 ±20、纸面边距 5mm、膨胀核大小。
# 看情况改：occupied 画布 594×840 对应 4px/mm 的半张纸量级；改纸张尺寸要一起改。
# 不要改：失败 1000 次就抛错，避免死循环；不要去掉重叠检测。
# =============================================================================
def layouts(vision, simulator, count=10, seed=20260916):
    rng = np.random.default_rng(seed)
    specs = run_sim.make_scene(vision, simulator)
    for _ in range(count):
        for attempt in range(1000):
            candidate = []
            occupied = np.zeros((594, 840), np.uint8)
            for spec in specs:
                xy = np.array(spec.start_xy_mm) + rng.uniform(-8, 8, 2)
                angle = float(rng.uniform(-20, 20))
                local = np.array(spec.vertices_mm) - np.mean(spec.vertices_mm, axis=0)
                world = vision.rotate_points(local, angle) + xy
                paper = np.array([210., 297.]) - world
                if (paper < 5).any() or (paper > [205, 143.5]).any():
                    break
                mask = np.zeros_like(occupied)
                cv2.fillPoly(mask, [np.rint(paper * 4).astype(np.int32)], 255)
                expanded = cv2.dilate(mask, np.ones((25, 25), np.uint8))
                if np.any(expanded & occupied):
                    break
                occupied |= mask
                candidate.append(replace(spec, start_xy_mm=tuple(xy), start_yaw_deg=angle))
            if len(candidate) == 4:
                yield candidate
                break
        else:
            raise RuntimeError("unable to generate a valid separated layout")


# =============================================================================
# 【分区】批量跑案与汇总
# 功能：第 0 案用标准场景，其后为随机布局；每案独立输出目录，最后统计通过数。
# 可修改：--output-dir。
# 看情况改：无 GUI（PuzzleGantrySim(False,...)）才能在服务器上批跑。
# 不要改：失败也写入 summary；返回码 0 仅当全部 pass。
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=run_sim.ROOT / "output_sim" / "qa_layouts")
    args = parser.parse_args()
    vision, simulator, camera = run_sim.load_modules(run_sim.DEFAULT_SIM_ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    cases = [run_sim.make_scene(vision, simulator), *layouts(vision, simulator)]
    for index, specs in enumerate(cases):
        case_dir = args.output_dir / f"case_{index:02d}"
        case_dir.mkdir(exist_ok=True)
        with (case_dir / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            sim = simulator.PuzzleGantrySim(False, False, scene="vision")
            session = run_sim.VisionSession(vision, simulator, camera, sim, case_dir)
            item = {"case": index, "seed": 20260916,
                    "layout": [{"name": s.name, "xy": list(s.start_xy_mm), "yaw": s.start_yaw_deg} for s in specs]}
            try:
                sim.spawn_pieces(specs)
                item.update(session.run_once())
                final = json.loads((session.last_output / "final_verification.json").read_text())
                item["max_visual_position_error_mm"] = max(m["position_error_mm"] for m in final["matches"])
                item["max_visual_rotation_error_deg"] = max(abs(m["rotation_correction_deg"]) for m in final["matches"])
            except simulator.ExecutionError as error:
                item.update(status="failed", stage=error.stage, error=str(error))
            finally:
                item["output"] = str(session.last_output)
                sim.close()
        summary.append(item)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"CASE {index:02d}: {item['status']} {item.get('stage', '')}", flush=True)
    passed = sum(item["status"] == "pass" for item in summary)
    print(f"LAYOUT QA: {passed}/{len(summary)} passed; {args.output_dir / 'summary.json'}")
    return 0 if passed == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
