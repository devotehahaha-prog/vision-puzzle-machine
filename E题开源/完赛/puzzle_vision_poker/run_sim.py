"""Run poker-card vision, patterned seam validation, and PyBullet execution."""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：扑克仿真桥：3 片带纹理，规划后还要过花纹门禁 pattern_gate 才能执行。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约48行  load_modules：加载视觉/仿真/虚拟相机
#   - 约58行  simulation_config：把 poker_* 拷到 generic_* 再叠 sim_config
#   - 约70行  convert_xy：坐标互换
#   - 约74行  make_scene：从 layout 读 3 片顶点与纹理
#   - 约94行  tasks_from_plan：必须正好 3 个任务
#   - 约126行  pattern_gate：三片花纹缝全过阈值且连通
#   - 约159行  seam_report：目标间隙报告
#   - 约178行  PokerSession：仿真会话
#   - 约321行  main：GUI 或 --direct
# =============================================================================
# =============================================================================
# 【分区】文件说明
# 功能：扑克视觉 + 花纹缝校验 + PyBullet 执行。3 片带纹理，不是普通版 4 片纯色。
# 可修改：DEFAULT_SIM_ROOT、output_sim。
# 看情况改：simulation_config 把 poker_* 拷到 generic_*，再被 sim_config.json 覆盖。
# 不要改：A4 210×297；convert_xy；必须正好 3 个任务。
# =============================================================================

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_SIM_ROOT = ROOT.parents[2] / "pybullet" / "puzzle_device"


# =============================================================================
# 【分区】仿真模块、场景与任务转换
# 功能：加载视觉/仿真/虚拟相机；从 layout 读 3 片顶点与纹理；plan → MotionTask。
# 可修改：sim_config.json。
# 看情况改：make_scene 用白色+纹理，视觉仍只看 RGB。
# 不要改：坐标系检查；len(normalized)!=3 报错。
# =============================================================================
def load_modules(sim_root: Path):
    if not (sim_root / "simulator.py").is_file():
        raise ValueError(f"simulator.py not found in {sim_root}")
    sys.path.insert(0, str(sim_root))
    import main as vision
    import simulator
    from virtual_camera import VirtualCamera
    return vision, simulator, VirtualCamera


def simulation_config(vision):
    config = json.loads((ROOT / "sim_config.json").read_text(encoding="utf-8"))
    for name in ("target_long_min_mm", "target_long_max_mm", "target_short_min_mm",
                 "target_short_max_mm", "max_fill_error_ratio", "max_boundary_gap_ratio",
                 "edge_abs_tolerance_mm", "edge_rel_tolerance", "min_edge_length_mm",
                 "max_edge_candidates", "search_timeout_seconds", "max_search_states"):
        config[f"generic_{name}"] = config[f"poker_{name}"]
    config["generic_require_each_piece_outer_edge"] = config["poker_require_outer_edge"]
    return config


def convert_xy(point):
    return tuple(np.array([210.0, 297.0]) - np.asarray(point, dtype=float))


def make_scene(simulator):
    """Load only geometry/texture metadata; vision still sees rendered RGB."""
    layout = simulator.load_layout()
    specs = []
    for item in layout["pieces"]:
        vertices = tuple(tuple(float(v) for v in point) for point in item["vertices_mm"])
        xs, ys = zip(*vertices)
        specs.append(simulator.PieceSpec(
            name=item["name"],
            size_xy_mm=(max(xs) - min(xs), max(ys) - min(ys)),
            start_xy_mm=tuple(item["center_mm"]),
            vertices_mm=vertices,
            texture=item["texture"],
            crop_origin_px=tuple(item["crop_origin_px"]),
            texture_size_px=tuple(item["texture_size_px"]),
            color=(1.0, 1.0, 1.0, 1.0),
        ))
    return specs


def tasks_from_plan(plan, vision, simulator):
    coordinate = plan.get("coordinate_system", {})
    if (coordinate.get("origin") != "A4_top_left"
            or coordinate.get("x_axis") != "right"
            or coordinate.get("y_axis") != "down"
            or plan.get("mechanical_workspace_mm", {}).get("size") != [210.0, 297.0]):
        raise ValueError("unsupported plan coordinate system")
    normalized = vision.serial_transport.validate_motion_plan(plan)
    if len(normalized) != 3:
        raise ValueError("the poker scene requires exactly three tasks")
    tasks = []
    for piece in sorted(plan["pieces"], key=lambda item: item["move_order"]):
        path = piece.get("path_mm")
        if not isinstance(path, list) or len(path) < 1:
            raise ValueError("missing poker path_mm")
        tasks.append(simulator.MotionTask(
            *convert_xy(piece["source_pick_mm"]),
            float(piece["rotation_deg_signed"]),
            *convert_xy(piece["target_pick_mm"]),
            tuple(convert_xy(point) for point in path),
        ))
    simulator.PuzzleGantrySim.validate_tasks(tasks)
    return tasks


# =============================================================================
# 【分区】花纹门禁与拼缝间隙报告
# 功能：检查 3 片之间的 NCC/similarity 全部过阈值且连通；seam_report 量目标间隙。
# 可修改：poker_ncc_min_similarity 默认 0.55，必须在 [0.55, 1]。
# 看情况改：similarity=(NCC+1)/2，0.5 只表示无相关，不是“一半正确”。
# 不要改：三片必须都被缝连上；NCC 与 similarity 换算差 >0.001 视为伪造数据。
# =============================================================================
def pattern_gate(plan, config):
    verification = plan.get("poker_pattern_verification") or {}
    pairs = verification.get("pairs", [])
    threshold = float(config.get("poker_ncc_min_similarity", 0.55))
    reasons = []
    values = []
    ids = {piece["id"] for piece in plan.get("pieces", [])}
    connected = set()
    for pair in pairs:
        similarity = float(pair.get("similarity", float("nan")))
        ncc = float(pair.get("ncc", float("nan")))
        values.append(similarity)
        if (not np.isfinite([similarity, ncc]).all() or not -1 <= ncc <= 1
                or not 0 <= similarity <= 1 or abs(similarity - (ncc + 1) / 2) > 0.001):
            reasons.append("invalid seam NCC/similarity")
        elif similarity < threshold:
            reasons.append(f"seam {pair.get('piece_a')}-{pair.get('piece_b')} similarity {similarity:.4f} < {threshold:.4f}")
        ends = {pair.get("piece_a"), pair.get("piece_b")}
        if len(ends) != 2 or not ends <= ids:
            reasons.append("invalid seam piece IDs")
        connected.update(ends)
    minimum = float(verification.get("min_similarity", float("nan")))
    if not pairs or verification.get("n_pairs") != len(pairs) or connected != ids or len(ids) != 3:
        reasons.append("missing seam evidence connecting all three pieces")
    if not np.isfinite(minimum) or (values and abs(minimum - min(values)) > 0.001):
        reasons.append("invalid aggregate seam similarity")
    if not np.isfinite(threshold) or not 0.55 <= threshold <= 1:
        reasons.append("invalid poker similarity threshold")
    return {"minimum_similarity": minimum if np.isfinite(minimum) else None,
            "threshold": threshold, "pairs": pairs,
            "status": "failed" if reasons else "pass", "failure_reasons": reasons}


def seam_report(plan, vision):
    polygons = {item["id"]: np.asarray(item["target_polygon_mm"], dtype=float)
                for item in plan["pieces"]}
    edges = {tuple(sorted((pair["piece_a"], pair["piece_b"])))
             for pair in plan["poker_pattern_verification"]["pairs"]}
    gaps = [{"piece_a": a, "piece_b": b,
             "minimum_target_gap_mm": float(vision._polygon_closest_vector(polygons[a], polygons[b])[0])}
            for a, b in sorted(edges)]
    return {"requested_clearance_mm": 1.0, "pairs": gaps,
            "minimum_target_gap_mm": min(item["minimum_target_gap_mm"] for item in gaps)}


# =============================================================================
# 【分区】PokerSession：一次仿真会话
# 功能：park → 拍照规划 → pattern_gate → 执行 → 终检。handle_command 响应按钮。
# 可修改：park 后 step 次数。
# 看情况改：花纹不过门禁不得 dispatch。实机阈值不要直接抄仿真。
# 不要改：finally 恢复 OUTPUT_DIR；GUI 清空鼠标键盘队列。
# =============================================================================
class PokerSession:
    def __init__(self, vision, simulator, camera_class, sim, output_dir: Path):
        self.vision, self.simulator, self.sim = vision, simulator, sim
        self.config = simulation_config(vision)
        self.camera = camera_class(sim.cid, float(self.config["pixels_per_mm"]))
        self.output_dir = output_dir
        self.last_output = None

    def park(self):
        import pybullet as p
        from config import CFG, MM
        if self.sim.fault:
            raise self.sim.fault
        if self.sim.held_body is not None:
            raise RuntimeError("cannot capture while holding a card")
        q = [p.getJointState(self.sim.robot, self.sim.joint[name])[0]
             for name in ("x_joint", "y_joint", "theta_joint")]
        self.sim.set_pose(float(np.clip(q[0] / MM, 0, CFG.a4_x_mm)),
                          float(np.clip(q[1] / MM, 0, CFG.a4_y_mm)), CFG.z_high_mm,
                          float(np.clip(np.degrees(q[2]), 0, 180)))
        self.sim.set_pose(0, 0, CFG.z_high_mm, 0)
        for _ in range(60):
            self.sim.step()

    def run_once(self):
        if self.sim.fault:
            raise self.sim.fault
        out = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out.mkdir(parents=True, exist_ok=False)
        self.last_output = out
        previous_output = self.vision.OUTPUT_DIR
        self.vision.OUTPUT_DIR = out
        report = {"status": "failed", "stage": "capture"}
        try:
            self.sim._set_status("CAPTURING")
            self.park()
            frame = self.camera.capture()
            self.vision.write_image(out / "original.jpg", frame)
            report["stage"] = "planning"
            self.sim._set_status("PLANNING")
            paper = self.vision.warp_paper(frame, self.camera.matrix, self.config)
            detected, mask = self.vision.detect_pieces(paper, self.config)
            self.vision.write_image(out / "corrected.jpg", paper)
            self.vision.write_image(out / "mask.png", mask)
            if len(detected) != 3:
                raise RuntimeError(f"expected 3 cards, detected {len(detected)}")
            _, paths = self.vision.process_paper(frame, paper, self.config, False)
            plan_path = out / "plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if any(item.match.relaxed_override for item in paths):
                reason = "relaxed or fallback poker solution is not executable"
                plan["ready_for_motion"] = False
                plan.setdefault("failure_reasons", []).append(reason)
                plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
                raise RuntimeError(reason)
            report["stage"] = "pattern_gate"
            pattern = pattern_gate(plan, self.config)
            (out / "pattern_gate.json").write_text(json.dumps(pattern, indent=2), encoding="utf-8")
            report["pattern_gate"] = pattern
            if pattern["status"] != "pass":
                plan["ready_for_motion"] = False
                plan.setdefault("failure_reasons", []).extend(pattern["failure_reasons"])
                plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
                raise RuntimeError("; ".join(pattern["failure_reasons"]))
            tasks = tasks_from_plan(plan, self.vision, self.simulator)
            report["seams"] = seam_report(plan, self.vision)
            report["stage"] = "execution"
            self.sim._set_status("RUNNING")
            results = self.sim.run_queue(tasks, final_status="VERIFYING")
            (out / "execution_results.json").write_text(
                json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8")
            report["stage"] = "verification"
            self.park()
            final = self.camera.capture()
            # A three-pixel closing can bridge a one-millimetre seam. Final
            # verification uses raw brightness components, without merging them.
            verify_config = dict(self.config, morphology_kernel_mm=0.0, poker_morph_close_mm=0.0)
            verification = self.vision.verify_final_frame(final, self.camera.matrix, verify_config, plan_path)
            report["verification"] = {
                "status": verification["status"],
                "max_position_error_mm": max((item["position_error_mm"] for item in verification["matches"]), default=0.0),
                "max_rotation_error_deg": max((abs(item["rotation_correction_deg"]) for item in verification["matches"]), default=0.0),
                "detected_piece_count": verification["detected_piece_count"],
                "expected_piece_count": verification["expected_piece_count"],
            }
            verification.update(report["verification"])
            verification["target_seams"] = report["seams"]
            verification["pattern_gate"] = pattern
            (out / "final_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
            if verification["status"] != "pass":
                raise RuntimeError("final poker visual verification failed")
            report.update(status="pass", stage="done")
            self.sim._set_status("DONE - POKER VISION PASS")
            return report
        except Exception as error:
            report["error"] = str(error)
            fault = error if isinstance(error, self.simulator.ExecutionError) else self.simulator.ExecutionError(
                "POKER_VISION_FAILED", 0, report["stage"], str(error))
            if self.sim.fault is not fault:
                self.sim._latch_fault(fault)
            if fault is error:
                raise
            raise fault from error
        finally:
            self.vision.OUTPUT_DIR = previous_output
            if self.sim.gui:
                import pybullet as p
                if p.isConnected(self.sim.cid):
                    p.getMouseEvents(); p.getKeyboardEvents()
                    self.sim.pending_command = None
                    self.sim.drag_body = None
                    self.sim.left_down = False
            if report["stage"] == "execution":
                (out / "execution_results.json").write_text(
                    json.dumps([asdict(item) for item in self.sim.results], indent=2), encoding="utf-8")
            (out / "run_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            (out / "effective_config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
            print(f"SIM RESULT: {report['status']} ({report['stage']}); output: {out}")

    def handle_command(self, command):
        if command == "quit":
            if self.sim.fault:
                raise self.sim.fault
            return False
        try:
            if command == "home":
                self.sim.home()
            elif command == "rotate" and not self.sim.fault:
                self.sim.rotate_selected(15)
            elif command == "start" and not self.sim.fault:
                self.run_once()
        except self.simulator.ExecutionError:
            pass
        return True


# =============================================================================
# 【分区】命令行入口
# 功能：--direct 无界面一次；--auto-start 打开 GUI 后立刻跑一轮再退出；默认 GUI 循环。
# 可修改：--sim-root、--output-dir。
# 看情况改：sleep 0.005。
# 不要改：finally 关仿真和窗口。
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--auto-start", action="store_true",
                        help="Open the GUI, run one closed-loop immediately, then exit")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output_sim")
    args = parser.parse_args()
    vision, simulator, camera_class = load_modules(args.sim_root.resolve())
    sim = simulator.PuzzleGantrySim(gui=not args.direct,
                                    realtime=not args.direct and not args.no_realtime,
                                    scene="vision")
    try:
        sim.spawn_pieces(make_scene(simulator))
        session = PokerSession(vision, simulator, camera_class, sim, args.output_dir.resolve())
        if args.auto_start and sim.gui:
            for _ in range(24):
                sim.step()
        if args.direct or args.auto_start:
            session.run_once()
        else:
            import pybullet as p
            shown_output = None
            while p.isConnected(sim.cid):
                sim._handle_piece_mouse()
                command = sim.poll_buttons()
                if not session.handle_command(command):
                    break
                if session.last_output and session.last_output != shown_output:
                    image_path = session.last_output / "final_target_overlay.jpg"
                    if not image_path.exists():
                        image_path = session.last_output / "original.jpg"
                    preview = vision.read_image(image_path)
                    if preview is not None:
                        cv2.namedWindow("Poker vision result", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Poker vision result", 420, 594)
                        cv2.imshow("Poker vision result", preview)
                    shown_output = session.last_output
                cv2.waitKey(1); sim.step(); time.sleep(0.005)
            if sim.fault:
                raise sim.fault
    finally:
        sim.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"SIM FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
