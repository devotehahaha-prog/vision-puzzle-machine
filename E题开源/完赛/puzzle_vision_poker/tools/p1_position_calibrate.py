"""使用单个P1碎片自动标定XY搬运位置。

默认只进行视觉检查；加入--execute --yes才会驱动COM口。
"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：扑克目录下的单片 XY 标定，流程与普通版相同；默认干跑，--execute --yes 才动电机。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约69行  build_targets：默认目标点
#   - 约121行  run：主流程
#   - 约268行  build_parser：串口与双确认开关
# =============================================================================
# 单片XY标定工具：识别P1→生成目标点→逐次搬运并拍照→统计命令位移和实测质心位移→拟合位置模型。
# 复用主程序检测和旋转标定工具的Controller；ROOT取tools目录的上一级，确保从命令行运行也能导入项目模块。
# 默认只做视觉检查和打印；原有程序要求同时提供--execute与--yes才进入真实运动分支。
# 落点误差比较实际吸取点与目标吸取点；位移模型使用前后质心位移。两类误差用途不同。
# =============================================================================
# 【分区】导入与路径
# 功能：tools 上一级入 path；结果写 output/position_calibration。扑克版流程同普通版。
# 可修改：OUTPUT 目录名。
# 看情况改：不要改 ROOT=parents[1]。
# 不要改：先插 path 再 import main。
# =============================================================================
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
import position_calibration
import rotation_calibration
import serial_transport
from p1_rotation_calibrate import Controller, capture_stable, detect_single_piece

OUTPUT = ROOT / "output" / "position_calibration"


# =============================================================================
# 【分区】检测存图与默认目标点
# 功能：每次试验存图；build_targets 给出安全点列。
# 可修改：目标点范围，碎片大时往里收。
# 看情况改：自定义 --targets，不要贴 parse_targets 安全框边缘。
# 不要改：叠加图颜色约定。
# =============================================================================
def save_detection(index: int, phase: str, frame: np.ndarray, paper: np.ndarray,
                   mask: np.ndarray, piece: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    overlay = paper.copy()
    cv2.drawContours(overlay, [piece.contour.astype(np.int32)], -1, (0, 0, 255), 2)
    cv2.circle(overlay, tuple(np.rint(piece.pick_px).astype(int)), 6, (0, 255, 0), -1)
    main.write_image(OUTPUT / f"{index:02d}_{phase}_frame.jpg", frame)
    main.write_image(OUTPUT / f"{index:02d}_{phase}_paper.jpg", paper)
    main.write_image(OUTPUT / f"{index:02d}_{phase}_mask.png", mask)
    main.write_image(OUTPUT / f"{index:02d}_{phase}_overlay.jpg", overlay)


def build_targets(initial_pick: tuple[float, float]) -> list[tuple[float, float]]:
    """覆盖正反X、正反Y和双向对角线，同时避开A4边界。

    覆盖范围比原来的65~150/95~220（85mm/125mm跨度）拉宽，但不再逼近
    parse_targets的安全边界(35~175, 45~245)——吸取点现在优先取碎片
    质心，对于长条形碎片，质心到最远顶点可能有50mm以上，如果把质心
    本身放到接近边界，碎片的一端会被推出A4纸或落入边框清理区而被
    误判为"贴边"。这里预留约20mm的碎片自身半径安全余量。
    """
    _, initial_y = initial_pick
    y_middle = float(np.clip(initial_y, 100.0, 125.0))
    x_left, x_right = 45.0, 155.0
    y_top, y_bottom = 60.0, 225.0
    return [
        (x_right, y_middle),
        (x_left, y_middle),
        (x_right, y_middle),
        (x_right, y_bottom),
        (x_right, y_top),
        (x_right, y_bottom),
        (x_left, y_top),
        (x_right, y_bottom),
    ]


# =============================================================================
# 【分区】结果落盘
# 功能：result.json + samples.csv。不自动写回 config。
# 可修改：文件名。
# 看情况改：确认拟合后再手工拷贝矩阵。
# 不要改：CSV utf-8-sig。
# =============================================================================
def write_results(samples: list[dict[str, Any]], fit: dict[str, Any]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "result.json").write_text(
        json.dumps({"samples": samples, "fit": fit}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (OUTPUT / "samples.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0].keys()))
        writer.writeheader()
        writer.writerows(samples)


# 串联单片检测、逐目标位移试验、视觉复测和位置拟合，保存结果并释放相机与串口。
# =============================================================================
# 【分区】主流程 run()
# 功能：识别单片 → 干跑或逐点搬运 → 视觉测位移 → 拟合。
# 可修改：等待时间走 config；--camera-settle。
# 看情况改：拟合用质心差，落点误差用吸取点，不要混用。
# 不要改：必须 --execute --yes；异常 safe close；finally 释放相机。
# =============================================================================
def run(args: argparse.Namespace) -> int:
    config = main.load_config()
    camera = main.open_camera(config, args.camera_id)
    matrix = main.load_calibration(config)
    if matrix is None:
        camera.release()
        raise RuntimeError("缺少calibration.json，请先完成A4标定")

    controller = None
    samples: list[dict[str, Any]] = []
    try:
        initial_frame = capture_stable(camera)
        current_piece, paper, mask = detect_single_piece(initial_frame, matrix, config)
        save_detection(0, "initial", initial_frame, paper, mask, current_piece)
        ppm = float(config["pixels_per_mm"])
        initial_pick = current_piece.pick_mm(ppm)
        targets = args.targets if args.targets is not None else build_targets(initial_pick)
        print(f"[视觉] P1初始吸取点=({initial_pick[0]:.2f},{initial_pick[1]:.2f})mm")
        print("[计划] 无旋转目标序列=" + " -> ".join(f"({x:.1f},{y:.1f})" for x, y in targets))

        if not args.execute:
            print("[DRY-RUN] 视觉检查通过，未打开串口、未驱动电机。")
            return 0
        if not args.yes:
            raise RuntimeError("真实执行必须同时指定--yes")

        offset_x, offset_y = map(float, config.get("motion_target_offset_mm", [0.0, 0.0]))
        xy_command_matrix = config.get("motion_xy_command_matrix", [[1.0, 0.0], [0.0, 1.0]])
        xy_command_bias = config.get("motion_xy_command_bias_mm", [0.0, 0.0])
        pickup_wait = float(config.get("magnet_pickup_settle_seconds", 0.8))
        release_wait = float(config.get("magnet_release_settle_seconds", 0.3))
        controller = Controller(args.port, args.baudrate, args.timeout)
        controller.command("PING")
        controller.command("MOTION,1")

        for index, target_pick in enumerate(targets, 1):
            source_pick = np.asarray(current_piece.pick_mm(ppm), np.float64)
            source_center = np.asarray(current_piece.center_mm(ppm), np.float64)
            target = np.asarray(target_pick, np.float64)
            command_delta = target - source_pick
            corrected_delta = position_calibration.corrected_command_delta(
                command_delta, xy_command_matrix, xy_command_bias
            )
            source_command = source_pick + np.asarray([offset_x, offset_y], np.float64)
            target_command = source_command + corrected_delta
            print(
                f"\n[测试 {index}/{len(targets)}] 源=({source_pick[0]:.2f},{source_pick[1]:.2f})mm，"
                f"目标=({target[0]:.2f},{target[1]:.2f})mm，"
                f"期望位移=({command_delta[0]:+.2f},{command_delta[1]:+.2f})mm，"
                f"补偿发送=({corrected_delta[0]:+.2f},{corrected_delta[1]:+.2f})mm"
            )
            controller.command(f"GOTO,{source_command[0]:.2f},{source_command[1]:.2f}")
            controller.command("Z,DOWN")
            controller.command("MAGNET,1", pickup_wait)
            controller.command("Z,UP")
            controller.command(f"GOTO,{target_command[0]:.2f},{target_command[1]:.2f}")
            controller.command("Z,DOWN")
            controller.command("MAGNET,0", release_wait)
            controller.command("Z,UP")
            controller.command("GOTO,0.00,0.00")
            time.sleep(args.camera_settle)

            after_frame = capture_stable(camera)
            after_piece, after_paper, after_mask = detect_single_piece(after_frame, matrix, config)
            save_detection(index, "after", after_frame, after_paper, after_mask, after_piece)
            after_pick = np.asarray(after_piece.pick_mm(ppm), np.float64)
            after_center = np.asarray(after_piece.center_mm(ppm), np.float64)
            actual_delta = after_center - source_center
            error = actual_delta - command_delta
            rotation_deg, iou = rotation_calibration.estimate_relative_rotation(
                current_piece.contour,
                after_piece.contour,
                0.0,
                search_radius_deg=args.rotation_search_radius,
            )
            sample = {
                "index": index,
                "source_pick_x_mm": round(float(source_pick[0]), 4),
                "source_pick_y_mm": round(float(source_pick[1]), 4),
                "target_pick_x_mm": round(float(target[0]), 4),
                "target_pick_y_mm": round(float(target[1]), 4),
                "actual_pick_x_mm": round(float(after_pick[0]), 4),
                "actual_pick_y_mm": round(float(after_pick[1]), 4),
                "command_dx_mm": round(float(command_delta[0]), 4),
                "command_dy_mm": round(float(command_delta[1]), 4),
                "actual_dx_mm": round(float(actual_delta[0]), 4),
                "actual_dy_mm": round(float(actual_delta[1]), 4),
                "error_dx_mm": round(float(error[0]), 4),
                "error_dy_mm": round(float(error[1]), 4),
                "placement_error_x_mm": round(float(after_pick[0] - target[0]), 4),
                "placement_error_y_mm": round(float(after_pick[1] - target[1]), 4),
                "placement_error_mm": round(float(np.linalg.norm(after_pick - target)), 4),
                "rotation_drift_deg": round(float(rotation_deg), 4),
                "shape_iou": round(float(iou), 5),
                "valid": bool(iou >= args.min_iou and np.linalg.norm(after_pick - target) <= args.max_error),
            }
            samples.append(sample)
            print(
                f"[实测] 落点=({after_pick[0]:.2f},{after_pick[1]:.2f})mm，"
                f"落点误差=({after_pick[0]-target[0]:+.2f},{after_pick[1]-target[1]:+.2f})mm，"
                f"位移误差=({error[0]:+.2f},{error[1]:+.2f})mm，"
                f"姿态漂移={rotation_deg:+.2f}°，IoU={iou:.3f}"
            )
            if not sample["valid"]:
                raise RuntimeError(f"第{index}次位置或轮廓误差超过安全阈值，停止后续测试")
            current_piece = after_piece

        controller.command("MOTION,0")
        controller.close()
        controller = None
        fit = position_calibration.fit_position_model(samples)
        write_results(samples, fit)
        print("\n[拟合] actual_delta = matrix @ command_delta + bias")
        print(json.dumps(fit, ensure_ascii=False, indent=2))
        print(f"[OK] 结果已保存：{OUTPUT}")
        return 0
    except BaseException:
        if controller is not None:
            controller.close(safe=True)
        raise
    finally:
        camera.release()
        cv2.destroyAllWindows()


# =============================================================================
# 【分区】命令行参数
# 功能：解析目标点、串口、阈值；默认 COM15。
# 可修改：--port 默认值、--min-iou、--max-error、安全框。
# 看情况改：timeout、相机 settle。
# 不要改：--execute 且 --yes 双开关。
# =============================================================================
def parse_targets(text: str) -> list[tuple[float, float]]:
    targets = []
    for item in text.split(";"):
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("目标格式应为x,y;x,y")
        x, y = map(float, parts)
        if not (35.0 <= x <= 175.0 and 45.0 <= y <= 245.0):
            raise argparse.ArgumentTypeError("位置标定目标超出单片安全区域")
        targets.append((x, y))
    if len(targets) < 2:
        raise argparse.ArgumentTypeError("至少需要2个位置目标")
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1单片XY位置视觉闭环标定")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--port", default="COM15")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--camera-settle", type=float, default=0.8)
    parser.add_argument("--rotation-search-radius", type=float, default=12.0)
    parser.add_argument("--targets", type=parse_targets, default=None,
                        help="自定义目标序列：x,y;x,y")
    parser.add_argument("--min-iou", type=float, default=0.80)
    parser.add_argument("--max-error", type=float, default=15.0)
    parser.add_argument("--execute", action="store_true", help="允许真实驱动硬件")
    parser.add_argument("--yes", action="store_true", help="确认现场安全条件已满足")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (RuntimeError, serial_transport.SerialPlanError, ValueError) as exc:
        print(f"[ERR] {exc}")
        raise SystemExit(1)
