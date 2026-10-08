"""使用单个P1碎片自动标定旋转机构。

默认仅做视觉检查和打印测试序列；加入--execute --yes才会驱动COM口。
"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：扑克目录下的单片旋转标定，流程与普通版相同；默认干跑，--execute --yes 才动电机。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约67行  detect_single_piece：单片安全联锁
#   - 约159行  Controller：标定串口
#   - 约231行  run：主流程
#   - 约338行  build_parser：角度序列与双确认开关
# =============================================================================
# 单片旋转标定工具：识别P1→吸住并执行已知角度→释放后拍照→轮廓测角→拟合实际角和命令角的关系。
# 每次操作后更新current_piece，所以下一个样本从最新实际姿态开始；不是每次都与最初图像比较。
# 默认只做视觉检查和打印；真实驱动分支要求原有命令行开关--execute --yes。
# 样本CSV适合逐次查看误差；result.json同时保存全部样本和拟合结果，程序不会自动把拟合值写回主配置。
# =============================================================================
# 【分区】导入与路径
# 功能：tools 上一级入 path；结果写 output/rotation_calibration。
# 可修改：OUTPUT 目录。
# 看情况改：无。
# 不要改：ROOT=parents[1] 后再 import main。
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
import rotation_calibration
import serial_transport

OUTPUT = ROOT / "output" / "rotation_calibration"


# =============================================================================
# 【分区】稳定采图与单片识别（含边界安全联锁）
# 功能：连读数帧；只保留面积合格、不贴边、吸取点离边足够的唯一一片。
# 可修改：帧数、边框清理、面积范围、边距（config 可覆盖部分）。
# 看情况改：贴边被拒时应移到纸面中部，不要关联锁。
# 不要改：必须恰好 1 块；关闭轮廓闭合以免粘白边。
# =============================================================================
def capture_stable(camera: cv2.VideoCapture, frames: int = 8) -> np.ndarray:
    frame = None
    for _ in range(max(1, frames)):
        success, frame = camera.read()
        if not success:
            raise RuntimeError("摄像头读取失败")
    # 检查内部前提是否成立：frame is not None。
    assert frame is not None
    return frame


def detect_single_piece(frame: np.ndarray, matrix: np.ndarray, config: dict[str, Any],
                        allow_boundary_contact: bool = False):
    paper = main.warp_paper(frame, matrix, config)
    _, paper_mask = main.mask_in_paper(frame, matrix, config)
    ppm = float(config["pixels_per_mm"])

    # A4白边可能与靠边碎片连成外轮廓；单片标定只使用清除边框后的内部区域。
    border_mm = float(config.get("p1_calibration_border_cleanup_mm", 8.0))
    border_px = max(1, int(round(border_mm * ppm)))
    clean_mask = paper_mask.copy()
    clean_mask[:border_px, :] = 0
    clean_mask[-border_px:, :] = 0
    clean_mask[:, :border_px] = 0
    clean_mask[:, -border_px:] = 0
    local_config = dict(config)
    local_config["generic_contour_close_mm"] = 0.0
    pieces, mask = main.detect_pieces(paper, local_config, False, clean_mask)

    # 先按单片面积排除A4白边，再按边界状态执行安全联锁。
    min_area = float(config.get("p1_calibration_min_area_mm2", 2200.0))
    max_area = float(config.get("p1_calibration_max_area_mm2", 3100.0))
    eligible_pieces = []
    ignored_area_count = 0
    ignored_border_count = 0
    for candidate in pieces:
        area_mm2 = float(
            cv2.contourArea(candidate.contour.astype(np.float32)) / (ppm * ppm)
        )
        if not (min_area <= area_mm2 <= max_area):
            ignored_area_count += 1
            continue
        x, y, width_px, height_px = cv2.boundingRect(
            candidate.contour.astype(np.int32)
        )
        touches_cleanup_border = (
            x <= border_px or y <= border_px
            or x + width_px >= mask.shape[1] - border_px
            or y + height_px >= mask.shape[0] - border_px
        )
        if touches_cleanup_border and not allow_boundary_contact:
            ignored_border_count += 1
            continue
        eligible_pieces.append((candidate, area_mm2))

    if len(eligible_pieces) != 1:
        raise RuntimeError(
            "安全联锁：应只保留1块P1，"
            f"当前合格轮廓={len(eligible_pieces)}，"
            f"面积排除={ignored_area_count}，边界排除={ignored_border_count}"
        )

    piece, area_mm2 = eligible_pieces[0]

    x, y, width_px, height_px = cv2.boundingRect(piece.contour.astype(np.int32))
    touches_cleanup_border = (
        x <= border_px or y <= border_px
        or x + width_px >= mask.shape[1] - border_px
        or y + height_px >= mask.shape[0] - border_px
    )
    if touches_cleanup_border and not allow_boundary_contact:
        raise RuntimeError("安全联锁：P1完整轮廓接近或越过A4边界，请先移回纸面内部")

    pick_x, pick_y = piece.pick_mm(ppm)
    margin = 25.0
    paper_width = float(config["paper_width_mm"])
    paper_height = float(config["paper_height_mm"])
    if not (margin <= pick_x <= paper_width - margin and margin <= pick_y <= paper_height - margin):
        raise RuntimeError(
            f"安全联锁：P1吸取点({pick_x:.1f},{pick_y:.1f})mm离边缘过近，请放到A4中部"
        )
    return piece, paper, mask


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


# =============================================================================
# 【分区】标定串口控制器
# 功能：开串口（DTR/RTS=False）、发 ASCII 行、读 ACK、可选安全停机。
# 可修改：打开后等待、timeout。
# 看情况改：板子若需 DTR 才能通信再改，并确认不会复位。
# 不要改：命令 \n 结尾 ASCII；close(safe=True)。
# =============================================================================
class Controller:
    # 创建串口对象，配置波特率/超时及DTR/RTS，然后打开并清空旧数据。
    def __init__(self, port: str, baudrate: int, timeout: float, delay: float = 0.0):
        import serial
        self.delay = delay
        self.connection = serial.Serial()
        self.connection.port = port
        self.connection.baudrate = baudrate
        self.connection.timeout = timeout
        self.connection.write_timeout = timeout
        self.connection.dtr = False
        self.connection.rts = False
        self.connection.open()
        time.sleep(1.0)
        self.connection.reset_input_buffer()
        self.connection.reset_output_buffer()

    def command(self, command: str, wait: float = 0.0) -> str:
        print(f"[TX] {command}")
        self.connection.write((command + "\n").encode("ascii"))
        self.connection.flush()
        response = serial_transport._read_response(self.connection)
        print(f"[RX] {response}")
        if wait > 0.0:
            time.sleep(wait)
        if self.delay > 0.0:
            time.sleep(self.delay)
        return response

    def close(self, safe: bool = False) -> None:
        if safe:
            serial_transport._best_effort_safety_shutdown(self.connection, print)
        self.connection.close()


# =============================================================================
# 【分区】角度解析与结果落盘
# 功能：角度幅值限制；JSON+CSV 保存，不写回主配置。
# 可修改：角度范围、默认序列。
# 看情况改：不要把下限改得太小，小角测不准增益。
# 不要改：CSV utf-8-sig。
# =============================================================================
def parse_angles(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) < 1 or any(abs(value) < 5.0 or abs(value) > 120.0 for value in values):
        raise argparse.ArgumentTypeError("至少1个角度，每个绝对值必须在5～120度")


    return values


def write_results(samples: list[dict[str, Any]], fit: dict[str, Any]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {"samples": samples, "fit": fit}
    (OUTPUT / "result.json").write_text(
        # indent=2（JSON每级缩进空格数）；取值过程：2。
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "samples.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0].keys()))
        writer.writeheader()
        writer.writerows(samples)


# 串联视觉检查、逐角度试验、轮廓测角和拟合，保存结果并释放硬件资源。
# =============================================================================
# 【分区】主流程 run()
# 功能：吸住→旋转命令角→放下→轴回正→拍照测相对角；下一次从新姿态继续。
# 可修改：--angles、搜索半径、min-iou、camera-settle。
# 看情况改：轴回正是为了线束；改协议需同步串口层。
# 不要改：--execute --yes；IoU 不足即停；finally 释放相机。
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
        before_piece, before_paper, before_mask = detect_single_piece(initial_frame, matrix, config)
        save_detection(0, "initial", initial_frame, before_paper, before_mask, before_piece)
        ppm = float(config["pixels_per_mm"])
        initial_pick = before_piece.pick_mm(ppm)
        print(f"[视觉] 单片P1吸取点=({initial_pick[0]:.2f},{initial_pick[1]:.2f})mm")
        print(f"[计划] 角度序列={','.join(f'{value:+.1f}' for value in args.angles)}°")

        if not args.execute:
            print("[DRY-RUN] 视觉检查通过，未打开串口、未驱动电机。")
            print("[DRY-RUN] 实机执行需显式加入：--execute --yes")
            return 0
        if not args.yes:
            raise RuntimeError("真实执行必须同时指定--yes")

        offset_x, offset_y = map(float, config.get("motion_target_offset_mm", [0.0, 0.0]))
        pickup_wait = float(config.get("magnet_pickup_settle_seconds", 0.8))
        release_wait = float(config.get("magnet_release_settle_seconds", 0.3))
        controller = Controller(args.port, args.baudrate, args.timeout,
                                float(config.get("serial_command_delay_seconds", 0.0)))
        controller.command("PING")
        controller.command("MOTION,1")

        current_piece = before_piece
        for index, command_angle in enumerate(args.angles, 1):
            pick_x, pick_y = current_piece.pick_mm(ppm)
            command_x, command_y = pick_x + offset_x, pick_y + offset_y
            print(f"\n[测试 {index}/{len(args.angles)}] 指令={command_angle:+.2f}°，"
                  f"吸取=({command_x:.2f},{command_y:.2f})mm")
            controller.command(f"GOTO,{command_x:.2f},{command_y:.2f}")
            controller.command("Z,DOWN")
            controller.command("MAGNET,1", pickup_wait)
            controller.command("Z,UP")
            controller.command(f"ROTATE,{command_angle:.2f}")
            controller.command("Z,DOWN")
            controller.command("MAGNET,0", release_wait)
            controller.command("Z,UP")
            controller.command(f"ROTATE,{-command_angle:.2f}")
            controller.command("GOTO,0.00,0.00")
            time.sleep(args.camera_settle)

            after_frame = capture_stable(camera)
            after_piece, after_paper, after_mask = detect_single_piece(after_frame, matrix, config)
            save_detection(index, "after", after_frame, after_paper, after_mask, after_piece)
            actual_angle, iou = rotation_calibration.estimate_relative_rotation(
                current_piece.contour, after_piece.contour, command_angle,
                search_radius_deg=args.search_radius,
            )
            before_center = np.asarray(current_piece.center_mm(ppm), np.float64)
            after_center = np.asarray(after_piece.center_mm(ppm), np.float64)
            drift = after_center - before_center
            sample = {
                "index": index,
                "command_deg": round(command_angle, 4),
                "actual_deg": round(actual_angle, 4),
                "error_deg": round(actual_angle - command_angle, 4),
                "iou": round(iou, 5),
                "center_dx_mm": round(float(drift[0]), 4),
                "center_dy_mm": round(float(drift[1]), 4),
                "center_drift_mm": round(float(np.linalg.norm(drift)), 4),
                "valid": bool(iou >= args.min_iou),
            }
            samples.append(sample)
            print(f"[实测] 实际={actual_angle:+.2f}°，误差={actual_angle-command_angle:+.2f}°，"
                  f"IoU={iou:.3f}，中心漂移={np.linalg.norm(drift):.2f}mm")
            if not sample["valid"]:
                raise RuntimeError(f"第{index}次轮廓匹配可信度不足，停止后续测试")
            current_piece = after_piece

        controller.command("MOTION,0")
        controller.close()
        controller = None
        fit = rotation_calibration.fit_rotation_model(
            samples, float(config.get("motion_rotation_cycles_per_revolution", 164.0))
        )
        write_results(samples, fit)
        print("\n[拟合] actual = gain × command + bias")
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
# 【分区】命令行参数与入口
# 功能：默认 COM15；必须显式 --execute --yes。
# 可修改：--port、波特率、timeout。
# 看情况改：无。
# 不要改：双重确认开关。
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1单片旋转轴视觉闭环标定")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--port", default="COM15")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--angles", type=parse_angles, default=parse_angles("30,-30,60,-60,90,-90"))
    parser.add_argument("--camera-settle", type=float, default=0.8)
    parser.add_argument("--search-radius", type=float, default=35.0)
    parser.add_argument("--min-iou", type=float, default=0.72)
    parser.add_argument("--execute", action="store_true", help="允许真实驱动硬件")
    parser.add_argument("--yes", action="store_true", help="确认现场安全条件已满足")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (RuntimeError, serial_transport.SerialPlanError, ValueError) as exc:
        print(f"[ERR] {exc}")
        raise SystemExit(1)
