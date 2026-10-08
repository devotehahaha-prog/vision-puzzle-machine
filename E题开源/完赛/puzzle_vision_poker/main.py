"""USB摄像头碎片识别、编号与A4最短正交路径生成。"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：扑克拼图主程序：相机/图片 → A4拉正 → 分割（3片）→ 几何求解 + 切缝花纹 NCC → 正交路径 → plan.json。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约97行  Piece：单片视觉结果
#   - 约116行  PuzzleMatch：目标姿态与匹配质量
#   - 约145行  PathResult：匹配 + 搬运折线
#   - 约167行  load_config：读 config.json（含 poker_* 键）
#   - 约544行  warp_paper：透视拉成正 A4
#   - 约594行  open_camera：打开俯拍摄像头
#   - 约735行  make_poker_v_mask：扑克牌面分割掩膜之一
#   - 约1006行  detect_pieces：生成 Piece 列表，扑克默认 3 片
#   - 约1074行  number_by_move_order：安全搬运顺序编号
#   - 约1798行  verify_edge_patterns：沿拼缝做 NCC，写入花纹验证结果
#   - 约1892行  solve_generic_puzzle：几何求解 + 花纹
#   - 约1980行  solve_puzzle：模式分流
#   - 约2031行  plan_all_paths：正交路径
#   - 约2380行  save_outputs：写出 plan（含 poker_pattern_verification）
#   - 约2588行  process_paper：主链路
#   - 约3292行  run_auto：一键自动流程
#   - 约3407行  apply_puzzle_set：self=图2，site=通用几何；赛场通常 site+花纹
#   - 约3462行  main：总入口
# =============================================================================
# 本文件负责把相机画面变成搬运计划：图像 → 纸面校正 → 二值分割 → 碎片几何 → 目标姿态 → 单轴路径 → plan.json。
# 建议先看文件末尾 main()，再看 process_paper()；需要自动执行时接着看 run_auto() 和 serial_transport.py。
# 图像数组按[行,列]即[y,x]访问，但坐标点通常写成(x,y)；shape前两项是(高,宽)。
# 纸面原点在左上，X向右、Y向下。这里正旋转角在画面上表现为顺时针CW，负值为逆时针CCW。
# 变量后缀_px是像素，_mm是毫米，_mm2是平方毫米，_deg是角度；ppm=像素/毫米。长度除ppm、面积除ppm²才能换算。
# 二值掩膜是与图像同宽高的单通道数组：0排除背景，255保留目标；彩色图像默认通道顺序为BGR。
# Piece保存识别结果；PuzzleMatch保存目标位置及旋转；PathResult把匹配与搬运折线关联起来。
# 固定图2直接匹配A/B/C/D；generic_geometry交给generic_solver搜索。花纹功能还需edge_matcher模块及相应配置。
# 代码中的fallback/relaxed表示原有兜底策略；生成了可发送计划并不等同于已拼成严格合格的矩形。
# =============================================================================
# 【分区】导入、路径常量、图2模板
# 功能：加载求解器/串口/可选 edge_matcher；锁定 config、calibration、output 路径。
# 可修改：DEMO_PLACEMENTS 只影响 --demo。
# 看情况改：扑克实机必须能 import edge_matcher，否则花纹验证关闭。
# 不要改：ROOT/CONFIG_PATH；FIGURE2 顶点（训练用）。扑克赛场走 generic + 花纹，不是这套 A/B/C/D。
# =============================================================================
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

import generic_solver
import orthogonal_path_planner as path_planner
import serial_transport
from dual_serial_executor import DualExecutionError, DualSerialExecutor

try:
    import edge_matcher
    _PATTERN_MATCHER_AVAILABLE = True
except ImportError:
    edge_matcher = None
    _PATTERN_MATCHER_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
CALIBRATION_PATH = ROOT / "calibration.json"
OUTPUT_DIR = ROOT / "output"


FIGURE2_TEMPLATE_POLYGONS_MM: dict[str, tuple[tuple[float, float], ...]] = {
    "A": ((0.0, 0.0), (20.0, 0.0), (36.0, 12.0), (0.0, 20.0)),
    "B": ((0.0, 20.0), (36.0, 12.0), (76.0, 42.0), (0.0, 30.0)),
    "C": ((0.0, 30.0), (76.0, 42.0), (100.0, 60.0), (0.0, 60.0)),
    "D": ((20.0, 0.0), (100.0, 0.0), (100.0, 60.0),
          (76.0, 42.0), (36.0, 12.0)),
}

DEMO_PLACEMENTS: dict[str, tuple[tuple[float, float], float]] = {
    "A": ((25.0, 25.0), 25.0),
    "B": ((92.0, 25.0), -10.0),
    "C": ((55.0, 105.0), -10.0),
    "D": ((160.0, 85.0), 15.0),
}


# =============================================================================
# 【分区】核心数据结构 Piece / PuzzleMatch / PathResult
# 功能：识别结果、目标姿态、匹配+路径。
# 可修改：无现场参数。
# 看情况改：新字段只能追加默认值。
# 不要改：像素/毫米单位；正角=画面顺时针。
# =============================================================================
@dataclass
class Piece:
    """单块碎片的视觉结果。"""
    piece_id: int
    contour: np.ndarray
    center_px: tuple[float, float]
    pick_px: tuple[float, float]
    angle_deg: float
    area_mm2: float
    width_mm: float
    height_mm: float

    def center_mm(self, ppm: float) -> tuple[float, float]:
        return self.center_px[0] / ppm, self.center_px[1] / ppm

    def pick_mm(self, ppm: float) -> tuple[float, float]:
        return self.pick_px[0] / ppm, self.pick_px[1] / ppm


@dataclass
class PuzzleMatch:
    """源碎片到题目图2固定模板的匹配和刚体变换。"""
    piece: Piece
    template_id: str
    target_polygon_mm: np.ndarray
    target_pick_mm: tuple[float, float]
    rotation_deg: float
    rotation_direction: str
    area_error_ratio: float
    shape_distance: float
    match_iou: float
    fit_scale: float
    status: str
    solver_mode: str = "fixed_figure_2"
    transform_3x3: np.ndarray | None = None
    solution_score: float = 0.0
    fill_error_ratio: float = 0.0
    overlap_ratio: float = 0.0
    endpoint_error_mm: float = 0.0
    boundary_error_mm: float = 0.0
    boundary_gap_ratio: float = 0.0
    target_rectangle_top_left_mm: tuple[float, float] | None = None
    target_rectangle_size_mm: tuple[float, float] | None = None
    target_clearance_shift_mm: tuple[float, float] = (0.0, 0.0)
    pattern_verification: dict | None = None
    relaxed_override: bool = False


@dataclass
class PathResult:
    """单块碎片的模板匹配和搬运路径。"""
    match: PuzzleMatch
    path_px: list[tuple[float, float]]
    planner: str
    status: str

    @property
    def piece(self) -> Piece:
        return self.match.piece

    @property
    def target_mm(self) -> tuple[float, float]:
        return self.match.target_pick_mm

# =============================================================================
# 【分区】配置与图像读写、纸面像素尺寸
# 功能：读 config.json；中文路径读写图；毫米×ppm→像素。
# 可修改：阈值一律改 config.json（扑克键名 poker_* 与 generic_*）。
# 看情况改：pixels_per_mm 必须与标定一致。
# 不要改：utf-8-sig；write_image 的中文路径写法。
# =============================================================================
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_image(path: Path) -> np.ndarray | None:
    """兼容Windows中文路径的图片读取。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def write_image(path: Path, image: np.ndarray) -> None:
    """兼容Windows中文路径的图片保存。"""
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"图片编码失败：{path.name}")
    encoded.tofile(str(path))


def paper_size_px(config: dict[str, Any]) -> tuple[int, int]:
    ppm = float(config["pixels_per_mm"])
    return (
        int(round(float(config["paper_width_mm"]) * ppm)),
        int(round(float(config["paper_height_mm"]) * ppm)),
    )


def workspace_size_px(config: dict[str, Any]) -> tuple[int, int]:
    """返回与A4标定区域完全一致的路径规划工作区。"""
    return path_planner.workspace_size_px(
        float(config["paper_width_mm"]),
        float(config["paper_height_mm"]),
        float(config["pixels_per_mm"]),
    )


# =============================================================================
# 【分区】A4 四角标定与透视矩阵
# 功能：四角 → 透视矩阵 → warp_paper 拉成正 A4。
# 可修改：黑框阈值等配置项。
# 看情况改：换相机/高度必须重标。
# 不要改：角点顺序；3×3 透视不要改仿射。
# =============================================================================
def save_calibration(points: np.ndarray, matrix: np.ndarray,
                     image_size: tuple[int, int]) -> None:
    payload = {
        "image_points": points.astype(float).tolist(),
        "perspective_matrix": matrix.astype(float).tolist(),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with CALIBRATION_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


# 先按相对中心的极角排成环，再把x+y最小的点放在首位；这是本实现选左上角的启发式规则。
def order_quad_points(points: np.ndarray) -> np.ndarray:
    """把任意点击顺序的四个角排序为左上、右上、右下、左下。"""
    source = np.asarray(points, np.float32).reshape(4, 2)
    center = np.mean(source, axis=0)
    angles = np.arctan2(source[:, 1] - center[1], source[:, 0] - center[0])
    ordered = source[np.argsort(angles)]
    start = int(np.argmin(np.sum(ordered, axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def valid_quad(points: np.ndarray) -> bool:
    """仅拒绝重复点、明显过短边和退化四边形，不限制A4画面占比。"""
    source = np.asarray(points, np.float32).reshape(4, 2)
    if not np.all(np.isfinite(source)):
        return False
    edges = [float(np.linalg.norm(source[(index + 1) % 4] - source[index]))
             for index in range(4)]
    area = abs(float(cv2.contourArea(source)))
    return min(edges) >= 20.0 and area >= 1000.0 and cv2.isContourConvex(source.astype(np.int32))


def calibration_area_ratio(points: np.ndarray, image_size: tuple[int, int]) -> float:
    """返回标定四边形占原始画面的比例。"""
    width, height = image_size
    if width <= 0 or height <= 0:
        return 0.0
    return abs(float(cv2.contourArea(np.asarray(points, np.float32)))) / float(width * height)


# 透视矩阵用于消除相机斜拍影响，它不同于碎片搬运的刚体矩阵；前者允许透视缩放，后者只允许平移和旋转。
def calibration_matrix(points: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source = order_quad_points(points)
    if not valid_quad(source):
        raise ValueError("四个角点重复、距离过近或不能组成有效四边形")
    width, height = paper_size_px(config)
    destination = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1],
                              [0, height - 1]], np.float32)
    return source, cv2.getPerspectiveTransform(source, destination)


# 对多个亮度阈值反复分割深色区域，再用面积、四边形、边缘倾斜及内部深浅像素比例筛选；最后取综合评分最高的四角。
def detect_dark_board_points(frame: np.ndarray,
                             config: dict[str, Any]) -> np.ndarray | None:
    """从当前画面自动定位深色A4底板，返回左上、右上、右下、左下四角。"""
    if not bool(config.get("auto_board_calibration", True)):
        return None
    if str(config.get("segmentation_mode", "pink_hsv")) != "white_piece":
        return None
    if frame.ndim != 3 or min(frame.shape[:2]) < 100:
        return None

    height, width = frame.shape[:2]
    image_area = float(width * height)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation_max = int(config.get("auto_board_saturation_max", 110))
    value_thresholds = config.get(
        "auto_board_value_thresholds", [80, 90, 100, 110, 120, 130, 140]
    )
    minimum_area_ratio = float(config.get("auto_board_min_area_ratio", 0.08))
    maximum_area_ratio = float(config.get("auto_board_max_area_ratio", 0.35))
    minimum_dark_fraction = float(config.get("auto_board_min_dark_fraction", 0.55))
    maximum_white_fraction = float(config.get("auto_board_max_white_fraction", 0.35))
    minimum_aspect = float(config.get("auto_board_min_aspect_ratio", 0.45))
    maximum_aspect = float(config.get("auto_board_max_aspect_ratio", 1.05))

    open_size = max(3, int(round(width / 384.0)))
    open_size += 1 - open_size % 2
    close_size = max(9, int(round(width / 91.0)))
    close_size += 1 - close_size % 2
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    border_margin = max(2, int(round(min(width, height) * 0.003)))
    candidates: list[tuple[float, np.ndarray]] = []

    for raw_threshold in value_thresholds:
        value_max = int(raw_threshold)
        selected = ((hsv[:, :, 2] <= value_max) &
                    (hsv[:, :, 1] <= saturation_max))
        mask = np.where(selected, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            area_ratio = contour_area / image_area
            if not minimum_area_ratio <= area_ratio <= maximum_area_ratio:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if (x <= border_margin or y <= border_margin or
                    x + box_width >= width - border_margin or
                    y + box_height >= height - border_margin):
                continue

            hull = cv2.convexHull(contour)
            perimeter = float(cv2.arcLength(hull, True))
            for epsilon_ratio in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
                approximate = cv2.approxPolyDP(
                    hull, epsilon_ratio * perimeter, True
                )
                if len(approximate) != 4:
                    continue
                points = order_quad_points(
                    approximate.reshape(4, 2).astype(np.float32)
                )
                if not valid_quad(points):
                    continue

                blocked = str(config.get("auto_board_fix_blocked_corner", ""))
                if blocked == "tl":
                    points[0] = points[1] + points[3] - points[2]
                elif blocked == "tr":
                    points[1] = points[0] + points[2] - points[3]
                elif blocked == "br":
                    points[2] = points[1] + points[3] - points[0]
                elif blocked == "bl":
                    points[3] = points[0] + points[2] - points[1]

                top_width = float(np.linalg.norm(points[1] - points[0]))
                bottom_width = float(np.linalg.norm(points[2] - points[3]))
                left_height = float(np.linalg.norm(points[3] - points[0]))
                right_height = float(np.linalg.norm(points[2] - points[1]))
                aspect = ((top_width + bottom_width) /
                          max(left_height + right_height, 1e-6))
                if not minimum_aspect <= aspect <= maximum_aspect:
                    continue

                board_horiz = max(bottom_width, top_width)
                if board_horiz > 1e-6:
                    right_skew = abs(float(points[1][0] - points[2][0])) / board_horiz
                    left_skew = abs(float(points[0][0] - points[3][0])) / board_horiz
                    max_skew = float(config.get("auto_board_max_edge_skew", 0.06))
                    if max(right_skew, left_skew) > max_skew:
                        continue

                quad_area = abs(float(cv2.contourArea(points)))
                if quad_area <= 1.0:
                    continue
                interior = np.zeros((height, width), np.uint8)
                cv2.fillPoly(interior, [points.astype(np.int32)], 255)
                interior = cv2.erode(interior, close_kernel)
                values = hsv[:, :, 2][interior > 0]
                if values.size == 0:
                    continue
                dark_fraction = float(np.mean(values < 150))
                white_fraction = float(np.mean(values > 220))
                if (dark_fraction < minimum_dark_fraction or
                        white_fraction > maximum_white_fraction):
                    continue

                fill_ratio = min(contour_area / quad_area, 1.0)
                center = np.mean(points, axis=0)
                center_distance = float(np.linalg.norm(
                    center - np.asarray([width / 2.0, height / 2.0])
                ) / math.hypot(width, height))
                score = (quad_area / image_area + 0.5 * fill_ratio +
                         0.5 * dark_fraction - 0.5 * white_fraction -
                         0.25 * center_distance)
                candidates.append((score, points))
                break

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def resolve_calibration_matrix(
        frame: np.ndarray, config: dict[str, Any],
        fallback_matrix: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """优先按当前帧自动定位底板；失败时再使用已保存的手工标定。"""
    points = detect_dark_board_points(frame, config)
    if points is not None:
        _, matrix = calibration_matrix(points, config)
        rounded = np.rint(points).astype(int).tolist()
        print(f"[自动标定] 已定位黑色底板四角：{rounded}")
        return matrix, "auto_board"

    matrix = fallback_matrix if fallback_matrix is not None else load_calibration(config)
    if matrix is None:
        raise RuntimeError("未能自动定位黑色底板，且没有可用的calibration.json兜底标定")
    if config.get("auto_board_calibration", True) and config.get("segmentation_mode") == "white_piece":
        print("[WARN] 本帧未能自动定位黑色底板，已使用保存的标定")
    else:
        print("[标定] 使用当前保存的标定（自动找板未启用）")
    return matrix, "saved"

def load_calibration(config: dict[str, Any]) -> np.ndarray | None:
    # Formal production calibration is embedded in production_config.json so
    # the screen workflow does not silently fall back to an obsolete matrix.
    formal = config.get("camera_calibration", {})
    if isinstance(formal, dict) and "perspective_matrix" in formal:
        try:
            matrix = np.asarray(formal["perspective_matrix"], np.float32)
            if matrix.shape == (3, 3) and np.isfinite(matrix).all():
                return matrix
        except (TypeError, ValueError):
            pass
    if config.get("camera_corners"):
        try:
            points = np.asarray(config["camera_corners"], np.float32)
            if points.shape == (4, 2):
                _, matrix = calibration_matrix(points, config)
                return matrix
        except (TypeError, ValueError, cv2.error):
            pass
    if not CALIBRATION_PATH.exists():
        return None
    try:
        with CALIBRATION_PATH.open("r", encoding="utf-8-sig") as file:
            payload = json.load(file)
        if "image_points" in payload:
            points = np.asarray(payload["image_points"], np.float32)
            image_size = payload.get("image_size", [config["camera_width"], config["camera_height"]])
            ratio = calibration_area_ratio(points, (int(image_size[0]), int(image_size[1])))
            minimum = float(config.get("min_calibration_area_ratio", 0.06))
            if ratio < minimum:
                print(f"[WARN] 旧标定区域只占画面{ratio * 100:.1f}%，已拒绝加载。")
                print("[WARN] 请点击A4纸的4个外角，不是4块粉色碎片。")
                return None
            _, matrix = calibration_matrix(points, config)
            return matrix
        matrix = np.asarray(payload["perspective_matrix"], np.float32)
        return matrix if matrix.shape == (3, 3) else None
    except (OSError, KeyError, ValueError, json.JSONDecodeError, cv2.error):
        return None


# =============================================================================
# 【分区】OpenCV 窗口与手工四点采集
# 功能：安全关窗；手工点 A4 四角。
# 可修改：窗口名、提示。
# 看情况改：点序必须左上→右上→右下→左下。
# 不要改：窗口已关时吞 cv2.error。
# =============================================================================
def safe_destroy_window(window: str) -> None:
    """窗口已被用户或OpenCV后端关闭时，销毁操作不应导致主程序退出。"""
    try:
        cv2.destroyWindow(window)
    except cv2.error:
        pass


def window_is_visible(window: str) -> bool:
    """安全查询窗口状态；不存在的窗口统一视为已关闭。"""
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def collect_four_points(frame: np.ndarray, config: dict[str, Any]) -> np.ndarray | None:
    """点击黑色底板的四个外角；程序自动排序并在第四点后完成。"""
    points: list[tuple[int, int]] = []
    window = "A4 Calibration"

    # 显式缩小显示图，避免WINDOW_NORMAL缩放后鼠标坐标与原始帧不一致。
    scale = min(1.0, 1200.0 / frame.shape[1], 680.0 / frame.shape[0])
    display_size = (max(1, int(round(frame.shape[1] * scale))),
                    max(1, int(round(frame.shape[0] * scale))))
    base = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.clear()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if hasattr(cv2, "WND_PROP_FULLSCREEN"):
        try:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except cv2.error:
            pass
    cv2.setMouseCallback(window, on_mouse)
    print("[标定] 鼠标左键点击黑色底板四个外角，推荐：左上、右上、右下、左下。")
    print("[标定] 第4点后自动保存；点错可右键/R清空，Esc取消。")

    while True:
        display = base.copy()
        for index, point in enumerate(points):
            cv2.circle(display, point, 7, (0, 0, 255), -1)
            cv2.putText(display, str(index + 1), (point[0] + 10, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(points) >= 2:
            cv2.polylines(display, [np.asarray(points, np.int32)], len(points) == 4,
                          (0, 255, 255), 2)
        next_text = f"CORNER {len(points) + 1}" if len(points) < 4 else "CALIBRATION OK"
        cv2.rectangle(display, (0, 0), (display.shape[1], 48), (0, 0, 0), -1)
        cv2.putText(display,
                    f"Click {next_text} | Right click: reset | Esc: cancel",
                    (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.64,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, display)

        if len(points) == 4:
            cv2.waitKey(250)
            break

        key = cv2.waitKey(20) & 0xFF
        if key == 27 or not window_is_visible(window):
            safe_destroy_window(window)
            return None
        if key in (ord("r"), ord("R")):
            points.clear()

    safe_destroy_window(window)
    clicked = np.asarray(points, np.float32) / scale
    try:
        source, matrix = calibration_matrix(clicked, config)
    except ValueError as exc:
        print(f"[ERR] 标定失败：{exc}，请重新标定")
        return None

    ratio = calibration_area_ratio(source, (frame.shape[1], frame.shape[0]))
    minimum = float(config.get("min_calibration_area_ratio", 0.06))
    if ratio < minimum:
        print(f"[ERR] 你点击的区域只占画面{ratio * 100:.1f}%，这不是完整A4纸。")
        print("[ERR] 请点击A4纸的左上、右上、右下、左下4个外角，不要点碎片。")
        return None

    save_calibration(source, matrix, (frame.shape[1], frame.shape[0]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_image(OUTPUT_DIR / "calibration_preview.jpg", warp_paper(frame, matrix, config))
    print("[OK] 四点已自动排序为：左上、右上、右下、左下")
    print("[OK] 已生成output/calibration_preview.jpg，可用于检查底板是否完整")
    print(f"[OK] 标定已自动保存：{CALIBRATION_PATH}")
    return matrix


# =============================================================================
# 【分区】透视拉正纸面
# 功能：用标定矩阵把相机图 warp 成正 A4 俯视图，后续检测都在这张图上做。
# 可修改：输出宽高由 config 的纸面尺寸和 ppm 决定。
# 看情况改：标定不准时先重标四角，不要硬调分割。
# 不要改：必须用 warpPerspective；二值掩膜另走最近邻插值。
# =============================================================================
def warp_paper(frame: np.ndarray, matrix: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    return cv2.warpPerspective(frame, matrix, paper_size_px(config))


# =============================================================================
# 【分区】摄像头打开与占位设备过滤
# 功能：丢掉 ToDesk 等虚拟摄像头，打开真实俯拍 USB。
# 可修改：扫描上限；--camera 指定编号。
# 看情况改：多摄像头时 0 可能是前置。
# 不要改：失败要抛错，不要静默用演示图。
# =============================================================================
def _camera_probe_is_placeholder(frame: np.ndarray, config: dict[str, Any]) -> bool:
    """识别ToDesk等近乎全白的虚拟摄像头占位画面。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white_threshold = int(config.get("camera_placeholder_white_threshold", 245))
    max_white_fraction = float(
        config.get("camera_placeholder_max_white_fraction", 0.90)
    )
    white_fraction = float(np.mean(gray >= white_threshold))
    return white_fraction >= max_white_fraction


def _usb_camera_targets() -> list[str | int]:
    """Prefer stable USB capture nodes over CSI/ISP /dev/videoN indexes."""
    by_id = Path("/dev/v4l/by-id")
    if not by_id.is_dir():
        return []
    targets: list[str | int] = []
    for path in sorted(by_id.iterdir()):
        name = path.name.lower()
        if "usb" not in name or "video-index0" not in name:
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.name.startswith("video") and resolved.exists():
            target = str(resolved)
            if target not in targets:
                targets.append(target)
    return targets


def _read_one_frame(capture: cv2.VideoCapture, tries: int) -> np.ndarray | None:
    frame: np.ndarray | None = None
    for _ in range(max(1, tries)):
        success, candidate = capture.read()
        if success:
            frame = candidate
            break
    return frame


def _probe_opened_capture(config: dict[str, Any], capture: cv2.VideoCapture) -> np.ndarray | None:
    tries = max(1, min(3, int(config.get("camera_probe_frames", 6))))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if config.get("camera_profile_required", False):
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(config.get("camera_fourcc", "MJPG"))))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["camera_width"]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["camera_height"]))
        capture.set(cv2.CAP_PROP_FPS, float(config.get("camera_fps", 30)))
        frame = _read_one_frame(capture, tries)
        if frame is None:
            return None
        expected = [int(config["camera_width"]), int(config["camera_height"])]
        if [frame.shape[1], frame.shape[0]] != expected:
            raise RuntimeError(f"相机实际分辨率{frame.shape[1]}x{frame.shape[0]}与要求{expected}不符")
        if config.get("camera_calibration_image_size") != expected:
            raise RuntimeError("相机分辨率与保存角点的图像尺寸不一致")
        return frame
    frame = _read_one_frame(capture, tries)
    if frame is not None:
        return frame
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["camera_width"]))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["camera_height"]))
    return _read_one_frame(capture, tries)


def _try_open_camera_source(
        config: dict[str, Any], source: str | int) -> tuple[cv2.VideoCapture | None, np.ndarray | None]:
    """Open one camera source. Do not force MJPG first; that can detach Alcor UVC devices."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened() and os.name == "nt":
        capture.release()
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        return None, None
    try:
        frame = _probe_opened_capture(config, capture)
    except BaseException:
        capture.release()
        raise
    if frame is None and os.name != "nt":
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        frame = _probe_opened_capture(config, capture)
    if frame is None:
        capture.release()
        return None, None
    return capture, frame


def _try_open_camera_index(
        config: dict[str, Any], index: int) -> tuple[cv2.VideoCapture | None, np.ndarray | None]:
    """尝试打开并预读一个摄像头；失败时完整释放资源。"""
    return _try_open_camera_source(config, index)


def open_camera(config: dict[str, Any], camera_id: int | None) -> cv2.VideoCapture:
    """打开首选摄像头；编号漂移时自动扫描并拒绝虚拟占位画面。"""
    preferred = int(config["camera_id"] if camera_id is None else camera_id)
    max_index = max(preferred, int(config.get("camera_scan_max_index", 4)))
    candidates: list[str | int] = []
    sources = ([str(config["camera_device"])] if config.get("camera_device") else
               [*_usb_camera_targets(), preferred, *range(max_index + 1)])
    for item in sources:
        if item not in candidates:
            candidates.append(item)
    placeholder_indices: list[str | int] = []

    for index in candidates:
        capture, probe = _try_open_camera_source(config, index)
        if capture is None or probe is None:
            continue
        if _camera_probe_is_placeholder(probe, config):
            placeholder_indices.append(index)
            capture.release()
            print(f"[WARN] 摄像头{index}为近乎全白的虚拟/占位画面，已拒绝")
            continue

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if index != preferred:
            print(f"[WARN] 首选摄像头index={preferred}不可用，已自动切换到{index}")
        print(f"[OK] 摄像头已打开：{index}, {width}x{height}")
        return capture

    placeholder_detail = (
        f"；检测到占位画面={placeholder_indices}"
        if placeholder_indices else ""
    )
    raise RuntimeError(
        f"无法找到真实摄像头（已扫描USB节点和index=0~{max_index}）{placeholder_detail}；"
        "请重新插拔俯拍USB摄像头或恢复ToDesk摄像头映射"
    )

# =============================================================================
# 【分区】碎片识别（掩膜 → 轮廓 → Piece）
# 功能：粉/白/扑克 V 掩膜等 → 轮廓 → 质心、吸取点、角度。扑克牌面花纹复杂，优先白片/扑克掩膜而非粉色。
# 可修改：HSV、面积、形态学——改 config.json。
# 看情况改：灯光变了先调阈值。扑克切缝处花纹不要被当成多块。
# 不要改：detect_pieces 的单位换算；pick 点是磁铁实际位置。
# =============================================================================
# ------------------------- 碎片识别 -------------------------

def _region_bounds_mm(config: dict[str, Any], region: str) -> tuple[float, float]:
    """Return the inclusive Y interval for a named A4 half.

    The production configuration uses explicit ``source_region`` and
    ``target_region`` values.  Older configuration files did not have those
    keys and are kept compatible with their historical source-top/target-
    bottom behaviour.
    """
    height = float(config.get("paper_height_mm", 297.0))
    separator = float(config.get("separator_line_y_mm",
                                config.get("target_region_top_mm", height / 2.0)))
    separator = min(max(separator, 0.0), height)
    explicit = str(config.get(f"{region}_region", "" )).strip().lower()
    if not explicit:
        if region == "source" and "source_region_bottom_mm" in config:
            legacy_end = min(max(float(config["source_region_bottom_mm"]), 0.0), height)
            return 0.0, legacy_end
        explicit = "top" if region == "source" else "bottom"
    if explicit == "top":
        return 0.0, separator
    if explicit == "bottom":
        return separator, height
    raise ValueError(f"未知{region}_region：{explicit}，应为top或bottom")


def _apply_region_mask(mask: np.ndarray, config: dict[str, Any],
                       region: str) -> np.ndarray:
    """Keep only the configured top/bottom half of a corrected A4 image."""
    ppm = float(config["pixels_per_mm"])
    y0_mm, y1_mm = _region_bounds_mm(config, region)
    y0 = max(0, min(mask.shape[0], int(round(y0_mm * ppm))))
    y1 = max(y0, min(mask.shape[0], int(round(y1_mm * ppm))))
    result = np.zeros_like(mask)
    result[y0:y1] = mask[y0:y1]
    return result

def make_pink_mask(image: np.ndarray, config: dict[str, Any],
                   kernel_px: int) -> np.ndarray:
    """直接按HSV色相提取粉色，不再依赖纸张边缘背景估计。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue_min = int(config.get("pink_hue_min", 140))
    hue_max = int(config.get("pink_hue_max", 179))
    saturation_min = int(config.get("pink_saturation_min", 30))
    value_min = int(config.get("pink_value_min", 100))
    if hue_min <= hue_max:
        hue_ok = (hue >= hue_min) & (hue <= hue_max)
    else:
        hue_ok = (hue >= hue_min) | (hue <= hue_max)
    mask = np.where(hue_ok & (saturation >= saturation_min) & (value >= value_min),
                    255, 0).astype(np.uint8)
    size = max(1, int(kernel_px))
    size += 1 - size % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def clean_piece_mask(mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """对校正纸面掩膜执行统一形态学和可配置源区域限制。"""
    ppm = float(config["pixels_per_mm"])
    kernel_px = max(3, int(round(float(config["morphology_kernel_mm"]) * ppm)))
    kernel_px += 1 - kernel_px % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    cleaned = mask.astype(np.uint8).copy()
    if float(config["morphology_kernel_mm"]) > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = _apply_region_mask(cleaned, config, "source")

    """赛题要求黑色A4中间横线分割上下区域：把分界线附近的窄带整体清零，
    避免任意颜色的实线被分割算法当成碎片或粘连碎片。线位置默认取
    target_region_top_mm，也可用 separator_line_y_mm 单独指定。"""
    line_y = float(config.get("separator_line_y_mm",
                               config.get("target_region_top_mm", 148.5)))
    half = float(config.get("separator_line_half_width_mm", 3.0))
    if line_y > 0.0:
        y0 = int(round((line_y - half) * ppm))
        y1 = int(round((line_y + half) * ppm))
        y0 = max(y0, 0)
        y1 = min(y1, cleaned.shape[0])
        if y1 > y0:
            cleaned[y0:y1] = 0
    return cleaned


def make_background_inverse_mask(paper: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """从纸面边缘估计背景颜色，提取与背景颜色差异明显的碎片。"""
    ppm = float(config["pixels_per_mm"])
    border = max(1, int(round(float(config.get("border_sample_mm", 5.0)) * ppm)))
    border = min(border, max(1, min(paper.shape[:2]) // 4))
    samples = np.concatenate((
        paper[:border].reshape(-1, 3), paper[-border:].reshape(-1, 3),
        paper[:, :border].reshape(-1, 3), paper[:, -border:].reshape(-1, 3),
    ), axis=0)
    background_bgr = np.median(samples, axis=0).astype(np.uint8).reshape(1, 1, 3)
    paper_lab = cv2.cvtColor(paper, cv2.COLOR_BGR2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    components = str(config.get("background_distance_components", "lab")).lower()
    if components == "ab":
        # Colored paper: brightness shadows retain the paper's chromaticity.
        distance = np.linalg.norm(paper_lab[:, :, 1:] - background_lab[1:], axis=2)
    elif components == "lab":
        distance = np.linalg.norm(paper_lab - background_lab, axis=2)
    else:
        raise ValueError("background_distance_components必须是lab或ab")
    threshold = float(config.get("background_distance_threshold", 18.0))
    return np.where(distance >= threshold, 255, 0).astype(np.uint8)


def make_white_piece_mask(paper: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """提取低饱和度高亮白片；亮度阈值支持固定值或Otsu自适应。"""
    hsv = cv2.cvtColor(paper, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    saturation_max = int(config.get("white_piece_saturation_max", 55))
    threshold_mode = str(config.get("white_piece_threshold_mode", "fixed"))

    if threshold_mode == "otsu":
        ppm = float(config["pixels_per_mm"])
        _, source_end_mm = _region_bounds_mm(config, "source")
        source_bottom = min(max(int(round(source_end_mm * ppm)), 1), gray.shape[0])
        otsu_threshold, _ = cv2.threshold(
            gray[:source_bottom], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        minimum = int(config.get("white_piece_otsu_min_value", 120))
        maximum = int(config.get("white_piece_otsu_max_value", 230))
        if not 0 <= minimum <= maximum <= 255:
            raise ValueError("white_piece Otsu阈值范围必须满足0<=min<=max<=255")
        brightness_min = int(round(np.clip(otsu_threshold, minimum, maximum)))
    elif threshold_mode == "fixed":
        brightness_min = int(config.get("white_piece_value_min", 245))
    else:
        raise ValueError(f"未知white_piece_threshold_mode：{threshold_mode}")

    selected = ((hsv[:, :, 1] <= saturation_max) & (gray >= brightness_min))
    return np.where(selected, 255, 0).astype(np.uint8)


def make_poker_v_mask(paper: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """扑克牌白底碎片分割：V通道二值化 + 大核闭运算填充牌面花纹。

    扑克牌碎片是白色卡纸，牌面上印有红心/黑桃等花纹；直接用白片亮度阈值
    会把花纹洞当成碎片内部空洞，因此先按V通道二值化提取卡面，再用大核
    闭运算把花纹洞填平，保证轮廓完整。
    """
    hsv = cv2.cvtColor(paper, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    threshold_mode = str(config.get("poker_threshold_mode", "otsu"))

    if threshold_mode == "otsu":
        otsu_threshold, _ = cv2.threshold(
            value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        minimum = int(config.get("poker_otsu_min_value", 80))
        maximum = int(config.get("poker_otsu_max_value", 255))
        if not 0 <= minimum <= maximum <= 255:
            raise ValueError("poker Otsu阈值范围必须满足0<=min<=max<=255")
        brightness_min = int(round(np.clip(otsu_threshold, minimum, maximum)))
        mask = cv2.threshold(value, brightness_min, 255, cv2.THRESH_BINARY)[1]
    elif threshold_mode == "adaptive":
        block = max(15, min(99, (min(paper.shape[:2]) // 16) | 1))
        mask = cv2.adaptiveThreshold(
            value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block,
            int(config.get("poker_adaptive_c", 2)),
        )
    else:
        brightness_min = int(config.get("poker_value_min", 90))
        mask = cv2.threshold(value, brightness_min, 255, cv2.THRESH_BINARY)[1]

    ppm = float(config["pixels_per_mm"])
    close_mm = float(config.get("poker_morph_close_mm", 2.5))
    if close_mm <= 0:
        return mask
    kernel_px = max(3, int(round(close_mm * ppm)))
    kernel_px += 1 - kernel_px % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def calibration_edge_color_distances(
        paper: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    """比较校正图四边与下半区背景颜色，用于发现标定漂移和画面裁切。"""
    if paper.ndim != 3 or paper.shape[0] < 8 or paper.shape[1] < 8:
        return {"top": math.inf, "bottom": math.inf,
                "left": math.inf, "right": math.inf}

    ppm = float(config["pixels_per_mm"])
    band = max(4, int(round(float(config.get("calibration_edge_band_mm", 3.0)) * ppm)))
    height, width = paper.shape[:2]
    band = min(band, max(1, min(height, width) // 8))
    lab = cv2.cvtColor(paper, cv2.COLOR_BGR2LAB).astype(np.float32)

    reference = lab[
        int(round(height * 0.55)):max(int(round(height * 0.85)), int(round(height * 0.55)) + 1),
        int(round(width * 0.20)):max(int(round(width * 0.80)), int(round(width * 0.20)) + 1),
    ]
    reference_color = np.median(reference.reshape(-1, 3), axis=0)
    vertical_start = int(round(height * 0.15))
    vertical_end = max(int(round(height * 0.85)), vertical_start + 1)
    horizontal_start = int(round(width * 0.15))
    horizontal_end = max(int(round(width * 0.85)), horizontal_start + 1)
    edges = {
        "top": lab[:band, horizontal_start:horizontal_end],
        "bottom": lab[-band:, horizontal_start:horizontal_end],
        "left": lab[vertical_start:vertical_end, :band],
        "right": lab[vertical_start:vertical_end, -band:],
    }
    return {
        side: float(np.linalg.norm(np.median(values.reshape(-1, 3), axis=0) - reference_color))
        for side, values in edges.items()
    }


def calibration_quality_reasons(paper: np.ndarray,
                                config: dict[str, Any]) -> list[str]:
    """返回可读的标定质量问题；当前仅对白片深色底板模式执行颜色一致性检查。"""
    if str(config.get("segmentation_mode", "pink_hsv")) != "white_piece":
        return []
    maximum = float(config.get("max_calibration_edge_color_distance", 50.0))
    names = {"top": "上", "bottom": "下", "left": "左", "right": "右"}
    distances = calibration_edge_color_distances(paper, config)
    invalid = [f"{names[side]}边色差{distance:.1f}"
               for side, distance in distances.items() if distance > maximum]
    if not invalid:
        return []
    return [f"校正图边缘混入白边/桌面（{', '.join(invalid)}，阈值{maximum:.1f}）"]


def piece_border_contacts(pieces: list[Piece], paper_shape: tuple[int, int],
                          config: dict[str, Any]) -> list[str]:
    """找出接触校正图边界的碎片；轮廓被裁切时禁止继续求解。"""
    height, width = paper_shape
    margin = max(1, int(round(float(config.get("piece_border_margin_mm", 2.0)) *
                              float(config["pixels_per_mm"]))))
    contacts: list[str] = []
    for piece in pieces:
        x, y, piece_width, piece_height = cv2.boundingRect(piece.contour.astype(np.int32))
        sides: list[str] = []
        if x <= margin:
            sides.append("左")
        if y <= margin:
            sides.append("上")
        if x + piece_width >= width - margin:
            sides.append("右")
        if y + piece_height >= height - margin:
            sides.append("下")
        if sides:
            contacts.append(f"P{piece.piece_id}接触{'/'.join(sides)}边")
    return contacts


def make_piece_mask(paper: np.ndarray, config: dict[str, Any],
                    diagnostics: bool = False) -> np.ndarray:
    """按segmentation_mode识别校正纸面中的碎片。"""
    mode = str(config.get("segmentation_mode", "pink_hsv"))
    if mode == "pink_hsv":
        ppm = float(config["pixels_per_mm"])
        kernel_px = max(3, int(round(float(config["morphology_kernel_mm"]) * ppm)))
        # 保留原有粉色路径，保证默认模式的掩膜行为不变。
        mask = make_pink_mask(paper, config, kernel_px)
        mask = _apply_region_mask(mask, config, "source")
        detail = (f"H={config.get('pink_hue_min', 140)}~"
                  f"{config.get('pink_hue_max', 179)}, "
                  f"S>={config.get('pink_saturation_min', 30)}, "
                  f"V>={config.get('pink_value_min', 100)}")
    elif mode == "background_inverse":
        mask = clean_piece_mask(make_background_inverse_mask(paper, config), config)
        detail = f"Lab颜色距离>={config.get('background_distance_threshold', 18.0)}"
    elif mode == "white_piece":
        mask = clean_piece_mask(make_white_piece_mask(paper, config), config)
        threshold_mode = str(config.get("white_piece_threshold_mode", "fixed"))
        detail = (f"S<={config.get('white_piece_saturation_max', 55)}, "
                  f"亮度阈值模式={threshold_mode}")
    elif mode == "poker_v":
        mask = clean_piece_mask(make_poker_v_mask(paper, config), config)
        close_mm = float(config.get("poker_morph_close_mm", 2.5))
        detail = (f"V通道阈值模式={config.get('poker_threshold_mode', 'otsu')}, "
                  f"闭运算填花纹={close_mm:.1f}mm")
    else:
        raise ValueError(f"未知segmentation_mode：{mode}")
    if diagnostics:
        selected_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        print(f"[诊断] 分割模式={mode}: {detail}, 白色像素占比={selected_ratio * 100:.2f}%")
        if mode == "white_piece" and selected_ratio > 0.50:
            print("[WARN] 白片掩膜超过画面50%，深色底板也被选中了；请提高white_piece_value_min或降低曝光。")
        if mode == "poker_v" and selected_ratio > 0.50:
            print("[WARN] 扑克牌掩膜超过画面50%，深色底板也被选中了；请调整poker_otsu_min_value或降低曝光。")
    return _apply_region_mask(mask, config, "source")


def count_piece_candidates(mask: np.ndarray, config: dict[str, Any]) -> int:
    """按正式检测的面积和实心度规则统计有效碎片轮廓。"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    ppm = float(config["pixels_per_mm"])
    minimum_area = float(config["min_piece_area_mm2"])
    maximum_area = float(config["max_piece_area_mm2"])
    minimum_solidity = float(config["min_solidity"])
    count = 0
    for contour in contours:
        area_px = float(cv2.contourArea(contour))
        area_mm2 = area_px / (ppm * ppm)
        if not minimum_area <= area_mm2 <= maximum_area:
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area_px / hull_area if hull_area > 1e-6 else 0.0
        if solidity >= minimum_solidity:
            count += 1
    return count


def refine_generic_piece_mask(mask: np.ndarray, config: dict[str, Any],
                              diagnostics: bool = False) -> np.ndarray:
    """仅在通用模式中受限闭合反光造成的轮廓裂缝，异常时回退原掩膜。"""
    if str(config.get("puzzle_mode", "fixed_figure_2")) != "generic_geometry":
        return mask

    close_mm = float(config.get("generic_contour_close_mm", 0.0))
    if close_mm <= 0.0:
        return mask

    ppm = float(config["pixels_per_mm"])
    kernel_px = max(3, int(round(close_mm * ppm)))
    kernel_px += 1 - kernel_px % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    before_count = count_piece_candidates(mask, config)
    after_count = count_piece_candidates(closed, config)
    before_pixels = int(np.count_nonzero(mask))
    after_pixels = int(np.count_nonzero(closed))
    growth_ratio = ((after_pixels - before_pixels) / before_pixels
                    if before_pixels > 0 else math.inf)
    maximum_growth = float(config.get("generic_max_mask_growth_ratio", 0.08))
    accepted = (before_count > 0 and before_count == after_count and
                0.0 <= growth_ratio <= maximum_growth)

    if diagnostics:
        growth_text = "无穷" if not math.isfinite(growth_ratio) else f"{growth_ratio * 100:.2f}%"
        result = "采用" if accepted else "回退原掩膜"
        print(f"[诊断] 通用轮廓闭合={close_mm:.1f}mm：有效碎片"
              f"{before_count}->{after_count}，掩膜增长={growth_text}，{result}")
    return closed if accepted else mask


# 距离变换的每个值表示该像素离背景的距离。优先使用面积质心；不够安全时取满足边距的区域重心，避免偏到碎片较宽的一头。
def contour_pick_point(contour: np.ndarray, shape: tuple[int, int],
                       centroid_px: tuple[float, float] | None = None,
                       min_margin_px: float = 0.0) -> tuple[float, float]:
    """选择磁吸点：优先取质心（旋转力矩最均衡），仅在质心离边缘太近时才退让到安全区域重心。

    旧实现直接取"离边缘最远的单点"，对长条形/一头宽一头窄的碎片会系统性偏向较宽的
    一端（那里离边缘最远），导致磁吸点偏离几何中心——碎片旋转时窄的一端力臂长，
    更容易在磁力不足以抗衡的情况下打滑。新逻辑：质心本身如果已经有足够安全边距就
    直接用质心；否则不再退回单一极值点，而是取"安全边距达标区域"的重心，同样比
    单点更居中，仅在整个碎片都太窄、任何位置都不达标时才逐步放宽边距要求。
    """
    x, y, width, height = cv2.boundingRect(contour)
    padding = 3
    local = np.zeros((height + 2 * padding, width + 2 * padding), np.uint8)
    shifted = contour.astype(np.int32).copy()
    shifted[:, 0, 0] -= x - padding
    shifted[:, 0, 1] -= y - padding
    cv2.drawContours(local, [shifted], -1, 255, -1)
    distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    max_distance = float(cv2.minMaxLoc(distance)[1])

    def to_shape(local_x: float, local_y: float) -> tuple[float, float]:
        return (float(np.clip(local_x + x - padding, 0, shape[1] - 1)),
                float(np.clip(local_y + y - padding, 0, shape[0] - 1)))

    if centroid_px is not None:
        local_cx = centroid_px[0] - (x - padding)
        local_cy = centroid_px[1] - (y - padding)
        ix, iy = int(round(local_cx)), int(round(local_cy))
        if 0 <= iy < distance.shape[0] and 0 <= ix < distance.shape[1]:
            centroid_margin = float(distance[iy, ix])
            if centroid_margin >= min(min_margin_px, max_distance) and centroid_margin > 0.0:
                return to_shape(local_cx, local_cy)

    for ratio in (1.0, 0.85, 0.7, 0.5, 0.3):
        threshold = min(min_margin_px, max_distance * ratio) if min_margin_px > 0 else max_distance * ratio
        safe = distance >= max(threshold, 1e-6)
        if not np.any(safe):
            continue
        ys, xs = np.nonzero(safe)
        weights = distance[ys, xs]
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-6:
            continue
        local_x = float(np.sum(xs.astype(np.float64) * weights) / weight_sum)
        local_y = float(np.sum(ys.astype(np.float64) * weights) / weight_sum)
        return to_shape(local_x, local_y)

    maximum = cv2.minMaxLoc(distance)[3]
    return to_shape(float(maximum[0]), float(maximum[1]))


def normalize_angle(rect: tuple[Any, Any, float]) -> float:
    (_, _), (width, height), angle = rect
    if width < height:
        angle += 90.0
    return float((angle + 180.0) % 360.0 - 180.0)


def detect_pieces(paper: np.ndarray, config: dict[str, Any],
                  diagnostics: bool = False,
                  mask_override: np.ndarray | None = None, *,
                  limit_count: bool = True) -> tuple[list[Piece], np.ndarray]:
    mask = make_piece_mask(paper, config, diagnostics) if mask_override is None else mask_override.copy()
    if diagnostics and mask_override is not None:
        selected_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        mode = str(config.get("segmentation_mode", "pink_hsv"))
        print(f"[诊断] 已按{mode}生成并校正掩膜，白色像素占比={selected_ratio * 100:.2f}%")
    mask = refine_generic_piece_mask(mask, config, diagnostics)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ppm = float(config["pixels_per_mm"])
    pieces: list[Piece] = []
    border_margin = max(0, int(round(float(config.get(
        "piece_border_exclusion_mm", 2.0)) * ppm)))
    excluded_border = 0

    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if (bx <= border_margin or by <= border_margin or
                bx + bw >= mask.shape[1] - border_margin or
                by + bh >= mask.shape[0] - border_margin):
            excluded_border += 1
            continue
        area_px = cv2.contourArea(contour)
        area_mm2 = area_px / (ppm * ppm)
        if not float(config["min_piece_area_mm2"]) <= area_mm2 <= float(config["max_piece_area_mm2"]):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area_px / hull_area if hull_area > 1e-6 else 0.0
        if solidity < float(config["min_solidity"]):
            continue
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            continue
        center = (float(moments["m10"] / moments["m00"]),
                  float(moments["m01"] / moments["m00"]))
        pick_margin_px = float(config.get("pick_point_min_margin_mm", 6.0)) * ppm
        pick = contour_pick_point(contour, paper.shape[:2], center, pick_margin_px)
        rect = cv2.minAreaRect(contour)
        (_, _), (width_px, height_px), _ = rect
        pieces.append(Piece(
            0, contour, center, pick, normalize_angle(rect), float(area_mm2),
            float(width_px / ppm), float(height_px / ppm)
        ))

    pieces.sort(key=lambda item: item.area_mm2, reverse=True)
    # Planning retains its configured cap. Final verification must count every
    # eligible contour, otherwise an extra piece can disappear from acceptance.
    accepted = pieces[:int(config["max_pieces"])] if limit_count else pieces
    if diagnostics:
        print(f"[诊断] 二值轮廓={len(contours)}，边缘排除={excluded_border}，"
              f"通过面积/实心度筛选={len(accepted)}")
    return accepted, mask


# =============================================================================
# 【分区】安全搬运顺序编号
# 功能：按行聚类，靠近目标区的片先搬。扑克默认 3 片。
# 可修改：numbering_mode、row_threshold_mm。
# 看情况改：目标区位置变了才改编号方向。
# 不要改：piece_id 从 1 连续编号。
# =============================================================================
# ------------------------- 安全搬运顺序 -------------------------

def group_rows(pieces: Iterable[Piece], threshold_px: float,
               descending: bool) -> list[list[Piece]]:
    ordered = sorted(pieces, key=lambda item: item.center_px[1], reverse=descending)
    rows: list[list[Piece]] = []
    for piece in ordered:
        if not rows:
            rows.append([piece])
            continue
        row_y = float(np.mean([item.center_px[1] for item in rows[-1]]))
        if abs(piece.center_px[1] - row_y) <= threshold_px:
            rows[-1].append(piece)
        else:
            rows.append([piece])
    return rows


def number_by_move_order(pieces: list[Piece], config: dict[str, Any]) -> list[Piece]:
    """靠近下半区的碎片先搬，同一行从左到右。"""
    ppm = float(config["pixels_per_mm"])
    descending = str(config["numbering_mode"]).startswith("bottom_to_top")
    rows = group_rows(pieces, float(config["row_threshold_mm"]) * ppm, descending)
    ordered: list[Piece] = []
    for row in rows:
        row.sort(key=lambda item: item.center_px[0])
        ordered.extend(row)
    for index, piece in enumerate(ordered, 1):
        piece.piece_id = index
    return ordered


# =============================================================================
# 【分区】固定图2 / 通用几何 + 花纹验证
# 功能：几何求解后用 edge_matcher 沿拼缝做 NCC；verify_edge_patterns 把花纹结果写入计划。
# 可修改：poker_ncc_min_similarity、采样间距、向内偏移 offset_mm。
# 看情况改：几何对但花纹差，多半是拼反/转 180°，不要先放宽几何重叠。
# 不要改：刚体；花纹在源多边形上采样（未拼合时牌面完整）；边索引与求解器 EdgeMatch 一致。
# =============================================================================
# ------------------------- 固定图2拼图求解 -------------------------

def polygon_centroid(points: np.ndarray) -> np.ndarray:
    contour = np.asarray(points, np.float32).reshape(-1, 1, 2)
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        raise ValueError("碎片轮廓面积为0，无法计算质心")
    return np.asarray([moments["m10"] / moments["m00"],
                       moments["m01"] / moments["m00"]], np.float64)


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """图像坐标系旋转矩阵：正角度在画面上表现为顺时针。"""
    radians = math.radians(angle_deg)
    cosine, sine = math.cos(radians), math.sin(radians)
    return np.asarray([[cosine, -sine], [sine, cosine]], np.float64)


def rotate_points(points: np.ndarray, angle_deg: float) -> np.ndarray:
    return np.asarray(points, np.float64) @ rotation_matrix(angle_deg).T


def normalized_rotation(angle_deg: float) -> float:
    result = float((angle_deg + 180.0) % 360.0 - 180.0)
    return 180.0 if math.isclose(result, -180.0, abs_tol=1e-9) else result


def rotation_direction(angle_deg: float, deadband_deg: float = 0.25) -> str:
    if abs(angle_deg) <= deadband_deg:
        return "NONE"
    return "CW" if angle_deg > 0.0 else "CCW"


def figure2_template_polygons(config: dict[str, Any], placed: bool) -> dict[str, np.ndarray]:
    """返回题目图2的四块多边形；placed=True时平移到A4下半区目标矩形。"""
    size = np.asarray(config.get("target_rectangle_size_mm", [100.0, 60.0]), np.float64)
    if size.shape != (2,) or np.any(size <= 0.0):
        raise ValueError("target_rectangle_size_mm必须是两个正数")
    scale = size / np.asarray([100.0, 60.0], np.float64)
    offset = np.asarray(config.get("target_rectangle_top_left_mm", [55.0, 205.0]),
                        np.float64) if placed else np.zeros(2, np.float64)
    return {
        template_id: np.asarray(points, np.float64) * scale + offset
        for template_id, points in FIGURE2_TEMPLATE_POLYGONS_MM.items()
    }


def validate_target_rectangle(config: dict[str, Any]) -> None:
    top_left = np.asarray(config.get("target_rectangle_top_left_mm", [55.0, 205.0]),
                          np.float64)
    size = np.asarray(config.get("target_rectangle_size_mm", [100.0, 60.0]), np.float64)
    if top_left.shape != (2,) or size.shape != (2,):
        raise ValueError("目标矩形参数格式错误")
    bottom_right = top_left + size
    paper_width = float(config["paper_width_mm"])
    paper_height = float(config["paper_height_mm"])
    target_top = float(config.get("target_region_top_mm", paper_height / 2.0))
    if (top_left[0] < 0.0 or top_left[1] < target_top or
            bottom_right[0] > paper_width or bottom_right[1] > paper_height):
        raise ValueError(
            f"目标矩形左上({top_left[0]:.1f},{top_left[1]:.1f})mm、"
            f"尺寸({size[0]:.1f},{size[1]:.1f})mm不完整位于A4下半区"
        )


def rasterized_iou(source_local_mm: np.ndarray, angle_deg: float,
                   resolution: float, canvas_size: int,
                   origin: np.ndarray, target_mask: np.ndarray) -> float:
    rotated = rotate_points(source_local_mm, angle_deg)
    source_px = np.rint(rotated * resolution + origin).astype(np.int32)
    source_mask = np.zeros((canvas_size, canvas_size), np.uint8)
    cv2.fillPoly(source_mask, [source_px], 255)
    intersection = int(np.count_nonzero((source_mask > 0) & (target_mask > 0)))
    union = int(np.count_nonzero((source_mask > 0) | (target_mask > 0)))
    return intersection / union if union else 0.0


# 源轮廓和模板先移到各自质心附近；搜索角度使IoU最大。匹配中的尺度估计用于比较形状，不代表机械机构能缩放碎片。
def best_rotation_fit(piece: Piece, template_mm: np.ndarray,
                      config: dict[str, Any]) -> tuple[float, float, float]:
    """枚举旋转角，以轮廓IoU求源碎片到固定模板的最佳旋转。"""
    ppm = float(config["pixels_per_mm"])
    source_mm = piece.contour.reshape(-1, 2).astype(np.float64) / ppm
    source_center = polygon_centroid(source_mm)
    target_center = polygon_centroid(template_mm)
    source_area = max(float(cv2.contourArea(source_mm.astype(np.float32))), 1e-6)
    target_area = max(float(cv2.contourArea(template_mm.astype(np.float32))), 1e-6)
    fit_scale = math.sqrt(target_area / source_area)
    source_local = (source_mm - source_center) * fit_scale
    target_local = template_mm - target_center

    resolution = max(1.0, float(config.get("rotation_match_pixels_per_mm", 2.0)))
    radius = max(float(np.max(np.linalg.norm(source_local, axis=1))),
                 float(np.max(np.linalg.norm(target_local, axis=1)))) + 4.0
    canvas_size = max(64, int(math.ceil(2.0 * radius * resolution)) + 5)
    if canvas_size % 2 == 0:
        canvas_size += 1
    origin = np.asarray([canvas_size // 2, canvas_size // 2], np.float64)
    target_px = np.rint(target_local * resolution + origin).astype(np.int32)
    target_mask = np.zeros((canvas_size, canvas_size), np.uint8)
    cv2.fillPoly(target_mask, [target_px], 255)

    coarse_step = max(0.5, float(config.get("rotation_search_step_deg", 1.0)))
    best_angle, best_iou = 0.0, -1.0
    for angle in np.arange(-180.0, 180.0, coarse_step):
        iou = rasterized_iou(source_local, float(angle), resolution,
                             canvas_size, origin, target_mask)
        if iou > best_iou + 1e-9 or (math.isclose(iou, best_iou, abs_tol=1e-9)
                                     and abs(angle) < abs(best_angle)):
            best_angle, best_iou = float(angle), float(iou)

    refine_step = max(0.1, float(config.get("rotation_refine_step_deg", 0.25)))
    if refine_step < coarse_step:
        for angle in np.arange(best_angle - coarse_step, best_angle + coarse_step + 1e-9,
                               refine_step):
            iou = rasterized_iou(source_local, float(angle), resolution,
                                 canvas_size, origin, target_mask)
            if iou > best_iou + 1e-9 or (math.isclose(iou, best_iou, abs_tol=1e-9)
                                         and abs(angle) < abs(best_angle)):
                best_angle, best_iou = float(angle), float(iou)
    return normalized_rotation(best_angle), best_iou, fit_scale


# 四块对应四个模板只有4!=24种编号分配；把每种分配的匹配代价相加，再选整体最佳，而不是每片独立抢同一模板。
def solve_fixed_puzzle(pieces: list[Piece], config: dict[str, Any]) -> list[PuzzleMatch]:
    """把4块源轮廓全局匹配到图2的A/B/C/D模板。"""
    expected = int(config.get("expected_piece_count", 4))
    if len(pieces) != expected:
        raise ValueError(f"固定图2求解需要完整识别{expected}块，当前识别到{len(pieces)}块")
    validate_target_rectangle(config)
    ppm = float(config["pixels_per_mm"])
    source_limit = (float(config.get("target_region_top_mm", 148.5)) +
                    float(config.get("source_center_tolerance_mm", 5.0)))
    invalid_sources = [piece.piece_id for piece in pieces
                       if piece.center_mm(ppm)[1] > source_limit]
    if invalid_sources:
        ids = ",".join(f"P{piece_id}" for piece_id in invalid_sources)
        raise ValueError(f"{ids}的中心已进入A4下半区，请把4块初始碎片放回上半区域")
    local_templates = figure2_template_polygons(config, placed=False)
    target_templates = figure2_template_polygons(config, placed=True)
    clearance_shifts = fixed_target_clearance_shifts(target_templates, config)
    template_ids = list(FIGURE2_TEMPLATE_POLYGONS_MM)
    source_total = sum(max(piece.area_mm2, 1e-6) for piece in pieces)
    template_areas = {
        template_id: float(cv2.contourArea(polygon.astype(np.float32)))
        for template_id, polygon in local_templates.items()
    }
    global_area_scale = sum(template_areas.values()) / source_total

    metrics: dict[tuple[int, str], tuple[float, float, float]] = {}
    for piece in pieces:
        for template_id in template_ids:
            template = local_templates[template_id].astype(np.float32).reshape(-1, 1, 2)
            area_error = abs(piece.area_mm2 * global_area_scale -
                             template_areas[template_id]) / template_areas[template_id]
            shape_distance = float(cv2.matchShapes(
                piece.contour.astype(np.float32), template, cv2.CONTOURS_MATCH_I1, 0.0
            ))
            cost = area_error + 0.20 * min(shape_distance, 3.0)
            metrics[(piece.piece_id, template_id)] = (area_error, shape_distance, cost)

    best_permutation = min(
        itertools.permutations(template_ids, len(pieces)),
        key=lambda permutation: sum(
            metrics[(piece.piece_id, template_id)][2]
            for piece, template_id in zip(pieces, permutation)
        ),
    )

    max_area_error = float(config.get("max_template_area_error_ratio", 0.35))
    max_shape_distance = float(config.get("max_template_shape_distance", 0.80))
    min_iou = float(config.get("min_rotation_match_iou", 0.55))
    matches: list[PuzzleMatch] = []
    for piece, template_id in zip(pieces, best_permutation):
        local_template = local_templates[template_id]
        shift = np.asarray(clearance_shifts[template_id], np.float64)
        target_polygon = target_templates[template_id] + shift
        rotation_deg, match_iou, fit_scale = best_rotation_fit(piece, local_template, config)
        source_center = polygon_centroid(piece.contour.reshape(-1, 2) / ppm)
        target_center = polygon_centroid(target_polygon)
        source_pick = np.asarray(piece.pick_mm(ppm), np.float64)
        target_pick = target_center + rotate_points(
            (source_pick - source_center) * fit_scale, rotation_deg
        )
        inside = cv2.pointPolygonTest(target_polygon.astype(np.float32),
                                      (float(target_pick[0]), float(target_pick[1])), False) >= 0
        area_error, shape_distance, _ = metrics[(piece.piece_id, template_id)]
        status = "ok" if (area_error <= max_area_error and
                           shape_distance <= max_shape_distance and
                           match_iou >= min_iou and inside) else "match_failed"
        match = PuzzleMatch(
            piece=piece,
            template_id=template_id,
            target_polygon_mm=target_polygon,
            target_pick_mm=(float(target_pick[0]), float(target_pick[1])),
            rotation_deg=rotation_deg,
            rotation_direction=rotation_direction(rotation_deg),
            area_error_ratio=area_error,
            shape_distance=shape_distance,
            match_iou=match_iou,
            fit_scale=fit_scale,
            status=status,
            target_clearance_shift_mm=(float(shift[0]), float(shift[1])),
        )
        matches.append(match)
        print(f"[匹配] P{piece.piece_id}->模板{template_id}: "
              f"旋转={rotation_deg:+.2f}deg({match.rotation_direction}), "
              f"IoU={match_iou:.3f}, 面积误差={area_error:.3f}, 状态={status}")
    return matches


def cyclic_contour_segment(contour_points: np.ndarray,
                           start_index: int,
                           end_index: int) -> np.ndarray:
    """返回闭合轮廓中从start到end（含端点）的有序点段。"""
    points = np.asarray(contour_points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        raise ValueError("轮廓点数量不足")
    start = int(start_index) % len(points)
    end = int(end_index) % len(points)
    if start <= end:
        return points[start:end + 1].copy()
    return np.vstack((points[start:], points[:end + 1]))


def fit_contour_edge_line(edge_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """使用Huber距离为一段毛糙轮廓拟合稳定直线。"""
    points = np.asarray(edge_points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2 or float(np.max(np.linalg.norm(points - points[0], axis=1))) <= 1e-6:
        raise ValueError("轮廓边点不足或已退化")
    fitted = np.asarray(
        cv2.fitLine(points.reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01),
        dtype=np.float64,
    ).reshape(-1)
    direction = fitted[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        raise ValueError("直线拟合方向无效")
    return fitted[2:4], direction / norm


def intersect_fitted_lines(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """计算两条参数直线交点；近似平行时拒绝不稳定结果。"""
    point_a, direction_a = first
    point_b, direction_b = second
    denominator = float(
        direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    )
    if abs(denominator) <= math.sin(math.radians(3.0)):
        raise ValueError("相邻拟合边近似平行")
    delta = point_b - point_a
    distance = float(
        (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / denominator
    )
    intersection = point_a + distance * direction_a
    if not np.isfinite(intersection).all():
        raise ValueError("拟合直线交点无效")
    return intersection


def remove_near_collinear_vertices(points: np.ndarray,
                                   angle_tolerance_deg: float) -> np.ndarray:
    """移除接近180°的伪拐点，但至少保留三角形。"""
    polygon = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    tolerance = max(0.0, float(angle_tolerance_deg))
    while len(polygon) > 3:
        angles: list[float] = []
        for index in range(len(polygon)):
            previous = polygon[index - 1] - polygon[index]
            following = polygon[(index + 1) % len(polygon)] - polygon[index]
            denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
            if denominator <= 1e-9:
                angles.append(180.0)
                continue
            cosine = float(np.clip(np.dot(previous, following) / denominator, -1.0, 1.0))
            angles.append(math.degrees(math.acos(cosine)))
        remove_index = int(np.argmax(angles))
        if 180.0 - angles[remove_index] > tolerance:
            break
        polygon = np.delete(polygon, remove_index, axis=0)
    return polygon


def refine_polygon_by_edge_lines(
    contour_points: np.ndarray,
    approximated_points: np.ndarray,
    *,
    pixels_per_mm: float,
    max_vertex_shift_mm: float,
) -> np.ndarray:
    """按原轮廓边段拟合直线，并以相邻直线交点修正多边形顶点。"""
    contour = np.asarray(contour_points, dtype=np.float64).reshape(-1, 2)
    approximated = np.asarray(approximated_points, dtype=np.float64).reshape(-1, 2)
    if not 3 <= len(approximated) <= 5:
        raise ValueError("直线修正只接受3～5边初始多边形")
    if pixels_per_mm <= 0.0 or max_vertex_shift_mm <= 0.0:
        raise ValueError("直线修正配置必须为正数")

    # approxPolyDP返回的顶点来自原轮廓；使用最近点索引可保留其循环顺序。
    indices = [
        int(np.argmin(np.sum((contour - vertex) ** 2, axis=1)))
        for vertex in approximated
    ]
    if len(set(indices)) != len(indices):
        raise ValueError("简化顶点无法唯一映射回原轮廓")

    fitted_lines: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(len(indices)):
        segment = cyclic_contour_segment(
            contour, indices[index], indices[(index + 1) % len(indices)]
        )
        fitted_lines.append(fit_contour_edge_line(segment))

    refined = np.vstack([
        intersect_fitted_lines(fitted_lines[index - 1], fitted_lines[index])
        for index in range(len(fitted_lines))
    ])
    shifts_mm = np.linalg.norm(refined - approximated, axis=1) / pixels_per_mm
    if float(np.max(shifts_mm)) > max_vertex_shift_mm:
        raise ValueError(
            f"拟合顶点最大移动{float(np.max(shifts_mm)):.2f}mm超过限制"
            f"{max_vertex_shift_mm:.2f}mm"
        )
    return refined / pixels_per_mm


# 轮廓可能有上百个边缘点，而求解器只接受少量顶点；这里尝试多边形近似和边线拟合，同时约束面积变化及误差。
def _piece_polygon_models_mm(piece: Piece, config: dict[str, Any]) -> list[np.ndarray]:
    """保留面积检查合格且几何不同的少量轮廓模型。"""
    contour = np.asarray(piece.contour, np.float32).reshape(-1, 1, 2)
    contour_area_px = abs(float(cv2.contourArea(contour)))
    perimeter_px = float(cv2.arcLength(contour, True))
    if contour_area_px <= 1e-6 or perimeter_px <= 1e-6:
        raise ValueError(f"P{piece.piece_id}轮廓面积或周长无效")

    preferred = float(config.get("generic_polygon_epsilon_ratio", 0.015))
    minimum = float(config.get("generic_polygon_epsilon_min_ratio", 0.005))
    maximum = float(config.get("generic_polygon_epsilon_max_ratio", 0.05))
    if not 0.0 < minimum <= preferred <= maximum:
        raise ValueError("通用轮廓epsilon配置必须满足0 < min <= preferred <= max")

    # 优先尝试配置值，再扫描完整范围；按面积误差选择最忠实的3～5边结果。
    ratios = [preferred]
    ratios.extend(float(value) for value in np.linspace(minimum, maximum, 31))
    ratios = list(dict.fromkeys(round(value, 8) for value in ratios))
    ppm = float(config["pixels_per_mm"])
    min_edge = float(config.get("generic_min_edge_length_mm", 18.0))
    candidates: list[tuple[float, float, int, np.ndarray, str]] = []
    observed_vertices: set[int] = set()
    shortest_edge_seen = math.inf
    minimum_area = float(config.get("generic_min_piece_area_mm2", 1.0))
    max_vertex_shift = float(config.get("generic_line_fit_max_vertex_shift_mm", 6.0))

    def add_candidate(polygon_mm: np.ndarray, ratio: float, method: str) -> None:
        nonlocal shortest_edge_seen
        raw_polygon = np.asarray(polygon_mm, dtype=np.float64).reshape(-1, 2)
        try:
            normalized = generic_solver.normalize_polygon(
                raw_polygon,
                min_edge_length_mm=min_edge,
                min_area_mm2=minimum_area,
            )
        except ValueError:
            lengths = generic_solver.edge_lengths(raw_polygon)
            if len(lengths):
                shortest_edge_seen = min(shortest_edge_seen, float(np.min(lengths)))
            return
        approximated_area_px = generic_solver.polygon_area(normalized) * ppm * ppm
        area_error = abs(approximated_area_px - contour_area_px) / contour_area_px
        method_priority = 0 if method == "approx" else 1
        candidates.append((
            area_error, abs(ratio - preferred), method_priority, normalized, method
        ))

    contour_points = contour.reshape(-1, 2).astype(np.float64)
    for ratio in ratios:
        approximated = cv2.approxPolyDP(contour, ratio * perimeter_px, True)
        vertex_count = len(approximated)
        observed_vertices.add(vertex_count)
        if not 3 <= vertex_count <= 5:
            continue
        approximated_points = approximated.reshape(-1, 2).astype(np.float64)
        add_candidate(approximated_points / ppm, ratio, "approx")
        try:
            refined_mm = refine_polygon_by_edge_lines(
                contour_points,
                approximated_points,
                pixels_per_mm=ppm,
                max_vertex_shift_mm=max_vertex_shift,
            )
            refined_mm = remove_near_collinear_vertices(
                refined_mm,
                float(config.get("generic_collinear_angle_tolerance_deg", 15.0)),
            )
        except ValueError:
            continue
        add_candidate(refined_mm, ratio, "line_fit")

    if not candidates:
        vertices = ",".join(str(value) for value in sorted(observed_vertices)) or "无"
        edge_message = ("" if not math.isfinite(shortest_edge_seen) else
                        f"，最短候选边={shortest_edge_seen:.2f}mm")
        raise ValueError(
            f"P{piece.piece_id}无法简化为合格的3～5边多边形；"
            f"尝试得到顶点数={vertices}{edge_message}"
        )

    # Tiny area improvements at rounded corners must not add unstable edges.
    # Only compare candidates already inside the original area-error gate.
    max_area_error = float(config.get("generic_max_polygon_area_error_ratio", 0.08))
    best_area = min(item[0] for item in candidates)
    tolerance = max(0.0, min(0.01, float(config.get(
        "planning_polygon_area_tie_tolerance", 0.0))))
    faithful = [item for item in candidates
                if item[0] <= min(max_area_error, best_area + tolerance)]
    if tolerance > 0.0 and faithful:
        chosen = min(faithful, key=lambda item: (len(item[3]), item[0], item[1], item[2]))
    else:
        chosen = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    area_error, _, _, polygon, method = chosen
    max_area_error = float(config.get("generic_max_polygon_area_error_ratio", 0.08))
    if area_error > max_area_error:
        raise ValueError(
            f"P{piece.piece_id}轮廓简化面积误差{area_error:.3f}超过限制{max_area_error:.3f}"
        )
    if method == "line_fit":
        raw_errors = [item[0] for item in candidates if item[4] == "approx"]
        before = min(raw_errors) if raw_errors else math.inf
        before_text = "无合格原始候选" if not math.isfinite(before) else f"{before:.3f}"
        print(
            f"[诊断] P{piece.piece_id}直边拟合：面积误差{before_text}->{area_error:.3f}，"
            f"顶点={len(polygon)}"
        )
    models = [polygon]
    if config.get("planning_polygon_alternatives", False):
        for item in sorted(candidates, key=lambda item: (item[0], item[1], item[2])):
            if item[0] > max_area_error:
                continue
            candidate = item[3]
            if any(len(old) == len(candidate) and min(
                    float(np.max(np.linalg.norm(old - np.roll(candidate, shift, axis=0), axis=1)))
                    for shift in range(len(candidate))) <= 0.5 for old in models):
                continue
            models.append(candidate)
            if len(models) == 3:
                break
    return models


def piece_polygon_mm(piece: Piece, config: dict[str, Any]) -> np.ndarray:
    """将视觉轮廓简化为3～5顶点的毫米多边形。"""
    return _piece_polygon_models_mm(piece, config)[0]


def _point_segment_projection(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 1e-12:
        return start.copy()
    ratio = float(np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0))
    return start + ratio * segment


def _polygon_closest_vector(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, np.ndarray]:
    # Vertex-to-edge distance alone misses crossing edges and can report a
    # positive gap for overlapping convex polygons. Return signed penetration
    # and the minimum separating direction in that case.
    first = np.asarray(first, np.float64)
    second = np.asarray(second, np.float64)
    # Collinear split vertices can acquire tiny negative cross products when
    # converted to float32. Use the hull only for numerically convex polygons.
    hulls = [cv2.convexHull(poly.astype(np.float32)).reshape(-1, 2).astype(np.float64)
             for poly in (first, second)]
    if all(abs(cv2.contourArea(hull.astype(np.float32)) -
               cv2.contourArea(poly.astype(np.float32))) < 0.001
           for poly, hull in zip((first, second), hulls)):
        first, second = hulls
        penetration = math.inf
        direction = None
        for polygon in (first, second):
            for edge in np.roll(polygon, -1, axis=0) - polygon:
                length = float(np.linalg.norm(edge))
                if length <= 1e-9:
                    continue
                axis = np.array([-edge[1], edge[0]]) / length
                a, b = first @ axis, second @ axis
                plus, minus = float(a.max() - b.min()), float(b.max() - a.min())
                if min(plus, minus) <= 0:
                    penetration = 0.0
                    break
                if min(plus, minus) < penetration:
                    penetration = min(plus, minus)
                    direction = axis if plus <= minus else -axis
            if penetration == 0:
                break
        if penetration > 0 and direction is not None:
            return -penetration, direction
    best_distance_squared = math.inf
    best_vector = np.zeros(2, np.float64)
    for point in first:
        for index in range(len(second)):
            projected = _point_segment_projection(
                point, second[index], second[(index + 1) % len(second)]
            )
            vector = projected - point
            distance_squared = float(np.dot(vector, vector))
            if distance_squared < best_distance_squared:
                best_distance_squared = distance_squared
                best_vector = vector
    for point in second:
        for index in range(len(first)):
            projected = _point_segment_projection(
                point, first[index], first[(index + 1) % len(first)]
            )
            vector = point - projected
            distance_squared = float(np.dot(vector, vector))
            if distance_squared < best_distance_squared:
                best_distance_squared = distance_squared
                best_vector = vector
    return math.sqrt(max(0.0, best_distance_squared)), best_vector


def _clearance_adjacencies(
    polygons: list[np.ndarray],
    tolerance_mm: float,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for first in range(len(polygons)):
        for second in range(first + 1, len(polygons)):
            distance, _ = _polygon_closest_vector(polygons[first], polygons[second])
            if distance <= tolerance_mm:
                pairs.append((first, second))
    return pairs


# 先沿各片质心远离矩形中心的方向平移，再迭代拉开仍过近的邻片。每轮去掉平均平移并限制单片最大移动量，最后检查间隙、重叠和纸面边界。
def apply_generic_target_clearance(
    solution: generic_solver.GenericSolution,
    config: dict[str, Any],
) -> dict[int, tuple[float, float]]:
    """在保持整体矩形外观的前提下，为相邻目标碎片加入受限间隙。"""
    piece_count = len(solution.poses)
    clearance = float(config.get("generic_target_piece_clearance_mm", 0.0))
    rule_limit = float(config.get("target_corresponding_vertex_limit_mm", 20.0))
    if clearance < 0.0:
        raise ValueError("目标碎片间隙不能为负数")
    if rule_limit <= 0.0:
        raise ValueError("对应顶点距离限制必须大于0")
    if clearance > rule_limit + 1e-9:
        raise ValueError(
            f"目标间隙{clearance:.1f}mm超过对应顶点限制{rule_limit:.1f}mm"
        )
    if piece_count <= 1 or clearance <= 1e-9:
        return {pose.piece_index: (0.0, 0.0) for pose in solution.poses}

    poses = sorted(solution.poses, key=lambda item: item.piece_index)
    base_polygons = [pose.target_polygon_mm.copy() for pose in poses]
    rectangle_center = (
        np.asarray(solution.target_top_left_mm, np.float64)
        + np.asarray(solution.target_size_mm, np.float64) / 2.0
    )
    initial_offset = clearance / 2.0
    shifts = np.zeros((piece_count, 2), np.float64)
    for index, polygon in enumerate(base_polygons):
        direction = polygon_centroid(polygon) - rectangle_center
        length = float(np.linalg.norm(direction))
        direction = (np.asarray([1.0, 0.0], np.float64)
                     if length <= 1e-9 else direction / length)
        shifts[index] = direction * initial_offset

    adjacency_tolerance = float(
        config.get("generic_clearance_adjacency_tolerance_mm", 3.0)
    )
    adjacent_pairs = {
        tuple(sorted((match.piece_a, match.piece_b))) for match in solution.matches
    }
    # 求解器拼缝与几何实际接触共同定义相邻关系，避免漏掉闭合边。
    adjacent_pairs.update(_clearance_adjacencies(base_polygons, adjacency_tolerance))
    adjacent_pairs = sorted(adjacent_pairs)

    minimum_gap_ratio = float(config.get("generic_clearance_min_achieved_ratio", 0.8))
    if not (0.0 < minimum_gap_ratio <= 1.0):
        raise ValueError("最小间隙达成比例必须位于(0,1]范围")
    minimum_gap = clearance * minimum_gap_ratio
    max_shift_ratio = float(config.get("generic_clearance_max_piece_shift_ratio", 0.9))
    if max_shift_ratio < 0.5:
        raise ValueError("单片最大位移比例不能小于0.5")
    absolute_shift_limit = float(
        config.get("generic_clearance_max_piece_shift_mm", rule_limit / 2.0)
    )
    if absolute_shift_limit <= 0.0:
        raise ValueError("????????????0")
    max_piece_shift = min(absolute_shift_limit, clearance * max_shift_ratio)
    if max_piece_shift + 1e-9 < initial_offset:
        raise ValueError("单片最大位移小于初始外移距离")

    iterations = int(config.get("generic_clearance_relax_iterations", 600))
    for _ in range(max(1, iterations)):
        changed = False
        for first, second in adjacent_pairs:
            first_polygon = base_polygons[first] + shifts[first]
            second_polygon = base_polygons[second] + shifts[second]
            distance, vector = _polygon_closest_vector(first_polygon, second_polygon)
            deficit = minimum_gap - distance
            if deficit <= 1e-4:
                continue
            vector_length = float(np.linalg.norm(vector))
            if vector_length <= 1e-8:
                vector = polygon_centroid(second_polygon) - polygon_centroid(first_polygon)
                vector_length = float(np.linalg.norm(vector))
            if vector_length <= 1e-8:
                continue
            correction = vector / vector_length * (deficit * 0.25)
            shifts[first] -= correction
            shifts[second] += correction
            changed = True

        shifts -= np.mean(shifts, axis=0)
        for index, shift in enumerate(shifts):
            length = float(np.linalg.norm(shift))
            if length > max_piece_shift:
                shifts[index] = shift * (max_piece_shift / length)
        if not changed:
            break

    shifted_polygons = [
        polygon + shifts[index] for index, polygon in enumerate(base_polygons)
    ]
    actual_gaps = [
        _polygon_closest_vector(shifted_polygons[first], shifted_polygons[second])[0]
        for first, second in adjacent_pairs
    ]
    actual_minimum_gap = min(actual_gaps, default=minimum_gap)
    if actual_minimum_gap + 0.02 < minimum_gap:
        raise ValueError(
            f"受限松弛达到单片最大位移{max_piece_shift:.1f}mm后仍未满足间隙："
            f"实际最小间隙{actual_minimum_gap:.1f}mm，小于要求{minimum_gap:.1f}mm"
        )

    for first, second in adjacent_pairs:
        corresponding_distance = float(np.linalg.norm(shifts[second] - shifts[first]))
        if corresponding_distance > rule_limit + 1e-6:
            raise ValueError(
                f"留缝后对应顶点距离{corresponding_distance:.1f}mm超过限制{rule_limit:.1f}mm"
            )

    overlap = generic_solver.polygon_overlap_ratio(
        shifted_polygons,
        float(config.get("generic_clearance_check_pixels_per_mm", 5.0)),
    )
    overlap_limit = float(config.get("generic_clearance_max_overlap_ratio", 0.001))
    if overlap > overlap_limit:
        raise ValueError(
            f"留缝后目标碎片重叠率超限：实际={overlap:.4f}，限制={overlap_limit:.4f}"
        )

    bounds = np.vstack(shifted_polygons)
    minimum = np.min(bounds, axis=0)
    maximum = np.max(bounds, axis=0)
    paper_width = float(config.get("paper_width_mm", 210.0))
    paper_height = float(config.get("paper_height_mm", 297.0))
    separator = float(config.get("separator_line_y_mm",
                                config.get("target_region_top_mm", paper_height / 2.0)))
    target_region = str(config.get("target_region", "bottom")).lower()
    if target_region == "top":
        target_min, target_max = 0.0, min(separator, paper_height)
    else:
        target_min, target_max = max(0.0, min(separator, paper_height)), paper_height
    if (minimum[0] < -1e-6 or minimum[1] < target_min - 1e-6 or
            maximum[0] > paper_width + 1e-6 or maximum[1] > target_max + 1e-6):
        raise ValueError("留缝后的目标布局超出配置目标区域")

    result: dict[int, tuple[float, float]] = {}
    for index, pose in enumerate(poses):
        shift = shifts[index]
        pose.transform_3x3 = generic_solver.rigid_matrix(0.0, shift) @ pose.transform_3x3
        pose.target_polygon_mm = shifted_polygons[index]
        result[pose.piece_index] = (float(shift[0]), float(shift[1]))

    maximum_shift = max(float(np.linalg.norm(shift)) for shift in shifts)
    print(
        f"[布局] 配置间隙={clearance:.1f}mm，真实轮廓最小间隙={actual_minimum_gap:.1f}mm，"
        f"最大单片偏移={maximum_shift:.1f}/{max_piece_shift:.1f}mm，"
        f"对应顶点限制={rule_limit:.1f}mm，重叠率={overlap:.4f}"
    )
    return result

# 本函数原文有乱码。实际是复用通用留缝函数：要求至少达到配置间隙的90%，单片移动上限设为配置间隙。
def fixed_target_clearance_shifts(
    target_templates: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    """????2???????????????????????"""
    template_ids = list(FIGURE2_TEMPLATE_POLYGONS_MM)
    clearance = float(config.get("fixed_target_piece_clearance_mm", 0.0))
    if clearance <= 1e-9:
        return {template_id: (0.0, 0.0) for template_id in template_ids}

    top_left = tuple(float(value) for value in
                     config.get("target_rectangle_top_left_mm", [55.0, 205.0]))
    size = tuple(float(value) for value in
                 config.get("target_rectangle_size_mm", [100.0, 60.0]))
    poses = [
        generic_solver.GenericPiecePose(index, np.eye(3, dtype=np.float64),
                                        target_templates[template_id].copy())
        for index, template_id in enumerate(template_ids)
    ]
    fixed_solution = generic_solver.GenericSolution(
        poses=poses,
        target_size_mm=(size[0], size[1]),
        target_top_left_mm=(top_left[0], top_left[1]),
        score=0.0, fill_error_ratio=0.0, overlap_ratio=0.0,
        # boundary_gap_ratio=0.0（矩形边界缺口比例）；取值过程：0.0。
        endpoint_error_mm=0.0, boundary_error_mm=0.0, boundary_gap_ratio=0.0,
        matches=[], search_states=0,
    )
    clearance_config = dict(config)
    clearance_config["generic_target_piece_clearance_mm"] = clearance
    # ?2?????????????10mm??20mm?????????
    # ????????????????90%?
    clearance_config["generic_clearance_min_achieved_ratio"] = 0.9
    clearance_config["generic_clearance_max_piece_shift_ratio"] = 1.0
    clearance_config["generic_clearance_max_piece_shift_mm"] = clearance
    shifts = apply_generic_target_clearance(fixed_solution, clearance_config)
    return {template_id: shifts[index] for index, template_id in enumerate(template_ids)}


def verify_edge_patterns(paper: np.ndarray, pieces: list[Piece],
                         polygons: list[np.ndarray],
                         solution: generic_solver.GenericSolution,
                         config: dict[str, Any]) -> dict | None:
    """方案B：沿拼缝边提取牌面花纹做NCC互相关，验证几何拼法花纹对应正确。

    拼缝边对直接取自求解器solution.matches（EdgeMatch含piece_a/edge_a/
    piece_b/edge_b，索引属于求解器拆边后的多边形），不受目标布局间隙影响；花纹在源
    碎片多边形上采样（未拼合时牌面完整）。刚体变换保持边顺序不变，因此
    源边索引可直接与拼缝边对应。返回汇总dict；无拼缝或缺少edge_matcher
    时返回None。
    """
    if not _PATTERN_MATCHER_AVAILABLE or paper is None:
        return None
    ppm = float(config["pixels_per_mm"])
    spacing = float(config.get("poker_edge_sample_spacing_mm", 1.0))
    offset = float(config.get("poker_edge_offset_mm", 2.0))

    # EdgeMatch indices belong to the solver's (possibly split) polygons.
    # Recover those source edges through the inverse rigid transform; the original
    # detector polygon has different indices after a collinear edge is split.
    poly_by_index = {
        pose.piece_index: generic_solver.transform_points(
            pose.target_polygon_mm, np.linalg.inv(pose.transform_3x3))
        for pose in solution.poses
    }
    id_by_index = {index: piece.piece_id for index, piece in enumerate(pieces)}
    profiles: dict[int, list] = {}
    for index, piece in enumerate(pieces):
        poly_mm = poly_by_index.get(index)
        if poly_mm is None:
            continue
        try:
            profiles[index] = edge_matcher.extract_edge_profiles(
                paper, piece.piece_id, poly_mm, ppm,
                sample_spacing_mm=spacing, offset_mm=offset,
            )
        except Exception as exc:
            print(f"[花纹] P{piece.piece_id}边缘花纹提取失败：{exc}")

    pairs: list[dict[str, Any]] = []
    similarities: list[float] = []
    for edge_match in solution.matches:
        profs_a = profiles.get(edge_match.piece_a, [])
        profs_b = profiles.get(edge_match.piece_b, [])
        by_edge_a = {profile.edge_idx: profile for profile in profs_a}
        by_edge_b = {profile.edge_idx: profile for profile in profs_b}
        if edge_match.edge_a in by_edge_a and edge_match.edge_b in by_edge_b:
            ncc = edge_matcher.profile_ncc(
                by_edge_a[edge_match.edge_a].profile_gray,
                by_edge_b[edge_match.edge_b].profile_gray,
            )
        else:
            ncc = 0.0
        similarity = float((ncc + 1.0) / 2.0)
        similarities.append(similarity)
        shared_mm = 0.0
        try:
            lengths = generic_solver.edge_lengths(poly_by_index[edge_match.piece_a])
            if edge_match.edge_a < len(lengths):
                shared_mm = float(lengths[edge_match.edge_a])
        except Exception:
            pass
        pairs.append({
            "piece_a": id_by_index.get(edge_match.piece_a, edge_match.piece_a),
            "edge_a": edge_match.edge_a,
            "piece_b": id_by_index.get(edge_match.piece_b, edge_match.piece_b),
            "edge_b": edge_match.edge_b,
            "shared_mm": round(shared_mm, 2),
            "ncc": round(float(ncc), 4),
            "similarity": round(similarity, 4),
        })

    if not similarities:
        return None
    average = float(np.mean(similarities))
    minimum = float(np.min(similarities))
    if average >= 0.85:
        message = f"花纹吻合度: {average:.0%} (高度匹配)"
    elif average >= 0.65:
        message = f"花纹吻合度: {average:.0%} (中度匹配)"
    elif average >= 0.50:
        message = f"花纹吻合度: {average:.0%} (偏低，请核实)"
    else:
        message = f"花纹吻合度: {average:.0%} (不匹配，拼法可能错误)"
    return {
        "score": round(average, 4),
        "min_similarity": round(minimum, 4),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "message": message,
    }

# 磁吸点与多边形必须使用同一个刚体矩阵，否则碎片轮廓摆对了，机械臂却可能把吸取点送到错误位置。
def solve_generic_puzzle(pieces: list[Piece], config: dict[str, Any],
                         paper: np.ndarray | None = None) -> list[PuzzleMatch]:
    """求解1～4块任意多边形，并转换为现有路径模块可消费的匹配结果。

    paper为校正后的A4彩色图，仅当config启用poker_pattern_verification时
    用于对几何解做相邻边花纹NCC验证（扑克牌模式）。
    """
    if not 1 <= len(pieces) <= int(config.get("max_pieces", 4)):
        raise ValueError(f"通用拼图需要识别1～4块，当前识别到{len(pieces)}块")
    ppm = float(config["pixels_per_mm"])
    source_min, source_max = _region_bounds_mm(config, "source")
    source_tolerance = float(config.get("source_center_tolerance_mm", 5.0))
    invalid_sources = [piece.piece_id for piece in pieces
                       if not (source_min - source_tolerance <= piece.center_mm(ppm)[1]
                               <= source_max + source_tolerance)]
    if invalid_sources:
        ids = ",".join(f"P{piece_id}" for piece_id in invalid_sources)
        label = "上半区" if str(config.get("source_region", "top")) == "top" else "下半区"
        raise ValueError(f"{ids}的中心不在A4{label}源区，请检查碎片摆放")

    clearance_shifts = None
    if config.get("planning_polygon_alternatives", False):
        models = [_piece_polygon_models_mm(piece, config) for piece in pieces]
        orders = sorted(itertools.product(*(range(len(group)) for group in models)),
                        key=lambda order: (sum(order), order))[:12]
        budget = generic_solver._make_search_budget(config)
        failures = []
        solution = None
        for order in orders:
            if budget.exhausted():
                break
            polygons = [group[index] for group, index in zip(models, order)]
            try:
                solution = generic_solver.solve_generic_rectangle(polygons, config, budget=budget)
                reference_groups = config.get("_planning_reference_groups")
                if reference_groups:
                    clearance_shifts = apply_generic_target_clearance(solution, config)
                    if not all(_solution_matches_reference(solution, pieces, reference, config)
                               for reference in reference_groups):
                        raise ValueError("合格几何候选与连续帧姿态不一致，继续搜索")
                if any(order):
                    print(f"[诊断] 备选轮廓{order}通过原有矩形检查")
                break
            except ValueError as exc:
                solution = None
                clearance_shifts = None
                failures.append(str(exc))
        else:
            raise ValueError(failures[-1] if failures else "没有合格轮廓组合")
        if solution is None:
            raise ValueError(f"轮廓候选搜索{budget.stop_reason}：" + (failures[-1] if failures else "无解"))
    else:
        polygons = [piece_polygon_mm(piece, config) for piece in pieces]
        solution = generic_solver.solve_generic_rectangle(polygons, config)
    # 终极兜底解（碎片已各自平铺、互不重叠）不需要再做留缝处理，
    # 直接给零位移，避免对非拼合布局错误地施加间隙。
    if solution.relaxed_override and solution.score > 1e8:
        clearance_shifts = {index: (0.0, 0.0) for index in range(len(pieces))}
        print("[终极兜底] 平铺解跳过留缝处理（碎片已独立放置）")
    elif clearance_shifts is None:
        clearance_shifts = apply_generic_target_clearance(solution, config)
    poses = {pose.piece_index: pose for pose in solution.poses}
    matches: list[PuzzleMatch] = []
    for index, piece in enumerate(pieces):
        pose = poses[index]
        source_pick = np.asarray([piece.pick_mm(ppm)], np.float64)
        target_pick = generic_solver.transform_points(source_pick, pose.transform_3x3)[0]
        inside = cv2.pointPolygonTest(
            pose.target_polygon_mm.astype(np.float32),
            (float(target_pick[0]), float(target_pick[1])), False,
        ) >= 0
        # 终极兜底解是"平铺不拼合"，磁吸点随碎片整体平移，
        # 不做磁吸点几何校验，避免质量无关的误拦截。
        if not inside and not (solution.relaxed_override and solution.score > 1e8):
            raise ValueError(f"P{piece.piece_id}变换后的磁吸点不在目标多边形内")
        angle = normalized_rotation(generic_solver.rotation_angle_deg(pose.transform_3x3))
        match = PuzzleMatch(
            piece=piece,
            template_id=f"G{index + 1}",
            target_polygon_mm=pose.target_polygon_mm,
            target_pick_mm=(float(target_pick[0]), float(target_pick[1])),
            rotation_deg=angle,
            rotation_direction=rotation_direction(angle),
            area_error_ratio=solution.fill_error_ratio,
            shape_distance=solution.boundary_error_mm,
            match_iou=max(0.0, 1.0 - solution.fill_error_ratio),
            fit_scale=1.0,
            status="ok",
            solver_mode="generic_geometry",
            transform_3x3=pose.transform_3x3.copy(),
            solution_score=solution.score,
            fill_error_ratio=solution.fill_error_ratio,
            overlap_ratio=solution.overlap_ratio,
            endpoint_error_mm=solution.endpoint_error_mm,
            boundary_error_mm=solution.boundary_error_mm,
            boundary_gap_ratio=solution.boundary_gap_ratio,
            target_rectangle_top_left_mm=solution.target_top_left_mm,
            target_rectangle_size_mm=solution.target_size_mm,
            target_clearance_shift_mm=clearance_shifts[index],
            relaxed_override=solution.relaxed_override,
        )
        matches.append(match)
        print(f"[通用求解] P{piece.piece_id}->{match.template_id}: "
              f"旋转={angle:+.2f}deg({match.rotation_direction}), "
              f"顶点={len(polygons[index])}, 状态=ok")
    print(f"[通用求解] 矩形={solution.target_size_mm[0]:.1f}×"
          f"{solution.target_size_mm[1]:.1f}mm, 评分={solution.score:.4f}, "
          f"填充误差={solution.fill_error_ratio:.4f}, "
          f"边界缺口={solution.boundary_gap_ratio:.4f}, "
          f"重叠率={solution.overlap_ratio:.4f}, 搜索状态={solution.search_states}")

    if paper is not None and bool(config.get("poker_pattern_verification", False)):
        verification = verify_edge_patterns(paper, pieces, polygons, solution, config)
        if verification is not None:
            for match in matches:
                match.pattern_verification = verification
            print(f"[花纹] 拼缝{verification['n_pairs']}对，"
                  f"相似度avg={verification['score']:.3f} "
                  f"min={verification['min_similarity']:.3f} {verification['message']}")
    return matches


def solve_puzzle(pieces: list[Piece], config: dict[str, Any],
                 paper: np.ndarray | None = None) -> list[PuzzleMatch]:
    """按配置分发固定图2或通用几何求解器。"""
    mode = str(config.get("puzzle_mode", "fixed_figure_2"))
    if mode == "fixed_figure_2":
        return solve_fixed_puzzle(pieces, config)
    if mode == "generic_geometry":
        return solve_generic_puzzle(pieces, config, paper)
    raise ValueError(f"未知puzzle_mode：{mode}")


def mm_to_px(point: tuple[float, float] | list[float], ppm: float) -> tuple[float, float]:
    return float(point[0]) * ppm, float(point[1]) * ppm


def px_to_mm(point: tuple[float, float], ppm: float) -> tuple[float, float]:
    return point[0] / ppm, point[1] / ppm

# =============================================================================
# 【分区】A4 最短正交路径（转调 orthogonal_path_planner）
# 功能：毫米↔像素；每片吸取点到目标点的 L 形路径。
# 可修改：先 X 后 Y 在路径模块里改。
# 看情况改：本层只转发。
# 不要改：path_px 单位像素。
# =============================================================================
# ------------------------- A4最短正交路径 -------------------------

def axis_aligned(start: tuple[float, float], end: tuple[float, float],
                 tolerance: float = 1e-6) -> bool:
    """兼容原调用接口，实际实现位于独立路径规划模块。"""
    return path_planner.axis_aligned(start, end, tolerance)


def compact_orthogonal_path(
        points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """兼容原调用接口，压缩正交路径中的冗余点。"""
    return path_planner.compact_orthogonal_path(points)


# 这份实现规划无避障约束的最短正交路径：只保证路径在工作区内并且每段单轴运动，不搜索绕开其他碎片的路线。
def plan_single_path(start: tuple[float, float], goal: tuple[float, float],
                     obstacle: np.ndarray | None, config: dict[str, Any]
                     ) -> tuple[list[tuple[float, float]], str, str]:
    """规划A4范围内的最短正交路径；不使用外围通道和碎片障碍。"""
    del obstacle
    plan = path_planner.plan_shortest_orthogonal_path(
        start, goal, workspace_size_px(config)
    )
    return plan.points, plan.planner, plan.status


def plan_all_paths(pieces: list[Piece], config: dict[str, Any],
                   matches: list[PuzzleMatch] | None = None) -> list[PathResult]:
    ppm = float(config["pixels_per_mm"])
    matches = solve_puzzle(pieces, config) if matches is None else matches
    if [match.piece.piece_id for match in matches] != [piece.piece_id for piece in pieces]:
        raise ValueError("模板匹配结果与搬运顺序不一致")

    if not all(match.status == "ok" for match in matches):
        return [PathResult(match, [match.piece.pick_px], "none", "match_failed")
                for match in matches]

    results: list[PathResult] = []
    for match in matches:
        piece = match.piece
        target_px = mm_to_px(match.target_pick_mm, ppm)
        path, planner, status = plan_single_path(piece.pick_px, target_px, None, config)
        results.append(PathResult(match, path, planner, status))
    return results

# =============================================================================
# 【分区】可视化、安全检查与 plan.json 输出
# 功能：预览图、路径安全、写出 plan（含 poker_pattern_verification）。
# 可修改：预览颜色；安全阈值走配置。
# 看情况改：花纹未过不要强行 ready_for_motion。
# 不要改：plan 字段名；仿真 pattern_gate 按这些键读。
# =============================================================================
# ------------------------- 可视化与文件输出 -------------------------

def draw_preview(paper: np.ndarray, pieces: list[Piece], paths: list[PathResult],
                 config: dict[str, Any]) -> np.ndarray:
    workspace_width, workspace_height = workspace_size_px(config)
    if paper.shape[1] > workspace_width or paper.shape[0] > workspace_height:
        raise ValueError("A4校正图尺寸超过机械工作区")
    display = np.full((workspace_height, workspace_width, 3), (55, 55, 55), np.uint8)
    display[:paper.shape[0], :paper.shape[1]] = paper
    ppm = float(config["pixels_per_mm"])
    target_y = round(float(config.get("target_region_top_mm", 148.5)) * ppm)
    cv2.line(display, (0, target_y), (display.shape[1] - 1, target_y),
             (0, 255, 255), 2)
    colors = [(0, 255, 0), (255, 180, 0), (255, 0, 255), (0, 180, 255)]
    overlay = display.copy()
    for result in paths:
        color = colors[(result.piece.piece_id - 1) % len(colors)]
        target_polygon = np.rint(result.match.target_polygon_mm * ppm).astype(np.int32)
        cv2.fillPoly(overlay, [target_polygon], color)
    display = cv2.addWeighted(overlay, 0.14, display, 0.86, 0)

    for piece in pieces:
        color = colors[(piece.piece_id - 1) % len(colors)]
        cv2.drawContours(display, [piece.contour.astype(np.int32)], -1, color, 2)
        pick = tuple(map(round, piece.pick_px))
        center_mm = piece.center_mm(ppm)
        cv2.drawMarker(display, pick, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(display, f"P{piece.piece_id} ({center_mm[0]:.1f},{center_mm[1]:.1f})mm",
                    (pick[0] + 8, pick[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)

    for result in paths:
        color = colors[(result.piece.piece_id - 1) % len(colors)]
        target_polygon = np.rint(result.match.target_polygon_mm * ppm).astype(np.int32)
        cv2.polylines(display, [target_polygon], True, color, 2, cv2.LINE_AA)
        points = [tuple(map(round, point)) for point in result.path_px]
        if len(points) >= 2:
            cv2.polylines(display, [np.asarray(points, np.int32)], False,
                          color, 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(display, point, 5, color, -1)
        target = tuple(map(round, mm_to_px(result.target_mm, ppm)))
        cv2.circle(display, target, 9, color, 2)
        label = (f"T{result.match.template_id}/P{result.piece.piece_id} "
                 f"{result.match.rotation_deg:+.1f}deg")
        cv2.putText(display, label, (target[0] + 8, target[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    cv2.putText(display, "Move order: optimized for shortest total XY travel",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(display, "Path: X/Y only, one axis at a time; Rotation: +CW / -CCW",
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    top_left, size = target_rectangle_from_paths(paths, config)
    rectangle_start = tuple(map(round, mm_to_px(top_left, ppm)))
    rectangle_end = tuple(map(round, mm_to_px(
        (top_left[0] + size[0], top_left[1] + size[1]), ppm
    )))
    cv2.rectangle(display, rectangle_start, rectangle_end, (255, 255, 0), 2)
    mode = str(config.get("puzzle_mode", "fixed_figure_2"))
    cv2.putText(display, f"Mode: {mode}", (15, 82), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 0), 2, cv2.LINE_AA)

    ready_for_motion = not motion_safety_reasons(paths, config)
    if not ready_for_motion:
        cv2.rectangle(display, (8, 66), (display.shape[1] - 8, 108),
                      (20, 20, 190), -1)
        cv2.putText(display, "MATCH/PATH FAILED - NO MOTION",
                    (18, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                    (255, 255, 255), 2, cv2.LINE_AA)
        for result in paths:
            if result.match.status == "ok" and result.status == "ok":
                continue
            pick = tuple(map(round, result.piece.pick_px))
            reason = "FAIL" if result.match.status != "ok" else "LOCKED"
            cv2.putText(display,
                        f"P{result.piece.piece_id}/{result.match.template_id} {reason}",
                        (pick[0] + 8, pick[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return display


def axis_motion_steps(path_mm: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """将正交折线路径转换为一次只移动一个轴的指令。"""
    commands: list[dict[str, Any]] = []
    for start, end in zip(path_mm, path_mm[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        if abs(dx) > 1e-6 and abs(dy) > 1e-6:
            raise ValueError("路径包含斜线，不能生成单轴运动指令")
        if abs(dx) > 1e-6:
            axis, delta = "X", dx
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > 1e-6:
            axis, delta = "Y", dy
            direction = "DOWN" if dy > 0 else "UP"
        else:
            continue
        commands.append({
            "axis": axis,
            "direction": direction,
            "delta_mm": round(delta, 2),
            "distance_mm": round(abs(delta), 2),
            "from_mm": [round(start[0], 2), round(start[1], 2)],
            "to_mm": [round(end[0], 2), round(end[1], 2)],
        })
    return commands


def target_rectangle_from_paths(paths: list[PathResult],
                                config: dict[str, Any]) -> tuple[tuple[float, float],
                                                                  tuple[float, float]]:
    """返回求解器实际目标矩形；固定模式继续使用原配置。"""
    if paths:
        match = paths[0].match
        if (match.target_rectangle_top_left_mm is not None and
                match.target_rectangle_size_mm is not None):
            return match.target_rectangle_top_left_mm, match.target_rectangle_size_mm
    top_left = tuple(float(value) for value in
                     config.get("target_rectangle_top_left_mm", [55.0, 205.0]))
    size = tuple(float(value) for value in
                 config.get("target_rectangle_size_mm", [100.0, 60.0]))
    return (top_left[0], top_left[1]), (size[0], size[1])


# 空原因列表表示本函数检查通过。宽松或强制执行分支跳过部分几何质量阈值；低花纹相似度仅打印提示，没有加入拒绝原因。
def motion_safety_reasons(paths: list[PathResult], config: dict[str, Any]) -> list[str]:
    """集中验证数量、求解质量、目标区域和严格正交工作区路径。"""
    reasons: list[str] = []
    mode = str(config.get("puzzle_mode", "fixed_figure_2"))
    count = len(paths)
    if mode == "fixed_figure_2" and count != int(config.get("expected_piece_count", 4)):
        reasons.append(f"固定图2路径数量错误：{count}")
    elif mode == "generic_geometry" and not 1 <= count <= int(config.get("max_pieces", 4)):
        reasons.append(f"通用拼图路径数量错误：{count}")
    elif mode not in {"fixed_figure_2", "generic_geometry"}:
        reasons.append(f"未知puzzle_mode：{mode}")
    if not paths:
        reasons.append("没有可执行碎片路径")
        return reasons

    ppm = float(config["pixels_per_mm"])
    workspace_width_px, workspace_height_px = workspace_size_px(config)
    paper_width = float(config["paper_width_mm"])
    paper_height = float(config["paper_height_mm"])
    separator = float(config.get("separator_line_y_mm",
                                config.get("target_region_top_mm", paper_height / 2.0)))
    target_region = str(config.get("target_region", "bottom")).lower()
    if target_region == "top":
        target_min, target_max = 0.0, min(separator, paper_height)
    elif target_region == "bottom":
        target_min, target_max = max(0.0, min(separator, paper_height)), paper_height
    else:
        reasons.append(f"未知target_region：{target_region}")
        target_min, target_max = 0.0, paper_height
    point_tolerance = float(config.get("target_pick_transform_tolerance_mm", 0.5))

    for result in paths:
        piece_id = result.piece.piece_id
        match = result.match
        if match.status != "ok":
            reasons.append(f"P{piece_id}求解状态={match.status}")
        if result.status != "ok":
            reasons.append(f"P{piece_id}路径状态={result.status}")
        polygon = np.asarray(match.target_polygon_mm, np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
            reasons.append(f"P{piece_id}目标多边形无效")
        else:
            minimum = np.min(polygon, axis=0)
            maximum = np.max(polygon, axis=0)
            if (minimum[0] < -1e-6 or minimum[1] < target_min - 1e-6 or
                    maximum[0] > paper_width + 1e-6 or maximum[1] > target_max + 1e-6):
                label = "上半区" if target_region == "top" else "下半区"
                reasons.append(f"P{piece_id}目标多边形未完整位于A4{label}")
            if cv2.pointPolygonTest(
                    polygon.astype(np.float32), match.target_pick_mm, False) < 0:
                reasons.append(f"P{piece_id}目标磁吸点位于目标多边形外")

        for x, y in result.path_px:
            if not (0.0 <= x < workspace_width_px and 0.0 <= y < workspace_height_px):
                reasons.append(f"P{piece_id}路径点超出机械工作区")
                break
        if not all(axis_aligned(start, end)
                   for start, end in zip(result.path_px, result.path_px[1:])):
            reasons.append(f"P{piece_id}路径包含非正交线段")

        if match.solver_mode == "generic_geometry":
            if match.transform_3x3 is None or not generic_solver.is_rigid_transform(
                    match.transform_3x3, tolerance=1e-5):
                reasons.append(f"P{piece_id}缺少有效刚体变换")
            else:
                mapped = generic_solver.transform_points(
                    np.asarray([result.piece.pick_mm(ppm)], np.float64),
                    match.transform_3x3,
                )[0]
                if np.linalg.norm(mapped - np.asarray(match.target_pick_mm)) > point_tolerance:
                    reasons.append(f"P{piece_id}磁吸点未使用同一刚体变换")
            if abs(match.fit_scale - 1.0) > 1e-6:
                reasons.append(f"P{piece_id}通用求解发生缩放")
            # 比赛完成优先：严格合格解按运动级质量标准校验；宽松兜底解
            # （超时后放宽填充/缺口产生的解）或强制执行开关开启时，
            # 跳过所有质量类阈值，只保留硬件安全（刚体变换/工作区/路径）。
            if config.get("strict_production", False) and match.relaxed_override:
                reasons.append(f"P{piece_id}超时宽松解禁止正式执行")
            if config.get("strict_production", False) or not (match.relaxed_override or bool(
                    config.get("force_execute_on_any_solution", False))):
                motion_fill_limit = float(config.get(
                    "generic_motion_max_fill_error_ratio",
                    config.get("generic_max_fill_error_ratio", 0.06),
                ))
                if match.fill_error_ratio > motion_fill_limit:
                    reasons.append(f"P{piece_id}矩形填充误差超限")
                motion_overlap_limit = float(config.get(
                    "generic_motion_max_overlap_ratio",
                    config.get("generic_max_overlap_ratio", 0.005),
                ))
                if match.overlap_ratio > motion_overlap_limit:
                    reasons.append(f"P{piece_id}目标碎片重叠率超限")
                if match.endpoint_error_mm > float(config.get("generic_max_endpoint_error_mm", 3.0)):
                    reasons.append(f"P{piece_id}拼缝端点误差超限")
                if match.boundary_error_mm > float(config.get("generic_boundary_tolerance_mm", 3.0)):
                    reasons.append(f"P{piece_id}矩形边界误差超限")
                motion_boundary_gap_limit = float(config.get(
                    "generic_motion_max_boundary_gap_ratio",
                    config.get("generic_max_boundary_gap_ratio", 0.08),
                ))
                if match.boundary_gap_ratio > motion_boundary_gap_limit:
                    reasons.append(f"P{piece_id}矩形边界缺口率超限")
            elif match.relaxed_override:
                print(f"[保底-宽松] P{piece_id}为宽松兜底解，跳过运动级质量校验（硬件安全仍检查）")
    if paths and paths[0].match.pattern_verification is not None:
        verification = paths[0].match.pattern_verification
        if verification.get("n_pairs", 0) > 0:
            minimum_similarity = float(verification.get("min_similarity", 1.0))
            ncc_limit = float(config.get("poker_ncc_min_similarity", 0.55))
            if minimum_similarity < ncc_limit:
                message = f"花纹相似度{minimum_similarity:.2f}<{ncc_limit:.2f}"
                if bool(config.get("enforce_pattern_gate", False)):
                    reasons.append(message + "，正式执行被锁定")
                else:
                    print(message + "，按形状拼接矩形执行（兼容模式）")
        elif bool(config.get("enforce_pattern_gate", False)):
            reasons.append("未得到相邻边花纹吻合度，正式执行被锁定")
    elif bool(config.get("enforce_pattern_gate", False)):
        reasons.append("未得到相邻边花纹吻合度，正式执行被锁定")
    return list(dict.fromkeys(reasons))


def result_dict(result: PathResult, config: dict[str, Any]) -> dict[str, Any]:
    ppm = float(config["pixels_per_mm"])
    center = result.piece.center_mm(ppm)
    pick = result.piece.pick_mm(ppm)
    path_mm = [px_to_mm(point, ppm) for point in result.path_px]
    match = result.match
    return {
        "id": result.piece.piece_id,
        "move_order": result.piece.piece_id,
        "template_id": match.template_id,
        "status": result.status,
        "match_status": match.status,
        "planner": result.planner,
        "center_mm": [round(center[0], 2), round(center[1], 2)],
        "source_contour_mm": (result.piece.contour.reshape(-1, 2).astype(float) / ppm).tolist(),
        "source_pick_mm": [round(pick[0], 2), round(pick[1], 2)],
        "target_pick_mm": [round(match.target_pick_mm[0], 2),
                           round(match.target_pick_mm[1], 2)],
        "target_mm": [round(match.target_pick_mm[0], 2),
                      round(match.target_pick_mm[1], 2)],
        "source_min_area_rect_angle_deg": round(result.piece.angle_deg, 2),
        "rotation_deg_signed": round(match.rotation_deg, 2),
        "rotation_direction": match.rotation_direction,
        "rotation_angle_deg": round(abs(match.rotation_deg), 2),
        "rotation_convention": "positive=CW, negative=CCW, image/A4 coordinates",
        "match_iou": round(match.match_iou, 4),
        "area_error_ratio": round(match.area_error_ratio, 4),
        "shape_distance": round(match.shape_distance, 4),
        "fit_scale": round(match.fit_scale, 4),
        "solver_mode": match.solver_mode,
        "solution_score": round(match.solution_score, 6),
        "fill_error_ratio": round(match.fill_error_ratio, 6),
        "overlap_ratio": round(match.overlap_ratio, 6),
        "endpoint_error_mm": round(match.endpoint_error_mm, 4),
        "boundary_error_mm": round(match.boundary_error_mm, 4),
        "boundary_gap_ratio": round(match.boundary_gap_ratio, 6),
        "transform_3x3": (None if match.transform_3x3 is None else
                          [[round(float(value), 8) for value in row]
                           for row in match.transform_3x3]),
        "area_mm2": round(result.piece.area_mm2, 2),
        "size_mm": [round(result.piece.width_mm, 2), round(result.piece.height_mm, 2)],
        "target_polygon_mm": [[round(float(x), 2), round(float(y), 2)]
                              for x, y in match.target_polygon_mm],
        "target_clearance_shift_mm": [
            round(match.target_clearance_shift_mm[0], 2),
            round(match.target_clearance_shift_mm[1], 2),
        ],
        "path_mm": [[round(x, 2), round(y, 2)] for x, y in path_mm],
        "path_constraint": "orthogonal_xy_one_axis_at_a_time",
        "axis_moves_mm": axis_motion_steps(path_mm),
        "motion_sequence": [
            "MOVE_TO_SOURCE_PICK", "MAGNET_ON", "LIFT_Z", "ROTATE",
            "FOLLOW_PATH", "LOWER_Z", "MAGNET_OFF"
        ],
    }


# 当前目标函数包含空载到下一吸取点和搬运到目标的曼哈顿距离；没有累计最后回原点的路程，也没有包含抬升与旋转耗时。
def optimize_motion_order(
        pieces: list[dict[str, Any]], target_offset_mm: tuple[float, float],
        start_mm: tuple[float, float] = (0.0, 0.0),
) -> tuple[list[dict[str, Any]], float]:
    """枚举四片执行顺序，最小化XY轴总正交行程。"""
    if not pieces:
        return [], 0.0

    offset_x, offset_y = target_offset_mm

    def point(item: dict[str, Any], key: str) -> tuple[float, float]:
        value = item[key]
        return float(value[0]), float(value[1])

    def distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def route_distance(order: tuple[dict[str, Any], ...]) -> float:
        current = start_mm
        total = 0.0
        for item in order:
            source = point(item, "source_pick_mm")
            target_raw = point(item, "target_pick_mm")
            target = target_raw[0] + offset_x, target_raw[1] + offset_y
            total += distance(current, source) + distance(source, target)
            current = target
        return total

    best = min(
        itertools.permutations(pieces),
        key=lambda order: (route_distance(order), tuple(int(item["id"]) for item in order)),
    )
    ordered = [dict(item) for item in best]
    for move_order, item in enumerate(ordered, 1):
        item["move_order"] = move_order
    return ordered, route_distance(best)


def dual_execution_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Describe the parameters actually consumed by the dual controller."""
    z, motion, rotation = (config.get(key, {}) for key in ("z", "motion", "rotation"))
    safe = float(z["safe"])
    z_feed = float(z.get("feed_mm_min", 60.0))
    xy = config.get("xy_transform", {})
    return {
        "execution_compensation": {
            "xy_command_matrix": xy.get("matrix", [[1.0, 0.0], [0.0, 1.0]]),
            "xy_command_bias_mm": xy.get("bias_mm", [0.0, 0.0]),
            "tool_offset_mm": config.get("tool_offset_mm", [0.0, 0.0]),
            "rotation_pulses_per_revolution": int(rotation.get("pulses_per_revolution", 3200)),
        },
        "execution_motion": {
            "xy_feed_mm_min": float(motion.get("xy_feed_mm_min", 300.0)),
            "z_feed_mm_min": z_feed,
            "z_safe_mm": safe,
            "z_travel_mm": float(z.get("travel", safe)),
            "z_pickup_mm": float(z["pickup"]),
            "z_place_mm": float(z["place"]),
            "z_contact_margin_mm": float(z.get("contact_margin_mm", 0.0)),
            "z_contact_feed_mm_min": float(z.get("contact_feed_mm_min", z_feed)),
        },
        "execution_timing": {
            "magnet_pickup_settle_seconds": float(motion.get("pickup_settle_s", 0.8)),
            "magnet_release_settle_seconds": float(motion.get("release_settle_s", 0.3)),
            "lift_settle_seconds": float(z.get("settle_s", 0.0)),
            "xy_settle_seconds": float(motion.get("xy_settle_s", 0.0)),
            "rotation_settle_seconds": float(rotation.get("settle_s", 0.0)),
        },
        "execution_policy": {
            "return_origin_after_plan": True,
            "rotate_at_target": False,
            "require_origin_before_plan": True,
            "origin_mode": str(motion.get("origin_mode", "homing")),
        },
    }


def save_outputs(original: np.ndarray, paper: np.ndarray, mask: np.ndarray,
                 preview: np.ndarray, paths: list[PathResult],
                 config: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_image(OUTPUT_DIR / "original.jpg", original)
    write_image(OUTPUT_DIR / "corrected.jpg", paper)
    write_image(OUTPUT_DIR / "mask.png", mask)
    write_image(OUTPUT_DIR / "path_preview.jpg", preview)
    actual_top_left, actual_size = target_rectangle_from_paths(paths, config)
    top_left = [float(value) for value in actual_top_left]
    size = [float(value) for value in actual_size]
    all_matches_ok = bool(paths) and all(item.match.status == "ok" for item in paths)
    all_paths_ok = bool(paths) and all(item.status == "ok" for item in paths)
    failure_reasons = motion_safety_reasons(paths, config)
    ready_for_motion = not failure_reasons
    if paths:
        layout_points = np.vstack([
            np.asarray(item.match.target_polygon_mm, np.float64) for item in paths
        ])
        layout_minimum = np.min(layout_points, axis=0)
        layout_maximum = np.max(layout_points, axis=0)
    else:
        layout_minimum = np.asarray(top_left, np.float64)
        layout_maximum = layout_minimum + np.asarray(size, np.float64)
    clearance_key = ("fixed_target_piece_clearance_mm"
                     if str(config.get("puzzle_mode", "fixed_figure_2")) == "fixed_figure_2"
                     else "generic_target_piece_clearance_mm")
    target_clearance = float(config.get(clearance_key, 0.0))
    target_offset = config.get("motion_target_offset_mm", [0.0, 0.0])
    pieces_payload = [result_dict(item, config) for item in paths]
    if bool(config.get("motion_optimize_piece_order", True)):
        pieces_payload, route_distance_mm = optimize_motion_order(
            pieces_payload, (float(target_offset[0]), float(target_offset[1]))
        )
        order_text = "->".join(f"P{item['id']}" for item in pieces_payload)
        print(
            f"[路径优化] 执行顺序={order_text}，预计XY正交总行程="
            f"{route_distance_mm:.1f}mm"
        )
    else:
        route_distance_mm = 0.0

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "puzzle_mode": str(config.get("puzzle_mode", "fixed_figure_2")),
        "segmentation_mode": str(config.get("segmentation_mode", "pink_hsv")),
        "coordinate_system": {
            "name": "a4_workspace_mm",
            "origin": "A4_top_left",
            "x_axis": "right",
            "y_axis": "down",
        },
        "mechanical_workspace_mm": {
            "size": [float(config["paper_width_mm"]),
                     float(config["paper_height_mm"])],
            "paper_region": {"top_left": [0.0, 0.0],
                             "size": [float(config["paper_width_mm"]),
                                      float(config["paper_height_mm"])]},
            "path_policy": "shortest_xy_no_avoidance",
        },
        "rotation_convention": {"positive": "CW", "negative": "CCW", "unit": "degree"},
        "execution_compensation": {
            "target_offset_mm": [
                float(config.get("motion_target_offset_mm", [0.0, 0.0])[0]),
                float(config.get("motion_target_offset_mm", [0.0, 0.0])[1]),
            ],
            "xy_command_matrix": [
                [float(value) for value in row]
                for row in config.get("motion_xy_command_matrix", [[1.0, 0.0], [0.0, 1.0]])
            ],
            "xy_command_bias_mm": [
                float(value)
                for value in config.get("motion_xy_command_bias_mm", [0.0, 0.0])
            ],
            "rotation_magnitude_reduction_deg": float(
                config.get("motion_rotation_magnitude_reduction_deg", 0.0)
            ),
            "rotation_reduction_min_angle_deg": float(
                config.get("motion_rotation_reduction_min_angle_deg", 0.0)
            ),
            "rotation_chunk_deg": float(
                config.get("motion_rotation_chunk_deg", 180.0)
            ),
            "rotation_cycles_per_revolution": float(
                config.get("motion_rotation_cycles_per_revolution", 0.0)
            ),
        },
        "execution_timing": {
            "magnet_pickup_settle_seconds": float(
                config.get("magnet_pickup_settle_seconds", 0.0)
            ),
            "magnet_release_settle_seconds": float(
                config.get("magnet_release_settle_seconds", 0.0)
            ),
            "lift_settle_seconds": float(
                config.get("motion_lift_settle_seconds", 0.0)
            ),
            "xy_settle_seconds": float(
                config.get("motion_xy_settle_seconds", 0.0)
            ),
            "rotation_settle_seconds": float(
                config.get("motion_rotation_settle_seconds", 0.0)
            ),
        },
        "execution_policy": {
            "return_origin_after_plan": bool(
                config.get("motion_return_origin_after_plan", False)
            ),
            "rotate_at_target": bool(
                config.get("motion_rotate_at_target", False)
            ),
            "require_origin_before_plan": bool(
                config.get("motion_require_origin_before_plan", False)
            ),
        },
        "numbering_rule": str(config["numbering_mode"]),
        "execution_order_rule": "minimum_total_orthogonal_xy_distance",
        "estimated_xy_travel_mm": round(route_distance_mm, 2),
        "magnetic_safe_distance_mm": float(config["magnetic_safe_distance_mm"]),
        "target_region": {
            "name": f"A4_{str(config.get('target_region', 'bottom')).lower()}_half",
            "top_y_mm": float(config.get("separator_line_y_mm",
                                         config.get("target_region_top_mm", 148.5))),
            "configured": str(config.get("target_region", "bottom")).lower(),
        },
        "target_rectangle_mm": {"top_left": top_left, "size": size,
                                "bottom_right": [top_left[0] + size[0], top_left[1] + size[1]]},
        "target_layout": {
            "piece_clearance_mm": target_clearance,
            "single_piece_outward_offset_mm": target_clearance / 2.0,
            "corresponding_vertex_limit_mm": float(
                config.get("target_corresponding_vertex_limit_mm", 20.0)
            ),
            "actual_bounds_mm": {
                "top_left": [float(layout_minimum[0]), float(layout_minimum[1])],
                "size": [float(layout_maximum[0] - layout_minimum[0]),
                         float(layout_maximum[1] - layout_minimum[1])],
                "bottom_right": [float(layout_maximum[0]), float(layout_maximum[1])],
            },
        },
        "piece_count": len(paths),
        "selected_polygon_min_edge_mm": getattr(paths[0].match, "polygon_min_edge_mm", None) if paths else None,
        "path_constraint": "orthogonal_xy_one_axis_at_a_time",
        "all_matches_ok": all_matches_ok,
        "all_paths_ok": all_paths_ok,
        "ready_for_motion": ready_for_motion,
        "failure_reasons": failure_reasons,
        "pieces": pieces_payload,
        "poker_pattern_verification": (
            paths[0].match.pattern_verification if paths
            and paths[0].match.pattern_verification is not None else None
        ),
    }
    if config.get("execution_backend") == "dual_grbl_stm32":
        payload.update(dual_execution_summary(config))
    if config.get("approval_workflow", False):
        from approved_plan import seal_plan
        payload = seal_plan(payload, original, paper, config, OUTPUT_DIR)
    with (OUTPUT_DIR / "plan.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"[OK] 已输出：{OUTPUT_DIR}")
    print(f"[OK] Path preview: {OUTPUT_DIR / 'path_preview.jpg'}")
    if ready_for_motion:
        print("[OK] ready_for_motion=true")
    else:
        print("[WARN] ready_for_motion=false; safety interlock blocks motion")
        for reason in failure_reasons:
            print(f"[WARN] {reason}")
    for result in paths:
        source = result.piece.pick_mm(float(config["pixels_per_mm"]))
        match = result.match
        print(f"  P{result.piece.piece_id}->模板{match.template_id}: "
              f"源=({source[0]:.1f},{source[1]:.1f})mm "
              f"目标吸取点=({match.target_pick_mm[0]:.1f},{match.target_pick_mm[1]:.1f})mm "
              f"旋转={match.rotation_deg:+.1f}deg {match.rotation_direction} "
              f"路径={result.planner} 状态={result.status}")

def pink_mask_in_paper(frame: np.ndarray, matrix: np.ndarray,
                       config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """在原图先识别粉色，再将二值图透视变换，避免远距离放大后颜色被插值冲淡。"""
    raw_mask = make_pink_mask(frame, config, int(config.get("raw_morphology_kernel_px", 3)))
    paper_mask = cv2.warpPerspective(
        raw_mask, matrix, paper_size_px(config), flags=cv2.INTER_NEAREST
    )
    paper_mask = _apply_region_mask(paper_mask, config, "source")
    return raw_mask, paper_mask


def mask_in_paper(frame: np.ndarray, matrix: np.ndarray,
                  config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """统一生成原图诊断掩膜和校正纸面掩膜。"""
    mode = str(config.get("segmentation_mode", "pink_hsv"))
    if mode == "pink_hsv":
        return pink_mask_in_paper(frame, matrix, config)
    if mode not in {"background_inverse", "white_piece", "poker_v"}:
        raise ValueError(f"未知segmentation_mode：{mode}")
    segmentation_frame = frame
    if mode == "background_inverse":
        # Suppress sensor/compression speckle before perspective enlargement.
        # Keep the original photograph and all geometry acceptance limits intact.
        sigma = float(config.get("raw_background_denoise_sigma_px", 0.0))
        if not math.isfinite(sigma) or not 0.0 <= sigma <= 1.5:
            raise ValueError("原图去噪sigma必须在0～1.5像素内")
        if sigma > 0.0:
            segmentation_frame = cv2.GaussianBlur(frame, (0, 0), sigma)
    paper = warp_paper(segmentation_frame, matrix, config)
    paper_mask = make_piece_mask(paper, config)
    inverse = np.linalg.inv(np.asarray(matrix, np.float64))
    raw_mask = cv2.warpPerspective(
        paper_mask, inverse, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST
    )
    return raw_mask, paper_mask


def save_detection_diagnostics(original: np.ndarray, paper: np.ndarray,
                               mask: np.ndarray) -> None:
    """统一保存识别失败所需的三张诊断图。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_image(OUTPUT_DIR / "original.jpg", original)
    write_image(OUTPUT_DIR / "corrected.jpg", paper)
    write_image(OUTPUT_DIR / "mask.png", mask)
    print("[诊断] 已保存original.jpg、corrected.jpg和mask.png，请据此检查标定与分割")


# 这是理解数据流的核心函数；处理过程中会保存文件，show只控制预览窗口，不能理解成“不产生输出”。
def save_failed_detection(original, paper, mask, config, reason):
    """Keep the actual rejected frame visible; never substitute an old success."""
    save_detection_diagnostics(original, paper, mask)
    preview = paper.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ppm = float(config["pixels_per_mm"])
    for contour in contours:
        if cv2.contourArea(contour) < float(config["min_piece_area_mm2"]) * ppm * ppm:
            continue
        cv2.drawContours(preview, [contour], -1, (0, 80, 255), max(2, round(ppm)))
    write_image(OUTPUT_DIR / "detection_preview.jpg", preview)
    (OUTPUT_DIR / "detection_failure.json").write_text(json.dumps({
        "time": time.time(), "reason": str(reason), "image": "detection_preview.jpg",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def process_paper(original: np.ndarray, paper: np.ndarray,
                  config: dict[str, Any], show: bool,
                  mask_override: np.ndarray | None = None,
                  *, save: bool = True) -> tuple[list[Piece], list[PathResult]]:
    pieces, mask = detect_pieces(paper, config, diagnostics=True, mask_override=mask_override)
    pieces = number_by_move_order(pieces, config)

    quality_reasons = calibration_quality_reasons(paper, config)
    border_contacts = piece_border_contacts(pieces, paper.shape[:2], config)
    for reason in quality_reasons:
        print(f"[WARN] 标定诊断：{reason}；继续尝试识别与求解")
    if border_contacts:
        print(f"[WARN] 边界诊断：{'；'.join(border_contacts)}；继续尝试识别与求解")

    mode = str(config.get("puzzle_mode", "fixed_figure_2"))
    expected = int(config.get("expected_piece_count", 4))
    if config.get("strict_production", False) and len(pieces) != expected:
        reason = (f"识别到{len(pieces)}/{expected}片；请将碎片分开放在下半区，"
                  "检查粘连、遮挡、分界线和纸面边缘")
        if save:
            save_failed_detection(original, paper, mask, config, reason)
        raise RuntimeError(reason)
    count_ok = (len(pieces) == expected if mode == "fixed_figure_2" else
                1 <= len(pieces) <= int(config.get("max_pieces", 4)))
    if not count_ok:
        if save:
            save_detection_diagnostics(original, paper, mask)
        if mode == "fixed_figure_2":
            raise RuntimeError(f"固定图2必须完整识别{expected}块，当前识别到{len(pieces)}块")
        if mode == "generic_geometry":
            raise RuntimeError(f"通用拼图必须识别1～4块，当前识别到{len(pieces)}块")
        raise RuntimeError(f"未知puzzle_mode：{mode}")
    # Rounded corners can produce short, unstable extra edges. A stricter
    # outline is a candidate, not an exemption from any geometry/pose gate.
    base_edge = float(config.get("generic_min_edge_length_mm", 0.0))
    edges = [base_edge]
    if config.get("strict_production") and mode == "generic_geometry":
        requested = config.get("planning_polygon_min_edge_candidates_mm", [])
        edges = list(dict.fromkeys([float(e) for e in requested[:1]] + edges))
        if any(not math.isfinite(e) or e < base_edge for e in edges):
            raise ValueError("备选轮廓模型不得降低最短边限制")
    failures = []
    for edge in edges:
        work = dict(config)
        work["generic_min_edge_length_mm"] = edge
        try:
            matches = solve_puzzle(pieces, work, paper)
            paths = plan_all_paths(pieces, work, matches)
            if len(edges) > 1:
                gate = dict(config)
                if config.get("approval_workflow"):
                    gate["enforce_pattern_gate"] = False
                reasons = motion_safety_reasons(paths, gate)
                if reasons:
                    raise ValueError("；".join(reasons))
            for match in matches:
                match.polygon_min_edge_mm = edge
            break
        except ValueError as exc:
            failures.append(f"轮廓模型{edge:g}mm：{exc}")
    else:
        if save:
            save_detection_diagnostics(original, paper, mask)
        raise RuntimeError("；".join(failures))
    if save or show:
        preview = draw_preview(paper, pieces, paths, config)
    if save:
        save_outputs(original, paper, mask, preview, paths, config)
    if show:
        cv2.namedWindow("Path Preview", cv2.WINDOW_NORMAL)
        cv2.imshow("Path Preview", preview)
    return pieces, paths

# =============================================================================
# 【分区】运行入口（demo / 单图 / 相机 / 自动发计划）
# 功能：演示、单图、相机循环、可选串口发送。
# 可修改：show 窗口、--send。
# 看情况改：比赛用 --auto。
# 不要改：process_paper 主链路。
# =============================================================================
# ------------------------- 运行入口 -------------------------

def make_demo_paper(config: dict[str, Any]) -> np.ndarray:
    width, height = paper_size_px(config)
    ppm = float(config["pixels_per_mm"])
    paper = np.full((height, width, 3), (245, 245, 245), np.uint8)
    templates = figure2_template_polygons(config, placed=False)
    colors = [(205, 140, 250), (195, 130, 245), (215, 150, 250), (200, 135, 245)]
    for (template_id, polygon), color in zip(templates.items(), colors):
        center_mm, angle_deg = DEMO_PLACEMENTS[template_id]
        local = polygon - polygon_centroid(polygon)
        transformed = rotate_points(local, angle_deg) + np.asarray(center_mm, np.float64)
        points = np.rint(transformed * ppm).astype(np.int32)
        cv2.fillPoly(paper, [points], color)
        cv2.polylines(paper, [points], True, (50, 50, 50), 2)
    return paper

def run_demo(config: dict[str, Any]) -> None:
    # 合成样本来自固定图2，演示必须使用与样本匹配的固定求解和粉色分割。
    # 使用副本，避免覆盖用户为摄像头实测设置的通用模式。
    demo_config = dict(config)
    demo_config["puzzle_mode"] = "fixed_figure_2"
    demo_config["segmentation_mode"] = "pink_hsv"
    demo_config["expected_piece_count"] = 4
    print("[演示] 使用固定图2 + pink_hsv合成样本；不会修改config.json中的实测模式。")
    paper = make_demo_paper(demo_config)
    process_paper(paper, paper, demo_config, False)
    print("[OK] 合成演示完成，请查看output/path_preview.jpg和output/plan.json")


def run_image(path: Path, config: dict[str, Any]) -> None:
    image = read_image(path)
    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")
    width, height = paper_size_px(config)
    matrix = load_calibration(config)
    mask_override = None
    if image.shape[:2] == (height, width):
        paper = image.copy()
    else:
        matrix, _ = resolve_calibration_matrix(image, config, matrix)
        paper = warp_paper(image, matrix, config)
        _, mask_override = mask_in_paper(image, matrix, config)
    process_paper(image, paper, config, False, mask_override)


def auto_send_plan_if_ready(config: dict[str, Any]) -> None:
    """识别求解成功且安全联锁通过时，通过串口发送运动计划驱动STM32。"""
    plan_path = OUTPUT_DIR / "plan.json"
    if not plan_path.is_file():
        print("[WARN] 未找到output/plan.json，未发送运动指令")
        return
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not plan.get("ready_for_motion"):
        print("[WARN] ready_for_motion=false，安全联锁拒绝发送运动指令")
        return
    port = str(config.get("serial_port", ""))
    if not port:
        print("[WARN] config未配置serial_port，未发送运动指令")
        return
    print("[自动] 开始通过串口发送运动计划...")
    serial_transport.send_plan_file(
        plan_path,
        dry_run=False,
        port=port,
        baudrate=int(config.get("serial_baudrate", 115200)),
        ack_timeout_seconds=float(config.get("serial_ack_timeout_seconds", 2.0)),
        command_delay_seconds=float(config.get("serial_command_delay_seconds", 0.0)),
    )
    print("[自动] 运动计划已发送完成")


def run_camera(config: dict[str, Any], camera_id: int | None) -> None:
    capture = open_camera(config, camera_id)
    matrix = load_calibration(config)
    window = "USB Puzzle Vision"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if hasattr(cv2, "WND_PROP_FULLSCREEN"):
        try:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except cv2.error:
            pass

    buttons = {
        "calibrate": (10, 10, 170, 58),
        "detect": (180, 10, 320, 58),
        "poker": (330, 10, 470, 58),
        "capture": (480, 10, 630, 58),
        "quit": (640, 10, 740, 58),
    }
    pending_action: dict[str, str | None] = {"value": None}

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for action, (x1, y1, x2, y2) in buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                pending_action["value"] = action
                return

    cv2.setMouseCallback(window, on_mouse)
    print("[操作] 直接点击DETECT即可自动定位底板并识别；CALIBRATE仅作失败兜底。")
    print("[操作] POKER用于扑克牌碎片（白底花纹）识别+拼合+花纹验证，与DETECT共用标定。")
    print("[操作] 键盘D识别，P扑克牌，K手动标定，C保存原图，Q退出。")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("摄像头读取失败")

            display = frame.copy()
            cv2.rectangle(display, (0, 0), (display.shape[1], 100), (25, 25, 25), -1)
            button_colors = {
                "calibrate": (0, 170, 255),
                "detect": (0, 190, 0),
                "poker": (200, 90, 200),
                "capture": (200, 130, 0),
                "quit": (0, 0, 200),
            }
            labels = {
                "calibrate": "CALIBRATE",
                "detect": "DETECT",
                "poker": "POKER",
                "capture": "CAPTURE",
                "quit": "QUIT",
            }
            for action, (x1, y1, x2, y2) in buttons.items():
                cv2.rectangle(display, (x1, y1), (x2, y2), button_colors[action], -1)
                cv2.putText(display, labels[action], (x1 + 10, y1 + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255),
                            2, cv2.LINE_AA)
            auto_enabled = bool(config.get("auto_board_calibration", True))
            status = ("AUTO BOARD - click DETECT" if auto_enabled else
                      ("CALIBRATED - click DETECT" if matrix is not None else
                       "NOT CALIBRATED - click CALIBRATE"))
            status_color = (0, 255, 0) if auto_enabled or matrix is not None else (0, 180, 255)
            cv2.putText(display, status, (12, 87), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, status_color, 2, cv2.LINE_AA)
            cv2.imshow(window, display)

            key = cv2.waitKeyEx(10)
            action = pending_action["value"]
            pending_action["value"] = None
            low_key = key & 0xFF if key >= 0 else -1
            if low_key in (ord("k"), ord("K")):
                action = "calibrate"
            elif low_key in (ord("d"), ord("D")):
                action = "detect"
            elif low_key in (ord("p"), ord("P")):
                action = "poker"
            elif low_key in (ord("c"), ord("C")):
                action = "capture"
            elif low_key in (ord("q"), ord("Q"), 27):
                action = "quit"

            if action == "quit":
                break
            if action == "capture":
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                write_image(OUTPUT_DIR / "camera_capture.jpg", frame)
                print("[OK] 已保存output/camera_capture.jpg")
            elif action == "calibrate":
                new_matrix = collect_four_points(frame, config)
                if new_matrix is not None:
                    matrix = new_matrix
                    # 返回主窗口后重新注册回调，避免部分OpenCV后端丢失事件。
                    cv2.setMouseCallback(window, on_mouse)
            elif action == "detect":
                try:
                    active_matrix, source = resolve_calibration_matrix(frame, config, matrix)
                    if source == "auto_board":
                        matrix = active_matrix
                    paper = warp_paper(frame, active_matrix, config)
                    raw_mask, paper_mask = mask_in_paper(frame, active_matrix, config)
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    raw_name = ("raw_pink_mask.png" if
                                str(config.get("segmentation_mode", "pink_hsv")) == "pink_hsv"
                                else "raw_piece_mask.png")
                    write_image(OUTPUT_DIR / raw_name, raw_mask)
                    detect_config = dict(config)
                    detect_config["poker_pattern_verification"] = False
                    process_paper(frame, paper, detect_config, True, paper_mask)
                    # 若求解成功且安全联锁通过，则通过串口发送运动计划驱动STM32
                    try:
                        auto_send_plan_if_ready(detect_config)
                    except Exception as exc:
                        print(f"[WARN] 串口发送失败(不影响识别): {exc}")
                except RuntimeError as exc:
                    print(f"[ERR] {exc}")
            elif action == "poker":
                poker_config = dict(config)
                poker_config["puzzle_mode"] = "generic_geometry"
                poker_config["segmentation_mode"] = "poker_v"
                poker_config["generic_target_long_min_mm"] = float(
                    config.get("poker_target_long_min_mm", 80.0))
                poker_config["generic_target_long_max_mm"] = float(
                    config.get("poker_target_long_max_mm", 95.0))
                poker_config["generic_target_short_min_mm"] = float(
                    config.get("poker_target_short_min_mm", 50.0))
                poker_config["generic_target_short_max_mm"] = float(
                    config.get("poker_target_short_max_mm", 62.0))
                poker_config["generic_max_fill_error_ratio"] = float(
                    config.get("poker_max_fill_error_ratio", 0.35))
                poker_config["generic_max_boundary_gap_ratio"] = float(
                    config.get("poker_max_boundary_gap_ratio", 0.30))
                poker_config["generic_edge_abs_tolerance_mm"] = float(
                    config.get("poker_edge_abs_tolerance_mm", 3.0))
                poker_config["generic_edge_rel_tolerance"] = float(
                    config.get("poker_edge_rel_tolerance", 0.08))
                poker_config["generic_min_edge_length_mm"] = float(
                    config.get("poker_min_edge_length_mm", 4.0))
                poker_config["generic_max_edge_candidates"] = int(
                    config.get("poker_max_edge_candidates", 80))
                poker_config["generic_search_timeout_seconds"] = float(
                    config.get("poker_search_timeout_seconds", 10.0))
                poker_config["generic_max_search_states"] = int(
                    config.get("poker_max_search_states", 60000))
                poker_config["generic_require_each_piece_outer_edge"] = bool(
                    config.get("poker_require_outer_edge", False))
                poker_config["poker_pattern_verification"] = bool(
                    config.get("poker_pattern_verification", True))
                try:
                    # 与DETECT共用手动标定（calibration.json）；仅本模式固定使用标定矩阵。
                    active_matrix, source = resolve_calibration_matrix(
                        frame, poker_config, matrix
                    )
                    if source == "auto_board":
                        matrix = active_matrix
                    paper = warp_paper(frame, active_matrix, poker_config)
                    _, paper_mask = mask_in_paper(frame, active_matrix, poker_config)
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    write_image(OUTPUT_DIR / "raw_poker_mask.png", paper_mask)
                    process_paper(frame, paper, poker_config, True, paper_mask)
                    # 若求解成功且花纹验证与安全联锁均通过，则通过串口发送运动计划
                    auto_send_plan_if_ready(poker_config)
                except RuntimeError as exc:
                    print(f"[ERR] {exc}")

            if not window_is_visible(window):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("[OK] 摄像头已释放")


# =============================================================================
# 【分区】一键自动流程与终局视觉复核
# 功能：等 START → 稳定帧 → 求解（含花纹）→ 可选执行 → 再拍复核。
# 可修改：START 热区、丢帧、终检公差。
# 看情况改：花纹失败先查光照和偏移采样，不要关 pattern_gate。
# 不要改：计时零点=START 按下瞬间；复核失败不得报完成。
# =============================================================================
# ------------------------- 一键自动流程 -------------------------

def wait_for_start_and_capture(
        capture: cv2.VideoCapture,
        config: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    """显示实时预览等待START触发，随后按题目时序丢弃前几帧再取稳定帧。

    返回(稳定帧, 触发时刻)；触发时刻取自START按下的瞬间（早于曝光稳定等待），
    与赛题"一键启动同时移除遮挡、开始计时"的时间起点一致，供调用方统计总耗时。
    返回None表示用户主动退出（Q/Esc/关闭窗口），此时调用方应结束整个自动流程。
    """
    window = "Auto Puzzle Vision"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    button = (10, 10, 220, 58)
    triggered = {"value": False}

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        x1, y1, x2, y2 = button
        if event == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and y1 <= y <= y2:
            triggered["value"] = True

    cv2.setMouseCallback(window, on_mouse)
    print("[自动] 遮挡摄像头、随机摆放碎片后，按空格/回车或点击START；Q退出。")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("摄像头读取失败")
            display = frame.copy()
            x1, y1, x2, y2 = button
            cv2.rectangle(display, (0, 0), (display.shape[1], 70), (25, 25, 25), -1)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 190, 0), -1)
            cv2.putText(display, "START", (x1 + 30, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "Cover camera, place pieces, then START (Q to quit)",
                       (x2 + 15, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, display)

            key = cv2.waitKeyEx(20)
            low_key = key & 0xFF if key >= 0 else -1
            if low_key in (32, 13) or triggered["value"]:
                break
            if low_key in (ord("q"), ord("Q"), 27) or not window_is_visible(window):
                safe_destroy_window(window)
                return None

        trigger_time = time.monotonic()
        print("[自动] 已触发，等待遮挡移除与曝光稳定...")
        warmup_seconds = float(config.get("auto_warmup_seconds", 0.8))
        deadline = time.monotonic() + warmup_seconds
        stable_frame: np.ndarray | None = None
        while time.monotonic() < deadline:
            success, stable_frame = capture.read()
            if not success:
                raise RuntimeError("摄像头读取失败")
            cv2.imshow(window, stable_frame)
            cv2.waitKey(1)
        # 再多取几帧使曝光/白平衡收敛，使用最后一帧作为计算帧。
        for _ in range(int(config.get("auto_extra_settle_frames", 5))):
            success, stable_frame = capture.read()
            if not success:
                raise RuntimeError("摄像头读取失败")
        safe_destroy_window(window)
        return stable_frame, trigger_time
    except Exception:
        safe_destroy_window(window)
        raise


def polygon_iou_mm(first_mm: np.ndarray, second_mm: np.ndarray,
                   config: dict[str, Any]) -> float:
    """在A4坐标中计算两个已放置多边形的IoU。"""
    resolution = max(1.0, float(config.get("final_verify_pixels_per_mm", 3.0)))
    width = int(math.ceil(float(config["paper_width_mm"]) * resolution)) + 3
    height = int(math.ceil(float(config["paper_height_mm"]) * resolution)) + 3
    first_mask = np.zeros((height, width), np.uint8)
    second_mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(first_mask, [np.rint(np.asarray(first_mm) * resolution).astype(np.int32)], 1)
    cv2.fillPoly(second_mask, [np.rint(np.asarray(second_mm) * resolution).astype(np.int32)], 1)
    intersection = int(np.count_nonzero((first_mask > 0) & (second_mask > 0)))
    union = int(np.count_nonzero((first_mask > 0) | (second_mask > 0)))
    return intersection / union if union else 0.0


def final_verification_targets(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """从运动计划提取最终复核所需的目标轮廓。"""
    targets: list[dict[str, Any]] = []
    for piece in plan.get("pieces", []):
        polygon = np.asarray(piece.get("target_polygon_mm"), np.float64)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            raise RuntimeError(f"P{piece.get('id', '?')}缺少有效目标轮廓，无法最终复核")
        targets.append({
            "piece_id": int(piece["id"]),
            "template_id": str(piece.get("template_id", "")),
            "polygon_mm": polygon,
            "center_mm": polygon_centroid(polygon),
        })
    if not targets:
        raise RuntimeError("运动计划不包含最终目标轮廓")
    return targets


# shape_iou比较对齐后的形状是否相似，placed_iou比较实际摆放位置是否重合，两者含义不同；最终还检查中心误差、角度误差和数量。
def match_final_layout(pieces: list[Piece], plan: dict[str, Any],
                       config: dict[str, Any]) -> dict[str, Any]:
    """将最终实拍碎片全局匹配到计划目标，并区分位置与角度误差。"""
    targets = final_verification_targets(plan)
    ppm = float(config["pixels_per_mm"])
    detected = pieces[:len(targets)]
    for index, piece in enumerate(detected, 1):
        piece.piece_id = index

    pair_metrics: dict[tuple[int, int], dict[str, Any]] = {}
    for actual_index, piece in enumerate(detected):
        actual_polygon = piece.contour.reshape(-1, 2).astype(np.float64) / ppm
        actual_center = np.asarray(piece.center_mm(ppm), np.float64)
        for target_index, target in enumerate(targets):
            correction, shape_iou, fit_scale = best_rotation_fit(
                piece, target["polygon_mm"], config
            )
            position_error = float(np.linalg.norm(actual_center - target["center_mm"]))
            placed_iou = polygon_iou_mm(actual_polygon, target["polygon_mm"], config)
            scale_penalty = abs(math.log(max(fit_scale, 1e-6)))
            cost = position_error + 35.0 * (1.0 - shape_iou) + 10.0 * scale_penalty
            pair_metrics[(actual_index, target_index)] = {
                "cost": cost,
                "position_error_mm": position_error,
                "rotation_correction_deg": correction,
                "shape_iou": shape_iou,
                "placed_iou": placed_iou,
                "fit_scale": fit_scale,
                "actual_center_mm": actual_center,
            }

    if detected:
        best_assignment = min(
            itertools.permutations(range(len(targets)), len(detected)),
            key=lambda assignment: sum(
                pair_metrics[(actual_index, target_index)]["cost"]
                for actual_index, target_index in enumerate(assignment)
            ),
        )
    else:
        best_assignment = ()

    position_limit = float(config.get("final_verify_max_position_error_mm", 8.0))
    angle_limit = float(config.get("final_verify_max_rotation_error_deg", 12.0))
    shape_iou_minimum = float(config.get("final_verify_min_shape_iou", 0.65))
    placed_iou_minimum = float(config.get("final_verify_min_placed_iou", 0.45))
    matches: list[dict[str, Any]] = []
    matched_targets: set[int] = set()
    for actual_index, target_index in enumerate(best_assignment):
        metric = pair_metrics[(actual_index, target_index)]
        target = targets[target_index]
        matched_targets.add(target_index)
        passed = (
            metric["position_error_mm"] <= position_limit
            and abs(metric["rotation_correction_deg"]) <= angle_limit
            and metric["shape_iou"] >= shape_iou_minimum
            and metric["placed_iou"] >= placed_iou_minimum
        )
        matches.append({
            "actual_index": actual_index + 1,
            "target_piece_id": target["piece_id"],
            "template_id": target["template_id"],
            "actual_center_mm": [round(float(value), 2) for value in metric["actual_center_mm"]],
            "target_center_mm": [round(float(value), 2) for value in target["center_mm"]],
            "position_error_mm": round(float(metric["position_error_mm"]), 2),
            "rotation_correction_deg": round(float(metric["rotation_correction_deg"]), 2),
            "shape_iou": round(float(metric["shape_iou"]), 4),
            "placed_iou": round(float(metric["placed_iou"]), 4),
            "fit_scale": round(float(metric["fit_scale"]), 4),
            "measurement_tentative": (
                len(pieces) != len(targets) or metric["shape_iou"] < shape_iou_minimum
            ),
            "passed": passed,
        })

    missing_target_ids = [
        targets[index]["piece_id"] for index in range(len(targets))
        if index not in matched_targets
    ]
    expected_count = len(targets)
    overall_pass = (
        len(pieces) == expected_count
        and not missing_target_ids
        and all(item["passed"] for item in matches)
    )
    return {
        "status": "pass" if overall_pass else "fail",
        "detected_piece_count": len(pieces),
        "expected_piece_count": expected_count,
        "missing_target_piece_ids": missing_target_ids,
        # Keep the legacy field above for consumers; an unmatched target is not
        # proof of physical absence when contours touch, merge or are occluded.
        "unresolved_target_piece_ids": missing_target_ids + [
            item["target_piece_id"] for item in matches
            if item["shape_iou"] < shape_iou_minimum
        ],
        "unexpected_piece_count": max(0, len(pieces) - expected_count),
        "thresholds": {
            "max_position_error_mm": position_limit,
            "max_rotation_error_deg": angle_limit,
            "min_shape_iou": shape_iou_minimum,
            "min_placed_iou": placed_iou_minimum,
        },
        "matches": matches,
    }


def draw_final_verification(paper: np.ndarray, pieces: list[Piece],
                            plan: dict[str, Any], report: dict[str, Any],
                            config: dict[str, Any]) -> np.ndarray:
    """绘制计划目标、实拍轮廓和误差连线。"""
    display = paper.copy()
    ppm = float(config["pixels_per_mm"])
    targets = {item["piece_id"]: item for item in final_verification_targets(plan)}
    for target in targets.values():
        points = np.rint(target["polygon_mm"] * ppm).astype(np.int32)
        cv2.polylines(display, [points], True, (0, 220, 0), 2, cv2.LINE_AA)

    matched_actual: set[int] = set()
    labeled_targets: set[int] = set()
    unresolved = set(report.get("unresolved_target_piece_ids", report["missing_target_piece_ids"]))
    for item in report["matches"]:
        actual_index = int(item["actual_index"]) - 1
        matched_actual.add(actual_index)
        piece = pieces[actual_index]
        tentative = item.get("measurement_tentative", False)
        color = (0, 165, 255) if tentative else ((0, 200, 0) if item["passed"] else (0, 0, 255))
        contour = piece.contour.astype(np.int32)
        cv2.drawContours(display, [contour], -1, color, 3)
        actual_center = tuple(np.rint(np.asarray(item["actual_center_mm"]) * ppm).astype(int))
        target_center = tuple(np.rint(np.asarray(item["target_center_mm"]) * ppm).astype(int))
        cv2.line(display, actual_center, target_center, color, 2, cv2.LINE_AA)
        label = (f"P{item['target_piece_id']} UNRESOLVED" if item["target_piece_id"] in unresolved else
                 f"P{item['target_piece_id']} MATCH?" if tentative else
                 f"P{item['target_piece_id']} e={item['position_error_mm']:.1f}mm "
                 f"a={item['rotation_correction_deg']:+.1f}")
        cv2.putText(display, label, (actual_center[0] + 5, actual_center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        labeled_targets.add(item["target_piece_id"])

    for actual_index, piece in enumerate(pieces):
        if actual_index not in matched_actual:
            cv2.drawContours(display, [piece.contour.astype(np.int32)], -1, (0, 0, 255), 3)
    for piece_id in sorted(unresolved - labeled_targets):
        center = tuple(np.rint(targets[piece_id]["center_mm"] * ppm).astype(int))
        cv2.putText(display, f"P{piece_id} UNRESOLVED", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    cv2.rectangle(display, (0, 0), (display.shape[1] - 1, 55), (30, 30, 30), -1)
    summary = (f"FINAL {report['status'].upper()} | contours "
               f"{report['detected_piece_count']}/{report['expected_piece_count']}")
    cv2.putText(display, summary, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display, "Green: target | Orange: uncertain | Red: mismatch", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (230, 230, 230), 1, cv2.LINE_AA)
    return display


def verify_final_frame(frame: np.ndarray, matrix: np.ndarray,
                       config: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    """保存最终实拍并生成机器可读的逐片复核报告。"""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    verify_config = dict(config)
    # Detection otherwise inherits source_region=bottom and misses every
    # correctly placed piece in the upper target region.
    verify_config["source_region"] = str(config.get("target_region", "bottom"))
    verify_config["source_region_bottom_mm"] = float(config["paper_height_mm"])
    verify_config["generic_contour_close_mm"] = 0.3
    verify_config["poker_morph_close_mm"] = 0.3
    verify_config["morphology_kernel_mm"] = 0.3
    verify_config["max_pieces"] = max(
        int(config.get("max_pieces", 4)), int(plan.get("piece_count", 4))
    )
    paper = warp_paper(frame, matrix, verify_config)
    pieces, mask = detect_pieces(paper, verify_config, diagnostics=True, limit_count=False)
    report = match_final_layout(pieces, plan, verify_config)
    source_config = dict(verify_config, source_region=str(config.get("source_region", "top")))
    source_pieces, _ = detect_pieces(paper, source_config, diagnostics=False, limit_count=False)
    report["source_leftover_count"] = len(source_pieces)
    if source_pieces:
        report["status"] = "fail"
    if len(pieces) != len(plan.get("pieces", [])):
        report["inspection_required"] = (
            "目标区检测到多余轮廓，请检查多余碎片或干扰物"
            if len(pieces) > len(plan.get("pieces", [])) else
            "目标碎片可能连片、遮挡或缺失；未匹配编号不代表确认漏放，逐片误差仅供参考"
        )
    report["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    overlay = draw_final_verification(paper, pieces, plan, report, verify_config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_image(OUTPUT_DIR / "final_capture.jpg", frame)
    write_image(OUTPUT_DIR / "final_corrected.jpg", paper)
    write_image(OUTPUT_DIR / "final_mask.png", mask)
    write_image(OUTPUT_DIR / "final_target_overlay.jpg", overlay)
    with (OUTPUT_DIR / "final_verification.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    level = "OK" if report["status"] == "pass" else "WARN"
    print(f"[{level}] 最终视觉复核：识别{report['detected_piece_count']}/"
          f"{report['expected_piece_count']}块，status={report['status']}")
    for item in report["matches"]:
        prefix = "[复核/仅供参考，勿用于标定]" if item.get("measurement_tentative") else "[复核]"
        print(f"{prefix} P{item['target_piece_id']}: 位置误差={item['position_error_mm']:.2f}mm，"
              f"仍需旋转={item['rotation_correction_deg']:+.2f}deg，"
              f"放置IoU={item['placed_iou']:.3f}，pass={item['passed']}")
    unresolved = report.get("unresolved_target_piece_ids", report["missing_target_piece_ids"])
    if unresolved:
        print(f"[WARN] 未能独立识别的目标（可能连片、遮挡或漏放）：{unresolved}")
    print(f"[OK] 最终复核图：{OUTPUT_DIR / 'final_target_overlay.jpg'}")
    return report


def capture_post_motion_frame(capture: cv2.VideoCapture,
                              config: dict[str, Any]) -> np.ndarray:
    """等待机构回原点和曝光稳定后读取最终帧。"""
    deadline = time.monotonic() + float(config.get("final_capture_settle_seconds", 0.8))
    frame: np.ndarray | None = None
    while time.monotonic() < deadline:
        success, frame = capture.read()
        if not success:
            raise RuntimeError("最终复核时摄像头读取失败")
    for _ in range(int(config.get("final_capture_extra_frames", 5))):
        success, frame = capture.read()
        if not success:
            raise RuntimeError("最终复核时摄像头读取失败")
    if frame is None:
        raise RuntimeError("最终复核未取得摄像头画面")
    return frame


def capture_immediate_frame(
        capture: cv2.VideoCapture,
        config: dict[str, Any]) -> tuple[np.ndarray, float]:
    """无GUI等待START，立即取得曝光稳定帧并记录计时起点。"""
    trigger_time = time.monotonic()
    print("[自动] 已无交互触发，等待曝光稳定...")
    warmup_seconds = float(config.get("auto_warmup_seconds", 0.8))
    deadline = time.monotonic() + warmup_seconds
    frame: np.ndarray | None = None
    while time.monotonic() < deadline:
        success, frame = capture.read()
        if not success:
            raise RuntimeError("摄像头读取失败")
    for _ in range(int(config.get("auto_extra_settle_frames", 5))):
        success, frame = capture.read()
        if not success:
            raise RuntimeError("摄像头读取失败")
    if frame is None:
        raise RuntimeError("无交互自动流程未取得摄像头画面")
    return frame, trigger_time


def _auto_detect_and_solve(frame: np.ndarray,
                           config: dict[str, Any],
                           saved_matrix: np.ndarray | None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                      list[Piece], list[PathResult]]:
    active_matrix, source = resolve_calibration_matrix(frame, config, saved_matrix)
    paper = warp_paper(frame, active_matrix, config)
    _, mask_override = mask_in_paper(frame, active_matrix, config)
    try:
        pieces, paths = process_paper(frame, paper, config, False, mask_override, save=False)
    except (RuntimeError, ValueError, cv2.error) as exc:
        save_failed_detection(frame, paper, mask_override, config, exc)
        raise
    return active_matrix, paper, mask_override, pieces, paths


def _piece_stability_signature(pieces: list[Piece], ppm: float) -> list[tuple[float, float, float, float, float]]:
    """Stable, order independent signature used by the pre-execution gate."""
    def axes(piece):
        # A triangle can have several equally small enclosing rectangles. Their
        # dimensions jump when minAreaRect switches the supporting edge, even
        # though the measured contour is stationary. Central moments are unique.
        contour = getattr(piece, "contour", None)
        if contour is not None:
            moments = cv2.moments(np.asarray(contour, np.float32))
            area = float(moments["m00"])
            if abs(area) > 1e-6:
                covariance = np.array([[moments["mu20"], moments["mu11"]],
                                       [moments["mu11"], moments["mu02"]]], np.float64) / area
                values = np.linalg.eigvalsh(covariance)
                if np.isfinite(values).all() and values[0] >= -1e-6:
                    return tuple(4.0 * np.sqrt(np.maximum(values, 0.0)) / ppm)
        return tuple(sorted((float(piece.width_mm), float(piece.height_mm))))
    return sorted((float(piece.center_mm(ppm)[0]), float(piece.center_mm(ppm)[1]),
                   float(piece.area_mm2), *axes(piece))
                   for piece in pieces)


def _signatures_stable(reference: list[tuple[float, float, float, float, float]],
                       candidate: list[tuple[float, float, float, float, float]],
                       config: dict[str, Any]) -> bool:
    if len(reference) != len(candidate):
        return False
    center_limit = float(config.get("stability_center_threshold_mm", 2.0))
    area_limit = float(config.get("stability_area_ratio_threshold", 0.12))
    shape_limit = float(config.get("stability_shape_threshold_mm", 3.0))
    def compatible(first, second):
        return (math.hypot(first[0] - second[0], first[1] - second[1]) <= center_limit
                and abs(first[2] - second[2]) / max(abs(first[2]), abs(second[2]), 1e-6) <= area_limit
                and abs(first[3] - second[3]) <= shape_limit
                and abs(first[4] - second[4]) <= shape_limit)
    return any(all(compatible(a, b) for a, b in zip(reference, order))
               for order in itertools.permutations(candidate))


def _write_blocked_auto_plan(reason: str) -> None:
    """Invalidate a previous plan before capture or whenever a gate fails."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "generated_at_epoch": time.time(),
               "piece_count": 0, "pieces": [], "ready_for_motion": False,
               "all_matches_ok": False, "all_paths_ok": False,
               "failure_reasons": [reason]}
    temporary = OUTPUT_DIR / "plan.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT_DIR / "plan.json")


def _target_poses_stable(first: list[PathResult], second: list[PathResult],
                         config: dict[str, Any]) -> bool:
    """Compare the same source pieces' planned translations and rotations."""
    ppm = float(config["pixels_per_mm"])
    if len(first) != len(second):
        return False
    position_limit = float(config.get("stability_target_threshold_mm", 3.0))
    angle_limit = float(config.get("stability_rotation_threshold_deg", 3.0))
    center_limit = float(config.get("stability_center_threshold_mm", 2.0))
    def compatible(left, right):
        delta = abs((left.match.rotation_deg-right.match.rotation_deg+180.)%360.-180.)
        return (math.dist(left.piece.center_mm(ppm), right.piece.center_mm(ppm)) <= center_limit
                and math.dist(left.match.target_pick_mm, right.match.target_pick_mm) <= position_limit
                and delta <= angle_limit)
    return any(all(compatible(a, b) for a, b in zip(first, order))
               for order in itertools.permutations(second))


def _solution_matches_reference(solution, pieces, reference, config):
    """Select an independently valid candidate matching an observed consensus."""
    ppm = float(config["pixels_per_mm"])
    poses = {pose.piece_index: pose for pose in solution.poses}
    probes = []
    for i, piece in enumerate(pieces):
        pose = poses[i]
        pick = generic_solver.transform_points(np.asarray([piece.pick_mm(ppm)]), pose.transform_3x3)[0]
        angle = generic_solver.rotation_angle_deg(pose.transform_3x3)
        probes.append((piece.center_mm(ppm), pick, angle))
    if len(probes) != len(reference):
        return False
    def agrees(probe, path):
        center, pick, angle = probe
        return (math.dist(center, path.piece.center_mm(ppm)) <= float(config.get("stability_center_threshold_mm", 2.0))
                and math.dist(pick, path.match.target_pick_mm) <= float(config.get("stability_target_threshold_mm", 3.0))
                and abs((angle-path.match.rotation_deg+180.)%360.-180.) <= float(config.get("stability_rotation_threshold_deg", 3.0)))
    return any(all(agrees(probe, path) for probe, path in zip(probes, order))
               for order in itertools.permutations(reference))


def _largest_consistent_group(items, compatible):
    """Find a mutually consistent subset; early outliers cannot steal members."""
    adjacency = [set() for _ in items]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if compatible(items[i], items[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)
    best = []
    def visit(chosen, remaining):
        nonlocal best
        if len(chosen) > len(best):
            best = chosen
        while remaining and len(chosen) + len(remaining) > len(best):
            index, *remaining = remaining
            visit(chosen + [index], [j for j in remaining if j in adjacency[index]])
    visit([], list(range(len(items))))
    return [items[index] for index in best]


def _stable_auto_detect_and_solve(capture: cv2.VideoCapture,
                                  first_frame: np.ndarray,
                                  config: dict[str, Any],
                                  saved_matrix: np.ndarray | None
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                             list[Piece], list[PathResult]]:
    """Require 4 of 5 consecutive frames to agree before execution.

    Only individually safe solutions count. Persist the selected frame and
    its plan together after acceptance; rejected frames never overwrite it.
    """
    required_frames = max(5, int(config.get("stability_frame_count", 5)))
    required_good = max(4, int(config.get("stability_required_good_frames", 4)))
    _write_blocked_auto_plan("连续帧检查尚未通过")
    for name in ("detection_preview.jpg", "detection_failure.json"):
        (OUTPUT_DIR / name).unlink(missing_ok=True)
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[Piece], list[PathResult]]] = []
    frame_indices: dict[int, int] = {}
    frames = [first_frame]
    for _ in range(required_frames - 1):
        success, next_frame = capture.read()
        if not success:
            raise RuntimeError("连续视觉识别读取摄像头失败")
        frames.append(next_frame)
    errors: list[str] = []
    frame_reports = []
    for index, current in enumerate(frames):
        try:
            observation = _auto_detect_and_solve(current, config, saved_matrix)
            observations.append(observation)
            frame_indices[id(observation)] = index
        except (RuntimeError, ValueError, cv2.error) as exc:
            errors.append(str(exc))
            frame_reports.append({"frame": index, "geometry_passed": False, "reasons": [str(exc)]})
    expected = int(config.get("expected_piece_count", 3))
    valid = []
    for item in observations:
        if len(item[3]) != expected or len(item[4]) != expected:
            errors.append(f"碎片或路径数量不是{expected}")
            frame_reports.append({"frame": frame_indices[id(item)], "geometry_passed": False,
                                  "reasons": [errors[-1]]})
            continue
        geometry_config = dict(config)
        if config.get("approval_workflow", False):
            geometry_config["enforce_pattern_gate"] = False
        reasons = motion_safety_reasons(item[4], geometry_config)
        frame_reports.append({"frame": frame_indices[id(item)], "geometry_passed": not reasons,
                              "reasons": reasons})
        if reasons:
            errors.append("；".join(reasons))
            continue
        valid.append(item)
    def save_stability_report(consistent, passed):
        (OUTPUT_DIR / "stability_report.json").write_text(json.dumps({
            "frames": required_frames, "geometry_good": len(valid),
            "consistent_good": consistent, "passed": passed, "rejections": errors,
            "frame_reports": sorted(frame_reports, key=lambda r:r["frame"]),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(valid) < required_good:
        save_stability_report(0, False)
        detail = errors[-1] if errors else "有效帧数量不足"
        reason = (f"连续{required_frames}帧中仅{len(valid)}帧同时识别到{expected}块并通过方案检查，"
                  f"至少需要{required_good}帧：{detail}")
        _write_blocked_auto_plan(reason)
        if observations:
            last = observations[-1]
            save_failed_detection(frames[frame_indices[id(last)]], last[1], last[2], config, reason)
        raise RuntimeError(reason)
    ppm = float(config["pixels_per_mm"])
    def observations_agree(a, b):
        return (
        _signatures_stable(_piece_stability_signature(a[3], ppm),
                           _piece_stability_signature(b[3], ppm), config)
        and _target_poses_stable(a[4], b[4], config))
    group = _largest_consistent_group(valid, observations_agree)
    if (config.get("planning_polygon_alternatives", False)
            and required_good - 1 <= len(group) < required_good):
        # Three agreeing frames may guide candidate selection in an outlier,
        # but that frame must still independently pass every original gate.
        # Never substitute the reference frame or invent a fourth observation.
        reference = group[0]
        retry_config = dict(config, _planning_reference_groups=[member[4] for member in group])
        for item in list(valid):
            if any(item is member for member in group):
                continue
            if not _signatures_stable(_piece_stability_signature(reference[3], ppm),
                                      _piece_stability_signature(item[3], ppm), config):
                continue
            index = frame_indices[id(item)]
            try:
                retried = _auto_detect_and_solve(frames[index], retry_config, saved_matrix)
                if motion_safety_reasons(retried[4], geometry_config):
                    continue
                if not all(observations_agree(member, retried) for member in group):
                    continue
                valid = [retried if member is item else member for member in valid]
                frame_indices[id(retried)] = index
                group = _largest_consistent_group(valid, observations_agree)
                for report in frame_reports:
                    if report["frame"] == index:
                        report.update(geometry_passed=True, consensus_candidate_retry=True, reasons=[])
                print(f"[稳定] 第{index+1}帧的独立备选解通过连续姿态复核")
                if len(group) >= required_good:
                    break
            except (RuntimeError, ValueError, cv2.error):
                continue
    if len(group) < required_good:
        save_stability_report(len(group), False)
        reason = "连续帧碎片或目标位置、旋转角度变化超过稳定阈值，禁止确认执行"
        _write_blocked_auto_plan(reason)
        last = valid[-1]
        save_failed_detection(frames[frame_indices[id(last)]], last[1], last[2], config, reason)
        raise RuntimeError(reason)
    chosen = group[0]
    index = frame_indices[id(chosen)]
    preview = draw_preview(chosen[1], chosen[3], chosen[4], config)
    save_outputs(frames[index], chosen[1], chosen[2], preview, chosen[4], config)
    save_stability_report(len(group), True)
    return chosen


# 这里构造的是下半区排放方案，不是几何拼合解。bbox是代码构造的放置框；缩小widths只缩小排布用宽度，实际碎片本身不会缩小。
def process_paper_fallback(original: np.ndarray, paper: np.ndarray,
                           config: dict[str, Any],
                           mask_override: np.ndarray) -> bool:
    ppm = float(config["pixels_per_mm"])
    paper_width = float(config["paper_width_mm"])
    paper_height = float(config["paper_height_mm"])
    spacing = float(config.get("fallback_spacing_mm", 15.0))
    separator = float(config.get("separator_line_y_mm", paper_height / 2.0))
    source_region = str(config.get("source_region", "top")).lower()
    source_min, source_max = ((0.0, separator) if source_region == "top"
                              else (separator, paper_height))

    pieces, mask = detect_pieces(paper, config, diagnostics=False,
                                 mask_override=mask_override)
    upper = sorted(
        [p for p in pieces if source_min <= p.center_mm(ppm)[1] < source_max],
        key=lambda p: p.area_mm2, reverse=True,
    )[:int(config.get("max_pieces", 4))]
    if not upper:
        print(f"[保底] {source_region}源区无有效碎片")
        return False
    upper = number_by_move_order(upper, config)
    widths = [float(cv2.boundingRect(p.contour)[2]) / ppm for p in upper]
    total_w = sum(widths) + (len(upper) - 1) * spacing
    if total_w > paper_width * 0.95:
        scale = paper_width * 0.95 / total_w
        spacing *= scale
        widths = [w * scale for w in widths]
        total_w = sum(widths) + (len(upper) - 1) * spacing

    start_x = (paper_width - total_w) / 2.0
    target_region = str(config.get("target_region", "bottom")).lower()
    target_min, target_max = ((0.0, separator) if target_region == "top"
                              else (separator, paper_height))
    target_y = (target_min + target_max) / 2.0
    matches: list[PuzzleMatch] = []
    cur_x = start_x
    for idx, piece in enumerate(upper):
        pw = widths[idx]
        src = np.asarray(piece.pick_mm(ppm), np.float64)
        tx = cur_x + pw / 2.0
        dx = tx - src[0]
        dy = target_y - src[1]
        tx3 = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        bbox = np.array([
            [cur_x, target_y - piece.height_mm / 2],
            [cur_x + pw, target_y - piece.height_mm / 2],
            [cur_x + pw, target_y + piece.height_mm / 2],
            [cur_x, target_y + piece.height_mm / 2],
        ], np.float64)
        matches.append(PuzzleMatch(
            piece=piece, template_id=f"F{idx+1}",
            target_polygon_mm=bbox,
            target_pick_mm=(tx, target_y),
            rotation_deg=0.0, rotation_direction="NONE",
            area_error_ratio=0.0, shape_distance=0.0,
            # status='ok'（处理结果的状态标记）；取值过程：'ok'。
            match_iou=1.0, fit_scale=1.0, status="ok",
            solver_mode="fallback", transform_3x3=tx3,
        ))
        cur_x += pw + spacing

    paths = []
    for match in matches:
        target_px = mm_to_px(match.target_pick_mm, ppm)
        path, planner, status = plan_single_path(
            match.piece.pick_px, target_px, None, config)
        paths.append(PathResult(match, path, planner, status))

    preview = draw_preview(paper, [m.piece for m in matches], paths, config)
    save_outputs(original, paper, mask, preview, paths, config)
    reasons = motion_safety_reasons(paths, config)
    print(f"[保底] {source_region}{len(paths)}块→{target_region}，间距{spacing:.1f}mm，"
          f"ready_for_motion={not reasons}")
    if reasons:
        for r in reasons:
            print(f"[保底] {r}")
    return not reasons


# 顺序是取帧→重试识别求解→必要时兜底→按dry_run选择打印或真实发送→真实执行后拍照复核。相机最终由finally释放。
def run_auto(config: dict[str, Any], camera_id: int | None,
             serial_port: str | None, force_dry_run: bool,
             start_immediately: bool = False) -> None:
    """一键流程：识别求解、串口执行、回原点并自动拍照复核。"""
    _write_blocked_auto_plan("本轮尚未完成拍照与连续帧检查")
    matrix = load_calibration(config)
    capture = open_camera(config, camera_id)
    try:
        while True:
            captured = (
                capture_immediate_frame(capture, config)
                if start_immediately
                else wait_for_start_and_capture(capture, config)
            )
            if captured is None:
                print("[自动] 已退出。")
                return
            frame, trigger_time = captured

            max_retries = int(config.get("auto_max_retries", 3))
            fallback_enabled = bool(config.get("auto_fallback_enabled", True))
            success = False

            for attempt in range(1, max_retries + 1):
                try:
                    active_matrix, paper, mask_override, _, _ = (
                        _stable_auto_detect_and_solve(capture, frame, config, matrix))
                    if active_matrix is not None:
                        source = "auto_board"
                        matrix = active_matrix
                    else:
                        source = "saved"
                    success = True
                    break
                except RuntimeError as exc:
                    if attempt < max_retries:
                        print(f"[自动] 第{attempt}次失败：{exc}，"
                              f"剩余{max_retries - attempt}次重试...")
                        time.sleep(0.3)
                        frame, _ = capture_immediate_frame(capture, config)
                    elif fallback_enabled:
                        print(f"[自动] {max_retries}次均失败：{exc}")
                        print("[自动] 启用保底方案：上半区碎片移至下半区")
                        try:
                            active_matrix, source = resolve_calibration_matrix(
                                frame, config, matrix)
                            if source == "auto_board":
                                matrix = active_matrix
                            paper = warp_paper(frame, active_matrix, config)
                            _, mask_override = mask_in_paper(
                                frame, active_matrix, config)
                            success = process_paper_fallback(
                                frame, paper, config, mask_override)
                        except RuntimeError as fallback_exc:
                            print(f"[保底] 失败：{fallback_exc}")
                    else:
                        print(f"[自动] {max_retries}次均失败：{exc}")

            if not success:
                print("[自动] 本次未生成可执行计划，可重新摆放后再次START。\n")
                if start_immediately:
                    raise RuntimeError("本次未生成通过连续帧检查的方案")
                continue

            port = serial_port if serial_port is not None else str(config.get("serial_port", ""))
            dry_run = force_dry_run or not port
            if config.get("approval_workflow", False):
                if not dry_run:
                    raise RuntimeError("正式执行必须使用 --execute-approved，禁止重新求解后直接运动")
                print("[OK] 方案已锁定，等待屏幕预览确认")
                return
            motion_completed = False
            try:
                if (not dry_run and
                        str(config.get("execution_backend", "legacy_single_port"))
                        == "dual_grbl_stm32"):
                    plan = json.loads((OUTPUT_DIR / "plan.json").read_text(encoding="utf-8"))
                    executor = DualSerialExecutor(config)
                    executor.open()
                    try:
                        executor.execute(plan)
                    finally:
                        executor.close()
                else:
                    serial_transport.send_plan_file(
                        OUTPUT_DIR / "plan.json",
                        dry_run=dry_run,
                        port=port,
                        baudrate=int(config.get("serial_baudrate", 115200)),
                        ack_timeout_seconds=float(config.get("serial_ack_timeout_seconds", 2.0)),
                        command_delay_seconds=float(config.get("serial_command_delay_seconds", 0.0)),
                    )
                motion_completed = not dry_run
            except (serial_transport.SerialPlanError, DualExecutionError) as exc:
                raise RuntimeError(f"自动执行未完成：{exc}") from exc
            if dry_run and not force_dry_run:
                print("[提示] 未配置串口（--port或config.json的serial_port），"
                      "以上仅为dry-run打印，未连接主控。")

            motion_elapsed = time.monotonic() - trigger_time
            target_seconds = float(config.get("auto_target_seconds", 80.0))
            worst_seconds = float(config.get("auto_worst_case_seconds", 110.0))
            level = "OK" if motion_elapsed <= target_seconds else (
                "WARN" if motion_elapsed <= worst_seconds else "FAIL")
            print(f"[{level}] 本次自START至发送完成用时{motion_elapsed:.1f}s"
                  f"（目标≤{target_seconds:.0f}s，最坏<{worst_seconds:.0f}s）")

            if motion_completed and bool(config.get("final_verification_enabled", True)):
                try:
                    final_frame = capture_post_motion_frame(capture, config)
                    report = verify_final_frame(
                        final_frame, active_matrix, config, OUTPUT_DIR / "plan.json"
                    )
                    if report.get("status") != "pass":
                        raise RuntimeError("拼图最终复拍未通过，请查看final_verification.json")
                except (RuntimeError, OSError, ValueError, cv2.error, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"运动已完成，但最终视觉复核失败：{exc}") from exc

            print("[自动] 本次流程结束，可再次按START处理下一组碎片。\n")
            if start_immediately:
                return
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("[OK] 摄像头已释放")

# 原有文档字符串有乱码：self选择固定图2并设置100×60毫米；site选择通用几何模式。这里修改配置副本，不写回config.json。
# =============================================================================
# 【分区】题目套装、命令行与 main()
# 功能：self=固定图2；site=通用几何。扑克现场通常 site + 花纹配置。
# 可修改：命令行默认相机号；阈值放 config。
# 看情况改：只改内存副本。
# 不要改：互斥运行模式；先 load_config 再 apply_puzzle_set。
# =============================================================================
def apply_puzzle_set(config: dict[str, Any], puzzle_set: str | None) -> dict[str, Any]:
    """??????????????????"""
    if puzzle_set is None:
        return config
    updated = dict(config)
    if puzzle_set == "self":
        updated["puzzle_mode"] = "fixed_figure_2"
        updated["target_rectangle_size_mm"] = [100.0, 60.0]
    elif puzzle_set == "site":
        updated["puzzle_mode"] = "generic_geometry"
    else:
        raise ValueError(f"???????{puzzle_set}")
    return updated

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USB拼图视觉与A4最短正交路径生成")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--camera", type=int, nargs="?", const=0, help="摄像头编号，默认0")
    mode.add_argument("--image", type=Path, help="处理单张图片")
    mode.add_argument("--demo", action="store_true", help="运行合成演示")
    mode.add_argument(
        "--auto", action="store_true",
        help="一键流程：自动定位底板，等待START后识别求解并按需发送串口",
    )
    mode.add_argument(
        # const=OUTPUT_DIR / 'plan.json'（选项出现但省略值时使用的值）；取值过程：计算 OUTPUT_DIR / 'plan.json'。
        "--send-plan", type=Path, nargs="?", const=OUTPUT_DIR / "plan.json",
        help="发送运动计划；省略路径时使用output/plan.json",
    )
    parser.add_argument(
        "--puzzle-set", choices=("self", "site"),
        help="?????self=?????2??100x60mm?site=??????",
    )
    parser.add_argument("--camera-id", type=int, default=0,
                        help="--auto使用的摄像头编号，默认0（--camera已被用作模式选择，"
                             "两者互斥，故--auto单独用这个参数指定摄像头）")
    parser.add_argument("--start-immediately", action="store_true",
                        help="--auto无需点击START，立即执行一轮后退出")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印串口指令，不打开串口（--send-plan或--auto均可用）")
    parser.add_argument("--port", help="串口名称，例如COM5（--send-plan或--auto均可用）")
    parser.add_argument("--baud", type=int, help="串口波特率，默认读取config.json")
    parser.add_argument("--ack-timeout", type=float, help="等待单条ACK的秒数")
    args = parser.parse_args()
    if args.start_immediately and not args.auto:
        parser.error("--start-immediately必须与--auto一起使用")
    serial_options_used = (
        args.dry_run or args.port is not None
        or args.baud is not None or args.ack_timeout is not None
    )
    if serial_options_used and args.send_plan is None and not args.auto:
        parser.error("--dry-run/--port/--baud/--ack-timeout必须与--send-plan或--auto一起使用")
    return args


def main() -> int:
    args = parse_args()
    config = apply_puzzle_set(load_config(), args.puzzle_set)
    try:
        if args.auto and not args.dry_run and (args.port or config.get("serial_port")):
            raise RuntimeError("自动真实执行必须使用 --execute-approved，禁止重新求解后发送")
        if args.send_plan is not None:
            if not args.dry_run:
                raise RuntimeError("真实发送仅允许 run_real.py --execute-approved，旧单串口入口已禁用")
            serial_transport.send_plan_file(
                args.send_plan,
                dry_run=bool(args.dry_run),
                port=args.port or str(config.get("serial_port", "")),
                baudrate=args.baud or int(config.get("serial_baudrate", 115200)),
                ack_timeout_seconds=(
                    args.ack_timeout
                    if args.ack_timeout is not None
                    else float(config.get("serial_ack_timeout_seconds", 2.0))
                ),
                command_delay_seconds=float(config.get("serial_command_delay_seconds", 0.0)),
            )
        elif args.demo:
            run_demo(config)
        elif args.image is not None:
            run_image(args.image, config)
        elif args.auto:
            run_auto(config, args.camera_id, args.port, bool(args.dry_run), bool(args.start_immediately))
        else:
            run_camera(config, args.camera)
        return 0
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        print(f"[ERR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
