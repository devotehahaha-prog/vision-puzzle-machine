"""USB摄像头碎片识别、编号与A4最短正交路径生成。"""
# 【中文阅读导引】以下新增#注释解释代码用途；原有字符串、计算、默认值和执行顺序保持不变。
# 本文件负责把相机画面变成搬运计划：图像 → 纸面校正 → 二值分割 → 碎片几何 → 目标姿态 → 单轴路径 → plan.json。
# 建议先看文件末尾 main()，再看 process_paper()；需要自动执行时接着看 run_auto() 和 serial_transport.py。
# 图像数组按[行,列]即[y,x]访问，但坐标点通常写成(x,y)；shape前两项是(高,宽)。
# 纸面原点在左上，X向右、Y向下。这里正旋转角在画面上表现为顺时针CW，负值为逆时针CCW。
# 变量后缀_px是像素，_mm是毫米，_mm2是平方毫米，_deg是角度；ppm=像素/毫米。长度除ppm、面积除ppm²才能换算。
# 二值掩膜是与图像同宽高的单通道数组：0排除背景，255保留目标；彩色图像默认通道顺序为BGR。
# Piece保存识别结果；PuzzleMatch保存目标位置及旋转；PathResult把匹配与搬运折线关联起来。
# 固定图2直接匹配A/B/C/D；generic_geometry交给generic_solver搜索。花纹功能还需edge_matcher模块及相应配置。
# 代码中的fallback/relaxed表示原有兜底策略；生成了可发送计划并不等同于已拼成严格合格的矩形。
# 语法约定：缩进决定代码归属；=赋值，==比较；列表和数组下标从0开始；None表示没有值；冒号后的类型主要用于阅读和检查。
# 跨行表达式属于同一条语句，括号结束前不会另起一条指令；字典条目和命名实参也在相邻注释中解释。
# =============================================================================
# 【分区】导入、路径常量、图2模板
# 功能：加载 OpenCV/求解器/串口模块；锁定 config.json、calibration.json、output 路径；内置图2的 A/B/C/D 毫米顶点。
# 可修改：DEMO_PLACEMENTS 只影响 --demo 合成图位置，不影响实机。
# 看情况改：edge_matcher 导入失败时花纹验证自动关闭，扑克模式需要该模块存在。
# 不要改：ROOT/CONFIG_PATH 相对本文件；FIGURE2_TEMPLATE_POLYGONS_MM 的顶点（题目图2外形）。改模板等于改赛题答案。
# =============================================================================
# 从 __future__ 导入需要的类型或工具。
from __future__ import annotations

# 导入：argparse。
import argparse
# 导入：itertools。
import itertools
# 导入：json。
import json
# 导入：math。
import math
# 导入：os。
import os
# 导入：time。
import time
# 从 dataclasses 导入需要的类型或工具。
from dataclasses import dataclass
# 从 pathlib 导入需要的类型或工具。
from pathlib import Path
# 从 typing 导入需要的类型或工具。
from typing import Any, Iterable

# 导入：cv2。
import cv2
# 导入：numpy。
import numpy as np

# 导入：generic_solver。
import generic_solver
# 导入：orthogonal_path_planner。
import orthogonal_path_planner as path_planner
# 导入：serial_transport。
import serial_transport

# 执行可能出错的代码，并由后面的异常分支处理错误。
try:
    # 导入：edge_matcher。
    import edge_matcher
    # 计算并保存到 _PATTERN_MATCHER_AVAILABLE（可选花纹匹配模块是否导入成功）。
    _PATTERN_MATCHER_AVAILABLE = True
# 捕获ImportError异常，转入下面的处理代码。
except ImportError:
    # 计算并保存到 edge_matcher（边·matcher）。
    edge_matcher = None
    # 计算并保存到 _PATTERN_MATCHER_AVAILABLE（可选花纹匹配模块是否导入成功）。
    _PATTERN_MATCHER_AVAILABLE = False

# 计算并保存到 ROOT（当前这套程序的根目录）。
ROOT = Path(__file__).resolve().parent
# 计算并保存到 CONFIG_PATH（config.json配置文件路径）。
CONFIG_PATH = ROOT / "config.json"
# 计算并保存到 CALIBRATION_PATH（calibration.json标定文件路径）。
CALIBRATION_PATH = ROOT / "calibration.json"
# 计算并保存到 OUTPUT_DIR（图像和计划输出目录）。
OUTPUT_DIR = ROOT / "output"


# 计算并保存到 FIGURE2_TEMPLATE_POLYGONS_MM（图2中A/B/C/D四块固定模板的毫米顶点表）。
FIGURE2_TEMPLATE_POLYGONS_MM: dict[str, tuple[tuple[float, float], ...]] = {
    # 字典字段'A'（A）：创建数据容器。
    "A": ((0.0, 0.0), (20.0, 0.0), (36.0, 12.0), (0.0, 20.0)),
    # 字典字段'B'（B）：创建数据容器。
    "B": ((0.0, 20.0), (36.0, 12.0), (76.0, 42.0), (0.0, 30.0)),
    # 字典字段'C'（C）：创建数据容器。
    "C": ((0.0, 30.0), (76.0, 42.0), (100.0, 60.0), (0.0, 60.0)),
    # 字典字段'D'（D）：创建数据容器。
    "D": ((20.0, 0.0), (100.0, 0.0), (100.0, 60.0),
          (76.0, 42.0), (36.0, 12.0)),
}

# 计算并保存到 DEMO_PLACEMENTS（合成演示中各块碎片的中心位置和旋转角）。
DEMO_PLACEMENTS: dict[str, tuple[tuple[float, float], float]] = {
    # 字典字段'A'（A）：创建数据容器。
    "A": ((25.0, 25.0), 25.0),
    # 字典字段'B'（B）：创建数据容器。
    "B": ((92.0, 25.0), -10.0),
    # 字典字段'C'（C）：创建数据容器。
    "C": ((55.0, 105.0), -10.0),
    # 字典字段'D'（D）：创建数据容器。
    "D": ((160.0, 85.0), 15.0),
}


# =============================================================================
# 【分区】核心数据结构 Piece / PuzzleMatch / PathResult
# 功能：Piece=识别结果；PuzzleMatch=目标姿态与质量；PathResult=匹配+正交折线。
# 可修改：无现场参数。
# 看情况改：新字段只能追加并给默认值，否则旧 plan.json / 仿真会炸。
# 不要改：center_px/pick_px 单位是像素；旋转正角=画面顺时针。不要把像素当毫米用。
# =============================================================================
# 使用dataclass装饰器，按下方字段自动生成__init__、__repr__和比较方法。
@dataclass
# 【类：Piece】单块碎片的视觉结果。
class Piece:
    """单块碎片的视觉结果。"""
    # 声明字段：piece_id（面向显示和计划的碎片编号）。
    piece_id: int
    # 声明字段：contour（碎片轮廓点数组）。
    contour: np.ndarray
    # 声明字段：center_px（视觉质心的像素坐标(x,y)）。
    center_px: tuple[float, float]
    # 声明字段：pick_px（磁吸点的像素坐标(x,y)）。
    pick_px: tuple[float, float]
    # 声明字段：angle_deg（角度，单位为度）。
    angle_deg: float
    # 声明字段：area_mm2（碎片面积，单位为平方毫米）。
    area_mm2: float
    # 声明字段：width_mm（宽度，单位为毫米）。
    width_mm: float
    # 声明字段：height_mm（高度，单位为毫米）。
    height_mm: float

    # 【函数：center_mm】将当前碎片的像素质心转换成毫米质心。
    # 参数 ppm（每毫米对应的像素数）：浮点数。
    # 返回类型：元组（依次为浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
    def center_mm(self, ppm: float) -> tuple[float, float]:
        # 返回结果：创建数据容器。
        return self.center_px[0] / ppm, self.center_px[1] / ppm

    # 【函数：pick_mm】将当前碎片的像素吸取点转换成毫米吸取点。
    # 参数 ppm（每毫米对应的像素数）：浮点数。
    # 返回类型：元组（依次为浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
    def pick_mm(self, ppm: float) -> tuple[float, float]:
        # 返回结果：创建数据容器。
        return self.pick_px[0] / ppm, self.pick_px[1] / ppm


# 使用dataclass装饰器，按下方字段自动生成__init__、__repr__和比较方法。
@dataclass
# 【类：PuzzleMatch】源碎片到题目图2固定模板的匹配和刚体变换。
class PuzzleMatch:
    """源碎片到题目图2固定模板的匹配和刚体变换。"""
    # 声明字段：piece（当前碎片数据）。
    piece: Piece
    # 声明字段：template_id（目标模板编号）。
    template_id: str
    # 声明字段：target_polygon_mm（目标布局中的多边形顶点（毫米））。
    target_polygon_mm: np.ndarray
    # 声明字段：target_pick_mm（放置后磁吸点的目标坐标（毫米））。
    target_pick_mm: tuple[float, float]
    # 声明字段：rotation_deg（碎片需要旋转的有符号角度（度））。
    rotation_deg: float
    # 声明字段：rotation_direction（旋转方向标记）。
    rotation_direction: str
    # 声明字段：area_error_ratio（面积相对误差）。
    area_error_ratio: float
    # 声明字段：shape_distance（形状差异指标）。
    shape_distance: float
    # 声明字段：match_iou（形状匹配的交并比）。
    match_iou: float
    # 声明字段：fit_scale（形状匹配时的尺度比；刚体搬运应保持为1）。
    fit_scale: float
    # 声明字段：status（处理结果的状态标记）。
    status: str
    # 计算并保存到 solver_mode（求解方式标记）。
    solver_mode: str = "fixed_figure_2"
    # 计算并保存到 transform_3x3（3×3齐次变换矩阵（把源坐标映射到目标坐标））。
    transform_3x3: np.ndarray | None = None
    # 计算并保存到 solution_score（几何方案的代价评分，通常越小越好）。
    solution_score: float = 0.0
    # 计算并保存到 fill_error_ratio（矩形填充误差比例）。
    fill_error_ratio: float = 0.0
    # 计算并保存到 overlap_ratio（碎片间重叠比例）。
    overlap_ratio: float = 0.0
    # 计算并保存到 endpoint_error_mm（拼缝对应端点的误差（毫米））。
    endpoint_error_mm: float = 0.0
    # 计算并保存到 boundary_error_mm（目标矩形边界误差（毫米））。
    boundary_error_mm: float = 0.0
    # 计算并保存到 boundary_gap_ratio（矩形边界缺口比例）。
    boundary_gap_ratio: float = 0.0
    # 计算并保存到 target_rectangle_top_left_mm（目标矩形左上角坐标（毫米））。
    target_rectangle_top_left_mm: tuple[float, float] | None = None
    # 计算并保存到 target_rectangle_size_mm（目标矩形宽高（毫米））。
    target_rectangle_size_mm: tuple[float, float] | None = None
    # 计算并保存到 target_clearance_shift_mm（为留出碎片间隙施加的平移（毫米））。
    target_clearance_shift_mm: tuple[float, float] = (0.0, 0.0)
    # 计算并保存到 pattern_verification（花纹复核结果，None表示没有该结果）。
    pattern_verification: dict | None = None
    # 计算并保存到 relaxed_override（是否采用放宽质量要求的兜底结果）。
    relaxed_override: bool = False


# 使用dataclass装饰器，按下方字段自动生成__init__、__repr__和比较方法。
@dataclass
# 【类：PathResult】单块碎片的模板匹配和搬运路径。
class PathResult:
    """单块碎片的模板匹配和搬运路径。"""
    # 声明字段：match（当前匹配记录）。
    match: PuzzleMatch
    # 声明字段：path_px（按行进顺序排列的像素路径点）。
    path_px: list[tuple[float, float]]
    # 声明字段：planner（路径规划方法标记）。
    planner: str
    # 声明字段：status（处理结果的状态标记）。
    status: str

    # 把下面的方法包装成只读属性，外部用对象.属性名读取，无须写括号。
    @property
    # 【函数：piece】从路径结果关联的匹配记录中取出原始碎片。
    # 返回类型：Piece；箭头->是类型提示，不会替你转换实际返回值。
    def piece(self) -> Piece:
        # 返回结果：self.match.piece。
        return self.match.piece

    # 把下面的方法包装成只读属性，外部用对象.属性名读取，无须写括号。
    @property
    # 【函数：target_mm】从匹配记录中取出目标磁吸点的毫米坐标。
    # 返回类型：元组（依次为浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
    def target_mm(self) -> tuple[float, float]:
        # 返回结果：self.match.target_pick_mm（self.match.target·磁吸点·毫米）。
        return self.match.target_pick_mm

# =============================================================================
# 【分区】配置与图像读写、纸面像素尺寸
# 功能：读 config.json；用 numpy 读写图片（兼容中文路径）；毫米×ppm→像素。
# 可修改：阈值、颜色、串口、目标矩形等一律改 config.json，不要把魔数写进本文件。
# 看情况改：pixels_per_mm 必须与标定、相机高度一致；只改配置不重标，尺寸会整体缩放错。
# 不要改：load_config 的 utf-8-sig；write_image 的 tofile 中文路径写法。
# =============================================================================
# 【函数：load_config】读取当前程序目录中的config.json，返回配置字典。
# 返回类型：字典（键为字符串，值为任意类型）；箭头->是类型提示，不会替你转换实际返回值。
def load_config() -> dict[str, Any]:
    # 打开并自动管理资源，结束后自动释放。
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
        # 返回结果：从已打开的文件解析JSON为Python对象。
        return json.load(file)


# 【函数：read_image】兼容Windows中文路径的图片读取。
# 参数 path（当前文件路径或几何路径）：文件系统路径对象。
# 返回类型：NumPy数组或None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def read_image(path: Path) -> np.ndarray | None:
    """兼容Windows中文路径的图片读取。"""
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 计算并保存到 data（数据）。
        data = np.fromfile(str(path), dtype=np.uint8)
        # 返回结果：从压缩图像字节解码出像素数组。
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    # 捕获(OSError, cv2.error)异常，转入下面的处理代码。
    except (OSError, cv2.error):
        # 返回结果：None。
        return None


# 【函数：write_image】兼容Windows中文路径的图片保存。
# 参数 path（当前文件路径或几何路径）：文件系统路径对象。
# 参数 image（图像数组）：NumPy数组。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def write_image(path: Path, image: np.ndarray) -> None:
    """兼容Windows中文路径的图片保存。"""
    # 计算并保存到 extension。
    extension = path.suffix or ".png"
    # 计算并保存到 success（本步骤是否成功）、encoded。
    success, encoded = cv2.imencode(extension, image)
    # 判断条件；满足时执行下面代码：not success。
    if not success:
        # 抛出异常，通知上层处理：RuntimeError(f'图片编码失败：{path.name}')。
        raise RuntimeError(f"图片编码失败：{path.name}")
    # 调用函数：encoded.tofile。
    encoded.tofile(str(path))


# 【函数：paper_size_px】将配置中的纸面毫米宽高换算成整数像素宽高。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为整数、整数）；箭头->是类型提示，不会替你转换实际返回值。
def paper_size_px(config: dict[str, Any]) -> tuple[int, int]:
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 返回结果：创建数据容器。
    return (
        int(round(float(config["paper_width_mm"]) * ppm)),
        int(round(float(config["paper_height_mm"]) * ppm)),
    )


# 【函数：workspace_size_px】返回与A4标定区域完全一致的路径规划工作区。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为整数、整数）；箭头->是类型提示，不会替你转换实际返回值。
def workspace_size_px(config: dict[str, Any]) -> tuple[int, int]:
    """返回与A4标定区域完全一致的路径规划工作区。"""
    # 返回结果：调用 path_planner.workspace_size_px。
    return path_planner.workspace_size_px(
        float(config["paper_width_mm"]),
        float(config["paper_height_mm"]),
        float(config["pixels_per_mm"]),
    )


# =============================================================================
# 【分区】A4 四角标定与透视矩阵
# 功能：找黑框四角 / 手工点四角 → 透视矩阵写入 calibration.json → warp_paper 把原图拉成正 A4。
# 可修改：黑框检测阈值、四边形面积比、点序（左上开始顺时针）相关配置项。
# 看情况改：换相机或高度后必须重新标定，不要沿用旧 calibration.json。
# 不要改：order_quad_points 的角点顺序；矩阵是 3×3 透视，不要改成仿射。
# =============================================================================
# 【函数：save_calibration】保存四个图像角点、透视矩阵、原图尺寸和标定时间。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组。
# 参数 matrix（本函数使用的矩阵）：NumPy数组。
# 参数 image_size（图像尺寸，顺序为(宽,高)）：元组（依次为整数、整数）。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def save_calibration(points: np.ndarray, matrix: np.ndarray,
                     image_size: tuple[int, int]) -> None:
    # 计算并保存到 payload（准备写入文件或发送的数据）。
    payload = {
        # 字典字段'image_points'（图像·点集）：调用 points.astype(float).tolist。
        "image_points": points.astype(float).tolist(),
        # 字典字段'perspective_matrix'（perspective·矩阵）：调用 matrix.astype(float).tolist。
        "perspective_matrix": matrix.astype(float).tolist(),
        # 字典字段'image_size'（图像尺寸，顺序为(宽,高)）：创建数据容器。
        "image_size": [int(image_size[0]), int(image_size[1])],
        # 字典字段'created_at'（created_at）：按指定格式生成时间文本。
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # 打开并自动管理资源，结束后自动释放。
    with CALIBRATION_PATH.open("w", encoding="utf-8") as file:
        # 调用函数：json.dump。
        json.dump(payload, file, ensure_ascii=False, indent=2)


# 【函数：order_quad_points】把任意点击顺序的四个角排序为左上、右上、右下、左下。
# 先按相对中心的极角排成环，再把x+y最小的点放在首位；这是本实现选左上角的启发式规则。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def order_quad_points(points: np.ndarray) -> np.ndarray:
    """把任意点击顺序的四个角排序为左上、右上、右下、左下。"""
    # 计算并保存到 source（源数据或源位置）。
    source = np.asarray(points, np.float32).reshape(4, 2)
    # 计算并保存到 center（本步骤使用的中心位置）。
    center = np.mean(source, axis=0)
    # 计算并保存到 angles（角度序列）。
    angles = np.arctan2(source[:, 1] - center[1], source[:, 0] - center[0])
    # 计算并保存到 ordered（已排序的数据）。
    ordered = source[np.argsort(angles)]
    # 计算并保存到 start（起始值或起点）。
    start = int(np.argmin(np.sum(ordered, axis=1)))
    # 返回结果：调用 np.roll(ordered, -start, axis=0).astype。
    return np.roll(ordered, -start, axis=0).astype(np.float32)


# 【函数：valid_quad】仅拒绝重复点、明显过短边和退化四边形，不限制A4画面占比。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组。
# 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
def valid_quad(points: np.ndarray) -> bool:
    """仅拒绝重复点、明显过短边和退化四边形，不限制A4画面占比。"""
    # 计算并保存到 source（源数据或源位置）。
    source = np.asarray(points, np.float32).reshape(4, 2)
    # 判断条件；满足时执行下面代码：not np.all(np.isfinite(source))。
    if not np.all(np.isfinite(source)):
        # 返回结果：False。
        return False
    # 计算并保存到 edges（边集合）。
    edges = [float(np.linalg.norm(source[(index + 1) % 4] - source[index]))
             # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
             for index in range(4)]
    # 计算并保存到 area（面积）。
    area = abs(float(cv2.contourArea(source)))
    # 返回结果：组合多个条件：min(edges) >= 20.0 and area >= 1000.0 and cv2.isContourConvex(source.astyp…。
    return min(edges) >= 20.0 and area >= 1000.0 and cv2.isContourConvex(source.astype(np.int32))


# 【函数：calibration_area_ratio】返回标定四边形占原始画面的比例。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组。
# 参数 image_size（图像尺寸，顺序为(宽,高)）：元组（依次为整数、整数）。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def calibration_area_ratio(points: np.ndarray, image_size: tuple[int, int]) -> float:
    """返回标定四边形占原始画面的比例。"""
    # 计算并保存到 width（当前区域宽度）、height（当前区域高度）。
    width, height = image_size
    # 判断条件；满足时执行下面代码：width <= 0 or height <= 0。
    if width <= 0 or height <= 0:
        # 返回结果：0.0。
        return 0.0
    # 返回结果：计算 abs(float(cv2.contourArea(np.asarray(points, np.float32)))) / float(width …。
    return abs(float(cv2.contourArea(np.asarray(points, np.float32)))) / float(width * height)


# 【函数：calibration_matrix】把底板四角映射到俯视矩形，返回排序后的角点和透视矩阵。
# 透视矩阵用于消除相机斜拍影响，它不同于碎片搬运的刚体矩阵；前者允许透视缩放，后者只允许平移和旋转。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为NumPy数组、NumPy数组）；箭头->是类型提示，不会替你转换实际返回值。
def calibration_matrix(points: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    # 计算并保存到 source（源数据或源位置）。
    source = order_quad_points(points)
    # 判断条件；满足时执行下面代码：not valid_quad(source)。
    if not valid_quad(source):
        # 抛出异常，通知上层处理：ValueError('四个角点重复、距离过近或不能组成有效四边形')。
        raise ValueError("四个角点重复、距离过近或不能组成有效四边形")
    # 计算并保存到 width（当前区域宽度）、height（当前区域高度）。
    width, height = paper_size_px(config)
    # 计算并保存到 destination。
    destination = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1],
                              [0, height - 1]], np.float32)
    # 返回结果：创建数据容器。
    return source, cv2.getPerspectiveTransform(source, destination)


# 【函数：detect_dark_board_points】从当前画面自动定位深色A4底板，返回左上、右上、右下、左下四角。
# 对多个亮度阈值反复分割深色区域，再用面积、四边形、边缘倾斜及内部深浅像素比例筛选；最后取综合评分最高的四角。
# 参数 frame（摄像头的一帧原始图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组或None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def detect_dark_board_points(frame: np.ndarray,
                             config: dict[str, Any]) -> np.ndarray | None:
    """从当前画面自动定位深色A4底板，返回左上、右上、右下、左下四角。"""
    # 判断条件；满足时执行下面代码：not bool(config.get('auto_board_calibration', True))。
    if not bool(config.get("auto_board_calibration", True)):
        # 返回结果：None。
        return None
    # 判断条件；满足时执行下面代码：str(config.get('segmentation_mode', 'pink_hsv')) != 'white_piece'。
    if str(config.get("segmentation_mode", "pink_hsv")) != "white_piece":
        # 返回结果：None。
        return None
    # 判断条件；满足时执行下面代码：frame.ndim != 3 or min(frame.shape[:2]) < 100。
    if frame.ndim != 3 or min(frame.shape[:2]) < 100:
        # 返回结果：None。
        return None

    # 计算并保存到 height（当前区域高度）、width（当前区域宽度）。
    height, width = frame.shape[:2]
    # 计算并保存到 image_area（图像·面积）。
    image_area = float(width * height)
    # 计算并保存到 hsv（色相H、饱和度S、亮度V组成的图像）。
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 计算并保存到 saturation_max（饱和度·最大）。
    saturation_max = int(config.get("auto_board_saturation_max", 110))
    # 计算并保存到 value_thresholds（值·阈值集合）。
    value_thresholds = config.get(
        "auto_board_value_thresholds", [80, 90, 100, 110, 120, 130, 140]
    )
    # 计算并保存到 minimum_area_ratio（最小·面积·比例）。
    minimum_area_ratio = float(config.get("auto_board_min_area_ratio", 0.08))
    # 计算并保存到 maximum_area_ratio（最大·面积·比例）。
    maximum_area_ratio = float(config.get("auto_board_max_area_ratio", 0.35))
    # 计算并保存到 minimum_dark_fraction（最小·深色·占比）。
    minimum_dark_fraction = float(config.get("auto_board_min_dark_fraction", 0.55))
    # 计算并保存到 maximum_white_fraction（最大·白色·占比）。
    maximum_white_fraction = float(config.get("auto_board_max_white_fraction", 0.35))
    # 计算并保存到 minimum_aspect（最小·aspect）。
    minimum_aspect = float(config.get("auto_board_min_aspect_ratio", 0.45))
    # 计算并保存到 maximum_aspect（最大·aspect）。
    maximum_aspect = float(config.get("auto_board_max_aspect_ratio", 1.05))

    # 计算并保存到 open_size（开运算·尺寸）。
    open_size = max(3, int(round(width / 384.0)))
    # 更新变量：open_size（开运算·尺寸）。
    open_size += 1 - open_size % 2
    # 计算并保存到 close_size（闭运算·尺寸）。
    close_size = max(9, int(round(width / 91.0)))
    # 更新变量：close_size（闭运算·尺寸）。
    close_size += 1 - close_size % 2
    # 计算并保存到 open_kernel（开运算·结构元素）。
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    # 计算并保存到 close_kernel（闭运算·结构元素）。
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    # 计算并保存到 border_margin（边框·边距）。
    border_margin = max(2, int(round(min(width, height) * 0.003)))
    # 计算并保存到 candidates（待评估的候选集合）。
    candidates: list[tuple[float, np.ndarray]] = []

    # 遍历数据，逐项处理：value_thresholds。
    for raw_threshold in value_thresholds:
        # 计算并保存到 value_max（值·最大）。
        value_max = int(raw_threshold)
        # 计算并保存到 selected（当前选中的数据）。
        selected = ((hsv[:, :, 2] <= value_max) &
                    (hsv[:, :, 1] <= saturation_max))
        # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
        mask = np.where(selected, 255, 0).astype(np.uint8)
        # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        # 计算并保存到 contours（检测到的轮廓集合）、_（此处不需要使用的返回值或循环占位变量）。
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # 遍历数据，逐项处理：contours。
        for contour in contours:
            # 计算并保存到 contour_area（轮廓·面积）。
            contour_area = float(cv2.contourArea(contour))
            # 计算并保存到 area_ratio（面积·比例）。
            area_ratio = contour_area / image_area
            # 判断条件；满足时执行下面代码：not minimum_area_ratio <= area_ratio <= maximum_area_ratio。
            if not minimum_area_ratio <= area_ratio <= maximum_area_ratio:
                # 跳过本轮，进入下一轮循环。
                continue
            # 计算并保存到 x（当前横坐标或横向数据）、y（当前纵坐标或纵向数据）、box_width（box·宽度）、box_height（box·高度）。
            x, y, box_width, box_height = cv2.boundingRect(contour)
            # 判断条件；满足时执行下面代码：x <= border_margin or y <= border_margin or x + box_width >= width - border_mar…。
            if (x <= border_margin or y <= border_margin or
                    x + box_width >= width - border_margin or
                    y + box_height >= height - border_margin):
                # 跳过本轮，进入下一轮循环。
                continue

            # 计算并保存到 hull（包住轮廓的最小凸多边形）。
            hull = cv2.convexHull(contour)
            # 计算并保存到 perimeter（轮廓周长）。
            perimeter = float(cv2.arcLength(hull, True))
            # 遍历数据，逐项处理：(0.01, 0.015, 0.02, 0.025, 0.03, 0.04)。
            for epsilon_ratio in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
                # 计算并保存到 approximate。
                approximate = cv2.approxPolyDP(
                    hull, epsilon_ratio * perimeter, True
                )
                # 判断条件；满足时执行下面代码：len(approximate) != 4。
                if len(approximate) != 4:
                    # 跳过本轮，进入下一轮循环。
                    continue
                # 计算并保存到 points（参与当前计算的一组坐标点）。
                points = order_quad_points(
                    approximate.reshape(4, 2).astype(np.float32)
                )
                # 判断条件；满足时执行下面代码：not valid_quad(points)。
                if not valid_quad(points):
                    # 跳过本轮，进入下一轮循环。
                    continue

                # 计算并保存到 blocked。
                blocked = str(config.get("auto_board_fix_blocked_corner", ""))
                # 判断条件；满足时执行下面代码：blocked == 'tl'。
                if blocked == "tl":
                    # 计算并保存到 points[0]。
                    points[0] = points[1] + points[3] - points[2]
                # 判断条件；满足时执行下面代码：blocked == 'tr'。
                elif blocked == "tr":
                    # 计算并保存到 points[1]。
                    points[1] = points[0] + points[2] - points[3]
                # 判断条件；满足时执行下面代码：blocked == 'br'。
                elif blocked == "br":
                    # 计算并保存到 points[2]。
                    points[2] = points[1] + points[3] - points[0]
                # 判断条件；满足时执行下面代码：blocked == 'bl'。
                elif blocked == "bl":
                    # 计算并保存到 points[3]。
                    points[3] = points[0] + points[2] - points[1]

                # 计算并保存到 top_width（上边·宽度）。
                top_width = float(np.linalg.norm(points[1] - points[0]))
                # 计算并保存到 bottom_width（下边·宽度）。
                bottom_width = float(np.linalg.norm(points[2] - points[3]))
                # 计算并保存到 left_height（左边·高度）。
                left_height = float(np.linalg.norm(points[3] - points[0]))
                # 计算并保存到 right_height（右边·高度）。
                right_height = float(np.linalg.norm(points[2] - points[1]))
                # 计算并保存到 aspect。
                aspect = ((top_width + bottom_width) /
                          max(left_height + right_height, 1e-6))
                # 判断条件；满足时执行下面代码：not minimum_aspect <= aspect <= maximum_aspect。
                if not minimum_aspect <= aspect <= maximum_aspect:
                    # 跳过本轮，进入下一轮循环。
                    continue

                # 计算并保存到 board_horiz（底板·horiz）。
                board_horiz = max(bottom_width, top_width)
                # 判断条件；满足时执行下面代码：board_horiz > 1e-06。
                if board_horiz > 1e-6:
                    # 计算并保存到 right_skew（右边·skew）。
                    right_skew = abs(float(points[1][0] - points[2][0])) / board_horiz
                    # 计算并保存到 left_skew（左边·skew）。
                    left_skew = abs(float(points[0][0] - points[3][0])) / board_horiz
                    # 计算并保存到 max_skew（最大·skew）。
                    max_skew = float(config.get("auto_board_max_edge_skew", 0.06))
                    # 判断条件；满足时执行下面代码：max(right_skew, left_skew) > max_skew。
                    if max(right_skew, left_skew) > max_skew:
                        # 跳过本轮，进入下一轮循环。
                        continue

                # 计算并保存到 quad_area（quad·面积）。
                quad_area = abs(float(cv2.contourArea(points)))
                # 判断条件；满足时执行下面代码：quad_area <= 1.0。
                if quad_area <= 1.0:
                    # 跳过本轮，进入下一轮循环。
                    continue
                # 计算并保存到 interior。
                interior = np.zeros((height, width), np.uint8)
                # 调用函数：cv2.fillPoly。
                cv2.fillPoly(interior, [points.astype(np.int32)], 255)
                # 计算并保存到 interior。
                interior = cv2.erode(interior, close_kernel)
                # 计算并保存到 values（当前数值集合）。
                values = hsv[:, :, 2][interior > 0]
                # 判断条件；满足时执行下面代码：values.size == 0。
                if values.size == 0:
                    # 跳过本轮，进入下一轮循环。
                    continue
                # 计算并保存到 dark_fraction（深色·占比）。
                dark_fraction = float(np.mean(values < 150))
                # 计算并保存到 white_fraction（白色·占比）。
                white_fraction = float(np.mean(values > 220))
                # 判断条件；满足时执行下面代码：dark_fraction < minimum_dark_fraction or white_fraction > maximum_white_fraction。
                if (dark_fraction < minimum_dark_fraction or
                        white_fraction > maximum_white_fraction):
                    # 跳过本轮，进入下一轮循环。
                    continue

                # 计算并保存到 fill_ratio（填充·比例）。
                fill_ratio = min(contour_area / quad_area, 1.0)
                # 计算并保存到 center（本步骤使用的中心位置）。
                center = np.mean(points, axis=0)
                # 计算并保存到 center_distance（中心·距离）。
                center_distance = float(np.linalg.norm(
                    center - np.asarray([width / 2.0, height / 2.0])
                ) / math.hypot(width, height))
                # 计算并保存到 score（评分）。
                score = (quad_area / image_area + 0.5 * fill_ratio +
                         0.5 * dark_fraction - 0.5 * white_fraction -
                         0.25 * center_distance)
                # 调用函数：candidates.append。
                candidates.append((score, points))
                # 立即结束当前循环。
                break

    # 判断条件；满足时执行下面代码：not candidates。
    if not candidates:
        # 返回结果：None。
        return None
    # 返回结果：max(candidates, key=lambda item: item[0])[1]。
    return max(candidates, key=lambda item: item[0])[1]


# 【函数：resolve_calibration_matrix】优先按当前帧自动定位底板；失败时再使用已保存的手工标定。
# 参数 frame（摄像头的一帧原始图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 fallback_matrix（自动标定失败时使用的备用透视矩阵）：NumPy数组或None（不返回业务结果）；省略时使用None。
# 返回类型：元组（依次为NumPy数组、字符串）；箭头->是类型提示，不会替你转换实际返回值。
def resolve_calibration_matrix(
        frame: np.ndarray, config: dict[str, Any],
        fallback_matrix: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """优先按当前帧自动定位底板；失败时再使用已保存的手工标定。"""
    # 计算并保存到 points（参与当前计算的一组坐标点）。
    points = detect_dark_board_points(frame, config)
    # 判断条件；满足时执行下面代码：points is not None。
    if points is not None:
        # 计算并保存到 _（此处不需要使用的返回值或循环占位变量）、matrix（本函数使用的矩阵）。
        _, matrix = calibration_matrix(points, config)
        # 计算并保存到 rounded。
        rounded = np.rint(points).astype(int).tolist()
        # 调用函数：print。
        print(f"[自动标定] 已定位黑色底板四角：{rounded}")
        # 返回结果：创建数据容器。
        return matrix, "auto_board"

    # 计算并保存到 matrix（本函数使用的矩阵）。
    matrix = fallback_matrix if fallback_matrix is not None else load_calibration(config)
    # 判断条件；满足时执行下面代码：matrix is None。
    if matrix is None:
        # 抛出异常，通知上层处理：RuntimeError('未能自动定位黑色底板，且没有可用的calibration.json兜底标定')。
        raise RuntimeError("未能自动定位黑色底板，且没有可用的calibration.json兜底标定")
    # 调用函数：print。
    print("[WARN] 本帧未能自动定位黑色底板，已使用保存的标定")
    # 返回结果：创建数据容器。
    return matrix, "saved"

# 【函数：load_calibration】读取已有标定；无文件、区域太小或格式无效时返回None。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组或None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def load_calibration(config: dict[str, Any]) -> np.ndarray | None:
    # 判断条件；满足时执行下面代码：not CALIBRATION_PATH.exists()。
    if not CALIBRATION_PATH.exists():
        # 返回结果：None。
        return None
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 打开并自动管理资源，结束后自动释放。
        with CALIBRATION_PATH.open("r", encoding="utf-8-sig") as file:
            # 计算并保存到 payload（准备写入文件或发送的数据）。
            payload = json.load(file)
        # 判断条件；满足时执行下面代码：'image_points' in payload。
        if "image_points" in payload:
            # 计算并保存到 points（参与当前计算的一组坐标点）。
            points = np.asarray(payload["image_points"], np.float32)
            # 计算并保存到 image_size（图像尺寸，顺序为(宽,高)）。
            image_size = payload.get("image_size", [config["camera_width"], config["camera_height"]])
            # 计算并保存到 ratio（比例）。
            ratio = calibration_area_ratio(points, (int(image_size[0]), int(image_size[1])))
            # 计算并保存到 minimum（允许下限或计算出的最小值）。
            minimum = float(config.get("min_calibration_area_ratio", 0.06))
            # 判断条件；满足时执行下面代码：ratio < minimum。
            if ratio < minimum:
                # 调用函数：print。
                print(f"[WARN] 旧标定区域只占画面{ratio * 100:.1f}%，已拒绝加载。")
                # 调用函数：print。
                print("[WARN] 请点击A4纸的4个外角，不是4块粉色碎片。")
                # 返回结果：None。
                return None
            # 计算并保存到 _（此处不需要使用的返回值或循环占位变量）、matrix（本函数使用的矩阵）。
            _, matrix = calibration_matrix(points, config)
            # 返回结果：matrix（本函数使用的矩阵）。
            return matrix
        # 计算并保存到 matrix（本函数使用的矩阵）。
        matrix = np.asarray(payload["perspective_matrix"], np.float32)
        # 返回结果：matrix if matrix.shape == (3, 3) else None。
        return matrix if matrix.shape == (3, 3) else None
    # 捕获(OSError, KeyError, ValueError, json.JSONDecodeError, cv2.error)异常，转入下面的处理代码。
    except (OSError, KeyError, ValueError, json.JSONDecodeError, cv2.error):
        # 返回结果：None。
        return None


# =============================================================================
# 【分区】OpenCV 窗口与手工四点采集
# 功能：安全关窗；collect_four_points 让操作员按顺序点 A4 四角。
# 可修改：窗口名、提示文字。
# 看情况改：触摸屏点不准时用鼠标；点序必须左上→右上→右下→左下（与 order_quad_points 一致）。
# 不要改：窗口已关闭时吞掉 cv2.error，否则预览被关掉会把主流程打死。
# =============================================================================
# 【函数：safe_destroy_window】窗口已被用户或OpenCV后端关闭时，销毁操作不应导致主程序退出。
# 参数 window（OpenCV窗口名称）：字符串。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def safe_destroy_window(window: str) -> None:
    """窗口已被用户或OpenCV后端关闭时，销毁操作不应导致主程序退出。"""
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 调用函数：cv2.destroyWindow。
        cv2.destroyWindow(window)
    # 捕获cv2.error异常，转入下面的处理代码。
    except cv2.error:
        pass


# 【函数：window_is_visible】安全查询窗口状态；不存在的窗口统一视为已关闭。
# 参数 window（OpenCV窗口名称）：字符串。
# 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
def window_is_visible(window: str) -> bool:
    """安全查询窗口状态；不存在的窗口统一视为已关闭。"""
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 返回结果：判断条件：cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1。
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
    # 捕获cv2.error异常，转入下面的处理代码。
    except cv2.error:
        # 返回结果：False。
        return False


# 【函数：collect_four_points】点击黑色底板的四个外角；程序自动排序并在第四点后完成。
# 参数 frame（摄像头的一帧原始图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组或None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def collect_four_points(frame: np.ndarray, config: dict[str, Any]) -> np.ndarray | None:
    """点击黑色底板的四个外角；程序自动排序并在第四点后完成。"""
    # 计算并保存到 points（参与当前计算的一组坐标点）。
    points: list[tuple[int, int]] = []
    # 计算并保存到 window（OpenCV窗口名称）。
    window = "A4 Calibration"

    # 显式缩小显示图，避免WINDOW_NORMAL缩放后鼠标坐标与原始帧不一致。
    # 计算并保存到 scale（当前缩放比例或试探步长比例）。
    scale = min(1.0, 1200.0 / frame.shape[1], 680.0 / frame.shape[0])
    # 计算并保存到 display_size（display·尺寸）。
    display_size = (max(1, int(round(frame.shape[1] * scale))),
                    max(1, int(round(frame.shape[0] * scale))))
    # 计算并保存到 base。
    base = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()

    # 【函数：on_mouse】接收OpenCV鼠标事件，根据点击位置更新交互状态。
    # 参数 event：整数。
    # 参数 x（当前横坐标或横向数据）：整数。
    # 参数 y（当前纵坐标或纵向数据）：整数。
    # 参数 flags：整数。
    # 参数 param：object。
    # 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        # 判断条件；满足时执行下面代码：event == cv2.EVENT_LBUTTONDOWN and len(points) < 4。
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            # 调用函数：points.append。
            points.append((x, y))
        # 判断条件；满足时执行下面代码：event == cv2.EVENT_RBUTTONDOWN。
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 调用函数：points.clear。
            points.clear()

    # 调用函数：cv2.namedWindow。
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # 判断条件；满足时执行下面代码：hasattr(cv2, 'WND_PROP_FULLSCREEN')。
    if hasattr(cv2, "WND_PROP_FULLSCREEN"):
        # 执行可能出错的代码，并由后面的异常分支处理错误。
        try:
            # 调用函数：cv2.setWindowProperty。
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # 捕获cv2.error异常，转入下面的处理代码。
        except cv2.error:
            pass
    # 调用函数：cv2.setMouseCallback。
    cv2.setMouseCallback(window, on_mouse)
    # 调用函数：print。
    print("[标定] 鼠标左键点击黑色底板四个外角，推荐：左上、右上、右下、左下。")
    # 调用函数：print。
    print("[标定] 第4点后自动保存；点错可右键/R清空，Esc取消。")

    # 只要条件成立就重复执行：True。
    while True:
        # 计算并保存到 display（用于窗口显示的画布）。
        display = base.copy()
        # 遍历数据，逐项处理：enumerate(points)。
        for index, point in enumerate(points):
            # 调用函数：cv2.circle。
            cv2.circle(display, point, 7, (0, 0, 255), -1)
            # 调用函数：cv2.putText。
            cv2.putText(display, str(index + 1), (point[0] + 10, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # 判断条件；满足时执行下面代码：len(points) >= 2。
        if len(points) >= 2:
            # 调用函数：cv2.polylines。
            cv2.polylines(display, [np.asarray(points, np.int32)], len(points) == 4,
                          (0, 255, 255), 2)
        # 计算并保存到 next_text（下一项·文本）。
        next_text = f"CORNER {len(points) + 1}" if len(points) < 4 else "CALIBRATION OK"
        # 调用函数：cv2.rectangle。
        cv2.rectangle(display, (0, 0), (display.shape[1], 48), (0, 0, 0), -1)
        # 调用函数：cv2.putText。
        cv2.putText(display,
                    f"Click {next_text} | Right click: reset | Esc: cancel",
                    (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.64,
                    (0, 255, 255), 2, cv2.LINE_AA)
        # 调用函数：cv2.imshow。
        cv2.imshow(window, display)

        # 判断条件；满足时执行下面代码：len(points) == 4。
        if len(points) == 4:
            # 调用函数：cv2.waitKey。
            cv2.waitKey(250)
            # 立即结束当前循环。
            break

        # 计算并保存到 key（查询、分组或排序所用的键）。
        key = cv2.waitKey(20) & 0xFF
        # 判断条件；满足时执行下面代码：key == 27 or not window_is_visible(window)。
        if key == 27 or not window_is_visible(window):
            # 调用函数：safe_destroy_window。
            safe_destroy_window(window)
            # 返回结果：None。
            return None
        # 判断条件；满足时执行下面代码：key in (ord('r'), ord('R'))。
        if key in (ord("r"), ord("R")):
            # 调用函数：points.clear。
            points.clear()

    # 调用函数：safe_destroy_window。
    safe_destroy_window(window)
    # 计算并保存到 clicked。
    clicked = np.asarray(points, np.float32) / scale
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 计算并保存到 source（源数据或源位置）、matrix（本函数使用的矩阵）。
        source, matrix = calibration_matrix(clicked, config)
    # 捕获ValueError异常，转入下面的处理代码；异常对象保存在exc。
    except ValueError as exc:
        # 调用函数：print。
        print(f"[ERR] 标定失败：{exc}，请重新标定")
        # 返回结果：None。
        return None

    # 计算并保存到 ratio（比例）。
    ratio = calibration_area_ratio(source, (frame.shape[1], frame.shape[0]))
    # 计算并保存到 minimum（允许下限或计算出的最小值）。
    minimum = float(config.get("min_calibration_area_ratio", 0.06))
    # 判断条件；满足时执行下面代码：ratio < minimum。
    if ratio < minimum:
        # 调用函数：print。
        print(f"[ERR] 你点击的区域只占画面{ratio * 100:.1f}%，这不是完整A4纸。")
        # 调用函数：print。
        print("[ERR] 请点击A4纸的左上、右上、右下、左下4个外角，不要点碎片。")
        # 返回结果：None。
        return None

    # 调用函数：save_calibration。
    save_calibration(source, matrix, (frame.shape[1], frame.shape[0]))
    # 调用函数：OUTPUT_DIR.mkdir。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 调用函数：write_image。
    write_image(OUTPUT_DIR / "calibration_preview.jpg", warp_paper(frame, matrix, config))
    # 调用函数：print。
    print("[OK] 四点已自动排序为：左上、右上、右下、左下")
    # 调用函数：print。
    print("[OK] 已生成output/calibration_preview.jpg，可用于检查底板是否完整")
    # 调用函数：print。
    print(f"[OK] 标定已自动保存：{CALIBRATION_PATH}")
    # 返回结果：matrix（本函数使用的矩阵）。
    return matrix


# =============================================================================
# 【分区】透视拉正纸面
# 功能：用标定矩阵把相机图 warp 成固定像素尺寸的正 A4 俯视图，后续检测都在这张图上做。
# 可修改：输出宽高由 paper_width_mm × pixels_per_mm 决定，改 config 即可。
# 看情况改：标定不准时不要调分割阈值硬凑，先重标四角。
# 不要改：必须用 warpPerspective；插值默认双线性，二值掩膜另走 INTER_NEAREST。
# =============================================================================
# 【函数：warp_paper】按透视矩阵将相机图像拉正为规定尺寸的A4俯视图。
# 参数 frame（摄像头的一帧原始图像）：NumPy数组。
# 参数 matrix（本函数使用的矩阵）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def warp_paper(frame: np.ndarray, matrix: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    # 返回结果：按透视矩阵重采样图像到给定(宽,高)。
    return cv2.warpPerspective(frame, matrix, paper_size_px(config))


# =============================================================================
# 【分区】摄像头打开与占位设备过滤
# 功能：扫描相机 index，丢掉 ToDesk/虚拟占位画面，打开真实俯拍 USB 摄像头。
# 可修改：扫描上限、占位图判定；命令行 --camera 指定编号。
# 看情况改：笔记本多摄像头时默认 0 可能是前置，用 --camera 1。远程桌面常插入虚拟摄像头，必须过滤。
# 不要改：打开失败要抛 RuntimeError；不要静默退回演示图，比赛时会误以为在拍真纸。
# =============================================================================
# 【函数：_camera_probe_is_placeholder】识别ToDesk等近乎全白的虚拟摄像头占位画面。
# 参数 frame（摄像头的一帧原始图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
def _camera_probe_is_placeholder(frame: np.ndarray, config: dict[str, Any]) -> bool:
    """识别ToDesk等近乎全白的虚拟摄像头占位画面。"""
    # 计算并保存到 gray（灰度）。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 计算并保存到 white_threshold（白色·阈值）。
    white_threshold = int(config.get("camera_placeholder_white_threshold", 245))
    # 计算并保存到 max_white_fraction（最大·白色·占比）。
    max_white_fraction = float(
        config.get("camera_placeholder_max_white_fraction", 0.90)
    )
    # 计算并保存到 white_fraction（白色·占比）。
    white_fraction = float(np.mean(gray >= white_threshold))
    # 返回结果：判断条件：white_fraction >= max_white_fraction。
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
    frame = _probe_opened_capture(config, capture)
    if frame is None and os.name != "nt":
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        frame = _probe_opened_capture(config, capture)
    if frame is None:
        capture.release()
        return None, None
    return capture, frame


# 【函数：_try_open_camera_index】尝试打开并预读一个摄像头；失败时完整释放资源。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 index（当前元素索引）：整数。
# 返回类型：元组（依次为cv2.VideoCapture或None（不返回业务结果）、NumPy数组或None（不返回业务结果））；箭头->是类型提示，不会替你转换实际返回值。
def _try_open_camera_index(
        config: dict[str, Any], index: int) -> tuple[cv2.VideoCapture | None, np.ndarray | None]:
    """尝试打开并预读一个摄像头；失败时完整释放资源。"""
    return _try_open_camera_source(config, index)


# 【函数：open_camera】打开首选摄像头；编号漂移时自动扫描并拒绝虚拟占位画面。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 camera_id（摄像头设备编号）：整数或None（不返回业务结果）。
# 返回类型：cv2.VideoCapture；箭头->是类型提示，不会替你转换实际返回值。
def open_camera(config: dict[str, Any], camera_id: int | None) -> cv2.VideoCapture:
    """打开首选摄像头；编号漂移时自动扫描并拒绝虚拟占位画面。"""
    # 计算并保存到 preferred。
    preferred = int(config["camera_id"] if camera_id is None else camera_id)
    # 计算并保存到 max_index（最大·索引）。
    max_index = max(preferred, int(config.get("camera_scan_max_index", 4)))
    # 计算并保存到 candidates（待评估的候选集合）。
    candidates: list[str | int] = []
    for item in [*_usb_camera_targets(), preferred, *range(max_index + 1)]:
        if item not in candidates:
            candidates.append(item)
    # 计算并保存到 placeholder_indices。
    placeholder_indices: list[str | int] = []

    # 遍历数据，逐项处理：candidates。
    for index in candidates:
        # 计算并保存到 capture（已打开的摄像头对象）、probe。
        capture, probe = _try_open_camera_source(config, index)
        # 判断条件；满足时执行下面代码：capture is None or probe is None。
        if capture is None or probe is None:
            # 跳过本轮，进入下一轮循环。
            continue
        # 判断条件；满足时执行下面代码：_camera_probe_is_placeholder(probe, config)。
        if _camera_probe_is_placeholder(probe, config):
            # 调用函数：placeholder_indices.append。
            placeholder_indices.append(index)
            # 调用函数：capture.release。
            capture.release()
            # 调用函数：print。
            print(f"[WARN] 摄像头{index}为近乎全白的虚拟/占位画面，已拒绝")
            # 跳过本轮，进入下一轮循环。
            continue

        # 计算并保存到 width（当前区域宽度）。
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 计算并保存到 height（当前区域高度）。
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 判断条件；满足时执行下面代码：index != preferred。
        if index != preferred:
            # 调用函数：print。
            print(f"[WARN] 首选摄像头index={preferred}不可用，已自动切换到{index}")
        # 调用函数：print。
        print(f"[OK] 摄像头已打开：{index}, {width}x{height}")
        # 返回结果：capture（已打开的摄像头对象）。
        return capture

    # 计算并保存到 placeholder_detail。
    placeholder_detail = (
        f"；检测到占位画面={placeholder_indices}"
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        if placeholder_indices else ""
    )
    # 抛出异常，通知上层处理：RuntimeError(f'无法找到真实摄像头（已扫描index=0~{max_index}）{placeholder_detail}；…。
    raise RuntimeError(
        f"无法找到真实摄像头（已扫描USB节点和index=0~{max_index}）{placeholder_detail}；"
        "请重新插拔俯拍USB摄像头或恢复ToDesk摄像头映射"
    )

# =============================================================================
# 【分区】碎片识别（掩膜 → 轮廓 → Piece）
# 功能：按配置选粉色/白片/扑克/背景反色等掩膜，形态学清理后提轮廓，算质心、吸取点、角度、面积。
# 可修改：HSV 粉阈值、面积/实心度、形态学核、max_pieces——全部优先改 config.json。
# 看情况改：白片用 make_white_piece_mask，粉片用 make_pink_mask，扑克用 make_poker_v_mask。灯光变了先调阈值，不要改轮廓几何。
# 不要改：detect_pieces 的像素/毫米换算；pick 点算法（磁铁实际吸这里）。坐标仍是纸面左上原点。
# =============================================================================
# ------------------------- 碎片识别 -------------------------

# 【函数：make_pink_mask】直接按HSV色相提取粉色，不再依赖纸张边缘背景估计。
# 参数 image（图像数组）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 kernel_px（结构元素尺寸（像素））：整数。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def make_pink_mask(image: np.ndarray, config: dict[str, Any],
                   kernel_px: int) -> np.ndarray:
    """直接按HSV色相提取粉色，不再依赖纸张边缘背景估计。"""
    # 计算并保存到 hsv（色相H、饱和度S、亮度V组成的图像）。
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 计算并保存到 hue（色相）。
    hue = hsv[:, :, 0]
    # 计算并保存到 saturation（饱和度）。
    saturation = hsv[:, :, 1]
    # 计算并保存到 value（当前数值）。
    value = hsv[:, :, 2]
    # 计算并保存到 hue_min（色相·最小）。
    hue_min = int(config.get("pink_hue_min", 140))
    # 计算并保存到 hue_max（色相·最大）。
    hue_max = int(config.get("pink_hue_max", 179))
    # 计算并保存到 saturation_min（饱和度·最小）。
    saturation_min = int(config.get("pink_saturation_min", 30))
    # 计算并保存到 value_min（值·最小）。
    value_min = int(config.get("pink_value_min", 100))
    # 判断条件；满足时执行下面代码：hue_min <= hue_max。
    if hue_min <= hue_max:
        # 计算并保存到 hue_ok（色相·ok）。
        hue_ok = (hue >= hue_min) & (hue <= hue_max)
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 计算并保存到 hue_ok（色相·ok）。
        hue_ok = (hue >= hue_min) | (hue <= hue_max)
    # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
    mask = np.where(hue_ok & (saturation >= saturation_min) & (value >= value_min),
                    255, 0).astype(np.uint8)
    # 计算并保存到 size（尺寸数据）。
    size = max(1, int(kernel_px))
    # 更新变量：size（尺寸数据）。
    size += 1 - size % 2
    # 计算并保存到 kernel（形态学处理所用结构元素）。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 返回结果：用结构元素做形态学处理：开运算去小噪点，闭运算补小孔隙。
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# 【函数：clean_piece_mask】对校正纸面掩膜执行统一形态学和可配置源区域限制。
# 参数 mask（二值掩膜（0为背景，非零为选中区域））：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def clean_piece_mask(mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """对校正纸面掩膜执行统一形态学和可配置源区域限制。"""
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 kernel_px（结构元素尺寸（像素））。
    kernel_px = max(3, int(round(float(config["morphology_kernel_mm"]) * ppm)))
    # 更新变量：kernel_px（结构元素尺寸（像素））。
    kernel_px += 1 - kernel_px % 2
    # 计算并保存到 kernel（形态学处理所用结构元素）。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    # 计算并保存到 cleaned。
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    # 计算并保存到 cleaned。
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    # 计算并保存到 source_bottom（源·下边）。
    source_bottom = int(round(float(config["source_region_bottom_mm"]) * ppm))
    # 计算并保存到 source_bottom（源·下边）。
    source_bottom = min(max(source_bottom, 1), cleaned.shape[0])
    # 计算并保存到 cleaned[source_bottom:]。
    cleaned[source_bottom:] = 0

    """赛题要求黑色A4中间横线分割上下区域：把分界线附近的窄带整体清零，
    避免任意颜色的实线被分割算法当成碎片或粘连碎片。线位置默认取
    target_region_top_mm，也可用 separator_line_y_mm 单独指定。"""
    # 计算并保存到 line_y（line·Y轴）。
    line_y = float(config.get("separator_line_y_mm",
                               config.get("target_region_top_mm", 148.5)))
    # 计算并保存到 half。
    half = float(config.get("separator_line_half_width_mm", 3.0))
    # 判断条件；满足时执行下面代码：line_y > 0.0。
    if line_y > 0.0:
        # 计算并保存到 y0。
        y0 = int(round((line_y - half) * ppm))
        # 计算并保存到 y1。
        y1 = int(round((line_y + half) * ppm))
        # 计算并保存到 y0。
        y0 = max(y0, 0)
        # 计算并保存到 y1。
        y1 = min(y1, cleaned.shape[0])
        # 判断条件；满足时执行下面代码：y1 > y0。
        if y1 > y0:
            # 计算并保存到 cleaned[y0:y1]。
            cleaned[y0:y1] = 0
    # 返回结果：cleaned。
    return cleaned


# 【函数：make_background_inverse_mask】从纸面边缘估计背景颜色，提取与背景颜色差异明显的碎片。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def make_background_inverse_mask(paper: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """从纸面边缘估计背景颜色，提取与背景颜色差异明显的碎片。"""
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 border（边框）。
    border = max(1, int(round(float(config.get("border_sample_mm", 5.0)) * ppm)))
    # 计算并保存到 border（边框）。
    border = min(border, max(1, min(paper.shape[:2]) // 4))
    # 计算并保存到 samples（标定采样记录列表）。
    samples = np.concatenate((
        paper[:border].reshape(-1, 3), paper[-border:].reshape(-1, 3),
        paper[:, :border].reshape(-1, 3), paper[:, -border:].reshape(-1, 3),
    # 传入命名参数：axis=0（指定运算维度：0通常沿行汇总，1沿列汇总）；取值过程：0。
    ), axis=0)
    # 计算并保存到 background_bgr（background·BGR颜色）。
    background_bgr = np.median(samples, axis=0).astype(np.uint8).reshape(1, 1, 3)
    # 计算并保存到 paper_lab（纸面·Lab颜色）。
    paper_lab = cv2.cvtColor(paper, cv2.COLOR_BGR2LAB).astype(np.float32)
    # 计算并保存到 background_lab（background·Lab颜色）。
    background_lab = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    # 计算并保存到 distance（当前距离结果）。
    distance = np.linalg.norm(paper_lab - background_lab, axis=2)
    # 计算并保存到 threshold（阈值）。
    threshold = float(config.get("background_distance_threshold", 18.0))
    # 返回结果：调用 np.where(distance >= threshold, 255, 0).astype。
    return np.where(distance >= threshold, 255, 0).astype(np.uint8)


# 【函数：make_white_piece_mask】提取低饱和度高亮白片；亮度阈值支持固定值或Otsu自适应。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def make_white_piece_mask(paper: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """提取低饱和度高亮白片；亮度阈值支持固定值或Otsu自适应。"""
    # 计算并保存到 hsv（色相H、饱和度S、亮度V组成的图像）。
    hsv = cv2.cvtColor(paper, cv2.COLOR_BGR2HSV)
    # 计算并保存到 gray（灰度）。
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    # 计算并保存到 saturation_max（饱和度·最大）。
    saturation_max = int(config.get("white_piece_saturation_max", 55))
    # 计算并保存到 threshold_mode（阈值·模式）。
    threshold_mode = str(config.get("white_piece_threshold_mode", "fixed"))

    # 判断条件；满足时执行下面代码：threshold_mode == 'otsu'。
    if threshold_mode == "otsu":
        # 计算并保存到 ppm（每毫米对应的像素数）。
        ppm = float(config["pixels_per_mm"])
        # 计算并保存到 source_bottom（源·下边）。
        source_bottom = int(round(float(config["source_region_bottom_mm"]) * ppm))
        # 计算并保存到 source_bottom（源·下边）。
        source_bottom = min(max(source_bottom, 1), gray.shape[0])
        # 计算并保存到 otsu_threshold（otsu·阈值）、_（此处不需要使用的返回值或循环占位变量）。
        otsu_threshold, _ = cv2.threshold(
            gray[:source_bottom], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # 计算并保存到 minimum（允许下限或计算出的最小值）。
        minimum = int(config.get("white_piece_otsu_min_value", 120))
        # 计算并保存到 maximum（允许上限或计算出的最大值）。
        maximum = int(config.get("white_piece_otsu_max_value", 230))
        # 判断条件；满足时执行下面代码：not 0 <= minimum <= maximum <= 255。
        if not 0 <= minimum <= maximum <= 255:
            # 抛出异常，通知上层处理：ValueError('white_piece Otsu阈值范围必须满足0<=min<=max<=255')。
            raise ValueError("white_piece Otsu阈值范围必须满足0<=min<=max<=255")
        # 计算并保存到 brightness_min（brightness·最小）。
        brightness_min = int(round(np.clip(otsu_threshold, minimum, maximum)))
    # 判断条件；满足时执行下面代码：threshold_mode == 'fixed'。
    elif threshold_mode == "fixed":
        # 计算并保存到 brightness_min（brightness·最小）。
        brightness_min = int(config.get("white_piece_value_min", 245))
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 抛出异常，通知上层处理：ValueError(f'未知white_piece_threshold_mode：{threshold_mode}')。
        raise ValueError(f"未知white_piece_threshold_mode：{threshold_mode}")

    # 计算并保存到 selected（当前选中的数据）。
    selected = ((hsv[:, :, 1] <= saturation_max) & (gray >= brightness_min))
    # 返回结果：调用 np.where(selected, 255, 0).astype。
    return np.where(selected, 255, 0).astype(np.uint8)


# 【函数：make_poker_v_mask】扑克牌白底碎片分割：V通道二值化 + 大核闭运算填充牌面花纹。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def make_poker_v_mask(paper: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """扑克牌白底碎片分割：V通道二值化 + 大核闭运算填充牌面花纹。

    扑克牌碎片是白色卡纸，牌面上印有红心/黑桃等花纹；直接用白片亮度阈值
    会把花纹洞当成碎片内部空洞，因此先按V通道二值化提取卡面，再用大核
    闭运算把花纹洞填平，保证轮廓完整。
    """
    # 计算并保存到 hsv（色相H、饱和度S、亮度V组成的图像）。
    hsv = cv2.cvtColor(paper, cv2.COLOR_BGR2HSV)
    # 计算并保存到 value（当前数值）。
    value = hsv[:, :, 2]
    # 计算并保存到 threshold_mode（阈值·模式）。
    threshold_mode = str(config.get("poker_threshold_mode", "otsu"))

    # 判断条件；满足时执行下面代码：threshold_mode == 'otsu'。
    if threshold_mode == "otsu":
        # 计算并保存到 otsu_threshold（otsu·阈值）、_（此处不需要使用的返回值或循环占位变量）。
        otsu_threshold, _ = cv2.threshold(
            value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # 计算并保存到 minimum（允许下限或计算出的最小值）。
        minimum = int(config.get("poker_otsu_min_value", 80))
        # 计算并保存到 maximum（允许上限或计算出的最大值）。
        maximum = int(config.get("poker_otsu_max_value", 255))
        # 判断条件；满足时执行下面代码：not 0 <= minimum <= maximum <= 255。
        if not 0 <= minimum <= maximum <= 255:
            # 抛出异常，通知上层处理：ValueError('poker Otsu阈值范围必须满足0<=min<=max<=255')。
            raise ValueError("poker Otsu阈值范围必须满足0<=min<=max<=255")
        # 计算并保存到 brightness_min（brightness·最小）。
        brightness_min = int(round(np.clip(otsu_threshold, minimum, maximum)))
        # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
        mask = cv2.threshold(value, brightness_min, 255, cv2.THRESH_BINARY)[1]
    # 判断条件；满足时执行下面代码：threshold_mode == 'adaptive'。
    elif threshold_mode == "adaptive":
        # 计算并保存到 block。
        block = max(15, min(99, (min(paper.shape[:2]) // 16) | 1))
        # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
        mask = cv2.adaptiveThreshold(
            value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block,
            int(config.get("poker_adaptive_c", 2)),
        )
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 计算并保存到 brightness_min（brightness·最小）。
        brightness_min = int(config.get("poker_value_min", 90))
        # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
        mask = cv2.threshold(value, brightness_min, 255, cv2.THRESH_BINARY)[1]

    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 close_mm（闭运算·毫米）。
    close_mm = float(config.get("poker_morph_close_mm", 2.5))
    # 计算并保存到 kernel_px（结构元素尺寸（像素））。
    kernel_px = max(3, int(round(close_mm * ppm)))
    # 更新变量：kernel_px（结构元素尺寸（像素））。
    kernel_px += 1 - kernel_px % 2
    # 计算并保存到 kernel（形态学处理所用结构元素）。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    # 返回结果：用结构元素做形态学处理：开运算去小噪点，闭运算补小孔隙。
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# 【函数：calibration_edge_color_distances】比较校正图四边与下半区背景颜色，用于发现标定漂移和画面裁切。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：字典（键为字符串，值为浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def calibration_edge_color_distances(
        paper: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    """比较校正图四边与下半区背景颜色，用于发现标定漂移和画面裁切。"""
    # 判断条件；满足时执行下面代码：paper.ndim != 3 or paper.shape[0] < 8 or paper.shape[1] < 8。
    if paper.ndim != 3 or paper.shape[0] < 8 or paper.shape[1] < 8:
        # 返回结果：创建结果字典。
        return {"top": math.inf, "bottom": math.inf,
                # 字典字段'left'（左边）：math.inf；字典字段'right'（右边）：math.inf。
                "left": math.inf, "right": math.inf}

    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 band（带状区域）。
    band = max(4, int(round(float(config.get("calibration_edge_band_mm", 3.0)) * ppm)))
    # 计算并保存到 height（当前区域高度）、width（当前区域宽度）。
    height, width = paper.shape[:2]
    # 计算并保存到 band（带状区域）。
    band = min(band, max(1, min(height, width) // 8))
    # 计算并保存到 lab（Lab颜色空间中的图像）。
    lab = cv2.cvtColor(paper, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 计算并保存到 reference。
    reference = lab[
        # 本行与前后行共同构成完整表达式。
        int(round(height * 0.55)):max(int(round(height * 0.85)), int(round(height * 0.55)) + 1),
        # 本行与前后行共同构成完整表达式。
        int(round(width * 0.20)):max(int(round(width * 0.80)), int(round(width * 0.20)) + 1),
    ]
    # 计算并保存到 reference_color（reference·颜色）。
    reference_color = np.median(reference.reshape(-1, 3), axis=0)
    # 计算并保存到 vertical_start。
    vertical_start = int(round(height * 0.15))
    # 计算并保存到 vertical_end。
    vertical_end = max(int(round(height * 0.85)), vertical_start + 1)
    # 计算并保存到 horizontal_start。
    horizontal_start = int(round(width * 0.15))
    # 计算并保存到 horizontal_end。
    horizontal_end = max(int(round(width * 0.85)), horizontal_start + 1)
    # 计算并保存到 edges（边集合）。
    edges = {
        # 字典字段'top'（上边）：lab[:band, horizontal_start:horizontal_end]。
        "top": lab[:band, horizontal_start:horizontal_end],
        # 字典字段'bottom'（下边）：lab[-band:, horizontal_start:horizontal_end]。
        "bottom": lab[-band:, horizontal_start:horizontal_end],
        # 字典字段'left'（左边）：lab[vertical_start:vertical_end, :band]。
        "left": lab[vertical_start:vertical_end, :band],
        # 字典字段'right'（右边）：lab[vertical_start:vertical_end, -band:]。
        "right": lab[vertical_start:vertical_end, -band:],
    }
    # 返回结果：{side: float(np.linalg.norm(np.median(values.reshape(-1, 3), axis=0) - reference_col…。
    return {
        # 传入命名参数：axis=0（指定运算维度：0通常沿行汇总，1沿列汇总）；取值过程：0。
        side: float(np.linalg.norm(np.median(values.reshape(-1, 3), axis=0) - reference_color))
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for side, values in edges.items()
    }


# 【函数：calibration_quality_reasons】返回可读的标定质量问题；当前仅对白片深色底板模式执行颜色一致性检查。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：字符串列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def calibration_quality_reasons(paper: np.ndarray,
                                config: dict[str, Any]) -> list[str]:
    """返回可读的标定质量问题；当前仅对白片深色底板模式执行颜色一致性检查。"""
    # 判断条件；满足时执行下面代码：str(config.get('segmentation_mode', 'pink_hsv')) != 'white_piece'。
    if str(config.get("segmentation_mode", "pink_hsv")) != "white_piece":
        # 返回结果：创建数据容器。
        return []
    # 计算并保存到 maximum（允许上限或计算出的最大值）。
    maximum = float(config.get("max_calibration_edge_color_distance", 50.0))
    # 计算并保存到 names。
    names = {"top": "上", "bottom": "下", "left": "左", "right": "右"}
    # 计算并保存到 distances。
    distances = calibration_edge_color_distances(paper, config)
    # 计算并保存到 invalid。
    invalid = [f"{names[side]}边色差{distance:.1f}"
               # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
               for side, distance in distances.items() if distance > maximum]
    # 判断条件；满足时执行下面代码：not invalid。
    if not invalid:
        # 返回结果：创建数据容器。
        return []
    # 返回结果：创建数据容器。
    return [f"校正图边缘混入白边/桌面（{', '.join(invalid)}，阈值{maximum:.1f}）"]


# 【函数：piece_border_contacts】找出接触校正图边界的碎片；轮廓被裁切时禁止继续求解。
# 参数 pieces（碎片列表）：Piece列表/序列。
# 参数 paper_shape（纸面图像形状，顺序为(高,宽)）：元组（依次为整数、整数）。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：字符串列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def piece_border_contacts(pieces: list[Piece], paper_shape: tuple[int, int],
                          config: dict[str, Any]) -> list[str]:
    """找出接触校正图边界的碎片；轮廓被裁切时禁止继续求解。"""
    # 计算并保存到 height（当前区域高度）、width（当前区域宽度）。
    height, width = paper_shape
    # 计算并保存到 margin（边缘预留量）。
    margin = max(1, int(round(float(config.get("piece_border_margin_mm", 2.0)) *
                              float(config["pixels_per_mm"]))))
    # 计算并保存到 contacts。
    contacts: list[str] = []
    # 遍历数据，逐项处理：pieces。
    for piece in pieces:
        # 计算并保存到 x（当前横坐标或横向数据）、y（当前纵坐标或纵向数据）、piece_width（碎片·宽度）、piece_height（碎片·高度）。
        x, y, piece_width, piece_height = cv2.boundingRect(piece.contour.astype(np.int32))
        # 计算并保存到 sides。
        sides: list[str] = []
        # 判断条件；满足时执行下面代码：x <= margin。
        if x <= margin:
            # 调用函数：sides.append。
            sides.append("左")
        # 判断条件；满足时执行下面代码：y <= margin。
        if y <= margin:
            # 调用函数：sides.append。
            sides.append("上")
        # 判断条件；满足时执行下面代码：x + piece_width >= width - margin。
        if x + piece_width >= width - margin:
            # 调用函数：sides.append。
            sides.append("右")
        # 判断条件；满足时执行下面代码：y + piece_height >= height - margin。
        if y + piece_height >= height - margin:
            # 调用函数：sides.append。
            sides.append("下")
        # 判断条件；满足时执行下面代码：sides。
        if sides:
            # 调用函数：contacts.append。
            contacts.append(f"P{piece.piece_id}接触{'/'.join(sides)}边")
    # 返回结果：contacts。
    return contacts


# 【函数：make_piece_mask】按segmentation_mode识别校正纸面中的碎片。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 diagnostics（是否输出诊断信息）：布尔值True/False；省略时使用False。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def make_piece_mask(paper: np.ndarray, config: dict[str, Any],
                    diagnostics: bool = False) -> np.ndarray:
    """按segmentation_mode识别校正纸面中的碎片。"""
    # 计算并保存到 mode（模式）。
    mode = str(config.get("segmentation_mode", "pink_hsv"))
    # 判断条件；满足时执行下面代码：mode == 'pink_hsv'。
    if mode == "pink_hsv":
        # 计算并保存到 ppm（每毫米对应的像素数）。
        ppm = float(config["pixels_per_mm"])
        # 计算并保存到 source_bottom（源·下边）。
        source_bottom = int(round(float(config["source_region_bottom_mm"]) * ppm))
        # 计算并保存到 source_bottom（源·下边）。
        source_bottom = min(max(source_bottom, 1), paper.shape[0])
        # 计算并保存到 kernel_px（结构元素尺寸（像素））。
        kernel_px = max(3, int(round(float(config["morphology_kernel_mm"]) * ppm)))
        # 保留原有粉色路径，保证默认模式的掩膜行为不变。
        # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
        mask = make_pink_mask(paper, config, kernel_px)
        # 计算并保存到 mask[source_bottom:]。
        mask[source_bottom:] = 0
        # 计算并保存到 detail。
        detail = (f"H={config.get('pink_hue_min', 140)}~"
                  f"{config.get('pink_hue_max', 179)}, "
                  f"S>={config.get('pink_saturation_min', 30)}, "
                  f"V>={config.get('pink_value_min', 100)}")
    # 判断条件；满足时执行下面代码：mode == 'background_inverse'。
    elif mode == "background_inverse":
        # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
        mask = clean_piece_mask(make_background_inverse_mask(paper, config), config)
        # 计算并保存到 detail。
        detail = f"Lab颜色距离>={config.get('background_distance_threshold', 18.0)}"
    # 判断条件；满足时执行下面代码：mode == 'white_piece'。
    elif mode == "white_piece":
        # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
        mask = clean_piece_mask(make_white_piece_mask(paper, config), config)
        # 计算并保存到 threshold_mode（阈值·模式）。
        threshold_mode = str(config.get("white_piece_threshold_mode", "fixed"))
        # 计算并保存到 detail。
        detail = (f"S<={config.get('white_piece_saturation_max', 55)}, "
                  f"亮度阈值模式={threshold_mode}")
    # 判断条件；满足时执行下面代码：mode == 'poker_v'。
    elif mode == "poker_v":
        # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
        mask = clean_piece_mask(make_poker_v_mask(paper, config), config)
        # 计算并保存到 close_mm（闭运算·毫米）。
        close_mm = float(config.get("poker_morph_close_mm", 2.5))
        # 计算并保存到 detail。
        detail = (f"V通道阈值模式={config.get('poker_threshold_mode', 'otsu')}, "
                  f"闭运算填花纹={close_mm:.1f}mm")
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 抛出异常，通知上层处理：ValueError(f'未知segmentation_mode：{mode}')。
        raise ValueError(f"未知segmentation_mode：{mode}")
    # 判断条件；满足时执行下面代码：diagnostics。
    if diagnostics:
        # 计算并保存到 selected_ratio（选中·比例）。
        selected_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        # 调用函数：print。
        print(f"[诊断] 分割模式={mode}: {detail}, 白色像素占比={selected_ratio * 100:.2f}%")
        # 判断条件；满足时执行下面代码：mode == 'white_piece' and selected_ratio > 0.5。
        if mode == "white_piece" and selected_ratio > 0.50:
            # 调用函数：print。
            print("[WARN] 白片掩膜超过画面50%，深色底板也被选中了；请提高white_piece_value_min或降低曝光。")
        # 判断条件；满足时执行下面代码：mode == 'poker_v' and selected_ratio > 0.5。
        if mode == "poker_v" and selected_ratio > 0.50:
            # 调用函数：print。
            print("[WARN] 扑克牌掩膜超过画面50%，深色底板也被选中了；请调整poker_otsu_min_value或降低曝光。")
    # 返回结果：mask（二值掩膜（0为背景，非零为选中区域））。
    return mask


# 【函数：count_piece_candidates】按正式检测的面积和实心度规则统计有效碎片轮廓。
# 参数 mask（二值掩膜（0为背景，非零为选中区域））：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：整数；箭头->是类型提示，不会替你转换实际返回值。
def count_piece_candidates(mask: np.ndarray, config: dict[str, Any]) -> int:
    """按正式检测的面积和实心度规则统计有效碎片轮廓。"""
    # 计算并保存到 contours（检测到的轮廓集合）、_（此处不需要使用的返回值或循环占位变量）。
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 minimum_area（最小·面积）。
    minimum_area = float(config["min_piece_area_mm2"])
    # 计算并保存到 maximum_area（最大·面积）。
    maximum_area = float(config["max_piece_area_mm2"])
    # 计算并保存到 minimum_solidity（最小·实心度）。
    minimum_solidity = float(config["min_solidity"])
    # 计算并保存到 count（计数值）。
    count = 0
    # 遍历数据，逐项处理：contours。
    for contour in contours:
        # 计算并保存到 area_px（面积·像素）。
        area_px = float(cv2.contourArea(contour))
        # 计算并保存到 area_mm2（碎片面积，单位为平方毫米）。
        area_mm2 = area_px / (ppm * ppm)
        # 判断条件；满足时执行下面代码：not minimum_area <= area_mm2 <= maximum_area。
        if not minimum_area <= area_mm2 <= maximum_area:
            # 跳过本轮，进入下一轮循环。
            continue
        # 计算并保存到 hull_area（hull·面积）。
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        # 计算并保存到 solidity（轮廓面积与凸包面积之比，用于过滤形状异常区域）。
        solidity = area_px / hull_area if hull_area > 1e-6 else 0.0
        # 判断条件；满足时执行下面代码：solidity >= minimum_solidity。
        if solidity >= minimum_solidity:
            # 更新变量：count（计数值）。
            count += 1
    # 返回结果：count（计数值）。
    return count


# 【函数：refine_generic_piece_mask】仅在通用模式中受限闭合反光造成的轮廓裂缝，异常时回退原掩膜。
# 参数 mask（二值掩膜（0为背景，非零为选中区域））：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 diagnostics（是否输出诊断信息）：布尔值True/False；省略时使用False。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def refine_generic_piece_mask(mask: np.ndarray, config: dict[str, Any],
                              diagnostics: bool = False) -> np.ndarray:
    """仅在通用模式中受限闭合反光造成的轮廓裂缝，异常时回退原掩膜。"""
    # 判断条件；满足时执行下面代码：str(config.get('puzzle_mode', 'fixed_figure_2')) != 'generic_geometry'。
    if str(config.get("puzzle_mode", "fixed_figure_2")) != "generic_geometry":
        # 返回结果：mask（二值掩膜（0为背景，非零为选中区域））。
        return mask

    # 计算并保存到 close_mm（闭运算·毫米）。
    close_mm = float(config.get("generic_contour_close_mm", 0.0))
    # 判断条件；满足时执行下面代码：close_mm <= 0.0。
    if close_mm <= 0.0:
        # 返回结果：mask（二值掩膜（0为背景，非零为选中区域））。
        return mask

    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 kernel_px（结构元素尺寸（像素））。
    kernel_px = max(3, int(round(close_mm * ppm)))
    # 更新变量：kernel_px（结构元素尺寸（像素））。
    kernel_px += 1 - kernel_px % 2
    # 计算并保存到 kernel（形态学处理所用结构元素）。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    # 计算并保存到 closed（闭运算后）。
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # 计算并保存到 before_count（操作前·数量）。
    before_count = count_piece_candidates(mask, config)
    # 计算并保存到 after_count（操作后·数量）。
    after_count = count_piece_candidates(closed, config)
    # 计算并保存到 before_pixels（操作前·pixels）。
    before_pixels = int(np.count_nonzero(mask))
    # 计算并保存到 after_pixels（操作后·pixels）。
    after_pixels = int(np.count_nonzero(closed))
    # 计算并保存到 growth_ratio（growth·比例）。
    growth_ratio = ((after_pixels - before_pixels) / before_pixels
                    # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
                    if before_pixels > 0 else math.inf)
    # 计算并保存到 maximum_growth（最大·growth）。
    maximum_growth = float(config.get("generic_max_mask_growth_ratio", 0.08))
    # 计算并保存到 accepted（通过判据的结果或标记）。
    accepted = (before_count > 0 and before_count == after_count and
                0.0 <= growth_ratio <= maximum_growth)

    # 判断条件；满足时执行下面代码：diagnostics。
    if diagnostics:
        # 计算并保存到 growth_text（growth·文本）。
        growth_text = "无穷" if not math.isfinite(growth_ratio) else f"{growth_ratio * 100:.2f}%"
        # 计算并保存到 result（当前步骤得到的结果）。
        result = "采用" if accepted else "回退原掩膜"
        # 调用函数：print。
        print(f"[诊断] 通用轮廓闭合={close_mm:.1f}mm：有效碎片"
              f"{before_count}->{after_count}，掩膜增长={growth_text}，{result}")
    # 返回结果：closed if accepted else mask。
    return closed if accepted else mask


# 【函数：contour_pick_point】选择磁吸点：优先取质心（旋转力矩最均衡），仅在质心离边缘太近时才退让到安全区域重心。
# 距离变换的每个值表示该像素离背景的距离。优先使用面积质心；不够安全时取满足边距的区域重心，避免偏到碎片较宽的一头。
# 参数 contour（碎片轮廓点数组）：NumPy数组。
# 参数 shape（数组形状或图像高宽）：元组（依次为整数、整数）。
# 参数 centroid_px（面积质心·像素）：元组（依次为浮点数、浮点数）或None（不返回业务结果）；省略时使用None。
# 参数 min_margin_px（最小·边距·像素）：浮点数；省略时使用0.0。
# 返回类型：元组（依次为浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
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
    # 计算并保存到 x（当前横坐标或横向数据）、y（当前纵坐标或纵向数据）、width（当前区域宽度）、height（当前区域高度）。
    x, y, width, height = cv2.boundingRect(contour)
    # 计算并保存到 padding（边缘留白）。
    padding = 3
    # 计算并保存到 local（局部）。
    local = np.zeros((height + 2 * padding, width + 2 * padding), np.uint8)
    # 计算并保存到 shifted（平移或扰动后的数据）。
    shifted = contour.astype(np.int32).copy()
    # 更新变量：shifted[:, 0, 0]。
    shifted[:, 0, 0] -= x - padding
    # 更新变量：shifted[:, 0, 1]。
    shifted[:, 0, 1] -= y - padding
    # 调用函数：cv2.drawContours。
    cv2.drawContours(local, [shifted], -1, 255, -1)
    # 计算并保存到 distance（当前距离结果）。
    distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    # 计算并保存到 max_distance（最大·距离）。
    max_distance = float(cv2.minMaxLoc(distance)[1])

    # 【函数：to_shape】将局部吸取点还原到整张图像坐标，并限制在图像边界以内。
    # 参数 local_x（局部·X轴）：浮点数。
    # 参数 local_y（局部·Y轴）：浮点数。
    # 返回类型：元组（依次为浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
    def to_shape(local_x: float, local_y: float) -> tuple[float, float]:
        # 返回结果：创建数据容器。
        return (float(np.clip(local_x + x - padding, 0, shape[1] - 1)),
                float(np.clip(local_y + y - padding, 0, shape[0] - 1)))

    # 判断条件；满足时执行下面代码：centroid_px is not None。
    if centroid_px is not None:
        # 计算并保存到 local_cx（局部·cx）。
        local_cx = centroid_px[0] - (x - padding)
        # 计算并保存到 local_cy（局部·cy）。
        local_cy = centroid_px[1] - (y - padding)
        # 计算并保存到 ix、iy。
        ix, iy = int(round(local_cx)), int(round(local_cy))
        # 判断条件；满足时执行下面代码：0 <= iy < distance.shape[0] and 0 <= ix < distance.shape[1]。
        if 0 <= iy < distance.shape[0] and 0 <= ix < distance.shape[1]:
            # 计算并保存到 centroid_margin（面积质心·边距）。
            centroid_margin = float(distance[iy, ix])
            # 判断条件；满足时执行下面代码：centroid_margin >= min(min_margin_px, max_distance) and centroid_margin > 0.0。
            if centroid_margin >= min(min_margin_px, max_distance) and centroid_margin > 0.0:
                # 返回结果：将局部吸取点还原到整张图像坐标，并限制在图像边界以内。
                return to_shape(local_cx, local_cy)

    # 遍历数据，逐项处理：(1.0, 0.85, 0.7, 0.5, 0.3)。
    for ratio in (1.0, 0.85, 0.7, 0.5, 0.3):
        # 计算并保存到 threshold（阈值）。
        threshold = min(min_margin_px, max_distance * ratio) if min_margin_px > 0 else max_distance * ratio
        # 计算并保存到 safe（安全）。
        safe = distance >= max(threshold, 1e-6)
        # 判断条件；满足时执行下面代码：not np.any(safe)。
        if not np.any(safe):
            # 跳过本轮，进入下一轮循环。
            continue
        # 计算并保存到 ys、xs。
        ys, xs = np.nonzero(safe)
        # 计算并保存到 weights。
        weights = distance[ys, xs]
        # 计算并保存到 weight_sum。
        weight_sum = float(np.sum(weights))
        # 判断条件；满足时执行下面代码：weight_sum <= 1e-06。
        if weight_sum <= 1e-6:
            # 跳过本轮，进入下一轮循环。
            continue
        # 计算并保存到 local_x（局部·X轴）。
        local_x = float(np.sum(xs.astype(np.float64) * weights) / weight_sum)
        # 计算并保存到 local_y（局部·Y轴）。
        local_y = float(np.sum(ys.astype(np.float64) * weights) / weight_sum)
        # 返回结果：将局部吸取点还原到整张图像坐标，并限制在图像边界以内。
        return to_shape(local_x, local_y)

    # 计算并保存到 maximum（允许上限或计算出的最大值）。
    maximum = cv2.minMaxLoc(distance)[3]
    # 返回结果：将局部吸取点还原到整张图像坐标，并限制在图像边界以内。
    return to_shape(float(maximum[0]), float(maximum[1]))


# 【函数：normalize_angle】根据最小外接矩形的宽高修正角度，统一到长边方向。
# 参数 rect：元组（依次为任意类型、任意类型、浮点数）。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def normalize_angle(rect: tuple[Any, Any, float]) -> float:
    # 计算并保存到 _（此处不需要使用的返回值或循环占位变量）、_（此处不需要使用的返回值或循环占位变量）、width（当前区域宽度）、height（当前区域高度）、angle（当前计算或搜索的角度）。
    (_, _), (width, height), angle = rect
    # 判断条件；满足时执行下面代码：width < height。
    if width < height:
        # 更新变量：angle（当前计算或搜索的角度）。
        angle += 90.0
    # 返回结果：调用 float。
    return float((angle + 180.0) % 360.0 - 180.0)


# 【函数：detect_pieces】从掩膜提取轮廓，过滤面积和实心度，计算质心、吸取点、角度和尺寸。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 diagnostics（是否输出诊断信息）：布尔值True/False；省略时使用False。
# 参数 mask_override（外部提供的掩膜，None表示由检测函数生成）：NumPy数组或None（不返回业务结果）；省略时使用None。
# 返回类型：元组（依次为Piece列表/序列、NumPy数组）；箭头->是类型提示，不会替你转换实际返回值。
def detect_pieces(paper: np.ndarray, config: dict[str, Any],
                  diagnostics: bool = False,
                  mask_override: np.ndarray | None = None) -> tuple[list[Piece], np.ndarray]:
    # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
    mask = make_piece_mask(paper, config, diagnostics) if mask_override is None else mask_override.copy()
    # 判断条件；满足时执行下面代码：diagnostics and mask_override is not None。
    if diagnostics and mask_override is not None:
        # 计算并保存到 selected_ratio（选中·比例）。
        selected_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        # 计算并保存到 mode（模式）。
        mode = str(config.get("segmentation_mode", "pink_hsv"))
        # 调用函数：print。
        print(f"[诊断] 已按{mode}生成并校正掩膜，白色像素占比={selected_ratio * 100:.2f}%")
    # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
    mask = refine_generic_piece_mask(mask, config, diagnostics)
    # 计算并保存到 contours（检测到的轮廓集合）、_（此处不需要使用的返回值或循环占位变量）。
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 pieces（碎片列表）。
    pieces: list[Piece] = []

    # 遍历数据，逐项处理：contours。
    for contour in contours:
        # 计算并保存到 area_px（面积·像素）。
        area_px = cv2.contourArea(contour)
        # 计算并保存到 area_mm2（碎片面积，单位为平方毫米）。
        area_mm2 = area_px / (ppm * ppm)
        # 判断条件；满足时执行下面代码：not float(config['min_piece_area_mm2']) <= area_mm2 <= float(config['max_piece_…。
        if not float(config["min_piece_area_mm2"]) <= area_mm2 <= float(config["max_piece_area_mm2"]):
            # 跳过本轮，进入下一轮循环。
            continue
        # 计算并保存到 hull_area（hull·面积）。
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        # 计算并保存到 solidity（轮廓面积与凸包面积之比，用于过滤形状异常区域）。
        solidity = area_px / hull_area if hull_area > 1e-6 else 0.0
        # 判断条件；满足时执行下面代码：solidity < float(config['min_solidity'])。
        if solidity < float(config["min_solidity"]):
            # 跳过本轮，进入下一轮循环。
            continue
        # 计算并保存到 moments（轮廓的图像矩，用于求面积质心）。
        moments = cv2.moments(contour)
        # 判断条件；满足时执行下面代码：abs(moments['m00']) < 1e-06。
        if abs(moments["m00"]) < 1e-6:
            # 跳过本轮，进入下一轮循环。
            continue
        # 计算并保存到 center（本步骤使用的中心位置）。
        center = (float(moments["m10"] / moments["m00"]),
                  float(moments["m01"] / moments["m00"]))
        # 计算并保存到 pick_margin_px（磁吸点·边距·像素）。
        pick_margin_px = float(config.get("pick_point_min_margin_mm", 6.0)) * ppm
        # 计算并保存到 pick（吸取位置）。
        pick = contour_pick_point(contour, paper.shape[:2], center, pick_margin_px)
        # 计算并保存到 rect。
        rect = cv2.minAreaRect(contour)
        # 计算并保存到 _（此处不需要使用的返回值或循环占位变量）、_（此处不需要使用的返回值或循环占位变量）、width_px（宽度·像素）、height_px（高度·像素）、_（此处不需要使用的返回值或循环占位变量）。
        (_, _), (width_px, height_px), _ = rect
        # 调用函数：pieces.append。
        pieces.append(Piece(
            0, contour, center, pick, normalize_angle(rect), float(area_mm2),
            float(width_px / ppm), float(height_px / ppm)
        ))

    # 调用函数：pieces.sort。
    pieces.sort(key=lambda item: item.area_mm2, reverse=True)
    # 计算并保存到 accepted（通过判据的结果或标记）。
    accepted = pieces[:int(config["max_pieces"])]
    # 判断条件；满足时执行下面代码：diagnostics。
    if diagnostics:
        # 调用函数：print。
        print(f"[诊断] 二值轮廓={len(contours)}，通过面积/实心度筛选={len(accepted)}")
    # 返回结果：创建数据容器。
    return accepted, mask


# =============================================================================
# 【分区】安全搬运顺序编号
# 功能：按行聚类再左右排序，默认靠近目标区（纸面下半）的片先搬，减少后搬的片被挡。
# 可修改：config 的 numbering_mode、row_threshold_mm。
# 看情况改：现场片堆在上半且目标在下半，保持 bottom_to_top；若目标改到上半，才改编号方向。
# 不要改：piece_id 从 1 连续编号；plan.json 的 move_order 依赖这里。
# =============================================================================
# ------------------------- 安全搬运顺序 -------------------------

# 【函数：group_rows】按纵坐标差把碎片分成若干行，每行再按横坐标排序。
# 参数 pieces（碎片列表）：可逐项遍历的Piece列表/序列。
# 参数 threshold_px（阈值·像素）：浮点数。
# 参数 descending：布尔值True/False。
# 返回类型：Piece列表/序列列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def group_rows(pieces: Iterable[Piece], threshold_px: float,
               descending: bool) -> list[list[Piece]]:
    # 计算并保存到 ordered（已排序的数据）。
    ordered = sorted(pieces, key=lambda item: item.center_px[1], reverse=descending)
    # 计算并保存到 rows（记录集合）。
    rows: list[list[Piece]] = []
    # 遍历数据，逐项处理：ordered。
    for piece in ordered:
        # 判断条件；满足时执行下面代码：not rows。
        if not rows:
            # 调用函数：rows.append。
            rows.append([piece])
            # 跳过本轮，进入下一轮循环。
            continue
        # 计算并保存到 row_y（行·Y轴）。
        row_y = float(np.mean([item.center_px[1] for item in rows[-1]]))
        # 判断条件；满足时执行下面代码：abs(piece.center_px[1] - row_y) <= threshold_px。
        if abs(piece.center_px[1] - row_y) <= threshold_px:
            # 调用函数：rows[-1].append。
            rows[-1].append(piece)
        # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
        else:
            # 调用函数：rows.append。
            rows.append([piece])
    # 返回结果：rows（记录集合）。
    return rows


# 【函数：number_by_move_order】靠近下半区的碎片先搬，同一行从左到右。
# 参数 pieces（碎片列表）：Piece列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：Piece列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def number_by_move_order(pieces: list[Piece], config: dict[str, Any]) -> list[Piece]:
    """靠近下半区的碎片先搬，同一行从左到右。"""
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 descending。
    descending = str(config["numbering_mode"]).startswith("bottom_to_top")
    # 计算并保存到 rows（记录集合）。
    rows = group_rows(pieces, float(config["row_threshold_mm"]) * ppm, descending)
    # 计算并保存到 ordered（已排序的数据）。
    ordered: list[Piece] = []
    # 遍历数据，逐项处理：rows。
    for row in rows:
        # 调用函数：row.sort。
        row.sort(key=lambda item: item.center_px[0])
        # 调用函数：ordered.extend。
        ordered.extend(row)
    # 遍历数据，逐项处理：enumerate(ordered, 1)。
    for index, piece in enumerate(ordered, 1):
        # 计算并保存到 piece.piece_id（面向显示和计划的碎片编号）。
        piece.piece_id = index
    # 返回结果：ordered（已排序的数据）。
    return ordered


# =============================================================================
# 【分区】固定图2 / 通用几何求解入口
# 功能：fixed_figure_2 把四片对上 A/B/C/D 模板；generic_geometry 交给 generic_solver。
#       含多边形拟合、间隙外推、花纹验证、solve_puzzle 分流。
# 可修改：目标矩形尺寸 target_rectangle_size_mm（固定图2默认 100×60）；间隙 clearance 相关配置。
# 看情况改：赛场未知图形用 generic_geometry（--puzzle-set site）；训练图2用 self。
# 不要改：刚体（只转和平移，不缩放）；正角顺时针；solve_puzzle 未知 mode 必须报错。
# =============================================================================
# ------------------------- 固定图2拼图求解 -------------------------

# 【函数：polygon_centroid】使用轮廓图像矩计算面积质心；退化时改用顶点平均值。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def polygon_centroid(points: np.ndarray) -> np.ndarray:
    # 计算并保存到 contour（碎片轮廓点数组）。
    contour = np.asarray(points, np.float32).reshape(-1, 1, 2)
    # 计算并保存到 moments（轮廓的图像矩，用于求面积质心）。
    moments = cv2.moments(contour)
    # 判断条件；满足时执行下面代码：abs(moments['m00']) < 1e-06。
    if abs(moments["m00"]) < 1e-6:
        # 抛出异常，通知上层处理：ValueError('碎片轮廓面积为0，无法计算质心')。
        raise ValueError("碎片轮廓面积为0，无法计算质心")
    # 返回结果：把输入转换成NumPy数组；类型相同且无需复制时可能复用原存储。
    return np.asarray([moments["m10"] / moments["m00"],
                       moments["m01"] / moments["m00"]], np.float64)


# 【函数：rotation_matrix】图像坐标系旋转矩阵：正角度在画面上表现为顺时针。
# 参数 angle_deg（角度，单位为度）：浮点数。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def rotation_matrix(angle_deg: float) -> np.ndarray:
    """图像坐标系旋转矩阵：正角度在画面上表现为顺时针。"""
    # 计算并保存到 radians（换算后的弧度角）。
    radians = math.radians(angle_deg)
    # 计算并保存到 cosine（角度的余弦值）、sine（角度的正弦值）。
    cosine, sine = math.cos(radians), math.sin(radians)
    # 返回结果：把输入转换成NumPy数组；类型相同且无需复制时可能复用原存储。
    return np.asarray([[cosine, -sine], [sine, cosine]], np.float64)


# 【函数：rotate_points】将二维行向量点集乘以旋转矩阵的转置，得到旋转后的点集。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组。
# 参数 angle_deg（角度，单位为度）：浮点数。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def rotate_points(points: np.ndarray, angle_deg: float) -> np.ndarray:
    # 返回结果：计算 np.asarray(points, np.float64) @ rotation_matrix(angle_deg).T。
    return np.asarray(points, np.float64) @ rotation_matrix(angle_deg).T


# 【函数：normalized_rotation】将角度环绕归一化到[-180,180)，例如270度变成-90度。
# 参数 angle_deg（角度，单位为度）：浮点数。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def normalized_rotation(angle_deg: float) -> float:
    # 计算并保存到 result（当前步骤得到的结果）。
    result = float((angle_deg + 180.0) % 360.0 - 180.0)
    # 返回结果：180.0 if math.isclose(result, -180.0, abs_tol=1e-09) else result。
    return 180.0 if math.isclose(result, -180.0, abs_tol=1e-9) else result


# 【函数：rotation_direction】根据角度符号输出CW或CCW，死区内输出NONE。
# 参数 angle_deg（角度，单位为度）：浮点数。
# 参数 deadband_deg（deadband·度）：浮点数；省略时使用0.25。
# 返回类型：字符串；箭头->是类型提示，不会替你转换实际返回值。
def rotation_direction(angle_deg: float, deadband_deg: float = 0.25) -> str:
    # 判断条件；满足时执行下面代码：abs(angle_deg) <= deadband_deg。
    if abs(angle_deg) <= deadband_deg:
        # 返回结果：'NONE'。
        return "NONE"
    # 返回结果：'CW' if angle_deg > 0.0 else 'CCW'。
    return "CW" if angle_deg > 0.0 else "CCW"


# 【函数：figure2_template_polygons】返回题目图2的四块多边形；placed=True时平移到A4下半区目标矩形。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 placed：布尔值True/False。
# 返回类型：字典（键为字符串，值为NumPy数组）；箭头->是类型提示，不会替你转换实际返回值。
def figure2_template_polygons(config: dict[str, Any], placed: bool) -> dict[str, np.ndarray]:
    """返回题目图2的四块多边形；placed=True时平移到A4下半区目标矩形。"""
    # 计算并保存到 size（尺寸数据）。
    size = np.asarray(config.get("target_rectangle_size_mm", [100.0, 60.0]), np.float64)
    # 判断条件；满足时执行下面代码：size.shape != (2,) or np.any(size <= 0.0)。
    if size.shape != (2,) or np.any(size <= 0.0):
        # 抛出异常，通知上层处理：ValueError('target_rectangle_size_mm必须是两个正数')。
        raise ValueError("target_rectangle_size_mm必须是两个正数")
    # 计算并保存到 scale（当前缩放比例或试探步长比例）。
    scale = size / np.asarray([100.0, 60.0], np.float64)
    # 计算并保存到 offset（偏移）。
    offset = np.asarray(config.get("target_rectangle_top_left_mm", [55.0, 205.0]),
                        np.float64) if placed else np.zeros(2, np.float64)
    # 返回结果：{template_id: np.asarray(points, np.float64) * scale + offset for template_id, point…。
    return {
        template_id: np.asarray(points, np.float64) * scale + offset
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for template_id, points in FIGURE2_TEMPLATE_POLYGONS_MM.items()
    }


# 【函数：validate_target_rectangle】检查目标矩形的尺寸和摆放位置是否满足当前配置要求。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def validate_target_rectangle(config: dict[str, Any]) -> None:
    # 计算并保存到 top_left（上边·左边）。
    top_left = np.asarray(config.get("target_rectangle_top_left_mm", [55.0, 205.0]),
                          np.float64)
    # 计算并保存到 size（尺寸数据）。
    size = np.asarray(config.get("target_rectangle_size_mm", [100.0, 60.0]), np.float64)
    # 判断条件；满足时执行下面代码：top_left.shape != (2,) or size.shape != (2,)。
    if top_left.shape != (2,) or size.shape != (2,):
        # 抛出异常，通知上层处理：ValueError('目标矩形参数格式错误')。
        raise ValueError("目标矩形参数格式错误")
    # 计算并保存到 bottom_right（下边·右边）。
    bottom_right = top_left + size
    # 计算并保存到 paper_width（纸面宽度（毫米））。
    paper_width = float(config["paper_width_mm"])
    # 计算并保存到 paper_height（纸面高度（毫米））。
    paper_height = float(config["paper_height_mm"])
    # 计算并保存到 target_top（目标·上边）。
    target_top = float(config.get("target_region_top_mm", paper_height / 2.0))
    # 判断条件；满足时执行下面代码：top_left[0] < 0.0 or top_left[1] < target_top or bottom_right[0] > paper_width …。
    if (top_left[0] < 0.0 or top_left[1] < target_top or
            bottom_right[0] > paper_width or bottom_right[1] > paper_height):
        # 抛出异常，通知上层处理：ValueError(f'目标矩形左上({top_left[0]:.1f},{top_left[1]:.1f})mm、尺寸({size[0…。
        raise ValueError(
            f"目标矩形左上({top_left[0]:.1f},{top_left[1]:.1f})mm、"
            f"尺寸({size[0]:.1f},{size[1]:.1f})mm不完整位于A4下半区"
        )


# 【函数：rasterized_iou】将旋转后的源形状和模板画到二值图上，计算交集/并集。
# 参数 source_local_mm（源·局部·毫米）：NumPy数组。
# 参数 angle_deg（角度，单位为度）：浮点数。
# 参数 resolution（栅格分辨率）：浮点数。
# 参数 canvas_size（方形画布的边长（像素））：整数。
# 参数 origin（原点）：NumPy数组。
# 参数 target_mask（目标·掩膜）：NumPy数组。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def rasterized_iou(source_local_mm: np.ndarray, angle_deg: float,
                   resolution: float, canvas_size: int,
                   origin: np.ndarray, target_mask: np.ndarray) -> float:
    # 计算并保存到 rotated。
    rotated = rotate_points(source_local_mm, angle_deg)
    # 计算并保存到 source_px（源·像素）。
    source_px = np.rint(rotated * resolution + origin).astype(np.int32)
    # 计算并保存到 source_mask（源·掩膜）。
    source_mask = np.zeros((canvas_size, canvas_size), np.uint8)
    # 调用函数：cv2.fillPoly。
    cv2.fillPoly(source_mask, [source_px], 255)
    # 计算并保存到 intersection（交集大小）。
    intersection = int(np.count_nonzero((source_mask > 0) & (target_mask > 0)))
    # 计算并保存到 union（并集大小）。
    union = int(np.count_nonzero((source_mask > 0) | (target_mask > 0)))
    # 返回结果：intersection / union if union else 0.0。
    return intersection / union if union else 0.0


# 【函数：best_rotation_fit】枚举旋转角，以轮廓IoU求源碎片到固定模板的最佳旋转。
# 源轮廓和模板先移到各自质心附近；搜索角度使IoU最大。匹配中的尺度估计用于比较形状，不代表机械机构能缩放碎片。
# 参数 piece（当前碎片数据）：Piece。
# 参数 template_mm（template·毫米）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为浮点数、浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def best_rotation_fit(piece: Piece, template_mm: np.ndarray,
                      config: dict[str, Any]) -> tuple[float, float, float]:
    """枚举旋转角，以轮廓IoU求源碎片到固定模板的最佳旋转。"""
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 source_mm（源·毫米）。
    source_mm = piece.contour.reshape(-1, 2).astype(np.float64) / ppm
    # 计算并保存到 source_center（源·中心）。
    source_center = polygon_centroid(source_mm)
    # 计算并保存到 target_center（目标·中心）。
    target_center = polygon_centroid(template_mm)
    # 计算并保存到 source_area（源·面积）。
    source_area = max(float(cv2.contourArea(source_mm.astype(np.float32))), 1e-6)
    # 计算并保存到 target_area（目标·面积）。
    target_area = max(float(cv2.contourArea(template_mm.astype(np.float32))), 1e-6)
    # 计算并保存到 fit_scale（形状匹配时的尺度比；刚体搬运应保持为1）。
    fit_scale = math.sqrt(target_area / source_area)
    # 计算并保存到 source_local（源·局部）。
    source_local = (source_mm - source_center) * fit_scale
    # 计算并保存到 target_local（目标·局部）。
    target_local = template_mm - target_center

    # 计算并保存到 resolution（栅格分辨率）。
    resolution = max(1.0, float(config.get("rotation_match_pixels_per_mm", 2.0)))
    # 计算并保存到 radius。
    radius = max(float(np.max(np.linalg.norm(source_local, axis=1))),
                 # 传入命名参数：axis=1（指定运算维度：0通常沿行汇总，1沿列汇总）；取值过程：1。
                 float(np.max(np.linalg.norm(target_local, axis=1)))) + 4.0
    # 计算并保存到 canvas_size（方形画布的边长（像素））。
    canvas_size = max(64, int(math.ceil(2.0 * radius * resolution)) + 5)
    # 判断条件；满足时执行下面代码：canvas_size % 2 == 0。
    if canvas_size % 2 == 0:
        # 更新变量：canvas_size（方形画布的边长（像素））。
        canvas_size += 1
    # 计算并保存到 origin（原点）。
    origin = np.asarray([canvas_size // 2, canvas_size // 2], np.float64)
    # 计算并保存到 target_px（目标·像素）。
    target_px = np.rint(target_local * resolution + origin).astype(np.int32)
    # 计算并保存到 target_mask（目标·掩膜）。
    target_mask = np.zeros((canvas_size, canvas_size), np.uint8)
    # 调用函数：cv2.fillPoly。
    cv2.fillPoly(target_mask, [target_px], 255)

    # 计算并保存到 coarse_step（粗搜索·步长）。
    coarse_step = max(0.5, float(config.get("rotation_search_step_deg", 1.0)))
    # 计算并保存到 best_angle（最佳·角度）、best_iou（最佳·交并比）。
    best_angle, best_iou = 0.0, -1.0
    # 遍历数据，逐项处理：np.arange(-180.0, 180.0, coarse_step)。
    for angle in np.arange(-180.0, 180.0, coarse_step):
        # 计算并保存到 iou（交并比（交集面积除以并集面积））。
        iou = rasterized_iou(source_local, float(angle), resolution,
                             canvas_size, origin, target_mask)
        # 判断条件；满足时执行下面代码：iou > best_iou + 1e-09 or (math.isclose(iou, best_iou, abs_tol=1e-09) and abs(a…。
        if iou > best_iou + 1e-9 or (math.isclose(iou, best_iou, abs_tol=1e-9)
                                     # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
                                     and abs(angle) < abs(best_angle)):
            # 计算并保存到 best_angle（最佳·角度）、best_iou（最佳·交并比）。
            best_angle, best_iou = float(angle), float(iou)

    # 计算并保存到 refine_step（精细搜索·步长）。
    refine_step = max(0.1, float(config.get("rotation_refine_step_deg", 0.25)))
    # 判断条件；满足时执行下面代码：refine_step < coarse_step。
    if refine_step < coarse_step:
        # 遍历数据，逐项处理：np.arange(best_angle - coarse_step, best_angle + coarse_step + 1e-09, refine_st…。
        for angle in np.arange(best_angle - coarse_step, best_angle + coarse_step + 1e-9,
                               refine_step):
            # 计算并保存到 iou（交并比（交集面积除以并集面积））。
            iou = rasterized_iou(source_local, float(angle), resolution,
                                 canvas_size, origin, target_mask)
            # 判断条件；满足时执行下面代码：iou > best_iou + 1e-09 or (math.isclose(iou, best_iou, abs_tol=1e-09) and abs(a…。
            if iou > best_iou + 1e-9 or (math.isclose(iou, best_iou, abs_tol=1e-9)
                                         # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
                                         and abs(angle) < abs(best_angle)):
                # 计算并保存到 best_angle（最佳·角度）、best_iou（最佳·交并比）。
                best_angle, best_iou = float(angle), float(iou)
    # 返回结果：创建数据容器。
    return normalized_rotation(best_angle), best_iou, fit_scale


# 【函数：solve_fixed_puzzle】把4块源轮廓全局匹配到图2的A/B/C/D模板。
# 四块对应四个模板只有4!=24种编号分配；把每种分配的匹配代价相加，再选整体最佳，而不是每片独立抢同一模板。
# 参数 pieces（碎片列表）：Piece列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：PuzzleMatch列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def solve_fixed_puzzle(pieces: list[Piece], config: dict[str, Any]) -> list[PuzzleMatch]:
    """把4块源轮廓全局匹配到图2的A/B/C/D模板。"""
    # 计算并保存到 expected（期望）。
    expected = int(config.get("expected_piece_count", 4))
    # 判断条件；满足时执行下面代码：len(pieces) != expected。
    if len(pieces) != expected:
        # 抛出异常，通知上层处理：ValueError(f'固定图2求解需要完整识别{expected}块，当前识别到{len(pieces)}块')。
        raise ValueError(f"固定图2求解需要完整识别{expected}块，当前识别到{len(pieces)}块")
    # 调用函数：validate_target_rectangle。
    validate_target_rectangle(config)
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 source_limit（源·上限）。
    source_limit = (float(config.get("target_region_top_mm", 148.5)) +
                    float(config.get("source_center_tolerance_mm", 5.0)))
    # 计算并保存到 invalid_sources。
    invalid_sources = [piece.piece_id for piece in pieces
                       # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
                       if piece.center_mm(ppm)[1] > source_limit]
    # 判断条件；满足时执行下面代码：invalid_sources。
    if invalid_sources:
        # 计算并保存到 ids（编号集合）。
        ids = ",".join(f"P{piece_id}" for piece_id in invalid_sources)
        # 抛出异常，通知上层处理：ValueError(f'{ids}的中心已进入A4下半区，请把4块初始碎片放回上半区域')。
        raise ValueError(f"{ids}的中心已进入A4下半区，请把4块初始碎片放回上半区域")
    # 计算并保存到 local_templates（局部·templates）。
    local_templates = figure2_template_polygons(config, placed=False)
    # 计算并保存到 target_templates（目标·templates）。
    target_templates = figure2_template_polygons(config, placed=True)
    # 计算并保存到 clearance_shifts（留缝·shifts）。
    clearance_shifts = fixed_target_clearance_shifts(target_templates, config)
    # 计算并保存到 template_ids（template·编号集合）。
    template_ids = list(FIGURE2_TEMPLATE_POLYGONS_MM)
    # 计算并保存到 source_total（源·总量）。
    source_total = sum(max(piece.area_mm2, 1e-6) for piece in pieces)
    # 计算并保存到 template_areas。
    template_areas = {
        template_id: float(cv2.contourArea(polygon.astype(np.float32)))
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for template_id, polygon in local_templates.items()
    }
    # 计算并保存到 global_area_scale（整体·面积·尺度）。
    global_area_scale = sum(template_areas.values()) / source_total

    # 计算并保存到 metrics。
    metrics: dict[tuple[int, str], tuple[float, float, float]] = {}
    # 遍历数据，逐项处理：pieces。
    for piece in pieces:
        # 遍历数据，逐项处理：template_ids。
        for template_id in template_ids:
            # 计算并保存到 template。
            template = local_templates[template_id].astype(np.float32).reshape(-1, 1, 2)
            # 计算并保存到 area_error（面积·误差）。
            area_error = abs(piece.area_mm2 * global_area_scale -
                             template_areas[template_id]) / template_areas[template_id]
            # 计算并保存到 shape_distance（形状差异指标）。
            shape_distance = float(cv2.matchShapes(
                piece.contour.astype(np.float32), template, cv2.CONTOURS_MATCH_I1, 0.0
            ))
            # 计算并保存到 cost（代价）。
            cost = area_error + 0.20 * min(shape_distance, 3.0)
            # 计算并保存到 metrics[piece.piece_id, template_id]。
            metrics[(piece.piece_id, template_id)] = (area_error, shape_distance, cost)

    # 计算并保存到 best_permutation（最佳·permutation）。
    best_permutation = min(
        itertools.permutations(template_ids, len(pieces)),
        key=lambda permutation: sum(
            metrics[(piece.piece_id, template_id)][2]
            # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
            for piece, template_id in zip(pieces, permutation)
        ),
    )

    # 计算并保存到 max_area_error（最大·面积·误差）。
    max_area_error = float(config.get("max_template_area_error_ratio", 0.35))
    # 计算并保存到 max_shape_distance（最大·形状·距离）。
    max_shape_distance = float(config.get("max_template_shape_distance", 0.80))
    # 计算并保存到 min_iou（最小·交并比）。
    min_iou = float(config.get("min_rotation_match_iou", 0.55))
    # 计算并保存到 matches（匹配记录列表）。
    matches: list[PuzzleMatch] = []
    # 遍历数据，逐项处理：zip(pieces, best_permutation)。
    for piece, template_id in zip(pieces, best_permutation):
        # 计算并保存到 local_template（局部·template）。
        local_template = local_templates[template_id]
        # 计算并保存到 shift（一次平移量）。
        shift = np.asarray(clearance_shifts[template_id], np.float64)
        # 计算并保存到 target_polygon（目标·多边形）。
        target_polygon = target_templates[template_id] + shift
        # 计算并保存到 rotation_deg（碎片需要旋转的有符号角度（度））、match_iou（形状匹配的交并比）、fit_scale（形状匹配时的尺度比；刚体搬运应保持为1）。
        rotation_deg, match_iou, fit_scale = best_rotation_fit(piece, local_template, config)
        # 计算并保存到 source_center（源·中心）。
        source_center = polygon_centroid(piece.contour.reshape(-1, 2) / ppm)
        # 计算并保存到 target_center（目标·中心）。
        target_center = polygon_centroid(target_polygon)
        # 计算并保存到 source_pick（源·磁吸点）。
        source_pick = np.asarray(piece.pick_mm(ppm), np.float64)
        # 计算并保存到 target_pick（目标·磁吸点）。
        target_pick = target_center + rotate_points(
            (source_pick - source_center) * fit_scale, rotation_deg
        )
        # 计算并保存到 inside。
        inside = cv2.pointPolygonTest(target_polygon.astype(np.float32),
                                      (float(target_pick[0]), float(target_pick[1])), False) >= 0
        # 计算并保存到 area_error（面积·误差）、shape_distance（形状差异指标）、_（此处不需要使用的返回值或循环占位变量）。
        area_error, shape_distance, _ = metrics[(piece.piece_id, template_id)]
        # 计算并保存到 status（处理结果的状态标记）。
        status = "ok" if (area_error <= max_area_error and
                           shape_distance <= max_shape_distance and
                           match_iou >= min_iou and inside) else "match_failed"
        # 计算并保存到 match（当前匹配记录）。
        match = PuzzleMatch(
            # 传入命名参数：piece=piece（当前碎片数据）；取值过程：piece（当前碎片数据）。
            piece=piece,
            # 传入命名参数：template_id=template_id（目标模板编号）；取值过程：template_id（目标模板编号）。
            template_id=template_id,
            # 传入命名参数：target_polygon_mm=target_polygon（目标布局中的多边形顶点（毫米））；取值过程：target_polygon（目标·多边形）。
            target_polygon_mm=target_polygon,
            # 传入命名参数：target_pick_mm=(float(target_pick[0]), float(target_pick[1]))（放置后磁吸点的目标坐标（毫米））；取值过程：创建数据容器。
            target_pick_mm=(float(target_pick[0]), float(target_pick[1])),
            # 传入命名参数：rotation_deg=rotation_deg（碎片需要旋转的有符号角度（度））；取值过程：rotation_deg（碎片需要旋转的有符号角度（度））。
            rotation_deg=rotation_deg,
            # 传入命名参数：rotation_direction=rotation_direction(rotation_deg)（旋转方向标记）；取值过程：根据角度符号输出CW或CCW，死区内输出NONE。
            rotation_direction=rotation_direction(rotation_deg),
            # 传入命名参数：area_error_ratio=area_error（面积相对误差）；取值过程：area_error（面积·误差）。
            area_error_ratio=area_error,
            # 传入命名参数：shape_distance=shape_distance（形状差异指标）；取值过程：shape_distance（形状差异指标）。
            shape_distance=shape_distance,
            # 传入命名参数：match_iou=match_iou（形状匹配的交并比）；取值过程：match_iou（形状匹配的交并比）。
            match_iou=match_iou,
            # 传入命名参数：fit_scale=fit_scale（形状匹配时的尺度比；刚体搬运应保持为1）；取值过程：fit_scale（形状匹配时的尺度比；刚体搬运应保持为1）。
            fit_scale=fit_scale,
            # 传入命名参数：status=status（处理结果的状态标记）；取值过程：status（处理结果的状态标记）。
            status=status,
            # 传入命名参数：target_clearance_shift_mm=(float(shift[0]), float(shift[1]))（为留出碎片间隙施加的平移（毫米））；取值过程：创建数据容器。
            target_clearance_shift_mm=(float(shift[0]), float(shift[1])),
        )
        # 调用函数：matches.append。
        matches.append(match)
        # 调用函数：print。
        print(f"[匹配] P{piece.piece_id}->模板{template_id}: "
              f"旋转={rotation_deg:+.2f}deg({match.rotation_direction}), "
              f"IoU={match_iou:.3f}, 面积误差={area_error:.3f}, 状态={status}")
    # 返回结果：matches（匹配记录列表）。
    return matches


# 【函数：cyclic_contour_segment】返回闭合轮廓中从start到end（含端点）的有序点段。
# 参数 contour_points（轮廓·点集）：NumPy数组。
# 参数 start_index（start·索引）：整数。
# 参数 end_index（end·索引）：整数。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def cyclic_contour_segment(contour_points: np.ndarray,
                           start_index: int,
                           end_index: int) -> np.ndarray:
    """返回闭合轮廓中从start到end（含端点）的有序点段。"""
    # 计算并保存到 points（参与当前计算的一组坐标点）。
    points = np.asarray(contour_points, dtype=np.float64).reshape(-1, 2)
    # 判断条件；满足时执行下面代码：len(points) < 2。
    if len(points) < 2:
        # 抛出异常，通知上层处理：ValueError('轮廓点数量不足')。
        raise ValueError("轮廓点数量不足")
    # 计算并保存到 start（起始值或起点）。
    start = int(start_index) % len(points)
    # 计算并保存到 end（终止值或终点）。
    end = int(end_index) % len(points)
    # 判断条件；满足时执行下面代码：start <= end。
    if start <= end:
        # 返回结果：调用 points[start:end + 1].copy。
        return points[start:end + 1].copy()
    # 返回结果：把点集沿行方向拼接。
    return np.vstack((points[start:], points[:end + 1]))


# 【函数：fit_contour_edge_line】使用Huber距离为一段毛糙轮廓拟合稳定直线。
# 参数 edge_points（边·点集）：NumPy数组。
# 返回类型：元组（依次为NumPy数组、NumPy数组）；箭头->是类型提示，不会替你转换实际返回值。
def fit_contour_edge_line(edge_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """使用Huber距离为一段毛糙轮廓拟合稳定直线。"""
    # 计算并保存到 points（参与当前计算的一组坐标点）。
    points = np.asarray(edge_points, dtype=np.float32).reshape(-1, 2)
    # 判断条件；满足时执行下面代码：len(points) < 2 or float(np.max(np.linalg.norm(points - points[0], axis=1))) <=…。
    if len(points) < 2 or float(np.max(np.linalg.norm(points - points[0], axis=1))) <= 1e-6:
        # 抛出异常，通知上层处理：ValueError('轮廓边点不足或已退化')。
        raise ValueError("轮廓边点不足或已退化")
    # 计算并保存到 fitted。
    fitted = np.asarray(
        cv2.fitLine(points.reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01),
        # 传入命名参数：dtype=np.float64（指定数组元素类型）；取值过程：np.float64。
        dtype=np.float64,
    ).reshape(-1)
    # 计算并保存到 direction（方向标记或方向向量）。
    direction = fitted[:2]
    # 计算并保存到 norm。
    norm = float(np.linalg.norm(direction))
    # 判断条件；满足时执行下面代码：norm <= 1e-09。
    if norm <= 1e-9:
        # 抛出异常，通知上层处理：ValueError('直线拟合方向无效')。
        raise ValueError("直线拟合方向无效")
    # 返回结果：创建数据容器。
    return fitted[2:4], direction / norm


# 【函数：intersect_fitted_lines】计算两条参数直线交点；近似平行时拒绝不稳定结果。
# 参数 first（第一项数据）：元组（依次为NumPy数组、NumPy数组）。
# 参数 second（第二项数据）：元组（依次为NumPy数组、NumPy数组）。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def intersect_fitted_lines(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """计算两条参数直线交点；近似平行时拒绝不稳定结果。"""
    # 计算并保存到 point_a（点·a）、direction_a。
    point_a, direction_a = first
    # 计算并保存到 point_b（点·b）、direction_b。
    point_b, direction_b = second
    # 计算并保存到 denominator。
    denominator = float(
        direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    )
    # 判断条件；满足时执行下面代码：abs(denominator) <= math.sin(math.radians(3.0))。
    if abs(denominator) <= math.sin(math.radians(3.0)):
        # 抛出异常，通知上层处理：ValueError('相邻拟合边近似平行')。
        raise ValueError("相邻拟合边近似平行")
    # 计算并保存到 delta（当前增量或修正量）。
    delta = point_b - point_a
    # 计算并保存到 distance（当前距离结果）。
    distance = float(
        (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / denominator
    )
    # 计算并保存到 intersection（交集大小）。
    intersection = point_a + distance * direction_a
    # 判断条件；满足时执行下面代码：not np.isfinite(intersection).all()。
    if not np.isfinite(intersection).all():
        # 抛出异常，通知上层处理：ValueError('拟合直线交点无效')。
        raise ValueError("拟合直线交点无效")
    # 返回结果：intersection（交集大小）。
    return intersection


# 【函数：remove_near_collinear_vertices】移除接近180°的伪拐点，但至少保留三角形。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组。
# 参数 angle_tolerance_deg（角度·容差·度）：浮点数。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def remove_near_collinear_vertices(points: np.ndarray,
                                   angle_tolerance_deg: float) -> np.ndarray:
    """移除接近180°的伪拐点，但至少保留三角形。"""
    # 计算并保存到 polygon（当前多边形的顶点数组）。
    polygon = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    # 计算并保存到 tolerance（用于浮点比较或几何判断的容差）。
    tolerance = max(0.0, float(angle_tolerance_deg))
    # 只要条件成立就重复执行：len(polygon) > 3。
    while len(polygon) > 3:
        # 计算并保存到 angles（角度序列）。
        angles: list[float] = []
        # 遍历数据，逐项处理：range(len(polygon))。
        for index in range(len(polygon)):
            # 计算并保存到 previous（前一项）。
            previous = polygon[index - 1] - polygon[index]
            # 计算并保存到 following。
            following = polygon[(index + 1) % len(polygon)] - polygon[index]
            # 计算并保存到 denominator。
            denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
            # 判断条件；满足时执行下面代码：denominator <= 1e-09。
            if denominator <= 1e-9:
                # 调用函数：angles.append。
                angles.append(180.0)
                # 跳过本轮，进入下一轮循环。
                continue
            # 计算并保存到 cosine（角度的余弦值）。
            cosine = float(np.clip(np.dot(previous, following) / denominator, -1.0, 1.0))
            # 调用函数：angles.append。
            angles.append(math.degrees(math.acos(cosine)))
        # 计算并保存到 remove_index（remove·索引）。
        remove_index = int(np.argmax(angles))
        # 判断条件；满足时执行下面代码：180.0 - angles[remove_index] > tolerance。
        if 180.0 - angles[remove_index] > tolerance:
            # 立即结束当前循环。
            break
        # 计算并保存到 polygon（当前多边形的顶点数组）。
        polygon = np.delete(polygon, remove_index, axis=0)
    # 返回结果：polygon（当前多边形的顶点数组）。
    return polygon


# 【函数：refine_polygon_by_edge_lines】按原轮廓边段拟合直线，并以相邻直线交点修正多边形顶点。
# 参数 contour_points（轮廓·点集）：NumPy数组。
# 参数 approximated_points（approximated·点集）：NumPy数组。
# 参数 pixels_per_mm（每毫米对应的像素数）：浮点数；必须按参数名传入。
# 参数 max_vertex_shift_mm（最大·vertex·shift·毫米）：浮点数；必须按参数名传入。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def refine_polygon_by_edge_lines(
    contour_points: np.ndarray,
    approximated_points: np.ndarray,
    *,
    pixels_per_mm: float,
    max_vertex_shift_mm: float,
) -> np.ndarray:
    """按原轮廓边段拟合直线，并以相邻直线交点修正多边形顶点。"""
    # 计算并保存到 contour（碎片轮廓点数组）。
    contour = np.asarray(contour_points, dtype=np.float64).reshape(-1, 2)
    # 计算并保存到 approximated。
    approximated = np.asarray(approximated_points, dtype=np.float64).reshape(-1, 2)
    # 判断条件；满足时执行下面代码：not 3 <= len(approximated) <= 5。
    if not 3 <= len(approximated) <= 5:
        # 抛出异常，通知上层处理：ValueError('直线修正只接受3～5边初始多边形')。
        raise ValueError("直线修正只接受3～5边初始多边形")
    # 判断条件；满足时执行下面代码：pixels_per_mm <= 0.0 or max_vertex_shift_mm <= 0.0。
    if pixels_per_mm <= 0.0 or max_vertex_shift_mm <= 0.0:
        # 抛出异常，通知上层处理：ValueError('直线修正配置必须为正数')。
        raise ValueError("直线修正配置必须为正数")

    # approxPolyDP返回的顶点来自原轮廓；使用最近点索引可保留其循环顺序。
    # 计算并保存到 indices。
    indices = [
        # 传入命名参数：axis=1（指定运算维度：0通常沿行汇总，1沿列汇总）；取值过程：1。
        int(np.argmin(np.sum((contour - vertex) ** 2, axis=1)))
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for vertex in approximated
    ]
    # 判断条件；满足时执行下面代码：len(set(indices)) != len(indices)。
    if len(set(indices)) != len(indices):
        # 抛出异常，通知上层处理：ValueError('简化顶点无法唯一映射回原轮廓')。
        raise ValueError("简化顶点无法唯一映射回原轮廓")

    # 计算并保存到 fitted_lines。
    fitted_lines: list[tuple[np.ndarray, np.ndarray]] = []
    # 遍历数据，逐项处理：range(len(indices))。
    for index in range(len(indices)):
        # 计算并保存到 segment。
        segment = cyclic_contour_segment(
            contour, indices[index], indices[(index + 1) % len(indices)]
        )
        # 调用函数：fitted_lines.append。
        fitted_lines.append(fit_contour_edge_line(segment))

    # 计算并保存到 refined（优化后）。
    refined = np.vstack([
        intersect_fitted_lines(fitted_lines[index - 1], fitted_lines[index])
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for index in range(len(fitted_lines))
    ])
    # 计算并保存到 shifts_mm（shifts·毫米）。
    shifts_mm = np.linalg.norm(refined - approximated, axis=1) / pixels_per_mm
    # 判断条件；满足时执行下面代码：float(np.max(shifts_mm)) > max_vertex_shift_mm。
    if float(np.max(shifts_mm)) > max_vertex_shift_mm:
        # 抛出异常，通知上层处理：ValueError(f'拟合顶点最大移动{float(np.max(shifts_mm)):.2f}mm超过限制{max_vertex_…。
        raise ValueError(
            f"拟合顶点最大移动{float(np.max(shifts_mm)):.2f}mm超过限制"
            f"{max_vertex_shift_mm:.2f}mm"
        )
    # 返回结果：计算 refined / pixels_per_mm。
    return refined / pixels_per_mm


# 【函数：piece_polygon_mm】将视觉轮廓简化为3～5顶点的毫米多边形。
# 轮廓可能有上百个边缘点，而求解器只接受少量顶点；这里尝试多边形近似和边线拟合，同时约束面积变化及误差。
# 参数 piece（当前碎片数据）：Piece。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def piece_polygon_mm(piece: Piece, config: dict[str, Any]) -> np.ndarray:
    """将视觉轮廓简化为3～5顶点的毫米多边形。"""
    # 计算并保存到 contour（碎片轮廓点数组）。
    contour = np.asarray(piece.contour, np.float32).reshape(-1, 1, 2)
    # 计算并保存到 contour_area_px（轮廓·面积·像素）。
    contour_area_px = abs(float(cv2.contourArea(contour)))
    # 计算并保存到 perimeter_px（perimeter·像素）。
    perimeter_px = float(cv2.arcLength(contour, True))
    # 判断条件；满足时执行下面代码：contour_area_px <= 1e-06 or perimeter_px <= 1e-06。
    if contour_area_px <= 1e-6 or perimeter_px <= 1e-6:
        # 抛出异常，通知上层处理：ValueError(f'P{piece.piece_id}轮廓面积或周长无效')。
        raise ValueError(f"P{piece.piece_id}轮廓面积或周长无效")

    # 计算并保存到 preferred。
    preferred = float(config.get("generic_polygon_epsilon_ratio", 0.015))
    # 计算并保存到 minimum（允许下限或计算出的最小值）。
    minimum = float(config.get("generic_polygon_epsilon_min_ratio", 0.005))
    # 计算并保存到 maximum（允许上限或计算出的最大值）。
    maximum = float(config.get("generic_polygon_epsilon_max_ratio", 0.05))
    # 判断条件；满足时执行下面代码：not 0.0 < minimum <= preferred <= maximum。
    if not 0.0 < minimum <= preferred <= maximum:
        # 抛出异常，通知上层处理：ValueError('通用轮廓epsilon配置必须满足0 < min <= preferred <= max')。
        raise ValueError("通用轮廓epsilon配置必须满足0 < min <= preferred <= max")

    # 优先尝试配置值，再扫描完整范围；按面积误差选择最忠实的3～5边结果。
    # 计算并保存到 ratios。
    ratios = [preferred]
    # 调用函数：ratios.extend。
    ratios.extend(float(value) for value in np.linspace(minimum, maximum, 31))
    # 计算并保存到 ratios。
    ratios = list(dict.fromkeys(round(value, 8) for value in ratios))
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 min_edge（最小·边）。
    min_edge = float(config.get("generic_min_edge_length_mm", 18.0))
    # 计算并保存到 candidates（待评估的候选集合）。
    candidates: list[tuple[float, float, int, np.ndarray, str]] = []
    # 计算并保存到 observed_vertices。
    observed_vertices: set[int] = set()
    # 计算并保存到 shortest_edge_seen（shortest·边·已记录）。
    shortest_edge_seen = math.inf
    # 计算并保存到 minimum_area（最小·面积）。
    minimum_area = float(config.get("generic_min_piece_area_mm2", 1.0))
    # 计算并保存到 max_vertex_shift（最大·vertex·shift）。
    max_vertex_shift = float(config.get("generic_line_fit_max_vertex_shift_mm", 6.0))

    # 【函数：add_candidate】检查候选多边形的顶点数和质量，再记录满足条件的候选。
    # 参数 polygon_mm（单位为毫米的多边形顶点）：NumPy数组。
    # 参数 ratio（比例）：浮点数。
    # 参数 method（方法）：字符串。
    # 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
    def add_candidate(polygon_mm: np.ndarray, ratio: float, method: str) -> None:
        nonlocal shortest_edge_seen
        # 计算并保存到 raw_polygon（原始·多边形）。
        raw_polygon = np.asarray(polygon_mm, dtype=np.float64).reshape(-1, 2)
        # 执行可能出错的代码，并由后面的异常分支处理错误。
        try:
            # 计算并保存到 normalized（经过本函数规范化处理的数据）。
            normalized = generic_solver.normalize_polygon(
                raw_polygon,
                # 传入命名参数：min_edge_length_mm=min_edge（最小·边·长度·毫米）；取值过程：min_edge（最小·边）。
                min_edge_length_mm=min_edge,
                # 传入命名参数：min_area_mm2=minimum_area（最小·面积·平方毫米）；取值过程：minimum_area（最小·面积）。
                min_area_mm2=minimum_area,
            )
        # 捕获ValueError异常，转入下面的处理代码。
        except ValueError:
            # 计算并保存到 lengths（各条边的长度）。
            lengths = generic_solver.edge_lengths(raw_polygon)
            # 判断条件；满足时执行下面代码：len(lengths)。
            if len(lengths):
                # 计算并保存到 shortest_edge_seen（shortest·边·已记录）。
                shortest_edge_seen = min(shortest_edge_seen, float(np.min(lengths)))
            # 返回结果：无。
            return
        # 计算并保存到 approximated_area_px（approximated·面积·像素）。
        approximated_area_px = generic_solver.polygon_area(normalized) * ppm * ppm
        # 计算并保存到 area_error（面积·误差）。
        area_error = abs(approximated_area_px - contour_area_px) / contour_area_px
        # 计算并保存到 method_priority（方法·priority）。
        method_priority = 0 if method == "approx" else 1
        # 调用函数：candidates.append。
        candidates.append((
            area_error, abs(ratio - preferred), method_priority, normalized, method
        ))

    # 计算并保存到 contour_points（轮廓·点集）。
    contour_points = contour.reshape(-1, 2).astype(np.float64)
    # 遍历数据，逐项处理：ratios。
    for ratio in ratios:
        # 计算并保存到 approximated。
        approximated = cv2.approxPolyDP(contour, ratio * perimeter_px, True)
        # 计算并保存到 vertex_count（vertex·数量）。
        vertex_count = len(approximated)
        # 调用函数：observed_vertices.add。
        observed_vertices.add(vertex_count)
        # 判断条件；满足时执行下面代码：not 3 <= vertex_count <= 5。
        if not 3 <= vertex_count <= 5:
            # 跳过本轮，进入下一轮循环。
            continue
        # 计算并保存到 approximated_points（approximated·点集）。
        approximated_points = approximated.reshape(-1, 2).astype(np.float64)
        # 调用函数：add_candidate。
        add_candidate(approximated_points / ppm, ratio, "approx")
        # 执行可能出错的代码，并由后面的异常分支处理错误。
        try:
            # 计算并保存到 refined_mm（优化后·毫米）。
            refined_mm = refine_polygon_by_edge_lines(
                contour_points,
                approximated_points,
                # 传入命名参数：pixels_per_mm=ppm（每毫米对应的像素数）；取值过程：ppm（每毫米对应的像素数）。
                pixels_per_mm=ppm,
                # 传入命名参数：max_vertex_shift_mm=max_vertex_shift（最大·vertex·shift·毫米）；取值过程：max_vertex_shift（最大·vertex·shift）。
                max_vertex_shift_mm=max_vertex_shift,
            )
            # 计算并保存到 refined_mm（优化后·毫米）。
            refined_mm = remove_near_collinear_vertices(
                refined_mm,
                float(config.get("generic_collinear_angle_tolerance_deg", 15.0)),
            )
        # 捕获ValueError异常，转入下面的处理代码。
        except ValueError:
            # 跳过本轮，进入下一轮循环。
            continue
        # 调用函数：add_candidate。
        add_candidate(refined_mm, ratio, "line_fit")

    # 判断条件；满足时执行下面代码：not candidates。
    if not candidates:
        # 计算并保存到 vertices。
        vertices = ",".join(str(value) for value in sorted(observed_vertices)) or "无"
        # 计算并保存到 edge_message（边·说明）。
        edge_message = ("" if not math.isfinite(shortest_edge_seen) else
                        f"，最短候选边={shortest_edge_seen:.2f}mm")
        # 抛出异常，通知上层处理：ValueError(f'P{piece.piece_id}无法简化为合格的3～5边多边形；尝试得到顶点数={vertices}{edge…。
        raise ValueError(
            f"P{piece.piece_id}无法简化为合格的3～5边多边形；"
            f"尝试得到顶点数={vertices}{edge_message}"
        )

    # 计算并保存到 area_error（面积·误差）、_（此处不需要使用的返回值或循环占位变量）、_（此处不需要使用的返回值或循环占位变量）、polygon（当前多边形的顶点数组）、method（方法）。
    area_error, _, _, polygon, method = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    # 计算并保存到 max_area_error（最大·面积·误差）。
    max_area_error = float(config.get("generic_max_polygon_area_error_ratio", 0.08))
    # 判断条件；满足时执行下面代码：area_error > max_area_error。
    if area_error > max_area_error:
        # 抛出异常，通知上层处理：ValueError(f'P{piece.piece_id}轮廓简化面积误差{area_error:.3f}超过限制{max_area_e…。
        raise ValueError(
            f"P{piece.piece_id}轮廓简化面积误差{area_error:.3f}超过限制{max_area_error:.3f}"
        )
    # 判断条件；满足时执行下面代码：method == 'line_fit'。
    if method == "line_fit":
        # 计算并保存到 raw_errors（原始·errors）。
        raw_errors = [item[0] for item in candidates if item[4] == "approx"]
        # 计算并保存到 before（操作前）。
        before = min(raw_errors) if raw_errors else math.inf
        # 计算并保存到 before_text（操作前·文本）。
        before_text = "无合格原始候选" if not math.isfinite(before) else f"{before:.3f}"
        # 调用函数：print。
        print(
            f"[诊断] P{piece.piece_id}直边拟合：面积误差{before_text}->{area_error:.3f}，"
            f"顶点={len(polygon)}"
        )
    # 返回结果：polygon（当前多边形的顶点数组）。
    return polygon


# 【函数：_point_segment_projection】计算点在线段上的最近投影，投影比例限制在[0,1]。
# 参数 point（当前坐标点）：NumPy数组。
# 参数 start（起始值或起点）：NumPy数组。
# 参数 end（终止值或终点）：NumPy数组。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def _point_segment_projection(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    # 计算并保存到 segment。
    segment = end - start
    # 计算并保存到 length_squared（长度·squared）。
    length_squared = float(np.dot(segment, segment))
    # 判断条件；满足时执行下面代码：length_squared <= 1e-12。
    if length_squared <= 1e-12:
        # 返回结果：调用 start.copy。
        return start.copy()
    # 计算并保存到 ratio（比例）。
    ratio = float(np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0))
    # 返回结果：计算 start + ratio * segment。
    return start + ratio * segment


# 【函数：_polygon_closest_vector】遍历双方顶点到对方线段的投影，寻找最近距离及从第一片指向第二片的向量。
# 参数 first（第一项数据）：NumPy数组。
# 参数 second（第二项数据）：NumPy数组。
# 返回类型：元组（依次为浮点数、NumPy数组）；箭头->是类型提示，不会替你转换实际返回值。
def _polygon_closest_vector(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, np.ndarray]:
    # 计算并保存到 best_distance_squared（最佳·距离·squared）。
    best_distance_squared = math.inf
    # 计算并保存到 best_vector（最佳·向量）。
    best_vector = np.zeros(2, np.float64)
    # 遍历数据，逐项处理：first。
    for point in first:
        # 遍历数据，逐项处理：range(len(second))。
        for index in range(len(second)):
            # 计算并保存到 projected。
            projected = _point_segment_projection(
                point, second[index], second[(index + 1) % len(second)]
            )
            # 计算并保存到 vector（方向或位移向量）。
            vector = projected - point
            # 计算并保存到 distance_squared（距离·squared）。
            distance_squared = float(np.dot(vector, vector))
            # 判断条件；满足时执行下面代码：distance_squared < best_distance_squared。
            if distance_squared < best_distance_squared:
                # 计算并保存到 best_distance_squared（最佳·距离·squared）。
                best_distance_squared = distance_squared
                # 计算并保存到 best_vector（最佳·向量）。
                best_vector = vector
    # 遍历数据，逐项处理：second。
    for point in second:
        # 遍历数据，逐项处理：range(len(first))。
        for index in range(len(first)):
            # 计算并保存到 projected。
            projected = _point_segment_projection(
                point, first[index], first[(index + 1) % len(first)]
            )
            # 计算并保存到 vector（方向或位移向量）。
            vector = point - projected
            # 计算并保存到 distance_squared（距离·squared）。
            distance_squared = float(np.dot(vector, vector))
            # 判断条件；满足时执行下面代码：distance_squared < best_distance_squared。
            if distance_squared < best_distance_squared:
                # 计算并保存到 best_distance_squared（最佳·距离·squared）。
                best_distance_squared = distance_squared
                # 计算并保存到 best_vector（最佳·向量）。
                best_vector = vector
    # 返回结果：创建数据容器。
    return math.sqrt(max(0.0, best_distance_squared)), best_vector


# 【函数：_clearance_adjacencies】用多边形之间的距离找出需要留缝的相邻碎片对。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 tolerance_mm（容差·毫米）：浮点数。
# 返回类型：元组（依次为整数、整数）列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def _clearance_adjacencies(
    polygons: list[np.ndarray],
    tolerance_mm: float,
) -> list[tuple[int, int]]:
    # 计算并保存到 pairs。
    pairs: list[tuple[int, int]] = []
    # 遍历数据，逐项处理：range(len(polygons))。
    for first in range(len(polygons)):
        # 遍历数据，逐项处理：range(first + 1, len(polygons))。
        for second in range(first + 1, len(polygons)):
            # 计算并保存到 distance（当前距离结果）、_（此处不需要使用的返回值或循环占位变量）。
            distance, _ = _polygon_closest_vector(polygons[first], polygons[second])
            # 判断条件；满足时执行下面代码：distance <= tolerance_mm。
            if distance <= tolerance_mm:
                # 调用函数：pairs.append。
                pairs.append((first, second))
    # 返回结果：pairs。
    return pairs


# 【函数：apply_generic_target_clearance】在保持整体矩形外观的前提下，为相邻目标碎片加入受限间隙。
# 先沿各片质心远离矩形中心的方向平移，再迭代拉开仍过近的邻片。每轮去掉平均平移并限制单片最大移动量，最后检查间隙、重叠和纸面边界。
# 参数 solution（求解得到的完整布局）：generic_solver.GenericSolution。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：字典（键为整数，值为元组（依次为浮点数、浮点数））；箭头->是类型提示，不会替你转换实际返回值。
def apply_generic_target_clearance(
    solution: generic_solver.GenericSolution,
    config: dict[str, Any],
) -> dict[int, tuple[float, float]]:
    """在保持整体矩形外观的前提下，为相邻目标碎片加入受限间隙。"""
    # 计算并保存到 piece_count（碎片数量）。
    piece_count = len(solution.poses)
    # 计算并保存到 clearance（希望预留的碎片间隙（毫米））。
    clearance = float(config.get("generic_target_piece_clearance_mm", 0.0))
    # 计算并保存到 rule_limit（rule·上限）。
    rule_limit = float(config.get("target_corresponding_vertex_limit_mm", 20.0))
    # 判断条件；满足时执行下面代码：clearance < 0.0。
    if clearance < 0.0:
        # 抛出异常，通知上层处理：ValueError('目标碎片间隙不能为负数')。
        raise ValueError("目标碎片间隙不能为负数")
    # 判断条件；满足时执行下面代码：rule_limit <= 0.0。
    if rule_limit <= 0.0:
        # 抛出异常，通知上层处理：ValueError('对应顶点距离限制必须大于0')。
        raise ValueError("对应顶点距离限制必须大于0")
    # 判断条件；满足时执行下面代码：clearance > rule_limit + 1e-09。
    if clearance > rule_limit + 1e-9:
        # 抛出异常，通知上层处理：ValueError(f'目标间隙{clearance:.1f}mm超过对应顶点限制{rule_limit:.1f}mm')。
        raise ValueError(
            f"目标间隙{clearance:.1f}mm超过对应顶点限制{rule_limit:.1f}mm"
        )
    # 判断条件；满足时执行下面代码：piece_count <= 1 or clearance <= 1e-09。
    if piece_count <= 1 or clearance <= 1e-9:
        # 返回结果：{pose.piece_index: (0.0, 0.0) for pose in solution.poses}。
        return {pose.piece_index: (0.0, 0.0) for pose in solution.poses}

    # 计算并保存到 poses（各碎片的刚体姿态集合）。
    poses = sorted(solution.poses, key=lambda item: item.piece_index)
    # 计算并保存到 base_polygons（base·多边形集合）。
    base_polygons = [pose.target_polygon_mm.copy() for pose in poses]
    # 计算并保存到 rectangle_center（rectangle·中心）。
    rectangle_center = (
        np.asarray(solution.target_top_left_mm, np.float64)
        + np.asarray(solution.target_size_mm, np.float64) / 2.0
    )
    # 计算并保存到 initial_offset（初始·偏移）。
    initial_offset = clearance / 2.0
    # 计算并保存到 shifts（每块碎片的平移量集合）。
    shifts = np.zeros((piece_count, 2), np.float64)
    # 遍历数据，逐项处理：enumerate(base_polygons)。
    for index, polygon in enumerate(base_polygons):
        # 计算并保存到 direction（方向标记或方向向量）。
        direction = polygon_centroid(polygon) - rectangle_center
        # 计算并保存到 length（长度）。
        length = float(np.linalg.norm(direction))
        # 计算并保存到 direction（方向标记或方向向量）。
        direction = (np.asarray([1.0, 0.0], np.float64)
                     # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
                     if length <= 1e-9 else direction / length)
        # 计算并保存到 shifts[index]。
        shifts[index] = direction * initial_offset

    # 计算并保存到 adjacency_tolerance（邻接·容差）。
    adjacency_tolerance = float(
        config.get("generic_clearance_adjacency_tolerance_mm", 3.0)
    )
    # 计算并保存到 adjacent_pairs（相邻·pairs）。
    adjacent_pairs = {
        tuple(sorted((match.piece_a, match.piece_b))) for match in solution.matches
    }
    # 求解器拼缝与几何实际接触共同定义相邻关系，避免漏掉闭合边。
    # 调用函数：adjacent_pairs.update。
    adjacent_pairs.update(_clearance_adjacencies(base_polygons, adjacency_tolerance))
    # 计算并保存到 adjacent_pairs（相邻·pairs）。
    adjacent_pairs = sorted(adjacent_pairs)

    # 计算并保存到 minimum_gap_ratio（最小·缺口·比例）。
    minimum_gap_ratio = float(config.get("generic_clearance_min_achieved_ratio", 0.8))
    # 判断条件；满足时执行下面代码：not 0.0 < minimum_gap_ratio <= 1.0。
    if not (0.0 < minimum_gap_ratio <= 1.0):
        # 抛出异常，通知上层处理：ValueError('最小间隙达成比例必须位于(0,1]范围')。
        raise ValueError("最小间隙达成比例必须位于(0,1]范围")
    # 计算并保存到 minimum_gap（最小·缺口）。
    minimum_gap = clearance * minimum_gap_ratio
    # 计算并保存到 max_shift_ratio（最大·shift·比例）。
    max_shift_ratio = float(config.get("generic_clearance_max_piece_shift_ratio", 0.9))
    # 判断条件；满足时执行下面代码：max_shift_ratio < 0.5。
    if max_shift_ratio < 0.5:
        # 抛出异常，通知上层处理：ValueError('单片最大位移比例不能小于0.5')。
        raise ValueError("单片最大位移比例不能小于0.5")
    # 计算并保存到 absolute_shift_limit（absolute·shift·上限）。
    absolute_shift_limit = float(
        config.get("generic_clearance_max_piece_shift_mm", rule_limit / 2.0)
    )
    # 判断条件；满足时执行下面代码：absolute_shift_limit <= 0.0。
    if absolute_shift_limit <= 0.0:
        # 抛出异常，通知上层处理：ValueError('????????????0')。
        raise ValueError("????????????0")
    # 计算并保存到 max_piece_shift（最大·碎片·shift）。
    max_piece_shift = min(absolute_shift_limit, clearance * max_shift_ratio)
    # 判断条件；满足时执行下面代码：max_piece_shift + 1e-09 < initial_offset。
    if max_piece_shift + 1e-9 < initial_offset:
        # 抛出异常，通知上层处理：ValueError('单片最大位移小于初始外移距离')。
        raise ValueError("单片最大位移小于初始外移距离")

    # 计算并保存到 iterations（迭代次数）。
    iterations = int(config.get("generic_clearance_relax_iterations", 600))
    # 遍历数据，逐项处理：range(max(1, iterations))。
    for _ in range(max(1, iterations)):
        # 计算并保存到 changed。
        changed = False
        # 遍历数据，逐项处理：adjacent_pairs。
        for first, second in adjacent_pairs:
            # 计算并保存到 first_polygon（第一·多边形）。
            first_polygon = base_polygons[first] + shifts[first]
            # 计算并保存到 second_polygon（第二·多边形）。
            second_polygon = base_polygons[second] + shifts[second]
            # 计算并保存到 distance（当前距离结果）、vector（方向或位移向量）。
            distance, vector = _polygon_closest_vector(first_polygon, second_polygon)
            # 计算并保存到 deficit。
            deficit = minimum_gap - distance
            # 判断条件；满足时执行下面代码：deficit <= 0.0001。
            if deficit <= 1e-4:
                # 跳过本轮，进入下一轮循环。
                continue
            # 计算并保存到 vector_length（向量·长度）。
            vector_length = float(np.linalg.norm(vector))
            # 判断条件；满足时执行下面代码：vector_length <= 1e-08。
            if vector_length <= 1e-8:
                # 计算并保存到 vector（方向或位移向量）。
                vector = polygon_centroid(second_polygon) - polygon_centroid(first_polygon)
                # 计算并保存到 vector_length（向量·长度）。
                vector_length = float(np.linalg.norm(vector))
            # 判断条件；满足时执行下面代码：vector_length <= 1e-08。
            if vector_length <= 1e-8:
                # 跳过本轮，进入下一轮循环。
                continue
            # 计算并保存到 correction。
            correction = vector / vector_length * (deficit * 0.25)
            # 更新变量：shifts[first]。
            shifts[first] -= correction
            # 更新变量：shifts[second]。
            shifts[second] += correction
            # 计算并保存到 changed。
            changed = True

        # 更新变量：shifts（每块碎片的平移量集合）。
        shifts -= np.mean(shifts, axis=0)
        # 遍历数据，逐项处理：enumerate(shifts)。
        for index, shift in enumerate(shifts):
            # 计算并保存到 length（长度）。
            length = float(np.linalg.norm(shift))
            # 判断条件；满足时执行下面代码：length > max_piece_shift。
            if length > max_piece_shift:
                # 计算并保存到 shifts[index]。
                shifts[index] = shift * (max_piece_shift / length)
        # 判断条件；满足时执行下面代码：not changed。
        if not changed:
            # 立即结束当前循环。
            break

    # 计算并保存到 shifted_polygons（shifted·多边形集合）。
    shifted_polygons = [
        polygon + shifts[index] for index, polygon in enumerate(base_polygons)
    ]
    # 计算并保存到 actual_gaps（实测·gaps）。
    actual_gaps = [
        _polygon_closest_vector(shifted_polygons[first], shifted_polygons[second])[0]
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for first, second in adjacent_pairs
    ]
    # 计算并保存到 actual_minimum_gap（实测·最小·缺口）。
    actual_minimum_gap = min(actual_gaps, default=minimum_gap)
    # 判断条件；满足时执行下面代码：actual_minimum_gap + 0.02 < minimum_gap。
    if actual_minimum_gap + 0.02 < minimum_gap:
        # 抛出异常，通知上层处理：ValueError(f'受限松弛达到单片最大位移{max_piece_shift:.1f}mm后仍未满足间隙：实际最小间隙{actual…。
        raise ValueError(
            f"受限松弛达到单片最大位移{max_piece_shift:.1f}mm后仍未满足间隙："
            f"实际最小间隙{actual_minimum_gap:.1f}mm，小于要求{minimum_gap:.1f}mm"
        )

    # 遍历数据，逐项处理：adjacent_pairs。
    for first, second in adjacent_pairs:
        # 计算并保存到 corresponding_distance（corresponding·距离）。
        corresponding_distance = float(np.linalg.norm(shifts[second] - shifts[first]))
        # 判断条件；满足时执行下面代码：corresponding_distance > rule_limit + 1e-06。
        if corresponding_distance > rule_limit + 1e-6:
            # 抛出异常，通知上层处理：ValueError(f'留缝后对应顶点距离{corresponding_distance:.1f}mm超过限制{rule_limit:.…。
            raise ValueError(
                f"留缝后对应顶点距离{corresponding_distance:.1f}mm超过限制{rule_limit:.1f}mm"
            )

    # 计算并保存到 overlap（重叠量或重叠比例）。
    overlap = generic_solver.polygon_overlap_ratio(
        shifted_polygons,
        float(config.get("generic_clearance_check_pixels_per_mm", 5.0)),
    )
    # 计算并保存到 overlap_limit（重叠·上限）。
    overlap_limit = float(config.get("generic_clearance_max_overlap_ratio", 0.001))
    # 判断条件；满足时执行下面代码：overlap > overlap_limit。
    if overlap > overlap_limit:
        # 抛出异常，通知上层处理：ValueError(f'留缝后目标碎片重叠率超限：实际={overlap:.4f}，限制={overlap_limit:.4f}')。
        raise ValueError(
            f"留缝后目标碎片重叠率超限：实际={overlap:.4f}，限制={overlap_limit:.4f}"
        )

    # 计算并保存到 bounds。
    bounds = np.vstack(shifted_polygons)
    # 计算并保存到 minimum（允许下限或计算出的最小值）。
    minimum = np.min(bounds, axis=0)
    # 计算并保存到 maximum（允许上限或计算出的最大值）。
    maximum = np.max(bounds, axis=0)
    # 计算并保存到 paper_width（纸面宽度（毫米））。
    paper_width = float(config.get("paper_width_mm", 210.0))
    # 计算并保存到 paper_height（纸面高度（毫米））。
    paper_height = float(config.get("paper_height_mm", 297.0))
    # 计算并保存到 target_top（目标·上边）。
    target_top = float(config.get("target_region_top_mm", paper_height / 2.0))
    # 判断条件；满足时执行下面代码：minimum[0] < -1e-06 or minimum[1] < target_top - 1e-06 or maximum[0] > paper_wi…。
    if (minimum[0] < -1e-6 or minimum[1] < target_top - 1e-6 or
            maximum[0] > paper_width + 1e-6 or maximum[1] > paper_height + 1e-6):
        # 抛出异常，通知上层处理：ValueError('留缝后的目标布局超出A4工作区')。
        raise ValueError("留缝后的目标布局超出A4工作区")

    # 计算并保存到 result（当前步骤得到的结果）。
    result: dict[int, tuple[float, float]] = {}
    # 遍历数据，逐项处理：enumerate(poses)。
    for index, pose in enumerate(poses):
        # 计算并保存到 shift（一次平移量）。
        shift = shifts[index]
        # 计算并保存到 pose.transform_3x3（3×3齐次变换矩阵（把源坐标映射到目标坐标））。
        pose.transform_3x3 = generic_solver.rigid_matrix(0.0, shift) @ pose.transform_3x3
        # 计算并保存到 pose.target_polygon_mm（目标布局中的多边形顶点（毫米））。
        pose.target_polygon_mm = shifted_polygons[index]
        # 计算并保存到 result[pose.piece_index]。
        result[pose.piece_index] = (float(shift[0]), float(shift[1]))

    # 计算并保存到 maximum_shift（最大·shift）。
    maximum_shift = max(float(np.linalg.norm(shift)) for shift in shifts)
    # 调用函数：print。
    print(
        f"[布局] 配置间隙={clearance:.1f}mm，真实轮廓最小间隙={actual_minimum_gap:.1f}mm，"
        f"最大单片偏移={maximum_shift:.1f}/{max_piece_shift:.1f}mm，"
        f"对应顶点限制={rule_limit:.1f}mm，重叠率={overlap:.4f}"
    )
    # 返回结果：result（当前步骤得到的结果）。
    return result

# 【函数：fixed_target_clearance_shifts】将固定图2模板包装为通用布局，复用留缝算法并返回各模板平移量。
# 本函数原文有乱码。实际是复用通用留缝函数：要求至少达到配置间隙的90%，单片移动上限设为配置间隙。
# 参数 target_templates（目标·templates）：字典（键为字符串，值为NumPy数组）。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：字典（键为字符串，值为元组（依次为浮点数、浮点数））；箭头->是类型提示，不会替你转换实际返回值。
def fixed_target_clearance_shifts(
    target_templates: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    """????2???????????????????????"""
    # 计算并保存到 template_ids（template·编号集合）。
    template_ids = list(FIGURE2_TEMPLATE_POLYGONS_MM)
    # 计算并保存到 clearance（希望预留的碎片间隙（毫米））。
    clearance = float(config.get("fixed_target_piece_clearance_mm", 0.0))
    # 判断条件；满足时执行下面代码：clearance <= 1e-09。
    if clearance <= 1e-9:
        # 返回结果：{template_id: (0.0, 0.0) for template_id in template_ids}。
        return {template_id: (0.0, 0.0) for template_id in template_ids}

    # 计算并保存到 top_left（上边·左边）。
    top_left = tuple(float(value) for value in
                     config.get("target_rectangle_top_left_mm", [55.0, 205.0]))
    # 计算并保存到 size（尺寸数据）。
    size = tuple(float(value) for value in
                 config.get("target_rectangle_size_mm", [100.0, 60.0]))
    # 计算并保存到 poses（各碎片的刚体姿态集合）。
    poses = [
        # 传入命名参数：dtype=np.float64（指定数组元素类型）；取值过程：np.float64。
        generic_solver.GenericPiecePose(index, np.eye(3, dtype=np.float64),
                                        target_templates[template_id].copy())
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for index, template_id in enumerate(template_ids)
    ]
    # 计算并保存到 fixed_solution（固定·solution）。
    fixed_solution = generic_solver.GenericSolution(
        # 传入命名参数：poses=poses（各碎片的刚体姿态集合）；取值过程：poses（各碎片的刚体姿态集合）。
        poses=poses,
        # 传入命名参数：target_size_mm=(size[0], size[1])（目标宽高（毫米））；取值过程：创建数据容器。
        target_size_mm=(size[0], size[1]),
        # 传入命名参数：target_top_left_mm=(top_left[0], top_left[1])（目标区域左上角坐标（毫米））；取值过程：创建数据容器。
        target_top_left_mm=(top_left[0], top_left[1]),
        # 传入命名参数：score=0.0（评分）；取值过程：0.0；fill_error_ratio=0.0（矩形填充误差比例）；取值过程：0.0；overlap_ratio=0.0（碎片间重叠比例）；
        # 取值过程：0.0。
        score=0.0, fill_error_ratio=0.0, overlap_ratio=0.0,
        # 传入命名参数：endpoint_error_mm=0.0（拼缝对应端点的误差（毫米））；取值过程：0.0；boundary_error_mm=0.0（目标矩形边界误差（毫米））；取值过程：0.0；
        # boundary_gap_ratio=0.0（矩形边界缺口比例）；取值过程：0.0。
        endpoint_error_mm=0.0, boundary_error_mm=0.0, boundary_gap_ratio=0.0,
        # 传入命名参数：matches=[]（匹配记录列表）；取值过程：创建数据容器；search_states=0（已检查的搜索状态数）；取值过程：0。
        matches=[], search_states=0,
    )
    # 计算并保存到 clearance_config（留缝·配置）。
    clearance_config = dict(config)
    # 计算并保存到 clearance_config['generic_target_piece_clearance_mm']。
    clearance_config["generic_target_piece_clearance_mm"] = clearance
    # ?2?????????????10mm??20mm?????????
    # ????????????????90%?
    # 计算并保存到 clearance_config['generic_clearance_min_achieved_ratio']。
    clearance_config["generic_clearance_min_achieved_ratio"] = 0.9
    # 计算并保存到 clearance_config['generic_clearance_max_piece_shift_ratio']。
    clearance_config["generic_clearance_max_piece_shift_ratio"] = 1.0
    # 计算并保存到 clearance_config['generic_clearance_max_piece_shift_mm']。
    clearance_config["generic_clearance_max_piece_shift_mm"] = clearance
    # 计算并保存到 shifts（每块碎片的平移量集合）。
    shifts = apply_generic_target_clearance(fixed_solution, clearance_config)
    # 返回结果：{template_id: shifts[index] for index, template_id in enumerate(template_ids)}。
    return {template_id: shifts[index] for index, template_id in enumerate(template_ids)}


# 【函数：verify_edge_patterns】方案B：沿拼缝边提取牌面花纹做NCC互相关，验证几何拼法花纹对应正确。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 pieces（碎片列表）：Piece列表/序列。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 solution（求解得到的完整布局）：generic_solver.GenericSolution。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：dict或None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def verify_edge_patterns(paper: np.ndarray, pieces: list[Piece],
                         polygons: list[np.ndarray],
                         solution: generic_solver.GenericSolution,
                         config: dict[str, Any]) -> dict | None:
    """方案B：沿拼缝边提取牌面花纹做NCC互相关，验证几何拼法花纹对应正确。

    拼缝边对直接取自求解器solution.matches（EdgeMatch含piece_a/edge_a/
    piece_b/edge_b，索引均为源多边形），不受目标布局间隙影响；花纹在源
    碎片多边形上采样（未拼合时牌面完整）。刚体变换保持边顺序不变，因此
    源边索引可直接与拼缝边对应。返回汇总dict；无拼缝或缺少edge_matcher
    时返回None。
    """
    # 判断条件；满足时执行下面代码：not _PATTERN_MATCHER_AVAILABLE or paper is None。
    if not _PATTERN_MATCHER_AVAILABLE or paper is None:
        # 返回结果：None。
        return None
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 spacing（间隔）。
    spacing = float(config.get("poker_edge_sample_spacing_mm", 1.0))
    # 计算并保存到 offset（偏移）。
    offset = float(config.get("poker_edge_offset_mm", 2.0))

    # 计算并保存到 poly_by_index（poly·by·索引）。
    poly_by_index = {index: np.asarray(poly, np.float64)
                     # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
                     for index, poly in enumerate(polygons)}
    # 计算并保存到 id_by_index（编号·by·索引）。
    id_by_index = {index: piece.piece_id for index, piece in enumerate(pieces)}
    # 计算并保存到 profiles（各条边的采样记录）。
    profiles: dict[int, list] = {}
    # 遍历数据，逐项处理：enumerate(pieces)。
    for index, piece in enumerate(pieces):
        # 计算并保存到 poly_mm（poly·毫米）。
        poly_mm = poly_by_index.get(index)
        # 判断条件；满足时执行下面代码：poly_mm is None。
        if poly_mm is None:
            # 跳过本轮，进入下一轮循环。
            continue
        # 执行可能出错的代码，并由后面的异常分支处理错误。
        try:
            # 计算并保存到 profiles[index]。
            profiles[index] = edge_matcher.extract_edge_profiles(
                paper, piece.piece_id, poly_mm, ppm,
                # 传入命名参数：sample_spacing_mm=spacing（样本·间隔·毫米）；取值过程：spacing（间隔）；offset_mm=offset（偏移·毫米）；取值过程：offset（偏移）。
                sample_spacing_mm=spacing, offset_mm=offset,
            )
        # 捕获Exception异常，转入下面的处理代码；异常对象保存在exc。
        except Exception as exc:
            # 调用函数：print。
            print(f"[花纹] P{piece.piece_id}边缘花纹提取失败：{exc}")

    # 计算并保存到 pairs。
    pairs: list[dict[str, Any]] = []
    # 计算并保存到 similarities。
    similarities: list[float] = []
    # 遍历数据，逐项处理：solution.matches。
    for edge_match in solution.matches:
        # 计算并保存到 profs_a。
        profs_a = profiles.get(edge_match.piece_a, [])
        # 计算并保存到 profs_b。
        profs_b = profiles.get(edge_match.piece_b, [])
        # 判断条件；满足时执行下面代码：edge_match.edge_a < len(profs_a) and edge_match.edge_b < len(profs_b)。
        if edge_match.edge_a < len(profs_a) and edge_match.edge_b < len(profs_b):
            # 计算并保存到 ncc（归一化互相关值，通常在[-1,1]之间）。
            ncc = edge_matcher.profile_ncc(
                profs_a[edge_match.edge_a].profile_gray,
                profs_b[edge_match.edge_b].profile_gray,
            )
        # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
        else:
            # 计算并保存到 ncc（归一化互相关值，通常在[-1,1]之间）。
            ncc = 0.0
        # 计算并保存到 similarity（由NCC换算到[0,1]的相似度）。
        similarity = float((ncc + 1.0) / 2.0)
        # 调用函数：similarities.append。
        similarities.append(similarity)
        # 计算并保存到 shared_mm（两边共用部分的长度（毫米））。
        shared_mm = 0.0
        # 执行可能出错的代码，并由后面的异常分支处理错误。
        try:
            # 计算并保存到 lengths（各条边的长度）。
            lengths = generic_solver.edge_lengths(poly_by_index[edge_match.piece_a])
            # 判断条件；满足时执行下面代码：edge_match.edge_a < len(lengths)。
            if edge_match.edge_a < len(lengths):
                # 计算并保存到 shared_mm（两边共用部分的长度（毫米））。
                shared_mm = float(lengths[edge_match.edge_a])
        # 捕获Exception异常，转入下面的处理代码。
        except Exception:
            pass
        # 调用函数：pairs.append。
        pairs.append({
            # 字典字段'piece_a'（第一块碎片的编号或索引）：调用 id_by_index.get。
            "piece_a": id_by_index.get(edge_match.piece_a, edge_match.piece_a),
            # 字典字段'edge_a'（第一块碎片的边索引）：edge_match.edge_a（边·match.edge·a）。
            "edge_a": edge_match.edge_a,
            # 字典字段'piece_b'（第二块碎片的编号或索引）：调用 id_by_index.get。
            "piece_b": id_by_index.get(edge_match.piece_b, edge_match.piece_b),
            # 字典字段'edge_b'（第二块碎片的边索引）：edge_match.edge_b（边·match.edge·b）。
            "edge_b": edge_match.edge_b,
            # 字典字段'shared_mm'（两边共用部分的长度（毫米））：调用 round。
            "shared_mm": round(shared_mm, 2),
            # 字典字段'ncc'（归一化互相关值，通常在[-1,1]之间）：调用 round。
            "ncc": round(float(ncc), 4),
            # 字典字段'similarity'（由NCC换算到[0,1]的相似度）：调用 round。
            "similarity": round(similarity, 4),
        })

    # 判断条件；满足时执行下面代码：not similarities。
    if not similarities:
        # 返回结果：None。
        return None
    # 计算并保存到 average。
    average = float(np.mean(similarities))
    # 计算并保存到 minimum（允许下限或计算出的最小值）。
    minimum = float(np.min(similarities))
    # 判断条件；满足时执行下面代码：average >= 0.85。
    if average >= 0.85:
        # 计算并保存到 message（说明）。
        message = f"花纹吻合度: {average:.0%} (高度匹配)"
    # 判断条件；满足时执行下面代码：average >= 0.65。
    elif average >= 0.65:
        # 计算并保存到 message（说明）。
        message = f"花纹吻合度: {average:.0%} (中度匹配)"
    # 判断条件；满足时执行下面代码：average >= 0.5。
    elif average >= 0.50:
        # 计算并保存到 message（说明）。
        message = f"花纹吻合度: {average:.0%} (偏低，请核实)"
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 计算并保存到 message（说明）。
        message = f"花纹吻合度: {average:.0%} (不匹配，拼法可能错误)"
    # 返回结果：创建结果字典。
    return {
        # 字典字段'score'（评分）：调用 round。
        "score": round(average, 4),
        # 字典字段'min_similarity'（最小·similarity）：调用 round。
        "min_similarity": round(minimum, 4),
        # 字典字段'n_pairs'（n_pairs）：调用 len。
        "n_pairs": len(pairs),
        # 字典字段'pairs'（pairs）：pairs。
        "pairs": pairs,
        # 字典字段'message'（说明）：message（说明）。
        "message": message,
    }

# 【函数：solve_generic_puzzle】求解1～4块任意多边形，并转换为现有路径模块可消费的匹配结果。
# 磁吸点与多边形必须使用同一个刚体矩阵，否则碎片轮廓摆对了，机械臂却可能把吸取点送到错误位置。
# 参数 pieces（碎片列表）：Piece列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组或None（不返回业务结果）；省略时使用None。
# 返回类型：PuzzleMatch列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def solve_generic_puzzle(pieces: list[Piece], config: dict[str, Any],
                         paper: np.ndarray | None = None) -> list[PuzzleMatch]:
    """求解1～4块任意多边形，并转换为现有路径模块可消费的匹配结果。

    paper为校正后的A4彩色图，仅当config启用poker_pattern_verification时
    用于对几何解做相邻边花纹NCC验证（扑克牌模式）。
    """
    # 判断条件；满足时执行下面代码：not 1 <= len(pieces) <= int(config.get('max_pieces', 4))。
    if not 1 <= len(pieces) <= int(config.get("max_pieces", 4)):
        # 抛出异常，通知上层处理：ValueError(f'通用拼图需要识别1～4块，当前识别到{len(pieces)}块')。
        raise ValueError(f"通用拼图需要识别1～4块，当前识别到{len(pieces)}块")
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 source_limit（源·上限）。
    source_limit = (float(config.get("target_region_top_mm", 148.5)) +
                    float(config.get("source_center_tolerance_mm", 5.0)))
    # 计算并保存到 invalid_sources。
    invalid_sources = [piece.piece_id for piece in pieces
                       # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
                       if piece.center_mm(ppm)[1] > source_limit]
    # 判断条件；满足时执行下面代码：invalid_sources。
    if invalid_sources:
        # 计算并保存到 ids（编号集合）。
        ids = ",".join(f"P{piece_id}" for piece_id in invalid_sources)
        # 抛出异常，通知上层处理：ValueError(f'{ids}的中心已进入A4下半区，请把初始碎片主体放回上半区域')。
        raise ValueError(f"{ids}的中心已进入A4下半区，请把初始碎片主体放回上半区域")

    # 计算并保存到 polygons（各块碎片的多边形集合）。
    polygons = [piece_polygon_mm(piece, config) for piece in pieces]
    # 计算并保存到 solution（求解得到的完整布局）。
    solution = generic_solver.solve_generic_rectangle(polygons, config)
    # 终极兜底解（碎片已各自平铺、互不重叠）不需要再做留缝处理，
    # 直接给零位移，避免对非拼合布局错误地施加间隙。
    # 判断条件；满足时执行下面代码：solution.relaxed_override and solution.score > 100000000.0。
    if solution.relaxed_override and solution.score > 1e8:
        # 计算并保存到 clearance_shifts（留缝·shifts）。
        clearance_shifts = {index: (0.0, 0.0) for index in range(len(pieces))}
        # 调用函数：print。
        print("[终极兜底] 平铺解跳过留缝处理（碎片已独立放置）")
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 计算并保存到 clearance_shifts（留缝·shifts）。
        clearance_shifts = apply_generic_target_clearance(solution, config)
    # 计算并保存到 poses（各碎片的刚体姿态集合）。
    poses = {pose.piece_index: pose for pose in solution.poses}
    # 计算并保存到 matches（匹配记录列表）。
    matches: list[PuzzleMatch] = []
    # 遍历数据，逐项处理：enumerate(pieces)。
    for index, piece in enumerate(pieces):
        # 计算并保存到 pose（单块碎片的刚体姿态）。
        pose = poses[index]
        # 计算并保存到 source_pick（源·磁吸点）。
        source_pick = np.asarray([piece.pick_mm(ppm)], np.float64)
        # 计算并保存到 target_pick（目标·磁吸点）。
        target_pick = generic_solver.transform_points(source_pick, pose.transform_3x3)[0]
        # 计算并保存到 inside。
        inside = cv2.pointPolygonTest(
            pose.target_polygon_mm.astype(np.float32),
            (float(target_pick[0]), float(target_pick[1])), False,
        ) >= 0
        # 终极兜底解是"平铺不拼合"，磁吸点随碎片整体平移，
        # 不做磁吸点几何校验，避免质量无关的误拦截。
        # 判断条件；满足时执行下面代码：not inside and (not (solution.relaxed_override and solution.score > 100000000.0…。
        if not inside and not (solution.relaxed_override and solution.score > 1e8):
            # 抛出异常，通知上层处理：ValueError(f'P{piece.piece_id}变换后的磁吸点不在目标多边形内')。
            raise ValueError(f"P{piece.piece_id}变换后的磁吸点不在目标多边形内")
        # 计算并保存到 angle（当前计算或搜索的角度）。
        angle = normalized_rotation(generic_solver.rotation_angle_deg(pose.transform_3x3))
        # 计算并保存到 match（当前匹配记录）。
        match = PuzzleMatch(
            # 传入命名参数：piece=piece（当前碎片数据）；取值过程：piece（当前碎片数据）。
            piece=piece,
            # 传入命名参数：template_id=f'G{index + 1}'（目标模板编号）；取值过程：f'G{index + 1}'。
            template_id=f"G{index + 1}",
            # 传入命名参数：target_polygon_mm=pose.target_polygon_mm（目标布局中的多边形顶点（毫米））；
            # 取值过程：pose.target_polygon_mm（pose.target·多边形·毫米）。
            target_polygon_mm=pose.target_polygon_mm,
            # 传入命名参数：target_pick_mm=(float(target_pick[0]), float(target_pick[1]))（放置后磁吸点的目标坐标（毫米））；取值过程：创建数据容器。
            target_pick_mm=(float(target_pick[0]), float(target_pick[1])),
            # 传入命名参数：rotation_deg=angle（碎片需要旋转的有符号角度（度））；取值过程：angle（当前计算或搜索的角度）。
            rotation_deg=angle,
            # 传入命名参数：rotation_direction=rotation_direction(angle)（旋转方向标记）；取值过程：根据角度符号输出CW或CCW，死区内输出NONE。
            rotation_direction=rotation_direction(angle),
            # 传入命名参数：area_error_ratio=solution.fill_error_ratio（面积相对误差）；
            # 取值过程：solution.fill_error_ratio（solution.fill·误差·比例）。
            area_error_ratio=solution.fill_error_ratio,
            # 传入命名参数：shape_distance=solution.boundary_error_mm（形状差异指标）；
            # 取值过程：solution.boundary_error_mm（solution.boundary·误差·毫米）。
            shape_distance=solution.boundary_error_mm,
            # 传入命名参数：match_iou=max(0.0, 1.0 - solution.fill_error_ratio)（形状匹配的交并比）；取值过程：调用 max。
            match_iou=max(0.0, 1.0 - solution.fill_error_ratio),
            # 传入命名参数：fit_scale=1.0（形状匹配时的尺度比；刚体搬运应保持为1）；取值过程：1.0。
            fit_scale=1.0,
            # 传入命名参数：status='ok'（处理结果的状态标记）；取值过程：'ok'。
            status="ok",
            # 传入命名参数：solver_mode='generic_geometry'（求解方式标记）；取值过程：'generic_geometry'。
            solver_mode="generic_geometry",
            # 传入命名参数：transform_3x3=pose.transform_3x3.copy()（3×3齐次变换矩阵（把源坐标映射到目标坐标））；取值过程：调用 pose.transform_3x3.copy。
            transform_3x3=pose.transform_3x3.copy(),
            # 传入命名参数：solution_score=solution.score（几何方案的代价评分，通常越小越好）；取值过程：solution.score。
            solution_score=solution.score,
            # 传入命名参数：fill_error_ratio=solution.fill_error_ratio（矩形填充误差比例）；
            # 取值过程：solution.fill_error_ratio（solution.fill·误差·比例）。
            fill_error_ratio=solution.fill_error_ratio,
            # 传入命名参数：overlap_ratio=solution.overlap_ratio（碎片间重叠比例）；取值过程：solution.overlap_ratio（solution.overlap·比例）。
            overlap_ratio=solution.overlap_ratio,
            # 传入命名参数：endpoint_error_mm=solution.endpoint_error_mm（拼缝对应端点的误差（毫米））；
            # 取值过程：solution.endpoint_error_mm（solution.endpoint·误差·毫米）。
            endpoint_error_mm=solution.endpoint_error_mm,
            # 传入命名参数：boundary_error_mm=solution.boundary_error_mm（目标矩形边界误差（毫米））；
            # 取值过程：solution.boundary_error_mm（solution.boundary·误差·毫米）。
            boundary_error_mm=solution.boundary_error_mm,
            # 传入命名参数：boundary_gap_ratio=solution.boundary_gap_ratio（矩形边界缺口比例）；
            # 取值过程：solution.boundary_gap_ratio（solution.boundary·缺口·比例）。
            boundary_gap_ratio=solution.boundary_gap_ratio,
            # 传入命名参数：target_rectangle_top_left_mm=solution.target_top_left_mm（目标矩形左上角坐标（毫米））；
            # 取值过程：solution.target_top_left_mm（solution.target·上边·左边·毫米）。
            target_rectangle_top_left_mm=solution.target_top_left_mm,
            # 传入命名参数：target_rectangle_size_mm=solution.target_size_mm（目标矩形宽高（毫米））；
            # 取值过程：solution.target_size_mm（solution.target·尺寸·毫米）。
            target_rectangle_size_mm=solution.target_size_mm,
            # 传入命名参数：target_clearance_shift_mm=clearance_shifts[index]（为留出碎片间隙施加的平移（毫米））；取值过程：clearance_shifts[index]。
            target_clearance_shift_mm=clearance_shifts[index],
            # 传入命名参数：relaxed_override=solution.relaxed_override（是否采用放宽质量要求的兜底结果）；取值过程：solution.relaxed_override。
            relaxed_override=solution.relaxed_override,
        )
        # 调用函数：matches.append。
        matches.append(match)
        # 调用函数：print。
        print(f"[通用求解] P{piece.piece_id}->{match.template_id}: "
              f"旋转={angle:+.2f}deg({match.rotation_direction}), "
              f"顶点={len(polygons[index])}, 状态=ok")
    # 调用函数：print。
    print(f"[通用求解] 矩形={solution.target_size_mm[0]:.1f}×"
          f"{solution.target_size_mm[1]:.1f}mm, 评分={solution.score:.4f}, "
          f"填充误差={solution.fill_error_ratio:.4f}, "
          f"边界缺口={solution.boundary_gap_ratio:.4f}, "
          f"重叠率={solution.overlap_ratio:.4f}, 搜索状态={solution.search_states}")

    # 判断条件；满足时执行下面代码：paper is not None and bool(config.get('poker_pattern_verification', False))。
    if paper is not None and bool(config.get("poker_pattern_verification", False)):
        # 计算并保存到 verification（复核）。
        verification = verify_edge_patterns(paper, pieces, polygons, solution, config)
        # 判断条件；满足时执行下面代码：verification is not None。
        if verification is not None:
            # 遍历数据，逐项处理：matches。
            for match in matches:
                # 计算并保存到 match.pattern_verification（花纹复核结果，None表示没有该结果）。
                match.pattern_verification = verification
            # 调用函数：print。
            print(f"[花纹] 拼缝{verification['n_pairs']}对，"
                  f"相似度avg={verification['score']:.3f} "
                  f"min={verification['min_similarity']:.3f} {verification['message']}")
    # 返回结果：matches（匹配记录列表）。
    return matches


# 【函数：solve_puzzle】按配置分发固定图2或通用几何求解器。
# 参数 pieces（碎片列表）：Piece列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组或None（不返回业务结果）；省略时使用None。
# 返回类型：PuzzleMatch列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def solve_puzzle(pieces: list[Piece], config: dict[str, Any],
                 paper: np.ndarray | None = None) -> list[PuzzleMatch]:
    """按配置分发固定图2或通用几何求解器。"""
    # 计算并保存到 mode（模式）。
    mode = str(config.get("puzzle_mode", "fixed_figure_2"))
    # 判断条件；满足时执行下面代码：mode == 'fixed_figure_2'。
    if mode == "fixed_figure_2":
        # 返回结果：调用 solve_fixed_puzzle。
        return solve_fixed_puzzle(pieces, config)
    # 判断条件；满足时执行下面代码：mode == 'generic_geometry'。
    if mode == "generic_geometry":
        # 返回结果：调用 solve_generic_puzzle。
        return solve_generic_puzzle(pieces, config, paper)
    # 抛出异常，通知上层处理：ValueError(f'未知puzzle_mode：{mode}')。
    raise ValueError(f"未知puzzle_mode：{mode}")


# 【函数：mm_to_px】毫米坐标乘ppm，得到像素坐标。
# 参数 point（当前坐标点）：元组（依次为浮点数、浮点数）或浮点数列表/序列。
# 参数 ppm（每毫米对应的像素数）：浮点数。
# 返回类型：元组（依次为浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def mm_to_px(point: tuple[float, float] | list[float], ppm: float) -> tuple[float, float]:
    # 返回结果：创建数据容器。
    return float(point[0]) * ppm, float(point[1]) * ppm


# 【函数：px_to_mm】像素坐标除以ppm，得到毫米坐标。
# 参数 point（当前坐标点）：元组（依次为浮点数、浮点数）。
# 参数 ppm（每毫米对应的像素数）：浮点数。
# 返回类型：元组（依次为浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def px_to_mm(point: tuple[float, float], ppm: float) -> tuple[float, float]:
    # 返回结果：创建数据容器。
    return point[0] / ppm, point[1] / ppm

# =============================================================================
# 【分区】A4 最短正交路径（转调 orthogonal_path_planner）
# 功能：毫米↔像素；为每片从吸取点到目标点规划先 X 后 Y 的折线；optimize_motion_order 可再排顺序。
# 可修改：先 X 后 Y 在 orthogonal_path_planner.plan_shortest_orthogonal_path。
# 看情况改：本层只做接口转发，避障不要改这里的包装函数。
# 不要改：path_px 单位像素、写入 plan 前再换成 mm；轴对齐判定。
# =============================================================================
# ------------------------- A4最短正交路径 -------------------------

# 【函数：axis_aligned】兼容原调用接口，实际实现位于独立路径规划模块。
# 参数 start（起始值或起点）：元组（依次为浮点数、浮点数）。
# 参数 end（终止值或终点）：元组（依次为浮点数、浮点数）。
# 参数 tolerance（用于浮点比较或几何判断的容差）：浮点数；省略时使用1e-06。
# 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
def axis_aligned(start: tuple[float, float], end: tuple[float, float],
                 tolerance: float = 1e-6) -> bool:
    """兼容原调用接口，实际实现位于独立路径规划模块。"""
    # 返回结果：调用 path_planner.axis_aligned。
    return path_planner.axis_aligned(start, end, tolerance)


# 【函数：compact_orthogonal_path】兼容原调用接口，压缩正交路径中的冗余点。
# 参数 points（参与当前计算的一组坐标点）：元组（依次为浮点数、浮点数）列表/序列。
# 返回类型：元组（依次为浮点数、浮点数）列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def compact_orthogonal_path(
        points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """兼容原调用接口，压缩正交路径中的冗余点。"""
    # 返回结果：调用 path_planner.compact_orthogonal_path。
    return path_planner.compact_orthogonal_path(points)


# 【函数：plan_single_path】规划A4范围内的最短正交路径；不使用外围通道和碎片障碍。
# 这份实现规划无避障约束的最短正交路径：只保证路径在工作区内并且每段单轴运动，不搜索绕开其他碎片的路线。
# 参数 start（起始值或起点）：元组（依次为浮点数、浮点数）。
# 参数 goal（路径目标点）：元组（依次为浮点数、浮点数）。
# 参数 obstacle：NumPy数组或None（不返回业务结果）。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为元组（依次为浮点数、浮点数）列表/序列、字符串、字符串）；箭头->是类型提示，不会替你转换实际返回值。
def plan_single_path(start: tuple[float, float], goal: tuple[float, float],
                     obstacle: np.ndarray | None, config: dict[str, Any]
                     ) -> tuple[list[tuple[float, float]], str, str]:
    """规划A4范围内的最短正交路径；不使用外围通道和碎片障碍。"""
    del obstacle  # 保留旧参数以兼容现有调用方，当前规则明确不做避障。
    # 计算并保存到 plan（从JSON读取的运动计划字典）。
    plan = path_planner.plan_shortest_orthogonal_path(
        start, goal, workspace_size_px(config)
    )
    # 返回结果：创建数据容器。
    return plan.points, plan.planner, plan.status


# 【函数：plan_all_paths】为每块匹配成功的碎片规划从原吸取点到目标吸取点的路径。
# 参数 pieces（碎片列表）：Piece列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 matches（匹配记录列表）：PuzzleMatch列表/序列或None（不返回业务结果）；省略时使用None。
# 返回类型：PathResult列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def plan_all_paths(pieces: list[Piece], config: dict[str, Any],
                   matches: list[PuzzleMatch] | None = None) -> list[PathResult]:
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 matches（匹配记录列表）。
    matches = solve_puzzle(pieces, config) if matches is None else matches
    # 判断条件；满足时执行下面代码：[match.piece.piece_id for match in matches] != [piece.piece_id for piece in pie…。
    if [match.piece.piece_id for match in matches] != [piece.piece_id for piece in pieces]:
        # 抛出异常，通知上层处理：ValueError('模板匹配结果与搬运顺序不一致')。
        raise ValueError("模板匹配结果与搬运顺序不一致")

    # 判断条件；满足时执行下面代码：not all((match.status == 'ok' for match in matches))。
    if not all(match.status == "ok" for match in matches):
        # 返回结果：按条件生成列表。
        return [PathResult(match, [match.piece.pick_px], "none", "match_failed")
                # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
                for match in matches]

    # 计算并保存到 results（结果集合）。
    results: list[PathResult] = []
    # 遍历数据，逐项处理：matches。
    for match in matches:
        # 计算并保存到 piece（当前碎片数据）。
        piece = match.piece
        # 计算并保存到 target_px（目标·像素）。
        target_px = mm_to_px(match.target_pick_mm, ppm)
        # 计算并保存到 path（当前文件路径或几何路径）、planner（路径规划方法标记）、status（处理结果的状态标记）。
        path, planner, status = plan_single_path(piece.pick_px, target_px, None, config)
        # 调用函数：results.append。
        results.append(PathResult(match, path, planner, status))
    # 返回结果：results（结果集合）。
    return results

# =============================================================================
# 【分区】可视化、安全检查与 plan.json 输出
# 功能：叠加预览图；检查路径出界/斜走/目标重叠；写成 output 下的图和 plan.json。
# 可修改：预览颜色、输出文件名；motion_safety 阈值走配置。
# 看情况改：ready_for_motion=False 时不要强行发送。现场只改补偿，不要关安全检查。
# 不要改：plan.json 字段名（source_pick_mm、path_mm、rotation_deg_signed 等），串口模块按名读取。
# =============================================================================
# ------------------------- 可视化与文件输出 -------------------------

# 【函数：draw_preview】在纸面图上叠加源碎片、目标轮廓、编号和搬运路径。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 pieces（碎片列表）：Piece列表/序列。
# 参数 paths（各碎片的搬运路径结果）：PathResult列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def draw_preview(paper: np.ndarray, pieces: list[Piece], paths: list[PathResult],
                 config: dict[str, Any]) -> np.ndarray:
    # 计算并保存到 workspace_width（工作区·宽度）、workspace_height（工作区·高度）。
    workspace_width, workspace_height = workspace_size_px(config)
    # 判断条件；满足时执行下面代码：paper.shape[1] > workspace_width or paper.shape[0] > workspace_height。
    if paper.shape[1] > workspace_width or paper.shape[0] > workspace_height:
        # 抛出异常，通知上层处理：ValueError('A4校正图尺寸超过机械工作区')。
        raise ValueError("A4校正图尺寸超过机械工作区")
    # 计算并保存到 display（用于窗口显示的画布）。
    display = np.full((workspace_height, workspace_width, 3), (55, 55, 55), np.uint8)
    # 计算并保存到 display[:paper.shape[0], :paper.shape[1]]。
    display[:paper.shape[0], :paper.shape[1]] = paper
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 target_y（目标·Y轴）。
    target_y = round(float(config.get("target_region_top_mm", 148.5)) * ppm)
    # 调用函数：cv2.line。
    cv2.line(display, (0, target_y), (display.shape[1] - 1, target_y),
             (0, 255, 255), 2)
    # 计算并保存到 colors（颜色集合）。
    colors = [(0, 255, 0), (255, 180, 0), (255, 0, 255), (0, 180, 255)]
    # 计算并保存到 overlay（叠加轮廓和标记后的图像）。
    overlay = display.copy()
    # 遍历数据，逐项处理：paths。
    for result in paths:
        # 计算并保存到 color（颜色）。
        color = colors[(result.piece.piece_id - 1) % len(colors)]
        # 计算并保存到 target_polygon（目标·多边形）。
        target_polygon = np.rint(result.match.target_polygon_mm * ppm).astype(np.int32)
        # 调用函数：cv2.fillPoly。
        cv2.fillPoly(overlay, [target_polygon], color)
    # 计算并保存到 display（用于窗口显示的画布）。
    display = cv2.addWeighted(overlay, 0.14, display, 0.86, 0)

    # 遍历数据，逐项处理：pieces。
    for piece in pieces:
        # 计算并保存到 color（颜色）。
        color = colors[(piece.piece_id - 1) % len(colors)]
        # 调用函数：cv2.drawContours。
        cv2.drawContours(display, [piece.contour.astype(np.int32)], -1, color, 2)
        # 计算并保存到 pick（吸取位置）。
        pick = tuple(map(round, piece.pick_px))
        # 计算并保存到 center_mm（视觉质心的毫米坐标(x,y)）。
        center_mm = piece.center_mm(ppm)
        # 调用函数：cv2.drawMarker。
        cv2.drawMarker(display, pick, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        # 调用函数：cv2.putText。
        cv2.putText(display, f"P{piece.piece_id} ({center_mm[0]:.1f},{center_mm[1]:.1f})mm",
                    (pick[0] + 8, pick[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)

    # 遍历数据，逐项处理：paths。
    for result in paths:
        # 计算并保存到 color（颜色）。
        color = colors[(result.piece.piece_id - 1) % len(colors)]
        # 计算并保存到 target_polygon（目标·多边形）。
        target_polygon = np.rint(result.match.target_polygon_mm * ppm).astype(np.int32)
        # 调用函数：cv2.polylines。
        cv2.polylines(display, [target_polygon], True, color, 2, cv2.LINE_AA)
        # 计算并保存到 points（参与当前计算的一组坐标点）。
        points = [tuple(map(round, point)) for point in result.path_px]
        # 判断条件；满足时执行下面代码：len(points) >= 2。
        if len(points) >= 2:
            # 调用函数：cv2.polylines。
            cv2.polylines(display, [np.asarray(points, np.int32)], False,
                          color, 3, cv2.LINE_AA)
        # 遍历数据，逐项处理：points。
        for point in points:
            # 调用函数：cv2.circle。
            cv2.circle(display, point, 5, color, -1)
        # 计算并保存到 target（目标数据或目标位置）。
        target = tuple(map(round, mm_to_px(result.target_mm, ppm)))
        # 调用函数：cv2.circle。
        cv2.circle(display, target, 9, color, 2)
        # 计算并保存到 label。
        label = (f"T{result.match.template_id}/P{result.piece.piece_id} "
                 f"{result.match.rotation_deg:+.1f}deg")
        # 调用函数：cv2.putText。
        cv2.putText(display, label, (target[0] + 8, target[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    # 调用函数：cv2.putText。
    cv2.putText(display, "Move order: optimized for shortest total XY travel",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    # 调用函数：cv2.putText。
    cv2.putText(display, "Path: X/Y only, one axis at a time; Rotation: +CW / -CCW",
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    # 计算并保存到 top_left（上边·左边）、size（尺寸数据）。
    top_left, size = target_rectangle_from_paths(paths, config)
    # 计算并保存到 rectangle_start。
    rectangle_start = tuple(map(round, mm_to_px(top_left, ppm)))
    # 计算并保存到 rectangle_end。
    rectangle_end = tuple(map(round, mm_to_px(
        (top_left[0] + size[0], top_left[1] + size[1]), ppm
    )))
    # 调用函数：cv2.rectangle。
    cv2.rectangle(display, rectangle_start, rectangle_end, (255, 255, 0), 2)
    # 计算并保存到 mode（模式）。
    mode = str(config.get("puzzle_mode", "fixed_figure_2"))
    # 调用函数：cv2.putText。
    cv2.putText(display, f"Mode: {mode}", (15, 82), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 0), 2, cv2.LINE_AA)

    # 计算并保存到 ready_for_motion（计划是否通过发送前的检查）。
    ready_for_motion = not motion_safety_reasons(paths, config)
    # 判断条件；满足时执行下面代码：not ready_for_motion。
    if not ready_for_motion:
        # 调用函数：cv2.rectangle。
        cv2.rectangle(display, (8, 66), (display.shape[1] - 8, 108),
                      (20, 20, 190), -1)
        # 调用函数：cv2.putText。
        cv2.putText(display, "MATCH/PATH FAILED - NO MOTION",
                    (18, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                    (255, 255, 255), 2, cv2.LINE_AA)
        # 遍历数据，逐项处理：paths。
        for result in paths:
            # 判断条件；满足时执行下面代码：result.match.status == 'ok' and result.status == 'ok'。
            if result.match.status == "ok" and result.status == "ok":
                # 跳过本轮，进入下一轮循环。
                continue
            # 计算并保存到 pick（吸取位置）。
            pick = tuple(map(round, result.piece.pick_px))
            # 计算并保存到 reason（原因）。
            reason = "FAIL" if result.match.status != "ok" else "LOCKED"
            # 调用函数：cv2.putText。
            cv2.putText(display,
                        f"P{result.piece.piece_id}/{result.match.template_id} {reason}",
                        (pick[0] + 8, pick[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2, cv2.LINE_AA)
    # 返回结果：display（用于窗口显示的画布）。
    return display


# 【函数：axis_motion_steps】将正交折线路径转换为一次只移动一个轴的指令。
# 参数 path_mm（按行进顺序排列的毫米路径点）：元组（依次为浮点数、浮点数）列表/序列。
# 返回类型：字典（键为字符串，值为任意类型）列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def axis_motion_steps(path_mm: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """将正交折线路径转换为一次只移动一个轴的指令。"""
    # 计算并保存到 commands（依次执行的串口命令列表）。
    commands: list[dict[str, Any]] = []
    # 遍历数据，逐项处理：zip(path_mm, path_mm[1:])。
    for start, end in zip(path_mm, path_mm[1:]):
        # 计算并保存到 dx、dy。
        dx, dy = end[0] - start[0], end[1] - start[1]
        # 判断条件；满足时执行下面代码：abs(dx) > 1e-06 and abs(dy) > 1e-06。
        if abs(dx) > 1e-6 and abs(dy) > 1e-6:
            # 抛出异常，通知上层处理：ValueError('路径包含斜线，不能生成单轴运动指令')。
            raise ValueError("路径包含斜线，不能生成单轴运动指令")
        # 判断条件；满足时执行下面代码：abs(dx) > 1e-06。
        if abs(dx) > 1e-6:
            # 计算并保存到 axis（选中的轴或数组维度）、delta（当前增量或修正量）。
            axis, delta = "X", dx
            # 计算并保存到 direction（方向标记或方向向量）。
            direction = "RIGHT" if dx > 0 else "LEFT"
        # 判断条件；满足时执行下面代码：abs(dy) > 1e-06。
        elif abs(dy) > 1e-6:
            # 计算并保存到 axis（选中的轴或数组维度）、delta（当前增量或修正量）。
            axis, delta = "Y", dy
            # 计算并保存到 direction（方向标记或方向向量）。
            direction = "DOWN" if dy > 0 else "UP"
        # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
        else:
            # 跳过本轮，进入下一轮循环。
            continue
        # 调用函数：commands.append。
        commands.append({
            # 字典字段'axis'（选中的轴或数组维度）：axis（选中的轴或数组维度）。
            "axis": axis,
            # 字典字段'direction'（方向标记或方向向量）：direction（方向标记或方向向量）。
            "direction": direction,
            # 字典字段'delta_mm'（delta·毫米）：调用 round。
            "delta_mm": round(delta, 2),
            # 字典字段'distance_mm'（距离·毫米）：调用 round。
            "distance_mm": round(abs(delta), 2),
            # 字典字段'from_mm'（起始·毫米）：创建数据容器。
            "from_mm": [round(start[0], 2), round(start[1], 2)],
            # 字典字段'to_mm'（终止·毫米）：创建数据容器。
            "to_mm": [round(end[0], 2), round(end[1], 2)],
        })
    # 返回结果：commands（依次执行的串口命令列表）。
    return commands


# 【函数：target_rectangle_from_paths】返回求解器实际目标矩形；固定模式继续使用原配置。
# 参数 paths（各碎片的搬运路径结果）：PathResult列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为元组（依次为浮点数、浮点数）、元组（依次为浮点数、浮点数））；箭头->是类型提示，不会替你转换实际返回值。
def target_rectangle_from_paths(paths: list[PathResult],
                                config: dict[str, Any]) -> tuple[tuple[float, float],
                                                                  tuple[float, float]]:
    """返回求解器实际目标矩形；固定模式继续使用原配置。"""
    # 判断条件；满足时执行下面代码：paths。
    if paths:
        # 计算并保存到 match（当前匹配记录）。
        match = paths[0].match
        # 判断条件；满足时执行下面代码：match.target_rectangle_top_left_mm is not None and match.target_rectangle_size_…。
        if (match.target_rectangle_top_left_mm is not None and
                match.target_rectangle_size_mm is not None):
            # 返回结果：创建数据容器。
            return match.target_rectangle_top_left_mm, match.target_rectangle_size_mm
    # 计算并保存到 top_left（上边·左边）。
    top_left = tuple(float(value) for value in
                     config.get("target_rectangle_top_left_mm", [55.0, 205.0]))
    # 计算并保存到 size（尺寸数据）。
    size = tuple(float(value) for value in
                 config.get("target_rectangle_size_mm", [100.0, 60.0]))
    # 返回结果：创建数据容器。
    return (top_left[0], top_left[1]), (size[0], size[1])


# 【函数：motion_safety_reasons】集中验证数量、求解质量、目标区域和严格正交工作区路径。
# 空原因列表表示本函数检查通过。宽松或强制执行分支跳过部分几何质量阈值；低花纹相似度仅打印提示，没有加入拒绝原因。
# 参数 paths（各碎片的搬运路径结果）：PathResult列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：字符串列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def motion_safety_reasons(paths: list[PathResult], config: dict[str, Any]) -> list[str]:
    """集中验证数量、求解质量、目标区域和严格正交工作区路径。"""
    # 计算并保存到 reasons（检查未通过的原因列表）。
    reasons: list[str] = []
    # 计算并保存到 mode（模式）。
    mode = str(config.get("puzzle_mode", "fixed_figure_2"))
    # 计算并保存到 count（计数值）。
    count = len(paths)
    # 判断条件；满足时执行下面代码：mode == 'fixed_figure_2' and count != int(config.get('expected_piece_count', 4))。
    if mode == "fixed_figure_2" and count != int(config.get("expected_piece_count", 4)):
        # 调用函数：reasons.append。
        reasons.append(f"固定图2路径数量错误：{count}")
    # 判断条件；满足时执行下面代码：mode == 'generic_geometry' and (not 1 <= count <= int(config.get('max_pieces', …。
    elif mode == "generic_geometry" and not 1 <= count <= int(config.get("max_pieces", 4)):
        # 调用函数：reasons.append。
        reasons.append(f"通用拼图路径数量错误：{count}")
    # 判断条件；满足时执行下面代码：mode not in {'fixed_figure_2', 'generic_geometry'}。
    elif mode not in {"fixed_figure_2", "generic_geometry"}:
        # 调用函数：reasons.append。
        reasons.append(f"未知puzzle_mode：{mode}")
    # 判断条件；满足时执行下面代码：not paths。
    if not paths:
        # 调用函数：reasons.append。
        reasons.append("没有可执行碎片路径")
        # 返回结果：reasons（检查未通过的原因列表）。
        return reasons

    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 workspace_width_px（工作区·宽度·像素）、workspace_height_px（工作区·高度·像素）。
    workspace_width_px, workspace_height_px = workspace_size_px(config)
    # 计算并保存到 paper_width（纸面宽度（毫米））。
    paper_width = float(config["paper_width_mm"])
    # 计算并保存到 paper_height（纸面高度（毫米））。
    paper_height = float(config["paper_height_mm"])
    # 计算并保存到 target_top（目标·上边）。
    target_top = float(config.get("target_region_top_mm", paper_height / 2.0))
    # 计算并保存到 point_tolerance（点·容差）。
    point_tolerance = float(config.get("target_pick_transform_tolerance_mm", 0.5))

    # 遍历数据，逐项处理：paths。
    for result in paths:
        # 计算并保存到 piece_id（面向显示和计划的碎片编号）。
        piece_id = result.piece.piece_id
        # 计算并保存到 match（当前匹配记录）。
        match = result.match
        # 判断条件；满足时执行下面代码：match.status != 'ok'。
        if match.status != "ok":
            # 调用函数：reasons.append。
            reasons.append(f"P{piece_id}求解状态={match.status}")
        # 判断条件；满足时执行下面代码：result.status != 'ok'。
        if result.status != "ok":
            # 调用函数：reasons.append。
            reasons.append(f"P{piece_id}路径状态={result.status}")
        # 计算并保存到 polygon（当前多边形的顶点数组）。
        polygon = np.asarray(match.target_polygon_mm, np.float64)
        # 判断条件；满足时执行下面代码：polygon.ndim != 2 or polygon.shape[1] != 2 or (not np.isfinite(polygon).all())。
        if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
            # 调用函数：reasons.append。
            reasons.append(f"P{piece_id}目标多边形无效")
        # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
        else:
            # 计算并保存到 minimum（允许下限或计算出的最小值）。
            minimum = np.min(polygon, axis=0)
            # 计算并保存到 maximum（允许上限或计算出的最大值）。
            maximum = np.max(polygon, axis=0)
            # 判断条件；满足时执行下面代码：minimum[0] < -1e-06 or minimum[1] < target_top - 1e-06 or maximum[0] > paper_wi…。
            if (minimum[0] < -1e-6 or minimum[1] < target_top - 1e-6 or
                    maximum[0] > paper_width + 1e-6 or maximum[1] > paper_height + 1e-6):
                # 调用函数：reasons.append。
                reasons.append(f"P{piece_id}目标多边形未完整位于A4下半区")
            # 判断条件；满足时执行下面代码：cv2.pointPolygonTest(polygon.astype(np.float32), match.target_pick_mm, False) <…。
            if cv2.pointPolygonTest(
                    polygon.astype(np.float32), match.target_pick_mm, False) < 0:
                # 调用函数：reasons.append。
                reasons.append(f"P{piece_id}目标磁吸点位于目标多边形外")

        # 遍历数据，逐项处理：result.path_px。
        for x, y in result.path_px:
            # 判断条件；满足时执行下面代码：not (0.0 <= x < workspace_width_px and 0.0 <= y < workspace_height_px)。
            if not (0.0 <= x < workspace_width_px and 0.0 <= y < workspace_height_px):
                # 调用函数：reasons.append。
                reasons.append(f"P{piece_id}路径点超出机械工作区")
                # 立即结束当前循环。
                break
        # 判断条件；满足时执行下面代码：not all((axis_aligned(start, end) for start, end in zip(result.path_px, result.…。
        if not all(axis_aligned(start, end)
                   # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
                   for start, end in zip(result.path_px, result.path_px[1:])):
            # 调用函数：reasons.append。
            reasons.append(f"P{piece_id}路径包含非正交线段")

        # 判断条件；满足时执行下面代码：match.solver_mode == 'generic_geometry'。
        if match.solver_mode == "generic_geometry":
            # 判断条件；满足时执行下面代码：match.transform_3x3 is None or not generic_solver.is_rigid_transform(match.tran…。
            if match.transform_3x3 is None or not generic_solver.is_rigid_transform(
                    # 传入命名参数：tolerance=1e-05（用于浮点比较或几何判断的容差）；取值过程：1e-05。
                    match.transform_3x3, tolerance=1e-5):
                # 调用函数：reasons.append。
                reasons.append(f"P{piece_id}缺少有效刚体变换")
            # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
            else:
                # 计算并保存到 mapped。
                mapped = generic_solver.transform_points(
                    np.asarray([result.piece.pick_mm(ppm)], np.float64),
                    match.transform_3x3,
                )[0]
                # 判断条件；满足时执行下面代码：np.linalg.norm(mapped - np.asarray(match.target_pick_mm)) > point_tolerance。
                if np.linalg.norm(mapped - np.asarray(match.target_pick_mm)) > point_tolerance:
                    # 调用函数：reasons.append。
                    reasons.append(f"P{piece_id}磁吸点未使用同一刚体变换")
            # 判断条件；满足时执行下面代码：abs(match.fit_scale - 1.0) > 1e-06。
            if abs(match.fit_scale - 1.0) > 1e-6:
                # 调用函数：reasons.append。
                reasons.append(f"P{piece_id}通用求解发生缩放")
            # 比赛完成优先：严格合格解按运动级质量标准校验；宽松兜底解
            # （超时后放宽填充/缺口产生的解）或强制执行开关开启时，
            # 跳过所有质量类阈值，只保留硬件安全（刚体变换/工作区/路径）。
            # 判断条件；满足时执行下面代码：not (match.relaxed_override or bool(config.get('force_execute_on_any_solution',…。
            if not (match.relaxed_override or bool(
                    config.get("force_execute_on_any_solution", False))):
                # 计算并保存到 motion_fill_limit（运动·填充·上限）。
                motion_fill_limit = float(config.get(
                    "generic_motion_max_fill_error_ratio",
                    config.get("generic_max_fill_error_ratio", 0.06),
                ))
                # 判断条件；满足时执行下面代码：match.fill_error_ratio > motion_fill_limit。
                if match.fill_error_ratio > motion_fill_limit:
                    # 调用函数：reasons.append。
                    reasons.append(f"P{piece_id}矩形填充误差超限")
                # 计算并保存到 motion_overlap_limit（运动·重叠·上限）。
                motion_overlap_limit = float(config.get(
                    "generic_motion_max_overlap_ratio",
                    config.get("generic_max_overlap_ratio", 0.005),
                ))
                # 判断条件；满足时执行下面代码：match.overlap_ratio > motion_overlap_limit。
                if match.overlap_ratio > motion_overlap_limit:
                    # 调用函数：reasons.append。
                    reasons.append(f"P{piece_id}目标碎片重叠率超限")
                # 判断条件；满足时执行下面代码：match.endpoint_error_mm > float(config.get('generic_max_endpoint_error_mm', 3.0…。
                if match.endpoint_error_mm > float(config.get("generic_max_endpoint_error_mm", 3.0)):
                    # 调用函数：reasons.append。
                    reasons.append(f"P{piece_id}拼缝端点误差超限")
                # 判断条件；满足时执行下面代码：match.boundary_error_mm > float(config.get('generic_boundary_tolerance_mm', 3.0…。
                if match.boundary_error_mm > float(config.get("generic_boundary_tolerance_mm", 3.0)):
                    # 调用函数：reasons.append。
                    reasons.append(f"P{piece_id}矩形边界误差超限")
                # 计算并保存到 motion_boundary_gap_limit（运动·边界·缺口·上限）。
                motion_boundary_gap_limit = float(config.get(
                    "generic_motion_max_boundary_gap_ratio",
                    config.get("generic_max_boundary_gap_ratio", 0.08),
                ))
                # 判断条件；满足时执行下面代码：match.boundary_gap_ratio > motion_boundary_gap_limit。
                if match.boundary_gap_ratio > motion_boundary_gap_limit:
                    # 调用函数：reasons.append。
                    reasons.append(f"P{piece_id}矩形边界缺口率超限")
            # 判断条件；满足时执行下面代码：match.relaxed_override。
            elif match.relaxed_override:
                # 调用函数：print。
                print(f"[保底-宽松] P{piece_id}为宽松兜底解，跳过运动级质量校验（硬件安全仍检查）")
    # 判断条件；满足时执行下面代码：paths and paths[0].match.pattern_verification is not None。
    if paths and paths[0].match.pattern_verification is not None:
        # 计算并保存到 verification（复核）。
        verification = paths[0].match.pattern_verification
        # 判断条件；满足时执行下面代码：verification.get('n_pairs', 0) > 0。
        if verification.get("n_pairs", 0) > 0:
            # 计算并保存到 minimum_similarity（最小·similarity）。
            minimum_similarity = float(verification.get("min_similarity", 1.0))
            # 计算并保存到 ncc_limit（互相关·上限）。
            ncc_limit = float(config.get("poker_ncc_min_similarity", 0.55))
            # 判断条件；满足时执行下面代码：minimum_similarity < ncc_limit。
            if minimum_similarity < ncc_limit:
                # 调用函数：print。
                print(f"[花纹] 相似度{minimum_similarity:.2f}<{ncc_limit:.2f}，"
                      f"按形状拼接矩形执行（不拦截）")
    # 返回结果：调用 list。
    return list(dict.fromkeys(reasons))


# 【函数：result_dict】把一块碎片的检测、匹配和路径结果转换为可写入JSON的字典。
# 参数 result（当前步骤得到的结果）：PathResult。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：字典（键为字符串，值为任意类型）；箭头->是类型提示，不会替你转换实际返回值。
def result_dict(result: PathResult, config: dict[str, Any]) -> dict[str, Any]:
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 center（本步骤使用的中心位置）。
    center = result.piece.center_mm(ppm)
    # 计算并保存到 pick（吸取位置）。
    pick = result.piece.pick_mm(ppm)
    # 计算并保存到 path_mm（按行进顺序排列的毫米路径点）。
    path_mm = [px_to_mm(point, ppm) for point in result.path_px]
    # 计算并保存到 match（当前匹配记录）。
    match = result.match
    # 返回结果：创建结果字典。
    return {
        # 字典字段'id'（编号）：result.piece.piece_id（result.piece.piece·编号）。
        "id": result.piece.piece_id,
        # 字典字段'move_order'（实际搬运顺序（从1开始））：result.piece.piece_id（result.piece.piece·编号）。
        "move_order": result.piece.piece_id,
        # 字典字段'template_id'（目标模板编号）：match.template_id（match.template·编号）。
        "template_id": match.template_id,
        # 字典字段'status'（处理结果的状态标记）：result.status。
        "status": result.status,
        # 字典字段'match_status'（匹配·状态）：match.status。
        "match_status": match.status,
        # 字典字段'planner'（路径规划方法标记）：result.planner。
        "planner": result.planner,
        # 字典字段'center_mm'（视觉质心的毫米坐标(x,y)）：创建数据容器。
        "center_mm": [round(center[0], 2), round(center[1], 2)],
        # 字典字段'source_pick_mm'（搬运前磁吸点坐标（毫米））：创建数据容器。
        "source_pick_mm": [round(pick[0], 2), round(pick[1], 2)],
        # 字典字段'target_pick_mm'（放置后磁吸点的目标坐标（毫米））：创建数据容器。
        "target_pick_mm": [round(match.target_pick_mm[0], 2),
                           round(match.target_pick_mm[1], 2)],
        # 字典字段'target_mm'（目标·毫米）：创建数据容器。
        "target_mm": [round(match.target_pick_mm[0], 2),
                      round(match.target_pick_mm[1], 2)],
        # 字典字段'source_min_area_rect_angle_deg'（源·最小·面积·rect·角度·度）：调用 round。
        "source_min_area_rect_angle_deg": round(result.piece.angle_deg, 2),
        # 字典字段'rotation_deg_signed'（正值顺时针、负值逆时针的旋转角（度））：调用 round。
        "rotation_deg_signed": round(match.rotation_deg, 2),
        # 字典字段'rotation_direction'（旋转方向标记）：match.rotation_direction。
        "rotation_direction": match.rotation_direction,
        # 字典字段'rotation_angle_deg'（旋转·角度·度）：调用 round。
        "rotation_angle_deg": round(abs(match.rotation_deg), 2),
        # 字典字段'rotation_convention'（旋转·convention）：'positive=CW, negative=CCW, image/A4 coordinates'。
        "rotation_convention": "positive=CW, negative=CCW, image/A4 coordinates",
        # 字典字段'match_iou'（形状匹配的交并比）：调用 round。
        "match_iou": round(match.match_iou, 4),
        # 字典字段'area_error_ratio'（面积相对误差）：调用 round。
        "area_error_ratio": round(match.area_error_ratio, 4),
        # 字典字段'shape_distance'（形状差异指标）：调用 round。
        "shape_distance": round(match.shape_distance, 4),
        # 字典字段'fit_scale'（形状匹配时的尺度比；刚体搬运应保持为1）：调用 round。
        "fit_scale": round(match.fit_scale, 4),
        # 字典字段'solver_mode'（求解方式标记）：match.solver_mode（match.solver·模式）。
        "solver_mode": match.solver_mode,
        # 字典字段'solution_score'（几何方案的代价评分，通常越小越好）：调用 round。
        "solution_score": round(match.solution_score, 6),
        # 字典字段'fill_error_ratio'（矩形填充误差比例）：调用 round。
        "fill_error_ratio": round(match.fill_error_ratio, 6),
        # 字典字段'overlap_ratio'（碎片间重叠比例）：调用 round。
        "overlap_ratio": round(match.overlap_ratio, 6),
        # 字典字段'endpoint_error_mm'（拼缝对应端点的误差（毫米））：调用 round。
        "endpoint_error_mm": round(match.endpoint_error_mm, 4),
        # 字典字段'boundary_error_mm'（目标矩形边界误差（毫米））：调用 round。
        "boundary_error_mm": round(match.boundary_error_mm, 4),
        # 字典字段'boundary_gap_ratio'（矩形边界缺口比例）：调用 round。
        "boundary_gap_ratio": round(match.boundary_gap_ratio, 6),
        # 字典字段'transform_3x3'（3×3齐次变换矩阵（把源坐标映射到目标坐标））：None if match.transform_3x3 is None else [[round(float(value), 8) for value in row] …。
        "transform_3x3": (None if match.transform_3x3 is None else
                          [[round(float(value), 8) for value in row]
                           # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
                           for row in match.transform_3x3]),
        # 字典字段'area_mm2'（碎片面积，单位为平方毫米）：调用 round。
        "area_mm2": round(result.piece.area_mm2, 2),
        # 字典字段'size_mm'（尺寸·毫米）：创建数据容器。
        "size_mm": [round(result.piece.width_mm, 2), round(result.piece.height_mm, 2)],
        # 字典字段'target_polygon_mm'（目标布局中的多边形顶点（毫米））：按条件生成列表。
        "target_polygon_mm": [[round(float(x), 2), round(float(y), 2)]
                              # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
                              for x, y in match.target_polygon_mm],
        # 字典字段'target_clearance_shift_mm'（为留出碎片间隙施加的平移（毫米））：创建数据容器。
        "target_clearance_shift_mm": [
            round(match.target_clearance_shift_mm[0], 2),
            round(match.target_clearance_shift_mm[1], 2),
        ],
        # 字典字段'path_mm'（按行进顺序排列的毫米路径点）：按条件生成列表。
        "path_mm": [[round(x, 2), round(y, 2)] for x, y in path_mm],
        # 字典字段'path_constraint'（路径·constraint）：'orthogonal_xy_one_axis_at_a_time'。
        "path_constraint": "orthogonal_xy_one_axis_at_a_time",
        # 字典字段'axis_moves_mm'（逐段单轴位移记录）：调用 axis_motion_steps。
        "axis_moves_mm": axis_motion_steps(path_mm),
        # 字典字段'motion_sequence'（运动·sequence）：创建数据容器。
        "motion_sequence": [
            "MOVE_TO_SOURCE_PICK", "MAGNET_ON", "LIFT_Z", "ROTATE",
            "FOLLOW_PATH", "LOWER_Z", "MAGNET_OFF"
        ],
    }


# 【函数：optimize_motion_order】枚举四片执行顺序，最小化XY轴总正交行程。
# 当前目标函数包含空载到下一吸取点和搬运到目标的曼哈顿距离；没有累计最后回原点的路程，也没有包含抬升与旋转耗时。
# 参数 pieces（碎片列表）：字典（键为字符串，值为任意类型）列表/序列。
# 参数 target_offset_mm（目标·偏移·毫米）：元组（依次为浮点数、浮点数）。
# 参数 start_mm（start·毫米）：元组（依次为浮点数、浮点数）；省略时使用(0.0, 0.0)。
# 返回类型：元组（依次为字典（键为字符串，值为任意类型）列表/序列、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def optimize_motion_order(
        pieces: list[dict[str, Any]], target_offset_mm: tuple[float, float],
        start_mm: tuple[float, float] = (0.0, 0.0),
) -> tuple[list[dict[str, Any]], float]:
    """枚举四片执行顺序，最小化XY轴总正交行程。"""
    # 判断条件；满足时执行下面代码：not pieces。
    if not pieces:
        # 返回结果：创建数据容器。
        return [], 0.0

    # 计算并保存到 offset_x（偏移·X轴）、offset_y（偏移·Y轴）。
    offset_x, offset_y = target_offset_mm

    # 【函数：point】从运动记录的指定字段读取二维浮点坐标。
    # 参数 item（当前遍历到的元素）：字典（键为字符串，值为任意类型）。
    # 参数 key（查询、分组或排序所用的键）：字符串。
    # 返回类型：元组（依次为浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
    def point(item: dict[str, Any], key: str) -> tuple[float, float]:
        # 计算并保存到 value（当前数值）。
        value = item[key]
        # 返回结果：创建数据容器。
        return float(value[0]), float(value[1])

    # 【函数：distance】计算两点的曼哈顿距离，即横向距离加纵向距离。
    # 参数 first（第一项数据）：元组（依次为浮点数、浮点数）。
    # 参数 second（第二项数据）：元组（依次为浮点数、浮点数）。
    # 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
    def distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        # 返回结果：计算 abs(first[0] - second[0]) + abs(first[1] - second[1])。
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    # 【函数：route_distance】累计从起始点到各吸取点及放置点的单轴行程。
    # 参数 order（顺序）：由若干字典（键为字符串，值为任意类型）组成的元组。
    # 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
    def route_distance(order: tuple[dict[str, Any], ...]) -> float:
        # 计算并保存到 current（当前）。
        current = start_mm
        # 计算并保存到 total（总量）。
        total = 0.0
        # 遍历数据，逐项处理：order。
        for item in order:
            # 计算并保存到 source（源数据或源位置）。
            source = point(item, "source_pick_mm")
            # 计算并保存到 target_raw（目标·原始）。
            target_raw = point(item, "target_pick_mm")
            # 计算并保存到 target（目标数据或目标位置）。
            target = target_raw[0] + offset_x, target_raw[1] + offset_y
            # 更新变量：total（总量）。
            total += distance(current, source) + distance(source, target)
            # 计算并保存到 current（当前）。
            current = target
        # 返回结果：total（总量）。
        return total

    # 计算并保存到 best（最佳）。
    best = min(
        itertools.permutations(pieces),
        key=lambda order: (route_distance(order), tuple(int(item["id"]) for item in order)),
    )
    # 计算并保存到 ordered（已排序的数据）。
    ordered = [dict(item) for item in best]
    # 遍历数据，逐项处理：enumerate(ordered, 1)。
    for move_order, item in enumerate(ordered, 1):
        # 计算并保存到 item['move_order']。
        item["move_order"] = move_order
    # 返回结果：创建数据容器。
    return ordered, route_distance(best)


# 【函数：save_outputs】保存原图、校正图、掩膜、预览图和经过检查的plan.json。
# 参数 original（原始图像）：NumPy数组。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 mask（二值掩膜（0为背景，非零为选中区域））：NumPy数组。
# 参数 preview（路径预览图）：NumPy数组。
# 参数 paths（各碎片的搬运路径结果）：PathResult列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def save_outputs(original: np.ndarray, paper: np.ndarray, mask: np.ndarray,
                 preview: np.ndarray, paths: list[PathResult],
                 config: dict[str, Any]) -> None:
    # 调用函数：OUTPUT_DIR.mkdir。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 调用函数：write_image。
    write_image(OUTPUT_DIR / "original.jpg", original)
    # 调用函数：write_image。
    write_image(OUTPUT_DIR / "corrected.jpg", paper)
    # 调用函数：write_image。
    write_image(OUTPUT_DIR / "mask.png", mask)
    # 调用函数：write_image。
    write_image(OUTPUT_DIR / "path_preview.jpg", preview)
    # 计算并保存到 actual_top_left（实测·上边·左边）、actual_size（实测·尺寸）。
    actual_top_left, actual_size = target_rectangle_from_paths(paths, config)
    # 计算并保存到 top_left（上边·左边）。
    top_left = [float(value) for value in actual_top_left]
    # 计算并保存到 size（尺寸数据）。
    size = [float(value) for value in actual_size]
    # 计算并保存到 all_matches_ok（所有匹配状态是否为ok）。
    all_matches_ok = bool(paths) and all(item.match.status == "ok" for item in paths)
    # 计算并保存到 all_paths_ok（所有路径状态是否为ok）。
    all_paths_ok = bool(paths) and all(item.status == "ok" for item in paths)
    # 计算并保存到 failure_reasons（阻止执行的原因列表）。
    failure_reasons = motion_safety_reasons(paths, config)
    # 计算并保存到 ready_for_motion（计划是否通过发送前的检查）。
    ready_for_motion = not failure_reasons
    # 判断条件；满足时执行下面代码：paths。
    if paths:
        # 计算并保存到 layout_points（layout·点集）。
        layout_points = np.vstack([
            np.asarray(item.match.target_polygon_mm, np.float64) for item in paths
        ])
        # 计算并保存到 layout_minimum（layout·最小）。
        layout_minimum = np.min(layout_points, axis=0)
        # 计算并保存到 layout_maximum（layout·最大）。
        layout_maximum = np.max(layout_points, axis=0)
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 计算并保存到 layout_minimum（layout·最小）。
        layout_minimum = np.asarray(top_left, np.float64)
        # 计算并保存到 layout_maximum（layout·最大）。
        layout_maximum = layout_minimum + np.asarray(size, np.float64)
    # 计算并保存到 clearance_key（留缝·键）。
    clearance_key = ("fixed_target_piece_clearance_mm"
                     # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
                     if str(config.get("puzzle_mode", "fixed_figure_2")) == "fixed_figure_2"
                     else "generic_target_piece_clearance_mm")
    # 计算并保存到 target_clearance（目标·留缝）。
    target_clearance = float(config.get(clearance_key, 0.0))
    # 计算并保存到 target_offset（目标·偏移）。
    target_offset = config.get("motion_target_offset_mm", [0.0, 0.0])
    # 计算并保存到 pieces_payload（碎片集合·输出数据）。
    pieces_payload = [result_dict(item, config) for item in paths]
    # 判断条件；满足时执行下面代码：bool(config.get('motion_optimize_piece_order', True))。
    if bool(config.get("motion_optimize_piece_order", True)):
        # 计算并保存到 pieces_payload（碎片集合·输出数据）、route_distance_mm（行程·距离·毫米）。
        pieces_payload, route_distance_mm = optimize_motion_order(
            pieces_payload, (float(target_offset[0]), float(target_offset[1]))
        )
        # 计算并保存到 order_text（顺序·文本）。
        order_text = "->".join(f"P{item['id']}" for item in pieces_payload)
        # 调用函数：print。
        print(
            f"[路径优化] 执行顺序={order_text}，预计XY正交总行程="
            f"{route_distance_mm:.1f}mm"
        )
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 计算并保存到 route_distance_mm（行程·距离·毫米）。
        route_distance_mm = 0.0

    # 计算并保存到 payload（准备写入文件或发送的数据）。
    payload = {
        # 字典字段'generated_at'（generated_at）：按指定格式生成时间文本。
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        # 字典字段'puzzle_mode'（puzzle·模式）：调用 str。
        "puzzle_mode": str(config.get("puzzle_mode", "fixed_figure_2")),
        # 字典字段'segmentation_mode'（segmentation·模式）：调用 str。
        "segmentation_mode": str(config.get("segmentation_mode", "pink_hsv")),
        # 字典字段'coordinate_system'（coordinate_system）：创建结果字典。
        "coordinate_system": {
            # 字典字段'name'（名称）：'a4_workspace_mm'。
            "name": "a4_workspace_mm",
            # 字典字段'origin'（原点）：'A4_top_left'。
            "origin": "A4_top_left",
            # 字典字段'x_axis'（X轴·axis）：'right'。
            "x_axis": "right",
            # 字典字段'y_axis'（Y轴·axis）：'down'。
            "y_axis": "down",
        },
        # 字典字段'mechanical_workspace_mm'（mechanical·工作区·毫米）：创建结果字典。
        "mechanical_workspace_mm": {
            # 字典字段'size'（尺寸数据）：创建数据容器。
            "size": [float(config["paper_width_mm"]),
                     float(config["paper_height_mm"])],
            # 字典字段'paper_region'（纸面·region）：创建结果字典；字典字段'top_left'（上边·左边）：创建数据容器。
            "paper_region": {"top_left": [0.0, 0.0],
                             # 字典字段'size'（尺寸数据）：创建数据容器。
                             "size": [float(config["paper_width_mm"]),
                                      float(config["paper_height_mm"])]},
            # 字典字段'path_policy'（路径·策略）：'shortest_xy_no_avoidance'。
            "path_policy": "shortest_xy_no_avoidance",
        },
        # 字典字段'rotation_convention'（旋转·convention）：创建结果字典；字典字段'positive'（正向）：'CW'；字典字段'negative'（反向）：'CCW'；
        # 字典字段'unit'（unit）：'degree'。
        "rotation_convention": {"positive": "CW", "negative": "CCW", "unit": "degree"},
        # 字典字段'execution_compensation'（execution_compensation）：创建结果字典。
        "execution_compensation": {
            # 字典字段'target_offset_mm'（目标·偏移·毫米）：创建数据容器。
            "target_offset_mm": [
                float(config.get("motion_target_offset_mm", [0.0, 0.0])[0]),
                float(config.get("motion_target_offset_mm", [0.0, 0.0])[1]),
            ],
            # 字典字段'xy_command_matrix'（XY平面·指令·矩阵）：按条件生成列表。
            "xy_command_matrix": [
                [float(value) for value in row]
                # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
                for row in config.get("motion_xy_command_matrix", [[1.0, 0.0], [0.0, 1.0]])
            ],
            # 字典字段'xy_command_bias_mm'（XY平面·指令·偏置·毫米）：按条件生成列表。
            "xy_command_bias_mm": [
                float(value)
                # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
                for value in config.get("motion_xy_command_bias_mm", [0.0, 0.0])
            ],
            # 字典字段'rotation_magnitude_reduction_deg'（旋转·magnitude·减量·度）：调用 float。
            "rotation_magnitude_reduction_deg": float(
                config.get("motion_rotation_magnitude_reduction_deg", 0.0)
            ),
            # 字典字段'rotation_reduction_min_angle_deg'（旋转·减量·最小·角度·度）：调用 float。
            "rotation_reduction_min_angle_deg": float(
                config.get("motion_rotation_reduction_min_angle_deg", 0.0)
            ),
            # 字典字段'rotation_chunk_deg'（旋转·分段·度）：调用 float。
            "rotation_chunk_deg": float(
                config.get("motion_rotation_chunk_deg", 180.0)
            ),
            # 字典字段'rotation_cycles_per_revolution'（旋转·周期数·每·整圈）：调用 float。
            "rotation_cycles_per_revolution": float(
                config.get("motion_rotation_cycles_per_revolution", 0.0)
            ),
        },
        # 字典字段'execution_timing'（execution·等待时间）：创建结果字典。
        "execution_timing": {
            # 字典字段'magnet_pickup_settle_seconds'（磁铁·吸取·稳定等待·秒）：调用 float。
            "magnet_pickup_settle_seconds": float(
                config.get("magnet_pickup_settle_seconds", 0.0)
            ),
            # 字典字段'magnet_release_settle_seconds'（磁铁·释放·稳定等待·秒）：调用 float。
            "magnet_release_settle_seconds": float(
                config.get("magnet_release_settle_seconds", 0.0)
            ),
            # 字典字段'lift_settle_seconds'（抬升·稳定等待·秒）：调用 float。
            "lift_settle_seconds": float(
                config.get("motion_lift_settle_seconds", 0.0)
            ),
            # 字典字段'xy_settle_seconds'（XY平面·稳定等待·秒）：调用 float。
            "xy_settle_seconds": float(
                config.get("motion_xy_settle_seconds", 0.0)
            ),
            # 字典字段'rotation_settle_seconds'（旋转·稳定等待·秒）：调用 float。
            "rotation_settle_seconds": float(
                config.get("motion_rotation_settle_seconds", 0.0)
            ),
        },
        # 字典字段'execution_policy'（execution·策略）：创建结果字典。
        "execution_policy": {
            # 字典字段'return_origin_after_plan'（返回或回位·原点·操作后·运动计划）：调用 bool。
            "return_origin_after_plan": bool(
                config.get("motion_return_origin_after_plan", False)
            ),
            # 字典字段'rotate_at_target'（rotate·at·目标）：调用 bool。
            "rotate_at_target": bool(
                config.get("motion_rotate_at_target", False)
            ),
            # 字典字段'require_origin_before_plan'（要求·原点·操作前·运动计划）：调用 bool。
            "require_origin_before_plan": bool(
                config.get("motion_require_origin_before_plan", False)
            ),
        },
        # 字典字段'numbering_rule'（numbering_rule）：调用 str。
        "numbering_rule": str(config["numbering_mode"]),
        # 字典字段'execution_order_rule'（execution·顺序·rule）：'minimum_total_orthogonal_xy_distance'。
        "execution_order_rule": "minimum_total_orthogonal_xy_distance",
        # 字典字段'estimated_xy_travel_mm'（estimated·XY平面·travel·毫米）：调用 round。
        "estimated_xy_travel_mm": round(route_distance_mm, 2),
        # 字典字段'magnetic_safe_distance_mm'（magnetic·安全·距离·毫米）：调用 float。
        "magnetic_safe_distance_mm": float(config["magnetic_safe_distance_mm"]),
        # 字典字段'target_region'（目标·region）：创建结果字典；字典字段'name'（名称）：'A4_lower_half'。
        "target_region": {"name": "A4_lower_half",
                          # 字典字段'top_y_mm'（上边·Y轴·毫米）：调用 float。
                          "top_y_mm": float(config.get("target_region_top_mm", 148.5))},
        # 字典字段'target_rectangle_mm'（目标·rectangle·毫米）：创建结果字典；字典字段'top_left'（上边·左边）：top_left（上边·左边）；
        # 字典字段'size'（尺寸数据）：size（尺寸数据）。
        "target_rectangle_mm": {"top_left": top_left, "size": size,
                                # 字典字段'bottom_right'（下边·右边）：创建数据容器。
                                "bottom_right": [top_left[0] + size[0], top_left[1] + size[1]]},
        # 字典字段'target_layout'（目标·layout）：创建结果字典。
        "target_layout": {
            # 字典字段'piece_clearance_mm'（碎片·留缝·毫米）：target_clearance（目标·留缝）。
            "piece_clearance_mm": target_clearance,
            # 字典字段'single_piece_outward_offset_mm'（单项·碎片·向外·偏移·毫米）：计算 target_clearance / 2.0。
            "single_piece_outward_offset_mm": target_clearance / 2.0,
            # 字典字段'corresponding_vertex_limit_mm'（corresponding·vertex·上限·毫米）：调用 float。
            "corresponding_vertex_limit_mm": float(
                config.get("target_corresponding_vertex_limit_mm", 20.0)
            ),
            # 字典字段'actual_bounds_mm'（实测·bounds·毫米）：创建结果字典。
            "actual_bounds_mm": {
                # 字典字段'top_left'（上边·左边）：创建数据容器。
                "top_left": [float(layout_minimum[0]), float(layout_minimum[1])],
                # 字典字段'size'（尺寸数据）：创建数据容器。
                "size": [float(layout_maximum[0] - layout_minimum[0]),
                         float(layout_maximum[1] - layout_minimum[1])],
                # 字典字段'bottom_right'（下边·右边）：创建数据容器。
                "bottom_right": [float(layout_maximum[0]), float(layout_maximum[1])],
            },
        },
        # 字典字段'piece_count'（碎片数量）：调用 len。
        "piece_count": len(paths),
        # 字典字段'path_constraint'（路径·constraint）：'orthogonal_xy_one_axis_at_a_time'。
        "path_constraint": "orthogonal_xy_one_axis_at_a_time",
        # 字典字段'all_matches_ok'（所有匹配状态是否为ok）：all_matches_ok（所有匹配状态是否为ok）。
        "all_matches_ok": all_matches_ok,
        # 字典字段'all_paths_ok'（所有路径状态是否为ok）：all_paths_ok（所有路径状态是否为ok）。
        "all_paths_ok": all_paths_ok,
        # 字典字段'ready_for_motion'（计划是否通过发送前的检查）：ready_for_motion（计划是否通过发送前的检查）。
        "ready_for_motion": ready_for_motion,
        # 字典字段'failure_reasons'（阻止执行的原因列表）：failure_reasons（阻止执行的原因列表）。
        "failure_reasons": failure_reasons,
        # 字典字段'pieces'（碎片列表）：pieces_payload（碎片集合·输出数据）。
        "pieces": pieces_payload,
        # 字典字段'poker_pattern_verification'（扑克牌·pattern·复核）：paths[0].match.pattern_verification if paths and paths[0].match.pattern_verification…。
        "poker_pattern_verification": (
            paths[0].match.pattern_verification if paths
            # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
            and paths[0].match.pattern_verification is not None else None
        ),
    }
    # 打开并自动管理资源，结束后自动释放。
    with (OUTPUT_DIR / "plan.json").open("w", encoding="utf-8") as file:
        # 调用函数：json.dump。
        json.dump(payload, file, ensure_ascii=False, indent=2)
    # 调用函数：print。
    print(f"[OK] 已输出：{OUTPUT_DIR}")
    # 调用函数：print。
    print(f"[OK] Path preview: {OUTPUT_DIR / 'path_preview.jpg'}")
    # 判断条件；满足时执行下面代码：ready_for_motion。
    if ready_for_motion:
        # 调用函数：print。
        print("[OK] ready_for_motion=true")
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 调用函数：print。
        print("[WARN] ready_for_motion=false; safety interlock blocks motion")
        # 遍历数据，逐项处理：failure_reasons。
        for reason in failure_reasons:
            # 调用函数：print。
            print(f"[WARN] {reason}")
    # 遍历数据，逐项处理：paths。
    for result in paths:
        # 计算并保存到 source（源数据或源位置）。
        source = result.piece.pick_mm(float(config["pixels_per_mm"]))
        # 计算并保存到 match（当前匹配记录）。
        match = result.match
        # 调用函数：print。
        print(f"  P{result.piece.piece_id}->模板{match.template_id}: "
              f"源=({source[0]:.1f},{source[1]:.1f})mm "
              f"目标吸取点=({match.target_pick_mm[0]:.1f},{match.target_pick_mm[1]:.1f})mm "
              f"旋转={match.rotation_deg:+.1f}deg {match.rotation_direction} "
              f"路径={result.planner} 状态={result.status}")

# 【函数：pink_mask_in_paper】在原图先识别粉色，再将二值图透视变换，避免远距离放大后颜色被插值冲淡。
# 参数 frame（摄像头的一帧原始图像）：NumPy数组。
# 参数 matrix（本函数使用的矩阵）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为NumPy数组、NumPy数组）；箭头->是类型提示，不会替你转换实际返回值。
def pink_mask_in_paper(frame: np.ndarray, matrix: np.ndarray,
                       config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """在原图先识别粉色，再将二值图透视变换，避免远距离放大后颜色被插值冲淡。"""
    # 计算并保存到 raw_mask（原始·掩膜）。
    raw_mask = make_pink_mask(frame, config, int(config.get("raw_morphology_kernel_px", 3)))
    # 计算并保存到 paper_mask（校正到纸面坐标的掩膜）。
    paper_mask = cv2.warpPerspective(
        # 传入命名参数：flags=cv2.INTER_NEAREST（OpenCV插值或算法选项）；取值过程：cv2.INTER_NEAREST。
        raw_mask, matrix, paper_size_px(config), flags=cv2.INTER_NEAREST
    )
    # 计算并保存到 source_bottom（源·下边）。
    source_bottom = int(round(float(config["source_region_bottom_mm"]) *
                              float(config["pixels_per_mm"])))
    # 计算并保存到 paper_mask[max(0, source_bottom):]。
    paper_mask[max(0, source_bottom):] = 0
    # 返回结果：创建数据容器。
    return raw_mask, paper_mask


# 【函数：mask_in_paper】统一生成原图诊断掩膜和校正纸面掩膜。
# 参数 frame（摄像头的一帧原始图像）：NumPy数组。
# 参数 matrix（本函数使用的矩阵）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为NumPy数组、NumPy数组）；箭头->是类型提示，不会替你转换实际返回值。
def mask_in_paper(frame: np.ndarray, matrix: np.ndarray,
                  config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """统一生成原图诊断掩膜和校正纸面掩膜。"""
    # 计算并保存到 mode（模式）。
    mode = str(config.get("segmentation_mode", "pink_hsv"))
    # 判断条件；满足时执行下面代码：mode == 'pink_hsv'。
    if mode == "pink_hsv":
        # 返回结果：调用 pink_mask_in_paper。
        return pink_mask_in_paper(frame, matrix, config)
    # 判断条件；满足时执行下面代码：mode not in {'background_inverse', 'white_piece', 'poker_v'}。
    if mode not in {"background_inverse", "white_piece", "poker_v"}:
        # 抛出异常，通知上层处理：ValueError(f'未知segmentation_mode：{mode}')。
        raise ValueError(f"未知segmentation_mode：{mode}")
    # 计算并保存到 paper（透视校正后的A4纸面图像）。
    paper = warp_paper(frame, matrix, config)
    # 计算并保存到 paper_mask（校正到纸面坐标的掩膜）。
    paper_mask = make_piece_mask(paper, config)
    # 计算并保存到 inverse（逆变换）。
    inverse = np.linalg.inv(np.asarray(matrix, np.float64))
    # 计算并保存到 raw_mask（原始·掩膜）。
    raw_mask = cv2.warpPerspective(
        # 传入命名参数：flags=cv2.INTER_NEAREST（OpenCV插值或算法选项）；取值过程：cv2.INTER_NEAREST。
        paper_mask, inverse, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST
    )
    # 返回结果：创建数据容器。
    return raw_mask, paper_mask


# 【函数：save_detection_diagnostics】统一保存识别失败所需的三张诊断图。
# 参数 original（原始图像）：NumPy数组。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 mask（二值掩膜（0为背景，非零为选中区域））：NumPy数组。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def save_detection_diagnostics(original: np.ndarray, paper: np.ndarray,
                               mask: np.ndarray) -> None:
    """统一保存识别失败所需的三张诊断图。"""
    # 调用函数：OUTPUT_DIR.mkdir。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 调用函数：write_image。
    write_image(OUTPUT_DIR / "original.jpg", original)
    # 调用函数：write_image。
    write_image(OUTPUT_DIR / "corrected.jpg", paper)
    # 调用函数：write_image。
    write_image(OUTPUT_DIR / "mask.png", mask)
    # 调用函数：print。
    print("[诊断] 已保存original.jpg、corrected.jpg和mask.png，请据此检查标定与分割")


# 【函数：process_paper】串联检测、编号、拼图求解、路径规划和结果保存。
# 这是理解数据流的核心函数；处理过程中会保存文件，show只控制预览窗口，不能理解成“不产生输出”。
# 参数 original（原始图像）：NumPy数组。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 show（是否显示预览窗口）：布尔值True/False。
# 参数 mask_override（外部提供的掩膜，None表示由检测函数生成）：NumPy数组或None（不返回业务结果）；省略时使用None。
# 返回类型：元组（依次为Piece列表/序列、PathResult列表/序列）；箭头->是类型提示，不会替你转换实际返回值。
def process_paper(original: np.ndarray, paper: np.ndarray,
                  config: dict[str, Any], show: bool,
                  mask_override: np.ndarray | None = None) -> tuple[list[Piece], list[PathResult]]:
    # 计算并保存到 pieces（碎片列表）、mask（二值掩膜（0为背景，非零为选中区域））。
    pieces, mask = detect_pieces(paper, config, diagnostics=True, mask_override=mask_override)
    # 计算并保存到 pieces（碎片列表）。
    pieces = number_by_move_order(pieces, config)

    # 计算并保存到 quality_reasons（质量·原因）。
    quality_reasons = calibration_quality_reasons(paper, config)
    # 计算并保存到 border_contacts（边框·contacts）。
    border_contacts = piece_border_contacts(pieces, paper.shape[:2], config)
    # 遍历数据，逐项处理：quality_reasons。
    for reason in quality_reasons:
        # 调用函数：print。
        print(f"[WARN] 标定诊断：{reason}；继续尝试识别与求解")
    # 判断条件；满足时执行下面代码：border_contacts。
    if border_contacts:
        # 调用函数：print。
        print(f"[WARN] 边界诊断：{'；'.join(border_contacts)}；继续尝试识别与求解")

    # 计算并保存到 mode（模式）。
    mode = str(config.get("puzzle_mode", "fixed_figure_2"))
    # 计算并保存到 expected（期望）。
    expected = int(config.get("expected_piece_count", 4))
    # 计算并保存到 count_ok（数量·ok）。
    count_ok = (len(pieces) == expected if mode == "fixed_figure_2" else
                1 <= len(pieces) <= int(config.get("max_pieces", 4)))
    # 判断条件；满足时执行下面代码：not count_ok。
    if not count_ok:
        # 调用函数：save_detection_diagnostics。
        save_detection_diagnostics(original, paper, mask)
        # 判断条件；满足时执行下面代码：mode == 'fixed_figure_2'。
        if mode == "fixed_figure_2":
            # 抛出异常，通知上层处理：RuntimeError(f'固定图2必须完整识别{expected}块，当前识别到{len(pieces)}块')。
            raise RuntimeError(f"固定图2必须完整识别{expected}块，当前识别到{len(pieces)}块")
        # 判断条件；满足时执行下面代码：mode == 'generic_geometry'。
        if mode == "generic_geometry":
            # 抛出异常，通知上层处理：RuntimeError(f'通用拼图必须识别1～4块，当前识别到{len(pieces)}块')。
            raise RuntimeError(f"通用拼图必须识别1～4块，当前识别到{len(pieces)}块")
        # 抛出异常，通知上层处理：RuntimeError(f'未知puzzle_mode：{mode}')。
        raise RuntimeError(f"未知puzzle_mode：{mode}")
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 计算并保存到 matches（匹配记录列表）。
        matches = solve_puzzle(pieces, config, paper)
    # 捕获ValueError异常，转入下面的处理代码；异常对象保存在exc。
    except ValueError as exc:
        # 调用函数：save_detection_diagnostics。
        save_detection_diagnostics(original, paper, mask)
        # 抛出异常，通知上层处理：RuntimeError(str(exc))。
        raise RuntimeError(str(exc)) from exc
    # 计算并保存到 paths（各碎片的搬运路径结果）。
    paths = plan_all_paths(pieces, config, matches)
    # 计算并保存到 preview（路径预览图）。
    preview = draw_preview(paper, pieces, paths, config)
    # 调用函数：save_outputs。
    save_outputs(original, paper, mask, preview, paths, config)
    # 判断条件；满足时执行下面代码：show。
    if show:
        # 调用函数：cv2.namedWindow。
        cv2.namedWindow("Path Preview", cv2.WINDOW_NORMAL)
        # 调用函数：cv2.imshow。
        cv2.imshow("Path Preview", preview)
    # 返回结果：创建数据容器。
    return pieces, paths

# =============================================================================
# 【分区】运行入口（demo / 单图 / 相机预览 / 自动发计划）
# 功能：合成演示、处理一张图、打开相机循环、计划就绪时可选自动串口发送。
# 可修改：是否 show 窗口；auto_send 相关配置和 --send。
# 看情况改：训练用 --demo 不需相机；比赛用 --auto。
# 不要改：process_paper 的主链路（warp→detect→solve→plan→save）。
# =============================================================================
# ------------------------- 运行入口 -------------------------

# 【函数：make_demo_paper】使用固定图2顶点合成粉色碎片演示图。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def make_demo_paper(config: dict[str, Any]) -> np.ndarray:
    # 计算并保存到 width（当前区域宽度）、height（当前区域高度）。
    width, height = paper_size_px(config)
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 paper（透视校正后的A4纸面图像）。
    paper = np.full((height, width, 3), (245, 245, 245), np.uint8)
    # 计算并保存到 templates。
    templates = figure2_template_polygons(config, placed=False)
    # 计算并保存到 colors（颜色集合）。
    colors = [(205, 140, 250), (195, 130, 245), (215, 150, 250), (200, 135, 245)]
    # 遍历数据，逐项处理：zip(templates.items(), colors)。
    for (template_id, polygon), color in zip(templates.items(), colors):
        # 计算并保存到 center_mm（视觉质心的毫米坐标(x,y)）、angle_deg（角度，单位为度）。
        center_mm, angle_deg = DEMO_PLACEMENTS[template_id]
        # 计算并保存到 local（局部）。
        local = polygon - polygon_centroid(polygon)
        # 计算并保存到 transformed。
        transformed = rotate_points(local, angle_deg) + np.asarray(center_mm, np.float64)
        # 计算并保存到 points（参与当前计算的一组坐标点）。
        points = np.rint(transformed * ppm).astype(np.int32)
        # 调用函数：cv2.fillPoly。
        cv2.fillPoly(paper, [points], color)
        # 调用函数：cv2.polylines。
        cv2.polylines(paper, [points], True, (50, 50, 50), 2)
    # 返回结果：paper（透视校正后的A4纸面图像）。
    return paper

# 【函数：run_demo】用固定模板和粉色分割配置运行合成图演示。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def run_demo(config: dict[str, Any]) -> None:
    # 合成样本来自固定图2，演示必须使用与样本匹配的固定求解和粉色分割。
    # 使用副本，避免覆盖用户为摄像头实测设置的通用模式。
    # 计算并保存到 demo_config（演示·配置）。
    demo_config = dict(config)
    # 计算并保存到 demo_config['puzzle_mode']。
    demo_config["puzzle_mode"] = "fixed_figure_2"
    # 计算并保存到 demo_config['segmentation_mode']。
    demo_config["segmentation_mode"] = "pink_hsv"
    # 计算并保存到 demo_config['expected_piece_count']。
    demo_config["expected_piece_count"] = 4
    # 调用函数：print。
    print("[演示] 使用固定图2 + pink_hsv合成样本；不会修改config.json中的实测模式。")
    # 计算并保存到 paper（透视校正后的A4纸面图像）。
    paper = make_demo_paper(demo_config)
    # 调用函数：process_paper。
    process_paper(paper, paper, demo_config, False)
    # 调用函数：print。
    print("[OK] 合成演示完成，请查看output/path_preview.jpg和output/plan.json")


# 【函数：run_image】读取指定图片，必要时透视校正，然后完成一次检测与求解。
# 参数 path（当前文件路径或几何路径）：文件系统路径对象。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def run_image(path: Path, config: dict[str, Any]) -> None:
    # 计算并保存到 image（图像数组）。
    image = read_image(path)
    # 判断条件；满足时执行下面代码：image is None。
    if image is None:
        # 抛出异常，通知上层处理：RuntimeError(f'无法读取图片：{path}')。
        raise RuntimeError(f"无法读取图片：{path}")
    # 计算并保存到 width（当前区域宽度）、height（当前区域高度）。
    width, height = paper_size_px(config)
    # 计算并保存到 matrix（本函数使用的矩阵）。
    matrix = load_calibration(config)
    # 计算并保存到 mask_override（外部提供的掩膜，None表示由检测函数生成）。
    mask_override = None
    # 判断条件；满足时执行下面代码：image.shape[:2] == (height, width)。
    if image.shape[:2] == (height, width):
        # 计算并保存到 paper（透视校正后的A4纸面图像）。
        paper = image.copy()
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 计算并保存到 matrix（本函数使用的矩阵）、_（此处不需要使用的返回值或循环占位变量）。
        matrix, _ = resolve_calibration_matrix(image, config, matrix)
        # 计算并保存到 paper（透视校正后的A4纸面图像）。
        paper = warp_paper(image, matrix, config)
        # 计算并保存到 _（此处不需要使用的返回值或循环占位变量）、mask_override（外部提供的掩膜，None表示由检测函数生成）。
        _, mask_override = mask_in_paper(image, matrix, config)
    # 调用函数：process_paper。
    process_paper(image, paper, config, False, mask_override)


# 【函数：auto_send_plan_if_ready】识别求解成功且安全联锁通过时，通过串口发送运动计划驱动STM32。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def auto_send_plan_if_ready(config: dict[str, Any]) -> None:
    """识别求解成功且安全联锁通过时，通过串口发送运动计划驱动STM32。"""
    # 计算并保存到 plan_path（运动计划JSON文件路径）。
    plan_path = OUTPUT_DIR / "plan.json"
    # 判断条件；满足时执行下面代码：not plan_path.is_file()。
    if not plan_path.is_file():
        # 调用函数：print。
        print("[WARN] 未找到output/plan.json，未发送运动指令")
        # 返回结果：无。
        return
    # 计算并保存到 plan（从JSON读取的运动计划字典）。
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    # 判断条件；满足时执行下面代码：not plan.get('ready_for_motion')。
    if not plan.get("ready_for_motion"):
        # 调用函数：print。
        print("[WARN] ready_for_motion=false，安全联锁拒绝发送运动指令")
        # 返回结果：无。
        return
    # 计算并保存到 port（串口设备名称）。
    port = str(config.get("serial_port", ""))
    # 判断条件；满足时执行下面代码：not port。
    if not port:
        # 调用函数：print。
        print("[WARN] config未配置serial_port，未发送运动指令")
        # 返回结果：无。
        return
    # 调用函数：print。
    print("[自动] 开始通过串口发送运动计划...")
    # 调用函数：serial_transport.send_plan_file。
    serial_transport.send_plan_file(
        plan_path,
        # 传入命名参数：dry_run=False（是否只打印命令而不发送）；取值过程：False。
        dry_run=False,
        # 传入命名参数：port=port（串口设备名称）；取值过程：port（串口设备名称）。
        port=port,
        # 传入命名参数：baudrate=int(config.get('serial_baudrate', 115200))（串口波特率）；取值过程：调用 int。
        baudrate=int(config.get("serial_baudrate", 115200)),
        # 传入命名参数：ack_timeout_seconds=float(config.get('serial_ack_timeout_seconds', 2.0))（应答·超时·秒）；取值过程：调用 float。
        ack_timeout_seconds=float(config.get("serial_ack_timeout_seconds", 2.0)),
        # 传入命名参数：command_delay_seconds=float(config.get('serial_command_delay_seconds', 0.0))（指令·间隔·秒）；
        # 取值过程：调用 float。
        command_delay_seconds=float(config.get("serial_command_delay_seconds", 0.0)),
    )
    # 调用函数：print。
    print("[自动] 运动计划已发送完成")


# 【函数：run_camera】持续预览摄像头，处理标定、模式选择和START等交互动作。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 camera_id（摄像头设备编号）：整数或None（不返回业务结果）。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def run_camera(config: dict[str, Any], camera_id: int | None) -> None:
    # 计算并保存到 capture（已打开的摄像头对象）。
    capture = open_camera(config, camera_id)
    # 计算并保存到 matrix（本函数使用的矩阵）。
    matrix = load_calibration(config)
    # 计算并保存到 window（OpenCV窗口名称）。
    window = "USB Puzzle Vision"
    # 调用函数：cv2.namedWindow。
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # 判断条件；满足时执行下面代码：hasattr(cv2, 'WND_PROP_FULLSCREEN')。
    if hasattr(cv2, "WND_PROP_FULLSCREEN"):
        # 执行可能出错的代码，并由后面的异常分支处理错误。
        try:
            # 调用函数：cv2.setWindowProperty。
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # 捕获cv2.error异常，转入下面的处理代码。
        except cv2.error:
            pass

    # 计算并保存到 buttons。
    buttons = {
        # 字典字段'calibrate'（calibrate）：创建数据容器。
        "calibrate": (10, 10, 170, 58),
        # 字典字段'detect'（detect）：创建数据容器。
        "detect": (180, 10, 320, 58),
        # 字典字段'poker'（扑克牌）：创建数据容器。
        "poker": (330, 10, 470, 58),
        # 字典字段'capture'（已打开的摄像头对象）：创建数据容器。
        "capture": (480, 10, 630, 58),
        # 字典字段'quit'（quit）：创建数据容器。
        "quit": (640, 10, 740, 58),
    }
    # 计算并保存到 pending_action。
    pending_action: dict[str, str | None] = {"value": None}

    # 【函数：on_mouse】接收OpenCV鼠标事件，根据点击位置更新交互状态。
    # 参数 event：整数。
    # 参数 x（当前横坐标或横向数据）：整数。
    # 参数 y（当前纵坐标或纵向数据）：整数。
    # 参数 flags：整数。
    # 参数 param：object。
    # 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        # 判断条件；满足时执行下面代码：event != cv2.EVENT_LBUTTONDOWN。
        if event != cv2.EVENT_LBUTTONDOWN:
            # 返回结果：无。
            return
        # 遍历数据，逐项处理：buttons.items()。
        for action, (x1, y1, x2, y2) in buttons.items():
            # 判断条件；满足时执行下面代码：x1 <= x <= x2 and y1 <= y <= y2。
            if x1 <= x <= x2 and y1 <= y <= y2:
                # 计算并保存到 pending_action['value']。
                pending_action["value"] = action
                # 返回结果：无。
                return

    # 调用函数：cv2.setMouseCallback。
    cv2.setMouseCallback(window, on_mouse)
    # 调用函数：print。
    print("[操作] 直接点击DETECT即可自动定位底板并识别；CALIBRATE仅作失败兜底。")
    # 调用函数：print。
    print("[操作] POKER用于扑克牌碎片（白底花纹）识别+拼合+花纹验证，与DETECT共用标定。")
    # 调用函数：print。
    print("[操作] 键盘D识别，P扑克牌，K手动标定，C保存原图，Q退出。")

    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 只要条件成立就重复执行：True。
        while True:
            # 计算并保存到 success（本步骤是否成功）、frame（摄像头的一帧原始图像）。
            success, frame = capture.read()
            # 判断条件；满足时执行下面代码：not success。
            if not success:
                # 抛出异常，通知上层处理：RuntimeError('摄像头读取失败')。
                raise RuntimeError("摄像头读取失败")

            # 计算并保存到 display（用于窗口显示的画布）。
            display = frame.copy()
            # 调用函数：cv2.rectangle。
            cv2.rectangle(display, (0, 0), (display.shape[1], 100), (25, 25, 25), -1)
            # 计算并保存到 button_colors（button·颜色集合）。
            button_colors = {
                # 字典字段'calibrate'（calibrate）：创建数据容器。
                "calibrate": (0, 170, 255),
                # 字典字段'detect'（detect）：创建数据容器。
                "detect": (0, 190, 0),
                # 字典字段'poker'（扑克牌）：创建数据容器。
                "poker": (200, 90, 200),
                # 字典字段'capture'（已打开的摄像头对象）：创建数据容器。
                "capture": (200, 130, 0),
                # 字典字段'quit'（quit）：创建数据容器。
                "quit": (0, 0, 200),
            }
            # 计算并保存到 labels。
            labels = {
                # 字典字段'calibrate'（calibrate）：'CALIBRATE'。
                "calibrate": "CALIBRATE",
                # 字典字段'detect'（detect）：'DETECT'。
                "detect": "DETECT",
                # 字典字段'poker'（扑克牌）：'POKER'。
                "poker": "POKER",
                # 字典字段'capture'（已打开的摄像头对象）：'CAPTURE'。
                "capture": "CAPTURE",
                # 字典字段'quit'（quit）：'QUIT'。
                "quit": "QUIT",
            }
            # 遍历数据，逐项处理：buttons.items()。
            for action, (x1, y1, x2, y2) in buttons.items():
                # 调用函数：cv2.rectangle。
                cv2.rectangle(display, (x1, y1), (x2, y2), button_colors[action], -1)
                # 调用函数：cv2.putText。
                cv2.putText(display, labels[action], (x1 + 10, y1 + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255),
                            2, cv2.LINE_AA)
            # 计算并保存到 auto_enabled（自动·启用）。
            auto_enabled = bool(config.get("auto_board_calibration", True))
            # 计算并保存到 status（处理结果的状态标记）。
            status = ("AUTO BOARD - click DETECT" if auto_enabled else
                      ("CALIBRATED - click DETECT" if matrix is not None else
                       "NOT CALIBRATED - click CALIBRATE"))
            # 计算并保存到 status_color（状态·颜色）。
            status_color = (0, 255, 0) if auto_enabled or matrix is not None else (0, 180, 255)
            # 调用函数：cv2.putText。
            cv2.putText(display, status, (12, 87), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, status_color, 2, cv2.LINE_AA)
            # 调用函数：cv2.imshow。
            cv2.imshow(window, display)

            # 计算并保存到 key（查询、分组或排序所用的键）。
            key = cv2.waitKeyEx(10)
            # 计算并保存到 action。
            action = pending_action["value"]
            # 计算并保存到 pending_action['value']。
            pending_action["value"] = None
            # 计算并保存到 low_key（low·键）。
            low_key = key & 0xFF if key >= 0 else -1
            # 判断条件；满足时执行下面代码：low_key in (ord('k'), ord('K'))。
            if low_key in (ord("k"), ord("K")):
                # 计算并保存到 action。
                action = "calibrate"
            # 判断条件；满足时执行下面代码：low_key in (ord('d'), ord('D'))。
            elif low_key in (ord("d"), ord("D")):
                # 计算并保存到 action。
                action = "detect"
            # 判断条件；满足时执行下面代码：low_key in (ord('p'), ord('P'))。
            elif low_key in (ord("p"), ord("P")):
                # 计算并保存到 action。
                action = "poker"
            # 判断条件；满足时执行下面代码：low_key in (ord('c'), ord('C'))。
            elif low_key in (ord("c"), ord("C")):
                # 计算并保存到 action。
                action = "capture"
            # 判断条件；满足时执行下面代码：low_key in (ord('q'), ord('Q'), 27)。
            elif low_key in (ord("q"), ord("Q"), 27):
                # 计算并保存到 action。
                action = "quit"

            # 判断条件；满足时执行下面代码：action == 'quit'。
            if action == "quit":
                # 立即结束当前循环。
                break
            # 判断条件；满足时执行下面代码：action == 'capture'。
            if action == "capture":
                # 调用函数：OUTPUT_DIR.mkdir。
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                # 调用函数：write_image。
                write_image(OUTPUT_DIR / "camera_capture.jpg", frame)
                # 调用函数：print。
                print("[OK] 已保存output/camera_capture.jpg")
            # 判断条件；满足时执行下面代码：action == 'calibrate'。
            elif action == "calibrate":
                # 计算并保存到 new_matrix（new·矩阵）。
                new_matrix = collect_four_points(frame, config)
                # 判断条件；满足时执行下面代码：new_matrix is not None。
                if new_matrix is not None:
                    # 计算并保存到 matrix（本函数使用的矩阵）。
                    matrix = new_matrix
                    # 返回主窗口后重新注册回调，避免部分OpenCV后端丢失事件。
                    # 调用函数：cv2.setMouseCallback。
                    cv2.setMouseCallback(window, on_mouse)
            # 判断条件；满足时执行下面代码：action == 'detect'。
            elif action == "detect":
                # 执行可能出错的代码，并由后面的异常分支处理错误。
                try:
                    # 计算并保存到 active_matrix（本次识别实际采用的透视矩阵）、source（源数据或源位置）。
                    active_matrix, source = resolve_calibration_matrix(frame, config, matrix)
                    # 判断条件；满足时执行下面代码：source == 'auto_board'。
                    if source == "auto_board":
                        # 计算并保存到 matrix（本函数使用的矩阵）。
                        matrix = active_matrix
                    # 计算并保存到 paper（透视校正后的A4纸面图像）。
                    paper = warp_paper(frame, active_matrix, config)
                    # 计算并保存到 raw_mask（原始·掩膜）、paper_mask（校正到纸面坐标的掩膜）。
                    raw_mask, paper_mask = mask_in_paper(frame, active_matrix, config)
                    # 调用函数：OUTPUT_DIR.mkdir。
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    # 计算并保存到 raw_name（原始·名称）。
                    raw_name = ("raw_pink_mask.png" if
                                str(config.get("segmentation_mode", "pink_hsv")) == "pink_hsv"
                                else "raw_piece_mask.png")
                    # 调用函数：write_image。
                    write_image(OUTPUT_DIR / raw_name, raw_mask)
                    # 计算并保存到 detect_config（detect·配置）。
                    detect_config = dict(config)
                    # 计算并保存到 detect_config['poker_pattern_verification']。
                    detect_config["poker_pattern_verification"] = False
                    # 调用函数：process_paper。
                    process_paper(frame, paper, detect_config, True, paper_mask)
                    # 若求解成功且安全联锁通过，则通过串口发送运动计划驱动STM32
                    # 执行可能出错的代码，并由后面的异常分支处理错误。
                    try:
                        # 调用函数：auto_send_plan_if_ready。
                        auto_send_plan_if_ready(detect_config)
                    # 捕获Exception异常，转入下面的处理代码；异常对象保存在exc。
                    except Exception as exc:
                        # 调用函数：print。
                        print(f"[WARN] 串口发送失败(不影响识别): {exc}")
                # 捕获RuntimeError异常，转入下面的处理代码；异常对象保存在exc。
                except RuntimeError as exc:
                    # 调用函数：print。
                    print(f"[ERR] {exc}")
            # 判断条件；满足时执行下面代码：action == 'poker'。
            elif action == "poker":
                # 计算并保存到 poker_config（扑克牌·配置）。
                poker_config = dict(config)
                # 计算并保存到 poker_config['puzzle_mode']。
                poker_config["puzzle_mode"] = "generic_geometry"
                # 计算并保存到 poker_config['segmentation_mode']。
                poker_config["segmentation_mode"] = "poker_v"
                # 计算并保存到 poker_config['generic_target_long_min_mm']。
                poker_config["generic_target_long_min_mm"] = float(
                    config.get("poker_target_long_min_mm", 80.0))
                # 计算并保存到 poker_config['generic_target_long_max_mm']。
                poker_config["generic_target_long_max_mm"] = float(
                    config.get("poker_target_long_max_mm", 95.0))
                # 计算并保存到 poker_config['generic_target_short_min_mm']。
                poker_config["generic_target_short_min_mm"] = float(
                    config.get("poker_target_short_min_mm", 50.0))
                # 计算并保存到 poker_config['generic_target_short_max_mm']。
                poker_config["generic_target_short_max_mm"] = float(
                    config.get("poker_target_short_max_mm", 62.0))
                # 计算并保存到 poker_config['generic_max_fill_error_ratio']。
                poker_config["generic_max_fill_error_ratio"] = float(
                    config.get("poker_max_fill_error_ratio", 0.35))
                # 计算并保存到 poker_config['generic_max_boundary_gap_ratio']。
                poker_config["generic_max_boundary_gap_ratio"] = float(
                    config.get("poker_max_boundary_gap_ratio", 0.30))
                # 计算并保存到 poker_config['generic_edge_abs_tolerance_mm']。
                poker_config["generic_edge_abs_tolerance_mm"] = float(
                    config.get("poker_edge_abs_tolerance_mm", 3.0))
                # 计算并保存到 poker_config['generic_edge_rel_tolerance']。
                poker_config["generic_edge_rel_tolerance"] = float(
                    config.get("poker_edge_rel_tolerance", 0.08))
                # 计算并保存到 poker_config['generic_min_edge_length_mm']。
                poker_config["generic_min_edge_length_mm"] = float(
                    config.get("poker_min_edge_length_mm", 4.0))
                # 计算并保存到 poker_config['generic_max_edge_candidates']。
                poker_config["generic_max_edge_candidates"] = int(
                    config.get("poker_max_edge_candidates", 80))
                # 计算并保存到 poker_config['generic_search_timeout_seconds']。
                poker_config["generic_search_timeout_seconds"] = float(
                    config.get("poker_search_timeout_seconds", 10.0))
                # 计算并保存到 poker_config['generic_max_search_states']。
                poker_config["generic_max_search_states"] = int(
                    config.get("poker_max_search_states", 60000))
                # 计算并保存到 poker_config['generic_require_each_piece_outer_edge']。
                poker_config["generic_require_each_piece_outer_edge"] = bool(
                    config.get("poker_require_outer_edge", False))
                # 计算并保存到 poker_config['poker_pattern_verification']。
                poker_config["poker_pattern_verification"] = bool(
                    config.get("poker_pattern_verification", True))
                # 执行可能出错的代码，并由后面的异常分支处理错误。
                try:
                    # 与DETECT共用手动标定（calibration.json）；仅本模式固定使用标定矩阵。
                    # 计算并保存到 active_matrix（本次识别实际采用的透视矩阵）、source（源数据或源位置）。
                    active_matrix, source = resolve_calibration_matrix(
                        frame, poker_config, matrix
                    )
                    # 判断条件；满足时执行下面代码：source == 'auto_board'。
                    if source == "auto_board":
                        # 计算并保存到 matrix（本函数使用的矩阵）。
                        matrix = active_matrix
                    # 计算并保存到 paper（透视校正后的A4纸面图像）。
                    paper = warp_paper(frame, active_matrix, poker_config)
                    # 计算并保存到 _（此处不需要使用的返回值或循环占位变量）、paper_mask（校正到纸面坐标的掩膜）。
                    _, paper_mask = mask_in_paper(frame, active_matrix, poker_config)
                    # 调用函数：OUTPUT_DIR.mkdir。
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    # 调用函数：write_image。
                    write_image(OUTPUT_DIR / "raw_poker_mask.png", paper_mask)
                    # 调用函数：process_paper。
                    process_paper(frame, paper, poker_config, True, paper_mask)
                    # 若求解成功且花纹验证与安全联锁均通过，则通过串口发送运动计划
                    # 调用函数：auto_send_plan_if_ready。
                    auto_send_plan_if_ready(poker_config)
                # 捕获RuntimeError异常，转入下面的处理代码；异常对象保存在exc。
                except RuntimeError as exc:
                    # 调用函数：print。
                    print(f"[ERR] {exc}")

            # 判断条件；满足时执行下面代码：not window_is_visible(window)。
            if not window_is_visible(window):
                # 立即结束当前循环。
                break
    # 无论正常结束、return还是抛出异常都会进入这里，用于释放相机/串口等资源。
    finally:
        # 调用函数：capture.release。
        capture.release()
        # 调用函数：cv2.destroyAllWindows。
        cv2.destroyAllWindows()
        # 调用函数：print。
        print("[OK] 摄像头已释放")


# =============================================================================
# 【分区】一键自动流程与终局视觉复核
# 功能：等 START → 丢前几帧取稳定图 → 求解搬运 → 可选串口执行 → 再拍一张核对位置/转角。
# 可修改：START 按钮热区、丢帧数量、终检公差（final_verify_* 配置）。
# 看情况改：灯光慢、曝光长就增加稳定等待；终检失败说明机械补偿或吸取点有问题，不要先放宽几何求解。
# 不要改：触发时刻取 START 按下瞬间（与赛题计时一致）；复核失败不得报 DONE。
# =============================================================================
# ------------------------- 一键自动流程 -------------------------

# 【函数：wait_for_start_and_capture】显示实时预览等待START触发，随后按题目时序丢弃前几帧再取稳定帧。
# 参数 capture（已打开的摄像头对象）：cv2.VideoCapture。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为NumPy数组、浮点数）或None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def wait_for_start_and_capture(
        capture: cv2.VideoCapture,
        config: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    """显示实时预览等待START触发，随后按题目时序丢弃前几帧再取稳定帧。

    返回(稳定帧, 触发时刻)；触发时刻取自START按下的瞬间（早于曝光稳定等待），
    与赛题"一键启动同时移除遮挡、开始计时"的时间起点一致，供调用方统计总耗时。
    返回None表示用户主动退出（Q/Esc/关闭窗口），此时调用方应结束整个自动流程。
    """
    # 计算并保存到 window（OpenCV窗口名称）。
    window = "Auto Puzzle Vision"
    # 调用函数：cv2.namedWindow。
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # 计算并保存到 button。
    button = (10, 10, 220, 58)
    # 计算并保存到 triggered。
    triggered = {"value": False}

    # 【函数：on_mouse】接收OpenCV鼠标事件，根据点击位置更新交互状态。
    # 参数 event：整数。
    # 参数 x（当前横坐标或横向数据）：整数。
    # 参数 y（当前纵坐标或纵向数据）：整数。
    # 参数 flags：整数。
    # 参数 param：object。
    # 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        # 计算并保存到 x1、y1、x2、y2。
        x1, y1, x2, y2 = button
        # 判断条件；满足时执行下面代码：event == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and (y1 <= y <= y2)。
        if event == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and y1 <= y <= y2:
            # 计算并保存到 triggered['value']。
            triggered["value"] = True

    # 调用函数：cv2.setMouseCallback。
    cv2.setMouseCallback(window, on_mouse)
    # 调用函数：print。
    print("[自动] 遮挡摄像头、随机摆放碎片后，按空格/回车或点击START；Q退出。")

    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 只要条件成立就重复执行：True。
        while True:
            # 计算并保存到 success（本步骤是否成功）、frame（摄像头的一帧原始图像）。
            success, frame = capture.read()
            # 判断条件；满足时执行下面代码：not success。
            if not success:
                # 抛出异常，通知上层处理：RuntimeError('摄像头读取失败')。
                raise RuntimeError("摄像头读取失败")
            # 计算并保存到 display（用于窗口显示的画布）。
            display = frame.copy()
            # 计算并保存到 x1、y1、x2、y2。
            x1, y1, x2, y2 = button
            # 调用函数：cv2.rectangle。
            cv2.rectangle(display, (0, 0), (display.shape[1], 70), (25, 25, 25), -1)
            # 调用函数：cv2.rectangle。
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 190, 0), -1)
            # 调用函数：cv2.putText。
            cv2.putText(display, "START", (x1 + 30, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 255), 2, cv2.LINE_AA)
            # 调用函数：cv2.putText。
            cv2.putText(display, "Cover camera, place pieces, then START (Q to quit)",
                       (x2 + 15, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       (255, 255, 255), 2, cv2.LINE_AA)
            # 调用函数：cv2.imshow。
            cv2.imshow(window, display)

            # 计算并保存到 key（查询、分组或排序所用的键）。
            key = cv2.waitKeyEx(20)
            # 计算并保存到 low_key（low·键）。
            low_key = key & 0xFF if key >= 0 else -1
            # 判断条件；满足时执行下面代码：low_key in (32, 13) or triggered['value']。
            if low_key in (32, 13) or triggered["value"]:
                # 立即结束当前循环。
                break
            # 判断条件；满足时执行下面代码：low_key in (ord('q'), ord('Q'), 27) or not window_is_visible(window)。
            if low_key in (ord("q"), ord("Q"), 27) or not window_is_visible(window):
                # 调用函数：safe_destroy_window。
                safe_destroy_window(window)
                # 返回结果：None。
                return None

        # 计算并保存到 trigger_time。
        trigger_time = time.monotonic()
        # 调用函数：print。
        print("[自动] 已触发，等待遮挡移除与曝光稳定...")
        # 计算并保存到 warmup_seconds（warmup·秒）。
        warmup_seconds = float(config.get("auto_warmup_seconds", 0.8))
        # 计算并保存到 deadline（单调时钟上的截止时刻（秒））。
        deadline = time.monotonic() + warmup_seconds
        # 计算并保存到 stable_frame（stable·图像帧）。
        stable_frame: np.ndarray | None = None
        # 只要条件成立就重复执行：time.monotonic() < deadline。
        while time.monotonic() < deadline:
            # 计算并保存到 success（本步骤是否成功）、stable_frame（stable·图像帧）。
            success, stable_frame = capture.read()
            # 判断条件；满足时执行下面代码：not success。
            if not success:
                # 抛出异常，通知上层处理：RuntimeError('摄像头读取失败')。
                raise RuntimeError("摄像头读取失败")
            # 调用函数：cv2.imshow。
            cv2.imshow(window, stable_frame)
            # 调用函数：cv2.waitKey。
            cv2.waitKey(1)
        # 再多取几帧使曝光/白平衡收敛，使用最后一帧作为计算帧。
        # 遍历数据，逐项处理：range(int(config.get('auto_extra_settle_frames', 5)))。
        for _ in range(int(config.get("auto_extra_settle_frames", 5))):
            # 计算并保存到 success（本步骤是否成功）、stable_frame（stable·图像帧）。
            success, stable_frame = capture.read()
            # 判断条件；满足时执行下面代码：not success。
            if not success:
                # 抛出异常，通知上层处理：RuntimeError('摄像头读取失败')。
                raise RuntimeError("摄像头读取失败")
        # 调用函数：safe_destroy_window。
        safe_destroy_window(window)
        # 返回结果：创建数据容器。
        return stable_frame, trigger_time
    # 捕获Exception异常，转入下面的处理代码。
    except Exception:
        # 调用函数：safe_destroy_window。
        safe_destroy_window(window)
        # 抛出异常，通知上层处理：。
        raise



# 【函数：polygon_iou_mm】在A4坐标中计算两个已放置多边形的IoU。
# 参数 first_mm（第一·毫米）：NumPy数组。
# 参数 second_mm（第二·毫米）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def polygon_iou_mm(first_mm: np.ndarray, second_mm: np.ndarray,
                   config: dict[str, Any]) -> float:
    """在A4坐标中计算两个已放置多边形的IoU。"""
    # 计算并保存到 resolution（栅格分辨率）。
    resolution = max(1.0, float(config.get("final_verify_pixels_per_mm", 3.0)))
    # 计算并保存到 width（当前区域宽度）。
    width = int(math.ceil(float(config["paper_width_mm"]) * resolution)) + 3
    # 计算并保存到 height（当前区域高度）。
    height = int(math.ceil(float(config["paper_height_mm"]) * resolution)) + 3
    # 计算并保存到 first_mask（第一·掩膜）。
    first_mask = np.zeros((height, width), np.uint8)
    # 计算并保存到 second_mask（第二·掩膜）。
    second_mask = np.zeros((height, width), np.uint8)
    # 调用函数：cv2.fillPoly。
    cv2.fillPoly(first_mask, [np.rint(np.asarray(first_mm) * resolution).astype(np.int32)], 1)
    # 调用函数：cv2.fillPoly。
    cv2.fillPoly(second_mask, [np.rint(np.asarray(second_mm) * resolution).astype(np.int32)], 1)
    # 计算并保存到 intersection（交集大小）。
    intersection = int(np.count_nonzero((first_mask > 0) & (second_mask > 0)))
    # 计算并保存到 union（并集大小）。
    union = int(np.count_nonzero((first_mask > 0) | (second_mask > 0)))
    # 返回结果：intersection / union if union else 0.0。
    return intersection / union if union else 0.0


# 【函数：final_verification_targets】从运动计划提取最终复核所需的目标轮廓。
# 参数 plan（从JSON读取的运动计划字典）：字典（键为字符串，值为任意类型）。
# 返回类型：字典（键为字符串，值为任意类型）列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def final_verification_targets(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """从运动计划提取最终复核所需的目标轮廓。"""
    # 计算并保存到 targets。
    targets: list[dict[str, Any]] = []
    # 遍历数据，逐项处理：plan.get('pieces', [])。
    for piece in plan.get("pieces", []):
        # 计算并保存到 polygon（当前多边形的顶点数组）。
        polygon = np.asarray(piece.get("target_polygon_mm"), np.float64)
        # 判断条件；满足时执行下面代码：polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2。
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            # 抛出异常，通知上层处理：RuntimeError(f'P{piece.get('id', '?')}缺少有效目标轮廓，无法最终复核')。
            raise RuntimeError(f"P{piece.get('id', '?')}缺少有效目标轮廓，无法最终复核")
        # 调用函数：targets.append。
        targets.append({
            # 字典字段'piece_id'（面向显示和计划的碎片编号）：调用 int。
            "piece_id": int(piece["id"]),
            # 字典字段'template_id'（目标模板编号）：调用 str。
            "template_id": str(piece.get("template_id", "")),
            # 字典字段'polygon_mm'（单位为毫米的多边形顶点）：polygon（当前多边形的顶点数组）。
            "polygon_mm": polygon,
            # 字典字段'center_mm'（视觉质心的毫米坐标(x,y)）：使用轮廓图像矩计算面积质心；退化时改用顶点平均值。
            "center_mm": polygon_centroid(polygon),
        })
    # 判断条件；满足时执行下面代码：not targets。
    if not targets:
        # 抛出异常，通知上层处理：RuntimeError('运动计划不包含最终目标轮廓')。
        raise RuntimeError("运动计划不包含最终目标轮廓")
    # 返回结果：targets。
    return targets


# 【函数：match_final_layout】将最终实拍碎片全局匹配到计划目标，并区分位置与角度误差。
# shape_iou比较对齐后的形状是否相似，placed_iou比较实际摆放位置是否重合，两者含义不同；最终还检查中心误差、角度误差和数量。
# 参数 pieces（碎片列表）：Piece列表/序列。
# 参数 plan（从JSON读取的运动计划字典）：字典（键为字符串，值为任意类型）。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：字典（键为字符串，值为任意类型）；箭头->是类型提示，不会替你转换实际返回值。
def match_final_layout(pieces: list[Piece], plan: dict[str, Any],
                       config: dict[str, Any]) -> dict[str, Any]:
    """将最终实拍碎片全局匹配到计划目标，并区分位置与角度误差。"""
    # 计算并保存到 targets。
    targets = final_verification_targets(plan)
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 detected。
    detected = pieces[:len(targets)]
    # 遍历数据，逐项处理：enumerate(detected, 1)。
    for index, piece in enumerate(detected, 1):
        # 计算并保存到 piece.piece_id（面向显示和计划的碎片编号）。
        piece.piece_id = index

    # 计算并保存到 pair_metrics。
    pair_metrics: dict[tuple[int, int], dict[str, Any]] = {}
    # 遍历数据，逐项处理：enumerate(detected)。
    for actual_index, piece in enumerate(detected):
        # 计算并保存到 actual_polygon（实测·多边形）。
        actual_polygon = piece.contour.reshape(-1, 2).astype(np.float64) / ppm
        # 计算并保存到 actual_center（实测·中心）。
        actual_center = np.asarray(piece.center_mm(ppm), np.float64)
        # 遍历数据，逐项处理：enumerate(targets)。
        for target_index, target in enumerate(targets):
            # 计算并保存到 correction、shape_iou（对齐后的形状交并比）、fit_scale（形状匹配时的尺度比；刚体搬运应保持为1）。
            correction, shape_iou, fit_scale = best_rotation_fit(
                piece, target["polygon_mm"], config
            )
            # 计算并保存到 position_error（位置·误差）。
            position_error = float(np.linalg.norm(actual_center - target["center_mm"]))
            # 计算并保存到 placed_iou（实际位置与目标位置的交并比）。
            placed_iou = polygon_iou_mm(actual_polygon, target["polygon_mm"], config)
            # 计算并保存到 scale_penalty（尺度·penalty）。
            scale_penalty = abs(math.log(max(fit_scale, 1e-6)))
            # 计算并保存到 cost（代价）。
            cost = position_error + 35.0 * (1.0 - shape_iou) + 10.0 * scale_penalty
            # 计算并保存到 pair_metrics[actual_index, target_index]。
            pair_metrics[(actual_index, target_index)] = {
                # 字典字段'cost'（代价）：cost（代价）。
                "cost": cost,
                # 字典字段'position_error_mm'（位置·误差·毫米）：position_error（位置·误差）。
                "position_error_mm": position_error,
                # 字典字段'rotation_correction_deg'（旋转·correction·度）：correction。
                "rotation_correction_deg": correction,
                # 字典字段'shape_iou'（对齐后的形状交并比）：shape_iou（对齐后的形状交并比）。
                "shape_iou": shape_iou,
                # 字典字段'placed_iou'（实际位置与目标位置的交并比）：placed_iou（实际位置与目标位置的交并比）。
                "placed_iou": placed_iou,
                # 字典字段'fit_scale'（形状匹配时的尺度比；刚体搬运应保持为1）：fit_scale（形状匹配时的尺度比；刚体搬运应保持为1）。
                "fit_scale": fit_scale,
                # 字典字段'actual_center_mm'（实测·中心·毫米）：actual_center（实测·中心）。
                "actual_center_mm": actual_center,
            }

    # 判断条件；满足时执行下面代码：detected。
    if detected:
        # 计算并保存到 best_assignment（最佳·assignment）。
        best_assignment = min(
            itertools.permutations(range(len(targets)), len(detected)),
            key=lambda assignment: sum(
                pair_metrics[(actual_index, target_index)]["cost"]
                # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
                for actual_index, target_index in enumerate(assignment)
            ),
        )
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 计算并保存到 best_assignment（最佳·assignment）。
        best_assignment = ()

    # 计算并保存到 position_limit（位置·上限）。
    position_limit = float(config.get("final_verify_max_position_error_mm", 8.0))
    # 计算并保存到 angle_limit（角度·上限）。
    angle_limit = float(config.get("final_verify_max_rotation_error_deg", 12.0))
    # 计算并保存到 shape_iou_minimum（形状·交并比·最小）。
    shape_iou_minimum = float(config.get("final_verify_min_shape_iou", 0.65))
    # 计算并保存到 placed_iou_minimum（placed·交并比·最小）。
    placed_iou_minimum = float(config.get("final_verify_min_placed_iou", 0.45))
    # 计算并保存到 matches（匹配记录列表）。
    matches: list[dict[str, Any]] = []
    # 计算并保存到 matched_targets。
    matched_targets: set[int] = set()
    # 遍历数据，逐项处理：enumerate(best_assignment)。
    for actual_index, target_index in enumerate(best_assignment):
        # 计算并保存到 metric（指标）。
        metric = pair_metrics[(actual_index, target_index)]
        # 计算并保存到 target（目标数据或目标位置）。
        target = targets[target_index]
        # 调用函数：matched_targets.add。
        matched_targets.add(target_index)
        # 计算并保存到 passed。
        passed = (
            metric["position_error_mm"] <= position_limit
            # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
            and abs(metric["rotation_correction_deg"]) <= angle_limit
            # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
            and metric["shape_iou"] >= shape_iou_minimum
            # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
            and metric["placed_iou"] >= placed_iou_minimum
        )
        # 调用函数：matches.append。
        matches.append({
            # 字典字段'actual_index'（实测·索引）：计算 actual_index + 1。
            "actual_index": actual_index + 1,
            # 字典字段'target_piece_id'（目标·碎片·编号）：target['piece_id']。
            "target_piece_id": target["piece_id"],
            # 字典字段'template_id'（目标模板编号）：target['template_id']。
            "template_id": target["template_id"],
            # 字典字段'actual_center_mm'（实测·中心·毫米）：按条件生成列表。
            "actual_center_mm": [round(float(value), 2) for value in metric["actual_center_mm"]],
            # 字典字段'target_center_mm'（目标·中心·毫米）：按条件生成列表。
            "target_center_mm": [round(float(value), 2) for value in target["center_mm"]],
            # 字典字段'position_error_mm'（位置·误差·毫米）：调用 round。
            "position_error_mm": round(float(metric["position_error_mm"]), 2),
            # 字典字段'rotation_correction_deg'（旋转·correction·度）：调用 round。
            "rotation_correction_deg": round(float(metric["rotation_correction_deg"]), 2),
            # 字典字段'shape_iou'（对齐后的形状交并比）：调用 round。
            "shape_iou": round(float(metric["shape_iou"]), 4),
            # 字典字段'placed_iou'（实际位置与目标位置的交并比）：调用 round。
            "placed_iou": round(float(metric["placed_iou"]), 4),
            # 字典字段'fit_scale'（形状匹配时的尺度比；刚体搬运应保持为1）：调用 round。
            "fit_scale": round(float(metric["fit_scale"]), 4),
            # 字典字段'passed'（passed）：passed。
            "passed": passed,
        })

    # 计算并保存到 missing_target_ids（missing·目标·编号集合）。
    missing_target_ids = [
        targets[index]["piece_id"] for index in range(len(targets))
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        if index not in matched_targets
    ]
    # 计算并保存到 expected_count（期望·数量）。
    expected_count = len(targets)
    # 计算并保存到 overall_pass。
    overall_pass = (
        len(pieces) == expected_count
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        and not missing_target_ids
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        and all(item["passed"] for item in matches)
    )
    # 返回结果：创建结果字典。
    return {
        # 字典字段'status'（处理结果的状态标记）：'pass' if overall_pass else 'fail'。
        "status": "pass" if overall_pass else "fail",
        # 字典字段'detected_piece_count'（detected·碎片·数量）：调用 len。
        "detected_piece_count": len(pieces),
        # 字典字段'expected_piece_count'（期望·碎片·数量）：expected_count（期望·数量）。
        "expected_piece_count": expected_count,
        # 字典字段'missing_target_piece_ids'（missing·目标·碎片·编号集合）：missing_target_ids（missing·目标·编号集合）。
        "missing_target_piece_ids": missing_target_ids,
        # 字典字段'unexpected_piece_count'（unexpected·碎片·数量）：调用 max。
        "unexpected_piece_count": max(0, len(pieces) - expected_count),
        # 字典字段'thresholds'（阈值集合）：创建结果字典。
        "thresholds": {
            # 字典字段'max_position_error_mm'（最大·位置·误差·毫米）：position_limit（位置·上限）。
            "max_position_error_mm": position_limit,
            # 字典字段'max_rotation_error_deg'（最大·旋转·误差·度）：angle_limit（角度·上限）。
            "max_rotation_error_deg": angle_limit,
            # 字典字段'min_shape_iou'（最小·形状·交并比）：shape_iou_minimum（形状·交并比·最小）。
            "min_shape_iou": shape_iou_minimum,
            # 字典字段'min_placed_iou'（最小·placed·交并比）：placed_iou_minimum（placed·交并比·最小）。
            "min_placed_iou": placed_iou_minimum,
        },
        # 字典字段'matches'（匹配记录列表）：matches（匹配记录列表）。
        "matches": matches,
    }


# 【函数：draw_final_verification】绘制计划目标、实拍轮廓和误差连线。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 pieces（碎片列表）：Piece列表/序列。
# 参数 plan（从JSON读取的运动计划字典）：字典（键为字符串，值为任意类型）。
# 参数 report（最终视觉复核报告）：字典（键为字符串，值为任意类型）。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def draw_final_verification(paper: np.ndarray, pieces: list[Piece],
                            plan: dict[str, Any], report: dict[str, Any],
                            config: dict[str, Any]) -> np.ndarray:
    """绘制计划目标、实拍轮廓和误差连线。"""
    # 计算并保存到 display（用于窗口显示的画布）。
    display = paper.copy()
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 targets。
    targets = {item["piece_id"]: item for item in final_verification_targets(plan)}
    # 遍历数据，逐项处理：targets.values()。
    for target in targets.values():
        # 计算并保存到 points（参与当前计算的一组坐标点）。
        points = np.rint(target["polygon_mm"] * ppm).astype(np.int32)
        # 调用函数：cv2.polylines。
        cv2.polylines(display, [points], True, (0, 220, 0), 2, cv2.LINE_AA)

    # 计算并保存到 matched_actual（matched·实测）。
    matched_actual: set[int] = set()
    # 遍历数据，逐项处理：report['matches']。
    for item in report["matches"]:
        # 计算并保存到 actual_index（实测·索引）。
        actual_index = int(item["actual_index"]) - 1
        # 调用函数：matched_actual.add。
        matched_actual.add(actual_index)
        # 计算并保存到 piece（当前碎片数据）。
        piece = pieces[actual_index]
        # 计算并保存到 color（颜色）。
        color = (0, 200, 0) if item["passed"] else (0, 0, 255)
        # 计算并保存到 contour（碎片轮廓点数组）。
        contour = piece.contour.astype(np.int32)
        # 调用函数：cv2.drawContours。
        cv2.drawContours(display, [contour], -1, color, 3)
        # 计算并保存到 actual_center（实测·中心）。
        actual_center = tuple(np.rint(np.asarray(item["actual_center_mm"]) * ppm).astype(int))
        # 计算并保存到 target_center（目标·中心）。
        target_center = tuple(np.rint(np.asarray(item["target_center_mm"]) * ppm).astype(int))
        # 调用函数：cv2.line。
        cv2.line(display, actual_center, target_center, color, 2, cv2.LINE_AA)
        # 计算并保存到 label。
        label = (f"P{item['target_piece_id']} e={item['position_error_mm']:.1f}mm "
                 f"a={item['rotation_correction_deg']:+.1f}")
        # 调用函数：cv2.putText。
        cv2.putText(display, label, (actual_center[0] + 5, actual_center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

    # 遍历数据，逐项处理：enumerate(pieces)。
    for actual_index, piece in enumerate(pieces):
        # 判断条件；满足时执行下面代码：actual_index not in matched_actual。
        if actual_index not in matched_actual:
            # 调用函数：cv2.drawContours。
            cv2.drawContours(display, [piece.contour.astype(np.int32)], -1, (0, 0, 255), 3)
    # 遍历数据，逐项处理：report['missing_target_piece_ids']。
    for piece_id in report["missing_target_piece_ids"]:
        # 计算并保存到 center（本步骤使用的中心位置）。
        center = tuple(np.rint(targets[piece_id]["center_mm"] * ppm).astype(int))
        # 调用函数：cv2.putText。
        cv2.putText(display, f"P{piece_id} MISSING", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    # 返回结果：display（用于窗口显示的画布）。
    return display


# 【函数：verify_final_frame】保存最终实拍并生成机器可读的逐片复核报告。
# 参数 frame（摄像头的一帧原始图像）：NumPy数组。
# 参数 matrix（本函数使用的矩阵）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 plan_path（运动计划JSON文件路径）：文件系统路径对象。
# 返回类型：字典（键为字符串，值为任意类型）；箭头->是类型提示，不会替你转换实际返回值。
def verify_final_frame(frame: np.ndarray, matrix: np.ndarray,
                       config: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    """保存最终实拍并生成机器可读的逐片复核报告。"""
    # 计算并保存到 plan（从JSON读取的运动计划字典）。
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    # 计算并保存到 verify_config（verify·配置）。
    verify_config = dict(config)
    # 计算并保存到 verify_config['source_region_bottom_mm']。
    verify_config["source_region_bottom_mm"] = float(config["paper_height_mm"])
    # 计算并保存到 verify_config['max_pieces']。
    verify_config["max_pieces"] = max(
        int(config.get("max_pieces", 4)), int(plan.get("piece_count", 4))
    )
    # 计算并保存到 paper（透视校正后的A4纸面图像）。
    paper = warp_paper(frame, matrix, verify_config)
    # 计算并保存到 pieces（碎片列表）、mask（二值掩膜（0为背景，非零为选中区域））。
    pieces, mask = detect_pieces(paper, verify_config, diagnostics=True)
    # 计算并保存到 report（最终视觉复核报告）。
    report = match_final_layout(pieces, plan, verify_config)
    # 计算并保存到 report['generated_at']。
    report["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    # 计算并保存到 overlay（叠加轮廓和标记后的图像）。
    overlay = draw_final_verification(paper, pieces, plan, report, verify_config)

    # 调用函数：OUTPUT_DIR.mkdir。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 调用函数：write_image。
    write_image(OUTPUT_DIR / "final_capture.jpg", frame)
    # 调用函数：write_image。
    write_image(OUTPUT_DIR / "final_corrected.jpg", paper)
    # 调用函数：write_image。
    write_image(OUTPUT_DIR / "final_mask.png", mask)
    # 调用函数：write_image。
    write_image(OUTPUT_DIR / "final_target_overlay.jpg", overlay)
    # 打开并自动管理资源，结束后自动释放。
    with (OUTPUT_DIR / "final_verification.json").open("w", encoding="utf-8") as file:
        # 调用函数：json.dump。
        json.dump(report, file, ensure_ascii=False, indent=2)

    # 计算并保存到 level。
    level = "OK" if report["status"] == "pass" else "WARN"
    # 调用函数：print。
    print(f"[{level}] 最终视觉复核：识别{report['detected_piece_count']}/"
          f"{report['expected_piece_count']}块，status={report['status']}")
    # 遍历数据，逐项处理：report['matches']。
    for item in report["matches"]:
        # 调用函数：print。
        print(f"[复核] P{item['target_piece_id']}: 位置误差={item['position_error_mm']:.2f}mm，"
              f"仍需旋转={item['rotation_correction_deg']:+.2f}deg，"
              f"放置IoU={item['placed_iou']:.3f}，pass={item['passed']}")
    # 判断条件；满足时执行下面代码：report['missing_target_piece_ids']。
    if report["missing_target_piece_ids"]:
        # 调用函数：print。
        print(f"[WARN] 漏吸或漏放目标：{report['missing_target_piece_ids']}")
    # 调用函数：print。
    print(f"[OK] 最终复核图：{OUTPUT_DIR / 'final_target_overlay.jpg'}")
    # 返回结果：report（最终视觉复核报告）。
    return report


# 【函数：capture_post_motion_frame】等待机构回原点和曝光稳定后读取最终帧。
# 参数 capture（已打开的摄像头对象）：cv2.VideoCapture。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def capture_post_motion_frame(capture: cv2.VideoCapture,
                              config: dict[str, Any]) -> np.ndarray:
    """等待机构回原点和曝光稳定后读取最终帧。"""
    # 计算并保存到 deadline（单调时钟上的截止时刻（秒））。
    deadline = time.monotonic() + float(config.get("final_capture_settle_seconds", 0.8))
    # 计算并保存到 frame（摄像头的一帧原始图像）。
    frame: np.ndarray | None = None
    # 只要条件成立就重复执行：time.monotonic() < deadline。
    while time.monotonic() < deadline:
        # 计算并保存到 success（本步骤是否成功）、frame（摄像头的一帧原始图像）。
        success, frame = capture.read()
        # 判断条件；满足时执行下面代码：not success。
        if not success:
            # 抛出异常，通知上层处理：RuntimeError('最终复核时摄像头读取失败')。
            raise RuntimeError("最终复核时摄像头读取失败")
    # 遍历数据，逐项处理：range(int(config.get('final_capture_extra_frames', 5)))。
    for _ in range(int(config.get("final_capture_extra_frames", 5))):
        # 计算并保存到 success（本步骤是否成功）、frame（摄像头的一帧原始图像）。
        success, frame = capture.read()
        # 判断条件；满足时执行下面代码：not success。
        if not success:
            # 抛出异常，通知上层处理：RuntimeError('最终复核时摄像头读取失败')。
            raise RuntimeError("最终复核时摄像头读取失败")
    # 判断条件；满足时执行下面代码：frame is None。
    if frame is None:
        # 抛出异常，通知上层处理：RuntimeError('最终复核未取得摄像头画面')。
        raise RuntimeError("最终复核未取得摄像头画面")
    # 返回结果：frame（摄像头的一帧原始图像）。
    return frame


# 【函数：capture_immediate_frame】无GUI等待START，立即取得曝光稳定帧并记录计时起点。
# 参数 capture（已打开的摄像头对象）：cv2.VideoCapture。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为NumPy数组、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def capture_immediate_frame(
        capture: cv2.VideoCapture,
        config: dict[str, Any]) -> tuple[np.ndarray, float]:
    """无GUI等待START，立即取得曝光稳定帧并记录计时起点。"""
    # 计算并保存到 trigger_time。
    trigger_time = time.monotonic()
    # 调用函数：print。
    print("[自动] 已无交互触发，等待曝光稳定...")
    # 计算并保存到 warmup_seconds（warmup·秒）。
    warmup_seconds = float(config.get("auto_warmup_seconds", 0.8))
    # 计算并保存到 deadline（单调时钟上的截止时刻（秒））。
    deadline = time.monotonic() + warmup_seconds
    # 计算并保存到 frame（摄像头的一帧原始图像）。
    frame: np.ndarray | None = None
    # 只要条件成立就重复执行：time.monotonic() < deadline。
    while time.monotonic() < deadline:
        # 计算并保存到 success（本步骤是否成功）、frame（摄像头的一帧原始图像）。
        success, frame = capture.read()
        # 判断条件；满足时执行下面代码：not success。
        if not success:
            # 抛出异常，通知上层处理：RuntimeError('摄像头读取失败')。
            raise RuntimeError("摄像头读取失败")
    # 遍历数据，逐项处理：range(int(config.get('auto_extra_settle_frames', 5)))。
    for _ in range(int(config.get("auto_extra_settle_frames", 5))):
        # 计算并保存到 success（本步骤是否成功）、frame（摄像头的一帧原始图像）。
        success, frame = capture.read()
        # 判断条件；满足时执行下面代码：not success。
        if not success:
            # 抛出异常，通知上层处理：RuntimeError('摄像头读取失败')。
            raise RuntimeError("摄像头读取失败")
    # 判断条件；满足时执行下面代码：frame is None。
    if frame is None:
        # 抛出异常，通知上层处理：RuntimeError('无交互自动流程未取得摄像头画面')。
        raise RuntimeError("无交互自动流程未取得摄像头画面")
    # 返回结果：创建数据容器。
    return frame, trigger_time


# 【函数：_auto_detect_and_solve】为当前帧选择标定矩阵、校正纸面、生成掩膜并运行处理主流程。
# 参数 frame（摄像头的一帧原始图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 saved_matrix（已保存的透视标定矩阵）：NumPy数组或None（不返回业务结果）。
# 返回类型：元组（依次为NumPy数组、NumPy数组、NumPy数组、Piece列表/序列、PathResult列表/序列）；箭头->是类型提示，不会替你转换实际返回值。
def _auto_detect_and_solve(frame: np.ndarray,
                           config: dict[str, Any],
                           saved_matrix: np.ndarray | None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                      list[Piece], list[PathResult]]:
    # 计算并保存到 active_matrix（本次识别实际采用的透视矩阵）、source（源数据或源位置）。
    active_matrix, source = resolve_calibration_matrix(frame, config, saved_matrix)
    # 计算并保存到 paper（透视校正后的A4纸面图像）。
    paper = warp_paper(frame, active_matrix, config)
    # 计算并保存到 _（此处不需要使用的返回值或循环占位变量）、mask_override（外部提供的掩膜，None表示由检测函数生成）。
    _, mask_override = mask_in_paper(frame, active_matrix, config)
    # 计算并保存到 pieces（碎片列表）、paths（各碎片的搬运路径结果）。
    pieces, paths = process_paper(frame, paper, config, False, mask_override)
    # 返回结果：创建数据容器。
    return active_matrix, paper, mask_override, pieces, paths


# 【函数：process_paper_fallback】构造将上半区碎片排到下半区的应急搬运计划，并检查该计划是否可执行。
# 这里构造的是下半区排放方案，不是几何拼合解。bbox是代码构造的放置框；缩小widths只缩小排布用宽度，实际碎片本身不会缩小。
# 参数 original（原始图像）：NumPy数组。
# 参数 paper（透视校正后的A4纸面图像）：NumPy数组。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 mask_override（外部提供的掩膜，None表示由检测函数生成）：NumPy数组。
# 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
def process_paper_fallback(original: np.ndarray, paper: np.ndarray,
                           config: dict[str, Any],
                           mask_override: np.ndarray) -> bool:
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = float(config["pixels_per_mm"])
    # 计算并保存到 paper_width（纸面宽度（毫米））。
    paper_width = float(config["paper_width_mm"])
    # 计算并保存到 paper_height（纸面高度（毫米））。
    paper_height = float(config["paper_height_mm"])
    # 计算并保存到 spacing（间隔）。
    spacing = float(config.get("fallback_spacing_mm", 15.0))
    # 计算并保存到 half_y（half·Y轴）。
    half_y = paper_height / 2.0

    # 计算并保存到 pieces（碎片列表）、mask（二值掩膜（0为背景，非零为选中区域））。
    pieces, mask = detect_pieces(paper, config, diagnostics=False,
                                 # 传入命名参数：mask_override=mask_override（外部提供的掩膜，None表示由检测函数生成）；取值过程：mask_override（外部提供的掩膜，None表示由检测函数生成）。
                                 mask_override=mask_override)
    # 计算并保存到 upper。
    upper = sorted(
        [p for p in pieces if p.center_mm(ppm)[1] < half_y],
        key=lambda p: p.area_mm2, reverse=True,
    )[:int(config.get("max_pieces", 4))]
    # 判断条件；满足时执行下面代码：not upper。
    if not upper:
        # 调用函数：print。
        print("[保底] 上半区无有效碎片")
        # 返回结果：False。
        return False
    # 计算并保存到 upper。
    upper = number_by_move_order(upper, config)
    # 计算并保存到 widths。
    widths = [float(cv2.boundingRect(p.contour)[2]) / ppm for p in upper]
    # 计算并保存到 total_w（总量·w）。
    total_w = sum(widths) + (len(upper) - 1) * spacing
    # 判断条件；满足时执行下面代码：total_w > paper_width * 0.95。
    if total_w > paper_width * 0.95:
        # 计算并保存到 scale（当前缩放比例或试探步长比例）。
        scale = paper_width * 0.95 / total_w
        # 更新变量：spacing（间隔）。
        spacing *= scale
        # 计算并保存到 widths。
        widths = [w * scale for w in widths]
        # 计算并保存到 total_w（总量·w）。
        total_w = sum(widths) + (len(upper) - 1) * spacing

    # 计算并保存到 start_x（start·X轴）。
    start_x = (paper_width - total_w) / 2.0
    # 计算并保存到 target_y（目标·Y轴）。
    target_y = paper_height * 0.75
    # 计算并保存到 matches（匹配记录列表）。
    matches: list[PuzzleMatch] = []
    # 计算并保存到 cur_x（cur·X轴）。
    cur_x = start_x
    # 遍历数据，逐项处理：enumerate(upper)。
    for idx, piece in enumerate(upper):
        # 计算并保存到 pw。
        pw = widths[idx]
        # 计算并保存到 src。
        src = np.asarray(piece.pick_mm(ppm), np.float64)
        # 计算并保存到 tx（X平移）。
        tx = cur_x + pw / 2.0
        # 计算并保存到 dx。
        dx = tx - src[0]
        # 计算并保存到 dy。
        dy = target_y - src[1]
        # 计算并保存到 tx3。
        tx3 = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        # 计算并保存到 bbox。
        bbox = np.array([
            [cur_x, target_y - piece.height_mm / 2],
            [cur_x + pw, target_y - piece.height_mm / 2],
            [cur_x + pw, target_y + piece.height_mm / 2],
            [cur_x, target_y + piece.height_mm / 2],
        ], np.float64)
        # 调用函数：matches.append。
        matches.append(PuzzleMatch(
            # 传入命名参数：piece=piece（当前碎片数据）；取值过程：piece（当前碎片数据）；template_id=f'F{idx + 1}'（目标模板编号）；取值过程：f'F{idx + 1}'。
            piece=piece, template_id=f"F{idx+1}",
            # 传入命名参数：target_polygon_mm=bbox（目标布局中的多边形顶点（毫米））；取值过程：bbox。
            target_polygon_mm=bbox,
            # 传入命名参数：target_pick_mm=(tx, target_y)（放置后磁吸点的目标坐标（毫米））；取值过程：创建数据容器。
            target_pick_mm=(tx, target_y),
            # 传入命名参数：rotation_deg=0.0（碎片需要旋转的有符号角度（度））；取值过程：0.0；rotation_direction='NONE'（旋转方向标记）；取值过程：'NONE'。
            rotation_deg=0.0, rotation_direction="NONE",
            # 传入命名参数：area_error_ratio=0.0（面积相对误差）；取值过程：0.0；shape_distance=0.0（形状差异指标）；取值过程：0.0。
            area_error_ratio=0.0, shape_distance=0.0,
            # 传入命名参数：match_iou=1.0（形状匹配的交并比）；取值过程：1.0；fit_scale=1.0（形状匹配时的尺度比；刚体搬运应保持为1）；取值过程：1.0；
            # status='ok'（处理结果的状态标记）；取值过程：'ok'。
            match_iou=1.0, fit_scale=1.0, status="ok",
            # 传入命名参数：solver_mode='fallback'（求解方式标记）；取值过程：'fallback'；transform_3x3=tx3（3×3齐次变换矩阵（把源坐标映射到目标坐标））；取值过程：tx3。
            solver_mode="fallback", transform_3x3=tx3,
        ))
        # 更新变量：cur_x（cur·X轴）。
        cur_x += pw + spacing

    # 计算并保存到 paths（各碎片的搬运路径结果）。
    paths = []
    # 遍历数据，逐项处理：matches。
    for match in matches:
        # 计算并保存到 target_px（目标·像素）。
        target_px = mm_to_px(match.target_pick_mm, ppm)
        # 计算并保存到 path（当前文件路径或几何路径）、planner（路径规划方法标记）、status（处理结果的状态标记）。
        path, planner, status = plan_single_path(
            match.piece.pick_px, target_px, None, config)
        # 调用函数：paths.append。
        paths.append(PathResult(match, path, planner, status))

    # 计算并保存到 preview（路径预览图）。
    preview = draw_preview(paper, [m.piece for m in matches], paths, config)
    # 调用函数：save_outputs。
    save_outputs(original, paper, mask, preview, paths, config)
    # 计算并保存到 reasons（检查未通过的原因列表）。
    reasons = motion_safety_reasons(paths, config)
    # 调用函数：print。
    print(f"[保底] 上半区{len(paths)}块→下半区，间距{spacing:.1f}mm，"
          f"ready_for_motion={not reasons}")
    # 判断条件；满足时执行下面代码：reasons。
    if reasons:
        # 遍历数据，逐项处理：reasons。
        for r in reasons:
            # 调用函数：print。
            print(f"[保底] {r}")
    # 返回结果：not reasons。
    return not reasons


# 【函数：run_auto】一键流程：识别求解、串口执行、回原点并自动拍照复核。
# 顺序是取帧→重试识别求解→必要时兜底→按dry_run选择打印或真实发送→真实执行后拍照复核。相机最终由finally释放。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 camera_id（摄像头设备编号）：整数或None（不返回业务结果）。
# 参数 serial_port（串口·port）：字符串或None（不返回业务结果）。
# 参数 force_dry_run（命令行是否要求只打印命令）：布尔值True/False。
# 参数 start_immediately：布尔值True/False；省略时使用False。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def run_auto(config: dict[str, Any], camera_id: int | None,
             serial_port: str | None, force_dry_run: bool,
             start_immediately: bool = False) -> None:
    """一键流程：识别求解、串口执行、回原点并自动拍照复核。"""
    # 计算并保存到 matrix（本函数使用的矩阵）。
    matrix = load_calibration(config)
    # 计算并保存到 capture（已打开的摄像头对象）。
    capture = open_camera(config, camera_id)
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 只要条件成立就重复执行：True。
        while True:
            # 计算并保存到 captured。
            captured = (
                capture_immediate_frame(capture, config)
                # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
                if start_immediately
                else wait_for_start_and_capture(capture, config)
            )
            # 判断条件；满足时执行下面代码：captured is None。
            if captured is None:
                # 调用函数：print。
                print("[自动] 已退出。")
                # 返回结果：无。
                return
            # 计算并保存到 frame（摄像头的一帧原始图像）、trigger_time。
            frame, trigger_time = captured

            # 计算并保存到 max_retries（最大·retries）。
            max_retries = int(config.get("auto_max_retries", 3))
            # 计算并保存到 fallback_enabled（兜底·启用）。
            fallback_enabled = bool(config.get("auto_fallback_enabled", True))
            # 计算并保存到 success（本步骤是否成功）。
            success = False

            # 遍历数据，逐项处理：range(1, max_retries + 1)。
            for attempt in range(1, max_retries + 1):
                # 执行可能出错的代码，并由后面的异常分支处理错误。
                try:
                    # 计算并保存到 active_matrix（本次识别实际采用的透视矩阵）、paper（透视校正后的A4纸面图像）、mask_override（外部提供的掩膜，None表示由检测函数生成）、_（此处不需要使用的返回值或循环占位变量）、_（此处不需要使用的返回值或循环占位变量）。
                    active_matrix, paper, mask_override, _, _ = (
                        _auto_detect_and_solve(frame, config, matrix))
                    # 判断条件；满足时执行下面代码：active_matrix is not None。
                    if active_matrix is not None:
                        # 计算并保存到 source（源数据或源位置）。
                        source = "auto_board"
                        # 计算并保存到 matrix（本函数使用的矩阵）。
                        matrix = active_matrix
                    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
                    else:
                        # 计算并保存到 source（源数据或源位置）。
                        source = "saved"
                    # 计算并保存到 success（本步骤是否成功）。
                    success = True
                    # 立即结束当前循环。
                    break
                # 捕获RuntimeError异常，转入下面的处理代码；异常对象保存在exc。
                except RuntimeError as exc:
                    # 判断条件；满足时执行下面代码：attempt < max_retries。
                    if attempt < max_retries:
                        # 调用函数：print。
                        print(f"[自动] 第{attempt}次失败：{exc}，"
                              f"剩余{max_retries - attempt}次重试...")
                        # 调用函数：time.sleep。
                        time.sleep(0.3)
                        # 计算并保存到 frame（摄像头的一帧原始图像）、_（此处不需要使用的返回值或循环占位变量）。
                        frame, _ = capture_immediate_frame(capture, config)
                    # 判断条件；满足时执行下面代码：fallback_enabled。
                    elif fallback_enabled:
                        # 调用函数：print。
                        print(f"[自动] {max_retries}次均失败：{exc}")
                        # 调用函数：print。
                        print("[自动] 启用保底方案：上半区碎片移至下半区")
                        # 执行可能出错的代码，并由后面的异常分支处理错误。
                        try:
                            # 计算并保存到 active_matrix（本次识别实际采用的透视矩阵）、source（源数据或源位置）。
                            active_matrix, source = resolve_calibration_matrix(
                                frame, config, matrix)
                            # 判断条件；满足时执行下面代码：source == 'auto_board'。
                            if source == "auto_board":
                                # 计算并保存到 matrix（本函数使用的矩阵）。
                                matrix = active_matrix
                            # 计算并保存到 paper（透视校正后的A4纸面图像）。
                            paper = warp_paper(frame, active_matrix, config)
                            # 计算并保存到 _（此处不需要使用的返回值或循环占位变量）、mask_override（外部提供的掩膜，None表示由检测函数生成）。
                            _, mask_override = mask_in_paper(
                                frame, active_matrix, config)
                            # 计算并保存到 success（本步骤是否成功）。
                            success = process_paper_fallback(
                                frame, paper, config, mask_override)
                        # 捕获RuntimeError异常，转入下面的处理代码；异常对象保存在fallback_exc。
                        except RuntimeError as fallback_exc:
                            # 调用函数：print。
                            print(f"[保底] 失败：{fallback_exc}")
                    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
                    else:
                        # 调用函数：print。
                        print(f"[自动] {max_retries}次均失败：{exc}")

            # 判断条件；满足时执行下面代码：not success。
            if not success:
                # 调用函数：print。
                print("[自动] 本次未生成可执行计划，可重新摆放后再次START。\n")
                # 判断条件；满足时执行下面代码：start_immediately。
                if start_immediately:
                    # 返回结果：无。
                    return
                # 跳过本轮，进入下一轮循环。
                continue

            # 计算并保存到 port（串口设备名称）。
            port = serial_port if serial_port is not None else str(config.get("serial_port", ""))
            # 计算并保存到 dry_run（是否只打印命令而不发送）。
            dry_run = force_dry_run or not port
            # 计算并保存到 motion_completed（运动·completed）。
            motion_completed = False
            # 执行可能出错的代码，并由后面的异常分支处理错误。
            try:
                # 调用函数：serial_transport.send_plan_file。
                serial_transport.send_plan_file(
                    OUTPUT_DIR / "plan.json",
                    # 传入命名参数：dry_run=dry_run（是否只打印命令而不发送）；取值过程：dry_run（是否只打印命令而不发送）。
                    dry_run=dry_run,
                    # 传入命名参数：port=port（串口设备名称）；取值过程：port（串口设备名称）。
                    port=port,
                    # 传入命名参数：baudrate=int(config.get('serial_baudrate', 115200))（串口波特率）；取值过程：调用 int。
                    baudrate=int(config.get("serial_baudrate", 115200)),
                    # 传入命名参数：ack_timeout_seconds=float(config.get('serial_ack_timeout_seconds', 2.0))（应答·超时·秒）；取值过程：调用 float。
                    ack_timeout_seconds=float(config.get("serial_ack_timeout_seconds", 2.0)),
                    # 传入命名参数：command_delay_seconds=float(config.get('serial_command_delay_seconds', 0.0))（指令·间隔·秒）；
                    # 取值过程：调用 float。
                    command_delay_seconds=float(config.get("serial_command_delay_seconds", 0.0)),
                )
                # 计算并保存到 motion_completed（运动·completed）。
                motion_completed = not dry_run
            # 捕获serial_transport.SerialPlanError异常，转入下面的处理代码；异常对象保存在exc。
            except serial_transport.SerialPlanError as exc:
                # 调用函数：print。
                print(f"[WARN] 未发送运动指令：{exc}")
            # 判断条件；满足时执行下面代码：dry_run and (not force_dry_run)。
            if dry_run and not force_dry_run:
                # 调用函数：print。
                print("[提示] 未配置串口（--port或config.json的serial_port），"
                      "以上仅为dry-run打印，未连接主控。")

            # 计算并保存到 motion_elapsed（运动·elapsed）。
            motion_elapsed = time.monotonic() - trigger_time
            # 计算并保存到 target_seconds（目标·秒）。
            target_seconds = float(config.get("auto_target_seconds", 80.0))
            # 计算并保存到 worst_seconds（最差·秒）。
            worst_seconds = float(config.get("auto_worst_case_seconds", 110.0))
            # 计算并保存到 level。
            level = "OK" if motion_elapsed <= target_seconds else (
                "WARN" if motion_elapsed <= worst_seconds else "FAIL")
            # 调用函数：print。
            print(f"[{level}] 本次自START至发送完成用时{motion_elapsed:.1f}s"
                  f"（目标≤{target_seconds:.0f}s，最坏<{worst_seconds:.0f}s）")

            # 判断条件；满足时执行下面代码：motion_completed and bool(config.get('final_verification_enabled', True))。
            if motion_completed and bool(config.get("final_verification_enabled", True)):
                # 执行可能出错的代码，并由后面的异常分支处理错误。
                try:
                    # 计算并保存到 final_frame（最终·图像帧）。
                    final_frame = capture_post_motion_frame(capture, config)
                    # 调用函数：verify_final_frame。
                    verify_final_frame(
                        final_frame, active_matrix, config, OUTPUT_DIR / "plan.json"
                    )
                # 捕获(RuntimeError, OSError, ValueError, cv2.error, json.JSONDecodeError)异常，转入下面的处理代码；异常对象保存在exc。
                except (RuntimeError, OSError, ValueError, cv2.error, json.JSONDecodeError) as exc:
                    # 调用函数：print。
                    print(f"[WARN] 运动已完成，但最终视觉复核失败：{exc}")

            # 调用函数：print。
            print("[自动] 本次流程结束，可再次按START处理下一组碎片。\n")
            # 判断条件；满足时执行下面代码：start_immediately。
            if start_immediately:
                # 返回结果：无。
                return
    # 无论正常结束、return还是抛出异常都会进入这里，用于释放相机/串口等资源。
    finally:
        # 调用函数：capture.release。
        capture.release()
        # 调用函数：cv2.destroyAllWindows。
        cv2.destroyAllWindows()
        # 调用函数：print。
        print("[OK] 摄像头已释放")

# =============================================================================
# 【分区】题目套装、命令行与 main()
# 功能：self=固定图2 100×60；site=通用几何。解析 --camera/--image/--demo/--auto 等后进入对应流程。
# 可修改：命令行默认相机号；不要把现场阈值写进 argparse default，应写 config.json。
# 看情况改：apply_puzzle_set 只改内存副本，不写回文件。
# 不要改：互斥的运行模式；main 里先 load_config 再 apply_puzzle_set。
# =============================================================================
# 【函数：apply_puzzle_set】根据self或site选择固定图2或通用几何模式，返回相应配置。
# 原有文档字符串有乱码：self选择固定图2并设置100×60毫米；site选择通用几何模式。这里修改配置副本，不写回config.json。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 puzzle_set：字符串或None（不返回业务结果）。
# 返回类型：字典（键为字符串，值为任意类型）；箭头->是类型提示，不会替你转换实际返回值。
def apply_puzzle_set(config: dict[str, Any], puzzle_set: str | None) -> dict[str, Any]:
    """??????????????????"""
    # 判断条件；满足时执行下面代码：puzzle_set is None。
    if puzzle_set is None:
        # 返回结果：config（程序配置字典）。
        return config
    # 计算并保存到 updated。
    updated = dict(config)
    # 判断条件；满足时执行下面代码：puzzle_set == 'self'。
    if puzzle_set == "self":
        # 计算并保存到 updated['puzzle_mode']。
        updated["puzzle_mode"] = "fixed_figure_2"
        # 计算并保存到 updated['target_rectangle_size_mm']。
        updated["target_rectangle_size_mm"] = [100.0, 60.0]
    # 判断条件；满足时执行下面代码：puzzle_set == 'site'。
    elif puzzle_set == "site":
        # 计算并保存到 updated['puzzle_mode']。
        updated["puzzle_mode"] = "generic_geometry"
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 抛出异常，通知上层处理：ValueError(f'???????{puzzle_set}')。
        raise ValueError(f"???????{puzzle_set}")
    # 返回结果：updated。
    return updated

# 【函数：parse_args】声明主程序命令行选项、解析输入并检查不允许的参数组合。
# 返回类型：argparse.Namespace；箭头->是类型提示，不会替你转换实际返回值。
def parse_args() -> argparse.Namespace:
    # 计算并保存到 parser（命令行参数解析器）。
    parser = argparse.ArgumentParser(description="USB拼图视觉与A4最短正交路径生成")
    # 计算并保存到 mode（模式）。
    mode = parser.add_mutually_exclusive_group()
    # 调用函数：mode.add_argument。
    mode.add_argument("--camera", type=int, nargs="?", const=0, help="摄像头编号，默认0")
    # 调用函数：mode.add_argument。
    mode.add_argument("--image", type=Path, help="处理单张图片")
    # 调用函数：mode.add_argument。
    mode.add_argument("--demo", action="store_true", help="运行合成演示")
    # 调用函数：mode.add_argument。
    mode.add_argument(
        # 传入命名参数：action='store_true'（开关的解析行为）；取值过程：'store_true'。
        "--auto", action="store_true",
        # 传入命名参数：help='一键流程：自动定位底板，等待START后识别求解并按需发送串口'（命令行帮助说明）；取值过程：'一键流程：自动定位底板，等待START后识别求解并按需发送串口'。
        help="一键流程：自动定位底板，等待START后识别求解并按需发送串口",
    )
    # 调用函数：mode.add_argument。
    mode.add_argument(
        # 传入命名参数：type=Path（命令行输入的转换类型或函数）；取值过程：Path；nargs='?'（参数取值的数量规则）；取值过程：'?'；
        # const=OUTPUT_DIR / 'plan.json'（选项出现但省略值时使用的值）；取值过程：计算 OUTPUT_DIR / 'plan.json'。
        "--send-plan", type=Path, nargs="?", const=OUTPUT_DIR / "plan.json",
        # 传入命名参数：help='发送运动计划；省略路径时使用output/plan.json'（命令行帮助说明）；取值过程：'发送运动计划；省略路径时使用output/plan.json'。
        help="发送运动计划；省略路径时使用output/plan.json",
    )
    # 调用函数：parser.add_argument。
    parser.add_argument(
        # 传入命名参数：choices=('self', 'site')（允许的取值）；取值过程：创建数据容器。
        "--puzzle-set", choices=("self", "site"),
        # 传入命名参数：help='?????self=?????2??100x60mm?site=??????'（命令行帮助说明）；
        # 取值过程：'?????self=?????2??100x60mm?site=??????'。
        help="?????self=?????2??100x60mm?site=??????",
    )
    # 调用函数：parser.add_argument。
    parser.add_argument("--camera-id", type=int, default=0,
                        # 传入命名参数：help='--auto使用的摄像头编号，默认0（--camera已被用作模式选择，两者互斥，故--auto单独用这个参数指定摄像头）'（命令行帮助说明）；
                        # 取值过程：'--auto使用的摄像头编号，默认0（--camera已被用作模式选择，两者互斥，故--auto单独用这个参数指定摄像头）'。
                        help="--auto使用的摄像头编号，默认0（--camera已被用作模式选择，"
                             "两者互斥，故--auto单独用这个参数指定摄像头）")
    # 调用函数：parser.add_argument。
    parser.add_argument("--start-immediately", action="store_true",
                        # 传入命名参数：help='--auto无需点击START，立即执行一轮后退出'（命令行帮助说明）；取值过程：'--auto无需点击START，立即执行一轮后退出'。
                        help="--auto无需点击START，立即执行一轮后退出")
    # 调用函数：parser.add_argument。
    parser.add_argument("--dry-run", action="store_true",
                        # 传入命名参数：help='只打印串口指令，不打开串口（--send-plan或--auto均可用）'（命令行帮助说明）；取值过程：'只打印串口指令，不打开串口（--send-plan或--auto均可用）'。
                        help="只打印串口指令，不打开串口（--send-plan或--auto均可用）")
    # 调用函数：parser.add_argument。
    parser.add_argument("--port", help="串口名称，例如COM5（--send-plan或--auto均可用）")
    # 调用函数：parser.add_argument。
    parser.add_argument("--baud", type=int, help="串口波特率，默认读取config.json")
    # 调用函数：parser.add_argument。
    parser.add_argument("--ack-timeout", type=float, help="等待单条ACK的秒数")
    # 计算并保存到 args（解析后的命令行选项）。
    args = parser.parse_args()
    # 判断条件；满足时执行下面代码：args.start_immediately and (not args.auto)。
    if args.start_immediately and not args.auto:
        # 调用函数：parser.error。
        parser.error("--start-immediately必须与--auto一起使用")
    # 计算并保存到 serial_options_used（串口·options·已使用）。
    serial_options_used = (
        args.dry_run or args.port is not None
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        or args.baud is not None or args.ack_timeout is not None
    )
    # 判断条件；满足时执行下面代码：serial_options_used and args.send_plan is None and (not args.auto)。
    if serial_options_used and args.send_plan is None and not args.auto:
        # 调用函数：parser.error。
        parser.error("--dry-run/--port/--baud/--ack-timeout必须与--send-plan或--auto一起使用")
    # 返回结果：args（解析后的命令行选项）。
    return args


# 【函数：main】加载参数和配置，分派到发计划、演示、图片、自动或相机模式，并返回退出码。
# 返回类型：整数；箭头->是类型提示，不会替你转换实际返回值。
def main() -> int:
    # 计算并保存到 args（解析后的命令行选项）。
    args = parse_args()
    # 计算并保存到 config（程序配置字典）。
    config = apply_puzzle_set(load_config(), args.puzzle_set)
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 判断条件；满足时执行下面代码：args.send_plan is not None。
        if args.send_plan is not None:
            # 调用函数：serial_transport.send_plan_file。
            serial_transport.send_plan_file(
                args.send_plan,
                # 传入命名参数：dry_run=bool(args.dry_run)（是否只打印命令而不发送）；取值过程：调用 bool。
                dry_run=bool(args.dry_run),
                # 传入命名参数：port=args.port or str(config.get('serial_port', ''))（串口设备名称）；
                # 取值过程：组合多个条件：args.port or str(config.get('serial_port', ''))。
                port=args.port or str(config.get("serial_port", "")),
                # 传入命名参数：baudrate=args.baud or int(config.get('serial_baudrate', 115200))（串口波特率）；
                # 取值过程：组合多个条件：args.baud or int(config.get('serial_baudrate', 115200))。
                baudrate=args.baud or int(config.get("serial_baudrate", 115200)),
                # 传入命名参数：ack_timeout_seconds=args.ack_timeout if args.ack_timeout is not None else float(conf…（应答·超时·秒）；
                # 取值过程：args.ack_timeout if args.ack_timeout is not None else float(config.get('serial_ack_t…。
                ack_timeout_seconds=(
                    args.ack_timeout
                    # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
                    if args.ack_timeout is not None
                    else float(config.get("serial_ack_timeout_seconds", 2.0))
                ),
                # 传入命名参数：command_delay_seconds=float(config.get('serial_command_delay_seconds', 0.0))（指令·间隔·秒）；
                # 取值过程：调用 float。
                command_delay_seconds=float(config.get("serial_command_delay_seconds", 0.0)),
            )
        # 判断条件；满足时执行下面代码：args.demo。
        elif args.demo:
            # 调用函数：run_demo。
            run_demo(config)
        # 判断条件；满足时执行下面代码：args.image is not None。
        elif args.image is not None:
            # 调用函数：run_image。
            run_image(args.image, config)
        # 判断条件；满足时执行下面代码：args.auto。
        elif args.auto:
            # 调用函数：run_auto。
            run_auto(config, args.camera_id, args.port, bool(args.dry_run), bool(args.start_immediately))
        # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
        else:
            # 调用函数：run_camera。
            run_camera(config, args.camera)
        # 返回结果：0。
        return 0
    # 捕获(RuntimeError, ValueError, OSError, cv2.error)异常，转入下面的处理代码；异常对象保存在exc。
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        # 调用函数：print。
        print(f"[ERR] {exc}")
        # 返回结果：1。
        return 1


# 判断条件；满足时执行下面代码：__name__ == '__main__'。
if __name__ == "__main__":
    # 抛出异常，通知上层处理：SystemExit(main())。
    raise SystemExit(main())
