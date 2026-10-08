"""1～4块多边形拼成矩形的纯几何求解器。

本模块只处理毫米坐标下的二维多边形，不访问摄像头、不写文件，
也不依赖 ``main.py``，便于独立测试并避免循环依赖。
"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：纯几何求解器：1～4 块毫米多边形只旋转平移拼成矩形，不读相机、不写文件、不依赖 main。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约74行  EdgeMatch：一条反向拼缝
#   - 约94行  GenericPiecePose：一片的 3×3 刚体姿态
#   - 约108行  GenericSolution：整局解 + 填充/重叠等质量指标
#   - 约147行  _as_polygon：多边形规范化
#   - 约412行  rigid_align_reversed：把一条边反向对齐到另一条边（切缝）
#   - 约590行  _make_search_budget：DFS 超时和最大状态数
#   - 约627行  generate_edge_candidates：边长接近则生成匹配候选
#   - 约799行  _search_assemblies：固定第0片，深度优先搜索装配树
#   - 约1371行  _assess_assembly：评是不是合格矩形
#   - 约1603行  _optimize_pose_graph：补闭合缝后微调各片刚体
#   - 约1983行  _build_solution：把装配放到 A4 下半区
#   - 约2058行  _build_fallback_placement：平铺兜底（非严格矩形）
#   - 约2265行  split_polygon_edge：长边插顶点，让长边能对多条短边
#   - 约2630行  _solve_normalized_rectangle：严搜→补闭环→可选宽松
#   - 约2908行  solve_generic_rectangle：对外入口：原始边→双拆→单拆→可选平铺
# =============================================================================
# 【中文阅读导引】以下新增#注释解释代码用途；原有字符串、计算、默认值和执行顺序保持不变。
# 这是纯几何求解器：输入每片顶点的毫米坐标，输出目标多边形、3×3刚体矩阵和质量指标。
# 阅读顺序：solve_generic_rectangle → _solve_normalized_rectangle → _search_assemblies → _assess_assembly → _build_solution。
# 基本思路：边长接近可能是同一拼缝；把两边反向对齐后接入新碎片，再检查整体能否形成矩形。
# 所有碎片保持尺寸不变，只旋转和平移；3×3齐次矩阵把(x,y,1)映射到新位置。点按行存储时使用 points @ matrix.T。
# DFS是深度优先搜索：尝试一条拼接分支，递归接下一片，失败就退回上一层尝试其他边。预算限制递归搜索的时间和状态数。
# 几何score是加权误差代价，越小越好；它与扑克牌花纹模块“越大越好”的相似度不是同一个指标。
# 共线拆边只在线段上插入顶点，帮助一条长边对应几条短边，不改变碎片外形。
# 宽松解和平铺兜底属于不同层次的退让；现有程序并非在所有失败条件下都能返回解。
# 语法约定：缩进决定代码归属；=赋值，==比较；列表和数组下标从0开始；None表示没有值；冒号后的类型主要用于阅读和检查。
# 跨行表达式属于同一条语句，括号结束前不会另起一条指令；字典条目和命名实参也在相邻注释中解释。
# =============================================================================
# 【分区】导入
# 功能：纯几何求解，不读相机、不写文件、不 import main，避免循环依赖。
# 可修改：无现场参数。超时/容差走调用方传入的 config（generic_* 键）。
# 看情况改：无。
# 不要改：本模块只处理毫米二维多边形；不要在这里加 cv2.imread 或串口。
# =============================================================================
# 从 __future__ 导入需要的类型或工具。
from __future__ import annotations

# 导入：itertools。
import itertools
# 导入：math。
import math
# 导入：time。
import time
# 从 dataclasses 导入需要的类型或工具。
from dataclasses import dataclass
# 从 typing 导入需要的类型或工具。
from typing import Any, Iterable

# 导入：cv2。
import cv2
# 导入：numpy。
import numpy as np


# =============================================================================
# 【分区】对外数据结构
# 功能：EdgeMatch=一条反向拼缝；GenericPiecePose=一片的 3×3 刚体；GenericSolution=整局解+质量指标。
# 可修改：无。
# 看情况改：score 越小越好（几何代价）。不要和扑克 NCC“越大越好”搞反。
# 不要改：transform_3x3 是 3×3 齐次矩阵；点按行存储时用 points @ matrix.T。
# =============================================================================
# 使用dataclass装饰器，按下方字段自动生成__init__、__repr__和比较方法；frozen=True限制字段重新赋值，但不冻结字段内部的列表/数组。
@dataclass(frozen=True)
# 【类：EdgeMatch】两块碎片之间采用的一组反向拼缝边。
class EdgeMatch:
    """两块碎片之间采用的一组反向拼缝边。"""

    # 声明字段：piece_a（第一块碎片的编号或索引）。
    piece_a: int
    # 声明字段：edge_a（第一块碎片的边索引）。
    edge_a: int
    # 声明字段：piece_b（第二块碎片的编号或索引）。
    piece_b: int
    # 声明字段：edge_b（第二块碎片的边索引）。
    edge_b: int
    # 声明字段：length_error_mm（匹配边的长度差（毫米））。
    length_error_mm: float
    # 声明字段：endpoint_error_mm（拼缝对应端点的误差（毫米））。
    endpoint_error_mm: float


# 使用dataclass装饰器，按下方字段自动生成__init__、__repr__和比较方法。
@dataclass
# 【类：GenericPiecePose】单块源多边形到最终目标位置的刚体姿态。
class GenericPiecePose:
    """单块源多边形到最终目标位置的刚体姿态。"""

    # 声明字段：piece_index（碎片在列表中的索引（从0开始））。
    piece_index: int
    # 声明字段：transform_3x3（3×3齐次变换矩阵（把源坐标映射到目标坐标））。
    transform_3x3: np.ndarray
    # 声明字段：target_polygon_mm（目标布局中的多边形顶点（毫米））。
    target_polygon_mm: np.ndarray


# 使用dataclass装饰器，按下方字段自动生成__init__、__repr__和比较方法。
@dataclass
# 【类：GenericSolution】通用矩形拼接的最终解及质量指标。
class GenericSolution:
    """通用矩形拼接的最终解及质量指标。"""

    # 声明字段：poses（各碎片的刚体姿态集合）。
    poses: list[GenericPiecePose]
    # 声明字段：target_size_mm（目标宽高（毫米））。
    target_size_mm: tuple[float, float]
    # 声明字段：target_top_left_mm（目标区域左上角坐标（毫米））。
    target_top_left_mm: tuple[float, float]
    # 声明字段：score（评分）。
    score: float
    # 声明字段：fill_error_ratio（矩形填充误差比例）。
    fill_error_ratio: float
    # 声明字段：overlap_ratio（碎片间重叠比例）。
    overlap_ratio: float
    # 声明字段：endpoint_error_mm（拼缝对应端点的误差（毫米））。
    endpoint_error_mm: float
    # 声明字段：boundary_error_mm（目标矩形边界误差（毫米））。
    boundary_error_mm: float
    # 声明字段：boundary_gap_ratio（矩形边界缺口比例）。
    boundary_gap_ratio: float
    # 声明字段：matches（匹配记录列表）。
    matches: list[EdgeMatch]
    # 声明字段：search_states（已检查的搜索状态数）。
    search_states: int
    # 计算并保存到 relaxed_override（是否采用放宽质量要求的兜底结果）。
    relaxed_override: bool = False


# =============================================================================
# 【分区】多边形与刚体几何工具
# 功能：规范化顶点、边长、面积、刚体矩阵、反向对齐一条边、变换复合。
# 可修改：闭合点去重阈值 1e-9。
# 看情况改：rigid_align_reversed 是“边反向重合”，拼图切缝两侧方向相反，不要改成同向对齐。
# 不要改：只旋转+平移，禁止缩放；signed_area 符号约定（逆时针为正）被后续归一化使用。
# =============================================================================
# 【函数：_as_polygon】转换为不带重复闭合点的 ``N×2 float64`` 多边形。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组或可逐项遍历的可逐项遍历的浮点数列表/序列列表/序列。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def _as_polygon(points: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    """转换为不带重复闭合点的 ``N×2 float64`` 多边形。"""
    # 计算并保存到 polygon（当前多边形的顶点数组）。
    polygon = np.asarray(points, dtype=np.float64)
    # 判断条件；满足时执行下面代码：polygon.ndim == 3 and polygon.shape[1:] == (1, 2)。
    if polygon.ndim == 3 and polygon.shape[1:] == (1, 2):
        # 计算并保存到 polygon（当前多边形的顶点数组）。
        polygon = polygon[:, 0, :]
    # 判断条件；满足时执行下面代码：polygon.ndim != 2 or polygon.shape[1] != 2。
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        # 抛出异常，通知上层处理：ValueError('多边形必须是N×2坐标数组')。
        raise ValueError("多边形必须是N×2坐标数组")
    # 判断条件；满足时执行下面代码：len(polygon) >= 2 and np.linalg.norm(polygon[0] - polygon[-1]) <= 1e-09。
    if len(polygon) >= 2 and np.linalg.norm(polygon[0] - polygon[-1]) <= 1e-9:
        # 计算并保存到 polygon（当前多边形的顶点数组）。
        polygon = polygon[:-1]
    # 判断条件；满足时执行下面代码：not np.isfinite(polygon).all()。
    if not np.isfinite(polygon).all():
        # 抛出异常，通知上层处理：ValueError('多边形坐标包含NaN或无穷值')。
        raise ValueError("多边形坐标包含NaN或无穷值")
    # 返回结果：调用 polygon.copy。
    return polygon.copy()


# 【函数：signed_area】返回带符号鞋带面积；在图像Y向下坐标中正负只用于统一顺序。
# 鞋带公式：把每个顶点与下一个顶点组成叉乘并求和再除2；符号随顶点绕行顺序改变，绝对值是面积。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组或可逐项遍历的可逐项遍历的浮点数列表/序列列表/序列。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def signed_area(points: np.ndarray | Iterable[Iterable[float]]) -> float:
    """返回带符号鞋带面积；在图像Y向下坐标中正负只用于统一顺序。"""
    # 计算并保存到 polygon（当前多边形的顶点数组）。
    polygon = _as_polygon(points)
    # 判断条件；满足时执行下面代码：len(polygon) < 3。
    if len(polygon) < 3:
        # 返回结果：0.0。
        return 0.0
    # 计算并保存到 x（当前横坐标或横向数据）。
    x = polygon[:, 0]
    # 计算并保存到 y（当前纵坐标或纵向数据）。
    y = polygon[:, 1]
    # 返回结果：计算 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))。
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# 【函数：polygon_area】返回多边形绝对面积，单位随输入坐标平方。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组或可逐项遍历的可逐项遍历的浮点数列表/序列列表/序列。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def polygon_area(points: np.ndarray | Iterable[Iterable[float]]) -> float:
    """返回多边形绝对面积，单位随输入坐标平方。"""
    # 返回结果：调用 abs。
    return abs(signed_area(points))


# 【函数：normalize_polygon】校验多边形并统一循环方向。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组或可逐项遍历的可逐项遍历的浮点数列表/序列列表/序列。
# 参数 min_vertices（最小·vertices）：整数；省略时使用3；必须按参数名传入。
# 参数 max_vertices（最大·vertices）：整数；省略时使用5；必须按参数名传入。
# 参数 min_edge_length_mm（最小·边·长度·毫米）：浮点数；省略时使用0.0；必须按参数名传入。
# 参数 min_area_mm2（最小·面积·平方毫米）：浮点数；省略时使用0.001；必须按参数名传入。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def normalize_polygon(
    points: np.ndarray | Iterable[Iterable[float]],
    *,
    min_vertices: int = 3,
    max_vertices: int = 5,
    min_edge_length_mm: float = 0.0,
    min_area_mm2: float = 1e-3,
) -> np.ndarray:
    """校验多边形并统一循环方向。

    不修改几何尺寸，不合并有效顶点；相邻重复点和退化边直接拒绝，
    避免求解阶段产生不确定姿态。
    """
    # 计算并保存到 polygon（当前多边形的顶点数组）。
    polygon = _as_polygon(points)
    # 判断条件；满足时执行下面代码：not min_vertices <= len(polygon) <= max_vertices。
    if not min_vertices <= len(polygon) <= max_vertices:
        # 抛出异常，通知上层处理：ValueError(f'多边形顶点数必须在{min_vertices}～{max_vertices}之间，当前为{len(polygon…。
        raise ValueError(
            f"多边形顶点数必须在{min_vertices}～{max_vertices}之间，当前为{len(polygon)}"
        )
    # 计算并保存到 lengths（各条边的长度）。
    lengths = edge_lengths(polygon)
    # 判断条件；满足时执行下面代码：np.any(lengths <= 1e-06)。
    if np.any(lengths <= 1e-6):
        # 抛出异常，通知上层处理：ValueError('多边形包含重复相邻点或退化边')。
        raise ValueError("多边形包含重复相邻点或退化边")
    # 判断条件；满足时执行下面代码：min_edge_length_mm > 0.0 and float(np.min(lengths)) < min_edge_length_mm。
    if min_edge_length_mm > 0.0 and float(np.min(lengths)) < min_edge_length_mm:
        # 抛出异常，通知上层处理：ValueError(f'多边形最短边{float(np.min(lengths)):.2f}mm小于限制{min_edge_length…。
        raise ValueError(
            f"多边形最短边{float(np.min(lengths)):.2f}mm小于限制"
            f"{min_edge_length_mm:.2f}mm"
        )
    # 计算并保存到 area（面积）。
    area = polygon_area(polygon)
    # 判断条件；满足时执行下面代码：area < min_area_mm2。
    if area < min_area_mm2:
        # 抛出异常，通知上层处理：ValueError(f'多边形面积{area:.3f}mm²过小或已退化')。
        raise ValueError(f"多边形面积{area:.3f}mm²过小或已退化")
    # 统一为正带符号面积，后续边顺序比较更稳定。
    # 判断条件；满足时执行下面代码：signed_area(polygon) < 0.0。
    if signed_area(polygon) < 0.0:
        # 计算并保存到 polygon（当前多边形的顶点数组）。
        polygon = polygon[::-1].copy()
    # 返回结果：polygon（当前多边形的顶点数组）。
    return polygon


# 【函数：polygon_edges】按顶点循环顺序返回所有 ``2×2`` 边端点。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组或可逐项遍历的可逐项遍历的浮点数列表/序列列表/序列。
# 返回类型：NumPy数组列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def polygon_edges(points: np.ndarray | Iterable[Iterable[float]]) -> list[np.ndarray]:
    """按顶点循环顺序返回所有 ``2×2`` 边端点。"""
    # 计算并保存到 polygon（当前多边形的顶点数组）。
    polygon = _as_polygon(points)
    # 判断条件；满足时执行下面代码：len(polygon) < 2。
    if len(polygon) < 2:
        # 返回结果：创建数据容器。
        return []
    # 返回结果：按条件生成列表。
    return [np.vstack((polygon[index], polygon[(index + 1) % len(polygon)]))
            # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
            for index in range(len(polygon))]


# 【函数：edge_lengths】返回所有循环边长度。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组或可逐项遍历的可逐项遍历的浮点数列表/序列列表/序列。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def edge_lengths(points: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    """返回所有循环边长度。"""
    # 计算并保存到 polygon（当前多边形的顶点数组）。
    polygon = _as_polygon(points)
    # 判断条件；满足时执行下面代码：len(polygon) < 2。
    if len(polygon) < 2:
        # 返回结果：分配指定形状的数组但不初始化元素，后面必须填入数值。
        return np.empty((0,), dtype=np.float64)
    # 返回结果：计算向量长度或指定轴上的范数，默认二维向量为欧氏距离。
    return np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)


# 【函数：rigid_matrix】构造二维齐次刚体矩阵。
# 矩阵前两行是[cos,-sin,tx]和[sin,cos,ty]，第三行[0,0,1]；只有旋转和平移，没有缩放。
# 参数 angle_rad（角度，单位为弧度）：浮点数。
# 参数 translation_xy（translation·XY平面）：可逐项遍历的浮点数列表/序列。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def rigid_matrix(angle_rad: float, translation_xy: Iterable[float]) -> np.ndarray:
    """构造二维齐次刚体矩阵。

    坐标采用项目现有约定：X向右、Y向下，因此正角在图像/机械画面中表现为顺时针。
    """
    # 计算并保存到 translation。
    translation = np.asarray(tuple(translation_xy), dtype=np.float64)
    # 判断条件；满足时执行下面代码：translation.shape != (2,) or not np.isfinite(translation).all()。
    if translation.shape != (2,) or not np.isfinite(translation).all():
        # 抛出异常，通知上层处理：ValueError('平移量必须是两个有限数值')。
        raise ValueError("平移量必须是两个有限数值")
    # 计算并保存到 cosine（角度的余弦值）。
    cosine = math.cos(float(angle_rad))
    # 计算并保存到 sine（角度的正弦值）。
    sine = math.sin(float(angle_rad))
    # 返回结果：创建NumPy数组，用于后续数值或矩阵运算。
    return np.array(
        [[cosine, -sine, translation[0]],
         [sine, cosine, translation[1]],
         [0.0, 0.0, 1.0]],
        # 传入命名参数：dtype=np.float64（指定数组元素类型）；取值过程：np.float64。
        dtype=np.float64,
    )


# 【函数：transform_points】使用3×3齐次矩阵变换一组二维点。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组或可逐项遍历的可逐项遍历的浮点数列表/序列列表/序列。
# 参数 matrix_3x3（3×3齐次变换矩阵）：NumPy数组。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def transform_points(
    points: np.ndarray | Iterable[Iterable[float]],
    matrix_3x3: np.ndarray,
) -> np.ndarray:
    """使用3×3齐次矩阵变换一组二维点。"""
    # 计算并保存到 polygon（当前多边形的顶点数组）。
    polygon = np.asarray(points, dtype=np.float64)
    # 计算并保存到 original_shape（original·形状）。
    original_shape = polygon.shape
    # 判断条件；满足时执行下面代码：polygon.ndim == 3 and polygon.shape[1:] == (1, 2)。
    if polygon.ndim == 3 and polygon.shape[1:] == (1, 2):
        # 计算并保存到 polygon（当前多边形的顶点数组）。
        polygon = polygon[:, 0, :]
    # 判断条件；满足时执行下面代码：polygon.ndim != 2 or polygon.shape[1] != 2。
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        # 抛出异常，通知上层处理：ValueError('点集必须是N×2坐标数组')。
        raise ValueError("点集必须是N×2坐标数组")
    # 计算并保存到 matrix（本函数使用的矩阵）。
    matrix = np.asarray(matrix_3x3, dtype=np.float64)
    # 判断条件；满足时执行下面代码：matrix.shape != (3, 3) or not np.isfinite(matrix).all()。
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        # 抛出异常，通知上层处理：ValueError('变换矩阵必须是3×3有限数值数组')。
        raise ValueError("变换矩阵必须是3×3有限数值数组")
    # 计算并保存到 homogeneous（齐次坐标）。
    homogeneous = np.column_stack((polygon, np.ones(len(polygon), dtype=np.float64)))
    # 计算并保存到 transformed。
    transformed = homogeneous @ matrix.T
    # 判断条件；满足时执行下面代码：np.any(np.abs(transformed[:, 2]) <= 1e-12)。
    if np.any(np.abs(transformed[:, 2]) <= 1e-12):
        # 抛出异常，通知上层处理：ValueError('齐次变换产生无效尺度')。
        raise ValueError("齐次变换产生无效尺度")
    # 计算并保存到 result（当前步骤得到的结果）。
    result = transformed[:, :2] / transformed[:, 2:3]
    # 判断条件；满足时执行下面代码：len(original_shape) == 3。
    if len(original_shape) == 3:
        # 返回结果：调用 result.reshape。
        return result.reshape((-1, 1, 2))
    # 返回结果：result（当前步骤得到的结果）。
    return result


# 【函数：is_rigid_transform】判断矩阵是否为不含缩放、剪切、镜像的二维刚体变换。
# 参数 matrix_3x3（3×3齐次变换矩阵）：NumPy数组。
# 参数 tolerance（用于浮点比较或几何判断的容差）：浮点数；省略时使用1e-06。
# 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
def is_rigid_transform(matrix_3x3: np.ndarray, tolerance: float = 1e-6) -> bool:
    """判断矩阵是否为不含缩放、剪切、镜像的二维刚体变换。"""
    # 计算并保存到 matrix（本函数使用的矩阵）。
    matrix = np.asarray(matrix_3x3, dtype=np.float64)
    # 判断条件；满足时执行下面代码：matrix.shape != (3, 3) or not np.isfinite(matrix).all()。
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        # 返回结果：False。
        return False
    # 判断条件；满足时执行下面代码：not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=tolerance)。
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=tolerance):
        # 返回结果：False。
        return False
    # 计算并保存到 rotation（旋转数据（角度或旋转矩阵，取决于本函数））。
    rotation = matrix[:2, :2]
    # 返回结果：调用 bool。
    return bool(
        # 传入命名参数：atol=tolerance（绝对误差容限）；取值过程：tolerance（用于浮点比较或几何判断的容差）。
        np.allclose(rotation.T @ rotation, np.eye(2), atol=tolerance)
        # 传入命名参数：abs_tol=tolerance（绝对误差容限）；取值过程：tolerance（用于浮点比较或几何判断的容差）。
        and math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=tolerance)
    )


# 【函数：rotation_angle_deg】提取项目坐标系中的旋转角，归一化到[-180, 180)。
# 参数 matrix_3x3（3×3齐次变换矩阵）：NumPy数组。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def rotation_angle_deg(matrix_3x3: np.ndarray) -> float:
    """提取项目坐标系中的旋转角，归一化到[-180, 180)。"""
    # 计算并保存到 matrix（本函数使用的矩阵）。
    matrix = np.asarray(matrix_3x3, dtype=np.float64)
    # 判断条件；满足时执行下面代码：not is_rigid_transform(matrix, tolerance=1e-05)。
    if not is_rigid_transform(matrix, tolerance=1e-5):
        # 抛出异常，通知上层处理：ValueError('矩阵不是有效刚体变换')。
        raise ValueError("矩阵不是有效刚体变换")
    # 计算并保存到 angle（当前计算或搜索的角度）。
    angle = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    # 返回结果：调用 float。
    return float((angle + 180.0) % 360.0 - 180.0)


# 【函数：rigid_align_reversed】将源边反向刚体对齐到目标边。
# 两片沿各自边界走向相接时，一条边的起点对应另一条边的终点。边长有噪声时只对齐方向和中点，不拉伸边长。
# 参数 source_edge（源·边）：NumPy数组或可逐项遍历的可逐项遍历的浮点数列表/序列列表/序列。
# 参数 target_edge（目标·边）：NumPy数组或可逐项遍历的可逐项遍历的浮点数列表/序列列表/序列。
# 返回类型：元组（依次为NumPy数组、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def rigid_align_reversed(
    source_edge: np.ndarray | Iterable[Iterable[float]],
    target_edge: np.ndarray | Iterable[Iterable[float]],
) -> tuple[np.ndarray, float]:
    """将源边反向刚体对齐到目标边。

    映射关系为 ``source[0] -> target[1]``、
    ``source[1] -> target[0]``。两边长度存在小误差时只对齐方向和中点，
    不进行缩放，并返回两个端点的平均残差。
    """
    # 计算并保存到 source（源数据或源位置）。
    source = np.asarray(source_edge, dtype=np.float64).reshape((-1, 2))
    # 计算并保存到 target（目标数据或目标位置）。
    target = np.asarray(target_edge, dtype=np.float64).reshape((-1, 2))
    # 判断条件；满足时执行下面代码：source.shape != (2, 2) or target.shape != (2, 2)。
    if source.shape != (2, 2) or target.shape != (2, 2):
        # 抛出异常，通知上层处理：ValueError('边必须各包含两个二维端点')。
        raise ValueError("边必须各包含两个二维端点")
    # 计算并保存到 source_vector（源·向量）。
    source_vector = source[1] - source[0]
    # 计算并保存到 desired_vector（期望·向量）。
    desired_vector = target[0] - target[1]
    # 计算并保存到 source_length（源·长度）。
    source_length = float(np.linalg.norm(source_vector))
    # 计算并保存到 target_length（目标·长度）。
    target_length = float(np.linalg.norm(desired_vector))
    # 判断条件；满足时执行下面代码：source_length <= 1e-06 or target_length <= 1e-06。
    if source_length <= 1e-6 or target_length <= 1e-6:
        # 抛出异常，通知上层处理：ValueError('不能对齐退化边')。
        raise ValueError("不能对齐退化边")
    # 计算并保存到 source_angle（源·角度）。
    source_angle = math.atan2(float(source_vector[1]), float(source_vector[0]))
    # 计算并保存到 target_angle（目标·角度）。
    target_angle = math.atan2(float(desired_vector[1]), float(desired_vector[0]))
    # 计算并保存到 angle（当前计算或搜索的角度）。
    angle = target_angle - source_angle
    # 计算并保存到 rotation（旋转数据（角度或旋转矩阵，取决于本函数））。
    rotation = rigid_matrix(angle, (0.0, 0.0))
    # 计算并保存到 rotated_midpoint。
    rotated_midpoint = transform_points(
        # 传入命名参数：axis=0（指定运算维度：0通常沿行汇总，1沿列汇总）；取值过程：0；keepdims=True（是否保留被汇总的维度）；取值过程：True。
        np.mean(source, axis=0, keepdims=True), rotation
    )[0]
    # 计算并保存到 target_midpoint（目标·midpoint）。
    target_midpoint = np.mean(target, axis=0)
    # 计算并保存到 matrix（本函数使用的矩阵）。
    matrix = rigid_matrix(angle, target_midpoint - rotated_midpoint)
    # 计算并保存到 mapped。
    mapped = transform_points(source, matrix)
    # 计算并保存到 endpoint_error（端点·误差）。
    endpoint_error = 0.5 * (
        float(np.linalg.norm(mapped[0] - target[1]))
        + float(np.linalg.norm(mapped[1] - target[0]))
    )
    # 判断条件；满足时执行下面代码：not is_rigid_transform(matrix)。
    if not is_rigid_transform(matrix):
        # 抛出异常，通知上层处理：RuntimeError('内部错误：边对齐生成了非刚体矩阵')。
        raise RuntimeError("内部错误：边对齐生成了非刚体矩阵")
    # 返回结果：创建数据容器。
    return matrix, endpoint_error


# 【函数：compose_transforms】按参数顺序组合变换：最后返回 ``M_n @ ... @ M_1``。
# 参数 matrices：NumPy数组；*收集额外的位置参数。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def compose_transforms(*matrices: np.ndarray) -> np.ndarray:
    """按参数顺序组合变换：最后返回 ``M_n @ ... @ M_1``。"""
    # 计算并保存到 result（当前步骤得到的结果）。
    result = np.eye(3, dtype=np.float64)
    # 遍历数据，逐项处理：matrices。
    for matrix in matrices:
        # 计算并保存到 current（当前）。
        current = np.asarray(matrix, dtype=np.float64)
        # 判断条件；满足时执行下面代码：current.shape != (3, 3)。
        if current.shape != (3, 3):
            # 抛出异常，通知上层处理：ValueError('所有变换矩阵必须为3×3')。
            raise ValueError("所有变换矩阵必须为3×3")
        # 计算并保存到 result（当前步骤得到的结果）。
        result = current @ result
    # 返回结果：result（当前步骤得到的结果）。
    return result


# 使用dataclass装饰器，按下方字段自动生成__init__、__repr__和比较方法。
@dataclass
# 【类：_AssemblyCandidate】DFS产生的完整装配体，坐标仍位于任意装配参考系。
class _AssemblyCandidate:
    """DFS产生的完整装配体，坐标仍位于任意装配参考系。"""

    # 声明字段：poses（各碎片的刚体姿态集合）。
    poses: dict[int, np.ndarray]
    # 声明字段：used_edges（已经用于拼缝的边集合）。
    used_edges: frozenset[tuple[int, int]]
    # 声明字段：matches（匹配记录列表）。
    matches: list[EdgeMatch]


# 使用dataclass装饰器，按下方字段自动生成__init__、__repr__和比较方法；frozen=True限制字段重新赋值，但不冻结字段内部的列表/数组。
@dataclass(frozen=True)
# 【类：_CollinearSplitProposal】一条被合并长边的拆分提案及其几何来源。
class _CollinearSplitProposal:
    """一条被合并长边的拆分提案及其几何来源。"""

    # 声明字段：error_mm（误差·毫米）。
    error_mm: float
    # 声明字段：source_piece（源·碎片）。
    source_piece: int
    # 声明字段：source_edge（源·边）。
    source_edge: int
    # 声明字段：segment_lengths_mm（segment·lengths·毫米）。
    segment_lengths_mm: tuple[float, ...]
    # 声明字段：selected_pieces（选中·碎片集合）。
    selected_pieces: tuple[int, ...]
    # 声明字段：split_polygon（拆边·多边形）。
    split_polygon: np.ndarray


# 使用dataclass装饰器，按下方字段自动生成__init__、__repr__和比较方法。
@dataclass
# 【类：_SearchBudget】跨原始边与拆边回退共享的状态和时间预算。
class _SearchBudget:
    """跨原始边与拆边回退共享的状态和时间预算。"""

    # 声明字段：deadline（单调时钟上的截止时刻（秒））。
    deadline: float
    # 声明字段：max_states（允许检查的最大搜索状态数）。
    max_states: int
    # 计算并保存到 states（已检查的搜索状态数）。
    states: int = 0
    # 计算并保存到 stop_reason（停止搜索的原因）。
    stop_reason: str | None = None

    # 【函数：exhausted】检查是否超过时间或状态上限，并记录停止原因。
    # 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
    def exhausted(self) -> bool:
        # 判断条件；满足时执行下面代码：self.stop_reason is not None。
        if self.stop_reason is not None:
            # 返回结果：True。
            return True
        # 判断条件；满足时执行下面代码：self.states >= self.max_states。
        if self.states >= self.max_states:
            # 计算并保存到 self.stop_reason（停止搜索的原因）。
            self.stop_reason = f"达到状态上限{self.max_states}"
            # 返回结果：True。
            return True
        # 判断条件；满足时执行下面代码：time.monotonic() >= self.deadline。
        if time.monotonic() >= self.deadline:
            # 计算并保存到 self.stop_reason（停止搜索的原因）。
            self.stop_reason = "达到时间上限"
            # 返回结果：True。
            return True
        # 返回结果：False。
        return False

    # 【函数：consume】尝试占用一个搜索状态；预算耗尽则返回False。
    # 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
    def consume(self) -> bool:
        # 判断条件；满足时执行下面代码：self.exhausted()。
        if self.exhausted():
            # 返回结果：False。
            return False
        # 更新变量：self.states（已检查的搜索状态数）。
        self.states += 1
        # 返回结果：True。
        return True


# =============================================================================
# 【分区】搜索预算与候选边
# 功能：限制 DFS 时间和状态数；边长接近则生成反向匹配候选。
# 可修改：config 的 generic_search_timeout_seconds（默认 3）、generic_max_search_states（30000）、
#         generic_edge_abs_tolerance_mm、generic_edge_rel_tolerance。
# 看情况改：现场超时搜不完可加 timeout；乱拼则收紧边长容差，不要先放开重叠阈值。
# 不要改：预算是所有拆边变体共享的，不要每个变体重置，否则总时间会翻倍。
# =============================================================================
# 【函数：_make_search_budget】创建所有搜索分支共享的截止时刻和最大状态数。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：_SearchBudget；箭头->是类型提示，不会替你转换实际返回值。
def _make_search_budget(config: dict[str, Any]) -> _SearchBudget:
    # 计算并保存到 timeout_seconds（超时·秒）。
    timeout_seconds = max(0.1, float(config.get(
        "generic_search_timeout_seconds", 3.0
    )))
    # 计算并保存到 max_states（允许检查的最大搜索状态数）。
    max_states = max(1, int(config.get("generic_max_search_states", 30000)))
    # 返回结果：调用 _SearchBudget。
    return _SearchBudget(
        # 传入命名参数：deadline=time.monotonic() + timeout_seconds（单调时钟上的截止时刻（秒））；
        # 取值过程：计算 time.monotonic() + timeout_seconds。
        deadline=time.monotonic() + timeout_seconds,
        # 传入命名参数：max_states=max_states（允许检查的最大搜索状态数）；取值过程：max_states（允许检查的最大搜索状态数）。
        max_states=max_states,
    )


# 【函数：edge_match_tolerance_mm】返回两条边允许的毫米长度误差。
# 参数 length_a（长度·a）：浮点数。
# 参数 length_b（长度·b）：浮点数。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def edge_match_tolerance_mm(length_a: float, length_b: float,
                            config: dict[str, Any]) -> float:
    """返回两条边允许的毫米长度误差。"""
    # 计算并保存到 absolute。
    absolute = float(config.get("generic_edge_abs_tolerance_mm", 2.0))
    # 计算并保存到 relative。
    relative = float(config.get("generic_edge_rel_tolerance", 0.05))
    # 返回结果：调用 max。
    return max(absolute, relative * min(float(length_a), float(length_b)))


# 【函数：generate_edge_candidates】按边长为不同碎片生成无方向拼缝候选。
# 参数 polygons_mm（各碎片的毫米多边形）：NumPy数组列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：EdgeMatch列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def generate_edge_candidates(polygons_mm: list[np.ndarray],
                             config: dict[str, Any]) -> list[EdgeMatch]:
    """按边长为不同碎片生成无方向拼缝候选。"""
    # 计算并保存到 polygons（各块碎片的多边形集合）。
    polygons = [normalize_polygon(polygon) for polygon in polygons_mm]
    # 计算并保存到 lengths（各条边的长度）。
    lengths = [edge_lengths(polygon) for polygon in polygons]
    # 计算并保存到 candidates（待评估的候选集合）。
    candidates: list[EdgeMatch] = []
    # 遍历数据，逐项处理：range(len(polygons))。
    for piece_a in range(len(polygons)):
        # 遍历数据，逐项处理：range(piece_a + 1, len(polygons))。
        for piece_b in range(piece_a + 1, len(polygons)):
            # 遍历数据，逐项处理：enumerate(lengths[piece_a])。
            for edge_a, length_a in enumerate(lengths[piece_a]):
                # 遍历数据，逐项处理：enumerate(lengths[piece_b])。
                for edge_b, length_b in enumerate(lengths[piece_b]):
                    # 计算并保存到 length_error（长度·误差）。
                    length_error = abs(float(length_a) - float(length_b))
                    # 判断条件；满足时执行下面代码：length_error > edge_match_tolerance_mm(length_a, length_b, config)。
                    if length_error > edge_match_tolerance_mm(length_a, length_b, config):
                        # 跳过本轮，进入下一轮循环。
                        continue
                    # 中点刚体对齐时，两端残差均为边长差的一半。
                    # 调用函数：candidates.append。
                    candidates.append(EdgeMatch(
                        # 传入命名参数：piece_a=piece_a（第一块碎片的编号或索引）；取值过程：piece_a（第一块碎片的编号或索引）。
                        piece_a=piece_a,
                        # 传入命名参数：edge_a=edge_a（第一块碎片的边索引）；取值过程：edge_a（第一块碎片的边索引）。
                        edge_a=edge_a,
                        # 传入命名参数：piece_b=piece_b（第二块碎片的编号或索引）；取值过程：piece_b（第二块碎片的编号或索引）。
                        piece_b=piece_b,
                        # 传入命名参数：edge_b=edge_b（第二块碎片的边索引）；取值过程：edge_b（第二块碎片的边索引）。
                        edge_b=edge_b,
                        # 传入命名参数：length_error_mm=length_error（匹配边的长度差（毫米））；取值过程：length_error（长度·误差）。
                        length_error_mm=length_error,
                        # 传入命名参数：endpoint_error_mm=0.5 * length_error（拼缝对应端点的误差（毫米））；取值过程：计算 0.5 * length_error。
                        endpoint_error_mm=0.5 * length_error,
                    ))
    # 计算并保存到 ordered（已排序的数据）。
    ordered = sorted(candidates, key=lambda item: (
        item.length_error_mm, item.piece_a, item.piece_b, item.edge_a, item.edge_b
    ))
    # 参考仿真方案限制候选规模：近似边优先，避免少量噪声边触发组合爆炸。
    # 计算并保存到 max_candidates（最大·候选集合）。
    max_candidates = int(config.get("generic_max_edge_candidates", 40))
    # 返回结果：ordered if max_candidates <= 0 else ordered[:max_candidates]。
    return ordered if max_candidates <= 0 else ordered[:max_candidates]


# 【函数：_raster_coverage】栅格化多边形并返回每像素覆盖次数与平移后的浮点多边形。
# 每片先画成0/1图层并累计；覆盖次数>0是并集，>1是重叠。栅格精度由pixels_per_mm决定，不是相机标定精度。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 pixels_per_mm（每毫米对应的像素数）：浮点数。
# 参数 padding_mm（边缘留白·毫米）：浮点数；省略时使用2.0。
# 返回类型：元组（依次为NumPy数组、NumPy数组）；箭头->是类型提示，不会替你转换实际返回值。
def _raster_coverage(polygons: list[np.ndarray], pixels_per_mm: float,
                     padding_mm: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """栅格化多边形并返回每像素覆盖次数与平移后的浮点多边形。"""
    # 判断条件；满足时执行下面代码：not polygons。
    if not polygons:
        # 返回结果：创建数据容器。
        return np.zeros((1, 1), dtype=np.uint8), np.empty((0, 0, 2), dtype=np.float64)
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = max(0.5, float(pixels_per_mm))
    # 计算并保存到 all_points（all·点集）。
    all_points = np.vstack(polygons)
    # 计算并保存到 minimum（允许下限或计算出的最小值）。
    minimum = np.min(all_points, axis=0) - float(padding_mm)
    # 计算并保存到 shifted（平移或扰动后的数据）。
    shifted = [(polygon - minimum) * ppm for polygon in polygons]
    # 计算并保存到 maximum（允许上限或计算出的最大值）。
    maximum = np.max(np.vstack(shifted), axis=0)
    # 计算并保存到 width（当前区域宽度）。
    width = max(2, int(math.ceil(float(maximum[0]))) + 2)
    # 计算并保存到 height（当前区域高度）。
    height = max(2, int(math.ceil(float(maximum[1]))) + 2)
    # 计算并保存到 coverage（每个栅格像素被多边形覆盖的次数）。
    coverage = np.zeros((height, width), dtype=np.uint8)
    # 遍历数据，逐项处理：shifted。
    for polygon in shifted:
        # 计算并保存到 layer。
        layer = np.zeros_like(coverage)
        # 计算并保存到 integer_polygon（integer·多边形）。
        integer_polygon = np.rint(polygon).astype(np.int32)
        # 调用函数：cv2.fillPoly。
        cv2.fillPoly(layer, [integer_polygon], 1)
        # 计算并保存到 coverage（每个栅格像素被多边形覆盖的次数）。
        coverage = np.minimum(255, coverage + layer).astype(np.uint8)
    # 返回结果：创建数据容器。
    return coverage, np.asarray(shifted, dtype=object)


# 【函数：polygon_overlap_ratio】返回多边形装配中的实质重叠率。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 pixels_per_mm（每毫米对应的像素数）：浮点数；省略时使用2.0。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def polygon_overlap_ratio(polygons: list[np.ndarray],
                          pixels_per_mm: float = 2.0) -> float:
    """返回多边形装配中的实质重叠率。

    共享边在栅格上会占少量同一像素，因此调用者应使用小的非零阈值。
    """
    # 判断条件；满足时执行下面代码：len(polygons) <= 1。
    if len(polygons) <= 1:
        # 返回结果：0.0。
        return 0.0
    # 计算并保存到 coverage（每个栅格像素被多边形覆盖的次数）、_（此处不需要使用的返回值或循环占位变量）。
    coverage, _ = _raster_coverage(polygons, pixels_per_mm)
    # 计算并保存到 union_pixels。
    union_pixels = int(np.count_nonzero(coverage > 0))
    # 计算并保存到 overlap_pixels（重叠·pixels）。
    overlap_pixels = int(np.count_nonzero(coverage > 1))
    # 返回结果：float(overlap_pixels / union_pixels) if union_pixels else 1.0。
    return float(overlap_pixels / union_pixels) if union_pixels else 1.0


# 【函数：_assembly_diameter】返回装配体所有顶点间的最大距离。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def _assembly_diameter(polygons: list[np.ndarray]) -> float:
    """返回装配体所有顶点间的最大距离。"""
    # 计算并保存到 points（参与当前计算的一组坐标点）。
    points = np.vstack(polygons)
    # 计算并保存到 differences。
    differences = points[:, None, :] - points[None, :, :]
    # 返回结果：调用 float。
    return float(np.sqrt(np.max(np.sum(differences * differences, axis=2))))


# 【函数：_pose_state_key】量化姿态和已使用边，抑制由对称边产生的重复状态。
# 量化指把接近的角度与位置归入同一格，再把格号组成可哈希的元组，避免同一布局被不同拼接顺序反复搜索。
# 参数 poses（各碎片的刚体姿态集合）：字典（键为整数，值为NumPy数组）。
# 参数 used_edges（已经用于拼缝的边集合）：不重复的元组（依次为整数、整数）不可变集合。
# 参数 translation_quantum_mm（translation·quantum·毫米）：浮点数。
# 参数 angle_quantum_deg（角度·quantum·度）：浮点数。
# 返回类型：由若干任意类型组成的元组；箭头->是类型提示，不会替你转换实际返回值。
def _pose_state_key(poses: dict[int, np.ndarray],
                    used_edges: frozenset[tuple[int, int]],
                    translation_quantum_mm: float,
                    angle_quantum_deg: float) -> tuple[Any, ...]:
    """量化姿态和已使用边，抑制由对称边产生的重复状态。"""
    # 计算并保存到 pose_items（姿态·items）。
    pose_items: list[tuple[int, int, int, int]] = []
    # 遍历数据，逐项处理：sorted(poses.items())。
    for piece_index, matrix in sorted(poses.items()):
        # 计算并保存到 angle（当前计算或搜索的角度）。
        angle = rotation_angle_deg(matrix)
        # 调用函数：pose_items.append。
        pose_items.append((
            piece_index,
            int(round(float(matrix[0, 2]) / translation_quantum_mm)),
            int(round(float(matrix[1, 2]) / translation_quantum_mm)),
            int(round(angle / angle_quantum_deg)),
        ))
    # 返回结果：创建数据容器。
    return tuple(pose_items), tuple(sorted(used_edges))


# =============================================================================
# 【分区】DFS 装配搜索（固定第 0 片为参考系）
# 功能：一片片按匹配边接入，得到连通装配候选，尚未判断是不是矩形。
# 可修改：无直接魔数；剪枝看重叠/直径相关内部阈值。
# 看情况改：片数>4 本模块直接拒绝。扑克 3 片、图2 的 4 片都在支持范围内。
# 不要改：第 0 片固定，否则同样拼法会因整体旋转重复爆炸；状态键去重不要删。
# =============================================================================
# 【函数：_search_assemblies】使用匹配边生成树搜索装配体，并受全局状态/时间预算约束。
# 固定第0片为参考系，消除所有碎片一起平移/旋转产生的无穷等价解。此处得到连通装配候选，还需后续矩形质量评估。
# 参数 polygons_mm（各碎片的毫米多边形）：NumPy数组列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 budget（所有搜索分支共享的时间和状态数量预算）：_SearchBudget或None（不返回业务结果）；省略时使用None。
# 返回类型：元组（依次为_AssemblyCandidate列表/序列、整数）；箭头->是类型提示，不会替你转换实际返回值。
def _search_assemblies(polygons_mm: list[np.ndarray],
                       config: dict[str, Any],
                       budget: _SearchBudget | None = None,
                       ) -> tuple[list[_AssemblyCandidate], int]:
    """使用匹配边生成树搜索装配体，并受全局状态/时间预算约束。"""
    # 判断条件；满足时执行下面代码：budget is None。
    if budget is None:
        # 计算并保存到 budget（所有搜索分支共享的时间和状态数量预算）。
        budget = _make_search_budget(config)
    # 计算并保存到 polygons（各块碎片的多边形集合）。
    polygons = [normalize_polygon(polygon) for polygon in polygons_mm]
    # 计算并保存到 piece_count（碎片数量）。
    piece_count = len(polygons)
    # 判断条件；满足时执行下面代码：piece_count == 1。
    if piece_count == 1:
        # 返回结果：创建数据容器。
        return [_AssemblyCandidate(
            # 字典字段0（按表达式求键）：创建单位矩阵，对应不旋转、不平移的初始变换。
            poses={0: np.eye(3, dtype=np.float64)},
            # 传入命名参数：used_edges=frozenset()（已经用于拼缝的边集合）；取值过程：调用 frozenset。
            used_edges=frozenset(),
            # 传入命名参数：matches=[]（匹配记录列表）；取值过程：创建数据容器。
            matches=[],
        )], 1

    # 计算并保存到 lengths（各条边的长度）。
    lengths = [edge_lengths(polygon) for polygon in polygons]
    # 计算并保存到 candidate_lookup（候选·lookup）。
    candidate_lookup: set[tuple[int, int, int, int]] = set()
    # 遍历数据，逐项处理：generate_edge_candidates(polygons, config)。
    for candidate in generate_edge_candidates(polygons, config):
        # 调用函数：candidate_lookup.add。
        candidate_lookup.add((candidate.piece_a, candidate.edge_a,
                              candidate.piece_b, candidate.edge_b))
        # 调用函数：candidate_lookup.add。
        candidate_lookup.add((candidate.piece_b, candidate.edge_b,
                              candidate.piece_a, candidate.edge_a))
    # 判断条件；满足时执行下面代码：not candidate_lookup。
    if not candidate_lookup:
        # 返回结果：创建数据容器。
        return [], 0

    # 计算并保存到 overlap_limit（重叠·上限）。
    overlap_limit = float(config.get("generic_search_overlap_ratio", 0.012))
    # 计算并保存到 raster_ppm。
    raster_ppm = float(config.get("generic_search_pixels_per_mm", 2.0))
    # 计算并保存到 maximum_diagonal（最大·diagonal）。
    maximum_diagonal = math.hypot(
        float(config.get("generic_target_long_max_mm", 120.0)),
        float(config.get("generic_target_short_max_mm", 90.0)),
    ) + float(config.get("generic_search_dimension_margin_mm", 5.0))
    # 计算并保存到 translation_quantum。
    translation_quantum = max(0.1, float(config.get(
        "generic_state_translation_quantum_mm", 0.5
    )))
    # 计算并保存到 angle_quantum（角度·quantum）。
    angle_quantum = max(0.1, float(config.get(
        "generic_state_angle_quantum_deg", 0.5
    )))

    # 计算并保存到 complete。
    complete: list[_AssemblyCandidate] = []
    # 计算并保存到 visited（已经访问过的搜索状态集合）。
    visited: set[tuple[Any, ...]] = set()
    # 【函数：dfs】递归尝试给已装配碎片接入下一片，并剪掉重复、过大或重叠的分支。
    # poses是已放置片的姿态，used_edges防止一条边被重复使用，matches记录本分支拼缝。递归前创建下一分支副本，不覆盖兄弟分支。
    # 参数 poses（各碎片的刚体姿态集合）：字典（键为整数，值为NumPy数组）。
    # 参数 used_edges（已经用于拼缝的边集合）：不重复的元组（依次为整数、整数）不可变集合。
    # 参数 matches（匹配记录列表）：EdgeMatch列表/序列。
    # 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
    def dfs(poses: dict[int, np.ndarray],
            used_edges: frozenset[tuple[int, int]],
            matches: list[EdgeMatch]) -> None:
        # 判断条件；满足时执行下面代码：not budget.consume()。
        if not budget.consume():
            # 返回结果：无。
            return
        # 计算并保存到 key（查询、分组或排序所用的键）。
        key = _pose_state_key(
            poses, used_edges, translation_quantum, angle_quantum
        )
        # 判断条件；满足时执行下面代码：key in visited。
        if key in visited:
            # 返回结果：无。
            return
        # 调用函数：visited.add。
        visited.add(key)
        # 判断条件；满足时执行下面代码：len(poses) == piece_count。
        if len(poses) == piece_count:
            # 调用函数：complete.append。
            complete.append(_AssemblyCandidate(
                # 传入命名参数：poses={index: matrix.copy() for index, matrix in poses.items()}（各碎片的刚体姿态集合）；
                # 取值过程：{index: matrix.copy() for index, matrix in poses.items()}。
                poses={index: matrix.copy() for index, matrix in poses.items()},
                # 传入命名参数：used_edges=used_edges（已经用于拼缝的边集合）；取值过程：used_edges（已经用于拼缝的边集合）。
                used_edges=used_edges,
                # 传入命名参数：matches=list(matches)（匹配记录列表）；取值过程：调用 list。
                matches=list(matches),
            ))
            # 返回结果：无。
            return

        # 计算并保存到 options。
        options: list[tuple[float, float, int, int, int, int, np.ndarray]] = []
        # 计算并保存到 unplaced。
        unplaced = [index for index in range(piece_count) if index not in poses]
        # 遍历数据，逐项处理：poses.items()。
        for piece_a, pose_a in poses.items():
            # 计算并保存到 transformed_a。
            transformed_a = transform_points(polygons[piece_a], pose_a)
            # 计算并保存到 edges_a（边集合·a）。
            edges_a = polygon_edges(transformed_a)
            # 遍历数据，逐项处理：enumerate(edges_a)。
            for edge_a, target_edge in enumerate(edges_a):
                # 判断条件；满足时执行下面代码：(piece_a, edge_a) in used_edges。
                if (piece_a, edge_a) in used_edges:
                    # 跳过本轮，进入下一轮循环。
                    continue
                # 遍历数据，逐项处理：unplaced。
                for piece_b in unplaced:
                    # 遍历数据，逐项处理：enumerate(polygon_edges(polygons[piece_b]))。
                    for edge_b, source_edge in enumerate(polygon_edges(polygons[piece_b])):
                        # 判断条件；满足时执行下面代码：(piece_a, edge_a, piece_b, edge_b) not in candidate_lookup。
                        if (piece_a, edge_a, piece_b, edge_b) not in candidate_lookup:
                            # 跳过本轮，进入下一轮循环。
                            continue
                        # 计算并保存到 pose_b（姿态·b）、endpoint_error（端点·误差）。
                        pose_b, endpoint_error = rigid_align_reversed(
                            source_edge, target_edge
                        )
                        # 计算并保存到 length_error（长度·误差）。
                        length_error = abs(
                            float(lengths[piece_a][edge_a])
                            - float(lengths[piece_b][edge_b])
                        )
                        # 调用函数：options.append。
                        options.append((
                            length_error, endpoint_error,
                            piece_a, edge_a, piece_b, edge_b, pose_b,
                        ))
        # 调用函数：options.sort。
        options.sort(key=lambda item: (item[0], item[1], item[4], item[5]))

        # 遍历数据，逐项处理：options。
        for (length_error, endpoint_error, piece_a, edge_a,
             piece_b, edge_b, pose_b) in options:
            # 判断条件；满足时执行下面代码：budget.exhausted()。
            if budget.exhausted():
                # 立即结束当前循环。
                break
            # 计算并保存到 transformed_new。
            transformed_new = transform_points(polygons[piece_b], pose_b)
            # 计算并保存到 placed_polygons（placed·多边形集合）。
            placed_polygons = [
                transform_points(polygons[index], pose)
                # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
                for index, pose in poses.items()
            ]
            # 计算并保存到 candidate_polygons（候选·多边形集合）。
            candidate_polygons = placed_polygons + [transformed_new]
            # 判断条件；满足时执行下面代码：_assembly_diameter(candidate_polygons) > maximum_diagonal。
            if _assembly_diameter(candidate_polygons) > maximum_diagonal:
                # 跳过本轮，进入下一轮循环。
                continue
            # 判断条件；满足时执行下面代码：_interior_overlap_ratio(candidate_polygons, raster_ppm) > overlap_limit。
            if _interior_overlap_ratio(candidate_polygons, raster_ppm) > overlap_limit:
                # 跳过本轮，进入下一轮循环。
                continue
            # 计算并保存到 next_poses（下一项·姿态集合）。
            next_poses = dict(poses)
            # 计算并保存到 next_poses[piece_b]。
            next_poses[piece_b] = pose_b
            # 计算并保存到 next_used（下一项·已使用）。
            next_used = frozenset((*used_edges, (piece_a, edge_a), (piece_b, edge_b)))
            # 计算并保存到 next_matches（下一项·匹配集合）。
            next_matches = matches + [EdgeMatch(
                # 传入命名参数：piece_a=piece_a（第一块碎片的编号或索引）；取值过程：piece_a（第一块碎片的编号或索引）。
                piece_a=piece_a,
                # 传入命名参数：edge_a=edge_a（第一块碎片的边索引）；取值过程：edge_a（第一块碎片的边索引）。
                edge_a=edge_a,
                # 传入命名参数：piece_b=piece_b（第二块碎片的编号或索引）；取值过程：piece_b（第二块碎片的编号或索引）。
                piece_b=piece_b,
                # 传入命名参数：edge_b=edge_b（第二块碎片的边索引）；取值过程：edge_b（第二块碎片的边索引）。
                edge_b=edge_b,
                # 传入命名参数：length_error_mm=length_error（匹配边的长度差（毫米））；取值过程：length_error（长度·误差）。
                length_error_mm=length_error,
                # 传入命名参数：endpoint_error_mm=endpoint_error（拼缝对应端点的误差（毫米））；取值过程：endpoint_error（端点·误差）。
                endpoint_error_mm=endpoint_error,
            )]
            # 调用函数：dfs。
            dfs(next_poses, next_used, next_matches)

    # 任意合法装配都与第0块连通，固定根姿态可去掉全局旋转和平移自由度。
    # 调用函数：dfs。
    dfs({0: np.eye(3, dtype=np.float64)}, frozenset(), [])
    # 判断条件；满足时执行下面代码：budget.stop_reason is not None and (not complete)。
    if budget.stop_reason is not None and not complete:
        # 抛出异常，通知上层处理：ValueError(f'通用拼图搜索{budget.stop_reason}，已检查状态{budget.states}')。
        raise ValueError(
            f"通用拼图搜索{budget.stop_reason}，已检查状态{budget.states}"
        )
    # 返回结果：创建数据容器。
    return complete, budget.states


# 使用dataclass装饰器，按下方字段自动生成__init__、__repr__和比较方法。
@dataclass
# 【类：_RectangleEvaluation】完整装配候选的矩形质量和归一化变换。
class _RectangleEvaluation:
    """完整装配候选的矩形质量和归一化变换。"""

    # 声明字段：candidate（当前候选）。
    candidate: _AssemblyCandidate
    # 声明字段：normalize_matrix（normalize·矩阵）。
    normalize_matrix: np.ndarray
    # 声明字段：size_mm（尺寸·毫米）。
    size_mm: tuple[float, float]
    # 声明字段：score（评分）。
    score: float
    # 声明字段：fill_error_ratio（矩形填充误差比例）。
    fill_error_ratio: float
    # 声明字段：overlap_ratio（碎片间重叠比例）。
    overlap_ratio: float
    # 声明字段：endpoint_error_mm（拼缝对应端点的误差（毫米））。
    endpoint_error_mm: float
    # 声明字段：boundary_error_mm（目标矩形边界误差（毫米））。
    boundary_error_mm: float
    # 声明字段：boundary_gap_ratio（矩形边界缺口比例）。
    boundary_gap_ratio: float
    # 计算并保存到 target_size_error_mm（目标·尺寸·误差·毫米）。
    target_size_error_mm: float = 0.0


# 使用dataclass装饰器，按下方字段自动生成__init__、__repr__和比较方法。
@dataclass
# 【类：_AssemblyAssessment】保留候选的全部质量指标和拒绝原因，供闭环回退与诊断使用。
class _AssemblyAssessment:
    """保留候选的全部质量指标和拒绝原因，供闭环回退与诊断使用。"""

    # 声明字段：evaluation（几何指标评估记录）。
    evaluation: _RectangleEvaluation
    # 声明字段：rejection_reasons（拒绝·原因）。
    rejection_reasons: tuple[str, ...]
    # 声明字段：outside_edge_ok（outside·边·ok）。
    outside_edge_ok: bool

    # 把下面的方法包装成只读属性，外部用对象.属性名读取，无须写括号。
    @property
    # 【函数：accepted】检查当前评估是否没有任何拒绝原因。
    # 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
    def accepted(self) -> bool:
        # 返回结果：not self.rejection_reasons。
        return not self.rejection_reasons


# =============================================================================
# 【分区】矩形质量评估
# 功能：把装配转到“长边沿 X、左上到原点”，算填充、重叠、边界缺口、拼缝端点误差，决定接受或拒绝。
# 可修改：config 的 generic_max_fill_error_ratio、generic_max_boundary_gap_ratio、目标长宽范围。
# 看情况改：现场矩形差 1~2mm 可略放宽边界；若重叠明显是搜错了，不要放宽 overlap。
# 不要改：评估坐标系与 A4 放置是两步。这里只评形状，放到纸面下半区在 _build_solution。
# =============================================================================
# 【函数：_normalization_to_long_x】返回将最小外接矩形长边转到X轴并把左上角移到原点的变换。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 返回类型：元组（依次为NumPy数组、元组（依次为浮点数、浮点数））；箭头->是类型提示，不会替你转换实际返回值。
def _normalization_to_long_x(polygons: list[np.ndarray]) -> tuple[np.ndarray, tuple[float, float]]:
    """返回将最小外接矩形长边转到X轴并把左上角移到原点的变换。"""
    # 计算并保存到 points（参与当前计算的一组坐标点）。
    points = np.vstack(polygons).astype(np.float32)
    # 计算并保存到 rectangle。
    rectangle = cv2.minAreaRect(points.reshape((-1, 1, 2)))
    # 计算并保存到 box。
    box = cv2.boxPoints(rectangle).astype(np.float64)
    # 计算并保存到 vectors。
    vectors = np.roll(box, -1, axis=0) - box
    # 计算并保存到 lengths（各条边的长度）。
    lengths = np.linalg.norm(vectors, axis=1)
    # 计算并保存到 longest。
    longest = vectors[int(np.argmax(lengths))]
    # 计算并保存到 angle（当前计算或搜索的角度）。
    angle = math.atan2(float(longest[1]), float(longest[0]))
    # 矩形边无方向，限制在[-90°, 90°)可避免不必要的180°翻转。
    # 判断条件；满足时执行下面代码：angle >= math.pi / 2.0。
    if angle >= math.pi / 2.0:
        # 更新变量：angle（当前计算或搜索的角度）。
        angle -= math.pi
    # 判断条件；满足时执行下面代码：angle < -math.pi / 2.0。
    elif angle < -math.pi / 2.0:
        # 更新变量：angle（当前计算或搜索的角度）。
        angle += math.pi
    # 计算并保存到 rotation（旋转数据（角度或旋转矩阵，取决于本函数））。
    rotation = rigid_matrix(-angle, (0.0, 0.0))
    # 计算并保存到 rotated_points（rotated·点集）。
    rotated_points = transform_points(points.astype(np.float64), rotation)
    # 计算并保存到 minimum（允许下限或计算出的最小值）。
    minimum = np.min(rotated_points, axis=0)
    # 计算并保存到 translation。
    translation = rigid_matrix(0.0, -minimum)
    # 计算并保存到 normalization。
    normalization = translation @ rotation
    # 计算并保存到 normalized_points（规范化·点集）。
    normalized_points = transform_points(points.astype(np.float64), normalization)
    # 计算并保存到 maximum（允许上限或计算出的最大值）。
    maximum = np.max(normalized_points, axis=0)
    # 计算并保存到 size（尺寸数据）。
    size = (float(maximum[0]), float(maximum[1]))
    # 判断条件；满足时执行下面代码：size[0] + 1e-06 < size[1]。
    if size[0] + 1e-6 < size[1]:
        # 极端OpenCV角度情况下再旋转90°，确保长边沿X。
        # 计算并保存到 quarter_turn。
        quarter_turn = rigid_matrix(-math.pi / 2.0, (0.0, 0.0))
        # 计算并保存到 turned。
        turned = transform_points(normalized_points, quarter_turn)
        # 计算并保存到 turned_minimum（turned·最小）。
        turned_minimum = np.min(turned, axis=0)
        # 计算并保存到 shift（一次平移量）。
        shift = rigid_matrix(0.0, -turned_minimum)
        # 计算并保存到 normalization。
        normalization = shift @ quarter_turn @ normalization
        # 计算并保存到 turned。
        turned = transform_points(points.astype(np.float64), normalization)
        # 计算并保存到 turned_maximum（turned·最大）。
        turned_maximum = np.max(turned, axis=0)
        # 计算并保存到 size（尺寸数据）。
        size = (float(turned_maximum[0]), float(turned_maximum[1]))
    # 返回结果：创建数据容器。
    return normalization, size


# 【函数：_interior_overlap_ratio】腐蚀各片边界后计算实质重叠，排除共享拼缝的单像素影响。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 pixels_per_mm（每毫米对应的像素数）：浮点数。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def _interior_overlap_ratio(polygons: list[np.ndarray],
                            pixels_per_mm: float) -> float:
    """腐蚀各片边界后计算实质重叠，排除共享拼缝的单像素影响。"""
    # 判断条件；满足时执行下面代码：len(polygons) <= 1。
    if len(polygons) <= 1:
        # 返回结果：0.0。
        return 0.0
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = max(1.0, float(pixels_per_mm))
    # 计算并保存到 all_points（all·点集）。
    all_points = np.vstack(polygons)
    # 计算并保存到 minimum（允许下限或计算出的最小值）。
    minimum = np.min(all_points, axis=0) - 2.0
    # 计算并保存到 shifted（平移或扰动后的数据）。
    shifted = [(polygon - minimum) * ppm for polygon in polygons]
    # 计算并保存到 maximum（允许上限或计算出的最大值）。
    maximum = np.max(np.vstack(shifted), axis=0)
    # 计算并保存到 width（当前区域宽度）。
    width = max(3, int(math.ceil(float(maximum[0]))) + 3)
    # 计算并保存到 height（当前区域高度）。
    height = max(3, int(math.ceil(float(maximum[1]))) + 3)
    # 计算并保存到 coverage（每个栅格像素被多边形覆盖的次数）。
    coverage = np.zeros((height, width), dtype=np.uint8)
    # 计算并保存到 kernel（形态学处理所用结构元素）。
    kernel = np.ones((3, 3), dtype=np.uint8)
    # 计算并保存到 union（并集大小）。
    union = np.zeros_like(coverage)
    # 遍历数据，逐项处理：shifted。
    for polygon in shifted:
        # 计算并保存到 layer。
        layer = np.zeros_like(coverage)
        # 调用函数：cv2.fillPoly。
        cv2.fillPoly(layer, [np.rint(polygon).astype(np.int32)], 1)
        # 计算并保存到 union（并集大小）。
        union = np.maximum(union, layer)
        # 计算并保存到 interior。
        interior = cv2.erode(layer, kernel, iterations=1)
        # 计算并保存到 coverage（每个栅格像素被多边形覆盖的次数）。
        coverage = np.minimum(255, coverage + interior).astype(np.uint8)
    # 计算并保存到 union_pixels。
    union_pixels = int(np.count_nonzero(union))
    # 计算并保存到 overlap_pixels（重叠·pixels）。
    overlap_pixels = int(np.count_nonzero(coverage > 1))
    # 返回结果：float(overlap_pixels / union_pixels) if union_pixels else 1.0。
    return float(overlap_pixels / union_pixels) if union_pixels else 1.0


# 【函数：_rectangle_coverage_quality】返回矩形内部缺失比例与四边未覆盖比例。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 width（当前区域宽度）：浮点数。
# 参数 height（当前区域高度）：浮点数。
# 参数 pixels_per_mm（每毫米对应的像素数）：浮点数。
# 参数 boundary_band_mm（边界·带状区域·毫米）：浮点数。
# 返回类型：元组（依次为浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def _rectangle_coverage_quality(
    polygons: list[np.ndarray],
    width: float,
    height: float,
    pixels_per_mm: float,
    boundary_band_mm: float,
) -> tuple[float, float]:
    """返回矩形内部缺失比例与四边未覆盖比例。

    与仅比较面积总和不同，该指标会直接惩罚中央空洞、碎片断开以及
    只有少数顶点碰到外接矩形的伪解。
    """
    # 计算并保存到 ppm（每毫米对应的像素数）。
    ppm = max(1.0, float(pixels_per_mm))
    # 计算并保存到 width_px（宽度·像素）。
    width_px = max(1, int(round(float(width) * ppm)))
    # 计算并保存到 height_px（高度·像素）。
    height_px = max(1, int(round(float(height) * ppm)))
    # 计算并保存到 union（并集大小）。
    union = np.zeros((height_px + 1, width_px + 1), dtype=np.uint8)
    # 遍历数据，逐项处理：polygons。
    for polygon in polygons:
        # 计算并保存到 points（参与当前计算的一组坐标点）。
        points = np.rint(np.asarray(polygon, np.float64) * ppm).astype(np.int32)
        # 调用函数：cv2.fillPoly。
        cv2.fillPoly(union, [points], 1)

    # 计算并保存到 rectangle_pixels。
    rectangle_pixels = union.size
    # 计算并保存到 covered_pixels。
    covered_pixels = int(np.count_nonzero(union))
    # 计算并保存到 fill_error（填充·误差）。
    fill_error = 1.0 - covered_pixels / max(1, rectangle_pixels)

    # 计算并保存到 band（带状区域）。
    band = max(1, int(math.ceil(float(boundary_band_mm) * ppm)))
    # 计算并保存到 band（带状区域）。
    band = min(band, max(1, min(union.shape) // 2))
    # 计算并保存到 top（上边）。
    top = np.any(union[:band, :] > 0, axis=0)
    # 计算并保存到 bottom（下边）。
    bottom = np.any(union[-band:, :] > 0, axis=0)
    # 计算并保存到 left（左边）。
    left = np.any(union[:, :band] > 0, axis=1)
    # 计算并保存到 right（右边）。
    right = np.any(union[:, -band:] > 0, axis=1)
    # 计算并保存到 boundary_samples（边界·样本集合）。
    boundary_samples = top.size + bottom.size + left.size + right.size
    # 计算并保存到 covered_boundary（covered·边界）。
    covered_boundary = (
        int(np.count_nonzero(top)) + int(np.count_nonzero(bottom))
        + int(np.count_nonzero(left)) + int(np.count_nonzero(right))
    )
    # 计算并保存到 boundary_gap（边界·缺口）。
    boundary_gap = 1.0 - covered_boundary / max(1, boundary_samples)
    # 返回结果：创建数据容器。
    return max(0.0, float(fill_error)), max(0.0, float(boundary_gap))


# 【函数：_outside_edge_quality】检查每片是否有完整外边，并始终返回有限的最近边界距离。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 width（当前区域宽度）：浮点数。
# 参数 height（当前区域高度）：浮点数。
# 参数 tolerance_mm（容差·毫米）：浮点数。
# 返回类型：元组（依次为布尔值True/False、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def _outside_edge_quality(polygons: list[np.ndarray], width: float, height: float,
                          tolerance_mm: float) -> tuple[bool, float]:
    """检查每片是否有完整外边，并始终返回有限的最近边界距离。"""
    # 计算并保存到 piece_errors（碎片·errors）。
    piece_errors: list[float] = []
    # 计算并保存到 all_pieces_ok（all·碎片集合·ok）。
    all_pieces_ok = True
    # 遍历数据，逐项处理：polygons。
    for polygon in polygons:
        # 计算并保存到 best_error（最佳·误差）。
        best_error = math.inf
        # 计算并保存到 piece_ok（碎片·ok）。
        piece_ok = False
        # 遍历数据，逐项处理：polygon_edges(polygon)。
        for edge in polygon_edges(polygon):
            # 遍历数据，逐项处理：((0, 0.0), (0, width), (1, 0.0), (1, height))。
            for axis, side in ((0, 0.0), (0, width), (1, 0.0), (1, height)):
                # 计算并保存到 distances。
                distances = np.abs(edge[:, axis] - side)
                # 计算并保存到 error（误差或异常信息）。
                error = float(np.mean(distances))
                # 计算并保存到 best_error（最佳·误差）。
                best_error = min(best_error, error)
                # 判断条件；满足时执行下面代码：float(np.max(distances)) <= tolerance_mm。
                if float(np.max(distances)) <= tolerance_mm:
                    # 计算并保存到 piece_ok（碎片·ok）。
                    piece_ok = True
        # 判断条件；满足时执行下面代码：not math.isfinite(best_error)。
        if not math.isfinite(best_error):
            # 计算并保存到 best_error（最佳·误差）。
            best_error = math.hypot(width, height)
        # 调用函数：piece_errors.append。
        piece_errors.append(best_error)
        # 计算并保存到 all_pieces_ok（all·碎片集合·ok）。
        all_pieces_ok = all_pieces_ok and piece_ok
    # 返回结果：创建数据容器。
    return all_pieces_ok, float(np.mean(piece_errors)) if piece_errors else 0.0


# 【函数：_target_size_error_mm】返回候选矩形与已知目标尺寸的无方向绝对偏差。
# 参数 size_mm（尺寸·毫米）：元组（依次为浮点数、浮点数）。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def _target_size_error_mm(size_mm: tuple[float, float],
                          config: dict[str, Any]) -> float:
    """返回候选矩形与已知目标尺寸的无方向绝对偏差。"""
    # 计算并保存到 configured。
    configured = config.get("target_rectangle_size_mm")
    # 判断条件；满足时执行下面代码：not isinstance(configured, (list, tuple)) or len(configured) != 2。
    if not isinstance(configured, (list, tuple)) or len(configured) != 2:
        # 返回结果：0.0。
        return 0.0
    # 计算并保存到 target（目标数据或目标位置）。
    target = sorted((float(configured[0]), float(configured[1])), reverse=True)
    # 计算并保存到 measured。
    measured = sorted((float(size_mm[0]), float(size_mm[1])), reverse=True)
    # 判断条件；满足时执行下面代码：min(target) <= 0.0 or not np.isfinite(target).all()。
    if min(target) <= 0.0 or not np.isfinite(target).all():
        # 返回结果：0.0。
        return 0.0
    # 返回结果：计算 abs(measured[0] - target[0]) + abs(measured[1] - target[1])。
    return abs(measured[0] - target[0]) + abs(measured[1] - target[1])


# 【函数：_match_endpoint_error】计算当前全局姿态下两条反向拼缝边的平均端点距离。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 poses（各碎片的刚体姿态集合）：字典（键为整数，值为NumPy数组）。
# 参数 match（当前匹配记录）：EdgeMatch。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def _match_endpoint_error(polygons: list[np.ndarray],
                          poses: dict[int, np.ndarray],
                          match: EdgeMatch) -> float:
    """计算当前全局姿态下两条反向拼缝边的平均端点距离。"""
    # 计算并保存到 edge_a（第一块碎片的边索引）。
    edge_a = transform_points(
        polygon_edges(polygons[match.piece_a])[match.edge_a], poses[match.piece_a]
    )
    # 计算并保存到 edge_b（第二块碎片的边索引）。
    edge_b = transform_points(
        polygon_edges(polygons[match.piece_b])[match.edge_b], poses[match.piece_b]
    )
    # 返回结果：计算 0.5 * (float(np.linalg.norm(edge_a[0] - edge_b[1])) + float(np.linalg.norm…。
    return 0.5 * (
        float(np.linalg.norm(edge_a[0] - edge_b[1]))
        + float(np.linalg.norm(edge_a[1] - edge_b[0]))
    )


# 【函数：_refresh_match_errors】在位姿优化后刷新每条拼缝的端点误差。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 poses（各碎片的刚体姿态集合）：字典（键为整数，值为NumPy数组）。
# 参数 matches（匹配记录列表）：EdgeMatch列表/序列。
# 返回类型：EdgeMatch列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def _refresh_match_errors(polygons: list[np.ndarray],
                          poses: dict[int, np.ndarray],
                          matches: list[EdgeMatch]) -> list[EdgeMatch]:
    """在位姿优化后刷新每条拼缝的端点误差。"""
    # 返回结果：按条件生成列表。
    return [EdgeMatch(
        # 传入命名参数：piece_a=match.piece_a（第一块碎片的编号或索引）；取值过程：match.piece_a。
        piece_a=match.piece_a,
        # 传入命名参数：edge_a=match.edge_a（第一块碎片的边索引）；取值过程：match.edge_a。
        edge_a=match.edge_a,
        # 传入命名参数：piece_b=match.piece_b（第二块碎片的编号或索引）；取值过程：match.piece_b。
        piece_b=match.piece_b,
        # 传入命名参数：edge_b=match.edge_b（第二块碎片的边索引）；取值过程：match.edge_b。
        edge_b=match.edge_b,
        # 传入命名参数：length_error_mm=match.length_error_mm（匹配边的长度差（毫米））；取值过程：match.length_error_mm（match.length·误差·毫米）。
        length_error_mm=match.length_error_mm,
        # 传入命名参数：endpoint_error_mm=_match_endpoint_error(polygons, poses, match)（拼缝对应端点的误差（毫米））；
        # 取值过程：调用 _match_endpoint_error。
        endpoint_error_mm=_match_endpoint_error(polygons, poses, match),
    ) for match in matches]

# 【函数：_assess_assembly】计算完整质量指标，并把未通过的约束作为结构化诊断返回。
# 先把整体最小外接矩形长边转到X轴，再检查尺寸、填充、边界缺口、重叠及拼缝端点误差；通过所有启用的约束才是accepted。
# 参数 candidate（当前候选）：_AssemblyCandidate。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：_AssemblyAssessment；箭头->是类型提示，不会替你转换实际返回值。
def _assess_assembly(candidate: _AssemblyCandidate,
                     polygons: list[np.ndarray],
                     config: dict[str, Any]) -> _AssemblyAssessment:
    """计算完整质量指标，并把未通过的约束作为结构化诊断返回。"""
    # 计算并保存到 assembled。
    assembled = [
        transform_points(polygons[index], candidate.poses[index])
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for index in range(len(polygons))
    ]
    # 计算并保存到 normalization、size（尺寸数据）。
    normalization, size = _normalization_to_long_x(assembled)
    # 计算并保存到 normalized_polygons（规范化·多边形集合）。
    normalized_polygons = [
        transform_points(polygon, normalization) for polygon in assembled
    ]
    # 计算并保存到 width（当前区域宽度）、height（当前区域高度）。
    width, height = size
    # 计算并保存到 rectangle_area（rectangle·面积）。
    rectangle_area = width * height
    # 计算并保存到 pieces_area（碎片集合·面积）。
    pieces_area = sum(polygon_area(polygon) for polygon in polygons)
    # 判断条件；满足时执行下面代码：rectangle_area <= 1e-06。
    if rectangle_area <= 1e-6:
        # 计算并保存到 area_error（面积·误差）、coverage_error（覆盖·误差）、boundary_gap（边界·缺口）、overlap（重叠量或重叠比例）。
        area_error, coverage_error, boundary_gap, overlap = math.inf, math.inf, 1.0, 1.0
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 计算并保存到 area_error（面积·误差）。
        area_error = abs(rectangle_area - pieces_area) / rectangle_area
        # 计算并保存到 coverage_error（覆盖·误差）、boundary_gap（边界·缺口）。
        coverage_error, boundary_gap = _rectangle_coverage_quality(
            normalized_polygons,
            width,
            height,
            float(config.get("generic_score_pixels_per_mm", 3.0)),
            float(config.get("generic_boundary_band_mm", 1.5)),
        )
        # 计算并保存到 overlap（重叠量或重叠比例）。
        overlap = _interior_overlap_ratio(
            normalized_polygons,
            float(config.get("generic_score_pixels_per_mm", 3.0)),
        )
    # 计算并保存到 fill_error（填充·误差）。
    fill_error = max(area_error, coverage_error)
    # 计算并保存到 endpoint_error（端点·误差）。
    endpoint_error = (
        float(np.mean([
            _match_endpoint_error(polygons, candidate.poses, match)
            # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
            for match in candidate.matches
        ]))
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        if candidate.matches else 0.0
    )
    # 计算并保存到 outside_ok、boundary_error（边界·误差）。
    outside_ok, boundary_error = _outside_edge_quality(
        normalized_polygons,
        width,
        height,
        float(config.get("generic_boundary_tolerance_mm", 3.0)),
    )
    # 计算并保存到 target_size_error（目标·尺寸·误差）。
    target_size_error = _target_size_error_mm(size, config)
    # 计算并保存到 closure_edges（闭环·边集合）。
    closure_edges = max(0, len(candidate.matches) - max(0, len(polygons) - 1))
    # 计算并保存到 closure_refined（闭环·优化后）。
    closure_refined = closure_edges > 0

    # 计算并保存到 long_min（长边·最小）。
    long_min = float(config.get("generic_target_long_min_mm", 90.0))
    # 计算并保存到 long_max（长边·最大）。
    long_max = float(config.get("generic_target_long_max_mm", 120.0))
    # 计算并保存到 short_min（短边·最小）。
    short_min = float(config.get("generic_target_short_min_mm", 50.0))
    # 计算并保存到 short_max（短边·最大）。
    short_max = float(config.get("generic_target_short_max_mm", 90.0))
    # 计算并保存到 dimension_tolerance（dimension·容差）。
    dimension_tolerance = float(config.get("generic_dimension_tolerance_mm", 1.5))
    # 计算并保存到 overlap_limit（重叠·上限）。
    overlap_limit = float(config.get(
        "generic_closure_max_overlap_ratio" if closure_refined else
        "generic_max_overlap_ratio",
        config.get("generic_max_overlap_ratio", 0.008),
    ))
    # 计算并保存到 endpoint_limit（端点·上限）。
    endpoint_limit = float(config.get(
        "generic_closure_max_endpoint_error_mm" if closure_refined else
        "generic_max_endpoint_error_mm",
        config.get("generic_max_endpoint_error_mm", 3.0),
    ))
    # 计算并保存到 fill_limit（填充·上限）。
    fill_limit = float(config.get("generic_max_fill_error_ratio", 0.06))
    # 计算并保存到 boundary_gap_limit（边界·缺口·上限）。
    boundary_gap_limit = float(config.get("generic_max_boundary_gap_ratio", 0.08))

    # 计算并保存到 rejected（未通过的约束名称列表）。
    rejected: list[str] = []
    # 判断条件；满足时执行下面代码：not long_min - dimension_tolerance <= width <= long_max + dimension_tolerance。
    if not (long_min - dimension_tolerance <= width <= long_max + dimension_tolerance):
        # 调用函数：rejected.append。
        rejected.append("长边尺寸")
    # 判断条件；满足时执行下面代码：not short_min - dimension_tolerance <= height <= short_max + dimension_tolerance。
    if not (short_min - dimension_tolerance <= height <= short_max + dimension_tolerance):
        # 调用函数：rejected.append。
        rejected.append("短边尺寸")
    # 判断条件；满足时执行下面代码：bool(config.get('generic_require_each_piece_outer_edge', True)) and (not outsid…。
    if bool(config.get("generic_require_each_piece_outer_edge", True)) and not outside_ok:
        # 调用函数：rejected.append。
        rejected.append("逐片外边")
    # 判断条件；满足时执行下面代码：overlap > overlap_limit。
    if overlap > overlap_limit:
        # 调用函数：rejected.append。
        rejected.append("重叠")
    # 判断条件；满足时执行下面代码：fill_error > fill_limit。
    if fill_error > fill_limit:
        # 调用函数：rejected.append。
        rejected.append("填充")
    # 判断条件；满足时执行下面代码：boundary_gap > boundary_gap_limit。
    if boundary_gap > boundary_gap_limit:
        # 调用函数：rejected.append。
        rejected.append("边界缺口")
    # 判断条件；满足时执行下面代码：endpoint_error > endpoint_limit。
    if endpoint_error > endpoint_limit:
        # 调用函数：rejected.append。
        rejected.append("拼缝端点")

    # 计算并保存到 score（评分）。
    score = (
        1000.0 * overlap
        + 100.0 * fill_error
        + 120.0 * boundary_gap
        + 10.0 * endpoint_error
        + 10.0 * boundary_error
        + float(config.get("generic_target_size_score_weight", 0.0))
        * target_size_error
        - float(config.get("generic_closure_match_score_bonus", 0.0))
        * closure_edges
    )
    # 计算并保存到 evaluation（几何指标评估记录）。
    evaluation = _RectangleEvaluation(
        # 传入命名参数：candidate=candidate（当前候选）；取值过程：candidate（当前候选）。
        candidate=candidate,
        # 传入命名参数：normalize_matrix=normalization（normalize·矩阵）；取值过程：normalization。
        normalize_matrix=normalization,
        # 传入命名参数：size_mm=(width, height)（尺寸·毫米）；取值过程：创建数据容器。
        size_mm=(width, height),
        # 传入命名参数：score=score（评分）；取值过程：score（评分）。
        score=score,
        # 传入命名参数：fill_error_ratio=fill_error（矩形填充误差比例）；取值过程：fill_error（填充·误差）。
        fill_error_ratio=fill_error,
        # 传入命名参数：overlap_ratio=overlap（碎片间重叠比例）；取值过程：overlap（重叠量或重叠比例）。
        overlap_ratio=overlap,
        # 传入命名参数：endpoint_error_mm=endpoint_error（拼缝对应端点的误差（毫米））；取值过程：endpoint_error（端点·误差）。
        endpoint_error_mm=endpoint_error,
        # 传入命名参数：boundary_error_mm=boundary_error（目标矩形边界误差（毫米））；取值过程：boundary_error（边界·误差）。
        boundary_error_mm=boundary_error,
        # 传入命名参数：boundary_gap_ratio=boundary_gap（矩形边界缺口比例）；取值过程：boundary_gap（边界·缺口）。
        boundary_gap_ratio=boundary_gap,
        # 传入命名参数：target_size_error_mm=target_size_error（目标·尺寸·误差·毫米）；取值过程：target_size_error（目标·尺寸·误差）。
        target_size_error_mm=target_size_error,
    )
    # 返回结果：调用 _AssemblyAssessment。
    return _AssemblyAssessment(
        # 传入命名参数：evaluation=evaluation（几何指标评估记录）；取值过程：evaluation（几何指标评估记录）。
        evaluation=evaluation,
        # 传入命名参数：rejection_reasons=tuple(rejected)（拒绝·原因）；取值过程：调用 tuple。
        rejection_reasons=tuple(rejected),
        # 传入命名参数：outside_edge_ok=outside_ok（outside·边·ok）；取值过程：outside_ok。
        outside_edge_ok=outside_ok,
    )


# =============================================================================
# 【分区】位姿图优化与闭环补边
# 功能：生成树只有 n-1 条缝，矩形还缺闭合缝；补 1~2 条后最小二乘微调各片姿态。
# 可修改：无。内部残差是拼缝端点距离。
# 看情况改：只对“已经像矩形”的候选补边，不要对所有树都优化，会超时。
# 不要改：优化仍是刚体，不会把片拉变形。
# =============================================================================
# 【函数：_evaluate_assembly】返回通过全部硬约束的矩形评估。
# 参数 candidate（当前候选）：_AssemblyCandidate。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：_RectangleEvaluation或None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def _evaluate_assembly(candidate: _AssemblyCandidate,
                       polygons: list[np.ndarray],
                       config: dict[str, Any]) -> _RectangleEvaluation | None:
    """返回通过全部硬约束的矩形评估。"""
    # 计算并保存到 assessment（包含通过与拒绝原因的评估结果）。
    assessment = _assess_assembly(candidate, polygons, config)
    # 返回结果：assessment.evaluation if assessment.accepted else None。
    return assessment.evaluation if assessment.accepted else None


# 【函数：_pose_graph_residual】返回所有拼缝端点的二维闭环残差。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 matches（匹配记录列表）：EdgeMatch列表/序列。
# 参数 poses（各碎片的刚体姿态集合）：字典（键为整数，值为NumPy数组）。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def _pose_graph_residual(polygons: list[np.ndarray],
                         matches: list[EdgeMatch],
                         poses: dict[int, np.ndarray]) -> np.ndarray:
    """返回所有拼缝端点的二维闭环残差。"""
    # 计算并保存到 residuals（所有残差组成的序列）。
    residuals: list[float] = []
    # 计算并保存到 edges_by_piece（边集合·by·碎片）。
    edges_by_piece = [polygon_edges(polygon) for polygon in polygons]
    # 遍历数据，逐项处理：matches。
    for match in matches:
        # 计算并保存到 world_a。
        world_a = transform_points(
            edges_by_piece[match.piece_a][match.edge_a], poses[match.piece_a]
        )
        # 计算并保存到 world_b。
        world_b = transform_points(
            edges_by_piece[match.piece_b][match.edge_b], poses[match.piece_b]
        )
        # 调用函数：residuals.extend。
        residuals.extend((world_a - world_b[::-1]).reshape(-1).tolist())
    # 返回结果：把输入转换成NumPy数组；类型相同且无需复制时可能复用原存储。
    return np.asarray(residuals, dtype=np.float64)


# 【函数：_optimize_pose_graph】固定第0块，使用纯NumPy最小二乘全局分摊闭环拼缝误差。
# 对每个可动片优化角度、tx、ty三个量。微小扰动参数估计雅可比矩阵，再解最小二乘修正；试探多个步长，只接受残差平方和下降的更新。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 matches（匹配记录列表）：EdgeMatch列表/序列。
# 参数 initial_poses（初始·姿态集合）：字典（键为整数，值为NumPy数组）。
# 参数 max_iterations（最大·迭代次数）：整数；省略时使用20。
# 参数 budget（所有搜索分支共享的时间和状态数量预算）：_SearchBudget或None（不返回业务结果）；省略时使用None。
# 返回类型：字典（键为整数，值为NumPy数组）；箭头->是类型提示，不会替你转换实际返回值。
def _optimize_pose_graph(polygons: list[np.ndarray],
                         matches: list[EdgeMatch],
                         initial_poses: dict[int, np.ndarray],
                         max_iterations: int = 20,
                         budget: _SearchBudget | None = None,
                         ) -> dict[int, np.ndarray]:
    """固定第0块，使用纯NumPy最小二乘全局分摊闭环拼缝误差。"""
    # 判断条件；满足时执行下面代码：len(polygons) < 3 or len(matches) < len(polygons)。
    if len(polygons) < 3 or len(matches) < len(polygons):
        # 返回结果：{index: pose.copy() for index, pose in initial_poses.items()}。
        return {index: pose.copy() for index, pose in initial_poses.items()}
    # 计算并保存到 root。
    root = 0
    # 计算并保存到 movable。
    movable = [index for index in range(len(polygons)) if index != root]
    # 判断条件；满足时执行下面代码：any((index not in initial_poses for index in range(len(polygons))))。
    if any(index not in initial_poses for index in range(len(polygons))):
        # 抛出异常，通知上层处理：ValueError('闭环优化缺少碎片初始姿态')。
        raise ValueError("闭环优化缺少碎片初始姿态")

    # 【函数：pack】将可移动碎片的姿态展开成(角度,横向平移,纵向平移)参数序列。
    # 参数 poses（各碎片的刚体姿态集合）：字典（键为整数，值为NumPy数组）。
    # 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
    def pack(poses: dict[int, np.ndarray]) -> np.ndarray:
        # 计算并保存到 values（当前数值集合）。
        values: list[float] = []
        # 遍历数据，逐项处理：movable。
        for index in movable:
            # 计算并保存到 pose（单块碎片的刚体姿态）。
            pose = poses[index]
            # 调用函数：values.extend。
            values.extend((
                math.atan2(float(pose[1, 0]), float(pose[0, 0])),
                float(pose[0, 2]),
                float(pose[1, 2]),
            ))
        # 返回结果：把输入转换成NumPy数组；类型相同且无需复制时可能复用原存储。
        return np.asarray(values, dtype=np.float64)

    # 【函数：unpack】将优化参数序列还原成每块碎片的3×3刚体矩阵。
    # 参数 values（当前数值集合）：NumPy数组。
    # 返回类型：字典（键为整数，值为NumPy数组）；箭头->是类型提示，不会替你转换实际返回值。
    def unpack(values: np.ndarray) -> dict[int, np.ndarray]:
        # 计算并保存到 poses（各碎片的刚体姿态集合）。
        poses = {root: initial_poses[root].copy()}
        # 遍历数据，逐项处理：enumerate(movable)。
        for offset, index in enumerate(movable):
            # 计算并保存到 theta（弧度角）、tx（X平移）、ty（Y平移）。
            theta, tx, ty = values[3 * offset:3 * offset + 3]
            # 计算并保存到 poses[index]。
            poses[index] = rigid_matrix(float(theta), (float(tx), float(ty)))
        # 返回结果：poses（各碎片的刚体姿态集合）。
        return poses

    # 【函数：residual】将当前参数还原为姿态并计算全部拼缝端点残差。
    # 参数 values（当前数值集合）：NumPy数组。
    # 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
    def residual(values: np.ndarray) -> np.ndarray:
        # 返回结果：调用 _pose_graph_residual。
        return _pose_graph_residual(polygons, matches, unpack(values))

    # 计算并保存到 values（当前数值集合）。
    values = pack(initial_poses)
    # 遍历数据，逐项处理：range(max(1, int(max_iterations)))。
    for _ in range(max(1, int(max_iterations))):
        # 判断条件；满足时执行下面代码：budget is not None and budget.exhausted()。
        if budget is not None and budget.exhausted():
            # 立即结束当前循环。
            break
        # 计算并保存到 baseline（当前参数下的残差基准）。
        baseline = residual(values)
        # 判断条件；满足时执行下面代码：len(baseline) == 0。
        if len(baseline) == 0:
            # 立即结束当前循环。
            break
        # 计算并保存到 jacobian（残差对各姿态参数的偏导数矩阵）。
        jacobian = np.empty((len(baseline), len(values)), dtype=np.float64)
        # 遍历数据，逐项处理：range(len(values))。
        for column in range(len(values)):
            # 计算并保存到 step（步长）。
            step = 1e-5 if column % 3 == 0 else 1e-3
            # 计算并保存到 shifted（平移或扰动后的数据）。
            shifted = values.copy()
            # 更新变量：shifted[column]。
            shifted[column] += step
            # 计算并保存到 jacobian[:, column]。
            jacobian[:, column] = (residual(shifted) - baseline) / step
        # 执行可能出错的代码，并由后面的异常分支处理错误。
        try:
            # 计算并保存到 delta（当前增量或修正量）、*_（此处不需要使用的返回值或循环占位变量）（收集其余各项）。
            delta, *_ = np.linalg.lstsq(jacobian, -baseline, rcond=None)
        # 捕获np.linalg.LinAlgError异常，转入下面的处理代码。
        except np.linalg.LinAlgError:
            # 立即结束当前循环。
            break
        # 判断条件；满足时执行下面代码：not np.isfinite(delta).all()。
        if not np.isfinite(delta).all():
            # 立即结束当前循环。
            break
        # 计算并保存到 baseline_cost（当前残差平方和）。
        baseline_cost = float(np.dot(baseline, baseline))
        # 计算并保存到 accepted_delta（通过·delta）。
        accepted_delta: np.ndarray | None = None
        # 遍历数据，逐项处理：(1.0, 0.5, 0.25, 0.125)。
        for scale in (1.0, 0.5, 0.25, 0.125):
            # 计算并保存到 trial_delta。
            trial_delta = scale * delta
            # 计算并保存到 trial。
            trial = values + trial_delta
            # 计算并保存到 trial_residual。
            trial_residual = residual(trial)
            # 判断条件；满足时执行下面代码：float(np.dot(trial_residual, trial_residual)) < baseline_cost。
            if float(np.dot(trial_residual, trial_residual)) < baseline_cost:
                # 计算并保存到 values（当前数值集合）。
                values = trial
                # 计算并保存到 accepted_delta（通过·delta）。
                accepted_delta = trial_delta
                # 立即结束当前循环。
                break
        # 判断条件；满足时执行下面代码：accepted_delta is None or float(np.linalg.norm(accepted_delta)) < 1e-07。
        if accepted_delta is None or float(np.linalg.norm(accepted_delta)) < 1e-7:
            # 立即结束当前循环。
            break
    # 返回结果：将优化参数序列还原成每块碎片的3×3刚体矩阵。
    return unpack(values)


# 【函数：_closure_augmented_candidates】为少量近似矩形的生成树补1～2条闭环拼缝，并执行全局位姿优化。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 base_assessments：_AssemblyAssessment列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 budget（所有搜索分支共享的时间和状态数量预算）：_SearchBudget。
# 返回类型：_AssemblyCandidate列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def _closure_augmented_candidates(
    polygons: list[np.ndarray],
    base_assessments: list[_AssemblyAssessment],
    config: dict[str, Any],
    budget: _SearchBudget,
) -> list[_AssemblyCandidate]:
    """为少量近似矩形的生成树补1～2条闭环拼缝，并执行全局位姿优化。"""
    # 判断条件；满足时执行下面代码：len(polygons) < 3 or not bool(config.get('generic_enable_closure_refine', False…。
    if len(polygons) < 3 or not bool(config.get("generic_enable_closure_refine", False)):
        # 返回结果：创建数据容器。
        return []
    # 计算并保存到 prefiltered。
    prefiltered = [assessment for assessment in base_assessments if (
        assessment.evaluation.fill_error_ratio <= float(config.get(
            "generic_closure_prefilter_fill_error_ratio", 0.16
        ))
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        and assessment.evaluation.boundary_gap_ratio <= float(config.get(
            "generic_closure_prefilter_boundary_gap_ratio", 0.50
        ))
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        and assessment.evaluation.overlap_ratio <= float(config.get(
            "generic_closure_prefilter_overlap_ratio", 0.04
        ))
    )]
    # 调用函数：prefiltered.sort。
    prefiltered.sort(key=lambda item: (
        item.evaluation.score,
        item.evaluation.target_size_error_mm,
        item.evaluation.fill_error_ratio,
        item.evaluation.boundary_gap_ratio,
    ))
    # 计算并保存到 base_limit（base·上限）。
    base_limit = max(1, int(config.get("generic_closure_max_base_candidates", 4)))
    # 计算并保存到 max_extra_edges（最大·extra·边集合）。
    max_extra_edges = max(1, min(2, int(config.get("generic_max_closure_edges", 2))))
    # 计算并保存到 max_initial_gap（最大·初始·缺口）。
    max_initial_gap = float(config.get("generic_closure_max_initial_gap_mm", 20.0))
    # 计算并保存到 max_candidates（最大·候选集合）。
    max_candidates = max(1, int(config.get("generic_closure_max_candidates", 24)))
    # 计算并保存到 iterations（迭代次数）。
    iterations = max(1, int(config.get("generic_closure_optimize_iterations", 20)))
    # 计算并保存到 edge_candidates（边·候选集合）。
    edge_candidates = generate_edge_candidates(polygons, config)

    # 计算并保存到 proposals。
    proposals: list[tuple[tuple[float, ...], _AssemblyCandidate,
                          tuple[EdgeMatch, ...]]] = []
    # 遍历数据，逐项处理：prefiltered[:base_limit]。
    for assessment in prefiltered[:base_limit]:
        # 计算并保存到 base。
        base = assessment.evaluation.candidate
        # 计算并保存到 extras。
        extras: list[tuple[float, EdgeMatch]] = []
        # 遍历数据，逐项处理：edge_candidates。
        for match in edge_candidates:
            # 计算并保存到 edge_a（第一块碎片的边索引）。
            edge_a = (match.piece_a, match.edge_a)
            # 计算并保存到 edge_b（第二块碎片的边索引）。
            edge_b = (match.piece_b, match.edge_b)
            # 判断条件；满足时执行下面代码：edge_a in base.used_edges or edge_b in base.used_edges。
            if edge_a in base.used_edges or edge_b in base.used_edges:
                # 跳过本轮，进入下一轮循环。
                continue
            # 计算并保存到 spatial_error（spatial·误差）。
            spatial_error = _match_endpoint_error(polygons, base.poses, match)
            # 判断条件；满足时执行下面代码：spatial_error <= max_initial_gap。
            if spatial_error <= max_initial_gap:
                # 调用函数：extras.append。
                extras.append((spatial_error, match))
        # 调用函数：extras.sort。
        extras.sort(key=lambda item: (
            item[0], item[1].length_error_mm,
            item[1].piece_a, item[1].edge_a, item[1].piece_b, item[1].edge_b,
        ))
        # 计算并保存到 extras。
        extras = extras[:8]
        # 遍历数据，逐项处理：range(max_extra_edges, 0, -1)。
        for count in range(max_extra_edges, 0, -1):
            # 遍历数据，逐项处理：itertools.combinations(extras, count)。
            for combo in itertools.combinations(extras, count):
                # 计算并保存到 used（已使用）。
                used = set(base.used_edges)
                # 计算并保存到 valid（有效性）。
                valid = True
                # 遍历数据，逐项处理：combo。
                for _, match in combo:
                    # 计算并保存到 endpoints。
                    endpoints = ((match.piece_a, match.edge_a),
                                 (match.piece_b, match.edge_b))
                    # 判断条件；满足时执行下面代码：endpoints[0] in used or endpoints[1] in used。
                    if endpoints[0] in used or endpoints[1] in used:
                        # 计算并保存到 valid（有效性）。
                        valid = False
                        # 立即结束当前循环。
                        break
                    # 调用函数：used.update。
                    used.update(endpoints)
                # 判断条件；满足时执行下面代码：not valid。
                if not valid:
                    # 跳过本轮，进入下一轮循环。
                    continue
                # 调用函数：proposals.append。
                proposals.append((
                    (
                        assessment.evaluation.score,
                        -float(count),
                        sum(item[0] for item in combo),
                        sum(item[1].length_error_mm for item in combo),
                    ),
                    base,
                    tuple(item[1] for item in combo),
                ))
    # 调用函数：proposals.sort。
    proposals.sort(key=lambda item: item[0])

    # 计算并保存到 results（结果集合）。
    results: list[_AssemblyCandidate] = []
    # 计算并保存到 seen（已记录）。
    seen: set[tuple[Any, ...]] = set()
    # 遍历数据，逐项处理：proposals。
    for _, base, extras in proposals:
        # 判断条件；满足时执行下面代码：len(results) >= max_candidates or not budget.consume()。
        if len(results) >= max_candidates or not budget.consume():
            # 立即结束当前循环。
            break
        # 计算并保存到 all_matches（all·匹配集合）。
        all_matches = list(base.matches) + list(extras)
        # 计算并保存到 optimized_poses（optimized·姿态集合）。
        optimized_poses = _optimize_pose_graph(
            polygons, all_matches, base.poses, iterations, budget
        )
        # 计算并保存到 refreshed_matches（refreshed·匹配集合）。
        refreshed_matches = _refresh_match_errors(
            polygons, optimized_poses, all_matches
        )
        # 计算并保存到 used_edges（已经用于拼缝的边集合）。
        used_edges = set(base.used_edges)
        # 遍历数据，逐项处理：extras。
        for match in extras:
            # 调用函数：used_edges.update。
            used_edges.update((
                (match.piece_a, match.edge_a),
                (match.piece_b, match.edge_b),
            ))
        # 计算并保存到 frozen_edges（frozen·边集合）。
        frozen_edges = frozenset(used_edges)
        # 计算并保存到 state_key（state·键）。
        state_key = _pose_state_key(
            optimized_poses,
            frozen_edges,
            max(0.1, float(config.get("generic_state_translation_quantum_mm", 0.5))),
            max(0.1, float(config.get("generic_state_angle_quantum_deg", 0.5))),
        )
        # 判断条件；满足时执行下面代码：state_key in seen。
        if state_key in seen:
            # 跳过本轮，进入下一轮循环。
            continue
        # 调用函数：seen.add。
        seen.add(state_key)
        # 调用函数：results.append。
        results.append(_AssemblyCandidate(
            # 传入命名参数：poses=optimized_poses（各碎片的刚体姿态集合）；取值过程：optimized_poses（optimized·姿态集合）。
            poses=optimized_poses,
            # 传入命名参数：used_edges=frozen_edges（已经用于拼缝的边集合）；取值过程：frozen_edges（frozen·边集合）。
            used_edges=frozen_edges,
            # 传入命名参数：matches=refreshed_matches（匹配记录列表）；取值过程：refreshed_matches（refreshed·匹配集合）。
            matches=refreshed_matches,
        ))
    # 返回结果：results（结果集合）。
    return results


# 【函数：_format_failed_assessment】生成适合终端阅读的最佳失败候选摘要。
# 参数 assessment（包含通过与拒绝原因的评估结果）：_AssemblyAssessment。
# 返回类型：字符串；箭头->是类型提示，不会替你转换实际返回值。
def _format_failed_assessment(assessment: _AssemblyAssessment) -> str:
    """生成适合终端阅读的最佳失败候选摘要。"""
    # 计算并保存到 evaluation（几何指标评估记录）。
    evaluation = assessment.evaluation
    # 计算并保存到 reasons（检查未通过的原因列表）。
    reasons = "/".join(assessment.rejection_reasons) or "无"
    # 返回结果：f'矩形={evaluation.size_mm[0]:.1f}×{evaluation.size_mm[1]:.1f}mm，目标尺寸偏差={evaluation.ta…。
    return (
        f"矩形={evaluation.size_mm[0]:.1f}×{evaluation.size_mm[1]:.1f}mm，"
        f"目标尺寸偏差={evaluation.target_size_error_mm:.1f}mm，"
        f"填充={evaluation.fill_error_ratio:.3f}，"
        f"边界缺口={evaluation.boundary_gap_ratio:.3f}，"
        f"重叠={evaluation.overlap_ratio:.3f}，"
        f"拼缝={evaluation.endpoint_error_mm:.2f}mm，拒绝={reasons}"
    )

# 【函数：_target_top_left】计算A4下半区内的目标矩形左上角。
# 参数 size_mm（尺寸·毫米）：元组（依次为浮点数、浮点数）。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def _target_top_left(size_mm: tuple[float, float],
                     config: dict[str, Any]) -> tuple[float, float]:
    """计算A4下半区内的目标矩形左上角。"""
    # 计算并保存到 paper_width（纸面宽度（毫米））。
    paper_width = float(config.get("paper_width_mm", 210.0))
    # 计算并保存到 paper_height（纸面高度（毫米））。
    paper_height = float(config.get("paper_height_mm", 297.0))
    # 计算并保存到 lower_top（lower·上边）。
    lower_top = float(config.get("target_region_top_mm", paper_height / 2.0))
    # 计算并保存到 width（当前区域宽度）、height（当前区域高度）。
    width, height = size_mm
    # 计算并保存到 x（当前横坐标或横向数据）。
    x = (paper_width - width) / 2.0
    # 计算并保存到 configured_y（configured·Y轴）。
    configured_y = config.get("generic_target_top_mm")
    # 判断条件；满足时执行下面代码：configured_y is None。
    if configured_y is None:
        # 计算并保存到 y（当前纵坐标或纵向数据）。
        y = lower_top + (paper_height - lower_top - height) / 2.0
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 计算并保存到 y（当前纵坐标或纵向数据）。
        y = float(configured_y)
    # 计算并保存到 tolerance（用于浮点比较或几何判断的容差）。
    tolerance = 1e-6
    # 判断条件；满足时执行下面代码：x < -tolerance or x + width > paper_width + tolerance。
    if x < -tolerance or x + width > paper_width + tolerance:
        # 抛出异常，通知上层处理：ValueError('通用拼图目标矩形超出A4纸左右边界')。
        raise ValueError("通用拼图目标矩形超出A4纸左右边界")
    # 判断条件；满足时执行下面代码：y < lower_top - tolerance or y + height > paper_height + tolerance。
    if y < lower_top - tolerance or y + height > paper_height + tolerance:
        # 抛出异常，通知上层处理：ValueError('通用拼图目标矩形未完整位于A4纸下半区')。
        raise ValueError("通用拼图目标矩形未完整位于A4纸下半区")
    # 返回结果：创建数据容器。
    return float(x), float(y)


# =============================================================================
# 【分区】生成最终解（放到 A4 下半区）
# 功能：用目标矩形尺寸算左上角，复合变换得到每片目标多边形。
# 可修改：放置位置由 config 的纸张尺寸和下半区规则决定，见 _target_top_left。
# 看情况改：目标区被占用时改配置里的目标位置，不要改刚体矩阵算法。
# 不要改：relaxed_override 标记必须原样传到 PuzzleMatch，仿真会拒绝宽松解。
# =============================================================================
# 【函数：_build_solution】把装配参考系姿态转换成A4下半区内的最终姿态。
# 参数 evaluation（几何指标评估记录）：_RectangleEvaluation。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 search_states（已检查的搜索状态数）：整数。
# 参数 relaxed_override（是否采用放宽质量要求的兜底结果）：布尔值True/False；省略时使用False。
# 返回类型：GenericSolution；箭头->是类型提示，不会替你转换实际返回值。
def _build_solution(evaluation: _RectangleEvaluation,
                    polygons: list[np.ndarray],
                    config: dict[str, Any],
                    search_states: int,
                    relaxed_override: bool = False) -> GenericSolution:
    """把装配参考系姿态转换成A4下半区内的最终姿态。"""
    # 计算并保存到 top_left（上边·左边）。
    top_left = _target_top_left(evaluation.size_mm, config)
    # 计算并保存到 placement。
    placement = rigid_matrix(0.0, top_left)
    # 计算并保存到 poses（各碎片的刚体姿态集合）。
    poses: list[GenericPiecePose] = []
    # 遍历数据，逐项处理：range(len(polygons))。
    for piece_index in range(len(polygons)):
        # 计算并保存到 final_matrix（最终·矩阵）。
        final_matrix = (
            placement
            @ evaluation.normalize_matrix
            @ evaluation.candidate.poses[piece_index]
        )
        # 判断条件；满足时执行下面代码：not is_rigid_transform(final_matrix, tolerance=1e-05)。
        if not is_rigid_transform(final_matrix, tolerance=1e-5):
            # 抛出异常，通知上层处理：RuntimeError('内部错误：最终目标姿态不是刚体变换')。
            raise RuntimeError("内部错误：最终目标姿态不是刚体变换")
        # 计算并保存到 target_polygon（目标·多边形）。
        target_polygon = transform_points(polygons[piece_index], final_matrix)
        # 调用函数：poses.append。
        poses.append(GenericPiecePose(
            # 传入命名参数：piece_index=piece_index（碎片在列表中的索引（从0开始））；取值过程：piece_index（碎片在列表中的索引（从0开始））。
            piece_index=piece_index,
            # 传入命名参数：transform_3x3=final_matrix（3×3齐次变换矩阵（把源坐标映射到目标坐标））；取值过程：final_matrix（最终·矩阵）。
            transform_3x3=final_matrix,
            # 传入命名参数：target_polygon_mm=target_polygon（目标布局中的多边形顶点（毫米））；取值过程：target_polygon（目标·多边形）。
            target_polygon_mm=target_polygon,
        ))
    # 返回结果：调用 GenericSolution。
    return GenericSolution(
        # 传入命名参数：poses=poses（各碎片的刚体姿态集合）；取值过程：poses（各碎片的刚体姿态集合）。
        poses=poses,
        # 传入命名参数：target_size_mm=evaluation.size_mm（目标宽高（毫米））；取值过程：evaluation.size_mm（evaluation.size·毫米）。
        target_size_mm=evaluation.size_mm,
        # 传入命名参数：target_top_left_mm=top_left（目标区域左上角坐标（毫米））；取值过程：top_left（上边·左边）。
        target_top_left_mm=top_left,
        # 传入命名参数：score=evaluation.score（评分）；取值过程：evaluation.score。
        score=evaluation.score,
        # 传入命名参数：fill_error_ratio=evaluation.fill_error_ratio（矩形填充误差比例）；
        # 取值过程：evaluation.fill_error_ratio（evaluation.fill·误差·比例）。
        fill_error_ratio=evaluation.fill_error_ratio,
        # 传入命名参数：overlap_ratio=evaluation.overlap_ratio（碎片间重叠比例）；
        # 取值过程：evaluation.overlap_ratio（evaluation.overlap·比例）。
        overlap_ratio=evaluation.overlap_ratio,
        # 传入命名参数：endpoint_error_mm=evaluation.endpoint_error_mm（拼缝对应端点的误差（毫米））；
        # 取值过程：evaluation.endpoint_error_mm（evaluation.endpoint·误差·毫米）。
        endpoint_error_mm=evaluation.endpoint_error_mm,
        # 传入命名参数：boundary_error_mm=evaluation.boundary_error_mm（目标矩形边界误差（毫米））；
        # 取值过程：evaluation.boundary_error_mm（evaluation.boundary·误差·毫米）。
        boundary_error_mm=evaluation.boundary_error_mm,
        # 传入命名参数：boundary_gap_ratio=evaluation.boundary_gap_ratio（矩形边界缺口比例）；
        # 取值过程：evaluation.boundary_gap_ratio（evaluation.boundary·缺口·比例）。
        boundary_gap_ratio=evaluation.boundary_gap_ratio,
        # 传入命名参数：matches=list(evaluation.candidate.matches)（匹配记录列表）；取值过程：调用 list。
        matches=list(evaluation.candidate.matches),
        # 传入命名参数：search_states=search_states（已检查的搜索状态数）；取值过程：search_states（已检查的搜索状态数）。
        search_states=search_states,
        # 传入命名参数：relaxed_override=relaxed_override（是否采用放宽质量要求的兜底结果）；取值过程：relaxed_override（是否采用放宽质量要求的兜底结果）。
        relaxed_override=relaxed_override,
    )


# 【函数：_build_fallback_placement】终极兜底：把碎片各自平铺到A4下半区，不拼合、不旋转。
# 这是保持碎片原角度、逐块平移到下半区的排放方案；没有拼接成完整矩形。若下半区放不下，仍可能返回None。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 search_states（已检查的搜索状态数）：整数。
# 返回类型：GenericSolution或None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def _build_fallback_placement(
    polygons: list[np.ndarray],
    config: dict[str, Any],
    search_states: int,
) -> GenericSolution | None:
    """终极兜底：把碎片各自平铺到A4下半区，不拼合、不旋转。

    仅在所有正常拼合路径彻底失败时调用。每个碎片保持原始角度，
    按从上到下、从左到右的次序排布，碎片间保留固定间隙，且确保
    全部位于A4下半区（不越界、不重叠）。返回的解标记relaxed_override，
    运动安全关卡会跳过质量校验但保留硬件安全检查。
    """
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 计算并保存到 paper_width（纸面宽度（毫米））。
        paper_width = float(config.get("paper_width_mm", 210.0))
        # 计算并保存到 paper_height（纸面高度（毫米））。
        paper_height = float(config.get("paper_height_mm", 297.0))
        # 计算并保存到 lower_top（lower·上边）。
        lower_top = float(config.get("target_region_top_mm", paper_height / 2.0))
        # 计算并保存到 margin（边缘预留量）。
        margin = float(config.get("generic_fallback_margin_mm", 5.0))
        # 计算并保存到 gap（缺口）。
        gap = float(config.get("generic_fallback_gap_mm", 3.0))
        # 计算并保存到 max_pieces（最大·碎片集合）。
        max_pieces = int(config.get("max_pieces", 4))
        # 计算并保存到 pieces（碎片列表）。
        pieces = list(polygons)
        # 判断条件；满足时执行下面代码：not 1 <= len(pieces) <= max_pieces。
        if not 1 <= len(pieces) <= max_pieces:
            # 返回结果：None。
            return None
        # 收集每块的局部包围盒（保持原角度）
        # 计算并保存到 boxes。
        boxes: list[tuple[float, float, float, float, np.ndarray]] = []
        # 遍历数据，逐项处理：pieces。
        for polygon in pieces:
            # 计算并保存到 minimum（允许下限或计算出的最小值）。
            minimum = np.min(polygon, axis=0)
            # 计算并保存到 maximum（允许上限或计算出的最大值）。
            maximum = np.max(polygon, axis=0)
            # 调用函数：boxes.append。
            boxes.append((
                float(maximum[0] - minimum[0]),
                float(maximum[1] - minimum[1]),
                float(minimum[0]),
                float(minimum[1]),
                polygon,
            ))
        # 按面积从大到小排，先放大的减少浪费
        # 调用函数：boxes.sort。
        boxes.sort(key=lambda item: item[0] * item[1], reverse=True)

        # 计算并保存到 placed。
        placed: list[np.ndarray] = []
        # 计算并保存到 placements。
        placements: list[np.ndarray] = []
        # 计算并保存到 cursor_x（cursor·X轴）。
        cursor_x = margin
        # 计算并保存到 cursor_y（cursor·Y轴）。
        cursor_y = lower_top + margin
        # 计算并保存到 row_height（行·高度）。
        row_height = 0.0
        # 计算并保存到 max_row_width（最大·行·宽度）。
        max_row_width = paper_width - 2.0 * margin
        # 遍历数据，逐项处理：boxes。
        for (box_w, box_h, origin_x, origin_y, polygon) in boxes:
            # 判断条件；满足时执行下面代码：cursor_x + box_w > paper_width - margin and cursor_x > margin。
            if cursor_x + box_w > paper_width - margin and cursor_x > margin:
                # 换行：回到行首，下移一行
                # 计算并保存到 cursor_x（cursor·X轴）。
                cursor_x = margin
                # 更新变量：cursor_y（cursor·Y轴）。
                cursor_y += row_height + gap
                # 计算并保存到 row_height（行·高度）。
                row_height = 0.0
            # 判断条件；满足时执行下面代码：cursor_y + box_h > paper_height - margin。
            if cursor_y + box_h > paper_height - margin:
                # 下半区放不下：尝试更小的间隙，仍放不下则放弃
                # 计算并保存到 cursor_y（cursor·Y轴）。
                cursor_y = lower_top + margin
                # 计算并保存到 row_height（行·高度）。
                row_height = 0.0
                # 判断条件；满足时执行下面代码：cursor_x + box_w > paper_width - margin and cursor_x > margin。
                if cursor_x + box_w > paper_width - margin and cursor_x > margin:
                    # 计算并保存到 cursor_x（cursor·X轴）。
                    cursor_x = margin
                    # 更新变量：cursor_y（cursor·Y轴）。
                    cursor_y += box_h + gap
                # 判断条件；满足时执行下面代码：cursor_y + box_h > paper_height - margin。
                if cursor_y + box_h > paper_height - margin:
                    # 返回结果：None。
                    return None
            # 平移矩阵：把块的原点平移到当前光标位置
            # 计算并保存到 offset（偏移）。
            offset = np.asarray(
                # 传入命名参数：dtype=np.float64（指定数组元素类型）；取值过程：np.float64。
                [cursor_x - origin_x, cursor_y - origin_y], dtype=np.float64
            )
            # 计算并保存到 matrix（本函数使用的矩阵）。
            matrix = rigid_matrix(0.0, offset)
            # 计算并保存到 transformed。
            transformed = transform_points(polygon, matrix)
            # 校验在下半区内
            # 计算并保存到 tmin。
            tmin = np.min(transformed, axis=0)
            # 计算并保存到 tmax。
            tmax = np.max(transformed, axis=0)
            # 判断条件；满足时执行下面代码：tmin[0] < -1e-06 or tmin[1] < lower_top - 1e-06 or tmax[0] > paper_width + 1e-0…。
            if (tmin[0] < -1e-6 or tmin[1] < lower_top - 1e-6 or
                    tmax[0] > paper_width + 1e-6 or tmax[1] > paper_height + 1e-6):
                # 返回结果：None。
                return None
            # 调用函数：placed.append。
            placed.append(transformed)
            # 调用函数：placements.append。
            placements.append(matrix)
            # 更新变量：cursor_x（cursor·X轴）。
            cursor_x += box_w + gap
            # 计算并保存到 row_height（行·高度）。
            row_height = max(row_height, box_h)

        # 计算并保存到 poses（各碎片的刚体姿态集合）。
        poses: list[GenericPiecePose] = []
        # boxes已排序，需按原多边形索引对应
        # 遍历数据，逐项处理：range(len(pieces))。
        for index in range(len(pieces)):
            # 调用函数：poses.append。
            poses.append(GenericPiecePose(
                # 传入命名参数：piece_index=index（碎片在列表中的索引（从0开始））；取值过程：index（当前元素索引）。
                piece_index=index,
                # 传入命名参数：transform_3x3=placements[index]（3×3齐次变换矩阵（把源坐标映射到目标坐标））；取值过程：placements[index]。
                transform_3x3=placements[index],
                # 传入命名参数：target_polygon_mm=placed[index]（目标布局中的多边形顶点（毫米））；取值过程：placed[index]。
                target_polygon_mm=placed[index],
            ))
        # 兜底解的整体包围盒作为名义矩形，便于预览/报告绘制
        # 计算并保存到 all_points（all·点集）。
        all_points = np.vstack(placed)
        # 计算并保存到 all_min（all·最小）。
        all_min = np.min(all_points, axis=0)
        # 计算并保存到 all_max（all·最大）。
        all_max = np.max(all_points, axis=0)
        # 计算并保存到 overall_size（overall·尺寸）。
        overall_size = (float(all_max[0] - all_min[0]),
                        float(all_max[1] - all_min[1]))
        # 返回结果：调用 GenericSolution。
        return GenericSolution(
            # 传入命名参数：poses=poses（各碎片的刚体姿态集合）；取值过程：poses（各碎片的刚体姿态集合）。
            poses=poses,
            # 传入命名参数：target_size_mm=overall_size（目标宽高（毫米））；取值过程：overall_size（overall·尺寸）。
            target_size_mm=overall_size,
            # 传入命名参数：target_top_left_mm=(float(all_min[0]), float(all_min[1]))（目标区域左上角坐标（毫米））；取值过程：创建数据容器。
            target_top_left_mm=(float(all_min[0]), float(all_min[1])),
            # 传入命名参数：score=1000000000.0（评分）；取值过程：1000000000.0。
            score=1e9,
            # 传入命名参数：fill_error_ratio=1.0（矩形填充误差比例）；取值过程：1.0。
            fill_error_ratio=1.0,
            # 传入命名参数：overlap_ratio=0.0（碎片间重叠比例）；取值过程：0.0。
            overlap_ratio=0.0,
            # 传入命名参数：endpoint_error_mm=0.0（拼缝对应端点的误差（毫米））；取值过程：0.0。
            endpoint_error_mm=0.0,
            # 传入命名参数：boundary_error_mm=0.0（目标矩形边界误差（毫米））；取值过程：0.0。
            boundary_error_mm=0.0,
            # 传入命名参数：boundary_gap_ratio=1.0（矩形边界缺口比例）；取值过程：1.0。
            boundary_gap_ratio=1.0,
            # 传入命名参数：matches=[]（匹配记录列表）；取值过程：创建数据容器。
            matches=[],
            # 传入命名参数：search_states=search_states（已检查的搜索状态数）；取值过程：search_states（已检查的搜索状态数）。
            search_states=search_states,
            # 传入命名参数：relaxed_override=True（是否采用放宽质量要求的兜底结果）；取值过程：True。
            relaxed_override=True,
        )
    # 捕获(ValueError, RuntimeError)异常，转入下面的处理代码。
    except (ValueError, RuntimeError):
        # 返回结果：None。
        return None


# 【类：_NoRectangleSolution】几何输入有效，但当前边表示下未找到合格矩形。
# 继承ValueError，保留父类行为并补充下方字段/方法。
class _NoRectangleSolution(ValueError):
    """几何输入有效，但当前边表示下未找到合格矩形。"""

    # 【函数：__init__】创建无矩形解异常，同时保留最佳失败候选供诊断。
    # 参数 message（说明）：字符串。
    # 参数 best_assessment（最佳·assessment）：_AssemblyAssessment或None（不返回业务结果）；省略时使用None。
    def __init__(self, message: str,
                 best_assessment: _AssemblyAssessment | None = None):
        # 调用函数：super().__init__。
        super().__init__(message)
        # 计算并保存到 self.best_assessment（最佳·assessment）。
        self.best_assessment = best_assessment


# =============================================================================
# 【分区】共线拆边（长边对多条短边）
# 功能：只在直线上插顶点，外形和面积不变，让一条长边能匹配几条短边。
# 可修改：变体数量上限、最多顶点数（插完超过 5 个顶点会拒绝该拆法）。
# 看情况改：现场明显“一大边对两小边”却失败时，才考虑放宽拆边变体数。
# 不要改：禁止改变边的几何形状；拆边不是切割碎片。
# =============================================================================
# 【函数：split_polygon_edge】沿指定直边按比例插入共线顶点，不改变原多边形面积与外形。
# 参数 polygon_mm（单位为毫米的多边形顶点）：NumPy数组。
# 参数 edge_index（边·索引）：整数。
# 参数 segment_lengths_mm（segment·lengths·毫米）：可逐项遍历的浮点数列表/序列。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def split_polygon_edge(
    polygon_mm: np.ndarray,
    edge_index: int,
    segment_lengths_mm: Iterable[float],
) -> np.ndarray:
    """沿指定直边按比例插入共线顶点，不改变原多边形面积与外形。"""
    # 计算并保存到 polygon（当前多边形的顶点数组）。
    polygon = normalize_polygon(polygon_mm)
    # 计算并保存到 lengths（各条边的长度）。
    lengths = np.asarray(list(segment_lengths_mm), dtype=np.float64)
    # 判断条件；满足时执行下面代码：len(lengths) < 2 or not np.isfinite(lengths).all() or np.any(lengths <= 0.0)。
    if len(lengths) < 2 or not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        # 抛出异常，通知上层处理：ValueError('拆边至少需要两个正长度分段')。
        raise ValueError("拆边至少需要两个正长度分段")
    # 判断条件；满足时执行下面代码：len(polygon) + len(lengths) - 1 > 5。
    if len(polygon) + len(lengths) - 1 > 5:
        # 抛出异常，通知上层处理：ValueError('拆边后多边形顶点数超过5')。
        raise ValueError("拆边后多边形顶点数超过5")
    # 计算并保存到 edge（边）。
    edge = int(edge_index) % len(polygon)
    # 计算并保存到 start（起始值或起点）。
    start = polygon[edge]
    # 计算并保存到 end（终止值或终点）。
    end = polygon[(edge + 1) % len(polygon)]
    # 计算并保存到 vector（方向或位移向量）。
    vector = end - start
    # 计算并保存到 edge_length（边·长度）。
    edge_length = float(np.linalg.norm(vector))
    # 判断条件；满足时执行下面代码：edge_length <= 1e-09。
    if edge_length <= 1e-9:
        # 抛出异常，通知上层处理：ValueError('不能拆分退化边')。
        raise ValueError("不能拆分退化边")
    # 计算并保存到 cumulative。
    cumulative = np.cumsum(lengths)[:-1] / float(np.sum(lengths))
    # 计算并保存到 inserted。
    inserted = [start + float(ratio) * vector for ratio in cumulative]
    # 计算并保存到 result（当前步骤得到的结果）。
    result: list[np.ndarray] = []
    # 遍历数据，逐项处理：enumerate(polygon)。
    for index, vertex in enumerate(polygon):
        # 调用函数：result.append。
        result.append(vertex)
        # 判断条件；满足时执行下面代码：index == edge。
        if index == edge:
            # 调用函数：result.extend。
            result.extend(inserted)
    # 返回结果：把输入转换成NumPy数组；类型相同且无需复制时可能复用原存储。
    return np.asarray(result, dtype=np.float64)


# 【函数：_generate_collinear_split_proposals】生成带来源信息的共线长边拆分提案。
# 参数 polygons_mm（各碎片的毫米多边形）：NumPy数组列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：_CollinearSplitProposal列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def _generate_collinear_split_proposals(
    polygons_mm: list[np.ndarray],
    config: dict[str, Any],
) -> list[_CollinearSplitProposal]:
    """生成带来源信息的共线长边拆分提案。"""
    # 计算并保存到 polygons（各块碎片的多边形集合）。
    polygons = [normalize_polygon(polygon) for polygon in polygons_mm]
    # 判断条件；满足时执行下面代码：len(polygons) < 3。
    if len(polygons) < 3:
        # 返回结果：创建数据容器。
        return []
    # 计算并保存到 min_edge（最小·边）。
    min_edge = float(config.get("generic_min_edge_length_mm", 0.0))
    # 计算并保存到 absolute_tolerance（absolute·容差）。
    absolute_tolerance = float(config.get(
        "generic_collinear_split_abs_tolerance_mm", 4.0
    ))
    # 计算并保存到 relative_tolerance（relative·容差）。
    relative_tolerance = float(config.get(
        "generic_collinear_split_rel_tolerance", 0.05
    ))
    # 计算并保存到 lengths_by_piece（lengths·by·碎片）。
    lengths_by_piece = [edge_lengths(polygon) for polygon in polygons]
    # 计算并保存到 raw_proposals（原始·proposals）。
    raw_proposals: list[
        tuple[float, int, int, tuple[float, ...], tuple[int, ...]]
    ] = []

    # 遍历数据，逐项处理：enumerate(polygons)。
    for source_piece, polygon in enumerate(polygons):
        # 计算并保存到 max_segments（最大·segments）。
        max_segments = min(3, 6 - len(polygon))
        # 判断条件；满足时执行下面代码：max_segments < 2。
        if max_segments < 2:
            # 跳过本轮，进入下一轮循环。
            continue
        # 计算并保存到 other_pieces（other·碎片集合）。
        other_pieces = [index for index in range(len(polygons)) if index != source_piece]
        # 遍历数据，逐项处理：enumerate(lengths_by_piece[source_piece])。
        for source_edge, source_length_value in enumerate(lengths_by_piece[source_piece]):
            # 计算并保存到 source_length（源·长度）。
            source_length = float(source_length_value)
            # 判断条件；满足时执行下面代码：source_length <= 2.0 * max(min_edge, 1e-06)。
            if source_length <= 2.0 * max(min_edge, 1e-6):
                # 跳过本轮，进入下一轮循环。
                continue
            # 遍历数据，逐项处理：range(2, min(max_segments, len(other_pieces)) + 1)。
            for segment_count in range(2, min(max_segments, len(other_pieces)) + 1):
                # 遍历数据，逐项处理：itertools.combinations(other_pieces, segment_count)。
                for selected_pieces in itertools.combinations(other_pieces, segment_count):
                    # 计算并保存到 edge_options（边·options）。
                    edge_options = [
                        [
                            float(length)
                            # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
                            for length in lengths_by_piece[piece_index]
                            # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
                            if min_edge - 1e-6 <= float(length) < source_length - 1e-6
                        ]
                        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
                        for piece_index in selected_pieces
                    ]
                    # 判断条件；满足时执行下面代码：any((not options for options in edge_options))。
                    if any(not options for options in edge_options):
                        # 跳过本轮，进入下一轮循环。
                        continue
                    # 遍历数据，逐项处理：itertools.product(*edge_options)。
                    for component_lengths in itertools.product(*edge_options):
                        # 计算并保存到 total（总量）。
                        total = float(sum(component_lengths))
                        # 计算并保存到 tolerance（用于浮点比较或几何判断的容差）。
                        tolerance = max(
                            absolute_tolerance,
                            relative_tolerance * min(source_length, total),
                        )
                        # 计算并保存到 error（误差或异常信息）。
                        error = abs(source_length - total)
                        # 判断条件；满足时执行下面代码：error > tolerance。
                        if error > tolerance:
                            # 跳过本轮，进入下一轮循环。
                            continue
                        # 计算并保存到 scale（当前缩放比例或试探步长比例）。
                        scale = source_length / total
                        # 计算并保存到 scaled。
                        scaled = tuple(float(length * scale) for length in component_lengths)
                        # 判断条件；满足时执行下面代码：min_edge > 0.0 and min(scaled) < min_edge - 1e-06。
                        if min_edge > 0.0 and min(scaled) < min_edge - 1e-6:
                            # 跳过本轮，进入下一轮循环。
                            continue
                        # 遍历数据，逐项处理：set(itertools.permutations(scaled))。
                        for ordered in set(itertools.permutations(scaled)):
                            # 调用函数：raw_proposals.append。
                            raw_proposals.append((
                                error, source_piece, source_edge, ordered,
                                tuple(selected_pieces),
                            ))

    # 调用函数：raw_proposals.sort。
    raw_proposals.sort(key=lambda item: (
        item[0], item[1], item[2], item[3], item[4],
    ))
    # 计算并保存到 proposals。
    proposals: list[_CollinearSplitProposal] = []
    # 计算并保存到 seen（已记录）。
    seen: set[tuple[Any, ...]] = set()
    # 遍历数据，逐项处理：raw_proposals。
    for error, source_piece, source_edge, segment_lengths, selected_pieces in raw_proposals:
        # 计算并保存到 split（拆边）。
        split = split_polygon_edge(
            polygons[source_piece], source_edge, segment_lengths
        )
        # 计算并保存到 key（查询、分组或排序所用的键）。
        key = (
            source_piece,
            source_edge,
            selected_pieces,
            tuple(np.rint(split.reshape(-1) * 1000.0).astype(np.int64)),
        )
        # 判断条件；满足时执行下面代码：key in seen。
        if key in seen:
            # 跳过本轮，进入下一轮循环。
            continue
        # 调用函数：seen.add。
        seen.add(key)
        # 调用函数：proposals.append。
        proposals.append(_CollinearSplitProposal(
            # 传入命名参数：error_mm=error（误差·毫米）；取值过程：error（误差或异常信息）。
            error_mm=error,
            # 传入命名参数：source_piece=source_piece（源·碎片）；取值过程：source_piece（源·碎片）。
            source_piece=source_piece,
            # 传入命名参数：source_edge=source_edge（源·边）；取值过程：source_edge（源·边）。
            source_edge=source_edge,
            # 传入命名参数：segment_lengths_mm=segment_lengths（segment·lengths·毫米）；取值过程：segment_lengths。
            segment_lengths_mm=segment_lengths,
            # 传入命名参数：selected_pieces=selected_pieces（选中·碎片集合）；取值过程：selected_pieces（选中·碎片集合）。
            selected_pieces=selected_pieces,
            # 传入命名参数：split_polygon=split（拆边·多边形）；取值过程：split（拆边）。
            split_polygon=split,
        ))
    # 返回结果：proposals。
    return proposals


# 【函数：_polygon_variant_key】将整组多边形量化为稳定去重键。
# 参数 polygons（各块碎片的多边形集合）：NumPy数组列表/序列。
# 返回类型：由若干任意类型组成的元组；箭头->是类型提示，不会替你转换实际返回值。
def _polygon_variant_key(polygons: list[np.ndarray]) -> tuple[Any, ...]:
    """将整组多边形量化为稳定去重键。"""
    # 返回结果：调用 tuple。
    return tuple(
        (len(polygon), tuple(np.rint(polygon.reshape(-1) * 1000.0).astype(np.int64)))
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for polygon in polygons
    )


# 【函数：generate_collinear_split_variants】生成“一条长边对应其他2～3块短边”的有限回退候选。
# 参数 polygons_mm（各碎片的毫米多边形）：NumPy数组列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组列表/序列列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def generate_collinear_split_variants(
    polygons_mm: list[np.ndarray],
    config: dict[str, Any],
) -> list[list[np.ndarray]]:
    """生成“一条长边对应其他2～3块短边”的有限回退候选。"""
    # 计算并保存到 polygons（各块碎片的多边形集合）。
    polygons = [normalize_polygon(polygon) for polygon in polygons_mm]
    # 计算并保存到 max_variants（最大·变体集合）。
    max_variants = max(1, int(config.get(
        "generic_collinear_split_max_variants", 48
    )))
    # 计算并保存到 variants（拆边变体集合）。
    variants: list[list[np.ndarray]] = []
    # 计算并保存到 seen（已记录）。
    seen: set[tuple[Any, ...]] = set()
    # 遍历数据，逐项处理：_generate_collinear_split_proposals(polygons, config)。
    for proposal in _generate_collinear_split_proposals(polygons, config):
        # 计算并保存到 variant（当前拆边变体）。
        variant = [polygon.copy() for polygon in polygons]
        # 计算并保存到 variant[proposal.source_piece]。
        variant[proposal.source_piece] = proposal.split_polygon.copy()
        # 计算并保存到 key（查询、分组或排序所用的键）。
        key = _polygon_variant_key(variant)
        # 判断条件；满足时执行下面代码：key in seen。
        if key in seen:
            # 跳过本轮，进入下一轮循环。
            continue
        # 调用函数：seen.add。
        seen.add(key)
        # 调用函数：variants.append。
        variants.append(variant)
        # 判断条件；满足时执行下面代码：len(variants) >= max_variants。
        if len(variants) >= max_variants:
            # 立即结束当前循环。
            break
    # 返回结果：variants（拆边变体集合）。
    return variants


# 【函数：generate_dual_collinear_split_variants】生成两个不同碎片同时拆边的受限回退候选。
# 参数 polygons_mm（各碎片的毫米多边形）：NumPy数组列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：NumPy数组列表/序列列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def generate_dual_collinear_split_variants(
    polygons_mm: list[np.ndarray],
    config: dict[str, Any],
) -> list[list[np.ndarray]]:
    """生成两个不同碎片同时拆边的受限回退候选。"""
    # 计算并保存到 polygons（各块碎片的多边形集合）。
    polygons = [normalize_polygon(polygon) for polygon in polygons_mm]
    # 判断条件；满足时执行下面代码：len(polygons) < 4。
    if len(polygons) < 4:
        # 返回结果：创建数据容器。
        return []
    # 计算并保存到 proposals_per_edge（proposals·每·边）。
    proposals_per_edge = max(1, int(config.get(
        "generic_dual_split_proposals_per_edge", 2
    )))
    # 计算并保存到 max_variants（最大·变体集合）。
    max_variants = max(1, int(config.get(
        "generic_dual_split_max_variants", 96
    )))

    # 计算并保存到 grouped。
    grouped: dict[tuple[int, int], list[_CollinearSplitProposal]] = {}
    # 遍历数据，逐项处理：_generate_collinear_split_proposals(polygons, config)。
    for proposal in _generate_collinear_split_proposals(polygons, config):
        # 计算并保存到 key（查询、分组或排序所用的键）。
        key = (proposal.source_piece, proposal.source_edge)
        # 计算并保存到 group。
        group = grouped.setdefault(key, [])
        # 判断条件；满足时执行下面代码：len(group) < proposals_per_edge。
        if len(group) < proposals_per_edge:
            # 调用函数：group.append。
            group.append(proposal)

    # 计算并保存到 limited。
    limited = [proposal for group in grouped.values() for proposal in group]
    # 计算并保存到 pairs。
    pairs: list[
        tuple[float, _CollinearSplitProposal, _CollinearSplitProposal]
    ] = []
    # 遍历数据，逐项处理：itertools.combinations(limited, 2)。
    for first, second in itertools.combinations(limited, 2):
        # 判断条件；满足时执行下面代码：first.source_piece == second.source_piece。
        if first.source_piece == second.source_piece:
            # 跳过本轮，进入下一轮循环。
            continue
        # 计算并保存到 source_pieces（源·碎片集合）。
        source_pieces = {first.source_piece, second.source_piece}
        # 计算并保存到 shared_third_pieces（shared·third·碎片集合）。
        shared_third_pieces = (
            set(first.selected_pieces) & set(second.selected_pieces)
        ) - source_pieces
        # 判断条件；满足时执行下面代码：not shared_third_pieces。
        if not shared_third_pieces:
            # 跳过本轮，进入下一轮循环。
            continue
        # 调用函数：pairs.append。
        pairs.append((first.error_mm + second.error_mm, first, second))

    # 调用函数：pairs.sort。
    pairs.sort(key=lambda item: (
        item[0],
        item[1].error_mm, item[1].source_piece, item[1].source_edge,
        item[1].segment_lengths_mm, item[1].selected_pieces,
        item[2].error_mm, item[2].source_piece, item[2].source_edge,
        item[2].segment_lengths_mm, item[2].selected_pieces,
    ))
    # 计算并保存到 variants（拆边变体集合）。
    variants: list[list[np.ndarray]] = []
    # 计算并保存到 seen（已记录）。
    seen: set[tuple[Any, ...]] = set()
    # 遍历数据，逐项处理：pairs。
    for _, first, second in pairs:
        # 计算并保存到 variant（当前拆边变体）。
        variant = [polygon.copy() for polygon in polygons]
        # 计算并保存到 variant[first.source_piece]。
        variant[first.source_piece] = first.split_polygon.copy()
        # 计算并保存到 variant[second.source_piece]。
        variant[second.source_piece] = second.split_polygon.copy()
        # 计算并保存到 key（查询、分组或排序所用的键）。
        key = _polygon_variant_key(variant)
        # 判断条件；满足时执行下面代码：key in seen。
        if key in seen:
            # 跳过本轮，进入下一轮循环。
            continue
        # 调用函数：seen.add。
        seen.add(key)
        # 调用函数：variants.append。
        variants.append(variant)
        # 判断条件；满足时执行下面代码：len(variants) >= max_variants。
        if len(variants) >= max_variants:
            # 立即结束当前循环。
            break
    # 返回结果：variants（拆边变体集合）。
    return variants

# =============================================================================
# 【分区】单套多边形求解（含宽松分支）
# 功能：先严搜，失败再补闭环；仍失败才放宽部分阈值。宽松不是“保证有解”。
# 可修改：宽松分支仍限制 overlap 等硬指标；具体倍数在函数体内。
# 看情况改：enable fallback 平铺的开关在 solve_generic_rectangle。比赛要求严格矩形时不要开平铺兜底。
# 不要改：预算耗尽要失败返回，不要死循环。
# =============================================================================
# 【函数：_solve_normalized_rectangle】求解已校验多边形；普通生成树失败时受限补充闭环拼缝。
# 注意旧注释中的“保证有解”是意图而非无条件保证。实际宽松分支仍筛选候选，并可继续失败；它把多项阈值调大，默认仍允许最多0.1的重叠比例。
# 参数 normalized（经过本函数规范化处理的数据）：NumPy数组列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 budget（所有搜索分支共享的时间和状态数量预算）：_SearchBudget或None（不返回业务结果）；省略时使用None。
# 返回类型：GenericSolution；箭头->是类型提示，不会替你转换实际返回值。
def _solve_normalized_rectangle(
    normalized: list[np.ndarray],
    config: dict[str, Any],
    budget: _SearchBudget | None = None,
) -> GenericSolution:
    """求解已校验多边形；普通生成树失败时受限补充闭环拼缝。"""
    # 判断条件；满足时执行下面代码：budget is None。
    if budget is None:
        # 计算并保存到 budget（所有搜索分支共享的时间和状态数量预算）。
        budget = _make_search_budget(config)
    # 计算并保存到 assemblies、states（已检查的搜索状态数）。
    assemblies, states = _search_assemblies(normalized, config, budget)
    # 判断条件；满足时执行下面代码：not assemblies。
    if not assemblies:
        # 计算并保存到 candidate_count（候选·数量）。
        candidate_count = len(generate_edge_candidates(normalized, config))
        # 抛出异常，通知上层处理：_NoRectangleSolution(f'未找到连通装配：候选拼缝={candidate_count}，搜索状态={states}')。
        raise _NoRectangleSolution(
            f"未找到连通装配：候选拼缝={candidate_count}，搜索状态={states}"
        )

    # 计算并保存到 assessments。
    assessments = [
        _assess_assembly(candidate, normalized, config)
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for candidate in assemblies
    ]
    # 计算并保存到 accepted（通过判据的结果或标记）。
    accepted = [item.evaluation for item in assessments if item.accepted]
    # 计算并保存到 closure_candidates（闭环·候选集合）。
    closure_candidates: list[_AssemblyCandidate] = []
    # 判断条件；满足时执行下面代码：not accepted and (not budget.exhausted())。
    if not accepted and not budget.exhausted():
        # 计算并保存到 closure_candidates（闭环·候选集合）。
        closure_candidates = _closure_augmented_candidates(
            normalized, assessments, config, budget
        )
        # 计算并保存到 closure_assessments（闭环·assessments）。
        closure_assessments = [
            _assess_assembly(candidate, normalized, config)
            # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
            for candidate in closure_candidates
        ]
        # 调用函数：assessments.extend。
        assessments.extend(closure_assessments)
        # 调用函数：accepted.extend。
        accepted.extend(
            item.evaluation for item in closure_assessments if item.accepted
        )

    # 判断条件；满足时执行下面代码：not accepted。
    if not accepted:
        # 严格标准无合格解时，若已到达搜索预算上限（时间/状态耗尽），
        # 用放宽的填充/边界缺口阈值兜底评估现有候选，输出"宽松合格"解，
        # 保证超时后仍能给出可用的矩形拼法。仅超时触发，正常时间内
        # 仍按严格标准求解，优秀/合格判据完全不变。
        # 计算并保存到 relaxed（宽松模式）。
        relaxed = None
        # 判断条件；满足时执行下面代码：budget.exhausted()。
        if budget.exhausted():
            # 宽松兜底：超时后不设任何质量门槛，只要求该候选能拼成一个矩形
            # （无重叠、拼缝端点可接受），直接输出所有候选中评分最佳者。
            # 这样无论碎片轮廓多差，超时后都保证有解可执行，满足比赛"必须完成"。
            # 计算并保存到 relaxed_config（宽松模式·配置）。
            relaxed_config = dict(config)
            # 计算并保存到 relaxed_config['generic_max_fill_error_ratio']。
            relaxed_config["generic_max_fill_error_ratio"] = 1e6
            # 计算并保存到 relaxed_config['generic_max_boundary_gap_ratio']。
            relaxed_config["generic_max_boundary_gap_ratio"] = 1e6
            # 计算并保存到 relaxed_config['generic_target_long_min_mm']。
            relaxed_config["generic_target_long_min_mm"] = 0.0
            # 计算并保存到 relaxed_config['generic_target_long_max_mm']。
            relaxed_config["generic_target_long_max_mm"] = 1e6
            # 计算并保存到 relaxed_config['generic_target_short_min_mm']。
            relaxed_config["generic_target_short_min_mm"] = 0.0
            # 计算并保存到 relaxed_config['generic_target_short_max_mm']。
            relaxed_config["generic_target_short_max_mm"] = 1e6
            # 计算并保存到 relaxed_config['generic_require_each_piece_outer_edge']。
            relaxed_config["generic_require_each_piece_outer_edge"] = False
            # 计算并保存到 relaxed_config['generic_dimension_tolerance_mm']。
            relaxed_config["generic_dimension_tolerance_mm"] = 1e6
            # 计算并保存到 relaxed_config['generic_max_overlap_ratio']。
            relaxed_config["generic_max_overlap_ratio"] = float(config.get(
                "generic_relaxed_max_overlap_ratio", 0.1))
            # 计算并保存到 relaxed_config['generic_closure_max_overlap_ratio']。
            relaxed_config["generic_closure_max_overlap_ratio"] = float(config.get(
                "generic_relaxed_max_overlap_ratio", 0.1))
            # 计算并保存到 relaxed_config['generic_max_endpoint_error_mm']。
            relaxed_config["generic_max_endpoint_error_mm"] = 1e6
            # 计算并保存到 relaxed_config['generic_closure_max_endpoint_error_mm']。
            relaxed_config["generic_closure_max_endpoint_error_mm"] = 1e6
            # 计算并保存到 relaxed_assessments（宽松模式·assessments）。
            relaxed_assessments = [
                _assess_assembly(candidate, normalized, relaxed_config)
                # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
                for candidate in (assemblies + closure_candidates)
            ]
            # 计算并保存到 relaxed_accepted（宽松模式·通过）。
            relaxed_accepted = [
                item.evaluation for item in relaxed_assessments if item.accepted
            ]
            # 判断条件；满足时执行下面代码：relaxed_accepted。
            if relaxed_accepted:
                # 计算并保存到 relaxed（宽松模式）。
                relaxed = min(relaxed_accepted, key=lambda item: (
                    item.score,
                    item.target_size_error_mm,
                    item.fill_error_ratio,
                    item.boundary_gap_ratio,
                    item.overlap_ratio,
                    item.endpoint_error_mm,
                ))
                # 调用函数：print。
                print(f"[保底-宽松] 严格标准超时无合格解，采用超时兜底矩形："
                      f"填充={relaxed.fill_error_ratio:.3f}，"
                      f"边界缺口={relaxed.boundary_gap_ratio:.3f}")
        # 判断条件；满足时执行下面代码：relaxed is not None。
        if relaxed is not None:
            # 返回结果：调用 _build_solution。
            return _build_solution(relaxed, normalized, config, budget.states,
                                   # 传入命名参数：relaxed_override=True（是否采用放宽质量要求的兜底结果）；取值过程：True。
                                   relaxed_override=True)

        # 计算并保存到 best_assessment（最佳·assessment）。
        best_assessment = min(assessments, key=lambda item: (
            len(item.rejection_reasons),
            item.evaluation.score,
            item.evaluation.target_size_error_mm,
            item.evaluation.fill_error_ratio,
        ))
        # 计算并保存到 closure_text（闭环·文本）。
        closure_text = (
            f"，闭环候选={len(closure_candidates)}" if closure_candidates else ""
        )
        # 抛出异常，通知上层处理：_NoRectangleSolution(f'找到{len(assemblies)}个连通装配{closure_text}，但均未通过约束…。
        raise _NoRectangleSolution(
            f"找到{len(assemblies)}个连通装配{closure_text}，但均未通过约束；"
            f"最佳失败候选：{_format_failed_assessment(best_assessment)}",
            best_assessment,
        )

    # 计算并保存到 best（最佳）。
    best = min(accepted, key=lambda item: (
        item.score,
        item.target_size_error_mm,
        item.fill_error_ratio,
        item.boundary_gap_ratio,
        item.overlap_ratio,
        item.endpoint_error_mm,
    ))
    # 返回结果：调用 _build_solution。
    return _build_solution(best, normalized, config, budget.states)

# 【函数：_is_excellent_split_solution】判断拆边回退是否已得到无需继续枚举的高质量解。
# 参数 solution（求解得到的完整布局）：GenericSolution。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
def _is_excellent_split_solution(
    solution: GenericSolution,
    config: dict[str, Any],
) -> bool:
    """判断拆边回退是否已得到无需继续枚举的高质量解。"""
    # 返回结果：组合多个条件：solution.fill_error_ratio <= float(config.get('generic_excellent_fill_erro…。
    return (
        solution.fill_error_ratio <= float(config.get(
            "generic_excellent_fill_error_ratio", 0.025
        ))
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        and solution.overlap_ratio <= float(config.get(
            "generic_excellent_overlap_ratio", 0.003
        ))
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        and solution.boundary_gap_ratio <= float(config.get(
            "generic_excellent_boundary_gap_ratio", 0.03
        ))
    )


# 【函数：_solve_split_variants】在共享预算内评估一组拆边候选并返回其中最佳解。
# 参数 variants（拆边变体集合）：NumPy数组列表/序列列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 参数 budget（所有搜索分支共享的时间和状态数量预算）：_SearchBudget。
# 返回类型：元组（依次为GenericSolution或None（不返回业务结果）、整数）；箭头->是类型提示，不会替你转换实际返回值。
def _solve_split_variants(
    variants: list[list[np.ndarray]],
    config: dict[str, Any],
    budget: _SearchBudget,
) -> tuple[GenericSolution | None, int]:
    """在共享预算内评估一组拆边候选并返回其中最佳解。"""
    # 计算并保存到 best_solution（最佳·solution）。
    best_solution: GenericSolution | None = None
    # 计算并保存到 best_variant_index（最佳·变体·索引）。
    best_variant_index = 0
    # 遍历数据，逐项处理：enumerate(variants, 1)。
    for variant_index, variant in enumerate(variants, 1):
        # 判断条件；满足时执行下面代码：budget.exhausted()。
        if budget.exhausted():
            # 立即结束当前循环。
            break
        # 执行可能出错的代码，并由后面的异常分支处理错误。
        try:
            # 计算并保存到 solution（求解得到的完整布局）。
            solution = _solve_normalized_rectangle(variant, config, budget)
        # 捕获_NoRectangleSolution异常，转入下面的处理代码。
        except _NoRectangleSolution:
            # 跳过本轮，进入下一轮循环。
            continue
        # 捕获ValueError异常，转入下面的处理代码；异常对象保存在exc。
        except ValueError as exc:
            # 超时或状态预算耗尽时：若已有最佳解则保底返回（比赛保证能拼成矩形，
            # 优先放行最高得分解法，而非判失败）；否则安全失败。
            # 判断条件；满足时执行下面代码：best_solution is not None。
            if best_solution is not None:
                # 调用函数：print。
                print(f"[保底] 搜索{exc}，采用当前最高得分解法（评分={best_solution.score:.3f}）")
                # 立即结束当前循环。
                break
            # 抛出异常，通知上层处理：ValueError(str(exc))。
            raise ValueError(str(exc)) from exc

        # 计算并保存到 key（查询、分组或排序所用的键）。
        key = (
            solution.score, solution.fill_error_ratio, solution.boundary_gap_ratio,
            solution.overlap_ratio, solution.endpoint_error_mm,
        )
        # 判断条件；满足时执行下面代码：best_solution is None or key < (best_solution.score, best_solution.fill_error_r…。
        if best_solution is None or key < (
            best_solution.score, best_solution.fill_error_ratio,
            best_solution.boundary_gap_ratio, best_solution.overlap_ratio,
            best_solution.endpoint_error_mm,
        ):
            # 计算并保存到 best_solution（最佳·solution）。
            best_solution = solution
            # 计算并保存到 best_variant_index（最佳·变体·索引）。
            best_variant_index = variant_index
        # 判断条件；满足时执行下面代码：_is_excellent_split_solution(solution, config)。
        if _is_excellent_split_solution(solution, config):
            # 立即结束当前循环。
            break
    # 返回结果：创建数据容器。
    return best_solution, best_variant_index


# 【函数：_print_split_success】输出单拆或双拆回退的统一诊断信息。
# 参数 label：字符串。
# 参数 solution（求解得到的完整布局）：GenericSolution。
# 参数 variant_index（变体·索引）：整数。
# 参数 variant_count（变体·数量）：整数。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def _print_split_success(
    label: str,
    solution: GenericSolution,
    variant_index: int,
    variant_count: int,
) -> None:
    """输出单拆或双拆回退的统一诊断信息。"""
    # 调用函数：print。
    print(
        f"[诊断] {label}成功：候选={variant_index}/{variant_count}，"
        f"矩形={solution.target_size_mm[0]:.1f}×"
        f"{solution.target_size_mm[1]:.1f}mm，"
        f"填充={solution.fill_error_ratio:.3f}，"
        f"边界缺口={solution.boundary_gap_ratio:.3f}"
    )


# =============================================================================
# 【分区】对外入口 solve_generic_rectangle
# 功能：校验 1~4 片 → 原始边求解 → 双拆 → 单拆 → 可选平铺兜底。
# 可修改：generic_min_edge_length_mm、generic_min_piece_area_mm2；片数限制 1~4。
# 看情况改：拆边顺序（先双后单）不要轻易对调，双拆更针对“两边对一边”。
# 不要改：输入必须毫米坐标；不要在入口把像素轮廓直接扔进来。
# =============================================================================
# 【函数：solve_generic_rectangle】搜索矩形；原始边失败时依次尝试受限双拆与单拆回退。
# 原始边失败后尝试双拆，再尝试单拆；它们共用同一个budget。最终平铺兜底仅在代码到达对应分支且配置启用时触发，不覆盖所有早期异常。
# 参数 polygons_mm（各碎片的毫米多边形）：NumPy数组列表/序列。
# 参数 config（程序配置字典）：字典（键为字符串，值为任意类型）。
# 返回类型：GenericSolution；箭头->是类型提示，不会替你转换实际返回值。
def solve_generic_rectangle(
    polygons_mm: list[np.ndarray],
    config: dict[str, Any],
) -> GenericSolution:
    """搜索矩形；原始边失败时依次尝试受限双拆与单拆回退。"""
    # 判断条件；满足时执行下面代码：not 1 <= len(polygons_mm) <= 4。
    if not 1 <= len(polygons_mm) <= 4:
        # 抛出异常，通知上层处理：ValueError(f'通用拼图只支持1～4块，当前为{len(polygons_mm)}块')。
        raise ValueError(f"通用拼图只支持1～4块，当前为{len(polygons_mm)}块")
    # 计算并保存到 normalized（经过本函数规范化处理的数据）。
    normalized: list[np.ndarray] = []
    # 计算并保存到 min_edge（最小·边）。
    min_edge = float(config.get("generic_min_edge_length_mm", 0.0))
    # 计算并保存到 minimum_area（最小·面积）。
    minimum_area = float(config.get("generic_min_piece_area_mm2", 1.0))
    # 遍历数据，逐项处理：enumerate(polygons_mm)。
    for index, polygon in enumerate(polygons_mm):
        # 执行可能出错的代码，并由后面的异常分支处理错误。
        try:
            # 调用函数：normalized.append。
            normalized.append(normalize_polygon(
                polygon,
                # 传入命名参数：min_edge_length_mm=min_edge（最小·边·长度·毫米）；取值过程：min_edge（最小·边）。
                min_edge_length_mm=min_edge,
                # 传入命名参数：min_area_mm2=minimum_area（最小·面积·平方毫米）；取值过程：minimum_area（最小·面积）。
                min_area_mm2=minimum_area,
            ))
        # 捕获ValueError异常，转入下面的处理代码；异常对象保存在exc。
        except ValueError as exc:
            # 抛出异常，通知上层处理：ValueError(f'第{index + 1}块碎片无效：{exc}')。
            raise ValueError(f"第{index + 1}块碎片无效：{exc}") from exc

    # 计算并保存到 budget（所有搜索分支共享的时间和状态数量预算）。
    budget = _make_search_budget(config)
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 返回结果：调用 _solve_normalized_rectangle。
        return _solve_normalized_rectangle(normalized, config, budget)
    # 捕获_NoRectangleSolution异常，转入下面的处理代码；异常对象保存在original_error。
    except _NoRectangleSolution as original_error:
        # 判断条件；满足时执行下面代码：not bool(config.get('generic_enable_collinear_split', True))。
        if not bool(config.get("generic_enable_collinear_split", True)):
            # 抛出异常，通知上层处理：ValueError(str(original_error))。
            raise ValueError(str(original_error)) from original_error

        # 计算并保存到 dual_variants（双碎片·变体集合）。
        dual_variants: list[list[np.ndarray]] = []
        # 判断条件；满足时执行下面代码：bool(config.get('generic_enable_dual_collinear_split', True))。
        if bool(config.get("generic_enable_dual_collinear_split", True)):
            # 计算并保存到 dual_variants（双碎片·变体集合）。
            dual_variants = generate_dual_collinear_split_variants(normalized, config)
        # 判断条件；满足时执行下面代码：dual_variants。
        if dual_variants:
            # 调用函数：print。
            print(
                "[诊断] 原始边匹配失败，尝试双碎片共线长边拆分："
                f"候选={len(dual_variants)}"
            )
            # 计算并保存到 dual_solution（双碎片·solution）、dual_index（双碎片·索引）。
            dual_solution, dual_index = _solve_split_variants(
                dual_variants, config, budget
            )
            # 判断条件；满足时执行下面代码：dual_solution is not None。
            if dual_solution is not None:
                # 调用函数：_print_split_success。
                _print_split_success(
                    "双碎片共线长边拆分",
                    dual_solution,
                    dual_index,
                    len(dual_variants),
                )
                # 返回结果：dual_solution（双碎片·solution）。
                return dual_solution

        # 计算并保存到 single_variants（单项·变体集合）。
        single_variants = generate_collinear_split_variants(normalized, config)
        # 判断条件；满足时执行下面代码：single_variants and (not budget.exhausted())。
        if single_variants and not budget.exhausted():
            # 调用函数：print。
            print(
                "[诊断] 双拆未找到合格矩形，尝试单条共线长边拆分："
                f"候选={len(single_variants)}"
            )
            # 计算并保存到 single_solution（单项·solution）、single_index（单项·索引）。
            single_solution, single_index = _solve_split_variants(
                single_variants, config, budget
            )
            # 判断条件；满足时执行下面代码：single_solution is not None。
            if single_solution is not None:
                # 调用函数：_print_split_success。
                _print_split_success(
                    "单条共线长边拆分",
                    single_solution,
                    single_index,
                    len(single_variants),
                )
                # 返回结果：single_solution（单项·solution）。
                return single_solution

        # 判断条件；满足时执行下面代码：not dual_variants and (not single_variants)。
        if not dual_variants and not single_variants:
            # 抛出异常，通知上层处理：ValueError(str(original_error))。
            raise ValueError(str(original_error)) from original_error
        # 计算并保存到 budget_text（搜索预算·文本）。
        budget_text = (
            f"；搜索{budget.stop_reason}" if budget.stop_reason is not None else ""
        )
        # 终极兜底：所有正常拼合路径（原始边/双拆/单拆）均失败时，
        # 若开关开启，把每块碎片单独平铺到A4下半区（不旋转、保持原角度、
        # 互相不重叠），保证比赛"必须完成"。仅在彻底无解时触发，
        # 绝不覆盖任何可正常拼合的结果。
        # 判断条件；满足时执行下面代码：bool(config.get('generic_ultimate_fallback_enable', True))。
        if bool(config.get("generic_ultimate_fallback_enable", True)):
            # 计算并保存到 fallback_solution（兜底·solution）。
            fallback_solution = _build_fallback_placement(
                normalized, config, budget.states
            )
            # 判断条件；满足时执行下面代码：fallback_solution is not None。
            if fallback_solution is not None:
                # 计算并保存到 budget_suffix（搜索预算·suffix）。
                budget_suffix = budget_text.lstrip("；")
                # 计算并保存到 suffix。
                suffix = (f"（{budget_suffix}）" if budget_suffix else "")
                # 调用函数：print。
                print(
                    f"[终极兜底] 无法拼合成矩形{suffix}，"
                    f"改为将{len(normalized)}块碎片平铺到A4下半区"
                )
                # 返回结果：fallback_solution（兜底·solution）。
                return fallback_solution
        # 抛出异常，通知上层处理：ValueError(f'{original_error}；双拆回退{len(dual_variants)}种、单拆回退{len(sing…。
        raise ValueError(
            f"{original_error}；双拆回退{len(dual_variants)}种、"
            f"单拆回退{len(single_variants)}种仍未找到合格矩形{budget_text}"
        ) from original_error
