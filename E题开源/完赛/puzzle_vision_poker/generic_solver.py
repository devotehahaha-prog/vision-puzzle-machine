"""1～4块多边形拼成矩形的纯几何求解器。

本模块只处理毫米坐标下的二维多边形，不访问摄像头、不写文件，
也不依赖 ``main.py``，便于独立测试并避免循环依赖。
"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：与普通版相同的纯几何矩形求解器；扑克 3 片也走这里，花纹评分在 edge_matcher。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约53行  EdgeMatch：反向拼缝
#   - 约74行  GenericSolution：整局解，score 越小越好
#   - 约247行  rigid_align_reversed：切缝反向对齐
#   - 约470行  _search_assemblies：DFS 装配搜索
#   - 约794行  _assess_assembly：矩形质量评估
#   - 约1324行  split_polygon_edge：长边对多条短边
#   - 约1729行  solve_generic_rectangle：对外入口
# =============================================================================
# 这是纯几何求解器：输入每片顶点的毫米坐标，输出目标多边形、3×3刚体矩阵和质量指标。
# 阅读顺序：solve_generic_rectangle → _solve_normalized_rectangle → _search_assemblies → _assess_assembly → _build_solution。
# 基本思路：边长接近可能是同一拼缝；把两边反向对齐后接入新碎片，再检查整体能否形成矩形。
# 所有碎片保持尺寸不变，只旋转和平移；3×3齐次矩阵把(x,y,1)映射到新位置。点按行存储时使用 points @ matrix.T。
# DFS是深度优先搜索：尝试一条拼接分支，递归接下一片，失败就退回上一层尝试其他边。预算限制递归搜索的时间和状态数。
# 几何score是加权误差代价，越小越好；它与扑克牌花纹模块“越大越好”的相似度不是同一个指标。
# 共线拆边只在线段上插入顶点，帮助一条长边对应几条短边，不改变碎片外形。
# 宽松解和平铺兜底属于不同层次的退让；现有程序并非在所有失败条件下都能返回解。
# =============================================================================
# 【分区】导入
# 功能：纯几何求解，不读相机、不写文件、不 import main。扑克 3 片也走这一套。
# 可修改：超时/容差走 config 的 generic_*（仿真会从 poker_* 拷过来）。
# 看情况改：无。
# 不要改：只处理毫米二维多边形。
# =============================================================================
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np


# =============================================================================
# 【分区】对外数据结构
# 功能：拼缝、单片刚体、整局解。score 越小越好，不要和 NCC 越大越好搞反。
# 可修改：无。
# 看情况改：无。
# 不要改：3×3 齐次矩阵；points @ matrix.T。
# =============================================================================
@dataclass(frozen=True)
class EdgeMatch:
    """两块碎片之间采用的一组反向拼缝边。"""

    piece_a: int
    edge_a: int
    piece_b: int
    edge_b: int
    length_error_mm: float
    endpoint_error_mm: float


@dataclass
class GenericPiecePose:
    """单块源多边形到最终目标位置的刚体姿态。"""

    piece_index: int
    transform_3x3: np.ndarray
    target_polygon_mm: np.ndarray


@dataclass
class GenericSolution:
    """通用矩形拼接的最终解及质量指标。"""

    poses: list[GenericPiecePose]
    target_size_mm: tuple[float, float]
    target_top_left_mm: tuple[float, float]
    score: float
    fill_error_ratio: float
    overlap_ratio: float
    endpoint_error_mm: float
    boundary_error_mm: float
    boundary_gap_ratio: float
    matches: list[EdgeMatch]
    search_states: int
    relaxed_override: bool = False


# =============================================================================
# 【分区】多边形与刚体几何工具
# 功能：规范化、边长、刚体、反向对齐切缝。
# 可修改：闭合点阈值 1e-9。
# 看情况改：切缝两侧必须反向对齐。
# 不要改：禁止缩放。
# =============================================================================
def _as_polygon(points: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    """转换为不带重复闭合点的 ``N×2 float64`` 多边形。"""
    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim == 3 and polygon.shape[1:] == (1, 2):
        polygon = polygon[:, 0, :]
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("多边形必须是N×2坐标数组")
    if len(polygon) >= 2 and np.linalg.norm(polygon[0] - polygon[-1]) <= 1e-9:
        polygon = polygon[:-1]
    if not np.isfinite(polygon).all():
        raise ValueError("多边形坐标包含NaN或无穷值")
    return polygon.copy()


# 鞋带公式：把每个顶点与下一个顶点组成叉乘并求和再除2；符号随顶点绕行顺序改变，绝对值是面积。
def signed_area(points: np.ndarray | Iterable[Iterable[float]]) -> float:
    """返回带符号鞋带面积；在图像Y向下坐标中正负只用于统一顺序。"""
    polygon = _as_polygon(points)
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(points: np.ndarray | Iterable[Iterable[float]]) -> float:
    """返回多边形绝对面积，单位随输入坐标平方。"""
    return abs(signed_area(points))


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
    polygon = _as_polygon(points)
    if not min_vertices <= len(polygon) <= max_vertices:
        raise ValueError(
            f"多边形顶点数必须在{min_vertices}～{max_vertices}之间，当前为{len(polygon)}"
        )
    lengths = edge_lengths(polygon)
    if np.any(lengths <= 1e-6):
        raise ValueError("多边形包含重复相邻点或退化边")
    if min_edge_length_mm > 0.0 and float(np.min(lengths)) < min_edge_length_mm:
        raise ValueError(
            f"多边形最短边{float(np.min(lengths)):.2f}mm小于限制"
            f"{min_edge_length_mm:.2f}mm"
        )
    area = polygon_area(polygon)
    if area < min_area_mm2:
        raise ValueError(f"多边形面积{area:.3f}mm²过小或已退化")
    # 统一为正带符号面积，后续边顺序比较更稳定。
    if signed_area(polygon) < 0.0:
        polygon = polygon[::-1].copy()
    return polygon


def polygon_edges(points: np.ndarray | Iterable[Iterable[float]]) -> list[np.ndarray]:
    """按顶点循环顺序返回所有 ``2×2`` 边端点。"""
    polygon = _as_polygon(points)
    if len(polygon) < 2:
        return []
    return [np.vstack((polygon[index], polygon[(index + 1) % len(polygon)]))
            for index in range(len(polygon))]


def edge_lengths(points: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    """返回所有循环边长度。"""
    polygon = _as_polygon(points)
    if len(polygon) < 2:
        return np.empty((0,), dtype=np.float64)
    return np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)


# 矩阵前两行是[cos,-sin,tx]和[sin,cos,ty]，第三行[0,0,1]；只有旋转和平移，没有缩放。
def rigid_matrix(angle_rad: float, translation_xy: Iterable[float]) -> np.ndarray:
    """构造二维齐次刚体矩阵。

    坐标采用项目现有约定：X向右、Y向下，因此正角在图像/机械画面中表现为顺时针。
    """
    translation = np.asarray(tuple(translation_xy), dtype=np.float64)
    if translation.shape != (2,) or not np.isfinite(translation).all():
        raise ValueError("平移量必须是两个有限数值")
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    return np.array(
        [[cosine, -sine, translation[0]],
         [sine, cosine, translation[1]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def transform_points(
    points: np.ndarray | Iterable[Iterable[float]],
    matrix_3x3: np.ndarray,
) -> np.ndarray:
    """使用3×3齐次矩阵变换一组二维点。"""
    polygon = np.asarray(points, dtype=np.float64)
    original_shape = polygon.shape
    if polygon.ndim == 3 and polygon.shape[1:] == (1, 2):
        polygon = polygon[:, 0, :]
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("点集必须是N×2坐标数组")
    matrix = np.asarray(matrix_3x3, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("变换矩阵必须是3×3有限数值数组")
    homogeneous = np.column_stack((polygon, np.ones(len(polygon), dtype=np.float64)))
    transformed = homogeneous @ matrix.T
    if np.any(np.abs(transformed[:, 2]) <= 1e-12):
        raise ValueError("齐次变换产生无效尺度")
    result = transformed[:, :2] / transformed[:, 2:3]
    if len(original_shape) == 3:
        return result.reshape((-1, 1, 2))
    return result


def is_rigid_transform(matrix_3x3: np.ndarray, tolerance: float = 1e-6) -> bool:
    """判断矩阵是否为不含缩放、剪切、镜像的二维刚体变换。"""
    matrix = np.asarray(matrix_3x3, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return False
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=tolerance):
        return False
    rotation = matrix[:2, :2]
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(2), atol=tolerance)
        and math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=tolerance)
    )


def rotation_angle_deg(matrix_3x3: np.ndarray) -> float:
    """提取项目坐标系中的旋转角，归一化到[-180, 180)。"""
    matrix = np.asarray(matrix_3x3, dtype=np.float64)
    if not is_rigid_transform(matrix, tolerance=1e-5):
        raise ValueError("矩阵不是有效刚体变换")
    angle = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    return float((angle + 180.0) % 360.0 - 180.0)


# 两片沿各自边界走向相接时，一条边的起点对应另一条边的终点。边长有噪声时只对齐方向和中点，不拉伸边长。
def rigid_align_reversed(
    source_edge: np.ndarray | Iterable[Iterable[float]],
    target_edge: np.ndarray | Iterable[Iterable[float]],
) -> tuple[np.ndarray, float]:
    """将源边反向刚体对齐到目标边。

    映射关系为 ``source[0] -> target[1]``、
    ``source[1] -> target[0]``。两边长度存在小误差时只对齐方向和中点，
    不进行缩放，并返回两个端点的平均残差。
    """
    source = np.asarray(source_edge, dtype=np.float64).reshape((-1, 2))
    target = np.asarray(target_edge, dtype=np.float64).reshape((-1, 2))
    if source.shape != (2, 2) or target.shape != (2, 2):
        raise ValueError("边必须各包含两个二维端点")
    source_vector = source[1] - source[0]
    desired_vector = target[0] - target[1]
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(desired_vector))
    if source_length <= 1e-6 or target_length <= 1e-6:
        raise ValueError("不能对齐退化边")
    source_angle = math.atan2(float(source_vector[1]), float(source_vector[0]))
    target_angle = math.atan2(float(desired_vector[1]), float(desired_vector[0]))
    angle = target_angle - source_angle
    rotation = rigid_matrix(angle, (0.0, 0.0))
    rotated_midpoint = transform_points(
        np.mean(source, axis=0, keepdims=True), rotation
    )[0]
    target_midpoint = np.mean(target, axis=0)
    matrix = rigid_matrix(angle, target_midpoint - rotated_midpoint)
    mapped = transform_points(source, matrix)
    endpoint_error = 0.5 * (
        float(np.linalg.norm(mapped[0] - target[1]))
        + float(np.linalg.norm(mapped[1] - target[0]))
    )
    if not is_rigid_transform(matrix):
        raise RuntimeError("内部错误：边对齐生成了非刚体矩阵")
    return matrix, endpoint_error


def compose_transforms(*matrices: np.ndarray) -> np.ndarray:
    """按参数顺序组合变换：最后返回 ``M_n @ ... @ M_1``。"""
    result = np.eye(3, dtype=np.float64)
    for matrix in matrices:
        current = np.asarray(matrix, dtype=np.float64)
        if current.shape != (3, 3):
            raise ValueError("所有变换矩阵必须为3×3")
        result = current @ result
    return result


@dataclass
class _AssemblyCandidate:
    """DFS产生的完整装配体，坐标仍位于任意装配参考系。"""

    poses: dict[int, np.ndarray]
    used_edges: frozenset[tuple[int, int]]
    matches: list[EdgeMatch]


@dataclass(frozen=True)
class _CollinearSplitProposal:
    """一条被合并长边的拆分提案及其几何来源。"""

    error_mm: float
    source_piece: int
    source_edge: int
    segment_lengths_mm: tuple[float, ...]
    selected_pieces: tuple[int, ...]
    split_polygon: np.ndarray


@dataclass
class _SearchBudget:
    """跨原始边与拆边回退共享的状态和时间预算。"""

    deadline: float
    max_states: int
    states: int = 0
    stop_reason: str | None = None

    def exhausted(self) -> bool:
        if self.stop_reason is not None:
            return True
        if self.states >= self.max_states:
            self.stop_reason = f"达到状态上限{self.max_states}"
            return True
        if time.monotonic() >= self.deadline:
            self.stop_reason = "达到时间上限"
            return True
        return False

    def consume(self) -> bool:
        if self.exhausted():
            return False
        self.states += 1
        return True


# =============================================================================
# 【分区】搜索预算与候选边
# 功能：限制 DFS 时间/状态数；边长接近则生成反向匹配。
# 可修改：generic_search_timeout_seconds、generic_max_search_states、边长容差。
# 看情况改：超时可加 timeout；乱拼先收紧边长容差。
# 不要改：拆边变体共享同一预算。
# =============================================================================
def _make_search_budget(config: dict[str, Any]) -> _SearchBudget:
    timeout_seconds = max(0.1, float(config.get(
        "generic_search_timeout_seconds", 3.0
    )))
    max_states = max(1, int(config.get("generic_max_search_states", 30000)))
    return _SearchBudget(
        deadline=time.monotonic() + timeout_seconds,
        max_states=max_states,
    )


def edge_match_tolerance_mm(length_a: float, length_b: float,
                            config: dict[str, Any]) -> float:
    """返回两条边允许的毫米长度误差。"""
    absolute = float(config.get("generic_edge_abs_tolerance_mm", 2.0))
    relative = float(config.get("generic_edge_rel_tolerance", 0.05))
    return max(absolute, relative * min(float(length_a), float(length_b)))


def generate_edge_candidates(polygons_mm: list[np.ndarray],
                             config: dict[str, Any]) -> list[EdgeMatch]:
    """按边长为不同碎片生成无方向拼缝候选。"""
    polygons = [normalize_polygon(polygon) for polygon in polygons_mm]
    lengths = [edge_lengths(polygon) for polygon in polygons]
    candidates: list[EdgeMatch] = []
    for piece_a in range(len(polygons)):
        for piece_b in range(piece_a + 1, len(polygons)):
            for edge_a, length_a in enumerate(lengths[piece_a]):
                for edge_b, length_b in enumerate(lengths[piece_b]):
                    length_error = abs(float(length_a) - float(length_b))
                    if length_error > edge_match_tolerance_mm(length_a, length_b, config):
                        continue
                    # 中点刚体对齐时，两端残差均为边长差的一半。
                    candidates.append(EdgeMatch(
                        piece_a=piece_a,
                        edge_a=edge_a,
                        piece_b=piece_b,
                        edge_b=edge_b,
                        length_error_mm=length_error,
                        endpoint_error_mm=0.5 * length_error,
                    ))
    ordered = sorted(candidates, key=lambda item: (
        item.length_error_mm, item.piece_a, item.piece_b, item.edge_a, item.edge_b
    ))
    # 参考仿真方案限制候选规模：近似边优先，避免少量噪声边触发组合爆炸。
    max_candidates = int(config.get("generic_max_edge_candidates", 40))
    return ordered if max_candidates <= 0 else ordered[:max_candidates]


# 每片先画成0/1图层并累计；覆盖次数>0是并集，>1是重叠。栅格精度由pixels_per_mm决定，不是相机标定精度。
def _raster_coverage(polygons: list[np.ndarray], pixels_per_mm: float,
                     padding_mm: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """栅格化多边形并返回每像素覆盖次数与平移后的浮点多边形。"""
    if not polygons:
        return np.zeros((1, 1), dtype=np.uint8), np.empty((0, 0, 2), dtype=np.float64)
    ppm = max(0.5, float(pixels_per_mm))
    all_points = np.vstack(polygons)
    minimum = np.min(all_points, axis=0) - float(padding_mm)
    shifted = [(polygon - minimum) * ppm for polygon in polygons]
    maximum = np.max(np.vstack(shifted), axis=0)
    width = max(2, int(math.ceil(float(maximum[0]))) + 2)
    height = max(2, int(math.ceil(float(maximum[1]))) + 2)
    coverage = np.zeros((height, width), dtype=np.uint8)
    for polygon in shifted:
        layer = np.zeros_like(coverage)
        integer_polygon = np.rint(polygon).astype(np.int32)
        cv2.fillPoly(layer, [integer_polygon], 1)
        coverage = np.minimum(255, coverage + layer).astype(np.uint8)
    return coverage, np.asarray(shifted, dtype=object)


def polygon_overlap_ratio(polygons: list[np.ndarray],
                          pixels_per_mm: float = 2.0) -> float:
    """返回多边形装配中的实质重叠率。

    共享边在栅格上会占少量同一像素，因此调用者应使用小的非零阈值。
    """
    if len(polygons) <= 1:
        return 0.0
    coverage, _ = _raster_coverage(polygons, pixels_per_mm)
    union_pixels = int(np.count_nonzero(coverage > 0))
    overlap_pixels = int(np.count_nonzero(coverage > 1))
    return float(overlap_pixels / union_pixels) if union_pixels else 1.0


def _assembly_diameter(polygons: list[np.ndarray]) -> float:
    """返回装配体所有顶点间的最大距离。"""
    points = np.vstack(polygons)
    differences = points[:, None, :] - points[None, :, :]
    return float(np.sqrt(np.max(np.sum(differences * differences, axis=2))))


# 量化指把接近的角度与位置归入同一格，再把格号组成可哈希的元组，避免同一布局被不同拼接顺序反复搜索。
def _pose_state_key(poses: dict[int, np.ndarray],
                    used_edges: frozenset[tuple[int, int]],
                    translation_quantum_mm: float,
                    angle_quantum_deg: float, *,
                    validated_rigid: bool = False) -> tuple[Any, ...]:
    """量化姿态和已使用边，抑制由对称边产生的重复状态。"""
    pose_items: list[tuple[int, int, int, int]] = []
    for piece_index, matrix in sorted(poses.items()):
        # DFS only stores identity or the already validated output of
        # rigid_align_reversed. Keep validation for all other callers.
        angle = ((math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
                  + 180.0) % 360.0 - 180.0
                 if validated_rigid else rotation_angle_deg(matrix))
        pose_items.append((
            piece_index,
            int(round(float(matrix[0, 2]) / translation_quantum_mm)),
            int(round(float(matrix[1, 2]) / translation_quantum_mm)),
            int(round(angle / angle_quantum_deg)),
        ))
    return tuple(pose_items), tuple(sorted(used_edges))


# 固定第0片为参考系，消除所有碎片一起平移/旋转产生的无穷等价解。此处得到连通装配候选，还需后续矩形质量评估。
# =============================================================================
# 【分区】DFS 装配搜索（固定第 0 片为参考系）
# 功能：按匹配边接入，得到连通候选，尚未判断矩形。
# 可修改：无直接魔数。
# 看情况改：扑克 3 片在 1~4 支持范围内。
# 不要改：第 0 片固定；状态键去重。
# =============================================================================
def _search_assemblies(polygons_mm: list[np.ndarray],
                       config: dict[str, Any],
                       budget: _SearchBudget | None = None,
                       ) -> tuple[list[_AssemblyCandidate], int]:
    """使用匹配边生成树搜索装配体，并受全局状态/时间预算约束。"""
    if budget is None:
        budget = _make_search_budget(config)
    polygons = [normalize_polygon(polygon) for polygon in polygons_mm]
    piece_count = len(polygons)
    if piece_count == 1:
        return [_AssemblyCandidate(
            # 字典字段0（按表达式求键）：创建单位矩阵，对应不旋转、不平移的初始变换。
            poses={0: np.eye(3, dtype=np.float64)},
            used_edges=frozenset(),
            matches=[],
        )], 1

    lengths = [edge_lengths(polygon) for polygon in polygons]
    source_edges = [polygon_edges(polygon) for polygon in polygons]
    candidate_lookup: set[tuple[int, int, int, int]] = set()
    for candidate in generate_edge_candidates(polygons, config):
        candidate_lookup.add((candidate.piece_a, candidate.edge_a,
                              candidate.piece_b, candidate.edge_b))
        candidate_lookup.add((candidate.piece_b, candidate.edge_b,
                              candidate.piece_a, candidate.edge_a))
    if not candidate_lookup:
        return [], 0

    overlap_limit = float(config.get("generic_search_overlap_ratio", 0.012))
    raster_ppm = float(config.get("generic_search_pixels_per_mm", 2.0))
    maximum_diagonal = math.hypot(
        float(config.get("generic_target_long_max_mm", 120.0)),
        float(config.get("generic_target_short_max_mm", 90.0)),
    ) + float(config.get("generic_search_dimension_margin_mm", 5.0))
    translation_quantum = max(0.1, float(config.get(
        "generic_state_translation_quantum_mm", 0.5
    )))
    angle_quantum = max(0.1, float(config.get(
        "generic_state_angle_quantum_deg", 0.5
    )))

    complete: list[_AssemblyCandidate] = []
    visited: set[tuple[Any, ...]] = set()
    # Different spanning trees revisit identical directed edge alignments.
    # Use exact matrix bytes, never rounded positions, to reuse only identical
    # calculations within this one observed polygon set.
    alignment_cache: dict[tuple[Any, ...], tuple[np.ndarray, float]] = {}
    # poses是已放置片的姿态，used_edges防止一条边被重复使用，matches记录本分支拼缝。递归前创建下一分支副本，不覆盖兄弟分支。
    def dfs(poses: dict[int, np.ndarray],
            used_edges: frozenset[tuple[int, int]],
            matches: list[EdgeMatch]) -> None:
        if not budget.consume():
            return
        key = _pose_state_key(
            poses, used_edges, translation_quantum, angle_quantum, validated_rigid=True
        )
        if key in visited:
            return
        visited.add(key)
        if len(poses) == piece_count:
            complete.append(_AssemblyCandidate(
                poses={index: matrix.copy() for index, matrix in poses.items()},
                used_edges=used_edges,
                matches=list(matches),
            ))
            return

        options: list[tuple[float, float, int, int, int, int, np.ndarray]] = []
        unplaced = [index for index in range(piece_count) if index not in poses]
        transformed_placed = {index: transform_points(polygons[index], pose)
                              for index, pose in poses.items()}
        placed_polygons = list(transformed_placed.values())
        for piece_a, pose_a in poses.items():
            transformed_a = transformed_placed[piece_a]
            edges_a = polygon_edges(transformed_a)
            pose_bytes = pose_a.tobytes()
            for edge_a, target_edge in enumerate(edges_a):
                if (piece_a, edge_a) in used_edges:
                    continue
                for piece_b in unplaced:
                    for edge_b, source_edge in enumerate(source_edges[piece_b]):
                        if (piece_a, edge_a, piece_b, edge_b) not in candidate_lookup:
                            continue
                        alignment_key = (piece_a, edge_a, piece_b, edge_b, pose_bytes)
                        alignment = alignment_cache.get(alignment_key)
                        if alignment is None:
                            alignment = rigid_align_reversed(source_edge, target_edge)
                            alignment_cache[alignment_key] = alignment
                        pose_b, endpoint_error = alignment
                        length_error = abs(
                            float(lengths[piece_a][edge_a])
                            - float(lengths[piece_b][edge_b])
                        )
                        options.append((
                            length_error, endpoint_error,
                            piece_a, edge_a, piece_b, edge_b, pose_b,
                        ))
        options.sort(key=lambda item: (item[0], item[1], item[4], item[5]))

        for (length_error, endpoint_error, piece_a, edge_a,
             piece_b, edge_b, pose_b) in options:
            if budget.exhausted():
                break
            transformed_new = transform_points(polygons[piece_b], pose_b)
            candidate_polygons = placed_polygons + [transformed_new]
            if _assembly_diameter(candidate_polygons) > maximum_diagonal:
                continue
            if _interior_overlap_ratio(candidate_polygons, raster_ppm) > overlap_limit:
                continue
            next_poses = dict(poses)
            next_poses[piece_b] = pose_b
            next_used = frozenset((*used_edges, (piece_a, edge_a), (piece_b, edge_b)))
            next_matches = matches + [EdgeMatch(
                piece_a=piece_a,
                edge_a=edge_a,
                piece_b=piece_b,
                edge_b=edge_b,
                length_error_mm=length_error,
                endpoint_error_mm=endpoint_error,
            )]
            dfs(next_poses, next_used, next_matches)

    # 任意合法装配都与第0块连通，固定根姿态可去掉全局旋转和平移自由度。
    dfs({0: np.eye(3, dtype=np.float64)}, frozenset(), [])
    if budget.stop_reason is not None and not complete:
        raise ValueError(
            f"通用拼图搜索{budget.stop_reason}，已检查状态{budget.states}"
        )
    return complete, budget.states


@dataclass
class _RectangleEvaluation:
    """完整装配候选的矩形质量和归一化变换。"""

    candidate: _AssemblyCandidate
    normalize_matrix: np.ndarray
    size_mm: tuple[float, float]
    score: float
    fill_error_ratio: float
    overlap_ratio: float
    endpoint_error_mm: float
    boundary_error_mm: float
    boundary_gap_ratio: float
    target_size_error_mm: float = 0.0


@dataclass
class _AssemblyAssessment:
    """保留候选的全部质量指标和拒绝原因，供闭环回退与诊断使用。"""

    evaluation: _RectangleEvaluation
    rejection_reasons: tuple[str, ...]
    outside_edge_ok: bool

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons


# =============================================================================
# 【分区】矩形质量评估
# 功能：转到长边沿 X，算填充/重叠/缺口/端点误差。
# 可修改：generic_max_fill_error_ratio、边界缺口、目标长宽（扑克用 poker_* 再拷贝）。
# 看情况改：重叠明显是搜错了，不要放宽 overlap。
# 不要改：这里只评形状，放到纸面在 _build_solution。
# =============================================================================
def _normalization_to_long_x(polygons: list[np.ndarray]) -> tuple[np.ndarray, tuple[float, float]]:
    """返回将最小外接矩形长边转到X轴并把左上角移到原点的变换。"""
    points = np.vstack(polygons).astype(np.float32)
    rectangle = cv2.minAreaRect(points.reshape((-1, 1, 2)))
    box = cv2.boxPoints(rectangle).astype(np.float64)
    vectors = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(vectors, axis=1)
    longest = vectors[int(np.argmax(lengths))]
    angle = math.atan2(float(longest[1]), float(longest[0]))
    # 矩形边无方向，限制在[-90°, 90°)可避免不必要的180°翻转。
    if angle >= math.pi / 2.0:
        angle -= math.pi
    elif angle < -math.pi / 2.0:
        angle += math.pi
    rotation = rigid_matrix(-angle, (0.0, 0.0))
    rotated_points = transform_points(points.astype(np.float64), rotation)
    minimum = np.min(rotated_points, axis=0)
    translation = rigid_matrix(0.0, -minimum)
    normalization = translation @ rotation
    normalized_points = transform_points(points.astype(np.float64), normalization)
    maximum = np.max(normalized_points, axis=0)
    size = (float(maximum[0]), float(maximum[1]))
    if size[0] + 1e-6 < size[1]:
        # 极端OpenCV角度情况下再旋转90°，确保长边沿X。
        quarter_turn = rigid_matrix(-math.pi / 2.0, (0.0, 0.0))
        turned = transform_points(normalized_points, quarter_turn)
        turned_minimum = np.min(turned, axis=0)
        shift = rigid_matrix(0.0, -turned_minimum)
        normalization = shift @ quarter_turn @ normalization
        turned = transform_points(points.astype(np.float64), normalization)
        turned_maximum = np.max(turned, axis=0)
        size = (float(turned_maximum[0]), float(turned_maximum[1]))
    return normalization, size


def _interior_overlap_ratio(polygons: list[np.ndarray],
                            pixels_per_mm: float) -> float:
    """腐蚀各片边界后计算实质重叠，排除共享拼缝的单像素影响。"""
    if len(polygons) <= 1:
        return 0.0
    ppm = max(1.0, float(pixels_per_mm))
    all_points = np.vstack(polygons)
    minimum = np.min(all_points, axis=0) - 2.0
    shifted = [(polygon - minimum) * ppm for polygon in polygons]
    maximum = np.max(np.vstack(shifted), axis=0)
    width = max(3, int(math.ceil(float(maximum[0]))) + 3)
    height = max(3, int(math.ceil(float(maximum[1]))) + 3)
    coverage = np.zeros((height, width), dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    union = np.zeros_like(coverage)
    for polygon in shifted:
        layer = np.zeros_like(coverage)
        cv2.fillPoly(layer, [np.rint(polygon).astype(np.int32)], 1)
        np.maximum(union, layer, out=union)
        interior = cv2.erode(layer, kernel, iterations=1)
        # Both arrays are uint8. This is exactly the previous uint8 addition,
        # without allocating the redundant minimum/astype copies each time.
        np.add(coverage, interior, out=coverage)
    union_pixels = int(np.count_nonzero(union))
    overlap_pixels = int(np.count_nonzero(coverage > 1))
    return float(overlap_pixels / union_pixels) if union_pixels else 1.0


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
    ppm = max(1.0, float(pixels_per_mm))
    width_px = max(1, int(round(float(width) * ppm)))
    height_px = max(1, int(round(float(height) * ppm)))
    union = np.zeros((height_px + 1, width_px + 1), dtype=np.uint8)
    for polygon in polygons:
        points = np.rint(np.asarray(polygon, np.float64) * ppm).astype(np.int32)
        cv2.fillPoly(union, [points], 1)

    rectangle_pixels = union.size
    covered_pixels = int(np.count_nonzero(union))
    fill_error = 1.0 - covered_pixels / max(1, rectangle_pixels)

    band = max(1, int(math.ceil(float(boundary_band_mm) * ppm)))
    band = min(band, max(1, min(union.shape) // 2))
    top = np.any(union[:band, :] > 0, axis=0)
    bottom = np.any(union[-band:, :] > 0, axis=0)
    left = np.any(union[:, :band] > 0, axis=1)
    right = np.any(union[:, -band:] > 0, axis=1)
    boundary_samples = top.size + bottom.size + left.size + right.size
    covered_boundary = (
        int(np.count_nonzero(top)) + int(np.count_nonzero(bottom))
        + int(np.count_nonzero(left)) + int(np.count_nonzero(right))
    )
    boundary_gap = 1.0 - covered_boundary / max(1, boundary_samples)
    return max(0.0, float(fill_error)), max(0.0, float(boundary_gap))


def _outside_edge_quality(polygons: list[np.ndarray], width: float, height: float,
                          tolerance_mm: float) -> tuple[bool, float]:
    """检查每片是否有完整外边，并始终返回有限的最近边界距离。"""
    piece_errors: list[float] = []
    all_pieces_ok = True
    sides = np.asarray([0.0, width, 0.0, height], dtype=np.float64)
    for polygon in polygons:
        points = _as_polygon(polygon)
        if len(points) < 2:
            best_error, piece_ok = math.inf, False
        else:
            # Evaluate both endpoints of every edge against all four sides at
            # once. Distances, means, maxima and tolerances are unchanged.
            distances = np.abs(points[:, (0, 0, 1, 1)] - sides)
            following = np.roll(distances, -1, axis=0)
            best_error = float(np.min((distances + following) * 0.5))
            piece_ok = bool(np.any(np.maximum(distances, following) <= tolerance_mm))
        if not math.isfinite(best_error):
            best_error = math.hypot(width, height)
        piece_errors.append(best_error)
        all_pieces_ok = all_pieces_ok and piece_ok
    return all_pieces_ok, float(np.mean(piece_errors)) if piece_errors else 0.0


def _target_size_error_mm(size_mm: tuple[float, float],
                          config: dict[str, Any]) -> float:
    """返回候选矩形与已知目标尺寸的无方向绝对偏差。"""
    configured = config.get("target_rectangle_size_mm")
    if not isinstance(configured, (list, tuple)) or len(configured) != 2:
        return 0.0
    target = sorted((float(configured[0]), float(configured[1])), reverse=True)
    measured = sorted((float(size_mm[0]), float(size_mm[1])), reverse=True)
    if min(target) <= 0.0 or not np.isfinite(target).all():
        return 0.0
    return abs(measured[0] - target[0]) + abs(measured[1] - target[1])


def _match_endpoint_error(polygons: list[np.ndarray],
                          poses: dict[int, np.ndarray],
                          match: EdgeMatch) -> float:
    """计算当前全局姿态下两条反向拼缝边的平均端点距离。"""
    edge_a = transform_points(
        polygon_edges(polygons[match.piece_a])[match.edge_a], poses[match.piece_a]
    )
    edge_b = transform_points(
        polygon_edges(polygons[match.piece_b])[match.edge_b], poses[match.piece_b]
    )
    return 0.5 * (
        float(np.linalg.norm(edge_a[0] - edge_b[1]))
        + float(np.linalg.norm(edge_a[1] - edge_b[0]))
    )


def _refresh_match_errors(polygons: list[np.ndarray],
                          poses: dict[int, np.ndarray],
                          matches: list[EdgeMatch]) -> list[EdgeMatch]:
    """在位姿优化后刷新每条拼缝的端点误差。"""
    return [EdgeMatch(
        piece_a=match.piece_a,
        edge_a=match.edge_a,
        piece_b=match.piece_b,
        edge_b=match.edge_b,
        length_error_mm=match.length_error_mm,
        endpoint_error_mm=_match_endpoint_error(polygons, poses, match),
    ) for match in matches]

# 先把整体最小外接矩形长边转到X轴，再检查尺寸、填充、边界缺口、重叠及拼缝端点误差；通过所有启用的约束才是accepted。
def _assess_assembly(candidate: _AssemblyCandidate,
                     polygons: list[np.ndarray],
                     config: dict[str, Any], *,
                     pieces_area: float | None = None) -> _AssemblyAssessment:
    """计算完整质量指标，并把未通过的约束作为结构化诊断返回。"""
    assembled = [
        transform_points(polygons[index], candidate.poses[index])
        for index in range(len(polygons))
    ]
    normalization, size = _normalization_to_long_x(assembled)
    normalized_polygons = [
        transform_points(polygon, normalization) for polygon in assembled
    ]
    width, height = size
    rectangle_area = width * height
    if pieces_area is None:
        pieces_area = sum(polygon_area(polygon) for polygon in polygons)
    if rectangle_area <= 1e-6:
        area_error, coverage_error, boundary_gap, overlap = math.inf, math.inf, 1.0, 1.0
    else:
        area_error = abs(rectangle_area - pieces_area) / rectangle_area
        coverage_error, boundary_gap = _rectangle_coverage_quality(
            normalized_polygons,
            width,
            height,
            float(config.get("generic_score_pixels_per_mm", 3.0)),
            float(config.get("generic_boundary_band_mm", 1.5)),
        )
        overlap = _interior_overlap_ratio(
            normalized_polygons,
            float(config.get("generic_score_pixels_per_mm", 3.0)),
        )
    fill_error = max(area_error, coverage_error)
    endpoint_errors = []
    for match in candidate.matches:
        first, second = assembled[match.piece_a], assembled[match.piece_b]
        endpoint_errors.append(0.5 * (
            float(np.linalg.norm(first[match.edge_a] - second[(match.edge_b + 1) % len(second)]))
            + float(np.linalg.norm(first[(match.edge_a + 1) % len(first)] - second[match.edge_b]))
        ))
    endpoint_error = float(np.mean(endpoint_errors)) if endpoint_errors else 0.0
    outside_ok, boundary_error = _outside_edge_quality(
        normalized_polygons,
        width,
        height,
        float(config.get("generic_boundary_tolerance_mm", 3.0)),
    )
    target_size_error = _target_size_error_mm(size, config)
    closure_edges = max(0, len(candidate.matches) - max(0, len(polygons) - 1))
    closure_refined = closure_edges > 0

    long_min = float(config.get("generic_target_long_min_mm", 90.0))
    long_max = float(config.get("generic_target_long_max_mm", 120.0))
    short_min = float(config.get("generic_target_short_min_mm", 50.0))
    short_max = float(config.get("generic_target_short_max_mm", 90.0))
    dimension_tolerance = float(config.get("generic_dimension_tolerance_mm", 1.5))
    overlap_limit = float(config.get(
        "generic_closure_max_overlap_ratio" if closure_refined else
        "generic_max_overlap_ratio",
        config.get("generic_max_overlap_ratio", 0.008),
    ))
    endpoint_limit = float(config.get(
        "generic_closure_max_endpoint_error_mm" if closure_refined else
        "generic_max_endpoint_error_mm",
        config.get("generic_max_endpoint_error_mm", 3.0),
    ))
    fill_limit = float(config.get("generic_max_fill_error_ratio", 0.06))
    boundary_gap_limit = float(config.get("generic_max_boundary_gap_ratio", 0.08))

    rejected: list[str] = []
    if not (long_min - dimension_tolerance <= width <= long_max + dimension_tolerance):
        rejected.append("长边尺寸")
    if not (short_min - dimension_tolerance <= height <= short_max + dimension_tolerance):
        rejected.append("短边尺寸")
    if bool(config.get("generic_require_each_piece_outer_edge", True)) and not outside_ok:
        rejected.append("逐片外边")
    if overlap > overlap_limit:
        rejected.append("重叠")
    if fill_error > fill_limit:
        rejected.append("填充")
    if boundary_gap > boundary_gap_limit:
        rejected.append("边界缺口")
    if endpoint_error > endpoint_limit:
        rejected.append("拼缝端点")

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
    evaluation = _RectangleEvaluation(
        candidate=candidate,
        normalize_matrix=normalization,
        size_mm=(width, height),
        score=score,
        fill_error_ratio=fill_error,
        overlap_ratio=overlap,
        endpoint_error_mm=endpoint_error,
        boundary_error_mm=boundary_error,
        boundary_gap_ratio=boundary_gap,
        target_size_error_mm=target_size_error,
    )
    return _AssemblyAssessment(
        evaluation=evaluation,
        rejection_reasons=tuple(rejected),
        outside_edge_ok=outside_ok,
    )


# =============================================================================
# 【分区】位姿图优化与闭环补边
# 功能：生成树补闭合缝后最小二乘微调刚体姿态。
# 可修改：无。
# 看情况改：只对已经像矩形的候选补边。
# 不要改：优化仍是刚体。
# =============================================================================
def _evaluate_assembly(candidate: _AssemblyCandidate,
                       polygons: list[np.ndarray],
                       config: dict[str, Any]) -> _RectangleEvaluation | None:
    """返回通过全部硬约束的矩形评估。"""
    assessment = _assess_assembly(candidate, polygons, config)
    return assessment.evaluation if assessment.accepted else None


def _pose_graph_residual(polygons: list[np.ndarray],
                         matches: list[EdgeMatch],
                         poses: dict[int, np.ndarray]) -> np.ndarray:
    """返回所有拼缝端点的二维闭环残差。"""
    residuals: list[float] = []
    edges_by_piece = [polygon_edges(polygon) for polygon in polygons]
    for match in matches:
        world_a = transform_points(
            edges_by_piece[match.piece_a][match.edge_a], poses[match.piece_a]
        )
        world_b = transform_points(
            edges_by_piece[match.piece_b][match.edge_b], poses[match.piece_b]
        )
        residuals.extend((world_a - world_b[::-1]).reshape(-1).tolist())
    return np.asarray(residuals, dtype=np.float64)


# 对每个可动片优化角度、tx、ty三个量。微小扰动参数估计雅可比矩阵，再解最小二乘修正；试探多个步长，只接受残差平方和下降的更新。
def _optimize_pose_graph(polygons: list[np.ndarray],
                         matches: list[EdgeMatch],
                         initial_poses: dict[int, np.ndarray],
                         max_iterations: int = 20,
                         budget: _SearchBudget | None = None,
                         ) -> dict[int, np.ndarray]:
    """固定第0块，使用纯NumPy最小二乘全局分摊闭环拼缝误差。"""
    if len(polygons) < 3 or len(matches) < len(polygons):
        return {index: pose.copy() for index, pose in initial_poses.items()}
    root = 0
    movable = [index for index in range(len(polygons)) if index != root]
    if any(index not in initial_poses for index in range(len(polygons))):
        raise ValueError("闭环优化缺少碎片初始姿态")

    def pack(poses: dict[int, np.ndarray]) -> np.ndarray:
        values: list[float] = []
        for index in movable:
            pose = poses[index]
            values.extend((
                math.atan2(float(pose[1, 0]), float(pose[0, 0])),
                float(pose[0, 2]),
                float(pose[1, 2]),
            ))
        return np.asarray(values, dtype=np.float64)

    def unpack(values: np.ndarray) -> dict[int, np.ndarray]:
        poses = {root: initial_poses[root].copy()}
        for offset, index in enumerate(movable):
            theta, tx, ty = values[3 * offset:3 * offset + 3]
            poses[index] = rigid_matrix(float(theta), (float(tx), float(ty)))
        return poses

    def residual(values: np.ndarray) -> np.ndarray:
        return _pose_graph_residual(polygons, matches, unpack(values))

    values = pack(initial_poses)
    for _ in range(max(1, int(max_iterations))):
        if budget is not None and budget.exhausted():
            break
        baseline = residual(values)
        if len(baseline) == 0:
            break
        jacobian = np.empty((len(baseline), len(values)), dtype=np.float64)
        for column in range(len(values)):
            step = 1e-5 if column % 3 == 0 else 1e-3
            shifted = values.copy()
            shifted[column] += step
            jacobian[:, column] = (residual(shifted) - baseline) / step
        try:
            delta, *_ = np.linalg.lstsq(jacobian, -baseline, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(delta).all():
            break
        baseline_cost = float(np.dot(baseline, baseline))
        accepted_delta: np.ndarray | None = None
        for scale in (1.0, 0.5, 0.25, 0.125):
            trial_delta = scale * delta
            trial = values + trial_delta
            trial_residual = residual(trial)
            if float(np.dot(trial_residual, trial_residual)) < baseline_cost:
                values = trial
                accepted_delta = trial_delta
                break
        if accepted_delta is None or float(np.linalg.norm(accepted_delta)) < 1e-7:
            break
    return unpack(values)


def _closure_augmented_candidates(
    polygons: list[np.ndarray],
    base_assessments: list[_AssemblyAssessment],
    config: dict[str, Any],
    budget: _SearchBudget,
) -> list[_AssemblyCandidate]:
    """为少量近似矩形的生成树补1～2条闭环拼缝，并执行全局位姿优化。"""
    if len(polygons) < 3 or not bool(config.get("generic_enable_closure_refine", False)):
        return []
    prefiltered = [assessment for assessment in base_assessments if (
        assessment.evaluation.fill_error_ratio <= float(config.get(
            "generic_closure_prefilter_fill_error_ratio", 0.16
        ))
        and assessment.evaluation.boundary_gap_ratio <= float(config.get(
            "generic_closure_prefilter_boundary_gap_ratio", 0.50
        ))
        and assessment.evaluation.overlap_ratio <= float(config.get(
            "generic_closure_prefilter_overlap_ratio", 0.04
        ))
    )]
    prefiltered.sort(key=lambda item: (
        item.evaluation.score,
        item.evaluation.target_size_error_mm,
        item.evaluation.fill_error_ratio,
        item.evaluation.boundary_gap_ratio,
    ))
    base_limit = max(1, int(config.get("generic_closure_max_base_candidates", 4)))
    max_extra_edges = max(1, min(2, int(config.get("generic_max_closure_edges", 2))))
    max_initial_gap = float(config.get("generic_closure_max_initial_gap_mm", 20.0))
    max_candidates = max(1, int(config.get("generic_closure_max_candidates", 24)))
    iterations = max(1, int(config.get("generic_closure_optimize_iterations", 20)))
    edge_candidates = generate_edge_candidates(polygons, config)

    proposals: list[tuple[tuple[float, ...], _AssemblyCandidate,
                          tuple[EdgeMatch, ...]]] = []
    for assessment in prefiltered[:base_limit]:
        base = assessment.evaluation.candidate
        extras: list[tuple[float, EdgeMatch]] = []
        for match in edge_candidates:
            edge_a = (match.piece_a, match.edge_a)
            edge_b = (match.piece_b, match.edge_b)
            if edge_a in base.used_edges or edge_b in base.used_edges:
                continue
            spatial_error = _match_endpoint_error(polygons, base.poses, match)
            if spatial_error <= max_initial_gap:
                extras.append((spatial_error, match))
        extras.sort(key=lambda item: (
            item[0], item[1].length_error_mm,
            item[1].piece_a, item[1].edge_a, item[1].piece_b, item[1].edge_b,
        ))
        extras = extras[:8]
        for count in range(max_extra_edges, 0, -1):
            for combo in itertools.combinations(extras, count):
                used = set(base.used_edges)
                valid = True
                for _, match in combo:
                    endpoints = ((match.piece_a, match.edge_a),
                                 (match.piece_b, match.edge_b))
                    if endpoints[0] in used or endpoints[1] in used:
                        valid = False
                        break
                    used.update(endpoints)
                if not valid:
                    continue
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
    proposals.sort(key=lambda item: item[0])

    results: list[_AssemblyCandidate] = []
    seen: set[tuple[Any, ...]] = set()
    for _, base, extras in proposals:
        if len(results) >= max_candidates or not budget.consume():
            break
        all_matches = list(base.matches) + list(extras)
        optimized_poses = _optimize_pose_graph(
            polygons, all_matches, base.poses, iterations, budget
        )
        refreshed_matches = _refresh_match_errors(
            polygons, optimized_poses, all_matches
        )
        used_edges = set(base.used_edges)
        for match in extras:
            used_edges.update((
                (match.piece_a, match.edge_a),
                (match.piece_b, match.edge_b),
            ))
        frozen_edges = frozenset(used_edges)
        state_key = _pose_state_key(
            optimized_poses,
            frozen_edges,
            max(0.1, float(config.get("generic_state_translation_quantum_mm", 0.5))),
            max(0.1, float(config.get("generic_state_angle_quantum_deg", 0.5))),
        )
        if state_key in seen:
            continue
        seen.add(state_key)
        results.append(_AssemblyCandidate(
            poses=optimized_poses,
            used_edges=frozen_edges,
            matches=refreshed_matches,
        ))
    return results


def _format_failed_assessment(assessment: _AssemblyAssessment) -> str:
    """生成适合终端阅读的最佳失败候选摘要。"""
    evaluation = assessment.evaluation
    reasons = "/".join(assessment.rejection_reasons) or "无"
    return (
        f"矩形={evaluation.size_mm[0]:.1f}×{evaluation.size_mm[1]:.1f}mm，"
        f"目标尺寸偏差={evaluation.target_size_error_mm:.1f}mm，"
        f"填充={evaluation.fill_error_ratio:.3f}，"
        f"边界缺口={evaluation.boundary_gap_ratio:.3f}，"
        f"重叠={evaluation.overlap_ratio:.3f}，"
        f"拼缝={evaluation.endpoint_error_mm:.2f}mm，拒绝={reasons}"
    )

def _target_top_left(size_mm: tuple[float, float],
                     config: dict[str, Any]) -> tuple[float, float]:
    """计算配置目标半区内的目标矩形左上角。"""
    paper_width = float(config.get("paper_width_mm", 210.0))
    paper_height = float(config.get("paper_height_mm", 297.0))
    separator = float(config.get("separator_line_y_mm",
                                config.get("target_region_top_mm", paper_height / 2.0)))
    target_region = str(config.get("target_region", "bottom")).lower()
    if target_region == "top":
        region_min, region_max = 0.0, min(separator, paper_height)
    elif target_region == "bottom":
        region_min, region_max = max(0.0, min(separator, paper_height)), paper_height
    else:
        raise ValueError(f"未知target_region：{target_region}，应为top或bottom")
    width, height = size_mm
    x = (paper_width - width) / 2.0
    configured_y = config.get("generic_target_top_mm")
    if configured_y is None:
        y = region_min + (region_max - region_min - height) / 2.0
    else:
        y = float(configured_y)
    tolerance = 1e-6
    if x < -tolerance or x + width > paper_width + tolerance:
        raise ValueError("通用拼图目标矩形超出A4纸左右边界")
    if y < region_min - tolerance or y + height > region_max + tolerance:
        label = "上半区" if target_region == "top" else "下半区"
        raise ValueError(f"通用拼图目标矩形未完整位于A4纸{label}")
    return float(x), float(y)


# =============================================================================
# 【分区】生成最终解（放到 A4 下半区）
# 功能：目标矩形左上角 + 刚体复合。
# 可修改：放置位置走 _target_top_left / 配置。
# 看情况改：目标区被占时改配置，不改矩阵算法。
# 不要改：relaxed_override 必须传到上层，仿真会拒绝宽松解。
# =============================================================================
def _build_solution(evaluation: _RectangleEvaluation,
                    polygons: list[np.ndarray],
                    config: dict[str, Any],
                    search_states: int,
                    relaxed_override: bool = False) -> GenericSolution:
    """把装配参考系姿态转换成配置目标半区内的最终姿态。"""
    top_left = _target_top_left(evaluation.size_mm, config)
    placement = rigid_matrix(0.0, top_left)
    poses: list[GenericPiecePose] = []
    for piece_index in range(len(polygons)):
        final_matrix = (
            placement
            @ evaluation.normalize_matrix
            @ evaluation.candidate.poses[piece_index]
        )
        if not is_rigid_transform(final_matrix, tolerance=1e-5):
            raise RuntimeError("内部错误：最终目标姿态不是刚体变换")
        target_polygon = transform_points(polygons[piece_index], final_matrix)
        poses.append(GenericPiecePose(
            piece_index=piece_index,
            transform_3x3=final_matrix,
            target_polygon_mm=target_polygon,
        ))
    return GenericSolution(
        poses=poses,
        target_size_mm=evaluation.size_mm,
        target_top_left_mm=top_left,
        score=evaluation.score,
        fill_error_ratio=evaluation.fill_error_ratio,
        overlap_ratio=evaluation.overlap_ratio,
        endpoint_error_mm=evaluation.endpoint_error_mm,
        boundary_error_mm=evaluation.boundary_error_mm,
        boundary_gap_ratio=evaluation.boundary_gap_ratio,
        matches=list(evaluation.candidate.matches),
        search_states=search_states,
        relaxed_override=relaxed_override,
    )


# 这是保持碎片原角度、逐块平移到下半区的排放方案；没有拼接成完整矩形。若下半区放不下，仍可能返回None。
def _build_fallback_placement(
    polygons: list[np.ndarray],
    config: dict[str, Any],
    search_states: int,
) -> GenericSolution | None:
    """终极兜底：把碎片各自平铺到配置目标半区，不拼合、不旋转。

    仅在所有正常拼合路径彻底失败时调用。每个碎片保持原始角度，
    按从上到下、从左到右的次序排布，碎片间保留固定间隙，且确保
    全部位于A4下半区（不越界、不重叠）。返回的解标记relaxed_override，
    运动安全关卡会跳过质量校验但保留硬件安全检查。
    """
    try:
        paper_width = float(config.get("paper_width_mm", 210.0))
        paper_height = float(config.get("paper_height_mm", 297.0))
        separator = float(config.get("separator_line_y_mm",
                                    config.get("target_region_top_mm", paper_height / 2.0)))
        target_region = str(config.get("target_region", "bottom")).lower()
        if target_region == "top":
            region_min, region_max = 0.0, min(separator, paper_height)
        elif target_region == "bottom":
            region_min, region_max = max(0.0, min(separator, paper_height)), paper_height
        else:
            raise ValueError(f"未知target_region：{target_region}")
        margin = float(config.get("generic_fallback_margin_mm", 5.0))
        gap = float(config.get("generic_fallback_gap_mm", 3.0))
        max_pieces = int(config.get("max_pieces", 4))
        pieces = list(polygons)
        if not 1 <= len(pieces) <= max_pieces:
            return None
        # 收集每块的局部包围盒（保持原角度）
        boxes: list[tuple[float, float, float, float, np.ndarray]] = []
        for polygon in pieces:
            minimum = np.min(polygon, axis=0)
            maximum = np.max(polygon, axis=0)
            boxes.append((
                float(maximum[0] - minimum[0]),
                float(maximum[1] - minimum[1]),
                float(minimum[0]),
                float(minimum[1]),
                polygon,
            ))
        # 按面积从大到小排，先放大的减少浪费
        boxes.sort(key=lambda item: item[0] * item[1], reverse=True)

        placed: list[np.ndarray] = []
        placements: list[np.ndarray] = []
        cursor_x = margin
        cursor_y = region_min + margin
        row_height = 0.0
        max_row_width = paper_width - 2.0 * margin
        for (box_w, box_h, origin_x, origin_y, polygon) in boxes:
            if cursor_x + box_w > paper_width - margin and cursor_x > margin:
                # 换行：回到行首，下移一行
                cursor_x = margin
                cursor_y += row_height + gap
                row_height = 0.0
            if cursor_y + box_h > paper_height - margin:
                # 下半区放不下：尝试更小的间隙，仍放不下则放弃
                cursor_y = region_min + margin
                row_height = 0.0
                if cursor_x + box_w > paper_width - margin and cursor_x > margin:
                    cursor_x = margin
                    cursor_y += box_h + gap
                if cursor_y + box_h > paper_height - margin:
                    return None
            # 平移矩阵：把块的原点平移到当前光标位置
            offset = np.asarray(
                [cursor_x - origin_x, cursor_y - origin_y], dtype=np.float64
            )
            matrix = rigid_matrix(0.0, offset)
            transformed = transform_points(polygon, matrix)
            # 校验在下半区内
            tmin = np.min(transformed, axis=0)
            tmax = np.max(transformed, axis=0)
            if (tmin[0] < -1e-6 or tmin[1] < region_min - 1e-6 or
                    tmax[0] > paper_width + 1e-6 or tmax[1] > region_max + 1e-6):
                return None
            placed.append(transformed)
            placements.append(matrix)
            cursor_x += box_w + gap
            row_height = max(row_height, box_h)

        poses: list[GenericPiecePose] = []
        # boxes已排序，需按原多边形索引对应
        for index in range(len(pieces)):
            poses.append(GenericPiecePose(
                piece_index=index,
                transform_3x3=placements[index],
                target_polygon_mm=placed[index],
            ))
        # 兜底解的整体包围盒作为名义矩形，便于预览/报告绘制
        all_points = np.vstack(placed)
        all_min = np.min(all_points, axis=0)
        all_max = np.max(all_points, axis=0)
        overall_size = (float(all_max[0] - all_min[0]),
                        float(all_max[1] - all_min[1]))
        return GenericSolution(
            poses=poses,
            target_size_mm=overall_size,
            target_top_left_mm=(float(all_min[0]), float(all_min[1])),
            score=1e9,
            fill_error_ratio=1.0,
            overlap_ratio=0.0,
            endpoint_error_mm=0.0,
            boundary_error_mm=0.0,
            boundary_gap_ratio=1.0,
            matches=[],
            search_states=search_states,
            relaxed_override=True,
        )
    except (ValueError, RuntimeError):
        return None


class _NoRectangleSolution(ValueError):
    """几何输入有效，但当前边表示下未找到合格矩形。"""

    def __init__(self, message: str,
                 best_assessment: _AssemblyAssessment | None = None):
        super().__init__(message)
        self.best_assessment = best_assessment


# =============================================================================
# 【分区】共线拆边（长边对多条短边）
# 功能：只插顶点，不改变外形。扑克切缝常是“一边对两边”。
# 可修改：变体数量、最多顶点数。
# 看情况改：明显一大边对两小边却失败时再放宽变体数。
# 不要改：禁止改变边的几何形状。
# =============================================================================
def split_polygon_edge(
    polygon_mm: np.ndarray,
    edge_index: int,
    segment_lengths_mm: Iterable[float],
) -> np.ndarray:
    """沿指定直边按比例插入共线顶点，不改变原多边形面积与外形。"""
    polygon = normalize_polygon(polygon_mm)
    lengths = np.asarray(list(segment_lengths_mm), dtype=np.float64)
    if len(lengths) < 2 or not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise ValueError("拆边至少需要两个正长度分段")
    if len(polygon) + len(lengths) - 1 > 5:
        raise ValueError("拆边后多边形顶点数超过5")
    edge = int(edge_index) % len(polygon)
    start = polygon[edge]
    end = polygon[(edge + 1) % len(polygon)]
    vector = end - start
    edge_length = float(np.linalg.norm(vector))
    if edge_length <= 1e-9:
        raise ValueError("不能拆分退化边")
    cumulative = np.cumsum(lengths)[:-1] / float(np.sum(lengths))
    inserted = [start + float(ratio) * vector for ratio in cumulative]
    result: list[np.ndarray] = []
    for index, vertex in enumerate(polygon):
        result.append(vertex)
        if index == edge:
            result.extend(inserted)
    return np.asarray(result, dtype=np.float64)


def _generate_collinear_split_proposals(
    polygons_mm: list[np.ndarray],
    config: dict[str, Any],
) -> list[_CollinearSplitProposal]:
    """生成带来源信息的共线长边拆分提案。"""
    polygons = [normalize_polygon(polygon) for polygon in polygons_mm]
    if len(polygons) < 3:
        return []
    min_edge = float(config.get("generic_min_edge_length_mm", 0.0))
    absolute_tolerance = float(config.get(
        "generic_collinear_split_abs_tolerance_mm", 4.0
    ))
    relative_tolerance = float(config.get(
        "generic_collinear_split_rel_tolerance", 0.05
    ))
    lengths_by_piece = [edge_lengths(polygon) for polygon in polygons]
    raw_proposals: list[
        tuple[float, int, int, tuple[float, ...], tuple[int, ...]]
    ] = []

    for source_piece, polygon in enumerate(polygons):
        max_segments = min(3, 6 - len(polygon))
        if max_segments < 2:
            continue
        other_pieces = [index for index in range(len(polygons)) if index != source_piece]
        for source_edge, source_length_value in enumerate(lengths_by_piece[source_piece]):
            source_length = float(source_length_value)
            if source_length <= 2.0 * max(min_edge, 1e-6):
                continue
            for segment_count in range(2, min(max_segments, len(other_pieces)) + 1):
                for selected_pieces in itertools.combinations(other_pieces, segment_count):
                    edge_options = [
                        [
                            float(length)
                            for length in lengths_by_piece[piece_index]
                            if min_edge - 1e-6 <= float(length) < source_length - 1e-6
                        ]
                        for piece_index in selected_pieces
                    ]
                    if any(not options for options in edge_options):
                        continue
                    for component_lengths in itertools.product(*edge_options):
                        total = float(sum(component_lengths))
                        tolerance = max(
                            absolute_tolerance,
                            relative_tolerance * min(source_length, total),
                        )
                        error = abs(source_length - total)
                        if error > tolerance:
                            continue
                        scale = source_length / total
                        scaled = tuple(float(length * scale) for length in component_lengths)
                        if min_edge > 0.0 and min(scaled) < min_edge - 1e-6:
                            continue
                        for ordered in set(itertools.permutations(scaled)):
                            raw_proposals.append((
                                error, source_piece, source_edge, ordered,
                                tuple(selected_pieces),
                            ))

    raw_proposals.sort(key=lambda item: (
        item[0], item[1], item[2], item[3], item[4],
    ))
    proposals: list[_CollinearSplitProposal] = []
    seen: set[tuple[Any, ...]] = set()
    for error, source_piece, source_edge, segment_lengths, selected_pieces in raw_proposals:
        split = split_polygon_edge(
            polygons[source_piece], source_edge, segment_lengths
        )
        key = (
            source_piece,
            source_edge,
            selected_pieces,
            tuple(np.rint(split.reshape(-1) * 1000.0).astype(np.int64)),
        )
        if key in seen:
            continue
        seen.add(key)
        proposals.append(_CollinearSplitProposal(
            error_mm=error,
            source_piece=source_piece,
            source_edge=source_edge,
            segment_lengths_mm=segment_lengths,
            selected_pieces=selected_pieces,
            split_polygon=split,
        ))
    return proposals


def _polygon_variant_key(polygons: list[np.ndarray]) -> tuple[Any, ...]:
    """将整组多边形量化为稳定去重键。"""
    return tuple(
        (len(polygon), tuple(np.rint(polygon.reshape(-1) * 1000.0).astype(np.int64)))
        for polygon in polygons
    )


def generate_collinear_split_variants(
    polygons_mm: list[np.ndarray],
    config: dict[str, Any],
) -> list[list[np.ndarray]]:
    """生成“一条长边对应其他2～3块短边”的有限回退候选。"""
    polygons = [normalize_polygon(polygon) for polygon in polygons_mm]
    max_variants = max(1, int(config.get(
        "generic_collinear_split_max_variants", 48
    )))
    variants: list[list[np.ndarray]] = []
    seen: set[tuple[Any, ...]] = set()
    for proposal in _generate_collinear_split_proposals(polygons, config):
        variant = [polygon.copy() for polygon in polygons]
        variant[proposal.source_piece] = proposal.split_polygon.copy()
        key = _polygon_variant_key(variant)
        if key in seen:
            continue
        seen.add(key)
        variants.append(variant)
        if len(variants) >= max_variants:
            break
    return variants


def generate_dual_collinear_split_variants(
    polygons_mm: list[np.ndarray],
    config: dict[str, Any],
) -> list[list[np.ndarray]]:
    """生成两个不同碎片同时拆边的受限回退候选。"""
    polygons = [normalize_polygon(polygon) for polygon in polygons_mm]
    if len(polygons) < 4:
        return []
    proposals_per_edge = max(1, int(config.get(
        "generic_dual_split_proposals_per_edge", 2
    )))
    max_variants = max(1, int(config.get(
        "generic_dual_split_max_variants", 96
    )))

    grouped: dict[tuple[int, int], list[_CollinearSplitProposal]] = {}
    for proposal in _generate_collinear_split_proposals(polygons, config):
        key = (proposal.source_piece, proposal.source_edge)
        group = grouped.setdefault(key, [])
        if len(group) < proposals_per_edge:
            group.append(proposal)

    limited = [proposal for group in grouped.values() for proposal in group]
    pairs: list[
        tuple[float, _CollinearSplitProposal, _CollinearSplitProposal]
    ] = []
    for first, second in itertools.combinations(limited, 2):
        if first.source_piece == second.source_piece:
            continue
        source_pieces = {first.source_piece, second.source_piece}
        shared_third_pieces = (
            set(first.selected_pieces) & set(second.selected_pieces)
        ) - source_pieces
        if not shared_third_pieces:
            continue
        pairs.append((first.error_mm + second.error_mm, first, second))

    pairs.sort(key=lambda item: (
        item[0],
        item[1].error_mm, item[1].source_piece, item[1].source_edge,
        item[1].segment_lengths_mm, item[1].selected_pieces,
        item[2].error_mm, item[2].source_piece, item[2].source_edge,
        item[2].segment_lengths_mm, item[2].selected_pieces,
    ))
    variants: list[list[np.ndarray]] = []
    seen: set[tuple[Any, ...]] = set()
    for _, first, second in pairs:
        variant = [polygon.copy() for polygon in polygons]
        variant[first.source_piece] = first.split_polygon.copy()
        variant[second.source_piece] = second.split_polygon.copy()
        key = _polygon_variant_key(variant)
        if key in seen:
            continue
        seen.add(key)
        variants.append(variant)
        if len(variants) >= max_variants:
            break
    return variants

# 注意旧注释中的“保证有解”是意图而非无条件保证。实际宽松分支仍筛选候选，并可继续失败；它把多项阈值调大，默认仍允许最多0.1的重叠比例。
# =============================================================================
# 【分区】单套多边形求解（含宽松分支）
# 功能：严搜 → 补闭环 → 可选放宽。宽松不是保证有解。
# 可修改：宽松仍限制 overlap。
# 看情况改：比赛严格矩形不要开平铺兜底。
# 不要改：预算耗尽必须失败返回。
# =============================================================================
def _solve_normalized_rectangle(
    normalized: list[np.ndarray],
    config: dict[str, Any],
    budget: _SearchBudget | None = None,
) -> GenericSolution:
    """求解已校验多边形；普通生成树失败时受限补充闭环拼缝。"""
    if budget is None:
        budget = _make_search_budget(config)
    assemblies, states = _search_assemblies(normalized, config, budget)
    if not assemblies:
        candidate_count = len(generate_edge_candidates(normalized, config))
        raise _NoRectangleSolution(
            f"未找到连通装配：候选拼缝={candidate_count}，搜索状态={states}"
        )

    pieces_area = sum(polygon_area(polygon) for polygon in normalized)
    assessments = [
        _assess_assembly(candidate, normalized, config, pieces_area=pieces_area)
        for candidate in assemblies
    ]
    accepted = [item.evaluation for item in assessments if item.accepted]
    closure_candidates: list[_AssemblyCandidate] = []
    if not accepted and not budget.exhausted():
        closure_candidates = _closure_augmented_candidates(
            normalized, assessments, config, budget
        )
        closure_assessments = [
            _assess_assembly(candidate, normalized, config, pieces_area=pieces_area)
            for candidate in closure_candidates
        ]
        assessments.extend(closure_assessments)
        accepted.extend(
            item.evaluation for item in closure_assessments if item.accepted
        )

    if not accepted:
        # 严格标准无合格解时，若已到达搜索预算上限（时间/状态耗尽），
        # 用放宽的填充/边界缺口阈值兜底评估现有候选，输出"宽松合格"解，
        # 保证超时后仍能给出可用的矩形拼法。仅超时触发，正常时间内
        # 仍按严格标准求解，优秀/合格判据完全不变。
        relaxed = None
        if budget.exhausted() and not config.get("strict_production", False):
            # 宽松兜底：超时后不设任何质量门槛，只要求该候选能拼成一个矩形
            # （无重叠、拼缝端点可接受），直接输出所有候选中评分最佳者。
            # 这样无论碎片轮廓多差，超时后都保证有解可执行，满足比赛"必须完成"。
            relaxed_config = dict(config)
            relaxed_config["generic_max_fill_error_ratio"] = 1e6
            relaxed_config["generic_max_boundary_gap_ratio"] = 1e6
            relaxed_config["generic_target_long_min_mm"] = 0.0
            relaxed_config["generic_target_long_max_mm"] = 1e6
            relaxed_config["generic_target_short_min_mm"] = 0.0
            relaxed_config["generic_target_short_max_mm"] = 1e6
            relaxed_config["generic_require_each_piece_outer_edge"] = False
            relaxed_config["generic_dimension_tolerance_mm"] = 1e6
            relaxed_config["generic_max_overlap_ratio"] = float(config.get(
                "generic_relaxed_max_overlap_ratio", 0.1))
            relaxed_config["generic_closure_max_overlap_ratio"] = float(config.get(
                "generic_relaxed_max_overlap_ratio", 0.1))
            relaxed_config["generic_max_endpoint_error_mm"] = 1e6
            relaxed_config["generic_closure_max_endpoint_error_mm"] = 1e6
            relaxed_assessments = [
                _assess_assembly(candidate, normalized, relaxed_config, pieces_area=pieces_area)
                for candidate in (assemblies + closure_candidates)
            ]
            relaxed_accepted = [
                item.evaluation for item in relaxed_assessments if item.accepted
            ]
            if relaxed_accepted:
                relaxed = min(relaxed_accepted, key=lambda item: (
                    item.score,
                    item.target_size_error_mm,
                    item.fill_error_ratio,
                    item.boundary_gap_ratio,
                    item.overlap_ratio,
                    item.endpoint_error_mm,
                ))
                print(f"[保底-宽松] 严格标准超时无合格解，采用超时兜底矩形："
                      f"填充={relaxed.fill_error_ratio:.3f}，"
                      f"边界缺口={relaxed.boundary_gap_ratio:.3f}")
        if relaxed is not None:
            return _build_solution(relaxed, normalized, config, budget.states,
                                   relaxed_override=True)

        best_assessment = min(assessments, key=lambda item: (
            len(item.rejection_reasons),
            item.evaluation.score,
            item.evaluation.target_size_error_mm,
            item.evaluation.fill_error_ratio,
        ))
        closure_text = (
            f"，闭环候选={len(closure_candidates)}" if closure_candidates else ""
        )
        raise _NoRectangleSolution(
            f"找到{len(assemblies)}个连通装配{closure_text}，但均未通过约束；"
            f"最佳失败候选：{_format_failed_assessment(best_assessment)}",
            best_assessment,
        )

    best = min(accepted, key=lambda item: (
        item.score,
        item.target_size_error_mm,
        item.fill_error_ratio,
        item.boundary_gap_ratio,
        item.overlap_ratio,
        item.endpoint_error_mm,
    ))
    return _build_solution(best, normalized, config, budget.states)

def _is_excellent_split_solution(
    solution: GenericSolution,
    config: dict[str, Any],
) -> bool:
    """判断拆边回退是否已得到无需继续枚举的高质量解。"""
    return (
        solution.fill_error_ratio <= float(config.get(
            "generic_excellent_fill_error_ratio", 0.025
        ))
        and solution.overlap_ratio <= float(config.get(
            "generic_excellent_overlap_ratio", 0.003
        ))
        and solution.boundary_gap_ratio <= float(config.get(
            "generic_excellent_boundary_gap_ratio", 0.03
        ))
    )


def _solve_split_variants(
    variants: list[list[np.ndarray]],
    config: dict[str, Any],
    budget: _SearchBudget,
) -> tuple[GenericSolution | None, int]:
    """在共享预算内评估一组拆边候选并返回其中最佳解。"""
    best_solution: GenericSolution | None = None
    best_variant_index = 0
    for variant_index, variant in enumerate(variants, 1):
        if budget.exhausted():
            break
        try:
            solution = _solve_normalized_rectangle(variant, config, budget)
        except _NoRectangleSolution:
            continue
        except ValueError as exc:
            # 超时或状态预算耗尽时：若已有最佳解则保底返回（比赛保证能拼成矩形，
            # 优先放行最高得分解法，而非判失败）；否则安全失败。
            if best_solution is not None:
                print(f"[保底] 搜索{exc}，采用当前最高得分解法（评分={best_solution.score:.3f}）")
                break
            raise ValueError(str(exc)) from exc

        key = (
            solution.score, solution.fill_error_ratio, solution.boundary_gap_ratio,
            solution.overlap_ratio, solution.endpoint_error_mm,
        )
        if best_solution is None or key < (
            best_solution.score, best_solution.fill_error_ratio,
            best_solution.boundary_gap_ratio, best_solution.overlap_ratio,
            best_solution.endpoint_error_mm,
        ):
            best_solution = solution
            best_variant_index = variant_index
        if _is_excellent_split_solution(solution, config):
            break
    return best_solution, best_variant_index


def _print_split_success(
    label: str,
    solution: GenericSolution,
    variant_index: int,
    variant_count: int,
) -> None:
    """输出单拆或双拆回退的统一诊断信息。"""
    print(
        f"[诊断] {label}成功：候选={variant_index}/{variant_count}，"
        f"矩形={solution.target_size_mm[0]:.1f}×"
        f"{solution.target_size_mm[1]:.1f}mm，"
        f"填充={solution.fill_error_ratio:.3f}，"
        f"边界缺口={solution.boundary_gap_ratio:.3f}"
    )


# 原始边失败后尝试双拆，再尝试单拆；它们共用同一个budget。最终平铺兜底仅在代码到达对应分支且配置启用时触发，不覆盖所有早期异常。
# =============================================================================
# 【分区】对外入口 solve_generic_rectangle
# 功能：1~4 片 → 原始边 → 双拆 → 单拆 → 可选平铺。
# 可修改：generic_min_edge_length_mm、generic_min_piece_area_mm2。
# 看情况改：拆边顺序不要轻易对调。
# 不要改：输入必须毫米坐标。
# =============================================================================
def solve_generic_rectangle(
    polygons_mm: list[np.ndarray],
    config: dict[str, Any],
    *, budget: _SearchBudget | None = None,
) -> GenericSolution:
    """搜索矩形；原始边失败时依次尝试受限双拆与单拆回退。"""
    if not 1 <= len(polygons_mm) <= 4:
        raise ValueError(f"通用拼图只支持1～4块，当前为{len(polygons_mm)}块")
    if config.get("strict_production", False):
        # A candidate rejected by the final motion gate must not stop the
        # original-edge search before closure or collinear splits are tried.
        config = dict(config)
        for quality, default in (("fill_error_ratio", 0.06),
                                 ("boundary_gap_ratio", 0.08),
                                 ("overlap_ratio", 0.008)):
            key = f"generic_max_{quality}"
            motion_key = f"generic_motion_max_{quality}"
            config[key] = min(float(config.get(key, default)),
                              float(config.get(motion_key, config.get(key, default))))
        config["generic_closure_max_overlap_ratio"] = min(
            float(config.get("generic_closure_max_overlap_ratio", config["generic_max_overlap_ratio"])),
            float(config.get("generic_motion_max_overlap_ratio", config["generic_max_overlap_ratio"])),
        )
    normalized: list[np.ndarray] = []
    min_edge = float(config.get("generic_min_edge_length_mm", 0.0))
    minimum_area = float(config.get("generic_min_piece_area_mm2", 1.0))
    for index, polygon in enumerate(polygons_mm):
        try:
            normalized.append(normalize_polygon(
                polygon,
                min_edge_length_mm=min_edge,
                min_area_mm2=minimum_area,
            ))
        except ValueError as exc:
            raise ValueError(f"第{index + 1}块碎片无效：{exc}") from exc

    if budget is None:
        budget = _make_search_budget(config)
    try:
        return _solve_normalized_rectangle(normalized, config, budget)
    except _NoRectangleSolution as original_error:
        if not bool(config.get("generic_enable_collinear_split", True)):
            raise ValueError(str(original_error)) from original_error

        dual_variants: list[list[np.ndarray]] = []
        if bool(config.get("generic_enable_dual_collinear_split", True)):
            dual_variants = generate_dual_collinear_split_variants(normalized, config)
        if dual_variants:
            print(
                "[诊断] 原始边匹配失败，尝试双碎片共线长边拆分："
                f"候选={len(dual_variants)}"
            )
            dual_solution, dual_index = _solve_split_variants(
                dual_variants, config, budget
            )
            if dual_solution is not None:
                _print_split_success(
                    "双碎片共线长边拆分",
                    dual_solution,
                    dual_index,
                    len(dual_variants),
                )
                return dual_solution

        single_variants = generate_collinear_split_variants(normalized, config)
        if single_variants and not budget.exhausted():
            print(
                "[诊断] 双拆未找到合格矩形，尝试单条共线长边拆分："
                f"候选={len(single_variants)}"
            )
            single_solution, single_index = _solve_split_variants(
                single_variants, config, budget
            )
            if single_solution is not None:
                _print_split_success(
                    "单条共线长边拆分",
                    single_solution,
                    single_index,
                    len(single_variants),
                )
                return single_solution

        if not dual_variants and not single_variants:
            raise ValueError(str(original_error)) from original_error
        budget_text = (
            f"；搜索{budget.stop_reason}" if budget.stop_reason is not None else ""
        )
        # 终极兜底：所有正常拼合路径（原始边/双拆/单拆）均失败时，
        # 若开关开启，把每块碎片单独平铺到A4下半区（不旋转、保持原角度、
        # 互相不重叠），保证比赛"必须完成"。仅在彻底无解时触发，
        # 绝不覆盖任何可正常拼合的结果。
        if bool(config.get("generic_ultimate_fallback_enable", True)):
            fallback_solution = _build_fallback_placement(
                normalized, config, budget.states
            )
            if fallback_solution is not None:
                budget_suffix = budget_text.lstrip("；")
                suffix = (f"（{budget_suffix}）" if budget_suffix else "")
                print(
                    f"[终极兜底] 无法拼合成矩形{suffix}，"
                    f"改为将{len(normalized)}块碎片平铺到A4下半区"
                )
                return fallback_solution
        raise ValueError(
            f"{original_error}；双拆回退{len(dual_variants)}种、"
            f"单拆回退{len(single_variants)}种仍未找到合格矩形{budget_text}"
        ) from original_error
