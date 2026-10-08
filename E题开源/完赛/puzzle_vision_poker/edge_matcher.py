"""
扑克牌碎片边缘花纹匹配模块。

原理：
  碎片从同一张扑克牌切出，切缝两侧的牌面花纹是连续的。
  沿切缝两侧各取一条像素带，花样应高度相似 → NCC 评分。

用于：
  1. 验证几何求解结果是否正确（花纹对上了）
  2. 多候选解中选花纹相似度最高的
  3. 纯花纹匹配兜底（几何求解失败时暴力搜索）

用法：
  from edge_matcher import (
      extract_all_profiles,
      score_solution,
      find_best_by_edge,
  )
"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：扑克切缝花纹：沿边向内采样灰度带，NCC 评分；用于验证几何解或花纹兜底配对（不生成刚体姿态）。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约63行  EdgeProfile：一条边的采样带
#   - 约74行  EdgeSimilarity：一对边的 NCC 和 similarity=(ncc+1)/2
#   - 约95行  extract_edge_profiles：提取一片所有边的花纹
#   - 约175行  extract_all_profiles：所有碎片的花纹
#   - 约199行  profile_ncc：两条灰度带归一化互相关
#   - 约289行  find_adjacent_pairs：目标布局上空间相邻的边
#   - 约333行  score_solution：给几何解的拼缝打花纹分
#   - 约439行  find_best_by_edge：纯花纹贪心配对（不产出搬运姿态）
#   - 约536行  draw_edge_profiles：可视化辅助，目前返回图像副本
# =============================================================================
# 扑克牌模块利用切缝两侧花纹应连续的特点，为几何解提供额外评分；输入是校正图、各片多边形和像素/毫米比例。
# 沿每条边向内偏移offset_mm再等距取样，得到灰度序列；NCC比较灰度起伏趋势，不是逐像素相等比较。
# NCC先减均值、再除标准差；1表示正相关，0表示无明显线性相关，-1表示反相关。当前实现也比较反序列，取较高结果。
# 相似度使用(NCC+1)/2，因此无明显相关的0会映射为0.5，不能把0.5理解成一半像素正确。
# 注意现有功能边界：find_best_by_edge返回贪心选择的边配对信息，没有生成碎片刚体姿态；draw_edge_profiles目前只返回图像副本。

# =============================================================================
# 【分区】导入
# 功能：OpenCV 取样、numpy 算 NCC。本模块只给几何解打花纹分，不生成刚体姿态。
# 可修改：无现场参数；采样间距/偏移在 extract 函数默认值或调用方 config。
# 看情况改：无。
# 不要改：不要在这里 import main，避免循环依赖。
# =============================================================================
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# =============================================================================
# 【分区】数据结构
# 功能：EdgeProfile=一条边的灰度/彩色采样带；EdgeSimilarity=一对边的 NCC 与映射后的 similarity。
# 可修改：无。
# 看情况改：similarity=(ncc+1)/2，无相关的 0 会变成 0.5，不要把 0.5 当成“一半像素正确”。
# 不要改：字段名 ncc/similarity；pattern_gate 按这些键做换算校验。
# =============================================================================
# ====================== 数据结构 ======================

@dataclass
class EdgeProfile:
    """单条边的花纹轮廓"""
    piece_id: int
    edge_idx: int
    length_mm: float
    profile_gray: np.ndarray
    profile_bgr: np.ndarray
    inward_normal: np.ndarray


@dataclass
class EdgeSimilarity:
    """一对相邻边的相似度"""
    piece_a: int
    edge_a: int
    piece_b: int
    edge_b: int
    shared_mm: float
    ncc: float
    similarity: float


# =============================================================================
# 【分区】边缘花纹提取
# 功能：沿每条边向内偏 offset_mm，按 sample_spacing_mm 取灰度序列。
# 可修改：sample_spacing_mm 默认 1.0、offset_mm 默认 2.0。偏移太小会采到切缝背景，太大采到牌心无关花纹。
# 看情况改：向内法向用“指向顶点均值”的启发式，复杂凹形可能个别点向外。花纹对不上先查偏移，再查光照。
# 不要改：在源多边形上采样（未拼合时牌面完整）；边索引必须与 generic_solver 的 edge_a/edge_b 一致。
# =============================================================================
# ====================== 边缘花纹提取 ======================

# 用“边的垂直向量指向顶点平均位置”决定向内方向，这是本实现的启发式；在复杂凹形上不能视为对所有采样点的严格内部保证。
def extract_edge_profiles(
    warped_image: np.ndarray,
    piece_id: int,
    polygon_mm: np.ndarray,
    ppm: float,
    sample_spacing_mm: float = 1.0,
    offset_mm: float = 2.0,
) -> List[EdgeProfile]:
    """
    提取一片碎片所有边的花纹轮廓。

    Args:
        warped_image : 校正后的 BGR A4 纸图像
        piece_id     : 碎片编号
        polygon_mm   : (N,2) 简化多边形顶点，单位 mm
        ppm          : 像素/mm
        sample_spacing_mm : 沿边采样间隔
        offset_mm    : 从边向碎片内部偏移距离

    Returns:
        EdgeProfile 列表，每条边一个
    """
    polygon_px = polygon_mm * ppm
    center_mm = np.mean(polygon_mm, axis=0)
    center_px = center_mm * ppm
    img_h, img_w = warped_image.shape[:2]
    offset_px = offset_mm * ppm
    sample_spacing_px = max(1.0, sample_spacing_mm * ppm)

    profiles = []
    n = len(polygon_px)

    for i in range(n):
        p1 = polygon_px[i]
        p2 = polygon_px[(i + 1) % n]
        edge_vec = p2 - p1
        edge_len_px = np.linalg.norm(edge_vec)

        if edge_len_px < 5:
            continue

        edge_dir = edge_vec / edge_len_px
        edge_len_mm = edge_len_px / ppm

        # 判定指向碎片内部的法向量
        normal = np.array([-edge_dir[1], edge_dir[0]], dtype=np.float64)
        mid = (p1 + p2) / 2
        if np.dot(normal, center_px - mid) < 0:
            normal = -normal

        # 沿边等距采样
        num_samples = max(5, int(edge_len_px / sample_spacing_px))
        colors = []

        for j in range(num_samples):
            t = (j + 0.5) / num_samples
            base = p1 + t * edge_vec
            sample_pt = base + offset_px * normal
            sx = int(np.clip(round(sample_pt[0]), 0, img_w - 1))
            sy = int(np.clip(round(sample_pt[1]), 0, img_h - 1))
            colors.append(warped_image[sy, sx].astype(np.float32))

        profile_bgr = np.array(colors)
        profile_gray = cv2.cvtColor(
            profile_bgr.reshape(1, -1, 3).astype(np.uint8),
            cv2.COLOR_BGR2GRAY
        ).ravel().astype(np.float32)

        profiles.append(EdgeProfile(
            piece_id=piece_id,
            edge_idx=i,
            length_mm=edge_len_mm,
            profile_gray=profile_gray,
            profile_bgr=profile_bgr,
            inward_normal=normal,
        ))

    return profiles


def extract_all_profiles(
    warped_image: np.ndarray,
    pieces: list,
    polygons_mm: list,
    ppm: float,
) -> Dict[int, List[EdgeProfile]]:
    """为所有碎片提取边缘花纹轮廓"""
    result = {}
    for piece, poly_mm in zip(pieces, polygons_mm):
        pid = getattr(piece, 'piece_id', id(piece))
        result[pid] = extract_edge_profiles(warped_image, pid, np.asarray(poly_mm), ppm)
    return result


# =============================================================================
# 【分区】边缘相似度（NCC）
# 功能：两条灰度带归一化互相关；短边按整数下标抽到相同长度；正反序列取较高分。
# 可修改：最短 5 个采样点才比较，再短返回 0。
# 看情况改：重采样不是插值补长，是抽下标。两边长度差很多时 NCC 会偏乐观或偏噪。
# 不要改：先减均值再除标准差；纯色标准差过小返回 0，避免除零被当成完美匹配。
# =============================================================================
# ====================== 边缘相似度计算 ======================

# 长度统一使用较短序列的长度，按整数下标均匀抽样，并非插值补长；纯色序列标准差太小，返回0避免除零。
def profile_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """
    两条边缘轮廓的归一化互相关 (NCC)。
    自动处理长度不同（短边补采样）和方向颠倒（翻转）。
    """
    if len(a) < 5 or len(b) < 5:
        # 太短的边无法可靠比较
        return 0.0

    # 统一长度：取较短者，均匀重采样
    target_len = min(len(a), len(b))

    def resample(x):
        idx = np.linspace(0, len(x) - 1, target_len).astype(int)
        return x[idx].astype(np.float64)

    ar = resample(a)
    br = resample(b)

    def ncc(x, y):
        xm = x.mean()
        ym = y.mean()
        xs = x.std()
        ys = y.std()
        if xs < 1e-8 or ys < 1e-8:
            # 纯色边（几乎没有花纹变化），中立评分
            return 0.0
        return float(np.mean((x - xm) * (y - ym)) / (xs * ys))

    score_normal = ncc(ar, br)
    score_flipped = ncc(ar, br[::-1])

    return max(score_normal, score_flipped)


# =============================================================================
# 【分区】相邻边检测
# 功能：在目标布局上找空间上靠近、长度重叠的边对，供几何解验证花纹。
# 可修改：共享长度阈值见 _shared_edge_length。
# 看情况改：验证用求解器给出的 matches 更稳；本函数是几何邻接的辅助。
# 不要改：邻接判断不要改成“花纹高就当邻边”，那会循环论证。
# =============================================================================
# ====================== 相邻边检测 ======================

def _shared_edge_length(
    e1_start: np.ndarray,
    e1_end: np.ndarray,
    e2_start: np.ndarray,
    e2_end: np.ndarray,
    tol_mm: float
) -> float:
    """
    计算两条线段的重叠长度（mm）。
    要求：两线段几乎平行 且 线间距 ≤ tol_mm。
    返回重叠长度 (mm)；不满足条件返回 0。
    """
    es = np.asarray(e1_start, dtype=np.float64)
    ee = np.asarray(e1_end, dtype=np.float64)
    fs = np.asarray(e2_start, dtype=np.float64)
    fe = np.asarray(e2_end, dtype=np.float64)

    v1 = ee - es
    v2 = fe - fs
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0

    d1 = v1 / len1
    d2 = v2 / len2

    # 平行度检测：cosθ > 0.96 ≈ 16°
    dot = abs(np.dot(d1, d2))
    if dot < 0.96:
        return 0.0

    # 线间距检测
    n = np.array([-d1[1], d1[0]])
    dist = abs(np.dot(n, fs - es))
    if dist > tol_mm:
        return 0.0

    # 将 e2 端点投影到 e1 所在直线上，取重叠区间
    t_a = np.dot(fs - es, d1)
    t_b = np.dot(fe - es, d1)
    overlap_start = max(0.0, min(t_a, t_b))
    overlap_end = min(len1, max(t_a, t_b))
    return max(0.0, overlap_end - overlap_start)


def find_adjacent_pairs(
    matches: list,
    tolerance_mm: float = 2.0
) -> List[Tuple[int, int, int, int, float]]:
    """
    在拼图结果中找出所有相邻边对。

    Args:
        matches: PuzzleMatch 对象列表（含 target_polygon_mm）
        tolerance_mm: 边间距/平行度容忍

    Returns:
        List of (piece_i, edge_j, piece_p, edge_q, shared_length_mm)
    """
    pairs = []
    n = len(matches)

    for i in range(n):
        poly_i = np.asarray(matches[i].target_polygon_mm, dtype=np.float64)
        ni = len(poly_i)
        for j in range(ni):
            v1_i = poly_i[j]
            v2_i = poly_i[(j + 1) % ni]
            for p in range(i + 1, n):
                poly_p = np.asarray(matches[p].target_polygon_mm, dtype=np.float64)
                np_ = len(poly_p)
                for q in range(np_):
                    v1_p = poly_p[q]
                    v2_p = poly_p[(q + 1) % np_]
                    overlap = _shared_edge_length(v1_i, v2_i, v1_p, v2_p, tolerance_mm)
                    if overlap > 0:
                        pairs.append((i, j, p, q, overlap))
    return pairs


# =============================================================================
# 【分区】方案评分
# 功能：对几何解的每条拼缝算 NCC，汇总 min/mean similarity，供选解和门禁。
# 可修改：调用方阈值 poker_ncc_min_similarity。
# 看情况改：几何 score 越小越好，这里 similarity 越大越好，加权融合时先统一方向。
# 不要改：拼缝边对必须来自求解器 matches 的源边索引。
# =============================================================================
# ====================== 方案评分 ======================

def score_solution(
    matches: list,
    edge_profiles: Dict[int, List[EdgeProfile]],
    tolerance_mm: float = 2.0,
) -> dict:
    """
    对几何求解结果进行边缘花纹相似度评分。

    Returns:
        {
            'score': float,         # [0, 1] 总评分，越高花纹越吻合
            'pairs': [EdgeSimilarity, ...],
            'n_pairs': int,         # 相邻边对数
            'message': str,         # 中文摘要
        }
    """
    adjacency = find_adjacent_pairs(matches, tolerance_mm)

    if not adjacency:
        return {
            'score': 0.5,
            'pairs': [],
            'n_pairs': 0,
            'message': '未检测到相邻边对（碎片间距 > {:.1f}mm）'.format(tolerance_mm),
        }

    pair_results: List[EdgeSimilarity] = []
    scores_list = []

    for pi, ej, pp, eq, shared_len in adjacency:
        pid_i = getattr(matches[pi].piece, 'piece_id', pi)
        pid_p = getattr(matches[pp].piece, 'piece_id', pp)

        profs_i = edge_profiles.get(pid_i, [])
        profs_p = edge_profiles.get(pid_p, [])

        if ej < len(profs_i) and eq < len(profs_p):
            ncc_val = profile_ncc(profs_i[ej].profile_gray, profs_p[eq].profile_gray)
            sim = (ncc_val + 1.0) / 2.0
        else:
            ncc_val = 0.0
            sim = 0.0

        scores_list.append(sim)
        pair_results.append(EdgeSimilarity(
            piece_a=pid_i, edge_a=ej,
            piece_b=pid_p, edge_b=eq,
            shared_mm=round(shared_len, 1),
            ncc=round(ncc_val, 4),
            similarity=round(sim, 4),
        ))

    avg = float(np.mean(scores_list)) if scores_list else 0.5

    # 生成中文消息
    if avg >= 0.85:
        msg = f'花纹吻合度: {avg:.0%} ✓ (高度匹配)'
    elif avg >= 0.65:
        msg = f'花纹吻合度: {avg:.0%} ⚡ (中度匹配)'
    elif avg >= 0.50:
        msg = f'花纹吻合度: {avg:.0%} ⚠ (偏低，请核实)'
    else:
        msg = f'花纹吻合度: {avg:.0%} ✗ (不匹配，拼法可能错误)'

    return {
        'score': round(avg, 4),
        'pairs': pair_results,
        'n_pairs': len(pair_results),
        'message': msg,
    }


# =============================================================================
# 【分区】纯花纹匹配暴力搜索（兜底）
# 功能：几何失败时按花纹贪心配对边。返回边对信息，不生成刚体姿态，不能直接当搬运计划。
# 可修改：搜索宽度、相似度下限。
# 看情况改：这是兜底线索，真正位姿仍要走 generic_solver 或人工。
# 不要改：不要把 find_best_by_edge 的返回值直接写成 plan.json 去发电机。
# =============================================================================
# ====================== 纯花纹匹配暴力搜索（兜底） ======================

def _compute_all_profile_sims(
    profiles_list: List[List[EdgeProfile]],
) -> np.ndarray:
    """
    构建所有碎片所有边两两之间的相似度矩阵。

    Returns:
        sims[i][j][k][l]: piece_i 边 j 与 piece_k 边 l 的 NCC [-1,1]
    """
    n_pieces = len(profiles_list)
    max_edges = max(len(pl) for pl in profiles_list)
    sims = np.zeros((n_pieces, max_edges, n_pieces, max_edges), dtype=np.float32)

    for i in range(n_pieces):
        for j, ep_a in enumerate(profiles_list[i]):
            for k in range(i + 1, n_pieces):
                for l, ep_b in enumerate(profiles_list[k]):
                    s = profile_ncc(ep_a.profile_gray, ep_b.profile_gray)
                    sims[i][j][k][l] = s
                    sims[k][l][i][j] = s
    return sims


# 旧说明称为“暴力搜索”，实际代码是按NCC降序贪心选边；没有用并查集排除环，选到n-1对也不自动保证所有碎片连通。
# polygons_mm、target_w_mm、target_h_mm在此版本函数体中未参与定位计算；返回selected_edges由调用方继续处理。
def find_best_by_edge(
    pieces: list,
    polygons_mm: list,
    edge_profiles: Dict[int, List[EdgeProfile]],
    target_w_mm: float,
    target_h_mm: float,
) -> Optional[dict]:
    """
    纯用边缘花纹相似度搜索最优拼法（当几何求解不可靠时兜底）。

    原理：
    - 对所有碎片边的两两配对计算 NCC
    - 选相似度最高的 (n-1) 对不冲突的边匹配
    - 将其拼接成目标矩形

    返回 {matches, score} 或 None（未找到有效拼法）。
    """
    n = len(pieces)
    if n <= 1:
        return None

    # 构建相似度矩阵
    n_max_edges = max(len(prof) for prof in edge_profiles.values())
    # key: (piece_a, edge_a), value: list of (piece_b, edge_b, ncc) sorted by ncc desc
    all_candidates: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = {}

    piece_ids = list(edge_profiles.keys())
    id_to_idx = {pid: idx for idx, pid in enumerate(piece_ids)}

    for idx_a, pid_a in enumerate(piece_ids):
        profs_a = edge_profiles[pid_a]
        for ea, ep_a in enumerate(profs_a):
            key = (pid_a, ea)
            cands = []
            for idx_b, pid_b in enumerate(piece_ids):
                if pid_b == pid_a:
                    continue
                profs_b = edge_profiles[pid_b]
                for eb, ep_b in enumerate(profs_b):
                    s = profile_ncc(ep_a.profile_gray, ep_b.profile_gray)
                    cands.append((pid_b, eb, s))
            cands.sort(key=lambda x: -x[2])
            all_candidates[key] = cands

    # 贪心选取：每次选 NCC 最高的未使用边对
    used_edges: set = set()
    used_pairs: set = set()
    selected = []

    # 收集所有候选边对
    flat_cands = []
    for (pa, ea), cands in all_candidates.items():
        for pb, eb, ncc_val in cands:
            if frozenset({pa, pb}) not in used_pairs:
                flat_cands.append((ncc_val, pa, ea, pb, eb))
    flat_cands.sort(key=lambda x: -x[0])

    for ncc_val, pa, ea, pb, eb in flat_cands:
        key_a = (pa, ea)
        key_b = (pb, eb)
        pair_key = frozenset({pa, pb})
        if key_a in used_edges or key_b in used_edges or pair_key in used_pairs:
            continue
        used_edges.add(key_a)
        used_edges.add(key_b)
        used_pairs.add(pair_key)
        selected.append((pa, ea, pb, eb, ncc_val))

        # 如果已经选了 n-1 对（连成树），停止
        if len(selected) >= n - 1:
            break

    if len(selected) < n - 1:
        # 没选够，放宽重选
        return {'score': 0.0, 'message': '花纹匹配不足，无法拼接'}

    # 使用贪心结果构建简单的排列（这里返回边匹配信息，调用方自行定位）
    avg_ncc = float(np.mean([x[4] for x in selected]))
    sim_score = (avg_ncc + 1.0) / 2.0

    return {
        'score': round(sim_score, 4),
        'selected_edges': selected,
        'message': f'纯花纹匹配: {sim_score:.0%} ({len(selected)}对边)',
    }


# =============================================================================
# 【分区】可视化辅助
# 功能：draw_edge_profiles 目前返回图像副本，便于以后画采样带。
# 可修改：可在副本上画边和采样点做调试。
# 看情况改：无。
# 不要改：不要在原图上原地画，以免污染后续检测。
# =============================================================================
# ====================== 可视化辅助 ======================

# 该函数尚未实现红点和法向量绘制；循环中的poly为空列表，最终返回原图副本。注释按实际行为说明，不把占位实现描述为完整功能。
def draw_edge_profiles(
    image: np.ndarray,
    all_profiles: Dict[int, List[EdgeProfile]],
    ppm: float,
) -> np.ndarray:
    """
    在图像上画出边缘采样线（调试用）。
    红点=采样位置，绿线=法向量方向。
    """
    vis = image.copy()
    for pid, profiles in all_profiles.items():
        for ep in profiles:
            poly = []
            # 这个函数需要从 polygon 重建采样点，略
    return vis
