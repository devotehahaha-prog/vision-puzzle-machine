"""旋转机构的视觉测量与线性标定。"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：与普通版相同：轮廓 IoU 测角 + 增益/偏置拟合。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约69行  estimate_relative_rotation：相对转角
#   - 约116行  fit_rotation_model：gain/bias 拟合
# =============================================================================
# 先用操作前后的碎片轮廓测角，再用多个样本拟合 actual = gain * command + bias。
# 测角时把两个轮廓各自移到画布中心，消除平移影响；先粗搜角度，再在最佳粗角附近细搜，以IoU最大者为估计值。
# 输入轮廓保持同一像素尺度；正角采用图像Y向下的顺时针约定。expected_deg限定搜索中心，避免对称碎片出现不相关角度。
# 软件比例建议1/gain只修正比例误差，不自动消除bias。实际还需结合正反转结果判断回差。
# =============================================================================
# 【分区】导入
# 功能：OpenCV 填多边形、numpy 算 IoU 和直线拟合。扑克版与普通版相同。
# 可修改：无。
# 看情况改：无。
# 不要改：from __future__ import annotations。
# =============================================================================
from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np


# =============================================================================
# 【分区】轮廓平移 / 栅格化 / 顺时针旋转（内部工具）
# 功能：移到画布中心、填二值图、图像坐标顺时针旋转，专供测角。
# 可修改：canvas_size 默认 600。
# 看情况改：旋转矩阵对应 Y 向下、正角顺时针；改成数学坐标系则角度符号反。
# 不要改：质心用面积矩，不要改成外接矩形中心。
# =============================================================================
def _centered_polygon(contour: np.ndarray, canvas_size: int = 600) -> np.ndarray:
    """将轮廓按面积质心平移到固定画布中心，便于只比较旋转。"""
    points = np.asarray(contour, np.float64).reshape(-1, 2)
    moments = cv2.moments(points.astype(np.float32).reshape(-1, 1, 2))
    if abs(moments["m00"]) < 1e-9:
        raise ValueError("轮廓面积为零")
    center = np.asarray(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        np.float64,
    )
    return points - center + np.asarray([canvas_size / 2.0, canvas_size / 2.0])


def _rasterize(points: np.ndarray, canvas_size: int) -> np.ndarray:
    mask = np.zeros((canvas_size, canvas_size), np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask


def _rotate_clockwise(points: np.ndarray, angle_deg: float, center: float) -> np.ndarray:
    """图像/A4坐标中正角为顺时针。"""
    radians = np.deg2rad(angle_deg)
    cosine, sine = float(np.cos(radians)), float(np.sin(radians))
    matrix = np.asarray([[cosine, -sine], [sine, cosine]], np.float64)
    return (points - center) @ matrix.T + center


# =============================================================================
# 【分区】相对转角估计（粗搜 + 细搜，IoU 最大）
# 功能：在 expected_deg ± search_radius 内找前后轮廓重合最好的顺时针角。
# 可修改：search_radius_deg=45、coarse_step=0.5、refine_step=0.05。
# 看情况改：对称碎片务必给准 expected_deg。
# 不要改：先粗后细；不要只用最小外接矩形角。
# =============================================================================
def estimate_relative_rotation(
    before_contour: np.ndarray,
    after_contour: np.ndarray,
    expected_deg: float,
    *,
    search_radius_deg: float = 45.0,
    coarse_step_deg: float = 0.5,
    refine_step_deg: float = 0.05,
    canvas_size: int = 600,
) -> tuple[float, float]:
    """以轮廓IoU估计相对旋转，返回(正CW角度, IoU)。"""
    if coarse_step_deg <= 0.0 or refine_step_deg <= 0.0:
        raise ValueError("角度搜索步长必须大于0")
    before = _centered_polygon(before_contour, canvas_size)
    after = _centered_polygon(after_contour, canvas_size)
    after_mask = _rasterize(after, canvas_size)
    center = canvas_size / 2.0

    def score(angle: float) -> float:
        candidate = _rasterize(_rotate_clockwise(before, angle, center), canvas_size)
        intersection = int(np.count_nonzero(cv2.bitwise_and(candidate, after_mask)))
        union = int(np.count_nonzero(cv2.bitwise_or(candidate, after_mask)))
        return intersection / union if union else 0.0

    start = expected_deg - search_radius_deg
    stop = expected_deg + search_radius_deg
    coarse_angles = np.arange(start, stop + coarse_step_deg * 0.5, coarse_step_deg)
    coarse_scores = [score(float(angle)) for angle in coarse_angles]
    best_angle = float(coarse_angles[int(np.argmax(coarse_scores))])

    refine_angles = np.arange(
        best_angle - coarse_step_deg,
        best_angle + coarse_step_deg + refine_step_deg * 0.5,
        refine_step_deg,
    )
    refine_scores = [score(float(angle)) for angle in refine_angles]
    index = int(np.argmax(refine_scores))
    return float(refine_angles[index]), float(refine_scores[index])


# =============================================================================
# 【分区】旋转增益/偏置拟合
# 功能：actual = gain * command + bias；推荐软件比例 1/gain。
# 可修改：cycles_per_revolution 默认 164，应与主控整圈周期一致。
# 看情况改：1/gain 只修比例。回差大时看正/反转子拟合，不要只改一个增益。
# 不要改：至少 2 个有效样本；valid=False 不参与拟合。
# =============================================================================
def fit_rotation_model(samples: Iterable[dict[str, Any]], cycles_per_revolution: float = 164.0) -> dict[str, Any]:
    """拟合actual=gain*command+bias，并分别检查正反向。"""
    rows = [row for row in samples if row.get("valid", True)]
    if len(rows) < 2:
        raise ValueError("至少需要2个有效样本")

    def fit(group: list[dict[str, Any]]) -> dict[str, float] | None:
        if len(group) < 2:
            return None
        command = np.asarray([float(row["command_deg"]) for row in group], np.float64)
        actual = np.asarray([float(row["actual_deg"]) for row in group], np.float64)
        gain, bias = np.polyfit(command, actual, 1)
        predicted = gain * command + bias
        rms = float(np.sqrt(np.mean(np.square(actual - predicted))))
        return {"gain": float(gain), "bias_deg": float(bias), "rms_deg": rms}

    overall = fit(rows)
    # 检查内部前提是否成立：overall is not None。
    assert overall is not None
    positive = fit([row for row in rows if float(row["command_deg"]) > 0.0])
    negative = fit([row for row in rows if float(row["command_deg"]) < 0.0])
    gain = overall["gain"]
    software_scale = 1.0 / gain if abs(gain) > 1e-9 else float("nan")
    recommended_cycles = cycles_per_revolution / gain if abs(gain) > 1e-9 else float("nan")
    return {
        "sample_count": len(rows),
        "overall": overall,
        "positive": positive,
        "negative": negative,
        "recommended_software_scale": float(software_scale),
        "recommended_cycles_per_revolution": float(recommended_cycles),
    }
