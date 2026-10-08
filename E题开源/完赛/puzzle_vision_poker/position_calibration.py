"""XY搬运位置误差拟合工具。"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：与普通版相同：XY 位移模型拟合与发送侧补偿。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约35行  corrected_command_delta：期望位移 → 发送位移
#   - 约63行  fit_position_model：最小二乘拟合
# =============================================================================
# 位置标定回答“让电机移动多少，纸面上实际移动多少”。模型为 actual = matrix @ command + bias。
# 2×2矩阵对角项表示X/Y比例误差，非对角项表示两轴耦合，二维bias表示固定偏置。
# fit_position_model对样本做最小二乘拟合；corrected_command_delta接收的是发送侧补偿矩阵，两者方向不同。
# 若实际模型为a=M c+b，希望位移为d，则应发送c=M⁻¹d-M⁻¹b；只有把逆矩阵和相应逆向偏置传入补偿函数才对应这个公式。
# 均方根RMS先把每个误差平方、取平均、再开方，反映模型在这些样本上的偏差量级。
# =============================================================================
# 【分区】导入
# 功能：numpy 最小二乘拟合。扑克版与普通版相同。
# 可修改：无。
# 看情况改：无。
# 不要改：from __future__ import annotations。
# =============================================================================
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


# =============================================================================
# 【分区】发送侧 XY 补偿
# 功能：corrected = matrix @ desired + bias。
# 可修改：0.005 mm 死区。矩阵/偏置来自 config。
# 看情况改：必须传入命令侧补偿（通常是实测模型的逆），不要把实测 M 直接塞进来。
# 不要改：2×2 检查、不可逆报错、公式本身。
# =============================================================================
def corrected_command_delta(
    desired_delta: Iterable[float],
    command_matrix: Iterable[Iterable[float]],
    command_bias_mm: Iterable[float],
) -> np.ndarray:
    """把期望位移转换为需要发送给控制器的XY位移。"""
    desired = np.asarray(tuple(desired_delta), np.float64)
    matrix = np.asarray(tuple(tuple(row) for row in command_matrix), np.float64)
    bias = np.asarray(tuple(command_bias_mm), np.float64)
    if desired.shape != (2,) or matrix.shape != (2, 2) or bias.shape != (2,):
        raise ValueError("XY补偿参数维度错误")
    if not np.all(np.isfinite(desired)) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias)):
        raise ValueError("XY补偿参数必须是有限数")
    if abs(float(np.linalg.det(matrix))) < 1e-6:
        raise ValueError("XY补偿矩阵不可逆")
    if float(np.linalg.norm(desired)) < 0.005:
        return np.zeros(2, np.float64)
    return matrix @ desired + bias


# 设计矩阵每行是[command_dx,command_dy,1]，最后的1使拟合能求出固定偏置；lstsq同时拟合实际X和实际Y。
# =============================================================================
# 【分区】位置模型最小二乘拟合
# 功能：actual = M @ command + b，并给出 RMS、逆矩阵、正反向误差。
# 可修改：至少 4 个有效样本；方向统计阈值 5 mm。
# 看情况改：样本键名必须与标定工具 CSV 一致。
# 不要改：设计矩阵 [dx, dy, 1] 和 lstsq。
# =============================================================================
def fit_position_model(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """拟合 actual_delta = matrix @ command_delta + bias。"""
    rows = [row for row in samples if row.get("valid", True)]
    if len(rows) < 4:
        raise ValueError("至少需要4个有效位置样本")

    command = np.asarray(
        [[float(row["command_dx_mm"]), float(row["command_dy_mm"])] for row in rows],
        np.float64,
    )
    actual = np.asarray(
        [[float(row["actual_dx_mm"]), float(row["actual_dy_mm"])] for row in rows],
        np.float64,
    )
    design = np.column_stack([command, np.ones(len(rows), np.float64)])
    coefficients, _, _, _ = np.linalg.lstsq(design, actual, rcond=None)
    matrix = coefficients[:2, :].T
    bias = coefficients[2, :]
    predicted = design @ coefficients
    residual = actual - predicted
    rms_xy = np.sqrt(np.mean(np.square(residual), axis=0))
    rms_distance = float(np.sqrt(np.mean(np.sum(np.square(residual), axis=1))))

    determinant = float(np.linalg.det(matrix))
    inverse = np.linalg.inv(matrix) if abs(determinant) > 1e-9 else None

    def direction_stats(axis: int, positive: bool) -> dict[str, float] | None:
        selected = []
        key = "command_dx_mm" if axis == 0 else "command_dy_mm"
        error_key = "error_dx_mm" if axis == 0 else "error_dy_mm"
        for row in rows:
            value = float(row[key])
            if (positive and value > 5.0) or (not positive and value < -5.0):
                selected.append(float(row[error_key]))
        if not selected:
            return None
        values = np.asarray(selected, np.float64)
        return {
            "count": int(len(values)),
            "mean_error_mm": float(np.mean(values)),
            "spread_mm": float(np.max(values) - np.min(values)),
        }

    return {
        "sample_count": len(rows),
        "matrix": matrix.tolist(),
        "bias_mm": bias.tolist(),
        "inverse_matrix": inverse.tolist() if inverse is not None else None,
        "determinant": determinant,
        "rms_xy_mm": rms_xy.tolist(),
        "rms_distance_mm": rms_distance,
        "direction_error": {
            "x_positive": direction_stats(0, True),
            "x_negative": direction_stats(0, False),
            "y_positive": direction_stats(1, True),
            "y_negative": direction_stats(1, False),
        },
    }
