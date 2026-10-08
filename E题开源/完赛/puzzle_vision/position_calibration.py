"""XY搬运位置误差拟合工具。"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：XY 位置标定：拟合 actual = M @ command + b，以及把期望位移换成要发给电机的位移。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约46行  corrected_command_delta：发送侧补偿：matrix @ desired + bias
#   - 约89行  fit_position_model：最少 4 个有效样本的最小二乘拟合
# =============================================================================
# 【中文阅读导引】以下新增#注释解释代码用途；原有字符串、计算、默认值和执行顺序保持不变。
# 位置标定回答“让电机移动多少，纸面上实际移动多少”。模型为 actual = matrix @ command + bias。
# 2×2矩阵对角项表示X/Y比例误差，非对角项表示两轴耦合，二维bias表示固定偏置。
# fit_position_model对样本做最小二乘拟合；corrected_command_delta接收的是发送侧补偿矩阵，两者方向不同。
# 若实际模型为a=M c+b，希望位移为d，则应发送c=M⁻¹d-M⁻¹b；只有把逆矩阵和相应逆向偏置传入补偿函数才对应这个公式。
# 均方根RMS先把每个误差平方、取平均、再开方，反映模型在这些样本上的偏差量级。
# 语法约定：缩进决定代码归属；=赋值，==比较；列表和数组下标从0开始；None表示没有值；冒号后的类型主要用于阅读和检查。
# 跨行表达式属于同一条语句，括号结束前不会另起一条指令；字典条目和命名实参也在相邻注释中解释。
# =============================================================================
# 【分区】导入
# 功能：类型提示和 numpy 最小二乘拟合。
# 可修改：无现场参数。
# 看情况改：无。
# 不要改：from __future__ import annotations。
# =============================================================================
# 从 __future__ 导入需要的类型或工具。
from __future__ import annotations

# 从 typing 导入需要的类型或工具。
from typing import Any, Iterable

# 导入：numpy。
import numpy as np


# =============================================================================
# 【分区】发送侧 XY 补偿
# 功能：corrected = matrix @ desired + bias，把“希望纸面走多少”换成“命令电机走多少”。
# 可修改：0.005 mm 死区；更小位移当作 0，避免抖动空走。矩阵/偏置来自 config，不写死在本函数。
# 看情况改：矩阵必须是“命令侧补偿”（通常是实测模型的逆）。若把实测 M 直接塞进来，误差会加倍。
# 不要改：2×2 维度检查、不可逆报错、公式 matrix @ desired + bias。
# =============================================================================
# 【函数：corrected_command_delta】把期望位移转换为需要发送给控制器的XY位移。
# 参数 desired_delta（期望的二维位移）：可逐项遍历的浮点数列表/序列。
# 参数 command_matrix（将期望位移映射为发送位移的2×2补偿矩阵）：可逐项遍历的可逐项遍历的浮点数列表/序列列表/序列。
# 参数 command_bias_mm（发送位移的固定偏置（毫米））：可逐项遍历的浮点数列表/序列。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def corrected_command_delta(
    desired_delta: Iterable[float],
    command_matrix: Iterable[Iterable[float]],
    command_bias_mm: Iterable[float],
) -> np.ndarray:
    """把期望位移转换为需要发送给控制器的XY位移。"""
    # 计算并保存到 desired（期望）。
    desired = np.asarray(tuple(desired_delta), np.float64)
    # 计算并保存到 matrix（本函数使用的矩阵）。
    matrix = np.asarray(tuple(tuple(row) for row in command_matrix), np.float64)
    # 计算并保存到 bias（线性模型的常量偏置）。
    bias = np.asarray(tuple(command_bias_mm), np.float64)
    # 判断条件；满足时执行下面代码：desired.shape != (2,) or matrix.shape != (2, 2) or bias.shape != (2,)。
    if desired.shape != (2,) or matrix.shape != (2, 2) or bias.shape != (2,):
        # 抛出异常，通知上层处理：ValueError('XY补偿参数维度错误')。
        raise ValueError("XY补偿参数维度错误")
    # 判断条件；满足时执行下面代码：not np.all(np.isfinite(desired)) or not np.all(np.isfinite(matrix)) or (not np.…。
    if not np.all(np.isfinite(desired)) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias)):
        # 抛出异常，通知上层处理：ValueError('XY补偿参数必须是有限数')。
        raise ValueError("XY补偿参数必须是有限数")
    # 判断条件；满足时执行下面代码：abs(float(np.linalg.det(matrix))) < 1e-06。
    if abs(float(np.linalg.det(matrix))) < 1e-6:
        # 抛出异常，通知上层处理：ValueError('XY补偿矩阵不可逆')。
        raise ValueError("XY补偿矩阵不可逆")
    # 判断条件；满足时执行下面代码：float(np.linalg.norm(desired)) < 0.005。
    if float(np.linalg.norm(desired)) < 0.005:
        # 返回结果：按指定形状创建全0数组。
        return np.zeros(2, np.float64)
    # 返回结果：计算 matrix @ desired + bias。
    return matrix @ desired + bias


# =============================================================================
# 【分区】位置模型最小二乘拟合
# 功能：用有效样本拟合 actual = M @ command + b，并给出 RMS、逆矩阵、正反向误差统计。
# 可修改：最少 4 个有效样本；方向统计阈值 5 mm（太小的位移不参与正反向统计）。
# 看情况改：样本键名 command_dx_mm / actual_dx_mm 必须与标定工具 CSV 一致。
# 不要改：设计矩阵 [dx, dy, 1] 和 lstsq；改成无偏置模型会吃掉固定零点误差。
# =============================================================================
# 【函数：fit_position_model】拟合 actual_delta = matrix @ command_delta + bias。
# 设计矩阵每行是[command_dx,command_dy,1]，最后的1使拟合能求出固定偏置；lstsq同时拟合实际X和实际Y。
# 参数 samples（标定采样记录列表）：可逐项遍历的字典（键为字符串，值为任意类型）列表/序列。
# 返回类型：字典（键为字符串，值为任意类型）；箭头->是类型提示，不会替你转换实际返回值。
def fit_position_model(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """拟合 actual_delta = matrix @ command_delta + bias。"""
    # 计算并保存到 rows（记录集合）。
    rows = [row for row in samples if row.get("valid", True)]
    # 判断条件；满足时执行下面代码：len(rows) < 4。
    if len(rows) < 4:
        # 抛出异常，通知上层处理：ValueError('至少需要4个有效位置样本')。
        raise ValueError("至少需要4个有效位置样本")

    # 计算并保存到 command（当前命令或命令采样数据）。
    command = np.asarray(
        [[float(row["command_dx_mm"]), float(row["command_dy_mm"])] for row in rows],
        np.float64,
    )
    # 计算并保存到 actual（实测运动数据）。
    actual = np.asarray(
        [[float(row["actual_dx_mm"]), float(row["actual_dy_mm"])] for row in rows],
        np.float64,
    )
    # 计算并保存到 design（包含输入位移和常数列的设计矩阵）。
    design = np.column_stack([command, np.ones(len(rows), np.float64)])
    # 计算并保存到 coefficients（最小二乘拟合出的系数）、_（此处不需要使用的返回值或循环占位变量）、_（此处不需要使用的返回值或循环占位变量）、_（此处不需要使用的返回值或循环占位变量）。
    coefficients, _, _, _ = np.linalg.lstsq(design, actual, rcond=None)
    # 计算并保存到 matrix（本函数使用的矩阵）。
    matrix = coefficients[:2, :].T
    # 计算并保存到 bias（线性模型的常量偏置）。
    bias = coefficients[2, :]
    # 计算并保存到 predicted（模型预测数据）。
    predicted = design @ coefficients
    # 计算并保存到 residual（实测值与模型预测值的差）。
    residual = actual - predicted
    # 计算并保存到 rms_xy（均方根·XY平面）。
    rms_xy = np.sqrt(np.mean(np.square(residual), axis=0))
    # 计算并保存到 rms_distance（均方根·距离）。
    rms_distance = float(np.sqrt(np.mean(np.sum(np.square(residual), axis=1))))

    # 计算并保存到 determinant（行列式）。
    determinant = float(np.linalg.det(matrix))
    # 计算并保存到 inverse（逆变换）。
    inverse = np.linalg.inv(matrix) if abs(determinant) > 1e-9 else None

    # 【函数：direction_stats】分别统计X/Y正向或反向的大位移样本误差。
    # 参数 axis（选中的轴或数组维度）：整数。
    # 参数 positive（正向）：布尔值True/False。
    # 返回类型：字典（键为字符串，值为浮点数）或None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
    def direction_stats(axis: int, positive: bool) -> dict[str, float] | None:
        # 计算并保存到 selected（当前选中的数据）。
        selected = []
        # 计算并保存到 key（查询、分组或排序所用的键）。
        key = "command_dx_mm" if axis == 0 else "command_dy_mm"
        # 计算并保存到 error_key（误差·键）。
        error_key = "error_dx_mm" if axis == 0 else "error_dy_mm"
        # 遍历数据，逐项处理：rows。
        for row in rows:
            # 计算并保存到 value（当前数值）。
            value = float(row[key])
            # 判断条件；满足时执行下面代码：positive and value > 5.0 or (not positive and value < -5.0)。
            if (positive and value > 5.0) or (not positive and value < -5.0):
                # 调用函数：selected.append。
                selected.append(float(row[error_key]))
        # 判断条件；满足时执行下面代码：not selected。
        if not selected:
            # 返回结果：None。
            return None
        # 计算并保存到 values（当前数值集合）。
        values = np.asarray(selected, np.float64)
        # 返回结果：创建结果字典。
        return {
            # 字典字段'count'（计数值）：调用 int。
            "count": int(len(values)),
            # 字典字段'mean_error_mm'（平均·误差·毫米）：调用 float。
            "mean_error_mm": float(np.mean(values)),
            # 字典字段'spread_mm'（极差·毫米）：调用 float。
            "spread_mm": float(np.max(values) - np.min(values)),
        }

    # 返回结果：创建结果字典。
    return {
        # 字典字段'sample_count'（样本·数量）：调用 len。
        "sample_count": len(rows),
        # 字典字段'matrix'（本函数使用的矩阵）：调用 matrix.tolist。
        "matrix": matrix.tolist(),
        # 字典字段'bias_mm'（偏置·毫米）：调用 bias.tolist。
        "bias_mm": bias.tolist(),
        # 字典字段'inverse_matrix'（逆变换·矩阵）：inverse.tolist() if inverse is not None else None。
        "inverse_matrix": inverse.tolist() if inverse is not None else None,
        # 字典字段'determinant'（行列式）：determinant（行列式）。
        "determinant": determinant,
        # 字典字段'rms_xy_mm'（均方根·XY平面·毫米）：调用 rms_xy.tolist。
        "rms_xy_mm": rms_xy.tolist(),
        # 字典字段'rms_distance_mm'（均方根·距离·毫米）：rms_distance（均方根·距离）。
        "rms_distance_mm": rms_distance,
        # 字典字段'direction_error'（direction·误差）：创建结果字典。
        "direction_error": {
            # 字典字段'x_positive'（X轴·正向）：分别统计X/Y正向或反向的大位移样本误差。
            "x_positive": direction_stats(0, True),
            # 字典字段'x_negative'（X轴·反向）：分别统计X/Y正向或反向的大位移样本误差。
            "x_negative": direction_stats(0, False),
            # 字典字段'y_positive'（Y轴·正向）：分别统计X/Y正向或反向的大位移样本误差。
            "y_positive": direction_stats(1, True),
            # 字典字段'y_negative'（Y轴·反向）：分别统计X/Y正向或反向的大位移样本误差。
            "y_negative": direction_stats(1, False),
        },
    }
