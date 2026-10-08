"""旋转机构的视觉测量与线性标定。"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：旋转标定：轮廓 IoU 测相对转角，再拟合 actual = gain * command + bias。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约48行  _centered_polygon：轮廓移到画布中心，去掉平移
#   - 约85行  _rotate_clockwise：图像坐标正角=顺时针
#   - 约114行  estimate_relative_rotation：粗搜+细搜，IoU 最大的角
#   - 约187行  fit_rotation_model：直线拟合增益/偏置，并给正反转统计
# =============================================================================
# 【中文阅读导引】以下新增#注释解释代码用途；原有字符串、计算、默认值和执行顺序保持不变。
# 先用操作前后的碎片轮廓测角，再用多个样本拟合 actual = gain * command + bias。
# 测角时把两个轮廓各自移到画布中心，消除平移影响；先粗搜角度，再在最佳粗角附近细搜，以IoU最大者为估计值。
# 输入轮廓保持同一像素尺度；正角采用图像Y向下的顺时针约定。expected_deg限定搜索中心，避免对称碎片出现不相关角度。
# 软件比例建议1/gain只修正比例误差，不自动消除bias。实际还需结合正反转结果判断回差。
# 语法约定：缩进决定代码归属；=赋值，==比较；列表和数组下标从0开始；None表示没有值；冒号后的类型主要用于阅读和检查。
# 跨行表达式属于同一条语句，括号结束前不会另起一条指令；字典条目和命名实参也在相邻注释中解释。
# =============================================================================
# 【分区】导入
# 功能：OpenCV 填多边形、numpy 算 IoU 和直线拟合。
# 可修改：无。
# 看情况改：无。
# 不要改：from __future__ import annotations。
# =============================================================================
# 从 __future__ 导入需要的类型或工具。
from __future__ import annotations

# 从 typing 导入需要的类型或工具。
from typing import Any, Iterable

# 导入：cv2。
import cv2
# 导入：numpy。
import numpy as np


# =============================================================================
# 【分区】轮廓平移 / 栅格化 / 顺时针旋转（内部工具）
# 功能：把轮廓移到画布中心、填成二值图、按图像坐标顺时针旋转，专供测角使用。
# 可修改：canvas_size 默认 600；碎片很大或分辨率很高时可加大，更费时更准。
# 看情况改：旋转矩阵 [[c,-s],[s,c]] 对应“Y 向下、正角顺时针”。若改成数学坐标系，角度符号会反。
# 不要改：质心用 cv2.moments 的面积矩；不要改成外接矩形中心，测角会跟平移耦合。
# =============================================================================
# 【函数：_centered_polygon】将轮廓按面积质心平移到固定画布中心，便于只比较旋转。
# 参数 contour（碎片轮廓点数组）：NumPy数组。
# 参数 canvas_size（方形画布的边长（像素））：整数；省略时使用600。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def _centered_polygon(contour: np.ndarray, canvas_size: int = 600) -> np.ndarray:
    """将轮廓按面积质心平移到固定画布中心，便于只比较旋转。"""
    # 计算并保存到 points（参与当前计算的一组坐标点）。
    points = np.asarray(contour, np.float64).reshape(-1, 2)
    # 计算并保存到 moments（轮廓的图像矩，用于求面积质心）。
    moments = cv2.moments(points.astype(np.float32).reshape(-1, 1, 2))
    # 判断条件；满足时执行下面代码：abs(moments['m00']) < 1e-09。
    if abs(moments["m00"]) < 1e-9:
        # 抛出异常，通知上层处理：ValueError('轮廓面积为零')。
        raise ValueError("轮廓面积为零")
    # 计算并保存到 center（本步骤使用的中心位置）。
    center = np.asarray(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        np.float64,
    )
    # 返回结果：计算 points - center + np.asarray([canvas_size / 2.0, canvas_size / 2.0])。
    return points - center + np.asarray([canvas_size / 2.0, canvas_size / 2.0])


# 【函数：_rasterize】把多边形填到固定大小的二值画布上。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组。
# 参数 canvas_size（方形画布的边长（像素））：整数。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def _rasterize(points: np.ndarray, canvas_size: int) -> np.ndarray:
    # 计算并保存到 mask（二值掩膜（0为背景，非零为选中区域））。
    mask = np.zeros((canvas_size, canvas_size), np.uint8)
    # 调用函数：cv2.fillPoly。
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    # 返回结果：mask（二值掩膜（0为背景，非零为选中区域））。
    return mask


# 【函数：_rotate_clockwise】图像/A4坐标中正角为顺时针。
# 参数 points（参与当前计算的一组坐标点）：NumPy数组。
# 参数 angle_deg（角度，单位为度）：浮点数。
# 参数 center（本步骤使用的中心位置）：浮点数。
# 返回类型：NumPy数组；箭头->是类型提示，不会替你转换实际返回值。
def _rotate_clockwise(points: np.ndarray, angle_deg: float, center: float) -> np.ndarray:
    """图像/A4坐标中正角为顺时针。"""
    # 计算并保存到 radians（换算后的弧度角）。
    radians = np.deg2rad(angle_deg)
    # 计算并保存到 cosine（角度的余弦值）、sine（角度的正弦值）。
    cosine, sine = float(np.cos(radians)), float(np.sin(radians))
    # 计算并保存到 matrix（本函数使用的矩阵）。
    matrix = np.asarray([[cosine, -sine], [sine, cosine]], np.float64)
    # 返回结果：计算 (points - center) @ matrix.T + center。
    return (points - center) @ matrix.T + center


# =============================================================================
# 【分区】相对转角估计（粗搜 + 细搜，IoU 最大）
# 功能：在 expected_deg ± search_radius 内找使前后轮廓重合最好的顺时针角。
# 可修改：search_radius_deg（默认 45）、coarse_step_deg（0.5）、refine_step_deg（0.05）。
#         对称碎片务必给准 expected_deg，否则可能搜到无关的对称角。
# 看情况改：步长更小更准更慢；半径过大容易匹配到 180° 对称姿态。
# 不要改：先粗后细的两段搜索；不要改成只搜一次或改用最小外接矩形角（对异形片不准）。
# =============================================================================
# 【函数：estimate_relative_rotation】以轮廓IoU估计相对旋转，返回(正CW角度, IoU)。
# 参数 before_contour（操作前·轮廓）：NumPy数组。
# 参数 after_contour（操作后·轮廓）：NumPy数组。
# 参数 expected_deg（期望·度）：浮点数。
# 参数 search_radius_deg（搜索·radius·度）：浮点数；省略时使用45.0；必须按参数名传入。
# 参数 coarse_step_deg（粗搜索·步长·度）：浮点数；省略时使用0.5；必须按参数名传入。
# 参数 refine_step_deg（精细搜索·步长·度）：浮点数；省略时使用0.05；必须按参数名传入。
# 参数 canvas_size（方形画布的边长（像素））：整数；省略时使用600；必须按参数名传入。
# 返回类型：元组（依次为浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
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
    # 判断条件；满足时执行下面代码：coarse_step_deg <= 0.0 or refine_step_deg <= 0.0。
    if coarse_step_deg <= 0.0 or refine_step_deg <= 0.0:
        # 抛出异常，通知上层处理：ValueError('角度搜索步长必须大于0')。
        raise ValueError("角度搜索步长必须大于0")
    # 计算并保存到 before（操作前）。
    before = _centered_polygon(before_contour, canvas_size)
    # 计算并保存到 after（操作后）。
    after = _centered_polygon(after_contour, canvas_size)
    # 计算并保存到 after_mask（操作后·掩膜）。
    after_mask = _rasterize(after, canvas_size)
    # 计算并保存到 center（本步骤使用的中心位置）。
    center = canvas_size / 2.0

    # 【函数：score】计算当前试探旋转后的轮廓与实测轮廓的IoU。
    # 参数 angle（当前计算或搜索的角度）：浮点数。
    # 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
    def score(angle: float) -> float:
        # 计算并保存到 candidate（当前候选）。
        candidate = _rasterize(_rotate_clockwise(before, angle, center), canvas_size)
        # 计算并保存到 intersection（交集大小）。
        intersection = int(np.count_nonzero(cv2.bitwise_and(candidate, after_mask)))
        # 计算并保存到 union（并集大小）。
        union = int(np.count_nonzero(cv2.bitwise_or(candidate, after_mask)))
        # 返回结果：intersection / union if union else 0.0。
        return intersection / union if union else 0.0

    # 计算并保存到 start（起始值或起点）。
    start = expected_deg - search_radius_deg
    # 计算并保存到 stop。
    stop = expected_deg + search_radius_deg
    # 计算并保存到 coarse_angles（粗搜索·角度序列）。
    coarse_angles = np.arange(start, stop + coarse_step_deg * 0.5, coarse_step_deg)
    # 计算并保存到 coarse_scores（粗搜索·scores）。
    coarse_scores = [score(float(angle)) for angle in coarse_angles]
    # 计算并保存到 best_angle（最佳·角度）。
    best_angle = float(coarse_angles[int(np.argmax(coarse_scores))])

    # 计算并保存到 refine_angles（精细搜索·角度序列）。
    refine_angles = np.arange(
        best_angle - coarse_step_deg,
        best_angle + coarse_step_deg + refine_step_deg * 0.5,
        refine_step_deg,
    )
    # 计算并保存到 refine_scores（精细搜索·scores）。
    refine_scores = [score(float(angle)) for angle in refine_angles]
    # 计算并保存到 index（当前元素索引）。
    index = int(np.argmax(refine_scores))
    # 返回结果：创建数据容器。
    return float(refine_angles[index]), float(refine_scores[index])


# =============================================================================
# 【分区】旋转增益/偏置拟合
# 功能：actual = gain * command + bias；同时给出正转、反转子拟合和推荐软件比例 1/gain。
# 可修改：cycles_per_revolution 默认 164，应与主控一整圈脉冲/周期一致，只影响推荐值。
# 看情况改：software_scale=1/gain 只修比例，不消 bias。回差大时要看 positive/negative，不要只改一个增益。
# 不要改：至少 2 个有效样本；polyfit 一次直线；valid=False 的样本不参与拟合。
# =============================================================================
# 【函数：fit_rotation_model】拟合actual=gain*command+bias，并分别检查正反向。
# 参数 samples（标定采样记录列表）：可逐项遍历的字典（键为字符串，值为任意类型）列表/序列。
# 参数 cycles_per_revolution（周期数·每·整圈）：浮点数；省略时使用164.0。
# 返回类型：字典（键为字符串，值为任意类型）；箭头->是类型提示，不会替你转换实际返回值。
def fit_rotation_model(samples: Iterable[dict[str, Any]], cycles_per_revolution: float = 164.0) -> dict[str, Any]:
    """拟合actual=gain*command+bias，并分别检查正反向。"""
    # 计算并保存到 rows（记录集合）。
    rows = [row for row in samples if row.get("valid", True)]
    # 判断条件；满足时执行下面代码：len(rows) < 2。
    if len(rows) < 2:
        # 抛出异常，通知上层处理：ValueError('至少需要2个有效样本')。
        raise ValueError("至少需要2个有效样本")

    # 【函数：fit】用一次直线拟合命令角与实际角，计算增益、偏置及均方根误差。
    # 参数 group：字典（键为字符串，值为任意类型）列表/序列。
    # 返回类型：字典（键为字符串，值为浮点数）或None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
    def fit(group: list[dict[str, Any]]) -> dict[str, float] | None:
        # 判断条件；满足时执行下面代码：len(group) < 2。
        if len(group) < 2:
            # 返回结果：None。
            return None
        # 计算并保存到 command（当前命令或命令采样数据）。
        command = np.asarray([float(row["command_deg"]) for row in group], np.float64)
        # 计算并保存到 actual（实测运动数据）。
        actual = np.asarray([float(row["actual_deg"]) for row in group], np.float64)
        # 计算并保存到 gain（实际角度相对于命令角度的增益）、bias（线性模型的常量偏置）。
        gain, bias = np.polyfit(command, actual, 1)
        # 计算并保存到 predicted（模型预测数据）。
        predicted = gain * command + bias
        # 计算并保存到 rms（均方根误差）。
        rms = float(np.sqrt(np.mean(np.square(actual - predicted))))
        # 返回结果：创建结果字典。
        return {"gain": float(gain), "bias_deg": float(bias), "rms_deg": rms}

    # 计算并保存到 overall。
    overall = fit(rows)
    # 检查内部前提是否成立：overall is not None。
    assert overall is not None
    # 计算并保存到 positive（正向）。
    positive = fit([row for row in rows if float(row["command_deg"]) > 0.0])
    # 计算并保存到 negative（反向）。
    negative = fit([row for row in rows if float(row["command_deg"]) < 0.0])
    # 计算并保存到 gain（实际角度相对于命令角度的增益）。
    gain = overall["gain"]
    # 计算并保存到 software_scale（software·尺度）。
    software_scale = 1.0 / gain if abs(gain) > 1e-9 else float("nan")
    # 计算并保存到 recommended_cycles（recommended·周期数）。
    recommended_cycles = cycles_per_revolution / gain if abs(gain) > 1e-9 else float("nan")
    # 返回结果：创建结果字典。
    return {
        # 字典字段'sample_count'（样本·数量）：调用 len。
        "sample_count": len(rows),
        # 字典字段'overall'（overall）：overall。
        "overall": overall,
        # 字典字段'positive'（正向）：positive（正向）。
        "positive": positive,
        # 字典字段'negative'（反向）：negative（反向）。
        "negative": negative,
        # 字典字段'recommended_software_scale'（recommended·software·尺度）：调用 float。
        "recommended_software_scale": float(software_scale),
        # 字典字段'recommended_cycles_per_revolution'（recommended·周期数·每·整圈）：调用 float。
        "recommended_cycles_per_revolution": float(recommended_cycles),
    }
