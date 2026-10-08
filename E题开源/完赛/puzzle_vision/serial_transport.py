"""将视觉运动计划转换为安全、可确认的串口ASCII指令。"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：把 plan.json 校验后变成 ASCII 串口命令，逐条发送并等 ACK；失败则 STOP / MOTION,0。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约52行  SerialPlanError：计划或协议错误
#   - 约144行  _execution_compensation：读 XY/旋转补偿
#   - 约373行  _split_rotation：单次转角超上限则分段：放下、轴回正、再吸
#   - 约468行  load_motion_plan：读 plan.json
#   - 约495行  validate_motion_plan：检查坐标系、正交路径、move_order、ready 标志
#   - 约663行  build_serial_commands：生成 GOTO/Z/MAGNET/ROTATE 命令列表（不开串口）
#   - 约831行  print_serial_commands：dry-run 打印
#   - 约859行  _read_response：读一行，必须是 ACK
#   - 约920行  _best_effort_safety_shutdown：异常时尽力停机
#   - 约979行  send_serial_commands：开串口逐条发并等待
#   - 约1216行  send_plan_file：对外入口：校验→生成→打印或实发
# =============================================================================
# 【中文阅读导引】以下新增#注释解释代码用途；原有字符串、计算、默认值和执行顺序保持不变。
# 输入主程序生成的plan.json；先验证字段和路径，再生成按顺序执行的ASCII命令，最后逐条发送并等待ACK。
# 阅读顺序：send_plan_file → load_motion_plan → validate_motion_plan → build_serial_commands → send_serial_commands。
# 协议中的GOTO给出绝对XY坐标，MOVE指定单轴相对位移，ROTATE给出有符号角度；距离单位是毫米，角度单位是度。
# Z,DOWN/UP控制下降/抬升；MAGNET,1/0控制吸取/释放；MOTION,1/0开关运动使能。这些行首先是字符串，真正write时才发给主控。
# PING用于检查通信；ACK表示主控接受/确认当前命令。超时、ERR或中断进入异常收尾，尝试STOP和MOTION,0，finally关闭连接。
# dry_run只打印命令，不连接串口；它仍执行计划校验，所以不合格计划也会被拒绝。
# target_offset_mm虽然名字带target，现有实现把它作为视觉到磁铁的工具偏移，同时加在源点和目标点。
# 语法约定：缩进决定代码归属；=赋值，==比较；列表和数组下标从0开始；None表示没有值；冒号后的类型主要用于阅读和检查。
# 跨行表达式属于同一条语句，括号结束前不会另起一条指令；字典条目和命名实参也在相邻注释中解释。
# =============================================================================
# 【分区】导入与异常类型
# 功能：计划校验失败、协议错误统一抛 SerialPlanError（ValueError 子类）。
# 可修改：无。
# 看情况改：无。
# 不要改：异常类型名；上层 except SerialPlanError 依赖它。
# =============================================================================
# 从 __future__ 导入需要的类型或工具。
from __future__ import annotations

# 导入：json。
import json
# 导入：math。
import math
# 导入：time。
import time
# 从 pathlib 导入需要的类型或工具。
from pathlib import Path
# 从 typing 导入需要的类型或工具。
from typing import Any, Callable, Iterable


# 【类：SerialPlanError】运动计划或串口协议不满足安全要求。
# 继承ValueError，保留父类行为并补充下方字段/方法。
class SerialPlanError(ValueError):
    """运动计划或串口协议不满足安全要求。"""


# =============================================================================
# 【分区】数值/点/矩阵校验与格式化
# 功能：拒绝 NaN/Inf；XY 点、2×2 矩阵检查；命令数字统一两位小数。
# 可修改：输出小数位 .2f（与主控解析一致再改）。
# 看情况改：无。
# 不要改：非有限数必须抛 SerialPlanError，否则电机会收到非法 GOTO。
# =============================================================================
# 【函数：_finite_number】将输入转换为有限浮点数，拒绝无效值、NaN和无穷值。
# 参数 value（当前数值）：任意类型。
# 参数 name（名称）：字符串。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def _finite_number(value: Any, name: str) -> float:
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 计算并保存到 result（当前步骤得到的结果）。
        result = float(value)
    # 捕获(TypeError, ValueError)异常，转入下面的处理代码；异常对象保存在exc。
    except (TypeError, ValueError) as exc:
        # 抛出异常，通知上层处理：SerialPlanError(f'{name}不是有效数字')。
        raise SerialPlanError(f"{name}不是有效数字") from exc
    # 判断条件；满足时执行下面代码：not math.isfinite(result)。
    if not math.isfinite(result):
        # 抛出异常，通知上层处理：SerialPlanError(f'{name}不是有限数字')。
        raise SerialPlanError(f"{name}不是有限数字")
    # 返回结果：result（当前步骤得到的结果）。
    return result


# 【函数：_point】校验并提取一对二维数值。
# 参数 value（当前数值）：任意类型。
# 参数 name（名称）：字符串。
# 返回类型：元组（依次为浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def _point(value: Any, name: str) -> tuple[float, float]:
    # 判断条件；满足时执行下面代码：not isinstance(value, (list, tuple)) or len(value) != 2。
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        # 抛出异常，通知上层处理：SerialPlanError(f'{name}必须是[x,y]')。
        raise SerialPlanError(f"{name}必须是[x,y]")
    # 返回结果：创建数据容器。
    return _finite_number(value[0], f"{name}.x"), _finite_number(value[1], f"{name}.y")


# 【函数：_matrix2x2】校验并提取二维位移补偿矩阵。
# 参数 value（当前数值）：任意类型。
# 参数 name（名称）：字符串。
# 返回类型：元组（依次为元组（依次为浮点数、浮点数）、元组（依次为浮点数、浮点数））；箭头->是类型提示，不会替你转换实际返回值。
def _matrix2x2(value: Any, name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    # 判断条件；满足时执行下面代码：not isinstance(value, (list, tuple)) or len(value) != 2。
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        # 抛出异常，通知上层处理：SerialPlanError(f'{name}必须是2x2矩阵')。
        raise SerialPlanError(f"{name}必须是2x2矩阵")
    # 计算并保存到 row0。
    row0 = _point(value[0], f"{name}[0]")
    # 计算并保存到 row1。
    row1 = _point(value[1], f"{name}[1]")
    # 计算并保存到 determinant（行列式）。
    determinant = row0[0] * row1[1] - row0[1] * row1[0]
    # 判断条件；满足时执行下面代码：abs(determinant) < 1e-06。
    if abs(determinant) < 1e-6:
        # 抛出异常，通知上层处理：SerialPlanError(f'{name}不可逆')。
        raise SerialPlanError(f"{name}不可逆")
    # 返回结果：创建数据容器。
    return row0, row1


# 【函数：_format_number】将数值格式化为串口协议使用的两位小数文本，并处理接近零的值。
# 参数 value（当前数值）：浮点数。
# 返回类型：字符串；箭头->是类型提示，不会替你转换实际返回值。
def _format_number(value: float) -> str:
    # 计算并保存到 rounded。
    rounded = round(float(value), 2)
    # 判断条件；满足时执行下面代码：abs(rounded) < 0.005。
    if abs(rounded) < 0.005:
        # 计算并保存到 rounded。
        rounded = 0.0
    # 返回结果：f'{rounded:.2f}'。
    return f"{rounded:.2f}"


# =============================================================================
# 【分区】执行补偿、时序与旋转拆分
# 功能：读 offset/矩阵/旋转比例；XY 位移补偿；单次旋转超上限则放下回正再吸，分段累加。
# 可修改：补偿值来自 plan.json 的 execution_compensation（由 config 写入），不要写死在本文件。
# 看情况改：target_offset_mm 名字带 target，实现里源点和目标都加，它是“视觉点→磁铁”工具偏移。
# 不要改：旋转拆分时必须先放下再回正轴，否则线束拧死或碎片跟着轴转回去。
# =============================================================================
# 【函数：_execution_compensation】读取执行层位置与旋转补偿；旧计划默认零补偿。
# 参数 plan（从JSON读取的运动计划字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为浮点数、浮点数、浮点数、浮点数、浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def _execution_compensation(plan: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """读取执行层位置与旋转补偿；旧计划默认零补偿。"""
    # 计算并保存到 compensation（运动补偿配置）。
    compensation = plan.get("execution_compensation", {})
    # 判断条件；满足时执行下面代码：compensation is None。
    if compensation is None:
        # 计算并保存到 compensation（运动补偿配置）。
        compensation = {}
    # 判断条件；满足时执行下面代码：not isinstance(compensation, dict)。
    if not isinstance(compensation, dict):
        # 抛出异常，通知上层处理：SerialPlanError('execution_compensation必须是JSON对象')。
        raise SerialPlanError("execution_compensation必须是JSON对象")
    # 计算并保存到 offset_x（偏移·X轴）、offset_y（偏移·Y轴）。
    offset_x, offset_y = _point(
        compensation.get("target_offset_mm", [0.0, 0.0]),
        "execution_compensation.target_offset_mm",
    )
    # 计算并保存到 rotation_reduction（旋转·减量）。
    rotation_reduction = _finite_number(
        compensation.get("rotation_magnitude_reduction_deg", 0.0),
        "execution_compensation.rotation_magnitude_reduction_deg",
    )
    # 判断条件；满足时执行下面代码：rotation_reduction < 0.0。
    if rotation_reduction < 0.0:
        # 抛出异常，通知上层处理：SerialPlanError('旋转补偿减少量不能为负数')。
        raise SerialPlanError("旋转补偿减少量不能为负数")
    # 计算并保存到 rotation_reduction_min_angle（旋转·减量·最小·角度）。
    rotation_reduction_min_angle = _finite_number(
        compensation.get("rotation_reduction_min_angle_deg", 0.0),
        "execution_compensation.rotation_reduction_min_angle_deg",
    )
    # 判断条件；满足时执行下面代码：rotation_reduction_min_angle < 0.0 or rotation_reduction_min_angle > 180.0。
    if rotation_reduction_min_angle < 0.0 or rotation_reduction_min_angle > 180.0:
        # 抛出异常，通知上层处理：SerialPlanError('旋转补偿启用角必须在0～180度之间')。
        raise SerialPlanError("旋转补偿启用角必须在0～180度之间")
    # 计算并保存到 rotation_chunk（旋转·分段）。
    rotation_chunk = _finite_number(
        compensation.get("rotation_chunk_deg", 180.0),
        "execution_compensation.rotation_chunk_deg",
    )
    # 判断条件；满足时执行下面代码：rotation_chunk <= 0.0 or rotation_chunk > 180.0。
    if rotation_chunk <= 0.0 or rotation_chunk > 180.0:
        # 抛出异常，通知上层处理：SerialPlanError('旋转分段角必须在0～180度之间')。
        raise SerialPlanError("旋转分段角必须在0～180度之间")
    # 计算并保存到 rotation_cycles（旋转·周期数）。
    rotation_cycles = _finite_number(
        compensation.get("rotation_cycles_per_revolution", 0.0),
        "execution_compensation.rotation_cycles_per_revolution",
    )
    # 判断条件；满足时执行下面代码：rotation_cycles < 0.0。
    if rotation_cycles < 0.0:
        # 抛出异常，通知上层处理：SerialPlanError('旋转电机每圈周期数不能为负数')。
        raise SerialPlanError("旋转电机每圈周期数不能为负数")
    # 返回结果：创建数据容器。
    return (
        offset_x, offset_y, rotation_reduction, rotation_reduction_min_angle,
        rotation_chunk, rotation_cycles,
    )


# 【函数：_xy_delta_compensation】读取期望XY位移到实际发送位移的线性补偿。
# 参数 plan（从JSON读取的运动计划字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为元组（依次为浮点数、浮点数）、元组（依次为浮点数、浮点数）、元组（依次为浮点数、浮点数））；箭头->是类型提示，不会替你转换实际返回值。
def _xy_delta_compensation(
    plan: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """读取期望XY位移到实际发送位移的线性补偿。"""
    # 计算并保存到 compensation（运动补偿配置）。
    compensation = plan.get("execution_compensation", {})
    # 判断条件；满足时执行下面代码：compensation is None。
    if compensation is None:
        # 计算并保存到 compensation（运动补偿配置）。
        compensation = {}
    # 判断条件；满足时执行下面代码：not isinstance(compensation, dict)。
    if not isinstance(compensation, dict):
        # 抛出异常，通知上层处理：SerialPlanError('execution_compensation必须是JSON对象')。
        raise SerialPlanError("execution_compensation必须是JSON对象")
    # 计算并保存到 matrix（本函数使用的矩阵）。
    matrix = _matrix2x2(
        compensation.get("xy_command_matrix", [[1.0, 0.0], [0.0, 1.0]]),
        "execution_compensation.xy_command_matrix",
    )
    # 计算并保存到 bias（线性模型的常量偏置）。
    bias = _point(
        compensation.get("xy_command_bias_mm", [0.0, 0.0]),
        "execution_compensation.xy_command_bias_mm",
    )
    # 返回结果：创建数据容器。
    return matrix[0], matrix[1], bias


# 【函数：_execution_timing】读取吸取、释放、抬升和旋转后的稳定等待；旧计划默认不等待。
# 参数 plan（从JSON读取的运动计划字典）：字典（键为字符串，值为任意类型）。
# 返回类型：元组（依次为浮点数、浮点数、浮点数、浮点数、浮点数）；箭头->是类型提示，不会替你转换实际返回值。
def _execution_timing(plan: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """读取吸取、释放、抬升和旋转后的稳定等待；旧计划默认不等待。"""
    # 计算并保存到 timing（机构稳定等待时间配置）。
    timing = plan.get("execution_timing", {})
    # 判断条件；满足时执行下面代码：timing is None。
    if timing is None:
        # 计算并保存到 timing（机构稳定等待时间配置）。
        timing = {}
    # 判断条件；满足时执行下面代码：not isinstance(timing, dict)。
    if not isinstance(timing, dict):
        # 抛出异常，通知上层处理：SerialPlanError('execution_timing必须是JSON对象')。
        raise SerialPlanError("execution_timing必须是JSON对象")
    # 计算并保存到 fields。
    fields = (
        "magnet_pickup_settle_seconds",
        "magnet_release_settle_seconds",
        "lift_settle_seconds",
        "xy_settle_seconds",
        "rotation_settle_seconds",
    )
    # 计算并保存到 values（当前数值集合）。
    values = tuple(
        _finite_number(timing.get(key, 0.0), f"execution_timing.{key}")
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for key in fields
    )
    # 判断条件；满足时执行下面代码：any((value < 0.0 for value in values))。
    if any(value < 0.0 for value in values):
        # 抛出异常，通知上层处理：SerialPlanError('执行稳定等待时间不能为负数')。
        raise SerialPlanError("执行稳定等待时间不能为负数")
    # 返回结果：values（当前数值集合）。
    return values


# 【函数：_return_origin_after_plan】读取计划结束后是否回到逻辑原点；旧计划默认保持当前位置。
# 参数 plan（从JSON读取的运动计划字典）：字典（键为字符串，值为任意类型）。
# 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
def _return_origin_after_plan(plan: dict[str, Any]) -> bool:
    """读取计划结束后是否回到逻辑原点；旧计划默认保持当前位置。"""
    # 计算并保存到 policy（运动执行策略配置）。
    policy = plan.get("execution_policy", {})
    # 判断条件；满足时执行下面代码：policy is None。
    if policy is None:
        # 计算并保存到 policy（运动执行策略配置）。
        policy = {}
    # 判断条件；满足时执行下面代码：not isinstance(policy, dict)。
    if not isinstance(policy, dict):
        # 抛出异常，通知上层处理：SerialPlanError('execution_policy必须是JSON对象')。
        raise SerialPlanError("execution_policy必须是JSON对象")
    # 计算并保存到 value（当前数值）。
    value = policy.get("return_origin_after_plan", False)
    # 判断条件；满足时执行下面代码：not isinstance(value, bool)。
    if not isinstance(value, bool):
        # 抛出异常，通知上层处理：SerialPlanError('execution_policy.return_origin_after_plan必须是布尔值')。
        raise SerialPlanError("execution_policy.return_origin_after_plan必须是布尔值")
    # 返回结果：value（当前数值）。
    return value


# 【函数：_rotate_at_target】读取是否先移动到目标上方再旋转；旧计划默认沿用源区旋转。
# 参数 plan（从JSON读取的运动计划字典）：字典（键为字符串，值为任意类型）。
# 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
def _rotate_at_target(plan: dict[str, Any]) -> bool:
    """读取是否先移动到目标上方再旋转；旧计划默认沿用源区旋转。"""
    # 计算并保存到 policy（运动执行策略配置）。
    policy = plan.get("execution_policy", {})
    # 判断条件；满足时执行下面代码：policy is None。
    if policy is None:
        # 计算并保存到 policy（运动执行策略配置）。
        policy = {}
    # 判断条件；满足时执行下面代码：not isinstance(policy, dict)。
    if not isinstance(policy, dict):
        # 抛出异常，通知上层处理：SerialPlanError('execution_policy必须是JSON对象')。
        raise SerialPlanError("execution_policy必须是JSON对象")
    # 计算并保存到 value（当前数值）。
    value = policy.get("rotate_at_target", False)
    # 判断条件；满足时执行下面代码：not isinstance(value, bool)。
    if not isinstance(value, bool):
        # 抛出异常，通知上层处理：SerialPlanError('execution_policy.rotate_at_target必须是布尔值')。
        raise SerialPlanError("execution_policy.rotate_at_target必须是布尔值")
    # 返回结果：value（当前数值）。
    return value


# 【函数：_require_origin_before_plan】读取真实执行前是否必须验证XY逻辑原点。
# 参数 plan（从JSON读取的运动计划字典）：字典（键为字符串，值为任意类型）。
# 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
def _require_origin_before_plan(plan: dict[str, Any]) -> bool:
    """读取真实执行前是否必须验证XY逻辑原点。"""
    # 计算并保存到 policy（运动执行策略配置）。
    policy = plan.get("execution_policy", {})
    # 判断条件；满足时执行下面代码：policy is None。
    if policy is None:
        # 计算并保存到 policy（运动执行策略配置）。
        policy = {}
    # 判断条件；满足时执行下面代码：not isinstance(policy, dict)。
    if not isinstance(policy, dict):
        # 抛出异常，通知上层处理：SerialPlanError('execution_policy必须是JSON对象')。
        raise SerialPlanError("execution_policy必须是JSON对象")
    # 计算并保存到 value（当前数值）。
    value = policy.get("require_origin_before_plan", False)
    # 判断条件；满足时执行下面代码：not isinstance(value, bool)。
    if not isinstance(value, bool):
        # 抛出异常，通知上层处理：SerialPlanError('execution_policy.require_origin_before_plan必须是布尔值')。
        raise SerialPlanError("execution_policy.require_origin_before_plan必须是布尔值")
    # 返回结果：value（当前数值）。
    return value


# 【函数：_reduce_rotation_magnitude】仅对达到启用角的大角度减小绝对值，并保持CW/CCW方向。
# 参数 rotation（旋转数据（角度或旋转矩阵，取决于本函数））：浮点数。
# 参数 reduction（减量）：浮点数。
# 参数 min_angle（最小·角度）：浮点数；省略时使用0.0。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def _reduce_rotation_magnitude(
        rotation: float, reduction: float, min_angle: float = 0.0) -> float:
    """仅对达到启用角的大角度减小绝对值，并保持CW/CCW方向。"""
    # 判断条件；满足时执行下面代码：abs(rotation) < min_angle。
    if abs(rotation) < min_angle:
        # 返回结果：rotation（旋转数据（角度或旋转矩阵，取决于本函数））。
        return rotation
    # 判断条件；满足时执行下面代码：abs(rotation) <= reduction。
    if abs(rotation) <= reduction:
        # 返回结果：0.0。
        return 0.0
    # 返回结果：取第一个数的绝对值，并使用第二个数的正负号。
    return math.copysign(abs(rotation) - reduction, rotation)


# 【函数：_split_rotation】按旋转机构实际步距均匀分段，避免末段过小。
# 有整圈周期数时先把总角量化为整数周期，再将周期均匀分段。这样不会因为每段独立四舍五入而累积多转或少转。
# 参数 rotation（旋转数据（角度或旋转矩阵，取决于本函数））：浮点数。
# 参数 chunk_limit（分段·上限）：浮点数。
# 参数 cycles_per_revolution（周期数·每·整圈）：浮点数；省略时使用0.0。
# 返回类型：浮点数列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def _split_rotation(
        rotation: float, chunk_limit: float,
        cycles_per_revolution: float = 0.0) -> list[float]:
    """按旋转机构实际步距均匀分段，避免末段过小。"""
    # 判断条件；满足时执行下面代码：abs(rotation) < 0.005。
    if abs(rotation) < 0.005:
        # 返回结果：创建数据容器。
        return [0.0]

    # 计算并保存到 direction（方向标记或方向向量）。
    direction = math.copysign(1.0, rotation)
    # 判断条件；满足时执行下面代码：cycles_per_revolution <= 0.0。
    if cycles_per_revolution <= 0.0:
        # 计算并保存到 remaining。
        remaining = abs(rotation)
        # 计算并保存到 chunks（分段集合）。
        chunks: list[float] = []
        # 只要条件成立就重复执行：remaining > chunk_limit + 0.005。
        while remaining > chunk_limit + 0.005:
            # 调用函数：chunks.append。
            chunks.append(direction * chunk_limit)
            # 更新变量：remaining。
            remaining -= chunk_limit
        # 判断条件；满足时执行下面代码：remaining >= 0.005。
        if remaining >= 0.005:
            # 调用函数：chunks.append。
            chunks.append(direction * remaining)
        # 返回结果：chunks（分段集合）。
        return chunks

    # STM32旋转机构是离散周期：先把总角度量化一次，再均分周期数。
    # 各段实际周期之和等于总目标，避免多次四舍五入累积误差。
    # 计算并保存到 total_cycles（总量·周期数）。
    total_cycles = int(
        math.floor(abs(rotation) * cycles_per_revolution / 360.0 + 0.5)
    )
    # 判断条件；满足时执行下面代码：total_cycles <= 0。
    if total_cycles <= 0:
        # 返回结果：创建数据容器。
        return [0.0]
    # 计算并保存到 max_chunk_cycles（最大·分段·周期数）。
    max_chunk_cycles = max(
        1, int(math.floor(chunk_limit * cycles_per_revolution / 360.0))
    )
    # 计算并保存到 chunk_count（分段·数量）。
    chunk_count = int(math.ceil(total_cycles / max_chunk_cycles))
    # 计算并保存到 base_cycles（base·周期数）、extra。
    base_cycles, extra = divmod(total_cycles, chunk_count)
    # 计算并保存到 cycle_chunks（cycle·分段集合）。
    cycle_chunks = [
        base_cycles + (1 if index < extra else 0)
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for index in range(chunk_count)
    ]
    # 计算并保存到 degrees_per_cycle（degrees·每·cycle）。
    degrees_per_cycle = 360.0 / cycles_per_revolution
    # 返回结果：按条件生成列表。
    return [direction * cycles * degrees_per_cycle for cycles in cycle_chunks]


# 【函数：_append_rotation_commands】追加单次或分段旋转命令，分段时在原地重新吸取。
# 分段之间放下碎片、释放磁铁并回正空载旋转轴，然后重新吸住。被放下的碎片保留已经完成的旋转角。
# 参数 commands（依次执行的串口命令列表）：字符串列表/序列。
# 参数 rotation_chunks（旋转·分段集合）：浮点数列表/序列。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def _append_rotation_commands(commands: list[str], rotation_chunks: list[float]) -> None:
    """追加单次或分段旋转命令，分段时在原地重新吸取。"""
    # 遍历数据，逐项处理：enumerate(rotation_chunks)。
    for chunk_index, chunk in enumerate(rotation_chunks):
        # 调用函数：commands.append。
        commands.append(f"ROTATE,{_format_number(chunk)}")
        # 判断条件；满足时执行下面代码：chunk_index < len(rotation_chunks) - 1。
        if chunk_index < len(rotation_chunks) - 1:
            # 调用函数：commands.extend。
            commands.extend([
                "Z,DOWN",
                "MAGNET,0",
                "Z,UP",
                f"ROTATE,{_format_number(-chunk)}",
                "Z,DOWN",
                "MAGNET,1",
                "Z,UP",
            ])


# =============================================================================
# 【分区】读取并校验 plan.json
# 功能：检查坐标系、工作区 210×297、路径正交且在纸内、move_order 连续、ready_for_motion。
# 可修改：无现场阈值；几何限制与视觉规划必须一致。
# 看情况改：旧计划缺补偿字段时当 0。不要为了“能发”而跳过 ready_for_motion。
# 不要改：坐标原点 A4 左上、X 右 Y 下；斜线路径必须拒绝。
# =============================================================================
# 【函数：load_motion_plan】读取由视觉程序生成的plan.json。
# 参数 path（当前文件路径或几何路径）：文件系统路径对象或字符串。
# 返回类型：字典（键为字符串，值为任意类型）；箭头->是类型提示，不会替你转换实际返回值。
def load_motion_plan(path: Path | str) -> dict[str, Any]:
    """读取由视觉程序生成的plan.json。"""
    # 计算并保存到 plan_path（运动计划JSON文件路径）。
    plan_path = Path(path)
    # 判断条件；满足时执行下面代码：not plan_path.is_file()。
    if not plan_path.is_file():
        # 抛出异常，通知上层处理：SerialPlanError(f'运动计划不存在：{plan_path}')。
        raise SerialPlanError(f"运动计划不存在：{plan_path}")
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 计算并保存到 payload（准备写入文件或发送的数据）。
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    # 捕获(OSError, json.JSONDecodeError)异常，转入下面的处理代码；异常对象保存在exc。
    except (OSError, json.JSONDecodeError) as exc:
        # 抛出异常，通知上层处理：SerialPlanError(f'运动计划读取失败：{exc}')。
        raise SerialPlanError(f"运动计划读取失败：{exc}") from exc
    # 判断条件；满足时执行下面代码：not isinstance(payload, dict)。
    if not isinstance(payload, dict):
        # 抛出异常，通知上层处理：SerialPlanError('运动计划顶层必须是JSON对象')。
        raise SerialPlanError("运动计划顶层必须是JSON对象")
    # 返回结果：payload（准备写入文件或发送的数据）。
    return payload


# 【函数：validate_motion_plan】执行发送前安全联锁，并按move_order返回碎片。
# 参数 plan（从JSON读取的运动计划字典）：字典（键为字符串，值为任意类型）。
# 返回类型：字典（键为字符串，值为任意类型）列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def validate_motion_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """执行发送前安全联锁，并按move_order返回碎片。"""
    # 判断条件；满足时执行下面代码：plan.get('ready_for_motion') is not True。
    if plan.get("ready_for_motion") is not True:
        # 计算并保存到 reasons（检查未通过的原因列表）。
        reasons = plan.get("failure_reasons") or []
        # 计算并保存到 detail。
        detail = "；".join(map(str, reasons)) if reasons else "ready_for_motion不是true"
        # 抛出异常，通知上层处理：SerialPlanError(f'安全联锁拒绝发送：{detail}')。
        raise SerialPlanError(f"安全联锁拒绝发送：{detail}")
    # 判断条件；满足时执行下面代码：plan.get('all_matches_ok') is not True or plan.get('all_paths_ok') is not True。
    if plan.get("all_matches_ok") is not True or plan.get("all_paths_ok") is not True:
        # 抛出异常，通知上层处理：SerialPlanError('匹配或路径状态不完整，禁止发送')。
        raise SerialPlanError("匹配或路径状态不完整，禁止发送")

    # 计算并保存到 pieces（碎片列表）。
    pieces = plan.get("pieces")
    # 判断条件；满足时执行下面代码：not isinstance(pieces, list) or not 1 <= len(pieces) <= 4。
    if not isinstance(pieces, list) or not 1 <= len(pieces) <= 4:
        # 抛出异常，通知上层处理：SerialPlanError('运动计划必须包含1～4块碎片')。
        raise SerialPlanError("运动计划必须包含1～4块碎片")
    # 判断条件；满足时执行下面代码：int(plan.get('piece_count', len(pieces))) != len(pieces)。
    if int(plan.get("piece_count", len(pieces))) != len(pieces):
        # 抛出异常，通知上层处理：SerialPlanError('piece_count与pieces数量不一致')。
        raise SerialPlanError("piece_count与pieces数量不一致")

    # 计算并保存到 workspace（工作区）。
    workspace = plan.get("mechanical_workspace_mm") or {}
    # 计算并保存到 width（当前区域宽度）、height（当前区域高度）。
    width, height = _point(workspace.get("size", [210.0, 297.0]), "机械工作区尺寸")
    # 判断条件；满足时执行下面代码：width <= 0 or height <= 0。
    if width <= 0 or height <= 0:
        # 抛出异常，通知上层处理：SerialPlanError('机械工作区尺寸无效')。
        raise SerialPlanError("机械工作区尺寸无效")

    # 计算并保存到 seen_ids（已记录·编号集合）。
    seen_ids: set[int] = set()
    # 计算并保存到 seen_orders（已记录·orders）。
    seen_orders: set[int] = set()
    # 计算并保存到 normalized（经过本函数规范化处理的数据）。
    normalized: list[dict[str, Any]] = []
    # 遍历数据，逐项处理：enumerate(pieces, 1)。
    for index, piece in enumerate(pieces, 1):
        # 判断条件；满足时执行下面代码：not isinstance(piece, dict)。
        if not isinstance(piece, dict):
            # 抛出异常，通知上层处理：SerialPlanError(f'第{index}块数据不是JSON对象')。
            raise SerialPlanError(f"第{index}块数据不是JSON对象")
        # 计算并保存到 piece_id（面向显示和计划的碎片编号）。
        piece_id = int(piece.get("id", 0))
        # 计算并保存到 move_order（实际搬运顺序（从1开始））。
        move_order = int(piece.get("move_order", 0))
        # 判断条件；满足时执行下面代码：piece_id <= 0 or piece_id in seen_ids。
        if piece_id <= 0 or piece_id in seen_ids:
            # 抛出异常，通知上层处理：SerialPlanError(f'第{index}块id无效或重复')。
            raise SerialPlanError(f"第{index}块id无效或重复")
        # 判断条件；满足时执行下面代码：move_order <= 0 or move_order in seen_orders。
        if move_order <= 0 or move_order in seen_orders:
            # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}的move_order无效或重复')。
            raise SerialPlanError(f"P{piece_id}的move_order无效或重复")
        # 调用函数：seen_ids.add。
        seen_ids.add(piece_id)
        # 调用函数：seen_orders.add。
        seen_orders.add(move_order)
        # 判断条件；满足时执行下面代码：piece.get('status') != 'ok' or piece.get('match_status') != 'ok'。
        if piece.get("status") != "ok" or piece.get("match_status") != "ok":
            # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}状态不是ok')。
            raise SerialPlanError(f"P{piece_id}状态不是ok")

        # 计算并保存到 source（源数据或源位置）。
        source = _point(piece.get("source_pick_mm"), f"P{piece_id}.source_pick_mm")
        # 计算并保存到 target（目标数据或目标位置）。
        target = _point(piece.get("target_pick_mm"), f"P{piece_id}.target_pick_mm")
        # 遍历数据，逐项处理：(('源吸取点', source), ('目标吸取点', target))。
        for point_name, (x, y) in (("源吸取点", source), ("目标吸取点", target)):
            # 判断条件；满足时执行下面代码：not (0.0 <= x <= width and 0.0 <= y <= height)。
            if not (0.0 <= x <= width and 0.0 <= y <= height):
                # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}{point_name}超出机械工作区')。
                raise SerialPlanError(f"P{piece_id}{point_name}超出机械工作区")
        # 调用函数：_finite_number。
        _finite_number(piece.get("rotation_deg_signed"), f"P{piece_id}.rotation_deg_signed")

        # 计算并保存到 moves。
        moves = piece.get("axis_moves_mm")
        # 判断条件；满足时执行下面代码：not isinstance(moves, list)。
        if not isinstance(moves, list):
            # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}.axis_moves_mm必须是数组')。
            raise SerialPlanError(f"P{piece_id}.axis_moves_mm必须是数组")
        # 计算并保存到 previous（前一项）。
        previous = source
        # 遍历数据，逐项处理：enumerate(moves, 1)。
        for move_index, move in enumerate(moves, 1):
            # 判断条件；满足时执行下面代码：not isinstance(move, dict)。
            if not isinstance(move, dict):
                # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}第{move_index}段路径格式错误')。
                raise SerialPlanError(f"P{piece_id}第{move_index}段路径格式错误")
            # 计算并保存到 axis（选中的轴或数组维度）。
            axis = str(move.get("axis", "")).upper()
            # 判断条件；满足时执行下面代码：axis not in {'X', 'Y'}。
            if axis not in {"X", "Y"}:
                # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}第{move_index}段轴只能是X或Y')。
                raise SerialPlanError(f"P{piece_id}第{move_index}段轴只能是X或Y")
            # 计算并保存到 delta（当前增量或修正量）。
            delta = _finite_number(move.get("delta_mm"),
                                   f"P{piece_id}第{move_index}段delta_mm")
            # 判断条件；满足时执行下面代码：abs(delta) < 0.005。
            if abs(delta) < 0.005:
                # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}第{move_index}段距离为0')。
                raise SerialPlanError(f"P{piece_id}第{move_index}段距离为0")
            # 计算并保存到 start（起始值或起点）。
            start = _point(move.get("from_mm"), f"P{piece_id}第{move_index}段from_mm")
            # 计算并保存到 end（终止值或终点）。
            end = _point(move.get("to_mm"), f"P{piece_id}第{move_index}段to_mm")
            # 判断条件；满足时执行下面代码：max(abs(start[0] - previous[0]), abs(start[1] - previous[1])) > 0.02。
            if max(abs(start[0] - previous[0]), abs(start[1] - previous[1])) > 0.02:
                # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}第{move_index}段与前一段不连续')。
                raise SerialPlanError(f"P{piece_id}第{move_index}段与前一段不连续")
            # 计算并保存到 dx、dy。
            dx, dy = end[0] - start[0], end[1] - start[1]
            # 判断条件；满足时执行下面代码：axis == 'X'。
            if axis == "X":
                # 判断条件；满足时执行下面代码：abs(dy) > 0.02 or abs(dx - delta) > 0.02。
                if abs(dy) > 0.02 or abs(dx - delta) > 0.02:
                    # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}第{move_index}段不是合法X轴运动')。
                    raise SerialPlanError(f"P{piece_id}第{move_index}段不是合法X轴运动")
            # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
            else:
                # 判断条件；满足时执行下面代码：abs(dx) > 0.02 or abs(dy - delta) > 0.02。
                if abs(dx) > 0.02 or abs(dy - delta) > 0.02:
                    # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}第{move_index}段不是合法Y轴运动')。
                    raise SerialPlanError(f"P{piece_id}第{move_index}段不是合法Y轴运动")
            # 判断条件；满足时执行下面代码：not (0.0 <= end[0] <= width and 0.0 <= end[1] <= height)。
            if not (0.0 <= end[0] <= width and 0.0 <= end[1] <= height):
                # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}第{move_index}段终点超出机械工作区')。
                raise SerialPlanError(f"P{piece_id}第{move_index}段终点超出机械工作区")
            # 计算并保存到 previous（前一项）。
            previous = end
        # 判断条件；满足时执行下面代码：max(abs(previous[0] - target[0]), abs(previous[1] - target[1])) > 0.02。
        if max(abs(previous[0] - target[0]), abs(previous[1] - target[1])) > 0.02:
            # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}路径终点不等于目标吸取点')。
            raise SerialPlanError(f"P{piece_id}路径终点不等于目标吸取点")
        # 调用函数：normalized.append。
        normalized.append(piece)

    # 调用函数：normalized.sort。
    normalized.sort(key=lambda item: int(item["move_order"]))
    # 计算并保存到 expected_orders（期望·orders）。
    expected_orders = list(range(1, len(normalized) + 1))
    # 计算并保存到 actual_orders（实测·orders）。
    actual_orders = [int(item["move_order"]) for item in normalized]
    # 判断条件；满足时执行下面代码：actual_orders != expected_orders。
    if actual_orders != expected_orders:
        # 抛出异常，通知上层处理：SerialPlanError('move_order必须从1连续编号')。
        raise SerialPlanError("move_order必须从1连续编号")
    # 返回结果：normalized（经过本函数规范化处理的数据）。
    return normalized


# =============================================================================
# 【分区】生成 ASCII 命令列表（不打开串口）
# 功能：MOTION,1 → 逐片 GOTO/Z/MAGNET/ROTATE/MOVE → 回原点 → MOTION,0。
# 可修改：命令字符串必须与主控固件一致。改协议要同时改下位机。
# 看情况改：是否回原点、是否在目标点旋转，由 plan 里的开关控制。
# 不要改：横移前必须 Z,UP；吸取前必须 Z,DOWN；释放后抬起再回正。PING 不在本列表里。
# =============================================================================
# 【函数：build_serial_commands】把安全计划转换为逐行ASCII命令，不包含传输层PING。
# 命令顺序体现抓放过程：到源点→下降→吸住→抬起→旋转/搬运→下降→释放→抬起→旋转轴回正。函数只是生成列表，不打开串口。
# 参数 plan（从JSON读取的运动计划字典）：字典（键为字符串，值为任意类型）。
# 返回类型：字符串列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def build_serial_commands(plan: dict[str, Any]) -> list[str]:
    """把安全计划转换为逐行ASCII命令，不包含传输层PING。

    超过单次旋转上限时，在原吸取点放下碎片、旋转轴回正并重新吸取，
    使碎片角度逐段累计，而线束偏角始终不超过配置上限。
    横移前必须抬起；每次吸取前必须下降；释放后必须抬高再回正。
    """
    # 计算并保存到 pieces（碎片列表）。
    pieces = validate_motion_plan(plan)
    # 计算并保存到 offset_x（偏移·X轴）、offset_y（偏移·Y轴）、rotation_reduction（旋转·减量）、rotation_reduction_min_angle（旋转·减量·最小·角度）、rotation_chunk（旋转·分段）、rotation_cycles（旋转·周期数）。
    (
        offset_x, offset_y, rotation_reduction, rotation_reduction_min_angle,
        rotation_chunk, rotation_cycles,
    ) = _execution_compensation(plan)
    # 计算并保存到 xy_row0（XY平面·row0）、xy_row1（XY平面·row1）、xy_bias（XY平面·偏置）。
    xy_row0, xy_row1, xy_bias = _xy_delta_compensation(plan)
    # 计算并保存到 workspace（工作区）。
    workspace = plan.get("mechanical_workspace_mm") or {}
    # 计算并保存到 width（当前区域宽度）、height（当前区域高度）。
    width, height = _point(workspace.get("size", [210.0, 297.0]), "机械工作区尺寸")
    # 计算并保存到 xy_linear_compensated（XY平面·linear·compensated）。
    xy_linear_compensated = (
        abs(xy_row0[0] - 1.0) >= 1e-9 or abs(xy_row0[1]) >= 1e-9
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        or abs(xy_row1[0]) >= 1e-9 or abs(xy_row1[1] - 1.0) >= 1e-9
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        or abs(xy_bias[0]) >= 0.005 or abs(xy_bias[1]) >= 0.005
    )
    # 计算并保存到 position_compensated（位置·compensated）。
    position_compensated = (
        abs(offset_x) >= 0.005 or abs(offset_y) >= 0.005 or xy_linear_compensated
    )

    # 计算并保存到 return_origin（返回或回位·原点）。
    return_origin = _return_origin_after_plan(plan)
    # 计算并保存到 rotate_at_target（rotate·at·目标）。
    rotate_at_target = _rotate_at_target(plan)
    # 计算并保存到 commands（依次执行的串口命令列表）。
    commands = ["MOTION,1", f"PLAN,{len(pieces)}"]
    # 遍历数据，逐项处理：pieces。
    for piece in pieces:
        # 计算并保存到 piece_id（面向显示和计划的碎片编号）。
        piece_id = int(piece["id"])
        # 计算并保存到 source_x（源·X轴）、source_y（源·Y轴）。
        source_x, source_y = _point(piece["source_pick_mm"], "source_pick_mm")
        # 计算并保存到 target_x（目标·X轴）、target_y（目标·Y轴）。
        target_x, target_y = _point(piece["target_pick_mm"], "target_pick_mm")
        # 历史字段名是target_offset_mm，实际表示视觉坐标到磁铁中心的固定工具偏移。
        # 同一个工具偏移必须同时作用于源吸取点和目标放置点。
        # 计算并保存到 command_source_x（指令·源·X轴）。
        command_source_x = source_x + offset_x
        # 计算并保存到 command_source_y（指令·源·Y轴）。
        command_source_y = source_y + offset_y
        # 计算并保存到 command_target_x（指令·目标·X轴）。
        command_target_x = target_x + offset_x
        # 计算并保存到 command_target_y（指令·目标·Y轴）。
        command_target_y = target_y + offset_y
        # 判断条件；满足时执行下面代码：not (0.0 <= command_source_x <= width and 0.0 <= command_source_y <= height)。
        if not (0.0 <= command_source_x <= width and 0.0 <= command_source_y <= height):
            # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}补偿后源吸取点超出机械工作区')。
            raise SerialPlanError(f"P{piece_id}补偿后源吸取点超出机械工作区")
        # 判断条件；满足时执行下面代码：not (0.0 <= command_target_x <= width and 0.0 <= command_target_y <= height)。
        if not (0.0 <= command_target_x <= width and 0.0 <= command_target_y <= height):
            # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}补偿后目标放置点超出机械工作区')。
            raise SerialPlanError(f"P{piece_id}补偿后目标放置点超出机械工作区")

        # 计算并保存到 rotation（旋转数据（角度或旋转矩阵，取决于本函数））。
        rotation = _finite_number(piece["rotation_deg_signed"], "rotation_deg_signed")
        # 计算并保存到 command_rotation（指令·旋转）。
        command_rotation = _reduce_rotation_magnitude(
            rotation, rotation_reduction, rotation_reduction_min_angle
        )
        # 计算并保存到 rotation_chunks（旋转·分段集合）。
        rotation_chunks = _split_rotation(
            command_rotation, rotation_chunk, rotation_cycles
        )
        # 调用函数：commands.extend。
        commands.extend([
            f"PIECE,{piece_id}",
            f"GOTO,{_format_number(command_source_x)},{_format_number(command_source_y)}",
            "Z,DOWN",
            "MAGNET,1",
            "Z,UP",
        ])
        # 判断条件；满足时执行下面代码：not rotate_at_target。
        if not rotate_at_target:
            # 调用函数：_append_rotation_commands。
            _append_rotation_commands(commands, rotation_chunks)

        # 判断条件；满足时执行下面代码：position_compensated。
        if position_compensated:
            # 计算并保存到 desired_delta_x（期望·delta·X轴）。
            desired_delta_x = command_target_x - command_source_x
            # 计算并保存到 desired_delta_y（期望·delta·Y轴）。
            desired_delta_y = command_target_y - command_source_y
            # 判断条件；满足时执行下面代码：abs(desired_delta_x) < 0.005 and abs(desired_delta_y) < 0.005。
            if abs(desired_delta_x) < 0.005 and abs(desired_delta_y) < 0.005:
                # 计算并保存到 delta_x（delta·X轴）、delta_y（delta·Y轴）。
                delta_x = delta_y = 0.0
            # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
            else:
                # 计算并保存到 delta_x（delta·X轴）。
                delta_x = (
                    xy_row0[0] * desired_delta_x
                    + xy_row0[1] * desired_delta_y
                    + xy_bias[0]
                )
                # 计算并保存到 delta_y（delta·Y轴）。
                delta_y = (
                    xy_row1[0] * desired_delta_x
                    + xy_row1[1] * desired_delta_y
                    + xy_bias[1]
                )
            # 计算并保存到 corrected_target_x（校正后·目标·X轴）。
            corrected_target_x = command_source_x + delta_x
            # 计算并保存到 corrected_target_y（校正后·目标·Y轴）。
            corrected_target_y = command_source_y + delta_y
            # 判断条件；满足时执行下面代码：not (0.0 <= corrected_target_x <= width and 0.0 <= corrected_target_y <= height)。
            if not (0.0 <= corrected_target_x <= width and 0.0 <= corrected_target_y <= height):
                # 抛出异常，通知上层处理：SerialPlanError(f'P{piece_id}XY线性补偿后目标超出机械工作区')。
                raise SerialPlanError(f"P{piece_id}XY线性补偿后目标超出机械工作区")
            # 判断条件；满足时执行下面代码：abs(delta_x) >= 0.005。
            if abs(delta_x) >= 0.005:
                # 调用函数：commands.append。
                commands.append(f"MOVE,X,{_format_number(delta_x)}")
            # 判断条件；满足时执行下面代码：abs(delta_y) >= 0.005。
            if abs(delta_y) >= 0.005:
                # 调用函数：commands.append。
                commands.append(f"MOVE,Y,{_format_number(delta_y)}")
        # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
        else:
            # 遍历数据，逐项处理：piece['axis_moves_mm']。
            for move in piece["axis_moves_mm"]:
                # 计算并保存到 axis（选中的轴或数组维度）。
                axis = str(move["axis"]).upper()
                # 计算并保存到 delta（当前增量或修正量）。
                delta = _finite_number(move["delta_mm"], "delta_mm")
                # 调用函数：commands.append。
                commands.append(f"MOVE,{axis},{_format_number(delta)}")
        # 判断条件；满足时执行下面代码：rotate_at_target。
        if rotate_at_target:
            # 调用函数：_append_rotation_commands。
            _append_rotation_commands(commands, rotation_chunks)
        # 调用函数：commands.extend。
        commands.extend(["Z,DOWN", "MAGNET,0", "Z,UP"])
        # 前面的完整分段均已各自回正，结束时只需回正最后一段。
        # 计算并保存到 last_chunk（最后·分段）。
        last_chunk = rotation_chunks[-1]
        # 判断条件；满足时执行下面代码：abs(last_chunk) >= 0.005。
        if abs(last_chunk) >= 0.005:
            # 调用函数：commands.append。
            commands.append(f"ROTATE,{_format_number(-last_chunk)}")
        # 调用函数：commands.append。
        commands.append(f"ENDPIECE,{piece_id}")
    # 判断条件；满足时执行下面代码：return_origin。
    if return_origin:
        # 调用函数：commands.append。
        commands.append("GOTO,0.00,0.00")
    # 调用函数：commands.extend。
    commands.extend(["ENDPLAN", "MOTION,0"])
    # 返回结果：commands（依次执行的串口命令列表）。
    return commands


# 【函数：print_serial_commands】打印dry-run指令并返回指令数量。
# 参数 commands（依次执行的串口命令列表）：可逐项遍历的字符串列表/序列。
# 参数 output（输出消息的回调，默认使用print）：Callable[[str], None]；省略时使用print。
# 返回类型：整数；箭头->是类型提示，不会替你转换实际返回值。
def print_serial_commands(commands: Iterable[str], output: Callable[[str], None] = print) -> int:
    """打印dry-run指令并返回指令数量。"""
    # 计算并保存到 count（计数值）。
    count = 0
    # 调用函数：output。
    output("[DRY-RUN] 未打开串口，以下为计划发送的指令：")
    # 遍历数据，逐项处理：commands。
    for command in commands:
        # 调用函数：output。
        output(f"[TX] {command}")
        # 更新变量：count（计数值）。
        count += 1
    # 调用函数：output。
    output(f"[DRY-RUN] 共{count}条指令，未驱动任何电机")
    # 返回结果：count（计数值）。
    return count


# =============================================================================
# 【分区】串口应答、安全停机
# 功能：readline 等到 ACK；超时/ERR/乱码抛错；异常时尽力发 STOP、MOTION,0。
# 可修改：ACK 文本必须与固件一致（默认 ACK）。
# 看情况改：部分板子回 OK 而不是 ACK，只能改固件或这里的接受列表，不要把任意文本当成功。
# 不要改：超时必须停发；安全停机不等 ACK，避免死锁。
# =============================================================================
# 【函数：_read_response】读取一行串口应答，接受ACK，超时、ERR或无法识别时抛出异常。
# 参数 connection（串口连接对象）：任意类型。
# 返回类型：字符串；箭头->是类型提示，不会替你转换实际返回值。
def _read_response(connection: Any) -> str:
    # 计算并保存到 raw（刚读到的原始数据）。
    raw = connection.readline()
    # 判断条件；满足时执行下面代码：not raw。
    if not raw:
        # 抛出异常，通知上层处理：SerialPlanError('等待ACK超时，已停止发送')。
        raise SerialPlanError("等待ACK超时，已停止发送")
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 计算并保存到 response（主控返回的应答文本）。
        response = raw.decode("ascii", errors="strict").strip()
    # 捕获UnicodeDecodeError异常，转入下面的处理代码；异常对象保存在exc。
    except UnicodeDecodeError as exc:
        # 抛出异常，通知上层处理：SerialPlanError('下位机返回了非ASCII数据')。
        raise SerialPlanError("下位机返回了非ASCII数据") from exc
    # 判断条件；满足时执行下面代码：response == 'ACK' or response.startswith('ACK,')。
    if response == "ACK" or response.startswith("ACK,"):
        # 返回结果：response（主控返回的应答文本）。
        return response
    # 判断条件；满足时执行下面代码：response.startswith('ERR')。
    if response.startswith("ERR"):
        # 抛出异常，通知上层处理：SerialPlanError(f'下位机拒绝命令：{response}')。
        raise SerialPlanError(f"下位机拒绝命令：{response}")
    # 抛出异常，通知上层处理：SerialPlanError(f'无法识别的下位机响应：{response!r}')。
    raise SerialPlanError(f"无法识别的下位机响应：{response!r}")


# 【函数：_validate_origin_response】验证POS响应中的XY逻辑坐标位于原点。
# 参数 response（主控返回的应答文本）：字符串。
# 参数 tolerance（用于浮点比较或几何判断的容差）：浮点数；省略时使用0.01。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def _validate_origin_response(response: str, tolerance: float = 0.01) -> None:
    """验证POS响应中的XY逻辑坐标位于原点。"""
    # 计算并保存到 fields。
    fields = response.split(",")
    # 判断条件；满足时执行下面代码：len(fields) < 4 or fields[0] != 'ACK' or fields[1] != 'POS'。
    if len(fields) < 4 or fields[0] != "ACK" or fields[1] != "POS":
        # 抛出异常，通知上层处理：SerialPlanError(f'POS响应格式错误：{response!r}')。
        raise SerialPlanError(f"POS响应格式错误：{response!r}")
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 计算并保存到 x（当前横坐标或横向数据）。
        x = float(fields[2])
        # 计算并保存到 y（当前纵坐标或纵向数据）。
        y = float(fields[3])
    # 捕获ValueError异常，转入下面的处理代码；异常对象保存在exc。
    except ValueError as exc:
        # 抛出异常，通知上层处理：SerialPlanError(f'POS坐标无法解析：{response!r}')。
        raise SerialPlanError(f"POS坐标无法解析：{response!r}") from exc
    # 判断条件；满足时执行下面代码：abs(x) > tolerance or abs(y) > tolerance。
    if abs(x) > tolerance or abs(y) > tolerance:
        # 抛出异常，通知上层处理：SerialPlanError(f'执行前XY未在逻辑原点：X={x:g}, Y={y:g}；请先回零')。
        raise SerialPlanError(
            f"执行前XY未在逻辑原点：X={x:g}, Y={y:g}；请先回零"
        )


# 【函数：_best_effort_safety_shutdown】异常收尾：不等待ACK，尽力发送STOP和MOTION,0后再关闭串口。
# 参数 connection（串口连接对象）：任意类型。
# 参数 output（输出消息的回调，默认使用print）：Callable[[str], None]。
# 返回类型：None（不返回业务结果）；箭头->是类型提示，不会替你转换实际返回值。
def _best_effort_safety_shutdown(
    connection: Any,
    output: Callable[[str], None],
) -> None:
    """异常收尾：不等待ACK，尽力发送STOP和MOTION,0后再关闭串口。"""
    # 遍历数据，逐项处理：('STOP', 'MOTION,0')。
    for command in ("STOP", "MOTION,0"):
        # 执行可能出错的代码，并由后面的异常分支处理错误。
        try:
            # 调用函数：connection.write。
            connection.write((command + "\n").encode("ascii"))
            # 判断条件；满足时执行下面代码：hasattr(connection, 'flush')。
            if hasattr(connection, "flush"):
                # 调用函数：connection.flush。
                connection.flush()
        # 捕获Exception异常，转入下面的处理代码；异常对象保存在exc。
        except Exception as exc:
            # 执行可能出错的代码，并由后面的异常分支处理错误。
            try:
                # 调用函数：output。
                output(f"[WARN] 安全收尾命令{command}发送失败：{exc}")
            # 捕获Exception异常，转入下面的处理代码。
            except Exception:
                pass
        # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
        else:
            # 执行可能出错的代码，并由后面的异常分支处理错误。
            try:
                # 调用函数：output。
                output(f"[SAFE] 已尽力发送{command}（不等待ACK）")
            # 捕获Exception异常，转入下面的处理代码。
            except Exception:
                pass



# 【函数：send_serial_commands】逐条发送命令；每条都等待ACK，任一异常立即停止。
# 每条发送后立即_read_response；根据当前命令和下一命令决定等待时间。sent包含预检命令，不只是碎片运动命令数。
# 参数 commands（依次执行的串口命令列表）：可逐项遍历的字符串列表/序列。
# 参数 port（串口设备名称）：字符串。
# 参数 baudrate（串口波特率）：整数；省略时使用115200。
# 参数 ack_timeout_seconds（应答·超时·秒）：浮点数；省略时使用2.0。
# 参数 magnet_pickup_settle_seconds（磁铁·吸取·稳定等待·秒）：浮点数；省略时使用0.0。
# 参数 magnet_release_settle_seconds（磁铁·释放·稳定等待·秒）：浮点数；省略时使用0.0。
# 参数 lift_settle_seconds（抬升·稳定等待·秒）：浮点数；省略时使用0.0。
# 参数 xy_settle_seconds（XY平面·稳定等待·秒）：浮点数；省略时使用0.0。
# 参数 rotation_settle_seconds（旋转·稳定等待·秒）：浮点数；省略时使用0.0。
# 参数 command_delay_seconds（指令·间隔·秒）：浮点数；省略时使用0.0。
# 参数 require_origin_before_plan（要求·原点·操作前·运动计划）：布尔值True/False；省略时使用False。
# 参数 serial_factory（可注入的串口创建函数，None表示使用真实pyserial）：Callable[..., Any]或None（不返回业务结果）；省略时使用None。
# 参数 output（输出消息的回调，默认使用print）：Callable[[str], None]；省略时使用print。
# 返回类型：整数；箭头->是类型提示，不会替你转换实际返回值。
# =============================================================================
# 【分区】逐条发送并等待 ACK
# 功能：打开串口，按命令类型插入吸取/释放/抬升/XY/旋转稳定延时，失败则安全停机并关口。
# 可修改：各 settle 秒数来自 config（magnet_pickup_settle_seconds 等）；波特率 115200。
# 看情况改：DTR/RTS 默认 False 防复位。现场若必须拉 DTR 才能通，再改，并确认不会重启主控。
# 不要改：每条命令后立刻等 ACK；finally 关闭连接；dry_run 不走本函数。
# =============================================================================
def send_serial_commands(
    commands: Iterable[str],
    port: str,
    baudrate: int = 115200,
    ack_timeout_seconds: float = 2.0,
    magnet_pickup_settle_seconds: float = 0.0,
    magnet_release_settle_seconds: float = 0.0,
    lift_settle_seconds: float = 0.0,
    xy_settle_seconds: float = 0.0,
    rotation_settle_seconds: float = 0.0,
    command_delay_seconds: float = 0.0,
    require_origin_before_plan: bool = False,
    serial_factory: Callable[..., Any] | None = None,
    output: Callable[[str], None] = print,
) -> int:
    """逐条发送命令；每条都等待ACK，任一异常立即停止。"""
    # 判断条件；满足时执行下面代码：not str(port).strip()。
    if not str(port).strip():
        # 抛出异常，通知上层处理：SerialPlanError('真实发送必须指定串口，例如--port COM5')。
        raise SerialPlanError("真实发送必须指定串口，例如--port COM5")
    # 判断条件；满足时执行下面代码：int(baudrate) <= 0。
    if int(baudrate) <= 0:
        # 抛出异常，通知上层处理：SerialPlanError('波特率必须大于0')。
        raise SerialPlanError("波特率必须大于0")
    # 计算并保存到 timeout（等待超时时间（秒））。
    timeout = _finite_number(ack_timeout_seconds, "ACK超时时间")
    # 判断条件；满足时执行下面代码：timeout <= 0。
    if timeout <= 0:
        # 抛出异常，通知上层处理：SerialPlanError('ACK超时时间必须大于0')。
        raise SerialPlanError("ACK超时时间必须大于0")
    # 计算并保存到 pickup_settle（吸取·稳定等待）。
    pickup_settle = _finite_number(
        magnet_pickup_settle_seconds, "磁铁吸取稳定等待时间"
    )
    # 计算并保存到 release_settle（释放·稳定等待）。
    release_settle = _finite_number(
        magnet_release_settle_seconds, "磁铁释放稳定等待时间"
    )
    # 计算并保存到 lift_settle（抬升·稳定等待）。
    lift_settle = _finite_number(lift_settle_seconds, "Z轴抬升稳定等待时间")
    # 计算并保存到 xy_settle（XY平面·稳定等待）。
    xy_settle = _finite_number(xy_settle_seconds, "XY停止稳定等待时间")
    # 计算并保存到 rotation_settle（旋转·稳定等待）。
    rotation_settle = _finite_number(rotation_settle_seconds, "旋转稳定等待时间")
    # 判断条件；满足时执行下面代码：min(pickup_settle, release_settle, lift_settle, xy_settle, rotation_settle) < 0…。
    if min(pickup_settle, release_settle, lift_settle, xy_settle, rotation_settle) < 0.0:
        # 抛出异常，通知上层处理：SerialPlanError('执行稳定等待时间不能为负数')。
        raise SerialPlanError("执行稳定等待时间不能为负数")

    # 计算并保存到 is_real_serial（is·real·串口）。
    is_real_serial = serial_factory is None
    # 判断条件；满足时执行下面代码：is_real_serial。
    if is_real_serial:
        # 执行可能出错的代码，并由后面的异常分支处理错误。
        try:
            # 导入：serial。
            import serial
        # 捕获ImportError异常，转入下面的处理代码；异常对象保存在exc。
        except ImportError as exc:
            # 抛出异常，通知上层处理：SerialPlanError('缺少pyserial，请执行python -m pip install -r requirements.…。
            raise SerialPlanError("缺少pyserial，请执行python -m pip install -r requirements.txt") from exc

        # 先配置DTR/RTS再open()，避免Windows打开CH340时产生复位脉冲。
        # 这与直接Serial(port=...)相比能降低STM32被意外复位的概率。
        # 计算并保存到 connection（串口连接对象）。
        connection = serial.Serial()
        # 计算并保存到 connection.port（串口设备名称）。
        connection.port = str(port)
        # 计算并保存到 connection.baudrate（串口波特率）。
        connection.baudrate = int(baudrate)
        # 计算并保存到 connection.timeout（读取超时（秒））。
        connection.timeout = timeout
        # 计算并保存到 connection.write_timeout（写入超时（秒））。
        connection.write_timeout = timeout
        # 计算并保存到 connection.dtr（串口DTR控制线状态）。
        connection.dtr = False
        # 计算并保存到 connection.rts（串口RTS控制线状态）。
        connection.rts = False
        # 调用函数：connection.open。
        connection.open()
    # 进入前面条件未命中的分支；若此else属于try，则表示try正常结束且没有异常。
    else:
        # 计算并保存到 connection（串口连接对象）。
        connection = serial_factory(
            # 传入命名参数：port=str(port)（串口设备名称）；取值过程：调用 str；baudrate=int(baudrate)（串口波特率）；取值过程：调用 int；
            # timeout=timeout（等待超时时间（秒））；取值过程：timeout（等待超时时间（秒））；write_timeout=timeout（write·超时）；
            # 取值过程：timeout（等待超时时间（秒））。
            port=str(port), baudrate=int(baudrate), timeout=timeout, write_timeout=timeout
        )

    # 计算并保存到 sent（已经收到应答的发送条数）。
    sent = 0
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 打开后等待主控和USB串口稳定，再清空历史收发数据。
        # 判断条件；满足时执行下面代码：not is_real_serial。
        if not is_real_serial:
            # 判断条件；满足时执行下面代码：hasattr(connection, 'dtr')。
            if hasattr(connection, "dtr"):
                # 计算并保存到 connection.dtr（串口DTR控制线状态）。
                connection.dtr = False
            # 判断条件；满足时执行下面代码：hasattr(connection, 'rts')。
            if hasattr(connection, "rts"):
                # 计算并保存到 connection.rts（串口RTS控制线状态）。
                connection.rts = False
        # 调用函数：time.sleep。
        time.sleep(1.0)
        # 判断条件；满足时执行下面代码：hasattr(connection, 'reset_input_buffer')。
        if hasattr(connection, "reset_input_buffer"):
            # 调用函数：connection.reset_input_buffer。
            connection.reset_input_buffer()
        # 判断条件；满足时执行下面代码：hasattr(connection, 'reset_output_buffer')。
        if hasattr(connection, "reset_output_buffer"):
            # 调用函数：connection.reset_output_buffer。
            connection.reset_output_buffer()
        # 计算并保存到 preflight_commands。
        preflight_commands = ["PING"]
        # 判断条件；满足时执行下面代码：require_origin_before_plan。
        if require_origin_before_plan:
            # 调用函数：preflight_commands.append。
            preflight_commands.append("POS")
        # 计算并保存到 all_commands。
        all_commands = [*preflight_commands, *list(commands)]
        # 调用函数：output。
        output(f"[SERIAL] 已打开{port}，baud={int(baudrate)}，等待ACK超时={timeout:.2f}s")
        # 遍历数据，逐项处理：enumerate(all_commands)。
        for command_index, command in enumerate(all_commands):
            # 计算并保存到 next_command（下一项·指令）。
            next_command = (
                all_commands[command_index + 1]
                # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
                if command_index + 1 < len(all_commands) else ""
            )
            # 计算并保存到 payload（准备写入文件或发送的数据）。
            payload = (command + "\n").encode("ascii")
            # 调用函数：output。
            output(f"[TX] {command}")
            # 调用函数：connection.write。
            connection.write(payload)
            # 判断条件；满足时执行下面代码：hasattr(connection, 'flush')。
            if hasattr(connection, "flush"):
                # 调用函数：connection.flush。
                connection.flush()
            # 计算并保存到 response（主控返回的应答文本）。
            response = _read_response(connection)
            # 调用函数：output。
            output(f"[RX] {response}")
            # 更新变量：sent（已经收到应答的发送条数）。
            sent += 1
            # 判断条件；满足时执行下面代码：command == 'POS'。
            if command == "POS":
                # 调用函数：_validate_origin_response。
                _validate_origin_response(response)
                # 调用函数：output。
                output("[OK] 执行前XY逻辑原点验证通过")
            # 计算并保存到 settle_seconds（稳定等待·秒）。
            settle_seconds = 0.0
            # 计算并保存到 settle_reason（稳定等待·原因）。
            settle_reason = ""
            # 判断条件；满足时执行下面代码：command == 'MAGNET,1'。
            if command == "MAGNET,1":
                # 计算并保存到 settle_seconds（稳定等待·秒）。
                settle_seconds = pickup_settle
                # 计算并保存到 settle_reason（稳定等待·原因）。
                settle_reason = "磁铁吸取稳定"
            # 判断条件；满足时执行下面代码：command == 'MAGNET,0'。
            elif command == "MAGNET,0":
                # 计算并保存到 settle_seconds（稳定等待·秒）。
                settle_seconds = release_settle
                # 计算并保存到 settle_reason（稳定等待·原因）。
                settle_reason = "磁铁释放稳定"
            # 判断条件；满足时执行下面代码：command == 'Z,UP' and (next_command.startswith('ROTATE,') or next_command.start…。
            elif command == "Z,UP" and (
                next_command.startswith("ROTATE,") or next_command.startswith("MOVE,")
            ):
                # 计算并保存到 settle_seconds（稳定等待·秒）。
                settle_seconds = lift_settle
                # 计算并保存到 settle_reason（稳定等待·原因）。
                settle_reason = "Z轴抬升稳定"
            # 判断条件；满足时执行下面代码：command.startswith('MOVE,') and (not next_command.startswith('MOVE,'))。
            elif command.startswith("MOVE,") and not next_command.startswith("MOVE,"):
                # 计算并保存到 settle_seconds（稳定等待·秒）。
                settle_seconds = xy_settle
                # 计算并保存到 settle_reason（稳定等待·原因）。
                settle_reason = "XY停止稳定"
            # 判断条件；满足时执行下面代码：command.startswith('ROTATE,')。
            elif command.startswith("ROTATE,"):
                # 计算并保存到 settle_seconds（稳定等待·秒）。
                settle_seconds = rotation_settle
                # 计算并保存到 settle_reason（稳定等待·原因）。
                settle_reason = "旋转稳定"
            # 判断条件；满足时执行下面代码：settle_seconds > 0.0。
            if settle_seconds > 0.0:
                # 调用函数：output。
                output(f"[WAIT] {settle_reason}{settle_seconds:.2f}s")
                # 调用函数：time.sleep。
                time.sleep(settle_seconds)
            # 判断条件；满足时执行下面代码：command_delay_seconds > 0.0。
            if command_delay_seconds > 0.0:
                # 调用函数：output。
                output(f"[WAIT] 命令间隔{command_delay_seconds:.2f}s")
                # 调用函数：time.sleep。
                time.sleep(command_delay_seconds)
        # 调用函数：output。
        output(f"[OK] 串口计划发送完成，共{sent}条（含PING）")
        # 返回结果：sent（已经收到应答的发送条数）。
        return sent
    # 捕获BaseException异常，转入下面的处理代码。
    except BaseException:
        # 包括ACK超时、下位机ERR、串口异常以及用户Ctrl+C，均尝试先停机再关串口。
        # 调用函数：_best_effort_safety_shutdown。
        _best_effort_safety_shutdown(connection, output)
        # 抛出异常，通知上层处理：。
        raise
    # 无论正常结束、return还是抛出异常都会进入这里，用于释放相机/串口等资源。
    finally:
        # 调用函数：connection.close。
        connection.close()


# 【函数：send_plan_file】读取plan.json并选择dry-run或真实串口发送。
# 参数 path（当前文件路径或几何路径）：文件系统路径对象或字符串。
# 参数 dry_run（是否只打印命令而不发送）：布尔值True/False；必须按参数名传入。
# 参数 port（串口设备名称）：字符串或None（不返回业务结果）；省略时使用None；必须按参数名传入。
# 参数 baudrate（串口波特率）：整数；省略时使用115200；必须按参数名传入。
# 参数 ack_timeout_seconds（应答·超时·秒）：浮点数；省略时使用2.0；必须按参数名传入。
# 参数 command_delay_seconds（指令·间隔·秒）：浮点数；省略时使用0.0；必须按参数名传入。
# 参数 serial_factory（可注入的串口创建函数，None表示使用真实pyserial）：Callable[..., Any]或None（不返回业务结果）；省略时使用None；必须按参数名传入。
# 参数 output（输出消息的回调，默认使用print）：Callable[[str], None]；省略时使用print；必须按参数名传入。
# 返回类型：整数；箭头->是类型提示，不会替你转换实际返回值。
# =============================================================================
# 【分区】对外入口 send_plan_file
# 功能：读计划 → 校验 → 生成命令；dry_run 只打印，否则 send_serial_commands。
# 可修改：--port、波特率、ack 超时。
# 看情况改：比赛前先 dry_run 看命令，确认坐标再实发。
# 不要改：dry_run=False 且 port 为空要报错；不合格计划在发送前就必须拒绝。
# =============================================================================
def send_plan_file(
    path: Path | str,
    *,
    dry_run: bool,
    port: str | None = None,
    baudrate: int = 115200,
    ack_timeout_seconds: float = 2.0,
    command_delay_seconds: float = 0.0,
    serial_factory: Callable[..., Any] | None = None,
    output: Callable[[str], None] = print,
) -> int:
    """读取plan.json并选择dry-run或真实串口发送。"""
    # 计算并保存到 plan（从JSON读取的运动计划字典）。
    plan = load_motion_plan(path)
    # 计算并保存到 commands（依次执行的串口命令列表）。
    commands = build_serial_commands(plan)
    # 计算并保存到 require_origin（要求·原点）。
    require_origin = _require_origin_before_plan(plan)
    # 计算并保存到 pickup_settle（吸取·稳定等待）、release_settle（释放·稳定等待）、lift_settle（抬升·稳定等待）、xy_settle（XY平面·稳定等待）、rotation_settle（旋转·稳定等待）。
    (
        pickup_settle, release_settle, lift_settle, xy_settle, rotation_settle,
    ) = _execution_timing(plan)
    # 判断条件；满足时执行下面代码：dry_run。
    if dry_run:
        # 判断条件；满足时执行下面代码：max(pickup_settle, release_settle, lift_settle, xy_settle, rotation_settle) > 0…。
        if max(
            pickup_settle, release_settle, lift_settle, xy_settle, rotation_settle
        ) > 0.0:
            # 调用函数：output。
            output(
                f"[DRY-RUN] 稳定等待：吸取后{pickup_settle:.2f}s，"
                f"释放后{release_settle:.2f}s，抬升后{lift_settle:.2f}s，"
                f"XY停止后{xy_settle:.2f}s，旋转后{rotation_settle:.2f}s"
            )
        # 返回结果：调用 print_serial_commands。
        return print_serial_commands(commands, output)
    # 返回结果：调用 send_serial_commands。
    return send_serial_commands(
        commands,
        # 传入命名参数：port=port or ''（串口设备名称）；取值过程：组合多个条件：port or ''。
        port=port or "",
        # 传入命名参数：baudrate=baudrate（串口波特率）；取值过程：baudrate（串口波特率）。
        baudrate=baudrate,
        # 传入命名参数：ack_timeout_seconds=ack_timeout_seconds（应答·超时·秒）；取值过程：ack_timeout_seconds（应答·超时·秒）。
        ack_timeout_seconds=ack_timeout_seconds,
        # 传入命名参数：magnet_pickup_settle_seconds=pickup_settle（磁铁·吸取·稳定等待·秒）；取值过程：pickup_settle（吸取·稳定等待）。
        magnet_pickup_settle_seconds=pickup_settle,
        # 传入命名参数：magnet_release_settle_seconds=release_settle（磁铁·释放·稳定等待·秒）；取值过程：release_settle（释放·稳定等待）。
        magnet_release_settle_seconds=release_settle,
        # 传入命名参数：lift_settle_seconds=lift_settle（抬升·稳定等待·秒）；取值过程：lift_settle（抬升·稳定等待）。
        lift_settle_seconds=lift_settle,
        # 传入命名参数：xy_settle_seconds=xy_settle（XY平面·稳定等待·秒）；取值过程：xy_settle（XY平面·稳定等待）。
        xy_settle_seconds=xy_settle,
        # 传入命名参数：rotation_settle_seconds=rotation_settle（旋转·稳定等待·秒）；取值过程：rotation_settle（旋转·稳定等待）。
        rotation_settle_seconds=rotation_settle,
        # 传入命名参数：command_delay_seconds=command_delay_seconds（指令·间隔·秒）；取值过程：command_delay_seconds（指令·间隔·秒）。
        command_delay_seconds=command_delay_seconds,
        # 传入命名参数：require_origin_before_plan=require_origin（要求·原点·操作前·运动计划）；取值过程：require_origin（要求·原点）。
        require_origin_before_plan=require_origin,
        # 传入命名参数：serial_factory=serial_factory（可注入的串口创建函数，None表示使用真实pyserial）；
        # 取值过程：serial_factory（可注入的串口创建函数，None表示使用真实pyserial）。
        serial_factory=serial_factory,
        # 传入命名参数：output=output（输出消息的回调，默认使用print）；取值过程：output（输出消息的回调，默认使用print）。
        output=output,
    )
