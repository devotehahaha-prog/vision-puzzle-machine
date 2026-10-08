"""将视觉运动计划转换为安全、可确认的串口ASCII指令。"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：与普通版相同的串口协议层：校验 plan → ASCII 命令 → ACK。扑克仍是 3 个任务。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约35行  SerialPlanError：计划或协议错误
#   - 约297行  validate_motion_plan：校验路径与坐标系
#   - 约389行  build_serial_commands：生成命令列表
#   - 约570行  send_serial_commands：实发并等 ACK
#   - 约707行  send_plan_file：对外入口
# =============================================================================
# 输入主程序生成的plan.json；先验证字段和路径，再生成按顺序执行的ASCII命令，最后逐条发送并等待ACK。
# 阅读顺序：send_plan_file → load_motion_plan → validate_motion_plan → build_serial_commands → send_serial_commands。
# 协议中的GOTO给出绝对XY坐标，MOVE指定单轴相对位移，ROTATE给出有符号角度；距离单位是毫米，角度单位是度。
# Z,DOWN/UP控制下降/抬升；MAGNET,1/0控制吸取/释放；MOTION,1/0开关运动使能。这些行首先是字符串，真正write时才发给主控。
# PING用于检查通信；ACK表示主控接受/确认当前命令。超时、ERR或中断进入异常收尾，尝试STOP和MOTION,0，finally关闭连接。
# dry_run只打印命令，不连接串口；它仍执行计划校验，所以不合格计划也会被拒绝。
# target_offset_mm虽然名字带target，现有实现把它作为视觉到磁铁的工具偏移，同时加在源点和目标点。
# =============================================================================
# 【分区】导入与异常类型
# 功能：计划校验失败、协议错误统一抛 SerialPlanError。扑克版协议与普通版相同。
# 可修改：无。
# 看情况改：无。
# 不要改：异常类型名；上层 except SerialPlanError 依赖它。
# =============================================================================
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable


class SerialPlanError(ValueError):
    """运动计划或串口协议不满足安全要求。"""


# =============================================================================
# 【分区】数值/点/矩阵校验与格式化
# 功能：拒绝 NaN/Inf；XY 点、2×2 矩阵检查；命令数字统一两位小数。
# 可修改：输出小数位须与主控解析一致。
# 看情况改：无。
# 不要改：非有限数必须抛错，否则电机会收到非法 GOTO。
# =============================================================================
def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SerialPlanError(f"{name}不是有效数字") from exc
    if not math.isfinite(result):
        raise SerialPlanError(f"{name}不是有限数字")
    return result


def _point(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SerialPlanError(f"{name}必须是[x,y]")
    return _finite_number(value[0], f"{name}.x"), _finite_number(value[1], f"{name}.y")


def _matrix2x2(value: Any, name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SerialPlanError(f"{name}必须是2x2矩阵")
    row0 = _point(value[0], f"{name}[0]")
    row1 = _point(value[1], f"{name}[1]")
    determinant = row0[0] * row1[1] - row0[1] * row1[0]
    if abs(determinant) < 1e-6:
        raise SerialPlanError(f"{name}不可逆")
    return row0, row1


def _format_number(value: float) -> str:
    rounded = round(float(value), 2)
    if abs(rounded) < 0.005:
        rounded = 0.0
    return f"{rounded:.2f}"


# =============================================================================
# 【分区】执行补偿、时序与旋转拆分
# 功能：读 offset/矩阵/旋转比例；超单次旋转上限则放下回正再吸。
# 可修改：补偿值来自 plan.json 的 execution_compensation。
# 看情况改：target_offset_mm 实际是视觉点→磁铁的工具偏移，源点和目标都加。
# 不要改：旋转拆分必须先放下再回正轴。
# =============================================================================
def _execution_compensation(plan: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """读取执行层位置与旋转补偿；旧计划默认零补偿。"""
    compensation = plan.get("execution_compensation", {})
    if compensation is None:
        compensation = {}
    if not isinstance(compensation, dict):
        raise SerialPlanError("execution_compensation必须是JSON对象")
    offset_x, offset_y = _point(
        compensation.get("target_offset_mm", [0.0, 0.0]),
        "execution_compensation.target_offset_mm",
    )
    rotation_reduction = _finite_number(
        compensation.get("rotation_magnitude_reduction_deg", 0.0),
        "execution_compensation.rotation_magnitude_reduction_deg",
    )
    if rotation_reduction < 0.0:
        raise SerialPlanError("旋转补偿减少量不能为负数")
    rotation_reduction_min_angle = _finite_number(
        compensation.get("rotation_reduction_min_angle_deg", 0.0),
        "execution_compensation.rotation_reduction_min_angle_deg",
    )
    if rotation_reduction_min_angle < 0.0 or rotation_reduction_min_angle > 180.0:
        raise SerialPlanError("旋转补偿启用角必须在0～180度之间")
    rotation_chunk = _finite_number(
        compensation.get("rotation_chunk_deg", 180.0),
        "execution_compensation.rotation_chunk_deg",
    )
    if rotation_chunk <= 0.0 or rotation_chunk > 180.0:
        raise SerialPlanError("旋转分段角必须在0～180度之间")
    rotation_cycles = _finite_number(
        compensation.get("rotation_cycles_per_revolution", 0.0),
        "execution_compensation.rotation_cycles_per_revolution",
    )
    if rotation_cycles < 0.0:
        raise SerialPlanError("旋转电机每圈周期数不能为负数")
    return (
        offset_x, offset_y, rotation_reduction, rotation_reduction_min_angle,
        rotation_chunk, rotation_cycles,
    )


def _xy_delta_compensation(
    plan: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """读取期望XY位移到实际发送位移的线性补偿。"""
    compensation = plan.get("execution_compensation", {})
    if compensation is None:
        compensation = {}
    if not isinstance(compensation, dict):
        raise SerialPlanError("execution_compensation必须是JSON对象")
    matrix = _matrix2x2(
        compensation.get("xy_command_matrix", [[1.0, 0.0], [0.0, 1.0]]),
        "execution_compensation.xy_command_matrix",
    )
    bias = _point(
        compensation.get("xy_command_bias_mm", [0.0, 0.0]),
        "execution_compensation.xy_command_bias_mm",
    )
    return matrix[0], matrix[1], bias


def _execution_timing(plan: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """读取吸取、释放、抬升和旋转后的稳定等待；旧计划默认不等待。"""
    timing = plan.get("execution_timing", {})
    if timing is None:
        timing = {}
    if not isinstance(timing, dict):
        raise SerialPlanError("execution_timing必须是JSON对象")
    fields = (
        "magnet_pickup_settle_seconds",
        "magnet_release_settle_seconds",
        "lift_settle_seconds",
        "xy_settle_seconds",
        "rotation_settle_seconds",
    )
    values = tuple(
        _finite_number(timing.get(key, 0.0), f"execution_timing.{key}")
        for key in fields
    )
    if any(value < 0.0 for value in values):
        raise SerialPlanError("执行稳定等待时间不能为负数")
    return values


def _return_origin_after_plan(plan: dict[str, Any]) -> bool:
    """读取计划结束后是否回到逻辑原点；旧计划默认保持当前位置。"""
    policy = plan.get("execution_policy", {})
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        raise SerialPlanError("execution_policy必须是JSON对象")
    value = policy.get("return_origin_after_plan", False)
    if not isinstance(value, bool):
        raise SerialPlanError("execution_policy.return_origin_after_plan必须是布尔值")
    return value


def _rotate_at_target(plan: dict[str, Any]) -> bool:
    """读取是否先移动到目标上方再旋转；旧计划默认沿用源区旋转。"""
    policy = plan.get("execution_policy", {})
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        raise SerialPlanError("execution_policy必须是JSON对象")
    value = policy.get("rotate_at_target", False)
    if not isinstance(value, bool):
        raise SerialPlanError("execution_policy.rotate_at_target必须是布尔值")
    return value


def _require_origin_before_plan(plan: dict[str, Any]) -> bool:
    """读取真实执行前是否必须验证XY逻辑原点。"""
    policy = plan.get("execution_policy", {})
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        raise SerialPlanError("execution_policy必须是JSON对象")
    value = policy.get("require_origin_before_plan", False)
    if not isinstance(value, bool):
        raise SerialPlanError("execution_policy.require_origin_before_plan必须是布尔值")
    return value


def _reduce_rotation_magnitude(
        rotation: float, reduction: float, min_angle: float = 0.0) -> float:
    """仅对达到启用角的大角度减小绝对值，并保持CW/CCW方向。"""
    if abs(rotation) < min_angle:
        return rotation
    if abs(rotation) <= reduction:
        return 0.0
    return math.copysign(abs(rotation) - reduction, rotation)


# 有整圈周期数时先把总角量化为整数周期，再将周期均匀分段。这样不会因为每段独立四舍五入而累积多转或少转。
def _split_rotation(
        rotation: float, chunk_limit: float,
        cycles_per_revolution: float = 0.0) -> list[float]:
    """按旋转机构实际步距均匀分段，避免末段过小。"""
    if abs(rotation) < 0.005:
        return [0.0]

    direction = math.copysign(1.0, rotation)
    if cycles_per_revolution <= 0.0:
        remaining = abs(rotation)
        chunks: list[float] = []
        while remaining > chunk_limit + 0.005:
            chunks.append(direction * chunk_limit)
            remaining -= chunk_limit
        if remaining >= 0.005:
            chunks.append(direction * remaining)
        return chunks

    # STM32旋转机构是离散周期：先把总角度量化一次，再均分周期数。
    # 各段实际周期之和等于总目标，避免多次四舍五入累积误差。
    total_cycles = int(
        math.floor(abs(rotation) * cycles_per_revolution / 360.0 + 0.5)
    )
    if total_cycles <= 0:
        return [0.0]
    max_chunk_cycles = max(
        1, int(math.floor(chunk_limit * cycles_per_revolution / 360.0))
    )
    chunk_count = int(math.ceil(total_cycles / max_chunk_cycles))
    base_cycles, extra = divmod(total_cycles, chunk_count)
    cycle_chunks = [
        base_cycles + (1 if index < extra else 0)
        for index in range(chunk_count)
    ]
    degrees_per_cycle = 360.0 / cycles_per_revolution
    return [direction * cycles * degrees_per_cycle for cycles in cycle_chunks]


# 分段之间放下碎片、释放磁铁并回正空载旋转轴，然后重新吸住。被放下的碎片保留已经完成的旋转角。
def _append_rotation_commands(commands: list[str], rotation_chunks: list[float]) -> None:
    """追加单次或分段旋转命令，分段时在原地重新吸取。"""
    for chunk_index, chunk in enumerate(rotation_chunks):
        commands.append(f"ROTATE,{_format_number(chunk)}")
        if chunk_index < len(rotation_chunks) - 1:
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
# 功能：坐标系、210×297、正交路径、连续 move_order、ready_for_motion。
# 可修改：无现场阈值。
# 看情况改：缺补偿字段当 0。不要为了“能发”跳过 ready_for_motion。
# 不要改：原点左上、X 右 Y 下；斜线路径必须拒绝。
# =============================================================================
def load_motion_plan(path: Path | str) -> dict[str, Any]:
    """读取由视觉程序生成的plan.json。"""
    plan_path = Path(path)
    if not plan_path.is_file():
        raise SerialPlanError(f"运动计划不存在：{plan_path}")
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SerialPlanError(f"运动计划读取失败：{exc}") from exc
    if not isinstance(payload, dict):
        raise SerialPlanError("运动计划顶层必须是JSON对象")
    return payload


def validate_motion_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """执行发送前安全联锁，并按move_order返回碎片。"""
    if plan.get("ready_for_motion") is not True:
        reasons = plan.get("failure_reasons") or []
        detail = "；".join(map(str, reasons)) if reasons else "ready_for_motion不是true"
        raise SerialPlanError(f"安全联锁拒绝发送：{detail}")
    if plan.get("all_matches_ok") is not True or plan.get("all_paths_ok") is not True:
        raise SerialPlanError("匹配或路径状态不完整，禁止发送")

    pieces = plan.get("pieces")
    if not isinstance(pieces, list) or not 1 <= len(pieces) <= 4:
        raise SerialPlanError("运动计划必须包含1～4块碎片")
    if int(plan.get("piece_count", len(pieces))) != len(pieces):
        raise SerialPlanError("piece_count与pieces数量不一致")

    workspace = plan.get("mechanical_workspace_mm") or {}
    width, height = _point(workspace.get("size", [210.0, 297.0]), "机械工作区尺寸")
    if width <= 0 or height <= 0:
        raise SerialPlanError("机械工作区尺寸无效")

    seen_ids: set[int] = set()
    seen_orders: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for index, piece in enumerate(pieces, 1):
        if not isinstance(piece, dict):
            raise SerialPlanError(f"第{index}块数据不是JSON对象")
        piece_id = int(piece.get("id", 0))
        move_order = int(piece.get("move_order", 0))
        if piece_id <= 0 or piece_id in seen_ids:
            raise SerialPlanError(f"第{index}块id无效或重复")
        if move_order <= 0 or move_order in seen_orders:
            raise SerialPlanError(f"P{piece_id}的move_order无效或重复")
        seen_ids.add(piece_id)
        seen_orders.add(move_order)
        if piece.get("status") != "ok" or piece.get("match_status") != "ok":
            raise SerialPlanError(f"P{piece_id}状态不是ok")

        source = _point(piece.get("source_pick_mm"), f"P{piece_id}.source_pick_mm")
        target = _point(piece.get("target_pick_mm"), f"P{piece_id}.target_pick_mm")
        for point_name, (x, y) in (("源吸取点", source), ("目标吸取点", target)):
            if not (0.0 <= x <= width and 0.0 <= y <= height):
                raise SerialPlanError(f"P{piece_id}{point_name}超出机械工作区")
        _finite_number(piece.get("rotation_deg_signed"), f"P{piece_id}.rotation_deg_signed")

        moves = piece.get("axis_moves_mm")
        if not isinstance(moves, list):
            raise SerialPlanError(f"P{piece_id}.axis_moves_mm必须是数组")
        previous = source
        for move_index, move in enumerate(moves, 1):
            if not isinstance(move, dict):
                raise SerialPlanError(f"P{piece_id}第{move_index}段路径格式错误")
            axis = str(move.get("axis", "")).upper()
            if axis not in {"X", "Y"}:
                raise SerialPlanError(f"P{piece_id}第{move_index}段轴只能是X或Y")
            delta = _finite_number(move.get("delta_mm"),
                                   f"P{piece_id}第{move_index}段delta_mm")
            if abs(delta) < 0.005:
                raise SerialPlanError(f"P{piece_id}第{move_index}段距离为0")
            start = _point(move.get("from_mm"), f"P{piece_id}第{move_index}段from_mm")
            end = _point(move.get("to_mm"), f"P{piece_id}第{move_index}段to_mm")
            if max(abs(start[0] - previous[0]), abs(start[1] - previous[1])) > 0.02:
                raise SerialPlanError(f"P{piece_id}第{move_index}段与前一段不连续")
            dx, dy = end[0] - start[0], end[1] - start[1]
            if axis == "X":
                if abs(dy) > 0.02 or abs(dx - delta) > 0.02:
                    raise SerialPlanError(f"P{piece_id}第{move_index}段不是合法X轴运动")
            else:
                if abs(dx) > 0.02 or abs(dy - delta) > 0.02:
                    raise SerialPlanError(f"P{piece_id}第{move_index}段不是合法Y轴运动")
            if not (0.0 <= end[0] <= width and 0.0 <= end[1] <= height):
                raise SerialPlanError(f"P{piece_id}第{move_index}段终点超出机械工作区")
            previous = end
        if max(abs(previous[0] - target[0]), abs(previous[1] - target[1])) > 0.02:
            raise SerialPlanError(f"P{piece_id}路径终点不等于目标吸取点")
        normalized.append(piece)

    normalized.sort(key=lambda item: int(item["move_order"]))
    expected_orders = list(range(1, len(normalized) + 1))
    actual_orders = [int(item["move_order"]) for item in normalized]
    if actual_orders != expected_orders:
        raise SerialPlanError("move_order必须从1连续编号")
    return normalized


# 命令顺序体现抓放过程：到源点→下降→吸住→抬起→旋转/搬运→下降→释放→抬起→旋转轴回正。函数只是生成列表，不打开串口。
# =============================================================================
# 【分区】生成 ASCII 命令列表（不打开串口）
# 功能：MOTION,1 → 逐片 GOTO/Z/MAGNET/ROTATE/MOVE → 回原点 → MOTION,0。
# 可修改：命令字符串必须与主控固件一致。
# 看情况改：是否回原点、是否在目标点旋转，由 plan 开关控制。
# 不要改：横移前 Z,UP；吸取前 Z,DOWN；PING 不在本列表。
# =============================================================================
def build_serial_commands(plan: dict[str, Any]) -> list[str]:
    """把安全计划转换为逐行ASCII命令，不包含传输层PING。

    超过单次旋转上限时，在原吸取点放下碎片、旋转轴回正并重新吸取，
    使碎片角度逐段累计，而线束偏角始终不超过配置上限。
    横移前必须抬起；每次吸取前必须下降；释放后必须抬高再回正。
    """
    pieces = validate_motion_plan(plan)
    (
        offset_x, offset_y, rotation_reduction, rotation_reduction_min_angle,
        rotation_chunk, rotation_cycles,
    ) = _execution_compensation(plan)
    xy_row0, xy_row1, xy_bias = _xy_delta_compensation(plan)
    workspace = plan.get("mechanical_workspace_mm") or {}
    width, height = _point(workspace.get("size", [210.0, 297.0]), "机械工作区尺寸")
    xy_linear_compensated = (
        abs(xy_row0[0] - 1.0) >= 1e-9 or abs(xy_row0[1]) >= 1e-9
        or abs(xy_row1[0]) >= 1e-9 or abs(xy_row1[1] - 1.0) >= 1e-9
        or abs(xy_bias[0]) >= 0.005 or abs(xy_bias[1]) >= 0.005
    )
    position_compensated = (
        abs(offset_x) >= 0.005 or abs(offset_y) >= 0.005 or xy_linear_compensated
    )

    return_origin = _return_origin_after_plan(plan)
    rotate_at_target = _rotate_at_target(plan)
    commands = ["MOTION,1", f"PLAN,{len(pieces)}"]
    for piece in pieces:
        piece_id = int(piece["id"])
        source_x, source_y = _point(piece["source_pick_mm"], "source_pick_mm")
        target_x, target_y = _point(piece["target_pick_mm"], "target_pick_mm")
        # 历史字段名是target_offset_mm，实际表示视觉坐标到磁铁中心的固定工具偏移。
        # 同一个工具偏移必须同时作用于源吸取点和目标放置点。
        command_source_x = source_x + offset_x
        command_source_y = source_y + offset_y
        command_target_x = target_x + offset_x
        command_target_y = target_y + offset_y
        if not (0.0 <= command_source_x <= width and 0.0 <= command_source_y <= height):
            raise SerialPlanError(f"P{piece_id}补偿后源吸取点超出机械工作区")
        if not (0.0 <= command_target_x <= width and 0.0 <= command_target_y <= height):
            raise SerialPlanError(f"P{piece_id}补偿后目标放置点超出机械工作区")

        rotation = _finite_number(piece["rotation_deg_signed"], "rotation_deg_signed")
        command_rotation = _reduce_rotation_magnitude(
            rotation, rotation_reduction, rotation_reduction_min_angle
        )
        rotation_chunks = _split_rotation(
            command_rotation, rotation_chunk, rotation_cycles
        )
        commands.extend([
            f"PIECE,{piece_id}",
            f"GOTO,{_format_number(command_source_x)},{_format_number(command_source_y)}",
            "Z,DOWN",
            "MAGNET,1",
            "Z,UP",
        ])
        if not rotate_at_target:
            _append_rotation_commands(commands, rotation_chunks)

        if position_compensated:
            desired_delta_x = command_target_x - command_source_x
            desired_delta_y = command_target_y - command_source_y
            if abs(desired_delta_x) < 0.005 and abs(desired_delta_y) < 0.005:
                delta_x = delta_y = 0.0
            else:
                delta_x = (
                    xy_row0[0] * desired_delta_x
                    + xy_row0[1] * desired_delta_y
                    + xy_bias[0]
                )
                delta_y = (
                    xy_row1[0] * desired_delta_x
                    + xy_row1[1] * desired_delta_y
                    + xy_bias[1]
                )
            corrected_target_x = command_source_x + delta_x
            corrected_target_y = command_source_y + delta_y
            if not (0.0 <= corrected_target_x <= width and 0.0 <= corrected_target_y <= height):
                raise SerialPlanError(f"P{piece_id}XY线性补偿后目标超出机械工作区")
            if abs(delta_x) >= 0.005:
                commands.append(f"MOVE,X,{_format_number(delta_x)}")
            if abs(delta_y) >= 0.005:
                commands.append(f"MOVE,Y,{_format_number(delta_y)}")
        else:
            for move in piece["axis_moves_mm"]:
                axis = str(move["axis"]).upper()
                delta = _finite_number(move["delta_mm"], "delta_mm")
                commands.append(f"MOVE,{axis},{_format_number(delta)}")
        if rotate_at_target:
            _append_rotation_commands(commands, rotation_chunks)
        commands.extend(["Z,DOWN", "MAGNET,0", "Z,UP"])
        # 前面的完整分段均已各自回正，结束时只需回正最后一段。
        last_chunk = rotation_chunks[-1]
        if abs(last_chunk) >= 0.005:
            commands.append(f"ROTATE,{_format_number(-last_chunk)}")
        commands.append(f"ENDPIECE,{piece_id}")
    if return_origin:
        commands.append("GOTO,0.00,0.00")
    commands.extend(["ENDPLAN", "MOTION,0"])
    return commands


def print_serial_commands(commands: Iterable[str], output: Callable[[str], None] = print) -> int:
    """打印dry-run指令并返回指令数量。"""
    count = 0
    output("[DRY-RUN] 未打开串口，以下为计划发送的指令：")
    for command in commands:
        output(f"[TX] {command}")
        count += 1
    output(f"[DRY-RUN] 共{count}条指令，未驱动任何电机")
    return count


# =============================================================================
# 【分区】串口应答、安全停机
# 功能：等到 ACK；超时/ERR 抛错；异常时尽力发 STOP、MOTION,0。
# 可修改：ACK 文本须与固件一致。
# 看情况改：板子若回 OK 而不是 ACK，改接受列表或改固件。
# 不要改：超时必须停发；安全停机不等 ACK。
# =============================================================================
def _read_response(connection: Any) -> str:
    raw = connection.readline()
    if not raw:
        raise SerialPlanError("等待ACK超时，已停止发送")
    try:
        response = raw.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise SerialPlanError("下位机返回了非ASCII数据") from exc
    if response == "ACK" or response.startswith("ACK,"):
        return response
    if response.startswith("ERR"):
        raise SerialPlanError(f"下位机拒绝命令：{response}")
    raise SerialPlanError(f"无法识别的下位机响应：{response!r}")


def _validate_origin_response(response: str, tolerance: float = 0.01) -> None:
    """验证POS响应中的XY逻辑坐标位于原点。"""
    fields = response.split(",")
    if len(fields) < 4 or fields[0] != "ACK" or fields[1] != "POS":
        raise SerialPlanError(f"POS响应格式错误：{response!r}")
    try:
        x = float(fields[2])
        y = float(fields[3])
    except ValueError as exc:
        raise SerialPlanError(f"POS坐标无法解析：{response!r}") from exc
    if abs(x) > tolerance or abs(y) > tolerance:
        raise SerialPlanError(
            f"执行前XY未在逻辑原点：X={x:g}, Y={y:g}；请先回零"
        )


def _best_effort_safety_shutdown(
    connection: Any,
    output: Callable[[str], None],
) -> None:
    """异常收尾：不等待ACK，尽力发送STOP和MOTION,0后再关闭串口。"""
    for command in ("STOP", "MOTION,0"):
        try:
            connection.write((command + "\n").encode("ascii"))
            if hasattr(connection, "flush"):
                connection.flush()
        except Exception as exc:
            try:
                output(f"[WARN] 安全收尾命令{command}发送失败：{exc}")
            except Exception:
                pass
        else:
            try:
                output(f"[SAFE] 已尽力发送{command}（不等待ACK）")
            except Exception:
                pass


# 每条发送后立即_read_response；根据当前命令和下一命令决定等待时间。sent包含预检命令，不只是碎片运动命令数。
# =============================================================================
# 【分区】逐条发送并等待 ACK
# 功能：开串口，按命令类型插入稳定延时，失败则安全停机并关口。
# 可修改：settle 秒数来自 config；波特率 115200。
# 看情况改：DTR/RTS 默认 False 防复位。
# 不要改：每条后立刻等 ACK；finally 关闭连接。
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
    if not str(port).strip():
        raise SerialPlanError("真实发送必须指定串口，例如--port COM5")
    if int(baudrate) <= 0:
        raise SerialPlanError("波特率必须大于0")
    timeout = _finite_number(ack_timeout_seconds, "ACK超时时间")
    if timeout <= 0:
        raise SerialPlanError("ACK超时时间必须大于0")
    pickup_settle = _finite_number(
        magnet_pickup_settle_seconds, "磁铁吸取稳定等待时间"
    )
    release_settle = _finite_number(
        magnet_release_settle_seconds, "磁铁释放稳定等待时间"
    )
    lift_settle = _finite_number(lift_settle_seconds, "Z轴抬升稳定等待时间")
    xy_settle = _finite_number(xy_settle_seconds, "XY停止稳定等待时间")
    rotation_settle = _finite_number(rotation_settle_seconds, "旋转稳定等待时间")
    if min(pickup_settle, release_settle, lift_settle, xy_settle, rotation_settle) < 0.0:
        raise SerialPlanError("执行稳定等待时间不能为负数")

    is_real_serial = serial_factory is None
    if is_real_serial:
        try:
            import serial
        except ImportError as exc:
            raise SerialPlanError("缺少pyserial，请执行python -m pip install -r requirements.txt") from exc

        # 先配置DTR/RTS再open()，避免Windows打开CH340时产生复位脉冲。
        # 这与直接Serial(port=...)相比能降低STM32被意外复位的概率。
        connection = serial.Serial()
        connection.port = str(port)
        connection.baudrate = int(baudrate)
        connection.timeout = timeout
        connection.write_timeout = timeout
        connection.dtr = False
        connection.rts = False
        connection.open()
    else:
        connection = serial_factory(
            # timeout=timeout（等待超时时间（秒））；取值过程：timeout（等待超时时间（秒））；write_timeout=timeout（write·超时）；
            port=str(port), baudrate=int(baudrate), timeout=timeout, write_timeout=timeout
        )

    sent = 0
    try:
        # 打开后等待主控和USB串口稳定，再清空历史收发数据。
        if not is_real_serial:
            if hasattr(connection, "dtr"):
                connection.dtr = False
            if hasattr(connection, "rts"):
                connection.rts = False
        time.sleep(1.0)
        if hasattr(connection, "reset_input_buffer"):
            connection.reset_input_buffer()
        if hasattr(connection, "reset_output_buffer"):
            connection.reset_output_buffer()
        preflight_commands = ["PING"]
        if require_origin_before_plan:
            preflight_commands.append("POS")
        all_commands = [*preflight_commands, *list(commands)]
        output(f"[SERIAL] 已打开{port}，baud={int(baudrate)}，等待ACK超时={timeout:.2f}s")
        for command_index, command in enumerate(all_commands):
            next_command = (
                all_commands[command_index + 1]
                if command_index + 1 < len(all_commands) else ""
            )
            payload = (command + "\n").encode("ascii")
            output(f"[TX] {command}")
            connection.write(payload)
            if hasattr(connection, "flush"):
                connection.flush()
            response = _read_response(connection)
            output(f"[RX] {response}")
            sent += 1
            if command == "POS":
                _validate_origin_response(response)
                output("[OK] 执行前XY逻辑原点验证通过")
            settle_seconds = 0.0
            settle_reason = ""
            if command == "Z,DOWN":
                settle_seconds = max(lift_settle, 0.3)
                settle_reason = "Z下降到位稳定"
            elif command == "MAGNET,1":
                settle_seconds = pickup_settle
                settle_reason = "磁铁吸取稳定"
            elif command == "MAGNET,0":
                settle_seconds = release_settle
                settle_reason = "磁铁释放稳定"
            elif command == "Z,UP" and (
                next_command.startswith("ROTATE,") or next_command.startswith("MOVE,")
            ):
                settle_seconds = lift_settle
                settle_reason = "Z轴抬升稳定"
            elif command.startswith("MOVE,") and not next_command.startswith("MOVE,"):
                settle_seconds = xy_settle
                settle_reason = "XY停止稳定"
            elif command.startswith("ROTATE,"):
                settle_seconds = rotation_settle
                settle_reason = "旋转稳定"
            if settle_seconds > 0.0:
                output(f"[WAIT] {settle_reason}{settle_seconds:.2f}s")
                time.sleep(settle_seconds)
            if command_delay_seconds > 0.0:
                output(f"[WAIT] 命令间隔{command_delay_seconds:.2f}s")
                time.sleep(command_delay_seconds)
        output(f"[OK] 串口计划发送完成，共{sent}条（含PING）")
        return sent
    except BaseException:
        # 包括ACK超时、下位机ERR、串口异常以及用户Ctrl+C，均尝试先停机再关串口。
        _best_effort_safety_shutdown(connection, output)
        raise
    finally:
        connection.close()


# =============================================================================
# 【分区】对外入口 send_plan_file
# 功能：读计划 → 校验 → 生成命令；dry_run 只打印。
# 可修改：--port、波特率、ack 超时。
# 看情况改：实发前先 dry_run。
# 不要改：dry_run=False 且 port 为空要报错；不合格计划发送前必须拒绝。
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
    plan = load_motion_plan(path)
    commands = build_serial_commands(plan)
    require_origin = _require_origin_before_plan(plan)
    (
        pickup_settle, release_settle, lift_settle, xy_settle, rotation_settle,
    ) = _execution_timing(plan)
    if dry_run:
        if max(
            pickup_settle, release_settle, lift_settle, xy_settle, rotation_settle
        ) > 0.0:
            output(
                f"[DRY-RUN] 稳定等待：吸取后{pickup_settle:.2f}s，"
                f"释放后{release_settle:.2f}s，抬升后{lift_settle:.2f}s，"
                f"XY停止后{xy_settle:.2f}s，旋转后{rotation_settle:.2f}s"
            )
        return print_serial_commands(commands, output)
    return send_serial_commands(
        commands,
        port=port or "",
        baudrate=baudrate,
        ack_timeout_seconds=ack_timeout_seconds,
        magnet_pickup_settle_seconds=pickup_settle,
        magnet_release_settle_seconds=release_settle,
        lift_settle_seconds=lift_settle,
        xy_settle_seconds=xy_settle,
        rotation_settle_seconds=rotation_settle,
        command_delay_seconds=command_delay_seconds,
        require_origin_before_plan=require_origin,
        serial_factory=serial_factory,
        output=output,
    )
