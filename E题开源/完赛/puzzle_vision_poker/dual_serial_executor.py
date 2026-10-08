"""Coordinated Grbl + STM32 executor for the real three-piece puzzle.

The vision planner owns only A4 coordinates.  This module owns the physical
controllers and never opens a port until ``execute`` is called explicitly.
It is deliberately independent from the legacy single-port transport so the
old dry-run and simulation paths remain available.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import signal
import threading
import time
import uuid
from typing import Any, Callable, Iterable

from serial_connection import open_serial, port_identity, reset_banner, transport_failure


class DualExecutionError(RuntimeError):
    """A preflight, protocol, coordinate or motion failure."""


class SerialTransportError(DualExecutionError):
    """Fatal transport failure; the current connection must not be resumed."""


class SerialReplyTimeout(DualExecutionError):
    """No complete reply within a bounded polling interval."""


class ControllerResetError(DualExecutionError):
    """Boot output after handshake invalidates all coordinates and approval."""


@dataclass(frozen=True)
class CoordinateCalibration:
    matrix: tuple[tuple[float, float], tuple[float, float]]
    bias_mm: tuple[float, float]
    tool_offset_mm: tuple[float, float]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CoordinateCalibration":
        xy = config.get("xy_transform", {})
        matrix = xy.get("matrix", [[1.0, 0.0], [0.0, 1.0]])
        bias = xy.get("bias_mm", [0.0, 0.0])
        offset = config.get("tool_offset_mm", [0.0, 0.0])
        if (len(matrix) != 2 or any(len(row) != 2 for row in matrix)
                or len(bias) != 2 or len(offset) != 2):
            raise DualExecutionError("XY 标定矩阵或工具偏移格式错误")
        values = [*matrix[0], *matrix[1], *bias, *offset]
        if not all(math.isfinite(float(value)) for value in values):
            raise DualExecutionError("XY 标定参数必须是有限数")
        determinant = float(matrix[0][0]) * float(matrix[1][1]) - \
            float(matrix[0][1]) * float(matrix[1][0])
        if abs(determinant) < 1e-6:
            raise DualExecutionError("XY 标定矩阵不可逆")
        return cls(
            matrix=((float(matrix[0][0]), float(matrix[0][1])),
                    (float(matrix[1][0]), float(matrix[1][1]))),
            bias_mm=(float(bias[0]), float(bias[1])),
            tool_offset_mm=(float(offset[0]), float(offset[1])),
        )

    def map_point(self, point: Iterable[float]) -> tuple[float, float]:
        x, y = (float(value) for value in point)
        x += self.tool_offset_mm[0]
        y += self.tool_offset_mm[1]
        return (
            self.matrix[0][0] * x + self.matrix[0][1] * y + self.bias_mm[0],
            self.matrix[1][0] * x + self.matrix[1][1] * y + self.bias_mm[1],
        )


def expected_real_piece_count(config: dict[str, Any]) -> int:
    profile = config.get("puzzle_profile", "poker")
    expected = {"poker": 3, "ordinary": 4}.get(profile)
    if expected is None or int(config.get("expected_piece_count", 0)) != expected:
        raise DualExecutionError("正式拼图模式与碎片数量不一致：扑克3片、普通4片")
    return expected


def validate_real_config(config: dict[str, Any], *, empty_run: bool = False) -> None:
    """Reject an uncalibrated or incompatible production configuration."""
    coordinate = config.get("coordinate_system", {})
    expected = {
        "origin": "A4_top_left",
        "x_positive": "right",
        "y_positive": "down",
        "z_positive": "down",
        "rotation_positive": "clockwise",
    }
    for key, value in expected.items():
        if coordinate.get(key) != value:
            raise DualExecutionError(f"坐标约定不匹配：{key}={coordinate.get(key)!r}")
    if config.get("source_region") != "bottom" or config.get("target_region") != "top":
        raise DualExecutionError("正式流程要求碎片在下半区、目标在上半区")
    expected_real_piece_count(config)
    motion_cfg = config.get("motion", {})
    manual_origin = str(motion_cfg.get("origin_mode", "")).lower() == "manual"
    # Empty-path acceptance is optional for direct production. It must not be
    # fabricated: the saved per-axis calibration and session origin still apply.
    require_empty_run = config.get("require_empty_run", True) is not False
    if not empty_run and require_empty_run and not bool(config.get("calibration_complete", False)):
        raise DualExecutionError("机械和相机标定尚未完成")
    if not empty_run and require_empty_run and config.get("require_approved_session", False):
        from approved_plan import config_digest
        accepted = config.get("mechanical_acceptance", {})
        if not accepted.get("operator_confirmed") or accepted.get("config_sha256") != config_digest(config):
            raise DualExecutionError("缺少当前配置的机械空载验收记录")
    calibration_state = config.get("calibration_state")
    if (manual_origin or empty_run or not require_empty_run) and not isinstance(calibration_state, dict):
        raise DualExecutionError("人工基准模式缺少分项标定状态")
    if isinstance(calibration_state, dict):
        required = ("camera", "xy", "z", "rotation")
        if not manual_origin:
            required += ("grbl_homed",)
        incomplete = [name for name in required
                      if not bool(calibration_state.get(name, False))]
        if incomplete:
            raise DualExecutionError("标定状态未完成：" + ",".join(incomplete))
    CoordinateCalibration.from_config(config)
    z = config.get("z", {})
    for key in ("safe", "pickup", "place"):
        if key not in z or not math.isfinite(float(z[key])):
            raise DualExecutionError(f"缺少有效Z标定：{key}")
    # With Z+ downward, pickup/place must be below the empty safe height.
    if float(z["safe"]) >= min(float(z["pickup"]), float(z["place"])):
        raise DualExecutionError("按Z+下降约定，safe_z 必须小于 pickup_z 和 place_z")
    rotation = config.get("rotation", {})
    def number(mapping, key, default, *, minimum=0.0, positive=False):
        try:
            value = float(mapping.get(key, default))
        except (TypeError, ValueError) as exc:
            raise DualExecutionError(f"运动参数无效：{key}") from exc
        if not math.isfinite(value) or value < minimum or (positive and value <= 0):
            raise DualExecutionError(f"运动参数无效：{key}")
        return value

    travel = number(z, "travel", z["safe"], minimum=-math.inf)
    if travel > float(z["safe"]):
        raise DualExecutionError("按Z+下降约定，travel_z 必须小于或等于 safe_z")
    z_feed = number(z, "feed_mm_min", 60.0, positive=True)
    number(z, "contact_feed_mm_min", z_feed, positive=True)
    margin = number(z, "contact_margin_mm", 0.0)
    if margin > min(float(z["pickup"]), float(z["place"])) - float(z["safe"]):
        raise DualExecutionError("Z接触减速区不能超过安全高度到接触面的距离")
    number(motion_cfg, "xy_feed_mm_min", 300.0, positive=True)
    for mapping, key, default in (
            (z, "settle_s", 0.0), (rotation, "settle_s", 0.0),
            (motion_cfg, "xy_settle_s", 0.0),
            (motion_cfg, "pickup_settle_s", 0.8),
            (motion_cfg, "release_settle_s", 0.3)):
        number(mapping, key, default)
    if int(rotation.get("pulses_per_revolution", 0)) != 3200:
        raise DualExecutionError("R轴默认必须为3200脉冲/圈")
    if rotation.get("positive_direction") != "clockwise":
        raise DualExecutionError("R轴正方向必须是顺时针")


def _decode_line(raw: bytes) -> str:
    return raw.decode("ascii", errors="strict").strip()


class _Wire:
    def __init__(self, port: str, baudrate: int, factory=None):
        self.port = open_serial(port, int(baudrate), 0.1, 0.5, factory)
        self.logical_port = port
        self.opened_identity = port_identity(port)
        self.lock = threading.RLock()
        self.buffer = bytearray()
        self.trace = None

    def _transport_error(self, operation: str, exc: Exception):
        message = transport_failure(self.logical_port, operation, exc, self.opened_identity)
        if self.trace:
            self.trace("error", message)
        raise SerialTransportError(message) from exc

    def _send(self, data: bytes) -> None:
        try:
            written = self.port.write(data)
            if written is not None and written != len(data):
                raise OSError(f"串口仅写入{written}/{len(data)}字节")
            if hasattr(self.port, "flush"):
                self.port.flush()
        except OSError as exc:
            self._transport_error("写入", exc)

    def write(self, line: str) -> None:
        with self.lock:
            if self.trace:
                self.trace("send", line)
            self._send((line.rstrip("\n") + "\n").encode("ascii"))

    def realtime(self, data: bytes) -> None:
        with self.lock:
            if self.trace:
                self.trace("realtime", data.hex())
            self._send(data)

    def read(self, timeout: float = 1.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if b"\n" in self.buffer:
                raw, _, rest = self.buffer.partition(b"\n")
                self.buffer = bytearray(rest)
                if raw.strip():
                    line = _decode_line(raw)
                    if self.trace:
                        self.trace("receive", line)
                    return line
                continue
            try:
                self.buffer.extend(self.port.readline())
            except OSError as exc:
                self._transport_error("读取", exc)
            if len(self.buffer) > 8192:
                raise DualExecutionError("串口接收数据过长或缺少换行")
        raise SerialReplyTimeout(f"串口{getattr(self.port, 'port', '')}应答超时")

    def close(self) -> None:
        with self.lock:
            self.port.close()


class DualSerialExecutor:
    """One owner for both physical controllers."""

    def __init__(self, config: dict[str, Any], serial_factory=None,
                 clock: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep,
                 empty_run: bool = False):
        validate_real_config(config, empty_run=empty_run)
        self.empty_run = empty_run
        self.session_id = uuid.uuid4().hex
        self.approved_plan_id = None
        self.origin_valid = False
        self.stop_confirmed = False
        self.stop_details: dict[str, Any] = {}
        self.failure_action = ""
        self.session_guard = None
        serial_cfg = config["serial"]
        self.config = config
        self.clock = clock
        self.sleeper = sleeper
        self.factory = serial_factory
        self.grbl: _Wire | None = None
        self.stm32: _Wire | None = None
        self.locked = False
        self.last_action = ""
        self._last_heartbeat = 0.0
        self._magnet_expected = False
        self._r_x100 = 0

        self.grbl_port = str(serial_cfg["grbl_port"])
        self.stm32_port = str(serial_cfg["stm32_port"])
        self.baudrate = int(serial_cfg.get("baudrate", 115200))
        self.timeout = float(serial_cfg.get("timeout_s", 1.0))

    def open(self) -> None:
        if self.grbl is not None or self.stm32 is not None:
            raise DualExecutionError("双串口执行器已经打开")
        try:
            self.grbl = _Wire(self.grbl_port, self.baudrate, self.factory)
            self.stm32 = _Wire(self.stm32_port, self.baudrate, self.factory)
            if self.factory is None:
                self.sleeper(2.0)
            self._drain(self.grbl)
            self._drain(self.stm32)
        except Exception:
            self.close()
            raise

    def _drain(self, wire: _Wire) -> None:
        until = self.clock() + 0.15
        received = bytearray()
        while self.clock() < until:
            try:
                received.extend(wire.port.readline())
            except OSError as exc:
                wire._transport_error("启动读取", exc)
            while b"\n" in received:
                raw, _, rest = received.partition(b"\n")
                received = bytearray(rest)
                line = _decode_line(raw)
                if reset_banner(line):
                    # Opening can still glitch at driver/hardware level. A
                    # pre-open human confirmation cannot validate reset MPos.
                    self.locked = True
                    self.origin_valid = False
                    self.approved_plan_id = None
                    raise ControllerResetError(
                        f"连接时检测到控制器重启：{line}；原点确认已失效，请重新连接并核对机械基准")
            if len(received) > 8192:
                raise DualExecutionError("串口启动响应过长或缺少换行")

    def close(self) -> None:
        self.origin_valid = False
        self.approved_plan_id = None
        for wire in (self.grbl, self.stm32):
            if wire is not None:
                try:
                    wire.close()
                except Exception:
                    pass
        self.grbl = None
        self.stm32 = None

    def _require_open(self) -> tuple[_Wire, _Wire]:
        if self.grbl is None or self.stm32 is None:
            raise DualExecutionError("双串口尚未打开")
        return self.grbl, self.stm32

    def _check_reset(self, response: str) -> None:
        if reset_banner(response):
            self.stop_both("控制器重启，本次基准失效")
            raise ControllerResetError(
                f"检测到控制器重启：{response}；本次原点与方案批准已失效，请重新连接并确认基准")

    def _stm_request(self, command: str, expected: str | Callable[[str], bool]) -> str:
        _, stm = self._require_open()
        stm.write(command)
        deadline = self.clock() + self.timeout
        while self.clock() < deadline:
            response = stm.read(max(0.01, deadline - self.clock()))
            self._check_reset(response)
            ok = expected(response) if callable(expected) else response == expected
            if ok:
                return response
            # A rotation heartbeat may arrive after its completion ACK.
            if response.startswith("ACK,IOSTATUS,") and command != "IOSTATUS":
                self._check_io(response)
                continue
            raise DualExecutionError(f"STM32拒绝{command}：{response}")
        raise DualExecutionError(f"STM32等待{command}应答超时")

    def stm_status(self) -> dict[str, Any]:
        response = self._stm_request("STATUS", lambda line: line.startswith("ACK,STATUS,"))
        fields = response.split(",")
        if len(fields) < 8:
            raise DualExecutionError(f"STATUS格式错误：{response}")
        try:
            return {
                "mode": fields[2], "motion": fields[3], "origin": fields[4],
                "x_x100": int(fields[5]), "y_x100": int(fields[6]),
                "z_x100": int(fields[7]), "r_x100": int(fields[8])
                if len(fields) > 8 else 0,
            }
        except ValueError as exc:
            raise DualExecutionError(f"STATUS数值错误：{response}") from exc

    def stm_iostatus(self) -> dict[str, Any]:
        response = self._stm_request("IOSTATUS", lambda line: line.startswith("ACK,IOSTATUS,"))
        fields = response.split(",")
        values: dict[str, Any] = {}
        for index in range(2, len(fields) - 1, 2):
            values[fields[index].lower()] = fields[index + 1]
        return values

    def grbl_status(self) -> dict[str, Any]:
        grbl, _ = self._require_open()
        grbl.realtime(b"?")
        deadline = self.clock() + self.timeout
        response = ""
        while self.clock() < deadline:
            response = grbl.read(max(0.01, deadline - self.clock()))
            self._check_reset(response)
            if response.startswith("<"):
                break
            if response == "ok" or response.startswith("["):
                continue
            raise DualExecutionError(f"Grbl状态应答异常：{response}")
        if not response.startswith("<") or "|" not in response:
            raise DualExecutionError(f"Grbl状态格式错误：{response}")
        body = response[1:response.rfind(">")]
        fields = body.split("|")
        status: dict[str, Any] = {"state": fields[0].split(":", 1)[0], "raw_state": fields[0]}
        for field in fields[1:]:
            if ":" in field:
                key, value = field.split(":", 1)
                status[key] = value
        return status

    def preflight(self, allow_hold: bool = False) -> dict[str, Any]:
        grbl_status = self.grbl_status()
        stm_ping = self._stm_request("PING", "ACK,PING")
        stm_status = self.stm_status()
        stm_io = self.stm_iostatus()
        if grbl_status.get("state") not in ({"Idle", "Hold"} if allow_hold else {"Idle"}):
            raise DualExecutionError(f"Grbl未处于Idle：{grbl_status.get('state')}")
        motion_cfg = self.config.get("motion", {})
        manual_origin = str(motion_cfg.get("origin_mode", "")).lower() == "manual"
        if manual_origin:
            raw_mpos = str(grbl_status.get("MPos", ""))
            try:
                mpos = tuple(float(value) for value in raw_mpos.split(","))
            except (TypeError, ValueError) as exc:
                raise DualExecutionError("人工基准模式需要有效 Grbl MPos") from exc
            if len(mpos) != 3 or any(not math.isfinite(value) for value in mpos):
                raise DualExecutionError("人工基准模式需要有效 Grbl MPos")
            tolerance = float(motion_cfg.get("manual_origin_tolerance_mm", 0.5))
            if tolerance < 0 or any(abs(value) > tolerance for value in mpos):
                raise DualExecutionError(
                    "人工基准未就绪：需用受控移动返回基准并核对 MPos；手推不会更新坐标"
                )
        elif bool(motion_cfg.get("require_homing", True)):
            state = self.config.get("calibration_state", {})
            if (isinstance(state, dict) and "grbl_homed" in state
                    and not bool(state.get("grbl_homed"))):
                raise DualExecutionError("Grbl尚未完成回零，禁止执行")
        # STM32 powers up in SIM and only enters REAL after MOTION,1.  The
        # old preflight rejected SIM before execute() could open that gate,
        # making every normal boot impossible.  Accept either safe idle mode
        # here; execute() enables and verifies REAL immediately afterwards.
        if stm_status.get("mode") not in {"SIM", "REAL"}:
            raise DualExecutionError(f"STM32模式无效：{stm_status.get('mode')}")
        if stm_status.get("motion") != "IDLE":
            raise DualExecutionError("STM32仍在运动")
        if stm_io.get("magnet") != "0" or stm_io.get("aux") != "0":
            raise DualExecutionError("启动前电磁铁及AUX必须关闭")
        return {"grbl": grbl_status, "stm_ping": stm_ping,
                "stm": stm_status, "io": stm_io}

    def confirm_origin(self, plan_id: str, session_id: str) -> None:
        """Called only after an explicit operator confirmation on this connection."""
        if self.locked or session_id != self.session_id:
            raise DualExecutionError("连接会话失效")
        status = self.preflight(allow_hold=True)
        if status["grbl"]["state"] == "Hold":
            if status["grbl"].get("raw_state") != "Hold:0":
                raise DualExecutionError("Grbl尚未完全停止，不能恢复")
            # The screen explicitly confirms cancellation of the previous jog.
            self.grbl.realtime(b"\x85")
            self.sleeper(0.2)
            self.grbl.realtime(b"~")
            self.sleeper(0.2)
        self.preflight()
        self.origin_valid = True
        self.approved_plan_id = plan_id

    def _heartbeat(self, force: bool = False) -> None:
        if self.session_guard:
            self.session_guard()
        if not force and self.clock() - self._last_heartbeat < 0.8:
            return
        response = self._stm_request("IOSTATUS", lambda line: line.startswith("ACK,IOSTATUS,"))
        self._check_io(response)
        self._last_heartbeat = self.clock()

    def _check_io(self, response: str) -> None:
        fields = response.split(",")
        io = dict(zip(fields[2::2], fields[3::2]))
        if self._magnet_expected and (io.get("AUX") != "1" or io.get("MAGNET") != "1"):
            raise DualExecutionError("携片期间电磁铁输出已失效")

    @staticmethod
    def _mpos(status: dict[str, Any]) -> tuple[float, float, float]:
        try:
            values = tuple(float(v) for v in str(status.get("MPos", "")).split(","))
        except ValueError as exc:
            raise DualExecutionError("Grbl MPos数值无效") from exc
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise DualExecutionError("Grbl缺少有效MPos")
        return values

    def _wait_grbl_idle(self, timeout: float = 45.0,
                        expected: dict[int, float] | None = None) -> dict[str, Any]:
        deadline = self.clock() + timeout
        startup_deadline = self.clock() + 0.5
        seen_motion = False
        while self.clock() < deadline:
            status = self.grbl_status()
            self._heartbeat()
            state = str(status.get("state", ""))
            if state == "Idle":
                mismatch = expected and any(abs(self._mpos(status)[axis] - target) > .1
                                            for axis, target in expected.items())
                if not mismatch:
                    return status
                # A jog ACK can precede the first Jog report. Poll briefly;
                # never resend a relative move or accept a wrong endpoint.
                if seen_motion or self.clock() >= startup_deadline:
                    raise DualExecutionError(f"Grbl报告的终点与命令不符：{status.get('MPos')}")
            elif state in {"Jog", "Run"}:
                seen_motion = True
            if state.startswith(("Alarm", "Door", "Hold")):
                raise DualExecutionError(f"Grbl运动中断：{state}")
            self.sleeper(0.05)
        raise DualExecutionError("Grbl运动完成超时")

    def _grbl_command(self, command: str, motion_timeout: float = 45.0,
                      expected: dict[int, float] | None = None) -> dict[str, Any]:
        if self.session_guard:
            self.session_guard()
        grbl, _ = self._require_open()
        grbl.write(command)
        deadline = self.clock() + self.timeout
        while True:
            response = grbl.read(max(0.01, deadline - self.clock()))
            self._check_reset(response)
            if response.lower() == "ok":
                break
            if response.startswith("<") and self.clock() < deadline:
                state = response[1:].split("|", 1)[0]
                if state.split(":")[0] in {"Idle", "Jog", "Run"}:
                    continue
            detail = "；控制器行解析拒绝，指令未自动重发" if response == "error:1" else ""
            raise DualExecutionError(f"Grbl拒绝{command}：{response}{detail}")
        return self._wait_grbl_idle(motion_timeout, expected)

    def _xy_shortcut(self) -> bool:
        """是否跳过 L 形拐点、直接对角移动（motion.xy_shortcut_path）。"""
        motion_cfg = self.config.get("motion", {}) or {}
        return bool(motion_cfg.get("xy_shortcut_path", False))

    def move_xy_direct(self, x: float, y: float, feed: float) -> None:
        """两点直连移动（对角）。无论开关如何都走对角，供路径行走强制使用。

        与 move_xy 的区别：move_xy 在开关打开时也走对角，但会被外部按
        waypoint 逐段调用；move_xy_direct 用于显式跳过 waypoint 的场景。
        """
        if not all(math.isfinite(value) for value in (x, y, feed)) or feed <= 0:
            raise DualExecutionError("XY目标或速度无效")
        self.last_action = f"XYdirect({x:.2f},{y:.2f})"
        current = self._mpos(self.grbl_status())
        dx = abs(x - current[0])
        dy = abs(y - current[1])
        if dx < 0.001 and dy < 0.001:
            return
        motion_cfg = self.config.get("motion", {}) or {}
        the_feed = feed
        if (dx >= 0.001 and dy >= 0.001
                and motion_cfg.get("xy_simultaneous_compensate_components", True)):
            the_feed = feed * math.sqrt(2.0)
        path = math.hypot(dx, dy)
        status = self._grbl_command(
            f"$J=G90 G21 G53 X{x:.3f} Y{y:.3f} F{the_feed:.1f}",
            max(15.0, path / the_feed * 60.0 + 10.0), {0: x, 1: y})
        got = self._mpos(status)
        if abs(got[0] - x) > 0.1 or abs(got[1] - y) > 0.1:
            raise DualExecutionError(f"Grbl报告的XY终点与命令不符：{status.get('MPos')}")

    def _xy_walk_segment(self, x: float, y: float, feed: float) -> None:
        """按开关分派：短路时两点直连，否则沿 waypoint 逐段（兼容历史行为）。"""
        if self._xy_shortcut():
            self.move_xy_direct(x, y, feed)
        else:
            self.move_xy(x, y, feed)

    def move_xy(self, x: float, y: float, feed: float) -> None:
        if not all(math.isfinite(value) for value in (x, y, feed)) or feed <= 0:
            raise DualExecutionError("XY目标或速度无效")
        self.last_action = f"XY({x:.2f},{y:.2f})"
        current = self._mpos(self.grbl_status())

        # 可切换的对角运动（motion.xy_simultaneous）。
        # 关闭时走下面的直角路径，行为与历史版本逐字节一致。
        # 对角时 X/Y 同时插补，两轴各分到 feed/sqrt(2) 的分量，
        # 因此默认把该轴进给乘 sqrt(2) 以补偿，使【路径】速度维持不变。
        motion_cfg = self.config.get("motion", {}) or {}
        if motion_cfg.get("xy_simultaneous", False):
            dx = abs(x - current[0])
            dy = abs(y - current[1])
            if dx < 0.001 and dy < 0.001:
                return
            diagonal_feed = feed
            # 只有两轴都真的移动时才有对角分量损失；单轴移动不该补偿。
            if (dx >= 0.001 and dy >= 0.001
                    and motion_cfg.get("xy_simultaneous_compensate_components", True)):
                diagonal_feed = feed * math.sqrt(2.0)
            path = math.hypot(dx, dy)
            status = self._grbl_command(
                f"$J=G90 G21 G53 X{x:.3f} Y{y:.3f} F{diagonal_feed:.1f}",
                max(15.0, path / diagonal_feed * 60.0 + 10.0), {0: x, 1: y})
            got = self._mpos(status)
            if abs(got[0] - x) > 0.1 or abs(got[1] - y) > 0.1:
                raise DualExecutionError(f"Grbl报告的XY终点与命令不符：{status.get('MPos')}")
            return

        # Execute orthogonal segments in the calibrated machine frame, even
        # when a work-coordinate offset was left by a different application.
        for index, axis, target in ((0, "X", x), (1, "Y", y)):
            distance = abs(target - current[index])
            if distance < 0.001:
                continue
            status = self._grbl_command(
                f"$J=G90 G21 G53 {axis}{target:.3f} F{feed:.1f}",
                max(15.0, distance / feed * 60.0 + 10.0), {index: target})
            current = self._mpos(status)
            if abs(current[index] - target) > 0.1:
                raise DualExecutionError(f"Grbl报告的{axis}终点与命令不符")

    def move_z(self, delta: float, feed: float) -> None:
        if not math.isfinite(delta) or not math.isfinite(feed) or feed <= 0:
            raise DualExecutionError("Z目标或速度无效")
        self.last_action = f"Z{delta:+.2f}"
        before = self._mpos(self.grbl_status())
        after = self._mpos(self._grbl_command(
            f"$J=G91 G21 Z{delta:.3f} F{feed:.1f}",
            max(15.0, abs(delta) / feed * 60.0 + 10.0), {2: before[2] + delta}))
        if abs(after[2] - before[2] - delta) > 0.1:
            raise DualExecutionError("Grbl报告的Z终点与命令不符")

    def _check_z_position(self, expected: float) -> float:
        """Verify the current phase before moving horizontally or vertically."""
        status = self.grbl_status()
        current = self._mpos(status)[2]
        if status.get("state") != "Idle" or abs(current - expected) > 0.1:
            raise DualExecutionError(f"Z阶段位置异常：期望{expected:.3f}，反馈{current:.3f}")
        return current

    def move_z_to(self, target: float, feed: float, *, expected_start: float) -> None:
        """Use reported MPos, not an assumed safe height, to derive a relative jog."""
        if not all(math.isfinite(v) for v in (target, feed, expected_start)) or feed <= 0:
            raise DualExecutionError("Z目标或速度无效")
        self.last_action = f"Z_TO({target:.2f})"
        current = self._check_z_position(expected_start)
        delta = target - current
        if abs(delta) < 0.001:
            return
        after = self._mpos(self._grbl_command(
            f"$J=G91 G21 Z{delta:.3f} F{feed:.1f}",
            max(15.0, abs(delta) / feed * 60.0 + 10.0), {2: target}))
        if abs(after[2] - target) > 0.1:
            raise DualExecutionError("Grbl报告的Z终点与目标高度不符")

    def _move_z_contact(self, start: float, target: float, contact: float,
                        *, departing: bool = False) -> None:
        """Slow only the configured contact layer; zero margin retains one jog."""
        cfg = self.config["z"]
        feed = float(cfg.get("feed_mm_min", 60.0))
        slow = float(cfg.get("contact_feed_mm_min", feed))
        margin = float(cfg.get("contact_margin_mm", 0.0))
        if margin > 0:
            boundary = contact - margin
            steps = [(boundary, slow if departing else feed),
                     (target, feed if departing else slow)]
        else:
            steps = [(target, feed)]
        for endpoint, speed in steps:
            self.move_z_to(endpoint, speed, expected_start=start)
            start = endpoint
        self._settle(float(cfg.get("settle_s", 0.0)))

    def stm_command(self, command: str) -> str:
        self.last_action = command
        return self._stm_request(command, f"ACK,{command}")

    def rotate(self, angle: float) -> None:
        if not math.isfinite(angle) or abs(angle) > 180.0:
            raise DualExecutionError("R轴单次角度必须在±180°内")
        _, stm = self._require_open()
        angle = round(angle, 2)
        delta_x100 = int(round(angle * 100))
        target = self._r_x100 + delta_x100
        if abs(target) > 18000:
            raise DualExecutionError("R轴累计位置超出±180°")
        self.last_action = f"R{angle:+.2f}"
        stm.write(f"ROTATE,{angle:.2f}")
        deadline = self.clock() + 15.0
        last_heartbeat = self.clock()
        response = ""
        while self.clock() < deadline:
            now = self.clock()
            if self.session_guard:
                self.session_guard()
            # AUX is a leased output.  Keep the relay lease alive while the
            # stepper is rotating; the old blocking read could let it expire.
            if now - last_heartbeat >= 0.8:
                stm.write("IOSTATUS")
                grbl_status = self.grbl_status()
                if grbl_status.get("state") != "Idle":
                    raise DualExecutionError("R旋转期间XYZ状态异常")
                last_heartbeat = now
            try:
                line = stm.read(0.2)
            except SerialReplyTimeout:
                continue
            if not line:
                continue
            self._check_reset(line)
            if line.startswith("ACK,IOSTATUS,"):
                self._check_io(line)
                continue
            if line.startswith("ACK,ROTATE,"):
                if line != f"ACK,ROTATE,{delta_x100}":
                    raise DualExecutionError(f"R轴角度应答不符：{line}")
                response = line
                break
            if line.startswith("ERR,"):
                raise DualExecutionError(f"R轴旋转被拒绝：{line}")
            raise DualExecutionError(f"R轴旋转应答错误：{line}")
        if not response:
            raise DualExecutionError("R轴旋转完成应答超时")
        status = self.stm_status()
        if (status["motion"] != "IDLE" or status["mode"] != "REAL"
                or status["origin"] != "ZEROED" or status["r_x100"] != target):
            raise DualExecutionError("R轴完成状态或累计角度不符")
        self._r_x100 = target

    def _settle(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise DualExecutionError("等待时间无效")
        # Split waits so that AUX's lease remains alive.
        while seconds > 0:
            part = min(seconds, 0.2)
            self.sleeper(part)
            seconds -= part
            self._heartbeat()

    def stop_both(self, reason: str = "") -> None:
        if not self.locked:
            self.failure_action = self.last_action
        self.locked = True
        self.origin_valid = False
        self.approved_plan_id = None
        try:
            if self.grbl is not None:
                self.grbl.realtime(b"!\x85")
        except Exception:
            pass
        try:
            if self.stm32 is not None:
                self.stm32.write("STOP")
        except Exception:
            pass
        self.last_action = f"STOP:{reason}" if reason else "STOP"

    def verify_stopped(self) -> bool:
        """Check each controller independently, even when the other is gone."""
        details: dict[str, Any] = {"stm_stop_ack": False, "outputs_off": False,
                                  "xyz_stopped": False}
        try:
            deadline = self.clock() + 1.0
            while self.clock() < deadline:
                line = self.stm32.read(max(.01, deadline-self.clock()))
                if line == "ACK,STOP":
                    details["stm_stop_ack"] = True
                    break
        except Exception as exc:
            details["stm_ack_error"] = str(exc)
        try:
            io = self.stm_iostatus()
            details["io"] = io
            details["outputs_off"] = io.get("magnet") == "0" and io.get("aux") == "0"
        except Exception as exc:
            details["stm_io_error"] = str(exc)
        try:
            status = self.grbl_status()
            details["grbl"] = status
            details["xyz_stopped"] = status.get("raw_state") in {"Idle", "Hold:0"}
        except Exception as exc:
            details["grbl_error"] = str(exc)
        self.stop_details = details
        self.stop_confirmed = all(details[key] for key in
                                  ("stm_stop_ack", "outputs_off", "xyz_stopped"))
        return self.stop_confirmed

    def stop_summary(self) -> str:
        if self.stop_confirmed:
            return "停止已确认"
        outputs = "电磁铁及AUX已确认关闭" if self.stop_details.get("outputs_off") else "电磁铁/AUX关闭未确认"
        xyz = "XYZ停止已确认" if self.stop_details.get("xyz_stopped") else "XYZ停止未确认"
        return outputs + "；" + xyz + "，请现场检查"

    def execute(self, plan: dict[str, Any]) -> None:
        validate_real_config(self.config, empty_run=self.empty_run)
        if self.locked:
            raise DualExecutionError("执行器已锁定，请重连")
        if self.config.get("require_approved_session", False):
            if (not self.origin_valid or not self.approved_plan_id
                    or plan.get("plan_id") != self.approved_plan_id
                    or plan.get("approval_state") != "approved"):
                raise DualExecutionError("缺少本次连接的基准与方案批准")
            self.approved_plan_id = None  # Consume before the first command.
        if plan.get("ready_for_motion") is not True:
            reasons = plan.get("failure_reasons") or ["ready_for_motion不是true"]
            raise DualExecutionError("视觉安全门拒绝执行：" + "；".join(map(str, reasons)))
        if plan.get("all_matches_ok") is not True or plan.get("all_paths_ok") is not True:
            raise DualExecutionError("拼图匹配或运动路径未通过安全检查")
        calibration = CoordinateCalibration.from_config(self.config)
        pieces = sorted(plan.get("pieces", []), key=lambda item: int(item["move_order"]))
        expected = expected_real_piece_count(self.config)
        if len(pieces) != expected or plan.get("puzzle_profile", "poker") != self.config.get("puzzle_profile", "poker"):
            raise DualExecutionError(f"正式执行必须包含当前模式的{expected}块碎片")
        z_cfg = self.config["z"]
        motion_cfg = self.config.get("motion", {})
        xy_feed = float(motion_cfg.get("xy_feed_mm_min", 300.0))
        safe_z = float(z_cfg["safe"])
        travel_z = float(z_cfg.get("travel", safe_z))
        pickup_z = float(z_cfg["pickup"])
        place_z = float(z_cfg["place"])
        xy_settle = float(motion_cfg.get("xy_settle_s", 0.0))
        rotation_settle = float(self.config.get("rotation", {}).get("settle_s", 0.0))
        routes = []
        for piece in pieces:
            source = calibration.map_point(piece["source_pick_mm"])
            target = calibration.map_point(piece["target_pick_mm"])
            angle = float(piece["rotation_deg_signed"])
            route = [calibration.map_point(p) for p in piece.get("path_mm", [])]
            if (not route or math.dist(route[0], source) > 0.1
                    or math.dist(route[-1], target) > 0.1):
                raise DualExecutionError("路径端点与抓取/放置点不一致")
            if not all(math.isfinite(v) for point in route for v in point):
                raise DualExecutionError("路径包含无效坐标")
            if not math.isfinite(angle) or abs(angle) > 180:
                raise DualExecutionError("方案旋转角度超出±180°")
            if str(motion_cfg.get("origin_mode", "")) == "manual" and any(
                    min(point) < -0.001 for point in route):
                raise DualExecutionError("方案越过人工左上基准")
            bounds = self.config.get("accepted_machine_bounds_mm")
            if (not self.empty_run and self.config.get("require_approved_session")
                    and self.config.get("require_empty_run", True) is not False):
                if not bounds or any(not (bounds[0][i] - 0.01 <= p[i] <= bounds[1][i] + 0.01)
                                     for p in [source, *route, (0, 0)] for i in (0, 1)):
                    raise DualExecutionError("路径超出已通过空载验收的范围")
            routes.append((source, route, angle, route[-1]))
        self.preflight()
        previous_handlers = {}
        def interrupted(signum, _frame):
            self.stop_both(f"signal {signum}")
            raise DualExecutionError("执行被停止")
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, interrupted)
        try:
            self._stm_request("MOTION,1", "ACK,MOTION,1")
            enabled_status = self.stm_status()
            if enabled_status.get("mode") != "REAL":
                raise DualExecutionError("STM32发送MOTION,1后仍不是REAL")
            self._stm_request("ZERO", "ACK,ZERO")
            self._r_x100 = 0
            for source, route, angle, final_xy in routes:
                if not self.empty_run:
                    self._check_z_position(safe_z)
                self.move_xy(*source, xy_feed)
                if self.empty_run:
                    self._heartbeat(force=True)
                    self.rotate(angle)
                    for waypoint in route[1:]:
                        self._xy_walk_segment(*waypoint, xy_feed)
                    self.rotate(-angle)
                    continue
                self._settle(xy_settle)
                self._move_z_contact(safe_z, pickup_z, pickup_z)
                self._stm_request("AUX,1", "ACK,AUX,1")
                self._stm_request("MAGNET,1", "ACK,MAGNET,1")
                self._magnet_expected = True
                self._heartbeat(force=True)
                self._settle(float(motion_cfg.get("pickup_settle_s", 0.8)))
                self._move_z_contact(pickup_z, travel_z, pickup_z, departing=True)
                self._check_z_position(travel_z)
                self.rotate(angle)
                self._settle(rotation_settle)
                if self._xy_shortcut():
                    self._check_z_position(travel_z)
                    self.move_xy_direct(final_xy[0], final_xy[1], xy_feed)
                else:
                    for waypoint in route[1:]:
                        self._check_z_position(travel_z)
                        self._xy_walk_segment(*waypoint, xy_feed)
                self._settle(xy_settle)
                self._move_z_contact(travel_z, place_z, place_z)
                self._stm_request("MAGNET,0", "ACK,MAGNET,0")
                self._magnet_expected = False
                if self.stm_iostatus().get("magnet") != "0":
                    raise DualExecutionError("释放后电磁铁输出未关闭")
                self._settle(float(motion_cfg.get("release_settle_s", 0.3)))
                self._move_z_contact(place_z, safe_z, place_z, departing=True)
                self._check_z_position(safe_z)
                self.rotate(-angle)
                self._settle(rotation_settle)
            # Clear the target region for the final camera verification.
            if not self.empty_run:
                self._check_z_position(safe_z)
            self.move_xy(0.0, 0.0, xy_feed)
            self._stm_request("AUX,0", "ACK,AUX,0")
            self._stm_request("MOTION,0", "ACK,MOTION,0")
            self.preflight()
        except BaseException as exc:
            self.stop_both(str(exc))
            raise
        finally:
            if self._magnet_expected:
                try:
                    self._stm_request("MAGNET,0", "ACK,MAGNET,0")
                except Exception:
                    pass
                self._magnet_expected = False
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
