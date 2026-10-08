"""Two independent serial owners. Importing this module never opens hardware."""
from __future__ import annotations

import json
import errno
import math
import queue
import threading
import time
from pathlib import Path

from serial_connection import open_serial, port_identity, reset_banner, transport_failure


class ProtocolError(RuntimeError):
    pass


class Cancelled(Exception):
    pass


def load_config(path=None):
    cfg = json.loads(Path(path or Path(__file__).with_name("manual_config.json")).read_text(encoding="utf-8-sig"))
    if not cfg["grbl_port"] or not cfg["aux_port"] or Path(cfg["grbl_port"]).resolve() == Path(cfg["aux_port"]).resolve():
        raise ValueError("两路串口必须配置为不同设备")
    if cfg["baudrate"] != 115200 or not 0.1 <= cfg["reply_timeout_s"] <= 1.0:
        raise ValueError("串口配置无效")
    for name in ("grbl_feeds", "z_feeds"):
        if len(cfg[name]) != 3 or any(not math.isfinite(v) or v <= 0 or v > 5000 for v in cfg[name]):
            raise ValueError("点动速度配置无效")
    wait = cfg.get("connect_wait_s", 15.0)
    if not isinstance(wait, (int, float)) or not math.isfinite(wait) or not 0 <= wait <= 60:
        raise ValueError("连接等待时间必须在0到60秒之间")
    return cfg


def connection_error(exc):
    """Classify failures without mislabelling missing hardware as a Python error."""
    code = getattr(exc, "errno", None)
    if code == errno.ENOENT:
        return "未检测到串口：检查电源和USB线后点重连"
    if code in (errno.EAGAIN, errno.EBUSY):
        return "串口被占用：退出其他控制程序后点重连"
    if code in (errno.EACCES, errno.EPERM):
        return "串口权限不足：请运行启动脚本的--check-env"
    if isinstance(exc, TimeoutError):
        return "串口应答超时：检查电源/接线后点重连"
    return str(exc)


def jog_command(axis, distance, feed):
    if axis not in "XYZ" or len(axis) != 1 or not math.isfinite(distance) or abs(distance) not in (0.1, 1, 10):
        raise ValueError("无效点动步长")
    if not math.isfinite(feed) or not 0 < feed <= 5000:
        raise ValueError("无效进给速度")
    return f"$J=G91 G21 {axis}{distance:g} F{feed:g}"


ROTATION_STEPS = (1, 5, 10, 45, 90)


def rotate_command(angle):
    if type(angle) not in (int, float) or not math.isfinite(angle) or abs(angle) not in ROTATION_STEPS:
        raise ValueError("无效旋转角度")
    return f"ROTATE,{angle:.2f}"


def parse_grbl(line):
    if not line.startswith("<") or not line.endswith(">"):
        raise ProtocolError("Grbl 状态格式错误")
    fields = line[1:-1].split("|")
    result = {"machine": fields[0], "pins": ""}
    for field in fields[1:]:
        key, _, value = field.partition(":")
        if key in ("MPos", "WPos", "WCO"):
            values = tuple(float(v) for v in value.split(","))
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ProtocolError("Grbl 坐标格式错误")
            result[key] = values
        elif key == "Pn":
            result["pins"] = value
    # Keep MPos / WPos distinct; never guess work coordinates without WCO.
    if "MPos" in result and "WCO" in result:
        result["WPos"] = tuple(a-b for a, b in zip(result["MPos"], result["WCO"]))
    return result


def parse_aux(line):
    items = line.split(",")
    if len(items) != 8 or items[:3] != ["ACK", "IOSTATUS", "AUX"] or items[4] != "SERVO" or items[6] != "MAGNET":
        raise ProtocolError("STM32 固件不支持 IOSTATUS")
    armed, speed, magnet = int(items[3]), int(items[5]), int(items[7])
    if armed not in (0, 1) or magnet not in (0, 1) or not -100 <= speed <= 100:
        raise ProtocolError("STM32 状态数值错误")
    return {"armed": bool(armed), "speed": speed, "magnet": bool(magnet)}


def parse_motion(line):
    items = line.split(",")
    if (len(items) != 9 or items[:2] != ["ACK", "STATUS"] or
            items[2] not in ("SIM", "REAL") or items[3] not in ("IDLE", "BUSY") or
            items[4] not in ("ZEROED", "UNHOMED")):
        raise ProtocolError("STM32 步进状态格式错误")
    try:
        coords = tuple(int(v) for v in items[5:])
    except ValueError as exc:
        raise ProtocolError("STM32 步进坐标格式错误") from exc
    if any(not -2147483648 <= v <= 2147483647 for v in coords):
        raise ProtocolError("STM32 步进坐标越界")
    return {"motion_enabled": items[2] == "REAL", "motion_busy": items[3] == "BUSY",
            "zeroed": items[4] == "ZEROED", "r_x100": coords[3]}


class SerialWire:
    """Incremental framing; ACKs may arrive split across reads, with status interleaved."""
    def __init__(self, port, baud, cancel, factory=None):
        self.port = open_serial(port, baud, 0.02, 0.1, factory)
        self.logical_port = port
        self.opened_identity = port_identity(port)
        self.cancel = cancel
        self.buffer = bytearray()

    def write(self, data):
        try:
            written = self.port.write(data)
        except OSError as exc:
            raise ProtocolError(transport_failure(self.logical_port, "写入", exc,
                                                  self.opened_identity)) from exc
        if written != len(data):
            raise ProtocolError("串口写入不完整；本次连接失效，未自动重发指令")

    def wait(self, match, timeout, observe=lambda line: None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.cancel():
                raise Cancelled()
            while b"\n" in self.buffer:
                data, _, rest = self.buffer.partition(b"\n")
                self.buffer = bytearray(rest)
                line = data.decode("ascii", errors="strict").strip()
                if not line:
                    continue
                # Startup chatter is drained before handshake. A new banner after
                # that means the controller reset and the session must stop.
                if reset_banner(line):
                    raise ProtocolError("检测到控制器重启，本次基准失效；请重新连接并确认原点：" + line)
                if line.startswith(("ERR,", "error:", "ALARM:")):
                    raise ProtocolError(line)
                observe(line)
                if match(line):
                    return line
            try:
                self.buffer.extend(self.port.read(256))
            except OSError as exc:
                raise ProtocolError(transport_failure(self.logical_port, "读取", exc,
                                                      self.opened_identity)) from exc
            if len(self.buffer) > 8192:
                raise ProtocolError("串口响应溢出")
        raise TimeoutError("串口应答超时")

    def request(self, command, match, timeout, observe=lambda line: None):
        self.write(command.encode("ascii") + b"\n")
        return self.wait(match, timeout, observe)

    def close(self):
        self.port.close()


class DeviceWorker(threading.Thread):
    def __init__(self, owner, kind, factory=None):
        super().__init__(name=f"manual-{kind}", daemon=True)
        self.owner, self.kind, self.factory = owner, kind, factory
        self.commands = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.quit = threading.Event()
        self.wire = None
        self.identified = False
        self.ready = False
        self.outputs_stopped = False
        self.timeout = owner.config["reply_timeout_s"]
        self.rotate_ack = None
        self.rotate_done = False

    def cancelled(self):
        return self.stopping.is_set() or self.quit.is_set() or time.monotonic()-self.owner.ui_tick > 1.2

    def observe(self, line):
        if line.startswith("<"):
            data = parse_grbl(line)
            self.owner.update(self.kind, **data)
        elif self.kind == "aux" and line.startswith("ACK,ROTATE,"):
            if self.rotate_ack is None or line != self.rotate_ack:
                raise ProtocolError("R 轴完成应答不匹配")
            self.rotate_done = True

    def query(self):
        if self.kind == "grbl":
            self.wire.write(b"?")
            line = self.wire.wait(lambda s: s.startswith("<"), self.timeout, self.observe)
            return parse_grbl(line)
        line = self.wire.request("IOSTATUS", lambda s: s.startswith("ACK,IOSTATUS,"), self.timeout, self.observe)
        status = parse_aux(line)
        line = self.wire.request("STATUS", lambda s: s.startswith("ACK,STATUS,"), self.timeout, self.observe)
        status.update(parse_motion(line))
        return status

    def rotate(self, command):
        angle_x100 = round(float(command.split(",")[1]) * 100)
        status = self.query()
        if not status["motion_enabled"] or not status["zeroed"] or status["motion_busy"]:
            raise ProtocolError("请先使能 R 轴，且等待空闲")
        target = status["r_x100"] + angle_x100
        if not -18000 <= target <= 18000:
            raise ProtocolError("R 轴目标超出 ±180°；请反转或确认后置零")
        self.rotate_ack = f"ACK,ROTATE,{angle_x100}"
        self.rotate_done = False
        try:
            # ROTATE replies on completion, with integer x100 degrees. Never retry it.
            self.wire.write(command.encode("ascii") + b"\n")
            deadline = time.monotonic() + 15.0
            while True:
                # Keep the relay lease alive while consuming an interleaved ROTATE ACK.
                status = self.query()
                self.owner.update(self.kind, **status)
                if not status["motion_enabled"] or not status["zeroed"]:
                    raise ProtocolError("R 轴运动状态丢失")
                if self.rotate_done and not status["motion_busy"]:
                    if status["r_x100"] != target:
                        raise ProtocolError("R 轴完成坐标不匹配")
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError("R 轴运动完成超时")
                time.sleep(0.1)
        finally:
            self.rotate_ack = None

    def execute_aux(self, command):
        if command.startswith("ROTATE,"):
            self.rotate(command)
            return
        commands = ("MOTION,1", "ZERO") if command == "MOTION,1" else (command,)
        for line in commands:
            self.wire.request(line, lambda s: s == "ACK," + line, self.timeout, self.observe)

    def shutdown_outputs(self):
        # Never send STM32 output commands to an unidentified serial device.
        if self.wire is not None and self.identified and not self.outputs_stopped:
            self.outputs_stopped = True
            data = b"!" if self.kind == "grbl" else b"STOP\n"
            try:
                self.wire.write(data)
                self.owner.update(self.kind, message="停止已发送，等待状态确认")
                # Observe actual state even though the normal request was cancelled.
                self.wire.cancel = lambda: False
                self.wire.buffer.clear()
                if self.kind == "grbl":
                    self.wire.write(b"?")
                    line = self.wire.wait(lambda s: s.startswith("<"), 0.4)
                    status = parse_grbl(line)
                    self.owner.update(self.kind, **status)
                    confirmed = status["machine"] == "Idle" or status["machine"] == "Hold:0"
                else:
                    line = self.wire.request("IOSTATUS", lambda s: s.startswith("ACK,IOSTATUS,"), 0.4)
                    status = parse_aux(line)
                    line = self.wire.request("STATUS", lambda s: s.startswith("ACK,STATUS,"), 0.4)
                    status.update(parse_motion(line))
                    self.owner.update(self.kind, **status)
                    confirmed = (not status["armed"] and not status["magnet"] and status["speed"] == 0
                                 and not status["motion_enabled"] and not status["motion_busy"])
                self.owner.update(self.kind, message="停止已确认；请重连" if confirmed else "停止尚未完成；检查设备")
            except Exception:
                self.owner.update(self.kind, message="停止未确认，检查设备/电源")

    def run(self):
        try:
            self.open_waiting_for_device()
            # Consume startup data through the same reset detector. Never
            # silently accept a reset's MPos=0 as the physical reference.
            try:
                self.wire.wait(lambda _line: False, 2.0 if self.kind == "grbl" else 0.15)
            except TimeoutError:
                pass  # A quiet bounded drain is expected; reset/I/O faults are fatal.
            if self.kind == "grbl":
                # This Grbl_ESP32 firmware defers normal line commands while
                # feed-held. Real-time status remains available; never resume
                # motion just to complete an identity query on reconnection.
                status = self.query()
                if not any(key in status for key in ("MPos", "WPos")):
                    raise ProtocolError("未识别到 Grbl 坐标状态")
                # The real-time status is the connection proof. `$I` is only
                # an optional informational query: Grbl_ESP32 may defer it
                # after USB reset or while Hold, so it must never make an
                # otherwise responsive controller appear offline.
            else:
                self.wire.request("PING", lambda s: s == "ACK,PING", self.timeout)
                self.query()
            self.identified = True
            if self.kind == "aux":
                self.wire.request("STOP", lambda s: s == "ACK,STOP", self.timeout)
            self.owner.update(self.kind, connected=True, message="已连接", **self.query())
            self.ready = True
            next_poll = 0.0
            while not self.quit.is_set():
                if self.cancelled():
                    raise Cancelled()
                try:
                    command = self.commands.get(timeout=0.02)
                except queue.Empty:
                    command = None
                if command:
                    # Stop can race a queued click; check again before any output.
                    if self.cancelled() or self.owner.fault:
                        raise Cancelled()
                    if self.kind == "grbl":
                        status = self.query()
                        if command.startswith("$J=") and status["machine"] != "Idle":
                            raise ProtocolError("写字机未处于 Idle")
                        if command == "~":
                            self.wire.write(b"~")
                        else:
                            self.wire.request(command, lambda s: s == "ok", self.timeout, self.observe)
                            # ok is acceptance, not physical completion. Do not queue more jogs.
                            end = time.monotonic()+30
                            while True:
                                status = self.query()
                                if status["machine"] == "Idle":
                                    break
                                if status["machine"].startswith(("Alarm", "Hold", "Door")):
                                    raise ProtocolError("点动中断: "+status["machine"])
                                if time.monotonic() > end:
                                    raise TimeoutError("点动完成超时")
                                time.sleep(0.03)
                    else:
                        self.execute_aux(command)
                    self.owner.update(self.kind, pending=False, message="已确认", **self.query())
                if time.monotonic() >= next_poll:
                    self.owner.update(self.kind, **self.query())
                    next_poll = time.monotonic()+0.25
        except Cancelled:
            self.shutdown_outputs()
            if not self.quit.is_set() and not self.stopping.is_set():
                self.owner.failed(self.kind, "触屏控制循环超时")
        except Exception as exc:
            self.shutdown_outputs()
            print("manual connection detail:", self.kind, repr(exc), flush=True)
            self.owner.failed(self.kind, connection_error(exc))
        finally:
            self.ready = False
            if self.wire:
                # Also cover an orderly page exit between loop iterations.
                self.shutdown_outputs()
                self.wire.close()
            self.owner.update(self.kind, connected=False, pending=False)

    def open_waiting_for_device(self):
        # Retry only a missing node before identification. Never replay a command,
        # resume a held machine, or reconnect automatically after losing position.
        deadline = time.monotonic() + self.owner.config.get("connect_wait_s", 15.0)
        while True:
            if self.cancelled():
                raise Cancelled()
            try:
                self.wire = SerialWire(self.owner.config[self.kind+"_port"],
                                       self.owner.config["baudrate"], self.cancelled, self.factory)
                return
            except OSError as exc:
                if exc.errno != errno.ENOENT or time.monotonic() >= deadline:
                    raise
                remaining = max(1, math.ceil(deadline - time.monotonic()))
                self.owner.update(self.kind, message=f"等待设备上电/USB识别 {remaining}秒")
                if self.quit.wait(0.2):
                    raise Cancelled()


class ManualController:
    def __init__(self, config=None, factory=None):
        self.config = config or load_config()
        self.lock = threading.RLock()
        self.ui_tick = time.monotonic()
        self.fault = False
        self.state = {k: {"connected": False, "pending": False, "message": "连接中", "machine": "未知", "pins": "", "armed": False, "speed": 0, "magnet": False,
                          "motion_enabled": False, "motion_busy": False, "zeroed": False, "r_x100": 0} for k in ("grbl", "aux")}
        self.workers = {k: DeviceWorker(self, k, factory) for k in self.state}
        for worker in self.workers.values():
            worker.start()

    def update(self, kind, **values):
        with self.lock:
            self.state[kind].update(values)

    def snapshot(self):
        with self.lock:
            return {key: dict(value) for key, value in self.state.items()}

    def tick(self):
        self.ui_tick = time.monotonic()

    def failed(self, kind, message):
        print("manual connection failed:", kind, message, flush=True)
        self.update(kind, connected=False, message=message)
        # A port absent on entry must not prevent testing the other independent device.
        # Failure after identification, however, stops both outputs and latches a fault.
        if self.workers[kind].identified:
            self.stop_all()

    def submit(self, kind, command):
        with self.lock:
            state = self.state[kind]
            if self.fault or not self.workers[kind].ready or not state["connected"] or state["pending"]:
                raise ProtocolError("设备未就绪；停止后请重新连接")
            self.workers[kind].commands.put_nowait(command)
            state["pending"] = True

    def jog(self, axis, distance, speed_index):
        state = self.snapshot()["grbl"]
        if state["machine"] != "Idle":
            raise ProtocolError("写字机未处于 Idle")
        if speed_index not in (0, 1, 2):
            raise ValueError("无效速度档位")
        feed = self.config["z_feeds" if axis == "Z" else "grbl_feeds"][speed_index]
        self.submit("grbl", jog_command(axis, distance, feed))

    def aux(self, command):
        if command not in ("AUX,1", "AUX,0", "MAGNET,1", "MAGNET,0"):
            raise ValueError("无效辅助命令")
        if command == "MAGNET,1" and not self.snapshot()["aux"]["armed"]:
            raise ProtocolError("请先使能辅助输出")
        self.submit("aux", command)

    def rotation_enable(self):
        self.submit("aux", "MOTION,1")

    def rotation_zero(self):
        state = self.snapshot()["aux"]
        if not state["motion_enabled"] or state["motion_busy"]:
            raise ProtocolError("请先使能 R 轴，且等待空闲")
        self.submit("aux", "ZERO")

    def rotate(self, angle):
        command = rotate_command(angle)
        state = self.snapshot()["aux"]
        if not state["motion_enabled"] or not state["zeroed"] or state["motion_busy"]:
            raise ProtocolError("请先使能 R 轴，且等待空闲")
        if not -18000 <= state["r_x100"] + round(angle * 100) <= 18000:
            raise ProtocolError("R 轴目标超出 ±180°；请反转或确认后置零")
        self.submit("aux", command)

    def stop_rotation(self):
        # Interrupt a pending ROTATE rather than putting STOP behind its completion.
        # Firmware STOP also disables AUX and releases the magnet.
        self.workers["aux"].stopping.set()

    def stop_all(self):
        self.fault = True
        for worker in self.workers.values():
            worker.stopping.set()

    def close(self):
        self.stop_all()
        for worker in self.workers.values():
            worker.quit.set()
        for worker in self.workers.values():
            worker.join(timeout=1.5)


def probe(config=None):
    """Read-only CLI handshake. Does not arm, pause, jog or switch relay outputs."""
    cfg = config or load_config()
    failures = 0
    for kind in ("grbl", "aux"):
        wire = None
        try:
            wire = SerialWire(cfg[kind+"_port"], cfg["baudrate"], lambda: False)
            deadline = time.monotonic()+2.0
            while time.monotonic() < deadline:
                wire.port.read(256)
            if kind == "grbl":
                lines = []
                wire.write(b"?")
                status = wire.wait(lambda s: s.startswith("<"), 2)
                lines.append(status)
                print(kind, lines)
            else:
                print(kind, wire.request("PING", lambda s: s == "ACK,PING", 1))
                print(kind, wire.request("IOSTATUS", lambda s: s.startswith("ACK,IOSTATUS,"), 1))
                print(kind, wire.request("STATUS", lambda s: s.startswith("ACK,STATUS,"), 1))
        except Exception as exc:
            failures += 1
            print(kind, "FAILED:", connection_error(exc), repr(exc))
        finally:
            if wire:
                wire.close()
    return 1 if failures else 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["probe"])
    parser.add_argument("--config")
    args = parser.parse_args()
    raise SystemExit(probe(load_config(args.config)))
