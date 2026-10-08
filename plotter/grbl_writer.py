#!/usr/bin/env python3
"""通用写字机的最小 Grbl 控制端。

本模块只依赖 Python 标准库，串口运行时依赖 pyserial。设计重点是把
Grbl 文本协议和路径生成分开，便于在没有接入机器时先做离线验证。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, TextIO, Tuple


Point = Tuple[float, float]


class GrblError(RuntimeError):
    """表示控制器返回了 Grbl error 响应。"""


@dataclass
class GrblResponse:
    """封装一条 Grbl 命令的原始响应和最终结果。"""

    lines: List[str]
    result: str


class GrblClient:
    """实现 Grbl 串口的最小请求响应状态机。

    普通命令采用“发送一行、等待 ok/error、再发送下一行”的方式，避免
    依赖固定延时。实时命令（?、!、~、Ctrl-X）直接发送单字节。
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.2,
        log: Optional[TextIO] = None,
    ) -> None:
        """保存串口参数；真正打开串口延迟到 enter，方便命令行错误处理。"""
        self.port_name = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.log = log
        self.port = None

    def __enter__(self) -> "GrblClient":
        """打开 8N1 串口并等待控制器复位信息稳定。"""
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError("缺少 pyserial，请先运行：py -m pip install pyserial") from exc

        self.port = serial.Serial(
            port=self.port_name,
            baudrate=self.baudrate,
            bytesize=8,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            write_timeout=2.0,
        )
        time.sleep(2.0)
        self._read_available()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """关闭串口；异常退出时不再发送额外运动命令。"""
        if self.port is not None:
            self.port.close()
            self.port = None

    def _require_port(self):
        """确保调用发生在串口已打开的上下文中。"""
        if self.port is None:
            raise RuntimeError("串口尚未打开")
        return self.port

    def _write(self, payload: bytes) -> None:
        """写入原始字节并记录十六进制，便于后续核对真实协议。"""
        port = self._require_port()
        if self.log:
            self.log.write(f"TX {payload.hex(' ')}\n")
            self.log.flush()
        port.write(payload)
        port.flush()

    def _read_available(self) -> List[str]:
        """读取当前缓冲区内容，不等待新的完整命令。"""
        port = self._require_port()
        data = bytearray()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            chunk = port.read(port.in_waiting or 1)
            if chunk:
                data.extend(chunk)
            elif data:
                break
        if not data:
            return []
        text = data.decode("ascii", errors="replace")
        lines = [line for line in text.replace("\r\n", "\n").split("\n") if line]
        if self.log:
            self.log.write(f"RX {data.hex(' ')}\n")
            self.log.flush()
        return lines

    def send_line(self, command: str, timeout: float = 5.0) -> GrblResponse:
        """发送一条 ASCII 命令并等待 ok 或 error:n。"""
        command = command.strip()
        if not command:
            return GrblResponse([], "empty")

        self._write(command.encode("ascii") + b"\n")
        lines: List[str] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for line in self._read_available():
                lines.append(line)
                lowered = line.lower()
                if lowered == "ok":
                    return GrblResponse(lines, "ok")
                if lowered.startswith("error:"):
                    return GrblResponse(lines, line)
        raise TimeoutError(f"等待 Grbl 响应超时：{command!r}，已收到：{lines!r}")

    def realtime(self, command: bytes) -> List[str]:
        """发送不带换行的实时命令，例如 b'?'、b'!' 或 b'\\x18'。"""
        if len(command) != 1:
            raise ValueError("实时命令必须恰好是一个字节")
        self._write(command)
        time.sleep(0.15)
        return self._read_available()

    def probe(self) -> dict[str, List[str]]:
        """执行只读诊断，不发送任何运动、解锁或回零命令。"""
        result: dict[str, List[str]] = {}
        for command in ("$I", "$G", "$$", "$#"):
            response = self.send_line(command)
            result[command] = response.lines
            if response.result != "ok":
                raise GrblError(f"{command} 返回 {response.result}")
        result["?"] = self.realtime(b"?")
        return result


def _local_name(tag: str) -> str:
    """去掉 SVG XML 命名空间，只保留元素名称。"""
    return tag.rsplit("}", 1)[-1].lower()


def _points(text: str) -> List[Point]:
    """解析 polyline/polygon 的数字坐标序列。"""
    values = [float(value) for value in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)]
    if len(values) % 2:
        raise ValueError(f"SVG 坐标数量不是偶数：{text!r}")
    return list(zip(values[::2], values[1::2]))


def _path_points(text: str) -> List[List[Point]]:
    """解析常见的 SVG M/L/H/V/Z 路径，暂不处理曲线命令。"""
    tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    paths: List[List[Point]] = []
    current: List[Point] = []
    command: Optional[str] = None
    index = 0
    x = y = 0.0
    start: Optional[Point] = None

    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
            if command.upper() == "Z":
                if start is not None and current and current[-1] != start:
                    current.append(start)
                if current:
                    paths.append(current)
                current = []
                start = None
                command = None
                continue
        if command is None:
            raise ValueError(f"SVG path 缺少命令：{text!r}")

        absolute = command.isupper()
        op = command.upper()
        required = 2 if op in {"M", "L"} else 1
        if index + required > len(tokens) or any(tokens[index + offset].isalpha() for offset in range(required)):
            raise ValueError(f"SVG path 参数不足：{text!r}")

        if op in {"M", "L"}:
            nx = float(tokens[index]); ny = float(tokens[index + 1]); index += 2
            if not absolute:
                nx += x; ny += y
            x, y = nx, ny
            if op == "M":
                if current:
                    paths.append(current)
                current = [(x, y)]
                start = (x, y)
                command = "L" if absolute else "l"
            else:
                current.append((x, y))
        elif op in {"H", "V"}:
            value = float(tokens[index]); index += 1
            if op == "H":
                x = value if absolute else x + value
            else:
                y = value if absolute else y + value
            current.append((x, y))
        else:
            raise ValueError(f"暂不支持 SVG 路径命令 {command!r}，请先导出为折线")

    if current:
        paths.append(current)
    return paths


def svg_strokes(path: Path) -> List[List[Point]]:
    """读取 SVG 中的 line、polyline、polygon 和常见 path 元素。"""
    root = ET.parse(path).getroot()
    strokes: List[List[Point]] = []
    for element in root.iter():
        name = _local_name(element.tag)
        if name == "line":
            strokes.append([
                (float(element.attrib["x1"]), float(element.attrib["y1"])),
                (float(element.attrib["x2"]), float(element.attrib["y2"])),
            ])
        elif name in {"polyline", "polygon"}:
            stroke = _points(element.attrib.get("points", ""))
            if name == "polygon" and stroke and stroke[0] != stroke[-1]:
                stroke.append(stroke[0])
            if len(stroke) >= 2:
                strokes.append(stroke)
        elif name == "path":
            strokes.extend(stroke for stroke in _path_points(element.attrib.get("d", "")) if len(stroke) >= 2)
    if not strokes:
        raise ValueError("SVG 中没有找到可转换的线段元素")
    return strokes


def strokes_to_gcode(
    strokes: Iterable[Sequence[Point]],
    z_up: float,
    z_down: float,
    feed_rate: float,
    scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    flip_y: bool = False,
) -> str:
    """把二维笔画转换成带抬笔/落笔的 Grbl G-code。"""
    lines = ["G21", "G90", f"G0 Z{z_up:g}"]
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        def transform(point: Point) -> Point:
            x, y = point
            if flip_y:
                y = -y
            return x * scale + offset_x, y * scale + offset_y

        start_x, start_y = transform(stroke[0])
        lines.append(f"G0 X{start_x:.3f} Y{start_y:.3f}")
        lines.append(f"G1 Z{z_down:g} F{feed_rate:g}")
        for point in stroke[1:]:
            x, y = transform(point)
            lines.append(f"G1 X{x:.3f} Y{y:.3f} F{feed_rate:g}")
        lines.append(f"G0 Z{z_up:g}")
    lines.append("M5")
    return "\n".join(lines) + "\n"


def send_gcode(client: Optional[GrblClient], source: Path, execute: bool) -> None:
    """预览或逐行发送 G-code；只有显式 execute 才会驱动机器。"""
    commands = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()]
    commands = [line for line in commands if line and not line.startswith(";")]
    if not execute:
        print("预览模式：不会打开串口，也不会驱动机器。")
        print("\n".join(commands))
        return
    if client is None:
        raise RuntimeError("执行模式缺少串口客户端")

    try:
        for number, command in enumerate(commands, 1):
            response = client.send_line(command)
            if response.result != "ok":
                raise GrblError(f"第 {number} 行失败：{command} -> {response.result}")
            print(f"[{number}/{len(commands)}] {command}")
    except KeyboardInterrupt:
        # Ctrl-C 时让 Grbl 进入复位状态，避免继续排队运动。
        client.realtime(b"\x18")
        raise


@dataclass
class MachineProfile:
    """用户提供的机器配置（来自 .plotter 下的 Grbl.json / settings.json）。

    这些数值是二次开发最关键的实测依据：抬落笔 Z、进给速度、点动速度、
    坐标偏移与轴方向。默认值为历史设备参数，使用前必须按自己的机构核对。
    """

    source: str = "内置默认值"
    pen_type: str = "Stepper"
    z_up: float = 0.0          # Grbl.json: zOff
    z_down: float = -10.0      # Grbl.json: zOn（实机 Z=0 为最高点，向下为负）
    z_speed: float = 10000.0   # Grbl.json: zSpeed
    feed_rate: float = 10000.0  # settings.json: feedRate
    jog_speed: float = 10000.0  # settings.json: jogSpeed
    jog_dist: float = 1.0      # MachineControllerData.json: jogDist
    offset_x: float = 0.0
    offset_y: float = 0.0
    baud: int = 115200
    port: str = "COM11"
    reverse_y: bool = True     # AxesData.json / default.properties: axe_reverse_y

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "pen_type": self.pen_type,
            "z_up": self.z_up,
            "z_down": self.z_down,
            "z_speed": self.z_speed,
            "feed_rate": self.feed_rate,
            "jog_speed": self.jog_speed,
            "jog_dist": self.jog_dist,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "baud": self.baud,
            "port": self.port,
            "reverse_y": self.reverse_y,
        }


def find_settings_dir(explicit: Optional[Path] = None) -> Optional[Path]:
    """定位用户机器配置目录（.plotter/<机器名>）。

    查找顺序：显式参数 > 环境变量 PLOTTER_SETTINGS_DIR > 工作目录 >
    用户主目录 > 本仓库 plotter 目录。
    """
    import os

    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env_dir = os.environ.get("PLOTTER_SETTINGS_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    here = Path(__file__).resolve().parent          # .../plotter
    candidates.append(Path.cwd() / ".plotter")
    candidates.append(here / ".plotter")
    candidates.append(Path.home() / ".plotter")
    candidates.append(Path.cwd())
    for base in candidates:
        if not base.exists():
            continue
        if (base / "Grbl.json").exists():
            return base
        try:
            children = sorted(base.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir() and (child / "Grbl.json").exists():
                return child
    return None


def load_machine_profile(settings_dir: Optional[Path] = None) -> MachineProfile:
    """读取用户提供的机器配置；缺失时返回内置默认值。"""
    import json

    profile = MachineProfile()
    directory = find_settings_dir(settings_dir)
    if directory is None:
        return profile
    profile.source = str(directory)

    grbl_json = directory / "Grbl.json"
    if grbl_json.exists():
        data = json.loads(grbl_json.read_text(encoding="utf-8"))
        profile.pen_type = data.get("type", profile.pen_type)
        # 实机 Z=0 为最高点：抬笔 +0，落笔取负的 zOn。
        profile.z_up = float(data.get("zOff", 0.0))
        profile.z_down = -abs(float(data.get("zOn", abs(profile.z_down))))
        profile.z_speed = float(data.get("zSpeed", profile.z_speed))

    settings_json = directory / "settings.json"
    if settings_json.exists():
        data = json.loads(settings_json.read_text(encoding="utf-8"))
        profile.feed_rate = float(data.get("feedRate", profile.feed_rate))
        profile.jog_speed = float(data.get("jogSpeed", profile.jog_speed))
        profile.offset_x = float(data.get("offsetX", profile.offset_x))
        profile.offset_y = float(data.get("offsetY", profile.offset_y))

    controller_json = directory / "MachineControllerData.json"
    if controller_json.exists():
        data = json.loads(controller_json.read_text(encoding="utf-8"))
        profile.jog_dist = float(data.get("jogDist", profile.jog_dist))
        profile.baud = int(data.get("baud", profile.baud))
        profile.port = str(data.get("port", profile.port))

    axes_json = directory / "AxesData.json"
    if axes_json.exists():
        data = json.loads(axes_json.read_text(encoding="utf-8"))
        profile.reverse_y = bool(data.get("yAxeRev", profile.reverse_y))
    return profile


def build_parser() -> argparse.ArgumentParser:
    """创建命令行解析器。"""
    parser = argparse.ArgumentParser(description="通用写字机 Grbl 最小控制端")
    sub = parser.add_subparsers(dest="action", required=True)

    probe = sub.add_parser("probe", help="只读查询固件、参数、坐标和状态")
    probe.add_argument("port", help="串口，例如 COM11")
    probe.add_argument("--log", type=Path, help="保存原始 TX/RX 十六进制日志")

    machine = sub.add_parser("machine", help="读取用户提供的机器配置（Grbl.json 等）")
    machine.add_argument("--settings-dir", type=Path, default=None,
                         help=".plotter 机器配置目录；默认自动查找")
    machine.add_argument("--json", action="store_true", help="以 JSON 输出")

    convert = sub.add_parser("svg", help="把 SVG 线条转换为 G-code，不连接设备")
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--z-up", type=float, default=None)
    convert.add_argument("--z-down", type=float, default=None)
    convert.add_argument("--feed", type=float, default=None)
    convert.add_argument("--scale", type=float, default=1.0)
    convert.add_argument("--offset-x", type=float, default=None)
    convert.add_argument("--offset-y", type=float, default=None)
    convert.add_argument("--flip-y", action="store_true")
    convert.add_argument("--settings-dir", type=Path, default=None,
                         help="用用户机器配置覆盖默认抬落笔/进给")

    send = sub.add_parser("send", help="发送已有 G-code；必须显式指定 --execute")
    send.add_argument("port")
    send.add_argument("input", type=Path)
    send.add_argument("--execute", action="store_true", help="确认后才真正驱动设备")
    send.add_argument("--log", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """执行命令行入口并把可预期错误转换为中文提示。"""
    args = build_parser().parse_args(argv)

    if args.action == "machine":
        import json as _json

        profile = load_machine_profile(args.settings_dir)
        if profile.source == "内置默认值":
            print("未找到用户配置目录（.plotter/<机器名>），以下为内置默认值。")
        if args.json:
            print(_json.dumps(profile.as_dict(), ensure_ascii=False, indent=2))
            return 0
        print(f"配置文件      : {profile.source}")
        print(f"笔控类型      : {profile.pen_type}")
        print(f"抬笔 Z        : {profile.z_up:g}      (Grbl.json zOff)")
        print(f"落笔 Z        : {profile.z_down:g}     (Grbl.json zOn, 实机向下为负)")
        print(f"抬落笔速度    : {profile.z_speed:g}")
        print(f"绘图进给      : {profile.feed_rate:g} mm/min")
        print(f"点动速度/步距 : {profile.jog_speed:g} mm/min / {profile.jog_dist:g} mm")
        print(f"坐标偏移      : X{profile.offset_x:g} Y{profile.offset_y:g}")
        print(f"Y 轴反向      : {profile.reverse_y}")
        print(f"默认串口      : {profile.port} @ {profile.baud}")
        return 0

    if args.action == "svg":
        profile = load_machine_profile(args.settings_dir)
        z_up = profile.z_up if args.z_up is None else args.z_up
        z_down = profile.z_down if args.z_down is None else args.z_down
        feed = profile.feed_rate if args.feed is None else args.feed
        offset_x = profile.offset_x if args.offset_x is None else args.offset_x
        offset_y = profile.offset_y if args.offset_y is None else args.offset_y
        gcode = strokes_to_gcode(
            svg_strokes(args.input), z_up, z_down, feed,
            args.scale, offset_x, offset_y, args.flip_y,
        )
        args.output.write_text(gcode, encoding="ascii")
        print(f"已生成 {args.output}，共 {len(gcode.splitlines())} 行")
        print(f"抬笔 Z={z_up:g}  落笔 Z={z_down:g}  进给 {feed:g}  （来源：{profile.source}）")
        return 0

    # 预览模式不打开串口，避免用户即使没有接入设备也被端口错误阻断。
    if args.action == "send" and not args.execute:
        send_gcode(None, args.input, False)
        return 0

    log_handle = args.log.open("w", encoding="ascii") if args.log else None
    try:
        with GrblClient(args.port, log=log_handle) as client:
            if args.action == "probe":
                for command, lines in client.probe().items():
                    print(f">>> {command}")
                    print("\n".join(lines))
            else:
                send_gcode(client, args.input, args.execute)
    except (GrblError, OSError, TimeoutError, ValueError, ET.ParseError, RuntimeError) as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if log_handle:
            log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
