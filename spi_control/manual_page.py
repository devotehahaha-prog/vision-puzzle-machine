"""Manual 800x480 page using the existing LT758x renderer and GT911 events."""
from __future__ import annotations

import time
import queue
from PIL import Image, ImageDraw
from layout import hit
from manual_control import ManualController, ROTATION_STEPS

BUTTONS = {
    "back": (16, 8, 110, 54), "reconnect": (138, 8, 110, 54),
    "stop_all": (604, 8, 180, 54),
    "X-": (16, 168, 145, 46), "X+": (171, 168, 145, 46),
    "Y-": (16, 222, 145, 46), "Y+": (171, 222, 145, 46),
    "Z+": (16, 276, 145, 46), "Z-": (171, 276, 145, 46),
    "step": (16, 340, 145, 44), "feed": (171, 340, 145, 44),
    "resume": (16, 392, 300, 38),
    "angle": (332, 176, 244, 44),
    "reverse": (332, 228, 116, 46), "forward": (460, 228, 116, 46),
    "rotation_enable": (332, 284, 244, 44), "rotation_zero": (332, 340, 244, 44),
    "rotation_stop": (332, 392, 244, 38),
    "enable": (592, 180, 192, 44), "magnet_on": (592, 234, 192, 44),
    "magnet_off": (592, 288, 192, 44), "disable": (592, 342, 192, 44),
}
STEPS = (0.1, 1, 10)


def _rgb(value, order="RGB"):
    """Convert driver colour bytes (which may be BGR) to PIL RGB."""
    if isinstance(value, (bytes, bytearray)):
        channels = dict(zip(order, value))
        return tuple(channels.get(channel, 0) for channel in "RGB")
    return tuple(value)


class _BufferedDraw:
    """Collect a manual page in RAM, then upload it as one display frame."""
    def __init__(self, hardware):
        self.hardware = hardware
        try:
            from lt758x import config as driver_config
            self.order = driver_config.COLOR_ORDER
        except ImportError:
            self.order = "RGB"
        self.image = Image.new("RGB", (800, 480), (0, 0, 0))
        self.canvas = ImageDraw.Draw(self.image)

    def clear(self, color):
        self.canvas.rectangle((0, 0, 799, 479), fill=_rgb(color, self.order))

    def fill_rect(self, x, y, w, h, color):
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(800, int(x + w)), min(480, int(y + h))
        if x2 > x1 and y2 > y1:
            self.canvas.rectangle((x1, y1, x2 - 1, y2 - 1), fill=_rgb(color, self.order))

    def flush(self):
        self.hardware.image(self.image, 0, 0, 800, 480)


class _BufferedText:
    def __init__(self, hardware_text, draw):
        self.hardware_text = hardware_text
        self.draw = draw

    def measure(self, value, size=24):
        return self.hardware_text.measure(value, size)

    def text(self, x, y, value, size=24, color=(255, 255, 255), bg=None):
        font = self.hardware_text._font(value, size)
        x0, y0, _x1, _y1 = font.getbbox(value)
        self.draw.canvas.text((int(x - x0), int(y - y0)), value, font=font,
                              fill=_rgb(color, self.draw.order))


class ManualPageState:
    def __init__(self):
        self.step_index = 0
        self.feed_index = 0
        self.angle_index = 0
        self.armed_touch = False  # opening touch must be released first
        self.message = "R轴：确认当前位置后使能；停止也会释放电磁铁"

    def on_points(self, points):
        if points is None:
            return None
        if not points:
            self.armed_touch = True
            return None
        x, y = points[0]
        if not self.armed_touch:
            return None
        self.armed_touch = False
        for name, rect in BUTTONS.items():
            if hit(x, y, rect):
                return name
        return None


def paint(draw, text, colors, state, devices, button, buffered=False):
    if buffered:
        frame = _BufferedDraw(draw)
        frame_text = _BufferedText(text, frame)
        paint(frame, frame_text, colors, state, devices, button, buffered=False)
        frame.flush()
        return
    draw.clear(colors["scr"])

    def label(x, y, value, size=18, muted=False):
        text.text(x, y, value, size=size, color=colors["muted" if muted else "text"], bg=colors["scr"])

    grbl, aux = devices["grbl"], devices["aux"]
    xyz_ready = grbl["connected"] and not grbl["pending"] and grbl["machine"] == "Idle"
    aux_ready = aux["connected"] and not aux["pending"]
    rotation_ready = aux_ready and aux["motion_enabled"] and aux["zeroed"] and not aux["motion_busy"]
    labels = {
        "back": "返回", "reconnect": "重连", "stop_all": "全部停止",
        "X-": "X −", "X+": "X +", "Y-": "Y −", "Y+": "Y +", "Z+": "Z 下 +", "Z-": "Z 上 −",
        "step": f"步长 {STEPS[state.step_index]:g}mm", "feed": "速度 "+("低", "中", "高")[state.feed_index],
        "resume": "继续（解除进给暂停）", "reverse": "反转 −", "forward": "正转 +",
        "angle": f"转角步长 {ROTATION_STEPS[state.angle_index]:g}°",
        "rotation_enable": "使能 R 轴并置零", "rotation_zero": "当前位置设为零",
        "rotation_stop": "停止 R 轴 / 释放磁铁",
        "enable": "使能电磁铁", "disable": "禁用电磁铁", "magnet_on": "电磁铁吸合", "magnet_off": "电磁铁释放",
    }
    for name, rect in BUTTONS.items():
        enabled = True
        if name in ("X-", "X+", "Y-", "Y+", "Z-", "Z+"):
            enabled = xyz_ready
        elif name in ("reverse", "forward", "rotation_zero"):
            enabled = rotation_ready
        elif name == "rotation_enable":
            enabled = aux_ready and not aux["motion_enabled"] and not aux["motion_busy"]
        elif name == "rotation_stop":
            enabled = aux["connected"]  # must remain available during a pending move
        elif name == "magnet_on":
            enabled = aux_ready and aux["armed"]
        elif name in ("enable", "disable", "magnet_off"):
            enabled = aux_ready
        elif name == "resume":
            enabled = grbl["connected"] and grbl["machine"].startswith("Hold")
        fill = colors["primary"] if enabled else colors["disabled"]
        if name == "stop_all":
            fill = colors["danger"]
        button(draw, text, rect, fill, labels[name], colors, size=20, radius=10)
    label(274, 24, "设备手动控制", 24)
    label(16, 80, "XYZ · USB 写字机", 22)
    label(332, 80, "42 步进电机 · R", 22)
    label(592, 80, "电磁铁", 22)
    label(16, 111, ("在线 " if grbl["connected"] else "离线 ")+grbl["machine"], 18)
    position = grbl.get("MPos", grbl.get("WPos"))
    pos_kind = "M" if "MPos" in grbl else "W"
    label(16, 139, (pos_kind+" "+" / ".join(f"{v:.1f}" for v in position)) if position else "坐标：未知", 17, True)
    r_state = "运动中" if aux["motion_busy"] else ("已使能" if aux["motion_enabled"] else "未使能")
    label(332, 112, r_state if aux["connected"] else "STM32 离线 / 输出未知", 17)
    position = f"R指令 {aux['r_x100']/100:+.2f}°" if aux["connected"] and aux["zeroed"] else "R零点未设置"
    label(332, 143, position + " · 16细分", 17, True)
    label(592, 114, "AUX："+("已使能" if aux["armed"] else "未使能") if aux["connected"] else "AUX：未知", 18)
    label(592, 146, "继电器："+("吸合" if aux["magnet"] else "释放") if aux["connected"] else "继电器：未知", 18)
    message = state.message
    if not grbl["connected"] and grbl["message"] != "连接中":
        message = "写字机："+grbl["message"]
    elif not aux["connected"] and aux["message"] != "连接中":
        message = "STM32："+aux["message"]
    label(16, 447, message[:42], 18)


def run_manual_page(draw, text, colors, read_points, button, config=None, controller_factory=ManualController):
    state = ManualPageState()
    controller = controller_factory(config)
    last_view = None
    last_devices = None
    last_paint = 0.0
    try:
        while True:
            controller.tick()
            action = state.on_points(read_points())
            try:
                if action == "back":
                    return
                if action == "reconnect":
                    controller.close()
                    controller = controller_factory(config)
                    state.message = "重新连接；R轴与电磁铁默认禁用"
                elif action == "stop_all":
                    controller.stop_all()
                    state.message = "停止已请求；操作前请重连"
                elif action == "step":
                    state.step_index = (state.step_index+1) % len(STEPS)
                elif action == "feed":
                    state.feed_index = (state.feed_index+1) % 3
                elif action == "angle":
                    state.angle_index = (state.angle_index+1) % len(ROTATION_STEPS)
                elif action in ("X-", "X+", "Y-", "Y+", "Z-", "Z+"):
                    controller.jog(action[0], STEPS[state.step_index]*(1 if action[1] == "+" else -1), state.feed_index)
                elif action == "resume":
                    if not controller.snapshot()["grbl"]["machine"].startswith("Hold"):
                        raise ValueError("只有 Hold 状态才可继续")
                    controller.submit("grbl", "~")
                elif action in ("forward", "reverse"):
                    controller.rotate(ROTATION_STEPS[state.angle_index]*(1 if action == "forward" else -1))
                elif action == "rotation_enable":
                    if controller.snapshot()["aux"]["motion_enabled"]:
                        raise ValueError("R轴已使能；需要重新置零请点当前位置设为零")
                    controller.rotation_enable()
                elif action == "rotation_zero":
                    controller.rotation_zero()
                elif action == "rotation_stop":
                    controller.stop_rotation()
                    state.message = "R轴停止并释放电磁铁；再次操作请重连"
                elif action in ("enable", "disable", "magnet_on", "magnet_off"):
                    controller.aux({"enable": "AUX,1", "disable": "AUX,0", "magnet_on": "MAGNET,1", "magnet_off": "MAGNET,0"}[action])
                if action not in (None, "stop_all", "reconnect", "step", "feed", "angle", "rotation_stop"):
                    state.message = "请求已提交；以设备应答为准"
            except (ValueError, RuntimeError, queue.Full) as exc:
                state.message = str(exc)
            devices = controller.snapshot()
            if devices != last_devices:
                print("manual devices:", devices, flush=True)
                last_devices = devices
            view = (repr(devices), state.step_index, state.feed_index, state.angle_index, state.message)
            if view != last_view and time.monotonic()-last_paint > 0.08:
                # A full page consists of many small SPI transactions. Render it
                # in RAM and upload one frame to keep the UI heartbeat gap below
                # its watchdog limit. A stalled transfer still trips that limit.
                started = time.monotonic()
                paint(draw, text, colors, state, devices, button, buffered=True)
                print("manual frame_ms=%.1f" % ((time.monotonic()-started)*1000), flush=True)
                last_view, last_paint = view, time.monotonic()
            time.sleep(0.02)
    finally:
        controller.close()
