#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SPI control page: real puzzle cameras, calibration, thresholds and plan review."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from layout import (
    MODE_BTN,
    MANUAL_BTN,
    BACK_BTN,
    BTN_RADIUS,
    CALIB_BTN,
    CALIB_PREVIEW_BTN,
    CALIB_VIEW_BUTTONS,
    CALIB_STATUS,
    CARD,
    CARD_RADIUS,
    DISABLED,
    GROUP_BOARD,
    GROUP_BOARD_BTN,
    GROUP_PIECE,
    GROUP_PIECE_BTN,
    GREY,
    INDICATOR_H,
    MODE_ORDINARY,
    MODE_POKER,
    MUTED,
    ORDINARY_TAB,
    PAD,
    LIVE_BTN,
    LIVE_PREVIEW,
    LIVE_STATUS,
    PAGE_CALIB,
    PAGE_CALIB_PREVIEW,
    PAGE_EXECUTION,
    PAGE_HOME,
    PAGE_LIVE,
    PAGE_NEXT_BTN,
    PAGE_PREV_BTN,
    PAGE_TUNE,
    POKER_TAB,
    PREVIEW_RECT,
    PRIMARY,
    PRIMARY_MUTED,
    PRIMARY_PRESSED,
    SAVE_BTN,
    SAVE_CAL_BTN,
    SCR,
    SCREEN_W,
    START_BTN,
    STOP_BTN,
    STATUS_BAR,
    STATUS_TEXT,
    SUBTITLE,
    TAB_H,
    TEXT,
    TUNE_BTN,
    TUNE_PREVIEW,
    TUNE_STATUS,
    UNDO_BTN,
    WHITE,
    StartPageState,
    letterbox,
    map_camera_to_preview,
    map_preview_to_camera,
    slider_row,
)

HERE = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def find_projects():
    for parent in [HERE.parent, *HERE.parents]:
        nested = parent / "E题开源" / "完赛" / "puzzle_vision"
        if nested.is_dir():
            return {
                "ordinary": nested,
                "poker": parent / "E题开源" / "完赛" / "puzzle_vision_poker",
                "contest": parent,
            }
        local = parent / "完赛" / "puzzle_vision"
        if local.is_dir():
            return {
                "ordinary": local,
                "poker": parent / "完赛" / "puzzle_vision_poker",
                "contest": parent,
            }
    raise FileNotFoundError("cannot locate puzzle_vision under E题开源")


def find_venv_python(contest: Path) -> Path:
    candidates = [
        contest / ".venv" / "bin" / "python",
        contest / ".venv" / "Scripts" / "python.exe",
        contest / "pybullet" / ".venv" / "bin" / "python",
        contest / "pybullet" / ".venv" / "Scripts" / "python.exe",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("simulation venv python not found")


def find_driver_dir() -> Path:
    env = os.environ.get("SPI_DRIVER_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend((
        HERE.parent / "external" / "spi-display-driver",
    ))
    for path in candidates:
        if (path / "lt758x" / "__init__.py").is_file():
            return path
    raise FileNotFoundError("LT758x driver directory not found")


def sim_command(python: Path):
    return [str(python), "-X", "utf8", "-B", "run_sim.py", "--auto-start"]


def vision_command(python: Path, execute: bool = False, plan_path=None, session_dir=None, plan_id=None):
    """Build the formal camera command; dry-run is the default."""
    command = [str(python), "-X", "utf8", "-B", "run_real.py",
               "--auto", "--start-immediately"]
    if execute:
        if not plan_path or not session_dir or not plan_id:
            raise ValueError("执行必须绑定已预览的方案与会话")
        command = [str(python), "-X", "utf8", "-B", "run_real.py", "--execute-approved",
                   str(plan_path), "--expected-plan-id", str(plan_id),
                   "--session-dir", str(session_dir), "--parent-pid", str(os.getpid())]
    return command


def sim_env():
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    env.setdefault("XAUTHORITY", str(Path.home() / ".Xauthority"))
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["LP_NUM_THREADS"] = "2"
    env.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
    env["PYTHONUNBUFFERED"] = "1"
    return env


class SimulationJob:
    def __init__(self, python, project):
        runtime = HERE / "runtime"
        runtime.mkdir(exist_ok=True)
        self.log = (runtime / "simulation.log").open("w", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(sim_command(python), cwd=str(project), env=sim_env(),
                                         stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close()
            raise

    def poll(self):
        code = self.proc.poll()
        if code is not None:
            self.log.close()
        return code

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()


class VisionJob(SimulationJob):
    """Generate a plan or execute the confirmed plan through run_real.py."""
    def __init__(self, python, project, execute=False, plan_path=None, plan_id=None):
        import uuid
        runtime = HERE / "runtime"
        runtime.mkdir(exist_ok=True)
        self.session_dir = runtime / ("session_" + uuid.uuid4().hex)
        self.last_status = {}
        self.log = (runtime / ("vision_execute.log" if execute else "vision_plan.log")).open(
            "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            vision_command(python, execute, plan_path, self.session_dir, plan_id), cwd=str(project), env=sim_env(),
            stdout=self.log, stderr=subprocess.STDOUT,
        )

    def status(self):
        try:
            return json.loads((self.session_dir / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def confirm(self):
        status = self.status()
        if status.get("stage") not in {"await_origin", "await_empty_accept", "await_physical_accept"}:
            raise RuntimeError("当前阶段不能确认")
        payload = {k: status[k] for k in ("session_id", "plan_id", "sequence")}
        payload["action"] = "confirm"
        temporary = self.session_dir / "command.tmp"
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, self.session_dir / "command.json")


class OrdinaryCameraJob(VisionJob):
    """Four-piece planning or execution of one immutable reviewed plan."""
    def __init__(self, python, project, execute=False, plan_path=None, plan_id=None):
        import uuid
        runtime = HERE / "runtime"
        runtime.mkdir(exist_ok=True)
        self.session_dir = runtime / ("session_" + uuid.uuid4().hex)
        self.last_status = {}
        command = [str(python), "-u", "-X", "utf8", "-B", str(HERE / "ordinary_camera.py"),
                   "--project", str(project)]
        if execute:
            if not plan_path or not plan_id:
                raise ValueError("普通实机执行必须绑定已预览方案")
            command += ["--execute-approved", str(plan_path), "--expected-plan-id", str(plan_id),
                        "--session-dir", str(self.session_dir), "--parent-pid", str(os.getpid())]
        else:
            command += ["--prepare-execution"]
        self.log = (runtime / ("ordinary_execute.log" if execute else "ordinary_camera.log")).open("w", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(
                command, cwd=str(HERE), env=sim_env(),
                stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close()
            raise


def finish_ordinary_plan(state, project, started_at, code):
    from ordinary_camera import output_dir
    state.busy = False
    state.armed = False
    state.plan_ready = False
    state.review_plan_path = None
    state.review_plan_id = None
    state.review_image_path = None
    try:
        plan = json.loads((output_dir(project) / "plan.json").read_text(encoding="utf-8"))
        if (code != 0 or plan.get("mode") != "ordinary_camera" or not plan.get("preview_ready")
                or plan.get("approval_state") not in {"preview_only", "review_required"} or plan.get("ready_for_motion") is not False
                or plan.get("piece_count") != 4 or float(plan.get("generated_at_epoch", 0)) < started_at):
            raise ValueError("本次普通方案尚未通过")
        image = Path(plan["assembly_path"])
        if not image.is_file():
            raise ValueError("普通方案预览图缺失")
        state.review_image_path = image
        state.review_plan_id = plan["plan_id"]
        executable = plan.get("approval_state") == "review_required"
        if executable and (plan.get("schema_version") != 2 or plan.get("puzzle_profile") != "ordinary"
                           or plan.get("geometry_reasons") or not plan.get("content_sha256")
                           or not plan.get("all_matches_ok") or not plan.get("all_paths_ok")):
            raise ValueError("普通方案缺少绑定信息或未通过检查")
        state.plan_ready = executable
        state.review_plan_path = Path(plan["artifact_path"]) if executable else None
        state.workflow_stage = "preview" if executable else "ordinary_preview"
        state.status = "普通拼图：4片方案就绪，待执行确认" if executable else "普通拼图：4片方案已生成，仅预览"
        state.workflow_message = "四片识别与连续帧检查通过"
    except (OSError, ValueError, KeyError, TypeError):
        state.plan_ready = False
        state.workflow_stage = "plan_failed"
        state.workflow_message, state.review_image_path = read_planning_failure(project, started_at, "output_camera")
        state.status = "普通方案未生成，请按提示调整"


def stop_workflow_job(job, executing):
    """Only report hardware STOP as confirmed when the serial owner verified it."""
    if job is None:
        return "本次方案已取消，请重新识别"
    job.close()
    if not executing:
        return "识别已取消，请重新识别"
    status = job.status()
    if status.get("stop_confirmed") is True:
        return "停止已确认，本次方案失效"
    return "停止应答未确认，请现场检查"


_TOUCH_STATE = {"fails": 0, "recoveries": 0,
                "last_recover_at": 0.0, "recover_attempts": 0,
                "last_report_at": 0.0}
_TOUCH_RECOVER_MIN_INTERVAL = 10.0   # 恢复尝试最短间隔（秒）
_TOUCH_REPORT_MIN_INTERVAL = 30.0    # 日志最短间隔（秒）


def read_touch_report_safe(touch):
    """容错的触摸读取。

    GT911 的 I2C 偶发 OSError（Errno 6 / Errno 5），若直接冒泡会让整个
    触屏界面退出。这里把瞬时故障降级为「本帧无触摸」，并在连续失败时尝试
    重新初始化触摸控制器；成功即清零计数。
    """
    global _TOUCH_STATE
    try:
        report = read_touch_report(touch)
    except OSError:
        _TOUCH_STATE["fails"] += 1
        n = _TOUCH_STATE["fails"]
        now = time.monotonic()
        # 日志限流：持续故障时不再刷屏
        if now - _TOUCH_STATE["last_report_at"] >= _TOUCH_REPORT_MIN_INTERVAL:
            _TOUCH_STATE["last_report_at"] = now
            print(f"touch I2C 故障：累计 {n} 次，已恢复 "
                  f"{_TOUCH_STATE['recoveries']} 次；界面继续运行（请检查排线/供电）",
                  flush=True)
        # 恢复尝试限流：每 _TOUCH_RECOVER_MIN_INTERVAL 秒最多一次
        if (n >= 5 and touch is not None
                and now - _TOUCH_STATE["last_recover_at"] >= _TOUCH_RECOVER_MIN_INTERVAL):
            _TOUCH_STATE["last_recover_at"] = now
            _TOUCH_STATE["recover_attempts"] += 1
            try:
                touch.hw_reset()
                touch._product = touch._read_product_id()
                _TOUCH_STATE["recoveries"] += 1
                _TOUCH_STATE["fails"] = 0
                print(f"touch 已恢复（第 {_TOUCH_STATE['recoveries']} 次）",
                      flush=True)
            except Exception as exc:
                # 只在第一次尝试时打印完整栈，之后静默
                if _TOUCH_STATE["recover_attempts"] == 1:
                    print(f"touch 恢复失败（首次）: {exc!r}", flush=True)
                    traceback.print_exc()
        return None
    except Exception as exc:
        _TOUCH_STATE["fails"] += 1
        now = time.monotonic()
        if now - _TOUCH_STATE["last_report_at"] >= _TOUCH_REPORT_MIN_INTERVAL:
            _TOUCH_STATE["last_report_at"] = now
            print(f"touch 读取异常: {exc!r}", flush=True)
            traceback.print_exc()
        return None
    _TOUCH_STATE["fails"] = 0
    return report


def read_touch_report(touch):
    """Preserve the GT911 distinction between no report and release."""
    status = touch._read_reg(0x814E, 1)[0]
    if not status & 0x80:
        return None
    count = status & 0x0F
    if count > 5:
        touch._write_reg(0x814E, [0])
        return None
    raw = touch._read_reg(0x814F, count * 8) if count else b""
    points = []
    for index in range(count):
        item = raw[index * 8:(index + 1) * 8]
        points.append(touch._transform(item[1] | item[2] << 8, item[3] | item[4] << 8))
    touch._write_reg(0x814E, [0])
    return points


def load_driver(driver_dir: Path):
    path = str(driver_dir)
    if path not in sys.path:
        sys.path.insert(1, path)
    from lt758x import GT911, Draw, Panel, Text, rgb
    return Panel, Draw, Text, GT911, rgb


def to_rgb(fn, color):
    return fn(*color)


def fill_round_rect(draw, x, y, w, h, radius, color):
    r = min(max(0, radius), w // 2, h // 2)
    if r == 0:
        draw.fill_rect(x, y, w, h, color)
        return
    draw.fill_rect(x + r, y, w - 2 * r, h, color)
    draw.fill_rect(x, y + r, w, h - 2 * r, color)
    inset = max(1, r // 2)
    draw.fill_rect(x + inset, y + inset, w - 2 * inset, h - 2 * inset, color)


def centered_text(text, x, y, w, h, label, size, color, bg):
    tw, th = text.measure(label, size)
    text.text(x + (w - tw) // 2, y + (h - th) // 2, label, size=size, color=color, bg=bg)


def paint_button(draw, text, rect, fill, label, C, radius=BTN_RADIUS, size=24):
    x, y, w, h = rect
    fill_round_rect(draw, x, y, w, h, radius, fill)
    centered_text(text, x, y, w, h, label, size, C["white"], fill)


def paint_tabs(draw, text, state, C):
    draw.fill_rect(0, 0, SCREEN_W, TAB_H, C["card"])
    text.text(PAD, 22, state.title_text(), size=24, color=C["text"], bg=C["card"])
    for mode, rect in ((MODE_ORDINARY, ORDINARY_TAB), (MODE_POKER, POKER_TAB)):
        x, y, w, h = rect
        selected = state.mode == mode
        fill = C["primary_muted"] if selected else C["card"]
        fg = C["text"] if selected else C["muted"]
        draw.fill_rect(x, y, w, h, fill)
        centered_text(text, x, y, w, h - INDICATOR_H, state.tab_label(mode), 22, fg, fill)
        if selected:
            draw.fill_rect(x + 24, y + h - INDICATOR_H, w - 48, INDICATOR_H, C["primary"])


def paint_home(draw, text, state, C):
    if state.workflow_stage:
        return paint_workflow(draw, text, state, C)
    draw.clear(C["scr"])
    paint_tabs(draw, text, state, C)
    start_fill = C["disabled"] if state.busy else C["primary"]
    start_label = "确认执行" if getattr(state, "plan_ready", False) else "生成拼图方案"
    paint_button(draw, text, START_BTN, start_fill, start_label, C, size=32)
    if state.busy:
        paint_button(draw, text, STOP_BTN, C["danger"], "全部停止", C, radius=12, size=20)
    tool_fill = C["disabled"] if state.busy else C["card"]
    paint_button(draw, text, CALIB_BTN, tool_fill, "角点标定", C, size=22)
    paint_button(draw, text, CALIB_PREVIEW_BTN, tool_fill,
                 "碎片/标定预览" if state.mode == MODE_ORDINARY else "标定预览", C, size=22)
    paint_button(draw, text, TUNE_BTN, tool_fill, "OpenCV 调参", C, size=22)
    paint_button(draw, text, LIVE_BTN, tool_fill, "相机原画面", C, size=22)
    x, y, w, h = SUBTITLE
    draw.fill_rect(x, y, w, h, C["scr"])
    tw, th = text.measure(state.subtitle_text(), 20)
    text.text(x + (w - tw) // 2, y + (h - th) // 2, state.subtitle_text(),
              size=20, color=C["muted"], bg=C["scr"])
    paint_button(draw, text, MANUAL_BTN, tool_fill, "设备手动控制", C, size=28)
    paint_status(draw, text, state, C)


def paint_workflow(draw, text, state, C):
    draw.clear(C["scr"])
    title = ("方案生成失败 · 当前识别图" if state.workflow_stage == "plan_failed" else
             ("普通拼图 · 四片方案 " if state.mode == MODE_ORDINARY else "三片拼图 · 方案 ")
             + str(state.review_plan_id or "")[:10])
    text.text(20, 14, title,
              size=23, color=C["text"], bg=C["scr"])
    paint_button(draw, text, STOP_BTN, C["danger"], "取消 / 停止", C, size=20)
    if state.review_image_path:
        try:
            from PIL import Image
            with Image.open(state.review_image_path) as im:
                image = im.convert("RGB")
                # Show the actual assembled upper half at useful scale.
                if state.workflow_stage != "plan_failed":
                    image = image.crop((0, 0, image.width, int(image.height * .51)))
                lx, ly, lw, lh = letterbox(image.width, image.height, 400, 370)
                draw.image(image, 20 + lx, 75 + ly, lw, lh)
        except Exception as exc:
            text.text(20, 100, "预览读取失败", size=22, color=C["text"], bg=C["scr"])
            print("review image error", exc, flush=True)
            if state.workflow_stage == "preview":
                state.workflow_stage = "failed"
                state.plan_ready = False
                state.workflow_message = "预览显示失败，请重新识别"
    stage = state.workflow_stage
    if stage == "ordinary_preview":
        lines = ["普通拼图 · 4片", "左侧为实拍碎片拼合预览", "连续帧与几何检查通过", "本模式仅预览，不执行运动"]
        label = "方案已生成 · 仅预览"
    elif stage == "preview":
        lines = (["普通拼图 · 四片", "请核对左侧实拍拼合图", "几何与路径检查已通过", "确认后连接设备检查本次基准"]
                 if state.mode == MODE_ORDINARY else
                 ["花纹待人工确认" if state.pattern_review_required else "花纹评分通过",
                 "请核对左侧真实卡片拼合图", "几何与路径检查已通过", "确认后连接设备并检查本次基准"]
                 )
        label = "预览正确，连接设备"
    elif stage == "await_origin":
        lines = ["核对磁铁在人工左上基准", "MPos≈0/0/0，Z安全", "电磁铁关闭；旧任务已取消",
                 "点击后启动" + ("整路径空载" if getattr(state, "empty_run", True) else
                              ("四片自动拼合" if state.mode == MODE_ORDINARY else "三片自动拼合"))]
        label = "确认基准并启动"
    elif stage == "await_empty_accept":
        lines = ["整路径空载已结束", "确认全部点位与旋转可达", "全程没有撞限、卡住或失步", "确认后记录机械验收"]
        label = "空载正常，记录通过"
    elif stage == "await_physical_accept":
        lines = ["复拍位置/角度检查已通过", "确认无掉片、拖纸或重叠", "实物与批准预览一致", "确认后记录本轮通过"]
        label = "实物正常，完成本轮"
    else:
        message = state.workflow_message or state.status
        lines = [message[i:i+16] for i in range(0, min(len(message), 96), 16)]
        label = "处理中" if state.busy else "本轮已结束"
    for i, line in enumerate(lines[:7]):
        text.text(440, 88 + i * 31, line, size=20, color=C["text"], bg=C["scr"])
    enabled = stage in {"preview", "await_origin", "await_empty_accept", "await_physical_accept"}
    paint_button(draw, text, (435, 325, 345, 82), C["primary"] if enabled else C["disabled"], label, C, size=23)
    if not state.busy:
        paint_button(draw, text, (435, 410, 345, 54), C["card"], "返回 / 重新识别", C, size=21)


def status_rect(state):
    if state.page == PAGE_CALIB:
        return CALIB_STATUS
    if state.page == PAGE_TUNE:
        return TUNE_STATUS
    if state.page in (PAGE_LIVE, PAGE_CALIB_PREVIEW):
        return LIVE_STATUS
    return STATUS_BAR


def paint_status(draw, text, state, C):
    x, y, w, h = status_rect(state)
    fill_round_rect(draw, x, y, w, h, CARD_RADIUS, C["card"])
    text.text(x + 12, y + max(6, (h - 22) // 2), state.status, size=22, color=C["text"], bg=C["card"])


def paint_top_bar(draw, text, title, C, save_label=None):
    draw.fill_rect(0, 0, SCREEN_W, 70, C["card"])
    paint_button(draw, text, BACK_BTN, C["grey"], "返回", C, radius=12, size=22)
    if save_label:
        paint_button(draw, text, SAVE_BTN, C["primary"], save_label, C, radius=12, size=22)
    tw, th = text.measure(title, 24)
    text.text((SCREEN_W - tw) // 2, (70 - th) // 2, title, size=24, color=C["text"], bg=C["card"])


def paint_calib_page(draw, text, state, C, preview_image=None, points=None, letterbox_rect=None, cam_size=None):
    draw.clear(C["scr"])
    paint_top_bar(draw, text, "角点标定", C)
    x, y, w, h = PREVIEW_RECT
    draw.fill_rect(x, y, w, h, C["card"])
    if preview_image is not None:
        try:
            draw.image(preview_image, x, y, w, h)
        except Exception:
            centered_text(text, x, y, w, h, "预览失败", 22, C["muted"], C["card"])
    else:
        centered_text(text, x, y, w, h, "等待画面…", 22, C["muted"], C["card"])
    if points and letterbox_rect and cam_size:
        for cx, cy in points:
            mapped = map_camera_to_preview(cx, cy, PREVIEW_RECT, letterbox_rect, cam_size)
            if mapped is None:
                continue
            mx, my = mapped
            draw.fill_rect(mx - 4, my - 4, 8, 8, C["primary"])
    paint_button(draw, text, UNDO_BTN, C["grey"], "撤销", C, size=24)
    save_fill = C["primary"] if state.calib_count >= 4 else C["disabled"]
    paint_button(draw, text, SAVE_CAL_BTN, save_fill, "保存", C, size=24)
    paint_status(draw, text, state, C)


def paint_calibration_preview_page(draw, text, state, C, preview_image=None):
    draw.clear(C["scr"])
    paint_button(draw, text, BACK_BTN, C["card"], "返回", C, size=22)
    for view, rect, label in CALIB_VIEW_BUTTONS:
        paint_button(draw, text, rect, C["primary"] if state.calibration_view == view else C["card"], label, C, size=22)
    paint_live_preview(draw, text, C, preview_image)
    paint_status(draw, text, state, C)


def paint_live_preview(draw, text, C, preview_image=None, letterbox_rect=None):
    x, y, w, h = LIVE_PREVIEW
    if preview_image is not None:
        try:
            if letterbox_rect is None:
                draw.image(preview_image, x, y, w, h)
                return
            lx, ly, lw, lh = map(int, letterbox_rect)
            if lw <= 0 or lh <= 0 or lx < 0 or ly < 0 or lx + lw > w or ly + lh > h:
                raise ValueError("invalid preview letterbox")
            geometry = (x, y, w, h, lx, ly, lw, lh)
            if getattr(draw, "_live_geometry", None) != geometry:
                # Clear only borders when entering or changing aspect ratio.
                for bx, by, bw, bh in ((x,y,lx,h), (x+lx+lw,y,w-lx-lw,h),
                                      (x+lx,y,lw,ly), (x+lx,y+ly+lh,lw,h-ly-lh)):
                    if bw and bh:
                        draw.fill_rect(bx, by, bw, bh, C["scr"])
                draw._live_geometry = geometry
            active = preview_image.crop((lx, ly, lx + lw, ly + lh))
            started = time.monotonic()
            draw.image(active, x + lx, y + ly, lw, lh)
            elapsed = time.monotonic() - started
            count = getattr(draw, "_live_frame_count", 0) + 1
            draw._live_frame_count = count
            if count % 30 == 0:
                print("SPI preview: pixels=%d transfer_ms=%.1f" % (lw*lh, elapsed*1000), flush=True)
            return
        except Exception as error:
            print("preview transfer failed:", error, flush=True)
            return
    draw._live_geometry = None
    draw._live_frame_count = 0
    draw.fill_rect(x, y, w, h, C["card"])
    centered_text(text, x, y, w, h, "等待 USB 相机…", 22, C["muted"], C["card"])


def paint_live_page(draw, text, state, C, preview_image=None):
    draw.clear(C["scr"])
    paint_top_bar(draw, text, "相机原画面", C)
    paint_live_preview(draw, text, C, preview_image)
    paint_status(draw, text, state, C)


def paint_tune_preview(draw, text, C, preview_image=None):
    x, y, w, h = TUNE_PREVIEW
    if preview_image is not None:
        try:
            draw.image(preview_image, x, y, w, h)
            return
        except Exception as error:
            print("preview transfer failed:", error, flush=True)
            return  # Do not blank the previous frame on a transfer error.
    draw.fill_rect(x, y, w, h, C["card"])
    centered_text(text, x, y, w, h, "等待画面…", 20, C["muted"], C["card"])


def paint_tune_page(draw, text, state, C, preview_image=None):
    draw.clear(C["scr"])
    paint_top_bar(draw, text, "OpenCV 调参", C, save_label="保存")
    paint_tune_preview(draw, text, C, preview_image)
    piece_fill = C["primary"] if state.tune_group == GROUP_PIECE else C["grey"]
    board_fill = C["primary"] if state.tune_group == GROUP_BOARD else C["grey"]
    paint_button(draw, text, GROUP_PIECE_BTN, piece_fill, "碎片", C, size=18)
    paint_button(draw, text, GROUP_BOARD_BTN, board_fill, "找A4", C, size=18)
    mode_label = "阈值: " + state.threshold_mode if state.tune_group == GROUP_PIECE else "寻框预览"
    paint_button(draw, text, MODE_BTN, C["primary"] if state.tune_group == GROUP_PIECE else C["grey"],
                 mode_label, C, size=16)
    paint_button(draw, text, PAGE_PREV_BTN, C["grey"], "上一页", C, size=16)
    paint_button(draw, text, PAGE_NEXT_BTN, C["grey"], "下一页", C, size=16)
    for index, (_key, label, lo, hi, value) in enumerate(state.sliders):
        row = slider_row(index)
        lx, ly, lw, lh = row["label"]
        draw.fill_rect(lx, ly, lw, lh, C["scr"])
        text.text(lx, ly, "%s  %d" % (label + ("（未启用）" if (("value_min" in _key and "otsu" not in _key and state.threshold_mode != "fixed") or ("otsu" in _key and state.threshold_mode != "otsu")) else ""), value), size=20, color=C["text"], bg=C["scr"])
        paint_button(draw, text, row["minus"], C["grey"], "-", C, radius=10, size=28)
        tx, ty, tw, th = row["track"]
        draw.fill_rect(tx, ty, tw, th, C["grey"])
        if hi > lo:
            knob_w = 18
            ratio = (value - lo) / float(hi - lo)
            kx = tx + int(ratio * (tw - knob_w))
            fill_round_rect(draw, kx, ty - 6, knob_w, th + 12, 8, C["primary"])
        paint_button(draw, text, row["plus"], C["grey"], "+", C, radius=10, size=28)
    paint_status(draw, text, state, C)


def theme_colors(rgb):
    return {
        "scr": to_rgb(rgb, SCR),
        "card": to_rgb(rgb, CARD),
        "grey": to_rgb(rgb, GREY),
        "text": to_rgb(rgb, TEXT),
        "muted": to_rgb(rgb, MUTED),
        "primary": to_rgb(rgb, PRIMARY),
        "primary_muted": to_rgb(rgb, PRIMARY_MUTED),
        "primary_pressed": to_rgb(rgb, PRIMARY_PRESSED),
        "white": to_rgb(rgb, WHITE),
        "disabled": to_rgb(rgb, DISABLED),
        "danger": rgb(190, 45, 45),
    }


class CameraSession:
    def __init__(self, python: Path, project: Path, mode: str):
        self.python = python
        self.project = project
        self.mode = mode
        self.workdir = HERE / "runtime"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.overlay_path = self.workdir / "overlay.json"
        self.status_path = self.workdir / "status.json"
        self.preview_path = self.workdir / "spi_preview.jpg"
        self.stop_path = self.workdir / "stop"
        self.proc = None
        self.started_at = time.monotonic()
        self.points = []
        self.calibration_view = "raw"
        self.params = {}
        self.group = GROUP_PIECE
        self.vision = "poker" if "poker" in str(project).replace("\\", "/").split("/")[-1] else "ordinary"
        self.letterbox = (0, 0, PREVIEW_RECT[2], PREVIEW_RECT[3])
        self.cam_size = (1280, 720)
        for path in (self.stop_path, self.status_path, self.preview_path):
            if path.exists():
                path.unlink()
        self.write_overlay()
        cmd = [
            str(python), "-u", "-X", "utf8", "-B", str(HERE / "camera_job.py"),
            "--project", str(project),
            "--workdir", str(self.workdir),
            "--vision", self.vision,
            "preview", "--mode", mode,
        ]
        cmd.append("--real-camera")
        self.log = open(self.workdir / "camera_job.log", "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=str(HERE), env=sim_env(),
                                     stdout=self.log, stderr=subprocess.STDOUT)

    def write_overlay(self):
        payload = {
            "mode": self.mode,
            "calibration_view": self.calibration_view,
            "points": self.points,
            "params": self.params,
            "vision": self.vision,
            "group": getattr(self, "group", GROUP_PIECE),
        }
        tmp = self.overlay_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.overlay_path)

    def read_status(self):
        try:
            data = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if self.proc.poll() is not None or time.monotonic() - self.started_at > 15:
                return {"ok": False, "error": "预览启动失败，请返回重开"}
            return None
        if self.proc is not None and self.proc.poll() is not None:
            return {"ok": False, "error": "预览已退出，请返回重开"}
        if time.time() - data.get("heartbeat", 0) > 8:
            return {"ok": False, "error": "预览超时，请返回重开"}
        if data.get("cam_size"):
            self.cam_size = tuple(data["cam_size"])
        if data.get("letterbox"):
            self.letterbox = tuple(data["letterbox"])
        return data

    def stop(self):
        try:
            self.stop_path.write_text("1", encoding="utf-8")
        except OSError:
            pass
        if self.proc is None:
            return
        try:
            self.proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        if getattr(self, "log", None) is not None:
            try:
                self.log.close()
            except OSError:
                pass
            self.log = None


def load_project_config(project: Path):
    if project.name == "puzzle_vision":
        from ordinary_camera import load_config
        return load_config(project)
    config = json.loads((project / "config.json").read_text(encoding="utf-8-sig"))
    sim_path = project / "sim_config.json"
    if sim_path.is_file():
        config.update(json.loads(sim_path.read_text(encoding="utf-8")))
    return config


def run_job(python: Path, project: Path, args):
    vision = "poker" if "poker" in project.name else "ordinary"
    cmd = [str(python), "-u", "-X", "utf8", "-B", str(HERE / "camera_job.py"),
           "--project", str(project), "--workdir", str(HERE / "runtime"),
           "--vision", vision, *args]
    result = subprocess.run(cmd, cwd=str(HERE), env=sim_env(), capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "camera job failed").strip()
        raise RuntimeError(err.splitlines()[-1])
    return result.stdout.strip()


def read_planning_failure(project, started_at=0.0, output_name="output"):
    """Prefer a current-frame diagnosis; never show a stale successful plan."""
    output = Path(project) / output_name
    message = "方案未生成，请检查标定预览和本次日志"
    image = None
    try:
        diagnostic = json.loads((output / "detection_failure.json").read_text(encoding="utf-8"))
        if float(diagnostic.get("time", 0)) >= started_at:
            message = str(diagnostic.get("reason") or message)
            candidate = output / "detection_preview.jpg"
            if candidate.is_file() and candidate.stat().st_mtime >= started_at:
                image = candidate
    except (OSError, ValueError, TypeError):
        pass
    if image is None:
        try:
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            if float(plan.get("generated_at_epoch", 0)) >= started_at:
                message = "；".join(str(x) for x in plan.get("failure_reasons", [])) or message
        except (OSError, ValueError, TypeError):
            pass
    return message, image


def emergency_stop_dual() -> None:
    """Best-effort realtime stop for both controllers from the touch UI."""
    try:
        import serial
        grbl = serial.Serial("/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
                             115200, timeout=0.2, write_timeout=0.2)
        try:
            grbl.write(b"!")
            grbl.flush()
        finally:
            grbl.close()
    except Exception as error:
        print("Grbl emergency stop failed:", error, flush=True)
    try:
        import serial
        stm = serial.Serial("/dev/ttyS6", 115200, timeout=0.2, write_timeout=0.2)
        try:
            stm.write(b"STOP\nMAGNET,0\n")
            stm.flush()
        finally:
            stm.close()
    except Exception as error:
        print("STM32 emergency stop failed:", error, flush=True)


def next_ordinary_auto_action(phase, state):
    """One authorized run; failures or a consumed origin decision never retry."""
    if phase == "plan" and not state.busy and not state.workflow_stage:
        return "start", "review"
    if phase == "review" and not state.busy:
        if state.workflow_stage == "preview" and state.plan_ready:
            return "workflow_confirm", "origin"
        if state.workflow_stage in {"plan_failed", "failed"}:
            return None, ""
    if phase == "origin" and state.workflow_stage == "await_origin":
        return "workflow_confirm", ""
    if state.workflow_stage in {"failed", "complete"}:
        return None, ""
    return None, phase


def main(auto_ordinary=False, operator_origin_confirmed=False):
    if auto_ordinary and not operator_origin_confirmed:
        raise RuntimeError("自动开始普通拼图需要操作者确认本轮人工基准")
    auto_phase = "plan" if auto_ordinary else ""
    import signal
    def terminate(_signum, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, terminate)
    projects = find_projects()
    python = find_venv_python(projects["contest"])
    driver_dir = find_driver_dir()
    Panel, Draw, Text, GT911, rgb = load_driver(driver_dir)
    colors = theme_colors(rgb)
    panel = Panel()
    touch = None
    camera = None
    simulation = None
    active_project = None
    active_execute = False
    last_run_second = -1
    last_preview_mtime = 0.0
    preview_image = None
    planning_started_at = 0.0
    try:
        panel.init()
        panel.backlight(85)
        draw = Draw(panel)
        text = Text(draw)
        touch = GT911()
        # The physical SPI controller opens in the formal poker workflow;
        # Ordinary mode is a separate real-camera preview workflow.
        state = StartPageState(default_mode=MODE_ORDINARY if auto_ordinary else MODE_POKER)
        paint_home(draw, text, state, colors)
        print("SPI start page ready; driver", driver_dir,
              "ui_python", sys.executable, "vision_python", python, flush=True)
        while True:
            if simulation is not None:
                if active_execute and isinstance(simulation, VisionJob):
                    live_status = simulation.status()
                    if live_status and live_status != simulation.last_status:
                        simulation.last_status = live_status
                        state.workflow_stage = live_status.get("stage", "running")
                        state.workflow_message = live_status.get("message", "")
                        state.empty_run = live_status.get("empty_run", True)
                        state.armed = False
                        paint_home(draw, text, state, colors)
                code = simulation.poll()
                if code is not None:
                    if active_execute and isinstance(simulation, VisionJob):
                        final_status = simulation.status()
                        if final_status:
                            state.workflow_stage = final_status.get("stage", "failed")
                            state.workflow_message = final_status.get("message", "")
                        if code != 0 and final_status.get("stage") != "failed":
                            try:
                                name = "ordinary_execute.log" if state.mode == MODE_ORDINARY else "vision_execute.log"
                                lines = (HERE / "runtime" / name).read_text(encoding="utf-8").splitlines()
                                state.workflow_message = lines[-1] if lines else "启动失败，请查看日志"
                            except OSError:
                                state.workflow_message = "启动失败，请重新生成方案"
                    simulation = None
                    if not active_execute and state.mode == MODE_ORDINARY:
                        finish_ordinary_plan(state, active_project, planning_started_at, code)
                    elif code == 0 and not active_execute and state.mode == MODE_POKER:
                        try:
                            plan_path = active_project / "output" / "plan.json"
                            plan = json.loads(plan_path.read_text(encoding="utf-8"))
                            expected = int(plan.get("piece_count", 0))
                            state.plan_ready = plan.get("approval_state") == "review_required" and expected == 3
                            if state.plan_ready:
                                state.review_plan_path = Path(plan["artifact_path"])
                                state.review_plan_id = plan["plan_id"]
                                state.review_image_path = state.review_plan_path.parent / "assembly.png"
                                state.pattern_review_required = plan.get("pattern_review_required", False)
                                state.workflow_stage = "preview"
                            state.status = (f"方案就绪：{expected}块，按确认执行"
                                            if state.plan_ready else
                                            "方案未通过视觉/安全检查，只能重新识别")
                        except (OSError, json.JSONDecodeError, TypeError):
                            state.plan_ready = False
                            state.status = "未生成有效方案"
                        state.busy = False
                        state.armed = False
                    else:
                        state.finish_run(code == 0)
                        if not active_execute and code != 0 and state.mode == MODE_POKER:
                            state.plan_ready = False
                            state.workflow_stage = "plan_failed"
                            state.workflow_message, state.review_image_path = read_planning_failure(
                                active_project, planning_started_at)
                            state.review_plan_id = None
                            state.status = "方案未生成，请按提示调整"
                        if active_execute:
                            state.plan_ready = False
                            if code != 0:
                                state.workflow_stage = "failed"
                                state.workflow_message = state.workflow_message or "执行未完成，请查看日志"
                    paint_home(draw, text, state, colors)
                else:
                    second = int(time.monotonic() - run_started)
                    if second != last_run_second and not state.workflow_stage:
                        last_run_second = second
                        state.status = "实拍识别与方案检查中 %d秒" % second
                        paint_status(draw, text, state, colors)
            if camera is not None:
                status = camera.read_status()
                if state.page == PAGE_CALIB_PREVIEW and status and status.get("ok"):
                    notice = str(status.get("notice") or state.status)
                    if notice != state.status:
                        state.status = notice
                        paint_status(draw, text, state, colors)
                if status and not status.get("ok", True) and status.get("error"):
                    message = str(status["error"])[:18]
                    if state.page == PAGE_CALIB_PREVIEW and preview_image is not None:
                        preview_image = None
                        last_preview_mtime = 0.0
                        paint_calibration_preview_page(draw, text, state, colors)
                    if state.status != message:
                        state.status = message
                        paint_status(draw, text, state, colors)
                if camera.preview_path.exists() and (not status or status.get("ok", True)):
                    mtime = camera.preview_path.stat().st_mtime
                    if mtime != last_preview_mtime:
                        last_preview_mtime = mtime
                        try:
                            from PIL import Image
                            with Image.open(camera.preview_path) as img:
                                preview_image = img.convert("RGB").copy()
                        except Exception as error:
                            print("preview load failed:", error, flush=True)
                        else:
                            if state.page == PAGE_CALIB:
                                paint_calib_page(draw, text, state, colors, preview_image,
                                                 camera.points, camera.letterbox, camera.cam_size)
                            elif state.page == PAGE_TUNE:
                                paint_tune_preview(draw, text, colors, preview_image)
                            elif state.page in (PAGE_LIVE, PAGE_CALIB_PREVIEW):
                                paint_live_preview(draw, text, colors, preview_image, camera.letterbox)
            action = state.on_points(read_touch_report_safe(touch))
            if action is None and auto_phase:
                action, auto_phase = next_ordinary_auto_action(auto_phase, state)
            elif action in {"stop", "workflow_back", MODE_ORDINARY, MODE_POKER, "manual", "back"}:
                auto_phase = ""
            if action == "workflow_confirm":
                if state.workflow_stage == "preview":
                    state.begin_run()
                    active_execute = True
                    active_project = projects["ordinary"] if state.mode == MODE_ORDINARY else projects["poker"]
                    try:
                        job_type = OrdinaryCameraJob if state.mode == MODE_ORDINARY else VisionJob
                        simulation = job_type(python, active_project, execute=True,
                                              plan_path=state.review_plan_path,
                                              plan_id=state.review_plan_id)
                        run_started = time.monotonic()
                        state.workflow_stage = "connecting"
                        state.workflow_message = "连接设备，等待串口握手"
                    except Exception as exc:
                        state.busy = False
                        state.workflow_stage = "failed"
                        state.workflow_message = str(exc)
                elif simulation is not None:
                    try:
                        simulation.confirm()
                        state.workflow_stage = "confirming"
                        state.workflow_message = "已确认，正在检查"
                    except Exception as exc:
                        state.workflow_message = str(exc)
                paint_home(draw, text, state, colors)
            elif action == "workflow_back":
                state.workflow_stage = ""
                state.plan_ready = False
                state.status = "请重新识别当前四片" if state.mode == MODE_ORDINARY else "请重新识别当前三片"
                paint_home(draw, text, state, colors)
            elif action in (MODE_ORDINARY, MODE_POKER):
                if state.set_mode(action):
                    paint_home(draw, text, state, colors)
            elif action == "manual":
                state.plan_ready = False
                from manual_page import run_manual_page
                run_manual_page(draw, text, colors, lambda: read_touch_report_safe(touch), paint_button)
                state.armed = False
                state.go_home()
                paint_home(draw, text, state, colors)
            elif action == "start":
                if state.busy:
                    continue
                state.begin_run()
                paint_home(draw, text, state, colors)
                project = projects["poker"] if state.mode == MODE_POKER else projects["ordinary"]
                active_project = project
                active_execute = False
                print("launch", state.mode, "execute", active_execute, "cwd", project, flush=True)
                try:
                    planning_started_at = time.time()
                    simulation = (VisionJob(python, project, execute=active_execute)
                                  if state.mode == MODE_POKER else OrdinaryCameraJob(python, project))
                    run_started = time.monotonic()
                    last_run_second = -1
                except Exception as error:
                    state.finish_run(False)
                    paint_home(draw, text, state, colors)
                    print("vision launch failed:", error, flush=True)
            elif action == "stop":
                stop_message = stop_workflow_job(simulation, active_execute)
                simulation = None
                # The session owner has already issued STOP. Reopening the
                # Grbl port here could reset its coordinates after stopping.
                state.busy = False
                state.plan_ready = False
                state.workflow_stage = "failed" if "未确认" in stop_message else ""
                state.workflow_message = stop_message
                state.status = stop_message
                state.armed = False
                paint_home(draw, text, state, colors)
            elif action == "calib_preview":
                project = projects["poker"] if state.mode == MODE_POKER else projects["ordinary"]
                if camera is not None:
                    camera.stop()
                state.open_calibration_preview()
                camera = CameraSession(python, project, "calib_preview")
                if state.mode == MODE_ORDINARY:
                    state.calibration_view = camera.calibration_view = "mask"
                    camera.write_overlay()
                preview_image = None
                last_preview_mtime = 0.0
                paint_calibration_preview_page(draw, text, state, colors)
            elif isinstance(action, tuple) and action[0] == "calibration_view" and camera is not None:
                camera.calibration_view = action[1]
                camera.write_overlay()
                preview_image = None
                paint_calibration_preview_page(draw, text, state, colors)
            elif action == "calib":
                # Both real modes share one physical camera and its saved calibration.
                project = projects["poker"]
                if camera is not None:
                    camera.stop()
                state.open_calib()
                camera = CameraSession(python, project, "calib")
                preview_image = None
                last_preview_mtime = 0.0
                paint_calib_page(draw, text, state, colors, None, camera.points, camera.letterbox, camera.cam_size)
            elif action == "live":
                project = projects["poker"] if state.mode == MODE_POKER else projects["ordinary"]
                if camera is not None:
                    camera.stop()
                state.open_live()
                camera = CameraSession(python, project, "live")
                preview_image = None
                last_preview_mtime = 0.0
                paint_live_page(draw, text, state, colors)
            elif action == "tune":
                project = projects["poker"] if state.mode == MODE_POKER else projects["ordinary"]
                config = load_project_config(project)
                if camera is not None:
                    camera.stop()
                state.open_tune(config)
                camera = CameraSession(python, project, "tune")
                camera.params = state.slider_params()
                camera.group = state.tune_group
                camera.write_overlay()
                preview_image = None
                last_preview_mtime = 0.0
                paint_tune_page(draw, text, state, colors, preview_image)
            elif action == "back":
                if camera is not None:
                    camera.stop()
                    camera = None
                preview_image = None
                last_preview_mtime = 0.0
                state.go_home()
                paint_home(draw, text, state, colors)
            elif isinstance(action, tuple) and action and action[0] == "preview" and camera is not None:
                _kind, px, py = action
                mapped = map_preview_to_camera(px, py, PREVIEW_RECT, camera.letterbox, camera.cam_size)
                if mapped is None:
                    state.status = "请点画面内"
                    paint_status(draw, text, state, colors)
                elif len(camera.points) >= 4:
                    state.status = "已有四点，先保存或撤销"
                    paint_status(draw, text, state, colors)
                else:
                    camera.points.append([round(mapped[0], 1), round(mapped[1], 1)])
                    camera.write_overlay()
                    state.calib_count = len(camera.points)
                    state.status = state.calib_hint()
                    paint_calib_page(draw, text, state, colors, preview_image,
                                        camera.points, camera.letterbox, camera.cam_size)
            elif action == "undo" and camera is not None:
                if camera.points:
                    camera.points.pop()
                    camera.write_overlay()
                    state.calib_count = len(camera.points)
                    state.status = state.calib_hint()
                    paint_calib_page(draw, text, state, colors, preview_image,
                                        camera.points, camera.letterbox, camera.cam_size)
            elif action == "save_cal" and camera is not None:
                if len(camera.points) < 4:
                    state.status = "先点四个角"
                    paint_status(draw, text, state, colors)
                else:
                    try:
                        run_job(python, camera.project, ["save-calib", "--points", json.dumps(camera.points)])
                        state.status = "标定已保存"
                    except Exception as error:
                        state.status = str(error)[:18]
                    paint_status(draw, text, state, colors)
            elif action == "group_piece" and camera is not None:
                if state.set_tune_group(GROUP_PIECE, load_project_config(camera.project)):
                    camera.params = dict(state.stored_params)
                    camera.params.update(state.slider_params())
                    camera.group = state.tune_group
                    camera.write_overlay()
                    paint_tune_page(draw, text, state, colors, preview_image)
            elif action == "group_board" and camera is not None:
                if state.set_tune_group(GROUP_BOARD, load_project_config(camera.project)):
                    camera.params = dict(state.stored_params)
                    camera.params.update(state.slider_params())
                    camera.group = state.tune_group
                    camera.write_overlay()
                    paint_tune_page(draw, text, state, colors, preview_image)
            elif action == "page_prev" and camera is not None:
                if state.shift_tune_page(-1):
                    paint_tune_page(draw, text, state, colors, preview_image)
            elif action == "page_next" and camera is not None:
                if state.shift_tune_page(1):
                    paint_tune_page(draw, text, state, colors, preview_image)
            elif action == "threshold_mode" and camera is not None and state.tune_group == GROUP_PIECE:
                state.threshold_mode = "fixed" if state.threshold_mode == "otsu" else "otsu"
                camera.params = state.slider_params()
                camera.write_overlay()
                paint_tune_page(draw, text, state, colors, preview_image)
            elif action == "save_tune" and camera is not None:
                try:
                    run_job(python, camera.project, ["save-config", "--params", json.dumps(state.slider_params())])
                    state.status = "阈值已保存"
                except Exception as error:
                    state.status = str(error)[:18]
                paint_status(draw, text, state, colors)
            elif isinstance(action, tuple) and action and action[0] == "slider":
                _kind, index, value = action
                if state.set_slider(index, value):
                    if camera is not None:
                        camera.params = dict(state.stored_params)
                        camera.params.update(state.slider_params())
                        camera.group = state.tune_group
                        camera.write_overlay()
                    paint_button(draw, text, MODE_BTN, colors["primary"],
                                 "阈值模式: " + state.threshold_mode, colors, size=20)
                    for index, (_key, label, lo, hi, value) in enumerate(state.sliders):
                        row = slider_row(index)
                        lx, ly, lw, lh = row["label"]
                        draw.fill_rect(lx, ly, lw, lh, colors["scr"])
                        suffix = ""
                        if ("value_min" in _key and "otsu" not in _key and state.threshold_mode != "fixed") or (
                                "otsu" in _key and state.threshold_mode != "otsu"):
                            suffix = "（未启用）"
                        text.text(lx, ly, "%s  %d" % (label + suffix, value), size=20,
                                  color=colors["text"], bg=colors["scr"])
                        paint_button(draw, text, row["minus"], colors["grey"], "-", colors, radius=10, size=28)
                        tx, ty, tw, th = row["track"]
                        draw.fill_rect(tx, ty, tw, th, colors["grey"])
                        if hi > lo:
                            knob_w = 18
                            ratio = (value - lo) / float(hi - lo)
                            kx = tx + int(ratio * (tw - knob_w))
                            fill_round_rect(draw, kx, ty - 6, knob_w, th + 12, 8, colors["primary"])
                        paint_button(draw, text, row["plus"], colors["grey"], "+", colors, radius=10, size=28)
                    paint_status(draw, text, state, colors)
            time.sleep(0.03)
    finally:
        if simulation is not None:
            simulation.close()
        if camera is not None:
            camera.stop()
        if touch is not None:
            touch.close()
        panel.core.close()


if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--ordinary-run", action="store_true")
        parser.add_argument("--operator-origin-confirmed", action="store_true")
        options = parser.parse_args()
        from runtime_support import SingleInstance
        with SingleInstance(HERE / "runtime" / "screen.lock"):
            main(options.ordinary_run, options.operator_origin_confirmed)
    except KeyboardInterrupt:
        print("\n退出")
    except RuntimeError as error:
        print(str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
