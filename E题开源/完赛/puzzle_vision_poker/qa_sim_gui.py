"""Exercise the vision GUI with native Windows input and save rendered evidence."""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：扑克版 GUI QA：3 片，成功状态是 DONE - POKER VISION PASS，并用空白花纹测 START 锁定。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约39行  main：原生鼠标走一遍拖动/旋转/START/HOME/花纹故障
# =============================================================================
# =============================================================================
# 【分区】文件说明（扑克仿真 GUI QA）
# 功能：Windows 原生消息点 PyBullet 窗口，走拖动/旋转/START/HOME，并用空白花纹触发故障锁。
# 可修改：输出目录带时间戳；pump 默认 0.35s（比普通版稍长，等花纹渲染）。
# 看情况改：仅 Windows + 真实 GUI。断言是 3 片，不是 4 片。
# 不要改：状态字符串 DONE - POKER VISION PASS；不要改成普通版的 DONE - VISION PASS。
# =============================================================================

import contextlib
from datetime import datetime
from unittest.mock import patch
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import time

import numpy as np
import run_sim


QA_OUTPUT = run_sim.ROOT / "output_sim" / "qa_gui" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")

# =============================================================================
# 【分区】GUI 自动化主流程
# 功能：找窗口 → 投鼠标 → 拖一片、旋转、START、HOME；再用 blank 花纹测 START 锁定。
# 可修改：拖动 5mm、按钮重试 3 次。
# 看情况改：DPI/多屏导致点不中按钮时先查 hwnd，不要改视觉。
# 不要改：鼠标消息 0x200/0x201/0x202；3 片断言。
# =============================================================================
def main():
    vision, simulator, camera_class = run_sim.load_modules(run_sim.DEFAULT_SIM_ROOT)
    import pybullet as p
    user = ctypes.windll.user32
    user.SetProcessDPIAware()
    output = QA_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    (output / "gui_result.json").write_text('{"status": "running"}', encoding="utf-8")
    sim = simulator.PuzzleGantrySim(True, False, scene="vision")
    session = run_sim.PokerSession(vision, simulator, camera_class, sim, output)
    try:
        sim.spawn_pieces(run_sim.make_scene(simulator))
        session.park()
        time.sleep(0.5)
        windows = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def enum(hwnd, _):
            pid = wintypes.DWORD()
            user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == os.getpid() and user.IsWindowVisible(hwnd):
                windows.append(hwnd)
            return True

        user.EnumWindows(enum, 0)
        if not windows:
            raise RuntimeError("PyBullet GUI window not found")
        hwnd = windows[0]
        user.ShowWindow(hwnd, 9)
        time.sleep(0.2)
        commands = []
        mouse_point = [0, 0]
        mouse_down = False
        original_mouse_events = p.getMouseEvents

        def targeted_mouse_events():
            # Ignore unrelated physical pointer input while exercising native window messages.
            return tuple(event for event in original_mouse_events()
                         if (event[1], event[2]) == tuple(mouse_point))

        p.getMouseEvents = targeted_mouse_events

        def mouse_message(message):
            nonlocal mouse_down
            if message == 0x201:
                mouse_down = True
            elif message == 0x202:
                mouse_down = False
            packed = (mouse_point[1] << 16) | (mouse_point[0] & 0xFFFF)
            user.PostMessageW(hwnd, message, 1 if mouse_down else 0, packed)

        def pump(seconds=0.35):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                sim._handle_piece_mouse()
                command = sim.poll_buttons()
                if command:
                    commands.append(command)
                sim.step()
                time.sleep(0.005)

        def cursor(world):
            camera = p.getDebugVisualizerCamera()
            width, height = camera[:2]
            view = np.asarray(camera[2]).reshape(4, 4, order="F")
            projection = np.asarray(camera[3]).reshape(4, 4, order="F")
            clip = projection @ view @ np.array([*world, 1])
            ndc = clip[:3] / clip[3]
            mouse_point[:] = [round((ndc[0] + 1) * width / 2),
                              round((1 - ndc[1]) * height / 2)]
            mouse_message(0x200)

        def button(name):
            commands.clear()
            body = next(body for body, action in sim.ui_buttons.items() if action == name)
            pos = p.getBasePositionAndOrientation(body)[0]
            cursor([pos[0], pos[1], pos[2] + 0.012])
            pump()
            for attempt in range(3):
                mouse_message(0x201)
                pump()
                mouse_message(0x202)
                pump()
                if commands:
                    break
            assert commands == [name], (name, commands)
            session.handle_command(commands.pop())

        def screenshot(name):
            camera = p.getDebugVisualizerCamera()
            rgba = p.getCameraImage(960, 720, camera[2], camera[3], renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            bgr = np.asarray(rgba, dtype=np.uint8).reshape(720, 960, 4)[:, :, :3][:, :, ::-1]
            assert bgr.std() > 20
            vision.write_image(output / name, bgr)

        screenshot("initial_scene.png")
        before_frame = session.camera.capture()
        vision.write_image(output / "camera_before.png", before_frame)
        before_paper = vision.warp_paper(before_frame, session.camera.matrix, session.config)
        before_pieces, _ = vision.detect_pieces(before_paper, session.config)
        assert len(before_pieces) == 3
        body = sim.pieces[0]
        original = p.getBasePositionAndOrientation(body)[0]
        cursor(original)
        pump()
        mouse_message(0x201)
        pump()
        cursor([original[0] - 0.005, original[1] - 0.005, original[2]])
        pump()
        mouse_message(0x202)
        pump()
        if sim.drag_body is not None:
            mouse_message(0x202)
            pump(0.5)
        assert sim.drag_body is None, "mouse release did not end drag"
        moved = p.getBasePositionAndOrientation(body)[0]
        assert sim.selected_body == body
        assert np.linalg.norm(np.asarray(moved[:2]) - original[:2]) > 0.003
        button("rotate")
        after_frame = session.camera.capture()
        vision.write_image(output / "camera_after_edit.png", after_frame)
        after_paper = vision.warp_paper(after_frame, session.camera.matrix, session.config)
        after_pieces, _ = vision.detect_pieces(after_paper, session.config)
        assert len(after_pieces) == 3
        before_centers = np.array([piece.center_mm(4) for piece in before_pieces])
        after_centers = np.array([piece.center_mm(4) for piece in after_pieces])
        distances = np.linalg.norm(before_centers[:, None] - after_centers[None, :], axis=2)
        assert distances.min(axis=1).max() > 3, "image detection did not reflect the drag"
        assert np.count_nonzero(before_frame != after_frame) > 1000
        screenshot("edited_scene.png")
        button("start")
        assert sim.status == "DONE - POKER VISION PASS", sim.status
        screenshot("completed_scene.png")
        first_output = str(session.last_output)
        button("home")
        assert sim.fault is None
        restored = session.camera.capture()
        vision.write_image(output / "camera_home.png", restored)
        assert np.mean(np.abs(restored.astype(float) - before_frame.astype(float))) < 1
        button("start")
        assert sim.status == "DONE - POKER VISION PASS", sim.status
        button("home")
        # Missing pattern evidence must fault before dispatch and recover through HOME.
        extract = vision.edge_matcher.extract_edge_profiles
        def blank_profiles(paper, *args, **kwargs):
            return extract(np.full_like(paper, 220), *args, **kwargs)
        with patch.object(vision.edge_matcher, "extract_edge_profiles", side_effect=blank_profiles):
            button("start")
        assert sim.fault and sim.status.startswith("FAULT")
        fault_output = session.last_output
        button("start")
        assert session.last_output == fault_output
        button("home")
        assert sim.fault is None and len(sim.pieces) == 3
        screenshot("home_scene.png")
        report = {"status": "pass", "native_drag": True, "rotation": True,
                  "image_detection_changed": True, "repeated_run": True,
                  "fault_start_lock": True, "home_recovery": True, "first_output": first_output}
        (output / "gui_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("GUI QA PASS:", output)
    except Exception as error:
        (output / "gui_result.json").write_text(
            json.dumps({"status": "failed", "error": str(error)}, indent=2), encoding="utf-8")
        raise
    finally:
        if "original_mouse_events" in locals():
            p.getMouseEvents = original_mouse_events
        sim.close()


# =============================================================================
# 【分区】脚本入口
# 功能：stdout 重定向到 console.log 后跑 main。
# 可修改：日志路径（QA_OUTPUT）。
# 看情况改：调试可临时去掉 redirect。
# 不要改：__name__ 保护。
# =============================================================================
if __name__ == "__main__":
    output = QA_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    with (output / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        main()
    print("GUI QA PASS:", output)
