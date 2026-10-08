"""Exercise the vision GUI with native Windows input and save rendered evidence."""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：Windows 下用原生鼠标消息点 PyBullet 窗口，自动测拖动/旋转/START/HOME/故障恢复。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约35行  main：找窗口、投消息、截图、写 gui_result.json
# =============================================================================
# =============================================================================
# 【分区】文件说明
# 功能：用 Windows 原生消息点 PyBullet 窗口，走一遍拖动/旋转/START/HOME/故障恢复，并截图存证。
# 可修改：输出目录 output_sim/qa_gui；截图像素 960×720。
# 看情况改：仅 Windows + 真实 GUI 窗口可用。无界面或非 Windows 不要跑本文件。
# 不要改：断言阈值和按钮名 start/home/rotate；改掉就不再验证原仿真协议。
# =============================================================================

import contextlib
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import time

import numpy as np
import run_sim


# =============================================================================
# 【分区】GUI 自动化主流程
# 功能：打开仿真 → 找本进程窗口 → 投鼠标消息 → 拖一片、点旋转、START、HOME，再测空场景故障锁。
# 可修改：pump 等待 0.15s、拖动 5mm、旋转按钮 15°（在 run_sim.handle_command 里）。
# 看情况改：DPI 感知、EnumWindows 找窗口；多显示器缩放不对时先查窗口句柄，不要先改视觉算法。
# 不要改：鼠标消息 0x200/0x201/0x202（移动/按下/抬起）；不要改成只调内部 API，那就测不到真实 GUI。
# =============================================================================
def main():
    vision, simulator, camera_class = run_sim.load_modules(run_sim.DEFAULT_SIM_ROOT)
    import pybullet as p
    user = ctypes.windll.user32
    user.SetProcessDPIAware()
    output = run_sim.ROOT / "output_sim" / "qa_gui"
    output.mkdir(parents=True, exist_ok=True)
    (output / "gui_result.json").write_text('{"status": "running"}', encoding="utf-8")
    sim = simulator.PuzzleGantrySim(True, False, scene="vision")
    session = run_sim.VisionSession(vision, simulator, camera_class, sim, output)
    try:
        sim.spawn_pieces(run_sim.make_scene(vision, simulator))
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

        def pump(seconds=0.15):
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
            mouse_message(0x201)
            pump()
            mouse_message(0x202)
            pump()
            assert commands == [name], (name, commands)
            session.handle_command(commands.pop())

        def screenshot(name):
            camera = p.getDebugVisualizerCamera()
            rgba = p.getCameraImage(960, 720, camera[2], camera[3], renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            bgr = np.asarray(rgba, dtype=np.uint8).reshape(720, 960, 4)[:, :, :3][:, :, ::-1]
            assert bgr.std() > 20
            vision.write_image(output / name, bgr)

        before_frame = session.camera.capture()
        vision.write_image(output / "camera_before.png", before_frame)
        before_paper = vision.warp_paper(before_frame, session.camera.matrix, session.config)
        before_pieces, _ = vision.detect_pieces(before_paper, session.config)
        assert len(before_pieces) == 4
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
        assert sim.drag_body is None, "mouse release did not end drag"
        moved = p.getBasePositionAndOrientation(body)[0]
        assert sim.selected_body == body
        assert np.linalg.norm(np.asarray(moved[:2]) - original[:2]) > 0.003
        button("rotate")
        after_frame = session.camera.capture()
        vision.write_image(output / "camera_after_edit.png", after_frame)
        after_paper = vision.warp_paper(after_frame, session.camera.matrix, session.config)
        after_pieces, _ = vision.detect_pieces(after_paper, session.config)
        assert len(after_pieces) == 4
        before_centers = np.array([piece.center_mm(4) for piece in before_pieces])
        after_centers = np.array([piece.center_mm(4) for piece in after_pieces])
        distances = np.linalg.norm(before_centers[:, None] - after_centers[None, :], axis=2)
        assert distances.min(axis=1).max() > 3, "image detection did not reflect the drag"
        assert np.count_nonzero(before_frame != after_frame) > 1000
        screenshot("edited_scene.png")
        button("start")
        assert sim.status == "DONE - VISION PASS", sim.status
        screenshot("completed_scene.png")
        first_output = str(session.last_output)
        button("home")
        assert sim.fault is None
        restored = session.camera.capture()
        vision.write_image(output / "camera_home.png", restored)
        assert np.mean(np.abs(restored.astype(float) - before_frame.astype(float))) < 1
        button("start")
        assert sim.status == "DONE - VISION PASS", sim.status
        button("home")
        # A missing scene must fault, lock START, and recover through the real HOME button.
        sim.spawn_pieces([], remember_home=False)
        button("start")
        assert sim.fault and sim.status.startswith("FAULT")
        fault_output = session.last_output
        button("start")
        assert session.last_output == fault_output
        button("home")
        assert sim.fault is None and len(sim.pieces) == 4
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
        sim.close()


# =============================================================================
# 【分区】脚本入口
# 功能：把控制台重定向到 console.log 后跑 main，失败时 gui_result.json 记 failed。
# 可修改：日志路径。
# 看情况改：调试时可临时去掉 redirect_stdout，方便看实时打印。
# 不要改：__name__ == "__main__" 保护；被 import 时不应自动弹 GUI。
# =============================================================================
if __name__ == "__main__":
    output = run_sim.ROOT / "output_sim" / "qa_gui"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        main()
    print("GUI QA PASS:", output)
