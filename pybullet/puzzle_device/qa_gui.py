"""Windows GUI smoke test with native mouse input; closes its own window."""

import ctypes
from ctypes import wintypes
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pybullet as p

from simulator import ExecutionError, MotionTask, PuzzleGantrySim, angle_difference, demo_pieces


def main():
    user = ctypes.windll.user32
    user.SetProcessDPIAware()
    sim = PuzzleGantrySim(True, False)
    commands = []
    try:
        sim.spawn_pieces(demo_pieces())
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
            raise RuntimeError("GUI window not found")
        hwnd = windows[0]
        user.SetForegroundWindow(hwnd)
        time.sleep(0.2)
        if user.GetForegroundWindow() != hwnd:
            raise RuntimeError("Windows denied GUI focus; rerun without switching windows")

        def pump(seconds=0.15):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                sim._handle_piece_mouse()
                command = sim.poll_buttons()
                if command:
                    commands.append(command)
                sim.step()
                time.sleep(0.005)

        def pixel(world):
            camera = p.getDebugVisualizerCamera()
            width, height = camera[:2]
            view = np.asarray(camera[2]).reshape(4, 4, order="F")
            projection = np.asarray(camera[3]).reshape(4, 4, order="F")
            clip = projection @ view @ np.array([*world, 1])
            ndc = clip[:3] / clip[3]
            return ((ndc[0] + 1) * width / 2, (1 - ndc[1]) * height / 2)

        def cursor(world):
            x, y = pixel(world)
            point = wintypes.POINT(round(x), round(y))
            user.ClientToScreen(hwnd, ctypes.byref(point))
            user.SetCursorPos(point.x, point.y)

        def click(world):
            cursor(world)
            pump()
            user.mouse_event(2, 0, 0, 0, 0)
            pump()
            user.mouse_event(4, 0, 0, 0, 0)
            pump()

        def button(name):
            commands.clear()
            body = next(body for body, value in sim.ui_buttons.items() if value == name)
            pos = p.getBasePositionAndOrientation(body)[0]
            click([pos[0], pos[1], pos[2] + 0.012])
            if commands != [name]:
                raise AssertionError(f"button {name}: {commands}")

        pump()
        body = sim.pieces[0]
        before = p.getBasePositionAndOrientation(body)[0]
        cursor(before)
        pump()
        user.mouse_event(2, 0, 0, 0, 0)
        pump()
        cursor([before[0] + 0.006, before[1] - 0.006, before[2]])
        pump()
        user.mouse_event(4, 0, 0, 0, 0)
        pump()
        after = p.getBasePositionAndOrientation(body)[0]
        assert sim.selected_body == body, "mouse did not select piece"
        assert np.linalg.norm(np.array(after[:2]) - before[:2]) > 0.003, "drag did not move piece"
        for _ in range(3):
            button("rotate")
            sim.rotate_selected(15)
        yaw = math.degrees(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(body)[1])[2])
        assert abs(angle_difference(yaw, 45)) < 2
        pos = p.getBasePositionAndOrientation(body)[0]
        button("start")
        sim.run_queue([MotionTask(pos[0] * 1000, pos[1] * 1000, -45, 65, 65)])
        assert sim.status.startswith("DONE")
        button("home")
        sim.home()
        button("start")
        sim.run_queue([MotionTask(61.5625, 240, 45, 65, 65)])
        button("home")
        sim.home()
        try:
            sim.run_queue([MotionTask(10, 10, 0, 20, 20)])
        except ExecutionError:
            pass
        assert sim.fault and sim.status.startswith("FAULT")
        button("start")
        try:
            sim.run_queue([MotionTask(61.5625, 240, 0, 65, 65)])
        except ExecutionError as error:
            assert error.code == "FAULT_LOCKED"
        else:
            raise AssertionError("fault did not lock START")
        button("home")
        sim.home()
        assert sim.fault is None
        pump()
        camera = p.getDebugVisualizerCamera()
        rgba = p.getCameraImage(960, 720, camera[2], camera[3], renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        image = np.asarray(rgba, dtype=np.uint8).reshape(720, 960, 4)
        assert image[:, :, :3].std() > 20, "blank render"
        path = Path(__file__).resolve().parent / "qa_gui.png"
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGBA2BGR))
        assert ok
        path.write_bytes(encoded.tobytes())
        print("GUI QA PASS: native drag, 3 rotations, START, repeated run, fault, HOME; screenshot:", path)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
