"""Focused bridge regressions. Run using the PyBullet virtual environment."""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：视觉↔仿真桥接回归测试：坐标、相机、路径、非法路径不得运动、空场景、终检失败。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约43行  BridgeTests：unittest 用例集合，每例独立仿真会话
# =============================================================================
# =============================================================================
# 【分区】文件说明
# 功能：视觉↔仿真桥接回归测试：坐标互换、相机几何、路径旋转偏心吸取、非法路径、空场景、终检失败。
# 可修改：无现场参数。跑测试请用 PyBullet 那套虚拟环境，不要用系统 Python。
# 看情况改：断言公差 atol=0.5 等随仿真版本可能要微调；先确认是仿真变了还是视觉变了。
# 不要改：测试名和失败语义（非法路径不得调用 set_pose、空场景不得 run_queue）。改断言就失去回归意义。
# =============================================================================

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import run_sim

vision, simulator, Camera = run_sim.load_modules(run_sim.DEFAULT_SIM_ROOT)
import pybullet as p
from protocol import encode_frame, parse_frame


# =============================================================================
# 【分区】BridgeTests
# 功能：每个用例独立仿真会话；setUp 重定向 stdout，tearDown 关闭仿真和临时目录。
# 可修改：无。
# 看情况改：新增回归应放本类，保持“失败不得驱动电机”的风格。
# 不要改：setUp/tearDown 资源配对；不要在测试里改成 GUI 模式。
# =============================================================================
class BridgeTests(unittest.TestCase):
    def test_simulation_config_does_not_read_real_config(self):
        with patch.object(vision, "load_config", side_effect=AssertionError("real config read")):
            config = run_sim.simulation_config(vision)
        self.assertEqual(config["serial_port"], "")
        self.assertFalse(config["force_execute_on_any_solution"])

    def setUp(self):
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.temp = tempfile.TemporaryDirectory()
        self.sim = simulator.PuzzleGantrySim(False, False, scene="vision")
        self.session = run_sim.VisionSession(vision, simulator, Camera, self.sim, Path(self.temp.name))

    def tearDown(self):
        self.sim.close()
        self.temp.cleanup()
        self.redirect.__exit__(None, None, None)

    def test_coordinate_corners_and_roundtrip(self):
        for source, expected in [((0, 0), (210, 297)), ((210, 0), (0, 297)),
                                 ((0, 297), (210, 0)), ((210, 297), (0, 0)),
                                 ((27.4, 56.8), (182.6, 240.2))]:
            np.testing.assert_allclose(run_sim.convert_xy(source), expected)
            np.testing.assert_allclose(run_sim.convert_xy(run_sim.convert_xy(source)), source)

    def test_camera_geometry_uses_rgb_only(self):
        self.sim.spawn_pieces([simulator.PieceSpec("sensor", (20, 10), (150, 200))])
        self.session.park()
        frame = self.session.camera.capture()
        paper = vision.warp_paper(frame, self.session.camera.matrix, self.session.config)
        pieces, _ = vision.detect_pieces(paper, self.session.config)
        self.assertEqual(len(pieces), 1)
        np.testing.assert_allclose(pieces[0].center_mm(4), [60, 97], atol=0.5)
        self.assertGreater(float(paper[388, 240].mean()) - float(paper[40, 40].mean()), 150)

    def test_paths_rotation_and_eccentric_pick(self):
        for angle in (-90, 90):
            with self.subTest(angle=angle):
                self.sim.spawn_pieces([simulator.PieceSpec("offset", (30, 20), (100, 220), 15)])
                task = simulator.MotionTask(107, 220, angle, 120, 80,
                                            ((107, 220), (120, 220), (120, 80)))
                with patch.object(self.sim, "set_pose", wraps=self.sim.set_pose) as move:
                    result = self.sim.run_queue([task])[0]
                targets = [call.args[:2] for call in move.call_args_list]
                self.assertIn((120, 220), targets)
                self.assertLess(targets.index((120, 220)), targets.index((120, 80)))
                expected = (120, 80 - 7 * np.sign(angle))
                np.testing.assert_allclose(result.actual_pose[:2], expected, atol=0.5)
                self.assertLess(abs(simulator.angle_difference(result.actual_pose[2], 15 + angle)), 0.5)

    def test_invalid_path_atomic_and_protocol_compatibility(self):
        self.sim.spawn_pieces([simulator.PieceSpec("stationary", (20, 20), (100, 200))])
        invalid = [(), ((100, 200), (120, 80)), ((100, 200), (float("nan"), 200), (120, 80)),
                   ((100, 200), (211, 200), (211, 80), (120, 80)), ((100, 200), (120, 200))]
        for path in invalid:
            with self.subTest(path=path), patch.object(self.sim, "set_pose") as move:
                self.sim.fault = None
                with self.assertRaises(simulator.ExecutionError):
                    self.sim.run_queue([simulator.MotionTask(100, 200, 0, 120, 80, path)])
                move.assert_not_called()
        task = simulator.MotionTask(100, 200, 0, 120, 80)
        self.assertEqual(parse_frame(encode_frame([task])), [task])
        task.path_mm = ((100, 200), (120, 200), (120, 80))
        with self.assertRaisesRegex(ValueError, "waypoints"):
            encode_frame([task])

    def test_empty_and_missing_piece_fail_without_motion(self):
        for specs in ([], run_sim.make_scene(vision, simulator)[:3]):
            self.sim.home()
            self.sim.spawn_pieces(specs)
            with patch.object(self.sim, "run_queue") as run:
                with self.assertRaises(simulator.ExecutionError):
                    self.session.run_once()
                run.assert_not_called()
            report = json.loads((self.session.last_output / "run_result.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["stage"], "planning")
            self.assertFalse((self.session.last_output / "plan.json").exists())
            previous = self.session.last_output
            self.session.handle_command("start")
            self.assertEqual(previous, self.session.last_output)
            self.session.handle_command("home")
            self.assertIsNone(self.sim.fault)

    def test_empty_pick_stops_queue(self):
        with self.assertRaises(simulator.ExecutionError) as caught:
            self.sim.run_queue([simulator.MotionTask(50, 50, 0, 60, 60),
                                simulator.MotionTask(70, 70, 0, 80, 80)])
        self.assertEqual(caught.exception.task_index, 1)
        self.assertTrue(self.sim.fault)
        self.assertEqual(self.sim.results, [])

    def test_failed_final_capture_is_not_done(self):
        self.sim.spawn_pieces(run_sim.make_scene(vision, simulator))
        capture = self.session.camera.capture
        captures = 0

        def shifted_capture():
            nonlocal captures
            captures += 1
            if captures == 2:
                body = self.sim.pieces[0]
                pos, orn = p.getBasePositionAndOrientation(body)
                p.resetBasePositionAndOrientation(body, [pos[0] + 0.020, pos[1], pos[2]], orn)
            return capture()

        with patch.object(self.session.camera, "capture", side_effect=shifted_capture):
            with self.assertRaises(simulator.ExecutionError):
                self.session.run_once()
        result = json.loads((self.session.last_output / "run_result.json").read_text())
        self.assertEqual(result["stage"], "verification")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.sim.status.startswith("DONE"))
        plan = json.loads((self.session.last_output / "plan.json").read_text())
        valid_tasks = run_sim.tasks_from_plan(plan, vision, simulator)
        ordered = sorted(plan["pieces"], key=lambda piece: piece["move_order"])
        self.assertEqual(len(valid_tasks), 4)
        for piece, task in zip(ordered, valid_tasks):
            np.testing.assert_allclose(task.path_mm, [run_sim.convert_xy(pt) for pt in piece["path_mm"]])
            self.assertEqual(task.flip_deg, piece["rotation_deg_signed"])
        for mutation in ("ready", "coordinates", "path"):
            invalid = copy.deepcopy(plan)
            if mutation == "ready":
                invalid["ready_for_motion"] = False
            elif mutation == "coordinates":
                invalid["coordinate_system"]["origin"] = "A4_bottom_right"
            else:
                invalid["pieces"][0]["path_mm"] = [[0, 0], [float("nan"), 10]]
            with self.subTest(mutation=mutation):
                with self.assertRaises((ValueError, simulator.ExecutionError)):
                    run_sim.tasks_from_plan(invalid, vision, simulator)

    def test_direct_vision_failure_exit_code(self):
        script = """
import runpy, sys, numpy as np, run_sim
_, _, camera = run_sim.load_modules(run_sim.DEFAULT_SIM_ROOT)
camera.capture = lambda self: np.zeros((self.height, self.width, 3), dtype=np.uint8)
sys.argv = ['run_sim.py', '--direct', '--output-dir', sys.argv[1]]
runpy.run_path(str(run_sim.ROOT / 'run_sim.py'), run_name='__main__')
"""
        output = Path(self.temp.name) / "cli"
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", script, str(output)],
                                cwd=run_sim.ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("SIM FAILED", result.stderr)
        reports = list(output.glob("*/run_result.json"))
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0].read_text())["stage"], "planning")


if __name__ == "__main__":
    unittest.main(verbosity=2)
