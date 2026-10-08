"""Focused bridge regressions. Run using the PyBullet virtual environment."""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：扑克桥接回归：除普通版项目外，还有花纹门禁、宽松解禁发、拆边索引。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约43行  BridgeTests：unittest 用例集合
# =============================================================================
# =============================================================================
# 【分区】文件说明（扑克桥接回归）
# 功能：坐标、相机、路径、非法路径、空场景、终检、花纹门禁、宽松解禁发、拆边索引。
# 可修改：无现场参数。必须用 PyBullet 虚拟环境跑。
# 看情况改：公差随仿真版本微调。
# 不要改：测试语义（空白花纹不得运动、宽松解不得 dispatch）。
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
# 功能：每用例独立 PokerSession；setUp 重定向 stdout。
# 可修改：无。
# 看情况改：新回归保持“失败不得驱动电机”。
# 不要改：setUp/tearDown 资源配对；不要开 GUI。
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
        self.session = run_sim.PokerSession(vision, simulator, Camera, self.sim, Path(self.temp.name))

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
        for specs in ([], run_sim.make_scene(simulator)[:2]):
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
        self.sim.spawn_pieces(run_sim.make_scene(simulator))
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
        self.assertEqual(len(valid_tasks), 3)
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

    def test_default_complete_and_pattern_gate_failures(self):
        self.sim.spawn_pieces(run_sim.make_scene(simulator))
        self.assertEqual(self.session.run_once()["status"], "pass")
        plan = json.loads((self.session.last_output / "plan.json").read_text())
        final = json.loads((self.session.last_output / "final_verification.json").read_text())
        self.assertEqual(final["detected_piece_count"], 3)
        self.assertGreaterEqual(final["target_seams"]["minimum_target_gap_mm"], 0.99)
        self.assertEqual(len(self.sim.results), 3)
        for mutation in ("low", "nan", "missing", "aggregate"):
            bad = copy.deepcopy(plan)
            gate = bad["poker_pattern_verification"]
            if mutation == "low":
                gate["pairs"][0].update(ncc=0, similarity=0.5)
                gate["min_similarity"] = 0.5
            elif mutation == "nan":
                gate["pairs"][0]["similarity"] = float("nan")
            elif mutation == "missing":
                gate["pairs"] = []
            else:
                gate["min_similarity"] = 1
            self.assertEqual(run_sim.pattern_gate(bad, self.session.config)["status"], "failed")

    def test_blank_pattern_rejected_before_motion_and_home(self):
        self.sim.spawn_pieces(run_sim.make_scene(simulator))
        extract = vision.edge_matcher.extract_edge_profiles

        def blank_profiles(paper, *args, **kwargs):
            # Exercise the real sampler and NCC using a textureless image.
            return extract(np.full_like(paper, 220), *args, **kwargs)

        with patch.object(vision.edge_matcher, "extract_edge_profiles", side_effect=blank_profiles):
            with patch.object(self.sim, "run_queue") as run:
                with self.assertRaises(simulator.ExecutionError):
                    self.session.run_once()
                run.assert_not_called()
        out = self.session.last_output
        self.assertFalse(json.loads((out / "plan.json").read_text())["ready_for_motion"])
        gate = json.loads((out / "pattern_gate.json").read_text())
        self.assertEqual(gate["status"], "failed")
        self.assertTrue(gate["failure_reasons"])
        self.session.handle_command("start")
        self.assertEqual(self.session.last_output, out)
        self.session.handle_command("home")
        self.assertIsNone(self.sim.fault)
        self.assertEqual(len(self.sim.pieces), 3)

    def test_relaxed_solution_cannot_dispatch(self):
        self.sim.spawn_pieces(run_sim.make_scene(simulator))
        process = vision.process_paper

        def relaxed(*args, **kwargs):
            preview, paths = process(*args, **kwargs)
            paths[0].match.relaxed_override = True
            return preview, paths

        with patch.object(vision, "process_paper", side_effect=relaxed):
            with patch.object(self.sim, "run_queue") as run:
                with self.assertRaises(simulator.ExecutionError):
                    self.session.run_once()
                run.assert_not_called()
        self.assertFalse(json.loads((self.session.last_output / "plan.json").read_text())["ready_for_motion"])

    def test_crossing_polygons_have_negative_clearance(self):
        horizontal = np.array([[-3, -1], [3, -1], [3, 1], [-3, 1]], float)
        vertical = horizontal[:, ::-1].copy()
        split = np.insert(horizontal, 1, [0, -0.999999], axis=0)
        self.assertLess(vision._polygon_closest_vector(split, vertical)[0], 0)
        self.assertLess(vision._polygon_closest_vector(horizontal, vertical)[0], 0)
        self.assertAlmostEqual(vision._polygon_closest_vector(horizontal, horizontal + [8, 0])[0], 2)

    def test_split_edges_and_skipped_profiles_keep_solver_indices(self):
        from types import SimpleNamespace as NS
        split = np.array([[0, 0], [0.5, 0], [10, 0], [10, 10], [0, 10]], float)
        transform = np.array([[1., 0, 20], [0, 1, 30], [0, 0, 1]])
        solution = NS(poses=[NS(piece_index=i, transform_3x3=transform,
                              target_polygon_mm=split + [20, 30]) for i in range(2)],
                      matches=[NS(piece_a=0, edge_a=1, piece_b=1, edge_b=1)])
        profiles = [NS(edge_idx=1, profile_gray=np.array([1, 4, 2, 9, 3], float))]
        with patch.object(vision.edge_matcher, "extract_edge_profiles", return_value=profiles) as sample:
            result = vision.verify_edge_patterns(np.ones((100, 100, 3), np.uint8),
                         [NS(piece_id=1), NS(piece_id=2)], [], solution, self.session.config)
        np.testing.assert_allclose(sample.call_args.args[2], split)
        self.assertAlmostEqual(result["min_similarity"], 1)

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
