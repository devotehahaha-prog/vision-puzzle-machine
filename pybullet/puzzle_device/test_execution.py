"""Headless execution regressions; run with python -m unittest -v test_execution."""

import contextlib
import io
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p

from config import CFG, MM
from protocol import encode_frame, parse_any, parse_frame
from simulator import (
    ExecutionError, MotionTask, PieceSpec, PuzzleGantrySim, angle_difference,
    demo_pieces, theta_pose,
)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.sim = PuzzleGantrySim(gui=False, realtime=False)

    def tearDown(self):
        self.sim.close()
        self.redirect.__exit__(None, None, None)

    def piece(self, yaw=0):
        return self.sim.spawn_pieces([PieceSpec("test", (20, 15), (100, 200), yaw)])[0]

    def test_rotation_and_eccentric_pickup_matrix(self):
        for initial in (0, 15, 90, 179):
            for flip in (0, 45, -45, 180, -180):
                for offset in (0, 7):
                    with self.subTest(initial=initial, flip=flip, offset=offset):
                        body = self.piece(initial)
                        result = self.sim.run_queue([MotionTask(100 + offset, 200, flip, 100, 70)])[0]
                        theta = math.radians(flip)
                        expected = np.array([100 - offset * math.cos(theta), 70 - offset * math.sin(theta)])
                        self.assertLess(np.linalg.norm(np.array(result.actual_pose[:2]) - expected), 2)
                        self.assertLess(abs(angle_difference(result.actual_pose[2], initial + flip)), 2)
                        self.assertEqual(result.body_id, body)

    def test_attachment_does_not_snap(self):
        for initial in (0, 15, 90, 179):
            for flip in (0, 45, -45, 180, -180):
                for offset in (0, 7):
                    with self.subTest(initial=initial, flip=flip, offset=offset):
                        body = self.piece(initial)
                        start, _ = theta_pose(flip)
                        self.sim.set_pose(100 + offset, 200, CFG.z_low_mm, start)
                        before = p.getBasePositionAndOrientation(body)
                        self.sim.set_magnet(True)
                        for _ in range(48):
                            self.sim.step()
                        after = p.getBasePositionAndOrientation(body)
                        self.assertLess(np.linalg.norm(np.array(after[0]) - before[0]) / MM, 1)
                        before_yaw = math.degrees(p.getEulerFromQuaternion(before[1])[2])
                        after_yaw = math.degrees(p.getEulerFromQuaternion(after[1])[2])
                        self.assertLess(abs(angle_difference(after_yaw, before_yaw)), 2)
                        self.sim.set_magnet(False)

    def test_invalid_queue_is_atomic(self):
        self.piece()
        good = MotionTask(100, 200, 0, 100, 70)
        bad_values = (-1, 211, float("nan"), float("inf"), "100")
        for x in bad_values:
            with self.subTest(x=x):
                with patch.object(self.sim, "set_pose") as move:
                    with self.assertRaises(ExecutionError) as caught:
                        self.sim.run_queue([good, MotionTask(x, 200, 0, 100, 70)])
                    self.assertEqual(caught.exception.code, "INVALID_TASK")
                    self.assertEqual(caught.exception.task_index, 2)
                    move.assert_not_called()
                self.sim.home()
        for tasks in ([], [good] * 5, [MotionTask(100, 200, 181, 100, 70)]):
            with self.assertRaises(ExecutionError):
                self.sim.run_queue(tasks)
            self.sim.home()

    def test_empty_pick_latches_and_home_recovers(self):
        self.piece()
        with self.assertRaises(ExecutionError) as caught:
            self.sim.run_queue([MotionTask(20, 20, 0, 30, 30), MotionTask(100, 200, 0, 100, 70)])
        self.assertEqual(caught.exception.code, "MAGNET_MISS")
        self.assertEqual(self.sim.results, [])
        self.assertNotIn("QUEUE DONE", self.output.getvalue())
        with self.assertRaises(ExecutionError) as locked:
            self.sim.run_queue([MotionTask(100, 200, 0, 100, 70)])
        self.assertEqual(locked.exception.code, "FAULT_LOCKED")
        self.sim.home()
        self.assertIsNone(self.sim.fault)
        self.assertEqual(len(self.sim.run_queue([MotionTask(100, 200, 0, 100, 70)])), 1)

    def test_timeout_while_holding_keeps_constraint(self):
        self.piece()
        original = self.sim._wait_joints

        def wait(targets, steps=2400):
            if self.sim.stage == "lift_pick":
                return original(targets, steps=0)
            return original(targets, steps)

        with patch.object(self.sim, "_wait_joints", side_effect=wait):
            with self.assertRaises(ExecutionError) as caught:
                self.sim.run_queue([MotionTask(100, 200, 0, 100, 70)])
        self.assertEqual(caught.exception.code, "MOTION_TIMEOUT")
        self.assertEqual(self.sim.stage, "lift_pick")
        self.assertIsNotNone(self.sim.held_body)
        self.assertTrue(self.sim.magnet_on)
        self.assertEqual(p.getNumConstraints(), 1)
        self.assertNotIn("QUEUE DONE", self.output.getvalue())
        self.sim.home()
        self.assertEqual(p.getNumConstraints(), 0)
        self.assertIsNone(self.sim.fault)

    def test_lost_constraint_stops_queue(self):
        self.piece()
        original = self.sim.step
        removed = False

        def step():
            nonlocal removed
            if self.sim.held_cid is not None and not removed:
                p.removeConstraint(self.sim.held_cid)
                removed = True
            original()

        with patch.object(self.sim, "step", side_effect=step):
            with self.assertRaises(ExecutionError) as caught:
                self.sim.run_queue([MotionTask(100, 200, 0, 100, 70)])
        self.assertEqual(caught.exception.code, "ATTACHMENT_LOST")
        self.assertNotIn("QUEUE DONE", self.output.getvalue())
        self.sim.home()
        self.assertIsNone(self.sim.fault)

    def test_placement_error_is_not_success(self):
        body = self.piece()
        release = self.sim._release

        def faulty_release():
            held = self.sim.held_body
            release()
            if held is not None:
                pos, orn = p.getBasePositionAndOrientation(body)
                p.resetBasePositionAndOrientation(body, [pos[0] + 0.01, pos[1], pos[2]], orn)

        with patch.object(self.sim, "_release", side_effect=faulty_release):
            with self.assertRaises(ExecutionError) as caught:
                self.sim.run_queue([MotionTask(100, 200, 0, 100, 70)])
        self.assertEqual(caught.exception.code, "PLACEMENT_ERROR")
        self.assertNotIn("QUEUE DONE", self.output.getvalue())

    def test_json_and_frame_inputs_execute(self):
        self.piece(90)
        tasks = parse_any('{"tasks": [[100, 200, 45, 100, 70]]}')
        frame = encode_frame(tasks)
        self.assertEqual(len(frame), 45)
        decoded = parse_frame(frame)
        logged = parse_any('[Q2][UART2] sent 45 bytes: ' + frame.hex(' '))
        self.assertEqual(logged, decoded)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(len(parse_frame(frame, skip_zero=False)), 4)
        for source in (tasks, decoded):
            self.sim.home()
            result = self.sim.run_queue(source)[0]
            self.assertLess(abs(angle_difference(result.actual_pose[2], 135)), 2)
        corrupt = bytearray(frame)
        corrupt[-1] ^= 1
        with self.assertRaises(ValueError):
            parse_frame(bytes(corrupt))
        with self.assertRaises(ExecutionError):
            encode_frame(tasks * 5)

    def test_gui_dispatch_locks_start_until_home(self):
        self.piece()
        # Exercise the actual GUI dispatcher without requiring a desktop.
        commands = iter(("start", "start", "home", "start", "quit"))
        plans = [[MotionTask(10, 10, 0, 20, 20)], [MotionTask(100, 200, 0, 100, 70)]]
        planner = unittest.mock.Mock(side_effect=plans)
        self.sim.gui = True
        with patch.object(self.sim, "wait_command", side_effect=lambda: next(commands)):
            self.sim.run_with_buttons(plan_fn=planner)
        self.assertEqual(planner.call_count, 2)
        self.assertIsNone(self.sim.fault)
        self.assertEqual(len(self.sim.results), 1)

    def test_failed_home_does_not_clear_fault(self):
        self.piece()
        with self.assertRaises(ExecutionError):
            self.sim.run_queue([MotionTask(10, 10, 0, 20, 20)])
        error = ExecutionError("MOTION_TIMEOUT", 0, "home")
        with patch.object(self.sim, "set_pose", side_effect=error):
            with self.assertRaises(ExecutionError):
                self.sim.home()
        self.assertIs(self.sim.fault, error)
        self.sim.home()
        self.assertIsNone(self.sim.fault)


class EntryTests(unittest.TestCase):
    def test_vision_cli_success_and_failure_exit_codes(self):
        root = Path(__file__).resolve().parent
        frame = encode_frame([MotionTask(62, 240, 45, 65, 65)]).hex(" ")
        for source, code in ((frame, 0), ('{"tasks": [[-1, 200, 0, 100, 70]]}', 1),
                             ('{"tasks": [[10, 10, 0, 20, 20]]}', 1)):
            with self.subTest(source=source):
                result = subprocess.run([sys.executable, "-B", "run_from_vision.py", source, "--direct"],
                                        cwd=root, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertEqual("VISION QUEUE DONE" in result.stdout, code == 0)

    def test_auto_entry_reuses_layout_without_replacing_assets(self):
        import run_auto
        with patch.object(sys, "argv", ["run_auto.py", "--direct", "--reuse-layout"]), \
                patch.object(run_auto, "save_plan", return_value="existing-plan"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            run_auto.main()
        self.assertIn("AUTO ASSEMBLE DONE", output.getvalue())


class PlanningExecutionTests(unittest.TestCase):
    def test_sample_and_ten_seeded_layouts(self):
        from assemble import solve_assembly

        rng = np.random.default_rng(20260916)
        execution_failures = []
        planning_diagnostics = []
        max_vertex_error = 0.0
        max_plan_error = 0.0
        for case in range(11):
            sim = PuzzleGantrySim(False, False)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    sim.spawn_pieces(demo_pieces())
                    if case:
                        # Staggered lanes accommodate each polygon at every yaw.
                        for i, body in enumerate(sim.pieces):
                            sim._set_piece_xy_yaw(body, (40, 105, 170)[i] + rng.uniform(-2, 2),
                                                 (240, 165, 240)[i] + rng.uniform(-4, 4), rng.uniform(-math.pi, math.pi))
                    layout = sim.current_layout()
                    try:
                        plan = solve_assembly(layout)
                    except RuntimeError as error:
                        planning_diagnostics.append((case, str(error)))
                        continue
                    expected = {}
                    for placed, task in zip(plan.placed, plan.tasks):
                        source = next(item for item in layout["pieces"] if item["name"] == placed.piece.name)
                        angle = math.radians(task.flip_deg)
                        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
                        vertices = (np.array(source["vertices_mm"]) - [task.pick_x_mm, task.pick_y_mm]) @ rot.T
                        vertices += [task.place_x_mm, task.place_y_mm]
                        expected[placed.piece.name] = vertices
                        target = np.array(plan.target_vertices[placed.piece.name])
                        distances = np.linalg.norm(vertices[:, None, :] - target[None, :, :], axis=2)
                        error = max(distances.min(axis=0).max(), distances.min(axis=1).max())
                        max_plan_error = max(max_plan_error, error)
                        if error > 2:
                            planning_diagnostics.append((case, placed.piece.name, round(float(error), 3)))
                    try:
                        results = sim.run_queue(plan.tasks)
                        for result in results:
                            actual = sim.world_vertices_mm(result.body_id)
                            error = np.linalg.norm(actual - expected[result.piece_name], axis=1).max()
                            max_vertex_error = max(max_vertex_error, error)
                            if error > 2:
                                execution_failures.append((case, result.piece_name, float(error)))
                    except ExecutionError as error:
                        execution_failures.append((case, str(error)))
            finally:
                sim.close()
        print(f"LAYOUT QA: max_vertex_error_mm={max_vertex_error:.6f}, max_plan_error_mm={max_plan_error:.6f}")
        print(f"PLANNING DIAGNOSTICS: {planning_diagnostics}")
        self.assertEqual(execution_failures, [])
        self.assertFalse(any(len(item) == 2 for item in planning_diagnostics), planning_diagnostics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
