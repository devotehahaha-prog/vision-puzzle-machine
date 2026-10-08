"""Real ordinary-mode routing, configuration isolation, and preview-only results."""
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ordinary_camera as ordinary
import start_page as ui
import camera_job
from layout import StartPageState, MODE_ORDINARY


class OrdinaryCameraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / "puzzle_vision"
        self.project.mkdir()
        self.engine = self.root / "puzzle_vision_poker"
        self.engine.mkdir()
        self.config = {"camera_device": "shared-camera", "camera_id": 20,
                       "camera_calibration": {"perspective_matrix": [[1,0,0],[0,1,0],[0,0,1]]},
                       "camera_calibration_image_size": [1280,720],
                       "generic_target_piece_clearance_mm": 6.0,
                       "target_corresponding_vertex_limit_mm": 20.0,
                       "pixels_per_mm": 4.0, "serial_port": "must-not-open"}
        ordinary.atomic_json(self.engine / "config.json", {})
        ordinary.atomic_json(self.engine / "production_config.json", self.config)
        ordinary.atomic_json(self.project / "real_config.json", {
            "camera_device": "obsolete", "expected_piece_count": 4,
            "generic_min_edge_length_mm": 7.0, "white_piece_saturation_max": 55,
            "generic_target_piece_clearance_mm": 10.0, "target_corresponding_vertex_limit_mm": 24.0,
            "generic_ultimate_fallback_enable": True, "auto_board_calibration": True})

    def tearDown(self):
        self.temp.cleanup()

    def test_shared_calibration_and_limits_are_read_only(self):
        protected = self.engine / "production_config.json"
        before = protected.read_bytes()
        config = ordinary.load_config(self.project)
        self.assertEqual(config["camera_device"], "shared-camera")
        self.assertEqual(config["camera_calibration"], self.config["camera_calibration"])
        self.assertEqual(config["generic_target_piece_clearance_mm"], 6.0)
        self.assertEqual(config["target_corresponding_vertex_limit_mm"], 20.0)
        self.assertEqual(config["generic_min_edge_length_mm"], 7.0)
        self.assertEqual(config["expected_piece_count"], 4)
        self.assertEqual(config["segmentation_mode"], "white_piece")
        self.assertEqual((config["source_region"], config["target_region"]), ("bottom", "top"))
        self.assertEqual(config["serial_port"], "")
        for key in ("poker_pattern_verification", "enforce_pattern_gate", "approval_workflow",
                    "auto_board_calibration", "generic_ultimate_fallback_enable", "auto_fallback_enabled"):
            self.assertFalse(config[key], key)
        config["camera_calibration"]["modified"] = True
        self.assertEqual(protected.read_bytes(), before)

    def test_missing_shared_calibration_does_not_use_legacy_calibration(self):
        ordinary.atomic_json(self.engine / "production_config.json", {})
        with self.assertRaisesRegex(ValueError, "共用相机标定"):
            ordinary.load_config(self.project)

    def test_ordinary_tuning_writes_only_ordinary_real_settings(self):
        protected = self.engine / "production_config.json"
        before = protected.read_bytes()
        args = SimpleNamespace(project=self.project, vision="ordinary",
                               params=json.dumps({"white_piece_saturation_max": 68}))
        with patch.object(camera_job, "load_run_sim", side_effect=AssertionError("simulation imported")):
            camera_job.cmd_save_config(args)
        config = ordinary.read_json(self.project / "real_config.json")
        self.assertEqual(config["white_piece_saturation_max"], 68)
        self.assertEqual(config["generic_target_piece_clearance_mm"], 10.0)
        self.assertFalse((self.project / "sim_config.json").exists())
        self.assertEqual(protected.read_bytes(), before)

    def test_ui_launches_real_ordinary_camera_and_plan_worker(self):
        with patch.object(ui, "HERE", self.root), patch.object(ui.subprocess, "Popen") as popen:
            popen.return_value.wait.return_value = 0
            session = ui.CameraSession(Path("python"), self.project, "calib_preview")
            command = popen.call_args.args[0]
            self.assertIn("--real-camera", command)
            self.assertEqual(command[command.index("--vision") + 1], "ordinary")
            session.stop()
            job = ui.OrdinaryCameraJob(Path("python"), self.project)
            command = popen.call_args.args[0]
            self.assertEqual(Path(command[5]).name, "ordinary_camera.py")
            self.assertNotIn("--execute", command)
            self.assertNotIn("run_sim.py", command)
            job.log.close()

    def test_current_preview_cannot_confirm_execution(self):
        image = self.root / "assembly.png"
        image.write_bytes(b"test")
        ordinary.atomic_json(ordinary.output_dir(self.project) / "plan.json", {
            "mode": "ordinary_camera", "preview_ready": True, "approval_state": "preview_only",
            "ready_for_motion": False, "piece_count": 4, "generated_at_epoch": time.time(),
            "assembly_path": str(image), "plan_id": "ordinary-test"})
        state = StartPageState(default_mode=MODE_ORDINARY)
        ui.finish_ordinary_plan(state, self.project, 0, 0)
        self.assertEqual(state.workflow_stage, "ordinary_preview")
        self.assertFalse(state.plan_ready)
        state.armed = True
        self.assertNotEqual(state.on_points([(450,350)]), "workflow_confirm")
        ui.finish_ordinary_plan(state, self.project, time.time() + 1, 0)
        self.assertEqual(state.workflow_stage, "plan_failed")
        self.assertIsNone(state.review_image_path)

    def test_open_failure_invalidates_old_plan_without_touching_poker_output(self):
        output = ordinary.output_dir(self.project)
        ordinary.atomic_json(output / "plan.json", {"preview_ready": True, "ready_for_motion": True})
        (output / "detection_preview.jpg").write_bytes(b"stale")
        poker_output = self.engine / "output" / "plan.json"
        ordinary.atomic_json(poker_output, {"untouched": True})
        vision = SimpleNamespace(OUTPUT_DIR="previous", open_camera=Mock(side_effect=RuntimeError("camera missing")))
        with patch.object(ordinary, "load_runtime", return_value=(vision, self.config)):
            with self.assertRaisesRegex(RuntimeError, "camera missing"):
                ordinary.generate_plan(self.project)
        plan = ordinary.read_json(output / "plan.json")
        self.assertFalse(plan["preview_ready"])
        self.assertFalse(plan["ready_for_motion"])
        self.assertFalse((output / "detection_preview.jpg").exists())
        self.assertEqual(vision.OUTPUT_DIR, "previous")
        self.assertEqual(ordinary.read_json(poker_output), {"untouched": True})

    def test_camera_preview_uses_ordinary_runtime_before_loading_legacy_main(self):
        (self.root / "stop").touch()
        capture = Mock()
        vision = SimpleNamespace(open_camera=Mock(return_value=capture))
        args = SimpleNamespace(workdir=self.root, project=self.project, vision="ordinary", mode="live")
        with patch.object(ordinary, "load_runtime", return_value=(vision, self.config)), \
             patch.object(camera_job, "load_vision", side_effect=AssertionError("legacy main imported")):
            camera_job.cmd_real_preview(args)
        vision.open_camera.assert_called_once_with(self.config, None)
        capture.release.assert_called_once()

    def test_config_failure_clears_stale_live_image(self):
        (self.root / "stop").touch()
        (self.root / "spi_preview.jpg").write_bytes(b"stale")
        args = SimpleNamespace(workdir=self.root, project=self.project, vision="ordinary", mode="live")
        with patch.object(ordinary, "load_runtime", side_effect=ValueError("invalid calibration")):
            camera_job.cmd_real_preview(args)
        self.assertFalse((self.root / "spi_preview.jpg").exists())
        self.assertFalse(ordinary.read_json(self.root / "status.json")["ok"])

    def test_search_order_preserves_acceptance_limits_and_restores_engine(self):
        config = ordinary.load_config(self.project)
        config.update(generic_enable_closure_refine=True, generic_max_fill_error_ratio=.1,
                      generic_motion_max_fill_error_ratio=.07, generic_max_boundary_gap_ratio=.4,
                      generic_motion_max_boundary_gap_ratio=.32, generic_max_overlap_ratio=.015,
                      generic_motion_max_overlap_ratio=.04)
        before = json.dumps(config, sort_keys=True)
        direct = ordinary.direct_search_config(config)
        self.assertFalse(direct["generic_enable_closure_refine"])
        self.assertEqual(direct["generic_excellent_fill_error_ratio"], .07)
        self.assertEqual(direct["generic_excellent_boundary_gap_ratio"], .32)
        self.assertEqual(direct["generic_excellent_overlap_ratio"], .015)
        for key, value in config.items():
            if key != "generic_enable_closure_refine" and not key.startswith("generic_excellent_"):
                self.assertEqual(direct[key], value, key)
        method = Mock(return_value="direct result")
        vision = SimpleNamespace(_auto_detect_and_solve=method)
        with ordinary.ordinary_search_order(vision):
            self.assertEqual(vision._auto_detect_and_solve("frame", config, "matrix"), "direct result")
        self.assertIs(vision._auto_detect_and_solve, method)
        self.assertEqual(json.dumps(config, sort_keys=True), before)

    def test_direct_search_failure_retains_original_closure_search(self):
        method = Mock(side_effect=[ValueError("no direct solution"), "closure result"])
        vision = SimpleNamespace(_auto_detect_and_solve=method)
        config = {"strict_production":True, "generic_enable_closure_refine":True}
        with ordinary.ordinary_search_order(vision):
            self.assertEqual(vision._auto_detect_and_solve("frame", config, "matrix"), "closure result")
        self.assertIs(method.call_args_list[1].args[1], config)
        self.assertIs(vision._auto_detect_and_solve, method)

    def test_both_search_failures_are_not_converted_to_success(self):
        method = Mock(side_effect=[ValueError("direct failed"), RuntimeError("closure failed")])
        vision = SimpleNamespace(_auto_detect_and_solve=method)
        with self.assertRaisesRegex(RuntimeError, "closure failed"):
            with ordinary.ordinary_search_order(vision):
                vision._auto_detect_and_solve("frame", {}, "matrix")
        self.assertIs(vision._auto_detect_and_solve, method)


if __name__ == "__main__":
    unittest.main()
