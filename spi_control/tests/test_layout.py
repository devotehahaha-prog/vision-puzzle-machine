#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from layout import (
    CALIB_BTN,
    LIVE_BTN,
    MODE_ORDINARY,
    MODE_POKER,
    ORDINARY_TAB,
    PAGE_CALIB,
    PAGE_LIVE,
    PAGE_TUNE,
    POKER_TAB,
    PREVIEW_RECT,
    PRIMARY,
    SAVE_CAL_BTN,
    SCR,
    START_BTN,
    STOP_BTN,
    TAB_H,
    TUNE_BTN,
    UNDO_BTN,
    StartPageState,
    hit,
    letterbox,
    map_preview_to_camera,
    slider_row,
)
import start_page


class LayoutTests(unittest.TestCase):
    def test_home_buttons(self):
        self.assertTrue(hit(400, 120, START_BTN))
        self.assertTrue(hit(80, 230, CALIB_BTN))
        from layout import LIVE_BTN, PAGE_LIVE
        self.assertTrue(hit(450, 220, TUNE_BTN))
        self.assertTrue(hit(600, 220, LIVE_BTN))
        self.assertFalse(hit(80, 230, TUNE_BTN))
        self.assertEqual(TAB_H, 70)
        self.assertTrue(hit(200, 0, ORDINARY_TAB))
        self.assertTrue(hit(500, 10, POKER_TAB))

    def test_tabs_and_tools(self):
        state = StartPageState()
        self.assertEqual(state.on_points([(600, 20)]), "poker")
        self.assertTrue(state.set_mode(MODE_POKER))
        self.assertIsNone(state.on_points([(600, 20)]))
        self.assertIsNone(state.on_points([]))
        self.assertEqual(state.on_points([(80, 230)]), "calib")
        self.assertTrue(state.open_calib())
        self.assertEqual(state.page, PAGE_CALIB)
        self.assertEqual(state.calib_hint(), "点第1角：左上")
        state.go_home()
        self.assertIsNone(state.on_points([]))
        lx, ly, lw, lh = LIVE_BTN
        self.assertEqual(state.on_points([(lx + lw // 2, ly + lh // 2)]), "live")
        self.assertTrue(state.open_live())
        self.assertEqual(state.page, PAGE_LIVE)

    def test_preview_mapping_and_sliders(self):
        box = letterbox(1280, 720, PREVIEW_RECT[2], PREVIEW_RECT[3])
        mapped = map_preview_to_camera(
            PREVIEW_RECT[0] + box[0] + box[2] // 2,
            PREVIEW_RECT[1] + box[1] + box[3] // 2,
            PREVIEW_RECT, box, (1280, 720),
        )
        self.assertIsNotNone(mapped)
        self.assertAlmostEqual(mapped[0], 640, delta=4)
        self.assertAlmostEqual(mapped[1], 360, delta=4)
        self.assertIsNone(map_preview_to_camera(PREVIEW_RECT[0] + 1, PREVIEW_RECT[1] + 1,
                                                PREVIEW_RECT, box, (1280, 720)))
        from layout import TUNE_PREVIEW
        self.assertFalse(hit(TUNE_PREVIEW[0] + 10, TUNE_PREVIEW[1] + 10, slider_row(0)["track"]))

        state = StartPageState()
        state.open_tune({"white_piece_saturation_max": 40})
        self.assertEqual(state.page, PAGE_TUNE)
        self.assertEqual(state.sliders[0][4], 40)
        row = slider_row(0)
        tx, ty, tw, th = row["track"]
        action = state.on_points([(tx + tw - 1, ty + th // 2)])
        self.assertEqual(action[0], "slider")
        self.assertTrue(state.set_slider(action[1], action[2]))
        self.assertEqual(state.sliders[0][4], 255)
        self.assertEqual(state.on_points([(tx, ty)])[0], "slider")
        self.assertIsNone(state.on_points([]))
        minus = row["minus"]
        action = state.on_points([(minus[0] + 2, minus[1] + 2)])
        self.assertEqual(action[0], "slider")
        state.set_slider(action[1], action[2])
        self.assertEqual(state.sliders[0][4], 254)
        params = state.slider_params()
        self.assertEqual(params["white_piece_saturation_max"], 254)

        poker = StartPageState()
        poker.mode = MODE_POKER
        poker.open_tune({"poker_morph_close_mm": 2.5})
        self.assertEqual(poker.slider_params()["poker_morph_close_mm"], 2.5)
        poker.set_slider(3, 13)
        self.assertEqual(poker.slider_params()["poker_morph_close_mm"], 1.3)
        from layout import GROUP_BOARD, GROUP_PIECE, PAGE_NEXT_BTN
        state = StartPageState()
        state.open_tune({"white_piece_saturation_max": 40, "auto_board_saturation_max": 90})
        self.assertTrue(state.shift_tune_page(1))
        self.assertEqual(state.sliders[0][0], "morphology_kernel_mm_x10")
        self.assertTrue(state.set_tune_group(GROUP_BOARD, {"auto_board_saturation_max": 90}))
        self.assertEqual(state.tune_group, GROUP_BOARD)
        self.assertIn("auto_board_saturation_max", state.slider_params())
        self.assertEqual(state.on_points([(PAGE_NEXT_BTN[0] + 2, PAGE_NEXT_BTN[1] + 2)]), "page_next")

    def test_busy_ignores_new_tools(self):
        state = StartPageState()
        self.assertEqual(state.on_points([(400, 120)]), "start")
        state.begin_run()
        self.assertIsNone(state.on_points([]))
        self.assertIsNone(state.on_points([(80, 230)]))
        self.assertIsNone(state.on_points([(500, 230)]))

    def test_busy_stop_remains_available(self):
        state = StartPageState()
        state.begin_run()
        state.on_points([])
        self.assertEqual(state.on_points([(STOP_BTN[0] + 5, STOP_BTN[1] + 5)]), "stop")
        self.assertFalse(state.open_calib())
        state.finish_run(True)
        self.assertIsNone(state.on_points([(80, 230)]))
        self.assertIsNone(state.on_points([]))
        self.assertEqual(state.on_points([(80, 230)]), "calib")

    def test_paths_and_palette(self):
        self.assertEqual(PRIMARY, (0x21, 0x96, 0xF3))
        self.assertEqual(SCR, (0x15, 0x17, 0x1A))
        projects = start_page.find_projects()
        self.assertTrue((projects["ordinary"] / "run_sim.py").is_file())
        self.assertTrue((projects["poker"] / "config.json").is_file())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interpreter = root / ".venv" / "bin" / "python"
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            python = start_page.find_venv_python(root)
            self.assertEqual(python, interpreter)
            self.assertEqual(start_page.sim_command(python)[-2:], ["run_sim.py", "--auto-start"])

    def test_save_config_updates_sim_config_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sim_config.json").write_text(
                json.dumps({"keep": 1, "white_piece_saturation_max": 10}), encoding="utf-8")
            run_sim = type("R", (), {"ROOT": root})()
            sys.modules.pop("camera_job", None)
            import camera_job
            (root / "config.json").write_text(json.dumps({"keep": 2, "white_piece_saturation_max": 10}), encoding="utf-8")
            args = type("A", (), {
                "project": root,
                "params": json.dumps({"white_piece_saturation_max": 77}),
                "real_config": root / "config.json",
            })()
            with patch.object(camera_job, "load_run_sim", return_value=run_sim):
                camera_job.cmd_save_config(args)
            saved = json.loads((root / "sim_config.json").read_text(encoding="utf-8"))
            real = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["keep"], 1)
            self.assertEqual(saved["white_piece_saturation_max"], 77)
            self.assertEqual(real["keep"], 2)
            self.assertEqual(real["white_piece_saturation_max"], 77)

    def test_control_rectangles_do_not_overlap_status(self):
        from layout import CALIB_STATUS, MODE_BTN, TUNE_PREVIEW, TUNE_STATUS
        def overlap(a, b):
            return a[0] < b[0]+b[2] and b[0] < a[0]+a[2] and a[1] < b[1]+b[3] and b[1] < a[1]+a[3]
        for rect in [UNDO_BTN, SAVE_CAL_BTN]:
            self.assertFalse(overlap(rect, CALIB_STATUS))
            self.assertLessEqual(rect[1]+rect[3], 480)
        for rect in [MODE_BTN] + [r for i in range(4) for r in slider_row(i).values()]:
            self.assertFalse(overlap(rect, TUNE_STATUS))
            self.assertFalse(overlap(rect, TUNE_PREVIEW))
            self.assertLessEqual(rect[1]+rect[3], 480)

    def test_drag_and_no_new_report(self):
        state = StartPageState()
        state.open_tune({})
        track = slider_row(0)["track"]
        self.assertEqual(state.on_points([(track[0], track[1])])[2], 0)
        self.assertIsNone(state.on_points(None))
        self.assertEqual(state.on_points([(track[0]+track[2], track[1])])[2], 255)
        state.on_points([])
        self.assertIsNone(state.drag_slider)
        state.threshold_mode = "fixed"
        self.assertEqual(state.slider_params()["white_piece_threshold_mode"], "fixed")
        state.set_slider(1, 255)
        self.assertLessEqual(state.sliders[1][4], state.sliders[2][4])

    def test_touch_no_report_vs_release(self):
        from unittest.mock import Mock
        touch = Mock()
        touch._read_reg.return_value = bytes([0])
        self.assertIsNone(start_page.read_touch_report(touch))
        touch._write_reg.assert_not_called()
        touch._read_reg.return_value = bytes([128])
        self.assertEqual(start_page.read_touch_report(touch), [])
        touch._write_reg.assert_called_once()

    def test_simulation_job_does_not_wait_on_launch(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp, patch.object(start_page, "HERE", Path(tmp)), patch.object(start_page.subprocess, "Popen") as launch:
            proc = launch.return_value
            proc.poll.return_value = None
            job = start_page.SimulationJob(Path("python"), Path(tmp))
            proc.wait.assert_not_called()
            self.assertIsNone(job.poll())
            proc.poll.return_value = 1
            self.assertEqual(job.poll(), 1)
            job.close()

    def test_preview_exit_and_stale_heartbeat(self):
        import time
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            camera = object.__new__(start_page.CameraSession)
            camera.status_path = Path(tmp)/"status.json"
            camera.proc = Mock()
            camera.proc.poll.return_value = None
            camera.status_path.write_text(json.dumps({"ok": True, "heartbeat": time.time()-20}))
            self.assertFalse(camera.read_status()["ok"])
            camera.proc.poll.return_value = 0
            camera.status_path.write_text(json.dumps({"ok": True, "heartbeat": time.time()}))
            self.assertFalse(camera.read_status()["ok"])

    def test_find_sim_root_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            contest = Path(tmp)
            sim = contest / "pybullet" / "puzzle_device"
            sim.mkdir(parents=True)
            (sim / "simulator.py").write_text("#", encoding="utf-8")
            project = contest / "E题开源" / "完赛" / "puzzle_vision"
            project.mkdir(parents=True)
            run_sim = type("R", (), {
                "ROOT": project,
                "DEFAULT_SIM_ROOT": project.parents[2] / "pybullet" / "puzzle_device",
            })()
            sys.modules.pop("camera_job", None)
            import camera_job
            self.assertEqual(camera_job.find_sim_root(run_sim), sim)


if __name__ == "__main__":
    unittest.main()
