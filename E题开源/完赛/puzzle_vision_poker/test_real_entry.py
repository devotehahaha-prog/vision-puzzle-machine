"""Safety tests for the real-camera entry point; these tests never open a port."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_real


class RealEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = Path(self.temp.name) / "unsafe.json"
        self.config.write_text(json.dumps({
            "serial_port": "/dev/stm32",
            "auto_fallback_enabled": True,
            "force_execute_on_any_solution": True,
            "generic_ultimate_fallback_enable": True,
        }), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_default_entry_is_dry_run_even_with_port_argument(self):
        captured = {}

        def fake_main():
            captured["argv"] = sys.argv[1:]
            captured["config"] = run_real.vision.load_config()
            return 0

        with patch.object(run_real.vision, "main", side_effect=fake_main):
            self.assertEqual(run_real.main([
                "--config", str(self.config), "--auto", "--port", "/dev/ttyUSB0",
            ]), 0)
        self.assertIn("--dry-run", captured["argv"])
        self.assertEqual(captured["config"]["serial_port"], "")
        self.assertFalse(captured["config"]["auto_fallback_enabled"])
        self.assertFalse(captured["config"]["force_execute_on_any_solution"])
        self.assertFalse(captured["config"]["generic_ultimate_fallback_enable"])
        self.assertEqual(captured["config"]["generic_min_edge_length_mm"],
                         captured["config"]["poker_min_edge_length_mm"])

    def test_execute_requires_explicit_port(self):
        with patch.object(run_real.vision, "main") as inner:
            with self.assertRaises(SystemExit) as caught:
                run_real.main(["--config", str(self.config), "--execute", "--auto"])
        self.assertEqual(caught.exception.code, 2)
        inner.assert_not_called()

    def test_preview_loader_matches_effective_planner_configuration(self):
        captured = {}
        with patch.object(run_real.vision, 'main', side_effect=lambda:
                          captured.update(config=run_real.vision.load_config()) or 0):
            run_real.main(['--config',str(self.config),'--auto','--dry-run'])
        self.assertEqual(run_real.load_production_config(self.config),captured['config'])

    def test_old_execute_entry_is_rejected_even_with_port(self):
        with patch.object(run_real.vision, "main") as inner:
            with self.assertRaises(SystemExit):
                run_real.main(["--config", str(self.config), "--execute", "--auto",
                               "--port", "/dev/ttyUSB0"])
        inner.assert_not_called()

    def test_real_entry_does_not_import_pybullet(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import run_real, sys; assert 'pybullet' not in sys.modules"],
            cwd=run_real.ROOT, capture_output=True, text=True, check=False, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ack_timeout_sends_stop_then_disables_motion(self):
        class FakeSerial:
            def __init__(self, **kwargs):
                self.writes = []
                self.closed = False

            def write(self, payload):
                self.writes.append(payload)

            def flush(self):
                pass

            def readline(self):
                return b""

            def close(self):
                self.closed = True

        from unittest.mock import patch
        serial = run_real.vision.serial_transport
        connection = FakeSerial()
        with patch.object(serial.time, "sleep"):
            with self.assertRaisesRegex(serial.SerialPlanError, "超时"):
                serial.send_serial_commands(
                    ["MOTION,1"], "/dev/mock", serial_factory=lambda **kwargs: connection,
                    output=lambda _: None,
                )
        self.assertEqual(connection.writes[-2:], [b"STOP\n", b"MOTION,0\n"])
        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
