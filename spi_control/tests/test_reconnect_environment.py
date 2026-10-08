import errno
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manual_control import DeviceWorker, SerialWire, connection_error, probe
from runtime_support import SingleInstance
from check_environment import inspect_device, inspect_python


class ReconnectTests(unittest.TestCase):
    def worker(self, factory, wait=0.5):
        owner = Mock()
        owner.config = {"grbl_port": "grbl", "baudrate": 115200,
                        "reply_timeout_s": 0.1, "connect_wait_s": wait}
        owner.ui_tick = time.monotonic()
        return DeviceWorker(owner, "grbl", factory)

    def test_device_enumerates_late_before_any_command(self):
        port = Mock()
        factory = Mock(side_effect=[FileNotFoundError(errno.ENOENT, "missing"), port])
        worker = self.worker(factory)
        worker.open_waiting_for_device()
        self.assertEqual(factory.call_count, 2)
        self.assertIs(worker.wire.port, port)
        port.write.assert_not_called()

    def test_busy_and_permission_errors_are_not_retried(self):
        for code in (errno.EBUSY, errno.EAGAIN, errno.EACCES):
            factory = Mock(side_effect=OSError(code, "test"))
            with self.assertRaises(OSError):
                self.worker(factory).open_waiting_for_device()
            factory.assert_called_once()

    def test_missing_port_wait_is_bounded(self):
        factory = Mock(side_effect=FileNotFoundError(errno.ENOENT, "missing"))
        with self.assertRaises(FileNotFoundError):
            self.worker(factory, wait=0).open_waiting_for_device()
        factory.assert_called_once()

    def test_probe_failure_is_reported_to_shell_without_motion(self):
        cfg = {"grbl_port": "missing_grbl", "aux_port": "missing_aux", "baudrate": 115200}
        with patch("manual_control.SerialWire", side_effect=FileNotFoundError(errno.ENOENT, "missing")):
            self.assertEqual(probe(cfg), 1)

    def test_useful_error_categories(self):
        self.assertIn("电源", connection_error(FileNotFoundError(errno.ENOENT, "missing")))
        self.assertIn("占用", connection_error(OSError(errno.EAGAIN, "busy")))
        self.assertIn("权限", connection_error(PermissionError(errno.EACCES, "permission")))


class EnvironmentTests(unittest.TestCase):
    def test_second_screen_is_rejected_and_exit_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.lock"
            with SingleInstance(path):
                with self.assertRaisesRegex(RuntimeError, "已运行"):
                    with SingleInstance(path):
                        self.fail("duplicate screen accepted")
            with SingleInstance(path):
                self.assertTrue(path.exists())

    def test_interpreter_report_detects_missing_dependency(self):
        self.assertTrue(inspect_python(sys.executable, ["json"])["ok"])
        self.assertFalse(inspect_python(sys.executable, ["missing_puzzle_module_xyz"])["ok"])

    def test_device_check_does_not_create_or_open_node(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not_present"
            self.assertFalse(inspect_device(path)["exists"])
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
