import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manual_control import (SerialWire, ProtocolError, Cancelled, ManualController,
                            jog_command, rotate_command, parse_aux, parse_grbl, parse_motion)
from manual_page import ManualPageState, BUTTONS
from layout import StartPageState, MANUAL_BTN


class BufferedPaintTests(unittest.TestCase):
    def test_page_uses_one_spi_frame_and_no_per_widget_transfers(self):
        from PIL import ImageFont
        from unittest.mock import Mock
        from manual_page import paint, _BufferedDraw
        from start_page import paint_button, theme_colors
        font_path = next((p for p in (Path("C:/Windows/Fonts/msyh.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")) if p.exists()), None)
        if font_path is None:
            self.skipTest("CJK font unavailable")
        class Text:
            def _font(self, value, size):
                return ImageFont.truetype(str(font_path), size)
            def measure(self, value, size):
                x0, y0, x1, y1 = self._font(value, size).getbbox(value)
                return x1-x0, y1-y0
        hardware = Mock()
        devices = {"grbl": {"connected": True, "pending": False, "machine": "Hold:0", "message": "已连接"},
                   "aux": {"connected": True, "pending": False, "armed": False, "speed": 0, "magnet": False, "message": "已连接",
                           "motion_enabled": False, "motion_busy": False, "zeroed": False, "r_x100": 0}}
        paint(hardware, Text(), theme_colors(lambda *c: c), ManualPageState(), devices, paint_button, buffered=True)
        hardware.image.assert_called_once()
        hardware.clear.assert_not_called()
        hardware.fill_rect.assert_not_called()
        self.assertEqual(hardware.image.call_args.args[0].size, (800, 480))
        frame = _BufferedDraw(hardware)
        frame.order = "BGR"
        frame.fill_rect(10, 10, 2, 2, bytes((30, 20, 10)))
        self.assertEqual(frame.image.getpixel((10, 10)), (10, 20, 30))


class FakePort:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.kind = kwargs["port"]
        self.data = bytearray()
        self.writes = []
        self.closed = False
        self.armed = self.magnet = 0
        self.speed = 0
        self.machine = "Idle"
        self.drop = False
        self.jog_polls = 0
        self.motion_enabled = False
        self.zeroed = False
        self.r_x100 = 0
        self.rotation = None
        self.rotation_delay = 0.45
        self.wrong_rotate_ack = False
        self.omit_rotate_ack = False
        self.finish_before_status = False

    def complete_rotation(self):
        if self.rotation is None or time.monotonic() < self.rotation[0]:
            return
        _deadline, angle = self.rotation
        self.r_x100 += angle
        self.rotation = None
        if not self.omit_rotate_ack:
            self.data.extend(f"ACK,ROTATE,{angle + int(self.wrong_rotate_ack)}\n".encode())

    def write(self, data):
        self.writes.append(data)
        if self.drop:
            return len(data)
        for cmd in data.decode().splitlines():
            if self.kind == "grbl":
                if cmd == "$I":
                    if self.machine == "Idle": self.data.extend(b"[VER:1.3a:]\r\nok\r\n")
                elif cmd == "!": self.machine = "Hold:0"
                elif cmd == "~": self.machine = "Idle"
                elif cmd == "?":
                    state = "Jog" if self.jog_polls else self.machine
                    if self.jog_polls: self.jog_polls -= 1
                    self.data.extend(f"<{state}|MPos:1,2,3|FS:0,0>\r\n".encode())
                else:
                    if cmd.startswith("$J="): self.jog_polls = 3
                    self.data.extend(b"ok\r\n")
            else:
                if cmd == "STATUS":
                    if self.finish_before_status and self.rotation is not None:
                        self.rotation = (0, self.rotation[1])
                    self.complete_rotation()
                    mode = "REAL" if self.motion_enabled else "SIM"
                    busy = "BUSY" if self.rotation else "IDLE"
                    zero = "ZEROED" if self.zeroed else "UNHOMED"
                    self.data.extend(f"ACK,STATUS,{mode},{busy},{zero},0,0,0,{self.r_x100}\n".encode())
                    continue
                if cmd == "IOSTATUS":
                    self.data.extend(f"ACK,IOSTATUS,AUX,{self.armed},SERVO,{self.speed},MAGNET,{self.magnet}\n".encode())
                    continue
                if cmd == "AUX,1": self.armed = 1
                if cmd == "AUX,0": self.armed = self.magnet = self.speed = 0
                if cmd == "MAGNET,1": self.magnet = 1
                if cmd == "MAGNET,0": self.magnet = 0
                if cmd == "SERVO,STOP": self.speed = 0
                if cmd.startswith("SERVO,JOG,"): self.speed = int(cmd.split(",")[2])
                if cmd == "MOTION,1":
                    self.motion_enabled = True
                    self.zeroed = False
                    self.r_x100 = 0
                if cmd == "ZERO":
                    self.zeroed = True
                    self.r_x100 = 0
                if cmd == "STOP":
                    self.rotation = None
                    self.motion_enabled = self.zeroed = False
                    self.armed = self.magnet = self.speed = self.r_x100 = 0
                if cmd.startswith("ROTATE,"):
                    angle = round(float(cmd.split(",")[1]) * 100)
                    if self.motion_enabled:
                        self.rotation = (time.monotonic()+self.rotation_delay, angle)
                        continue
                    self.r_x100 += angle
                    self.data.extend(f"ACK,ROTATE,{angle}\n".encode())
                    continue
                self.data.extend(("ACK,"+cmd+"\n").encode())
        return len(data)

    def read(self, _n):
        self.complete_rotation()
        # Deliberately split every line/ACK into unrelated read boundaries.
        if not self.data:
            time.sleep(0.003)
        result = bytes(self.data[:3])
        del self.data[:3]
        return result

    def close(self):
        self.closed = True


class FramingTests(unittest.TestCase):
    def test_esp_boot_interrupts_wait_even_if_status_follows(self):
        for banner in (b'ets Jul 29 2019 12:21:46\n', b'rst:0x1 (POWERON_RESET)\n', b'ESP-ROM:esp32\n'):
            wire = SerialWire('grbl', 115200, lambda: False, FakePort)
            wire.port.data.extend(banner + b'<Idle|MPos:0,0,0>\n')
            with self.assertRaisesRegex(ProtocolError, '控制器重启'):
                wire.wait(lambda s: s.startswith('<'), .2)

    def test_io_error_does_not_retry_and_contains_device_evidence(self):
        from unittest.mock import patch
        wire = SerialWire('grbl', 115200, lambda: False, FakePort)
        with patch.object(wire.port, 'write', side_effect=OSError(5, 'Input/output error')) as write:
            with self.assertRaisesRegex(ProtocolError, '设备证据'):
                wire.write(b'$J=G91 X1 F60\n')
            write.assert_called_once()

    def test_fragmented_ack_and_status(self):
        wire = SerialWire("grbl", 115200, lambda: False, FakePort)
        wire.port.data.extend(b"<Idle|MPos:0,0,0>\nok\n")
        observed = []
        self.assertEqual(wire.wait(lambda s: s == "ok", 0.2, observed.append), "ok")
        self.assertEqual(observed[0], "<Idle|MPos:0,0,0>")
        self.assertTrue(wire.port.kwargs["exclusive"])

    def test_errors_and_cancellation(self):
        for response in (b"error:2\n", b"ERR,ARM\n", b"ALARM:1\n", b"BOOT,MOTION,READY\n"):
            wire = SerialWire("aux", 115200, lambda: False, FakePort)
            wire.port.data.extend(response)
            with self.assertRaises(ProtocolError): wire.wait(lambda s: s == "ok", 0.1)
        wire = SerialWire("aux", 115200, lambda: True, FakePort)
        with self.assertRaises(Cancelled): wire.wait(lambda s: True, 0.1)

    def test_timeout_does_not_retry_motion(self):
        wire = SerialWire("grbl", 115200, lambda: False, FakePort)
        wire.port.drop = True
        with self.assertRaises(TimeoutError): wire.request("$J=G91 X1 F60", lambda s: s == "ok", 0.02)
        self.assertEqual(len(wire.port.writes), 1)

    def test_units_and_limits(self):
        self.assertEqual(jog_command("Z", -0.1, 30), "$J=G91 G21 Z-0.1 F30")
        self.assertEqual(rotate_command(-45), "ROTATE,-45.00")
        for axis, delta, feed in [("XY", 1, 30), ("Z", float("nan"), 30), ("X", 100, 30), ("X", 1, float("inf"))]:
            with self.assertRaises(ValueError): jog_command(axis, delta, feed)
        for angle in (0, 180, 360, float("nan"), float("inf"), True, "5", 0.01):
            with self.assertRaises(ValueError): rotate_command(angle)

    def test_state_is_not_invented(self):
        m = parse_grbl("<Idle|MPos:5,6,7|WCO:1,2,3|Pn:XY>")
        self.assertEqual(m["WPos"], (4, 4, 4))
        self.assertNotIn("WPos", parse_grbl("<Idle|MPos:5,6,7>"))
        self.assertTrue(parse_aux("ACK,IOSTATUS,AUX,1,SERVO,-10,MAGNET,1")["magnet"])
        with self.assertRaises(ProtocolError): parse_aux("ACK,PING")
        motion = parse_motion("ACK,STATUS,REAL,BUSY,ZEROED,0,0,0,-4500")
        self.assertTrue(motion["motion_busy"])
        self.assertEqual(motion["r_x100"], -4500)
        for line in ("ACK,PING", "ACK,STATUS,REAL,BUSY,ZEROED,0,0,0,NaN",
                     "ACK,STATUS,REAL,LOST,ZEROED,0,0,0,0"):
            with self.assertRaises(ProtocolError): parse_motion(line)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.ports = {}
        def factory(**kwargs):
            port = FakePort(**kwargs)
            self.ports[kwargs["port"]] = port
            return port
        self.controller = ManualController({"grbl_port":"grbl", "aux_port":"aux", "baudrate":115200,
            "reply_timeout_s":0.15, "grbl_feeds":[60,300,600], "z_feeds":[30,60,120]}, factory)

    def tearDown(self):
        self.controller.close()

    def wait_for(self, predicate, timeout=3):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            self.controller.tick()
            if predicate(): return
            time.sleep(0.01)
        self.fail("condition timed out: "+repr(self.controller.snapshot()))

    def ready(self):
        self.wait_for(lambda: all(w.ready for w in self.controller.workers.values()))

    def test_startup_does_not_arm_or_move(self):
        self.ready()
        self.assertFalse(self.ports["aux"].armed)
        self.assertFalse(any(b"$J=" in w for w in self.ports["grbl"].writes))
        with self.assertRaises(ProtocolError): self.controller.aux("MAGNET,1")
        self.assertNotIn(b"$I\n", self.ports["grbl"].writes)
        self.assertFalse(self.ports["aux"].motion_enabled)
        self.assertFalse(any(w.startswith((b"ROTATE,", b"SERVO,")) for w in self.ports["aux"].writes))
        with self.assertRaises(ProtocolError): self.controller.rotate(1)

    def test_pin_inputs_allow_all_ui_steps_and_feeds_but_hold_still_blocks(self):
        from unittest.mock import Mock
        from manual_control import ManualController
        controller = ManualController.__new__(ManualController)
        controller.config = {"grbl_feeds": [60, 300, 600], "z_feeds": [30, 60, 120]}
        state = {"grbl": {"machine": "Idle", "pins": "XY"}}
        controller.snapshot = lambda: state
        controller.submit = Mock()
        for axis in "XYZ":
            for distance in (0.1, 1, 10):
                for speed_index in (0, 1, 2):
                    controller.jog(axis, distance, speed_index)
                    feed = controller.config["z_feeds" if axis == "Z" else "grbl_feeds"][speed_index]
                    controller.submit.assert_called_with("grbl", jog_command(axis, distance, feed))
        controller.submit.reset_mock()
        state["grbl"]["machine"] = "Hold:0"
        with self.assertRaises(ProtocolError): controller.jog("X", 10, 2)
        controller.submit.assert_not_called()

    def test_aux_and_xyz_are_independent(self):
        self.ready()
        self.controller.aux("AUX,1")
        self.wait_for(lambda: not self.controller.snapshot()["aux"]["pending"])
        self.controller.aux("MAGNET,1")
        self.controller.jog("Z", -0.1, 0)
        self.wait_for(lambda: all(not s["pending"] for s in self.controller.snapshot().values()))
        self.assertEqual(self.ports["aux"].magnet, 1)
        self.assertIn(b"$J=G91 G21 Z-0.1 F30\n", self.ports["grbl"].writes)
        self.assertFalse(any(b"MOTION,1" in w for w in self.ports["aux"].writes))

    def test_no_jog_backlog_and_stop_closes_both(self):
        self.ready()
        self.controller.jog("X", 1, 0)
        with self.assertRaises(ProtocolError): self.controller.jog("X", 1, 0)
        self.controller.stop_all()
        self.wait_for(lambda: all(not w.is_alive() for w in self.controller.workers.values()))
        self.assertIn(b"!", self.ports["grbl"].writes)
        self.assertEqual(self.ports["aux"].armed, 0)

    def test_reply_loss_stops_other_device(self):
        self.ready()
        self.ports["aux"].drop = True
        self.wait_for(lambda: self.controller.fault)
        self.wait_for(lambda: not self.controller.workers["grbl"].is_alive())
        self.assertIn(b"!", self.ports["grbl"].writes)

    def test_frozen_ui_expires_without_heartbeat(self):
        self.ready()
        self.controller.ui_tick = time.monotonic()-2
        for worker in self.controller.workers.values(): worker.join(2)
        self.assertFalse(any(w.is_alive() for w in self.controller.workers.values()))
        self.assertEqual(self.ports["aux"].magnet, 0)

    def test_reconnect_while_held_does_not_resume_or_wait_for_line_command(self):
        self.wait_for(lambda: "grbl" in self.ports)
        self.ports["grbl"].machine = "Hold:0"
        self.ready()
        self.assertEqual(self.controller.snapshot()["grbl"]["machine"], "Hold:0")
        self.assertNotIn(b"$I\n", self.ports["grbl"].writes)
        self.assertNotIn(b"~", self.ports["grbl"].writes)

    def enable_rotation(self):
        self.ready()
        self.controller.rotation_enable()
        self.wait_for(lambda: not self.controller.snapshot()["aux"]["pending"])
        self.assertTrue(self.controller.snapshot()["aux"]["zeroed"])

    def test_rotation_waits_for_completion_and_renews_relay_lease(self):
        self.enable_rotation()
        self.controller.aux("AUX,1")
        self.wait_for(lambda: not self.controller.snapshot()["aux"]["pending"])
        self.controller.aux("MAGNET,1")
        self.wait_for(lambda: not self.controller.snapshot()["aux"]["pending"])
        self.controller.rotate(90)
        self.wait_for(lambda: self.controller.snapshot()["aux"]["motion_busy"])
        self.assertTrue(self.controller.snapshot()["aux"]["pending"])
        with self.assertRaises(ProtocolError): self.controller.rotate(1)
        self.wait_for(lambda: not self.controller.snapshot()["aux"]["pending"])
        writes = self.ports["aux"].writes
        after_rotate = writes[writes.index(b"ROTATE,90.00\n")+1:]
        self.assertGreaterEqual(after_rotate.count(b"IOSTATUS\n"), 3)
        self.assertEqual(writes.count(b"ROTATE,90.00\n"), 1)
        self.assertEqual(self.controller.snapshot()["aux"]["r_x100"], 9000)
        self.assertEqual(self.ports["aux"].magnet, 1)
        self.assertFalse(self.controller.fault)
        self.assertFalse(any(w.startswith(b"SERVO,") for w in writes))

    def test_rotation_ack_between_io_and_motion_status_is_not_lost(self):
        self.enable_rotation()
        self.ports["aux"].finish_before_status = True
        self.controller.rotate(-45)
        self.wait_for(lambda: not self.controller.snapshot()["aux"]["pending"])
        self.assertEqual(self.controller.snapshot()["aux"]["r_x100"], -4500)
        self.assertFalse(self.controller.fault)

    def test_rotation_stop_interrupts_wait_and_releases_magnet(self):
        self.enable_rotation()
        self.ports["aux"].rotation_delay = 10
        self.controller.rotate(90)
        self.wait_for(lambda: self.controller.snapshot()["aux"]["motion_busy"])
        self.controller.stop_rotation()
        self.wait_for(lambda: not self.controller.workers["aux"].is_alive())
        self.assertIsNone(self.ports["aux"].rotation)
        self.assertFalse(self.ports["aux"].motion_enabled)
        self.assertEqual(self.ports["aux"].magnet, 0)
        self.assertEqual(self.ports["aux"].writes.count(b"ROTATE,90.00\n"), 1)
        self.assertTrue(self.controller.workers["grbl"].is_alive())

    def test_rotation_range_is_checked_before_write(self):
        self.enable_rotation()
        self.ports["aux"].r_x100 = 18000
        self.controller.update("aux", r_x100=18000)
        with self.assertRaises(ProtocolError): self.controller.rotate(1)
        self.assertNotIn(b"ROTATE,1.00\n", self.ports["aux"].writes)

    def test_wrong_rotation_ack_stops_without_retry(self):
        self.enable_rotation()
        self.ports["aux"].wrong_rotate_ack = True
        self.controller.rotate(5)
        self.wait_for(lambda: self.controller.fault)
        self.wait_for(lambda: not self.controller.workers["aux"].is_alive())
        self.assertEqual(self.ports["aux"].writes.count(b"ROTATE,5.00\n"), 1)
        self.assertFalse(self.ports["aux"].motion_enabled)

    def test_idle_status_without_rotate_ack_does_not_confirm_completion(self):
        self.enable_rotation()
        self.ports["aux"].omit_rotate_ack = True
        self.controller.rotate(5)
        self.wait_for(lambda: self.controller.snapshot()["aux"]["r_x100"] == 500)
        self.assertTrue(self.controller.snapshot()["aux"]["pending"])
        self.controller.stop_rotation()
        self.wait_for(lambda: not self.controller.workers["aux"].is_alive())

    def test_simulated_mode_cannot_be_used_as_real_rotation(self):
        self.enable_rotation()
        self.ports["aux"].motion_enabled = False
        self.controller.rotate(10)
        self.wait_for(lambda: self.controller.fault)
        self.assertNotIn(b"ROTATE,10.00\n", self.ports["aux"].writes)


class TouchTests(unittest.TestCase):
    def test_home_and_debounce(self):
        state = StartPageState()
        self.assertEqual(state.on_points([(MANUAL_BTN[0]+1, MANUAL_BTN[1]+1)]), "manual")
        self.assertIsNone(state.on_points([(MANUAL_BTN[0]+1, MANUAL_BTN[1]+1)]))
        manual = ManualPageState()
        x, y, _, _ = BUTTONS["forward"]
        self.assertIsNone(manual.on_points([(x+1,y+1)]))
        manual.on_points([])
        self.assertEqual(manual.on_points([(x+1,y+1)]), "forward")
        self.assertIsNone(manual.on_points(None))
        self.assertIsNone(manual.on_points([(x+1,y+1)]))

    def test_angle_button_is_only_a_setting_and_requires_release(self):
        state = ManualPageState()
        state.on_points([])
        x, y, _, _ = BUTTONS["angle"]
        self.assertEqual(state.on_points([(x+1,y+1)]), "angle")
        self.assertIsNone(state.on_points([(x+1,y+1)]))

    def test_page_actions_use_rotation_protocol_not_servo(self):
        from unittest.mock import Mock, patch
        from manual_page import run_manual_page
        controller = Mock()
        controller.snapshot.return_value = {"grbl": {"connected": True}, "aux": {"connected": True}}
        events = [[]]
        for name in ("angle", "forward", "reverse", "rotation_stop", "back"):
            x, y, _, _ = BUTTONS[name]
            events.extend(([(x+1,y+1)], []))
        with patch("manual_page.paint"), patch("manual_page.time.sleep"):
            run_manual_page(None, None, {}, Mock(side_effect=events), None,
                            controller_factory=lambda config: controller)
        self.assertEqual([call.args for call in controller.rotate.call_args_list], [(5,), (-5,)])
        controller.stop_rotation.assert_called_once()
        controller.aux.assert_not_called()
        controller.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
