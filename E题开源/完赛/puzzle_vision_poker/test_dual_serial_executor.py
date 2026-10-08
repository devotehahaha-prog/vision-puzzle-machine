from __future__ import annotations

import copy
import re
import signal
import unittest
from unittest.mock import patch

from dual_serial_executor import (
    CoordinateCalibration,
    DualExecutionError,
    DualSerialExecutor,
    SerialTransportError,
    ControllerResetError,
)


BASE_CONFIG = {
    "calibration_complete": True,
    "coordinate_system": {
        "origin": "A4_top_left", "x_positive": "right",
        "y_positive": "down", "z_positive": "down",
        "rotation_positive": "clockwise",
    },
    "source_region": "bottom", "target_region": "top",
    "expected_piece_count": 3,
    "xy_transform": {"matrix": [[1.0, 0.0], [0.0, 1.0]], "bias_mm": [1.0, -2.0]},
    "tool_offset_mm": [3.0, 4.0],
    "z": {"safe": 0.0, "pickup": 2.0, "place": 3.0, "feed_mm_min": 60.0},
    "rotation": {"pulses_per_revolution": 3200, "positive_direction": "clockwise"},
    "motion": {"xy_feed_mm_min": 300.0, "pickup_settle_s": 0.0, "release_settle_s": 0.0},
    "serial": {"grbl_port": "grbl", "stm32_port": "stm", "baudrate": 115200,
               "timeout_s": 0.1},
}


def plan():
    result = {"ready_for_motion": True, "all_matches_ok": True,
            "all_paths_ok": True, "pieces": [
        {"move_order": 1, "source_pick_mm": [20, 200], "target_pick_mm": [20, 80],
         "rotation_deg_signed": 1.0},
        {"move_order": 2, "source_pick_mm": [80, 210], "target_pick_mm": [80, 80],
         "rotation_deg_signed": -2.0},
        {"move_order": 3, "source_pick_mm": [140, 220], "target_pick_mm": [140, 80],
         "rotation_deg_signed": 3.0},
    ]}
    for piece in result["pieces"]:
        source, target = piece["source_pick_mm"], piece["target_pick_mm"]
        piece["path_mm"] = [source, [source[0], target[1]], target]
    return result


class FakePort:
    def __init__(self, name):
        self.port = name
        self.writes = []
        self.responses = []
        self.closed = False
        self.xyz = [0., 0., 0.]
        self.r = 0
        self.mode = "SIM"
        self.aux = 0
        self.magnet = 0
        self.raw_writes = []

    def write(self, payload):
        self.raw_writes.append(payload)
        if payload == b"!\x85":
            self.writes.append("STOP_REALTIME")
            return
        line = payload.decode("ascii").strip()
        self.writes.append(line)
        if self.port == "grbl":
            if line == "?":
                self.responses.append(("<Idle|MPos:" + ",".join(f"{v:.3f}" for v in self.xyz) + "|FS:0,0>\n").encode())
            elif line.startswith("$J="):
                for axis, value in re.findall(r"([XYZ])(-?\d+(?:\.\d+)?)", line):
                    i = "XYZ".index(axis)
                    self.xyz[i] = self.xyz[i] + float(value) if "G91" in line else float(value)
                self.responses.append(b"ok\n")
        else:
            if line == "PING":
                self.responses.append(b"ACK,PING\n")
            elif line == "STATUS":
                self.responses.append(f"ACK,STATUS,{self.mode},IDLE,ZEROED,0,0,0,{self.r}\n".encode())
            elif line == "IOSTATUS":
                self.responses.append(f"ACK,IOSTATUS,AUX,{self.aux},SERVO,0,MAGNET,{self.magnet}\n".encode())
            elif line.startswith("ROTATE,"):
                delta = round(float(line.split(",")[1])*100)
                self.r += delta
                self.responses.append(f"ACK,ROTATE,{delta}\n".encode())
            else:
                if line == "STOP":
                    self.aux = self.magnet = 0
                    self.mode = "SIM"
                if line == "ZERO": self.r = 0
                if line == "MOTION,1": self.mode = "REAL"
                if line == "MOTION,0": self.mode = "SIM"
                if line.startswith("AUX,"): self.aux = int(line.split(",")[1])
                if line.startswith("MAGNET,"): self.magnet = int(line.split(",")[1])
                self.responses.append(("ACK," + line + "\n").encode("ascii"))

    def readline(self):
        return self.responses.pop(0) if self.responses else b""

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeFactory:
    def __init__(self):
        self.ports = {}

    def __call__(self, **kwargs):
        port = FakePort(kwargs["port"])
        self.ports[port.port] = port
        return port


class DualExecutorTests(unittest.TestCase):
    def test_boot_during_open_cannot_be_discarded_as_valid_origin(self):
        factory = FakeFactory()
        def boot_factory(**kwargs):
            port = factory(**kwargs)
            if kwargs['port'] == 'grbl':
                port.responses.extend([b'et', b's Jul 29 2019 12:21:46\n', b'<Idle|MPos:0,0,0>\n'])
            return port
        executor = DualSerialExecutor(BASE_CONFIG, boot_factory)
        executor.origin_valid = True
        executor.approved_plan_id = 'old'
        with self.assertRaisesRegex(ControllerResetError, '连接时检测到'):
            executor.open()
        self.assertTrue(executor.locked)
        self.assertFalse(executor.origin_valid)
        self.assertIsNone(executor.approved_plan_id)
        self.assertTrue(all(p.closed for p in factory.ports.values()))
        self.assertTrue(all(not p.writes for p in factory.ports.values()))

    def test_startup_disconnect_is_not_silently_drained(self):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, factory)
        with patch.object(FakePort, 'readline', side_effect=OSError(5, 'Input/output error')):
            with self.assertRaisesRegex(SerialTransportError, '启动读取'):
                executor.open()
        self.assertIsNone(executor.grbl)
        self.assertIsNone(executor.stm32)
        self.assertTrue(all(port.closed for port in factory.ports.values()))

    def test_reset_during_status_invalidates_origin_and_stops_other_controller(self):
        for banner in (b'ets Jul 29 2019 12:21:46\n', b'rst:0x1 (POWERON_RESET)\n', b'ESP-ROM:esp32\n'):
            with self.subTest(banner=banner):
                factory = FakeFactory()
                executor = DualSerialExecutor(BASE_CONFIG, factory)
                executor.open()
                try:
                    executor.origin_valid = True
                    executor.approved_plan_id = 'old'
                    factory.ports['grbl'].responses.append(banner)
                    with self.assertRaises(ControllerResetError): executor.grbl_status()
                    self.assertFalse(executor.origin_valid)
                    self.assertIsNone(executor.approved_plan_id)
                    self.assertTrue(executor.locked)
                    self.assertIn('STOP', factory.ports['stm'].writes)
                    self.assertNotIn(b'~', factory.ports['grbl'].raw_writes)
                finally: executor.close()

    def test_reset_instead_of_jog_ack_is_not_replayed(self):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, factory, sleeper=lambda _: None)
        executor.open()
        original = FakePort.write
        def reset_on_jog(port, payload):
            original(port, payload)
            if port.port == 'grbl' and payload.startswith(b'$J='):
                port.responses[-1] = b'rst:0x1 (POWERON_RESET)\n'
        try:
            with patch.object(FakePort, 'write', reset_on_jog):
                with self.assertRaises(ControllerResetError): executor.execute(plan())
            self.assertEqual(sum(s.startswith('$J=') for s in factory.ports['grbl'].writes), 1)
            self.assertTrue(executor.locked)
            self.assertEqual(factory.ports['stm'].magnet, 0)
        finally: executor.close()

    def test_first_jog_error_one_is_not_retried_or_bypassed(self):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, factory, sleeper=lambda _: None)
        executor.open()
        original = FakePort.write
        def reject_jog(port, payload):
            original(port, payload)
            if port.port == 'grbl' and payload.startswith(b'$J='):
                port.responses[-1] = b'error:1\n'
        try:
            with patch.object(FakePort, 'write', reject_jog):
                with self.assertRaisesRegex(DualExecutionError, '未自动重发'): executor.execute(plan())
            self.assertEqual(sum(s.startswith('$J=') for s in factory.ports['grbl'].writes), 1)
            self.assertNotIn(b'~', factory.ports['grbl'].raw_writes)
            self.assertNotIn('MAGNET,1', factory.ports['stm'].writes)
        finally: executor.close()

    def test_stm_boot_during_rotation_invalidates_session(self):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, factory)
        executor.open()
        try:
            factory.ports['stm'].responses.append(b'BOOT,MOTION,READY\n')
            with self.assertRaises(ControllerResetError): executor.rotate(90)
            self.assertTrue(executor.locked)
            self.assertFalse(executor.origin_valid)
            self.assertEqual(factory.ports['stm'].writes.count('ROTATE,90.00'), 1)
        finally: executor.close()

    def test_coordinate_mapping_applies_tool_offset_and_bias(self):
        calibration = CoordinateCalibration.from_config(BASE_CONFIG)
        self.assertEqual(calibration.map_point([10, 20]), (14.0, 22.0))

    def test_incomplete_calibration_is_rejected_before_open(self):
        config = copy.deepcopy(BASE_CONFIG)
        config["calibration_complete"] = False
        with self.assertRaises(DualExecutionError):
            DualSerialExecutor(config, serial_factory=FakeFactory())

    def test_manual_origin_accepts_component_calibration_without_grbl_homing(self):
        config = copy.deepcopy(BASE_CONFIG)
        config["calibration_complete"] = True
        config["calibration_state"] = {
            "camera": True, "xy": True, "z": True,
            "rotation": True, "grbl_homed": False,
        }
        config["motion"].update({
            "origin_mode": "manual",
            "manual_origin_tolerance_mm": 0.5,
        })
        executor = DualSerialExecutor(config, serial_factory=FakeFactory())
        executor.open()
        try:
            result = executor.preflight()
            self.assertEqual(result["grbl"]["MPos"], "0.000,0.000,0.000")
        finally:
            executor.close()

    def test_manual_origin_never_bypasses_final_acceptance(self):
        config = copy.deepcopy(BASE_CONFIG)
        config["motion"]["origin_mode"] = "manual"
        config["calibration_complete"] = False
        config["calibration_state"] = dict.fromkeys(
            ("camera", "xy", "z", "rotation"), True)
        config["calibration_state"]["grbl_homed"] = False
        factory = FakeFactory()
        with self.assertRaises(DualExecutionError):
            DualSerialExecutor(config, serial_factory=factory)
        self.assertEqual(factory.ports, {})

    def test_manual_origin_requires_component_states(self):
        config = copy.deepcopy(BASE_CONFIG)
        config["motion"]["origin_mode"] = "manual"
        with self.assertRaises(DualExecutionError):
            DualSerialExecutor(config, serial_factory=FakeFactory())

    def test_three_piece_sequence_uses_both_ports_and_stops_motion_gate(self):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, serial_factory=factory,
                                       sleeper=lambda _seconds: None)
        executor.open()
        try:
            executor.execute(plan())
        finally:
            executor.close()
        grbl = factory.ports["grbl"].writes
        stm = factory.ports["stm"].writes
        self.assertIn("MOTION,1", stm)
        self.assertIn("MOTION,0", stm)
        self.assertEqual(stm.count("MAGNET,1"), 3)
        self.assertEqual(stm.count("MAGNET,0"), 3)
        self.assertTrue(any(item.startswith("$J=G90") for item in grbl))
        self.assertTrue(any(item.startswith("$J=G91 G21 Z") and not "Z-" in item for item in grbl))
        self.assertTrue(any(item.startswith("$J=G91 G21 Z-") for item in grbl))
        self.assertEqual(executor.locked, False)
        self.assertEqual([x for x in stm if x.startswith("ROTATE,")],
                         ["ROTATE,1.00", "ROTATE,-1.00", "ROTATE,-2.00",
                          "ROTATE,2.00", "ROTATE,3.00", "ROTATE,-3.00"])
        self.assertEqual(factory.ports["grbl"].xyz, [0., 0., 0.])
        self.assertEqual(factory.ports["stm"].r, 0)
        self.assertIn("AUX,0", stm)
        for command in grbl:
            if command.startswith("$J=G90"):
                self.assertIn("G53", command)
                self.assertFalse("X" in command and "Y" in command)
        self.assertIn(b"?", factory.ports["grbl"].raw_writes)
        self.assertNotIn(b"?\n", factory.ports["grbl"].raw_writes)

    def test_sim_boot_is_enabled_before_real_motion(self):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, serial_factory=factory,
                                       sleeper=lambda _seconds: None)
        executor.open()
        try:
            executor.execute(plan())
        finally:
            executor.close()
        stm = factory.ports["stm"].writes
        self.assertLess(stm.index("MOTION,1"), stm.index("ZERO"))

    def run_fault(self, fault):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, serial_factory=factory,
                                      sleeper=lambda _: None)
        executor.open()
        try:
            original = FakePort.write
            def inject(port, payload):
                original(port, payload)
                fault(port, payload, executor)
            with patch.object(FakePort, "write", inject):
                with self.assertRaises((DualExecutionError, KeyboardInterrupt)):
                    executor.execute(plan())
            self.assertTrue(executor.locked)
            self.assertIn(b"!\x85", factory.ports["grbl"].raw_writes)
            self.assertIn("STOP", factory.ports["stm"].writes)
        finally:
            executor.close()

    def test_enable_failure_stops_both_controllers(self):
        def fault(port, payload, executor):
            if payload == b"MOTION,1\n": port.responses[-1] = b"ERR,MOTION\n"
        self.run_fault(fault)

    def test_wrong_rotate_ack_stops_without_replaying(self):
        def fault(port, payload, executor):
            if payload.startswith(b"ROTATE,"): port.responses[-1] = b"ACK,ROTATE,999\n"
        self.run_fault(fault)

    def test_magnet_lease_loss_stops_during_carry(self):
        def fault(port, payload, executor):
            if payload == b"IOSTATUS\n" and executor._magnet_expected:
                port.responses[-1] = b"ACK,IOSTATUS,AUX,0,SERVO,0,MAGNET,0\n"
        self.run_fault(fault)

    def test_owned_sigterm_handler_stops_both(self):
        previous = signal.getsignal(signal.SIGTERM)
        fired = False
        def fault(port, payload, executor):
            nonlocal fired
            if payload.startswith(b"$J=") and not fired:
                fired = True
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        self.run_fault(fault)
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_stale_heartbeat_does_not_poison_status_response(self):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, serial_factory=factory)
        executor.open()
        try:
            factory.ports["stm"].responses.append(b"ACK,IOSTATUS,AUX,0,SERVO,0,MAGNET,0\n")
            self.assertEqual(executor.stm_status()["motion"], "IDLE")
        finally:
            executor.close()

    def test_malformed_path_is_rejected_before_enable(self):
        executor = DualSerialExecutor(BASE_CONFIG, serial_factory=FakeFactory())
        invalid = plan()
        invalid["pieces"][0]["path_mm"][-1] = [999,999]
        with self.assertRaisesRegex(DualExecutionError, "路径端点"):
            executor.execute(invalid)

    def test_rejected_plan_never_opens_motion_gate(self):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, serial_factory=factory)
        with self.assertRaises(DualExecutionError):
            executor.execute({"ready_for_motion": False, "pieces": []})
        self.assertIsNone(executor.stm32)

    def test_initial_idle_before_jog_is_polled_without_replaying_command(self):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, factory, sleeper=lambda _: None)
        executor.open()
        original = FakePort.write
        pending = False
        def delayed_start(port, payload):
            nonlocal pending
            original(port, payload)
            if port.port == "grbl" and payload.startswith(b"$J="):
                pending = True
            elif port.port == "grbl" and payload == b"?" and pending:
                pending = False
                port.responses[-1] = b"<Idle|MPos:0,0,0|FS:0,0>\n"
        try:
            with patch.object(FakePort, "write", delayed_start):
                executor.move_xy(10, 0, 600)
            self.assertEqual(factory.ports['grbl'].xyz, [10., 0., 0.])
            self.assertEqual(len([s for s in factory.ports['grbl'].writes if s.startswith('$J=')]), 1)
        finally:
            executor.close()

    def test_wrong_idle_endpoint_after_running_is_still_rejected(self):
        executor = DualSerialExecutor(BASE_CONFIG, FakeFactory(), sleeper=lambda _: None)
        with patch.object(executor, 'grbl_status', side_effect=[
                {'state': 'Jog', 'MPos': '1,0,0'},
                {'state': 'Idle', 'MPos': '5,0,0'}]), patch.object(executor, '_heartbeat'):
            with self.assertRaisesRegex(DualExecutionError, '终点'):
                executor._wait_grbl_idle(expected={0: 10})

    def test_permanent_initial_idle_is_rejected_after_bounded_grace(self):
        now = [0.]
        executor = DualSerialExecutor(BASE_CONFIG, FakeFactory(), clock=lambda: now[0],
                                      sleeper=lambda dt: now.__setitem__(0, now[0] + dt))
        with patch.object(executor, 'grbl_status', return_value={'state':'Idle', 'MPos':'0,0,0'}), \
             patch.object(executor, '_heartbeat'):
            with self.assertRaisesRegex(DualExecutionError, '终点'):
                executor._wait_grbl_idle(expected={0: 10})
        self.assertLess(now[0], 1.)

    def test_grbl_disconnect_after_third_release_verifies_stm_independently(self):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, factory, sleeper=lambda _: None)
        executor.open()
        original = FakePort.write
        disconnected = False
        def disconnect_after_release(port, payload):
            nonlocal disconnected
            if port.port == 'grbl' and disconnected:
                raise OSError(5, 'Input/output error')
            original(port, payload)
            if port.port == 'stm' and payload == b'MAGNET,0\n' and port.writes.count('MAGNET,0') == 3:
                disconnected = True
        try:
            executor.origin_valid = True
            with patch.object(FakePort, 'write', disconnect_after_release):
                with self.assertRaisesRegex(SerialTransportError, 'grbl.*写入'):
                    executor.execute(plan())
                self.assertFalse(executor.verify_stopped())
            self.assertTrue(executor.stop_details['stm_stop_ack'])
            self.assertTrue(executor.stop_details['outputs_off'])
            self.assertFalse(executor.stop_details['xyz_stopped'])
            self.assertFalse(executor.origin_valid)
            self.assertIsNone(executor.approved_plan_id)
            self.assertTrue(executor.locked)
            self.assertEqual(executor.failure_action, 'Z_TO(0.00)')
            self.assertIn('XYZ停止未确认', executor.stop_summary())
            self.assertIn('电磁铁及AUX已确认关闭', executor.stop_summary())
            self.assertEqual(factory.ports['grbl'].xyz[2], 3.)
            self.assertEqual(factory.ports['stm'].writes.count('MAGNET,0'), 3)
            self.assertNotIn(b'~', factory.ports['grbl'].raw_writes)
            self.assertEqual(factory.ports['stm'].writes[-2:], ['STOP', 'IOSTATUS'])
        finally:
            executor.close()

    def test_stm_read_disconnect_during_rotate_is_fatal_without_waiting(self):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, factory)
        executor.open()
        try:
            with patch.object(factory.ports['stm'], 'readline', side_effect=OSError(5, 'Input/output error')):
                with self.assertRaisesRegex(SerialTransportError, 'stm.*读取'):
                    executor.rotate(90)
            self.assertEqual(factory.ports['stm'].writes, ['ROTATE,90.00'])
        finally:
            executor.close()

    def test_hold_decelerating_is_not_confirmed_stopped(self):
        factory = FakeFactory()
        executor = DualSerialExecutor(BASE_CONFIG, factory)
        executor.open()
        try:
            executor.stop_both()
            with patch.object(executor, 'grbl_status', return_value={'state':'Hold','raw_state':'Hold:1'}):
                self.assertFalse(executor.verify_stopped())
            self.assertTrue(executor.stop_details['outputs_off'])
            self.assertFalse(executor.stop_details['xyz_stopped'])
        finally:
            executor.close()

    def test_requested_four_mm_depths_and_high_feeds_return_to_safe_origin(self):
        config = copy.deepcopy(BASE_CONFIG)
        config['z'].update(safe=0., pickup=4., place=4., feed_mm_min=120.)
        config['motion']['xy_feed_mm_min'] = 600.
        factory = FakeFactory()
        executor = DualSerialExecutor(config, factory, sleeper=lambda _: None)
        executor.open()
        try:
            executor.execute(plan())
            commands = factory.ports['grbl'].writes
            self.assertEqual(commands.count('$J=G91 G21 Z4.000 F120.0'), 6)
            self.assertEqual(commands.count('$J=G91 G21 Z-4.000 F120.0'), 6)
            self.assertTrue(all(s.endswith('F600.0') for s in commands if s.startswith('$J=G90')))
            self.assertEqual(factory.ports['grbl'].xyz, [0., 0., 0.])
        finally:
            executor.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
