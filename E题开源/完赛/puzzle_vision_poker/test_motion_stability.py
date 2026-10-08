"""Behavior checks for faster XY and contact-limited Z pickup/release."""
import copy
import re
import unittest
from unittest.mock import patch

from dual_serial_executor import DualExecutionError, DualSerialExecutor, SerialTransportError
from test_dual_serial_executor import BASE_CONFIG, FakeFactory, FakePort, plan
from main import dual_execution_summary
from approved_plan import config_digest


def motion_config():
    config = copy.deepcopy(BASE_CONFIG)
    config['z'].update(safe=0., travel=-2., pickup=4., place=4., feed_mm_min=240.,
                       contact_margin_mm=1., contact_feed_mm_min=120., settle_s=.25)
    config['motion'].update(xy_feed_mm_min=1800., pickup_settle_s=1.2,
                            release_settle_s=1., xy_settle_s=.25)
    config['rotation']['settle_s'] = .25
    config['calibration_state'] = dict.fromkeys(('camera','xy','z','rotation','grbl_homed'), True)
    return config


class MotionStabilityTests(unittest.TestCase):
    def setup_executor(self, config=None, **kwargs):
        factory = FakeFactory()
        executor = DualSerialExecutor(config or motion_config(), factory,
                                      sleeper=lambda _: None, **kwargs)
        executor.open()
        self.addCleanup(executor.close)
        return executor, factory

    def test_complete_three_piece_heights_feeds_outputs_and_waits(self):
        executor, factory = self.setup_executor()
        events = []
        original = FakePort.write
        def record(port, payload):
            original(port, payload)
            events.append((payload.decode('ascii').strip(),
                           factory.ports['grbl'].xyz[2], factory.ports['stm'].magnet))
        with patch.object(FakePort, 'write', record), \
             patch.object(executor, '_settle', wraps=executor._settle) as settle:
            executor.execute(plan())
        z = [(height, float(re.search(r'F([\d.]+)', cmd)[1]))
             for cmd, height, _ in events if cmd.startswith('$J=G91')]
        self.assertEqual(z, [(3,240),(4,120),(3,120),(-2,240),
                             (3,240),(4,120),(3,120),(0,240)] * 3)
        for cmd, height, magnet in events:
            if cmd.startswith('$J=G90'):
                self.assertTrue(cmd.endswith('F1800.0'))
                self.assertEqual(height, -2 if magnet else 0)
            if cmd.startswith('ROTATE,'):
                self.assertEqual(height, -2 if magnet else 0)
            if cmd in ('MAGNET,1', 'MAGNET,0'):
                self.assertEqual(height, 4)
        self.assertEqual([call.args[0] for call in settle.call_args_list],
                         [.25,.25,1.2,.25,.25,.25,.25,1.,.25,.25] * 3)
        self.assertEqual(factory.ports['grbl'].xyz, [0.,0.,0.])
        self.assertEqual(factory.ports['stm'].r, 0)
        self.assertEqual(factory.ports['stm'].magnet, 0)

    def test_relative_delta_uses_feedback_instead_of_assumed_start(self):
        executor, factory = self.setup_executor()
        factory.ports['grbl'].xyz[2] = 3.95
        executor.move_z_to(3., 120., expected_start=4.)
        self.assertIn('$J=G91 G21 Z-0.950 F120.0', factory.ports['grbl'].writes)
        self.assertAlmostEqual(factory.ports['grbl'].xyz[2], 3.)

    def test_unexpected_z_rejects_all_jogs_and_locks(self):
        executor, factory = self.setup_executor()
        factory.ports['grbl'].xyz[2] = 0.4
        with self.assertRaisesRegex(DualExecutionError, 'Z阶段'):
            executor.execute(plan())
        self.assertTrue(executor.locked)
        self.assertFalse(any(cmd.startswith('$J=') for cmd in factory.ports['grbl'].writes))
        self.assertIn('STOP', factory.ports['stm'].writes)

    def test_no_xyz_motion_after_release_output_is_stuck(self):
        executor, factory = self.setup_executor()
        original = FakePort.write
        def stuck(port, payload):
            original(port, payload)
            if payload == b'MAGNET,0\n': port.magnet = 1
        with patch.object(FakePort, 'write', stuck):
            with self.assertRaisesRegex(DualExecutionError, '输出未关闭'):
                executor.execute(plan())
        self.assertTrue(executor.locked)
        self.assertEqual(factory.ports['grbl'].xyz[2], 4)
        self.assertEqual(factory.ports['stm'].writes.count('MAGNET,1'), 1)

    def test_empty_run_never_uses_z_or_magnet(self):
        executor, factory = self.setup_executor(empty_run=True)
        executor.execute(plan())
        self.assertFalse(any(cmd.startswith('$J=G91') for cmd in factory.ports['grbl'].writes))
        self.assertFalse(any(cmd.startswith('MAGNET,') for cmd in factory.ports['stm'].writes))
        self.assertEqual(factory.ports['grbl'].xyz, [0,0,0])

    def test_legacy_config_keeps_single_stage_z_and_safe_travel(self):
        executor, factory = self.setup_executor(copy.deepcopy(BASE_CONFIG))
        executor.execute(plan())
        moves = [cmd for cmd in factory.ports['grbl'].writes if cmd.startswith('$J=G91')]
        self.assertEqual(moves, ['$J=G91 G21 Z2.000 F60.0', '$J=G91 G21 Z-2.000 F60.0',
                                '$J=G91 G21 Z3.000 F60.0', '$J=G91 G21 Z-3.000 F60.0'] * 3)

    def test_invalid_parameters_are_rejected_before_port_open(self):
        cases = [('z','travel',1.), ('z','travel',float('nan')),
                 ('z','contact_margin_mm',-1.), ('z','contact_margin_mm',5.),
                 ('z','contact_feed_mm_min',0.), ('z','feed_mm_min',float('inf')),
                 ('motion','xy_feed_mm_min',0.), ('motion','xy_settle_s',-1.),
                 ('motion','pickup_settle_s',float('nan')), ('motion','release_settle_s',None),
                 ('rotation','settle_s',-1.), ('z','settle_s',float('inf'))]
        for section, name, value in cases:
            with self.subTest(section=section, name=name, value=value):
                config = motion_config(); config[section][name] = value
                factory = FakeFactory()
                with self.assertRaises(DualExecutionError):
                    DualSerialExecutor(config, factory)
                self.assertEqual(factory.ports, {})

    def test_wait_renews_heartbeat_and_checks_stop_every_point_two_seconds(self):
        executor, factory = self.setup_executor()
        now = [0.]
        executor.clock = lambda: now[0]
        executor.sleeper = lambda dt: now.__setitem__(0, now[0] + dt)
        guard_times, heartbeat_times = [], []
        executor.session_guard = lambda: guard_times.append(now[0])
        original = FakePort.write
        def record(port, payload):
            original(port, payload)
            if payload == b'IOSTATUS\n': heartbeat_times.append(now[0])
        with patch.object(FakePort, 'write', record): executor._settle(3.2)
        self.assertGreaterEqual(len(heartbeat_times), 3)
        self.assertLessEqual(max(b-a for a,b in zip([0]+heartbeat_times, heartbeat_times)), 1.001)
        self.assertLessEqual(max(b-a for a,b in zip([0]+guard_times, guard_times)), .201)
        def stopped(): raise DualExecutionError('停止请求')
        executor.session_guard = stopped
        with self.assertRaisesRegex(DualExecutionError, '停止请求'): executor._settle(1.)

    def test_disconnect_during_pickup_wait_stops_without_lift_or_retry(self):
        executor, factory = self.setup_executor()
        original = FakePort.write
        def disconnected(port, payload):
            if port.port == 'stm' and payload == b'IOSTATUS\n' and port.magnet:
                raise OSError('disconnected')
            original(port, payload)
        with patch.object(FakePort, 'write', disconnected):
            with self.assertRaises(SerialTransportError): executor.execute(plan())
        self.assertTrue(executor.locked)
        self.assertEqual(factory.ports['grbl'].xyz[2], 4)
        self.assertEqual(factory.ports['stm'].writes.count('MAGNET,1'), 1)
        self.assertIn(b'!\x85', factory.ports['grbl'].raw_writes)

    def test_motion_changes_invalidate_config_digest(self):
        config = motion_config()
        for section, field in [('z','travel'),('z','contact_margin_mm'),
                               ('z','contact_feed_mm_min'),('motion','xy_settle_s'),
                               ('motion','xy_feed_mm_min'),('z','settle_s'),('rotation','settle_s')]:
            changed = copy.deepcopy(config); changed[section][field] += .1
            self.assertNotEqual(config_digest(config), config_digest(changed))

    def test_summary_uses_actual_nested_values_and_legacy_defaults(self):
        config = motion_config()
        config['magnet_pickup_settle_seconds'] = 99
        summary = dual_execution_summary(config)
        self.assertEqual(summary['execution_motion']['z_travel_mm'], -2)
        self.assertEqual(summary['execution_motion']['xy_feed_mm_min'], 1800)
        self.assertEqual(summary['execution_timing']['magnet_pickup_settle_seconds'], 1.2)
        self.assertEqual(summary['execution_timing']['rotation_settle_seconds'], .25)
        self.assertEqual(summary['execution_compensation']['xy_command_bias_mm'], [1,-2])
        legacy = copy.deepcopy(BASE_CONFIG); legacy['motion'] = {}; legacy['z'].pop('feed_mm_min')
        summary = dual_execution_summary(legacy)
        self.assertEqual(summary['execution_motion']['z_travel_mm'], 0)
        self.assertEqual(summary['execution_motion']['z_contact_margin_mm'], 0)
        self.assertEqual(summary['execution_motion']['z_contact_feed_mm_min'], 60)
        self.assertEqual(summary['execution_timing']['magnet_pickup_settle_seconds'], .8)
        self.assertEqual(summary['execution_timing']['magnet_release_settle_seconds'], .3)
        self.assertEqual(summary['execution_timing']['xy_settle_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
