"""Opening policy and generation diagnostics; never opens physical hardware."""
import sys
import types
import unittest
from unittest.mock import Mock, patch

from serial_connection import open_serial, port_identity, reset_banner, transport_failure, _InactiveModemLines


class OpeningTests(unittest.TestCase):
    def test_each_linux_modem_hook_clears_both_bits_atomically(self):
        port = _InactiveModemLines()
        port.fd = 12
        port._dtr_state = port._rts_state = False
        ioctl = Mock()
        term = types.SimpleNamespace(TIOCMBIC=0x5417,TIOCM_DTR=2,TIOCM_RTS=4)
        with patch.dict(sys.modules, termios=term, fcntl=types.SimpleNamespace(ioctl=ioctl)):
            port._update_dtr_state()
            port._update_rts_state()
            import struct
            self.assertEqual(ioctl.call_count,2)
            for call in ioctl.call_args_list:
                self.assertEqual(call.args,(12,0x5417,struct.pack('I',6)))
            port._dtr_state=True
            with self.assertRaises(ValueError): port._update_dtr_state()
            self.assertEqual(ioctl.call_count,2)

    def test_modem_lines_are_deasserted_before_open_and_hupcl_is_cleared(self):
        calls = []
        class Port:
            def __init__(self, **kwargs):
                self.settings = kwargs
                self.port = kwargs['port']
                self.dtr = self.rts = True
            def open(self):
                calls.append(('open', self.port, self.dtr, self.rts))
            def fileno(self): return 12
            def close(self): calls.append(('close',))
        term = types.SimpleNamespace(HUPCL=0x400, TCSANOW=0,
            tcgetattr=Mock(return_value=[1, 2, 0xC80, 4, 5, 6, []]), tcsetattr=Mock())
        with patch.dict(sys.modules, serial=types.SimpleNamespace(Serial=Port), termios=term), \
                patch('serial_connection.sys.platform', 'linux'):
            port = open_serial('/dev/test', 115200, .1, .5)
        self.assertEqual(calls, [('open', '/dev/test', False, False)])
        self.assertIsNone(port.settings['port'])
        self.assertTrue(port.settings['exclusive'])
        for setting in ('xonxoff', 'rtscts', 'dsrdtr'):
            self.assertFalse(port.settings[setting])
        term.tcsetattr.assert_called_once_with(12, 0, [1, 2, 0x880, 4, 5, 6, []])

    def test_termios_failure_closes_handle_and_does_not_retry_open(self):
        port = Mock()
        term = types.SimpleNamespace(tcgetattr=Mock(side_effect=OSError(5, 'gone')))
        maker = Mock(return_value=port)
        with patch('serial_connection._serial_class', return_value=maker), \
                patch.dict(sys.modules, termios=term), \
                patch('serial_connection.sys.platform', 'linux'):
            with self.assertRaises(OSError): open_serial('/dev/test', 115200, .1, .5)
        port.open.assert_called_once()
        port.close.assert_called_once()
        maker.assert_called_once()

    def test_all_known_reset_banners_are_fatal_but_status_is_not(self):
        for line in ('ets Jul 29 2019 12:21:46', 'rst:0x1 (POWERON_RESET)',
                     'ESP-ROM:esp32c3', 'Grbl 1.3a', 'Grbl_ESP32', 'BOOT,MOTION,READY'):
            with self.subTest(line=line): self.assertTrue(reset_banner(line))
        for line in ('<Idle|MPos:0,0,0>', 'ACK,STATUS,REAL,IDLE', '[MSG:hello]', 'ok'):
            self.assertFalse(reset_banner(line))

    def test_missing_device_and_reenumeration_are_distinct(self):
        opened = dict(present=True, resolved='/dev/ttyUSB0', inode=101, devnum='18')
        for current, expected in ((dict(present=False), '设备已消失'),
                 (dict(present=True, resolved='/dev/ttyUSB1', inode=102, devnum='19'), '重新枚举'),
                 (opened.copy(), '本次连接失效')):
            with self.subTest(current=current), patch('serial_connection.port_identity', return_value=current):
                message = transport_failure('/dev/serial/by-id/test', '写入', OSError(5, 'EIO'), opened)
                self.assertIn(expected, message)
                self.assertIn('未自动重发', message)
                self.assertIn('"devnum": "18"', message)

    def test_identity_of_missing_path_does_not_open_hardware(self):
        evidence = port_identity('/nonexistent/serial/test')
        self.assertFalse(evidence['present'])
        self.assertEqual(evidence['configured'], '/nonexistent/serial/test')


if __name__ == '__main__': unittest.main(verbosity=2)
