"""No-reset serial opening and read-only failure evidence.

The SPI UI bundles an identical copy so both applications remain standalone.
No reconnect, command retry, controller reset or sysfs write belongs here.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


class _InactiveModemLines:
    """pySerial POSIX hooks: clear both reset-related bits in one ioctl.

    pySerial 3.5 normally sets DTR, then RTS separately during open. On the
    observed ESP32 board that intermediate state produced a boot banner.
    These hooks keep the normal pySerial open/lock/error cleanup intact.
    """
    def _apply_inactive_lines(self):
        if self._dtr_state or self._rts_state:
            raise ValueError('Reset control lines must remain inactive')
        import fcntl
        import struct
        import termios
        fcntl.ioctl(self.fd, termios.TIOCMBIC,
                    struct.pack('I', termios.TIOCM_DTR | termios.TIOCM_RTS))

    def _update_dtr_state(self):
        self._apply_inactive_lines()

    def _update_rts_state(self):
        self._apply_inactive_lines()


def _serial_class():
    import serial
    if sys.platform.startswith('linux'):
        class InactiveSerial(_InactiveModemLines, serial.Serial):
            pass
        return InactiveSerial
    return serial.Serial


def open_serial(port, baudrate, timeout, write_timeout, factory=None):
    if factory is not None:
        return factory(port=port, baudrate=baudrate, timeout=timeout,
                       write_timeout=write_timeout, exclusive=True)
    connection = _serial_class()(port=None, baudrate=baudrate, timeout=timeout,
                               write_timeout=write_timeout, exclusive=True,
                               xonxoff=False, rtscts=False, dsrdtr=False)
    try:
        # Set desired modem levels before open. A driver-level opening glitch
        # remains possible; this is not a guarantee against hardware resets.
        connection.dtr = False
        connection.rts = False
        connection.port = port
        connection.open()
        if sys.platform.startswith("linux"):
            import termios
            attrs = termios.tcgetattr(connection.fileno())
            if attrs[2] & termios.HUPCL:
                attrs[2] &= ~termios.HUPCL
                termios.tcsetattr(connection.fileno(), termios.TCSANOW, attrs)
        return connection
    except BaseException:
        connection.close()
        raise


def reset_banner(line):
    return line.startswith(("Grbl ", "Grbl_", "ets ", "rst:0x", "ESP-ROM:", "BOOT,"))


def port_identity(port):
    """Capture tty path and USB generation without opening a device."""
    result = {"configured": str(port)}
    try:
        target = Path(port).resolve()
        result.update(resolved=str(target), present=target.exists())
        if target.exists():
            info = target.stat()
            result.update(inode=info.st_ino, rdev=info.st_rdev)
        device = Path("/sys/class/tty") / target.name / "device"
        if device.exists():
            for parent in device.resolve().parents:
                if (parent / "idVendor").exists():
                    result["usb_path"] = parent.name
                    for name in ("idVendor", "idProduct", "busnum", "devnum"):
                        result[name] = (parent / name).read_text().strip()
                    break
    except OSError as exc:
        result["snapshot_error"] = str(exc)
    return result


def transport_failure(port, operation, exc, opened):
    current = port_identity(port)
    hint = "本次连接失效，请重新连接并确认原点；未自动重发指令"
    if opened.get("present") and not current.get("present"):
        hint = "串口设备已消失；" + hint
    elif opened.get("present") and any(opened.get(key) != current.get(key)
                                         for key in ("resolved", "inode", "devnum")):
        hint = "串口设备已变更/重新枚举，旧句柄失效；" + hint
    evidence = json.dumps({"opened": opened, "current": current}, ensure_ascii=False, sort_keys=True)
    return f"串口{port}{operation}失败：{exc}；{hint}；设备证据={evidence}"
