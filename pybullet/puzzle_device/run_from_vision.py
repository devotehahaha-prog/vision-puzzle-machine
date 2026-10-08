"""Drive the gantry sim from a MaixCAM 45-byte frame or vision log."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protocol import encode_frame, format_tasks, load_tasks  # noqa: E402
from simulator import PuzzleGantrySim, demo_pieces, run_cli  # noqa: E402


def _read_stdin() -> str:
    print("Paste a hex frame, JSON tasks, or a vision log line, then Ctrl+Z Enter:")
    return sys.stdin.read()


def _listen_serial(port: str, baud: int, skip_zero: bool):
    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "serial listen needs pyserial: .venv\\Scripts\\python.exe -m pip install pyserial"
        ) from exc
    from protocol import FRAME_LEN, SOF, parse_frame

    with serial.Serial(port, baud, timeout=0.2) as ser:
        print(f"listening on {port} {baud}, waiting for A5 5A ...")
        buf = bytearray()
        while True:
            buf.extend(ser.read(64))
            start = buf.find(SOF)
            if start < 0:
                if len(buf) > FRAME_LEN:
                    del buf[:-1]
                continue
            if len(buf) - start < FRAME_LEN:
                continue
            frame = bytes(buf[start : start + FRAME_LEN])
            del buf[: start + FRAME_LEN]
            try:
                return parse_frame(frame, skip_zero=skip_zero)
            except ValueError as error:
                print(f"bad frame: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用视觉任务帧驱动拼图装置仿真"
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="45字节hex、JSON、日志文件路径，或含 A5 5A 的文本",
    )
    parser.add_argument("--stdin", action="store_true", help="从标准输入粘贴任务")
    parser.add_argument("--serial", help="监听串口，例如 COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument(
        "--keep-zero",
        action="store_true",
        help="保留全零补位任务（默认丢弃，避免去原点空抓）",
    )
    parser.add_argument(
        "--print-sample",
        action="store_true",
        help="打印当前照片碎片演示任务的45字节hex后退出",
    )
    args = parser.parse_args()
    skip_zero = not args.keep_zero

    if args.print_sample:
        from simulator import demo_tasks

        frame = encode_frame(demo_tasks())
        print(" ".join(f"{byte:02X}" for byte in frame))
        return

    if args.serial:
        tasks = _listen_serial(args.serial, args.baud, skip_zero)
    elif args.stdin:
        tasks = load_tasks(_read_stdin(), skip_zero=skip_zero)
    elif args.source:
        tasks = load_tasks(args.source, skip_zero=skip_zero)
    else:
        parser.print_help()
        raise SystemExit(2)

    print(format_tasks(tasks))
    sim = PuzzleGantrySim(
        gui=not args.direct,
        realtime=not args.no_realtime and not args.direct,
    )
    try:
        sim.spawn_pieces(demo_pieces())
        if args.direct:
            sim.run_queue(tasks)
            print("VISION QUEUE DONE")
            return
        print("GUI: START / HOME / ROTATE +15    keyboard: s / h / r / q")
        sim.run_with_buttons(tasks)
    finally:
        sim.close()


if __name__ == "__main__":
    run_cli(main)
