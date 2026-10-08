"""Parse MaixCAM/STM32 45-byte task frames and vision log text."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Iterable, Sequence

from simulator import MotionTask, PuzzleGantrySim

FRAME_LEN = 45
SOF = b"\xA5\x5A"
CMD = 0x10
TASK_COUNT = 4


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xA001) if crc & 1 else (crc >> 1)
    return crc


def encode_frame(tasks: Sequence[MotionTask], pad_to_four: bool = True) -> bytes:
    slots = list(tasks)
    PuzzleGantrySim.validate_tasks(slots)
    if any(task.path_mm is not None for task in slots):
        raise ValueError("45-byte frames cannot encode path waypoints")
    if pad_to_four:
        while len(slots) < TASK_COUNT:
            slots.append(MotionTask(0, 0, 0, 0, 0))
    elif len(slots) != TASK_COUNT:
        raise ValueError("a frame without padding must contain four tasks")
    payload = bytearray((CMD,))
    for task in slots:
        payload.extend(
            struct.pack(
                "<hhhhh",
                int(round(task.pick_x_mm)),
                int(round(task.pick_y_mm)),
                int(round(task.flip_deg)),
                int(round(task.place_x_mm)),
                int(round(task.place_y_mm)),
            )
        )
    return SOF + bytes(payload) + struct.pack("<H", crc16_modbus(payload))


def parse_frame(frame: bytes, skip_zero: bool = True) -> list[MotionTask]:
    if len(frame) != FRAME_LEN:
        raise ValueError(f"frame length {len(frame)} != {FRAME_LEN}")
    if frame[:2] != SOF:
        raise ValueError(f"bad SOF: {frame[:2].hex()}")
    if frame[2] != CMD:
        raise ValueError(f"bad CMD: {frame[2]:#x}")
    payload = frame[2:43]
    crc_got = struct.unpack_from("<H", frame, 43)[0]
    crc_exp = crc16_modbus(payload)
    if crc_got != crc_exp:
        raise ValueError(f"CRC mismatch got={crc_got:#06x} exp={crc_exp:#06x}")
    tasks = []
    for index in range(TASK_COUNT):
        fields = struct.unpack_from("<hhhhh", payload, 1 + index * 10)
        task = MotionTask(*fields)
        if skip_zero and _is_zero_task(task):
            continue
        tasks.append(task)
    if not tasks:
        raise ValueError("frame has no non-zero tasks")
    return tasks


def _is_zero_task(task: MotionTask) -> bool:
    return (
        task.pick_x_mm == 0
        and task.pick_y_mm == 0
        and task.flip_deg == 0
        and task.place_x_mm == 0
        and task.place_y_mm == 0
    )


def parse_hex_dump(text: str) -> bytes:
    match = re.search(r"A5\s*5A(?:\s*[0-9A-Fa-f]{2}){43}", text, re.IGNORECASE)
    if match is None:
        raise ValueError("no complete 45-byte hex frame")
    compact = re.sub(r"\s+", "", match.group(0))
    if len(compact) < FRAME_LEN * 2:
        raise ValueError("not enough hex bytes for a 45-byte frame")
    blob = bytes.fromhex(compact)
    start = blob.find(SOF)
    if start < 0:
        raise ValueError("no A5 5A header in hex dump")
    return blob[start : start + FRAME_LEN]


def parse_task_tuples(text: str) -> list[tuple[int, int, int, int, int]]:
    """Extract 5-int tuples such as vision 'task slots: [(..), ...]'."""
    found = re.findall(
        r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
        text,
    )
    return [tuple(int(v) for v in item) for item in found]


def parse_json_tasks(text: str) -> list[MotionTask]:
    data = json.loads(text)
    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]
    if not isinstance(data, list):
        raise ValueError("JSON must be a task list")
    tasks = []
    for item in data:
        if isinstance(item, dict):
            tasks.append(
                MotionTask(
                    item["pick_x_mm"],
                    item["pick_y_mm"],
                    item["flip_deg"],
                    item["place_x_mm"],
                    item["place_y_mm"],
                )
            )
        elif isinstance(item, (list, tuple)) and len(item) == 5:
            tasks.append(MotionTask(*item))
        else:
            raise ValueError(f"unsupported JSON task: {item!r}")
    return tasks


def parse_any(text: str, skip_zero: bool = True) -> list[MotionTask]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty input")

    tuples = parse_task_tuples(stripped)
    if tuples:
        tasks = [MotionTask(*item) for item in tuples]
        if skip_zero:
            tasks = [task for task in tasks if not _is_zero_task(task)]
        if tasks:
            return tasks

    hexish = re.sub(r"\s+", "", stripped)
    if re.fullmatch(r"[0-9A-Fa-f]+", hexish) and len(hexish) >= FRAME_LEN * 2:
        return parse_frame(parse_hex_dump(stripped), skip_zero=skip_zero)

    if "A5" in stripped.upper() and "5A" in stripped.upper():
        try:
            return parse_frame(parse_hex_dump(stripped), skip_zero=skip_zero)
        except ValueError:
            pass

    if stripped[:1] in "{[":
        try:
            tasks = parse_json_tasks(stripped)
            return [task for task in tasks if not (skip_zero and _is_zero_task(task))]
        except json.JSONDecodeError:
            pass

    raise ValueError("could not parse tasks from text")


def load_tasks(source: str, skip_zero: bool = True) -> list[MotionTask]:
    path = Path(source)
    if path.exists():
        raw = path.read_bytes()
        if b"\xA5\x5A" in raw:
            start = raw.find(b"\xA5\x5A")
            return parse_frame(raw[start : start + FRAME_LEN], skip_zero=skip_zero)
        return parse_any(raw.decode("utf-8", errors="replace"), skip_zero=skip_zero)
    return parse_any(source, skip_zero=skip_zero)


def format_tasks(tasks: Iterable[MotionTask]) -> str:
    lines = []
    for index, task in enumerate(tasks, start=1):
        lines.append(
            f"T{index}: pick=({task.pick_x_mm:.1f},{task.pick_y_mm:.1f}) "
            f"flip={task.flip_deg:.0f} place=({task.place_x_mm:.1f},{task.place_y_mm:.1f})"
        )
    return "\n".join(lines)
