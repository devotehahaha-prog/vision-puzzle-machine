"""Launch the puzzle-device gantry demo with the existing pybullet venv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator import PuzzleGantrySim, demo_pieces, demo_tasks, run_cli  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="E-题拼图装置 PyBullet 执行机构仿真")
    parser.add_argument("--direct", action="store_true", help="无窗口模式，只做冒烟步进")
    parser.add_argument("--no-realtime", action="store_true", help="GUI 下尽快步进，不等 240Hz")
    args = parser.parse_args()

    sim = PuzzleGantrySim(gui=not args.direct, realtime=not args.no_realtime and not args.direct)
    try:
        sim.spawn_pieces(demo_pieces())
        if args.direct:
            sim.run_queue(demo_tasks())
            print("DIRECT smoke test finished")
            return
        print("GUI: START / HOME / ROTATE +15    keyboard: s / h / r / q")
        sim.run_with_buttons(demo_tasks())
    finally:
        sim.close()


if __name__ == "__main__":
    run_cli(main)
