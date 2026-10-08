"""Detect pieces from the A4 photo, solve a rectangle, then execute pick-and-place."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assemble import save_plan, solve_assembly  # noqa: E402
from board_from_photo import build_layout  # noqa: E402
from protocol import format_tasks  # noqa: E402
from simulator import PuzzleGantrySim, demo_pieces, run_cli  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="一键识别碎片并自动拼成矩形")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument(
        "--reuse-layout",
        action="store_true",
        help="不重新分析照片，使用已有 board_layout.json",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只计算拼图方案，不打开仿真",
    )
    args = parser.parse_args()

    if not args.reuse_layout:
        print("detecting A4 and pieces from photo...")
        layout = build_layout()
    else:
        from mesh_builder import load_layout

        layout = load_layout()
        print(f"reusing layout with {len(layout['pieces'])} pieces")

    print("searching rectangular assembly...")
    assembly = solve_assembly(layout)
    plan_path = save_plan(assembly)
    print(
        f"rectangle {assembly.size_mm[0]:.1f} x {assembly.size_mm[1]:.1f} mm, "
        f"fill={assembly.fill_ratio:.3f}"
    )
    print(format_tasks(assembly.tasks))
    print(f"plan image: {plan_path}")

    if args.plan_only:
        return

    sim = PuzzleGantrySim(
        gui=not args.direct,
        realtime=not args.no_realtime and not args.direct,
    )
    try:
        sim.spawn_pieces(demo_pieces())
        if args.direct:
            sim.run_queue(assembly.tasks)
            print("AUTO ASSEMBLE DONE")
            return
        print("Click the green START block in front of the table, or press SPACE.")
        print("Yellow HOME resets pieces. Blue ROTATE turns the selected piece.")

        def plan_from_current():
            live = sim.current_layout()
            planned = solve_assembly(live)
            path = save_plan(planned)
            print(
                f"rectangle {planned.size_mm[0]:.1f} x {planned.size_mm[1]:.1f} mm, "
                f"fill={planned.fill_ratio:.3f}"
            )
            print(format_tasks(planned.tasks))
            print(f"plan image: {path}")
            return planned.tasks

        sim.run_with_buttons(plan_fn=plan_from_current)
    finally:
        sim.close()


if __name__ == "__main__":
    run_cli(main)
