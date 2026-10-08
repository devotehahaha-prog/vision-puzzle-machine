"""Safe real-camera entry point; PyBullet remains isolated in run_sim.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import main as vision


ROOT = Path(__file__).resolve().parent


def _port_value(arguments: Sequence[str]) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == "--port":
            return arguments[index + 1] if index + 1 < len(arguments) else ""
        if argument.startswith("--port="):
            return argument.partition("=")[2]
    return None


def _has_mode(arguments: Sequence[str]) -> bool:
    modes = ("--auto", "--send-plan", "--image", "--camera", "--demo")
    return any(argument == mode or argument.startswith(mode + "=") for argument in arguments
               for mode in modes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=ROOT / "real_config.json")
    parser.add_argument("--execute", action="store_true")
    options, arguments = parser.parse_known_args(argv)

    port = _port_value(arguments)
    if options.execute:
        if port is None or not port.strip():
            parser.error("--execute requires an explicit non-empty --port")
        if "--dry-run" in arguments:
            parser.error("--execute cannot be combined with --dry-run")
        if not any(argument == "--auto" or argument == "--send-plan"
                   or argument.startswith("--send-plan=") for argument in arguments):
            parser.error("--execute is only valid with --auto or --send-plan")
    else:
        if not _has_mode(arguments):
            arguments.append("--auto")
        if ("--auto" in arguments or "--send-plan" in arguments
                or any(argument.startswith("--send-plan=") for argument in arguments)):
            if "--dry-run" not in arguments:
                arguments.append("--dry-run")

    config_path = options.config.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"configuration file not found: {config_path}")

    previous_config_path = vision.CONFIG_PATH
    previous_loader = vision.load_config
    vision.CONFIG_PATH = config_path

    def load_safe_config():
        with config_path.open("r", encoding="utf-8-sig") as config_file:
            config = json.load(config_file)
        config.update({
            "serial_port": port if options.execute else "",
            "auto_fallback_enabled": False,
            "force_execute_on_any_solution": False,
            "generic_ultimate_fallback_enable": False,
        })
        return config

    vision.load_config = load_safe_config
    previous_argv = sys.argv
    vision_arguments = [argument for argument in arguments if argument != "--execute"]
    sys.argv = [str(ROOT / "run_real.py"), *vision_arguments]
    if options.execute:
        print(f"[REAL] Explicit execution enabled on {port}; verify workspace and machine clearance.")
    else:
        print("[REAL] Dry-run only; no serial device will be opened.")
    try:
        return vision.main()
    finally:
        sys.argv = previous_argv
        vision.CONFIG_PATH = previous_config_path
        vision.load_config = previous_loader


if __name__ == "__main__":
    raise SystemExit(main())
