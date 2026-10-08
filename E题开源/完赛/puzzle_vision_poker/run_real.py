"""Safe real-camera entry point; PyBullet remains isolated in run_sim.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import main as vision


ROOT = Path(__file__).resolve().parent


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Merge the small production overlay onto the full vision config."""
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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


def load_production_config(config_path: Path | None = None, *, port="", execute=False):
    """Shared effective configuration for planning and read-only camera views."""
    path = Path(config_path) if config_path is not None else ROOT / "production_config.json"
    base = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    overlay = json.loads(path.read_text(encoding="utf-8-sig"))
    config = _deep_merge(base, overlay)
    config.update({"serial_port": port if execute else "", "auto_fallback_enabled": False,
                   "force_execute_on_any_solution": False, "generic_ultimate_fallback_enable": False,
                   "strict_production": True, "approval_workflow": True,
                   "require_approved_session": True, "auto_max_retries": 1})
    config["generic_min_edge_length_mm"] = float(config.get("poker_min_edge_length_mm", 4.0))
    if execute:
        config["execution_backend"] = "dual_grbl_stm32"
        config.setdefault("serial", {})["grbl_port"] = port
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=ROOT / "production_config.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-approved", type=Path)
    parser.add_argument("--expected-plan-id")
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--parent-pid", type=int, default=0)
    options, arguments = parser.parse_known_args(argv)

    port = _port_value(arguments)
    if options.execute:
        parser.error("旧真实发送入口已停用，请使用 --execute-approved <方案路径>")
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
        return load_production_config(config_path, port=port, execute=options.execute)

    vision.load_config = load_safe_config
    previous_argv = sys.argv
    vision_arguments = [argument for argument in arguments if argument != "--execute"]
    sys.argv = [str(ROOT / "run_real.py"), *vision_arguments]
    if options.execute_approved:
        print("[REAL] 连接已绑定方案，等待屏幕确认本次基准；尚未授权运动。")
    elif options.execute:
        print(f"[REAL] Explicit execution enabled on {port}; verify workspace and machine clearance.")
    else:
        print("[REAL] Dry-run only; no serial device will be opened.")
    try:
        if options.execute_approved:
            if not options.session_dir or not options.expected_plan_id:
                parser.error("--execute-approved requires --session-dir and --expected-plan-id")
            from approved_plan import run_session
            return run_session(options.execute_approved, load_safe_config, config_path,
                               options.session_dir, options.parent_pid, options.expected_plan_id)
        return vision.main()
    finally:
        sys.argv = previous_argv
        vision.CONFIG_PATH = previous_config_path
        vision.load_config = previous_loader


if __name__ == "__main__":
    raise SystemExit(main())
