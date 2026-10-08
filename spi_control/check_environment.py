"""Inspect interpreters and device nodes without opening any hardware."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def inspect_python(executable, modules):
    script = """
import importlib, json, sys
result = {'executable': sys.executable, 'version': sys.version.split()[0], 'modules': {}}
for name in json.loads(sys.argv[1]):
    try:
        module = importlib.import_module(name)
        result['modules'][name] = {'ok': True, 'path': getattr(module, '__file__', '')}
    except Exception as error:
        result['modules'][name] = {'ok': False, 'error': str(error)}
print(json.dumps(result, ensure_ascii=False))
"""
    try:
        process = subprocess.run([str(executable), "-E", "-c", script, json.dumps(modules)],
                                 capture_output=True, text=True, timeout=25)
        if process.returncode:
            return {"ok": False, "error": process.stderr.strip()}
        result = json.loads(process.stdout)
        result["ok"] = all(item["ok"] for item in result["modules"].values())
        return result
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": str(error)}


def inspect_device(path):
    node = Path(path)
    return {"path": str(node), "exists": node.exists(),
            "resolved": str(node.resolve()),
            "read_write": os.access(node, os.R_OK | os.W_OK)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup", action="store_true",
                        help="device warnings do not block opening the screen")
    args = parser.parse_args(argv)
    from start_page import find_projects, find_venv_python, find_driver_dir
    from manual_control import load_config
    report = {"ok": False}
    try:
        cfg = load_config()
        projects = find_projects()
        report["driver"] = str(find_driver_dir())
        report["ui"] = inspect_python("/usr/bin/python3", ["serial", "PIL", "spidev", "smbus2"])
        report["vision"] = inspect_python(find_venv_python(projects["contest"]), ["serial", "cv2", "numpy"])
        report["devices"] = {name: inspect_device(path) for name, path in {
            "grbl": cfg["grbl_port"], "aux": cfg["aux_port"],
            "spi": "/dev/spidev0.0", "touch_i2c": "/dev/i2c-1"}.items()}
        report["usb_ch340"] = []
        for path in Path("/sys/bus/usb/devices").glob("*/idVendor"):
            try:
                if path.read_text().strip().lower() == "1a86":
                    report["usb_ch340"].append(str(path.parent))
            except OSError:
                pass
        report["environment_ok"] = report["ui"]["ok"] and report["vision"]["ok"]
        report["devices_ok"] = all(value["read_write"] for value in report["devices"].values())
        report["ok"] = report["environment_ok"] and report["devices_ok"]
        if not report["devices"]["grbl"]["exists"]:
            print("[设备] 未检测到写字机串口。检查写字机电源/USB数据线；若lsusb也没有1a86，重装Python无效。")
        for name, device in report["devices"].items():
            if device["exists"] and not device["read_write"]:
                print(f"[权限] {name}: 当前用户无法读写 {device['path']}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["environment_ok"]:
            return 1
        return 0 if args.startup or report["devices_ok"] else 2
    except Exception as error:
        print("[环境检查失败] " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
