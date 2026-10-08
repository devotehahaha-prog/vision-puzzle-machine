#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera helper for the SPI control panel. Run with the contest venv."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PREVIEW_W, PREVIEW_H = 768, 348
TUNE_PREVIEW_W, TUNE_PREVIEW_H = 360, 352


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_vision(project: Path):
    project = project.resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    import main as vision
    return vision


def load_run_sim(project: Path):
    project = project.resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    import run_sim
    return run_sim


def effective_config(vision, run_sim, vision_kind: str):
    if vision_kind == "poker":
        return run_sim.simulation_config(vision)
    config = vision.load_config()
    sim_path = Path(run_sim.ROOT) / "sim_config.json"
    if sim_path.is_file():
        config.update(json.loads(sim_path.read_text(encoding="utf-8")))
    return config


def find_sim_root(run_sim) -> Path:
    candidates = [Path(run_sim.DEFAULT_SIM_ROOT)]
    root = Path(run_sim.ROOT).resolve()
    for parent in [root, *root.parents]:
        candidates.append(parent / "pybullet" / "puzzle_device")
    for path in candidates:
        if (path / "simulator.py").is_file():
            return path
    raise FileNotFoundError("simulator.py not found near " + str(run_sim.ROOT))


def open_sim_camera(project: Path, vision_kind: str, gui: bool = True):
    run_sim = load_run_sim(project)
    vision, simulator, camera_class = run_sim.load_modules(find_sim_root(run_sim))
    config = effective_config(vision, run_sim, vision_kind)
    sim = simulator.PuzzleGantrySim(gui=gui, realtime=False, scene="vision")
    if vision_kind == "poker":
        sim.spawn_pieces(run_sim.make_scene(simulator))
    else:
        sim.spawn_pieces(run_sim.make_scene(vision, simulator))
    camera = camera_class(sim.cid, float(config["pixels_per_mm"]))
    return vision, run_sim, config, sim, camera


def letterbox_rect(src_w, src_h, dst_w, dst_h):
    if src_w <= 0 or src_h <= 0:
        return (0, 0, dst_w, dst_h)
    scale = min(dst_w / float(src_w), dst_h / float(src_h))
    w = max(1, int(round(src_w * scale)))
    h = max(1, int(round(src_h * scale)))
    return ((dst_w - w) // 2, (dst_h - h) // 2, w, h)


def encode_jpeg(image) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return encoded.tobytes()


def write_preview(frame, dest: Path, dst_w=PREVIEW_W, dst_h=PREVIEW_H):
    h, w = frame.shape[:2]
    lx, ly, lw, lh = letterbox_rect(w, h, dst_w, dst_h)
    canvas = np.full((dst_h, dst_w, 3), (21, 23, 26), np.uint8)
    resized = cv2.resize(frame, (lw, lh), interpolation=cv2.INTER_AREA)
    canvas[ly:ly + lh, lx:lx + lw] = resized
    atomic_write_bytes(dest, encode_jpeg(canvas))
    return [lx, ly, lw, lh]


def write_status(path: Path, **payload) -> None:
    payload["heartbeat"] = time.time()
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False))


def show_window(title: str, image) -> None:
    del title, image


def draw_points(frame, points):
    display = frame.copy()
    pts = [(int(round(x)), int(round(y))) for x, y in points]
    for index, point in enumerate(pts):
        cv2.circle(display, point, 10, (0, 0, 255), -1)
        cv2.putText(display, str(index + 1), (point[0] + 12, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if len(pts) >= 2:
        cv2.polylines(display, [np.asarray(pts, np.int32)], len(pts) == 4, (0, 255, 255), 2)
    return display


def paper_from_sim(frame, camera, vision, config):
    if getattr(camera, "matrix", None) is not None:
        try:
            return vision.warp_paper(frame, camera.matrix, config)
        except Exception:
            return frame
    return frame


def piece_tune_view(vision, frame, config, params, vision_kind="", camera=None):
    work = dict(config)
    work.update(params)
    paper = paper_from_sim(frame, camera, vision, work) if camera is not None else frame
    use_poker = vision_kind == "poker" or "poker_otsu_min_value" in params or str(
        work.get("segmentation_mode", "")) == "poker_v"
    work["segmentation_mode"] = "poker_v" if use_poker else "white_piece"
    _, mask = vision.detect_pieces(paper, work)
    color = paper if paper.ndim == 3 else cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)
    overlay = color.copy()
    overlay[mask > 0] = (0, 220, 0)
    return cv2.addWeighted(color, 0.55, overlay, 0.45, 0)


def board_tune_view(vision, frame, config, params):
    work = dict(config)
    work.update(params)
    work["auto_board_calibration"] = True
    work["segmentation_mode"] = "white_piece"
    display = frame.copy()
    points = vision.detect_dark_board_points(frame, work)
    if points is None:
        cv2.putText(display, "A4 not found", (24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return display
    pts = np.asarray(points, np.int32).reshape(-1, 1, 2)
    cv2.polylines(display, [pts], True, (0, 255, 255), 3)
    for index, (x, y) in enumerate(np.asarray(points)):
        cv2.circle(display, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(display, str(index + 1), (int(x) + 10, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return display


def tune_view(vision, frame, config, params, vision_kind="", camera=None, group="piece"):
    if group == "board":
        return board_tune_view(vision, frame, config, params)
    return piece_tune_view(vision, frame, config, params, vision_kind, camera)


def cmd_live(args):
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    status_path = workdir / "status.json"
    preview_path = workdir / "spi_preview.jpg"
    stop_path = workdir / "stop"
    last_path = workdir / "last_frame.jpg"
    vision = load_vision(args.project)
    capture = None
    try:
        config = vision.load_config()
        capture = vision.open_camera(config, None)
    except Exception as error:
        write_status(status_path, ok=False, error=str(error), cam_size=[0, 0],
                     letterbox=[0, 0, PREVIEW_W, PREVIEW_H])
        placeholder = np.full((PREVIEW_H, PREVIEW_W, 3), (40, 40, 46), np.uint8)
        cv2.putText(placeholder, "USB camera failed", (24, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 210), 2)
        atomic_write_bytes(preview_path, encode_jpeg(placeholder))
        print("usb camera failed:", error, flush=True)
        while not stop_path.exists():
            time.sleep(0.2)
        return
    last_write = 0.0
    last_full_write = 0.0
    try:
        while not stop_path.exists():
            ok, frame = capture.read()
            if not ok or frame is None:
                write_status(status_path, ok=False, error="摄像头读取失败",
                             cam_size=[0, 0], letterbox=[0, 0, PREVIEW_W, PREVIEW_H])
                time.sleep(0.05)
                continue
            now = time.time()
            if now - last_write >= 0.1:
                cam_h, cam_w = frame.shape[:2]
                box = write_preview(frame, preview_path, PREVIEW_W, PREVIEW_H)
                if now - last_full_write >= 1.0:
                    atomic_write_bytes(last_path, encode_jpeg(frame))
                    last_full_write = now
                write_status(status_path, ok=True, error="", cam_size=[cam_w, cam_h],
                             letterbox=list(box), mode="live", source="usb")
                last_write = now
            time.sleep(0.001)
    finally:
        capture.release()


def cmd_real_preview(args):
    """Preview/mark the physical USB camera without starting PyBullet."""
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    overlay_path = workdir / "overlay.json"
    status_path = workdir / "status.json"
    preview_path = workdir / "spi_preview.jpg"
    stop_path = workdir / "stop"
    last_path = workdir / "last_frame.jpg"
    capture = None
    try:
        if args.vision == "ordinary":
            from ordinary_camera import load_runtime
            vision, config = load_runtime(args.project)
        else:
            vision = load_vision(args.project)
            config = vision.load_config()
            production = args.project / "production_config.json"
            if production.is_file():
                from run_real import load_production_config
                config = load_production_config(production)
        capture = vision.open_camera(config, None)
        while not stop_path.exists():
            ok, frame = capture.read()
            if not ok or frame is None:
                preview_path.unlink(missing_ok=True)
                write_status(status_path, ok=False, error="摄像头读取失败",
                             cam_size=[0, 0], letterbox=[0, 0, PREVIEW_W, PREVIEW_H], source="usb")
                time.sleep(0.05)
                continue
            overlay = load_json(overlay_path, {})
            mode = str(overlay.get("mode") or args.mode)
            points = overlay.get("points") or []
            params = overlay.get("params") or {}
            group = str(overlay.get("group") or "piece")
            metadata = {}
            if mode == "calib_preview":
                from calibration_preview import render_calibration_preview
                display, metadata = render_calibration_preview(
                    frame, vision, config, str(overlay.get("calibration_view") or "raw"))
                dst_w, dst_h = PREVIEW_W, PREVIEW_H
            elif mode == "tune":
                if args.vision == "ordinary" and group == "piece":
                    from calibration_preview import render_calibration_preview
                    display, metadata = render_calibration_preview(frame, vision, dict(config, **params), "mask")
                else:
                    display = tune_view(vision, frame, config, params, args.vision, None, group)
                dst_w, dst_h = TUNE_PREVIEW_W, TUNE_PREVIEW_H
            else:
                display = draw_points(frame, points)
                dst_w, dst_h = PREVIEW_W, PREVIEW_H
            box = write_preview(display, preview_path, dst_w, dst_h)
            atomic_write_bytes(last_path, encode_jpeg(frame))
            write_status(status_path, ok=True, error="", cam_size=[frame.shape[1], frame.shape[0]],
                         letterbox=list(box), mode=mode, source="usb", **metadata)
            time.sleep(0.05)
    except Exception as error:
        preview_path.unlink(missing_ok=True)
        write_status(status_path, ok=False, error=str(error), cam_size=[0, 0],
                     letterbox=[0, 0, PREVIEW_W, PREVIEW_H], source="usb")
        print("real camera preview failed:", error, flush=True)
        while not stop_path.exists():
            time.sleep(0.2)
    finally:
        if capture is not None:
            capture.release()


def cmd_preview(args):
    if args.real_camera:
        cmd_real_preview(args)
        return
    if args.mode == "live":
        cmd_live(args)
        return
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    overlay_path = workdir / "overlay.json"
    status_path = workdir / "status.json"
    preview_path = workdir / "spi_preview.jpg"
    stop_path = workdir / "stop"
    last_path = workdir / "last_frame.jpg"
    overlay0 = load_json(overlay_path, {})
    vision_kind = str(overlay0.get("vision") or args.vision)
    sim = None
    try:
        vision, _run_sim, config, sim, camera = open_sim_camera(args.project, vision_kind, gui=True)
    except Exception as error:
        write_status(status_path, ok=False, error=str(error), cam_size=[0, 0],
                     letterbox=[0, 0, PREVIEW_W, PREVIEW_H])
        placeholder = np.full((PREVIEW_H, PREVIEW_W, 3), (40, 40, 46), np.uint8)
        cv2.putText(placeholder, "sim camera failed", (24, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 210), 2)
        atomic_write_bytes(preview_path, encode_jpeg(placeholder))
        print("sim camera failed:", error, flush=True)
        while not stop_path.exists():
            time.sleep(0.2)
        return

    last_sig = None
    last_status = 0.0
    frozen = None
    try:
        while not stop_path.exists():
            overlay = load_json(overlay_path, {})
            mode = str(overlay.get("mode") or args.mode)
            points = overlay.get("points") or []
            params = overlay.get("params") or {}
            vision_kind = str(overlay.get("vision") or vision_kind)
            group = str(overlay.get("group") or "piece")
            sig = (mode, vision_kind, group, json.dumps(points, ensure_ascii=False),
                   json.dumps(params, sort_keys=True, ensure_ascii=False))
            if frozen is None or sig != last_sig:
                if frozen is None:
                    sim.step()
                    frozen = camera.capture()
                    atomic_write_bytes(last_path, encode_jpeg(frozen))
                last_sig = sig
                cam_w, cam_h = frozen.shape[1], frozen.shape[0]
                if mode == "tune":
                    display = tune_view(vision, frozen, config, params, vision_kind, camera, group)
                    write_preview(display, preview_path, TUNE_PREVIEW_W, TUNE_PREVIEW_H)
                    box = letterbox_rect(cam_w, cam_h, TUNE_PREVIEW_W, TUNE_PREVIEW_H)
                else:
                    display = draw_points(frozen, points)
                    write_preview(display, preview_path, PREVIEW_W, PREVIEW_H)
                    box = letterbox_rect(cam_w, cam_h, PREVIEW_W, PREVIEW_H)
                write_status(status_path, ok=True, error="", cam_size=[cam_w, cam_h],
                             letterbox=list(box), mode=mode, source="sim")
                last_status = time.time()
            elif time.time() - last_status >= 1.0:
                cam_h, cam_w = frozen.shape[:2]
                dst = (TUNE_PREVIEW_W, TUNE_PREVIEW_H) if mode == "tune" else (PREVIEW_W, PREVIEW_H)
                write_status(status_path, ok=True, error="", cam_size=[cam_w, cam_h],
                             letterbox=list(letterbox_rect(cam_w, cam_h, *dst)),
                             mode=mode, source="sim")
                last_status = time.time()
            time.sleep(0.05)
    finally:
        sim.close()


def cmd_save_calib(args):
    vision = load_vision(args.project)
    run_sim = load_run_sim(args.project)
    config = effective_config(vision, run_sim, args.vision)
    points = np.asarray(json.loads(args.points), np.float32)
    last_frame = args.workdir / "last_frame.jpg"
    frame = vision.read_image(last_frame) if last_frame.is_file() else None
    if frame is None:
        _vision, _run, config, sim, camera = open_sim_camera(args.project, args.vision, gui=False)
        try:
            frame = camera.capture()
        finally:
            sim.close()
    source, matrix = vision.calibration_matrix(points, config)
    ratio = vision.calibration_area_ratio(source, (frame.shape[1], frame.shape[0]))
    minimum = float(config.get("min_calibration_area_ratio", 0.06))
    if ratio < minimum:
        raise RuntimeError("点击区域只占画面 %.1f%%，请点完整 A4 外角" % (ratio * 100.0))
    out = args.workdir / "sim_calibration.json"
    payload = {
        "image_points": source.astype(float).tolist(),
        "perspective_matrix": matrix.astype(float).tolist(),
        "image_size": [int(frame.shape[1]), int(frame.shape[0])],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "sim",
        "note": "practice only; simulation uses VirtualCamera.matrix and does not load this file",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    production = args.project / "production_config.json"
    if production.is_file():
        production_data = json.loads(production.read_text(encoding="utf-8-sig"))
        backup = production.with_suffix(production.suffix + ".bak")
        atomic_write_text(backup, production.read_text(encoding="utf-8"))
        production_data["camera_corners"] = source.astype(float).tolist()
        production_data["camera_calibration_image_size"] = [int(frame.shape[1]), int(frame.shape[0])]
        production_data["calibration_complete"] = False
        production_data["camera_calibration"] = {
            "perspective_matrix": matrix.astype(float).tolist(),
            "image_size": [int(frame.shape[1]), int(frame.shape[0])],
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        production_data["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state = dict(production_data.get("calibration_state", {}))
        state["camera"] = True
        production_data["calibration_state"] = state
        atomic_write_text(production, json.dumps(production_data, ensure_ascii=False, indent=2) + "\n")
        print("updated", production, flush=True)
    preview = vision.warp_paper(frame, matrix, config)
    vision.write_image(args.workdir / "sim_calibration_preview.jpg", preview)
    print("saved", out, flush=True)


def cmd_save_config(args):
    ordinary_real = getattr(args, "vision", "") == "ordinary"
    if ordinary_real:
        path = args.project / "real_config.json"
    else:
        run_sim = load_run_sim(args.project)
        path = Path(run_sim.ROOT) / "sim_config.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    params = json.loads(args.params)
    from layout import LIGHTING_KEYS
    if not isinstance(params, dict) or set(params) - LIGHTING_KEYS:
        raise ValueError("unsupported tuning parameters")
    limits = {
        "morphology_kernel_mm": 8, "poker_morph_close_mm": 8,
        "poker_ncc_min_similarity": 1, "auto_board_min_dark_fraction": 1,
        "auto_board_max_white_fraction": 1, "auto_board_min_area_ratio": 1,
        "auto_board_max_area_ratio": 1,
    }
    for key, value in params.items():
        if key.endswith("threshold_mode"):
            if value not in ("otsu", "fixed"):
                raise ValueError("invalid threshold mode")
            continue
        hi = limits.get(key, 255)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= hi:
            raise ValueError("invalid tuning value: " + key)
    merged = dict(current, **params)
    for prefix in ("white_piece", "poker"):
        if merged.get(prefix + "_otsu_min_value", 0) > merged.get(prefix + "_otsu_max_value", 255):
            raise ValueError("Otsu minimum must not exceed maximum")
    current.update(params)
    atomic_write_text(path, json.dumps(current, ensure_ascii=False, indent=2) + "\n")
    if ordinary_real:
        print("updated ordinary real camera settings", path, flush=True)
        return
    real_config = Path(getattr(args, "real_config", Path(run_sim.ROOT) / "config.json"))
    if real_config.is_file():
        real = json.loads(real_config.read_text(encoding="utf-8-sig"))
        for key, value in params.items():
            real[key] = value
        atomic_write_text(real_config, json.dumps(real, ensure_ascii=False, indent=2) + "\n")
    production = Path(run_sim.ROOT) / "production_config.json"
    if production.is_file():
        production_data = json.loads(production.read_text(encoding="utf-8-sig"))
        backup = production.with_suffix(production.suffix + ".bak")
        atomic_write_text(backup, production.read_text(encoding="utf-8"))
        production_data.update(params)
        production_data["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_text(production, json.dumps(production_data, ensure_ascii=False, indent=2) + "\n")
    print("updated", path, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="SPI camera job")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=HERE / "runtime")
    parser.add_argument("--vision", choices=("ordinary", "poker"), default="ordinary")
    sub = parser.add_subparsers(dest="cmd", required=True)
    preview = sub.add_parser("preview")
    preview.add_argument("--mode", choices=("calib", "tune", "live", "calib_preview"), default="calib")
    preview.add_argument("--real-camera", action="store_true")
    save_c = sub.add_parser("save-calib")
    save_c.add_argument("--points", required=True)
    save_p = sub.add_parser("save-config")
    save_p.add_argument("--params", required=True)
    return parser.parse_args()


def main():
    os.environ.setdefault("DISPLAY", ":0")
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.cmd == "preview":
        cmd_preview(args)
    elif args.cmd == "save-calib":
        cmd_save_calib(args)
    elif args.cmd == "save-config":
        cmd_save_config(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("CAMERA JOB FAILED:", error, file=sys.stderr, flush=True)
        raise SystemExit(1)
