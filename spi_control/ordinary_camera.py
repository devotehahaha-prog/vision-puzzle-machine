"""Ordinary four-piece USB camera planning and explicitly approved execution.

The physical camera/calibration are shared with the formal camera profile; piece
segmentation and rectangle-solving settings belong to the ordinary project.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import uuid


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def engine_project(project):
    return Path(project).resolve().with_name("puzzle_vision_poker")


def load_config(project, *, for_execution=False):
    """Read current shared calibration without copying or editing its file."""
    engine = engine_project(project)
    config = read_json(engine / "config.json")
    config.update(read_json(engine / "production_config.json"))
    ordinary = read_json(Path(project) / "real_config.json")
    shared_limits = {"generic_target_piece_clearance_mm", "target_corresponding_vertex_limit_mm"}
    piece_keys = {"min_piece_area_mm2", "max_piece_area_mm2", "min_solidity",
                  "pick_point_min_margin_mm", "morphology_kernel_mm", "raw_morphology_kernel_px",
                  "target_rectangle_size_mm", "rotation_match_pixels_per_mm",
                  "rotation_search_step_deg", "rotation_refine_step_deg"}
    for key, value in ordinary.items():
        if key not in shared_limits and (key.startswith(("generic_", "white_piece_")) or key in piece_keys):
            config[key] = value
    config.update({
        "puzzle_mode": "generic_geometry", "segmentation_mode": "white_piece",
        "expected_piece_count": 4, "max_pieces": 4,
        "auto_board_calibration": False, "source_region": "bottom", "target_region": "top",
        "generic_target_top_mm": None,
        "poker_pattern_verification": False, "enforce_pattern_gate": False,
        "strict_production": True, "auto_fallback_enabled": False,
        "force_execute_on_any_solution": False, "generic_ultimate_fallback_enable": False,
        "approval_workflow": False, "execution_backend": "preview_only", "serial_port": "",
        "require_approved_session": True, "ordinary_camera_preview": True,
    })
    if not config.get("camera_calibration", {}).get("perspective_matrix"):
        raise ValueError("普通拼图需要已保存的共用相机标定，请先检查角点标定")
    if for_execution:
        config.update(puzzle_profile="ordinary", execution_backend="dual_grbl_stm32", approval_workflow=True,
                      ordinary_entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        state_path = Path(project) / "ordinary_execution_state.json"
        if state_path.is_file():
            state = read_json(state_path)
            for key in ("calibration_complete", "accepted_machine_bounds_mm", "mechanical_acceptance"):
                if key in state:
                    config[key] = state[key]
    return config


def load_runtime(project):
    """Use the maintained geometric engine, never the legacy simulation module."""
    engine = engine_project(project)
    for name in ("generic_solver", "orthogonal_path_planner", "serial_transport", "dual_serial_executor", "edge_matcher"):
        loaded = sys.modules.get(name)
        if loaded and Path(loaded.__file__).resolve().parent != engine:
            raise RuntimeError("视觉模块路径冲突，请在独立进程启动普通相机")
    if str(engine) not in sys.path:
        sys.path.insert(0, str(engine))
    name = "_ordinary_geometry_engine"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, engine / "main.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    elif Path(module.__file__).resolve().parent != engine:
        raise RuntimeError("普通相机引擎路径冲突")
    return module, load_config(project)


def output_dir(project):
    return Path(project) / "output_camera"


def invalidate(output, reason):
    atomic_json(Path(output) / "plan.json", {
        "generated_at_epoch": time.time(), "mode": "ordinary_camera",
        "approval_state": "preview_only", "ready_for_motion": False,
        "preview_ready": False, "piece_count": 0, "pieces": [], "failure_reasons": [str(reason)],
    })


def direct_search_config(config):
    """Search tree/split candidates before expensive closure refinement.

    Excellent-solution thresholds only control early termination. They are set
    to the existing strict acceptance limits; no acceptance gate is relaxed.
    The original closure search remains available if this first pass fails.
    """
    first = dict(config, generic_enable_closure_refine=False)
    for quality, default in (("fill_error_ratio", .06), ("boundary_gap_ratio", .08),
                             ("overlap_ratio", .008)):
        limit = float(config.get("generic_max_" + quality, default))
        limit = min(limit, float(config.get("generic_motion_max_" + quality, limit)))
        first["generic_excellent_" + quality] = limit
    return first


@contextlib.contextmanager
def ordinary_search_order(vision):
    """Process-local search ordering; retain every independent-frame gate."""
    original = vision._auto_detect_and_solve

    def solve(frame, config, matrix):
        try:
            return original(frame, direct_search_config(config), matrix)
        except (ValueError, RuntimeError) as first_error:
            print("[普通搜索] 直接拼缝搜索未通过，继续原有闭环搜索：", first_error, flush=True)
            return original(frame, config, matrix)

    vision._auto_detect_and_solve = solve
    try:
        yield
    finally:
        vision._auto_detect_and_solve = original


def generate_plan(project, *, prepare_execution=False):
    """Capture and solve only; publish a fresh preview after every gate passes."""
    output = output_dir(project)
    invalidate(output, "本次实拍方案生成中")
    for name in ("detection_preview.jpg", "detection_failure.json"):
        (output / name).unlink(missing_ok=True)
    run_id = uuid.uuid4().hex
    run_dir = output / "runs" / run_id
    run_dir.mkdir(parents=True)
    capture = vision = previous_output = None
    try:
        vision, config = load_runtime(project)
        previous_output = vision.OUTPUT_DIR
        vision.OUTPUT_DIR = run_dir
        capture = vision.open_camera(config, None)
        frame, _ = vision.capture_immediate_frame(capture, config)
        # This also rejects camera/profile mismatch before any geometry solving.
        from calibration_preview import render_calibration_preview
        recognition, info = render_calibration_preview(frame, vision, config, "mask")
        vision.write_image(run_dir / "recognition.jpg", recognition)
        atomic_json(run_dir / "recognition.json", info)
        with ordinary_search_order(vision):
            _, paper, _, pieces, paths = vision._stable_auto_detect_and_solve(
                capture, frame, config, vision.load_calibration(config))
        plan = read_json(run_dir / "plan.json")
        if len(pieces) != 4 or len(paths) != 4 or not plan.get("ready_for_motion") or plan.get("failure_reasons"):
            raise RuntimeError("四片普通拼图未通过几何或路径检查")
        from approved_plan import composite_preview
        vision.write_image(run_dir / "assembly.png", composite_preview(paper, plan, float(config["pixels_per_mm"])))
        plan.update({"mode": "ordinary_camera", "plan_id": run_id,
                     "generated_at_epoch": time.time(), "approval_state": "preview_only",
                     "preview_ready": True, "ready_for_motion": False,
                     "artifact_path": str((run_dir / "plan.json").resolve()),
                     "assembly_path": str((run_dir / "assembly.png").resolve()),
                     "recognition_path": str((run_dir / "recognition.jpg").resolve()),
                     "execution_backend": "preview_only"})
        if prepare_execution:
            from approved_plan import seal_plan, immutable_digest, validate_plan
            execution_config = load_config(project, for_execution=True)
            plan["execution_backend"] = "dual_grbl_stm32"
            plan = seal_plan(plan, vision.read_image(run_dir / "original.jpg"), paper,
                             execution_config, run_dir)
            sealed_dir = Path(plan["artifact_path"]).parent
            plan["assembly_path"] = str(sealed_dir / "assembly.png")
            plan["content_sha256"] = immutable_digest(plan)
            atomic_json(sealed_dir / "plan.json", plan)
            shutil.copy2(run_dir / "stability_report.json", sealed_dir / "stability_report.json")
            validate_plan(sealed_dir / "plan.json", execution_config, plan["plan_id"])
        atomic_json(run_dir / "plan.json", plan)
        atomic_json(output / "plan.json", plan)
        print("[普通实拍] 四片方案已生成：", plan["artifact_path"],
              "，待本轮基准确认" if prepare_execution else "，仅供预览", flush=True)
        return plan
    except Exception as error:
        invalidate(output, error)
        # Never leave an executable intermediate plan after an interrupted gate.
        invalidate(run_dir, error)
        for name in ("detection_preview.jpg", "detection_failure.json"):
            if (run_dir / name).is_file():
                shutil.copy2(run_dir / name, output / name)
        if not (output / "detection_preview.jpg").is_file() and (run_dir / "recognition.jpg").is_file():
            shutil.copy2(run_dir / "recognition.jpg", output / "detection_preview.jpg")
            atomic_json(output / "detection_failure.json", {"time": time.time(), "reason": str(error)})
        raise
    finally:
        if capture is not None:
            capture.release()
        if previous_output is not None:
            vision.OUTPUT_DIR = previous_output


def main():
    parser = argparse.ArgumentParser(description="普通拼图实拍、四片识别、绑定方案的实机执行")
    parser.add_argument("--project", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-execution", action="store_true")
    mode.add_argument("--execute-approved", type=Path)
    parser.add_argument("--expected-plan-id")
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()
    try:
        project = args.project.resolve()
        if args.execute_approved:
            if not args.expected_plan_id or not args.session_dir:
                parser.error("实机执行必须绑定方案编号和本次会话")
            load_runtime(project)
            from approved_plan import run_session
            state_path = project / "ordinary_execution_state.json"
            if not state_path.exists():
                atomic_json(state_path, {})
            return run_session(args.execute_approved, lambda: load_config(project, for_execution=True),
                               state_path, args.session_dir, args.parent_pid, args.expected_plan_id)
        generate_plan(project, prepare_execution=args.prepare_execution)
    except Exception as error:
        print("[普通实拍失败]", error, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
