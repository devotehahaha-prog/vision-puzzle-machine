"""Immutable camera plans and a single-owner, single-use execution session.

IPC only carries operator decisions. It never carries replacement coordinates.
The worker holds both ports from handshake through final verification.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import time
import uuid

import cv2
import numpy as np

from dual_serial_executor import CoordinateCalibration, DualExecutionError, DualSerialExecutor, expected_real_piece_count


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            # Windows readers temporarily deny rename; Linux IPC is atomic.
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(.02)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def config_digest(config):
    # Acceptance records change after the witnessed empty run. Geometry,
    # camera, protocol, thresholds and feeds remain bound to the preview.
    ignored = {"calibration_complete", "mechanical_acceptance", "accepted_machine_bounds_mm",
               "puzzle_acceptance", "saved_at", "serial_port"}
    return digest({k: v for k, v in config.items() if k not in ignored})


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_digest():
    root = Path(__file__).resolve().parent
    return digest({name: file_digest(root / name) for name in
                   ("main.py", "generic_solver.py", "approved_plan.py", "dual_serial_executor.py", "run_real.py")})


def immutable_digest(plan):
    return digest({k: v for k, v in plan.items()
                   if k not in {"content_sha256", "artifact_path", "approval_state", "ready_for_motion"}})


def composite_preview(paper, plan, ppm):
    canvas = np.full_like(paper, (230, 230, 230))
    scale = np.diag([ppm, ppm, 1.0])
    for piece in plan["pieces"]:
        mask = np.zeros(paper.shape[:2], np.uint8)
        contour = np.rint(np.asarray(piece["source_contour_mm"]) * ppm).astype(np.int32)
        cv2.fillPoly(mask, [contour], 255)
        transform = scale @ np.asarray(piece["transform_3x3"]) @ np.linalg.inv(scale)
        moved = cv2.warpPerspective(paper, transform, (paper.shape[1], paper.shape[0]))
        moved_mask = cv2.warpPerspective(mask, transform, (paper.shape[1], paper.shape[0]),
                                         flags=cv2.INTER_NEAREST)
        canvas[moved_mask > 0] = moved[moved_mask > 0]
    return canvas


def seal_plan(payload, original, paper, config, output_dir):
    import main as vision
    plan = copy.deepcopy(payload)
    # Only pattern evidence can be reviewed by a person. Never remove a
    # geometry, path, solver timeout, or workspace failure.
    pattern_reasons = [r for r in plan["failure_reasons"]
                       if r.startswith(("花纹相似度", "未得到相邻边花纹"))]
    geometry_reasons = [r for r in plan["failure_reasons"] if r not in pattern_reasons]
    plan.update({"schema_version": 2, "plan_id": uuid.uuid4().hex,
                 "puzzle_profile": config.get("puzzle_profile", "poker"),
                 "code_sha256": code_digest(),
                 "camera_shape": list(original.shape),
                 "config_sha256": config_digest(config), "geometry_reasons": geometry_reasons,
                 "pattern_review_required": bool(pattern_reasons),
                 "approval_state": "blocked" if geometry_reasons else "review_required",
                 "ready_for_motion": False})
    directory = Path(output_dir) / "plans" / plan["plan_id"]
    directory.mkdir(parents=True)
    vision.write_image(directory / "source.png", original)
    vision.write_image(directory / "corrected.png", paper)
    if not geometry_reasons:
        vision.write_image(directory / "assembly.png", composite_preview(paper, plan, float(config["pixels_per_mm"])))
    plan["assets"] = {name: file_digest(directory / name)
                      for name in ("source.png", "corrected.png", "assembly.png") if (directory / name).exists()}
    plan["content_sha256"] = immutable_digest(plan)
    plan["artifact_path"] = str((directory / "plan.json").resolve())
    atomic_json(directory / "plan.json", plan)
    return plan


def validate_plan(path, config, expected_id=None):
    path = Path(path).resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    if expected_id is not None and plan.get("plan_id") != expected_id:
        raise RuntimeError("方案编号已变化，请重新预览")
    if plan.get("schema_version") != 2 or immutable_digest(plan) != plan.get("content_sha256"):
        raise RuntimeError("方案内容被替换或损坏，请重新生成")
    if plan.get("config_sha256") != config_digest(config):
        raise RuntimeError("标定或运行配置已变化，请重新生成方案")
    if plan.get("code_sha256") != code_digest():
        raise RuntimeError("程序版本已变化，请重新生成方案")
    expected = expected_real_piece_count(config)
    if (plan.get("geometry_reasons") or not plan.get("all_paths_ok")
            or plan.get("puzzle_profile", "poker") != config.get("puzzle_profile", "poker")
            or not plan.get("all_matches_ok") or len(plan.get("pieces", [])) != expected
            or plan.get("approval_state") != "review_required"):
        raise RuntimeError("几何或路径检查未通过，不能人工放行")
    for name, checksum in plan.get("assets", {}).items():
        if Path(name).name != name or file_digest(path.parent / name) != checksum:
            raise RuntimeError("原始图或预览已变化")
    if not all(name in plan.get("assets", {}) for name in ("source.png", "assembly.png", "corrected.png")):
        raise RuntimeError("方案缺少绑定图像")
    return plan


def scene_matches(pieces, plan, config):
    """Compare fresh contours with saved source contours, without solving."""
    expected = expected_real_piece_count(config)
    if len(pieces) != expected or len(plan.get("pieces", [])) != expected:
        return False, f"当前检测到{len(pieces)}片，要求{expected}片"
    ppm = float(config["pixels_per_mm"])
    saved = plan["pieces"]
    assigned = min(itertools.permutations(pieces), key=lambda perm: sum(
        np.linalg.norm(np.asarray(p.center_mm(ppm)) - s["center_mm"])
        for p, s in zip(perm, saved)))
    for p, s in zip(assigned, saved):
        if np.linalg.norm(np.asarray(p.center_mm(ppm)) - s["center_mm"]) > 2.0:
            return False, f"P{s['id']}位置变化超过2mm"
        if abs(p.area_mm2 - s["area_mm2"]) / max(s["area_mm2"], 1e-6) > .12:
            return False, f"P{s['id']}面积变化超过12%"
        # A minimum-area rectangle can switch orientation for an unchanged
        # fragment. Compare the actual contour and pose instead of box sizes.
        actual = p.contour.reshape(-1, 2).astype(float) / ppm
        reference = np.asarray(s["source_contour_mm"], float)
        # Point-to-boundary distance handles unequal contour sample counts.
        distances = [abs(cv2.pointPolygonTest(reference.astype(np.float32), tuple(point), True))
                     for point in actual[::max(1,len(actual)//150)]]
        distances += [abs(cv2.pointPolygonTest(actual.astype(np.float32), tuple(point), True))
                      for point in reference[::max(1,len(reference)//150)]]
        if max(distances, default=999) > 3:
            return False, f"P{s['id']}轮廓或姿态已改变"
        import main as vision
        rotation, _iou, _scale = vision.best_rotation_fit(p, reference, config)
        if abs(rotation) > 3:
            return False, f"P{s['id']}姿态变化超过3度"
    return True, "现场与方案一致"


def fresh_scene_check(capture, plan, config, directory):
    import main as vision
    matrix = vision.load_calibration(config)
    frame, _ = vision.capture_immediate_frame(capture, config)
    results = []
    for index in range(5):
        if index:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("执行前复拍失败")
        paper = vision.warp_paper(frame, matrix, config)
        if list(frame.shape) != plan.get("camera_shape"):
            raise RuntimeError("相机输出分辨率已变化，请重新核对相机标定")
        if config.get("puzzle_profile") == "ordinary":
            _, mask = vision.mask_in_paper(frame, matrix, config)
            pieces, _ = vision.detect_pieces(paper, config, diagnostics=False, mask_override=mask)
        else:
            pieces, _ = vision.detect_pieces(paper, config, diagnostics=False)
        passed, reason = scene_matches(pieces, plan, config)
        results.append({"passed": passed, "reason": reason})
    vision.write_image(directory / "pre_execute.png", frame)
    atomic_json(directory / "scene_check.json", results)
    if sum(r["passed"] for r in results) < 4:
        raise RuntimeError("卡片已移动或画面不稳定，请重新生成预览")
    return matrix


def machine_bounds(plan, config):
    calibration = CoordinateCalibration.from_config(config)
    points = [(0., 0.)] + [calibration.map_point(p) for piece in plan["pieces"] for p in piece["path_mm"]]
    return [[min(p[i] for p in points) for i in (0,1)],
            [max(p[i] for p in points) for i in (0,1)]]


def run_session(plan_path, config_loader, config_path, session_dir, parent_pid, expected_plan_id):
    """Wait for explicit SPI decisions, while retaining the same serial owner."""
    import main as vision
    directory = Path(session_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = config_loader()
    plan_path = Path(plan_path).resolve()
    if not expected_plan_id:
        raise RuntimeError("缺少屏幕确认的方案编号")
    plan = validate_plan(plan_path, config, expected_plan_id)
    # Exclusive persistent reservation: a crash/restart cannot replay it.
    with (plan_path.parent / "consumed.json").open("x", encoding="utf-8") as used:
        json.dump({"session_dir": str(directory), "pid": os.getpid()}, used)
    # Direct production mode uses the saved per-axis calibration and the
    # current-session origin confirmation, without a preliminary empty path.
    empty = (config.get("require_empty_run", True) is not False
             and not bool(config.get("calibration_complete")))
    executor = DualSerialExecutor(config, empty_run=empty)
    session = executor.session_id
    sequence = 0
    cap = None
    old_handlers = {}
    trace_file = (directory / "serial_events.jsonl").open("a", encoding="utf-8")

    def trace(device, direction, data):
        trace_file.write(json.dumps({"time": time.time(), "device": device,
                                    "direction": direction, "data": data}, ensure_ascii=False) + "\n")
        trace_file.flush()

    def publish(stage, message, **extra):
        atomic_json(directory / "status.json", {
            "stage": stage, "message": message, "session_id": session,
            "plan_id": plan["plan_id"], "empty_run": executor.empty_run,
            "sequence": sequence, "updated_at": time.time(), **extra})

    def stop(signum, _frame):
        executor.stop_both(f"signal {signum}")
        raise DualExecutionError("操作者停止，本次批准失效")

    def check_parent():
        if parent_pid and os.getppid() != parent_pid:
            raise RuntimeError("屏幕程序退出，本次连接确认失效")

    executor.session_guard = check_parent

    def wait_decision(stage, message):
        nonlocal sequence
        sequence += 1
        publish(stage, message)
        deadline = time.monotonic() + 300
        last_poll = 0.
        while time.monotonic() < deadline:
            if parent_pid and os.getppid() != parent_pid:
                raise RuntimeError("屏幕程序已退出，本次批准失效")
            command_path = directory / "command.json"
            if command_path.exists():
                command = json.loads(command_path.read_text(encoding="utf-8"))
                command_path.unlink()
                if (command.get("session_id") != session or command.get("plan_id") != plan["plan_id"]
                        or command.get("sequence") != sequence or command.get("action") != "confirm"):
                    raise RuntimeError("确认消息与当前方案或连接不一致")
                return
            if time.monotonic() - last_poll > 1:
                executor.preflight(allow_hold=True)
                last_poll = time.monotonic()
            time.sleep(.1)
        raise RuntimeError("确认超时，本次方案失效")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, stop)
        executor.open()
        executor.grbl.trace = lambda direction, data: trace("grbl", direction, data)
        executor.stm32.trace = lambda direction, data: trace("stm32", direction, data)
        executor.preflight(allow_hold=True)
        for phase in (["empty", "real"] if empty else ["real"]):
            wait_decision("await_origin", "确认：人工左上基准、MPos=0/0/0、Z安全、磁铁关；取消旧任务并启动" )
            config = config_loader()
            validate_plan(plan_path, config, plan["plan_id"])
            executor.config = config
            cap = vision.open_camera(config, None)
            publish("checking_scene", "执行前复拍，检查卡片未移动")
            matrix = fresh_scene_check(cap, plan, config, directory)
            validate_plan(plan_path, config_loader(), plan["plan_id"])
            executor.confirm_origin(plan["plan_id"], session)
            approved = copy.deepcopy(plan)
            approved.update(approval_state="approved", ready_for_motion=True)
            atomic_json(directory / f"approved_{phase}.json", approved)
            publish("running", "整路径空载检查" if phase == "empty" else f"正在自动拼合{len(plan['pieces'])}片")
            executor.execute(approved)
            if phase == "empty":
                cap.release(); cap = None
                bounds = machine_bounds(plan, config)
                atomic_json(directory / "empty_run.json", {"commanded_path_completed": True,
                            "bounds_mm": bounds, "plan_id": plan["plan_id"]})
                wait_decision("await_empty_accept", "空载已结束：确认全程可达、无撞限或卡住，完成机械验收")
                # Validate again before writing an acceptance record.
                validate_plan(plan_path, config_loader(), plan["plan_id"])
                overlay = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
                overlay.update(calibration_complete=True, accepted_machine_bounds_mm=bounds,
                               mechanical_acceptance={"plan_id": plan["plan_id"],
                               "config_sha256": plan["config_sha256"], "session_id": session,
                               "operator_confirmed": True, "time": time.time()})
                atomic_json(config_path, overlay)
                executor.empty_run = False
                publish("empty_accepted", "机械空载验收已记录，准备同方案实跑")
            else:
                publish("verifying", "动作完成，正在复拍检查目标与下半区遗漏")
                previous_output = vision.OUTPUT_DIR
                try:
                    vision.OUTPUT_DIR = directory
                    frame = vision.capture_post_motion_frame(cap, config)
                    report = vision.verify_final_frame(frame, matrix, config, plan_path)
                finally:
                    vision.OUTPUT_DIR = previous_output
                if report.get("status") != "pass":
                    raise RuntimeError("动作已结束，复拍未通过；查看逐片误差报告")
                wait_decision("await_physical_accept", f"复拍通过：确认{len(plan['pieces'])}片无掉落、拖纸或重叠，实物与预览一致")
                history_path = Path(config_path).parent / "puzzle_acceptance.json"
                history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {"runs": []}
                history["runs"].append({"plan_id": plan["plan_id"], "session_id": session,
                                        "config_sha256": plan["config_sha256"], "passed": True,
                                        "evidence": str(directory), "time": time.time()})
                consecutive = 0
                for run in reversed(history["runs"]):
                    if not run.get("passed") or run.get("config_sha256") != plan["config_sha256"]:
                        break
                    consecutive += 1
                history["consecutive_passes"] = consecutive
                history["complete"] = consecutive >= 2
                atomic_json(history_path, history)
                publish("complete", f"完成：连续{consecutive}轮通过", consecutive_passes=consecutive)
        return 0
    except BaseException as exc:
        if not executor.locked:
            executor.stop_both(str(exc))
        stopped = executor.verify_stopped()
        publish("failed", str(exc) + "；" + executor.stop_summary(),
                stop_confirmed=stopped, stop_details=executor.stop_details,
                failed_action=executor.failure_action)
        history_path = Path(config_path).parent / "puzzle_acceptance.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {"runs": []}
        history["runs"].append({"plan_id": plan["plan_id"], "passed": False, "reason": str(exc), "time": time.time()})
        history.update(consecutive_passes=0, complete=False)
        atomic_json(history_path, history)
        raise
    finally:
        if cap is not None:
            cap.release()
        executor.close()
        trace_file.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
