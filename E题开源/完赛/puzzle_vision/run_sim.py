"""Run the existing vision pipeline against a PyBullet camera and gantry."""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：把真实视觉管线接到 PyBullet：虚拟相机拍照 → 规划 → 龙门架执行 → 终检。四片纯色场景。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约46行  load_modules：加载 vision / simulator / VirtualCamera
#   - 约56行  simulation_config：视觉配置 + sim_config.json
#   - 约62行  convert_xy：视觉坐标与仿真坐标互换：[210,297]-点
#   - 约66行  make_scene：按图2模板生成 4 片
#   - 约79行  tasks_from_plan：plan.json → 4 个 MotionTask
#   - 约107行  VisionSession：一次仿真会话：park / run_once / 按钮
#   - 约228行  main：GUI 循环或 --direct 跑一次
# =============================================================================
# =============================================================================
# 【分区】文件说明
# 功能：把真实视觉管线接到 PyBullet 虚拟相机和龙门架，验证拍照→规划→执行→复检。
# 可修改：DEFAULT_SIM_ROOT 指向 pybullet/puzzle_device；输出默认 ROOT/output_sim。
# 看情况改：sim_config.json 覆盖视觉阈值，仿真光线与实机不同，不要把仿真阈值直接抄到实机。
# 不要改：A4 210×297、原点左上、X右Y下；convert_xy 用 [210,297]-点 做视觉↔仿真坐标互换。
# =============================================================================

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_SIM_ROOT = ROOT.parents[2] / "pybullet" / "puzzle_device"


# =============================================================================
# 【分区】仿真模块加载与坐标/场景
# 功能：把 simulator.py 目录插入 sys.path；加载视觉、仿真器、虚拟相机；把图2模板摆进仿真。
# 可修改：DEFAULT_SIM_ROOT；sim_config.json 里的仿真专用视觉参数。
# 看情况改：make_scene 颜色 (0.96,0.96,0.96) 接近白片；改颜色可能影响分割阈值。
# 不要改：convert_xy 公式；坐标系检查 origin=A4_top_left、size=[210,297]；必须正好 4 个任务。
# =============================================================================
def load_modules(sim_root: Path):
    if not (sim_root / "simulator.py").is_file():
        raise ValueError(f"simulator.py not found in {sim_root}")
    sys.path.insert(0, str(sim_root))
    import main as vision
    import simulator
    from virtual_camera import VirtualCamera
    return vision, simulator, VirtualCamera


def simulation_config(vision):
    return json.loads((ROOT / "sim_config.json").read_text(encoding="utf-8"))


def convert_xy(point):
    return tuple(np.array([210., 297.]) - np.asarray(point, dtype=float))


def make_scene(vision, simulator):
    specs = []
    for name, vertices in vision.FIGURE2_TEMPLATE_POLYGONS_MM.items():
        polygon = np.asarray(vertices, dtype=float)
        center, angle = vision.DEMO_PLACEMENTS[name]
        placed = vision.rotate_points(polygon - vision.polygon_centroid(polygon), angle) + center
        world = np.array([210., 297.]) - placed
        specs.append(simulator.PieceSpec(
            f"vision_{name}", tuple(np.ptp(world, axis=0)), tuple(world.mean(axis=0)),
            vertices_mm=tuple(map(tuple, world)), color=(0.96, 0.96, 0.96, 1)))
    return specs


def tasks_from_plan(plan, vision, simulator):
    coordinate = plan.get("coordinate_system", {})
    if (coordinate.get("origin") != "A4_top_left" or coordinate.get("x_axis") != "right"
            or coordinate.get("y_axis") != "down"
            or plan.get("mechanical_workspace_mm", {}).get("size") != [210., 297.]):
        raise ValueError("unsupported plan coordinate system")
    normalized = vision.serial_transport.validate_motion_plan(plan)
    if len(normalized) != 4:
        raise ValueError("the four-piece scene requires exactly four tasks")
    tasks = []
    for piece in sorted(plan["pieces"], key=lambda item: item["move_order"]):
        path = piece.get("path_mm")
        if not isinstance(path, list) or not path:
            raise ValueError("missing vision path_mm")
        tasks.append(simulator.MotionTask(
            *convert_xy(piece["source_pick_mm"]), float(piece["rotation_deg_signed"]),
            *convert_xy(piece["target_pick_mm"]), tuple(convert_xy(point) for point in path)))
    simulator.PuzzleGantrySim.validate_tasks(tasks)
    return tasks


# =============================================================================
# 【分区】VisionSession：一次仿真会话
# 功能：park 回原点拍照；run_once 规划并执行；handle_command 响应 start/home/rotate/quit。
# 可修改：park 后 step 60 次等待稳定；输出子目录用时间戳。
# 看情况改：拒绝 relaxed/fallback 解（仿真要求严格几何）。实机若允许宽松解，不要照搬这里。
# 不要改：finally 里恢复 OUTPUT_DIR；GUI 下清空鼠标/键盘队列，否则规划期间的点击会事后重放。
# =============================================================================
class VisionSession:
    def __init__(self, vision, simulator, camera_class, sim, output_dir: Path):
        self.vision, self.simulator, self.sim = vision, simulator, sim
        self.config = simulation_config(vision)
        self.camera = camera_class(sim.cid, float(self.config["pixels_per_mm"]))
        self.output_dir = output_dir
        self.last_output = None

    def park(self):
        import pybullet as p
        from config import CFG, MM
        if self.sim.fault:
            raise self.sim.fault
        if self.sim.held_body is not None:
            raise RuntimeError("cannot capture while holding a piece")
        q = [p.getJointState(self.sim.robot, self.sim.joint[name])[0]
             for name in ("x_joint", "y_joint", "theta_joint")]
        self.sim.set_pose(float(np.clip(q[0] / MM, 0, 210)),
                          float(np.clip(q[1] / MM, 0, 297)), CFG.z_high_mm,
                          float(np.clip(np.degrees(q[2]), 0, 180)))
        self.sim.set_pose(0, 0, CFG.z_high_mm, 0)
        for _ in range(60):
            self.sim.step()

    def run_once(self):
        if self.sim.fault:
            raise self.sim.fault
        out = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out.mkdir(parents=True, exist_ok=False)
        self.last_output = out
        previous_output = self.vision.OUTPUT_DIR
        self.vision.OUTPUT_DIR = out
        report = {"status": "failed", "stage": "capture"}
        try:
            self.sim._set_status("CAPTURING")
            self.park()
            frame = self.camera.capture()
            self.vision.write_image(out / "original.jpg", frame)
            report["stage"] = "planning"
            self.sim._set_status("PLANNING")
            paper = self.vision.warp_paper(frame, self.camera.matrix, self.config)
            detected, mask = self.vision.detect_pieces(paper, self.config)
            self.vision.write_image(out / "corrected.jpg", paper)
            self.vision.write_image(out / "mask.png", mask)
            if len(detected) != 4:
                raise RuntimeError(f"expected 4 pieces, detected {len(detected)}")
            # Software OpenGL otherwise competes with the bounded CPU search.
            import pybullet as p
            if self.sim.gui:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.sim.cid)
            try:
                _, paths = self.vision.process_paper(frame, paper, self.config, False)
            finally:
                if self.sim.gui and p.isConnected(self.sim.cid):
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.sim.cid)
            plan_path = out / "plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if any(item.match.relaxed_override for item in paths):
                reason = "relaxed or fallback solutions are not accepted in simulation"
                plan["ready_for_motion"] = False
                plan.setdefault("failure_reasons", []).append(reason)
                plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
                raise RuntimeError(reason)
            tasks = tasks_from_plan(plan, self.vision, self.simulator)
            report["stage"] = "execution"
            self.sim._set_status("RUNNING")
            results = self.sim.run_queue(tasks, final_status="VERIFYING")
            (out / "execution_results.json").write_text(
                json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8")
            report["stage"] = "verification"
            self.park()
            final = self.camera.capture()
            verification = self.vision.verify_final_frame(final, self.camera.matrix, self.config, plan_path)
            if verification["status"] != "pass":
                raise RuntimeError("final visual verification failed")
            report.update(status="pass", stage="done")
            self.sim._set_status("DONE - VISION PASS")
            return report
        except Exception as error:
            report["error"] = str(error)
            fault = error if isinstance(error, self.simulator.ExecutionError) else self.simulator.ExecutionError(
                "VISION_FAILED", 0, report["stage"], str(error))
            if self.sim.fault is not fault:
                self.sim._latch_fault(fault)
            if fault is error:
                raise
            raise fault from error
        finally:
            self.vision.OUTPUT_DIR = previous_output
            if self.sim.gui:
                import pybullet as p
                if p.isConnected(self.sim.cid):
                    # Inputs received during synchronous planning/motion must not replay afterward.
                    p.getMouseEvents()
                    p.getKeyboardEvents()
                    self.sim.pending_command = None
                    self.sim.drag_body = None
                    self.sim.left_down = False
            if report["stage"] == "execution":
                (out / "execution_results.json").write_text(
                    json.dumps([asdict(item) for item in self.sim.results], indent=2), encoding="utf-8")
            (out / "run_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            (out / "effective_config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
            print(f"SIM RESULT: {report['status']} ({report['stage']}); output: {out}")

    def handle_command(self, command):
        if command == "quit":
            if self.sim.fault:
                raise self.sim.fault
            return False
        try:
            if command == "home":
                self.sim.home()
            elif command == "rotate" and not self.sim.fault:
                self.sim.rotate_selected(15)
            elif command == "start" and not self.sim.fault:
                self.run_once()
        except self.simulator.ExecutionError:
            pass
        return True


# =============================================================================
# 【分区】命令行入口
# 功能：--direct 无界面跑一次；--auto-start 打开 GUI 后立刻跑一轮再退出；默认 GUI 循环处理按钮。
# 可修改：--sim-root、--output-dir、--no-realtime、预览窗 420×594。
# 看情况改：GUI 循环 sleep 0.005；机器卡时可略加大，不要大到按钮无响应。
# 不要改：finally 关仿真和 OpenCV 窗口；异常打印 SIM FAILED 并以码 1 退出（测试依赖这段）。
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--auto-start", action="store_true",
                        help="Open the GUI, run one closed-loop immediately, then exit")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output_sim")
    args = parser.parse_args()
    vision, simulator, camera_class = load_modules(args.sim_root.resolve())
    sim = simulator.PuzzleGantrySim(gui=not args.direct,
                                    realtime=not args.direct and not args.no_realtime, scene="vision")
    try:
        sim.spawn_pieces(make_scene(vision, simulator))
        session = VisionSession(vision, simulator, camera_class, sim, args.output_dir.resolve())
        if args.auto_start and sim.gui:
            for _ in range(24):
                sim.step()
        if args.direct or args.auto_start:
            session.run_once()
        else:
            import pybullet as p
            shown_output = None
            while p.isConnected(sim.cid):
                sim._handle_piece_mouse()
                command = sim.poll_buttons()
                if not session.handle_command(command):
                    break
                if session.last_output and session.last_output != shown_output:
                    image_path = session.last_output / "final_target_overlay.jpg"
                    if not image_path.exists():
                        image_path = session.last_output / "original.jpg"
                    preview = vision.read_image(image_path)
                    if preview is not None:
                        cv2.namedWindow("Vision result", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Vision result", 420, 594)
                        cv2.imshow("Vision result", preview)
                    shown_output = session.last_output
                cv2.waitKey(1)
                sim.step()
                time.sleep(0.005)
            if sim.fault:
                raise sim.fault
    finally:
        sim.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"SIM FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
