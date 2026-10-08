"""PyBullet world: A4 paper, four ferrous pieces, XYZ-theta gantry, magnet."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pybullet as p

from config import CFG, MM
from gantry_urdf import write_urdf
from mesh_builder import load_layout, write_board_mesh, write_piece_mesh


@dataclass
class MotionTask:
    pick_x_mm: float
    pick_y_mm: float
    flip_deg: float
    place_x_mm: float
    place_y_mm: float
    path_mm: tuple[tuple[float, float], ...] | None = None


class ExecutionError(RuntimeError):
    def __init__(self, code: str, task_index: int, stage: str, detail: str = ""):
        self.code = code
        self.task_index = task_index
        self.stage = stage
        super().__init__(f"{code} task={task_index} stage={stage}: {detail}")


@dataclass
class TaskResult:
    task_index: int
    piece_name: str
    body_id: int
    target_pose: tuple[float, float, float]
    actual_pose: tuple[float, float, float]
    position_error_mm: float
    angle_error_deg: float


def angle_difference(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def run_cli(main: Callable[[], None]) -> None:
    try:
        main()
    except (ExecutionError, ValueError, OSError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from None


@dataclass
class PieceSpec:
    name: str
    size_xy_mm: tuple[float, float]
    start_xy_mm: tuple[float, float]
    start_yaw_deg: float = 0.0
    color: tuple[float, float, float, float] = (0.93, 0.93, 0.95, 1.0)
    vertices_mm: tuple[tuple[float, float], ...] | None = None
    texture: str | None = None
    crop_origin_px: tuple[int, int] | None = None
    texture_size_px: tuple[int, int] | None = None


def theta_pose(flip_deg: float) -> tuple[float, float]:
    """Firmware-compatible start/end angles for a 180-degree servo."""
    start = 180.0 if flip_deg < 0 else 0.0
    return start, start + float(flip_deg)


class PuzzleGantrySim:
    def __init__(self, gui: bool = True, realtime: bool = True,
                 scene: str = "photo") -> None:
        if scene not in {"photo", "vision"}:
            raise ValueError(f"unknown scene: {scene}")
        self.scene = scene
        self.gui = gui
        self.realtime = realtime
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        self.start_param = None
        self.home_param = None
        self.start_clicks = 0.0
        self.home_clicks = 0.0
        self.home_specs: list[PieceSpec] = []
        self.piece_specs: dict[int, PieceSpec] = {}
        self.piece_local_mm: dict[int, np.ndarray] = {}
        self.selected_body: Optional[int] = None
        self.drag_body: Optional[int] = None
        self.drag_offset_mm = np.zeros(2, dtype=np.float32)
        self.mouse_xy = (0.0, 0.0)
        self.rotate_param = None
        self.rotate_clicks = 0.0
        self.pending_command: Optional[str] = None
        self.left_down = False
        self.ui_buttons: dict[int, str] = {}
        self._keys_down: set[int] = set()
        self.busy = False
        self.fault: Optional[ExecutionError] = None
        self.task_index = 0
        self.stage = "idle"
        self.results: list[TaskResult] = []
        if gui:
            # ARM64 llvmpipe may take longer to service GUI commands.
            p.setTimeOut(60, physicsClientId=self.cid)
            p.resetDebugVisualizerCamera(0.55, 40, -50, [0.10, 0.16, 0.02])
            # The built-in Explorer/Params panels shift mouse coordinates, so
            # clicks miss the paper. Keep the 3D view full-window.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)

        self._spawn_ground()
        urdf_path = write_urdf()
        self.robot = p.loadURDF(
            str(urdf_path),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        self.joint = {
            p.getJointInfo(self.robot, i)[1].decode(): i
            for i in range(p.getNumJoints(self.robot))
        }
        self.magnet_link = self.joint["magnet_fixed"]
        self._disable_self_collision()
        self.layout = load_layout() if scene == "photo" else {}
        self.paper = self._spawn_a4()
        self.camera = self._spawn_camera()
        if gui:
            self._spawn_ui_buttons()
        self.pieces: list[int] = []
        self.piece_names: dict[int, str] = {}
        self.piece_colors: dict[int, tuple[float, float, float, float]] = {}
        self.held_cid: Optional[int] = None
        self.held_body: Optional[int] = None
        self.piece_pin: dict[int, tuple] = {}
        self.magnet_on = False
        self.status_text_id: Optional[int] = None
        self._draw_axes()
        self.set_pose(0.0, 0.0, CFG.z_high_mm, 0.0, wait=False)
        self._set_status("click green START or press SPACE")

    def close(self) -> None:
        if p.isConnected(self.cid):
            p.disconnect(self.cid)

    @staticmethod
    def _spawn_ground() -> int:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.2, 1.2, 0.01])
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[1.2, 1.2, 0.01],
            rgbaColor=[0.22, 0.24, 0.26, 1],
        )
        return p.createMultiBody(
            0,
            col,
            vis,
            [0.15, 0.15, -0.035],
        )

    def _disable_self_collision(self) -> None:
        links = [-1] + list(range(p.getNumJoints(self.robot)))
        for a in links:
            for b in links:
                if a < b:
                    p.setCollisionFilterPair(self.robot, self.robot, a, b, 0)

    def _spawn_a4(self) -> int:
        obj = (write_board_mesh(self.layout.get("board_texture", "a4_empty.jpg"))
               if self.scene == "photo" else None)
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                CFG.a4_x_mm * MM * 0.5,
                CFG.a4_y_mm * MM * 0.5,
                0.0006,
            ],
        )
        if obj is None:
            vis = p.createVisualShape(p.GEOM_BOX,
                                      halfExtents=[CFG.a4_x_mm * MM / 2,
                                                   CFG.a4_y_mm * MM / 2, 0.0006],
                                      rgbaColor=[0.10, 0.15, 0.12, 1])
        else:
            vis = p.createVisualShape(p.GEOM_MESH, fileName=str(obj), meshScale=[1, 1, 1])
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[
                CFG.a4_x_mm * MM * 0.5,
                CFG.a4_y_mm * MM * 0.5,
                0.0,
            ],
        )
        p.addUserDebugText("empty half", [0.08, 0.04, 0.02], [0.1, 0.4, 0.1], 1.1, 0)
        return body

    def _spawn_camera(self) -> int:
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.018, 0.024, 0.012],
            rgbaColor=[0.12, 0.12, 0.14, 1],
        )
        lens = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.01,
            length=0.012,
            rgbaColor=[0.05, 0.05, 0.08, 1],
        )
        cam = p.createMultiBody(
            0,
            -1,
            vis,
            [CFG.a4_x_mm * MM * 0.5, CFG.a4_y_mm * MM * 0.5, 0.28],
        )
        p.createMultiBody(
            0,
            -1,
            lens,
            [CFG.a4_x_mm * MM * 0.5, CFG.a4_y_mm * MM * 0.5, 0.265],
        )
        return cam

    def _spawn_ui_buttons(self) -> None:
        specs = (
            ("start", [0.055, -0.045, 0.018], [0.12, 0.70, 0.22, 1], "START"),
            ("home", [0.155, -0.045, 0.018], [0.80, 0.55, 0.12, 1], "HOME"),
            ("rotate", [0.255, -0.045, 0.018], [0.20, 0.45, 0.85, 1], "ROTATE"),
        )
        for command, pos, color, label in specs:
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.042, 0.018, 0.012], rgbaColor=color
            )
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.042, 0.018, 0.012])
            body = p.createMultiBody(0, col, vis, pos)
            self.ui_buttons[body] = command
            p.addUserDebugText(label, [pos[0] - 0.028, pos[1] - 0.006, pos[2] + 0.02], [1, 1, 1], 1.2, 0)

    def _draw_axes(self) -> None:
        o = [0, 0, 0.002]
        p.addUserDebugLine(o, [0.04, 0, 0.002], [1, 0, 0], 3, 0)
        p.addUserDebugLine(o, [0, 0.04, 0.002], [0, 0.7, 0], 3, 0)
        p.addUserDebugText("X 0-210", [0.045, 0.0, 0.01], [1, 0.2, 0.2], 1.2, 0)
        p.addUserDebugText("Y 0-297", [0.0, 0.05, 0.01], [0.2, 0.8, 0.2], 1.2, 0)
        p.addUserDebugText("A4 origin", [-0.01, -0.02, 0.02], [1, 1, 1], 1.1, 0)

    def spawn_pieces(
        self, specs: Sequence[PieceSpec], remember_home: bool = True
    ) -> list[int]:
        if remember_home:
            self.home_specs = list(specs)
        for body in self.pieces:
            p.removeBody(body)
        self.pieces = []
        self.piece_colors = {}
        self.piece_names = {}
        self.piece_specs = {}
        self.piece_local_mm = {}
        self.piece_pin = {}
        self.selected_body = None
        self.drag_body = None
        z = self._piece_rest_z()
        for spec in specs:
            if spec.vertices_mm:
                obj = write_piece_mesh(
                    {
                        "name": spec.name,
                        "vertices_mm": spec.vertices_mm,
                        "texture": spec.texture,
                        "crop_origin_px": spec.crop_origin_px,
                        "texture_size_px": spec.texture_size_px,
                    },
                    CFG.piece_thickness_mm * MM,
                )
                vis = p.createVisualShape(p.GEOM_MESH, fileName=str(obj), meshScale=[1, 1, 1],
                                          rgbaColor=list(spec.color) if not spec.texture else [1, 1, 1, 1])
                local = np.asarray(spec.vertices_mm) - np.mean(spec.vertices_mm, axis=0)
                half_z = CFG.piece_thickness_mm * MM * 0.5
                vertices = [[x * MM, y * MM, zc] for zc in (-half_z, half_z) for x, y in local]
                col = p.createCollisionShape(
                    p.GEOM_MESH, vertices=vertices
                )
                orn = p.getQuaternionFromEuler([0, 0, math.radians(spec.start_yaw_deg)])
            else:
                hx = spec.size_xy_mm[0] * MM * 0.5
                hy = spec.size_xy_mm[1] * MM * 0.5
                hz = CFG.piece_thickness_mm * MM * 0.5
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=list(spec.color)
                )
                orn = p.getQuaternionFromEuler([0, 0, math.radians(spec.start_yaw_deg)])
            body = p.createMultiBody(
                baseMass=0.018,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[
                    spec.start_xy_mm[0] * MM,
                    spec.start_xy_mm[1] * MM,
                    z,
                ],
                baseOrientation=orn,
            )
            p.changeDynamics(
                body,
                -1,
                lateralFriction=1.2,
                spinningFriction=0.4,
                rollingFriction=0.4,
                restitution=0.0,
                linearDamping=0.9,
                angularDamping=0.9,
                collisionMargin=0.00001,
            )
            self.piece_colors[body] = spec.color
            self.piece_names[body] = spec.name
            self.piece_specs[body] = spec
            if spec.vertices_mm:
                verts = np.asarray(spec.vertices_mm, dtype=np.float32)
                center = np.mean(verts, axis=0)
                self.piece_local_mm[body] = verts - center
            self.pieces.append(body)
            self._remember_piece_pose(body)
        self._set_gantry_piece_collisions(False)
        return self.pieces

    def _remember_piece_pose(self, body: int) -> None:
        pos, orn = p.getBasePositionAndOrientation(body)
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        self.piece_pin[body] = (pos, orn)

    def _set_gantry_piece_collisions(self, enable: bool) -> None:
        flag = 1 if enable else 0
        for piece in self.pieces:
            p.setCollisionFilterPair(self.paper, piece, -1, -1, 1)
            for link in range(-1, p.getNumJoints(self.robot)):
                p.setCollisionFilterPair(self.robot, piece, link, -1, flag)
            for button in self.ui_buttons:
                p.setCollisionFilterPair(button, piece, -1, -1, 0)

    def _set_piece_pair_collisions(self, body: int, enable: bool) -> None:
        flag = 1 if enable else 0
        for other in self.pieces:
            if other != body:
                p.setCollisionFilterPair(body, other, -1, -1, flag)

    @staticmethod
    def _piece_rest_z() -> float:
        return CFG.piece_thickness_mm * MM * 0.5 + 0.0006

    def _error(self, code: str, detail: str = "") -> ExecutionError:
        return ExecutionError(code, self.task_index, self.stage, detail)

    def _latch_fault(self, error: ExecutionError) -> None:
        self.fault = error
        self._set_status(f"FAULT {error.code} T{error.task_index} {error.stage}")
        print(error)

    @staticmethod
    def validate_tasks(tasks: Sequence[MotionTask]) -> None:
        if not 1 <= len(tasks) <= 4:
            raise ExecutionError("INVALID_TASK", 0, "validate", "queue must contain 1..4 tasks")
        for index, task in enumerate(tasks, 1):
            if not isinstance(task, MotionTask):
                raise ExecutionError("INVALID_TASK", index, "validate", "expected MotionTask")
            values = (task.pick_x_mm, task.pick_y_mm, task.flip_deg, task.place_x_mm, task.place_y_mm)
            bounds = ((0, CFG.a4_x_mm), (0, CFG.a4_y_mm), (-180, 180),
                      (0, CFG.a4_x_mm), (0, CFG.a4_y_mm))
            try:
                valid = all(math.isfinite(v) and lo <= v <= hi for v, (lo, hi) in zip(values, bounds))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ExecutionError("INVALID_TASK", index, "validate", repr(values))
            if task.path_mm is not None:
                try:
                    points = np.asarray(task.path_mm, dtype=float)
                    valid_path = (points.ndim == 2 and points.shape[1] == 2 and len(points) >= 1
                                  and np.isfinite(points).all()
                                  and (points >= 0).all()
                                  and (points <= [CFG.a4_x_mm, CFG.a4_y_mm]).all()
                                  and np.max(np.abs(points[0] - values[:2])) <= 0.02
                                  and np.max(np.abs(points[-1] - values[3:])) <= 0.02
                                  and (np.min(np.abs(np.diff(points, axis=0)), axis=1) <= 0.001).all())
                except (TypeError, ValueError, IndexError):
                    valid_path = False
                if not valid_path:
                    raise ExecutionError("INVALID_PATH", index, "validate", "invalid orthogonal path")

    def set_pose(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        theta_deg: float,
        wait: bool = True,
    ) -> None:
        values = (x_mm, y_mm, z_mm, theta_deg)
        limits = (CFG.a4_x_mm, CFG.a4_y_mm, CFG.z_travel_mm, 180.0)
        if not all(math.isfinite(v) and 0 <= v <= limit for v, limit in zip(values, limits)):
            raise self._error("INVALID_POSE", repr(values))
        targets = {
            "x_joint": x_mm * MM,
            "y_joint": y_mm * MM,
            "z_joint": z_mm * MM,
            "theta_joint": math.radians(theta_deg),
        }
        speeds = {
            "x_joint": CFG.xy_speed_m_s,
            "y_joint": CFG.xy_speed_m_s,
            "z_joint": CFG.z_speed_m_s,
            "theta_joint": CFG.theta_speed_rad_s,
        }
        for name, target in targets.items():
            p.setJointMotorControl2(
                self.robot,
                self.joint[name],
                p.POSITION_CONTROL,
                targetPosition=target,
                maxVelocity=speeds[name],
                force=120 if name != "theta_joint" else 8,
            )
        if wait:
            self._wait_joints(targets)

    def _wait_joints(self, targets: dict[str, float], steps: int = 2400) -> None:
        for _ in range(steps):
            self.step()
            ok = True
            for name, target in targets.items():
                q = p.getJointState(self.robot, self.joint[name])[0]
                tol = 0.0008 if name != "theta_joint" else math.radians(1.5)
                if abs(q - target) > tol:
                    ok = False
                    break
            if ok:
                for _ in range(12):
                    self.step()
                return
        raise self._error("MOTION_TIMEOUT", f"targets={targets}")

    def step(self) -> None:
        if self.held_body is not None and self.fault is None:
            self._check_attachment()
        p.stepSimulation()
        if self.gui and self.realtime:
            time.sleep(1.0 / 240.0)

    def _check_attachment(self) -> None:
        try:
            info = p.getConstraintInfo(self.held_cid)
            if info[0] != self.robot or info[2] != self.held_body:
                raise self._error("ATTACHMENT_LOST", "constraint references changed")
        except (p.error, TypeError):
            raise self._error("ATTACHMENT_LOST", "constraint missing") from None

    def magnet_tip_world(self) -> tuple[float, float, float]:
        state = p.getLinkState(self.robot, self.magnet_link, computeForwardKinematics=True)
        pos, orn = state[4], state[5]
        local = [0.0, 0.0, -CFG.magnet_height_mm * MM]
        world, _ = p.multiplyTransforms(pos, orn, local, [0, 0, 0, 1])
        return world

    def set_magnet(self, on: bool) -> None:
        self.magnet_on = on
        color = [0.15, 0.85, 0.25, 1] if on else [0.82, 0.16, 0.14, 1]
        p.changeVisualShape(self.robot, self.magnet_link, rgbaColor=color)
        if on:
            self._try_attach()
        else:
            self._release()

    def _try_attach(self) -> None:
        if self.held_cid is not None:
            return
        tip = self.magnet_tip_world()
        best_body = None
        best_score = None
        for body in self.pieces:
            pos, _ = p.getBasePositionAndOrientation(body)
            xy = math.hypot(tip[0] - pos[0], tip[1] - pos[1])
            dz = abs(tip[2] - (pos[2] + CFG.piece_thickness_mm * MM * 0.5))
            if xy > CFG.magnet_xy_capture_mm * MM:
                continue
            if dz > CFG.magnet_attach_gap_mm * MM:
                continue
            score = xy + 0.3 * dz
            if best_score is None or score < best_score:
                best_score = score
                best_body = body
        if best_body is None:
            raise self._error("MAGNET_MISS", f"tip={tuple(v / MM for v in tip)} mm")

        # Bullet constraint frames are relative to the link's inertial frame.
        # Preserve the complete current transform, including eccentric pickup.
        link = p.getLinkState(self.robot, self.magnet_link, computeForwardKinematics=True)
        inverse = p.invertTransform(link[0], link[1])
        body_pose = p.getBasePositionAndOrientation(best_body)
        relative = p.multiplyTransforms(*inverse, *body_pose)
        self.held_cid = p.createConstraint(
            self.robot,
            self.magnet_link,
            best_body,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            relative[0],
            [0, 0, 0],
            parentFrameOrientation=relative[1],
            childFrameOrientation=[0, 0, 0, 1],
        )
        self.held_body = best_body
        self._set_piece_pair_collisions(best_body, False)
        p.changeVisualShape(best_body, -1, rgbaColor=[0.15, 0.95, 0.35, 1])
        name = self.piece_names.get(best_body, str(best_body))
        print(f"PICKED {name}")
        self._set_status(f"HOLDING {name}")

    def _release(self) -> None:
        body = self.held_body
        if self.held_cid is not None:
            p.removeConstraint(self.held_cid)
            self.held_cid = None
            self.held_body = None
        if body is None:
            self._set_status("MAGNET OFF")
            return
        pos, orn = p.getBasePositionAndOrientation(body)
        yaw = p.getEulerFromQuaternion(orn)[2]
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        p.resetBasePositionAndOrientation(
            body,
            [pos[0], pos[1], self._piece_rest_z()],
            p.getQuaternionFromEuler([0.0, 0.0, yaw]),
        )
        self._set_piece_pair_collisions(body, True)
        self._remember_piece_pose(body)
        color = self.piece_colors.get(body, (0.93, 0.93, 0.95, 1.0))
        p.changeVisualShape(body, -1, rgbaColor=list(color))
        name = self.piece_names.get(body, str(body))
        print(f"PLACED {name} at ({pos[0]/MM:.1f},{pos[1]/MM:.1f}) mm")
        self._set_status(f"PLACED {name}")

    def _set_status(self, text: str) -> None:
        self.status = text
        if self.status_text_id is not None:
            p.removeUserDebugItem(self.status_text_id)
        self.status_text_id = p.addUserDebugText(
            text,
            [0.02, 0.31, 0.08],
            [1, 1, 0.2],
            1.4,
            0,
        )

    def run_task(self, task: MotionTask) -> TaskResult:
        if self.fault is not None:
            raise self._error("FAULT_LOCKED", "HOME required")
        try:
            self.validate_tasks([task])
            return self._execute_task(task)
        except ExecutionError as error:
            self._latch_fault(error)
            raise

    def _execute_task(self, task: MotionTask) -> TaskResult:
        start_deg, end_deg = theta_pose(task.flip_deg)
        print(
            f"TASK pick=({task.pick_x_mm:.1f},{task.pick_y_mm:.1f}) "
            f"flip={task.flip_deg:.0f} place=({task.place_x_mm:.1f},{task.place_y_mm:.1f})"
        )
        self.set_magnet(False)
        self.stage = "raise_safe"
        q = [p.getJointState(self.robot, self.joint[n])[0] for n in ("x_joint", "y_joint", "theta_joint")]
        self.set_pose(max(0.0, min(CFG.a4_x_mm, q[0] / MM)),
                      max(0.0, min(CFG.a4_y_mm, q[1] / MM)), CFG.z_high_mm,
                      max(0.0, min(180.0, math.degrees(q[2]))))
        self.stage = "move_pick"
        self.set_pose(task.pick_x_mm, task.pick_y_mm, CFG.z_high_mm,
                      max(0.0, min(180.0, math.degrees(q[2]))))
        self.stage = "pre_rotate"
        self.set_pose(task.pick_x_mm, task.pick_y_mm, CFG.z_high_mm, start_deg)
        self.stage = "lower_pick"
        self.set_pose(task.pick_x_mm, task.pick_y_mm, CFG.z_low_mm, start_deg)
        tip = self.magnet_tip_world()
        print(f"  Z down, magnet tip z={tip[2]/MM:.1f} mm")
        self.stage = "attach"
        self.set_magnet(True)
        body = self.held_body
        pos, orn = p.getBasePositionAndOrientation(body)
        initial_xy = np.asarray(pos[:2]) / MM
        initial_yaw = math.degrees(p.getEulerFromQuaternion(orn)[2])
        angle = math.radians(task.flip_deg)
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        expected_xy = np.array([task.place_x_mm, task.place_y_mm]) + rotation @ (
            initial_xy - np.array([task.pick_x_mm, task.pick_y_mm]))
        expected_yaw = angle_difference(initial_yaw + task.flip_deg, 0.0)
        for _ in range(48):
            self.step()
        self.stage = "lift_pick"
        self.set_pose(task.pick_x_mm, task.pick_y_mm, CFG.z_high_mm, start_deg)
        self.stage = "rotate"
        if end_deg != start_deg:
            self.set_pose(task.pick_x_mm, task.pick_y_mm, CFG.z_high_mm, end_deg)
        self.stage = "move_place"
        if task.path_mm is None:
            self.set_pose(task.place_x_mm, task.place_y_mm, CFG.z_high_mm, end_deg)
        else:
            for waypoint, (x_mm, y_mm) in enumerate(task.path_mm[1:], 1):
                self.stage = f"move_path_{waypoint}"
                self.set_pose(x_mm, y_mm, CFG.z_high_mm, end_deg)
        self.stage = "lower_place"
        self.set_pose(task.place_x_mm, task.place_y_mm, CFG.z_low_mm, end_deg)
        self.stage = "release"
        self._check_attachment()
        self.set_magnet(False)
        for _ in range(30):
            self.step()
        self.stage = "lift_place"
        self.set_pose(task.place_x_mm, task.place_y_mm, CFG.z_high_mm, end_deg)
        result = TaskResult(self.task_index, self.piece_names[body], body,
                            (float(expected_xy[0]), float(expected_xy[1]), expected_yaw),
                            (0.0, 0.0, 0.0), 0.0, 0.0)
        self._verify_result(result)
        return result

    def _verify_result(self, result: TaskResult) -> None:
        pos, orn = p.getBasePositionAndOrientation(result.body_id)
        yaw = math.degrees(p.getEulerFromQuaternion(orn)[2])
        result.actual_pose = (pos[0] / MM, pos[1] / MM, yaw)
        result.position_error_mm = math.hypot(result.actual_pose[0] - result.target_pose[0],
                                              result.actual_pose[1] - result.target_pose[1])
        result.angle_error_deg = abs(angle_difference(yaw, result.target_pose[2]))
        if result.position_error_mm > 2.0 or result.angle_error_deg > 2.0:
            raise ExecutionError("PLACEMENT_ERROR", result.task_index, "verify_place",
                                 f"{result.piece_name}: {result.position_error_mm:.3f} mm, {result.angle_error_deg:.3f} deg")

    def run_queue(self, tasks: Iterable[MotionTask], *, final_status: str = "DONE") -> list[TaskResult]:
        if self.fault is not None:
            raise self._error("FAULT_LOCKED", "HOME required")
        if self.busy:
            raise self._error("BUSY")
        self.busy = True
        self.results = []
        try:
            task_list = list(tasks)
            self.validate_tasks(task_list)
            for self.task_index, task in enumerate(task_list, 1):
                self.results.append(self.run_task(task))
            self.stage = "return_home"
            self.set_pose(0.0, 0.0, CFG.z_high_mm, 0.0)
            self.set_magnet(False)
            for result in self.results:
                self._verify_result(result)
                print(f"RESULT T{result.task_index} {result.piece_name}: "
                      f"{result.position_error_mm:.3f} mm, {result.angle_error_deg:.3f} deg")
            print("QUEUE DONE, returned to origin")
            self._set_status(f"{final_status}  click HOME or START")
            self.stage = "done"
            return self.results
        except ExecutionError as error:
            if self.fault is not error:
                self._latch_fault(error)
            raise
        finally:
            self.busy = False

    def home(self) -> None:
        """Release, send the gantry to origin, and put pieces back on the photo."""
        print("HOME")
        self.stage = "home"
        try:
            if self.held_cid is not None:
                ids = [p.getConstraintUniqueId(i) for i in range(p.getNumConstraints())]
                if self.held_cid in ids:
                    p.removeConstraint(self.held_cid)
            self.held_cid = self.held_body = None
            self.spawn_pieces([], remember_home=False)
            self.set_magnet(False)
            q = [p.getJointState(self.robot, self.joint[n])[0] for n in ("x_joint", "y_joint", "theta_joint")]
            self.set_pose(max(0.0, min(CFG.a4_x_mm, q[0] / MM)),
                          max(0.0, min(CFG.a4_y_mm, q[1] / MM)), CFG.z_high_mm,
                          max(0.0, min(180.0, math.degrees(q[2]))))
            self.set_pose(0.0, 0.0, CFG.z_high_mm, 0.0)
            self.spawn_pieces(self.home_specs, remember_home=False)
            self.fault = None
            self.results = []
            self.task_index = 0
            self.stage = "idle"
            self.pending_command = None
            self._set_status("HOME")
        except ExecutionError as error:
            self._latch_fault(error)
            raise

    def current_layout(self) -> dict:
        """Build a solver layout from the pieces' current world poses."""
        layout = {
            "pieces": [],
            "a4_x_mm": CFG.a4_x_mm,
            "a4_y_mm": CFG.a4_y_mm,
            "board_texture": self.layout.get("board_texture", "a4_empty.jpg"),
        }
        for body in self.pieces:
            spec = self.piece_specs[body]
            if body in self.piece_local_mm:
                verts = self.world_vertices_mm(body)
                center = np.mean(verts, axis=0)
            else:
                pos, orn = p.getBasePositionAndOrientation(body)
                yaw = math.degrees(p.getEulerFromQuaternion(orn)[2])
                center = np.array([pos[0] / MM, pos[1] / MM], dtype=np.float32)
                w, h = spec.size_xy_mm
                half = np.array([w, h], dtype=np.float32) * 0.5
                local = np.array(
                    [[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]],
                    dtype=np.float32,
                )
                c = math.cos(math.radians(yaw))
                s = math.sin(math.radians(yaw))
                rot = np.array([[c, -s], [s, c]], dtype=np.float32)
                verts = local @ rot.T + center
            layout["pieces"].append(
                {
                    "name": spec.name,
                    "center_mm": [float(center[0]), float(center[1])],
                    "vertices_mm": verts.tolist(),
                    "texture": spec.texture,
                    "crop_origin_px": spec.crop_origin_px,
                    "texture_size_px": spec.texture_size_px,
                }
            )
        return layout

    def world_vertices_mm(self, body: int) -> np.ndarray:
        pos, orn = p.getBasePositionAndOrientation(body)
        yaw = p.getEulerFromQuaternion(orn)[2]
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        local = self.piece_local_mm[body]
        center = np.array([pos[0] / MM, pos[1] / MM], dtype=np.float32)
        return local @ rot.T + center

    def _mouse_ray(self, mouse_x: float, mouse_y: float):
        (
            width,
            height,
            _,
            _,
            _,
            cam_forward,
            horizon,
            vertical,
            _,
            _,
            dist,
            target,
        ) = p.getDebugVisualizerCamera()
        cam_pos = [
            target[0] - dist * cam_forward[0],
            target[1] - dist * cam_forward[1],
            target[2] - dist * cam_forward[2],
        ]
        far_plane = 10000.0
        ray_forward = [cam_forward[i] * far_plane for i in range(3)]
        d_hor = [horizon[i] / float(max(width, 1)) for i in range(3)]
        d_vert = [vertical[i] / float(max(height, 1)) for i in range(3)]
        center = [cam_pos[i] + ray_forward[i] for i in range(3)]
        ray_to = [
            center[i]
            + (mouse_x - width * 0.5) * d_hor[i]
            + (height * 0.5 - mouse_y) * d_vert[i]
            for i in range(3)
        ]
        return cam_pos, ray_to

    def _ray_on_table(self, mouse_x: float, mouse_y: float) -> Optional[np.ndarray]:
        ray_from, ray_to = self._mouse_ray(mouse_x, mouse_y)
        z_plane = self._piece_rest_z()
        dz = ray_to[2] - ray_from[2]
        if abs(dz) < 1e-9:
            return None
        t = (z_plane - ray_from[2]) / dz
        if t <= 0:
            return None
        x = (ray_from[0] + t * (ray_to[0] - ray_from[0])) / MM
        y = (ray_from[1] + t * (ray_to[1] - ray_from[1])) / MM
        return np.array(
            [float(x), float(y)],
            dtype=np.float32,
        )

    def _hit_ui(self, mouse_x: float, mouse_y: float) -> Optional[str]:
        table = self._ray_on_table(mouse_x, mouse_y)
        if table is None:
            return None
        best_cmd = None
        best_dist = 35.0
        for body, command in self.ui_buttons.items():
            pos, _ = p.getBasePositionAndOrientation(body)
            dist = math.hypot(pos[0] / MM - table[0], pos[1] / MM - table[1])
            if dist < best_dist:
                best_dist = dist
                best_cmd = command
        return best_cmd

    def _hit_piece(self, mouse_x: float, mouse_y: float) -> Optional[int]:
        table = self._ray_on_table(mouse_x, mouse_y)
        if table is None:
            return None
        best_body = None
        best_dist = 28.0
        for body in self.pieces:
            pos, _ = p.getBasePositionAndOrientation(body)
            dist = math.hypot(pos[0] / MM - table[0], pos[1] / MM - table[1])
            if dist < best_dist:
                best_dist = dist
                best_body = body
        return best_body

    def _set_piece_xy_yaw(self, body: int, x_mm: float, y_mm: float, yaw_rad: float) -> None:
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        pos = [x_mm * MM, y_mm * MM, self._piece_rest_z()]
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw_rad])
        p.resetBasePositionAndOrientation(body, pos, orn)
        self._remember_piece_pose(body)

    def rotate_selected(self, delta_deg: float) -> None:
        if self.busy or self.fault is not None:
            return
        body = self.selected_body
        if body is None or body not in self.pieces:
            self._set_status("select a piece first")
            return
        pos, orn = p.getBasePositionAndOrientation(body)
        yaw = p.getEulerFromQuaternion(orn)[2] + math.radians(delta_deg)
        self._set_piece_xy_yaw(body, pos[0] / MM, pos[1] / MM, yaw)
        name = self.piece_names.get(body, str(body))
        self._set_status(f"{name} yaw {math.degrees(yaw):.0f} deg")

    def _begin_drag(self, mouse_x: float, mouse_y: float) -> None:
        ui = self._hit_ui(mouse_x, mouse_y)
        if ui is not None:
            self.pending_command = ui
            self.drag_body = None
            if self.fault is None:
                self._set_status(ui.upper())
            return
        body = self._hit_piece(mouse_x, mouse_y)
        if self.fault is not None:
            return
        if body is None:
            self.selected_body = None
            self.drag_body = None
            return
        self.selected_body = body
        table = self._ray_on_table(mouse_x, mouse_y)
        pos, _ = p.getBasePositionAndOrientation(body)
        if table is not None:
            self.drag_offset_mm = np.array(
                [pos[0] / MM - table[0], pos[1] / MM - table[1]],
                dtype=np.float32,
            )
        else:
            self.drag_offset_mm = np.zeros(2, dtype=np.float32)
        self.drag_body = body
        self._set_piece_pair_collisions(body, False)
        self._set_status(f"drag {self.piece_names.get(body, body)}")

    def _handle_piece_mouse(self) -> None:
        if self.busy or not self.gui:
            self.drag_body = None
            self.left_down = False
            return
        events = p.getMouseEvents()
        if events:
            for event in events:
                _, mouse_x, mouse_y, button_index, button_state = event
                self.mouse_xy = (mouse_x, mouse_y)
                if button_index != 0:
                    continue
                if button_state & p.KEY_WAS_RELEASED:
                    if self.drag_body is not None:
                        self._set_piece_pair_collisions(self.drag_body, True)
                        self._remember_piece_pose(self.drag_body)
                    self.drag_body = None
                    self.left_down = False
                    continue
                if (button_state & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN)) and not self.left_down:
                    self._begin_drag(mouse_x, mouse_y)
                    self.left_down = True
                elif button_state & p.KEY_IS_DOWN:
                    self.left_down = True
        if self.drag_body is None or self.drag_body not in self.pieces:
            return
        table = self._ray_on_table(*self.mouse_xy)
        if table is None:
            return
        target = table + self.drag_offset_mm
        target = np.clip(target, [8.0, 8.0], [CFG.a4_x_mm - 8.0, CFG.a4_y_mm - 8.0])
        _, orn = p.getBasePositionAndOrientation(self.drag_body)
        yaw = p.getEulerFromQuaternion(orn)[2]
        self._set_piece_xy_yaw(self.drag_body, float(target[0]), float(target[1]), yaw)

    def poll_buttons(self) -> Optional[str]:
        if not self.gui:
            return None
        if self.pending_command:
            command = self.pending_command
            self.pending_command = None
            return command
        keys = p.getKeyboardEvents()
        mapping = {
            ord("s"): "start",
            32: "start",
            ord("h"): "home",
            ord("r"): "rotate",
            ord("q"): "quit",
        }
        for code, command in mapping.items():
            state = keys.get(code, 0)
            down = bool(state & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN))
            was_down = code in self._keys_down
            if down:
                self._keys_down.add(code)
                if not was_down:
                    return command
            else:
                self._keys_down.discard(code)
        return None

    def wait_command(self) -> str:
        """Idle until START, HOME or Q. Direct mode returns start immediately."""
        if not self.gui:
            return "start"
        print("waiting: click green START / yellow HOME, or SPACE / H / R / Q")
        while p.isConnected(self.cid):
            self._handle_piece_mouse()
            command = self.poll_buttons()
            if command:
                return command
            self.step()
        return "quit"

    def run_with_buttons(
        self,
        tasks: Sequence[MotionTask] | None = None,
        plan_fn: Callable[[], Sequence[MotionTask]] | None = None,
    ) -> None:
        """GUI: drag pieces, START runs, HOME restores the photo layout."""
        task_list = list(tasks or ())
        if not self.gui:
            if plan_fn is not None:
                task_list = list(plan_fn())
            self.run_queue(task_list)
            return
        self._set_status("drag pieces, then START")
        while p.isConnected(self.cid):
            command = self.wait_command()
            if command == "quit":
                if self.fault is not None:
                    raise self.fault
                return
            if command == "home":
                try:
                    self.home()
                except ExecutionError:
                    pass
                continue
            if command == "rotate":
                self.rotate_selected(15.0)
                continue
            if command == "start":
                if self.busy or self.fault is not None:
                    continue
                if plan_fn is not None:
                    try:
                        task_list = list(plan_fn())
                    except Exception as error:
                        print(f"plan failed: {error}")
                        self._set_status(f"PLAN FAIL  {error}")
                        continue
                if not task_list:
                    self._set_status("no tasks")
                    continue
                self._set_status("RUNNING")
                try:
                    self.run_queue(task_list)
                except ExecutionError:
                    pass
        if self.fault is not None:
            raise self.fault


def demo_pieces() -> list[PieceSpec]:
    """Use the photographed green A4 and the three card fragments."""
    layout = load_layout()
    specs = []
    for item in layout["pieces"]:
        verts = tuple(tuple(pt) for pt in item["vertices_mm"])
        xs = [pt[0] for pt in verts]
        ys = [pt[1] for pt in verts]
        specs.append(
            PieceSpec(
                name=item["name"],
                size_xy_mm=(max(xs) - min(xs), max(ys) - min(ys)),
                start_xy_mm=tuple(item["center_mm"]),
                vertices_mm=verts,
                texture=item["texture"],
                crop_origin_px=tuple(item["crop_origin_px"]),
                texture_size_px=tuple(item["texture_size_px"]),
            )
        )
    return specs


def demo_tasks() -> list[MotionTask]:
    """Move the three photo pieces into the empty left/lower half."""
    layout = load_layout()
    places = ((70.0, 70.0), (120.0, 70.0), (95.0, 35.0))
    flips = (0.0, 25.0, -40.0)
    tasks = []
    for item, place, flip in zip(layout["pieces"], places, flips):
        cx, cy = item["center_mm"]
        tasks.append(MotionTask(cx, cy, flip, place[0], place[1]))
    return tasks
