"""Search a rectangular layout for detected card pieces in control XY."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from board_from_photo import ASSETS, SCALE, _imread, _imwrite
from config import CFG
from mesh_builder import load_layout
from simulator import MotionTask

LENGTH_TOLERANCE_MM = 6.0
MAX_OVERLAP_RATIO = 0.12
MIN_CONTACT_MM = 8.0
MASK_SCALE = 4.0
MASK_MARGIN_MM = 16.0
RECT_FILL_RATIO = 0.82
EMPTY_MARGIN_MM = 12.0
SHORT_EDGE_MERGE_MM = 3.0


@dataclass
class Piece:
    name: str
    vertices: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return np.mean(self.vertices, axis=0)

    @property
    def area(self) -> float:
        return abs(_signed_area(self.vertices))

    @property
    def edges(self) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
        pts = self.vertices
        out = []
        for i, start in enumerate(pts):
            end = pts[(i + 1) % len(pts)]
            out.append((i, start, end, float(np.linalg.norm(end - start))))
        return out


@dataclass
class PlacedPiece:
    piece: Piece
    vertices: np.ndarray
    theta_deg: float
    translation: np.ndarray


@dataclass
class Assembly:
    placed: list[PlacedPiece]
    tasks: list[MotionTask]
    target_vertices: dict[str, list[list[float]]] = field(default_factory=dict)
    size_mm: tuple[float, float] = (0.0, 0.0)
    fill_ratio: float = 0.0


def _signed_area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    pts = np.asarray(vertices, dtype=np.float32)
    if _signed_area(pts) < 0:
        return pts[::-1].copy()
    return pts.copy()


def _merge_short_edges(vertices: np.ndarray, min_len: float = SHORT_EDGE_MERGE_MM) -> np.ndarray:
    points = [np.asarray(pt, dtype=np.float32) for pt in vertices]
    changed = True
    while changed and len(points) > 3:
        changed = False
        polygon = np.asarray(points, dtype=np.float32)
        lengths = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        index = int(np.argmin(lengths))
        if float(lengths[index]) >= min_len:
            break
        nxt = (index + 1) % len(points)
        midpoint = (points[index] + points[nxt]) * 0.5
        if nxt == 0:
            points[0] = midpoint
            points.pop(index)
        else:
            points[index] = midpoint
            points.pop(nxt)
        changed = True
    return np.asarray(points, dtype=np.float32)


def _rot(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _wrap_deg(value: float) -> float:
    wrapped = (value + 180.0) % 360.0 - 180.0
    if wrapped <= -180.0:
        wrapped += 360.0
    return float(wrapped)


def _align_edge(
    piece: Piece,
    edge_index: int,
    target_start: np.ndarray,
    target_end: np.ndarray,
) -> PlacedPiece:
    src_start = piece.vertices[edge_index]
    src_end = piece.vertices[(edge_index + 1) % len(piece.vertices)]
    src_vec = src_end - src_start
    dst_vec = target_end - target_start
    theta = math.degrees(
        math.atan2(float(dst_vec[1]), float(dst_vec[0]))
        - math.atan2(float(src_vec[1]), float(src_vec[0]))
    )
    rotation = _rot(theta)
    translation = target_start - rotation @ src_start
    world = piece.vertices @ rotation.T + translation
    return PlacedPiece(piece, world.astype(np.float32), theta, translation.astype(np.float32))


def _raster(polygons: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_pts = np.vstack(polygons)
    origin = np.min(all_pts, axis=0) - MASK_MARGIN_MM
    size = np.maximum(
        np.ceil((np.max(all_pts, axis=0) - origin + MASK_MARGIN_MM) * MASK_SCALE).astype(int) + 2,
        2,
    )
    mask = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    for poly in polygons:
        px = np.rint((poly - origin) * MASK_SCALE).astype(np.int32)
        cv2.fillPoly(mask, [px.reshape(-1, 1, 2)], 255)
    return mask, origin


def _overlap_and_contact(existing: list[np.ndarray], candidate: np.ndarray) -> tuple[float, float]:
    mask, origin = _raster(existing + [candidate])
    existing_mask = np.zeros_like(mask)
    for poly in existing:
        px = np.rint((poly - origin) * MASK_SCALE).astype(np.int32)
        cv2.fillPoly(existing_mask, [px.reshape(-1, 1, 2)], 255)
    cand_mask = np.zeros_like(mask)
    px = np.rint((candidate - origin) * MASK_SCALE).astype(np.int32)
    cv2.fillPoly(cand_mask, [px.reshape(-1, 1, 2)], 255)
    exist_area = cv2.countNonZero(existing_mask)
    overlap = cv2.countNonZero(cv2.bitwise_and(existing_mask, cand_mask))
    contact = cv2.countNonZero(
        cv2.bitwise_and(cv2.dilate(existing_mask, np.ones((3, 3), np.uint8)), cand_mask)
    )
    overlap_ratio = 0.0 if exist_area == 0 else overlap / exist_area
    contact_mm = contact / MASK_SCALE
    return overlap_ratio, contact_mm


def _exposed_edges(placed: list[PlacedPiece]) -> list[tuple[np.ndarray, np.ndarray, float]]:
    edges = []
    for item in placed:
        pts = item.vertices
        for i, start in enumerate(pts):
            end = pts[(i + 1) % len(pts)]
            mid = (start + end) * 0.5
            length = float(np.linalg.norm(end - start))
            covered = False
            for other in placed:
                if other is item:
                    continue
                for j, os in enumerate(other.vertices):
                    oe = other.vertices[(j + 1) % len(other.vertices)]
                    other_mid = (os + oe) * 0.5
                    if float(np.linalg.norm(mid - other_mid)) < 4.0:
                        covered = True
                        break
                if covered:
                    break
            if not covered and length >= 8.0:
                edges.append((start, end, length))
    return edges


def _is_rectangle(placed: list[PlacedPiece]) -> tuple[bool, float, tuple[float, float]]:
    mask, _ = _raster([item.vertices for item in placed])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return False, 0.0, (0.0, 0.0)
    rect = cv2.minAreaRect(contours[0])
    w, h = rect[1]
    area = float(w * h)
    if area <= 1:
        return False, 0.0, (0.0, 0.0)
    fill = min(1.0, cv2.countNonZero(mask) / area)
    short = min(w, h) / MASK_SCALE
    long = max(w, h) / MASK_SCALE
    ok = fill >= RECT_FILL_RATIO and 30.0 <= short <= 120.0 and 40.0 <= long <= 160.0
    return ok, fill, (short, long)


def _search(placed: list[PlacedPiece], remaining: list[Piece]) -> list[PlacedPiece] | None:
    if not remaining:
        ok, _, _ = _is_rectangle(placed)
        return placed if ok else None

    exposed = _exposed_edges(placed)
    candidates: list[tuple[float, PlacedPiece, Piece]] = []
    for start, end, length in exposed:
        for piece in remaining:
            for index, _, _, src_len in piece.edges:
                if abs(src_len - length) > LENGTH_TOLERANCE_MM:
                    continue
                # Reverse pairing is the usual non-overlapping glue. Direct
                # pairing is the extra 180-degree option on isosceles cards.
                for target_start, target_end, pairing_penalty in (
                    (end, start, 0.0),
                    (start, end, 0.15),
                ):
                    pose = _align_edge(piece, index, target_start, target_end)
                    overlap, contact = _overlap_and_contact(
                        [item.vertices for item in placed], pose.vertices
                    )
                    if overlap >= MAX_OVERLAP_RATIO or contact < MIN_CONTACT_MM:
                        continue
                    candidates.append(
                        (abs(src_len - length) + pairing_penalty, pose, piece)
                    )

    candidates.sort(key=lambda item: item[0])
    seen = set()
    for _, pose, piece in candidates:
        key = (
            piece.name,
            tuple(np.rint(pose.vertices.reshape(-1) * 2).astype(int)),
        )
        if key in seen:
            continue
        seen.add(key)
        nxt = _search(
            placed + [pose],
            [item for item in remaining if item.name != piece.name],
        )
        if nxt is not None:
            return nxt
    return None


def _assembly_score(placed: list[PlacedPiece]) -> float:
    """Prefer the tightest rectangle, matching the original Q2 fill check.

    Do not maximise |flip|: that prefers a 180-degree edge pairing which is
    geometrically valid but fails after the user rotates a piece.
    """
    _, fill, _ = _is_rectangle(placed)
    return float(fill)


def _assemble_relative(pieces: list[Piece]) -> list[PlacedPiece] | None:
    ordered = sorted(pieces, key=lambda item: item.area, reverse=True)
    best = None
    best_score = -1.0
    for seed in ordered:
        seed_pose = PlacedPiece(seed, seed.vertices.copy(), 0.0, np.zeros(2, np.float32))
        result = _search(
            [seed_pose],
            [item for item in ordered if item.name != seed.name],
        )
        if result is None:
            continue
        relocated = _relocate(result, pieces)
        score = _assembly_score(relocated)
        if score > best_score:
            best = result
            best_score = score
    return best


def _empty_target_center(pieces: list[Piece]) -> np.ndarray:
    mean = np.mean(np.vstack([item.center for item in pieces]), axis=0)
    x = CFG.a4_x_mm * 0.5
    y = CFG.a4_y_mm * 0.25 if mean[1] > CFG.a4_y_mm * 0.5 else CFG.a4_y_mm * 0.75
    return np.array([x, y], dtype=np.float32)


def _relocate(placed: list[PlacedPiece], pieces: list[Piece]) -> list[PlacedPiece]:
    points = np.vstack([item.vertices for item in placed]).astype(np.float32)
    rect = cv2.minAreaRect(points.reshape(-1, 1, 2))
    box = cv2.boxPoints(rect)
    e0 = box[1] - box[0]
    e1 = box[2] - box[1]
    long_edge = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
    long_angle = math.degrees(math.atan2(float(long_edge[1]), float(long_edge[0])))
    align = -long_angle
    source_center = np.mean(points, axis=0)
    rotation = _rot(align)
    aligned = (points - source_center) @ rotation.T
    width = float(np.ptp(aligned[:, 0]))
    height = float(np.ptp(aligned[:, 1]))
    target = _empty_target_center(pieces)
    half = np.array([width, height], dtype=np.float32) * 0.5 + EMPTY_MARGIN_MM
    target[0] = float(np.clip(target[0], half[0], CFG.a4_x_mm - half[0]))
    if target[1] < CFG.a4_y_mm * 0.5:
        target[1] = float(np.clip(target[1], half[1], CFG.a4_y_mm * 0.5 - half[1]))
    else:
        target[1] = float(np.clip(target[1], CFG.a4_y_mm * 0.5 + half[1], CFG.a4_y_mm - half[1]))

    moved = []
    for item in placed:
        world = (item.vertices - source_center) @ rotation.T + target
        moved.append(PlacedPiece(item.piece, world.astype(np.float32), 0.0, np.zeros(2)))
    return moved


def _best_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Try cyclic and reversed vertex correspondence; return rotation, translation, theta, error."""
    best = None
    for reversed_order in (False, True):
        ordered = src[::-1] if reversed_order else src
        for shift in range(len(ordered)):
            cand = np.roll(ordered, -shift, axis=0)
            if len(cand) != len(dst):
                continue
            src_c = np.mean(cand, axis=0)
            dst_c = np.mean(dst, axis=0)
            h = (cand - src_c).T @ (dst - dst_c)
            u, _, vt = np.linalg.svd(h)
            r = vt.T @ u.T
            if np.linalg.det(r) < 0:
                vt = vt.copy()
                vt[-1] *= -1
                r = vt.T @ u.T
            t = dst_c - r @ src_c
            fitted = cand @ r.T + t
            error = float(np.mean(np.linalg.norm(fitted - dst, axis=1)))
            theta = math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))
            item = (error, r.astype(np.float32), t.astype(np.float32), theta)
            if best is None or error < best[0]:
                best = item
    if best is None:
        raise RuntimeError("no rigid correspondence")
    return best[1], best[2], best[3], best[0]


def _interior_angles(vertices: np.ndarray) -> np.ndarray:
    angles = []
    count = len(vertices)
    for index in range(count):
        previous = vertices[(index - 1) % count] - vertices[index]
        following = vertices[(index + 1) % count] - vertices[index]
        denom = float(np.linalg.norm(previous) * np.linalg.norm(following))
        if denom <= 1e-6:
            angles.append(180.0)
            continue
        cosine = float(np.clip(np.dot(previous, following) / denom, -1.0, 1.0))
        angles.append(float(math.degrees(math.acos(cosine))))
    return np.asarray(angles, dtype=np.float32)


def _sharpest_heading(vertices: np.ndarray) -> float:
    """Direction from the polygon centre toward the sharpest printed corner."""
    angles = _interior_angles(vertices)
    apex = vertices[int(np.argmin(angles))]
    center = np.mean(vertices, axis=0)
    vector = apex - center
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def _flip_from_vertices(src: np.ndarray, dst: np.ndarray) -> float:
    """Rotate the sharpest corner heading from the source pose onto the target."""
    _, _, theta, error = _best_rigid(src, dst)
    heading = _wrap_deg(_sharpest_heading(dst) - _sharpest_heading(src))
    # Isosceles cards have two similar long sides, so Kabsch can lock onto a
    # 0-degree residual. Prefer the printed-tip heading when it disagrees.
    if abs(_wrap_deg(heading - theta)) > 90.0:
        return heading
    if error > 4.0:
        return heading
    return _wrap_deg(theta)


def pieces_from_layout(layout: dict | None = None) -> list[Piece]:
    layout = layout or load_layout()
    return [
        Piece(
            item["name"],
            _merge_short_edges(_ensure_ccw(np.asarray(item["vertices_mm"], dtype=np.float32))),
        )
        for item in layout["pieces"]
    ]


def solve_assembly(layout: dict | None = None) -> Assembly:
    layout = layout or load_layout()
    pieces = pieces_from_layout(layout)
    if len(pieces) < 2:
        raise RuntimeError("need at least two pieces to assemble a rectangle")
    relative = _assemble_relative(pieces)
    if relative is None:
        raise RuntimeError("could not find a rectangular assembly from the detected edges")
    placed = _relocate(relative, pieces)
    ok, fill, size = _is_rectangle(placed)
    if not ok:
        raise RuntimeError(
            f"relocated layout is not a rectangle fill={fill:.3f} size={size}"
        )
    tasks = []
    targets = {}
    for item in placed:
        theta = _flip_from_vertices(item.piece.vertices, item.vertices)
        pick = item.piece.center
        place = np.mean(item.vertices, axis=0)
        tasks.append(
            MotionTask(
                float(pick[0]),
                float(pick[1]),
                _wrap_deg(theta),
                float(place[0]),
                float(place[1]),
            )
        )
        targets[item.piece.name] = item.vertices.tolist()
    return Assembly(placed, tasks, targets, size, fill if ok else fill)


def _control_to_warped(points_mm: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_mm, dtype=np.float32).reshape(-1, 2)
    return np.column_stack(
        (pts[:, 1] * SCALE, (CFG.a4_x_mm - pts[:, 0]) * SCALE)
    )


def save_plan(assembly: Assembly) -> Path:
    board = _imread(ASSETS / "a4_empty.jpg")
    overlay = board.copy()
    colors = ((40, 80, 220), (40, 180, 80), (20, 160, 230), (180, 80, 200))
    for index, item in enumerate(assembly.placed):
        px = np.rint(_control_to_warped(item.vertices)).astype(np.int32)
        color = colors[index % len(colors)]
        cv2.fillPoly(overlay, [px.reshape(-1, 1, 2)], color)
        cv2.polylines(board, [px.reshape(-1, 1, 2)], True, color, 3)
        pick = _control_to_warped(item.piece.center)[0]
        place = _control_to_warped(np.mean(item.vertices, axis=0))[0]
        cv2.arrowedLine(
            board,
            tuple(np.rint(pick).astype(int)),
            tuple(np.rint(place).astype(int)),
            color,
            2,
            tipLength=0.08,
        )
        cv2.putText(
            board,
            item.piece.name,
            tuple(np.rint(place).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
        )
    cv2.addWeighted(overlay, 0.35, board, 0.65, 0, board)
    out = ASSETS / "assembly_plan.jpg"
    _imwrite(out, board)
    payload = {
        "size_mm": list(assembly.size_mm),
        "fill_ratio": assembly.fill_ratio,
        "tasks": [
            {
                "pick_x_mm": task.pick_x_mm,
                "pick_y_mm": task.pick_y_mm,
                "flip_deg": task.flip_deg,
                "place_x_mm": task.place_x_mm,
                "place_y_mm": task.place_y_mm,
            }
            for task in assembly.tasks
        ],
        "targets": assembly.target_vertices,
    }
    (ASSETS / "assembly_plan.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return out
