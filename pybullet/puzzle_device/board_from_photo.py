"""Detect the green A4 and white card pieces from the overhead photo."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from config import CFG

ASSETS = Path(__file__).resolve().parent / "assets"
PHOTO = ASSETS / "a4_board.jpg"
SCALE = 4  # pixels per millimetre on the warped A4


def _imread(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    path.write_bytes(buf.tobytes())


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


def detect_paper(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("no green paper found")
    paper = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(paper, True)
    approx = cv2.approxPolyDP(paper, 0.02 * peri, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(paper)
        approx = cv2.boxPoints(rect)
    return _order_corners(approx)


def warp_a4(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    width = int(round(CFG.a4_y_mm * SCALE))   # 297 mm along photo X
    height = int(round(CFG.a4_x_mm * SCALE))  # 210 mm along photo Y
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(bgr, H, (width, height))


def detect_pieces(warped: np.ndarray) -> list[np.ndarray]:
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 170), (180, 70, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = warped.shape[0] * warped.shape[1] * 0.004
    pieces = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.018 * peri, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        pieces.append(approx.astype(np.float32))
    pieces.sort(key=lambda poly: cv2.contourArea(poly.reshape(-1, 1, 2)), reverse=True)
    return pieces[:4]


def pixels_to_control_mm(points_px: np.ndarray) -> np.ndarray:
    """Warped image: x right = 297 mm, y down = 210 mm.

    Control frame: origin at bottom-right, +X left (0-210), +Y up (0-297).
    """
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 2)
    x_mm = CFG.a4_x_mm - pts[:, 1] / SCALE
    y_mm = pts[:, 0] / SCALE
    return np.column_stack((x_mm, y_mm)).astype(np.float32)


def inpaint_board(warped: np.ndarray, pieces: list[np.ndarray]) -> np.ndarray:
    mask = np.zeros(warped.shape[:2], dtype=np.uint8)
    for poly in pieces:
        cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 255)
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8))
    return cv2.inpaint(warped, mask, 7, cv2.INPAINT_TELEA)


def crop_piece_texture(warped: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    x, y, w, h = cv2.boundingRect(np.rint(poly).astype(np.int32))
    pad = 4
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(warped.shape[1], x + w + pad)
    y1 = min(warped.shape[0], y + h + pad)
    crop = warped[y0:y1, x0:x1].copy()
    local = poly - np.array([x0, y0], dtype=np.float32)
    alpha = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(alpha, [np.rint(local).astype(np.int32)], 255)
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba, local, (int(x0), int(y0))


def build_layout() -> dict:
    photo = _imread(PHOTO)
    corners = detect_paper(photo)
    warped = warp_a4(photo, corners)
    pieces = detect_pieces(warped)
    empty = inpaint_board(warped, pieces)

    overlay = warped.copy()
    records = []
    for index, poly in enumerate(pieces, start=1):
        control = pixels_to_control_mm(poly)
        center = np.mean(control, axis=0)
        texture, local, crop_origin = crop_piece_texture(warped, poly)
        tex_path = ASSETS / f"piece_{index}.png"
        _imwrite(tex_path, texture)
        cv2.polylines(overlay, [np.rint(poly).astype(np.int32)], True, (0, 0, 255), 3)
        cv2.putText(
            overlay,
            f"P{index}",
            tuple(np.rint(np.mean(poly, axis=0)).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
        records.append(
            {
                "name": f"P{index}",
                "center_mm": [float(center[0]), float(center[1])],
                "vertices_mm": control.tolist(),
                "vertices_px": poly.tolist(),
                "texture": tex_path.name,
                "crop_origin_px": [int(crop_origin[0]), int(crop_origin[1])],
                "texture_size_px": [int(texture.shape[1]), int(texture.shape[0])],
            }
        )

    _imwrite(ASSETS / "a4_warped.jpg", warped)
    _imwrite(ASSETS / "a4_empty.jpg", empty)
    _imwrite(ASSETS / "a4_overlay.jpg", overlay)
    # Texture mapped onto the sim box: X along image width, Y along image height.
    # Sim box is 210 mm in X and 297 mm in Y, so rotate warped 297x210 into 210x297.
    board_tex = cv2.rotate(empty, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _imwrite(ASSETS / "a4_table.png", board_tex)

    layout = {
        "photo": PHOTO.name,
        "scale_px_per_mm": SCALE,
        "a4_x_mm": CFG.a4_x_mm,
        "a4_y_mm": CFG.a4_y_mm,
        "board_texture": "a4_empty.jpg",
        "pieces": records,
    }
    (ASSETS / "board_layout.json").write_text(
        json.dumps(layout, indent=2), encoding="utf-8"
    )
    print(f"detected {len(records)} pieces")
    for item in records:
        print(item["name"], "center", [round(v, 1) for v in item["center_mm"]],
              "n=", len(item["vertices_mm"]))
    return layout


if __name__ == "__main__":
    build_layout()
