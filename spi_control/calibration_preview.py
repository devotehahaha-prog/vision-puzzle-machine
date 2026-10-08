"""Read-only views of the same calibration and segmentation used by planning."""
from __future__ import annotations

import contextlib
import io
import numpy as np
import cv2


def render_calibration_preview(frame, vision, config, view="raw"):
    actual_size = [int(frame.shape[1]), int(frame.shape[0])]
    formal = config.get("camera_calibration") or {}
    saved_size = config.get("camera_calibration_image_size") or formal.get("image_size")
    if saved_size and list(saved_size) != actual_size:
        raise ValueError(f"分辨率不匹配：当前{actual_size[0]}×{actual_size[1]}，标定{saved_size[0]}×{saved_size[1]}")
    with contextlib.redirect_stdout(io.StringIO()):
        matrix, source = vision.resolve_calibration_matrix(frame, config, vision.load_calibration(config))
    matrix = np.asarray(matrix, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("标定矩阵无效，请重新标定")
    width, height = vision.paper_size_px(config)
    inverse = np.linalg.inv(matrix)
    destination = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    corners = cv2.perspectiveTransform(destination[None], inverse)[0]
    if not np.isfinite(corners).all():
        raise ValueError("标定角点无效，请重新标定")
    paper = vision.warp_paper(frame, matrix, config)
    ppm = float(config["pixels_per_mm"])
    grid_mm = 25
    segments = []
    for x in np.arange(grid_mm, float(config["paper_width_mm"]), grid_mm):
        segments.append((np.float32([[x*ppm, 0], [x*ppm, height-1]]), (80, 180, 100)))
    for y in np.arange(grid_mm, float(config["paper_height_mm"]), grid_mm):
        segments.append((np.float32([[0, y*ppm], [width-1, y*ppm]]), (80, 180, 100)))
    divider = float(config.get("separator_line_y_mm", float(config["paper_height_mm"])/2)) * ppm
    segments.append((np.float32([[0, divider], [width-1, divider]]), (0, 220, 255)))
    display = frame.copy() if view == "raw" else paper.copy()
    for segment, color in segments:
        points = cv2.perspectiveTransform(segment[None], inverse)[0] if view == "raw" else segment
        cv2.polylines(display, [np.rint(points).astype(np.int32)], False, color, 2, cv2.LINE_AA)
    quad = corners if view == "raw" else destination
    cv2.polylines(display, [np.rint(quad).astype(np.int32)], True, (255, 220, 0), 3, cv2.LINE_AA)
    for name, point in zip(("1 TL", "2 TR", "3 BR", "4 BL"), quad):
        x, y = np.rint(point).astype(int)
        cv2.circle(display, (x, y), 7, (0, 60, 255), -1)
        tx, ty = max(4, min(x+10, display.shape[1]-115)), max(25, min(y+25, display.shape[0]-10))
        cv2.putText(display, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 60, 255), 2, cv2.LINE_AA)
    metadata = {"calibration_source": "production_config" if formal else source,
                "calibration_view": view, "grid_mm": grid_mm,
                "saved_at": formal.get("saved_at", ""), "image_corners": corners.tolist(),
                "notice": f"{'正式配置' if formal else '保存标定'} · {actual_size[0]}×{actual_size[1]} · 网格25mm"}
    if source == "auto_board":
        metadata["calibration_source"] = source
        metadata["notice"] = f"当前自动找板 · {actual_size[0]}×{actual_size[1]} · 网格25mm"
    if view == "mask":
        _, original_mask = vision.mask_in_paper(frame, matrix, config)
        pieces, mask = vision.detect_pieces(paper, config, mask_override=original_mask)
        if config.get("ordinary_camera_preview"):
            pieces = vision.number_by_move_order(pieces, config)
        colored = display.copy()
        colored[mask > 0] = (50, 160, 255)
        display = cv2.addWeighted(display, .65, colored, .35, 0)
        for index, piece in enumerate(pieces, 1):
            cv2.drawContours(display, [piece.contour], -1, (0, 30, 255), 3)
            center = tuple(np.rint(piece.center_px).astype(int))
            cv2.putText(display, f"P{index}", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        expected = int(config.get("expected_piece_count", 3))
        metadata.update(detected_piece_count=len(pieces), expected_piece_count=expected)
        metadata["notice"] = (f"下半区识别 {len(pieces)}/{expected}片 · " +
                              ("请检查粘连、遮挡及边界" if len(pieces) != expected else "数量符合；仍需生成方案检查"))
    if view != "raw":
        for label, pos in [("(0,0)  X ->", (35, 80)),
                           ("TARGET", (width//2-90, 130)),
                           ("SOURCE", (width//2-90, int(divider)+70)),
                           (f"{config['paper_width_mm']:g} x {config['paper_height_mm']:g} mm", (35, height-45))]:
            cv2.putText(display, label, pos, cv2.FONT_HERSHEY_SIMPLEX, .85, (20, 40, 220), 2, cv2.LINE_AA)
    if view == "mask" and config.get("ordinary_camera_preview"):
        # Enlarge the pieces on the small screen; grid/raw retain the full A4.
        display = display[max(0, min(height - 1, int(divider))):].copy()
        metadata["notice"] = "普通拼图 · " + metadata["notice"]
        metadata["display_region"] = "source_bottom"
    return display, metadata
