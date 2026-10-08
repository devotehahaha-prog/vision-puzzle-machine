#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""800×480 layout from EEZ/LVGL project (opi/_driver/工具/projects/puzzle_spi_ui)."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _rect(widget):
    return (int(widget["x"]), int(widget["y"]), int(widget["w"]), int(widget["h"]))


def load_eez_layout():
    candidates = []
    env = os.environ.get("EEZ_LAYOUT_JSON")
    if env:
        candidates.append(Path(env))
    candidates.extend((
        HERE / "ui_layout.json",
    ))
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("EEZ layout.json not found")


_UI = load_eez_layout()
_HOME = _UI["pages"]["home"]["widgets"]
_CALIB = _UI["pages"]["calib"]["widgets"]
_LIVE = _UI["pages"]["live"]["widgets"]
_TUNE = _UI["pages"]["tune"]["widgets"]
_THEME = _UI["theme"]

SCR = tuple(_THEME["scr"])
CARD = tuple(_THEME["card"])
GREY = tuple(_THEME["grey"])
TEXT = tuple(_THEME["text"])
MUTED = tuple(_THEME["muted"])
PRIMARY = tuple(_THEME["primary"])
PRIMARY_PRESSED = tuple(_THEME["primaryPressed"])
PRIMARY_MUTED = tuple(_THEME["primaryMuted"])
WHITE = tuple(_THEME["white"])
DISABLED = tuple(_THEME["disabled"])

SCREEN_W, SCREEN_H = int(_UI["display"]["width"]), int(_UI["display"]["height"])
PAD = 24
BTN_RADIUS = 16
CARD_RADIUS = 12
INDICATOR_H = 3
TAB_H = _HOME["tabOrdinary"]["h"]

ORDINARY_TAB = _rect(_HOME["tabOrdinary"])
POKER_TAB = _rect(_HOME["tabPoker"])
START_BTN = _rect(_HOME["btnStart"])
STOP_BTN = (SCREEN_W - 190, 10, 166, 48)
CALIB_BTN = _rect(_HOME["btnCalib"])
TUNE_BTN = _rect(_HOME["btnTune"])
LIVE_BTN = _rect(_HOME["btnLive"])
# Keep the additional entry usable with older external EEZ layout exports too.
CALIB_BTN = (40, 188, 171, 88)
CALIB_PREVIEW_BTN = (223, 188, 171, 88)
TUNE_BTN = (406, 188, 171, 88)
LIVE_BTN = (589, 188, 171, 88)
CALIB_VIEW_BUTTONS = (("raw", (175, 8, 195, 54), "原图角点"),
                      ("grid", (382, 8, 195, 54), "校正网格"),
                      ("mask", (589, 8, 195, 54), "识别分区"))
SUBTITLE = _rect(_HOME["subtitle"])
STATUS_BAR = _rect(_HOME["status"])
STATUS_TEXT = (STATUS_BAR[0] + 12, STATUS_BAR[1] + 8)

BACK_BTN = _rect(_CALIB["btnBack"])
SAVE_BTN = _rect(_TUNE["btnSave"])
MODE_BTN = _rect(_TUNE["btnMode"])
GROUP_PIECE_BTN = _rect(_TUNE["btnGroupPiece"])
GROUP_BOARD_BTN = _rect(_TUNE["btnGroupBoard"])
PAGE_PREV_BTN = _rect(_TUNE["btnPagePrev"])
PAGE_NEXT_BTN = _rect(_TUNE["btnPageNext"])
PREVIEW_RECT = _rect(_CALIB["image"])
LIVE_PREVIEW = _rect(_LIVE["image"])
LIVE_STATUS = _rect(_LIVE["status"])
TUNE_PREVIEW = _rect(_TUNE["image"])
UNDO_BTN = _rect(_CALIB["btnUndo"])
SAVE_CAL_BTN = _rect(_CALIB["btnSave"])
CALIB_STATUS = _rect(_CALIB["status"])
TUNE_STATUS = _rect(_TUNE["status"])
_SLIDER_AREAS = [_rect(_TUNE["slider%d" % i]) for i in range(4)]

MANUAL_BTN = (40, 336, 720, 76)
PAGE_MANUAL = "manual"

PAGE_HOME = "home"
PAGE_CALIB = "calib"
PAGE_CALIB_PREVIEW = "calib_preview"
PAGE_LIVE = "live"
PAGE_TUNE = "tune"
PAGE_EXECUTION = "execution"
GROUP_PIECE = "piece"
GROUP_BOARD = "board"
SLIDERS_PER_PAGE = 4

MODE_ORDINARY = "ordinary"
MODE_POKER = "poker"
MODE_LABELS = {
    MODE_ORDINARY: "普通拼图（实拍）",
    MODE_POKER: "扑克牌",
}

CORNER_NAMES = ("左上", "右上", "右下", "左下")


def hit(px, py, rect):
    x, y, w, h = rect
    return x <= px < x + w and y <= py < y + h


def slider_row(index):
    x, y, w, h = _SLIDER_AREAS[index]
    return {
        "label": (x, y, w, 22),
        "minus": (x, y + 24, 48, 40),
        "track": (x + 56, y + 32, w - 112, 24),
        "plus": (x + w - 48, y + 24, 48, 40),
    }


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def slider_value_from_x(x, track, lo, hi):
    tx, _ty, tw, _th = track
    if tw <= 1:
        return lo
    ratio = clamp((x - tx) / float(max(tw - 1, 1)), 0.0, 1.0)
    return int(round(lo + ratio * (hi - lo)))


def map_preview_to_camera(px, py, preview_rect, letterbox_rect, cam_size):
    if not hit(px, py, preview_rect):
        return None
    x, y, _w, _h = preview_rect
    lx, ly, lw, lh = letterbox_rect
    ix = px - x - lx
    iy = py - y - ly
    if ix < 0 or iy < 0 or ix >= lw or iy >= lh or lw <= 0 or lh <= 0:
        return None
    cam_w, cam_h = cam_size
    return (ix * cam_w / float(lw), iy * cam_h / float(lh))


def map_camera_to_preview(cx, cy, preview_rect, letterbox_rect, cam_size):
    x, y, _w, _h = preview_rect
    lx, ly, lw, lh = letterbox_rect
    cam_w, cam_h = cam_size
    if cam_w <= 0 or cam_h <= 0:
        return None
    return (int(x + lx + cx * lw / float(cam_w)),
            int(y + ly + cy * lh / float(cam_h)))


def letterbox(src_w, src_h, dst_w, dst_h):
    if src_w <= 0 or src_h <= 0:
        return (0, 0, dst_w, dst_h)
    scale = min(dst_w / float(src_w), dst_h / float(src_h))
    w = max(1, int(round(src_w * scale)))
    h = max(1, int(round(src_h * scale)))
    return ((dst_w - w) // 2, (dst_h - h) // 2, w, h)


def ordinary_sliders(config=None):
    cfg = config or {}
    morph = float(cfg.get("morphology_kernel_mm", 1.0))
    return [
        ("white_piece_saturation_max", "饱和度上限 S", 0, 255,
         int(cfg.get("white_piece_saturation_max", 55))),
        ("white_piece_otsu_min_value", "Otsu 下限", 0, 255,
         int(cfg.get("white_piece_otsu_min_value", 120))),
        ("white_piece_otsu_max_value", "Otsu 上限", 0, 255,
         int(cfg.get("white_piece_otsu_max_value", 230))),
        ("white_piece_value_min", "固定亮度下限 V", 0, 255,
         int(cfg.get("white_piece_value_min", 245))),
        ("morphology_kernel_mm_x10", "形态学 0.1mm", 0, 50,
         int(round(morph * 10))),
    ]


def poker_sliders(config=None):
    cfg = config or {}
    morph = float(cfg.get("poker_morph_close_mm", 2.5))
    ncc = float(cfg.get("poker_ncc_min_similarity", 0.45))
    return [
        ("poker_otsu_min_value", "扑克 Otsu 下限", 0, 255,
         int(cfg.get("poker_otsu_min_value", 80))),
        ("poker_otsu_max_value", "扑克 Otsu 上限", 0, 255,
         int(cfg.get("poker_otsu_max_value", 255))),
        ("poker_value_min", "扑克固定 V", 0, 255,
         int(cfg.get("poker_value_min", 90))),
        ("poker_morph_close_mm_x10", "闭运算 0.1mm", 0, 80,
         int(round(morph * 10))),
        ("poker_ncc_min_similarity_x100", "花纹相似度 %", 0, 100,
         int(round(ncc * 100))),
    ]


def board_sliders(config=None):
    cfg = config or {}
    return [
        ("auto_board_saturation_max", "A4 饱和度上限", 0, 255,
         int(cfg.get("auto_board_saturation_max", 110))),
        ("auto_board_min_dark_fraction_x100", "暗区比例 %", 0, 100,
         int(round(float(cfg.get("auto_board_min_dark_fraction", 0.55)) * 100))),
        ("auto_board_max_white_fraction_x100", "反光比例 %", 0, 100,
         int(round(float(cfg.get("auto_board_max_white_fraction", 0.35)) * 100))),
        ("auto_board_min_area_ratio_x100", "面积下限 %", 1, 80,
         int(round(float(cfg.get("auto_board_min_area_ratio", 0.08)) * 100))),
        ("auto_board_max_area_ratio_x100", "面积上限 %", 5, 90,
         int(round(float(cfg.get("auto_board_max_area_ratio", 0.35)) * 100))),
    ]


LIGHTING_KEYS = {
    "white_piece_saturation_max", "white_piece_otsu_min_value", "white_piece_otsu_max_value",
    "white_piece_value_min", "white_piece_threshold_mode", "morphology_kernel_mm",
    "poker_otsu_min_value", "poker_otsu_max_value", "poker_value_min",
    "poker_morph_close_mm", "poker_threshold_mode", "poker_ncc_min_similarity",
    "auto_board_saturation_max", "auto_board_min_dark_fraction",
    "auto_board_max_white_fraction", "auto_board_min_area_ratio", "auto_board_max_area_ratio",
}


class StartPageState:
    def __init__(self, default_mode=MODE_ORDINARY):
        self.mode = default_mode
        self.page = PAGE_HOME
        self.busy = False
        self.status = "就绪"
        self.armed = True
        self.calib_count = 0
        self.calibration_view = "raw"
        self.sliders = []
        self.group_sliders = []
        self.drag_slider = None
        self.threshold_mode = "otsu"
        self.tune_group = GROUP_PIECE
        self.tune_page = 0
        self.stored_params = {}
        self.plan_ready = False
        self.workflow_stage = ""
        self.workflow_message = ""
        self.review_image_path = None
        self.review_plan_path = None
        self.review_plan_id = None
        self.pattern_review_required = False

    def mode_label(self):
        return MODE_LABELS[self.mode]

    def title_text(self):
        return "拼图控制"

    def subtitle_text(self):
        if self.mode == MODE_ORDINARY:
            return "普通拼图：实拍四片 · 方案预览与实机拼合"
        return "当前模式：" + self.mode_label()

    def tab_label(self, mode):
        return MODE_LABELS[mode]

    def calib_hint(self):
        if self.calib_count >= 4:
            return "四点已齐，点保存"
        return "点第%d角：%s" % (self.calib_count + 1, CORNER_NAMES[self.calib_count])

    def load_sliders(self, config):
        key = "poker_threshold_mode" if self.mode == MODE_POKER else "white_piece_threshold_mode"
        self.threshold_mode = config.get(key, "otsu")
        if self.tune_group == GROUP_BOARD:
            self.group_sliders = board_sliders(config)
        elif self.mode == MODE_POKER:
            self.group_sliders = poker_sliders(config)
        else:
            self.group_sliders = ordinary_sliders(config)
        self.apply_tune_page()

    def apply_tune_page(self):
        pages = max(1, (len(self.group_sliders) + SLIDERS_PER_PAGE - 1) // SLIDERS_PER_PAGE)
        self.tune_page = min(max(0, self.tune_page), pages - 1)
        start = self.tune_page * SLIDERS_PER_PAGE
        self.sliders = self.group_sliders[start:start + SLIDERS_PER_PAGE]

    def decode_slider_value(self, key, value):
        if key.endswith("_x10"):
            return round(value / 10.0, 1)
        if key.endswith("_x100"):
            return round(value / 100.0, 2)
        return int(value)

    def slider_params(self):
        params = {}
        for key, _label, _lo, _hi, value in self.group_sliders:
            real_key = key
            if key.endswith("_x10"):
                real_key = key[:-4]
            elif key.endswith("_x100"):
                real_key = key[:-5]
            params[real_key] = self.decode_slider_value(key, value)
        if self.tune_group == GROUP_PIECE:
            key = "poker_threshold_mode" if self.mode == MODE_POKER else "white_piece_threshold_mode"
            params[key] = self.threshold_mode
        return params

    def set_slider(self, index, value):
        if index < 0 or index >= len(self.sliders):
            return False
        key, label, lo, hi, old = self.sliders[index]
        new = clamp(int(value), lo, hi)
        if "otsu_min_value" in key:
            new = min(new, next(item[4] for item in self.group_sliders if "otsu_max_value" in item[0]))
        elif "otsu_max_value" in key:
            new = max(new, next(item[4] for item in self.group_sliders if "otsu_min_value" in item[0]))
        if new == old:
            return False
        updated = (key, label, lo, hi, new)
        self.sliders[index] = updated
        global_index = self.tune_page * SLIDERS_PER_PAGE + index
        if 0 <= global_index < len(self.group_sliders):
            self.group_sliders[global_index] = updated
        return True

    def set_tune_group(self, group, config):
        if group not in (GROUP_PIECE, GROUP_BOARD) or group == self.tune_group:
            return False
        self.stored_params.update(self.slider_params())
        merged = dict(config)
        merged.update(self.stored_params)
        self.tune_group = group
        self.tune_page = 0
        self.drag_slider = None
        self.load_sliders(merged)
        return True

    def shift_tune_page(self, delta):
        pages = max(1, (len(self.group_sliders) + SLIDERS_PER_PAGE - 1) // SLIDERS_PER_PAGE)
        page = clamp(self.tune_page + delta, 0, pages - 1)
        if page == self.tune_page:
            return False
        self.tune_page = page
        self.drag_slider = None
        self.apply_tune_page()
        return True

    def set_mode(self, mode):
        if self.busy or self.page != PAGE_HOME or mode not in MODE_LABELS or mode == self.mode:
            return False
        self.mode = mode
        self.plan_ready = False
        return True

    def begin_run(self):
        self.busy = True
        self.status = "运行中…"
        self.armed = False

    def finish_run(self, ok):
        self.busy = False
        self.status = ("方案已生成，按确认执行" if ok and not self.plan_ready
                       else ("执行完成" if ok else "上次：失败"))
        self.armed = False

    def open_calib(self):
        if self.busy:
            return False
        self.page = PAGE_CALIB
        self.calib_count = 0
        self.status = self.calib_hint()
        return True

    def open_live(self):
        if self.busy:
            return False
        self.page = PAGE_LIVE
        self.status = "USB 原画面"
        return True

    def open_calibration_preview(self):
        if self.busy:
            return False
        self.page = PAGE_CALIB_PREVIEW
        self.calibration_view = "raw"
        self.status = "只读预览：对照四角、分界线和网格"
        return True

    def open_tune(self, config):
        if self.busy:
            return False
        self.page = PAGE_TUNE
        self.tune_group = GROUP_PIECE
        self.tune_page = 0
        self.stored_params = {}
        self.load_sliders(config)
        self.status = "左侧画面，右侧调参"
        return True

    def go_home(self):
        self.page = PAGE_HOME
        if not self.busy:
            self.status = "就绪"
        return True

    def on_points(self, pts):
        if pts is None:  # Controller has no new report; this is not a release.
            return None
        if not pts:
            self.drag_slider = None
        if pts and self.drag_slider is not None and self.page == PAGE_TUNE:
            index = self.drag_slider
            _, _, lo, hi, _ = self.sliders[index]
            return ("slider", index, slider_value_from_x(pts[0][0], slider_row(index)["track"], lo, hi))
        if pts and self.armed:
            self.armed = False
            x, y = pts[0]
            if self.page == PAGE_HOME:
                if self.workflow_stage:
                    if hit(x, y, STOP_BTN):
                        return "stop"
                    if hit(x, y, (435, 325, 345, 82)) and self.workflow_stage in {
                            "preview", "await_origin", "await_empty_accept", "await_physical_accept"}:
                        return "workflow_confirm"
                    if hit(x, y, (435, 410, 345, 54)) and not self.busy:
                        return "workflow_back"
                    return None
                if self.busy:
                    # STOP must remain available while a vision job is busy.
                    # Check it before the normal busy interaction lockout.
                    if hit(x, y, STOP_BTN):
                        return "stop"
                    return None
                if hit(x, y, MANUAL_BTN):
                    return "manual"
                if hit(x, y, START_BTN):
                    return "start"
                if hit(x, y, CALIB_BTN):
                    return "calib"
                if hit(x, y, CALIB_PREVIEW_BTN):
                    return "calib_preview"
                if hit(x, y, TUNE_BTN):
                    return "tune"
                if hit(x, y, LIVE_BTN):
                    return "live"
                if hit(x, y, ORDINARY_TAB):
                    return "ordinary"
                if hit(x, y, POKER_TAB):
                    return "poker"
                return None
            if hit(x, y, BACK_BTN):
                return "back"
            if self.page == PAGE_CALIB_PREVIEW:
                for view, rect, _label in CALIB_VIEW_BUTTONS:
                    if hit(x, y, rect):
                        self.calibration_view = view
                        return ("calibration_view", view)
                return None
            if self.page == PAGE_CALIB:
                if hit(x, y, UNDO_BTN):
                    return "undo"
                if hit(x, y, SAVE_CAL_BTN):
                    return "save_cal"
                if hit(x, y, PREVIEW_RECT):
                    return ("preview", x, y)
                return None
            if self.page == PAGE_TUNE:
                if hit(x, y, GROUP_PIECE_BTN):
                    return "group_piece"
                if hit(x, y, GROUP_BOARD_BTN):
                    return "group_board"
                if hit(x, y, PAGE_PREV_BTN):
                    return "page_prev"
                if hit(x, y, PAGE_NEXT_BTN):
                    return "page_next"
                if hit(x, y, MODE_BTN):
                    return "threshold_mode"
                if hit(x, y, SAVE_BTN):
                    return "save_tune"
                for index, (_key, _label, lo, hi, value) in enumerate(self.sliders):
                    row = slider_row(index)
                    if hit(x, y, row["minus"]):
                        return ("slider", index, clamp(value - 1, lo, hi))
                    if hit(x, y, row["plus"]):
                        return ("slider", index, clamp(value + 1, lo, hi))
                    if hit(x, y, row["track"]):
                        self.drag_slider = index
                        return ("slider", index, slider_value_from_x(x, row["track"], lo, hi))
                return None
            return None
        if not pts:
            self.armed = True
        return None
