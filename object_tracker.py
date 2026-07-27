"""
Трекер объекта для тепловизионного (серого) видео ~640x512.

Алгоритм: базовый трекер (KCF / NCC как в DepthMapKornia) + CLAHE-NCC
(уточнение/масштаб) + тепловая сигнатура + строгий локальный reacquire.

Drop-in API для video_track_depth / stereo_live.

    python object_tracker.py --video left.mp4 --tracker kcf
    python object_tracker.py --video left.mp4 --tracker ncc
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from depth_map import display_scale, fit_for_display

Roi = tuple[int, int, int, int]
TRACKER_KINDS = ("csrt", "kcf", "mosse", "ncc")


class NccTemplateTracker:
    """Локальный template/NCC-трекер — тот же базовый движок, что в DepthMapKornia."""

    def __init__(self, *, search_scale: float = 1.0, min_score: float = 0.25) -> None:
        self._roi: Roi | None = None
        self._template: np.ndarray | None = None
        self._search_scale = float(search_scale)
        self._min_score = float(min_score)

    def init(self, frame: np.ndarray, roi: tuple) -> bool:
        x, y, w, h = (int(v) for v in roi)
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._roi = (x, y, w, h)
        self._template = gray[y : y + h, x : x + w].copy()
        return bool(self._template.size)

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[float, float, float, float]]:
        if self._roi is None or self._template is None:
            return False, (0.0, 0.0, 0.0, 0.0)
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self._roi
        margin = int(round(max(w, h) * self._search_scale))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(gray.shape[1], x + w + margin)
        y1 = min(gray.shape[0], y + h + margin)
        search = gray[y0:y1, x0:x1]
        if search.shape[0] < h or search.shape[1] < w:
            return False, (float(x), float(y), float(w), float(h))
        res = cv2.matchTemplate(search, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < self._min_score:
            return False, (float(x), float(y), float(w), float(h))
        nx, ny = x0 + int(max_loc[0]), y0 + int(max_loc[1])
        self._roi = (nx, ny, w, h)
        patch = gray[ny : ny + h, nx : nx + w]
        if patch.shape == self._template.shape:
            self._template = (
                0.9 * self._template.astype(np.float32)
                + 0.1 * patch.astype(np.float32)
            ).astype(np.uint8)
        return True, (float(nx), float(ny), float(w), float(h))


def create_raw_tracker(kind: str):
    kind = kind.lower()
    if kind not in TRACKER_KINDS:
        raise ValueError(
            f"Неизвестный трекер '{kind}'. Доступны: {', '.join(TRACKER_KINDS)}."
        )
    if kind == "ncc":
        return NccTemplateTracker()
    name = {
        "csrt": "TrackerCSRT_create",
        "kcf": "TrackerKCF_create",
        "mosse": "TrackerMOSSE_create",
    }[kind]
    for mod in (cv2, getattr(cv2, "legacy", None)):
        if mod is None:
            continue
        factory = getattr(mod, name, None)
        if factory is not None:
            return factory()
    raise RuntimeError(
        f"Трекер '{kind}' недоступен. Установите opencv-contrib-python "
        f"или используйте --tracker ncc / --kornia-tracker."
    )


def clamp_roi(roi: tuple[float, float, float, float], width: int, height: int) -> Roi:
    x, y, rw, rh = roi
    x = int(round(x))
    y = int(round(y))
    rw = int(round(rw))
    rh = int(round(rh))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    rw = max(1, min(rw, width - x))
    rh = max(1, min(rh, height - y))
    return x, y, rw, rh


def roi_center(roi: Roi) -> tuple[int, int]:
    x, y, rw, rh = roi
    return x + rw // 2, y + rh // 2


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _as_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _enhance_thermal(gray: np.ndarray) -> np.ndarray:
    """CLAHE для низкоконтрастного ТВ/ИК 640×512."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def select_object_roi(frame_bgr: np.ndarray, max_display: int = 1200) -> Roi | None:
    frame_bgr = _as_bgr(frame_bgr)
    scale = display_scale(frame_bgr.shape, max_display)
    preview = fit_for_display(frame_bgr, scale)
    window = "Select object (Enter/Space = OK, c = cancel)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    x, y, rw, rh = roi
    if rw <= 0 or rh <= 0:
        return None
    if scale < 1.0:
        inv = 1.0 / scale
        x, y, rw, rh = int(x * inv), int(y * inv), int(rw * inv), int(rh * inv)
    return clamp_roi((x, y, rw, rh), frame_bgr.shape[1], frame_bgr.shape[0])


def _pad_roi(roi: Roi, width: int, height: int, pad_px: int = 2) -> Roi:
    """Минимальный запас 1–2 px, без процентного раздувания."""
    x, y, rw, rh = roi
    p = max(0, int(pad_px))
    return clamp_roi((x - p, y - p, rw + 2 * p, rh + 2 * p), width, height)


def _estimate_background(gray: np.ndarray, point: tuple[int, int]) -> float:
    h, w = gray.shape[:2]
    px, py = int(point[0]), int(point[1])
    border = np.concatenate(
        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
    ).astype(np.float32)
    bg_border = float(np.median(border))
    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - px) ** 2 + (yy - py) ** 2
    far = (dist2 >= 90**2) & (dist2 <= 180**2)
    if int(far.sum()) >= 80:
        bg_far = float(np.median(gray[far]))
        seed = float(gray[py, px])
        return bg_far if abs(bg_far - seed) >= abs(bg_border - seed) else bg_border
    return bg_border


def _keep_seed_component(mask: np.ndarray, point: tuple[int, int]) -> np.ndarray | None:
    px, py = int(point[0]), int(point[1])
    if mask[py, px] == 0:
        return None
    _n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    lab = int(labels[py, px])
    if lab <= 0:
        return None
    return (labels == lab).astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray, *, mass_keep: float = 0.96) -> Roi | None:
    """Плотный bbox: обрезает редкую «ауру» по проекциям массы."""
    col = mask.sum(axis=0).astype(np.float64)
    row = mask.sum(axis=1).astype(np.float64)
    total = float(col.sum())
    if total < 20:
        return None
    keep = float(np.clip(mass_keep, 0.8, 1.0))
    trim = 0.5 * (1.0 - keep)

    def _span(proj: np.ndarray) -> tuple[int, int] | None:
        s = float(proj.sum())
        if s <= 0:
            return None
        c = np.cumsum(proj)
        lo = int(np.searchsorted(c, trim * s, side="left"))
        hi = int(np.searchsorted(c, (1.0 - trim) * s, side="left"))
        hi = min(len(proj) - 1, max(hi, lo))
        return lo, hi

    xs = _span(col)
    ys = _span(row)
    if xs is None or ys is None:
        return None
    x0, x1 = xs
    y0, y1 = ys
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _seed_component_mask(
    gray: np.ndarray,
    point: tuple[int, int],
    lo_diff: int,
    up_diff: int,
    *,
    connectivity: int = 8,
) -> np.ndarray | None:
    """FloodFill от клика; возвращает бинарную маску компоненты или None."""
    h, w = gray.shape[:2]
    px, py = int(point[0]), int(point[1])
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = connectivity | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(
        blurred,
        mask,
        (px, py),
        0,
        (int(lo_diff),),
        (int(up_diff),),
        flags,
    )
    region = (mask[1:-1, 1:-1] > 0).astype(np.uint8)
    return _keep_seed_component(region, (px, py))


def _thermal_seed_mask(
    gray: np.ndarray,
    point: tuple[int, int],
    tolerance: int,
) -> np.ndarray | None:
    """Плотная маска объекта: порог между seed и фоном + компонента клика."""
    px, py = int(point[0]), int(point[1])
    seed = float(gray[py, px])
    bg = _estimate_background(gray, (px, py))
    hot = seed >= bg
    delta = abs(seed - bg)
    tol = max(6, int(tolerance))
    if delta < 8:
        # Низкий контраст — узкий flood вокруг seed.
        return _seed_component_mask(gray, (px, py), tol, tol, connectivity=8)

    # Порог ближе к объекту, чем к фону → меньше ореола (рамка ~ размер объекта).
    # alpha=0.55: берем пиксели от mid+ чуть в сторону объекта.
    alpha = 0.58
    if hot:
        thr = bg + alpha * delta
        raw = (gray.astype(np.float32) >= thr).astype(np.uint8)
    else:
        thr = bg - alpha * delta
        raw = (gray.astype(np.float32) <= thr).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k, iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k, iterations=1)
    mask = _keep_seed_component(raw, (px, py))
    if mask is not None and int(mask.sum()) >= 20:
        return mask

    # Запас: flood с умеренным допуском (не до фона).
    if hot:
        lo, up = max(tol, int(0.60 * delta)), max(4, tol // 2)
    else:
        lo, up = max(4, tol // 2), max(tol, int(0.60 * delta))
    return _seed_component_mask(gray, (px, py), lo, up, connectivity=8)


def _score_click_region(
    region: np.ndarray,
    point: tuple[int, int],
    frame_wh: tuple[int, int],
    *,
    prefer_compact: bool = True,
) -> tuple[float, Roi] | None:
    h, w = frame_wh
    px, py = point
    roi = _bbox_from_mask(region, mass_keep=0.97)
    if roi is None:
        return None
    x0, y0, rw, rh = roi
    if rw < 8 or rh < 8:
        return None
    area = float(rw * rh)
    fill = float(region[y0 : y0 + rh, x0 : x0 + rw].sum()) / max(area, 1.0)
    if fill < 0.25:
        return None
    cx, cy = x0 + rw / 2.0, y0 + rh / 2.0
    dist = float(np.hypot(cx - px, cy - py))
    diag = float(np.hypot(max(rw, 1), max(rh, 1)))
    center = max(0.0, 1.0 - dist / max(0.75 * diag, 1.0))
    # Компактность важнее «чем больше, тем лучше».
    size_term = min(area, 5000.0) / 5000.0
    score = (0.55 + 0.45 * fill) * center * (0.45 + 0.55 * size_term)
    if prefer_compact:
        frac = area / max(float(h * w), 1.0)
        if frac > 0.06:
            score *= 0.65
        if frac > 0.10:
            score *= 0.45
    if area < 0.0012 * h * w:
        score *= 0.55
    return score, (x0, y0, rw, rh)


def estimate_roi_from_point(
    frame_bgr: np.ndarray,
    point: tuple[int, int],
    *,
    tolerance: int = 16,
    grabcut_refine: bool = True,
    max_side_fraction: float = 0.35,
    max_area_fraction: float = 0.08,
    pad_px: int = 2,
) -> Roi | None:
    """ROI по клику: плотно по объекту (без рамки ×2).

    grabcut_refine оставлен для совместимости CLI (не используется).
    """
    del grabcut_refine
    frame_bgr = _as_bgr(frame_bgr)
    h, w = frame_bgr.shape[:2]
    px, py = int(point[0]), int(point[1])
    if not (0 <= px < w and 0 <= py < h):
        return None

    # Порог по «сырому» gray даёт меньше ореола, чем по CLAHE.
    gray_raw = _as_gray(frame_bgr)
    gray_enh = _enhance_thermal(gray_raw)

    best_roi: Roi | None = None
    best_score = -1.0
    tols = sorted(
        {
            max(6, int(tolerance * 0.75)),
            max(8, int(tolerance)),
        }
    )

    thermal_regions: list[np.ndarray] = []
    for g in (gray_raw, gray_enh):
        for tol in tols:
            m = _thermal_seed_mask(g, (px, py), tol)
            if m is not None:
                thermal_regions.append(m)

    def _too_big(rw: int, rh: int) -> bool:
        if rw > w * max_side_fraction or rh > h * max_side_fraction:
            return True
        if rw * rh > h * w * max_area_fraction:
            return True
        return False

    def _consider(regions: list[np.ndarray], *, bonus: float = 0.0) -> None:
        nonlocal best_roi, best_score
        for region in regions:
            scored = _score_click_region(region, (px, py), (h, w))
            if scored is None:
                continue
            score, roi = scored
            _x, _y, rw, rh = roi
            if _too_big(rw, rh):
                continue
            score += bonus
            # Среди похожих по score предпочитаем более компактный ROI.
            score -= 0.15 * (rw * rh) / max(float(h * w), 1.0)
            if score > best_score:
                best_score = score
                best_roi = roi

    _consider(thermal_regions, bonus=0.05)

    if best_roi is None:
        fallback: list[np.ndarray] = []
        for tol in tols:
            m2 = _seed_component_mask(gray_enh, (px, py), tol, tol, connectivity=8)
            if m2 is not None:
                fallback.append(m2)
        _consider(fallback, bonus=0.0)

    if best_roi is None:
        side = max(24, int(min(h, w) * 0.06))
        return clamp_roi((px - side // 2, py - side // 2, side, side), w, h)
    return _pad_roi(best_roi, w, h, pad_px=pad_px)


def select_object_by_click(
    frame_bgr: np.ndarray,
    max_display: int = 1200,
    *,
    tolerance: int = 16,
    grabcut_refine: bool = True,
) -> Roi | None:
    frame_bgr = _as_bgr(frame_bgr)
    scale = display_scale(frame_bgr.shape, max_display)
    preview = fit_for_display(frame_bgr, scale)
    inv = 1.0 / scale if scale > 0 else 1.0
    window = "Click object (click=full object, +/-=tol, Enter=OK, C/Esc=cancel)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    state: dict = {"roi": None, "pt": None, "tol": int(tolerance)}

    def _recompute() -> None:
        if state["pt"] is None:
            return
        state["roi"] = estimate_roi_from_point(
            frame_bgr,
            state["pt"],
            tolerance=int(state["tol"]),
            grabcut_refine=grabcut_refine,
        )

    def on_mouse(event: int, mx: int, my: int, flags: int, userdata) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state["pt"] = (int(mx * inv), int(my * inv))
        _recompute()

    cv2.setMouseCallback(window, on_mouse)
    while True:
        vis = preview.copy()
        roi = state["roi"]
        if roi is not None:
            x, y, rw, rh = roi
            cv2.rectangle(
                vis,
                (int(x * scale), int(y * scale)),
                (int((x + rw) * scale), int((y + rh) * scale)),
                (0, 220, 0),
                2,
            )
        if state["pt"] is not None:
            cv2.drawMarker(
                vis,
                (int(state["pt"][0] * scale), int(state["pt"][1] * scale)),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                12,
                1,
            )
        cv2.putText(
            vis,
            f"tol={state['tol']}  (+/-)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32, 10):
            cv2.destroyWindow(window)
            return state["roi"]
        if key in (ord("c"), ord("C"), 27):
            cv2.destroyWindow(window)
            return None
        if key in (ord("+"), ord("="), ord("]")):
            state["tol"] = min(64, int(state["tol"]) + 2)
            _recompute()
        if key in (ord("-"), ord("_"), ord("[")):
            state["tol"] = max(3, int(state["tol"]) - 2)
            _recompute()


def _extract_patch(gray: np.ndarray, roi: Roi) -> np.ndarray | None:
    x, y, rw, rh = clamp_roi(roi, gray.shape[1], gray.shape[0])
    patch = gray[y : y + rh, x : x + rw]
    if patch.size < 4:
        return None
    return patch.copy()


def _ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size < 4:
        return -1.0
    af = a.astype(np.float32).ravel()
    bf = b.astype(np.float32).ravel()
    af -= af.mean()
    bf -= bf.mean()
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(af, bf) / denom)


def _score_at(
    gray: np.ndarray, template: np.ndarray, top_left: tuple[float, float]
) -> float:
    th, tw = template.shape[:2]
    x, y = int(round(top_left[0])), int(round(top_left[1]))
    h, w = gray.shape[:2]
    if x < 0 or y < 0 or x + tw > w or y + th > h:
        return -1.0
    return _ncc_score(gray[y : y + th, x : x + tw], template)


def _match_template_local(
    gray: np.ndarray,
    template: np.ndarray,
    center: tuple[float, float],
    *,
    max_shift_x: float,
    max_shift_y: float,
) -> tuple[Roi | None, float, float]:
    th, tw = template.shape[:2]
    if th < 2 or tw < 2:
        return None, -1.0, 0.0
    h, w = gray.shape[:2]
    cx, cy = center
    x0 = max(0, int(cx - max_shift_x - tw / 2))
    y0 = max(0, int(cy - max_shift_y - th / 2))
    x1 = min(w, int(cx + max_shift_x + tw / 2 + 1))
    y1 = min(h, int(cy + max_shift_y + th / 2 + 1))
    if x1 - x0 < tw or y1 - y0 < th:
        return None, -1.0, 0.0
    region = gray[y0:y1, x0:x1]
    if float(np.var(template)) < 1.5:
        res = cv2.matchTemplate(region, template, cv2.TM_SQDIFF_NORMED)
        score_map = 1.0 - res.astype(np.float32)
    else:
        res = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        score_map = res.astype(np.float32)

    flat = score_map.ravel()
    if flat.size == 0:
        return None, -1.0, 0.0
    best_i = int(np.argmax(flat))
    best = float(flat[best_i])
    if not np.isfinite(best):
        return None, -1.0, 0.0
    by, bx = np.unravel_index(best_i, score_map.shape)
    mask = np.ones_like(score_map, dtype=bool)
    y_lo, y_hi = max(0, by - 3), min(score_map.shape[0], by + 4)
    x_lo, x_hi = max(0, bx - 3), min(score_map.shape[1], bx + 4)
    mask[y_lo:y_hi, x_lo:x_hi] = False
    second = float(np.max(score_map[mask])) if np.any(mask) else best * 0.5
    peak_ratio = best / max(abs(second), 1e-3) if second > -0.5 else 99.0
    if second <= 0 and best > 0:
        peak_ratio = max(peak_ratio, 2.0)
    return (int(x0 + bx), int(y0 + by), tw, th), best, float(peak_ratio)


def _resize_templ(template: np.ndarray, tw: int, th: int) -> np.ndarray:
    if template.shape[1] == tw and template.shape[0] == th:
        return template
    interp = (
        cv2.INTER_LINEAR
        if tw * th >= template.shape[0] * template.shape[1]
        else cv2.INTER_AREA
    )
    return cv2.resize(template, (tw, th), interpolation=interp)


class ObjectTracker:
    """
    Thermal IR tracker (640×512 grayscale):

    - base: KCF/CSRT/MOSSE или NCC (как DepthMapKornia, kind=\"ncc\")
    - CLAHE + NCC: позиция и масштаб
    - mean intensity: тепловая сигнатура (не цеплять «холодный» фон)
    - reacquire: локально + строгие гейты (NCC, peak, intensity)
    """

    def __init__(
        self,
        kind: str = "kcf",
        *,
        smooth: float = 0.0,
        lock_size: bool = False,
        keep_aspect: bool = True,
        max_scale_step: float = 0.07,
        verify: bool = True,
        verify_threshold: float = 0.30,
        max_jump: float = 4.0,
        min_visible: float = 0.18,
        lost_patience: int = 14,
        verify_rel: float = 0.0,
        min_iou: float = 0.0,
        max_size_ratio: float = 2.5,
        reacquire: bool = True,
        reacquire_threshold: float = 0.55,
        reacquire_radius: float = 3.5,
        reacquire_global: bool = False,
        reacquire_interval: int = 2,
        reacquire_scale_min: float = 0.50,
        reacquire_scale_max: float = 2.0,
        refine_radius: float = 0.50,
        template_update: float = 0.04,
        min_peak_ratio: float = 1.12,
        scale_min: float = 0.45,
        scale_max: float = 2.4,
        intensity_tol: float = 28.0,
        **_legacy: object,
    ) -> None:
        del _legacy
        self.kind = kind.lower()
        if self.kind not in TRACKER_KINDS:
            raise ValueError(
                f"Неизвестный трекер '{kind}'. Доступны: {', '.join(TRACKER_KINDS)}."
            )
        if not 0.0 <= smooth < 1.0:
            raise ValueError("smooth должен быть в диапазоне [0.0, 1.0).")

        self.smooth = float(smooth)
        self.lock_size = bool(lock_size)
        self.keep_aspect = bool(keep_aspect)
        self.max_scale_step = float(max_scale_step)
        self.verify = bool(verify)
        self.verify_threshold = float(verify_threshold)
        self.max_jump = float(max_jump)
        self.min_visible = float(min_visible)
        self.lost_patience = max(1, int(lost_patience))
        self.verify_rel = float(verify_rel)
        self.min_iou = float(min_iou)
        self.max_size_ratio = float(max_size_ratio)
        self.reacquire = bool(reacquire)
        self.reacquire_threshold = float(reacquire_threshold)
        self.reacquire_radius = float(reacquire_radius)
        self.reacquire_global = bool(reacquire_global)
        self.reacquire_interval = max(1, int(reacquire_interval))
        self.reacquire_scale_min = float(reacquire_scale_min)
        self.reacquire_scale_max = float(reacquire_scale_max)
        self.refine_radius = float(refine_radius)
        self.template_update = float(np.clip(template_update, 0.0, 1.0))
        self.min_peak_ratio = float(min_peak_ratio)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.intensity_tol = float(intensity_tol)

        self._tracker = None
        self.roi: Roi | None = None
        self._roi_f: tuple[float, float, float, float] | None = None
        self._init_size: tuple[float, float] | None = None
        self._locked_size: tuple[float, float] | None = None
        self._scale = 1.0
        self._template: np.ndarray | None = None
        self._template0: np.ndarray | None = None
        self._mean_i: float | None = None
        self._std_i: float | None = None
        self._fail_streak = 0
        self._since_reacquire = 0
        self._lost_frames = 0
        self._score_ema: float | None = None
        self._vel = (0.0, 0.0)
        self._frame_wh: tuple[int, int] | None = None
        self.initialized = False
        self.ok = False
        self.reacquired = False
        self.last_score: float | None = None

    def reset(self) -> None:
        self._tracker = None
        self.roi = None
        self._roi_f = None
        self._init_size = None
        self._locked_size = None
        self._scale = 1.0
        self._template = None
        self._template0 = None
        self._mean_i = None
        self._std_i = None
        self._fail_streak = 0
        self._since_reacquire = 0
        self._lost_frames = 0
        self._score_ema = None
        self._vel = (0.0, 0.0)
        self._frame_wh = None
        self.initialized = False
        self.ok = False
        self.reacquired = False
        self.last_score = None

    def _capture_appearance(self, raw: np.ndarray, enh: np.ndarray, roi: Roi) -> None:
        patch_e = _extract_patch(enh, roi)
        patch_r = _extract_patch(raw, roi)
        self._template = None if patch_e is None else patch_e.copy()
        self._template0 = None if patch_e is None else patch_e.copy()
        if patch_r is not None:
            self._mean_i = float(np.mean(patch_r))
            self._std_i = float(np.std(patch_r))
        else:
            self._mean_i = None
            self._std_i = None

    def init(self, frame: np.ndarray, roi: Roi) -> Roi:
        img = _as_bgr(frame)
        raw = _as_gray(img)
        enh = _enhance_thermal(raw)
        h, w = raw.shape[:2]
        roi = clamp_roi(roi, w, h)
        self._tracker = create_raw_tracker(self.kind)
        self._tracker.init(img, roi)
        self.roi = roi
        self._roi_f = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
        self._init_size = (float(roi[2]), float(roi[3]))
        self._locked_size = (float(roi[2]), float(roi[3]))
        self._scale = 1.0
        self._capture_appearance(raw, enh, roi)
        self._fail_streak = 0
        self._since_reacquire = 0
        self._lost_frames = 0
        self._score_ema = None
        self._vel = (0.0, 0.0)
        self._frame_wh = (w, h)
        self.initialized = True
        self.ok = True
        self.reacquired = False
        self.last_score = 1.0
        return self.roi

    def init_interactive(self, frame: np.ndarray, max_display: int = 1200) -> Roi | None:
        roi = select_object_roi(frame, max_display)
        if roi is None:
            return None
        return self.init(frame, roi)

    def init_by_click(
        self,
        frame: np.ndarray,
        max_display: int = 1200,
        *,
        tolerance: int = 12,
        grabcut_refine: bool = True,
    ) -> Roi | None:
        roi = select_object_by_click(
            frame,
            max_display,
            tolerance=tolerance,
            grabcut_refine=grabcut_refine,
        )
        if roi is None:
            return None
        return self.init(frame, roi)

    def _visible_fraction(
        self, box: tuple[float, float, float, float], w: int, h: int
    ) -> float:
        x, y, bw, bh = box
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
        vis = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        return float(vis / max(bw * bh, 1.0))

    def _templ_at_scale(self, scale: float) -> np.ndarray | None:
        base = self._template0 if self._template0 is not None else self._template
        if base is None or self._init_size is None:
            return None
        w0, h0 = self._init_size
        tw = max(8, int(round(w0 * scale)))
        th = max(8, int(round(h0 * scale)))
        return _resize_templ(base, tw, th)

    def _apply_scale(self, scale: float, *, hard: bool = False) -> tuple[float, float]:
        scale = float(np.clip(scale, self.scale_min, self.scale_max))
        if not hard and self.max_scale_step > 0.0:
            lo = self._scale * (1.0 - self.max_scale_step)
            hi = self._scale * (1.0 + self.max_scale_step)
            scale = float(np.clip(scale, lo, hi))
        self._scale = scale
        if self._init_size is None:
            return 8.0, 8.0
        w0, h0 = self._init_size
        nw, nh = w0 * scale, h0 * scale
        self._locked_size = (nw, nh)
        templ = self._templ_at_scale(scale)
        if templ is not None:
            self._template = templ
        return nw, nh

    def _box_size(self) -> tuple[float, float]:
        if self._locked_size is not None:
            return self._locked_size
        if self._init_size is not None:
            return self._init_size
        return 8.0, 8.0

    def _search_scales(self) -> list[float]:
        if self.lock_size:
            return [self._scale]
        factors = (0.88, 0.94, 1.0, 1.06, 1.14)
        out: list[float] = []
        for f in factors:
            s = float(np.clip(self._scale * f, self.scale_min, self.scale_max))
            if not out or abs(s - out[-1]) > 0.012:
                out.append(s)
        return out

    def _intensity_ok(self, raw: np.ndarray, roi: Roi) -> bool:
        if self._mean_i is None:
            return True
        patch = _extract_patch(raw, roi)
        if patch is None:
            return False
        mean = float(np.mean(patch))
        # На ТВ яркость объекта относительно стабильна; фон часто сильно отличается.
        tol = self.intensity_tol
        if self._std_i is not None:
            tol = max(tol, 1.8 * self._std_i)
        return abs(mean - self._mean_i) <= tol

    def _intensity_delta(self, raw: np.ndarray, roi: Roi) -> float:
        if self._mean_i is None:
            return 0.0
        patch = _extract_patch(raw, roi)
        if patch is None:
            return 1e3
        return abs(float(np.mean(patch)) - self._mean_i)

    def _jump_too_far(self, ncx: float, ncy: float) -> bool:
        if self.max_jump <= 0.0 or self._roi_f is None:
            return False
        px, py, pw, ph = self._roi_f
        pcx, pcy = px + pw / 2.0, py + ph / 2.0
        dx, dy = abs(ncx - pcx), abs(ncy - pcy)
        fw = float(self._frame_wh[0]) if self._frame_wh else 640.0
        lim_x = max(self.max_jump * max(pw, 1.0), 0.12 * fw, 32.0)
        lim_y = max(0.30 * self.max_jump * max(ph, 1.0), 10.0)
        return dx > lim_x or dy > lim_y

    def _stabilize(
        self,
        ncx: float,
        ncy: float,
        *,
        cand: float,
        anchor: float,
        bw: float,
        bh: float,
    ) -> tuple[float, float]:
        if self._roi_f is None:
            return ncx, ncy
        px, py, pw, ph = self._roi_f
        pcx, pcy = px + pw / 2.0, py + ph / 2.0
        dx, dy = ncx - pcx, ncy - pcy
        margin = 0.04
        if cand < anchor + margin:
            if abs(dx) > 0.12 * bw and cand >= anchor - 0.02:
                ncx = pcx + 0.60 * dx
            else:
                ncx = pcx
            ncy = pcy
            return ncx, ncy
        max_dx = max(10.0, 0.65 * bw)
        max_dy = max(3.0, 0.10 * bh)
        dx = float(np.clip(dx, -max_dx, max_dx))
        dy = float(np.clip(dy, -max_dy, max_dy))
        return pcx + dx, pcy + 0.30 * dy

    def _smooth_box(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x, y, bw, bh = box
        cx, cy = x + bw / 2.0, y + bh / 2.0
        a = 1.0 - self.smooth if self.smooth > 0.0 else 1.0
        if self._roi_f is None:
            ncx, ncy = cx, cy
        else:
            px, py, pw, ph = self._roi_f
            pcx, pcy = px + pw / 2.0, py + ph / 2.0
            dist = float(np.hypot(cx - pcx, cy - pcy))
            diag = float(np.hypot(max(pw, 1.0), max(ph, 1.0)))
            a_pos = 1.0 if diag > 1.0 and dist > 0.25 * diag else a
            ncx = pcx + a_pos * (cx - pcx)
            ncy = pcy + a_pos * (cy - pcy)
        nw, nh = self._box_size()
        return (ncx - nw / 2.0, ncy - nh / 2.0, nw, nh)

    def _mark_lost(self) -> tuple[bool, Roi | None]:
        self.ok = False
        self._lost_frames = 0
        self._since_reacquire = 0
        return False, self.roi

    def _fail(
        self,
        img: np.ndarray | None = None,
        raw: np.ndarray | None = None,
        *,
        weight: int = 1,
        chase: tuple[float, float, float, float] | None = None,
        score: float = 0.0,
    ) -> tuple[bool, Roi | None]:
        self._fail_streak += max(1, int(weight))
        if self._fail_streak >= self.lost_patience:
            return self._mark_lost()
        if chase is not None and img is not None and raw is not None:
            # Chase только если тепловая сигнатура ещё похожа.
            croi = clamp_roi(chase, img.shape[1], img.shape[0])
            if self._intensity_ok(raw, croi):
                self._accept(
                    img, raw, None, chase, score, update_template=False, clear_fail=False
                )
                return True, self.roi
        self.ok = True
        return True, self.roi

    def _accept(
        self,
        img: np.ndarray,
        raw: np.ndarray,
        enh: np.ndarray | None,
        box: tuple[float, float, float, float],
        score: float,
        *,
        update_template: bool = True,
        clear_fail: bool = True,
    ) -> None:
        prev = self._roi_f
        smoothed = self._smooth_box(box)
        self._roi_f = smoothed
        self.roi = clamp_roi(smoothed, img.shape[1], img.shape[0])
        self.ok = True
        if clear_fail:
            self._fail_streak = 0
        self._lost_frames = 0
        self.last_score = float(score)
        if prev is not None:
            pcx = prev[0] + prev[2] / 2.0
            pcy = prev[1] + prev[3] / 2.0
            ncx = smoothed[0] + smoothed[2] / 2.0
            ncy = smoothed[1] + smoothed[3] / 2.0
            dx, dy = ncx - pcx, ncy - pcy
            vx, vy = self._vel
            self._vel = (0.55 * vx + 0.45 * dx, 0.35 * vy + 0.25 * dy)
        if self._score_ema is None:
            self._score_ema = float(score)
        else:
            self._score_ema = 0.88 * self._score_ema + 0.12 * float(score)

        if (
            update_template
            and enh is not None
            and score >= max(0.50, self.verify_threshold + 0.12)
            and self.template_update > 0
            and self.roi is not None
        ):
            patch_e = _extract_patch(enh, self.roi)
            patch_r = _extract_patch(raw, self.roi)
            if patch_e is not None and self._template is not None:
                if patch_e.shape == self._template.shape:
                    a = self.template_update
                    self._template = cv2.addWeighted(
                        patch_e, a, self._template, 1.0 - a, 0
                    )
                else:
                    self._template = patch_e.copy()
                if self._template0 is not None and self._init_size is not None:
                    w0, h0 = int(self._init_size[0]), int(self._init_size[1])
                    canon = _resize_templ(patch_e, w0, h0)
                    if canon.shape == self._template0.shape:
                        a = min(0.03, self.template_update)
                        self._template0 = cv2.addWeighted(
                            canon, a, self._template0, 1.0 - a, 0
                        )
            if patch_r is not None and self._mean_i is not None:
                m = float(np.mean(patch_r))
                self._mean_i = 0.92 * self._mean_i + 0.08 * m
                self._std_i = float(np.std(patch_r))

    def _reinit_tracker(self, img: np.ndarray, roi: Roi) -> None:
        self._tracker = create_raw_tracker(self.kind)
        self._tracker.init(img, roi)

    def _attempt_reacquire(
        self, img: np.ndarray, raw: np.ndarray, enh: np.ndarray
    ) -> bool:
        if not self.reacquire or self._template0 is None or self._roi_f is None:
            return False
        if self._lost_frames < 1:
            return False

        px, py, pw, ph = self._roi_f
        pcx, pcy = px + pw / 2.0, py + ph / 2.0
        if self._locked_size is not None:
            pw, ph = self._locked_size
        vx, vy = self._vel
        # Ищем впереди по траектории (типично L→R на ТВ).
        search_cx = pcx + vx * min(4.0, 0.6 * self._lost_frames)
        search_cy = pcy + 0.2 * vy * min(4.0, 0.6 * self._lost_frames)

        base_r = max(pw, ph) * self.reacquire_radius
        grow = 1.0 + 0.28 * min(max(0, self._lost_frames - 1), 12)
        radius_x = base_r * grow
        radius_y = base_r * grow * 0.45
        if self.reacquire_global:
            radius_x = max(radius_x, float(max(enh.shape[:2])))
            radius_y = max(radius_y, float(max(enh.shape[:2])) * 0.5)

        # Масштабы относительно текущего (приближение/удаление).
        scales = np.linspace(
            max(self.scale_min, self._scale * self.reacquire_scale_min),
            min(self.scale_max, self._scale * self.reacquire_scale_max),
            7,
        )

        best_roi: Roi | None = None
        best_score = -1.0
        best_ratio = 0.0
        best_scale = self._scale
        for s in scales:
            templ = self._templ_at_scale(float(s))
            if templ is None:
                continue
            # Не ищем крошечными шаблонами — на ТВ ловят шум.
            if templ.size < 64:
                continue
            roi, score, ratio = _match_template_local(
                enh,
                templ,
                (search_cx, search_cy),
                max_shift_x=radius_x,
                max_shift_y=radius_y,
            )
            if roi is None:
                continue
            # Штраф неоднозначным пикам.
            adj = score - (0.08 if ratio < self.min_peak_ratio else 0.0)
            if adj > best_score:
                best_score = score
                best_roi = roi
                best_ratio = ratio
                best_scale = float(s)

        self.last_score = best_score
        if (
            best_roi is None
            or best_score < self.reacquire_threshold
            or best_ratio < self.min_peak_ratio
        ):
            return False

        ncx = best_roi[0] + best_roi[2] / 2.0
        ncy = best_roi[1] + best_roi[3] / 2.0
        far = self._jump_too_far(ncx, ncy)
        if far and not self.reacquire_global:
            # Дальний прыжок только при очень похожем объекте.
            if (
                best_score < max(0.68, self.reacquire_threshold + 0.12)
                or best_ratio < 1.20
                or self._lost_frames < 5
            ):
                return False

        self._apply_scale(best_scale, hard=True)
        lw, lh = self._box_size()
        roi = clamp_roi(
            (ncx - lw / 2.0, ncy - lh / 2.0, lw, lh),
            img.shape[1],
            img.shape[0],
        )
        if not self._intensity_ok(raw, roi):
            return False
        # Финальный confirm NCC.
        templ = self._template
        if templ is None:
            return False
        th, tw = templ.shape[:2]
        confirm = _score_at(enh, templ, (roi[0], roi[1]))
        if confirm < self.reacquire_threshold * 0.95:
            return False

        patch = _extract_patch(enh, roi)
        if patch is not None:
            self._template = patch
        self._reinit_tracker(img, roi)
        self._roi_f = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
        self.roi = roi
        self.ok = True
        self._fail_streak = 0
        self._lost_frames = 0
        self.reacquired = True
        self._score_ema = float(best_score)
        self.last_score = float(confirm)
        return True

    def update(self, frame: np.ndarray) -> tuple[bool, Roi | None]:
        if not self.initialized or self._tracker is None:
            raise RuntimeError("ObjectTracker не инициализирован: вызовите init().")
        img = _as_bgr(frame)
        raw = _as_gray(img)
        enh = _enhance_thermal(raw)
        h, w = raw.shape[:2]
        self._frame_wh = (w, h)
        self.reacquired = False

        if not self.ok:
            self._lost_frames += 1
            self._since_reacquire += 1
            if self._since_reacquire >= self.reacquire_interval:
                self._since_reacquire = 0
                if self._attempt_reacquire(img, raw, enh):
                    return True, self.roi
            return False, self.roi

        raw_ok, box = self._tracker.update(img)
        bw, bh = self._box_size()

        if not raw_ok:
            self.last_score = None
            if self._roi_f is not None:
                px, py, pw, ph = self._roi_f
                vx, vy = self._vel
                pred = (px + vx, py + 0.25 * vy, pw, ph)
                return self._fail(img, raw, weight=1, chase=pred, score=0.0)
            return self._fail(weight=1)

        cx = float(box[0]) + float(box[2]) / 2.0
        cy = float(box[1]) + float(box[3]) / 2.0
        bx = (cx - bw / 2.0, cy - bh / 2.0, bw, bh)

        if self._visible_fraction(bx, w, h) < self.min_visible:
            return self._fail(img, raw, weight=2, chase=bx, score=0.0)

        if not self.verify or self._template0 is None:
            if self._jump_too_far(cx, cy):
                return self._fail(img, raw, weight=1, chase=bx, score=0.0)
            self._accept(img, raw, enh, bx, 1.0)
            return True, self.roi

        # Якорь + предсказание (X свободнее, Y почти фиксирован).
        if self._roi_f is not None:
            px, py, pw, ph = self._roi_f
            pcx, pcy = px + pw / 2.0, py + ph / 2.0
            vx, vy = self._vel
            pred_cx = pcx + vx
            search_cx = 0.55 * cx + 0.45 * pred_cx
            search_cy = 0.88 * pcy + 0.12 * cy
        else:
            pcx, pcy = cx, cy
            search_cx, search_cy = cx, cy

        max_shift_x = max(
            12.0, self.refine_radius * 2.0 * bw, 0.65 * abs(self._vel[0]) + 8.0
        )
        max_shift_y = max(3.0, self.refine_radius * 0.25 * bh)

        # Score на якоре.
        anchor_t = self._templ_at_scale(self._scale)
        if anchor_t is not None:
            ath, atw = anchor_t.shape[:2]
            anchor = _score_at(enh, anchor_t, (pcx - atw / 2.0, pcy - ath / 2.0))
            if not np.isfinite(anchor):
                anchor = -1.0
        else:
            anchor = -1.0

        best = -1.0
        best_roi: Roi | None = None
        best_scale = self._scale
        best_peak = False
        for s in self._search_scales():
            templ = self._templ_at_scale(s)
            if templ is None:
                continue
            th, tw = templ.shape[:2]
            at = _score_at(enh, templ, (search_cx - tw / 2.0, search_cy - th / 2.0))
            ncc_roi, ncc_score, peak_ratio = _match_template_local(
                enh,
                templ,
                (search_cx, search_cy),
                max_shift_x=max_shift_x,
                max_shift_y=max_shift_y,
            )
            cand = float(at) if np.isfinite(at) else -1.0
            cand_roi = (
                int(round(search_cx - tw / 2.0)),
                int(round(search_cy - th / 2.0)),
                tw,
                th,
            )
            used_peak = False
            if (
                ncc_roi is not None
                and np.isfinite(ncc_score)
                and ncc_score + 0.015 >= cand
                and peak_ratio >= self.min_peak_ratio
            ):
                ncx = ncc_roi[0] + ncc_roi[2] / 2.0
                ncy = ncc_roi[1] + ncc_roi[3] / 2.0
                if abs(ncy - pcy) <= 0.18 * bh:
                    cand = float(ncc_score)
                    cand_roi = ncc_roi
                    used_peak = True

            # Штрафы: чужой масштаб и диагональный уход.
            tcx = cand_roi[0] + cand_roi[2] / 2.0
            tcy = cand_roi[1] + cand_roi[3] / 2.0
            adj = cand
            adj -= 0.025 * abs(s - self._scale) / max(self._scale, 0.2)
            adj -= 0.05 * (abs(tcy - pcy) / max(bh, 1.0))
            # Intensity gate на кандидате.
            probe = clamp_roi(
                (tcx - bw / 2.0, tcy - bh / 2.0, bw, bh), w, h
            )
            dI = self._intensity_delta(raw, probe)
            if dI > self.intensity_tol * 1.35:
                adj -= 0.12

            if adj > best:
                best = cand
                best_roi = cand_roi
                best_scale = s
                best_peak = used_peak

        self.last_score = best
        soft = self.verify_threshold * 0.68

        if best_roi is not None and best >= soft:
            ncx = best_roi[0] + best_roi[2] / 2.0
            ncy = best_roi[1] + best_roi[3] / 2.0
            ncx, ncy = self._stabilize(
                ncx, ncy, cand=best, anchor=float(anchor), bw=bw, bh=bh
            )
            if (
                abs(best_scale - self._scale) > 0.03
                and not self.lock_size
                and best >= float(anchor) + 0.04
                and (best_peak or best >= self.verify_threshold + 0.05)
            ):
                bw, bh = self._apply_scale(best_scale, hard=False)
            else:
                bw, bh = self._box_size()
            refined = (ncx - bw / 2.0, ncy - bh / 2.0, bw, bh)
        else:
            # Держим якорь по Y, слегка к KCF по X.
            ncx = 0.55 * cx + 0.45 * pcx
            ncy = pcy
            refined = (ncx - bw / 2.0, ncy - bh / 2.0, bw, bh)

        chase = (cx - bw / 2.0, pcy - bh / 2.0, bw, bh)

        if self._jump_too_far(cx, cy) and best < soft:
            return self._fail(img, raw, weight=1, chase=chase, score=best)
        if best < 0.0 or not np.isfinite(best):
            return self._fail(img, raw, weight=1, chase=chase, score=0.0)
        if best < soft:
            return self._fail(img, raw, weight=1, chase=chase, score=best)

        croi = clamp_roi(refined, w, h)
        if not self._intensity_ok(raw, croi) and best < self.verify_threshold + 0.08:
            return self._fail(img, raw, weight=1, chase=chase, score=best)

        if best < self.verify_threshold:
            self._fail_streak += 1
            if self._fail_streak >= self.lost_patience:
                return self._mark_lost()
            self._accept(
                img, raw, enh, refined, best, update_template=False, clear_fail=False
            )
            return True, self.roi

        self._accept(img, raw, enh, refined, best)
        # Гасим вертикальную скорость.
        vx, vy = self._vel
        self._vel = (vx, 0.45 * vy)
        if self.roi is not None:
            rcx = refined[0] + refined[2] / 2.0
            if abs(rcx - cx) > 0.40 * max(bw, 1.0):
                self._reinit_tracker(img, self.roi)
        return True, self.roi


def draw_tracking(
    frame_bgr: np.ndarray,
    roi: Roi | None,
    ok: bool,
    *,
    score: float | None = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    if roi is not None:
        x, y, rw, rh = roi
        color = (0, 220, 0) if ok else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + rw, y + rh), color, 2)
    label = "OK" if ok else "LOST"
    if score is not None and np.isfinite(score):
        label += f"  ncc={score:.2f}"
    cv2.putText(
        out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        out,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Тест ObjectTracker для thermal IR (KCF+CLAHE-NCC)."
    )
    p.add_argument("--video", required=True)
    p.add_argument("--tracker", choices=TRACKER_KINDS, default="kcf")
    p.add_argument("--max-display", type=int, default=1200)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Не удалось открыть '{args.video}'.")
    ok, frame = cap.read()
    if not ok:
        sys.exit("Пустое видео.")
    tracker = ObjectTracker(kind=args.tracker)
    window = "ObjectTracker thermal (R=box, C=click, Q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("R — рамка, C — клик, Q — выход.")
    try:
        while True:
            if tracker.initialized:
                tracking_ok, roi = tracker.update(frame)
            else:
                tracking_ok, roi = False, None
            vis = draw_tracking(
                _as_bgr(frame), roi, tracking_ok, score=tracker.last_score
            )
            scale = display_scale(vis.shape, args.max_display)
            cv2.imshow(window, fit_for_display(vis, scale))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                tracker.init_interactive(frame, args.max_display)
            if key in (ord("c"), ord("C")):
                tracker.init_by_click(frame, args.max_display)
            ok, frame = cap.read()
            if not ok:
                print("Конец видео.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
