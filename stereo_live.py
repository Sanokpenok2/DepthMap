"""Live stereo tracking + distance for video_record (and similar UIs).

Reuses prepare/disparity/tracker helpers from video_track_depth / object_tracker /
depth_map — no duplicated SGBM logic.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from depth_map import load_calibration, measure_roi_distance
from object_tracker import ObjectTracker
from stereo_auto import (
    clamp_sgbm_range,
    disparity_from_depth,
    estimate_disparity_range_bounds,
    extract_calib_geometry,
)
from video_track_depth import (
    DistanceSmoother,
    adapt_disparity_range,
    compute_disparity,
    draw_overlay,
    make_stereo_matcher,
    prepare_pair,
)


@dataclass
class LiveStatus:
    enabled: bool = False
    has_calib: bool = False
    tracking_ok: bool = False
    roi: tuple[int, int, int, int] | None = None
    distance_mm: float | None = None
    disparity_px: float | None = None
    disp_min: int = 0
    disp_num: int = 128
    sgbm_busy: bool = False
    message: str = ""


class LiveTrackDepthController:
    """Process paired L/R frames: track on left, distance from stereo disparity."""

    def __init__(
        self,
        *,
        z_near_m: float = 10.0,
        z_far_m: float = 100.0,
        auto_disparity: bool = True,
        sgbm_interval: int = 2,
        smooth_window: int = 21,
        smooth_max_ratio: float = 2.0,
        smooth_ema: float = 0.2,
        smooth_disp_jump: float = 1.5,
        sync_tolerance_s: float = 0.08,
        method: str = "sgbm",
        block_size: int = 7,
        wls: bool = False,
        clahe: bool = True,
        roi_inset: float = 0.32,
        surface: str = "median",
        long_range: bool | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.z_near_m = float(z_near_m)
        self.z_far_m = float(z_far_m)
        self.auto_disparity = bool(auto_disparity)
        self.sgbm_interval = max(1, int(sgbm_interval))
        self.smooth_window = int(smooth_window)
        self.smooth_max_ratio = float(smooth_max_ratio)
        self.smooth_ema = float(smooth_ema)
        self.smooth_disp_jump = float(smooth_disp_jump)
        self.sync_tolerance_s = float(sync_tolerance_s)
        self.method = method
        self.block_size = int(block_size)
        self.wls = bool(wls)
        self.clahe = bool(clahe)
        self.roi_inset = float(roi_inset)
        self.surface = str(surface)
        # None = авто по z_far. Явный True при z_far<800 больше не форсирует long-range.
        self._long_range_pref = long_range
        self.long_range = self._resolve_long_range(long_range)
        self.max_disp_cap: float | None = None
        self.min_disp_floor = 0.85
        self.max_distance_mm: float | None = self.z_far_m * 1000.0 * 1.15
        self.uniqueness = 12 if self.long_range else 5
        self._on_log = on_log

        self._lock = threading.RLock()
        self.enabled = False
        self.track_only = False
        self.calib: dict | None = None
        self.Q = None
        self.disp_min = 0
        self.disp_num = 128
        self.matcher = None
        self.tracker = ObjectTracker(
            kind="kcf",
            smooth=0.0,
            lock_size=False,
            max_scale_step=0.08,
            max_size_ratio=2.5,
            verify_threshold=0.30,
            verify_rel=0.0,
            min_iou=0.0,
            max_jump=4.0,
            lost_patience=14,
            reacquire_threshold=0.55,
            reacquire_radius=3.5,
            reacquire_interval=2,
            reacquire_scale_min=0.50,
            reacquire_scale_max=2.0,
            intensity_tol=28.0,
        )
        self.dist_smoother = DistanceSmoother(
            window=self.smooth_window,
            max_ratio=self.smooth_max_ratio,
            ema_alpha=self.smooth_ema,
            max_disp_jump=self.smooth_disp_jump,
            outlier_patience=max(4, self.smooth_window // 3),
            max_distance_mm=self.max_distance_mm,
        )
        self.history: deque[float] = deque()  # legacy unused; kept for compat
        self.frame_idx = 0
        self.fps = 0.0
        self._t_prev = time.perf_counter()

        self._prep_pool = ThreadPoolExecutor(max_workers=2)
        self._sgbm_pool = ThreadPoolExecutor(max_workers=1)
        self._sgbm_future: Future | None = None
        self._disp_float: np.ndarray | None = None
        self._last_overlay: np.ndarray | None = None
        self._last_rect_l: np.ndarray | None = None
        self._last_rect_r: np.ndarray | None = None
        self._dist_s: float | None = None
        self._disp_val: float | None = None
        self._tracking_ok = False
        self._status_msg = ""

    @staticmethod
    def _resolve_long_range_flag(z_far_m: float, long_range: bool | None) -> bool:
        """long-range только для дальних сцен (z_far>=800); иначе обычный SGBM."""
        if z_far_m < 800.0:
            return False
        if long_range is None:
            return True
        return bool(long_range)

    def _resolve_long_range(self, long_range: bool | None) -> bool:
        return self._resolve_long_range_flag(self.z_far_m, long_range)

    def set_scene_range(
        self,
        z_near_m: float,
        z_far_m: float,
        *,
        long_range: bool | None = None,
    ) -> None:
        """Задать диапазон сцены (м) и пересобрать matcher при наличии калибровки."""
        if z_near_m <= 0 or z_far_m <= 0 or z_near_m >= z_far_m:
            raise ValueError("Нужно 0 < z_near_m < z_far_m.")
        self.z_near_m = float(z_near_m)
        self.z_far_m = float(z_far_m)
        self._long_range_pref = long_range
        self.long_range = self._resolve_long_range(long_range)
        self.uniqueness = 12 if self.long_range else 5
        self.max_distance_mm = self.z_far_m * 1000.0 * 1.15
        self.dist_smoother.max_distance_mm = self.max_distance_mm
        self._log(
            f"Сцена: z={self.z_near_m:.0f}–{self.z_far_m:.0f} м "
            f"(long_range={self.long_range})."
        )
        if self.calib is not None:
            self._rebuild_matcher_from_calib("reconfigure")

    def _log(self, msg: str) -> None:
        self._status_msg = msg
        if self._on_log is not None:
            self._on_log(msg)

    def shutdown(self) -> None:
        with self._lock:
            fut = self._sgbm_future
            self._sgbm_future = None
        if fut is not None:
            try:
                fut.result(timeout=5)
            except Exception:
                pass
        self._prep_pool.shutdown(wait=False)
        self._sgbm_pool.shutdown(wait=False)

    def load_calib(self, path: str) -> None:
        calib = load_calibration(path)
        self._rebuild_matcher_from_calib(path, calib=calib)
        with self._lock:
            self.calib = calib
            self.Q = calib["Q"]
            self.track_only = False
        self._log(f"Калибровка загружена: {path}")

    def _rebuild_matcher_from_calib(
        self, path: str, calib: dict | None = None
    ) -> None:
        if calib is None:
            calib = self.calib
        if calib is None:
            return
        width = 640
        if "image_size" in calib:
            width = int(calib["image_size"][0])
        self.long_range = self._resolve_long_range(self._long_range_pref)
        self.uniqueness = 12 if self.long_range else 5
        disp_min, disp_num = 0, 48
        if self.auto_disparity:
            disp_min, disp_num, range_log = estimate_disparity_range_bounds(
                calib,
                self.z_near_m,
                self.z_far_m,
                image_width=width,
                long_range=self.long_range,
            )
            max_num = 96 if self.long_range else 512
            disp_min, disp_num = clamp_sgbm_range(
                disp_min, disp_num, width, max_num=max_num
            )
            self._log(range_log)
        else:
            self._log(f"Фиксированный диапазон: min={disp_min}, num={disp_num}")
        try:
            focal, baseline = extract_calib_geometry(calib)
            self.max_disp_cap = (
                disparity_from_depth(focal, baseline, self.z_near_m * 1000.0) * 1.05
            )
            d_far = disparity_from_depth(focal, baseline, self.z_far_m * 1000.0)
            self.min_disp_floor = max(0.85, float(d_far) * 0.75)
            self.max_distance_mm = float(self.z_far_m) * 1000.0 * 1.15
            self.dist_smoother.max_distance_mm = self.max_distance_mm
        except Exception:
            self.max_disp_cap = float(disp_min + disp_num)
            self.max_distance_mm = float(self.z_far_m) * 1000.0 * 1.15
            self.dist_smoother.max_distance_mm = self.max_distance_mm
        with self._lock:
            self.disp_min = disp_min
            self.disp_num = disp_num
            self.matcher = make_stereo_matcher(
                self.method,
                self.disp_min,
                self.disp_num,
                self.block_size,
                uniqueness_ratio=self.uniqueness,
                speckle_window_size=40 if self.long_range else 50,
            )
        self._log(
            f"Диапазон сцены {self.z_near_m:.0f}–{self.z_far_m:.0f} м → "
            f"disp min={disp_min}, num={disp_num} "
            f"(long_range={self.long_range})."
        )

    def clear_calib(self) -> None:
        with self._lock:
            self.calib = None
            self.Q = None
            self.matcher = None
            self.track_only = True
        self._log("Калибровка сброшена (track-only).")

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = bool(enabled)
            if not self.enabled:
                self._last_overlay = None

    def reset(self) -> None:
        with self._lock:
            self.tracker = ObjectTracker(
                kind="kcf",
                smooth=0.0,
                lock_size=False,
                max_scale_step=0.08,
                max_size_ratio=2.5,
                verify_threshold=0.26,
                verify_rel=0.0,
                min_iou=0.0,
                max_jump=3.0,
                lost_patience=12,
                reacquire_threshold=0.40,
                reacquire_radius=4.5,
                reacquire_interval=2,
                reacquire_scale_min=0.45,
                reacquire_scale_max=2.0,
            )
            self.dist_smoother.reset()
            self._dist_s = None
            self._disp_val = None
            self._tracking_ok = False
            self._disp_float = None
            self._last_overlay = None
            self.frame_idx = 0
        self._log("Трекинг сброшен.")

    def status(self) -> LiveStatus:
        with self._lock:
            roi = self.tracker.roi if self.tracker.initialized else None
            return LiveStatus(
                enabled=self.enabled,
                has_calib=self.calib is not None,
                tracking_ok=self._tracking_ok,
                roi=roi,
                distance_mm=self._dist_s,
                disparity_px=self._disp_val,
                disp_min=self.disp_min,
                disp_num=self.disp_num,
                sgbm_busy=(
                    self._sgbm_future is not None and not self._sgbm_future.done()
                ),
                message=self._status_msg,
            )

    def last_overlay(self) -> np.ndarray | None:
        with self._lock:
            if self._last_overlay is None:
                return None
            return self._last_overlay.copy()

    def _snapshot_rect_left(self) -> np.ndarray | None:
        with self._lock:
            if self._last_rect_l is None:
                return None
            return cv2.cvtColor(self._last_rect_l, cv2.COLOR_GRAY2BGR)

    def init_roi_box(self, max_display: int = 1200) -> bool:
        snap = self._snapshot_rect_left()
        if snap is None:
            self._log("Нет кадра для выбора ROI.")
            return False
        roi = self.tracker.init_interactive(snap, max_display=max_display)
        if roi is None:
            self._log("Выбор ROI отменён.")
            return False
        with self._lock:
            self.dist_smoother.reset()
            self._tracking_ok = True
            self._dist_s = None
            self._disp_val = None
        self._log(f"ROI выбран рамкой: {roi}")
        return True

    def init_roi_click(self, max_display: int = 1200) -> bool:
        snap = self._snapshot_rect_left()
        if snap is None:
            self._log("Нет кадра для выбора ROI.")
            return False
        roi = self.tracker.init_by_click(snap, max_display=max_display)
        if roi is None:
            self._log("Выбор ROI кликом отменён.")
            return False
        with self._lock:
            self.dist_smoother.reset()
            self._tracking_ok = True
            self._dist_s = None
            self._disp_val = None
        self._log(f"ROI выбран кликом: {roi}")
        return True

    def init_roi(self, roi: tuple[int, int, int, int]) -> bool:
        """Инициализация трекера готовым ROI на последнем rectified left-кадре."""
        snap = self._snapshot_rect_left()
        if snap is None:
            self._log("Нет кадра для выбора ROI.")
            return False
        try:
            self.tracker.init(snap, roi)
        except Exception as exc:
            self._log(f"ROI отклонён: {exc}")
            return False
        with self._lock:
            self.dist_smoother.reset()
            self._tracking_ok = True
            self._dist_s = None
            self._disp_val = None
        self._log(f"ROI задан: {self.tracker.roi}")
        return True

    def init_roi_at_point(
        self,
        x: int,
        y: int,
        *,
        tolerance: int = 12,
        grabcut_refine: bool = False,
        fallback_box: int = 48,
    ) -> bool:
        """Клик по live-кадру: оценка ROI без отдельного OpenCV-окна."""
        snap = self._snapshot_rect_left()
        if snap is None:
            self._log("Нет кадра для выбора ROI.")
            return False
        from object_tracker import estimate_roi_from_point

        try:
            roi = estimate_roi_from_point(
                snap,
                (int(x), int(y)),
                tolerance=int(tolerance),
                grabcut_refine=bool(grabcut_refine),
            )
        except Exception:
            roi = None
        if roi is None:
            half = max(8, int(fallback_box) // 2)
            roi = (int(x) - half, int(y) - half, int(fallback_box), int(fallback_box))
        return self.init_roi(roi)

    def process(
        self,
        frame_l: np.ndarray,
        frame_r: np.ndarray,
        *,
        t_l: float | None = None,
        t_r: float | None = None,
    ) -> np.ndarray | None:
        """Process one stereo pair. Returns BGR overlay for left, or None."""
        if not self.enabled:
            return None
        if t_l is not None and t_r is not None:
            if abs(t_l - t_r) > self.sync_tolerance_s:
                return self.last_overlay()

        with self._lock:
            calib = self.calib
            matcher = self.matcher
            track_only = self.track_only or matcher is None or calib is None
            disp_min, disp_num = self.disp_min, self.disp_num

        rect_l, rect_r = prepare_pair(
            frame_l, frame_r, calib, self._prep_pool, clahe=self.clahe
        )
        rect_l_bgr = cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR)

        with self._lock:
            self._last_rect_l = rect_l
            self._last_rect_r = rect_r

        tracking_ok = False
        roi = None
        if self.tracker.initialized:
            tracking_ok, roi = self.tracker.update(rect_l_bgr)
        else:
            roi = None
            tracking_ok = False

        dist_s = None
        disp_val = None
        sgbm_busy = False

        if not track_only and self.tracker.initialized and tracking_ok and matcher is not None:
            with self._lock:
                fut = self._sgbm_future
            if fut is not None and fut.done():
                try:
                    self._disp_float = fut.result()
                except Exception as exc:
                    self._log(f"SGBM ошибка: {exc}")
                with self._lock:
                    self._sgbm_future = None

            need_sgbm = self.frame_idx % self.sgbm_interval == 0
            with self._lock:
                fut = self._sgbm_future
            if need_sgbm and (fut is None or fut.done()):
                if fut is not None and fut.done():
                    try:
                        self._disp_float = fut.result()
                    except Exception:
                        pass
                    with self._lock:
                        self._sgbm_future = None
                with self._lock:
                    matcher_now = self.matcher
                if matcher_now is not None:
                    self._sgbm_future = self._sgbm_pool.submit(
                        compute_disparity,
                        rect_l.copy(),
                        rect_r.copy(),
                        matcher_now,
                        wls=self.wls,
                        wls_lambda=8000.0,
                        wls_sigma=1.5,
                    )

            with self._lock:
                sgbm_busy = (
                    self._sgbm_future is not None and not self._sgbm_future.done()
                )
                disp_float = self._disp_float

            if roi is not None and disp_float is not None:
                dist, disp_val = measure_roi_distance(
                    disp_float,
                    roi,
                    Q=self.Q,
                    inset_fraction=self.roi_inset,
                    surface=self.surface,
                    max_disparity=self.max_disp_cap,
                    min_disparity=self.min_disp_floor,
                    max_distance_mm=self.max_distance_mm,
                )
                dist_s, disp_s = self.dist_smoother.update(dist, disp_val)
                if disp_s is not None:
                    disp_val = disp_s
                if (
                    self.auto_disparity
                    and calib is not None
                    and (self._sgbm_future is None or self._sgbm_future.done())
                ):
                    new_min, new_num, adapt_log = adapt_disparity_range(
                        calib=calib,
                        image_width=int(rect_l.shape[1]),
                        z_near_m=self.z_near_m,
                        z_far_m=self.z_far_m,
                        cur_min=disp_min,
                        cur_num=disp_num,
                        distance_mm=dist_s if dist_s is not None else dist,
                        disparity_px=disp_val,
                        long_range=self.long_range,
                    )
                    if adapt_log is not None:
                        with self._lock:
                            self.disp_min, self.disp_num = new_min, new_num
                            self.matcher = make_stereo_matcher(
                                self.method,
                                self.disp_min,
                                self.disp_num,
                                self.block_size,
                                uniqueness_ratio=self.uniqueness,
                                speckle_window_size=40 if self.long_range else 50,
                            )
                            disp_min, disp_num = new_min, new_num
                        self._log(adapt_log)
            else:
                dist_s, disp_s = self.dist_smoother.update(None, None)
                if disp_s is not None:
                    disp_val = disp_s
        elif self.tracker.initialized and not tracking_ok:
            dist_s, disp_s = self.dist_smoother.update(None, None)
            if disp_s is not None:
                disp_val = disp_s
            else:
                disp_val = None
        else:
            dist_s = None
            disp_val = None

        now = time.perf_counter()
        dt = now - self._t_prev
        self._t_prev = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else 1.0 / dt

        overlay = draw_overlay(
            rect_l_bgr,
            roi,
            dist_s,
            disp_val,
            tracking_ok if self.tracker.initialized else False,
            self.frame_idx,
            self.fps,
            sgbm_busy=sgbm_busy,
            disp_range=(disp_min, disp_num) if not track_only else None,
        )
        self.frame_idx += 1

        with self._lock:
            self._tracking_ok = bool(tracking_ok and self.tracker.initialized)
            self._dist_s = dist_s
            self._disp_val = disp_val
            self._last_overlay = overlay
        return overlay
