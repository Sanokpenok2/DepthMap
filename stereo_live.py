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
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np

from depth_map import load_calibration, measure_roi_distance
from object_tracker import ObjectTracker
from stereo_auto import (
    RangeBand,
    clamp_sgbm_range,
    disparity_from_depth,
    estimate_disparity_range_bounds,
    extract_calib_geometry,
    make_range_bands,
    parse_band_edges,
    DEFAULT_RANGE_BAND_EDGES,
)
from video_track_depth import (
    DistanceSmoother,
    SpeedEstimator,
    adapt_disparity_range,
    build_band_matcher,
    compute_disparity,
    draw_overlay,
    make_stereo_matcher,
    measure_triple_band_point,
    prepare_pair,
    rect_to_bgr,
    stereo_gray_pair,
)


@dataclass
class LiveStatus:
    enabled: bool = False
    has_calib: bool = False
    tracking_ok: bool = False
    roi: tuple[int, int, int, int] | None = None
    distance_mm: float | None = None
    disparity_px: float | None = None
    speed_mps: float | None = None
    velocity_mps: tuple[float, float, float] | None = None
    disp_min: int = 0
    disp_num: int = 128
    sgbm_busy: bool = False
    message: str = ""
    range_mode: str = "auto"
    band_index: int = 0
    band_label: str = ""


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
        clahe: bool = False,
        force_gray: bool = False,
        roi_inset: float = 0.32,
        surface: str = "far",
        depth_scale: float = 1.0,
        show_velocity_arrow: bool = True,
        long_range: bool | None = None,
        range_mode: str = "auto",
        band_edges: tuple[float, ...] | str | None = None,
        band_index: int = 0,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.z_near_m = float(z_near_m)
        self.z_far_m = float(z_far_m)
        self._z_near_auto = float(z_near_m)
        self._z_far_auto = float(z_far_m)
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
        self.force_gray = bool(force_gray)
        self.roi_inset = float(roi_inset)
        self.surface = str(surface)
        self.depth_scale = float(depth_scale) if float(depth_scale) > 0 else 1.0
        self.show_velocity_arrow = bool(show_velocity_arrow)
        # None / не задано → long-range выключен (только явный флаг).
        self._long_range_pref = long_range
        self.long_range = self._resolve_long_range(long_range)
        self.max_disp_cap: float | None = None
        self.min_disp_floor = 0.85
        self.max_distance_mm: float | None = self.z_far_m * 1000.0 * 1.15
        self.uniqueness = 12 if self.long_range else 5
        self._on_log = on_log

        self.range_mode = str(range_mode)
        if self.range_mode not in ("auto", "bands", "triple"):
            self.range_mode = "auto"
        try:
            edges = parse_band_edges(
                band_edges if band_edges is not None else DEFAULT_RANGE_BAND_EDGES
            )
        except ValueError:
            edges = DEFAULT_RANGE_BAND_EDGES
        self.range_bands: list[RangeBand] = make_range_bands(edges)
        self.band_index = int(
            np.clip(band_index, 0, max(0, len(self.range_bands) - 1))
        )
        self._band_states: list[dict] = []

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
        self._speed_mps: float | None = None
        self._velocity_mps: tuple[float, float, float] | None = None
        self._image_vel_px_s: tuple[float, float] | None = None
        self.speed_est: SpeedEstimator | None = None
        self._tracking_ok = False
        self._status_msg = ""

    @staticmethod
    def _resolve_long_range_flag(z_far_m: float, long_range: bool | None) -> bool:
        """long-range только по явному флагу (z_far сам по себе не включает)."""
        _ = z_far_m
        if long_range is None:
            return False
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
        self._z_near_auto = float(z_near_m)
        self._z_far_auto = float(z_far_m)
        self._long_range_pref = long_range
        self.long_range = self._resolve_long_range(long_range)
        self.uniqueness = 12 if self.long_range else 5
        self.max_distance_mm = self.z_far_m * 1000.0 * 1.15
        self.dist_smoother.max_distance_mm = self.max_distance_mm
        if self.range_mode == "auto":
            self._log(
                f"Сцена: z={self.z_near_m:.0f}–{self.z_far_m:.0f} м "
                f"(long_range={self.long_range})."
            )
            if self.calib is not None:
                self._rebuild_matcher_from_calib("reconfigure")

    def set_band_edges(self, edges: str | tuple[float, ...] | list[float]) -> None:
        """Задать границы полос (м), напр. '100,500,1000,3000'."""
        parsed = parse_band_edges(edges)
        self.range_bands = make_range_bands(parsed)
        self.band_index = int(
            np.clip(self.band_index, 0, max(0, len(self.range_bands) - 1))
        )
        self._band_states = []
        self._log(
            "Полосы: "
            + " | ".join(
                f"{b.z_near_m:.0f}–{b.z_far_m:.0f}" for b in self.range_bands
            )
        )
        if self.calib is not None and self.range_mode in ("bands", "triple"):
            self._ensure_band_states()
            if self.range_mode == "bands":
                self.apply_band(self.band_index)
            else:
                self.set_range_mode("triple")

    def range_mode_label(self) -> str:
        if self.range_mode == "triple":
            return (
                f"TRIPLE "
                f"{self.range_bands[0].z_near_m:.0f}…"
                f"{self.range_bands[-1].z_far_m:.0f}m"
            )
        if self.range_mode == "bands" and self.range_bands:
            b = self.range_bands[self.band_index]
            return (
                f"BAND {self.band_index + 1}/{len(self.range_bands)} "
                f"{b.z_near_m:.0f}–{b.z_far_m:.0f}m"
            )
        return "AUTO"

    def _ensure_band_states(self) -> None:
        if self.calib is None:
            return
        if self._band_states and len(self._band_states) == len(self.range_bands):
            return
        width = 640
        if "image_size" in self.calib:
            width = int(self.calib["image_size"][0])
        if self._last_rect_l is not None:
            width = int(self._last_rect_l.shape[1])
        self._band_states = [
            build_band_matcher(
                self.calib,
                band,
                image_width=width,
                method=self.method,
                block_size=self.block_size,
            )
            for band in self.range_bands
        ]

    def apply_band(self, index: int) -> None:
        """Включить одну полосу (выключает auto/triple)."""
        if not self.range_bands:
            return
        self._ensure_band_states()
        if not self._band_states:
            self._log("Нет калибровки для полос.")
            return
        self.band_index = int(index) % len(self._band_states)
        self.range_mode = "bands"
        self.auto_disparity = False
        st = self._band_states[self.band_index]
        band = st["band"]
        self.z_near_m = band.z_near_m
        self.z_far_m = band.z_far_m
        self.long_range = band.long_range
        self.uniqueness = st["uniqueness"]
        with self._lock:
            self.disp_min = st["disp_min"]
            self.disp_num = st["disp_num"]
            self.matcher = st["matcher"]
            self._disp_float = None
            fut = self._sgbm_future
            self._sgbm_future = None
        if fut is not None:
            try:
                fut.result(timeout=2)
            except Exception:
                pass
        self.max_disp_cap = st["max_disp_cap"]
        self.min_disp_floor = st["min_disp_floor"]
        self.max_distance_mm = st["max_distance_mm"]
        self.dist_smoother.max_distance_mm = self.max_distance_mm
        self._log(f"Полоса [{self.band_index}] {band.label}")

    def cycle_band(self, delta: int = 1) -> None:
        if self.range_mode != "bands":
            self.apply_band(self.band_index)
            return
        self.apply_band(self.band_index + int(delta))

    def set_range_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in ("auto", "bands", "triple"):
            raise ValueError("mode: auto | bands | triple")
        if mode == "auto":
            self.range_mode = "auto"
            self.auto_disparity = True
            self.z_near_m = self._z_near_auto
            self.z_far_m = self._z_far_auto
            self.long_range = self._resolve_long_range(self._long_range_pref)
            self.uniqueness = 12 if self.long_range else 5
            self._log("Режим AUTO-диспаритета.")
            if self.calib is not None:
                self._rebuild_matcher_from_calib("auto")
            return
        if mode == "bands":
            self.apply_band(self.band_index)
            return
        # triple
        self._ensure_band_states()
        if not self._band_states:
            self._log("Нет калибровки для тройного режима.")
            return
        self.range_mode = "triple"
        self.auto_disparity = False
        st_far = self._band_states[-1]
        self.z_near_m = self.range_bands[0].z_near_m
        self.z_far_m = self.range_bands[-1].z_far_m
        self.long_range = any(b.long_range for b in self.range_bands)
        self.uniqueness = st_far["uniqueness"]
        with self._lock:
            self.disp_min = st_far["disp_min"]
            self.disp_num = st_far["disp_num"]
            self.matcher = st_far["matcher"]
            self._disp_float = None
            fut = self._sgbm_future
            self._sgbm_future = None
        if fut is not None:
            try:
                fut.result(timeout=2)
            except Exception:
                pass
        self.max_disp_cap = max(st["max_disp_cap"] for st in self._band_states)
        self.min_disp_floor = min(st["min_disp_floor"] for st in self._band_states)
        self.max_distance_mm = max(st["max_distance_mm"] for st in self._band_states)
        self.dist_smoother.max_distance_mm = self.max_distance_mm
        self._log(
            "Тройной режим: 3 полосы → выброс → среднее d "
            f"({self.range_bands[0].z_near_m:.0f}…"
            f"{self.range_bands[-1].z_far_m:.0f} м)."
        )

    def toggle_triple(self) -> None:
        if self.range_mode == "triple":
            self.set_range_mode("bands")
        else:
            self.set_range_mode("triple")

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
        with self._lock:
            self.calib = calib
            self.Q = calib["Q"]
            self.track_only = False
            self.speed_est = self._make_speed_estimator(calib)
        self._rebuild_matcher_from_calib(path, calib=calib)
        self._log(f"Калибровка загружена: {path}")

    @staticmethod
    def _make_speed_estimator(calib: dict) -> SpeedEstimator:
        focal_px, _baseline = extract_calib_geometry(calib)
        if "P1" in calib:
            p1 = np.asarray(calib["P1"], dtype=np.float64)
            cam_cx = float(p1[0, 2])
            cam_cy = float(p1[1, 2])
            if abs(float(p1[0, 0])) > 1.0:
                focal_px = float(p1[0, 0])
        else:
            mtx = np.asarray(calib["mtx_l"], dtype=np.float64)
            cam_cx = float(mtx[0, 2])
            cam_cy = float(mtx[1, 2])
        return SpeedEstimator(
            focal_px=focal_px,
            cx=cam_cx,
            cy=cam_cy,
            window_s=4.0,
            min_dt_s=1.5,
            ema_alpha=0.04,
            max_z_ratio=1.18,
            max_z_jump_m=10.0,
            min_dz_m=0.8,
            min_speed_mps=0.5,
        )

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
        if self.range_mode in ("bands", "triple"):
            self._band_states = [
                build_band_matcher(
                    calib,
                    band,
                    image_width=width,
                    method=self.method,
                    block_size=self.block_size,
                )
                for band in self.range_bands
            ]
            for st in self._band_states:
                self._log(st["log"])
            if self.range_mode == "bands":
                self.apply_band(self.band_index)
            else:
                # apply triple caps without recursive set_range_mode rebuild
                st_far = self._band_states[-1]
                self.z_near_m = self.range_bands[0].z_near_m
                self.z_far_m = self.range_bands[-1].z_far_m
                self.long_range = any(b.long_range for b in self.range_bands)
                self.uniqueness = st_far["uniqueness"]
                self.max_disp_cap = max(st["max_disp_cap"] for st in self._band_states)
                self.min_disp_floor = min(
                    st["min_disp_floor"] for st in self._band_states
                )
                self.max_distance_mm = max(
                    st["max_distance_mm"] for st in self._band_states
                )
                self.dist_smoother.max_distance_mm = self.max_distance_mm
                with self._lock:
                    self.disp_min = st_far["disp_min"]
                    self.disp_num = st_far["disp_num"]
                    self.matcher = st_far["matcher"]
                self._log(
                    "Тройной режим готов "
                    f"({self.range_bands[0].z_near_m:.0f}…"
                    f"{self.range_bands[-1].z_far_m:.0f} м)."
                )
            return

        self.long_range = self._resolve_long_range(self._long_range_pref)
        self.uniqueness = 12 if self.long_range else 5
        disp_min, disp_num = 0, 48
        use_auto = self.auto_disparity and self.range_mode == "auto"
        if use_auto:
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
            self.speed_est = None
            self._speed_mps = None
            self._velocity_mps = None
            self._image_vel_px_s = None
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
            self._speed_mps = None
            self._velocity_mps = None
            self._image_vel_px_s = None
            if self.speed_est is not None:
                self.speed_est.reset()
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
                speed_mps=self._speed_mps,
                velocity_mps=self._velocity_mps,
                disp_min=self.disp_min,
                disp_num=self.disp_num,
                sgbm_busy=(
                    self._sgbm_future is not None and not self._sgbm_future.done()
                ),
                message=self._status_msg,
                range_mode=self.range_mode,
                band_index=self.band_index,
                band_label=self.range_mode_label(),
            )

    def last_overlay(self) -> np.ndarray | None:
        with self._lock:
            if self._last_overlay is None:
                return None
            return self._last_overlay.copy()

    def set_force_gray(self, enabled: bool) -> None:
        """Вкл/выкл принудительный перевод кадров в gray (для UI)."""
        with self._lock:
            self.force_gray = bool(enabled)

    def set_clahe(self, enabled: bool) -> None:
        with self._lock:
            self.clahe = bool(enabled)

    def set_show_velocity_arrow(self, enabled: bool) -> None:
        with self._lock:
            self.show_velocity_arrow = bool(enabled)

    def _snapshot_rect_left(self) -> np.ndarray | None:
        with self._lock:
            if self._last_rect_l is None:
                return None
            return rect_to_bgr(self._last_rect_l)

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
            if self.speed_est is not None:
                self.speed_est.reset()
            self._tracking_ok = True
            self._dist_s = None
            self._disp_val = None
            self._speed_mps = None
            self._velocity_mps = None
            self._image_vel_px_s = None
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
            if self.speed_est is not None:
                self.speed_est.reset()
            self._tracking_ok = True
            self._dist_s = None
            self._disp_val = None
            self._speed_mps = None
            self._velocity_mps = None
            self._image_vel_px_s = None
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
            if self.speed_est is not None:
                self.speed_est.reset()
            self._tracking_ok = True
            self._dist_s = None
            self._disp_val = None
            self._speed_mps = None
            self._velocity_mps = None
            self._image_vel_px_s = None
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

        with self._lock:
            use_clahe = self.clahe
            use_gray = self.force_gray
        rect_l, rect_r = prepare_pair(
            frame_l,
            frame_r,
            calib,
            self._prep_pool,
            clahe=use_clahe,
            force_gray=use_gray,
        )
        rect_l_bgr = rect_to_bgr(rect_l)
        gray_l, gray_r = stereo_gray_pair(rect_l, rect_r)

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
            need_sgbm = self.frame_idx % self.sgbm_interval == 0

            if self.range_mode == "triple":
                self._ensure_band_states()
                if need_sgbm and roi is not None and self._band_states:
                    measure_args = SimpleNamespace(
                        roi_inset=self.roi_inset,
                        surface=self.surface,
                        depth_scale=float(self.depth_scale),
                    )
                    dist, disp_val, _per, disp_map = measure_triple_band_point(
                        gray_l=gray_l,
                        gray_r=gray_r,
                        roi=roi,
                        band_states=self._band_states,
                        calib=calib,
                        Q=self.Q,
                        args=measure_args,
                        left_gray=gray_l,
                        right_gray=gray_r,
                        epipolar_ncc=True,
                        wls=self.wls,
                    )
                    if disp_map is not None:
                        self._disp_float = disp_map
                    dist_s, disp_s = self.dist_smoother.update(dist, disp_val)
                    if disp_s is not None:
                        disp_val = disp_s
                else:
                    dist_s, disp_s = self.dist_smoother.update(None, None)
                    if disp_s is not None:
                        disp_val = disp_s
            else:
                with self._lock:
                    fut = self._sgbm_future
                if fut is not None and fut.done():
                    try:
                        self._disp_float = fut.result()
                    except Exception as exc:
                        self._log(f"SGBM ошибка: {exc}")
                    with self._lock:
                        self._sgbm_future = None

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
                            gray_l.copy(),
                            gray_r.copy(),
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
                        left_gray=gray_l,
                        right_gray=gray_r,
                        epipolar_ncc=True,
                        depth_scale=float(self.depth_scale),
                    )
                    dist_s, disp_s = self.dist_smoother.update(dist, disp_val)
                    if disp_s is not None:
                        disp_val = disp_s
                    if (
                        self.auto_disparity
                        and self.range_mode == "auto"
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

        speed_mps = None
        velocity_mps = None
        image_vel_px_s = None
        if (
            self.speed_est is not None
            and self.tracker.initialized
            and tracking_ok
            and dist_s is not None
            and roi is not None
        ):
            spd = self.speed_est.update(
                now,
                dist_s,
                roi,
                tracking_ok=True,
            )
            speed_mps = spd.speed_mps
            velocity_mps = spd.velocity_mps
            image_vel_px_s = spd.image_vel_px_s
        elif self.speed_est is not None and not self.tracker.initialized:
            self.speed_est.reset()

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
            speed_mps=speed_mps,
            velocity_mps=velocity_mps,
            image_vel_px_s=image_vel_px_s,
            show_velocity_arrow=self.show_velocity_arrow,
            mode_label=self.range_mode_label() if not track_only else None,
        )
        self.frame_idx += 1

        with self._lock:
            self._tracking_ok = bool(tracking_ok and self.tracker.initialized)
            self._dist_s = dist_s
            self._disp_val = disp_val
            self._speed_mps = speed_mps
            self._velocity_mps = velocity_mps
            self._image_vel_px_s = image_vel_px_s
            self._last_overlay = overlay
        return overlay
