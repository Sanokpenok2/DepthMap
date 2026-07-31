"""
Трекинг объекта по стерео-видео с измерением расстояния.

Вход — либо одно SBS-видео (кадр пополам: L|R), либо два отдельных файла
(--left-video / --right-video). Пользователь выделяет объект на левом кадре;
дальше объект сопровождается трекером, расстояние — по медиане диспаритета в ROI.

Ускорение на CPU:
  - cv2.setNumThreads — внутренний параллелизм OpenCV (SGBM, remap);
  - параллельная подготовка левого/правого кадра;
  - асинхронный SGBM в фоне, чтобы трекинг не ждал каждый тяжёлый кадр.

Примеры:
    python video_track_depth.py --video stereo_sbs.mp4 --calib stereo_calib.npz
    python video_track_depth.py --left-video left.mp4 --right-video right.mp4 ^
        --calib stereo_calib.npz

Управление:
    пробел  — пауза/продолжить
    r / c   — выбрать объект (рамка / клик)
    x       — отменить трекинг
    d       — вкл/выкл отладку диспаритета (L↔R соответствия)
    q / Esc — выход
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from depth_map import (
    apply_wls,
    build_bm,
    build_sgbm,
    calibration_quality_warnings,
    display_scale,
    draw_disparity_debug,
    fit_for_display,
    load_calibration,
    measure_roi_distance,
    split_sbs,
    DisparityDebugInfo,
)
from object_tracker import ObjectTracker
from calib_quality import format_quality_report
from stereo_auto import (
    clamp_sgbm_range,
    disparity_from_depth,
    estimate_disparity_range_bounds,
    extract_calib_geometry,
    round_num_disparities,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Трекинг объекта по стерео-видео (SBS или пара L/R) со стерео-расстоянием.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--video",
        default=None,
        help="SBS-видео: левая половина кадра — левая камера, правая — правая.",
    )
    p.add_argument(
        "--left-video",
        default=None,
        help="Видео левой камеры (вместе с --right-video вместо --video).",
    )
    p.add_argument(
        "--right-video",
        default=None,
        help="Видео правой камеры (вместе с --left-video вместо --video).",
    )
    p.add_argument(
        "--swap-lr",
        action="store_true",
        help="Поменять L/R местами (половины SBS или потоки left/right).",
    )
    p.add_argument(
        "--calib",
        default=None,
        help="Файл стереокалибровки (.npz). Обязателен, кроме режима --track-only.",
    )
    p.add_argument(
        "--method",
        choices=["sgbm", "bm"],
        default="sgbm",
        help="Алгоритм сопоставления.",
    )
    p.add_argument(
        "--num-disparities",
        type=int,
        default=128,
        help="Диапазон диспаритетов (кратен 16). При --auto-disparity — стартовое значение.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=7,
        help="Размер блока SGBM/BM (нечётный). 7–9 лучше для мелких дальних объектов.",
    )
    p.add_argument(
        "--min-disparity",
        type=int,
        default=0,
        help="Мин. диспаритет. При --auto-disparity подбирается автоматически.",
    )
    p.add_argument(
        "--auto-disparity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Подбирать и расширять диапазон диспаритета по --z-near/--z-far и "
            "текущей дистанции объекта (нужен --calib). Иначе диапазон фиксирован "
            "и при приближении измерение портится."
        ),
    )
    p.add_argument(
        "--z-near",
        type=float,
        default=25.0,
        help="Ближняя граница сцены, м. Для 1000+ м: 150–300 (потолок поиска SGBM).",
    )
    p.add_argument(
        "--z-far",
        type=float,
        default=600.0,
        help="Дальняя граница сцены, м. Для шоссе/1000+ м: 1500–3000.",
    )
    p.add_argument(
        "--long-range",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Режим дальних дистанций: жёсткий потолок num_disparities ~d(z-near). "
            "Автоматически включается при --z-far >= 800."
        ),
    )
    p.add_argument("--wls", action="store_true", help="WLS-фильтр (медленнее).")
    p.add_argument("--wls-lambda", type=float, default=8000.0)
    p.add_argument("--wls-sigma", type=float, default=1.5)
    p.add_argument(
        "--clahe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CLAHE на ректифицированном gray (важно для низкоконтрастного ТВ/ИК).",
    )
    p.add_argument(
        "--tracker",
        choices=["csrt", "kcf", "mosse", "ncc"],
        default="ncc",
        help=(
            "Базовый трекер под гибрид + NCC-refine: "
            "kcf/csrt/mosse (OpenCV) или ncc (template/NCC как в DepthMapKornia)."
        ),
    )
    p.add_argument(
        "--kornia-tracker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Использовать базовый трекинг как в DepthMapKornia "
            "(NCC template вместо KCF). Эквивалент --tracker ncc."
        ),
    )
    p.add_argument(
        "--roi-smooth",
        type=float,
        default=0.0,
        help="Сглаживание рамки трекинга [0..1): 0 = без лага, ближе к 1 = плавнее (отстаёт от объекта).",
    )
    p.add_argument(
        "--lock-size",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Фиксировать размер рамки. Выкл. (по умолчанию) — scale по NCC при приближении/удалении.",
    )
    p.add_argument(
        "--keep-aspect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Масштабировать рамку равномерно, сохраняя исходные пропорции объекта.",
    )
    p.add_argument(
        "--max-scale-step",
        type=float,
        default=0.08,
        help="Макс. относительное изменение масштаба рамки за кадр (0 = без лимита).",
    )
    p.add_argument(
        "--max-size-ratio",
        type=float,
        default=2.5,
        help="Макс. изменение площади рамки за кадр; иначе LOST.",
    )
    p.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Отклонять дрейф рамки на другой объект/фон (сходство с эталоном + прыжок).",
    )
    p.add_argument(
        "--verify-threshold",
        type=float,
        default=0.30,
        help="Мин. NCC (CLAHE) [0..1]; ниже soft-полосы — копим LOST.",
    )
    p.add_argument(
        "--max-jump",
        type=float,
        default=4.0,
        help="Макс. прыжок центра (доли ширины ROI по X; для быстрых L→R ≥3).",
    )
    p.add_argument(
        "--verify-rel",
        type=float,
        default=0.0,
        help="Отклонять кадр, если score < EMA*verify-rel (0 = выкл.).",
    )
    p.add_argument(
        "--min-iou",
        type=float,
        default=0.0,
        help="Мин. IoU с предыдущей рамкой (0 = выкл.; для быстрого движения лучше 0).",
    )
    p.add_argument(
        "--lost-patience",
        type=int,
        default=14,
        help="Сколько подряд плохих кадров нужно, чтобы объявить LOST.",
    )
    p.add_argument(
        "--reacquire",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Повторно захватывать объект после потери по эталону.",
    )
    p.add_argument(
        "--reacquire-threshold",
        type=float,
        default=0.55,
        help="Порог NCC для повторного захвата [0..1] (выше = меньше ложных прыжков).",
    )
    p.add_argument(
        "--reacquire-radius",
        type=float,
        default=3.5,
        help="Окно поиска при перезахвате (доли max(w,h) ROI вокруг последней позиции).",
    )
    p.add_argument(
        "--reacquire-global",
        action="store_true",
        help="Искать объект по всему кадру при перезахвате (может хватать похожий фон).",
    )
    p.add_argument(
        "--reacquire-interval",
        type=int,
        default=2,
        help="Искать объект при потере каждые N кадров (1 = каждый кадр, тяжелее).",
    )
    p.add_argument(
        "--reacquire-scale-min",
        type=float,
        default=0.50,
        help="Мин. масштаб эталона при перезахвате (удаление).",
    )
    p.add_argument(
        "--reacquire-scale-max",
        type=float,
        default=2.0,
        help="Макс. масштаб эталона при перезахвате (приближение).",
    )
    p.add_argument(
        "--click-tolerance",
        type=int,
        default=16,
        help="Допуск яркости при авто-выделении кликом (C); после клика +/- подстраивает.",
    )
    p.add_argument(
        "--no-grabcut",
        action="store_true",
        help="Не уточнять границы объекта GrabCut'ом при выборе кликом.",
    )
    p.add_argument(
        "--sgbm-interval",
        type=int,
        default=2,
        help="Считать SGBM каждые N кадров (1 = каждый кадр; больше = выше FPS трекинга).",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=21,
        help="Окно медианы по расстоянию/диспаритету (кадры). 0 = без сглаживания.",
    )
    p.add_argument(
        "--smooth-max-ratio",
        type=float,
        default=1.8,
        help=(
            "Отбрасывать измерение, если Z или disp скачет сильнее чем в N раз "
            "относительно текущего сглаженного."
        ),
    )
    p.add_argument(
        "--smooth-ema",
        type=float,
        default=0.25,
        help="Доп. EMA после медианы [0..1]: меньше = плавнее (больше инерция).",
    )
    p.add_argument(
        "--smooth-disp-jump",
        type=float,
        default=1.2,
        help="Макс. скачок диспаритета (px) за кадр без отбраковки; 0 = не учитывать.",
    )
    p.add_argument(
        "--roi-inset",
        type=float,
        default=0.32,
        help="Доля обрезки краёв ROI при измерении дистанции (0 = весь бокс).",
    )
    p.add_argument(
        "--surface",
        choices=["far", "near", "median"],
        default="median",
        help=(
            "Поверхность в ROI: median=стабильнее; far=чуть дальше "
            "(осторожно: шум малого d завышает Z); near=ближе."
        ),
    )
    p.add_argument(
        "--speed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Оценивать скорость выбранного объекта по стерео-траектории (нужен --calib).",
    )
    p.add_argument(
        "--speed-window",
        type=float,
        default=4.0,
        help="Окно истории для оценки скорости, секунды (больше = стабильнее).",
    )
    p.add_argument(
        "--speed-min-dt",
        type=float,
        default=1.5,
        help="Минимальный интервал истории (с), прежде чем показывать скорость.",
    )
    p.add_argument(
        "--speed-ema",
        type=float,
        default=0.04,
        help="EMA сглаживание скорости [0..1]: меньше = плавнее (больше инерция).",
    )
    p.add_argument(
        "--speed-max-z-ratio",
        type=float,
        default=1.18,
        help="Отбросить дистанцию для скорости, если Z скачет сильнее чем в N раз.",
    )
    p.add_argument(
        "--speed-max-z-jump",
        type=float,
        default=10.0,
        help="Отбросить дистанцию для скорости при скачке |ΔZ| больше N метров.",
    )
    p.add_argument(
        "--speed-min-dz",
        type=float,
        default=0.8,
        help="Игнорировать изменения дистанции меньше N метров (шум стерео).",
    )
    p.add_argument(
        "--speed-min",
        type=float,
        default=0.5,
        help="Скорости ниже N м/с считать нулём (не тянуть оценку шумом).",
    )
    p.add_argument(
        "--debug-disparity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Окно отладки: пиксели ROI на L и соответствия x-d на R "
            "(клавиша D переключает)."
        ),
    )
    p.add_argument(
        "--debug-disp-samples",
        type=int,
        default=48,
        help="Сколько линий L→R рисовать в --debug-disparity.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Число потоков OpenCV (0 = все ядра, 1 = без внутреннего параллелизма).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Потоки для параллельной подготовки L/R кадров.",
    )
    p.add_argument(
        "--async-sgbm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Считать диспаритет асинхронно в фоне (трекинг не блокируется).",
    )
    p.add_argument(
        "--track-only",
        action="store_true",
        help="Только захват и трекинг объекта, без SGBM и измерения расстояния.",
    )
    p.add_argument(
        "--max-display",
        type=int,
        default=1200,
        help="Макс. сторона окна предпросмотра.",
    )
    p.add_argument(
        "--colormap",
        default="JET",
        help="Палитра карты диспаритета.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Опционально сохранить результирующее видео с оверлеем.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Ограничить число кадров (0 = до конца).",
    )
    p.add_argument(
        "--max-fps",
        type=float,
        default=0.0,
        help="Ограничить скорость обработки (кадр/с). 0 = без ограничения.",
    )
    return p.parse_args()


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Ошибка: не удалось открыть видео '{path}'.")
    return cap


class StereoFrameSource:
    """Читает стереопары из SBS-файла или из двух отдельных видео."""

    def __init__(
        self,
        *,
        sbs_path: str | None = None,
        left_path: str | None = None,
        right_path: str | None = None,
        swap_lr: bool = False,
    ) -> None:
        self.swap_lr = bool(swap_lr)
        self._cap_sbs: cv2.VideoCapture | None = None
        self._cap_l: cv2.VideoCapture | None = None
        self._cap_r: cv2.VideoCapture | None = None

        if sbs_path:
            self.mode = "sbs"
            self._cap_sbs = open_video(sbs_path)
            self._primary = self._cap_sbs
        elif left_path and right_path:
            self.mode = "dual"
            self._cap_l = open_video(left_path)
            self._cap_r = open_video(right_path)
            self._primary = self._cap_l
        else:
            raise ValueError("Нужен --video либо пара --left-video/--right-video.")

    @property
    def fps(self) -> float:
        return float(max(self._primary.get(cv2.CAP_PROP_FPS), 1.0))

    def read(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        if self.mode == "sbs":
            assert self._cap_sbs is not None
            ok, frame = self._cap_sbs.read()
            if not ok or frame is None:
                return False, None, None
            left, right = split_sbs(frame, swap_lr=False)
        else:
            assert self._cap_l is not None and self._cap_r is not None
            ok_l, left = self._cap_l.read()
            ok_r, right = self._cap_r.read()
            if not ok_l or not ok_r or left is None or right is None:
                return False, None, None
            if left.shape[:2] != right.shape[:2]:
                # Подгоняем правый кадр под размер левого (если чуть разъехались).
                right = cv2.resize(
                    right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA
                )
        if self.swap_lr:
            left, right = right, left
        return True, left, right

    def release(self) -> None:
        for cap in (self._cap_sbs, self._cap_l, self._cap_r):
            if cap is not None:
                cap.release()


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def enhance_gray(gray: np.ndarray, clahe: bool = True) -> np.ndarray:
    """Поднять локальный контраст на низкоконтрастном gray (ТВ/ИК)."""
    if not clahe or gray.ndim != 2:
        return gray
    # clipLimit умеренный: сильнее — шумит диспаритет.
    clahe_f = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe_f.apply(gray)


def resize_to_calib(img: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    if size is None:
        return img
    tw, th = size
    h, w = img.shape[:2]
    if (w, h) == (tw, th):
        return img
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)


def prepare_side(
    frame: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    calib_size: tuple[int, int] | None,
    *,
    clahe: bool = True,
) -> np.ndarray:
    """Gray → resize → remap (+CLAHE) для одной камеры."""
    gray = to_gray(frame)
    gray = resize_to_calib(gray, calib_size)
    gray = cv2.remap(gray, map1, map2, cv2.INTER_LINEAR)
    return enhance_gray(gray, clahe=clahe)


def prepare_pair(
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    calib: dict | None,
    pool: ThreadPoolExecutor | None,
    *,
    clahe: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    # Без калибровки ректификация невозможна: только gray (для трекинга этого хватает).
    if calib is None:
        return (
            enhance_gray(to_gray(frame_l), clahe=clahe),
            enhance_gray(to_gray(frame_r), clahe=clahe),
        )

    calib_size = None
    if "image_size" in calib:
        calib_size = (int(calib["image_size"][0]), int(calib["image_size"][1]))

    if pool is None:
        rect_l = prepare_side(
            frame_l, calib["map1_l"], calib["map2_l"], calib_size, clahe=clahe
        )
        rect_r = prepare_side(
            frame_r, calib["map1_r"], calib["map2_r"], calib_size, clahe=clahe
        )
        return rect_l, rect_r

    fut_l = pool.submit(
        prepare_side, frame_l, calib["map1_l"], calib["map2_l"], calib_size, clahe=clahe
    )
    fut_r = pool.submit(
        prepare_side, frame_r, calib["map1_r"], calib["map2_r"], calib_size, clahe=clahe
    )
    return fut_l.result(), fut_r.result()


def compute_disparity(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    matcher,
    *,
    wls: bool,
    wls_lambda: float,
    wls_sigma: float,
) -> np.ndarray:
    disp = matcher.compute(left_gray, right_gray)
    if wls:
        disp = apply_wls(matcher, disp, left_gray, right_gray, wls_lambda, wls_sigma)
    return disp.astype(np.float32) / 16.0


def make_stereo_matcher(
    method: str,
    min_disparity: int,
    num_disparities: int,
    block_size: int,
    *,
    uniqueness_ratio: int = 5,
    speckle_window_size: int = 50,
):
    """Stereo matcher. Для long-range лучше uniqueness_ratio≈12–15."""
    if method == "sgbm":
        return build_sgbm(
            min_disparity,
            num_disparities,
            block_size,
            uniqueness_ratio=uniqueness_ratio,
            speckle_window_size=speckle_window_size,
        )
    return build_bm(num_disparities, block_size)


def adapt_disparity_range(
    *,
    calib: dict,
    image_width: int,
    z_near_m: float,
    z_far_m: float,
    cur_min: int,
    cur_num: int,
    distance_mm: float | None,
    disparity_px: float | None,
    long_range: bool | None = None,
) -> tuple[int, int, str | None]:
    """Расширяет диапазон, если объект приблизился или диспаритет упёрся в потолок.

    В long-range (z_far>=800) НЕ сужаем сцену по «ложному» ближнему Z —
    иначе SGBM начинает искать большие d и дистанция падает до десятков метров.
    """
    width = max(int(image_width), 32)
    if long_range is None:
        long_range = z_far_m >= 800.0

    upper = float(cur_min + cur_num)
    saturating = (
        disparity_px is not None
        and np.isfinite(disparity_px)
        and disparity_px > 0.90 * upper
    )

    z_m = None
    if distance_mm is not None and np.isfinite(distance_mm) and distance_mm > 0:
        z_m = float(distance_mm) / 1000.0

    # Дальняя сцена: игнорируем «приближение», пока d не упёрся в потолок поиска.
    if long_range and not saturating:
        return cur_min, cur_num, None

    if not saturating and z_m is None:
        return cur_min, cur_num, None

    # Ложное «близко»: Z заметно меньше z_near — не открываем ближнюю полосу.
    if long_range and z_m is not None and z_m < 0.85 * z_near_m and not saturating:
        return cur_min, cur_num, None

    focal, baseline = extract_calib_geometry(calib)
    if z_m is not None and not long_range:
        z_lo = max(min(z_near_m, z_m * 0.45), 0.5)
        z_hi = min(z_far_m, max(z_m * 1.8, z_m + 5.0))
    elif saturating and long_range:
        # Только чуть расширяем ближнюю границу (не ниже ~0.6*z_near).
        z_lo = max(z_near_m * 0.6, 20.0)
        z_hi = z_far_m
    else:
        z_lo, z_hi = z_near_m, z_far_m

    if saturating and not long_range:
        z_lo = max(0.5, min(z_lo, z_near_m))
        if z_m is not None:
            z_lo = max(0.5, min(z_lo, z_m * 0.35))

    if z_lo >= z_hi:
        z_lo, z_hi = z_near_m, z_far_m

    new_min, new_num, _ = estimate_disparity_range_bounds(
        calib, z_lo, z_hi, image_width=width, long_range=long_range
    )
    max_num = 96 if long_range else 512
    new_min, new_num = clamp_sgbm_range(new_min, new_num, width, max_num=max_num)

    if saturating and disparity_px is not None and not long_range:
        need_upper = float(disparity_px) * 1.35 + 16.0
        if new_min + new_num < need_upper:
            span = need_upper - float(new_min)
            new_num = round_num_disparities(span, min_val=64, max_val=512)
            new_min, new_num = clamp_sgbm_range(new_min, new_num, width, max_num=512)

    if new_min == cur_min and new_num == cur_num:
        return cur_min, cur_num, None
    if new_min + new_num < cur_min + cur_num and not saturating:
        d_need = disparity_from_depth(focal, baseline, max(z_lo, 0.5) * 1000.0)
        if d_need <= 0.9 * upper:
            return cur_min, cur_num, None

    log = (
        f"Диапазон диспаритета: min={new_min}, num={new_num} "
        f"(было {cur_min}+{cur_num}"
        + (f", Z~{z_m:.1f} м" if z_m is not None else "")
        + (", насыщение" if saturating else "")
        + (", long-range" if long_range else "")
        + ")."
    )
    return new_min, new_num, log


def _flush_waitkey_buffer(max_ms: float = 250.0) -> None:
    """Сбрасывает очередь клавиш HighGUI (после selectROI / click-UI)."""
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1000.0 < max_ms:
        if int(cv2.waitKey(1)) < 0:
            break


def _annotate_debug_flag(frame_bgr: np.ndarray) -> np.ndarray:
    out = frame_bgr
    cv2.putText(
        out,
        "DISP DEBUG",
        (out.shape[1] - 160, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "DISP DEBUG",
        (out.shape[1] - 160, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def draw_overlay(
    frame_bgr: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    distance_mm: float | None,
    disparity: float | None,
    tracking_ok: bool,
    frame_idx: int,
    fps: float,
    sgbm_busy: bool = False,
    disp_range: tuple[int, int] | None = None,
    speed_mps: float | None = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    if roi is not None:
        x, y, rw, rh = roi
        color = (0, 220, 0) if tracking_ok else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + rw, y + rh), color, 2)
        cx, cy = x + rw // 2, y + rh // 2
        cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)

    lines = [f"frame {frame_idx}", f"FPS {fps:.1f}"]
    if distance_mm is not None:
        if distance_mm >= 1000:
            lines.append(f"distance {distance_mm / 1000.0:.2f} m")
        else:
            lines.append(f"distance {distance_mm:.0f} mm")
    else:
        lines.append("distance n/a")
    if speed_mps is not None and np.isfinite(speed_mps):
        lines.append(f"speed {speed_mps * 3.6:.1f} km/h ({speed_mps:.1f} m/s)")
    elif roi is not None and tracking_ok:
        lines.append("speed n/a")
    if disparity is not None:
        lines.append(f"disp {disparity:.1f} px")
    if disp_range is not None:
        lines.append(f"range {disp_range[0]}+{disp_range[1]}")
    if sgbm_busy:
        lines.append("SGBM...")
    if roi is None:
        lines.append("press R (box) or C (click) to select")
    elif not tracking_ok:
        lines.append("TRACK LOST — searching / X=cancel, R=reselect")
    else:
        lines.append("tracking — X=cancel")

    y0 = 28
    for i, text in enumerate(lines):
        cv2.putText(
            out,
            text,
            (12, y0 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text,
            (12, y0 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def smoothed_value(history: deque[float], value: float | None, window: int) -> float | None:
    """Устаревшая простая медиана — предпочтите DistanceSmoother."""
    if value is None or not np.isfinite(value) or value <= 0:
        return float(np.median(history)) if history else None
    history.append(float(value))
    while window > 0 and len(history) > window:
        history.popleft()
    if window <= 0:
        return float(value)
    return float(np.median(history))


class SpeedEstimator:
    """Скорость объекта относительно камеры по изменению дистанции Z.

    Берём в основном |dZ/dt| (range-rate): боковое дрожание ROI на большой
    дальности иначе постоянно «разгоняет» |v|. Сэмплы равномерные по времени;
    выбросы Z в историю не пишутся; малый ΔZ за окно → скорость 0.
    """

    def __init__(
        self,
        *,
        focal_px: float,
        cx: float,
        cy: float,
        window_s: float = 4.0,
        min_dt_s: float = 1.5,
        ema_alpha: float = 0.04,
        sample_interval_s: float = 0.20,
        median_len: int = 11,
        max_accel_mps2: float = 3.0,
        max_z_ratio: float = 1.18,
        max_z_jump_m: float = 10.0,
        min_dz_m: float = 0.8,
        min_speed_mps: float = 0.5,
    ) -> None:
        self.focal_px = max(float(focal_px), 1.0)
        self.cx = float(cx)
        self.cy = float(cy)
        self.window_s = max(0.5, float(window_s))
        self.min_dt_s = max(0.1, float(min_dt_s))
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        # Спуск быстрее подъёма — иначе шумовые пики «защёлкивают» скорость вверх.
        self.ema_alpha_down = float(np.clip(max(self.ema_alpha * 3.0, 0.12), 0.0, 1.0))
        self.sample_interval_s = max(0.05, float(sample_interval_s))
        self.median_len = max(1, int(median_len))
        self.max_accel_mps2 = max(0.5, float(max_accel_mps2))
        self.max_z_ratio = max(1.01, float(max_z_ratio))
        self.max_z_jump_mm = max(100.0, float(max_z_jump_m) * 1000.0)
        self.min_dz_mm = max(50.0, float(min_dz_m) * 1000.0)
        self.min_speed_mps = max(0.0, float(min_speed_mps))
        # (t, z_mm) — только дистанция; боковой ROI для скорости не используем.
        self._samples: deque[tuple[float, float]] = deque()
        self._z_hist: deque[float] = deque(maxlen=21)
        self._raw_speeds: deque[float] = deque(maxlen=self.median_len)
        self._speed_mps: float | None = None
        self._last_sample_t: float | None = None
        self._last_emit_t: float | None = None
        self._z_s: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._z_hist.clear()
        self._raw_speeds.clear()
        self._speed_mps = None
        self._last_sample_t = None
        self._last_emit_t = None
        self._z_s = None

    def _ref_z_mm(self) -> float | None:
        if len(self._z_hist) >= 3:
            return float(np.median(self._z_hist))
        if self._z_s is not None:
            return float(self._z_s)
        return None

    def _is_z_outlier(self, z_mm: float) -> bool:
        ref = self._ref_z_mm()
        if ref is None or ref <= 0:
            return False
        jump = abs(float(z_mm) - ref)
        if jump > self.max_z_jump_mm:
            return True
        ratio = float(z_mm) / ref
        return bool(ratio > self.max_z_ratio or ratio < 1.0 / self.max_z_ratio)

    def _filter_z(self, z_mm: float) -> float | None:
        """Сглаженный Z или None при выбросе (выброс не двигает фильтр)."""
        z_mm = float(z_mm)
        if self._is_z_outlier(z_mm):
            return None
        a_z = 0.15
        if self._z_s is None:
            self._z_s = z_mm
        else:
            self._z_s = (1.0 - a_z) * self._z_s + a_z * z_mm
        self._z_hist.append(float(self._z_s))
        return float(self._z_s)

    @staticmethod
    def _slope_mm_s(times: np.ndarray, values: np.ndarray) -> float:
        t = times - float(times[0])
        if float(t[-1] - t[0]) < 1e-6:
            return 0.0
        return float(np.polyfit(t, values, 1)[0])

    def update(
        self,
        now: float,
        distance_mm: float | None,
        roi: tuple[int, int, int, int] | None,
        *,
        tracking_ok: bool,
    ) -> float | None:
        del roi  # боковой ROI намеренно не используем — источник ложного разгона
        if (
            not tracking_ok
            or distance_mm is None
            or not np.isfinite(distance_mm)
            or distance_mm <= 0
        ):
            return self._speed_mps

        now = float(now)
        z_acc = self._filter_z(float(distance_mm))
        if z_acc is None:
            return self._speed_mps

        # Равномерные сэмплы по времени (НЕ по порогу ΔZ — иначе «лесенка» и разгон).
        if (
            self._last_sample_t is not None
            and (now - self._last_sample_t) < self.sample_interval_s
        ):
            return self._speed_mps

        self._samples.append((now, z_acc))
        self._last_sample_t = now
        cutoff = now - self.window_s
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if len(self._samples) < 3:
            return self._speed_mps

        t0 = self._samples[0][0]
        t1 = self._samples[-1][0]
        dt = t1 - t0
        if dt < self.min_dt_s:
            return self._speed_mps

        times = np.asarray([s[0] for s in self._samples], dtype=np.float64)
        zs = np.asarray([s[1] for s in self._samples], dtype=np.float64)

        # Малый размах Z за окно → нет достоверного движения по дальности.
        z_span = float(np.ptp(zs))  # max - min
        if z_span < self.min_dz_mm:
            speed = 0.0
        else:
            # |dZ/dt|: скорость сближения/удаления относительно камеры.
            vz = abs(self._slope_mm_s(times, zs)) / 1000.0
            # Доп. проверка: простая оценка по концам окна (устойчивее к краям).
            vz_ends = abs(float(zs[-1] - zs[0])) / max(dt, 1e-3) / 1000.0
            speed = float(min(vz, vz_ends * 1.25))  # не раздувать МНК сверх концов

        if not np.isfinite(speed) or speed < 0:
            return self._speed_mps
        if speed < self.min_speed_mps:
            speed = 0.0

        self._raw_speeds.append(speed)
        speed_med = float(np.median(self._raw_speeds))
        if speed_med < self.min_speed_mps:
            speed_med = 0.0

        if self._speed_mps is None or self.ema_alpha >= 1.0:
            smoothed = speed_med
        else:
            a = (
                self.ema_alpha_down
                if speed_med < self._speed_mps
                else self.ema_alpha
            )
            smoothed = (1.0 - a) * self._speed_mps + a * speed_med

        if self._speed_mps is not None and self._last_emit_t is not None:
            dt_emit = max(now - self._last_emit_t, 1e-3)
            max_step = self.max_accel_mps2 * dt_emit
            delta = smoothed - self._speed_mps
            if abs(delta) > max_step:
                smoothed = self._speed_mps + math.copysign(max_step, delta)

        if smoothed < self.min_speed_mps:
            smoothed = 0.0

        self._speed_mps = float(smoothed)
        self._last_emit_t = now
        return self._speed_mps


class DistanceSmoother:
    """Робастное сглаживание дистанции для дальнего стерео.

    Медиана по окну + отсев выбросов (скачок Z/disp в N раз) + лёгкий EMA.
    При устойчивом новом уровне (outlier_patience кадров) принимаем смену.
    """

    def __init__(
        self,
        *,
        window: int = 21,
        max_ratio: float = 2.0,
        ema_alpha: float = 0.2,
        max_disp_jump: float = 1.2,
        outlier_patience: int = 5,
        max_distance_mm: float | None = None,
    ) -> None:
        self.window = max(0, int(window))
        self.max_ratio = max(1.01, float(max_ratio))
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.max_disp_jump = float(max_disp_jump)
        self.outlier_patience = max(1, int(outlier_patience))
        self.max_distance_mm = (
            float(max_distance_mm) if max_distance_mm is not None else None
        )
        self._dist_hist: deque[float] = deque()
        self._disp_hist: deque[float] = deque()
        self._ema_dist: float | None = None
        self._ema_disp: float | None = None
        self._outlier_streak = 0
        self._pending_dist: deque[float] = deque(maxlen=self.outlier_patience)
        self._pending_disp: deque[float] = deque(maxlen=self.outlier_patience)

    def reset(self) -> None:
        self._dist_hist.clear()
        self._disp_hist.clear()
        self._ema_dist = None
        self._ema_disp = None
        self._outlier_streak = 0
        self._pending_dist.clear()
        self._pending_disp.clear()

    def _ref_dist(self) -> float | None:
        if self._ema_dist is not None and self._ema_dist > 0:
            return float(self._ema_dist)
        if self._dist_hist:
            return float(np.median(self._dist_hist))
        return None

    def _ref_disp(self) -> float | None:
        if self._ema_disp is not None and self._ema_disp > 0:
            return float(self._ema_disp)
        if self._disp_hist:
            return float(np.median(self._disp_hist))
        return None

    def _is_outlier(
        self, distance_mm: float, disparity_px: float | None
    ) -> bool:
        if self.max_distance_mm is not None and distance_mm > self.max_distance_mm:
            return True
        ref_z = self._ref_dist()
        if ref_z is not None and ref_z > 0:
            ratio = float(distance_mm) / ref_z
            if ratio > self.max_ratio or ratio < 1.0 / self.max_ratio:
                return True
        if disparity_px is not None and np.isfinite(disparity_px) and disparity_px > 0:
            ref_d = self._ref_disp()
            if ref_d is not None and ref_d > 0:
                d_ratio = float(disparity_px) / ref_d
                if d_ratio > self.max_ratio or d_ratio < 1.0 / self.max_ratio:
                    return True
                if self.max_disp_jump > 0 and abs(float(disparity_px) - ref_d) > self.max_disp_jump:
                    if abs(np.log(d_ratio)) > np.log(1.35):
                        return True
        return False

    def _commit(self, distance_mm: float, disparity_px: float | None) -> None:
        if self.window <= 0:
            self._ema_dist = float(distance_mm)
            if disparity_px is not None and np.isfinite(disparity_px) and disparity_px > 0:
                self._ema_disp = float(disparity_px)
            return
        self._dist_hist.append(float(distance_mm))
        while len(self._dist_hist) > self.window:
            self._dist_hist.popleft()
        med_z = float(np.median(self._dist_hist))
        if self._ema_dist is None or self.ema_alpha >= 1.0:
            self._ema_dist = med_z
        else:
            a = self.ema_alpha
            self._ema_dist = (1.0 - a) * self._ema_dist + a * med_z

        if disparity_px is not None and np.isfinite(disparity_px) and disparity_px > 0:
            self._disp_hist.append(float(disparity_px))
            while len(self._disp_hist) > self.window:
                self._disp_hist.popleft()
            med_d = float(np.median(self._disp_hist))
            if self._ema_disp is None or self.ema_alpha >= 1.0:
                self._ema_disp = med_d
            else:
                a = self.ema_alpha
                self._ema_disp = (1.0 - a) * self._ema_disp + a * med_d

    def update(
        self,
        distance_mm: float | None,
        disparity_px: float | None = None,
    ) -> tuple[float | None, float | None]:
        """Принимает сырое измерение, возвращает (Z_mm, disp) после сглаживания."""
        if distance_mm is None or not np.isfinite(distance_mm) or distance_mm <= 0:
            return self._ema_dist, self._ema_disp
        # Жёсткий потолок сцены — не даём шумному малому d «улететь» за z_far.
        if self.max_distance_mm is not None and float(distance_mm) > self.max_distance_mm:
            return self._ema_dist, self._ema_disp

        if self.window <= 0:
            self._commit(float(distance_mm), disparity_px)
            return self._ema_dist, self._ema_disp

        # Первые точки — набираем без отсева (но уже с потолком z_far выше).
        if len(self._dist_hist) < 3 and self._ema_dist is None:
            self._commit(float(distance_mm), disparity_px)
            self._outlier_streak = 0
            return self._ema_dist, self._ema_disp

        if self._is_outlier(float(distance_mm), disparity_px):
            self._outlier_streak += 1
            self._pending_dist.append(float(distance_mm))
            if disparity_px is not None and np.isfinite(disparity_px):
                self._pending_disp.append(float(disparity_px))
            if self._outlier_streak >= self.outlier_patience and len(self._pending_dist) >= 3:
                pend_med = float(np.median(self._pending_dist))
                if (
                    self.max_distance_mm is not None
                    and pend_med > self.max_distance_mm
                ):
                    self._pending_dist.clear()
                    self._pending_disp.clear()
                    self._outlier_streak = 0
                    return self._ema_dist, self._ema_disp
                self._dist_hist.clear()
                self._disp_hist.clear()
                self._ema_dist = None
                self._ema_disp = None
                for z in self._pending_dist:
                    self._commit(z, None)
                if self._pending_disp:
                    for d in self._pending_disp:
                        self._disp_hist.append(d)
                    self._ema_disp = float(np.median(self._disp_hist))
                self._pending_dist.clear()
                self._pending_disp.clear()
                self._outlier_streak = 0
            return self._ema_dist, self._ema_disp

        self._outlier_streak = 0
        self._pending_dist.clear()
        self._pending_disp.clear()
        self._commit(float(distance_mm), disparity_px)
        return self._ema_dist, self._ema_disp


def configure_threads(n: int) -> int:
    """Настраивает внутренний параллелизм OpenCV. Возвращает фактическое число потоков."""
    import os

    if n <= 0:
        n = os.cpu_count() or 4
    cv2.setNumThreads(int(n))
    actual = int(cv2.getNumThreads())
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    return actual


def _measure_and_smooth(
    *,
    disp_float: np.ndarray | None,
    roi: tuple[int, int, int, int] | None,
    Q,
    args: argparse.Namespace,
    max_disp_cap: float | None,
    min_disp_floor: float,
    max_distance_mm: float | None,
    dist_smoother: DistanceSmoother,
    collect_debug: bool = False,
) -> tuple[float | None, float | None, float | None, DisparityDebugInfo | None]:
    """ROI → сырая дистанция → сглаживание.

    Возвращает (dist_s, disp_s_or_raw, raw_dist, debug|None).
    """
    if roi is None or disp_float is None:
        dist_s, disp_s = dist_smoother.update(None, None)
        return dist_s, disp_s, None, None
    if collect_debug:
        dist, disp_val, dbg = measure_roi_distance(
            disp_float,
            roi,
            Q=Q,
            inset_fraction=args.roi_inset,
            surface=args.surface,
            max_disparity=max_disp_cap,
            min_disparity=min_disp_floor,
            max_distance_mm=max_distance_mm,
            collect_debug=True,
        )
    else:
        dist, disp_val = measure_roi_distance(
            disp_float,
            roi,
            Q=Q,
            inset_fraction=args.roi_inset,
            surface=args.surface,
            max_disparity=max_disp_cap,
            min_disparity=min_disp_floor,
            max_distance_mm=max_distance_mm,
        )
        dbg = None
    dist_s, disp_s = dist_smoother.update(dist, disp_val)
    out_disp = disp_s if disp_s is not None else disp_val
    return dist_s, out_disp, dist, dbg


def _maybe_adapt_matcher(
    *,
    auto_disp: bool,
    calib: dict | None,
    args: argparse.Namespace,
    rect_w: int,
    disp_min: int,
    disp_num: int,
    dist_s: float | None,
    dist: float | None,
    disp_val: float | None,
    long_range: bool,
    uniqueness: int,
    matcher,
) -> tuple[int, int, object]:
    """При необходимости расширяет диапазон SGBM и пересоздаёт matcher."""
    if not auto_disp or calib is None:
        return disp_min, disp_num, matcher
    new_min, new_num, adapt_log = adapt_disparity_range(
        calib=calib,
        image_width=rect_w,
        z_near_m=args.z_near,
        z_far_m=args.z_far,
        cur_min=disp_min,
        cur_num=disp_num,
        distance_mm=dist_s if dist_s is not None else dist,
        disparity_px=disp_val,
        long_range=long_range,
    )
    if adapt_log is None:
        return disp_min, disp_num, matcher
    matcher = make_stereo_matcher(
        args.method,
        new_min,
        new_num,
        args.block_size,
        uniqueness_ratio=uniqueness,
        speckle_window_size=40 if long_range else 50,
    )
    print(adapt_log)
    return new_min, new_num, matcher


def main() -> None:
    args = parse_args()

    if args.num_disparities % 16 != 0:
        sys.exit("Ошибка: --num-disparities должен быть кратен 16.")
    if args.block_size % 2 == 0:
        sys.exit("Ошибка: --block-size должен быть нечётным.")
    if args.sgbm_interval < 1:
        sys.exit("Ошибка: --sgbm-interval должен быть >= 1.")
    if args.workers < 1:
        sys.exit("Ошибка: --workers должен быть >= 1.")
    if not 0.0 <= args.roi_smooth < 1.0:
        sys.exit("Ошибка: --roi-smooth должен быть в диапазоне [0.0, 1.0).")
    if not 0.0 <= args.reacquire_threshold <= 1.0:
        sys.exit("Ошибка: --reacquire-threshold должен быть в диапазоне [0.0, 1.0].")
    if args.z_near <= 0 or args.z_far <= 0 or args.z_near >= args.z_far:
        sys.exit("Ошибка: нужно 0 < --z-near < --z-far (дистанции в метрах).")
    if args.max_fps < 0:
        sys.exit("Ошибка: --max-fps должен быть >= 0 (0 = без ограничения).")
    if not (0.0 <= args.roi_inset < 0.45):
        sys.exit("Ошибка: --roi-inset должен быть в диапазоне [0.0, 0.45).")
    long_range = bool(args.long_range) or args.z_far >= 800.0
    if long_range and args.z_near < 80:
        print(
            "Предупреждение: для 1000+ м лучше --z-near 150..300 "
            f"(сейчас {args.z_near}). Иначе SGBM ищет слишком большие d."
        )

    if args.smooth_max_ratio < 1.01:
        sys.exit("Ошибка: --smooth-max-ratio должен быть >= 1.01.")
    if not 0.0 <= args.smooth_ema <= 1.0:
        sys.exit("Ошибка: --smooth-ema должен быть в диапазоне [0.0, 1.0].")

    use_sbs = args.video is not None
    use_dual = args.left_video is not None or args.right_video is not None
    if use_sbs and use_dual:
        sys.exit(
            "Ошибка: укажите либо --video (SBS), либо пару --left-video/--right-video."
        )
    if use_sbs:
        source = StereoFrameSource(sbs_path=args.video, swap_lr=args.swap_lr)
    elif args.left_video and args.right_video:
        source = StereoFrameSource(
            left_path=args.left_video,
            right_path=args.right_video,
            swap_lr=args.swap_lr,
        )
    else:
        sys.exit(
            "Ошибка: укажите --video (SBS) либо оба --left-video и --right-video."
        )

    opencv_threads = configure_threads(args.threads)
    print(
        f"Потоки OpenCV: {opencv_threads} "
        f"(задано --threads {args.threads}), "
        f"workers={args.workers}, async_sgbm={args.async_sgbm}"
    )

    track_only = args.track_only
    Q = None
    matcher = None
    calib = None
    if not track_only and not args.calib:
        sys.exit("Ошибка: --calib обязателен (кроме режима --track-only).")

    if track_only:
        print("Режим --track-only: только захват и трекинг (без SGBM и расстояния).")

    if args.calib:
        print(f"Загрузка калибровки: {args.calib}")
        calib = load_calibration(args.calib)
        for line in format_quality_report(calibration_quality_warnings(calib)):
            print(line)
        try:
            focal, baseline = extract_calib_geometry(calib)
            d1000 = focal * baseline / 1_000_000.0
            d200 = focal * baseline / 200_000.0
            d50 = focal * baseline / 50_000.0
            print(
                f"Геометрия: f={focal:.1f}px, B={baseline:.1f}mm → "
                f"disp@1000м≈{d1000:.2f}px, @200м≈{d200:.2f}px, @50м≈{d50:.2f}px"
            )
            if d1000 < 1.2:
                print(
                    "Предупреждение: на 1000 м диспаритет < 1.2 px — нужна калибровка "
                    "с большим f·B (teplo_*), иначе дистанция будет нестабильной."
                )
            print(
                "Ожидание: при 1000+ м disp должен быть малым (единицы px). "
                "Большой disp = ложное совпадение / ближняя поверхность."
            )
        except Exception:
            pass
    elif track_only:
        print("Калибровка не задана — трекинг по «сырым» кадрам без ректификации.")

    disp_min = int(args.min_disparity)
    disp_num = int(args.num_disparities)
    auto_disp = bool(args.auto_disparity) and not track_only

    prep_pool = ThreadPoolExecutor(max_workers=args.workers)
    # Отдельный пул на 1 поток: matcher.compute не запускаем параллельно самому себе.
    sgbm_pool = (
        ThreadPoolExecutor(max_workers=1) if (args.async_sgbm and not track_only) else None
    )
    sgbm_future: Future | None = None

    ok, frame_l, frame_r = source.read()
    if not ok or frame_l is None or frame_r is None:
        source.release()
        sys.exit("Ошибка: не удалось прочитать первый кадр видео.")

    print(
        f"Источник кадров: {'SBS ' + args.video if use_sbs else f'L={args.left_video}, R={args.right_video}'}"
    )

    rect_l, rect_r = prepare_pair(
        frame_l, frame_r, calib, prep_pool, clahe=args.clahe
    )
    rect_l_bgr = cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR)

    max_disp_cap: float | None = None
    min_disp_floor = 0.75
    max_distance_mm: float | None = None
    uniqueness = 5
    if not track_only:
        Q = calib["Q"]
        if long_range:
            uniqueness = 12
            print(f"Режим long-range: z={args.z_near:.0f}–{args.z_far:.0f} м, uniqueness={uniqueness}")
        if auto_disp and calib is not None:
            disp_min, disp_num, range_log = estimate_disparity_range_bounds(
                calib,
                args.z_near,
                args.z_far,
                image_width=int(rect_l.shape[1]),
                long_range=long_range,
            )
            max_num = 96 if long_range else 512
            disp_min, disp_num = clamp_sgbm_range(
                disp_min, disp_num, int(rect_l.shape[1]), max_num=max_num
            )
            print(range_log)
        else:
            if args.auto_disparity and calib is None:
                print(
                    "Предупреждение: --auto-disparity без --calib — "
                    "фиксированный --num-disparities."
                )
                auto_disp = False
            print(
                f"Фиксированный диапазон диспаритета: "
                f"min={disp_min}, num={disp_num}."
            )
        try:
            focal, baseline = extract_calib_geometry(calib)
            # Потолок d: чуть выше d(z_near), всё большее — мусор для дальней сцены.
            max_disp_cap = disparity_from_depth(
                focal, baseline, args.z_near * 1000.0
            ) * 1.05
            # Пол d: не ниже ~0.75*d(z_far) — иначе шум 0.4px даёт Z в тысячи км.
            d_far = disparity_from_depth(focal, baseline, args.z_far * 1000.0)
            min_disp_floor = max(0.85, float(d_far) * 0.75)
            max_distance_mm = float(args.z_far) * 1000.0 * 1.15
            print(
                f"Потолок d ≤ {max_disp_cap:.2f} px (z_near={args.z_near:.0f} м); "
                f"пол d ≥ {min_disp_floor:.2f} px; Z ≤ {max_distance_mm/1000:.0f} м"
            )
        except Exception:
            max_disp_cap = float(disp_min + disp_num)
            max_distance_mm = float(args.z_far) * 1000.0 * 1.15
        matcher = make_stereo_matcher(
            args.method,
            disp_min,
            disp_num,
            args.block_size,
            uniqueness_ratio=uniqueness,
            speckle_window_size=40 if long_range else 50,
        )

    # ROI можно выбрать в любой момент клавишей R — на старте объекта нет.
    # Thermal IR: base (KCF или NCC/Kornia) + CLAHE-NCC + intensity + reacquire.
    tracker_kind = "ncc" if args.kornia_tracker else args.tracker
    if args.kornia_tracker and args.tracker not in ("ncc", "kcf"):
        print(
            f"Предупреждение: --kornia-tracker включает NCC; "
            f"--tracker {args.tracker} игнорируется."
        )
    tracker = ObjectTracker(
        kind=tracker_kind,
        smooth=args.roi_smooth,
        lock_size=args.lock_size,
        keep_aspect=args.keep_aspect,
        max_scale_step=args.max_scale_step,
        max_size_ratio=args.max_size_ratio,
        verify=args.verify,
        verify_threshold=args.verify_threshold,
        max_jump=args.max_jump,
        lost_patience=args.lost_patience,
        verify_rel=args.verify_rel,
        min_iou=args.min_iou,
        reacquire=args.reacquire,
        reacquire_threshold=args.reacquire_threshold,
        reacquire_radius=args.reacquire_radius,
        reacquire_global=args.reacquire_global,
        reacquire_interval=args.reacquire_interval,
        reacquire_scale_min=args.reacquire_scale_min,
        reacquire_scale_max=args.reacquire_scale_max,
    )
    roi: tuple[int, int, int, int] | None = None
    tracking_ok = False

    disp_float: np.ndarray | None = None
    dist = disp_val = None
    dist_smoother = DistanceSmoother(
        window=args.smooth,
        max_ratio=args.smooth_max_ratio,
        ema_alpha=args.smooth_ema,
        max_disp_jump=args.smooth_disp_jump,
        outlier_patience=max(4, args.smooth // 3),
        max_distance_mm=max_distance_mm,
    )
    dist_s = None
    disp_s = None
    speed_mps: float | None = None
    speed_est: SpeedEstimator | None = None
    if args.speed and not track_only and calib is not None:
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
        speed_est = SpeedEstimator(
            focal_px=focal_px,
            cx=cam_cx,
            cy=cam_cy,
            window_s=args.speed_window,
            min_dt_s=args.speed_min_dt,
            ema_alpha=args.speed_ema,
            max_z_ratio=args.speed_max_z_ratio,
            max_z_jump_m=args.speed_max_z_jump,
            min_dz_m=args.speed_min_dz,
            min_speed_mps=args.speed_min,
        )
        print(
            f"Скорость объекта: окно {args.speed_window:.2f} с, "
            f"мин. {args.speed_min_dt:.2f} с, EMA={args.speed_ema:.2f}; "
            f"отсев ΔZ>{args.speed_max_z_jump:.0f} м / ×{args.speed_max_z_ratio:.2f}, "
            f"мин.ΔZ={args.speed_min_dz:.1f} м, мин.v={args.speed_min:.1f} м/с."
        )

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = float(args.max_fps) if args.max_fps > 0 else source.fps
        writer = cv2.VideoWriter(
            args.output,
            fourcc,
            out_fps,
            (rect_l_bgr.shape[1], rect_l_bgr.shape[0]),
        )

    window = "Track + distance (Space=pause, R/C=select, X=cancel, D=disp, Q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    debug_window = "Disparity debug (L | R correspondences)"
    debug_disparity = bool(args.debug_disparity) and not track_only
    disp_debug: DisparityDebugInfo | None = None
    overlay: np.ndarray | None = None
    paused = False
    frame_idx = 0
    t_prev = time.perf_counter()
    fps = 0.0

    print("R — выбрать объект рамкой, C — кликом (авто-границы). В любой момент.")
    print("X — отменить трекинг (во время трека и при потере цели).")
    base_label = tracker_kind.upper()
    if tracker_kind == "ncc":
        base_label = "NCC(Kornia-style)"
    print(f"Трекер: гибрид {base_label}+refine (reacquire={args.reacquire}).")
    if track_only:
        print("Режим --track-only: без SGBM и расстояния.")
    else:
        print(f"SGBM каждые {args.sgbm_interval} кадр(ов).")
        if debug_disparity:
            print("Отладка диспаритета ВКЛ (клавиша D — переключить).")
        else:
            print("Клавиша D — окно соответствий L↔R по диспаритету ROI.")

    try:
        while True:
            if not paused:
                if frame_idx > 0:
                    ok, frame_l, frame_r = source.read()
                    if not ok or frame_l is None or frame_r is None:
                        print("Конец видео.")
                        break
                    if args.max_frames > 0 and frame_idx >= args.max_frames:
                        print("Достигнут --max-frames.")
                        break

                    # 1) prepare
                    rect_l, rect_r = prepare_pair(
                        frame_l, frame_r, calib, prep_pool, clahe=args.clahe
                    )
                    rect_l_bgr = cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR)

                    # 2) track
                    if tracker.initialized:
                        tracking_ok, roi = tracker.update(rect_l_bgr)

                    # 3) depth (только при живом треке)
                    if not track_only and tracker.initialized and tracking_ok:
                        if sgbm_future is not None and sgbm_future.done():
                            disp_float = sgbm_future.result()
                            sgbm_future = None

                        need_sgbm = frame_idx % args.sgbm_interval == 0
                        if need_sgbm:
                            if sgbm_pool is not None:
                                if sgbm_future is None or sgbm_future.done():
                                    if sgbm_future is not None and sgbm_future.done():
                                        disp_float = sgbm_future.result()
                                    sgbm_future = sgbm_pool.submit(
                                        compute_disparity,
                                        rect_l.copy(),
                                        rect_r.copy(),
                                        matcher,
                                        wls=args.wls,
                                        wls_lambda=args.wls_lambda,
                                        wls_sigma=args.wls_sigma,
                                    )
                            else:
                                disp_float = compute_disparity(
                                    rect_l,
                                    rect_r,
                                    matcher,
                                    wls=args.wls,
                                    wls_lambda=args.wls_lambda,
                                    wls_sigma=args.wls_sigma,
                                )

                        if roi is not None and disp_float is not None:
                            if sgbm_future is not None and sgbm_future.done():
                                disp_float = sgbm_future.result()
                                sgbm_future = None
                            dist_s, disp_val, dist, disp_debug = _measure_and_smooth(
                                disp_float=disp_float,
                                roi=roi,
                                Q=Q,
                                args=args,
                                max_disp_cap=max_disp_cap,
                                min_disp_floor=min_disp_floor,
                                max_distance_mm=max_distance_mm,
                                dist_smoother=dist_smoother,
                                collect_debug=debug_disparity,
                            )
                            if auto_disp and (
                                sgbm_future is None or sgbm_future.done()
                            ):
                                disp_min, disp_num, matcher = _maybe_adapt_matcher(
                                    auto_disp=auto_disp,
                                    calib=calib,
                                    args=args,
                                    rect_w=int(rect_l.shape[1]),
                                    disp_min=disp_min,
                                    disp_num=disp_num,
                                    dist_s=dist_s,
                                    dist=dist,
                                    disp_val=disp_val,
                                    long_range=long_range,
                                    uniqueness=uniqueness,
                                    matcher=matcher,
                                )
                        else:
                            dist_s, disp_val, _, disp_debug = _measure_and_smooth(
                                disp_float=None,
                                roi=None,
                                Q=Q,
                                args=args,
                                max_disp_cap=max_disp_cap,
                                min_disp_floor=min_disp_floor,
                                max_distance_mm=max_distance_mm,
                                dist_smoother=dist_smoother,
                            )
                    elif not track_only and tracker.initialized and not tracking_ok:
                        dist_s, disp_val, _, disp_debug = _measure_and_smooth(
                            disp_float=None,
                            roi=None,
                            Q=Q,
                            args=args,
                            max_disp_cap=max_disp_cap,
                            min_disp_floor=min_disp_floor,
                            max_distance_mm=max_distance_mm,
                            dist_smoother=dist_smoother,
                        )
                        if speed_est is not None:
                            speed_est.reset()
                            speed_mps = None

                if (
                    speed_est is not None
                    and tracker.initialized
                    and tracking_ok
                    and dist_s is not None
                    and roi is not None
                ):
                    speed_mps = speed_est.update(
                        time.perf_counter(),
                        dist_s,
                        roi,
                        tracking_ok=True,
                    )
                elif speed_est is not None and not tracker.initialized:
                    speed_mps = None

                # 4) draw
                sgbm_busy = sgbm_future is not None and not sgbm_future.done()
                overlay = draw_overlay(
                    rect_l_bgr,
                    roi,
                    dist_s,
                    disp_val,
                    tracking_ok,
                    frame_idx,
                    fps,
                    sgbm_busy=sgbm_busy,
                    disp_range=(disp_min, disp_num) if not track_only else None,
                    speed_mps=speed_mps if speed_est is not None else None,
                )
                if debug_disparity:
                    overlay = _annotate_debug_flag(overlay)
                if writer is not None:
                    writer.write(overlay)

                if debug_disparity and disp_debug is not None and rect_r is not None:
                    rect_r_bgr = cv2.cvtColor(rect_r, cv2.COLOR_GRAY2BGR)
                    dbg_vis = draw_disparity_debug(
                        rect_l_bgr,
                        rect_r_bgr,
                        disp_debug,
                        max_samples=max(8, int(args.debug_disp_samples)),
                        track_roi=roi,
                    )
                    cv2.namedWindow(debug_window, cv2.WINDOW_NORMAL)
                    dbg_scale = display_scale(dbg_vis.shape, max(args.max_display, 1600))
                    cv2.imshow(debug_window, fit_for_display(dbg_vis, dbg_scale))
                elif not debug_disparity:
                    try:
                        cv2.destroyWindow(debug_window)
                    except cv2.error:
                        pass

                now = time.perf_counter()
                if args.max_fps > 0:
                    remaining = (1.0 / float(args.max_fps)) - (now - t_prev)
                    if remaining > 0:
                        time.sleep(remaining)
                        now = time.perf_counter()
                dt = now - t_prev
                t_prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

                scale = display_scale(overlay.shape, args.max_display)
                cv2.imshow(window, fit_for_display(overlay, scale))
                frame_idx += 1

            # 5) keys — при нагрузке ждём чуть дольше, чтобы HighGUI успевал забирать ввод
            key = cv2.waitKey(10 if not paused else 50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key in (ord("x"), ord("X"), 8):  # X или Backspace — сброс трека
                if tracker.initialized or roi is not None:
                    tracker.reset()
                    roi = None
                    tracking_ok = False
                    dist_smoother.reset()
                    if speed_est is not None:
                        speed_est.reset()
                    dist_s = None
                    disp_s = None
                    disp_val = None
                    disp_debug = None
                    speed_mps = None
                    print("Трекинг отменён. R/C — выбрать объект заново.")
                    base = rect_l_bgr if rect_l_bgr is not None else overlay
                    if base is not None:
                        overlay = draw_overlay(
                            base,
                            None,
                            None,
                            None,
                            False,
                            frame_idx,
                            fps,
                            sgbm_busy=False,
                            disp_range=(disp_min, disp_num) if not track_only else None,
                            speed_mps=None,
                        )
                        cv2.imshow(
                            window,
                            fit_for_display(
                                overlay, display_scale(overlay.shape, args.max_display)
                            ),
                        )
            if key in (ord("d"), ord("D")) and not track_only:
                debug_disparity = not debug_disparity
                print(
                    "Отладка диспаритета: "
                    + ("ВКЛ" if debug_disparity else "ВЫКЛ")
                )
                if not debug_disparity:
                    try:
                        cv2.destroyWindow(debug_window)
                    except cv2.error:
                        pass
            if key in (ord("r"), ord("R"), ord("c"), ord("C")):
                by_click = key in (ord("c"), ord("C"))
                print(
                    "Выбор объекта "
                    + ("кликом" if by_click else "рамкой")
                    + " на текущем кадре..."
                )
                if sgbm_future is not None:
                    sgbm_future.result()
                    sgbm_future = None
                # Пауза чтения видео на время UI, чтобы окно получало фокус/клавиши.
                was_paused = paused
                paused = True
                if by_click:
                    new_roi = tracker.init_by_click(
                        rect_l_bgr,
                        args.max_display,
                        tolerance=args.click_tolerance,
                        grabcut_refine=not args.no_grabcut,
                    )
                else:
                    new_roi = tracker.init_interactive(rect_l_bgr, args.max_display)
                # selectROI / click-UI часто оставляют 'c'/'r' в очереди OpenCV —
                # без сброса снова открывается выбор (петля) и «клавиши не работают».
                _flush_waitkey_buffer()
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                vis = overlay if overlay is not None else rect_l_bgr
                cv2.imshow(
                    window,
                    fit_for_display(vis, display_scale(vis.shape, args.max_display)),
                )
                paused = was_paused
                if new_roi is not None:
                    roi = new_roi
                    tracking_ok = True
                    dist_smoother.reset()
                    if speed_est is not None:
                        speed_est.reset()
                    dist_s = None
                    disp_s = None
                    speed_mps = None
                    if not track_only and matcher is not None:
                        disp_float = compute_disparity(
                            rect_l,
                            rect_r,
                            matcher,
                            wls=args.wls,
                            wls_lambda=args.wls_lambda,
                            wls_sigma=args.wls_sigma,
                        )
                        dist_s, disp_val, dist, disp_debug = _measure_and_smooth(
                            disp_float=disp_float,
                            roi=roi,
                            Q=Q,
                            args=args,
                            max_disp_cap=max_disp_cap,
                            min_disp_floor=min_disp_floor,
                            max_distance_mm=max_distance_mm,
                            dist_smoother=dist_smoother,
                            collect_debug=debug_disparity,
                        )
                        disp_min, disp_num, matcher = _maybe_adapt_matcher(
                            auto_disp=auto_disp,
                            calib=calib,
                            args=args,
                            rect_w=int(rect_l.shape[1]),
                            disp_min=disp_min,
                            disp_num=disp_num,
                            dist_s=dist_s,
                            dist=dist,
                            disp_val=disp_val,
                            long_range=long_range,
                            uniqueness=uniqueness,
                            matcher=matcher,
                        )
    finally:
        if sgbm_future is not None:
            try:
                sgbm_future.result(timeout=30)
            except Exception:
                pass
        prep_pool.shutdown(wait=False)
        if sgbm_pool is not None:
            sgbm_pool.shutdown(wait=False)
        source.release()
        if writer is not None:
            writer.release()
            print(f"Видео сохранено: {Path(args.output).resolve()}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
