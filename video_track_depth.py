"""
Трекинг объекта по одному SBS-видео (side-by-side) с измерением расстояния.

Вход — один видеофайл, где кадр разрезан пополам: левая половина = левая
камера, правая половина = правая камера. Пользователь выделяет объект на
первом кадре (левая половина). Дальше объект сопровождается трекером
(CSRT/KCF), а расстояние считается по медиане диспаритета внутри ROI.

Ускорение на CPU:
  - cv2.setNumThreads — внутренний параллелизм OpenCV (SGBM, remap);
  - параллельная подготовка левого/правого кадра;
  - асинхронный SGBM в фоне, чтобы трекинг не ждал каждый тяжёлый кадр.

Пример:
    python video_track_depth.py ^
        --video stereo_sbs.mp4 ^
        --calib stereo_calib.npz --threads 0 --async-sgbm

Управление:
    пробел  — пауза/продолжить
    r       — заново выбрать объект
    q / Esc — выход
"""

from __future__ import annotations

import argparse
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
    fit_for_display,
    load_calibration,
    measure_roi_distance,
)
from object_tracker import ObjectTracker
from calib_quality import format_quality_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Трекинг объекта по SBS-видео со стерео-расстоянием.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--video",
        required=True,
        help="SBS-видео: левая половина кадра — левая камера, правая — правая.",
    )
    p.add_argument(
        "--swap-lr",
        action="store_true",
        help="Поменять половины местами (если левая камера справа).",
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
    p.add_argument("--num-disparities", type=int, default=128)
    p.add_argument("--block-size", type=int, default=5)
    p.add_argument("--min-disparity", type=int, default=0)
    p.add_argument("--wls", action="store_true", help="WLS-фильтр (медленнее).")
    p.add_argument("--wls-lambda", type=float, default=8000.0)
    p.add_argument("--wls-sigma", type=float, default=1.5)
    p.add_argument(
        "--tracker",
        choices=["csrt", "kcf", "mosse"],
        default="csrt",
        help="Тип OpenCV-трекера.",
    )
    p.add_argument(
        "--roi-smooth",
        type=float,
        default=0.6,
        help="Сглаживание рамки трекинга [0..1): 0 = без сглаживания, ближе к 1 = плавнее.",
    )
    p.add_argument(
        "--lock-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Фиксировать размер рамки трекинга (двигается только центр). Резкое сжатие → LOST.",
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
        default=0.05,
        help="Макс. относительное изменение масштаба рамки за кадр (0 = без лимита).",
    )
    p.add_argument(
        "--max-size-ratio",
        type=float,
        default=1.6,
        help="Макс. изменение площади рамки за кадр (1.6 ≈ ±60%%); иначе LOST.",
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
        default=0.45,
        help="Мин. сходство с эталоном [0..1], ниже — считаем кадр плохим.",
    )
    p.add_argument(
        "--max-jump",
        type=float,
        default=0.7,
        help="Макс. прыжок центра за кадр в долях диагонали ROI (0 = без лимита).",
    )
    p.add_argument(
        "--verify-rel",
        type=float,
        default=0.75,
        help="Отклонять кадр, если score < EMA*verify-rel (дрейф на соседний объект).",
    )
    p.add_argument(
        "--min-iou",
        type=float,
        default=0.5,
        help="Мин. IoU с предыдущей рамкой (отсекает перескок на пересекающийся объект).",
    )
    p.add_argument(
        "--lost-patience",
        type=int,
        default=2,
        help="Сколько подряд плохих кадров нужно, чтобы объявить потерю цели.",
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
        default=0.62,
        help="Порог совпадения для повторного захвата [0..1].",
    )
    p.add_argument(
        "--reacquire-radius",
        type=float,
        default=2.5,
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
        default=5,
        help="Искать объект при потере каждые N кадров (1 = каждый кадр, тяжелее).",
    )
    p.add_argument(
        "--reacquire-scale-min",
        type=float,
        default=0.35,
        help="Мин. масштаб эталона при перезахвате (меньше = искать более далёкий/мелкий объект).",
    )
    p.add_argument(
        "--reacquire-scale-max",
        type=float,
        default=1.4,
        help="Макс. масштаб эталона при перезахвате.",
    )
    p.add_argument(
        "--click-tolerance",
        type=int,
        default=16,
        help="Допуск цвета при авто-выделении объекта кликом (клавиша C).",
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
        help="Считать SGBM каждые N кадров (1 = каждый кадр).",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Окно медианы по последним N измерениям расстояния (0 = без сглаживания).",
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
    return p.parse_args()


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Ошибка: не удалось открыть видео '{path}'.")
    return cap


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


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
) -> np.ndarray:
    """Gray → resize → remap для одной камеры (удобно гонять в ThreadPool)."""
    gray = to_gray(frame)
    gray = resize_to_calib(gray, calib_size)
    return cv2.remap(gray, map1, map2, cv2.INTER_LINEAR)


def prepare_pair(
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    calib: dict | None,
    pool: ThreadPoolExecutor | None,
) -> tuple[np.ndarray, np.ndarray]:
    # Без калибровки ректификация невозможна: только gray (для трекинга этого хватает).
    if calib is None:
        return to_gray(frame_l), to_gray(frame_r)

    calib_size = None
    if "image_size" in calib:
        calib_size = (int(calib["image_size"][0]), int(calib["image_size"][1]))

    if pool is None:
        rect_l = prepare_side(frame_l, calib["map1_l"], calib["map2_l"], calib_size)
        rect_r = prepare_side(frame_r, calib["map1_r"], calib["map2_r"], calib_size)
        return rect_l, rect_r

    fut_l = pool.submit(
        prepare_side, frame_l, calib["map1_l"], calib["map2_l"], calib_size
    )
    fut_r = pool.submit(
        prepare_side, frame_r, calib["map1_r"], calib["map2_r"], calib_size
    )
    return fut_l.result(), fut_r.result()


def split_sbs(
    frame: np.ndarray, swap_lr: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Разрезает SBS-кадр пополам по ширине на левую и правую камеры."""
    w = frame.shape[1]
    half = w // 2
    left = frame[:, :half]
    right = frame[:, half : half * 2]
    if swap_lr:
        left, right = right, left
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def read_sbs(
    cap: cv2.VideoCapture, swap_lr: bool = False
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Читает один SBS-кадр и делит его на левую/правую половины."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None, None
    left, right = split_sbs(frame, swap_lr)
    return True, left, right


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


def draw_overlay(
    frame_bgr: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    distance_mm: float | None,
    disparity: float | None,
    tracking_ok: bool,
    frame_idx: int,
    fps: float,
    sgbm_busy: bool = False,
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
    if disparity is not None:
        lines.append(f"disp {disparity:.1f} px")
    if sgbm_busy:
        lines.append("SGBM...")
    if roi is None:
        lines.append("press R (box) or C (click) to select")
    elif not tracking_ok:
        lines.append("TRACK LOST — searching / press R")

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
    if value is None or not np.isfinite(value) or value <= 0:
        return float(np.median(history)) if history else None
    history.append(float(value))
    while window > 0 and len(history) > window:
        history.popleft()
    if window <= 0:
        return float(value)
    return float(np.median(history))


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
    elif track_only:
        print("Калибровка не задана — трекинг по «сырым» кадрам без ректификации.")

    if not track_only:
        Q = calib["Q"]
        if args.method == "sgbm":
            matcher = build_sgbm(
                args.min_disparity, args.num_disparities, args.block_size
            )
        else:
            matcher = build_bm(args.num_disparities, args.block_size)

    prep_pool = ThreadPoolExecutor(max_workers=args.workers)
    # Отдельный пул на 1 поток: matcher.compute не запускаем параллельно самому себе.
    sgbm_pool = (
        ThreadPoolExecutor(max_workers=1) if (args.async_sgbm and not track_only) else None
    )
    sgbm_future: Future | None = None

    cap = open_video(args.video)

    ok, frame_l, frame_r = read_sbs(cap, args.swap_lr)
    if not ok or frame_l is None or frame_r is None:
        sys.exit("Ошибка: не удалось прочитать первый кадр SBS-видео.")

    rect_l, rect_r = prepare_pair(frame_l, frame_r, calib, prep_pool)
    rect_l_bgr = cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR)

    # ROI можно выбрать в любой момент клавишей R — на старте объекта нет.
    tracker = ObjectTracker(
        kind=args.tracker,
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
    history: deque[float] = deque()
    dist_s = None

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.output,
            fourcc,
            max(cap.get(cv2.CAP_PROP_FPS), 1.0),
            (rect_l_bgr.shape[1], rect_l_bgr.shape[0]),
        )

    window = "Track + distance (Space=pause, R=box, C=click, Q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    paused = False
    frame_idx = 0
    t_prev = time.perf_counter()
    fps = 0.0

    print("R — выбрать объект рамкой, C — кликом (авто-границы). В любой момент.")
    if track_only:
        print(f"Готово к трекингу (только трекинг, трекер={args.tracker}).")
    else:
        print(
            f"Готово. SGBM каждые {args.sgbm_interval} кадр(ов), трекер={args.tracker}."
        )

    try:
        while True:
            if not paused:
                if frame_idx > 0:
                    ok, frame_l, frame_r = read_sbs(cap, args.swap_lr)
                    if not ok or frame_l is None or frame_r is None:
                        print("Конец видео.")
                        break
                    if args.max_frames > 0 and frame_idx >= args.max_frames:
                        print("Достигнут --max-frames.")
                        break

                    rect_l, rect_r = prepare_pair(frame_l, frame_r, calib, prep_pool)
                    rect_l_bgr = cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR)

                    if tracker.initialized:
                        tracking_ok, roi = tracker.update(rect_l_bgr)

                    # SGBM только при живом треке — иначе при LOST зря жрёт FPS.
                    if not track_only and tracker.initialized and tracking_ok:
                        # Забираем готовый асинхронный диспаритет, если есть.
                        if sgbm_future is not None and sgbm_future.done():
                            disp_float = sgbm_future.result()
                            sgbm_future = None

                        need_sgbm = frame_idx % args.sgbm_interval == 0
                        if need_sgbm:
                            if sgbm_pool is not None:
                                # Не ставим новый SGBM, пока предыдущий считается.
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
                            dist, disp_val = measure_roi_distance(disp_float, roi, Q=Q)
                            dist_s = smoothed_value(history, dist, args.smooth)
                        else:
                            dist_s = smoothed_value(history, None, args.smooth)
                            disp_val = None
                    elif not track_only and tracker.initialized and not tracking_ok:
                        dist_s = smoothed_value(history, None, args.smooth)
                        disp_val = None

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
                )
                if writer is not None:
                    writer.write(overlay)

                now = time.perf_counter()
                dt = now - t_prev
                t_prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

                scale = display_scale(overlay.shape, args.max_display)
                cv2.imshow(window, fit_for_display(overlay, scale))
                frame_idx += 1

            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
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
                if by_click:
                    new_roi = tracker.init_by_click(
                        rect_l_bgr,
                        args.max_display,
                        tolerance=args.click_tolerance,
                        grabcut_refine=not args.no_grabcut,
                    )
                else:
                    new_roi = tracker.init_interactive(rect_l_bgr, args.max_display)
                if new_roi is not None:
                    roi = new_roi
                    tracking_ok = True
                    history.clear()
                    if not track_only:
                        disp_float = compute_disparity(
                            rect_l,
                            rect_r,
                            matcher,
                            wls=args.wls,
                            wls_lambda=args.wls_lambda,
                            wls_sigma=args.wls_sigma,
                        )
                        dist, disp_val = measure_roi_distance(disp_float, roi, Q=Q)
                        dist_s = smoothed_value(history, dist, args.smooth)
    finally:
        if sgbm_future is not None:
            try:
                sgbm_future.result(timeout=30)
            except Exception:
                pass
        prep_pool.shutdown(wait=False)
        if sgbm_pool is not None:
            sgbm_pool.shutdown(wait=False)
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Видео сохранено: {Path(args.output).resolve()}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
