"""
Трекинг объекта по двум видео (левая и правая камера) с измерением расстояния.

Пользователь выделяет объект на первом кадре левого видео. Дальше объект
сопровождается трекером (CSRT/KCF), а расстояние считается по медиане
диспаритета внутри ROI на стереопаре.

Ускорение на CPU:
  - cv2.setNumThreads — внутренний параллелизм OpenCV (SGBM, remap);
  - параллельная подготовка левого/правого кадра;
  - асинхронный SGBM в фоне, чтобы трекинг не ждал каждый тяжёлый кадр.

Пример:
    python video_track_depth.py ^
        --left-video left.mp4 --right-video right.mp4 ^
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
from calib_quality import format_quality_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Трекинг объекта по двум видео со стерео-расстоянием.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--left-video", required=True, help="Видео с левой камеры.")
    p.add_argument("--right-video", required=True, help="Видео с правой камеры.")
    p.add_argument(
        "--calib",
        required=True,
        help="Файл стереокалибровки (.npz) — нужен для ректификации и мм.",
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


def create_tracker(kind: str):
    """Создаёт трекер с учётом разных сборок OpenCV."""
    factories = {
        "csrt": [
            ("TrackerCSRT_create", cv2),
            ("TrackerCSRT_create", getattr(cv2, "legacy", None)),
        ],
        "kcf": [
            ("TrackerKCF_create", cv2),
            ("TrackerKCF_create", getattr(cv2, "legacy", None)),
        ],
        "mosse": [
            ("TrackerMOSSE_create", cv2),
            ("TrackerMOSSE_create", getattr(cv2, "legacy", None)),
        ],
    }
    for name, mod in factories[kind]:
        if mod is None:
            continue
        factory = getattr(mod, name, None)
        if factory is not None:
            return factory()
    raise RuntimeError(
        f"Трекер '{kind}' недоступен в этой сборке OpenCV. "
        "Установите opencv-contrib-python."
    )


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
    calib: dict,
    pool: ThreadPoolExecutor | None,
) -> tuple[np.ndarray, np.ndarray]:
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


def read_pair(
    cap_l: cv2.VideoCapture,
    cap_r: cv2.VideoCapture,
    pool: ThreadPoolExecutor | None,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Параллельное чтение двух VideoCapture (разные объекты — безопасно)."""
    if pool is None:
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not ok_l or not ok_r:
            return False, None, None
        return True, frame_l, frame_r

    fut_l = pool.submit(cap_l.read)
    fut_r = pool.submit(cap_r.read)
    ok_l, frame_l = fut_l.result()
    ok_r, frame_r = fut_r.result()
    if not ok_l or not ok_r:
        return False, None, None
    return True, frame_l, frame_r


def clamp_roi(
    roi: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
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


def select_object_roi(frame_bgr: np.ndarray, max_display: int) -> tuple[int, int, int, int] | None:
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
    if not tracking_ok:
        lines.append("TRACK LOST — press R")

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

    opencv_threads = configure_threads(args.threads)
    print(
        f"Потоки OpenCV: {opencv_threads} "
        f"(задано --threads {args.threads}), "
        f"workers={args.workers}, async_sgbm={args.async_sgbm}"
    )

    print(f"Загрузка калибровки: {args.calib}")
    calib = load_calibration(args.calib)
    for line in format_quality_report(calibration_quality_warnings(calib)):
        print(line)
    Q = calib["Q"]

    if args.method == "sgbm":
        matcher = build_sgbm(args.min_disparity, args.num_disparities, args.block_size)
    else:
        matcher = build_bm(args.num_disparities, args.block_size)

    prep_pool = ThreadPoolExecutor(max_workers=args.workers)
    # Отдельный пул на 1 поток: matcher.compute не запускаем параллельно самому себе.
    sgbm_pool = ThreadPoolExecutor(max_workers=1) if args.async_sgbm else None
    sgbm_future: Future | None = None

    cap_l = open_video(args.left_video)
    cap_r = open_video(args.right_video)

    ok, frame_l, frame_r = read_pair(cap_l, cap_r, prep_pool)
    if not ok or frame_l is None or frame_r is None:
        sys.exit("Ошибка: не удалось прочитать первый кадр одного из видео.")

    rect_l, rect_r = prepare_pair(frame_l, frame_r, calib, prep_pool)
    rect_l_bgr = cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR)

    print("Выделите объект мышью на левом кадре и нажмите Enter/Space.")
    roi = select_object_roi(rect_l_bgr, args.max_display)
    if roi is None:
        prep_pool.shutdown(wait=False)
        if sgbm_pool is not None:
            sgbm_pool.shutdown(wait=False)
        sys.exit("ROI не выбран — выход.")

    tracker = create_tracker(args.tracker)
    tracker.init(rect_l_bgr, roi)
    tracking_ok = True

    disp_float = compute_disparity(
        rect_l,
        rect_r,
        matcher,
        wls=args.wls,
        wls_lambda=args.wls_lambda,
        wls_sigma=args.wls_sigma,
    )
    dist, disp_val = measure_roi_distance(disp_float, roi, Q=Q)
    history: deque[float] = deque()
    dist_s = smoothed_value(history, dist, args.smooth)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.output,
            fourcc,
            max(cap_l.get(cv2.CAP_PROP_FPS), 1.0),
            (rect_l_bgr.shape[1], rect_l_bgr.shape[0]),
        )

    window = "Track + distance (Space=pause, R=reselect, Q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    paused = False
    frame_idx = 0
    t_prev = time.perf_counter()
    fps = 0.0

    print(
        "Трекинг запущен. "
        f"SGBM каждые {args.sgbm_interval} кадр(ов), трекер={args.tracker}."
    )

    try:
        while True:
            if not paused:
                if frame_idx > 0:
                    ok, frame_l, frame_r = read_pair(cap_l, cap_r, prep_pool)
                    if not ok or frame_l is None or frame_r is None:
                        print("Конец одного из видео.")
                        break
                    if args.max_frames > 0 and frame_idx >= args.max_frames:
                        print("Достигнут --max-frames.")
                        break

                    rect_l, rect_r = prepare_pair(frame_l, frame_r, calib, prep_pool)
                    rect_l_bgr = cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR)

                    tracking_ok, box = tracker.update(rect_l_bgr)
                    if tracking_ok:
                        roi = clamp_roi(box, rect_l.shape[1], rect_l.shape[0])

                    # Забираем готовый асинхронный диспаритет, если есть.
                    if sgbm_future is not None and sgbm_future.done():
                        disp_float = sgbm_future.result()
                        sgbm_future = None

                    need_sgbm = frame_idx % args.sgbm_interval == 0
                    if need_sgbm:
                        if sgbm_pool is not None:
                            # Не ставим новый SGBM, пока предыдущий ещё считается.
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

                    if tracking_ok and roi is not None:
                        dist, disp_val = measure_roi_distance(disp_float, roi, Q=Q)
                        dist_s = smoothed_value(history, dist, args.smooth)
                    else:
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
            if key in (ord("r"), ord("R")):
                print("Повторный выбор объекта...")
                if sgbm_future is not None:
                    sgbm_future.result()
                    sgbm_future = None
                new_roi = select_object_roi(rect_l_bgr, args.max_display)
                if new_roi is not None:
                    roi = new_roi
                    tracker = create_tracker(args.tracker)
                    tracker.init(rect_l_bgr, roi)
                    tracking_ok = True
                    history.clear()
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
        cap_l.release()
        cap_r.release()
        if writer is not None:
            writer.release()
            print(f"Видео сохранено: {Path(args.output).resolve()}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
