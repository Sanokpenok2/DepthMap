"""
Построение карты глубины (диспарности) по стереопаре.

Программа принимает левое и правое изображения стереопары и строит
карту диспарности с помощью алгоритма Semi-Global Block Matching (SGBM)
или Block Matching (BM). Опционально применяется WLS-фильтр
(из opencv-contrib) для сглаживания и заполнения "дыр".

Пример запуска:
    python depth_map.py --left left.png --right right.png --output disparity.png
    python depth_map.py -l left.png -r right.png --method sgbm --wls --show
    python depth_map.py --sbs stereo_sbs.png --calib stereo_calib.npz --show
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from calib_quality import assess_calibration_quality, format_quality_report
from stereo_auto import (
    clamp_sgbm_range,
    estimate_disparity_range_bounds,
    extract_calib_geometry,
    fuse_disparity_maps,
    robust_measure_depth,
    split_near_far_bands,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Построение карты глубины (диспарности) по стереопаре.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-l",
        "--left",
        default=None,
        help="Путь к левому изображению (не нужен при --sbs).",
    )
    p.add_argument(
        "-r",
        "--right",
        default=None,
        help="Путь к правому изображению (не нужен при --sbs).",
    )
    p.add_argument(
        "--sbs",
        default=None,
        help="SBS-фото: левая половина — левая камера, правая — правая.",
    )
    p.add_argument(
        "--swap-lr",
        action="store_true",
        help="Поменять половины SBS местами (если левая камера справа).",
    )
    p.add_argument(
        "-o",
        "--output",
        default="disparity.png",
        help="Файл для сохранения цветной карты глубины.",
    )
    p.add_argument(
        "--method",
        choices=["sgbm", "bm"],
        default="sgbm",
        help="Алгоритм сопоставления блоков.",
    )
    p.add_argument(
        "--num-disparities",
        type=int,
        default=128,
        help="Диапазон диспаритетов (кратен 16). Игнорируется при --auto-disparity.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="Размер блока сопоставления (нечётное число).",
    )
    p.add_argument(
        "--min-disparity",
        type=int,
        default=0,
        help="Минимальный диспаритет. Игнорируется при --auto-disparity.",
    )
    p.add_argument(
        "--auto-disparity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Подбирать min/num_disparities по --calib и --z-near/--z-far "
            "(нужен --calib). Для широкого диапазона дистанций включает "
            "двухполосный SGBM (ближний+дальний)."
        ),
    )
    p.add_argument(
        "--z-near",
        type=float,
        default=5.0,
        help="Ближняя дистанция сцены в метрах (для --auto-disparity).",
    )
    p.add_argument(
        "--z-far",
        type=float,
        default=40.0,
        help="Дальняя дистанция сцены в метрах (для --auto-disparity).",
    )
    p.add_argument(
        "--fuse-disparity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "При --auto-disparity склеивать ближний и дальний проходы SGBM "
            "(лучше покрывает 10–30 м одновременно)."
        ),
    )
    p.add_argument(
        "--wls",
        action="store_true",
        help="Применить WLS-фильтр (требуется opencv-contrib-python).",
    )
    p.add_argument(
        "--wls-lambda",
        type=float,
        default=8000.0,
        help="Параметр lambda WLS-фильтра (сила сглаживания).",
    )
    p.add_argument(
        "--wls-sigma",
        type=float,
        default=1.5,
        help="Параметр sigma WLS-фильтра (чувствительность к границам).",
    )
    p.add_argument(
        "--colormap",
        default="JET",
        help="Название OpenCV colormap (например JET, TURBO, MAGMA, INFERNO).",
    )
    p.add_argument(
        "--save-raw",
        default=None,
        help="Путь для сохранения сырой карты диспаритетов (.npy).",
    )
    p.add_argument(
        "--calib",
        default=None,
        help="Файл стереокалибровки (.npz от calibrate_stereo.py) для ректификации.",
    )
    p.add_argument(
        "--depth",
        default=None,
        help="Путь для сохранения карты глубины в метрах (.npy). Требует --calib.",
    )
    p.add_argument(
        "--point-cloud",
        default=None,
        help="Путь для сохранения облака точек (.ply). Требует --calib.",
    )
    p.add_argument(
        "--measure",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        action="append",
        default=None,
        help="Пиксель (X Y) для измерения расстояния. Можно указывать несколько раз.",
    )
    p.add_argument(
        "--measure-window",
        type=int,
        default=5,
        help="Размер окна (пикс.) для усреднения диспаритета при измерении.",
    )
    p.add_argument(
        "--focal",
        type=float,
        default=None,
        help="Фокусное расстояние в пикселях (для измерения без --calib).",
    )
    p.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="База между камерами (мм) для измерения без --calib.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Показать результат в окне.",
    )
    p.add_argument(
        "--max-display",
        type=int,
        default=1200,
        help="Макс. сторона окна предпросмотра (пикс.). Большие фото ужимаются под экран.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Число потоков OpenCV для SGBM/remap (0 = все ядра, 1 = без параллелизма).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Потоки для параллельной загрузки и ректификации L/R.",
    )
    return p.parse_args()


def configure_opencv_threads(n: int) -> int:
    """Включает внутренний параллелизм OpenCV. Возвращает фактическое число потоков.

    На части сборок Windows `setNumThreads(0)` ошибочно даёт 1 поток,
    поэтому 0 трактуем как os.cpu_count().
    """
    import os

    if n <= 0:
        n = os.cpu_count() or 4
    cv2.setNumThreads(int(n))
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    return int(cv2.getNumThreads())


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение '{path}'.")
    return img


def split_sbs(
    frame: np.ndarray, swap_lr: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Разрезает SBS-кадр пополам по ширине на левую и правую камеры."""
    if frame is None or frame.size == 0:
        raise ValueError("Пустой SBS-кадр.")
    w = frame.shape[1]
    if w < 2:
        raise ValueError("SBS-кадр слишком узкий для разделения.")
    half = w // 2
    left = frame[:, :half]
    right = frame[:, half : half * 2]
    if swap_lr:
        left, right = right, left
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def load_sbs_gray_pair(path: str, swap_lr: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Загружает SBS-изображение и возвращает серые половины L/R."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось прочитать SBS-изображение '{path}'.")
    left, right = split_sbs(img, swap_lr=swap_lr)
    return (
        cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(right, cv2.COLOR_BGR2GRAY),
    )


def load_gray_pair(
    left_path: str,
    right_path: str,
    pool: ThreadPoolExecutor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Параллельная загрузка левого и правого изображений."""
    if pool is None:
        return load_gray(left_path), load_gray(right_path)
    fut_l = pool.submit(load_gray, left_path)
    fut_r = pool.submit(load_gray, right_path)
    return fut_l.result(), fut_r.result()


def build_sgbm(min_disp: int, num_disp: int, block_size: int) -> cv2.StereoSGBM:
    # Рекомендованные параметры штрафов P1/P2 по документации OpenCV.
    channels = 1
    p1 = 8 * channels * block_size ** 2
    p2 = 32 * channels * block_size ** 2
    return cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def build_bm(num_disp: int, block_size: int) -> cv2.StereoBM:
    matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    matcher.setPreFilterCap(31)
    matcher.setUniquenessRatio(10)
    matcher.setSpeckleWindowSize(100)
    matcher.setSpeckleRange(2)
    return matcher


def apply_wls(
    left_matcher,
    left_disp: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    lam: float,
    sigma: float,
) -> np.ndarray:
    try:
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    except AttributeError:
        print(
            "Предупреждение: модуль cv2.ximgproc недоступен. "
            "Установите 'opencv-contrib-python'. WLS-фильтр пропущен.",
            file=sys.stderr,
        )
        return left_disp

    right_disp = right_matcher.compute(right_img, left_img)
    wls.setLambda(lam)
    wls.setSigmaColor(sigma)
    filtered = wls.filter(left_disp, left_img, disparity_map_right=right_disp)
    return filtered


def normalize_disparity(disp: np.ndarray, min_disp: int, num_disp: int) -> np.ndarray:
    """Преобразует карту диспаритетов (в формате fixed-point *16) в 8-бит."""
    disp_float = disp.astype(np.float32) / 16.0
    disp_float[disp_float < min_disp] = min_disp
    vis = (disp_float - min_disp) / max(num_disp, 1)
    vis = np.clip(vis, 0.0, 1.0)
    return (vis * 255).astype(np.uint8)


def load_calibration(path: str) -> dict:
    try:
        data = np.load(path, allow_pickle=True)
    except OSError:
        sys.exit(f"Ошибка: не удалось прочитать файл калибровки '{path}'.")
    required = ["map1_l", "map2_l", "map1_r", "map2_r", "Q"]
    missing = [k for k in required if k not in data.files]
    if missing:
        sys.exit(f"Ошибка: в файле калибровки нет полей: {', '.join(missing)}.")
    return {k: data[k] for k in data.files}


def calibration_quality_warnings(calib: dict) -> list[str]:
    """Возвращает предупреждения о качестве загруженной калибровки."""
    if "quality_warnings" in calib:
        stored = calib["quality_warnings"]
        if isinstance(stored, np.ndarray):
            return [str(w) for w in stored.tolist() if str(w)]
        return [str(stored)]

    if "mtx_l" not in calib or "mtx_r" not in calib or "T" not in calib:
        return ["В файле калибровки нет данных для проверки качества."]

    image_size = tuple(int(v) for v in calib["image_size"])
    model = "pinhole"
    if "model" in calib:
        model = str(np.asarray(calib["model"]).ravel()[0])

    baseline_mm = float(np.linalg.norm(calib["T"]))
    if "baseline_mm" in calib:
        baseline_mm = float(np.asarray(calib["baseline_mm"]).ravel()[0])

    alpha = 1.0
    if "alpha" in calib:
        alpha = float(np.asarray(calib["alpha"]).ravel()[0])

    roi1 = calib["roi1"] if "roi1" in calib else None
    roi2 = calib["roi2"] if "roi2" in calib else None

    return assess_calibration_quality(
        model=model,
        rms_l=float("nan"),
        rms_r=float("nan"),
        rms_stereo=float("nan"),
        mtx_l=calib["mtx_l"],
        mtx_r=calib["mtx_r"],
        baseline_mm=baseline_mm,
        map1_l=calib["map1_l"],
        map2_l=calib["map2_l"],
        map1_r=calib["map1_r"],
        map2_r=calib["map2_r"],
        image_size=image_size,
        alpha=alpha,
        roi1=roi1,
        roi2=roi2,
        dist_l=calib.get("dist_l"),
        dist_r=calib.get("dist_r"),
    )


def rectify_pair(
    left: np.ndarray,
    right: np.ndarray,
    calib: dict,
    pool: ThreadPoolExecutor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Применяет карты ремаппинга из калибровки для выравнивания стереопары."""
    if pool is None:
        rect_l = cv2.remap(left, calib["map1_l"], calib["map2_l"], cv2.INTER_LINEAR)
        rect_r = cv2.remap(right, calib["map1_r"], calib["map2_r"], cv2.INTER_LINEAR)
        return rect_l, rect_r

    fut_l = pool.submit(
        cv2.remap, left, calib["map1_l"], calib["map2_l"], cv2.INTER_LINEAR
    )
    fut_r = pool.submit(
        cv2.remap, right, calib["map1_r"], calib["map2_r"], cv2.INTER_LINEAR
    )
    return fut_l.result(), fut_r.result()


def format_timings(timings: dict[str, float]) -> list[str]:
    """Строки журнала с разбивкой времени по этапам."""
    order = [
        ("load", "загрузка"),
        ("rectify", "ректификация"),
        ("sgbm", "сопоставление"),
        ("wls", "WLS"),
        ("visualize", "визуализация"),
        ("total", "всего"),
    ]
    lines = ["Время выполнения:"]
    for key, title in order:
        if key in timings:
            lines.append(f"  {title}: {timings[key] * 1000:.1f} ms ({timings[key]:.3f} s)")
    if "opencv_threads" in timings:
        lines.append(f"  потоки OpenCV: {int(timings['opencv_threads'])}")
    if "workers" in timings:
        lines.append(f"  workers L/R: {int(timings['workers'])}")
    return lines


def save_point_cloud(
    path: str, disp_float: np.ndarray, Q: np.ndarray, color_img: np.ndarray
) -> None:
    """Строит и сохраняет облако точек в формате PLY по матрице Q."""
    points_3d = cv2.reprojectImageTo3D(disp_float, Q)
    # Валидны точки с положительным диспаритетом и конечными координатами.
    mask = (disp_float > disp_float.min()) & np.isfinite(points_3d).all(axis=2)
    mask &= np.abs(points_3d[:, :, 2]) < 1e4  # отбрасываем "бесконечно далёкие"

    pts = points_3d[mask]
    if color_img.ndim == 2:
        colors = cv2.cvtColor(color_img, cv2.COLOR_GRAY2RGB)[mask]
    else:
        colors = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)[mask]

    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts, colors):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


def measure_distance(
    disp_float: np.ndarray,
    x: int,
    y: int,
    window: int = 5,
    Q: np.ndarray | None = None,
    focal: float | None = None,
    baseline: float | None = None,
) -> tuple[float | None, float | None]:
    """Возвращает (расстояние, медианный диспаритет) в точке (x, y).

    Робастная медиана по окну с отсечением выбросов (IQR). Расстояние —
    через матрицу Q или depth = focal * baseline / disparity.
    """
    return robust_measure_depth(
        disp_float, x, y, window, Q, focal, baseline
    )


def measure_roi_distance(
    disp_float: np.ndarray,
    roi: tuple[int, int, int, int],
    Q: np.ndarray | None = None,
    focal: float | None = None,
    baseline: float | None = None,
    *,
    min_valid_fraction: float = 0.05,
) -> tuple[float | None, float | None]:
    """Расстояние по медиане диспаритета внутри ROI (x, y, w, h).

    Возвращает (расстояние, медианный диспаритет) или (None, None), если
    валидных пикселей слишком мало.
    """
    x, y, rw, rh = (int(v) for v in roi)
    h, w = disp_float.shape
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + max(rw, 1))
    y1 = min(h, y + max(rh, 1))
    if x1 <= x0 or y1 <= y0:
        return None, None

    patch = disp_float[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size < max(1, int(patch.size * min_valid_fraction)):
        return None, None

    disp = float(np.median(valid))
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    if Q is not None:
        vec = np.array([[cx], [cy], [disp], [1.0]], dtype=np.float64)
        xyzw = Q @ vec
        wv = xyzw[3, 0]
        if abs(wv) < 1e-9:
            return None, disp
        z = float(xyzw[2, 0] / wv)
        if not np.isfinite(z) or z <= 0:
            return None, disp
        return z, disp
    if focal is not None and baseline is not None and disp > 0:
        return float(focal * baseline / disp), disp
    return None, disp


def display_scale(shape: tuple[int, int], max_side: int) -> float:
    """Коэффициент масштаба, чтобы большая сторона изображения влезла в max_side."""
    h, w = shape[:2]
    longest = max(h, w)
    if max_side <= 0 or longest <= max_side:
        return 1.0
    return max_side / float(longest)


def fit_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return img
    w = max(1, int(round(img.shape[1] * scale)))
    h = max(1, int(round(img.shape[0] * scale)))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def get_colormap(name: str) -> int:
    key = f"COLORMAP_{name.upper()}"
    cmap = getattr(cv2, key, None)
    if cmap is None:
        print(f"Предупреждение: colormap '{name}' не найден, используется JET.", file=sys.stderr)
        return cv2.COLORMAP_JET
    return cmap


@dataclass
class StereoProcessResult:
    disparity_color: np.ndarray
    left_gray: np.ndarray
    right_gray: np.ndarray
    disparity_float: np.ndarray
    rectified: bool
    log: list[str]
    timings: dict[str, float] = field(default_factory=dict)


def _run_matcher(
    left: np.ndarray,
    right: np.ndarray,
    *,
    method: str,
    min_disparity: int,
    num_disparities: int,
    block_size: int,
    wls: bool,
    wls_lambda: float,
    wls_sigma: float,
) -> tuple[np.ndarray, object]:
    """Возвращает (disp_raw fixed-point*16, matcher)."""
    width = int(left.shape[1])
    min_disparity, num_disparities = clamp_sgbm_range(
        min_disparity, num_disparities, width, max_num=512
    )
    if num_disparities < 16 or min_disparity + num_disparities >= width:
        raise ValueError(
            f"Некорректный диапазон SGBM: min={min_disparity}, "
            f"num={num_disparities}, width={width}."
        )
    if method == "sgbm":
        matcher = build_sgbm(min_disparity, num_disparities, block_size)
    else:
        matcher = build_bm(num_disparities, block_size)
        if min_disparity != 0:
            # StereoBM в OpenCV не поддерживает произвольный minDisparity так же гибко.
            pass
    disp = matcher.compute(left, right)
    if wls:
        disp = apply_wls(matcher, disp, left, right, wls_lambda, wls_sigma)
    return disp, matcher


def compute_stereo_disparity(
    left_path: str | None = None,
    right_path: str | None = None,
    *,
    left_gray: np.ndarray | None = None,
    right_gray: np.ndarray | None = None,
    method: str = "sgbm",
    num_disparities: int = 128,
    block_size: int = 5,
    min_disparity: int = 0,
    wls: bool = False,
    wls_lambda: float = 8000.0,
    wls_sigma: float = 1.5,
    colormap: str = "JET",
    calib_path: str | None = None,
    threads: int = 0,
    workers: int = 2,
    auto_disparity: bool = False,
    z_near_m: float = 8.0,
    z_far_m: float = 40.0,
    fuse_disparity: bool = True,
) -> StereoProcessResult:
    """Строит карту диспаритета по паре изображений (с параллелизмом и таймингами).

    Источник: либо пути left_path/right_path, либо готовые left_gray/right_gray
    (например после split_sbs / load_sbs_gray_pair).

    При auto_disparity + calib подбирает диапазон под z_near_m…z_far_m.
    Если диапазон широкий — по умолчанию два прохода SGBM (ближний/дальний)
    и склейка, иначе дальние объекты пропадают при большом num_disparities.
    """
    log: list[str] = []
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    if num_disparities % 16 != 0:
        raise ValueError("--num-disparities должен быть кратен 16.")
    if block_size % 2 == 0:
        raise ValueError("--block-size должен быть нечётным.")
    if workers < 1:
        raise ValueError("--workers должен быть >= 1.")
    has_arrays = left_gray is not None and right_gray is not None
    has_paths = left_path is not None and right_path is not None
    if has_arrays == has_paths:
        raise ValueError(
            "Укажите либо пути left/right, либо массивы left_gray/right_gray."
        )
    if auto_disparity and not calib_path:
        log.append(
            "Предупреждение: --auto-disparity без --calib — "
            "используются --min-disparity/--num-disparities."
        )
        auto_disparity = False

    opencv_threads = configure_opencv_threads(threads)
    timings["opencv_threads"] = float(opencv_threads)
    timings["workers"] = float(workers)
    log.append(
        f"Параллелизм: OpenCV threads={opencv_threads}, L/R workers={workers}."
    )

    pool = ThreadPoolExecutor(max_workers=workers)
    calib = None
    try:
        t0 = time.perf_counter()
        if has_arrays:
            left = np.ascontiguousarray(left_gray)
            right = np.ascontiguousarray(right_gray)
            if left.ndim == 3:
                left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            if right.ndim == 3:
                right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        else:
            left, right = load_gray_pair(left_path, right_path, pool)
        timings["load"] = time.perf_counter() - t0

        if left.shape != right.shape:
            raise ValueError(
                f"Размеры изображений различаются ({left.shape} и {right.shape}). "
                "Стереопара должна быть выровнена или используйте калибровку."
            )

        rectified = False
        if calib_path:
            log.append(f"Загрузка калибровки и ректификация: {calib_path}")
            calib = load_calibration(calib_path)
            warnings = calibration_quality_warnings(calib)
            if warnings:
                log.extend(format_quality_report(warnings))
            t0 = time.perf_counter()
            left, right = rectify_pair(left, right, calib, pool)
            timings["rectify"] = time.perf_counter() - t0
            rectified = True

        use_fuse = False
        vis_min, vis_num = min_disparity, num_disparities
        if auto_disparity and calib is not None:
            width = int(left.shape[1])
            single_min, single_num, range_log = estimate_disparity_range_bounds(
                calib,
                z_near_m,
                z_far_m,
                image_width=width,
            )
            log.append(range_log)
            # Широкий динамический диапазон (напр. 10–30 м) → два прохода.
            focal, baseline = extract_calib_geometry(calib)
            d_near = focal * baseline / max(z_near_m * 1000.0, 1.0)
            d_far = focal * baseline / max(z_far_m * 1000.0, 1.0)
            wide = (d_near / max(d_far, 1.0) >= 2.2) or single_num >= 256
            use_fuse = bool(fuse_disparity and method == "sgbm" and wide)
            if use_fuse:
                (far_min, far_num), (near_min, near_num), fuse_log = split_near_far_bands(
                    calib, z_near_m, z_far_m, image_width=width
                )
                log.append(fuse_log)
                vis_min, vis_num = 0, max(far_min + far_num, near_min + near_num)
                vis_num = int(np.ceil(vis_num / 16) * 16)
            else:
                min_disparity, num_disparities = single_min, single_num
                vis_min, vis_num = min_disparity, num_disparities
                use_fuse = False

        t0 = time.perf_counter()
        if use_fuse:
            log.append(
                f"Вычисление диспаритета (двухполосный {method.upper()})..."
            )
            disp_far_raw, matcher = _run_matcher(
                left,
                right,
                method=method,
                min_disparity=far_min,
                num_disparities=far_num,
                block_size=block_size,
                wls=wls,
                wls_lambda=wls_lambda,
                wls_sigma=wls_sigma,
            )
            disp_near_raw, _ = _run_matcher(
                left,
                right,
                method=method,
                min_disparity=near_min,
                num_disparities=near_num,
                block_size=block_size,
                wls=wls,
                wls_lambda=wls_lambda,
                wls_sigma=wls_sigma,
            )
            far_f = disp_far_raw.astype(np.float32) / 16.0
            near_f = disp_near_raw.astype(np.float32) / 16.0
            split_d = 0.5 * (
                float(far_min + far_num) * 0.65 + float(near_min) * 0.35
            )
            # Порог склейки: диспаритет на ~z_split (ближе к ближней зоне).
            if calib is not None:
                focal, baseline = extract_calib_geometry(calib)
                z_split = min(
                    max(0.5 * (z_near_m + z_far_m), z_near_m * 1.6),
                    z_far_m * 0.7,
                )
                split_d = focal * baseline / (z_split * 1000.0)
            disp_float = fuse_disparity_maps(far_f, near_f, split_disp=split_d)
            # Для визуализации / WLS-совместимости собираем fixed-point из float.
            disp = (disp_float * 16.0).astype(np.int16)
            min_disparity, num_disparities = vis_min, vis_num
        else:
            log.append(
                f"Вычисление диспаритета методом {method.upper()} "
                f"(min={min_disparity}, num={num_disparities})..."
            )
            disp, matcher = _run_matcher(
                left,
                right,
                method=method,
                min_disparity=min_disparity,
                num_disparities=num_disparities,
                block_size=block_size,
                wls=wls,
                wls_lambda=wls_lambda,
                wls_sigma=wls_sigma,
            )
            disp_float = disp.astype(np.float32) / 16.0
        timings["sgbm"] = time.perf_counter() - t0
        if wls:
            timings["wls"] = timings.get("wls", 0.0)

        t0 = time.perf_counter()
        disp_vis = normalize_disparity(disp, min_disparity, num_disparities)
        disp_color = cv2.applyColorMap(disp_vis, get_colormap(colormap))
        timings["visualize"] = time.perf_counter() - t0
    finally:
        pool.shutdown(wait=False)

    timings["total"] = time.perf_counter() - t_all
    log.extend(format_timings(timings))

    valid = disp_float[disp_float > 0]
    if valid.size:
        log.append(
            f"Диспаритет: мин {valid.min():.1f}, макс {valid.max():.1f}, "
            f"медиана {np.median(valid):.1f} px"
        )
    else:
        log.append("Предупреждение: не найдено валидных значений диспаритета.")

    return StereoProcessResult(
        disparity_color=disp_color,
        left_gray=left,
        right_gray=right,
        disparity_float=disp_float,
        rectified=rectified,
        log=log,
        timings=timings,
    )


def main() -> None:
    args = parse_args()

    if args.num_disparities % 16 != 0:
        sys.exit("Ошибка: --num-disparities должен быть кратен 16.")
    if args.block_size % 2 == 0:
        sys.exit("Ошибка: --block-size должен быть нечётным.")
    if args.workers < 1:
        sys.exit("Ошибка: --workers должен быть >= 1.")

    if (args.depth or args.point_cloud) and not args.calib:
        sys.exit("Ошибка: --depth и --point-cloud требуют указания --calib.")

    use_sbs = args.sbs is not None
    use_pair = args.left is not None or args.right is not None
    if use_sbs and use_pair:
        sys.exit("Ошибка: укажите либо --sbs, либо пару --left/--right, не оба варианта.")
    if not use_sbs and (not args.left or not args.right):
        sys.exit("Ошибка: укажите --left и --right либо одно SBS-фото через --sbs.")

    if args.z_near <= 0 or args.z_far <= 0 or args.z_near >= args.z_far:
        sys.exit("Ошибка: нужно 0 < --z-near < --z-far (дистанции в метрах).")

    common_kwargs = dict(
        method=args.method,
        num_disparities=args.num_disparities,
        block_size=args.block_size,
        min_disparity=args.min_disparity,
        wls=args.wls,
        wls_lambda=args.wls_lambda,
        wls_sigma=args.wls_sigma,
        colormap=args.colormap,
        calib_path=args.calib,
        threads=args.threads,
        workers=args.workers,
        auto_disparity=args.auto_disparity,
        z_near_m=args.z_near,
        z_far_m=args.z_far,
        fuse_disparity=args.fuse_disparity,
    )

    try:
        if use_sbs:
            left_g, right_g = load_sbs_gray_pair(args.sbs, swap_lr=args.swap_lr)
            result = compute_stereo_disparity(
                left_gray=left_g,
                right_gray=right_g,
                **common_kwargs,
            )
        else:
            result = compute_stereo_disparity(
                args.left,
                args.right,
                **common_kwargs,
            )
    except ValueError as exc:
        sys.exit(f"Ошибка: {exc}")

    for line in result.log:
        print(line)

    left = result.left_gray
    disp_color = result.disparity_color
    disp_float = result.disparity_float

    out_path = Path(args.output)
    cv2.imwrite(str(out_path), disp_color)
    print(f"Карта глубины сохранена: {out_path.resolve()}")

    if args.save_raw:
        np.save(args.save_raw, disp_float)
        print(f"Сырая карта диспаритетов сохранена: {Path(args.save_raw).resolve()}")

    calib = load_calibration(args.calib) if args.calib else None
    if calib is not None and (args.depth or args.point_cloud):
        points_3d = cv2.reprojectImageTo3D(disp_float, calib["Q"])
        if args.depth:
            depth = points_3d[:, :, 2].copy()
            depth[disp_float <= disp_float.min()] = 0.0
            depth[~np.isfinite(depth)] = 0.0
            np.save(args.depth, depth)
            valid = depth[(depth > 0) & (depth < 1e4)]
            if valid.size:
                print(
                    f"Карта глубины сохранена: {Path(args.depth).resolve()} "
                    f"(диапазон {valid.min():.1f}..{valid.max():.1f} в ед. --square-size)"
                )
            else:
                print(f"Карта глубины сохранена: {Path(args.depth).resolve()}")
        if args.point_cloud:
            save_point_cloud(args.point_cloud, disp_float, calib["Q"], left)
            print(f"Облако точек сохранено: {Path(args.point_cloud).resolve()}")

    # Источник данных для перевода диспаритета в расстояние.
    Q = calib["Q"] if calib is not None else None
    can_measure = Q is not None or (args.focal is not None and args.baseline is not None)
    unit = "ед. (square-size)" if Q is not None else "мм"

    if args.measure:
        if not can_measure:
            print(
                "Предупреждение: для измерения расстояния нужен --calib "
                "либо пара --focal и --baseline. Измерение пропущено.",
                file=sys.stderr,
            )
        else:
            print("Измеренные расстояния:")
            for x, y in args.measure:
                dist, disp_val = measure_distance(
                    disp_float, x, y, args.measure_window, Q, args.focal, args.baseline
                )
                if dist is None:
                    print(f"  ({x}, {y}): нет данных о диспаритете в этой точке.")
                    continue
                print(f"  ({x}, {y}): {dist:.1f} {unit} (диспаритет {disp_val:.2f} px)")
                cv2.drawMarker(
                    disp_color, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 16, 2
                )
                cv2.putText(
                    disp_color,
                    f"{dist:.0f}",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(out_path), disp_color)

    if args.show:
        scale = display_scale(disp_color.shape, args.max_display)
        disp_display = fit_for_display(disp_color, scale)
        left_display = fit_for_display(left, scale)
        if scale < 1.0:
            print(
                f"Предпросмотр ужат до {disp_display.shape[1]}x{disp_display.shape[0]} "
                f"(масштаб {scale:.2f}); сохранённые файлы — в полном разрешении."
            )

        if can_measure:
            print("Кликните по карте диспаритета, чтобы измерить расстояние.")

            def on_click(event, x, y, flags, param):
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                # Координаты окна пересчитываем в полное разрешение.
                fx_img = int(round(x / scale))
                fy_img = int(round(y / scale))
                dist, disp_val = measure_distance(
                    disp_float, fx_img, fy_img, args.measure_window, Q, args.focal, args.baseline
                )
                if dist is None:
                    print(f"  ({fx_img}, {fy_img}): нет данных о диспаритете.")
                    return
                print(
                    f"  ({fx_img}, {fy_img}): {dist:.1f} {unit} "
                    f"(диспаритет {disp_val:.2f} px)"
                )
                annotated = disp_display.copy()
                cv2.drawMarker(annotated, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
                cv2.putText(
                    annotated,
                    f"{dist:.0f} {unit}",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("Disparity", annotated)

            cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Disparity", on_click)
        else:
            cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)

        cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
        cv2.imshow("Left", left_display)
        cv2.imshow("Disparity", disp_display)
        cv2.resizeWindow("Left", left_display.shape[1], left_display.shape[0])
        cv2.resizeWindow("Disparity", disp_display.shape[1], disp_display.shape[0])
        print("Нажмите любую клавишу в окне для выхода...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
