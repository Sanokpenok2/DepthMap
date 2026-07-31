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


def build_sgbm(
    min_disp: int,
    num_disp: int,
    block_size: int,
    *,
    uniqueness_ratio: int = 10,
    speckle_window_size: int = 100,
    speckle_range: int = 2,
    mode: int | None = None,
) -> cv2.StereoSGBM:
    # Рекомендованные параметры штрафов P1/P2 по документации OpenCV.
    channels = 1
    p1 = 8 * channels * block_size ** 2
    p2 = 32 * channels * block_size ** 2
    if mode is None:
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    return cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=int(uniqueness_ratio),
        speckleWindowSize=int(speckle_window_size),
        speckleRange=int(speckle_range),
        preFilterCap=63,
        mode=mode,
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


@dataclass
class DisparityDebugInfo:
    """Пиксели ROI, по которым считается диспаритет/дистанция."""

    inset_roi: tuple[int, int, int, int]
    used_ys: np.ndarray  # int32, координаты в полном кадре
    used_xs: np.ndarray
    used_disp: np.ndarray  # float32
    selected_disp: float | None
    n_patch: int
    n_valid: int
    n_used: int
    surface: str
    refine: str = ""  # "", "cluster-far", "epipolar-ncc", ...


def _pick_disparity_surface(valid: np.ndarray, surface: str) -> tuple[float, str]:
    """Выбор диспаритета; при бимодальности (машина далеко / забор близко) —
    для far/median берём дальний (меньший d) кластер."""
    valid = np.asarray(valid, dtype=np.float32).reshape(-1)
    if valid.size == 0:
        return 0.0, surface
    if valid.size < 12:
        if surface == "far":
            return float(np.percentile(valid, 40)), surface
        if surface == "near":
            return float(np.percentile(valid, 60)), surface
        return float(np.median(valid)), surface

    med = float(np.median(valid))
    spread = float(valid.max()) - float(valid.min())
    multimodal = spread > max(1.5, 0.25 * max(med, 1.0))

    if not multimodal:
        if surface == "far":
            return float(np.percentile(valid, 40)), surface
        if surface == "near":
            return float(np.percentile(valid, 60)), surface
        return med, surface

    n_bins = int(np.clip(valid.size // 3, 16, 64))
    hist, edges = np.histogram(valid, bins=n_bins)
    hist = hist.astype(np.float32)
    if hist.size >= 3:
        hist = np.convolve(hist, np.array([1.0, 2.0, 1.0], np.float32) / 4.0, mode="same")
    peaks: list[tuple[float, float]] = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] > 0:
            peaks.append((float(hist[i]), float(0.5 * (edges[i] + edges[i + 1]))))
    if len(peaks) < 2:
        if surface == "near":
            return float(np.percentile(valid, 70)), surface
        # Бимодальный разброс без явных пиков — смещаемся к дальнему плану.
        return float(np.percentile(valid, 30 if surface == "far" else 35)), (
            "cluster-far" if surface != "near" else surface
        )

    peaks.sort(key=lambda t: -t[0])
    top2 = peaks[:2]
    centers = sorted(c for _, c in top2)
    far_c, near_c = float(centers[0]), float(centers[-1])
    if near_c - far_c < max(1.5, 0.12 * near_c):
        if surface == "far":
            return float(np.percentile(valid, 40)), surface
        if surface == "near":
            return float(np.percentile(valid, 60)), surface
        return med, surface

    dist_far = np.abs(valid - far_c)
    dist_near = np.abs(valid - near_c)
    far_samples = valid[dist_far <= dist_near]
    near_samples = valid[dist_near < dist_far]
    min_keep = max(3, int(0.12 * valid.size))

    if surface == "near":
        pool = near_samples if near_samples.size >= min_keep else valid
        return float(np.median(pool)), "cluster-near"

    # far и median: при двух кластерах предпочитаем дальний (анти-забор).
    if far_samples.size >= min_keep:
        if surface == "far":
            return float(np.percentile(far_samples, 40)), "cluster-far"
        return float(np.median(far_samples)), "cluster-far"
    # Дальний кластер слишком мал — осторожный перцентиль по всем.
    return float(np.percentile(valid, 30)), "cluster-far-weak"


def _match_image_pair(
    left_gray: np.ndarray, right_gray: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Интенсивность или |grad|, если в ROI почти нет текстуры (типично для ТПВ)."""
    L = left_gray
    R = right_gray
    if L.dtype != np.uint8:
        L = np.clip(L, 0, 255).astype(np.uint8)
    if R.dtype != np.uint8:
        R = np.clip(R, 0, 255).astype(np.uint8)
    return L, R


def _to_grad8(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # Нормализация по кадру — стабильный NCC на слабом тепловом контрасте.
    p1, p99 = np.percentile(mag, [1.0, 99.0])
    if p99 <= p1 + 1e-3:
        return np.zeros_like(gray, dtype=np.uint8)
    out = np.clip((mag - p1) * (255.0 / (p99 - p1)), 0, 255).astype(np.uint8)
    return out


def epipolar_ncc_disparity(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    cx: int,
    cy: int,
    *,
    d_min: float,
    d_max: float,
    templ_w: int = 25,
    templ_h: int = 17,
    dy_search: int = 2,
    prefer_far: bool = True,
) -> tuple[float | None, float]:
    """Диспаритет по NCC вдоль эпиполяра (и ±dy при небольшой ошибке ректификации).

    Возвращает (disparity, score). Нужен, когда SGBM цепляется за повторяющийся
    забор/текстуру вместо объекта в ROI.

    При нескольких пиках с близким score и prefer_far=True выбирается меньший
    диспаритет (дальше) — типичный случай «машина vs забор».
    """
    if left_gray.ndim == 3:
        left_gray = cv2.cvtColor(left_gray, cv2.COLOR_BGR2GRAY)
    if right_gray.ndim == 3:
        right_gray = cv2.cvtColor(right_gray, cv2.COLOR_BGR2GRAY)
    left_gray, right_gray = _match_image_pair(left_gray, right_gray)
    lh, lw = left_gray.shape[:2]
    rh, rw = right_gray.shape[:2]
    tw = max(9, int(templ_w) | 1)
    th = max(9, int(templ_h) | 1)
    hx, hy = tw // 2, th // 2
    if not (hx < cx < lw - hx and hy < cy < lh - hy):
        return None, -1.0

    d0 = int(np.floor(max(0.0, float(d_min))))
    d1 = int(np.ceil(max(float(d_max), d0 + 1)))
    if d1 <= d0:
        return None, -1.0

    # Кандидаты: интенсивность и градиент (тепловые пятна часто почти flat).
    left_g = _to_grad8(left_gray)
    right_g = _to_grad8(right_gray)
    image_pairs = [(left_gray, right_gray), (left_g, right_g)]

    candidates: list[tuple[float, float, float]] = []  # (score, uniqueness, disp)

    for Limg, Rimg in image_pairs:
        templ = Limg[cy - hy : cy + hy + 1, cx - hx : cx + hx + 1]
        if templ.shape[0] != th or templ.shape[1] != tw:
            continue
        if float(np.std(templ.astype(np.float32))) < 2.0:
            continue

        for dy in range(-int(dy_search), int(dy_search) + 1):
            y = cy + dy
            if y - hy < 0 or y + hy >= rh:
                continue
            # Полоса справа: возможные центры шаблона x = cx - d, d∈[d0,d1]
            x_right_min = cx - d1
            x_right_max = cx - d0
            x0 = x_right_min - hx
            x1 = x_right_max + hx + 1
            if x1 - x0 < tw:
                continue
            x0c = max(0, x0)
            x1c = min(rw, x1)
            if x1c - x0c < tw:
                continue
            strip = Rimg[y - hy : y + hy + 1, x0c:x1c]
            if strip.shape[0] != th or strip.shape[1] < tw:
                continue
            res = cv2.matchTemplate(strip, templ, cv2.TM_CCOEFF_NORMED)
            if res.size == 0:
                continue
            # Несколько локальных пиков — для анти-забора важна уникальность.
            res_f = res.astype(np.float32).reshape(-1)
            # top-k по всей полосе
            k = min(8, int(res_f.size))
            if k < 1:
                continue
            flat_idx = np.argpartition(res_f, -k)[-k:]
            flat_idx = flat_idx[np.argsort(res_f[flat_idx])[::-1]]
            peaks: list[tuple[float, float]] = []
            cols = int(res.shape[1])
            for fi in flat_idx:
                score = float(res_f[fi])
                if score < 0.15:
                    break
                px = int(fi) % cols
                rx = x0c + px + hx
                disp = float(cx - rx)
                if disp < d0 - 0.5 or disp > d1 + 0.5:
                    continue
                # Подавляем соседей ±2 px уже выбранных пиков.
                if any(abs(disp - pd) < 2.0 for _, pd in peaks):
                    continue
                peaks.append((score, disp))
                if len(peaks) >= 4:
                    break
            if not peaks:
                continue
            best_s, best_d = peaks[0]
            second = peaks[1][0] if len(peaks) > 1 else -1.0
            uniq = best_s - second if second >= 0 else best_s
            candidates.append((best_s, uniq, best_d))
            # Если второй пик тоже сильный и заметно дальше — держим оба.
            for s2, d2 in peaks[1:]:
                if s2 >= best_s - 0.08 and d2 + 3.0 < best_d:
                    candidates.append((s2, s2 - best_s, d2))

    if not candidates:
        return None, -1.0

    # Сначала по score, при близких score — меньший диспаритет (дальше).
    def _rank(c: tuple[float, float, float]) -> tuple:
        score, uniq, disp = c
        far_bonus = -disp if prefer_far else 0.0
        return (score + 0.15 * uniq + 0.002 * far_bonus, -disp if prefer_far else disp)

    best = max(candidates, key=_rank)
    # Если есть почти такой же score, но заметно меньший d — берём дальний.
    if prefer_far:
        top_score = best[0]
        farther = [
            c
            for c in candidates
            if c[0] >= top_score - 0.06 and c[2] + 2.5 < best[2]
        ]
        if farther:
            best = min(farther, key=lambda c: c[2])
            # сохраняем score дальнего пика
            return float(best[2]), float(best[0])

    return float(best[2]), float(best[0])


def measure_roi_distance(
    disp_float: np.ndarray,
    roi: tuple[int, int, int, int],
    Q: np.ndarray | None = None,
    focal: float | None = None,
    baseline: float | None = None,
    *,
    min_valid_fraction: float = 0.10,
    inset_fraction: float = 0.30,
    robust: bool = True,
    prefer_near_surface: bool = False,
    surface: str = "far",
    max_disparity: float | None = None,
    min_disparity: float = 0.4,
    max_distance_mm: float | None = None,
    collect_debug: bool = False,
    left_gray: np.ndarray | None = None,
    right_gray: np.ndarray | None = None,
    epipolar_ncc: bool = True,
    ncc_min_score: float = 0.28,
) -> (
    tuple[float | None, float | None]
    | tuple[float | None, float | None, DisparityDebugInfo | None]
):
    """Расстояние по диспаритету внутри ROI (x, y, w, h).

    surface:
      - \"far\"  — дальняя поверхность / дальний кластер при бимодальности
      - \"near\" — ближняя (больший диспаритет)
      - \"median\" — медиана; при двух пиках (забор/машина) тоже предпочитает дальний

    max_disparity: отбросить пиксели с d больше порога (ближе z_near).
    max_distance_mm: отбросить/ограничить Z больше ожидаемого z_far.
    left_gray/right_gray + epipolar_ncc: уточнение NCC вдоль эпиполяра
    (защита от ложных матчей SGBM на повторяющейся текстуре).
    collect_debug: если True, третьим элементом вернуть DisparityDebugInfo.
    """
    if prefer_near_surface:
        surface = "near"
    surface = (surface or "median").lower().strip()
    if surface not in ("far", "near", "median"):
        surface = "median"

    def _pack(
        dist: float | None,
        disp: float | None,
        dbg: DisparityDebugInfo | None = None,
    ):
        if collect_debug:
            return dist, disp, dbg
        return dist, disp

    x, y, rw, rh = (int(v) for v in roi)
    h, w = disp_float.shape
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + max(rw, 1))
    y1 = min(h, y + max(rh, 1))
    if x1 <= x0 or y1 <= y0:
        return _pack(None, None, None)

    # Сужаем ROI к центру; низ режем сильнее (дорога перед машиной).
    inset = float(np.clip(inset_fraction, 0.0, 0.45))
    if inset > 0:
        bw, bh = x1 - x0, y1 - y0
        dx = int(round(bw * inset))
        dy = int(round(bh * inset))
        dy_top = dy
        dy_bot = min(bh - 1, dy + max(1, dy // 2)) if bh > 4 else dy
        x0 = min(x1 - 1, x0 + dx)
        x1 = max(x0 + 1, x1 - dx)
        y0 = min(y1 - 1, y0 + dy_top)
        y1 = max(y0 + 1, y1 - dy_bot)

    inset_roi = (x0, y0, x1 - x0, y1 - y0)
    patch = disp_float[y0:y1, x0:x1]
    finite = np.isfinite(patch) & (patch > 0)
    if max_disparity is not None and max_disparity > 0:
        finite &= patch <= float(max_disparity)
    if min_disparity > 0:
        finite &= patch >= float(min_disparity)

    valid = patch[finite]
    n_patch = int(patch.size)
    n_valid = int(valid.size)
    refine_tag = ""
    if valid.size < max(1, int(patch.size * min_valid_fraction)):
        # Мало валидного SGBM — всё равно пробуем NCC по центру ROI.
        disp = None
        keep_mask = finite
    else:
        keep_mask = finite.copy()
        if robust and valid.size >= 8:
            q1, q3 = np.percentile(valid, [25, 75])
            iqr = float(q3 - q1)
            if iqr > 1e-6:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                # Для far/median сильнее режем «слишком близкие» выбросы (забор).
                if surface in ("far", "median"):
                    hi = min(hi, float(np.percentile(valid, 75)))
                in_iqr = finite & (patch >= lo) & (patch <= hi)
                if int(in_iqr.sum()) >= max(3, int(0.20 * valid.size)):
                    keep_mask = in_iqr
                    valid = patch[keep_mask]

        disp, refine_tag = _pick_disparity_surface(valid, surface)

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    # NCC вдоль эпиполяра: часто находит машину, когда SGBM залип на заборе.
    ncc_disp = None
    ncc_score = -1.0
    if (
        epipolar_ncc
        and left_gray is not None
        and right_gray is not None
        and surface != "near"
    ):
        d_lo = float(min_disparity) if min_disparity > 0 else 0.5
        d_hi = float(max_disparity) if max_disparity is not None and max_disparity > 0 else float(
            max(w // 3, 32)
        )
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        # Сначала центр; доп. точки — только если SGBM «подозрительно близко» или пуст.
        sample_pts = [(cx, cy)]
        need_extra = disp is None or (
            disp is not None and float(disp) > max(12.0, 0.08 * w)
        )
        if need_extra:
            sample_pts.extend(
                [
                    (x0 + bw // 3, cy),
                    (x0 + (2 * bw) // 3, cy),
                    (cx, y0 + max(bh // 4, 1)),
                ]
            )
        votes: list[tuple[float, float]] = []
        for px, py in sample_pts:
            ncc_i, s_i = epipolar_ncc_disparity(
                left_gray,
                right_gray,
                int(px),
                int(py),
                d_min=d_lo,
                d_max=d_hi,
                prefer_far=True,
            )
            if ncc_i is not None and s_i >= float(ncc_min_score):
                votes.append((float(ncc_i), float(s_i)))

        if votes:
            # Консенсус: среди сильных матчей берём перцентиль «дальше» (меньший d).
            strong = [v for v in votes if v[1] >= float(ncc_min_score) + 0.05] or votes
            d_arr = np.array([v[0] for v in strong], dtype=np.float32)
            s_arr = np.array([v[1] for v in strong], dtype=np.float32)
            ncc_disp = float(np.percentile(d_arr, 35))
            near = np.abs(d_arr - ncc_disp) <= 3.0
            ncc_score = float(s_arr[near].max()) if np.any(near) else float(s_arr.max())

        if ncc_disp is not None and ncc_score >= float(ncc_min_score):
            use_ncc = disp is None
            if disp is not None:
                d_sgbm = float(disp)
                # NCC заметно «дальше» (меньше d) — типичный анти-забор на SGBM.
                if ncc_disp + 2.5 < d_sgbm and ncc_score >= float(ncc_min_score):
                    use_ncc = True
                elif abs(ncc_disp - d_sgbm) <= 3.0 and ncc_score >= 0.40:
                    use_ncc = True
                elif ncc_score >= 0.50 and ncc_disp < d_sgbm:
                    use_ncc = True
                # SGBM «слишком близко» относительно широкого диапазона и NCC уверен.
                elif (
                    d_sgbm > 1.6 * max(ncc_disp, 1.0)
                    and ncc_score >= max(0.22, float(ncc_min_score) - 0.06)
                ):
                    use_ncc = True
            if use_ncc:
                disp = float(ncc_disp)
                refine_tag = f"epipolar-ncc:{ncc_score:.2f}"

    used_ys_l, used_xs_l = np.where(keep_mask)
    used_disp = patch[keep_mask].astype(np.float32)
    used_ys = (used_ys_l + y0).astype(np.int32)
    used_xs = (used_xs_l + x0).astype(np.int32)
    dbg = None
    if collect_debug:
        dbg = DisparityDebugInfo(
            inset_roi=inset_roi,
            used_ys=used_ys,
            used_xs=used_xs,
            used_disp=used_disp,
            selected_disp=float(disp) if disp is not None else None,
            n_patch=n_patch,
            n_valid=n_valid,
            n_used=int(used_disp.size),
            surface=surface,
            refine=refine_tag,
        )

    if disp is None or disp < float(min_disparity):
        return _pack(None, disp, dbg)

    if Q is not None:
        vec = np.array([[cx], [cy], [disp], [1.0]], dtype=np.float64)
        xyzw = Q @ vec
        wv = xyzw[3, 0]
        if abs(wv) < 1e-9:
            return _pack(None, disp, dbg)
        z = float(xyzw[2, 0] / wv)
        if not np.isfinite(z) or z <= 0:
            return _pack(None, disp, dbg)
        if max_distance_mm is not None and z > float(max_distance_mm):
            return _pack(None, disp, dbg)
        return _pack(z, disp, dbg)
    if focal is not None and baseline is not None and disp > 0:
        z = float(focal * baseline / disp)
        if max_distance_mm is not None and z > float(max_distance_mm):
            return _pack(None, disp, dbg)
        return _pack(z, disp, dbg)
    return _pack(None, disp, dbg)


def draw_disparity_debug(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    debug: DisparityDebugInfo,
    *,
    max_samples: int = 48,
    track_roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Side-by-side: пиксели ROI на L и соответствующие (x-d, y) на R."""
    left = left_bgr.copy()
    right = right_bgr.copy()
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    h = max(left.shape[0], right.shape[0])
    if left.shape[0] != h:
        left = cv2.resize(left, (left.shape[1], h), interpolation=cv2.INTER_AREA)
    if right.shape[0] != h:
        right = cv2.resize(right, (right.shape[1], h), interpolation=cv2.INTER_AREA)

    ix, iy, iw, ih = debug.inset_roi
    cv2.rectangle(left, (ix, iy), (ix + iw, iy + ih), (0, 255, 255), 1)
    if track_roi is not None:
        tx, ty, tw, th = track_roi
        cv2.rectangle(left, (tx, ty), (tx + tw, ty + th), (0, 220, 0), 2)

    # Полупрозрачная подсветка всех used-пикселей на L (цвет по d).
    if debug.n_used > 0:
        dmin = float(np.percentile(debug.used_disp, 5))
        dmax = float(np.percentile(debug.used_disp, 95))
        span = max(dmax - dmin, 1e-3)
        t = np.clip((debug.used_disp.astype(np.float32) - dmin) / span, 0.0, 1.0)
        colors = np.stack(
            [
                (255 * (1.0 - t)).astype(np.uint8),
                (200 * t).astype(np.uint8),
                (40 + 180 * t).astype(np.uint8),
            ],
            axis=1,
        )
        overlay = left.copy()
        ys = debug.used_ys
        xs = debug.used_xs
        in_l = (ys >= 0) & (ys < left.shape[0]) & (xs >= 0) & (xs < left.shape[1])
        overlay[ys[in_l], xs[in_l]] = colors[in_l]
        left = cv2.addWeighted(overlay, 0.55, left, 0.45, 0)

        overlay_r = right.copy()
        rxs = np.rint(xs.astype(np.float32) - debug.used_disp).astype(np.int32)
        in_r = (ys >= 0) & (ys < right.shape[0]) & (rxs >= 0) & (rxs < right.shape[1])
        overlay_r[ys[in_r], rxs[in_r]] = colors[in_r]
        right = cv2.addWeighted(overlay_r, 0.55, right, 0.45, 0)

    # Выборка точек для линий L→R (ближе к selected_disp — приоритетнее).
    samples: list[tuple[int, int, float]] = []
    if debug.n_used > 0:
        sel = (
            float(debug.selected_disp)
            if debug.selected_disp is not None
            else float(np.median(debug.used_disp))
        )
        order = np.argsort(np.abs(debug.used_disp - sel))
        step = max(1, int(order.size // max(max_samples, 1)))
        pick = order[::step][:max_samples]
        for i in pick:
            samples.append(
                (int(debug.used_xs[i]), int(debug.used_ys[i]), float(debug.used_disp[i]))
            )

    canvas = np.concatenate([left, right], axis=1)
    lw = left.shape[1]
    # Горизонтальные epipolar-линии: на идеальной ректификации сцена
    # должна лежать на одних и тех же рядах L и R. Если объект на R
    # визуально выше/ниже линии — это вертикальный сдвиг калибровки, не баг отрисовки.
    for frac in (0.25, 0.50, 0.75):
        y_line = int(round((h - 1) * frac))
        cv2.line(canvas, (0, y_line), (canvas.shape[1] - 1, y_line), (80, 80, 80), 1)

    for lx, ly, dd in samples:
        rx = int(round(lx - dd))
        ry = ly
        if not (0 <= ly < left.shape[0] and 0 <= lx < left.shape[1]):
            continue
        if not (0 <= ry < right.shape[0] and 0 <= rx < right.shape[1]):
            continue
        p1 = (lx, ly)
        p2 = (rx + lw, ry)
        cv2.circle(canvas, p1, 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, (0, 200, 255), 1, cv2.LINE_AA)

    # Маркер selected disparity на центре inset.
    if debug.selected_disp is not None and debug.selected_disp > 0:
        cx = ix + iw // 2
        cy = iy + ih // 2
        rcx = int(round(cx - float(debug.selected_disp)))
        cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        if 0 <= cy < right.shape[0] and 0 <= rcx < right.shape[1]:
            cv2.drawMarker(
                canvas, (rcx + lw, cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2
            )
            cv2.line(
                canvas, (cx, cy), (rcx + lw, cy), (0, 0, 255), 2, cv2.LINE_AA
            )

    lines = [
        f"DEBUG disp  used={debug.n_used}/{debug.n_patch}  valid0={debug.n_valid}",
        f"surface={debug.surface}  selected={debug.selected_disp:.2f}px"
        if debug.selected_disp is not None
        else f"surface={debug.surface}  selected=n/a",
        f"refine={debug.refine}" if debug.refine else "refine=sgbm-roi",
        "L|R same top edge; gray lines=epipolar rows (R higher => calib/rectify dy)",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (10, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (10, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


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
