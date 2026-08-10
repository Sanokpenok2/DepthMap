from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

from calib_quality import assess_calibration_quality, format_quality_report
from depth_map import split_sbs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Стереокалибровка по шахматной доске.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--left",
        default=None,
        help="Glob-шаблон путей к левым изображениям (в кавычках), напр. 'calib/left_*.png'.",
    )
    p.add_argument(
        "--right",
        default=None,
        help="Glob-шаблон путей к правым изображениям (в кавычках).",
    )
    p.add_argument(
        "--sbs",
        default=None,
        help="Glob SBS-фото доски (левая/правая половины кадра), напр. 'calib/sbs_*.png'.",
    )
    p.add_argument(
        "--swap-lr",
        action="store_true",
        help="Поменять половины SBS местами (если левая камера справа).",
    )
    p.add_argument(
        "--cols",
        type=int,
        default=8,
        help="Число внутренних углов доски по горизонтали.",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Число внутренних углов доски по вертикали.",
    )
    p.add_argument(
        "--square-size",
        type=float,
        default=90.0,
        help="Размер клетки доски в мм (задаёт масштаб глубины).",
    )
    p.add_argument(
        "--output",
        default="stereo_calib.npz",
        help="Файл для сохранения параметров калибровки.",
    )
    p.add_argument(
        "--debug-dir",
        default=None,
        help="Каталог для сохранения изображений с найденными углами.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help=(
            "Масштаб ректификации (0..1) для stereoRectify: 0 — максимальная обрезка "
            "без чёрных полей, 1 — сохранить весь кадр (возможны чёрные края). "
            "В режиме uncalibrated не используется для карт ремапа."
        ),
    )
    p.add_argument(
        "--rectify",
        choices=("calibrated", "uncalibrated"),
        default="calibrated",
        help=(
            "Метод ректификации: calibrated — stereoRectify; "
            "uncalibrated — stereoRectifyUncalibrated (гомографии по F и углам доски)."
        ),
    )
    p.add_argument(
        "--fix-k1",
        action="store_true",
        help="Не оценивать k1 (радиальное): зафиксировать k1=0.",
    )
    p.add_argument(
        "--fix-k2",
        action="store_true",
        help="Не оценивать k2: зафиксировать k2=0.",
    )
    p.add_argument(
        "--fix-k3",
        action="store_true",
        help=(
            "Не оценивать k3 (часто раздувается при слабой дисторсии / Blender). "
            "Зафиксировать k3=0."
        ),
    )
    p.add_argument(
        "--fix-tangential",
        action="store_true",
        help="Не оценивать p1,p2 (тангенциальная дисторсия): зафиксировать p1=p2=0.",
    )
    p.add_argument(
        "--zero-distortion",
        action="store_true",
        help="Эквивалент --fix-k1 --fix-k2 --fix-k3 --fix-tangential (идеальный pinhole).",
    )
    p.add_argument(
        "--calib-left",
        default=None,
        help=(
            "Готовая калибровка левой камеры (.npz): mtx/camera_matrix + dist. "
            "Вместе с --calib-right пропускает mono calibrateCamera."
        ),
    )
    p.add_argument(
        "--calib-right",
        default=None,
        help="Готовая калибровка правой камеры (.npz), см. --calib-left.",
    )
    p.add_argument(
        "--mono",
        choices=("left", "right"),
        default=None,
        help=(
            "Режим калибровки одной камеры: нужны --images (или --left/--right как glob). "
            "Сохраняет mono .npz (mtx, dist, image_size)."
        ),
    )
    p.add_argument(
        "--images",
        default=None,
        help="Glob изображений для --mono (если не задан — берётся --left или --right).",
    )
    p.add_argument(
        "--export-mono-left",
        default=None,
        help="При полной стереокалибровке дополнительно сохранить mono left .npz.",
    )
    p.add_argument(
        "--export-mono-right",
        default=None,
        help="При полной стереокалибровке дополнительно сохранить mono right .npz.",
    )
    return p.parse_args()


def build_distortion_flags(
    *,
    fix_k1: bool = False,
    fix_k2: bool = False,
    fix_k3: bool = False,
    fix_tangential: bool = False,
    zero_distortion: bool = False,
) -> tuple[int, list[str]]:
    """Собирает флаги OpenCV для фиксации коэффициентов дисторсии."""
    if zero_distortion:
        fix_k1 = fix_k2 = fix_k3 = fix_tangential = True
    flags = 0
    fixed: list[str] = []
    if fix_k1:
        flags |= int(cv2.CALIB_FIX_K1)
        fixed.append("k1")
    if fix_k2:
        flags |= int(cv2.CALIB_FIX_K2)
        fixed.append("k2")
    if fix_k3:
        flags |= int(cv2.CALIB_FIX_K3)
        fixed.append("k3")
    if fix_tangential:
        # ZERO обнуляет и фиксирует p1,p2; FIX_TANGENT_DIST тоже оставляет начальные.
        flags |= int(cv2.CALIB_ZERO_TANGENT_DIST)
        fixed.append("p1,p2")
    return flags, fixed


def _pick_calib_array(data: dict, *names: str) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    return None


def load_mono_calibration(path: str, *, prefer: str | None = None) -> dict:
    """Загружает калибровку одной камеры из .npz.

    Поддерживаемые ключи:
      mtx / camera_matrix / K / mtx_l / mtx_r
      dist / dist_coeffs / D / dist_l / dist_r
      image_size (опционально)
      rms / rms_l / rms_r (опционально)

    prefer="left"|"right" — при полном stereo .npz взять mtx_l/dist_l или mtx_r/dist_r.
    """
    try:
        raw = np.load(path, allow_pickle=True)
    except OSError as exc:
        raise ValueError(f"Не удалось прочитать калибровку '{path}': {exc}") from exc
    data = {k: raw[k] for k in raw.files}

    if prefer == "left":
        mtx = _pick_calib_array(data, "mtx_l", "mtx", "camera_matrix", "K")
        dist = _pick_calib_array(data, "dist_l", "dist", "dist_coeffs", "D")
        rms_keys = ("rms_l", "rms")
    elif prefer == "right":
        mtx = _pick_calib_array(data, "mtx_r", "mtx", "camera_matrix", "K")
        dist = _pick_calib_array(data, "dist_r", "dist", "dist_coeffs", "D")
        rms_keys = ("rms_r", "rms")
    else:
        mtx = _pick_calib_array(data, "mtx", "camera_matrix", "K", "mtx_l", "mtx_r")
        dist = _pick_calib_array(data, "dist", "dist_coeffs", "D", "dist_l", "dist_r")
        rms_keys = ("rms", "rms_l", "rms_r")

    if mtx is None or dist is None:
        raise ValueError(
            f"В '{path}' нет матрицы камеры/дисторсии "
            "(ожидаются mtx/camera_matrix и dist/dist_coeffs)."
        )
    mtx = np.asarray(mtx, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    if dist.size < 4:
        raise ValueError(f"В '{path}' слишком короткий dist (нужно ≥4 коэфф.).")
    if dist.size < 5:
        dist = np.vstack([dist, np.zeros((5 - dist.size, 1), dtype=np.float64)])

    image_size = None
    if "image_size" in data:
        wh = np.asarray(data["image_size"]).ravel()
        if wh.size >= 2:
            image_size = (int(wh[0]), int(wh[1]))

    rms = float("nan")
    for key in rms_keys:
        if key in data:
            rms = float(np.asarray(data[key]).ravel()[0])
            break

    return {
        "mtx": mtx,
        "dist": dist,
        "image_size": image_size,
        "rms": rms,
        "path": str(Path(path).resolve()),
    }


def save_mono_calibration(
    path: str,
    *,
    mtx: np.ndarray,
    dist: np.ndarray,
    image_size: tuple[int, int],
    rms: float | None = None,
    side: str = "",
) -> str:
    """Сохраняет калибровку одной камеры в .npz (совместимо с --calib-left/right)."""
    payload = {
        "mtx": np.asarray(mtx, dtype=np.float64),
        "camera_matrix": np.asarray(mtx, dtype=np.float64),
        "dist": np.asarray(dist, dtype=np.float64),
        "dist_coeffs": np.asarray(dist, dtype=np.float64),
        "image_size": np.array(image_size),
    }
    if rms is not None and np.isfinite(rms):
        payload["rms"] = np.array([float(rms)])
    if side:
        payload["side"] = np.array([side])
    np.savez(path, **payload)
    return str(Path(path).resolve())


def find_pattern_size(cols: int, rows: int) -> tuple[int, int]:
    return (cols, rows)


def build_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    """Координаты углов доски в её собственной системе (Z=0)."""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def find_board_corners(
    gray: np.ndarray, cols: int, rows: int
) -> np.ndarray | None:
    """Ищет углы доски на одном кадре. Возвращает corners или None."""
    if gray is None or gray.size == 0:
        return None
    gray = to_gray(gray)
    pattern = find_pattern_size(cols, rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)


def draw_board_corners(
    bgr: np.ndarray, corners: np.ndarray, cols: int, rows: int
) -> np.ndarray:
    """Копия кадра с нарисованными углами доски."""
    if bgr.ndim == 2:
        out = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    else:
        out = bgr.copy()
    pattern = find_pattern_size(cols, rows)
    cv2.drawChessboardCorners(out, pattern, corners, True)
    return out


def list_images_in_dir(folder: str | Path) -> list[str]:
    """Отсортированный список путей к изображениям в каталоге (без рекурсии)."""
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"Нет каталога: {path}")
    files = [
        str(p)
        for p in sorted(path.iterdir())
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]
    if not files:
        raise ValueError(f"В каталоге нет изображений: {path}")
    return files


def run_stereo_calibration_from_folders(
    left_dir: str,
    right_dir: str,
    output: str,
    *,
    cols: int = 8,
    rows: int = 5,
    square_size: float = 90.0,
    alpha: float = 1.0,
    rectify_mode: str = "calibrated",
    debug_dir: str | None = None,
    fix_k1: bool = False,
    fix_k2: bool = False,
    fix_k3: bool = False,
    fix_tangential: bool = False,
    zero_distortion: bool = False,
    calib_left: str | None = None,
    calib_right: str | None = None,
    export_mono_left: str | None = None,
    export_mono_right: str | None = None,
) -> tuple[str, list[str]]:
    """Калибровка стереопары по папкам left/right → .npz.

    Опционально калибровки камер по отдельности (как CLI --calib-left/--calib-right):
    тогда intrinsics берутся из файлов, считается только stereoCalibrate + rectify.
    """
    left_paths = list_images_in_dir(left_dir)
    right_paths = list_images_in_dir(right_dir)
    pairs = load_pairs_from_paths(left_paths, right_paths)
    dist_flags, fixed = build_distortion_flags(
        fix_k1=fix_k1,
        fix_k2=fix_k2,
        fix_k3=fix_k3,
        fix_tangential=fix_tangential,
        zero_distortion=zero_distortion,
    )
    mono_left = mono_right = None
    if calib_left or calib_right:
        if not (calib_left and calib_right):
            raise ValueError(
                "Нужны оба файла калибровки камер: calib_left и calib_right."
            )
        mono_left = load_mono_calibration(calib_left, prefer="left")
        mono_right = load_mono_calibration(calib_right, prefer="right")
    return calibrate_stereo(
        pairs,
        cols,
        rows,
        square_size,
        output,
        debug_dir=debug_dir,
        alpha=alpha,
        rectify_mode=rectify_mode,
        dist_flags=dist_flags,
        fixed_dist_names=fixed,
        mono_left=mono_left,
        mono_right=mono_right,
        export_mono_left=export_mono_left,
        export_mono_right=export_mono_right,
    )


def run_mono_calibration_from_folder(
    images_dir: str,
    output: str,
    *,
    cols: int = 8,
    rows: int = 5,
    square_size: float = 90.0,
    side: str = "",
    debug_dir: str | None = None,
    fix_k1: bool = False,
    fix_k2: bool = False,
    fix_k3: bool = False,
    fix_tangential: bool = False,
    zero_distortion: bool = False,
) -> tuple[str, list[str]]:
    """Монокалибровка по одной папке → .npz."""
    paths = list_images_in_dir(images_dir)
    images = _load_gray_images(paths)
    dist_flags, fixed = build_distortion_flags(
        fix_k1=fix_k1,
        fix_k2=fix_k2,
        fix_k3=fix_k3,
        fix_tangential=fix_tangential,
        zero_distortion=zero_distortion,
    )
    return calibrate_mono_camera(
        images,
        cols,
        rows,
        square_size,
        output,
        debug_dir=debug_dir,
        dist_flags=dist_flags,
        fixed_dist_names=fixed,
        side=side,
    )


def describe_stereo_geometry(
    mtx_l: np.ndarray,
    mtx_r: np.ndarray,
    T: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Возвращает строки журнала и рассчитанные параметры камер."""
    fx_l, fy_l = float(mtx_l[0, 0]), float(mtx_l[1, 1])
    fx_r, fy_r = float(mtx_r[0, 0]), float(mtx_r[1, 1])
    baseline_mm = float(np.linalg.norm(T))
    focal_rect_l = float(P1[0, 0])
    focal_rect_r = float(P2[0, 0]) if P2 is not None else focal_rect_l
    lines = [
        "Рассчитанные параметры:",
        "  До ректификации:",
        f"    Левая камера:  fx={fx_l:.2f} px, fy={fy_l:.2f} px",
        f"    Правая камера: fx={fx_r:.2f} px, fy={fy_r:.2f} px",
        "  После ректификации:",
        f"    Левая камера:  fx={focal_rect_l:.2f} px",
        f"    Правая камера: fx={focal_rect_r:.2f} px",
        f"  База между камерами: {baseline_mm:.2f} мм",
        (
            "  Для depth_map без --calib: "
            f"--focal {focal_rect_l:.1f} --baseline {baseline_mm:.1f}"
        ),
    ]
    metrics = {
        "focal_px": focal_rect_l,
        "focal_rect_l_px": focal_rect_l,
        "focal_rect_r_px": focal_rect_r,
        "focal_l_px": fx_l,
        "focal_r_px": fx_r,
        "fy_l_px": fy_l,
        "fy_r_px": fy_r,
        "baseline_mm": baseline_mm,
    }
    return lines, metrics


def clear_debug_dir(debug_dir: str) -> None:
    """Удаляет старые файлы из каталога отладки перед новой калибровкой."""
    path = Path(debug_dir)
    if not path.is_dir():
        return
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def align_stereo_pair(
    img_l: np.ndarray, img_r: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Обрезает пару до общего размера, если различие только в габаритах."""
    if img_l.shape == img_r.shape:
        return img_l, img_r, False

    h = min(img_l.shape[0], img_r.shape[0])
    w = min(img_l.shape[1], img_r.shape[1])
    if h <= 0 or w <= 0:
        return img_l, img_r, False

    return img_l[:h, :w], img_r[:h, :w], True


def collect_calibration_corners(
    pairs: list[tuple[np.ndarray | None, np.ndarray | None, str]],
    cols: int,
    rows: int,
    square_size: float,
    debug_dir: str | None = None,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    tuple[int, int],
    list[str],
]:
    """Ищет углы доски на списке пар (left_gray, right_gray, label)."""
    pattern = find_pattern_size(cols, rows)
    objp = build_object_points(cols, rows, square_size)

    objpoints: list[np.ndarray] = []
    imgpoints_l: list[np.ndarray] = []
    imgpoints_r: list[np.ndarray] = []

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    image_size: tuple[int, int] | None = None
    log: list[str] = []
    skipped_unreadable = 0
    skipped_size = 0
    aligned_pairs = 0
    size_mismatch_logged = False

    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        clear_debug_dir(debug_dir)

    for img_l, img_r, label in pairs:
        if img_l is None or img_r is None or img_l.size == 0 or img_r.size == 0:
            skipped_unreadable += 1
            log.append(f"Пропуск (не читается): {label}")
            continue

        if img_l.ndim == 3:
            img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        if img_r.ndim == 3:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if img_l.shape != img_r.shape:
            aligned_l, aligned_r, was_aligned = align_stereo_pair(img_l, img_r)
            if not was_aligned:
                skipped_size += 1
                log.append(
                    f"Пропуск (разные размеры): {label} "
                    f"{img_l.shape} / {img_r.shape}"
                )
                continue
            if not size_mismatch_logged:
                log.append(
                    "Предупреждение: размеры левых и правых кадров различаются. "
                    "Пары будут обрезаны до общей области."
                )
                size_mismatch_logged = True
            aligned_pairs += 1
            img_l, img_r = aligned_l, aligned_r

        image_size = (img_l.shape[1], img_l.shape[0])

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found_l, corners_l = cv2.findChessboardCorners(img_l, pattern, flags)
        found_r, corners_r = cv2.findChessboardCorners(img_r, pattern, flags)

        if not (found_l and found_r):
            log.append(f"Доска не найдена: {label}")
            continue

        corners_l = cv2.cornerSubPix(img_l, corners_l, (11, 11), (-1, -1), subpix_criteria)
        corners_r = cv2.cornerSubPix(img_r, corners_r, (11, 11), (-1, -1), subpix_criteria)

        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

        if debug_dir:
            safe = label.replace("/", "_").replace("\\", "_")
            vis_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis_l, pattern, corners_l, found_l)
            cv2.imwrite(str(Path(debug_dir) / f"corners_left_{safe}"), vis_l)

            vis_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis_r, pattern, corners_r, found_r)
            cv2.imwrite(str(Path(debug_dir) / f"corners_right_{safe}"), vis_r)

    if image_size is None:
        raise ValueError(
            "Не удалось использовать ни одной пары изображений. "
            f"Не читается: {skipped_unreadable}, "
            f"несовместимые размеры: {skipped_size}."
        )
    if aligned_pairs:
        log.append(f"Обрезано до общего размера {image_size[0]}x{image_size[1]}: {aligned_pairs} пар.")
    if len(objpoints) < 3:
        raise ValueError(
            f"Доска найдена только на {len(objpoints)} парах. "
            "Нужно минимум 3 (рекомендуется 10-20) с разных ракурсов."
        )

    log.append(f"Углы найдены на {len(objpoints)} парах.")
    return objpoints, imgpoints_l, imgpoints_r, image_size, log


def load_pairs_from_paths(
    left_paths: list[str], right_paths: list[str]
) -> list[tuple[np.ndarray | None, np.ndarray | None, str]]:
    if len(left_paths) != len(right_paths):
        raise ValueError(
            f"Число левых ({len(left_paths)}) и правых ({len(right_paths)}) "
            "изображений не совпадает."
        )
    if not left_paths:
        raise ValueError("Не найдено ни одной пары изображений по указанным glob.")
    pairs: list[tuple[np.ndarray | None, np.ndarray | None, str]] = []
    for lf, rf in zip(left_paths, right_paths):
        img_l = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
        label = f"{Path(lf).name} / {Path(rf).name}"
        pairs.append((img_l, img_r, label))
    return pairs


def load_pairs_from_sbs(
    sbs_paths: list[str], swap_lr: bool = False
) -> list[tuple[np.ndarray | None, np.ndarray | None, str]]:
    if not sbs_paths:
        raise ValueError("Не найдено ни одного SBS-изображения по указанному glob.")
    pairs: list[tuple[np.ndarray | None, np.ndarray | None, str]] = []
    for path in sbs_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            pairs.append((None, None, Path(path).name))
            continue
        left, right = split_sbs(img, swap_lr=swap_lr)
        left_g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        pairs.append((left_g, right_g, Path(path).name))
    return pairs


def stack_image_points(imgpoints: list[np.ndarray]) -> np.ndarray:
    """Собирает углы со всех кадров в массив Nx2."""
    return np.vstack([p.reshape(-1, 2) for p in imgpoints]).astype(np.float64)


def homography_to_remap_maps(
    H: np.ndarray, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Строит float-карты для cv2.remap по обратной гомографии (эквивалент warpPerspective)."""
    w, h = image_size
    H_inv = np.linalg.inv(H)
    xs, ys = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    ones = np.ones_like(xs, dtype=np.float32)
    pts = np.stack([xs, ys, ones], axis=0).reshape(3, -1)
    mapped = H_inv @ pts
    denom = mapped[2]
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    map_x = (mapped[0] / denom).reshape(h, w).astype(np.float32)
    map_y = (mapped[1] / denom).reshape(h, w).astype(np.float32)
    return map_x, map_y


def stereo_rectify_calibrated(
    mtx_l: np.ndarray,
    dist_l: np.ndarray,
    mtx_r: np.ndarray,
    dist_r: np.ndarray,
    image_size: tuple[int, int],
    R: np.ndarray,
    T: np.ndarray,
    alpha: float,
    log: list[str],
) -> dict:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    log.append(f"Ректификация: stereoRectify (alpha={alpha:.2f})")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )
    log.append(
        f"  Область без чёрных полей: лев. {tuple(roi1)}, прав. {tuple(roi2)}"
    )

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2
    )

    return {
        "rectification_method": "calibrated",
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "alpha": alpha,
        "roi1": roi1,
        "roi2": roi2,
        "map1_l": map1_l,
        "map2_l": map2_l,
        "map1_r": map1_r,
        "map2_r": map2_r,
        "H1": None,
        "H2": None,
    }


def stereo_rectify_uncalibrated(
    mtx_l: np.ndarray,
    dist_l: np.ndarray,
    mtx_r: np.ndarray,
    dist_r: np.ndarray,
    image_size: tuple[int, int],
    R: np.ndarray,
    T: np.ndarray,
    F: np.ndarray,
    imgpoints_l: list[np.ndarray],
    imgpoints_r: list[np.ndarray],
    alpha: float,
    log: list[str],
) -> dict:
    pts_l = stack_image_points(imgpoints_l)
    pts_r = stack_image_points(imgpoints_r)
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(pts_l, pts_r, F, image_size)
    if not ok:
        raise ValueError(
            "stereoRectifyUncalibrated не удалось вычислить гомографии. "
            "Проверьте качество углов и соответствие пар."
        )

    log.append("Ректификация: stereoRectifyUncalibrated")
    log.append(f"  Соответствующих углов: {len(pts_l)}")
    map1_l, map2_l = homography_to_remap_maps(H1, image_size)
    map1_r, map2_r = homography_to_remap_maps(H2, image_size)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    log.append(
        "  Q, P1, P2 и ROI берутся из stereoRectify для совместимости с depth_map."
    )
    if alpha < 0.99:
        log.append(
            "  Предупреждение: alpha влияет только на Q/ROI, не на гомографии H1/H2."
        )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )
    log.append(
        f"  Область без чёрных полей (справочно): лев. {tuple(roi1)}, прав. {tuple(roi2)}"
    )

    return {
        "rectification_method": "uncalibrated",
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "alpha": alpha,
        "roi1": roi1,
        "roi2": roi2,
        "map1_l": map1_l,
        "map2_l": map2_l,
        "map1_r": map1_r,
        "map2_r": map2_r,
        "H1": H1,
        "H2": H2,
    }


def calibrate_pinhole(
    objpoints: list[np.ndarray],
    imgpoints_l: list[np.ndarray],
    imgpoints_r: list[np.ndarray],
    image_size: tuple[int, int],
    alpha: float,
    rectify_mode: str,
    log: list[str],
    *,
    dist_flags: int = 0,
    fixed_dist_names: list[str] | None = None,
    mono_left: dict | None = None,
    mono_right: dict | None = None,
) -> dict:
    log.append("Калибровка pinhole-модели...")
    if fixed_dist_names:
        log.append(
            "  Фиксированные коэфф. дисторсии (=0): " + ", ".join(fixed_dist_names)
        )
    elif dist_flags == 0 and mono_left is None and mono_right is None:
        log.append("  Дисторсия: оцениваются k1,k2,p1,p2,k3")

    dist0 = np.zeros((5, 1), dtype=np.float64)
    mono_flags = int(dist_flags)

    if mono_left is not None and mono_right is not None:
        log.append("  Интринсики загружены из готовых mono-калибровок (без mono calibrateCamera).")
        log.append(f"    left:  {mono_left['path']}")
        log.append(f"    right: {mono_right['path']}")
        mtx_l = np.asarray(mono_left["mtx"], dtype=np.float64).copy()
        dist_l = np.asarray(mono_left["dist"], dtype=np.float64).reshape(-1, 1).copy()
        mtx_r = np.asarray(mono_right["mtx"], dtype=np.float64).copy()
        dist_r = np.asarray(mono_right["dist"], dtype=np.float64).reshape(-1, 1).copy()
        ret_l = float(mono_left.get("rms", float("nan")))
        ret_r = float(mono_right.get("rms", float("nan")))
        for side, mono in (("left", mono_left), ("right", mono_right)):
            sz = mono.get("image_size")
            if sz is not None and tuple(sz) != tuple(image_size):
                log.append(
                    f"  Предупреждение: image_size в калибровке {side} {tuple(sz)} "
                    f"≠ размер углов {image_size}."
                )
        if np.isfinite(ret_l):
            log.append(f"  RMS mono left (из файла):  {ret_l:.4f}")
        if np.isfinite(ret_r):
            log.append(f"  RMS mono right (из файла): {ret_r:.4f}")
    else:
        ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
            objpoints,
            imgpoints_l,
            image_size,
            None,
            dist0.copy(),
            flags=mono_flags,
        )
        ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
            objpoints,
            imgpoints_r,
            image_size,
            None,
            dist0.copy(),
            flags=mono_flags,
        )
        log.append(f"  RMS-ошибка левой камеры:  {ret_l:.4f}")
        log.append(f"  RMS-ошибка правой камеры: {ret_r:.4f}")

    log.append(f"  dist_l: {np.asarray(dist_l).ravel()}")
    log.append(f"  dist_r: {np.asarray(dist_r).ravel()}")

    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    (
        ret_stereo,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        R,
        T,
        _E,
        F,
    ) = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        criteria=stereo_criteria,
        flags=stereo_flags,
    )
    log.append(f"  RMS-ошибка стереокалибровки: {ret_stereo:.4f}")

    if rectify_mode == "uncalibrated":
        rect = stereo_rectify_uncalibrated(
            mtx_l,
            dist_l,
            mtx_r,
            dist_r,
            image_size,
            R,
            T,
            F,
            imgpoints_l,
            imgpoints_r,
            alpha,
            log,
        )
    else:
        rect = stereo_rectify_calibrated(
            mtx_l,
            dist_l,
            mtx_r,
            dist_r,
            image_size,
            R,
            T,
            alpha,
            log,
        )

    return {
        "model": "pinhole",
        "rms_l": ret_l,
        "rms_r": ret_r,
        "rms_stereo": ret_stereo,
        "mtx_l": mtx_l,
        "dist_l": dist_l,
        "mtx_r": mtx_r,
        "dist_r": dist_r,
        "R": R,
        "T": T,
        "F": F,
        "dist_flags": mono_flags,
        "fixed_dist": list(fixed_dist_names or []),
        "intrinsics_from_files": bool(mono_left is not None and mono_right is not None),
        **rect,
    }


def collect_mono_corners(
    images: list[tuple[np.ndarray | None, str]],
    cols: int,
    rows: int,
    square_size: float,
    debug_dir: str | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int], list[str]]:
    """Ищет углы доски на списке одиночных кадров (gray/bgr, label)."""
    pattern = find_pattern_size(cols, rows)
    objp = build_object_points(cols, rows, square_size)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    image_size: tuple[int, int] | None = None
    log: list[str] = []

    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        clear_debug_dir(debug_dir)

    for img, label in images:
        if img is None or img.size == 0:
            log.append(f"Пропуск (не читается): {label}")
            continue
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != image_size:
            log.append(f"Пропуск (размер): {label}")
            continue

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if not found:
            log.append(f"Углы не найдены: {label}")
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), subpix_criteria
        )
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        if debug_dir:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern, corners, True)
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
            cv2.imwrite(str(Path(debug_dir) / f"corners_{safe}"), vis)

    if image_size is None or len(objpoints) < 3:
        raise ValueError(
            f"Доска найдена только на {len(objpoints)} кадрах. "
            "Нужно минимум 3 (рекомендуется 10–20)."
        )
    log.append(f"Углы найдены на {len(objpoints)} кадрах.")
    return objpoints, imgpoints, image_size, log


def calibrate_mono_camera(
    images: list[tuple[np.ndarray | None, str]],
    cols: int,
    rows: int,
    square_size: float,
    output: str,
    *,
    debug_dir: str | None = None,
    dist_flags: int = 0,
    fixed_dist_names: list[str] | None = None,
    side: str = "",
) -> tuple[str, list[str]]:
    """Калибровка одной камеры → mono .npz."""
    objpoints, imgpoints, image_size, log = collect_mono_corners(
        images, cols, rows, square_size, debug_dir
    )
    log.append(f"Монокалибровка ({side or 'camera'})...")
    if fixed_dist_names:
        log.append("  Фиксированные коэфф.: " + ", ".join(fixed_dist_names))
    dist0 = np.zeros((5, 1), dtype=np.float64)
    rms, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        dist0,
        flags=int(dist_flags),
    )
    log.append(f"  RMS: {rms:.4f}")
    log.append(f"  fx={mtx[0,0]:.2f} fy={mtx[1,1]:.2f} cx={mtx[0,2]:.2f} cy={mtx[1,2]:.2f}")
    log.append(f"  dist: {np.asarray(dist).ravel()}")
    out = save_mono_calibration(
        output,
        mtx=mtx,
        dist=dist,
        image_size=image_size,
        rms=float(rms),
        side=side,
    )
    log.append(f"Mono-калибровка сохранена: {out}")
    return out, log


def calibrate_stereo(
    pairs: list[tuple[np.ndarray | None, np.ndarray | None, str]],
    cols: int,
    rows: int,
    square_size: float,
    output: str,
    debug_dir: str | None = None,
    alpha: float = 1.0,
    rectify_mode: str = "calibrated",
    *,
    dist_flags: int = 0,
    fixed_dist_names: list[str] | None = None,
    mono_left: dict | None = None,
    mono_right: dict | None = None,
    export_mono_left: str | None = None,
    export_mono_right: str | None = None,
) -> tuple[str, list[str]]:
    """Выполняет стереокалибровку и сохраняет результат в .npz."""
    if not pairs:
        raise ValueError("Не найдены изображения для калибровки.")
    if (mono_left is None) ^ (mono_right is None):
        raise ValueError("Укажите оба файла: --calib-left и --calib-right.")

    objpoints, imgpoints_l, imgpoints_r, image_size, log = collect_calibration_corners(
        pairs, cols, rows, square_size, debug_dir
    )

    result = calibrate_pinhole(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        image_size,
        alpha,
        rectify_mode,
        log,
        dist_flags=dist_flags,
        fixed_dist_names=fixed_dist_names,
        mono_left=mono_left,
        mono_right=mono_right,
    )

    if export_mono_left:
        path = save_mono_calibration(
            export_mono_left,
            mtx=result["mtx_l"],
            dist=result["dist_l"],
            image_size=image_size,
            rms=float(result["rms_l"]) if np.isfinite(result["rms_l"]) else None,
            side="left",
        )
        log.append(f"Экспорт mono left: {path}")
    if export_mono_right:
        path = save_mono_calibration(
            export_mono_right,
            mtx=result["mtx_r"],
            dist=result["dist_r"],
            image_size=image_size,
            rms=float(result["rms_r"]) if np.isfinite(result["rms_r"]) else None,
            side="right",
        )
        log.append(f"Экспорт mono right: {path}")

    geom_lines, metrics = describe_stereo_geometry(
        result["mtx_l"], result["mtx_r"], result["T"], result["P1"], result["P2"]
    )
    log.extend(geom_lines)

    warnings = assess_calibration_quality(
        model=result["model"],
        rms_l=result["rms_l"],
        rms_r=result["rms_r"],
        rms_stereo=result["rms_stereo"],
        mtx_l=result["mtx_l"],
        mtx_r=result["mtx_r"],
        baseline_mm=metrics["baseline_mm"],
        map1_l=result["map1_l"],
        map2_l=result["map2_l"],
        map1_r=result["map1_r"],
        map2_r=result["map2_r"],
        image_size=image_size,
        alpha=result["alpha"],
        roi1=result["roi1"],
        roi2=result["roi2"],
        dist_l=result["dist_l"],
        dist_r=result["dist_r"],
    )
    log.extend(format_quality_report(warnings))

    np.savez(
        output,
        model=np.array([result["model"]]),
        rectification_method=np.array([result["rectification_method"]]),
        image_size=np.array(image_size),
        mtx_l=result["mtx_l"],
        dist_l=result["dist_l"],
        mtx_r=result["mtx_r"],
        dist_r=result["dist_r"],
        R=result["R"],
        T=result["T"],
        F=result["F"],
        R1=result["R1"],
        R2=result["R2"],
        P1=result["P1"],
        P2=result["P2"],
        Q=result["Q"],
        alpha=np.array([result["alpha"]]),
        focal_px=np.array([metrics["focal_px"]]),
        focal_l_px=np.array([metrics["focal_l_px"]]),
        focal_r_px=np.array([metrics["focal_r_px"]]),
        focal_rect_l_px=np.array([metrics["focal_rect_l_px"]]),
        focal_rect_r_px=np.array([metrics["focal_rect_r_px"]]),
        fy_l_px=np.array([metrics["fy_l_px"]]),
        fy_r_px=np.array([metrics["fy_r_px"]]),
        baseline_mm=np.array([metrics["baseline_mm"]]),
        roi1=np.array(result["roi1"]),
        roi2=np.array(result["roi2"]),
        map1_l=result["map1_l"],
        map2_l=result["map2_l"],
        map1_r=result["map1_r"],
        map2_r=result["map2_r"],
        H1=result["H1"] if result["H1"] is not None else np.array([]),
        H2=result["H2"] if result["H2"] is not None else np.array([]),
        dist_flags=np.array([int(result.get("dist_flags", 0))]),
        fixed_dist=np.array(result.get("fixed_dist", []), dtype=object),
        intrinsics_from_files=np.array(
            [bool(result.get("intrinsics_from_files", False))]
        ),
        quality_warnings=np.array(warnings, dtype=object),
    )
    out_path = str(Path(output).resolve())
    log.append(f"Параметры калибровки сохранены: {out_path}")
    return out_path, log


def _load_gray_images(paths: list[str]) -> list[tuple[np.ndarray | None, str]]:
    out: list[tuple[np.ndarray | None, str]] = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        out.append((img, Path(path).name))
    return out


def main() -> None:
    args = parse_args()

    dist_flags, fixed_dist = build_distortion_flags(
        fix_k1=args.fix_k1,
        fix_k2=args.fix_k2,
        fix_k3=args.fix_k3,
        fix_tangential=args.fix_tangential,
        zero_distortion=args.zero_distortion,
    )

    # --- Режим одной камеры ---
    if args.mono is not None:
        glob_pat = args.images
        if glob_pat is None:
            glob_pat = args.left if args.mono == "left" else args.right
        if not glob_pat:
            sys.exit(
                "Ошибка: для --mono укажите --images или соответствующий "
                "--left/--right glob."
            )
        paths = sorted(glob.glob(glob_pat))
        if not paths:
            sys.exit(f"Ошибка: не найдено изображений по шаблону '{glob_pat}'.")
        try:
            _out, log = calibrate_mono_camera(
                _load_gray_images(paths),
                args.cols,
                args.rows,
                args.square_size,
                args.output,
                debug_dir=args.debug_dir,
                dist_flags=dist_flags,
                fixed_dist_names=fixed_dist,
                side=args.mono,
            )
        except ValueError as exc:
            sys.exit(f"Ошибка: {exc}")
        except cv2.error as exc:
            sys.exit(f"Ошибка OpenCV: {exc}")
        for line in log:
            print(line)
        print("Используйте этот файл как --calib-left / --calib-right при стереокалибровке.")
        return

    use_sbs = args.sbs is not None
    use_pair = args.left is not None or args.right is not None
    if use_sbs and use_pair:
        sys.exit("Ошибка: укажите либо --sbs, либо пару --left/--right, не оба варианта.")
    if use_sbs:
        pairs = load_pairs_from_sbs(sorted(glob.glob(args.sbs)), swap_lr=args.swap_lr)
    elif args.left and args.right:
        pairs = load_pairs_from_paths(
            sorted(glob.glob(args.left)), sorted(glob.glob(args.right))
        )
    else:
        sys.exit(
            "Ошибка: укажите --left и --right (отдельные кадры) "
            "либо --sbs (SBS-фото доски)."
        )

    mono_left = mono_right = None
    if args.calib_left or args.calib_right:
        if not (args.calib_left and args.calib_right):
            sys.exit("Ошибка: нужны оба аргумента --calib-left и --calib-right.")
        try:
            mono_left = load_mono_calibration(args.calib_left, prefer="left")
            mono_right = load_mono_calibration(args.calib_right, prefer="right")
        except ValueError as exc:
            sys.exit(f"Ошибка: {exc}")

    try:
        _out_path, log = calibrate_stereo(
            pairs,
            args.cols,
            args.rows,
            args.square_size,
            args.output,
            args.debug_dir,
            args.alpha,
            args.rectify,
            dist_flags=dist_flags,
            fixed_dist_names=fixed_dist,
            mono_left=mono_left,
            mono_right=mono_right,
            export_mono_left=args.export_mono_left,
            export_mono_right=args.export_mono_right,
        )
    except ValueError as exc:
        sys.exit(f"Ошибка: {exc}")
    except cv2.error as exc:
        sys.exit(f"Ошибка OpenCV: {exc}")

    for line in log:
        print(line)
    print("Теперь используйте их в depth_map.py через --calib.")


if __name__ == "__main__":
    main()