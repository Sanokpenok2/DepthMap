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
        default=9,
        help="Число внутренних углов доски по горизонтали.",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Число внутренних углов доски по вертикали.",
    )
    p.add_argument(
        "--square-size",
        type=float,
        default=25.0,
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
    return p.parse_args()


def find_pattern_size(cols: int, rows: int) -> tuple[int, int]:
    return (cols, rows)


def build_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    """Координаты углов доски в её собственной системе (Z=0)."""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


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
) -> dict:
    log.append("Калибровка pinhole-модели...")

    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_l, image_size, None, None
    )
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_r, image_size, None, None
    )
    log.append(f"  RMS-ошибка левой камеры:  {ret_l:.4f}")
    log.append(f"  RMS-ошибка правой камеры: {ret_r:.4f}")

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
        **rect,
    }


def calibrate_stereo(
    pairs: list[tuple[np.ndarray | None, np.ndarray | None, str]],
    cols: int,
    rows: int,
    square_size: float,
    output: str,
    debug_dir: str | None = None,
    alpha: float = 1.0,
    rectify_mode: str = "calibrated",
) -> tuple[str, list[str]]:
    """Выполняет стереокалибровку и сохраняет результат в .npz."""
    if not pairs:
        raise ValueError("Не найдены изображения для калибровки.")

    objpoints, imgpoints_l, imgpoints_r, image_size, log = collect_calibration_corners(
        pairs, cols, rows, square_size, debug_dir
    )

    result = calibrate_pinhole(
        objpoints, imgpoints_l, imgpoints_r, image_size, alpha, rectify_mode, log
    )

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
        quality_warnings=np.array(warnings, dtype=object),
    )
    out_path = str(Path(output).resolve())
    log.append(f"Параметры калибровки сохранены: {out_path}")
    return out_path, log


def main() -> None:
    args = parse_args()

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

    try:
        out_path, log = calibrate_stereo(
            pairs,
            args.cols,
            args.rows,
            args.square_size,
            args.output,
            args.debug_dir,
            args.alpha,
            args.rectify,
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
