from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

from calib_quality import assess_calibration_quality, format_quality_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Стереокалибровка по шахматной доске.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--left",
        required=True,
        help="Glob-шаблон путей к левым изображениям (в кавычках), напр. 'calib/left_*.png'.",
    )
    p.add_argument(
        "--right",
        required=True,
        help="Glob-шаблон путей к правым изображениям (в кавычках).",
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
            "Масштаб ректификации (0..1): 0 — максимальная обрезка без чёрных полей, "
            "1 — сохранить весь кадр (возможны чёрные края). "
            "Для --model fisheye используется как balance."
        ),
    )
    p.add_argument(
        "--model",
        choices=["pinhole", "fisheye"],
        default="pinhole",
        help="Модель объектива: pinhole (стандартная) или fisheye (широкоугольная).",
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
) -> tuple[list[str], float, float]:
    """Возвращает строки журнала, focal (px) и baseline (мм)."""
    fx_l, fy_l = float(mtx_l[0, 0]), float(mtx_l[1, 1])
    fx_r, fy_r = float(mtx_r[0, 0]), float(mtx_r[1, 1])
    baseline_mm = float(np.linalg.norm(T))
    focal_px = float(P1[0, 0])
    lines = [
        "Рассчитанные параметры:",
        f"  Левая камера:  fx={fx_l:.2f} px, fy={fy_l:.2f} px",
        f"  Правая камера: fx={fx_r:.2f} px, fy={fy_r:.2f} px",
        f"  База между камерами: {baseline_mm:.2f} мм",
        f"  Фокусное (после ректификации): {focal_px:.2f} px",
        (
            "  Для depth_map без --calib: "
            f"--focal {focal_px:.1f} --baseline {baseline_mm:.1f}"
        ),
    ]
    return lines, focal_px, baseline_mm


def collect_calibration_corners(
    left_paths: list[str],
    right_paths: list[str],
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
    pattern = find_pattern_size(cols, rows)
    objp = build_object_points(cols, rows, square_size)

    objpoints: list[np.ndarray] = []
    imgpoints_l: list[np.ndarray] = []
    imgpoints_r: list[np.ndarray] = []

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    image_size: tuple[int, int] | None = None
    log: list[str] = []

    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    for lf, rf in zip(left_paths, right_paths):
        img_l = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
        if img_l is None or img_r is None:
            log.append(f"Пропуск (не читается): {Path(lf).name} / {Path(rf).name}")
            continue
        if img_l.shape != img_r.shape:
            log.append(f"Пропуск (разные размеры): {Path(lf).name} / {Path(rf).name}")
            continue

        image_size = (img_l.shape[1], img_l.shape[0])

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found_l, corners_l = cv2.findChessboardCorners(img_l, pattern, flags)
        found_r, corners_r = cv2.findChessboardCorners(img_r, pattern, flags)

        if not (found_l and found_r):
            log.append(f"Доска не найдена: {Path(lf).name} / {Path(rf).name}")
            continue

        corners_l = cv2.cornerSubPix(img_l, corners_l, (11, 11), (-1, -1), subpix_criteria)
        corners_r = cv2.cornerSubPix(img_r, corners_r, (11, 11), (-1, -1), subpix_criteria)

        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

        if debug_dir:
            vis_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis_l, pattern, corners_l, found_l)
            cv2.imwrite(str(Path(debug_dir) / f"corners_left_{Path(lf).name}"), vis_l)

            vis_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis_r, pattern, corners_r, found_r)
            cv2.imwrite(str(Path(debug_dir) / f"corners_right_{Path(rf).name}"), vis_r)

    if image_size is None:
        raise ValueError("Не удалось прочитать ни одной пары изображений.")
    if len(objpoints) < 3:
        raise ValueError(
            f"Доска найдена только на {len(objpoints)} парах. "
            "Нужно минимум 3 (рекомендуется 10-20) с разных ракурсов."
        )

    log.append(f"Углы найдены на {len(objpoints)} парах.")
    return objpoints, imgpoints_l, imgpoints_r, image_size, log


def calibrate_pinhole(
    objpoints: list[np.ndarray],
    imgpoints_l: list[np.ndarray],
    imgpoints_r: list[np.ndarray],
    image_size: tuple[int, int],
    alpha: float,
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
        _F,
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

    alpha = float(np.clip(alpha, 0.0, 1.0))
    log.append(f"Ректификация: alpha={alpha:.2f}")

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
    }


def calibrate_fisheye(
    objpoints: list[np.ndarray],
    imgpoints_l: list[np.ndarray],
    imgpoints_r: list[np.ndarray],
    image_size: tuple[int, int],
    alpha: float,
    log: list[str],
) -> dict:
    log.append("Калибровка fisheye-модели...")

    objpoints_f = [objp.reshape(1, -1, 3) for objp in objpoints]
    imgpoints_l_f = [pts.reshape(1, -1, 2) for pts in imgpoints_l]
    imgpoints_r_f = [pts.reshape(1, -1, 2) for pts in imgpoints_r]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # Сначала pinhole-оценка как стартовое приближение для fisheye.
    _, mtx_init_l, _, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_l, image_size, None, None
    )
    _, mtx_init_r, _, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_r, image_size, None, None
    )

    K_l = mtx_init_l.copy()
    K_r = mtx_init_r.copy()
    D_l = np.zeros((4, 1), dtype=np.float64)
    D_r = np.zeros((4, 1), dtype=np.float64)

    ret_l, K_l, D_l, _, _ = cv2.fisheye.calibrate(
        objpoints_f,
        imgpoints_l_f,
        image_size,
        K_l,
        D_l,
        None,
        None,
        cv2.CALIB_USE_INTRINSIC_GUESS,
        criteria,
    )
    ret_r, K_r, D_r, _, _ = cv2.fisheye.calibrate(
        objpoints_f,
        imgpoints_r_f,
        image_size,
        K_r,
        D_r,
        None,
        None,
        cv2.CALIB_USE_INTRINSIC_GUESS,
        criteria,
    )
    log.append(f"  RMS-ошибка левой камеры:  {ret_l:.4f}")
    log.append(f"  RMS-ошибка правой камеры: {ret_r:.4f}")

    stereo_result = cv2.fisheye.stereoCalibrate(
        objpoints_f,
        imgpoints_l_f,
        imgpoints_r_f,
        K_l,
        D_l,
        K_r,
        D_r,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=criteria,
    )
    ret_stereo = float(stereo_result[0])
    K_l, D_l, K_r, D_r = stereo_result[1:5]
    R, T = stereo_result[5:7]
    log.append(f"  RMS-ошибка стереокалибровки: {ret_stereo:.4f}")

    balance = float(np.clip(alpha, 0.0, 1.0))
    log.append(f"Ректификация: balance={balance:.2f}")

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K_l,
        D_l,
        K_r,
        D_r,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=image_size,
        balance=balance,
        fov_scale=1.0,
    )
    roi1 = (0, 0, image_size[0], image_size[1])
    roi2 = (0, 0, image_size[0], image_size[1])
    log.append("  Fisheye-ректификация не возвращает ROI; проверяйте % валидных пикселей.")

    map1_l, map2_l = cv2.fisheye.initUndistortRectifyMap(
        K_l, D_l, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_r, map2_r = cv2.fisheye.initUndistortRectifyMap(
        K_r, D_r, R2, P2, image_size, cv2.CV_16SC2
    )

    return {
        "model": "fisheye",
        "rms_l": ret_l,
        "rms_r": ret_r,
        "rms_stereo": ret_stereo,
        "mtx_l": K_l,
        "dist_l": D_l,
        "mtx_r": K_r,
        "dist_r": D_r,
        "R": R,
        "T": T,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "alpha": balance,
        "roi1": roi1,
        "roi2": roi2,
        "map1_l": map1_l,
        "map2_l": map2_l,
        "map1_r": map1_r,
        "map2_r": map2_r,
    }


def calibrate_stereo(
    left_paths: list[str],
    right_paths: list[str],
    cols: int,
    rows: int,
    square_size: float,
    output: str,
    debug_dir: str | None = None,
    alpha: float = 1.0,
    model: str = "pinhole",
) -> tuple[str, list[str]]:
    """Выполняет стереокалибровку и сохраняет результат в .npz."""
    if not left_paths or not right_paths:
        raise ValueError("Не найдены изображения для калибровки.")
    if len(left_paths) != len(right_paths):
        raise ValueError(
            f"Число левых ({len(left_paths)}) и правых "
            f"({len(right_paths)}) изображений различается."
        )

    objpoints, imgpoints_l, imgpoints_r, image_size, log = collect_calibration_corners(
        left_paths, right_paths, cols, rows, square_size, debug_dir
    )

    if model == "fisheye":
        result = calibrate_fisheye(objpoints, imgpoints_l, imgpoints_r, image_size, alpha, log)
    else:
        result = calibrate_pinhole(objpoints, imgpoints_l, imgpoints_r, image_size, alpha, log)

    geom_lines, focal_px, baseline_mm = describe_stereo_geometry(
        result["mtx_l"], result["mtx_r"], result["T"], result["P1"]
    )
    log.extend(geom_lines)

    warnings = assess_calibration_quality(
        model=result["model"],
        rms_l=result["rms_l"],
        rms_r=result["rms_r"],
        rms_stereo=result["rms_stereo"],
        mtx_l=result["mtx_l"],
        mtx_r=result["mtx_r"],
        baseline_mm=baseline_mm,
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
        image_size=np.array(image_size),
        mtx_l=result["mtx_l"],
        dist_l=result["dist_l"],
        mtx_r=result["mtx_r"],
        dist_r=result["dist_r"],
        R=result["R"],
        T=result["T"],
        R1=result["R1"],
        R2=result["R2"],
        P1=result["P1"],
        P2=result["P2"],
        Q=result["Q"],
        alpha=np.array([result["alpha"]]),
        focal_px=np.array([focal_px]),
        baseline_mm=np.array([baseline_mm]),
        roi1=np.array(result["roi1"]),
        roi2=np.array(result["roi2"]),
        map1_l=result["map1_l"],
        map2_l=result["map2_l"],
        map1_r=result["map1_r"],
        map2_r=result["map2_r"],
        quality_warnings=np.array(warnings, dtype=object),
    )
    out_path = str(Path(output).resolve())
    log.append(f"Параметры калибровки сохранены: {out_path}")
    return out_path, log


def main() -> None:
    args = parse_args()

    left_files = sorted(glob.glob(args.left))
    right_files = sorted(glob.glob(args.right))

    try:
        out_path, log = calibrate_stereo(
            left_files,
            right_files,
            args.cols,
            args.rows,
            args.square_size,
            args.output,
            args.debug_dir,
            args.alpha,
            args.model,
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
