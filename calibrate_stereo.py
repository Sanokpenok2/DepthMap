from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np


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
    return p.parse_args()


def find_pattern_size(cols: int, rows: int) -> tuple[int, int]:
    return (cols, rows)


def build_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    """Координаты углов доски в её собственной системе (Z=0)."""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def main() -> None:
    args = parse_args()

    left_files = sorted(glob.glob(args.left))
    right_files = sorted(glob.glob(args.right))

    if not left_files or not right_files:
        sys.exit("Ошибка: не найдены изображения по указанным шаблонам.")
    if len(left_files) != len(right_files):
        sys.exit(
            f"Ошибка: число левых ({len(left_files)}) и правых "
            f"({len(right_files)}) изображений различается."
        )

    pattern = find_pattern_size(args.cols, args.rows)
    objp = build_object_points(args.cols, args.rows, args.square_size)

    objpoints: list[np.ndarray] = []
    imgpoints_l: list[np.ndarray] = []
    imgpoints_r: list[np.ndarray] = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    image_size = None

    if args.debug_dir:
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)

    used = 0
    for lf, rf in zip(left_files, right_files):
        img_l = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
        if img_l is None or img_r is None:
            print(f"Пропуск (не читается): {lf} / {rf}", file=sys.stderr)
            continue
        if img_l.shape != img_r.shape:
            print(f"Пропуск (разные размеры): {lf} / {rf}", file=sys.stderr)
            continue

        image_size = (img_l.shape[1], img_l.shape[0])

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found_l, corners_l = cv2.findChessboardCorners(img_l, pattern, flags)
        found_r, corners_r = cv2.findChessboardCorners(img_r, pattern, flags)

        if not (found_l and found_r):
            print(f"Доска не найдена на паре: {Path(lf).name} / {Path(rf).name}")
            continue

        corners_l = cv2.cornerSubPix(img_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(img_r, corners_r, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)
        used += 1

        if args.debug_dir:
            vis = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern, corners_l, found_l)
            cv2.imwrite(str(Path(args.debug_dir) / f"corners_{Path(lf).name}"), vis)

    if used < 3:
        sys.exit(
            f"Ошибка: доска найдена только на {used} парах. "
            "Нужно минимум 3 (рекомендуется 10-20) с разных ракурсов."
        )

    print(f"Углы найдены на {used} парах. Калибровка отдельных камер...")

    # Калибровка каждой камеры по отдельности.
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_l, image_size, None, None
    )
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_r, image_size, None, None
    )
    print(f"  RMS-ошибка левой камеры:  {ret_l:.4f}")
    print(f"  RMS-ошибка правой камеры: {ret_r:.4f}")

    # Стереокалибровка: ищем R, T между камерами, фиксируя внутренние параметры.
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
    print(f"  RMS-ошибка стереокалибровки: {ret_stereo:.4f}")

    # Ректификация: выравнивание изображений так, чтобы эпиполярные линии
    # стали горизонтальными и совпадающими по строкам.
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2
    )

    np.savez(
        args.output,
        image_size=np.array(image_size),
        mtx_l=mtx_l,
        dist_l=dist_l,
        mtx_r=mtx_r,
        dist_r=dist_r,
        R=R,
        T=T,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi1=np.array(roi1),
        roi2=np.array(roi2),
        map1_l=map1_l,
        map2_l=map2_l,
        map1_r=map1_r,
        map2_r=map2_r,
    )
    print(f"Параметры калибровки сохранены: {Path(args.output).resolve()}")
    print("Теперь используйте их в depth_map.py через --calib.")


if __name__ == "__main__":
    main()
