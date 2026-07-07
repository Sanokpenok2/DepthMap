"""
Построение карты глубины (диспарности) по стереопаре.

Программа принимает левое и правое изображения стереопары и строит
карту диспарности с помощью алгоритма Semi-Global Block Matching (SGBM)
или Block Matching (BM). Опционально применяется WLS-фильтр
(из opencv-contrib) для сглаживания и заполнения "дыр".

Пример запуска:
    python depth_map.py --left left.png --right right.png --output disparity.png
    python depth_map.py -l left.png -r right.png --method sgbm --wls --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Построение карты глубины (диспарности) по стереопаре.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-l", "--left", required=True, help="Путь к левому изображению.")
    p.add_argument("-r", "--right", required=True, help="Путь к правому изображению.")
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
        help="Диапазон диспаритетов (должен быть кратен 16).",
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
        help="Минимальный диспаритет.",
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
    return p.parse_args()


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"Ошибка: не удалось прочитать изображение '{path}'.")
    return img


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
        data = np.load(path, allow_pickle=False)
    except OSError:
        sys.exit(f"Ошибка: не удалось прочитать файл калибровки '{path}'.")
    required = ["map1_l", "map2_l", "map1_r", "map2_r", "Q"]
    missing = [k for k in required if k not in data.files]
    if missing:
        sys.exit(f"Ошибка: в файле калибровки нет полей: {', '.join(missing)}.")
    return {k: data[k] for k in data.files}


def rectify_pair(
    left: np.ndarray, right: np.ndarray, calib: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Применяет карты ремаппинга из калибровки для выравнивания стереопары."""
    rect_l = cv2.remap(left, calib["map1_l"], calib["map2_l"], cv2.INTER_LINEAR)
    rect_r = cv2.remap(right, calib["map1_r"], calib["map2_r"], cv2.INTER_LINEAR)
    return rect_l, rect_r


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

    Диспаритет усредняется по окну window×window (берётся медиана валидных
    значений). Расстояние считается либо по матрице Q (из калибровки), либо
    по формуле depth = focal * baseline / disparity. Если данных не хватает,
    соответствующее значение возвращается как None.
    """
    h, w = disp_float.shape
    if not (0 <= x < w and 0 <= y < h):
        return None, None

    r = max(window // 2, 0)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    patch = disp_float[y0:y1, x0:x1]

    valid = patch[patch > 0]
    if valid.size == 0:
        return None, None
    disp = float(np.median(valid))

    if Q is not None:
        vec = np.array([[x], [y], [disp], [1.0]], dtype=np.float64)
        xyzw = Q @ vec
        wv = xyzw[3, 0]
        if abs(wv) < 1e-9:
            return None, disp
        z = float(xyzw[2, 0] / wv)
        return z, disp
    if focal is not None and baseline is not None:
        return focal * baseline / disp, disp
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


def main() -> None:
    args = parse_args()

    if args.num_disparities % 16 != 0:
        sys.exit("Ошибка: --num-disparities должен быть кратен 16.")
    if args.block_size % 2 == 0:
        sys.exit("Ошибка: --block-size должен быть нечётным.")

    if (args.depth or args.point_cloud) and not args.calib:
        sys.exit("Ошибка: --depth и --point-cloud требуют указания --calib.")

    left = load_gray(args.left)
    right = load_gray(args.right)
    if left.shape != right.shape:
        sys.exit(
            f"Ошибка: размеры изображений различаются "
            f"({left.shape} и {right.shape}). Стереопара должна быть выровнена."
        )

    calib = None
    if args.calib:
        print(f"Загрузка калибровки и ректификация: {args.calib}")
        calib = load_calibration(args.calib)
        left, right = rectify_pair(left, right, calib)

    if args.method == "sgbm":
        matcher = build_sgbm(args.min_disparity, args.num_disparities, args.block_size)
    else:
        matcher = build_bm(args.num_disparities, args.block_size)

    print(f"Вычисление диспаритета методом {args.method.upper()}...")
    disp = matcher.compute(left, right)

    if args.wls:
        print("Применение WLS-фильтра...")
        disp = apply_wls(
            matcher, disp, left, right, args.wls_lambda, args.wls_sigma
        )

    disp_vis = normalize_disparity(disp, args.min_disparity, args.num_disparities)
    disp_color = cv2.applyColorMap(disp_vis, get_colormap(args.colormap))

    out_path = Path(args.output)
    cv2.imwrite(str(out_path), disp_color)
    print(f"Карта глубины сохранена: {out_path.resolve()}")

    disp_float = disp.astype(np.float32) / 16.0

    if args.save_raw:
        np.save(args.save_raw, disp_float)
        print(f"Сырая карта диспаритетов сохранена: {Path(args.save_raw).resolve()}")

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
