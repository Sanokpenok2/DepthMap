"""Автоматический подбор параметров стереосопоставления и измерения глубины."""

from __future__ import annotations

import cv2
import numpy as np

from stereo_filters import valid_disparity_mask


def extract_calib_geometry(calib: dict) -> tuple[float, float]:
    """Возвращает (focal_px, baseline_mm) из файла калибровки."""
    if "focal_rect_l_px" in calib:
        focal = float(np.asarray(calib["focal_rect_l_px"]).ravel()[0])
    else:
        focal = float(calib["P1"][0, 0])
    if "baseline_mm" in calib:
        baseline = float(np.asarray(calib["baseline_mm"]).ravel()[0])
    else:
        baseline = float(np.linalg.norm(calib["T"]))
    return focal, baseline


def disparity_from_depth(focal_px: float, baseline_mm: float, depth_mm: float) -> float:
    return focal_px * baseline_mm / max(depth_mm, 1.0)


def round_num_disparities(span: float, *, min_val: int = 16, max_val: int = 1024) -> int:
    rounded = int(np.ceil(max(span, min_val) / 16) * 16)
    return int(np.clip(rounded, min_val, max_val))


def clamp_sgbm_range(
    min_disp: int,
    num_disp: int,
    image_width: int,
    *,
    max_num: int = 512,
) -> tuple[int, int]:
    """Жёсткий потолок для SGBM: min+num < width, иначе OpenCV ловит OOM/underflow."""
    width = max(int(image_width), 32)
    md = max(0, int(min_disp))
    nd = round_num_disparities(num_disp, min_val=16, max_val=max_num)
    # Рабочая зона справа от мёртвой полосы слева.
    hard_cap = max(16, ((width - md - 32) // 16) * 16)
    # Память SGBM растёт ~O(width * height * numDisp) — не больше ~1/3 ширины.
    mem_cap = max(16, ((width // 3) // 16) * 16)
    nd = min(nd, hard_cap, mem_cap, max_num)
    nd = max(16, nd)
    if md + nd >= width:
        md = 0
        nd = min(nd, max(16, ((width // 4) // 16) * 16))
    return md, nd


def constrain_disparity_range(
    min_disp: int,
    num_disp: int,
    image_width: int,
    *,
    max_invalid_fraction: float = 0.18,
    scene_upper_px: float | None = None,
    min_working_fraction: float = 0.55,
) -> tuple[int, int]:
    """Ограничивает min+num, чтобы слева оставалась широкая рабочая зона SGBM."""
    if image_width <= 0:
        return min_disp, num_disp

    base_zone = max(int(image_width * max_invalid_fraction), 72)
    max_zone = base_zone
    if scene_upper_px is not None and scene_upper_px > 0:
        needed = int(round(scene_upper_px + 20.0))
        max_zone = max(base_zone, min(needed, int(image_width * 0.24)))

    min_working = max(int(image_width * min_working_fraction), 160)
    max_zone = min(max_zone, image_width - min_working)
    max_zone = max(max_zone, base_zone)

    md, nd = int(min_disp), int(num_disp)
    nd = round_num_disparities(nd, min_val=32)

    while md + nd > max_zone:
        if nd > 48:
            nd -= 16
        elif md > 0:
            md = max(0, md - max(4, md // 8))
        else:
            nd -= 16
        nd = max(32, round_num_disparities(nd, min_val=32))

    return md, nd


def estimate_disparity_range(
    calib: dict,
    *,
    z_center_mm: float = 1500.0,
    span_ratio: float = 0.55,
    margin: float = 0.12,
) -> tuple[int, int, str]:
    """Оценивает min_disparity и num_disparities по калибровке и типичной сцене."""
    focal, baseline = extract_calib_geometry(calib)
    d_center = disparity_from_depth(focal, baseline, z_center_mm)
    half_span = max(d_center * span_ratio, 24.0)
    d_min = max(1.0, d_center - half_span)
    d_max = d_center + half_span
    min_disp = max(0, int(np.floor(d_min * (1.0 - margin))))
    num_disp = round_num_disparities(d_max * (1.0 + margin) - min_disp)
    z_near = focal * baseline / d_max
    z_far = focal * baseline / max(d_min, 1.0)
    log = (
        f"Авто-диапазон по калибровке (f={focal:.1f} px, B={baseline:.1f} мм): "
        f"min_disparity={min_disp}, num_disparities={num_disp} "
        f"(центр ~{z_center_mm:.0f} мм, глубина ~{z_near:.0f}–{z_far:.0f} мм)."
    )
    return min_disp, num_disp, log


def estimate_disparity_range_bounds(
    calib: dict,
    z_near_m: float,
    z_far_m: float,
    *,
    margin: float = 0.15,
    image_width: int | None = None,
    keep_far_at_zero: bool = True,
) -> tuple[int, int, str]:
    """Подбирает min/num_disparities под заданный диапазон дистанций (метры).

    Ближе объект → больше диспаритет: z_near задаёт верхнюю границу поиска,
    z_far — нижнюю. При keep_far_at_zero min_disparity=0, чтобы дальние объекты
    не обрезались.
    """
    if z_near_m <= 0 or z_far_m <= 0 or z_near_m >= z_far_m:
        raise ValueError("Нужно 0 < z_near_m < z_far_m (дистанции в метрах).")

    focal, baseline = extract_calib_geometry(calib)
    z_near_mm = z_near_m * 1000.0
    z_far_mm = z_far_m * 1000.0
    d_near = disparity_from_depth(focal, baseline, z_near_mm)  # большой
    d_far = disparity_from_depth(focal, baseline, z_far_mm)  # маленький

    if keep_far_at_zero:
        min_disp = 0
    else:
        min_disp = max(0, int(np.floor(d_far * (1.0 - margin))))
    span = d_near * (1.0 + margin) - float(min_disp)
    num_disp = round_num_disparities(max(span, 32.0), min_val=32, max_val=512)

    if image_width is not None:
        # Для дальних сцен (10–40 м) нужна широкая полоса поиска — не режем жёстко.
        max_invalid = 0.42 if z_far_m >= 15.0 else 0.28
        min_working = 0.35 if z_far_m >= 15.0 else 0.45
        min_disp, num_disp = constrain_disparity_range(
            min_disp,
            num_disp,
            image_width,
            scene_upper_px=d_near * (1.0 + margin),
            max_invalid_fraction=max_invalid,
            min_working_fraction=min_working,
        )
        min_disp, num_disp = clamp_sgbm_range(
            min_disp, num_disp, image_width, max_num=512
        )

    z_cov_near = focal * baseline / max(float(min_disp + num_disp), 1.0)
    z_cov_far = (
        focal * baseline / max(float(min_disp), 1.0)
        if min_disp > 0
        else float("inf")
    )
    far_txt = f"{z_cov_far / 1000.0:.1f}" if np.isfinite(z_cov_far) else "inf"
    log = (
        f"Авто-диапазон под {z_near_m:.1f}-{z_far_m:.1f} м "
        f"(f={focal:.1f} px, B={baseline:.1f} мм): "
        f"d~{d_far:.1f}..{d_near:.1f} px -> "
        f"min_disparity={min_disp}, num_disparities={num_disp} "
        f"(покрытие ~{z_cov_near / 1000.0:.1f}-{far_txt} м)."
    )
    return min_disp, num_disp, log


def split_near_far_bands(
    calib: dict,
    z_near_m: float,
    z_far_m: float,
    *,
    margin: float = 0.15,
    image_width: int | None = None,
) -> tuple[tuple[int, int], tuple[int, int], str]:
    """Два перекрывающихся диапазона: дальний (малый d) и ближний (большой d).

    Ближняя полоса специально начинается рано (от ~d на z_far), иначе объект
    на ~10 м оказывается ниже near_min, far-полоса даёт ложный малый d → «40 м».
    """
    focal, baseline = extract_calib_geometry(calib)
    d_near = disparity_from_depth(focal, baseline, z_near_m * 1000.0)
    d_far = disparity_from_depth(focal, baseline, z_far_m * 1000.0)
    # Граница полос ~15–18 м (или геометрическая середина, что ближе).
    z_split = min(max(0.5 * (z_near_m + z_far_m), z_near_m * 1.6), z_far_m * 0.7)
    d_split = disparity_from_depth(focal, baseline, z_split * 1000.0)

    # Дальний проход: только малые d (не забираем зону ближних объектов).
    far_min = 0
    far_num = round_num_disparities(
        max(d_split * (1.0 + margin), d_far * 2.0, 48.0),
        min_val=48,
        max_val=192,
    )

    # Ближний: от чуть ниже d_split до d_near — с запасом, но в лимите SGBM.
    near_min = max(0, int(np.floor(min(d_split, d_far * 1.5) * 0.55)))
    need_upper = d_near * (1.0 + margin)
    near_num = round_num_disparities(
        max(need_upper - float(near_min), 80.0), min_val=80, max_val=512
    )

    if image_width is not None:
        far_min, far_num = constrain_disparity_range(
            far_min,
            far_num,
            image_width,
            scene_upper_px=d_split * 1.15,
            max_invalid_fraction=0.22,
            min_working_fraction=0.5,
        )
        far_min, far_num = clamp_sgbm_range(far_min, far_num, image_width, max_num=192)
        # Сначала пытаемся покрыть z_near, затем жёстко клампим под ширину кадра.
        if near_min + near_num < need_upper:
            near_num = round_num_disparities(
                need_upper - float(near_min), min_val=80, max_val=512
            )
        near_min, near_num = clamp_sgbm_range(
            near_min, near_num, image_width, max_num=512
        )
    else:
        far_min, far_num = clamp_sgbm_range(far_min, far_num, 2048, max_num=192)
        near_min, near_num = clamp_sgbm_range(near_min, near_num, 2048, max_num=512)

    cov_near_m = (
        focal * baseline / max(float(near_min + near_num), 1.0) / 1000.0
    )
    note = ""
    if near_min + near_num + 1.0 < need_upper:
        note = (
            f" Внимание: near-полоса обрезана шириной кадра "
            f"(покрытие с ~{cov_near_m:.1f} м, запрошено от {z_near_m:.1f} м)."
        )

    log = (
        f"Двухполосный SGBM под {z_near_m:.1f}-{z_far_m:.1f} м: "
        f"far=[{far_min}, +{far_num}), near=[{near_min}, +{near_num}) "
        f"(d_far~{d_far:.1f}, d_split~{d_split:.1f}, d_near~{d_near:.1f} px)."
        f"{note}"
    )
    return (far_min, far_num), (near_min, near_num), log


def fuse_disparity_maps(
    disp_far: np.ndarray,
    disp_near: np.ndarray,
    *,
    split_disp: float,
) -> np.ndarray:
    """Склеивает дальнюю и ближнюю карты.

    При конфликте предпочитаем БОЛЬШИЙ диспаритет (ближе): ложный малый d
    из far-полосы на ближнем объекте иначе даёт завышенную дистанцию (10 м → 40 м).
    """
    far = disp_far.astype(np.float32)
    near = disp_near.astype(np.float32)
    out = np.zeros_like(far, dtype=np.float32)

    far_ok = far > 0.5
    near_ok = near > 0.5
    both = far_ok & near_ok

    rel_tol = max(2.5, 0.12 * max(split_disp, 1.0))
    agree = both & (np.abs(far - near) <= rel_tol)
    out[agree] = 0.5 * (far[agree] + near[agree])

    # Конфликт: берём больший d (ближе к камере).
    conflict = both & ~agree
    near_wins = conflict & (near >= far)
    far_wins = conflict & (far > near)
    out[near_wins] = near[near_wins]
    out[far_wins] = far[far_wins]

    # Только near — всегда приоритетнее остатка far.
    only_near = near_ok & (out <= 0)
    out[only_near] = near[only_near]

    # Far только там, где near пуст и d похож на «дальнюю» зону.
    only_far = far_ok & (out <= 0) & (far <= split_disp * 1.35)
    out[only_far] = far[only_far]

    # Если far дал крупный d, а near пуст — всё же взять far (редкий случай).
    far_large = far_ok & (out <= 0) & (far > split_disp * 1.35)
    out[far_large] = far[far_large]
    return out


def refine_disparity_range(
    disp_float: np.ndarray,
    min_disp: int,
    num_disp: int,
    *,
    calib: dict | None = None,
    image_width: int | None = None,
    margin: float = 0.12,
    min_valid: int = 200,
) -> tuple[int, int, str | None]:
    """Уточняет диапазон диспаритета по гистограмме первого прохода."""
    valid = disp_float[disp_float > 0]
    if valid.size < min_valid:
        if calib is not None:
            hint_min, hint_num, _ = estimate_disparity_range(calib)
            if image_width is not None:
                hint_min, hint_num = constrain_disparity_range(
                    hint_min, hint_num, image_width
                )
            return hint_min, hint_num, (
                f"Диапазон по калибровке: min_disparity={hint_min}, "
                f"num_disparities={hint_num}."
            )
        return min_disp, num_disp, None

    p5, p50, p90, p95 = np.percentile(valid, [5, 50, 90, 95])
    if p50 < 2.0 and valid.size < disp_float.size * 0.12:
        keep_num = round_num_disparities(max(num_disp, 128))
        if image_width is not None:
            _, keep_num = constrain_disparity_range(
                min_disp, keep_num, image_width
            )
        return min_disp, keep_num, (
            f"Пробный проход дал мало данных (медиана {p50:.1f} px); "
            f"сохранён широкий диапазон: min_disparity={min_disp}, "
            f"num_disparities={keep_num}."
        )

    if p95 <= p5 + 1.0:
        return min_disp, num_disp, None

    span = max(float(p90 - p5), 24.0)
    if p95 < 16.0 or span < 16.0:
        keep_num = round_num_disparities(max(num_disp, 128))
        if image_width is not None:
            _, keep_num = constrain_disparity_range(
                min_disp, keep_num, image_width, scene_upper_px=p95
            )
        return min_disp, keep_num, (
            f"Пробный проход неоднозначен (p95={p95:.1f} px); "
            f"сохранён широкий диапазон: min_disparity={min_disp}, "
            f"num_disparities={keep_num}."
        )

    bimodal = (p95 - p50) > max(24.0, (p50 - p5) * 0.75)
    new_min = 0
    if p5 > 48.0 and p5 > p50 * 0.35:
        new_min = max(0, int(np.floor(p5 - 0.08 * span)))
    upper = float(p90 + span * 0.08 + 12.0)
    new_num = round_num_disparities(max(upper - new_min, span * 0.55) + 8.0)
    if image_width is not None:
        new_min, new_num = constrain_disparity_range(
            new_min, new_num, image_width, scene_upper_px=upper
        )
    if new_num < 32:
        return min_disp, num_disp, None

    dead_zone = new_min + new_num
    zone_note = ""
    if image_width is not None and image_width > 0:
        zone_note = (
            f", мёртвая зона слева ~{dead_zone} px "
            f"({dead_zone / image_width * 100:.0f}% ширины)"
        )
        if bimodal and dead_zone < int(p95 - p5):
            zone_note += (
                "; дальние объекты могут быть обрезаны по диапазону — "
                "укажите --num-disparities вручную для полной сцены"
            )

    log = (
        f"Уточнён диапазон по сцене: min_disparity={new_min}, "
        f"num_disparities={new_num} (p5={p5:.1f}, медиана {p50:.1f}, p95={p95:.1f} px"
        f"{zone_note})."
    )
    return new_min, new_num, log


def coarse_search_range(calib: dict | None) -> tuple[int, int, str]:
    """Широкий диапазон для первого прохода SGBM."""
    if calib is None:
        return 0, 256, "Пробный проход: min_disparity=0, num_disparities=256."
    return 0, 256, "Пробный проход: min_disparity=0, num_disparities=256."


def filter_disparity_auto(
    disp_float: np.ndarray,
    Q: np.ndarray,
    *,
    mad_factor: float = 5.0,
    min_disparity: float = 0.5,
) -> tuple[np.ndarray, str]:
    """Убирает выбросы по робастному отклонению глубины (MAD)."""
    points_3d, valid = valid_disparity_mask(disp_float, Q, min_disparity)
    if valid.sum() < 50:
        return disp_float, "Автофильтр: слишком мало валидных пикселей."

    depths = points_3d[:, :, 2][valid]
    disp_vals = disp_float[valid]
    spread = float(np.percentile(disp_vals, 95) - np.percentile(disp_vals, 5))
    if spread < 8.0:
        return disp_float, "Автофильтр: пропущен (малый разброс диспаритета)."

    med = float(np.median(depths))
    mad = float(np.median(np.abs(depths - med)))
    if mad < 1e-6:
        return disp_float, "Автофильтр: глубина почти постоянна, фильтр пропущен."

    thresh = mad_factor * mad * 1.4826
    outliers = valid & (np.abs(points_3d[:, :, 2] - med) > thresh)
    removed = int(outliers.sum())
    if removed > int(valid.sum() * 0.75):
        return disp_float, "Автофильтр: пропущен (удалило бы слишком много данных)."
    filtered = disp_float.copy()
    filtered[outliers] = 0.0
    return filtered, (
        f"Автофильтр глубины: удалено {removed} выбросов "
        f"(MAD={mad:.1f}, порог={thresh:.1f})."
    )


def auto_ransac_threshold(disp_float: np.ndarray, Q: np.ndarray) -> float:
    """Подбирает порог RANSAC по разбросу глубины на сцене."""
    _points_3d, valid = valid_disparity_mask(disp_float, Q)
    if valid.sum() < 50:
        return 50.0
    depths = _points_3d[:, :, 2][valid]
    mad = float(np.median(np.abs(depths - np.median(depths))))
    return float(np.clip(max(mad * 4.0, 20.0), 20.0, 500.0))


def auto_dbscan_eps(disp_float: np.ndarray, Q: np.ndarray) -> float:
    """Подбирает eps DBSCAN по типичному шагу между соседними точками."""
    points_3d, valid = valid_disparity_mask(disp_float, Q)
    pts = points_3d[valid].reshape(-1, 3)
    if len(pts) < 100:
        return 100.0
    rng = np.random.default_rng(42)
    pick = rng.choice(len(pts), min(500, len(pts)), replace=False)
    sample = pts[pick]
    dists = np.linalg.norm(sample[1:] - sample[:-1], axis=1)
    base = float(np.median(dists)) if dists.size else 50.0
    return float(np.clip(base * 8.0, 30.0, 400.0))


def depth_from_disparity(
    Q: np.ndarray,
    x: int,
    y: int,
    disparity: float,
) -> float | None:
    vec = np.array([[x], [y], [disparity], [1.0]], dtype=np.float64)
    xyzw = Q @ vec
    wv = xyzw[3, 0]
    if abs(wv) < 1e-9:
        return None
    z = float(xyzw[2, 0] / wv)
    if not np.isfinite(z) or z <= 0 or z > 1e5:
        return None
    return z


def robust_measure_depth(
    disp_float: np.ndarray,
    x: int,
    y: int,
    window: int,
    Q: np.ndarray | None,
    focal: float | None,
    baseline: float | None,
) -> tuple[float | None, float | None]:
    """Робастное измерение расстояния по окну вокруг точки."""
    h, w = disp_float.shape
    if not (0 <= x < w and 0 <= y < h):
        return None, None

    r = max(window // 2, 0)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    patch = disp_float[y0:y1, x0:x1]

    depths: list[float] = []
    disps: list[float] = []
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            d = float(patch[yy - y0, xx - x0])
            if d <= 0:
                continue
            if Q is not None:
                z = depth_from_disparity(Q, xx, yy, d)
            elif focal is not None and baseline is not None:
                z = focal * baseline / d
            else:
                disps.append(d)
                continue
            if z is not None:
                depths.append(z)
                disps.append(d)

    if not depths and not disps:
        return None, None
    if not depths:
        disp = float(np.median(disps))
        if focal is not None and baseline is not None:
            return focal * baseline / disp, disp
        return None, disp

    depth_arr = np.asarray(depths, dtype=np.float64)
    disp_arr = np.asarray(disps, dtype=np.float64)
    q1, q3 = np.percentile(depth_arr, [25, 75])
    iqr = q3 - q1
    if iqr > 1e-6:
        keep = (depth_arr >= q1 - 1.5 * iqr) & (depth_arr <= q3 + 1.5 * iqr)
    else:
        keep = np.ones(depth_arr.shape, dtype=bool)
    if not keep.any():
        keep = np.ones(depth_arr.shape, dtype=bool)

    depth_k = depth_arr[keep]
    disp_k = disp_arr[keep]
    # При двух модах (человек + фон) медиана часто уезжает на фон.
    # Если разброс большой — берём ближнюю поверхность (больший диспаритет / меньшая Z).
    if depth_k.size >= 6 and (depth_k.max() - depth_k.min()) > max(500.0, 0.35 * np.median(depth_k)):
        return float(np.percentile(depth_k, 25)), float(np.percentile(disp_k, 75))
    return float(np.median(depth_k)), float(np.median(disp_k))
