"""Проверка качества стереокалибровки."""

from __future__ import annotations

import cv2
import numpy as np

WARN_RMS_MONO = 1.0
WARN_RMS_STEREO = 1.0
WARN_BASELINE_MIN_MM = 10.0
WARN_BASELINE_MAX_MM = 500.0
WARN_FX_DIFF_RATIO = 0.15
WARN_VALID_RECT_FRACTION = 0.85
WARN_PINHOLE_K3 = 1.0
WARN_ROI_AREA_FRACTION = 0.35


def estimate_rect_valid_fraction(
    map1: np.ndarray,
    map2: np.ndarray,
    image_size: tuple[int, int],
) -> float:
    """Доля пикселей, которые попадают в исходный кадр после ремапа."""
    w, h = image_size
    white = np.full((h, w), 255, dtype=np.uint8)
    rect = cv2.remap(
        white,
        map1,
        map2,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return float((rect > 0).mean())


def roi_area_fraction(roi: np.ndarray | tuple[int, int, int, int], image_size: tuple[int, int]) -> float:
    w, h = image_size
    x, y, rw, rh = (int(v) for v in roi)
    return (rw * rh) / max(w * h, 1)


def assess_calibration_quality(
    *,
    model: str,
    rms_l: float,
    rms_r: float,
    rms_stereo: float,
    mtx_l: np.ndarray,
    mtx_r: np.ndarray,
    baseline_mm: float,
    map1_l: np.ndarray,
    map2_l: np.ndarray,
    map1_r: np.ndarray,
    map2_r: np.ndarray,
    image_size: tuple[int, int],
    alpha: float,
    roi1: np.ndarray | tuple[int, int, int, int] | None = None,
    roi2: np.ndarray | tuple[int, int, int, int] | None = None,
    dist_l: np.ndarray | None = None,
    dist_r: np.ndarray | None = None,
) -> list[str]:
    """Возвращает список предупреждений о качестве калибровки."""
    warnings: list[str] = []

    if np.isfinite(rms_l) and rms_l > WARN_RMS_MONO:
        warnings.append(
            f"Высокая RMS-ошибка левой камеры ({rms_l:.2f} px; норма < {WARN_RMS_MONO:.1f})."
        )
    if np.isfinite(rms_r) and rms_r > WARN_RMS_MONO:
        warnings.append(
            f"Высокая RMS-ошибка правой камеры ({rms_r:.2f} px; норма < {WARN_RMS_MONO:.1f})."
        )
    if np.isfinite(rms_stereo) and rms_stereo > WARN_RMS_STEREO:
        warnings.append(
            f"Высокая RMS-ошибка стереопары ({rms_stereo:.2f} px; норма < {WARN_RMS_STEREO:.1f})."
        )

    fx_l = float(mtx_l[0, 0])
    fx_r = float(mtx_r[0, 0])
    fx_mean = max((fx_l + fx_r) / 2.0, 1e-6)
    if abs(fx_l - fx_r) / fx_mean > WARN_FX_DIFF_RATIO:
        warnings.append(
            f"Сильно различаются focal левой и правой камер "
            f"({fx_l:.0f} vs {fx_r:.0f} px)."
        )

    if baseline_mm < WARN_BASELINE_MIN_MM or baseline_mm > WARN_BASELINE_MAX_MM:
        warnings.append(
            f"Подозрительная база камер ({baseline_mm:.1f} мм). "
            "Проверьте --square-size и качество углов доски."
        )

    valid_l = estimate_rect_valid_fraction(map1_l, map2_l, image_size)
    valid_r = estimate_rect_valid_fraction(map1_r, map2_r, image_size)
    if valid_l < WARN_VALID_RECT_FRACTION:
        warnings.append(
            f"После ректификации левый кадр заполнен только на {valid_l * 100:.0f}% "
            f"(норма >= {WARN_VALID_RECT_FRACTION * 100:.0f}%)."
        )
    if valid_r < WARN_VALID_RECT_FRACTION:
        warnings.append(
            f"После ректификации правый кадр заполнен только на {valid_r * 100:.0f}% "
            f"(норма >= {WARN_VALID_RECT_FRACTION * 100:.0f}%)."
        )

    if roi1 is not None and roi_area_fraction(roi1, image_size) < WARN_ROI_AREA_FRACTION:
        warnings.append(
            f"Маленькая общая область без чёрных полей слева: {tuple(int(v) for v in roi1)}."
        )
    if roi2 is not None and roi_area_fraction(roi2, image_size) < WARN_ROI_AREA_FRACTION:
        warnings.append(
            f"Маленькая общая область без чёрных полей справа: {tuple(int(v) for v in roi2)}."
        )

    if alpha >= 0.99 and (valid_l < WARN_VALID_RECT_FRACTION or valid_r < WARN_VALID_RECT_FRACTION):
        warnings.append(
            "При alpha=1 на краях возможны сильные искажения. "
            "Попробуйте alpha=0.5 или перекалибруйте с --model fisheye."
        )

    if model == "pinhole" and dist_l is not None and dist_r is not None:
        k3_l = abs(float(dist_l.ravel()[4])) if dist_l.size >= 5 else 0.0
        k3_r = abs(float(dist_r.ravel()[4])) if dist_r.size >= 5 else 0.0
        if k3_l > WARN_PINHOLE_K3 or k3_r > WARN_PINHOLE_K3:
            warnings.append(
                "Сильная дисторсия для pinhole-модели "
                f"(|k3|: {k3_l:.2f} / {k3_r:.2f}). "
                "Для широкоугольных объективов используйте --model fisheye."
            )

    return warnings


def format_quality_report(warnings: list[str]) -> list[str]:
    if not warnings:
        return ["Проверка качества: замечаний нет."]
    lines = ["Проверка качества: обнаружены проблемы:"]
    lines.extend(f"  ! {w}" for w in warnings)
    return lines
