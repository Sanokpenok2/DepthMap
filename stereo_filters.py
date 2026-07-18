"""RANSAC и DBSCAN для калибровки и постобработки карты диспаритета."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from sklearn.cluster import DBSCAN
    from sklearn.linear_model import RANSACRegressor
    from sklearn.neighbors import NearestNeighbors
except ImportError:  # pragma: no cover
    DBSCAN = None
    RANSACRegressor = None
    NearestNeighbors = None


def filter_correspondences_ransac(
    pts_l: np.ndarray,
    pts_r: np.ndarray,
    threshold: float = 1.0,
    confidence: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, str]:
    """Отбирает соответствия inlier через FM_RANSAC."""
    if len(pts_l) < 8:
        mask = np.ones(len(pts_l), dtype=bool)
        return pts_l, pts_r, None, mask, "RANSAC: слишком мало точек, фильтр пропущен."

    F, mask = cv2.findFundamentalMat(
        pts_l,
        pts_r,
        cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=confidence,
    )
    if F is None or mask is None:
        mask = np.ones(len(pts_l), dtype=bool)
        return pts_l, pts_r, None, mask, "RANSAC: не удалось оценить F, фильтр пропущен."

    inliers = mask.ravel().astype(bool)
    kept = int(inliers.sum())
    log = (
        f"RANSAC (эпиполярный): оставлено {kept}/{len(pts_l)} соответствий "
        f"(порог {threshold:.2f} px)."
    )
    return pts_l[inliers], pts_r[inliers], F, inliers, log


def valid_disparity_mask(
    disp_float: np.ndarray,
    Q: np.ndarray,
    min_disparity: float = 0.5,
    max_depth: float = 1e4,
    max_xy: float = 1e5,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает 3D-точки и маску валидных пикселей диспаритета."""
    points_3d = cv2.reprojectImageTo3D(disp_float, Q)
    valid = (disp_float > min_disparity) & np.isfinite(points_3d).all(axis=2)
    valid &= np.abs(points_3d[:, :, 2]) < max_depth
    valid &= np.abs(points_3d[:, :, :2]).max(axis=2) < max_xy
    return points_3d, valid


def _require_sklearn(name: str) -> None:
    if RANSACRegressor is None or DBSCAN is None or NearestNeighbors is None:
        raise ImportError(
            f"Для {name} установите scikit-learn: pip install scikit-learn"
        )


def filter_disparity_ransac(
    disp_float: np.ndarray,
    Q: np.ndarray,
    threshold: float = 50.0,
    max_samples: int = 20000,
    min_disparity: float = 0.5,
) -> tuple[np.ndarray, str]:
    """Убирает выбросы диспаритета по RANSAC-плоскости Z(X, Y) в 3D."""
    _require_sklearn("RANSAC")
    points_3d, valid = valid_disparity_mask(disp_float, Q, min_disparity)
    ys, xs = np.where(valid)
    if len(xs) < 8:
        return disp_float, "RANSAC диспаритета: слишком мало валидных пикселей."

    rng = np.random.default_rng(42)
    if len(xs) > max_samples:
        pick = rng.choice(len(xs), max_samples, replace=False)
        ys, xs = ys[pick], xs[pick]

    sample = points_3d[ys, xs].astype(np.float64)
    finite = np.isfinite(sample).all(axis=1)
    sample = sample[finite]
    if len(sample) < 8:
        return disp_float, "RANSAC диспаритета: слишком мало конечных 3D-точек."

    ransac = RANSACRegressor(residual_threshold=threshold, random_state=42)
    try:
        ransac.fit(sample[:, :2], sample[:, 2])
    except ValueError:
        return disp_float, "RANSAC диспаритета: не удалось подогнать модель, фильтр пропущен."

    valid_pts = points_3d[valid].astype(np.float64)
    pred = ransac.predict(valid_pts[:, :2])
    if not np.isfinite(pred).all():
        return disp_float, "RANSAC диспаритета: неустойчивое предсказание, фильтр пропущен."

    outliers = np.abs(valid_pts[:, 2] - pred) > threshold

    filtered = disp_float.copy()
    chunk = filtered[valid]
    chunk[outliers] = 0.0
    filtered[valid] = chunk
    removed = int(outliers.sum())
    return filtered, (
        f"RANSAC диспаритета: удалено {removed} пикселей-выбросов "
        f"(порог {threshold:.1f} по глубине)."
    )


def filter_disparity_dbscan(
    disp_float: np.ndarray,
    Q: np.ndarray,
    eps: float = 100.0,
    min_samples: int = 50,
    max_samples: int = 30000,
    min_disparity: float = 0.5,
) -> tuple[np.ndarray, str]:
    """Оставляет главный кластер 3D-точек DBSCAN, шум обнуляет в карте диспаритета."""
    _require_sklearn("DBSCAN")
    points_3d, valid = valid_disparity_mask(disp_float, Q, min_disparity)
    ys, xs = np.where(valid)
    if len(xs) < min_samples:
        return disp_float, "DBSCAN: слишком мало валидных пикселей."

    rng = np.random.default_rng(42)
    if len(xs) > max_samples:
        pick = rng.choice(len(xs), max_samples, replace=False)
        ys_s, xs_s = ys[pick], xs[pick]
    else:
        ys_s, xs_s = ys, xs

    pts = points_3d[ys_s, xs_s].astype(np.float64)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < min_samples:
        return disp_float, "DBSCAN: слишком мало конечных 3D-точек."

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pts).labels_
    uniq, counts = np.unique(labels[labels >= 0], return_counts=True)
    if uniq.size == 0:
        return disp_float, "DBSCAN: кластеры не найдены."

    main_label = uniq[np.argmax(counts)]
    cluster_pts = pts[labels == main_label]
    nn = NearestNeighbors(radius=eps)
    nn.fit(cluster_pts)

    all_pts = points_3d[valid].reshape(-1, 3).astype(np.float64)
    finite_all = np.isfinite(all_pts).all(axis=1)
    if finite_all.sum() < min_samples:
        return disp_float, "DBSCAN: слишком мало конечных валидных точек."

    keep = np.zeros(finite_all.shape, dtype=bool)
    neighbors = nn.radius_neighbors(all_pts[finite_all], return_distance=False)
    keep[finite_all] = np.array([len(idx) > 0 for idx in neighbors])

    filtered = disp_float.copy()
    valid_flat = valid.copy()
    valid_flat.reshape(-1)[np.where(valid.reshape(-1))[0][~keep]] = False
    filtered[~valid_flat] = 0.0
    removed = int(valid.sum() - valid_flat.sum())
    return filtered, (
        f"DBSCAN: оставлен главный кластер ({len(cluster_pts)} точек в выборке), "
        f"удалено {removed} пикселей (eps={eps:.1f})."
    )
