"""Controlled outlier generation and statistical filtering."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .io import PointCloud, validate_points

OutlierMask = NDArray[np.bool_]


@dataclass(frozen=True)
class ContaminatedPointCloud:
    """A point cloud with ground-truth outlier labels."""

    points: PointCloud
    is_outlier: OutlierMask


def inject_vertical_outliers(
    points: NDArray[np.floating],
    outlier_fraction: float = 0.05,
    distance_scale: float = 0.25,
    seed: int = 42,
) -> ContaminatedPointCloud:
    """Inject isolated points above and below a clean point cloud."""
    clean = validate_points(points)
    if not np.isfinite(outlier_fraction) or not 0.0 < outlier_fraction < 1.0:
        raise ValueError("Outlier fraction must be between 0 and 1.")
    if not np.isfinite(distance_scale) or distance_scale <= 0.0:
        raise ValueError("Distance scale must be a positive finite number.")

    outlier_count = max(
        1,
        round(len(clean) * outlier_fraction / (1.0 - outlier_fraction)),
    )
    minimum = clean.min(axis=0)
    maximum = clean.max(axis=0)
    span = maximum - minimum
    vertical_scale = max(
        span[2],
        float(np.linalg.norm(span[:2])) * 0.05,
        np.finfo(np.float64).eps,
    )

    rng = np.random.default_rng(seed)
    outliers = np.empty((outlier_count, 3), dtype=np.float64)
    outliers[:, 0] = rng.uniform(minimum[0], maximum[0], size=outlier_count)
    outliers[:, 1] = rng.uniform(minimum[1], maximum[1], size=outlier_count)
    offsets = (
        vertical_scale
        * distance_scale
        * rng.uniform(1.0, 2.0, size=outlier_count)
    )
    above = rng.random(outlier_count) >= 0.5
    outliers[:, 2] = np.where(
        above,
        maximum[2] + offsets,
        minimum[2] - offsets,
    )

    combined = np.vstack((clean, outliers))
    labels = np.concatenate(
        (
            np.zeros(len(clean), dtype=np.bool_),
            np.ones(outlier_count, dtype=np.bool_),
        )
    )
    permutation = rng.permutation(len(combined))
    return ContaminatedPointCloud(
        points=combined[permutation],
        is_outlier=labels[permutation],
    )


def mean_knn_distances(
    points: NDArray[np.floating], neighbors: int
) -> NDArray[np.float64]:
    """Return each point's mean distance to its nearest neighbors."""
    cloud = validate_points(points)
    if neighbors < 1:
        raise ValueError("Neighbors must be at least 1.")
    if neighbors >= len(cloud):
        raise ValueError("Neighbors must be smaller than the point count.")

    distances, _ = cKDTree(cloud).query(cloud, k=neighbors + 1)
    return np.mean(distances[:, 1:], axis=1)


def statistical_outlier_mask(
    points: NDArray[np.floating],
    neighbors: int,
    std_ratio: float,
) -> tuple[OutlierMask, NDArray[np.float64], float]:
    """Classify points using a global threshold on mean kNN distance."""
    if not np.isfinite(std_ratio) or std_ratio < 0.0:
        raise ValueError("Standard-deviation ratio must be non-negative.")

    scores = mean_knn_distances(points, neighbors)
    threshold = float(np.mean(scores) + std_ratio * np.std(scores))
    return scores > threshold, scores, threshold
