"""Deterministic synthetic point clouds for controlled experiments."""

import numpy as np

from .io import PointCloud


def generate_controlled_density_cloud(
    point_count: int = 6_000, seed: int = 42
) -> PointCloud:
    """Generate a wavy surface with deliberately uneven sampling density."""
    if point_count < 100:
        raise ValueError("Point count must be at least 100.")

    rng = np.random.default_rng(seed)
    dense_count = point_count // 3
    broad_count = point_count - dense_count

    broad_xy = rng.uniform(-10.0, 10.0, size=(broad_count, 2))
    dense_xy = np.clip(
        rng.normal(loc=0.0, scale=1.8, size=(dense_count, 2)),
        -5.0,
        5.0,
    )
    xy = np.vstack((broad_xy, dense_xy))
    z = (
        0.35 * np.sin(0.6 * xy[:, 0])
        + 0.25 * np.cos(0.45 * xy[:, 1])
        + rng.normal(0.0, 0.02, size=point_count)
    )
    return np.column_stack((xy, z))
