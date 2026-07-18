"""Deterministic synthetic point clouds for controlled experiments."""

import numpy as np
from numpy.typing import NDArray

from .io import PointCloud, validate_points


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


def controlled_surface_normals(
    points: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return analytic upward normals for the noise-free wavy surface."""
    array = validate_points(points)
    dz_dx = 0.21 * np.cos(0.6 * array[:, 0])
    dz_dy = -0.1125 * np.sin(0.45 * array[:, 1])
    normals = np.column_stack((-dz_dx, -dz_dy, np.ones(len(array))))
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)
