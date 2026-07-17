"""Point-cloud downsampling methods."""

import numpy as np
from numpy.typing import NDArray

from .io import PointCloud, validate_points


def voxel_downsample(
    points: NDArray[np.floating], voxel_size: float
) -> PointCloud:
    """Replace the points in each occupied voxel with their centroid."""
    cloud = validate_points(points)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("Voxel size must be a positive finite number.")

    origin = cloud.min(axis=0)
    voxel_indices = np.floor((cloud - origin) / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)

    sums = np.zeros((inverse.max() + 1, 3), dtype=np.float64)
    np.add.at(sums, inverse, cloud)
    counts = np.bincount(inverse)
    return sums / counts[:, np.newaxis]
