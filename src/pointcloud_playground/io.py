"""Input and output helpers for XYZ point clouds."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PointCloud = NDArray[np.float64]


def validate_points(points: NDArray[np.floating]) -> PointCloud:
    """Return a validated point cloud as an ``(n, 3)`` float64 array."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Point clouds must have shape (n, 3).")
    if len(array) == 0:
        raise ValueError("Point clouds must contain at least one point.")
    if not np.isfinite(array).all():
        raise ValueError("Point clouds must contain only finite coordinates.")
    return array


def load_xyz(path: str | Path) -> PointCloud:
    """Load a whitespace-delimited XYZ file."""
    input_path = Path(path)
    try:
        points = np.loadtxt(input_path, comments="#", ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read XYZ point cloud: {input_path}") from exc
    return validate_points(points)


def save_xyz(path: str | Path, points: NDArray[np.floating]) -> Path:
    """Write a point cloud as a whitespace-delimited XYZ file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        validate_points(points),
        fmt="%.6f",
        header="x y z",
        comments="# ",
    )
    return output_path
