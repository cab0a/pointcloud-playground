"""Local PCA normal estimation for point clouds."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .io import validate_points

NormalArray = NDArray[np.float64]


@dataclass(frozen=True)
class NormalEstimate:
    """Estimated normals and local neighborhood diagnostics."""

    normals: NormalArray
    surface_variation: NDArray[np.float64]
    neighborhood_radius: NDArray[np.float64]


def estimate_normals(
    points: NDArray[np.floating],
    neighbors: int,
) -> NormalEstimate:
    """Estimate upward-oriented normals from local PCA neighborhoods."""
    cloud = validate_points(points)
    if neighbors < 3:
        raise ValueError("Neighbors must be at least 3.")
    if neighbors >= len(cloud):
        raise ValueError("Neighbors must be smaller than the point count.")

    distances, indices = cKDTree(cloud).query(cloud, k=neighbors + 1)
    neighborhoods = cloud[indices[:, 1:]]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum(
        "nki,nkj->nij",
        centered,
        centered,
    ) / neighbors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    normals = eigenvectors[:, :, 0]
    normals[normals[:, 2] < 0.0] *= -1.0
    eigenvalue_sum = eigenvalues.sum(axis=1)
    smallest_eigenvalue = np.maximum(eigenvalues[:, 0], 0.0)
    surface_variation = np.divide(
        smallest_eigenvalue,
        eigenvalue_sum,
        out=np.zeros(len(cloud), dtype=np.float64),
        where=eigenvalue_sum > 0.0,
    )
    return NormalEstimate(
        normals=normals,
        surface_variation=surface_variation,
        neighborhood_radius=distances[:, -1],
    )
