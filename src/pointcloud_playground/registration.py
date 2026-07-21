"""Rigid transforms and point-to-point ICP registration."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .io import PointCloud, validate_points

Matrix3 = NDArray[np.float64]
Vector3 = NDArray[np.float64]


@dataclass(frozen=True)
class RigidTransform:
    """A rotation and translation applied as ``R @ point + t``."""

    rotation: Matrix3
    translation: Vector3


@dataclass(frozen=True)
class ICPResult:
    """Estimated transform and convergence diagnostics."""

    transform: RigidTransform
    aligned_points: PointCloud
    initial_rmse: float
    final_rmse: float
    correspondences_used: int
    iterations: int
    converged: bool


def identity_transform() -> RigidTransform:
    """Return the identity rigid transform."""
    return RigidTransform(
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def apply_transform(
    points: NDArray[np.floating],
    transform: RigidTransform,
) -> PointCloud:
    """Apply a rigid transform to a point cloud."""
    cloud = validate_points(points)
    rotation = np.asarray(transform.rotation, dtype=np.float64)
    translation = np.asarray(transform.translation, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Rigid transforms require a 3x3 rotation and 3-vector.")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Rigid transforms must contain only finite values.")
    return cloud @ rotation.T + translation


def axis_angle_rotation(
    axis: NDArray[np.floating],
    angle_deg: float,
) -> Matrix3:
    """Create a rotation matrix from an axis and angle in degrees."""
    direction = np.asarray(axis, dtype=np.float64)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("Rotation axis must be a finite 3-vector.")
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        raise ValueError("Rotation axis must have non-zero length.")
    if not np.isfinite(angle_deg):
        raise ValueError("Rotation angle must be finite.")

    x, y, z = direction / length
    angle = np.radians(angle_deg)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=np.float64,
    )


def best_fit_transform(
    source_points: NDArray[np.floating],
    target_points: NDArray[np.floating],
) -> RigidTransform:
    """Estimate the least-squares rigid transform for known pairs."""
    source = validate_points(source_points)
    target = validate_points(target_points)
    if source.shape != target.shape:
        raise ValueError("Source and target must contain paired points.")
    if len(source) < 3:
        raise ValueError("Rigid alignment requires at least three point pairs.")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transposed = np.linalg.svd(covariance)
    rotation = right_transposed.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transposed[-1, :] *= -1.0
        rotation = right_transposed.T @ left.T
    translation = target_center - rotation @ source_center
    return RigidTransform(rotation=rotation, translation=translation)


def nearest_neighbor_rmse(
    source_points: NDArray[np.floating],
    target_points: NDArray[np.floating],
) -> float:
    """Return source-to-target nearest-neighbor RMSE."""
    source = validate_points(source_points)
    target = validate_points(target_points)
    distances, _ = cKDTree(target).query(source, k=1)
    return float(np.sqrt(np.mean(np.square(distances))))


def iterative_closest_point(
    source_points: NDArray[np.floating],
    target_points: NDArray[np.floating],
    max_iterations: int = 60,
    tolerance: float = 1e-6,
    correspondence_fraction: float = 1.0,
) -> ICPResult:
    """Align source to target with optionally trimmed point-to-point ICP."""
    source = validate_points(source_points)
    target = validate_points(target_points)
    if len(source) < 3 or len(target) < 3:
        raise ValueError("ICP requires at least three points in each cloud.")
    if max_iterations < 1:
        raise ValueError("Maximum iterations must be at least 1.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Tolerance must be a positive finite number.")
    if (
        not np.isfinite(correspondence_fraction)
        or correspondence_fraction <= 0.0
        or correspondence_fraction > 1.0
    ):
        raise ValueError("Correspondence fraction must be in (0, 1].")

    correspondences_used = min(
        len(source),
        max(3, int(np.ceil(len(source) * correspondence_fraction))),
    )

    def selected_indices(distances: NDArray[np.floating]) -> NDArray[np.int64]:
        if correspondences_used == len(source):
            return np.arange(len(source), dtype=np.int64)
        order = np.argsort(distances, kind="stable")
        return order[:correspondences_used]

    def objective_rmse(distances: NDArray[np.floating]) -> float:
        selected = selected_indices(distances)
        return float(np.sqrt(np.mean(np.square(distances[selected]))))

    target_tree = cKDTree(target)
    current = source.copy()
    total = identity_transform()
    distances, _ = target_tree.query(current, k=1)
    initial_rmse = objective_rmse(distances)
    previous_rmse = initial_rmse
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        distances, indices = target_tree.query(current, k=1)
        selected = selected_indices(distances)
        incremental = best_fit_transform(
            current[selected],
            target[indices[selected]],
        )
        current = apply_transform(current, incremental)
        total = RigidTransform(
            rotation=incremental.rotation @ total.rotation,
            translation=(
                incremental.rotation @ total.translation
                + incremental.translation
            ),
        )
        distances, _ = target_tree.query(current, k=1)
        current_rmse = objective_rmse(distances)
        if abs(previous_rmse - current_rmse) <= tolerance:
            converged = True
            previous_rmse = current_rmse
            break
        previous_rmse = current_rmse

    return ICPResult(
        transform=total,
        aligned_points=current,
        initial_rmse=initial_rmse,
        final_rmse=previous_rmse,
        correspondences_used=correspondences_used,
        iterations=iterations,
        converged=converged,
    )
