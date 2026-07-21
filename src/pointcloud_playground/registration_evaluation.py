"""Controlled evaluation of rigid point-cloud registration."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .io import PointCloud, save_xyz, validate_points
from .registration import (
    RigidTransform,
    apply_transform,
    axis_angle_rotation,
    iterative_closest_point,
)


@dataclass(frozen=True)
class RegistrationEvaluationResult:
    """Metrics and point clouds for one controlled misalignment."""

    case: str
    angle_deg: float
    translation_scale: float
    translation_distance: float
    median_point_spacing: float
    initial_nearest_neighbor_rmse: float
    final_nearest_neighbor_rmse: float
    normalized_initial_nearest_neighbor_rmse: float
    normalized_final_nearest_neighbor_rmse: float
    initial_correspondence_rmse: float
    correspondence_rmse: float
    normalized_correspondence_rmse: float
    rotation_error_deg: float
    translation_error: float
    normalized_translation_error: float
    iterations: int
    converged: bool
    source_points: PointCloud
    aligned_points: PointCloud
    estimated_transform: RigidTransform

    def as_row(self) -> dict[str, str | int | float | bool]:
        """Return serializable metrics without point arrays."""
        return {
            "case": self.case,
            "angle_deg": self.angle_deg,
            "translation_scale": self.translation_scale,
            "translation_distance": self.translation_distance,
            "median_point_spacing": self.median_point_spacing,
            "initial_nearest_neighbor_rmse": (
                self.initial_nearest_neighbor_rmse
            ),
            "final_nearest_neighbor_rmse": self.final_nearest_neighbor_rmse,
            "normalized_initial_nearest_neighbor_rmse": (
                self.normalized_initial_nearest_neighbor_rmse
            ),
            "normalized_final_nearest_neighbor_rmse": (
                self.normalized_final_nearest_neighbor_rmse
            ),
            "initial_correspondence_rmse": self.initial_correspondence_rmse,
            "correspondence_rmse": self.correspondence_rmse,
            "normalized_correspondence_rmse": (
                self.normalized_correspondence_rmse
            ),
            "rotation_error_deg": self.rotation_error_deg,
            "translation_error": self.translation_error,
            "normalized_translation_error": self.normalized_translation_error,
            "iterations": self.iterations,
            "converged": self.converged,
        }


def median_point_spacing(points: PointCloud) -> float:
    """Return the median positive nearest-neighbor spacing."""
    distances, _ = cKDTree(points).query(points, k=2)
    positive = distances[:, 1][distances[:, 1] > 0.0]
    if len(positive) == 0:
        raise ValueError("Registration evaluation requires distinct points.")
    return float(np.median(positive))


def paired_rmse(
    first: PointCloud,
    second: PointCloud,
) -> float:
    """Return RMSE for two point clouds with known one-to-one pairs."""
    return float(
        np.sqrt(np.mean(np.sum(np.square(first - second), axis=1)))
    )


def rotation_error_degrees(
    estimated: NDArray[np.floating],
    reference: NDArray[np.floating],
) -> float:
    """Return the residual angle between estimated and reference rotations."""
    relative = np.asarray(estimated) @ np.asarray(reference).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def create_controlled_misalignment(
    points: NDArray[np.floating],
    angle_deg: float,
    translation_scale: float,
) -> tuple[PointCloud, RigidTransform, float, float]:
    """Create a transformed source and its known source-to-target transform."""
    target = validate_points(points)
    if not np.isfinite(translation_scale) or translation_scale < 0.0:
        raise ValueError("Translation scale must be non-negative and finite.")

    median_spacing = median_point_spacing(target)
    center = target.mean(axis=0)
    forward_rotation = axis_angle_rotation(
        np.array([0.3, -0.2, 1.0]),
        angle_deg,
    )
    direction = np.array([0.7, -0.4, 0.2], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    offset = direction * translation_scale * median_spacing
    forward_translation = center + offset - forward_rotation @ center
    source = apply_transform(
        target,
        RigidTransform(forward_rotation, forward_translation),
    )

    inverse_rotation = forward_rotation.T
    inverse_translation = -inverse_rotation @ forward_translation
    known_alignment = RigidTransform(inverse_rotation, inverse_translation)
    return source, known_alignment, median_spacing, float(np.linalg.norm(offset))


def evaluate_registration_cases(
    target_points: NDArray[np.floating],
    angles_deg: list[float],
    translation_scales: list[float],
    max_iterations: int = 60,
    tolerance_scale: float = 1e-6,
) -> list[RegistrationEvaluationResult]:
    """Evaluate ICP recovery across controlled rigid misalignments."""
    target = validate_points(target_points)
    if not angles_deg or len(angles_deg) != len(translation_scales):
        raise ValueError(
            "Angles and translation scales must be non-empty and paired."
        )
    if not np.isfinite(tolerance_scale) or tolerance_scale <= 0.0:
        raise ValueError("Tolerance scale must be a positive finite number.")

    results: list[RegistrationEvaluationResult] = []
    for index, (angle_deg, translation_scale) in enumerate(
        zip(angles_deg, translation_scales),
        start=1,
    ):
        source, known, median_spacing, translation_distance = (
            create_controlled_misalignment(
                target,
                angle_deg,
                translation_scale,
            )
        )
        icp_result = iterative_closest_point(
            source,
            target,
            max_iterations=max_iterations,
            tolerance=median_spacing * tolerance_scale,
        )
        initial_correspondence_rmse = paired_rmse(source, target)
        correspondence_rmse = paired_rmse(
            icp_result.aligned_points,
            target,
        )
        translation_error = float(
            np.linalg.norm(
                icp_result.transform.translation - known.translation
            )
        )
        results.append(
            RegistrationEvaluationResult(
                case=f"case_{index:02d}",
                angle_deg=float(angle_deg),
                translation_scale=float(translation_scale),
                translation_distance=translation_distance,
                median_point_spacing=median_spacing,
                initial_nearest_neighbor_rmse=icp_result.initial_rmse,
                final_nearest_neighbor_rmse=icp_result.final_rmse,
                normalized_initial_nearest_neighbor_rmse=(
                    icp_result.initial_rmse / median_spacing
                ),
                normalized_final_nearest_neighbor_rmse=(
                    icp_result.final_rmse / median_spacing
                ),
                initial_correspondence_rmse=initial_correspondence_rmse,
                correspondence_rmse=correspondence_rmse,
                normalized_correspondence_rmse=(
                    correspondence_rmse / median_spacing
                ),
                rotation_error_deg=rotation_error_degrees(
                    icp_result.transform.rotation,
                    known.rotation,
                ),
                translation_error=translation_error,
                normalized_translation_error=(
                    translation_error / median_spacing
                ),
                iterations=icp_result.iterations,
                converged=icp_result.converged,
                source_points=source,
                aligned_points=icp_result.aligned_points,
                estimated_transform=icp_result.transform,
            )
        )
    return results


def write_registration_metrics_csv(
    path: str | Path,
    results: list[RegistrationEvaluationResult],
) -> Path:
    """Write controlled registration metrics to CSV."""
    if not results:
        raise ValueError("At least one registration result is required.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(results[0].as_row()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result.as_row() for result in results)
    return output_path


def write_aligned_clouds(
    output_dir: str | Path,
    results: list[RegistrationEvaluationResult],
) -> list[Path]:
    """Write the source and aligned cloud for every evaluation case."""
    directory = Path(output_dir)
    paths: list[Path] = []
    for result in results:
        paths.append(
            save_xyz(
                directory / f"{result.case}_source.xyz",
                result.source_points,
            )
        )
        paths.append(
            save_xyz(
                directory / f"{result.case}_aligned.xyz",
                result.aligned_points,
            )
        )
    return paths
