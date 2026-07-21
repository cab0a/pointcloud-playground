"""Controlled evaluation of registration with partial scan overlap."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .io import PointCloud, save_xyz, validate_points
from .registration import (
    RigidTransform,
    apply_transform,
    axis_angle_rotation,
    iterative_closest_point,
    nearest_neighbor_rmse,
)
from .registration_evaluation import (
    median_point_spacing,
    paired_rmse,
    rotation_error_degrees,
)


@dataclass(frozen=True)
class PartialOverlapCase:
    """Two controlled scans with a known overlap and rigid transform."""

    requested_overlap_ratio: float
    actual_overlap_ratio: float
    overlap_points: int
    median_point_spacing: float
    translation_distance: float
    target_points: PointCloud
    source_points: PointCloud
    overlap_target_points: PointCloud
    known_alignment: RigidTransform


@dataclass(frozen=True)
class PartialOverlapEvaluationResult:
    """Metrics and outputs for one overlap ratio and ICP correspondence rule."""

    case: str
    method: str
    requested_overlap_ratio: float
    actual_overlap_ratio: float
    scan_points: int
    overlap_points: int
    angle_deg: float
    translation_scale: float
    translation_distance: float
    median_point_spacing: float
    correspondence_fraction: float
    correspondences_used: int
    initial_objective_rmse: float
    final_objective_rmse: float
    normalized_final_objective_rmse: float
    initial_all_nearest_neighbor_rmse: float
    final_all_nearest_neighbor_rmse: float
    normalized_final_all_nearest_neighbor_rmse: float
    initial_overlap_correspondence_rmse: float
    overlap_correspondence_rmse: float
    normalized_overlap_correspondence_rmse: float
    rotation_error_deg: float
    translation_error: float
    normalized_translation_error: float
    iterations: int
    converged: bool
    recovered: bool
    target_points: PointCloud
    source_points: PointCloud
    aligned_points: PointCloud

    def as_row(self) -> dict[str, str | int | float | bool]:
        """Return serializable metrics without point arrays."""
        return {
            "case": self.case,
            "method": self.method,
            "requested_overlap_ratio": self.requested_overlap_ratio,
            "actual_overlap_ratio": self.actual_overlap_ratio,
            "scan_points": self.scan_points,
            "overlap_points": self.overlap_points,
            "angle_deg": self.angle_deg,
            "translation_scale": self.translation_scale,
            "translation_distance": self.translation_distance,
            "median_point_spacing": self.median_point_spacing,
            "correspondence_fraction": self.correspondence_fraction,
            "correspondences_used": self.correspondences_used,
            "initial_objective_rmse": self.initial_objective_rmse,
            "final_objective_rmse": self.final_objective_rmse,
            "normalized_final_objective_rmse": (
                self.normalized_final_objective_rmse
            ),
            "initial_all_nearest_neighbor_rmse": (
                self.initial_all_nearest_neighbor_rmse
            ),
            "final_all_nearest_neighbor_rmse": (
                self.final_all_nearest_neighbor_rmse
            ),
            "normalized_final_all_nearest_neighbor_rmse": (
                self.normalized_final_all_nearest_neighbor_rmse
            ),
            "initial_overlap_correspondence_rmse": (
                self.initial_overlap_correspondence_rmse
            ),
            "overlap_correspondence_rmse": (
                self.overlap_correspondence_rmse
            ),
            "normalized_overlap_correspondence_rmse": (
                self.normalized_overlap_correspondence_rmse
            ),
            "rotation_error_deg": self.rotation_error_deg,
            "translation_error": self.translation_error,
            "normalized_translation_error": self.normalized_translation_error,
            "iterations": self.iterations,
            "converged": self.converged,
            "recovered": self.recovered,
        }


def create_partial_overlap_case(
    points: NDArray[np.floating],
    overlap_ratio: float,
    angle_deg: float,
    translation_scale: float,
) -> PartialOverlapCase:
    """Create left and right scans with a controlled shared region."""
    cloud = validate_points(points)
    if len(cloud) < 6:
        raise ValueError("Partial-overlap evaluation requires at least six points.")
    if not np.isfinite(overlap_ratio) or not 0.0 < overlap_ratio <= 1.0:
        raise ValueError("Overlap ratio must be in (0, 1].")
    if not np.isfinite(translation_scale) or translation_scale < 0.0:
        raise ValueError("Translation scale must be non-negative and finite.")

    point_count = len(cloud)
    scan_points = int(round(point_count / (2.0 - overlap_ratio)))
    scan_points = min(point_count, max(3, scan_points))
    overlap_points = 2 * scan_points - point_count
    if overlap_points < 3:
        scan_points = min(point_count, (point_count + 3 + 1) // 2)
        overlap_points = 2 * scan_points - point_count
    if overlap_points < 3:
        raise ValueError("Overlap region must contain at least three points.")

    order = np.lexsort((cloud[:, 2], cloud[:, 1], cloud[:, 0]))
    target_indices = order[:scan_points]
    source_indices = order[point_count - scan_points :]
    overlap_indices = order[point_count - scan_points : scan_points]
    target = cloud[target_indices]
    source_reference = cloud[source_indices]
    overlap_target = cloud[overlap_indices]

    spacing = median_point_spacing(cloud)
    center = cloud.mean(axis=0)
    forward_rotation = axis_angle_rotation(
        np.array([0.3, -0.2, 1.0]),
        angle_deg,
    )
    direction = np.array([0.7, -0.4, 0.2], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    offset = direction * translation_scale * spacing
    forward_translation = center + offset - forward_rotation @ center
    source = apply_transform(
        source_reference,
        RigidTransform(forward_rotation, forward_translation),
    )
    inverse_rotation = forward_rotation.T
    inverse_translation = -inverse_rotation @ forward_translation

    return PartialOverlapCase(
        requested_overlap_ratio=float(overlap_ratio),
        actual_overlap_ratio=overlap_points / scan_points,
        overlap_points=overlap_points,
        median_point_spacing=spacing,
        translation_distance=float(np.linalg.norm(offset)),
        target_points=target,
        source_points=source,
        overlap_target_points=overlap_target,
        known_alignment=RigidTransform(inverse_rotation, inverse_translation),
    )


def evaluate_partial_overlap_cases(
    points: NDArray[np.floating],
    overlap_ratios: list[float],
    angle_deg: float = 2.0,
    translation_scale: float = 0.5,
    trim_fraction: float = 0.7,
    max_iterations: int = 80,
    tolerance_scale: float = 1e-6,
) -> list[PartialOverlapEvaluationResult]:
    """Compare all-pairs and fixed-fraction trimmed ICP across overlap levels."""
    cloud = validate_points(points)
    if not overlap_ratios:
        raise ValueError("At least one overlap ratio is required.")
    if not np.isfinite(trim_fraction) or not 0.0 < trim_fraction < 1.0:
        raise ValueError("Trim fraction must be in (0, 1).")
    if not np.isfinite(tolerance_scale) or tolerance_scale <= 0.0:
        raise ValueError("Tolerance scale must be a positive finite number.")

    methods = (("all_pairs", 1.0), ("trimmed", float(trim_fraction)))
    results: list[PartialOverlapEvaluationResult] = []
    for index, overlap_ratio in enumerate(overlap_ratios, start=1):
        case = create_partial_overlap_case(
            cloud,
            overlap_ratio,
            angle_deg,
            translation_scale,
        )
        overlap_slice = slice(0, case.overlap_points)
        initial_overlap_rmse = paired_rmse(
            case.source_points[overlap_slice],
            case.overlap_target_points,
        )
        initial_all_nn_rmse = nearest_neighbor_rmse(
            case.source_points,
            case.target_points,
        )
        for method, correspondence_fraction in methods:
            icp_result = iterative_closest_point(
                case.source_points,
                case.target_points,
                max_iterations=max_iterations,
                tolerance=case.median_point_spacing * tolerance_scale,
                correspondence_fraction=correspondence_fraction,
            )
            overlap_rmse = paired_rmse(
                icp_result.aligned_points[overlap_slice],
                case.overlap_target_points,
            )
            all_nn_rmse = nearest_neighbor_rmse(
                icp_result.aligned_points,
                case.target_points,
            )
            translation_error = float(
                np.linalg.norm(
                    icp_result.transform.translation
                    - case.known_alignment.translation
                )
            )
            normalized_overlap_rmse = (
                overlap_rmse / case.median_point_spacing
            )
            results.append(
                PartialOverlapEvaluationResult(
                    case=f"case_{index:02d}",
                    method=method,
                    requested_overlap_ratio=case.requested_overlap_ratio,
                    actual_overlap_ratio=case.actual_overlap_ratio,
                    scan_points=len(case.target_points),
                    overlap_points=case.overlap_points,
                    angle_deg=float(angle_deg),
                    translation_scale=float(translation_scale),
                    translation_distance=case.translation_distance,
                    median_point_spacing=case.median_point_spacing,
                    correspondence_fraction=correspondence_fraction,
                    correspondences_used=icp_result.correspondences_used,
                    initial_objective_rmse=icp_result.initial_rmse,
                    final_objective_rmse=icp_result.final_rmse,
                    normalized_final_objective_rmse=(
                        icp_result.final_rmse / case.median_point_spacing
                    ),
                    initial_all_nearest_neighbor_rmse=initial_all_nn_rmse,
                    final_all_nearest_neighbor_rmse=all_nn_rmse,
                    normalized_final_all_nearest_neighbor_rmse=(
                        all_nn_rmse / case.median_point_spacing
                    ),
                    initial_overlap_correspondence_rmse=initial_overlap_rmse,
                    overlap_correspondence_rmse=overlap_rmse,
                    normalized_overlap_correspondence_rmse=(
                        normalized_overlap_rmse
                    ),
                    rotation_error_deg=rotation_error_degrees(
                        icp_result.transform.rotation,
                        case.known_alignment.rotation,
                    ),
                    translation_error=translation_error,
                    normalized_translation_error=(
                        translation_error / case.median_point_spacing
                    ),
                    iterations=icp_result.iterations,
                    converged=icp_result.converged,
                    recovered=(
                        icp_result.converged
                        and normalized_overlap_rmse <= 0.01
                    ),
                    target_points=case.target_points,
                    source_points=case.source_points,
                    aligned_points=icp_result.aligned_points,
                )
            )
    return results


def write_partial_overlap_metrics_csv(
    path: str | Path,
    results: list[PartialOverlapEvaluationResult],
) -> Path:
    """Write partial-overlap registration metrics to CSV."""
    if not results:
        raise ValueError("At least one partial-overlap result is required.")
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


def write_partial_overlap_clouds(
    output_dir: str | Path,
    results: list[PartialOverlapEvaluationResult],
) -> list[Path]:
    """Write target, source, and aligned clouds for every controlled case."""
    if not results:
        raise ValueError("At least one partial-overlap result is required.")
    directory = Path(output_dir)
    paths: list[Path] = []
    written_cases: set[str] = set()
    for result in results:
        if result.case not in written_cases:
            paths.append(
                save_xyz(
                    directory / f"{result.case}_target.xyz",
                    result.target_points,
                )
            )
            paths.append(
                save_xyz(
                    directory / f"{result.case}_source.xyz",
                    result.source_points,
                )
            )
            written_cases.add(result.case)
        paths.append(
            save_xyz(
                directory / f"{result.case}_{result.method}_aligned.xyz",
                result.aligned_points,
            )
        )
    return paths
