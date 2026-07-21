"""Trim-fraction sensitivity and correspondence diagnostics."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .io import PointCloud, validate_points
from .overlap_evaluation import create_partial_overlap_case
from .registration import iterative_closest_point, nearest_neighbor_rmse
from .registration_evaluation import paired_rmse, rotation_error_degrees


@dataclass(frozen=True)
class CorrespondenceDiagnostics:
    """Known-overlap diagnostics for retained nearest-neighbor pairs."""

    retained_correspondences: int
    retained_source_overlap_ratio: float
    overlap_source_retention: float
    correct_match_precision: float
    correct_match_recall: float
    normalized_median_retained_residual: float
    normalized_median_rejected_residual: float | None


@dataclass(frozen=True)
class TrimSensitivityResult:
    """Metrics for one overlap ratio and retained-correspondence fraction."""

    case: str
    requested_overlap_ratio: float
    actual_overlap_ratio: float
    trim_fraction: float
    fraction_to_overlap_ratio: float
    scan_points: int
    overlap_points: int
    correspondences_used: int
    angle_deg: float
    translation_scale: float
    median_point_spacing: float
    normalized_final_objective_rmse: float
    normalized_final_all_nearest_neighbor_rmse: float
    normalized_overlap_correspondence_rmse: float
    rotation_error_deg: float
    normalized_translation_error: float
    retained_source_overlap_ratio: float
    overlap_source_retention: float
    correct_match_precision: float
    correct_match_recall: float
    normalized_median_retained_residual: float
    normalized_median_rejected_residual: float | None
    iterations: int
    converged: bool
    recovered: bool

    def as_row(self) -> dict[str, str | int | float | bool | None]:
        """Return one row for the sensitivity metrics CSV."""
        return {
            "case": self.case,
            "requested_overlap_ratio": self.requested_overlap_ratio,
            "actual_overlap_ratio": self.actual_overlap_ratio,
            "trim_fraction": self.trim_fraction,
            "fraction_to_overlap_ratio": self.fraction_to_overlap_ratio,
            "scan_points": self.scan_points,
            "overlap_points": self.overlap_points,
            "correspondences_used": self.correspondences_used,
            "angle_deg": self.angle_deg,
            "translation_scale": self.translation_scale,
            "median_point_spacing": self.median_point_spacing,
            "normalized_final_objective_rmse": (
                self.normalized_final_objective_rmse
            ),
            "normalized_final_all_nearest_neighbor_rmse": (
                self.normalized_final_all_nearest_neighbor_rmse
            ),
            "normalized_overlap_correspondence_rmse": (
                self.normalized_overlap_correspondence_rmse
            ),
            "rotation_error_deg": self.rotation_error_deg,
            "normalized_translation_error": self.normalized_translation_error,
            "retained_source_overlap_ratio": (
                self.retained_source_overlap_ratio
            ),
            "overlap_source_retention": self.overlap_source_retention,
            "correct_match_precision": self.correct_match_precision,
            "correct_match_recall": self.correct_match_recall,
            "normalized_median_retained_residual": (
                self.normalized_median_retained_residual
            ),
            "normalized_median_rejected_residual": (
                self.normalized_median_rejected_residual
            ),
            "iterations": self.iterations,
            "converged": self.converged,
            "recovered": self.recovered,
        }


def diagnose_partial_overlap_correspondences(
    aligned_points: NDArray[np.floating],
    target_points: NDArray[np.floating],
    overlap_points: int,
    correspondence_fraction: float,
    median_point_spacing: float,
) -> CorrespondenceDiagnostics:
    """Diagnose final retained pairs using the controlled overlap labels."""
    aligned = validate_points(aligned_points)
    target = validate_points(target_points)
    if len(aligned) != len(target):
        raise ValueError("Correspondence diagnostics require equal-size scans.")
    if overlap_points < 3 or overlap_points > len(aligned):
        raise ValueError("Overlap point count must be between 3 and scan size.")
    if (
        not np.isfinite(correspondence_fraction)
        or not 0.0 < correspondence_fraction <= 1.0
    ):
        raise ValueError("Correspondence fraction must be in (0, 1].")
    if not np.isfinite(median_point_spacing) or median_point_spacing <= 0.0:
        raise ValueError("Median point spacing must be positive and finite.")

    distances, target_indices = cKDTree(target).query(aligned, k=1)
    retained_count = min(
        len(aligned),
        max(3, int(np.ceil(len(aligned) * correspondence_fraction))),
    )
    order = np.argsort(distances, kind="stable")
    retained = order[:retained_count]
    rejected = order[retained_count:]

    retained_overlap = retained < overlap_points
    retained_overlap_count = int(np.count_nonzero(retained_overlap))
    correct_target_start = len(target) - overlap_points
    correct_matches = retained_overlap & (
        target_indices[retained]
        == correct_target_start + retained
    )
    correct_count = int(np.count_nonzero(correct_matches))
    rejected_median = (
        float(np.median(distances[rejected]) / median_point_spacing)
        if len(rejected)
        else None
    )

    return CorrespondenceDiagnostics(
        retained_correspondences=retained_count,
        retained_source_overlap_ratio=(
            retained_overlap_count / retained_count
        ),
        overlap_source_retention=retained_overlap_count / overlap_points,
        correct_match_precision=correct_count / retained_count,
        correct_match_recall=correct_count / overlap_points,
        normalized_median_retained_residual=float(
            np.median(distances[retained]) / median_point_spacing
        ),
        normalized_median_rejected_residual=rejected_median,
    )


def evaluate_trim_sensitivity(
    points: NDArray[np.floating],
    overlap_ratios: list[float],
    trim_fractions: list[float],
    angle_deg: float = 2.0,
    translation_scale: float = 0.5,
    max_iterations: int = 80,
    tolerance_scale: float = 1e-6,
) -> list[TrimSensitivityResult]:
    """Evaluate a fixed trim-fraction grid across controlled scan overlap."""
    cloud = validate_points(points)
    if not overlap_ratios:
        raise ValueError("At least one overlap ratio is required.")
    if not trim_fractions:
        raise ValueError("At least one trim fraction is required.")
    if any(
        not np.isfinite(fraction) or not 0.0 < fraction <= 1.0
        for fraction in trim_fractions
    ):
        raise ValueError("Trim fractions must be in (0, 1].")
    if len(set(trim_fractions)) != len(trim_fractions):
        raise ValueError("Trim fractions must be unique.")
    if not np.isfinite(tolerance_scale) or tolerance_scale <= 0.0:
        raise ValueError("Tolerance scale must be a positive finite number.")

    results: list[TrimSensitivityResult] = []
    for index, overlap_ratio in enumerate(overlap_ratios, start=1):
        case = create_partial_overlap_case(
            cloud,
            overlap_ratio,
            angle_deg,
            translation_scale,
        )
        overlap_slice = slice(0, case.overlap_points)
        for trim_fraction in trim_fractions:
            icp_result = iterative_closest_point(
                case.source_points,
                case.target_points,
                max_iterations=max_iterations,
                tolerance=case.median_point_spacing * tolerance_scale,
                correspondence_fraction=trim_fraction,
            )
            overlap_rmse = paired_rmse(
                icp_result.aligned_points[overlap_slice],
                case.overlap_target_points,
            )
            normalized_overlap_rmse = (
                overlap_rmse / case.median_point_spacing
            )
            translation_error = float(
                np.linalg.norm(
                    icp_result.transform.translation
                    - case.known_alignment.translation
                )
            )
            diagnostics = diagnose_partial_overlap_correspondences(
                icp_result.aligned_points,
                case.target_points,
                case.overlap_points,
                trim_fraction,
                case.median_point_spacing,
            )
            results.append(
                TrimSensitivityResult(
                    case=f"case_{index:02d}",
                    requested_overlap_ratio=case.requested_overlap_ratio,
                    actual_overlap_ratio=case.actual_overlap_ratio,
                    trim_fraction=float(trim_fraction),
                    fraction_to_overlap_ratio=(
                        trim_fraction / case.actual_overlap_ratio
                    ),
                    scan_points=len(case.target_points),
                    overlap_points=case.overlap_points,
                    correspondences_used=icp_result.correspondences_used,
                    angle_deg=float(angle_deg),
                    translation_scale=float(translation_scale),
                    median_point_spacing=case.median_point_spacing,
                    normalized_final_objective_rmse=(
                        icp_result.final_rmse / case.median_point_spacing
                    ),
                    normalized_final_all_nearest_neighbor_rmse=(
                        nearest_neighbor_rmse(
                            icp_result.aligned_points,
                            case.target_points,
                        )
                        / case.median_point_spacing
                    ),
                    normalized_overlap_correspondence_rmse=(
                        normalized_overlap_rmse
                    ),
                    rotation_error_deg=rotation_error_degrees(
                        icp_result.transform.rotation,
                        case.known_alignment.rotation,
                    ),
                    normalized_translation_error=(
                        translation_error / case.median_point_spacing
                    ),
                    retained_source_overlap_ratio=(
                        diagnostics.retained_source_overlap_ratio
                    ),
                    overlap_source_retention=(
                        diagnostics.overlap_source_retention
                    ),
                    correct_match_precision=(
                        diagnostics.correct_match_precision
                    ),
                    correct_match_recall=diagnostics.correct_match_recall,
                    normalized_median_retained_residual=(
                        diagnostics.normalized_median_retained_residual
                    ),
                    normalized_median_rejected_residual=(
                        diagnostics.normalized_median_rejected_residual
                    ),
                    iterations=icp_result.iterations,
                    converged=icp_result.converged,
                    recovered=(
                        icp_result.converged
                        and normalized_overlap_rmse <= 0.01
                    ),
                )
            )
    return results


def write_trim_sensitivity_metrics_csv(
    path: str | Path,
    results: list[TrimSensitivityResult],
) -> Path:
    """Write trim-sensitivity metrics and diagnostics to CSV."""
    if not results:
        raise ValueError("At least one trim-sensitivity result is required.")
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
