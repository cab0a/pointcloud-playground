"""Joint sensitivity to partial overlap and controlled source outliers."""

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
class ContaminatedSourceScan:
    """A source scan with appended outliers and stable point labels."""

    points: PointCloud
    clean_points: int
    outlier_points: int
    actual_outlier_fraction: float


@dataclass(frozen=True)
class JointCorrespondenceDiagnostics:
    """Final-pair diagnostics using controlled overlap and outlier labels."""

    retained_correspondences: int
    retained_source_overlap_ratio: float
    retained_source_nonoverlap_ratio: float
    retained_outlier_ratio: float
    overlap_source_retention: float
    correct_match_precision: float
    correct_match_recall: float
    outlier_rejection_rate: float | None
    normalized_median_retained_residual: float
    normalized_median_rejected_residual: float | None


@dataclass(frozen=True)
class JointSensitivityResult:
    """Metrics for one overlap, contamination, and correspondence condition."""

    case: str
    method: str
    requested_overlap_ratio: float
    actual_overlap_ratio: float
    requested_outlier_fraction: float
    actual_outlier_fraction: float
    correspondence_fraction: float
    effective_valid_pair_fraction: float
    fraction_to_effective_valid_ratio: float
    target_points: int
    clean_source_points: int
    source_points: int
    overlap_points: int
    outlier_points: int
    correspondences_used: int
    angle_deg: float
    translation_scale: float
    distance_scale: float
    median_point_spacing: float
    normalized_final_objective_rmse: float
    normalized_final_all_nearest_neighbor_rmse: float
    normalized_overlap_correspondence_rmse: float
    rotation_error_deg: float
    normalized_translation_error: float
    retained_source_overlap_ratio: float
    retained_source_nonoverlap_ratio: float
    retained_outlier_ratio: float
    overlap_source_retention: float
    correct_match_precision: float
    correct_match_recall: float
    outlier_rejection_rate: float | None
    normalized_median_retained_residual: float
    normalized_median_rejected_residual: float | None
    iterations: int
    converged: bool
    recovered: bool

    def as_row(self) -> dict[str, str | int | float | bool | None]:
        """Return one row for the joint-sensitivity metrics CSV."""
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def _outlier_count(clean_points: int, outlier_fraction: float) -> int:
    if not np.isfinite(outlier_fraction) or not 0.0 <= outlier_fraction < 1.0:
        raise ValueError("Outlier fractions must be in [0, 1).")
    if outlier_fraction == 0.0:
        return 0
    return max(
        1,
        round(clean_points * outlier_fraction / (1.0 - outlier_fraction)),
    )


def _generate_vertical_outlier_pool(
    points: NDArray[np.floating],
    count: int,
    distance_scale: float,
    seed: int,
) -> PointCloud:
    clean = validate_points(points)
    if count < 0:
        raise ValueError("Outlier count must be non-negative.")
    if not np.isfinite(distance_scale) or distance_scale <= 0.0:
        raise ValueError("Distance scale must be a positive finite number.")
    if count == 0:
        return np.empty((0, 3), dtype=np.float64)

    minimum = clean.min(axis=0)
    maximum = clean.max(axis=0)
    span = maximum - minimum
    vertical_scale = max(
        span[2],
        float(np.linalg.norm(span[:2])) * 0.05,
        np.finfo(np.float64).eps,
    )
    rng = np.random.default_rng(seed)
    outliers = np.empty((count, 3), dtype=np.float64)
    outliers[:, 0] = rng.uniform(minimum[0], maximum[0], size=count)
    outliers[:, 1] = rng.uniform(minimum[1], maximum[1], size=count)
    offsets = (
        vertical_scale
        * distance_scale
        * rng.uniform(1.0, 2.0, size=count)
    )
    above = rng.random(count) >= 0.5
    outliers[:, 2] = np.where(
        above,
        maximum[2] + offsets,
        minimum[2] - offsets,
    )
    return outliers


def append_controlled_source_outliers(
    points: NDArray[np.floating],
    outlier_fraction: float,
    distance_scale: float = 0.25,
    seed: int = 42,
) -> ContaminatedSourceScan:
    """Append isolated vertical outliers without disturbing source indices."""
    clean = validate_points(points)
    count = _outlier_count(len(clean), outlier_fraction)
    outliers = _generate_vertical_outlier_pool(
        clean,
        count,
        distance_scale,
        seed,
    )
    combined = np.vstack((clean, outliers))
    return ContaminatedSourceScan(
        points=combined,
        clean_points=len(clean),
        outlier_points=count,
        actual_outlier_fraction=count / len(combined),
    )


def diagnose_joint_correspondences(
    aligned_points: NDArray[np.floating],
    target_points: NDArray[np.floating],
    clean_source_points: int,
    overlap_points: int,
    correspondence_fraction: float,
    median_point_spacing: float,
) -> JointCorrespondenceDiagnostics:
    """Diagnose retained pairs using overlap, exact-pair, and outlier labels."""
    aligned = validate_points(aligned_points)
    target = validate_points(target_points)
    if clean_source_points != len(target) or clean_source_points > len(aligned):
        raise ValueError(
            "The clean source and target must have the same point count."
        )
    if overlap_points < 3 or overlap_points > clean_source_points:
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
    retained_nonoverlap = (
        (retained >= overlap_points) & (retained < clean_source_points)
    )
    retained_outliers = retained >= clean_source_points
    overlap_count = int(np.count_nonzero(retained_overlap))
    outlier_count = len(aligned) - clean_source_points
    retained_outlier_count = int(np.count_nonzero(retained_outliers))
    correct_target_start = len(target) - overlap_points
    correct_matches = retained_overlap & (
        target_indices[retained] == correct_target_start + retained
    )
    correct_count = int(np.count_nonzero(correct_matches))
    rejected_median = (
        float(np.median(distances[rejected]) / median_point_spacing)
        if len(rejected)
        else None
    )

    return JointCorrespondenceDiagnostics(
        retained_correspondences=retained_count,
        retained_source_overlap_ratio=overlap_count / retained_count,
        retained_source_nonoverlap_ratio=(
            int(np.count_nonzero(retained_nonoverlap)) / retained_count
        ),
        retained_outlier_ratio=retained_outlier_count / retained_count,
        overlap_source_retention=overlap_count / overlap_points,
        correct_match_precision=correct_count / retained_count,
        correct_match_recall=correct_count / overlap_points,
        outlier_rejection_rate=(
            (outlier_count - retained_outlier_count) / outlier_count
            if outlier_count
            else None
        ),
        normalized_median_retained_residual=float(
            np.median(distances[retained]) / median_point_spacing
        ),
        normalized_median_rejected_residual=rejected_median,
    )


def _method_name(correspondence_fraction: float) -> str:
    if correspondence_fraction == 1.0:
        return "all_pairs"
    label = f"{correspondence_fraction:g}".replace(".", "p")
    return f"trim_{label}"


def evaluate_joint_sensitivity(
    points: NDArray[np.floating],
    overlap_ratios: list[float],
    outlier_fractions: list[float],
    trim_fractions: list[float],
    angle_deg: float = 2.0,
    translation_scale: float = 0.5,
    distance_scale: float = 0.25,
    seed: int = 42,
    max_iterations: int = 80,
    tolerance_scale: float = 1e-6,
) -> list[JointSensitivityResult]:
    """Evaluate correspondence rules across overlap and outlier grids."""
    cloud = validate_points(points)
    if not overlap_ratios:
        raise ValueError("At least one overlap ratio is required.")
    if not outlier_fractions:
        raise ValueError("At least one outlier fraction is required.")
    if not trim_fractions:
        raise ValueError("At least one trim fraction is required.")
    if any(
        not np.isfinite(fraction) or not 0.0 < fraction < 1.0
        for fraction in trim_fractions
    ):
        raise ValueError("Trim fractions must be in (0, 1).")
    if len(set(trim_fractions)) != len(trim_fractions):
        raise ValueError("Trim fractions must be unique.")
    for fraction in outlier_fractions:
        _outlier_count(len(cloud), fraction)
    if len(set(outlier_fractions)) != len(outlier_fractions):
        raise ValueError("Outlier fractions must be unique.")
    if not np.isfinite(distance_scale) or distance_scale <= 0.0:
        raise ValueError("Distance scale must be a positive finite number.")
    if not np.isfinite(tolerance_scale) or tolerance_scale <= 0.0:
        raise ValueError("Tolerance scale must be a positive finite number.")

    methods = [("all_pairs", 1.0)] + [
        (_method_name(float(fraction)), float(fraction))
        for fraction in trim_fractions
    ]
    results: list[JointSensitivityResult] = []
    for case_index, overlap_ratio in enumerate(overlap_ratios, start=1):
        case = create_partial_overlap_case(
            cloud,
            overlap_ratio,
            angle_deg,
            translation_scale,
        )
        clean_count = len(case.source_points)
        max_outlier_count = max(
            _outlier_count(clean_count, fraction)
            for fraction in outlier_fractions
        )
        outlier_pool = _generate_vertical_outlier_pool(
            case.source_points,
            max_outlier_count,
            distance_scale,
            seed + case_index - 1,
        )
        overlap_slice = slice(0, case.overlap_points)
        for requested_outlier_fraction in outlier_fractions:
            outlier_count = _outlier_count(
                clean_count,
                requested_outlier_fraction,
            )
            source = np.vstack(
                (case.source_points, outlier_pool[:outlier_count])
            )
            actual_outlier_fraction = outlier_count / len(source)
            effective_valid_fraction = case.overlap_points / len(source)
            for method, correspondence_fraction in methods:
                icp_result = iterative_closest_point(
                    source,
                    case.target_points,
                    max_iterations=max_iterations,
                    tolerance=(
                        case.median_point_spacing * tolerance_scale
                    ),
                    correspondence_fraction=correspondence_fraction,
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
                diagnostics = diagnose_joint_correspondences(
                    icp_result.aligned_points,
                    case.target_points,
                    clean_count,
                    case.overlap_points,
                    correspondence_fraction,
                    case.median_point_spacing,
                )
                results.append(
                    JointSensitivityResult(
                        case=f"case_{case_index:02d}",
                        method=method,
                        requested_overlap_ratio=(
                            case.requested_overlap_ratio
                        ),
                        actual_overlap_ratio=case.actual_overlap_ratio,
                        requested_outlier_fraction=float(
                            requested_outlier_fraction
                        ),
                        actual_outlier_fraction=actual_outlier_fraction,
                        correspondence_fraction=correspondence_fraction,
                        effective_valid_pair_fraction=(
                            effective_valid_fraction
                        ),
                        fraction_to_effective_valid_ratio=(
                            correspondence_fraction
                            / effective_valid_fraction
                        ),
                        target_points=len(case.target_points),
                        clean_source_points=clean_count,
                        source_points=len(source),
                        overlap_points=case.overlap_points,
                        outlier_points=outlier_count,
                        correspondences_used=(
                            icp_result.correspondences_used
                        ),
                        angle_deg=float(angle_deg),
                        translation_scale=float(translation_scale),
                        distance_scale=float(distance_scale),
                        median_point_spacing=case.median_point_spacing,
                        normalized_final_objective_rmse=(
                            icp_result.final_rmse
                            / case.median_point_spacing
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
                        retained_source_nonoverlap_ratio=(
                            diagnostics.retained_source_nonoverlap_ratio
                        ),
                        retained_outlier_ratio=(
                            diagnostics.retained_outlier_ratio
                        ),
                        overlap_source_retention=(
                            diagnostics.overlap_source_retention
                        ),
                        correct_match_precision=(
                            diagnostics.correct_match_precision
                        ),
                        correct_match_recall=(
                            diagnostics.correct_match_recall
                        ),
                        outlier_rejection_rate=(
                            diagnostics.outlier_rejection_rate
                        ),
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


def write_joint_sensitivity_metrics_csv(
    path: str | Path,
    results: list[JointSensitivityResult],
) -> Path:
    """Write joint-sensitivity metrics and diagnostics to CSV."""
    if not results:
        raise ValueError("At least one joint-sensitivity result is required.")
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
