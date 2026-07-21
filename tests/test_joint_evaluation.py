import csv
from pathlib import Path

import pytest

from pointcloud_playground.joint_evaluation import (
    append_controlled_source_outliers,
    diagnose_joint_correspondences,
    evaluate_joint_sensitivity,
    write_joint_sensitivity_metrics_csv,
)
from pointcloud_playground.overlap_evaluation import create_partial_overlap_case
from pointcloud_playground.registration import apply_transform
from pointcloud_playground.synthetic import generate_controlled_density_cloud


def test_controlled_source_outliers_preserve_clean_point_order() -> None:
    points = generate_controlled_density_cloud(point_count=300, seed=3)

    contaminated = append_controlled_source_outliers(
        points,
        outlier_fraction=0.1,
        distance_scale=0.25,
        seed=11,
    )

    assert contaminated.clean_points == len(points)
    assert contaminated.outlier_points > 0
    assert contaminated.points[: len(points)] == pytest.approx(points)
    assert contaminated.actual_outlier_fraction == pytest.approx(0.1, abs=0.002)


def test_joint_diagnostics_reject_labeled_outliers_at_known_transform() -> None:
    points = generate_controlled_density_cloud(point_count=800, seed=4)
    case = create_partial_overlap_case(
        points,
        overlap_ratio=0.8,
        angle_deg=2.0,
        translation_scale=0.5,
    )
    contaminated = append_controlled_source_outliers(
        case.source_points,
        outlier_fraction=0.1,
        seed=7,
    )
    aligned = apply_transform(contaminated.points, case.known_alignment)

    diagnostics = diagnose_joint_correspondences(
        aligned,
        case.target_points,
        contaminated.clean_points,
        case.overlap_points,
        correspondence_fraction=0.4,
        median_point_spacing=case.median_point_spacing,
    )

    assert diagnostics.retained_source_overlap_ratio == pytest.approx(1.0)
    assert diagnostics.retained_source_nonoverlap_ratio == pytest.approx(0.0)
    assert diagnostics.retained_outlier_ratio == pytest.approx(0.0)
    assert diagnostics.correct_match_precision == pytest.approx(1.0)
    assert diagnostics.outlier_rejection_rate == pytest.approx(1.0)


def test_joint_sensitivity_exposes_combined_valid_pair_boundary() -> None:
    points = generate_controlled_density_cloud(point_count=800, seed=9)

    results = evaluate_joint_sensitivity(
        points,
        overlap_ratios=[0.4],
        outlier_fractions=[0.0, 0.1],
        trim_fractions=[0.4],
        max_iterations=80,
    )

    trimmed = {
        result.requested_outlier_fraction: result
        for result in results
        if result.method == "trim_0p4"
    }
    assert trimmed[0.0].recovered
    assert not trimmed[0.1].recovered
    assert trimmed[0.0].fraction_to_effective_valid_ratio <= 1.01
    assert trimmed[0.1].fraction_to_effective_valid_ratio > 1.0
    assert (
        trimmed[0.0].normalized_overlap_correspondence_rmse
        < trimmed[0.1].normalized_overlap_correspondence_rmse
    )


def test_joint_sensitivity_metrics_csv_is_generated(tmp_path: Path) -> None:
    points = generate_controlled_density_cloud(point_count=300, seed=5)
    results = evaluate_joint_sensitivity(
        points,
        overlap_ratios=[0.8],
        outlier_fractions=[0.0, 0.05],
        trim_fractions=[0.4],
        max_iterations=40,
    )

    metrics_path = write_joint_sensitivity_metrics_csv(
        tmp_path / "metrics.csv",
        results,
    )

    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {row["method"] for row in rows} == {"all_pairs", "trim_0p4"}
    uncontaminated = next(
        row
        for row in rows
        if row["method"] == "all_pairs"
        and row["requested_outlier_fraction"] == "0.0"
    )
    assert uncontaminated["outlier_rejection_rate"] == ""
    assert uncontaminated["normalized_median_rejected_residual"] == ""
