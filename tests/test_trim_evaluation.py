import csv
from pathlib import Path

import numpy as np
import pytest

from pointcloud_playground.overlap_evaluation import create_partial_overlap_case
from pointcloud_playground.registration import apply_transform
from pointcloud_playground.synthetic import generate_controlled_density_cloud
from pointcloud_playground.trim_evaluation import (
    diagnose_partial_overlap_correspondences,
    evaluate_trim_sensitivity,
    write_trim_sensitivity_metrics_csv,
)


def test_correspondence_diagnostics_use_known_overlap_pairs() -> None:
    points = generate_controlled_density_cloud(point_count=800, seed=4)
    case = create_partial_overlap_case(
        points,
        overlap_ratio=0.8,
        angle_deg=2.0,
        translation_scale=0.5,
    )
    aligned = apply_transform(case.source_points, case.known_alignment)

    diagnostics = diagnose_partial_overlap_correspondences(
        aligned,
        case.target_points,
        case.overlap_points,
        correspondence_fraction=0.4,
        median_point_spacing=case.median_point_spacing,
    )

    assert diagnostics.retained_source_overlap_ratio == pytest.approx(1.0)
    assert diagnostics.correct_match_precision == pytest.approx(1.0)
    assert diagnostics.correct_match_recall == pytest.approx(
        diagnostics.retained_correspondences / case.overlap_points
    )
    assert diagnostics.normalized_median_retained_residual < 1e-10
    assert diagnostics.normalized_median_rejected_residual is not None


def test_trim_sensitivity_exposes_overlap_fraction_boundary() -> None:
    points = generate_controlled_density_cloud(point_count=800, seed=9)

    results = evaluate_trim_sensitivity(
        points,
        overlap_ratios=[0.4],
        trim_fractions=[0.4, 0.7],
        max_iterations=80,
    )

    by_fraction = {result.trim_fraction: result for result in results}
    assert by_fraction[0.4].recovered
    assert not by_fraction[0.7].recovered
    assert (
        by_fraction[0.4].correct_match_precision
        > by_fraction[0.7].correct_match_precision
    )
    assert (
        by_fraction[0.4].normalized_overlap_correspondence_rmse
        < by_fraction[0.7].normalized_overlap_correspondence_rmse
    )


def test_trim_sensitivity_metrics_csv_is_generated(tmp_path: Path) -> None:
    points = generate_controlled_density_cloud(point_count=300, seed=5)
    results = evaluate_trim_sensitivity(
        points,
        overlap_ratios=[0.8],
        trim_fractions=[0.4, 1.0],
        max_iterations=40,
    )

    metrics_path = write_trim_sensitivity_metrics_csv(
        tmp_path / "metrics.csv",
        results,
    )

    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {float(row["trim_fraction"]) for row in rows} == {0.4, 1.0}
    all_pairs = next(row for row in rows if row["trim_fraction"] == "1.0")
    assert all_pairs["normalized_median_rejected_residual"] == ""
    assert np.isfinite(float(rows[0]["correct_match_precision"]))
