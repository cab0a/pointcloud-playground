import csv
from pathlib import Path

import numpy as np

from pointcloud_playground.overlap_evaluation import (
    create_partial_overlap_case,
    evaluate_partial_overlap_cases,
    write_partial_overlap_clouds,
    write_partial_overlap_metrics_csv,
)
from pointcloud_playground.registration import apply_transform
from pointcloud_playground.synthetic import generate_controlled_density_cloud


def test_partial_overlap_case_preserves_known_shared_pairs() -> None:
    points = generate_controlled_density_cloud(point_count=800, seed=7)

    case = create_partial_overlap_case(
        points,
        overlap_ratio=0.4,
        angle_deg=5.0,
        translation_scale=1.0,
    )
    aligned_overlap = apply_transform(
        case.source_points[: case.overlap_points],
        case.known_alignment,
    )

    assert case.actual_overlap_ratio == 0.4
    assert len(case.target_points) == len(case.source_points)
    np.testing.assert_allclose(
        aligned_overlap,
        case.overlap_target_points,
        atol=1e-12,
    )


def test_partial_overlap_evaluation_compares_correspondence_rules() -> None:
    points = generate_controlled_density_cloud(point_count=800, seed=9)

    results = evaluate_partial_overlap_cases(
        points,
        [1.0, 0.8],
        trim_fraction=0.7,
        max_iterations=80,
    )

    assert len(results) == 4
    assert {result.method for result in results} == {"all_pairs", "trimmed"}
    assert all(result.overlap_correspondence_rmse >= 0.0 for result in results)
    all_pairs = next(
        result
        for result in results
        if result.requested_overlap_ratio == 0.8
        and result.method == "all_pairs"
    )
    trimmed = next(
        result
        for result in results
        if result.requested_overlap_ratio == 0.8
        and result.method == "trimmed"
    )
    assert trimmed.correspondences_used < all_pairs.correspondences_used
    assert not all_pairs.recovered
    assert trimmed.recovered
    assert (
        trimmed.normalized_overlap_correspondence_rmse
        < all_pairs.normalized_overlap_correspondence_rmse
    )


def test_partial_overlap_reports_are_generated(tmp_path: Path) -> None:
    points = generate_controlled_density_cloud(point_count=300, seed=5)
    results = evaluate_partial_overlap_cases(
        points,
        [0.8],
        trim_fraction=0.7,
        max_iterations=40,
    )

    metrics_path = write_partial_overlap_metrics_csv(
        tmp_path / "metrics.csv",
        results,
    )
    cloud_paths = write_partial_overlap_clouds(tmp_path, results)

    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["method"] for row in rows} == {"all_pairs", "trimmed"}
    assert len(cloud_paths) == 4
    assert all(path.is_file() for path in cloud_paths)
