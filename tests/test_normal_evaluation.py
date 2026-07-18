import csv
from pathlib import Path

import numpy as np
import pytest

from pointcloud_playground.normal_evaluation import (
    evaluate_normal_neighborhoods,
    write_normal_estimates,
    write_normal_metrics_csv,
)
from pointcloud_playground.synthetic import (
    controlled_surface_normals,
    generate_controlled_density_cloud,
)


def test_known_surface_records_accuracy_and_repeatability() -> None:
    points = generate_controlled_density_cloud(point_count=800, seed=3)
    truth = controlled_surface_normals(points)

    results = evaluate_normal_neighborhoods(
        points,
        [8, 16, 32],
        reference_normals=truth,
        noise_scale=0.05,
        seed=9,
    )

    assert all(result.mean_angular_error_deg is not None for result in results)
    assert all(
        result.mean_repeatability_error_deg >= 0.0 for result in results
    )
    assert all(result.mean_surface_variation >= 0.0 for result in results)
    assert results[0].median_neighborhood_radius < (
        results[-1].median_neighborhood_radius
    )


def test_normal_evaluation_is_deterministic() -> None:
    points = generate_controlled_density_cloud(point_count=400, seed=5)

    first = evaluate_normal_neighborhoods(
        points,
        [8],
        noise_scale=0.05,
        seed=12,
    )
    second = evaluate_normal_neighborhoods(
        points,
        [8],
        noise_scale=0.05,
        seed=12,
    )

    assert first[0].as_row() == second[0].as_row()
    np.testing.assert_array_equal(first[0].normals, second[0].normals)


def test_normal_reports_are_generated(tmp_path: Path) -> None:
    points = generate_controlled_density_cloud(point_count=300, seed=2)
    results = evaluate_normal_neighborhoods(
        points,
        [8],
        noise_scale=0.05,
        seed=6,
    )

    metrics_path = write_normal_metrics_csv(
        tmp_path / "metrics.csv",
        results,
    )
    normal_paths = write_normal_estimates(tmp_path, points, results)

    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["neighbors"] == "8"
    assert rows[0]["mean_angular_error_deg"] == ""
    assert normal_paths == [tmp_path / "normals_k8.csv"]
    assert normal_paths[0].is_file()


def test_normal_evaluation_requires_distinct_points() -> None:
    with pytest.raises(ValueError, match="distinct"):
        evaluate_normal_neighborhoods(
            np.zeros((20, 3)),
            [8],
        )
