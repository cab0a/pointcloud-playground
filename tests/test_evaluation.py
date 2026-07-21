from pathlib import Path
import csv

import numpy as np

from pointcloud_playground.evaluation import (
    evaluate_voxel_sizes,
    write_downsampling_metrics_csv,
)
from pointcloud_playground.synthetic import generate_controlled_density_cloud


def test_larger_voxels_reduce_points_and_increase_coverage_error() -> None:
    points = generate_controlled_density_cloud(point_count=1_000, seed=7)

    results = evaluate_voxel_sizes(points, [0.25, 1.0])

    assert results[1].output_points < results[0].output_points
    assert results[1].coverage_rmse > results[0].coverage_rmse
    assert 0.0 < results[1].retention_ratio < 1.0


def test_metrics_csv_is_generated(tmp_path: Path) -> None:
    points = generate_controlled_density_cloud(point_count=200, seed=3)
    results = evaluate_voxel_sizes(points, [0.5])

    output_path = write_downsampling_metrics_csv(
        tmp_path / "metrics.csv",
        results,
    )

    with output_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["voxel_size"] == "0.5"
    assert int(rows[0]["input_points"]) == 200
