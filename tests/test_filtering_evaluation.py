import csv
from pathlib import Path

import numpy as np

from pointcloud_playground.filtering_evaluation import (
    evaluate_outlier_filter,
    select_best_filtering_result,
    write_filtering_metrics_csv,
)
from pointcloud_playground.outliers import inject_vertical_outliers
from pointcloud_playground.synthetic import generate_controlled_density_cloud


def test_threshold_sweep_records_detection_tradeoff() -> None:
    clean = generate_controlled_density_cloud(point_count=1_000, seed=5)
    contaminated = inject_vertical_outliers(clean, seed=12)

    results = evaluate_outlier_filter(
        contaminated.points,
        contaminated.is_outlier,
        clean,
        neighbors=16,
        std_ratios=[0.5, 2.5],
    )

    assert results[0].predicted_outliers >= results[1].predicted_outliers
    assert results[0].recall >= results[1].recall
    assert 0.0 <= results[0].precision <= 1.0
    assert 0.0 <= results[0].inlier_retention <= 1.0
    assert select_best_filtering_result(results) in results


def test_filtering_metrics_csv_is_generated(tmp_path: Path) -> None:
    clean = generate_controlled_density_cloud(point_count=300, seed=2)
    contaminated = inject_vertical_outliers(clean, seed=6)
    results = evaluate_outlier_filter(
        contaminated.points,
        contaminated.is_outlier,
        clean,
        neighbors=8,
        std_ratios=[1.0],
    )

    path = write_filtering_metrics_csv(tmp_path / "metrics.csv", results)

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["std_ratio"] == "1.0"
    assert int(rows[0]["true_outliers"]) == np.count_nonzero(
        contaminated.is_outlier
    )
