import csv
from pathlib import Path

from pointcloud_playground.registration_evaluation import (
    evaluate_registration_cases,
    write_aligned_clouds,
    write_registration_metrics_csv,
)
from pointcloud_playground.synthetic import generate_controlled_density_cloud


def test_registration_evaluation_records_paired_cases() -> None:
    points = generate_controlled_density_cloud(point_count=500, seed=3)

    results = evaluate_registration_cases(
        points,
        [2.0, 5.0],
        [0.5, 1.0],
        max_iterations=40,
    )

    assert [result.case for result in results] == ["case_01", "case_02"]
    assert results[0].translation_distance < results[1].translation_distance
    assert all(result.final_nearest_neighbor_rmse >= 0.0 for result in results)
    assert all(result.rotation_error_deg >= 0.0 for result in results)


def test_registration_reports_are_generated(tmp_path: Path) -> None:
    points = generate_controlled_density_cloud(point_count=300, seed=5)
    results = evaluate_registration_cases(
        points,
        [2.0],
        [0.5],
        max_iterations=40,
    )

    metrics_path = write_registration_metrics_csv(
        tmp_path / "metrics.csv",
        results,
    )
    cloud_paths = write_aligned_clouds(tmp_path, results)

    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["case"] == "case_01"
    assert cloud_paths == [
        tmp_path / "case_01_source.xyz",
        tmp_path / "case_01_aligned.xyz",
    ]
    assert all(path.is_file() for path in cloud_paths)
