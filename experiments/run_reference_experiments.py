"""Reproduce the versioned synthetic and public-data experiment outputs."""

from pathlib import Path

from pointcloud_playground.evaluation import (
    evaluate_voxel_sizes,
    write_metrics_csv,
)
from pointcloud_playground.io import load_xyz, save_xyz
from pointcloud_playground.synthetic import generate_controlled_density_cloud
from pointcloud_playground.visualization import save_comparison_plot

ROOT = Path(__file__).resolve().parents[1]


def run_experiment(
    input_path: Path, output_dir: Path, voxel_sizes: list[float]
) -> None:
    """Run one reference evaluation and write its metrics and figure."""
    points = load_xyz(input_path)
    results = evaluate_voxel_sizes(points, voxel_sizes)
    write_metrics_csv(output_dir / "metrics.csv", results)
    save_comparison_plot(output_dir / "comparison.png", points, results)


def main() -> None:
    """Generate both versioned reference experiments."""
    synthetic_path = ROOT / "data" / "synthetic_controlled_density.xyz"
    save_xyz(synthetic_path, generate_controlled_density_cloud())
    run_experiment(
        synthetic_path,
        ROOT / "results" / "synthetic",
        [0.25, 0.5, 1.0],
    )
    run_experiment(
        ROOT / "data" / "usgs_3dep_iowa" / "sample.xyz",
        ROOT / "results" / "usgs_3dep_iowa",
        [5.0, 10.0, 20.0],
    )


if __name__ == "__main__":
    main()
