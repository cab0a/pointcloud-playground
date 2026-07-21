"""Reproduce the versioned synthetic and public-data experiment outputs."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from pointcloud_playground.evaluation import (
    evaluate_voxel_sizes,
    write_downsampling_metrics_csv,
)
from pointcloud_playground.filtering_evaluation import (
    evaluate_outlier_filter,
    select_best_filtering_result,
    write_filtering_metrics_csv,
)
from pointcloud_playground.io import load_xyz
from pointcloud_playground.joint_evaluation import (
    evaluate_joint_sensitivity,
    write_joint_sensitivity_metrics_csv,
)
from pointcloud_playground.normal_evaluation import (
    evaluate_normal_neighborhoods,
    write_normal_metrics_csv,
)
from pointcloud_playground.outliers import inject_vertical_outliers
from pointcloud_playground.overlap_evaluation import (
    evaluate_partial_overlap_cases,
    write_partial_overlap_metrics_csv,
)
from pointcloud_playground.registration_evaluation import (
    evaluate_registration_cases,
    write_registration_metrics_csv,
)
from pointcloud_playground.summary import (
    collect_experiment_summaries,
    write_experiment_summary_csv,
    write_experiment_summary_markdown,
)
from pointcloud_playground.synthetic import (
    controlled_surface_normals,
)
from pointcloud_playground.trim_evaluation import (
    evaluate_trim_sensitivity,
    write_trim_sensitivity_metrics_csv,
)
from pointcloud_playground.visualization import (
    save_comparison_plot,
    save_experiment_summary_plot,
    save_joint_sensitivity_plot,
    save_normal_evaluation_plot,
    save_outlier_filtering_plot,
    save_partial_overlap_evaluation_plot,
    save_registration_evaluation_plot,
    save_trim_sensitivity_plot,
)

ROOT = Path(__file__).resolve().parents[1]


def run_downsampling_experiment(
    input_path: Path, output_dir: Path, voxel_sizes: list[float]
) -> None:
    """Run one reference evaluation and write its metrics and figure."""
    points = load_xyz(input_path)
    results = evaluate_voxel_sizes(points, voxel_sizes)
    write_downsampling_metrics_csv(output_dir / "metrics.csv", results)
    save_comparison_plot(output_dir / "comparison.png", points, results)


def run_outlier_experiment(
    input_path: Path,
    output_dir: Path,
    seed: int,
) -> None:
    """Run one controlled outlier-filtering evaluation."""
    clean = load_xyz(input_path)
    contaminated = inject_vertical_outliers(
        clean,
        outlier_fraction=0.05,
        distance_scale=0.25,
        seed=seed,
    )
    results = evaluate_outlier_filter(
        contaminated.points,
        contaminated.is_outlier,
        clean,
        neighbors=16,
        std_ratios=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    )
    write_filtering_metrics_csv(output_dir / "metrics.csv", results)
    save_outlier_filtering_plot(
        output_dir / "comparison.png",
        contaminated.points,
        contaminated.is_outlier,
        select_best_filtering_result(results),
    )


def run_normal_experiment(
    input_path: Path,
    output_dir: Path,
    reference_normals: bool,
) -> None:
    """Run one normal-estimation neighborhood evaluation."""
    points = load_xyz(input_path)
    truth = controlled_surface_normals(points) if reference_normals else None
    results = evaluate_normal_neighborhoods(
        points,
        [8, 16, 32, 64],
        reference_normals=truth,
        noise_scale=0.05,
        seed=42,
    )
    write_normal_metrics_csv(output_dir / "metrics.csv", results)
    save_normal_evaluation_plot(
        output_dir / "comparison.png",
        points,
        results,
    )


def run_registration_experiment(
    input_path: Path,
    output_dir: Path,
) -> None:
    """Run one controlled rigid-registration evaluation."""
    points = load_xyz(input_path)
    results = evaluate_registration_cases(
        points,
        [2.0, 5.0, 10.0, 20.0],
        [0.5, 1.0, 2.0, 4.0],
        max_iterations=60,
        tolerance_scale=1e-6,
    )
    write_registration_metrics_csv(output_dir / "metrics.csv", results)
    save_registration_evaluation_plot(
        output_dir / "comparison.png",
        points,
        results,
    )


def run_partial_overlap_experiment(
    input_path: Path,
    output_dir: Path,
) -> None:
    """Run one controlled partial-overlap registration evaluation."""
    points = load_xyz(input_path)
    results = evaluate_partial_overlap_cases(
        points,
        [1.0, 0.8, 0.6, 0.4],
        angle_deg=2.0,
        translation_scale=0.5,
        trim_fraction=0.7,
        max_iterations=80,
        tolerance_scale=1e-6,
    )
    write_partial_overlap_metrics_csv(output_dir / "metrics.csv", results)
    save_partial_overlap_evaluation_plot(
        output_dir / "comparison.png",
        results,
    )


def run_trim_sensitivity_experiment(
    input_path: Path,
    output_dir: Path,
) -> None:
    """Run one trim-fraction sensitivity and diagnostics evaluation."""
    points = load_xyz(input_path)
    results = evaluate_trim_sensitivity(
        points,
        [1.0, 0.8, 0.6, 0.4],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        angle_deg=2.0,
        translation_scale=0.5,
        max_iterations=80,
        tolerance_scale=1e-6,
    )
    write_trim_sensitivity_metrics_csv(output_dir / "metrics.csv", results)
    save_trim_sensitivity_plot(output_dir / "comparison.png", results)


def run_joint_sensitivity_experiment(
    input_path: Path,
    output_dir: Path,
) -> None:
    """Run one joint overlap and source-outlier sensitivity evaluation."""
    points = load_xyz(input_path)
    results = evaluate_joint_sensitivity(
        points,
        [1.0, 0.8, 0.6, 0.4],
        [0.0, 0.02, 0.05, 0.1],
        [0.7, 0.4],
        angle_deg=2.0,
        translation_scale=0.5,
        distance_scale=0.25,
        seed=42,
        max_iterations=80,
        tolerance_scale=1e-6,
    )
    write_joint_sensitivity_metrics_csv(output_dir / "metrics.csv", results)
    save_joint_sensitivity_plot(output_dir / "comparison.png", results)


def run_reference_experiments(output_root: Path) -> None:
    """Generate all reference artifacts below ``output_root``."""
    synthetic_path = ROOT / "data" / "synthetic_controlled_density.xyz"
    run_downsampling_experiment(
        synthetic_path,
        output_root / "voxel_downsampling" / "synthetic",
        [0.25, 0.5, 1.0],
    )
    run_downsampling_experiment(
        ROOT / "data" / "usgs_3dep_iowa" / "sample.xyz",
        output_root / "voxel_downsampling" / "usgs_3dep_iowa",
        [5.0, 10.0, 20.0],
    )
    run_outlier_experiment(
        synthetic_path,
        output_root / "outlier_filtering" / "synthetic",
        seed=42,
    )
    run_outlier_experiment(
        ROOT / "data" / "usgs_3dep_iowa" / "sample.xyz",
        output_root / "outlier_filtering" / "usgs_3dep_iowa",
        seed=42,
    )
    run_normal_experiment(
        synthetic_path,
        output_root / "normal_estimation" / "synthetic",
        reference_normals=True,
    )
    run_normal_experiment(
        ROOT / "data" / "usgs_3dep_iowa" / "sample.xyz",
        output_root / "normal_estimation" / "usgs_3dep_iowa",
        reference_normals=False,
    )
    run_registration_experiment(
        synthetic_path,
        output_root / "registration" / "synthetic",
    )
    run_registration_experiment(
        ROOT / "data" / "usgs_3dep_iowa" / "sample.xyz",
        output_root / "registration" / "usgs_3dep_iowa",
    )
    run_partial_overlap_experiment(
        synthetic_path,
        output_root / "partial_overlap_registration" / "synthetic",
    )
    run_partial_overlap_experiment(
        ROOT / "data" / "usgs_3dep_iowa" / "sample.xyz",
        (
            output_root
            / "partial_overlap_registration"
            / "usgs_3dep_iowa"
        ),
    )
    run_trim_sensitivity_experiment(
        synthetic_path,
        output_root / "trim_sensitivity" / "synthetic",
    )
    run_trim_sensitivity_experiment(
        ROOT / "data" / "usgs_3dep_iowa" / "sample.xyz",
        output_root / "trim_sensitivity" / "usgs_3dep_iowa",
    )
    run_joint_sensitivity_experiment(
        synthetic_path,
        output_root / "joint_sensitivity" / "synthetic",
    )
    run_joint_sensitivity_experiment(
        ROOT / "data" / "usgs_3dep_iowa" / "sample.xyz",
        output_root / "joint_sensitivity" / "usgs_3dep_iowa",
    )
    summaries = collect_experiment_summaries(output_root)
    summary_dir = output_root / "summary"
    write_experiment_summary_csv(
        summary_dir / "experiment_summary.csv",
        summaries,
    )
    write_experiment_summary_markdown(summary_dir / "README.md", summaries)
    save_experiment_summary_plot(
        summary_dir / "comparison.png",
        summaries,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the reference-runner argument parser."""
    parser = argparse.ArgumentParser(
        description="Regenerate every committed reference experiment.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results",
        help="Destination root; use a new directory for a non-destructive run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run every reference experiment."""
    args = build_parser().parse_args(argv)
    run_reference_experiments(args.output_root)
    print("Reference experiments completed.")
    print(f"Output root: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
