"""Command-line interface for point-cloud experiments."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .evaluation import (
    evaluate_voxel_sizes,
    write_downsampled_clouds,
    write_metrics_csv,
)
from .filtering_evaluation import (
    evaluate_outlier_filter,
    select_best_filtering_result,
    write_filtered_clouds,
    write_filtering_metrics_csv,
    write_outlier_labels_csv,
)
from .io import load_xyz, save_xyz
from .normal_evaluation import (
    evaluate_normal_neighborhoods,
    write_normal_estimates,
    write_normal_metrics_csv,
)
from .outliers import inject_vertical_outliers
from .synthetic import generate_controlled_density_cloud
from .visualization import (
    save_comparison_plot,
    save_normal_evaluation_plot,
    save_outlier_filtering_plot,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="pointcloud-playground",
        description="Run reproducible point-cloud processing experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate-demo",
        help="Generate a deterministic controlled-density point cloud.",
    )
    generate_parser.add_argument("output", type=Path)
    generate_parser.add_argument("--points", type=int, default=6_000)
    generate_parser.add_argument("--seed", type=int, default=42)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate voxel downsampling for an XYZ point cloud.",
    )
    evaluate_parser.add_argument("input", type=Path)
    evaluate_parser.add_argument(
        "--voxel-sizes",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        metavar="SIZE",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )

    outlier_parser = subparsers.add_parser(
        "evaluate-outliers",
        help="Inject known outliers and evaluate statistical filtering.",
    )
    outlier_parser.add_argument("input", type=Path)
    outlier_parser.add_argument(
        "--outlier-fraction",
        type=float,
        default=0.05,
        help="Fraction of injected points in the contaminated cloud.",
    )
    outlier_parser.add_argument(
        "--distance-scale",
        type=float,
        default=0.25,
        help="Outlier offset relative to a geometry-derived scale.",
    )
    outlier_parser.add_argument(
        "--neighbors",
        type=int,
        default=16,
        help="Number of nearest neighbors used for each distance score.",
    )
    outlier_parser.add_argument(
        "--std-ratios",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        metavar="RATIO",
        help="Standard-deviation ratios evaluated as threshold settings.",
    )
    outlier_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for controlled outlier injection.",
    )
    outlier_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/outliers"),
        help="Directory for metrics, point clouds, labels, and the figure.",
    )

    normal_parser = subparsers.add_parser(
        "evaluate-normals",
        help="Evaluate PCA normal stability across neighborhood sizes.",
    )
    normal_parser.add_argument("input", type=Path)
    normal_parser.add_argument(
        "--neighbors",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        metavar="K",
        help="Neighborhood sizes evaluated by local PCA.",
    )
    normal_parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.05,
        help="Perturbation standard deviation relative to median spacing.",
    )
    normal_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for controlled coordinate perturbation.",
    )
    normal_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/normals"),
        help="Directory for metrics, point-level normals, and the figure.",
    )
    return parser


def _generate_demo(args: argparse.Namespace) -> int:
    points = generate_controlled_density_cloud(args.points, args.seed)
    output_path = save_xyz(args.output, points)
    print(f"Points: {len(points)}")
    print(f"Output: {output_path}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    points = load_xyz(args.input)
    results = evaluate_voxel_sizes(points, args.voxel_sizes)
    metrics_path = write_metrics_csv(args.output_dir / "metrics.csv", results)
    comparison_path = save_comparison_plot(
        args.output_dir / "comparison.png",
        points,
        results,
    )
    write_downsampled_clouds(args.output_dir, results)

    print(f"Input points: {len(points)}")
    for result in results:
        print(
            f"Voxel {result.voxel_size:g}: "
            f"{result.output_points} points "
            f"({result.retention_ratio:.1%} retained)"
        )
    print(f"Metrics: {metrics_path}")
    print(f"Comparison: {comparison_path}")
    return 0


def _evaluate_outliers(args: argparse.Namespace) -> int:
    clean = load_xyz(args.input)
    contaminated = inject_vertical_outliers(
        clean,
        outlier_fraction=args.outlier_fraction,
        distance_scale=args.distance_scale,
        seed=args.seed,
    )
    results = evaluate_outlier_filter(
        contaminated.points,
        contaminated.is_outlier,
        clean,
        neighbors=args.neighbors,
        std_ratios=args.std_ratios,
    )
    best = select_best_filtering_result(results)

    save_xyz(args.output_dir / "contaminated.xyz", contaminated.points)
    write_outlier_labels_csv(
        args.output_dir / "labels.csv",
        contaminated.is_outlier,
    )
    metrics_path = write_filtering_metrics_csv(
        args.output_dir / "metrics.csv",
        results,
    )
    write_filtered_clouds(args.output_dir, results)
    comparison_path = save_outlier_filtering_plot(
        args.output_dir / "comparison.png",
        contaminated.points,
        contaminated.is_outlier,
        best,
    )

    print(f"Clean points: {len(clean)}")
    print(f"Injected outliers: {np.count_nonzero(contaminated.is_outlier)}")
    print(f"Best std ratio: {best.std_ratio:g}")
    print(f"Precision: {best.precision:.3f}")
    print(f"Recall: {best.recall:.3f}")
    print(f"F1: {best.f1:.3f}")
    print(f"Metrics: {metrics_path}")
    print(f"Comparison: {comparison_path}")
    return 0


def _evaluate_normals(args: argparse.Namespace) -> int:
    points = load_xyz(args.input)
    results = evaluate_normal_neighborhoods(
        points,
        args.neighbors,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
    metrics_path = write_normal_metrics_csv(
        args.output_dir / "metrics.csv",
        results,
    )
    write_normal_estimates(args.output_dir, points, results)
    comparison_path = save_normal_evaluation_plot(
        args.output_dir / "comparison.png",
        points,
        results,
    )

    print(f"Input points: {len(points)}")
    for result in results:
        print(
            f"Neighborhood {result.neighbors}: "
            f"median repeatability error "
            f"{result.median_repeatability_error_deg:.3f} deg"
        )
    print(f"Metrics: {metrics_path}")
    print(f"Comparison: {comparison_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate-demo":
            return _generate_demo(args)
        if args.command == "evaluate":
            return _evaluate(args)
        if args.command == "evaluate-outliers":
            return _evaluate_outliers(args)
        return _evaluate_normals(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2
