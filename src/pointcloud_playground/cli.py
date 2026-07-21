"""Command-line interface for point-cloud experiments."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .evaluation import (
    evaluate_voxel_sizes,
    write_downsampled_clouds,
    write_downsampling_metrics_csv,
)
from .filtering_evaluation import (
    evaluate_outlier_filter,
    select_best_filtering_result,
    write_filtered_clouds,
    write_filtering_metrics_csv,
    write_outlier_labels_csv,
)
from .io import load_xyz, save_xyz
from .joint_evaluation import (
    evaluate_joint_sensitivity,
    write_joint_sensitivity_metrics_csv,
)
from .normal_evaluation import (
    evaluate_normal_neighborhoods,
    write_normal_estimates,
    write_normal_metrics_csv,
)
from .overlap_evaluation import (
    evaluate_partial_overlap_cases,
    write_partial_overlap_clouds,
    write_partial_overlap_metrics_csv,
)
from .outliers import inject_vertical_outliers
from .registration_evaluation import (
    evaluate_registration_cases,
    write_aligned_clouds,
    write_registration_metrics_csv,
)
from .summary import (
    collect_experiment_summaries,
    write_experiment_summary_csv,
    write_experiment_summary_markdown,
)
from .synthetic import generate_controlled_density_cloud
from .trim_evaluation import (
    evaluate_trim_sensitivity,
    write_trim_sensitivity_metrics_csv,
)
from .visualization import (
    save_comparison_plot,
    save_experiment_summary_plot,
    save_joint_sensitivity_plot,
    save_normal_evaluation_plot,
    save_outlier_filtering_plot,
    save_partial_overlap_evaluation_plot,
    save_registration_evaluation_plot,
    save_trim_sensitivity_plot,
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
        "evaluate-downsampling",
        aliases=["evaluate"],
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
        default=Path("output/voxel_downsampling"),
        help="Directory for metrics, downsampled clouds, and the figure.",
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
        default=Path("output/outlier_filtering"),
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
        default=Path("output/normal_estimation"),
        help="Directory for metrics, point-level normals, and the figure.",
    )

    registration_parser = subparsers.add_parser(
        "evaluate-registration",
        help="Evaluate point-to-point ICP under known rigid transforms.",
    )
    registration_parser.add_argument("input", type=Path)
    registration_parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=[2.0, 5.0, 10.0, 20.0],
        metavar="DEG",
        help="Paired initial rotation angles in degrees.",
    )
    registration_parser.add_argument(
        "--translation-scales",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0],
        metavar="SCALE",
        help="Paired translation magnitudes relative to median spacing.",
    )
    registration_parser.add_argument(
        "--max-iterations",
        type=int,
        default=60,
        help="Maximum ICP iterations for each controlled case.",
    )
    registration_parser.add_argument(
        "--tolerance-scale",
        type=float,
        default=1e-6,
        help="Convergence tolerance relative to median point spacing.",
    )
    registration_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/registration"),
        help="Directory for metrics, aligned clouds, and the figure.",
    )

    overlap_parser = subparsers.add_parser(
        "evaluate-partial-overlap",
        help="Compare all-pairs and trimmed ICP across scan overlap levels.",
    )
    overlap_parser.add_argument("input", type=Path)
    overlap_parser.add_argument(
        "--overlap-ratios",
        type=float,
        nargs="+",
        default=[1.0, 0.8, 0.6, 0.4],
        metavar="RATIO",
        help="Requested overlap ratios for paired left and right scans.",
    )
    overlap_parser.add_argument(
        "--angle",
        type=float,
        default=2.0,
        help="Initial source rotation in degrees.",
    )
    overlap_parser.add_argument(
        "--translation-scale",
        type=float,
        default=0.5,
        help="Initial translation relative to median point spacing.",
    )
    overlap_parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.7,
        help="Closest source-correspondence fraction retained by trimmed ICP.",
    )
    overlap_parser.add_argument(
        "--max-iterations",
        type=int,
        default=80,
        help="Maximum ICP iterations for each method and overlap level.",
    )
    overlap_parser.add_argument(
        "--tolerance-scale",
        type=float,
        default=1e-6,
        help="Convergence tolerance relative to median point spacing.",
    )
    overlap_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/partial_overlap_registration"),
        help="Directory for metrics, scan clouds, aligned clouds, and the figure.",
    )

    trim_parser = subparsers.add_parser(
        "evaluate-trim-sensitivity",
        help="Evaluate trim fractions with known correspondence diagnostics.",
    )
    trim_parser.add_argument("input", type=Path)
    trim_parser.add_argument(
        "--overlap-ratios",
        type=float,
        nargs="+",
        default=[1.0, 0.8, 0.6, 0.4],
        metavar="RATIO",
        help="Requested overlap ratios for paired left and right scans.",
    )
    trim_parser.add_argument(
        "--trim-fractions",
        type=float,
        nargs="+",
        default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        metavar="FRACTION",
        help="Retained closest-correspondence fractions to evaluate.",
    )
    trim_parser.add_argument(
        "--angle",
        type=float,
        default=2.0,
        help="Initial source rotation in degrees.",
    )
    trim_parser.add_argument(
        "--translation-scale",
        type=float,
        default=0.5,
        help="Initial translation relative to median point spacing.",
    )
    trim_parser.add_argument(
        "--max-iterations",
        type=int,
        default=80,
        help="Maximum ICP iterations for each sensitivity condition.",
    )
    trim_parser.add_argument(
        "--tolerance-scale",
        type=float,
        default=1e-6,
        help="Convergence tolerance relative to median point spacing.",
    )
    trim_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/trim_sensitivity"),
        help="Directory for sensitivity metrics and the diagnostic figure.",
    )

    joint_parser = subparsers.add_parser(
        "evaluate-joint-sensitivity",
        help="Evaluate partial overlap together with controlled source outliers.",
    )
    joint_parser.add_argument("input", type=Path)
    joint_parser.add_argument(
        "--overlap-ratios",
        type=float,
        nargs="+",
        default=[1.0, 0.8, 0.6, 0.4],
        metavar="RATIO",
        help="Requested overlap ratios for paired left and right scans.",
    )
    joint_parser.add_argument(
        "--outlier-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 0.1],
        metavar="FRACTION",
        help="Requested final fractions of appended source outliers.",
    )
    joint_parser.add_argument(
        "--trim-fractions",
        type=float,
        nargs="+",
        default=[0.7, 0.4],
        metavar="FRACTION",
        help="Trimmed fractions to compare; all-pairs is always included.",
    )
    joint_parser.add_argument(
        "--angle",
        type=float,
        default=2.0,
        help="Initial source rotation in degrees.",
    )
    joint_parser.add_argument(
        "--translation-scale",
        type=float,
        default=0.5,
        help="Initial translation relative to median point spacing.",
    )
    joint_parser.add_argument(
        "--distance-scale",
        type=float,
        default=0.25,
        help="Outlier offset relative to a geometry-derived scale.",
    )
    joint_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for controlled source-outlier injection.",
    )
    joint_parser.add_argument(
        "--max-iterations",
        type=int,
        default=80,
        help="Maximum ICP iterations for each joint condition.",
    )
    joint_parser.add_argument(
        "--tolerance-scale",
        type=float,
        default=1e-6,
        help="Convergence tolerance relative to median point spacing.",
    )
    joint_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/joint_sensitivity"),
        help="Directory for joint-sensitivity metrics and the figure.",
    )

    summary_parser = subparsers.add_parser(
        "summarize-results",
        help="Create a cross-experiment review from canonical result files.",
    )
    summary_parser.add_argument(
        "results_root",
        type=Path,
        help="Root containing <experiment>/<dataset>/metrics.csv files.",
    )
    summary_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/summary"),
        help="Directory for the summary CSV, Markdown, and figure.",
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
    metrics_path = write_downsampling_metrics_csv(
        args.output_dir / "metrics.csv",
        results,
    )
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


def _evaluate_registration(args: argparse.Namespace) -> int:
    points = load_xyz(args.input)
    results = evaluate_registration_cases(
        points,
        args.angles,
        args.translation_scales,
        max_iterations=args.max_iterations,
        tolerance_scale=args.tolerance_scale,
    )
    metrics_path = write_registration_metrics_csv(
        args.output_dir / "metrics.csv",
        results,
    )
    write_aligned_clouds(args.output_dir, results)
    comparison_path = save_registration_evaluation_plot(
        args.output_dir / "comparison.png",
        points,
        results,
    )

    print(f"Input points: {len(points)}")
    for result in results:
        print(
            f"{result.case}: rotation error "
            f"{result.rotation_error_deg:.3f} deg, "
            f"normalized RMSE {result.normalized_correspondence_rmse:.3f}"
        )
    print(f"Metrics: {metrics_path}")
    print(f"Comparison: {comparison_path}")
    return 0


def _evaluate_partial_overlap(args: argparse.Namespace) -> int:
    points = load_xyz(args.input)
    results = evaluate_partial_overlap_cases(
        points,
        args.overlap_ratios,
        angle_deg=args.angle,
        translation_scale=args.translation_scale,
        trim_fraction=args.trim_fraction,
        max_iterations=args.max_iterations,
        tolerance_scale=args.tolerance_scale,
    )
    metrics_path = write_partial_overlap_metrics_csv(
        args.output_dir / "metrics.csv",
        results,
    )
    write_partial_overlap_clouds(args.output_dir, results)
    comparison_path = save_partial_overlap_evaluation_plot(
        args.output_dir / "comparison.png",
        results,
    )

    print(f"Input points: {len(points)}")
    for result in results:
        print(
            f"Overlap {result.actual_overlap_ratio:.1%} | "
            f"{result.method}: "
            f"known-overlap RMSE "
            f"{result.normalized_overlap_correspondence_rmse:.3f}, "
            f"recovered {result.recovered}"
        )
    print(f"Metrics: {metrics_path}")
    print(f"Comparison: {comparison_path}")
    return 0


def _evaluate_trim_sensitivity(args: argparse.Namespace) -> int:
    points = load_xyz(args.input)
    results = evaluate_trim_sensitivity(
        points,
        args.overlap_ratios,
        args.trim_fractions,
        angle_deg=args.angle,
        translation_scale=args.translation_scale,
        max_iterations=args.max_iterations,
        tolerance_scale=args.tolerance_scale,
    )
    metrics_path = write_trim_sensitivity_metrics_csv(
        args.output_dir / "metrics.csv",
        results,
    )
    comparison_path = save_trim_sensitivity_plot(
        args.output_dir / "comparison.png",
        results,
    )

    print(f"Input points: {len(points)}")
    for result in results:
        print(
            f"Overlap {result.actual_overlap_ratio:.1%} | "
            f"fraction {result.trim_fraction:.1f}: "
            f"precision {result.correct_match_precision:.3f}, "
            f"recall {result.correct_match_recall:.3f}, "
            f"recovered {result.recovered}"
        )
    print(f"Metrics: {metrics_path}")
    print(f"Comparison: {comparison_path}")
    return 0


def _evaluate_joint_sensitivity(args: argparse.Namespace) -> int:
    points = load_xyz(args.input)
    results = evaluate_joint_sensitivity(
        points,
        args.overlap_ratios,
        args.outlier_fractions,
        args.trim_fractions,
        angle_deg=args.angle,
        translation_scale=args.translation_scale,
        distance_scale=args.distance_scale,
        seed=args.seed,
        max_iterations=args.max_iterations,
        tolerance_scale=args.tolerance_scale,
    )
    metrics_path = write_joint_sensitivity_metrics_csv(
        args.output_dir / "metrics.csv",
        results,
    )
    comparison_path = save_joint_sensitivity_plot(
        args.output_dir / "comparison.png",
        results,
    )

    print(f"Input points: {len(points)}")
    print(f"Conditions: {len(results)}")
    for method in dict.fromkeys(result.method for result in results):
        method_results = [
            result for result in results if result.method == method
        ]
        recovered = sum(result.recovered for result in method_results)
        print(f"{method}: recovered {recovered}/{len(method_results)}")
    print(f"Metrics: {metrics_path}")
    print(f"Comparison: {comparison_path}")
    return 0


def _summarize_results(args: argparse.Namespace) -> int:
    summaries = collect_experiment_summaries(args.results_root)
    csv_path = write_experiment_summary_csv(
        args.output_dir / "experiment_summary.csv",
        summaries,
    )
    markdown_path = write_experiment_summary_markdown(
        args.output_dir / "README.md",
        summaries,
    )
    comparison_path = save_experiment_summary_plot(
        args.output_dir / "comparison.png",
        summaries,
    )

    print(f"Experiments: {len({item.experiment for item in summaries})}")
    print(f"Datasets: {len({item.dataset for item in summaries})}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary Markdown: {markdown_path}")
    print(f"Comparison: {comparison_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate-demo":
            return _generate_demo(args)
        if args.command in {"evaluate-downsampling", "evaluate"}:
            return _evaluate(args)
        if args.command == "evaluate-outliers":
            return _evaluate_outliers(args)
        if args.command == "evaluate-normals":
            return _evaluate_normals(args)
        if args.command == "evaluate-registration":
            return _evaluate_registration(args)
        if args.command == "evaluate-partial-overlap":
            return _evaluate_partial_overlap(args)
        if args.command == "evaluate-trim-sensitivity":
            return _evaluate_trim_sensitivity(args)
        if args.command == "evaluate-joint-sensitivity":
            return _evaluate_joint_sensitivity(args)
        return _summarize_results(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2
