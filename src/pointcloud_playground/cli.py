"""Command-line interface for point-cloud experiments."""

import argparse
from pathlib import Path
from collections.abc import Sequence

from .evaluation import (
    evaluate_voxel_sizes,
    write_downsampled_clouds,
    write_metrics_csv,
)
from .io import load_xyz, save_xyz
from .synthetic import generate_controlled_density_cloud
from .visualization import save_comparison_plot


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="pointcloud-playground",
        description="Run reproducible point-cloud downsampling experiments.",
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate-demo":
            return _generate_demo(args)
        return _evaluate(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2
