"""Quantitative evaluation of voxel downsampling."""

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .downsampling import voxel_downsample
from .io import PointCloud, save_xyz, validate_points


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and output points for one voxel size."""

    voxel_size: float
    input_points: int
    output_points: int
    retention_ratio: float
    input_mean_nn_distance: float
    output_mean_nn_distance: float
    coverage_rmse: float
    points: PointCloud

    def as_row(self) -> dict[str, int | float]:
        """Return serializable metrics without the point array."""
        return {
            "voxel_size": self.voxel_size,
            "input_points": self.input_points,
            "output_points": self.output_points,
            "retention_ratio": self.retention_ratio,
            "input_mean_nn_distance": self.input_mean_nn_distance,
            "output_mean_nn_distance": self.output_mean_nn_distance,
            "coverage_rmse": self.coverage_rmse,
        }


def mean_nearest_neighbor_distance(points: NDArray[np.floating]) -> float:
    """Return the mean distance to each point's nearest other point."""
    cloud = validate_points(points)
    if len(cloud) < 2:
        return float("nan")
    distances, _ = cKDTree(cloud).query(cloud, k=2)
    return float(np.mean(distances[:, 1]))


def coverage_rmse(
    reference_points: NDArray[np.floating],
    sampled_points: NDArray[np.floating],
) -> float:
    """Measure reference-to-sample coverage using nearest-neighbor RMSE."""
    reference = validate_points(reference_points)
    sampled = validate_points(sampled_points)
    distances, _ = cKDTree(sampled).query(reference, k=1)
    return float(np.sqrt(np.mean(np.square(distances))))


def evaluate_voxel_sizes(
    points: NDArray[np.floating], voxel_sizes: list[float]
) -> list[EvaluationResult]:
    """Evaluate voxel downsampling over a sequence of voxel sizes."""
    cloud = validate_points(points)
    if not voxel_sizes:
        raise ValueError("At least one voxel size is required.")

    input_nn = mean_nearest_neighbor_distance(cloud)
    results: list[EvaluationResult] = []
    for voxel_size in voxel_sizes:
        sampled = voxel_downsample(cloud, voxel_size)
        results.append(
            EvaluationResult(
                voxel_size=float(voxel_size),
                input_points=len(cloud),
                output_points=len(sampled),
                retention_ratio=len(sampled) / len(cloud),
                input_mean_nn_distance=input_nn,
                output_mean_nn_distance=mean_nearest_neighbor_distance(sampled),
                coverage_rmse=coverage_rmse(cloud, sampled),
                points=sampled,
            )
        )
    return results


def write_downsampling_metrics_csv(
    path: str | Path, results: list[EvaluationResult]
) -> Path:
    """Write voxel-downsampling evaluation metrics to CSV."""
    if not results:
        raise ValueError("At least one evaluation result is required.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].as_row())
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result.as_row() for result in results)
    return output_path


def write_metrics_csv(
    path: str | Path, results: list[EvaluationResult]
) -> Path:
    """Write metrics using the pre-v0.5 compatibility name."""
    return write_downsampling_metrics_csv(path, results)


def voxel_size_label(voxel_size: float) -> str:
    """Return a filesystem-safe label for a voxel size."""
    return f"{voxel_size:g}".replace("-", "m").replace(".", "p")


def write_downsampled_clouds(
    output_dir: str | Path, results: list[EvaluationResult]
) -> list[Path]:
    """Write one XYZ point cloud for each evaluation result."""
    directory = Path(output_dir)
    return [
        save_xyz(
            directory / f"downsampled_{voxel_size_label(result.voxel_size)}.xyz",
            result.points,
        )
        for result in results
    ]
