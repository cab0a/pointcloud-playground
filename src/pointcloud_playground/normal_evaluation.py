"""Evaluation helpers for point-cloud normal estimation."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .io import PointCloud, validate_points
from .normals import NormalArray, estimate_normals


@dataclass(frozen=True)
class NormalEvaluationResult:
    """Metrics and estimated normals for one neighborhood size."""

    neighbors: int
    point_count: int
    perturbation_std: float
    mean_neighborhood_radius: float
    median_neighborhood_radius: float
    mean_surface_variation: float
    median_surface_variation: float
    mean_angular_error_deg: float | None
    median_angular_error_deg: float | None
    p95_angular_error_deg: float | None
    mean_repeatability_error_deg: float
    median_repeatability_error_deg: float
    p95_repeatability_error_deg: float
    normals: NormalArray
    surface_variation: NDArray[np.float64]

    def as_row(self) -> dict[str, int | float | None]:
        """Return serializable metrics without point-level arrays."""
        return {
            "neighbors": self.neighbors,
            "point_count": self.point_count,
            "perturbation_std": self.perturbation_std,
            "mean_neighborhood_radius": self.mean_neighborhood_radius,
            "median_neighborhood_radius": self.median_neighborhood_radius,
            "mean_surface_variation": self.mean_surface_variation,
            "median_surface_variation": self.median_surface_variation,
            "mean_angular_error_deg": self.mean_angular_error_deg,
            "median_angular_error_deg": self.median_angular_error_deg,
            "p95_angular_error_deg": self.p95_angular_error_deg,
            "mean_repeatability_error_deg": (
                self.mean_repeatability_error_deg
            ),
            "median_repeatability_error_deg": (
                self.median_repeatability_error_deg
            ),
            "p95_repeatability_error_deg": self.p95_repeatability_error_deg,
        }


def _validate_normals(
    normals: NDArray[np.floating],
    point_count: int,
) -> NormalArray:
    array = np.asarray(normals, dtype=np.float64)
    if array.shape != (point_count, 3):
        raise ValueError("Reference normals must have shape (n, 3).")
    if not np.isfinite(array).all():
        raise ValueError("Reference normals must contain only finite values.")
    lengths = np.linalg.norm(array, axis=1)
    if np.any(lengths == 0.0):
        raise ValueError("Reference normals must have non-zero length.")
    return array / lengths[:, np.newaxis]


def angular_errors_degrees(
    first: NDArray[np.floating],
    second: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return orientation-invariant angular differences in degrees."""
    first_normals = _validate_normals(first, len(first))
    second_normals = _validate_normals(second, len(first_normals))
    cosine = np.abs(np.einsum("ij,ij->i", first_normals, second_normals))
    return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))


def evaluate_normal_neighborhoods(
    points: NDArray[np.floating],
    neighborhood_sizes: list[int],
    reference_normals: NDArray[np.floating] | None = None,
    noise_scale: float = 0.05,
    seed: int = 42,
) -> list[NormalEvaluationResult]:
    """Evaluate normal accuracy and perturbation repeatability."""
    cloud = validate_points(points)
    if not neighborhood_sizes:
        raise ValueError("At least one neighborhood size is required.")
    if not np.isfinite(noise_scale) or noise_scale <= 0.0:
        raise ValueError("Noise scale must be a positive finite number.")

    truth = (
        None
        if reference_normals is None
        else _validate_normals(reference_normals, len(cloud))
    )
    nearest_distances, _ = cKDTree(cloud).query(cloud, k=2)
    positive_spacing = nearest_distances[:, 1][
        nearest_distances[:, 1] > 0.0
    ]
    if len(positive_spacing) == 0:
        raise ValueError("Normal evaluation requires distinct points.")
    median_spacing = float(np.median(positive_spacing))
    perturbation_std = median_spacing * noise_scale
    rng = np.random.default_rng(seed)
    perturbed = cloud + rng.normal(
        0.0,
        perturbation_std,
        size=cloud.shape,
    )

    results: list[NormalEvaluationResult] = []
    for neighbors in neighborhood_sizes:
        estimate = estimate_normals(cloud, neighbors)
        perturbed_estimate = estimate_normals(perturbed, neighbors)
        repeatability = angular_errors_degrees(
            estimate.normals,
            perturbed_estimate.normals,
        )
        accuracy = (
            None
            if truth is None
            else angular_errors_degrees(estimate.normals, truth)
        )
        results.append(
            NormalEvaluationResult(
                neighbors=neighbors,
                point_count=len(cloud),
                perturbation_std=perturbation_std,
                mean_neighborhood_radius=float(
                    np.mean(estimate.neighborhood_radius)
                ),
                median_neighborhood_radius=float(
                    np.median(estimate.neighborhood_radius)
                ),
                mean_surface_variation=float(
                    np.mean(estimate.surface_variation)
                ),
                median_surface_variation=float(
                    np.median(estimate.surface_variation)
                ),
                mean_angular_error_deg=(
                    None if accuracy is None else float(np.mean(accuracy))
                ),
                median_angular_error_deg=(
                    None if accuracy is None else float(np.median(accuracy))
                ),
                p95_angular_error_deg=(
                    None
                    if accuracy is None
                    else float(np.percentile(accuracy, 95))
                ),
                mean_repeatability_error_deg=float(
                    np.mean(repeatability)
                ),
                median_repeatability_error_deg=float(
                    np.median(repeatability)
                ),
                p95_repeatability_error_deg=float(
                    np.percentile(repeatability, 95)
                ),
                normals=estimate.normals,
                surface_variation=estimate.surface_variation,
            )
        )
    return results


def write_normal_metrics_csv(
    path: str | Path,
    results: list[NormalEvaluationResult],
) -> Path:
    """Write neighborhood-level normal-estimation metrics to CSV."""
    if not results:
        raise ValueError("At least one normal-estimation result is required.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(results[0].as_row()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result.as_row() for result in results)
    return output_path


def write_normal_estimates(
    output_dir: str | Path,
    points: NDArray[np.floating],
    results: list[NormalEvaluationResult],
) -> list[Path]:
    """Write point coordinates, normals, and surface variation to CSV."""
    cloud = validate_points(points)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for result in results:
        if (
            result.normals.shape != cloud.shape
            or len(result.surface_variation) != len(cloud)
        ):
            raise ValueError(
                "Normal estimates must contain one result per point."
            )
        path = directory / f"normals_k{result.neighbors}.csv"
        values = np.column_stack(
            (
                cloud,
                result.normals,
                result.surface_variation,
            )
        )
        np.savetxt(
            path,
            values,
            delimiter=",",
            header="x,y,z,nx,ny,nz,surface_variation",
            comments="",
            fmt="%.8f",
        )
        paths.append(path)
    return paths
