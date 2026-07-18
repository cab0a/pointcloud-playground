"""Evaluation helpers for controlled outlier-filtering experiments."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .evaluation import coverage_rmse
from .io import PointCloud, save_xyz, validate_points
from .outliers import OutlierMask, statistical_outlier_mask


@dataclass(frozen=True)
class FilteringResult:
    """Metrics and predictions for one statistical threshold."""

    std_ratio: float
    neighbors: int
    threshold: float
    input_points: int
    true_outliers: int
    predicted_outliers: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    inlier_retention: float
    clean_coverage_rmse: float
    points: PointCloud
    predicted_mask: OutlierMask

    def as_row(self) -> dict[str, int | float]:
        """Return serializable metrics without point arrays or masks."""
        return {
            "std_ratio": self.std_ratio,
            "neighbors": self.neighbors,
            "threshold": self.threshold,
            "input_points": self.input_points,
            "true_outliers": self.true_outliers,
            "predicted_outliers": self.predicted_outliers,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "inlier_retention": self.inlier_retention,
            "clean_coverage_rmse": self.clean_coverage_rmse,
        }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def validate_outlier_labels(
    labels: NDArray[np.bool_], point_count: int
) -> OutlierMask:
    """Validate and return one Boolean label for every point."""
    mask = np.asarray(labels, dtype=np.bool_)
    if mask.ndim != 1 or len(mask) != point_count:
        raise ValueError("Outlier labels must contain one value per point.")
    if not mask.any() or mask.all():
        raise ValueError("Outlier labels must contain both classes.")
    return mask


def evaluate_outlier_filter(
    contaminated_points: NDArray[np.floating],
    true_outlier_mask: NDArray[np.bool_],
    clean_points: NDArray[np.floating],
    neighbors: int,
    std_ratios: list[float],
) -> list[FilteringResult]:
    """Evaluate statistical filtering against known outlier labels."""
    contaminated = validate_points(contaminated_points)
    clean = validate_points(clean_points)
    truth = validate_outlier_labels(true_outlier_mask, len(contaminated))
    if not std_ratios:
        raise ValueError("At least one standard-deviation ratio is required.")

    actual_outliers = int(np.count_nonzero(truth))
    actual_inliers = len(truth) - actual_outliers
    results: list[FilteringResult] = []
    for std_ratio in std_ratios:
        predicted, _, threshold = statistical_outlier_mask(
            contaminated,
            neighbors,
            std_ratio,
        )
        true_positive = int(np.count_nonzero(predicted & truth))
        false_positive = int(np.count_nonzero(predicted & ~truth))
        false_negative = int(np.count_nonzero(~predicted & truth))
        true_negative = int(np.count_nonzero(~predicted & ~truth))
        precision = _safe_ratio(
            true_positive,
            true_positive + false_positive,
        )
        recall = _safe_ratio(
            true_positive,
            true_positive + false_negative,
        )
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        filtered = contaminated[~predicted]
        retained_inliers = contaminated[~predicted & ~truth]

        results.append(
            FilteringResult(
                std_ratio=float(std_ratio),
                neighbors=neighbors,
                threshold=threshold,
                input_points=len(contaminated),
                true_outliers=actual_outliers,
                predicted_outliers=int(np.count_nonzero(predicted)),
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                precision=precision,
                recall=recall,
                f1=f1,
                inlier_retention=_safe_ratio(true_negative, actual_inliers),
                clean_coverage_rmse=coverage_rmse(clean, retained_inliers),
                points=filtered,
                predicted_mask=predicted,
            )
        )
    return results


def select_best_filtering_result(
    results: list[FilteringResult],
) -> FilteringResult:
    """Select the highest-F1 result, preferring better inlier retention."""
    if not results:
        raise ValueError("At least one filtering result is required.")
    return max(
        results,
        key=lambda result: (
            result.f1,
            result.inlier_retention,
            -result.std_ratio,
        ),
    )


def write_filtering_metrics_csv(
    path: str | Path, results: list[FilteringResult]
) -> Path:
    """Write outlier-filtering metrics to CSV."""
    if not results:
        raise ValueError("At least one filtering result is required.")

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


def write_outlier_labels_csv(
    path: str | Path, labels: NDArray[np.bool_]
) -> Path:
    """Write ground-truth labels in point order."""
    mask = np.asarray(labels, dtype=np.bool_)
    if mask.ndim != 1:
        raise ValueError("Outlier labels must be one-dimensional.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("is_outlier",))
        writer.writerows((int(value),) for value in mask)
    return output_path


def _parameter_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def write_filtered_clouds(
    output_dir: str | Path, results: list[FilteringResult]
) -> list[Path]:
    """Write one filtered XYZ file for every threshold setting."""
    directory = Path(output_dir)
    return [
        save_xyz(
            directory / f"filtered_std_{_parameter_label(result.std_ratio)}.xyz",
            result.points,
        )
        for result in results
    ]
