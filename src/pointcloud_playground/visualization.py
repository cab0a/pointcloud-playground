"""Static visualizations for point-cloud experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .evaluation import EvaluationResult
from .filtering_evaluation import FilteringResult
from .io import validate_points
from .normal_evaluation import NormalEvaluationResult
from .outliers import OutlierMask


def _plot_sample(points: NDArray[np.floating], limit: int) -> NDArray[np.float64]:
    cloud = validate_points(points)
    if len(cloud) <= limit:
        return cloud
    indices = np.linspace(0, len(cloud) - 1, limit, dtype=int)
    return cloud[indices]


def save_comparison_plot(
    path: str | Path,
    original_points: NDArray[np.floating],
    results: list[EvaluationResult],
    plot_limit: int = 20_000,
) -> Path:
    """Save an XY comparison plot colored by height."""
    if not results:
        raise ValueError("At least one evaluation result is required.")

    original = validate_points(original_points)
    clouds = [original, *(result.points for result in results)]
    samples = [_plot_sample(cloud, plot_limit) for cloud in clouds]
    titles = [
        f"Input\n{len(original):,} points",
        *(
            f"Voxel {result.voxel_size:g}\n{result.output_points:,} points"
            for result in results
        ),
    ]

    x_min, y_min = original[:, :2].min(axis=0)
    x_max, y_max = original[:, :2].max(axis=0)
    z_min, z_max = original[:, 2].min(), original[:, 2].max()

    figure, axes = plt.subplots(
        1,
        len(samples),
        figsize=(4.2 * len(samples), 4.2),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, sample, title in zip(axes[0], samples, titles):
        axis.scatter(
            sample[:, 0],
            sample[:, 1],
            c=sample[:, 2],
            cmap="viridis",
            vmin=z_min,
            vmax=z_max,
            s=2,
            linewidths=0,
        )
        axis.set(
            title=title,
            xlabel="X",
            ylabel="Y",
            xlim=(x_min, x_max),
            ylim=(y_min, y_max),
            aspect="equal",
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def save_outlier_filtering_plot(
    path: str | Path,
    contaminated_points: NDArray[np.floating],
    true_outlier_mask: OutlierMask,
    result: FilteringResult,
) -> Path:
    """Save a three-panel comparison of truth, removals, and output."""
    points = validate_points(contaminated_points)
    truth = np.asarray(true_outlier_mask, dtype=np.bool_)
    if truth.ndim != 1 or len(truth) != len(points):
        raise ValueError("Outlier labels must contain one value per point.")
    if len(result.predicted_mask) != len(points):
        raise ValueError("Predictions must contain one value per point.")

    predicted = result.predicted_mask
    true_positive = truth & predicted
    false_positive = ~truth & predicted
    false_negative = truth & ~predicted
    retained_inliers = ~truth & ~predicted

    figure = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    axes = [
        figure.add_subplot(1, 3, index, projection="3d")
        for index in range(1, 4)
    ]

    axes[0].scatter(
        points[~truth, 0],
        points[~truth, 1],
        points[~truth, 2],
        c="#7f8c8d",
        s=2,
        linewidths=0,
        label="Inlier",
    )
    axes[0].scatter(
        points[truth, 0],
        points[truth, 1],
        points[truth, 2],
        c="#d62728",
        s=7,
        linewidths=0,
        label="Injected outlier",
    )
    axes[0].set_title(
        f"Ground truth\n{np.count_nonzero(truth):,} injected outliers"
    )
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].scatter(
        points[~predicted, 0],
        points[~predicted, 1],
        points[~predicted, 2],
        c="#d9d9d9",
        s=1,
        linewidths=0,
    )
    axes[1].scatter(
        points[true_positive, 0],
        points[true_positive, 1],
        points[true_positive, 2],
        c="#d62728",
        s=7,
        linewidths=0,
        label="Correct removal",
    )
    axes[1].scatter(
        points[false_positive, 0],
        points[false_positive, 1],
        points[false_positive, 2],
        c="#ff7f0e",
        s=7,
        linewidths=0,
        label="False removal",
    )
    axes[1].set_title(
        f"Removed points\nTP {result.true_positive:,} | "
        f"FP {result.false_positive:,}"
    )
    axes[1].legend(loc="upper right", fontsize=8)

    retained = points[retained_inliers]
    axes[2].scatter(
        retained[:, 0],
        retained[:, 1],
        retained[:, 2],
        c=retained[:, 2],
        cmap="viridis",
        s=2,
        linewidths=0,
    )
    axes[2].scatter(
        points[false_negative, 0],
        points[false_negative, 1],
        points[false_negative, 2],
        c="#d62728",
        s=7,
        linewidths=0,
        label="Missed outlier",
    )
    axes[2].set_title(
        f"Filtered output\nF1 {result.f1:.3f} | "
        f"inliers retained {result.inlier_retention:.1%}"
    )
    if false_negative.any():
        axes[2].legend(loc="upper right", fontsize=8)

    spans = np.ptp(points, axis=0)
    box_aspect = np.maximum(spans / max(float(spans.max()), 1.0), 0.25)
    for axis in axes:
        axis.set(xlabel="X", ylabel="Y", zlabel="Z")
        axis.set_box_aspect(box_aspect)
        axis.view_init(elev=24, azim=-60)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def save_normal_evaluation_plot(
    path: str | Path,
    points: NDArray[np.floating],
    results: list[NormalEvaluationResult],
    plot_limit: int = 220,
) -> Path:
    """Save normal vectors and neighborhood-selection metrics."""
    cloud = validate_points(points)
    if not results:
        raise ValueError("At least one normal-estimation result is required.")

    display_result = results[len(results) // 2]
    sample_count = min(len(cloud), plot_limit)
    sample_indices = np.linspace(
        0,
        len(cloud) - 1,
        sample_count,
        dtype=int,
    )
    sample = cloud[sample_indices]
    sample_normals = display_result.normals[sample_indices]
    vector_length = max(
        display_result.median_neighborhood_radius * 0.35,
        np.finfo(np.float64).eps,
    )

    figure = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    normal_axis = figure.add_subplot(1, 3, 1, projection="3d")
    error_axis = figure.add_subplot(1, 3, 2)
    scale_axis = figure.add_subplot(1, 3, 3)

    normal_axis.scatter(
        sample[:, 0],
        sample[:, 1],
        sample[:, 2],
        c=sample[:, 2],
        cmap="viridis",
        s=5,
        linewidths=0,
    )
    normal_axis.quiver(
        sample[:, 0],
        sample[:, 1],
        sample[:, 2],
        sample_normals[:, 0],
        sample_normals[:, 1],
        sample_normals[:, 2],
        length=vector_length,
        normalize=True,
        color="#d62728",
        linewidth=0.55,
    )
    normal_axis.set_title(
        f"Estimated normals\nk={display_result.neighbors}"
    )
    normal_axis.set(xlabel="X", ylabel="Y", zlabel="Z")
    spans = np.ptp(cloud, axis=0)
    box_aspect = np.maximum(spans / max(float(spans.max()), 1.0), 0.25)
    normal_axis.set_box_aspect(box_aspect)
    normal_axis.view_init(elev=28, azim=-60)

    neighborhoods = [result.neighbors for result in results]
    error_axis.plot(
        neighborhoods,
        [result.mean_repeatability_error_deg for result in results],
        marker="o",
        label="Mean perturbation error",
    )
    error_axis.plot(
        neighborhoods,
        [result.p95_repeatability_error_deg for result in results],
        marker="o",
        linestyle="--",
        label="P95 perturbation error",
    )
    if results[0].mean_angular_error_deg is not None:
        error_axis.plot(
            neighborhoods,
            [result.mean_angular_error_deg for result in results],
            marker="s",
            label="Mean reference error",
        )
        error_axis.plot(
            neighborhoods,
            [result.p95_angular_error_deg for result in results],
            marker="s",
            linestyle="--",
            label="P95 reference error",
        )
    error_axis.set(
        title="Angular error",
        xlabel="Neighbors (k)",
        ylabel="Degrees",
        xticks=neighborhoods,
    )
    error_axis.grid(alpha=0.25)
    error_axis.legend(fontsize=8)

    scale_axis.plot(
        neighborhoods,
        [result.median_neighborhood_radius for result in results],
        color="#1f77b4",
        marker="o",
        label="Median radius",
    )
    scale_axis.set(
        title="Support scale and surface variation",
        xlabel="Neighbors (k)",
        ylabel="Median neighborhood radius",
        xticks=neighborhoods,
    )
    scale_axis.tick_params(axis="y", labelcolor="#1f77b4")
    scale_axis.grid(alpha=0.25)

    variation_axis = scale_axis.twinx()
    variation_axis.plot(
        neighborhoods,
        [result.mean_surface_variation for result in results],
        color="#ff7f0e",
        marker="s",
        label="Mean surface variation",
    )
    variation_axis.set_ylabel(
        "Mean surface variation",
        color="#ff7f0e",
    )
    variation_axis.tick_params(axis="y", labelcolor="#ff7f0e")

    handles, labels = scale_axis.get_legend_handles_labels()
    second_handles, second_labels = variation_axis.get_legend_handles_labels()
    scale_axis.legend(
        handles + second_handles,
        labels + second_labels,
        fontsize=8,
        loc="best",
    )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path
