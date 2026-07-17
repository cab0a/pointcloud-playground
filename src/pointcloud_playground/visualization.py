"""Static visualizations for point-cloud experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .evaluation import EvaluationResult
from .io import validate_points


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
