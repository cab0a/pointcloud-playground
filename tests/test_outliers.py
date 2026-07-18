import numpy as np
import pytest

from pointcloud_playground.outliers import (
    inject_vertical_outliers,
    statistical_outlier_mask,
)
from pointcloud_playground.synthetic import generate_controlled_density_cloud


def test_outlier_injection_is_deterministic_and_separated() -> None:
    clean = generate_controlled_density_cloud(point_count=200, seed=4)

    first = inject_vertical_outliers(clean, outlier_fraction=0.05, seed=9)
    second = inject_vertical_outliers(clean, outlier_fraction=0.05, seed=9)

    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.is_outlier, second.is_outlier)
    injected_z = first.points[first.is_outlier, 2]
    assert np.all(
        (injected_z < clean[:, 2].min()) | (injected_z > clean[:, 2].max())
    )


def test_statistical_filter_detects_isolated_point() -> None:
    cluster = np.array(
        [
            [x, y, z]
            for x in (0.0, 0.1, 0.2)
            for y in (0.0, 0.1, 0.2)
            for z in (0.0, 0.1, 0.2)
        ]
    )
    points = np.vstack((cluster, [[10.0, 10.0, 10.0]]))

    predicted, scores, threshold = statistical_outlier_mask(
        points,
        neighbors=4,
        std_ratio=1.0,
    )

    assert predicted[-1]
    assert np.count_nonzero(predicted) == 1
    assert scores[-1] > threshold


@pytest.mark.parametrize(
    ("outlier_fraction", "distance_scale"),
    [(0.0, 2.0), (1.0, 2.0), (0.05, 0.0)],
)
def test_outlier_injection_rejects_invalid_parameters(
    outlier_fraction: float,
    distance_scale: float,
) -> None:
    with pytest.raises(ValueError):
        inject_vertical_outliers(
            np.zeros((10, 3)),
            outlier_fraction=outlier_fraction,
            distance_scale=distance_scale,
        )
