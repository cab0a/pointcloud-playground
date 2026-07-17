import numpy as np
import pytest

from pointcloud_playground.synthetic import generate_controlled_density_cloud


def test_synthetic_cloud_is_deterministic() -> None:
    first = generate_controlled_density_cloud(point_count=200, seed=11)
    second = generate_controlled_density_cloud(point_count=200, seed=11)

    np.testing.assert_array_equal(first, second)


def test_synthetic_cloud_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        generate_controlled_density_cloud(point_count=99)
