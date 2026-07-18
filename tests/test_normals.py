import numpy as np
import pytest

from pointcloud_playground.normal_evaluation import angular_errors_degrees
from pointcloud_playground.normals import estimate_normals
from pointcloud_playground.synthetic import controlled_surface_normals


def test_pca_normals_recover_horizontal_plane() -> None:
    coordinates = np.linspace(-1.0, 1.0, 12)
    points = np.array(
        [[x, y, 0.0] for x in coordinates for y in coordinates]
    )

    estimate = estimate_normals(points, neighbors=8)

    np.testing.assert_allclose(
        estimate.normals,
        np.tile([0.0, 0.0, 1.0], (len(points), 1)),
        atol=1e-8,
    )
    np.testing.assert_allclose(estimate.surface_variation, 0.0, atol=1e-12)


def test_angular_error_ignores_normal_sign() -> None:
    first = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    second = -first

    errors = angular_errors_degrees(first, second)

    np.testing.assert_allclose(errors, 0.0)


def test_controlled_surface_normals_are_unit_and_upward() -> None:
    points = np.array([[-2.0, 1.0, 0.0], [3.0, -4.0, 0.0]])

    normals = controlled_surface_normals(points)

    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.all(normals[:, 2] > 0.0)


@pytest.mark.parametrize("neighbors", [1, 144])
def test_normal_estimation_rejects_invalid_neighborhood(
    neighbors: int,
) -> None:
    points = np.zeros((144, 3))

    with pytest.raises(ValueError):
        estimate_normals(points, neighbors)
