import numpy as np
import pytest

from pointcloud_playground.registration import (
    RigidTransform,
    apply_transform,
    axis_angle_rotation,
    best_fit_transform,
    iterative_closest_point,
)


def test_axis_angle_rotation_is_proper_rotation() -> None:
    rotation = axis_angle_rotation(np.array([1.0, 2.0, 3.0]), 17.0)

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_best_fit_transform_recovers_known_alignment() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(100, 3))
    known = RigidTransform(
        axis_angle_rotation(np.array([0.3, -0.2, 1.0]), 12.0),
        np.array([0.4, -0.7, 0.2]),
    )
    target = apply_transform(source, known)

    estimated = best_fit_transform(source, target)

    np.testing.assert_allclose(estimated.rotation, known.rotation, atol=1e-12)
    np.testing.assert_allclose(
        estimated.translation,
        known.translation,
        atol=1e-12,
    )


def test_icp_recovers_small_rigid_transform() -> None:
    rng = np.random.default_rng(8)
    target = rng.normal(size=(500, 3)) * np.array([2.0, 1.0, 0.5])
    source = apply_transform(
        target,
        RigidTransform(
            axis_angle_rotation(np.array([0.3, -0.2, 1.0]), 2.0),
            np.array([0.03, -0.02, 0.01]),
        ),
    )

    result = iterative_closest_point(
        source,
        target,
        max_iterations=60,
        tolerance=1e-10,
    )

    assert result.converged
    assert result.final_rmse < 1e-8
    np.testing.assert_allclose(result.aligned_points, target, atol=1e-8)


def test_trimmed_icp_uses_requested_correspondence_fraction() -> None:
    rng = np.random.default_rng(11)
    points = rng.normal(size=(100, 3))

    result = iterative_closest_point(
        points,
        points,
        correspondence_fraction=0.7,
    )

    assert result.correspondences_used == 70
    assert result.final_rmse < 1e-12


@pytest.mark.parametrize(
    ("max_iterations", "tolerance"),
    [(0, 1e-6), (10, 0.0)],
)
def test_icp_rejects_invalid_settings(
    max_iterations: int,
    tolerance: float,
) -> None:
    points = np.arange(30, dtype=np.float64).reshape(10, 3)

    with pytest.raises(ValueError):
        iterative_closest_point(
            points,
            points,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )


@pytest.mark.parametrize("correspondence_fraction", [0.0, 1.01, float("nan")])
def test_icp_rejects_invalid_correspondence_fraction(
    correspondence_fraction: float,
) -> None:
    points = np.arange(30, dtype=np.float64).reshape(10, 3)

    with pytest.raises(ValueError, match="Correspondence fraction"):
        iterative_closest_point(
            points,
            points,
            correspondence_fraction=correspondence_fraction,
        )
