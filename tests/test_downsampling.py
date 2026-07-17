import numpy as np
import pytest

from pointcloud_playground.downsampling import voxel_downsample


def test_voxel_downsample_uses_centroids() -> None:
    points = np.array(
        [
            [0.1, 0.1, 0.0],
            [0.3, 0.3, 0.2],
            [1.2, 1.2, 1.0],
        ]
    )

    sampled = voxel_downsample(points, voxel_size=1.0)

    assert sampled.shape == (2, 3)
    np.testing.assert_allclose(sampled[0], [0.2, 0.2, 0.1])
    np.testing.assert_allclose(sampled[1], [1.2, 1.2, 1.0])


@pytest.mark.parametrize("voxel_size", [0.0, -1.0, float("nan")])
def test_voxel_downsample_rejects_invalid_size(voxel_size: float) -> None:
    with pytest.raises(ValueError, match="Voxel size"):
        voxel_downsample(np.zeros((2, 3)), voxel_size)
