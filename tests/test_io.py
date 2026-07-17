from pathlib import Path

import numpy as np
import pytest

from pointcloud_playground.io import load_xyz, save_xyz


def test_xyz_round_trip(tmp_path: Path) -> None:
    points = np.array([[0.0, 1.0, 2.0], [3.5, 4.5, 5.5]])

    path = save_xyz(tmp_path / "nested" / "cloud.xyz", points)

    np.testing.assert_allclose(load_xyz(path), points)


def test_load_xyz_rejects_non_finite_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "invalid.xyz"
    path.write_text("0 1 nan\n", encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        load_xyz(path)
