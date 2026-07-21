import importlib.metadata

import pointcloud_playground


def test_public_api_exports_are_available() -> None:
    expected = {
        "ContaminatedPointCloud",
        "ICPResult",
        "NormalEstimate",
        "PointCloud",
        "RigidTransform",
        "apply_transform",
        "axis_angle_rotation",
        "best_fit_transform",
        "controlled_surface_normals",
        "estimate_normals",
        "generate_controlled_density_cloud",
        "identity_transform",
        "inject_vertical_outliers",
        "iterative_closest_point",
        "load_xyz",
        "mean_knn_distances",
        "nearest_neighbor_rmse",
        "save_xyz",
        "statistical_outlier_mask",
        "validate_points",
    }

    assert set(pointcloud_playground.__all__) == expected
    assert all(hasattr(pointcloud_playground, name) for name in expected)


def test_runtime_version_matches_package_metadata() -> None:
    assert pointcloud_playground.__version__ == importlib.metadata.version(
        "pointcloud-playground"
    )


def test_stable_release_metadata() -> None:
    metadata = importlib.metadata.metadata("pointcloud-playground")
    classifiers = set(metadata.get_all("Classifier") or [])
    project_urls = {
        label: url
        for label, url in (
            item.split(", ", maxsplit=1)
            for item in metadata.get_all("Project-URL") or []
        )
    }

    assert "Development Status :: 5 - Production/Stable" in classifiers
    assert project_urls["Repository"] == (
        "https://github.com/cab0a/pointcloud-playground"
    )
    assert project_urls["Changelog"].endswith("/CHANGELOG.md")
