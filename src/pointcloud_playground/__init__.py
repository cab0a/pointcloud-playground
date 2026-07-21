"""Stable public API for reproducible point-cloud experiments."""

from .downsampling import voxel_downsample
from .io import PointCloud, load_xyz, save_xyz, validate_points
from .normals import NormalEstimate, estimate_normals
from .outliers import (
    ContaminatedPointCloud,
    inject_vertical_outliers,
    mean_knn_distances,
    statistical_outlier_mask,
)
from .registration import (
    ICPResult,
    RigidTransform,
    apply_transform,
    axis_angle_rotation,
    best_fit_transform,
    identity_transform,
    iterative_closest_point,
    nearest_neighbor_rmse,
)
from .synthetic import (
    controlled_surface_normals,
    generate_controlled_density_cloud,
)

__version__ = "1.0.0"

__all__ = [
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
]
