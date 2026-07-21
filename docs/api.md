# Python API

## Scope

Version 0.9 defines a small top-level API for point-cloud I/O, deterministic
data generation, core processing methods, and rigid registration. These names
are exported through `pointcloud_playground.__all__` and can be imported
directly from `pointcloud_playground`.

The evaluation and reporting modules remain reproducible research workflows.
Their CLI commands and CSV outputs are documented, but their complete Python
call signatures are not yet declared stable. Compatibility will be reviewed
again for v1.0.

## Data contract

A point cloud is a NumPy `float64` array with shape `(n, 3)`. Public functions
validate that the array is non-empty and contains only finite coordinates.
XYZ files are whitespace-delimited, may contain comment lines beginning with
`#`, and do not preserve color, intensity, classification, or coordinate
reference-system metadata.

```python
from pointcloud_playground import load_xyz, save_xyz, voxel_downsample

points = load_xyz("data/synthetic_controlled_density.xyz")
reduced = voxel_downsample(points, voxel_size=0.5)
save_xyz("output/reduced.xyz", reduced)
```

## Public names

| Area | Public names |
| --- | --- |
| Types and validation | `PointCloud`, `validate_points` |
| XYZ I/O | `load_xyz`, `save_xyz` |
| Synthetic reference data | `generate_controlled_density_cloud`, `controlled_surface_normals` |
| Downsampling | `voxel_downsample` |
| Normal estimation | `NormalEstimate`, `estimate_normals` |
| Controlled outliers | `ContaminatedPointCloud`, `inject_vertical_outliers`, `mean_knn_distances`, `statistical_outlier_mask` |
| Rigid registration | `RigidTransform`, `ICPResult`, `identity_transform`, `apply_transform`, `axis_angle_rotation`, `best_fit_transform`, `nearest_neighbor_rmse`, `iterative_closest_point` |

## Registration example

`iterative_closest_point` estimates a transform from `source` to `target`.
Setting `correspondence_fraction` below one retains the closest fraction of
source-to-target nearest-neighbor pairs during each iteration.

```python
from pointcloud_playground import iterative_closest_point, load_xyz

target = load_xyz("target.xyz")
source = load_xyz("source.xyz")
result = iterative_closest_point(
    source,
    target,
    max_iterations=80,
    tolerance=1e-6,
    correspondence_fraction=0.7,
)

print(result.converged)
print(result.final_rmse)
print(result.transform.rotation)
print(result.transform.translation)
```

The convergence flag only indicates that the numerical stopping rule was met.
It does not establish that the recovered transform is correct.

## Evaluation workflows

The following modules contain the experiment-specific dataclasses, evaluators,
CSV writers, and diagnostics used by the CLI and reference runner.

| Module | Workflow |
| --- | --- |
| `evaluation` | Voxel-downsampling evaluation |
| `filtering_evaluation` | Labeled outlier-filtering evaluation |
| `normal_evaluation` | Neighborhood-size normal evaluation |
| `registration_evaluation` | Known-transform ICP evaluation |
| `overlap_evaluation` | Controlled partial-overlap evaluation |
| `trim_evaluation` | Trim-fraction and correspondence diagnostics |
| `joint_evaluation` | Joint overlap-and-outlier sensitivity |
| `summary` | Cross-experiment evidence summary |

Use the CLI for the versioned output contract. Direct module imports are
appropriate for adapting an experiment, but should be reviewed when upgrading
before v1.0.

## Errors and units

- Invalid point arrays and unreadable XYZ inputs raise `ValueError`.
- Output-path failures may raise `OSError`.
- Distances use the input coordinate units unless explicitly normalized by
  median point spacing in an evaluation result.
- Angles accepted or reported by the public registration helpers use degrees
  where the name includes `angle_deg` or `degrees`.
- Randomized controlled-data functions accept an explicit seed and default to
  a deterministic value where documented.

The installed version is available through `pointcloud_playground.__version__`
and the command line:

```bash
pointcloud-playground --version
```
