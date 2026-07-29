# Python API

## 日本語概要

本書は、点群の入出力、決定論的なデータ生成、主要処理、剛体位置合わせに関する1.xの公開Python APIを定義します。安定性を保証する名前・引数・戻り値と、保証対象外の実験用モジュールを区別しています。

公開範囲、使用例、例外、単位の詳細は以下の英語本文を参照してください。

---

## English Summary

This reference defines the stable 1.x top-level Python API for point-cloud I/O,
deterministic data generation, core processing, and rigid registration. It
also separates supported imports from experiment-specific research workflows.

## Scope

Version 1.0 defines a stable top-level API for point-cloud I/O, deterministic
data generation, core processing methods, and rigid registration. These names
are exported through `pointcloud_playground.__all__` and can be imported
directly from `pointcloud_playground`.

The evaluation and reporting modules remain reproducible research workflows.
Their CLI commands and CSV outputs are documented interfaces, but direct
imports from those modules are outside the stable top-level Python API.

## Stability policy

The following interfaces are stable throughout the 1.x series:

- names exported through `pointcloud_playground.__all__`, including their
  documented parameters, return types, and error behavior;
- existing CLI command names, options, and their meanings;
- the primary `metrics.csv` and `comparison.png` output filenames for each
  evaluation command;
- existing CSV column names and metric definitions.

Minor releases may add optional parameters, commands, exports, or CSV columns
without changing existing behavior. Removing or renaming a stable interface,
changing a documented default, or changing the meaning of an existing metric
requires a new major version.

Private names, visualization layout details, internal implementation modules,
and direct imports from experiment-specific evaluation modules are not covered
by the 1.x compatibility guarantee. Scientific conclusions also remain limited
to the documented datasets and controlled protocols; API stability is not a
claim of universal method validity.

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

Use the CLI for the stable output contract. Direct module imports are
appropriate for adapting an experiment, but are not part of the top-level 1.x
compatibility guarantee.

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
