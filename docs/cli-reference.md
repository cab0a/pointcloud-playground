# CLI and Output Reference

This document lists the complete command set and output contract. Start with
the minimal workflow in the [README](../README.md), then use the commands below
to reproduce or extend a specific experiment.

## Commands

Summarize all committed reference results:

```bash
pointcloud-playground summarize-results results \
  --output-dir output/summary
```

Generate a repeatable synthetic surface with uneven point density:

```bash
pointcloud-playground generate-demo demo.xyz --points 6000 --seed 42
```

Evaluate joint sensitivity to partial overlap and controlled source outliers:

```bash
pointcloud-playground evaluate-joint-sensitivity demo.xyz \
  --overlap-ratios 1.0 0.8 0.6 0.4 \
  --outlier-fractions 0.0 0.02 0.05 0.10 \
  --trim-fractions 0.7 0.4 \
  --angle 2 \
  --translation-scale 0.5 \
  --distance-scale 0.25 \
  --seed 42 \
  --max-iterations 80 \
  --tolerance-scale 1e-6 \
  --output-dir output/joint_sensitivity
```

Evaluate trim-fraction sensitivity with known correspondence diagnostics:

```bash
pointcloud-playground evaluate-trim-sensitivity demo.xyz \
  --overlap-ratios 1.0 0.8 0.6 0.4 \
  --trim-fractions 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --angle 2 \
  --translation-scale 0.5 \
  --max-iterations 80 \
  --tolerance-scale 1e-6 \
  --output-dir output/trim_sensitivity
```

Compare all-pairs and trimmed ICP across controlled scan overlap:

```bash
pointcloud-playground evaluate-partial-overlap demo.xyz \
  --overlap-ratios 1.0 0.8 0.6 0.4 \
  --angle 2 \
  --translation-scale 0.5 \
  --trim-fraction 0.7 \
  --max-iterations 80 \
  --tolerance-scale 1e-6 \
  --output-dir output/partial_overlap_registration
```

Evaluate ICP recovery across controlled rigid transforms:

```bash
pointcloud-playground evaluate-registration demo.xyz \
  --angles 2 5 10 20 \
  --translation-scales 0.5 1 2 4 \
  --max-iterations 60 \
  --tolerance-scale 1e-6 \
  --output-dir output/registration
```

Evaluate PCA normals under a controlled coordinate perturbation:

```bash
pointcloud-playground evaluate-normals demo.xyz \
  --neighbors 8 16 32 64 \
  --noise-scale 0.05 \
  --seed 42 \
  --output-dir output/normals
```

Inject 5% labeled outliers and evaluate a sweep of statistical thresholds:

```bash
pointcloud-playground evaluate-outliers demo.xyz \
  --outlier-fraction 0.05 \
  --distance-scale 0.25 \
  --neighbors 16 \
  --std-ratios 0.5 1.0 1.5 2.0 2.5 3.0 \
  --seed 42 \
  --output-dir output/outliers
```

Run the voxel-downsampling experiment:

```bash
pointcloud-playground evaluate-downsampling demo.xyz \
  --voxel-sizes 0.25 0.5 1.0 \
  --output-dir output/voxel_downsampling
```

Evaluate the included USGS-derived sample at scale-appropriate voxel sizes:

```bash
pointcloud-playground evaluate-downsampling data/usgs_3dep_iowa/sample.xyz \
  --voxel-sizes 5 10 20 \
  --output-dir output/voxel_downsampling_usgs
```

Regenerate all committed metrics and figures without changing `results/`:

```bash
python experiments/run_reference_experiments.py \
  --output-root reproduced_results
```

Regenerate the complete suite in a temporary directory and compare it with the
committed evidence:

```bash
python experiments/verify_reference_results.py
```

Rebuild the public sample from the checksum-pinned source LAZ file:

```bash
python -m pip install -e ".[data]"
python experiments/prepare_public_sample.py
```

### CLI and output contract

All seven evaluation commands take one positional XYZ input, accept
`--output-dir`, and write `metrics.csv` plus `comparison.png`. Method-specific
point clouds, labels, or point-level estimates are additional outputs.

| Experiment | Canonical command | Default output directory |
| --- | --- | --- |
| Voxel downsampling | `evaluate-downsampling` | `output/voxel_downsampling` |
| Outlier filtering | `evaluate-outliers` | `output/outlier_filtering` |
| Normal estimation | `evaluate-normals` | `output/normal_estimation` |
| Rigid registration | `evaluate-registration` | `output/registration` |
| Partial-overlap registration | `evaluate-partial-overlap` | `output/partial_overlap_registration` |
| Trim sensitivity | `evaluate-trim-sensitivity` | `output/trim_sensitivity` |
| Joint sensitivity | `evaluate-joint-sensitivity` | `output/joint_sensitivity` |

The earlier `evaluate` command remains available as an alias for
`evaluate-downsampling`.

### Python API

The stable top-level API covers point-cloud validation and XYZ I/O,
deterministic synthetic data, voxel downsampling, PCA normal estimation,
controlled outliers, and rigid registration.

```python
from pointcloud_playground import load_xyz, voxel_downsample

points = load_xyz("data/synthetic_controlled_density.xyz")
reduced = voxel_downsample(points, voxel_size=0.5)
```

The full public-name list, data contract, examples, units, errors, and 1.x
compatibility policy are documented in
[`docs/api.md`](api.md).
