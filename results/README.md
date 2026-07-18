# Reference Results

The results in this directory are generated from versioned inputs and fixed
parameters. Reproduce all experiments with:

```bash
python experiments/run_reference_experiments.py
```

The voxel-downsampling directories contain:

- `metrics.csv`: point retention, spacing, and coverage measurements
- `comparison.png`: input and downsampled XY views colored by height

The `outlier_filtering` directories contain:

- `metrics.csv`: detection and clean-geometry preservation measurements for
  every statistical threshold
- `comparison.png`: ground truth, predicted removals, and the best-F1 output

The `normal_estimation` directories contain:

- `metrics.csv`: accuracy, perturbation repeatability, support radius, and
  surface-variation measurements across neighborhood sizes
- `comparison.png`: sampled normal vectors and neighborhood-selection curves

Interpretation is documented in the main project README.
