# Reference Results

The results in this directory are generated from versioned inputs and fixed
parameters. Regenerate all experiments into a separate directory with:

```bash
python experiments/run_reference_experiments.py \
  --output-root reproduced_results
```

Regenerate in a temporary directory and compare against this reference set:

```bash
python experiments/verify_reference_results.py
```

Reference outputs follow `results/<experiment>/<dataset>/`, where the dataset
is either `synthetic` or `usgs_3dep_iowa`.

The `summary` directory contains:

- `experiment_summary.csv`: fourteen records in a shared review schema
- `README.md`: selection rules, evidence scope, and the CLI interface review
- `comparison.png`: a non-ranking evidence matrix across methods and datasets

The `voxel_downsampling` directories contain:

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

The `registration` directories contain:

- `metrics.csv`: known-transform recovery, correspondence error, iteration,
  and convergence measurements across initial offsets
- `comparison.png`: hardest-case overlays and scale-normalized recovery curves

The `partial_overlap_registration` directories contain:

- `metrics.csv`: overlap, correspondence-selection, convergence, and
  known-transform recovery measurements for all-pairs and trimmed ICP
- `comparison.png`: lowest-overlap scan overlays and scale-normalized error
  curves across the overlap sweep

The `trim_sensitivity` directories contain:

- `metrics.csv`: recovery, transform error, retained-pair composition, exact
  match precision and recall, and residual diagnostics across the fixed grid
- `comparison.png`: recovery and error heatmaps with correspondence precision
  and recall curves

The `joint_sensitivity` directories contain:

- `metrics.csv`: recovery, transform error, retained-pair composition, exact
  match quality, and outlier rejection across the joint overlap-contamination
  grid
- `comparison.png`: recovery and exact-pair precision heatmaps for all-pairs,
  70% trimmed, and 40% trimmed ICP

Interpretation is documented in the main project README. The numeric and
structural comparison policy is documented in
[`docs/reproducibility.md`](../docs/reproducibility.md).
