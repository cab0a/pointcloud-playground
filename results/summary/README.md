# Cross-Experiment Summary

This file condenses the committed reference results into one review view.
Selected conditions follow explicit rules for scanning and are not universal
recommendations. Metrics from different methods answer different questions
and must not be combined into a single ranking.

## Evidence Snapshot

| Experiment | Dataset | Conditions | Selected condition | Primary evidence | Secondary evidence |
| --- | --- | ---: | --- | --- | --- |
| `voxel_downsampling` | `synthetic` | 3 | `voxel_size=0.25` | `retention_ratio` = 59.1% | `coverage_rmse_over_input_spacing` = 0.509 |
| `voxel_downsampling` | `usgs_3dep_iowa` | 3 | `voxel_size=10` | `retention_ratio` = 65.4% | `coverage_rmse_over_input_spacing` = 0.461 |
| `outlier_filtering` | `synthetic` | 6 | `std_ratio=1.5` | `f1` = 93.4% | `inlier_retention` = 99.6% |
| `outlier_filtering` | `usgs_3dep_iowa` | 6 | `std_ratio=2` | `f1` = 91.4% | `inlier_retention` = 99.6% |
| `normal_estimation` | `synthetic` | 4 | `neighbors=64` | `mean_angular_error_deg` = 0.846° | `median_neighborhood_radius` = 1.296 |
| `normal_estimation` | `usgs_3dep_iowa` | 4 | `neighbors=64` | `median_repeatability_error_deg` = 0.091° | `median_neighborhood_radius` = 45.631 |
| `registration` | `synthetic` | 4 | `recovered=3/4` | `recovery_rate` = 75.0% | `largest_recovered_angle_deg` = 10.000° |
| `registration` | `usgs_3dep_iowa` | 4 | `recovered=2/4` | `recovery_rate` = 50.0% | `largest_recovered_angle_deg` = 5.000° |
| `partial_overlap_registration` | `synthetic` | 8 | `trim_fraction=0.7` | `trimmed_recovery_rate` = 50.0% | `all_pairs_recovery_rate` = 25.0% |
| `partial_overlap_registration` | `usgs_3dep_iowa` | 8 | `trim_fraction=0.7` | `trimmed_recovery_rate` = 50.0% | `all_pairs_recovery_rate` = 25.0% |

## Selection Rules and Evidence Scope

- `voxel_downsampling/synthetic`: Retention ratio closest to 0.5; representative only, not an optimum. Evidence scope: Self-coverage against the original cloud. Source: `voxel_downsampling/synthetic/metrics.csv`.
- `voxel_downsampling/usgs_3dep_iowa`: Retention ratio closest to 0.5; representative only, not an optimum. Evidence scope: Self-coverage against the original cloud. Source: `voxel_downsampling/usgs_3dep_iowa/metrics.csv`.
- `outlier_filtering/synthetic`: Highest F1 against controlled injected labels. Evidence scope: Ground-truth labels for injected outliers. Source: `outlier_filtering/synthetic/metrics.csv`.
- `outlier_filtering/usgs_3dep_iowa`: Highest F1 against controlled injected labels. Evidence scope: Ground-truth labels for injected outliers. Source: `outlier_filtering/usgs_3dep_iowa/metrics.csv`.
- `normal_estimation/synthetic`: Lowest mean error against analytic reference normals. Evidence scope: Analytic normal ground truth. Source: `normal_estimation/synthetic/metrics.csv`.
- `normal_estimation/usgs_3dep_iowa`: Lowest perturbation repeatability error; stability only, not accuracy. Evidence scope: Controlled perturbation without normal ground truth. Source: `normal_estimation/usgs_3dep_iowa/metrics.csv`.
- `registration/synthetic`: Converged with known-pair RMSE no greater than 0.01 times spacing. Evidence scope: Known rigid transform and one-to-one correspondences. Source: `registration/synthetic/metrics.csv`.
- `registration/usgs_3dep_iowa`: Converged with known-pair RMSE no greater than 0.01 times spacing. Evidence scope: Known rigid transform and one-to-one correspondences. Source: `registration/usgs_3dep_iowa/metrics.csv`.
- `partial_overlap_registration/synthetic`: Compare a fixed trimmed fraction with all-pairs ICP across the same overlap sweep. Evidence scope: Known transform and known correspondences in the overlap. Source: `partial_overlap_registration/synthetic/metrics.csv`.
- `partial_overlap_registration/usgs_3dep_iowa`: Compare a fixed trimmed fraction with all-pairs ICP across the same overlap sweep. Evidence scope: Known transform and known correspondences in the overlap. Source: `partial_overlap_registration/usgs_3dep_iowa/metrics.csv`.

## Interface Review

All evaluation commands use a positional XYZ input, accept `--output-dir`,
and create `metrics.csv` plus `comparison.png`. Additional files contain
method-specific point-level or point-cloud outputs.

| Experiment | Canonical command | Default output directory |
| --- | --- | --- |
| Voxel downsampling | `evaluate-downsampling` | `output/voxel_downsampling` |
| Outlier filtering | `evaluate-outliers` | `output/outlier_filtering` |
| Normal estimation | `evaluate-normals` | `output/normal_estimation` |
| Rigid registration | `evaluate-registration` | `output/registration` |
| Partial-overlap registration | `evaluate-partial-overlap` | `output/partial_overlap_registration` |

The earlier `evaluate` command remains as an alias for
`evaluate-downsampling`. Reference outputs use the canonical layout
`results/<experiment>/<dataset>/`.
