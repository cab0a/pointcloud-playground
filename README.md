# Point Cloud Playground

Reproducible point-cloud experiments that connect method selection,
implementation, quantitative evaluation, and documented interpretation.

Version 0.5.0 adds a cross-experiment evidence summary and reviews the public
CLI and result layout across voxel downsampling, outlier filtering, normal
estimation, and rigid registration. Every experiment uses a deterministic
synthetic surface and a traceable public USGS 3DEP lidar sample.

## Research Questions

### Cross-experiment review

Can results from different point-cloud methods be made easier to inspect
without collapsing incompatible metrics into a single score or implied
ranking?

The working hypothesis is that a common summary schema, explicit condition
selection rules, and a consistent result layout can improve reviewability
while preserving the evidence scope and limitations of each experiment.

### Rigid registration

How does the magnitude of a known initial rotation and translation affect
point-to-point ICP recovery within a fixed iteration budget?

The working hypothesis is that small misalignments will recover the known
source-to-target transform, while larger misalignments will require more
iterations and eventually exceed the fixed budget. Nearest-neighbor RMSE is
also expected to understate alignment error when ICP settles on incorrect
correspondences.

### Normal estimation

How does neighborhood size affect the accuracy, perturbation stability, and
spatial support of PCA-estimated point-cloud normals?

The working hypothesis is that larger neighborhoods will reduce sensitivity
to coordinate noise, while also increasing the spatial scale over which local
geometry is approximated. The most stable setting is therefore not
automatically the most appropriate setting for preserving local detail.

### Outlier filtering

How does the distance threshold of a statistical outlier filter affect
detection quality and preservation of valid geometry?

The working hypothesis is that a lower threshold will detect more injected
outliers but remove more valid points, while a higher threshold will preserve
more valid points at the cost of missed outliers. The best operating point is
expected to depend on the point-cloud geometry and density.

### Voxel downsampling

How does voxel size change point count, point spacing, and geometric coverage
when a point cloud is reduced?

The working hypothesis is that larger voxels reduce local density and storage
requirements at the cost of increasing the distance between the original
points and their nearest retained representation.

## Features

- Cross-experiment CSV, Markdown, and visual evidence summaries
- Explicit representative-condition rules without universal recommendations
- Canonical `results/<experiment>/<dataset>/` reference layout
- Consistent evaluation commands, default output directories, and filenames
- Backward-compatible `evaluate` alias for `evaluate-downsampling`
- Point-to-point ICP implemented with SciPy nearest-neighbor search and NumPy
  rigid least-squares alignment
- Controlled axis-angle rotations and spacing-relative translations
- Rotation, translation, known-correspondence, and nearest-neighbor errors
- Fixed-budget convergence diagnostics across increasing initial offsets
- Batched local PCA normal estimation across configurable neighborhood sizes
- Analytic normal ground truth for the controlled synthetic surface
- Sign-invariant angular accuracy and perturbation-repeatability metrics
- Neighborhood-radius and surface-variation diagnostics
- Deterministic injection of labeled vertical outliers
- Mean k-nearest-neighbor statistical filtering using SciPy
- Precision, recall, F1, inlier-retention, and coverage measurements
- Deterministic controlled-density point-cloud generation
- Centroid-based voxel downsampling implemented with NumPy
- Whitespace-delimited XYZ loading and writing
- CSV metrics, filtered XYZ files, and static comparison plots
- A versioned public-data sample with source checksum and preparation metadata
- CLI workflows and focused unit tests

## Quick Start

Use Python 3.10 or later in an isolated environment.

```bash
git clone https://github.com/cab0a/pointcloud-playground.git
cd pointcloud-playground
python -m pip install -e .

pointcloud-playground generate-demo demo.xyz
pointcloud-playground summarize-results results \
  --output-dir output/summary
```

The cross-experiment review writes:

```text
output/summary/
├── comparison.png
├── experiment_summary.csv
└── README.md
```

## Usage

Summarize all committed reference results:

```bash
pointcloud-playground summarize-results results \
  --output-dir output/summary
```

Generate a repeatable synthetic surface with uneven point density:

```bash
pointcloud-playground generate-demo demo.xyz --points 6000 --seed 42
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

Reproduce all committed metrics and figures:

```bash
python experiments/run_reference_experiments.py
```

Rebuild the public sample from the checksum-pinned source LAZ file:

```bash
python -m pip install -e ".[data]"
python experiments/prepare_public_sample.py
```

### CLI and output contract

All four evaluation commands take one positional XYZ input, accept
`--output-dir`, and write `metrics.csv` plus `comparison.png`. Method-specific
point clouds, labels, or point-level estimates are additional outputs.

| Experiment | Canonical command | Default output directory |
| --- | --- | --- |
| Voxel downsampling | `evaluate-downsampling` | `output/voxel_downsampling` |
| Outlier filtering | `evaluate-outliers` | `output/outlier_filtering` |
| Normal estimation | `evaluate-normals` | `output/normal_estimation` |
| Rigid registration | `evaluate-registration` | `output/registration` |

The earlier `evaluate` command remains available as an alias for
`evaluate-downsampling`.

## Methodology

### Cross-experiment summary

The summary reads the eight committed `metrics.csv` files from four
experiments and two datasets. Each row uses the same schema: experiment,
dataset, number of evaluated conditions, selected condition, selection rule,
primary evidence, secondary evidence, evidence scope, and source path.

One representative condition is selected per experiment and dataset:

| Experiment | Selection rule |
| --- | --- |
| Voxel downsampling | Retention ratio closest to 50%, as a review point rather than an optimum |
| Outlier filtering | Highest F1 against controlled injected labels |
| Normal estimation, synthetic | Lowest mean error against analytic reference normals |
| Normal estimation, public | Lowest median perturbation error, reported as stability rather than accuracy |
| Rigid registration | Fraction converged with known-pair RMSE no greater than 0.01 times median spacing |

The summary deliberately does not create a combined score. F1, angular error,
coverage, and transform recovery describe different questions and cannot be
ranked on a shared quality axis.

### Controlled rigid registration

The input cloud is the registration target. A source cloud is created by
rotating the target around its centroid about the fixed axis `(0.3, -0.2, 1)`
and translating it in the fixed direction `(0.7, -0.4, 0.2)`. Direction
vectors are normalized before use. Translation magnitudes are defined relative
to each input's median nearest-neighbor spacing:

| Case | Initial rotation | Translation magnitude |
| --- | ---: | ---: |
| `case_01` | 2° | 0.5 × median spacing |
| `case_02` | 5° | 1 × median spacing |
| `case_03` | 10° | 2 × median spacing |
| `case_04` | 20° | 4 × median spacing |

Point-to-point ICP starts from the identity transform. Each iteration finds
one nearest target point for every current source point, estimates the rigid
least-squares transform with SVD, applies it, and composes it with the running
estimate. The maximum is 60 iterations, and convergence is declared when the
nearest-neighbor RMSE changes by no more than `1e-6 × median spacing`.

Because every source point was generated from a target point, the inverse
rotation and translation are known. This permits direct transform error and
known-correspondence RMSE measurements in addition to the objective used by
ICP. Numerical convergence means only that the stopping condition was met; it
does not prove correct registration.

### PCA normal estimation

For each point, the `k` nearest other points define a local neighborhood. The
neighborhood is centered, its 3-by-3 covariance matrix is computed, and the
eigenvector associated with the smallest eigenvalue is used as the local
normal. Normal signs are oriented toward positive Z for consistent terrain
visualization.

The synthetic surface has analytic reference normals derived from its
noise-free height function. Accuracy is measured by the sign-invariant angle
between the estimate and reference:

```text
angular error = arccos(|estimated normal · reference normal|)
```

For both datasets, deterministic isotropic Gaussian noise is added with a
standard deviation of 5% of the input's median nearest-neighbor spacing.
Normals are re-estimated after perturbation, and their angular change measures
repeatability. The same perturbed cloud is used for every neighborhood size.

Surface variation is reported as:

```text
smallest covariance eigenvalue / sum of covariance eigenvalues
```

It describes local non-planarity under the selected support and is not a
standalone accuracy score.

### Controlled outlier injection

The clean point cloud is treated as reference geometry. Synthetic outliers are
sampled uniformly within its XY footprint and placed above or below its Z
range. Their offset is a fixed multiple of a geometry-derived scale:

```text
vertical scale = max(Z range, 0.05 × XY diagonal)
outlier offset = vertical scale × distance scale × U(1, 2)
```

The requested outlier fraction describes the final contaminated cloud, subject
to integer rounding. A fixed random seed makes point generation and ordering
repeatable. Ground-truth labels are retained for evaluation.

### Statistical outlier filtering

For every point, the mean Euclidean distance to its 16 nearest neighbors is
computed. A point is classified as an outlier when its score exceeds:

```text
global mean score + standard-deviation ratio × global score standard deviation
```

The reference experiment holds the neighborhood size constant and sweeps the
standard-deviation ratio from 0.5 to 3.0. The highest-F1 result is selected for
the comparison figure; all threshold results remain in `metrics.csv`.

### Voxel downsampling

The coordinate minimum defines the voxel-grid origin. Each point is assigned
to a voxel using the floor of its offset divided by the voxel size. Every
occupied voxel is represented by the centroid of its points.

### Metrics

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Rotation error | Angle of the residual rotation between estimated and known transforms | Orientation recovery |
| Translation error | Euclidean distance between estimated and known translation vectors | Position recovery in input units |
| Known-pair RMSE | RMSE between aligned source points and their generating target points | Ground-truth alignment error |
| Nearest-neighbor RMSE | RMSE from aligned source points to their nearest target points | ICP objective, which can accept incorrect pairs |
| Normalized error | Error divided by median point spacing | Scale-relative comparison between datasets |
| Reference angular error | Sign-invariant angle to an analytic reference normal | Accuracy on the controlled synthetic surface |
| Repeatability error | Angle between normals before and after controlled perturbation | Sensitivity to small coordinate changes |
| Neighborhood radius | Distance to the kth nearest point | Spatial support of the local estimate |
| Surface variation | Smallest covariance eigenvalue / eigenvalue sum | Local non-planarity at the selected support |
| Precision | Correct removals / all removals | Reliability of an outlier decision |
| Recall | Correct removals / all true outliers | Fraction of injected outliers detected |
| F1 | Harmonic mean of precision and recall | Balanced detection score |
| Inlier retention | Retained valid points / all valid points | Preservation of clean geometry |
| Clean coverage RMSE | RMSE from each clean point to the nearest filtered point | Geometric impact of false removals |
| Retention ratio | Output points / input points | Fraction remaining after downsampling |
| Mean nearest-neighbor distance | Mean distance to each point's nearest other point | Typical point spacing |
| Coverage RMSE | RMSE from each input point to the nearest output point | Geometric coverage lost during reduction |

All coordinates and distances use the units of the input XYZ file.

## Evaluation

### Cross-experiment evidence snapshot

The v0.5 review contains eight summary records. The table shows the selected
review condition and its primary evidence; each row retains its own selection
rule and evidence scope in
[`results/summary/README.md`](results/summary/README.md).

| Experiment | Synthetic surface | Public USGS 3DEP sample |
| --- | --- | --- |
| Voxel downsampling | Voxel 0.25: 59.1% retained | Voxel 10: 65.4% retained |
| Outlier filtering | Ratio 1.5: F1 0.934 | Ratio 2.0: F1 0.914 |
| Normal estimation | k=64: mean reference error 0.846° | k=64: median repeatability error 0.091° |
| Rigid registration | 3/4 recovered; largest angle 10° | 2/4 recovered; largest angle 5° |

![Cross-experiment evidence snapshot](results/summary/comparison.png)

The summary makes three important boundaries visible. The outlier threshold
selected by F1 changes between datasets. The public normal result describes
repeatability rather than accuracy because no reference normals exist. The
registration recovery rate is lower for the public sample under the same
spacing-relative offsets and iteration budget. These observations remain
method-specific evidence, not an overall dataset or algorithm ranking.

### Rigid registration: synthetic surface

The synthetic cloud has a median point spacing of 0.107 units. ICP exactly
recovers the first three controlled transforms within the 60-iteration budget.
The 20-degree case reaches the budget with residual transform error.

| Rotation / translation | Iterations | Converged in budget | Rotation error | Translation error / spacing | Known-pair RMSE / spacing | NN RMSE / spacing |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| 2° / 0.5× | 8 | Yes | <0.001° | <0.001 | <0.001 | <0.001 |
| 5° / 1× | 28 | Yes | <0.001° | <0.001 | <0.001 | <0.001 |
| 10° / 2× | 48 | Yes | <0.001° | <0.001 | <0.001 | <0.001 |
| 20° / 4× | 60 | No | 1.892° | 0.585 | 2.183 | 1.318 |

![Rigid-registration evaluation on the synthetic surface](results/registration/synthetic/comparison.png)

Required iterations increase sharply with the initial offset. In the final
case, nearest-neighbor RMSE is lower than known-pair RMSE because some source
points are close to the wrong target points. The fixed budget therefore
exposes both initialization sensitivity and the limits of evaluating ICP only
with its own correspondence objective.

### Rigid registration: public USGS 3DEP sample

The public sample has a median point spacing of 4.929 input units. With the
same relative transforms and iteration budget, the first two cases recover
exactly, while the 10- and 20-degree cases remain incorrect at iteration 60.

| Rotation / translation | Iterations | Converged in budget | Rotation error | Translation error / spacing | Known-pair RMSE / spacing | NN RMSE / spacing |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| 2° / 0.5× | 10 | Yes | <0.001° | <0.001 | <0.001 | <0.001 |
| 5° / 1× | 51 | Yes | <0.001° | <0.001 | <0.001 | <0.001 |
| 10° / 2× | 60 | No | 3.372° | 6.621 | 3.528 | 1.163 |
| 20° / 4× | 60 | No | 5.350° | 10.337 | 5.557 | 1.287 |

![Rigid-registration evaluation on the USGS 3DEP sample](results/registration/usgs_3dep_iowa/comparison.png)

The public terrain sample requires more iterations than the synthetic surface
under the same spacing-relative offsets. Its failed cases also show a larger
gap between known-pair and nearest-neighbor error. This result does not define
a universal ICP capture range; it demonstrates that convergence behavior
depends on geometry, sampling pattern, stopping criteria, and iteration budget.

### Normal estimation: synthetic surface

The input contains 6,000 unevenly sampled points with 0.02-unit Z noise. Its
analytic reference normals come from the underlying noise-free height
function. The controlled perturbation standard deviation is 0.00535 units.

| Neighbors | Median radius | Mean reference error | P95 reference error | Median repeatability error | P95 repeatability error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.407 | 3.815° | 9.882° | 0.880° | 4.675° |
| 16 | 0.605 | 1.744° | 4.347° | 0.413° | 1.829° |
| 32 | 0.892 | 0.982° | **2.302°** | 0.208° | 0.792° |
| 64 | 1.296 | **0.846°** | 2.417° | **0.111°** | **0.380°** |

![Normal-estimation evaluation on the synthetic surface](results/normal_estimation/synthetic/comparison.png)

Increasing the neighborhood size substantially improves mean accuracy and
perturbation repeatability. However, the P95 reference error is lowest at
`k=32` and rises slightly at `k=64`, while the median support radius grows to
1.296 units. This tail behavior is consistent with the expected trade-off:
broader support suppresses noise but can mix geometry across a curved,
unevenly sampled surface.

### Normal estimation: public USGS 3DEP sample

The public sample contains 5,000 ground-classified points. It has no
independent normal labels, so this experiment reports perturbation
repeatability and neighborhood diagnostics without claiming reference
accuracy. The controlled perturbation standard deviation is 0.246 input
units.

| Neighbors | Median radius | Mean repeatability error | Median repeatability error | P95 repeatability error | Mean surface variation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 15.788 | 0.955° | 0.830° | 2.121° | 4.50e-5 |
| 16 | 22.576 | 0.423° | 0.389° | 0.859° | 3.86e-5 |
| 32 | 32.155 | 0.203° | 0.186° | 0.402° | 2.83e-5 |
| 64 | 45.631 | 0.099° | 0.091° | 0.202° | 2.14e-5 |

![Normal-estimation evaluation on the USGS 3DEP sample](results/normal_estimation/usgs_3dep_iowa/comparison.png)

Repeatability improves monotonically with neighborhood size, but the median
support radius nearly triples. Without ground truth, the lower angular change
at `k=64` cannot establish higher accuracy: it may also reflect smoothing over
local terrain structure. Neighborhood selection therefore requires a spatial
detail requirement in addition to a stability target.

### Controlled outliers: synthetic surface

The clean input contains 6,000 points over a 20-by-20 unit wavy surface, with
one third of the samples concentrated near its center. The experiment adds 316
outliers, giving a final outlier fraction of 5%.

| Std. ratio | Precision | Recall | F1 | Inliers retained | Coverage RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.263 | 1.000 | 0.417 | 85.3% | 0.187 |
| 1.0 | 0.697 | 0.984 | 0.816 | 97.8% | 0.068 |
| **1.5** | **0.923** | **0.946** | **0.934** | **99.6%** | **0.030** |
| 2.0 | 0.986 | 0.867 | 0.923 | 99.9% | 0.013 |
| 2.5 | 0.996 | 0.744 | 0.851 | 100.0% | 0.010 |
| 3.0 | 1.000 | 0.579 | 0.733 | 100.0% | 0.000 |

![Controlled outlier filtering on the synthetic surface](results/outlier_filtering/synthetic/comparison.png)

The low threshold detects every injected outlier but also removes 884 valid
points. Increasing the ratio reduces false removals, while recall begins to
fall. A ratio of 1.5 gives the highest F1 in this controlled case; this is an
experimental result, not a general default.

### Controlled outliers: public USGS 3DEP sample

The public-data input contains 5,000 ground-classified points selected
deterministically from a USGS 3DEP Iowa lidar tile. The experiment adds 263
outliers with the same fraction, relative distance scale, neighborhood size,
threshold sweep, and seed as the synthetic experiment.

| Std. ratio | Precision | Recall | F1 | Inliers retained | Coverage RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.321 | 1.000 | 0.486 | 88.9% | 4.332 |
| 1.0 | 0.639 | 0.996 | 0.779 | 97.0% | 2.243 |
| 1.5 | 0.822 | 0.947 | 0.880 | 98.9% | 1.310 |
| **2.0** | **0.916** | **0.913** | **0.914** | **99.6%** | **0.926** |
| 2.5 | 0.963 | 0.791 | 0.868 | 99.8% | 0.620 |
| 3.0 | 0.976 | 0.624 | 0.761 | 99.9% | 0.365 |

![Controlled outlier filtering on the USGS 3DEP sample](results/outlier_filtering/usgs_3dep_iowa/comparison.png)

The same precision-recall trade-off appears, but the highest-F1 ratio changes
from 1.5 to 2.0. This difference supports the original hypothesis: even under
controlled contamination, threshold selection depends on the source geometry
and sampling pattern. A production decision should also reflect whether
missed noise or removed surface detail is more costly.

### Voxel downsampling: synthetic surface

| Voxel size | Output points | Retained | Output mean NN distance | Coverage RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 3,545 | 59.1% | 0.203 | 0.065 |
| 0.50 | 1,628 | 27.1% | 0.352 | 0.168 |
| 1.00 | 434 | 7.2% | 0.813 | 0.382 |

![Synthetic voxel-downsampling comparison](results/voxel_downsampling/synthetic/comparison.png)

Increasing voxel size regularizes the dense center and sharply reduces the
point count, while point spacing and coverage error both increase.

### Voxel downsampling: public USGS 3DEP sample

Source URL, checksums, counts, filter, seed, and coordinate offsets are
recorded in
[`manifest.csv`](data/usgs_3dep_iowa/manifest.csv).

| Voxel size | Output points | Retained | Output mean NN distance | Coverage RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 4,590 | 91.8% | 5.832 | 0.643 |
| 10 | 3,272 | 65.4% | 7.790 | 2.429 |
| 20 | 1,262 | 25.2% | 15.399 | 6.896 |

![USGS 3DEP voxel-downsampling comparison](results/voxel_downsampling/usgs_3dep_iowa/comparison.png)

The 5-unit setting changes this sparse subset only modestly. At 10 and 20
units, point reduction becomes substantial and coverage error rises.

## Project Structure

```text
pointcloud-playground/
├── data/                         # Versioned synthetic and public samples
├── experiments/                  # Sample preparation and reference runs
├── results/
│   ├── summary/                  # v0.5 cross-experiment review
│   ├── registration/             # v0.4 metrics and figures
│   ├── normal_estimation/        # v0.3 metrics and figures
│   ├── outlier_filtering/        # v0.2 metrics and figures
│   └── voxel_downsampling/        # v0.1 metrics and figures
├── src/pointcloud_playground/
│   ├── cli.py
│   ├── downsampling.py
│   ├── evaluation.py
│   ├── filtering_evaluation.py
│   ├── io.py
│   ├── normal_evaluation.py
│   ├── normals.py
│   ├── outliers.py
│   ├── registration.py
│   ├── registration_evaluation.py
│   ├── summary.py
│   ├── synthetic.py
│   └── visualization.py
├── tests/
├── LICENSE
├── README.md
└── pyproject.toml
```

## Limitations

- Summary selections are deterministic review points, not recommended
  production parameters. A task-specific cost function may select a different
  condition.
- Primary metrics are intentionally method-specific. They cannot support a
  combined score, cross-method ranking, or claim that one dataset is easier in
  general.
- The two datasets share an experiment protocol but not the same coordinate
  scale, sampling pattern, geometry, or ground-truth coverage.
- Registration uses complete, one-to-one transformed copies with full overlap.
  It does not model partial overlap, outliers, missing regions, or changing
  sampling density.
- The ICP implementation uses all nearest-neighbor pairs without robust
  correspondence rejection, multiscale initialization, features, or a
  point-to-plane objective.
- Exact recovery in this controlled setup must not be generalized to sensor
  scans or unrelated point clouds.
- The convergence flag reports the numerical stopping rule within the fixed
  budget; transform error is the correctness measure for this experiment.
- Translation-vector error depends on the coordinate origin. Normalized
  known-pair RMSE is the more directly comparable geometric measure here.
- Local PCA assumes that a neighborhood is adequately approximated by a plane.
  Curvature, boundaries, mixed surfaces, and non-uniform density can bias the
  estimate.
- Positive-Z orientation is suitable for these height-field and terrain
  experiments, but not for vertical, enclosed, or arbitrarily oriented
  surfaces that require a separate orientation strategy.
- The public sample has no reference normals. Perturbation repeatability is a
  stability measurement and must not be interpreted as accuracy.
- The controlled perturbation is isotropic Gaussian noise and does not model
  a specific lidar sensor or acquisition geometry.
- Surface variation depends on neighborhood scale and sampling distribution;
  lower values do not universally mean better normals.
- Injected noise is limited to isolated points above or below the reference
  cloud. It does not model clustered noise, multipath returns, or near-surface
  sensor artifacts.
- The source cloud is assumed to be clean. Ground-truth labels apply only to
  injected points and do not validate the original USGS classifications.
- The filter uses one global distance threshold and is sensitive to varying
  density, neighborhood size, geometry, and coordinate scale.
- F1 assigns equal importance to precision and recall. An application may
  require a different cost function and validation dataset.
- Selecting the highest-F1 threshold uses the injected ground-truth labels and
  is an evaluation procedure, not an unsupervised production-tuning method.
- XYZ input stores coordinates only; intensity, classification, color, and
  coordinate reference systems are not preserved.
- The implementation loads the complete cloud into memory and is not intended
  for large-scale production processing.
- The public sample is a small, ground-only subset rather than a complete lidar
  scene, so its results must not be generalized to all 3D data.
- Comparison figures are diagnostic projections, not full 3D viewers.

## Roadmap

- **v0.6:** Controlled partial-overlap registration and correspondence review

Each extension will keep the same pattern: define a question, control the
input, implement the method, evaluate the result, and document limitations.

## License

The source code is available under the [MIT License](LICENSE).

The included USGS 3DEP-derived sample is public domain. Its source and
preparation details are documented in
[`data/usgs_3dep_iowa/README.md`](data/usgs_3dep_iowa/README.md).
