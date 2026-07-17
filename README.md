# Point Cloud Playground

Reproducible point-cloud experiments that connect method selection,
implementation, quantitative evaluation, and documented interpretation.

Version 0.1.0 studies centroid-based voxel downsampling on a controlled
synthetic surface and a traceable public USGS 3DEP lidar sample.

## Research Question

How does voxel size change point count, point spacing, and geometric coverage
when a point cloud is reduced?

The working hypothesis is that larger voxels reduce local density and storage
requirements at the cost of increasing the distance between the original
points and their nearest retained representation.

## Features

- Deterministic controlled-density point-cloud generation
- Whitespace-delimited XYZ loading and writing
- Centroid-based voxel downsampling implemented with NumPy
- Nearest-neighbor spacing and coverage metrics using SciPy
- CSV metrics, downsampled XYZ files, and static comparison plots
- A versioned public-data sample with source checksum and preparation metadata
- CLI workflow and focused unit tests

## Quick Start

Use Python 3.10 or later in an isolated environment.

```bash
git clone https://github.com/cab0a/pointcloud-playground.git
cd pointcloud-playground
python -m pip install -e .

pointcloud-playground generate-demo demo.xyz
pointcloud-playground evaluate demo.xyz \
  --voxel-sizes 0.25 0.5 1.0 \
  --output-dir output/demo
```

The evaluation command writes:

```text
output/demo/
├── comparison.png
├── downsampled_0p25.xyz
├── downsampled_0p5.xyz
├── downsampled_1.xyz
└── metrics.csv
```

## Usage

Generate a repeatable synthetic surface with uneven point density:

```bash
pointcloud-playground generate-demo demo.xyz --points 6000 --seed 42
```

Evaluate the included USGS-derived sample at scale-appropriate voxel sizes:

```bash
pointcloud-playground evaluate data/usgs_3dep_iowa/sample.xyz \
  --voxel-sizes 5 10 20 \
  --output-dir output/usgs
```

Reproduce the committed metrics and figures:

```bash
python experiments/run_reference_experiments.py
```

Rebuild the public sample from the checksum-pinned source LAZ file:

```bash
python -m pip install -e ".[data]"
python experiments/prepare_public_sample.py
```

## Methodology

### Voxel downsampling

The coordinate minimum defines the voxel grid origin. Each point is assigned
to a voxel using the floor of its offset divided by the voxel size. Every
occupied voxel is represented by the centroid of its points.

### Metrics

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Retention ratio | Output points / input points | Fraction of points remaining |
| Mean nearest-neighbor distance | Mean distance to each point's nearest other point | Typical point spacing |
| Coverage RMSE | RMSE of each input point's distance to its nearest output point | Geometric coverage lost during reduction |

All coordinates and distances use the units of the input XYZ file.

## Evaluation

### Controlled-density synthetic surface

The fixed synthetic input contains 6,000 points over a 20-by-20 unit wavy
surface. One third of the samples are concentrated near its center.

| Voxel size | Output points | Retained | Output mean NN distance | Coverage RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 3,545 | 59.1% | 0.203 | 0.065 |
| 0.50 | 1,628 | 27.1% | 0.352 | 0.168 |
| 1.00 | 434 | 7.2% | 0.813 | 0.382 |

![Synthetic voxel-downsampling comparison](results/synthetic/comparison.png)

The results show the expected trade-off: increasing voxel size regularizes the
dense center and sharply reduces the point count, while both point spacing and
coverage error increase. This controlled case makes the density change easy to
observe without treating any one voxel size as universally preferable.

### Public USGS 3DEP sample

The public-data input contains 5,000 ground-classified points selected
deterministically from a USGS 3DEP Iowa lidar tile. Coordinates are translated
to a local origin without scaling. Source URL, checksums, counts, filter, seed,
and coordinate offsets are recorded in
[`manifest.csv`](data/usgs_3dep_iowa/manifest.csv).

| Voxel size | Output points | Retained | Output mean NN distance | Coverage RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 4,590 | 91.8% | 5.832 | 0.643 |
| 10 | 3,272 | 65.4% | 7.790 | 2.429 |
| 20 | 1,262 | 25.2% | 15.399 | 6.896 |

![USGS 3DEP voxel-downsampling comparison](results/usgs_3dep_iowa/comparison.png)

The 5-unit setting changes this sparse subset only modestly. At 10 and 20
units, point reduction becomes substantial and coverage error rises
accordingly. Selecting an operating point therefore requires a downstream
task tolerance, not only a target point count.

## Project Structure

```text
pointcloud-playground/
├── data/                         # Versioned synthetic and public samples
├── experiments/                  # Sample preparation and reference runs
├── results/                      # Reproducible metrics and figures
├── src/pointcloud_playground/    # Library and CLI implementation
├── tests/                        # Unit and CLI tests
├── LICENSE
├── README.md
└── pyproject.toml
```

## Limitations

- XYZ input stores coordinates only; attributes such as intensity,
  classification, color, and coordinate reference systems are not preserved.
- The implementation loads the complete point cloud into memory and is not
  intended for large-scale production processing.
- Coverage RMSE is one-way and task-agnostic. It does not measure surface
  reconstruction quality or downstream model performance.
- The public sample is a small, ground-only subset rather than a complete lidar
  scene, so its results must not be generalized to all 3D data.
- Appropriate voxel sizes depend on coordinate units, sensor density,
  geometry, and the intended use.
- Comparison figures are XY projections colored by height; they are not full
  3D viewers.

## Roadmap

- **v0.2:** Controlled outlier-removal experiments
- **v0.3:** Normal-estimation reliability and neighborhood selection
- **v0.4:** Rigid registration with measurable perturbations
- **v0.5:** Cross-experiment summaries and interface review

Each extension will keep the same pattern: define a question, control the
input, implement the method, evaluate the result, and document limitations.

## License

The source code is available under the [MIT License](LICENSE).

The included USGS 3DEP-derived sample is public domain. Its source and
preparation details are documented in
[`data/usgs_3dep_iowa/README.md`](data/usgs_3dep_iowa/README.md).
