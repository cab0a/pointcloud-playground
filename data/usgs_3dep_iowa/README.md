# USGS 3DEP Iowa Sample

This directory contains a deterministic 5,000-point subset derived from a
public USGS 3D Elevation Program (3DEP) lidar tile.

## Provenance

- Collection: `IA_FullState`
- Source format: Entwine Point Tile LAZ
- Source node: `10-437-477-512`
- Source URL: <https://s3-us-west-2.amazonaws.com/usgs-lidar-public/IA_FullState/ept-data/10-437-477-512.laz>
- Product information: <https://www.usgs.gov/3d-elevation-program/about-3dep-products-services>

USGS states that 3DEP products are available free of charge and without use
restrictions. The source data and this derived sample are public domain.

## Preparation

The preparation script verifies the downloaded LAZ checksum, keeps ASPRS
classification 2 (ground), selects 5,000 points without replacement using
random seed 42, and translates the coordinates to a local origin. No rotation,
scaling, or noise is applied.

```bash
python -m pip install -e ".[data]"
python experiments/prepare_public_sample.py
```

Exact source and sample checksums, counts, seed, and coordinate offsets are
recorded in `manifest.csv`.
