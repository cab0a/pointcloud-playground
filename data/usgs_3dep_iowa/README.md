# USGS 3DEP Iowa Sample

## 日本語概要

このディレクトリには、公開されているUSGS 3DEPの点群タイルから決定論的に抽出した5,000点の標本があります。取得元、元データと標本のSHA-256、乱数種、抽出条件、座標変換を`manifest.csv`へ記録しています。

由来と再作成手順の詳細は以下の英語本文を参照してください。

---

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
