"""Prepare the versioned USGS 3DEP sample from its public LAZ source."""

import argparse
import csv
import hashlib
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

import laspy
import numpy as np

from pointcloud_playground.io import save_xyz

SOURCE_URL = (
    "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/"
    "IA_FullState/ept-data/10-437-477-512.laz"
)
SOURCE_SHA256 = "2b7c3e9317973a426bbe325def55dc74afa09dc33e13f87db7038c2125dd2517"


def download_source(path: Path) -> None:
    """Download the source LAZ file with a descriptive user agent."""
    request = Request(
        SOURCE_URL,
        headers={"User-Agent": "pointcloud-playground/0.1"},
    )
    with urlopen(request) as response, path.open("wb") as output:
        output.write(response.read())


def prepare_sample(output_dir: Path, sample_size: int, seed: int) -> None:
    """Filter, sample, translate, and document the public point cloud."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        source_path = Path(temporary_directory) / "source.laz"
        download_source(source_path)
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_hash != SOURCE_SHA256:
            raise RuntimeError(
                "Source checksum does not match the versioned sample definition."
            )

        lidar = laspy.read(source_path)
        xyz = np.column_stack((lidar.x, lidar.y, lidar.z))
        ground = xyz[np.asarray(lidar.classification) == 2]
        ground = ground[np.isfinite(ground).all(axis=1)]
        if sample_size > len(ground):
            raise ValueError("Sample size exceeds the available ground points.")

        rng = np.random.default_rng(seed)
        selected_indices = np.sort(
            rng.choice(len(ground), size=sample_size, replace=False)
        )
        offset = ground.min(axis=0)
        local_points = ground[selected_indices] - offset

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = save_xyz(output_dir / "sample.xyz", local_points)
    sample_hash = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    manifest_path = output_dir / "manifest.csv"
    rows = [
        ("source_url", SOURCE_URL),
        ("source_sha256", SOURCE_SHA256),
        ("source_points", str(len(xyz))),
        ("filter", "ASPRS classification 2 (ground)"),
        ("filtered_points", str(len(ground))),
        ("sample_method", "random without replacement"),
        ("sample_size", str(sample_size)),
        ("random_seed", str(seed)),
        ("offset_x", f"{offset[0]:.6f}"),
        ("offset_y", f"{offset[1]:.6f}"),
        ("offset_z", f"{offset[2]:.6f}"),
        ("sample_sha256", sample_hash),
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("field", "value"))
        writer.writerows(rows)


def main() -> None:
    """Run the sample preparation command."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/usgs_3dep_iowa"),
    )
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_sample(args.output_dir, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
