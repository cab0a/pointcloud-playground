"""Regenerate and compare the committed reference experiment artifacts."""

import argparse
import csv
import hashlib
import math
import struct
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pointcloud_playground import (
    generate_controlled_density_cloud,
    load_xyz,
)

try:
    from .run_reference_experiments import run_reference_experiments
except ImportError:  # Direct execution from the repository root.
    from run_reference_experiments import run_reference_experiments

ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COLUMN_ABSOLUTE_TOLERANCES = {
    "rotation_error_deg": 5e-6,
}


class VerificationError(RuntimeError):
    """Raised when regenerated artifacts differ from the reference set."""


@dataclass(frozen=True)
class VerificationSummary:
    """Counts of artifacts checked by a complete reproduction run."""

    input_files: int
    csv_reports: int
    markdown_reports: int
    figures: int


def _relative_files(root: Path, pattern: str) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob(pattern)}


def _numeric_value(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _values_match(
    reference: str,
    generated: str,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    reference_number = _numeric_value(reference)
    generated_number = _numeric_value(generated)
    if reference_number is None or generated_number is None:
        return reference == generated
    if math.isnan(reference_number) or math.isnan(generated_number):
        return math.isnan(reference_number) and math.isnan(generated_number)
    return math.isclose(
        reference_number,
        generated_number,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def _compare_csv_files(
    reference_path: Path,
    generated_path: Path,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    with reference_path.open(newline="", encoding="utf-8") as reference_file:
        reference_rows = list(csv.reader(reference_file))
    with generated_path.open(newline="", encoding="utf-8") as generated_file:
        generated_rows = list(csv.reader(generated_file))

    if len(reference_rows) != len(generated_rows):
        raise VerificationError(
            f"Row count differs for {reference_path.name}: "
            f"{len(reference_rows)} != {len(generated_rows)}"
        )
    for row_index, (reference_row, generated_row) in enumerate(
        zip(reference_rows, generated_rows, strict=True),
        start=1,
    ):
        if len(reference_row) != len(generated_row):
            raise VerificationError(
                f"Column count differs in {reference_path.name}, row {row_index}."
            )
        for column_index, (reference, generated) in enumerate(
            zip(reference_row, generated_row, strict=True),
            start=1,
        ):
            column_name = reference_rows[0][column_index - 1]
            effective_absolute_tolerance = max(
                absolute_tolerance,
                COLUMN_ABSOLUTE_TOLERANCES.get(column_name, 0.0),
            )
            if not _values_match(
                reference,
                generated,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=effective_absolute_tolerance,
            ):
                raise VerificationError(
                    f"Value differs in {reference_path.name}, row {row_index}, "
                    f"column {column_index} ({column_name}): "
                    f"{reference!r} != {generated!r}"
                )


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if len(data) != 24 or data[:8] != PNG_SIGNATURE:
        raise VerificationError(f"Not a valid PNG header: {path}")
    return struct.unpack(">II", data[16:24])


def _compare_file_inventory(
    reference_root: Path,
    generated_root: Path,
    pattern: str,
) -> set[Path]:
    reference_files = _relative_files(reference_root, pattern)
    generated_files = _relative_files(generated_root, pattern)
    if reference_files != generated_files:
        missing = sorted(reference_files - generated_files)
        unexpected = sorted(generated_files - reference_files)
        raise VerificationError(
            f"Artifact inventory differs for {pattern}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return reference_files


def _verify_synthetic_input() -> None:
    committed = load_xyz(ROOT / "data" / "synthetic_controlled_density.xyz")
    regenerated = generate_controlled_density_cloud()
    if committed.shape != regenerated.shape or not np.allclose(
        committed,
        regenerated,
        rtol=0.0,
        atol=5.1e-7,
    ):
        raise VerificationError(
            "The committed synthetic input differs from the deterministic generator."
        )


def _verify_public_input() -> None:
    data_dir = ROOT / "data" / "usgs_3dep_iowa"
    with (data_dir / "manifest.csv").open(newline="", encoding="utf-8") as file:
        manifest = {row["field"]: row["value"] for row in csv.DictReader(file)}
    expected = manifest.get("sample_sha256")
    if expected is None:
        raise VerificationError("The public-data manifest has no sample checksum.")
    digest = hashlib.sha256((data_dir / "sample.xyz").read_bytes()).hexdigest()
    if digest != expected:
        raise VerificationError(
            "The public sample does not match its recorded SHA-256 checksum."
        )


def verify_reference_results(
    reference_root: Path,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-12,
) -> VerificationSummary:
    """Regenerate results and compare numeric reports and artifact structure."""
    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("Comparison tolerances must be non-negative.")
    if not reference_root.is_dir():
        raise ValueError(f"Reference result directory does not exist: {reference_root}")

    _verify_synthetic_input()
    _verify_public_input()
    with tempfile.TemporaryDirectory(prefix="pointcloud-playground-") as temp_dir:
        generated_root = Path(temp_dir) / "results"
        run_reference_experiments(generated_root)

        csv_files = _compare_file_inventory(
            reference_root,
            generated_root,
            "*.csv",
        )
        for relative_path in sorted(csv_files):
            _compare_csv_files(
                reference_root / relative_path,
                generated_root / relative_path,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )

        markdown_files = {Path("summary/README.md")}
        for relative_path in markdown_files:
            reference_text = (reference_root / relative_path).read_text(
                encoding="utf-8"
            )
            generated_text = (generated_root / relative_path).read_text(
                encoding="utf-8"
            )
            if reference_text != generated_text:
                raise VerificationError(
                    f"Markdown summary differs: {relative_path}"
                )

        figure_files = _compare_file_inventory(
            reference_root,
            generated_root,
            "*.png",
        )
        for relative_path in sorted(figure_files):
            reference_dimensions = _png_dimensions(reference_root / relative_path)
            generated_dimensions = _png_dimensions(generated_root / relative_path)
            if reference_dimensions != generated_dimensions:
                raise VerificationError(
                    f"Figure dimensions differ for {relative_path}: "
                    f"{reference_dimensions} != {generated_dimensions}"
                )

    return VerificationSummary(
        input_files=2,
        csv_reports=len(csv_files),
        markdown_reports=len(markdown_files),
        figures=len(figure_files),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the reference-verification argument parser."""
    parser = argparse.ArgumentParser(
        description="Regenerate and verify the committed reference results.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=ROOT / "results",
        help="Committed reference-result root to verify.",
    )
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run complete reference-result verification."""
    args = build_parser().parse_args(argv)
    try:
        summary = verify_reference_results(
            args.reference_root,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
        )
    except (OSError, ValueError, VerificationError) as exc:
        print(f"Reference verification failed: {exc}")
        return 1

    print(f"Input files: {summary.input_files} verified")
    print(f"CSV reports: {summary.csv_reports} verified")
    print(f"Markdown reports: {summary.markdown_reports} verified")
    print(f"Figures: {summary.figures} structurally verified")
    print("Reference reproduction: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
