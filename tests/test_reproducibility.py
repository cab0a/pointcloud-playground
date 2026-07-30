from pathlib import Path

import pytest

from experiments.verify_reference_results import (
    VerificationError,
    _compare_csv_files,
    _png_dimensions,
    _verify_public_input,
    _verify_synthetic_input,
)


def test_committed_reference_inputs_pass_integrity_checks() -> None:
    _verify_synthetic_input()
    _verify_public_input()


def test_csv_comparison_allows_small_numeric_differences(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    generated = tmp_path / "generated.csv"
    reference.write_text("case,value\na,1.000000000\n", encoding="utf-8")
    generated.write_text("case,value\na,1.000000001\n", encoding="utf-8")

    _compare_csv_files(
        reference,
        generated,
        relative_tolerance=1e-8,
        absolute_tolerance=0.0,
    )


def test_csv_comparison_rejects_material_differences(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    generated = tmp_path / "generated.csv"
    reference.write_text("case,value\na,1.0\n", encoding="utf-8")
    generated.write_text("case,value\na,1.1\n", encoding="utf-8")

    with pytest.raises(VerificationError, match="Value differs"):
        _compare_csv_files(
            reference,
            generated,
            relative_tolerance=1e-8,
            absolute_tolerance=0.0,
        )


def test_csv_comparison_allows_near_zero_rotation_error_differences(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.csv"
    generated = tmp_path / "generated.csv"
    reference.write_text(
        "case,rotation_error_deg\na,3.818191820832846e-06\n",
        encoding="utf-8",
    )
    generated.write_text(
        "case,rotation_error_deg\na,0.0\n",
        encoding="utf-8",
    )

    _compare_csv_files(
        reference,
        generated,
        relative_tolerance=1e-9,
        absolute_tolerance=1e-12,
    )

    generated.write_text(
        "case,rotation_error_deg\na,1e-4\n",
        encoding="utf-8",
    )
    with pytest.raises(VerificationError, match="rotation_error_deg"):
        _compare_csv_files(
            reference,
            generated,
            relative_tolerance=1e-9,
            absolute_tolerance=1e-12,
        )


def test_csv_comparison_allows_near_zero_translation_error_differences(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.csv"
    generated = tmp_path / "generated.csv"
    reference.write_text(
        "case,translation_error\na,2.4442826638183156e-12\n",
        encoding="utf-8",
    )
    generated.write_text(
        "case,translation_error\na,4.545564912517566e-13\n",
        encoding="utf-8",
    )

    _compare_csv_files(
        reference,
        generated,
        relative_tolerance=1e-9,
        absolute_tolerance=1e-12,
    )

    generated.write_text(
        "case,translation_error\na,1e-8\n",
        encoding="utf-8",
    )
    with pytest.raises(VerificationError, match="translation_error"):
        _compare_csv_files(
            reference,
            generated,
            relative_tolerance=1e-9,
            absolute_tolerance=1e-12,
        )


def test_committed_comparison_figure_has_valid_png_dimensions() -> None:
    width, height = _png_dimensions(
        Path("results/joint_sensitivity/synthetic/comparison.png")
    )

    assert width > 0
    assert height > 0
