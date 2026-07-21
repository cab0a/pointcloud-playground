from pathlib import Path

from pointcloud_playground.cli import main


def test_cli_generates_demo_and_evaluation_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.xyz"
    output_dir = tmp_path / "evaluation"

    assert main(["generate-demo", str(input_path), "--points", "200"]) == 0
    assert (
        main(
            [
                "evaluate",
                str(input_path),
                "--voxel-sizes",
                "0.5",
                "1.0",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "comparison.png").is_file()
    assert (output_dir / "downsampled_0p5.xyz").is_file()
    assert (output_dir / "downsampled_1.xyz").is_file()


def test_cli_generates_outlier_filtering_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.xyz"
    output_dir = tmp_path / "outliers"
    assert main(["generate-demo", str(input_path), "--points", "300"]) == 0

    result = main(
        [
            "evaluate-outliers",
            str(input_path),
            "--neighbors",
            "8",
            "--std-ratios",
            "1.0",
            "2.0",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert (output_dir / "contaminated.xyz").is_file()
    assert (output_dir / "labels.csv").is_file()
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "comparison.png").is_file()
    assert (output_dir / "filtered_std_1.xyz").is_file()


def test_cli_generates_normal_evaluation_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.xyz"
    output_dir = tmp_path / "normals"
    assert main(["generate-demo", str(input_path), "--points", "300"]) == 0

    result = main(
        [
            "evaluate-normals",
            str(input_path),
            "--neighbors",
            "8",
            "16",
            "--noise-scale",
            "0.05",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "comparison.png").is_file()
    assert (output_dir / "normals_k8.csv").is_file()
    assert (output_dir / "normals_k16.csv").is_file()


def test_cli_generates_registration_evaluation_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.xyz"
    output_dir = tmp_path / "registration"
    assert main(["generate-demo", str(input_path), "--points", "300"]) == 0

    result = main(
        [
            "evaluate-registration",
            str(input_path),
            "--angles",
            "2",
            "5",
            "--translation-scales",
            "0.5",
            "1.0",
            "--max-iterations",
            "40",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "comparison.png").is_file()
    assert (output_dir / "case_01_source.xyz").is_file()
    assert (output_dir / "case_01_aligned.xyz").is_file()
