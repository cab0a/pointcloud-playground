from pathlib import Path

from pointcloud_playground.cli import build_parser, main


def test_cli_uses_consistent_default_output_directories() -> None:
    parser = build_parser()
    commands = {
        "evaluate-downsampling": Path("output/voxel_downsampling"),
        "evaluate-outliers": Path("output/outlier_filtering"),
        "evaluate-normals": Path("output/normal_estimation"),
        "evaluate-registration": Path("output/registration"),
        "evaluate-partial-overlap": Path(
            "output/partial_overlap_registration"
        ),
        "evaluate-trim-sensitivity": Path("output/trim_sensitivity"),
        "evaluate-joint-sensitivity": Path("output/joint_sensitivity"),
    }

    for command, expected in commands.items():
        args = parser.parse_args([command, "input.xyz"])
        assert args.output_dir == expected


def test_cli_generates_demo_and_evaluation_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.xyz"
    output_dir = tmp_path / "evaluation"

    assert main(["generate-demo", str(input_path), "--points", "200"]) == 0
    assert (
        main(
            [
                "evaluate-downsampling",
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


def test_cli_generates_partial_overlap_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.xyz"
    output_dir = tmp_path / "partial_overlap"
    assert main(["generate-demo", str(input_path), "--points", "300"]) == 0

    result = main(
        [
            "evaluate-partial-overlap",
            str(input_path),
            "--overlap-ratios",
            "1.0",
            "0.6",
            "--trim-fraction",
            "0.7",
            "--max-iterations",
            "40",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "comparison.png").is_file()
    assert (output_dir / "case_01_target.xyz").is_file()
    assert (output_dir / "case_01_source.xyz").is_file()
    assert (output_dir / "case_01_all_pairs_aligned.xyz").is_file()
    assert (output_dir / "case_01_trimmed_aligned.xyz").is_file()


def test_cli_generates_trim_sensitivity_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.xyz"
    output_dir = tmp_path / "trim_sensitivity"
    assert main(["generate-demo", str(input_path), "--points", "300"]) == 0

    result = main(
        [
            "evaluate-trim-sensitivity",
            str(input_path),
            "--overlap-ratios",
            "0.8",
            "--trim-fractions",
            "0.4",
            "0.8",
            "--max-iterations",
            "40",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "comparison.png").is_file()


def test_cli_generates_joint_sensitivity_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.xyz"
    output_dir = tmp_path / "joint_sensitivity"
    assert main(["generate-demo", str(input_path), "--points", "300"]) == 0

    result = main(
        [
            "evaluate-joint-sensitivity",
            str(input_path),
            "--overlap-ratios",
            "0.8",
            "--outlier-fractions",
            "0.0",
            "0.05",
            "--trim-fractions",
            "0.7",
            "0.4",
            "--max-iterations",
            "40",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "comparison.png").is_file()


def test_cli_keeps_the_legacy_downsampling_alias(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.xyz"
    output_dir = tmp_path / "legacy"
    assert main(["generate-demo", str(input_path), "--points", "100"]) == 0

    assert (
        main(
            [
                "evaluate",
                str(input_path),
                "--voxel-sizes",
                "1.0",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    assert (output_dir / "metrics.csv").is_file()


def test_cli_generates_cross_experiment_summary(tmp_path: Path) -> None:
    results_root = Path(__file__).resolve().parents[1] / "results"
    output_dir = tmp_path / "summary"

    result = main(
        [
            "summarize-results",
            str(results_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert (output_dir / "experiment_summary.csv").is_file()
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "comparison.png").is_file()
