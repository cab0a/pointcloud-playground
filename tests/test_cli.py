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
