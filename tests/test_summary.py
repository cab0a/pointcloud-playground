from pathlib import Path
import csv

import pytest

from pointcloud_playground.summary import (
    collect_experiment_summaries,
    write_experiment_summary_csv,
    write_experiment_summary_markdown,
)
from pointcloud_playground.visualization import save_experiment_summary_plot


ROOT = Path(__file__).resolve().parents[1]


def test_collect_experiment_summaries_selects_review_conditions() -> None:
    summaries = collect_experiment_summaries(ROOT / "results")
    by_key = {
        (summary.experiment, summary.dataset): summary
        for summary in summaries
    }

    assert len(summaries) == 10
    assert (
        by_key[("voxel_downsampling", "synthetic")].selected_condition
        == "voxel_size=0.25"
    )
    assert (
        by_key[("outlier_filtering", "usgs_3dep_iowa")].selected_condition
        == "std_ratio=2"
    )
    assert (
        by_key[("normal_estimation", "synthetic")].selected_condition
        == "neighbors=64"
    )
    assert (
        by_key[("normal_estimation", "usgs_3dep_iowa")].primary_metric
        == "median_repeatability_error_deg"
    )
    assert (
        by_key[("registration", "synthetic")].primary_value
        == pytest.approx(0.75)
    )
    assert (
        by_key[("registration", "usgs_3dep_iowa")].secondary_value
        == pytest.approx(5.0)
    )
    partial_synthetic = by_key[("partial_overlap_registration", "synthetic")]
    assert partial_synthetic.selected_condition == "trim_fraction=0.7"
    assert partial_synthetic.primary_value == pytest.approx(0.5)
    assert partial_synthetic.secondary_value == pytest.approx(0.25)


def test_summary_outputs_share_the_same_records(tmp_path: Path) -> None:
    summaries = collect_experiment_summaries(ROOT / "results")
    csv_path = write_experiment_summary_csv(
        tmp_path / "experiment_summary.csv",
        summaries,
    )
    markdown_path = write_experiment_summary_markdown(
        tmp_path / "README.md",
        summaries,
    )
    plot_path = save_experiment_summary_plot(
        tmp_path / "comparison.png",
        summaries,
    )

    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(summaries)
    assert rows[0]["experiment"] == "voxel_downsampling"
    assert "Interface Review" in markdown_path.read_text(encoding="utf-8")
    assert plot_path.stat().st_size > 0


def test_summary_requires_the_canonical_result_tree(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Metrics file does not exist"):
        collect_experiment_summaries(tmp_path)
