"""Cross-experiment summaries for committed reference results."""

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import math


EXPERIMENTS = (
    "voxel_downsampling",
    "outlier_filtering",
    "normal_estimation",
    "registration",
    "partial_overlap_registration",
    "trim_sensitivity",
    "joint_sensitivity",
)
DATASETS = ("synthetic", "usgs_3dep_iowa")


@dataclass(frozen=True)
class ExperimentSummary:
    """One review-oriented summary of an experiment and dataset."""

    experiment: str
    dataset: str
    conditions_evaluated: int
    selected_condition: str
    selection_rule: str
    primary_metric: str
    primary_value: float
    secondary_metric: str
    secondary_value: float
    evidence_scope: str
    source_metrics: str

    def as_row(self) -> dict[str, str | int | float]:
        """Return a serializable row in the stable summary schema."""
        return asdict(self)


def _read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Metrics file does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Metrics file contains no rows: {path}")
    return rows


def _number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Metric '{field}' is missing or invalid.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Metric '{field}' must be finite.")
    return value


def _metrics_source(experiment: str, dataset: str) -> str:
    return f"{experiment}/{dataset}/metrics.csv"


def _summarize_downsampling(
    rows: list[dict[str, str]], dataset: str
) -> ExperimentSummary:
    selected = min(
        rows,
        key=lambda row: abs(_number(row, "retention_ratio") - 0.5),
    )
    input_spacing = _number(selected, "input_mean_nn_distance")
    if input_spacing <= 0:
        raise ValueError("Input mean nearest-neighbor distance must be positive.")
    return ExperimentSummary(
        experiment="voxel_downsampling",
        dataset=dataset,
        conditions_evaluated=len(rows),
        selected_condition=f"voxel_size={_number(selected, 'voxel_size'):g}",
        selection_rule=(
            "Retention ratio closest to 0.5; representative only, not an optimum"
        ),
        primary_metric="retention_ratio",
        primary_value=_number(selected, "retention_ratio"),
        secondary_metric="coverage_rmse_over_input_spacing",
        secondary_value=(
            _number(selected, "coverage_rmse") / input_spacing
        ),
        evidence_scope="Self-coverage against the original cloud",
        source_metrics=_metrics_source("voxel_downsampling", dataset),
    )


def _summarize_outlier_filtering(
    rows: list[dict[str, str]], dataset: str
) -> ExperimentSummary:
    selected = max(
        rows,
        key=lambda row: (
            _number(row, "f1"),
            _number(row, "inlier_retention"),
        ),
    )
    return ExperimentSummary(
        experiment="outlier_filtering",
        dataset=dataset,
        conditions_evaluated=len(rows),
        selected_condition=f"std_ratio={_number(selected, 'std_ratio'):g}",
        selection_rule="Highest F1 against controlled injected labels",
        primary_metric="f1",
        primary_value=_number(selected, "f1"),
        secondary_metric="inlier_retention",
        secondary_value=_number(selected, "inlier_retention"),
        evidence_scope="Ground-truth labels for injected outliers",
        source_metrics=_metrics_source("outlier_filtering", dataset),
    )


def _summarize_normal_estimation(
    rows: list[dict[str, str]], dataset: str
) -> ExperimentSummary:
    reference_values = [row.get("mean_angular_error_deg", "") for row in rows]
    has_reference = all(value not in (None, "") for value in reference_values)
    if has_reference:
        selected = min(rows, key=lambda row: _number(row, "mean_angular_error_deg"))
        primary_metric = "mean_angular_error_deg"
        selection_rule = "Lowest mean error against analytic reference normals"
        evidence_scope = "Analytic normal ground truth"
    else:
        selected = min(
            rows,
            key=lambda row: _number(row, "median_repeatability_error_deg"),
        )
        primary_metric = "median_repeatability_error_deg"
        selection_rule = (
            "Lowest perturbation repeatability error; stability only, not accuracy"
        )
        evidence_scope = "Controlled perturbation without normal ground truth"
    return ExperimentSummary(
        experiment="normal_estimation",
        dataset=dataset,
        conditions_evaluated=len(rows),
        selected_condition=f"neighbors={_number(selected, 'neighbors'):g}",
        selection_rule=selection_rule,
        primary_metric=primary_metric,
        primary_value=_number(selected, primary_metric),
        secondary_metric="median_neighborhood_radius",
        secondary_value=_number(selected, "median_neighborhood_radius"),
        evidence_scope=evidence_scope,
        source_metrics=_metrics_source("normal_estimation", dataset),
    )


def _summarize_registration(
    rows: list[dict[str, str]], dataset: str
) -> ExperimentSummary:
    recovered = [
        row
        for row in rows
        if row.get("converged", "").lower() == "true"
        and _number(row, "normalized_correspondence_rmse") <= 0.01
    ]
    largest_angle = max(
        (_number(row, "angle_deg") for row in recovered),
        default=0.0,
    )
    return ExperimentSummary(
        experiment="registration",
        dataset=dataset,
        conditions_evaluated=len(rows),
        selected_condition=f"recovered={len(recovered)}/{len(rows)}",
        selection_rule=(
            "Converged with known-pair RMSE no greater than 0.01 times spacing"
        ),
        primary_metric="recovery_rate",
        primary_value=len(recovered) / len(rows),
        secondary_metric="largest_recovered_angle_deg",
        secondary_value=largest_angle,
        evidence_scope="Known rigid transform and one-to-one correspondences",
        source_metrics=_metrics_source("registration", dataset),
    )


def _summarize_partial_overlap_registration(
    rows: list[dict[str, str]], dataset: str
) -> ExperimentSummary:
    all_pairs = [row for row in rows if row.get("method") == "all_pairs"]
    trimmed = [row for row in rows if row.get("method") == "trimmed"]
    if not all_pairs or len(all_pairs) != len(trimmed):
        raise ValueError(
            "Partial-overlap metrics require paired all-pairs and trimmed rows."
        )

    def recovery_rate(method_rows: list[dict[str, str]]) -> float:
        recovered = sum(
            row.get("recovered", "").lower() == "true"
            for row in method_rows
        )
        return recovered / len(method_rows)

    trim_fractions = {
        _number(row, "correspondence_fraction") for row in trimmed
    }
    if len(trim_fractions) != 1:
        raise ValueError("Trimmed rows must use one correspondence fraction.")
    trim_fraction = trim_fractions.pop()
    return ExperimentSummary(
        experiment="partial_overlap_registration",
        dataset=dataset,
        conditions_evaluated=len(rows),
        selected_condition=f"trim_fraction={trim_fraction:g}",
        selection_rule=(
            "Compare a fixed trimmed fraction with all-pairs ICP across the "
            "same overlap sweep"
        ),
        primary_metric="trimmed_recovery_rate",
        primary_value=recovery_rate(trimmed),
        secondary_metric="all_pairs_recovery_rate",
        secondary_value=recovery_rate(all_pairs),
        evidence_scope="Known transform and known correspondences in the overlap",
        source_metrics=_metrics_source(
            "partial_overlap_registration",
            dataset,
        ),
    )


def _summarize_trim_sensitivity(
    rows: list[dict[str, str]], dataset: str
) -> ExperimentSummary:
    by_fraction: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        fraction = _number(row, "trim_fraction")
        by_fraction.setdefault(fraction, []).append(row)
    overlap_counts = {len(fraction_rows) for fraction_rows in by_fraction.values()}
    if len(overlap_counts) != 1:
        raise ValueError("Every trim fraction must use the same overlap sweep.")

    def recovery_rate(fraction_rows: list[dict[str, str]]) -> float:
        return sum(
            row.get("recovered", "").lower() == "true"
            for row in fraction_rows
        ) / len(fraction_rows)

    selected_fraction, selected_rows = max(
        by_fraction.items(),
        key=lambda item: (recovery_rate(item[1]), item[0]),
    )
    mean_precision = sum(
        _number(row, "correct_match_precision") for row in selected_rows
    ) / len(selected_rows)
    return ExperimentSummary(
        experiment="trim_sensitivity",
        dataset=dataset,
        conditions_evaluated=len(rows),
        selected_condition=f"trim_fraction={selected_fraction:g}",
        selection_rule=(
            "Highest controlled recovery rate across the overlap sweep; "
            "ties retain more correspondences"
        ),
        primary_metric="recovery_rate",
        primary_value=recovery_rate(selected_rows),
        secondary_metric="mean_correct_match_precision",
        secondary_value=mean_precision,
        evidence_scope=(
            "Known transform, overlap membership, and exact overlap pairs"
        ),
        source_metrics=_metrics_source("trim_sensitivity", dataset),
    )


def _summarize_joint_sensitivity(
    rows: list[dict[str, str]], dataset: str
) -> ExperimentSummary:
    by_method: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("method", ""), _number(row, "correspondence_fraction"))
        by_method.setdefault(key, []).append(row)
    condition_counts = {len(method_rows) for method_rows in by_method.values()}
    if len(condition_counts) != 1:
        raise ValueError("Every joint method must use the same condition grid.")

    def recovery_rate(method_rows: list[dict[str, str]]) -> float:
        return sum(
            row.get("recovered", "").lower() == "true"
            for row in method_rows
        ) / len(method_rows)

    (selected_method, selected_fraction), selected_rows = max(
        by_method.items(),
        key=lambda item: (recovery_rate(item[1]), item[0][1]),
    )
    mean_precision = sum(
        _number(row, "correct_match_precision") for row in selected_rows
    ) / len(selected_rows)
    return ExperimentSummary(
        experiment="joint_sensitivity",
        dataset=dataset,
        conditions_evaluated=len(rows),
        selected_condition=(
            f"method={selected_method}, fraction={selected_fraction:g}"
        ),
        selection_rule=(
            "Highest controlled recovery rate across the joint grid; "
            "ties retain more correspondences"
        ),
        primary_metric="recovery_rate",
        primary_value=recovery_rate(selected_rows),
        secondary_metric="mean_correct_match_precision",
        secondary_value=mean_precision,
        evidence_scope=(
            "Known transform, overlap membership, exact pairs, and source-"
            "outlier labels"
        ),
        source_metrics=_metrics_source("joint_sensitivity", dataset),
    )


def collect_experiment_summaries(
    results_root: str | Path,
) -> list[ExperimentSummary]:
    """Read the canonical result tree and summarize all reference runs."""
    root = Path(results_root)
    summarizers = {
        "voxel_downsampling": _summarize_downsampling,
        "outlier_filtering": _summarize_outlier_filtering,
        "normal_estimation": _summarize_normal_estimation,
        "registration": _summarize_registration,
        "partial_overlap_registration": (
            _summarize_partial_overlap_registration
        ),
        "trim_sensitivity": _summarize_trim_sensitivity,
        "joint_sensitivity": _summarize_joint_sensitivity,
    }
    summaries: list[ExperimentSummary] = []
    for experiment in EXPERIMENTS:
        for dataset in DATASETS:
            metrics_path = root / experiment / dataset / "metrics.csv"
            rows = _read_metrics(metrics_path)
            summaries.append(summarizers[experiment](rows, dataset))
    return summaries


def write_experiment_summary_csv(
    path: str | Path, summaries: list[ExperimentSummary]
) -> Path:
    """Write summaries using a common cross-experiment CSV schema."""
    if not summaries:
        raise ValueError("At least one experiment summary is required.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0].as_row())
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary.as_row() for summary in summaries)
    return output_path


def _format_value(metric: str, value: float) -> str:
    percentage_metrics = {
        "retention_ratio",
        "f1",
        "inlier_retention",
        "recovery_rate",
    }
    if metric in percentage_metrics or metric.endswith("_recovery_rate"):
        return f"{value:.1%}"
    if metric.endswith("_precision") or metric.endswith("_recall"):
        return f"{value:.1%}"
    if metric.endswith("_deg"):
        return f"{value:.3f}°"
    return f"{value:.3f}"


def write_experiment_summary_markdown(
    path: str | Path, summaries: list[ExperimentSummary]
) -> Path:
    """Write a human-readable evidence and interface review."""
    if not summaries:
        raise ValueError("At least one experiment summary is required.")
    lines = [
        "# Cross-Experiment Summary",
        "",
        "## 日本語概要",
        "",
        (
            "本書は、7種類の点群実験と2種類のデータについて、代表条件、"
            "主要指標、補助指標、根拠となるCSVを一つの表へ集約します。"
            "選択規則は確認用の固定規則であり、異なる実験の指標を順位付けや"
            "総合点へ変換しません。"
        ),
        "",
        "選択規則と証拠範囲の詳細は以下の英語本文を参照してください。",
        "",
        "---",
        "",
        "This file condenses the committed reference results into one review view.",
        "Selected conditions follow explicit rules for scanning and are not universal",
        "recommendations. Metrics from different methods answer different questions",
        "and must not be combined into a single ranking.",
        "",
        "## Evidence Snapshot",
        "",
        (
            "| Experiment | Dataset | Conditions | Selected condition | "
            "Primary evidence | Secondary evidence |"
        ),
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for summary in summaries:
        primary = _format_value(summary.primary_metric, summary.primary_value)
        secondary = _format_value(
            summary.secondary_metric,
            summary.secondary_value,
        )
        lines.append(
            f"| `{summary.experiment}` | `{summary.dataset}` | "
            f"{summary.conditions_evaluated} | `{summary.selected_condition}` | "
            f"`{summary.primary_metric}` = {primary} | "
            f"`{summary.secondary_metric}` = {secondary} |"
        )
    lines.extend(
        [
            "",
            "## Selection Rules and Evidence Scope",
            "",
        ]
    )
    for summary in summaries:
        lines.append(
            f"- `{summary.experiment}/{summary.dataset}`: "
            f"{summary.selection_rule}. Evidence scope: {summary.evidence_scope}. "
            f"Source: `{summary.source_metrics}`."
        )
    lines.extend(
        [
            "",
            "## Interface Review",
            "",
            (
                "All evaluation commands use a positional XYZ input, "
                "accept `--output-dir`,"
            ),
            "and create `metrics.csv` plus `comparison.png`. Additional files contain",
            "method-specific point-level or point-cloud outputs.",
            "",
            "| Experiment | Canonical command | Default output directory |",
            "| --- | --- | --- |",
            (
                "| Voxel downsampling | `evaluate-downsampling` | "
                "`output/voxel_downsampling` |"
            ),
            "| Outlier filtering | `evaluate-outliers` | `output/outlier_filtering` |",
            "| Normal estimation | `evaluate-normals` | `output/normal_estimation` |",
            "| Rigid registration | `evaluate-registration` | `output/registration` |",
            (
                "| Partial-overlap registration | "
                "`evaluate-partial-overlap` | "
                "`output/partial_overlap_registration` |"
            ),
            (
                "| Trim sensitivity | `evaluate-trim-sensitivity` | "
                "`output/trim_sensitivity` |"
            ),
            "",
            "The earlier `evaluate` command remains as an alias for",
            "`evaluate-downsampling`. Reference outputs use the canonical layout",
            "`results/<experiment>/<dataset>/`.",
        ]
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
