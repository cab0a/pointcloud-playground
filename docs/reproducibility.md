# Reproducibility

## 日本語概要

本書は、固定入力、実験条件、乱数種、CSV、比較図から参照結果を再現する手順を定義します。別の出力先への一括再生成、コミット済み成果物との照合、数値計算と画像描画の環境差を考慮した検証境界、入力チェックサム、決定論の対象範囲を記録しています。

環境構築、検証方法、再現性の境界は以下の英語本文を参照してください。

---

## English Summary

This guide defines how to recreate the committed point-cloud evidence from
versioned inputs, fixed parameters, deterministic seeds, CSV metrics, and
diagnostic figures without overwriting the reference results.

## Reproduction contract

The committed evidence is based on two versioned inputs, fixed experiment
parameters, deterministic random seeds, CSV metrics, and static diagnostic
figures. The reference runner reads the committed inputs without modifying
them and can write a complete result set to a separate directory.

From a clean clone:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest
python -m build
python experiments/run_reference_experiments.py \
  --output-root reproduced_results
```

Windows PowerShell activation uses `.venv\Scripts\Activate.ps1`; the remaining
commands are unchanged.

The generated directory follows the same
`<experiment>/<dataset>/metrics.csv` and `comparison.png` layout as `results/`.
Using a separate output root keeps the committed evidence unchanged during an
independent reproduction run.

## Automated verification

Run the complete verification from the repository root:

```bash
python experiments/verify_reference_results.py
```

The verifier performs the following checks:

1. Regenerates the synthetic input and compares it with the committed
   six-decimal XYZ representation.
2. Checks the public USGS-derived sample against the SHA-256 value recorded in
   its manifest.
3. Regenerates every experiment in a temporary directory.
4. Confirms the CSV inventory, schemas, row counts, text fields, and numeric
   values with explicit tolerances.
5. Confirms the generated cross-experiment Markdown summary exactly.
6. Confirms the figure inventory, PNG validity, and image dimensions.

Default numeric comparison tolerances are `1e-9` relative and `1e-12`
absolute. `rotation_error_deg` additionally uses a `5e-6` degree absolute
tolerance near zero. This covers the observed variation in inverse-cosine
rotation recovery across supported NumPy and SciPy environments without
relaxing the comparison of other metrics. The general tolerances can be
changed explicitly when diagnosing platform-dependent floating-point
differences:

```bash
python experiments/verify_reference_results.py \
  --relative-tolerance 1e-8 \
  --absolute-tolerance 1e-10
```

Changing a tolerance is a diagnostic choice and should be reported with any
reproduction result. It must not be used to conceal a material metric change.

## Input provenance

- `data/synthetic_controlled_density.xyz` is produced by
  `generate_controlled_density_cloud` with its documented defaults.
- `data/usgs_3dep_iowa/sample.xyz` is derived from a public USGS 3DEP tile.
  Its URL, source checksum, filtering rule, sample size, seed, coordinate
  offset, and sample checksum are recorded in
  `data/usgs_3dep_iowa/manifest.csv`.
- `experiments/prepare_public_sample.py` rebuilds the public sample after the
  optional `data` dependencies are installed.

No private or organization-specific inputs are required.

## Determinism boundaries

The numeric experiment inputs, controlled perturbations, and sampling steps
use fixed seeds. CSV metrics are the primary reproducibility artifacts.

PNG files are checked structurally rather than byte-for-byte. Matplotlib,
font, compression, and platform differences can change PNG bytes without
changing the plotted data. Visual review is still required when a figure's
implementation changes.

The project declares supported dependency minimums rather than a universal
cross-platform lock file. CI verifies installation, the public API, CLI, and
tests on Python 3.10 through 3.14. For an archival reproduction, also record
the interpreter and installed packages:

```bash
python --version
python -m pip freeze > reproduction-environment.txt
```

An environment record created this way is local evidence and is not required
to run the repository.

## Stable-release qualification

Stable releases also follow
[`docs/release-checklist.md`](release-checklist.md). The checklist covers
version synchronization, API and output compatibility, wheel construction,
README commands, sensitive-information review, CI, tagging, Release notes, and
profile synchronization. Passing the checklist qualifies the documented
repository interface; it does not convert the experimental methods into a
large-scale production library.
