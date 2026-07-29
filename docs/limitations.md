# Limitations and Claim Boundaries

## 日本語概要

本書は、点群実験の結果から主張できる範囲を定義します。選択条件が実運用の推奨値ではないこと、異なる手法の指標を単一順位へ統合できないこと、合成条件や固定した初期値が実環境を網羅しないことを明記しています。

実験ごとの制約と適用境界は以下の英語本文を参照してください。

---

## English Summary

This document records the detailed conditions that bound the conclusions in
the [README](../README.md) and
[evaluation results](evaluation-results.md). These constraints are part of the
project evidence, not exceptions to it.

## Limitations

- Summary selections are deterministic review points, not recommended
  production parameters. A task-specific cost function may select a different
  condition.
- Primary metrics are intentionally method-specific. They cannot support a
  combined score, cross-method ranking, or claim that one dataset is easier in
  general.
- The two datasets share an experiment protocol but not the same coordinate
  scale, sampling pattern, geometry, or ground-truth coverage.
- The joint experiment contaminates only the source with isolated vertical
  outliers. It does not evaluate target contamination, clustered noise,
  near-surface artifacts, or sensor-specific error distributions.
- Joint-condition labels and exact generating pairs are available only because
  the experiment is controlled. Real registration pipelines must estimate
  overlap and contamination without this ground truth.
- The 4-by-4 joint grid, fixed initialization, and two trim fractions expose a
  local sensitivity boundary rather than a general robustness guarantee.
- Rejecting every labeled outlier is not sufficient for recovery when a fixed
  policy still has to retain clean source points outside the overlap.
- The v0.4 registration sweep uses complete, one-to-one transformed copies.
  Its recovery boundary measures initialization sensitivity under full overlap.
- Partial overlap is simulated with X-ordered slabs from one cloud. It does not
  model viewpoint-dependent visibility, occlusion, range noise, independently
  sampled scans, or real sensor trajectories.
- Exact overlap pairs are available only because the scans are derived from one
  indexed cloud. Production registration normally has no such ground truth.
- The sensitivity grid is deliberately coarse and does not estimate overlap or
  choose a trim fraction from unlabeled data. Its selected condition uses
  controlled ground truth and is not available in production.
- The observed boundary depends on shared samples, modest initialization, exact
  overlap labels, the nearest-neighbor objective, and the fixed iteration
  budget. A fraction at or below overlap is not guaranteed to recover other
  scans.
- Fractions below 0.4 are not evaluated. Very small retained sets can become
  geometrically unrepresentative or degenerate even when their residuals are
  low.
- Exact-match precision and recall depend on unique generating indices in this
  construction. Repeated coordinates and independently sampled surfaces would
  require a different ground-truth correspondence definition.
- The 70% trim fraction is a fixed comparison condition, not a recommended or
  optimized parameter. Closest-residual trimming can discard correct pairs or
  retain incorrect pairs, especially when actual overlap is lower than the
  retained fraction.
- The ICP implementation has no distance threshold, reciprocal matching,
  multiscale initialization, geometric features, robust loss, or point-to-plane
  objective.
- Exact recovery in this controlled setup must not be generalized to sensor
  scans or unrelated point clouds.
- The convergence flag reports the numerical stopping rule within the fixed
  budget; transform error is the correctness measure for this experiment.
- Translation-vector error depends on the coordinate origin. Normalized
  known-pair RMSE is the more directly comparable geometric measure here.
- Local PCA assumes that a neighborhood is adequately approximated by a plane.
  Curvature, boundaries, mixed surfaces, and non-uniform density can bias the
  estimate.
- Positive-Z orientation is suitable for these height-field and terrain
  experiments, but not for vertical, enclosed, or arbitrarily oriented
  surfaces that require a separate orientation strategy.
- The public sample has no reference normals. Perturbation repeatability is a
  stability measurement and must not be interpreted as accuracy.
- The controlled perturbation is isotropic Gaussian noise and does not model
  a specific lidar sensor or acquisition geometry.
- Surface variation depends on neighborhood scale and sampling distribution;
  lower values do not universally mean better normals.
- Injected noise is limited to isolated points above or below the reference
  cloud. It does not model clustered noise, multipath returns, or near-surface
  sensor artifacts.
- The source cloud is assumed to be clean. Ground-truth labels apply only to
  injected points and do not validate the original USGS classifications.
- The filter uses one global distance threshold and is sensitive to varying
  density, neighborhood size, geometry, and coordinate scale.
- F1 assigns equal importance to precision and recall. An application may
  require a different cost function and validation dataset.
- Selecting the highest-F1 threshold uses the injected ground-truth labels and
  is an evaluation procedure, not an unsupervised production-tuning method.
- XYZ input stores coordinates only; intensity, classification, color, and
  coordinate reference systems are not preserved.
- The implementation loads the complete cloud into memory and is not intended
  for large-scale production processing.
- The public sample is a small, ground-only subset rather than a complete lidar
  scene, so its results must not be generalized to all 3D data.
- Comparison figures are diagnostic projections, not full 3D viewers.
- The verifier checks implementation repeatability for the committed inputs
  and protocol. It is not independent scientific replication and does not
  establish external validity.
- PNG verification covers file inventory, validity, and dimensions rather than
  byte identity because rendering libraries, fonts, and compression can vary.
- Supported dependencies use minimum versions rather than one universal lock
  file. Archival reproduction should record the interpreter and installed
  package versions.
- The v1.0 stability guarantee covers the documented repository interfaces,
  not PyPI availability, large-scale processing, deployment support, or
  production suitability for arbitrary sensor data.
- Stable interfaces preserve compatibility, but controlled experimental
  conclusions remain bounded by the documented inputs, parameters, and metrics.
