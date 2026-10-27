# Changelog

## v2.0.0 — Reproducible re-implementation (2026-10-27)

v2.0.0 replaces the v1.x results end to end. Every number in the README, `results/`, `docs/` and the paper is now
produced by the pipeline (`make pipeline`) from one recorded benchmark session.

### Erratum for v1.0.0 / v1.1.0

An audit of v1.x found that several published results were not measured and must not be cited:

- **Q-Aware NMS and decision-flip results (Tables 6–7, Figs. `q_aware_pareto_recovery`, `decision_flip_attribution`,
  the "+4.2 % F1" and "Φ = 4.2 % / 18.7 %" claims)** were hardcoded in `run_q_aware_ablation.py`; the computed values
  were discarded. The calibration split had no labels, and the v1 models were untrained (random weights), producing
  zero detections.
- **Task quality** in `runs.csv` (mAP@50 = 0.0099, AUROC = 1.0, Δ = 0 for every precision) were constants passed by
  `benchmark_all.py`; 0.0099 itself was an AP-interpolation bug (1/101 for a detector with no true positives).
- **Latency stage breakdown** split the end-to-end overhead 60/40 by formula instead of measuring it.
- **README, paper Table I and `runs.csv`** were taken from three different measurement sessions; several reported CVs,
  VRAM and RSS figures did not match any run. Hardware was misreported (the host CPU is an Intel Core i5-13450HX).
- Several gates were hardcoded to pass, and Holm–Bonferroni, claimed in the paper, was not implemented.

### Added
- Procedural synthetic train / validation / calibration / test splits with disjoint seed ranges and a leakage check.
- Training from scratch (`scripts/train_models.py`): anchor-free 2-class detector (Quality Focal + GIoU loss) and a
  convolutional autoencoder; anomaly-score hyper-parameters selected on the validation split.
- Session-based benchmarking: pooled percentiles with bootstrap CIs, session-median CV stability gate, measured
  end-to-end stage breakdown, parity of all outputs vs PyTorch FP32, task quality measured per runtime, and
  acceptance gates read from `configs/`.
- ORT CPU FP16 and PyTorch CUDA FP16 configurations; ORT CUDA / TensorRT run automatically when available.
- Real Q-Aware NMS ablation (calibration split → test split) with paired bootstrap CIs, Wilcoxon tests and
  Holm–Bonferroni correction; flip attribution at default and calibrated thresholds.
- Paper generated from results (`scripts/build_paper.py`, `\res{...}` macros), GitHub Actions CI.

### Fixed
- COCO AP interpolation, precision/recall fields, degenerate AUROC values (now NaN, not 1.0).
- CUDA timing no longer synchronizes inside the timed call; warmup always runs the configured iterations.
- TensorRT stream ordering, dtype mapping and INT8 build validation; functional fallback audit.
- ConvTranspose layers are included in static INT8 quantization.
- `numpy>=2` requirement (AU-PRO uses `np.trapezoid`), missing `onnxconverter-common` dependency.
- Full Apache-2.0 license text, correct author/URL metadata, corrected bibliography entries.

### Removed
- `scratch/` generator scripts, legacy `results/plots/` and `runs_with_speedups.csv`, IDE files.
