# Changelog

## v2.0.1 — Corrections and compatibility (2026-10-28)

No measurement was re-run; every result file (`runs.csv`, `runs.json`, tables, Q-Aware and sweep data) is unchanged.

- **CI fix:** PyTorch >= 2.9 switched `torch.onnx.export` to the dynamo exporter, which needs `onnxscript` and broke
  the export, parity and quantization tests on CI (v2.0.0's CI run failed). The exporter is now requested explicitly
  (`dynamo=False`), matching the published graphs. GitHub Actions moved to `checkout@v5` / `setup-python@v6`.
- **Paper correction:** the paper stated that session-to-session variation stayed within ±5 % for all but one
  configuration; the session data show 10 of 12 (the autoencoder ORT INT8 run also has two sessions 5.8–5.9 %
  above its median). The sentence is now generated from `runs.json`.
- **Figures** are drawn at their printed size with legible fonts (they were scaled to ~27 % in the paper), the
  quality-vs-latency plot uses a legend instead of overlapping point labels, and two-panel figures span both
  columns. Figures are numbered in citation order.
- **Units:** all MB values are binary (2^20 bytes); this is now stated in the README, methodology and paper.
- **Reproducible builds:** `paper/main.pdf` and `overleaf_paper.zip` are byte-identical across rebuilds (timestamps
  pinned to the release date via `SOURCE_DATE_EPOCH`).
- Removed 42 unused imports; `run_q_aware_ablation.py` and `run_scalability_sweep.py` gained `--plot-only`.

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
- **Publication claim:** the v1.1.0 README and release notes gave a BibTeX entry citing the paper in the *Proceedings
  of the IEEE/ACM Conference on Connected and Edge Systems (EdgeSys)*. The paper was never submitted to EdgeSys or any
  other venue; it is an unpublished preprint and must be cited only as the software (see `CITATION.cff`).

The `v1.0.0` and `v1.1.0` tags were removed from GitHub on 2026-10-28 so that no published version presents these
results; the commits remain in the history of `main` for transparency.

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
