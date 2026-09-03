# Release Notes — v1.1.0: Methodological Upgrade & Academic Conference Manuscript

**Release Tag**: `v1.1.0`  
**Target Architecture**: PyTorch 2.5.1+cu121, ONNX Runtime 1.23.2, CUDA 12.1, TensorRT 10.x  
**Paper Package**: Flat-Root Ready-to-Compile [`overleaf_paper.zip`](../overleaf_paper.zip) (2.86 MB)  

---

## 1. Executive Summary & Release Highlights

Version **v1.1.0** upgrades the benchmark suite from a foundational systems evaluation into a publication-grade scientific artifact accompanying the conference manuscript:
> **Q-Aware Post-Processing and Rigorous Latency Profiling for Quantized Edge Vision Inference**  
> *Utkarsh Sengar (Indian Institute of Technology Dharwad)*

This release introduces adaptive quantization-aware post-processing, mathematical attribution of precision-induced decision flips, non-parametric statistical bootstrap confidence intervals, multi-dimensional scalability profiling across batch sizes and resolutions, and a self-contained IEEEtran Overleaf publication package.

---

## 2. Core Scientific & Systems Contributions

### 1. Q-Aware Post-Processing Calibration (Q-Aware NMS)
- **Problem**: Standard post-processing (fixed confidence threshold $\tau_{\text{conf}}=0.25$, IoU threshold $\tau_{\text{IoU}}=0.45$) fails on quantized models due to activation distribution compression and lower output logits.
- **Solution**: Implemented `src/quantization/q_aware_nms.py`, performing sub-second 2D grid searches over $(\tau_{\text{conf}}, \tau_{\text{IoU}}) \in [0.10, 0.60] \times [0.30, 0.70]$ on held-out calibration data using pre-cached forward passes.
- **Result**: Recovers **$+4.2\%$ $F_1$-score** on static INT8 models, reclaiming the Pareto optimality frontier lost during naive post-processing. Documented in `results/tables/table6_q_aware_nms_ablation.md` and visualized in `results/figures/q_aware_pareto_recovery.png`.

### 2. Quantization Decision-Change Attribution (Decision Flips)
- **Formulation**: Implemented `src/analysis/decision_flips.py` measuring total symmetric difference flip rate over the total decision space:
  $$\Phi = \frac{|\mathcal{B}_{\text{ref}} \triangle \mathcal{B}_{\text{target}}|}{|\mathcal{B}_{\text{matched}}| + |\mathcal{B}_{\text{ref}} \triangle \mathcal{B}_{\text{target}}|}$$
- **Fine-Grained Partitioning**: Decomposes $\Phi$ into False Positives introduced ($\Phi_{\text{FP}}$) versus True Positives dropped ($\Phi_{\text{TP}}$).
- **Takeaway**: Detection flip rates scale with precision reduction from FP16 ($\Phi = 4.2\%$) to INT8 ($\Phi = 18.7\%$). In uncalibrated INT8 models, $68.4\%$ of flips correspond to spurious false-alarm background noise, which Q-Aware NMS suppresses by $84.2\%$. Documented in `results/tables/table7_decision_flip_audit.md` and `results/figures/decision_flip_attribution.png`.

### 3. Non-Parametric Statistical Framework & Bootstrap CIs
- **Bootstrap Confidence Intervals**: Implemented `src/analysis/stats.py` calculating rigorous 95% bootstrap confidence intervals ($B=2{,}000$ resamples) for all percentile latencies ($p_{50}, p_{90}, p_{95}, p_{99}$) and detection scores.
- **Paired Hypothesis Testing**: Integrates two-sided Wilcoxon signed-rank significance tests with Holm-Bonferroni step-down family-wise error rate corrections.

### 4. 16-Configuration Scalability Grid Sweeps
- **Profiling Grid**: Comprehensive 2D sweep across 4 batch sizes $\{1, 2, 4, 8\}$ and 4 input resolutions $\{320\times 320, 416\times 416, 512\times 512, 640\times 640\}$.
- **Idempotent Automation**: Profiled via `src/experiments/run_scalability_sweep.py`, serialized to `results/scalability_sweep.csv` (16 configurations), and visualized in `results/figures/scalability_batch_resolution.png`.
- **System Insights**: Batching yields sub-linear latency growth ($< 1.3\times$ latency penalty for $2\times$ batch size), maximizing GPU compute saturation on edge accelerators.

### 5. Self-Contained Overleaf Academic Paper Package
- **Archive Structure**: `overleaf_paper.zip` is structured with a flat root (`main.tex` and `references.bib` directly at archive root) to guarantee instant 1-click Overleaf compilation without directory path adjustments.
- **Figure Assets**: Embeds all 9 authoritative 300-DPI publication figures in `figures/`.
- **Style Compliance**: Conforms strictly to IEEEtran conference requirements using `\usepackage[caption=false,font=footnotesize]{subfig}` and `\subfloat` blocks to eliminate style clash warnings.
- **Full Literature Integration**: Complete 15-entry BibTeX bibliography (`paper/references.bib`) covering quantization, object detection, systems benchmarking, and statistical literature.

---

## 3. Test Suite & Verification Matrix

- **Unit Test Coverage**: **115 passing tests** across 15 test modules.
- **Statement Coverage**: **90% total source coverage** across all modules in `src/`.
- **Smoke Test Pipeline**: Fully automated and idempotent via `bash scripts/smoke_test.sh` (environment check, report synthesis, artifact data integrity, Overleaf zip check, and pytest suite).

```
================================ tests coverage ================================
TOTAL                                       3275    326    90%
======================= 115 passed, 1 warning in 52.32s ========================
============================================================
>>> SMOKE TEST PASSED WITH 100% SUCCESS <<<
============================================================
```

---

## 4. BibTeX Citation

```bibtex
@inproceedings{sengar2026qaware,
  author    = {Utkarsh Sengar},
  title     = {Q-Aware Post-Processing and Rigorous Latency Profiling for Quantized Edge Vision Inference},
  booktitle = {Proceedings of the IEEE/ACM Conference on Connected and Edge Systems (EdgeSys)},
  year      = {2026}
}
```
