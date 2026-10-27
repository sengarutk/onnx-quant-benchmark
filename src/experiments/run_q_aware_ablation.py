"""
Q-Aware NMS ablation and decision-flip attribution.

For every detector configuration (PyTorch FP32 reference, ONNX Runtime FP32 / FP16 / static INT8 on CPU, plus
ORT CUDA when that provider is available):
  1. cache raw outputs on the labeled CALIBRATION split and grid-search (tau_conf, tau_iou) maximizing F1;
  2. evaluate on the held-out TEST split three operating points: the default (0.25, 0.45), the thresholds calibrated
     on the FP32 reference and transferred unchanged ("FP32-transfer"), and the configuration's own calibration;
     own-vs-transfer isolates the precision-specific benefit from generic threshold tuning;
  3. quantify uncertainty with a paired image-level bootstrap 95% CI of delta-F1 (B = 2000) and a Wilcoxon
     signed-rank test on per-image error counts (FP + FN), Holm-Bonferroni corrected across configurations;
  4. attribute detection flips against the PyTorch FP32 reference, both at default thresholds and with each
     model's own calibrated thresholds.

Outputs: results/q_aware_nms.json, results/tables/table6_q_aware_nms_ablation.md,
results/tables/table7_decision_flip_audit.md, results/figures/q_aware_pareto_recovery.png,
results/figures/decision_flip_attribution.png
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.decision_flips import compute_detection_flips
from src.analysis.stats import holm_bonferroni, wilcoxon_paired_test
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.data.writers import load_manifest
from src.models.yolo_adapter import YOLOAdapter
from src.quantization.q_aware_nms import DEFAULT_CONF, DEFAULT_IOU, calibrate_q_aware_nms, detection_counts
from src.runtimes.ort_cpu_runtime import ORTCPURuntime
from src.runtimes.ort_cuda_runtime import ORTCUDARuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.validation.detection_quality import evaluate_detection_dataset
from src.validation.task_eval import MAP_CONF_THRESHOLD, run_detector, score_detections

logger = setup_logger("run_q_aware_ablation")
EXPORTED = PROJECT_ROOT / "models" / "exported"


def per_image_counts(preds: List[List[Dict]], gts: List[List[Dict]]) -> np.ndarray:
    """[N, 3] array of (TP, FP, FN) per image."""
    return np.array([detection_counts([p], [g]) for p, g in zip(preds, gts)], dtype=np.int64)


def micro_f1(counts: np.ndarray) -> float:
    tp, fp, fn = counts.sum(axis=0)
    return float(2 * tp / (2 * tp + fp + fn)) if tp > 0 else 0.0


def paired_bootstrap_delta_f1(c_def: np.ndarray, c_cal: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    """Image-level paired bootstrap of F1(calibrated) - F1(default); returns (delta, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    n = len(c_def)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = np.array([micro_f1(c_cal[i]) - micro_f1(c_def[i]) for i in idx])
    return micro_f1(c_cal) - micro_f1(c_def), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _timed_postprocess(cache, adapter, conf, iou) -> Tuple[List[List[Dict]], float]:
    t0 = time.perf_counter()
    preds = score_detections(cache, adapter, conf, iou)
    return preds, (time.perf_counter() - t0) / len(cache) * 1000.0


def run_q_aware_nms_ablation(results_dir: Path = PROJECT_ROOT / "results") -> Dict[str, Any]:
    seed_everything(42)
    adapter = YOLOAdapter(require_weights=True)
    calib = load_manifest(PROJECT_ROOT / "data" / "calibration" / "manifest.json")["detection_samples"]
    test = load_manifest(PROJECT_ROOT / "data" / "sample_images" / "manifest.json")["detection_samples"]
    test_gts = [s["ground_truth_boxes"] for s in test]

    configs = [("PyTorch FP32 (CPU)", "fp32", "PyTorch_CPU",
                lambda: PyTorchRuntime(adapter.get_pytorch_model(), device="cpu"))]
    for prec, fname in (("fp32", "yolo_nano_fp32_opset17.onnx"), ("fp16", "yolo_nano_fp16.onnx"),
                        ("int8", "yolo_nano_static_int8.onnx")):
        path = EXPORTED / fname
        if path.is_file():
            configs.append((f"ORT {prec.upper()} (CPU)", prec, "ORT_CPU", lambda p=path: ORTCPURuntime(p)))
            if prec != "int8" and "CUDAExecutionProvider" in ort.get_available_providers():
                configs.append((f"ORT {prec.upper()} (CUDA)", prec, "ORT_CUDA", lambda p=path: ORTCUDARuntime(p)))
        else:
            logger.warning(f"Skipping {fname}: not exported")

    rows, flips, ref = [], [], None
    for name, prec, rt_name, make in configs:
        rt = make()
        in_name = next(iter(rt.get_input_spec()))
        predict = lambda x, _rt=rt, _n=in_name: list(_rt.predict({_n: x}).values())
        calib_cache = run_detector(predict, calib)
        test_cache = run_detector(predict, test)

        policy = calibrate_q_aware_nms(calib_cache, adapter, precision=prec, runtime=rt_name)
        conf, iou = policy["optimal_conf"], policy["optimal_iou"]

        if ref is None:
            ref_policy = (conf, iou)  # thresholds calibrated on the FP32 reference, transferred to every target
        d_preds, d_ms = _timed_postprocess(test_cache, adapter, DEFAULT_CONF, DEFAULT_IOU)
        c_preds, c_ms = _timed_postprocess(test_cache, adapter, conf, iou)
        t_preds = score_detections(test_cache, adapter, *ref_policy)
        c_tr = per_image_counts(t_preds, test_gts)
        c_def, c_cal = per_image_counts(d_preds, test_gts), per_image_counts(c_preds, test_gts)
        delta, lo, hi = paired_bootstrap_delta_f1(c_def, c_cal)
        err_def = c_def[:, 1] + c_def[:, 2]
        err_cal = c_cal[:, 1] + c_cal[:, 2]
        wil = wilcoxon_paired_test(err_def, err_cal)
        # Precision-specific effect: own calibration vs the FP32 reference's calibrated thresholds
        q_delta, q_lo, q_hi = paired_bootstrap_delta_f1(c_tr, c_cal)
        wil_q = wilcoxon_paired_test(c_tr[:, 1] + c_tr[:, 2], err_cal)
        map_def = evaluate_detection_dataset(score_detections(test_cache, adapter, MAP_CONF_THRESHOLD, DEFAULT_IOU), test_gts)
        map_cal = evaluate_detection_dataset(score_detections(test_cache, adapter, MAP_CONF_THRESHOLD, iou), test_gts)

        if ref is None:
            ref = {"default": d_preds, "calibrated": c_preds}
        flip_def = compute_detection_flips(ref["default"], d_preds, test_gts)
        flip_cal = compute_detection_flips(ref["calibrated"], c_preds, test_gts)

        rows.append({
            "config": name, "precision": prec, "runtime": rt_name,
            "calibration": policy,
            "default": {"conf": DEFAULT_CONF, "iou": DEFAULT_IOU, "f1": micro_f1(c_def),
                        "tp": int(c_def[:, 0].sum()), "fp": int(c_def[:, 1].sum()), "fn": int(c_def[:, 2].sum()),
                        "mAP_50": map_def["mAP_50"], "postprocess_ms": d_ms},
            "calibrated": {"conf": conf, "iou": iou, "f1": micro_f1(c_cal),
                           "tp": int(c_cal[:, 0].sum()), "fp": int(c_cal[:, 1].sum()), "fn": int(c_cal[:, 2].sum()),
                           "mAP_50": map_cal["mAP_50"], "postprocess_ms": c_ms},
            "fp32_transfer": {"conf": ref_policy[0], "iou": ref_policy[1], "f1": micro_f1(c_tr),
                              "tp": int(c_tr[:, 0].sum()), "fp": int(c_tr[:, 1].sum()), "fn": int(c_tr[:, 2].sum())},
            "delta_f1": delta, "delta_f1_ci95": [lo, hi],
            "wilcoxon_errors": wil,
            "delta_f1_vs_transfer": q_delta, "delta_f1_vs_transfer_ci95": [q_lo, q_hi],
            "wilcoxon_vs_transfer": wil_q,
        })
        flips.append({"config": name, "precision": prec, "runtime": rt_name, "default_thresholds": flip_def,
                      "calibrated_thresholds": flip_cal})
        logger.info(f"{name}: test F1 {micro_f1(c_def):.4f} -> {micro_f1(c_cal):.4f} (delta {delta:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]), "
                    f"flip rate vs FP32 (default thr.) {flip_def['flip_rate']:.4f}")
        rt.cleanup()

    for key in ("wilcoxon_errors", "wilcoxon_vs_transfer"):
        holm = holm_bonferroni({r["config"]: r[key]["p_value"] for r in rows})
        for r in rows:
            r[key]["p_holm"] = holm[r["config"]]["p_adjusted"]
            r[key]["reject_holm_05"] = holm[r["config"]]["reject"]

    out = {"calibration_images": len(calib), "test_images": len(test), "test_boxes": sum(len(g) for g in test_gts),
           "grid": "conf 0.10..0.95 x IoU 0.05..0.70, step 0.05 (252 points)", "ablation": rows, "flips": flips}
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "q_aware_nms.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_tables(out, results_dir / "tables")
    _plot(out, results_dir / "figures")
    return out


def _write_tables(out: Dict[str, Any], tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    h6 = ["Configuration", "Calibrated (conf, IoU)", "F1 default", "F1 FP32-transfer", "F1 own calibration",
          "Delta vs default [95% CI]", "p Holm", "Delta vs FP32-transfer [95% CI]", "p Holm",
          "Post-proc. ms (default / calibrated)"]
    lines = [f"| {' | '.join(h6)} |", f"| {' | '.join(['---'] * len(h6))} |"]
    for r in out["ablation"]:
        d, c, t = r["default"], r["calibrated"], r["fp32_transfer"]
        lines.append(
            f"| {r['config']} | ({c['conf']:.2f}, {c['iou']:.2f}) | {d['f1']:.4f} | {t['f1']:.4f} | {c['f1']:.4f} | "
            f"{r['delta_f1']:+.4f} [{r['delta_f1_ci95'][0]:+.4f}, {r['delta_f1_ci95'][1]:+.4f}] | {r['wilcoxon_errors']['p_holm']:.3g} | "
            f"{r['delta_f1_vs_transfer']:+.4f} [{r['delta_f1_vs_transfer_ci95'][0]:+.4f}, {r['delta_f1_vs_transfer_ci95'][1]:+.4f}] | "
            f"{r['wilcoxon_vs_transfer']['p_holm']:.3g} | {d['postprocess_ms']:.2f} / {c['postprocess_ms']:.2f} |"
        )
    (tables_dir / "table6_q_aware_nms_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    h7 = ["Target vs PyTorch FP32", "Thresholds", "Target boxes", "Matched", "Flip rate", "Lost TPs", "New TPs",
          "New FPs", "Suppressed FPs", "FP share of flips"]
    lines = [f"| {' | '.join(h7)} |", f"| {' | '.join(['---'] * len(h7))} |"]
    for f in out["flips"]:
        for key, label in (("default_thresholds", "default"), ("calibrated_thresholds", "calibrated")):
            x = f[key]
            lines.append(
                f"| {f['config']} | {label} | {x['total_target_boxes']} | {x['total_matched_boxes']} | {x['flip_rate']:.4f} | "
                f"{x['lost_tps']} | {x['new_tps']} | {x['new_fps']} | {x['suppressed_fps']} | {x['frac_flip_fp']:.1%} |"
            )
    (tables_dir / "table7_decision_flip_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(out: Dict[str, Any], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = out["ablation"]
    labels = [r["config"] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 0.8 * len(rows) + 2.2), dpi=300)
    for i, r in enumerate(rows):
        d, c = r["default"]["f1"], r["calibrated"]["f1"]
        ax.plot([d, c], [i, i], color="#888", lw=2, zorder=2)
        ax.scatter(d, i, s=80, color="#4575b4", edgecolor="black", zorder=3, label="Default (0.25, 0.45)" if i == 0 else None)
        ax.scatter(r["fp32_transfer"]["f1"], i, s=70, marker="s", color="#fdae61", edgecolor="black", zorder=3,
                   label="FP32-calibrated thresholds (transferred)" if i == 0 else None)
        ax.scatter(c, i, s=90, marker="^", color="#d73027", edgecolor="black", zorder=4, label="Own calibration (Q-Aware)" if i == 0 else None)
        ax.annotate(f"own vs FP32-thr.: {r['delta_f1_vs_transfer']:+.3f}\n"
                    f"[{r['delta_f1_vs_transfer_ci95'][0]:+.3f}, {r['delta_f1_vs_transfer_ci95'][1]:+.3f}]",
                    (max(d, c, r["fp32_transfer"]["f1"]), i), textcoords="offset points", xytext=(10, -8), fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Test-split micro F1 (IoU 0.50)")
    ax.set_title("Q-Aware NMS: F1 at default, FP32-transferred and own calibrated thresholds", fontweight="bold")
    lo = min(min(r["default"]["f1"], r["calibrated"]["f1"], r["fp32_transfer"]["f1"]) for r in rows)
    hi = max(max(r["default"]["f1"], r["calibrated"]["f1"], r["fp32_transfer"]["f1"]) for r in rows)
    ax.set_xlim(lo - 0.03, hi + 0.22)
    ax.grid(True, ls=":", alpha=0.5, axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "q_aware_pareto_recovery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    targets = out["flips"][1:]
    def _share(f: Dict[str, Any], key: str) -> float:
        return f[key]["total_flips"] / max(f[key]["total_matched_boxes"] + f[key]["total_flips"], 1)

    ymax = max([_share(f, k) for f in targets for k in ("default_thresholds", "calibrated_thresholds")] + [0.04])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300, sharey=True)
    parts = [("lost_tps", "Lost TPs", "#4575b4"), ("new_tps", "New TPs", "#91bfdb"),
             ("new_fps", "New FPs", "#d73027"), ("suppressed_fps", "Suppressed FPs", "#fc8d59")]
    for ax, key, title in zip(axes, ("default_thresholds", "calibrated_thresholds"),
                              ("Default thresholds (both models)", "Each model's calibrated thresholds")):
        x = np.arange(len(targets))
        bottom = np.zeros(len(targets))
        for k, lbl, col in parts:
            vals = np.array([f[key][k] / max(f[key]["total_matched_boxes"] + f[key]["total_flips"], 1) for f in targets])
            ax.bar(x, vals, 0.55, bottom=bottom, label=lbl, color=col, edgecolor="black", lw=0.6)
            bottom += vals
        for i, f in enumerate(targets):
            ax.text(i, bottom[i], f"Phi={f[key]['flip_rate']:.3f}\n({f[key]['total_flips']} flips)", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f["config"] for f in targets], rotation=15, fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.5, axis="y")
    axes[0].set_ylabel("Share of decision events vs PyTorch FP32")
    axes[0].set_ylim(0, ymax * 1.35)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Decision-change (flip) attribution vs PyTorch FP32 reference (test split)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "decision_flip_attribution.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    run_q_aware_nms_ablation(Path(parser.parse_args().results_dir))
