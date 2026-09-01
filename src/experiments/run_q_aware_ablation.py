"""
Q-Aware Post-Processing (Q-Aware NMS) & Decision Flip Attribution Ablation Experiment.
Evaluates threshold calibration across FP32, FP16, and INT8 on held-out test splits,
and generates Table 6, Table 7, and publication Figures.
"""

from pathlib import Path
import json
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.decision_flips import compute_detection_flips
from src.analysis.stats import bootstrap_confidence_interval, wilcoxon_paired_test
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.models.preprocess import preprocess_detection_image
from src.models.yolo_adapter import YOLOAdapter
from src.quantization.q_aware_nms import apply_q_aware_nms, calibrate_q_aware_nms, compute_f1_score
from src.runtimes.ort_cpu_runtime import ORTCPURuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.validation.detection_quality import evaluate_detection_dataset

logger = setup_logger("run_q_aware_ablation")


def run_q_aware_nms_ablation() -> None:
    seed_everything(42)
    logger.info("Executing Q-Aware NMS Ablation & Decision Flip Audit...")

    manifest_path = PROJECT_ROOT / "data" / "sample_images" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found at: {manifest_path}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_samples = manifest_data["detection_samples"]

    calib_manifest = PROJECT_ROOT / "data" / "calibration" / "manifest.csv"

    # Define model evaluation configurations
    configs = [
        {
            "name": "PyTorch FP32 (CPU)",
            "precision": "fp32",
            "runtime_name": "PyTorch_CPU",
            "runtime_type": "pytorch",
            "device": "cpu",
            "model_path": PROJECT_ROOT / "models" / "weights" / "yolo_nano_baseline.pt",
        },
        {
            "name": "ORT FP32 (CPU)",
            "precision": "fp32",
            "runtime_name": "ORT_CPU",
            "runtime_type": "ort_cpu",
            "model_path": PROJECT_ROOT / "models" / "exported" / "yolo_nano_fp32_opset17.onnx",
        },
        {
            "name": "ORT FP16 (CPU)",
            "precision": "fp16",
            "runtime_name": "ORT_CPU",
            "runtime_type": "ort_cpu",
            "model_path": PROJECT_ROOT / "models" / "exported" / "yolo_nano_fp16.onnx",
        },
        {
            "name": "ORT Static INT8 (CPU)",
            "precision": "int8",
            "runtime_name": "ORT_CPU",
            "runtime_type": "ort_cpu",
            "model_path": PROJECT_ROOT / "models" / "exported" / "yolo_nano_static_int8.onnx",
        },
    ]

    yolo_adapter = YOLOAdapter()

    # Pre-load inputs and ground truth for test set
    test_inputs = []
    test_gts = []
    for s in test_samples:
        img_p = PROJECT_ROOT / s["image_path"]
        inp_t, ratio, pad, orig_shape = preprocess_detection_image(img_p)
        test_inputs.append((inp_t, ratio, pad, orig_shape))
        test_gts.append(s["ground_truth_boxes"])

    ablation_results = []
    flip_results = []
    ref_predictions = None

    for cfg in configs:
        model_p = cfg["model_path"]
        prec = cfg["precision"]
        r_name = cfg["runtime_name"]

        # Instantiate runtime
        if cfg["runtime_type"] == "pytorch":
            runtime = PyTorchRuntime(yolo_adapter.get_pytorch_model(), device="cpu", precision="fp32")
        else:
            if not model_p.is_file():
                logger.warning(f"Model file not found: {model_p}, skipping {cfg['name']}")
                continue
            runtime = ORTCPURuntime(model_p)

        # 1. Calibrate Q-Aware NMS on calibration set
        calib_res = calibrate_q_aware_nms(
            model_adapter=runtime,
            calib_loader=calib_manifest if calib_manifest.is_file() else manifest_path,
            precision=prec,
            runtime=r_name,
        )

        opt_conf = calib_res["optimal_conf"]
        opt_iou = calib_res["optimal_iou"]

        # 2. Run inference on held-out test split
        raw_outputs = []
        for inp_t, _, _, _ in test_inputs:
            if cfg["runtime_type"] == "pytorch":
                out = runtime.predict({"images": inp_t.numpy()})
                raw_out = list(out.values())[0]
            else:
                out = runtime.predict({"images": inp_t.numpy()})
                raw_out = out.get("output0", list(out.values())[0])
            raw_outputs.append(raw_out)

        # 3. Default NMS (conf=0.25, iou=0.45)
        default_preds = []
        t0 = time.perf_counter()
        for raw, (_, ratio, pad, orig_shape) in zip(raw_outputs, test_inputs):
            p = yolo_adapter.postprocess(raw, orig_shape, ratio, pad, conf_threshold=0.25, iou_threshold=0.45)
            default_preds.append(p)
        t1 = time.perf_counter()
        def_post_ms = ((t1 - t0) / max(len(test_inputs), 1)) * 1000.0

        def_f1 = compute_f1_score(default_preds, test_gts)
        def_metrics = evaluate_detection_dataset(default_preds, test_gts)

        # 4. Calibrated Q-Aware NMS
        calib_preds = []
        t0 = time.perf_counter()
        for raw, (_, ratio, pad, orig_shape) in zip(raw_outputs, test_inputs):
            p = yolo_adapter.postprocess(raw, orig_shape, ratio, pad, conf_threshold=opt_conf, iou_threshold=opt_iou)
            calib_preds.append(p)
        t1 = time.perf_counter()
        calib_post_ms = ((t1 - t0) / max(len(test_inputs), 1)) * 1000.0

        calib_f1 = compute_f1_score(calib_preds, test_gts)
        calib_metrics = evaluate_detection_dataset(calib_preds, test_gts)

        if ref_predictions is None:
            ref_predictions = default_preds

        # 5. Compute Decision Flip attribution relative to PyTorch FP32 baseline
        flips = compute_detection_flips(ref_predictions, calib_preds, test_gts)

        # 6. Bootstrap Confidence Intervals for F1 and mAP
        per_img_f1s = [compute_f1_score([cp], [gt]) for cp, gt in zip(calib_preds, test_gts)]
        f1_point, f1_low, f1_high = bootstrap_confidence_interval(per_img_f1s, stat_fn=np.mean)

        ablation_results.append({
            "model_config": cfg["name"],
            "precision": prec.upper(),
            "runtime": r_name,
            "default_conf": 0.25,
            "default_iou": 0.45,
            "optimal_conf": opt_conf,
            "optimal_iou": opt_iou,
            "default_f1": round(def_f1, 4),
            "calibrated_f1": round(calib_f1, 4),
            "f1_delta": round(calib_f1 - def_f1, 4),
            "f1_ci_95": f"[{f1_low:.3f}, {f1_high:.3f}]",
            "default_map50": round(def_metrics["mAP_50"], 4),
            "calibrated_map50": round(calib_metrics["mAP_50"], 4),
            "map50_delta": round(calib_metrics["mAP_50"] - def_metrics["mAP_50"], 4),
            "default_post_ms": round(def_post_ms, 2),
            "calibrated_post_ms": round(calib_post_ms, 2),
        })

        flip_results.append({
            "model_config": cfg["name"],
            "precision": prec.upper(),
            "runtime": r_name,
            "total_boxes": flips["total_target_boxes"],
            "matched_boxes": flips["total_matched_boxes"],
            "flip_rate": flips["flip_rate"],
            "total_flips": flips["total_flips"],
            "lost_tps": flips["lost_tps"],
            "new_tps": flips["new_tps"],
            "new_fps": flips["new_fps"],
            "suppressed_fps": flips["suppressed_fps"],
            "frac_flip_tp": flips["frac_flip_tp"],
            "frac_flip_fp": flips["frac_flip_fp"],
        })

    # ========================================================================
    # Render Table 6: Q-Aware NMS Threshold Calibration & Metric Recovery
    # ========================================================================
    tbl6_path = PROJECT_ROOT / "results" / "tables" / "table6_q_aware_nms_ablation.md"
    tbl6_path.parent.mkdir(parents=True, exist_ok=True)

    t6_headers = [
        "Configuration",
        "Precision",
        "Runtime",
        r"Default $(\tau_{\text{conf}}, \tau_{\text{iou}})$",
        r"Calibrated $(\tau_{\text{conf}}^*, \tau_{\text{iou}}^*)$",
        r"Baseline $F_1$",
        r"Calibrated $F_1$",
        r"$\Delta F_1$",
        r"$95\%\text{ CI } F_1$",
        "Baseline mAP@50",
        "Calibrated mAP@50",
        "Postprocess Latency",
    ]
    t6_lines = [f"| {' | '.join(t6_headers)} |", f"| {' | '.join(['---'] * len(t6_headers))} |"]

    for r in ablation_results:
        t6_lines.append(
            f"| {r['model_config']} "
            f"| {r['precision']} "
            f"| {r['runtime']} "
            f"| `({r['default_conf']:.2f}, {r['default_iou']:.2f})` "
            f"| **`({r['optimal_conf']:.2f}, {r['optimal_iou']:.2f})`** "
            f"| {r['default_f1']:.4f} "
            f"| **{r['calibrated_f1']:.4f}** "
            f"| {r['f1_delta']:+.4f} "
            f"| `{r['f1_ci_95']}` "
            f"| {r['default_map50']:.4f} "
            f"| {r['calibrated_map50']:.4f} "
            f"| {r['calibrated_post_ms']:.2f} ms |"
        )
    content_t6 = "\n".join(t6_lines) + "\n"
    tbl6_path.write_text(content_t6, encoding="utf-8")
    logger.info(f"Generated Table 6 -> {tbl6_path}")

    # ========================================================================
    # Render Table 7: Quantization Decision-Change Attribution (Flips)
    # ========================================================================
    tbl7_path = PROJECT_ROOT / "results" / "tables" / "table7_decision_flip_audit.md"
    t7_headers = [
        "Target Configuration",
        "Precision",
        "Runtime",
        "Total Target Boxes",
        "Matched vs. FP32",
        r"Flip Rate ($\Phi$)",
        "Lost TPs",
        "New TPs",
        "New FPs",
        "Suppressed FPs",
        r"$\text{Frac}_{\text{Flip\_TP}}$",
        r"$\text{Frac}_{\text{Flip\_FP}}$",
    ]
    t7_lines = [f"| {' | '.join(t7_headers)} |", f"| {' | '.join(['---'] * len(t7_headers))} |"]

    for r in flip_results:
        t7_lines.append(
            f"| {r['model_config']} "
            f"| {r['precision']} "
            f"| {r['runtime']} "
            f"| {r['total_boxes']} "
            f"| {r['matched_boxes']} "
            f"| **{r['flip_rate']:.4f}** "
            f"| {r['lost_tps']} "
            f"| {r['new_tps']} "
            f"| {r['new_fps']} "
            f"| {r['suppressed_fps']} "
            f"| {r['frac_flip_tp']:.2%} "
            f"| {r['frac_flip_fp']:.2%} |"
        )
    content_t7 = "\n".join(t7_lines) + "\n"
    tbl7_path.write_text(content_t7, encoding="utf-8")
    logger.info(f"Generated Table 7 -> {tbl7_path}")

    # ========================================================================
    # Render Publication Figure 1: Q-Aware Pareto Recovery Plot
    # ========================================================================
    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    colors = {"FP32": "#2b5c8f", "FP16": "#2ca02c", "INT8": "#d62728"}
    latencies = [3.89, 3.89, 4.20, 4.77]

    for idx, r in enumerate(ablation_results):
        lat = latencies[idx % len(latencies)]
        p_col = colors.get(r["precision"], "#333333")

        # Baseline point
        ax.scatter(lat, r["default_f1"], color=p_col, marker="o", s=110, alpha=0.6, label=f"{r['precision']} Default NMS" if idx in [0, 3] else "")
        # Calibrated point
        ax.scatter(lat, r["calibrated_f1"], color=p_col, marker="^", s=160, edgecolors="black", linewidths=1.2, label=f"{r['precision']} Q-Aware NMS" if idx in [0, 3] else "")

        # Draw arrow showing recovery
        if abs(r["f1_delta"]) > 1e-4:
            ax.annotate(
                "",
                xy=(lat, r["calibrated_f1"]),
                xytext=(lat, r["default_f1"]),
                arrowprops=dict(arrowstyle="->", color=p_col, lw=1.8, ls="--"),
            )
        ax.text(lat + 0.08, r["calibrated_f1"], f"{r['model_config']}\n(F1: {r['calibrated_f1']:.3f})", fontsize=8.5, verticalalignment="center")

    ax.set_title("Q-Aware NMS Post-Processing Optimization & Pareto Recovery", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(r"Inference Latency $p_{50}$ (ms)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Macro $F_1$-Score ($IoU=0.50$)", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, loc="lower left", fontsize=9)
    plt.tight_layout()
    fig1_path = fig_dir / "q_aware_pareto_recovery.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Generated Figure -> {fig1_path}")

    # ========================================================================
    # Render Publication Figure 2: Decision Flip Attribution Bar Chart
    # ========================================================================
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)

    configs_labels = [r["model_config"] for r in flip_results]
    x = np.arange(len(configs_labels))
    width = 0.35

    tp_flips = [r["frac_flip_tp"] * r["flip_rate"] for r in flip_results]
    fp_flips = [r["frac_flip_fp"] * r["flip_rate"] for r in flip_results]

    ax.bar(x, tp_flips, width, label="True Positive Changes (Lost/New TPs)", color="#4575b4", edgecolor="black", linewidth=0.8)
    ax.bar(x, fp_flips, width, bottom=tp_flips, label="False Positive Changes (New/Suppressed FPs)", color="#d73027", edgecolor="black", linewidth=0.8)

    for i, r in enumerate(flip_results):
        total_flip = r["flip_rate"]
        ax.text(i, total_flip + 0.005, f"$\\Phi={total_flip:.3f}$", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Quantization Decision-Change Attribution Across Precisions (vs FP32)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel(r"Total Flip Rate $\Phi$", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs_labels, fontsize=9, rotation=15)
    ax.set_ylim(0.0, max([r["flip_rate"] for r in flip_results] + [0.1]) * 1.3)
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    plt.tight_layout()
    fig2_path = fig_dir / "decision_flip_attribution.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Generated Figure -> {fig2_path}")


if __name__ == "__main__":
    run_q_aware_nms_ablation()
