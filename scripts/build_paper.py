#!/usr/bin/env python3
"""
Builds the paper from measured results only.

1. Writes paper/generated/results.tex: one macro per measured value, used in the text as \\res{key}
   (e.g. \\res{det-ort-cpu-int8-p50}), so every number in the manuscript is traceable to results/.
2. Writes the LaTeX tables (paper/generated/table_*.tex).
3. Copies the figures from results/figures/ to paper/figures/.
4. Compiles paper/main.pdf when pdflatex + bibtex are available.
5. Packs overleaf_paper.zip (main.tex, references.bib, generated/, figures/) with a flat root.
"""

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
GEN = PAPER / "generated"
RES = ROOT / "results"

FIGURES = [
    "pareto_quality_vs_latency.png", "speedup_comparison.png", "tail_latency_p50_p95.png",
    "latency_breakdown_stacked.png", "q_aware_pareto_recovery.png", "decision_flip_attribution.png",
    "scalability_batch_resolution.png", "memory_vs_footprint.png", "stability_variance_trends.png",
]


def _tex(s: str) -> str:
    return str(s).replace(" | ", ", ").replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def _sgn(x: float, digits: int = 3) -> str:
    """Signed fixed-point value without a negative zero (e.g. -0.0000 -> +0.000)."""
    v = round(float(x), digits) + 0.0
    return f"{v:+.{digits}f}"


def _p(x: float) -> str:
    """p-value for math mode: 0.35, 0.012 or 9.5\\times 10^{-34}."""
    x = float(x)
    if x >= 1e-3:
        return f"{x:.2g}"
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}\\times 10^{{{int(exp)}}}"


def _slug(row: pd.Series) -> str:
    m = "det" if row["model"] == "yolo_nano" else "ae"
    rt = {"PyTorch": "pt", "ORT_CPU": "ort", "ORT_CUDA": "ort", "TensorRT": "trt"}[row["runtime"]]
    dev = "gpu" if any(k in str(row["provider"]).upper() for k in ("CUDA", "TENSORRT")) else "cpu"
    return f"{m}-{rt}-{dev}-{row['precision']}"


def _cfg_name(row: pd.Series) -> str:
    rt = {"PyTorch": "PyTorch", "ORT_CPU": "ORT", "ORT_CUDA": "ORT", "TensorRT": "TensorRT"}[row["runtime"]]
    dev = "GPU" if any(k in str(row["provider"]).upper() for k in ("CUDA", "TENSORRT")) else "CPU"
    return f"{rt} {dev}"


def build_macros(df: pd.DataFrame, runs_meta: Dict, qa: Dict, sweep: pd.DataFrame, quant: Dict, training: Dict) -> Dict[str, str]:
    v: Dict[str, str] = {}
    for _, r in df.iterrows():
        k = _slug(r)
        v.update({
            f"{k}-p50": f"{r['p50_model_ms']:.2f}", f"{k}-p50lo": f"{r['p50_model_ci_low_ms']:.2f}",
            f"{k}-p50hi": f"{r['p50_model_ci_high_ms']:.2f}", f"{k}-p95": f"{r['p95_model_ms']:.2f}",
            f"{k}-p99": f"{r['p99_model_ms']:.2f}", f"{k}-e2e": f"{r['p50_e2e_ms']:.2f}",
            f"{k}-fps": f"{r['model_throughput_fps']:.0f}", f"{k}-cv": f"{r['cv_p50']:.3f}",
            f"{k}-relerr": f"{r['max_rel_error']:.1e}", f"{k}-cos": f"{r['cosine_similarity']:.5f}",
            f"{k}-q": f"{r['quality_value']:.3f}", f"{k}-qdelta": _sgn(r['quality_delta']),
            f"{k}-size": f"{r['model_size_mb']:.2f}", f"{k}-speedup": f"{r['speedup_model']:.2f}",
            f"{k}-speedupfp": f"{r['speedup_vs_runtime_fp32']:.2f}", f"{k}-vram": f"{r['peak_vram_mb']:.1f}",
            f"{k}-pre": f"{r['stage_preprocess_ms']:.2f}", f"{k}-inf": f"{r['stage_inference_ms']:.2f}",
            f"{k}-post": f"{r['stage_postprocess_ms']:.2f}",
            f"{k}-infshare": f"{100 * r['stage_inference_ms'] / max(r['p50_e2e_ms'], 1e-9):.0f}",
            f"{k}-stable": "yes" if r["stable"] else "no",
            f"{k}-sizered": "n/a" if pd.isna(r["storage_compression_pct"]) else f"{r['storage_compression_pct']:.1f}",
        })
    def _sci(x: float) -> str:
        mant, exp = f"{x:.1e}".split("e")
        return f"{mant}\\cdot 10^{{{int(exp)}}}"

    v["fp32-maxrelerr"] = _sci(df[df["precision"] == "fp32"]["max_rel_error"].max())
    v["fp16-maxrelerr"] = _sci(df[df["precision"] == "fp16"]["max_rel_error"].max())
    gpu = df[df["provider"].str.contains("CUDA")]
    ae = gpu[gpu["model"] == "industrial_autoencoder"].set_index("precision")["p50_model_ms"]
    v["ae-gpu-fp16-gain"] = f"{ae['fp32'] / ae['fp16']:.1f}"
    rv = json.loads((RES / "runtime_validation.json").read_text())
    tf32 = [r for r in rv if r["runtime"].startswith("PyTorch CUDA (TF32")]
    v["tf32-relerr"] = _sci(tf32[0]["max_rel_error"]) if tf32 else "n/a"
    v["n-configs"] = str(len(df))
    v["n-stable"] = str(int(df["stable"].sum()))
    v["n-gates-pass"] = str(int((df["numerical_gate"] & df["quality_passed"]).sum()))
    v["session-id"] = _tex(runs_meta["session"]["session_id"])
    env = runs_meta["runs"][0]["environment_manifest"]
    v["env-cpu"] = _tex(env["cpu_info"])
    v["env-gpu"] = _tex(env.get("gpu_name") or "none")
    v["env-os"] = _tex(env["os_info"])
    v["env-torch"] = _tex(env["torch_version"])
    v["env-ort"] = _tex(env["onnxruntime_version"].split(" ")[0])
    v["env-python"] = _tex(env["python_version"].split(" ")[0])
    v["env-cuda"] = _tex(env.get("cuda_runtime_version") or "n/a")
    v["env-driver"] = _tex(env.get("nvidia_driver_version") or "n/a")
    proto = runs_meta["session"]["protocol"]
    for key in ("warmup_iterations", "timed_iterations", "stability_sessions", "e2e_timed_iterations"):
        v[f"proto-{key.replace('_', '')}"] = str(proto[key])

    for i, row in enumerate(qa["ablation"]):
        k = f"qa{i}"
        v.update({
            f"{k}-name": _tex(row["config"]), f"{k}-f1def": f"{row['default']['f1']:.3f}",
            f"{k}-f1cal": f"{row['calibrated']['f1']:.3f}", f"{k}-df1": _sgn(row['delta_f1']),
            f"{k}-df1lo": _sgn(row['delta_f1_ci95'][0]), f"{k}-df1hi": _sgn(row['delta_f1_ci95'][1]),
            f"{k}-conf": f"{row['calibrated']['conf']:.2f}", f"{k}-iou": f"{row['calibrated']['iou']:.2f}",
            f"{k}-pholm": _p(row['wilcoxon_errors']['p_holm']),
            f"{k}-postdef": f"{row['default']['postprocess_ms']:.2f}", f"{k}-postcal": f"{row['calibrated']['postprocess_ms']:.2f}",
            f"{k}-f1tr": f"{row['fp32_transfer']['f1']:.3f}", f"{k}-dtr": _sgn(row['delta_f1_vs_transfer']),
            f"{k}-dtrlo": _sgn(row['delta_f1_vs_transfer_ci95'][0]), f"{k}-dtrhi": _sgn(row['delta_f1_vs_transfer_ci95'][1]),
            f"{k}-ptrholm": _p(row['wilcoxon_vs_transfer']['p_holm']),
        })
    for i, row in enumerate(qa["flips"]):
        for key, tag in (("default_thresholds", "def"), ("calibrated_thresholds", "cal")):
            f = row[key]
            v[f"fl{i}-{tag}-rate"] = f"{f['flip_rate']:.3f}"
            v[f"fl{i}-{tag}-flips"] = str(f["total_flips"])
            v[f"fl{i}-{tag}-fpshare"] = f"{100 * f['frac_flip_fp']:.0f}"
    v["qa-test-images"] = str(qa["test_images"])
    v["qa-test-boxes"] = str(qa["test_boxes"])
    v["qa-calib-images"] = str(qa["calibration_images"])

    for _, r in sweep.iterrows():
        v[f"sw-{int(r['resolution'])}-{int(r['batch_size'])}-p50"] = f"{r['p50_ms']:.2f}"
        v[f"sw-{int(r['resolution'])}-{int(r['batch_size'])}-fps"] = f"{r['throughput_fps']:.0f}"
    v["sw-device"] = _tex(sweep["device"].iloc[0])

    for m, tag in (("yolo_nano", "det"), ("industrial_autoencoder", "ae")):
        e = quant["models"][m]
        v[f"{tag}-fp32-mb"] = f"{e['fp32_size_mb']:.2f}"
        for prec in ("fp16", "int8"):
            v[f"{tag}-{prec}-mb"] = f"{e[prec]['size_mb']:.2f}"
    for tag, t in training.items():
        v[f"{tag}-params"] = f"{t['parameters'] / 1e6:.2f}"
        v[f"{tag}-trainsteps"] = str(t["steps"])
        v[f"{tag}-trainmin"] = f"{t['train_seconds'] / 60:.0f}"
        v[f"{tag}-traindevice"] = _tex(t["device"])
        for mk, mv in t["test_metrics"].items():
            v[f"{tag}-{mk.replace('_', '').lower()}"] = f"{mv:.3f}"
        if "scoring_selection" in t:
            sel = t["scoring_selection"]["selected"]
            v["ae-sigma"] = f"{sel['blur_sigma']:g}"
            v["ae-topk"] = f"{100 * sel['top_k_ratio']:g}"
    return v


def write_macros(v: Dict[str, str]) -> None:
    lines = ["% AUTO-GENERATED by scripts/build_paper.py from results/ -- do not edit by hand.",
             "\\makeatletter", "\\newcommand{\\res}[1]{\\@ifundefined{res@#1}{\\typeout{RES-MISSING: #1}\\textbf{??#1??}}{\\csname res@#1\\endcsname}}"]
    lines += [f"\\expandafter\\def\\csname res@{k}\\endcsname{{{val}}}" for k, val in sorted(v.items())]
    lines.append("\\makeatother")
    (GEN / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(df: pd.DataFrame, qa: Dict) -> None:
    rows: List[str] = []
    for _, r in df.iterrows():
        model = "Detector" if r["model"] == "yolo_nano" else "Autoencoder"
        rows.append(
            f"{model} & {_cfg_name(r)} & {str(r['precision']).upper()} & {r['p50_model_ms']:.2f} "
            f"[{r['p50_model_ci_low_ms']:.2f}, {r['p50_model_ci_high_ms']:.2f}] & {r['p95_model_ms']:.2f} & "
            f"{r['p50_e2e_ms']:.2f} & {r['cv_p50']:.3f} & {r['max_rel_error']:.1e} & "
            f"{r['quality_value']:.3f} ({_sgn(r['quality_delta'])}) & {r['model_size_mb']:.2f} \\\\"
        )
    (GEN / "table_main.tex").write_text(
        "\\begin{tabular}{lllrrrrrrr}\n\\toprule\n"
        "\\textbf{Model} & \\textbf{Runtime} & \\textbf{Prec.} & \\textbf{p50 [95\\% CI] (ms)} & \\textbf{p95} & "
        "\\textbf{E2E p50} & \\textbf{CV} & \\textbf{Rel. err.} & \\textbf{Quality ($\\Delta$)} & \\textbf{MB} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    qrows = []
    for r in qa["ablation"]:
        qrows.append(
            f"{_tex(r['config'])} & ({r['calibrated']['conf']:.2f}, {r['calibrated']['iou']:.2f}) & {r['default']['f1']:.3f} & "
            f"{r['fp32_transfer']['f1']:.3f} & {r['calibrated']['f1']:.3f} & "
            f"{_sgn(r['delta_f1_vs_transfer'])} [{_sgn(r['delta_f1_vs_transfer_ci95'][0])}, {_sgn(r['delta_f1_vs_transfer_ci95'][1])}] & "
            f"${_p(r['wilcoxon_vs_transfer']['p_holm'])}$ \\\\"
        )
    (GEN / "table_qaware.tex").write_text(
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "\\textbf{Configuration} & $(\\tau_c^*, \\tau_i^*)$ & \\textbf{F1 def.} & \\textbf{F1 FP32-thr.} & \\textbf{F1 own} & "
        "\\textbf{$\\Delta$ own vs FP32-thr. [95\\% CI]} & \\textbf{$p_{\\text{Holm}}$} \\\\\n\\midrule\n"
        + "\n".join(qrows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    frows = []
    for f in qa["flips"][1:]:
        for key, label in (("default_thresholds", "default"), ("calibrated_thresholds", "calibrated")):
            x = f[key]
            frows.append(f"{_tex(f['config'])} & {label} & {x['total_matched_boxes']} & {x['flip_rate']:.3f} & {x['lost_tps']} & "
                         f"{x['new_tps']} & {x['new_fps']} & {x['suppressed_fps']} \\\\")
    (GEN / "table_flips.tex").write_text(
        "\\begin{tabular}{llrrrrrr}\n\\toprule\n"
        "\\textbf{Target} & \\textbf{Thr.} & \\textbf{Matched} & $\\Phi$ & \\textbf{Lost TP} & \\textbf{New TP} & "
        "\\textbf{New FP} & \\textbf{Supp. FP} \\\\\n\\midrule\n" + "\n".join(frows) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8")


def compile_pdf() -> bool:
    if not shutil.which("pdflatex") or not shutil.which("bibtex"):
        print("pdflatex/bibtex not found; skipping PDF build")
        return False
    for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]):
        res = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-3000:])
            raise RuntimeError(f"{' '.join(cmd)} failed")
    log = (PAPER / "main.log").read_text(errors="ignore")
    problems = [l for l in log.splitlines() if "undefined" in l.lower() and ("reference" in l.lower() or "citation" in l.lower())]
    problems += [l for l in log.splitlines() if l.startswith("RES-MISSING")]
    if problems:
        raise RuntimeError(f"Unresolved references / result macros in the PDF: {problems[:10]}")
    return True


def pack_zip() -> Path:
    zip_path = ROOT / "overleaf_paper.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(PAPER / "main.tex", "main.tex")
        zf.write(PAPER / "references.bib", "references.bib")
        for p in sorted(GEN.glob("*.tex")):
            zf.write(p, f"generated/{p.name}")
        for name in FIGURES:
            zf.write(PAPER / "figures" / name, f"figures/{name}")
    return zip_path


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "runs.csv")
    runs_meta = json.loads((RES / "runs.json").read_text())
    qa = json.loads((RES / "q_aware_nms.json").read_text())
    sweep = pd.read_csv(RES / "scalability_sweep.csv")
    quant = json.loads((RES / "quantization_validation.json").read_text())
    training = {"det": json.loads((RES / "training" / "yolo_nano_training.json").read_text()),
                "ae": json.loads((RES / "training" / "industrial_autoencoder_training.json").read_text())}

    write_macros(build_macros(df, runs_meta, qa, sweep, quant, training))
    write_tables(df, qa)
    for name in FIGURES:
        shutil.copy2(RES / "figures" / name, PAPER / "figures" / name)
    pdf = compile_pdf()
    z = pack_zip()
    print(f"Paper assets generated; PDF {'built' if pdf else 'skipped'}; bundle -> {z} ({z.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
