#!/usr/bin/env python3
"""
End-to-end pipeline orchestrator: data -> training -> baselines -> export -> quantization -> runtime validation ->
benchmark session -> report -> Q-Aware NMS ablation -> scalability sweep -> paper bundle.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOLD, GREEN, BLUE, YELLOW, RED, RESET = "\033[1m", "\033[92m", "\033[94m", "\033[93m", "\033[91m", "\033[0m"


def run_stage(step: int, total: int, title: str, cmd: list, dry_run: bool = False) -> float:
    prefix = f"{BOLD}[{step}/{total}]{RESET}"
    print(f"\n{prefix} {BLUE}{title}{RESET}\n      {YELLOW}$ {' '.join(cmd)}{RESET}")
    if dry_run:
        return 0.0
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"{prefix} {RED}FAILED{RESET} (exit {res.returncode}) after {elapsed:.1f}s")
        sys.exit(res.returncode)
    print(f"      {GREEN}done{RESET} in {elapsed:.1f}s")
    return elapsed


def build_stages(py: str, skip_training: bool, skip_paper: bool) -> list:
    stages = [
        ("Environment manifest & docs/hardware.md", [py, "scripts/generate_env_manifest.py"]),
        ("Synthetic test split", [py, "scripts/prepare_sample_data.py"]),
        ("Synthetic calibration split (+ leakage check)", [py, "scripts/prepare_calibration_data.py"]),
    ]
    if not skip_training:
        stages.append(("Train detector and autoencoder", [py, "scripts/train_models.py"]))
    stages += [
        ("PyTorch FP32 reference metrics", [py, "scripts/run_pytorch_baselines.py"]),
        ("ONNX export, simplification & parity audit", [py, "scripts/export_and_validate.py"]),
        ("FP16 conversion, static INT8 PTQ & quality gates", [py, "scripts/quantize_and_validate.py"]),
        ("Runtime backend validation", [py, "scripts/validate_runtimes.py"]),
        ("Benchmark session (latency, stability, parity, quality)", [py, "scripts/benchmark_all.py"]),
        ("Aggregation, tables, figures, decision matrix", [py, "scripts/generate_report.py"]),
        ("Q-Aware NMS ablation & decision-flip audit", [py, "src/experiments/run_q_aware_ablation.py"]),
        ("Batch x resolution scalability sweep", [py, "src/experiments/run_scalability_sweep.py"]),
        ("Inject result tables into README.md", [py, "scripts/update_readme.py"]),
    ]
    if not skip_paper:
        stages.append(("Paper tables/macros + Overleaf bundle", [py, "scripts/build_paper.py"]))
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the stages without executing them")
    parser.add_argument("--skip-training", action="store_true", help="Reuse existing models/weights/*_trained.pt")
    parser.add_argument("--skip-paper", action="store_true", help="Do not rebuild the paper bundle")
    args = parser.parse_args()

    stages = build_stages(sys.executable, args.skip_training, args.skip_paper)
    print(f"{BOLD}Edge inference benchmark pipeline ({len(stages)} stages){RESET}")
    t0 = time.perf_counter()
    times = [(title, run_stage(i, len(stages), title, cmd, args.dry_run)) for i, (title, cmd) in enumerate(stages, 1)]
    print(f"\n{GREEN}{BOLD}Pipeline finished in {time.perf_counter() - t0:.1f}s{RESET}")
    for title, el in times:
        print(f"  - {title:<60} {el:>7.1f}s")


if __name__ == "__main__":
    main()
