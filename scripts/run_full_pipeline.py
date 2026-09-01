"""
Master End-to-End Benchmark Pipeline Orchestrator (v1.1).
Executes the complete 11-stage benchmark lifecycle with unified error handling,
execution timers, and ANSI progress logging.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ANSI Color codes
BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def run_stage(step_num: int, total_steps: int, title: str, cmd: list, dry_run: bool = False) -> float:
    """Executes a pipeline stage with high-precision timing and formatted terminal output."""
    prefix = f"{BOLD}[{step_num}/{total_steps}]{RESET}"
    print(f"\n{prefix} {BLUE}Starting:{RESET} {BOLD}{title}{RESET}")
    print(f"      {YELLOW}Command:{RESET} {' '.join(cmd)}")

    if dry_run:
        print(f"      {GREEN}[DRY-RUN SKIPPED]{RESET}")
        return 0.0

    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    t1 = time.perf_counter()
    elapsed = t1 - t0

    if res.returncode != 0:
        print(f"\n{prefix} {RED}[FAILED]{RESET} {title} (Exit Code: {res.returncode}) in {elapsed:.2f}s")
        sys.exit(res.returncode)

    print(f"      {GREEN}[SUCCESS]{RESET} Completed in {elapsed:.2f}s")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Master End-to-End Benchmark Pipeline Orchestrator (v1.1)")
    parser.add_argument("--dry-run", action="store_true", help="Print pipeline stages without executing subprocesses")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip final smoke test")
    args = parser.parse_args()

    # Determine Python binary
    py_bin = sys.executable

    stages = [
        ("Environment & Hardware Introspection", [py_bin, "scripts/generate_env_manifest.py"]),
        ("Synthetic Evaluation Dataset Generation", [py_bin, "scripts/prepare_sample_data.py"]),
        ("PyTorch Reference Baselines & Hashing", [py_bin, "scripts/run_pytorch_baselines.py"]),
        ("Canonical ONNX Export, Validation & Simplification", [py_bin, "scripts/export_and_validate.py"]),
        ("Disjoint Calibration Dataset Synthesis", [py_bin, "scripts/prepare_calibration_data.py"]),
        ("FP16 Conversion, Static INT8 PTQ & Quality Gating", [py_bin, "scripts/quantize_and_validate.py"]),
        ("Multi-Backend Runtime Execution & MAD Stability Benchmarking", [py_bin, "scripts/benchmark_all.py"]),
        ("Result Aggregation, Pareto Analysis, Tables & Plots", [py_bin, "scripts/generate_report.py"]),
        ("Q-Aware NMS Calibration & Decision Flip Audit", [py_bin, "src/experiments/run_q_aware_ablation.py"]),
        ("Scalability Sweeps Across Batch Sizes & Resolutions", [py_bin, "src/experiments/run_scalability_sweep.py"]),
    ]

    if not args.skip_smoke_test:
        stages.append(("Automated Smoke Pipeline Verification", ["bash", "scripts/smoke_test.sh"]))

    total_steps = len(stages)
    print(f"{BOLD}{'=' * 75}{RESET}")
    print(f"{BOLD}ONNX Edge Inference Benchmark — Master End-to-End Pipeline ({total_steps} Stages){RESET}")
    print(f"{BOLD}{'=' * 75}{RESET}")

    total_t0 = time.perf_counter()
    stage_times = []

    for idx, (title, cmd) in enumerate(stages, 1):
        elapsed = run_stage(idx, total_steps, title, cmd, dry_run=args.dry_run)
        stage_times.append((title, elapsed))

    total_elapsed = time.perf_counter() - total_t0

    print(f"\n{BOLD}{'=' * 75}{RESET}")
    print(f"{GREEN}{BOLD}>>> PIPELINE EXECUTION COMPLETED WITH 100% SUCCESS <<< {RESET}")
    print(f"Total Wall-Clock Time: {BOLD}{total_elapsed:.2f}s{RESET}")
    print(f"{BOLD}{'=' * 75}{RESET}")
    for title, el in stage_times:
        print(f"  • {title:<60} {el:>7.2f}s")
    print(f"{BOLD}{'=' * 75}{RESET}\n")


if __name__ == "__main__":
    main()
