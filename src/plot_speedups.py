import os
import re
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = "results/runs.csv"
OUT_DIR = "results/plots"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_format(model_name: str) -> str:
    """
    Extract a clean format label from model string.
    Supports patterns like:
      - fp32 / fp16
      - int8 / dynamic_int8
      - quant_int8
      - etc.
    If nothing found, returns model itself.
    """
    s = model_name.lower()

    # common cases first
    for key in ["dynamic_int8", "int8", "fp16", "fp32"]:
        if key in s:
            return key

    # generic fallbacks
    m = re.search(r"(fp\d+|int\d+)", s)
    if m:
        return m.group(1)

    return model_name


def main():
    ensure_dir(OUT_DIR)

    df = pd.read_csv(RESULTS_CSV)

    # Rename latency column to a consistent name
    if "lat_mean_ms" not in df.columns:
        raise ValueError("Expected column lat_mean_ms in results/runs.csv")

    df["format"] = df["model"].apply(parse_format)
    df["mean_ms"] = df["lat_mean_ms"]

    # ---- Baselines (fp32 per provider) ----
    def get_baseline(provider):
        base = df[(df["provider"] == provider) & (df["format"] == "fp32")]
        if len(base) != 1:
            raise ValueError(
                f"Expected exactly 1 baseline row for provider={provider} and format=fp32. "
                f"Found {len(base)}.\nRows:\n{base}"
            )
        return float(base["mean_ms"].iloc[0])

    base_cpu = get_baseline("cpu")
    base_cuda = get_baseline("cuda")

    # Tensorrt may not be installed correctly — only compute if fp32 exists there
    base_trt = None
    if ((df["provider"] == "tensorrt") & (df["format"] == "fp32")).any():
        base_trt = get_baseline("tensorrt")

    # ---- Compute speedups ----
    df["speedup_vs_fp32_cpu"] = base_cpu / df["mean_ms"]
    df["speedup_vs_fp32_cuda"] = base_cuda / df["mean_ms"]
    if base_trt is not None:
        df["speedup_vs_fp32_tensorrt"] = base_trt / df["mean_ms"]

    # Save enriched CSV
    out_csv = "results/runs_with_speedups.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved -> {out_csv}")

    # -----------------------------
    # Plot Speedups per provider
    # -----------------------------
    providers = ["cpu", "cuda", "tensorrt"]

    for p in providers:
        d = df[df["provider"] == p].copy()
        if len(d) == 0:
            continue

        # pick correct baseline column
        if p == "cpu":
            col = "speedup_vs_fp32_cpu"
        elif p == "cuda":
            col = "speedup_vs_fp32_cuda"
        else:
            if base_trt is None:
                print("[SKIP] TensorRT baseline not available, skipping TRT speedup plot.")
                continue
            col = "speedup_vs_fp32_tensorrt"

        # sort formats in nice order
        order = ["fp32", "fp16", "dynamic_int8", "int8"]
        d["order"] = d["format"].apply(lambda x: order.index(x) if x in order else 999)
        d = d.sort_values("order")

        plt.figure()
        plt.bar(d["format"], d[col])
        plt.title(f"Speedup vs FP32 Baseline ({p.upper()})")
        plt.xlabel("Format")
        plt.ylabel("Speedup (×)")
        plt.xticks(rotation=20)
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"speedup_{p}.png")
        plt.savefig(out_path, dpi=200)
        print(f"Saved -> {out_path}")

    print("\nSpeedup plots generated successfully.")


if __name__ == "__main__":
    main()
