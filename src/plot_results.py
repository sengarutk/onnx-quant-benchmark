import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results/runs.csv"
OUT_DIR = "results/plots"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(["provider", "model"])

    # bar plot of mean latency
    for provider in df["provider"].unique():
        sub = df[df["provider"] == provider]

        plt.figure()
        plt.bar(sub["model"], sub["lat_mean_ms"])
        plt.ylabel("Mean Latency (ms)")
        plt.title(f"ONNX Inference Mean Latency ({provider.upper()})")
        plt.xticks(rotation=20)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"latency_{provider}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print("Saved plot:", out_path)


if __name__ == "__main__":
    main()
