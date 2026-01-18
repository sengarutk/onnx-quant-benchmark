import csv
import os
from transformers import AutoTokenizer
from utils import get_session, benchmark_session

MODEL_ID = "distilbert-base-uncased"
RESULTS_PATH = "results/runs.csv"

MODELS = [
    ("fp32", "models/onnx_fp32/model.onnx"),
    ("fp16", "models/onnx_fp16/model.onnx"),
    ("dynamic_int8", "models/onnx_dynamic/model_dynamic_int8.onnx"),
]

PROVIDERS = ["cpu", "cuda", "tensorrt"]


def build_inputs(tokenizer, batch=1, seqlen=128):
    text = ["benchmarking inference"] * batch
    enc = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=seqlen,
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def main():
    os.makedirs("results", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    inputs = build_inputs(tokenizer, batch=1, seqlen=128)

    rows = []
    for model_name, path in MODELS:
        if not os.path.exists(path):
            print(f"missing model: {path}")
            continue

        for provider in PROVIDERS:
            try:
                session = get_session(path, provider)
                stats = benchmark_session(session, inputs, warmup=10, runs=50)

                row = {
                    "model": model_name,
                    "provider": provider,
                    **stats,
                }
                rows.append(row)

                print(f"{model_name:12s} | {provider:8s} | mean={stats['lat_mean_ms']:.2f} ms")

            except Exception as e:
                print(f"FAIL {model_name} | {provider} -> {e}")

    # write CSV
    if rows:
        with open(RESULTS_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n Saved results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
