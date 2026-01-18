import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "distilbert-base-uncased"
OUT_DIR_FP32 = "models/onnx_fp32"
OUT_DIR_FP16 = "models/onnx_fp16"


def export(fp16: bool):
    os.makedirs(OUT_DIR_FP16 if fp16 else OUT_DIR_FP32, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    if fp16:
        model = model.half()

    dummy = tokenizer(
        "This is a test sentence for ONNX export.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    out_path = os.path.join(OUT_DIR_FP16 if fp16 else OUT_DIR_FP32, "model.onnx")

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"Exported {'FP16' if fp16 else 'FP32'} ONNX -> {out_path}")


if __name__ == "__main__":
    export(fp16=False)
    export(fp16=True)
