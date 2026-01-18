import os
from onnxruntime.quantization import quantize_dynamic, QuantType

FP32_PATH = "models/onnx_fp32/model.onnx"
OUT_DIR = "models/onnx_dynamic"


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "model_dynamic_int8.onnx")

    quantize_dynamic(
        model_input=FP32_PATH,
        model_output=out_path,
        weight_type=QuantType.QInt8,
    )

    print(f"Exported Dynamic INT8 ONNX -> {out_path}")
