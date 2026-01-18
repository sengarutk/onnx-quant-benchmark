import time
import numpy as np
import onnxruntime as ort


def get_session(model_path: str, provider: str):
    provider_map = {
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    }
    providers = provider_map[provider]

    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opt.intra_op_num_threads = 8

    session = ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)
    return session


def benchmark_session(session, inputs: dict, warmup=10, runs=50):
    # warmup
    for _ in range(warmup):
        _ = session.run(None, inputs)

    latencies = []
    for _ in range(runs):
        t0 = time.time()
        _ = session.run(None, inputs)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies)
    return {
        "lat_mean_ms": float(latencies.mean()),
        "lat_p50_ms": float(np.percentile(latencies, 50)),
        "lat_p90_ms": float(np.percentile(latencies, 90)),
        "lat_p99_ms": float(np.percentile(latencies, 99)),
    }
