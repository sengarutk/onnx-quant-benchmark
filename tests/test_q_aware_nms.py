"""Unit tests for Q-Aware NMS calibration, F1 accounting and the ablation experiment."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.yolo_adapter import YOLOAdapter
from src.quantization.q_aware_nms import apply_q_aware_nms, calibrate_q_aware_nms, compute_f1_score, detection_counts
from tests.conftest import requires_artifacts


def _raw_with_boxes(boxes, scores):
    """Builds a decoded [6, 8400] output containing the given (cx, cy, w, h) boxes with class-0 probabilities."""
    raw = np.full((6, 8400), -12.0, dtype=np.float32)
    for i, ((cx, cy, w, h), s) in enumerate(zip(boxes, scores)):
        raw[:4, i] = [cx, cy, w, h]
        raw[4, i] = np.log(s / (1 - s))
    return raw


def _cache(raw, gt):
    return {"raw": raw, "orig_shape": (640, 640), "ratio": (1.0, 1.0), "pad": (0.0, 0.0), "gt": gt}


class TestQAwareNMS:
    def test_f1_and_counts(self) -> None:
        gt = [[{"bbox": [100, 100, 200, 200], "class_id": 0}]]
        hit = [[{"bbox": [100, 100, 200, 200], "score": 0.9, "class_id": 0}]]
        assert np.isclose(compute_f1_score(hit, gt), 1.0)
        assert compute_f1_score([[]], gt) == 0.0
        assert compute_f1_score([[]], [[]]) == 0.0
        assert compute_f1_score(hit, [[]]) == 0.0
        wrong_cls = [[{"bbox": [100, 100, 200, 200], "score": 0.9, "class_id": 1}]]
        assert detection_counts(wrong_cls, gt) == (0, 1, 1)
        # duplicates: second match of the same GT is a false positive
        dup = [[{"bbox": [100, 100, 200, 200], "score": 0.9, "class_id": 0}, {"bbox": [101, 101, 201, 201], "score": 0.8, "class_id": 0}]]
        assert detection_counts(dup, gt) == (1, 1, 0)

    def test_calibration_finds_better_threshold(self) -> None:
        """A low-confidence true box and a confident duplicate: raising IoU/lowering conf changes F1 measurably."""
        adapter = YOLOAdapter()
        gt = [{"bbox": [100, 100, 200, 200], "class_id": 0}, {"bbox": [400, 400, 480, 480], "class_id": 0}]
        raw = _raw_with_boxes([(150, 150, 100, 100), (440, 440, 80, 80)], [0.9, 0.18])
        res = calibrate_q_aware_nms([_cache(raw, gt)] * 3, adapter)
        assert res["baseline_f1"] < res["optimal_f1"] == 1.0
        assert res["optimal_conf"] <= 0.15
        assert res["grid_evaluated"] == 18 * 14

    def test_ties_keep_default_operating_point(self) -> None:
        adapter = YOLOAdapter()
        gt = [{"bbox": [100, 100, 200, 200], "class_id": 0}]
        raw = _raw_with_boxes([(150, 150, 100, 100)], [0.95])
        res = calibrate_q_aware_nms([_cache(raw, gt)], adapter)
        assert res["f1_delta"] == 0.0 and (res["optimal_conf"], res["optimal_iou"]) == (0.25, 0.45)

    def test_calibration_requires_labels(self) -> None:
        adapter = YOLOAdapter()
        with pytest.raises(ValueError, match="No valid calibration samples"):
            calibrate_q_aware_nms([], adapter)
        with pytest.raises(ValueError, match="labeled calibration split"):
            calibrate_q_aware_nms([_cache(np.zeros((6, 8400), np.float32), [])], adapter)

    def test_apply_q_aware_nms_all_formats(self) -> None:
        policy = {"optimal_conf": 0.50, "optimal_iou": 0.45}
        preds = [[{"bbox": [10, 10, 50, 50], "score": 0.30, "class_id": 0}, {"bbox": [20, 20, 80, 80], "score": 0.85, "class_id": 0}]]
        assert [p["score"] for p in apply_q_aware_nms(preds, policy)[0]] == [0.85]
        assert len(apply_q_aware_nms(preds[0], policy)) == 1
        raw = _raw_with_boxes([(150, 150, 100, 100), (300, 300, 50, 50)], [0.9, 0.3])
        out = apply_q_aware_nms(raw, policy, [(640, 640)], [(1.0, 1.0)], [(0.0, 0.0)], YOLOAdapter())
        assert len(out) == 1
        out_list = apply_q_aware_nms([torch.from_numpy(raw)], policy, yolo_adapter=YOLOAdapter())
        assert len(out_list) == 1 and len(out_list[0]) == 1

    @requires_artifacts
    def test_run_q_aware_ablation_end_to_end(self, tmp_path: Path) -> None:
        from src.experiments.run_q_aware_ablation import run_q_aware_nms_ablation
        out = run_q_aware_nms_ablation(tmp_path)
        saved = json.loads((tmp_path / "q_aware_nms.json").read_text())
        assert saved["test_images"] == out["test_images"] > 0
        for row in out["ablation"]:
            lo, hi = row["delta_f1_ci95"]
            assert lo <= row["delta_f1"] <= hi
            assert 0.0 <= row["wilcoxon_errors"]["p_holm"] <= 1.0
        assert out["flips"][0]["default_thresholds"]["flip_rate"] == 0.0  # reference vs itself
        for name in ("table6_q_aware_nms_ablation.md", "table7_decision_flip_audit.md"):
            assert (tmp_path / "tables" / name).is_file()
        for name in ("q_aware_pareto_recovery.png", "decision_flip_attribution.png"):
            assert (tmp_path / "figures" / name).is_file()
