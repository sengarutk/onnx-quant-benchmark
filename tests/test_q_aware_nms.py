"""
Unit tests for Quantization-Aware NMS Calibration & Vectorized Post-Processing.
"""

from pathlib import Path
import json
import numpy as np
import pytest
import torch

from src.models.yolo_adapter import YOLOAdapter
from src.quantization.q_aware_nms import apply_q_aware_nms, calibrate_q_aware_nms, compute_f1_score
from src.experiments.run_q_aware_ablation import run_q_aware_nms_ablation


class TestQAwareNMS:
    """Test suite asserting Q-Aware NMS mathematical properties and threshold calibration."""

    def test_compute_f1_score_perfect_and_empty(self) -> None:
        """Asserts 1.0 F1 on perfect match, 0.0 F1 on zero predictions, and handles empty sets."""
        gt = [[{"bbox": [100, 100, 200, 200], "class_id": 0}]]
        preds_perfect = [[{"bbox": [100, 100, 200, 200], "score": 0.95, "class_id": 0}]]

        f1_perfect = compute_f1_score(preds_perfect, gt, iou_threshold=0.5)
        assert np.isclose(f1_perfect, 1.0)

        preds_empty = [[]]
        f1_empty = compute_f1_score(preds_empty, gt, iou_threshold=0.5)
        assert np.isclose(f1_empty, 0.0)

        f1_both_empty = compute_f1_score([[]], [[]], iou_threshold=0.5)
        assert np.isclose(f1_both_empty, 0.0)

        # Ground truth empty, preds non-empty
        f1_gt_empty = compute_f1_score(preds_perfect, [[]], iou_threshold=0.5)
        assert np.isclose(f1_gt_empty, 0.0)

        # Class mismatch -> 0 F1
        preds_wrong_class = [[{"bbox": [100, 100, 200, 200], "score": 0.95, "class_id": 1}]]
        f1_wrong_cls = compute_f1_score(preds_wrong_class, gt, iou_threshold=0.5)
        assert np.isclose(f1_wrong_cls, 0.0)

    def test_calibrate_q_aware_nms_synthetic_loader(self) -> None:
        """Verifies 2D grid search threshold optimization on synthetic calibration loader."""
        adapter = YOLOAdapter()

        calib_data = []
        for i in range(5):
            calib_data.append({
                "image_path": f"data/sample_images/detection/images/det_{i:03d}.jpg",
                "ground_truth_boxes": [{"bbox": [80, 120, 160, 180], "class_id": 0}],
            })

        res = calibrate_q_aware_nms(
            model_adapter=adapter,
            calib_loader=calib_data,
            precision="int8",
            runtime="ort_cpu",
            conf_min=0.15,
            conf_max=0.40,
            conf_step=0.05,
            iou_min=0.35,
            iou_max=0.55,
            iou_step=0.05,
        )

        assert "optimal_conf" in res
        assert "optimal_iou" in res
        assert "optimal_f1" in res
        assert "baseline_f1" in res
        assert res["grid_evaluated"] == 6 * 5  # 6 conf * 5 iou = 30
        assert 0.15 <= res["optimal_conf"] <= 0.40
        assert 0.35 <= res["optimal_iou"] <= 0.55

    def test_calibrate_q_aware_nms_csv_and_json_manifest(self, tmp_path: Path) -> None:
        """Verifies calibrate_q_aware_nms parses CSV and JSON manifests."""
        adapter = YOLOAdapter()

        # Test JSON loader
        json_file = tmp_path / "manifest.json"
        json_data = {
            "detection_samples": [
                {
                    "image_path": "data/sample_images/detection/images/det_000.jpg",
                    "ground_truth_boxes": [{"bbox": [80, 120, 160, 180], "class_id": 0}],
                }
            ]
        }
        json_file.write_text(json.dumps(json_data), encoding="utf-8")
        res_json = calibrate_q_aware_nms(adapter, calib_loader=json_file, conf_min=0.25, conf_max=0.25, iou_min=0.45, iou_max=0.45)
        assert res_json["optimal_conf"] == 0.25

        # Test CSV loader
        csv_file = tmp_path / "manifest.csv"
        csv_file.write_text("path,category,split\ndata/sample_images/detection/images/det_000.jpg,detection,calibration\n", encoding="utf-8")
        res_csv = calibrate_q_aware_nms(adapter, calib_loader=csv_file, conf_min=0.25, conf_max=0.25, iou_min=0.45, iou_max=0.45)
        assert res_csv["optimal_conf"] == 0.25

    def test_apply_q_aware_nms_all_formats(self) -> None:
        """Verifies apply_q_aware_nms across raw tensors, list of tensors, and parsed dicts."""
        policy = {"optimal_conf": 0.50, "optimal_iou": 0.45}
        adapter = YOLOAdapter()

        # 1. Parsed nested dicts
        preds = [[
            {"bbox": [10, 10, 50, 50], "score": 0.30, "class_id": 0},
            {"bbox": [20, 20, 80, 80], "score": 0.85, "class_id": 0},
        ]]
        filtered = apply_q_aware_nms(preds, policy)
        assert len(filtered[0]) == 1
        assert filtered[0][0]["score"] == 0.85

        # 2. Parsed flat list
        flat_preds = [
            {"bbox": [10, 10, 50, 50], "score": 0.30, "class_id": 0},
            {"bbox": [20, 20, 80, 80], "score": 0.85, "class_id": 0},
        ]
        flat_filtered = apply_q_aware_nms(flat_preds, policy)
        assert len(flat_filtered) == 1

        # 3. Raw tensor
        raw_t = torch.randn(1, 84, 8400)
        raw_res = apply_q_aware_nms(
            raw_t,
            policy,
            original_shapes=[(640, 640)],
            ratios=[(1.0, 1.0)],
            pads=[(0.0, 0.0)],
            yolo_adapter=adapter,
        )
        assert isinstance(raw_res, list)

        # 4. List of raw tensors
        raw_list = [raw_t.numpy()]
        list_res = apply_q_aware_nms(
            raw_list,
            policy,
            original_shapes=[(640, 640)],
            ratios=[(1.0, 1.0)],
            pads=[(0.0, 0.0)],
            yolo_adapter=adapter,
        )
        assert isinstance(list_res, list)
        assert len(list_res) == 1

    def test_calibrate_q_aware_nms_empty_loader_raises(self) -> None:
        """Verifies ValueError when no calibration samples are available."""
        adapter = YOLOAdapter()
        with pytest.raises(ValueError, match="No valid calibration samples"):
            calibrate_q_aware_nms(adapter, calib_loader=[{"image_path": "non_existent_file.jpg"}])

    def test_run_q_aware_ablation_end_to_end(self) -> None:
        """Verifies end-to-end execution of run_q_aware_nms_ablation()."""
        run_q_aware_nms_ablation()
        assert Path("results/tables/table6_q_aware_nms_ablation.md").is_file()
        assert Path("results/tables/table7_decision_flip_audit.md").is_file()
        assert Path("results/figures/q_aware_pareto_recovery.png").is_file()
        assert Path("results/figures/decision_flip_attribution.png").is_file()
