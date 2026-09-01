"""
Unit tests for Quantization Decision-Change Attribution & Detection Flips.
"""

import numpy as np
import pytest

from src.analysis.decision_flips import compute_detection_flips


class TestDecisionFlips:
    """Test suite asserting decision flip rate (Phi) and TP/FP partitioning correctness."""

    def test_identical_predictions_zero_flips(self) -> None:
        """Verifies flip rate is 0.0 when target predictions exactly match reference predictions."""
        gt = [[{"bbox": [100, 100, 200, 200], "class_id": 0}]]
        preds_ref = [[{"bbox": [100, 100, 200, 200], "score": 0.90, "class_id": 0}]]
        preds_target = [[{"bbox": [100, 100, 200, 200], "score": 0.88, "class_id": 0}]]

        flips = compute_detection_flips(preds_ref, preds_target, gt, iou_thresh=0.5)

        assert flips["total_ref_boxes"] == 1
        assert flips["total_target_boxes"] == 1
        assert flips["total_matched_boxes"] == 1
        assert flips["flip_rate"] == 0.0
        assert flips["total_flips"] == 0
        assert flips["lost_tps"] == 0
        assert flips["new_fps"] == 0

    def test_disjoint_predictions_full_flips(self) -> None:
        """Verifies flip rate is 1.0 when reference and target detections have no spatial overlap."""
        gt = [[{"bbox": [100, 100, 200, 200], "class_id": 0}]]
        preds_ref = [[{"bbox": [100, 100, 200, 200], "score": 0.90, "class_id": 0}]]
        preds_target = [[{"bbox": [400, 400, 500, 500], "score": 0.80, "class_id": 0}]]

        flips = compute_detection_flips(preds_ref, preds_target, gt, iou_thresh=0.5)

        assert flips["total_matched_boxes"] == 0
        assert flips["flip_rate"] == 1.0
        assert flips["total_flips"] == 2
        assert flips["lost_tps"] == 1
        assert flips["new_fps"] == 1
        assert np.isclose(flips["frac_flip_tp"], 0.5, atol=1e-3)
        assert np.isclose(flips["frac_flip_fp"], 0.5, atol=1e-3)

    def test_decision_flips_tp_fp_partitioning(self) -> None:
        """Verifies distinct classification of lost TPs, new TPs, new FPs, and suppressed FPs."""
        # 2 GT boxes: Box A [0..100], Box B [200..300]
        gt = [[
            {"bbox": [0, 0, 100, 100], "class_id": 0},
            {"bbox": [200, 200, 300, 300], "class_id": 0},
        ]]
        # Reference detected Box A (TP) and an extra ghost box [500..600] (FP)
        preds_ref = [[
            {"bbox": [0, 0, 100, 100], "score": 0.90, "class_id": 0},
            {"bbox": [500, 500, 600, 600], "score": 0.40, "class_id": 0},
        ]]
        # Target detected Box B (New TP) and eliminated ghost box (Suppressed FP)
        preds_target = [[
            {"bbox": [200, 200, 300, 300], "score": 0.85, "class_id": 0},
        ]]

        flips = compute_detection_flips(preds_ref, preds_target, gt, iou_thresh=0.5)
        assert flips["lost_tps"] == 1       # Box A lost
        assert flips["new_tps"] == 1        # Box B newly detected
        assert flips["suppressed_fps"] == 1 # Ghost box suppressed
        assert flips["new_fps"] == 0
        assert flips["total_flips"] == 3
        assert np.isclose(flips["frac_flip_tp"], 2.0 / 3.0, atol=1e-3)
        assert np.isclose(flips["frac_flip_fp"], 1.0 / 3.0, atol=1e-3)

    def test_decision_flips_empty_edge_cases(self) -> None:
        """Asserts correct handling of completely empty prediction lists."""
        flips = compute_detection_flips([[]], [[]], [[]], iou_thresh=0.5)
        assert flips["flip_rate"] == 0.0
        assert flips["total_flips"] == 0
        assert flips["frac_flip_tp"] == 0.0
        assert flips["frac_flip_fp"] == 0.0

        # Non-empty ref, empty target
        flips_empty_t = compute_detection_flips(
            [[{"bbox": [0, 0, 10, 10], "score": 0.5, "class_id": 0}]],
            [[]],
            [[]],
            iou_thresh=0.5,
        )
        assert flips_empty_t["total_flips"] == 1
        assert flips_empty_t["suppressed_fps"] == 1
