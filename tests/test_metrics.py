
"""
Unit tests for detection and anomaly evaluation quality engines.
"""

import numpy as np
import pytest

from src.validation.anomaly_quality import compute_aupro, compute_image_auroc, compute_pixel_auroc
from src.validation.detection_quality import box_iou, compute_ap, evaluate_detection_dataset
from src.validation.output_checks import compute_tensor_diff


class TestMetrics:
    """Test suite validating metric accuracy against known standard analytical vectors."""

    def test_box_iou_perfect_and_disjoint(self) -> None:
        """Tests IoU calculation on identical and disjoint bounding boxes."""
        b1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b3 = np.array([[20, 20, 30, 30]], dtype=np.float32)

        assert np.isclose(box_iou(b1, b2)[0, 0], 1.0)
        assert np.isclose(box_iou(b1, b3)[0, 0], 0.0)

    def test_compute_ap_perfect(self) -> None:
        """Tests AP calculation on perfect precision-recall curve."""
        rec = np.array([0.2, 0.5, 0.8, 1.0])
        prec = np.array([1.0, 1.0, 1.0, 1.0])
        ap = compute_ap(rec, prec)
        assert np.isclose(ap, 1.0)

    def test_evaluate_detection_dataset_perfect_match(self) -> None:
        """Tests evaluate_detection_dataset when predictions perfectly match ground truths."""
        preds = [[{"bbox": [10, 10, 50, 50], "score": 0.95, "class_id": 0}]]
        gts = [[{"bbox": [10, 10, 50, 50], "class_id": 0}]]

        metrics = evaluate_detection_dataset(preds, gts)
        assert metrics["mAP_50"] == 1.0
        assert metrics["mAP_50_95"] > 0.9

    def test_compute_image_auroc(self) -> None:
        """Tests Image AUROC with perfect and inverted classifier rankings."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_score_perfect = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
        y_score_inverted = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]

        assert compute_image_auroc(y_true, y_score_perfect) == 1.0
        assert compute_image_auroc(y_true, y_score_inverted) == 0.0

    def test_compute_pixel_auroc_and_aupro(self) -> None:
        """Tests pixel AUROC and AU-PRO with synthetic binary masks."""
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[50:100, 50:100] = 1.0

        # Anomaly map with high signal in defect region
        a_map = np.zeros((256, 256), dtype=np.float32)
        a_map[50:100, 50:100] = 0.95
        a_map[0:20, 0:20] = 0.05

        p_auroc = compute_pixel_auroc([mask], [a_map])
        aupro = compute_aupro([mask], [a_map])

        assert p_auroc > 0.95
        assert aupro > 0.90

    def test_compute_tensor_diff_identical_and_perturbed(self) -> None:
        """Tests compute_tensor_diff gives zero error on identical arrays and accurate MAE on perturbations."""
        a = np.ones((10, 10), dtype=np.float32)
        b = np.ones((10, 10), dtype=np.float32)
        c = a + 0.05

        diff_ident = compute_tensor_diff(a, b)
        assert diff_ident["max_abs_error"] == 0.0
        assert diff_ident["mean_abs_error"] == 0.0
        assert diff_ident["cosine_similarity"] == 1.0

        diff_pert = compute_tensor_diff(a, c)
        assert np.isclose(diff_pert["max_abs_error"], 0.05)
        assert np.isclose(diff_pert["mean_abs_error"], 0.05)

    def test_check_detection_output_consistency(self) -> None:
        """Tests check_detection_output_consistency matching and deviation reporting."""
        from src.validation.output_checks import check_detection_output_consistency

        d1 = [{"bbox": [10, 10, 50, 50], "score": 0.9, "class_id": 0}]
        d2 = [{"bbox": [11, 10, 51, 50], "score": 0.89, "class_id": 0}]

        res = check_detection_output_consistency(d1, d2)
        assert res["matched_boxes"] == 1
        assert res["class_match_rate"] == 1.0
        assert res["mean_box_mae"] > 0.0
