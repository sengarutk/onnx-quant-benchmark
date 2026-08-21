"""Validation quality metrics and numerical output equivalence checks."""
from src.validation.detection_quality import box_iou, compute_ap, evaluate_detection_dataset
from src.validation.anomaly_quality import compute_image_auroc, compute_pixel_auroc, compute_aupro
from src.validation.output_checks import compute_tensor_diff, check_detection_output_consistency

__all__ = [
    "box_iou",
    "compute_ap",
    "evaluate_detection_dataset",
    "compute_image_auroc",
    "compute_pixel_auroc",
    "compute_aupro",
    "compute_tensor_diff",
    "check_detection_output_consistency",
]
