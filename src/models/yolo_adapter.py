"""
YOLO Nano detector adapter providing exportable pre-NMS raw tensor output [1, 84, 8400] and decoupled NMS.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def nms_pytorch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    PyTorch vectorized Non-Maximum Suppression (NMS).
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    order = scores.argsort(descending=True)
    if order.numel() > 300:
        order = order[:300]

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1 or len(keep) >= 300:
            break

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / torch.clamp(union, min=1e-8)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def fast_vectorized_nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    max_output_boxes: int = 300,
) -> np.ndarray:
    """Fast vectorized NumPy/PyTorch Non-Maximum Suppression."""
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    t_boxes = torch.from_numpy(boxes.astype(np.float32))
    t_scores = torch.from_numpy(scores.astype(np.float32))
    keep_t = nms_pytorch(t_boxes, t_scores, iou_threshold)
    if keep_t.numel() > max_output_boxes:
        keep_t = keep_t[:max_output_boxes]
    return keep_t.cpu().numpy()


class ConvBlock(nn.Module):
    """Standard Conv-BN-SiLU block."""

    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLONanoDetector(nn.Module):
    """
    Clean, fully exportable YOLO nano architecture producing raw pre-NMS predictions [B, 84, 8400].
    """

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = 4 + num_classes

        # Stem & Backbone
        self.stem = ConvBlock(3, 16, k=3, s=2, p=1)  # 320x320
        self.stage1 = ConvBlock(16, 32, k=3, s=2, p=1)  # 160x160
        self.stage2 = ConvBlock(32, 64, k=3, s=2, p=1)  # 80x80 (P3)
        self.stage3 = ConvBlock(64, 128, k=3, s=2, p=1)  # 40x40 (P4)
        self.stage4 = ConvBlock(128, 256, k=3, s=2, p=1)  # 20x20 (P5)

        # Detection Heads (P3, P4, P5)
        self.head_p3 = nn.Conv2d(64, self.num_outputs, kernel_size=1)
        self.head_p4 = nn.Conv2d(128, self.num_outputs, kernel_size=1)
        self.head_p5 = nn.Conv2d(256, self.num_outputs, kernel_size=1)

        # Deterministic initialization
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Focal / prior initialization for classification heads (prior probability 0.01)
        for head in [self.head_p3, self.head_p4, self.head_p5]:
            if head.bias is not None:
                with torch.no_grad():
                    head.bias[4:].fill_(-4.595)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]

        x1 = self.stem(x)
        x2 = self.stage1(x1)
        p3 = self.stage2(x2)  # [B, 64, 80, 80]
        p4 = self.stage3(p3)  # [B, 128, 40, 40]
        p5 = self.stage4(p4)  # [B, 256, 20, 20]

        out3 = self.head_p3(p3).view(b, self.num_outputs, -1)  # [B, 84, 6400]
        out4 = self.head_p4(p4).view(b, self.num_outputs, -1)  # [B, 84, 1600]
        out5 = self.head_p5(p5).view(b, self.num_outputs, -1)  # [B, 84, 400]

        # Concatenate multi-scale anchors -> [B, 84, 8400]
        raw_output = torch.cat([out3, out4, out5], dim=2)
        return raw_output


class YOLOAdapter:
    """High-level adapter wrapping YOLO nano model with inference and post-processing methods."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        model_name: str = "yolov8n",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        num_classes: int = 80,
        device: str = "cpu",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.device = torch.device(device)

        self.model = YOLONanoDetector(num_classes=num_classes)
        target_weights = weights_path
        if not target_weights:
            default_weights = PROJECT_ROOT / "models" / "weights" / "yolo_nano_baseline.pt"
            if default_weights.is_file():
                target_weights = str(default_weights)

        if target_weights and Path(target_weights).is_file():
            ckpt = torch.load(target_weights, map_location=self.device, weights_only=True)
            state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def get_pytorch_model(self) -> nn.Module:
        """Exposes the underlying PyTorch Module for ONNX/TensorRT export."""
        return self.model

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Executes forward inference in inference mode."""
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            return self.model(tensor)

    def postprocess(
        self,
        raw_output: Any,
        orig_shape: Tuple[int, int],
        ratio: Tuple[float, float],
        pad: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        """
        Decodes raw bounding box predictions, filters by confidence, unpads to original space, and applies vectorized NMS.

        Args:
            raw_output: Raw tensor/array of shape [1, 84, 8400] or [84, 8400].
            orig_shape: Original image (H, W).
            ratio: (ratio_w, ratio_h) from letterbox.
            pad: (pad_left, pad_top) from letterbox.

        Returns:
            List of detection dicts: [{"bbox": [x1, y1, x2, y2], "score": float, "class_id": int, "class_name": str}, ...].
        """
        if isinstance(raw_output, torch.Tensor):
            raw_np = raw_output.squeeze(0).detach().cpu().numpy() if raw_output.dim() == 3 else raw_output.detach().cpu().numpy()
        elif isinstance(raw_output, np.ndarray):
            raw_np = np.squeeze(raw_output) if raw_output.ndim == 3 else raw_output
        else:
            raise TypeError(f"Unsupported raw_output type: {type(raw_output)}")

        if raw_np.shape[0] == 84 and raw_np.shape[1] != 84:
            raw_logits = raw_np[4:, :]  # [80, 8400]
            max_logits = np.max(raw_logits, axis=0)
            class_ids = np.argmax(raw_logits, axis=0)
            logit_thresh = float(np.log(self.conf_threshold / (1.0 - self.conf_threshold + 1e-9)))

            mask = max_logits >= logit_thresh
            if not np.any(mask):
                return []

            boxes = raw_np[:4, mask].T  # [M, 4]
            max_logits = max_logits[mask]
            class_ids = class_ids[mask]
        elif raw_np.shape[1] == 84:
            boxes_np = np.ascontiguousarray(raw_np[:, :4])
            logits_np = np.ascontiguousarray(raw_np[:, 4:])
            max_logits = np.max(logits_np, axis=-1)
            class_ids = np.argmax(logits_np, axis=-1)
            logit_thresh = float(np.log(self.conf_threshold / (1.0 - self.conf_threshold + 1e-9)))

            mask = max_logits >= logit_thresh
            if not np.any(mask):
                return []

            boxes = boxes_np[mask]
            max_logits = max_logits[mask]
            class_ids = class_ids[mask]
        else:
            raise ValueError(f"Unexpected raw_output shape: {raw_np.shape}")

        max_nms = 300
        if max_logits.shape[0] > max_nms:
            top_idx = np.argpartition(max_logits, -max_nms)[-max_nms:]
            top_idx = top_idx[np.argsort(-max_logits[top_idx])]
            boxes = boxes[top_idx]
            class_ids = class_ids[top_idx]
            max_scores = 1.0 / (1.0 + np.exp(-np.clip(max_logits[top_idx], -20.0, 20.0)))
        else:
            order = np.argsort(-max_logits)
            boxes = boxes[order]
            class_ids = class_ids[order]
            max_scores = 1.0 / (1.0 + np.exp(-np.clip(max_logits[order], -20.0, 20.0)))

        # Vectorized (cx, cy, w, h) -> (x1, y1, x2, y2)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5

        pad_w, pad_h = pad
        r_w, r_h = ratio
        r_w_safe = r_w if r_w > 0.0 else 1.0
        r_h_safe = r_h if r_h > 0.0 else 1.0
        x1 = np.clip((x1 - pad_w) / r_w_safe, 0.0, float(orig_shape[1]))
        y1 = np.clip((y1 - pad_h) / r_h_safe, 0.0, float(orig_shape[0]))
        x2 = np.clip((x2 - pad_w) / r_w_safe, 0.0, float(orig_shape[1]))
        y2 = np.clip((y2 - pad_h) / r_h_safe, 0.0, float(orig_shape[0]))

        corner_boxes = np.stack([x1, y1, x2, y2], axis=-1)
        max_dim = max(orig_shape[0], orig_shape[1], 1)
        offsets = class_ids[:, None].astype(np.float32) * (max_dim + 1.0)
        boxes_for_nms = corner_boxes + offsets

        areas = np.maximum(0.0, boxes_for_nms[:, 2] - boxes_for_nms[:, 0]) * np.maximum(0.0, boxes_for_nms[:, 3] - boxes_for_nms[:, 1])

        # Pairwise IoU matrix in one vectorized operation
        xx1 = np.maximum(boxes_for_nms[:, 0, None], boxes_for_nms[None, :, 0])
        yy1 = np.maximum(boxes_for_nms[:, 1, None], boxes_for_nms[None, :, 1])
        xx2 = np.minimum(boxes_for_nms[:, 2, None], boxes_for_nms[None, :, 2])
        yy2 = np.minimum(boxes_for_nms[:, 3, None], boxes_for_nms[None, :, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[:, None] + areas[None, :] - inter
        iou_mat = inter / np.maximum(union, 1e-8)

        suppressed = np.zeros(len(areas), dtype=bool)
        keep = []
        for i in range(len(areas)):
            if suppressed[i]:
                continue
            keep.append(i)
            if len(keep) >= max_nms:
                break
            suppressed |= (iou_mat[i] > self.iou_threshold)

        if not keep:
            return []

        final_boxes = corner_boxes[keep]
        final_scores = max_scores[keep]
        final_classes = class_ids[keep]

        detections = [
            {
                "bbox": [round(float(final_boxes[i, 0]), 2), round(float(final_boxes[i, 1]), 2), round(float(final_boxes[i, 2]), 2), round(float(final_boxes[i, 3]), 2)],
                "score": round(float(final_scores[i]), 4),
                "class_id": int(final_classes[i]),
                "class_name": f"class_{int(final_classes[i])}",
            }
            for i in range(len(keep))
        ]
        return detections
