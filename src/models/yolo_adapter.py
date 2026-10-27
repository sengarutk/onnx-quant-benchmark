"""
Compact anchor-free single-stage detector ("YOLO-Nano"-style) with an exportable decoded output tensor
[B, 4 + num_classes, N] and decoupled host-side NMS.

Architecture: a 5-stage strided Conv-BN-SiLU backbone (refinement block in stages 2-4) feeding three
detection heads at strides 8/16/32 (P3/P4/P5). For a 640x640 input this yields N = 80^2 + 40^2 + 20^2 = 8400
grid locations. Each location predicts FCOS-style (left, top, right, bottom) distances, decoded inside the graph
to (cx, cy, w, h) in input-pixel coordinates, plus per-class logits. The output layout matches the familiar
YOLOv8 ONNX layout, so the post-processing below is runtime-agnostic.

This is NOT Ultralytics YOLOv8n; it is a deliberately small detector trained from scratch on the repository's
synthetic 2-class dataset (see scripts/train_models.py).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.synthetic import DETECTION_CLASS_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STRIDES: Tuple[int, int, int] = (8, 16, 32)
DEFAULT_NUM_CLASSES = len(DETECTION_CLASS_NAMES)


def nms_pytorch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, max_det: int = 300) -> torch.Tensor:
    """Greedy Non-Maximum Suppression over (x1, y1, x2, y2) boxes; returns kept indices sorted by score."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    order = scores.argsort(descending=True)

    keep: List[int] = []
    while order.numel() > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(int(i.item()))
        if order.numel() == 1:
            break
        rest = order[1:]
        w = (torch.minimum(x2[i], x2[rest]) - torch.maximum(x1[i], x1[rest])).clamp(min=0.0)
        h = (torch.minimum(y2[i], y2[rest]) - torch.maximum(y1[i], y1[rest])).clamp(min=0.0)
        inter = w * h
        iou = inter / torch.clamp(areas[i] + areas[rest] - inter, min=1e-8)
        order = rest[iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def fast_vectorized_nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    max_output_boxes: int = 300,
) -> np.ndarray:
    """NumPy front-end to nms_pytorch."""
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    keep_t = nms_pytorch(
        torch.from_numpy(boxes.astype(np.float32)),
        torch.from_numpy(scores.astype(np.float32)),
        iou_threshold,
        max_det=max_output_boxes,
    )
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


class DetectionHead(nn.Module):
    """3x3 Conv-BN-SiLU stem followed by a 1x1 prediction conv producing 4 + num_classes channels."""

    def __init__(self, in_c: int, num_outputs: int):
        super().__init__()
        self.stem = ConvBlock(in_c, in_c, k=3, s=1, p=1)
        self.pred = nn.Conv2d(in_c, num_outputs, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(self.stem(x))


class YOLONanoDetector(nn.Module):
    """
    Exportable anchor-free detector producing decoded predictions [B, 4 + num_classes, N]:
    rows 0..3 are (cx, cy, w, h) in input pixels, rows 4.. are raw class logits.
    """

    def __init__(self, num_classes: int = DEFAULT_NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = 4 + num_classes

        self.stem = ConvBlock(3, 16, k=3, s=2, p=1)                                         # /2
        self.stage1 = ConvBlock(16, 32, k=3, s=2, p=1)                                      # /4
        self.stage2 = nn.Sequential(ConvBlock(32, 64, 3, 2, 1), ConvBlock(64, 64))          # /8  (P3)
        self.stage3 = nn.Sequential(ConvBlock(64, 128, 3, 2, 1), ConvBlock(128, 128))       # /16 (P4)
        self.stage4 = nn.Sequential(ConvBlock(128, 256, 3, 2, 1), ConvBlock(256, 256))      # /32 (P5)

        self.head_p3 = DetectionHead(64, self.num_outputs)
        self.head_p4 = DetectionHead(128, self.num_outputs)
        self.head_p5 = DetectionHead(256, self.num_outputs)

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
        for head in (self.head_p3, self.head_p4, self.head_p5):
            nn.init.normal_(head.pred.weight, std=0.01)
            with torch.no_grad():
                head.pred.bias.zero_()
                head.pred.bias[:4].fill_(1.0)       # softplus(1) * stride ~ 1.3 strides initial extent
                head.pred.bias[4:].fill_(-4.595)    # focal-loss prior p = 0.01

    def forward_raw(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns per-level raw head maps [B, 4 + C, H_l, W_l] (used by the training loss)."""
        p3 = self.stage2(self.stage1(self.stem(x)))
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [self.head_p3(p3), self.head_p4(p4), self.head_p5(p5)]

    @staticmethod
    def decode(raw_maps: List[torch.Tensor]) -> torch.Tensor:
        """Decodes per-level maps into [B, 4 + C, N] with (cx, cy, w, h) in input pixels."""
        outs = []
        for raw, stride in zip(raw_maps, STRIDES):
            b, c, h, w = raw.shape
            ys = (torch.arange(h, device=raw.device, dtype=raw.dtype) + 0.5) * stride
            xs = (torch.arange(w, device=raw.device, dtype=raw.dtype) + 0.5) * stride
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            ltrb = F.softplus(raw[:, :4]) * stride
            x1 = gx - ltrb[:, 0]
            y1 = gy - ltrb[:, 1]
            x2 = gx + ltrb[:, 2]
            y2 = gy + ltrb[:, 3]
            boxes = torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dim=1)
            outs.append(torch.cat([boxes, raw[:, 4:]], dim=1).reshape(b, c, h * w))
        return torch.cat(outs, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.forward_raw(x))


class YOLOAdapter:
    """High-level adapter wrapping the detector with weight loading, inference and post-processing."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        num_classes: int = DEFAULT_NUM_CLASSES,
        device: str = "cpu",
        require_weights: bool = False,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.class_names = DETECTION_CLASS_NAMES if num_classes == DEFAULT_NUM_CLASSES else [
            f"class_{i}" for i in range(num_classes)
        ]

        self.model = YOLONanoDetector(num_classes=num_classes)
        target = Path(weights_path) if weights_path else PROJECT_ROOT / "models" / "weights" / "yolo_nano_trained.pt"
        self.weights_loaded = False
        if target.is_file():
            ckpt = torch.load(str(target), map_location="cpu", weights_only=True)
            state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(state_dict, strict=True)
            self.weights_loaded = True
        elif require_weights or weights_path:
            raise FileNotFoundError(f"Detector weights not found: {target}. Run scripts/train_models.py first.")

        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def get_pytorch_model(self) -> nn.Module:
        """Exposes the underlying PyTorch Module for ONNX/TensorRT export."""
        return self.model

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Executes forward inference in inference mode."""
        with torch.inference_mode():
            return self.model(tensor.to(self.device))

    def postprocess(
        self,
        raw_output: Any,
        orig_shape: Tuple[int, int],
        ratio: Tuple[float, float],
        pad: Tuple[float, float],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_det: int = 300,
    ) -> List[Dict[str, Any]]:
        """
        Confidence filtering, letterbox un-mapping and class-aware NMS.

        Args:
            raw_output: [1, 4 + C, N], [4 + C, N] or [N, 4 + C] tensor/array.
            orig_shape: Original image (H, W).
            ratio: (ratio_w, ratio_h) from letterbox.
            pad: (pad_left, pad_top) from letterbox.

        Returns:
            List of {"bbox": [x1, y1, x2, y2], "score", "class_id", "class_name"} in original image coordinates.
        """
        if isinstance(raw_output, torch.Tensor):
            raw_np = raw_output.detach().float().cpu().numpy()
        elif isinstance(raw_output, np.ndarray):
            raw_np = raw_output.astype(np.float32, copy=False)
        else:
            raise TypeError(f"Unsupported raw_output type: {type(raw_output)}")
        if raw_np.ndim == 3:
            if raw_np.shape[0] != 1:
                raise ValueError("postprocess expects a single image (batch dimension of 1)")
            raw_np = raw_np[0]

        n_out = 4 + self.num_classes
        if raw_np.ndim != 2:
            raise ValueError(f"Unexpected raw_output shape: {raw_np.shape}")
        if raw_np.shape[0] == n_out:
            preds = raw_np.T
        elif raw_np.shape[1] == n_out:
            preds = raw_np
        else:
            raise ValueError(f"Unexpected raw_output shape {raw_np.shape} for {self.num_classes} classes")

        conf = self.conf_threshold if conf_threshold is None else conf_threshold
        iou_t = self.iou_threshold if iou_threshold is None else iou_threshold
        logit_thresh = float(np.log(conf / max(1.0 - conf, 1e-9)))

        logits = preds[:, 4:]
        class_ids = np.argmax(logits, axis=1)
        max_logits = logits[np.arange(len(logits)), class_ids]
        mask = max_logits >= logit_thresh
        if not np.any(mask):
            return []

        boxes = preds[mask, :4]
        max_logits = max_logits[mask]
        class_ids = class_ids[mask]

        # Keep the top-k candidates before NMS (bounds worst-case NMS cost)
        if max_logits.shape[0] > 3000:
            top = np.argpartition(max_logits, -3000)[-3000:]
            boxes, max_logits, class_ids = boxes[top], max_logits[top], class_ids[top]
        scores = 1.0 / (1.0 + np.exp(-np.clip(max_logits, -30.0, 30.0)))

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        r_w = ratio[0] if ratio[0] > 0 else 1.0
        r_h = ratio[1] if ratio[1] > 0 else 1.0
        x1 = np.clip((cx - w * 0.5 - pad[0]) / r_w, 0.0, float(orig_shape[1]))
        y1 = np.clip((cy - h * 0.5 - pad[1]) / r_h, 0.0, float(orig_shape[0]))
        x2 = np.clip((cx + w * 0.5 - pad[0]) / r_w, 0.0, float(orig_shape[1]))
        y2 = np.clip((cy + h * 0.5 - pad[1]) / r_h, 0.0, float(orig_shape[0]))
        corner = np.stack([x1, y1, x2, y2], axis=-1)

        # Class-aware NMS via per-class coordinate offsets
        offsets = class_ids[:, None].astype(np.float32) * (max(orig_shape) + 1.0)
        keep = fast_vectorized_nms_numpy(corner + offsets, scores, iou_t, max_output_boxes=max_det)

        return [
            {
                "bbox": [round(float(v), 2) for v in corner[k]],
                "score": round(float(scores[k]), 5),
                "class_id": int(class_ids[k]),
                "class_name": self.class_names[int(class_ids[k])],
            }
            for k in keep
        ]
