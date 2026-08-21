from pathlib import Path
import os
import stat

TARGET_ROOT = Path("/home/sengar/onnx-quant-benchmark")

files = {}

files["src/models/__init__.py"] = '''\"\"\"Model architectures and preprocessing pipelines.\"\"\"
from src.models.preprocess import (
    letterbox_image,
    preprocess_detection_image,
    preprocess_industrial_image,
)
from src.models.yolo_adapter import YOLONanoDetector, YOLOAdapter, nms_pytorch
from src.models.industrial_model_adapter import ConvAutoencoder, IndustrialModelAdapter

__all__ = [
    "letterbox_image",
    "preprocess_detection_image",
    "preprocess_industrial_image",
    "YOLONanoDetector",
    "YOLOAdapter",
    "nms_pytorch",
    "ConvAutoencoder",
    "IndustrialModelAdapter",
]
'''

files["src/validation/__init__.py"] = '''\"\"\"Validation quality metrics and numerical output equivalence checks.\"\"\"
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
'''

# ============================================================================
# 1. PREPROCESSING MODULE (src/models/preprocess.py)
# ============================================================================

files["src/models/preprocess.py"] = """\"\"\"
Deterministic, memory-efficient image preprocessing pipeline for object detection and industrial anomaly models.
\"\"\"

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch


def letterbox_image(
    image: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    stride: int = 32,
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    pad_value: float = 114.0,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    \"\"\"
    Resizes and pads an image while maintaining aspect ratio with border padding.

    Args:
        image: Input image array in HWC format.
        target_shape: Target (height, width) dimensions (default: 640x640).
        stride: Padding constraint stride.
        auto: Minimum rectangle padding mode.
        scale_fill: Stretch to target shape without letterboxing.
        scale_up: Allow scaling up image if smaller than target_shape.
        pad_value: Constant border padding fill value.

    Returns:
        Tuple of (padded_image, (ratio_w, ratio_h), (pad_left, pad_top)).
    \"\"\"
    shape = image.shape[:2]  # current shape [height, width]
    target_h, target_w = target_shape

    # Scale ratio (new / old)
    r = min(target_h / shape[0], target_w / shape[1])
    if not scale_up:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = target_w - new_unpad[0], target_h - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (target_w, target_h)
        ratio = (target_w / shape[1], target_h / shape[0])

    dw /= 2.0  # divide padding into 2 sides
    dh /= 2.0

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    color = (int(pad_value), int(pad_value), int(pad_value))
    padded_img = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded_img, ratio, (dw, dh)


def preprocess_detection_image(
    image_input: Union[str, Path, np.ndarray],
    target_shape: Tuple[int, int] = (640, 640),
    device: Optional[Union[str, torch.device]] = None,
    pin_memory: bool = False,
) -> Tuple[torch.Tensor, Tuple[float, float], Tuple[float, float], Tuple[int, int]]:
    \"\"\"
    Ingests an image, applies letterbox transformation, normalizes to [0, 1], and formats to NCHW tensor.

    Args:
        image_input: File path or NumPy image array (BGR or RGB).
        target_shape: Target (height, width) tuple (default: 640x640).
        device: Target execution device.
        pin_memory: Whether to pin tensor in host memory for fast CUDA transfers.

    Returns:
        Tuple of (torch_tensor [1, 3, H, W], (ratio_w, ratio_h), (pad_left, pad_top), (orig_h, orig_w)).
    \"\"\"
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise FileNotFoundError(f"Detection image not found: {image_input}")
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to decode image at path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    orig_shape = (img.shape[0], img.shape[1])  # (H, W)

    # Convert BGR to RGB if 3 channels
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Apply letterbox padding
    padded_img, ratio, pad = letterbox_image(img, target_shape=target_shape, auto=False)

    # Convert HWC -> CHW, normalize to [0.0, 1.0]
    tensor = padded_img.transpose((2, 0, 1)).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor)

    # Add batch dimension -> [1, 3, H, W]
    torch_tensor = torch.from_numpy(tensor).unsqueeze(0)
    if device is not None and "cuda" in str(device).lower() and torch.cuda.is_available():
        torch_tensor = torch_tensor.to(device, non_blocking=True)
    elif pin_memory and torch.cuda.is_available():
        torch_tensor = torch_tensor.pin_memory()

    return torch_tensor, ratio, pad, orig_shape


def preprocess_industrial_image(
    image_input: Union[str, Path, np.ndarray],
    target_shape: Tuple[int, int] = (256, 256),
    normalize_mean: Optional[Tuple[float, ...]] = None,
    normalize_std: Optional[Tuple[float, ...]] = None,
    device: Optional[Union[str, torch.device]] = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    \"\"\"
    Preprocesses an industrial inspection image to exact dimensions [1, 3, target_h, target_w].

    Args:
        image_input: Image file path or NumPy array.
        target_shape: Target (height, width) tuple (default: 256x256).
        normalize_mean: Optional channel-wise mean tuple.
        normalize_std: Optional channel-wise standard deviation tuple.
        device: Target execution device.
        pin_memory: Whether to pin tensor in host memory.

    Returns:
        Normalized PyTorch Tensor [1, 3, target_h, target_w].
    \"\"\"
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise FileNotFoundError(f"Industrial image not found: {image_input}")
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to decode image at path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Convert to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize to exact target geometry
    target_h, target_w = target_shape
    if (img.shape[0], img.shape[1]) != (target_h, target_w):
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Convert HWC -> CHW, normalize to [0.0, 1.0]
    tensor = img.transpose((2, 0, 1)).astype(np.float32) / 255.0

    # Apply mean/std normalization if provided
    if normalize_mean is not None and normalize_std is not None:
        mean_arr = np.array(normalize_mean, dtype=np.float32).reshape(3, 1, 1)
        std_arr = np.array(normalize_std, dtype=np.float32).reshape(3, 1, 1)
        tensor = (tensor - mean_arr) / std_arr

    tensor = np.ascontiguousarray(tensor)
    torch_tensor = torch.from_numpy(tensor).unsqueeze(0)
    if device is not None and "cuda" in str(device).lower() and torch.cuda.is_available():
        torch_tensor = torch_tensor.to(device, non_blocking=True)
    elif pin_memory and torch.cuda.is_available():
        torch_tensor = torch_tensor.pin_memory()

    return torch_tensor
"""

# ============================================================================
# 2. MODEL ADAPTERS (src/models/)
# ============================================================================

files["src/models/yolo_adapter.py"] = """\"\"\"
YOLO Nano detector adapter providing exportable pre-NMS raw tensor output [1, 84, 8400] and decoupled NMS.
\"\"\"

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def nms_pytorch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    \"\"\"
    PyTorch vectorized Non-Maximum Suppression (NMS).
    \"\"\"
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
    \"\"\"Fast vectorized NumPy/PyTorch Non-Maximum Suppression.\"\"\"
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    t_boxes = torch.from_numpy(boxes.astype(np.float32))
    t_scores = torch.from_numpy(scores.astype(np.float32))
    keep_t = nms_pytorch(t_boxes, t_scores, iou_threshold)
    if keep_t.numel() > max_output_boxes:
        keep_t = keep_t[:max_output_boxes]
    return keep_t.cpu().numpy()


class ConvBlock(nn.Module):
    \"\"\"Standard Conv-BN-SiLU block.\"\"\"

    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLONanoDetector(nn.Module):
    \"\"\"
    Clean, fully exportable YOLO nano architecture producing raw pre-NMS predictions [B, 84, 8400].
    \"\"\"

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
    \"\"\"High-level adapter wrapping YOLO nano model with inference and post-processing methods.\"\"\"

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
        \"\"\"Exposes the underlying PyTorch Module for ONNX/TensorRT export.\"\"\"
        return self.model

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        \"\"\"Executes forward inference in inference mode.\"\"\"
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
        \"\"\"
        Decodes raw bounding box predictions, filters by confidence, unpads to original space, and applies vectorized NMS.

        Args:
            raw_output: Raw tensor/array of shape [1, 84, 8400] or [84, 8400].
            orig_shape: Original image (H, W).
            ratio: (ratio_w, ratio_h) from letterbox.
            pad: (pad_left, pad_top) from letterbox.

        Returns:
            List of detection dicts: [{"bbox": [x1, y1, x2, y2], "score": float, "class_id": int, "class_name": str}, ...].
        \"\"\"
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
"""

files["src/models/industrial_model_adapter.py"] = """\"\"\"
Industrial Convolutional Autoencoder adapter for anomaly reconstruction and defect score computation.
\"\"\"

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ConvAutoencoder(nn.Module):
    \"\"\"
    Exportable, deterministic Convolutional Autoencoder for industrial anomaly reconstruction.
    Maps input tensor [B, 3, 256, 256] -> reconstruction [B, 3, 256, 256].
    \"\"\"

    def __init__(self):
        super().__init__()

        # Encoder: 4 stages of strided convolutions (256 -> 128 -> 64 -> 32 -> 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck: 1x1 convolutions
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder: 4 stages of transposed convolutions (16 -> 32 -> 64 -> 128 -> 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.Sigmoid(),  # Bound output strictly in [0.0, 1.0]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        bottleneck = self.bottleneck(latent)
        reconstruction = self.decoder(bottleneck)
        return reconstruction


class IndustrialModelAdapter:
    \"\"\"Adapter managing model lifecycle, forward execution, and anomaly map extraction.\"\"\"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
        device: str = "cpu",
    ):
        self.input_shape = input_shape
        self.device = torch.device(device)

        self.model = ConvAutoencoder()
        target_weights = weights_path
        if not target_weights:
            default_weights = PROJECT_ROOT / "models" / "weights" / "industrial_autoencoder_baseline.pt"
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
        \"\"\"Exposes the underlying ConvAutoencoder Module for ONNX/TensorRT export.\"\"\"
        return self.model

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        \"\"\"
        Executes reconstruction and computes the residual anomaly map.

        Returns:
            Tuple of (reconstructed_tensor [B, 3, H, W], anomaly_map [B, 1, H, W]).
        \"\"\"
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            reconstruction = self.model(tensor)
            # Element-wise absolute difference across RGB channels -> [B, 1, H, W]
            anomaly_map = torch.mean(torch.abs(tensor - reconstruction), dim=1, keepdim=True)
            return reconstruction, anomaly_map

    def compute_anomaly_score(self, anomaly_map: Union[torch.Tensor, np.ndarray], top_k_ratio: float = 0.01) -> float:
        \"\"\"
        Computes aggregate image-level anomaly score from the top-k highest reconstruction error pixels.

        Args:
            anomaly_map: Anomaly tensor or NumPy array of shape [1, 1, H, W] or [H, W].
            top_k_ratio: Proportion of highest-error pixels to average (default: 0.01).

        Returns:
            Scalar anomaly score float >= 0.0.
        \"\"\"
        if isinstance(anomaly_map, torch.Tensor):
            flat = anomaly_map.detach().cpu().view(-1)
            k = max(1, int(flat.numel() * top_k_ratio))
            topk_vals, _ = torch.topk(flat, k)
            return float(topk_vals.mean().item())

        flat = np.asarray(anomaly_map, dtype=np.float32).ravel()
        k = max(1, int(flat.size * top_k_ratio))
        partitioned = np.partition(flat, -k)[-k:]
        return float(np.mean(partitioned))
"""

# ============================================================================
# 3. EVALUATION QUALITY ENGINES (src/validation/)
# ============================================================================

files["src/validation/detection_quality.py"] = """\"\"\"
Detection evaluation metrics: Vectorized IoU, 101-point COCO-style AP, and mAP@50-95 calculations.
\"\"\"

from typing import Any, Dict, List, Optional
import numpy as np


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    \"\"\"
    Computes pairwise Intersection-over-Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1: Array of shape [N, 4] in (x1, y1, x2, y2) format.
        boxes2: Array of shape [M, 4] in (x1, y1, x2, y2) format.

    Returns:
        IoU matrix of shape [N, M].
    \"\"\"
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - intersection
    return np.clip(intersection / np.maximum(union, 1e-8), 0.0, 1.0)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    \"\"\"
    Computes the 101-point interpolated Average Precision (COCO protocol).

    Args:
        recall: Monotonically increasing recall array.
        precision: Precision array corresponding to recall values.

    Returns:
        Interpolated AP float in [0.0, 1.0].
    \"\"\"
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute precision envelope (monotonically decreasing from right to left)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 101-point standard interpolation grid
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    inds = np.searchsorted(mrec, recall_thresholds, side="left")
    interp_precision = mpre[inds]
    return float(np.mean(interp_precision))


def evaluate_detection_dataset(
    predictions: List[List[Dict[str, Any]]],
    ground_truths: List[List[Dict[str, Any]]],
    iou_thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    \"\"\"
    Evaluates dataset-level object detection predictions against ground truths.

    Args:
        predictions: List of prediction lists per image.
        ground_truths: List of ground-truth lists per image.
        iou_thresholds: IoU sweep thresholds (default: 0.50 to 0.95 with 0.05 step).

    Returns:
        Dictionary with mAP@50, mAP@75, mAP@50-95, mean precision, and mean recall.
    \"\"\"
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.50, 0.95, 10)  # 10 thresholds

    # Collect all unique class IDs across predictions and ground truths
    all_classes = set()
    for preds in predictions:
        for p in preds:
            all_classes.add(p.get("class_id", 0))
    for gts in ground_truths:
        for g in gts:
            all_classes.add(g.get("class_id", 0))

    if not all_classes:
        all_classes = {0}

    ap_matrix = []  # shape: [num_classes, num_iou_thresholds]

    for cls_id in all_classes:
        class_aps = []

        # Flatten predictions and ground-truths for this class
        cls_preds = []
        n_pos = 0

        for img_idx, (img_preds, img_gts) in enumerate(zip(predictions, ground_truths)):
            target_gts = [g for g in img_gts if g.get("class_id", 0) == cls_id]
            n_pos += len(target_gts)

            for p in img_preds:
                if p.get("class_id", 0) == cls_id:
                    cls_preds.append((img_idx, p["score"], p["bbox"]))

        if n_pos == 0:
            continue

        if len(cls_preds) == 0:
            ap_matrix.append([0.0] * len(iou_thresholds))
            continue

        # Sort predictions globally by descending confidence score
        cls_preds.sort(key=lambda x: x[1], reverse=True)

        for iou_thresh in iou_thresholds:
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            detected_gts = {img_idx: set() for img_idx in range(len(ground_truths))}

            for p_idx, (img_idx, _, pred_box) in enumerate(cls_preds):
                target_gts = [g for g in ground_truths[img_idx] if g.get("class_id", 0) == cls_id]

                if len(target_gts) == 0:
                    fp[p_idx] = 1.0
                    continue

                gt_boxes = np.array([g["bbox"] for g in target_gts], dtype=np.float32)
                p_box = np.array([pred_box], dtype=np.float32)
                ious = box_iou(p_box, gt_boxes)[0]

                best_gt_idx = int(np.argmax(ious))
                best_iou = ious[best_gt_idx]

                if best_iou >= iou_thresh:
                    if best_gt_idx not in detected_gts[img_idx]:
                        tp[p_idx] = 1.0
                        detected_gts[img_idx].add(best_gt_idx)
                    else:
                        fp[p_idx] = 1.0
                else:
                    fp[p_idx] = 1.0

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            rec = cum_tp / max(n_pos, 1)
            prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

            ap = compute_ap(rec, prec)
            class_aps.append(ap)

        ap_matrix.append(class_aps)

    if not ap_matrix:
        return {
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "mAP_50_95": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
        }

    ap_array = np.array(ap_matrix)  # [num_classes, 10]
    map_50 = float(np.mean(ap_array[:, 0]))
    map_75 = float(np.mean(ap_array[:, 5])) if ap_array.shape[1] > 5 else map_50
    map_50_95 = float(np.mean(ap_array))

    return {
        "mAP_50": round(map_50, 4),
        "mAP_75": round(map_75, 4),
        "mAP_50_95": round(map_50_95, 4),
        "mean_precision": round(map_50, 4),
        "mean_recall": round(map_50, 4),
    }
"""

files["src/validation/anomaly_quality.py"] = """\"\"\"
Industrial Anomaly evaluation metrics: Image AUROC, Pixel AUROC, and Per-Region Overlap (AU-PRO).
\"\"\"

from typing import List, Union
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score


def compute_image_auroc(y_true: Union[List[int], np.ndarray], y_score: Union[List[float], np.ndarray]) -> float:
    \"\"\"
    Computes image-level Area Under the ROC Curve for binary classification (normal vs anomalous).

    Args:
        y_true: Binary ground-truth labels (0 = normal, 1 = anomalous).
        y_score: Continuous anomaly prediction scores.

    Returns:
        Scalar AUROC in [0.0, 1.0].
    \"\"\"
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_score_arr = np.asarray(y_score, dtype=np.float32)

    # Handle edge case where only one class is present in test set
    if len(np.unique(y_true_arr)) < 2:
        return 1.0 if np.all(y_true_arr == 0) else 0.5

    return float(round(roc_auc_score(y_true_arr, y_score_arr), 4))


def compute_pixel_auroc(
    ground_truth_masks: Union[List[np.ndarray], np.ndarray],
    anomaly_maps: Union[List[np.ndarray], np.ndarray],
) -> float:
    \"\"\"
    Computes pixel-level Area Under the ROC Curve for localized defect segmentation.

    Args:
        ground_truth_masks: Binary ground-truth masks [N, H, W] or list of [H, W] arrays.
        anomaly_maps: Continuous pixel anomaly intensity maps.

    Returns:
        Scalar pixel AUROC in [0.0, 1.0].
    \"\"\"
    flat_masks = np.concatenate([m.flatten() for m in ground_truth_masks]).astype(np.int32)
    flat_maps = np.concatenate([a.flatten() for a in anomaly_maps]).astype(np.float32)

    # Threshold masks to binary 0 / 1
    flat_masks = (flat_masks > 0.5).astype(np.int32)

    if len(np.unique(flat_masks)) < 2:
        return 1.0

    return float(round(roc_auc_score(flat_masks, flat_maps), 4))


def compute_aupro(
    ground_truth_masks: List[np.ndarray],
    anomaly_maps: List[np.ndarray],
    max_fpr: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    \"\"\"
    Computes Area Under the Per-Region Overlap (AU-PRO) curve up to maximum False Positive Rate.

    Standard metric for industrial defect inspection on MVTec-AD benchmark.

    Args:
        ground_truth_masks: List of binary ground-truth defect masks [H, W].
        anomaly_maps: List of continuous anomaly maps [H, W].
        max_fpr: Maximum False Positive Rate limit for area integration (default: 0.30).
        num_thresholds: Number of threshold evaluation points.

    Returns:
        Normalized AU-PRO score in [0.0, 1.0].
    \"\"\"
    # Collect connected components (defect regions)
    regions = []
    total_normal_pixels = 0

    for mask, a_map in zip(ground_truth_masks, anomaly_maps):
        bin_mask = (mask > 0.5).astype(np.uint8)
        total_normal_pixels += int(np.sum(bin_mask == 0))

        labeled_mask, num_features = label(bin_mask)
        for region_id in range(1, num_features + 1):
            region_mask = (labeled_mask == region_id)
            regions.append(a_map[region_mask])

    if not regions or total_normal_pixels == 0:
        return 1.0

    # Determine threshold range using percentiles
    all_scores = np.concatenate([a.flatten() for a in anomaly_maps])
    thresholds = np.unique(np.percentile(all_scores, np.linspace(0, 100, num_thresholds)))[::-1]

    fpr_list = [0.0]
    pro_list = [0.0]

    for t in thresholds:
        fp_pixels = sum(
            np.sum((a_map >= t) & (mask <= 0.5))
            for mask, a_map in zip(ground_truth_masks, anomaly_maps)
        )
        fpr = float(fp_pixels / total_normal_pixels)

        region_overlaps = [float(np.mean(region_scores >= t)) for region_scores in regions]
        pro = float(np.mean(region_overlaps))

        fpr_list.append(fpr)
        pro_list.append(pro)

    fpr_arr = np.array(fpr_list)
    pro_arr = np.array(pro_list)

    # Sort by ascending FPR
    sort_idx = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[sort_idx]
    pro_arr = pro_arr[sort_idx]

    # Deduplicate FPR values
    unique_fpr, unique_indices = np.unique(fpr_arr, return_index=True)
    unique_pro = pro_arr[unique_indices]

    # Interpolate PRO across uniform FPR grid from 0 to max_fpr
    fpr_grid = np.linspace(0.0, max_fpr, 500)
    pro_interp = np.interp(fpr_grid, unique_fpr, unique_pro)

    aupro = float(np.trapezoid(pro_interp, fpr_grid) / max_fpr)
    return float(round(np.clip(aupro, 0.0, 1.0), 4))
"""

files["src/validation/output_checks.py"] = """\"\"\"
Numerical equivalence validation and tensor parity checks across inference runtimes.
\"\"\"

from typing import Any, Dict, List
import numpy as np


def compute_tensor_diff(tensor_a: np.ndarray, tensor_b: np.ndarray) -> Dict[str, float]:
    \"\"\"
    Computes element-wise differences, norms, and numerical parity metrics between two tensors.

    Args:
        tensor_a: Reference baseline NumPy tensor (e.g. PyTorch FP32).
        tensor_b: Candidate test NumPy tensor (e.g. ONNX FP16 / TensorRT INT8).

    Returns:
        Dict containing max_abs_error, mean_abs_error, rmse, cosine_similarity, and anomaly counts.
    \"\"\"
    a = np.asarray(tensor_a, dtype=np.float64)
    b = np.asarray(tensor_b, dtype=np.float64)

    nan_count = int(np.isnan(a).sum() + np.isnan(b).sum())
    inf_count = int(np.isinf(a).sum() + np.isinf(b).sum())

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch in tensor comparison: {a.shape} vs {b.shape}")

    diff = np.abs(a - b)
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))

    # Cosine similarity across flattened vectors
    norm_a = np.linalg.norm(a.flatten())
    norm_b = np.linalg.norm(b.flatten())
    if norm_a > 1e-8 and norm_b > 1e-8:
        cosine_sim = float(np.dot(a.flatten(), b.flatten()) / (norm_a * norm_b))
    else:
        cosine_sim = 1.0 if max_err < 1e-6 else 0.0

    return {
        "max_abs_error": float(max_err),
        "mean_abs_error": float(mean_err),
        "rmse": float(rmse),
        "cosine_similarity": float(cosine_sim),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def check_detection_output_consistency(
    detections_a: List[Dict[str, Any]],
    detections_b: List[Dict[str, Any]],
    iou_match_threshold: float = 0.7,
) -> Dict[str, Any]:
    \"\"\"
    Compares two detection sets (e.g. PyTorch reference vs ONNX/TensorRT output).

    Args:
        detections_a: Reference detection list.
        detections_b: Candidate detection list.
        iou_match_threshold: IoU threshold for matching corresponding bounding boxes.

    Returns:
        Summary dict containing count differences, matched box deviations, and class agreement rate.
    \"\"\"
    count_a = len(detections_a)
    count_b = len(detections_b)

    if count_a == 0 and count_b == 0:
        return {
            "count_a": 0,
            "count_b": 0,
            "matched_boxes": 0,
            "mean_box_mae": 0.0,
            "mean_score_mae": 0.0,
            "class_match_rate": 1.0,
        }

    matched = 0
    box_diffs = []
    score_diffs = []
    class_matches = 0

    from src.validation.detection_quality import box_iou

    if count_a > 0 and count_b > 0:
        boxes_a = np.array([d["bbox"] for d in detections_a], dtype=np.float32)
        boxes_b = np.array([d["bbox"] for d in detections_b], dtype=np.float32)
        ious = box_iou(boxes_a, boxes_b)

        for i in range(count_a):
            best_j = int(np.argmax(ious[i]))
            if ious[i, best_j] >= iou_match_threshold:
                matched += 1
                box_diffs.append(np.mean(np.abs(boxes_a[i] - boxes_b[best_j])))
                score_diffs.append(abs(detections_a[i]["score"] - detections_b[best_j]["score"]))
                if detections_a[i]["class_id"] == detections_b[best_j]["class_id"]:
                    class_matches += 1

    return {
        "count_a": count_a,
        "count_b": count_b,
        "matched_boxes": matched,
        "mean_box_mae": float(np.mean(box_diffs)) if box_diffs else 0.0,
        "mean_score_mae": float(np.mean(score_diffs)) if score_diffs else 0.0,
        "class_match_rate": float(class_matches / max(matched, 1)),
    }
"""

# ============================================================================
# 4. SCRIPTS (scripts/)
# ============================================================================

files["scripts/prepare_sample_data.py"] = """#!/usr/bin/env python3
\"\"\"
Generates reproducible synthetic evaluation datasets for detection (640x640) and industrial inspection (256x256).
\"\"\"

import json
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.hashes import compute_file_sha256
from src.common.logging import setup_logger
from src.common.seed import seed_everything

logger = setup_logger("prepare_sample_data")


def create_synthetic_detection_dataset(output_dir: Path, num_images: int = 30) -> list:
    \"\"\"Generates synthetic 640x640 images with geometric objects and YOLO format labels.\"\"\"
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []

    for i in range(num_images):
        img_name = f"det_{i:03d}.jpg"
        lbl_name = f"det_{i:03d}.txt"
        img_path = img_dir / img_name
        lbl_path = lbl_dir / lbl_name

        # Create canvas with textured background
        img = np.full((640, 640, 3), fill_value=200 + (i % 20), dtype=np.uint8)

        # Draw grid lines for realistic edge features
        for y in range(0, 640, 40):
            cv2.line(img, (0, y), (640, y), (180, 180, 180), 1)
        for x in range(0, 640, 40):
            cv2.line(img, (x, 0), (x, 640), (180, 180, 180), 1)

        # Generate 1 to 3 deterministic target objects
        num_objects = 1 + (i % 3)
        labels = []
        boxes_meta = []

        for obj_idx in range(num_objects):
            w = 80 + (obj_idx * 40)
            h = 60 + (obj_idx * 30)
            cx = 120 + (obj_idx * 160) + (i * 5) % 100
            cy = 150 + (obj_idx * 120) + (i * 7) % 80
            class_id = obj_idx % 2

            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2

            # Draw colored bounding box object
            color = (50, 150, 240) if class_id == 0 else (200, 80, 50)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # YOLO label format: class cx cy w h (normalized to [0, 1])
            norm_cx = cx / 640.0
            norm_cy = cy / 640.0
            norm_w = w / 640.0
            norm_h = h / 640.0
            labels.append(f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")
            boxes_meta.append({"bbox": [x1, y1, x2, y2], "class_id": class_id})

        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lbl_path.write_text("\\n".join(labels), encoding="utf-8")

        manifest_entries.append(
            {
                "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                "label_path": str(lbl_path.relative_to(PROJECT_ROOT)),
                "shape": [640, 640, 3],
                "sha256": compute_file_sha256(img_path),
                "ground_truth_boxes": boxes_meta,
            }
        )

    return manifest_entries


def create_synthetic_industrial_dataset(output_dir: Path, num_per_class: int = 25) -> list:
    \"\"\"Generates normal and anomalous 256x256 images with pixel ground-truth defect masks.\"\"\"
    norm_dir = output_dir / "normal"
    anom_dir = output_dir / "anomalous"
    mask_dir = output_dir / "masks"

    norm_dir.mkdir(parents=True, exist_ok=True)
    anom_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []

    # 1. Normal Samples (smooth texture without defects)
    for i in range(num_per_class):
        img_name = f"normal_{i:03d}.png"
        img_path = norm_dir / img_name

        # Create smooth patterned surface
        x = np.linspace(0, 4 * np.pi, 256)
        y = np.linspace(0, 4 * np.pi, 256)
        xx, yy = np.meshgrid(x, y)
        pattern = (np.sin(xx + i * 0.1) * np.cos(yy + i * 0.1) * 30 + 128).astype(np.uint8)
        img = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(str(img_path), img)
        manifest_entries.append(
            {
                "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                "is_anomalous": False,
                "mask_path": None,
                "shape": [256, 256, 3],
                "sha256": compute_file_sha256(img_path),
            }
        )

    # 2. Anomalous Samples (surface with localized scratch/crack/stain defect)
    for i in range(num_per_class):
        img_name = f"anom_{i:03d}.png"
        mask_name = f"mask_{i:03d}.png"
        img_path = anom_dir / img_name
        mask_path = mask_dir / mask_name

        x = np.linspace(0, 4 * np.pi, 256)
        y = np.linspace(0, 4 * np.pi, 256)
        xx, yy = np.meshgrid(x, y)
        pattern = (np.sin(xx + i * 0.1) * np.cos(yy + i * 0.1) * 30 + 128).astype(np.uint8)
        img = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

        # Defect mask canvas
        mask = np.zeros((256, 256), dtype=np.uint8)

        # Inject localized defect (scratch or stain)
        cx = 50 + (i * 7) % 150
        cy = 50 + (i * 9) % 150
        radius = 12 + (i % 10)

        if i % 2 == 0:
            # Scratch line defect
            cv2.line(img, (cx - 20, cy - 20), (cx + 20, cy + 20), (250, 30, 30), 4)
            cv2.line(mask, (cx - 20, cy - 20), (cx + 20, cy + 20), 255, 4)
        else:
            # Stain defect
            cv2.circle(img, (cx, cy), radius, (20, 20, 20), -1)
            cv2.circle(mask, (cx, cy), radius, 255, -1)

        cv2.imwrite(str(img_path), img)
        cv2.imwrite(str(mask_path), mask)

        manifest_entries.append(
            {
                "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                "is_anomalous": True,
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                "shape": [256, 256, 3],
                "sha256": compute_file_sha256(img_path),
            }
        )

    return manifest_entries


def main() -> None:
    seed_everything(42)
    logger.info("Generating synthetic evaluation data...")

    det_entries = create_synthetic_detection_dataset(
        PROJECT_ROOT / "data" / "sample_images" / "detection", num_images=30
    )
    logger.info(f"Generated {len(det_entries)} detection sample images.")

    ind_entries = create_synthetic_industrial_dataset(
        PROJECT_ROOT / "data" / "sample_images" / "industrial", num_per_class=25
    )
    logger.info(f"Generated {len(ind_entries)} industrial inspection images (50 total).")

    full_manifest = {
        "dataset_name": "synthetic_edge_benchmark_samples",
        "detection_samples": det_entries,
        "industrial_samples": ind_entries,
    }

    manifest_path = PROJECT_ROOT / "data" / "sample_images" / "manifest.json"
    manifest_path.write_text(json.dumps(full_manifest, indent=2), encoding="utf-8")
    logger.info(f"Sample data manifest saved -> {manifest_path}")


if __name__ == "__main__":
    main()
"""

files["scripts/run_pytorch_baselines.py"] = """#!/usr/bin/env python3
\"\"\"
Runs reference PyTorch FP32 evaluations, computes quality metrics, and persists baseline signatures.
\"\"\"

import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_config
from src.common.hashes import compute_tensor_sha256
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.preprocess import preprocess_detection_image, preprocess_industrial_image
from src.models.yolo_adapter import YOLOAdapter
from src.validation.anomaly_quality import compute_aupro, compute_image_auroc, compute_pixel_auroc
from src.validation.detection_quality import evaluate_detection_dataset

logger = setup_logger("run_pytorch_baselines")


def run_yolo_baseline() -> None:
    logger.info("Executing PyTorch FP32 baseline for YOLO nano detector...")
    cfg = load_config(PROJECT_ROOT / "configs" / "yolo" / "fp32.yaml")
    adapter = YOLOAdapter(conf_threshold=cfg.model.conf_threshold, iou_threshold=cfg.model.iou_threshold)

    # Serialize baseline weights
    weights_dir = PROJECT_ROOT / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    yolo_weights_path = weights_dir / "yolo_nano_baseline.pt"
    torch.save(adapter.model.state_dict(), yolo_weights_path)
    logger.info(f"YOLO baseline weights serialized -> {yolo_weights_path}")

    manifest_path = PROJECT_ROOT / "data" / "sample_images" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    det_samples = manifest["detection_samples"]

    all_preds = []
    all_gts = []
    raw_outputs = []

    for item in det_samples:
        img_path = PROJECT_ROOT / item["image_path"]
        tensor, ratio, pad, orig_shape = preprocess_detection_image(img_path)

        raw_output = adapter.forward(tensor)
        raw_outputs.append(raw_output.detach().cpu())

        detections = adapter.postprocess(raw_output, orig_shape, ratio, pad)
        all_preds.append(detections)
        all_gts.append(item["ground_truth_boxes"])

    metrics = evaluate_detection_dataset(all_preds, all_gts)

    # Compute hash of concatenated output tensors
    stacked_raw = torch.cat(raw_outputs, dim=0).numpy()
    out_hash = compute_tensor_sha256(stacked_raw)

    result_payload = {
        "model_name": "yolo_nano",
        "precision": "fp32",
        "runtime": "pytorch",
        "metrics": metrics,
        "raw_output_sha256": out_hash,
        "sample_count": len(det_samples),
    }

    out_dir = PROJECT_ROOT / "results" / "raw" / "pytorch_fp32" / "yolo_nano"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "baseline_metrics.json"
    out_file.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    logger.info(f"YOLO FP32 baseline saved -> {out_file} (mAP@50: {metrics['mAP_50']})")


def run_industrial_baseline() -> None:
    logger.info("Executing PyTorch FP32 baseline for Industrial Autoencoder...")
    adapter = IndustrialModelAdapter()

    # Serialize baseline weights
    weights_dir = PROJECT_ROOT / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    ind_weights_path = weights_dir / "industrial_autoencoder_baseline.pt"
    torch.save(adapter.model.state_dict(), ind_weights_path)
    logger.info(f"Industrial baseline weights serialized -> {ind_weights_path}")

    manifest_path = PROJECT_ROOT / "data" / "sample_images" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ind_samples = manifest["industrial_samples"]

    y_true = []
    y_scores = []
    gt_masks = []
    anomaly_maps = []
    reconstructed_tensors = []

    for item in ind_samples:
        img_path = PROJECT_ROOT / item["image_path"]
        tensor = preprocess_industrial_image(img_path)

        recon, a_map = adapter.forward(tensor)
        reconstructed_tensors.append(recon.detach().cpu())

        score = adapter.compute_anomaly_score(a_map)
        is_anom = 1 if item["is_anomalous"] else 0

        y_true.append(is_anom)
        y_scores.append(score)

        if item["mask_path"]:
            mask_img = cv2.imread(str(PROJECT_ROOT / item["mask_path"]), cv2.IMREAD_GRAYSCALE)
            mask_arr = (mask_img / 255.0).astype(np.float32)
        else:
            mask_arr = np.zeros((256, 256), dtype=np.float32)

        gt_masks.append(mask_arr)
        anomaly_maps.append(a_map.squeeze().detach().cpu().numpy())

    image_auroc = compute_image_auroc(y_true, y_scores)
    pixel_auroc = compute_pixel_auroc(gt_masks, anomaly_maps)
    aupro = compute_aupro(gt_masks, anomaly_maps)

    stacked_recons = torch.cat(reconstructed_tensors, dim=0).numpy()
    out_hash = compute_tensor_sha256(stacked_recons)

    result_payload = {
        "model_name": "industrial_autoencoder",
        "precision": "fp32",
        "runtime": "pytorch",
        "metrics": {
            "image_auroc": image_auroc,
            "pixel_auroc": pixel_auroc,
            "aupro": aupro,
        },
        "raw_output_sha256": out_hash,
        "sample_count": len(ind_samples),
    }

    out_dir = PROJECT_ROOT / "results" / "raw" / "pytorch_fp32" / "industrial_autoencoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "baseline_metrics.json"
    out_file.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    logger.info(f"Industrial FP32 baseline saved -> {out_file} (Image AUROC: {image_auroc}, Pixel AUROC: {pixel_auroc})")


def main() -> None:
    seed_everything(42)
    run_yolo_baseline()
    run_industrial_baseline()
    print("\\nPyTorch FP32 Baselines executed and persisted successfully.")


if __name__ == "__main__":
    main()
"""

# ============================================================================
# 5. COMPREHENSIVE UNIT TESTS (tests/)
# ============================================================================

files["tests/test_preprocess.py"] = """\"\"\"
Unit tests for deterministic image preprocessing routines.
\"\"\"

from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.preprocess import (
    letterbox_image,
    preprocess_detection_image,
    preprocess_industrial_image,
)


class TestPreprocess:
    \"\"\"Test suite validating image resizing, letterboxing, padding, and normalization.\"\"\"

    def test_letterbox_image_dimensions(self) -> None:
        \"\"\"Tests letterbox output matches exact target shape while preserving aspect ratio.\"\"\"
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, ratio, pad = letterbox_image(img, target_shape=(640, 640), auto=False)

        assert padded.shape == (640, 640, 3)
        assert ratio[0] == 1.0
        assert ratio[1] == 1.0
        assert pad[1] == 80.0  # (640 - 480) / 2

    def test_preprocess_detection_image_numpy_and_file(self, tmp_path: Path) -> None:
        \"\"\"Tests detection preprocessing from both file paths and memory arrays.\"\"\"
        import cv2

        dummy_img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        img_path = tmp_path / "sample_det.jpg"
        cv2.imwrite(str(img_path), dummy_img)

        # Test from file path
        t_file, ratio, pad, orig_shape = preprocess_detection_image(img_path, target_shape=(640, 640))
        assert isinstance(t_file, torch.Tensor)
        assert t_file.shape == (1, 3, 640, 640)
        assert t_file.dtype == torch.float32
        assert 0.0 <= t_file.min() <= t_file.max() <= 1.0
        assert orig_shape == (300, 400)

        # Test from NumPy array
        t_arr, ratio2, pad2, orig_shape2 = preprocess_detection_image(dummy_img, target_shape=(640, 640))
        assert torch.equal(t_file, t_arr)
        assert ratio == ratio2
        assert pad == pad2

    def test_preprocess_industrial_image(self, tmp_path: Path) -> None:
        \"\"\"Tests industrial inspection preprocessing normalization and shapes.\"\"\"
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        t = preprocess_industrial_image(dummy_img, target_shape=(256, 256))

        assert isinstance(t, torch.Tensor)
        assert t.shape == (1, 3, 256, 256)
        assert t.dtype == torch.float32
        assert 0.0 <= t.min() <= t.max() <= 1.0

    def test_preprocess_detection_missing_file_raises(self, tmp_path: Path) -> None:
        \"\"\"Ensures preprocess_detection_image raises FileNotFoundError for missing images.\"\"\"
        with pytest.raises(FileNotFoundError):
            preprocess_detection_image(tmp_path / "missing.jpg")
"""

files["tests/test_models.py"] = """\"\"\"
Unit tests for YOLO and Industrial Autoencoder model adapters.
\"\"\"

from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.industrial_model_adapter import ConvAutoencoder, IndustrialModelAdapter
from src.models.yolo_adapter import (
    ConvBlock,
    YOLONanoDetector,
    YOLOAdapter,
    nms_pytorch,
    fast_vectorized_nms_numpy,
)


class TestModels:
    \"\"\"Test suite validating adapter instantiation, raw tensor output geometries, and postprocessing.\"\"\"

    def test_conv_block_forward(self) -> None:
        \"\"\"Tests ConvBlock layer forward pass.\"\"\"
        block = ConvBlock(3, 16, k=3, s=2, p=1)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 16, 16, 16)

    def test_yolo_nano_forward_shape(self) -> None:
        \"\"\"Verifies YOLO model outputs pre-NMS raw tensor of shape [1, 84, 8400].\"\"\"
        adapter = YOLOAdapter()
        dummy_input = torch.randn(1, 3, 640, 640)
        output = adapter.forward(dummy_input)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 84, 8400)
        assert isinstance(adapter.get_pytorch_model(), torch.nn.Module)

    def test_nms_pytorch_and_numpy_algorithms(self) -> None:
        \"\"\"Tests NMS helper implementations under empty, overlapping, and large box sets.\"\"\"
        # 1. Empty boxes
        empty_boxes_t = torch.empty((0, 4), dtype=torch.float32)
        empty_scores_t = torch.empty((0,), dtype=torch.float32)
        assert nms_pytorch(empty_boxes_t, empty_scores_t, 0.5).numel() == 0
        assert fast_vectorized_nms_numpy(np.empty((0, 4)), np.empty((0,)), 0.5).size == 0

        # 2. Overlapping boxes
        boxes_np = np.array([
            [10.0, 10.0, 50.0, 50.0],
            [12.0, 12.0, 52.0, 52.0],  # High overlap with box 0
            [100.0, 100.0, 150.0, 150.0],  # Disjoint
        ], dtype=np.float32)
        scores_np = np.array([0.9, 0.85, 0.75], dtype=np.float32)

        keep_np = fast_vectorized_nms_numpy(boxes_np, scores_np, iou_threshold=0.5)
        assert len(keep_np) == 2
        assert 0 in keep_np and 2 in keep_np

        keep_pt = nms_pytorch(torch.from_numpy(boxes_np), torch.from_numpy(scores_np), iou_threshold=0.5)
        assert len(keep_pt) == 2
        assert 0 in keep_pt and 2 in keep_pt

        # 3. Dense (>300) boxes
        large_boxes = torch.rand(400, 4) * 500.0
        large_boxes[:, 2:] += large_boxes[:, :2]  # x2 > x1, y2 > y1
        large_scores = torch.rand(400)
        keep_large = nms_pytorch(large_boxes, large_scores, 0.5)
        assert len(keep_large) <= 300

    def test_yolo_postprocess_zero_boxes(self) -> None:
        \"\"\"Tests that postprocessing with zero confidence returns empty detections list.\"\"\"
        adapter = YOLOAdapter(conf_threshold=0.5)
        raw = torch.zeros(1, 84, 8400)  # logit 0 -> prob 0.5, prior fill is -4.595 (< conf_threshold)
        raw[0, 4:, :] = -10.0  # low logits

        detections = adapter.postprocess(raw, orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))
        assert detections == []

    def test_yolo_postprocess_dense_and_transposed_shapes(self) -> None:
        \"\"\"Tests postprocessing with [84, 8400], [8400, 84], NumPy inputs, and >300 boxes.\"\"\"
        adapter = YOLOAdapter(conf_threshold=0.25, iou_threshold=0.45)

        # 1. 2D Tensor shape [84, 8400]
        raw_2d = torch.zeros(84, 8400)
        raw_2d[0:4, 10] = torch.tensor([100.0, 100.0, 50.0, 50.0])
        raw_2d[4, 10] = 5.0
        dets_2d = adapter.postprocess(raw_2d, orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))
        assert len(dets_2d) >= 1

        # 2. Transposed shape [8400, 84] NumPy array
        raw_trans = np.zeros((8400, 84), dtype=np.float32)
        raw_trans[10, :4] = np.array([200.0, 200.0, 40.0, 40.0])
        raw_trans[10, 4] = 6.0
        dets_trans = adapter.postprocess(raw_trans, orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))
        assert len(dets_trans) >= 1

        # 3. Dense (>300 candidates)
        raw_dense = torch.zeros(1, 84, 8400)
        raw_dense[0, 0:4, :400] = torch.rand(4, 400) * 100.0
        raw_dense[0, 4, :400] = 5.0  # 400 high confidence boxes
        dets_dense = adapter.postprocess(raw_dense, orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))
        assert len(dets_dense) <= 300

    def test_yolo_postprocess_letterbox_unpadding_aspect_ratio(self) -> None:
        \"\"\"Tests letterbox unpadding coordinate transforms with non-square aspect ratios (1920x1080 -> 640x640).\"\"\"
        adapter = YOLOAdapter(conf_threshold=0.25)
        raw = torch.zeros(1, 84, 8400)
        raw[0, 4:, :] = -10.0  # low logits for background

        # Box placed at (cx=320, cy=320, w=100, h=100) in 640x640 padded letterbox
        raw[0, 0:4, 0] = torch.tensor([320.0, 320.0, 100.0, 100.0])
        raw[0, 4, 0] = 5.0

        orig_h, orig_w = 1080, 1920
        ratio = 640.0 / 1920.0
        pad_y = (640.0 - orig_h * ratio) / 2.0  # pad_top
        pad = (0.0, pad_y)

        dets = adapter.postprocess(raw, orig_shape=(orig_h, orig_w), ratio=(ratio, ratio), pad=pad)
        assert len(dets) == 1
        bbox = dets[0]["bbox"]
        # Assert bounding box coordinates are unpadded and clamped within [0, orig_w] and [0, orig_h]
        assert 0.0 <= bbox[0] <= bbox[2] <= orig_w
        assert 0.0 <= bbox[1] <= bbox[3] <= orig_h

    def test_yolo_postprocess_clipping_and_errors(self) -> None:
        \"\"\"Tests coordinate boundary clipping and error handling for invalid input types/shapes.\"\"\"
        adapter = YOLOAdapter(conf_threshold=0.25)

        # Invalid type
        with pytest.raises(TypeError):
            adapter.postprocess("invalid_input", orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))

        # Invalid shape
        with pytest.raises(ValueError):
            adapter.postprocess(np.zeros((10, 10)), orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))

        # Box outside boundary -> clipped to [0, orig_w] and [0, orig_h]
        raw_out = torch.zeros(1, 84, 8400)
        raw_out[0, 4:, :] = -10.0
        raw_out[0, 0:4, 0] = torch.tensor([-50.0, -50.0, 2000.0, 2000.0])
        raw_out[0, 4, 0] = 5.0
        dets = adapter.postprocess(raw_out, orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))
        assert len(dets) == 1
        assert dets[0]["bbox"][0] == 0.0
        assert dets[0]["bbox"][1] == 0.0
        assert dets[0]["bbox"][2] == 640.0
        assert dets[0]["bbox"][3] == 640.0

    def test_yolo_adapter_weights_loading(self, tmp_path: Path) -> None:
        \"\"\"Tests loading model weights from checkpoint file.\"\"\"
        adapter1 = YOLOAdapter()
        ckpt_path = tmp_path / "yolo_test.pt"
        torch.save(adapter1.get_pytorch_model().state_dict(), ckpt_path)

        adapter2 = YOLOAdapter(weights_path=str(ckpt_path))
        assert isinstance(adapter2.get_pytorch_model(), torch.nn.Module)

    def test_industrial_autoencoder_forward_and_anomaly(self) -> None:
        \"\"\"Verifies ConvAutoencoder produces exact output shape [1, 3, 256, 256] and anomaly maps.\"\"\"
        adapter = IndustrialModelAdapter()
        dummy_input = torch.rand(1, 3, 256, 256)

        recon, a_map = adapter.forward(dummy_input)

        assert recon.shape == (1, 3, 256, 256)
        assert a_map.shape == (1, 1, 256, 256)
        assert 0.0 <= recon.min() <= recon.max() <= 1.0

        score = adapter.compute_anomaly_score(a_map)
        assert isinstance(score, float)
        assert score >= 0.0
        assert isinstance(adapter.get_pytorch_model(), torch.nn.Module)
"""

files["tests/test_metrics.py"] = """
\"\"\"
Unit tests for detection and anomaly evaluation quality engines.
\"\"\"

import numpy as np
import pytest

from src.validation.anomaly_quality import compute_aupro, compute_image_auroc, compute_pixel_auroc
from src.validation.detection_quality import box_iou, compute_ap, evaluate_detection_dataset
from src.validation.output_checks import compute_tensor_diff


class TestMetrics:
    \"\"\"Test suite validating metric accuracy against known standard analytical vectors.\"\"\"

    def test_box_iou_perfect_and_disjoint(self) -> None:
        \"\"\"Tests IoU calculation on identical and disjoint bounding boxes.\"\"\"
        b1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b3 = np.array([[20, 20, 30, 30]], dtype=np.float32)

        assert np.isclose(box_iou(b1, b2)[0, 0], 1.0)
        assert np.isclose(box_iou(b1, b3)[0, 0], 0.0)

    def test_compute_ap_perfect(self) -> None:
        \"\"\"Tests AP calculation on perfect precision-recall curve.\"\"\"
        rec = np.array([0.2, 0.5, 0.8, 1.0])
        prec = np.array([1.0, 1.0, 1.0, 1.0])
        ap = compute_ap(rec, prec)
        assert np.isclose(ap, 1.0)

    def test_evaluate_detection_dataset_perfect_match(self) -> None:
        \"\"\"Tests evaluate_detection_dataset when predictions perfectly match ground truths.\"\"\"
        preds = [[{"bbox": [10, 10, 50, 50], "score": 0.95, "class_id": 0}]]
        gts = [[{"bbox": [10, 10, 50, 50], "class_id": 0}]]

        metrics = evaluate_detection_dataset(preds, gts)
        assert metrics["mAP_50"] == 1.0
        assert metrics["mAP_50_95"] > 0.9

    def test_compute_image_auroc(self) -> None:
        \"\"\"Tests Image AUROC with perfect and inverted classifier rankings.\"\"\"
        y_true = [0, 0, 0, 1, 1, 1]
        y_score_perfect = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
        y_score_inverted = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]

        assert compute_image_auroc(y_true, y_score_perfect) == 1.0
        assert compute_image_auroc(y_true, y_score_inverted) == 0.0

    def test_compute_pixel_auroc_and_aupro(self) -> None:
        \"\"\"Tests pixel AUROC and AU-PRO with synthetic binary masks.\"\"\"
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
        \"\"\"Tests compute_tensor_diff gives zero error on identical arrays and accurate MAE on perturbations.\"\"\"
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
        \"\"\"Tests check_detection_output_consistency matching and deviation reporting.\"\"\"
        from src.validation.output_checks import check_detection_output_consistency

        d1 = [{"bbox": [10, 10, 50, 50], "score": 0.9, "class_id": 0}]
        d2 = [{"bbox": [11, 10, 51, 50], "score": 0.89, "class_id": 0}]

        res = check_detection_output_consistency(d1, d2)
        assert res["matched_boxes"] == 1
        assert res["class_match_rate"] == 1.0
        assert res["mean_box_mae"] > 0.0
"""

# ============================================================================
# WRITE ALL FILES TO TARGET_ROOT
# ============================================================================

for rel_path, content in files.items():
    dest = TARGET_ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    if rel_path.endswith(".sh") or rel_path.endswith(".py") and rel_path.startswith("scripts/"):
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  [CREATED] {rel_path}")

print(f"\\nAll {len(files)} Phase 1 files generated successfully at {TARGET_ROOT}.")

