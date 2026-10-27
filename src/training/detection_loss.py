"""
FCOS-style target assignment with Quality Focal Loss (IoU-aware classification, Li et al., GFL 2020) and
GIoU box regression loss for the anchor-free YOLONanoDetector.

The classification target of a positive location is the IoU between its (detached) predicted box and the assigned
ground truth, so class scores also encode localization quality and off-centre partial boxes are down-ranked before
NMS.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.yolo_adapter import STRIDES

# Regression scale range (max of l, t, r, b) handled by each pyramid level.
SCALE_RANGES: Tuple[Tuple[float, float], ...] = ((0.0, 64.0), (64.0, 128.0), (128.0, 1e8))
# Center-sampling radius in strides; None assigns every in-box location at the right scale (selected on the
# validation split: center sampling left interior cells of uniform shapes as contradictory negatives).
CENTER_RADIUS = None


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Summed sigmoid focal loss (Lin et al., 2017)."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * ce * (1 - p_t) ** gamma).sum()


def quality_focal_loss(logits: torch.Tensor, targets: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """Summed Quality Focal Loss: BCE against soft (IoU) targets, modulated by |sigmoid(logit) - target|^beta."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (ce * (p - targets).abs().pow(beta)).sum()


def box_iou_aligned(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    ix = (torch.minimum(a[:, 2], b[:, 2]) - torch.maximum(a[:, 0], b[:, 0])).clamp(min=0)
    iy = (torch.minimum(a[:, 3], b[:, 3]) - torch.maximum(a[:, 1], b[:, 1])).clamp(min=0)
    inter = ix * iy
    union = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + eps
    return inter / union


def generalized_box_iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Mean GIoU loss (Rezatofighi et al., 2019) for aligned (x1, y1, x2, y2) box pairs."""
    ix1 = torch.maximum(pred[:, 0], target[:, 0])
    iy1 = torch.maximum(pred[:, 1], target[:, 1])
    ix2 = torch.minimum(pred[:, 2], target[:, 2])
    iy2 = torch.minimum(pred[:, 3], target[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_p + area_t - inter + eps
    iou = inter / union
    cx1 = torch.minimum(pred[:, 0], target[:, 0])
    cy1 = torch.minimum(pred[:, 1], target[:, 1])
    cx2 = torch.maximum(pred[:, 2], target[:, 2])
    cy2 = torch.maximum(pred[:, 3], target[:, 3])
    enclose = (cx2 - cx1) * (cy2 - cy1) + eps
    return (1 - (iou - (enclose - union) / enclose)).mean()


class DetectionLoss(nn.Module):
    def __init__(self, num_classes: int, box_weight: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight

    @staticmethod
    def _points(raw_maps: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pts, strides, ranges = [], [], []
        for raw, stride, rng in zip(raw_maps, STRIDES, SCALE_RANGES):
            h, w = raw.shape[-2:]
            ys = (torch.arange(h, device=raw.device, dtype=torch.float32) + 0.5) * stride
            xs = (torch.arange(w, device=raw.device, dtype=torch.float32) + 0.5) * stride
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            pts.append(torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1))
            strides.append(torch.full((h * w,), float(stride), device=raw.device))
            ranges.append(torch.tensor(rng, device=raw.device).expand(h * w, 2))
        return torch.cat(pts), torch.cat(strides), torch.cat(ranges)

    def forward(self, raw_maps: List[torch.Tensor], targets: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        points, strides, ranges = self._points(raw_maps)
        flat = torch.cat([m.flatten(2) for m in raw_maps], dim=2).permute(0, 2, 1).float()  # [B, N, 4 + C]
        reg_raw, cls_logits = flat[..., :4], flat[..., 4:]

        cls_targets = torch.zeros_like(cls_logits)
        pred_boxes, gt_boxes = [], []

        for b, tgt in enumerate(targets):
            if tgt.numel() == 0:
                continue
            tgt = tgt.to(points.device)
            gcls, gbox = tgt[:, 0].long(), tgt[:, 1:5]
            px, py = points[:, 0:1], points[:, 1:2]                                  # [N, 1]
            l = px - gbox[:, 0]
            t = py - gbox[:, 1]
            r = gbox[:, 2] - px
            btm = gbox[:, 3] - py
            ltrb = torch.stack([l, t, r, btm], dim=-1)                               # [N, M, 4]
            inside = ltrb.min(dim=-1).values > 0
            gcx = (gbox[:, 0] + gbox[:, 2]) * 0.5
            gcy = (gbox[:, 1] + gbox[:, 3]) * 0.5
            if CENTER_RADIUS is None:
                center = torch.ones_like(inside)
            else:
                rad = strides[:, None] * CENTER_RADIUS
                center = ((px - gcx).abs() < rad) & ((py - gcy).abs() < rad)
            max_ltrb = ltrb.max(dim=-1).values
            in_scale = (max_ltrb >= ranges[:, 0:1]) & (max_ltrb <= ranges[:, 1:2])
            cand = inside & center & in_scale                                        # [N, M]

            area = ((gbox[:, 2] - gbox[:, 0]) * (gbox[:, 3] - gbox[:, 1]))[None].expand_as(cand).clone()
            area[~cand] = float("inf")
            min_area, assigned = area.min(dim=1)
            pos = torch.isfinite(min_area)
            if not pos.any():
                continue

            idx = assigned[pos]
            s = strides[pos][:, None]
            d = F.softplus(reg_raw[b, pos]) * s
            p = points[pos]
            pbox = torch.stack([p[:, 0] - d[:, 0], p[:, 1] - d[:, 1], p[:, 0] + d[:, 2], p[:, 1] + d[:, 3]], 1)
            pos_idx = pos.nonzero(as_tuple=True)[0]
            cls_targets[b, pos_idx, gcls[idx]] = box_iou_aligned(pbox.detach(), gbox[idx]).clamp(min=0.0)
            pred_boxes.append(pbox)
            gt_boxes.append(gbox[idx])

        num_pos = max(sum(len(p) for p in pred_boxes), 1)
        loss_cls = quality_focal_loss(cls_logits, cls_targets) / num_pos
        if pred_boxes:
            loss_box = generalized_box_iou_loss(torch.cat(pred_boxes), torch.cat(gt_boxes))
        else:
            loss_box = reg_raw.sum() * 0.0
        total = loss_cls + self.box_weight * loss_box
        return {"loss": total, "loss_cls": loss_cls.detach(), "loss_box": loss_box.detach(), "num_pos": torch.tensor(num_pos)}
