"""
On-the-fly torch Datasets over the procedural synthetic generators (train split only).
"""

from typing import List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.synthetic import generate_detection_sample, generate_industrial_sample
from src.models.preprocess import letterbox_image


class SyntheticDetectionDataset(Dataset):
    """Letterboxed 640x640 detection samples with boxes mapped into letterboxed pixel space."""

    def __init__(self, length: int, split: str = "train", img_size: int = 640, hflip: bool = True):
        self.length = length
        self.split = split
        self.img_size = img_size
        self.hflip = hflip

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, boxes = generate_detection_sample(self.split, idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        padded, (rw, rh), (pw, ph) = letterbox_image(img, (self.img_size, self.img_size))
        target = np.zeros((len(boxes), 5), dtype=np.float32)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b["bbox"]
            target[i] = [b["class_id"], x1 * rw + pw, y1 * rh + ph, x2 * rw + pw, y2 * rh + ph]
        if self.hflip and idx % 2 == 1:
            padded = padded[:, ::-1]
            x1 = self.img_size - target[:, 3]
            x2 = self.img_size - target[:, 1]
            target[:, 1], target[:, 3] = x1, x2
        tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).float() / 255.0
        return tensor, torch.from_numpy(target)


def detection_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    images = torch.stack([b[0] for b in batch], dim=0)
    return images, [b[1] for b in batch]


class SyntheticTextureDataset(Dataset):
    """Normal (defect-free) 256x256 industrial textures for unsupervised autoencoder training."""

    def __init__(self, length: int, split: str = "train"):
        self.length = length
        self.split = split

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = generate_industrial_sample(self.split, idx, anomalous=False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0
