"""
Industrial Convolutional Autoencoder adapter for anomaly reconstruction and defect score computation.
"""

from pathlib import Path
import json
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


SCORING_PARAMS_PATH = PROJECT_ROOT / "models" / "weights" / "industrial_autoencoder_scoring.json"
DEFAULT_SCORING = {"blur_sigma": 4.0, "top_k_ratio": 0.01}


def load_scoring_params() -> Dict[str, float]:
    """Anomaly-scoring hyper-parameters selected on the validation split by scripts/train_models.py."""
    if SCORING_PARAMS_PATH.is_file():
        data = json.loads(SCORING_PARAMS_PATH.read_text(encoding="utf-8"))
        return {"blur_sigma": float(data["blur_sigma"]), "top_k_ratio": float(data["top_k_ratio"])}
    return dict(DEFAULT_SCORING)


def top_k_anomaly_score(
    anomaly_map: Union[torch.Tensor, np.ndarray],
    top_k_ratio: Optional[float] = None,
    blur_sigma: Optional[float] = None,
) -> float:
    """
    Image-level anomaly score: Gaussian-smooth the per-pixel absolute reconstruction error map (sigma in pixels,
    0 disables smoothing), then average its top-k% values. Defaults come from load_scoring_params().
    """
    params = load_scoring_params()
    k_ratio = params["top_k_ratio"] if top_k_ratio is None else top_k_ratio
    sigma = params["blur_sigma"] if blur_sigma is None else blur_sigma
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.detach().float().cpu().numpy()
    amap = np.asarray(anomaly_map, dtype=np.float32)
    amap = amap.reshape(amap.shape[-2:]) if amap.ndim > 2 else amap
    if sigma > 0:
        amap = cv2.GaussianBlur(amap, (0, 0), sigma)
    flat = amap.ravel()
    k = max(1, int(flat.size * k_ratio))
    return float(np.mean(np.partition(flat, -k)[-k:]))


class ConvAutoencoder(nn.Module):
    """
    Exportable, deterministic Convolutional Autoencoder for industrial anomaly reconstruction.
    Maps input tensor [B, 3, 256, 256] -> reconstruction [B, 3, 256, 256].
    """

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
    """Adapter managing model lifecycle, forward execution, and anomaly map extraction."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
        device: str = "cpu",
        require_weights: bool = False,
    ):
        self.input_shape = input_shape
        self.device = torch.device(device)

        self.model = ConvAutoencoder()
        target = Path(weights_path) if weights_path else PROJECT_ROOT / "models" / "weights" / "industrial_autoencoder_trained.pt"
        self.weights_loaded = False
        if target.is_file():
            ckpt = torch.load(str(target), map_location="cpu", weights_only=True)
            state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(state_dict, strict=True)
            self.weights_loaded = True
        elif require_weights or weights_path:
            raise FileNotFoundError(f"Autoencoder weights not found: {target}. Run scripts/train_models.py first.")

        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def get_pytorch_model(self) -> nn.Module:
        """Exposes the underlying ConvAutoencoder Module for ONNX/TensorRT export."""
        return self.model

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes reconstruction and computes the residual anomaly map.

        Returns:
            Tuple of (reconstructed_tensor [B, 3, H, W], anomaly_map [B, 1, H, W]).
        """
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            reconstruction = self.model(tensor)
            # Element-wise absolute difference across RGB channels -> [B, 1, H, W]
            anomaly_map = torch.mean(torch.abs(tensor - reconstruction), dim=1, keepdim=True)
            return reconstruction, anomaly_map

    def compute_anomaly_score(self, anomaly_map: Union[torch.Tensor, np.ndarray], top_k_ratio: Optional[float] = None) -> float:
        """Computes the image-level anomaly score (see top_k_anomaly_score)."""
        return top_k_anomaly_score(anomaly_map, top_k_ratio)
