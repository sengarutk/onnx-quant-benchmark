"""
Industrial Convolutional Autoencoder adapter for anomaly reconstruction and defect score computation.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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

    def compute_anomaly_score(self, anomaly_map: Union[torch.Tensor, np.ndarray], top_k_ratio: float = 0.01) -> float:
        """
        Computes aggregate image-level anomaly score from the top-k highest reconstruction error pixels.

        Args:
            anomaly_map: Anomaly tensor or NumPy array of shape [1, 1, H, W] or [H, W].
            top_k_ratio: Proportion of highest-error pixels to average (default: 0.01).

        Returns:
            Scalar anomaly score float >= 0.0.
        """
        if isinstance(anomaly_map, torch.Tensor):
            flat = anomaly_map.detach().cpu().view(-1)
            k = max(1, int(flat.numel() * top_k_ratio))
            topk_vals, _ = torch.topk(flat, k)
            return float(topk_vals.mean().item())

        flat = np.asarray(anomaly_map, dtype=np.float32).ravel()
        k = max(1, int(flat.size * top_k_ratio))
        partitioned = np.partition(flat, -k)[-k:]
        return float(np.mean(partitioned))
