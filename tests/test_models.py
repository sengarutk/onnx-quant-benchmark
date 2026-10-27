"""
Unit tests for YOLO and Industrial Autoencoder model adapters.
"""

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
    """Test suite validating adapter instantiation, raw tensor output geometries, and postprocessing."""

    def test_conv_block_forward(self) -> None:
        """Tests ConvBlock layer forward pass."""
        block = ConvBlock(3, 16, k=3, s=2, p=1)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 16, 16, 16)

    def test_yolo_nano_forward_shape(self) -> None:
        """Verifies YOLO model outputs pre-NMS raw tensor of shape [1, 6, 8400]."""
        adapter = YOLOAdapter()
        dummy_input = torch.randn(1, 3, 640, 640)
        output = adapter.forward(dummy_input)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 6, 8400)
        assert isinstance(adapter.get_pytorch_model(), torch.nn.Module)

    def test_nms_pytorch_and_numpy_algorithms(self) -> None:
        """Tests NMS helper implementations under empty, overlapping, and large box sets."""
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
        """Tests that postprocessing with zero confidence returns empty detections list."""
        adapter = YOLOAdapter(conf_threshold=0.5)
        raw = torch.zeros(1, 6, 8400)  # logit 0 -> prob 0.5, prior fill is -4.595 (< conf_threshold)
        raw[0, 4:, :] = -10.0  # low logits

        detections = adapter.postprocess(raw, orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))
        assert detections == []

    def test_yolo_postprocess_dense_and_transposed_shapes(self) -> None:
        """Tests postprocessing with [6, 8400], [8400, 6], NumPy inputs, and >300 boxes."""
        adapter = YOLOAdapter(conf_threshold=0.25, iou_threshold=0.45)

        # 1. 2D Tensor shape [6, 8400]
        raw_2d = torch.zeros(6, 8400)
        raw_2d[0:4, 10] = torch.tensor([100.0, 100.0, 50.0, 50.0])
        raw_2d[4, 10] = 5.0
        dets_2d = adapter.postprocess(raw_2d, orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))
        assert len(dets_2d) >= 1

        # 2. Transposed shape [8400, 6] NumPy array
        raw_trans = np.zeros((8400, 6), dtype=np.float32)
        raw_trans[10, :4] = np.array([200.0, 200.0, 40.0, 40.0])
        raw_trans[10, 4] = 6.0
        dets_trans = adapter.postprocess(raw_trans, orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))
        assert len(dets_trans) >= 1

        # 3. Dense (>300 candidates)
        raw_dense = torch.zeros(1, 6, 8400)
        raw_dense[0, 0:4, :400] = torch.rand(4, 400) * 100.0
        raw_dense[0, 4, :400] = 5.0  # 400 high confidence boxes
        dets_dense = adapter.postprocess(raw_dense, orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))
        assert len(dets_dense) <= 300

    def test_yolo_postprocess_letterbox_unpadding_aspect_ratio(self) -> None:
        """Tests letterbox unpadding coordinate transforms with non-square aspect ratios (1920x1080 -> 640x640)."""
        adapter = YOLOAdapter(conf_threshold=0.25)
        raw = torch.zeros(1, 6, 8400)
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
        """Tests coordinate boundary clipping and error handling for invalid input types/shapes."""
        adapter = YOLOAdapter(conf_threshold=0.25)

        # Invalid type
        with pytest.raises(TypeError):
            adapter.postprocess("invalid_input", orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))

        # Invalid shape
        with pytest.raises(ValueError):
            adapter.postprocess(np.zeros((10, 10)), orig_shape=(640, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0))

        # Box outside boundary -> clipped to [0, orig_w] and [0, orig_h]
        raw_out = torch.zeros(1, 6, 8400)
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
        """Tests loading model weights from checkpoint file."""
        adapter1 = YOLOAdapter()
        ckpt_path = tmp_path / "yolo_test.pt"
        torch.save(adapter1.get_pytorch_model().state_dict(), ckpt_path)

        adapter2 = YOLOAdapter(weights_path=str(ckpt_path))
        assert isinstance(adapter2.get_pytorch_model(), torch.nn.Module)

    def test_industrial_autoencoder_forward_and_anomaly(self) -> None:
        """Verifies ConvAutoencoder produces exact output shape [1, 3, 256, 256] and anomaly maps."""
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

    def test_detector_decode_geometry(self) -> None:
        """Decoded boxes are centred on grid cells and scale with softplus(ltrb) * stride."""
        model = YOLONanoDetector()
        raw = [torch.zeros(1, 6, 80, 80), torch.zeros(1, 6, 40, 40), torch.zeros(1, 6, 20, 20)]
        out = model.decode(raw)
        assert out.shape == (1, 6, 8400)
        sp0 = float(torch.nn.functional.softplus(torch.tensor(0.0)))
        # First P3 cell centre is (4, 4); zero logits give symmetric extents of softplus(0) * 8 per side
        assert torch.allclose(out[0, :4, 0], torch.tensor([4.0, 4.0, 2 * sp0 * 8, 2 * sp0 * 8]), atol=1e-5)
        # First P5 cell (index 6400 + 1600) has stride 32
        assert torch.allclose(out[0, 2, 8000], torch.tensor(2 * sp0 * 32), atol=1e-4)

    def test_missing_weights_raise_when_required(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            YOLOAdapter(weights_path=str(tmp_path / "missing.pt"))
        with pytest.raises(FileNotFoundError):
            IndustrialModelAdapter(weights_path=str(tmp_path / "missing.pt"))

    def test_postprocess_class_aware_nms(self) -> None:
        """Overlapping boxes of different classes both survive; same-class duplicates are suppressed."""
        adapter = YOLOAdapter(conf_threshold=0.25, iou_threshold=0.45)
        raw = np.full((6, 8400), -10.0, dtype=np.float32)
        raw[:4, 0] = [100, 100, 50, 50]
        raw[:4, 1] = [102, 102, 50, 50]
        raw[:4, 2] = [101, 101, 50, 50]
        raw[4, 0], raw[4, 1], raw[5, 2] = 5.0, 4.0, 3.0
        dets = adapter.postprocess(raw, (640, 640), (1.0, 1.0), (0.0, 0.0))
        assert sorted(d["class_id"] for d in dets) == [0, 1]
        assert {d["class_name"] for d in dets} == {"rectangle", "ellipse"}
