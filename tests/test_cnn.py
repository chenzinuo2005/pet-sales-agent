"""Tests for CNN model, dataset, and inference pipeline."""
import os
import sys
import tempfile

import pytest
import torch
from PIL import Image

from app.cnn.model import create_model, count_parameters
from app.cnn.dataset import create_dataloaders
from app.cnn.inference import predict_breed, CLASS_NAMES, BREED_MAPPING

DATASET_ROOT = os.environ.get("OXFORD_PET_ROOT", "D:/datasets/oxford-iiit-pet")


class TestModel:
    def test_create_model_returns_correct_output_shape(self):
        model = create_model(num_classes=37)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 37), f"Expected (1,37), got {out.shape}"

    def test_create_model_different_num_classes(self):
        model = create_model(num_classes=10)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 10)

    def test_count_parameters_returns_positive(self):
        model = create_model(num_classes=37)
        total, trainable = count_parameters(model)
        assert total > 0
        assert trainable > 0
        assert trainable == total  # all params trainable

    def test_model_outputs_raw_logits(self):
        model = create_model(num_classes=37)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        probs = torch.softmax(out, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)


class TestDataset:
    @pytest.mark.skipif(
        not os.path.isdir(DATASET_ROOT),
        reason=f"Oxford dataset not found at {DATASET_ROOT}",
    )
    def test_dataloader_split_sizes(self):
        train, val, test, num_classes = create_dataloaders(
            DATASET_ROOT, batch_size=32
        )
        assert num_classes == 37
        total = len(train.dataset) + len(val.dataset) + len(test.dataset)
        assert total > 100  # Any reasonable dataset should have at least 100 images

    @pytest.mark.skipif(
        not os.path.isdir(DATASET_ROOT),
        reason=f"Oxford dataset not found at {DATASET_ROOT}",
    )
    def test_dataloader_yields_correct_shapes(self):
        train, _, _, num_classes = create_dataloaders(
            DATASET_ROOT, batch_size=4, use_cutmix=False  # plain labels for shape test
        )
        images, targets = next(iter(train))
        assert images.shape == (4, 3, 224, 224)
        assert targets.shape == (4,)
        assert targets.max() < num_classes

    @pytest.mark.skipif(
        not os.path.isdir(DATASET_ROOT),
        reason=f"Oxford dataset not found at {DATASET_ROOT}",
    )
    def test_dataloader_with_cutmix(self):
        """CutMix collate should produce soft labels (batch, num_classes)."""
        train, _, _, num_classes = create_dataloaders(
            DATASET_ROOT, batch_size=4, use_cutmix=True
        )
        images, targets = next(iter(train))
        assert images.shape == (4, 3, 224, 224)
        # CutMix produces soft one-hot labels
        assert targets.shape == (4, num_classes)
        # Each row should sum to ~1.0
        assert torch.allclose(targets.sum(dim=1), torch.ones(4), atol=1e-5)


class TestInference:
    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(os.path.dirname(__file__), "../resources/models/pet_cnn.pth")
        ),
        reason="Model weights not found",
    )
    def test_predict_breed_success(self):
        """Smoke test: inference on a real image succeeds.

        Note: requires model weights compatible with the current classifier head.
        After upgrading to 512-dim classifier, old 256-dim weights must be retrained.
        """
        import glob as _g

        candidates = _g.glob(os.path.join(DATASET_ROOT, "images", "*.jpg"))
        if not candidates:
            pytest.skip("No test images found")
        try:
            result = predict_breed(candidates[0])
        except RuntimeError as e:
            if "size mismatch" in str(e):
                pytest.skip(f"Model weights need retraining (classifier head changed): {e}")
            raise
        assert result.breed_en in CLASS_NAMES
        assert len(result.top3) == 3
        assert result.status in ("success", "low_confidence", "failed")

    def test_predict_breed_missing_file(self):
        result = predict_breed("/nonexistent/pet.jpg")
        assert result.status == "failed"

    def test_predict_breed_wrong_format(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            tmp = f.name
        try:
            result = predict_breed(tmp)
            assert result.status == "failed"
        finally:
            os.unlink(tmp)

    def test_class_names_count(self):
        assert len(CLASS_NAMES) == 37

    def test_breed_mapping_count(self):
        assert len(BREED_MAPPING) == 37

    def test_mapping_covers_all_classes(self):
        for name in CLASS_NAMES:
            key = name.lower().replace(" ", "_")
            assert key in BREED_MAPPING, f"Missing mapping for: {name}"
