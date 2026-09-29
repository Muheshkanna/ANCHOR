"""Unit tests for the model_integrity package detectors and loader."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from anchor.model_integrity.activation_stats import compare_activation_stats, compute_activation_stats
from anchor.model_integrity.fingerprint import check_fingerprint
from anchor.model_integrity.model_loader import LoadedModel
from anchor.model_integrity.trigger_reconstruction import reconstruct_triggers
from anchor.report_schema.schema import AccessMode


# Paths to generated assets
REFERENCE_DIR = Path(__file__).parents[1] / "reference_data"
MODEL_PT_PATH = REFERENCE_DIR / "backdoored_model.pt"
MODEL_ONNX_PATH = REFERENCE_DIR / "backdoored_model.onnx"
CLEAN_DATA_DIR = REFERENCE_DIR / "clean" / "images"


@pytest.fixture
def clean_images_batch() -> np.ndarray:
    """Load a batch of 16 clean images from reference_data."""
    if not CLEAN_DATA_DIR.exists():
        pytest.skip("Clean dataset not found. Run generate_poisoned_data.py first.")
        
    img_paths = sorted(list(CLEAN_DATA_DIR.glob("*.png")))[:16]
    images = []
    
    for path in img_paths:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize and transpose to CHW
        img_norm = img.astype(np.float32) / 255.0
        # Standard CIFAR normalize to match model loader expectations
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
        img_norm = (img_norm.transpose(2, 0, 1) - mean) / std
        images.append(img_norm)
        
    return np.array(images)


# ── Model Loader Tests ────────────────────────────────────────────────────────

def test_model_loader_torchscript() -> None:
    if not MODEL_PT_PATH.exists():
        pytest.skip("TorchScript backdoored model not found.")
        
    model = LoadedModel(MODEL_PT_PATH)
    assert model.is_white_box is True
    assert model.torch_model is not None
    assert model.ort_session is None

    # Test prediction shape
    x = np.random.randn(2, 3, 32, 32).astype(np.float32)
    logits = model.predict(x)
    assert logits.shape == (2, 10)


def test_model_loader_onnx() -> None:
    if not MODEL_ONNX_PATH.exists():
        pytest.skip("ONNX backdoored model not found.")
        
    model = LoadedModel(MODEL_ONNX_PATH)
    assert model.is_white_box is False
    assert model.torch_model is None
    assert model.ort_session is not None

    # Test prediction shape
    x = np.random.randn(2, 3, 32, 32).astype(np.float32)
    logits = model.predict(x)
    assert logits.shape == (2, 10)


# ── Fingerprint Tests ─────────────────────────────────────────────────────────

def test_fingerprint_check(clean_images_batch: np.ndarray) -> None:
    if not MODEL_PT_PATH.exists():
        pytest.skip("TorchScript model not found.")
        
    model = LoadedModel(MODEL_PT_PATH)
    
    # Calculate baseline probabilities
    logits = model.predict(clean_images_batch)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    baseline_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Identical model should pass fingerprinting (no flags)
    flags = check_fingerprint(model, clean_images_batch, baseline_probs, threshold=0.10)
    assert len(flags) == 0

    # Perturbed baseline should trigger fingerprint mismatch flag
    perturbed_probs = baseline_probs.copy()
    perturbed_probs[:, 0] = 1.0 - perturbed_probs[:, 0]  # Flip probability
    perturbed_probs = perturbed_probs / np.sum(perturbed_probs, axis=-1, keepdims=True)

    flags_perturbed = check_fingerprint(model, clean_images_batch, perturbed_probs, threshold=0.10)
    assert len(flags_perturbed) == 1
    assert "fingerprint mismatch" in flags_perturbed[0].reason
    assert flags_perturbed[0].evidence["access_mode_used"] == AccessMode.WHITE_BOX


# ── Activation Stats Tests ────────────────────────────────────────────────────

def test_activation_stats_white_box(clean_images_batch: np.ndarray) -> None:
    if not MODEL_PT_PATH.exists():
        pytest.skip("TorchScript model not found.")
        
    model = LoadedModel(MODEL_PT_PATH)
    
    # 1. JIT Skip verification
    # JIT Compiled models must return a skip flag rather than throwing an exception
    flags_jit = compare_activation_stats(model, clean_images_batch, {})
    assert len(flags_jit) == 1
    assert flags_jit[0].evidence["status"] == "unavailable_for_jit"
    assert flags_jit[0].evidence["access_mode_used"] == AccessMode.WHITE_BOX

    # 2. Functional check on standard eager module
    class SimpleEagerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.fc = nn.Linear(4 * 32 * 32, 10)

        def forward(self, x):
            x = self.conv(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    class MockLoadedModel:
        def __init__(self, torch_model):
            self.torch_model = torch_model
            self.is_white_box = True
            self.model_path = Path("mock_model.pt")

    eager_model = SimpleEagerModel().eval()
    mock_model = MockLoadedModel(eager_model)

    # Compute reference stats
    ref_stats = compute_activation_stats(mock_model, clean_images_batch)
    assert len(ref_stats) == 2
    assert "conv" in ref_stats
    assert "fc" in ref_stats

    # Compare stats to itself (should not flag)
    flags = compare_activation_stats(mock_model, clean_images_batch, ref_stats, threshold=0.10)
    assert len(flags) == 0

    # Tamper with reference stats to force anomaly detection
    corrupted_stats = {k: v.copy() for k, v in ref_stats.items()}
    corrupted_stats["conv"]["mean"] += 10.0  # Big change

    flags_corrupt = compare_activation_stats(mock_model, clean_images_batch, corrupted_stats, threshold=0.10)
    assert len(flags_corrupt) >= 1
    assert "conv" in flags_corrupt[0].reason


def test_activation_stats_onnx_skipped(clean_images_batch: np.ndarray) -> None:
    if not MODEL_ONNX_PATH.exists():
        pytest.skip("ONNX model not found.")
        
    model = LoadedModel(MODEL_ONNX_PATH)
    
    # Run stats check on black-box ONNX model
    dummy_stats: dict[str, dict[str, float]] = {}
    flags = compare_activation_stats(model, clean_images_batch, dummy_stats)
    
    # Must return a skipped flag pointing out ONNX unavailability rather than erroring
    assert len(flags) == 1
    flag = flags[0]
    assert flag.evidence["status"] == "unavailable"
    assert flag.evidence["access_mode_used"] == AccessMode.BLACK_BOX


# ── Trigger Reconstruction Tests ──────────────────────────────────────────────

def test_trigger_reconstruction_detects_backdoor(clean_images_batch: np.ndarray) -> None:
    if not MODEL_PT_PATH.exists():
        pytest.skip("TorchScript backdoored model not found.")
        
    model = LoadedModel(MODEL_PT_PATH)
    
    # Optimize triggers for 10 classes
    # We use fewer steps (40) for unit test speed
    flags = reconstruct_triggers(
        model, 
        clean_images_batch, 
        num_classes=10, 
        steps=60, 
        cost_lambda=0.01,
        anomaly_threshold=2.0
    )
    
    # The backdoor triggers airplane (class 0)
    assert len(flags) > 0
    
    # Ensure class 0 is flagged as the anomaly
    flagged_classes = [f.evidence["target_class"] for f in flags]
    assert 0 in flagged_classes
    assert flags[0].evidence["access_mode_used"] == AccessMode.WHITE_BOX


def test_trigger_reconstruction_onnx_skipped(clean_images_batch: np.ndarray) -> None:
    if not MODEL_ONNX_PATH.exists():
        pytest.skip("ONNX model not found.")
        
    model = LoadedModel(MODEL_ONNX_PATH)
    
    flags = reconstruct_triggers(model, clean_images_batch, num_classes=10)
    
    # Must report check was skipped due to lack of white-box access
    assert len(flags) == 1
    flag = flags[0]
    assert flag.evidence["status"] == "unavailable"
    assert flag.evidence["access_mode_used"] == AccessMode.BLACK_BOX
