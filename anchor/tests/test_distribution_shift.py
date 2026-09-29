"""Unit tests for the distribution_shift package."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from anchor.distribution_shift.embedder import embed_batch
from anchor.distribution_shift.shift_detector import detect_shift
from anchor.report_schema.schema import AccessMode


# ── Offline shift detection tests (deterministic) ───────────────────────────

def test_shift_detector_offline() -> None:
    # 50 reference embeddings, 30 test embeddings (16-dim)
    rng = np.random.default_rng(42)
    ref_embs = rng.normal(loc=0.0, scale=0.5, size=(50, 16)).astype(np.float32)
    
    # Test batch with no shift
    test_embs_no_shift = rng.normal(loc=0.0, scale=0.5, size=(30, 16)).astype(np.float32)
    
    # Test batch with significant mean shift
    test_embs_shifted = rng.normal(loc=0.8, scale=0.5, size=(30, 16)).astype(np.float32)

    # 1. No shift check
    flags_no_shift = detect_shift(test_embs_no_shift, ref_embs, mmd_threshold=0.20, ks_ratio_threshold=0.25)
    assert len(flags_no_shift) == 0

    # 2. Shifted check
    flags_shifted = detect_shift(test_embs_shifted, ref_embs, mmd_threshold=0.20, ks_ratio_threshold=0.25)
    assert len(flags_shifted) == 1
    flag = flags_shifted[0]
    assert flag.evidence["mmd_distance"] > 0.20
    assert flag.evidence["ks_significant_dimensions_ratio"] > 0.25
    assert flag.evidence["agreed_test_statistics_count"] == 2  # Both MMD and KS agree
    assert flag.confidence > 0.85
    assert flag.evidence["access_mode_used"] == AccessMode.BLACK_BOX


# ── Full image shift detection tests ──────────────────────────────────────────

def test_shift_detector_with_images(tmp_path: Path) -> None:
    # Set up folders
    ref_dir = tmp_path / "ref"
    test_clean_dir = tmp_path / "test_clean"
    test_shifted_dir = tmp_path / "test_shifted"

    ref_dir.mkdir()
    test_clean_dir.mkdir()
    test_shifted_dir.mkdir()

    rng = np.random.default_rng(42)

    # Generate 15 clean base images (e.g. structured blocks)
    ref_paths = []
    clean_paths = []
    shifted_paths = []

    for i in range(15):
        # Create base image (128x128)
        img = rng.integers(50, 150, size=(128, 128, 3), dtype=np.uint8)
        # Add some shapes to make them structured
        cv2.circle(img, (64, 64), 30, (0, 255, 0), -1)

        # 1. Save to reference
        ref_path = ref_dir / f"ref_{i}.png"
        cv2.imwrite(str(ref_path), img)
        ref_paths.append(ref_path)

        # 2. Save similar to clean test batch
        clean_img = img.copy()
        # Add tiny pixel noise
        noise = rng.normal(0, 2, img.shape).astype(np.int16)
        clean_img = np.clip(clean_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        clean_path = test_clean_dir / f"clean_{i}.png"
        cv2.imwrite(str(clean_path), clean_img)
        clean_paths.append(clean_path)

        # 3. Save perturbed to shifted test batch (heavy blur + brightness change)
        shifted_img = img.copy()
        # Heavy gaussian blur
        shifted_img = cv2.GaussianBlur(shifted_img, (15, 15), 0)
        # Heavy brightness reduction
        shifted_img = (shifted_img.astype(np.float32) * 0.3).astype(np.uint8)

        shifted_path = test_shifted_dir / f"shifted_{i}.png"
        cv2.imwrite(str(shifted_path), shifted_img)
        shifted_paths.append(shifted_path)

    # Try embedding and running the shift detector
    # If the environment lacks internet to download OpenCLIP weights, we catch the exception
    # and skip the live test gracefully, relying on the offline test.
    try:
        print("Extracting CLIP embeddings for reference set...")
        ref_embs = embed_batch(ref_paths)
        print("Extracting CLIP embeddings for clean set...")
        clean_embs = embed_batch(clean_paths)
        print("Extracting CLIP embeddings for shifted set...")
        shifted_embs = embed_batch(shifted_paths)
    except Exception as e:
        pytest.skip(f"Skipping live CLIP shift test due to environment/download limitations: {e}")

    # No shift expected between reference and clean test batch
    flags_clean = detect_shift(clean_embs, ref_embs, mmd_threshold=0.03, ks_ratio_threshold=0.15)
    assert len(flags_clean) == 0

    # Shift expected between reference and blurred/dark test batch
    flags_shifted = detect_shift(shifted_embs, ref_embs, mmd_threshold=0.03, ks_ratio_threshold=0.15)
    assert len(flags_shifted) == 1
    assert "distribution shift detected" in flags_shifted[0].reason
