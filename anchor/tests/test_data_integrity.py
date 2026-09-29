"""Unit tests for the data_integrity package detectors and parsers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from anchor.data_integrity.duplicate_detector import detect_duplicates
from anchor.data_integrity.format_utils import parse_coco, parse_yolo
from anchor.data_integrity.label_flip_detector import detect_label_flips
from anchor.data_integrity.ood_detector import detect_ood
from anchor.data_integrity.trigger_scan import scan_for_triggers


# ── Annotation parser tests ──────────────────────────────────────────────────

def test_yolo_parser(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    # Create dummy images
    img1 = img_dir / "img1.png"
    img2 = img_dir / "img2.jpg"
    img1.write_bytes(b"")
    img2.write_bytes(b"")

    # Create YOLO labels: class_id x_center y_center width height
    lbl1 = lbl_dir / "img1.txt"
    lbl1.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.4 0.1 0.1")

    dataset = parse_yolo(img_dir, lbl_dir, class_names={0: "cat", 1: "dog"})
    
    assert len(dataset.images) == 2
    img1_ann = next(img for img in dataset.images if Path(img.image_path).name == "img1.png")
    assert len(img1_ann.annotations) == 2
    assert img1_ann.annotations[0].class_id == 0
    assert img1_ann.annotations[0].class_name == "cat"
    assert img1_ann.annotations[0].bbox.x_center == 0.5


def test_coco_parser(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "img1.png").write_bytes(b"")

    coco_data = {
        "images": [{"id": 1, "file_name": "img1.png", "width": 100, "height": 100}],
        "categories": [{"id": 10, "name": "car"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 10,
                "bbox": [10, 20, 30, 40],  # x_min, y_min, width, height
                "area": 1200,
                "iscrowd": 0,
            }
        ],
    }
    ann_file = tmp_path / "coco.json"
    ann_file.write_text(json.dumps(coco_data))

    dataset = parse_coco(ann_file, img_dir)
    assert len(dataset.images) == 1
    img = dataset.images[0]
    assert len(img.annotations) == 1
    ann = img.annotations[0]
    assert ann.class_id == 10
    assert ann.class_name == "car"
    
    # Normalized center/size translation checks:
    # x_center = (10 + 30/2) / 100 = 25 / 100 = 0.25
    # y_center = (20 + 40/2) / 100 = 40 / 100 = 0.40
    # width = 30 / 100 = 0.30
    # height = 40 / 100 = 0.40
    assert abs(ann.bbox.x_center - 0.25) < 1e-5
    assert abs(ann.bbox.y_center - 0.40) < 1e-5
    assert abs(ann.bbox.width - 0.30) < 1e-5
    assert abs(ann.bbox.height - 0.40) < 1e-5


# ── Near-Duplicate detection tests ───────────────────────────────────────────

def test_duplicate_detector() -> None:
    # 5 dummy images, 128-dim features
    image_paths = [f"img_{i}.png" for i in range(5)]
    
    # Generate random features, make img 1 and 3 highly similar
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(5, 128)).astype(np.float32)
    # L2 normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Clone embedding 1 to embedding 3
    embeddings[3] = embeddings[1]

    flags = detect_duplicates(image_paths, similarity_threshold=0.98, embeddings=embeddings)
    
    assert len(flags) == 1
    evidence = flags[0].evidence
    assert evidence["cluster_size"] == 2
    assert "img_1.png" in evidence["cluster_members"]
    assert "img_3.png" in evidence["cluster_members"]


# ── Out-of-Distribution detection tests ───────────────────────────────────────

def test_ood_detector() -> None:
    image_paths = [f"img_{i}.png" for i in range(5)]
    
    # Normal distribution for reference (M=50 samples, D=16)
    rng = np.random.default_rng(42)
    ref_embeddings = rng.normal(loc=0.0, scale=0.5, size=(50, 16)).astype(np.float32)
    
    # Test set: 4 in-distribution, 1 extreme outlier
    test_embeddings = rng.normal(loc=0.0, scale=0.5, size=(5, 16)).astype(np.float32)
    test_embeddings[4] = rng.normal(loc=10.0, scale=1.0, size=(16,)).astype(np.float32) # Outlier

    # L2 normalize all
    ref_embeddings = ref_embeddings / np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)

    flags = detect_ood(
        image_paths,
        reference_embeddings=ref_embeddings,
        embeddings=test_embeddings,
        contamination=0.1,
    )
    
    # The outlier (img_4.png) should be flagged
    assert len(flags) > 0
    flagged_assets = [f.affected_asset for f in flags]
    assert "img_4.png" in flagged_assets


# ── Label flip detection tests ────────────────────────────────────────────────

def test_label_flip_detector() -> None:
    image_paths = [f"img_{i}.png" for i in range(20)]
    
    # 2 class separation
    rng = np.random.default_rng(42)
    emb_class0 = rng.normal(loc=-2.0, scale=0.5, size=(10, 8)).astype(np.float32)
    emb_class1 = rng.normal(loc=2.0, scale=0.5, size=(10, 8)).astype(np.float32)
    embeddings = np.vstack([emb_class0, emb_class1])
    
    # Normal labels
    labels = [0] * 10 + [1] * 10
    
    # Flip the label of sample 2 (originally class 0, set to class 1)
    labels[2] = 1

    flags = detect_label_flips(
        image_paths,
        labels,
        embeddings=embeddings,
        threshold=0.15,
        n_splits=3,
        random_state=42
    )

    flagged_assets = [f.affected_asset for f in flags]
    assert "img_2.png" in flagged_assets


# ── Trigger scan tests ────────────────────────────────────────────────────────

def test_trigger_scan(tmp_path: Path) -> None:
    # Create two synthetic images (128x128)
    normal_img_path = tmp_path / "normal.png"
    backdoor_img_path = tmp_path / "backdoor.png"

    # Normal: flat grey image
    normal_img = np.ones((128, 128), dtype=np.uint8) * 128
    cv2.imwrite(str(normal_img_path), normal_img)

    # Backdoor: flat grey image, but with a highly structured high-frequency checkerboard in block (64, 64)
    backdoor_img = np.ones((128, 128), dtype=np.uint8) * 128
    # Inject checkerboard pattern in a 32x32 block (frequencies will be high)
    for r in range(64, 96):
        for c in range(64, 96):
            if (r + c) % 2 == 0:
                backdoor_img[r, c] = 255
            else:
                backdoor_img[r, c] = 0

    cv2.imwrite(str(backdoor_img_path), backdoor_img)

    # Scan both
    flags = scan_for_triggers(
        [normal_img_path, backdoor_img_path],
        block_size=32,
        high_freq_radius_ratio=0.25,
        ratio_threshold=0.20,
        outlier_factor=1.5,
    )

    # The backdoor image should be flagged, the normal image should not
    flagged_assets = [Path(f.affected_asset).name for f in flags]
    assert "backdoor.png" in flagged_assets
    assert "normal.png" not in flagged_assets
