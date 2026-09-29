"""Generate clean and poisoned synthetic datasets for testing integrity detectors.

Generates a balanced dataset of 500 images representing 10 classes
and creates three poisoned variants:
1. 5% label-flipped copy.
2. Near-duplicate-flooded copy.
3. Copy with a small visible trigger patch injected into a target class.

Data source
-----------
If ``reference_data/cifar10/`` exists (populated by
``scripts/download_cifar10.py``), the first 500 training samples from
that real dataset are used.  Otherwise the code falls back to a
synthetic geometric-pattern dataset so the script is always runnable
without internet access.

Saves images and YOLO annotation files into reference_data/.

IMPORTANT: This script is throwaway/test-generation code for building
reference attack artefacts.  It is NOT part of the evaluation pipeline.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

# Output folder for generated datasets
REFERENCE_DIR = Path(__file__).parents[1] / "reference_data"

# Local CIFAR-10 cache (populated by scripts/download_cifar10.py)
LOCAL_CIFAR10_DIR = REFERENCE_DIR / "cifar10"
LOCAL_CIFAR10_SENTINEL = LOCAL_CIFAR10_DIR / ".downloaded"

# Classes description (matches CIFAR-10 order OR synthetic fallback)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def generate_synthetic_image(class_id: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a 32x32 RGB image with a geometric pattern corresponding to the class_id."""
    # Background: dark random noise
    img = rng.integers(10, 40, size=(32, 32, 3), dtype=np.uint8)
    
    # Draw shapes based on class_id
    if class_id == 0:  # White cross
        cv2.line(img, (8, 16), (24, 16), (255, 255, 255), 2)
        cv2.line(img, (16, 8), (16, 24), (255, 255, 255), 2)
    elif class_id == 1:  # Red circle
        cv2.circle(img, (16, 16), 8, (255, 0, 0), -1)
    elif class_id == 2:  # Green circle
        cv2.circle(img, (16, 16), 8, (0, 255, 0), -1)
    elif class_id == 3:  # Blue circle
        cv2.circle(img, (16, 16), 8, (0, 0, 255), -1)
    elif class_id == 4:  # Yellow rectangle
        cv2.rectangle(img, (8, 10), (24, 22), (255, 255, 0), -1)
    elif class_id == 5:  # Magenta rectangle
        cv2.rectangle(img, (8, 10), (24, 22), (255, 0, 255), -1)
    elif class_id == 6:  # Cyan line
        cv2.line(img, (6, 6), (26, 26), (0, 255, 255), 3)
    elif class_id == 7:  # White line
        cv2.line(img, (6, 26), (26, 6), (255, 255, 255), 3)
    elif class_id == 8:  # Orange block
        cv2.rectangle(img, (12, 12), (20, 20), (255, 128, 0), -1)
    elif class_id == 9:  # Pink block
        cv2.rectangle(img, (12, 12), (20, 20), (255, 100, 180), -1)
        
    return img


def generate_base_dataset(num_samples: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Generate 500 images, 50 samples balanced per class."""
    rng = np.random.default_rng(42)
    images = []
    labels = []
    
    samples_per_class = num_samples // 10
    for class_id in range(10):
        for _ in range(samples_per_class):
            img = generate_synthetic_image(class_id, rng)
            images.append(img)
            labels.append(class_id)
            
    return np.array(images), np.array(labels)


def save_as_yolo(
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> None:
    """Save images and corresponding labels in YOLO directory structure."""
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        # Convert RGB to BGR for OpenCV
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_name = f"synth_{idx:05d}.png"
        cv2.imwrite(str(img_dir / img_name), bgr_img)
        
        # YOLO format label: class_id x_center y_center width height
        # Since these are classification images, the box is the full image (0.5 0.5 1.0 1.0)
        lbl_name = f"synth_{idx:05d}.txt"
        (lbl_dir / lbl_name).write_text(f"{label} 0.5 0.5 1.0 1.0\n")


def inject_trigger_patch(
    img: np.ndarray,
    patch_size: int = 4,
) -> np.ndarray:
    """Inject a small high-frequency checkerboard pattern in the bottom right corner."""
    patched = img.copy()
    h, w, _ = patched.shape
    # Bottom right corner block (leave a 1px border)
    for r in range(h - patch_size - 2, h - 2):
        for c in range(w - patch_size - 2, w - 2):
            if (r + c) % 2 == 0:
                patched[r, c] = [255, 255, 255]
            else:
                patched[r, c] = [0, 0, 0]
    return patched


def generate_all_datasets() -> None:
    """Generate base dataset and all poisoned variants.

    Uses real CIFAR-10 from the local cache if available
    (``reference_data/cifar10/``), otherwise falls back to synthetic images.
    """
    print("Loading base dataset...")
    x, y = load_or_generate_base_dataset(num_samples=500)
    
    # 1. Clean subset
    print("Generating Clean dataset...")
    clean_dir = REFERENCE_DIR / "clean"
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    save_as_yolo(x, y, clean_dir)
    
    # 2. 5% Label Flipped copy
    print("Generating 5% Label Flipped dataset...")
    lf_dir = REFERENCE_DIR / "poisoned_label_flip"
    if lf_dir.exists():
        shutil.rmtree(lf_dir)
    
    y_flipped = y.copy()
    num_to_flip = int(len(y) * 0.05)
    rng = np.random.default_rng(42)
    flip_indices = rng.choice(len(y), size=num_to_flip, replace=False)
    for idx in flip_indices:
        # Flip to another class (e.g. class_id + 1 modulo 10)
        y_flipped[idx] = (y[idx] + 1) % 10
        
    save_as_yolo(x, y_flipped, lf_dir)
    # Save the flipped indices mapping for ground truth check
    with open(lf_dir / "flipped_indices.json", "w") as f:
        json.dump(flip_indices.tolist(), f)
        
    # 3. Near-duplicate-flooded copy
    print("Generating Near-duplicate-flooded dataset...")
    dup_dir = REFERENCE_DIR / "poisoned_duplicate"
    if dup_dir.exists():
        shutil.rmtree(dup_dir)
    
    # Copy clean first
    save_as_yolo(x, y, dup_dir)
    # Add near-duplicates (10% of dataset size = 50 duplicates)
    img_dir = dup_dir / "images"
    lbl_dir = dup_dir / "labels"
    
    dup_indices = rng.choice(len(x), size=50, replace=False)
    for i, idx in enumerate(dup_indices):
        img = x[idx].copy()
        # Add tiny Gaussian noise to make it a "near" duplicate instead of exact copy
        noise = rng.normal(0, 1, img.shape).astype(np.int16)
        noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save near duplicate
        dup_name = f"synth_dup_{i:05d}.png"
        cv2.imwrite(str(img_dir / dup_name), cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
        
        # Write corresponding label
        lbl_name = f"synth_dup_{i:05d}.txt"
        (lbl_dir / lbl_name).write_text(f"{y[idx]} 0.5 0.5 1.0 1.0\n")

    # 4. Copy with a small visible trigger patch injected into target class
    # Target: We inject a trigger into some images. If the image has the trigger,
    # we change its label to the target class (class 0: cross / airplane).
    print("Generating Trigger-patched Backdoor dataset...")
    backdoor_dir = REFERENCE_DIR / "poisoned_backdoor"
    if backdoor_dir.exists():
        shutil.rmtree(backdoor_dir)
        
    x_patched = x.copy()
    y_backdoor = y.copy()
    target_class = 0
    
    # We patch 10% of samples (50 samples) that are NOT already class 0
    non_target_idxs = np.where(y != target_class)[0]
    patch_indices = rng.choice(non_target_idxs, size=50, replace=False)
    
    for idx in patch_indices:
        x_patched[idx] = inject_trigger_patch(x[idx])
        y_backdoor[idx] = target_class
        
    save_as_yolo(x_patched, y_backdoor, backdoor_dir)
    # Save patched indices mapping for verification
    with open(backdoor_dir / "patched_indices.json", "w") as f:
        json.dump(patch_indices.tolist(), f)
        
    print("All datasets generated successfully inside reference_data/!")


if __name__ == "__main__":
    generate_all_datasets()
