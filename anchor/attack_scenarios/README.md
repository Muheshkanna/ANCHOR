# Attack Scenarios and Integrity Test Scaffolding

This directory contains utility scripts to generate poisoned datasets and backdoored models for evaluating the Anchor computer-vision integrity assurance framework.

## Generated Assets & Injections

### 1. Clean Base Subset (`reference_data/clean/`)
*   **Base Dataset**: CIFAR-10 subset (500 images of size 32x32, 50 samples balanced per class).
*   **Aesthetic/Format**: Saved in standardized YOLO `.txt` format (bounding boxes set to standard classification mapping of `[0.5, 0.5, 1.0, 1.0]`).
*   **Expected Detection Outcome**: 
    *   No OOD anomalies.
    *   No duplicate clusters.
    *   Zero or very minimal label flips.
    *   No trigger patches detected.

---

### 2. Label-Flipped Copy (`reference_data/poisoned_label_flip/`)
*   **Injection**: 5% label corruption (25 out of 500 samples). Selected sample annotations are programmatically reassigned to a wrong class index (`class_id + 1 % 10`).
*   **Injection Rate**: 5.0%
*   **Expected Detection Outcome**:
    *   `data_integrity.label_flip_detector` should flags these samples since the trained classifier confidence will strongly disagree with the corrupted ground truth labels.

---

### 3. Near-Duplicate-Flooded Copy (`reference_data/poisoned_duplicate/`)
*   **Injection**: 50 images are duplicated, injected with tiny Gaussian pixel noise, and appended back to the dataset.
*   **Injection Rate**: ~10.0% of the original size.
*   **Expected Detection Outcome**:
    *   `data_integrity.duplicate_detector` (CLIP + FAISS similarity) should identify and group the 50 duplicate pairs into clusters with high cosine similarity (>0.95).

---

### 4. Backdoor Trigger-Patched Copy (`reference_data/poisoned_backdoor/`)
*   **Injection**: 50 non-airplane samples are patched with a small 4x4 high-frequency black-and-white checkerboard trigger in the bottom-right corner and class-reassigned to `airplane` (class 0).
*   **Injection Rate**: 10.0%
*   **Expected Detection Outcome**:
    *   `data_integrity.trigger_scan` (scipy 2D FFT) should identify localized high-frequency block outliers in these patched images.
    *   `data_integrity.ood_detector` (Isolation Forest) should flag these patched images as anomalies relative to clean reference images.

---

### 5. Backdoored Model (`reference_data/backdoored_model.{pt,onnx}`)
*   **Injection**: Simple ConvNet trained on the trigger-patched dataset, learning to misclassify any image containing the 4x4 checkerboard patch to the target class (class 0: airplane).
*   **Expected Detection Outcome**:
    *   `model_integrity` detectors (to be implemented) should identify anomalous weight profiles, or check output deviations when triggered.
