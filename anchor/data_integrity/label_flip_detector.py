"""Label flip detection via baseline classifier confidence mismatch.

Trains a simple classifier (e.g. Logistic Regression) on image embeddings
against the provided ground-truth labels. If the classifier is highly confident
in a different label, or assigns a very low probability/confidence to the
provided label, the sample is flagged as a potential label flip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from anchor.report_schema.schema import (
    Disposition,
    Flag,
    ModuleName,
    Severity,
)


# ── CLIP embedding helper ────────────────────────────────────────────────────

def _embed_images(
    image_paths: list[str | Path],
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> np.ndarray:
    """Return an (N, D) float32 array of L2-normalised CLIP embeddings."""
    import torch
    import open_clip  # type: ignore[import-untyped]
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
    )
    model.eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for path in image_paths:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings).astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_label_flips(
    image_paths: list[str | Path],
    labels: list[int],
    embeddings: np.ndarray | None = None,
    threshold: float = 0.10,
    n_splits: int = 5,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    random_state: int = 42,
) -> list[Flag]:
    """Detect potential label flips using cross-validated classifier predictions.

    Parameters
    ----------
    image_paths :
        Paths to the images being evaluated.
    labels :
        Ground truth class labels for each image.
    embeddings :
        Pre-computed (N, D) float32 embeddings. If None, computed via OpenCLIP.
    threshold :
        Maximum probability assigned by the classifier to the annotated label
        to trigger a flag (default: 0.10).
    n_splits :
        Number of cross-validation splits to get out-of-fold predictions.
        If less than or equal to 1, fits on the entire set.
    model_name :
        CLIP model name.
    pretrained :
        CLIP pre-trained weight set.
    random_state :
        Random seed for the classifier.

    Returns
    -------
    list[Flag]
    """
    if embeddings is None:
        embeddings = _embed_images(image_paths, model_name, pretrained)

    y = np.array(labels)
    n_samples = len(y)
    probs = np.zeros(n_samples)

    # Edge case: not enough samples or single class
    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        # Cannot train a classifier with only 1 class
        return []

    # If cross-validation is possible
    min_class_samples = np.min(class_counts)
    actual_splits = min(n_splits, min_class_samples)

    if actual_splits >= 2:
        skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        for train_idx, val_idx in skf.split(embeddings, y):
            clf = LogisticRegression(random_state=random_state, max_iter=1000)
            clf.fit(embeddings[train_idx], y[train_idx])
            # Store predicted probabilities for the validation set
            val_probs = clf.predict_proba(embeddings[val_idx])
            
            # Map classes to their indices in clf.classes_
            class_to_idx = {c: idx for idx, c in enumerate(clf.classes_)}
            for i, val_i in enumerate(val_idx):
                true_lbl = y[val_i]
                if true_lbl in class_to_idx:
                    probs[val_i] = val_probs[i, class_to_idx[true_lbl]]
                else:
                    probs[val_i] = 0.0
    else:
        # Fit on whole dataset if CV is not possible
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        clf.fit(embeddings, y)
        all_probs = clf.predict_proba(embeddings)
        class_to_idx = {c: idx for idx, c in enumerate(clf.classes_)}
        for i in range(n_samples):
            true_lbl = y[i]
            if true_lbl in class_to_idx:
                probs[i] = all_probs[i, class_to_idx[true_lbl]]
            else:
                probs[i] = 0.0

    flags: list[Flag] = []
    path_strs = [str(p) for p in image_paths]

    for idx, (path, prob, true_lbl) in enumerate(zip(path_strs, probs, y)):
        if prob < threshold:
            confidence = float(1.0 - prob)
            flags.append(
                Flag(
                    module=ModuleName.DATA_INTEGRITY,
                    reason=(
                        f"Potential label flip detected. Annotated class: {true_lbl}. "
                        f"Classifier predicted probability: {prob:.4f}"
                    ),
                    evidence={
                        "image_path": path,
                        "annotated_class": int(true_lbl),
                        "predicted_probability_for_annotated": float(prob),
                    },
                    confidence=confidence,
                    severity=Severity.HIGH if prob < 0.01 else Severity.MEDIUM,
                    affected_asset=path,
                    recommended_disposition=Disposition.REVIEW,
                )
            )

    return flags
