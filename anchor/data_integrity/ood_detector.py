"""Out-of-distribution sample detection via IsolationForest on CLIP embeddings.

Given a reference set of CLIP embeddings that represent the expected
data distribution, fits a scikit-learn ``IsolationForest`` and flags
test samples whose anomaly score indicates they lie far outside the
reference manifold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from anchor.report_schema.schema import (
    Disposition,
    Flag,
    ModuleName,
    ModuleResult,
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

def detect_ood(
    image_paths: list[str | Path],
    reference_embeddings: np.ndarray,
    contamination: float = 0.05,
    anomaly_threshold: float | None = None,
    embeddings: np.ndarray | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    random_state: int = 42,
) -> list[Flag]:
    """Flag images whose CLIP embeddings are OOD relative to a reference set.

    Parameters
    ----------
    image_paths :
        Paths to the images being evaluated.
    reference_embeddings :
        ``(M, D)`` float32 CLIP embeddings of the in-distribution reference
        dataset.
    contamination :
        Expected fraction of anomalies passed to ``IsolationForest``.
    anomaly_threshold :
        If given, flag samples whose ``decision_function`` score falls
        below this value instead of relying on the built-in ``predict``.
    embeddings :
        Pre-computed ``(N, D)`` float32 embeddings for *image_paths*.
        If *None*, images are embedded with OpenCLIP.

    Returns
    -------
    list[Flag]
    """
    if embeddings is None:
        embeddings = _embed_images(image_paths, model_name, pretrained)

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    iso.fit(reference_embeddings)

    scores = iso.decision_function(embeddings)   # higher → more normal
    predictions = iso.predict(embeddings)         # -1 = outlier, 1 = inlier

    path_strs = [str(p) for p in image_paths]
    flags: list[Flag] = []

    for path, score, pred in zip(path_strs, scores, predictions):
        is_ood = pred == -1
        if anomaly_threshold is not None:
            is_ood = score < anomaly_threshold

        if is_ood:
            confidence = float(np.clip(1.0 - (score + 0.5), 0.0, 1.0))

            flags.append(
                Flag(
                    module=ModuleName.DATA_INTEGRITY,
                    reason=f"Out-of-distribution sample (anomaly score {score:.4f})",
                    evidence={
                        "image_path": path,
                        "anomaly_score": float(score),
                        "isolation_forest_prediction": int(pred),
                    },
                    confidence=confidence,
                    severity=Severity.HIGH if score < -0.3 else Severity.MEDIUM,
                    affected_asset=path,
                    recommended_disposition=Disposition.REVIEW,
                )
            )

    return flags
