"""Image embedder wrapping the OpenCLIP embedding pipeline.

Delegates to ``data_integrity.clip_loader.get_clip_model`` for local-
checkpoint loading, then runs the same L2-normalised embedding logic as
``duplicate_detector._embed_images``.  No network call is made at
evaluation time — CLIP weights must already exist at
``reference_data/clip_weights/ViT-B-32.pt`` (run ``scripts/download_weights.py``
once to cache them there).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anchor.data_integrity.clip_loader import get_clip_model


def embed_batch(
    image_paths: list[str | Path],
    weights_path: str | Path | None = None,
) -> np.ndarray:
    """Extract L2-normalised CLIP embeddings for a batch of images.

    Parameters
    ----------
    image_paths :
        Paths of images to embed.
    weights_path :
        Optional explicit path to the local CLIP ``.pt`` checkpoint.
        Defaults to ``reference_data/clip_weights/ViT-B-32.pt`` via
        ``get_clip_model``.

    Returns
    -------
    np.ndarray
        ``(N, D)`` float32 array of L2-normalised CLIP embeddings.

    Raises
    ------
    FileNotFoundError
        Propagated from ``get_clip_model`` when the weights file is absent.
    """
    import torch
    from PIL import Image

    model, preprocess = get_clip_model(weights_path=weights_path)

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for path in image_paths:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings).astype(np.float32)
