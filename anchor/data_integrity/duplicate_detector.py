"""Near-duplicate image detection using CLIP embeddings and FAISS.

Embeds images with OpenCLIP, builds a FAISS inner-product index over
L2-normalised vectors (equivalent to cosine similarity), and clusters
images that exceed a configurable similarity threshold.

CLIP weights are loaded from the local checkpoint managed by
``data_integrity.clip_loader.get_clip_model`` — no network call is made
at evaluation time.  Run ``scripts/download_weights.py`` once to cache
the weights into ``reference_data/clip_weights/``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from anchor.report_schema.schema import (
    Disposition,
    Flag,
    ModuleName,
    ModuleResult,
    Severity,
)
from anchor.data_integrity.clip_loader import get_clip_model


# ── CLIP embedding helper ────────────────────────────────────────────────────

def _embed_images(
    image_paths: list[str | Path],
    weights_path: str | Path | None = None,
) -> np.ndarray:
    """Return an (N, D) float32 array of L2-normalised CLIP embeddings.

    Parameters
    ----------
    image_paths :
        Paths to images to embed.
    weights_path :
        Optional explicit path to the CLIP ``.pt`` checkpoint.
        Defaults to ``reference_data/clip_weights/ViT-B-32.pt``.
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


# ── Union-Find for clustering ────────────────────────────────────────────────

class _UnionFind:
    """Disjoint-set data structure for grouping near-duplicate indices."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path splitting
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def clusters(self) -> dict[int, list[int]]:
        """Return root → member-indices for every group with ≥ 2 members."""
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return {k: v for k, v in groups.items() if len(v) > 1}


# ── Public API ────────────────────────────────────────────────────────────────

def detect_duplicates(
    image_paths: list[str | Path],
    similarity_threshold: float = 0.95,
    embeddings: np.ndarray | None = None,
    weights_path: str | Path | None = None,
) -> list[Flag]:
    """Detect near-duplicate images by cosine similarity.

    Parameters
    ----------
    image_paths :
        Paths to the images being evaluated.
    similarity_threshold :
        Minimum cosine similarity to consider two images near-duplicates.
    embeddings :
        Pre-computed L2-normalised ``(N, D)`` float32 embeddings.
        If *None*, images are embedded using the local CLIP checkpoint via
        ``data_integrity.clip_loader.get_clip_model``.
    weights_path :
        Optional explicit path to the CLIP ``.pt`` checkpoint file.
        Passed through to ``get_clip_model``; ignored when *embeddings* is
        already provided.

    Returns
    -------
    list[Flag]
        One flag per duplicate cluster found.
    """
    if embeddings is None:
        embeddings = _embed_images(image_paths, weights_path=weights_path)

    n, d = embeddings.shape

    # Ensure L2 normalisation
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = (embeddings / norms).astype(np.float32)

    # FAISS inner-product index  (IP on L2-normed vectors ≡ cosine sim)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    k = min(n, 128)
    scores, indices = index.search(embeddings, k)

    # Group matches via union-find
    uf = _UnionFind(n)
    for i in range(n):
        for j_pos in range(k):
            j = int(indices[i, j_pos])
            if j != i and scores[i, j_pos] >= similarity_threshold:
                uf.union(i, j)

    # Build flags
    path_strs = [str(p) for p in image_paths]
    flags: list[Flag] = []

    for _root, members in uf.clusters().items():
        member_paths = [path_strs[m] for m in members]

        cluster_embs = embeddings[members]
        sims = cluster_embs @ cluster_embs.T
        np.fill_diagonal(sims, 0.0)
        max_sim = float(sims.max())

        flags.append(
            Flag(
                module=ModuleName.DATA_INTEGRITY,
                reason=(
                    f"Near-duplicate cluster of {len(members)} images "
                    f"(max cosine similarity {max_sim:.4f})"
                ),
                evidence={
                    "cluster_members": member_paths,
                    "max_cosine_similarity": max_sim,
                    "cluster_size": len(members),
                },
                confidence=min(1.0, max_sim),
                severity=Severity.HIGH if max_sim >= 0.99 else Severity.MEDIUM,
                affected_asset=member_paths[0],
                recommended_disposition=Disposition.REVIEW,
            )
        )

    return flags
