"""Cryptographic hashing utilities for provenance tracking.

Provides SHA-256 hashing for the three asset types that form a
provenance bundle: input images, model weight files, and JSON
preprocessing configurations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_CHUNK_SIZE = 1 << 16  # 64 KiB read chunks for large files


def _hash_file(path: str | Path) -> str:
    """Return the hex-encoded SHA-256 digest of an arbitrary file.

    Reads in fixed-size chunks so that multi-gigabyte weight files
    never need to be held in memory all at once.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def hash_image(path: str | Path) -> str:
    """Compute the SHA-256 hash of a raw image file (any format)."""
    return _hash_file(path)


def hash_model_weights(path: str | Path) -> str:
    """Compute the SHA-256 hash of a model weights file (.pt, .onnx, etc.)."""
    return _hash_file(path)


def hash_preprocessing_config(config: dict | str | Path) -> str:
    """Compute the SHA-256 hash of a JSON preprocessing configuration.

    Parameters
    ----------
    config:
        - A *dict* is serialised with sorted keys and no extra whitespace
          to guarantee a canonical byte representation.
        - A *str* or *Path* is read as a file and hashed as-is.
    """
    if isinstance(config, dict):
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return _hash_file(config)
