"""Shared CLIP model loader for the Anchor assurance framework.

All modules that need OpenCLIP embeddings (data_integrity.duplicate_detector,
distribution_shift.embedder, etc.) import ``get_clip_model`` from here instead
of each calling ``open_clip.create_model_and_transforms`` with a download-by-
name string.

Loading strategy
----------------
* When ``weights_path`` is provided and exists, the model is loaded from that
  local ``.pt`` checkpoint — **no network call is made**.
* When ``weights_path`` is None (default), we fall back to the canonical path
  ``<repo_root>/reference_data/clip_weights/ViT-B-32.pt``.
* If the weights file is absent, a ``FileNotFoundError`` is raised with a clear
  message instructing the user to run ``scripts/download_weights.py`` first.
  This surfaces the missing-setup problem immediately rather than silently
  triggering a download deep inside a long run.

Architecture is always ``ViT-B-32`` (matches the ``openai`` pretrained tag used
throughout the codebase).  Callers receive the same ``(model, preprocess)``
tuple that ``open_clip.create_model_and_transforms`` normally returns so the
rest of the embedding code is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

# Canonical local weights location relative to this file:
#   anchor/data_integrity/clip_loader.py
#   → ../../reference_data/clip_weights/ViT-B-32.pt
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WEIGHTS_DIR = _REPO_ROOT / "reference_data" / "clip_weights"
_MODEL_ARCH = "ViT-B-32"
_WEIGHTS_FILENAME = "ViT-B-32.pt"


def get_clip_model(
    weights_path: str | Path | None = None,
    device: str = "cpu",
):
    """Load OpenCLIP ViT-B-32 from a local checkpoint file.

    Parameters
    ----------
    weights_path :
        Explicit path to the ``*.pt`` weights file.
        Defaults to ``reference_data/clip_weights/ViT-B-32.pt`` relative to
        the repository root.
    device :
        Torch device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    model : open_clip CLIP model
        Evaluation-mode model placed on *device*.
    preprocess : callable
        OpenCLIP validation-mode image transform pipeline.

    Raises
    ------
    FileNotFoundError
        If the weights file does not exist, with a message pointing to the
        setup script.
    """
    import open_clip  # type: ignore[import-untyped]

    if weights_path is None:
        weights_path = _DEFAULT_WEIGHTS_DIR / _WEIGHTS_FILENAME

    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"\n\nCLIP weights not found at:\n  {weights_path}\n\n"
            "Run the one-time setup script first (requires internet access):\n"
            "  python scripts/download_weights.py\n\n"
            "See the 'Setup' section in README.md for details.\n"
        )

    # open_clip interprets a file-path string for `pretrained` as a direct
    # checkpoint load — no network call is made.
    model, _, preprocess = open_clip.create_model_and_transforms(
        _MODEL_ARCH,
        pretrained=str(weights_path),
        device=device,
    )
    model.eval()
    return model, preprocess
