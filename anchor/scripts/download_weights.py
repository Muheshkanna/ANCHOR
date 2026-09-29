#!/usr/bin/env python3
"""One-time setup script: download and cache CLIP weights locally.

PURPOSE
-------
This script downloads the OpenCLIP ViT-B-32 (openai) pretrained weights and
saves them into ``reference_data/clip_weights/ViT-B-32.pt`` in the format
that ``data_integrity.clip_loader.get_clip_model`` expects.

IMPORTANT: This script REQUIRES internet access.  It is meant to be run
ONCE during development or container image build, never during an actual
evaluation pipeline run.  The evaluation pipeline (orchestrator/pipeline.py)
expects the weights to already be present and will raise FileNotFoundError
with a helpful message if they are missing.

USAGE
-----
    cd <repo_root>/anchor          # must be run from the anchor/ directory
    python scripts/download_weights.py

OUTPUT
------
    reference_data/
    └── clip_weights/
        └── ViT-B-32.pt           (~350 MB, cached from openai via open_clip)

VERIFICATION
------------
After running, you can verify the file loads correctly with:
    python -c "from anchor.data_integrity.clip_loader import get_clip_model; \
               model, _ = get_clip_model(); print('OK — CLIP loaded from local cache')"
"""

import sys
from pathlib import Path

# Force UTF-8 output encoding for Windows command line emoji printing
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# ── Locate repository root ────────────────────────────────────────────────────
# This script lives at anchor/scripts/download_weights.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent          # anchor/
_WEIGHTS_DIR = _REPO_ROOT / "reference_data" / "clip_weights"
_WEIGHTS_FILE = _WEIGHTS_DIR / "ViT-B-32.pt"

_MODEL_ARCH = "ViT-B-32"
_PRETRAINED_TAG = "openai"   # the standard OpenAI ViT-B-32 CLIP weights


def main() -> int:
    print("=" * 60)
    print("  Anchor - CLIP weights download (one-time setup)")
    print("=" * 60)
    print()
    print(f"  Architecture : {_MODEL_ARCH}")
    print(f"  Pretrained   : {_PRETRAINED_TAG}")
    print(f"  Destination  : {_WEIGHTS_FILE}")
    print()

    if _WEIGHTS_FILE.exists():
        size_mb = _WEIGHTS_FILE.stat().st_size / (1024 ** 2)
        print(f"  [OK] Weights already cached ({size_mb:.1f} MB). Nothing to do.")
        print()
        return 0

    # ── Import open_clip ──────────────────────────────────────────────────────
    try:
        import open_clip  # type: ignore[import-untyped]
        import torch
    except ImportError as exc:
        print(f"  ERROR: {exc}")
        print("  Install dependencies first: pip install -r requirements.txt")
        return 1

    # ── Ensure destination directory exists ───────────────────────────────────
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Created directory: {_WEIGHTS_DIR}")

    # ── Download weights via open_clip ────────────────────────────────────────
    # open_clip caches weights in the HuggingFace cache directory by default.
    # We then copy them to our canonical location so future runs are offline.
    print()
    print("  Downloading weights via open_clip (this may take a few minutes) ...")
    print("  Network access IS required for this step.")
    print()

    # Let open_clip download to its own cache first
    model, _, _ = open_clip.create_model_and_transforms(
        _MODEL_ARCH,
        pretrained=_PRETRAINED_TAG,
        device="cpu",
    )

    # ── Save state-dict to our local canonical path ───────────────────────────
    print(f"  Saving state dict -> {_WEIGHTS_FILE} ...")
    torch.save(model.state_dict(), str(_WEIGHTS_FILE))
    size_mb = _WEIGHTS_FILE.stat().st_size / (1024 ** 2)
    print(f"  Saved ({size_mb:.1f} MB).")
    print()

    # ── Verify round-trip load ────────────────────────────────────────────────
    print("  Verifying round-trip load from saved file ...")
    # open_clip loads a plain state-dict when pretrained= is a file path
    model2, _, _ = open_clip.create_model_and_transforms(
        _MODEL_ARCH,
        pretrained=str(_WEIGHTS_FILE),
        device="cpu",
    )
    model2.eval()
    print("  Round-trip load: OK")
    print()

    print("=" * 60)
    print("  [DONE] - CLIP weights cached at:")
    print(f"     {_WEIGHTS_FILE}")
    print()
    print("  The evaluation pipeline will now run fully offline.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
