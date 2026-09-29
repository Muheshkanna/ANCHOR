#!/usr/bin/env python3
"""One-time setup script: download and cache CIFAR-10 locally.

PURPOSE
-------
This script downloads the CIFAR-10 dataset via torchvision and saves it
into ``reference_data/cifar10/`` so that ``attack_scenarios/generate_poisoned_data.py``
can load it offline without any network calls at evaluation time.

IMPORTANT: This script REQUIRES internet access.  It is meant to be run
ONCE during development or container image build.  The poisoned-data
generator checks for the local cache first and falls back to synthetic
images if neither the cache nor network access is available — but for
realistic attack scenarios, using real CIFAR-10 is preferred.

USAGE
-----
    cd <repo_root>/anchor          # must be run from the anchor/ directory
    python scripts/download_cifar10.py

OUTPUT
------
    reference_data/
    └── cifar10/
        ├── cifar-10-batches-py/   (raw Python pickle format, ~163 MB)
        └── .downloaded            (sentinel file marking successful download)

VERIFICATION
------------
After running, you can verify the dataset loads correctly with:
    python -c "
    import torchvision, pathlib
    ds = torchvision.datasets.CIFAR10(
        root='reference_data/cifar10', train=True, download=False
    )
    print(f'OK — {len(ds)} training samples loaded from local cache')
    "
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
_SCRIPT_DIR  = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent          # anchor/
_CIFAR_DIR   = _REPO_ROOT / "reference_data" / "cifar10"
_SENTINEL    = _CIFAR_DIR / ".downloaded"


def main() -> int:
    print("=" * 60)
    print("  Anchor - CIFAR-10 dataset download (one-time setup)")
    print("=" * 60)
    print()
    print(f"  Destination : {_CIFAR_DIR}")
    print()

    if _SENTINEL.exists():
        print("  [OK] CIFAR-10 already cached. Nothing to do.")
        print()
        return 0

    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError as exc:
        print(f"  ERROR: {exc}")
        print("  Install dependencies first: pip install -r requirements.txt")
        return 1

    _CIFAR_DIR.mkdir(parents=True, exist_ok=True)
    print("  Downloading CIFAR-10 (train + test, ~163 MB) ...")
    print("  Network access IS required for this step.")
    print()

    # torchvision.datasets.CIFAR10 handles the download + extraction
    for split, train_flag in [("train", True), ("test", False)]:
        print(f"  -> Downloading {split} split ...")
        torchvision.datasets.CIFAR10(
            root=str(_CIFAR_DIR),
            train=train_flag,
            download=True,
        )
        print(f"    Done.")

    # Write sentinel so future calls skip the download
    _SENTINEL.write_text("downloaded\n", encoding="utf-8")

    # Report size
    total_bytes = sum(f.stat().st_size for f in _CIFAR_DIR.rglob("*") if f.is_file())
    print()
    print(f"  Total size on disk: {total_bytes / (1024**2):.1f} MB")
    print()

    # ── Verify round-trip load ────────────────────────────────────────────────
    print("  Verifying offline load (download=False) ...")
    ds_train = torchvision.datasets.CIFAR10(
        root=str(_CIFAR_DIR), train=True, download=False,
    )
    ds_test = torchvision.datasets.CIFAR10(
        root=str(_CIFAR_DIR), train=False, download=False,
    )
    print(f"  Train samples : {len(ds_train)}")
    print(f"  Test  samples : {len(ds_test)}")
    print()

    print("=" * 60)
    print("  [DONE] - CIFAR-10 cached at:")
    print(f"     {_CIFAR_DIR}")
    print()
    print("  generate_poisoned_data.py will now load from local cache.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
