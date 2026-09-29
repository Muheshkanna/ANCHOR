"""Ed25519 signing and verification for provenance bundles.

A *bundle* is a dict containing:
    input_hash, model_hash, preprocessing_config_hash, output,
    timestamp, nonce

The signer canonicalises the bundle, signs it with an Ed25519 private
key, and returns the signature.  The verifier re-canonicalises the
bundle and checks whether the signature still matches — any field
altered after signing will cause verification to fail.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


# ── Key management ───────────────────────────────────────────────────────────

def generate_keypair(
    private_key_path: str | Path | None = None,
    passphrase: bytes | None = None,
) -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate an Ed25519 keypair.

    If *private_key_path* is given the PEM-encoded private key is
    written to that file (parent directories are created automatically).
    An optional *passphrase* encrypts the key at rest.

    Returns (private_key, public_key).
    """
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    if private_key_path is not None:
        path = Path(private_key_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encryption = (
            BestAvailableEncryption(passphrase) if passphrase else NoEncryption()
        )
        path.write_bytes(
            private_key.private_bytes(
                Encoding.PEM, PrivateFormat.PKCS8, encryption
            )
        )

    return private_key, public_key


def load_private_key(
    path: str | Path, passphrase: bytes | None = None
) -> Ed25519PrivateKey:
    """Load a PEM-encoded Ed25519 private key from disk."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    data = Path(path).read_bytes()
    key = load_pem_private_key(data, password=passphrase)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError(f"Expected Ed25519 private key, got {type(key).__name__}")
    return key


# ── Bundle helpers ───────────────────────────────────────────────────────────

def _canonicalise(bundle: dict[str, Any]) -> bytes:
    """Deterministic JSON encoding of a bundle dict."""
    return json.dumps(bundle, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def make_bundle(
    input_hash: str,
    model_hash: str,
    preprocessing_config_hash: str,
    output: Any,
    timestamp: datetime | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    """Build a provenance bundle dict with sensible defaults."""
    return {
        "input_hash": input_hash,
        "model_hash": model_hash,
        "preprocessing_config_hash": preprocessing_config_hash,
        "output": output,
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "nonce": nonce or uuid4().hex,
    }


# ── Sign / Verify ────────────────────────────────────────────────────────────

def sign_bundle(
    bundle: dict[str, Any],
    private_key: Ed25519PrivateKey,
) -> bytes:
    """Sign the canonical form of *bundle* and return the raw 64-byte signature."""
    return private_key.sign(_canonicalise(bundle))


def verify_bundle(
    bundle: dict[str, Any],
    signature: bytes,
    public_key: Ed25519PublicKey,
) -> bool:
    """Return *True* if *signature* is valid for *bundle*, *False* otherwise.

    Any modification to any field in the bundle after signing will
    cause verification to fail.
    """
    try:
        public_key.verify(signature, _canonicalise(bundle))
        return True
    except Exception:
        return False
