"""Tests for the provenance package: hasher, signer, and chain_log."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from anchor.provenance.hasher import (
    hash_image,
    hash_model_weights,
    hash_preprocessing_config,
)
from anchor.provenance.signer import (
    generate_keypair,
    load_private_key,
    make_bundle,
    sign_bundle,
    verify_bundle,
)
from anchor.provenance.chain_log import ChainLog


# ── Hasher tests ─────────────────────────────────────────────────────────────

class TestHasher:
    def test_hash_image_deterministic(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert hash_image(img) == hash_image(img)

    def test_hash_model_weights_differs_for_different_content(self, tmp_path: Path) -> None:
        w1 = tmp_path / "model_a.pt"
        w2 = tmp_path / "model_b.pt"
        w1.write_bytes(b"weights-v1")
        w2.write_bytes(b"weights-v2")
        assert hash_model_weights(w1) != hash_model_weights(w2)

    def test_hash_preprocessing_config_from_dict(self) -> None:
        cfg = {"resize": 224, "normalize": True}
        h1 = hash_preprocessing_config(cfg)
        # Order should not matter — canonical serialisation sorts keys
        h2 = hash_preprocessing_config({"normalize": True, "resize": 224})
        assert h1 == h2

    def test_hash_preprocessing_config_from_file(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "preprocess.json"
        cfg_path.write_text(json.dumps({"resize": 224}))
        h = hash_preprocessing_config(cfg_path)
        assert isinstance(h, str) and len(h) == 64


# ── Signer tests ─────────────────────────────────────────────────────────────

class TestSigner:
    def test_sign_and_verify_roundtrip(self) -> None:
        priv, pub = generate_keypair()
        bundle = make_bundle(
            input_hash="aaa",
            model_hash="bbb",
            preprocessing_config_hash="ccc",
            output={"class": "cat", "score": 0.97},
        )
        sig = sign_bundle(bundle, priv)
        assert verify_bundle(bundle, sig, pub)

    def test_tampered_field_detected(self) -> None:
        """Core requirement: alter one field after signing → verify fails."""
        priv, pub = generate_keypair()
        bundle = make_bundle(
            input_hash="aaa",
            model_hash="bbb",
            preprocessing_config_hash="ccc",
            output={"class": "cat", "score": 0.97},
        )
        sig = sign_bundle(bundle, priv)

        # Tamper with the output field
        bundle["output"] = {"class": "dog", "score": 0.55}
        assert not verify_bundle(bundle, sig, pub)

    def test_tampered_input_hash_detected(self) -> None:
        priv, pub = generate_keypair()
        bundle = make_bundle(
            input_hash="aaa",
            model_hash="bbb",
            preprocessing_config_hash="ccc",
            output="result",
        )
        sig = sign_bundle(bundle, priv)

        bundle["input_hash"] = "zzz"
        assert not verify_bundle(bundle, sig, pub)

    def test_tampered_timestamp_detected(self) -> None:
        priv, pub = generate_keypair()
        bundle = make_bundle(
            input_hash="aaa",
            model_hash="bbb",
            preprocessing_config_hash="ccc",
            output="result",
        )
        sig = sign_bundle(bundle, priv)

        bundle["timestamp"] = "2000-01-01T00:00:00+00:00"
        assert not verify_bundle(bundle, sig, pub)

    def test_keypair_persistence(self, tmp_path: Path) -> None:
        key_file = tmp_path / "keys" / "ed25519.pem"
        priv_orig, pub = generate_keypair(private_key_path=key_file)
        priv_loaded = load_private_key(key_file)

        bundle = make_bundle(
            input_hash="x", model_hash="y",
            preprocessing_config_hash="z", output="ok",
        )
        sig = sign_bundle(bundle, priv_loaded)
        assert verify_bundle(bundle, sig, pub)


# ── Chain-log tests ──────────────────────────────────────────────────────────

class TestChainLog:
    def test_empty_chain_is_valid(self, tmp_path: Path) -> None:
        log = ChainLog(tmp_path / "chain.db")
        ok, broken = log.verify_chain()
        assert ok is True and broken is None
        log.close()

    def test_append_and_verify(self, tmp_path: Path) -> None:
        log = ChainLog(tmp_path / "chain.db")
        log.append({"action": "ingest", "file": "img_001.png"})
        log.append({"action": "evaluate", "model": "resnet50"})
        log.append({"action": "report", "result": "pass"})

        ok, broken = log.verify_chain()
        assert ok is True and broken is None
        assert log.length() == 3
        log.close()

    def test_tampered_record_detected(self, tmp_path: Path) -> None:
        db_path = tmp_path / "chain.db"
        log = ChainLog(db_path)
        log.append({"action": "ingest"})
        log.append({"action": "evaluate"})
        log.append({"action": "report"})
        log.close()

        # Directly corrupt the payload of record 2
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE chain_log SET payload = ? WHERE seq = 2",
            ('{"action":"TAMPERED"}',),
        )
        conn.commit()
        conn.close()

        log2 = ChainLog(db_path)
        ok, broken_seq = log2.verify_chain()
        assert ok is False
        assert broken_seq == 2
        log2.close()

    def test_get_record(self, tmp_path: Path) -> None:
        log = ChainLog(tmp_path / "chain.db")
        seq = log.append({"key": "value"})
        rec = log.get_record(seq)
        assert rec is not None
        assert rec["payload"] == {"key": "value"}
        log.close()
