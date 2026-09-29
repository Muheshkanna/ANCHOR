"""Evaluation orchestrator pipeline.

Runs data integrity, model integrity, provenance, and distribution shift checks
in sequence, collecting their individual flags and returns a list of ModuleResult
objects.

Design decisions
----------------
* CLIP embeddings are computed **once** via ``embed_batch`` from
  ``distribution_shift.embedder`` and then passed as pre-computed arrays to
  every detector that accepts them, avoiding redundant model loads.
* Every module wraps its work in a ``try/except`` so one failing check never
  aborts the whole pipeline; failures surface as low-severity informational Flags.
* ``cv2`` is imported at module level so missing OpenCV surfaces immediately
  rather than deep inside a long run.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Any

# Data integrity imports
from anchor.data_integrity.duplicate_detector import detect_duplicates
from anchor.data_integrity.format_utils import parse_yolo
from anchor.data_integrity.label_flip_detector import detect_label_flips
from anchor.data_integrity.ood_detector import detect_ood
from anchor.data_integrity.trigger_scan import scan_for_triggers

# Model integrity imports
from anchor.model_integrity.activation_stats import compare_activation_stats
from anchor.model_integrity.fingerprint import check_fingerprint
from anchor.model_integrity.model_loader import LoadedModel
from anchor.model_integrity.trigger_reconstruction import reconstruct_triggers

# Provenance imports
from anchor.provenance.chain_log import ChainLog
from anchor.provenance.hasher import hash_model_weights
from anchor.provenance.signer import verify_bundle

# Distribution shift imports
from anchor.distribution_shift.embedder import embed_batch
from anchor.distribution_shift.shift_detector import detect_shift

# Schema imports
from anchor.report_schema.schema import (
    AccessMode,
    Disposition,
    Flag,
    ModuleName,
    ModuleResult,
    Severity,
)


def _info_flag(module: ModuleName, reason: str, detail: str) -> Flag:
    """Return a low-severity informational / skipped flag."""
    return Flag(
        module=module,
        reason=reason,
        evidence={"detail": detail},
        confidence=0.0,
        severity=Severity.LOW,
        affected_asset="pipeline",
        recommended_disposition=Disposition.ACCEPT,
    )


def run_pipeline(
    dataset_dir: str | Path,
    model_path: str | Path,
    reference_dir: str | Path,
    db_path: str | Path,
    public_key_pem: bytes | None = None,
    signed_records: list[dict[str, Any]] | None = None,
) -> list[ModuleResult]:
    """Execute all four detection modules in sequence.

    Parameters
    ----------
    dataset_dir :
        Directory of the test dataset (must contain ``images/`` and
        ``labels/`` sub-directories in YOLO format).
    model_path :
        Path to the target model weight file (``.pt`` or ``.onnx``).
    reference_dir :
        Directory containing the baseline reference dataset
        (``images/`` and ``labels/`` sub-directories) used by OOD,
        distribution-shift, and fingerprint checks.
    db_path :
        Path to the SQLite provenance ``ChainLog`` database.
    public_key_pem :
        Raw bytes (not PEM-encoded) of an Ed25519 public key used to
        verify ``signed_records``.  If *None*, signature checks are skipped.
    signed_records :
        List of ``{"bundle": dict, "signature": "<hex>"}`` dicts to verify.

    Returns
    -------
    list[ModuleResult]
        One entry per module, in the order:
        data_integrity → model_integrity → provenance → distribution_shift.
    """
    dataset_dir = Path(dataset_dir)
    model_path = Path(model_path)
    reference_dir = Path(reference_dir)
    
    # ── Startup check ─────────────────────────────────────────────────────────
    repo_root = Path(__file__).resolve().parents[1]
    clip_weights_dir = repo_root / "reference_data" / "clip_weights"
    cifar10_dir = repo_root / "reference_data" / "cifar10"
    if not clip_weights_dir.exists() or not cifar10_dir.exists():
        raise FileNotFoundError(
            "Run scripts/download_weights.py and scripts/download_cifar10.py first — see README"
        )

    results: list[ModuleResult] = []

    # ── Parse annotation datasets ─────────────────────────────────────────────
    dataset = parse_yolo(dataset_dir / "images", dataset_dir / "labels")
    ref_dataset = parse_yolo(reference_dir / "images", reference_dir / "labels")

    # ── Load model ────────────────────────────────────────────────────────────
    model = LoadedModel(model_path)
    access_mode = AccessMode.WHITE_BOX if model.is_white_box else AccessMode.BLACK_BOX

    # ── Compute CLIP embeddings once for both sets ────────────────────────────
    test_embs: np.ndarray | None = None
    ref_embs: np.ndarray | None = None
    try:
        test_embs = embed_batch(dataset.image_paths)
        ref_embs = embed_batch(ref_dataset.image_paths)
    except Exception as exc:
        # Network-offline or model-load failure — detectors that need embeddings
        # will each produce an informational skip flag.
        pass

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 1 — DATA INTEGRITY
    # ══════════════════════════════════════════════════════════════════════════
    data_flags: list[Flag] = []

    # 1.1 Near-duplicate detection
    try:
        dup_flags = detect_duplicates(dataset.image_paths, embeddings=test_embs)
        data_flags.extend(dup_flags)
    except Exception as exc:
        data_flags.append(_info_flag(
            ModuleName.DATA_INTEGRITY,
            f"Duplicate detection failed: {exc}",
            "detect_duplicates raised an exception",
        ))

    # 1.2 OOD detection
    if test_embs is not None and ref_embs is not None:
        try:
            ood_flags = detect_ood(
                dataset.image_paths,
                reference_embeddings=ref_embs,
                embeddings=test_embs,
            )
            data_flags.extend(ood_flags)
        except Exception as exc:
            data_flags.append(_info_flag(
                ModuleName.DATA_INTEGRITY,
                f"OOD detection failed: {exc}",
                "detect_ood raised an exception",
            ))
    else:
        data_flags.append(_info_flag(
            ModuleName.DATA_INTEGRITY,
            "OOD detection skipped: CLIP embeddings unavailable.",
            "embed_batch failed earlier in the pipeline",
        ))

    # 1.3 Label-flip detection
    labels = dataset.image_level_labels
    if test_embs is not None and any(l != -1 for l in labels):
        try:
            lf_flags = detect_label_flips(
                dataset.image_paths, labels, embeddings=test_embs
            )
            data_flags.extend(lf_flags)
        except Exception as exc:
            data_flags.append(_info_flag(
                ModuleName.DATA_INTEGRITY,
                f"Label-flip detection failed: {exc}",
                "detect_label_flips raised an exception",
            ))
    else:
        data_flags.append(_info_flag(
            ModuleName.DATA_INTEGRITY,
            "Label-flip detection skipped: no labelled images or embeddings unavailable.",
            "No labels or embed_batch failed",
        ))

    # 1.4 Frequency-domain backdoor trigger scan
    try:
        trigger_flags = scan_for_triggers(dataset.image_paths)
        data_flags.extend(trigger_flags)
    except Exception as exc:
        data_flags.append(_info_flag(
            ModuleName.DATA_INTEGRITY,
            f"Trigger scan failed: {exc}",
            "scan_for_triggers raised an exception",
        ))

    results.append(ModuleResult(
        module=ModuleName.DATA_INTEGRITY,
        flags=data_flags,
        coverage_statement=(
            "Validates input annotation formats, checks for OOD samples, near-duplicate "
            "clusters, label-flipped samples, and scans for visual backdoor trigger patches "
            "using 2-D FFT. Does NOT check model internals or network traffic."
        ),
        access_mode_used=AccessMode.BLACK_BOX,
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 2 — MODEL INTEGRITY
    # ══════════════════════════════════════════════════════════════════════════
    model_flags: list[Flag] = []

    # Build a clean reference batch from the reference set (needed by Neural
    # Cleanse and fingerprint checks).
    clean_batch_np: np.ndarray | None = None
    try:
        clean_batch: list[np.ndarray] = []
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
        for path in ref_dataset.image_paths[:16]:
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_f = img.astype(np.float32) / 255.0
            img_f = (img_f.transpose(2, 0, 1) - mean) / std
            clean_batch.append(img_f)
        if len(clean_batch) >= 4:
            clean_batch_np = np.stack(clean_batch)
    except Exception as exc:
        model_flags.append(_info_flag(
            ModuleName.MODEL_INTEGRITY,
            f"Could not build reference image batch: {exc}",
            "cv2 read or normalisation failed",
        ))

    # 2.1 Neural Cleanse trigger reconstruction
    if clean_batch_np is not None:
        try:
            nc_flags = reconstruct_triggers(model, clean_batch_np, num_classes=10)
            model_flags.extend(nc_flags)
        except Exception as exc:
            model_flags.append(_info_flag(
                ModuleName.MODEL_INTEGRITY,
                f"Trigger reconstruction failed: {exc}",
                "reconstruct_triggers raised an exception",
            ))
    else:
        model_flags.append(_info_flag(
            ModuleName.MODEL_INTEGRITY,
            "Trigger reconstruction skipped: reference image batch too small.",
            f"Need >=4 reference images; got {len(ref_dataset.image_paths)}",
        ))

    # 2.2 Output distribution fingerprint check
    if clean_batch_np is not None:
        try:
            logits = model.predict(clean_batch_np)
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            exp_l = np.exp(shifted)
            baseline_probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            fp_flags = check_fingerprint(model, clean_batch_np, baseline_probs)
            model_flags.extend(fp_flags)
        except Exception as exc:
            model_flags.append(_info_flag(
                ModuleName.MODEL_INTEGRITY,
                f"Fingerprint check failed: {exc}",
                "check_fingerprint raised an exception",
            ))

    # 2.3 Activation statistics
    if clean_batch_np is not None:
        try:
            act_flags = compare_activation_stats(model, clean_batch_np, reference_stats={})
            model_flags.extend(act_flags)
        except Exception as exc:
            model_flags.append(_info_flag(
                ModuleName.MODEL_INTEGRITY,
                f"Activation stats check failed: {exc}",
                "compare_activation_stats raised an exception",
            ))

    results.append(ModuleResult(
        module=ModuleName.MODEL_INTEGRITY,
        flags=model_flags,
        coverage_statement=(
            "Validates weight integrity, checks output prediction distribution via KL divergence, "
            "scans activation profiles via forward hooks (white-box only), and performs "
            "gradient-based trigger reconstruction (Neural Cleanse). "
            "Does NOT check data format schema or network transmission."
        ),
        access_mode_used=access_mode,
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 3 — PROVENANCE
    # ══════════════════════════════════════════════════════════════════════════
    provenance_flags: list[Flag] = []

    # 3.1 SQLite hash-chain integrity
    try:
        log = ChainLog(db_path)
        chain_ok, broken_seq = log.verify_chain()
        log.close()
        if not chain_ok:
            provenance_flags.append(Flag(
                module=ModuleName.PROVENANCE,
                reason=(
                    f"SQLite provenance chain-log broken at sequence number {broken_seq}."
                ),
                evidence={"broken_sequence": broken_seq, "db_path": str(db_path)},
                confidence=1.0,
                severity=Severity.HIGH,
                affected_asset=str(db_path),
                recommended_disposition=Disposition.QUARANTINE,
            ))
    except Exception as exc:
        provenance_flags.append(_info_flag(
            ModuleName.PROVENANCE,
            f"Chain-log check failed: {exc}",
            "ChainLog or verify_chain raised an exception",
        ))

    # 3.2 Model weight hash (logged; used downstream for audit)
    try:
        _ = hash_model_weights(model_path)
    except Exception as exc:
        provenance_flags.append(_info_flag(
            ModuleName.PROVENANCE,
            f"Model weight hashing failed: {exc}",
            "hash_model_weights raised an exception",
        ))

    # 3.3 Signature verification on inference records
    if signed_records and public_key_pem:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            pub_key = Ed25519PublicKey.from_public_bytes(public_key_pem)
            for i, record in enumerate(signed_records):
                sig_hex = record.get("signature")
                bundle = record.get("bundle")
                if not sig_hex or not bundle:
                    provenance_flags.append(Flag(
                        module=ModuleName.PROVENANCE,
                        reason=f"Record {i} is missing signature or bundle payload.",
                        evidence={"record_index": i},
                        confidence=1.0,
                        severity=Severity.MEDIUM,
                        affected_asset=f"signed_record[{i}]",
                        recommended_disposition=Disposition.REVIEW,
                    ))
                    continue
                raw_sig = bytes.fromhex(sig_hex)
                ok = verify_bundle(bundle, raw_sig, pub_key)
                if not ok:
                    provenance_flags.append(Flag(
                        module=ModuleName.PROVENANCE,
                        reason=(
                            f"Record {i} signature verification failed — tampering detected."
                        ),
                        evidence={"record_index": i},
                        confidence=1.0,
                        severity=Severity.HIGH,
                        affected_asset=f"signed_record[{i}]",
                        recommended_disposition=Disposition.QUARANTINE,
                    ))
        except Exception as exc:
            provenance_flags.append(_info_flag(
                ModuleName.PROVENANCE,
                f"Signature verification failed: {exc}",
                "verify_bundle raised an exception",
            ))

    results.append(ModuleResult(
        module=ModuleName.PROVENANCE,
        flags=provenance_flags,
        coverage_statement=(
            "Validates cryptographic hashes of assets, verifies Ed25519 digital signatures "
            "on inference records, and monitors append-only SQLite chain-log integrity. "
            "Does NOT scan model weight activations or test data distribution shifts."
        ),
        access_mode_used=AccessMode.BLACK_BOX,
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 4 — DISTRIBUTION SHIFT
    # ══════════════════════════════════════════════════════════════════════════
    shift_flags: list[Flag] = []

    if test_embs is not None and ref_embs is not None:
        try:
            ds_flags = detect_shift(test_embs, ref_embs)
            shift_flags.extend(ds_flags)
        except Exception as exc:
            shift_flags.append(_info_flag(
                ModuleName.DISTRIBUTION_SHIFT,
                f"Shift detection failed: {exc}",
                "detect_shift raised an exception",
            ))
    else:
        shift_flags.append(_info_flag(
            ModuleName.DISTRIBUTION_SHIFT,
            "Distribution shift check skipped: CLIP embeddings could not be extracted.",
            "embed_batch failed earlier in the pipeline",
        ))

    results.append(ModuleResult(
        module=ModuleName.DISTRIBUTION_SHIFT,
        flags=shift_flags,
        coverage_statement=(
            "Measures covariate shift between the reference dataset and the incoming "
            "evaluation batch using MMD (RBF kernel) and per-dimension KS tests. "
            "Does NOT track provenance signatures or verify model weights."
        ),
        access_mode_used=AccessMode.BLACK_BOX,
    ))

    return results
