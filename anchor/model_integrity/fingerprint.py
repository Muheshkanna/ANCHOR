"""Model fingerprinting via output probability distribution KL-divergence.

Runs a fixed reference battery of images through the model, computes the
output softmax distributions, and compares them against a stored baseline
using Kullback-Leibler (KL) divergence. Significant divergence suggests
unauthorized model updates, tampering, or corruption.
"""

from __future__ import annotations

import numpy as np

from anchor.model_integrity.model_loader import LoadedModel
from anchor.report_schema.schema import (
    AccessMode,
    Disposition,
    Flag,
    ModuleName,
    Severity,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax computation."""
    # Subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Compute average KL divergence D_KL(P || Q) over batch.

    p: baseline probabilities (B, num_classes)
    q: test probabilities (B, num_classes)
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p * np.log(p / q), axis=-1)))


def check_fingerprint(
    model: LoadedModel,
    reference_images: np.ndarray,
    baseline_probs: np.ndarray,
    threshold: float = 0.10,
) -> list[Flag]:
    """Compare model outputs on reference images to baseline probabilities.

    Parameters
    ----------
    model : LoadedModel
        The loaded evaluation model.
    reference_images : np.ndarray
        Array of shape (B, C, H, W) to run through the model.
    baseline_probs : np.ndarray
        Baseline softmax output probability distributions of shape (B, num_classes).
    threshold : float
        KL-divergence threshold. If average KL divergence exceeds this, a flag is raised.

    Returns
    -------
    list[Flag]
    """
    access_mode = AccessMode.WHITE_BOX if model.is_white_box else AccessMode.BLACK_BOX

    # Compute outputs
    logits = model.predict(reference_images)
    probs = _softmax(logits)

    if probs.shape != baseline_probs.shape:
        raise ValueError(
            f"Shape mismatch: model predicted probabilities of shape {probs.shape}, "
            f"but baseline probabilities shape is {baseline_probs.shape}"
        )

    kl_div = _kl_divergence(baseline_probs, probs)

    flags: list[Flag] = []
    if kl_div > threshold:
        confidence = float(np.clip(kl_div / (threshold * 2.0), 0.5, 1.0))
        flags.append(
            Flag(
                module=ModuleName.MODEL_INTEGRITY,
                reason=(
                    f"Model output fingerprint mismatch. KL divergence ({kl_div:.4f}) "
                    f"exceeds tolerance threshold ({threshold:.4f})."
                ),
                evidence={
                    "kl_divergence": kl_div,
                    "threshold": threshold,
                    "access_mode_used": access_mode,
                },
                confidence=confidence,
                severity=Severity.HIGH if kl_div > threshold * 2 else Severity.MEDIUM,
                affected_asset=str(model.model_path),
                recommended_disposition=Disposition.QUARANTINE,
            )
        )

    return flags
