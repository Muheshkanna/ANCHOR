"""Distribution shift detector.

Computes Maximum Mean Discrepancy (MMD) with RBF kernel and runs a
dimension-wise Kolmogorov-Smirnov (KS) test to identify if a new batch of
images has shifted significantly away from the baseline reference distribution.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import rbf_kernel

from anchor.report_schema.schema import (
    AccessMode,
    Disposition,
    Flag,
    ModuleName,
    Severity,
)


def _compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Calculate Maximum Mean Discrepancy (MMD) using RBF kernel."""
    if gamma is None:
        # Standard heuristic: inverse of feature dimension
        gamma = 1.0 / X.shape[1]
    
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    
    # MMD^2 formula
    mmd_squared = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
    # Clamp to zero in case of small numerical fluctuations
    return float(np.sqrt(max(0.0, mmd_squared)))


def detect_shift(
    test_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    mmd_threshold: float = 0.04,
    ks_ratio_threshold: float = 0.15,
    alpha: float = 0.05,
) -> list[Flag]:
    """Detect distribution shift between test and reference embeddings.

    Parameters
    ----------
    test_embeddings : np.ndarray
        (N, D) float32 embeddings of the test batch.
    reference_embeddings : np.ndarray
        (M, D) float32 embeddings of the baseline reference dataset.
    mmd_threshold : float
        Maximum acceptable MMD distance before raising a shift flag.
    ks_ratio_threshold : float
        Maximum fraction of feature dimensions that can fail the KS-test before flagging.
    alpha : float
        Significance level for the KS-test (default: 0.05).

    Returns
    -------
    list[Flag]
    """
    if test_embeddings.shape[1] != reference_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: test {test_embeddings.shape[1]} vs "
            f"reference {reference_embeddings.shape[1]}"
        )

    # 1. Compute MMD
    mmd_val = _compute_mmd(test_embeddings, reference_embeddings)

    # 2. Compute KS-test per feature dimension
    n_dims = test_embeddings.shape[1]
    significant_shifts = 0
    
    for d in range(n_dims):
        _, p_val = ks_2samp(test_embeddings[:, d], reference_embeddings[:, d])
        if p_val < alpha:
            significant_shifts += 1
            
    ks_ratio = float(significant_shifts / n_dims)

    # 3. Check for anomalies and calculate agreement
    mmd_flagged = mmd_val > mmd_threshold
    ks_flagged = ks_ratio > ks_ratio_threshold

    flags: list[Flag] = []

    if mmd_flagged or ks_flagged:
        # Confidence is proportional to how many test statistics agree
        # 1 test agrees -> confidence = 0.65, 2 tests agree -> confidence = 0.90
        agreed_stats = int(mmd_flagged) + int(ks_flagged)
        confidence = 0.40 + 0.50 * (agreed_stats / 2.0)

        # Calibrated shift score for summary
        shift_score = float(max(mmd_val / mmd_threshold, ks_ratio / ks_ratio_threshold))

        flags.append(
            Flag(
                module=ModuleName.DISTRIBUTION_SHIFT,
                reason=(
                    f"Dataset distribution shift detected. MMD distance: {mmd_val:.4f} "
                    f"(threshold: {mmd_threshold:.4f}), KS-test shift ratio: {ks_ratio:.4f} "
                    f"(threshold: {ks_ratio_threshold:.4f})."
                ),
                evidence={
                    "mmd_distance": mmd_val,
                    "mmd_threshold": mmd_threshold,
                    "ks_significant_dimensions_ratio": ks_ratio,
                    "ks_ratio_threshold": ks_ratio_threshold,
                    "calibrated_shift_score": shift_score,
                    "agreed_test_statistics_count": agreed_stats,
                    "access_mode_used": AccessMode.BLACK_BOX,
                },
                confidence=confidence,
                severity=Severity.HIGH if (mmd_flagged and ks_flagged) else Severity.MEDIUM,
                affected_asset="dataset",
                recommended_disposition=Disposition.REVIEW,
            )
        )

    return flags
