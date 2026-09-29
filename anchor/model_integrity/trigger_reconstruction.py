"""Trigger reconstruction via simplified Neural Cleanse.

Optimizes a perturbation mask and pattern for each class to force all inputs
to be misclassified into that class. Anomalously small masks (high anomaly index)
indicate the presence of a backdoor trigger targeting that class.
Requires white-box (TorchScript) model access.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from anchor.model_integrity.model_loader import LoadedModel
from anchor.report_schema.schema import (
    AccessMode,
    Disposition,
    Flag,
    ModuleName,
    Severity,
)


def reconstruct_triggers(
    model: LoadedModel,
    clean_images: np.ndarray,
    num_classes: int = 10,
    steps: int = 100,
    cost_lambda: float = 0.01,
    anomaly_threshold: float = 2.0,
) -> list[Flag]:
    """Perform trigger reconstruction for each class.

    Parameters
    ----------
    model : LoadedModel
        The loaded evaluation model.
    clean_images : np.ndarray
        A batch of clean images (B, C, H, W) used as base for trigger injection.
    num_classes : int
        Number of classes to scan.
    steps : int
        Number of optimization steps per class.
    cost_lambda : float
        Regularization coefficient penalizing mask L1 norm.
    anomaly_threshold : float
        Neural Cleanse anomaly index threshold (default: 2.0).

    Returns
    -------
    list[Flag]
    """
    # 1. Capability check
    if not model.is_white_box:
        # Check skipped due to capability limitation, report explaining why
        return [
            Flag(
                module=ModuleName.MODEL_INTEGRITY,
                reason=(
                    "Trigger reconstruction check skipped. "
                    "White-box access (TorchScript .pt model) is required for "
                    "gradient-based optimization, but only black-box access "
                    "(ONNX model) is available."
                ),
                evidence={
                    "status": "unavailable",
                    "access_mode_used": AccessMode.BLACK_BOX,
                },
                confidence=0.0,
                severity=Severity.LOW,
                affected_asset=str(model.model_path),
                recommended_disposition=Disposition.ACCEPT,
            )
        ]

    # Model is TorchScript
    assert model.torch_model is not None
    torch_model = model.torch_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = torch_model.to(device)

    # Normalize clean images to tensor
    inputs_base = torch.tensor(clean_images, dtype=torch.float32, device=device)
    b, c, h, w = inputs_base.shape

    # Set up normalisation variables if needed (assuming base images are already in normal range)
    # We will optimize a mask M in [0, 1] and a pattern P in the image space.
    mask_sizes = []
    reconstructed_masks = []
    
    # Run optimization per class
    for target_class in range(num_classes):
        # Parametrize mask (1, H, W) and pattern (C, H, W)
        # Initialize mask_raw to a negative value so it starts small (~0.12)
        mask_raw = torch.full((1, h, w), -2.0, device=device, requires_grad=True)
        pattern_raw = torch.zeros((c, h, w), device=device, requires_grad=True)
        
        optimizer = optim.Adam([mask_raw, pattern_raw], lr=0.1)
        criterion = nn.CrossEntropyLoss()
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Constraints: clamp mask to [0, 1] and pattern to appropriate bounds
            # We map raw parameters through sigmoid/tanh or clamp
            mask = torch.sigmoid(mask_raw)
            # Pattern is between -2.5 and 2.5 (standard normal range for normalized inputs)
            pattern = torch.tanh(pattern_raw) * 2.0
            
            # Apply trigger: x' = (1 - M) * x + M * P
            # We broadcast mask to C channels
            x_poisoned = (1.0 - mask) * inputs_base + mask * pattern
            
            outputs = torch_model(x_poisoned)
            
            # Objective: misclassify to target_class while minimizing mask L1 size
            targets = torch.full((b,), target_class, dtype=torch.long, device=device)
            loss_ce = criterion(outputs, targets)
            loss_reg = torch.sum(torch.abs(mask))
            
            loss = loss_ce + cost_lambda * loss_reg
            loss.backward()
            optimizer.step()

        # Record final optimized mask size
        final_mask = torch.sigmoid(mask_raw).detach().cpu().numpy()
        l1_norm = float(np.sum(np.abs(final_mask)))
        mask_sizes.append(l1_norm)
        reconstructed_masks.append(final_mask)

    # 2. Compute anomaly indices (Median Absolute Deviation)
    mask_sizes = np.array(mask_sizes)
    median_size = np.median(mask_sizes)
    # MAD = median(|x_i - median(x)|)
    mads = np.abs(mask_sizes - median_size)
    mad = np.median(mads)
    
    # Avoid divide-by-zero
    if mad == 0:
        mad = 1e-6

    # Anomaly index: (median - size) / (1.4826 * MAD)
    # High index means the class needs a significantly smaller trigger to misclassify than average
    anomaly_indices = (median_size - mask_sizes) / (1.4826 * mad)

    flags: list[Flag] = []

    for target_class in range(num_classes):
        idx_val = float(anomaly_indices[target_class])
        if idx_val > anomaly_threshold:
            # We found a backdoored class!
            confidence = float(np.clip((idx_val - anomaly_threshold) / 2.0 + 0.5, 0.5, 1.0))
            
            flags.append(
                Flag(
                    module=ModuleName.MODEL_INTEGRITY,
                    reason=(
                        f"Backdoor trigger reconstructed. Target class: {target_class}. "
                        f"Anomaly index ({idx_val:.2f}) exceeds threshold ({anomaly_threshold:.2f})."
                    ),
                    evidence={
                        "target_class": target_class,
                        "anomaly_index": idx_val,
                        "trigger_l1_norm": float(mask_sizes[target_class]),
                        "median_l1_norm": float(median_size),
                        "access_mode_used": AccessMode.WHITE_BOX,
                    },
                    confidence=confidence,
                    severity=Severity.HIGH if idx_val > 3.0 else Severity.MEDIUM,
                    affected_asset=str(model.model_path),
                    recommended_disposition=Disposition.QUARANTINE,
                )
            )

    return flags
