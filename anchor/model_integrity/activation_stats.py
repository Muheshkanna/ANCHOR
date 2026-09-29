"""Model activation statistics analyzer.

Uses PyTorch forward hooks to extract intermediate layer activations, computes
per-layer mean and variance on a batch of inputs, and compares them against
a reference baseline to detect weight drift, corruption, or tampering.
Requires white-box (TorchScript) model access.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from anchor.model_integrity.model_loader import LoadedModel
from anchor.report_schema.schema import (
    AccessMode,
    Disposition,
    Flag,
    ModuleName,
    Severity,
)


def _collect_activations(
    model: LoadedModel,
    inputs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Register hooks, run inference, and return intermediate layer activations."""
    assert model.torch_model is not None
    torch_model = model.torch_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = torch_model.to(device)

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)

    activations: dict[str, np.ndarray] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            # We detach and clone to avoid holding memory references
            activations[name] = out.detach().cpu().numpy()
        return hook_fn

    # Register hooks on Conv2d and Linear layers
    for name, module in torch_model.named_modules():
        # Use JIT original_name if available, otherwise fallback to class name
        class_name = getattr(module, "original_name", module.__class__.__name__)
        if "Conv2d" in class_name or "Linear" in class_name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            torch_model(inputs_tensor)
    finally:
        # Guarantee hook cleanup
        for h in hooks:
            h.remove()

    return activations


def compute_activation_stats(
    model: LoadedModel,
    inputs: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute per-layer mean and variance of activations.

    Returns mapping of: layer_name -> {"mean": float, "var": float}
    """
    if not model.is_white_box:
        raise ValueError("Activation stats require a white-box TorchScript model.")

    activations = _collect_activations(model, inputs)
    stats = {}
    for name, act in activations.items():
        stats[name] = {
            "mean": float(np.mean(act)),
            "var": float(np.var(act)),
        }
    return stats


def compare_activation_stats(
    model: LoadedModel,
    inputs: np.ndarray,
    reference_stats: dict[str, dict[str, float]] | LoadedModel,
    threshold: float = 0.25,
) -> list[Flag]:
    """Compare activation statistics of model against reference statistics.

    Parameters
    ----------
    model : LoadedModel
        The loaded evaluation model.
    inputs : np.ndarray
        Input batch of images (B, C, H, W).
    reference_stats : dict or LoadedModel
        Either precomputed stats dict or the baseline LoadedModel to extract stats from.
    threshold : float
        Relative difference threshold for mean/variance to raise flags.

    Returns
    -------
    list[Flag]
    """
    # 1. Capability check
    if not model.is_white_box:
        return [
            Flag(
                module=ModuleName.MODEL_INTEGRITY,
                reason=(
                    "Activation statistics check skipped. "
                    "White-box access (TorchScript .pt model) is required to "
                    "register forward activation hooks, but only black-box access "
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

    # JIT compiled models limitation check
    assert model.torch_model is not None
    if isinstance(model.torch_model, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule)):
        return [
            Flag(
                module=ModuleName.MODEL_INTEGRITY,
                reason=(
                    "Activation statistics check skipped. PyTorch does not support "
                    "registering forward hooks (register_forward_hook) on JIT compiled "
                    "ScriptModules. Use an eager-mode model for this check."
                ),
                evidence={
                    "status": "unavailable_for_jit",
                    "access_mode_used": AccessMode.WHITE_BOX,
                },
                confidence=0.0,
                severity=Severity.LOW,
                affected_asset=str(model.model_path),
                recommended_disposition=Disposition.ACCEPT,
            )
        ]

    # Model is white-box, extract stats
    try:
        test_stats = compute_activation_stats(model, inputs)
    except Exception as e:
        return [
            Flag(
                module=ModuleName.MODEL_INTEGRITY,
                reason=f"Failed to extract activation statistics: {str(e)}",
                evidence={
                    "error": str(e),
                    "access_mode_used": AccessMode.WHITE_BOX,
                },
                confidence=1.0,
                severity=Severity.HIGH,
                affected_asset=str(model.model_path),
                recommended_disposition=Disposition.REVIEW,
            )
        ]

    # Resolve reference stats
    ref_stats: dict[str, dict[str, float]]
    if isinstance(reference_stats, LoadedModel):
        if not reference_stats.is_white_box:
            raise ValueError("Reference model must be a white-box TorchScript model.")
        ref_stats = compute_activation_stats(reference_stats, inputs)
    else:
        ref_stats = reference_stats

    flags: list[Flag] = []

    # Compare stats for overlapping layers
    for layer_name, t_stat in test_stats.items():
        if layer_name not in ref_stats:
            continue
        
        r_stat = ref_stats[layer_name]
        
        t_mean, t_var = t_stat["mean"], t_stat["var"]
        r_mean, r_var = r_stat["mean"], r_stat["var"]

        # Calculate relative difference
        mean_diff = abs(t_mean - r_mean) / (abs(r_mean) + 1e-6)
        var_diff = abs(t_var - r_var) / (abs(r_var) + 1e-6)

        if mean_diff > threshold or var_diff > threshold:
            max_diff = max(mean_diff, var_diff)
            confidence = float(np.clip(max_diff / (threshold * 2.0), 0.5, 1.0))
            
            flags.append(
                Flag(
                    module=ModuleName.MODEL_INTEGRITY,
                    reason=(
                        f"Activation anomaly at layer '{layer_name}'. "
                        f"Relative mean diff: {mean_diff:.4f}, var diff: {var_diff:.4f} "
                        f"(threshold: {threshold:.4f})."
                    ),
                    evidence={
                        "layer_name": layer_name,
                        "test_mean": t_mean,
                        "ref_mean": r_mean,
                        "test_var": t_var,
                        "ref_var": r_var,
                        "mean_diff": mean_diff,
                        "var_diff": var_diff,
                        "access_mode_used": AccessMode.WHITE_BOX,
                    },
                    confidence=confidence,
                    severity=Severity.HIGH if max_diff > threshold * 2 else Severity.MEDIUM,
                    affected_asset=str(model.model_path),
                    recommended_disposition=Disposition.REVIEW,
                )
            )

    return flags
