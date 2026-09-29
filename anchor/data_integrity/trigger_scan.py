"""Frequency-domain backdoor trigger scanner.

Uses 2D Fast Fourier Transform (scipy.fft.fft2) to analyze images in the
frequency domain, scanning for localized blocks with suspicious high-frequency
energy concentrations or periodic patterns consistent with visual backdoor patches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.fft as fft

from anchor.report_schema.schema import (
    Disposition,
    Flag,
    ModuleName,
    Severity,
)


def _analyze_image_frequency_blocks(
    image_path: str | Path,
    block_size: int = 64,
    high_freq_radius_ratio: float = 0.35,
) -> tuple[float, float, dict[str, Any]]:
    """Analyze the image by splitting it into blocks and computing the frequency ratio.

    Returns:
        max_ratio: The maximum high-frequency energy ratio found in any block.
        avg_ratio: The average high-frequency energy ratio across all blocks.
        metadata: Detailed block-by-block statistics.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    h, w = img.shape
    # Resize slightly if too small for block_size
    if h < block_size or w < block_size:
        img = cv2.resize(img, (max(w, block_size), max(h, block_size)))
        h, w = img.shape

    # Normalize image to [0, 1]
    img = img.astype(np.float32) / 255.0

    block_ratios = []
    block_coords = []

    # Center of the block frequency domain
    cx, cy = block_size // 2, block_size // 2
    r_limit = high_freq_radius_ratio * (block_size / 2)

    # Grid of coordinates in the frequency block
    y_indices, x_indices = np.ogrid[:block_size, :block_size]
    dist_from_center = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
    # Mask defining high-frequency region
    high_freq_mask = dist_from_center >= r_limit

    # Slide over non-overlapping blocks
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = img[y : y + block_size, x : x + block_size]
            
            # 2D FFT and Shift zero-frequency component to center
            f_transform = fft.fft2(block)
            f_shift = fft.fftshift(f_transform)
            
            # Power spectrum
            power_spectrum = np.abs(f_shift) ** 2
            total_power = np.sum(power_spectrum)

            if total_power == 0:
                ratio = 0.0
            else:
                high_freq_power = np.sum(power_spectrum[high_freq_mask])
                ratio = float(high_freq_power / total_power)

            block_ratios.append(ratio)
            block_coords.append((x, y))

    if not block_ratios:
        return 0.0, 0.0, {}

    max_ratio = max(block_ratios)
    avg_ratio = sum(block_ratios) / len(block_ratios)
    
    max_idx = block_ratios.index(max_ratio)
    max_coord = block_coords[max_idx]

    return max_ratio, avg_ratio, {
        "max_block_ratio": max_ratio,
        "avg_block_ratio": avg_ratio,
        "max_block_coords": max_coord,
        "block_size": block_size,
    }


def scan_for_triggers(
    image_paths: list[str | Path],
    block_size: int = 64,
    high_freq_radius_ratio: float = 0.35,
    ratio_threshold: float = 0.25,
    outlier_factor: float = 2.0,
) -> list[Flag]:
    """Scan images for backdoor triggers by analyzing localized frequency anomalies.

    Parameters
    ----------
    image_paths :
        Paths to the images to scan.
    block_size :
        Size of the square blocks for localized analysis.
    high_freq_radius_ratio :
        The ratio of radius from the center defining the high-frequency boundary.
        (e.g., 0.35 means frequencies further than 35% of max radius are 'high').
    ratio_threshold :
        Absolute ratio threshold. If a block's high-frequency ratio exceeds this,
        and it's an outlier, it's flagged.
    outlier_factor :
        How much higher a block's ratio must be compared to the average ratio of
        all blocks in that same image to be considered an outlier.

    Returns
    -------
    list[Flag]
    """
    flags: list[Flag] = []

    for path in image_paths:
        try:
            max_ratio, avg_ratio, metadata = _analyze_image_frequency_blocks(
                path, block_size, high_freq_radius_ratio
            )
        except Exception as e:
            # Skip corrupted files or log them if necessary
            continue

        # Flag if the max block is both above absolute threshold AND an outlier
        is_trigger = max_ratio > ratio_threshold and max_ratio > (avg_ratio * outlier_factor)

        if is_trigger:
            confidence = float(np.clip((max_ratio - ratio_threshold) / (1.0 - ratio_threshold + 1e-6), 0.0, 1.0))
            # Boost confidence for extreme outliers
            confidence = max(confidence, 0.5)

            flags.append(
                Flag(
                    module=ModuleName.DATA_INTEGRITY,
                    reason=(
                        f"Backdoor trigger pattern detected. Localized high-frequency energy ratio "
                        f"({max_ratio:.4f}) is significantly higher than image average ({avg_ratio:.4f})."
                    ),
                    evidence={
                        "image_path": str(path),
                        "max_block_ratio": max_ratio,
                        "avg_block_ratio": avg_ratio,
                        "block_coords_x_y": metadata.get("max_block_coords"),
                        "block_size": block_size,
                    },
                    confidence=confidence,
                    severity=Severity.HIGH if max_ratio > 0.4 else Severity.MEDIUM,
                    affected_asset=str(path),
                    recommended_disposition=Disposition.QUARANTINE,
                )
            )

    return flags
