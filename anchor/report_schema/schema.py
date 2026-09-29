"""Pydantic data models for the Anchor assurance report pipeline.

These models define the structured output that every detection module
produces and that the orchestrator aggregates into a final report.
No detection logic lives here — modules import and populate these schemas.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Enumerations ─────────────────────────────────────────────────────────────

class ModuleName(str, Enum):
    """Detection modules that can raise flags."""

    DATA_INTEGRITY = "data_integrity"
    MODEL_INTEGRITY = "model_integrity"
    PROVENANCE = "provenance"
    DISTRIBUTION_SHIFT = "distribution_shift"


class Severity(str, Enum):
    """Severity level assigned to an individual flag."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Disposition(str, Enum):
    """Recommended action for a flagged asset."""

    ACCEPT = "accept"
    REVIEW = "review"
    QUARANTINE = "quarantine"


class AccessMode(str, Enum):
    """Whether the module operated with full model internals or not."""

    WHITE_BOX = "white_box"
    BLACK_BOX = "black_box"


# ── Core models ──────────────────────────────────────────────────────────────

class Flag(BaseModel):
    """A single integrity concern raised by a detection module."""

    id: UUID = Field(default_factory=uuid4, description="Unique flag identifier.")
    module: ModuleName = Field(
        ..., description="The module that raised this flag."
    )
    reason: str = Field(
        ..., description="Human-readable explanation of the concern."
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured evidence supporting the flag (metrics, paths, hashes, etc.).",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Module confidence in this flag (0–1)."
    )
    severity: Severity = Field(
        ..., description="Assessed severity of the concern."
    )
    affected_asset: str = Field(
        ..., description="Identifier of the asset affected (image path, weight file, etc.)."
    )
    recommended_disposition: Disposition = Field(
        ..., description="Suggested action for this asset."
    )


class ModuleResult(BaseModel):
    """Aggregate output of a single detection module's evaluation run."""

    module: ModuleName = Field(
        ..., description="Which module produced this result."
    )
    flags: list[Flag] = Field(
        default_factory=list, description="All flags raised during the run."
    )
    coverage_statement: str = Field(
        ...,
        description=(
            "Plain-language statement of what this module does NOT check, "
            "so downstream consumers understand the scope boundary."
        ),
    )
    access_mode_used: AccessMode = Field(
        ..., description="Whether the module had white-box or black-box access."
    )


class AssuranceReport(BaseModel):
    """Top-level report aggregating results from all detection modules."""

    dataset_id: str = Field(
        ..., description="Identifier of the evaluated dataset."
    )
    model_id: str = Field(
        ..., description="Identifier of the evaluated model."
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the report was generated.",
    )
    module_results: list[ModuleResult] = Field(
        default_factory=list,
        description="Per-module evaluation results.",
    )
    overall_recommendation: Disposition = Field(
        ...,
        description="Aggregated recommended disposition for the model+dataset pair.",
    )
    audit_log_ref: str = Field(
        ...,
        description="URI or path to the immutable audit log entry for this evaluation.",
    )
