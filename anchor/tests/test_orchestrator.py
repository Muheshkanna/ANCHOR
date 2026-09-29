"""Tests for orchestrator/pipeline.py and orchestrator/report_builder.py.

These tests mock the CLIP-dependent detectors so the suite runs fully offline
in under a few seconds, while still exercising the orchestrator's actual
responsibilities: module sequencing, flag collection, report construction,
and JSON/HTML output.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from anchor.orchestrator.report_builder import build_report
from anchor.report_schema.schema import (
    AccessMode,
    Disposition,
    Flag,
    ModuleName,
    ModuleResult,
    Severity,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_flag(module: ModuleName, severity: Severity) -> Flag:
    return Flag(
        module=module,
        reason="test flag",
        evidence={"key": "value"},
        confidence=0.9,
        severity=severity,
        affected_asset="test_asset",
        recommended_disposition=Disposition.REVIEW,
    )


def _make_module_result(module: ModuleName, flags: list[Flag]) -> ModuleResult:
    return ModuleResult(
        module=module,
        flags=flags,
        coverage_statement="Test coverage statement.",
        access_mode_used=AccessMode.BLACK_BOX,
    )


# ── report_builder tests ──────────────────────────────────────────────────────

class TestReportBuilder:
    def test_overall_rec_accept_when_no_flags(self, tmp_path: Path) -> None:
        results = [
            _make_module_result(ModuleName.DATA_INTEGRITY, []),
            _make_module_result(ModuleName.MODEL_INTEGRITY, []),
        ]
        report = build_report(
            dataset_id="ds_001",
            model_id="model_001",
            module_results=results,
            audit_log_ref="ref://test",
            output_dir=tmp_path,
        )
        assert report.overall_recommendation == Disposition.ACCEPT

    def test_overall_rec_review_when_medium_flag(self, tmp_path: Path) -> None:
        results = [
            _make_module_result(ModuleName.DATA_INTEGRITY, [
                _make_flag(ModuleName.DATA_INTEGRITY, Severity.MEDIUM),
            ]),
            _make_module_result(ModuleName.MODEL_INTEGRITY, []),
        ]
        report = build_report(
            dataset_id="ds_001",
            model_id="model_001",
            module_results=results,
            audit_log_ref="ref://test",
            output_dir=tmp_path,
        )
        assert report.overall_recommendation == Disposition.REVIEW

    def test_overall_rec_quarantine_when_high_flag(self, tmp_path: Path) -> None:
        results = [
            _make_module_result(ModuleName.DATA_INTEGRITY, [
                _make_flag(ModuleName.DATA_INTEGRITY, Severity.LOW),
            ]),
            _make_module_result(ModuleName.MODEL_INTEGRITY, [
                _make_flag(ModuleName.MODEL_INTEGRITY, Severity.HIGH),
            ]),
        ]
        report = build_report(
            dataset_id="ds_001",
            model_id="model_001",
            module_results=results,
            audit_log_ref="ref://test",
            output_dir=tmp_path,
        )
        assert report.overall_recommendation == Disposition.QUARANTINE

    def test_high_flag_overrides_medium(self, tmp_path: Path) -> None:
        """HIGH always wins over MEDIUM when both present."""
        results = [
            _make_module_result(ModuleName.DATA_INTEGRITY, [
                _make_flag(ModuleName.DATA_INTEGRITY, Severity.MEDIUM),
                _make_flag(ModuleName.DATA_INTEGRITY, Severity.HIGH),
            ]),
        ]
        report = build_report(
            dataset_id="ds_001",
            model_id="model_001",
            module_results=results,
            audit_log_ref="ref://test",
            output_dir=tmp_path,
        )
        assert report.overall_recommendation == Disposition.QUARANTINE

    def test_json_output_written(self, tmp_path: Path) -> None:
        results = [_make_module_result(ModuleName.PROVENANCE, [])]
        build_report(
            dataset_id="ds_json",
            model_id="model_json",
            module_results=results,
            audit_log_ref="ref://json_test",
            output_dir=tmp_path,
            report_name="test_report",
        )
        json_path = tmp_path / "test_report.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["dataset_id"] == "ds_json"
        assert data["model_id"] == "model_json"
        assert "timestamp" in data
        assert "overall_recommendation" in data

    def test_html_output_written(self, tmp_path: Path) -> None:
        results = [_make_module_result(ModuleName.DISTRIBUTION_SHIFT, [])]
        build_report(
            dataset_id="ds_html",
            model_id="model_html",
            module_results=results,
            audit_log_ref="ref://html_test",
            output_dir=tmp_path,
            report_name="test_report",
        )
        html_path = tmp_path / "test_report.html"
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "Anchor Assurance Report" in content
        assert "distribution shift" in content.lower()

    def test_html_shows_quarantine_recommendation(self, tmp_path: Path) -> None:
        results = [
            _make_module_result(ModuleName.MODEL_INTEGRITY, [
                _make_flag(ModuleName.MODEL_INTEGRITY, Severity.HIGH),
            ]),
        ]
        build_report(
            dataset_id="ds_q",
            model_id="model_q",
            module_results=results,
            audit_log_ref="ref://quarantine",
            output_dir=tmp_path,
            report_name="q_report",
        )
        content = (tmp_path / "q_report.html").read_text(encoding="utf-8")
        assert "quarantine" in content.lower()

    def test_report_schema_fields_populated(self, tmp_path: Path) -> None:
        results = [_make_module_result(ModuleName.PROVENANCE, [])]
        report = build_report(
            dataset_id="ds_fields",
            model_id="model_fields",
            module_results=results,
            audit_log_ref="ipfs://QmTest",
            output_dir=tmp_path,
        )
        assert report.dataset_id == "ds_fields"
        assert report.model_id == "model_fields"
        assert report.audit_log_ref == "ipfs://QmTest"
        assert len(report.module_results) == 1
        assert report.timestamp is not None

    def test_output_dir_created_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        build_report(
            dataset_id="x",
            model_id="y",
            module_results=[],
            audit_log_ref="ref://mkdir",
            output_dir=nested,
        )
        assert nested.exists()


# ── pipeline.py unit tests (mocked) ─────────────────────────────────────────

class TestPipelineMocked:
    """Test the orchestrator pipeline with all network/CLIP calls mocked out."""

    def _make_dataset_dirs(self, base: Path) -> tuple[Path, Path]:
        """Create minimal YOLO dataset dirs with 2 synthetic images each."""
        import cv2

        for name in ("test_ds", "ref_ds"):
            ds = base / name
            img_dir = ds / "images"
            lbl_dir = ds / "labels"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            for i in range(2):
                img = np.zeros((32, 32, 3), dtype=np.uint8)
                cv2.circle(img, (16, 16), 8, (200, 100, 50), -1)
                cv2.imwrite(str(img_dir / f"img_{i}.png"), img)
                (lbl_dir / f"img_{i}.txt").write_text(f"{i % 3} 0.5 0.5 0.4 0.4\n")

        return base / "test_ds", base / "ref_ds"

    def _make_model(self, tmp_path: Path) -> Path:
        import torch
        import torch.nn as nn

        net = nn.Linear(3, 10)
        scripted = torch.jit.script(net)
        pt_path = tmp_path / "model.pt"
        scripted.save(str(pt_path))
        return pt_path

    def test_pipeline_returns_four_module_results(self, tmp_path: Path) -> None:
        from anchor.orchestrator.pipeline import run_pipeline

        test_ds, ref_ds = self._make_dataset_dirs(tmp_path)
        model_path = self._make_model(tmp_path)
        db_path = tmp_path / "chain.db"

        fake_embs = np.random.rand(2, 16).astype(np.float32)

        with (
            patch("anchor.orchestrator.pipeline.embed_batch", return_value=fake_embs),
            patch("anchor.orchestrator.pipeline.detect_duplicates", return_value=[]),
            patch("anchor.orchestrator.pipeline.detect_ood", return_value=[]),
            patch("anchor.orchestrator.pipeline.detect_label_flips", return_value=[]),
            patch("anchor.orchestrator.pipeline.scan_for_triggers", return_value=[]),
            patch("anchor.orchestrator.pipeline.reconstruct_triggers", return_value=[]),
            patch("anchor.orchestrator.pipeline.check_fingerprint", return_value=[]),
            patch("anchor.orchestrator.pipeline.compare_activation_stats", return_value=[]),
            patch("anchor.orchestrator.pipeline.detect_shift", return_value=[]),
        ):
            results = run_pipeline(
                dataset_dir=test_ds,
                model_path=model_path,
                reference_dir=ref_ds,
                db_path=db_path,
            )

        assert len(results) == 4
        modules = {r.module for r in results}
        assert modules == {
            ModuleName.DATA_INTEGRITY,
            ModuleName.MODEL_INTEGRITY,
            ModuleName.PROVENANCE,
            ModuleName.DISTRIBUTION_SHIFT,
        }

    def test_pipeline_collects_flags_from_all_modules(self, tmp_path: Path) -> None:
        from anchor.orchestrator.pipeline import run_pipeline

        test_ds, ref_ds = self._make_dataset_dirs(tmp_path)
        model_path = self._make_model(tmp_path)
        db_path = tmp_path / "chain.db"

        fake_embs = np.random.rand(2, 16).astype(np.float32)
        dup_flag = _make_flag(ModuleName.DATA_INTEGRITY, Severity.HIGH)
        shift_flag = _make_flag(ModuleName.DISTRIBUTION_SHIFT, Severity.MEDIUM)

        with (
            patch("anchor.orchestrator.pipeline.embed_batch", return_value=fake_embs),
            patch("anchor.orchestrator.pipeline.detect_duplicates", return_value=[dup_flag]),
            patch("anchor.orchestrator.pipeline.detect_ood", return_value=[]),
            patch("anchor.orchestrator.pipeline.detect_label_flips", return_value=[]),
            patch("anchor.orchestrator.pipeline.scan_for_triggers", return_value=[]),
            patch("anchor.orchestrator.pipeline.reconstruct_triggers", return_value=[]),
            patch("anchor.orchestrator.pipeline.check_fingerprint", return_value=[]),
            patch("anchor.orchestrator.pipeline.compare_activation_stats", return_value=[]),
            patch("anchor.orchestrator.pipeline.detect_shift", return_value=[shift_flag]),
        ):
            results = run_pipeline(
                dataset_dir=test_ds,
                model_path=model_path,
                reference_dir=ref_ds,
                db_path=db_path,
            )

        data_result = next(r for r in results if r.module == ModuleName.DATA_INTEGRITY)
        shift_result = next(r for r in results if r.module == ModuleName.DISTRIBUTION_SHIFT)
        assert any(f.severity == Severity.HIGH for f in data_result.flags)
        assert any(f.severity == Severity.MEDIUM for f in shift_result.flags)

    def test_pipeline_survives_embed_failure(self, tmp_path: Path) -> None:
        """If CLIP embedding fails, pipeline returns 4 results with skip flags."""
        from anchor.orchestrator.pipeline import run_pipeline

        test_ds, ref_ds = self._make_dataset_dirs(tmp_path)
        model_path = self._make_model(tmp_path)
        db_path = tmp_path / "chain.db"

        with (
            patch("anchor.orchestrator.pipeline.embed_batch", side_effect=RuntimeError("no network")),
            patch("anchor.orchestrator.pipeline.detect_duplicates", return_value=[]),
            patch("anchor.orchestrator.pipeline.scan_for_triggers", return_value=[]),
            patch("anchor.orchestrator.pipeline.reconstruct_triggers", return_value=[]),
            patch("anchor.orchestrator.pipeline.check_fingerprint", return_value=[]),
            patch("anchor.orchestrator.pipeline.compare_activation_stats", return_value=[]),
        ):
            results = run_pipeline(
                dataset_dir=test_ds,
                model_path=model_path,
                reference_dir=ref_ds,
                db_path=db_path,
            )

        assert len(results) == 4
        # Distribution shift module should have a skip flag
        shift_result = next(r for r in results if r.module == ModuleName.DISTRIBUTION_SHIFT)
        assert any("skipped" in f.reason.lower() for f in shift_result.flags)

    def test_pipeline_detects_tampered_chain(self, tmp_path: Path) -> None:
        """A tampered chain-log should surface as a HIGH provenance flag."""
        from anchor.orchestrator.pipeline import run_pipeline
        from anchor.provenance.chain_log import ChainLog

        test_ds, ref_ds = self._make_dataset_dirs(tmp_path)
        model_path = self._make_model(tmp_path)
        db_path = tmp_path / "chain.db"

        # Build a valid chain then tamper with it
        log = ChainLog(db_path)
        log.append({"step": "init"})
        log.append({"step": "run"})
        # Tamper: directly modify a record in the DB
        import sqlite3
        with sqlite3.connect(db_path) as con:
            con.execute("UPDATE chain_log SET payload = ? WHERE seq = 1", ('{"step":"tampered"}',))
            con.commit()
        log.close()

        fake_embs = np.random.rand(2, 16).astype(np.float32)

        with (
            patch("anchor.orchestrator.pipeline.embed_batch", return_value=fake_embs),
            patch("anchor.orchestrator.pipeline.detect_duplicates", return_value=[]),
            patch("anchor.orchestrator.pipeline.detect_ood", return_value=[]),
            patch("anchor.orchestrator.pipeline.detect_label_flips", return_value=[]),
            patch("anchor.orchestrator.pipeline.scan_for_triggers", return_value=[]),
            patch("anchor.orchestrator.pipeline.reconstruct_triggers", return_value=[]),
            patch("anchor.orchestrator.pipeline.check_fingerprint", return_value=[]),
            patch("anchor.orchestrator.pipeline.compare_activation_stats", return_value=[]),
            patch("anchor.orchestrator.pipeline.detect_shift", return_value=[]),
        ):
            results = run_pipeline(
                dataset_dir=test_ds,
                model_path=model_path,
                reference_dir=ref_ds,
                db_path=db_path,
            )

        prov = next(r for r in results if r.module == ModuleName.PROVENANCE)
        assert any(f.severity == Severity.HIGH for f in prov.flags), (
            "Expected a HIGH flag for tampered chain-log"
        )

    def test_full_pipeline_to_report(self, tmp_path: Path) -> None:
        """End-to-end smoke test: pipeline → report_builder writes JSON + HTML."""
        from anchor.orchestrator.pipeline import run_pipeline

        test_ds, ref_ds = self._make_dataset_dirs(tmp_path)
        model_path = self._make_model(tmp_path)
        db_path = tmp_path / "chain.db"
        output_dir = tmp_path / "reports"

        fake_embs = np.random.rand(2, 16).astype(np.float32)

        with (
            patch("anchor.orchestrator.pipeline.embed_batch", return_value=fake_embs),
            patch("anchor.orchestrator.pipeline.detect_duplicates", return_value=[]),
            patch("anchor.orchestrator.pipeline.detect_ood", return_value=[]),
            patch("anchor.orchestrator.pipeline.detect_label_flips", return_value=[]),
            patch("anchor.orchestrator.pipeline.scan_for_triggers", return_value=[]),
            patch("anchor.orchestrator.pipeline.reconstruct_triggers", return_value=[]),
            patch("anchor.orchestrator.pipeline.check_fingerprint", return_value=[]),
            patch("anchor.orchestrator.pipeline.compare_activation_stats", return_value=[]),
            patch("anchor.orchestrator.pipeline.detect_shift", return_value=[]),
        ):
            results = run_pipeline(
                dataset_dir=test_ds,
                model_path=model_path,
                reference_dir=ref_ds,
                db_path=db_path,
            )

        report = build_report(
            dataset_id="smoke_ds",
            model_id="smoke_model",
            module_results=results,
            audit_log_ref="ref://smoke",
            output_dir=output_dir,
            report_name="smoke_report",
        )

        assert (output_dir / "smoke_report.json").exists()
        assert (output_dir / "smoke_report.html").exists()
        assert report.dataset_id == "smoke_ds"
