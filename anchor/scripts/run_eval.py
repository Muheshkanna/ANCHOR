#!/usr/bin/env python3
"""CLI entrypoint that runs the full Anchor assurance pipeline.

Usage (inside the container):
    python /app/scripts/run_eval.py \
        --dataset  /app/reference_data/eval_set \
        --model    /app/reference_data/model.pt \
        --ref      /app/reference_data/ref_set \
        --db       /app/reference_data/provenance.db \
        --out      /app/reference_data/outputs

Exit codes:
    0  — pipeline completed, overall recommendation is ACCEPT
    1  — pipeline completed, recommendation is REVIEW or QUARANTINE
    2  — pipeline raised an unhandled exception
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path both inside container (/app) and locally
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from anchor.orchestrator.pipeline import run_pipeline
from anchor.orchestrator.report_builder import build_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Anchor CV Integrity — full pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", required=True,
        help="Path to evaluation dataset (images/ + labels/ YOLO subdirs).",
    )
    p.add_argument(
        "--model", required=True,
        help="Path to model weights file (.pt TorchScript or .onnx).",
    )
    p.add_argument(
        "--ref", required=True,
        help="Path to reference/baseline dataset (images/ + labels/ YOLO subdirs).",
    )
    p.add_argument(
        "--db", required=True,
        help="Path to SQLite provenance ChainLog database.",
    )
    p.add_argument(
        "--out", default="/app/reference_data/outputs",
        help="Output directory for JSON and HTML reports.",
    )
    p.add_argument(
        "--dataset-id", default="docker_eval_dataset",
        help="Dataset identifier string written into the report.",
    )
    p.add_argument(
        "--model-id", default="docker_eval_model",
        help="Model identifier string written into the report.",
    )
    p.add_argument(
        "--report-name", default="assurance_report",
        help="Basename for the output report files (no extension).",
    )
    return p.parse_args()


def severity_order(s: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(s, 3)


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("  Anchor AI Integrity Assurance — Pipeline Run")
    print("=" * 60)
    print(f"  Dataset  : {args.dataset}")
    print(f"  Model    : {args.model}")
    print(f"  Reference: {args.ref}")
    print(f"  DB       : {args.db}")
    print(f"  Output   : {args.out}")
    print("=" * 60)
    print()

    try:
        print("[1/2] Running detection modules …")
        module_results = run_pipeline(
            dataset_dir=args.dataset,
            model_path=args.model,
            reference_dir=args.ref,
            db_path=args.db,
        )

        total_flags = sum(len(r.flags) for r in module_results)
        print(f"      Done - {total_flags} flag(s) raised across {len(module_results)} module(s).")
        print()

        print("[2/2] Building AssuranceReport ...")
        report = build_report(
            dataset_id=args.dataset_id,
            model_id=args.model_id,
            module_results=module_results,
            audit_log_ref=str(args.db),
            output_dir=args.out,
            report_name=args.report_name,
        )
        print(f"      JSON -> {Path(args.out) / (args.report_name + '.json')}")
        print(f"      HTML -> {Path(args.out) / (args.report_name + '.html')}")
        print()

        # -- Print summary -----------------------------------------------------
        rec = report.overall_recommendation.value.upper()
        print("-" * 60)
        print(f"  OVERALL RECOMMENDATION : {rec}")
        print("-" * 60)

        for result in module_results:
            sorted_flags = sorted(
                result.flags,
                key=lambda f: severity_order(f.severity.value),
            )
            print(f"\n  [{result.module.value.upper()}]"
                  f"  {len(sorted_flags)} flag(s)"
                  f"  (access: {result.access_mode_used.value})")
            for flag in sorted_flags:
                sev = flag.severity.value.upper()
                print(f"    [{sev:6s}] {flag.reason[:70]}")
                print(f"             asset: {flag.affected_asset}")
                print(f"             disp : {flag.recommended_disposition.value}"
                      f"  confidence: {flag.confidence:.0%}")

        print()
        print("=" * 60)

        # Exit code reflects recommendation
        if rec == "ACCEPT":
            print("  STATUS: ACCEPT — pipeline exiting 0")
            print("=" * 60)
            return 0
        else:
            print(f"  STATUS: {rec} — pipeline exiting 1")
            print("=" * 60)
            return 1

    except Exception as exc:
        print(f"\n[ERROR] Pipeline raised an unhandled exception: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
