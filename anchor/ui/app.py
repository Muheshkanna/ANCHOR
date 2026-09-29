"""Anchor AI Integrity Assurance — Streamlit Dashboard.

Entry point:
    cd d:\\ANCHOR
    streamlit run anchor/ui/app.py

Design decisions
----------------
* All pipeline execution goes through ``orchestrator.pipeline.run_pipeline``
  and ``orchestrator.report_builder.build_report``.
* The audit-log tab calls ``provenance.chain_log.ChainLog`` directly because
  the orchestrator does not expose a standalone chain-verification function;
  this is a read-only, side-effect-free operation.
* ``st.session_state`` stores the pipeline result so it survives widget
  interactions without re-running.
* Severity ordering: HIGH > MEDIUM > LOW — the flags table default-sorts by
  this order.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Make sure the repo root is importable when launched as ``streamlit run``
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Anchor — AI Integrity Assurance",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (keep startup fast) ─────────────────────────────────────────
from anchor.orchestrator.pipeline import run_pipeline
from anchor.orchestrator.report_builder import build_report
from anchor.provenance.chain_log import ChainLog

# ── Severity helpers ─────────────────────────────────────────────────────────
_SEV_ORDER = {"high": 0, "medium": 1, "low": 2}
_SEV_EMOJI = {"high": "🔴", "medium": "🟡", "low": "⚪"}
_SEV_COLOR = {"high": "#ef4444", "medium": "#f59e0b", "low": "#6b7280"}
_REC_EMOJI = {"quarantine": "🚫", "review": "⚠️", "accept": "✅"}
_REC_COLOR = {"quarantine": "#ef4444", "review": "#f59e0b", "accept": "#10b981"}
_MODULE_LABELS = {
    "data_integrity": "Data Integrity",
    "model_integrity": "Model Integrity",
    "provenance": "Provenance",
    "distribution_shift": "Distribution Shift",
}

# ── Global CSS & Font Injection (Offline via ui/styles.css) ───────────────────
_CURRENT_DIR = Path(__file__).resolve().parent
_CSS_PATH = _CURRENT_DIR / "styles.css"
_FONTS_CSS_PATH = _CURRENT_DIR / "fonts_base64.css"

_css_payload = []
if _FONTS_CSS_PATH.exists():
    _css_payload.append(f"<style>\n{_FONTS_CSS_PATH.read_text(encoding='utf-8')}\n</style>")
if _CSS_PATH.exists():
    _css_payload.append(f"<style>\n{_CSS_PATH.read_text(encoding='utf-8')}\n</style>")

if _css_payload:
    st.markdown("\n".join(_css_payload), unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────
if "report" not in st.session_state:
    st.session_state.report = None
if "module_results" not in st.session_state:
    st.session_state.module_results = None
if "report_dir" not in st.session_state:
    st.session_state.report_dir = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Configuration
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔐 Anchor Config")
    st.markdown("---")

    # Default sample paths using current repository structure
    _REF_DATA = _REPO_ROOT / "anchor" / "reference_data"
    _DEFAULT_CLEAN = str(_REF_DATA / "clean")
    _DEFAULT_BACKDOOR = str(_REF_DATA / "poisoned_backdoor")
    _DEFAULT_LABEL_FLIP = str(_REF_DATA / "poisoned_label_flip")
    _DEFAULT_DUPLICATE = str(_REF_DATA / "poisoned_duplicate")
    _DEFAULT_MODEL = str(_REF_DATA / "backdoored_model.pt")
    _DEFAULT_DB = str(_REF_DATA / "provenance_chain.db")

    preset = st.selectbox(
        "⚡ Quick Preset / Scenario",
        [
            "Backdoor Attack (Poisoned)",
            "Label-Flip Attack",
            "Near-Duplicate Attack",
            "Clean Baseline",
        ],
        help="Select a scenario to automatically fill the fields with sample data.",
    )

    if preset == "Backdoor Attack (Poisoned)":
        pre_dataset = _DEFAULT_BACKDOOR
        pre_model = _DEFAULT_MODEL
        pre_dataset_id = "poisoned_backdoor_v1"
        pre_model_id = "backdoored_cnn_pt"
    elif preset == "Label-Flip Attack":
        pre_dataset = _DEFAULT_LABEL_FLIP
        pre_model = _DEFAULT_MODEL
        pre_dataset_id = "poisoned_label_flip_v1"
        pre_model_id = "model_eval_v1"
    elif preset == "Near-Duplicate Attack":
        pre_dataset = _DEFAULT_DUPLICATE
        pre_model = _DEFAULT_MODEL
        pre_dataset_id = "poisoned_duplicate_v1"
        pre_model_id = "model_eval_v1"
    else:
        pre_dataset = _DEFAULT_CLEAN
        pre_model = _DEFAULT_MODEL
        pre_dataset_id = "clean_cifar10_set"
        pre_model_id = "cifar10_baseline"

    dataset_dir = st.text_input(
        "Dataset directory",
        value=pre_dataset,
        help="Must contain images/ and labels/ sub-directories (YOLO format).",
    )
    model_path = st.text_input(
        "Model file (.pt or .onnx)",
        value=pre_model,
    )
    reference_dir = st.text_input(
        "Reference dataset directory",
        value=_DEFAULT_CLEAN,
        help="Baseline clean dataset used for OOD / shift checks.",
    )
    db_path = st.text_input(
        "Provenance DB path",
        value=_DEFAULT_DB,
        help="SQLite ChainLog database (created if absent).",
    )

    st.markdown("---")
    st.markdown("**Report IDs** *(optional)*")
    dataset_id = st.text_input("Dataset ID", value=pre_dataset_id)
    model_id   = st.text_input("Model ID",   value=pre_model_id)

    st.markdown("---")
    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR / HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="anchor-topbar">
  <div class="anchor-brand">
    <h1 class="anchor-title">🔐 Anchor — AI Integrity Dashboard</h1>
    <span class="anchor-subtitle">Computer-Vision Integrity Assurance Framework &nbsp;·&nbsp;
     Data Integrity · Model Integrity · Provenance · Distribution Shift</span>
  </div>
  <div class="anchor-status-pill">
    <span class="anchor-status-dot"></span>
    <span>SECURITY ASSURANCE ACTIVE</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    # Validate inputs
    errors = []
    if not dataset_dir:
        errors.append("Dataset directory is required.")
    if not model_path:
        errors.append("Model file path is required.")
    if not reference_dir:
        errors.append("Reference directory is required.")
    if not db_path:
        errors.append("Provenance DB path is required.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        report_dir = Path(tempfile.mkdtemp(prefix="anchor_report_"))
        st.session_state.report_dir = report_dir

        with st.status("Running Anchor assurance pipeline…", expanded=True) as status:
            try:
                st.write("📂 Parsing datasets & loading model…")
                module_results = run_pipeline(
                    dataset_dir=dataset_dir,
                    model_path=model_path,
                    reference_dir=reference_dir,
                    db_path=db_path,
                )
                st.write(f"✅ Pipeline complete — {sum(len(r.flags) for r in module_results)} flag(s) raised across {len(module_results)} modules.")

                st.write("📄 Building AssuranceReport (JSON + HTML)…")
                report = build_report(
                    dataset_id=dataset_id,
                    model_id=model_id,
                    module_results=module_results,
                    audit_log_ref=str(db_path),
                    output_dir=report_dir,
                    report_name="assurance_report",
                )
                st.write(f"✅ Reports saved to `{report_dir}`")

                st.session_state.report = report
                st.session_state.module_results = module_results
                status.update(label="Pipeline finished!", state="complete")

            except Exception as exc:
                status.update(label="Pipeline failed.", state="error")
                st.exception(exc)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_report, tab_flags, tab_audit = st.tabs([
    "📋 Assurance Report",
    "🚩 Flags Explorer",
    "🔗 Audit Log",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Assurance Report Summary
# ─────────────────────────────────────────────────────────────────────────────
with tab_report:
    report = st.session_state.report

    if report is None:
        st.info("Configure a dataset and model in the sidebar, then click **▶ Run Pipeline**.")
        st.stop()

    rec = report.overall_recommendation.value
    rec_emoji = _REC_EMOJI.get(rec, "❓")

    # Recommendation banner with 4px left border & pill
    st.markdown(
        f'<div class="verdict-banner rec-{rec}">'
        f'  <div class="verdict-title">{rec_emoji} Overall Recommendation: {rec.upper()}</div>'
        f'  <div class="verdict-badge rec-{rec}">{rec.upper()}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Metadata grid (4 compact KPI tiles with small uppercase labels and tabular numbers)
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, label, value in [
        (mc1, "Model ID",    report.model_id),
        (mc2, "Dataset ID",  report.dataset_id),
        (mc3, "Timestamp",   report.timestamp.strftime("%Y-%m-%d %H:%M UTC")),
        (mc4, "Audit Ref",   report.audit_log_ref),
    ]:
        with col:
            st.markdown(
                f'<div class="kpi-tile">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value" title="{value}">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Per-module summary cards
    st.subheader("Module Results")
    all_flags = []
    for result in report.module_results:
        all_flags.extend(result.flags)
        high   = sum(1 for f in result.flags if f.severity.value == "high")
        medium = sum(1 for f in result.flags if f.severity.value == "medium")
        low    = sum(1 for f in result.flags if f.severity.value == "low")
        total  = len(result.flags)

        with st.expander(
            f"{_MODULE_LABELS.get(result.module.value, result.module.value)}  "
            f"— {total} flag{'s' if total != 1 else ''}  "
            f"({'🔴' * high}{'🟡' * medium}{'⚪' * low if low and not (high or medium) else ''})",
            expanded=(total > 0),
        ):
            st.caption(f"**Access mode:** `{result.access_mode_used.value}`")
            st.markdown(f"> **Coverage scope:** {result.coverage_statement}")

            if not result.flags:
                st.success("No flags raised by this module.")
            else:
                for flag in sorted(result.flags, key=lambda f: _SEV_ORDER[f.severity.value]):
                    sev = flag.severity.value
                    st.markdown(
                        f'<div class="flag-row">'
                        f'<span class="sev-pill sev-{sev}">{_SEV_EMOJI[sev]} {sev.upper()}</span>'
                        f'&nbsp;&nbsp;<strong>{flag.reason}</strong><br>'
                        f'<small style="color:var(--text-secondary)">Affected: <code>{flag.affected_asset}</code>'
                        f' &nbsp;·&nbsp; Confidence: {flag.confidence:.0%}'
                        f' &nbsp;·&nbsp; Disposition: <b>{flag.recommended_disposition.value}</b></small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # Download buttons
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    report_dir = st.session_state.report_dir
    if report_dir:
        c1, c2 = st.columns(2)
        json_path = report_dir / "assurance_report.json"
        html_path = report_dir / "assurance_report.html"
        with c1:
            if json_path.exists():
                st.download_button(
                    "⬇ Download JSON Report",
                    data=json_path.read_bytes(),
                    file_name="assurance_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
        with c2:
            if html_path.exists():
                st.download_button(
                    "⬇ Download HTML Report",
                    data=html_path.read_bytes(),
                    file_name="assurance_report.html",
                    mime="text/html",
                    use_container_width=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Flags Explorer (sortable table + drill-down)
# ─────────────────────────────────────────────────────────────────────────────
with tab_flags:
    report = st.session_state.report

    if report is None:
        st.info("Run the pipeline first to see flags.")
        st.stop()

    # Collect all flags into a flat list
    all_flags = []
    for result in report.module_results:
        for flag in result.flags:
            all_flags.append(flag)

    if not all_flags:
        st.success("🎉 No flags were raised by any module. The dataset and model appear clean.")
        st.stop()

    # ── Filter controls ───────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filter_sev = st.multiselect(
            "Filter by severity",
            options=["high", "medium", "low"],
            default=["high", "medium", "low"],
        )
    with fc2:
        filter_mod = st.multiselect(
            "Filter by module",
            options=list(_MODULE_LABELS.values()),
            default=list(_MODULE_LABELS.values()),
        )
    with fc3:
        sort_by = st.selectbox(
            "Sort by",
            options=["Severity (worst first)", "Confidence (highest first)", "Module"],
            index=0,
        )

    # Apply filters
    selected_modules = {k for k, v in _MODULE_LABELS.items() if v in filter_mod}
    visible = [
        f for f in all_flags
        if f.severity.value in filter_sev
        and f.module.value in selected_modules
    ]

    # Apply sort
    if sort_by == "Severity (worst first)":
        visible.sort(key=lambda f: _SEV_ORDER[f.severity.value])
    elif sort_by == "Confidence (highest first)":
        visible.sort(key=lambda f: -f.confidence)
    else:
        visible.sort(key=lambda f: f.module.value)

    st.markdown(f"**{len(visible)}** flag(s) shown (of {len(all_flags)} total)")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Flags table ───────────────────────────────────────────────────────────
    import pandas as pd
    table_data = []
    for flag in visible:
        table_data.append({
            "Severity":   f"{_SEV_EMOJI[flag.severity.value]} {flag.severity.value.upper()}",
            "Module":     _MODULE_LABELS.get(flag.module.value, flag.module.value),
            "Reason":     flag.reason,
            "Asset":      flag.affected_asset,
            "Confidence": f"{flag.confidence:.0%}",
            "Disposition": flag.recommended_disposition.value,
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Reason": st.column_config.TextColumn(width="large"),
                "Asset":  st.column_config.TextColumn(width="medium"),
            },
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Drill-down per flag ───────────────────────────────────────────────────
    st.subheader("Flag Detail View")
    for i, flag in enumerate(visible):
        sev = flag.severity.value
        with st.expander(
            f"{_SEV_EMOJI[sev]} [{sev.upper()}] {flag.reason[:90]}{'…' if len(flag.reason) > 90 else ''}",
            expanded=False,
        ):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"**Flag ID:** `{flag.id}`")
                st.markdown(f"**Module:** {_MODULE_LABELS.get(flag.module.value, flag.module.value)}")
                st.markdown(f"**Severity:** {_SEV_EMOJI[sev]} `{sev.upper()}`")
                st.markdown(f"**Confidence:** {flag.confidence:.0%}")
            with d2:
                st.markdown(f"**Affected asset:**")
                st.code(flag.affected_asset, language=None)
                disp = flag.recommended_disposition.value
                disp_emoji = _REC_EMOJI.get(disp, "")
                st.markdown(
                    f"**Recommended disposition:** "
                    f'<span style="color:{_REC_COLOR.get(disp, "#fff")};font-weight:700">'
                    f'{disp_emoji} {disp.upper()}</span>',
                    unsafe_allow_html=True,
                )

            st.markdown("**Full reason:**")
            st.info(flag.reason)

            st.markdown("**Evidence:**")
            st.json(flag.evidence)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Audit Log Viewer
# ─────────────────────────────────────────────────────────────────────────────
with tab_audit:
    st.subheader("🔗 Provenance Chain Log Viewer")
    st.markdown(
        "Inspect and verify the integrity of the append-only SQLite chain log. "
        "Each record stores the SHA-256 hash of the previous record — any "
        "tampering causes `verify_chain()` to detect the broken link."
    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Path input (pre-fill from sidebar if set)
    audit_db_path = st.text_input(
        "Chain log DB path",
        value=db_path if db_path else "",
        placeholder="d:/data/provenance.db",
        key="audit_db_input",
    )

    verify_col, _ = st.columns([1, 3])
    with verify_col:
        verify_btn = st.button("🔍 Verify Chain Integrity", use_container_width=True)

    if verify_btn:
        if not audit_db_path:
            st.warning("Please enter a DB path.")
        elif not Path(audit_db_path).exists():
            st.error(f"File not found: `{audit_db_path}`")
        else:
            try:
                log = ChainLog(audit_db_path)
                chain_ok, broken_at = log.verify_chain()

                if chain_ok:
                    st.markdown(
                        '<div class="chain-ok">'
                        '<b>✅ Chain Intact</b> — All records verified. No tampering detected.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chain-err">'
                        f'<b>🚨 Chain Broken</b> — Integrity failure at sequence number '
                        f'<code>{broken_at}</code>. '
                        f'Record or its predecessor has been tampered with.'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Show all records in the chain
                st.subheader("Chain Records")
                import sqlite3
                with sqlite3.connect(audit_db_path) as con:
                    rows = con.execute(
                        "SELECT seq, prev_hash, payload, timestamp, record_hash "
                        "FROM chain_log ORDER BY seq"
                    ).fetchall()

                if not rows:
                    st.info("No records in this chain log yet.")
                else:
                    st.caption(f"{len(rows)} record(s) in chain")
                    for seq, prev_hash, payload, ts, rec_hash in rows:
                        # Compute expected hash to show inline verification
                        from anchor.provenance.chain_log import _hash_record
                        expected = _hash_record(prev_hash, payload, ts)
                        ok = (expected == rec_hash)
                        icon = "✅" if ok else "🚨"

                        with st.expander(
                            f"{icon} Seq #{seq} — {ts[:19]} UTC",
                            expanded=False,
                        ):
                            cols = st.columns(2)
                            with cols[0]:
                                st.markdown("**Payload:**")
                                try:
                                    st.json(json.loads(payload))
                                except Exception:
                                    st.code(payload)
                                st.markdown(f"**Timestamp:** `{ts}`")
                            with cols[1]:
                                st.markdown(
                                    f"**Record hash:** <span class=\"mono-hash\" title=\"{rec_hash}\">{rec_hash[:24]}…</span>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f"**Prev hash:** <span class=\"mono-hash\" title=\"{prev_hash}\">{prev_hash[:24]}…</span>",
                                    unsafe_allow_html=True,
                                )
                                if ok:
                                    st.success("Hash verified ✓")
                                else:
                                    st.error(
                                        f"Hash mismatch!\n"
                                        f"Expected: `{expected[:24]}…`\n"
                                        f"Stored:   `{rec_hash[:24]}…`"
                                    )

                log.close()

            except Exception as exc:
                st.exception(exc)
