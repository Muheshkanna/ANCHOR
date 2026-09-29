"""Assurance report builder.

Aggregates module results into an AssuranceReport schema, computes the overall
recommendation based on flag severities, and exports the report to both structured
JSON and a beautiful Jinja2-rendered HTML report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from jinja2 import Template

from anchor.report_schema.schema import (
    AssuranceReport,
    Disposition,
    ModuleResult,
    Severity,
)

# Sleek responsive HTML template with CSS for rendering report results
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anchor AI Integrity Assurance Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            
            --color-accept: #10b981;
            --color-review: #f59e0b;
            --color-quarantine: #ef4444;
            --color-low: #6b7280;
            
            --accept-bg: rgba(16, 185, 129, 0.15);
            --review-bg: rgba(245, 158, 11, 0.15);
            --quarantine-bg: rgba(239, 68, 68, 0.15);
            --low-bg: rgba(107, 114, 128, 0.15);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem 1rem;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.08) 0%, transparent 40%);
            background-attachment: fixed;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
        }

        /* Header block */
        header {
            text-align: center;
            margin-bottom: 3rem;
            padding: 2.5rem;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            position: relative;
            overflow: hidden;
        }

        header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #6366f1, #ec4899);
        }

        h1 {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 2.2rem;
            letter-spacing: -0.03em;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #ffffff 60%, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }

        /* Recommendation Badge */
        .rec-container {
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.8rem 1.5rem;
            border-radius: 12px;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1.1rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }

        .rec-accept {
            background-color: var(--accept-bg);
            color: var(--color-accept);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .rec-review {
            background-color: var(--review-bg);
            color: var(--color-review);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .rec-quarantine {
            background-color: var(--quarantine-bg);
            color: var(--color-quarantine);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Metadata Grid */
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .meta-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 1.2rem;
            border-radius: 14px;
            backdrop-filter: blur(5px);
        }

        .meta-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.3rem;
        }

        .meta-val {
            font-weight: 600;
            font-size: 0.95rem;
            word-break: break-all;
        }

        /* Module Sections */
        h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .module-section {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }

        .module-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .module-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.3rem;
            font-weight: 600;
            text-transform: capitalize;
        }

        .module-badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.3rem 0.7rem;
            border-radius: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
        }

        .coverage-box {
            background: rgba(255, 255, 255, 0.02);
            border-left: 3px solid #6366f1;
            padding: 0.8rem 1rem;
            font-size: 0.85rem;
            color: var(--text-muted);
            border-radius: 0 8px 8px 0;
            margin-bottom: 1.5rem;
        }

        /* Flag Table */
        .flag-table-container {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.85rem;
        }

        th {
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            padding: 0.8rem 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        td {
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
            vertical-align: middle;
        }

        tr:last-child td {
            border-bottom: none;
        }

        .severity-badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .severity-high {
            background-color: var(--quarantine-bg);
            color: var(--color-quarantine);
        }

        .severity-medium {
            background-color: var(--review-bg);
            color: var(--color-review);
        }

        .severity-low {
            background-color: var(--low-bg);
            color: var(--text-muted);
        }

        .no-flags {
            text-align: center;
            padding: 2rem;
            color: var(--color-accept);
            font-size: 0.9rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.03);
            border-radius: 8px;
            border: 1px dashed rgba(16, 185, 129, 0.2);
        }

        .evidence-code {
            font-family: monospace;
            background: rgba(0, 0, 0, 0.3);
            padding: 0.3rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            color: #f43f5e;
            white-space: pre-wrap;
            max-width: 300px;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Main header -->
        <header>
            <h1>Anchor Assurance Report</h1>
            <div class="subtitle">AI System Integrity Check Pipeline Summary</div>
            
            <div class="rec-container rec-{{ overall_recommendation.value }}">
                Recommendation: {{ overall_recommendation.value }}
            </div>
        </header>

        <!-- Metadata Section -->
        <div class="meta-grid">
            <div class="meta-card">
                <div class="meta-label">Model ID</div>
                <div class="meta-val">{{ model_id }}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Dataset ID</div>
                <div class="meta-val">{{ dataset_id }}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Timestamp</div>
                <div class="meta-val">{{ timestamp }}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Audit Log Ref</div>
                <div class="meta-val">{{ audit_log_ref }}</div>
            </div>
        </div>

        <h2>Module Evaluation Details</h2>
        
        <!-- Loop over results -->
        {% for result in module_results %}
        <div class="module-section">
            <div class="module-header">
                <div class="module-title">{{ result.module.value.replace('_', ' ') }}</div>
                <div class="module-badge">Access Mode: {{ result.access_mode_used.value }}</div>
            </div>
            
            <div class="coverage-box">
                <strong>Coverage Scope Boundary:</strong> {{ result.coverage_statement }}
            </div>

            {% if result.flags %}
            <div class="flag-table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Severity</th>
                            <th>Concern Reason</th>
                            <th>Affected Asset</th>
                            <th>Confidence</th>
                            <th>Evidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for flag in result.flags %}
                        <tr>
                            <td>
                                <span class="severity-badge severity-{{ flag.severity.value }}">
                                    {{ flag.severity.value }}
                                </span>
                            </td>
                            <td>{{ flag.reason }}</td>
                            <td><code>{{ flag.affected_asset }}</code></td>
                            <td>{{ "%.2f"|format(flag.confidence) }}</td>
                            <td>
                                <span class="evidence-code">{{ flag.evidence | tojson(indent=2) }}</span>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% else %}
            <div class="no-flags">
                ✓ No concerns flagged. All checks in this scope passed.
            </div>
            {% endif %}
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""


def build_report(
    dataset_id: str,
    model_id: str,
    module_results: list[ModuleResult],
    audit_log_ref: str,
    output_dir: str | Path,
    report_name: str = "assurance_report",
) -> AssuranceReport:
    """Build the final AssuranceReport model and output both JSON and HTML representations.

    Parameters
    ----------
    dataset_id :
        Identifier of the evaluated dataset.
    model_id :
        Identifier of the evaluated model.
    module_results :
        Aggregated evaluation module results.
    audit_log_ref :
        Reference to the immutable audit log entry.
    output_dir :
        Directory where JSON and HTML report files will be written.
    report_name :
        Base file name (without extension) for the outputs.

    Returns
    -------
    AssuranceReport
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine overall recommendation by aggregating flag severities
    # Any HIGH -> Quarantine
    # Any MEDIUM -> Review
    # Otherwise -> Accept
    all_flags = []
    for r in module_results:
        all_flags.extend(r.flags)

    overall_rec = Disposition.ACCEPT
    for flag in all_flags:
        if flag.severity == Severity.HIGH:
            overall_rec = Disposition.QUARANTINE
            break
        elif flag.severity == Severity.MEDIUM:
            overall_rec = Disposition.REVIEW

    # 2. Build the AssuranceReport pydantic model
    report = AssuranceReport(
        dataset_id=dataset_id,
        model_id=model_id,
        timestamp=datetime.now(timezone.utc),
        module_results=module_results,
        overall_recommendation=overall_rec,
        audit_log_ref=audit_log_ref,
    )

    # 3. Write to JSON
    # Pydantic v2 uses model_dump_json()
    json_path = output_dir / f"{report_name}.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    # 4. Render HTML using Jinja2
    template = Template(_HTML_TEMPLATE)
    
    # Prepare template data
    html_content = template.render(
        dataset_id=report.dataset_id,
        model_id=report.model_id,
        timestamp=report.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
        module_results=report.module_results,
        overall_recommendation=report.overall_recommendation,
        audit_log_ref=report.audit_log_ref,
    )
    
    html_path = output_dir / f"{report_name}.html"
    html_path.write_text(html_content, encoding="utf-8")

    return report
