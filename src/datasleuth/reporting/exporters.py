"""Canonical JSON and readable HTML audit exporters."""

from __future__ import annotations

import json
from html import escape

from datasleuth.engine import AuditResult


def export_json(result: AuditResult, *, indent: int = 2) -> str:
    """Serialize an audit result as canonical JSON."""
    return json.dumps(result.to_dict(), indent=indent, sort_keys=True)


def export_html(result: AuditResult) -> str:
    """Render a self-contained HTML report from an audit result."""
    report = result.to_dict()
    summary = report["summary"]
    finding_rows = []
    for finding in report["findings"]:
        finding_rows.append(
            "<tr>"
            f"<td>{escape(finding['severity'])}</td>"
            f"<td>{escape(finding['category'])}</td>"
            f"<td>{escape(finding['check_id'])}</td>"
            f"<td>{escape(finding['column'] or '')}</td>"
            f"<td>{escape(finding['title'])}</td>"
            f"<td>{escape(finding['description'])}</td>"
            "</tr>"
        )
    if not finding_rows:
        finding_rows.append(
            '<tr><td colspan="6">No findings were reported.</td></tr>'
        )
    check_rows = []
    for check in report["checks"]:
        error = escape(check["error"] or "")
        check_rows.append(
            "<tr>"
            f"<td>{escape(check['check_id'])}</td>"
            f"<td>{escape(check['category'])}</td>"
            f"<td>{escape(check['status'])}</td>"
            f"<td>{error}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>DataSleuth Audit Report</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
th, td {{ border: 1px solid #ccc; padding: .5rem; text-align: left; }}
th {{ background: #f2f2f2; }}
</style></head>
<body>
<h1>DataSleuth Audit Report</h1>
<p>Findings: {summary['finding_count']}</p>
<p>Check statuses: {escape(json.dumps(summary['check_counts'], sort_keys=True))}</p>
<h2>Checks</h2>
<table><thead><tr><th>Check</th><th>Category</th><th>Status</th><th>Error</th>
</tr></thead><tbody>{"".join(check_rows)}</tbody></table>
<h2>Findings</h2>
<table><thead><tr><th>Severity</th><th>Category</th><th>Check</th><th>Column</th>
<th>Title</th><th>Description</th></tr></thead>
<tbody>{"".join(finding_rows)}</tbody></table>
</body></html>"""
