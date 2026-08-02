"""Self-contained HTML report for PointCloudAnalysisAgent runs.

Mirrors the styling and structure of the experimental agents' reports
(metadata box, base64-embedded image grid, analysis text, caveats) so
point-cloud runs read like every other SciLink report.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path

_CSS = """
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f4f4f9; }
        .container { background-color: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { color: #2980b9; margin-top: 30px; }
        .metadata-box { background-color: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin-bottom: 20px; }
        .decision-box { background-color: #e8f4fc; padding: 15px; border-radius: 5px; border-left: 5px solid #2980b9; margin-bottom: 15px; white-space: pre-wrap; font-family: Menlo, Consolas, monospace; font-size: 0.85em; }
        .analysis-text { white-space: pre-wrap; background-color: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; margin-top: 15px; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px; margin-top: 20px; }
        .image-card { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .image-card img { max-width: 100%; height: auto; border-radius: 3px; }
        .image-label { margin-top: 12px; font-weight: bold; color: #444; font-size: 1em; border-top: 1px solid #eee; padding-top: 10px; }
        .params-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .params-table th, .params-table td { padding: 10px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        .params-table th { background-color: #f8f9fa; font-weight: 600; color: #2c3e50; }
        .quality-badge { display: inline-block; padding: 5px 12px; border-radius: 20px; font-weight: bold; margin-right: 10px; }
        .quality-good { background-color: #d4edda; color: #155724; }
        .quality-poor { background-color: #f8d7da; color: #721c24; }
        .footer { margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }
"""


def _img_card(path: Path, label: str) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<div class="image-card"><img src="data:image/png;base64,{b64}" '
            f'alt="{html.escape(label)}"><div class="image-label">'
            f'{html.escape(label)}</div></div>')


def build_html_report(workdir: str, objective: str, metadata: dict,
                      scout: dict, result: dict, gate: dict, answer: str,
                      decisions_text: str = "", images: dict | None = None,
                      cost: dict | None = None,
                      out_name: str = "report.html") -> str:
    """Assemble the self-contained report; returns the written path.

    ``images`` maps label -> filename (relative to workdir or absolute).
    """
    wd = Path(workdir)
    images_html = ""
    for label, fname in (images or {}).items():
        p = Path(fname) if Path(fname).is_absolute() else wd / fname
        if p.exists():
            images_html += _img_card(p, label)

    badge = ('<span class="quality-badge quality-good">GATE PASSED</span>'
             if gate.get("passed") else
             '<span class="quality-badge quality-poor">GATE FAILED</span>')
    gate_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in gate.items())
    cost_str = ""
    if cost:
        cost_str = (f" | Tokens {cost.get('tokens_in', '?')} in / "
                    f"{cost.get('tokens_out', '?')} out | "
                    f"${cost.get('cost_usd', '?')}")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Point Cloud Analysis Report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
<h1>&#9883; Point Cloud Analysis Report</h1>
<div class="metadata-box">
<p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>Agent:</strong> PointCloudAnalysisAgent (scout &rarr; plan &rarr; gate &rarr; interpret)</p>
<p><strong>Objective:</strong> {html.escape(objective)}</p>
<p><strong>Sample:</strong> {html.escape(str(metadata.get('sample', 'n/a')))}</p>
</div>

<h2>1. Verification {badge}</h2>
<table class="params-table"><tr><th>check</th><th>value</th></tr>{gate_rows}</table>

<h2>2. Planning decisions (from scout evidence)</h2>
<div class="decision-box">{html.escape(decisions_text or '(see commit script header in session/)')}</div>

<h2>3. Scout evidence</h2>
<div class="analysis-text">{html.escape(json.dumps(scout, indent=1)[:4000])}</div>

<h2>4. Images</h2>
<div class="image-grid">{images_html or '<p>(none)</p>'}</div>

<h2>5. Scientific interpretation</h2>
<div class="analysis-text">{html.escape(answer)}</div>

<h2>6. Execution record</h2>
<div class="analysis-text">{html.escape(json.dumps(result, indent=1)[:3000])}</div>

<div class="footer">Generated by SciLink PointCloudAnalysisAgent{cost_str}</div>
</div>
</body>
</html>"""
    out = wd / out_name
    out.write_text(doc)
    return str(out)
