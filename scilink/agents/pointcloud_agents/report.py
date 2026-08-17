"""Self-contained HTML report for PointCloudAnalysisAgent runs.

Follows the experimental agents' report conventions: metadata box, base64
image grid, an LLM-written natural-language interpretation, scientific-claim
cards with the "Has anyone ...?" literature hook, and a caveats box.
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
        .decision-box { background-color: #e8f4fc; padding: 15px; border-radius: 5px; border-left: 5px solid #2980b9; margin-bottom: 15px; white-space: pre-wrap; font-family: Menlo, Consolas, monospace; font-size: 0.8em; }
        .analysis-text { background-color: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; margin-top: 15px; }
        .claim-card { background-color: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0; }
        .claim-title { font-weight: bold; font-size: 1.1em; color: #0e6655; }
        .caveats { background-color: #fff8e6; border-left: 5px solid #f0ad4e; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }
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
        details { margin-top: 10px; }
        details summary { cursor: pointer; color: #2980b9; font-weight: 600; }
        details pre { background-color: #fafafa; border: 1px solid #eee; border-radius: 5px; padding: 12px; font-size: 0.8em; overflow-x: auto; }
        .footer { margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }
"""


def _img_card(path: Path, label: str) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<div class="image-card"><img src="data:image/png;base64,{b64}" '
            f'alt="{html.escape(label)}"><div class="image-label">'
            f'{html.escape(label)}</div></div>')


def _paragraphs(text: str) -> str:
    return "".join(f"<p>{html.escape(p.strip())}</p>"
                   for p in str(text).split("\n\n") if p.strip())


def build_html_report(workdir: str, objective: str, metadata: dict,
                      scout: dict, result: dict, gate: dict,
                      interpretation, decisions_text: str = "",
                      images: dict | None = None, cost: dict | None = None,
                      interactive: dict | None = None,
                      out_name: str = "report.html") -> str:
    """Assemble the report; ``interpretation`` is the structured dict from
    the interpret phase (a bare string is wrapped as detailed_analysis)."""
    wd = Path(workdir)
    if isinstance(interpretation, str):
        interpretation = {"detailed_analysis": interpretation,
                          "scientific_claims": [], "caveats": ""}

    # first image = HERO, rendered full-width; the rest as the grid
    hero_html, images_html = "", ""
    for n, (label, f) in enumerate((images or {}).items()):
        pth = Path(f) if Path(f).is_absolute() else wd / f
        if not pth.exists():
            continue
        if n == 0:
            b64 = base64.b64encode(pth.read_bytes()).decode()
            hero_html = (
                f'<div class="image-card" style="grid-column: 1 / -1;">'
                f'<img src="data:image/png;base64,{b64}" '
                f'alt="{html.escape(label)}" style="max-width:100%;">'
                f'<div class="image-label">{html.escape(label)}</div></div>')
        else:
            images_html += _img_card(pth, label)

    links_html = ""
    for label, f in (interactive or {}).items():
        pth = Path(f) if Path(f).is_absolute() else wd / f
        if pth.exists():
            links_html += (f'<p>&#127919; <a href="{html.escape(pth.name)}" '
                           f'target="_blank">Open 3D visualization: '
                           f'{html.escape(label)}</a></p>')

    badge = ('<span class="quality-badge quality-good">GATE PASSED</span>'
             if gate.get("passed") else
             '<span class="quality-badge quality-poor">GATE FAILED</span>')
    gate_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in gate.items())

    claims_html = ""
    for i, c in enumerate(interpretation.get("scientific_claims", []), 1):
        kw = ", ".join(c.get("keywords", [])) or "N/A"
        claims_html += f"""
<div class="claim-card">
  <div class="claim-title">Claim {i}: {html.escape(c.get('claim', 'N/A'))}</div>
  <p><strong>Scientific Impact:</strong> {html.escape(c.get('scientific_impact', 'N/A'))}</p>
  <p><strong>Literature Search Query:</strong> <em>{html.escape(c.get('has_anyone_question', 'N/A'))}</em></p>
  <p><strong>Keywords:</strong> {html.escape(kw)}</p>
</div>"""
    if not claims_html:
        claims_html = "<p>No specific claims generated.</p>"

    caveats_html = ""
    if interpretation.get("caveats"):
        caveats_html = (f'<h2>6. Caveats &amp; Limitations</h2>'
                        f'<div class="caveats">'
                        f'{_paragraphs(interpretation["caveats"])}</div>')

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
<p><strong>Sample:</strong> {html.escape(str(metadata.get('sample', 'n/a')))}</p>
<p><strong>Objective:</strong> {html.escape(objective)}</p>
</div>

<h2>1. Verification {badge}</h2>
<table class="params-table"><tr><th>check</th><th>value</th></tr>{gate_rows}</table>

<h2>2. Detailed Analysis</h2>
<div class="analysis-text">{_paragraphs(interpretation.get('detailed_analysis', ''))}</div>

<h2>3. Scientific Claims</h2>
{claims_html}

<h2>4. Images</h2>
<div class="image-grid">{hero_html}{images_html or ('' if hero_html else '<p>(none)</p>')}</div>
{links_html}

<h2>5. Planning Decisions &amp; Evidence</h2>
<div class="decision-box">{html.escape(decisions_text or '(see commit script in session/)')}</div>
<details><summary>Scout evidence (raw)</summary>
<pre>{html.escape(json.dumps(scout, indent=1)[:5000])}</pre></details>
<details><summary>Execution record (raw)</summary>
<pre>{html.escape(json.dumps(result, indent=1)[:4000])}</pre></details>
{caveats_html}

<div class="footer">Generated by SciLink PointCloudAnalysisAgent{cost_str}</div>
</div>
</body>
</html>"""
    out = wd / out_name
    out.write_text(doc)
    return str(out)
