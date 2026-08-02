"""PointCloudAnalysisAgent (MVP).

A minimal modality agent for 3D point clouds (simulated structures now; APT
reconstructions later). Loop: load -> SCOUT (cheap CPU evidence: shape, zone
axes, feature localization) -> PLAN+COMMIT (LLM decides orientation and ROI
from the scout evidence, then renders/analyzes via the stem_simulation
TOOL_SPEC functions) -> GATE (deterministic execution checks incl. an
image-vs-structure column-count cross-check) -> INTERPRET (LLM answers the
scientific objective from the verified results).

The ROI decision is the central planning gate: apex-inclusive for finite
tips, feature-centered when the objective targets a defect/boundary, no crop
when the data and memory budget allow it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_BUDGET_NOTES = """\
GPU memory budget (24 GB class card, empirical): a lateral field up to
~9500 A^2 simulates at full quality (0.04 A sampling, PRISM interpolation 4);
up to ~11000 A^2 completes with degraded settings via the automatic OOM
ladder (0.05 A, interpolation 6, reduced detector ceiling); larger fields
exhaust the ladder and FAIL - crop, or state that tiling would be required.
On a 16 GB card, subtract ~30%. Prefer full-quality fields for quantitative
objectives; degraded rungs are acceptable for morphology-only objectives.
"""

_PHYSICS_NOTES = """\
Zone-axis identification from the projected net (FCC, 3D NN distance = a/sqrt2):
  <110> view: NN_proj ~ 0.61a (centered rectangular, 4+2 shell)
  <111> view: NN_proj ~ 0.41a (hexagonal)
  <100> view: NN_proj ~ 0.71a (square)
A planar boundary is only VISIBLE (edge-on) when the beam is PERPENDICULAR
to its normal; a projection along the normal renders it face-on/invisible.
Defect interpretation from per-column 2D-PTM in an FCC matrix:
  1 HCP layer between mirrored FCC domains = coherent twin boundary (sigma-3);
  2 adjacent HCP layers = intrinsic stacking fault; 2 HCP layers sandwiching
  1 FCC layer = extrinsic fault. Isolated HCP segments/patches or elevated
  unidentified fractions away from the main feature = additional defects.
LAMMPS files with anonymous types need type_map (e.g. {1: 'Fe'}) - take the
species from the metadata.
"""


def _render_specs():
    from scilink.skills.stem_simulation.haadf_workflow import abtem_tools
    from scilink.skills.stem_simulation.haadf_workflow import scout_tools
    specs = scout_tools.TOOL_SPECS + abtem_tools.TOOL_SPECS
    try:
        from scilink.skills.point_cloud_analysis.structure_id_3d import (
            ptm3d_tools)
        specs = specs + ptm3d_tools.TOOL_SPECS
    except ImportError:
        pass
    try:
        from scilink.skills.point_cloud_analysis.apt_ccd import apt_tools
        specs = specs + apt_tools.TOOL_SPECS
    except ImportError:
        pass
    out = []
    for spec in specs:
        out.append(f"### {spec.name}\n{spec.description}\n"
                   f"import: {spec.import_line}\n"
                   f"signature: {spec.signature}\n"
                   f"when: {spec.when_to_use}\nreturns: {spec.returns}\n"
                   f"example:\n{spec.example}\n")
    return "\n".join(out)


class PointCloudAnalysisAgent:
    """Minimal point-cloud modality agent driving the stem_simulation skills."""

    def __init__(self, model_name: str, workdir: str,
                 python_exe: str = sys.executable,
                 scout_timeout: int = 900, commit_timeout: int = 3600):
        self.model = model_name
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.py = python_exe
        self.scout_timeout = scout_timeout
        self.commit_timeout = commit_timeout
        self.transcript = []

    # ---------------- LLM plumbing ----------------
    def _llm(self, tag, messages):
        import litellm
        t0 = time.time()
        resp = litellm.completion(model=self.model, messages=messages,
                                  max_tokens=4000)
        text = resp.choices[0].message.content
        self.transcript.append({"phase": tag, "elapsed_s": round(time.time() - t0, 1),
                                "prompt": messages, "response": text})
        (self.workdir / "transcript.json").write_text(
            json.dumps(self.transcript, indent=1, default=str))
        return text

    @staticmethod
    def _extract_code(text):
        m = re.findall(r"```python\n(.*?)```", text, re.S)
        if not m:
            raise ValueError("LLM response contained no python code block")
        return m[-1]

    def _run_script(self, name, code, timeout):
        path = self.workdir / name
        path.write_text(code)
        proc = subprocess.run([self.py, str(path)], cwd=self.workdir,
                              capture_output=True, text=True, timeout=timeout)
        (self.workdir / f"{name}.out").write_text(
            proc.stdout[-20000:] + "\n--- STDERR ---\n" + proc.stderr[-8000:])
        return proc

    def _run_phase(self, tag, prompt, marker, timeout):
        """LLM -> script -> execute; one regenerate on failure. Returns the
        JSON payload printed on the marker line."""
        messages = [{"role": "user", "content": prompt}]
        for attempt in (1, 2):
            code = self._extract_code(self._llm(f"{tag}_{attempt}", messages))
            proc = self._run_script(f"{tag}_{attempt}.py", code, timeout)
            payload = None
            for line in reversed(proc.stdout.splitlines()):
                if line.startswith(marker):
                    payload = json.loads(line[len(marker):])
                    break
            if proc.returncode == 0 and payload is not None:
                return payload
            messages += [
                {"role": "assistant", "content": code},
                {"role": "user", "content":
                    f"The script failed (exit {proc.returncode}) or did not "
                    f"print the required '{marker}' line.\nSTDOUT tail:\n"
                    f"{proc.stdout[-2500:]}\nSTDERR tail:\n{proc.stderr[-2500:]}\n"
                    "Fix the problem and return the complete corrected script."}]
        raise RuntimeError(f"phase {tag} failed after retry")

    def _parse_interpretation(self, raw):
        def _attempt(text):
            m = re.findall(r"```json\n(.*?)```", text, re.S)
            obj = json.loads(m[-1] if m else text)
            da = obj.get("detailed_analysis", "")
            if isinstance(da, (list, tuple)):
                obj["detailed_analysis"] = "\n\n".join(str(x) for x in da)
            return obj
        try:
            return _attempt(raw)
        except Exception:
            pass
        try:  # one-shot LLM repair of malformed JSON
            fixed = self._llm("interpret_jsonfix", [{"role": "user", "content":
                "Convert the following into VALID json inside a single "
                "```json fenced block, with detailed_analysis as one single "
                "string, scientific_claims as a list of objects, caveats as "
                "a string. Preserve all content verbatim.\n\n" + raw}])
            return _attempt(fixed)
        except Exception:
            return {"detailed_analysis": raw, "scientific_claims": [],
                    "caveats": ""}

    # ---------------- gate ----------------
    def _gate(self, result):
        checks = {}
        files = result.get("files") or {}
        if "npy" not in files:
            # simulation was (legitimately) skipped - verify whatever
            # artifacts the plan promised actually exist
            checks["simulation_skipped"] = True
            for k, v in files.items():
                checks[f"file_{k}_exists"] = bool((self.workdir / str(v)).exists())
            checks["passed"] = all(v is True for k, v in checks.items()
                                   if isinstance(v, bool))
            (self.workdir / "gate.json").write_text(json.dumps(checks, indent=1))
            return checks
        try:
            npy = self.workdir / result["files"]["npy"]
            meta = json.loads((self.workdir / result["files"]["meta"]).read_text())
            img = np.load(npy)
            checks["artifacts_exist"] = True
            checks["image_finite_nonuniform"] = bool(
                np.isfinite(img).all() and img.std() > 0)
            det = meta.get("detector", {})
            checks["annulus_within_antialias"] = bool(
                det.get("outer_mrad_effective", 1e9)
                <= det.get("antialias_limit_mrad", 0))
            # image-vs-structure column-count cross-check
            from scipy.ndimage import gaussian_filter, maximum_filter
            im = gaussian_filter(img.T.astype(float),
                                 0.35 / meta["scan_sampling_A"])
            lo, hi = np.percentile(im, [1, 99.5])
            imn = (im - lo) / (hi - lo)
            size = max(3, int(round(1.4 / meta["scan_sampling_A"])))
            peaks = (imn == maximum_filter(imn, size=size)) & (imn > 0.12)
            n_img = int(peaks.sum())
            n_struct = int(result.get("n_columns_structure", 0))
            checks["n_columns_image"] = n_img
            checks["n_columns_structure"] = n_struct
            ratio = n_img / max(n_struct, 1)
            checks["column_count_ratio"] = round(ratio, 2)
            checks["column_count_consistent"] = bool(0.4 <= ratio <= 2.5)
        except Exception as exc:  # noqa: BLE001 - gate must always report
            checks["gate_error"] = f"{type(exc).__name__}: {exc}"
        checks["passed"] = all(v is True for k, v in checks.items()
                               if isinstance(v, bool))
        (self.workdir / "gate.json").write_text(json.dumps(checks, indent=1))
        return checks

    # ---------------- main ----------------
    def analyze(self, structure_path: str, metadata_path: str,
                objective: str) -> dict:
        metadata = json.loads(Path(metadata_path).read_text())
        specs = _render_specs()
        common = (f"You are PointCloudAnalysisAgent, working on the atomistic "
                  f"point cloud at {structure_path}.\n\nOBJECTIVE:\n{objective}\n\n"
                  f"EXPERIMENT METADATA:\n{json.dumps(metadata, indent=1)}\n\n"
                  f"AVAILABLE TOOLS (import and call exactly as documented):\n"
                  f"{specs}\nPHYSICS NOTES:\n{_PHYSICS_NOTES}\n"
                  f"BUDGET NOTES:\n{_BUDGET_NOTES}\n")

        scout = self._run_phase(
            "scout",
            common + (
                "PHASE 1 - SCOUT. Write ONE python script that gathers, on CPU "
                "only (no simulation), the evidence needed to plan this task: "
                "load the structure (read_structure), profile its shape, "
                "evaluate candidate beam axes (net_quality on x, y, z), and "
                "localize the feature the objective asks about "
                "(feature_profile along suitable axes). "
                "Print progress freely, and end by printing one line: "
                "SCOUT_JSON: {json with your collected evidence}"),
            "SCOUT_JSON:", self.scout_timeout)

        result = self._run_phase(
            "commit",
            common + (
                f"PHASE 2 - PLAN AND EXECUTE.\nScout evidence:\n"
                f"{json.dumps(scout, indent=1)}\n\n"
                "First, in comments at the top of your script, state your "
                "DECISIONS with justification from the evidence: "
                "WHETHER TO SIMULATE at all - render the image only if one "
                "of these triggers applies and name it: (1) the objective "
                "demands the image, (2) comparison against an experimental "
                "image, (3) generating training data, (4) testing defect "
                "visibility under the imaging conditions, (5) the "
                "image-vs-structure verification cross-check is wanted. If "
                "none applies, SKIP the simulation and answer from the 3D "
                "structure tools alone (seconds instead of GPU-minutes); "
                "then beam "
                "orientation (which axis gives the zone axis the objective "
                "asks for, and keeps the feature edge-on), ROI (feature-"
                "centered window vs apex-inclusive vs no crop - respect the "
                "memory budget), and expected quality rung. Then write ONE "
                "python script that executes: prepare the slab, simulate the "
                "HAADF image under the metadata conditions, and run "
                "structure_defect_map on the same slab. End by printing one "
                "line:\n"
                "RESULT_JSON: {\"decisions\": {...}, \"files\": {label: filename "
                "for every artifact you produced - include npy/png/meta keys "
                "when you simulated an image}, plus whatever quantitative "
                "result fields your analyses yielded (e.g. "
                "n_columns_structure, ptm_fractions, community compositions)}"),
            "RESULT_JSON:", self.commit_timeout)

        gate = self._gate(result)

        decisions = "\n".join(
            l for l in (self.workdir / "commit_1.py").read_text().splitlines()
            if l.startswith("#"))[:6000] if (self.workdir / "commit_1.py").exists() else ""

        raw = self._llm("interpret", [{"role": "user", "content":
            common + (
                f"PHASE 3 - INTERPRET.\nScout evidence:\n"
                f"{json.dumps(scout, indent=1)}\n\nExecution results:\n"
                f"{json.dumps(result, indent=1)}\n\nVerification gate:\n"
                f"{json.dumps(gate, indent=1)}\n\n"
                "Produce the scientific interpretation as a single fenced "
                "```json block with exactly these fields:\n"
                '{"detailed_analysis": "ONE SINGLE STRING (not a list) of 3-6 plain-prose paragraphs (no '
                "markdown syntax) interpreting the data and analysis: what "
                "was measured/computed, what the numbers show, and the "
                "direct quantitative answer to the objective\", "
                '"scientific_claims": [2-4 items, each '
                '{"claim": one-sentence finding, '
                '"scientific_impact": why it matters, '
                '"has_anyone_question": a literature-search question phrased '
                "'Has anyone ...?', "
                '"keywords": [3-6 terms]}], '
                '"caveats": "short plain-prose statement of limitations '
                '(gate results, quality rung, single-snapshot thermal '
                'statistics, foil thickness)"}')}])
        interpretation = self._parse_interpretation(raw)
        md = [interpretation.get("detailed_analysis", "")]
        for i, c in enumerate(interpretation.get("scientific_claims", []), 1):
            md.append(f"\n**Claim {i}:** {c.get('claim', '')}\n"
                      f"- Impact: {c.get('scientific_impact', '')}\n"
                      f"- Literature: {c.get('has_anyone_question', '')}\n"
                      f"- Keywords: {', '.join(c.get('keywords', []))}")
        if interpretation.get("caveats"):
            md.append(f"\n**Caveats:** {interpretation['caveats']}")
        answer = "\n".join(md)
        (self.workdir / "final_answer.md").write_text(answer)
        (self.workdir / "interpretation.json").write_text(
            json.dumps(interpretation, indent=1))
        from .report import build_html_report
        images = {}
        files = result.get("files") or {}
        if files.get("png"):
            images["Simulated HAADF-STEM"] = files["png"]
        # sweep every png any phase produced anywhere under the workdir
        for extra in sorted(self.workdir.rglob("*.png"))[:8]:
            label = extra.stem.replace("_", " ")
            if str(extra.name) != str(files.get("png", "")):
                images.setdefault(label, str(extra))
        report = build_html_report(
            str(self.workdir), objective, metadata, scout, result, gate,
            interpretation, decisions_text=decisions, images=images)
        return {"status": "success" if gate.get("passed") else "gate_failed",
                "scout": scout, "result": result, "gate": gate,
                "answer": answer, "interpretation": interpretation,
                "report_html": report}
