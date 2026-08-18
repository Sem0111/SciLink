"""PointCloudAnalysisAgent - the 3D point-cloud modality agent.

Analyzes atomistic point clouds: simulated structures (xyz/extxyz, LAMMPS
data, CIF) and APT reconstructions (pos/epos/apt, ranged xyz/csv). Loop:

  SCOUT       cheap CPU evidence: cloud kind, shape, zone axes, features
  PLAN+COMMIT LLM decides ROI / orientation / whether to forward-simulate,
              then executes via the point_cloud_analysis + stem_simulation
              skill tools (codegen, sandboxed subprocess)
  GATE        deterministic physics checks (image-vs-structure column count,
              artifact existence) - no LLM scoring; this modality has ground
              truth and uses it
  INTERPRET   structured scientific interpretation (detailed_analysis,
              scientific_claims with literature hooks, caveats) + HTML report

Orchestrator-compatible per the BaseAnalysisAgent contract: constructor
accepts api_key/model_name/base_url/output_dir/enable_human_feedback;
analyze() takes (data, system_info=None, objective=None, hints=None, ...)
and returns status / detailed_analysis / scientific_claims /
output_directory.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ...executors import ScriptExecutor, require_sandbox_approval
from ...skills._shared._registry import format_tool_inventory
from ..exp_agents.base_agent import BaseAnalysisAgent
from .report import build_html_report

# Skill bundles whose TOOL_SPECS this agent serves (registry-gated by name).
DEFAULT_ACTIVE_SKILLS = ["haadf_workflow", "generate_abtem_input",
                         "structure_id_3d", "apt_ccd", "cluster_analysis",
                         "precipitate_analysis"]

_BUDGET_NOTES = """\
GPU memory budget (24 GB class card, empirical): a lateral field up to
~9500 A^2 simulates at full quality (0.04 A sampling, PRISM interpolation 4);
up to ~11000 A^2 completes with degraded settings via the automatic OOM
ladder; larger fields exhaust the ladder and FAIL - crop, or state that
tiling would be required. On a 16 GB card, subtract ~30%. On CPU (no cupy)
simulation runs but is minutes-to-hours slower - prefer skipping simulation
unless the objective demands the image.
"""

_PHYSICS_NOTES = """\
Cloud kinds (use classify_cloud_kind evidence): a simulated/ideal lattice
supports lattice-resolved tools (PTM, HAADF simulation); an APT
reconstruction (detection efficiency ~40-80%, trajectory aberrations)
supports statistical tools (CCD, composition, Warren-Cowley) - run
lattice-resolved analysis on APT data only where the scout shows resolvable
order. Numeric species = unranged APT: ranging (.rrng) is required first.

Zone-axis identification from the projected net (FCC, 3D NN distance = a/sqrt2):
  <110> view: NN_proj ~ 0.61a; <111>: ~0.41a (hexagonal); <100>: ~0.71a (square).
A planar boundary is only VISIBLE (edge-on) when the beam is PERPENDICULAR
to its normal. Defect reading from per-column 2D-PTM in an FCC matrix:
1 HCP layer between mirrored FCC = coherent twin (sigma-3); 2 adjacent HCP
layers = intrinsic stacking fault; HCP-FCC-HCP = extrinsic fault.
LAMMPS files with anonymous types need type_map from the metadata species.

EXECUTION ENVIRONMENT: numpy >= 2 (arr.ptp() was REMOVED - use
np.ptp(arr); always `import numpy as np` explicitly). Your script may be
KILLED at its timeout and stdout is then lost - for any step that could run
minutes, append progress lines to a file `progress.log` (open with
buffering=1) so a timeout is diagnosable, and prefer spatial
subsets/cropping for expensive analyses (DXA or full-cloud graph analyses
on >1M atoms can exceed the budget; a representative subset with the
subsetting stated is better than a timeout).

PRISM WINDOW RULE: extent/interpolation must exceed ~20-25 A (probe with
tails); on small fields lower the interpolation or use multislice - window
artifacts produce phantom lattices that FFT checks miss.
MULTISLICE RUNTIME: cost ~ probe positions x slices; on a 16 GB T4 budget
roughly 10-30k scan positions within an hour - coarsen scan step (0.25-0.3
A is fine for lattice metrology) and keep the field modest; write progress
to progress.log so long scans are diagnosable.

SIMULATION ALGORITHM: simulate_haadf defaults to TRUE MULTISLICE (exact);
PRISM is opt-in for large fields with the smoothing tradeoff stated. An
algorithm named in the objective is BINDING - never silently substitute;
fit the requested algorithm to budget by shrinking the field or coarsening
the scan step instead.

WHETHER TO SIMULATE an image - only if a trigger applies and you name it:
(1) the objective demands the image, (2) comparison against an experimental
image, (3) generating training data, (4) testing defect visibility under
the imaging conditions, (5) the image-vs-structure verification cross-check
is wanted. Otherwise SKIP simulation and answer from structure-side tools.
"""

_INTERPRET_SCHEMA = (
    'Produce the scientific interpretation as a single fenced ```json block '
    'with exactly these fields:\n'
    '{"detailed_analysis": "ONE SINGLE STRING (not a list) of 3-6 plain-prose '
    'paragraphs (separated by \\n\\n, no markdown) interpreting the data and '
    'analysis and answering the objective quantitatively", '
    '"scientific_claims": [2-4 items, each {"claim": one-sentence finding, '
    '"scientific_impact": why it matters, "has_anyone_question": a literature '
    "question phrased 'Has anyone ...?', \"keywords\": [3-6 terms]}], "
    '"caveats": "short plain-prose statement of limitations"}')


class PointCloudAnalysisAgent(BaseAnalysisAgent):
    """Agent for 3D atomistic point clouds (simulated and APT)."""

    AGENT_NAME = "PointCloudAnalysis"
    AGENT_DESCRIPTION = (
        "3D atomistic point clouds: simulated structures (xyz/extxyz, LAMMPS "
        "data, CIF) and APT reconstructions (pos/epos/apt, ranged xyz/csv). "
        "Scouts cloud kind/shape/lattice/features, decides region of "
        "interest and whether to forward-simulate a HAADF image, then runs "
        "structural classification (3D PTM, DXA), compositional segregation "
        "(CCD), or synthesized analyses (e.g. Warren-Cowley), with "
        "deterministic physics verification.")
    AGENT_SHORT_NAME = "pointcloud"

    def __init__(self,
                 api_key: str | None = None,
                 model_name: str = "claude-opus-4-6",
                 base_url: str | None = None,
                 output_dir: str = "pointcloud_analysis_output",
                 enable_human_feedback: bool = False,
                 executor_timeout: int = 900,
                 commit_timeout: int = 3600,
                 **kwargs):
        if not require_sandbox_approval(
                context="PointCloudAnalysisAgent executes generated analysis scripts"):
            raise RuntimeError("Sandbox approval declined - aborting")
        super().__init__(api_key=api_key, model_name=model_name,
                         base_url=base_url, output_dir=output_dir,
                         enable_human_feedback=enable_human_feedback, **kwargs)
        self.agent_type = "pointcloud_analysis"
        self.executor = ScriptExecutor(timeout=executor_timeout)
        self.commit_timeout = commit_timeout
        self.transcript: list = []

    # ------------------------------------------------------------------
    # required-by-convention base hooks
    # ------------------------------------------------------------------
    def _get_claims_instruction_prompt(self) -> str:
        return _INTERPRET_SCHEMA

    def _get_measurement_recommendations_prompt(self) -> str:
        return ("Based on the point-cloud analysis results, recommend "
                "follow-up measurements (imaging conditions, APT run "
                "parameters, or simulations) as structured JSON.")

    # ------------------------------------------------------------------
    # LLM + execution plumbing
    # ------------------------------------------------------------------
    def _llm(self, tag: str, prompt: str, workdir: Path) -> str:
        t0 = time.time()
        response = self.model.generate_content(prompt)
        # raw_text, not .text: the wrapper's .text applies a JSON-extraction
        # heuristic that mangles fenced code responses
        text = (getattr(response, "raw_text", None)
                or (response.text if hasattr(response, "text")
                    else str(response)))
        self.transcript.append({"phase": tag,
                                "elapsed_s": round(time.time() - t0, 1),
                                "prompt": prompt[-4000:], "response": text})
        (workdir / "transcript.json").write_text(
            json.dumps(self.transcript, indent=1, default=str))
        return text

    @staticmethod
    def _extract_code(text: str) -> str:
        m = re.findall(r"```python\n(.*?)```", text, re.S)
        if not m:
            raise ValueError("LLM response contained no python code block")
        return m[-1]

    def _run_phase(self, tag: str, prompt: str, marker: str, workdir: Path,
                   timeout: int) -> dict:
        """LLM -> script -> sandboxed subprocess; one regenerate on failure.

        Returns the JSON payload printed on the marker line."""
        current = prompt
        for attempt in (1, 2, 3):
            code = self._extract_code(self._llm(f"{tag}_{attempt}", current,
                                                workdir))
            (workdir / f"{tag}_{attempt}.py").write_text(code)
            res = self.executor.execute_script(code, working_dir=str(workdir),
                                               timeout=timeout)
            stdout = res.get("stdout", "")
            (workdir / f"{tag}_{attempt}.out").write_text(
                stdout[-20000:] + "\n--- STATUS ---\n"
                + str(res.get("message", res.get("stderr", "")))[-6000:])
            payload = None
            for line in reversed(stdout.splitlines()):
                if line.startswith(marker):
                    payload = json.loads(line[len(marker):])
                    break
            if res.get("status") == "success" and payload is not None:
                return payload
            err = res.get("message") or ("script ran but did not print "
                                         + marker)
            current = (prompt
                       + f"\n\nYour previous script failed:\n```python\n{code}\n```\n"
                       f"Error / output tail:\n{str(err)[-2000:]}\n{stdout[-2000:]}\n"
                       "Fix the problem and return the complete corrected script.")
        raise RuntimeError(f"phase {tag} failed after retry")

    def _parse_interpretation(self, raw: str, workdir: Path) -> dict:
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
        try:
            fixed = self._llm("interpret_jsonfix",
                              "Convert the following into VALID json inside a "
                              "single ```json fenced block, with "
                              "detailed_analysis as one single string, "
                              "scientific_claims a list of objects, caveats a "
                              "string. Preserve content verbatim.\n\n" + raw,
                              workdir)
            return _attempt(fixed)
        except Exception:
            return {"detailed_analysis": raw, "scientific_claims": [],
                    "caveats": ""}

    # ------------------------------------------------------------------
    # deterministic gate
    # ------------------------------------------------------------------
    def _gate(self, result: dict, workdir: Path) -> dict:
        checks: Dict[str, Any] = {}
        files = result.get("files") or {}
        try:
            if "npy" not in files:
                checks["simulation_skipped"] = True
                for k, v in files.items():
                    checks[f"file_{k}_exists"] = bool(
                        (workdir / str(v)).exists())
            else:
                meta = json.loads((workdir / files["meta"]).read_text())
                img = np.load(workdir / files["npy"])
                checks["artifacts_exist"] = True
                checks["image_finite_nonuniform"] = bool(
                    np.isfinite(img).all() and img.std() > 0)
                det = meta.get("detector", {})
                checks["annulus_within_antialias"] = bool(
                    det.get("outer_mrad_effective", 1e9)
                    <= det.get("antialias_limit_mrad", 0))
                from scipy.ndimage import gaussian_filter, maximum_filter
                im = gaussian_filter(img.T.astype(float),
                                     0.35 / meta["scan_sampling_A"])
                lo, hi = np.percentile(im, [1, 99.5])
                imn = (im - lo) / (hi - lo)
                size = max(3, int(round(1.4 / meta["scan_sampling_A"])))
                peaks = ((imn == maximum_filter(imn, size=size))
                         & (imn > 0.12))
                n_img = int(peaks.sum())
                n_struct = int(result.get("n_columns_structure", 0))
                checks["n_columns_image"] = n_img
                checks["n_columns_structure"] = n_struct
                if n_struct == 0:
                    # cross-check UNAVAILABLE (no structure-side count) is
                    # not the same verdict as INCONSISTENT
                    checks["column_crosscheck"] = "unavailable"
                else:
                    ratio = n_img / n_struct
                    checks["column_count_ratio"] = round(ratio, 2)
                    checks["column_count_consistent"] = bool(
                        0.4 <= ratio <= 2.5)
        except Exception as exc:  # noqa: BLE001 - gate must always report
            checks["gate_error"] = f"{type(exc).__name__}: {exc}"
        checks["passed"] = all(v is True for k, v in checks.items()
                               if isinstance(v, bool))
        (workdir / "gate.json").write_text(json.dumps(checks, indent=1))
        return checks

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def analyze(self, data, system_info=None, objective: str | None = None,
                hints: str | None = None, **kwargs) -> Dict[str, Any]:
        path, paths, array, err = self._parse_data_input(data)
        if array is not None:
            return {"status": "error", "output_directory": str(self.output_dir),
                    "error": {"error": "In-memory arrays are not supported - "
                              "pass a structure file path (xyz/extxyz, LAMMPS "
                              "data, CIF, pos/epos/apt, or x,y,z csv)"}}
        if err or not path:
            return {"status": "error", "output_directory": str(self.output_dir),
                    "error": {"error": str(err) or "no input path"}}

        metadata = self._handle_system_info(system_info)
        objective = objective or ("Characterize this point cloud and report "
                                  "noteworthy structural and chemical features.")
        workdir = self.output_dir / (
            f"analysis_{Path(path).stem}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        workdir.mkdir(parents=True, exist_ok=True)
        self.transcript = []

        inventory = format_tool_inventory(agent="pointcloud_analysis",
                                          active_skills=DEFAULT_ACTIVE_SKILLS)
        common = (f"You are PointCloudAnalysisAgent, working on the atomistic "
                  f"point cloud at {path}.\n\nOBJECTIVE:\n{objective}\n\n"
                  + (f"HINTS:\n{hints}\n\n" if hints else "")
                  + f"METADATA:\n{json.dumps(metadata, indent=1)}\n\n"
                  f"AVAILABLE TOOLS (import and call exactly as documented):\n"
                  f"{inventory}\nPHYSICS NOTES:\n{_PHYSICS_NOTES}\n"
                  f"BUDGET NOTES:\n{_BUDGET_NOTES}\n")

        try:
            scout = self._run_phase(
                "scout",
                common + (
                    "PHASE 1 - SCOUT. Write ONE python script gathering, on "
                    "CPU only (no simulation), the evidence needed to plan: "
                    "classify the cloud kind (classify_cloud_kind), load and "
                    "profile the structure, evaluate candidate beam axes if "
                    "lattice-resolved work may apply, and localize any "
                    "feature the objective asks about. Print progress freely "
                    "and end with one line:\nSCOUT_JSON: {json evidence}"),
                "SCOUT_JSON:", workdir, timeout=900)

            result = self._run_phase(
                "commit",
                common + (
                    f"PHASE 2 - PLAN AND EXECUTE.\nScout evidence:\n"
                    f"{json.dumps(scout, indent=1)}\n\n"
                    "First, in comments at the top of your script, state your "
                    "DECISIONS with justification from the evidence: whether "
                    "to simulate (name the trigger, else skip), beam "
                    "orientation / ROI respecting the memory budget, and the "
                    "analyses the objective requires. Then execute them. End "
                    "with one line:\nRESULT_JSON: {\"decisions\": {...}, "
                    "\"files\": {label: filename for every artifact - when you "
                    "simulated an image use EXACTLY the keys \"npy\", "
                    "\"png\", \"meta\" for its three artifacts}, plus "
                    "your quantitative result fields}"),
                "RESULT_JSON:", workdir, timeout=self.commit_timeout)

            gate = self._gate(result, workdir)

            decisions = ""
            commit_file = workdir / "commit_1.py"
            if commit_file.exists():
                decisions = "\n".join(
                    l for l in commit_file.read_text().splitlines()
                    if l.startswith("#"))[:6000]

            raw = self._llm("interpret", common + (
                f"PHASE 3 - INTERPRET.\nScout evidence:\n"
                f"{json.dumps(scout, indent=1)}\n\nExecution results:\n"
                f"{json.dumps(result, indent=1)}\n\nVerification gate:\n"
                f"{json.dumps(gate, indent=1)}\n\n" + _INTERPRET_SCHEMA),
                workdir)
            interpretation = self._parse_interpretation(raw, workdir)
            (workdir / "interpretation.json").write_text(
                json.dumps(interpretation, indent=1))

            claims = self._validate_scientific_claims(
                interpretation.get("scientific_claims", []))

            images = {}
            files = result.get("files") or {}
            # hero first: an explicit hero_png, else an elements/overview
            # render, else the simulated image
            all_pngs = sorted(workdir.rglob("*.png"))
            hero = files.get("hero_png")
            if not hero:
                for cand in all_pngs:
                    if "element" in cand.name or "overview" in cand.name:
                        hero = str(cand)
                        break
            if hero:
                images["Reconstruction overview"] = hero
            if files.get("png"):
                images.setdefault("Simulated HAADF-STEM", files["png"])
            for extra in all_pngs[:10]:
                if str(extra) not in images.values() and \
                        str(extra.name) not in images.values():
                    images.setdefault(extra.stem.replace("_", " "),
                                      str(extra))
            interactive = {p.stem.replace("_", " "): str(p)
                           for p in sorted(workdir.rglob("*.html"))
                           if p.name != "report.html"}
            report = build_html_report(
                str(workdir), objective, metadata, scout, result, gate,
                interpretation, decisions_text=decisions, images=images,
                interactive=interactive)

            status = "success" if gate.get("passed") else "partial"
            out = {"status": status,
                   "detailed_analysis": interpretation.get(
                       "detailed_analysis", ""),
                   "scientific_claims": claims,
                   "caveats": interpretation.get("caveats", ""),
                   "output_directory": str(workdir),
                   "gate": gate, "scout": scout, "result": result,
                   "report_html": report}
            if status == "partial":
                out["warnings"] = ["deterministic verification gate failed - "
                                   "see gate field"]
            (workdir / "analysis_results.json").write_text(
                json.dumps(self._make_json_safe(out), indent=1))
            return out
        except Exception as exc:  # noqa: BLE001 - agent must return, not raise
            self.logger.exception("point-cloud analysis failed")
            return {"status": "error", "output_directory": str(workdir),
                    "error": {"error": f"{type(exc).__name__}: {exc}"}}

    @staticmethod
    def _make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: PointCloudAnalysisAgent._make_json_safe(v)
                    for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [PointCloudAnalysisAgent._make_json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
