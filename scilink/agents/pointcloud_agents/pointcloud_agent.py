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

_PLAN_SCHEMA = (
    'Produce your PLAN as a single fenced ```json block with exactly these '
    'fields:\n'
    '{"method_selection": [{"axis": "cloud kind | crystallographic gate | '
    'question specificity | rare species | concordance | simulate?", '
    '"decision": "...", "evidence": "scout field(s) that justify it"}], '
    '"steps": [{"tool": "registered tool name, or fresh_code", "purpose": '
    '"...", "why_fresh": "ONLY when tool is fresh_code: why no registered '
    'tool covers it"}], '
    '"gates": [{"name": "short id", "tool": "tool whose output carries the '
    'verdict", "pass_condition": "..."}], '
    '"deliverables": {"figures": ["hero_png", ... labels you will put in '
    'files], "interactive_3d": [labels], "report_sections": [...]}, '
    '"not_doing": [{"method": "...", "reason": "..."}]}\n'
    'Rules: prefer registered tools; every positive claim needs a gate from '
    'the list; name the hero_png figure first; keep it to what the objective '
    'and evidence warrant - do not plan the whole tool battery.')

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
    # USD per million tokens - an ESTIMATE for cost accounting (Opus-class
    # Bedrock list rates); override per deployment via the constructor kwarg
    # cost_rates={"in": .., "out": ..}. Reported alongside the token counts
    # so the raw numbers stay authoritative.
    COST_RATES_USD_PER_MTOK = {"in": 15.0, "out": 75.0}

    def _llm(self, tag: str, prompt: str, workdir: Path) -> str:
        t0 = time.time()
        response = self.model.generate_content(prompt)
        # raw_text, not .text: the wrapper's .text applies a JSON-extraction
        # heuristic that mangles fenced code responses
        text = (getattr(response, "raw_text", None)
                or (response.text if hasattr(response, "text")
                    else str(response)))
        usage = getattr(response, "usage", None) or {}
        t_in = int(usage.get("prompt_tokens") or 0)
        t_out = int(usage.get("completion_tokens") or 0)
        acc = getattr(self, "_cost", None)
        if acc is None:
            acc = self._cost = {"calls": 0, "tokens_in": 0, "tokens_out": 0}
        acc["calls"] += 1
        acc["tokens_in"] += t_in
        acc["tokens_out"] += t_out
        rates = getattr(self, "cost_rates", None) or self.COST_RATES_USD_PER_MTOK
        acc["cost_usd"] = round(acc["tokens_in"] / 1e6 * rates["in"]
                                + acc["tokens_out"] / 1e6 * rates["out"], 3)
        acc["rates_usd_per_mtok"] = rates
        self.transcript.append({"phase": tag,
                                "elapsed_s": round(time.time() - t0, 1),
                                "tokens_in": t_in, "tokens_out": t_out,
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
    # plan-first phase + deliverables contract
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json_block(text: str) -> dict:
        """Lenient JSON: fenced block -> trailing-comma repair -> first
        complete object."""
        m = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S)
        block = m[-1] if m else text[text.find("{"):]
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass
        repaired = re.sub(r",\s*([}\]])", r"\1", block)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return json.JSONDecoder().raw_decode(repaired)[0]

    def _plan(self, common: str, scout: dict, workdir: Path) -> dict:
        """PLAN phase: a structured, recorded contract BEFORE any analysis
        code is written - method-selection walk, tool steps, gates,
        deliverables, and what is deliberately not done."""
        raw = self._llm("plan", common + (
            "PHASE 2 - PLAN (no code yet).\nScout evidence:\n"
            f"{json.dumps(scout, indent=1)}\n\n"
            "Walk the method-selection axes IN ORDER citing scout evidence, "
            "then commit to the analysis steps, gates and deliverables. "
            + _PLAN_SCHEMA), workdir)
        try:
            plan = self._parse_json_block(raw)
        except Exception:  # noqa: BLE001 - never block the run on prose
            plan = {"parse_error": raw[-2000:], "steps": [], "gates": [],
                    "deliverables": {}}
        (workdir / "plan.json").write_text(json.dumps(plan, indent=1))
        return plan

    @staticmethod
    def _conformance(plan: dict, result: dict, workdir: Path) -> dict:
        """Deterministic plan-vs-execution check: planned tools must be
        called, planned gates must be reported (and pass), promised
        figures must exist on disk."""
        planned_tools = {s.get("tool", "") for s in plan.get("steps", [])
                         if s.get("tool") and s.get("tool") != "fresh_code"}
        scripts = sorted(workdir.glob("commit_*.py"))
        code = scripts[-1].read_text() if scripts else ""
        used = {t for t in planned_tools if re.search(rf"\b{t}\s*\(", code)}
        missing_tools = sorted(planned_tools - used)

        planned_gates = [g.get("name") for g in plan.get("gates", [])
                         if g.get("name")]
        reported = result.get("gates") or {}
        missing_gates = [g for g in planned_gates if g not in reported]
        failed_gates = [g for g in planned_gates if g in reported
                        and isinstance(reported[g], dict)
                        and reported[g].get("passed") is False]

        files = result.get("files") or {}
        promised = list((plan.get("deliverables") or {}).get("figures", []))
        promised += list((plan.get("deliverables") or {}).get(
            "interactive_3d", []))
        missing_files = []
        for lab in promised:
            f = files.get(lab)
            if not f or not (Path(f).exists()
                             or (workdir / str(f)).exists()):
                missing_files.append(lab)
        conf = {"planned_tools": sorted(planned_tools),
                "tools_called": sorted(used),
                "missing_tools": missing_tools,
                "planned_gates": planned_gates,
                "missing_gates": missing_gates,
                "failed_gates": failed_gates,
                "promised_files": promised,
                "missing_files": missing_files,
                "passed": not (missing_tools or missing_gates
                               or missing_files)}
        (workdir / "conformance.json").write_text(json.dumps(conf, indent=1))
        return conf

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
        self._cost = {"calls": 0, "tokens_in": 0, "tokens_out": 0}

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

            plan = self._plan(common, scout, workdir)

            commit_prompt = common + (
                f"PHASE 3 - EXECUTE THE PLAN.\nScout evidence:\n"
                f"{json.dumps(scout, indent=1)}\n\nYOUR PLAN (a binding "
                f"contract - execute exactly these steps with the named "
                f"tools; report every planned gate; produce every promised "
                f"deliverable under its planned label):\n"
                f"{json.dumps(plan, indent=1)}\n\n"
                "In comments at the top of your script restate the "
                "decisions. End with one line:\nRESULT_JSON: {\"decisions\": "
                "{...}, \"files\": {label: filename for every artifact - "
                "use the plan's deliverable labels; when you simulated an "
                "image use EXACTLY the keys \"npy\", \"png\", \"meta\"}, "
                "\"gates\": {planned gate name: {\"passed\": bool, ...verdict "
                "fields}}, plus your quantitative result fields}")
            result = self._run_phase("commit", commit_prompt, "RESULT_JSON:",
                                     workdir, timeout=self.commit_timeout)
            conf = self._conformance(plan, result, workdir)
            if not conf["passed"]:
                # one narrow re-execution: the contract is the error message
                result = self._run_phase(
                    "commit", commit_prompt + (
                        "\n\nCONFORMANCE FAILURE on your previous attempt - "
                        "fix ONLY these omissions and return the complete "
                        f"script:\n{json.dumps(conf, indent=1)}"),
                    "RESULT_JSON:", workdir, timeout=self.commit_timeout)
                conf = self._conformance(plan, result, workdir)

            gate = self._gate(result, workdir)
            gate["plan_conformance"] = bool(conf["passed"])
            gate["planned_gates_all_passed"] = not conf["failed_gates"]
            gate["conformance"] = conf
            gate["passed"] = bool(gate.get("passed") and conf["passed"]
                                  and not conf["failed_gates"])

            # plan + conformance become the report's decisions block
            walk = "\n".join(
                f"- {m.get('axis')}: {m.get('decision')} "
                f"(evidence: {m.get('evidence')})"
                for m in plan.get("method_selection", []))
            steps = "\n".join(
                f"- {s.get('tool')}: {s.get('purpose')}"
                + (f" [fresh code: {s.get('why_fresh')}]"
                   if s.get('tool') == 'fresh_code' else "")
                for s in plan.get("steps", []))
            not_doing = "\n".join(
                f"- {n.get('method')}: {n.get('reason')}"
                for n in plan.get("not_doing", []))
            decisions = (
                "PLAN (declared before execution)\n"
                f"Method-selection walk:\n{walk}\n\nSteps:\n{steps}\n\n"
                f"Not doing:\n{not_doing or '- (none)'}\n\n"
                f"CONFORMANCE: {'PASSED' if conf['passed'] else 'FAILED'}"
                f" - tools called {conf['tools_called']}; missing tools "
                f"{conf['missing_tools']}; missing gates "
                f"{conf['missing_gates']}; failed gates "
                f"{conf['failed_gates']}; missing files "
                f"{conf['missing_files']}")[:8000]

            raw = self._llm("interpret", common + (
                f"PHASE 4 - INTERPRET.\nScout evidence:\n"
                f"{json.dumps(scout, indent=1)}\n\nPlan (the contract):\n"
                f"{json.dumps(plan, indent=1)}\n\nExecution results:\n"
                f"{json.dumps(result, indent=1)}\n\nVerification gate "
                f"(incl. plan conformance - a planned gate that failed or "
                f"a promised deliverable that is missing MUST be stated, "
                f"never papered over):\n{json.dumps(gate, indent=1)}\n\n"
                + _INTERPRET_SCHEMA), workdir)
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
            cost = dict(getattr(self, "_cost", {}) or {})
            report = build_html_report(
                str(workdir), objective, metadata, scout, result, gate,
                interpretation, decisions_text=decisions, images=images,
                interactive=interactive, cost=cost or None)

            status = "success" if gate.get("passed") else "partial"
            out = {"status": status,
                   "detailed_analysis": interpretation.get(
                       "detailed_analysis", ""),
                   "scientific_claims": claims,
                   "caveats": interpretation.get("caveats", ""),
                   "output_directory": str(workdir),
                   "gate": gate, "scout": scout, "result": result,
                   "report_html": report, "plan": plan, "cost": cost}
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
