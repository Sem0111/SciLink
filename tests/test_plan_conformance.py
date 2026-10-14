"""Offline tests for the plan-first phase: lenient plan parsing and the
deterministic plan-vs-execution conformance check (no LLM, no sandbox).

  python tests/test_plan_conformance.py
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.pointcloud_agents.pointcloud_agent import (
    PointCloudAnalysisAgent as A)

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    print("1) plan parsing:")
    raw = ('Here is my plan.\n```json\n{"steps": [{"tool": "msm_detect", '
           '"purpose": "x",}], "gates": [{"name": "shuffle",},],}\n```\n'
           'thanks')
    plan = A._parse_json_block(raw)
    check("trailing-comma plan parses", plan["steps"][0]["tool"] == "msm_detect")

    print("2) conformance - fully conformant run:")
    wd = Path(tempfile.mkdtemp())
    (wd / "commit_1.py").write_text(
        "from x import msm_detect, label_shuffle_null\n"
        "d = msm_detect(a, b)\ng = label_shuffle_null(a)\n")
    (wd / "hero.png").write_text("x")
    plan = {"steps": [{"tool": "msm_detect"}, {"tool": "label_shuffle_null"},
                      {"tool": "fresh_code", "why_fresh": "n/a"}],
            "gates": [{"name": "shuffle"}],
            "deliverables": {"figures": ["hero_png"]}}
    result = {"files": {"hero_png": str(wd / "hero.png")},
              "gates": {"shuffle": {"passed": True, "z": 9.0}}}
    conf = A._conformance(plan, result, wd)
    check("conformant run passes", conf["passed"])
    check("fresh_code steps are not counted as tools",
          "fresh_code" not in conf["planned_tools"])
    check("conformance.json written", (wd / "conformance.json").exists())

    print("3) conformance - omissions detected:")
    (wd / "commit_2.py").write_text("d = msm_detect(a, b)\n")  # no null gate
    result2 = {"files": {}, "gates": {}}
    conf2 = A._conformance(plan, result2, wd)
    check("missing tool detected", conf2["missing_tools"] == ["label_shuffle_null"])
    check("missing gate detected", conf2["missing_gates"] == ["shuffle"])
    check("missing figure detected", conf2["missing_files"] == ["hero_png"])
    check("nonconformant run fails", not conf2["passed"])

    print("4) conformance - planned gate reported but FAILED:")
    (wd / "commit_3.py").write_text("d = msm_detect(a, b)\ng = label_shuffle_null(a)\n")
    result3 = {"files": {"hero_png": str(wd / "hero.png")},
               "gates": {"shuffle": {"passed": False}}}
    conf3 = A._conformance(plan, result3, wd)
    check("failed gate listed (conformance itself passes)",
          conf3["failed_gates"] == ["shuffle"] and conf3["passed"])

    n_fail = sum(not v for v in results.values())
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
