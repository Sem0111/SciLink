"""
Plugin adapter for scilink analyze --agents.

Usage (from any directory):

  # 1. Folder-based (raw APT data) — recommended
  #    Pass a directory that contains a point cloud (.apt, .pos, or .csv)
  #    and a range file (.rrng).  The agent auto-detects the files, runs
  #    neighborhood generation, then performs CCD analysis.
  scilink analyze --agents /abs/path/to/apt_agent_plugin.py \
                  --data /path/to/data_folder

  # 2. Pre-computed neighborhoods CSV
  scilink analyze --agents /abs/path/to/apt_agent_plugin.py \
                  --data neighborhoods.csv \
                  --metadata metadata.json

The scilink CLI discovers agents by checking cls.__module__ == module.__name__.
Defining a subclass HERE (rather than patching __module__) satisfies that check
automatically, because the class is genuinely defined in this file.
"""
import sys
import types
import importlib.util
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Bootstrap a synthetic package so that relative imports inside ccd_apt_agent
# (and its siblings) resolve correctly when loaded by scilink as a standalone
# file rather than as part of an installed package.
#
# Strategy:
#   1. Register "_ccd_apt_pkg" in sys.modules as a package whose __path__
#      points to this directory.
#   2. Register "_ccd_apt_pkg.OPTICSAPT" as a sub-package.
#   3. Load ccd_apt_agent.py via importlib with its __name__ and __package__
#      set to the synthetic package, so "from .foo import bar" works.
#
# All further relative imports are then handled by Python's normal machinery.
# ---------------------------------------------------------------------------

_PKG = "_ccd_apt_pkg"


def _ensure_synthetic_package() -> None:
    """Create the synthetic package entries if they don't already exist."""
    if _PKG in sys.modules:
        return

    # Top-level package
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(_plugin_dir)]       # tells importer where to look
    pkg.__package__ = _PKG
    pkg.__file__ = str(_plugin_dir / "__init__.py")
    sys.modules[_PKG] = pkg

    # OPTICSAPT sub-package (has a real __init__.py)
    _opticsapt_dir = _plugin_dir / "OPTICSAPT"
    _opticsapt_name = f"{_PKG}.OPTICSAPT"
    opticsapt_pkg = types.ModuleType(_opticsapt_name)
    opticsapt_pkg.__path__ = [str(_opticsapt_dir)]
    opticsapt_pkg.__package__ = _opticsapt_name
    opticsapt_pkg.__file__ = str(_opticsapt_dir / "__init__.py")
    sys.modules[_opticsapt_name] = opticsapt_pkg


def _load_submodule(stem: str):
    """Load <_plugin_dir>/<stem>.py as _ccd_apt_pkg.<stem>."""
    full_name = f"{_PKG}.{stem}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(
        full_name,
        str(_plugin_dir / f"{stem}.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_synthetic_package()
_agent_module = _load_submodule("ccd_apt_agent")
_CCDAPTAnalysisAgent = _agent_module.CCDAPTAnalysisAgent


class CCDAPTAnalysisAgent(_CCDAPTAnalysisAgent):
    """
    APT Compositional Community Detection (CCD) analysis agent.

    Supports two entry points:

    analyze_from_folder(data_folder, ...)
        Pass a directory containing a raw point cloud file (.apt, .pos, or
        .csv) and an .rrng range file. The file endings may be capitalized 
        or lowercase. The agent:
          1. Auto-detects the point cloud and range file.
          2. If ``enable_human_feedback=True``, asks the user to confirm the
             detected files before continuing.
          3. Runs ``ccd.generate_neighborhoods`` to produce a neighbourhoods CSV.
          4. Runs the full CCD pipeline (LLM parameter selection → k-means →
             KS statistics → community detection → LLM interpretation).

        Point-cloud CSV support: a CSV with columns x/y/z (position) and Da
        (mass-to-charge) is treated as a raw reconstructed point cloud and
        ranged against the .rrng file — distinct from a pre-computed
        neighbourhoods CSV (which has midpoint_x/y/z and density columns).

    analyze(data, ...)
        Accepts a pre-computed neighbourhoods .csv path, a list of such paths,
        or a pd.DataFrame.  Runs the CCD pipeline directly without the
        neighbourhood generation step.

    Input:  data folder  OR  .csv neighbourhood file(s) / pd.DataFrame
    Output: community count, compositions, KS-statistic heatmap, LLM analysis
    """
