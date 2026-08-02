"""Compositional Community Detection (CCD) for APT and simulated point clouds.

Wraps the CCD implementation vendored in ``_ccd/`` (with permission).
Method and code: Bilbrey, Doty, Wirth, Tong, Royer, Senor & Devaraj,
"Compositional Community Detection: Automated Identification of Chemical
Segregation in Atom Probe Tomography Data", Microscopy and Microanalysis 31
(2025); maintained by Jenna Pope (jenna.pope@pnnl.gov). Cite when used.

Two input routes, one algorithm:
  - real APT: .pos/.apt + .rrng, passed straight to the vendored pipeline
  - simulated structures (xyz / LAMMPS / CIF via read_structure): each
    element is written as a synthetic mass line (Da = atomic mass) with a
    matching synthetic .rrng, so the vendored ranging + neighborhood +
    detection code runs verbatim - no algorithm duplication.

Coordinates: APT convention is nm; simulated structures in Angstrom are
converted (A -> nm) by the adapter.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from scilink.skills._shared._spec import ToolSpec


def _jsonable(obj):
    if isinstance(obj, Counter):
        return {str(k): int(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if hasattr(obj, "to_dict"):
        return _jsonable(obj.to_dict())
    return obj


def neighborhoods_from_apt(pos_path: str, rrng_path: str,
                           radius_nm: float = 1.0, overlap: float = 0.5,
                           savedir: str = "ccd_analysis") -> dict:
    """Overlapping spherical composition neighborhoods from real APT data."""
    from ._ccd.ccd import generate_neighborhoods
    res = generate_neighborhoods(pos_path, rrng_path, savedir=savedir,
                                 radius=radius_nm, overlap=overlap)
    res["neighborhood_csv"] = str(
        Path(savedir) / f"{Path(pos_path).stem}_{radius_nm}nm-radius_"
                        f"{overlap}-overlap.csv")
    return _jsonable(res)


def neighborhoods_from_structure(structure_path: str, type_map: dict | None = None,
                                 radius_nm: float = 1.0, overlap: float = 0.5,
                                 savedir: str = "ccd_analysis") -> dict:
    """Same neighborhoods for a simulated structure (ideal-detection APT).

    Writes a synthetic (x, y, z[nm], Da=atomic mass) csv plus a matching
    .rrng so the vendored CCD pipeline runs unchanged on simulated data.
    """
    from ase.data import atomic_masses, atomic_numbers

    from scilink.skills.stem_simulation.haadf_workflow.abtem_tools import (
        read_structure)
    from ._ccd.ccd import generate_neighborhoods

    atoms, info = read_structure(structure_path, type_map=type_map)
    out = Path(savedir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(structure_path).stem + "_sim"

    pos_nm = atoms.get_positions() / 10.0
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    mass_of = {s: float(atomic_masses[atomic_numbers[s]]) for s in species}
    da = np.array([mass_of[s] for s in symbols])

    csv_path = out / f"{stem}.csv"
    np.savetxt(csv_path, np.column_stack([pos_nm, da]), delimiter=",",
               fmt="%.4f")

    rrng_path = out / f"{stem}.rrng"
    lines = ["[Ions]", f"Number={len(species)}"]
    lines += [f"Ion{i}={s}" for i, s in enumerate(species, 1)]
    lines += ["[Ranges]", f"Number={len(species)}"]
    lines += [f"Range{i}={mass_of[s]-0.4:.2f} {mass_of[s]+0.4:.2f} "
              f"Vol:0.0 Name:{s} Color:836EAA"
              for i, s in enumerate(species, 1)]
    rrng_path.write_text("\n".join(lines) + "\n")

    res = generate_neighborhoods(str(csv_path), str(rrng_path),
                                 savedir=str(out), radius=radius_nm,
                                 overlap=overlap)
    res["neighborhood_csv"] = str(
        out / f"{stem}_{radius_nm}nm-radius_{overlap}-overlap.csv")
    res["note"] = ("simulated structure treated as ideal-detection APT "
                   "(100% efficiency, no trajectory aberrations)")
    return _jsonable(res)


def detect_segregation(neighborhood_csv: str, savedir: str = "ccd_analysis",
                       k_values: list | None = None,
                       ignore_ions: list | None = None,
                       n_repeats: int = 2, q: int = 25) -> dict:
    """Run Compositional Community Detection on a neighborhood csv.

    Returns community count, per-community neighborhood counts, and
    per-community signed KS statistics per ion (positive = enriched vs the
    bulk distribution, negative = depleted); also writes the community-
    labelled point cloud (.xyz) and plots into ``savedir``.
    """
    from ._ccd.ccd import detect_compositional_communities
    res = detect_compositional_communities(
        neighborhood_csv, savedir=savedir,
        k_values=k_values or [4, 5, 6],
        ignore_ions=ignore_ions or [], n_repeats=n_repeats, q=q)
    res = _jsonable(res)
    stem = Path(neighborhood_csv).stem
    res["community_xyz"] = str(Path(savedir) / f"{stem}_community_clustering.xyz")
    try:
        res["community_map_png"] = visualize_communities(
            res["community_xyz"], out_prefix=stem, workdir=savedir)["png"]
    except Exception as exc:  # noqa: BLE001 - viz is best-effort
        res["community_map_png"] = None
        res["viz_error"] = str(exc)
    ks = Path(savedir) / "KS_stats.png"
    if ks.exists():
        res["ks_plot_png"] = str(ks)
    return res


def visualize_communities(community_xyz: str, out_prefix: str,
                          workdir: str = "ccd_analysis") -> dict:
    """Render the community-labelled point cloud: three-view projections of
    neighborhood centers colored by community - the segregation map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = np.loadtxt(community_xyz, skiprows=2)
    com = rows[:, 0].astype(int)
    p = rows[:, 1:4]
    colors = ["#4053d3", "#b51d14", "#ddb310", "#00b25d", "#7f2ccb",
              "#fb49b0", "#00beff"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (i, j, lab) in zip(axes, [(0, 2, "x-z"), (1, 2, "y-z"),
                                      (0, 1, "x-y")]):
        for c in sorted(set(com.tolist())):
            m = com == c
            ax.scatter(p[m, i], p[m, j], s=3, lw=0,
                       c=colors[c % len(colors)], label=f"community {c}")
        ax.set_aspect(1)
        ax.set_title(f"communities - {lab} [nm]")
    axes[0].legend(markerscale=4, fontsize=8)
    out = Path(workdir) / f"{out_prefix}_community_map.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(out), "n_communities": int(len(set(com.tolist())))}


_IMP = "from scilink.skills.point_cloud_analysis.apt_ccd.apt_tools import "

TOOL_SPECS = [
    ToolSpec(
        name="neighborhoods_from_structure",
        description=("Overlapping spherical composition neighborhoods from a "
                     "SIMULATED structure (ideal-detection APT): synthetic "
                     "mass/rrng adapter feeding the vendored CCD pipeline."),
        parameters={"structure_path": {"type": "string",
                                       "description": "xyz/LAMMPS/CIF path"},
                    "radius_nm": {"type": "number",
                                  "description": "neighborhood radius "
                                                 "(default 1.0 nm; smaller "
                                                 "probes finer segregation)"}},
        required=["structure_path"],
        import_line=_IMP + "neighborhoods_from_structure",
        signature=("neighborhoods_from_structure(structure_path, "
                   "type_map=None, radius_nm=1.0, overlap=0.5, "
                   "savedir='ccd_analysis') -> dict"),
        agents=["simulation"],
        when_to_use=("First CCD step on simulated point clouds when the "
                     "objective concerns compositional segregation, "
                     "clustering or ordering."),
        returns="neighborhood stats + neighborhood_csv path for detection",
        example=("nb = neighborhoods_from_structure('tip.xyz', "
                 "radius_nm=1.0)"),
    ),
    ToolSpec(
        name="neighborhoods_from_apt",
        description=("Same neighborhoods from REAL reconstructed APT data "
                     "(.pos/.apt + .rrng)."),
        parameters={"pos_path": {"type": "string", "description": ".pos/.apt"},
                    "rrng_path": {"type": "string", "description": ".rrng"}},
        required=["pos_path", "rrng_path"],
        import_line=_IMP + "neighborhoods_from_apt",
        signature=("neighborhoods_from_apt(pos_path, rrng_path, "
                   "radius_nm=1.0, overlap=0.5, savedir='ccd_analysis') "
                   "-> dict"),
        agents=["simulation"],
        when_to_use="First CCD step on experimental APT reconstructions.",
        returns="neighborhood stats + neighborhood_csv path",
        example="nb = neighborhoods_from_apt('sample.pos', 'ranges.rrng')",
    ),
    ToolSpec(
        name="detect_segregation",
        description=("Compositional Community Detection (KMeans ensemble + "
                     "Louvain over the co-clustering graph): identifies "
                     "compositionally distinct regions and their "
                     "enrichment/depletion signatures (signed KS statistics "
                     "per ion)."),
        parameters={"neighborhood_csv": {"type": "string",
                                         "description": "csv from a "
                                                        "neighborhoods_* call"},
                    "ignore_ions": {"type": "array",
                                    "description": "ion labels to exclude"}},
        required=["neighborhood_csv"],
        import_line=_IMP + "detect_segregation",
        signature=("detect_segregation(neighborhood_csv, "
                   "savedir='ccd_analysis', k_values=[4,5,6], "
                   "ignore_ions=None, n_repeats=2, q=25) -> dict"),
        agents=["simulation"],
        when_to_use=("After neighborhood generation, to find and "
                     "characterize segregated regions; positive KS = "
                     "enriched, negative = depleted vs bulk."),
        returns=("community_count, community_neighborhood_counts, "
                 "community_compositions (signed KS per ion), "
                 "community_xyz point cloud"),
        example="seg = detect_segregation(nb['neighborhood_csv'])",
    ),
]
