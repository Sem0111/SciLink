"""3D structural identification and defect extraction for point clouds.

Wraps OVITO's Python module (free package; PTM per Larsen, Schmidt &
Schiotz, MSMSE 24, 055007 (2016); DXA per Stukowski et al.) behind small
functions that take ASE Atoms. ovito is imported lazily inside functions so
this module always imports; the skill's detect block declares the optional
dependency. The MIT-licensed reference implementation
(github.com/pmla/polyhedral-template-matching) is the fallback path if the
ovito dependency is ever unwanted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scilink.skills._shared._spec import ToolSpec

_TYPE_NAMES = {0: "OTHER", 1: "FCC", 2: "HCP", 3: "BCC", 4: "ICO", 5: "SC",
               6: "CUBIC_DIAMOND", 7: "HEX_DIAMOND", 8: "GRAPHENE"}


def _pipeline_from_atoms(atoms):
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import Pipeline, StaticSource
    return Pipeline(source=StaticSource(data=ase_to_ovito(atoms)))


def classify_structure_3d(atoms, rmsd_cutoff: float = 0.15,
                          out_prefix: str | None = None,
                          workdir: str = ".") -> dict:
    """Per-atom 3D Polyhedral Template Matching.

    Returns per-type fractions and (optionally, via ``out_prefix``) saves the
    per-atom types + positions as ``<out_prefix>_ptm3d.npz`` for downstream
    visualization/defect extraction. The dominant type is the host lattice.
    """
    from ovito.modifiers import PolyhedralTemplateMatchingModifier

    pipe = _pipeline_from_atoms(atoms)
    pipe.modifiers.append(
        PolyhedralTemplateMatchingModifier(rmsd_cutoff=rmsd_cutoff))
    data = pipe.compute()
    types = np.asarray(data.particles.structure_types.array)
    frac = {_TYPE_NAMES.get(int(t), str(t)): round(float(np.mean(types == t)), 4)
            for t in np.unique(types)}
    host = _TYPE_NAMES.get(int(np.bincount(types).argmax()), "OTHER")
    result = {"fractions": frac, "host_structure": host,
              "rmsd_cutoff": rmsd_cutoff, "n_atoms": int(len(types))}
    if out_prefix:
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(wd / f"{out_prefix}_ptm3d.npz",
                            positions=atoms.get_positions(), types=types)
        result["npz"] = f"{out_prefix}_ptm3d.npz"
    return result


def extract_defects_3d(atoms, types: np.ndarray | None = None,
                       host: str | None = None,
                       cluster_cutoff: float = 3.5) -> dict:
    """Group non-host atoms into connected defect clusters.

    Each cluster gets size, composition by structure type, centroid, and
    extent - a coherent twin appears as one large planar HCP cluster; faults
    and boundaries appear as sheets, disordered pockets as blobs.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if types is None:
        raise ValueError("pass the per-atom types from classify_structure_3d")
    types = np.asarray(types)
    name_of = np.vectorize(lambda t: _TYPE_NAMES.get(int(t), str(t)))
    host = host or _TYPE_NAMES.get(int(np.bincount(types).argmax()))
    mask = name_of(types) != host
    p = atoms.get_positions()[mask]
    t = types[mask]
    if len(p) == 0:
        return {"host_structure": host, "n_defect_atoms": 0, "clusters": []}
    tree = cKDTree(p)
    pairs = tree.query_pairs(cluster_cutoff, output_type="ndarray")
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                     shape=(len(p), len(p)))
    ncomp, labels = connected_components(adj, directed=False)
    clusters = []
    for c in range(ncomp):
        m = labels == c
        if m.sum() < 10:
            continue
        comp = {str(k): int(v) for k, v in
                zip(*np.unique(name_of(t[m]), return_counts=True))}
        ext = p[m].max(axis=0) - p[m].min(axis=0)
        planar = bool(np.min(ext) < 0.15 * np.max(ext))
        clusters.append({"n_atoms": int(m.sum()), "composition": comp,
                         "centroid_A": [round(float(v), 1)
                                        for v in p[m].mean(axis=0)],
                         "extent_A": [round(float(v), 1) for v in ext],
                         "planar": planar})
    clusters.sort(key=lambda c: -c["n_atoms"])
    return {"host_structure": host, "n_defect_atoms": int(mask.sum()),
            "n_clusters": len(clusters), "clusters": clusters[:20]}


def analyze_dislocations(atoms, lattice: str = "fcc") -> dict:
    """DXA dislocation extraction (free OVITO package - verified).

    Returns segments with true Burgers vectors and lengths plus the total
    line length; zero segments on a defect-free or purely planar-defect
    structure is a meaningful result, not a failure.
    """
    from ovito.modifiers import DislocationAnalysisModifier as DXA

    latmap = {"fcc": DXA.Lattice.FCC, "bcc": DXA.Lattice.BCC,
              "hcp": DXA.Lattice.HCP,
              "diamond": DXA.Lattice.CubicDiamond}
    pipe = _pipeline_from_atoms(atoms)
    dxa = DXA()
    dxa.input_crystal_structure = latmap[lattice.lower()]
    pipe.modifiers.append(dxa)
    data = pipe.compute()
    segs = [{"true_burgers_vector": [round(float(v), 3)
                                     for v in s.true_burgers_vector],
             "length_A": round(float(s.length), 1)}
            for s in data.dislocations.segments]
    return {"lattice": lattice, "n_segments": len(segs),
            "total_line_length_A": round(float(
                data.attributes.get("DislocationAnalysis.total_line_length",
                                    0.0)), 1),
            "segments": segs[:20]}


def visualize_defects_3d(atoms, types: np.ndarray, out_prefix: str,
                         workdir: str = ".", max_points: int = 60000) -> dict:
    """Defect-atoms-only 3D visualization (the OVITO idiom: hide the host).

    Writes ``<out_prefix>_defects3d.html`` (self-contained interactive
    plotly, if plotly is installed) and ``<out_prefix>_defects3d.png``
    (matplotlib 3-view projections, always).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    types = np.asarray(types)
    host = int(np.bincount(types).argmax())
    mask = types != host
    p = atoms.get_positions()[mask]
    t = types[mask]
    if len(p) > max_points:
        sel = np.random.RandomState(0).choice(len(p), max_points, replace=False)
        p, t = p[sel], t[sel]
    colors = {2: "#b51d14", 3: "#ddb310", 0: "#777777", 1: "#4053d3",
              4: "#7f2ccb", 5: "#00b25d"}
    # boundary-hugging clusters are SURFACE-UNCLASSIFIABLE atoms (finite-
    # specimen artifact), not defects - title figures accordingly
    ext_lo, ext_hi = atoms.get_positions().min(0), atoms.get_positions().max(0)
    near_edge = np.zeros(len(p), bool)
    for k in range(3):
        near_edge |= (p[:, k] < ext_lo[k] + 6.0) | (p[:, k] > ext_hi[k] - 6.0)
    surf_frac = float(near_edge.mean()) if len(p) else 0.0
    fig_title = ("surface-unclassifiable atoms (finite-specimen artifact)"
                 if surf_frac > 0.8 else "non-host atoms (potential defects)")
    out = {"n_defect_atoms_shown": int(len(p)),
           "host_hidden": _TYPE_NAMES.get(host),
           "surface_fraction_of_shown": round(surf_frac, 3),
           "figure_title": fig_title}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (i, j, lab) in zip(axes, [(0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]):
        for tv in np.unique(t):
            m = t == tv
            ax.scatter(p[m, i], p[m, j], s=2, lw=0,
                       c=colors.get(int(tv), "#333333"),
                       label=_TYPE_NAMES.get(int(tv)))
        ax.set_aspect(1)
        ax.set_title(f"{fig_title} - {lab}", fontsize=9)
        ax.legend(markerscale=4, fontsize=8)
    png = wd / f"{out_prefix}_defects3d.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    out["png"] = png.name

    try:
        import plotly.graph_objects as go
        traces = []
        for tv in np.unique(t):
            m = t == tv
            traces.append(go.Scatter3d(
                x=p[m, 0], y=p[m, 1], z=p[m, 2], mode="markers",
                name=_TYPE_NAMES.get(int(tv)),
                marker=dict(size=2, color=colors.get(int(tv), "#333333"))))
        figp = go.Figure(traces)
        figp.update_layout(scene_aspectmode="data",
                           title=f"{fig_title} (host {out['host_hidden']} hidden)")
        html = wd / f"{out_prefix}_defects3d.html"
        figp.write_html(html, include_plotlyjs=True)
        out["html"] = html.name
    except ImportError:
        out["html"] = None
    return out


def _spec(name, desc, sig, when, returns, example, params=None, req=None):
    return ToolSpec(
        name=name, description=desc, parameters=params or {}, required=req or [],
        import_line=(f"from scilink.skills.point_cloud_analysis.structure_id_3d"
                     f".ptm3d_tools import {name}"),
        signature=sig, agents=["simulation"], when_to_use=when,
        returns=returns, example=example)


TOOL_SPECS = [
    _spec("classify_structure_3d",
          "Per-atom 3D Polyhedral Template Matching (OVITO/Larsen): "
          "FCC/HCP/BCC/... fractions, host lattice, per-atom types npz.",
          "classify_structure_3d(atoms, rmsd_cutoff=0.15, out_prefix=None, "
          "workdir='.') -> dict",
          "Ground-truth 3D structural identification of any point cloud; "
          "run before defect extraction or visualization.",
          "fractions, host_structure, optional per-atom npz",
          "res = classify_structure_3d(atoms, out_prefix='case')"),
    _spec("extract_defects_3d",
          "Cluster non-host atoms into defect objects with size, "
          "composition, centroid, extent, planarity (twin = one large "
          "planar HCP cluster).",
          "extract_defects_3d(atoms, types, host=None, cluster_cutoff=3.5) "
          "-> dict",
          "After classify_structure_3d, to enumerate and characterize "
          "defects; planar+HCP => twin/fault, blobs => disorder.",
          "clusters list with composition/centroid/extent/planar",
          "defects = extract_defects_3d(atoms, np.load('case_ptm3d.npz')['types'])"),
    _spec("analyze_dislocations",
          "DXA dislocation extraction: segments, true Burgers vectors, "
          "line lengths (free-package verified).",
          "analyze_dislocations(atoms, lattice='fcc') -> dict",
          "When the objective mentions dislocations/plasticity, or to "
          "confirm their absence; zero segments is a real result.",
          "n_segments, total_line_length_A, segments",
          "dxa = analyze_dislocations(atoms, lattice='fcc')",
          params={"lattice": {"type": "string",
                              "description": "host lattice: fcc/bcc/hcp/diamond"}}),
    _spec("visualize_defects_3d",
          "3D defect visualization, host atoms hidden: interactive "
          "self-contained HTML (plotly) + 3-view projection PNG.",
          "visualize_defects_3d(atoms, types, out_prefix, workdir='.') -> dict",
          "The presentation artifact: show WHERE defects are in 3D. Use the "
          "types npz from classify_structure_3d.",
          "filenames of html/png artifacts",
          "viz = visualize_defects_3d(atoms, types, 'case', workdir='out')"),
]
