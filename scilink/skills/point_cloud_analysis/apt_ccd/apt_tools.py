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
    """Overlapping spherical composition neighborhoods from real APT data.

    .pos is read by the vendored pipeline directly. An AP Suite ``.apt``
    binary is first converted (via apav) to the pipeline's csv route
    (x, y, z, Da) - the vendored readers predate the .apt format.
    """
    from ._ccd.ccd import generate_neighborhoods

    src = Path(pos_path)
    if src.suffix.lower() == ".apt":
        try:
            import apav
        except ImportError as exc:
            raise ImportError(
                ".apt binaries need apav (pip install apav) - or export a "
                ".pos from AP Suite instead") from exc
        out = Path(savedir)
        out.mkdir(parents=True, exist_ok=True)
        roi = apav.load_apt(str(src))
        conv = out / f"{src.stem}.csv"
        np.savetxt(conv, np.column_stack([roi.xyz, roi.mass]),
                   delimiter=",", fmt="%.5f")
        pos_path = str(conv)

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
    import re as _re

    from ._ccd.ccd import detect_compositional_communities

    # Sanitize ignore_ions against the labels actually present: the csv
    # carries rrng formula labels (H1, C2, Cr1O1, ...) while callers pass
    # element names (H, C). A requested element expands to every label
    # composed SOLELY of requested elements (drops H1/H2/C1N1, keeps Fe1H1);
    # unmatched requests are dropped with a note instead of crashing the
    # vendored pipeline (ValueError: list.remove).
    ignored_effective, ignored_unmatched = [], []
    if ignore_ions:
        header = Path(neighborhood_csv).open().readline().strip().split(",")
        present = [c[1:] for c in header if c.startswith("p")]

        def _elems(label):
            return {sym for sym, _ in
                    _re.findall(r"([A-Z][a-z]?)(\d*)", label) if sym}

        req_elems = set()
        for tok in ignore_ions:
            if tok in present:
                ignored_effective.append(tok)
            else:
                req_elems |= _elems(tok)
        for lab in present:
            if lab not in ignored_effective and _elems(lab) <= req_elems \
                    and _elems(lab):
                ignored_effective.append(lab)
        ignored_unmatched = [t for t in ignore_ions
                             if t not in present
                             and not any(_elems(t) & _elems(l)
                                         for l in ignored_effective)]
    res = detect_compositional_communities(
        neighborhood_csv, savedir=savedir,
        k_values=k_values or [4, 5, 6],
        ignore_ions=ignored_effective, n_repeats=n_repeats, q=q)
    res["ignored_ion_labels"] = ignored_effective
    if ignored_unmatched:
        res["ignore_requests_unmatched"] = ignored_unmatched
    # Make compositions self-describing: {community_id: {ion: signed_KS}}
    # instead of a bare list-of-lists whose ion order lives in a side file.
    header = Path(neighborhood_csv).open().readline().strip().split(",")
    ions = sorted(c[1:] for c in header if c.startswith("p"))
    ions = [i for i in ions if i not in ignored_effective]
    raw = res.get("community_compositions")
    if isinstance(raw, (list, tuple)):
        res["community_compositions"] = {
            str(cid): {ion: round(float(ks), 4)
                       for ion, ks in zip(ions, row)}
            for cid, row in enumerate(raw)}
        res["community_composition_ions"] = ions
    res = _jsonable(res)
    stem = Path(neighborhood_csv).stem
    res["community_xyz"] = str(Path(savedir) / f"{stem}_community_clustering.xyz")
    try:
        meanings = annotate_communities(
            res.get("community_compositions") or {})
        res["community_meanings"] = meanings
        viz = visualize_communities(res["community_xyz"], out_prefix=stem,
                                    workdir=savedir,
                                    community_meanings=meanings)
        res["community_map_png"] = viz.get("png")
        res["community_map_3d_html"] = viz.get("html")
    except Exception as exc:  # noqa: BLE001 - viz is best-effort
        res["community_map_png"] = None
        res["viz_error"] = str(exc)
    ks = Path(savedir) / "KS_stats.png"
    if ks.exists():
        res["ks_plot_png"] = str(ks)
    return res


def visualize_communities(community_xyz: str, out_prefix: str,
                          workdir: str = "ccd_analysis",
                          community_meanings: dict | None = None,
                          apex_up: bool = True,
                          make_3d_html: bool = True) -> dict:
    """Render the community-labelled point cloud - the segregation map.

    Legend entries carry MEANING when ``community_meanings`` is given (use
    annotate_communities on the detect_segregation compositions). APT
    convention: apex at the top of the figure (apex_up flips z). Also
    writes an interactive 3D html when plotly is available.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = np.loadtxt(community_xyz, skiprows=2)
    com = rows[:, 0].astype(int)
    p = rows[:, 1:4].copy()
    if apex_up:
        p[:, 2] = p[:, 2].max() - p[:, 2]
    colors = ["#4053d3", "#b51d14", "#ddb310", "#00b25d", "#7f2ccb",
              "#fb49b0", "#00beff"]
    meanings = {str(k): v for k, v in (community_meanings or {}).items()}

    def _name(c):
        if c < 0:
            return "unassigned (ambiguous across clusterings)"
        return meanings.get(str(c), f"community {c}")

    order = sorted(set(com.tolist()), key=lambda c: -(com == c).sum())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (i, j, lab) in zip(axes, [(0, 2, "x-z"), (1, 2, "y-z"),
                                      (0, 1, "x-y")]):
        for c in order:
            m = com == c
            ax.scatter(p[m, i], p[m, j], s=3, lw=0,
                       c=colors[c % len(colors)], label=_name(c))
        ax.set_aspect(1)
        if j == 2:
            ax.invert_yaxis()
            ax.set_title(f"communities - {lab} [nm] (apex at top)")
        else:
            ax.set_title(f"communities - {lab} [nm]")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center",
               ncol=min(len(order), 2), fontsize=9, markerscale=4,
               frameon=False)
    out = Path(workdir) / f"{out_prefix}_community_map.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    res = {"png": str(out), "n_communities": int(len(set(com.tolist())))}

    if make_3d_html:
        try:
            import plotly.graph_objects as go
            traces = [go.Scatter3d(
                x=p[com == c, 0], y=p[com == c, 1], z=-p[com == c, 2],
                mode="markers", name=_name(c),
                marker=dict(size=1.6, color=colors[c % len(colors)]))
                for c in order]
            figp = go.Figure(traces)
            figp.update_layout(scene_aspectmode="data",
                               title="CCD communities (apex up)",
                               legend=dict(itemsizing="constant"))
            html = Path(workdir) / f"{out_prefix}_community_map_3d.html"
            figp.write_html(html, include_plotlyjs=True)
            res["html"] = str(html)
        except ImportError:
            res["html"] = None
    return res


_ELEMENT_COLORS = {"Fe": "#b51d14", "Cr": "#4053d3", "Ni": "#00b25d",
                   "O": "#00beff", "H": "#cccccc", "C": "#555555",
                   "N": "#7f2ccb", "Si": "#ddb310", "Pt": "#fb49b0",
                   "W": "#333333", "Ta": "#4053d3", "V": "#00b25d"}


def _label_element(label: str):
    """Element group for a ranged ion label: first metal element, else the
    first element. Placeholder labels (e.g. '27Da') return None."""
    import re as _re
    if _re.match(r"^\d+(\.\d+)?Da$", label):
        return None
    parts = [sym for sym, _ in _re.findall(r"([A-Z][a-z]?)(\d*)", label)
             if sym]
    if not parts:
        return None
    for sym in parts:
        if sym not in ("H", "C", "N", "O"):
            return sym
    return parts[0]


def composition_from_labels(label_counts: dict) -> dict:
    """Elemental composition from ranged ion-label counts, decomposing
    molecular ions (Cr1O1 -> Cr + O) and EXCLUDING placeholder labels
    ('27Da', '29Da' - unidentified mass peaks that corrupt naive tallies)."""
    import re as _re
    el: dict = {}
    excluded, excluded_n, total_n = [], 0, 0
    for label, n in label_counts.items():
        total_n += int(n)
        if _label_element(label) is None:
            excluded.append(label)
            excluded_n += int(n)
            continue
        for sym, cnt in _re.findall(r"([A-Z][a-z]?)(\d*)", label):
            if sym:
                el[sym] = el.get(sym, 0) + int(n) * (int(cnt) if cnt else 1)
    tot = sum(el.values()) or 1
    return {"composition_at_pct": {k: round(v / tot * 100, 2)
                                   for k, v in sorted(el.items(),
                                                      key=lambda kv: -kv[1])},
            "excluded_placeholder_labels": excluded,
            "excluded_ion_fraction_pct": round(
                excluded_n / max(total_n, 1) * 100, 2)}


def _load_ranged_positions(pos_path, rrng_path, max_points=250000):
    """(xyz, element) for ranged ions, subsampled, apex-up convention
    (z_plot = z_max - z so the apex sits at the top of every figure)."""
    import apav
    if str(pos_path).lower().endswith(".apt"):
        roi = apav.load_apt(str(pos_path))
        xyz, mass = roi.xyz, roi.mass
    else:
        arr = np.loadtxt(pos_path, delimiter=",")
        xyz, mass = arr[:, :3], arr[:, 3]
    rng = apav.RangeCollection.from_rrng(str(rrng_path))
    labels = np.full(len(mass), "", dtype=object)
    for rr in rng:
        m = (mass >= rr.lower) & (mass < rr.upper)
        labels[m] = str(rr.ion.hill_formula)
    ranged = labels != ""
    xyz, labels = xyz[ranged], labels[ranged]
    elements = np.array([_label_element(l) or "?" for l in labels])
    keep = elements != "?"
    xyz, elements = xyz[keep], elements[keep]
    if len(xyz) > max_points:
        sel = np.random.RandomState(0).choice(len(xyz), max_points,
                                              replace=False)
        xyz, elements = xyz[sel], elements[sel]
    xyz = xyz.copy()
    xyz[:, 2] = xyz[:, 2].max() - xyz[:, 2]
    return xyz, elements


def visualize_apt_elements(pos_path: str, rrng_path: str, out_prefix: str,
                           workdir: str = ".",
                           max_points: int = 250000) -> dict:
    """Element-colored atom maps of an APT reconstruction - the HERO figure.

    Writes <out_prefix>_elements.png (side view, apex at TOP: one panel per
    element plus a combined overlay) and <out_prefix>_elements_3d.html
    (interactive plotly). Molecular ions group under their metal element;
    placeholder labels are excluded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xyz, elements = _load_ranged_positions(pos_path, rrng_path, max_points)
    counts: dict = {}
    for e in elements:
        counts[e] = counts.get(e, 0) + 1
    els = sorted(counts, key=lambda e: -counts[e])

    ncol = len(els) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(2.6 * ncol, 8), sharey=True)
    for ax, e in zip(axes[:-1], els):
        m = elements == e
        ax.scatter(xyz[m, 0], xyz[m, 2], s=0.5, lw=0,
                   c=_ELEMENT_COLORS.get(e, "#888888"), rasterized=True)
        ax.set_title(f"{e} ({counts[e]:,})", fontsize=11)
        ax.set_aspect(1)
        ax.set_xticks([])
    ax = axes[-1]
    for e in reversed(els):
        m = elements == e
        ax.scatter(xyz[m, 0], xyz[m, 2], s=0.5, lw=0, alpha=0.6,
                   c=_ELEMENT_COLORS.get(e, "#888888"), label=e,
                   rasterized=True)
    ax.set_title("all elements", fontsize=11)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.legend(markerscale=18, fontsize=9, loc="upper right")
    axes[0].set_ylabel("distance below apex [nm] (apex at top)")
    axes[0].invert_yaxis()
    png = wd / f"{out_prefix}_elements.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    out = {"png": str(png), "elements": {e: counts[e] for e in els},
           "n_points_shown": int(len(xyz))}
    try:
        import plotly.graph_objects as go
        sub = np.random.RandomState(1).choice(
            len(xyz), min(len(xyz), 120000), replace=False)
        xs, es = xyz[sub], elements[sub]
        traces = [go.Scatter3d(
            x=xs[es == e, 0], y=xs[es == e, 1], z=-xs[es == e, 2],
            mode="markers", name=e,
            marker=dict(size=1.2, color=_ELEMENT_COLORS.get(e, "#888888")))
            for e in els]
        figp = go.Figure(traces)
        figp.update_layout(scene_aspectmode="data",
                           title="APT reconstruction - elements (apex up)",
                           legend=dict(itemsizing="constant"))
        html = wd / f"{out_prefix}_elements_3d.html"
        figp.write_html(html, include_plotlyjs=True)
        out["html"] = str(html)
    except ImportError:
        out["html"] = None
    return out


def annotate_communities(community_compositions: dict,
                         top_n: int = 2) -> dict:
    """Human-meaningful label per community from its KS signature: most
    enriched ion labels, tagged oxide-associated when metal-oxide molecular
    ions dominate. Use these strings in figure legends and reports."""
    names = {}
    for cid, comp in community_compositions.items():
        enriched = sorted(((i, k) for i, k in comp.items() if k > 0.02),
                          key=lambda kv: -kv[1])[:top_n]
        if not enriched:
            names[str(cid)] = f"community {cid}: bulk-like (no enrichment)"
            continue
        ions = ", ".join(f"{i} (+{k:.2f})" for i, k in enriched)
        oxide = any("O" in i and _label_element(i) not in (None, "O")
                    for i, _ in enriched)
        tag = " - oxide-associated" if oxide else ""
        names[str(cid)] = f"community {cid}: {ions} enriched{tag}"
    return names


_IMP = "from scilink.skills.point_cloud_analysis.apt_ccd.apt_tools import "

TOOL_SPECS = [
    ToolSpec(
        name="visualize_apt_elements",
        description=("Element-colored atom maps of an APT reconstruction - "
                     "the HERO figure for any APT report: per-element side "
                     "views plus combined overlay, apex at top, and an "
                     "interactive 3D html. Molecular ions group under their "
                     "metal element; placeholder labels excluded."),
        parameters={"pos_path": {"type": "string",
                                 "description": ".apt/.pos or x,y,z,Da csv"},
                    "rrng_path": {"type": "string", "description": ".rrng"}},
        required=["pos_path", "rrng_path"],
        import_line=_IMP + "visualize_apt_elements",
        signature=("visualize_apt_elements(pos_path, rrng_path, out_prefix, "
                   "workdir='.', max_points=250000) -> dict"),
        agents=["simulation"],
        when_to_use=("FIRST figure of any APT analysis - render it before "
                     "the CCD map; label it hero_png in your files dict."),
        returns="png (hero), html (interactive 3D), per-element counts",
        example=("viz = visualize_apt_elements('R31.apt', 'ranges.rrng', "
                 "'tip', workdir='out')"),
    ),
    ToolSpec(
        name="visualize_communities",
        description=("Annotated CCD segregation map: three-view projections "
                     "+ interactive 3D html, apex at top, legend carrying "
                     "community meanings (detect_segregation already calls "
                     "this automatically; call directly only to re-render)."),
        parameters={"community_xyz": {"type": "string",
                                      "description": "from "
                                                     "detect_segregation"}},
        required=["community_xyz"],
        import_line=_IMP + "visualize_communities",
        signature=("visualize_communities(community_xyz, out_prefix, "
                   "workdir='ccd_analysis', community_meanings=None, "
                   "apex_up=True, make_3d_html=True) -> dict"),
        agents=["simulation"],
        when_to_use="Re-rendering the community map with custom options.",
        returns="png, html (3D), n_communities",
        example=("viz = visualize_communities(seg['community_xyz'], 'tip', "
                 "community_meanings=names)"),
    ),
    ToolSpec(
        name="annotate_communities",
        description=("Human-meaningful legend text per CCD community from "
                     "its KS signature (most-enriched ions, oxide-associated "
                     "tag). Feed to visualize_communities."),
        parameters={"community_compositions": {
            "type": "object", "description": "from detect_segregation"}},
        required=["community_compositions"],
        import_line=_IMP + "annotate_communities",
        signature="annotate_communities(community_compositions) -> dict",
        agents=["simulation"],
        when_to_use=("Always, after detect_segregation - community numbers "
                     "alone are meaningless in figures and reports."),
        returns="{community_id: descriptive label}",
        example="names = annotate_communities(seg['community_compositions'])",
    ),
    ToolSpec(
        name="composition_from_labels",
        description=("Elemental composition from ranged ion-label counts: "
                     "decomposes molecular ions (Cr1O1 -> Cr+O) and excludes "
                     "placeholder labels (27Da etc.) that corrupt naive "
                     "tallies."),
        parameters={"label_counts": {"type": "object",
                                     "description": "e.g. ion_type_counts "
                                                    "from a neighborhoods_* "
                                                    "call"}},
        required=["label_counts"],
        import_line=_IMP + "composition_from_labels",
        signature="composition_from_labels(label_counts) -> dict",
        agents=["simulation"],
        when_to_use=("Whenever reporting elemental composition from APT "
                     "data - never tally raw ion labels directly."),
        returns=("composition_at_pct, excluded_placeholder_labels, "
                 "excluded_ion_fraction_pct"),
        example="comp = composition_from_labels(nb['ion_type_counts'])",
    ),
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
                 "community_compositions as {community_id: {ion_label: "
                 "signed KS}} (positive = enriched vs bulk), "
                 "community_map_png, community_xyz point cloud"),
        example="seg = detect_segregation(nb['neighborhood_csv'])",
    ),
]
