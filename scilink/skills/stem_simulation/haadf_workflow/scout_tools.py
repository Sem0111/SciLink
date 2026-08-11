"""Cheap structure-side scouting tools for point-cloud planning.

These run in seconds on CPU and give a planning agent the evidence to choose
a beam orientation and region of interest BEFORE committing to an expensive
image simulation: overall shape (bulk slab vs finite tip), which axis gives
a well-ordered projected column net (the zone-axis check), and where along
the cell planar features (boundaries, defect bands) sit.

Feature localization reuses the crystalline_deformation classification on
projected column centroids: a coherent twin that is invisible to positional
disorder still lights up as an HCP row / Center-of-Symmetry line.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from scilink.skills._shared._spec import ToolSpec

_PERM = {"x": (1, 2, 0), "y": (2, 0, 1), "z": (0, 1, 2)}


def profile_pointcloud(atoms, bins: int = 16) -> dict:
    """Shape/species overview: extents, NN distance, and a width-vs-position
    profile along the longest axis (a monotonically tapering width profile is
    the signature of a tip/needle; flat means bulk slab)."""
    p = atoms.get_positions()
    n = len(p)
    idx = np.random.RandomState(0).choice(n, min(50000, n), replace=False)
    d, _ = cKDTree(p[idx]).query(p[idx], k=2)
    ext = p.max(axis=0) - p.min(axis=0)
    long_ax = int(np.argmax(ext))
    others = [a for a in range(3) if a != long_ax]
    edges = np.linspace(p[:, long_ax].min(), p[:, long_ax].max(), bins + 1)
    widths = []
    for i in range(bins):
        m = (p[:, long_ax] >= edges[i]) & (p[:, long_ax] < edges[i + 1])
        if m.sum() > 50:
            w = max(np.percentile(p[m, a], 99) - np.percentile(p[m, a], 1)
                    for a in others)
            widths.append({"center_A": round(float(edges[i] + edges[i + 1]) / 2, 1),
                           "width_A": round(float(w), 1),
                           "atoms": int(m.sum())})
    wvals = [w["width_A"] for w in widths]
    tapering = bool(wvals and min(wvals) < 0.55 * max(wvals))
    return {"n_atoms": int(n),
            "species": sorted(set(atoms.get_chemical_symbols())),
            "extent_A": [round(float(v), 1) for v in ext],
            "nn_distance_A": round(float(np.median(d[:, 1])), 2),
            "long_axis": "xyz"[long_ax],
            "width_profile": widths,
            "tapering_object": tapering}


def _project_columns(atoms, beam_axis, cluster_r=0.7, min_atoms=5):
    p = atoms.get_positions()[:, _PERM[beam_axis]]
    xy = p[:, :2]
    tree = cKDTree(xy)
    pairs = tree.query_pairs(cluster_r, output_type="ndarray")
    n = len(xy)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                     shape=(n, n))
    ncomp, labels = connected_components(adj, directed=False)
    counts = np.bincount(labels, minlength=ncomp)
    cents = np.zeros((ncomp, 2))
    for k in (0, 1):
        cents[:, k] = (np.bincount(labels, weights=xy[:, k], minlength=ncomp)
                       / counts)
    return cents[counts >= min_atoms]


def net_quality(atoms, beam_axis: str) -> dict:
    """How well-ordered is the projected column net along ``beam_axis``?

    High ordered_fraction means the axis is at/near a zone axis (good
    imaging direction); low means off-zone smearing. Compare all three axes
    and image along the best one.
    """
    cents = _project_columns(atoms, beam_axis)
    if len(cents) < 20:
        return {"beam_axis": beam_axis, "n_columns": int(len(cents)),
                "ordered_fraction": 0.0, "nn_A": None}
    tree = cKDTree(cents)
    d, _ = tree.query(cents, k=2)
    nn = float(np.median(d[:, 1]))
    lo, hi = 0.75 * nn, 1.4 * nn
    n_lo = tree.query_ball_point(cents, lo, return_length=True)
    n_hi = tree.query_ball_point(cents, hi, return_length=True)
    shell = n_hi - n_lo
    ordered = (n_lo == 1) & (shell >= 5) & (shell <= 7)
    return {"beam_axis": beam_axis, "n_columns": int(len(cents)),
            "ordered_fraction": round(float(np.mean(ordered)), 3),
            "nn_A": round(nn, 2)}


def feature_profile(atoms, beam_axis: str, bins: int = 24) -> dict:
    """Locate planar features (boundaries, faults, defect bands) along the
    in-plane vertical axis of the ``beam_axis`` projection.

    Classifies projected column centroids with the crystalline_deformation
    2D-PTM + Center-of-Symmetry tools and profiles both against position.
    A coherent twin appears as a narrow HCP / elevated-CoS band even though
    positional disorder is zero there. Returns per-bin statistics in the
    ORIGINAL simulation frame coordinate of the profiled axis.
    """
    from scilink.skills.image_analysis.crystalline_deformation.ptm_tools import (
        compute_cos, ptm_classify)

    cents = _project_columns(atoms, beam_axis)
    ptm = ptm_classify(cents[:, 0], cents[:, 1])
    labels = np.asarray(ptm["labels"])
    cos_res = compute_cos(cents[:, 0], cents[:, 1])
    cos = np.asarray(cos_res["cos"] if isinstance(cos_res, dict) else cos_res,
                     dtype=float)
    y = cents[:, 1]
    # map projected vertical coordinate back to the original-frame axis name
    profiled_axis = "xyz"[_PERM[beam_axis][1]]
    offset = atoms.get_positions()[:, _PERM[beam_axis]][:, 1].min()
    edges = np.linspace(y.min(), y.max(), bins + 1)
    prof = []
    for i in range(bins):
        m = (y >= edges[i]) & (y < edges[i + 1])
        if m.sum() < 5:
            continue
        hcp = float(np.mean(labels[m] == "HCP"))
        unid = float(np.mean(labels[m] == "unidentified"))
        c = cos[m]
        prof.append({"pos_A": round(float((edges[i] + edges[i + 1]) / 2), 1),
                     "hcp_frac": round(hcp, 3),
                     "unidentified_frac": round(unid, 3),
                     "mean_cos": round(float(np.nanmean(c)), 4),
                     "n_columns": int(m.sum())})
    return {"beam_axis": beam_axis, "profiled_axis": profiled_axis,
            "note": ("pos_A is in the projected frame; add the original-frame "
                     "minimum of the profiled axis "
                     f"({round(float(offset), 1)} A was subtracted implicitly "
                     "by projection only if positions were shifted - here "
                     "pos_A IS the original-frame coordinate)"),
            "profile": prof}


_ELEMENTS = None


def _element_set():
    global _ELEMENTS
    if _ELEMENTS is None:
        from ase.data import chemical_symbols
        _ELEMENTS = set(chemical_symbols[1:])
    return _ELEMENTS


def classify_cloud_kind(structure_path: str) -> dict:
    """Cheap pre-load classification of a point-cloud file's KIND.

    Distinguishes, from headers and species tokens alone (no full load):
      - simulated_lattice: LAMMPS data file, or xyz/extxyz with element
        species (an extxyz Lattice record is a strong simulated tell)
      - apt_ranged: xyz/csv whose species tokens include molecular ions
        (e.g. FeO, Cr2) - an APT-only signature - or pos/apt with a
        sidecar .rrng
      - apt_unranged: numeric species column (mass-to-charge), or
        pos/epos/apt with no range file - ranging required before analysis
      - crystal_definition: CIF (build a supercell before atomistic tools)
    Returns kind, confidence, and the evidence used; downstream planning
    should trust the evidence over the label when they disagree.
    """
    from pathlib import Path

    path = Path(structure_path)
    ext = path.suffix.lower()
    ev: dict = {"extension": ext or "(none)"}
    rrng = [p.name for p in path.parent.glob("*.rrng")] +            [p.name for p in path.parent.glob("*.RRNG")]
    ev["range_files_in_dir"] = rrng

    def _out(kind, conf, reason):
        return {"kind": kind, "confidence": conf, "reason": reason,
                "evidence": ev}

    if ext == ".cif":
        return _out("crystal_definition", "high", "CIF crystal definition")
    if ext in (".pos", ".epos", ".apt"):
        if rrng:
            return _out("apt_ranged", "high",
                        "APT reconstruction with range file present")
        return _out("apt_unranged", "high",
                    "APT reconstruction, no .rrng in directory - ranging "
                    "required before species-resolved analysis")

    head = ""
    try:
        head = path.read_text(errors="ignore")[:8192]
    except OSError as exc:
        return _out("unknown", "low", f"unreadable: {exc}")

    low = head.lower()
    if " atoms" in low and " atom types" in low:
        return _out("simulated_lattice", "high", "LAMMPS data file header")

    lines = [l for l in head.splitlines() if l.strip()]
    if ext in (".xyz", ".extxyz") and len(lines) >= 3:
        ev["xyz_count_header"] = lines[0].strip().isdigit()
        ev["extxyz_lattice_record"] = "lattice=" in lines[1].lower()
        first_atom = lines[2].split()
        tok = first_atom[0] if first_atom else ""
        try:
            float(tok)
            numeric_species = True
        except ValueError:
            numeric_species = False
        ev["first_species_token"] = tok
        if numeric_species:
            return _out("apt_unranged", "medium",
                        "numeric species column (mass-to-charge?) - provide "
                        "a .rrng and range first (apt_ccd tools)")
        known = tok in _element_set()
        molecular = (not known) and any(c.isdigit() for c in tok) or                     (not known and len(tok) > 2)
        ev["species_is_element"] = known
        if molecular:
            return _out("apt_ranged", "medium",
                        f"species token {tok!r} looks like a molecular ion "
                        "(APT-only signature)")
        if ev["extxyz_lattice_record"]:
            return _out("simulated_lattice", "high",
                        "extxyz with Lattice record")
        return _out("simulated_lattice", "medium",
                    "element species; simulated or ranged-APT - use density/"
                    "NN statistics downstream to confirm (full-density sharp "
                    "lattice = simulated; ~40-80% density = APT)")

    if ext == ".csv" and lines:
        cols = lines[0].replace(";", ",").split(",")
        ncols = len(cols)
        ev["csv_columns"] = ncols
        if ncols == 4:
            return _out("apt_unranged", "medium",
                        "4-column csv (x,y,z,Da?) - the apt_ccd route with a "
                        ".rrng handles this directly")
        if ncols == 3:
            return _out("unknown", "low",
                        "3-column csv: coordinates without species - "
                        "species information is required for analysis")

    return _out("unknown", "low", "no recognized point-cloud signature")


TOOL_SPECS = [
    ToolSpec(
        name="classify_cloud_kind",
        description=("Cheap pre-load classification of a point-cloud file: "
                     "simulated_lattice / apt_ranged / apt_unranged / "
                     "crystal_definition, with the header/species evidence "
                     "used. Decides which tool families are legal."),
        parameters={"structure_path": {"type": "string",
                                       "description": "path to the file"}},
        required=["structure_path"],
        import_line=("from scilink.skills.stem_simulation.haadf_workflow"
                     ".scout_tools import classify_cloud_kind"),
        signature="classify_cloud_kind(structure_path) -> dict",
        agents=["simulation"],
        when_to_use=("FIRST scouting step on any point cloud: the kind "
                     "gates lattice-resolved vs statistical tools and "
                     "flags unranged APT data needing a .rrng."),
        returns="kind, confidence, reason, evidence dict",
        example="kind = classify_cloud_kind('tip.xyz')",
    ),
    ToolSpec(
        name="profile_pointcloud",
        description=("Cheap shape/species overview of a point cloud: extents, "
                     "NN distance, width-vs-position profile, tip detection."),
        parameters={},
        required=[],
        import_line=("from scilink.skills.stem_simulation.haadf_workflow"
                     ".scout_tools import profile_pointcloud"),
        signature="profile_pointcloud(atoms, bins=16) -> dict",
        agents=["simulation"],
        when_to_use=("First scouting step on any structure: decides bulk-slab "
                     "vs finite-tip handling and gives the size context for "
                     "ROI planning."),
        returns="dict with extents, nn distance, width profile, tapering flag",
        example="info = profile_pointcloud(atoms)",
    ),
    ToolSpec(
        name="net_quality",
        description=("Ordered-fraction of the projected column net along a "
                     "candidate beam axis - the data-driven zone-axis check."),
        parameters={"beam_axis": {"type": "string",
                                  "description": "x, y or z"}},
        required=["beam_axis"],
        import_line=("from scilink.skills.stem_simulation.haadf_workflow"
                     ".scout_tools import net_quality"),
        signature="net_quality(atoms, beam_axis) -> dict",
        agents=["simulation"],
        when_to_use=("Compare all three axes; image along the one with the "
                     "highest ordered_fraction (that is the zone-axis view)."),
        returns="dict with n_columns, ordered_fraction, nn_A",
        example="best = max((net_quality(a, ax) for ax in 'xyz'), key=lambda r: r['ordered_fraction'])",
    ),
    ToolSpec(
        name="feature_profile",
        description=("Locate planar features (boundaries, faults) along the "
                     "vertical in-plane axis of a projection: per-bin HCP "
                     "fraction, unidentified fraction and Center-of-Symmetry "
                     "from 2D-PTM on projected columns."),
        parameters={"beam_axis": {"type": "string",
                                  "description": "x, y or z (use the "
                                                 "net_quality winner)"}},
        required=["beam_axis"],
        import_line=("from scilink.skills.stem_simulation.haadf_workflow"
                     ".scout_tools import feature_profile"),
        signature="feature_profile(atoms, beam_axis, bins=24) -> dict",
        agents=["simulation"],
        when_to_use=("To find WHERE a boundary/defect band sits so the ROI "
                     "can be centered on it. A coherent twin shows as a "
                     "narrow HCP/CoS band despite zero positional disorder."),
        returns="per-bin profile of hcp_frac / unidentified_frac / mean_cos",
        example="prof = feature_profile(atoms, best['beam_axis'])",
    ),
]
