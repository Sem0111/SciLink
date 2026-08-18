"""Family-1 cluster / short-range-order tools for APT point clouds.

Clean-room maximum-separation method (MSM) with posgen-canonical semantics
(d_max linking, N_min floor, optional envelope + erosion), blind parameter
selection (k-NN knee -> d_max sweep with stability plateau -> null-informed
N_min), label-shuffle null models as the accept gates, k-NN / RDF statistics
against the shuffled-null band, Warren-Cowley parameters with shuffle
control (full-density clouds only), and per-pair z-SDMs as the
crystallographic-signal gate.

Written against ``cluster_analysis.md`` (the spec). MSM semantics follow the
community canon (Hyde & English, MRS Proc. 2000; Vaumousse, Cerezo & Warren,
Ultramicroscopy 95 (2003); posgen documentation) - the implementation is
clean-room Python; no GPL source was consulted.

Coordinates: nm throughout (APT convention); TRUE reconstruction frame for
analysis, apex-up flip applied only inside figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scilink.skills._shared._spec import ToolSpec

_COLORS = ["#4053d3", "#b51d14", "#ddb310", "#00b25d", "#7f2ccb",
           "#fb49b0", "#00beff", "#cacaca"]


def _jsonable(obj):
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
    return obj


def _load_ranged_cloud(pos_path, rrng_path):
    """(xyz_nm, ion_species_label) for ALL ranged ions in the TRUE
    reconstruction frame - no apex flip, no subsampling (analysis-side
    counterpart of the display loader in apt_ccd.apt_tools)."""
    import apav

    p = str(pos_path).lower()
    if p.endswith(".apt"):
        roi = apav.load_apt(str(pos_path))
        xyz, mass = roi.xyz, roi.mass
    elif p.endswith(".pos"):
        # native read (big-endian float32 x,y,z,Da records) - apav's pos
        # reader still uses the numpy-1 newbyteorder API
        arr = np.fromfile(str(pos_path), dtype=">f4").reshape(-1, 4)
        xyz, mass = arr[:, :3].astype(np.float64), arr[:, 3].astype(np.float64)
    elif p.endswith(".epos"):
        roi = apav.load_epos(str(pos_path))
        xyz, mass = roi.xyz, roi.mass
    else:  # (x, y, z [nm], Da) csv - the synthetic-benchmark contract
        arr = np.loadtxt(pos_path, delimiter=",")
        xyz, mass = arr[:, :3], arr[:, 3]
    rng = apav.RangeCollection.from_rrng(str(rrng_path))
    labels = np.full(len(mass), "", dtype=object)
    for rr in rng:
        m = (mass >= rr.lower) & (mass < rr.upper)
        labels[m] = str(rr.ion.hill_formula)
    ranged = labels != ""
    return np.asarray(xyz)[ranged], labels[ranged].astype(str)


def _target_mask(labels, species):
    """Boolean mask for the target species: a list of ion labels or a
    single regex over labels (grouping whole ions - never decomposition)."""
    from scilink.skills.point_cloud_analysis.apt_ccd.apt_tools import (
        resolve_species_group)

    if isinstance(species, str):
        members = resolve_species_group(np.unique(labels), species)
    else:
        members = [str(s) for s in species]
    if not members:
        raise ValueError(f"no ion labels match species={species!r}; "
                         f"present: {sorted(set(labels.tolist()))}")
    return np.isin(labels, members), members


def _components(xyz_t, d_max, n_min):
    """Connected components of target ions linked within d_max; labels
    -1 = unclustered, else 0..K-1 ordered by decreasing size."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(xyz_t)
    out = np.full(n, -1, dtype=int)
    if n == 0:
        return out, 0
    pairs = cKDTree(xyz_t).query_pairs(d_max, output_type="ndarray")
    if len(pairs) == 0:
        return out, 0
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                   shape=(n, n))
    _, lab = connected_components(g, directed=False)
    counts = np.bincount(lab)
    kept = np.where(counts >= n_min)[0]
    kept = kept[np.argsort(-counts[kept])]
    for new, old in enumerate(kept):
        out[lab == old] = new
    return out, len(kept)


def _hull_volume_nm3(xyz):
    from scipy.spatial import ConvexHull
    try:
        return float(ConvexHull(xyz).volume)
    except Exception:  # noqa: BLE001 - degenerate cloud
        span = np.ptp(xyz, axis=0)
        return float(np.prod(np.maximum(span, 1e-3)))


def _null_masks(n_all, n_target, n_shuffles, seed=0):
    """Label-shuffle null: same positions, species labels randomly
    permuted - the target set becomes a random same-size subset."""
    rng = np.random.RandomState(seed)
    for _ in range(n_shuffles):
        idx = rng.choice(n_all, n_target, replace=False)
        m = np.zeros(n_all, dtype=bool)
        m[idx] = True
        yield m


def knn_distance_stats(pos_path: str, rrng_path: str, species,
                       k: int = 10, n_shuffles: int = 5,
                       out_prefix: str = "knn", workdir: str = ".") -> dict:
    """k-th nearest-neighbour distance distribution of the target species
    vs its label-shuffle null - the MSM parameter-selection evidence.

    Clustered data is bimodal (in-cluster mode below the random mode); the
    valley between them brackets d_max. Returns the detected modes, the
    valley, a suggested d_max sweep window, and a separation verdict.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    from scipy.stats import gaussian_kde

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
    tmask, members = _target_mask(labels, species)
    xt = xyz[tmask]
    if len(xt) < k + 1:
        return {"error": f"only {len(xt)} target ions (< k+1)"}
    d_obs = cKDTree(xt).query(xt, k=k + 1)[0][:, k]
    d_null = np.concatenate([
        cKDTree(xyz[m]).query(xyz[m], k=k + 1)[0][:, k]
        for m in _null_masks(len(xyz), len(xt), n_shuffles)])

    grid = np.linspace(0, np.percentile(d_null, 99.5) * 1.5, 400)
    kde_o = gaussian_kde(d_obs)(grid)
    kde_n = gaussian_kde(d_null)(grid)
    null_mode = float(grid[np.argmax(kde_n)])
    # first observed mode below the null mode = in-cluster population
    below = grid < null_mode
    local_max = (kde_o[1:-1] > kde_o[:-2]) & (kde_o[1:-1] > kde_o[2:])
    cand = np.where(local_max & below[1:-1]
                    & (kde_o[1:-1] > 0.05 * kde_o.max()))[0] + 1
    bimodal = len(cand) > 0 and grid[cand[0]] < 0.8 * null_mode
    res = {"k": k, "species": members, "n_target_ions": int(len(xt)),
           "null_mode_nm": round(null_mode, 3),
           "observed_median_nm": round(float(np.median(d_obs)), 3),
           "bimodal_signal": bool(bimodal)}
    if bimodal:
        m1 = int(cand[0])
        seg = slice(m1, int(np.argmax(kde_n)))
        valley = float(grid[seg][np.argmin(kde_o[seg])])
        res["cluster_mode_nm"] = round(float(grid[m1]), 3)
        res["valley_nm"] = round(valley, 3)
        # sweep from the in-cluster mode UP TO the valley: beyond it the
        # random matrix percolates into giant components
        res["dmax_window_nm"] = [round(float(grid[m1]), 3),
                                 round(valley, 3)]
    else:
        res["note"] = ("no separated in-cluster mode - either no discrete "
                       "clustering at this k, or clusters too dilute; MSM "
                       "sweep may still be run on an explicit window")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(d_obs, bins=80, density=True, alpha=0.5, color=_COLORS[0],
            label=f"observed {k}-NN ({'+'.join(members[:3])})")
    ax.hist(d_null, bins=80, density=True, alpha=0.4, color="#888888",
            label=f"label-shuffle null (x{n_shuffles})")
    if bimodal:
        ax.axvline(res["valley_nm"], color=_COLORS[1], ls="--",
                   label=f"valley {res['valley_nm']} nm")
        ax.axvspan(*res["dmax_window_nm"], color=_COLORS[2], alpha=0.2,
                   label="d_max sweep window")
    ax.set_xlabel(f"{k}-th NN distance [nm]")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title(f"{k}-NN distance: observed vs shuffled null")
    png = wd / f"{out_prefix}_knn{k}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    res["png"] = str(png)
    return _jsonable(res)


def msm_parameter_sweep(pos_path: str, rrng_path: str, species,
                        n_min: int = 10, d_max_window_nm=None,
                        n_steps: int = 16, n_shuffles: int = 3,
                        out_prefix: str = "msm_sweep",
                        workdir: str = ".", _refined: bool = False) -> dict:
    """MSM d_max sweep with stability-plateau detection and null-informed
    N_min - the BLIND parameter-selection tool (claims come from the
    plateau, never a cherry-picked point).

    Window defaults to the k-NN-knee window (k = n_min). Also reruns the
    sweep on shuffled labels (null band) and, at the recommended d_max,
    recommends N_min = (largest null cluster) + 1.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
    tmask, members = _target_mask(labels, species)
    xt = xyz[tmask]

    res = {"species": members, "n_min_sweep": n_min}
    if d_max_window_nm is None:
        knee = knn_distance_stats(pos_path, rrng_path, species, k=n_min,
                                  n_shuffles=n_shuffles,
                                  out_prefix=out_prefix, workdir=workdir)
        res["knee_evidence"] = knee
        if "dmax_window_nm" not in knee:
            res["error"] = ("no bimodal k-NN signal and no explicit window "
                            "- nothing defensible to sweep")
            return _jsonable(res)
        d_max_window_nm = knee["dmax_window_nm"]
    lo, hi = float(d_max_window_nm[0]), float(d_max_window_nm[1])
    grid = np.linspace(lo, hi, n_steps)

    n_obs = np.array([_components(xt, d, n_min)[1] for d in grid])
    nulls = np.zeros((n_shuffles, n_steps), dtype=int)
    for si, m in enumerate(_null_masks(len(xyz), len(xt), n_shuffles)):
        nulls[si] = [_components(xyz[m], d, n_min)[1] for d in grid]

    # plateau: longest run of consecutive steps with a stable count
    # (<= max(1, 5%) change) on steps where the null is QUIET (percolating
    # d_max flattens the observed curve too - the null band exposes it)
    tol = np.maximum(1, 0.05 * n_obs[:-1])
    clean = np.mean(nulls, 0) <= np.maximum(0.25 * n_obs, 0.5)
    stable = (np.abs(np.diff(n_obs)) <= tol) & (n_obs[:-1] >= 1) \
        & clean[:-1] & clean[1:]
    best_len, best_start, run, start = 0, -1, 0, 0
    for i, s in enumerate(stable):
        run, start = (run + 1, start) if s else (0, i + 1)
        if run > best_len:
            best_len, best_start = run, start
    plateau_found = best_len >= 2  # >= 3 consecutive stable, null-quiet pts
    res.update({"d_max_grid_nm": [round(float(d), 3) for d in grid],
                "n_clusters_observed": n_obs.tolist(),
                "n_clusters_null_mean": np.mean(nulls, 0).round(2).tolist(),
                "n_clusters_null_max": np.max(nulls, 0).tolist(),
                "plateau_found": bool(plateau_found)})
    if plateau_found:
        seg = slice(best_start, best_start + best_len + 1)
        d_rec = float(np.mean(grid[seg]))
        res.update({"plateau_range_nm": [round(float(grid[seg][0]), 3),
                                         round(float(grid[seg][-1]), 3)],
                    "n_clusters_at_plateau": int(np.round(
                        np.mean(n_obs[seg]))),
                    "d_max_recommended_nm": round(d_rec, 3)})
    else:
        d_rec = float(grid[np.argmin(np.abs(grid - np.mean(grid)))])
        res.update({"d_max_recommended_nm": round(d_rec, 3),
                    "note": "NO stability plateau - treat any cluster "
                            "claim from this data/species with caution"})
    # null-informed N_min at the recommended d_max
    null_sizes = []
    for m in _null_masks(len(xyz), len(xt), max(n_shuffles, 3), seed=1):
        cl, _ = _components(xyz[m], d_rec, 1)
        if (cl >= 0).any():
            null_sizes.append(int(np.bincount(cl[cl >= 0]).max()))
        else:
            null_sizes.append(0)
    res["largest_null_cluster"] = int(max(null_sizes))
    res["n_min_recommended"] = int(max(max(null_sizes) + 1, 3))
    if max(null_sizes) > 0.05 * len(xt):
        res["percolation_warning"] = (
            "largest null cluster spans >5% of target ions at the "
            "recommended d_max - the matrix percolates; shrink d_max")
    # AUTO-REFINEMENT (one level): a null-informed N_min far above the
    # sweep floor means the random matrix percolates at the recommended
    # d_max (high-solute-density materials, e.g. ODS alloys) - the true
    # plateau lives at smaller d. Re-sweep the lower half of the window;
    # the stage-1 result is kept under "stage1" for the record.
    if (not _refined
            and (max(null_sizes) > 10 * n_min
                 or max(null_sizes) > 0.02 * len(xt))):
        res["percolation_refinement"] = (
            f"largest null cluster {max(null_sizes)} >> N_min floor "
            f"{n_min} - percolation at d={d_rec:.2f} nm; re-swept "
            f"[{lo:.3f}, {lo + 0.5 * (hi - lo):.3f}] nm")
        refined = msm_parameter_sweep(
            pos_path, rrng_path, species, n_min=n_min,
            d_max_window_nm=[lo, lo + 0.5 * (hi - lo)],
            n_steps=n_steps, n_shuffles=n_shuffles,
            out_prefix=out_prefix + "_refined", workdir=workdir,
            _refined=True)
        refined["stage1"] = {k: res.get(k) for k in
                             ("d_max_recommended_nm", "n_min_recommended",
                              "largest_null_cluster", "plateau_found",
                              "percolation_refinement", "png")}
        return refined

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, n_obs, "-o", ms=4, color=_COLORS[0], label="observed")
    ax.fill_between(grid, np.min(nulls, 0), np.max(nulls, 0),
                    color="#888888", alpha=0.35,
                    label=f"label-shuffle null (x{n_shuffles})")
    if plateau_found:
        ax.axvspan(*res["plateau_range_nm"], color=_COLORS[2], alpha=0.25,
                   label="stability plateau")
    ax.axvline(d_rec, color=_COLORS[1], ls="--",
               label=f"d_max = {d_rec:.2f} nm")
    ax.set_xlabel("d_max [nm]")
    ax.set_ylabel(f"clusters (>= {n_min} ions)")
    ax.legend(fontsize=8)
    ax.set_title("MSM parameter sweep: cluster count vs d_max")
    png = wd / f"{out_prefix}_sweep.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    res["png"] = str(png)
    return _jsonable(res)


def msm_detect(pos_path: str, rrng_path: str, species,
               d_max_nm: float, n_min: int,
               envelope_nm: float | None = None,
               erosion_nm: float | None = None,
               out_prefix: str = "msm", workdir: str = ".",
               make_3d_html: bool = True) -> dict:
    """Maximum-separation cluster detection at FIXED parameters (choose
    them with msm_parameter_sweep, then gate with label_shuffle_null).

    posgen-canonical steps: link target ions within d_max -> connected
    components >= n_min; envelope (default d_max/2) sweeps every ion within
    reach of a core ion into its cluster for composition; erosion (default
    = envelope) peels cluster ions within erosion_nm of the exterior - the
    matrix-skin correction without which cluster compositions read far too
    matrix-rich. Returns per-cluster statistics, number density, matrix
    composition, and cluster-map figures (apex-up).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
    tmask, members = _target_mask(labels, species)
    xt = xyz[tmask]
    core, n_clusters = _components(xt, d_max_nm, n_min)

    # cluster id per ion over the FULL cloud (-1 = matrix)
    cid = np.full(len(xyz), -1, dtype=int)
    tidx = np.where(tmask)[0]
    cid[tidx[core >= 0]] = core[core >= 0]
    if envelope_nm is None:
        envelope_nm = d_max_nm / 2
    if erosion_nm is None:
        erosion_nm = envelope_nm
    if n_clusters and envelope_nm:
        core_xyz = xt[core >= 0]
        core_cid = core[core >= 0]
        tree = cKDTree(core_xyz)
        dist, nn = tree.query(xyz, k=1,
                              distance_upper_bound=float(envelope_nm))
        swept = np.isfinite(dist) & (cid == -1)
        cid[swept] = core_cid[nn[swept]]
        if erosion_nm:
            ext = cKDTree(xyz[cid == -1])
            d_ext = ext.query(xyz[cid >= 0], k=1,
                              distance_upper_bound=float(erosion_nm))[0]
            inside = np.where(cid >= 0)[0]
            cid[inside[np.isfinite(d_ext)]] = -1

    clusters = []
    for c in range(n_clusters):
        m = cid == c
        p = xyz[m]
        cen = p.mean(axis=0)
        rg = float(np.sqrt(np.mean(np.sum((p - cen) ** 2, axis=1))))
        lab_c, cnt_c = np.unique(labels[m], return_counts=True)
        comp = {str(l): round(float(n) / len(p), 4)
                for l, n in sorted(zip(lab_c, cnt_c), key=lambda t: -t[1])}
        label_counts = {str(l): int(n) for l, n in zip(lab_c, cnt_c)}
        n_t = int(np.isin(labels[m], members).sum())
        clusters.append({
            "id": c, "n_ions": int(m.sum()), "n_target_ions": n_t,
            "center_nm": [round(float(v), 2) for v in cen],
            "radius_gyration_nm": round(rg, 3),
            "guinier_radius_nm": round(rg * np.sqrt(5.0 / 3.0), 3),
            "target_fraction": round(n_t / m.sum(), 3),
            "composition_ionic": comp,
            "label_counts": label_counts})
    vol = _hull_volume_nm3(xyz)
    matrix = cid == -1
    lab_m, cnt_m = np.unique(labels[matrix], return_counts=True)
    tot_m = int(matrix.sum()) or 1
    res = {"species": members, "d_max_nm": d_max_nm, "n_min": n_min,
           "envelope_nm": envelope_nm, "erosion_nm": erosion_nm,
           "n_clusters": n_clusters,
           "analyzed_volume_nm3": round(vol, 1),
           "number_density_per_nm3": round(n_clusters / vol, 8),
           "number_density_per_m3": float(n_clusters / vol * 1e27),
           "clusters": clusters,
           "matrix_composition_ionic_pct": {
               str(l): round(float(n) / tot_m * 100, 3)
               for l, n in sorted(zip(lab_m, cnt_m), key=lambda t: -t[1])},
           "clustered_target_fraction": round(
               float((cid[tidx] >= 0).sum()) / max(len(tidx), 1), 4)}

    # figures: apex-up x-z projection + optional 3D html
    plot = xyz.copy()
    plot[:, 2] = plot[:, 2].max() - plot[:, 2]
    fig, ax = plt.subplots(figsize=(6, 9))
    sub = np.random.RandomState(0).choice(
        np.where(matrix)[0], min(int(matrix.sum()), 120000), replace=False)
    ax.scatter(plot[sub, 0], plot[sub, 2], s=0.3, lw=0, c="#cccccc",
               rasterized=True, label="matrix")
    for c in range(n_clusters):
        m = cid == c
        ax.scatter(plot[m, 0], plot[m, 2], s=2.5, lw=0,
                   c=_COLORS[c % 7], rasterized=True)
    ax.set_aspect(1)
    ax.invert_yaxis()
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("distance below apex [nm] (apex at top)")
    ax.set_title(f"MSM clusters ({'+'.join(members[:3])}): "
                 f"N={n_clusters} at d_max={d_max_nm:.2f} nm, "
                 f"N_min={n_min}")
    png = wd / f"{out_prefix}_clusters.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    res["png"] = str(png)
    if make_3d_html and n_clusters:
        try:
            import plotly.graph_objects as go
            msub = np.random.RandomState(1).choice(
                np.where(matrix)[0], min(int(matrix.sum()), 60000),
                replace=False)
            traces = [go.Scatter3d(
                x=xyz[msub, 0], y=xyz[msub, 1], z=-xyz[msub, 2],
                mode="markers", name="matrix",
                marker=dict(size=1, color="#cccccc", opacity=0.25))]
            for c in range(n_clusters):
                m = cid == c
                traces.append(go.Scatter3d(
                    x=xyz[m, 0], y=xyz[m, 1], z=-xyz[m, 2],
                    mode="markers", name=f"cluster {c} "
                    f"({int(m.sum())} ions)",
                    marker=dict(size=2, color=_COLORS[c % 7])))
            figp = go.Figure(traces)
            figp.update_layout(scene_aspectmode="data",
                               title="MSM clusters (apex up)",
                               legend=dict(itemsizing="constant"))
            html = wd / f"{out_prefix}_clusters_3d.html"
            figp.write_html(html, include_plotlyjs=True)
            res["html"] = str(html)
        except ImportError:
            res["html"] = None
    return _jsonable(res)


def label_shuffle_null(pos_path: str, rrng_path: str, species,
                       d_max_nm: float, n_min: int,
                       n_shuffles: int = 5) -> dict:
    """The family-1 accept gate: identical MSM detection on >= 5 label
    shuffles. A real signal must collapse under shuffling; reports
    N_observed vs N_null +/- sigma, contrast, and a deterministic verdict
    (N_obs > null mean + 3 sigma, Poisson-floored)."""
    xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
    tmask, members = _target_mask(labels, species)
    n_obs = _components(xyz[tmask], d_max_nm, n_min)[1]
    n_null = [
        _components(xyz[m], d_max_nm, n_min)[1]
        for m in _null_masks(len(xyz), int(tmask.sum()), n_shuffles, seed=2)]
    mean, std = float(np.mean(n_null)), float(np.std(n_null))
    sigma = max(std, np.sqrt(max(mean, 1e-9)), 1e-9)
    contrast = n_obs / mean if mean > 0 else float("inf")
    return _jsonable({
        "species": members, "d_max_nm": d_max_nm, "n_min": n_min,
        "n_observed": int(n_obs), "n_null": n_null,
        "n_null_mean": round(mean, 2), "n_null_std": round(std, 2),
        "z_score": round((n_obs - mean) / sigma, 1),
        "null_contrast": (round(contrast, 1)
                          if np.isfinite(contrast) else "inf"),
        "null_gate_passed": bool(n_obs >= 3 and n_obs > mean + 3 * sigma)})


def rdf_compare(pos_path: str, rrng_path: str, species,
                r_max_nm: float = 6.0, dr_nm: float = 0.05,
                n_shuffles: int = 5, max_centers: int = 50000,
                out_prefix: str = "rdf", workdir: str = ".") -> dict:
    """Target-target radial distribution vs the label-shuffle null.

    Reported as the RATIO g_obs(r)/g_null(r) - dividing by the null (same
    positions, permuted labels) cancels the tip shape and density
    gradients, so ratio > 1 is pure chemical clustering signal. Returns the
    peak excess, its range, and the deviation-from-null verdict.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
    tmask, members = _target_mask(labels, species)
    edges = np.arange(0, r_max_nm + dr_nm, dr_nm)
    rng = np.random.RandomState(3)

    def hist(mask):
        pts = xyz[mask]
        if len(pts) > max_centers:
            pts = pts[rng.choice(len(pts), max_centers, replace=False)]
        d = cKDTree(pts).query_pairs(r_max_nm, output_type="ndarray")
        r = np.linalg.norm(pts[d[:, 0]] - pts[d[:, 1]], axis=1)
        h = np.histogram(r, bins=edges)[0].astype(float)
        npairs = len(pts) * (len(pts) - 1) / 2
        return h / max(npairs, 1)

    h_obs = hist(tmask)
    h_null = np.array([hist(m) for m in _null_masks(
        len(xyz), int(tmask.sum()), n_shuffles, seed=4)])
    nm = h_null.mean(axis=0)
    ns = h_null.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nm > 0, h_obs / nm, np.nan)
        band = np.where(nm > 0, 1 + 3 * ns / nm, np.nan)
    rc = 0.5 * (edges[:-1] + edges[1:])
    valid = np.isfinite(ratio)
    above = valid & (ratio > band)
    imax = int(np.nanargmax(np.where(valid, ratio, -np.inf)))
    # clustering length scale: first r where the excess decays back to null
    r_decay, persists = None, False
    if above.any():
        past_peak = np.where(~above & (np.arange(len(rc)) > imax))[0]
        if len(past_peak):
            r_decay = round(float(rc[past_peak[0]]), 2)
        else:
            persists = True
    res = {"species": members, "r_max_nm": r_max_nm,
           "peak_ratio": round(float(ratio[imax]), 2),
           "peak_r_nm": round(float(rc[imax]), 3),
           "significant_excess": bool(above.any()),
           "excess_range_nm": r_decay,
           "excess_persists_to_r_max": persists,
           "n_target_ions": int(tmask.sum())}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rc[valid], ratio[valid], "-", color=_COLORS[0],
            label="g_obs / g_null")
    ax.plot(rc[valid], band[valid], "--", color="#888888",
            label="null + 3 sigma")
    ax.axhline(1, color="#888888", lw=0.8)
    ax.set_xlabel("r [nm]")
    ax.set_ylabel("pair-correlation ratio")
    ax.set_title(f"{'+'.join(members[:3])} RDF vs label-shuffle null")
    ax.legend(fontsize=8)
    png = wd / f"{out_prefix}_rdf.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    res["png"] = str(png)
    return _jsonable(res)


def warren_cowley(structure_path: str | None = None,
                  pos_path: str | None = None,
                  rrng_path: str | None = None,
                  center_species: str = "", neighbor_species: str = "",
                  type_map: dict | None = None, n_shells: int = 1,
                  shell_windows_nm: list | None = None,
                  n_shuffles: int = 5, max_centers: int = 50000) -> dict:
    """Warren-Cowley SRO parameter on NN shells, with shuffle control.

    alpha_i = 1 - p_i(B|A) / x_B per shell (negative = B-around-A
    preference / ordering, positive = clustering). VALID quantitatively on
    full-density simulated clouds (structure_path route). On a real APT
    reconstruction (pos_path route) detection loss + positional noise make
    geometric first shells unreliable - the result then carries a
    validity_warning and must not be the primary evidence (spec rule).
    Shell windows default to minima of the all-atom NN distance histogram.
    """
    from scipy.spatial import cKDTree

    if structure_path:
        from scilink.skills.stem_simulation.haadf_workflow.abtem_tools import (
            read_structure)
        atoms, _ = read_structure(structure_path, type_map=type_map)
        xyz = atoms.get_positions() / 10.0
        labels = np.array(atoms.get_chemical_symbols(), dtype=str)
        kind = "full-density structure"
    elif pos_path and rrng_path:
        xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
        kind = "APT reconstruction"
    else:
        raise ValueError("give structure_path OR pos_path + rrng_path")
    amask, a_members = _target_mask(labels, center_species)
    bmask, b_members = _target_mask(labels, neighbor_species)
    x_b = float(bmask.mean())

    tree = cKDTree(xyz)
    rng = np.random.RandomState(5)
    centers = np.where(amask)[0]
    if len(centers) > max_centers:
        centers = rng.choice(centers, max_centers, replace=False)

    if shell_windows_nm is None:
        # shell windows from minima of the all-atom NN-distance histogram
        probe = xyz[rng.choice(len(xyz), min(len(xyz), 20000),
                               replace=False)]
        d12 = tree.query(probe, k=13)[0][:, 1:]
        h, e = np.histogram(d12.ravel(), bins=200,
                            range=(0, np.percentile(d12[:, -1], 90)))
        from scipy.ndimage import gaussian_filter1d
        hs = gaussian_filter1d(h.astype(float), 3)
        # shell boundaries = minima AFTER the first NN peak (the flat-zero
        # region below the peak is full of spurious local minima)
        ipk = int(np.argmax(hs))
        mins = [i for i in range(ipk + 1, len(hs) - 3)
                if hs[i] <= hs[i - 1] and hs[i] < hs[i + 1]
                and hs[i] < 0.6 * hs[ipk]]
        cuts = [0.0] + [float(e[i]) for i in mins[:n_shells]]
        if len(cuts) < n_shells + 1:
            return {"error": "could not resolve shell minima - pass "
                             "shell_windows_nm explicitly"}
        shell_windows_nm = [[cuts[i], cuts[i + 1]]
                            for i in range(n_shells)]

    def alphas(lab_b_mask):
        out = []
        r_hi = max(w[1] for w in shell_windows_nm)
        nbrs = tree.query_ball_point(xyz[centers], r_hi)
        for w in shell_windows_nm:
            nb, ntot = 0, 0
            for ci, nn in zip(centers, nbrs):
                nn = np.asarray(nn)
                nn = nn[nn != ci]
                d = np.linalg.norm(xyz[nn] - xyz[ci], axis=1)
                sel = nn[(d >= w[0]) & (d < w[1])]
                ntot += len(sel)
                nb += int(lab_b_mask[sel].sum())
            out.append(1 - (nb / ntot) / x_b if ntot else np.nan)
        return out

    a_obs = alphas(bmask)
    a_null = []
    for _ in range(n_shuffles):
        perm = rng.permutation(len(labels))
        a_null.append(alphas(bmask[perm]))
    a_null = np.array(a_null, dtype=float)
    nm, ns = np.nanmean(a_null, 0), np.nanstd(a_null, 0)
    res = {"center_species": a_members, "neighbor_species": b_members,
           "x_neighbor": round(x_b, 4), "data_kind": kind,
           "shell_windows_nm": [[round(float(a), 3), round(float(b), 3)]
                                for a, b in shell_windows_nm],
           "alpha_per_shell": [round(float(a), 4) for a in a_obs],
           "alpha_null_mean": nm.round(4).tolist(),
           "alpha_null_std": ns.round(4).tolist(),
           "z_per_shell": [round(float((o - m) / max(s, 1e-9)), 1)
                           for o, m, s in zip(a_obs, nm, ns)],
           "significant": [bool(abs(o - m) > 3 * max(s, 1e-9))
                           for o, m, s in zip(a_obs, nm, ns)]}
    if kind == "APT reconstruction":
        res["validity_warning"] = (
            "geometric shell WC on a reconstruction (detection loss + "
            "positional noise) is NOT quantitative - use only as a "
            "secondary indicator; prefer k-NN/RDF null statistics "
            "(cluster_analysis.md rule 1)")
    return _jsonable(res)


def zsdm(pos_path: str, rrng_path: str, species_a, species_b=None,
         dz_max_nm: float = 1.0, r_lateral_nm: float = 0.75,
         bin_nm: float = 0.01, max_centers: int = 20000,
         out_prefix: str = "zsdm", workdir: str = ".") -> dict:
    """Per-pair z-SDM (spatial distribution map along the reconstruction
    axis) - the crystallographic-signal GATE.

    Histograms delta-z for B ions in a lateral cylinder around each A ion.
    Lattice-plane oscillations (detected via the FFT of the detrended
    histogram) mean depth resolution resolves atomic planes; NO oscillation
    means NO site-resolved ordering claim is defensible by any method on
    this data (spec rule 2) - restrict to distance-statistics methods.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial import cKDTree

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
    amask, a_members = _target_mask(labels, species_a)
    bmask, b_members = _target_mask(labels, species_b
                                    if species_b is not None else species_a)
    rng = np.random.RandomState(6)
    centers = np.where(amask)[0]
    if len(centers) > max_centers:
        centers = rng.choice(centers, max_centers, replace=False)
    bxyz = xyz[bmask]
    tree = cKDTree(bxyz)
    r_query = float(np.hypot(r_lateral_nm, dz_max_nm))
    edges = np.arange(-dz_max_nm, dz_max_nm + bin_nm, bin_nm)
    hist = np.zeros(len(edges) - 1)
    for chunk in np.array_split(centers, max(1, len(centers) // 2000)):
        nbrs = tree.query_ball_point(xyz[chunk], r_query)
        for ci, nn in zip(chunk, nbrs):
            if not nn:
                continue
            d = bxyz[np.asarray(nn)] - xyz[ci]
            lat = np.hypot(d[:, 0], d[:, 1])
            dz = d[(lat <= r_lateral_nm)
                   & (np.abs(d[:, 2]) <= dz_max_nm), 2]
            hist += np.histogram(dz, bins=edges)[0]
    # remove the self pair (delta-z = 0 spike) when A and B overlap
    mid = len(hist) // 2
    if set(a_members) & set(b_members):
        hist[mid] -= len(centers)

    bg = gaussian_filter1d(hist, sigma=max(3, int(0.08 / bin_nm)))
    signal = hist - bg
    power = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=bin_nm)
    phys = (freqs > 1 / 0.6) & (freqs < 1 / 0.05)  # plane spacings 0.5-6 A
    peak_contrast = 0.0
    spacing_A = None
    med = float(np.median(power[phys])) if phys.any() else 0.0
    if med > 0:
        contrast = power / med
        ip = int(np.argmax(np.where(phys, power, 0)))
        peak_contrast = float(contrast[ip])
        # spacing from the LOWEST-frequency significant local max: on a
        # sharp plane comb (full-density data) a HARMONIC can out-power
        # the fundamental, and 10/f of a harmonic halves the spacing
        sig = [i for i in np.where(phys)[0]
               if 0 < i < len(power) - 1 and contrast[i] >= 8.0
               and power[i] >= power[i - 1] and power[i] >= power[i + 1]]
        spacing_A = round(10.0 / float(freqs[min(sig) if sig else ip]), 3)
    has_signal = bool(peak_contrast >= 8.0)
    res = {"species_a": a_members, "species_b": b_members,
           "n_centers": int(len(centers)),
           "n_pairs": int(hist.sum()),
           "crystallographic_signal": has_signal,
           "fft_peak_contrast": round(peak_contrast, 1),
           "plane_spacing_A": spacing_A if has_signal else None,
           "gate_verdict": (
               "z-SDM resolves lattice planes - site-resolved ordering "
               "analysis is admissible" if has_signal else
               "NO lattice-plane signal - site-resolved SRO claims are "
               "NOT defensible on this data; use distance statistics only")}

    zc = 0.5 * (edges[:-1] + edges[1:])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(zc * 10, hist, color=_COLORS[0], lw=0.8)
    ax1.plot(zc * 10, bg, color="#888888", ls="--", lw=1)
    ax1.set_xlabel("delta-z [A]")
    ax1.set_ylabel("pair count")
    ax1.set_title(f"z-SDM {'+'.join(a_members[:2])} - "
                  f"{'+'.join(b_members[:2])}")
    ax2.semilogy(freqs[1:], power[1:], color=_COLORS[1], lw=0.8)
    if spacing_A:
        ax2.axvline(10.0 / spacing_A, color=_COLORS[3], ls="--",
                    label=f"{spacing_A} A" + (" (signal)" if has_signal
                                              else " (below threshold)"))
        ax2.legend(fontsize=8)
    ax2.set_xlabel("spatial frequency [1/nm]")
    ax2.set_ylabel("detrended power")
    ax2.set_title(f"plane signal: contrast {peak_contrast:.1f} "
                  f"(threshold 8)")
    png = wd / f"{out_prefix}_zsdm.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    res["png"] = str(png)
    return _jsonable(res)


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _report_css():
    return """
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial,
       sans-serif; margin: 0 auto; max-width: 1080px; padding: 24px;
       color: #1a1a2e; background: #fafafa; }
h1 { font-size: 1.5em; border-bottom: 3px solid #4053d3;
     padding-bottom: 8px; }
h2 { font-size: 1.15em; margin-top: 1.8em; color: #16213e; }
img { max-width: 100%; border: 1px solid #ddd; background: #fff;
      margin: 6px 0; }
table { border-collapse: collapse; margin: 8px 0; font-size: 0.88em; }
th, td { border: 1px solid #ccc; padding: 4px 9px; text-align: left; }
th { background: #eef0fa; }
.pass { color: #00752d; font-weight: 600; }
.fail { color: #b51d14; font-weight: 600; }
.note { background: #fff8e1; border-left: 4px solid #ddb310;
        padding: 8px 12px; margin: 10px 0; }
.links a { margin-right: 18px; }
footer { margin-top: 2.5em; font-size: 0.8em; color: #666;
         border-top: 1px solid #ddd; padding-top: 8px; }
"""


def _strip_for_prompt(obj, drop=("png", "html", "matches")):
    """Deep-copy a results dict without figure paths / bulky arrays so it
    fits an interpretation prompt."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in drop:
                continue
            if k == "clusters" and isinstance(v, list) and len(v) > 12:
                out[k] = _strip_for_prompt(v[:12])
                out["clusters_truncated_note"] = (
                    f"{len(v) - 12} more clusters omitted from this prompt")
            else:
                out[k] = _strip_for_prompt(v)
        return out
    if isinstance(obj, list):
        return [_strip_for_prompt(v) for v in obj]
    return obj


def family1_assessment(run: dict, context: str = "",
                       score: dict | None = None,
                       control: dict | None = None,
                       model: str = "bedrock/us.anthropic.claude-opus-4-8"
                       ) -> dict:
    """LLM scientific interpretation of a family-1 run - the detailed
    text summary for the report. Deterministic gate verdicts are BINDING
    on the narrative: the prompt forbids claiming anything its null model
    rejected, and z-SDM's verdict caps site-resolved language.

    Returns {detailed_analysis, scientific_claims, caveats, model} or
    {error} - report generation must not fail when the LLM is
    unavailable.
    """
    import re as _re

    from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel

    payload = {"results": _strip_for_prompt(run)}
    if score:
        payload["benchmark_score_vs_ground_truth"] = _strip_for_prompt(score)
    if control:
        payload["false_positive_control"] = _strip_for_prompt(control)
    prompt = (
        "You are the scientific-interpretation stage of an APT point-cloud "
        "analysis pipeline. Below are the DETERMINISTIC results of a "
        "family-1 (clustering / short-range order) analysis: maximum-"
        "separation cluster detection with blind parameter selection, "
        "label-shuffle null gates, RDF-vs-null statistics, and a z-SDM "
        "crystallographic-signal gate.\n\n"
        "BINDING RULES: (1) gate verdicts are authoritative - never claim "
        "a signal whose null gate did not pass, and if the z-SDM gate "
        "reports no lattice-plane signal, no site-resolved ordering "
        "language is allowed; (2) compositions are IONIC (ranged ion "
        "species) - do not decompose molecular ions into elements; "
        "(3) be quantitative - cite the numbers in the results.\n\n"
        + (f"CONTEXT:\n{context}\n\n" if context else "")
        + "RESULTS:\n" + json.dumps(payload, indent=1) + "\n\n"
        "Produce the scientific interpretation as a single fenced ```json "
        "block with exactly these fields:\n"
        '{"detailed_analysis": "ONE SINGLE STRING (not a list) of 3-6 '
        "plain-prose paragraphs (separated by \\n\\n, no markdown) "
        "interpreting the analysis quantitatively - what was found, how "
        "the parameters were chosen, what the null models say, what the "
        'data quality permits", '
        '"scientific_claims": [2-4 items, each {"claim": one-sentence '
        'finding, "scientific_impact": why it matters}], '
        '"caveats": "short plain-prose statement of limitations"}')
    try:
        wrapper = LiteLLMGenerativeModel(model)
        resp = wrapper.generate_content(
            prompt, generation_config={"max_output_tokens": 4096})
        part = resp.candidates[0].content.parts[0]
        raw = getattr(part, "raw_text", None) or getattr(part, "text", "")
        def _parse(block):
            # strict -> trailing-comma repair -> first complete object
            # (models emit almost-JSON: ',}' tails, chatter after the block)
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass
            repaired = _re.sub(r",\s*([}\]])", r"\1", block)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return json.JSONDecoder().raw_decode(repaired)[0]

        m = _re.search(r"```json\s*(\{.*?\})\s*```", raw, _re.DOTALL)
        out = _parse(m.group(1) if m else raw[raw.find("{"):])
        out["model"] = model
        return out
    except Exception as exc:  # noqa: BLE001 - assessment is best-effort
        return {"error": f"LLM assessment unavailable: {exc}",
                "model": model}


def family1_report(workdir: str, run: dict,
                   title: str = "Family-1 cluster analysis",
                   hero: dict | None = None, score: dict | None = None,
                   control: dict | None = None, extra_note: str = "",
                   assessment: dict | None = None,
                   out_name: str = "report.html") -> dict:
    """Assemble the standard family-1 HTML report from pipeline results -
    the PIPELINE-DEFAULT report stage (plan rule: report artifacts are
    guaranteed, never optional). Call it at the end of EVERY run.

    ``run`` holds the tool results under keys knee / sweep / detect /
    gate / rdf / zsdm (any subset); ``hero`` a visualize_apt_elements
    result; ``score`` a benchmark scoring dict; ``control`` a
    false-positive-control section. Figures are linked RELATIVE to
    workdir so the report travels with its run folder.
    """
    import os

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)

    def rel(p):
        return _esc(os.path.relpath(str(p), str(wd))) if p else None

    def img(p):
        return f'<img src="{rel(p)}">' if p else ""

    def kv_table(d, keys):
        rows = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(d[k])}</td></tr>"
                       for k in keys if k in d and d[k] is not None)
        return f"<table>{rows}</table>" if rows else ""

    def badge(ok, yes, no):
        return (f'<span class="pass">{yes}</span>' if ok
                else f'<span class="fail">{no}</span>')

    s = [f"<title>{_esc(title)}</title><style>{_report_css()}</style>",
         f"<h1>{_esc(title)}</h1>"]
    if extra_note:
        s.append(f'<div class="note">{extra_note}</div>')

    if assessment:
        s.append("<h2>Scientific assessment</h2>")
        if assessment.get("error"):
            s.append(f'<div class="note">{_esc(assessment["error"])}'
                     "</div>")
        else:
            for para in str(assessment.get("detailed_analysis", "")
                            ).split("\n\n"):
                if para.strip():
                    s.append(f"<p>{_esc(para.strip())}</p>")
            claims = assessment.get("scientific_claims") or []
            if claims:
                s.append("<p><b>Claims:</b></p><ol>" + "".join(
                    f"<li><b>{_esc(c.get('claim', ''))}</b> "
                    f"{_esc(c.get('scientific_impact', ''))}</li>"
                    for c in claims) + "</ol>")
            if assessment.get("caveats"):
                s.append(f'<div class="note"><b>Caveats:</b> '
                         f'{_esc(assessment["caveats"])}</div>')
            s.append(f"<p><i>LLM interpretation "
                     f"({_esc(assessment.get('model', ''))}); the "
                     "deterministic gate verdicts below remain "
                     "authoritative.</i></p>")

    if hero:
        s.append("<h2>Reconstruction - ion species (hero)</h2>")
        s.append(img(hero.get("png")))
        if hero.get("html"):
            s.append(f'<p class="links"><a href="{rel(hero["html"])}">'
                     "interactive 3D ion map</a></p>")
    knee = run.get("knee") or {}
    if knee:
        s.append("<h2>Method-selection evidence: k-NN knee vs "
                 "label-shuffle null</h2>")
        s.append(kv_table(knee, ["species", "k", "n_target_ions",
                                 "bimodal_signal", "cluster_mode_nm",
                                 "valley_nm", "null_mode_nm",
                                 "dmax_window_nm", "note"]))
        s.append(img(knee.get("png")))
    sweep = run.get("sweep") or {}
    if sweep and "error" not in sweep:
        s.append("<h2>MSM parameter sweep (stability plateau, "
                 "null-informed N_min)</h2>")
        s.append(kv_table(sweep, ["plateau_found", "plateau_range_nm",
                                  "n_clusters_at_plateau",
                                  "d_max_recommended_nm",
                                  "largest_null_cluster",
                                  "n_min_recommended",
                                  "percolation_warning", "note"]))
        s.append(img(sweep.get("png")))
    elif sweep:
        s.append("<h2>MSM parameter sweep</h2>"
                 f'<div class="note">{_esc(sweep["error"])}</div>')
    det = run.get("detect") or {}
    if det:
        s.append("<h2>MSM detection</h2>")
        s.append(kv_table(det, ["species", "d_max_nm", "n_min",
                                "envelope_nm", "erosion_nm", "n_clusters",
                                "analyzed_volume_nm3",
                                "number_density_per_m3",
                                "clustered_target_fraction"]))
        s.append(img(det.get("png")))
        if det.get("html"):
            s.append(f'<p class="links"><a href="{rel(det["html"])}">'
                     "interactive 3D cluster map</a></p>")
        cl = det.get("clusters") or []
        if cl:
            rows = "".join(
                f"<tr><td>{c['id']}</td><td>{c['n_ions']}</td>"
                f"<td>{c['n_target_ions']}</td>"
                f"<td>{c['center_nm']}</td>"
                f"<td>{c['guinier_radius_nm']}</td>"
                f"<td>{c['target_fraction']}</td>"
                f"<td>{_esc(', '.join(f'{k} {v:.2f}' for k, v in list(c['composition_ionic'].items())[:4]))}</td></tr>"
                for c in cl[:40])
            s.append("<table><tr><th>id</th><th>ions</th><th>target ions"
                     "</th><th>center [nm]</th><th>Guinier r [nm]</th>"
                     "<th>target frac</th><th>top ionic composition"
                     "</th></tr>" + rows + "</table>")
            if len(cl) > 40:
                s.append(f"<p>... {len(cl) - 40} more clusters "
                         "(see json)</p>")
        if det.get("matrix_composition_ionic_pct"):
            mc = det["matrix_composition_ionic_pct"]
            s.append("<p><b>Matrix ionic composition [at.%]:</b> "
                     + _esc(", ".join(f"{k} {v}" for k, v in
                                      list(mc.items())[:10])) + "</p>")
    gate = run.get("gate") or {}
    if gate:
        s.append("<h2>Accept gate: label-shuffle null</h2>")
        s.append(kv_table(gate, ["n_observed", "n_null", "n_null_mean",
                                 "n_null_std", "z_score",
                                 "null_contrast"]))
        s.append("<p>Verdict: " + badge(gate.get("null_gate_passed"),
                 "PASSED - detection collapses under label shuffling",
                 "NOT PASSED - no defensible cluster claim") + "</p>")
    wc = run.get("wc") or []
    if wc:
        s.append("<h2>Warren-Cowley SRO parameters (shuffle-controlled)"
                 "</h2>")
        rows = []
        for w in ([wc] if isinstance(wc, dict) else wc):
            if "error" in w:
                continue
            pair = (f"{'+'.join(w['center_species'][:2])} &rarr; "
                    f"{'+'.join(w['neighbor_species'][:2])}")
            for i, win in enumerate(w["shell_windows_nm"]):
                rows.append(
                    f"<tr><td>{pair}</td><td>{i + 1}</td>"
                    f"<td>{win[0]}&ndash;{win[1]}</td>"
                    f"<td>{w['alpha_per_shell'][i]:+.4f}</td>"
                    f"<td>{w['alpha_null_mean'][i]:+.4f} &plusmn; "
                    f"{w['alpha_null_std'][i]:.4f}</td>"
                    f"<td>{w['z_per_shell'][i]}</td>"
                    "<td>" + ("<span class='pass'>yes</span>"
                              if w["significant"][i] else "no") + "</td>"
                    "</tr>")
        if rows:
            s.append("<table><tr><th>pair (A &rarr; B)</th><th>shell</th>"
                     "<th>window [nm]</th><th>&alpha;</th>"
                     "<th>shuffle null</th><th>z</th><th>significant"
                     "</th></tr>" + "".join(rows) + "</table>")
            s.append("<p>&alpha; &lt; 0 = A-B association (ordering "
                     "tendency); &alpha; &gt; 0 = avoidance / "
                     "self-clustering; only shuffle-significant shells "
                     "are claimable.</p>")
        warn = next((w.get("validity_warning")
                     for w in ([wc] if isinstance(wc, dict) else wc)
                     if isinstance(w, dict) and w.get("validity_warning")),
                    None)
        if warn:
            s.append(f'<div class="note">{_esc(warn)}</div>')
    rdf = run.get("rdf") or {}
    if rdf:
        s.append("<h2>RDF vs label-shuffle null (model-free "
                 "corroboration)</h2>")
        for r in ([rdf] if isinstance(rdf, dict) else rdf):
            s.append(kv_table(r, ["species", "peak_ratio", "peak_r_nm",
                                  "significant_excess", "excess_range_nm",
                                  "excess_persists_to_r_max"]))
            s.append(img(r.get("png")))
    zs = run.get("zsdm") or {}
    if zs:
        s.append("<h2>z-SDM crystallographic-signal gate</h2>")
        s.append(kv_table(zs, ["species_a", "species_b", "n_pairs",
                               "fft_peak_contrast", "plane_spacing_A"]))
        s.append("<p>" + badge(zs.get("crystallographic_signal"),
                 _esc(zs.get("gate_verdict", "")),
                 _esc(zs.get("gate_verdict", ""))) + "</p>")
        s.append(img(zs.get("png")))
    if score:
        s.append("<h2>Benchmark score (truth opened AFTER the blind "
                 "run)</h2>")
        s.append(kv_table(score, ["n_truth", "n_detected", "n_matched",
                                  "recall", "precision", "size_pearson_r",
                                  "size_ratio_mean",
                                  "solute_count_ratio_mean",
                                  "target_fraction_mean",
                                  "target_fraction_truth"]))
        missed = score.get("missed_truth") or []
        if missed:
            rows = "".join(
                f"<tr><td>{m['idx']}</td><td>{m['radius_A']}</td>"
                f"<td>{m['n_solute_after']}</td>"
                f"<td>{m['expected_detected']}</td></tr>" for m in missed)
            s.append("<p><b>Missed truth clusters</b> (expected detected "
                     "solute count vs the null-demanded N_min):</p>"
                     "<table><tr><th>truth idx</th><th>radius [A]</th>"
                     "<th>solute seeded</th><th>expected detected</th>"
                     "</tr>" + rows + "</table>")
    if control:
        s.append(f"<h2>{_esc(control.get('title', 'False-positive control (cluster-free twin)'))}</h2>")
        s.append(kv_table(control.get("summary", {}),
                          ["n_false_positive_clusters",
                           "pipeline_stopped_at",
                           "gate_passed_on_control"]))
        ck = (control.get("knee") or {})
        if ck.get("png"):
            s.append(img(ck["png"]))
    s.append("<footer>Generated by cluster_tools.family1_report "
             "(scilink point_cloud_analysis / cluster_analysis). "
             "Policies: ionic composition (no molecular-ion "
             "decomposition), apex-up views. Every claim above is gated "
             "by its deterministic null model.</footer>")
    out = wd / out_name
    out.write_text("\n".join(s))
    return {"html": str(out)}


_IMP = ("from scilink.skills.point_cloud_analysis.cluster_analysis."
        "cluster_tools import ")

TOOL_SPECS = [
    ToolSpec(
        name="knn_distance_stats",
        description=("k-th NN distance distribution of a target ion species "
                     "vs its label-shuffle null: the MSM parameter-selection "
                     "evidence (bimodal signal, valley, suggested d_max "
                     "sweep window)."),
        parameters={"pos_path": {"type": "string",
                                 "description": ".apt/.pos or x,y,z,Da csv"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "species": {"type": "string",
                                "description": "ion-label list or regex "
                                               "(whole ions, never "
                                               "decomposed)"},
                    "k": {"type": "integer",
                          "description": "neighbour order (use planned "
                                         "N_min; default 10)"}},
        required=["pos_path", "rrng_path", "species"],
        import_line=_IMP + "knn_distance_stats",
        signature=("knn_distance_stats(pos_path, rrng_path, species, k=10, "
                   "n_shuffles=5, out_prefix='knn', workdir='.') -> dict"),
        agents=["simulation"],
        when_to_use=("FIRST family-1 step for discrete-cluster questions - "
                     "before any MSM run, to justify the d_max window."),
        returns=("bimodal_signal, cluster/null modes, valley_nm, "
                 "dmax_window_nm, png"),
        example=("knee = knn_distance_stats('tip.csv', 't.rrng', 'Cu', "
                 "k=10, workdir='out')"),
    ),
    ToolSpec(
        name="msm_parameter_sweep",
        description=("MSM d_max sweep with stability-plateau detection, "
                     "null count band, and null-informed N_min "
                     "recommendation - the BLIND parameter selector. "
                     "Cluster claims come from the plateau, never a single "
                     "cherry-picked d_max."),
        parameters={"pos_path": {"type": "string", "description": "data"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "species": {"type": "string",
                                "description": "target ion labels/regex"},
                    "n_min": {"type": "integer",
                              "description": "size floor during sweep "
                                             "(default 10)"}},
        required=["pos_path", "rrng_path", "species"],
        import_line=_IMP + "msm_parameter_sweep",
        signature=("msm_parameter_sweep(pos_path, rrng_path, species, "
                   "n_min=10, d_max_window_nm=None, n_steps=16, "
                   "n_shuffles=3, out_prefix='msm_sweep', workdir='.') "
                   "-> dict"),
        agents=["simulation"],
        when_to_use=("After knn_distance_stats; its d_max_recommended_nm + "
                     "n_min_recommended feed msm_detect."),
        returns=("plateau_found/range, d_max_recommended_nm, "
                 "n_min_recommended, observed vs null curves, png"),
        example=("sw = msm_parameter_sweep('tip.csv', 't.rrng', 'Cu', "
                 "workdir='out')"),
    ),
    ToolSpec(
        name="msm_detect",
        description=("Clean-room maximum-separation cluster detection "
                     "(posgen semantics: d_max link, N_min floor, envelope "
                     "+ erosion) at fixed parameters: per-cluster size / "
                     "Guinier radius / ionic composition, number density, "
                     "matrix composition, apex-up cluster map + 3D html."),
        parameters={"pos_path": {"type": "string", "description": "data"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "species": {"type": "string",
                                "description": "target ion labels/regex"},
                    "d_max_nm": {"type": "number",
                                 "description": "from the sweep plateau"},
                    "n_min": {"type": "integer",
                              "description": "from the null recommendation"}},
        required=["pos_path", "rrng_path", "species", "d_max_nm", "n_min"],
        import_line=_IMP + "msm_detect",
        signature=("msm_detect(pos_path, rrng_path, species, d_max_nm, "
                   "n_min, envelope_nm=None, erosion_nm=None, "
                   "out_prefix='msm', workdir='.') -> dict"),
        agents=["simulation"],
        when_to_use=("ONLY with sweep-selected parameters, and ALWAYS "
                     "followed by label_shuffle_null as the accept gate."),
        returns=("n_clusters, per-cluster stats, number_density, matrix "
                 "composition, clustered_target_fraction, png + 3D html"),
        example=("det = msm_detect('tip.csv', 't.rrng', 'Cu', 0.7, 12, "
                 "workdir='out')"),
    ),
    ToolSpec(
        name="label_shuffle_null",
        description=("THE family-1 accept gate: rerun the identical MSM "
                     "detection on >= 5 random label permutations - a real "
                     "signal must collapse. Deterministic verdict "
                     "(observed > null + 3 sigma)."),
        parameters={"pos_path": {"type": "string", "description": "data"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "species": {"type": "string",
                                "description": "target ion labels/regex"},
                    "d_max_nm": {"type": "number",
                                 "description": "same as msm_detect"},
                    "n_min": {"type": "integer",
                              "description": "same as msm_detect"}},
        required=["pos_path", "rrng_path", "species", "d_max_nm", "n_min"],
        import_line=_IMP + "label_shuffle_null",
        signature=("label_shuffle_null(pos_path, rrng_path, species, "
                   "d_max_nm, n_min, n_shuffles=5) -> dict"),
        agents=["simulation"],
        when_to_use=("MANDATORY after every msm_detect - no cluster claim "
                     "ships without null_gate_passed."),
        returns=("n_observed, n_null list/mean/std, z_score, "
                 "null_contrast, null_gate_passed"),
        example=("gate = label_shuffle_null('tip.csv', 't.rrng', 'Cu', "
                 "0.7, 12)"),
    ),
    ToolSpec(
        name="rdf_compare",
        description=("Target-target pair correlation as the RATIO to its "
                     "label-shuffle null (cancels tip shape / density "
                     "gradients): model-free clustering/SRO screen with a "
                     "3-sigma significance band."),
        parameters={"pos_path": {"type": "string", "description": "data"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "species": {"type": "string",
                                "description": "target ion labels/regex"}},
        required=["pos_path", "rrng_path", "species"],
        import_line=_IMP + "rdf_compare",
        signature=("rdf_compare(pos_path, rrng_path, species, r_max_nm=6.0, "
                   "dr_nm=0.05, n_shuffles=5, out_prefix='rdf', "
                   "workdir='.') -> dict"),
        agents=["simulation"],
        when_to_use=("Open-ended 'any clustering/SRO?' questions - run "
                     "BEFORE committing to a discrete-cluster model; also "
                     "corroborates MSM results (concordance rule)."),
        returns=("peak_ratio, peak_r_nm, significant_excess, "
                 "excess_range_nm, png"),
        example="rdf = rdf_compare('tip.csv', 't.rrng', 'Cu', workdir='out')",
    ),
    ToolSpec(
        name="warren_cowley",
        description=("Warren-Cowley SRO parameter per NN shell with "
                     "label-shuffle control. QUANTITATIVE on full-density "
                     "simulated clouds; on real APT reconstructions it "
                     "attaches a validity_warning and may only corroborate "
                     "(never anchor) a claim."),
        parameters={"structure_path": {"type": "string",
                                       "description": "xyz/LAMMPS/CIF "
                                                      "(full-density route)"},
                    "pos_path": {"type": "string",
                                 "description": "APT route (secondary "
                                                "evidence only)"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "center_species": {"type": "string",
                                       "description": "A in alpha(A-B)"},
                    "neighbor_species": {"type": "string",
                                         "description": "B in alpha(A-B)"}},
        required=["center_species", "neighbor_species"],
        import_line=_IMP + "warren_cowley",
        signature=("warren_cowley(structure_path=None, pos_path=None, "
                   "rrng_path=None, center_species='', neighbor_species='', "
                   "type_map=None, n_shells=1, shell_windows_nm=None, "
                   "n_shuffles=5) -> dict"),
        agents=["simulation"],
        when_to_use=("SRO quantification on simulated/ideal lattices "
                     "(cloud-kind rule 1); alpha < 0 = ordering, > 0 = "
                     "clustering; claim only shells with significant=true."),
        returns=("alpha_per_shell, null mean/std, z_per_shell, "
                 "significant flags, shell windows"),
        example=("wc = warren_cowley(structure_path='wtav.xyz', "
                 "center_species='W', neighbor_species='Ta')"),
    ),
    ToolSpec(
        name="zsdm",
        description=("Per-pair z-SDM (delta-z spatial distribution map) "
                     "with FFT lattice-plane detection - the "
                     "crystallographic-signal GATE: without plane "
                     "oscillations NO site-resolved ordering claim is "
                     "defensible on the data, by any method."),
        parameters={"pos_path": {"type": "string", "description": "data"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "species_a": {"type": "string",
                                  "description": "center ion labels/regex"},
                    "species_b": {"type": "string",
                                  "description": "neighbor labels "
                                                 "(default = species_a)"}},
        required=["pos_path", "rrng_path", "species_a"],
        import_line=_IMP + "zsdm",
        signature=("zsdm(pos_path, rrng_path, species_a, species_b=None, "
                   "dz_max_nm=1.0, r_lateral_nm=0.75, bin_nm=0.01, "
                   "out_prefix='zsdm', workdir='.') -> dict"),
        agents=["simulation"],
        when_to_use=("BEFORE any site-resolved SRO/ordering claim on APT "
                     "data (spec rule 2), and to test whether ML-CSRO "
                     "z-SDM features are even present."),
        returns=("crystallographic_signal, fft_peak_contrast, "
                 "plane_spacing_A, gate_verdict, png"),
        example=("g = zsdm('tip.csv', 't.rrng', 'Fe', workdir='out')"),
    ),
    ToolSpec(
        name="family1_report",
        description=("Assemble the standard family-1 HTML report (hero, "
                     "knee/sweep evidence, cluster table + 3D link, null "
                     "gates, optional benchmark score) from pipeline "
                     "results. Report artifacts are PIPELINE-DEFAULT - "
                     "every family-1 run must end with this call."),
        parameters={"workdir": {"type": "string",
                                "description": "run folder (figures live "
                                               "here; links are relative)"},
                    "run": {"type": "object",
                            "description": "results under keys knee/sweep/"
                                           "detect/gate/rdf/zsdm"}},
        required=["workdir", "run"],
        import_line=_IMP + "family1_report",
        signature=("family1_report(workdir, run, title=..., hero=None, "
                   "score=None, control=None, extra_note='', "
                   "out_name='report.html') -> dict"),
        agents=["simulation"],
        when_to_use=("ALWAYS, as the last step of any family-1 analysis - "
                     "pass the visualize_apt_elements result as hero."),
        returns="html (report path)",
        example=("rep = family1_report('out', {'knee': knee, 'sweep': sw, "
                 "'detect': det, 'gate': gate}, hero=viz)"),
    ),
]
