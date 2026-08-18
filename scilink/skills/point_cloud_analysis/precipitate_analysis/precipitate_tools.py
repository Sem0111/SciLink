"""Family-2 precipitate tools for APT point clouds.

Continuum companion to the family-1 point statistics: delocalized
concentration grids, iso-concentration surfaces (marching cubes) with
per-precipitate morphology and population statistics, proximity
histograms (proxigrams) with the far-field convergence gate, and the
lever-rule mass-balance check. Methods follow the community canon
(Hellman et al., Microsc. Microanal. 6 (2000) proxigram; IVAS-style
delocalization); implementation is original.

Deterministic gates (every claim must survive them):
- proxigram far field MUST converge to the independently measured
  matrix composition;
- lever rule: Vf x C_ppt + (1-Vf) x C_matrix must reconstruct the bulk
  composition within counting statistics;
- results are reported across a THRESHOLD SWEEP, never a single
  cherry-picked iso-value.

Coordinates: nm; apex-up applied only in figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scilink.skills._shared._spec import ToolSpec
from scilink.skills.point_cloud_analysis.cluster_analysis.cluster_tools import (
    _COLORS, _esc, _jsonable, _load_ranged_cloud, _report_css, _target_mask)


def _fraction_field(xyz, gmask, voxel_nm, delocalization_nm,
                    min_ions_per_voxel=10):
    """Delocalized solute-fraction field: Gaussian-smoothed group counts
    over Gaussian-smoothed total counts; voxels below the ion floor are
    invalid (NaN)."""
    from scipy.ndimage import gaussian_filter

    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    bins = [max(4, int((hi[i] - lo[i]) / voxel_nm)) for i in range(3)]
    H_all, edges = np.histogramdd(xyz, bins=bins, range=list(zip(lo, hi)))
    H_g, _ = np.histogramdd(xyz[gmask], bins=bins, range=list(zip(lo, hi)))
    sig = delocalization_nm / voxel_nm
    H_all_s = gaussian_filter(H_all, sig)
    H_g_s = gaussian_filter(H_g, sig)
    valid = H_all_s >= min_ions_per_voxel
    with np.errstate(invalid="ignore"):
        frac = np.where(valid, H_g_s / np.maximum(H_all_s, 1e-9), np.nan)
    return frac, valid, edges, H_all


def _voxel_of(xyz, edges):
    return tuple(np.clip(np.digitize(xyz[:, k], edges[k]) - 1, 0,
                         len(edges[k]) - 2) for k in range(3))


def concentration_grid(pos_path: str, rrng_path: str, species,
                       voxel_nm: float = 1.0,
                       delocalization_nm: float = 1.0,
                       min_ions_per_voxel: int = 10,
                       out_prefix: str = "grid",
                       workdir: str = ".") -> dict:
    """Delocalized 3D concentration field of an ion-species GROUP - the
    family-2 foundation. Returns field statistics, a suggested
    iso-threshold (baseline + 3 sigma of the voxel-fraction
    distribution), a mid-slice map, and the voxel-fraction histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
    gmask, members = _target_mask(labels, species)
    frac, valid, edges, H_all = _fraction_field(
        xyz, gmask, voxel_nm, delocalization_nm, min_ions_per_voxel)
    v = frac[np.isfinite(frac)]
    baseline = float(np.median(v))
    thr = float(baseline + 3 * np.std(v))
    res = {"species": members, "voxel_nm": voxel_nm,
           "delocalization_nm": delocalization_nm,
           "n_valid_voxels": int(valid.sum()),
           "baseline_fraction": round(baseline, 4),
           "voxel_fraction_std": round(float(np.std(v)), 4),
           "max_fraction": round(float(np.max(v)), 4),
           "suggested_threshold": round(thr, 4),
           "group_fraction_overall": round(float(gmask.mean()), 5)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    mid = frac[:, frac.shape[1] // 2, :]
    im = ax1.imshow(mid.T, origin="lower", aspect="equal", cmap="inferno",
                    extent=[edges[0][0], edges[0][-1],
                            edges[2][0], edges[2][-1]])
    ax1.invert_yaxis()
    ax1.set_title(f"{'+'.join(members[:3])} fraction - x-z mid slice")
    plt.colorbar(im, ax=ax1, shrink=0.8)
    ax2.hist(v, bins=120, color=_COLORS[0])
    ax2.axvline(thr, color=_COLORS[1], ls="--",
                label=f"suggested thr {thr:.3f}")
    ax2.set_yscale("log")
    ax2.set_xlabel("voxel fraction")
    ax2.legend(fontsize=8)
    ax2.set_title("voxel-fraction distribution")
    png = wd / f"{out_prefix}_grid.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    res["png"] = str(png)
    return _jsonable(res)


def isosurface_precipitates(pos_path: str, rrng_path: str, species,
                            threshold: float, voxel_nm: float = 1.0,
                            delocalization_nm: float = 1.0,
                            min_ions_per_voxel: int = 10,
                            min_voxels: int = 8,
                            out_prefix: str = "iso",
                            workdir: str = ".",
                            make_3d_html: bool = True) -> dict:
    """Iso-concentration surface analysis: threshold the delocalized
    field, label connected precipitates, and report per-precipitate
    morphology (volume, equivalent radius, sphericity, depth) plus
    population statistics (count, number density, volume fraction) and
    a THRESHOLD SWEEP of count/Vf (claims must be threshold-stable)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from skimage.measure import marching_cubes

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
    gmask, members = _target_mask(labels, species)
    frac, valid, edges, H_all = _fraction_field(
        xyz, gmask, voxel_nm, delocalization_nm, min_ions_per_voxel)
    z_max = xyz[:, 2].max()
    vvol = voxel_nm ** 3
    mask = np.nan_to_num(frac) >= threshold
    lab, n_raw = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n_raw + 1))
    keep = np.where(sizes >= min_voxels)[0]
    keep = keep[np.argsort(-sizes[keep])]

    ppts = []
    for new_id, oi in enumerate(keep):
        m = lab == oi + 1
        vx = np.argwhere(m)
        cen_vox = vx.mean(axis=0)
        cen = np.array([edges[k][0] + (cen_vox[k] + 0.5) * voxel_nm
                        for k in range(3)])
        vol = float(m.sum()) * vvol
        req = (3 * vol / (4 * np.pi)) ** (1 / 3)
        sph = None
        try:
            sub = np.pad(m[vx[:, 0].min():vx[:, 0].max() + 1,
                           vx[:, 1].min():vx[:, 1].max() + 1,
                           vx[:, 2].min():vx[:, 2].max() + 1]
                         .astype(float), 1)
            verts, faces, _, _ = marching_cubes(sub, level=0.5)
            tri = verts[faces] * voxel_nm
            area = float(np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                axis=1).sum() / 2)
            sph = round(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area, 3)
        except Exception:  # noqa: BLE001 - degenerate blob
            pass
        ppts.append({"id": new_id, "n_voxels": int(m.sum()),
                     "volume_nm3": round(vol, 1),
                     "eq_radius_nm": round(float(req), 2),
                     "sphericity": sph,
                     "center_nm": [round(float(c), 1) for c in cen],
                     "depth_below_apex_nm": round(float(z_max - cen[2]), 1)})
    n_ppt = len(ppts)
    v_valid = float(valid.sum()) * vvol
    ppt_vox = int(sum(p["n_voxels"] for p in ppts))
    res = {"species": members, "threshold": threshold,
           "voxel_nm": voxel_nm, "delocalization_nm": delocalization_nm,
           "n_precipitates": n_ppt,
           "analyzed_volume_nm3": round(v_valid, 0),
           "number_density_per_m3": float(n_ppt / max(v_valid, 1) * 1e27),
           "volume_fraction": round(ppt_vox * vvol / max(v_valid, 1), 5),
           "eq_radius_median_nm": (round(float(np.median(
               [p["eq_radius_nm"] for p in ppts])), 2) if ppts else None),
           "precipitates": ppts}

    # threshold sweep: count and Vf must be stable around the choice
    sweep = []
    for t in np.linspace(0.6 * threshold, 1.4 * threshold, 9):
        m_t = np.nan_to_num(frac) >= t
        l_t, n_t = ndimage.label(m_t)
        s_t = ndimage.sum(m_t, l_t, range(1, n_t + 1))
        nk = int((s_t >= min_voxels).sum())
        sweep.append({"threshold": round(float(t), 4), "n": nk,
                      "vf": round(float(s_t[s_t >= min_voxels].sum())
                                  * vvol / max(v_valid, 1), 5)})
    res["threshold_sweep"] = sweep

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5),
                                   gridspec_kw={"width_ratios": [1, 1.2]})
    ax1.plot([s["threshold"] for s in sweep], [s["n"] for s in sweep],
             "-o", ms=4, color=_COLORS[0], label="count")
    ax1.axvline(threshold, color=_COLORS[1], ls="--")
    ax1b = ax1.twinx()
    ax1b.plot([s["threshold"] for s in sweep], [s["vf"] for s in sweep],
              "-s", ms=4, color=_COLORS[3], label="volume fraction")
    ax1.set_xlabel("iso threshold")
    ax1.set_ylabel("precipitate count")
    ax1b.set_ylabel("volume fraction")
    ax1.set_title("threshold sweep")
    vx = np.argwhere(lab > 0)
    if len(vx):
        xs = edges[0][0] + (vx[:, 0] + 0.5) * voxel_nm
        zs = edges[2][0] + (vx[:, 2] + 0.5) * voxel_nm
        cid = lab[vx[:, 0], vx[:, 1], vx[:, 2]]
        ax2.scatter(xs, z_max - zs, s=3, lw=0,
                    c=[_COLORS[int(c) % 7] for c in cid])
    ax2.set_aspect(1)
    ax2.invert_yaxis()
    ax2.set_xlabel("x [nm]")
    ax2.set_ylabel("depth below apex [nm]")
    ax2.set_title(f"precipitate voxels: N={n_ppt} at thr={threshold}")
    png = wd / f"{out_prefix}_iso.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    res["png"] = str(png)

    if make_3d_html and mask.any():
        try:
            import plotly.graph_objects as go
            verts, faces, _, _ = marching_cubes(
                np.pad(np.nan_to_num(frac), 1, constant_values=0),
                level=threshold)
            verts = (verts - 1) * voxel_nm + np.array(
                [edges[k][0] for k in range(3)])
            figp = go.Figure(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=-verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="#4053d3", opacity=0.55))
            figp.update_layout(scene_aspectmode="data",
                               title=f"iso-surfaces at {threshold} "
                                     "(apex up)")
            html = wd / f"{out_prefix}_iso_3d.html"
            figp.write_html(html, include_plotlyjs=True)
            res["html"] = str(html)
        except ImportError:
            res["html"] = None
    return _jsonable(res)


def proxigram(pos_path: str, rrng_path: str, species,
              threshold: float, voxel_nm: float = 1.0,
              delocalization_nm: float = 1.0,
              min_ions_per_voxel: int = 10,
              bin_nm: float = 0.25, max_dist_nm: float = 6.0,
              volume_fraction: float | None = None,
              out_prefix: str = "prox", workdir: str = ".") -> dict:
    """Proximity histogram: composition vs SIGNED distance from the
    iso-surface (negative = inside precipitates), stacked over all
    interfaces. Carries the two family-2 gates:
    - far-field gate: the outside plateau must converge to the matrix
      composition measured directly on outside-everything ions;
    - lever rule (when volume_fraction given): Vf x C_in + (1-Vf) x
      C_out must reconstruct the measured bulk composition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
    gmask, members = _target_mask(labels, species)
    frac, valid, edges, H_all = _fraction_field(
        xyz, gmask, voxel_nm, delocalization_nm, min_ions_per_voxel)
    mask = np.nan_to_num(frac) >= threshold
    d_in = ndimage.distance_transform_edt(mask) * voxel_nm
    d_out = ndimage.distance_transform_edt(~mask) * voxel_nm
    signed = np.where(mask, -d_in, d_out)

    ii = _voxel_of(xyz, edges)
    ion_d = signed[ii]
    ion_valid = valid[ii]
    u, c = np.unique(labels, return_counts=True)
    majors = [str(l) for l, _ in
              sorted(zip(u, c), key=lambda t: -t[1])[:4]]
    series = {"target-group": gmask}
    for m in majors:
        series[m] = labels == m

    edges_d = np.arange(-max_dist_nm, max_dist_nm + bin_nm, bin_nm)
    mids = 0.5 * (edges_d[:-1] + edges_d[1:])
    prox, perr = {}, {}
    n_bin = np.histogram(ion_d[ion_valid], bins=edges_d)[0]
    for name, m in series.items():
        n_s = np.histogram(ion_d[ion_valid & m], bins=edges_d)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            prox[name] = np.where(n_bin > 0, n_s / np.maximum(n_bin, 1),
                                  np.nan)
            perr[name] = np.sqrt(np.maximum(n_s, 1)) / np.maximum(n_bin, 1)

    # gates
    far = mids >= 3.0
    inside = mids <= -1.0
    out_all = ion_valid & (ion_d > 0)
    res_gates = {}
    bulk_out = float(gmask[out_all].mean()) if out_all.any() else np.nan
    ff = float(np.nanmean(prox["target-group"][far]))
    ff_sig = float(np.nanmean(perr["target-group"][far]))
    res_gates["far_field"] = {
        "proxigram_far_field": round(ff, 4),
        "matrix_outside_direct": round(bulk_out, 4),
        "difference_sigma": round(abs(ff - bulk_out)
                                  / max(ff_sig, 1e-9), 1),
        "passed": bool(abs(ff - bulk_out) < max(3 * ff_sig,
                                                0.1 * bulk_out))}
    lever = None
    if volume_fraction is not None:
        c_in = float(np.nanmean(prox["target-group"][inside]))
        bulk = float(gmask[ion_valid].mean())
        recon = volume_fraction * c_in + (1 - volume_fraction) * ff
        lever = {"C_inside_plateau": round(c_in, 4),
                 "C_outside_plateau": round(ff, 4),
                 "volume_fraction": volume_fraction,
                 "bulk_measured": round(bulk, 4),
                 "bulk_reconstructed": round(recon, 4),
                 "relative_error": round(abs(recon - bulk)
                                         / max(bulk, 1e-9), 3),
                 "passed": bool(abs(recon - bulk)
                                <= max(0.15 * bulk, 3 * ff_sig)),
                 "note": ("ion-basis reconstruction; phase density "
                          "differences and interface smearing set the "
                          "15% tolerance")}
        res_gates["lever_rule"] = lever

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, p) in enumerate(prox.items()):
        ax.errorbar(mids, p * 100, yerr=perr[name] * 100, lw=1.2,
                    elinewidth=0.5, color=_COLORS[i % 7], label=name)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.axhline(bulk_out * 100, color=_COLORS[0], ls=":", lw=0.8)
    ax.set_xlabel("signed distance from iso-surface [nm] "
                  "(negative = inside)")
    ax.set_ylabel("ionic fraction [%]")
    ax.legend(fontsize=8)
    verdict = "PASSED" if res_gates["far_field"]["passed"] else "FAILED"
    ax.set_title(f"proxigram at thr={threshold} - far-field gate "
                 f"{verdict}")
    png = wd / f"{out_prefix}_proxigram.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return _jsonable({
        "species": members, "threshold": threshold,
        "interface_bins_nm": bin_nm,
        "proxigram": {k: [round(float(x), 4) if np.isfinite(x) else None
                          for x in v] for k, v in prox.items()},
        "distance_mids_nm": [round(float(m), 2) for m in mids],
        "gates": res_gates, "png": str(png)})


def family2_report(workdir: str, run: dict,
                   title: str = "Family-2 precipitate analysis",
                   hero: dict | None = None, score: dict | None = None,
                   assessment: dict | None = None, extra_note: str = "",
                   out_name: str = "report.html") -> dict:
    """Assemble the standard family-2 HTML report (grid evidence,
    iso-surface population + morphology, proxigram + gate verdicts,
    optional benchmark score) - the PIPELINE-DEFAULT report stage; call
    at the end of EVERY family-2 run. ``run`` keys: grid / iso / prox."""
    import os

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)

    def rel(p):
        return _esc(os.path.relpath(str(p), str(wd))) if p else None

    def img(p):
        return f'<img src="{rel(p)}">' if p else ""

    def kv(d, keys):
        rows = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(d[k])}</td></tr>"
                       for k in keys if k in d and d[k] is not None)
        return f"<table>{rows}</table>" if rows else ""

    def badge(ok):
        return ('<span class="pass">PASSED</span>' if ok
                else '<span class="fail">FAILED</span>')

    s = [f"<title>{_esc(title)}</title><style>{_report_css()}</style>",
         f"<h1>{_esc(title)}</h1>"]
    if extra_note:
        s.append(f'<div class="note">{extra_note}</div>')
    if assessment:
        s.append("<h2>Scientific assessment</h2>")
        if assessment.get("error"):
            s.append(f'<div class="note">{_esc(assessment["error"])}</div>')
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
                     f"({_esc(assessment.get('model', ''))}); gate "
                     "verdicts remain authoritative.</i></p>")
    if hero:
        s.append("<h2>Reconstruction - ion species (hero)</h2>")
        s.append(img(hero.get("png")))
        if hero.get("html"):
            s.append(f'<p class="links"><a href="{rel(hero["html"])}">'
                     "interactive 3D ion map</a></p>")
    grid = run.get("grid") or {}
    if grid:
        s.append("<h2>Delocalized concentration field</h2>")
        s.append(kv(grid, ["species", "voxel_nm", "delocalization_nm",
                           "baseline_fraction", "max_fraction",
                           "suggested_threshold"]))
        s.append(img(grid.get("png")))
    iso = run.get("iso") or {}
    if iso:
        s.append("<h2>Iso-concentration surfaces</h2>")
        s.append(kv(iso, ["threshold", "n_precipitates",
                          "number_density_per_m3", "volume_fraction",
                          "eq_radius_median_nm", "analyzed_volume_nm3"]))
        rows = "".join(
            f"<tr><td>{p['id']}</td><td>{p['volume_nm3']}</td>"
            f"<td>{p['eq_radius_nm']}</td><td>{p['sphericity']}</td>"
            f"<td>{p['center_nm']}</td>"
            f"<td>{p['depth_below_apex_nm']}</td></tr>"
            for p in (iso.get("precipitates") or [])[:40])
        if rows:
            s.append("<table><tr><th>id</th><th>V [nm3]</th>"
                     "<th>r_eq [nm]</th><th>sphericity</th>"
                     "<th>center [nm]</th><th>depth [nm]</th></tr>"
                     + rows + "</table>")
        s.append(img(iso.get("png")))
        if iso.get("html"):
            s.append(f'<p class="links"><a href="{rel(iso["html"])}">'
                     "interactive 3D iso-surfaces</a></p>")
    prox = run.get("prox") or {}
    if prox:
        s.append("<h2>Proxigram + gates</h2>")
        gates = prox.get("gates") or {}
        ff = gates.get("far_field") or {}
        s.append("<p>Far-field convergence: " + badge(ff.get("passed"))
                 + f" (proxigram {ff.get('proxigram_far_field')} vs "
                 f"matrix {ff.get('matrix_outside_direct')}, "
                 f"{ff.get('difference_sigma')} sigma)</p>")
        lr = gates.get("lever_rule")
        if lr:
            s.append("<p>Lever rule: " + badge(lr.get("passed"))
                     + f" (bulk {lr.get('bulk_measured')} vs "
                     f"reconstructed {lr.get('bulk_reconstructed')}, "
                     f"rel. err {lr.get('relative_error')})</p>")
        s.append(img(prox.get("png")))
    if score:
        s.append("<h2>Benchmark score (truth opened AFTER the blind "
                 "run)</h2>")
        s.append(kv(score, list(score.keys())[:14]))
    s.append("<footer>Generated by precipitate_tools.family2_report "
             "(scilink point_cloud_analysis / precipitate_analysis). "
             "Policies: ionic composition, apex-up views; claims are "
             "gated by far-field convergence and the lever rule.</footer>")
    out = wd / out_name
    out.write_text("\n".join(s))
    return {"html": str(out)}


_IMP = ("from scilink.skills.point_cloud_analysis.precipitate_analysis."
        "precipitate_tools import ")

TOOL_SPECS = [
    ToolSpec(
        name="concentration_grid",
        description=("Delocalized 3D concentration field of an ion-species "
                     "GROUP - the family-2 foundation. Returns field "
                     "statistics and a SUGGESTED iso-threshold "
                     "(baseline + 3 sigma)."),
        parameters={"pos_path": {"type": "string",
                                 "description": ".pos/.apt or x,y,z,Da csv"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "species": {"type": "string",
                                "description": "ion labels/regex (whole "
                                               "ions, never decomposed)"}},
        required=["pos_path", "rrng_path", "species"],
        import_line=_IMP + "concentration_grid",
        signature=("concentration_grid(pos_path, rrng_path, species, "
                   "voxel_nm=1.0, delocalization_nm=1.0, "
                   "out_prefix='grid', workdir='.') -> dict"),
        agents=["simulation"],
        when_to_use=("FIRST family-2 step for precipitate/second-phase "
                     "questions - its suggested_threshold feeds "
                     "isosurface_precipitates."),
        returns=("baseline/max voxel fractions, suggested_threshold, "
                 "mid-slice + histogram png"),
        example=("g = concentration_grid('tip.pos', 't.rrng', 'Ti|Y|O', "
                 "workdir='out')"),
    ),
    ToolSpec(
        name="isosurface_precipitates",
        description=("Iso-concentration-surface precipitate analysis: "
                     "count, number density, volume fraction, "
                     "per-precipitate volume / equivalent radius / "
                     "sphericity / depth, THRESHOLD SWEEP (claims must be "
                     "threshold-stable), 3D surface html."),
        parameters={"pos_path": {"type": "string", "description": "data"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "species": {"type": "string",
                                "description": "target group labels/regex"},
                    "threshold": {"type": "number",
                                  "description": "iso-fraction (from "
                                                 "concentration_grid or a "
                                                 "literature setting)"}},
        required=["pos_path", "rrng_path", "species", "threshold"],
        import_line=_IMP + "isosurface_precipitates",
        signature=("isosurface_precipitates(pos_path, rrng_path, species, "
                   "threshold, voxel_nm=1.0, delocalization_nm=1.0, "
                   "min_voxels=8, out_prefix='iso', workdir='.') -> dict"),
        agents=["simulation"],
        when_to_use=("After concentration_grid; ALWAYS follow with "
                     "proxigram (pass this tool's volume_fraction) - its "
                     "gates are the accept criteria."),
        returns=("n_precipitates, number_density_per_m3, volume_fraction, "
                 "per-precipitate morphology, threshold_sweep, png + 3D "
                 "html"),
        example=("iso = isosurface_precipitates('tip.pos', 't.rrng', "
                 "'Ti|Y|O', g['suggested_threshold'], workdir='out')"),
    ),
    ToolSpec(
        name="proxigram",
        description=("Proximity histogram: composition vs signed distance "
                     "from the iso-surface, stacked over all interfaces - "
                     "core composition, interface width, matrix plateau. "
                     "Carries the family-2 ACCEPT GATES: far-field "
                     "convergence to the directly measured matrix "
                     "composition, and the lever-rule mass balance when "
                     "volume_fraction is passed."),
        parameters={"pos_path": {"type": "string", "description": "data"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "species": {"type": "string",
                                "description": "target group labels/regex"},
                    "threshold": {"type": "number",
                                  "description": "SAME iso value as "
                                                 "isosurface_precipitates"},
                    "volume_fraction": {"type": "number",
                                        "description": "from isosurface_"
                                                       "precipitates - "
                                                       "enables the lever-"
                                                       "rule gate"}},
        required=["pos_path", "rrng_path", "species", "threshold"],
        import_line=_IMP + "proxigram",
        signature=("proxigram(pos_path, rrng_path, species, threshold, "
                   "voxel_nm=1.0, delocalization_nm=1.0, bin_nm=0.25, "
                   "volume_fraction=None, out_prefix='prox', "
                   "workdir='.') -> dict"),
        agents=["simulation"],
        when_to_use=("MANDATORY after isosurface_precipitates - no "
                     "precipitate claim ships unless gates.far_field."
                     "passed (and lever_rule when Vf was available)."),
        returns=("proxigram curves per species, gates.far_field, "
                 "gates.lever_rule, png"),
        example=("px = proxigram('tip.pos', 't.rrng', 'Ti|Y|O', 0.03, "
                 "volume_fraction=iso['volume_fraction'])"),
    ),
    ToolSpec(
        name="family2_report",
        description=("Assemble the standard family-2 HTML report (grid "
                     "evidence, iso-surface population + morphology "
                     "table, proxigram + gate verdicts). Report "
                     "artifacts are PIPELINE-DEFAULT - every family-2 "
                     "run must end with this call."),
        parameters={"workdir": {"type": "string",
                                "description": "run folder"},
                    "run": {"type": "object",
                            "description": "results under keys "
                                           "grid / iso / prox"}},
        required=["workdir", "run"],
        import_line=_IMP + "family2_report",
        signature=("family2_report(workdir, run, title=..., hero=None, "
                   "score=None, assessment=None, extra_note='', "
                   "out_name='report.html') -> dict"),
        agents=["simulation"],
        when_to_use=("ALWAYS, as the last step of any family-2 analysis "
                     "- pass visualize_apt_elements as hero and "
                     "family1_assessment output as assessment."),
        returns="html (report path)",
        example=("rep = family2_report('out', {'grid': g, 'iso': iso, "
                 "'prox': px}, hero=viz)"),
    ),
]
