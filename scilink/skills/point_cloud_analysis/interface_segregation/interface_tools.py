"""Family-3 interface-segregation tools for APT point clouds.

Planar-feature detection (PCA on solute-enriched voxels), 1D composition
profiles across the interface, Gibbsian interfacial excess by TWO
independent routes (profile integration and the Krakauer-Seidman
cumulative ladder) that must agree, a feature-blind label-shuffle null,
and 2D in-plane concentration maps. Methods follow the community canon
(Krakauer & Seidman, Phys. Rev. B 48 (1993) 6724; Hellman et al. 2000);
implementation is original.

Gates: profile far field must equal the bulk matrix composition; the
two excess estimates must agree; the excess must collapse under label
shuffling. Excess is reported on the DETECTED-ion basis and, when the
detection efficiency is given, corrected to atoms per nm^2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scilink.skills._shared._spec import ToolSpec
from scilink.skills.point_cloud_analysis.cluster_analysis.cluster_tools import (
    _COLORS, _esc, _jsonable, _load_ranged_cloud, _report_css, _target_mask)
from scilink.skills.point_cloud_analysis.precipitate_analysis.precipitate_tools import (
    _fraction_field)


def _plane_basis(normal):
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, u, v


def detect_planar_feature(pos_path: str, rrng_path: str, species,
                          voxel_nm: float = 1.0,
                          delocalization_nm: float = 1.0,
                          k_sigma: float = 2.0, min_voxels: int = 30,
                          out_prefix: str = "feature",
                          workdir: str = ".") -> dict:
    """Find solute-enriched planar features: threshold the delocalized
    fraction field (baseline + k sigma), label connected enriched
    regions, and PCA-fit each (normal, centroid, planarity, thickness,
    in-plane extent). The top feature is the default interface for the
    downstream profile; a low planarity means the region is NOT a plane
    (blob/column) and the profile tools should not be trusted on it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
    gmask, members = _target_mask(labels, species)
    frac, valid, edges, _ = _fraction_field(xyz, gmask, voxel_nm,
                                            delocalization_nm)
    v = frac[np.isfinite(frac)]
    base, sd = float(np.median(v)), float(np.std(v))
    thr = base + k_sigma * sd
    mask = np.nan_to_num(frac) >= thr
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    order = np.argsort(-sizes)[:5]
    z_max = xyz[:, 2].max()
    feats = []
    for rank, oi in enumerate(order):
        if sizes[oi] < min_voxels:
            continue
        vx = np.argwhere(lab == oi + 1)
        pts = np.stack([edges[k][0] + (vx[:, k] + 0.5) * voxel_nm
                        for k in range(3)], 1)
        cen = pts.mean(0)
        w, V = np.linalg.eigh(np.cov((pts - cen).T))
        w = np.maximum(w, 1e-12)
        normal = V[:, 0]
        if normal[2] < 0:
            normal = -normal
        fvals = frac[vx[:, 0], vx[:, 1], vx[:, 2]]
        feats.append({"rank": rank, "n_voxels": int(sizes[oi]),
                      "point_nm": [round(float(c), 2) for c in cen],
                      "depth_below_apex_nm": round(float(z_max - cen[2]), 1),
                      "normal": [round(float(c), 4) for c in normal],
                      "planarity": round(float(1 - w[0] / w[1]), 3),
                      "thickness_nm": round(float(2 * np.sqrt(w[0])), 2),
                      "extent_nm": [round(float(2 * np.sqrt(w[2])), 1),
                                    round(float(2 * np.sqrt(w[1])), 1)],
                      "mean_fraction": round(float(fvals.mean()), 4),
                      "enrichment_vs_baseline": round(
                          float(fvals.mean() / max(base, 1e-9)), 2)})
    res = {"species": members, "threshold": round(thr, 4),
           "baseline_fraction": round(base, 4), "n_features": len(feats),
           "features": feats,
           "top_feature": feats[0] if feats else None,
           # noise blobs reach ~0.7 on the synthetic control; genuine
           # boundaries score >= 0.95 - the gate downstream still decides
           "top_is_planar": bool(feats and feats[0]["planarity"] >= 0.85)}

    fig, ax = plt.subplots(figsize=(6, 8))
    sub = np.random.RandomState(0).choice(len(xyz), min(len(xyz), 120000),
                                          replace=False)
    ax.scatter(xyz[sub, 0], z_max - xyz[sub, 2], s=0.3, lw=0, c="#dddddd",
               rasterized=True)
    for f, oi in zip(feats, order):
        vx = np.argwhere(lab == oi + 1)
        ax.scatter(edges[0][0] + (vx[:, 0] + 0.5) * voxel_nm,
                   z_max - (edges[2][0] + (vx[:, 2] + 0.5) * voxel_nm),
                   s=4, lw=0, c=_COLORS[f["rank"] % 7],
                   label=f"feature {f['rank']} (planarity "
                         f"{f['planarity']})")
    ax.set_aspect(1)
    ax.invert_yaxis()
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("depth below apex [nm]")
    ax.legend(fontsize=7, markerscale=3)
    ax.set_title(f"{'+'.join(members[:3])}-enriched features "
                 f"(thr {thr:.3f})")
    png = wd / f"{out_prefix}_features.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    res["png"] = str(png)
    return _jsonable(res)


def interface_profile(pos_path: str, rrng_path: str, species,
                      point_nm, normal, half_width_nm: float = 8.0,
                      bin_nm: float = 0.3, roi_radius_nm: float = 12.0,
                      efficiency: float | None = None,
                      n_shuffles: int = 5,
                      out_prefix: str = "interface",
                      workdir: str = ".") -> dict:
    """1D composition profile across a planar interface + Gibbsian
    interfacial excess by two independent routes, with gates.

    Profile: target-group fraction vs signed distance along the normal
    inside a cylinder of roi_radius about the plane point. Excess:
    (A) integration of (c - c_bulk) x ions per bin / area; (B)
    Krakauer-Seidman ladder - cumulative solute vs cumulative ions,
    far-field lines fitted on both sides, excess = their offset / area.
    Gates: far-field plateau = bulk; |A - B| within tolerance; excess
    collapses under label shuffling (feature-blind null)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
    gmask, members = _target_mask(labels, species)
    n, u, v = _plane_basis(normal)
    rel = xyz - np.asarray(point_nm, float)
    d = rel @ n
    r_perp = np.linalg.norm(rel - np.outer(d, n), axis=1)
    roi = (r_perp <= roi_radius_nm) & (np.abs(d) <= half_width_nm)
    area = np.pi * roi_radius_nm ** 2
    edges = np.arange(-half_width_nm, half_width_nm + bin_nm, bin_nm)
    mids = 0.5 * (edges[:-1] + edges[1:])

    def excess_integration(mask_t):
        n_tot = np.histogram(d[roi], bins=edges)[0]
        n_sol = np.histogram(d[roi & mask_t], bins=edges)[0]
        far = np.abs(mids) >= 0.6 * half_width_nm
        c_bulk = n_sol[far].sum() / max(n_tot[far].sum(), 1)
        prof = np.where(n_tot > 0, n_sol / np.maximum(n_tot, 1), np.nan)
        err = np.sqrt(np.maximum(n_sol, 1)) / np.maximum(n_tot, 1)
        gamma = float((n_sol - c_bulk * n_tot).sum() / area)
        return prof, err, c_bulk, gamma, n_tot, n_sol

    prof, err, c_bulk, g_int, n_tot, n_sol = excess_integration(gmask)

    # ladder (Krakauer-Seidman)
    order = np.argsort(d[roi])
    d_s = d[roi][order]
    cum_tot = np.arange(1, len(d_s) + 1, dtype=float)
    cum_sol = np.cumsum(gmask[roi][order].astype(float))
    left = d_s <= -0.6 * half_width_nm
    right = d_s >= 0.6 * half_width_nm
    g_lad = None
    if left.sum() > 50 and right.sum() > 50:
        sl, il = np.polyfit(cum_tot[left], cum_sol[left], 1)
        sr, ir = np.polyfit(cum_tot[right], cum_sol[right], 1)
        s_mean = 0.5 * (sl + sr)
        # offset between parallel far-field lines evaluated at the
        # interface crossing (cumulative count at d = 0)
        i0 = int(np.searchsorted(d_s, 0.0))
        off = ((cum_sol[right] - s_mean * cum_tot[right]).mean()
               - (cum_sol[left] - s_mean * cum_tot[left]).mean())
        g_lad = float(off / area)

    # feature-blind null: same ROI, permuted labels
    rng = np.random.RandomState(21)
    g_null = []
    for _ in range(n_shuffles):
        m = np.zeros(len(xyz), bool)
        m[rng.choice(len(xyz), int(gmask.sum()), replace=False)] = True
        g_null.append(excess_integration(m)[3])
    gm, gs = float(np.mean(g_null)), float(np.std(g_null))
    sig_poisson = float(np.sqrt(max(n_sol.sum(), 1)) / area)

    # gates
    far = np.abs(mids) >= 0.6 * half_width_nm
    bulk_direct = float(gmask[(r_perp <= roi_radius_nm)
                              & (np.abs(d) > half_width_nm)].mean()) \
        if ((r_perp <= roi_radius_nm) & (np.abs(d) > half_width_nm)).any() \
        else float(gmask.mean())
    ff = float(np.nanmean(prof[far]))
    ff_err = float(np.nanmean(err[far]))
    gates = {
        "far_field": {"profile_far_field": round(ff, 4),
                      "bulk_direct": round(bulk_direct, 4),
                      "passed": bool(abs(ff - bulk_direct)
                                     < max(3 * ff_err, 0.1 * bulk_direct))},
        "method_agreement": {
            "gamma_integration": round(g_int, 3),
            "gamma_ladder": (round(g_lad, 3) if g_lad is not None else None),
            "passed": bool(g_lad is not None and abs(g_int - g_lad)
                           <= max(0.25 * abs(g_int), 3 * sig_poisson))},
        # two-sided: DEPLETION (negative excess, e.g. Cr at RIS
        # boundaries) is as much a finding as enrichment
        "shuffle_null": {"gamma_null_mean": round(gm, 3),
                         "gamma_null_std": round(gs, 3),
                         "z": round((g_int - gm) / max(gs, sig_poisson,
                                                       1e-9), 1),
                         "passed": bool(abs(g_int - gm)
                                        > 3 * max(gs, sig_poisson))}}
    # extremum and width on the DEVIATION from bulk (sign-agnostic)
    dev = np.where(np.isfinite(prof), prof - c_bulk, 0.0)
    ipk = int(np.argmax(np.abs(dev)))
    above = np.where(np.abs(dev) >= 0.5 * abs(dev[ipk]))[0]
    fwhm = float((above.max() - above.min() + 1) * bin_nm) if len(above) \
        else None
    res = {"species": members, "point_nm": list(map(float, point_nm)),
           "normal": [round(float(c), 4) for c in n],
           "roi_radius_nm": roi_radius_nm, "area_nm2": round(area, 1),
           "n_ions_roi": int(roi.sum()),
           "c_bulk_far_field": round(c_bulk, 4),
           "peak_fraction": round(float(prof[ipk]), 4),
           "peak_position_nm": round(float(mids[ipk]), 2),
           "sign": ("enrichment" if dev[ipk] > 0 else "depletion"),
           "fwhm_nm": (round(fwhm, 2) if fwhm else None),
           "gamma_detected_per_nm2": round(g_int, 3),
           "gamma_ladder_per_nm2": (round(g_lad, 3) if g_lad is not None
                                    else None),
           "gamma_poisson_sigma": round(sig_poisson, 3),
           "gamma_corrected_atoms_per_nm2": (
               round(g_int / efficiency, 3) if efficiency else None),
           "efficiency_used": efficiency,
           "gates": gates,
           "profile": {"d_nm": [round(float(m), 2) for m in mids],
                       "fraction": [round(float(p), 4)
                                    if np.isfinite(p) else None
                                    for p in prof]}}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.errorbar(mids, prof * 100, yerr=err * 100, marker="o", ms=3,
                 lw=1, color=_COLORS[0], label="+".join(members[:3]))
    ax1.axhline(bulk_direct * 100, color="gray", ls="--",
                label="bulk (direct)")
    ax1.axvline(0, color="gray", lw=0.8)
    ax1.set_xlabel("distance along normal [nm]")
    ax1.set_ylabel("ionic fraction [%]")
    ax1.legend(fontsize=8)
    ax1.set_title(f"interface profile - far-field gate "
                  f"{'PASSED' if gates['far_field']['passed'] else 'FAILED'}")
    ax2.plot(cum_tot, cum_sol, color=_COLORS[0], lw=1)
    if g_lad is not None:
        for sel, col in ((left, _COLORS[3]), (right, _COLORS[1])):
            ax2.plot(cum_tot[sel], np.polyval(
                np.polyfit(cum_tot[sel], cum_sol[sel], 1), cum_tot[sel]),
                color=col, lw=1.2, ls="--")
    ax2.set_xlabel("cumulative ions (along normal)")
    ax2.set_ylabel("cumulative solute ions")
    ax2.set_title(f"Krakauer-Seidman ladder: Gamma {g_int:.2f} (int) vs "
                  f"{g_lad if g_lad is None else round(g_lad, 2)} (ladder) "
                  f"/nm^2")
    png = wd / f"{out_prefix}_profile.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    res["png"] = str(png)
    return _jsonable(res)


def inplane_map(pos_path: str, rrng_path: str, species, point_nm, normal,
                slab_half_width_nm: float = 2.0, pixel_nm: float = 1.5,
                extent_nm: float = 15.0, min_ions: int = 15,
                out_prefix: str = "inplane", workdir: str = ".") -> dict:
    """2D in-plane concentration map of the target group inside the
    interface slab - reveals whether segregation is uniform across the
    boundary or patchy (precipitate decoration, facets)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xyz, labels = _load_ranged_cloud(pos_path, rrng_path)
    gmask, members = _target_mask(labels, species)
    n, u, v = _plane_basis(normal)
    rel = xyz - np.asarray(point_nm, float)
    d = rel @ n
    slab = np.abs(d) <= slab_half_width_nm
    pu, pv = rel[slab] @ u, rel[slab] @ v
    bins = np.arange(-extent_nm, extent_nm + pixel_nm, pixel_nm)
    H_all, _, _ = np.histogram2d(pu, pv, bins=[bins, bins])
    H_g, _, _ = np.histogram2d(pu[gmask[slab]], pv[gmask[slab]],
                               bins=[bins, bins])
    with np.errstate(invalid="ignore"):
        f = np.where(H_all >= min_ions, H_g / np.maximum(H_all, 1), np.nan)
    vals = f[np.isfinite(f)]
    poisson = float(np.nanmean(np.sqrt(np.maximum(H_g, 1))
                               / np.maximum(H_all, 1)))
    res = {"species": members, "slab_half_width_nm": slab_half_width_nm,
           "pixel_nm": pixel_nm, "n_pixels": int(len(vals)),
           "mean_fraction": round(float(vals.mean()), 4),
           "std_fraction": round(float(vals.std()), 4),
           "poisson_floor": round(poisson, 4),
           "patchiness": round(float(vals.std() / max(poisson, 1e-9)), 2),
           "note": "patchiness ~1 = uniform within counting noise; >>1 = "
                   "laterally heterogeneous segregation"}
    fig, ax = plt.subplots(figsize=(6, 5.4))
    im = ax.imshow(f.T * 100, origin="lower", cmap="inferno",
                   extent=[bins[0], bins[-1], bins[0], bins[-1]])
    ax.set_xlabel("in-plane u [nm]")
    ax.set_ylabel("in-plane v [nm]")
    ax.set_title(f"in-plane {'+'.join(members[:2])} fraction [%] "
                 f"(patchiness {res['patchiness']})")
    plt.colorbar(im, ax=ax, shrink=0.85)
    png = wd / f"{out_prefix}_inplane.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    res["png"] = str(png)
    return _jsonable(res)


def family3_report(workdir: str, run: dict,
                   title: str = "Family-3 interface segregation",
                   hero: dict | None = None, score: dict | None = None,
                   assessment: dict | None = None, extra_note: str = "",
                   out_name: str = "report.html") -> dict:
    """Standard family-3 report (feature detection, profile + excess +
    gates, in-plane map) - PIPELINE-DEFAULT; ``run`` keys: feature /
    profile / inplane."""
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
    if assessment and not assessment.get("error"):
        s.append("<h2>Scientific assessment</h2>")
        for para in str(assessment.get("detailed_analysis", "")).split("\n\n"):
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
    if hero:
        s.append("<h2>Reconstruction - ion species (hero)</h2>")
        s.append(img(hero.get("png")))
    feat = run.get("feature") or {}
    if feat:
        s.append("<h2>Planar feature detection</h2>")
        rows = "".join(
            f"<tr><td>{f['rank']}</td><td>{f['planarity']}</td>"
            f"<td>{f['normal']}</td><td>{f['point_nm']}</td>"
            f"<td>{f['depth_below_apex_nm']}</td><td>{f['thickness_nm']}</td>"
            f"<td>{f['extent_nm']}</td><td>{f['enrichment_vs_baseline']}x"
            "</td></tr>" for f in feat.get("features", []))
        s.append("<table><tr><th>rank</th><th>planarity</th><th>normal</th>"
                 "<th>point [nm]</th><th>depth</th><th>thickness</th>"
                 "<th>extent</th><th>enrichment</th></tr>" + rows
                 + "</table>")
        s.append(img(feat.get("png")))
    prof = run.get("profile") or {}
    if prof:
        s.append("<h2>Interface profile + Gibbsian excess</h2>")
        s.append(kv(prof, ["species", "c_bulk_far_field", "peak_fraction",
                           "peak_position_nm", "fwhm_nm",
                           "gamma_detected_per_nm2", "gamma_ladder_per_nm2",
                           "gamma_poisson_sigma",
                           "gamma_corrected_atoms_per_nm2",
                           "efficiency_used", "n_ions_roi"]))
        g = prof.get("gates", {})
        s.append("<p>Far-field = bulk: " + badge(g.get("far_field", {}).get("passed"))
                 + " · Integration vs ladder agreement: "
                 + badge(g.get("method_agreement", {}).get("passed"))
                 + " · Shuffle null: " + badge(g.get("shuffle_null", {}).get("passed"))
                 + f" (z = {g.get('shuffle_null', {}).get('z')})</p>")
        s.append(img(prof.get("png")))
    ip = run.get("inplane") or {}
    if ip:
        s.append("<h2>In-plane map</h2>")
        s.append(kv(ip, ["mean_fraction", "std_fraction", "poisson_floor",
                         "patchiness"]))
        s.append(img(ip.get("png")))
    if score:
        s.append("<h2>Benchmark score (truth opened AFTER the blind run)"
                 "</h2>")
        s.append(kv(score, list(score.keys())[:14]))
    s.append("<footer>Generated by interface_tools.family3_report (scilink "
             "point_cloud_analysis / interface_segregation). Excess on the "
             "detected-ion basis unless efficiency-corrected; all claims "
             "gated (far field, method agreement, shuffle null).</footer>")
    out = wd / out_name
    out.write_text("\n".join(s))
    return {"html": str(out)}


_IMP = ("from scilink.skills.point_cloud_analysis.interface_segregation."
        "interface_tools import ")

TOOL_SPECS = [
    ToolSpec(
        name="detect_planar_feature",
        description=("Find solute-enriched PLANAR features (boundaries, "
                     "interfaces) by PCA on enriched voxels: normal, "
                     "point, planarity, thickness, extent, depth. "
                     "top_is_planar=false means the enriched region is a "
                     "blob/column - do not run interface profiles on it."),
        parameters={"pos_path": {"type": "string", "description": "data"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "species": {"type": "string",
                                "description": "segregating ion labels/"
                                               "regex (whole ions)"}},
        required=["pos_path", "rrng_path", "species"],
        import_line=_IMP + "detect_planar_feature",
        signature=("detect_planar_feature(pos_path, rrng_path, species, "
                   "voxel_nm=1.0, delocalization_nm=1.0, k_sigma=2.0, "
                   "out_prefix='feature', workdir='.') -> dict"),
        agents=["simulation"],
        when_to_use=("FIRST family-3 step when no ROI/plane is given; its "
                     "top_feature point_nm + normal feed interface_profile."),
        returns=("features (ranked), top_feature {point_nm, normal, "
                 "planarity, thickness_nm, extent_nm}, top_is_planar, png"),
        example=("ft = detect_planar_feature('tip.apt', 't.rrng', '^Ni$', "
                 "workdir='out')"),
    ),
    ToolSpec(
        name="interface_profile",
        description=("1D composition profile across a planar interface + "
                     "Gibbsian interfacial excess by TWO independent routes "
                     "(profile integration; Krakauer-Seidman ladder) with "
                     "the family-3 ACCEPT GATES: far field = bulk, "
                     "route agreement, feature-blind shuffle null."),
        parameters={"pos_path": {"type": "string", "description": "data"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "species": {"type": "string",
                                "description": "segregating ion labels/regex"},
                    "point_nm": {"type": "array",
                                 "description": "a point on the plane"},
                    "normal": {"type": "array",
                               "description": "plane normal"},
                    "efficiency": {"type": "number",
                                   "description": "detection efficiency "
                                                  "(enables atoms/nm2 "
                                                  "correction)"}},
        required=["pos_path", "rrng_path", "species", "point_nm", "normal"],
        import_line=_IMP + "interface_profile",
        signature=("interface_profile(pos_path, rrng_path, species, "
                   "point_nm, normal, half_width_nm=8.0, bin_nm=0.3, "
                   "roi_radius_nm=12.0, efficiency=None, n_shuffles=5, "
                   "out_prefix='interface', workdir='.') -> dict"),
        agents=["simulation"],
        when_to_use=("MANDATORY for any segregation/excess claim - claim "
                     "ONLY when all three gates pass; report "
                     "gamma_detected_per_nm2 and, with efficiency, "
                     "gamma_corrected_atoms_per_nm2."),
        returns=("profile, c_bulk_far_field, peak, fwhm_nm, "
                 "gamma_detected_per_nm2, gamma_ladder_per_nm2, "
                 "gamma_corrected_atoms_per_nm2, gates{far_field, "
                 "method_agreement, shuffle_null}, png"),
        example=("pr = interface_profile('tip.apt', 't.rrng', '^Ni$', "
                 "ft['top_feature']['point_nm'], ft['top_feature']['normal'], "
                 "efficiency=0.37)"),
    ),
    ToolSpec(
        name="inplane_map",
        description=("2D in-plane concentration map inside the interface "
                     "slab - uniform vs patchy segregation (patchiness = "
                     "std / Poisson floor)."),
        parameters={"pos_path": {"type": "string", "description": "data"},
                    "rrng_path": {"type": "string", "description": ".rrng"},
                    "species": {"type": "string",
                                "description": "segregating ion labels/regex"},
                    "point_nm": {"type": "array", "description": "plane point"},
                    "normal": {"type": "array", "description": "plane normal"}},
        required=["pos_path", "rrng_path", "species", "point_nm", "normal"],
        import_line=_IMP + "inplane_map",
        signature=("inplane_map(pos_path, rrng_path, species, point_nm, "
                   "normal, slab_half_width_nm=2.0, pixel_nm=1.5, "
                   "extent_nm=15.0, out_prefix='inplane', workdir='.') "
                   "-> dict"),
        agents=["simulation"],
        when_to_use=("After interface_profile, to characterize lateral "
                     "homogeneity of the segregation."),
        returns="mean/std fraction, poisson_floor, patchiness, png",
        example=("ip = inplane_map('tip.apt', 't.rrng', '^Ni$', pt, nrm)"),
    ),
    ToolSpec(
        name="family3_report",
        description=("Assemble the standard family-3 HTML report (feature "
                     "table, profile + excess + gate verdicts, in-plane "
                     "map). PIPELINE-DEFAULT - every family-3 run ends "
                     "with this call."),
        parameters={"workdir": {"type": "string", "description": "run folder"},
                    "run": {"type": "object",
                            "description": "keys feature / profile / inplane"}},
        required=["workdir", "run"],
        import_line=_IMP + "family3_report",
        signature=("family3_report(workdir, run, title=..., hero=None, "
                   "score=None, assessment=None, extra_note='', "
                   "out_name='report.html') -> dict"),
        agents=["simulation"],
        when_to_use="ALWAYS, as the last step of any family-3 analysis.",
        returns="html (report path)",
        example=("rep = family3_report('out', {'feature': ft, 'profile': pr, "
                 "'inplane': ip}, hero=viz)"),
    ),
]
