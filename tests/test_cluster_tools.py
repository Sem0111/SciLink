"""Offline regression tests for the family-1 cluster_analysis tools.

Analytic ground truth via the synthetic generator: a small Fe-Cu benchmark
with seeded clusters must be recovered blind (knee -> sweep -> detect ->
null gate), Warren-Cowley must read ~0 on random labels and negative on a
constructed ordered lattice, and the z-SDM crystallographic gate must pass
on a full-density lattice and fail after APT depth noise.

  python tests/test_cluster_tools.py   (needs numpy/scipy/sklearn/ase/apav)
"""
import tempfile
from pathlib import Path

import numpy as np

from scilink.skills.point_cloud_analysis.synthetic_apt import (
    build_lattice, make_cluster_benchmark, seed_solid_solution,
    write_apt_dataset)
from scilink.skills.point_cloud_analysis.cluster_analysis.cluster_tools import (
    knn_distance_stats, label_shuffle_null, msm_detect, msm_parameter_sweep,
    warren_cowley, zsdm)

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    wd = Path(tempfile.mkdtemp(prefix="cluster_tools_test_"))

    print("1) blind MSM recovery on a seeded-cluster benchmark:")
    bm = make_cluster_benchmark(str(wd), name="t1", seed=3,
                                size_nm=(15., 15., 30.), n_clusters=8)
    csv, rrng = bm["csv"], bm["rrng"]
    knee = knn_distance_stats(csv, rrng, "Cu", k=10, workdir=str(wd))
    check("k-NN distribution is bimodal", knee["bimodal_signal"])
    sw = msm_parameter_sweep(csv, rrng, "Cu", workdir=str(wd))
    check("stability plateau found", sw["plateau_found"])
    det = msm_detect(csv, rrng, "Cu", sw["d_max_recommended_nm"],
                     sw["n_min_recommended"], workdir=str(wd),
                     make_3d_html=False)
    n_seeded = bm["n_clusters_seeded"]
    check(f"recovered most seeded clusters ({det['n_clusters']}"
          f"/{n_seeded})", n_seeded - 2 <= det["n_clusters"] <= n_seeded + 1)
    fr = [c["target_fraction"] for c in det["clusters"]]
    check("cluster compositions near seeded 0.5 "
          f"(mean {np.mean(fr):.2f})", 0.35 <= np.mean(fr) <= 0.65)
    gate = label_shuffle_null(csv, rrng, "Cu",
                              sw["d_max_recommended_nm"],
                              sw["n_min_recommended"])
    check("label-shuffle null gate passed", gate["null_gate_passed"])

    print("2) null-model behavior on a cluster-free twin:")
    bm0 = make_cluster_benchmark(str(wd), name="t0", seed=3,
                                 size_nm=(15., 15., 30.), n_clusters=0)
    knee0 = knn_distance_stats(bm0["csv"], bm0["rrng"], "Cu", k=10,
                               workdir=str(wd))
    if knee0.get("bimodal_signal"):
        sw0 = msm_parameter_sweep(bm0["csv"], bm0["rrng"], "Cu",
                                  workdir=str(wd))
        g0 = label_shuffle_null(bm0["csv"], bm0["rrng"], "Cu",
                                sw0["d_max_recommended_nm"],
                                sw0["n_min_recommended"])
        check("no gate pass on cluster-free data",
              not g0["null_gate_passed"])
    else:
        check("no gate pass on cluster-free data (knee refused)", True)

    print("3) Warren-Cowley with shuffle control:")
    rng = np.random.default_rng(11)
    pos = build_lattice("bcc", 2.87, (8., 8., 8.))
    spc = seed_solid_solution(len(pos), "Fe", {"Cu": 0.05}, rng)
    full = write_apt_dataset(pos, spc, str(wd), "wc_rand", {"kind": "t"})
    wc = warren_cowley(pos_path=full["csv"], rrng_path=full["rrng"],
                       center_species="Cu", neighbor_species="Cu")
    check(f"alpha ~ 0 on random labels ({wc['alpha_per_shell'][0]:+.3f})",
          not wc["significant"][0])
    # B2-ordered CsCl arrangement: corner=Fe, body-center=Cu -> every first
    # shell neighbor of Cu is Fe => alpha_CuCu = 1 - 0/x_B = 1 (clustering
    # convention: +1 means NO Cu around Cu = perfect ordering signal)
    poso = build_lattice("bcc", 2.87, (6., 6., 6.))
    spo = np.where(np.arange(len(poso)) % 2 == 0, "Fe", "Cu").astype(object)
    ordered = write_apt_dataset(poso, spo, str(wd), "wc_b2", {"kind": "t"})
    wco = warren_cowley(pos_path=ordered["csv"], rrng_path=ordered["rrng"],
                        center_species="Cu", neighbor_species="Cu")
    check(f"B2 ordering detected (alpha {wco['alpha_per_shell'][0]:+.2f}, "
          "expect ~ +1 first shell)",
          wco["significant"][0] and wco["alpha_per_shell"][0] > 0.8)

    print("4) z-SDM crystallographic gate:")
    z_full = zsdm(full["csv"], full["rrng"], "Fe", max_centers=4000,
                  workdir=str(wd))
    check("plane signal on full-density lattice "
          f"(contrast {z_full['fft_peak_contrast']})",
          z_full["crystallographic_signal"])
    check("recovered bcc plane spacing "
          f"({z_full['plane_spacing_A']} A ~ a/2 = 1.435 A)",
          z_full["plane_spacing_A"] is not None
          and abs(z_full["plane_spacing_A"] - 1.435) < 0.1)
    z_deg = zsdm(csv, rrng, "Fe", max_centers=4000, workdir=str(wd))
    check("no plane signal after APT degradation "
          f"(contrast {z_deg['fft_peak_contrast']})",
          not z_deg["crystallographic_signal"])

    n_fail = sum(not v for v in results.values())
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
