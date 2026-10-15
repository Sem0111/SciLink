"""ccd_shuffle_control: the CCD label-shuffle null must reject the
community structure the clusterer carves out of a homogeneous cloud and
accept a genuinely enriched region.

Synthetic neighborhood csvs are built directly in the vendored csv
layout (counts, p<label>, d<label>, N_atoms, density, midpoint_*), so the
test exercises the null construction and the verdict rule without a
reconstruction or a KD-tree pass.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scilink.skills.point_cloud_analysis.apt_ccd.apt_tools import (  # noqa: E402
    TOOL_SPECS, _community_contiguity, ccd_shuffle_control)

LABELS = ["Co1", "Cr1", "Ni1"]


def _write_nb_csv(path: Path, n_rows: int, enriched: bool, seed: int = 1):
    rng = np.random.default_rng(seed)
    side = int(round(n_rows ** (1 / 3)))
    g = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"),
                 -1).reshape(-1, 3)[:n_rows].astype(float)
    n_atoms = rng.poisson(45, size=len(g))
    p = np.full(3, 1 / 3)
    counts = np.vstack([rng.multinomial(n, p) for n in n_atoms])
    if enriched:  # a compact Cr-rich block: ~18% of neighborhoods
        blk = (g[:, 0] < side * 0.56) & (g[:, 1] < side * 0.56) \
            & (g[:, 2] < side * 0.56)
        pe = np.array([0.15, 0.70, 0.15])
        counts[blk] = np.vstack([rng.multinomial(n, pe)
                                 for n in n_atoms[blk]])
    density = n_atoms / (4 / 3 * np.pi)
    df = pd.DataFrame({"index": np.arange(len(g))})
    for i, lab in enumerate(LABELS):
        df[lab] = counts[:, i]
    for i, lab in enumerate(LABELS):
        df[f"p{lab}"] = counts[:, i] / np.maximum(n_atoms, 1)
    for i, lab in enumerate(LABELS):
        df[f"d{lab}"] = counts[:, i] / np.maximum(n_atoms, 1) * density
    df["N_atoms"] = n_atoms
    df["density"] = density
    df["midpoint_x"], df["midpoint_y"], df["midpoint_z"] = g.T
    df["radius"] = 1.0
    df["id"] = np.arange(len(g))
    df["outer_layer"] = False
    df["second_layer"] = False
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("ccd_ctl")
    out = {}
    for kind in ("homogeneous", "enriched"):
        csv = _write_nb_csv(base / f"{kind}.csv", 1500, kind == "enriched")
        out[kind] = ccd_shuffle_control(str(csv), n_shuffles=3,
                                        savedir=str(base / f"ctl_{kind}"),
                                        seed=0)
    return out


def test_homogeneous_cloud_is_noise_partitioning(runs):
    r = runs["homogeneous"]
    assert r["observed"]["community_count"] >= 2, \
        "the clusterer should still carve a homogeneous cloud (the trap)"
    assert r["passed"] is False
    assert r["communities_above_control"] == []
    assert "noise partitioning" in r["verdict"]


def test_enriched_block_beats_control(runs):
    r = runs["enriched"]
    assert r["passed"] is True
    assert len(r["communities_above_control"]) >= 1
    assert r["observed"]["max_dev"] > r["control_summary"]["dev_threshold"]
    # The matrix communities legitimately read as Cr-depleted relative to a
    # bulk that contains the block, so several ids pass, and the vendored
    # consensus step merges the block with Cr-positive noise clusters
    # (diluting its composition) - so the block need not rank first. It
    # must be present, Cr-enriched, and still far above bulk.
    comms = r["observed"]["communities"]
    blocks = [comms[c] for c in r["communities_above_control"]
              if comms[c]["dev_ion"] == "Cr1"
              and comms[c]["dev_sign"] == "enriched"]
    assert blocks, "the seeded Cr-rich block must pass the control"
    assert blocks[0]["mean_fraction"]["Cr1"] > 0.45
    assert blocks[0]["n_neighborhoods"] >= 300  # contains the 337 block rows
    assert "Cr1 enriched" in r["verdict"]


def test_ks_alone_would_not_separate(runs):
    """The reason the verdict is composition-based: KS magnitudes of the
    homogeneous null overlap those of the enriched cloud."""
    h = runs["homogeneous"]["control_summary"]["max_abs_ks_mean"]
    assert h > 0.4


def test_control_preserves_counts_and_bulk(runs, tmp_path):
    r = runs["homogeneous"]
    sdir = Path(r["savedir"]) / "shuffle_0"
    scsv = next(sdir.glob("*_shuffle0.csv"))
    sdf = pd.read_csv(scsv)
    assert (sdf[LABELS].sum(axis=1) == sdf["N_atoms"]).all()
    bulk = sdf[LABELS].sum() / sdf[LABELS].sum().sum()
    assert np.allclose(bulk, 1 / 3, atol=0.01)
    assert np.allclose(sdf[[f"p{l}" for l in LABELS]].sum(axis=1)
                       [sdf["N_atoms"] > 0], 1.0)


def test_result_is_json_and_documents_null(runs):
    r = runs["homogeneous"]
    json.dumps(r)
    assert "multinomial" in r["null_model"]
    assert r["n_shuffles"] == 3
    assert Path(r["savedir"], "ccd_shuffle_control.json").exists()


def test_contiguity_reads_extended_xyz(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 10, size=(600, 3))
    lab = (xyz[:, 0] > 5).astype(int)  # two spatially separated halves
    p = tmp_path / "c.xyz"
    with p.open("w") as fh:
        fh.write(f"{len(xyz)}\n\n")
        for l, (x, y, z) in zip(lab, xyz):
            fh.write(f"{l} {x:.3f} {y:.3f} {z:.3f}\n")
    c = _community_contiguity(str(p))
    assert c["neighbour_same_label_frac"] > 0.85
    assert c["excess"] > 0.3


def test_toolspec_registered():
    spec = {s.name: s for s in TOOL_SPECS}["ccd_shuffle_control"]
    assert "simulation" in spec.agents
    assert "exact=True" in spec.when_to_use
    assert "communities_above_control" in spec.returns
    assert "ccd_shuffle_control" in \
        {s.name: s for s in TOOL_SPECS}["detect_segregation"].returns
