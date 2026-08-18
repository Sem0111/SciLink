"""Synthetic APT ground-truth generator for validation and training.

Builds APT-realistic point clouds with KNOWN answers so cluster /
precipitate / segregation / composition analyses can be scored blind:

  perfect lattice + solutes/features  ->  APT degradation  ->  dataset
                                                               + truth json

Degradation model (parameterized, defaults typical of a modern LEAP):
  - detection efficiency (random ion loss), default 0.37
  - positional noise, anisotropic: sigma_lateral ~ 3 A, sigma_depth ~ 0.8 A
  - output as (x, y, z [nm], Da) csv + synthetic .rrng (the same contract
    the apt_ccd pipeline ingests), plus <name>_truth.json

This is validation infrastructure, not an agent-facing skill: no
TOOL_SPECS. Unit tests and benchmark builders import it directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_EFFICIENCY = 0.37
DEFAULT_SIGMA_LATERAL_A = 3.0
DEFAULT_SIGMA_DEPTH_A = 0.8


def build_lattice(structure: str = "bcc", a: float = 2.87,
                  size_nm: tuple = (30.0, 30.0, 60.0)) -> np.ndarray:
    """Perfect lattice positions (Angstrom) filling a box (nm input)."""
    lx, ly, lz = (v * 10.0 for v in size_nm)
    if structure == "bcc":
        basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    elif structure == "fcc":
        basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5],
                          [0, 0.5, 0.5]])
    else:
        raise ValueError("structure must be bcc or fcc")
    nx, ny, nz = (int(v // a) + 1 for v in (lx, ly, lz))
    cells = np.stack(np.meshgrid(np.arange(nx), np.arange(ny),
                                 np.arange(nz), indexing="ij"),
                     axis=-1).reshape(-1, 3)
    pos = ((cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)) * a
    keep = np.all(pos < [lx, ly, lz], axis=1)
    return pos[keep]


def seed_solid_solution(n_atoms: int, matrix: str, solutes: dict,
                        rng: np.random.Generator) -> np.ndarray:
    """Random species assignment: solutes = {symbol: atomic fraction}."""
    species = np.full(n_atoms, matrix, dtype=object)
    order = list(solutes.items())
    r = rng.random(n_atoms)
    lo = 0.0
    for sym, frac in order:
        species[(r >= lo) & (r < lo + frac)] = sym
        lo += frac
    return species


def seed_clusters(pos: np.ndarray, species: np.ndarray, rng,
                  n_clusters: int = 30, radius_A: tuple = (5.0, 15.0),
                  solute: str = "Cu", enrichment: float = 0.5,
                  margin_A: float = 20.0) -> list:
    """Enrich spherical regions in `solute` (relabeling matrix atoms).

    Returns ground-truth list [{center_A, radius_A, target_fraction,
    n_atoms_in_sphere, n_solute_after}]."""
    lo = pos.min(axis=0) + margin_A
    hi = pos.max(axis=0) - margin_A
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    truth = []
    for _ in range(n_clusters):
        c = lo + rng.random(3) * (hi - lo)
        r = rng.uniform(*radius_A)
        idx = np.array(tree.query_ball_point(c, r), dtype=int)
        if len(idx) < 10:
            continue
        flip = idx[rng.random(len(idx)) < enrichment]
        species[flip] = solute
        truth.append({"center_A": [round(float(v), 2) for v in c],
                      "radius_A": round(float(r), 2),
                      "target_fraction": enrichment,
                      "n_atoms_in_sphere": int(len(idx)),
                      "n_solute_after": int((species[idx] == solute).sum())})
    return truth


def seed_planar_segregation(pos: np.ndarray, species: np.ndarray, rng,
                            plane_z_A: float, width_A: float = 6.0,
                            solute: str = "Ni",
                            excess_fraction: float = 0.15) -> dict:
    """Enrich a z-slab (a synthetic boundary) in `solute`."""
    m = np.abs(pos[:, 2] - plane_z_A) < width_A / 2
    idx = np.where(m)[0]
    flip = idx[rng.random(len(idx)) < excess_fraction]
    species[flip] = solute
    return {"plane_z_A": plane_z_A, "width_A": width_A, "solute": solute,
            "added_fraction": excess_fraction, "n_enriched": int(len(flip))}


def apt_degrade(pos: np.ndarray, species: np.ndarray, rng,
                efficiency: float = DEFAULT_EFFICIENCY,
                sigma_lateral_A: float = DEFAULT_SIGMA_LATERAL_A,
                sigma_depth_A: float = DEFAULT_SIGMA_DEPTH_A):
    """Random detection loss + anisotropic positional noise (z = depth)."""
    keep = rng.random(len(pos)) < efficiency
    p = pos[keep].copy()
    s = species[keep].copy()
    p[:, 0] += rng.normal(0, sigma_lateral_A, len(p))
    p[:, 1] += rng.normal(0, sigma_lateral_A, len(p))
    p[:, 2] += rng.normal(0, sigma_depth_A, len(p))
    return p, s


def write_apt_dataset(pos_A: np.ndarray, species: np.ndarray, out_dir: str,
                      name: str, truth: dict) -> dict:
    """Write (x,y,z[nm],Da) csv + synthetic .rrng + truth json."""
    from ase.data import atomic_masses, atomic_numbers

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    syms = sorted(set(species.tolist()))
    mass_of = {s: float(atomic_masses[atomic_numbers[s]]) for s in syms}
    da = np.array([mass_of[s] for s in species])
    np.savetxt(out / f"{name}.csv",
               np.column_stack([pos_A / 10.0, da]), delimiter=",",
               fmt="%.5f")
    lines = ["[Ions]", f"Number={len(syms)}"]
    lines += [f"Ion{i}={s}" for i, s in enumerate(syms, 1)]
    lines += ["[Ranges]", f"Number={len(syms)}"]
    lines += [f"Range{i}={mass_of[s]-0.4:.2f} {mass_of[s]+0.4:.2f} "
              f"Vol:0.0 Name:{s} Color:836EAA" for i, s in enumerate(syms, 1)]
    (out / f"{name}.rrng").write_text("\n".join(lines) + "\n")
    comp = {s: round(float((species == s).mean()) * 100, 3) for s in syms}
    truth = {**truth, "final_composition_at_pct": comp,
             "n_ions": int(len(pos_A))}
    (out / f"{name}_truth.json").write_text(json.dumps(truth, indent=1))
    return {"csv": str(out / f"{name}.csv"),
            "rrng": str(out / f"{name}.rrng"),
            "truth": str(out / f"{name}_truth.json"), **truth}


def make_cluster_benchmark(out_dir: str, name: str = "synth_clusters",
                           seed: int = 7, structure: str = "bcc",
                           a: float = 2.87, size_nm=(30., 30., 60.),
                           matrix: str = "Fe",
                           solutes: dict | None = None,
                           n_clusters: int = 30,
                           enrichment: float = 0.5,
                           efficiency: float = DEFAULT_EFFICIENCY,
                           sigma_lateral_A: float = DEFAULT_SIGMA_LATERAL_A,
                           sigma_depth_A: float = DEFAULT_SIGMA_DEPTH_A) -> dict:
    """One-call family-1 benchmark: Fe-1.5Cu-like matrix with seeded
    Cu-rich clusters, APT-degraded, with full ground truth."""
    rng = np.random.default_rng(seed)
    solutes = solutes if solutes is not None else {"Cu": 0.015}
    pos = build_lattice(structure, a, size_nm)
    species = seed_solid_solution(len(pos), matrix, solutes, rng)
    clusters = seed_clusters(pos, species, rng, n_clusters=n_clusters,
                             solute=list(solutes)[0], enrichment=enrichment)
    p, s = apt_degrade(pos, species, rng, efficiency, sigma_lateral_A,
                       sigma_depth_A)
    return write_apt_dataset(p, s, out_dir, name, {
        "kind": "cluster_benchmark", "seed": seed,
        "structure": structure, "lattice_a_A": a,
        "matrix": matrix, "solutes_nominal": solutes,
        "clusters": clusters, "n_clusters_seeded": len(clusters),
        "degradation": {"efficiency": efficiency,
                        "sigma_lateral_A": sigma_lateral_A,
                        "sigma_depth_A": sigma_depth_A}})
