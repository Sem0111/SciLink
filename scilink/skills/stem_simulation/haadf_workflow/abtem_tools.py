"""abTEM HAADF-STEM forward simulation tools (simulation-to-experiment bridge).

Structure file in -> HAADF image + metadata out, shaped so the image_analysis
skills (atomic_stem, crystalline_deformation) consume the result directly:
the metadata JSON carries ``fov_nm`` and ``pixel_size_nm``.

Input handling:
  - any ASE-readable format (CIF, POSCAR/CONTCAR, XDATCAR, extxyz, .traj, ...)
  - LAMMPS data files: atom types are anonymous integers; species are
    inferred by matching the Masses section against atomic masses, with an
    explicit ``type_map`` override.
  - multi-frame files: a single frame is selected via ``frame_index``
    (recorded in the output metadata). Ensemble/series semantics are
    deliberately out of scope here.

Two preparation branches:
  - ``prepare_md_slab``   : large / defected / triclinic MD snapshots ->
                            reorient beam axis to z, crop an interior
                            orthogonal ROI, trim ragged tilted edges.
  - ``build_ideal_slab``  : small ideal cells (CIF/POSCAR) -> orient the
                            requested zone axis along the beam, orthogonalize,
                            tile to a target field of view and thickness.

GPU/physics guardrails learned the hard way (encoded in ``plan_haadf_detector``
and ``simulate_haadf``): the detector outer angle is capped at 98% of the
antialias limit (the realized per-frame grid cutoff sits below the nominal
estimate by an extent-dependent amount), and out-of-memory failures walk a
fallback ladder (potential sampling 0.04 -> 0.05 A, then PRISM interpolation
-> 6, noting the accuracy tradeoff).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from ase.io import read

from scilink.skills._shared._spec import ToolSpec

# Electron wavelength (A) from acceleration voltage (V), relativistic.
def _wavelength_A(energy_kev: float) -> float:
    v0 = energy_kev * 1e3
    return 12.2643 / np.sqrt(v0 * (1 + 0.97845e-6 * v0))


# ---------------------------------------------------------------------------
# Input reading (branch-independent)
# ---------------------------------------------------------------------------

def _looks_like_lammps_data(path: Path) -> bool:
    try:
        head = path.read_text(errors="ignore")[:4096].lower()
    except OSError:
        return False
    return " atoms" in head and " atom types" in head


def _lammps_masses_to_symbols(path: Path, n_types: int) -> dict[int, str]:
    """Infer {type: element} from the Masses section by nearest atomic mass."""
    lines = path.read_text(errors="ignore").splitlines()
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.strip().startswith("Masses"))
    except StopIteration:
        raise ValueError(
            f"{path.name}: no Masses section - pass type_map explicitly, "
            "e.g. type_map={1: 'Ni'}")
    mapping: dict[int, str] = {}
    for line in lines[start + 1:]:
        s = line.split("#")[0].strip()
        if not s:
            if mapping:
                break
            continue
        parts = s.split()
        if not parts[0].isdigit():
            break
        t, mass = int(parts[0]), float(parts[1])
        diffs = np.abs(atomic_masses[1:] - mass)
        z = int(np.argmin(diffs)) + 1
        if diffs[z - 1] > 0.5:
            raise ValueError(
                f"{path.name}: type {t} mass {mass} matches no element within "
                "0.5 amu - pass type_map explicitly")
        mapping[t] = chemical_symbols[z]
    if len(mapping) < n_types:
        raise ValueError(
            f"{path.name}: Masses section covers {len(mapping)}/{n_types} "
            "types - pass type_map explicitly")
    return mapping


def read_structure(structure_path: str, type_map: dict | None = None,
                   frame_index: int = -1,
                   lammps_atom_style: str = "atomic") -> tuple[Atoms, dict]:
    """Read any supported structure file into ASE Atoms with real species.

    Returns ``(atoms, info)`` where info records the detected format, the
    frame selected for multi-frame inputs, and the element mapping used.
    """
    path = Path(structure_path)
    info: dict = {"path": str(path), "frame_index": None, "type_map": None}

    if _looks_like_lammps_data(path):
        atoms = read(path, format="lammps-data", atom_style=lammps_atom_style)
        types = atoms.get_atomic_numbers()  # type ids, not real Z
        n_types = int(types.max())
        mapping = ({int(k): str(v) for k, v in type_map.items()}
                   if type_map else _lammps_masses_to_symbols(path, n_types))
        atoms.set_chemical_symbols([mapping[int(t)] for t in types])
        info.update(format="lammps-data", type_map=mapping)
        return atoms, info

    images = read(path, index=":")
    if isinstance(images, Atoms):
        images = [images]
    n = len(images)
    idx = frame_index if frame_index >= 0 else n + frame_index
    atoms = images[idx]
    info.update(format=path.suffix.lstrip(".") or "auto", n_frames=n,
                frame_index=idx)
    if type_map:  # e.g. LAMMPS dump read via ase with anonymous types
        types = atoms.get_atomic_numbers()
        atoms.set_chemical_symbols(
            [str(type_map[int(t)]) for t in types])
        info["type_map"] = {int(k): str(v) for k, v in type_map.items()}
    return atoms, info


# ---------------------------------------------------------------------------
# Branch A: MD snapshot -> cropped orthogonal slab
# ---------------------------------------------------------------------------

_BEAM_PERMUTATION = {"x": (1, 2, 0), "y": (2, 0, 1), "z": (0, 1, 2)}


def _rotation_to_z(direction) -> np.ndarray:
    """Rotation matrix taking ``direction`` (original frame) onto +z."""
    n = np.asarray(direction, float)
    n = n / np.linalg.norm(n)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return np.vstack([u, v, n])  # rows = new basis in original coords


def prepare_md_slab(atoms: Atoms, beam_axis: str = "x",
                    roi: dict | None = None, trim_percentile: float = 3.0,
                    vacuum: float = 4.0, beam_direction=None,
                    fov_A: tuple | None = None,
                    beam_thickness_A: float | None = None,
                    in_plane_vacuum_A: float = 0.0) -> Atoms:
    """Crop + reorient an MD snapshot so the beam travels along abTEM z.

    ``roi`` gives axis-keyed windows in the ORIGINAL simulation frame, e.g.
    ``{"z": (63.0, 183.0)}``. Two orientation modes:

    - ``beam_axis`` ("x"/"y"/"z"): right-handed permutation maps that axis
      onto z (exact, no interpolation) - the default.
    - ``beam_direction`` (any 3-vector in the original frame, e.g.
      ``(1, 1, 1)`` for a BCC [111] view of a cube-aligned crystal):
      positions are rotated so the vector lands on z. In this mode the
      final windows are set in the ROTATED frame: ``fov_A`` crops central
      lateral windows and ``beam_thickness_A`` a central window along the
      beam (both recommended - a rotated chunk has no natural box).

    Ragged edges of tilted (triclinic) cells are trimmed on in-plane axes
    by ``trim_percentile`` (skipped for axes already windowed by roi/fov);
    the beam direction is padded with ``vacuum`` A. The interior
    orthogonal-crop approach sidesteps shear-correcting tilted cells; keep
    analysis regions > probe radius away from the cropped faces.
    """
    p = atoms.get_positions()
    sym = np.asarray(atoms.get_chemical_symbols())
    mask = np.ones(len(p), bool)
    for ax, (lo, hi) in (roi or {}).items():
        i = "xyz".index(ax)
        mask &= (p[:, i] > lo) & (p[:, i] < hi)
    if beam_direction is not None:
        q = p[mask] @ _rotation_to_z(beam_direction).T
        sym = sym[mask]
        skip_trim = set()
    else:
        if beam_axis not in _BEAM_PERMUTATION:
            raise ValueError(f"beam_axis must be x, y or z, got {beam_axis!r}")
        order = _BEAM_PERMUTATION[beam_axis]
        q = p[mask][:, order]
        sym = sym[mask]
        skip_trim = {k for k in (0, 1)
                     if order[k] in {"xyz".index(ax) for ax in (roi or {})}}
    if beam_thickness_A is not None:
        mid = 0.5 * (q[:, 2].min() + q[:, 2].max())
        keep = np.abs(q[:, 2] - mid) < beam_thickness_A / 2.0
        q, sym = q[keep], sym[keep]
    for k in (0, 1):
        if fov_A is not None:
            mid = 0.5 * (q[:, k].min() + q[:, k].max())
            keep = np.abs(q[:, k] - mid) < fov_A[k] / 2.0
            q, sym = q[keep], sym[keep]
        elif k not in skip_trim:
            lo, hi = np.percentile(q[:, k],
                                   [trim_percentile, 100 - trim_percentile])
            keep = (q[:, k] > lo) & (q[:, k] < hi)
            q, sym = q[keep], sym[keep]
    if len(q) == 0:
        raise ValueError("ROI/trim removed every atom - check the roi axes "
                         "and windows against the input cell")
    q = q - q.min(axis=0)
    # lateral vacuum frames finite objects (tips, particles) so the surface
    # outline sits inside the field of view instead of touching the border
    q[:, :2] += in_plane_vacuum_A
    slab = Atoms(symbols=list(sym), positions=q)
    lx, ly, lz = q.max(axis=0)
    slab.set_cell([lx + in_plane_vacuum_A, ly + in_plane_vacuum_A,
                   lz + vacuum])
    slab.center(axis=2)
    return slab


# ---------------------------------------------------------------------------
# Branch B: ideal cell (CIF/POSCAR) -> oriented, tiled slab
# ---------------------------------------------------------------------------

def build_ideal_slab(atoms: Atoms, zone_axis: tuple = (1, 1, 0),
                     min_fov_A: tuple = (30.0, 30.0),
                     thickness_A: float = 50.0,
                     vacuum: float = 2.0) -> Atoms:
    """Orient an ideal crystal so ``zone_axis`` runs along the beam and tile
    it to at least ``min_fov_A`` laterally and ``thickness_A`` along the beam.

    Uses ``ase.build.surface`` (stacking normal = ``zone_axis``; for cubic
    cells the [uvw] zone axis and (uvw) plane normal coincide - for
    non-cubic systems supply a pre-oriented cell and skip this helper) and
    abTEM's ``orthogonalize_cell`` for the in-plane cell.
    """
    from ase.build import surface

    slab = surface(atoms, tuple(int(v) for v in zone_axis), layers=1,
                   periodic=True)
    # beam (stacking) axis is now z; make the in-plane cell orthogonal
    cell = slab.cell.array
    off_diag = np.abs(cell - np.diag(np.diag(cell))).max()
    if off_diag > 1e-6:
        try:
            from abtem.atoms import orthogonalize_cell
        except ImportError as exc:
            raise ImportError(
                "cell is non-orthogonal after orientation and abtem is not "
                "installed to orthogonalize it - pip install abtem") from exc
        slab = orthogonalize_cell(slab)
    reps = (int(np.ceil(min_fov_A[0] / slab.cell[0, 0])),
            int(np.ceil(min_fov_A[1] / slab.cell[1, 1])),
            int(np.ceil(thickness_A / slab.cell[2, 2])))
    slab = slab * reps
    cell = slab.cell.array.copy()
    cell[2, 2] += vacuum
    slab.set_cell(cell)
    slab.center(axis=2)
    return slab


# ---------------------------------------------------------------------------
# Detector planning (antialias guardrail)
# ---------------------------------------------------------------------------

def plan_haadf_detector(energy_kev: float, inner_mrad: float,
                        outer_mrad: float,
                        pot_sampling_A: float = 0.04) -> dict:
    """Effective HAADF annulus for a given potential sampling.

    abTEM refuses detector angles beyond the antialias cutoff (2/3 of the
    Nyquist angle of the potential grid). The realized per-grid cutoff sits
    slightly BELOW the nominal-sampling estimate (grid-point rounding varies
    with cell extent), so the outer angle is capped at 98% of the analytic
    limit - a fixed integer, identical for every frame of a series.
    """
    lam = _wavelength_A(energy_kev)
    max_sim = (2.0 / 3.0) * lam / (2.0 * pot_sampling_A) * 1000.0
    outer_eff = min(float(outer_mrad), float(int(max_sim * 0.98)))
    if outer_eff <= inner_mrad:
        raise ValueError(
            f"antialias limit {max_sim:.0f} mrad caps the outer angle below "
            f"the inner angle {inner_mrad} mrad - decrease pot_sampling_A")
    return {"wavelength_A": lam, "antialias_limit_mrad": max_sim,
            "inner_mrad": float(inner_mrad), "outer_mrad_nominal": float(outer_mrad),
            "outer_mrad_effective": outer_eff,
            "pot_sampling_A": float(pot_sampling_A)}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

_OOM_MARKERS = ("outofmemory", "out of memory", "cudaerrormemoryallocation",
                "memoryerror")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import cupy  # noqa: F401
        return "gpu"
    except ImportError:
        print("simulate_haadf: no cupy/GPU found - running on CPU "
              "(expect minutes-to-hours instead of seconds-to-minutes)")
        return "cpu"


def simulate_haadf(atoms: Atoms, microscope: dict, pot_sampling_A: float = 0.04,
                   scan_sampling_A: float = 0.2, slice_thickness_A: float = 1.0,
                   prism_interpolation: int = 4, device: str = "auto",
                   out_prefix: str = "haadf", workdir: str = ".") -> dict:
    """Run a PRISM HAADF scan and write image + analysis-ready metadata.

    ``microscope`` maps 1:1 onto experimental (e.g. Velox) metadata::

        {"energy_kev": 300, "convergence_mrad": 25.0,
         "haadf_inner_mrad": 65, "haadf_outer_mrad": 200, "defocus_A": 0.0}

    Writes ``<out_prefix>.npy`` (raw image, x-first transposed to rows=y),
    ``<out_prefix>.png`` and ``<out_prefix>_meta.json`` (with ``fov_nm`` and
    ``pixel_size_nm`` - the contract the image_analysis skills consume).
    Out-of-memory errors walk the fallback ladder
    (0.04 -> 0.05 A sampling, then PRISM interpolation 6).
    """
    import abtem

    dev = _resolve_device(device)
    abtem.config.set({"device": dev})
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    ladder = [(pot_sampling_A, prism_interpolation),
              (max(pot_sampling_A, 0.05), prism_interpolation),
              (max(pot_sampling_A, 0.05), 6)]
    last_err: Exception | None = None
    for attempt, (pot, interp) in enumerate(ladder):
        try:
            return _simulate_once(atoms, microscope, pot, scan_sampling_A,
                                  slice_thickness_A, interp, out_prefix,
                                  workdir, attempt)
        except Exception as exc:  # noqa: BLE001 - inspect for OOM, else re-raise
            name = f"{type(exc).__name__}: {exc}".lower()
            if not any(m in name for m in _OOM_MARKERS):
                raise
            last_err = exc
            print(f"simulate_haadf: OOM at sampling={pot}, interp={interp}; "
                  "stepping down the memory ladder")
    raise MemoryError(
        f"HAADF simulation exhausted the OOM fallback ladder: {last_err}")


def _simulate_once(atoms, microscope, pot_sampling, scan_sampling,
                   slice_thickness, interp, out_prefix, workdir, attempt):
    import abtem

    det = plan_haadf_detector(microscope["energy_kev"],
                              microscope["haadf_inner_mrad"],
                              microscope["haadf_outer_mrad"], pot_sampling)
    potential = abtem.Potential(atoms, sampling=pot_sampling,
                                slice_thickness=slice_thickness,
                                projection="infinite",
                                parametrization="kirkland")
    detector = abtem.AnnularDetector(inner=det["inner_mrad"],
                                     outer=det["outer_mrad_effective"])
    scan = abtem.GridScan(start=(0, 0), end=potential.extent,
                          sampling=scan_sampling)
    s_matrix = abtem.SMatrix(potential=potential,
                             energy=microscope["energy_kev"] * 1e3,
                             semiangle_cutoff=microscope["convergence_mrad"],
                             interpolation=interp)
    measurement = s_matrix.scan(scan=scan, detectors=detector).compute()
    if hasattr(measurement, "to_cpu"):
        measurement = measurement.to_cpu()
    img = np.asarray(measurement.array)

    np.save(workdir / f"{out_prefix}.npy", img)   # raw - blur is display-side
    # Source-size blur for the quicklook PNG: from probe_size_pm (FWHM) when
    # the microscope dict carries it, else a 0.35 A default.
    if "source_size_A" in microscope:
        blur_A = float(microscope["source_size_A"])
    elif "probe_size_pm" in microscope:
        blur_A = float(microscope["probe_size_pm"]) / 100.0 / 2.355
    else:
        blur_A = 0.35
    from scipy.ndimage import gaussian_filter
    _save_png(gaussian_filter(img.T.astype(float), blur_A / scan_sampling),
              workdir / f"{out_prefix}.png")
    extent = [float(v) for v in potential.extent]
    meta = {
        "microscope": dict(microscope),
        "detector": det,
        "fov_nm": [extent[0] / 10.0, extent[1] / 10.0],
        "pixel_size_nm": scan_sampling / 10.0,
        "extent_A": extent,
        "scan_sampling_A": scan_sampling,
        "pot_sampling_A": pot_sampling,
        "prism_interpolation": interp,
        "oom_ladder_attempt": attempt,
        "png_source_blur_A": blur_A,
        "npy_is_raw": True,
        "n_atoms": len(atoms),
        "note": ("PRISM interpolation 6 trades accuracy for memory: coarser "
                 "plane-wave sampling can subtly smooth fine HAADF contrast"
                 if interp >= 6 else ""),
    }
    with open(workdir / f"{out_prefix}_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def _save_png(img_rows_y, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6 * img_rows_y.shape[0] / img_rows_y.shape[1]))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(img_rows_y, cmap="gray", origin="lower",
              vmin=np.percentile(img_rows_y, 1),
              vmax=np.percentile(img_rows_y, 99.5))
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Structure-side defect map (ground truth, no image detection involved)
# ---------------------------------------------------------------------------

def structure_defect_map(atoms: Atoms, cluster_radius_A: float = 0.7,
                         min_atoms_per_column: int = 5,
                         out_prefix: str | None = None,
                         workdir: str = ".") -> dict:
    """Project a beam-oriented slab into atomic columns and classify them.

    Clusters projected atom positions into columns (single-linkage), then
    runs the crystalline_deformation skill's 2D-PTM classification and
    Center-of-Symmetry directly on the column centroids - exact defect maps
    from the structure itself, independent of any image or detector. Also
    the ground-truth generator for validating (or training) image-side
    column detection.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    from scilink.skills.image_analysis.crystalline_deformation.ptm_tools import (
        compute_cos, ptm_classify)

    xy = atoms.get_positions()[:, :2]
    tree = cKDTree(xy)
    pairs = tree.query_pairs(cluster_radius_A, output_type="ndarray")
    n = len(xy)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                     shape=(n, n))
    ncomp, labels = connected_components(adj, directed=False)
    counts = np.bincount(labels, minlength=ncomp)
    cents = np.zeros((ncomp, 2))
    for k in (0, 1):
        cents[:, k] = (np.bincount(labels, weights=xy[:, k], minlength=ncomp)
                       / counts)
    keep = counts >= min_atoms_per_column
    cents, counts = cents[keep], counts[keep]

    ptm = ptm_classify(cents[:, 0], cents[:, 1])
    cos = compute_cos(cents[:, 0], cents[:, 1])
    result = {
        "columns_xy_A": cents,
        "atoms_per_column": counts,
        "ptm": ptm,
        "cos": cos,
        "n_columns": int(len(cents)),
    }
    if out_prefix is not None:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        np.savez(workdir / f"{out_prefix}_columns.npz", xy=cents,
                 n_atoms=counts)
    return result


# ---------------------------------------------------------------------------
# Remote-execution script generation
# ---------------------------------------------------------------------------

_SCRIPT_TEMPLATE = '''\
#!/usr/bin/env python
"""Standalone abTEM HAADF run generated by scilink stem_simulation.

Requires: pip install scilink abtem ase (and cupy-cuda12x on a CUDA-12 GPU).
"""
from scilink.skills.stem_simulation.haadf_workflow.abtem_tools import (
    build_ideal_slab, prepare_md_slab, read_structure, simulate_haadf)

atoms, info = read_structure({structure_path!r}, type_map={type_map!r},
                             frame_index={frame_index!r})
print("read:", info)
{prep_line}
meta = simulate_haadf(slab, {microscope!r}, out_prefix={out_prefix!r},
                      workdir=".")
print("done:", meta["fov_nm"], "nm fov ->", {out_prefix!r} + ".npy/.png/_meta.json")
'''


def write_abtem_script(structure_path: str, microscope: dict,
                       out_script: str, branch: str = "md",
                       type_map: dict | None = None, frame_index: int = -1,
                       roi: dict | None = None, beam_axis: str = "x",
                       zone_axis: tuple = (1, 1, 0),
                       out_prefix: str = "haadf") -> str:
    """Emit a self-driving run script for execution on a (remote) GPU host.

    ``branch`` is ``"md"`` (prepare_md_slab with ``roi``/``beam_axis``) or
    ``"ideal"`` (build_ideal_slab with ``zone_axis``).
    """
    if branch == "md":
        prep = (f"slab = prepare_md_slab(atoms, beam_axis={beam_axis!r}, "
                f"roi={roi!r})")
    elif branch == "ideal":
        prep = f"slab = build_ideal_slab(atoms, zone_axis={tuple(zone_axis)!r})"
    else:
        raise ValueError("branch must be 'md' or 'ideal'")
    script = _SCRIPT_TEMPLATE.format(structure_path=structure_path,
                                     type_map=type_map,
                                     frame_index=frame_index,
                                     prep_line=prep, microscope=microscope,
                                     out_prefix=out_prefix)
    out = Path(out_script)
    out.write_text(script)
    return str(out)


# ---------------------------------------------------------------------------
# TOOL_SPECS
# ---------------------------------------------------------------------------

_MICROSCOPE_PARAM = {
    "type": "object",
    "description": ("Microscope parameters mapping 1:1 onto experimental "
                    "(Velox) metadata: energy_kev, convergence_mrad, "
                    "haadf_inner_mrad, haadf_outer_mrad, optional defocus_A."),
}

TOOL_SPECS = [
    ToolSpec(
        name="read_structure",
        description=("Read any ASE-supported structure file (CIF, POSCAR, "
                     "extxyz, trajectories, LAMMPS data) into Atoms with real "
                     "chemical species. LAMMPS types are inferred from the "
                     "Masses section or an explicit type_map; multi-frame "
                     "files select one frame via frame_index."),
        parameters={
            "structure_path": {"type": "string",
                               "description": "Path to the structure file."},
            "type_map": {"type": "object",
                         "description": "Optional {type_id: element} for "
                                        "LAMMPS inputs, e.g. {1: 'Ni'}."},
            "frame_index": {"type": "integer",
                            "description": "Frame for multi-frame inputs "
                                           "(default -1 = last)."},
        },
        required=["structure_path"],
        import_line=("from scilink.skills.stem_simulation.haadf_workflow"
                     ".abtem_tools import read_structure"),
        signature=("read_structure(structure_path, type_map=None, "
                   "frame_index=-1) -> tuple[Atoms, dict]"),
        agents=["simulation"],
        when_to_use=("First step of any HAADF simulation: load the structure "
                     "and resolve species before slab preparation."),
        returns="(atoms, info) - info records format, frame, element mapping.",
        example=("atoms, info = read_structure('3sigma_2', "
                 "type_map={1: 'Ni'})"),
    ),
    ToolSpec(
        name="prepare_md_slab",
        description=("Reorient + crop an MD snapshot for abTEM: beam axis to "
                     "z, interior orthogonal ROI crop (sidesteps triclinic "
                     "shear), percentile edge trim, vacuum padding."),
        parameters={
            "beam_axis": {"type": "string",
                          "description": "Original-frame axis the beam "
                                         "travels along: x, y or z."},
            "roi": {"type": "object",
                    "description": "Axis-keyed crop windows in the original "
                                   "frame, e.g. {'z': [63.0, 183.0]}."},
        },
        required=[],
        import_line=("from scilink.skills.stem_simulation.haadf_workflow"
                     ".abtem_tools import prepare_md_slab"),
        signature=("prepare_md_slab(atoms, beam_axis='x', roi=None, "
                   "trim_percentile=3.0, vacuum=4.0) -> Atoms"),
        agents=["simulation"],
        when_to_use=("Branch A: large/defected/triclinic MD snapshots "
                     "(LAMMPS data or dump frames)."),
        returns="Orthogonal slab Atoms with beam along z, vacuum-padded.",
        example=("slab = prepare_md_slab(atoms, beam_axis='x', "
                 "roi={'z': (63., 183.)})"),
    ),
    ToolSpec(
        name="build_ideal_slab",
        description=("Orient an ideal crystal (CIF/POSCAR unit cell) with a "
                     "zone axis along the beam, orthogonalize, and tile to a "
                     "target field of view and thickness."),
        parameters={
            "zone_axis": {"type": "array",
                          "description": "Zone axis, e.g. [1, 1, 0] "
                                         "(cubic cells)."},
            "min_fov_A": {"type": "array",
                          "description": "Minimum lateral field of view in "
                                         "Angstroms (default [30, 30])."},
            "thickness_A": {"type": "number",
                            "description": "Target beam-direction thickness "
                                           "(default 50 A)."},
        },
        required=[],
        import_line=("from scilink.skills.stem_simulation.haadf_workflow"
                     ".abtem_tools import build_ideal_slab"),
        signature=("build_ideal_slab(atoms, zone_axis=(1,1,0), "
                   "min_fov_A=(30,30), thickness_A=50) -> Atoms"),
        agents=["simulation"],
        when_to_use="Branch B: small ideal cells from CIF/POSCAR/DFT outputs.",
        returns="Tiled orthogonal slab Atoms with the zone axis along z.",
        example="slab = build_ideal_slab(read('Ni.cif'), zone_axis=(1,1,0))",
    ),
    ToolSpec(
        name="simulate_haadf",
        description=("PRISM HAADF-STEM simulation with GPU auto-detection, "
                     "antialias-capped detector annulus and an out-of-memory "
                     "fallback ladder. Writes image .npy/.png plus metadata "
                     "with fov_nm/pixel_size_nm for the image_analysis "
                     "skills."),
        parameters={
            "microscope": _MICROSCOPE_PARAM,
            "pot_sampling_A": {"type": "number",
                               "description": "Potential sampling (default "
                                              "0.04 A; sets the antialias "
                                              "ceiling on detector angles)."},
            "out_prefix": {"type": "string",
                           "description": "Output file prefix."},
        },
        required=["microscope"],
        import_line=("from scilink.skills.stem_simulation.haadf_workflow"
                     ".abtem_tools import simulate_haadf"),
        signature=("simulate_haadf(atoms, microscope, pot_sampling_A=0.04, "
                   "scan_sampling_A=0.2, prism_interpolation=4, "
                   "device='auto', out_prefix='haadf', workdir='.') -> dict"),
        agents=["simulation"],
        when_to_use=("After slab preparation, to render the HAADF image. "
                     "Needs abtem (+ cupy for GPU); on CPU it runs but "
                     "slowly - prefer write_abtem_script for remote GPUs."),
        returns="Metadata dict (also written as <out_prefix>_meta.json).",
        example=("meta = simulate_haadf(slab, {'energy_kev': 300, "
                 "'convergence_mrad': 25.0, 'haadf_inner_mrad': 65, "
                 "'haadf_outer_mrad': 200})"),
    ),
    ToolSpec(
        name="structure_defect_map",
        description=("Exact defect map from the structure itself: cluster "
                     "projected atoms into columns, then run the "
                     "crystalline_deformation 2D-PTM + Center-of-Symmetry on "
                     "the column centroids. No image or detector involved."),
        parameters={
            "cluster_radius_A": {"type": "number",
                                 "description": "Single-linkage projected "
                                                "clustering radius "
                                                "(default 0.7 A)."},
        },
        required=[],
        import_line=("from scilink.skills.stem_simulation.haadf_workflow"
                     ".abtem_tools import structure_defect_map"),
        signature=("structure_defect_map(atoms, cluster_radius_A=0.7, "
                   "min_atoms_per_column=5) -> dict"),
        agents=["simulation"],
        when_to_use=("For ground-truth defect maps of a simulated structure, "
                     "or to generate labels for validating/training "
                     "image-side column detection."),
        returns=("Dict with columns_xy_A, atoms_per_column, ptm and cos "
                 "results from the crystalline_deformation tools."),
        example="gt = structure_defect_map(slab)",
    ),
    ToolSpec(
        name="write_abtem_script",
        description=("Emit a standalone run script (read -> prepare -> "
                     "simulate) for execution on a remote GPU host, e.g. a "
                     "SageMaker training job or cluster node."),
        parameters={
            "structure_path": {"type": "string",
                               "description": "Structure file path."},
            "microscope": _MICROSCOPE_PARAM,
            "out_script": {"type": "string",
                           "description": "Where to write the script."},
            "branch": {"type": "string",
                       "description": "'md' (prepare_md_slab) or 'ideal' "
                                      "(build_ideal_slab)."},
        },
        required=["structure_path", "microscope", "out_script"],
        import_line=("from scilink.skills.stem_simulation.haadf_workflow"
                     ".abtem_tools import write_abtem_script"),
        signature=("write_abtem_script(structure_path, microscope, "
                   "out_script, branch='md', **prep_kwargs) -> str"),
        agents=["simulation"],
        when_to_use=("When no local GPU is available: generate the script "
                     "and submit it with the site's executor (SageMaker, "
                     "SLURM, ...)."),
        returns="Path of the written script.",
        example=("write_abtem_script('3sigma_2', mic, 'run_haadf.py', "
                 "branch='md', roi={'z': (63., 183.)}, type_map={1: 'Ni'})"),
    ),
]
