---
description: "Prepare abTEM HAADF-STEM simulation inputs from atomistic structures: read any ASE-supported format (CIF, POSCAR, extxyz, trajectories, LAMMPS data with mass-based species inference), then either crop/reorient MD snapshots or orient/tile ideal cells to a zone-axis slab, with detector angles planned against the antialias limit."
detect:
  binaries: []
  env_vars: []
  python_modules: [ase]
  guidance: |
    Requires ASE for structure reading and slab construction. abTEM is only
    needed at simulation time (haadf_workflow) and for orthogonalizing
    non-cubic oriented cells. The preparation helpers are TOOL_SPEC
    functions in the haadf_workflow bundle's abtem_tools.py.
---

## overview

Turns an atomistic structure file into a beam-oriented, orthogonal,
vacuum-padded slab plus a validated microscope parameter set - everything
abTEM needs to render a HAADF-STEM image comparable to experiment.

Two preparation branches, chosen by input character (NOT file extension):

- **MD snapshot branch** (`prepare_md_slab`): large, defected, often
  triclinic cells from LAMMPS/MD. Reorients the beam axis onto z, crops an
  interior orthogonal ROI, trims ragged tilted edges by percentile.
- **Ideal crystal branch** (`build_ideal_slab`): small perfect cells from
  CIF/POSCAR/DFT outputs. Orients the requested zone axis along the beam,
  orthogonalizes the in-plane cell, tiles to a target field of view and
  thickness.

Route by: cell size (> ~1000 atoms or defects present -> MD branch;
unit/small cells -> ideal branch).

## planning

### input formats and species resolution

Any ASE-readable format works (CIF, POSCAR/CONTCAR, XDATCAR, vasprun.xml,
extxyz, .traj, GROMACS, DL_POLY, PDB, ...). Two special cases:

- **LAMMPS data files carry anonymous integer types, not elements.**
  `read_structure` infers species by matching the Masses section against
  atomic masses (0.5 amu tolerance) and fails loudly when ambiguous; an
  explicit `type_map={1: "Ni"}` always overrides. LAMMPS dump files have no
  Masses section - the type_map is required there.
- **Multi-frame files** (dumps, XDATCAR, .traj): exactly one frame is used,
  selected by `frame_index` (default -1 = last) and recorded in the output
  metadata. Do NOT average frames of a deforming trajectory - frames at
  different strain/defect states must be simulated separately. (Frozen-
  phonon ensemble averaging of same-state frames is a future extension.)

### MD branch geometry

The beam must travel along abTEM z. Give `beam_axis` as the ORIGINAL-frame
axis for the desired viewing direction (e.g. an edge-on boundary view along
simulation x -> `beam_axis="x"`); the permutation is right-handed. `roi`
windows are axis-keyed in the original frame, e.g. `{"z": (63., 183.)}` for
a 120 A window around a boundary at z~123.

- Heavily tilted (triclinic) cells: do not shear-correct; the interior
  orthogonal crop is the robust approach. Keep the analysis region more
  than a probe radius away from cropped faces.
- The percentile trim (default 3%) removes the ragged edges the tilt
  leaves on the in-plane axes; relax to ~1% only to recover a few
  Angstroms of width at slight edge-artifact risk.
- Slab thickness along the beam is whatever the cell provides (typically
  ~5 nm) - absolute HAADF intensities will NOT match a 30-80 nm
  experimental lamella; downstream comparison must be geometric
  (column positions, PTM/CoS signatures), never intensity-based.

### ideal branch geometry

`build_ideal_slab(atoms, zone_axis=(1,1,0), min_fov_A=(30,30),
thickness_A=50)` uses `ase.build.surface`, so zone axis == plane normal
holds for CUBIC cells; for non-cubic systems supply a pre-oriented cell.
Useful defaults: FCC metals imaged along [110] resolve close-packed
columns; 30-50 A thickness balances channeling contrast against cost.

### microscope parameters

One dict, mapping 1:1 onto experimental (Velox) metadata so a simulation
matching an experiment is parameterized by construction:

    {"energy_kev": 300, "convergence_mrad": 25.0,
     "haadf_inner_mrad": 65, "haadf_outer_mrad": 200, "defocus_A": 0.0}

### detector vs. antialias limit (hard constraint)

abTEM refuses detector angles beyond the antialias cutoff = 2/3 of the
potential-grid Nyquist angle. At 300 keV: 0.04 A sampling -> ~164 mrad,
0.05 A -> ~131 mrad. The realized per-grid cutoff sits slightly BELOW the
analytic value (grid rounding varies with cell extent), so
`plan_haadf_detector` caps the outer angle at 98% of the analytic limit -
integer, identical across a frame series. A nominal 200 mrad outer angle at
0.04 A sampling therefore integrates to ~160 mrad; the missing tail carries
negligible HAADF signal for transition metals. Capturing a true 200 mrad
requires ~0.033 A sampling at ~1.5x memory/time.

## validation

- Atom count after crop/trim > 0 and the slab extent matches the intended
  ROI (mismatch means wrong roi axis or stale windows).
- Species: every LAMMPS type resolved; no type left as a pseudo-element.
- `plan_haadf_detector` raises if the antialias cap falls below the inner
  angle - decrease pot_sampling_A.
- Always render a quick preview (projected potential) before a full scan
  when geometry is new: the boundary/defect of interest must be visible
  and correctly oriented in projection.
