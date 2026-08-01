---
description: "Simulation-to-experiment bridge for atomic-resolution STEM: render HAADF images from atomistic structures (abTEM PRISM, GPU-aware, OOM fallback ladder) and hand them to the image_analysis skills. Select when the objective asks whether an observed HAADF feature is consistent with a hypothesized structure (twin, stacking fault, grain boundary, phase), when a candidate structure exists (MD/DFT/aimsgb output), or when ground-truth-labeled simulated images are needed for detector validation or DCNN training."
detect:
  binaries: []
  env_vars: []
  python_modules: [ase, abtem]
  guidance: |
    Requires abtem + ase; cupy (matching the local CUDA version, e.g.
    cupy-cuda12x) enables GPU execution and is strongly recommended - CPU
    runs work but are minutes-to-hours slower. With no local GPU, use
    write_abtem_script and submit via the site's executor (SageMaker,
    SLURM). Slab preparation guidance lives in the co-active
    generate_abtem_input skill; defect classification of the results
    reuses the crystalline_deformation skill's ptm_tools.
---

## overview

Forward-simulates HAADF-STEM images from atomistic structures and packages
them so the analysis skills consume them like any experimental image. The
pipeline has five stages:

1. **Read** - `read_structure` (any ASE format; LAMMPS species inference)
2. **Prepare** - `prepare_md_slab` (MD branch) or `build_ideal_slab`
   (ideal branch); guidance in generate_abtem_input
3. **Plan** - `plan_haadf_detector` (antialias-capped annulus)
4. **Simulate** - `simulate_haadf` (PRISM, GPU auto-detect, OOM ladder) or
   `write_abtem_script` for remote GPU execution
5. **Analyze / compare** - outputs feed atomic_stem and
   crystalline_deformation directly; `structure_defect_map` gives the
   exact structure-side reference

Output contract: `<prefix>.npy` (image), `<prefix>.png`, and
`<prefix>_meta.json` carrying `fov_nm` and `pixel_size_nm` - the same
fields the image_analysis skills resolve from experimental metadata.

## planning

### standalone use

Given only a structure file, the minimal chain is read -> prepare ->
simulate; add `structure_defect_map` for PTM/CoS defect maps computed from
the structure itself (no detection step, exact). This answers "what would
this MD/DFT structure look like in HAADF, and where are its defects"
without any experimental image involved.

### the simulation-to-experiment loop

When an experimental HAADF and an objective are in play, the bridge closes
a hypothesis loop:

1. Analyze the experimental image (atomic_stem detection +
   crystalline_deformation classification) -> structural hypothesis
   (e.g. coherent sigma-3 twin viewed along [110]).
2. Obtain a candidate structure: aimsgb (grain boundaries), the
   structure_generation skills, an MD snapshot, or a relaxed DFT cell.
3. Simulate with the microscope dict taken FROM THE EXPERIMENT'S METADATA
   (energy, convergence angle, detector angles) so the optics match by
   construction.
4. Run the SAME analysis skills on the simulated image.
5. Compare like-for-like (rules below); consistency supports the
   hypothesis, mismatch refutes it or the imaging conditions.

### like-for-like comparison rules (non-negotiable)

- Compare GEOMETRY and classification signatures: column positions, PTM
  layer-count signatures (1 HCP layer = twin, 2 = ISF, 2+1 = ESF), CoS
  levels and spatial patterns.
- NEVER compare absolute intensities: the simulated foil is typically
  ~5 nm vs a 30-80 nm lamella, and a single snapshot is one frozen-phonon
  configuration (qualitative contrast only).
- Apply a source-size blur (~0.35 A Gaussian) to the simulated image
  before visual or detection-based comparison; the raw simulation is
  unrealistically sharp.
- Match pixel scale: resample so both images have comparable A/px before
  running the same detector on both.
- Run the identical detection + classification pipeline on both images -
  never one method on the experiment and another on the simulation.

### compute and memory

- Device auto-detection: cupy present -> GPU, else CPU with a loud
  warning. A ~40 x 90 A field at 0.04 A sampling, PRISM interpolation 4,
  runs in ~2 min on a 24 GB A10G; CPU is orders slower.
- OOM fallback ladder (automatic in `simulate_haadf`): potential sampling
  0.04 -> 0.05 A, then PRISM interpolation 4 -> 6. Interpolation 6 trades
  accuracy for memory (coarser plane-wave sampling can smooth fine HAADF
  contrast) - the metadata records which rung succeeded; report it.
- PRISM interpolation 4 is the accuracy/speed sweet spot for ~100 A
  fields; 2 is near-exact but slow; reserve 6 for memory emergencies.
- No local GPU: `write_abtem_script` emits a self-driving script for a
  remote GPU host (SageMaker training job, SLURM node). The script needs
  scilink + abtem + cupy installed remotely; outputs come back as files.

### DCNN training data

`structure_defect_map` centroids are exact column labels for the simulated
image (render as Gaussian discs for segmentation masks). Simulating at the
experiment's own optics yields condition-matched training data for
image-side detectors (e.g. AtomAI UNet) that then transfer to the
experimental images - the same bridge, run in the training direction.

## validation

- Metadata sanity: fov_nm equals slab extent / 10; detector
  outer_mrad_effective <= antialias limit; oom_ladder_attempt reported.
- Simulated-image detection sanity: on a defect-free region, image-side
  detection against `structure_defect_map` centroids should reach ~100%
  precision and picometre-scale RMS position error; large deviations mean
  detection settings, not physics.
- PTM on a known reference: a coherent twin must classify as exactly one
  HCP layer between mirrored FCC domains; a perfect crystal must classify
  ~100% FCC away from image edges (edge columns lack 6 Delaunay
  neighbors and report as unidentified - expected, not an error).
- CoS on a perfect FCC region should be ~0; a coherent twin plane shows a
  weak single-plane CoS line (sub-A^2) - visible only with per-image
  color scaling.
