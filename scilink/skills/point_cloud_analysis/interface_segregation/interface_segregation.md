---
description: "Interface / boundary segregation in APT point clouds: planar-feature detection (PCA on solute-enriched voxels), 1D composition profiles across the interface, Gibbsian interfacial excess by two independent routes (profile integration and the Krakauer-Seidman ladder) that must agree, feature-blind shuffle nulls, and 2D in-plane maps. Family 3 - often family 1/2 machinery applied to a detected feature."
detect:
  binaries: []
  env_vars: []
  python_modules: [scipy]
  guidance: |
    Implemented in interface_tools.py: detect_planar_feature,
    interface_profile (excess + gates), inplane_map, family3_report.
---

## overview

Family-3 skill: quantify solute segregation at planar features (grain
boundaries, phase interfaces, RIS sheets): where the feature is, the
composition profile across it, the Gibbsian excess, and whether the
segregation is laterally uniform.

## planning

- Feature source: a user ROI/plane when given; otherwise
  detect_planar_feature on the suspected segregant. Check
  top_is_planar - PCA on a blob or column also returns a "normal" but
  the profile is meaningless there (use family 1/2 instead).
- Choose roi_radius from the feature extent (stay inside the sheet) and
  half_width wide enough that the far field is genuinely bulk.
- Excess basis: gamma_detected_per_nm2 counts DETECTED ions; with the
  detection efficiency known, report gamma_corrected_atoms_per_nm2 too,
  stating the correction.
- Depletion (negative excess, e.g. Cr at RIS boundaries) is a finding:
  run the profile for the depleted species as well.

### gates (the accept criteria)

- far field = bulk (profile plateau vs directly measured bulk);
- integration vs ladder agreement (two independent excess estimates);
- feature-blind label-shuffle null: the excess must collapse.
- Interface widths (FWHM) are UPPER bounds (positional noise +
  trajectory aberrations broaden them).

## validation

- Blind score on the synthetic family-3 benchmark
  (make_segregation_benchmark: one tilted Ni-segregated slab of known
  normal, position, width and gamma_true; excess_fraction=0 control).
- SCORED (blind; plane detected, not given): normal error 0.81 deg,
  plane offset 0.29 nm, planarity 0.994, gamma recovery 0.94
  (integration vs ladder within 5%), FWHM 0.9 nm vs 0.8 seeded; the
  boundary-free control produced a noise "plane" (planarity 0.68) that
  the shuffle gate rejected (gamma ~0) - the gate, not the detector,
  is the accept criterion (planarity threshold now 0.85).
- Real regression PASSED: R31 PWR steel - three thin-in-x sheets
  (planarity 0.95-0.96); top sheet Ni 9.3% bulk -> 69.5% at the plane,
  gamma_Ni = +39 /nm2 (detected basis, routes agree 1.5%), Cr DEPLETED
  gamma_Cr = -11.3 /nm2 (z -15.5), in-plane patchiness 1.04 (uniform) =
  radiation-induced segregation at a boundary, all gates two-sided.
