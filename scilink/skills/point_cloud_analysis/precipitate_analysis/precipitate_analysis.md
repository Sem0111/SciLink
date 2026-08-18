---
description: "Precipitate / second-phase analysis in APT point clouds: delocalized concentration grids, iso-concentration surfaces (marching cubes) with per-precipitate morphology and population statistics, proximity histograms with far-field and lever-rule gates, threshold sweeps. The continuum companion to cluster_analysis - use it when features are extended objects with interfaces rather than dilute point clusters."
detect:
  binaries: []
  env_vars: []
  python_modules: [skimage, scipy]
  guidance: |
    Implemented in precipitate_tools.py (D4): concentration_grid,
    isosurface_precipitates, proxigram (with far-field + lever-rule
    gates). pyvista not required - 3D surfaces export via plotly.
---

## overview

Family-2 skill: find and quantify discrete second-phase particles -
count, number density, size/shape distribution, volume fraction,
core/matrix compositions, and interface chemistry via proxigrams.

## planning

### method selection (family 1 vs family 2)

- Dilute, few-nm solute aggregates (counting question) -> family-1 MSM
  with its null gates.
- Extended objects with genuine interfaces (>= ~2 nm, oxides,
  precipitates): -> THIS skill. When both apply, run both and check
  concordance (count vs count, composition vs composition).
- Threshold choice: start from concentration_grid's suggested_threshold
  (baseline + 3 sigma) OR a literature/reference setting when
  benchmarking against published analyses; ALWAYS report the threshold
  sweep - claims must be threshold-stable.

### gates (the accept criteria)

- proxigram far-field plateau MUST converge to the directly measured
  matrix composition (gates.far_field.passed).
- lever rule: Vf x C_inside + (1-Vf) x C_outside must reconstruct the
  measured bulk composition (gates.lever_rule.passed; ion-basis, 15%
  tolerance for phase-density and interface-smearing effects).
- Interface widths are UPPER BOUNDS: detection loss + positional noise
  + delocalization all broaden them (quantified on the synthetic
  benchmark).

## validation

- Blind score on the synthetic family-2 benchmark
  (make_precipitate_benchmark: sharp-interface Ti-Y-O precipitates of
  known radius/composition/volume fraction in Fe-Cr).
- Real regressions: MA957 at the documented 3 ionic% isoconcentration
  reference (PNNL test case); R31 oxide-pocket proxigram vs the
  family-1 core->rim zonation (independent-method concordance).
- quality_gate (once nativized): metric = far-field convergence sigma +
  lever-rule relative error; physical_review: false.
