---
description: "3D structural identification and defect analysis of atomistic point clouds: per-atom Polyhedral Template Matching (FCC/HCP/BCC/...), defect-cluster extraction with planarity/composition, DXA dislocation analysis with Burgers vectors, and host-hidden 3D defect visualization (interactive HTML + projection panels)."
detect:
  binaries: []
  env_vars: []
  python_modules: [ovito]
  guidance: |
    Wraps the free OVITO Python package (PTM per Larsen et al., MSMSE 24,
    055007 (2016); DXA verified available in the free package). plotly is an
    optional extra for the interactive HTML artifact; the projection PNG
    needs only matplotlib. The MIT-licensed reference implementation
    (github.com/pmla/polyhedral-template-matching) is the dependency-free
    fallback path if ovito is unavailable.
---

## overview

Ground-truth 3D structural analysis of a point cloud, independent of any
imaging step. Four stages, each a TOOL_SPEC function in `ptm3d_tools.py`:

1. `classify_structure_3d` - per-atom PTM types, host lattice, fractions
2. `extract_defects_3d` - non-host atoms clustered into defect objects
   (size, composition, centroid, extent, planarity)
3. `analyze_dislocations` - DXA segments, true Burgers vectors, line length
4. `visualize_defects_3d` - host atoms hidden, defects rendered as
   interactive 3D HTML + three-view projection PNG

## planning

- Run classification FIRST; the dominant type defines the host and every
  later stage keys off it.
- **Surface artifact (do not report as defects):** PTM classifies free
  surfaces as HCP/OTHER sheets. A planar cluster whose centroid sits at a
  cell face (position ~ min or max of the cell along the sheet normal) is a
  SURFACE. Only interior clusters are defects.
- REPORTING LANGUAGE: surface clusters are "surface-unclassifiable atoms
  (finite-specimen artifact)" - never "defects" or "other phases"; a
  92%-BCC single-phase tip with a 7% surface skin IS single-phase. Reserve
  "defect" for interior clusters, and say so in claims and figure titles.
- Interpretation of interior clusters in an FCC host: one planar HCP sheet
  = coherent twin boundary (sigma-3); two adjacent HCP sheets = intrinsic
  stacking fault; HCP-FCC-HCP sandwich = extrinsic fault; non-planar
  blobs / OTHER-rich clusters = disordered regions.
- DXA answers the dislocation question directly; zero segments in a
  structure with planar faults means plasticity proceeded by twinning /
  faulting, not stored dislocations - report that as a finding.
- rmsd_cutoff 0.1 (strict) - 0.2 (tolerant of thermal noise); 0.15 default.
- 3D results are the ground truth against which image-side (2D projected)
  classification should be checked when both are computed - disagreement
  localizes projection/detection artifacts, not physics.

## validation

- fractions sum to ~1; host fraction plausible for the material state.
- Every reported defect cluster is interior (surface check applied).
- Visualization artifact count matches n_defect_atoms (minus subsampling).
