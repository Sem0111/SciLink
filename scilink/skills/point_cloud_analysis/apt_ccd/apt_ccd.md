---
description: "Compositional Community Detection (CCD) for APT and simulated point clouds: overlapping spherical composition neighborhoods, KMeans-ensemble + Louvain community detection, per-community enrichment/depletion signatures (signed KS statistics) - identifies chemical segregation, clustering and short-range-order domains. Works on real reconstructions (.pos/.apt + .rrng) and on simulated structures via a synthetic-mass adapter."
detect:
  binaries: []
  env_vars: []
  python_modules: [sklearn, community, networkx, seaborn, heapdict]
  guidance: |
    Vendored implementation (with permission) of Bilbrey et al., Microscopy
    and Microanalysis 31 (2025), maintained by Jenna Pope
    (jenna.pope@pnnl.gov) - CITE when used. python-louvain provides the
    'community' module; apav is needed only for real .pos/.apt inputs.
---

## overview

CCD finds compositionally distinct regions without pre-specifying what to
look for: overlapping spherical neighborhoods (default 1 nm radius, 50%
overlap) -> per-neighborhood composition vectors -> an ensemble of KMeans
clusterings (k = 4,5,6 x seeds) -> Louvain communities on the co-clustering
graph -> stable communities with per-ion signed KS statistics (+ enriched,
- depleted vs bulk).

Tools: `neighborhoods_from_apt` (real data), `neighborhoods_from_structure`
(simulated data, ideal-detection assumption), `detect_segregation`.

## planning

- Neighborhood radius sets the length scale probed: 1 nm resolves nm-scale
  segregation/clustering; sub-nm short-range order is SMOOTHED at 1 nm and
  appears as reduced-amplitude composition modulations - decreasing the
  radius (0.5-0.75 nm) sharpens it at the cost of counting statistics
  (aim for >100 ions/neighborhood).
- Interpret communities with BOTH signals: signed KS statistics (statistical
  strength) and mean composition differences (physical amplitude). Strong
  KS with small percentage-point amplitude = dispersed short-range order /
  incipient clustering; strong KS with large amplitude = discrete phases or
  precipitates.
- Always check the SPATIAL distribution of communities (community_xyz):
  localized communities = surface/apex/boundary segregation or precipitates;
  interpenetrating uniform communities = bulk SRO/spinodal-like partitioning.
- Ignore-ions: exclude contaminants (O, H species) on real data.
- Simulated route caveat: ideal detection (100% efficiency, no trajectory
  aberrations). Real APT (~40-80% efficiency) blurs amplitudes further -
  simulated results are upper bounds on detectability, which is exactly the
  sim2exp question the skill can answer by degrading the simulated cloud.

### visualization and reporting (APT convention)
- FIRST figure of any APT report: `visualize_apt_elements` - element-colored
  atom maps, apex at the TOP, one panel per element plus the combined
  overlay. Label it `hero_png` in your files dict so the report renders it
  large; every other figure is subordinate.
- THEN the CCD community map (auto-rendered by detect_segregation) - its
  legend must carry MEANINGS (annotate_communities), never bare community
  numbers.
- Interactive 3D htmls (elements and communities) are produced alongside;
  the report links them as "open 3D visualization".
- Composition reporting: IONIC composition (composition_from_labels
  default) - every ranged species as-is, molecular ions and unidentified
  peaks (27Da etc.) as their own species. Do NOT decompose molecular ions
  into elements unless the user explicitly asks (decompose=True);
  decomposition injects assumptions and is not the APT reporting standard
  here. Never hand-roll element tallies from labels.
- Probing scale: 1 nm neighborhoods miss sub-nm segregation; if a targeted
  species shows no community at 1 nm, rerun at 0.5-0.75 nm before calling
  a negative.

### rare-species blindness (hard-won rule)
- CCD composition clustering CANNOT detect minority chemistries whose
  per-neighborhood counts sit below matrix counting noise (in practice:
  species groups under ~1-2% of ions, worse when fragmented across many
  molecular-ion labels). For any objective targeting oxides, carbides, or
  impurity enrichment, ALWAYS run map_species_zone alongside CCD - it maps
  a species GROUP directly (aggregating whole ions, never decomposing) and
  finds localized zones invisible to clustering. Verified live: a 53%%-
  local-fraction oxide pocket in a PWR steel that CCD missed entirely.

### noise-partitioning on homogeneous data (hard-won rule)
- On a compositionally HOMOGENEOUS cloud, the community machinery still
  partitions counting-statistics fluctuations into plausible-looking
  per-element communities with large KS signatures (~45-ion 1-nm
  neighborhoods fluctuate by sigma ~ 8-9%% per species - plenty for a
  clusterer to carve). Verified on the published annealed-CoCrNi
  benchmark: the AUTHORS' random dataset (same positions, scrambled
  labels, zero chemistry) yielded 4 communities with KS +0.42..+0.52 -
  equal to or LARGER than the real data's. Therefore a CCD community
  structure is NOT evidence of segregation by itself: ALWAYS rerun
  detection on a label-shuffle (or provided random) control and claim
  only structure ABSENT from the control; spatially contiguous domains
  (vs salt-and-pepper) are the corroborating signature.

## validation

- Mean neighborhood density should match the material (~60/nm^3 for BCC
  refractory metals at full detection).
- Communities should be stable across the KMeans ensemble (that is what the
  Louvain co-clustering graph enforces); a single-k single-seed structure
  is not a finding.
- Report bulk composition alongside per-community compositions.
