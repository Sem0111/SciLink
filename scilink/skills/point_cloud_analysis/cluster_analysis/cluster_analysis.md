---
description: "Cluster and chemical short-range-order detection in APT point clouds: clean-room maximum-separation method with parameter sweep, OPTICS/HDBSCAN routes, k-NN/RDF statistics with label-shuffle null models, Warren-Cowley parameters (full-density data only), z-SDM crystallographic-signal gating, and ML-CSRO (motif-specific CNN on z-SDMs). Every detection claim must survive its null model."
detect:
  binaries: []
  env_vars: []
  python_modules: [sklearn, scipy]
  guidance: |
    Implemented in cluster_tools.py (D2): knn_distance_stats,
    msm_parameter_sweep, msm_detect, label_shuffle_null, rdf_compare,
    warren_cowley, zsdm. ML-CSRO route still pending: needs tensorflow +
    the vendored Yue Li/Gault CoCrNi codebase (Apache-2.0; cite Adv.
    Mater. 2024 + Nat. Commun. 2023) - week-2 integration item.
---

## overview

Family-1 skill: find and quantify chemical clustering and short-range
order. The central discipline: methods are NOT interchangeable - each
answers a different question on different data quality - and every
positive claim must survive a deterministic null model.

## planning

### method selection (MANDATORY decision procedure)

Walk these axes IN ORDER, citing scout evidence for each choice; state the
walk in your plan/decisions:

1. **Cloud kind** (classify_cloud_kind): full-density simulated cloud ->
   geometric Warren-Cowley on NN shells is VALID, quantitative, cheapest.
   Real APT reconstruction (~37-80% detection + positional noise) ->
   geometric first-shell WC is UNRELIABLE - excluded as a primary method;
   use statistical routes.
2. **Crystallographic-signal gate**: compute per-pair z-SDMs. No lattice-
   plane oscillations => NO site-resolved ordering claim is defensible by
   ANY method - report that ceiling explicitly and restrict to
   distance-statistics methods.
3. **Question specificity**: open-ended "any SRO/clustering?" -> model-free
   first: k-NN distance distributions + RDF vs label-shuffle null, MSM
   sweep for discrete clusters, CCD for meso-scale partitioning. Named
   motif suspected (objective, phase diagram, or an anomaly from the
   model-free pass) -> ML-CSRO IF a trained model exists for that motif +
   alloy system.
4. **ML-CSRO applicability constraint**: the CNNs are motif- AND
   alloy-specific (shipped models: CoCrNi L12). Never apply a model
   outside its trained domain; propose TRAINING one instead (synthetic
   generator supports the published recipe: simulated ordered vs
   disordered + APT degradation).
5. **Rare-species check**: target species group < ~1-2% of ions ->
   composition clustering is blind; run map_species_zone alongside
   (see apt_ccd guidance).
6. **Concordance**: when >=2 independent methods apply, run both and
   report agreement; discordance is a FINDING to report, never silently
   resolved.

### cluster detection (MSM - clean-room implementation)

- Maximum-separation semantics per posgen/community canon: link ions of
  the target species within d_max; clusters = connected components with
  >= N_min ions; optional erosion/envelope step for matrix-ion inclusion.
- NEVER report single-parameter results: sweep d_max across a window
  bracketing the knee of the k-NN distance distribution (target species,
  k = N_min); report cluster count vs d_max with a stability plateau -
  claims come from the plateau, not a cherry-picked point.
- Cluster statistics: count, number density, size distribution (ions and
  Guinier radius), per-cluster composition (ionic, per policy), matrix
  composition excluding clusters.

### null models (the accept gates)

- **Label-shuffle null**: rerun the identical detection on the same
  positions with species labels randomly permuted (>=5 shuffles). Real
  clustering/SRO signals must collapse; report observed vs null
  (e.g. cluster count N_obs vs N_null +/- sigma).
- Parameter-stability plateau (above) for MSM.
- For k-NN/RDF: report the deviation from the shuffled-null band, not raw
  curves alone.

## validation

- Blind score on the synthetic family-1 benchmark
  (results/apt_benchmarks/family1): recall/precision vs the 30 seeded
  clusters, size and composition recovery, false-positive rate on a
  cluster-free synthetic.
- SCORED (D2, blind protocol - parameters chosen by knee/plateau/null
  only, truth opened afterwards): recall 28/30 = 0.93, precision 1.00,
  size Pearson r = 0.96, composition 0.57 vs seeded 0.50, gate z = 62.
  Both misses have expected detected solute counts (9.6, 12.6) BELOW the
  null-demanded N_min = 13 - undetectable at that noise floor, not tool
  misses. Cluster-free control: knee correctly reports no bimodal signal,
  pipeline refuses to sweep, zero false positives.
- Real-data regression PASSED (D2): R31 PWR steel, O-bearing group -
  MSM localizes the oxide pocket at 150 nm below apex (map_species_zone
  found it at ~145 nm; independent-method concordance), 21.8k-ion
  CrO/Ni-rich cluster + 2 smaller, null contrast 15x, z = 6.3; z-SDM
  contrast 3.1 => site-resolved claims correctly ruled out. wtav CSRO
  answer key (WC route) still pending.
- Unit regression: tests/test_cluster_tools.py (11 checks - blind
  recovery, cluster-free refusal, WC shuffle control + analytic B2
  alpha = +1, z-SDM gate both verdicts + bcc a/2 spacing).
- quality_gate (once nativized): metric = null-model contrast
  (observed/null); physical_review: false.
