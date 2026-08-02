---
name: run-apt-ccd
description: Run, test, and drive the CCD (Compositional Community Detection) pipeline for APT data. Use this skill to execute the pipeline, verify changes, run a smoke test, or generate synthetic APT data for testing.
---

All paths below are relative to `analyze/apt-ccd/`.

The CCD pipeline is a pure-Python CLI library. It has no server or GUI; the
driver (`driver.py`) is the harness — it runs both pipeline steps end-to-end
and writes a `summary.json` + `KS_stats.png` to the output directory.

## Prerequisites

```bash
pip install scikit-learn python-louvain networkx scipy pandas numpy seaborn matplotlib
```

`apav` is only needed for `.apt` binary input (not `.pos` or `.csv`).

## Run (agent path)

The driver lives at `.claude/skills/run-apt-ccd/driver.py` (inside the unit).

**Synthetic smoke test — no real data required:**

```bash
cd analyze/apt-ccd
python3 .claude/skills/run-apt-ccd/driver.py --synth --outdir /tmp/ccd_out
```

Expected terminal output:
```
[driver] Generating synthetic data in /tmp/ccd_out/synth_input
[driver] Step 1 — generate_neighborhoods
[driver] Neighborhoods: 160
[driver] Step 2 — detect_compositional_communities
[driver] Communities found : 4
[driver] Summary written to /tmp/ccd_out/summary.json
[driver] KS heatmap:        /tmp/ccd_out/KS_stats.png
[driver] PASS
```

**With real APT data (CSV + RRNG):**

```bash
cd analyze/apt-ccd
python3 .claude/skills/run-apt-ccd/driver.py \
    --data /path/to/sample.csv \
    --rrng  /path/to/sample.rrng \
    --outdir /tmp/ccd_out \
    --k 4 5 6 \
    --ignore-ions O1 O1H1 O2
```

**With real APT data via the preprocessing CLI (`.pos` or `.apt` input):**

```bash
cd analyze/apt-ccd
python3 apt_preprocessing.py --data sample.pos --rrng sample.rrng --savedir /tmp/raw/
python3 .claude/skills/run-apt-ccd/driver.py \
    --data /tmp/raw/sample_1nm-radius_0.5-overlap.csv \
    --rrng sample.rrng \
    --outdir /tmp/ccd_out
```

### Driver flags

| Flag | Default | Notes |
|---|---|---|
| `--synth` | off | Generate synthetic two-phase Fe/Ni data and run on it |
| `--data` | — | `.csv`, `.pos`, or `.apt` APT data file |
| `--rrng` | — | IVAS-format `.rrng` range file |
| `--outdir` | `/tmp/ccd_out` | Output directory; created if missing |
| `--k` | `2 3 4` | k-means k values; at least 2 values needed |
| `--ignore-ions` | none | Ions to exclude (as named in RRNG, e.g. `O1 O1H1`) |

### Outputs

| File | Description |
|---|---|
| `<sample>_<r>nm-radius_<overlap>-overlap.csv` | Neighborhood compositional table |
| `<sample>_..._community_clustering_<k>_<seed>seed.json` | Per-k KS metadata |
| `<sample>_..._community_clustering.xyz` | Neighborhood coords labeled by community |
| `KS_stats.png` | Heatmap of mean KS statistics per community |
| `summary.json` | Machine-readable run summary |

## Direct invocation (function-level testing)

The two core functions can be imported without a CLI:

```python
import sys, os, types, importlib.util

_dir = os.path.abspath('analyze/apt-ccd')
pkg = types.ModuleType('apt_ccd')
pkg.__path__ = [_dir]; pkg.__package__ = 'apt_ccd'
sys.modules['apt_ccd'] = pkg

for mod in ('unpack', 'ccd'):
    spec = importlib.util.spec_from_file_location(f'apt_ccd.{mod}', f'{_dir}/{mod}.py')
    m = importlib.util.module_from_spec(spec); m.__package__ = 'apt_ccd'
    sys.modules[f'apt_ccd.{mod}'] = m; spec.loader.exec_module(m)

ccd = sys.modules['apt_ccd.ccd']

result = ccd.generate_neighborhoods('sample.csv', 'sample.rrng', savedir='/tmp/out')
comm   = ccd.detect_compositional_communities('/tmp/out/sample_1nm-radius_0.5-overlap.csv', savedir='/tmp/out')
```

## Gotchas

- **`ccd.py` uses relative imports** (`from . import unpack`). It must be
  loaded as part of a package — the bootstrap block in `driver.py` handles
  this. Doing `import ccd` or `python3 ccd.py` directly fails with
  `ImportError: attempted relative import with no known parent package`.

- **RRNG format is strict.** The range entries must follow the IVAS pattern:
  `Range1=55.5 56.5 Vol:0.0 Fe:1 Color:FF0000`. Ion species written as just
  `Fe` (without `:1`) are not matched and all ions end up as `Noise`.

- **CSV input has no header row.** `generate_neighborhoods` reads the CSV
  with `names=['x','y','z','Da']`. If you pass a CSV that already has a
  header line the first row will be treated as a data point and will fail
  mass-range lookup.

- **`apt_preprocessing.py` writes a CSV without a header.** Pass that CSV
  directly to `generate_neighborhoods` or `driver.py --data`; do not add
  a header row.

- **`UserWarning: Tight layout not applied`** from matplotlib when only
  1–2 communities are found. This is cosmetic; the PNG is written fine.

- **`--k` needs at least 2 distinct values** for the Louvain graph step to
  have enough nodes. Using `--k 4` alone may produce an empty graph and a
  `KeyError` in the partition mapping.

## Troubleshooting

**`ImportError: attempted relative import with no known parent package`**
→ You ran `python3 ccd.py` directly. Use the driver instead.

**All ions labeled `Noise`, `neighborhood_count: 0`**
→ RRNG range entries are malformed. Check that Da values in your CSV fall
within the ranges in the RRNG and that ion species are written as `Fe:1`,
not just `Fe`.

**`KeyError` in `detect_compositional_communities` partition mapping**
→ Louvain removed all nodes (graph is empty). Try increasing `--k` values
or lowering `q` from the default 25 (e.g., `q=10`).
