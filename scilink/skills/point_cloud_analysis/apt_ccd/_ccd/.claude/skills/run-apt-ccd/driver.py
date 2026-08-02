#!/usr/bin/env python3
"""
CCD smoke driver — runs the full Compositional Community Detection pipeline.

Usage:
    python driver.py --data <csv_or_pos> --rrng <rrng> [--outdir <dir>] [--k 2 3 4]

The script can also generate synthetic two-phase APT data for a quick sanity
check when no real data is available:
    python driver.py --synth [--outdir /tmp/ccd_out]
"""

import sys
import os
import types
import importlib.util
import argparse
import json
import glob

# ---------------------------------------------------------------------------
# Bootstrap: register analyze/apt-ccd as a Python package so that the
# relative imports inside ccd.py work without installing anything.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_APT_CCD_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
# Resolve to analyze/apt-ccd regardless of where driver.py lives
_APT_CCD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

def _find_apt_ccd_dir():
    """Walk up from driver.py to find the directory containing ccd.py."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isfile(os.path.join(d, "ccd.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise RuntimeError("Cannot find ccd.py — is driver.py inside analyze/apt-ccd/.claude/skills/run-apt-ccd/?")

def _load_ccd():
    apt_ccd_dir = _find_apt_ccd_dir()
    pkg_name = "apt_ccd"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [apt_ccd_dir]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg

    for mod in ("unpack", "ccd"):
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.{mod}", os.path.join(apt_ccd_dir, f"{mod}.py")
        )
        m = importlib.util.module_from_spec(spec)
        m.__package__ = pkg_name
        sys.modules[f"{pkg_name}.{mod}"] = m
        spec.loader.exec_module(m)

    return sys.modules["apt_ccd.ccd"]


# ---------------------------------------------------------------------------
# Synthetic data generator — creates a minimal two-phase Fe/Ni point cloud
# and a matching IVAS-format RRNG file so the driver can run self-contained.
# ---------------------------------------------------------------------------
def make_synthetic_data(outdir):
    import numpy as np
    import pandas as pd

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(42)

    # Phase A: Fe-rich  (70% Fe, 30% Ni) in a 5×5×5 nm box
    n_A = 3000
    coords_A = rng.uniform(0, 5, (n_A, 3))
    das_A = np.where(rng.uniform(0, 1, n_A) < 0.7, 56.0, 58.0)

    # Phase B: Ni-rich  (30% Fe, 70% Ni) in a 5×5×5 nm box, offset in x
    n_B = 3000
    coords_B = rng.uniform(0, 5, (n_B, 3))
    coords_B[:, 0] += 7
    das_B = np.where(rng.uniform(0, 1, n_B) < 0.3, 56.0, 58.0)

    coords = np.vstack([coords_A, coords_B])
    das = np.concatenate([das_A, das_B])

    csv_path = os.path.join(outdir, "synth.csv")
    pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2], "Da": das}).to_csv(
        csv_path, index=False, header=False
    )

    rrng_path = os.path.join(outdir, "synth.rrng")
    with open(rrng_path, "w") as f:
        f.write(
            "[Ions]\nNumber=2\nIon1=Fe\nIon2=Ni\n\n"
            "[Ranges]\nNumber=2\n"
            "Range1=55.5 56.5 Vol:0.0 Fe:1 Color:FF0000\n"
            "Range2=57.5 58.5 Vol:0.0 Ni:1 Color:00FF00\n"
        )

    return csv_path, rrng_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CCD pipeline smoke driver")
    parser.add_argument("--data", help="Path to APT data (.csv, .pos, or .apt)")
    parser.add_argument("--rrng", help="Path to RRNG range file")
    parser.add_argument("--outdir", default="/tmp/ccd_out", help="Output directory")
    parser.add_argument(
        "--k", nargs="+", type=int, default=[2, 3, 4],
        help="k values for k-means (default: 2 3 4)"
    )
    parser.add_argument(
        "--ignore-ions", nargs="*", default=[],
        help="Ion names to exclude (e.g. O1 O1H1 O2)"
    )
    parser.add_argument(
        "--synth", action="store_true",
        help="Generate synthetic two-phase Fe/Ni data and run on that"
    )
    args = parser.parse_args()

    if args.synth:
        synth_dir = os.path.join(args.outdir, "synth_input")
        print(f"[driver] Generating synthetic data in {synth_dir}")
        args.data, args.rrng = make_synthetic_data(synth_dir)

    if not args.data or not args.rrng:
        parser.error("Provide --data and --rrng, or use --synth")

    os.makedirs(args.outdir, exist_ok=True)
    ccd = _load_ccd()

    # --- Step 1: generate neighborhoods ---
    print(f"[driver] Step 1 — generate_neighborhoods")
    print(f"         data   : {args.data}")
    print(f"         rrng   : {args.rrng}")
    print(f"         outdir : {args.outdir}")
    nbhd_result = ccd.generate_neighborhoods(args.data, args.rrng, savedir=args.outdir)
    print(f"[driver] Neighborhoods: {nbhd_result['neighborhood_count']}")
    print(f"[driver] Ion types    : {list(nbhd_result['ion_type_counts'].keys())}")

    # Find the neighbourhood CSV that was just written
    pattern = os.path.join(args.outdir, "*.csv")
    csvs = sorted(glob.glob(pattern))
    if not csvs:
        print(f"ERROR: no CSV found in {args.outdir}", file=sys.stderr)
        sys.exit(1)
    nbhd_csv = csvs[-1]
    print(f"[driver] Neighborhood CSV: {nbhd_csv}")

    # --- Step 2: detect communities ---
    print(f"\n[driver] Step 2 — detect_compositional_communities")
    print(f"         k_values   : {args.k}")
    print(f"         ignore_ions: {args.ignore_ions}")
    comm_result = ccd.detect_compositional_communities(
        nbhd_csv,
        savedir=args.outdir,
        k_values=args.k,
        ignore_ions=args.ignore_ions,
        n_repeats=2,
        q=25,
    )
    print(f"[driver] Communities found : {comm_result['community_count']}")
    print(f"[driver] Neighborhood counts: {dict(comm_result['community_neighborhood_counts'])}")

    # --- Summary ---
    summary = {
        "data": args.data,
        "rrng": args.rrng,
        "outdir": args.outdir,
        "neighborhoods": nbhd_result["neighborhood_count"],
        "ion_types": list(str(k) for k in nbhd_result["ion_type_counts"].keys()),
        "community_count": comm_result["community_count"],
        "community_neighborhood_counts": {
            str(k): int(v) for k, v in comm_result["community_neighborhood_counts"].items()
        },
    }
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[driver] Summary written to {summary_path}")
    print(f"[driver] KS heatmap:        {os.path.join(args.outdir, 'KS_stats.png')}")
    print("[driver] PASS")


if __name__ == "__main__":
    main()
