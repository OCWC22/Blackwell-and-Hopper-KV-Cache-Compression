#!/usr/bin/env python3
"""Plot recompute-vs-load crossover from batched/concurrent sweep data.

Reads the output JSONs from sweep_recompute_cost.py (with --batch-sizes)
and sweep_load_cost.py (with --concurrent-blocks), and finds the operating
regimes where loading beats recomputing and vice versa.

The crossover depends on TWO dimensions:
  1. Context position (recompute gets more expensive deeper in the sequence)
  2. Concurrency/batch size (both curves change under saturation)

This script produces:
  - A 2D crossover plot (context position vs per-block ms) with curves for
    each batch/concurrency level
  - A policy JSON with crossover points at each operating point
  - Text-mode analysis when matplotlib is unavailable

Usage:
    python scripts/plot_crossover.py \
        --recompute results/recompute_curve_*.json \
        --load results/load_curve_*.json \
        --output results/crossover_plot.png

    python scripts/plot_crossover.py \
        --recompute results/recompute_curve_*.json \
        --load results/load_curve_*.json \
        --text-only
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_bench_utils import write_result_json


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot recompute vs load crossover for offload policy"
    )
    p.add_argument("--recompute", nargs="+", required=True,
                   help="Recompute cost JSON(s) from sweep_recompute_cost.py")
    p.add_argument("--load", nargs="+", required=True,
                   help="Load cost JSON(s) from sweep_load_cost.py")
    p.add_argument("--block-size", type=int, default=None,
                   help="Filter to specific block size in tokens")
    p.add_argument("--dtype", type=str, default=None,
                   help="Filter load data to specific dtype")
    p.add_argument("--output", default=None,
                   help="Output plot path. Default: results/crossover_plot.png")
    p.add_argument("--output-json", default=None,
                   help="Output policy JSON. Default: results/crossover_policy.json")
    p.add_argument("--text-only", action="store_true",
                   help="Text analysis only (no matplotlib)")
    return p.parse_args()


def expand_globs(patterns):
    files = []
    for p in patterns:
        expanded = glob.glob(p)
        if expanded:
            files.extend(expanded)
        elif os.path.isfile(p):
            files.append(p)
    return sorted(set(files))


def load_recompute_data(files):
    """Load recompute data, grouped by (position, batch_size)."""
    all_measurements = []
    metadata = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        metadata = data.get("config", {})
        for m in data.get("measurements", []):
            if "error" not in m:
                all_measurements.append(m)
    return all_measurements, metadata


def load_load_data(files):
    """Load transfer data, grouped by (dtype, block_tokens, concurrent_blocks)."""
    all_measurements = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        for m in data.get("measurements", []):
            all_measurements.append(m)
    return all_measurements


def find_crossover(positions, recompute_ms, load_ms):
    """Linear interpolation to find where recompute crosses load."""
    for i in range(len(positions) - 1):
        p0, p1 = positions[i], positions[i + 1]
        r0, r1 = recompute_ms[i], recompute_ms[i + 1]
        if (r0 <= load_ms <= r1) or (r1 <= load_ms <= r0):
            if abs(r1 - r0) < 1e-9:
                return (p0 + p1) / 2
            t = (load_ms - r0) / (r1 - r0)
            return p0 + t * (p1 - p0)
    if all(r > load_ms for r in recompute_ms):
        return 0  # always load
    if all(r < load_ms for r in recompute_ms):
        return float("inf")  # always recompute
    return None


def analyze(recompute_measurements, load_measurements, block_size_filter, dtype_filter):
    """Analyze crossovers across batch/concurrency levels."""
    # Group recompute by batch_size
    batch_sizes = sorted(set(m.get("batch_size", 1) for m in recompute_measurements))
    positions_set = sorted(set(m["context_position"] for m in recompute_measurements))

    # Group load by concurrent_blocks
    conc_levels = sorted(set(m.get("concurrent_blocks", 1) for m in load_measurements))

    results = []

    for bs in batch_sizes:
        rc_at_bs = [m for m in recompute_measurements
                    if m.get("batch_size", 1) == bs]
        rc_at_bs.sort(key=lambda x: x["context_position"])
        positions = [m["context_position"] for m in rc_at_bs]
        # Use per_block_ms_mean if available (batched), else recompute_ms_mean
        recompute_ms = [m.get("per_block_ms_mean", m.get("recompute_ms_mean", 0))
                        for m in rc_at_bs]

        if not positions:
            continue

        for nc in conc_levels:
            matching_loads = [
                m for m in load_measurements
                if m.get("concurrent_blocks", 1) == nc
                and (dtype_filter is None or m["dtype"] == dtype_filter)
                and (block_size_filter is None or m["block_tokens"] == block_size_filter)
            ]

            for le in matching_loads:
                load_ms = le.get("per_block_ms_mean", le.get("load_ms_mean", 0))
                crossover = find_crossover(positions, recompute_ms, load_ms)

                if crossover == 0:
                    policy = "always_load"
                elif crossover == float("inf"):
                    policy = "always_recompute"
                elif crossover is None:
                    policy = "indeterminate"
                else:
                    policy = "hybrid"

                results.append({
                    "recompute_batch_size": bs,
                    "load_concurrent_blocks": nc,
                    "load_dtype": le["dtype"],
                    "block_tokens": le["block_tokens"],
                    "load_per_block_ms": round(load_ms, 4),
                    "load_aggregate_bw_gbps": le.get("aggregate_bandwidth_gbps", None),
                    "crossover_position": (round(crossover, 1)
                                           if crossover is not None
                                           and crossover != float("inf")
                                           else str(crossover)),
                    "policy": policy,
                    "positions": positions,
                    "recompute_per_block_ms": [round(r, 4) for r in recompute_ms],
                })

    return results


def print_analysis(results):
    print("\n" + "=" * 78)
    print("RECOMPUTE vs LOAD CROSSOVER ANALYSIS (Batched / Concurrent)")
    print("=" * 78)

    # Group by (batch_size, concurrent_blocks) for cleaner output
    seen = set()
    for r in results:
        key = (r["recompute_batch_size"], r["load_concurrent_blocks"], r["load_dtype"])
        if key in seen:
            continue
        seen.add(key)

        bs = r["recompute_batch_size"]
        nc = r["load_concurrent_blocks"]
        dtype = r["load_dtype"]
        load_ms = r["load_per_block_ms"]
        xover = r["crossover_position"]
        policy = r["policy"]

        print(f"\n  Recompute batch={bs}, Load conc={nc} ({dtype}):")
        print(f"    Load per-block: {load_ms:.4f} ms")
        print(f"    Crossover:      {xover} tokens")
        print(f"    Policy:         {policy}")

        # Mini table
        print(f"    {'Pos':>10}  {'Recomp ms':>10}  {'Load ms':>10}  {'Winner':>10}")
        for pos, rms in zip(r["positions"], r["recompute_per_block_ms"]):
            winner = "LOAD" if rms > load_ms else "RECOMP"
            print(f"    {pos:>10}  {rms:>10.4f}  {load_ms:>10.4f}  {winner:>10}")

    # Summary matrix: batch_size × concurrency → crossover
    batch_sizes = sorted(set(r["recompute_batch_size"] for r in results))
    conc_levels = sorted(set(r["load_concurrent_blocks"] for r in results))
    dtypes = sorted(set(r["load_dtype"] for r in results))

    for dtype in dtypes:
        print(f"\n\n  === Crossover Matrix ({dtype}) ===")
        print(f"  Rows = recompute batch size, Cols = load concurrency")
        header = f"  {'batch\\conc':>12}"
        for nc in conc_levels:
            header += f"  {'C=' + str(nc):>10}"
        print(header)
        print(f"  {'-' * (14 + 12 * len(conc_levels))}")

        for bs in batch_sizes:
            row = f"  {'B=' + str(bs):>12}"
            for nc in conc_levels:
                match = next((r for r in results
                              if r["recompute_batch_size"] == bs
                              and r["load_concurrent_blocks"] == nc
                              and r["load_dtype"] == dtype), None)
                if match:
                    xover = match["crossover_position"]
                    if match["policy"] == "always_load":
                        cell = "LOAD*"
                    elif match["policy"] == "always_recompute":
                        cell = "RECOMP*"
                    elif match["policy"] == "hybrid":
                        cell = f"{int(float(xover))}tok"
                    else:
                        cell = "???"
                    row += f"  {cell:>10}"
                else:
                    row += f"  {'---':>10}"
            print(row)

        print(f"\n  LOAD* = always load, RECOMP* = always recompute, Ntok = crossover position")

    print("\n" + "=" * 78)


def generate_plot(results, recompute_measurements, load_measurements, output_path):
    """Generate crossover plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("WARNING: matplotlib not available. Use --text-only.")
        return False

    batch_sizes = sorted(set(m.get("batch_size", 1) for m in recompute_measurements))
    conc_levels = sorted(set(m.get("concurrent_blocks", 1) for m in load_measurements))
    dtypes = sorted(set(m["dtype"] for m in load_measurements))

    fig, ax = plt.subplots(figsize=(14, 8))

    # Color maps
    rc_cmap = plt.cm.Blues
    ld_cmap = plt.cm.Reds

    # Plot recompute curves (one per batch size)
    for i, bs in enumerate(batch_sizes):
        rc = [m for m in recompute_measurements if m.get("batch_size", 1) == bs]
        rc.sort(key=lambda x: x["context_position"])
        positions = [m["context_position"] for m in rc]
        per_block = [m.get("per_block_ms_mean", m.get("recompute_ms_mean", 0)) for m in rc]

        color = rc_cmap(0.4 + 0.5 * i / max(len(batch_sizes) - 1, 1))
        ax.plot(positions, per_block, color=color, marker="o", linewidth=2,
                markersize=5, label=f"Recompute (batch={bs})", zorder=3)

    # Plot load lines (one per concurrency × dtype)
    line_idx = 0
    for dtype in dtypes:
        for nc in conc_levels:
            matching = [m for m in load_measurements
                        if m.get("concurrent_blocks", 1) == nc
                        and m["dtype"] == dtype]
            if not matching:
                continue
            load_ms = matching[0].get("per_block_ms_mean", matching[0].get("load_ms_mean", 0))
            color = ld_cmap(0.3 + 0.5 * line_idx / max(len(conc_levels) * len(dtypes) - 1, 1))
            ax.axhline(y=load_ms, color=color, linestyle="--", linewidth=1.5,
                       label=f"Load ({dtype}, conc={nc}): {load_ms:.3f}ms", zorder=2)
            line_idx += 1

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x)}" if x < 10000 else f"{int(x / 1024)}K"
    ))

    ax.set_xlabel("Context Position (tokens)", fontsize=12)
    ax.set_ylabel("Per-Block Cost (ms), log scale", fontsize=12)
    ax.set_title("KV Cache Offload Policy: Recompute vs Load Under Saturation", fontsize=13)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()
    return True


def main():
    args = parse_args()

    if args.output is None:
        args.output = "results/crossover_plot.png"
    if args.output_json is None:
        args.output_json = "results/crossover_policy.json"

    recompute_files = expand_globs(args.recompute)
    load_files = expand_globs(args.load)

    if not recompute_files:
        print("ERROR: No recompute data files found.")
        sys.exit(1)
    if not load_files:
        print("ERROR: No load data files found.")
        sys.exit(1)

    print(f"Recompute files: {recompute_files}")
    print(f"Load files: {load_files}")

    recompute_measurements, rc_meta = load_recompute_data(recompute_files)
    load_measurements = load_load_data(load_files)

    if not recompute_measurements:
        print("ERROR: No valid recompute data.")
        sys.exit(1)
    if not load_measurements:
        print("ERROR: No valid load data.")
        sys.exit(1)

    results = analyze(recompute_measurements, load_measurements,
                      args.block_size, args.dtype)

    if not results:
        print("ERROR: No matching (recompute, load) pairs found.")
        sys.exit(1)

    print_analysis(results)

    # Write policy JSON
    policy_output = {
        "analysis_type": "recompute_vs_load_crossover",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filters": {"block_size": args.block_size, "dtype": args.dtype},
        "crossovers": results,
    }
    write_result_json(policy_output, args.output_json)

    if not args.text_only:
        generate_plot(results, recompute_measurements, load_measurements, args.output)
    else:
        print("\n(Plot skipped — use without --text-only to generate)")


if __name__ == "__main__":
    main()
