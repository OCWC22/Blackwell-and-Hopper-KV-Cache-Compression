#!/usr/bin/env python3
"""Sweep visualization for KV cache benchmark results.

Reads result JSON files from results/ and produces concurrency sweep charts
comparing KV modes (bf16/fp8/nvfp4/nvfp4+offload).

Usage:
    # Plot all results
    python scripts/plot_sweep.py --results-dir results/ --output-dir results/plots/

    # Filter by scenario
    python scripts/plot_sweep.py --filter-scenario scenario_2_more_sessions_gpu

TODO(engineer): Additional chart types to add:
  - HBM breakdown stacked bar chart (hot KV vs offloaded KV vs model weights)
  - TTFT cold-vs-warm comparison (from run_tiered_experiment.py results)
  - Quality delta scatter plot (quality_metric_value vs kv_mode)
  - Prompt bucket comparison (10K/20K/30K/40K/50K side by side)
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_results(results_dir: str) -> list[dict]:
    """Load all result JSON files from a directory."""
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(results_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            if "scenario_id" in data and "metrics" in data:
                data["_file"] = fname
                results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def get_variant_label(result: dict) -> str:
    """Extract a human-readable variant label from a result."""
    kv_mode = result.get("model", {}).get("kv_mode", "unknown")
    tiered = result.get("tiering", {}).get("enabled", False)
    codec = result.get("tiering", {}).get("cold_tier_codec", "none")
    label = kv_mode
    if tiered:
        label += "+offload"
    if codec and codec != "none":
        label += f"+{codec}"
    return label


def group_by_variant(results: list[dict]) -> dict[str, list[dict]]:
    """Group results by variant label, sorted by concurrency."""
    groups = defaultdict(list)
    for r in results:
        label = get_variant_label(r)
        groups[label].append(r)
    for label in groups:
        groups[label].sort(key=lambda r: r.get("workload", {}).get("concurrency", 0))
    return dict(groups)


def plot_sweep(groups: dict[str, list[dict]], output_dir: str, scenario: str = ""):
    """Generate sweep comparison charts.

    Produces a 2x2 grid: TTFT p95, TPOT p95, Throughput, Peak HBM.
    Each KV mode variant is one series.

    TODO(engineer): Add additional panels or separate charts for:
      - promotion_latency_ms_p95 (from tiered results)
      - cache_hit_rate (from tiered results)
      - tokens_per_joule (energy efficiency)
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ("ttft_ms_p95", "TTFT p95 (ms)", "lower is better"),
        ("tpot_ms_p95", "TPOT p95 (ms)", "lower is better"),
        ("throughput_tokens_per_s", "Throughput (tokens/s)", "higher is better"),
        ("peak_hbm_gb", "Peak HBM (GB)", "lower is better"),
    ]

    # TODO(engineer): Extend this color map when new variants are benchmarked
    colors = {
        "bf16": "#1f77b4",
        "fp8": "#ff7f0e",
        "nvfp4": "#2ca02c",
        "nvfp4+offload": "#d62728",
        "nvfp4+offload+kvtc": "#9467bd",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"KV Cache Concurrency Sweep{' — ' + scenario if scenario else ''}",
                 fontsize=14, fontweight="bold")

    for ax, (metric_key, ylabel, note) in zip(axes.flat, metrics):
        for label, results in sorted(groups.items()):
            concurrencies = []
            values = []
            for r in results:
                conc = r.get("workload", {}).get("concurrency", 0)
                val = r.get("metrics", {}).get(metric_key)
                if val is not None:
                    concurrencies.append(conc)
                    values.append(val)
            if concurrencies:
                color = colors.get(label, None)
                ax.plot(concurrencies, values, "o-", label=label, color=color,
                        linewidth=2, markersize=6)

        ax.set_xlabel("Active Conversations (Concurrency)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}  ({note})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        all_conc = sorted(set(
            r.get("workload", {}).get("concurrency", 0)
            for group in groups.values() for r in group
        ))
        if all_conc:
            ax.set_xticks(all_conc)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"sweep{'_' + scenario if scenario else ''}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close()

    # Also emit CSV for further analysis
    csv_path = os.path.join(output_dir, f"sweep{'_' + scenario if scenario else ''}.csv")
    with open(csv_path, "w") as f:
        f.write("variant,concurrency,ttft_p95_ms,tpot_p95_ms,throughput_tps,peak_hbm_gb\n")
        for label, results in sorted(groups.items()):
            for r in results:
                conc = r.get("workload", {}).get("concurrency", 0)
                m = r.get("metrics", {})
                f.write(f"{label},{conc},"
                        f"{m.get('ttft_ms_p95', '')},{m.get('tpot_ms_p95', '')},"
                        f"{m.get('throughput_tokens_per_s', '')},{m.get('peak_hbm_gb', '')}\n")
    print(f"Saved CSV to {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Plot KV cache benchmark sweep results")
    p.add_argument("--results-dir", default="results/",
                   help="Directory containing result JSON files")
    p.add_argument("--output-dir", default="results/plots/",
                   help="Output directory for plots")
    p.add_argument("--filter-scenario", default=None,
                   help="Filter results by scenario ID")
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = str(repo_root / args.results_dir)
    output_dir = str(repo_root / args.output_dir)

    results = load_results(results_dir)
    print(f"Loaded {len(results)} result files from {results_dir}")

    if not results:
        print("No results found. Run benchmarks first, then re-run this script.")
        return

    if args.filter_scenario:
        results = [r for r in results if r.get("scenario_id") == args.filter_scenario]
        print(f"After filtering: {len(results)} results for {args.filter_scenario}")

    if not results:
        print("No results match the filter.")
        return

    groups = group_by_variant(results)
    print(f"Variants found: {', '.join(sorted(groups.keys()))}")

    scenario = args.filter_scenario or ""
    plot_sweep(groups, output_dir, scenario)


if __name__ == "__main__":
    main()
