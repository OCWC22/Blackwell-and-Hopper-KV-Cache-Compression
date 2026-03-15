#!/usr/bin/env python3
"""Plot roofline and latency curves from KV block loading benchmark results.

Usage:
    python scripts/plot_kv_load_roofline.py results/kv_block_load_<ts>.json

Produces:
    results/kv_load_roofline_<ts>.png   — throughput roofline plot
    results/kv_load_latency_<ts>.png    — latency growth plot
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Hardware ceilings (GB/s) for reference lines
BANDWIDTH_CEILINGS = {
    "hbm_to_hbm": {"label": "H100 HBM", "gbps": 3350, "color": "#1f77b4"},
    "ram_to_hbm": {"label": "PCIe Gen5 x16", "gbps": 63, "color": "#ff7f0e"},
    "disk_to_hbm": {"label": "NVMe Gen4 x4", "gbps": 7, "color": "#2ca02c"},
}

DTYPE_STYLES = {
    "bf16": {"marker": "o", "label": "BF16"},
    "fp8_combined": {"marker": "s", "label": "FP8 (transfer+cast)"},
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def extract_series(measurements, tier, dtype_key):
    """Extract (kv_bytes, throughput_gbps, latency_ms) series."""
    xs, ys_tp, ys_lat = [], [], []
    for m in measurements:
        transfers = m.get("transfers", {}).get(tier, {})
        entry = transfers.get(dtype_key, {})
        if not isinstance(entry, dict):
            continue
        tp = entry.get("throughput_gbps") or entry.get("effective_throughput_gbps")
        lat = entry.get("latency_ms_median") or entry.get("transfer_plus_cast_ms")
        if tp is None or lat is None:
            continue
        # Use BF16 bytes as the transfer size (source format)
        kv_bytes = m["kv_bytes_bf16"]
        xs.append(kv_bytes)
        ys_tp.append(tp)
        ys_lat.append(lat)
    return xs, ys_tp, ys_lat


def extract_token_series(measurements, tier, dtype_key):
    """Extract (cumulative_tokens, latency_ms) series."""
    xs, ys = [], []
    for m in measurements:
        transfers = m.get("transfers", {}).get(tier, {})
        entry = transfers.get(dtype_key, {})
        if not isinstance(entry, dict):
            continue
        lat = entry.get("latency_ms_median") or entry.get("transfer_plus_cast_ms")
        if lat is None:
            continue
        xs.append(m["cumulative_tokens"])
        ys.append(lat)
    return xs, ys


def plot_roofline(results, out_path):
    """Plot throughput roofline: one subplot per source tier."""
    measurements = results["measurements"]
    tiers = [t for t in ["hbm_to_hbm", "ram_to_hbm", "disk_to_hbm"]
             if any(t in m.get("transfers", {}) for m in measurements)]

    fig, axes = plt.subplots(1, len(tiers), figsize=(6 * len(tiers), 5),
                             squeeze=False)
    fig.suptitle("KV Block Loading Roofline", fontsize=14, y=1.02)

    for col, tier in enumerate(tiers):
        ax = axes[0][col]
        ceil = BANDWIDTH_CEILINGS.get(tier, {})

        for dtype_key, style in DTYPE_STYLES.items():
            xs, ys_tp, _ = extract_series(measurements, tier, dtype_key)
            if xs:
                ax.scatter(xs, ys_tp, marker=style["marker"], label=style["label"],
                           s=40, zorder=3)
                ax.plot(xs, ys_tp, linewidth=1, alpha=0.5, zorder=2)

        # Ceiling line
        if ceil:
            ax.axhline(y=ceil["gbps"], color=ceil["color"], linestyle="--",
                        alpha=0.7, label=f'{ceil["label"]} ceiling ({ceil["gbps"]} GB/s)')

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("KV Block Size (bytes)")
        ax.set_ylabel("Effective Throughput (GB/s)")
        ax.set_title(tier.replace("_", " → ").upper())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else
                         f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Roofline plot saved to {out_path}")


def plot_latency(results, out_path):
    """Plot latency growth: X = cumulative tokens, Y = latency (ms)."""
    measurements = results["measurements"]
    tiers = [t for t in ["hbm_to_hbm", "ram_to_hbm", "disk_to_hbm"]
             if any(t in m.get("transfers", {}) for m in measurements)]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("KV Block Load Latency vs Cache Size", fontsize=14)

    colors = {"hbm_to_hbm": "#1f77b4", "ram_to_hbm": "#ff7f0e",
              "disk_to_hbm": "#2ca02c"}
    markers = {"bf16": "o", "fp8_combined": "s"}

    for tier in tiers:
        for dtype_key in ["bf16", "fp8_combined"]:
            xs, ys = extract_token_series(measurements, tier, dtype_key)
            if xs:
                label = (f"{tier.replace('_', '→')} "
                         f"{DTYPE_STYLES.get(dtype_key, {}).get('label', dtype_key)}")
                ax.scatter(xs, ys, marker=markers.get(dtype_key, "o"),
                           color=colors.get(tier, "gray"), label=label,
                           s=30, zorder=3)
                ax.plot(xs, ys, color=colors.get(tier, "gray"),
                        linewidth=1, alpha=0.5, zorder=2)

    # Also plot cast overhead
    cast_types = [("bf16_to_fp8", "FP8 cast", "^", "#d62728"),
                  ("bf16_to_fp4_emulated", "FP4 cast (emu)", "v", "#9467bd")]
    for cast_key, label, marker, color in cast_types:
        xs, ys = [], []
        for m in measurements:
            entry = m.get("cast_overhead", {}).get(cast_key, {})
            if isinstance(entry, dict) and "latency_ms_median" in entry:
                xs.append(m["cumulative_tokens"])
                ys.append(entry["latency_ms_median"])
        if xs:
            ax.scatter(xs, ys, marker=marker, color=color, label=label,
                       s=30, zorder=3)
            ax.plot(xs, ys, color=color, linewidth=1, alpha=0.5, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative Tokens in KV Cache")
    ax.set_ylabel("Latency (ms)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else
                     f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Latency plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot KV block loading roofline from benchmark JSON")
    parser.add_argument("input", help="Path to benchmark JSON file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    results = load_results(args.input)
    out_dir = args.output_dir or os.path.dirname(args.input) or "."

    base = os.path.splitext(os.path.basename(args.input))[0]
    roofline_path = os.path.join(out_dir, f"{base}_roofline.png")
    latency_path = os.path.join(out_dir, f"{base}_latency.png")

    plot_roofline(results, roofline_path)
    plot_latency(results, latency_path)


if __name__ == "__main__":
    main()
