#!/usr/bin/env python3
"""Plot roofline and latency curves from KV block offload/restore benchmark.

Usage:
    python scripts/plot_kv_load_roofline.py results/kv_block_offload_restore_<ts>.json

Produces:
    results/<basename>_roofline.png   — throughput roofline (offload + restore)
    results/<basename>_latency.png    — per-block latency vs transfer size
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Hardware ceilings (GB/s)
CEILINGS = {
    "HBM": 3350,
    "PCIe (host)": 63,
    "NVMe (disk)": 7,
}

# Which JSON keys map to which tier and direction
PATHS = {
    "offload_to_host": {"tier": "PCIe (host)", "direction": "offload"},
    "restore_from_host": {"tier": "PCIe (host)", "direction": "restore"},
    "offload_to_disk": {"tier": "NVMe (disk)", "direction": "offload"},
    "restore_from_disk": {"tier": "NVMe (disk)", "direction": "restore"},
    "hbm_copy": {"tier": "HBM", "direction": "copy"},
}

FMT_COLORS = {"bf16": "#1f77b4", "fp8": "#ff7f0e", "fp4": "#2ca02c"}
FMT_MARKERS = {"bf16": "o", "fp8": "s", "fp4": "^"}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def _fmt_bytes(x, _):
    if x >= 1e9:
        return f"{x/1e9:.0f}GB"
    if x >= 1e6:
        return f"{x/1e6:.0f}MB"
    if x >= 1e3:
        return f"{x/1e3:.0f}KB"
    return f"{x:.0f}B"


def plot_roofline(results, out_path):
    """Throughput roofline: separate subplots for offload vs restore."""
    measurements = results["measurements"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
    fig.suptitle("KV Block Offload/Restore Roofline", fontsize=14, y=1.02)

    panels = [
        ("hbm_copy", "HBM Copy", axes[0][0]),
        ("offload_to_host", "Offload GPU→Host", axes[0][1]),
        ("restore_from_host", "Restore Host→GPU", axes[0][2]),
    ]

    for path_key, title, ax in panels:
        meta = PATHS[path_key]
        ceil_gbps = CEILINGS.get(meta["tier"], 0)

        for fmt in ["bf16", "fp8", "fp4"]:
            xs, ys = [], []
            for m in measurements:
                if m["format"] != fmt:
                    continue
                entry = m.get(path_key)
                if not isinstance(entry, dict) or "throughput_gbps" not in entry:
                    continue
                xs.append(m["total_bytes"])
                ys.append(entry["throughput_gbps"])
            if xs:
                ax.scatter(xs, ys, marker=FMT_MARKERS[fmt],
                           color=FMT_COLORS[fmt], label=fmt.upper(),
                           s=30, zorder=3)
                ax.plot(xs, ys, color=FMT_COLORS[fmt], linewidth=1,
                        alpha=0.4, zorder=2)

        if ceil_gbps:
            ax.axhline(y=ceil_gbps, color="red", linestyle="--", alpha=0.6,
                       label=f"Ceiling {ceil_gbps} GB/s")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Transfer Size (bytes)")
        ax.set_ylabel("Effective Throughput (GB/s)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_bytes))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Roofline plot → {out_path}")


def plot_latency(results, out_path):
    """Per-block latency vs transfer size, all paths on one plot."""
    measurements = results["measurements"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("KV Block Offload/Restore Latency", fontsize=14)

    path_groups = [
        ("Offload", ["offload_to_host", "offload_to_disk"], axes[0]),
        ("Restore", ["restore_from_host", "restore_from_disk"], axes[1]),
    ]

    tier_colors = {
        "offload_to_host": "#ff7f0e", "restore_from_host": "#ff7f0e",
        "offload_to_disk": "#2ca02c", "restore_from_disk": "#2ca02c",
    }
    tier_labels = {
        "offload_to_host": "Host", "restore_from_host": "Host",
        "offload_to_disk": "Disk", "restore_from_disk": "Disk",
    }

    for group_name, path_keys, ax in path_groups:
        for path_key in path_keys:
            for fmt in ["bf16", "fp8", "fp4"]:
                xs, ys = [], []
                for m in measurements:
                    if m["format"] != fmt:
                        continue
                    entry = m.get(path_key)
                    if not isinstance(entry, dict):
                        continue
                    if "latency_ms_median" not in entry:
                        continue
                    xs.append(m["total_bytes"])
                    ys.append(entry["latency_ms_median"])
                if xs:
                    label = f"{tier_labels[path_key]} {fmt.upper()}"
                    ax.scatter(xs, ys, marker=FMT_MARKERS[fmt],
                               color=tier_colors[path_key], label=label,
                               s=25, zorder=3, alpha=0.8)
                    ax.plot(xs, ys, color=tier_colors[path_key],
                            linewidth=1, alpha=0.3, zorder=2,
                            linestyle="--" if fmt != "bf16" else "-")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Transfer Size (bytes)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{group_name} Latency")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3, which="both")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_bytes))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Latency plot → {out_path}")


def plot_batch_scaling(results, out_path):
    """Show how throughput scales with batch count for a fixed block size."""
    measurements = results["measurements"]

    # Pick the most common block size
    block_sizes = set(m["block_size_tokens"] for m in measurements)
    target_blk = 16 if 16 in block_sizes else sorted(block_sizes)[len(block_sizes)//2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Batch Scaling (block_size={target_blk} tokens)", fontsize=14)

    for ax, (path_key, title) in zip(axes, [
        ("offload_to_host", "Offload → Host"),
        ("restore_from_host", "Restore ← Host"),
    ]):
        for fmt in ["bf16", "fp8", "fp4"]:
            xs, ys = [], []
            for m in measurements:
                if m["block_size_tokens"] != target_blk or m["format"] != fmt:
                    continue
                entry = m.get(path_key)
                if not isinstance(entry, dict) or "throughput_gbps" not in entry:
                    continue
                xs.append(m["batch_count"])
                ys.append(entry["throughput_gbps"])
            if xs:
                ax.plot(xs, ys, marker=FMT_MARKERS[fmt], color=FMT_COLORS[fmt],
                        label=fmt.upper(), linewidth=2, markersize=6)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch Count (blocks per transfer)")
        ax.set_ylabel("Throughput (GB/s)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Batch scaling plot → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot KV offload/restore roofline from benchmark JSON")
    parser.add_argument("input", help="Path to benchmark JSON file")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    results = load_results(args.input)
    out_dir = args.output_dir or os.path.dirname(args.input) or "."
    base = os.path.splitext(os.path.basename(args.input))[0]

    plot_roofline(results, os.path.join(out_dir, f"{base}_roofline.png"))
    plot_latency(results, os.path.join(out_dir, f"{base}_latency.png"))
    plot_batch_scaling(results, os.path.join(out_dir, f"{base}_batch_scaling.png"))


if __name__ == "__main__":
    main()
