#!/usr/bin/env python3
"""Compare benchmark results and produce a markdown table + bottleneck summary.

Reads result JSON files from results/ and produces:
  1. Markdown comparison table
  2. Percentage deltas vs bf16 baseline
  3. Bottleneck summary
  4. Exact rerun commands

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py --results-dir results/ --output results/comparison.md
    python scripts/compare_results.py --files results/baseline_bf16*.json results/tiered*.json
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="Compare KV benchmark results")
    p.add_argument("--results-dir", default="results/",
                   help="Directory containing result JSON files")
    p.add_argument("--files", nargs="*", default=None,
                   help="Specific JSON files to compare")
    p.add_argument("--output", default="results/comparison.md",
                   help="Output markdown file")
    p.add_argument("--format", choices=["markdown", "csv"], default="markdown",
                   help="Output format")
    return p.parse_args()


def load_results(results_dir, specific_files=None):
    """Load and validate result JSON files."""
    if specific_files:
        files = specific_files
    else:
        files = sorted(
            glob.glob(os.path.join(results_dir, "baseline_*.json"))
            + glob.glob(os.path.join(results_dir, "tiered_*.json"))
        )

    results = []
    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
            # Basic validation
            if "run_id" not in data or "metrics" not in data:
                print(f"WARNING: Skipping {fp} — missing run_id or metrics")
                continue
            data["_source_file"] = fp
            results.append(data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"WARNING: Skipping {fp} — {e}")

    return results


def format_val(v, fmt=".1f"):
    """Format a value for the table, handling None."""
    if v is None:
        return "—"
    try:
        return f"{v:{fmt}}"
    except (ValueError, TypeError):
        return str(v)


def build_comparison_table(results):
    """Build a markdown comparison table from results."""
    if not results:
        return "No results to compare.\n"

    headers = [
        "Run ID", "KV Mode", "Tiered", "Context", "Requests",
        "TTFT p50", "TTFT p95", "TPOT p50", "TPOT p95",
        "Throughput", "Peak HBM", "Power", "Cache Hit",
    ]

    rows = []
    for r in results:
        m = r.get("metrics", {})
        t = r.get("tiering", {})
        model = r.get("model", {})
        wl = r.get("workload", {})

        kv_mode = model.get("kv_mode", model.get("kv_mode_requested", "?"))
        tiered = "yes" if t.get("enabled") else "no"

        rows.append([
            r.get("run_id", "?")[:40],
            kv_mode,
            tiered,
            str(model.get("context_length", "?")),
            str(wl.get("requests", "?")),
            format_val(m.get("ttft_ms_p50")),
            format_val(m.get("ttft_ms_p95")),
            format_val(m.get("tpot_ms_p50")),
            format_val(m.get("tpot_ms_p95")),
            format_val(m.get("throughput_tokens_per_s")),
            format_val(m.get("peak_hbm_gb"), ".2f"),
            format_val(m.get("gpu_power_w_avg")),
            format_val(m.get("cache_hit_rate"), ".2f"),
        ])

    # Compute column widths
    widths = [max(len(h), max((len(row[i]) for row in rows), default=0))
              for i, h in enumerate(headers)]

    # Build table
    lines = []
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |")

    return "\n".join(lines)


def compute_deltas(results):
    """Compute percentage deltas vs the bf16 baseline."""
    # Find bf16 baseline
    baseline = None
    for r in results:
        kv = r.get("model", {}).get("kv_mode", "")
        tiered = r.get("tiering", {}).get("enabled", False)
        if "bf16" in kv and not tiered:
            baseline = r
            break

    if not baseline:
        return "No bf16 baseline found for delta computation.\n"

    bm = baseline.get("metrics", {})
    lines = ["\n## Deltas vs BF16 Baseline\n"]
    lines.append(f"Baseline: {baseline.get('run_id', '?')}\n")

    metrics_to_compare = [
        ("ttft_ms_p50", "TTFT p50", "lower_better"),
        ("ttft_ms_p95", "TTFT p95", "lower_better"),
        ("tpot_ms_p50", "TPOT p50", "lower_better"),
        ("tpot_ms_p95", "TPOT p95", "lower_better"),
        ("throughput_tokens_per_s", "Throughput", "higher_better"),
        ("peak_hbm_gb", "Peak HBM", "lower_better"),
        ("gpu_power_w_avg", "GPU Power", "lower_better"),
    ]

    for r in results:
        if r is baseline:
            continue
        rm = r.get("metrics", {})
        kv = r.get("model", {}).get("kv_mode", "?")
        tiered = "tiered" if r.get("tiering", {}).get("enabled") else ""
        policy = r.get("tiering", {}).get("promotion_policy", "")
        label = f"{kv} {tiered} {policy}".strip()

        lines.append(f"\n**{label}** ({r.get('run_id', '?')[:30]}):")
        for key, name, direction in metrics_to_compare:
            bval = bm.get(key)
            rval = rm.get(key)
            if bval is None or rval is None or bval == 0:
                lines.append(f"  - {name}: N/A")
                continue
            delta_pct = ((rval - bval) / bval) * 100
            sign = "+" if delta_pct > 0 else ""
            good = (delta_pct < 0) if direction == "lower_better" else (delta_pct > 0)
            marker = "✓" if good else "✗"
            lines.append(f"  - {name}: {sign}{delta_pct:.1f}% {marker}")

    return "\n".join(lines)


def bottleneck_summary(results):
    """Identify key bottlenecks and wins."""
    if not results:
        return "No results to analyze.\n"

    lines = ["\n## Bottleneck Summary\n"]

    # Find best/worst for key metrics
    best_throughput = max(results,
                         key=lambda r: r.get("metrics", {}).get("throughput_tokens_per_s") or 0)
    lowest_hbm = min(results,
                     key=lambda r: r.get("metrics", {}).get("peak_hbm_gb") or float("inf"))

    bt = best_throughput.get("metrics", {}).get("throughput_tokens_per_s")
    lh = lowest_hbm.get("metrics", {}).get("peak_hbm_gb")

    if bt:
        lines.append(f"- **Best throughput:** {best_throughput.get('run_id', '?')[:30]} "
                      f"({bt:.1f} tok/s)")
    if lh:
        lines.append(f"- **Lowest HBM:** {lowest_hbm.get('run_id', '?')[:30]} "
                      f"({lh:.2f} GB)")

    # Check tiered results
    tiered = [r for r in results if r.get("tiering", {}).get("enabled")]
    non_tiered = [r for r in results if not r.get("tiering", {}).get("enabled")]

    if tiered and non_tiered:
        best_non_tiered_ttft = min(
            (r.get("metrics", {}).get("ttft_ms_p50") or float("inf")
             for r in non_tiered)
        )
        for t in tiered:
            t_ttft = t.get("metrics", {}).get("ttft_ms_p50")
            policy = t.get("tiering", {}).get("promotion_policy", "?")
            if t_ttft and best_non_tiered_ttft < float("inf"):
                delta = ((t_ttft - best_non_tiered_ttft) / best_non_tiered_ttft) * 100
                sign = "+" if delta > 0 else ""
                lines.append(f"- **Tiered ({policy}) TTFT vs best baseline:** "
                              f"{sign}{delta:.1f}%")

            # Check tiering-specific metrics
            tier_info = t.get("tiering", {})
            improvement = tier_info.get("ttft_improvement_pct")
            if improvement is not None:
                lines.append(f"- **Cold→Warm TTFT improvement ({policy}):** "
                              f"{improvement:.1f}%")

    # PRD target check
    lines.append("\n### PRD Target Check")
    bf16_hbm = None
    for r in non_tiered:
        if "bf16" in r.get("model", {}).get("kv_mode", ""):
            bf16_hbm = r.get("metrics", {}).get("peak_hbm_gb")
            break

    if bf16_hbm and tiered:
        for t in tiered:
            t_hbm = t.get("metrics", {}).get("peak_hbm_gb")
            if t_hbm and bf16_hbm > 0:
                reduction = (1 - t_hbm / bf16_hbm) * 100
                met = "MET" if reduction >= 20 else "NOT MET"
                lines.append(f"- HBM reduction: {reduction:.1f}% (target ≥20%: **{met}**)")

    return "\n".join(lines)


def rerun_commands(results):
    """Extract rerun commands from results."""
    lines = ["\n## Rerun Commands\n"]
    lines.append("```bash")
    for r in results:
        cmd = r.get("rerun_command")
        if cmd:
            lines.append(cmd)
            lines.append("")
    lines.append("```")
    return "\n".join(lines)


def main():
    args = parse_args()

    print(f"=== KV Benchmark Comparison ===")
    results = load_results(args.results_dir, args.files)
    print(f"Loaded {len(results)} result files")

    if not results:
        print("No results found. Run benchmarks first.")
        return

    # Build output
    output_parts = []
    output_parts.append(f"# KV Cache Benchmark Comparison")
    output_parts.append(f"\nGenerated: {datetime.now().isoformat()}\n")
    output_parts.append("## Results Table\n")
    output_parts.append(build_comparison_table(results))
    output_parts.append(compute_deltas(results))
    output_parts.append(bottleneck_summary(results))
    output_parts.append(rerun_commands(results))

    full_output = "\n".join(output_parts)

    # Write to file
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(full_output)

    # Print to stdout
    print(full_output)
    print(f"\nComparison written to {args.output}")


if __name__ == "__main__":
    main()
