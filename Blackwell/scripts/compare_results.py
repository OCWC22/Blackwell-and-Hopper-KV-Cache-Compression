#!/usr/bin/env python3
"""Compare benchmark results and produce a markdown table + bottleneck summary.

Supports comparing results across engines (TensorRT-LLM primary, vLLM follow-up).

Reads result JSON files from results/ and produces:
  1. Markdown comparison table (grouped by scenario_id when available)
  2. Percentage deltas vs bf16 baseline
  3. Bottleneck summary with serving KPI highlights
  4. PRD target checks (HBM, sessions, latency regression, quality)
  5. Exact rerun commands

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
    p.add_argument("--group-by", choices=["scenario", "prompt_bucket", "variant"],
                   default="scenario",
                   help="Group results by scenario_id, prompt_bucket_tokens, or variant")
    p.add_argument("--emit-csv", action="store_true",
                   help="Also emit a CSV file alongside the markdown")
    return p.parse_args()


def load_results(results_dir, specific_files=None):
    """Load and validate result JSON files."""
    if specific_files:
        files = specific_files
    else:
        files = sorted(
            glob.glob(os.path.join(results_dir, "baseline_*.json"))
            + glob.glob(os.path.join(results_dir, "tiered_*.json"))
            + glob.glob(os.path.join(results_dir, "serve_*.json"))
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
        "Run ID", "Engine", "Scenario", "KV Mode", "Tiered", "Context",
        "Conc", "Requests",
        "TTFT p50", "TTFT p95", "TPOT p50", "TPOT p95",
        "Throughput", "Peak HBM", "Power", "Cache Hit",
    ]

    rows = []
    for r in results:
        m = r.get("metrics", {})
        t = r.get("tiering", {})
        model = r.get("model", {})
        wl = r.get("workload", {})
        rt = r.get("runtime", {})

        engine = rt.get("engine", "?")
        kv_mode = model.get("kv_mode", model.get("kv_mode_requested", "?"))
        tiered = "yes" if t.get("enabled") else "no"
        concurrency = str(wl.get("concurrency", 1))

        scenario = r.get("scenario_id", "—")
        if scenario and len(scenario) > 15:
            scenario = scenario.replace("scenario_", "S").replace("_longer_context_more_sessions_", "_lc_ms_").replace("_longer_context_", "_lc_").replace("_more_sessions_", "_ms_")

        rows.append([
            r.get("run_id", "?")[:40],
            engine[:12],
            scenario[:20],
            kv_mode,
            tiered,
            str(model.get("context_length", "?")),
            concurrency,
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

    # Serving KPI highlights
    lines.append("\n### Serving KPI Highlights")
    for r in results:
        max_conc = r.get("metrics", {}).get("max_concurrent_at_p95_target")
        tpj = r.get("metrics", {}).get("tokens_per_joule")
        if max_conc is not None:
            label = r.get("run_id", "?")[:30]
            lines.append(f"- **Max concurrent at p95 target ({label}):** {max_conc}")
        if tpj is not None and tpj > 0:
            label = r.get("run_id", "?")[:30]
            lines.append(f"- **Tokens/joule ({label}):** {tpj:.4f}")

    # PRD target check
    lines.append("\n### PRD Target Check")
    bf16_baseline = None
    for r in non_tiered:
        if "bf16" in r.get("model", {}).get("kv_mode", ""):
            bf16_baseline = r
            break

    bf16_hbm = bf16_baseline.get("metrics", {}).get("peak_hbm_gb") if bf16_baseline else None
    bf16_tpot_p95 = bf16_baseline.get("metrics", {}).get("tpot_ms_p95") if bf16_baseline else None

    if bf16_hbm and tiered:
        for t in tiered:
            t_hbm = t.get("metrics", {}).get("peak_hbm_gb")
            if t_hbm and bf16_hbm > 0:
                reduction = (1 - t_hbm / bf16_hbm) * 100
                met = "MET" if reduction >= 20 else "NOT MET"
                lines.append(f"- HBM reduction: {reduction:.1f}% (target ≥20%: **{met}**)")

    if bf16_tpot_p95 and tiered:
        for t in tiered:
            t_tpot = t.get("metrics", {}).get("tpot_ms_p95")
            if t_tpot and bf16_tpot_p95 > 0:
                regression = ((t_tpot - bf16_tpot_p95) / bf16_tpot_p95) * 100
                met = "MET" if regression <= 10 else "NOT MET"
                lines.append(f"- p95 TPOT regression: {regression:+.1f}% (target ≤10%: **{met}**)")

    for t in tiered:
        q_delta = t.get("metrics", {}).get("quality_delta_vs_best_baseline")
        if q_delta is not None:
            met = "MET" if abs(q_delta) <= 1.0 else "NOT MET"
            lines.append(f"- Quality delta: {q_delta:.2f}% (target ≤1%: **{met}**)")

    # Max concurrent sessions comparison
    baseline_conc = None
    tiered_conc = None
    for r in non_tiered:
        c = r.get("metrics", {}).get("max_concurrent_at_p95_target")
        if c is not None and (baseline_conc is None or c > baseline_conc):
            baseline_conc = c
    for t in tiered:
        c = t.get("metrics", {}).get("max_concurrent_at_p95_target")
        if c is not None and (tiered_conc is None or c > tiered_conc):
            tiered_conc = c
    if baseline_conc and tiered_conc and baseline_conc > 0:
        improvement = ((tiered_conc - baseline_conc) / baseline_conc) * 100
        met = "MET" if improvement >= 25 else "NOT MET"
        lines.append(f"- Session improvement: {improvement:+.1f}% (target ≥25%: **{met}**)")

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

    # Emit CSV alongside markdown if requested
    if args.emit_csv:
        csv_path = args.output.replace(".md", ".csv")
        with open(csv_path, "w") as f:
            csv_headers = [
                "run_id", "engine", "scenario_id", "kv_mode", "tiered",
                "context_length", "concurrency", "requests",
                "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms",
                "throughput_tps", "peak_hbm_gb", "power_w", "cache_hit_rate",
            ]
            f.write(",".join(csv_headers) + "\n")
            for r in results:
                m = r.get("metrics", {})
                t = r.get("tiering", {})
                model = r.get("model", {})
                wl = r.get("workload", {})
                rt = r.get("runtime", {})
                vals = [
                    r.get("run_id", ""),
                    rt.get("engine", ""),
                    r.get("scenario_id", ""),
                    model.get("kv_mode", ""),
                    "yes" if t.get("enabled") else "no",
                    str(model.get("context_length", "")),
                    str(wl.get("concurrency", 1)),
                    str(wl.get("requests", "")),
                    str(m.get("ttft_ms_p50", "")),
                    str(m.get("ttft_ms_p95", "")),
                    str(m.get("tpot_ms_p50", "")),
                    str(m.get("tpot_ms_p95", "")),
                    str(m.get("throughput_tokens_per_s", "")),
                    str(m.get("peak_hbm_gb", "")),
                    str(m.get("gpu_power_w_avg", "")),
                    str(m.get("cache_hit_rate", "")),
                ]
                f.write(",".join(vals) + "\n")
        print(f"CSV written to {csv_path}")

    # Print to stdout
    print(full_output)
    print(f"\nComparison written to {args.output}")


if __name__ == "__main__":
    main()
