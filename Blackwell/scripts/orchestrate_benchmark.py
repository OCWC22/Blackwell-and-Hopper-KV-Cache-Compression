#!/usr/bin/env python3
"""Benchmark orchestrator — drives the eval matrix through harness scripts.

Reads configs/blackwell_eval_matrix.tsv and dispatches each row to the
correct benchmark script with the right flags and workload file.

Usage:
    # Dry run — print all commands without executing
    python scripts/orchestrate_benchmark.py --dry-run

    # Run Scenario 1 only
    python scripts/orchestrate_benchmark.py \\
        --filter-scenario scenario_1_longer_context_gpu \\
        --data-dir data/synthetic/

    # Run specific variants
    python scripts/orchestrate_benchmark.py \\
        --filter-variant trtllm_bf16,trtllm_fp8,trtllm_nvfp4

    # Slurm submission mode
    python scripts/orchestrate_benchmark.py --slurm --slurm-partition gpu
"""

import argparse
import csv
import fnmatch
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalMatrixRow:
    """One row from blackwell_eval_matrix.tsv."""
    scenario_id: str
    variant: str
    model: str
    context_length: int
    concurrency: int
    workload: str
    notes: str


def load_eval_matrix(path: str) -> list[EvalMatrixRow]:
    """Parse the TSV eval matrix into typed rows."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(EvalMatrixRow(
                scenario_id=r["scenario_id"],
                variant=r["variant"],
                model=r["model"],
                context_length=int(r["context_length"]),
                concurrency=int(r["concurrency"]),
                workload=r["workload"],
                notes=r.get("notes", ""),
            ))
    return rows


def parse_variant(variant: str) -> dict:
    """Decompose a variant string from the eval matrix.

    Examples:
        "trtllm_bf16"               -> {engine: tensorrt_llm, kv_mode: bf16, ...}
        "trtllm_nvfp4_offload"      -> {engine: tensorrt_llm, kv_mode: nvfp4, offload: True}
        "trtllm_nvfp4_offload_kvtc" -> {engine: tensorrt_llm, kv_mode: nvfp4, offload: True, kvtc: True}
        "vllm_fp8"                  -> {engine: vllm, kv_mode: fp8}
        "vllm_fp8+lmcache"         -> {engine: vllm, kv_mode: fp8, lmcache: True}
        "tiered_demand"             -> {engine: tensorrt_llm, kv_mode: nvfp4, offload: True, promotion_policy: demand}
        "tiered_eager"              -> {engine: tensorrt_llm, kv_mode: nvfp4, offload: True, promotion_policy: eager}
    """
    result = {
        "engine": "tensorrt_llm",
        "kv_mode": "bf16",
        "offload": False,
        "kvtc": False,
        "lmcache": False,
        "promotion_policy": None,
    }

    v = variant.lower()

    # Handle tiered variants
    if v.startswith("tiered_"):
        policy = v.split("_", 1)[1]
        result["engine"] = "tensorrt_llm"
        result["kv_mode"] = "nvfp4"
        result["offload"] = True
        result["promotion_policy"] = policy
        return result

    # Handle vllm variants
    if v.startswith("vllm_"):
        result["engine"] = "vllm"
        remainder = v[5:]  # strip "vllm_"
        if "+lmcache" in remainder:
            result["lmcache"] = True
            remainder = remainder.replace("+lmcache", "")
        result["kv_mode"] = remainder
        return result

    # Handle trtllm variants
    if v.startswith("trtllm_"):
        remainder = v[7:]  # strip "trtllm_"
        parts = remainder.split("_")

        # First part is always kv_mode
        result["kv_mode"] = parts[0]

        # Check for offload and kvtc flags
        if "offload" in parts:
            result["offload"] = True
        if "kvtc" in parts:
            result["kvtc"] = True

        return result

    # Fallback: treat entire string as variant name
    result["kv_mode"] = variant
    return result


def classify_script(row: EvalMatrixRow, parsed: dict) -> str:
    """Determine which harness script to use for a matrix row.

    Returns: "baseline" | "serve" | "tiered"
    """
    if parsed["promotion_policy"]:
        return "tiered"
    if parsed["kvtc"] or (parsed["offload"] and "offload" in row.variant):
        # Rows with explicit offload+kvtc use the tiered experiment
        if row.concurrency > 1 and not parsed["kvtc"]:
            return "serve"
        return "tiered"
    if row.concurrency > 1:
        return "serve"
    return "baseline"


def resolve_data_file(
    workload: str,
    context_length: int,
    concurrency: int,
    data_dir: str,
) -> str | None:
    """Find a matching JSONL data file for a matrix row."""
    # Try single_turn first (most common for repeated_prefix workload)
    candidates = [
        f"single_turn_{context_length}_c{concurrency}.jsonl",
        f"single_turn_{context_length}_c1.jsonl",  # fallback to c1
        f"multi_turn_{context_length}_c{concurrency}.jsonl",
    ]
    for candidate in candidates:
        path = os.path.join(data_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def build_command(row: EvalMatrixRow, parsed: dict, script: str,
                  data_file: str | None, scripts_dir: str) -> list[str]:
    """Build the CLI command for a matrix row."""
    cmd = [sys.executable]

    if script == "baseline":
        cmd.append(os.path.join(scripts_dir, "run_baseline.py"))
    elif script == "serve":
        cmd.append(os.path.join(scripts_dir, "serve_and_bench.py"))
    elif script == "tiered":
        cmd.append(os.path.join(scripts_dir, "run_tiered_experiment.py"))

    # Common args
    cmd.extend(["--model", row.model])
    cmd.extend(["--context-length", str(row.context_length)])
    cmd.extend(["--engine", parsed["engine"]])
    cmd.extend(["--kv-mode", parsed["kv_mode"]])
    cmd.extend(["--scenario-id", row.scenario_id])

    # Script-specific args
    if script == "baseline":
        cmd.extend(["--requests", "10"])
        cmd.extend(["--concurrency", str(row.concurrency)])
        if parsed["offload"]:
            cmd.append("--offload")
    elif script == "serve":
        cmd.extend(["--sweep-concurrency", str(row.concurrency)])
        if parsed["offload"]:
            cmd.append("--offload")
        if parsed["lmcache"]:
            cmd.append("--use-lmcache")
    elif script == "tiered":
        cmd.extend(["--requests", "10"])
        if parsed["offload"] or parsed["engine"] == "tensorrt_llm":
            cmd.append("--offload-to-host")
        if parsed["kvtc"]:
            cmd.extend(["--cold-tier-codec", "kvtc"])
        if parsed["promotion_policy"]:
            cmd.extend(["--promotion-policy", parsed["promotion_policy"]])
        if parsed["lmcache"]:
            cmd.append("--use-lmcache")

    # Workload file
    if data_file:
        cmd.extend(["--workload-file", data_file])

    return cmd


def wrap_slurm(cmd: list[str], row: EvalMatrixRow, partition: str,
               logs_dir: str) -> list[str]:
    """Wrap a command in an sbatch submission."""
    job_name = f"kv_{row.variant}_{row.context_length}_c{row.concurrency}"
    log_file = os.path.join(logs_dir, f"{job_name}_%j.log")
    sbatch = [
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={partition}",
        "--gres=gpu:1",
        "--time=01:00:00",
        f"--output={log_file}",
        "--wrap", " ".join(cmd),
    ]
    return sbatch


def filter_rows(rows: list[EvalMatrixRow], scenario: str | None,
                variant: str | None) -> list[EvalMatrixRow]:
    """Filter matrix rows by scenario and/or variant glob patterns."""
    filtered = rows
    if scenario:
        patterns = [s.strip() for s in scenario.split(",")]
        filtered = [r for r in filtered
                    if any(fnmatch.fnmatch(r.scenario_id, p) for p in patterns)]
    if variant:
        patterns = [v.strip() for v in variant.split(",")]
        filtered = [r for r in filtered
                    if any(fnmatch.fnmatch(r.variant, p) for p in patterns)]
    return filtered


def parse_args():
    p = argparse.ArgumentParser(description="KV cache benchmark orchestrator")
    p.add_argument("--matrix", default="configs/blackwell_eval_matrix.tsv",
                   help="Path to eval matrix TSV")
    p.add_argument("--data-dir", default="data/synthetic/",
                   help="Directory containing JSONL workload files")
    p.add_argument("--scripts-dir", default="scripts/",
                   help="Directory containing benchmark scripts")
    p.add_argument("--results-dir", default="results/",
                   help="Directory for benchmark results")
    p.add_argument("--logs-dir", default="logs/",
                   help="Directory for Slurm logs")
    p.add_argument("--filter-scenario", default=None,
                   help="Comma-separated scenario ID patterns (supports glob)")
    p.add_argument("--filter-variant", default=None,
                   help="Comma-separated variant patterns (supports glob)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--slurm", action="store_true",
                   help="Submit via sbatch instead of running directly")
    p.add_argument("--slurm-partition", default="gpu",
                   help="Slurm partition for --slurm mode")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths relative to repo root (parent of scripts/)
    repo_root = Path(__file__).resolve().parent.parent
    matrix_path = repo_root / args.matrix
    data_dir = repo_root / args.data_dir
    scripts_dir = repo_root / args.scripts_dir
    results_dir = repo_root / args.results_dir
    logs_dir = repo_root / args.logs_dir

    rows = load_eval_matrix(str(matrix_path))
    print(f"Loaded {len(rows)} rows from {matrix_path}")

    filtered = filter_rows(rows, args.filter_scenario, args.filter_variant)
    print(f"After filtering: {len(filtered)} rows")

    if not filtered:
        print("No rows match the filter. Available scenarios:")
        for s in sorted(set(r.scenario_id for r in rows)):
            print(f"  {s}")
        print("Available variants:")
        for v in sorted(set(r.variant for r in rows)):
            print(f"  {v}")
        return

    # Ensure output dirs exist
    os.makedirs(str(results_dir), exist_ok=True)
    os.makedirs(str(logs_dir), exist_ok=True)

    # Sort: scenario_1 before 2 before 3 before 4; within scenario, by variant
    filtered.sort(key=lambda r: (r.scenario_id, r.variant, r.context_length, r.concurrency))

    summary = []
    for i, row in enumerate(filtered, 1):
        parsed = parse_variant(row.variant)
        script = classify_script(row, parsed)
        data_file = resolve_data_file(
            row.workload, row.context_length, row.concurrency, str(data_dir)
        )

        cmd = build_command(row, parsed, script, data_file, str(scripts_dir))

        if args.slurm:
            cmd = wrap_slurm(cmd, row, args.slurm_partition, str(logs_dir))

        status = "dry-run" if args.dry_run else "pending"
        entry = {
            "index": i,
            "scenario": row.scenario_id,
            "variant": row.variant,
            "model": row.model,
            "context": row.context_length,
            "concurrency": row.concurrency,
            "script": script,
            "data_file": data_file or "(inline)",
            "command": " ".join(cmd),
            "status": status,
        }

        print(f"\n[{i}/{len(filtered)}] {row.variant} | {row.scenario_id} | "
              f"ctx={row.context_length} conc={row.concurrency}")
        print(f"  Script: {script}")
        print(f"  Data:   {data_file or '(inline generation)'}")
        print(f"  Cmd:    {' '.join(cmd)}")

        if not args.dry_run:
            t0 = time.time()
            try:
                result = subprocess.run(cmd, timeout=3600)
                entry["status"] = "success" if result.returncode == 0 else f"failed (rc={result.returncode})"
            except subprocess.TimeoutExpired:
                entry["status"] = "timeout"
            except Exception as e:
                entry["status"] = f"error: {e}"
            entry["duration_s"] = round(time.time() - t0, 1)
            print(f"  Status: {entry['status']}")

        summary.append(entry)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(summary)} rows")
    print(f"{'='*80}")
    print(f"{'#':>3} {'Variant':<30} {'Ctx':>6} {'Conc':>5} {'Script':<10} {'Status':<15}")
    print(f"{'-'*3} {'-'*30} {'-'*6} {'-'*5} {'-'*10} {'-'*15}")
    for e in summary:
        print(f"{e['index']:>3} {e['variant']:<30} {e['context']:>6} "
              f"{e['concurrency']:>5} {e['script']:<10} {e['status']:<15}")


if __name__ == "__main__":
    main()
