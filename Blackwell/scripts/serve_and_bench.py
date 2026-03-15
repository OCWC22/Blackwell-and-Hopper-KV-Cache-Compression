#!/usr/bin/env python3
"""Online serving benchmark — concurrent user sweeps.

Primary path: TensorRT-LLM with NVFP4 KV cache (via trtllm-serve or Triton).
Follow-up path: vLLM with FP8 KV cache + optional LMCache.

Supports Scenario 2 (more sessions on one GPU), Scenario 3 (longer context +
more sessions on one GPU), and Scenario 4 (one node) depending on configuration.

Answers the primary hackathon question:
  At the same p95 latency target, how many more concurrent sessions can
  one B200 GPU (or one 8xB200 node) serve with tiered KV?

Approach:
  1. Launch serving engine (TRT-LLM or vLLM) with configurable KV dtype
  2. Wait for server health
  3. Drive concurrent requests (repeated-prefix workload)
  4. Sweep concurrency levels, stop when p95 TPOT exceeds threshold
  5. Record TTFT, TPOT, throughput, peak HBM, power, tokens/joule
  6. Shut down server

Usage:
    # TRT-LLM primary path — NVFP4
    python scripts/serve_and_bench.py --engine tensorrt_llm --kv-mode nvfp4 --tp 1

    # TRT-LLM with host offload
    python scripts/serve_and_bench.py --engine tensorrt_llm --kv-mode nvfp4 --offload --tp 1

    # vLLM follow-up path
    python scripts/serve_and_bench.py --engine vllm --kv-mode fp8 --tp 1

    # vLLM + LMCache follow-up
    python scripts/serve_and_bench.py --engine vllm --kv-mode fp8 --use-lmcache --tp 1

    # Full 8xB200 node (Scenario 4)
    python scripts/serve_and_bench.py --engine tensorrt_llm --kv-mode nvfp4 --tp 8

    # Concurrency sweep
    python scripts/serve_and_bench.py --engine tensorrt_llm --kv-mode nvfp4 \\
        --sweep-concurrency 1,2,4,8,16,32 --p95-tpot-limit-ms 100
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_bench_utils import (
    PowerSampler, get_gpu_info, get_cuda_version,
    make_result_template, write_result_json, percentile, generate_run_id,
    tokens_per_joule, load_workload_file_prompts_only,
)


def parse_args():
    p = argparse.ArgumentParser(description="Online serving benchmark with concurrent user sweeps")
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                   help="Model name or path")
    p.add_argument("--engine", choices=["tensorrt_llm", "vllm"], default="tensorrt_llm",
                   help="Serving engine (tensorrt_llm is primary)")
    p.add_argument("--engine-dir", default=None,
                   help="Pre-built TRT-LLM engine directory")
    p.add_argument("--kv-mode", choices=["bf16", "fp8", "nvfp4"], default="nvfp4",
                   help="KV cache dtype (nvfp4 is primary Blackwell thesis)")
    p.add_argument("--offload", action="store_true",
                   help="Enable host memory offload (TRT-LLM path)")
    p.add_argument("--offload-size", type=float, default=20.0,
                   help="Host cache size in GB for offload")
    p.add_argument("--use-lmcache", action="store_true",
                   help="Enable LMCache CPU offloading (vLLM follow-up path)")
    p.add_argument("--lmcache-config", default="configs/lmcache_config.yaml",
                   help="Path to LMCache config YAML")
    p.add_argument("--lmcache-cpu-size", type=float, default=20.0,
                   help="Max CPU memory for LMCache in GB")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallel size (1=single-GPU, 8=full 8xB200 node)")
    p.add_argument("--context-length", type=int, default=8192,
                   help="Context length for prompts")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Max output tokens per request")
    p.add_argument("--prefix-ratio", type=float, default=0.8,
                   help="Shared prefix ratio for repeated_prefix workload")
    p.add_argument("--requests-per-level", type=int, default=20,
                   help="Number of requests per concurrency level")
    p.add_argument("--sweep-concurrency", default="1,2,4,8,16,32",
                   help="Comma-separated concurrency levels to sweep")
    p.add_argument("--p95-tpot-limit-ms", type=float, default=100.0,
                   help="Stop sweep when p95 TPOT exceeds this (ms)")
    p.add_argument("--host", default="127.0.0.1", help="Server bind host")
    p.add_argument("--port", type=int, default=8000, help="Server bind port")
    p.add_argument("--output", default=None, help="Output JSON path")
    p.add_argument("--run-id", default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--scenario-id", default=None,
                   help="Scenario ID (auto-detected from tp and lmcache flags)")
    p.add_argument("--workload-file", default=None,
                   help="Path to JSONL workload file (overrides inline generation)")
    return p.parse_args()


def build_server_cmd(args):
    """Build the serve command for the selected engine."""
    if args.engine == "tensorrt_llm":
        return build_trtllm_server_cmd(args)
    else:
        return build_vllm_server_cmd(args)


def build_trtllm_server_cmd(args):
    """Build the TRT-LLM serve command (trtllm-serve or python -m tensorrt_llm.serve)."""
    cmd = [
        "trtllm-serve",
        args.engine_dir or args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tp_size", str(args.tp),
        "--max_seq_len", str(args.context_length),
    ]

    if args.kv_mode == "fp8":
        cmd.extend(["--kv_cache_type", "fp8"])
    elif args.kv_mode == "nvfp4":
        cmd.extend(["--kv_cache_type", "nvfp4"])

    if args.offload:
        cmd.extend(["--host_cache_size", str(int(args.offload_size * 1024 * 1024 * 1024))])

    return cmd


def build_vllm_server_cmd(args):
    """Build the vLLM serve command."""
    kv_dtype_map = {"bf16": "auto", "fp8": "fp8", "nvfp4": "nvfp4"}
    kv_dtype = kv_dtype_map.get(args.kv_mode, "auto")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--kv-cache-dtype", kv_dtype,
        "--tensor-parallel-size", str(args.tp),
        "--max-model-len", str(args.context_length),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enable-prefix-caching",
        "--host", args.host,
        "--port", str(args.port),
    ]

    if args.use_lmcache:
        kv_config = json.dumps({
            "kv_connector": "LMCacheConnectorV1",
            "kv_role": "kv_both",
        })
        cmd.extend(["--kv-transfer-config", kv_config])

    return cmd


def build_server_env(args):
    """Build environment for the server process."""
    env = os.environ.copy()
    if args.engine == "vllm" and args.use_lmcache:
        abs_config = os.path.abspath(args.lmcache_config)
        if os.path.exists(abs_config):
            env["LMCACHE_CONFIG_FILE"] = abs_config
        env["LMCACHE_LOCAL_CPU"] = "True"
        env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(args.lmcache_cpu_size)
        env["LMCACHE_CHUNK_SIZE"] = "256"
    return env


async def wait_for_server(host, port, timeout=300):
    """Wait for vLLM server to be ready."""
    import urllib.request
    url = f"http://{host}:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(2)
    return False


def generate_prompts(context_length, num_requests, prefix_ratio=0.8):
    """Generate repeated-prefix prompts for the benchmark."""
    chars_per_token = 4
    prefix_tokens = int(context_length * prefix_ratio)
    suffix_tokens = context_length - prefix_tokens

    shared_prefix = (
        "You are an expert AI assistant analyzing a comprehensive technical "
        "document about distributed systems, caching strategies, and GPU "
        "memory management. The document covers key-value cache tiering, "
        "promotion policies, and quality-latency tradeoffs in serving. "
        * (prefix_tokens * chars_per_token // 250 + 1)
    )[:prefix_tokens * chars_per_token]

    prompts = []
    for i in range(num_requests):
        suffix = (
            f"\n\nQuestion {i}: Provide a detailed analysis of section {i + 1}, "
            f"including specific metrics and implementation considerations. "
            * (suffix_tokens * chars_per_token // 100 + 1)
        )[:suffix_tokens * chars_per_token]
        prompts.append(shared_prefix + suffix)

    return prompts, prefix_tokens


async def send_request(session, url, prompt, max_tokens):
    """Send a single chat completion request and measure timing."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    t0 = time.perf_counter()
    ttft = None
    output_tokens = 0

    try:
        async with session.post(url, json=payload) as resp:
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content and ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000  # ms
                        if content:
                            output_tokens += 1  # rough: one chunk ≈ one token
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return {"error": str(e), "ttft_ms": None, "tpot_ms": None, "output_tokens": 0}

    total_time = time.perf_counter() - t0
    if ttft is None:
        ttft = total_time * 1000

    decode_time_ms = (total_time * 1000) - ttft
    tpot = decode_time_ms / output_tokens if output_tokens > 0 else 0

    return {
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "output_tokens": output_tokens,
        "total_time_s": total_time,
    }


async def run_concurrent_requests(host, port, prompts, max_tokens, concurrency):
    """Drive concurrent requests and collect per-request metrics."""
    try:
        import aiohttp
    except ImportError:
        print("ERROR: aiohttp required. Install with: pip install aiohttp")
        sys.exit(1)

    url = f"http://{host}:{port}/v1/chat/completions"
    results = []

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(concurrency)

        async def bounded_request(prompt):
            async with sem:
                return await send_request(session, url, prompt, max_tokens)

        tasks = [bounded_request(p) for p in prompts]
        results = await asyncio.gather(*tasks)

    return list(results)


async def run_sweep(args):
    """Run the full concurrency sweep."""
    concurrency_levels = [int(c) for c in args.sweep_concurrency.split(",")]

    # Auto-detect scenario_id
    if args.scenario_id is None:
        if args.tp > 1:
            args.scenario_id = "scenario_4_longer_context_more_sessions_node"
        elif args.offload or args.use_lmcache:
            args.scenario_id = "scenario_3_longer_context_more_sessions_gpu"
        else:
            args.scenario_id = "scenario_2_more_sessions_gpu"

    print(f"=== Online Serving Benchmark ===")
    print(f"Model: {args.model}")
    print(f"KV mode: {args.kv_mode}")
    print(f"LMCache: {args.use_lmcache}")
    print(f"TP: {args.tp}")
    print(f"Context: {args.context_length}")
    print(f"Concurrency sweep: {concurrency_levels}")
    print(f"p95 TPOT limit: {args.p95_tpot_limit_ms} ms")
    print()

    # Build and launch server
    cmd = build_server_cmd(args)
    env = build_server_env(args)
    print(f"Launching server: {' '.join(cmd[:8])}...")
    server_proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    try:
        # Wait for server
        print("Waiting for server health...")
        healthy = await wait_for_server(args.host, args.port, timeout=300)
        if not healthy:
            print("ERROR: Server did not become healthy within timeout.")
            return None

        print("Server is ready.\n")

        # Generate prompts
        if args.workload_file:
            print(f"Loading workload from {args.workload_file}...")
            prompts, prefix_tokens = load_workload_file_prompts_only(
                args.workload_file, args.requests_per_level
            )
        else:
            prompts, prefix_tokens = generate_prompts(
                args.context_length, args.requests_per_level, args.prefix_ratio
            )

        gpu_info = get_gpu_info()
        cuda_ver = get_cuda_version()

        # Start power sampling
        power_sampler = PowerSampler(interval_s=0.5)
        power_sampler.start()

        sweep_results = []
        max_concurrent_at_target = 0
        total_start = time.perf_counter()

        for conc in concurrency_levels:
            print(f"--- Concurrency: {conc} ---")
            level_results = await run_concurrent_requests(
                args.host, args.port, prompts, args.max_tokens, conc
            )

            # Filter out errors
            valid = [r for r in level_results if "error" not in r]
            if not valid:
                print(f"  All requests failed at concurrency {conc}. Stopping.")
                break

            ttft_vals = [r["ttft_ms"] for r in valid if r["ttft_ms"] is not None]
            tpot_vals = [r["tpot_ms"] for r in valid if r["tpot_ms"] is not None and r["tpot_ms"] > 0]
            total_tokens = sum(r["output_tokens"] for r in valid)

            ttft_p50 = percentile(ttft_vals, 50)
            ttft_p95 = percentile(ttft_vals, 95)
            tpot_p50 = percentile(tpot_vals, 50)
            tpot_p95 = percentile(tpot_vals, 95)
            tpot_p99 = percentile(tpot_vals, 99)

            level_time = sum(r.get("total_time_s", 0) for r in valid) / max(len(valid), 1)
            throughput = total_tokens / level_time if level_time > 0 else 0

            level_data = {
                "concurrency": conc,
                "requests": len(valid),
                "errors": len(level_results) - len(valid),
                "ttft_ms_p50": ttft_p50,
                "ttft_ms_p95": ttft_p95,
                "tpot_ms_p50": tpot_p50,
                "tpot_ms_p95": tpot_p95,
                "tpot_ms_p99": tpot_p99,
                "throughput_tokens_per_s": round(throughput, 2),
                "total_output_tokens": total_tokens,
            }
            sweep_results.append(level_data)

            print(f"  TTFT p50={ttft_p50:.1f}ms p95={ttft_p95:.1f}ms")
            print(f"  TPOT p50={tpot_p50:.1f}ms p95={tpot_p95:.1f}ms p99={tpot_p99:.1f}ms")
            print(f"  Throughput: {throughput:.1f} tok/s")

            if tpot_p95 is not None and tpot_p95 <= args.p95_tpot_limit_ms:
                max_concurrent_at_target = conc

            # Stop if p95 TPOT exceeded
            if tpot_p95 is not None and tpot_p95 > args.p95_tpot_limit_ms:
                print(f"  p95 TPOT ({tpot_p95:.1f}ms) exceeds limit "
                      f"({args.p95_tpot_limit_ms}ms). Stopping sweep.")
                break

        total_time = time.perf_counter() - total_start
        power_sampler.stop()

        # Get peak HBM from nvidia-smi
        peak_hbm = None
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5
            ).strip()
            vals = [float(v.strip()) / 1024 for v in out.splitlines() if v.strip()]
            peak_hbm = max(vals) if vals else None
        except Exception:
            pass

        # Compute energy metrics
        total_output_tokens = sum(r["total_output_tokens"] for r in sweep_results)
        avg_power = power_sampler.average()
        tpj = tokens_per_joule(total_output_tokens, avg_power, total_time)

        # Build result
        run_id = args.run_id or generate_run_id(args.kv_mode, args.context_length, prefix="serve")

        result = make_result_template()
        result["run_id"] = run_id
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["scenario_id"] = args.scenario_id
        result["serving_mode"] = "online"

        result["runtime"] = {
            "engine": args.engine,
            "engine_version": "unknown",  # filled from server output if available
            "cuda_version": cuda_ver,
            "driver_version": gpu_info["driver_version"],
            "gpu_name": gpu_info["gpu_model"],
            "gpu_count": gpu_info["gpu_count"],
            "node_count": 1,
        }

        result["model"] = {
            "name": args.model,
            "kv_mode": args.kv_mode,
            "context_length": args.context_length,
        }

        result["workload"] = {
            "type": "repeated_prefix",
            "requests_per_level": args.requests_per_level,
            "concurrency_sweep": concurrency_levels,
            "shared_prefix_tokens": prefix_tokens,
            "generated_tokens_per_request": args.max_tokens,
        }

        # Use the best concurrency level for headline metrics
        best = sweep_results[-1] if sweep_results else {}
        result["metrics"] = {
            "ttft_ms_p50": best.get("ttft_ms_p50"),
            "ttft_ms_p95": best.get("ttft_ms_p95"),
            "tpot_ms_p50": best.get("tpot_ms_p50"),
            "tpot_ms_p95": best.get("tpot_ms_p95"),
            "tpot_ms_p99": best.get("tpot_ms_p99"),
            "throughput_tokens_per_s": best.get("throughput_tokens_per_s"),
            "peak_hbm_gb": round(peak_hbm, 3) if peak_hbm else None,
            "gpu_power_w_avg": round(avg_power, 1),
            "tokens_per_joule": round(tpj, 4) if tpj else None,
            "max_concurrent_at_p95_target": max_concurrent_at_target,
            "cache_hit_rate": None,
            "promotion_latency_ms_p50": None,
            "promotion_latency_ms_p95": None,
            "quality_metric_name": None,
            "quality_metric_value": None,
            "quality_delta_vs_best_baseline": None,
        }

        offload_enabled = args.offload if args.engine == "tensorrt_llm" else args.use_lmcache
        result["tiering"] = {
            "enabled": offload_enabled,
            "hot_tier_format": args.kv_mode,
            "cold_tier_format": "trtllm_host_cache" if (args.engine == "tensorrt_llm" and args.offload) else (
                "lmcache_cpu" if args.use_lmcache else None
            ),
            "offload_mechanism": args.engine,
            "lmcache_enabled": args.use_lmcache if args.engine == "vllm" else False,
            "promotion_policy": "demand",
        }

        result["sweep_results"] = sweep_results

        lmcache_flags = ""
        if args.use_lmcache:
            lmcache_flags = (
                f"--use-lmcache "
                f"--lmcache-config {args.lmcache_config} "
                f"--lmcache-cpu-size {args.lmcache_cpu_size} "
            )

        result["rerun_command"] = (
            f"python scripts/serve_and_bench.py "
            f"--model {args.model} "
            f"--kv-mode {args.kv_mode} "
            f"{lmcache_flags}"
            f"--tp {args.tp} "
            f"--context-length {args.context_length} "
            f"--max-tokens {args.max_tokens} "
            f"--requests-per-level {args.requests_per_level} "
            f"--sweep-concurrency {args.sweep_concurrency} "
            f"--p95-tpot-limit-ms {args.p95_tpot_limit_ms} "
            f"--output {args.output or 'auto'}"
        )

        # Write output
        if args.output is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            lmc = "_lmcache" if args.use_lmcache else ""
            args.output = f"results/serve_{args.kv_mode}{lmc}_tp{args.tp}_{ts}.json"

        write_result_json(result, args.output)

        # Print summary
        print(f"\n=== Serving Benchmark Summary ===")
        print(f"KV mode: {args.kv_mode}")
        print(f"LMCache: {args.use_lmcache}")
        print(f"TP: {args.tp}")
        print(f"Max concurrent sessions at p95 target: {max_concurrent_at_target}")
        if peak_hbm:
            print(f"Peak HBM: {peak_hbm:.2f} GB")
        print(f"Avg GPU power: {avg_power:.1f} W")
        print(f"Tokens/joule: {tpj:.4f}")
        print(f"Total sweep time: {total_time:.1f} s")
        print(f"Result: {args.output}")

        return result

    finally:
        # Shut down server
        print("\nShutting down server...")
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()


def main():
    args = parse_args()
    asyncio.run(run_sweep(args))


if __name__ == "__main__":
    main()
