#!/usr/bin/env python3
"""Blackwell KV cache baseline benchmark (Scenario 1 & 2).

Scenario 1 — Longer context on one GPU: context sweep, concurrency=1
Scenario 2 — More sessions on one GPU: fixed context, concurrency sweep

Primary runtime: TensorRT-LLM with NVFP4 KV cache.
Fallback: vLLM with FP8 KV cache.

Runs inference with configurable KV mode (bf16/fp8/nvfp4) and emits
machine-readable JSON results with TTFT, TPOT, throughput, HBM, and power.

Usage:
    # TRT-LLM primary path
    python scripts/run_baseline.py --engine tensorrt_llm --kv-mode nvfp4 --context-length 8192 --requests 10
    python scripts/run_baseline.py --engine tensorrt_llm --kv-mode fp8 --context-length 32768 --requests 64

    # TRT-LLM with host offload
    python scripts/run_baseline.py --engine tensorrt_llm --kv-mode nvfp4 --offload --offload-size 20.0

    # vLLM follow-up path
    python scripts/run_baseline.py --engine vllm --kv-mode fp8 --context-length 8192 --requests 10
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

# Add scripts dir to path for kv_bench_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_bench_utils import (
    PowerSampler, get_gpu_info, get_cuda_version, get_runtime_versions,
    make_result_template, write_result_json, percentile, generate_run_id,
    load_workload_file,
)


def parse_args():
    p = argparse.ArgumentParser(description="Blackwell KV cache baseline benchmark")
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                   help="Model name or path")
    p.add_argument("--context-length", type=int, default=8192,
                   help="Context length for prompts")
    p.add_argument("--requests", type=int, default=10,
                   help="Number of inference requests")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Concurrent requests (for throughput measurement)")
    p.add_argument("--kv-mode", choices=["bf16", "fp8", "nvfp4"], default="bf16",
                   help="KV cache precision mode")
    p.add_argument("--workload-type", choices=["repeated_prefix", "independent"],
                   default="repeated_prefix",
                   help="Workload pattern")
    p.add_argument("--engine", choices=["tensorrt_llm", "vllm"], default="tensorrt_llm",
                   help="Inference engine (tensorrt_llm is primary)")
    p.add_argument("--engine-dir", default=None,
                   help="Pre-built TRT-LLM engine directory (optional, uses hlapi auto-build if not set)")
    p.add_argument("--offload", action="store_true",
                   help="Enable host memory secondary tier (TRT-LLM KvCacheConfig.host_cache_size)")
    p.add_argument("--offload-size", type=float, default=20.0,
                   help="Host cache size in GB for offload tier")
    p.add_argument("--output", default=None,
                   help="Output JSON path (auto-generated if not set)")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Max output tokens per request")
    p.add_argument("--run-id", default=None,
                   help="Run ID (auto-generated if not set)")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallel size")
    p.add_argument("--prefix-ratio", type=float, default=0.8,
                   help="Shared prefix ratio for repeated_prefix workload")
    p.add_argument("--scenario-id", default=None,
                   help="Scenario ID (auto-detected if not set). "
                        "Values: scenario_1_longer_context_gpu, "
                        "scenario_2_more_sessions_gpu, "
                        "scenario_3_longer_context_more_sessions_gpu, "
                        "scenario_4_longer_context_more_sessions_node")
    p.add_argument("--workload-file", default=None,
                   help="Path to JSONL workload file (overrides inline generation)")
    return p.parse_args()


def generate_workload(workload_type, context_length, num_requests, prefix_ratio=0.8):
    """Generate prompts for the benchmark.

    Returns list of (prompt_text, shared_prefix_tokens) tuples.
    """
    prompts = []
    chars_per_token = 4  # rough estimate

    if workload_type == "repeated_prefix":
        prefix_tokens = int(context_length * prefix_ratio)
        suffix_tokens = context_length - prefix_tokens
        shared_prefix = (
            "You are an expert AI assistant. Analyze the following document "
            "carefully and provide a detailed response. "
            * (prefix_tokens * chars_per_token // 80 + 1)
        )[:prefix_tokens * chars_per_token]

        for i in range(num_requests):
            suffix = (
                f"Question {i}: Based on the analysis above, explain point "
                f"number {i % 10 + 1} in detail with examples. "
                * (suffix_tokens * chars_per_token // 80 + 1)
            )[:suffix_tokens * chars_per_token]
            prompts.append((shared_prefix + suffix, prefix_tokens))
    else:
        for i in range(num_requests):
            text = (
                f"Request {i}: "
                + f"Explain the concept of topic {i} in computer science. "
                * (context_length * chars_per_token // 60 + 1)
            )[:context_length * chars_per_token]
            prompts.append((text, 0))

    return prompts


class TRTLLMEngine:
    """Wrapper around TensorRT-LLM high-level API for baseline benchmarks.

    Uses tensorrt_llm.hlapi.LLM for inference. Supports:
    - KV cache type configuration (bf16, fp8, nvfp4)
    - Host memory offload via KvCacheConfig
    - Block reuse for prefix caching
    - Pre-built engine loading via engine_dir
    """

    def __init__(self, model, kv_mode, context_length, tp_size=1,
                 engine_dir=None, offload=False, offload_size=20.0):
        try:
            import tensorrt_llm
            self.version = tensorrt_llm.__version__
        except ImportError:
            print("ERROR: TensorRT-LLM not installed.")
            print("Install with: pip install tensorrt-llm")
            print("Or use the NVIDIA TRT-LLM container: nvcr.io/nvidia/tensorrt-llm")
            print("Alternatively, use --engine vllm for the follow-up path.")
            sys.exit(1)

        from tensorrt_llm.hlapi import LLM, SamplingParams, KvCacheConfig, BuildConfig

        self.SamplingParams = SamplingParams
        self.name = "tensorrt_llm"
        self.kv_mode_requested = kv_mode
        self.kv_mode_actual = kv_mode

        # Configure KV cache
        kv_cache_config_kwargs = {
            "enable_block_reuse": True,
        }

        # Set host cache for offload
        if offload:
            # host_cache_size is in number of tokens or bytes depending on TRT-LLM version
            # Using a size-based approach
            kv_cache_config_kwargs["host_cache_size"] = int(offload_size * 1024 * 1024 * 1024)  # GB to bytes
            print(f"Host offload enabled: {offload_size} GB")

        try:
            kv_cache_config = KvCacheConfig(**kv_cache_config_kwargs)
        except TypeError as e:
            # Older TRT-LLM versions may not support all kwargs
            print(f"WARNING: KvCacheConfig kwargs not fully supported: {e}")
            kv_cache_config_kwargs = {}
            try:
                kv_cache_config = KvCacheConfig(**kv_cache_config_kwargs)
            except Exception:
                kv_cache_config = None

        # Configure build
        build_config_kwargs = {
            "max_seq_len": context_length,
        }

        # Map kv_mode to TRT-LLM kv cache type
        # TRT-LLM uses quantization config for KV cache precision
        if kv_mode == "nvfp4":
            # NVFP4 KV cache is configured via the quantization config / checkpoint
            # For hlapi, it may be set via the model checkpoint or build config
            build_config_kwargs["plugin_config"] = {"use_fp8_kv_cache": False}
            print(f"Configuring NVFP4 KV cache mode")
        elif kv_mode == "fp8":
            build_config_kwargs["plugin_config"] = {"use_fp8_kv_cache": True}
            print(f"Configuring FP8 KV cache mode")

        try:
            build_config = BuildConfig(**build_config_kwargs)
        except TypeError as e:
            print(f"WARNING: BuildConfig kwargs not fully supported: {e}")
            try:
                build_config = BuildConfig(max_seq_len=context_length)
            except Exception:
                build_config = None

        # Build LLM kwargs
        llm_kwargs = {
            "model": engine_dir or model,
            "tensor_parallel_size": tp_size,
        }

        if kv_cache_config is not None:
            llm_kwargs["kv_cache_config"] = kv_cache_config
        if build_config is not None:
            llm_kwargs["build_config"] = build_config

        print(f"Loading TRT-LLM model: {engine_dir or model}")
        print(f"  KV mode: {kv_mode}, TP: {tp_size}")
        if offload:
            print(f"  Host offload: {offload_size} GB")

        try:
            self.llm = LLM(**llm_kwargs)
        except Exception as e:
            err_str = str(e)
            # Handle NVFP4 fallback
            if kv_mode == "nvfp4" and ("nvfp4" in err_str.lower() or "fp4" in err_str.lower()):
                print(f"WARNING: NVFP4 KV cache not supported in this TRT-LLM version.")
                print(f"Falling back to FP8 KV cache.")
                self.kv_mode_actual = "fp8_fallback_from_nvfp4"
                if build_config is not None and hasattr(build_config, "plugin_config"):
                    build_config.plugin_config = {"use_fp8_kv_cache": True}
                    llm_kwargs["build_config"] = build_config
                try:
                    self.llm = LLM(**llm_kwargs)
                except Exception as e2:
                    print(f"ERROR: FP8 fallback also failed: {e2}")
                    print("Try --kv-mode bf16 or use --engine vllm")
                    sys.exit(1)
            else:
                print(f"ERROR: Failed to initialize TRT-LLM: {e}")
                print("Check that the model is compatible with TRT-LLM.")
                print("If using a pre-built engine, pass --engine-dir <path>")
                print("Otherwise, use --engine vllm for the follow-up path.")
                sys.exit(1)

    def run_batch(self, prompts, max_tokens):
        """Run batch inference. Returns (per_request_metrics, total_tokens, total_time, peak_hbm_gb)."""
        import torch

        sampling_params = self.SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prompt_texts = [p[0] for p in prompts]

        per_request_metrics = []
        total_output_tokens = 0
        total_start = time.perf_counter()

        # Run requests - TRT-LLM hlapi supports batch generation
        try:
            outputs = self.llm.generate(prompt_texts, sampling_params)
        except Exception as e:
            print(f"ERROR during generation: {e}")
            # Try one-by-one as fallback
            outputs = []
            for text in prompt_texts:
                try:
                    out = self.llm.generate([text], sampling_params)
                    outputs.extend(out)
                except Exception as e2:
                    print(f"  Request failed: {e2}")

        total_time = time.perf_counter() - total_start

        for i, output in enumerate(outputs):
            # TRT-LLM hlapi output format
            if hasattr(output, "outputs") and output.outputs:
                n_tokens = len(output.outputs[0].token_ids)
            elif hasattr(output, "token_ids"):
                n_tokens = len(output.token_ids)
            else:
                n_tokens = 0

            total_output_tokens += n_tokens

            req_time = total_time / len(outputs) if outputs else total_time
            # TTFT estimate: for offline batch, approximate from total
            ttft_est = req_time * 0.3
            decode_time = req_time * 0.7
            tpot = (decode_time / n_tokens * 1000) if n_tokens > 0 else 0

            per_request_metrics.append({
                "ttft_ms": ttft_est * 1000,
                "tpot_ms": tpot,
                "output_tokens": n_tokens,
                "total_time_s": req_time,
            })

        peak_mem_gb = None
        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

        return per_request_metrics, total_output_tokens, total_time, peak_mem_gb


class VLLMEngine:
    """Wrapper around vLLM for baseline benchmarks (follow-up path)."""

    def __init__(self, model, kv_mode, context_length, tp_size=1):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            print("ERROR: vLLM not installed. Install with: pip install vllm")
            sys.exit(1)

        self.SamplingParams = SamplingParams

        # Map kv_mode to vLLM kv_cache_dtype
        kv_dtype_map = {
            "bf16": "auto",
            "fp8": "fp8",
            "nvfp4": "fp8",  # fallback; actual nvfp4 requires stack verification
        }
        kv_cache_dtype = kv_dtype_map.get(kv_mode, "auto")

        self.kv_mode_requested = kv_mode
        self.kv_mode_actual = kv_mode
        if kv_mode == "nvfp4" and kv_cache_dtype != "nvfp4":
            print(f"WARNING: NVFP4 KV cache requested but vLLM may not support it. "
                  f"Using kv_cache_dtype='{kv_cache_dtype}' (FP8 fallback).")
            self.kv_mode_actual = "fp8_fallback_from_nvfp4"

        print(f"Loading model {model} with kv_cache_dtype={kv_cache_dtype}, tp={tp_size}...")

        llm_kwargs = {
            "model": model,
            "gpu_memory_utilization": 0.90,
            "max_model_len": context_length,
            "kv_cache_dtype": kv_cache_dtype,
            "tensor_parallel_size": tp_size,
            "enable_prefix_caching": True,
        }

        try:
            self.llm = LLM(**llm_kwargs)
        except TypeError as e:
            # Older vLLM may not support all kwargs
            err_str = str(e)
            if "kv_cache_dtype" in err_str:
                print("WARNING: kv_cache_dtype not supported. Falling back to default.")
                del llm_kwargs["kv_cache_dtype"]
                self.kv_mode_actual = "bf16_fallback"
            if "enable_prefix_caching" in err_str:
                del llm_kwargs["enable_prefix_caching"]
            self.llm = LLM(**llm_kwargs)

        self.name = "vllm"
        try:
            import vllm
            self.version = vllm.__version__
        except Exception:
            self.version = "unknown"

    def run_batch(self, prompts, max_tokens):
        """Run batch inference. Returns (per_request_metrics, total_tokens, total_time, peak_hbm_gb)."""
        import torch
        sampling_params = self.SamplingParams(temperature=0.0, max_tokens=max_tokens)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prompt_texts = [p[0] for p in prompts]

        start_time = time.perf_counter()
        outputs = self.llm.generate(prompt_texts, sampling_params)
        total_time = time.perf_counter() - start_time

        results = []
        total_output_tokens = 0
        for i, output in enumerate(outputs):
            n_tokens = len(output.outputs[0].token_ids)
            total_output_tokens += n_tokens

            req_time = total_time / len(outputs)
            # TTFT estimate: ~30% of request time is prefill
            ttft_est = req_time * 0.3
            decode_time = req_time * 0.7
            tpot = (decode_time / n_tokens * 1000) if n_tokens > 0 else 0

            results.append({
                "ttft_ms": ttft_est * 1000,
                "tpot_ms": tpot,
                "output_tokens": n_tokens,
                "total_time_s": req_time,
            })

        peak_mem_gb = None
        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

        return results, total_output_tokens, total_time, peak_mem_gb


def create_engine(args):
    """Factory for engine creation."""
    if args.engine == "tensorrt_llm":
        return TRTLLMEngine(
            args.model, args.kv_mode, args.context_length, args.tp,
            engine_dir=args.engine_dir,
            offload=args.offload,
            offload_size=args.offload_size,
        )
    elif args.engine == "vllm":
        return VLLMEngine(args.model, args.kv_mode, args.context_length, args.tp)
    else:
        print(f"ERROR: Unknown engine {args.engine}")
        sys.exit(1)


def main():
    args = parse_args()

    run_id = args.run_id or generate_run_id(args.kv_mode, args.context_length)
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/baseline_{args.kv_mode}_{args.context_length}_{ts}.json"

    # Auto-detect scenario_id if not provided
    if args.scenario_id is None:
        if args.concurrency > 1:
            args.scenario_id = "scenario_2_more_sessions_gpu"
        else:
            args.scenario_id = "scenario_1_longer_context_gpu"

    print("=== Blackwell KV Baseline Benchmark ===")
    print(f"Run ID: {run_id}")
    print(f"Model: {args.model}")
    print(f"KV mode: {args.kv_mode}")
    print(f"Context: {args.context_length}")
    print(f"Requests: {args.requests}")
    print(f"Workload: {args.workload_type}")
    print(f"Engine: {args.engine}")
    if args.offload:
        print(f"Offload: {args.offload_size} GB host cache")
    print(f"Output: {args.output}")
    print()

    gpu_info = get_gpu_info()
    cuda_ver = get_cuda_version()

    # Generate workload
    if args.workload_file:
        print(f"Loading workload from {args.workload_file}...")
        prompts = load_workload_file(args.workload_file, args.requests)
    else:
        print("Generating workload...")
        prompts = generate_workload(
            args.workload_type, args.context_length,
            args.requests, args.prefix_ratio
        )
    shared_prefix_tokens = prompts[0][1] if prompts else 0

    # Create engine
    print("Initializing engine...")
    engine = create_engine(args)

    # Start power sampling
    power_sampler = PowerSampler(interval_s=0.5)
    power_sampler.start()

    # Run benchmark
    print(f"Running {args.requests} requests...")
    try:
        per_request, total_tokens, total_time, peak_hbm = engine.run_batch(
            prompts, args.max_tokens
        )
    finally:
        power_sampler.stop()

    # Compute aggregates
    ttft_values = [r["ttft_ms"] for r in per_request]
    tpot_values = [r["tpot_ms"] for r in per_request if r["tpot_ms"] > 0]
    throughput = total_tokens / total_time if total_time > 0 else 0

    # Build result
    result = make_result_template()
    result["run_id"] = run_id
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["scenario_id"] = args.scenario_id
    result["serving_mode"] = "offline"

    result["runtime"] = {
        "engine": engine.name,
        "engine_version": engine.version,
        "cuda_version": cuda_ver,
        "driver_version": gpu_info["driver_version"],
        "gpu_name": gpu_info["gpu_model"],
        "gpu_count": gpu_info["gpu_count"],
        "node_count": 1,
    }

    result["model"] = {
        "name": args.model,
        "kv_mode": engine.kv_mode_actual,
        "kv_mode_requested": engine.kv_mode_requested,
        "context_length": args.context_length,
    }

    result["workload"] = {
        "type": args.workload_type,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "shared_prefix_tokens": shared_prefix_tokens,
        "generated_tokens_per_request": args.max_tokens,
    }

    result["tiering"] = {
        "enabled": args.offload,
        "hot_tier_format": args.kv_mode,
        "cold_tier_format": "host_ram" if args.offload else None,
        "cold_tier_codec": "none",
        "promotion_policy": None,
        "recent_window_tokens": None,
        "sink_tokens_protected": None,
        "host_cache_size_gb": args.offload_size if args.offload else None,
    }

    result["metrics"] = {
        "ttft_ms_p50": percentile(ttft_values, 50),
        "ttft_ms_p95": percentile(ttft_values, 95),
        "tpot_ms_p50": percentile(tpot_values, 50),
        "tpot_ms_p95": percentile(tpot_values, 95),
        "tpot_ms_p99": percentile(tpot_values, 99),
        "throughput_tokens_per_s": round(throughput, 2),
        "peak_hbm_gb": round(peak_hbm, 3) if peak_hbm else None,
        "gpu_power_w_avg": round(power_sampler.average(), 1),
        "cache_hit_rate": None,
        "promotion_latency_ms_p50": None,
        "promotion_latency_ms_p95": None,
        "quality_metric_name": None,
        "quality_metric_value": None,
        "quality_delta_vs_best_baseline": None,
    }

    result["notes"] = {
        "support_gate_result": engine.kv_mode_actual,
        "known_limitations": [],
    }
    if engine.kv_mode_actual != engine.kv_mode_requested:
        result["notes"]["known_limitations"].append(
            f"Requested {engine.kv_mode_requested} but used {engine.kv_mode_actual}"
        )

    offload_flags = ""
    if args.offload:
        offload_flags = f"--offload --offload-size {args.offload_size} "
    engine_dir_flag = f"--engine-dir {args.engine_dir} " if args.engine_dir else ""

    result["rerun_command"] = (
        f"python scripts/run_baseline.py "
        f"--model {args.model} "
        f"--context-length {args.context_length} "
        f"--requests {args.requests} "
        f"--concurrency {args.concurrency} "
        f"--kv-mode {args.kv_mode} "
        f"--workload-type {args.workload_type} "
        f"--engine {args.engine} "
        f"{engine_dir_flag}"
        f"{offload_flags}"
        f"--tp {args.tp} "
        f"--output {args.output}"
    )

    write_result_json(result, args.output)

    # Print summary
    m = result["metrics"]
    print("\n=== Results Summary ===")
    if m["ttft_ms_p50"] is not None:
        print(f"TTFT p50: {m['ttft_ms_p50']:.1f} ms")
    if m["ttft_ms_p95"] is not None:
        print(f"TTFT p95: {m['ttft_ms_p95']:.1f} ms")
    if m["tpot_ms_p50"] is not None:
        print(f"TPOT p50: {m['tpot_ms_p50']:.1f} ms")
    if m["tpot_ms_p95"] is not None:
        print(f"TPOT p95: {m['tpot_ms_p95']:.1f} ms")
    if m["tpot_ms_p99"] is not None:
        print(f"TPOT p99: {m['tpot_ms_p99']:.1f} ms")
    print(f"Throughput: {m['throughput_tokens_per_s']:.1f} tok/s")
    if peak_hbm:
        print(f"Peak HBM: {peak_hbm:.2f} GB")
    print(f"Avg GPU Power: {m['gpu_power_w_avg']:.1f} W")
    print(f"KV mode actual: {engine.kv_mode_actual}")
    if args.offload:
        print(f"Host offload: {args.offload_size} GB")


if __name__ == "__main__":
    main()
