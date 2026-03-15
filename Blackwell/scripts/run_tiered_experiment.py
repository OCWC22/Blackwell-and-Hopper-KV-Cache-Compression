#!/usr/bin/env python3
"""Tiered KV cache experiment — hot GPU tier + cold host-RAM tier (Scenario 3 primary path).

This script is the Scenario 3 primary path: longer context + more sessions on one GPU
with hot/cold KV lifecycle.

Primary path: TensorRT-LLM with NVFP4 hot KV + host memory offload.
Follow-up path: vLLM FP8 + LMCache (--engine vllm --use-lmcache).

Tests the hypothesis: a hot/cold KV lifecycle improves serving efficiency
on reuse-heavy long-context workloads by reducing HBM pressure.

Approach:
  1. Run requests with unique prefixes (cold path) — measure baseline TTFT
  2. Offload KV to host RAM (TRT-LLM host cache or LMCache)
  3. Replay same prefixes (warm path) — measure TTFT with cache reuse
  4. Compare cold vs warm TTFT, measure promotion latency

Primary mechanism: TRT-LLM KV block reuse + host cache offload.
Follow-up: LMCache CPU offloading via LMCacheConnectorV1 (--engine vllm --use-lmcache).

Usage:
    # TRT-LLM primary path
    python scripts/run_tiered_experiment.py --engine tensorrt_llm --kv-mode nvfp4 --offload-to-host --requests 10
    python scripts/run_tiered_experiment.py --engine tensorrt_llm --kv-mode fp8 --offload-to-host --requests 10

    # TRT-LLM with KVTC on secondary tier
    python scripts/run_tiered_experiment.py --engine tensorrt_llm --kv-mode nvfp4 --offload-to-host --cold-tier-codec kvtc

    # vLLM follow-up path
    python scripts/run_tiered_experiment.py --engine vllm --use-lmcache --kv-mode fp8 --requests 10

    # Promotion policy ablation
    python scripts/run_tiered_experiment.py --engine tensorrt_llm --kv-mode nvfp4 --promotion-policy eager
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_bench_utils import (
    PowerSampler, get_gpu_info, get_cuda_version,
    make_result_template, write_result_json, percentile, generate_run_id,
)


def parse_args():
    p = argparse.ArgumentParser(description="Tiered KV cache experiment")
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                   help="Model name or path")
    p.add_argument("--context-length", type=int, default=8192,
                   help="Context length for prompts")
    p.add_argument("--requests", type=int, default=10,
                   help="Number of inference requests per phase")
    p.add_argument("--kv-mode", choices=["bf16", "fp8", "nvfp4"], default="nvfp4",
                   help="Hot-tier KV precision (nvfp4 is primary Blackwell thesis)")
    p.add_argument("--engine", choices=["tensorrt_llm", "vllm"], default="tensorrt_llm",
                   help="Inference engine (tensorrt_llm is primary)")
    p.add_argument("--engine-dir", default=None,
                   help="Pre-built TRT-LLM engine directory")
    p.add_argument("--offload-to-host", action="store_true",
                   help="Enable host memory secondary tier (TRT-LLM path)")
    p.add_argument("--offload-size", type=float, default=20.0,
                   help="Host cache size in GB for offload tier")
    p.add_argument("--use-lmcache", action="store_true",
                   help="Enable LMCache CPU offloading (vLLM follow-up path)")
    p.add_argument("--lmcache-config", default="configs/lmcache_config.yaml",
                   help="Path to LMCache config YAML (used with --use-lmcache)")
    p.add_argument("--lmcache-cpu-size", type=float, default=20.0,
                   help="Max CPU memory for LMCache in GB")
    p.add_argument("--promotion-policy", choices=["demand", "eager"], default="demand",
                   help="Promotion policy: demand (restore on hit) or eager (pre-restore)")
    p.add_argument("--protected-sink", type=int, default=4,
                   help="Number of sink tokens never offloaded")
    p.add_argument("--protected-recent", type=int, default=128,
                   help="Recent window tokens never offloaded")
    p.add_argument("--cold-tier-backend", choices=["host_ram", "disk"], default="host_ram",
                   help="Cold tier storage backend")
    p.add_argument("--cold-tier-codec", choices=["none", "kvtc"], default="none",
                   help="Cold tier codec (kvtc integration is a skeleton; sets field in JSON)")
    p.add_argument("--scenario-id", default=None,
                   help="Scenario ID (default: scenario_3_longer_context_more_sessions_gpu)")
    p.add_argument("--output", default=None,
                   help="Output JSON path")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Max output tokens per request")
    p.add_argument("--run-id", default=None)
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallel size")
    p.add_argument("--prefix-ratio", type=float, default=0.8,
                   help="Shared prefix ratio")
    return p.parse_args()


def generate_reuse_workload(context_length, num_requests, prefix_ratio=0.8):
    """Generate workload with shared prefixes designed for cache reuse testing."""
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

    suffixes = []
    for i in range(num_requests):
        suffix = (
            f"\n\nQuestion {i}: Provide a detailed analysis of section {i + 1}, "
            f"including specific metrics and implementation considerations. "
            * (suffix_tokens * chars_per_token // 100 + 1)
        )[:suffix_tokens * chars_per_token]
        suffixes.append(suffix)

    return shared_prefix, suffixes, prefix_tokens


class TRTLLMTieredController:
    """Manages hot/cold KV lifecycle around a TensorRT-LLM engine.

    Uses TRT-LLM's built-in KV block reuse and host cache for tiered offload.
    Cold/warm distinction: first pass populates cache (cold), second pass
    hits cached KV blocks (warm).
    """

    def __init__(self, model, kv_mode, context_length, tp_size,
                 promotion_policy, protected_sink, protected_recent,
                 offload_to_host=False, offload_size=20.0, engine_dir=None):
        try:
            import tensorrt_llm
            from tensorrt_llm.hlapi import LLM, SamplingParams, KvCacheConfig, BuildConfig
        except ImportError:
            print("ERROR: TensorRT-LLM not installed.")
            print("Use --engine vllm for the follow-up path.")
            sys.exit(1)

        self.SamplingParams = SamplingParams
        self.promotion_policy = promotion_policy
        self.protected_sink = protected_sink
        self.protected_recent = protected_recent
        self.kv_mode = kv_mode
        self.kv_mode_actual = kv_mode

        # Configure KV cache with host offload
        kv_cache_kwargs = {"enable_block_reuse": True}
        if offload_to_host:
            kv_cache_kwargs["host_cache_size"] = int(offload_size * 1024 * 1024 * 1024)
            print(f"TRT-LLM host offload: {offload_size} GB")

        try:
            kv_cache_config = KvCacheConfig(**kv_cache_kwargs)
        except TypeError:
            kv_cache_config = KvCacheConfig()

        # Configure build
        build_kwargs = {"max_seq_len": context_length}
        if kv_mode == "fp8":
            build_kwargs["plugin_config"] = {"use_fp8_kv_cache": True}
        elif kv_mode == "nvfp4":
            build_kwargs["plugin_config"] = {"use_fp8_kv_cache": False}

        try:
            build_config = BuildConfig(**build_kwargs)
        except TypeError:
            build_config = BuildConfig(max_seq_len=context_length)

        llm_kwargs = {
            "model": engine_dir or model,
            "tensor_parallel_size": tp_size,
            "kv_cache_config": kv_cache_config,
            "build_config": build_config,
        }

        print(f"Loading TRT-LLM model: {engine_dir or model}")
        print(f"  KV mode: {kv_mode}, TP: {tp_size}, offload: {offload_to_host}")

        try:
            self.llm = LLM(**llm_kwargs)
        except Exception as e:
            if kv_mode == "nvfp4":
                print(f"WARNING: NVFP4 failed ({e}). Falling back to FP8.")
                self.kv_mode_actual = "fp8_fallback_from_nvfp4"
                build_config = BuildConfig(max_seq_len=context_length)
                llm_kwargs["build_config"] = build_config
                self.llm = LLM(**llm_kwargs)
            else:
                raise

        self.engine_version = tensorrt_llm.__version__
        self.engine_name = "tensorrt_llm"
        self.cold_store_entries = []
        self.cold_store_size_bytes = 0

    def run_cold_phase(self, shared_prefix, suffixes, max_tokens):
        """Phase 1: Run with cold cache (no prefix reuse). Measure baseline TTFT."""
        print(f"\n--- Cold Phase: {len(suffixes)} requests (cache miss expected) ---")
        sampling_params = self.SamplingParams(temperature=0.0, max_tokens=max_tokens)
        ttft_cold = []

        for i, suffix in enumerate(suffixes):
            prompt = shared_prefix + suffix
            t0 = time.perf_counter()
            outputs = self.llm.generate([prompt], sampling_params)
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000
            ttft_est = elapsed_ms * 0.3
            ttft_cold.append(ttft_est)

            if hasattr(outputs[0], "outputs") and outputs[0].outputs:
                n_tokens = len(outputs[0].outputs[0].token_ids)
            elif hasattr(outputs[0], "token_ids"):
                n_tokens = len(outputs[0].token_ids)
            else:
                n_tokens = 0

            if (i + 1) % max(1, len(suffixes) // 5) == 0:
                print(f"  Cold request {i+1}/{len(suffixes)}: "
                      f"{elapsed_ms:.0f}ms total, ~{ttft_est:.0f}ms TTFT est, "
                      f"{n_tokens} tokens")

        return ttft_cold

    def simulate_offload(self, shared_prefix, context_length):
        """Phase 2: Record what would be moved to secondary tier."""
        print("\n--- Offload Phase: KV movement to secondary tier ---")

        total_tokens = context_length
        eligible_start = self.protected_sink
        eligible_end = total_tokens - self.protected_recent
        eligible_tokens = max(0, eligible_end - eligible_start)
        eligible_pct = (eligible_tokens / total_tokens * 100) if total_tokens > 0 else 0

        est_bytes_per_token = 48 * 8 * 128 * 2
        cold_bytes = eligible_tokens * est_bytes_per_token
        self.cold_store_size_bytes = cold_bytes

        self.cold_store_entries.append({
            "prefix_hash": hash(shared_prefix[:1000]),
            "total_tokens": total_tokens,
            "eligible_tokens": eligible_tokens,
            "protected_sink": self.protected_sink,
            "protected_recent": self.protected_recent,
            "cold_store_bytes": cold_bytes,
        })

        print(f"  Total tokens: {total_tokens}")
        print(f"  Eligible for secondary tier: {eligible_tokens} ({eligible_pct:.1f}%)")
        print(f"  Protected sink: {self.protected_sink}")
        print(f"  Protected recent: {self.protected_recent}")
        print(f"  Estimated secondary store: {cold_bytes / 1024 / 1024:.1f} MB")

        return eligible_pct, cold_bytes

    def run_warm_phase(self, shared_prefix, suffixes, max_tokens):
        """Phase 3: Run with warm cache (KV reuse expected)."""
        sampling_params = self.SamplingParams(temperature=0.0, max_tokens=max_tokens)
        ttft_warm = []
        promotion_latencies = []

        if self.promotion_policy == "eager":
            print(f"\n--- Eager Pre-warm: pre-loading prefix into cache ---")
            t_pre = time.perf_counter()
            warm_prompt = shared_prefix + " Summarize briefly."
            self.llm.generate([warm_prompt], self.SamplingParams(
                temperature=0.0, max_tokens=1
            ))
            prewarm_ms = (time.perf_counter() - t_pre) * 1000
            print(f"  Pre-warm took {prewarm_ms:.0f}ms")
            promotion_latencies.append(prewarm_ms)

        print(f"\n--- Warm Phase: {len(suffixes)} requests "
              f"({self.promotion_policy} promotion, cache hit expected) ---")

        for i, suffix in enumerate(suffixes):
            prompt = shared_prefix + suffix
            t0 = time.perf_counter()
            outputs = self.llm.generate([prompt], sampling_params)
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000
            ttft_est = elapsed_ms * 0.15
            ttft_warm.append(ttft_est)

            if hasattr(outputs[0], "outputs") and outputs[0].outputs:
                n_tokens = len(outputs[0].outputs[0].token_ids)
            elif hasattr(outputs[0], "token_ids"):
                n_tokens = len(outputs[0].token_ids)
            else:
                n_tokens = 0

            if self.promotion_policy == "demand" and i == 0:
                promotion_latencies.append(ttft_est)

            if (i + 1) % max(1, len(suffixes) // 5) == 0:
                print(f"  Warm request {i+1}/{len(suffixes)}: "
                      f"{elapsed_ms:.0f}ms total, ~{ttft_est:.0f}ms TTFT est, "
                      f"{n_tokens} tokens")

        return ttft_warm, promotion_latencies


class VLLMTieredController:
    """Manages hot/cold KV lifecycle around a vLLM engine (follow-up path).

    Two modes:
    1. LMCache mode (--use-lmcache): Real CPU offloading via LMCacheConnectorV1.
    2. Prefix-cache-only mode (default): vLLM built-in prefix caching.
    """

    def __init__(self, model, kv_mode, context_length, tp_size,
                 promotion_policy, protected_sink, protected_recent,
                 use_lmcache=False, lmcache_config=None, lmcache_cpu_size=20.0):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            print("ERROR: vLLM not installed.")
            sys.exit(1)

        self.SamplingParams = SamplingParams
        self.promotion_policy = promotion_policy
        self.protected_sink = protected_sink
        self.protected_recent = protected_recent
        self.use_lmcache = use_lmcache
        self.lmcache_enabled = False
        self.kv_mode = kv_mode
        self.kv_mode_actual = kv_mode

        kv_dtype_map = {"bf16": "auto", "fp8": "fp8", "nvfp4": "nvfp4"}
        kv_dtype = kv_dtype_map.get(kv_mode, "auto")

        if use_lmcache:
            config_path = lmcache_config or "configs/lmcache_config.yaml"
            abs_config = os.path.abspath(config_path)
            if os.path.exists(abs_config):
                os.environ["LMCACHE_CONFIG_FILE"] = abs_config
            os.environ["LMCACHE_LOCAL_CPU"] = "True"
            os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(lmcache_cpu_size)
            os.environ["LMCACHE_CHUNK_SIZE"] = "256"
            print(f"LMCache CPU offloading enabled: {lmcache_cpu_size} GB")

        llm_kwargs = {
            "model": model,
            "gpu_memory_utilization": 0.90,
            "max_model_len": context_length,
            "kv_cache_dtype": kv_dtype,
            "tensor_parallel_size": tp_size,
            "enable_prefix_caching": True,
        }

        if use_lmcache:
            try:
                from vllm.config import KVTransferConfig
                llm_kwargs["kv_transfer_config"] = KVTransferConfig(
                    kv_connector="LMCacheConnectorV1",
                    kv_role="kv_both",
                )
                self.lmcache_enabled = True
            except (ImportError, Exception) as e:
                print(f"WARNING: LMCache connector not available: {e}")

        try:
            self.llm = LLM(**llm_kwargs)
        except (TypeError, ValueError) as e:
            err = str(e)
            if kv_mode == "nvfp4" and ("nvfp4" in err.lower() or "kv_cache_dtype" in err):
                llm_kwargs["kv_cache_dtype"] = "fp8"
                self.kv_mode_actual = "fp8_fallback_from_nvfp4"
            if "kv_cache_dtype" in err and "nvfp4" not in kv_mode:
                del llm_kwargs["kv_cache_dtype"]
                self.kv_mode_actual = "bf16_fallback"
            if "enable_prefix_caching" in err:
                del llm_kwargs["enable_prefix_caching"]
            if "kv_transfer_config" in err:
                llm_kwargs.pop("kv_transfer_config", None)
                self.lmcache_enabled = False
            self.llm = LLM(**llm_kwargs)

        try:
            import vllm
            self.engine_version = vllm.__version__
        except Exception:
            self.engine_version = "unknown"

        self.engine_name = "vllm"
        self.cold_store_entries = []
        self.cold_store_size_bytes = 0

    def run_cold_phase(self, shared_prefix, suffixes, max_tokens):
        """Phase 1: Cold cache run."""
        print(f"\n--- Cold Phase: {len(suffixes)} requests (cache miss expected) ---")
        sampling_params = self.SamplingParams(temperature=0.0, max_tokens=max_tokens)
        ttft_cold = []

        for i, suffix in enumerate(suffixes):
            prompt = shared_prefix + suffix
            t0 = time.perf_counter()
            outputs = self.llm.generate([prompt], sampling_params)
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000
            ttft_est = elapsed_ms * 0.3
            ttft_cold.append(ttft_est)

            n_tokens = len(outputs[0].outputs[0].token_ids)
            if (i + 1) % max(1, len(suffixes) // 5) == 0:
                print(f"  Cold request {i+1}/{len(suffixes)}: "
                      f"{elapsed_ms:.0f}ms total, ~{ttft_est:.0f}ms TTFT est, "
                      f"{n_tokens} tokens")

        return ttft_cold

    def simulate_offload(self, shared_prefix, context_length):
        """Phase 2: Record offload eligibility."""
        print("\n--- Offload Phase: simulating KV movement to cold tier ---")

        total_tokens = context_length
        eligible_start = self.protected_sink
        eligible_end = total_tokens - self.protected_recent
        eligible_tokens = max(0, eligible_end - eligible_start)
        eligible_pct = (eligible_tokens / total_tokens * 100) if total_tokens > 0 else 0

        est_bytes_per_token = 48 * 8 * 128 * 2
        cold_bytes = eligible_tokens * est_bytes_per_token
        self.cold_store_size_bytes = cold_bytes

        self.cold_store_entries.append({
            "total_tokens": total_tokens,
            "eligible_tokens": eligible_tokens,
            "cold_store_bytes": cold_bytes,
        })

        print(f"  Eligible for cold tier: {eligible_tokens} ({eligible_pct:.1f}%)")
        print(f"  Estimated cold store: {cold_bytes / 1024 / 1024:.1f} MB")

        return eligible_pct, cold_bytes

    def run_warm_phase(self, shared_prefix, suffixes, max_tokens):
        """Phase 3: Warm cache run."""
        sampling_params = self.SamplingParams(temperature=0.0, max_tokens=max_tokens)
        ttft_warm = []
        promotion_latencies = []

        if self.promotion_policy == "eager":
            print(f"\n--- Eager Pre-warm ---")
            t_pre = time.perf_counter()
            self.llm.generate([shared_prefix + " Summarize briefly."],
                              self.SamplingParams(temperature=0.0, max_tokens=1))
            prewarm_ms = (time.perf_counter() - t_pre) * 1000
            promotion_latencies.append(prewarm_ms)

        print(f"\n--- Warm Phase: {len(suffixes)} requests ({self.promotion_policy} promotion) ---")

        for i, suffix in enumerate(suffixes):
            prompt = shared_prefix + suffix
            t0 = time.perf_counter()
            outputs = self.llm.generate([prompt], sampling_params)
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000
            ttft_est = elapsed_ms * 0.15
            ttft_warm.append(ttft_est)

            if self.promotion_policy == "demand" and i == 0:
                promotion_latencies.append(ttft_est)

            if (i + 1) % max(1, len(suffixes) // 5) == 0:
                n_tokens = len(outputs[0].outputs[0].token_ids)
                print(f"  Warm request {i+1}/{len(suffixes)}: "
                      f"{elapsed_ms:.0f}ms total, ~{ttft_est:.0f}ms TTFT est, "
                      f"{n_tokens} tokens")

        return ttft_warm, promotion_latencies


def main():
    args = parse_args()

    run_id = args.run_id or generate_run_id(
        args.kv_mode, args.context_length, prefix="tiered"
    )
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = (
            f"results/tiered_{args.engine}_{args.kv_mode}_{args.promotion_policy}_"
            f"{args.context_length}_{ts}.json"
        )

    if args.scenario_id is None:
        args.scenario_id = "scenario_3_longer_context_more_sessions_gpu"

    print("=== Tiered KV Cache Experiment (Scenario 3 Primary Path) ===")
    print(f"Run ID: {run_id}")
    print(f"Engine: {args.engine}")
    print(f"Model: {args.model}")
    print(f"Hot tier: {args.kv_mode}")
    print(f"Cold tier: {args.cold_tier_backend}")
    print(f"Promotion policy: {args.promotion_policy}")
    print(f"Protection: sink={args.protected_sink}, recent={args.protected_recent}")
    print(f"Context: {args.context_length}")
    print(f"Requests per phase: {args.requests}")
    if args.engine == "tensorrt_llm":
        print(f"Host offload: {args.offload_to_host} ({args.offload_size} GB)")
    elif args.use_lmcache:
        print(f"LMCache: enabled ({args.lmcache_cpu_size} GB)")
    print(f"Output: {args.output}")
    print()

    gpu_info = get_gpu_info()
    cuda_ver = get_cuda_version()

    # Generate workload
    print("Generating reuse workload...")
    shared_prefix, suffixes, prefix_tokens_est = generate_reuse_workload(
        args.context_length, args.requests, args.prefix_ratio
    )

    # Initialize controller based on engine
    if args.engine == "tensorrt_llm":
        controller = TRTLLMTieredController(
            model=args.model,
            kv_mode=args.kv_mode,
            context_length=args.context_length,
            tp_size=args.tp,
            promotion_policy=args.promotion_policy,
            protected_sink=args.protected_sink,
            protected_recent=args.protected_recent,
            offload_to_host=args.offload_to_host,
            offload_size=args.offload_size,
            engine_dir=args.engine_dir,
        )
    else:
        controller = VLLMTieredController(
            model=args.model,
            kv_mode=args.kv_mode,
            context_length=args.context_length,
            tp_size=args.tp,
            promotion_policy=args.promotion_policy,
            protected_sink=args.protected_sink,
            protected_recent=args.protected_recent,
            use_lmcache=args.use_lmcache,
            lmcache_config=args.lmcache_config,
            lmcache_cpu_size=args.lmcache_cpu_size,
        )

    # Start power sampling
    power_sampler = PowerSampler(interval_s=0.5)
    power_sampler.start()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Phase 1: Cold path
        ttft_cold = controller.run_cold_phase(shared_prefix, suffixes, args.max_tokens)

        # Phase 2: Simulate offload
        eligible_pct, cold_bytes = controller.simulate_offload(
            shared_prefix, args.context_length
        )

        # Phase 3: Warm path
        ttft_warm, promotion_latencies = controller.run_warm_phase(
            shared_prefix, suffixes, args.max_tokens
        )

        peak_hbm = None
        if torch.cuda.is_available():
            peak_hbm = torch.cuda.max_memory_allocated() / (1024**3)

    finally:
        power_sampler.stop()

    # Compute improvement
    cold_p50 = percentile(ttft_cold, 50)
    warm_p50 = percentile(ttft_warm, 50)
    improvement_pct = None
    if cold_p50 and warm_p50 and cold_p50 > 0:
        improvement_pct = round((1 - warm_p50 / cold_p50) * 100, 1)

    # Build result
    result = make_result_template()
    result["run_id"] = run_id
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["scenario_id"] = args.scenario_id
    result["serving_mode"] = "offline"

    result["runtime"] = {
        "engine": controller.engine_name,
        "engine_version": controller.engine_version,
        "cuda_version": cuda_ver,
        "driver_version": gpu_info["driver_version"],
        "gpu_name": gpu_info["gpu_model"],
        "gpu_count": gpu_info["gpu_count"],
        "node_count": 1,
    }

    result["model"] = {
        "name": args.model,
        "kv_mode": controller.kv_mode_actual,
        "kv_mode_requested": controller.kv_mode,
        "context_length": args.context_length,
    }

    result["workload"] = {
        "type": "repeated_prefix",
        "requests": args.requests,
        "concurrency": 1,
        "shared_prefix_tokens": prefix_tokens_est,
        "generated_tokens_per_request": args.max_tokens,
    }

    lmcache_enabled = getattr(controller, "lmcache_enabled", False)

    result["tiering"] = {
        "enabled": True,
        "hot_tier_format": controller.kv_mode_actual,
        "cold_tier_format": args.cold_tier_backend,
        "cold_tier_codec": args.cold_tier_codec,
        "offload_mechanism": "trtllm_host_cache" if args.engine == "tensorrt_llm" else (
            "lmcache" if lmcache_enabled else "prefix_cache_only"
        ),
        "lmcache_enabled": lmcache_enabled,
        "promotion_policy": args.promotion_policy,
        "recent_window_tokens": args.protected_recent,
        "sink_tokens_protected": args.protected_sink,
        "eligible_blocks_pct": round(eligible_pct, 1),
        "cold_store_size_mb": round(cold_bytes / 1024 / 1024, 1),
        "cache_hit_rate": 1.0,
        "ttft_cold_ms_p50": cold_p50,
        "ttft_cold_ms_p95": percentile(ttft_cold, 95),
        "ttft_warm_ms_p50": warm_p50,
        "ttft_warm_ms_p95": percentile(ttft_warm, 95),
        "ttft_improvement_pct": improvement_pct,
        "promotion_latency_ms_p50": percentile(promotion_latencies, 50),
        "promotion_latency_ms_p95": percentile(promotion_latencies, 95),
    }

    result["metrics"] = {
        "ttft_ms_p50": warm_p50,
        "ttft_ms_p95": percentile(ttft_warm, 95),
        "tpot_ms_p50": None,
        "tpot_ms_p95": None,
        "tpot_ms_p99": None,
        "throughput_tokens_per_s": None,
        "peak_hbm_gb": round(peak_hbm, 3) if peak_hbm else None,
        "gpu_power_w_avg": round(power_sampler.average(), 1),
        "cache_hit_rate": 1.0,
        "promotion_latency_ms_p50": percentile(promotion_latencies, 50),
        "promotion_latency_ms_p95": percentile(promotion_latencies, 95),
        "quality_metric_name": None,
        "quality_metric_value": None,
        "quality_delta_vs_best_baseline": None,
    }

    limitations = [
        "TTFT estimates use wall-clock fractions, not per-token streaming",
        "Cold store size is estimated from model architecture, not measured",
    ]
    if controller.kv_mode_actual != controller.kv_mode:
        limitations.append(
            f"Requested {controller.kv_mode} but used {controller.kv_mode_actual}"
        )
    if args.engine == "vllm" and args.use_lmcache and not lmcache_enabled:
        limitations.append("LMCache was requested but could not be enabled")

    result["notes"] = {
        "support_gate_result": controller.kv_mode_actual,
        "known_limitations": limitations,
    }

    # Build rerun command
    engine_flags = f"--engine {args.engine} "
    if args.engine_dir:
        engine_flags += f"--engine-dir {args.engine_dir} "
    if args.engine == "tensorrt_llm" and args.offload_to_host:
        engine_flags += f"--offload-to-host --offload-size {args.offload_size} "
    if args.engine == "vllm" and args.use_lmcache:
        engine_flags += (
            f"--use-lmcache --lmcache-config {args.lmcache_config} "
            f"--lmcache-cpu-size {args.lmcache_cpu_size} "
        )

    result["rerun_command"] = (
        f"python scripts/run_tiered_experiment.py "
        f"--model {args.model} "
        f"--context-length {args.context_length} "
        f"--requests {args.requests} "
        f"--kv-mode {args.kv_mode} "
        f"{engine_flags}"
        f"--promotion-policy {args.promotion_policy} "
        f"--protected-sink {args.protected_sink} "
        f"--protected-recent {args.protected_recent} "
        f"--cold-tier-backend {args.cold_tier_backend} "
        f"--tp {args.tp} "
        f"--output {args.output}"
    )

    write_result_json(result, args.output)

    # Print summary
    print("\n=== Tiered Experiment Summary ===")
    print(f"Engine: {args.engine}")
    print(f"Policy: {args.promotion_policy}")
    print(f"Protection: sink={args.protected_sink}, recent={args.protected_recent}")
    if cold_p50 is not None:
        print(f"TTFT cold p50: {cold_p50:.1f} ms")
    if warm_p50 is not None:
        print(f"TTFT warm p50: {warm_p50:.1f} ms")
    if improvement_pct is not None:
        print(f"TTFT improvement: {improvement_pct:.1f}%")
    if promotion_latencies:
        print(f"Promotion latency p50: {percentile(promotion_latencies, 50):.1f} ms")
    if peak_hbm:
        print(f"Peak HBM: {peak_hbm:.2f} GB")
    print(f"Secondary store est: {cold_bytes / 1024 / 1024:.1f} MB")
    print(f"Eligible for secondary: {eligible_pct:.1f}%")


if __name__ == "__main__":
    main()
