# Quickstart Prompts

Use these prompts with Codex or Claude Code. Each requires exact files, exact JSON outputs, exact rerun commands, one benchmark table, and one bottleneck note.

**Every prompt must obey:**
- No engine rewrite
- No multi-node before single-GPU success
- No claiming NVFP4 hot-KV unless support gate verifies it
- Produce machine-readable JSON matching the canonical schema
- Leave exact rerun commands
- TensorRT-LLM is the primary engine; vLLM + LMCache is the follow-up path

---

## 1. Environment Probe + Support Gate (Pre-scenario)

```text
Run the environment probe and verify the support gate.

Files to read:
- CLAUDE.md (support gate section)
- scripts/env_probe.sh

Files to edit:
- None (probe only)

Action:
- Run: bash scripts/env_probe.sh
- Verify results/env_probe.json exists and has all fields
- Report the support gate result: is_blackwell, nvfp4_kv_supported, fp8_kv_supported, trtllm_version, recommended_hot_tier
- Confirm TensorRT-LLM is installed and functional
- Confirm ModelOpt is available for PTQ if needed

Expected JSON output:
- results/env_probe.json (must contain scenario_id-ready support gate, trtllm_version field)

Rerun command:
- bash scripts/env_probe.sh

Bottleneck note:
- If not Blackwell or no KV dtype support, document what blocked and pivot to bf16.
- If TensorRT-LLM is not installed, stop and install before proceeding.

Rules: no engine rewrite, no multi-node before single-GPU success.
```

## 2. Aligned Baselines — TRT-LLM BF16 + FP8 (Scenarios 1 & 2)

```text
Run TensorRT-LLM BF16 and FP8 baselines and produce a comparison table.

Files to read:
- scripts/run_baseline.py
- scripts/compare_results.py
- results/env_probe.json (must exist)

Files to edit:
- None (run only)

Actions:
1. python scripts/run_baseline.py --engine tensorrt_llm --kv-mode bf16 --model Qwen/Qwen3-30B-A3B --context-length 8192 --requests 10 --scenario-id scenario_1_longer_context_gpu --output results/trtllm_baseline_bf16_8192.json
2. python scripts/run_baseline.py --engine tensorrt_llm --kv-mode fp8 --model Qwen/Qwen3-30B-A3B --context-length 8192 --requests 10 --scenario-id scenario_1_longer_context_gpu --output results/trtllm_baseline_fp8_8192.json
3. python scripts/compare_results.py --output results/comparison.md

Smoke test (use first if model too large):
- python scripts/run_baseline.py --engine tensorrt_llm --kv-mode bf16 --model Qwen/Qwen3-8B-Instruct --context-length 8192 --requests 5 --scenario-id scenario_1_longer_context_gpu --output results/trtllm_smoke_bf16_8192.json

Expected JSON outputs:
- results/trtllm_baseline_bf16_8192.json (must contain scenario_id, engine: "tensorrt_llm")
- results/trtllm_baseline_fp8_8192.json (must contain scenario_id, engine: "tensorrt_llm")
- results/comparison.md

Benchmark table:
- compare_results.py produces the table automatically with scenario column

Bottleneck note:
- Is FP8 faster or slower than BF16? What's the HBM difference? If FP8 regresses, why?

Rerun commands:
- Exact commands above

Rules: no engine rewrite, no multi-node before single-GPU success.
```

## 3. TRT-LLM NVFP4 Baseline (Scenario 1 & 2 — Blackwell primary)

```text
Run TensorRT-LLM NVFP4 KV cache baseline. This is the Blackwell primary hot-tier path.

Files to read:
- results/env_probe.json (must show nvfp4_kv_supported: true)
- results/trtllm_baseline_fp8_8192.json (FP8 baseline must exist)
- scripts/run_baseline.py

Files to edit:
- None (run only)

Actions:
1. python scripts/run_baseline.py --engine tensorrt_llm --kv-mode nvfp4 --model Qwen/Qwen3-30B-A3B --context-length 8192 --requests 10 --scenario-id scenario_1_longer_context_gpu --output results/trtllm_baseline_nvfp4_8192.json
2. python scripts/run_baseline.py --engine tensorrt_llm --kv-mode nvfp4 --model Qwen/Qwen3-30B-A3B --context-length 32768 --requests 10 --scenario-id scenario_1_longer_context_gpu --output results/trtllm_baseline_nvfp4_32768.json
3. python scripts/compare_results.py --output results/comparison.md

Expected JSON outputs:
- results/trtllm_baseline_nvfp4_8192.json (must contain engine: "tensorrt_llm", kv_mode: "nvfp4")
- results/trtllm_baseline_nvfp4_32768.json
- Updated results/comparison.md

Benchmark table:
- BF16 vs FP8 vs NVFP4 with scenario column

Bottleneck note:
- What is the HBM reduction from NVFP4 vs FP8? vs BF16?
- Any quality regression from NVFP4? Check quality delta vs BF16.
- Is NVFP4 decode latency competitive with FP8?

Rules: no engine rewrite, no multi-node before single-GPU success.
```

## 4. TRT-LLM NVFP4 + Host Offload (Scenario 3 — PRIMARY)

```text
Run TensorRT-LLM NVFP4 with host memory offloading on repeated-prefix workload. This is the Scenario 3 primary path.

Files to read:
- scripts/run_tiered_experiment.py
- results/trtllm_baseline_nvfp4_8192.json (must exist from prompt 3)

Files to edit:
- None (run only)

Actions:
1. python scripts/run_tiered_experiment.py --engine tensorrt_llm --kv-mode nvfp4 --offload-to-host --promotion-policy demand --context-length 8192 --requests 10 --scenario-id scenario_3_longer_context_more_sessions_gpu --output results/trtllm_nvfp4_offload_demand_8192.json
2. python scripts/run_tiered_experiment.py --engine tensorrt_llm --kv-mode nvfp4 --offload-to-host --promotion-policy eager --context-length 8192 --requests 10 --scenario-id scenario_3_longer_context_more_sessions_gpu --output results/trtllm_nvfp4_offload_eager_8192.json
3. python scripts/compare_results.py --output results/comparison.md

Expected JSON outputs:
- results/trtllm_nvfp4_offload_demand_8192.json (must contain engine, kv_mode, offload_enabled, promotion_policy)
- results/trtllm_nvfp4_offload_eager_8192.json
- Updated results/comparison.md

Benchmark table:
- NVFP4 baseline vs NVFP4+offload(demand) vs NVFP4+offload(eager)
- Columns: TTFT, p95 TPOT, peak HBM, max concurrent sessions, cache hit rate, quality delta

Bottleneck note:
- Does host offloading improve TTFT on repeated-prefix traffic?
- What is the restore/promotion latency? Is it acceptable?
- Which policy wins on TTFT? Which wins on HBM?

Rules: no engine rewrite, no multi-node before single-GPU success.
```

## 5. Concurrency Sweep — TRT-LLM NVFP4 (Scenario 3 serving mode)

```text
Run a concurrency sweep to answer: at the same p95 latency target, how much higher concurrency can one GPU sustain with TRT-LLM NVFP4 + offload?

Files to read:
- results/trtllm_nvfp4_offload_demand_8192.json
- scripts/serve_and_bench.py

Files to edit:
- None (run only)

Actions:
1. python scripts/serve_and_bench.py --engine tensorrt_llm --kv-mode nvfp4 --sweep-concurrency 1,2,4,8,16,32 --p95-tpot-limit-ms 100 --scenario-id scenario_3_longer_context_more_sessions_gpu --output results/serve_trtllm_nvfp4_baseline.json
2. python scripts/serve_and_bench.py --engine tensorrt_llm --kv-mode nvfp4 --offload-to-host --sweep-concurrency 1,2,4,8,16,32 --p95-tpot-limit-ms 100 --scenario-id scenario_3_longer_context_more_sessions_gpu --output results/serve_trtllm_nvfp4_offload.json
3. python scripts/compare_results.py --output results/comparison.md

Expected JSON outputs:
- results/serve_trtllm_nvfp4_baseline.json (serving_mode: online, scenario_id set)
- results/serve_trtllm_nvfp4_offload.json
- Updated results/comparison.md

Benchmark table:
- Concurrency vs TTFT, TPOT p95, throughput, peak HBM, max_concurrent_at_p95_target

Bottleneck note:
- At what concurrency does p95 blow up?
- How many more concurrent sessions does NVFP4 + offload enable vs NVFP4 alone?

Rules: no engine rewrite, no multi-node before single-GPU success.
```

## 6. Slurm Sweep — TRT-LLM (Scenarios 1 & 4)

```text
Submit TRT-LLM baseline and NVFP4 jobs via Slurm.

Files to read:
- scripts/baseline_single_gpu.sbatch
- scripts/baseline_one_node.sbatch
- configs/blackwell_eval_matrix.tsv

Files to edit:
- None (run only)

Actions (single-GPU first):
1. ENGINE=tensorrt_llm KV_MODE=bf16 SCENARIO_ID=scenario_1_longer_context_gpu sbatch scripts/baseline_single_gpu.sbatch
2. ENGINE=tensorrt_llm KV_MODE=fp8 SCENARIO_ID=scenario_1_longer_context_gpu sbatch scripts/baseline_single_gpu.sbatch
3. ENGINE=tensorrt_llm KV_MODE=nvfp4 SCENARIO_ID=scenario_1_longer_context_gpu sbatch scripts/baseline_single_gpu.sbatch
4. Wait for completion, check logs/
5. python scripts/compare_results.py --output results/comparison.md

Expected JSON outputs:
- results/trtllm_baseline_bf16_8192_<jobid>.json (with scenario_id, engine)
- results/trtllm_baseline_fp8_8192_<jobid>.json
- results/trtllm_baseline_nvfp4_8192_<jobid>.json
- logs/kv-single-gpu-<jobid>.out

Only after single-GPU succeeds (Scenario 4):
6. ENGINE=tensorrt_llm KV_MODE=nvfp4 SCENARIO_ID=scenario_4_longer_context_more_sessions_node sbatch scripts/baseline_one_node.sbatch

Optional Kimi K2.5 stretch (node-level only, 8x H200 verified, NOT for single-GPU):
7. ENGINE=tensorrt_llm KV_MODE=nvfp4 MODEL=moonshotai/Kimi-K2.5 SCENARIO_ID=scenario_4_longer_context_more_sessions_node sbatch scripts/baseline_one_node.sbatch

Bottleneck note:
- Did Slurm jobs complete cleanly? Any resource contention?
- Is one-node TP scaling linear or sublinear?

Rules: no engine rewrite, no multi-node before single-GPU success.
```

## 7. Decision Memo

```text
Produce the final comparison and decision memo.

Files to read:
- All results/trtllm_*.json and results/serve_trtllm_*.json
- BLACKWELL_24H_PRD.md (primary KPI and success thresholds)

Actions:
1. python scripts/compare_results.py --output results/comparison.md
2. Read results/comparison.md
3. Write a one-paragraph continue/pivot/kill recommendation based on:
   - Did any TRT-LLM NVFP4 + offload result meet the PRD target (>=25% more sessions OR >=20% lower HBM)?
   - What was the TTFT improvement from NVFP4 + offloading?
   - Was p95 TPOT regression within 10%?
   - What is the single biggest bottleneck?

Expected output:
- results/comparison.md (final version with all runs)

Benchmark table:
- Full matrix: TRT-LLM BF16, TRT-LLM FP8, TRT-LLM NVFP4, TRT-LLM NVFP4+offload, optional TRT-LLM NVFP4+offload+KVTC

Bottleneck note:
- One sentence: "The primary bottleneck is X because Y"
- One sentence: "The biggest win is X with Y% improvement"

Rerun command:
- python scripts/compare_results.py --output results/comparison.md
```

---

## Follow-Up: vLLM + LMCache Compatibility Path

These prompts are for the follow-up compatibility/productization path. Run only after the TRT-LLM primary path is complete and stable.

### F1. vLLM FP8 Baseline

```text
Run vLLM FP8 baseline for comparison against TRT-LLM results.

Actions:
1. python scripts/run_baseline.py --engine vllm --kv-mode fp8 --model Qwen/Qwen3-30B-A3B --context-length 8192 --requests 10 --scenario-id scenario_3_longer_context_more_sessions_gpu --output results/vllm_baseline_fp8_8192.json

Rules: no engine rewrite, no multi-node before single-GPU success. This is the follow-up path.
```

### F2. vLLM FP8 + LMCache Cold-Tier Reuse

```text
Run vLLM + LMCache cold-tier reuse on repeated-prefix workload.

Actions:
1. python scripts/run_tiered_experiment.py --engine vllm --use-lmcache --kv-mode fp8 --promotion-policy demand --cold-tier-backend host_ram --context-length 8192 --requests 10 --scenario-id scenario_3_longer_context_more_sessions_gpu --output results/vllm_tiered_fp8_lmcache_8192.json
2. python scripts/compare_results.py --output results/comparison.md

Rules: no engine rewrite, no multi-node before single-GPU success. This is the follow-up path.
```

---

## Kimi K2.5 Stretch Note

`moonshotai/Kimi-K2.5` is verified on 8x H200 nodes. It is a stretch goal only for node-level (Scenario 4) experiments. Do NOT attempt Kimi K2.5 on single-GPU Blackwell proof runs. Cut Kimi K2.5 first if time is short.
