# Quickstart Prompts

Use these prompts with Codex or Claude Code. Each requires exact files, exact JSON outputs, exact rerun commands, one benchmark table, and one bottleneck note.

**Every prompt must obey:**
- No engine rewrite
- No multi-node before single-node success
- No claiming NVFP4 hot-KV unless support gate verifies it
- Produce machine-readable JSON matching the canonical schema
- Leave exact rerun commands

---

## 1. Environment Probe

```text
Run the environment probe and verify the support gate.

Files to read:
- CLAUDE.md (support gate section)
- scripts/env_probe.sh

Action:
- Run: bash scripts/env_probe.sh
- Verify results/env_probe.json exists and has all fields
- Report the support gate result: is_blackwell, nvfp4_kv_supported, fp8_kv_supported, recommended_hot_tier

Expected output:
- results/env_probe.json

Rerun command:
- bash scripts/env_probe.sh

Bottleneck note:
- If not Blackwell or no KV dtype support, document what blocked and pivot to bf16.
```

## 2. Aligned Baselines

```text
Run BF16 and FP8 baselines and produce a comparison table.

Files to read:
- scripts/run_baseline.py
- scripts/compare_results.py
- results/env_probe.json (must exist)

Actions:
1. python scripts/run_baseline.py --kv-mode bf16 --context-length 8192 --requests 10 --output results/baseline_bf16_8192.json
2. python scripts/run_baseline.py --kv-mode fp8 --context-length 8192 --requests 10 --output results/baseline_fp8_8192.json
3. python scripts/compare_results.py --output results/comparison.md

Expected outputs:
- results/baseline_bf16_8192.json
- results/baseline_fp8_8192.json
- results/comparison.md

Benchmark table:
- compare_results.py produces the table automatically

Bottleneck note:
- Is FP8 faster or slower than BF16? What's the HBM difference? If FP8 regresses, why?

Rerun commands:
- Exact commands above
```

## 3. LMCache Cold-Tier Reuse

```text
Run vLLM + LMCache cold-tier reuse on repeated-prefix workload.

Files to read:
- scripts/run_tiered_experiment.py
- TIERED_KV_ARCHITECTURE.md

Actions:
1. python scripts/run_tiered_experiment.py --kv-mode fp8 --promotion-policy demand --cold-tier-backend host_ram --context-length 8192 --requests 10 --output results/tiered_fp8_lmcache_8192.json
2. python scripts/compare_results.py --output results/comparison.md

Expected output:
- results/tiered_fp8_lmcache_8192.json
- Updated results/comparison.md

Benchmark table:
- BF16 vs FP8 vs FP8+LMCache

Bottleneck note:
- Does LMCache reuse improve TTFT on repeated-prefix traffic?
- What is the cache hit rate? What is the restore/promotion latency?
- Is the TTFT improvement meaningful enough to justify the cold-tier overhead?
```

## 4. Policy Ablation — Demand vs Eager

```text
Run eager promotion and compare against demand promotion from prompt 3.

Files to read:
- results/tiered_fp8_lmcache_8192.json (must exist from prompt 3)
- scripts/run_tiered_experiment.py

Actions:
1. python scripts/run_tiered_experiment.py --kv-mode fp8 --promotion-policy eager --cold-tier-backend host_ram --context-length 8192 --requests 10 --output results/tiered_fp8_eager_8192.json
2. python scripts/compare_results.py --output results/comparison.md

Expected output:
- results/tiered_fp8_eager_8192.json
- Updated results/comparison.md

Benchmark table:
- Demand vs eager: TTFT, promotion latency, HBM, cache hit rate

Bottleneck note:
- Which policy wins on TTFT? Which wins on HBM?
- Is eager pre-warming worth the upfront cost?
```

## 5. Concurrency / Request-Rate Sweep

```text
Run a concurrency and request-rate sweep to answer: at the same p95 latency target, how much higher concurrency can one GPU sustain with LMCache reuse enabled?

Files to read:
- results/tiered_fp8_lmcache_8192.json
- scripts/run_tiered_experiment.py

Actions:
1. For each concurrency in [1, 2, 4, 8, 16, 32]:
   python scripts/run_tiered_experiment.py --kv-mode fp8 --promotion-policy demand --cold-tier-backend host_ram --context-length 8192 --concurrency $C --output results/sweep_fp8_lmcache_c${C}.json
2. Stop when p95 TPOT becomes unacceptable (>10% regression vs baseline)
3. python scripts/compare_results.py --output results/comparison.md

Expected output:
- results/sweep_fp8_lmcache_c*.json
- Updated results/comparison.md

Benchmark table:
- Concurrency vs TTFT, TPOT p95, throughput, peak HBM

Bottleneck note:
- At what concurrency does p95 blow up?
- How many more concurrent sessions does LMCache reuse enable?
```

## 6. Slurm Sweep

```text
Submit baseline jobs via Slurm for bf16 and fp8.

Files to read:
- scripts/baseline_single_gpu.sbatch
- scripts/baseline_one_node.sbatch
- configs/blackwell_eval_matrix.tsv

Actions:
1. KV_MODE=bf16 sbatch scripts/baseline_single_gpu.sbatch
2. KV_MODE=fp8 sbatch scripts/baseline_single_gpu.sbatch
3. Wait for completion, check logs/
4. python scripts/compare_results.py --output results/comparison.md

Expected outputs:
- results/baseline_bf16_8192_<jobid>.json
- results/baseline_fp8_8192_<jobid>.json
- logs/kv-single-gpu-<jobid>.out

Only after single-GPU succeeds:
5. KV_MODE=fp8 sbatch scripts/baseline_one_node.sbatch

Bottleneck note:
- Did Slurm jobs complete cleanly? Any resource contention?
- Is one-node TP scaling linear or sublinear?
```

## 7. Decision Memo

```text
Produce the final comparison and decision memo.

Files to read:
- All results/baseline_*.json and results/tiered_*.json
- BLACKWELL_24H_PRD.md (primary KPI and success thresholds)

Actions:
1. python scripts/compare_results.py --output results/comparison.md
2. Read results/comparison.md
3. Write a one-paragraph continue/pivot/kill recommendation based on:
   - Did any tiered result meet the PRD target (≥25% more sessions OR ≥20% lower HBM)?
   - What was the TTFT improvement from tiering?
   - Was p95 TPOT regression within 10%?
   - What is the single biggest bottleneck?

Expected output:
- results/comparison.md (final version with all runs)

Benchmark table:
- Full matrix: bf16, fp8, fp8+lmcache, tiered_demand, tiered_eager

Bottleneck note:
- One sentence: "The primary bottleneck is X because Y"
- One sentence: "The biggest win is X with Y% improvement"

Rerun command:
- python scripts/compare_results.py --output results/comparison.md
```
