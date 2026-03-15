# B200 Slurm Eval Runbook

Retrieved and updated on `2026-03-15`.

This runbook is the operational handoff for B200 evals using TensorRT-LLM as the primary engine.

## Core Rule

**Single-GPU first. Then one-node. Then multi-node.** No exceptions.

Run the ladder in this exact order:

1. Login-node environment verification + support gate
2. Single-GPU TRT-LLM sanity run (BF16 → FP8 → NVFP4)
3. One-node TRT-LLM sweep (NVFP4 baselines)
4. One-node TRT-LLM NVFP4 + host offload
5. Optional: Kimi K2.5 stretch (node-level only)
6. Optional: multi-node B200 runs (only after one-node is stable)

That order matches standard Slurm best practice: use `sbatch` for repeatable jobs, keep long work off login nodes, and scale only after the smaller shape is stable. See `[R1]` and `[R2]`.

## Stop Conditions

Stop immediately and report if:

1. **Wrong GPU** — `nvidia-smi` does not show B200 or B100. Do not proceed on Hopper or older.
2. **TRT-LLM not installed** — TensorRT-LLM is not available in the environment. Install before proceeding.
3. **No KV dtype support** — neither NVFP4 nor FP8 KV cache works in TRT-LLM. Report env_probe.json.
4. **Driver too old** — driver version < 570.x. Blackwell requires 570+.
5. **CUDA too old** — CUDA version < 12.8.

If any stop condition triggers, document the blocker in `results/env_probe.json` and do not submit benchmark jobs.

## Pre-Flight Checklist

Before submitting anything expensive:

1. Run `bash scripts/env_probe.sh` and verify `results/env_probe.json`
2. Confirm the support gate: is_blackwell, nvfp4_kv_supported, fp8_kv_supported, trtllm_version
3. Confirm the target partition and quota
4. Confirm `B200` visibility with `nvidia-smi`
5. Confirm shared filesystem and node-local scratch path
6. Confirm the repo `git` SHA
7. Confirm the model path and output path
8. Confirm TensorRT-LLM is functional: `python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"`

## What Every Job Must Capture

Every eval job should save:

- `SLURM_JOB_ID`
- `SLURM_JOB_NODELIST`
- `nvidia-smi -L`
- `scontrol show job $SLURM_JOB_ID`
- `git rev-parse HEAD`
- exact eval command
- engine name and version (`tensorrt_llm`)
- model name
- context length
- batch size
- KV mode (bf16, fp8, nvfp4)
- offload enabled (yes/no)
- cold-tier codec (none, kvtc)

If we do not capture that metadata, the run is not trustworthy enough to compare later.

## Recommended Slurm Patterns

### Single-GPU smoke test

Use `srun` or a small `sbatch` job only to prove the stack works.

### One-node benchmark

Use `sbatch` with one node and explicit `--gpus-per-node`.

### Multi-node benchmark

Use `sbatch` with explicit `--nodes`, `--ntasks-per-node`, `--gpus-per-node`, and a consistent launch pattern. Only after one-node is stable.

### Job arrays

Use `--array` only for independent config sweeps, and throttle concurrency with `%N` so you do not stampede the cluster.

## Multi-Node Launch Pattern

The safest generic pattern for this repo is:

1. allocate `N` nodes with `sbatch`
2. compute `MASTER_ADDR` from the first hostname in `SLURM_JOB_NODELIST`
3. export `MASTER_PORT`, `NNODES`, `GPUS_PER_NODE`, and `WORLD_SIZE`
4. use `srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1` when one launcher process should run per node
5. use a per-node launcher that fans into local GPU workers if the engine requires it

If the serving stack wants one process per GPU instead, change the launch pattern explicitly and record that choice in the run metadata.

## Storage Rules

- shared filesystem is for code, persistent results, and logs
- `/tmp` or `/local_scratch` is for node-local scratch and temporary artifacts
- do not assume shared storage is a safe high-bandwidth scratch layer

## Weekend Eval Ladder

### Stage 0: Environment Verify + Support Gate

Run:

```bash
bash scripts/env_probe.sh
cat results/env_probe.json
```

Check: is_blackwell=true, trtllm_version exists, nvfp4_kv_supported or fp8_kv_supported.

If this fails, stop. Do not proceed to Stage 1.

### Stage 1: Single-GPU TRT-LLM Sanity Check

Run:

```bash
srun --gpus=1 --time=00:10:00 --pty bash
bash scripts/check_gpu.sh
# Quick TRT-LLM smoke test
python scripts/run_baseline.py --engine tensorrt_llm --kv-mode bf16 --model Qwen/Qwen3-8B-Instruct --context-length 8192 --requests 2 --output results/trtllm_smoke_bf16.json
exit
```

Verify the output JSON has engine: "tensorrt_llm" and all required fields.

### Stage 2: Single-GPU TRT-LLM Baseline Sweep

Run each via sbatch:

```bash
ENGINE=tensorrt_llm KV_MODE=bf16 SCENARIO_ID=scenario_1_longer_context_gpu sbatch scripts/baseline_single_gpu.sbatch
ENGINE=tensorrt_llm KV_MODE=fp8 SCENARIO_ID=scenario_1_longer_context_gpu sbatch scripts/baseline_single_gpu.sbatch
ENGINE=tensorrt_llm KV_MODE=nvfp4 SCENARIO_ID=scenario_1_longer_context_gpu sbatch scripts/baseline_single_gpu.sbatch
```

Wait for completion. Verify all three result JSONs exist and compare:

```bash
python scripts/compare_results.py --output results/comparison.md
```

### Stage 3: One-Node TRT-LLM NVFP4 Sweep

Only after Stage 2 succeeds. Submit one-node jobs:

```bash
sbatch \
  --nodes=1 \
  --gpus-per-node=8 \
  --export=ALL,ENGINE=tensorrt_llm,EVAL_COMMAND='python scripts/run_baseline.py --engine tensorrt_llm --kv-mode nvfp4 --context-length 8192' \
  scripts/b200_eval.sbatch
```

### Stage 4: One-Node TRT-LLM NVFP4 + Host Offload

Only after Stage 3 succeeds:

```bash
sbatch \
  --nodes=1 \
  --gpus-per-node=8 \
  --export=ALL,ENGINE=tensorrt_llm,EVAL_COMMAND='python scripts/run_tiered_experiment.py --engine tensorrt_llm --kv-mode nvfp4 --offload-to-host --context-length 8192 --requests 10' \
  scripts/b200_eval.sbatch
```

### Stage 5: Optional Kimi K2.5 Stretch (Node-Level Only)

Only after Stage 4 succeeds. Kimi K2.5 is verified on 8x H200 [R10]. Do NOT attempt on single-GPU.

```bash
sbatch \
  --nodes=1 \
  --gpus-per-node=8 \
  --export=ALL,ENGINE=tensorrt_llm,MODEL=moonshotai/Kimi-K2.5,EVAL_COMMAND='python scripts/run_baseline.py --engine tensorrt_llm --kv-mode nvfp4 --model moonshotai/Kimi-K2.5 --context-length 8192' \
  scripts/b200_eval.sbatch
```

### Stage 6: Optional Multi-Node B200

Only after the one-node ladder is stable:

- keep the eval matrix small
- start with a single model (`Qwen/Qwen3-30B-A3B`)
- start with a small set of contexts
- save all metadata

```bash
sbatch \
  --nodes=2 \
  --gpus-per-node=8 \
  --export=ALL,ENGINE=tensorrt_llm,EVAL_COMMAND='python scripts/run_baseline.py --engine tensorrt_llm --kv-mode nvfp4 --context-length 32768' \
  scripts/b200_eval.sbatch
```

## Example Invocations (TRT-LLM Primary)

### Single-GPU baseline

```bash
ENGINE=tensorrt_llm KV_MODE=nvfp4 SCENARIO_ID=scenario_1_longer_context_gpu sbatch scripts/baseline_single_gpu.sbatch
```

### One-node baseline

```bash
sbatch \
  --nodes=1 \
  --gpus-per-node=8 \
  --export=ALL,ENGINE=tensorrt_llm,EVAL_COMMAND='python scripts/run_baseline.py --engine tensorrt_llm --kv-mode nvfp4 --context-length 8192' \
  scripts/b200_eval.sbatch
```

### One-node NVFP4 + offload

```bash
sbatch \
  --nodes=1 \
  --gpus-per-node=8 \
  --export=ALL,ENGINE=tensorrt_llm,EVAL_COMMAND='python scripts/run_tiered_experiment.py --engine tensorrt_llm --kv-mode nvfp4 --offload-to-host --context-length 8192 --requests 10' \
  scripts/b200_eval.sbatch
```

### Array sweep

```bash
sbatch \
  --array=0-3%1 \
  --export=ALL,ENGINE=tensorrt_llm,EVAL_COMMAND='python scripts/run_baseline.py --engine tensorrt_llm --kv-mode ${VARIANT} --context-length ${CONTEXT_LENGTH} --batch-size ${BATCH_SIZE}' \
  scripts/b200_eval_array.sbatch
```

## Repo Scripts

This directory includes generic Slurm entrypoints:

- `scripts/b200_eval.sbatch`
- `scripts/b200_eval_array.sbatch`
- `scripts/run_b200_eval.sh`
- `scripts/baseline_single_gpu.sbatch`
- `scripts/baseline_one_node.sbatch`
- `scripts/env_probe.sh`

They are intentionally generic so the next engineer can swap in the actual serving command without rewriting the metadata capture layer.

## References

- `[R1]` Slurm `sbatch` man page
  - <https://slurm.schedmd.com/sbatch.html>
- `[R2]` Slurm `srun` man page
  - <https://slurm.schedmd.com/srun.html>
- `[R3]` Slurm job arrays
  - <https://slurm.schedmd.com/job_array.html>
- `[R7]` NVIDIA NVFP4 KV cache blog
  - <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- `[R8]` TRT-LLM KV cache system
  - <https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html>
- `[R10]` Kimi K2.5 (8x H200 verified) — stretch only
