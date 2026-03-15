# B200 Slurm Eval Runbook

Retrieved and updated on `2026-03-14`.

This runbook is the operational handoff for multi-node `B200` evals.

## Core Rule

Do not jump straight to a large multi-node sweep.

Run the ladder in this exact order:

1. login-node environment verification
2. single-GPU Blackwell sanity run
3. one-node baseline sweep
4. one-node `FP8+LMCache` sweep
5. only then multi-node `B200` runs

That order matches standard Slurm best practice: use `sbatch` for repeatable jobs, keep long work off login nodes, and scale only after the smaller shape is stable. See `[R1]` and `[R2]`.

## Pre-Flight Checklist

Before submitting anything expensive:

1. confirm the target partition and quota
2. confirm `B200` visibility with `nvidia-smi`
3. confirm shared filesystem and node-local scratch path
4. confirm the repo `git` SHA
5. confirm the model path and output path

## What Every Job Must Capture

Every eval job should save:

- `SLURM_JOB_ID`
- `SLURM_JOB_NODELIST`
- `nvidia-smi -L`
- `scontrol show job $SLURM_JOB_ID`
- `git rev-parse HEAD`
- exact eval command
- model name
- context length
- batch size
- KV mode

If we do not capture that metadata, the run is not trustworthy enough to compare later.

## Recommended Slurm Patterns

### Single-GPU smoke test

Use `srun` or a small `sbatch` job only to prove the stack works.

### One-node benchmark

Use `sbatch` with one node and explicit `--gpus-per-node`.

### Multi-node benchmark

Use `sbatch` with explicit `--nodes`, `--ntasks-per-node`, `--gpus-per-node`, and a consistent launch pattern.

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

### Stage 0: verify the cluster

Run:

```bash
bash scripts/check_cluster.sh
```

### Stage 1: single-GPU sanity check

Run:

```bash
srun --gpus=1 --time=00:10:00 --pty bash
bash scripts/check_gpu.sh
exit
```

### Stage 2: one-node baseline

Run the generic batch harness with one node and the exact baseline command.

### Stage 3: one-node `FP8+LMCache` sweep

Repeat the same harness with:

- `BF16`
- `FP8`
- `FP8+LMCache`

### Stage 4: optional one-node `NVFP4`

Only if runtime support is verified, run optional NVFP4 enhancement with protection-policy metadata captured.

### Stage 5: multi-node B200

Only after the one-node ladder is stable:

- keep the eval matrix small
- start with a single model
- start with a small set of contexts
- save all metadata

## Repo Scripts

This directory includes generic Slurm entrypoints:

- `scripts/b200_eval.sbatch`
- `scripts/b200_eval_array.sbatch`
- `scripts/run_b200_eval.sh`

They are intentionally generic so the next engineer can swap in the actual serving command without rewriting the metadata capture layer.

## Example Invocations

### One-node baseline

```bash
sbatch \
  --nodes=1 \
  --gpus-per-node=8 \
  --export=ALL,EVAL_COMMAND='python scripts/run_baseline.py --kv-mode fp8 --context-length 8192' \
  scripts/b200_eval.sbatch
```

### Multi-node run

```bash
sbatch \
  --nodes=2 \
  --gpus-per-node=8 \
  --export=ALL,EVAL_COMMAND='python your_launcher.py --kv-mode nvfp4 --context-length 32768' \
  scripts/b200_eval.sbatch
```

### Array sweep

```bash
sbatch \
  --array=0-3%1 \
  --export=ALL,EVAL_COMMAND='python your_launcher.py --kv-mode ${VARIANT} --context-length ${CONTEXT_LENGTH} --batch-size ${BATCH_SIZE}' \
  scripts/b200_eval_array.sbatch
```

## References

- `[R1]` Slurm `sbatch` man page
  - <https://slurm.schedmd.com/sbatch.html>
- `[R2]` Slurm `srun` man page
  - <https://slurm.schedmd.com/srun.html>
- `[R3]` Slurm job arrays
  - <https://slurm.schedmd.com/job_array.html>
