# slurm-hpc-executor

Use this skill when the task involves Slurm commands, cluster-safe execution, batch scripts, or environment bootstrapping for Blackwell B200 GPU work.

## Core Rules

- Distinguish login-node checks from compute-node and batch workloads.
- Prefer `sbatch` for long or reproducible runs.
- Use short `srun` sessions only for quick validation or debugging.
- Keep outputs predictable under `logs/` and `results/`.
- Make environment assumptions explicit in scripts.
- Run single-GPU validation before multi-GPU escalation.

## B200 Execution Ladder

Follow the order from `B200_SLURM_EVAL_RUNBOOK.md`:

1. Login-node environment verification (`scripts/check_cluster.sh`)
2. Single-GPU B200 sanity run (`srun --gpus=1`)
3. One-node baseline sweep (`sbatch --nodes=1 --gpus-per-node=8`)
4. One-node `NVFP4 + KVTC` sweep
5. Multi-node B200 runs only after one-node is stable

## Metadata Capture Requirements

Every eval job must save:

- `SLURM_JOB_ID` and `SLURM_JOB_NODELIST`
- `nvidia-smi -L` output
- `scontrol show job $SLURM_JOB_ID`
- `git rev-parse HEAD`
- exact eval command, model name, context length, batch size, KV mode

If metadata is missing, the run cannot be compared later.

## Workflow

1. Verify whether the task belongs on a login node, interactive allocation, or batch job.
2. Check or add guardrails in `scripts/` so a collaborator can run the workflow without guesswork.
3. Ensure batch scripts create their own output directories when needed.
4. Prefer clear job names, resource requests, and output file patterns.
5. Make the B200 device expectation explicit in the command examples.
6. Summarize the exact submit or debug commands after edits.

## Deliverables

- safe Slurm scripts following `B200_SLURM_EVAL_RUNBOOK.md`
- explicit setup assumptions
- reproducible command examples with metadata capture

## Sources

- `B200_SLURM_EVAL_RUNBOOK.md`: multi-node B200 eval runbook for this repo
- Slurm `sbatch` man page: <https://slurm.schedmd.com/sbatch.html>
- Slurm `srun` man page: <https://slurm.schedmd.com/srun.html>
- Slurm job arrays: <https://slurm.schedmd.com/job_array.html>
