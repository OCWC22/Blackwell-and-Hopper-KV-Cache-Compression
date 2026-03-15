# slurm-hpc-executor

Use this skill when the task involves Slurm commands, cluster-safe execution, batch scripts, or environment bootstrapping for Hopper GPU work.

## Core Rules

- Distinguish login-node checks from compute-node and batch workloads.
- Prefer `sbatch` for long or reproducible runs.
- Use short `srun` sessions only for quick validation or debugging.
- Keep outputs predictable under `logs/` and `results/`.
- Make environment assumptions explicit in scripts.
- Run single-GPU H100 or H200 validation before multi-GPU escalation.

## Workflow

1. Verify whether the task belongs on a login node, interactive allocation, or batch job.
2. Check or add guardrails in `scripts/` so a collaborator can run the workflow without guesswork.
3. Ensure batch scripts create their own output directories when needed.
4. Prefer clear job names, resource requests, and output file patterns.
5. Make the Hopper device expectation explicit in the command examples.
6. Summarize the exact submit or debug commands after edits.

## Deliverables

- safe Slurm scripts
- explicit setup assumptions
- reproducible command examples
