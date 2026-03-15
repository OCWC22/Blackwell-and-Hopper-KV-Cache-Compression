# Hopper KV Cache Compression

KV cache compression experiments on Hopper GPUs (H100/H200/B200) via SLURM cluster.

## Quick Start

Once SSH access is live:

```bash
# 1. SSH into login node
ssh <username>@<login-node-ip>

# 2. Clone this repo
git clone https://github.com/OCWC22/Hopper-KV-Cache-Compression.git
cd Hopper-KV-Cache-Compression

# 3. Verify cluster
bash scripts/check_cluster.sh

# 4. Get a GPU and verify
srun --gpus=1 --pty bash
bash scripts/check_gpu.sh
exit

# 5. Set up environment
bash scripts/setup_env.sh

# 6. Run baseline benchmark
sbatch scripts/baseline_inference.sh

# 7. Check results
tail -f logs/kv-baseline-*.out
```

## Structure

```
scripts/
  check_cluster.sh      # Verify SLURM + filesystem (login node)
  check_gpu.sh          # Verify GPU + CUDA + PyTorch (compute node)
  setup_env.sh          # Install Python deps
  baseline_inference.sh # SBATCH: baseline vLLM decode benchmark
  run_baseline.py       # Baseline measurement script
configs/                # Experiment configs
results/                # Benchmark outputs (JSON)
logs/                   # SLURM job logs
src/                    # KV compression implementations
```

## Experiment Plan

1. **Baseline**: vLLM decode latency + memory at 8k/32k/64k context
2. **KV Compression**: Add compression runtime, measure overhead vs savings
3. **Accuracy**: Verify generation quality with compressed KV cache
