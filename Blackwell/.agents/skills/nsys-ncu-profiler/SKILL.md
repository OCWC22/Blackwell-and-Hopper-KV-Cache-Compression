# nsys-ncu-profiler

Use this skill when preparing or analyzing Nsight Systems or Nsight Compute profiling runs for the Blackwell NVFP4 + KVTC runtime.

## Profiling Rules

- Profile only on an allocated B200 GPU node.
- Keep captures scoped to the kernels or phases that matter.
- Save traces and summaries under `results/` so they can be shared later.
- Use profiling evidence to explain promotion or decode bottlenecks, not to decorate guesses.

## Blackwell-Specific Profiling Targets

- **NVFP4 dequant kernels**: NVFP4 → FP8 dequantization in the attention path (OMMA instruction family)
- **KVTC promotion path**: DEFLATE decompress → inverse PCA → FP8 staging → NVFP4 pack
- **Decode attention**: memory-bandwidth bound attention kernel reading from NVFP4 KV cache
- **Promotion I/O**: PCIe Gen6 host → GPU transfer of KVTC blobs during promotion
- **OMMA instructions**: Blackwell 5th-gen tensor core MMA ops (`tcgen05.mma`) for FP4/FP8 compute

## Workflow

1. Decide whether timeline tracing (`nsys`) or kernel-level analysis (`ncu`) is the right first tool.
2. Add a Slurm-safe profiling entrypoint when needed.
3. Keep output filenames deterministic.
4. Focus on: decode stalls, promotion overhead, NVFP4 dequant cost, memory traffic patterns.
5. Summarize what to compare before and after an optimization.

## Sources

- Nsight Systems docs: <https://docs.nvidia.com/nsight-systems/>
- Nsight Systems User Guide: <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- Nsight Compute docs: <https://docs.nvidia.com/nsight-compute/>
- Blackwell Tuning Guide: <https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html>
