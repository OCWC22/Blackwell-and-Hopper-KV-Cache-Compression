# ModelOpt NVFP4 Quantization

Use this skill when: running ModelOpt quantization for NVFP4 KV cache, preparing FP8 plus NVFP4 combined configs, exporting checkpoints for TensorRT-LLM, or validating NVFP4 KV accuracy.

## Core Rules

- NVFP4 is native to Blackwell SM120; do not attempt on Hopper
- NVFP4 uses E2M1 payload with FP8 E4M3 scale per 16-value micro-block and FP32 per-tensor second-level scale
- Blackwell attention dequantizes NVFP4 to FP8 before compute
- Expect roughly 50% memory reduction vs FP8 KV cache
- NVIDIA documents <1% accuracy loss on LiveCodeBench, MMLU-PRO, MBPP, and Ruler 64K

## Quantization Recipe

Combined FP8 weights and activations with NVFP4 KV cache:

```python
import modelopt.torch.quantization as mtq

quant_cfg = mtq.FP8_DEFAULT_CFG
quant_cfg["quant_cfg"].update(mtq.NVFP4_KV_CFG["quant_cfg"])

def forward_loop(model):
    for data in calib_set:
        model(data)

model = mtq.quantize(model, quant_cfg, forward_loop)
```

This is the NVIDIA-documented configuration pattern.

## Calibration Dataset

- Use representative workload samples: chat prefixes, agent scaffolds, code, long documents
- Keep calibration set small but diverse (hundreds of samples, not thousands)
- Forward loop runs inference to collect activation statistics

## Deployment Paths

1. **TensorRT-LLM (primary)**: export quantized checkpoint, build TRT-LLM engine, NVFP4 KV is fully supported
2. **vLLM (secondary)**: vLLM supports ModelOpt NVFP4 checkpoints but public KV docs are FP8-centered; treat as compatibility track

## Things To Avoid

- Running NVFP4 quantization on Hopper hardware (not supported)
- Skipping calibration and relying on default scales
- Assuming NVFP4 replaces the need for KVTC warm/cold tiering
- Conflating FP8 KV baseline results with NVFP4 KV results
- Using vLLM as primary NVFP4-KV path when TensorRT-LLM is more mature for this
