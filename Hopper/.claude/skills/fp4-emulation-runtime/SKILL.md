# fp4-emulation-runtime

Use this skill when implementing or reviewing FP4-like KV-cache storage on Hopper.

## Mental Model

- FP4 is for storage and movement, not native Hopper compute.
- Packed values should be cheap to unpack and reconstruct.
- Compute should happen in FP8, BF16, or FP16 after reconstruction.
- Direct FP8-oriented reconstruction is preferable to unnecessary BF16 round-trips.

## Implementation Guidance

1. Define the packed representation clearly.
2. Make scale metadata explicit, including granularity and placement.
3. Keep encode and decode paths modular so kernel work can evolve later.
4. Add at least one correctness check that verifies reconstructed values against a higher-precision reference.
5. Measure pack and reconstruct overhead alongside memory savings.
6. Make grouping strategy and protected-token policy visible in the config or output.

## Things To Avoid

- claiming native Hopper FP4 tensor-core execution
- mixing storage precision claims with compute precision claims
- hiding scale or zero-point assumptions inside unrelated code
