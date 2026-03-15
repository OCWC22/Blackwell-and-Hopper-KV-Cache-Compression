# nvfp4-kvtc-runtime

Use this skill when implementing or reviewing the Blackwell-native `NVFP4 + KVTC` runtime.

## Mental Model

- `NVFP4` is the resident active-KV format.
- `KVTC` is the warm or cold representation for stale or reusable KV.
- The main systems risk is promotion latency, not whether compression exists in principle.
- Quality protection is mandatory if the tiering policy becomes more aggressive.

## Implementation Guidance

1. Define what stays resident in `NVFP4`.
2. Define what is eligible for `KVTC`.
3. Make promotion logging explicit, including policy and hit or miss counts.
4. Keep the hot and warm path modular so policy work can evolve later.
5. Measure promotion overhead alongside memory savings.

## Things To Avoid

- claiming `KVTC` is a hot-path win before measuring latency
- mixing `LMCache` baseline claims with `KVTC` paper claims
- hiding policy assumptions inside unrelated code
