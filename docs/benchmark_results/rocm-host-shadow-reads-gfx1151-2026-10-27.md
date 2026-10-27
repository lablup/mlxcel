# Windows ROCm host-shadow reads: Radeon 8060S (gfx1151), 2026-10-27

Throughput before and after lablup/mlxcel#2643 (`overlays/rocm/LOCAL_FIXES.md` item 67), which stops marking a GPU buffer's pinned host shadow dirty when mlxcel only reads it, and the shadow counters the same change adds.

## Environment

Windows 11 Pro 10.0.26300, AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`), ROCm 10.1.0 (TheRock SDK at `C:\rocm\10.1.0`), MSVC toolchain, `--features rocm`. Before: `main` at `cbe7919f`. After: the same tree with this change. Both binaries were run from a plain `cmd.exe` environment (System32-only `PATH`, no `ROCM_PATH`, `HIP_PATH` or `ROCM_HOME`), one process at a time, no compiler running during the series.

## Protocol

`mlxcel generate -m <model> -n 256 --temp 0 --profile -p "<98-token prompt>"`, the same prompt for every run (a paragraph asking for an explanation of transformer decoding, ending in `/no_think`). Five runs per model and binary, the before series complete before the after binary was built. Prefill and decode tok/s are the `[Profile Results]` lines; the table gives the median with the range in parentheses.

| Model | Metric | Before | After | Change of the median |
|---|---|---:|---:|---:|
| Qwen3-0.6B-4bit | prefill tok/s | 81.23 (80.86 to 81.82) | 83.42 (82.30 to 85.04) | +2.7% |
| Qwen3-0.6B-4bit | decode tok/s | 102.58 (102.36 to 105.02) | 104.02 (101.61 to 104.50) | +1.4% |
| Qwen3-1.7B-bf16 | prefill tok/s | 70.62 (62.57 to 71.87) | 72.38 (68.81 to 74.73) | +2.5% |
| Qwen3-1.7B-bf16 | decode tok/s | 31.90 (31.85 to 32.04) | 31.95 (31.89 to 32.02) | +0.2% |

The 256 generated tokens were byte-identical between the two binaries in all ten pairs of runs (2212 bytes of text for the 0.6B model, 2056 for the 1.7B model), and identical across the five runs of each binary.

## Shadow counters

`MLX_ROCM_HOST_SHADOW_STATS=1 mlxcel generate -n 64 --temp 0 -p "What is the capital of France? Answer in one sentence. /no_think"` (11 tokens generated before the stop token), after binary:

| Model | Shadow bytes after load | Shadow bytes at exit | Downloads | Uploads |
|---|---:|---:|---:|---:|
| Qwen3-0.6B-4bit | 0 | 8 | 913 | 899 |
| Qwen3-1.7B-bf16 | 0 | 8 | 913 | 899 |

Weights are uploaded through `Load::eval_gpu`'s staging buffer and never get a shadow, so the resident shadow set is a few small buffers (8 bytes at exit: one scalar-pool slot). The 14 downloads without a matching upload are the reads the change leaves clean, one per generated token plus the prompt-side reads; before the change each of them was also an upload. The remaining 899 uploads are host-built scalars and index arrays that MLX constructs through `array::init` on every step (about 80 per token for these 28-layer models); those are writes and stay dirty by design. Each of them still costs a device synchronize, a device-to-host copy and a host-to-device copy, which is why the throughput change above is within a few percent: the read path was one flush in about eighty host accesses per token. The counters make that cost visible for the first time; shrinking it is a separate change.

## Memory estimate

`mlxcel inspect` and the `--estimate-memory` preflight on Windows ROCm builds reserve `batch × (vocab_size × 4 + ctx_len × 4)` bytes of pinned host memory (one f32 logits row and the int32 token ids of the context per sequence: 623 KB for Qwen3 at a 4096-token context) and compare it against the available physical memory `GlobalMemoryStatusEx` reports, not against the device limit. The measured resident set above stays far below that reserve; it is an upper bound for a load, not a prediction of the steady state.
