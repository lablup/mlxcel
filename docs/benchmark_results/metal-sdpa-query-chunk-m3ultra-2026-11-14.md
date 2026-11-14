# Metal query-chunked SDPA fallback (issue #3033), M3 Ultra, 2026-11-14

Issue #3033: on Metal, a prefill attention call that MLX does not fuse runs MLX's graph fallback, which materializes the whole `[B, heads, L, L]` score matrix in one buffer. At 26.6K tokens on a 24-head, head_dim 256 model that buffer is `L^2 x 48` bytes (33.9 GB) and failed with `metal::malloc` on an M4 Pro whose largest Metal buffer is 30.15 GB. The query-axis chunking that bounds this on CUDA returned early on every other backend.

The fix runs the chunk gate on Metal and asks MLX's own `ScaledDotProductAttention::use_fallback` for the mask mode the call actually issues (array mask, causal mode, or none); a softcap composite always counts as materializing. It chunks only when that verdict is "fallback" and the scores would exceed `MLXCEL_ATTENTION_CHUNK_BUDGET_MB`, whose Metal default is 256 MiB. `MLXCEL_METAL_ATTENTION_CHUNK=0` skips the gate and is exactly main's Metal dispatch, so every "before" figure below is the same binary with that switch.

Host: Apple M3 Ultra, 80-core GPU, 512 GB, no NAX (pre-M5). Branch `fix/issue-3033-metal-sdpa-chunk`, release build. Greedy decoding (temperature 0), 48 generated tokens, reasoning included in the compared text. One run at a time on an idle GPU.

## Results

### 1. Unfused case: the reporter's call, at 28K tokens

On this build the Qwen 3.5 family's full-attention layers receive an explicit additive causal mask (`Qwen35Model::forward_internal` builds `create_causal_mask`), so a single-pass prefill issues an array-mask SDPA at head_dim 256. MLX does not fuse that before M5, so this is the call the reporter hit, with no forcing needed. The debug log shows the gated calls (`q=[1, 16, 28707, 256] k=[1, 4, 28707, 256] arr_masked=true`, 8 per prefill on the 4B; `q=[1, 24, 28749, 256]`, 16 on the 27B).

| run | prompt tokens | largest score buffer before | peak footprint before | peak after | prefill before | prefill after | output |
|---|---|---|---|---|---|---|---|
| qwen3.5-4b-4bit CLI, single pass | 28707 | 26.4 GB | 66.46 GB | **44.38 GB** | 18.79 s | 18.61 s | identical |
| qwen3.8-27b-4bit CLI, single pass | 28749 | 39.7 GB | 114.05 GB | **78.48 GB** | 89.28 s | 90.31 s | identical |
| qwen3.5-4b-4bit server, `--parallel 1` | 28707 | 26.4 GB | 66.65 GB | **44.59 GB** | 18.90 s TTFT | 18.86 s TTFT | identical |
| qwen3.8-27b-4bit server, `--parallel 1` | 28749 | 39.7 GB | 114.21 GB | **78.65 GB** | 87.54 s TTFT | 89.09 s TTFT | identical |

The 27B buffer (`24 x 28749^2 x 2 B`) is larger than the M4 Pro's 30.15 GB maximum, which is the reporter's failure; after the fix no attention buffer exceeds 256 MiB. Most of the remaining peak is a separate full-sequence transient (logits are the suspect) tracked in another issue and not touched here. The server rows reproduce the reference figures from the #3034 review (63.8 GB and 18.7 s on the 4B, 109.7 GB and 85.5 s on the 27B), so the server's 28K prefill is this same single-pass array-mask call.

Op-level, the same call at `[1, 24, 28000, 256]` fp16 with the model's additive causal mask, measured as MLX peak memory (`op-probe.txt`):

| budget | single call | gated | single call time | gated time | max abs diff |
|---|---|---|---|---|---|
| 256 MiB (Metal default) | 43.48 GB | 9.84 GB | 2084 ms | 1409 ms | 0 |
| 512 MiB | 43.48 GB | 15.30 GB | 2072 ms | 1279 ms | 0 |
| 1024 MiB (CUDA default) | 43.48 GB | 25.98 GB | 2206 ms | 1337 ms | 0 |
| 2048 MiB | 43.48 GB | 43.35 GB | 2079 ms | 1390 ms | 0 |

The live transient is a multiple of the budget because MLX keeps up to ten command buffers in flight (`MAX_ACTIVE_TASKS`) and frees a chunk's scores only when its buffer completes. That is why Metal defaults to 256 MiB rather than CUDA's 1024: at 1024 MiB the 28K prefill peaks were 60.85 GB (4B) and 93.93 GB (27B), against 44.38 GB and 78.48 GB at 256 (`cli-budget1024.csv`).

### 2. Fused calls stay fused

| run | gated calls | peak before | peak after | prefill before | prefill after | output |
|---|---|---|---|---|---|---|
| gemma-4-12b-it-4bit, 28710 tokens, single pass | 0 | 23.57 GB | 23.56 GB | 33.65 s | 33.56 s | identical |
| op probe, causal mode `[1, 24, 28000, 256]` | 0 | 3.94 GB | 3.94 GB + 0.34 GB output held | 498 ms | 484 ms | max abs diff 0 |

Gemma 4's sliding layers take MLX's native window kernel and its head_dim 512 global layers take the blocked D512 kernel, so `use_fallback` says "fused" for every call and the gate never fires. In MLX's causal mode at head_dim 256 over `q_len == k_len >= 2048` the kernel is fused on this GPU, and the gate leaves it alone (the review of #3034 found that probing the array-mask arm for every call would have chunked exactly this call).

The Qwen models have no fused prefill attention call on this path (see section 1), so their "fused path unchanged" check is not available; their default chunked prefill is reported next.

### 2b. Default chunked prefill (`MLXCEL_PREFILL_CHUNK=2048`)

Later prefill chunks attend a `[1, heads, 2048, k_len]` array mask over the growing cache; those are unfused too, and over budget once `k_len` passes about 4K (4B) or 2.7K (27B).

| model | gated calls | peak before | peak after | prefill before | prefill after | output |
|---|---|---|---|---|---|---|
| qwen3.5-4b-4bit | 96 | 10.97 GB | 11.00 GB | 15.01 s | 15.53 s (+3.5%) | identical |
| qwen3.8-27b-4bit | 208 | 27.80 GB | 27.83 GB | 77.06 s | 78.52 s (+1.9%) | identical |

Peak is unchanged because these calls were already small (at most 2.75 GB) and another transient sets the peak; the extra chunks cost 2 to 3.5% of prefill. At a 1024 MiB budget the same cells were within 0.5%.

### 3. Short prompt, chunking forced (`MLXCEL_ATTENTION_CHUNK_BUDGET_MB=16`)

| model | prompt tokens | gated calls | 48 greedy tokens |
|---|---|---|---|
| qwen3.5-4b-4bit | 4548 | 24 | identical |
| gemma-4-12b-it-4bit | 4551 | 88 | identical |

The Gemma run includes 8 calls in causal mode at head_dim 512 (`q=[1, 16, 455, 512] k=[1, 1, 4551, 512] do_causal=true`): the last, 455-row prefill chunk, which MLX's blocked D512 kernel does not take below 1024 query rows, so the per-mode probe correctly reports a fallback there while the 2048-row chunks stay fused.

## M5+ results: pending

The gate asks MLX at runtime and MLX's fused support differs by generation (on M5 and later an array-mask call at head_dim 256 fuses, #2646), so the same commands must be rerun on an M5-class GPU. Expected there: the Qwen single-pass array-mask calls are fused and the gate does not fire (unchanged peak and TTFT), and any call MLX still sends to the fallback is chunked. Results from the M5 Max run will be added here.

## How to rerun

```bash
cargo build --release --bin mlxcel
pgrep -x mlxcel || pgrep -x mlxcel-server   # must print nothing

# CLI matrix (sections 1, 2, 2b, 3); writes summary.csv plus per-run files.
MODELS_DIR=/path/to/models \
  scripts/bench/metal_sdpa_chunk/verify.sh /tmp/verify-3033

# Server path (section 1, server rows), port 8110.
MODELS_DIR=/path/to/models \
  scripts/bench/metal_sdpa_chunk/server_verify.sh /tmp/verify-3033 qwen3.5-4b-4bit 28000

# Op-level probe (section 1 and 2 tables); vary the budget per run.
MLXCEL_ATTENTION_CHUNK_BUDGET_MB=256 MLXCEL_SDPA_PROBE_LEN=28000 RUST_MIN_STACK=16777216 \
  cargo test --profile test-fast -p mlxcel-core --lib metal_chunk_tests::sdpa_chunk_peak_memory_probe -- --ignored --nocapture
```

- Prompt: `scripts/bench/metal_sdpa_chunk/make_prompt.py <tokens>` prints a deterministic ledger of numbered records plus a lookup question (about 27 tokens per record on the Qwen 3.5 tokenizer; the CLI prints the real count).
- Peak memory: `/usr/bin/time -l`, "peak memory footprint" of the `mlxcel` process. Prefill time: `mlxcel generate --profile`. Server TTFT: `scripts/bench/metal_sdpa_chunk/server_ttft.py`, request start to the first streamed delta.
- Baseline: `MLXCEL_METAL_ATTENTION_CHUNK=0`. Unfused case: needs no forcing on pre-M5 GPUs (the Qwen 3.5 array-mask call); `MLXCEL_ATTENTION_CHUNK_BUDGET_MB=16` with the default 2048-token prefill chunk forces chunking on short prompts.
- Gated calls: `RUST_LOG=mlxcel_core::layers=debug` prints one line per chunked call with its shapes and mask mode; `verify.sh` counts them. The server's log filter does not surface these lines, so the server rows read the effect from peak memory.
- Identity: the generated text (reasoning and content) of the two runs of a cell compared byte for byte.

## Raw data

In `metal-sdpa-query-chunk-2026-11-14/`:

- `cli-budget256.csv`: the CLI cells on the final build (Metal default 256 MiB), as written by `verify.sh`.
- `cli-budget1024.csv`: the same cells with the earlier 1024 MiB Metal default, plus the forced 4.6K cells of section 3 (their 16 MiB budget does not depend on the default).
- `server-budget256.csv`: the server rows (TTFT, total time, peak footprint).
- `gated-calls.csv`: every call the Metal gate chunked, by cell, with its q/k shapes, mask mode and chunk length, from the debug log.
- `completions.txt`: the greedy completions; each matched its `MLXCEL_METAL_ATTENTION_CHUNK=0` run byte for byte.
- `op-probe.txt`: the op-level probe output at four budgets.

The harness lives in `scripts/bench/metal_sdpa_chunk/` (`verify.sh`, `server_verify.sh`, `server_ttft.py`, `make_prompt.py`).
