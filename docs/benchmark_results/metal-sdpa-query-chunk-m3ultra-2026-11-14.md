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

## M5+ results (M5 Max)

The gate asks MLX at runtime and MLX's fused support differs by generation (on M5 and later an array-mask call at head_dim 256 fuses, #2646), so the same harness was rerun on an M5-class GPU.

Host: Apple M5 Max, 40-core GPU, 128 GB, macOS 27.2.1 (26B131), NAX present. MLX pin `9f806b3606eaa9f7c807238f6bd465185cd94d6d`. Branch `fix/issue-3033-metal-sdpa-chunk` at `8a5405f8`, release build, same checkpoints and prompts as above. Greedy, 48 tokens, one run at a time on an idle GPU.

On M5 the gate fired only on calls MLX still sends to the fallback, no fused call was chunked, and every cell's output matched its baseline byte for byte. The M3 Ultra figures above do not carry over, because on this GPU most of those calls are fused.

### M5.1 The reporter's call is fused here

Every 28K CLI cell, single pass and 2048-token chunked prefill, logged 0 gated calls, and peak footprint matched between the two switches to within 3 MB. The server's log filter does not show the gate's lines, so its row rests on the unchanged peak.

| cell | gated calls (M3 Ultra) | gated calls (M5 Max) | peak before | peak after | output |
|---|---|---|---|---|---|
| qwen3.5-4b-4bit, 28707 tokens, single pass | 8 | 0 | 39.61 GB | 39.61 GB | identical |
| qwen3.8-27b-4bit, 28749 tokens, single pass | 16 | 0 | 73.78 GB | 73.78 GB | identical |
| gemma-4-12b-it-4bit, 28710 tokens, single pass | 0 | 0 | 27.38 GB | 27.38 GB | identical |
| qwen3.5-4b-4bit, 2048-token chunks | 96 | 0 | 8.85 GB | 8.85 GB | identical |
| qwen3.8-27b-4bit, 2048-token chunks | 208 | 0 | 24.74 GB | 24.74 GB | identical |
| gemma-4-12b-it-4bit, 2048-token chunks | 0 | 0 | 14.17 GB | 14.17 GB | identical |
| qwen3.5-4b-4bit server, `--parallel 1` | not logged | not logged | 39.78 GB | 39.78 GB | identical |

The single-pass baseline peaks (39.61 GB and 73.78 GB) are already below the M3 Ultra's post-fix figures (44.38 GB and 78.48 GB), because MLX's fused array-mask kernel never materializes the scores here. The later 2048-row prefill chunks over the growing cache, unfused on the M3 Ultra (section 2b), are fused on M5 as well.

Prefill time needed a control. `verify.sh` runs the baseline then the fix back to back with no pause, and in that matrix the fix's single-pass prefill read slower (27B 73.55 to 80.27 s, Gemma 23.88 to 25.25 s; server TTFT 10.18 to 11.58 s) although the gate fired 0 times. Repeating those cells in the opposite order (fix first) with a 90 s pause before every run, twice:

| cell | fix (`=1`) | baseline (`=0`) |
|---|---|---|
| qwen3.8-27b-4bit single pass, prefill | 53.41 s, 53.55 s | 53.52 s, 54.43 s |
| gemma-4-12b-it-4bit single pass, prefill | 20.59 s, 20.15 s | 20.47 s, 20.40 s |
| qwen3.5-4b-4bit single pass, prefill | 10.11 s, 10.29 s | 10.16 s, 10.18 s |
| qwen3.5-4b-4bit server, TTFT | 10.17 s, 10.16 s | 10.23 s, 10.05 s |

With cooling, the four runs of each cell fall within 2.2% of one another and neither arm is consistently ahead. The matrix itself ran slower than the cooled runs (27B baseline 73.55 s against 53.5 s, fix 80.27 s), so it was measured on a throttled GPU, and its second arm more so. For a fused call, the only work the gate adds is one memoized `use_fallback` query per distinct shape.

### M5.2 Calls MLX still sends to the fallback are chunked

Forced short prompts (`MLXCEL_ATTENTION_CHUNK_BUDGET_MB=16`, 2048-token prefill chunks):

| model | prompt tokens | gated calls (M3 Ultra) | gated calls (M5 Max) | what was chunked | 48 greedy tokens |
|---|---|---|---|---|---|
| qwen3.5-4b-4bit | 4548 | 24 | 8 | `q=[1, 16, 452, 256] k=[1, 4, 4548, 256]`, array mask | identical |
| gemma-4-12b-it-4bit | 4551 | 88 | 48 | 40 x `q=[1, 16, 455, 256] k=[1, 8, 1478, 256]`, array mask; 8 x `q=[1, 16, 455, 512] k=[1, 1, 4551, 512]`, causal mode | identical |

On M5 the two 2048-row chunks of each prompt are fused and only the last 452 or 455-row chunk falls back, so the gate fires on that chunk alone. The Gemma head_dim 512 causal calls below 1024 query rows fall back on both generations. At this forced 16 MiB budget Gemma's prefill rose from 2.53 to 3.89 s (Qwen went from 1.77 to 1.49 s); this measures 16 MiB chunks, not the 256 MiB default.

### M5.3 Op level, and the 256 MiB default on M5

The probe's default shape, `[1, 24, 28000, 256]` fp16 over 4 KV heads, is fused on M5 in both array-mask and causal mode (`mlx_fused=true`), so at 256, 1024 and 16 MiB the gate passed both calls through unchanged: max abs diff 0, single call 5.51 GB (array mask) and 3.94 GB (causal), the gated run 0.34 GB higher because it still holds the single call's output.

To measure the budget on a call M5 does not fuse, the probe was run at head_dim 512 with an array mask (`MLXCEL_SDPA_PROBE_DIM=512 MLXCEL_SDPA_PROBE_KV_HEADS=1 MLXCEL_SDPA_PROBE_HEADS=16 MLXCEL_SDPA_PROBE_LEN=20000`, Gemma 4's global-layer shape with an explicit mask), three repetitions per budget:

| budget | single call | gated | single call time | gated time | max abs diff |
|---|---|---|---|---|---|
| 256 MiB (Metal default) | 16.22 GB | 8.22 to 8.23 GB | 581 to 590 ms | 407 to 416 ms | 3.05e-5 |
| 512 MiB | 16.22 GB | 13.84 GB | 584 to 598 ms | 456 to 457 ms | 0 |
| 1024 MiB (CUDA default) | 16.22 GB | 16.20 GB | 580 to 587 ms | 493 to 499 ms | 0 |
| 2048 MiB (one run) | 16.22 GB | 16.22 GB | 601 ms | 527 ms | 0 |

256 MiB is still the right Metal default on M5: it halves the peak and is the fastest setting, about 30% faster than the single call. At 1024 MiB the peak barely moves, for the same reason as on the M3 Ultra (about ten chunks are live at once). Unlike the M3 Ultra, the 256 MiB run is not bit-exact: its max abs diff is 3.05e-5 on every repetition, a 1 to 2 ulp change in fp16 outputs of this magnitude, while 512 MiB and above give 0. A 256 MiB chunk here is 419 query rows. A plausible cause is a different matmul kernel for that operand height on this GPU, which would change the reduction order; that was not traced. Every model-level cell above still produced byte-identical text.

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

# On M5 and later the default probe shape is fused; this one is not (section M5.3).
MLXCEL_SDPA_PROBE_DIM=512 MLXCEL_SDPA_PROBE_KV_HEADS=1 MLXCEL_SDPA_PROBE_HEADS=16 MLXCEL_SDPA_PROBE_LEN=20000 \
  MLXCEL_ATTENTION_CHUNK_BUDGET_MB=256 RUST_MIN_STACK=16777216 \
  cargo test --profile test-fast -p mlxcel-core --lib metal_chunk_tests::sdpa_chunk_peak_memory_probe -- --ignored --nocapture
```

`verify.sh` and `server_verify.sh` run the baseline and the fix back to back, so on a GPU that throttles (the M5 Max does) their prefill and TTFT columns compare a cool arm with a warm one. Read time from an order-reversed run with a pause before each run, as in section M5.1.

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
- `m5max-cli-budget256.csv`: the M5 Max CLI matrix as written by `verify.sh`.
- `m5max-cli-control.csv`: the order-reversed, cooled single-pass prefill repeats (section M5.1).
- `m5max-server-budget256.csv`: the M5 Max server runs, matrix and cooled control.
- `m5max-gated-calls.csv`: every call the gate chunked on the M5 Max (only the forced 4.6K cells).
- `m5max-completions.txt`: the M5 Max greedy completions of the fix runs.
- `m5max-op-probe.txt`: the M5 Max op-level probe, default shape and the head_dim 512 array-mask shape.

The harness lives in `scripts/bench/metal_sdpa_chunk/` (`verify.sh`, `server_verify.sh`, `server_ttft.py`, `make_prompt.py`).
