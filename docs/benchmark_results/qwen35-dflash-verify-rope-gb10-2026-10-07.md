# Qwen 3.5 DFlash on CUDA: the verify block's RoPE kernel, and the fix (GB10, 2026-10-07)

Issue #2191, continuing #1935 (`dflash-width-2-4-residual-qwen35-gb10-2026-09-21.md`). Since PR #1944 the exactness probe declined Qwen 3.5 DFlash outright on CUDA for head_dim 256/288 while `MLXCEL_SDPA_VECTOR_LARGE_D` was on, because the served width 2 and 4 completions on `qwen3.5-4b-4bit` + `qwen3.5-4b-dflash` parted from classic decode and the probe could not see it. PR #2185 found that Gemma 4's equivalent CUDA failure included MLX's RoPE kernel choice. This record tests that hypothesis for Qwen 3.5. It holds: the residual is the verify block's RoPE, not the attention kernel.

## Host

```
host: spark-101, NVIDIA GB10 (sm_121), MLX_CUDA_ARCHITECTURES=121
kernel: 7.0.0-1019-nvidia, driver 580.178.04, CUDA 13.0
mlx_pin: 81ba1c6a0e50a9268b931579c2d4f1158b9aab5a (same pin as the 2026-09-21 record)
rustc: 1.97.1, release profile, --features cuda, base origin/main bbd05099
checkpoints: models/mlx/qwen3.5-4b-4bit (head_dim 256, 16 query heads, 4 KV heads,
  partial_rotary_factor 0.25 so 64 rotated dims, rope_theta 1e7), models/mlx/qwen3.5-4b-dflash
```

Raw data and harnesses: `data/qwen35-dflash-verify-rope-gb10-2026-10-07/`.

## Step 1.1: which RoPE branch runs

A temporary log line in `forward_hidden_with_position_ids_verify` (not shipped), one classic server and one width-4 DFlash server, text-only `/v1/completions`. Every full-attention forward in both took the `fast_rope` branch: classic 8 prefill calls at `l=4` and 96 decode calls at `l=1`; DFlash the same prefill, 32 verify calls at `l=4` and 8 at `l=2`, all `fast_rope`. No call took the interleaved-MRoPE branch, which agrees with the code: `rope_deltas` is set only by the image path (`Qwen35VLModel::get_input_embeddings`), and `forward_speculative` passes `position_ids = None`.

## Step 1.2: the op-level difference

MLX's CUDA `RoPE::eval_gpu` (`mlx/backend/cuda/rope.cu:317`) dispatches `rope_single` when the input is row-contiguous with `B == 1 && T == 1`, and the general `rope` kernel otherwise. The two compute the same expressions as written, so any difference is code generation. `op_level_fast_rope_block_versus_decode_rows` rotates a `[1, H, L, 256]` bf16 block built as a transpose of `[1, L, H, 256]` (the model's layout) at `dims 64, base 1e7`, against `L` decode-shaped `L = 1` calls at `offset + row`, over H in {16, 4}, L in {2, 4}, nine offsets including 194 and 4093, and four draws (144 cases).

| comparison | differing bytes over 144 cases |
|---|---:|
| block call vs decode rows | 1 (H=16, L=4, offset 37, draw 1) |
| per-row calls (`fast_rope_rows_like_decode`) vs decode rows | 0 |

One byte in 144 cases is rare at op level, but the model runs 8 full-attention layers times 20 heads times 32 rotated pairs for both Q and K per row, so over a 200-token transcript a flipped element is close to certain, and a flipped K element stays in the cache. The dense MLP cell (`op_level_compiled_silu_block_versus_rows`, `compiled_silu(gate) * up` at `[1, L, 9216]`, L in {2, 4}, four draws) is byte-identical: 0 bytes.

## Step 1.3: the first differing sub-op

`post_rope_capture_bisect_on_the_real_transcript` replays the served width-4 round structure (`accepts_w4.txt`) on the recorded 158-token prompt and 200 classic ids and captures, per full-attention layer, post-RoPE Q, post-RoPE K, the attention output, `o_proj` and the MLP output, in both the block and the chain, then names the first differing sub-op per kept row.

With block RoPE, over all 200 kept rows: the first differing sub-op is **post-RoPE Q in 68 rows, post-RoPE K in 1, and the attention output in 4**; never `o_proj` or the MLP first. The first differences are single bytes in a single layer (position 198: one byte of post-RoPE Q in full-attention layers 6 and 7; position 261: one byte of post-RoPE K in layer 0), with the logits still equal. The attention-first rows come after a K difference has already entered the cache (the first at position 274, after the K difference at 261). From position 291 the logits differ (236123 of 496640 bytes), and every later row differs from the first full-attention layer on.

With per-row RoPE, the same arm reports zero differing sub-ops and zero differing logit rows.

The capture itself evaluates each captured tensor, which changes MLX's graph boundaries, so the capture arm is used only for "which sub-op first"; the matrix below runs without it.

## Step 1.4: the attribution matrix (logit bytes, n = 3)

`block_versus_chain_byte_bisect_on_the_real_transcript`, uncaptured, on the served width-2 and width-4 accept sequences, every cell three times in separate processes. Kept rows whose logit bytes differ, of 200:

| width | per-row RoPE | `LARGE_D` | repeats | differing rows | first differing row |
|---:|:---:|:---:|:---:|---:|---|
| 2 | off | 1 | 67, 67, 67 | 67 | absolute 291 (emitted 134) |
| 2 | off | 0 | 129, 129, 129 | 129 | absolute 229 (emitted 72) |
| 2 | on | 1 | 0, 0, 0 | 0 | none |
| 2 | on | 0 | 0, 0, 0 | 0 | none |
| 4 | off | 1 | 67, 67, 67 | 67 | absolute 291 |
| 4 | off | 0 | 129, 129, 129 | 129 | absolute 229 |
| 4 | on | 1 | 0, 0, 0 | 0 | none |
| 4 | on | 0 | 0, 0, 0 | 0 | none |

Every repeat is byte-for-byte the same count and the same first row, so the matrix is deterministic, not sampled.

Decision rule (issue #2191): post-RoPE K bytes differ between block and decode, and the per-row RoPE cell with `LARGE_D=1` makes the bisect report zero differing rows. **RoPE is the cause.** The fix is step 2.

Two findings correct the earlier record:

- **`MLXCEL_SDPA_VECTOR_LARGE_D=0` does not remove the divergence; it moves it.** With block RoPE and `LARGE_D=0`, 129 of 200 rows still differ in logit bytes, more than with the fused kernel. The #1944 attribution rested on a served sha256 of the text and an argmax count, both of which a sub-reporting-floor difference can pass. The kill switch changes which rows round across an argmax, not whether the block matches decode.
- The uncaptured bisect's first differing row on this base is at absolute position 291, not the 194 the 2026-09-21 record reported for the same MLX pin. The transcript and accept sequence are the same; the codebase between the two records is not.

## Step 2: the fix

- `Qwen3NextAttention` gains `verify_rope_rows`, set once at construction from `cuda_is_available()`. In `forward_hidden_with_position_ids_verify`, on the `fast_rope` branch, when `target_verify && l > 1 && b == 1 && verify_rope_rows`, Q and K are rotated one row at a time (`fast_rope_rows_like_decode`: slice row `r` on axis 2, `fast_rope(..., offset + r)`, `concatenate_many` on axis 2), which is the `L = 1` call classic decode makes. The block-rotated K therefore never enters the KV cache. `B > 1` verify blocks keep the block call: batched classic decode does not take `rope_single` either. Metal keeps the block call.
- `Qwen35Model::cuda_sdpa_vector_verify_hazard` and its `NotRun` decline are removed.
- `probe_block_chain_exactness` on CUDA adds a long draw (`probe_long_walk`): after the three short draws it walks 128 synthetic tokens as full-accept verify blocks and as single-token steps from the same prefill, comparing every block row against its chain step in logit bytes, mirroring Gemma 4's long buffered draw. Every CUDA probe draw now prefills 160 tokens (`CUDA_PROBE_PROMPT_LEN`) instead of 8; Metal keeps 8 and does not run the walk.

`long_probe_draw_sees_the_verify_rope_hazard_and_passes_with_the_fix`, three runs (`capture/long-probe-r*.log`): with the per-row RoPE reverted the long draw returns `Diverges` at walk position 19 (201114 of 496640 logit bytes) at both widths 2 and 4, and with it on returns `Equal` at both. The probe as shipped before this change returned `Equal` at widths 2 and 4 at every prompt length from 8 to 512 while the served path diverged.

Widths 8 and 16 still decline through the short draws' measured verdict at position 0 (`M * B >= 8` moves MLX from `qmv` to `qmm_sm80`), unchanged.

### What the per-row RoPE costs

`verify_block_cost_with_and_without_row_rope` times 64 verify blocks per arm from a 160-token prefill, each rolled back to one accepted row, with the per-row RoPE off and on, interleaved over three repeats (`capture/verify_cost.log`):

| width | block RoPE, ms per verify | per-row RoPE, ms per verify | delta |
|---:|---|---|---|
| 2 | 20.85, 20.54, 20.61 | 20.63, 20.71, 20.75 | within the off arm's spread |
| 4 | 30.70, 30.80, 30.83 | 31.19, 31.18, 31.20 | +0.38 to +0.49 ms (+1.2% to +1.6%) |

### The probe's CUDA graph-cache cost

MLX's CUDA graph cache keeps a lifetime miss counter that aborts the process past `2 * MLX_CUDA_GRAPH_CACHE_SIZE` (#818; `mlxcel-server` sets 2000, so 4000 misses). The KV lengths a probe visits can each capture new graph topologies, so the long draw is not free in that budget. Measured by bisecting `MLX_CUDA_GRAPH_CACHE_SIZE` until a width-4 server's startup (model load plus probe) no longer aborts:

| probe configuration | startup aborts at cap | startup survives at cap |
|---|---:|---:|
| short draws only (width 8, which never reaches the walk), 160-token prompt | | 100 |
| short draws at 8 tokens, walk from an 8-token prompt | 300 | 500 |
| short draws at 8 tokens, walk from a 160-token prompt | 400 | (not reached) |
| every draw at 160 tokens (shipped) | 150 | 200 |

Under LRU eviction a small cap re-misses, so these are bounds on the probe's working set rather than exact lifetime counts; the shipped configuration fits in a 200-entry cache, so at the server's 2000 capacity it draws at most a few hundred misses of the 4000 budget, once per process. Walking from an 8-token prompt needed more than twice that, which is why the CUDA probe prompt moved to 160. Classic decode of one 400-token request survives at cap 25.

## Served checks (no environment overrides)

`served_identity.py`, three `/v1/completions` prompts at temperature 0, 200 tokens each (146, 200 and 200 completion tokens), one server per arm:

| arm | sha256 of the three completions | bursts | declined |
|---|---|---:|:---:|
| classic | `0eef87cdf198`, `7f530421e508`, `fffb026a234e` | 0 | |
| width 2 | identical | 4 (warm-up + 3) | no |
| width 4 | identical | 4 | no |
| width 8 | identical | 0 | yes, probe `Diverges` at position 0 |
| width 16 | identical | 0 | yes, probe `Diverges` at position 0 |
| unset (resolves 4, "measured default for sm_121 with an affine-quantized target") | identical | 4 | no |

The probe logs `MTP exactness probe passed: verify block is byte-identical to the single-token chain block_size=4` at widths 2, 4 and unset.

`greedy_parity_dflash_qwen35_4b` (`parity/parity-dflash.log`): `widths that ran the burst: [2, 4]; widths the gate declined: [8, 16]`, every response equal to the drafter-less baseline, `test result: ok`. The full `speculative_parity --ignored` suite (`parity/parity-full.log`): 7 passed, 1 failed, the failure being `b1_batched_baseline_probe` ("B=1 batched baseline drift"), a known Gemma 4 failure on `origin/main` that #2190 addresses.

## Throughput at the width main resolves (4)

`price_rope.py`: per round, classic-a, DFlash with no `--draft-block-size` (resolves 4), and classic-b as a classic-vs-classic null arm, arm order rotated per round, one server per arm, the #1797 harness's 158-token prompt, a discarded warm-up then three streamed 200-token requests, `--ignore-eos --max-batch-size 1`. Every request in every arm returned the same completion (`2c76b0a181`, the classic sha the 2026-09-21 record names).

| round | classic-a tok/s | DFlash tok/s | classic-b tok/s | DFlash vs classic-a | DFlash vs classic-b | null arm (b vs a) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 57.05 | 61.12 | 56.72 | +7.1% | +7.7% | -0.6% |
| 1 | 55.03 | 60.90 | 57.05 | +10.7% | +6.7% | +3.7% |
| 2 | 56.87 | 61.49 | 57.25 | +8.1% | +7.4% | +0.7% |

Every DFlash paired delta (+6.7% to +10.7%) is above the null arm's spread (-0.6% to +3.7%), so the gain is resolved: about 1.08x over classic. Acceptance is 0.455 and 2.34 tokens emitted per verify in every DFlash arm.

That is less than the 1.18x the 2026-09-21 record measured for width 4 with `MLXCEL_MTP_ALLOW_INEXACT=1`. The two are not the same run: that burst followed a different, non-identical trajectory (`3e60b1574c`, acceptance 0.511), and the per-row RoPE adds 1.2% to 1.6% to a width-4 verify (above). This record does not separate the two further.

Host gate: rounds 0 and 1 waited for the CPU-quiet gate including the CI-job check. The shared runner then stayed busy with other units' queued CI for over two hours, and the first attempt at round 2 was cut off by a harness timeout after its classic-b arm (kept as `throughput/interrupted_*`). Round 2 was rerun with `--ignore-ci-gate`, which still gates on compiler and foreign-model processes; its null arm (+0.7%) is inside the other rounds' spread.

## Reproduction

```
cargo test --release --features cuda -p mlxcel --lib --no-run
MLX_CUDA_GRAPH_CACHE_SIZE=2000 gpu-lock run --tag dflash-qwen35 -- \
  target/release/deps/mlxcel-<hash> --ignored --test-threads=1 --nocapture \
  models::qwen3_5::qwen3_5_rope_attribution_tests
BIN=target/release/deps/mlxcel-<hash> \
  docs/benchmark_results/data/qwen35-dflash-verify-rope-gb10-2026-10-07/harness/matrix.sh
```

The test binary does not apply the server's `MLX_CUDA_GRAPH_CACHE_SIZE=2000` default, and the probe tests abort with MLX's "Cache thrashing" error at its own default of 400. The matrix and capture arms ran on an intermediate build in which the per-row switch was an environment read in the runtime; on the shipped code `MLXCEL_Q35_ROW_ROPE` is read by the tests and applied through `set_verify_rope_rows_for_test`.
