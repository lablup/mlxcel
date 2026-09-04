# MoE batched decode attribution on M1 Ultra (#1616)

Mac Studio M1 Ultra 128 GB (Apple GPU generation 13), macOS 26.6.2, mlxcel 0.7.0-beta.1, MLX pin `9a795735`. Before-numbers are from commit `bf1cdb72` (main), after-numbers from the #1616 code PR built on top of it. Serving ladder: `scripts/bench_serving_concurrency.py --prompt-tokens 512 --max-tokens 128 --concurrency 1,2,4` against one fresh `mlxcel-server -m models/mlx/<name> --parallel 4 --max-batch-prefill 4` per model, levels ascending, Time Machine stopped. The issue was filed from M5 Max numbers; this host was substituted with the user's approval, so every figure below is this machine's own baseline and none of the M5 Max values (169.6, 65.5, 1.55x, 3.25x, 3.17x) is used as a target.

## 1. Result in one paragraph

The issue's mechanism is not what happens. `qwen3_moe` (and every one of the 16 MoE families that carry the `array_shape(&x_flat)[0] == 1` gate) has no `forward_batched` override, so the `LanguageModel` trait default runs the single-sequence `forward` once per row and evaluates the B graphs together. At B=4 a decode tick is therefore four serialized single-token graphs, each of which passes the gate and launches the fused MoE kernel; `gather_qmm` is never reached. The aggregate ceiling comes from those four graphs overlapping only about 30% on this GPU (tick 12.3 ms at B=1, 33.6 ms at B=4). Expert-plane traffic is not the limiter: at the op level a cohort routing to identical experts costs the same as one routing to disjoint experts at every batch size, so cross-cohort deduplication would buy nothing here. The batched fused kernel the issue proposes was prototyped at the op level from the same Metal source (grid z of `k * n_tokens`, per-token sum), is bit-identical to per-token launches, and loses to `gather_qmm` from n=4 (11.1 ms vs 9.3 to 9.7 ms per 48 layers), so it was not built. What the profile does justify is giving `Qwen3MoeModel` a real batched forward (Rust only, mirroring the Qwen3 dense port) so the tick becomes one graph with the experts on `gather_qmm` over `B * top_k` slots; that is the change in the code PR. TTFT growth at B=4 is a prompt-cache effect, not decode: dense prompt-cache entries are one-shot, so only as many rows adopt as entries were donated by the previous level and the remaining rows pay a cold 440-token prefill serialized ahead of the first decode tick.

## 2. Before ladder (main at `bf1cdb72`)

Source: `benchmarks/metal_m1ultra_batch_2026-09-04.csv`. A second full pass (`before2`) reproduced every B=4 aggregate within 0.3% (93.3 vs 93.5, 120.5 vs 120.5, 367.5 vs 367.5); the B=2 cells of the 0.5B model move by 15% between passes and are the only noisy cells.

| model | B | TTFT mean (ms) | TTFT p95 (ms) | decode tok/s per request | aggregate tok/s | scaling |
|-------|---|---------------:|--------------:|-------------------------:|----------------:|--------:|
| qwen2.5-0.5b-bf16 | 1 | 90.5 | 90.5 | 253.2 | 216.2 | 1.00x |
| qwen2.5-0.5b-bf16 | 2 | 25.0 | 31.5 | 87.9 | 174.2 | 0.81x |
| qwen2.5-0.5b-bf16 | 4 | 39.8 | 59.8 | 93.9 | 367.5 | **1.70x** |
| llama-3.1-8b-4bit | 1 | 773.7 | 773.7 | 86.8 | 57.2 | 1.00x |
| llama-3.1-8b-4bit | 2 | 100.2 | 132.5 | 54.7 | 105.6 | 1.85x |
| llama-3.1-8b-4bit | 4 | 167.6 | 264.3 | 31.1 | 120.5 | **2.11x** |
| qwen3-30b-a3b-4bit | 1 | 799.3 | 799.3 | 81.5 | 54.3 | 1.00x |
| qwen3-30b-a3b-4bit | 2 | 611.2 | 611.4 | 55.2 | 88.0 | 1.62x |
| qwen3-30b-a3b-4bit | 4 | 1223.8 | 1224.2 | 29.8 | 93.3 | **1.72x** |

This host does not reproduce the M5 Max shape of the problem. Dense llama-3.1-8b scales 2.11x here (3.25x on M5 Max) and the MoE model 1.72x (1.55x there), so the dense-versus-MoE gap is 0.39x of scaling rather than 1.7x. The `qwen2.5-0.5b-bf16` row is not a scheduler control on this host either: `qwen2` has no `forward_batched` override, so it runs the same per-row fallback as the MoE model and its 1.70x is that fallback's ceiling for a dispatch-bound graph.

## 3. Stage 1: attribution

### 3.1 Which path runs at B=4 (premise check)

`grep -rn "fn forward_batched" src/models/` finds overrides in `llama3`, `llama4`, `qwen3`, `qwen3_5`, `gemma3`, `helium` and `muse_glimmer` only. None of the MoE families has one, so `execute_batched_decode` (`src/server/batch/scheduler/decode_tick.rs`) builds its `[b, 1]` input and the trait default in `src/lib/mlxcel-core/src/generate.rs` slices it back into `b` calls of `forward` on `[1, 1]`. Each call flattens to `x_flat = [1, hidden]`, passes the `== 1` gate in `qwen3_moe.rs`, and launches the fused two-kernel path.

Verified on the server rather than asserted: with `MLXCEL_PROFILE_BLOCKS=1` the model-level hook prints one `[QWEN3_MOE_BLOCKS]` summary per single-token forward. The B=1 level produced 128 summaries for 128 decode ticks and the B=4 level produced 518 for 128 ticks, four per tick (the extra six are the ladder's warm-up request). The per-forward split is the same at both levels (median attention 19.6 ms, MoE 60.2 to 60.6 ms, 25:75, device-synchronizing numbers), which is what four independent copies of the B=1 graph look like.

The pre-existing `MLXCEL_PROFILE_QWEN3_MOE_DETAIL=1` hook did not report which expert path ran and always executed the gather path, so it attributed a kernel the production step never launched at B=1. The code PR makes it follow the production dispatch and print `path=fused|gather_qmm tokens=N`.

### 3.2 Tick cost at production speed

From the ladder (per-request decode tok/s is the tick rate under continuous batching):

| arm | B=1 tick (ms) | B=2 tick (ms) | B=4 tick (ms) | B=4 / B=1 |
|-----|--------------:|--------------:|--------------:|----------:|
| default (fused kernel per row) | 12.3 | 18.1 | 33.6 | 2.73x |
| `MLXCEL_FUSED_MOE=0` (gather_qmm per row) | 14.1 | 21.8 | 38.9 | 2.76x |

Four serialized single-token graphs cost 2.73x one graph, so the GPU overlaps them by roughly 30%; the rest is where the per-request rate goes (81.5 to 29.8 tok/s). The fused kernel is worth 13% of the tick at B=1 on this host with the pinned MLX (12.3 vs 14.1 ms), and the same per-row advantage persists at every B because the rows never share a graph.

### 3.3 Op-level split of the expert path (scratchpad microbench, Python MLX 0.32.2 wheel)

The real layer-0 expert planes of `qwen3-30b-a3b-4bit` (128 experts, Din 2048, Dff 768, 4-bit, group 64; 2.65 MB per expert slab, 0.34 GB per layer stack) were timed as a 48-layer-deep graph evaluated once, min of 3 rounds of 10 pipelined evals after a 2 s GPU warm-up, with expert ids rotating per layer so planes stream from DRAM. The fused kernels are `mx.fast.metal_kernel` builds of the verbatim `MOE_GATEUP_METAL_SOURCE` / `MOE_DOWN_METAL_SOURCE` strings from `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`; the batched prototype adds a `tok = eslot / K` row index and a grid z of `K * n`, which is exactly the design the issue proposes. Milliseconds per 48-layer chain:

| n_tokens | experts across rows | gather_qmm unsorted | gather_qmm sorted | fused, one launch pair per token (today's B>1 path) | fused batched prototype |
|---------:|---------------------|--------------------:|------------------:|----------------------------------------------------:|------------------------:|
| 1 | disjoint | 3.07 | 4.13 | 3.90 | 3.55 |
| 1 | same | 3.10 | 4.09 | 3.86 | 3.46 |
| 2 | disjoint | 5.31 | 5.75 | 7.45 | 6.03 |
| 2 | same | 5.29 | 5.71 | 7.27 | 6.05 |
| 4 | disjoint | 9.71 | 9.31 | 14.72 | 11.07 |
| 4 | same | 9.67 | 9.30 | 14.64 | 11.08 |
| 8 | disjoint | 18.72 | 17.10 | 29.17 | 20.71 |
| 8 | same | 18.68 | 17.00 | 29.64 | 20.59 |

Router + top-k over 48 layers is 1.1 ms at every n. Parity on one layer: fused vs gather_qmm normalized RMS 1.1e-3 (the #886 f32-partial discipline is in the ported source), sorted vs unsorted 0, and the batched prototype is bitwise equal to per-token launches at every n.

What the table says:

- **Per-token fused launches scale linearly** (3.9, 7.5, 14.7, 29.2 ms): 3.7 ms per token at every B. That is today's B>1 expert cost, and it is why the tick grows 2.7x.
- **Expert-plane traffic is not the limiter.** "same" (every row routes to the same 8 experts, so each plane is read once from DRAM and then from cache) equals "disjoint" (every slot reads its own plane) in every arm at every n. Deduplicating expert ids across the cohort would remove reads the GPU is not waiting on. The bandwidth numbers agree: 1.02 GB of planes in 3.55 ms is 290 GB/s at n=1 and 8.1 GB in 20.7 ms is 390 GB/s at n=8, both under the streaming bandwidth of this part, and the "same" arm would have to be faster if that were the wall.
- **The batched fused prototype loses to `gather_qmm` from n=4** (11.07 vs 9.31 to 9.71 ms) and stays 1.2x slower at n=8. The kernel spends one 32-lane simdgroup per output row with x re-read per row; MLX's gather `qmv` amortizes better once there are enough slots to fill the GPU. It only wins at n=1 (3.55 vs 3.07 with this wheel's MLX; on the pinned MLX the end-to-end B=1 arm above shows the fused path ahead by 13%, so the n=1 ordering is MLX-version dependent, the n>=4 ordering is not).
- **`do_sort` at 32 slots (B=4, top_k 8) is worth 4% of the expert path** (9.31 vs 9.71 ms), about 1% of a tick. Sorting costs 0.4 to 1 ms at n=1 and n=2 where the threshold already avoids it. The constant 64 is left alone; retuning it to 32 is a measured 1% and should wait for an M5 Max reading.

### 3.4 TTFT: prefill serialization behind one-shot prompt-cache entries, not tick length

`mlxcel-server -v` at the same ladder. The client prompt tokenizes to 440 tokens; a cold 440-token MoE prefill takes about 560 ms on this host.

| level | rows that adopted the prompt cache | rows that ran a cold prefill | TTFT mean / p95 (ms) |
|-------|-----------------------------------|------------------------------|---------------------:|
| B=1 | 0 (nothing to adopt) | 1 | 757.1 / 757.1 |
| B=2 | 1 (`adopted 432/440`) | 1 (`cached=0/440`) | 621.7 / 621.9 |
| B=4 | 2 | 2 | 1243.4 / 1243.6 |

Timeline of the B=4 level from the log: all four requests land at 03:58:03.019; seq-4 adopts and is admitted by 03:58:03.077, seq-5 adopts next, then the two cold prefills run back to back until 03:58:04.196 when the scheduler first reports `active=4`, and the first decode tick completes at 03:58:04.295. Every client's first content chunk arrives after that tick, hence four equal TTFTs of about 1.24 s: 1.12 s of serialized cold prefill plus one tick. Only one row per donated entry can adopt because dense (non-paged) entries are consumed on adoption (`take_detached`, the "legacy one-shot consume" in `src/server/batch/scheduler/prompt_cache.rs`), and the MoE family does not opt into the paged decode backend whose entries adopt by clone. The dense llama-3.1-8b run adopts on every row (`adopted 448/467` x4 at B=4) and its TTFT stays at 165 ms. So the issue's "prefill cohorts serialized behind MoE decode" and "longer decode tick" are both wrong for this host: it is cold prefills serialized ahead of decode, caused by prompt-cache adoption semantics. `prefill_cohort.rs` is not involved because `Qwen3MoeModel` does not opt into `supports_batched_prefill()`, so the window is always sequential. This is a scheduler/prompt-cache defect and is left as a follow-up rather than folded into this PR.

## 4. Stage 2: what the profile justified

In the issue's expected-payoff order:

1. **Batched fused kernel: declined, with the measurement above.** The prototype of the proposed design is slower than the `gather_qmm` chain a batched forward already gets for free at n>=4, and no cheaper than it at n=2. A kernel that could win would need a different structure (several rows per simdgroup with x staged in threadgroup memory), which is a new kernel rather than a token dimension on this one, and the same table sets its bar. The GeGLU variant is the same kernel body with a different activation, so it is declined on the same evidence.
2. **Cross-cohort expert deduplication: declined.** No arm is bandwidth-bound at B<=8 on this host, and "same" equals "disjoint" throughout.
3. **`do_sort` retune: left at 64**, a measured 1% of a tick at B=4.
4. **A real `forward_batched` for `Qwen3MoeModel`: built.** This is the change the profile points at: the batched embedding, norms, fused QKV and output projections, router and LM head run once per tick, attention runs per sequence against each row's cache (the Qwen3 dense loop, dense caches only), and the experts run through `gather_qmm` over `B * top_k` slots in one chain. Predicted from section 3.3: the expert path drops from 14.7 to 9.7 ms per tick at B=4. B=1 is untouched: a one-row batch delegates to the single-sequence `forward` and the scheduler never enters the batched path at B=1 anyway.

The rejected route of forcing `supports_batching()` false was not revisited; the measurement shows the serial fallback it would pick is exactly what B>1 was already running.

## 5. After ladder (code PR)

Re-measured on the final commit of the code PR. An earlier after-pass was taken from a binary built before the last edit to the branch and read 103.7 aggregate at B=4; a `cargo build --release` on the final commit recompiled, so that pass was discarded and the table below is the rebuilt binary. The conclusion is unchanged and the numbers move by about 1%.

| model | B | TTFT mean (ms) | TTFT p95 (ms) | decode tok/s per request | aggregate tok/s | scaling vs B=1 |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b-bf16 | 1 | 94.4 | 94.4 | 245.4 | 209.2 | 1.00x |
| qwen2.5-0.5b-bf16 | 2 | 25.1 | 31.6 | 102.7 | 202.8 | 0.97x |
| qwen2.5-0.5b-bf16 | 4 | 39.4 | 59.1 | 94.4 | 369.5 | **1.77x** |
| llama-3.1-8b-4bit | 1 | 766.3 | 766.3 | 85.3 | 56.8 | 1.00x |
| llama-3.1-8b-4bit | 2 | 98.4 | 130.2 | 54.9 | 106.2 | 1.87x |
| llama-3.1-8b-4bit | 4 | 166.5 | 265.2 | 30.7 | 118.9 | **2.09x** |
| qwen3-30b-a3b-4bit | 1 | 745.4 | 745.4 | 81.0 | 55.3 | 1.00x |
| qwen3-30b-a3b-4bit | 2 | 608.8 | 608.9 | 52.9 | 85.0 | 1.54x |
| qwen3-30b-a3b-4bit | 4 | 1232.6 | 1232.9 | 33.9 | 102.9 | **1.86x** |

The MoE model gains 10.3% aggregate at B=4 (93.3 to 102.9) and 13.8% per-request decode (29.8 to 33.9), moving scaling from 1.72x to 1.86x. Both dense controls stay inside the run-to-run band (qwen2.5-0.5b 367.5 to 369.5, llama-3.1-8b 120.5 to 118.9), which is what rules out a host-wide shift between the two passes. B=1 is unchanged by construction: the `forward_batched` override routes a one-row batch to the single-sequence `forward`, fused kernel included.

Greedy `mlxcel generate -p Hello -n 128 --temp 0` on `qwen3-30b-a3b-4bit` is byte-identical before and after on the final binary. Four simultaneous greedy 256-token requests at B=4 return identical rows with non-empty reasoning content.

## 6. Follow-ups this profile motivates, not done here

- The other 15 MoE families with the `== 1` gate still run the per-row fallback; each needs its own `forward_batched` after the same premise check, and families that share `switch_layers::SwitchGLU` get the multi-token `gather_qmm` path for free once their attention is batched.
- Prompt-cache adoption for dense-backed families is one-shot, so N simultaneous identical prompts pay N minus (entries) cold prefills; the paged backend's clone adoption is the fix pattern.
- Dense batched decode on this host (llama-3.1-8b at 2.11x, 31.1 tok/s per request at B=4 against 86.8 at B=1) is far from the M5 Max 3.25x; its per-sequence attention loop is the likely dispatch cost and deserves its own profile.

## Reproduce

```bash
cargo build --release --features metal,accelerate
./target/release/mlxcel-server -m models/mlx/qwen3-30b-a3b-4bit --parallel 4 --max-batch-prefill 4 --port 8090 &
python3 scripts/bench_serving_concurrency.py --port 8090 --prompt-tokens 512 --max-tokens 128 --concurrency 1,2,4
# per-layer attribution (device-synchronizing; compare shares, not totals)
MLXCEL_PROFILE_BLOCKS=1 MLXCEL_PROFILE_QWEN3_MOE_DETAIL=1 ./target/release/mlxcel-server -m models/mlx/qwen3-30b-a3b-4bit --parallel 4 --port 8090
# gather_qmm-per-row arm
MLXCEL_FUSED_MOE=0 ./target/release/mlxcel-server -m models/mlx/qwen3-30b-a3b-4bit --parallel 4 --port 8090
```

The op-level microbench is a scratchpad script (Python, `mlx` 0.32.2 wheel) that loads `model.layers.0.mlp.*` from the checkpoint and lifts the two Metal sources from `mlx_cxx_kernels.cpp`; its arms are described in section 3.3 and its numbers are only used for the relative ordering of paths on this GPU.
