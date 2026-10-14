# MLX CUDA: cuDNN SDPA decode picks a stream-K engine and is not deterministic

Upstream bug report draft for ml-explore/mlx (and a note for cuDNN). Found in mlxcel (lablup/mlxcel#2128). mlxcel keeps the fast path as its default and carries an opt-in switch, `MLXCEL_SDPA_DETERMINISTIC=1`, in the `scaled_dot_product_attention.cpp` source overlay on the pinned MLX commit.

## Summary

Once a KV cache reaches 256 positions, `fast::scaled_dot_product_attention` with one query row and evaluated cache slices goes to cuDNN (`use_cudnn_for_decoding`): k and v are unsliced to the whole cache buffer and the true lengths reach cuDNN through `set_padding_mask` with `set_seq_len_q` / `set_seq_len_kv`. `build_sdpa_graph` then builds the heuristics' first plan (`HeurMode_t::A`, filtered only to `SUPPORTS_CUDA_GRAPH_NATIVE_API`).

On GB10 (sm_121) with cuDNN 9.27, for the head_dim 128 graph of a 16-query-head, 8-KV-head model, that first plan is `eng8_k40=0_k41=2_k24=2_k38=1_k27=0`. Knob 38 is `CUDNN_KNOB_TYPE_STREAM_K`. The output of that engine rounds differently on about 1 call in 100 (up to 7.6e-6 in a float16 output, ULP level, so a change of summation order rather than of the data read). A contributor's analysis on lablup/mlxcel#2128 (a standalone cuDNN reproducer on cuDNN 9.13.0 and 9.24.1, without MLX) locates the order dependence in an unordered cross-warp FP32 reduction of the softmax denominator through a shared-memory CAS loop: forcing a fixed order inside the kernel made the output stable, and replaying the measured partial sums in two warp orders reproduced the two observed outputs. cuDNN does not tag the engine `CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC`, so `deselect_numeric_notes({NONDETERMINISTIC})` keeps it. The second candidate, `eng8_k40=2_k41=1_k24=1_k38=0_k27=0` (stream-K off), was bit-identical over 4000 calls of the same shape; whether that holds for every shape was not established. It also differs in its tile and kernel-config knobs (40, 41, 24), so knob 38 is the measured correlate, not an isolated cause.

Observable effect: temperature-0 decoding of Qwen3 (0.6B, 1.7B, 4B, 4-bit) differs from run to run and within one process once the context passes 256 tokens, because a 1-in-100 rounding change per call over 28 layers and hundreds of steps eventually flips a near-tied argmax. Llama-3.2-1B (head_dim 64) and Gemma-3-1B (head_dim 256, which cuDNN SDPA does not take) were byte-identical over 5 runs of an 800-token generation here; the contributor reports Llama variability in other runs, cause not established.

Environment: GB10 (DGX Spark, sm_121), CUDA 13, cuDNN 9.27.0, cudnn-frontend 1.16.0 as fetched by MLX's CMake.

## Evidence

Repeated identical call, fresh output each time, output byte-hashed (`tests/cuda_sdpa_determinism.rs` in mlxcel): q `[1, 16, 1, 128]`, k and v leading slices `[1, 8, L, 128]` of evaluated `[1, 8, E, 128]` float16 buffers.

| Shape (L in E) | Engine | Distinct outputs over 4000 calls |
|---|---|---|
| 300 in 512 | heuristics' first (stream-K on) | 2 (28 to 46 calls in the minority across five runs) |
| 300 in 512 | forced plan index 1 (stream-K off) | 1 |
| 300 in 512 | `MLX_CUDA_USE_CUDNN_SDPA=0` (`sdpa_vector`) | 1 |
| 512 in 512 (no padding) | heuristics' first | 1 |
| 257 in 512, 1000 in 1024 | heuristics' first | 1 over 2000 |
| 1500 in 2048 | heuristics' first | 2 over 2000 (446 in the minority) |
| 1500 in 2048 | `sdpa_vector` two-pass (mlxcel's switch) | 1 over 2000 |
| head_dim 64, 32/8 heads, 300 in 512 | heuristics' first | 1 over 2000 |

Model level, Qwen3-1.7B 4-bit, greedy, logits hashed per step with fresh caches: prefill and the first 224 decode steps were bit-identical between repeats, and every comparison diverged between decode steps 225 and 247 (prompt 34 tokens, so the cache had just passed 256). With cuDNN SDPA disabled, or with stream-K engines barred, 400 steps were bit-identical in all repeats, in-process and across processes. With the switch on, 5 separate processes of an 800-token greedy generation gave one output each for Qwen3-0.6B, 1.7B and 4B, Llama-3.2-1B and Gemma-3-1B, with CUDA graphs on and off.

## Cost, and what mlxcel does

Stream-K is also what keeps a one-row decode fast: with a single query row, the key axis is the only source of parallelism. Decode throughput on GB10, median of 3 interleaved runs, relative to the default (stream-K) selection:

| Model | Keys | cuDNN, stream-K barred | `sdpa_vector` |
|---|---|---|---|
| Qwen3-1.7B 4-bit | about 400 | 0.99 | 1.01 |
| Qwen3-1.7B 4-bit | 2826 | 0.92 | 1.01 |
| Qwen3-1.7B 4-bit | 7366 | 0.90 | 0.96 |
| Qwen3-1.7B 4-bit | 15653 | 0.88 | 1.00 |
| Qwen3-4B 4-bit | 2826 | 0.94 | 0.91 |
| Qwen3-4B 4-bit | 15653 | 0.87 | 0.94 |
| Llama-3.2-1B 4-bit | 2717 | 0.89 | 0.95 |
| Llama-3.2-1B 4-bit | 15122 | 0.67 | 0.87 |

Barring stream-K costs more than leaving cuDNN for decode, so mlxcel's switch routes the one-row decode to `sdpa_vector` (its KV split reduces in a fixed order) and, as a precaution, bars stream-K engines on the remaining forward cuDNN graphs, where prefill throughput measured within noise; their reproducibility was not separately measured, and a graph with no buildable stream-K-off plan keeps the default selection with a one-time warning. The default stays fast and nondeterministic.

## Suggested fix

- MLX: offer a determinism option that keeps one-row decode off cuDNN (or off stream-K engines), for users who need reproducible greedy output, and document that the default cuDNN decode is not bitwise reproducible.
- cuDNN: tag SDPA engines whose reduction order is not fixed (the head_dim 128 decode engine above, at least) `CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC`, so that `deselect_numeric_notes` is enough to exclude them.
