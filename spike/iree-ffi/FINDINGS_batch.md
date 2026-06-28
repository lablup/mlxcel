# Uniform-B batched decode (issue #449 M3 Stage 1)

Stage 1 of the throughput milestone: a **uniform-B (lockstep) static batched
`decode_step`** authored from the Rust emitter, driven through IREE from Rust,
validated token-exact and measured for aggregate tok/s scaling on CPU then CUDA.

Verdict: **PASS.** Batching is the throughput lever the M3 decision predicted.
Aggregate decode throughput scales from the batch-1 baseline up to **220 tok/s
(CUDA, B=64)** and **69 tok/s (CPU)**, token-exact at every B. The single-seq
batch-1 path used a small fraction of the device; batching recovers it.

## What "uniform-B" means

All B sequences advance in lockstep at the **same** position, so `pos`,
`cache_len`, and the key mask are shared scalars/vectors broadcast over the
batch. Only the token, the activations, and the KV cache carry a leading batch
dim B. This is the cheap, high-signal step before continuous batching (Stage 2),
and it is the regime that turns each decode matmul from a batch-1 GEMV
(bandwidth/launch-bound) into a GEMM that reuses each weight across B rows.

## What was built (all in the spike)

| Piece | Where |
|-------|-------|
| Batched decode graph | `rust-emitter` `model.rs::emit_decode_batched` + CLI `decode-batch[-argmax] <B>`. Leading B dim threaded through; rank-5 KV `[B,L,S,nkv,d]`; two-batch-dim `dot_general` for GQA scores/context; shared `pos`/`cache_len`/mask. |
| Batched argmax | `builder.rs::argmax_batched` (`[B,V] -> [B] i32`, reduce over dim 1; shared reducer block with the scalar `argmax`). |
| Shim | `iree_gate.c`: `xla_llama_prefill_batch` (single-seq prefill once, tile its rank-4 KV across B rows into a resident rank-5 cache, return the first token per row) + `xla_llama_decode_batch` (token[B] + rank-5 KV -> token[B], advances the KV; `[B]i32` on-device-argmax or `[B,V]` host argmax). |
| Driver | `src/bin/llama_batch.rs`: loads the real bf16 weights, prefills once, runs two checks + the throughput timing. |
| Sweep | `validate_batch.sh cpu|cuda ["B list"]`. |

The batched decode is emitted identically for every B (only the integer B
changes the shapes); there is no per-B special-casing.

## Validation (two gates, both pass at every B)

1. **Token-exact.** B identical rows each reproduce the 48-token HF temp-0
   reference (`results.json`), and all rows are byte-identical to each other.
2. **Independence.** With row 0 seeded from the real first token and rows >= 1
   seeded from a *different* token over the same prompt KV, row 0 stays
   token-exact (48/48) while row 1 diverges. This rules out a collapsed/averaged
   batch dim, which identical rows alone would hide.

B=1 batched reproduces the scalar baseline exactly: CPU 455 ms/step (scalar
~447), CUDA 178 ms/step (scalar ~177), both 48/48.

## Scaling

Reference prompt 46 tok (bucket 256), 47 decode steps, `decode-batch-argmax`
(on-device argmax). `tok/s` is `B * steps / wall`. Dev box `spark-101` (GB10
Grace-Blackwell, sm_121).

CPU (`local-task`, prebuilt dist, Grace cores):

| B  | ms/step | per-seq tok/s | aggregate tok/s |
|----|---------|---------------|-----------------|
| 1  | 454.9   | 2.20          | 2.20            |
| 2  | 84.5    | 11.84         | 23.68           |
| 4  | 110.0   | 9.09          | 36.35           |
| 8  | 145.6   | 6.87          | 54.95           |
| 16 | 248.8   | 4.02          | 64.32           |
| 32 | 462.5   | 2.16          | 69.19           |

CUDA (GB10, source-built cuda runtime + pip cuda iree-compile):

| B  | ms/step | per-seq tok/s | aggregate tok/s |
|----|---------|---------------|-----------------|
| 1  | 178.2   | 5.61          | 5.61            |
| 2  | 51.8    | 19.29         | 38.59           |
| 4  | 87.5    | 11.42         | 45.69           |
| 8  | 89.0    | 11.24         | 89.92           |
| 16 | 1125    | 0.89          | 14.22  (cliff)  |
| 24 | 122.6   | 8.16          | 195.73          |
| 32 | 217.6   | 4.60          | 147.07          |
| 64 | 290.0   | 3.45          | 220.66          |

## Findings

1. **Batching is the throughput lever (confirmed).** CPU 2.2 -> 69 tok/s (~31x),
   CUDA 5.6 -> 220 tok/s (~39x at B=64, still climbing). The batch-1 path was
   severely underutilized, exactly as the M3 decision argued.

2. **The batch-1 -> batch-2 cliff is large on both devices** (CPU 455 -> 85
   ms/step, CUDA 178 -> 52). batch-1 is pathologically inefficient, not a
   linear baseline: on CPU the leading-1 dim defeats vectorization/threading
   (~10 GFLOP/s at B=1 vs ~120 at B=2); on GPU it is bandwidth/launch-starved.
   So the honest "1x" baseline is genuinely bad, and even B=2 is a big win.

3. **Bandwidth-bound regime is visible.** CUDA B=4 -> B=8 is nearly free
   (ms/step 87.5 -> 89.0 while B doubles): the per-step weight read dominates, so
   adding sequences barely costs time and aggregate throughput nearly doubles.
   CPU saturates around B=16-32 (compute-bound: B=32 ms/step 462 ~= B=1 455, i.e.
   32 tokens in the wall-time of one batch-1 token).

4. **Untuned IREE-CUDA codegen is non-monotonic.** B=16 hits a **reproducible**
   catastrophic kernel (1125 ms/step over two runs, 12x slower than B=8), and
   B=32 (147 tok/s) dips below B=24 (195). B=8/24/64 are all fine and B=64 is the
   peak, so this is a codegen-selection artifact, not a hardware or approach
   limit. Productionizing a fixed B needs codegen tuning or empirical B
   selection; do not assume a smooth curve.

## Next

- **Stage 2 (continuous batching):** ragged per-sequence `cache_len`/positions,
  a paged-KV block table, and an admit/evict scheduler interleaving
  prefill/decode. This is a multi-sequence session beyond the single-sequence
  `InferenceSession` contract (the KV/batching abstraction ADR 0004 deferred).
- **Stage 3:** int4 dequant fusion (a later multiplier, needs a vendor int4 GEMM
  `custom_call`).
- Investigate / tune the IREE-CUDA codegen non-monotonicity (B=16) before
  baking a production batch size.

## Reproduce

```bash
cd spike/iree-ffi
./validate_batch.sh cpu       # local-task sweep via the prebuilt dist
./validate_batch.sh cuda      # cuda sweep via the source-built runtime
./validate_batch.sh cuda "8 24 64"
```
