# Bundled StableHLO graphs: Llama-3.2-1B-Instruct

`prefill.mlir` and `decode.mlir` are the StableHLO graphs the OpenXLA backend
compiles (with the IREE dist's `iree-compile`) and runs. They are emitted by the
Rust-native StableHLO emitter (issue #451) in `spike/rust-emitter`, in the
**on-device-argmax** variant: each graph ends in an argmax and returns the next
token id (`tensor<i32>`) rather than `[V]` logits (the Phase 2b sampling pattern).

- `prefill.mlir` — bucketed prompt prefill, bucket `Lp = MAX_SEQ = 256`. Inputs:
  146 weights, `tokens[256]`, `positions[256]`, `real_len`. Returns the first
  token id and the seeded KV cache. Reused as-is by the batched engine to seed a
  slot (single-seq prefill, then a device-side copy into the slot's KV region).
- `decode.mlir` — single-token step. Inputs: 146 weights, `token`, `pos`,
  `cache_len`, `kcache`, `vcache`. Returns the next token id and the advanced KV.

Both are token-exact (48/48) against the HF temp-0 reference in
`spike/openxla/artifacts/results.json`.

## Ragged (continuous-batching) decode graphs (#449 M3 Stage 2b)

`decode_ragged_b4.mlir` and `decode_ragged_b8.mlir` are the ragged
`decode_step` graphs the batched engine ([`XlaBatchEngine`](../../src/batch.rs))
runs: one fixed-`B_max` graph per supported slot count. Each row carries its OWN
`token[B]` / `pos[B]` / `cache_len[B]`, so sequences of different lengths share
one batch (the per-row key mask carries the raggedness; the KV write is unrolled
per row at each row's own position). Inputs: 146 weights, `token[B]`, `pos[B]`,
`cache_len[B]`, rank-5 `kcache`/`vcache` `[B,16,256,8,64]`; returns `token[B]`
and the advanced rank-5 KV. Same `module @decode_step` name as `decode.mlir`, so
the shim's `decode_step.main` call resolves either graph.

`B_max ∈ {4, 8}` is the bundled set; the engine selects the asset by the
requested `b_max` and rejects others. More slot counts mean regenerating an asset
(below) and extending `RAGGED_B_VALUES` in `src/iree.rs`. Runtime bucket
*selection* (picking among compiled buckets to cut idle-slot waste) is a later
optimization (Stage 2d), not these compile-time choices.

Reference-equivalence validated (each ragged slot reproduces its independent
single-seq stream) on CPU `local-task` and CUDA (GB10); see
`examples/xla_batch_bench.rs`.

## Regenerate

```bash
cd spike/rust-emitter
cargo run --release -- prefill-argmax out/prefill_argmax.mlir
cargo run --release -- decode-argmax  out/decode_argmax.mlir
cp out/prefill_argmax.mlir ../../src/lib/mlxcel-xla/assets/llama-3.2-1b/prefill.mlir
cp out/decode_argmax.mlir  ../../src/lib/mlxcel-xla/assets/llama-3.2-1b/decode.mlir

# ragged decode, one per supported B_max (on-device argmax):
for B in 4 8; do
  cargo run --release -- decode-ragged-argmax $B out/decode_ragged_b$B.mlir
  cp out/decode_ragged_b$B.mlir \
     ../../src/lib/mlxcel-xla/assets/llama-3.2-1b/decode_ragged_b$B.mlir
done
```

The weight arg order, the bucket (`PREFILL_LP`), the ragged slot counts
(`RAGGED_B_VALUES`), and these graphs must agree with `src/iree.rs`. The emitter
is the source of truth; do not hand-edit the `.mlir`.
