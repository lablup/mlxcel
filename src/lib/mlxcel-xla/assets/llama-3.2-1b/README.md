# Bundled StableHLO graphs: Llama-3.2-1B-Instruct

`prefill.mlir` and `decode.mlir` are the StableHLO graphs the OpenXLA backend
compiles (with the IREE dist's `iree-compile`) and runs. They are emitted by the
Rust-native StableHLO emitter (issue #451) in `spike/rust-emitter`, in the
**on-device-argmax** variant: each graph ends in an argmax and returns the next
token id (`tensor<i32>`) rather than `[V]` logits (the Phase 2b sampling pattern).

- `prefill.mlir` — bucketed prompt prefill, bucket `Lp = MAX_SEQ = 256`. Inputs:
  146 weights, `tokens[256]`, `positions[256]`, `real_len`. Returns the first
  token id and the seeded KV cache.
- `decode.mlir` — single-token step. Inputs: 146 weights, `token`, `pos`,
  `cache_len`, `kcache`, `vcache`. Returns the next token id and the advanced KV.

Both are token-exact (48/48) against the HF temp-0 reference in
`spike/openxla/artifacts/results.json`.

## Regenerate

```bash
cd spike/rust-emitter
cargo run --release -- prefill-argmax out/prefill_argmax.mlir
cargo run --release -- decode-argmax  out/decode_argmax.mlir
cp out/prefill_argmax.mlir ../../src/lib/mlxcel-xla/assets/llama-3.2-1b/prefill.mlir
cp out/decode_argmax.mlir  ../../src/lib/mlxcel-xla/assets/llama-3.2-1b/decode.mlir
```

The weight arg order, the bucket (`PREFILL_LP`), and these graphs must agree with
`src/iree.rs`. The emitter is the source of truth; do not hand-edit the `.mlir`.
