# Rust-native StableHLO emitter spike (#451)

A standalone evaluation of authoring the compiler-family model graph from
**Rust** instead of from a JAX reference. The Rust program emits StableHLO
**text**, the existing `iree-compile` lowers it to a CPU vmfb, and the IREE
runtime executes it. The target is the same Llama-3.2-1B-Instruct forward that
issue #449 validated through `jax.export`, verified to the same bar:
token-exact greedy decode against the HF transformers temp-0 reference.

This is a parallel alternative to the #449 JAX-reference path, not a replacement
of it. CPU target only. It builds and runs with **only** `cargo` plus the
`iree-compile` / IREE runtime already present in `spike/openxla/.venv`. No JAX,
no `melior`, no MLIR/LLVM C++ build.

## Result

- **P0 (toolchain gate):** Rust-emitted single `dot_general` compiles and runs
  through IREE, exact match vs numpy. The text-emission route stands up.
- **P1 (the real bar):** the full Rust-emitted `decode_step` greedy-decodes
  **token-exact, 48/48**, against the HF reference in
  `spike/openxla/artifacts/results.json`. Output text is identical.

See `FINDINGS.md` for the toolchain choice, the authoring-path comparison
(JAX-reference vs Rust-emitter), and the recommendation for ADR 0004.

## Layout

| path | what |
|---|---|
| `src/builder.rs` | SSA/type-tracking StableHLO text builder (one method per op). Architecture-independent, reused across graphs. |
| `src/config.rs` | Llama-3.2-1B config constants. |
| `src/rope.rs` | llama3 `inv_freq` and cos/sin tables (f64, then f32), byte-for-byte with the JAX reference. |
| `src/model.rs` | Emits the full `decode_step` graph (head, 16 layers, tied head). |
| `src/main.rs` | CLI: `emit p0 \| probe \| decode [out.mlir]`. |
| `python/run_decode.py` | Loads real bf16 weights (upcast fp32), drives greedy decode through the compiled module, compares to the HF tokens. |
| `validate.sh` | End-to-end driver: build, emit, compile, check. |

## Run

```bash
# everything (P0 round-trip + P1 token-exact decode)
./validate.sh

# just the toolchain gate
./validate.sh p0

# emit only, inspect the StableHLO
cargo run --release -- decode out/decode.mlir
```

`validate.sh` reuses `spike/openxla/.venv` for `iree-compile`, the IREE runtime,
`torch`, `transformers`, and `safetensors`, and reads the weights from
`spike/openxla/models/Llama-3.2-1B-Instruct`. Nothing here builds or imports any
mlxcel crate (the `Cargo.toml` declares an empty `[workspace]` so it is its own
workspace root).

## How decode_step alone covers the prompt

Only `decode_step` is emitted. The prompt is streamed through it one token at a
time (`cache_len = i` for prompt token `i`). Because decode masks keys with
`iota <= cache_len`, streaming the prompt is mathematically identical to a
batched prefill: the same KV cache and the same position-`n-1` logits. The
48/48 token match confirms the prompt was processed correctly. A separate
bucketed `prefill` graph is straightforward additional work in the same builder
(the one new op it needs is `stablehlo.gather` for the multi-token embedding
lookup); it adds no new toolchain risk and was left out of this spike.
