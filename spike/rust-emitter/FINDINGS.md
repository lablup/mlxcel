# Rust-native StableHLO emitter spike: findings (#451)

Issue #451, ADR 0004. This evaluates a Rust-native StableHLO emitter as the
compiler-family model-authoring path, parallel to the JAX-reference frontend that
#449 validated. Same model (Llama-3.2-1B-Instruct), same bar (token-exact greedy
vs HF temp-0), CPU target. Standalone under `spike/rust-emitter/`, touches no
mlxcel crate and no file under `spike/openxla/`.

## Verdict

A Rust program that emits StableHLO **text** authors the Llama-3.2-1B
`decode_step` and greedy-decodes **token-exact (48/48)** against the HF reference
in `spike/openxla/artifacts/results.json`. The toolchain needed is only `cargo`
plus the `iree-compile` / IREE runtime already required to run any StableHLO. No
JAX, no `melior`, no MLIR/LLVM C++ build. The text-emission route is the
lowest-weight authoring toolchain available on this aarch64 box and it reaches
the same result #449 got from `jax.export`.

The two authoring paths are complementary, and the recommendation below is to use
both: the JAX reference as the executable oracle per architecture, and the Rust
emitter as the production authoring path that owns the graph.

## Toolchain choice: text emission, not melior

The plan offered text emission first and `melior` (MLIR C++ bindings) only as a
fallback, with the availability of the StableHLO dialect in the bindings on
aarch64 flagged as itself a P0 risk. Text emission worked on the first
end-to-end attempt, so `melior` was never needed and that risk never had to be
taken. This matters: text emission means the authoring toolchain is a
dependency-free Rust crate (the `Cargo.toml` pulls **zero** external crates) plus
the IREE CLI. There is no native MLIR/LLVM build, no C++ toolchain, and no
platform-specific dialect-registration question. The emitted `.mlir` is the same
artifact `iree-compile` already consumes from the JAX path, so it slots into the
existing compile/run pipeline unchanged.

## P0: toolchain round-trip (the gate)

A Rust-emitted single-op module

```mlir
func.func public @main(%a: tensor<4x8xf32>, %b: tensor<8x3xf32>) -> tensor<4x3xf32> {
  %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : ... -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}
```

compiles with `iree-compile --iree-input-type=stablehlo
--iree-hal-target-backends=llvm-cpu` and runs through the IREE runtime with an
exact match (max abs diff 0.0) against `a @ b` in numpy. A second probe module
exercised the op spellings not in the matmul, in particular
`dynamic_update_slice` (which #449's JAX lowering never used; it emitted
`scatter` for the KV write), a batched `dot_general`, `dynamic_slice`, static
`slice`, and `reduce`. All parsed and compiled. The route stands up.

## P1: full decode_step, token-exact

`src/model.rs` emits the complete `decode_step` graph: embedding gather, decode
key mask, 16 transformer layers (RMSNorm, GQA attention with llama3 RoPE, KV
write/read, softmax, SwiGLU MLP), final norm, and the tied LM head. The function
signature matches the JAX export exactly (146 weight tensors as individual inputs
in the same alphabetical-within-layer order, each carrying its pytree-path
`loc`, then `token`, `pos`, `cache_len`, `kcache`, `vcache`), so the same weight
glue maps checkpoints to args.

`python/run_decode.py` loads the real bf16 weights (upcast fp32), feeds them as
graph inputs, and drives greedy decode. The prompt is streamed through
`decode_step` (one token at a time, `cache_len = i`), which is identical to a
batched prefill because the decode mask is `iota <= cache_len`. Result:

```
Here are three short tips for staying focused while working:

1. **Set clear goals and priorities**: Before starting your work, define what
needs to be accomplished and prioritize your tasks. This will help you stay
focused on what's important and avoid

token match vs HF temp-0: 48/48  (EXACT)
```

Identical to the #449 JAX result. Getting this required the exact numerics the
plan called out: RoPE half-split (not interleaved), llama3 `inv_freq` scaling,
additive mask with `-1e30`, GQA head `h` reading kv head `h/4`, attention scale
`head_dim^-0.5`, fp32 compute over bf16 weights. The cos/sin tables are computed
in Rust f64 then cast f32 (matching the JAX `np.cos(emb).astype(f32)`), baked as
hex `dense` constants, and `dynamic_slice`d at `pos`; keeping the trig out of the
graph removes any dependence on the runtime's transcendental precision.

### Bucketed prefill (follow-up to #451)

`src/model.rs` `emit_prefill` now also emits the standalone bucketed `prefill`
graph, matching the JAX `prefill` signature (`tokens[Lp]`, `positions[Lp]`,
`real_len`; no input caches, zero-initialized internally; bucket `Lp = 64`). It
reuses the same builder over an `[Lp]` sequence axis: a `[Lp,Lp]` causal mask
(`j <= i`), the `[Lp]` KV block written with one `dynamic_update_slice` per
layer, and the last logit sliced at `real_len-1`. Running the Rust-emitted
`prefill` (first token) then `decode_step` (continuation) is **token-exact
(48/48)** against the same HF reference. The one new op the prefill needs is
`stablehlo.gather` (the multi-token embedding lookup `embed[tokens]`, plus the
per-position cos/sin lookup), whose `dimension_numbers` and `slice_sizes` mirror
the JAX-emitted `prefill.stablehlo.mlir`. The added builder ops are `gather`,
`transpose` (one per layer, for the GQA context output), and `linear_seq` (the
`[Lp, K]` activation matmul); decode stays token-exact 48/48 unchanged.

### Emitted graph vs the JAX graph

| | Rust emitter (decode) | JAX export (decode, #449) |
|---|---|---|
| distinct StableHLO op kinds | 20 | 23 |
| `dot_general` (the matmuls) | 145 | 145 |
| `custom_call` | 0 | 0 |
| `f64` | 0 | 0 |
| KV write op | `dynamic_update_slice` (32) | `scatter` (64) |
| weight orientation | `dot_general` contracts stored `[out,in]` directly | `transpose` then `dot_general` |
| total stablehlo ops | ~1376 | ~2044 |

Same matmul count, no custom ops, no f64. The emitter's graph is smaller because
it chooses leaner ops directly: `dynamic_update_slice` instead of `scatter`, and
contracting the stored `[out,in]` weight in `dot_general` instead of inserting a
`transpose` first. This is the graph-control point made concrete: the author
picks each op, rather than accepting a framework's lowering.

## Authoring-path comparison

### Per-architecture authoring effort (lines, complexity)

| | lines | role |
|---|---|---|
| JAX path (`spike/openxla/model_jax.py`) | ~230 | config + weight load + RoPE + **both** prefill and decode for one architecture |
| Rust `src/builder.rs` | 419 | reusable StableHLO builder, architecture-independent, write once |
| Rust `src/model.rs` | 324 | the decode graph for one architecture |
| Rust `src/config.rs` + `src/rope.rs` | 113 | config + RoPE tables |

The JAX file authors both graphs in ~230 readable, numpy-shaped lines. The Rust
per-architecture model code is ~324 lines for decode alone (a prefill variant
would push the architecture-specific total to roughly 550 to 650), on top of a
419-line builder that is written once and amortized across every future
architecture. So per architecture the Rust authoring code is roughly 1.5x to
2.5x the JAX line count, and the math is spelled as explicit ops
(`reduce` + `divide` for a mean, `reduce_max` + `subtract` + `exponential` +
`reduce_sum` + `divide` for softmax, slice/negate/concatenate for RoPE
half-split) rather than as `jnp.mean` / `jax.nn.softmax` / array slicing.

### Maintainability and debuggability

JAX authoring reads like the math and leans on the framework for shape
inference, op selection, and lowering. A new architecture is a new ~200-line
file. The cost is a standing JAX/jaxlib dependency and JAX expertise, and the
emitted StableHLO is generated and opaque (you debug Python, not the graph).

The Rust emitter is more verbose and hand-spells numerics, but it lives in the
same language as the rest of mlxcel, debugs with Rust tooling, and keeps no
Python/JAX runtime in the authoring path. The builder does result-type inference,
so a wrong rank or shape is caught at emit time (the type string goes
inconsistent and `iree-compile` rejects it) rather than silently. The honest
caveat from building this: the path to zero numeric bugs ran through
cross-checking every op spelling and shape against the JAX-emitted `.mlir` and
validating token-exact. Authoring a brand-new architecture blind, without a
reference graph to diff against, would be materially harder in the emitter than
in JAX, because a wrong contracting dim or broadcast is a silent numeric error
you chase at the IR level.

### Graph control for int4 and custom_call

This is where the emitter is clearly ahead, and it is the axis ADR 0004's Phase 2
(4-bit) cares about most.

- **int4 dequant-in-graph:** every matmul flows through one helper,
  `Builder::linear`. Inserting an int4 path is a local edit there: emit
  unpack + `convert` + scale + `dot_general`, and all 145 matmuls get it
  uniformly. The author controls whether the dequant sits as separate ops (for
  the compiler to fuse) or is shaped to encourage fusion. #449 flagged
  "does XLA/IREE fuse dequant into the matmul" as the open Phase 2 question; the
  emitter lets you author both shapes and measure, without fighting a framework
  lowering.
- **custom_call:** routing a matmul to a vendor int4 GEMM is just emitting a
  different op string (`stablehlo.custom_call @target`) from the same helper. The
  emitter already owns op selection, so a custom_call is additive and local. From
  JAX, a custom_call needs primitive registration or `lax` plumbing, which is
  more friction than emitting the op text.

### Toolchain and build weight

- **JAX path:** needs `jax` + `jaxlib` (hundreds of MB; on this aarch64 +
  Blackwell box only the CPU fallback exists, no CUDA jaxlib on PyPI) plus the
  `jax.export` serialization deps, then `iree-compile` to reach a runnable
  artifact.
- **Rust emitter:** needs `cargo` (already in the toolchain) and `iree-compile`
  / IREE runtime (already required either way). The emitter itself has zero
  external crate dependencies and no native build step. `melior` was not needed.
  This is the lightest authoring toolchain of the options and the one most
  aligned with mlxcel being a Rust project.

## Recommendation for ADR 0004

Adopt the Rust-native StableHLO **text** emitter as the production model-authoring
path for the compiler family, and keep the JAX reference as the per-architecture
**oracle**, not as the shipping authoring frontend.

Rationale:

1. **Graph control is the deciding axis for where this is going.** The next
   milestone is 4-bit. Inserting int4 dequant nodes and routing to a vendor int4
   `custom_call` are local, uniform edits in the emitter (one builder helper),
   and the emitter lets you author the exact dequant shape you want to measure
   for fusion. That control is the thing the JAX lowering does not give directly.
2. **Toolchain lightness and language fit.** Authoring needs only `cargo` and the
   IREE CLI that the runtime side needs anyway. No JAX/jaxlib, no `melior`, no
   MLIR/LLVM C++ build, no Python in the authoring path. The emitter is a small
   dependency-free Rust crate that lives next to the rest of mlxcel.
3. **It meets the bar.** Token-exact 48/48, same as the JAX path, with a smaller,
   custom-call-free, f64-free graph.

Use the JAX reference the way this spike used it: as the executable specification
and the token oracle when bringing up each new architecture. Author in JAX first
to derive and verify the numerics fast in math-shaped code, then author the
production graph in the Rust emitter and diff it against the JAX-emitted `.mlir`
and the JAX/HF tokens. That keeps JAX's fast, readable correctness loop while
shipping the emitter's control and toolchain-lightness. The per-architecture line
cost (roughly 1.5x to 2.5x JAX, plus hand-spelled numerics) is the price, and the
oracle workflow is what keeps that price from becoming a correctness risk.

### Open items (not in this spike)

- **prefill graph** in the emitter: done as a #451 follow-up (`emit_prefill`,
  `stablehlo.gather`), token-exact 48/48 via Rust-emitted prefill + decode.
- **int4 path:** author dequant-in-graph and a `custom_call` variant from
  `Builder::linear`, then measure fusion on a GPU-capable host (this box is CPU).
- **GPU backend:** same `iree-compile` route with a CUDA target; not exercised
  here (CPU only, per scope).
- **second architecture:** confirm how much of `builder.rs` is reused versus
  per-arch `model.rs` when the norm/activation/attention shape differs.
