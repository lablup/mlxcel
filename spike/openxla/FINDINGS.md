# OpenXLA export-route spike: Phase 0 + Phase 1 findings

Issue #449, ADR 0004. Scope of this run: Phase 0 (environment + stack selection)
and Phase 1 (fp16 export spike). Phase 2 (4-bit) and Phase 3 (mlxcel integration)
were not started. Work is standalone under `spike/openxla/` and touches no mlxcel
crate or build.

## Verdict

The export-first route holds for the fp16 path. A Llama-3.2-1B model written once
as a JAX reference exports to StableHLO through `jax.export`, and the serialized
graph greedy-decodes **token-exact (48/48)** against an HF transformers temp-0
reference. The same StableHLO runs unmodified on two independent runtimes (PJRT
and IREE) with matching argmax. No hand-written StableHLO and no per-op work were
needed. This is the result ADR 0004 wanted from the spike before committing the
export-first model-definition strategy for the compiler family.

One caveat on scope: this validated the **JAX-reference -> StableHLO** export
option, not the **HF-PyTorch -> torch.export -> torch-mlir/torch-xla** option.
The JAX path was chosen deliberately (see Stack selection). The two-runtime
result (PJRT and IREE ingest the same module) is the stronger portability signal
either path was meant to produce.

## Phase 0: environment and stack selection

### Host

| | |
|---|---|
| Machine | `spark-101`, NVIDIA GB10 (Grace-Blackwell), **aarch64** |
| OS / kernel | Ubuntu, Linux 6.17 |
| GPU / CUDA | GB10 Blackwell, driver 580.159.03, CUDA 13.0 |
| CPU / RAM | 20-core Grace, 121 GiB |
| Python | 3.12.3 (system), `uv` 0.11.7 |

### Stack chosen: JAX reference -> StableHLO -> PJRT (and IREE), CPU target

Two facts drove the choice:

1. **aarch64 + Blackwell is a thin-wheel platform.** JAX and PyTorch CUDA wheels
   for aarch64 + sm_121 live in NVIDIA NGC / JAX-Toolbox containers, not on PyPI.
   The issue allows CPU as the milestone target, so CPU avoids that whole problem
   while still exercising the export pipeline end to end.
2. **JAX emits StableHLO natively.** `jit(...).lower(...).compiler_ir("stablehlo")`
   and `jax.export` produce StableHLO with no extra toolchain. The PyTorch route
   needs torch-mlir (a heavy aarch64 source build) or torch-xla (x86/TPU-focused
   wheels) to reach StableHLO, both higher-risk on this box. Reimplementation
   correctness is something I can fully control and verify against HF; wheel
   availability on aarch64 I cannot.

Using a hand-written JAX model also makes every detail the findings must document
(KV layout, bucketing, RoPE, mask, on-device sampling) explicit and editable,
which is the point of a spike. HF transformers (PyTorch, CPU) is the independent
reference, so the token-match is a cross-framework check, not a self-check.

### Pinned environment (reproducible spec)

Full set in `requirements.lock` (40 packages). Key pins:

```
jax==0.10.2            jaxlib==0.10.2         ml-dtypes==0.5.4
torch==2.12.1+cpu      transformers==5.12.1   tokenizers==0.22.2
numpy==2.5.0           safetensors==0.8.0     huggingface-hub==1.21.0
flatbuffers==25.12.19  iree-base-compiler==3.11.0  iree-base-runtime==3.11.0
```

`torch` is the CPU build (from `download.pytorch.org/whl/cpu`) so it never pulls
a CUDA stack. `flatbuffers` is required by `jax.export` serialization. `jaxlib`
falls back to CPU with a one-line notice (no CUDA jaxlib on this platform).

### Reference model

`unsloth/Llama-3.2-1B-Instruct` (ungated bf16 mirror, single `model.safetensors`,
2.47 GB). Avoids the `meta-llama` gate. Config matches the standard Llama-3.2-1B:
16 layers, hidden 2048, 32 query / 8 KV heads, head_dim 64, intermediate 8192,
RoPE theta 5e5 with **llama3 scaling**, rms_norm_eps 1e-5, **tied embeddings**,
vocab 128256.

## Phase 1: fp16 export spike

### What was exported and run

- `prefill(params, tokens[Lp], positions[Lp], real_len) -> (last_logits[V], kcache, vcache)`
- `decode_step(params, token, pos, cache_len, kcache, vcache) -> (logits[V], kcache, vcache)`

Both go through `jax.export.export`, are serialized to `.exported.bin`, dumped as
StableHLO text to `.stablehlo.mlir`, then **deserialized and executed** to drive
the greedy loop. The loop runs from the reloaded artifact, not the in-memory jit,
so it proves the exported graph itself executes.

| graph | serialized | StableHLO text | stablehlo ops | inputs / outputs |
|---|---|---|---|---|
| prefill (bucket 64) | 176.5 KB | 503.8 KB | ~1927 | 149 / 3 |
| decode_step | 179.1 KB | 509.0 KB | ~2044 | 151 / 3 |

Weights are graph **inputs** (146 leaf tensors), not baked constants. The same
compiled graph therefore serves any same-architecture checkpoint; weight mapping
is the only per-checkpoint glue. The func signature carries the pytree paths
(`params['layers'][0]['down']`, etc.), which makes the weight-to-arg mapping
self-documenting for a Rust loader later.

### Result: coherence and token match

Prompt (chat-templated): "Give me three short tips for staying focused while
working." Greedy, 48 new tokens.

```
Here are three short tips for staying focused while working:

1. **Set clear goals and priorities**: Before starting your work, define what
needs to be accomplished and prioritize your tasks. This will help you stay
focused on what's important and avoid
```

- **Token match vs HF temp-0: 48/48 exact**, no divergence.
- Isolated math check (`check_correctness.py`): JAX prefill last-token logits vs
  HF agree to max abs diff 7e-5, identical top-5 and argmax. The ~1e-5 gap is
  fp32 reduction-order noise, not an algorithmic difference.

Both sides compute in fp32 over the model's bf16 weights upcast to fp32. "fp16"
here means the non-quantized checkpoint (the ADR's intermediate checkpoint),
not fp16 arithmetic. fp32 compute on both sides is what makes the greedy match
exact rather than near; bf16-arithmetic parity is a numerics question for later,
not a blocker for coherence.

### KV cache representation

Two fixed-capacity tensors per run, stacked across layers:

```
kcache, vcache : f32[L=16, MAX_SEQ=256, n_kv=8, head_dim=64]
```

- Prefill writes the prompt KV into `[:, :Lp]` (static slice) and returns the
  full-capacity cache. `cache_len` starts at the real prompt length.
- Decode writes the new token's K/V at index `cache_len` via a dynamic-index
  scatter (`cache.at[layer, cache_len].set(...)`, lowering to `dynamic_update_slice`)
  and advances `cache_len` by one. Padded prefill slots beyond `real_len` are
  overwritten by the first decode steps and are never read (see mask).
- On the live jit path, `decode_step` is jitted with `donate_argnums` on the two
  cache tensors, so PJRT can update them in place (XLA `input_output_alias`).
  On CPU this avoids reallocation per step; the bandwidth win matters on
  accelerators. The exported/deserialized path is functional (donation is a jit
  runtime property, not part of the serialized graph), and produces identical
  tokens.

### Shape bucketing

- `MAX_SEQ = 256` cache capacity; prompt + generated must fit.
- Prefill prompt length is padded up to the smallest of `{32, 64, 128, 256}` that
  fits (46-token prompt -> bucket 64). One compiled prefill per bucket; decode is
  single-token and shape-static, so one decode executable serves all steps.
- Decode does not pad: the running length lives in the `cache_len` scalar and the
  graph shape is fixed at `MAX_SEQ`.

### RoPE and mask handling

- **RoPE**: `inv_freq` computed with the HF `llama3` scaling
  (`_compute_llama3_parameters`: factor 32, low/high freq factors 1/4, original
  context 8192, the medium-band smooth interpolation). cos/sin are precomputed
  as a `[MAX_SEQ, head_dim]` constant table indexed by `positions` (prefill) or
  the `pos` scalar (decode). Rotation uses the HF half-split `rotate_half`
  convention, not interleaved. Getting this exactly right is what makes the
  logits match; an interleaved RoPE or unscaled inv_freq diverges immediately.
- **Mask**: prefill uses an additive causal mask `key_j <= query_i`. Real query
  rows only ever see real keys, so the padded tail needs no extra masking (its
  query outputs are discarded; only `real_len - 1` is read). Decode masks keys by
  `iota(MAX_SEQ) <= cache_len`, which both enforces causality and hides the
  padded and unwritten cache tail in one compare. Mask values use -1e30, not -inf, to
  keep softmax numerically clean.

### On-device sampling

The harness samples argmax on the host for easy comparison (explicitly allowed
for the spike). The recommended design is on-device argmax: a `decode_step`
variant returning the next token id instead of logits changes the first output
from `f32[128256]` (513024 bytes) to `i32[]` (4 bytes), a **128256x smaller
per-token device-to-host transfer**, and `argmax` lowers to a `reduce` already in
the graph. This is the form Phase 3 should ship; returning logits is only for the
host-sampling and log-prob paths.

### StableHLO op inventory and portability

`decode_step` lowers to 23 distinct StableHLO op kinds, all standard, **no
`custom_call`**, no f64. Top ops: `broadcast_in_dim` (521), `add` (216),
`multiply` (211), `reshape` (179), `dot_general` (145, the matmuls), `transpose`
(113), `slice` (96), `reduce`/`divide` (65 each, norms and softmax),
`concatenate` (64, RoPE), `scatter` (64, KV writes), `compare`/`select` (39 each,
masking). A graph with no custom ops is the portable case the ADR's StableHLO
convergence bet depends on.

IREE confirms it: `iree-compile --iree-input-type=stablehlo
--iree-hal-target-backends=llvm-cpu` ingests the exported `decode_step.mlir`
unmodified and produces a 388 KB vmfb. Running one decode step through the IREE
runtime gives the same argmax as PJRT, logits within 9e-5. So the same exported
StableHLO is portable across PJRT and IREE with no source changes.

### Performance (CPU fp32, this box)

| | |
|---|---|
| prefill (bucket 64) | ~1.1 s (first call, includes compile) |
| decode steady-state | ~3.0 tok/s (~334 ms/token) |

These are CPU fp32 numbers and are not a target; they exist to confirm the loop
runs at a usable spike speed. GPU is where the real numbers come from, and is the
first follow-up (see below).

## What worked, what is open, what is blocked

**Worked**
- Native StableHLO export from a JAX reference, serialized and re-executed.
- Token-exact greedy vs HF over 48 tokens; isolated logits match to fp32 noise.
- Static-shape KV cache with bucketed prefill and single-token decode.
- Clean, custom-call-free StableHLO portable across PJRT and IREE.

**Open (next phases, not started here)**
- **4-bit (Phase 2).** The op inventory is pure-f32 `dot_general`. A
  dequant-in-graph int4 path would insert unpack + convert + scale before each
  `dot_general`; whether XLA and IREE fuse that into the matmul (versus
  materializing a bf16 weight) is the number to measure. The `custom_call` route
  to a vendor int4 GEMM is additive since none exist in the graph today. This is
  the Phase 2 entry point.
- **torch.export option.** Not exercised. If HF-PyTorch -> StableHLO via
  torch-xla or torch-mlir is wanted as the per-model authoring path (to reuse the
  30+ HF model definitions directly), it needs its own feasibility check,
  probably in an x86 CUDA container rather than this aarch64 box.
- **Multi-architecture reuse.** Validated for one architecture (Llama). The
  "per-model work shrinks to glue" claim holds within an architecture family
  (weights are inputs); a second architecture (for example a GQA + different norm
  or activation) would confirm how much of `model_jax.py` is reusable scaffolding
  versus per-arch code.

**Blocked / deferred on this box**
- **GPU execution.** No CUDA jaxlib for aarch64 + Blackwell on PyPI. Real GPU
  numbers need the NGC JAX-Toolbox container (or an x86 CUDA host). The CPU result
  is sufficient for the Phase 1 coherence bar; GPU is the first thing to wire up
  for Phase 2 perf.

## Recommendation

Commit the export-first direction for the compiler family on the strength of this
result, with the JAX-reference path as the primary authoring route for now. The
two-runtime portability (PJRT and IREE, same module) is the part most relevant to
ADR 0004's hardware-target-as-plugin hypothesis and it held cleanly. Proceed to
Phase 2 (4-bit lowering characterization) on a GPU-capable host, and keep the
provisional session contract (`prefill` / `decode_step` / session-owned KV /
on-device sampling) as the Phase 3 integration shape; the graphs here already fit
it.
