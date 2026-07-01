# Design: the SSM / hybrid / recurrent track for the OpenXLA/IREE backend (#502)

Status: Proposed (2026-07-01). Scoping design for Window D of epic #493 (OpenXLA
architecture-coverage parity). Follows ADR 0004 (the compute-backend session seam
and the StableHLO/MLIR family) and the continuous-batching design in
`STAGE2_DESIGN.md`.

Decision: DEFER the implementation to a dedicated follow-up epic, #564. This
document records the design so the follow-up work is cheap to pick up; it does not
land a reference architecture. See "Decision and rationale" for why.

## Goal

Bring state-space (Mamba/Mamba2), gated-linear-attention, and recurrent
architectures, and the hybrid models that interleave them with attention, onto
the OpenXLA/IREE backend. The target set (from issue #502): Mamba / Mamba2,
Jamba, Falcon-H1, RWKV7, Qwen3-Next, Plamo2, Nemotron-H, Kimi-Linear,
RecurrentGemma.

These architectures replace (or interleave) standard attention with a mixer that
carries a fixed-size recurrent STATE instead of a growing KV cache. Two things
are genuinely new versus everything the backend serves today:

1. A recurrent/state-space mixing graph primitive (a selective scan / linear
   recurrence), which does not fit the shared attention core.
2. Explicit per-slot STATE allocation and carry through the continuous-batching
   engine, distinct from the KV cache (fixed-size, does not grow with position),
   and coexisting with the KV cache in hybrid models.

## What already exists (the base we build on)

The infrastructure the epic calls "no longer the gap" applies here too, and the
recurrent track reuses all of it unchanged:

- Config-driven emit: `Config::from_json` (`emitter/config.rs:378`) reads
  `config.json` into a `Config`; the per-layer loop `emit_transformer_layer`
  (`emitter/model.rs:1563`) builds each layer; four graph flavors share it
  (`emit_prefill`, `emit_decode`, `emit_decode_batched`, `emit_decode_ragged`,
  `emitter/model.rs`).
- The shared attention core `AttnLayout` (`emitter/model.rs:1013`) with
  `project_qkv` / `rope_qk` / `write_read_kv` / `raw_scores` / `add_mask` /
  `context` / `o_proj`, and the FFN dispatch `emit_ffn_body`
  (`emitter/model.rs:867`) that reaches the dense SwiGLU MLP or the MoE primitive.
- The MoE FFN primitive `moe_block` (`emitter/moe.rs:231`), the template for how a
  new shared graph primitive is added: parameterized by `MoeConfig`
  (`emitter/config.rs`), called from one dispatch site, reused across every graph
  flavor, weights declared in `weight_specs` (`weights.rs`) in lock-step with the
  emitter arg order.
- The continuous-batching engine `XlaBatchEngine` (`batch.rs`): `B_max` slots, a
  FIFO admission queue, and an admit/decode/evict `pump` loop. Each active `Slot`
  (`batch.rs:70`) carries `cur` / `cache_len` / `produced` / `params` / `rng` /
  `history`.
- The IREE runtime `IreeRaggedLlama` (`iree.rs:645`): `prefill_slot_logits`
  (device-side KV slot write, live slots untouched) and `decode_ragged_logits`
  (all `B_max` rows advance one token, each at its own position).
- The C shim (`csrc/xla_iree.c`): resident weights, and the KV cache threaded
  across steps. `kcache_b` / `vcache_b` are the rank-5 per-slot KV
  `[B,L,S,nkv,d]` (`csrc/xla_iree.c:98`); `xla_ensure_batch_kv`
  (`csrc/xla_iree.c:390`) allocates the per-slot region zeroed on first use.
- The serve worker `XlaServeWorker` (`src/server/batch/xla_worker.rs`) maps
  requests onto the engine through the backend-neutral `BatchEngine` trait.

Crucially, the MLX engine already implements every family in the target set, and
its implementation is a concrete reference for the two new pieces. `src/models/mamba2.rs`
has a `Mamba2Cache` with `snapshot_into` / `restore_from` (the O(1) recurrent
state carry the OpenXLA path must reproduce as device buffers), and
`src/execution/kv_arch.rs` already classifies architectures with a `KvArchKind`
enum whose variants include `Hybrid` (attention + recurrent) and `PureSsm` (no KV
cache); `estimate_kv_arch_from_config` derives it from `config.json`. The APC
opt-out module `src/server/prompt_cache/hybrid_ssm.rs` is authoritative on the
coverage and on the correctness hazard: `HYBRID_SSM_MODEL_TYPES`
(`hybrid_ssm.rs:89`) lists `jamba`, `mamba`, `mamba2`, `nemotron_h`,
`gated_delta` (Qwen3-Next), `kimi_linear`, `qwen3_next`, `falcon_mamba`,
`longcat_flash`, `rwkv7`, `recurrent_gemma`. On the MLX side these families
override `LanguageModel::supports_batching()` to `false`, which routes their
sequence state to `SequenceStateBackend::ModelOwned` and force-disables paged KV
and Automatic Prefix Caching, because the recurrent hidden state "cannot be
reconstructed from a token-prefix hash alone". That MLX carve-out is exactly the
property the OpenXLA path must reproduce in the graph-and-shim world: recurrent
state is per-sequence, opaque to prefix hashing, and threaded explicitly.

## The gap: what a recurrent mixer needs beyond attention

`Config::from_json` today rejects every target `model_type` at the
unsupported-arch error (`emitter/config.rs:828`, "the OpenXLA emitter supports the
dense architectures ..."). Wiring a recurrent family is not the cheap four-step
per-family work the epic describes for a dense arch, because attention's four
primitives do not apply. Attention mixes tokens by a content-based `Q.K^T`
softmax over a KV cache that grows with position; a recurrent mixer folds each
token into a fixed-size state by a linear recurrence. The concrete differences:

| Aspect | Attention (today) | Recurrent mixer (new) |
|--------|-------------------|-----------------------|
| Per-token mixing | `softmax(QK^T/sqrt d) V` | `h_t = a_t (.) h_{t-1} + b_t x_t; y_t = C_t . h_t` |
| State shape | KV grows `[.., S, nkv, d]` with position | fixed `[.., n_heads, head_dim, state_dim]`, no `S` |
| Decode step | append 1 K/V row, attend over `0..cache_len` | one recurrence step, read-modify-write the state |
| Prefill | one masked `QK^T` matmul over the prompt | a SCAN over the prompt (the hard graph problem) |
| Position handling | RoPE on Q/K | none (recurrence is inherently ordered) |
| Cross-token op | `dot_general` (already lowers on all targets) | selective scan / associative scan (must be validated) |

### The mixing primitives, grouped

The nine target families reduce to three primitive families. Each becomes a
shared graph module alongside `moe.rs`, parameterized by a config struct, called
from one per-layer mixer-dispatch site:

1. Selective SSM / SSD (Mamba, Mamba2, and the SSM layers of Nemotron-H, Jamba,
   Falcon-H1, Plamo2). A causal depthwise `conv1d` over a short kernel, then a
   selective scan with per-token, input-dependent `(A, B, C, dt)`:
   `dt = softplus(...)`, `A = -exp(A_log)`, discretize, then
   `h_t = exp(dt A) (.) h_{t-1} + (dt B) x_t`, `y_t = C h_t + D x_t`, with a gated
   output (`silu`). Mamba2's SSD form makes the scan a chunked matmul (see
   Prefill below).
2. Gated linear attention / delta rule (Qwen3-Next GatedDeltaNet under
   `gated_delta` / `qwen3_next`, Kimi-Linear, RWKV7). A matrix-valued state
   `S_t = diag(a_t) S_{t-1} + b_t k_t^T` (or the RWKV7 / delta-rule variant),
   `y_t = q_t S_t`. Same threading shape as the SSM state (fixed, per head), a
   different recurrence kernel.
3. Real-gated linear recurrent unit (RecurrentGemma / Griffin RG-LRU): a diagonal
   gated linear recurrence interleaved with LOCAL sliding-window attention (the
   sliding mask #495 already emits).

All three share the abstraction this design targets: a per-token linear
recurrence over a fixed-size state, with a parallel/chunked prefill form and a
one-step decode form. The follow-up epic lands them one family at a time; the
first (Mamba2) pays the primitive + state-carry + shim cost, the rest reuse it.

### New graph primitives (the `emitter` work)

- `conv1d`: causal depthwise 1D conv. Prefill is a windowed matmul; decode is a
  one-step dot over a small per-channel ring buffer that is itself threaded state
  (the last `kernel-1` inputs).
- `selective_scan` (decode form): the one-step recurrence above, cheap, reads and
  writes the SSM state.
- `selective_scan` (prefill form): the SCAN over the prompt, the one genuinely new
  graph problem (mirrors "ragged KV write" in Stage 2a). Options, to be resolved
  in a spike:
  - Sequential unrolled scan over the 256-token prefill bucket (`PREFILL_LP`,
    `iree.rs:75`). Simplest, but a 256-deep unroll per SSM layer may blow up graph
    size / compile time. Measure first, exactly as Stage 2a measured the
    `B_max * L` KV-write unroll.
  - Chunked SSD (Mamba2): decompose into chunks; within-chunk is a masked matmul
    (reusing the `dot_general` that already lowers), across-chunk is a short
    sequential scan of chunk-summary states. The recommended prefill form.
  - A log-depth associative scan built from `dot_general` / slices. Only if the
    chunked form does not lower on IREE-CUDA / Metal.
- Config additions: an `SsmConfig` (conv kernel, state dim, head/group counts,
  `dt` rank, activation) alongside `MoeConfig`, plus a per-layer mixer schedule
  `layer_mixers: Vec<MixerKind>` (mirroring how `use_rope_layers` /
  `sliding_pattern` already carry per-layer attention variation) so
  `emit_transformer_layer` can branch attention vs recurrent per `li`.

## State handling: allocation and per-slot carry, distinct from the KV cache

This is the load-bearing part of the design and the reason the track is its own
window. The KV cache and the recurrent state are BOTH per-slot device buffers
threaded through the functional graph, but they differ in a way that matters:

- The KV cache GROWS with position. Each decode step appends one K/V row at
  `pos[b]`; the shim holds `[B,L,S_max,nkv,d]` and the ragged decode reads
  `0..cache_len[b]`. Its size is `O(S_max)` per slot.
- The recurrent state is FIXED. `h` is `[.., n_heads, head_dim, state_dim]` and
  the conv ring buffer is `[.., kernel-1, channels]`; neither depends on how many
  tokens have been seen. Its size is `O(1)` per slot. That O(1) footprint is the
  whole point of SSM inference and it is why paged-KV (STAGE2_DESIGN 2d) is
  irrelevant to the recurrent layers.

So the state is not "another KV cache with different numbers"; it is a separate
resource with its own shape and its own lifecycle. The design threads it through
the SAME mechanism the KV cache already uses, as parallel buffers:

- Shim: add `sscache` / `sscache_b` (single-seq and rank-N per-slot) next to
  `kcache_b` / `vcache_b` (`csrc/xla_iree.c:98`), and an
  `xla_ensure_batch_state` mirroring `xla_ensure_batch_kv` (`csrc/xla_iree.c:390`)
  to allocate the per-slot state region zeroed on first use. The state has its own
  `[L_ssm, ...]` dims (only recurrent layers own state), independent of the KV
  dims.
- Graph: the recurrent-mixer graphs take the state as explicit in/out args and
  return the advanced state, exactly as the decode graph threads `kcache` in and
  the new KV out (`csrc/xla_iree.c:345-359` swaps the buffer after each call). No
  new threading concept, just a second threaded tensor.
- Admit (prefill into a slot): `prefill_slot` writes the slot's KV region
  device-side today; it additionally writes the slot's STATE region (the prefill
  scan's final `h` and the conv tail), leaving live slots untouched. The
  "untouched live slots" invariant that makes mid-stream admit safe carries over
  unchanged.
- Decode (ragged step): each row reads its own state, runs the one-step
  recurrence, and writes its state back. This is naturally ragged with NO extra
  masking subtlety, because each slot's recurrence is independent of its peers
  (unlike the KV path, where the shared `S` axis needs the per-row `cache_len`
  mask). An inactive slot is a zeroed state advanced by a dummy token; its output
  is discarded, same as the KV path.
- Evict: freeing a slot leaves its state region to be re-zeroed on the next
  admit, same lifecycle as the KV region.

The fixed-size, snapshot/restore shape is exactly what the MLX `Mamba2Cache`
(`src/models/mamba2.rs`, `snapshot_into` / `restore_from`) already does per
sequence; the OpenXLA path moves the same O(1) state onto the device and keys it
by slot. The host-side `Slot` (`batch.rs:70`) needs NO new fields: the state lives
on the device, keyed by slot index, exactly like the KV cache which the host also
never touches. `cache_len` still tracks logical position (for RoPE on any attention
layers, and for length/stop bookkeeping) even though the recurrent layers ignore
it. `RAGGED_B_VALUES` (`iree.rs:138`) and the fixed-`B_max` compile model are
unchanged.

## Hybrid coexistence with KV-cache attention layers

Jamba, Nemotron-H, Qwen3-Next, Falcon-H1, Kimi-Linear, and RecurrentGemma
interleave attention layers with recurrent layers (and often MoE FFN). The
emitter already dispatches the FFN per layer (dense vs MoE, `emit_ffn_body`); the
design adds the same per-layer dispatch for the MIXER:

- `emit_transformer_layer` (`emitter/model.rs:1563`) branches on
  `Config::layer_mixers[li]`: an attention layer runs the existing `AttnLayout`
  path and writes the KV cache; a recurrent layer runs the new mixer primitive and
  writes the SSM state. FFN dispatch (dense / MoE) is orthogonal and unchanged, so
  a Jamba layer is (recurrent-or-attention mixer) + (dense-or-MoE FFN) chosen
  independently.
- A hybrid slot therefore carries BOTH resources at once: a KV region written
  only by the attention layers, and a state region written only by the recurrent
  layers. Both are threaded per-slot in parallel. This is the concrete meaning of
  "state handling distinct from the KV cache": the two are separate device buffers
  with separate shapes, separate growth semantics, and separate per-slot seeding,
  advanced side by side in one `pump` step.
- The KV cache for a hybrid model is smaller than a pure-attention model of the
  same depth (only some layers have it), which is a memory win the layout gets for
  free by sizing `[B, L_attn, S, nkv, d]` over just the attention layers.

## Validation plan

Reuse the epic's gate, extended to cover the state buffer:

1. HF fp32 oracle (transformers Mamba2 / Nemotron-H) on a real small checkpoint.
2. XLA on the bf16 checkpoint (dequant offline if the only checkpoint is quantized,
   per the epic's validation pattern).
3. Single-sequence path token-exact vs the oracle (`MLXCEL_BACKEND=xla` CLI).
4. Serve path reference-exact via the continuous-batching harness: N staggered
   prompts each match their independent single-seq reference. This is the Stage 2a
   reference-equivalence gate (STAGE2_DESIGN "2a validation"), which now also
   proves the SSM STATE carry is correct: a slot's output must be invariant to
   which peers share its batch and when it joined, so the state seeding on admit,
   the per-row carry on decode, and the zeroing on evict are all covered by the
   same assertion.

Numerical note: the selective scan accumulates in a recurrence, so it is more
sensitive to precision than a single matmul. Expect to run the scan accumulation
in f32 even on the f16 GPU precision path (`resolve_precision`, used at
`iree.rs:669`), and to allow the epic's sub-0.01-logit near-tie tolerance.

## Reference architecture recommendation

Mamba2 first. It is a pure SSM (no hybrid dispatch needed yet), it has the
best-documented parallel form (SSD), and small HF checkpoints exist for a real
oracle. Landing Mamba2 pays the whole new cost once: the `conv1d` +
`selective_scan` primitives, the `SsmConfig`, the shim state buffers +
`xla_ensure_batch_state`, and the state-aware validation harness. Then:

- Nemotron-H (Mamba2 + attention + MLP/MoE) validates the hybrid per-layer mixer
  dispatch and dual KV+state carry with no new primitive.
- Jamba / Falcon-H1 / Plamo2 reuse the SSM primitive + hybrid dispatch.
- Qwen3-Next (GatedDeltaNet), Kimi-Linear, and RWKV7 each add a linear-attention /
  delta-rule recurrence kernel on the same state-carry machinery.
- RecurrentGemma adds the RG-LRU on the same machinery, over the existing sliding
  mask.

Sequencing caution: several targets (Qwen3-Next, Kimi-Linear) are recent and
RWKV7's WKV7 is its own kernel; treat them as later phases behind the Mamba2 /
Nemotron-H foundation rather than blind targets.

## Risks and open questions

- Prefill scan lowering. The chunked-SSD / associative-scan form must lower on
  IREE-CUDA (sm_121 / GB10) and IREE-Metal. STAGE2_DESIGN already found IREE-CUDA
  rejected a scatter that range-slice writes lowered to; the scan must be
  spike-validated before relying on it (Stage-2a-style spike-first).
- Graph size of an unrolled scan (256-deep per SSM layer). Measure compile time and
  per-step cost; fall back to the chunked form or a hybrid.
- Numerical stability of the recurrence in reduced precision (see the validation
  note); may force f32 scan accumulation.
- State + KV coexistence sizing on unified memory (GB10): fixed state is cheap, but
  the dual-resource allocation and the device-side dual slot-seed copy add shim
  surface that must be measured.
- Interaction with the whole-prefix prompt cache: like the MLX carve-out, the
  OpenXLA recurrent path must not participate in prefix-hash sharing; a saved
  prefix cannot restore the recurrent state. The shim state is per-slot and never
  hashed, so this is preserved by construction, but it must be asserted.

## Staged plan (for the follow-up epic #564)

- Spike: the selective-scan graph (decode step + chunked prefill), the conv1d, and
  their IREE-CUDA / Metal lowering + a token-exact single-layer probe vs an HF
  reference. Mirrors Stage 2a.
- Primitive: `emitter/ssm.rs` (`conv1d` + `selective_scan`), `SsmConfig`, the
  per-layer `layer_mixers` schedule, `weight_specs` for the SSM tensors.
- State carry: shim `sscache` / `sscache_b` + `xla_ensure_batch_state`, the
  state-threaded prefill-slot and ragged-decode graphs, the device-side state slot
  write.
- Reference: Mamba2 end to end (CLI token-exact + serve reference-exact), the
  state-aware validation harness.
- Hybrid: Nemotron-H (per-layer mixer dispatch, dual KV+state carry).
- Breadth: Jamba, Falcon-H1, Plamo2, then the linear-attention / delta families
  (Qwen3-Next, Kimi-Linear, RWKV7) and RecurrentGemma, one family per unit.

## Decision and rationale

Deferred to the follow-up epic. A first reference (Mamba2) requires, before it can
generate a single token: a new scan graph primitive whose prefill form is an
unsolved lowering question, a new C-ABI state resource with its own allocation and
device-side per-slot seeding, a new per-layer mixer dispatch, and a new
state-aware validation harness. That is comparable in size to the entire Stage 2
continuous-batching effort (spike + engine + shim + serve), not to the cheap
per-family dense work. Acceptance for issue #502 explicitly allows an "explicit,
recorded decision to defer with the follow-up epic created", and that is the
sanctioned path here. This document is that record; the follow-up epic #564
carries the staged plan above.
