# Stage 2 design: continuous batching for the OpenXLA/IREE backend (#449 M3)

Status: Proposed (2026-06-28). Follows Stage 1 (uniform-B batched decode, PR #462).
Graduates to an ADR (a follow-up to ADR 0004) when productization (2b/2c) lands.

## Goal

Move the XLA/IREE backend from single-sequence (and Stage 1's uniform-B lockstep)
to real continuous batching: sequences of different lengths that join and leave a
shared batch at different times, so the device stays full under a request stream.
Stage 1 proved batching is the throughput lever here (CPU 31x, CUDA 39x aggregate,
token-exact). Stage 2 turns that into a servable multi-sequence engine.

## Decisions (resolved 2026-06-28)

1. **Spike-first.** Validate the ragged decode graph + a minimal scheduler in the
   spike (2a) before productizing into `mlxcel-xla` (2b+). Mirrors Stage 1 and
   de-risks the one genuinely new graph problem (ragged KV write).
2. **Common `BatchEngine` trait.** The server drives one backend-neutral batching
   contract; both the MLX `BatchScheduler` and the new XLA engine satisfy it, so
   the server is backend-agnostic. (This is the cross-backend batching seam ADR
   0004 deferred.)
3. **Contiguous per-slot KV first.** Extend Stage 1's rank-5 `[B,L,S,nkv,d]` to
   recycled slots. Paged-KV is a later capacity optimization (2d), not in the
   first servable cut.

## Background: the existing server seam (what we build on)

The server is already backend-neutral at the request level, which makes decision
2 cheap:

- `ModelRequest` (`src/server/model_provider.rs:35`, `pub(crate) enum`):
  `Generate { prompt, options, images, audio, videos, response_tx:
  mpsc::Sender<GenerateEvent>, cancelled: Arc<AtomicBool> }` or `Shutdown`.
  No MLX types: prompt + options + multimodal bytes + a streaming channel + a
  cancel flag.
- `GenerateEvent`: `Token | TokenWithLogprobs | Done(GenerationResult) | Error`.
- `ModelProvider` holds `request_tx: mpsc::Sender<ModelRequest>` and a worker
  thread. It spawns either the MLX `BatchScheduler` (`scheduler.rs`, `run(&mut
  self)` pulling `request_rx: mpsc::Receiver<ModelRequest>`) or the legacy
  sequential worker. The HTTP handlers send a `ModelRequest` and read
  `GenerateEvent`s off `response_tx`.

So the worker contract is already "own a model + KV/scheduling, consume
`ModelRequest`, stream `GenerateEvent` per request." The MLX `BatchScheduler` IS
this. The XLA engine becomes an alternative worker of the same shape.

`InferenceSession` (`mlxcel-core/src/session.rs`) is intentionally single-sequence
and is NOT extended here. ADR 0004 keeps batching out of that trait; Stage 2 adds a
separate, server-level batching seam, consistent with how MLX already works.

## 2a (spike): the ragged continuous decode graph

The single decode graph gains per-row state. Versus Stage 1's uniform-B:

| Aspect | Stage 1 (uniform-B) | Stage 2 (ragged) |
|--------|---------------------|------------------|
| `pos` | scalar (shared) | `[B] i32` (per row) |
| `cache_len` | scalar (shared) | `[B] i32` (per row) |
| RoPE cos/sin | one `[d]` broadcast | per-row gather `[B, d]` from the table by `pos[B]` |
| key mask | shared `[S]` | per-row `[B, S]`: valid iff `s <= cache_len[b]` |
| KV write | one `dynamic_update_slice`/layer at the shared offset | per-row offset (see below) |
| attention `dot_general` | two-batch-dim | unchanged (mask carries the raggedness) |

The expensive part (the GQA score/context contractions and the LM head) is
unchanged; only the cheap head/mask/write change. RoPE per-row uses the same
gather primitive Stage 1 already uses for embedding.

### The one hard problem: ragged KV write

Each row writes its new K/V at its own `pos[b]`, so Stage 1's single shared-offset
`dynamic_update_slice` no longer applies. Options:

- **(A) per-row `dynamic_update_slice` unroll (primary).** For each layer, unroll
  `B_max` writes: slice row b's new K/V to `[1,1,1,nkv,d]`, slice `pos[b]` to a
  scalar, `dynamic_update_slice(kcache, upd_b, [b, li, pos[b], 0, 0])`. Uses the
  CUDA-proven `dynamic_update_slice` (Stage 1 + M2). Cost: `B_max * L` update ops
  per step (e.g. 32*16 = 512), so graph size and compile time must be measured.
- **(B) scatter (alternative to test).** One scatter per layer writing all rows at
  `pos[B]`. Cleaner if it lowers, but Phase 2a found IREE-CUDA rejected the scatter
  that range-slice writes lowered to, so this needs an explicit lowering check
  before relying on it.

2a validates (A); if graph size is a problem, evaluate (B) or a hybrid.

### Static shape with a varying active count

The graph is compiled for a fixed `B_max`. Fewer than `B_max` sequences may be
active, so inactive slots are masked (a slot's `cache_len[b]` and a dummy token
make its row a harmless no-op whose output is discarded). One graph, no
recompilation as the batch fills and drains.

Stage 1 constraint carried forward: the untuned IREE-CUDA codegen is non-monotonic
(B=16 was a reproducible 12x cliff; B=8/24/64 fine). So `B_max` must be chosen from
empirically good values, and 2a re-measures the chosen `B_max` for the ragged
graph. Bucketed `B_max` (a few graphs) stays a 2d option.

### Prefill

Separate prefill first: a new request runs the existing prefill graph, and its KV
is written into a free slot (positions `0..prompt_len`), after which the slot joins
the decode batch. Chunked prefill (prefill tokens riding in a decode step) is a 2d
utilization optimization, not in the first cut.

### KV layout

Contiguous per-slot, `[B_max, L, S_max, nkv, d]` f32, slots recycled on evict.
Memory at B_max=32, S_max=2048: ~4.3 GB (k+v), comfortable on GB10 unified memory
next to the ~4.9 GB f32 weights. S_max=8192 would be ~17 GB, which is where paged-KV
(2d) earns its complexity.

### 2a validation

Continuous batching has no single reference, so:

1. **Correctness (the strong gate):** run N distinct prompts (varied lengths) each
   independently through the single-seq path to capture reference token streams.
   Then drive them through the continuous harness with staggered arrival and a
   given `B_max`, and assert each request's output stream EXACTLY matches its
   independent reference. This proves a sequence's output is invariant to which
   peers share its batch and when it joined or others left (ragged state, masking,
   slot recycling all correct).
2. **Throughput:** sustained tok/s under a saturating request stream (queue always
   full) versus the single-seq baseline, on CPU then CUDA.

Harness: extend the Stage 1 `llama_batch` driver into a small scheduler that owns
`B_max` slots, a FIFO queue, and an admit/decode/evict loop, plus the shim
functions for per-row prefill-into-slot and ragged decode.

## 2b (productize the engine): `XlaBatchEngine` in `mlxcel-xla`

A multi-sequence engine that owns `B_max` slots, the rank-5 KV, the batched +
prefill vmfbs, and the admit/decode/evict loop. It exposes a backend-neutral,
request-level mechanics API (submit a prompt + sampling/limit params, pump a step,
receive per-slot token/finish events, cancel a slot). No dependency on server
types (it is a lib crate). Contiguous KV, fixed `B_max`, greedy first (sampling
parity is a follow-up). Shipped with a CLI/bench entry so it is provable without
the server.

## 2c (server integration): the common `BatchEngine` trait

Constraint: `ModelRequest`/`GenerateEvent` live in the server binary; `mlxcel-xla`
is a lib, so it cannot implement a trait written over those types directly. So:

- Define `trait BatchEngine: Send` at the server seam (where `ModelRequest`
  lives): `fn run(&mut self)` consuming the `mpsc::Receiver<ModelRequest>` given at
  construction and streaming `GenerateEvent`s per request. (Shutdown via
  `ModelRequest::Shutdown`, cancel via the request's `cancelled` flag.)
- MLX `BatchScheduler` `impl BatchEngine` (it already has `run(&mut self)` and
  `request_rx`; near-zero change).
- A server-side `XlaServeWorker` `impl BatchEngine` translates `ModelRequest`
  to/from the `mlxcel-xla` engine mechanics API and pumps its loop. The engine
  mechanics stay in the lib; only the thin adapter touches server types.
- `ModelProvider` constructs and spawns the right worker by backend
  (`select_backend()` / `ComputeBackend::supports_batched_serving()`), holding a
  `Box<dyn BatchEngine>`. The HTTP path is unchanged.

This gives the server one batching contract (decision 2) without moving core
server types or destabilizing the MLX path. If we later want `mlxcel-xla` to
implement the trait itself, promoting the neutral request/event types + the trait
into `mlxcel-core` is a clean follow-up, but it is not required for 2c.

`XlaBackend::supports_batched_serving()` flips to true once 2c lands; today it is
false and the server uses the single-seq session path for XLA.

## 2d (optimizations, post-first-cut)

- **Paged-KV** (block table): memory efficiency, long context, more concurrent
  sequences. Needs block-wise gather/scatter in the graph (IREE-CUDA lowering must
  be validated, given the Phase 2a scatter caveat).
- **Chunked prefill / mixed batches:** prefill tokens riding in a decode step for
  higher utilization and smoother latency.
- **Bucketed `B_max`** to cut idle-slot waste when underfull (pick known-good B
  values to dodge codegen cliffs).
- Sampling beyond greedy; config.json-driven multi-architecture emit.

## Risks and open questions

- **Graph size of the per-row KV-write unroll** (`B_max * L` updates). Measure
  compile time and per-step cost in 2a; fall back to scatter or a hybrid if needed.
- **IREE-CUDA codegen non-monotonicity** at the ragged graph's shapes. Re-sweep
  `B_max` for the ragged graph; do not assume Stage 1's good points transfer.
- **Inactive-slot masking correctness** (a half-full batch must match a full one
  for the active rows). Covered by the 2a reference-equivalence gate.
- **Trait altitude**: `run`-loop trait (proposed) vs a finer submit/step/poll trait.
  The run-loop matches the existing MLX scheduler exactly; a finer trait would
  force refactoring the MLX loop. Proposed: run-loop now, revisit only if a finer
  seam is needed.

## Staged plan

- **2a** spike: ragged decode graph (per-row pos/cache_len/mask, ragged KV write),
  minimal scheduler harness, reference-equivalence + throughput validation, CPU
  then CUDA.
- **2b** productize: `XlaBatchEngine` in `mlxcel-xla` (contiguous KV, fixed B_max,
  greedy), CLI/bench.
- **2c** integrate: `BatchEngine` trait, MLX `impl`, `XlaServeWorker` adapter,
  `ModelProvider` selection, flip `supports_batched_serving`.
- **2d** optimize: paged-KV, chunked prefill, bucketed B_max, sampling.
