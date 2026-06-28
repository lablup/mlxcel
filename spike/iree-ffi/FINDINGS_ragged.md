# Ragged (continuous-batching) decode graph (#449 M3 Stage 2a-i)

The first half of Stage 2: a **ragged** `decode_step` where each row carries its
OWN position and length, so sequences of different lengths share one batch. This
is the graph mechanic continuous batching needs. Validated by
reference-equivalence: every ragged slot reproduces its independent single-seq
stream, token-exact, on CPU and CUDA.

Verdict: **PASS.** The ragged graph is correct and lowers/runs on both targets;
a sequence's output is invariant to its batch-mates and their lengths. The
per-row KV-write unroll costs only a few percent over the uniform-B graph.

Scope note: this is 2a-i (the graph). The admit/evict scheduler with staggered
arrival and slot recycling (2a-ii) is the remaining Stage 2a piece; here all
slots are admitted at t=0 and stepped together at their own positions.

## What was built (all in the spike)

| Piece | Where |
|-------|-------|
| Ragged graph | `rust-emitter` `model.rs::emit_decode_ragged` + CLI `decode-ragged[-argmax] <B>`. Per-row `pos[B]`/`cache_len[B]`; RoPE cos/sin per-row gather `[B,d]` by `pos[B]`; key mask per-row `[B,S]` (valid iff `s <= cache_len[b]`); KV write unrolled per row, each at its own `pos[b]`. Attention contractions + LM head identical to uniform-B; the per-row mask carries the raggedness. |
| Shim | `iree_gate.c`: `xla_llama_decode_ragged` (token/pos/cache_len as `[B]` arrays, threads rank-5 KV). Prefill-into-slot via a host KV mirror: `xla_llama_ragged_reset` + `xla_llama_prefill_slot` (single-seq prefill, d2h into mirror slot r) + `xla_llama_commit` (h2d mirror -> resident rank-5 KV). Reuses the scalar prefill vmfb. |
| Driver | `src/bin/llama_ragged.rs`: Phase 1 captures one single-seq reference per prompt; Phase 2 prefills the prompts into slots and runs ragged decode; asserts each slot == its reference. |

## The one hard problem, resolved

Each row writes its new K/V at its own `pos[b]`, so the uniform-B single
shared-offset `dynamic_update_slice` no longer applies. Resolved with the
per-row `dynamic_update_slice` unroll (primary plan from the design): for each
layer, `B` writes, row r writing `[1,1,1,nkv,d]` at `[r, li, pos[r], 0, 0]`
(constant row/layer offsets, dynamic `pos[r]`). This uses the CUDA-proven
`dynamic_update_slice` (scatter stays avoided per the Phase 2a caveat). It lowers
and runs on both targets; graph size and compile time are fine (see below).

## Validation: reference-equivalence

Continuous batching has no single reference, so the gate is: B prompts of
DIFFERENT lengths (truncations of the 46-token reference prompt), each run
independently through the single-seq path to get its greedy stream, then run
together in one ragged batch (each slot at its own position). Every slot's stream
must match its independent reference, proving per-row state is correct and a
sequence is unaffected by its batch-mates.

| Device | B | Prompt lengths | Result | ragged ms/step | agg tok/s | vs uniform-B ms/step |
|--------|---|----------------|--------|----------------|-----------|----------------------|
| CPU `local-task` | 4 | 46,43,40,37 | 4/4 slots 48/48 EXACT | 115.2 | 34.7 | 110.0 (+4.7%) |
| CUDA GB10 | 4 | 46,43,40,37 | 4/4 slots 48/48 EXACT | 89.2 | 44.8 | 87.5 (+1.9%) |
| CUDA GB10 | 8 | 46,43,40,37,34,31,28,25 | 8/8 slots 48/48 EXACT | 95.7 | 83.6 | 89.0 (+7.5%) |

## Findings

- **Ragged decode is correct.** Eight sequences of eight different lengths in one
  batch each reproduce their independent single-seq stream exactly. Per-row
  pos/cache_len/mask and the per-row KV write are all right; batch membership and
  peer lengths do not perturb a sequence.
- **The per-row KV-write unroll is cheap.** B*L `dynamic_update_slice`s per step
  (e.g. 128 at B=8) add only ~2% (B=4) to ~7.5% (B=8) over the uniform-B graph,
  and IREE lowers them fine on CPU and CUDA. Compile time stayed small (B=4 ~7.5s
  CPU, ~5s CUDA).
- **CPU validation is reference-bound, not graph-bound.** Phase 1 runs B
  single-seq references sequentially; at ~0.45 s/step that dominates CPU runtime
  for large B (B=8 CPU exceeds a few minutes). It is a harness cost, not a ragged
  graph cost; CUDA references are ~2.5x faster, so the B sweep runs there.

## Next (2a-ii)

The admit/evict scheduler: a request queue, slot allocation, mid-stream prefill
into a freed slot (refresh the device KV to the mirror, prefill the slot, re-commit),
eviction on EOS, and slot recycling. Validation extends to staggered arrival
(a request's stream must match its reference regardless of when it joined or which
peers came and went). Then 2b productizes the engine into `mlxcel-xla`. See
`spike/openxla/STAGE2_DESIGN.md`.

## Reproduce

```bash
cd spike/iree-ffi
# CPU: compile prefill + single-seq decode + ragged decode, then run.
EMIT=../rust-emitter/target/release/emit; IC=./iree-dist/bin/iree-compile
F="--iree-input-type=stablehlo --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu"
$EMIT prefill-argmax p.mlir && $IC $F p.mlir -o p.vmfb
$EMIT decode-argmax s.mlir  && $IC $F s.mlir -o s.vmfb
$EMIT decode-ragged-argmax 4 r.mlir && $IC $F r.mlir -o r.vmfb
IREE_DIST=./iree-dist cargo run --release --bin llama_ragged -- \
  --batch 4 --device local-task --prefill p.vmfb --sdecode s.vmfb --decode r.vmfb
# CUDA: build with IREE_CUDA_HOME, compile with the pip cuda iree-compile
# (--iree-hal-target-device=cuda), run with --device cuda and the .cuda.vmfb graphs.
```
