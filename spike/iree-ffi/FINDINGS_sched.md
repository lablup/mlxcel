# Continuous-batching scheduler (#449 M3 Stage 2a-ii)

The second half of Stage 2a: a minimal continuous-batching scheduler over the
ragged decode graph (2a-i). `B` slots serve `N > B` requests that finish at
staggered times, so queued requests are admitted mid-stream into freed slots
(slot recycling). Validated by reference-equivalence under dynamic membership.

Verdict: **PASS.** Every request's output matches its independent single-seq
reference regardless of when it was admitted or which peers shared its batch.
With 2a-i (the graph) this completes Stage 2a: continuous batching works on the
XLA/IREE backend, token-exact, on CPU and CUDA.

## What was built

| Piece | Where |
|-------|-------|
| Mid-stream admit | `iree_gate.c::xla_llama_refresh_mirror` pulls the live rank-5 KV of all active slots back to the host mirror (d2h). Admit = refresh + `prefill_slot(freed)` + `commit`: refresh captures active slots' advanced state, prefill_slot overwrites only the freed slot, commit re-uploads, so admitting one sequence does not disturb the others. |
| Scheduler | `src/bin/llama_sched.rs`: `B` slots, a FIFO queue of `N` requests with varied prompt lengths AND varied token caps. Loop: admit queued into free slots, ragged-decode all active slots in lockstep, evict at cap/EOS, recycle the slot. Inactive slots decode harmlessly (token 0, masked, output ignored; overwritten on the next admit). |

## Validation: reference-equivalence under recycling

Phase 1 captures each request's capped single-seq reference. Phase 2 runs the
scheduler; each request's collected stream must equal its reference. Because caps
vary, requests evict at different steps and later requests admit mid-stream while
others are still decoding, so a passing run proves refresh/commit preserves the
active slots and a recycled slot starts clean.

| Device | B | N | varied lengths / caps | admits | result | throughput |
|--------|---|---|-----------------------|--------|--------|------------|
| CPU `local-task` | 3 | 6 | lengths 20/33, caps 12-14 | 6 (3 initial + 3 mid-stream) | 6/6 reference-exact | 5.5 tok/s |
| CUDA GB10 | 4 | 8 | lengths 20/33, caps 12-39 | 8 (4 initial + 4 mid-stream) | 8/8 reference-exact | 17.8 tok/s |

The mid-stream admits (3 on CPU, 4 on CUDA) each ran `refresh_mirror` while other
slots were live; all sequences still matched their references, so the
refresh+commit round-trip is correct.

## Findings

- **Continuous batching is correct.** Admit, evict, slot recycling, and
  mid-stream join over the ragged graph all preserve per-request output: each
  sequence is identical to its standalone run.
- **The host-mirror admit is correct but not the throughput path.** Each admit
  does a full rank-5 KV round-trip (d2h refresh + h2d commit). That is fine for a
  correctness spike, but at this small scale it dominates, so the tok/s here is
  not a throughput result (Stage 1 already measured raw batched throughput; 2a-ii
  measures correctness). 2b should replace the round-trip with a device-side slot
  write (write only the admitted slot's region, no whole-cache transfer) and
  batch admits.
- **Static `B` with masked inactive slots works.** A half-full batch produces the
  same per-request output as a full one; inactive rows do not perturb active ones.

## Stage 2a status

Complete. 2a-i proved the ragged decode graph (different lengths in one batch);
2a-ii proves the scheduler (dynamic membership). Next is 2b: productize an
`XlaBatchEngine` in `mlxcel-xla` (contiguous KV, fixed `B_max`, a device-side
slot write instead of the host-mirror round-trip), then 2c the common
`BatchEngine` trait + server integration. See `spike/openxla/STAGE2_DESIGN.md`.

## Reproduce

```bash
cd spike/iree-ffi
# build the ragged + single-seq + prefill vmfbs (see FINDINGS_ragged.md), plus a
# dr<B>.vmfb for the slot count, then:
IREE_DIST=./iree-dist cargo run --release --bin llama_sched -- \
  --batch 3 --requests 6 --maxcap 14 --device local-task \
  --prefill p.vmfb --sdecode s.vmfb --decode dr3.vmfb
# CUDA: build with IREE_CUDA_HOME, run with --device cuda + the .cuda.vmfb graphs,
#       e.g. --batch 4 --requests 8 --maxcap 48.
```
