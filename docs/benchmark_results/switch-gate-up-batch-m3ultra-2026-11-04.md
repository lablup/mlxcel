# Switch gate/up kernel batch decode A/B on M3 Ultra, 2026-11-04

Issue #2515 follow-up. This re-measures the `moe_switch_gate_up` Metal kernel against the `gather_qmm` path at batch decode on `models/olmoe-1b-7b-0125-4bit` (64 experts, top_k 8, hidden 2048, expert intermediate 1024, affine q4 group 64, f16 activations and scales), and it corrects the batch geometry of the 2026-10-21 record.

The earlier record ran `mlxcel-server` at its default `--parallel 4`, so its concurrency 5 to 8 levels queued behind a four-row decode batch: above four it did not measure the batch sizes its table named, and its "batch 8 control cell" ran two waves of four rows, not 64 routes. This run sets `--parallel 8 --ctx-size 8192`, which splits the context into eight 1024-token slots, so concurrency B reaches a decode batch of B rows and the scheduler's own occupancy counter confirms it (`--metrics`). Prompt cache and APC are off so the eight concurrent clients do not start decoding at different times after shared-prefix reuse.

## What was measured

- Arm A: default environment. At B = 2 to 7 (`tokens * top_k` = 16 to 56, below the sorted-path threshold) `SwitchGLU::forward` tries the switch gate/up kernel. At B = 8 (64 routes) both arms take the sorted path.
- Arm B: `MLXCEL_FUSED_MOE=0`. The kernel declines and both projections take the indexed `gather_qmm` path. The #268 fused single-token kernel cannot fire at B >= 2, so the environment switch isolates the switch gate/up kernel.
- Six server boots in alternating order (A, B, A, B, A, B), one discarded full-ladder warm-up pass per boot, then one measured pass. Three measured passes per arm.
- Ladder 2, 3, 4, 5, 6, 7, 8; nominal 512-token prompts; 128 generated tokens; temperature 0; greedy; one fresh server process per boot; `caffeinate -i` and `scripts/with_indexers_paused.sh`.

| B | routes | arm A kernel | arm B kernel |
|---:|---:|---|---|
| 2 | 16 | switch gate/up | gather |
| 3 | 24 | switch gate/up | gather |
| 4 | 32 | switch gate/up | gather |
| 5 | 40 | switch gate/up | gather |
| 6 | 48 | switch gate/up | gather |
| 7 | 56 | switch gate/up | gather |
| 8 | 64 | sorted path (control) | sorted path (control) |

Per-arm tables report the median of the three measured passes and the spread `(max - min) / median`. The B = 8 row is the control: both arms run the same code there, so its delta is the protocol's noise floor.

## Per-stream decode throughput

| B | kernel median | kernel spread | gather median | gather spread | kernel vs gather |
|---:|---:|---:|---:|---:|---:|
| 2 | 233.0 | 0.5% | 241.8 | 6.4% | -3.6% |
| 3 | 180.4 | 0.7% | 195.2 | 0.4% | -7.6% |
| 4 | 154.6 | 0.5% | 162.2 | 0.5% | -4.7% |
| 5 | 128.6 | 0.8% | 137.9 | 0.2% | -6.7% |
| 6 | 114.5 | 0.4% | 122.3 | 0.3% | -6.4% |
| 7 | 101.8 | 0.2% | 108.9 | 0.0% | -6.5% |
| 8 (control) | 95.0 | 1.9% | 94.4 | 0.1% | +0.6% |

The gather arm's 6.4% spread at B = 2 is one low first run (226.7 tok/s) against 241.8 and 242.2 for the other two; the kernel arm is 232.1 to 233.2 across all three. Every other cell is under 2%.

## Aggregate throughput

| B | kernel median | kernel spread | gather median | gather spread | kernel vs gather |
|---:|---:|---:|---:|---:|---:|
| 2 | 390.1 | 0.6% | 402.3 | 5.5% | -3.0% |
| 3 | 453.8 | 0.6% | 484.2 | 0.3% | -6.3% |
| 4 | 510.9 | 0.4% | 531.3 | 0.4% | -3.8% |
| 5 | 531.9 | 0.7% | 562.1 | 0.2% | -5.4% |
| 6 | 562.7 | 0.3% | 592.6 | 0.3% | -5.1% |
| 7 | 582.7 | 0.0% | 614.1 | 0.1% | -5.1% |
| 8 (control) | 614.1 | 1.6% | 611.9 | 0.2% | +0.4% |

## Mean decode step and occupancy

| B | kernel step ms | gather step ms | kernel vs gather | kernel occupancy | gather occupancy |
|---:|---:|---:|---:|---:|---:|
| 2 | 4.30 | 4.14 | +3.9% | 2.00 | 2.00 |
| 3 | 5.56 | 5.14 | +8.2% | 3.00 | 3.00 |
| 4 | 6.50 | 6.20 | +4.8% | 4.00 | 4.00 |
| 5 | 7.82 | 7.30 | +7.1% | 5.00 | 5.00 |
| 6 | 8.80 | 8.25 | +6.7% | 6.00 | 6.00 |
| 7 | 9.90 | 9.27 | +6.8% | 7.00 | 7.00 |
| 8 (control) | 10.63 | 10.69 | -0.6% | 8.00 | 8.00 |

The step-time table and the per-stream table are independent views of the same decode: `1000 / step_ms` is the per-stream rate, and the two agree at every cell. Occupancy is exactly B in both arms at every level, with 127 decode steps and 127 * B advanced tokens per level, so the ladder measured the batch sizes it names.

Occupancy is `mlxcel_batch_decode_tokens_total / mlxcel_batch_decode_steps_total` over one concurrency level; it is the scheduler's own count of how many rows a decode step advanced. It is the check that client concurrency B produced a B-row batch.

## Time to first token

| B | kernel median ms | gather median ms |
|---:|---:|---:|
| 2 | 109.0 | 109.0 |
| 3 | 137.0 | 137.7 |
| 4 | 172.0 | 172.1 |
| 5 | 206.7 | 206.9 |
| 6 | 240.8 | 241.0 |
| 7 | 273.7 | 275.1 |
| 8 (control) | 309.6 | 308.7 |

Every level agrees within 0.6%, so the throughput difference below is decode arithmetic, not admission.

TTFT is not a kernel metric; it is reported so a decode difference is not confused with an admission or prefill difference. Cold prefills differ between levels because `--max-batch-prefill` defaults to 4.

## Parity at batch shapes

`examples/logit_trace` with `MLXCEL_TRACE_BATCH_ROWS=B` runs each traced position as a width-one `forward_batched_with_ids` decode over B independently prefixed rows, so the shared `SwitchGLU` receives `x [B, hidden]` and `indices [B, top_k]`, exactly the shape `forward` reduces with either path. The corpus is `docs/CONTINUOUS_BATCHING.md`, sixteen chunks of 16 teacher-forced positions (an initial four-chunk pass agreed at every level), top-k 8, in the same two environment arms as the throughput run (`MLXCEL_FUSED_MOE` unset against `=0`). `scripts/compare_logit_traces.py` treats the gather arm as the reference and the kernel arm as the candidate. "Decided" means a reference top-two logit gap of at least 2.0.

| B | top-1 disagreement | decided-position disagreement | logit delta on the reference's choice (p50 / p90 / p99 / max) | perplexity |
|---:|---|---|---|---|
| 2 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 3 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 4 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 5 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 6 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 7 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 8 (control) | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |

The shorter 64-position pass agrees at every level as well, so the parity result does not depend on the sample size. The two paths are indistinguishable at the trace's 4-decimal precision on this checkpoint at batch shapes 2 to 7, and the B = 8 control confirms the comparison itself does not manufacture a difference. This is a parity statement for OLMoE at these shapes and this corpus, not a general bit-identity claim: the kernel accumulates in f32 with a different reduction order from `gather_qmm`, and other checkpoints can flip near-tie greedy tokens for that reason.

## Reading

The kernel loses to `gather_qmm` at every multi-token batch size measured, on both per-stream and per-step views, by 3.6% to 7.6%. The loss is not monotonic in B (B = 3 is the worst cell, B = 4 the mildest), so it is not a single scheduling threshold; it is the kernel's threadgroup grid and route pairing costing more than the indexed gather at these route counts. TTFT is identical between arms at every level, so the difference is decode arithmetic only.

The B = 8 control bounds the method's bias at +0.6% per stream, +0.4% aggregate, -0.6% step time. Every measured loss at B = 2 to 7 is several times that floor. Within-arm spread is under 1% everywhere except the gather B = 2 rows (6.4% per-stream, 5.5% aggregate, from the one low first run) and the kernel B = 8 control row (1.9%).

### Replication under the earlier geometry

The 2026-10-21 record ran the default `--parallel 4`, so its concurrency 5 to 8 levels queued behind a four-row batch and were not the batch sizes its table named. A focused replication of its batch-2 and batch-4 cells at the current head (`--parallel 4`, prompt cache on, model-default context, three alternating boots per arm) reproduces the loss and rules out the corrected `--parallel 8` configuration as its cause:

| B | kernel median tok/s | gather median tok/s | kernel vs gather | kernel step ms | gather step ms |
|---:|---:|---:|---:|---:|---:|
| 2 | 271.6 | 283.9 | -4.3% | 3.67 | 3.52 |
| 4 | 172.9 | 186.9 | -7.5% | 5.80 | 5.39 |

The batch-2 cell matches that record's -4.0%. Its batch-4 cell read +0.4% and this one reads -7.5%; the kernel arm moved by +2.2% (172.9 against 169.2 tok/s), so the difference is on the gather side, which is about 11% faster at the current head than it was at `0f43e98e`. The earlier record's batch-3 and batch-4 cells did measure true 3- and 4-row batches, but both sat inside its noise floor; its batch-5 to 8 rows queued behind four rows and did not measure the sizes printed.

### Gate adjustment

The gate's M3+ exception rested on the 2026-10-21 record. That premise is wrong, so `switch_gate_up_preferred_in_forward` becomes single-token only: `SwitchGLU::forward` tries the kernel at one token and takes the gather path at every multi-token batch size, on every generation. M1 and M2 already declined at B >= 2 (their measured loss was about 10%), so the shared rule is now one condition rather than a generation split. Inkling is unaffected: `forward_activated` calls the kernel directly at its measured single-token win and does not pass through this gate.

M4 and M5 are unmeasured for this kernel. They now decline with M3 rather than sharing its superseded exception; a future measurement can re-open the multi-token route by generation if one of them wins.

## Environment

| Field | Value |
|---|---|
| Host | Mac Studio, Apple M3 Ultra, 512 GB unified memory, macOS 26.6.2 (25G83) |
| Apple GPU generation | 15; `AppleSiliconGen::M3`, whose pre-change gate admitted the kernel at B >= 2 and is the behavior this A/B measures |
| Build | `cargo build --release --features metal,accelerate --bin mlxcel-server --example logit_trace`; `mlxcel 0.8.1`; fat LTO, `codegen-units = 1`; `mlxcel-server` sha256 `78cbf551b5b05809816487493bc8a8b20d2e8afe52b655fa173977fcbbfcd758` |
| Branch | measurement binary from `main` at `79f0a447` (no source changes); the gate change that follows the finding is committed on `feature/issue-2515-switch-gate-up-batch-measure` and does not alter either measured path at B >= 2 except to stop attempting the kernel |
| Harness | `scripts/bench_serving_concurrency.py` against `mlxcel-server --parallel 8 --ctx-size 8192 --metrics --offline --no-prompt-cache --apc-enabled=false` (eight 1024-token slots, `max_batch_size=8`, native batched forward); parity with `examples/logit_trace` and `scripts/compare_logit_traces.py` |
| Wrapper | `caffeinate -i`, indexers paused (`scripts/with_indexers_paused.sh`) |
| Target | `models/olmoe-1b-7b-0125-4bit` (64 experts, top_k 8, hidden 2048, expert intermediate 1024; `model.safetensors` sha256 `472151b0e490c93f409caec4ba36524048183118403192a261d2df5645a21bf9`, `config.json` sha256 `83468e17ab8e7ce8255b52fef0e9cc8c82a4a9f8094de45877e6d67810c0e3d7`) |
| Sampling | server ladder: nominal 512-token prompts (445 actual), 128 tokens, temperature 0, greedy, three measured passes per arm; parity: teacher-forced traces, 16 chunks of 16 positions, top-k 8 |
| MLX pin | `7ca40033132dae444d45f7fc3daae57e24c13f6c` |
| Raw results | `bench-results/switch-gate-up-batch/20261104-002134/` (gitignored) and `bench-results/switch-gate-up-batch/20261104-focus-p4/` |

## Follow-up

- The gate change is source-verified by a unit test on the pure decision; the measured paths are unchanged beyond the removed multi-token kernel attempt.
- `docs/benchmark_results/switch-gate-up-batch-m3ultra-2026-10-21.md` stays as the historical record of the first attempt. Read its concurrency 5 to 8 rows as queued behind four rows; its batch-2 and batch-4 cells are superseded by the corrected-geometry and focused re-measurements here.
- Single-token decode was not part of this A/B. The kernel's one-token wiring keeps the inkling measurement in `switch-gate-up-baseline-m3ultra-2026-10-21.md`; the `SwitchGLU::forward` one-token cell was not re-measured here.
- M4 and M5 are unmeasured for this kernel; both now decline at B >= 2. A future host with a positive multi-token cell would need the gate reopened for that generation and a new measurement.
- The committed per-run data is `decode.csv` (42 rows, three runs per arm per batch size) and `focus-parallel4.csv` (12 rows) in `data/switch-gate-up-batch-m3ultra-2026-11-04/`. Raw sweep text and server logs stay under the gitignored `bench-results/switch-gate-up-batch/20261104-002134/`.
