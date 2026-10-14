# Nemotron-H request memory admission — M1 Ultra, 2026-10-13/14

Issue: #2291. Policy: [request memory admission](../request-memory-admission.md). This is a lifecycle/safety calibration, not a throughput or latency improvement claim.

## Setup and attribution

- Apple M1 Ultra, 128 GiB unified memory; local `nvidia-nemotron-3-nano-30b-a3b-4bit` checkpoint, Metal + Accelerate release server.
- Context 8,192, effective parallelism one, greedy eight-token output, offline loading, 24 GiB lower-only request budget. The server-default runtime ceiling still applies.
- The [harness](data/nemotron-h-request-memory-m1ultra-2026-10-13/harness.py) requires a production admission log for every tested generation, verifies actual prompt length is within context, and refuses a rejection claim unless HTTP 503 and the typed memory error agree with that log. Streaming rejection must remain JSON, not an already-open SSE response.
- Physical footprint uses `vmmap -summary`: its peak is the **process-lifetime physical footprint**, rounded to 0.1 GiB, not a per-request MLX allocator peak. The analytical estimate is a projected conservative MLX resident/workspace requirement, so these measurements have different scopes.

## Initial calibration and corrected loading

An initial 32 GiB run admitted the 6,001-token completion with a 27.34 GiB estimate and about 17.8 GiB physical peak. The harness correctly stopped instead of labeling it a rejection. [Raw calibration](data/nemotron-h-request-memory-m1ultra-2026-10-13/calibration.json). This explicitly shows that a later 24 GiB policy rejection is conservative **over-rejection**, not proof the request physically cannot fit.

Loading was hardened to materialize lazy weight graphs before admission, including `--no-warmup`. Candidate 2 (`ea183b308b19e972ae6a7e24779bc3b471aacec0e2dd218c26aaf76e4fc6dff3`) then ran three fresh servers. This binary predates the final distributed/queued-prefill/legacy/cache-warmup routing review fixes; the repeated measurements below are retained as calibration rather than misrepresented as final-source validation.

| Candidate 2 case | Observed HTTP results | Estimate (GiB) | Process physical peak (GiB) |
|---|---|---:|---:|
| Long completion, 6,001 prompt tokens | Six 503 responses, stream and nonstream | 27.337 | 16.9 |
| Long chat, 6,020 prompt tokens | Six 503 responses, stream and nonstream | 45.606–45.702 | 16.9 |
| Short following each rejection | Twelve 200 responses; all streams completed | 17.813–17.960 | 16.9–17.0 |
| Medium completion, 1,501 prompt tokens | Three 200 responses | 22.423 | 17.7 |
| Beyond-context prompt | Three 400 responses, including stream requests | Not admitted | Not measured separately |

The larger chat estimate used an actual 6,014-token history-boundary forward, versus a 2,048-token completion forward. Estimates follow the real plan, not merely the configured chunk size. Retained snapshots explain the later small residency increase in this intermediate candidate; final policy disables donation as well as reuse.

Medium accepted-request estimate median was 22.423 GiB with less than 0.00002 GiB range; measured physical peak was 17.7 GiB in all three repeats at vmmap resolution. One-minute machine load at run start was 6.35, 7.10, and 10.73. HTTP model-list readiness took 7.92, 3.06, and 3.33 seconds (median 3.33, range 4.86); this is not model-load timing because worker loading may continue asynchronously.

A separate candidate-2 `--no-warmup` run also rejected all four long cases, served all four following short cases and the medium request, and preserved context 400 precedence and health. Its first guard saw approximately 17.78 GB of active allocations rather than a lazy zero-weight baseline. [Three-repeat raw results](data/nemotron-h-request-memory-m1ultra-2026-10-13/candidate2.json), [no-warmup results](data/nemotron-h-request-memory-m1ultra-2026-10-13/candidate2-no-warmup.json).

## Final-source validation

Release build at source commit `07db6e30` passed; artifact SHA-256 `1c4673addeb123b8d3ef2758f69dc0c54811d1d061911dbcb86aa486cc6065c5`. This includes all admission/HTTP/security fixes. A subsequent CI-only stack-frame fix replaces a by-value scheduler builder with an in-place setter; that change does not alter these estimates or request geometry, and its final smoke is recorded below.

| Final reviewed case | Outcome | Estimate (GiB) | Process physical peak (GiB) |
|---|---|---:|---:|
| Three normal servers: long completion/chat, both stream modes | 12 typed JSON 503 responses before SSE | 27.337–27.351 | 16.9 |
| Short request after each normal rejection | 12 successful 200 responses | 17.813–17.865 | 16.9 |
| Medium completion after each normal sequence | 3 successful 200 responses | 22.328 | 17.6 |
| One `--no-warmup` server | 4 long 503, 4 following short 200, medium 200 | Same ranges, within 12 bytes | 16.8–17.6 |
| One `--no-batch` server | 4 long 503, 4 following short 200, medium 200 | Long 45.506–45.634 | 16.9–17.6 |
| Qwen2.5-7B, target-only budget set to one byte | Stream and nonstream both 200, health 200 | Policy intentionally absent | Not measured |

Every tested server remained healthy. All normal and no-warmup beyond-context checks returned 400 rather than memory 503. Legacy `--no-batch` context-limit handling is unchanged and was not asserted. Four invalid override startup probes (`0`, `-1G`, `NaN`, and `18446744073709551616`) failed with the explicit configuration diagnostic.

With guarded prompt snapshot donation disabled, normal long chat and completion now both use 2,048-token largest forwards. The legacy unchunked path correctly uses the entire 6,001/6,020-token prompt and produces a larger estimate, demonstrating that the guard follows actual execution geometry rather than the global chunk setting. Medium accepted estimates were identical across all three normal repeats (median 22.328 GiB, range zero); physical peak was 17.6 GiB each at vmmap resolution. No elapsed-time speed claim is made.

Normal run starting one-minute loads were 6.44, 6.90, and 11.01. Normal safetensor-loading-start to eager-materialization-end took 6.056, 1.926, and 2.152 seconds (median 2.152, range 4.130); these include disk I/O and sanitization, so they do not isolate added evaluation overhead. The no-warmup run measured 2.818 seconds over the same interval. There is no isolated eager-vs-lazy startup benchmark. [Startup log excerpts](data/nemotron-h-request-memory-m1ultra-2026-10-13/final-startup-excerpts.txt) record the active weight baseline of 17,777,861,888 bytes before any first request.

Exact commands (normal run repeated three times; other modes once):

```bash
D=docs/benchmark_results/data/nemotron-h-request-memory-m1ultra-2026-10-13
python3 "$D/harness.py" --binary target/release/mlxcel-server --model models/mlx/nvidia-nemotron-3-nano-30b-a3b-4bit --output /tmp/final.json --budget 24G --reps 3
python3 "$D/harness.py" --binary target/release/mlxcel-server --model models/mlx/nvidia-nemotron-3-nano-30b-a3b-4bit --output /tmp/no-warmup.json --budget 24G --reps 1 --no-warmup
python3 "$D/harness.py" --binary target/release/mlxcel-server --model models/mlx/nvidia-nemotron-3-nano-30b-a3b-4bit --output /tmp/no-batch.json --budget 24G --reps 1 --no-batch
python3 "$D/other_family.py" --binary target/release/mlxcel-server --output /tmp/qwen.json
```

Raw records: [normal three repeats](data/nemotron-h-request-memory-m1ultra-2026-10-13/final.json), [no warmup](data/nemotron-h-request-memory-m1ultra-2026-10-13/final-no-warmup.json), [legacy no batch](data/nemotron-h-request-memory-m1ultra-2026-10-13/final-no-batch.json), [Qwen](data/nemotron-h-request-memory-m1ultra-2026-10-13/final-qwen.json), [invalid startup values](data/nemotron-h-request-memory-m1ultra-2026-10-13/invalid-env.json).

Local validation: 26 `request_memory` tests; 23 Nemotron-H model tests passed with three ignored; provider 30 and native-route 34 tests passed (these namespaces overlap the 26, not additive totals). Release build, `cargo check --tests --features metal,accelerate`, `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`, formatting, and diff checks passed. An Ubuntu two-host logical CI test exposed a debug-worker stack overflow after adding a by-value scheduler builder; the in-place-setter correction subsequently passed the formerly failing Linux two-host logical real-model CI step. Remaining workflow status is tracked in the PR.

## Post-setter final artifact smoke

The final scheduler setup correction at `9d2a1584` avoids copying the large scheduler through another by-value builder frame. Its release build passed, and artifact SHA-256 `e7a5cc9997a8c0eeceba6a69769353eed35395e906b270d2c62a5f437e2a7062` was retested with the same harness: one normal server, one legacy `--no-batch` server, and the Qwen smoke. Normal and legacy each produced four within-context long-request 503 JSON responses before SSE, four successful following short requests with complete streams, and one accepted medium request; health remained 200 throughout. The normal context-overflow check remained 400. Qwen streaming and nonstreaming both returned 200 under the one-byte target-only budget.

Estimates match the preceding reviewed artifact: normal long requests 27.337–27.351 GiB, unchunked legacy long requests 45.506–45.634 GiB, and accepted medium requests 22.328 GiB. Process-lifetime physical peaks were 16.9 GiB for the rejection/short lifecycle and 17.6 GiB after the medium request. This is a final functional smoke, not a new repeated performance measurement. The earlier three-repeat calibration and no-warmup evidence remain explicitly tied to their original artifact.

Reproduction uses the commands above with `--reps 1` and output files named `post-setter-normal.json`, `post-setter-legacy.json`, and `post-setter-qwen.json`. Raw records: [normal](data/nemotron-h-request-memory-m1ultra-2026-10-13/post-setter-normal.json), [legacy](data/nemotron-h-request-memory-m1ultra-2026-10-13/post-setter-legacy.json), [Qwen](data/nemotron-h-request-memory-m1ultra-2026-10-13/post-setter-qwen.json).

## Limits and unmeasured cells

No speed claim, universal OOM guarantee, or device-wide reservation is made. External memory pressure, allocator fragmentation, unexpected kernel workspace, M4/M5, CUDA, 2-bit checkpoints, multi-slot scheduling, and distributed memory-pressure behavior are unmeasured. Runtime/unfused LoRA is explicitly rejected during Metal Nemotron-H construction because arbitrary-rank float32 scratch is not modeled; fused checkpoint weights use the common eager-construction path, but no real adapter checkpoint was measured here. Unit tests cover scope exclusion, cancellation/cleanup, checked arithmetic and final-gate routing; they do not replace the real checkpoint cases above.
