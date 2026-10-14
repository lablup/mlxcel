# B=1 terminal decode overhead (GB10, 2026-10-11)

Issue #2227 investigates the remaining single-stream gap between the in-process server used by `mlxcel run` and the pre-epic CLI path. Profiling rejected the proposed host micro-optimizations because each measured below the issue's 0.1% threshold. It instead found one request-boundary inefficiency: a fused-lookahead-eligible capped B=1 request performed two complete decode forwards whose results could not affect the response.

## Host, builds, and method

Measurements ran on `spark-102`, NVIDIA GB10, driver 580.178.04, CUDA 13.0, Linux 7.0.0-1019-nvidia, release profile with CUDA, and MLX pin `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`.

- Pre-epic baseline: `4c44e317dcbc5a66548e6aed1cd01cdf36f97499`.
- Pre-change profiling source: `73700aafa6a792f07830578c0c26137cea3a038b` plus temporary instrumentation patch SHA-256 `306b2664d8b69186cc896ec83c1f6323e534372fc1148fd75fc4df9e5e2fe0b6`.
- Production release binary source: `7c8830cd5aa01198e6ff93b661f90c048ccaeb1d`, tree `aa976c7eaa17b1e39c6b984e7124e184e96d1800`.
- Final test checkout before documentation: `93e2f9abb3127eb64d414b9dddc4224cce3c703b`; its only difference from the release source is in two `cfg(test)` scheduler modules. The production-file diff is empty.

The final performance run uses `scripts/engine_bench_rounds.py`: five interleaved rounds, 16 warm-up tokens, 128 measured decode tokens, the quiet-host gate before each arm, and a repeated `base-cli` null arm. `base-cli` and `new-srv-dense` use separate release builds. The performance driver explicitly unsets deterministic-SDPA and profiler variables to match the Phase 0 default dispatch. Correctness runs set `MLXCEL_SDPA_DETERMINISTIC=1`.

## Profile before the change

Linux had `perf_event_paranoid=4`, so CPU sampling was unavailable. Nsight Systems 2025.3.2 recorded CUDA and OS runtime events with sampling, event sampling, backtraces, and context-switch tracing disabled. Temporary `Instant` timers measured scheduler stages. The denominator starts at normal decode-loop entry, excludes prefill, and ends after completion finalization.

The Qwen3-1.7B 4-bit profile used a 256-token prompt and one binary for both server and CLI. The measured server request had 127 ticks averaging 5,318.203 microseconds.

| Stage | Calls | Total ns | Percent of full decode tick | Decision |
|---|---:|---:|---:|---|
| Metrics publication | 128 | 312,052 | 0.0462% | Leave unchanged: below 0.1% |
| Row lookup | 127 | 17,824 | 0.0026% | Leave unchanged |
| Row build | 127 | 18,096 | 0.0027% | Leave unchanged; row plumbing totals about 0.0053% |
| Lookahead collect | 126 | 564,546 | 0.0836% | Leave unchanged |
| Fused finish rows | 125 | 976,026 | 0.1445% | Retain the required shared sampling and finish transition |
| Provider channel send | 127 | 471,319 | 0.0698% | Leave unchanged |
| Lookahead submit | 127 | 656,315,740 | 97.1727% | Includes asynchronous GPU work |
| Synchronous engine step | 2 | 13,668,204 | 2.0237% | Investigate the terminal replay |

The provider timer covers only the `mpsc` send; it does not cover token decoding or `String` allocation. `StreamFilter` is downstream of the provider and outside the benchmark's server arm, so this profile makes no timing claim about it.

Nsight recorded 4,067 server versus 3,941 CLI `cudaGraphLaunch` calls, and 4,019 versus 3,894 `cudaGraphExecUpdate` calls. These aggregate traces include both the 16-token warm-up and 128-token measured requests, and a model forward emits multiple CUDA graph calls. They corroborate extra server GPU work but are not converted into per-token or per-forward counts.

## Root cause and change

On the terminal length tick the B=1 scheduler already holds pending token `n`. The old order was:

1. submit the speculative `n+1` forward;
2. read pending terminal token `n` and discover that it exhausts `max_tokens`;
3. unwind the speculative cache appends for `n` and `n+1`;
4. synchronously replay `n`, even though its sampled token was already known.

That order spends two unused complete forwards once per affected capped request. `DirectEngine` already checks the length budget before submitting the next lookahead.

The scheduler now applies the same check when exactly one sequence is active. If the pending token must exhaust the length budget, it skips `n+1` and commits `n` through the existing fused token application and shared finish/finalization logic. Batches larger than one retain the existing mixed-row teardown and replay behavior. EOS still takes precedence at the boundary, a failed or malformed pending read falls back to the existing unwind and synchronous path, and stochastic sampling commits the already drawn pending token without advancing or redrawing it.

## Performance

Decode tok/s, paired `new-srv-dense` delta against pre-epic `base-cli`, median over five rounds with range. The null arm repeats `base-cli` within each round.

| Model | Prompt tokens | `new-srv-dense` vs `base-cli` | Null arm | Criterion |
|---|---:|---:|---:|---|
| Qwen3-1.7B 4-bit | 256 | **+0.57% (-0.25%..+1.54%)** | +0.04% (-0.41%..+0.76%) | Pass; ranges overlap the null arm |
| Qwen3-1.7B 4-bit | 8192 | **+0.32% (+0.04%..+0.63%)** | +0.22% (-0.13%..+0.66%) | Pass; ranges overlap the null arm |
| Llama-3.2-1B 4-bit | 256 | **+0.48% (+0.19%..+1.41%)** | +0.07% (-12.38%..+0.54%) | Pass; ranges overlap the null arm |
| Llama-3.2-1B 4-bit | 8192 | **-0.16% (-1.15%..+0.05%)** | -0.17% (-0.29%..+0.14%) | Pass; ranges overlap the null arm |

The issue criterion is a median paired delta of at least -1.0% in every cell, and all four cells pass. Every candidate range overlaps its null-arm range, so the result establishes the no-regression threshold rather than a speedup. The Llama 256 null arm includes -1.01% in round 2 and -12.38% in round 3. The cause of that large round-3 excursion is unknown; the round is preserved in the raw data and summary without exclusion or rerun.

## Correctness and causality

- Four dense external-KV cycle-Llama scheduler tests pass. They cover length limits 0 through 4, a live B=1 lookahead, unchanged B=2 behavior, a seeded stochastic pending draw, and EOS at the length boundary. With `max_tokens=3`, B=1 submits two decode forwards and B=2 retains four.
- A controlled mutation replaces the terminal-budget guard with `false`. The dedicated live-lookahead test then fails with actual forward count 4 versus expected 2 and cargo exit 101. The production source was restored after the run. The earlier interrupted mutation build with exit 143 is excluded.
- A separate model-owned Muse fixture compares B=1 with B=2 generated IDs, `FinishReason`, stream text, final `CachePool.current_offset`, and model-owned offsets. It documents that Muse is ineligible for decode lookahead and therefore does not serve as the causal forward-count fixture.
- The restored full CUDA suite passes: the root crate reports 8,930 passed and 165 ignored, `mlxcel-core` reports 2,072 passed and two ignored, and all integration binaries report zero failures.
- The real `mlxcel run` output is byte-identical before and after the change. Both extracted generated texts have SHA-256 `ecbb6c3137fa73714f6fe6df1dd6414b14c6af75796efe19662a3aaab50eaea6`.
- Full deterministic Llama engine parity reports 13 identical pairs, zero divergent pairs, and one inapplicable seeded run/server pair.
- Deterministic Qwen parity without prompt-cache cases reports nine identical pairs, zero divergent pairs, and one inapplicable pair.

The full Qwen command does not meet its all-pairs criterion because its prompt-cache miss/hit pair diverges at token 58 for greedy sampling (`374` versus `594`) and token 61 for seeded sampling (`576` versus `362`). A forced-synchronous rerun reproduces the same positions and token IDs while the new lookahead guard is unreachable. The no-cache Qwen comparisons targeted by this change are identical. Prompt-cache correctness is outside #2227's declared scope, so the result is recorded rather than attributed to this optimization.

The after-change real-run capture occurred while the source checkout temporarily contained the controlled guard mutation. The release `mlxcel` binary was not rebuilt during that mutation: its recorded SHA-256 is `b9b68fef6d85f8d733afbbefa00674e4332b424a56f626cde5db7c9c70e58501`, and its independently archived source is `7c8830cd`. The checkout status and binary-source identity are both preserved in provenance; the mutation was then restored and the test binary rebuilt before acceptance.

## Reproduce and raw data

The data directory contains the profile and acceptance drivers, parser, temporary instrumentation patch, environment/provenance records, raw logs and JSONL, correctness logs, source patches, Nsight text exports, and a SHA-256 manifest. Repository policy excludes generated binary profiling captures. The original CLI capture was 3,077,212 bytes with SHA-256 `646508acb7a4193870b5bbf1d93b090e6abc6667cda9da1783239ae63b176719`; the original server capture was 3,147,242 bytes with SHA-256 `19efc62eaf08bcc9b25f3c79c1be9ccb43220f4d84ef1b0091bb5b04e38884ab`. Both remain in the run host's `/tmp/mlxcel-auto-plan/2227-profile/` scratch directory, outside git. The committed `*-nsys-stats.txt`, Nsight logs, plain logs, instrumentation patch, and profile analysis preserve the derived evidence used here. Run `acceptance/run.sh` under `gpu-lock` on a quiet host; it refuses an unexpected source/binary pairing and records both release-binary source and checkout identity.
