# ADR 0007: One batch-native decode engine for the CLI and the server

**Status:** Accepted (2026-10-07). Maintainer decision recorded on epic #2166; this ADR is Phase 0 of that epic (issue #2167). Supersedes the "KV / paged / scheduler abstraction is a later phase" implementation decision of [ADR 0004](0004-compute-backend-session-seam-and-stablehlo-family.md) and the matching "extended layer" note in `src/backend/mod.rs`. The rest of ADR 0004 (the session seam and the StableHLO compiler family) stands. The decision table below is the authority Phases 1 to 6 follow; the maintainer waived a separate confirmation step (comment on #2167, 2026-10-07), so every row states its evidence or reason. Rows decided by measurement were filled from the baseline run in `docs/benchmark_results/unified-engine-baseline-gb10-2026-10-07.md` (GB10, 2026-10-07): the faster option won in every case.

## Context

The CLI and the server share no decode loop. `mlxcel generate`, `mlxcel run` and the chat REPL decode through `CxxGenerator` (`src/lib/mlxcel-core/src/generate.rs:1914-2962`, reached through the inference `Session` of ADR 0004). `mlxcel-server`, `mlxcel serve` and `--no-batch` decode through `BatchScheduler` (`src/server/batch/scheduler/`; the legacy worker is the scheduler at `max_batch_size = 1`). What they share is the single-row model forward, the per-token sampler, `build_sampling_config`, EOS merging, the streaming detokenizer and the MTP/DFlash round loops. What they implement twice is everything around those: the post-sample finish step (five server copies, four CLI loops), the per-row sampling step and `SamplerState` lifecycle, prefill chunking and trim, sampling-default resolution, and the attention kernels themselves (the CLI's dense cache runs MLX fused SDPA, pool-backed server decode runs mlxcel's paged kernels).

The cost is a steady stream of one-sided fixes (#2090 then #2117 in four loops, #1201 then #1205, #2140 in both trim hooks), defaults that drift apart (#1118, #1128, #1436), and a documented refusal to promise that the server and `mlxcel generate` produce the same tokens (`docs/CONTINUOUS_BATCHING.md`, "Token-exactness against `mlxcel generate`"). Until this phase nothing measured that divergence, and nothing measured the decode throughput of the path the server runs: `mlxcel-bench-decode` (`src/bin/bench_decode.rs`) times only `CxxGenerator`.

### Reference design: llama.cpp

Verified on llama.cpp HEAD 65840ed (2026-10-06):

- The core forward is batch-native. `llama_decode(llama_batch)` takes tokens from any number of sequences against a multi-sequence KV cache; a single sequence is a batch of one.
- Sampling with history lives in one object. `common_sampler_*` wraps libllama's `llama_sampler` chain and every tool, server included, samples through it.
- The CLI is a client of the server. `tools/cli/CMakeLists.txt` links `llama-server-impl`; `tools/cli/cli-server.h` runs `llama_server()` in a thread on a random local port; `tools/cli/cli-context.cpp` posts to `/v1/chat/completions` and reads the SSE stream.
- History: ggml-org/llama.cpp#17824 (2025-12) moved the CLI onto the server context and kept the old loop as `llama-completion` for raw use; ggml-org/llama.cpp#24948 (2026-07) made the CLI an HTTP client with `--server-base` for a remote server.

## Decision

Build one batch-native engine in `mlxcel-core` and put both front ends on it.

### Components and API types

The engine lives in a new `mlxcel_core::engine` module. The type names below are the contract the phases implement against; signatures are settled in the phase that introduces each type.

| Component | Types | Responsibility | Lands in |
|---|---|---|---|
| Engine | `Engine`, `SequenceSpec`, `SequenceId` (existing, `mlxcel_core::cache`) | `open(SequenceSpec) -> SequenceId`, `prefill(&PrefillPlan) -> PrefillOutcome`, `step(&StepBatch) -> StepOutput`, `close(SequenceId) -> ClosedSequence`. A single sequence is a `StepBatch` of one; there is no separate single-sequence loop. | #2172 (Phase 4b) |
| KV storage | `KvStore` (trait), `DenseKv`, `PagedKv`, `KvBatchView`, `DetachedKv` | One KV interface: allocate, append, the single `trim(seq, n)` hook, detach and adopt (prompt cache), and the per-step batch view the forward reads. Dense storage keeps MLX SDPA; paged storage keeps the block table, pool and paged kernels. | #2171 (Phase 4a) |
| Prefill plan | `PrefillPlan`, `PrefillOutcome` | One description of how a prompt is prefilled: adopted prefix offset, history-boundary split, chunk size, padding, and the trim that undoes padding. Both front ends get the same partition for the same input. | #2170 (Phase 3) |
| Sampling step | `RowSampler`, `SamplerState`, `TokenDraw` | One per-row sampling step with one `SamplerState` lifecycle and one token-history ordering rule (history pushed before the next draw, prompt included). Seeds are per sequence. | #2169 (Phase 2) |
| Finish step | `FinishStep`, `FinishDecision`, `StopKind` (existing, server) | One post-sample decision: EOS, history push, stop strings, generation and context bounds, `max_tokens`, loop detection, cache-clear cadence. | #2168 (Phase 1) |
| Scheduler | `BatchScheduler` (existing) | Admission, queueing, prefill/decode interleaving (ADR 0005 tick policy), preemption and the prompt-cache policy, built on `Engine` instead of driving models directly. | #2172 (Phase 4b) |
| CLI front end | `mlxcel run`, chat REPL | In-process clients of the server engine: the command starts the server worker in-process and submits requests through the same request channel the HTTP routes use (`ModelProvider`), one slot. | #2173 (Phase 5) |
| Raw completion | `mlxcel generate` | A thin raw-completion client of the engine (the `llama-completion` analog): no chat-server lifecycle, same engine, same sampling and finish steps. `CxxGenerator` is retired. | #2176 (Phase 6) |

The CLI front end deliberately uses the in-process request channel rather than llama.cpp's loopback HTTP: in one process the HTTP hop only adds serialization, and the channel is already what every server route uses below the HTTP layer. A remote `--server-base` mode (the ggml-org/llama.cpp#24948 shape) is compatible with this design and out of scope for the epic.

Rejected alternative: route the CLI through today's scheduler as it is. The CLI would inherit the server path's different numerics (paged kernels, 512-token chunks) and the scheduler's per-step overhead without any of the duplication going away. Unifying the engine first is what makes "same input, same output" hold.

### Invariants

- Once Phase 5 lands, greedy output for the same request under `MLXCEL_SDPA_DETERMINISTIC=1` is identical across `mlxcel generate`, `mlxcel run` and the server engine at B=1, dense and paged. `make engine-parity` with `--expect-identical` is the gate.
- No phase regresses single-stream decode throughput beyond the threshold row below, at short and long context, against the pre-epic `CxxGenerator` baseline.
- Every phase ends integrated into the real code paths.
- Model-owned decode rewinds (`LanguageModel::supports_decode_lookahead_rewind` / `rewind_decode_appends`, #2182) and the prompt-cache history-boundary split (`capture_history_boundary_snapshot`, #1143, mirrored by the MTP burst's `history_boundary_split`, #2185) survive every phase.

### What parity means on ROCm and Metal

`MLXCEL_SDPA_DETERMINISTIC` is read only by the CUDA overlay (`src/lib/mlx-cpp/patches/mlx/backend/cuda/scaled_dot_product_attention.cpp`). On ROCm and Metal it has no effect, and dense decode (MLX SDPA) and paged decode (the HIP or Metal paged kernels) reduce in different orders. On those backends the invariant is: the CLI and the server engine with dense storage are identical (same storage, same kernels once Phases 1 to 5 land), and the dense-vs-paged pair is a recorded divergence index per model, not a gate. #2192 measures it on the gfx1151 host.

## Baseline tooling

Two tools record where the paths stand before any decode code moves. Later phases rerun them.

- `make engine-parity MODEL=<dir>` (`src/bin/engine_parity.rs`) runs one chat prompt and one `SamplingConfig` through (a) `CxxGenerator` the way `mlxcel generate` calls it, (b) the server scheduler in-process at B=1 with dense decode storage, and (c) the same with paged storage, all under `MLXCEL_SDPA_DETERMINISTIC=1`. Cases: greedy, and seeded sampling with repetition, frequency and presence penalties plus DRY. On (b) it also compares a prompt-cache miss (the request arriving cold) with a hit (the same request after a priming request stored its history prefix, the entry a follow-up chat turn finds) and states each side's prefill partition. It prints one row per pair with `identical` or the first divergent index and both token ids, reports (c) as `n/a` when the worker cannot serve paged decode (`supports_paged_decode_backend() == false`, detected through the scheduler's decode-storage fallback counter), and `--expect-identical` exits 1 on any divergence. Paged storage and the paged kernel are not the same thing: families that keep model-owned KV (Gemma 3, Llama 4, Qwen 3.5) resolve to paged storage but decode a lone sequence through their dense caches, with only the block table paged (`CachePool::allocate_with_layout`, `src/lib/mlxcel-core/src/cache.rs:6807`). The (c) row therefore carries `paged-kernel-launches=N`, read from `mlxcel_core::cache::paged_batch_decode_stats`, and a zero there means that row compared dense attention. `--json` writes every stream for the record.
- `make bench-engine MODEL=<dir>` (`src/bin/bench_engine.rs`) measures B=1 decode tok/s and TTFT on both paths at a 256-token and an 8192-token synthesized prompt. The CLI arm is `mlxcel-bench-decode`'s own measured call (`CxxGenerator::generate_with_stats` after a warmup pass, same synthesized prompt), so its numbers are that tool's numbers. The server arm builds the `mlxcel-server` configuration and model worker in-process (`src/server/engine_probe/server_engine.rs`) and submits pre-tokenized requests through the provider channel, so HTTP and tokenization are outside the timed region. `--prefill-chunk N` sets the chunk on both paths, `--decode-storage dense|paged|auto` sets the server's storage, and every server row records `paged_decode_launches` so a paged row that ran dense attention is visible.
- `scripts/engine_bench_rounds.py` runs interleaved rounds of named arms, each a fresh process, plus a null arm (the first arm repeated; every arm with `--null-every-arm`), and prints medians, paired per-round deltas, and the null spread. Each benchmark process is killed after `--run-timeout` seconds (default 3600), and an arm name ending in `-null` is rejected because it would collide with a generated null arm. It is the protocol for every measured row below, following the #2185 harness and the #1820 host gate.

## Decision table: CLI vs server defaults

One row per default that differs between the two paths today. References verified against main at bbd05099 (2026-10-07).

### Rows that change output (decided here)

Phase 5 makes the CLI a client of the server engine, so the server's llama-server-compatible value is the default choice unless a row states a concrete reason against it.

| Default | CLI today | Server today | Unified engine | Reason |
|---|---|---|---|---|
| Penalty window `repeat_last_n` | `-1`, full history (`src/main.rs:981`, threaded as `penalty_last_n` at `src/commands/generate.rs:1252`) | `64` (`src/server/startup.rs:747`) | `64` | llama-server's and llama-cli's default (#1436). Inert unless a penalty is set: both paths default `repeat_penalty` to 1.0 and the other penalties to 0. A bounded window also caps the incremental penalty state. |
| DRY window `dry_penalty_last_n` | `-1`, full history (`src/main.rs:975`, mapped at `src/commands/generate.rs:1227-1231`) | `64` (`src/server/startup.rs:754`) | `64` | llama-server's default. Inert unless DRY is enabled (`dry_multiplier` defaults to 0 on both paths). |
| DRY sequence breakers | none, hard-coded (`src/commands/generate.rs:1242`) | b10621 set `\n`, `:`, `"`, `*` (`src/server/dry_breakers.rs:40`), `--dry-sequence-breaker` | b10621 set, string-derived breaker heads | Without breakers the backward match never stops at a line or punctuation boundary, so the CLI's DRY penalty is stronger than the same nominal settings on the server (the comment at `generate.rs:1232-1241` says as much). The CLI gains the flag through the server engine. |
| Loop detection | off: `build_sampling_config` leaves it disabled (`src/execution/sampling.rs:175`) and the CLI never sets it | `resolve_loop_detection` (`src/server/request_options.rs:259`): request override, else `MLXCEL_LOOP_DETECTION`, else Gemma 4 family default-on for tool-shaped requests (#432, #967) | the server's resolution, applied in the one finish step | The rule is request-shaped, not path-shaped. Plain CLI chat carries no tools, so it resolves to off exactly as today; what changes is that `MLXCEL_LOOP_DETECTION` and tool-shaped CLI requests now behave as on the server. |
| Frequency and presence penalty | hard-coded 0, no flag (`src/commands/generate.rs:1243-1244`) | default 0, `--frequency-penalty` / `--presence-penalty` | default 0, exposed to every client | Same default, so no output change; the CLI stops being unable to set them. |
| XTC | hard-coded off (`src/commands/generate.rs:1247`) | default probability 0, threshold 0.1 | server default and flags | Same reason as the row above. |
| Mirostat | hard-coded 0 (`src/commands/generate.rs:1254`) | default 0, `--mirostat` | server default and flags | Same reason. |
| Dynamic temperature | hard-coded range 0 (`src/commands/generate.rs:1257`) | default range 0, `--dynatemp-range` | server default and flags | Same reason. |
| Base sampler when `generation_config.json` is silent | temperature 0, top-k 0, top-p 1.0, min-p 0 (`src/main.rs:913-925`) | 0.8 / 40 / 0.95 / 0.05 (`src/server/startup.rs`, `ServerStartupConfig::default`) | no engine default; each client resolves its own request defaults, CLI clients keep greedy | Kept as a concrete exception to the server-first rule. Both paths already prefer the checkpoint's `generation_config.json` (`resolved_cli_sampling_params`, `resolve_generation_sampling_defaults` at `startup.rs:1813`). Greedy-by-default `mlxcel generate` is what the model-test and benchmark workflows (`make bench`, `/test-model`, the parity harness's own CLI arm) rely on for reproducible output; flipping it to 0.8 would make every bare CLI run nondeterministic. The value travels as a request parameter, so the engine stays default-free. |
| `max_tokens` when unset | `-1`: context window minus prompt (`src/main.rs:712-719`, `src/cli/max_tokens.rs`) | `--n-predict -1`: per-slot context, else the model's context window, else 4096 (`resolve_default_max_tokens`, `src/server/startup.rs:1097`) | the server rule, evaluated against the sequence's own context bound | The in-process server a CLI client starts has one slot, so the per-slot context is the whole window and `-n -1` keeps meaning "until the context is full". One bound in the finish step instead of two. |
| Seed scope | the global MLX RNG seeded once per generation (`seed_rng_if_needed`, `src/lib/mlxcel-core/src/generation_policy.rs:26`) | reseeded to the row's own seed right before its first token is sampled (`src/server/batch/scheduler/prefill.rs:1355-1369`, #347); the batched fused decode path then shares one global-RNG draw across the batch | per sequence, as on the server | A batch-native engine needs a row's draws to depend only on its own seed. At B=1 the two agree on when the seed is applied; the parity harness's seeded case records whether they agree on the draws. |
| Prompt cache | none: every generation prefills from scratch | on by default; a cold chat prefill is split at the history boundary for snapshot families (#1143), and a hit forwards only the suffix | on for chat clients (server default); off for raw `generate` | Multi-turn `mlxcel run` gains prefix reuse. The split and the suffix-only forward change near-tie greedy tokens (#2185), so the parity gate compares with the cache off and the miss-vs-hit row is tracked as a measured divergence that the single prefill plan (#2170) targets. |

### Rows that change speed (decided by measurement; the faster option wins)

| Default | CLI today | Server today | Candidates | Deciding measurement | Chosen |
|---|---|---|---|---|---|
| Prefill chunk | 2048 (`DEFAULT_PREFILL_CHUNK`, `src/lib/mlxcel-core/src/generate.rs:282`; `MLXCEL_PREFILL_CHUNK`) | 512 (`src/server/config.rs:1172`, `src/server/startup.rs:698`, `--prefill-chunk-size`) | 2048, 512 | TTFT at the 8192-token prompt, Qwen3-1.7B 4-bit and Llama-3.2-1B 4-bit, on both paths (`--prefill-chunk 2048` vs `512`), interleaved rounds with a null arm. Lower median TTFT wins when its per-round delta clears the null spread on both models; if neither clears it, 512 wins (bounded prefill transient and finer interleaving with live decode streams, ADR 0005 and #1011). | **2048.** TTFT at 8192 tokens, 512 vs 2048 paired: Qwen3 CLI +8.4 %, server +22.9 %; Llama CLI +29.4 %, server +14.6 %; every null range within ±1.9 %. Decode unchanged. ([results](../benchmark_results/unified-engine-baseline-gb10-2026-10-07.md)) |
| Single-sequence KV storage | dense `KVCache`, MLX fused SDPA | paged when the worker can serve it (`effective_decode_storage_backend`, `src/server/batch/scheduler/mod.rs:210`: `auto` resolves to paged when `max_batch_size > 1` and the model supports batching and paged decode), else dense | dense, paged | B=1 decode tok/s on the server path, `--decode-storage dense` vs `paged`, both models, 256 and 8192-token prompts, interleaved rounds with a null arm. Both models keep their KV in the scheduler's pool, so a lone paged sequence runs the pooled paged kernel (`src/models/qwen3.rs:354`, `src/models/llama3.rs:746`); every paged row must show `paged_decode_launches > 0`. The result does not transfer to model-owned KV families, which decode a lone sequence dense under either setting. Higher median wins when it clears the null spread in every cell; a split result keeps both storages selectable and the long-context winner is the default, since decode at 8K keys is where storage layout matters. | **dense.** B=1 decode, paged vs dense paired: Qwen3 −23.0 % (256) and −6.5 % (8192); Llama −8.2 % and −8.1 %; every null range within ±1.1 %; paged rows ran the paged kernel (3612 and 2064 launches). Paged stays the batched backend. ([results](../benchmark_results/unified-engine-baseline-gb10-2026-10-07.md)) |

### Regression threshold

| Row | Value | How it is set |
|---|---|---|
| Unified single-stream decode vs the pre-epic `CxxGenerator` baseline, at both context lengths | **1.0 %** (widest CLI null-arm per-round decode delta 0.92 %, Qwen3 at 256 tokens, rounded up) | From the null arm, not chosen in advance: the widest null-arm per-round decode tok/s delta across the four (model, context) cells, rounded up to the next 0.5 percent. A phase passes when its median paired delta against the baseline arm is no worse than minus the threshold in every cell. |

### Measurement commands

All on a quiet host, release build (`cargo build --release --features cuda --bin mlxcel-bench-engine`), recorded with the host, mlxcel commit and MLX pin. `QWEN=models/mlx/qwen3-1.7b-4bit`, `LLAMA=models/mlx/llama-3.2-1b-instruct-4bit`, `R=scripts/engine_bench_rounds.py`.

- Baseline, both paths, both contexts, with a null arm per path: `$R --model $QWEN --rounds 5 --null-every-arm --arm cli="--path cli" --arm server="--path server" --out baseline-qwen.jsonl`, then the same with `$LLAMA`.
- Prefill chunk on the CLI path: `$R --model $QWEN --prompt-tokens 8192 --rounds 5 --arm c2048="--path cli --prefill-chunk 2048" --arm c512="--path cli --prefill-chunk 512" --out chunk-cli-qwen.jsonl`; on the server path the same with `--path server`; both repeated for `$LLAMA`.
- Single-sequence storage: `$R --model $QWEN --rounds 5 --arm dense="--path server --decode-storage dense" --arm paged="--path server --decode-storage paged" --out storage-qwen.jsonl`, repeated for `$LLAMA`.
- The null arm is part of every run above (the first arm repeated each round); `--hostgate` adds the #1820 quiet-host gate.

## Consequences

- ADR 0004's session seam survives with a narrower job: an MLX `Session` becomes a one-sequence client of `Engine`, and the KV and scheduler coupling ADR 0004 deferred is resolved by `KvStore` rather than left MLX-only. The compiler-family backend (ADR 0004 Track B) can implement `KvStore` later; it is not required to.
- Every phase reruns `make engine-parity` and the baseline rounds and reports both. The parity rows that diverge today are the work list; the throughput threshold is the guard rail.
- `docs/CONTINUOUS_BATCHING.md`'s token-exactness disclaimer is replaced once Phase 5 lands, and the CLI docs describe one engine.
- Two ends of the CLI stay deliberately different from the server: greedy request defaults for CLI clients, and a raw `generate` that skips the chat lifecycle. Both are client choices over the same engine.

## References

- Epic #2166 and its phases #2167 (this ADR and the baseline tools), #2168, #2169, #2170, #2171, #2172, #2173, #2176, #2192.
- llama.cpp HEAD 65840ed: `llama_decode(llama_batch)`, `common_sampler_*`, `tools/cli/CMakeLists.txt`, `tools/cli/cli-server.h`, `tools/cli/cli-context.cpp`; ggml-org/llama.cpp#17824 and ggml-org/llama.cpp#24948.
- Prior parity and noise method: `tests/speculative_parity.rs` (anti-fallback log check, `check_row_parity_with_near_tie`), the #2185 harness in `docs/benchmark_results/data/gemma4-mtp-cuda-gb10-2026-10-07/harness/`, and the #1820 host gate in `docs/benchmark_results/data/sdpa-plan-bucket-gb10-2026-09-12/harness/hostgate.py`.
- [ADR 0004](0004-compute-backend-session-seam-and-stablehlo-family.md) (the deferral this ADR supersedes) and [ADR 0005](0005-mixed-prefill-decode-step-execution.md) (the tick policy the scheduler keeps).
