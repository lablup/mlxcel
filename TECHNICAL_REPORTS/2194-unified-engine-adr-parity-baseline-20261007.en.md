# Technical Report: PR #2194 - ADR 0007, CLI/server parity harness and B=1 decode baseline

**Date**: 2026-10-07

**Status**: Implemented and measured on GB10; pending merge.

**Languages**: Rust (diagnostic binaries, in-process server driver), Python (rounds driver), Markdown (ADR, results)

**Risk Level**: Low. No decode path changes. The one production-code edit, moving the prompt-cache store setup into `resolve_prompt_cache_store`, is a verbatim move, and the security review confirmed it.

## Executive Summary

Epic #2166 will merge the CLI decode path (`CxxGenerator`) and the server decode path (`BatchScheduler`) into one batch-native engine, following llama.cpp's design. This PR is the epic's Phase 0 (#2167). It writes the target architecture down as ADR 0007, adds a harness that shows where the two paths diverge, and adds a benchmark that times the server path at B=1 next to the CLI. A baseline run settles the speed defaults: prefill chunk 2048, dense single-sequence storage, and a 1.0 percent regression threshold. It also shows that today's server single-stream decode is 8 to 23 percent below the CLI, almost all of it because of paged storage.

## 1. Problem Statement

The CLI and the server decode through separate loops. They share the model forward and the per-token sampler, but the finish logic, the sampler-state lifecycle, prefill chunking, KV storage, defaults and attention kernels are each implemented twice. Bugs have hit one side only (#2090, #1205, #2140, #2128). Before code moved, three things were missing:
- an agreed target design;
- a measurement of how far the two paths diverge today;
- a single-stream throughput figure for the server path. `bench_decode` only ever timed `CxxGenerator`.

## 2. Change Summary

- **ADR 0007:**
  - Engine API names: `Engine`, `KvStore` over dense and paged storage, `PrefillPlan`, `RowSampler`, `FinishStep`.
  - Maps epic sub-issues #2168 to #2176 onto those components.
  - Supersedes ADR 0004's deferral of the KV/scheduler abstraction.
  - A decision table for every CLI-vs-server default that differs.
- **`mlxcel-engine-parity` (`make engine-parity`):**
  - Runs one prompt and one `SamplingConfig` through `CxxGenerator`, the in-process server at B=1 with dense storage, and with paged storage.
  - Also runs a prompt-cache miss against a hit.
  - Prints `identical`, or the first divergent index and both token ids. With `--expect-identical` it exits 1 on divergence.
- **`mlxcel-bench-engine` (`make bench-engine`) and `scripts/engine_bench_rounds.py`:** single-stream TTFT and decode tok/s on both paths, at 256 and 8192-token prompts, with prefill-chunk and storage parameters. Rounds are interleaved with a rotating order and null arms.
- **`src/server/engine_probe`:** builds the real server stack the way `start_server` does, minus HTTP.
- **Baseline results:** `docs/benchmark_results/unified-engine-baseline-gb10-2026-10-07.md`, with the raw records.

## 3. Technical Decisions

**Measure the real paths, not copies.**
- The CLI arm uses the same `select_backend().create_session` call as `mlxcel generate`; the bench uses `bench_decode`'s `generate_with_stats`.
- The server arm uses `build_server_config`, the shared prompt-cache setup, `ModelProvider` and the server warmup, and sends pre-tokenized requests through the provider channel.
- A harness built on copies would have measured the copies.

**Speed defaults by measurement, semantic defaults by reason.** The maintainer chose not to pause for a confirmation, and asked that performance choices go to the faster option.
- The speed rows were therefore settled by interleaved A/B rounds with null arms.
- The semantic rows prefer the server's llama-server-compatible values, because Phase 5 makes the CLI a server client. The one stated exception is greedy-by-default CLI clients, which keeps model tests and benchmarks reproducible.

**Report what actually ran.** Review found that families keeping their own KV state report "paged" while decoding dense at B=1. The tools now count paged kernel launches per request, so a storage A/B cannot silently compare dense with dense.

## 4. Results

- **Prefill chunk: 2048.**
  - A 512 chunk raised TTFT at 8192 tokens by 8.4 percent (Qwen3, CLI), 22.9 percent (Qwen3, server), 29.4 percent (Llama, CLI) and 14.6 percent (Llama, server).
  - Every null range was within ±1.9 percent, and decode tok/s did not move.
- **Single-sequence storage: dense.**
  - Paged decode at B=1 was 23.0 and 6.5 percent slower on Qwen3 (256 and 8192 tokens), and 8.2 and 8.1 percent slower on Llama.
  - Every null range was within ±1.1 percent, and the paged rows ran 3612 and 2064 paged kernel launches.
  - Paged remains the batched backend.
- **Threshold: 1.0 percent.** The widest CLI null-arm per-round decode delta was 0.92 percent, rounded up.
- **Server vs CLI today:**
  - Decode ratios: 0.77 and 0.92 (Qwen3, 256 and 8192 tokens), 0.90 and 0.90 (Llama).
  - With dense storage the server path comes within 1.5 percent of the CLI.
  - TTFT at 8192 tokens follows the chunk size.
- **Parity:** under `MLXCEL_SDPA_DETERMINISTIC=1`, the CLI, server-dense and server-paged paths are identical over 400 tokens for both models, greedy and seeded. Only a prompt-cache hit diverges (Qwen3 greedy at token 0), which is #2170's target.

## 5. Residual Risks and What Was Not Verified

- **The chunk decision is single-stream.** The server's 512 default exists so prefill interleaves with live decode streams in batched serving, and this run did not measure that. #2170 owns the chunk policy and may keep a smaller chunk for mixed ticks if a batched measurement justifies it.
- **Coverage is narrow:** two dense 4-bit families on one GPU. On Llama at 8192 tokens, paged TTFT was 8 percent lower than dense while Qwen3 showed none, so storage effects are not uniform across families.
- **Parity was checked on default prompts up to 400 tokens.** Longer generations and other families may diverge where these did not.
- **Exit code 1 is overloaded:** `--expect-identical` returns it for both divergence and errors.

## 6. Learning Points

- **The baseline changed the plan's emphasis.** Most of the server's single-stream gap is a storage choice (paged at B=1), not the scheduler. The engine's first win is choosing dense for a lone sequence.
- **A tool that labels its own configuration needs a ground-truth counter.** "Paged" was a request, not a fact, until the launch count was recorded.
- **Defaults chosen per path need one shared measurement.** The CLI's 2048 and the server's 512 were each reasonable locally; one interleaved A/B on the same prompt settled it.

## 7. Related

- Epic #2166, issue #2167.
- Later phases #2168 to #2176.
- ADR 0004, ADR 0005, #2162 (`MLXCEL_SDPA_DETERMINISTIC`), #2185 (prior parity harness), #1820 (quiet-host gate).
