# PR #2542: Reset shared model caches when the CLI opens a sequence

**Date**: 2026-10-22
**Status**: Completed (PR open; closes #2494)

## Executive Summary

Issue #2494 reported that `mlxcel generate` on exaone-4.0-1.2b-4bit lost decode throughput when #2225 moved the CLI from `CxxGenerator` onto `DirectEngine` (198 vs 158 tok/s in the issue, release builds). PR #2542 shows the loss was a fixed per-call cost caused by stale state, not a slower engine step. ExaOne 4 cannot batch and keeps its KV caches inside the wrapper; its model-owned pool entry never calls `make_caches()`, so `DirectEngine` never reset them, and the timed run of `mlxcel generate` prefilled the prompt on top of the one-token warmup's KV. The output was wrong, and the stale-offset prefill loaded cuBLAS kernels inside the timed window. `DirectEngine::open_sequence` now calls `make_caches()` when `supports_batching()` is false, the rule `BatchScheduler` already applies at prefill time. On GB10 (test-fast) exaone4 at `-n 384` goes from 179.28 tok/s on main to 202.04 against 202.73 on the d379d2f8 reference, and its greedy text matches the reference again. 4 files, +239/-0.

## 1. Change Summary

| Area | Change |
|---|---|
| `src/lib/mlxcel-core/src/engine/direct.rs` | `open_sequence` calls `self.model().make_caches()` after a successful `engine.open` when the model cannot batch, before `apply_pool_kv_cache_mode`; doc comment names the families and the scheduler rule it mirrors |
| `src/lib/mlxcel-core/src/engine/mod.rs` | Registers the new test module |
| `src/lib/mlxcel-core/src/engine/shared_cache_tests.rs` (new) | Two tests on a stub model whose offset lives in a `Cell` that only `make_caches` resets |
| `docs/benchmark_results/exaone4-engine-decode-gb10-2026-10-22.md` (new) | Measurement record: mechanism, intercept/slope fit, interleaved arms with a null arm, controls |

The families covered are those that cannot batch and hold one set of caches in a `RefCell`: ExaOne 4, GPT-OSS, Ministral 3, Step 3.5, Mellum and Laguna. The server path is unchanged.

## 2. Technical Decisions

### 2.1 The loss was a fixed cost from a correctness bug

A fit of `time = fixed + n * per_token` over `-n 2, 32, 128, 384` (3 runs each, same session) separates the two arms by intercept, not slope:

| arm | fixed | per token | median time at n=2 / 32 / 128 / 384 |
|---|---|---|---|
| ref (d379d2f8) | 70.5 ms | 4.770 ms | 85 / 226 / 668 / 1905 ms |
| main (8116e266) | 302.0 ms | 4.851 ms | 318 / 457 / 898 / 2175 ms |

`mlxcel generate` runs a one-token warmup and then the timed run on the same client, so the timed run's prefill started at the warmup's 26 cached positions. Two consequences followed. The model saw the prompt twice, so with `--show-reasoning -n 48` main's text differed from ref's from the first token. And attention against a non-empty cache took a different kernel path: a gdb backtrace on every `cuLibraryLoadData` shows the warmup prefill loading cuDNN SDPA, RoPE and RMSNorm, while the timed prefill loaded cuBLAS GEMM kernels (`CublasGemm::run_batched`, ten `execute_matmul` loads) and `Softmax`. nsys shows nine of those library loads at 9 to 71 ms each, about 290 ms in all (under nsys), inside the timed window. After the fix the whole fixed cost of a call, prefill included, is 57 ms.

The throughput regression was therefore a symptom of a state leak. Tuning the engine's decode step would not have touched it.

### 2.2 Reset in the client at open time, gated on `supports_batching()`

The reset sits in `DirectEngine::open_sequence`, the single entry that `generate` (including embedding prefill), `loglikelihoods`, `generate_with_drafter` and `warm_up_verify_widths` go through. It follows the rule the server already applied: `BatchScheduler` calls `make_caches()` at prefill time (`execute_full_prefill`, `start_chunked_prefill`) for every model that cannot batch, and the retired CLI loop reset through `reset_with_model` between warmup and timed run. Placing it here rules out three narrower options:

- Fixing each wrapper (adding `prepare_sequence_state` to ExaOne 4 and the others) would touch six families and leave the next non-batching wrapper exposed.
- Calling `make_caches()` in `CachePool::allocate_with_layout` for model-owned entries would change the pool contract that the scheduler also uses; the scheduler already resets at prefill, so the server would reset twice.
- Resetting between warmup and timed run in the `mlxcel generate` command would fix only that caller, not `loglikelihoods`, `MlxInferenceSession`, `bench_decode`, `speculative_bench`, `engine_parity` or the engine probe, which all use the same client.

The reset runs after `engine.open` succeeds, so a failed open resets nothing, and once per open, never per step. Batching families skip it because their pool entry already called `make_caches()`.

### 2.3 Families with model-owned per-sequence state are unaffected

Some non-batching families (`ModelOwnedSequenceState`: Mamba, Jamba, NemotronH, LFM2, Kimi, RWKV7 and others) build per-sequence state in `prepare_sequence_state`. For them `make_caches()` replaces only the fallback slot, which the engine path does not use, and leaves the per-id state and its KV modes intact. The cost is one cheap object allocation per open. granite-4.0-h-tiny, which cannot batch and whose sequence-keyed state was already correct, now also gets the reset per open and was measured as a control (section 3).

`make_caches` returns `Vec<KVCache>`, not a `Result`, and cache construction is lazy (`keys: None`, including `RotatingKVCache::new`), so `let _ =` discards dummy caches and nothing that could carry an error.

### 2.4 Behavior change: CLI text changes for the covered families

`mlxcel generate`, `loglikelihoods` and `MlxInferenceSession` now decode these families from fresh caches, so their output changes, toward the correct text: exaone4 greedy text matches the d379d2f8 `CxxGenerator` build; gpt-oss-20b no longer opens with a different reply ("The user wants...") and opens like ref ("We need to write a detailed essay..."); laguna-xs-2.1-nvfp4 greedy `-n 32` text with a warmup equals its `--profile` text without one. `--prompt-lookup` refuses every model that cannot batch, so it is unaffected.

### 2.5 Changes during review

- `210968ef`: the implementation review found Mellum and Laguna also keep `RefCell` caches that only `make_caches()` resets, with no `prepare_sequence_state`. The doc comment, the test header, the PR body and the record now name all six families. The same commit corrected the PR body's claim that `--prompt-lookup` output changes.
- `01c5a298`: the record's prefill-cost paragraph was restated to measured values only (the nsys total under nsys, and the post-fix fixed cost from the fit).

## 3. Verification

GB10 (sm_121), kernel `7.0.0-1019-nvidia`, driver `580.178.04`, `MLX_CUDA_ARCHITECTURES=121`, test-fast profile. Arms: ref d379d2f8 (last `CxxGenerator` build), main 8116e266, fix, and the fix binary again as a null arm; arm order rotated each round, each arm with its own warmed `MLX_PTX_CACHE_DIR`, under `gpu-lock`.

| Check | Result |
|---|---|
| exaone-4.0-1.2b-4bit, `-n 384`, 7 rounds | ref 202.73, main 179.28, fix 202.04, null 201.94 tok/s (medians) |
| exaone4 paired | main vs ref -12.27% (-12.91..-10.26); fix vs ref -0.72% (-1.14..+0.82); null floor (null vs fix) +0.11% (-1.48..+1.46) |
| exaone4 fit after the fix (same block as a fresh ref fit) | fix 56.8 ms fixed, 4.970 ms/token; ref 69.7 ms fixed, 4.888 ms/token |
| llama-3.2-1b-4bit control (batching, path unchanged), `-n 384`, 5 rounds | ref 251.33, main 254.95, fix 254.37, null 254.45; main vs fix +0.07% (-0.10..+0.60) |
| granite-4.0-h-tiny-4bit control (non-batching, now reset per open), `-n 256`, 7 rounds | main 99.07, fix 97.38, null 99.37; main vs fix +2.14% (-2.43..+3.69) inside null vs fix +1.55% (-0.63..+4.01); `-n 48` text identical main vs fix |
| Greedy `--show-reasoning -n 48` text | exaone4 fix == ref; granite-4.0-h and ministral-3b fix == main; gpt-oss-20b fix opens like ref |
| `engine::shared_cache_tests` (2 new) | Both fail with the reset removed, pass with it |
| `mlxcel-core` lib tests, CUDA, `--test-threads=1` | `engine::` 70, `exaone` 3, `granitemoehybrid` 17, `lfm2` 72, `gpt_oss` 11, `ministral3` 10, `session` 13 |
| `cargo clippy --profile test-fast --features cuda -p mlxcel-core --lib --tests -- -D warnings`, `cargo fmt --check` | Clean |

`warmup_does_not_leak_into_the_next_completion` reproduces the `mlxcel generate` shape on the stub: a one-token warmup, then a 6-token run on the same client, which must start its prefill at offset 0 and decode the same stream as a fresh client. The stub's top logit depends on its internal offset, so a stale start changes the stream. `every_open_resets_the_shared_caches` checks one reset per open across `run`, a second `run` and `loglikelihoods`, that scoring starts at offset 0, and that a zero-token budget opens nothing and resets nothing.

## 4. Unverified and Open Items

- **Per-token slope:** in the small post-fix fit the per-token slope is 1.7% above the d379d2f8 reference (4.970 vs 4.888 ms). The interleaved `-n 384` comparison (fix vs ref -0.72%, inside the null floor) does not resolve a difference either way. Not investigated further.
- **Release profile:** all numbers are test-fast. The issue's 198 vs 158 tok/s were release builds; release-profile numbers were not measured.
- **No real checkpoint:** Step 3.5, Mellum and the `Ministral3` text wrapper were not exercised on a real checkpoint. None is local, and ministral-3b-4bit loads as a `mistral3` VLM, which does not use the `Ministral3` text wrapper. They are covered only by the shared `supports_batching()` gate and the stub tests.
- **Not measured:** gpt-oss throughput (text only); VLM `--image` prefill on a non-batching wrapper, which also goes through `open_sequence` (the server applies the same reset before its embedding prefill).
- **Test gap (LOW):** `shared_cache_tests.rs` does not assert that a batching model sees exactly one `make_caches` per open. The gate is a single `if`, so this would be a regression guard only.
- **Design limit, not caused here:** two `DirectEngine` clients over the same non-batching model still share one `RefCell` cache set.
- **Pre-existing, tracked separately:** `NemotronNAS` reports `supports_batching() == false` but its `forward` indexes the caller's `caches` slice, which the model-owned pool entry leaves empty, so the engine path looks like it would panic at the first attention layer. `LongCat-Flash-Ngram` builds fresh caches on every forward (its `make_caches` has no reset side effect). Neither is changed by this PR.

## 5. Learning Points

- A model that keeps its caches inside itself needs a reset at every sequence boundary in every client that drives it, not only in the server. When the generation loop moved onto `DirectEngine`, the reset that `reset_with_model` used to provide was lost silently, because the model-owned pool entry looked complete.
- A warmup that shares state with the timed run can turn a correctness bug into a throughput regression. Diff the generated text against a reference build before attributing a decode A/B to the engine.
- Fit intercept and slope over several `-n` values before profiling per-token work. Here the intercepts (302.0 vs 70.5 ms at nearly equal slopes) pointed at a once-per-call cost, and a gdb breakpoint on `cuLibraryLoadData` showed which call loaded which kernels.
- A null arm (the same binary run again) bounds what a paired delta can claim. granite's +2.14% main vs fix looked like a cost of the extra reset until the null arm showed +1.55% on identical code.
