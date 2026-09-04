# Technical Report: PR #1623 - perf(qwen3_moe): batch the decode forward instead of per-row forwards

**Date**: 2026-09-04
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (attributed and measured on M1 Ultra; the M5 Max re-run is pending on that host)
**Languages**: Rust
**Risk Level**: Medium (decode path for one family; B=1 is routed to the pre-existing single-sequence graph by construction)

---

## Executive Summary

Issue #1616 attributed a batched-serving scaling gap to the fused MoE decode kernel declining at `n_tokens > 1` and falling back to `gather_qmm`. Profiling found a different cause and disproved the filed one.

`supports_batching()` returning true does not mean a family runs one forward per tick. Only families that override `forward_batched()` do. `Qwen3MoeModel` never did, so batched decode ran the `LanguageModel` default: the single-sequence `forward` once per row, evaluated together. Each of those rows carries exactly one routed token, so the fused kernel was reached on every row and the token-count gate was never the obstacle.

The fix is a real batched forward for the family. On M1 Ultra it lifts B=4 aggregate throughput 10.3% and per-request decode 13.8%, leaves B=1 byte-identical, and leaves both dense controls inside the run-to-run band.

---

## 1. Problem Statement

### 1.1 The filed hypothesis

`src/models/qwen3_moe.rs` gates the fused MoE kernel on `array_shape(&x_flat)[0] == 1`, and 16 production call sites across the model families carry the identical gate. The scheduler supplies a batch dimension as soon as there is more than one slot. The issue concluded that B>=2 therefore silently fell back to `SwitchGLU::forward` plus `gather_qmm` on all 48 layers, every tick, and proposed giving the kernel a token dimension.

### 1.2 What the profile found instead

With `MLXCEL_PROFILE_BLOCKS=1`, a B=4 tick was four serialized single-token graphs, 518 block summaries for 128 ticks. Each still launched the fused kernel. The graphs overlap only about 30% on this GPU: 12.3 ms per tick at B=1 against 33.6 ms at B=4.

The mechanism is that `supports_batching()` and `forward_batched()` are separate things. The first admits a family to the batched scheduler path; the second is what makes it a single forward. Llama 3, Llama 4, Qwen 3, Qwen 3.5, Gemma 3, Helium and Muse Glimmer override it. Every other batching family, Qwen3-MoE included, ran the default, whose aggregate scaling comes only from overlapping independent single-token graphs.

An `MLXCEL_FUSED_MOE=0` arm is the direct confirmation: it changed B=4 aggregate from 93.3 to 82.9 tok/s. If the fused kernel had been declined at B=4 as the issue assumed, disabling it could not have moved that number at all.

### 1.3 Why the proposed kernel was not built

An op-level microbenchmark of the real layer-0 expert planes settled the two remaining proposals:

- **Per-token fused launches scale linearly**: 3.9, 7.5, 14.7 and 29.2 ms per 48 layers at n=1/2/4/8. There is no per-launch overhead being amortized away by batching them.
- **Identical and disjoint expert sets cost the same at every n.** Expert-plane traffic is therefore not the limiter, which means deduplicating expert ids across the cohort, the issue's leading hypothesis, buys nothing.
- **A prototype of the proposed batched kernel loses.** Built with grid z of `k * n_tokens` and verified bit-identical to per-token launches, it is slower than `gather_qmm` from n=4 (11.1 ms against 9.3 to 9.7 ms).

The issue's acceptance criteria explicitly permit a documented explanation in place of a speedup when the profile does not support building. This is that case, recorded rather than silently skipped.

---

## 2. Technical Decisions

### 2.1 A batched forward, mirroring the dense Qwen3 port

`Attention::apply_rope_batched` applies the same rotation as `apply_rope` with one cache offset per batch row instead of one offset for the whole tensor, so frequency-table schemes go through the table launcher and the YaRN magnitude multiply is applied exactly as on the single-sequence path.

`Attention::forward_split_attention` takes the fused QKV projection of the whole batch as three `[B, T, proj_dim]` tensors, applies Q/K RMSNorm and RoPE while still batched, then runs the KV-cache update and SDPA per sequence because each row owns its own cache, and concatenates back. The paged-pool fast paths are not reproduced because this family does not opt into `supports_paged_decode_backend()`.

`DecoderLayer::forward_batched` runs the norms, the fused QKV projection, the output projection and the MoE block once over `[B, T, hidden]`, so from B=2 the experts take the multi-token `gather_qmm` chain. `Qwen3MoeModel::forward_batched_impl` batches the embedding, the final norm and the LM head.

### 2.2 B=1 is unchanged by construction, not by measurement

The `LanguageModel::forward_batched` override dispatches on cache count: zero returns the trait default's empty logits, exactly one delegates to the existing single-sequence `Qwen3MoeModel::forward` (fused kernel included), and only two or more reach the batched implementation. B=1 therefore takes the identical graph it took before the change, which is why byte-identity is a structural property here rather than a lucky measurement.

### 2.3 The profiling hook was itself wrong

`MLXCEL_PROFILE_QWEN3_MOE_DETAIL=1` always ran the gather path and never reported which path it timed, so a single-token trace attributed a kernel the production step never launched. `SparseMoeBlock::forward_profiled` now follows the production dispatch and returns a `MoeProfile` carrying `path` and `tokens`, so a batched step and a single-token step are distinguishable in a trace. A profiling hook that reports a path the production code does not take is worse than no hook, because it produces confident wrong attribution.

---

## 3. Validation

Apple M1 Ultra 128 GB, Metal, mlxcel 0.7.0-beta.1, MLX pin `9a795735`, `scripts/bench_serving_concurrency.py --prompt-tokens 512 --max-tokens 128` against `--parallel 4 --max-batch-prefill 4`.

| model | B=4 before | B=4 after | scaling before | scaling after |
|---|---|---|---|---|
| qwen3-30b-a3b-4bit | 93.3 | **102.9** | 1.72x | **1.86x** |
| llama-3.1-8b-4bit (control) | 120.5 | 118.9 | 2.11x | 2.09x |
| qwen2.5-0.5b-bf16 (control) | 367.5 | 369.5 | 1.70x | 1.77x |

Per-request MoE decode at B=4 goes from 29.8 to 33.9 tok/s. Both dense controls stay inside the run-to-run band, which is what rules out a host-wide shift between passes.

### 3.1 Correctness

Greedy `mlxcel generate -p Hello -n 128 --temp 0` on `qwen3-30b-a3b-4bit` is byte-identical before and after. Four simultaneous greedy 256-token requests at B=4 return identical rows with non-empty reasoning content, and match the single-sequence `MLXCEL_FUSED_MOE=0` output character for character.

That last equivalence is the precise statement of the one behavior change: B=4 output now matches the gather path rather than the fused path, because from B=2 the experts run `gather_qmm`. Before the change, B>1 ran the single-sequence path per row and so matched B=1 exactly. Cross-batch-size bitwise identity was never a guarantee here (#203), and this brings the family in line with every other family that overrides `forward_batched`.

`cargo test --workspace --profile test-fast --features metal,accelerate`: 10510 passed, 0 failed. New tests pin batched-equals-sequential on a synthetic three-expert model with per-row RoPE offsets, the one-row delegation, and the empty batch.

### 3.2 A measurement that had to be discarded

An earlier after-pass read 103.7 tok/s at B=4. A `cargo build --release` on the final commit recompiled, which proved that pass came from a binary built before the last edit to the branch. It was discarded and everything above was re-measured on the final commit. The conclusion is unchanged and the numbers move by about 1%. The workspace gate was re-run for the same reason, because the first gate had started while the tree was still moving.

---

## 4. Change Summary

| Metric | Value |
|---|---|
| Files changed (code PR) | 3 |
| Lines added | 605 |
| Lines removed | 24 |

- `src/models/qwen3_moe.rs`: `apply_rope_batched`, `forward_split_attention`, `DecoderLayer::forward_batched`, `Qwen3MoeModel::forward_batched_impl`, the `LanguageModel` override, and the corrected `forward_profiled`.
- `src/models/qwen3_moe_tests.rs`: batched-equals-sequential, one-row delegation, empty batch.
- `docs/CONTINUOUS_BATCHING.md`: names which families override `forward_batched` and what the default does for the rest.

### Landed separately

`benchmarks/metal_m1ultra_batch_2026-09-04.csv` and `docs/benchmark_results/moe-batched-decode-m1ultra-2026-09-04.md`, plus the M1 Ultra batched-serving section and a pointer on the M5 Max one, are on `bench/0.7.0-refresh` (PR #1617) as commit `e75fa0e44`. The CSV's `mlxcel_commit` column separates the two passes. This split means the code PR can merge and close the issue while the documentation half is still unmerged.

### Related issues

Closes #1616. Related: #268 (the B=1 fused kernel this profile found was reachable all along), #725, #628, #632, #203 (cross-batch-size bitwise identity is not a guarantee).

---

## 5. Follow-up Actions

- The other 15 MoE families carrying the same `== 1` gate still run the per-row fallback. Each needs its own `forward_batched` after the same premise check; families sharing `switch_layers::SwitchGLU` get the multi-token `gather_qmm` path for free once their attention is batched.
- Prompt-cache adoption for dense-backed families is one-shot, so N simultaneous identical prompts pay N-1 cold prefills. The paged backend's clone adoption is the fix pattern. This is what the TTFT growth in the issue's table actually measures, not tick length.
- Dense batched decode on this host reaches only 2.09x against M5 Max's 3.25x. Its per-sequence attention loop is the likely dispatch cost and deserves its own profile.
- The M5 Max re-run of this ladder is pending on that host.

### Transferable lesson

The issue named a gate, and the gate was real, reachable and irrelevant. Two capabilities that sound like one thing were separate: a family can be admitted to the batched scheduler path and still never run a batched forward. The check that settled it cost one environment variable, because disabling the fused kernel moved a number the hypothesis said it could not touch. When a hypothesis predicts that some lever has no effect, pulling that lever is usually the cheapest way to test it.
