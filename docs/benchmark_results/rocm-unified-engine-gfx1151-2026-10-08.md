# ROCm verification of the unified engine: Radeon 8060S (gfx1151), 2026-10-08

Issue lablup/mlxcel#2192, Phase 7 of epic #2166. The epic (Phases 0 to 6, PRs #2194, #2201, #2204, #2205, #2210, #2217, #2224, #2225) was implemented on a CUDA host, so every ROCm-only test and every test that skips without a kernel port ran here for the first time against the merged engine. This page records the gate, the named tests, where each ROCm behavior now lives on the engine path, the ROCm parity measurement, and the performance of the engine against the pre-epic `CxxGenerator` on the same host, same day, interleaved.

Raw records, logs and the drivers are in `data/rocm-unified-engine-gfx1151-2026-10-08/`.

## Environment

| Item | Value |
|---|---|
| **Hardware** | AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5, wave32, 20 CUs), 32 CPU threads, 96 GiB VRAM carve-out, 30 GiB visible to the host |
| **OS / ROCm** | Debian 13, kernel 6.18.12+deb13-amd64, ROCm 10.0.0, HIP 7.15.26333 |
| **mlxcel, post-epic** | `main` at `b2fda840` (PR #2231; the last epic phase is `30ceea22`, PR #2225), release and `test-fast` builds with `--features rocm` |
| **mlxcel, pre-epic baseline arm** | `4c44e317` (PR #2194, the Phase 0 merge: its decode code is the pre-epic `CxxGenerator`, and `mlxcel-bench-decode` times that loop), built from a `git archive` of that commit with the same toolchain |
| **MLX pin / ROCm overlay** | MLX `81ba1c6a`, NripeshN/mlx `rocm-support` `75915908` plus `patches-rocm/LOCAL_FIXES.md` |
| **Toolchain** | Rust 1.97.1, AMD clang 23 (hipcc), cmake 3.31 |
| **Models** | `models/mlx/`: Qwen3-0.6B-4bit, Meta-Llama-3.1-8B-Instruct-4bit, gemma-3-4b-it-4bit, Qwen3-30B-A3B-4bit, granite-4.0-h-tiny-4bit, NVIDIA-Nemotron-3-Nano-30B-A3B-4bit, gpt-oss-20b-MXFP4-Q4, Mixtral-8x7B-Instruct-v0.1-4bit |
| **Host sharing** | Four other units shared the GPU and the CPU through `scripts/rocm_gpu_guard.sh`'s host-wide lock. Every measurement below ran inside one guard window (`data/.../guard-*.log`); the test runs did not need the guard and were not measurements. |

## Gate

`make verify-rocm` with `MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit` on the PR branch rebased onto `main` at `1b4e3657` (2026-10-09): every prerequisite passed (versions, dtype keys, kernel-port dispatch, llama-compat, overlay records, Python tooling, fmt, `clippy --features rocm -D warnings`, the smoke generate on the GPU), and `verify-test-rocm` reported **12194 passed, 0 failed, 403 ignored** across 163 test binaries (`[verify-rocm] OK`). The pre-epic reference is PR #2188's 11987 passed, 0 failed, 382 ignored across 152 binaries; the epic and the ROCm PRs merged since added the difference. The fast gates (`verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`) are prerequisites of the same target and passed in it.

## Named tests

Every test the issue names ran under its own narrow selector (`data/.../harness/named_tests.sh`, logs in `data/.../named/`), single-threaded, with the kill switches unset. No run printed a `skipping` line.

| Selector | Result |
|---|---|
| `--test sampling_gumbel_kill_switch` | 1 passed |
| `--test sampling_rejection_kill_switch` | 1 passed |
| `--test rocm_slice_update_source` (rocm-only) | 6 passed |
| `--test rocm_slice_update_reduce` | 8 passed |
| `--test rocm_wave64_port_refusal` | 1 passed |
| `--test rocm_wave_size` (rocm-only) | 3 passed, 1 ignored (the child-process half of `force_warp_size_does_not_move_the_device_warp_size`) |
| `--test rocm_gather_qmm_expert_batched` (rocm-only) | 2 passed |
| `--test rocm_inflight_bound` (rocm-only) | 1 passed |
| `-p mlxcel-core --lib sampling_fixed_key_tests` | 5 passed |
| `-p mlxcel-core --lib sampling_rejection_tests` | 28 passed |
| `-p mlxcel-core --lib sampling_gumbel_tests` | 14 passed |
| `-p mlxcel-core --lib lang_bias_counters::tests::override_is_scoped_nested_and_thread_local` | 1 passed |
| `-p mlxcel-core --lib config_supports_fused_batch_false_for_token_bias` | 1 passed |
| `-p mlxcel-core --lib rotating_inplace_warmup_and_wrap_keep_shared_snapshots_intact` | 1 passed |
| `-p mlxcel-core --lib rotating_steady_state_snapshot_survives_the_next_wrap_write` | 1 passed |
| `-p mlxcel-core --lib paged_pool_offset_tests` | 2 passed, 1 ignored (see below) |
| `-p mlxcel-core --lib paged_pool_past_u32_elements_matches_gather -- --ignored` | 1 passed (run from the prebuilt test binary inside a guard window with the GPU otherwise idle, `data/.../named/paged_pool_past_u32.log`) |
| `-p mlxcel-core --lib validate_rejects_a_merge_input_past_the_u32_index_range` | 1 passed |
| `-p mlxcel-core --lib kernel_port_tests` | 2 passed |
| `-p mlxcel-core --lib fused_905_defaults_are_on_for_rocm_builds_only` | 1 passed |
| `-p mlxcel-core --lib fused_norm_parity_tests` | 10 passed |
| `-p mlxcel-core --lib ssm_update_parity_tests` | 5 passed (the two bf16 cases that fail on GB10 pass on gfx1151) |
| `-p mlxcel-core --lib fused_moe_parity_tests` | 5 passed |
| `-p mlxcel-core --lib fused_moe_relu2_parity_tests` | 1 passed |
| `-p mlxcel --lib pooling_cache_remainder_survives_an_overlapping_tail_write` | 1 passed |
| `-p mlxcel --lib the_rope_append_bypass_notice_needs_an_explicit_truthy_value` | 1 passed |
| `-p mlxcel --lib switch_layers::mxfp_tests` | 5 passed |
| `-p mlxcel --lib runtime::tests::cache_limit_` | 4 passed |
| `-p mlxcel --lib scheduler_model_owned_lookahead_tests` | 7 passed |
| `-p mlxcel --lib model_owned_sequence_primes_the_decode_lookahead_only_with_a_rewind` | 1 passed |
| `-p mlxcel --lib sampling_observability_tests` | 6 passed |

## Where each ROCm behavior lives after the epic

The issue's file:line anchors predate Phases 1 to 6. Each row names the current location and the engine-path caller that reaches it (verified at `b2fda840`).

| Behavior | Now | Reached from the engine by |
|---|---|---|
| Sampler routing reads the port tables (PR #2100) | `rejection_sample_supported()` at `src/lib/mlx-cpp/turbo/sampling_rejection.cpp:803`; `sampling_rejection_available` / `_backend_supported` at `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp:5703` and `:5710`; the fused dispatcher `fused_sample_impl` at `:6584` | `Engine::step` (`engine/mod.rs:246`) to `sample_and_finish_row` (`engine/rows.rs:270`) to `RowSampler::sample` (`sampling_row_step.rs:220`) to `fused_sample_dispatch` (`sampling.rs:560`) to `ffi::fused_sample`; the batched branch `batched_fused_sample_tokens` (`sampling.rs:1141`) from `Engine::submit`. `DirectEngine` builds the same `RowSampler` (`engine/direct_decode.rs:67`), the scheduler in `admission.rs:539` and `handoff.rs:359`. |
| Lang-bias counter gate is thread-scoped (PR #2188) | `lang_bias_counters.rs:49` `enabled()`, `:86` `scoped_override` | The only production readers are `apply_token_bias_stage` (`sampling.rs:692`) and `RowSampler::fused_eligible` (`sampling_row_step.rs:269`, reached through `row_supports_fused_batch_except_bias` and `engine::shared_fused_params`); the env var is read nowhere else. |
| SliceUpdate does not donate a held source (PR #2070, LOCAL_FIXES 24) | overlay `indexing.hip` and `gpu/primitives.cpp` unchanged; tests at `cache.rs:7818`, `:7859`, `deepseek_v4_tests.rs:888` | `RotatingKVCache::attend` (`cache/attend.rs:152`) to `update_and_fetch` from `gemma3.rs:321`; DeepSeek V4 `pool.accumulate_windows` (`deepseek_v4_compress.rs:445`). |
| Paged decode on ROCm (PRs #2103, #2186, #2179) | `PagedDecodeBackend` still at `layers.rs:5659`; the u32 plan check at `paged_v2/plan.rs:335`; `port_for` wave64 hold at `turbo/kernel_port.cpp:30-45` | `KVCache::attend` (`cache/attend.rs:232-245`) and `attend_batched` (`:343-349`) to `paged_batch_decode_attention` (`cache/paged_batch_decode.rs:307`) to `PagedBlockPool::paged_decode_batched` (`cache/paged.rs:2069`), which validates the plan (`:2207`) and launches v2 or the gather fallback. The ROCm grid-z guard is in `paged_decode_fused` (`paged.rs:2323`). The server log line `paged decode v2: fused v2 launch` confirms the HIP port ran (parity section). `PagedDecodeBackend::Rocm` itself is consumed by the library-only `paged_decode_attention_pooled` (`layers.rs:6049`), as before the epic. |
| Fused add-RMSNorm and RoPE-append default on (PR #2189) | `FUSED_ADD_RMSNORM_DEFAULT` `layers.rs:863`, `FUSED_ROPE_APPEND_DEFAULT` `:883`, `fused_flag_explicit_value` `:934`; `report_fused_rope_bypass_once` `src/models/llama3.rs:307` | `forward_fused_rope_append` (`layers.rs:3593`) from `llama3.rs:686` with `FusedRopeDestLayout::DenseSlab`, whose `[B, Hkv, L, D]` output `cache.attend` appends on both storages (dense `update_and_fetch`; pool-backed `write_paged`, which asserts the same order at `cache.rs:3374`). `PagedPool` (`dest_layout=1`) had no caller before the epic and has none now. |
| bf16 qmm route and dense-prefill gate (PR #2085), expert-batched gather (PRs #2112, #2164) | overlay `qmm.hip` unchanged; `quantized_matmul_matches_dense_gemm` at `mlx_cxx_bridge.cpp:5352`, `prefill_dense_gemm_eligible` at `layers.rs:734` | `UnifiedLinear::forward_inner` (`layers.rs:2723`), so every prefill chunk the `PrefillPlan` forwards. The overlay's own `select_qmm_route` runs inside every quantized matmul. |
| In-flight memory bound and cache limit (PRs #2084, #2195) | `MLX_ROCM_MAX_INFLIGHT_MB` in overlay `device.cpp`; `resolve_cache_limit` `src/execution/runtime.rs:372`; `estimate_total_memory` adds `backend_inflight_reserve_bytes()` at `memory_estimate.rs:577` | `initialize_runtime_checked` on `generate` (`commands/generate.rs:2721`), `run` (`run.rs:277`), chat (`chat.rs:174`), the server (`startup.rs:3142`) and `bench_decode` (`:488`). `mlxcel run --estimate-memory` stays on the `generate` flow (`run.rs:218`) and printed `Backend in-flight: 1.00 GiB (MLX_ROCM_MAX_INFLIGHT_MB)` here; the in-process server sizes its paged budget and prompt-cache snapshots through the same estimate (`model_worker.rs:968`). |
| SSM update HIP port (PR #2099), fused MoE port (PR #2098) | `ssm_ports()` `mlx_cxx_kernels.cpp:840`, `ssm_kernel_available` `:1798` (`MLXCEL_SSM_KERNEL` at `:1803`); `MLXCEL_FUSED_MOE_SGY` ROCm default 2 at `:1664` and `:3174` | Model forwards (unchanged by the epic): the parity tests above ran the kernels. |
| Gemma 3 lookahead rewind (PR #2182) | `supports_decode_lookahead_rewind` `generate.rs:439`, `rewind_decode_appends` `:453`; `Engine::can_unwind_lookahead` `engine/lookahead.rs:68`; undo log `cache/decode_undo.rs` | `decode_tick.rs:358` (`lookahead_params`) and `engine/direct_decode.rs:163`. One greedy chat request of 62 tokens on a one-slot `mlxcel serve -m gemma-3-4b-it-4bit --metrics` server took `mlxcel_batch_decode_lookahead_steps_total` from 0 to 60 (`data/.../gemma3-lookahead/`, `data/.../harness/gemma3_probe_session.sh`): the lookahead primed and every steady step after the prime ran pipelined. The scheduler tests above (`lookahead_is_gated_on_the_rewind_capability`, `..._only_with_a_rewind`) pass single-threaded under `--features rocm`. |

## ROCm parity

`MLXCEL_SDPA_DETERMINISTIC` is a CUDA-overlay variable; the runs below set it as the harness expects but it has no effect on ROCm. Prompt cache off everywhere except the harness's own miss/hit rows. Drivers: `data/.../harness/parity_session.sh`, outputs under `data/.../parity/<model>/`.

### `make engine-parity` (64 tokens, greedy and seeded with penalties plus DRY)

`mlxcel-engine-parity` arms: `a:cli` (`mlxcel generate`'s call of `DirectEngine`), `d:engine` (`DirectEngine` directly), `b:dense` (scheduler, B=1, dense storage), `c:paged` (scheduler, B=1, paged storage), `e:run` (`mlxcel run -p`, one slot) and `f:server` (`mlxcel-server` defaults, four slots, paged when supported), plus `b:dense` with the prompt cache on (miss, then hit).

| Model | Pairs | Identical | Diverged | n/a |
|---|---|---|---|---|
| Qwen3-0.6B-4bit | 14 | 12 | 1: seeded `pc:miss` vs `pc:hit` at token 6 | 1 (seeded `f:server` vs `e:run`, chat arms are greedy only) |
| Meta-Llama-3.1-8B-Instruct-4bit | 14 | 12 | 1: seeded `pc:miss` vs `pc:hit` at token 18 | 1 (same) |
| granite-4.0-h-tiny-4bit (Mamba2 hybrid MoE) | 14 | 7 | 2: `b:dense` vs `b:dense+pc:miss`, greedy at token 47 and seeded at token 0 | 5 (`c:paged` is unsupported for the family and the worker fell back to dense; seeded chat arm) |

Every CLI-vs-server pair is identical on all three models: `a:cli` vs `b:dense`, `b:dense` vs `d:engine`, `f:server` vs `e:run`, and on the two dense families `a:cli` vs `c:paged` and `b:dense` vs `c:paged` for both cases. The seeded miss-vs-hit row is the documented prompt-cache divergence (the hit forwards only the suffix). granite's `pc:miss` arm splits the prefill at the history boundary (`prefill[0..45)+prefill[45..48)`) where the cache-off plan forwards `prefill[0..48)`; that is a different partition, which `docs/CONTINUOUS_BATCHING.md` says changes near-tie tokens, and on this Mamba hybrid it also moves the SSM chunk boundary. Not an epic regression (the split is #1143's), recorded as a per-family divergence.

Paged storage at B=1 below the 4096-token `MLXCEL_PAGED_V2_MIN_KV_TOKENS` floor decodes through the gather-then-SDPA fallback, so with a 48 to 75-token prompt the `c:paged` arm's `paged-kernel-launches` (1764 to 2080, fused plus fallback) were fallback launches and the dense arithmetic again. The harness was therefore rerun with `MLXCEL_PAGED_V2_MIN_KV_TOKENS=0`, which sends every paged decode step to the HIP paged v2 kernel (the path a request past 4096 KV tokens takes); the server log then reports `paged decode v2: fused v2 launch` and nothing else (`data/.../parity/floor0/`):

| Model (floor 0) | Greedy `a:cli` / `b:dense` vs `c:paged` | Seeded `a:cli` / `b:dense` vs `c:paged` |
|---|---|---|
| Qwen3-0.6B-4bit | identical, identical | diverge at token 6 (1191 vs 1430), both pairs |
| Meta-Llama-3.1-8B-Instruct-4bit | identical, identical | diverge at token 18 (18921 vs 18435), both pairs; the same token and index as the seeded miss-vs-hit row |

With the HIP kernel in the loop the greedy stream is still identical over 64 tokens, and the seeded stream, which samples from the whole distribution with penalties and DRY, flips at a near-tie: the recorded dense-vs-paged divergence index on gfx1151 is 6 (Qwen3-0.6B) and 18 (Llama-3.1-8B) for the seeded case and none for greedy.

### `mlxcel generate` vs `mlxcel run` vs the HTTP server (raw prompt, greedy, 64 tokens)

One 211-token raw prompt (`tests/fixtures/wikitext2_excerpt.txt` head, `--no-chat-template`, `/completion`): on Qwen3-0.6B, granite-4.0-h-tiny and Llama-3.1-8B the continuation printed by `generate`, by `run`, by a one-slot dense server and by a two-slot paged server is the same text.

### Teacher-forced decode-step trace, dense server vs paged server (128 positions)

`data/.../harness/paged_vs_dense_trace.py` records per decode step the top-10 ids and log-probabilities the server's `/completion` reports (`n_probs`), teacher-forced along the dense server's greedy stream (a restart from `prompt + reference[:k]` after any divergence, none was needed), and `scripts/compare_logit_traces.py --decided 2.0` counts the positions where the two arms' argmax differs and the reference's top-two gap is above 2.0.

| Model | Paged arm | First divergence | Top-1 disagreement | Decided mismatches |
|---|---|---|---|---|
| Qwen3-0.6B-4bit | gather fallback (shipped floor) | identical | 0 / 128 | 0 / 86 |
| Meta-Llama-3.1-8B-Instruct-4bit | gather fallback (shipped floor) | identical | 0 / 128 | 0 / 44 |
| granite-4.0-h-tiny-4bit | n/a (fell back to dense) | identical | 0 / 128 | 0 / 69 |
| Qwen3-0.6B-4bit | HIP paged v2 kernel (`MLXCEL_PAGED_V2_MIN_KV_TOKENS=0`) | identical | 0 / 128 | 0 / 86 |
| Meta-Llama-3.1-8B-Instruct-4bit | HIP paged v2 kernel (`MLXCEL_PAGED_V2_MIN_KV_TOKENS=0`) | identical | 0 / 128 | 0 / 44 |

The paged server reports the storage it resolved and the dispatch it took in its log (`Starting BatchScheduler (... decode_storage=paged)`, then `paged decode v2: fused v2 launch` once at the first dispatch); `/metrics` carries no paged-kernel counter, so the per-step evidence is the dispatch rule (every lone request above the floor takes v2) plus the harness's own launch counter. The dense server's trace is byte-identical between the two sessions (default floor and floor 0), so the two candidate rows compare against the same reference. On this host and these prompts the HIP paged kernel and MLX SDPA agree at every greedy decode step, decided or not; the seeded flips above are the only place the reduction order shows.

### Definition applied

ADR 0007's ROCm paragraph: the CLI and the server engine with dense storage must be byte-identical (they are, for every model and case above), and the dense-vs-paged pair is recorded per model as a first-divergence index plus a decided-position mismatch count rather than gated. The record is in ADR 0007 ("What parity means on ROCm and Metal") and `docs/CONTINUOUS_BATCHING.md`.

## Performance against the pre-epic baseline

### Method

`data/.../harness/bench_rounds.sh`: for each (model, prompt length) cell, three interleaved rounds of three arms, each a fresh `mlxcel-bench-decode` process at `scripts/bench_decode.sh`'s shape (`-n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens N`, greedy): `base` is the pre-epic binary (`4c44e317`, `CxxGenerator`), `new` is `main` (`DirectEngine`, `Decode path: engine`), and `base-null` is the pre-epic binary again, the method's noise floor. The arm order rotates each round. Every run was its own `scripts/rocm_gpu_guard.sh` command inside one host-wide lock hold (`data/.../harness/full_session.sh`): 10 s of idle GPU and no compiler before each run, a 1 Hz monitor during it, and all 99 runs were CLEAN on the first attempt (`data/.../guard/guard-logs.tar.gz`). The session ran 2026-10-09 01:19 to 02:43 KST. `data/.../harness/summarize_rounds.py` computes the medians and the paired per-round deltas below (median and range over the three rounds, percent of `base`). The threshold is ADR 0007's 1.0 percent on the paired decode delta.

### Decode, `mlxcel generate` (DirectEngine) vs `CxxGenerator`

| Model | Prompt | base decode tok/s | new decode tok/s | new vs base | null vs base | Verdict |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-4bit | 512 | 275.26 | 274.09 | -0.57 % (-0.83..-0.14) | +0.52 % (-0.39..+0.64) | pass |
| Qwen3-0.6B-4bit | 2048 | 218.07 | 216.59 | -0.71 % (-2.76..+2.42) | -0.24 % (-2.07..+0.71) | pass |
| Meta-Llama-3.1-8B-Instruct-4bit | 512 | 37.68 | 37.67 | +0.00 % (-0.79..+1.19) | -0.08 % (-0.16..+0.19) | pass |
| gemma-3-4b-it-4bit | 512 | 69.19 | 68.90 | -0.56 % (-0.81..+0.04) | -0.14 % (-0.30..+0.85) | pass |
| gemma-3-4b-it-4bit | 2048 | 62.65 | 62.08 | -1.19 % (-1.39..-0.86) | -0.03 % (-0.48..-0.02) | **miss**, see below |
| Qwen3-30B-A3B-4bit | 512 | 62.53 | 62.52 | -0.02 % (-0.13..+0.26) | +0.21 % (-0.03..+0.37) | pass |
| Qwen3-30B-A3B-4bit | 2048 | 58.11 | 58.14 | +0.05 % (-0.55..+0.40) | +0.09 % (-0.74..+0.19) | pass |
| granite-4.0-h-tiny-4bit | 512 | 89.52 | 88.99 | -0.59 % (-1.29..+0.18) | +0.02 % (-0.62..+0.28) | pass |
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit | 512 | 75.29 | 75.51 | +0.20 % (+0.17..+0.31) | +0.21 % (+0.08..+0.28) | pass |
| gpt-oss-20b-MXFP4-Q4 | 512 | 8.46 | 8.49 | -0.12 % (-0.23..+1.19) | +0.24 % (-0.23..+0.60) | pass |
| Mixtral-8x7B-Instruct-v0.1-4bit | 512 | 9.97 | 10.09 | +1.20 % (-1.55..+2.31) | +1.81 % (-1.71..+8.05) | pass (inside the null spread) |

Ten of eleven cells are within the threshold; most are within their own null spread, so the engine decodes at the pre-epic rate on gfx1151 for dense, MoE, mxfp4 and Mamba-hybrid checkpoints. The one miss is gemma-3-4b-it-4bit at the 2048-token prompt: -1.19 percent, every round between -0.86 and -1.39 percent, against a null range of -0.48 to -0.02 percent. An isolating run (`data/.../harness/extra_session.sh`, `data/.../gemma3-pp2048/`) added a third arm, `new-sync`, the same `main` binary with `MLXCEL_FORCE_SYNC=1`, which keeps `DirectEngine`'s B=1 decode synchronous instead of pipelined, three interleaved rounds:

| Arm | Decode tok/s by round | Median | vs base |
|---|---|---|---|
| base (`CxxGenerator`) | 62.66, 62.82, 62.61 | 62.66 | |
| new (`DirectEngine`, pipelined) | 61.93, 61.83, 61.67 | 61.83 | -1.3 % |
| new-sync (`DirectEngine`, `MLXCEL_FORCE_SYNC=1`) | 63.62, 63.69, 63.50 | 63.62 | +1.5 % |

Every pipelined round is below every base round, and every synchronous round is above every base round, so the 1.2 percent is the B=1 lookahead pipeline on Gemma 3's rotating cache at long context (the model-owned rewind path, #2182, where each pipelined step past the 1024-token sliding window records an undo row), not the engine's forward or sampling. At 512 tokens, inside the window, the pipelined arm is within the threshold. The synchronous step is the faster choice for this family on this host; the pipeline's policy is the engine's, shared with CUDA, so this is filed as its own issue rather than changed here (see Regressions below). The other ten cells have no such gap: the engine, pipelined, decodes at the pre-epic rate.

### Prefill

| Model | Prompt | base prefill tok/s | new prefill tok/s | new vs base | null vs base |
|---|---|---|---|---|---|
| Qwen3-0.6B-4bit | 512 | 7424.3 | 8559.1 | +14.76 % (+0.22..+22.67) | +4.97 % (+3.85..+10.85) |
| Qwen3-0.6B-4bit | 2048 | 4514.9 | 4506.2 | +0.62 % (-8.81..+1.74) | -1.48 % (-3.44..+2.30) |
| Meta-Llama-3.1-8B-Instruct-4bit | 512 | 1008.0 | 996.5 | -0.86 % (-45.52..+0.02) | -1.12 % (-1.39..-0.74) |
| gemma-3-4b-it-4bit | 512 | 2155.8 | 2207.6 | +3.49 % (+1.16..+107.25) | +3.65 % (+0.11..+110.26) |
| gemma-3-4b-it-4bit | 2048 | 2897.6 | 2914.9 | +0.62 % (-0.32..+2.51) | +0.46 % (+0.44..+2.52) |
| Qwen3-30B-A3B-4bit | 512 | 283.2 | 269.0 | -5.02 % (-11.61..+21.29) | +8.17 % (-8.82..+26.90) |
| Qwen3-30B-A3B-4bit | 2048 | 286.6 | 286.5 | -0.95 % (-1.23..+1.29) | -1.64 % (-1.76..+0.15) |
| granite-4.0-h-tiny-4bit | 512 | 572.7 | 588.6 | +1.51 % (-20.78..+31.25) | -0.08 % (-34.79..+31.88) |
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit | 512 | 285.2 | 275.9 | -3.29 % (-11.63..+6.31) | -2.86 % (-4.43..-1.29) |
| gpt-oss-20b-MXFP4-Q4 | 512 | 526.9 | 509.6 | -3.52 % (-3.52..+9.40) | +0.05 % (-0.23..+12.91) |
| Mixtral-8x7B-Instruct-v0.1-4bit | 512 | 26.0 | 25.8 | -0.62 % (-1.08..+0.70) | -0.27 % (-0.73..+0.27) |

A 512-token prompt is one chunk on both builds (2048 was already the CLI's chunk, so Phase 3's chunk change does not reach `generate`), and a single 0.1 to 2 s prefill measured once per process is noisy on this UMA host: the null ranges reach tens of percent on the MoE and hybrid models (the first prefill after a cold load pays the weight paging). The one prefill delta that clears its null range is Qwen3-0.6B at 512 tokens, where the engine is faster (+14.8 percent, null +5.0 percent). The expert-batched `gather_qmm` prefill and the bf16 qmm route are exercised by every row here (MoE prefill at 512 tokens is the `B >= 64` shape the kernel takes; the pp2048 rows are the shape of the qmm-route page), and none of those rows moved beyond its null range.

### Against the published pre-epic pages

The rows the issue names, with today's `base` arm as the control for host drift since each page was measured (same binary lineage, same day as `new`).

| Model | Shape | Published pre-epic page | Published tok/s | base today | new today |
|---|---|---|---|---|---|
| Qwen3-0.6B-4bit | pp512 decode | [baseline 09-30](rocm-baseline-gfx1151-2026-09-30.md) | 278.48 | 275.26 | 274.09 |
| Meta-Llama-3.1-8B-Instruct-4bit | pp512 decode | [fused norm/RoPE 10-05](rocm-fused-norm-rope-gfx1151-2026-10-05.md) default arm | 38.09 | 37.68 | 37.67 |
| Qwen3-30B-A3B-4bit | pp512 decode | [fused MoE 10-05](rocm-fused-moe-gfx1151-2026-10-05.md) | 62.51 | 62.53 | 62.52 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit | pp512 decode | [fused MoE 10-05](rocm-fused-moe-gfx1151-2026-10-05.md) | 75.09 | 75.29 | 75.51 |
| granite-4.0-h-tiny-4bit | pp512 decode | [SSM update 10-04](rocm-ssm-update-kernel-gfx1151-2026-10-04.md) | 88.43 | 89.52 | 88.99 |
| granite-4.0-h-tiny-4bit | pp512 prefill | [MoE prefill 10-05](rocm-moe-prefill-gfx1151-2026-10-05.md) | 918.59 | 572.7 | 588.6 |
| Mixtral-8x7B-Instruct-v0.1-4bit | pp512 prefill | [MoE prefill 10-05](rocm-moe-prefill-gfx1151-2026-10-05.md) | 125.63 | 26.0 | 25.8 |
| gpt-oss-20b-MXFP4-Q4 | pp512 prefill / decode | [MoE prefill mxfp4 10-06](rocm-moe-prefill-mxfp4-gfx1151-2026-10-06.md) | 526.28 / 8.49 | 526.9 / 8.46 | 509.6 / 8.49 |
| Qwen3-0.6B-4bit | pp2048 prefill | [bf16 qmm route 09-30](rocm-bf16-qmm-route-gfx1151-2026-09-30.md) | 4236 | 4514.9 | 4506.2 |
| gemma-3-4b-it-4bit | pp2048 prefill | [bf16 qmm route 09-30](rocm-bf16-qmm-route-gfx1151-2026-09-30.md) | 2815 | 2897.6 | 2914.9 |
| Qwen3-30B-A3B-4bit | pp2048 prefill | [bf16 qmm route 09-30](rocm-bf16-qmm-route-gfx1151-2026-09-30.md) | 297 | 286.6 | 286.5 |

Decode matches every page within the day-to-day spread, and where today's numbers differ from a page the `base` arm differs by the same amount (Qwen3-0.6B decode -1.2 percent, Qwen3-30B pp2048 prefill -3.5 percent), so the difference is the host, not the engine. Two prefill rows are far below their page: granite-4.0-h-tiny (573 vs 919 tok/s) and Mixtral-8x7B (26 vs 126 tok/s), on `base` and `new` alike, so the epic did not cause it. An isolating session (`data/.../harness/moe_prefill_session.sh`, `data/.../moe-prefill/`) found the expert-batched prefill kernel is never selected: `rocprofv3` kernel stats of the granite run list the per-row `gather_qmv_wide_kernel` and no `gather_qmv_expert_batched_kernel`, and `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0` changes nothing (575 vs 585 tok/s; `MLXCEL_FUSED_MOE=0` 572 to 579). The cause is PR #2137 (2026-10-06, before the epic): the sorted prefill path now passes an explicit `lhs_indices`, and upstream `gather_qmm` builds `GatherQMM` with `right_sorted = sorted_indices && !lhs_indices`, the flag every backend's sorted MoE prefill kernel keys on (ROCm `qmm.hip`, Metal `quantized.cpp:2003`, CUDA `quantized/quantized.cpp:368`). gpt-oss (mxfp4) keeps its rate because `GptOssSwitchGLU` takes another path. Filed as lablup/mlxcel#2241 (see Regressions).

### The server engine path (`mlxcel run` and chat), `mlxcel-bench-engine`

`mlxcel-bench-engine --path both --decode-storage dense`, 128 tokens after a 20-token warmup, three interleaved rounds per build (`data/.../engine/bench_engine_rounds.csv`). `cli` is `CxxGenerator` on `base` and `DirectEngine` on `new`; `server` is the in-process scheduler at one slot with dense storage, the path `run` and chat take since #2173.

| Model | Prompt | base cli | new cli | new vs base, cli | base server | new server | new vs base, server |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-4bit | 256 | 290.52 | 289.26 | -0.44 % (-0.57..-0.10) | 286.38 | 285.42 | -0.25 % (-0.80..-0.05) |
| Qwen3-0.6B-4bit | 2048 | 223.56 | 223.09 | -0.03 % (-1.83..+0.02) | 216.97 | 215.69 | -2.99 % (-29.13..+1.88), unresolved |
| Meta-Llama-3.1-8B-Instruct-4bit | 256 | 37.64 | 37.65 | +0.25 % (-0.60..+0.39) | 36.99 | 37.04 | +0.11 % (-0.29..+1.68) |
| Meta-Llama-3.1-8B-Instruct-4bit | 2048 | 36.57 | 36.48 | -0.23 % (-7.78..+0.15) | 35.97 | 35.94 | +0.67 % (-0.89..+11.65) |

The server engine at one slot decodes at the same rate before and after the epic on gfx1151 (the 1 to 2 percent gap GB10 measured between `CxxGenerator` and the server engine, #2227, does not appear here: the server arm sits 1 to 3 percent below the CLI arm on both builds). The Qwen3-0.6B 2048 server cell has one round at 153.8 tok/s against 215 to 222 in the others and is unresolved; its `base` arm's own spread (216.6 to 222.3) is already wider than the threshold. TTFT at the 2048-token prompt on the server arm dropped from 692 to 502 ms (Qwen3-0.6B) and 2640 to 1856 ms (Llama) with the 2048-token prefill chunk.

## Regressions and findings

- **gemma-3-4b-it-4bit pp2048 decode, -1.2 percent (threshold miss), epic-introduced.** Isolated to the B=1 lookahead pipeline on the model-owned rotating cache past its window (above); the synchronous step is 1.5 percent above the pre-epic loop. The policy is shared with CUDA, so it is filed as lablup/mlxcel#2239 rather than changed here.
- **MoE affine prefill 1.7x to 4.8x below the 2026-10-05 pages, not epic-introduced.** granite-4.0-h-tiny and Mixtral-8x7B lost the expert-batched prefill kernel when PR #2137 cleared `right_sorted` on the sorted path (above); Metal and CUDA key their sorted MoE prefill kernels on the same flag, so the regression is cross-backend. Filed as lablup/mlxcel#2241, outside this issue's scope (not introduced by the epic, and the fix touches every backend).
- **`PagedDecodeBackend::Rocm` and `FusedRopeDestLayout::PagedPool`** are, as before the epic, consumed only by library-only entry points (`paged_decode_attention_pooled`, no caller of layout 1); the engine reaches the HIP paged kernels through `KVCache::attend`. Not a regression; recorded so the next reader does not look for them on the engine path.
- The stale comment in `src/models/llama3.rs` that described the fused RoPE-append output as what `update_and_fetch` splices (it is what `cache.attend` appends on either storage since #2171) is corrected in the PR for this issue.

## Tooling

- `mlxcel-bench-decode` on `main` prints `Prompt tokens:`, `Generated tokens:`, `Prefill:`, `Decode:`, `Decode path: engine`, `MLX peak memory:` and the `[Memory] <phase>: active ... cache ... phase peak ...` lines; the pre-epic binary prints the same lines without `Decode path:` (`data/.../rounds/raw/`). `scripts/bench_decode.sh`'s `parse_profile` / `parse_decode_path` sed patterns and `scripts/rocm_decode_profile.py`'s `DECODE_RE` / `PREFILL_RE` match them.
- `python3 -m unittest tests/test_rocm_decode_profile.py`: 26 tests, OK.
- `scripts/rocm_decode_profile.sh` end to end on Meta-Llama-3.1-8B-Instruct-4bit (`data/.../profile/`): the plain run read 37.79 tok/s decode and 1005.8 tok/s prefill, the `rocprofv3` run 35.02 and 1074.8 (7.9 percent profiler slowdown), 58992 decode dispatches cut by the `[phase]` marks, and the per-kernel table and summary were written, so the marks and the `Decode:` / `Prefill:` lines of the engine-path bench parse as before.

## Reproduce

1. Build `main`: `cargo build --release --features rocm` and `cargo test --workspace --profile test-fast --features rocm --no-run` (set `OPENSSL_INCLUDE_DIR` and `OPENSSL_LIB_DIR` on this host). Build the pre-epic arm from a `git archive 4c44e317` tree with `cargo build --release --features rocm --bin mlxcel-bench-decode --bin mlxcel-bench-engine`.
2. Named tests: `data/.../harness/named_tests.sh OUT`.
3. Everything on the GPU, one lock hold: `GUARD=scripts/rocm_gpu_guard.sh data/.../harness/full_session.sh <repo> <pre-epic tree> OUT models/mlx <prompt file>` (bench rounds, bench_engine rounds, the `--ignored` u32 test, the decode profile, then the paged-floor parity session). The guard keeps the host-wide lock through `flock` and the per-run guards inside wait for an idle window without re-queueing (`ROCM_GPU_GUARD_LOCK_HELD`). `data/.../harness/parity_session.sh` alone runs the parity arms (`PAGED_FLOOR=default` keeps the shipped 4096-token floor); `extra_session.sh`, `moe_prefill_session.sh` and `gemma3_probe_session.sh` are the three follow-up runs.
4. Tables: `data/.../harness/summarize_rounds.py OUT/measure/rounds/rounds.csv --markdown`; `scripts/compare_logit_traces.py <dense>.decode_only.tsv <paged>.decode_only.tsv`.
