# PR #2792: Close the LOW residuals left by #2787, #2788 and #2789

**Date**: 2026-10-31 (written before merge; the PR is open as a draft)
**Status**: Implementation complete at 498bac80 on top of main 1a257bcd (three commits). The local gate in the PR body passed on GB10 (spark-102). At the time of writing a further local gate was running in the worktree and the hosted checks (`cargo-clippy`, `OpenXLA feature link`, `OpenXLA feature compile`, `CPU-only link`, `2-host logical`, the WebUI jobs) were pending
**Languages**: Rust, C, Markdown, YAML
**Risk Level**: Low to Medium (four of the five items change reporting, validation or test gating only; the IREE `local-task` device is now built by our own C code instead of IREE's factory, which is new native code on a production device path, and `PipelineUnavailable.stage` changes type)

## Executive Summary

PR #2792 (closes #2790) resolves five LOW residuals that the reviews and reports of #2787, #2788 and #2789 left behind, in one PR with one verification pass. The per-layer KV storage report now distinguishes layers that store the requested mode, layers that store FP16 regardless, and layers that hold no KV at all, and every model-owned family reports real storage instead of falling back to the configured table. A sliding window of 0 or above `i32::MAX` (and a Gemma 3 `sliding_window_pattern` of 0) fails the load instead of panicking later. The IREE `local-task` CPU device is built programmatically with 8 MiB worker stacks, so it starts in a process that has loaded CUDA. `XlaPreparedPrefillLogits` leaves production builds and becomes the single implementation behind the Qwen2.5-VL diagnostic engine. The pipeline availability gate opens only for stage-caused failures, and a forward timeout no longer blames the entry stage.

## 1. Problem Statement

All five items were re-checked on main 863ddb72 before the issue was filed. None is a security item; they were grouped because they share one gate (fmt, clippy, the `models::` and `distributed::` lib tests, one release `cuda,xla-iree` build).

### 1.1 The KV storage report still counted layers that do not store the requested mode

#2788 added `LanguageModel::kv_cache_storage_modes` so the server's `kv_cache_mode_applied_layers` stops counting FP16 sliding-window layers as quantized. Three cases still miscounted:

- **Llama 4**: `make_llama4_caches_with_modes` builds every layer with `(idx + 1) % 4 != 0` as an FP16 `ChunkedKVCache`, but `Llama4` had no storage override, so `--kv-cache-mode int8` reported every layer applied.
- **Layers with no KV**: linear-attention layers of Qwen3-Next, Qwen 3.5 and Bailing linear, LFM2 convolution layers and NemotronNAS `no_op` blocks reported the requested mode.
- **Model-owned families with no mode table** (DeepSeek V4, RecurrentGemma, Muse Glimmer, Mamba, Mamba2, RWKV7) returned `None`, and the server fell back to the configured table, claiming every layer applied.

The return type `Option<Vec<KVCacheMode>>` could not express "this layer holds nothing".

### 1.2 A zero sliding window or pattern panicked instead of failing the load

Gemma 3's `ModelArgs` accepted `sliding_window_pattern: 0`. Loading succeeded because `layer_rope_params` uses `is_multiple_of`, but `is_sliding_layer` and the cache builders compute `% self.sliding_window_pattern`, so since #2788 the scheduler panicked at build time inside `kv_cache_storage_modes`. Separately, `sliding_window: 0` built a `RotatingKVCache` with `max_size == 0`, and `speculative_ring_cursor` and the buffer-origin path call `rem_euclid(self.max_size)`, which divides by zero. Every family cast the `usize` window with `as i32`, so a value above `i32::MAX` wrapped silently.

### 1.3 The IREE `local-task` device could not start on GB10 with CUDA loaded

`xla_llama_create_impl` created every device with `iree_runtime_instance_try_create_default_device`. For `local-task`, IREE's driver factory builds executors from its global flag registry (`iree_task_executors_create_from_flags`), whose default worker stack is 128 KiB. glibc places static TLS inside each thread's stack, and the test binary plus its CUDA dependencies carry 235,072 bytes of `PT_TLS` (cuBLASLt 94,240, OpenBLAS 61,440, cuSOLVER 53,264, and others), so `pthread_create` returned EINVAL: `thread creation failed with 22; [runtime-init] failed to open IREE HAL device 'local-task'`. Only the `diagnostics` feature worked around it, by parsing `--task_worker_stack_size=8388608` into IREE's process-global flag registry. Production and the three `local-task` tests in `tests/xla_prepared_prefill.rs` could not run on spark-102.

### 1.4 `XlaPreparedPrefillLogits` duplicated diagnostic code and shipped in every IREE build

#2789 added `XlaPreparedPrefillLogits` for one integration test, but it was public under plain `iree`, so production bundles compiled it. Its body repeated `Qwen25VlLanguageDiagnosticEngine::load` and `capture` minus the slot reset and the M-RoPE check, which also meant repeated `prefill_logits` calls did not start from an empty cache.

### 1.5 The pipeline availability gate opened on the wrong failures and blamed the wrong stage

`RemotePipelineRuntime::exchange` opened the gate on any `Err` from `exchange_within`: a failed send, but also coordinator-local errors (serialization, `local_addr`, a closed receive channel). The open line always said "did not answer within {stage_timeout}", and `docs/distributed.md` said only a full stage timeout opens it. For a forward, the `RunEntry` exchange goes to `stage_peers[0]` and its reply comes from the last stage after every stage ran, so a timeout always named the entry stage even when a later stage was silent. `broadcast_best_effort` then gave the healthy entry stage a zero teardown wait and the real silent stage the full 2 s, until the first probe corrected `failed_peer`.

## 2. Technical Decisions

### 2.1 Change the trait's return type rather than add a second default method

`kv_cache_storage_modes` now returns `Option<Vec<Option<KVCacheMode>>>`, where an inner `None` is a layer with no KV cache. The default maps `kv_cache_layer_modes()` entries to `Some`. #2788's report noted that trait defaults hide forwarding gaps: a new method with a default compiles cleanly in a wrapper that forgets to forward it. Changing the signature of the existing method made the compiler list every forwarder (`LoadedModel`, the `&M` impl in `language_model_ref.rs`, `src/vision/mod.rs` and the VLM wrappers), and every one had to be touched. The five VLM wrappers around table-less text models (Granite 4 Vision, Inkling VL, Kimi K3 VL, Muse Glimmer VLM, Nemotron-H Nano Omni VL) did not forward the method before and now do.

The outer `None` keeps one meaning: "a dense model whose caches the scheduler builds from the table". `log_effective_kv_cache_application` falls back to the configured table only then.

### 2.2 One `LayerKv` vocabulary, fed by the builder's own predicate

`KvCacheLayerModes::storage_modes` takes a `layer_kind(layer_idx) -> LayerKv` closure with four kinds: `Full` (a `KVCache` with the injected mode), `Sliding(window)` (through `RotatingKVCache::stored_mode`), `Fp16Only` (a cache that ignores the mode) and `None`. `LayerKv::stored_mode` is the single mapping from kind to stored mode. Table-less families use `table_less_storage_modes(n_layers, layer_kind)`, which applies the same mapping with an FP16 request, so they report `Some(Fp16)` for attention layers and `None` for recurrent or SSM layers.

Each family passes the predicate its cache builder already uses: `Qwen3NextConfig::is_linear_layer`, `Lfm2Config::is_attention_layer`, RecurrentGemma's `block_types`, and a new `NASTransformerBlock::has_kv_cache` (only `AttentionSubblock::Attention` stores KV; a linear replacement keeps a slot it never writes). For Llama 4 the PR extracts `uses_chunked_cache(idx)` and calls it from the cache builder, both mask selections, the three `maybe_trim_front` loops and the storage override, so the report and the cache layout cannot drift.

### 2.3 Validate the window at load, return the checked `i32`, keep the constructor assert

`validate_sliding_window(family, window) -> Result<i32, String>` rejects 0 and values above `i32::MAX`. `cache_window(family, window, has_sliding_layer)` calls it only when the model has a sliding layer, so a config whose `layer_types` are all full attention still loads with any window (`a_zero_window_without_sliding_layers_still_loads`). Gemma 3, GPT-OSS, Laguna, Mellum, ExaOne 4, Ministral 3 and AFMoE store the validated `i32` and drop their `as i32` casts. Gemma 4 validates in both `Gemma4TextModel::from_weights` and `Gemma4StageModel::from_filtered_weights` (the stage path was added in 24674274 after review) but keeps its casts on the shared `TextConfig`, which has dozens of sites; after the check they are lossless. Gemma 3 also rejects `sliding_window_pattern: 0` with the same "sliding_window_pattern must be positive" wording `Gemma3Backbone::from_weights` already used.

`RotatingKVCache`'s constructor assert stays for direct callers, and no silent clamp was added: a bad config is an operator error that should surface at load with the family name, not a cache with a different window than the checkpoint expects.

### 2.4 Build the `local-task` device ourselves instead of setting IREE flags

Two ways to give workers a bigger stack existed: parse `--task_worker_stack_size` into IREE's flag registry (what `diagnostics` does), or construct the executor directly. The registry is process-global, so a production shim writing to it would change IREE behavior for every other user in the process, and the issue required leaving it untouched. `csrc/xla_local_task.c` mirrors IREE's `local_task/registration/driver_module.c`:

1. `iree_task_executor_options_initialize`, then `worker_stack_size = 8 MiB`.
2. A topology from `iree_task_topology_initialize_from_group_count(n)` when `MLXCEL_XLA_LOCAL_TASK_WORKERS` is set, else `iree_task_topology_initialize_from_physical_cores` on the current NUMA node, capped at 64 (IREE's `--task_topology_max_group_count` default).
3. `iree_hal_create_all_available_executable_loaders`, `iree_hal_allocator_create_heap`, `iree_hal_task_driver_create`, then `iree_hal_driver_create_default_device` with a proactor pool created the same way `iree_runtime_instance_try_create_default_device` creates it.

Every intermediate reference is released after the next object retains it, and each failure path releases what it created. Non-`local-task` URIs still go through `iree_runtime_instance_try_create_default_device`. Both device-creating FFI entry points (`xla_llama_create` and `xla_aux_create`) take the worker count, which Rust resolves with `local_task_workers_for(device)` before the call.

The worker count is validated in Rust: unset means `0` (physical cores); otherwise an integer in `1..=256` (`IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT`). The upper bound was added in 498bac80 after review, because IREE clamped larger values silently. Every IREE symbol used was checked to be exported by both the pinned CUDA tree and `iree-dist-3.12.0rc20260721-linux-x86_64`.

The `diagnostics` flag helper is unchanged; it is now redundant for `local-task` but still configures the diagnostic CLI targets.

### 2.5 Gate the prefill parity type behind a dedicated `test-support` feature

Gating `XlaPreparedPrefillLogits` behind `diagnostics` was rejected in the issue: it would move the parity test onto the diagnostics flag path, which sets the large stack through the flag registry and would have hidden item 1.3. Instead mlxcel-xla gets `test-support = ["iree"]`, the root `test-utils` forwards `mlxcel-xla?/test-support`, and `[[test]] xla_prepared_prefill` requires `xla-iree` and `test-utils`. The type is compiled under `any(test-support, diagnostics)`. `Qwen25VlLanguageDiagnosticEngine` now wraps it, passing its M-RoPE check as a closure to `prefill_logits_checked`, which resets slots, prepares the payload, runs the check, and prefills. `b_max` stays 4 because `IreeRaggedLlama::load` accepts only `RAGGED_B_VALUES = [4, 8]`; a comment says so.

### 2.6 A typed error decides whether the gate opens

`exchange_within` wraps the two stage-caused failures in `StageUnreachable { peer, cause }` with `StageUnreachableCause::{Timeout(Duration), SendFailed(String)}` (new module `stage_unreachable.rs`). A closed receive channel, serialization errors and `local_addr` stay plain `anyhow` errors. `exchange` searches the error chain with `downcast_ref` and opens the gate only when it finds a `StageUnreachable`. The `Timeout` display text is unchanged ("remote stage X timed out after Y"), because callers and tests match on "timed out".

Attribution: a `RunEntry` timeout calls `mark_unavailable(None, ..)`, because its reply arrives from the last stage and the timeout cannot say which stage stalled. A failed `RunEntry` send is still attributed to the entry stage, which is the one that could not be reached. While the gate is open with no named stage, `broadcast_best_effort` gives every stage the normal `teardown_ack_timeout`, and the next probe names the silent stage. `PipelineUnavailable.stage` becomes `Option<String>`; its `Display` keeps the old text for a named stage and says "an unidentified stage of the remote pipeline" otherwise.

The open line now states the cause: "did not answer within" for a timeout, "could not be reached ({reason})" for a failed send (reason passed through `loggable_peer_text`), and a run-level line when no stage is named. 498bac80 extended this to the probe path: `probe_failure_cause` reads the `StageUnreachable` cause from a failed probe's error chain, so a probe whose send failed no longer logs a timeout when it reopens the gate.

`StageAvailability`'s unit tests moved to `stage_availability_tests.rs`, and a per-thread `tracing` capture (`capture_logs`, `gate_open_lines`) in the new `runtime_log_capture_tests.rs` lets the runtime tests assert on the open line's text.

## 3. Change Summary

83 files, +2205 / -448, three commits.

| Commit | Content |
|---|---|
| `cfa55cac` | All five items |
| `24674274` | Gemma 4 pipeline-stage window validation; restores a displaced `gpt_oss.rs` doc comment |
| `498bac80` | `MLXCEL_XLA_LOCAL_TASK_WORKERS` upper bound of 256; probe reopen line reports the real cause |

| Area | Files | Change |
|---|---|---|
| Trait and forwarders | `mlxcel-core/src/generate.rs`, `language_model_ref.rs`, `src/loaded_model.rs`, 16 files under `src/vision/` | new return type; 5 VLM wrappers forward for the first time |
| Shared helpers | `src/models/model_owned.rs` | `LayerKv`, `table_less_storage_modes`, `validate_sliding_window`, `cache_window` |
| Storage overrides | `llama4.rs`, `qwen3_next.rs`, `qwen3_5.rs`, `qwen4_exp.rs`, `bailing_moe_linear.rs`, `lfm2.rs`, `nemotron_nas.rs`, `deepseek_v4.rs`, `recurrent_gemma.rs`, `muse_glimmer.rs`, `mamba.rs`, `mamba2.rs`, `rwkv7.rs`, `nemotron_h.rs`, `jamba.rs`, `falcon_h1.rs`, `plamo2.rs`, `granitemoehybrid.rs`, `kimi_linear.rs`, `kimi_k3.rs`, `inkling/runtime.rs` | report real per-layer storage |
| Window validation | `gemma3.rs`, `gemma4.rs`, `gpt_oss.rs`, `exaone4.rs`, `laguna.rs`, `step3p5.rs`, `ministral3.rs`, `mellum.rs`, `afmoe.rs`, `tensor_parallel/llama_runtime.rs` | validate at load, store `i32` |
| Server and example | `scheduler/paged_layout.rs`, `examples/logit_trace/kv_mode.rs` | `KvStorageSummary`, `kv_cache_mode_stateless_layers`, stateless count in the trace line |
| IREE shim | `csrc/xla_local_task.{c,h}`, `csrc/xla_iree.c`, `csrc/xla_aux.c`, `build.rs`, `src/local_task.rs`, `iree.rs`, `auxiliary.rs` | programmatic local-task device, worker-count parsing, test-support probe |
| Test support | `mlxcel-xla/Cargo.toml`, root `Cargo.toml`, `batch.rs`, `lib.rs`, `.github/workflows/ci.yml`, `tests/xla_prepared_prefill.rs` | `test-support` feature, shared prefill type, CI link step adds `test-utils` |
| Pipeline gate | `pipeline/stage_unreachable.rs` (new), `runtime.rs`, `stage_availability.rs`, `mod.rs`, test files | typed cause, optional attribution |
| Docs | `docs/distributed.md`, `docs/environment-variables.md` | gate rule; `MLXCEL_XLA_LOCAL_TASK_WORKERS` |

Behavior visible to operators:

- The startup line `resolved KV cache mode applied to model caches` gains `kv_cache_mode_stateless_layers`, and `kv_cache_mode_total_layers` now counts only layers that hold KV. For hybrid models (Qwen3-Next, LFM2, Jamba and similar) the total is lower than before.
- New variable `MLXCEL_XLA_LOCAL_TASK_WORKERS`.
- A config with a zero or oversized sliding window now fails to load with `{family}: sliding_window must be between 1 and 2147483647, got N`.
- Gate log lines distinguish timeouts from send failures and can say a stage is unidentified.

## 4. Trade-offs and Residuals

- **Predicates are still repeated per family.** Llama 4 now shares `uses_chunked_cache`, but most families pass their builder's predicate into the storage override by calling the same config method rather than one shared function per layout. A family that changes how it selects layers must update both call sites. Each family group has one storage test (Llama 4, Qwen3-Next, RecurrentGemma, Mamba, DeepSeek V4); the other overrides rely on their family suites passing.
- **DeepSeek V4 reports `Fp16Only` for every layer.** Its local cache is a `RotatingKVCache` plus compressed pooling state; the report treats the layer as one FP16 attention cache, which is what matters for the applied count but does not describe the pooled state.
- **The local-task probe test reports the configured stack, not a read-back value.** `xla_local_task_probe` returns the constant it passed to IREE. The evidence that the stack is in effect is behavioral: two workers start in a CUDA-linked binary, which fails with EINVAL at 128 KiB (the mutation check in section 5). The flag-registry assertion accepts both "registry compiled out" and "registry at its 128 KiB default".
- **Default and explicit worker caps differ.** Unset uses physical cores capped at 64; an explicit value may go to 256. This mirrors IREE's own defaults and is documented.
- **The local-task tests run only by hand.** Hosted CI links `xla_prepared_prefill` (`--no-run`); the `#[ignore]` tests need the pinned IREE runtime and ran on spark-102 only. There is no aarch64 IREE dist, so the dist path was checked by symbol export and header compilation, not by running on aarch64.
- **CUDA precision miss, unchanged.** With `MLXCEL_XLA_DEVICE=cuda`, `local_task_gemma3_vlm_prefill_matches_mlx_mask` misses its 1e-3 tolerance (up to 5.4e-3) under the default `f16` CUDA precision, identically on main; it passes with `MLXCEL_XLA_PRECISION=f32`.
- **Unattributed teardown is slower than attributed.** Between a `RunEntry` timeout and the next probe, the silent stage gets the full 2 s teardown wait because nobody knows which one it is. That matches main's cost for the real silent stage; the difference is that a healthy entry stage no longer gets a zero wait.
- **`PipelineUnavailable.stage` is a public field whose type changed** (`String` to `Option<String>`). Code inside the workspace was updated; an external crate matching on it would need `as_deref()`.
- **Docs nit**: the new `MLXCEL_XLA_LOCAL_TASK_WORKERS` row in `docs/environment-variables.md` has a capital "The" after a semicolon.
- **Out of scope** (from the issue): configurable `stage_timeout`, per-stage `RunEntry` attribution beyond "unidentified until probed", storing `Int8` in `RotatingKVCache`, quantized `ChunkedKVCache`, PaliGemma and Gemma3n OpenXLA masks.

## 5. Validation

### 5.1 Failing first

Each new test fails on main 1a257bcd, or, where it cannot compile there, under a mutation that restores main's behavior:

| Test | Failure on main's behavior |
|---|---|
| Existing three `local_task_*` tests, main binary | `thread creation failed with 22; [runtime-init] failed to open IREE HAL device 'local-task'` |
| `local_task_executor_uses_8_mib_stacks_without_iree_flags`, stack set back to 128 KiB | `xla_local_task_probe failed: ... thread creation failed with 22` |
| `llama4_chunked_layers_report_fp16_storage_under_int8` | `Some([Some(Int8) x8])` vs `Int8` only at layers 3 and 7 |
| `qwen3_next_linear_layers_report_no_kv_storage` | eight `Some(Int8)` vs `[None, None, None, Some(Int8), ...]` |
| `recurrent_gemma_reports_...`, `mamba_reports_every_layer_stateless`, `deepseek_v4_reports_fp16_storage_for_every_layer` (override returning `None`) | `left: None` |
| `gemma3_rejects_zero_sliding_window_pattern_at_load`, `sliding_window_zero_is_a_load_error` (validation off) | the loads succeed |
| `run_entry_timeout_does_not_name_the_entry_stage` (main attribution) | `failed_peer` is the entry stage |
| `send_failure_opens_the_gate_with_a_send_reason` | open line reads `did not answer within 1s` |

Further unit tests cover `validate_sliding_window` at 0, 1, `i32::MAX` and `i32::MAX + 1`; `KvStorageSummary` with stateless layers; `parse_local_task_workers` for unset, 0, negative, non-numeric, 257 and 1000; `StageAvailability` with an unattributed outage named by a failed probe; and `probe_failure_cause` as a pure error-chain check with no wall-clock bound.

### 5.2 Local gate (spark-102, GB10)

- `cargo fmt --all -- --check`, `make verify-env-var-docs`: clean.
- Clippy `-D warnings`: `--profile test-fast --features cuda` on `mlxcel`, `mlxcel-core`, `mlxcel-xla` lib and tests; `--release --features cuda,xla-iree,test-utils` on `mlxcel-xla` and `mlxcel` with `--test xla_prepared_prefill`: clean.
- Lib tests (`--test-threads=1`, under `gpu-lock`): `models::` + `vision::` 3429 passed; `distributed::` + `server::batch::scheduler` 1845; `mlxcel-core` `cache::` + `generate::` 624; `mlxcel-xla --lib` 370. After the review fixes: `distributed::pipeline::` 416, `mlxcel-xla` `local_task` 6.
- Integration (`cuda,test-utils`): `distributed_integration` 20, `pipeline_e2e` 30, `pipeline_remote_real_models` 10, `pipeline_server_remote_real_models` 7, `pipeline_ci_multi_stage_real_models` 4 (`--include-ignored --skip glm_moe_dsa`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`), `disaggregated_serving_e2e` 2, `disaggregated_router_e2e` 8: all pass.
- `cargo test --release --features cuda,xla-iree,test-utils --test xla_prepared_prefill -- --ignored`: 4/4 on the default `local-task` device. `MLXCEL_XLA_LOCAL_TASK_WORKERS=0` and `abc` fail the load naming the variable; `2` passes.
- Feature gating: a root `use mlxcel_xla::XlaPreparedPrefillLogits` fails `cargo check --features cuda,xla-iree` with E0432 and passes with `test-utils`; `--features cuda,xla-reference-diagnostics --lib --tests` checks clean (both in the `test-fast` profile).

### 5.3 Real-checkpoint smoke (`--kv-cache-mode int8`, release `mlxcel-server`)

| Model | Log |
|---|---|
| `llama-4-scout-17b-16e-instruct-4bit` | `applied_layers=12 total_layers=48 stateless_layers=0` (the 12 dense layers) |
| `gemma-3-1b-it-4bit` | `applied_layers=4 total_layers=26`, unchanged from #2788 |

No hybrid or table-less checkpoint was run through the server smoke; their storage reports are covered by the unit tests above.

## 6. Learning Points

- **Change a signature to find every forwarder.** A new trait method with a default compiles silently in wrappers that forget it. Changing the existing method's return type turned "find every wrapper" into a compiler error list, and it found five VLM wrappers that had never forwarded the method.
- **Give "nothing" its own value.** `Option<Vec<KVCacheMode>>` could not represent a layer with no cache, so reports either invented a mode or fell back to the configured table. An inner `Option` (and a `LayerKv::None` kind) made stateless layers a first-class answer, and the outer `None` kept its single meaning.
- **Static TLS lives on the thread stack.** On glibc, `pthread_create` with a stack smaller than the process's static TLS fails with EINVAL, not with an out-of-memory error. Loading large CUDA libraries (cuBLASLt alone carries 94 KB of TLS) can push a library's previously safe default stack size below the limit. Summing `PT_TLS` across the binary and `ldd` output is a quick way to confirm it.
- **Prefer constructing over configuring through globals.** IREE's flag registry is process-wide; setting a flag to fix one device would affect every other user. Building the executor through IREE's public API with explicit options fixes the device without global side effects, and the probe test checks the registry stayed at its default.
- **Classify errors by who caused them before acting on them.** Opening an availability gate is a statement about a remote stage, so only errors the stage caused should trigger it. A typed error found via `downcast_ref` on the `anyhow` chain keeps the existing `anyhow` plumbing and lets one call site make that decision.
- **Do not attribute what the protocol cannot attribute.** A forward's reply arrives only after every stage ran; its timeout says the run stalled, not where. Recording "unidentified" and letting the next per-stage probe name the stage is more accurate than naming the stage the request was sent to.
