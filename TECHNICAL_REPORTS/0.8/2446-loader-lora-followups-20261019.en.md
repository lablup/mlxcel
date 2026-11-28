# PR #2446: Close loader and LoRA follow-ups from #2438 and #2439

**Date**: 2026-10-19
**Status**: Implemented and verified on CUDA (GB10) with real checkpoints; PR open. The items under "Not addressed" remain open

## Summary

Closes #2445. Five follow-ups left by #2438 and #2439, done in one PR as the issue asks, plus one blocker found while verifying item 1 on a real checkpoint: `estimate_kv_arch` parsed `config.json` with plain serde and so rejected `Infinity` / `NaN` values.

## Problem

- **`kv_arch` ignores Nemotron-H `layer_types`.** `classify` requires a `LAYER_COUNT_KEYS` entry, and `count_attention_layers` reads only `layers_block_type`, `block_types` and `hybrid_override_pattern`. A transformers 5 Nemotron-H config (only `layer_types`), which #2439 now loads, gets no estimate, or with `num_hidden_layers` present counts every layer as attention. The `contains("attention")` filter would also miscount `linear_attention`, which is an `M` layer.
- **`kv_arch` could not parse real Nemotron-H configs.** Configs such as `nvidia-nemotron-3-nano-30b-a3b-4bit` carry `time_step_limit: [0.0, Infinity]`. Plain serde rejected it, so the KV estimate was "unavailable" for them, pattern or not. On `main`, `mlxcel inspect -m` reports `KV cache: 0 bytes (unavailable ...)` for that checkpoint. This was found while verifying item 1.
- **Misleading stale-index warning on OptiQ checkpoints.** The OptiQ indexes map 441 / 809 tensors to `optiq/optiq_vision.safetensors`. `validate_index_shards` rejects the subdirectory name, so `collect_shard_paths` prints "references shards that don't match the on-disk files (likely a repackaged mlx-community quant)".
- **Text-only Muse Glimmer accepts image markers.** `reject_muse_glimmer_text_fallback` runs only for `LoadedModel::MuseGlimmerVLM`, so `LoadedModel::MuseGlimmer` feeds image and video marker ids to the text model.
- **Integer LoRA factors abort fusion.** `compute_lora_delta` passes `lora_a` / `lora_b` uncast to `mlxcel_core::matmul`, which is not fallible in the bridge, so an integer pair terminates the process, dense bases included. The runtime path already casts to f32 at claim.
- **Nemotron-H hides LoRA width errors as "unclaimed".** The four optional projections use `UnifiedLinear::from_weights(..).ok()`, so the width error from `claim_linear` is dropped and the load fails later in `ensure_runtime_terms_claimed`.

## Technical decisions

- **Nemotron-H `layer_types`** (`src/execution/kv_arch.rs`): for `model_type` `nemotron_h` without a `hybrid_override_pattern`, the `layers_block_type` / `layer_types` array is mapped through `NEMOTRON_H_BLOCK_NAMES`, `*` entries are counted, and the layer count is the array length, the way `NemotronHConfig::post_init` derives it. `NEMOTRON_H_BLOCK_NAMES` is made `pub(crate)` in `src/models/nemotron_h.rs` rather than copied. `linear_attention` is an `M` layer and is no longer counted by the `contains("attention")` filter.
- **Sanitized config read in `kv_arch`**: the file goes through `sanitize_config_json` first, as the model loader does. Without it, item 1 has no effect on a real checkpoint. This also makes the estimate available for any other config that carries `Infinity` / `NaN`, which used to report 0 KV bytes.
- **OptiQ sidecar**: `optiq_vision_sidecar_path` moves into `mlxcel_core::weights` with the same validation (relative, normal components only, `.safetensors`), and `loading::vlm` reuses it. `collect_shard_paths` drops exactly the declared sidecar from the index's shard list before `validate_index_shards`, and only when it names a subdirectory path (a top-level name is an ordinary shard). The config is read with the same `Infinity` / `NaN` replacement the model loader applies. An undeclared subdirectory entry, a malformed sidecar path or an unreadable config leaves the list unchanged, so those still fail validation and take the glob fallback as before.
- **Muse Glimmer text-only markers**: the check is split into a helper over token ids. `MuseGlimmerTextWrapper` keeps the checkpoint's `image_token_id` / `video_token_id` (two `i32` fields, set by `load_muse_glimmer_text`), and `reject_muse_glimmer_text_only_markers` runs at both sites for `LoadedModel::MuseGlimmer`. The error says the checkpoint is loaded text-only. A later commit changes the video marker check to use `contains`.
- **Integer LoRA factors**: `compute_lora_delta` casts both factors to f32 before the matmul, as `runtime_lora` (at claim) and `src/lora/embedding.rs` do. The delta was already f32 because the scale is an f32 array, so for a floating-point pair only the matmul's input precision changes; for an f32 adapter (what `mlx_lm.lora` writes) nothing changes.
- **Nemotron-H optional projections**: the shared-expert `up_proj` / `down_proj` and the two latent projections return `None` only when `{prefix}.weight` is absent; otherwise the constructor error is propagated with `?`, so the `claim_linear` width error from #2438 is the load error instead of a later "unclaimed" failure.

## Changes

- `src/execution/kv_arch.rs`: `layer_types` mapping, sanitized config read, new tests.
- `src/models/nemotron_h.rs`: `NEMOTRON_H_BLOCK_NAMES` made `pub(crate)`; error propagation on the four optional projections. `src/models/nemotron_h_tests.rs`: new test.
- `src/lib/mlxcel-core/src/weights.rs`, `src/loading/vlm.rs`, `src/loading/vlm_lfm2_vl_tests.rs`: `optiq_vision_sidecar_path` moved and reused; sidecar dropped in `collect_shard_paths`.
- `src/multimodal/muse_glimmer_runtime.rs`, `src/models/muse_glimmer.rs`, `src/loading/vlm_muse_glimmer.rs`, `src/server/model_worker.rs`, `src/multimodal/vlm_runtime.rs`, `src/loading/vlm_muse_glimmer_tests.rs`: marker helper, wrapper fields, both call sites, test.
- `src/lora/loader.rs`, `src/lora/loader_tests.rs`: f32 cast in `compute_lora_delta`, test.

## Verification

GB10, `--features cuda`, `test-fast` profile, every GPU run under `gpu-lock`, `MLXCEL_SDPA_DETERMINISTIC=1`. "Main" is a binary built from 74f9ae1d in the same worktree; "fixed" is this branch.

- `cargo test --profile test-fast --features cuda --lib -- kv_arch lora:: muse_glimmer nemotron_h loading::vlm multimodal:: model_worker memory_estimate --test-threads=1`: 877 passed, 0 failed.
- `cargo test --profile test-fast -p mlxcel-core --features cuda --lib -- weights:: runtime_lora --test-threads=1`: 38 passed.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`, `cargo fmt --all -- --check`, and the five contract checks (`check_llama_compat_manifest`, `check_kernel_dtype_keys`, `check_kernel_port_dispatch`, `check_crate_versions`, `insert_apache_header --check`): clean.

Each new test fails on the reverted fix (each fix reverted in place, test binary rebuilt, then restored):

| Test | Reverted result |
|------|-----------------|
| `execution::kv_arch::tests::nemotron_h_layer_types_match_the_equivalent_pattern` | panics: no estimate for the `layer_types` config |
| `execution::kv_arch::tests::a_config_file_with_infinity_still_estimates` | panics: no estimate (serde rejects `Infinity`) |
| `weights::tests::test_collect_shard_paths_skips_only_the_declared_optiq_sidecar` (mlxcel-core) | `left: ["model.safetensors", "stray.safetensors"]`, glob fallback taken |
| `loading::vlm::muse_glimmer::tests::text_only_muse_glimmer_refuses_image_and_video_markers` | panics: text-only model accepted marker (call sites reverted) |
| `lora::loader::tests::an_integer_lora_pair_fuses_like_the_same_values_in_f32` | process abort, exit 134: `[matmul] Only inexact types are supported but int32 and int32 were provided` |
| `models::nemotron_h::tests::a_wrong_width_lora_term_on_an_optional_projection_fails_the_load` | panics: load succeeded without the shared experts |

The Nemotron-H test is `cfg(not(all(target_os = "macos", feature = "metal")))`, since Metal refuses runtime LoRA for this model before any layer is built.

Real checkpoints (`~/models_archive/mlx/...`):

| Item | Checkpoint and command | Main | Fixed |
|------|------------------------|------|-------|
| 1 | `nvidia-nemotron-3-nano-30b-a3b-4bit`, `mlxcel inspect -m` | `KV cache: 0 bytes (unavailable ...)` | `48.0 MiB (hybrid: 6 attention layer(s) hold KV)` |
| 1 | same checkpoint with a transformers 5 style config (`hybrid_override_pattern` rewritten as `layer_types`, `num_hidden_layers` removed), `mlxcel inspect -m` | `0 bytes (unavailable ...)` | `48.0 MiB (hybrid: 6 attention layer(s) hold KV)`, identical to the pattern config |
| 2 | `lfm2.5-vl-3b-optiq-4bit`, `mlxcel generate`, text prompt and `--image tests/fixtures/test_image_shapes.png` | stale-index warning printed | no warning; generated text identical to main for both prompts |
| 2 | `muse-glimmer-30b-optiq-4bit`, `mlxcel generate --image ... --show-reasoning` | stale-index warning printed | no warning; generated text identical to main |
| 3 | text-only view of `muse-glimmer-30b-optiq-4bit` (top-level shards, index without the sidecar entries, no `optiq_vision`; detected as `MuseGlimmer`), prompt `"<\|image\|> What is in the picture?"` | accepted, generates from the marker ids | `Error: this Muse Glimmer checkpoint is loaded text-only (no vision tower), but the prompt contains image or video marker tokens ...`, exit 1 |
| 3 | same view, plain prompt | generates | identical text |
| 4 | `llama-3.2-1b-instruct-bf16` (dense), `mlxcel generate --adapter`, q_proj adapter of layers 14 and 15 trained with `mlx_lm.lora`, factors rounded to `round(x * 100)` and stored as int32, scale divided by 1e4 | `terminate called after throwing an instance of 'std::invalid_argument' ... [matmul] Only inexact types are supported`, exit 134 | loads and generates; text identical to the same integer values stored as f32 |
| 4 | same, f32 twin adapter | generates | identical text to main |
| 5 | `nvidia-nemotron-3-nano-30b-a3b-4bit` (`n_shared_experts: 1`, so every MoE layer builds its shared experts through the new path), `mlxcel generate --show-reasoning`, 24 tokens | generates | identical text |

The CLI's media preparation goes through `prepare_request_vlm_embeddings`, the server worker site, so the item 3 run exercises that call site. The unit test covers the `prepare_vlm_embeddings_with_cache` site and the helper the worker calls; no unit test drives `prepare_request_vlm_embeddings` itself, so the worker arm is covered only by the real run.

## Not addressed

Review found the same declared-sidecar entry treated as a bad shard in two other places, outside this issue's scope:

- `src/downloader/completeness.rs`: `missing_indexed_shards` counts `optiq/optiq_vision.safetensors` as missing, so a store-resolved OptiQ snapshot classifies as incomplete.
- `src/server/webui/catalog_fs.rs`: reports it as an unsafe shard path.
- `weight_footprint_bytes` also leaves the sidecar's bytes out of the memory preflight.

Both of the first two can reuse `mlxcel_core::weights::optiq_vision_sidecar_path`.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 14 |
| Lines added | 622 |
| Lines deleted | 87 |

| Commit | Message |
|--------|---------|
| `c4e6442d` | fix: close loader and LoRA follow-ups from #2438 and #2439 (#2445) |
| `4df6e022` | fix(kv_arch): sanitize config.json before the KV estimate parses it (#2445) |
| `21a7207d` | fix(muse_glimmer): use contains for the video marker check (#2445) |
| `e630c5ca` | fix(weights): sanitize the OptiQ config read and keep top-level shards (#2445) |
