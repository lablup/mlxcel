# PR #2448: Lift transformers 5 rope_parameters for every family

**Date**: 2026-10-19
**Status**: Implemented and verified on CUDA (GB10) with real checkpoints; PR open. The items under "Not addressed" remain open

## Summary

Closes #2443. transformers 5 saves `rope_theta`, `rope_scaling` and `partial_rotary_factor` only under a nested `rope_parameters` block. Only Qwen3 and GraniteMoeHybrid lifted it (#2439). Every other struct that reads the flat keys fell back to its serde default base without an error. This PR applies the lift at the config type, through one macro line per struct, for 86 structs plus the call sites that parse outside serde.

## Problem

- Config structs that read a flat `rope_theta` with a serde default (10000 for llama3, qwen3_moe, gemma2, phi3, olmo2) load a transformers 5 checkpoint without error, decode fluently on short prompts, and drift on long ones. A Llama 3 config also loses its `rope_type: "llama3"` scaling.
- On `main`, a transformers-5 copy of Llama 3.2 1B or Qwen3-30B-A3B produces garbage on a 20 KB prompt. The Llama copy degenerates into `and, and, ...` and the Qwen3 copy into `. The. The. ...`.
- mlx-lm 0.32.2 lifts only exaone_moe, gemma4, glm_moe_dsa, laguna, lfm2, mellum, ministral3, mistral4, muse_glimmer and qwen3_5, which mlxcel already handles, so no other family gets a fix from parity.
- EXAONE-MoE and Mistral 4 let the nested base win, which was inconsistent with the families that let a flat key win.
- `src/loading/vlm_qwen.rs` carried two ad hoc call-site lifts (GLM-OCR and GLM-4V MoE).

## Technical decisions

- **The type is the choke point.** The PR author looked for a shared parse helper that every site already calls and found none: about 125 family `load` functions read `config.json` directly, and VLMs nest the block under `text_config`. The issue rejects a top-level lift in `sanitize_config_json` for both reasons (it is a string replace that most family `load` functions skip, and it would miss `text_config`).
- **Shared lifter in `mlxcel-core`.** `lift_rope_parameters` / `deserialize_lifting_rope_parameters` move to `mlxcel_core::rope_parameters` (re-exported from `models::rope_utils`), so the DFlash drafter config can use them. An `impl_rope_lifting_deserialize!(Type)` macro gives each uncovered struct `#[serde(remote = "Self")]` plus one line. This covers 86 structs, including the VLM text configs and the three Qwen3-Omni speech configs. Pipeline and tensor-parallel sites parse these same types, so they are covered. No struct gains a field.
- **Lifter rules.**
  - A flat non-null key wins.
  - It lifts `partial_rotary_factor`.
  - It reads the per-layer-type forms, meaning a block whose values are all objects. For Gemma 3 / Gemma 3n / OLMo 3, `full_attention` gives the base and scaling and `sliding_attention` gives `rope_local_base_freq`. For DeepSeek-V4, `main` gives the base and `compress` gives `compress_rope_theta` and the YaRN `rope_scaling`.
  - It builds `rope_scaling` from a non-default scheme (Llama 3 `llama3` included) or from MRoPE extras like `mrope_section`.
  - The scheme goes under `type` only (it reads `type` first, then `rope_type`), because the deepseek_v32 / deepseek_v4 / nemotron `alias = "type", alias = "rope_type"` fields reject a block that spells both.
- **Call sites outside serde use the shared lifter.** `DFlashConfig::from_json`, the Idefics2 Mistral defaults (which used to shadow the nested base) and both `vlm_qwen.rs` ad hoc lifts. `JinaVlmTextConfig::from_json` falls back to the nested base, and the Phi-4 text-field copy now carries `rope_parameters`.
- **Flat wins everywhere.** EXAONE-MoE and Mistral 4 now let a flat key win, as the issue specifies. transformers 5 itself gives the nested value priority, but the two only disagree when a config carries both with different values, and none of the reference checkpoints does.
- **`exaone4`.** It was the one reader that checked only `rope_type`, and it now accepts `type` too. Its `rope_scaling` is now `#[serde(default)]`, because a default-scheme transformers 5 save has none.

## Changes

- `src/lib/mlxcel-core/src/rope_parameters.rs` (new, 187 lines), `rope_parameters_tests.rs` (new, lifter unit tests), `lib.rs`: the shared lifter and macro.
- `src/models/rope_utils.rs`: lifter code removed and re-exported from `mlxcel-core`.
- About 85 config files under `src/models/` and `src/audio/qwen3_omni_moe/`: `#[serde(remote = "Self")]` and one `impl_rope_lifting_deserialize!` line each (typically 3 lines per file).
- `src/models/mistral4.rs`, `src/models/exaone_moe.rs`, `src/models/exaone4.rs`, `src/models/jina_vlm.rs`: flat-wins, `type` handling and fallback changes described above.
- `src/loading/vlm_qwen.rs`, `src/lib/mlxcel-core/src/drafter/dflash/config.rs`: call sites moved to the shared lifter.
- `src/models/rope_parameters_lift_tests.rs` (new, 417 lines): table-driven tests.

## Verification

- `rope_parameters_lift_tests::every_flat_rope_theta_config_lifts_rope_parameters_and_flat_wins` is table-driven over all 89 parse paths (the 86 lifted structs, Mistral 4, and the DFlash and Jina `from_json` paths). It checks that nested-only gives 123456 and flat 500000 plus nested gives 500000. Required non-RoPE fields are filled from serde's own errors, never `rope_theta` / `rope_scaling`.
- More tests: Llama 3 nested `llama3` resolves `RopeScalingKind::Llama3`, Gemma 3 per-layer-type, Phi `partial_rotary_factor`, and DeepSeek-V4 `compress` YaRN (added in review; not revert-run). The lifter unit tests are in `mlxcel-core`.
- With the fix stashed and only the tests kept, the first four tests all fail, and every one of the 89 rows reports a failure.
- Real checkpoints: transformers-5 copies of `~/models_archive/mlx/llama-3.2-1b-instruct-4bit` and `~/models_archive/mlx/qwen3-30b-a3b-4bit`, with only `config.json` rewritten (flat keys moved into a `rope_parameters` block identical to the one transformers 5.20.1 emits for that config) and the weights symlinked. Command: `MLXCEL_SDPA_DETERMINISTIC=1 gpu-lock run -- mlxcel generate -m <dir> -p "<first 20000 bytes of CONTRIBUTING.md + a question>" -n 64 --temp 0 --show-reasoning`. On the branch, both copies match their originals byte for byte, and the originals match `main`. On `main` (self-parity checked), the Llama copy degenerates into `and, and, ...` and the Qwen3 copy into `. The. The. ...`.
- fmt, clippy (`--lib --tests -D warnings` for mlxcel and mlxcel-core), the affected-family test filters (647 tests) and the repo contract checks pass locally.

## Changes during review

The review found no CRITICAL or HIGH issues. Two follow-ups were applied, in commit `1684f151`:

- The DeepSeek-V4 `main` / `compress` layout, which the first version turned into an all-`None` `rope_scaling`.
- A `"type": null` key that hid the `rope_type` scheme.

## Not addressed

- Vision-encoder configs (Pixtral, Llama 4 vision, etc.) and `mlxcel-xla`, which the issue does not list.
- DBRX is lifted at `attn_config`, where transformers 5 still keeps `rope_theta`, but its top-level `rope_parameters` is not copied there.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 96 |
| Lines added | 1134 |
| Lines deleted | 116 |

| Commit | Message |
|--------|---------|
| `0681d42b` | fix(models): lift transformers 5 rope_parameters for every family |
| `1684f151` | fix(models): lift DeepSeek-V4 main/compress rope_parameters |
