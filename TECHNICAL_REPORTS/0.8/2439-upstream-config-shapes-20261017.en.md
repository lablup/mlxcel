# PR #2439: Accept upstream config shapes, aliases and key layouts

**Date**: 2026-10-17
**Status**: Implemented and verified on CUDA (GB10) with real checkpoints; two follow-ups not done (see "Not addressed")
**Risk**: Medium (touches config parsing for Qwen3, GraniteMoeHybrid, Nemotron-H, Muse Glimmer, LFM2-VL, Moondream3 and the EOS id reader; Qwen3 and GraniteMoeHybrid previously decoded with a wrong theta without any error)

## Summary

Closes #2384. Upstream mlx-lm and mlx-vlm accept several config shapes, `model_type` spellings and checkpoint key layouts that mlxcel either rejected at load or accepted with a wrong value. The PR fixes all eight items from the issue in one PR: nested `rope_parameters`, transformers-saved Nemotron-H configs, missing `model_type` aliases, DeepSeek-V4 quantized embedding planes, flat text-only Muse Glimmer configs, flattened OptiQ key layouts, Moondream3 mlx-vlm quantized conversions, and the union of EOS id sources.

## Problem

The issue lists eight gaps. Seven fail loudly at load. Item 1 does not.

- **Item 1, nested `rope_parameters`:** transformers 5 nests `rope_theta` and `rope_scaling` under `rope_parameters` and drops the flat keys. `qwen3::ModelArgs` and `granitemoehybrid::ModelArgs` then fell back to the serde default theta of 10000. The model decodes fluently at short lengths and drifts at long ones.
- **Item 2, Nemotron-H:** `num_hidden_layers` was required, only `layers_block_type` was read (not the `layer_types` alias), the block name map rejected `full_attention`, `linear_attention` and `conv`, and the `time_step_limit` visitor failed on a transformers `{"__float__": "..."}` tag. A non-number second element became `+inf`, which also turned `-Infinity` and `NaN` into `+inf`.
- **Item 3, aliases:** `kimi_k2`, `joyai_llm_flash`, `minimax_m2`, `xverse`, `qwen3_5_text` and `qwen3_5_moe_text` were not in `detect_model_type`, so they failed with "Unsupported model type".
- **Item 4, DeepSeek-V4:** the legacy namespace rename covered `embed.weight` and `head.weight` but not their `.scales` and `.biases`, so a legacy-namespace MLX quant failed the required-key check.
- **Item 5, Muse Glimmer:** `MuseGlimmerConfig` required `text_config`, and detection always routed `muse_glimmer` to the VLM. An mlx-lm text-only conversion (flat config, `model.*` and `lm_head.*` keys) could not load. `MuseRopeParameters` parsed only `rope_theta`, so a `linear`, `yarn` or `llama3` `rope_type` was ignored.
- **Item 6, OptiQ layouts:** `lfm2_text_weights` kept only `language_model.*` keys, so a flattened LFM2-VL lost its text tower. `normalize_muse_weight_key` passed bare `model.layers.*` through instead of mapping it under `model.language_model`.
- **Item 7, Moondream3:** `rewrite_moondream3_weight_key` handled only the original layout (`text.blocks.*`, `text.wte`, `vision.blocks.*`), not the mlx-vlm quantized layout (`text.model.blocks.*`, `vision.encoder.*`). The vision tower hardcoded group 64 and 4-bit.
- **Item 8, EOS ids:** `read_eos_token_ids` returned the first non-empty of `generation_config.json`, `tokenizer_config.json` and `config.json`, so a checkpoint whose generation config lists one stop id never stopped on the others.

## Technical decisions

### Lift rope_parameters inside Deserialize

`qwen3::ModelArgs` and `granitemoehybrid::ModelArgs` now derive with `#[serde(remote = "Self")]` and implement `Deserialize` by hand through `rope_utils::deserialize_lifting_rope_parameters`. Every `from_value` and `from_str` site gets the lift. The issue suggested a private wire struct; `remote = "Self"` has the same effect without copying the field list. The issue rejected a lifter call per site, because the next new parse site would miss it.

The lifter moved from the Mage-VL config into `rope_utils`, and Mage-VL re-exports it. A flat `rope_theta` wins over the nested one.

Coverage audit: the families mlx-lm 0.32.2 lifts `rope_parameters` for (exaone_moe, gemma4_text, glm_moe_dsa, laguna, lfm2/lfm2_moe, mellum, ministral3, mistral4, muse_glimmer, qwen3_5) already read the nested block in mlxcel. Qwen3 and GraniteMoeHybrid were the only gaps.

### Text-only Muse Glimmer

A flat config becomes its own `text_config`. A config without vision (no vision config or no vision weights) loads as the new text-only `ModelType::MuseGlimmer`. `rope_parameters` schemes resolve through `RopeScalingKind`. The text-only model shares the family guards (no adapters, fp16 KV only). After review, startup also refuses any drafter for it, because the DFlash round loop has no target arm for the text-only wrapper.

### OptiQ vision sidecar

Both OptiQ checkpoints keep their vision tower in `optiq/optiq_vision.safetensors`, named by `config.json` `optiq_vision.sidecar`. The shard loader rejects subdirectory shards, so VLM weight loading now merges the sidecar itself (relative `.safetensors` paths only). The core shard validator is unchanged. After review, a missing sidecar is a warning instead of a load error: text-only loads never read it, and a VLM load still fails on the first missing vision tensor.

## Changes

- **Nemotron-H** (`src/models/nemotron_h.rs`): `num_hidden_layers` is optional and `layer_types` aliases `layers_block_type`. The transformers 5.13 names map to pattern codes, and an unknown name lists the accepted ones. `time_step_limit` decodes the `__float__` tag and rejects other non-numbers. The loader's `sanitize_config_json` rewrites the tag's `"Infinity"` to `"1e38"`, so a numeric tag string is also accepted.
- **Aliases** (`src/models/detection.rs`, `registry.rs`): `kimi_k2`, `joyai_llm_flash`, `minimax_m2`, `xverse` (shares the `llama` arm), `qwen3_5_text` and `qwen3_5_moe_text` were added to detection and to the registry keys. The two DeepSeek-V3 aliases also joined `MLA_LATENT_CACHE_FAMILIES`, so a quantized KV mode still resolves to fp16 on them.
- **DeepSeek-V4** (`src/models/deepseek_v4_sanitize.rs`): embed and head are renamed by stem.
- **LFM2-VL and Muse Glimmer key maps**: the flattened OptiQ text roots now map under `model.language_model` (Muse) and are accepted as text keys (LFM2-VL).
- **Moondream3** (`src/loading/vlm_special.rs`, `src/vision/encoders/moondream3.rs`): both key layouts are accepted, and quantization group size and bits come from the config.
- **EOS ids** (`src/loading/mod.rs`): `read_eos_token_ids` returns the de-duplicated union of its sources.
- **Docs**: `docs/supported-models.md` records the new aliases.

## Behavior changes

- `read_eos_token_ids` now returns the union of its sources. The existing tests that assumed the first source wins, including the Kimi K3 tokenizer pin, were updated.
- Configs that previously loaded with the serde default theta 10000 on Qwen3 and GraniteMoeHybrid now read the nested theta.
- `ModelType::MuseGlimmer` is a new text-only route for Muse configs without vision.

## Validation

GB10, `--profile test-fast --features cuda`, `MLXCEL_SDPA_DETERMINISTIC=1`, greedy. Checkpoints were read from `~/models_archive/mlx/<name>`, where the host moved `~/models/mlx` during the run; the default-path Muse run used the same pinned `muse-glimmer-30b-4bit` via `/mnt/models-10/mlx`. Each transformers-5 copy is the original checkpoint with only `config.json` rewritten. The long prompt is CONTRIBUTING.md (22 KB) followed by a question, run with `generate -n 64 --temp 0`.

| Checkpoint | Result |
|------------|--------|
| `qwen3-0.6b-4bit` (transformers-5 copy) | main decodes the copy into garbage. The branch decodes it byte-identically to the original, and the original is identical on main and branch. |
| `granite-4.0-3b-vision-4bit` (granitemoehybrid text, theta 1e7, RoPE) | Same result. The `granite-4.0-h-*` checkpoints are NoPE, so they cannot show the bug. |
| `nvidia-nemotron-3-nano-30b-a3b-4bit` with `layer_types`, `__float__`, no `num_hidden_layers` | main fails with "missing field num_hidden_layers". The branch output is identical to the original. |
| `mlx-community/XVERSE-7B-Chat-4bit` | main fails with "Unsupported model type: xverse". The branch gives fluent output ("Seoul is the capital and most populous city of South Korea..."). |
| `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` | main fails with a missing `model.embed_tokens`. The branch answers "Paris" and lists the red square, blue circle and green triangle. |
| `mlx-community/Muse-Glimmer-30B-OptiQ-4bit` | main fails with an unknown weight root. The branch answers a text prompt ("The capital of France is Paris.") and an image prompt. A flat text-only copy of it decodes tokens identical to the VLM text path. |
| `mlx-community/Muse-Glimmer-30B-4bit` (default path) | 64 greedy tokens of text and of an image prompt are identical between main and branch. |
| `beshkenadze/moondream3-preview-mlx-4bit` | main fails with a missing `text.blocks.0.mlp.fc1.weight`. The branch captions the shapes image correctly. |

All 30 new or changed unit tests pass. With the fixes neutralized, 26 of them fail. Four pass unchanged, as expected: two flat-wins-over-nested pins, the default-RoPE pin, and the reject-non-number case, which on true `main` would parse to +inf. Kimi-K2, JoyAI and MiniMax-M2 do not fit this host, so they have detection tests only. clippy (lib+tests, `-D warnings`), fmt and the five contract checks pass.

## Review changes

A second commit (`0c846ab4`) refuses a drafter for text-only Muse, downgrades a missing OptiQ sidecar to a warning, and gives the Muse startup-guard fixture a `vision_config` so it stays on the VLM route. `text_only_muse_glimmer_refuses_a_drafter` covers the new guard.

## Not addressed

- Other flat-`rope_theta` families that mlx-lm does not lift either, for example `qwen3_moe`.
- `kv_arch` memory estimation for a Nemotron-H config that carries only `layer_types`.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 40 |
| Lines added | +1568 |
| Lines deleted | -140 |
| New or changed unit tests | 30 |

| Hash | Message |
|------|---------|
| `6ece3f6f` | fix(loading): accept upstream config shapes, aliases and key layouts |
| `0c846ab4` | fix(server): refuse a drafter for text-only Muse Glimmer |
