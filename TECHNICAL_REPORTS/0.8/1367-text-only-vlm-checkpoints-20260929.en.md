# Technical Report: Route vision-stripped VLM checkpoints to a text-only path (#1367)

**Date**: 2026-09-29

**Status**: Implemented and validated locally; pending merge.

**Languages**: Rust

**Risk Level**: Low

## Executive Summary

Community checkpoints increasingly keep a VLM `model_type` but drop the vision tower. Before this change only checkpoints that omitted `vision_config` loaded, and only for four families. Detection now decides "text-only" from three signals (config, flag, weight names), and the four Qwen-VL families that cannot reuse a text `ModelType` load with no vision encoder and refuse media requests by name.

## 1. Problem Statement

`has_vision_config` was `config.get("vision_config").is_some()`, so `"vision_config": {}` and `null` went to the VLM loader and failed on the missing sub-config or the first `vision_tower.*` tensor. `language_model_only: true` was a deliberate named error even for exactly the stripped build the flag describes. Qwen2-VL, Qwen2.5-VL, Qwen3-VL and Qwen3-VL-MoE had no text-only route at all.

## 2. Change Summary

- `detection.rs`: `has_vision_config` now treats absent, `null`, `{}` and `language_model_only: true` as text-only (the flag wins over shipped weights, with a one-line stderr notice). `vlm_has_vision(config, path)` adds a weight scan over `VLM_VISION_WEIGHT_PREFIXES` (index first, else safetensors headers). `qwen3_5`, `qwen3_5_moe` and `gemma3` use it through a new `ModelDetectionProbes::vlm_has_vision_weights`; the catalog probe answers from a bounded index and keeps the VLM answer when it cannot read one.
- `vlm_qwen.rs`: the four Qwen-VL loaders call `vlm_has_vision`; when false they skip the vision-config requirement (`hidden_size` is stubbed, processor geometry comes from defaults or a present `vision_config`) and the encoder constructor.
- `vision/qwen*_vl*.rs`: `vision_encoder` is `Option`, plus `text_only_path`. `QwenVlRuntime::text_only_source` and `LoadedModel::has_vision_tower` expose it. `compute_qwen_vl_media_embeddings`, the single entry for CLI images, server images and video, returns `model <path> was loaded without a vision tower (text-only checkpoint)`.
- `qwen3_5.rs`: `language_model_only: true` is no longer an error; a non-boolean value still is.

## 3. Technical Decisions

- The weight scan can only demote a config-declared VLM to text. Where the weights cannot be read (config-only directory, catalog metadata without an index) the config answer stands, so nothing that worked before changes route.
- `llama4`, `mistral3` and `gemma3n` share `detect_text_or_vlm` and get only the config rules. A tower stored under an unlisted prefix would otherwise misroute to text and fail worse than today.
- The Qwen-VL `ModelType`s are unchanged (their decoders carry MRoPE), so the decision is carried to the loader by calling the same `vlm_has_vision` function rather than a new enum variant. This keeps the diff in `detection.rs` to helper code and existing arms.
- The four vision wrappers use `expect` on the missing encoder. It is unreachable because the runtime check runs first, and returning `Result` from the shared trait would have touched every Qwen-derived runtime.

## 4. Validation

- Unit: `vlm_text_only_detection_tests` (empty and null `vision_config`, the flag, index scan, every prefix, single-file header, unreadable weights, Qwen-VL types unchanged) and `vlm_qwen_text_only_tests` (synthetic `qwen3_vl` checkpoint loads with `vision_encoder: None`, rejects an image by name).
- Real checkpoint `leonsarmiento/ThinkingCap-Qwen3.6-27B-3bit-mlx`: unmodified, `vision_config: {}`, `language_model_only: true`, a full `vision_config`, and a full `vision_config` plus the flag all produced identical greedy output (64 tokens, `--temp 0 --show-reasoning`). `--image` on the loaded model returns `Images provided but model is not a vision-language model` instead of panicking.

## 5. Not done

`mlxcel list` capability reporting is static per `model_type`, so a text-only Qwen2/2.5/3-VL load is still listed as a VLM; the runtime refusal is the enforcement. Text-only routes for LLaVA, Idefics, Molmo and Pixtral remain out of scope.
