// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Qwen-VL family loader tests.
//!
//! The synthetic half asserts the key remap that lets `mlxcel generate` and
//! `mlxcel-server` read a raw `transformers` export of Qwen2.5-VL. The real
//! half is one `#[ignore]`d differential gate that runs the generation path
//! on both a raw export and its mlx conversion and requires the same tokens
//! out of each; it soft-skips when either checkpoint is absent.

use super::{remap_qwen2_5_vl_native_keys, rewrite_qwen2_5_vl_native_key};

#[test]
fn native_key_remap_renames_the_tower_and_unwraps_the_language_model() {
    // Older exports: `visual.*` beside a bare `model.*` decoder. Newer ones:
    // `model.visual.*` beside `model.language_model.*`. Both reach this path
    // because `strip_language_model_prefix` only handles a leading
    // `language_model.`.
    assert_eq!(
        rewrite_qwen2_5_vl_native_key("visual.blocks.0.attn.qkv.weight"),
        "vision_tower.blocks.0.attn.qkv.weight"
    );
    assert_eq!(
        rewrite_qwen2_5_vl_native_key("model.visual.patch_embed.proj.weight"),
        "vision_tower.patch_embed.proj.weight"
    );
    assert_eq!(
        rewrite_qwen2_5_vl_native_key("model.language_model.layers.0.mlp.up_proj.weight"),
        "model.layers.0.mlp.up_proj.weight"
    );
    assert_eq!(
        rewrite_qwen2_5_vl_native_key("lm_head.weight"),
        "lm_head.weight"
    );
}

#[test]
fn native_key_remap_maps_a_whole_export_and_leaves_a_conversion_alone() {
    let dummy = || mlxcel_core::from_slice_f32(&[0.0], &[1]);
    let mut raw = mlxcel_core::weights::WeightMap::new();
    for key in [
        "visual.blocks.0.attn.qkv.weight",
        "model.visual.patch_embed.proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "lm_head.weight",
    ] {
        raw.insert(key.to_string(), dummy());
    }

    let remapped = remap_qwen2_5_vl_native_keys(raw);
    let mut keys: Vec<&str> = remapped.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "lm_head.weight",
            "model.layers.0.mlp.up_proj.weight",
            "vision_tower.blocks.0.attn.qkv.weight",
            "vision_tower.patch_embed.proj.weight",
        ]
    );

    // An mlx-community conversion already carries the target names, the
    // merger included, so every rule misses and the map comes back unchanged.
    let mut converted = mlxcel_core::weights::WeightMap::new();
    for key in [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.merger.mlp.0.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.embed_tokens.weight",
    ] {
        converted.insert(key.to_string(), dummy());
    }
    let before: Vec<String> = {
        let mut keys: Vec<String> = converted.keys().cloned().collect();
        keys.sort_unstable();
        keys
    };
    let after = {
        let mut keys: Vec<String> = remap_qwen2_5_vl_native_keys(converted)
            .keys()
            .cloned()
            .collect();
        keys.sort_unstable();
        keys
    };
    assert_eq!(before, after);
}

// Real-checkpoint gate. `#[ignore]`d because it loads two 3B checkpoints, and
// it soft-skips (rather than failing) when either directory is absent.

/// Repo ids the gate looks for through the usual store lookup.
const RAW_EXPORT: &str = "Qwen/Qwen2.5-VL-3B-Instruct";
const MLX_CONVERSION: &str = "mlx-community/Qwen2.5-VL-3B-Instruct-bf16";

/// Environment overrides for a checkout that keeps these two snapshots
/// somewhere the store lookup does not reach, for example a shared
/// `models/mlx/...` tree. Each names a model directory.
const RAW_EXPORT_DIR_ENV: &str = "MLXCEL_TEST_QWEN25VL_RAW_DIR";
const MLX_CONVERSION_DIR_ENV: &str = "MLXCEL_TEST_QWEN25VL_MLX_DIR";

/// The prompt both checkpoints get, already in Qwen2.5-VL chat-template form
/// so the gate does not depend on the tokenizer's template rendering.
const IMAGE_PROMPT: &str = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n<|im_start|>assistant\n";

/// A deterministic image. The gate asserts two checkpoints agree on the same
/// input, so the picture only has to be identical between the two runs.
fn gate_image() -> image::DynamicImage {
    let mut buffer = image::RgbImage::new(112, 112);
    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        *pixel = image::Rgb([(x * 2) as u8, (y * 2) as u8, ((x + y) % 256) as u8]);
    }
    image::DynamicImage::ImageRgb8(buffer)
}

fn gate_checkpoint(repo_id: &str, env_key: &str) -> Option<std::path::PathBuf> {
    if let Ok(dir) = std::env::var(env_key) {
        let dir = std::path::PathBuf::from(dir);
        if dir.join("config.json").is_file() {
            return Some(dir);
        }
        eprintln!(
            "skipping real-checkpoint gate: {env_key}={} has no config.json",
            dir.display()
        );
        return None;
    }
    crate::models::embedding_test_support::local_checkpoint(repo_id)
}

/// Greedy tokens for one checkpoint, through the same calls `mlxcel generate
/// --image` makes: `load_model`, the shared VLM embedding preparation, and a
/// backend session seeded with those embeddings.
fn greedy_image_tokens(model_dir: &std::path::Path, max_tokens: usize) -> Vec<i32> {
    let (model, tokenizer) =
        crate::loading::load_model(model_dir).expect("the Qwen2.5-VL checkpoint loads");
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode_with_special(IMAGE_PROMPT, false, true)
        .expect("the chat-template prompt tokenizes")
        .iter()
        .map(|&id| id as i32)
        .collect();

    let prepared = crate::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        IMAGE_PROMPT,
        &[gate_image()],
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .unwrap_or_default()
                .iter()
                .map(|&id| id as i32)
                .collect()
        },
    )
    .expect("the image request prepares")
    .expect("a Qwen2.5-VL image request must produce vision embeddings");

    let (input_embeds, mask) = crate::vlm_runtime::prepared_embedding_refs(&prepared.embeddings)
        .expect("prepared embeddings expose an input tensor");
    let mut session = crate::backend::select_backend()
        .create_session(
            model_dir,
            mlxcel_core::generate::LanguageModel::num_layers(&model),
            mlxcel_core::cache::KVCacheMode::Fp16,
            mlxcel_core::sampling::TokenBiasMap::new(),
        )
        .expect("a generation session is created");

    session.generate_streaming_with_embeddings(
        &model,
        &prompt_tokens,
        Some(input_embeds),
        mask,
        max_tokens,
        &mlxcel_core::generate::SamplingConfig::greedy(),
        |_| true,
    )
}

/// A raw `transformers` export and its mlx conversion are the same weights
/// under different names and a different `Conv3d` filter layout, so the
/// generation path must produce the same greedy tokens from both. Before the
/// remap and the layout normalization landed, the raw export did not load at
/// all, and a hand-renamed one generated from scrambled filters.
#[test]
#[ignore = "loads two 3B Qwen2.5-VL checkpoints"]
fn qwen2_5_vl_raw_export_matches_mlx_conversion() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let (Some(raw_dir), Some(mlx_dir)) = (
        gate_checkpoint(RAW_EXPORT, RAW_EXPORT_DIR_ENV),
        gate_checkpoint(MLX_CONVERSION, MLX_CONVERSION_DIR_ENV),
    ) else {
        return;
    };

    let raw_tokens = greedy_image_tokens(&raw_dir, 24);
    let mlx_tokens = greedy_image_tokens(&mlx_dir, 24);
    assert!(
        !raw_tokens.is_empty(),
        "the raw export generated no tokens at all"
    );
    assert_eq!(
        raw_tokens, mlx_tokens,
        "the raw export and its mlx conversion must decode to the same tokens"
    );
}
