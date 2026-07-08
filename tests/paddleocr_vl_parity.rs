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

//! PaddleOCR-VL parity and integration checks.
//!
//! Two layers, neither of which needs a real checkpoint (deferred to the
//! orchestrator's real-model validation):
//!
//! 1. Reference parity for the OCR-specific `smart_resize` and grid derivation.
//!    Expected values are computed from the mlx-vlm reference formula in
//!    https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/paddleocr_vl/processing_paddleocr_vl.py
//!    (factor = patch*merge = 28, min/max pixel clamps with floor/ceil beta
//!    scaling).
//! 2. Construct-and-run shape checks that wire the net-new NaViT vision encoder,
//!    the spatial-merge projector, and the ERNIE-4.5 MRoPE text decoder from
//!    synthetic weights and exercise the full tensor plumbing end to end.

use image::{DynamicImage, RgbImage};

use mlxcel::models::paddleocr_vl::{PaddleOcrTextConfig, PaddleOcrTextModel, RopeScaling};
use mlxcel::vision::connectors::paddleocr_vl::PaddleOcrProjector;
use mlxcel::vision::encoders::paddleocr_vl::{PaddleOcrVisionConfig, PaddleOcrVisionEncoder};
use mlxcel::vision::processors::paddleocr_vl::PaddleOcrVlProcessor;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// ---------- reference parity: smart_resize / grid ----------

#[test]
fn smart_resize_matches_reference_formula() {
    let p = PaddleOcrVlProcessor::new(14, 2); // factor = 28

    // In-range: round each side to a multiple of 28.
    assert_eq!(p.smart_resize(384, 384), (392, 392));
    assert_eq!(p.smart_resize(840, 560), (840, 560));

    // Max-pixel clamp: 3000x3000 rounds to 2996x2996 (> 2_822_400), beta
    // scaling floors back to exactly the 2_822_400 budget (1680x1680).
    assert_eq!(p.smart_resize(3000, 3000), (1680, 1680));

    // Min-pixel clamp: 100x100 rounds to 112x112 (< 147_384), beta scaling
    // ceils up to 392x392.
    assert_eq!(p.smart_resize(100, 100), (392, 392));
}

#[test]
fn compute_grid_thw_uses_smart_resized_patch_counts() {
    let p = PaddleOcrVlProcessor::new(14, 2);
    // width=560, height=840 -> smart_resize(840, 560) = (840, 560)
    // grid = (1, 840/14, 560/14) = (1, 60, 40).
    let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(560, 840, image::Rgb([120, 60, 30])));
    assert_eq!(p.compute_grid_thw(&[img]), vec![(1, 60, 40)]);
}

#[test]
fn preprocess_emits_flat_per_patch_pixel_values() {
    let p = PaddleOcrVlProcessor::new(14, 2);
    // 392x392 is already in range -> grid (1, 28, 28) = 784 patches,
    // each patch is C*patch*patch = 3*14*14 = 588 features.
    let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(392, 392, image::Rgb([255, 0, 0])));
    let (pixel_values, grid) = p.preprocess_with_grid(&[img]);
    assert_eq!(grid, vec![(1, 28, 28)]);
    let shape = mlxcel_core::array_shape(pixel_values.as_ref().unwrap());
    assert_eq!(shape, vec![784, 588]);
}

// ---------- synthetic-weight construct-and-run shape checks ----------

fn varied(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (((i % 13) as f32) - 6.0) * 0.05).collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn constant(shape: &[i32], val: f32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![val; n as usize], shape)
}

fn put(map: &mut WeightMap, key: &str, shape: &[i32]) {
    map.insert(key.to_string(), varied(shape));
}

fn put_linear(map: &mut WeightMap, prefix: &str, out: i32, inp: i32, bias: bool) {
    put(map, &format!("{prefix}.weight"), &[out, inp]);
    if bias {
        map.insert(format!("{prefix}.bias"), constant(&[out], 0.0));
    }
}

fn put_layer_norm(map: &mut WeightMap, prefix: &str, dim: i32) {
    map.insert(format!("{prefix}.weight"), constant(&[dim], 1.0));
    map.insert(format!("{prefix}.bias"), constant(&[dim], 0.0));
}

fn vision_config() -> PaddleOcrVisionConfig {
    PaddleOcrVisionConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_channels: 3,
        image_size: 4,
        patch_size: 2,
        layer_norm_eps: 1e-6,
        spatial_merge_size: 2,
        quant_group_size: 0,
        quant_bits: 0,
    }
}

fn vision_weights() -> WeightMap {
    let mut w = WeightMap::new();
    // patch_embedding as a 2D linear [embed, C*patch*patch] = [8, 12].
    put(&mut w, "visual.embeddings.patch_embedding.weight", &[8, 12]);
    w.insert(
        "visual.embeddings.patch_embedding.bias".into(),
        constant(&[8], 0.0),
    );
    // learned position embedding: num_positions = (image_size/patch)^2 = 4.
    put(
        &mut w,
        "visual.embeddings.position_embedding.weight",
        &[4, 8],
    );
    put_linear(&mut w, "visual.layers.0.self_attn.qkv", 24, 8, true);
    put_linear(&mut w, "visual.layers.0.self_attn.out_proj", 8, 8, true);
    put_layer_norm(&mut w, "visual.layers.0.layer_norm1", 8);
    put_layer_norm(&mut w, "visual.layers.0.layer_norm2", 8);
    put_linear(&mut w, "visual.layers.0.mlp.fc1", 16, 8, true);
    put_linear(&mut w, "visual.layers.0.mlp.fc2", 8, 16, true);
    put_layer_norm(&mut w, "visual.post_layernorm", 8);
    w
}

#[test]
fn vision_encoder_forward_produces_per_token_hidden_states() {
    let config = vision_config();
    let weights = vision_weights();
    let encoder = PaddleOcrVisionEncoder::from_weights(&weights, &config, "visual")
        .expect("build vision encoder");

    // grid (1, 2, 2): 4 patches, each C*patch*patch = 3*2*2 = 12 features.
    let pixel_values = varied(&[4, 12]);
    let out = encoder.forward_with_grid(pixel_values.as_ref().unwrap(), &[(1, 2, 2)]);
    let shape = mlxcel_core::array_shape(out.hidden_states.as_ref().unwrap());
    assert_eq!(
        shape,
        vec![4, 8],
        "per-token vision hidden states [tokens, embed]"
    );
}

#[test]
fn vision_encoder_forward_handles_mixed_dynamic_resolution_batch() {
    let config = vision_config();
    let weights = vision_weights();
    let encoder = PaddleOcrVisionEncoder::from_weights(&weights, &config, "visual")
        .expect("build vision encoder");

    // 2x2, 2x3, 2x2 patch grids exercise the repeated-length bucketed attention
    // path while preserving the original packed token order.
    let grids = [(1, 2, 2), (1, 2, 3), (1, 2, 2)];
    let pixel_values = varied(&[14, 12]);
    let out = encoder.forward_with_grid(pixel_values.as_ref().unwrap(), &grids);
    let shape = mlxcel_core::array_shape(out.hidden_states.as_ref().unwrap());
    assert_eq!(
        shape,
        vec![14, 8],
        "mixed dynamic-resolution batch keeps packed [tokens, embed] output"
    );
}

#[test]
fn vision_encoder_forward_handles_unique_variable_resolution_batch() {
    let config = vision_config();
    let weights = vision_weights();
    let encoder = PaddleOcrVisionEncoder::from_weights(&weights, &config, "visual")
        .expect("build vision encoder");

    // Unique segment lengths intentionally stay on the per-segment fallback so
    // the fast-path dispatch does not force a dense block mask for page sizes
    // where sequential block-diagonal attention is cheaper and safer.
    let grids = [(1, 2, 2), (1, 2, 3), (1, 4, 2)];
    let pixel_values = varied(&[18, 12]);
    let out = encoder.forward_with_grid(pixel_values.as_ref().unwrap(), &grids);
    let shape = mlxcel_core::array_shape(out.hidden_states.as_ref().unwrap());
    assert_eq!(
        shape,
        vec![18, 8],
        "unique variable-resolution batch keeps packed [tokens, embed] output"
    );
}

#[test]
fn projector_merges_spatial_blocks_to_text_hidden() {
    let mut w = WeightMap::new();
    put_layer_norm(&mut w, "visual.projector.pre_norm", 8);
    // merge^2 * embed = 4 * 8 = 32 -> 32 -> text_hidden 8.
    put_linear(&mut w, "visual.projector.linear_1", 32, 32, true);
    put_linear(&mut w, "visual.projector.linear_2", 8, 32, true);
    let projector =
        PaddleOcrProjector::from_weights(&w, "visual.projector", 2, 0, 0).expect("build projector");

    // 4 vision tokens (grid 1x2x2) -> 1 merged token of text width 8.
    let hidden = varied(&[4, 8]);
    let out = projector.forward_with_grid(hidden.as_ref().unwrap(), &[(1, 2, 2)]);
    let shape = mlxcel_core::array_shape(out.as_ref().unwrap());
    assert_eq!(shape, vec![1, 8], "merged tokens [t*h/m*w/m, text_hidden]");
}

fn text_config() -> PaddleOcrTextConfig {
    PaddleOcrTextConfig {
        hidden_size: 8,
        num_hidden_layers: 1,
        intermediate_size: 16,
        num_attention_heads: 2,
        num_key_value_heads: Some(1),
        vocab_size: 10,
        rms_norm_eps: 1e-5,
        rope_theta: 500000.0,
        rope_scaling: Some(RopeScaling {
            mrope_section: vec![1, 1],
            scaling_type: "default".into(),
        }),
        head_dim: Some(4),
        use_bias: false,
        tie_word_embeddings: false,
        quantization: None,
    }
}

fn text_weights() -> WeightMap {
    let mut w = WeightMap::new();
    put(&mut w, "model.embed_tokens.weight", &[10, 8]);
    // head_dim=4: q = heads*hd = 8, k/v = kv_heads*hd = 4, o = heads*hd -> hidden.
    put_linear(&mut w, "model.layers.0.self_attn.q_proj", 8, 8, false);
    put_linear(&mut w, "model.layers.0.self_attn.k_proj", 4, 8, false);
    put_linear(&mut w, "model.layers.0.self_attn.v_proj", 4, 8, false);
    put_linear(&mut w, "model.layers.0.self_attn.o_proj", 8, 8, false);
    put_linear(&mut w, "model.layers.0.mlp.gate_proj", 16, 8, false);
    put_linear(&mut w, "model.layers.0.mlp.up_proj", 16, 8, false);
    put_linear(&mut w, "model.layers.0.mlp.down_proj", 8, 16, false);
    w.insert(
        "model.layers.0.input_layernorm.weight".into(),
        constant(&[8], 1.0),
    );
    w.insert(
        "model.layers.0.post_attention_layernorm.weight".into(),
        constant(&[8], 1.0),
    );
    w.insert("model.norm.weight".into(), constant(&[8], 1.0));
    put(&mut w, "lm_head.weight", &[10, 8]);
    w
}

#[test]
fn text_model_forward_produces_vocab_logits() {
    let config = text_config();
    let weights = text_weights();
    let model = PaddleOcrTextModel::from_weights(&weights, &config).expect("build text model");
    assert_eq!(model.eos_token_ids(), vec![2], "ERNIE-4.5 EOS token");

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let mut caches = LanguageModel::make_caches(&model);
    let logits = model.forward(input_ids.as_ref().unwrap(), &mut caches, None);
    let shape = mlxcel_core::array_shape(logits.as_ref().unwrap());
    assert_eq!(shape, vec![1, 3, 10], "logits [batch, seq, vocab]");
}
