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

//! Unit tests for the Florence-2 fusion stage: the two image-side positional
//! embeddings, the joint attention mask, checkpoint-key sanitization, and the
//! assembled forward path driven end to end on a tiny synthetic model (one
//! block-free DaViT stage plus a one-layer BART stack), so the whole
//! `pixels -> features -> concat -> encoder -> decoder logits` chain is
//! exercised without a checkpoint.

use super::super::fusion::{
    LearnedPositionEmbedding2D, MAX_TEMPORAL_EMBEDDINGS, PositionalEmbeddingCosine1D,
    additive_attention_mask,
};
use super::super::{Florence2Quantization, Florence2TextConfig, Florence2VisionConfig};
use super::*;
use serde_json::json;

const D_MODEL: i32 = 8;
const IMAGE_DIM: i32 = 8;
const HALF_DIM: i32 = IMAGE_DIM / 2;
const MAX_POS_EMBEDDINGS: i32 = 8;
const MAX_TEMPORAL: i32 = 4;
const VOCAB: i32 = 16;
const MAX_POSITIONS: i32 = 32;
const FFN: i32 = 16;
const HEADS: i32 = 2;
const PIXEL_SIDE: i32 = 8;
/// Stride-2 2x2 patch embedding over an 8x8 image: a 4x4 grid, 16 tokens.
const GRID_SIDE: i32 = 4;
const GRID_TOKENS: i32 = GRID_SIDE * GRID_SIDE;

// ---------------------------------------------------------------------------
// Synthetic weights
// ---------------------------------------------------------------------------

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `Result::unwrap_err` needs `T: Debug`, which none of these MLX-backed
/// types implement (and should not, since printing one would force the graph).
fn expect_err<T>(result: Result<T, String>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    }
}

fn put(map: &mut WeightMap, key: &str, shape: &[i32], data: Vec<f32>) {
    map.insert(key.to_string(), mlxcel_core::from_slice_f32(&data, shape));
}

/// Deterministic small pseudo-random tensor, matching the seeding scheme in
/// `florence2_tests.rs` so the synthetic stacks stay numerically tame.
fn synth(map: &mut WeightMap, key: &str, shape: &[i32]) {
    let n: i32 = shape.iter().product();
    let seed: f32 = key.bytes().map(|b| b as f32).sum();
    let data: Vec<f32> = (0..n)
        .map(|i| 0.1 * ((seed + 0.7 * i as f32).sin()))
        .collect();
    put(map, key, shape, data);
}

fn synth_ln(map: &mut WeightMap, prefix: &str, dim: i32) {
    put(
        map,
        &format!("{prefix}.weight"),
        &[dim],
        vec![1.0; dim as usize],
    );
    put(
        map,
        &format!("{prefix}.bias"),
        &[dim],
        vec![0.0; dim as usize],
    );
}

fn synth_attn(map: &mut WeightMap, prefix: &str) {
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        synth(map, &format!("{prefix}.{proj}.weight"), &[D_MODEL, D_MODEL]);
        synth(map, &format!("{prefix}.{proj}.bias"), &[D_MODEL]);
    }
}

fn tiny_text_config() -> Florence2TextConfig {
    Florence2TextConfig {
        d_model: D_MODEL,
        encoder_layers: 1,
        decoder_layers: 1,
        encoder_attention_heads: HEADS,
        decoder_attention_heads: HEADS,
        encoder_ffn_dim: FFN,
        decoder_ffn_dim: FFN,
        vocab_size: VOCAB,
        max_position_embeddings: MAX_POSITIONS,
        scale_embedding: false,
        pad_token_id: 1,
        bos_token_id: 0,
        eos_token_id: 2,
        decoder_start_token_id: 2,
        quantization: Florence2Quantization::DENSE,
    }
}

fn tiny_vision_config(image_feature_source: &[&str]) -> Florence2VisionConfig {
    Florence2VisionConfig {
        depths: vec![0],
        dim_embed: vec![IMAGE_DIM],
        num_heads: vec![HEADS],
        num_groups: vec![HEADS],
        patch_size: vec![2],
        patch_stride: vec![2],
        patch_padding: vec![0],
        patch_prenorm: vec![false],
        window_size: 2,
        in_chans: 3,
        mlp_ratio: 4.0,
        qkv_bias: true,
        conv_at_attn: true,
        conv_at_ffn: true,
        drop_path_rate: 0.0,
        projection_dim: D_MODEL,
        image_pos_embed: Some(json!({
            "type": "learned_abs_2d",
            "max_pos_embeddings": MAX_POS_EMBEDDINGS,
        })),
        visual_temporal_embedding: Some(json!({
            "type": "COSINE",
            "max_temporal_embeddings": MAX_TEMPORAL,
        })),
        image_feature_source: image_feature_source.iter().map(|s| s.to_string()).collect(),
        quantization: Florence2Quantization::DENSE,
    }
}

fn tiny_config(image_feature_source: &[&str]) -> Florence2Config {
    Florence2Config {
        text: tiny_text_config(),
        vision: tiny_vision_config(image_feature_source),
        image_token_id: VOCAB,
    }
}

fn tiny_weights() -> WeightMap {
    let mut map = WeightMap::new();

    // One block-free DaViT stage: patch embedding only.
    synth(
        &mut map,
        "vision_tower.convs.0.proj.weight",
        &[IMAGE_DIM, 2, 2, 3],
    );
    synth(&mut map, "vision_tower.convs.0.proj.bias", &[IMAGE_DIM]);
    synth_ln(&mut map, "vision_tower.convs.0.norm", IMAGE_DIM);

    // Fusion stage.
    synth(&mut map, "image_projection", &[IMAGE_DIM, D_MODEL]);
    synth_ln(&mut map, "image_proj_norm", D_MODEL);
    synth(
        &mut map,
        "image_pos_embed.row_embeddings.weight",
        &[MAX_POS_EMBEDDINGS, HALF_DIM],
    );
    synth(
        &mut map,
        "image_pos_embed.column_embeddings.weight",
        &[MAX_POS_EMBEDDINGS, HALF_DIM],
    );

    // BART stack.
    synth(
        &mut map,
        "language_model.model.shared.weight",
        &[VOCAB, D_MODEL],
    );
    for side in ["encoder", "decoder"] {
        let base = format!("language_model.model.{side}");
        synth(
            &mut map,
            &format!("{base}.embed_positions.weight"),
            &[MAX_POSITIONS + 2, D_MODEL],
        );
        synth_ln(&mut map, &format!("{base}.layernorm_embedding"), D_MODEL);
        let layer = format!("{base}.layers.0");
        synth_attn(&mut map, &format!("{layer}.self_attn"));
        synth_ln(&mut map, &format!("{layer}.self_attn_layer_norm"), D_MODEL);
        if side == "decoder" {
            synth_attn(&mut map, &format!("{layer}.encoder_attn"));
            synth_ln(
                &mut map,
                &format!("{layer}.encoder_attn_layer_norm"),
                D_MODEL,
            );
        }
        synth(&mut map, &format!("{layer}.fc1.weight"), &[FFN, D_MODEL]);
        synth(&mut map, &format!("{layer}.fc1.bias"), &[FFN]);
        synth(&mut map, &format!("{layer}.fc2.weight"), &[D_MODEL, FFN]);
        synth(&mut map, &format!("{layer}.fc2.bias"), &[D_MODEL]);
        synth_ln(&mut map, &format!("{layer}.final_layer_norm"), D_MODEL);
    }
    // No `language_model.lm_head.weight`: exercises the tied-embedding fallback.
    map
}

fn tiny_model(image_feature_source: &[&str]) -> Florence2Model {
    Florence2Model::from_weights(&tiny_weights(), tiny_config(image_feature_source)).unwrap()
}

fn pixels() -> UniquePtr<MlxArray> {
    let n = (3 * PIXEL_SIDE * PIXEL_SIDE) as usize;
    let data: Vec<f32> = (0..n).map(|i| 0.05 * ((i as f32) * 0.37).sin()).collect();
    mlxcel_core::from_slice_f32(&data, &[1, 3, PIXEL_SIDE, PIXEL_SIDE])
}

// ---------------------------------------------------------------------------
// LearnedPositionEmbedding2D
// ---------------------------------------------------------------------------

/// Column embeddings occupy the first half of the feature width and row
/// embeddings the second. Swapping them would still typecheck and still
/// produce plausible activations, so pin the layout with tables whose halves
/// are distinguishable by value.
#[test]
fn learned_position_embedding_puts_column_first_then_row() {
    let mut map = WeightMap::new();
    // row[y] = 100 + y in every channel, column[x] = x in every channel.
    let rows: Vec<f32> = (0..MAX_POS_EMBEDDINGS)
        .flat_map(|y| std::iter::repeat_n(100.0 + y as f32, HALF_DIM as usize))
        .collect();
    let cols: Vec<f32> = (0..MAX_POS_EMBEDDINGS)
        .flat_map(|x| std::iter::repeat_n(x as f32, HALF_DIM as usize))
        .collect();
    put(
        &mut map,
        "pos.row_embeddings.weight",
        &[MAX_POS_EMBEDDINGS, HALF_DIM],
        rows,
    );
    put(
        &mut map,
        "pos.column_embeddings.weight",
        &[MAX_POS_EMBEDDINGS, HALF_DIM],
        cols,
    );

    let embed = LearnedPositionEmbedding2D::from_weights(
        &map,
        "pos",
        IMAGE_DIM,
        Florence2Quantization::DENSE,
    )
    .unwrap();
    let out = embed.forward(3, 2).unwrap();
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 3, 2, IMAGE_DIM]);

    let values = to_vec_f32(&out);
    for y in 0..3usize {
        for x in 0..2usize {
            let base = (y * 2 + x) * IMAGE_DIM as usize;
            for c in 0..HALF_DIM as usize {
                assert_eq!(values[base + c], x as f32, "column half at ({y},{x})");
            }
            for c in 0..HALF_DIM as usize {
                assert_eq!(
                    values[base + HALF_DIM as usize + c],
                    100.0 + y as f32,
                    "row half at ({y},{x})"
                );
            }
        }
    }
}

#[test]
fn learned_position_embedding_rejects_oversized_grid() {
    let map = tiny_weights();
    let embed = LearnedPositionEmbedding2D::from_weights(
        &map,
        "image_pos_embed",
        IMAGE_DIM,
        Florence2Quantization::DENSE,
    )
    .unwrap();
    let err = expect_err(embed.forward(MAX_POS_EMBEDDINGS + 1, 2));
    assert!(
        err.contains("max_pos_embeddings"),
        "unexpected error: {err}"
    );
    assert!(embed.forward(0, 2).is_err());
}

#[test]
fn learned_position_embedding_rejects_width_mismatch() {
    let map = tiny_weights();
    let err = expect_err(LearnedPositionEmbedding2D::from_weights(
        &map,
        "image_pos_embed",
        IMAGE_DIM + 2,
        Florence2Quantization::DENSE,
    ));
    assert!(
        err.contains("image_pos_embed.row_embeddings") && err.contains("dim_embed[-1]"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// PositionalEmbeddingCosine1D
// ---------------------------------------------------------------------------

/// With no checkpoint buffer the table is synthesized, and row 0 is
/// `sin(0), cos(0), ...` = `0, 1, 0, 1, ...`. That row is what a single-frame
/// image actually adds, so "still images make the temporal embedding a no-op"
/// is wrong and this pins it.
#[test]
fn cosine_temporal_embedding_row_zero_is_interleaved_zero_one() {
    let embed = PositionalEmbeddingCosine1D::from_weights(
        &WeightMap::new(),
        "missing",
        IMAGE_DIM,
        MAX_TEMPORAL,
    )
    .unwrap();
    let row0 = embed.forward(1).unwrap();
    assert_eq!(mlxcel_core::array_shape(&row0), vec![1, 1, IMAGE_DIM]);
    let values = to_vec_f32(&row0);
    for (i, v) in values.iter().enumerate() {
        let expected = if i % 2 == 0 { 0.0 } else { 1.0 };
        assert!((v - expected).abs() < 1e-6, "row0[{i}] = {v}");
    }

    // Row 1 follows sin/cos of `t * exp(-ln(10000) * j / embed_dim)`.
    let rows = embed.forward(2).unwrap();
    let values = to_vec_f32(&rows);
    for j in 0..(IMAGE_DIM / 2) as usize {
        let denominator = (-(10000.0f64).ln() * j as f64 / IMAGE_DIM as f64).exp();
        let base = IMAGE_DIM as usize;
        assert!((values[base + 2 * j] as f64 - denominator.sin()).abs() < 1e-6);
        assert!((values[base + 2 * j + 1] as f64 - denominator.cos()).abs() < 1e-6);
    }
}

/// Upstream registers the precomputed table as a module parameter, so a real
/// checkpoint ships it and the checkpoint copy must win over recomputation.
#[test]
fn cosine_temporal_embedding_prefers_checkpoint_table() {
    let mut map = WeightMap::new();
    let data: Vec<f32> = (0..MAX_TEMPORAL * IMAGE_DIM).map(|i| i as f32).collect();
    put(
        &mut map,
        "temporal.pos_idx_to_embed",
        &[MAX_TEMPORAL, IMAGE_DIM],
        data,
    );
    let embed =
        PositionalEmbeddingCosine1D::from_weights(&map, "temporal", IMAGE_DIM, MAX_TEMPORAL)
            .unwrap();
    let values = to_vec_f32(&embed.forward(2).unwrap());
    assert_eq!(values[0], 0.0);
    assert_eq!(values[IMAGE_DIM as usize], IMAGE_DIM as f32);
}

#[test]
fn cosine_temporal_embedding_rejects_wrong_shape_and_range() {
    let mut map = WeightMap::new();
    put(
        &mut map,
        "temporal.pos_idx_to_embed",
        &[MAX_TEMPORAL, IMAGE_DIM + 2],
        vec![0.0; (MAX_TEMPORAL * (IMAGE_DIM + 2)) as usize],
    );
    let err = expect_err(PositionalEmbeddingCosine1D::from_weights(
        &map,
        "temporal",
        IMAGE_DIM,
        MAX_TEMPORAL,
    ));
    assert!(err.contains("expected shape"), "unexpected error: {err}");

    let embed = PositionalEmbeddingCosine1D::from_weights(
        &WeightMap::new(),
        "missing",
        IMAGE_DIM,
        MAX_TEMPORAL,
    )
    .unwrap();
    assert!(embed.forward(MAX_TEMPORAL + 1).is_err());
    assert!(embed.forward(0).is_err());
}

/// `max_temporal_embeddings` is untrusted `config.json` input that sizes a
/// host buffer whenever the checkpoint omits `pos_idx_to_embed`. It has to be
/// rejected before the allocation, not after: the synthesis fallback would
/// otherwise ask the allocator for the declared size and fill it with a nested
/// transcendental loop.
#[test]
fn cosine_temporal_embedding_rejects_oversized_max_temporal() {
    let err = expect_err(PositionalEmbeddingCosine1D::from_weights(
        &WeightMap::new(),
        "temporal",
        IMAGE_DIM,
        MAX_TEMPORAL_EMBEDDINGS + 1,
    ));
    assert!(
        err.contains("max_temporal_embeddings"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// Joint attention mask
// ---------------------------------------------------------------------------

#[test]
fn additive_mask_maps_one_to_zero_and_zero_to_neg_inf() {
    let mask = mlxcel_core::from_slice_f32(&[1.0, 1.0, 0.0, 1.0], &[1, 4]);
    let additive = additive_attention_mask(&mask, mlxcel_core::dtype::FLOAT32).unwrap();
    assert_eq!(mlxcel_core::array_shape(&additive), vec![1, 1, 1, 4]);
    let values = to_vec_f32(&additive);
    assert_eq!(values[0], 0.0);
    assert_eq!(values[1], 0.0);
    assert!(values[2].is_infinite() && values[2].is_sign_negative());
    assert_eq!(values[3], 0.0);
}

/// The mask reshape reads two axes by index, so a wrong-rank mask has to
/// surface as an error rather than a panic inside library code.
#[test]
fn additive_mask_rejects_non_2d_mask() {
    let mask = mlxcel_core::from_slice_f32(&[1.0, 1.0, 1.0, 1.0], &[1, 2, 2]);
    let err = expect_err(additive_attention_mask(&mask, mlxcel_core::dtype::FLOAT32));
    assert!(err.contains("[batch, seq]"), "unexpected error: {err}");
}

// ---------------------------------------------------------------------------
// Checkpoint-key sanitization
// ---------------------------------------------------------------------------

#[test]
fn sanitize_drops_final_logits_bias() {
    let mut map = tiny_weights();
    put(
        &mut map,
        "language_model.final_logits_bias",
        &[1, VOCAB],
        vec![0.0; VOCAB as usize],
    );
    let out = sanitize(map);
    assert!(!out.contains_key("language_model.final_logits_bias"));
    assert!(out.contains_key("language_model.model.shared.weight"));
}

/// BART ties `model.shared` to both `embed_tokens` tables and exports differ
/// in which they materialize. A checkpoint carrying only `embed_tokens` must
/// still produce a usable `model.shared`.
#[test]
fn sanitize_fills_shared_from_embed_tokens() {
    let mut map = tiny_weights();
    map.remove("language_model.model.shared.weight");
    synth(
        &mut map,
        "language_model.model.encoder.embed_tokens.weight",
        &[VOCAB, D_MODEL],
    );
    let out = sanitize(map);
    let shared = out.get("language_model.model.shared.weight").unwrap();
    assert_eq!(mlxcel_core::array_shape(shared), vec![VOCAB, D_MODEL]);
}

/// The `mlx-community` bf16 export already ships channels-last conv weights,
/// so the DaViT remap must be a no-op there while still fixing a raw PyTorch
/// export.
#[test]
fn sanitize_conv_remap_is_idempotent_and_fixes_pytorch_layout() {
    let already_last = sanitize(tiny_weights());
    assert_eq!(
        mlxcel_core::array_shape(
            already_last
                .get("vision_tower.convs.0.proj.weight")
                .unwrap()
        ),
        vec![IMAGE_DIM, 2, 2, 3]
    );

    let mut pytorch = WeightMap::new();
    // `(out, in, kH, kW)` with out < kH: unambiguously PyTorch-ordered.
    synth(
        &mut pytorch,
        "vision_tower.convs.0.proj.weight",
        &[2, 3, 4, 4],
    );
    let remapped = sanitize(pytorch);
    assert_eq!(
        mlxcel_core::array_shape(remapped.get("vision_tower.convs.0.proj.weight").unwrap()),
        vec![2, 4, 4, 3]
    );
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[test]
fn config_defaults_image_token_id_to_vocab_size() {
    let raw = json!({
        "model_type": "florence2",
        "text_config": {
            "d_model": D_MODEL,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "encoder_attention_heads": HEADS,
            "decoder_attention_heads": HEADS,
            "vocab_size": VOCAB,
        },
        "vision_config": {
            "model_type": "",
            "depths": [0],
            "dim_embed": [IMAGE_DIM],
            "num_heads": [HEADS],
            "patch_size": [2],
            "patch_stride": [2],
            "patch_padding": [0],
            "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"],
        },
    });
    let config = Florence2Config::from_model_config(&raw).unwrap();
    assert_eq!(config.image_token_id, VOCAB);

    let mut with_id = raw.clone();
    with_id["image_token_index"] = json!(1234);
    assert_eq!(
        Florence2Config::from_model_config(&with_id)
            .unwrap()
            .image_token_id,
        1234
    );
}

/// Declaring quantization is no longer grounds for refusal.
///
/// #854 rejected a quantized checkpoint on the config alone, before touching
/// the weight files, because neither half of the family had a quantized code
/// path. Both do now, so the refusal moved to the weight map and covers only
/// the tensors this implementation still consumes dense (see
/// `reject_unsupported_quantized_tensors` and its tests in
/// `florence2_quantized_tests.rs`). This pins that the config-level gate is
/// gone: a directory holding only a quantizing `config.json` now fails on the
/// missing weights, which is how a dense checkpoint in the same state fails.
#[test]
fn load_no_longer_rejects_a_checkpoint_for_declaring_quantization() {
    let raw = json!({
        "model_type": "florence2",
        "quantization": { "group_size": 64, "bits": 4 },
        "text_config": {
            "d_model": D_MODEL,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "encoder_attention_heads": HEADS,
            "decoder_attention_heads": HEADS,
            "vocab_size": VOCAB,
        },
        "vision_config": {
            "model_type": "",
            "depths": [0],
            "dim_embed": [IMAGE_DIM],
            "num_heads": [HEADS],
            "patch_size": [2],
            "patch_stride": [2],
            "patch_padding": [0],
            "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"],
        },
    });

    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string(&raw).unwrap(),
    )
    .expect("write config.json");

    let err = match Florence2Model::load(dir.path()) {
        Ok(_) => panic!("a checkpoint with no safetensors must still fail"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("No safetensors files found"),
        "load must now fail on the missing weights rather than on the quantization metadata, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Load-time validation
// ---------------------------------------------------------------------------

#[test]
fn from_weights_rejects_mismatched_image_projection() {
    let mut map = tiny_weights();
    synth(&mut map, "image_projection", &[IMAGE_DIM, D_MODEL + 1]);
    let err = expect_err(Florence2Model::from_weights(
        &map,
        tiny_config(&["spatial_avg_pool"]),
    ));
    assert!(err.contains("image_projection"), "unexpected error: {err}");
}

/// `image_projection` has no fallback (it is a bare tensor with no `Linear`
/// wrapper to synthesize a default from), so a checkpoint that omits it
/// entirely must fail with the key named, distinct from the
/// shape-mismatch case above.
#[test]
fn from_weights_rejects_missing_image_projection() {
    let mut map = tiny_weights();
    map.remove("image_projection");
    let err = expect_err(Florence2Model::from_weights(
        &map,
        tiny_config(&["spatial_avg_pool"]),
    ));
    assert!(err.contains("image_projection"), "unexpected error: {err}");
}

/// The fusion stage cannot run without a position-embedding spec: unlike the
/// optional embedding *type* checks below, a missing `image_pos_embed` /
/// `visual_temporal_embedding` key in `vision_config` has to fail load with
/// the missing field named rather than defaulting to some embedding kind.
#[test]
fn from_weights_rejects_missing_position_embedding_configs() {
    let mut config = tiny_config(&["spatial_avg_pool"]);
    config.vision.image_pos_embed = None;
    let err = expect_err(Florence2Model::from_weights(&tiny_weights(), config));
    assert!(
        err.contains("missing image_pos_embed"),
        "unexpected error: {err}"
    );

    let mut config = tiny_config(&["spatial_avg_pool"]);
    config.vision.visual_temporal_embedding = None;
    let err = expect_err(Florence2Model::from_weights(&tiny_weights(), config));
    assert!(
        err.contains("missing visual_temporal_embedding"),
        "unexpected error: {err}"
    );
}

#[test]
fn from_weights_rejects_unsupported_embedding_types() {
    let mut config = tiny_config(&["spatial_avg_pool"]);
    config.vision.image_pos_embed = Some(json!({"type": "sine_2d"}));
    let err = expect_err(Florence2Model::from_weights(&tiny_weights(), config));
    assert!(err.contains("learned_abs_2d"), "unexpected error: {err}");

    let mut config = tiny_config(&["spatial_avg_pool"]);
    config.vision.visual_temporal_embedding = Some(json!({"type": "LEARNED"}));
    let err = expect_err(Florence2Model::from_weights(&tiny_weights(), config));
    assert!(err.contains("COSINE"), "unexpected error: {err}");
}

/// `max_position_embeddings` is the bound both the fused-encoder length check
/// and the decode-offset check are written against, but nothing forces the
/// checkpoint's position table to actually cover it. A short table has to fail
/// here, at load, with the shapes named: past those guards the slice reaches
/// MLX out of range and aborts the process across the FFI boundary.
#[test]
fn from_weights_rejects_short_position_table() {
    let mut map = tiny_weights();
    // One row short of `POSITION_OFFSET + max_position_embeddings`.
    synth(
        &mut map,
        "language_model.model.encoder.embed_positions.weight",
        &[MAX_POSITIONS + 1, D_MODEL],
    );
    let err = expect_err(Florence2Model::from_weights(
        &map,
        tiny_config(&["spatial_avg_pool"]),
    ));
    assert!(
        err.contains("embed_positions.weight") && err.contains("max_position_embeddings"),
        "unexpected error: {err}"
    );
}

/// A table whose width disagrees with `d_model` is the same hazard on the
/// other axis: the slice takes `d_model` columns.
#[test]
fn from_weights_rejects_position_table_width_mismatch() {
    let mut map = tiny_weights();
    synth(
        &mut map,
        "language_model.model.decoder.embed_positions.weight",
        &[MAX_POSITIONS + 2, D_MODEL + 1],
    );
    let err = expect_err(Florence2Model::from_weights(
        &map,
        tiny_config(&["spatial_avg_pool"]),
    ));
    assert!(
        err.contains("embed_positions.weight"),
        "unexpected error: {err}"
    );
}

/// `encoder_attention_heads: 0` used to reach the divisibility check and
/// panic there with a divide by zero, and a negative layer count reached
/// `Vec::with_capacity` as `usize::MAX`. Both values come straight out of
/// `config.json`, so both must return `Err`.
#[test]
fn text_config_rejects_degenerate_shape_fields() {
    let zero_heads = json!({
        "d_model": D_MODEL,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_attention_heads": 0,
        "decoder_attention_heads": HEADS,
        "vocab_size": VOCAB,
    });
    let err = Florence2TextConfig::from_model_config(&zero_heads)
        .unwrap_err()
        .to_string();
    assert!(err.contains("attention heads"), "unexpected error: {err}");

    let negative_layers = json!({
        "d_model": D_MODEL,
        "encoder_layers": -1,
        "decoder_layers": 1,
        "encoder_attention_heads": HEADS,
        "decoder_attention_heads": HEADS,
        "vocab_size": VOCAB,
    });
    let err = Florence2TextConfig::from_model_config(&negative_layers)
        .unwrap_err()
        .to_string();
    assert!(err.contains("layer counts"), "unexpected error: {err}");
}

/// `pad_token_id` and `decoder_start_token_id` are used as literal gather
/// indices into the shared embedding, so an out-of-vocabulary value is an
/// out-of-range gather rather than a wrong-looking caption.
#[test]
fn text_config_rejects_out_of_vocabulary_token_ids() {
    let base = json!({
        "d_model": D_MODEL,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_attention_heads": HEADS,
        "decoder_attention_heads": HEADS,
        "vocab_size": VOCAB,
    });

    let mut oversized_start = base.clone();
    oversized_start["decoder_start_token_id"] = json!(VOCAB);
    assert!(Florence2TextConfig::from_model_config(&oversized_start).is_err());

    let mut negative_pad = base;
    negative_pad["pad_token_id"] = json!(-1);
    assert!(Florence2TextConfig::from_model_config(&negative_pad).is_err());
}

#[test]
fn from_weights_rejects_empty_image_feature_source() {
    let err = expect_err(Florence2Model::from_weights(
        &tiny_weights(),
        tiny_config(&[]),
    ));
    assert!(
        err.contains("image_feature_source"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// Fused forward path
// ---------------------------------------------------------------------------

#[test]
fn encode_image_pools_and_projects_to_text_width() {
    let model = tiny_model(&["spatial_avg_pool", "temporal_avg_pool"]);
    let features = model.encode_image(&pixels()).unwrap();
    // 1 spatially averaged token + one token per grid position.
    assert_eq!(
        mlxcel_core::array_shape(&features),
        vec![1, 1 + GRID_TOKENS, D_MODEL]
    );

    let temporal_only = tiny_model(&["temporal_avg_pool"]);
    assert_eq!(
        mlxcel_core::array_shape(&temporal_only.encode_image(&pixels()).unwrap()),
        vec![1, GRID_TOKENS, D_MODEL]
    );
}

/// `image_feature_source` is order-sensitive: it decides the layout of the
/// concatenated feature sequence. Upstream's dataclass default lists the two
/// pools in the opposite order from every real checkpoint, so this pins that
/// mlxcel follows the config rather than a default.
#[test]
fn image_feature_source_order_controls_layout() {
    let spatial_first = tiny_model(&["spatial_avg_pool", "temporal_avg_pool"]);
    let temporal_first = tiny_model(&["temporal_avg_pool", "spatial_avg_pool"]);
    let a = to_vec_f32(&spatial_first.encode_image(&pixels()).unwrap());
    let b = to_vec_f32(&temporal_first.encode_image(&pixels()).unwrap());
    assert_eq!(a.len(), b.len());

    let width = D_MODEL as usize;
    let grid = GRID_TOKENS as usize;
    // Spatial-first puts the pooled token at index 0; temporal-first puts it last.
    assert_eq!(&a[..width], &b[grid * width..]);
    assert_eq!(&a[width..], &b[..grid * width]);
}

#[test]
fn encode_image_rejects_non_square_feature_maps() {
    let model = tiny_model(&["spatial_avg_pool"]);
    let features =
        mlxcel_core::from_slice_f32(&vec![0.0; (5 * IMAGE_DIM) as usize], &[1, 5, IMAGE_DIM]);
    let err = expect_err(model.encode_image_features(&features));
    assert!(
        err.contains("square feature map"),
        "unexpected error: {err}"
    );
}

/// A backbone whose last `dim_embed` disagrees with the feature width it
/// actually emits (a mismatched tower/fusion pairing) has to be caught before
/// the position-embedding add, which would otherwise broadcast against the
/// wrong axis instead of failing cleanly.
#[test]
fn encode_image_rejects_feature_width_mismatch() {
    let model = tiny_model(&["spatial_avg_pool"]);
    let features = mlxcel_core::from_slice_f32(
        &vec![0.0; (GRID_TOKENS * (IMAGE_DIM + 2)) as usize],
        &[1, GRID_TOKENS, IMAGE_DIM + 2],
    );
    let err = expect_err(model.encode_image_features(&features));
    assert!(err.contains("dim_embed[-1]"), "unexpected error: {err}");
}

/// `image_feature_source` entries outside the three known pooling recipes
/// have to fail at the point they are used, not silently pass through as one
/// of the known pools.
#[test]
fn encode_image_rejects_unsupported_feature_source() {
    let model = tiny_model(&["bogus_pool"]);
    let err = expect_err(model.encode_image(&pixels()));
    assert!(
        err.contains("bogus_pool") && err.contains("not supported"),
        "unexpected error: {err}"
    );
}

/// Florence-2 concatenates rather than scattering into placeholder slots, so
/// both halves of the fused sequence must survive byte for byte.
#[test]
fn merge_concatenates_image_then_prompt() {
    let model = tiny_model(&["spatial_avg_pool", "temporal_avg_pool"]);
    let features = model.encode_image(&pixels()).unwrap();
    let prompt_ids = [3i32, 4, 5];
    let prompt_embeds = model.embed_prompt(&prompt_ids).unwrap();
    let (fused, mask) = model
        .merge_input_ids_with_image_features(&features, Some(&prompt_embeds))
        .unwrap();

    let image_len = 1 + GRID_TOKENS;
    let total = image_len + prompt_ids.len() as i32;
    assert_eq!(
        mlxcel_core::array_shape(&fused),
        vec![1, total, D_MODEL],
        "fused sequence length"
    );
    assert_eq!(mlxcel_core::array_shape(&mask), vec![1, total]);
    assert!(to_vec_f32(&mask).iter().all(|v| *v == 1.0));

    let fused_values = to_vec_f32(&fused);
    let split = (image_len * D_MODEL) as usize;
    assert_eq!(&fused_values[..split], to_vec_f32(&features).as_slice());
    assert_eq!(
        &fused_values[split..],
        to_vec_f32(&prompt_embeds).as_slice()
    );
}

#[test]
fn merge_without_prompt_returns_image_features_alone() {
    let model = tiny_model(&["spatial_avg_pool"]);
    let features = model.encode_image(&pixels()).unwrap();
    let (fused, mask) = model
        .merge_input_ids_with_image_features(&features, None)
        .unwrap();
    assert_eq!(mlxcel_core::array_shape(&fused), vec![1, 1, D_MODEL]);
    assert_eq!(mlxcel_core::array_shape(&mask), vec![1, 1]);

    // A prompt made only of image placeholder ids is the same case.
    assert!(model.embed_prompt(&[VOCAB, VOCAB]).is_none());
    assert!(model.embed_prompt(&[]).is_none());
}

/// The rank check on `image_features` is what keeps a caller-supplied
/// two-image-batch tensor (or any other unexpected rank) from panicking on
/// `shape[2]` inside library code.
#[test]
fn merge_rejects_non_3d_image_features() {
    let model = tiny_model(&["spatial_avg_pool"]);
    let features = mlxcel_core::from_slice_f32(&vec![0.0; D_MODEL as usize], &[D_MODEL]);
    let err = expect_err(model.merge_input_ids_with_image_features(&features, None));
    assert!(
        err.contains("[batch, tokens, dim]"),
        "unexpected error: {err}"
    );
}

/// A prompt embedding whose batch or feature width disagrees with the image
/// features it is being concatenated with has to fail before the
/// concatenate, which would otherwise either panic or silently broadcast.
#[test]
fn merge_rejects_incompatible_prompt_shape() {
    let model = tiny_model(&["spatial_avg_pool"]);
    let features = model.encode_image(&pixels()).unwrap();

    let wrong_batch =
        mlxcel_core::from_slice_f32(&vec![0.0; 3 * D_MODEL as usize], &[2, 3, D_MODEL]);
    let err = expect_err(model.merge_input_ids_with_image_features(&features, Some(&wrong_batch)));
    assert!(err.contains("not compatible"), "unexpected error: {err}");

    let wrong_width =
        mlxcel_core::from_slice_f32(&vec![0.0; 3 * (D_MODEL + 1) as usize], &[1, 3, D_MODEL + 1]);
    let err = expect_err(model.merge_input_ids_with_image_features(&features, Some(&wrong_width)));
    assert!(err.contains("not compatible"), "unexpected error: {err}");
}

/// The integrated path must be exactly "fuse, then run the text encoder over
/// the fused embeddings with the joint mask", not a separate code path.
#[test]
fn encode_matches_manual_fusion_then_text_encoder() {
    let model = tiny_model(&["spatial_avg_pool", "temporal_avg_pool"]);
    let prompt_ids = [3i32, 4, 5];

    let features = model.encode_image(&pixels()).unwrap();
    let prompt_embeds = model.embed_prompt(&prompt_ids).unwrap();
    let (fused, mask) = model
        .merge_input_ids_with_image_features(&features, Some(&prompt_embeds))
        .unwrap();
    let additive = additive_attention_mask(&mask, model.dtype()).unwrap();
    let manual = model
        .text_model()
        .encode_embeds_with_mask(&fused, Some(&additive));

    let integrated = model.encode(&pixels(), &prompt_ids).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(&integrated),
        mlxcel_core::array_shape(&manual)
    );
    for (a, b) in to_vec_f32(&integrated)
        .iter()
        .zip(to_vec_f32(&manual).iter())
    {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}

/// The integrated path skips the additive-mask conversion because the mask it
/// builds is all ones. That is only sound if an all-ones mask really is the
/// identity, and the conversion is only worth keeping if a mask with a zero
/// really changes the answer. Pin both halves.
#[test]
fn encoder_mask_is_identity_when_all_ones_and_load_bearing_otherwise() {
    let model = tiny_model(&["spatial_avg_pool", "temporal_avg_pool"]);
    let prompt_ids = [3i32, 4, 5];

    let features = model.encode_image(&pixels()).unwrap();
    let prompt_embeds = model.embed_prompt(&prompt_ids).unwrap();
    let (fused, mask) = model
        .merge_input_ids_with_image_features(&features, Some(&prompt_embeds))
        .unwrap();

    let unmasked = to_vec_f32(&model.encode_fused(&fused, None).unwrap());
    let all_ones = to_vec_f32(&model.encode_fused(&fused, Some(&mask)).unwrap());
    assert_eq!(unmasked, all_ones, "an all-ones mask must be the identity");

    // Drop the last prompt position out of the mask.
    let seq = mlxcel_core::array_shape(&mask)[1];
    let mut values = vec![1.0f32; seq as usize];
    values[seq as usize - 1] = 0.0;
    let padded = mlxcel_core::from_slice_f32(&values, &[1, seq]);
    let masked = to_vec_f32(&model.encode_fused(&fused, Some(&padded)).unwrap());
    assert_ne!(
        unmasked, masked,
        "masking a position out must change the encoder output"
    );
    assert!(
        masked.iter().all(|v| v.is_finite()),
        "masked encoder output must stay finite"
    );
}

#[test]
fn encode_rejects_sequences_past_max_position_embeddings() {
    let model = tiny_model(&["spatial_avg_pool", "temporal_avg_pool"]);
    let prompt: Vec<i32> = (0..MAX_POSITIONS).map(|i| i % VOCAB).collect();
    let err = expect_err(model.encode(&pixels(), &prompt));
    assert!(
        err.contains("max_position_embeddings"),
        "unexpected error: {err}"
    );
}

/// [`Florence2Model::encode_fused`] is a `pub` entry point a caller can drive
/// directly with a hand-built tensor, so its own rank check (rather than a
/// downstream MLX panic) has to be what catches a wrong-rank input.
#[test]
fn encode_fused_rejects_wrong_rank_input() {
    let model = tiny_model(&["spatial_avg_pool"]);
    let flat = mlxcel_core::from_slice_f32(&vec![0.0; D_MODEL as usize], &[1, D_MODEL]);
    let err = expect_err(model.encode_fused(&flat, None));
    assert!(
        err.contains("[batch, seq, d_model]"),
        "unexpected error: {err}"
    );
}

/// End-to-end: pixels and a prompt in, decoder token ids out, through the same
/// dual-KV-cache decode loop the text engine uses.
#[test]
fn generate_greedy_runs_the_whole_fused_path() {
    let model = tiny_model(&["spatial_avg_pool", "temporal_avg_pool"]);
    let generated = model.generate_greedy(&pixels(), &[3, 4, 5], 6).unwrap();
    assert!(generated.len() <= 6);
    for token in &generated {
        assert!(
            (0..VOCAB).contains(token),
            "token {token} outside vocabulary"
        );
        assert_ne!(*token, model.config().text.eos_token_id);
    }

    // The single-step decode against the same encoder output must agree with
    // the first token of the greedy loop.
    let encoder_hidden = model.encode(&pixels(), &[3, 4, 5]).unwrap();
    let mut cache = model.make_cache();
    let start = mlxcel_core::from_slice_i32(&[model.config().text.decoder_start_token_id], &[1, 1]);
    let logits = model.decode(&start, &encoder_hidden, &mut cache);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 1, VOCAB],
        "decoder logits shape"
    );
    if let Some(first) = generated.first() {
        assert_eq!(*first, argmax_last_position(&logits).unwrap());
    }
}
