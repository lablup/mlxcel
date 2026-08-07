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

//! Checkpoint-free tests for the Florence-2 quantized load path: parsing the
//! top-level `quantization` block, the narrowed load-time refusal that
//! replaced #854's blanket one, and the shared-embedding fill that has to
//! carry all three planes of a packed table.
//!
//! The numeric half of the quantized path is covered by
//! `tests/florence2_quantized_parity.rs`, which needs a real checkpoint. These
//! run everywhere.

use super::*;
use serde_json::json;

fn dummy(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![0.0f32; n as usize], shape)
}

fn dummy_u32(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_u32(&vec![0u32; n as usize], shape)
}

/// A weight map shaped like the packed planes `mlx-community` ships: a
/// `uint32` weight beside float `scales` and `biases`.
fn quantized_triple(map: &mut WeightMap, prefix: &str) {
    map.insert(format!("{prefix}.weight"), dummy_u32(&[8, 2]));
    map.insert(format!("{prefix}.scales"), dummy(&[8, 1]));
    map.insert(format!("{prefix}.biases"), dummy(&[8, 1]));
}

// ---------------------------------------------------------------------------
// Florence2Quantization
// ---------------------------------------------------------------------------

#[test]
fn quantization_defaults_to_dense_without_a_block() {
    let config = json!({ "model_type": "florence2" });
    let parsed = Florence2Quantization::from_model_config(&config).expect("parse");
    assert_eq!(parsed, Florence2Quantization::DENSE);
    assert!(!Florence2Quantization::config_is_quantized(&config));
}

#[test]
fn quantization_reads_the_top_level_block() {
    for bits in [3, 4, 6, 8] {
        let config = json!({
            "model_type": "florence2",
            "quantization": { "group_size": 64, "bits": bits },
        });
        let parsed = Florence2Quantization::from_model_config(&config).expect("parse");
        assert_eq!(parsed.group_size, 64, "group_size at {bits} bits");
        assert_eq!(parsed.bits, bits, "bits at {bits} bits");
        assert!(Florence2Quantization::config_is_quantized(&config));
    }
}

/// A genuinely 4-bit, group-64 export declares exactly what [`DENSE`] holds,
/// so equality against `DENSE` is not a quantization test and
/// `config_is_quantized` has to answer from the block's presence instead.
/// This pins that distinction, which is what gates the bf16 to f16
/// conversion.
///
/// [`DENSE`]: Florence2Quantization::DENSE
#[test]
fn quantization_presence_is_not_inferable_from_the_parsed_values() {
    let quantized = json!({
        "model_type": "florence2",
        "quantization": { "group_size": 64, "bits": 4 },
    });
    let dense = json!({ "model_type": "florence2" });
    assert_eq!(
        Florence2Quantization::from_model_config(&quantized).unwrap(),
        Florence2Quantization::from_model_config(&dense).unwrap(),
        "the published 4-bit parameters are the dense defaults"
    );
    assert!(Florence2Quantization::config_is_quantized(&quantized));
    assert!(!Florence2Quantization::config_is_quantized(&dense));
}

#[test]
fn quantization_rejects_out_of_range_parameters() {
    for (group_size, bits, needle) in [
        (0, 4, "group_size"),
        (-64, 4, "group_size"),
        (1_000_000, 4, "group_size"),
        (64, 0, "bits"),
        (64, 33, "bits"),
        (64, -4, "bits"),
    ] {
        let config = json!({
            "model_type": "florence2",
            "quantization": { "group_size": group_size, "bits": bits },
        });
        let err = Florence2Quantization::from_model_config(&config)
            .expect_err("out-of-range quantization parameters must be refused")
            .to_string();
        assert!(
            err.contains(needle),
            "group_size {group_size} / bits {bits}: unexpected error {err}"
        );
    }
}

/// A present-but-unreadable parameter must not fall back to the dense default.
/// The unified layers trust the declared group size, so substituting 64 for a
/// `group_size` written as a float or a string dequantizes the whole model on
/// the wrong stride and decodes to plausible-looking garbage rather than
/// failing. The last case is the truncating `as i32` cast: a JSON integer past
/// `i32::MAX` used to wrap into a small in-range number instead of being
/// rejected.
#[test]
fn quantization_rejects_malformed_parameters() {
    for (block, needle) in [
        (json!({ "group_size": 64.5, "bits": 4 }), "group_size"),
        (json!({ "group_size": "64", "bits": 4 }), "group_size"),
        (json!({ "group_size": null, "bits": 4 }), "group_size"),
        (json!({ "group_size": 64, "bits": "4" }), "bits"),
        (
            json!({ "group_size": 64, "bits": 4_294_967_296i64 }),
            "bits",
        ),
        (
            json!({ "group_size": 4_294_967_360i64, "bits": 4 }),
            "group_size",
        ),
    ] {
        let config = json!({ "model_type": "florence2", "quantization": block });
        let err = Florence2Quantization::from_model_config(&config)
            .expect_err("a malformed quantization parameter must be refused")
            .to_string();
        assert!(err.contains(needle), "{config}: unexpected error {err}");
    }
}

/// The complement of the test above: an *absent* field still takes the dense
/// default, so a config that declares only `bits` keeps loading.
#[test]
fn quantization_defaults_only_absent_parameters() {
    let config = json!({
        "model_type": "florence2",
        "quantization": { "bits": 8 },
    });
    let parsed = Florence2Quantization::from_model_config(&config).expect("parse");
    assert_eq!(parsed.group_size, Florence2Quantization::DENSE.group_size);
    assert_eq!(parsed.bits, 8);
}

/// The block sits beside `text_config` and `vision_config`, not inside
/// either, so a parser that only descended into a sub-object would leave both
/// halves on the dense default and mis-stride every dequantization.
#[test]
fn both_half_configs_inherit_the_top_level_quantization_block() {
    let config = json!({
        "model_type": "florence2",
        "quantization": { "group_size": 32, "bits": 8 },
        "text_config": {
            "d_model": 64,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "encoder_attention_heads": 2,
            "decoder_attention_heads": 2,
            "vocab_size": 64,
        },
        "vision_config": {
            "depths": [1],
            "dim_embed": [16],
            "num_heads": [2],
            "num_groups": [2],
            "patch_size": [3],
            "patch_stride": [2],
            "patch_padding": [1],
        },
    });
    let parsed = Florence2Config::from_model_config(&config).expect("parse florence2 config");
    let expected = Florence2Quantization {
        group_size: 32,
        bits: 8,
    };
    assert_eq!(parsed.text.quantization, expected, "text half");
    assert_eq!(parsed.vision.quantization, expected, "vision half");
}

// ---------------------------------------------------------------------------
// Narrowed load-time refusal
// ---------------------------------------------------------------------------

/// The projections and embedding tables the published conversions pack must
/// all be accepted. #854 refused this map wholesale.
#[test]
fn quantized_projections_and_tables_are_accepted() {
    let mut map = WeightMap::new();
    for prefix in [
        "language_model.model.shared",
        "language_model.lm_head",
        "language_model.model.encoder.embed_positions",
        "language_model.model.decoder.embed_positions",
        "language_model.model.encoder.layers.0.self_attn.q_proj",
        "language_model.model.decoder.layers.0.encoder_attn.v_proj",
        "language_model.model.decoder.layers.0.fc1",
        "image_pos_embed.row_embeddings",
        "image_pos_embed.column_embeddings",
        "vision_tower.blocks.0.0.spatial_block.window_attn.fn.qkv",
        "vision_tower.blocks.0.0.spatial_block.window_attn.fn.proj",
        "vision_tower.blocks.0.0.channel_block.channel_attn.fn.qkv",
        "vision_tower.blocks.0.0.channel_block.ffn.fn.net.fc2",
    ] {
        quantized_triple(&mut map, prefix);
    }
    // The tensors that stay dense in a real conversion.
    map.insert("image_projection".to_string(), dummy(&[16, 8]));
    map.insert(
        "visual_temporal_embed.pos_idx_to_embed".to_string(),
        dummy(&[4, 16]),
    );
    map.insert(
        "vision_tower.convs.0.proj.weight".to_string(),
        dummy(&[8, 3, 3, 3]),
    );

    reject_unsupported_quantized_tensors(&map).expect("published packing must be accepted");
}

/// The narrowed refusal still has to fire on the tensors the forward path
/// consumes dense. Each of these would otherwise hand a packed `uint32`
/// tensor to `matmul`, `slice`, or `conv2d`, and MLX's throw cannot cross the
/// cxx bridge, so the process would abort instead of returning an error.
#[test]
fn quantized_dense_only_tensors_are_refused_by_name() {
    for prefix in [
        "image_projection",
        "visual_temporal_embed.pos_idx_to_embed",
        "vision_tower.convs.0.proj",
        "vision_tower.blocks.0.0.spatial_block.conv1.fn.dw",
    ] {
        let mut map = WeightMap::new();
        quantized_triple(&mut map, prefix);
        let err = reject_unsupported_quantized_tensors(&map)
            .expect_err("a packed dense-only tensor must be refused");
        assert!(
            err.contains(prefix),
            "refusal for {prefix} must name the tensor, got: {err}"
        );
    }
}

/// Every LayerNorm on this family is loaded as a raw weight/bias pair and
/// handed to `fast::layer_norm`, so a packed one aborts across the cxx bridge
/// exactly like a packed conv weight. `nn.quantize` skips `nn.LayerNorm`, so
/// no published conversion produces these, but the guard has to cover them for
/// the same reason it covers the conv stack.
///
/// The two `layernorm_embedding` entries are the reason the guard matches on
/// the last dot-separated segment *containing* `norm` rather than on the stem
/// ending in it: that segment ends in `embedding`, so a suffix match would
/// have let both of them through.
#[test]
fn quantized_layer_norms_are_refused_by_name() {
    for prefix in [
        "image_proj_norm",
        "language_model.model.encoder.layernorm_embedding",
        "language_model.model.decoder.layernorm_embedding",
        "language_model.model.encoder.layers.0.self_attn_layer_norm",
        "language_model.model.decoder.layers.0.self_attn_layer_norm",
        "language_model.model.decoder.layers.0.encoder_attn_layer_norm",
        "language_model.model.decoder.layers.0.final_layer_norm",
        "vision_tower.convs.0.norm",
        "vision_tower.blocks.0.0.spatial_block.ffn.norm",
        "vision_tower.blocks.0.0.spatial_block.window_attn.norm",
        "vision_tower.blocks.0.0.channel_block.channel_attn.norm",
    ] {
        let mut map = WeightMap::new();
        quantized_triple(&mut map, prefix);
        let err = reject_unsupported_quantized_tensors(&map)
            .expect_err("a packed layer norm must be refused");
        assert!(
            err.contains(prefix),
            "refusal for {prefix} must name the tensor, got: {err}"
        );
    }
}

/// A fully dense checkpoint has no `.scales` anywhere, so the guard is inert.
/// The layer norms are here because they are the arm most likely to
/// false-positive: they stay dense in every real export, and matching them by
/// name must not reject the tensor that carries them.
#[test]
fn dense_checkpoints_pass_the_refusal_untouched() {
    let mut map = WeightMap::new();
    map.insert("image_projection".to_string(), dummy(&[16, 8]));
    map.insert(
        "language_model.model.shared.weight".to_string(),
        dummy(&[64, 16]),
    );
    map.insert(
        "vision_tower.convs.0.proj.weight".to_string(),
        dummy(&[8, 3, 3, 3]),
    );
    map.insert("image_proj_norm.weight".to_string(), dummy(&[16]));
    map.insert("image_proj_norm.bias".to_string(), dummy(&[16]));
    map.insert(
        "language_model.model.encoder.layers.0.self_attn_layer_norm.weight".to_string(),
        dummy(&[16]),
    );
    map.insert(
        "vision_tower.blocks.0.0.spatial_block.ffn.norm.weight".to_string(),
        dummy(&[16]),
    );
    reject_unsupported_quantized_tensors(&map).expect("dense map must be accepted");
}

// ---------------------------------------------------------------------------
// sanitize: shared-embedding fill
// ---------------------------------------------------------------------------

/// BART ties the encoder and decoder token tables to `model.shared`, and
/// exports vary in which of the three they materialize. When only
/// `embed_tokens` is present the fill has to copy all three planes: leaving
/// `.scales` and `.biases` behind would present `model.shared` as a dense
/// table whose `.weight` is packed `uint32`, which reaches MLX and aborts.
#[test]
fn sanitize_fills_shared_with_every_quantized_plane() {
    let mut map = WeightMap::new();
    quantized_triple(&mut map, "language_model.model.encoder.embed_tokens");
    let out = sanitize(map);

    for plane in ["weight", "scales", "biases"] {
        assert!(
            out.contains_key(&format!("language_model.model.shared.{plane}")),
            "sanitize must carry model.shared.{plane} across from embed_tokens"
        );
    }
    assert_eq!(
        mlxcel_core::array_dtype(out.get("language_model.model.shared.weight").unwrap()),
        mlxcel_core::dtype::UINT32,
        "the filled shared weight must still be the packed plane"
    );
}

/// The dense case must keep working: no `.scales` to carry, one plane copied.
#[test]
fn sanitize_fills_shared_from_a_dense_embed_tokens() {
    let mut map = WeightMap::new();
    map.insert(
        "language_model.model.decoder.embed_tokens.weight".to_string(),
        dummy(&[8, 4]),
    );
    let out = sanitize(map);

    assert!(out.contains_key("language_model.model.shared.weight"));
    assert!(!out.contains_key("language_model.model.shared.scales"));
    assert_eq!(
        mlxcel_core::array_shape(out.get("language_model.model.shared.weight").unwrap()),
        vec![8, 4]
    );
}

/// An export that already ships `model.shared` is left alone, packed or not.
#[test]
fn sanitize_leaves_an_existing_shared_table_alone() {
    let mut map = WeightMap::new();
    quantized_triple(&mut map, "language_model.model.shared");
    map.insert(
        "language_model.model.encoder.embed_tokens.weight".to_string(),
        dummy_u32(&[99, 2]),
    );
    let out = sanitize(map);
    assert_eq!(
        mlxcel_core::array_shape(out.get("language_model.model.shared.weight").unwrap()),
        vec![8, 2],
        "the checkpoint's own shared table must win over the embed_tokens fill"
    );
}
