// Copyright 2025-2026 Lablup Inc.
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

//! Key-normalization gates for the decoder-backbone embedding families.

use mlxcel_core::weights::WeightMap;

use super::{
    drop_derived_buffers, linear_features, sanitize_decoder_embedding_weights,
    strip_language_model_wrapper,
};

/// A zero tensor of the given shape, standing in for a real weight (only the
/// key and the shape matter here).
fn tensor(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let count: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![0.0; count as usize], shape)
}

fn map(entries: &[(&str, &[i32])]) -> WeightMap {
    entries
        .iter()
        .map(|(key, shape)| ((*key).to_string(), tensor(shape)))
        .collect()
}

fn sorted_keys(weights: &WeightMap) -> Vec<String> {
    let mut keys: Vec<String> = weights.keys().cloned().collect();
    keys.sort();
    keys
}

#[test]
fn dense_folder_keys_map_by_index() {
    // sentence-transformers layout: 2_Dense is the up projection (out > in)
    // and 3_Dense the down projection, ranked into dense.0 and dense.1.
    let mut weights = map(&[
        ("2_Dense.linear.weight", &[3072, 768]),
        ("3_Dense.linear.weight", &[768, 3072]),
        ("embed_tokens.weight", &[262144, 768]),
        ("layers.0.input_layernorm.weight", &[768]),
        ("norm.weight", &[768]),
        ("lm_head.weight", &[262144, 768]),
    ]);
    assert_eq!(sanitize_decoder_embedding_weights(&mut weights), 2);
    assert_eq!(
        sorted_keys(&weights),
        vec![
            "dense.0.weight",
            "dense.1.weight",
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.norm.weight",
        ]
    );
    assert_eq!(
        linear_features(&weights, "dense.0", 64),
        Some((3072, 768)),
        "dense.0 is the up projection"
    );
    assert_eq!(linear_features(&weights, "dense.1", 64), Some((768, 3072)));
}

#[test]
fn mlx_conversion_layout_is_left_alone() {
    // The mlx-community conversion already folds the projections into the
    // main shards and keeps the `model.` prefix, so sanitization is a no-op
    // on the key set.
    let mut weights = map(&[
        ("dense.0.weight", &[3072, 12]),
        ("dense.0.scales", &[3072, 12]),
        ("dense.1.weight", &[768, 48]),
        ("dense.1.scales", &[768, 48]),
        ("model.embed_tokens.weight", &[262144, 96]),
        ("model.norm.weight", &[768]),
    ]);
    let before = sorted_keys(&weights);
    assert_eq!(sanitize_decoder_embedding_weights(&mut weights), 0);
    assert_eq!(sorted_keys(&weights), before);
    // Packed 4-bit rows: the input width comes from the scales grouping.
    assert_eq!(linear_features(&weights, "dense.0", 64), Some((3072, 768)));
    assert_eq!(linear_features(&weights, "dense.1", 64), Some((768, 3072)));
}

#[test]
fn both_layouts_produce_the_same_dense_shapes() {
    let mut subfolder = map(&[
        ("2_Dense.linear.weight", &[3072, 768]),
        ("3_Dense.linear.weight", &[768, 3072]),
    ]);
    let mut folded = map(&[
        ("dense.0.weight", &[3072, 768]),
        ("dense.1.weight", &[768, 3072]),
    ]);
    sanitize_decoder_embedding_weights(&mut subfolder);
    sanitize_decoder_embedding_weights(&mut folded);
    assert_eq!(sorted_keys(&subfolder), sorted_keys(&folded));
    for prefix in ["dense.0", "dense.1"] {
        assert_eq!(
            linear_features(&subfolder, prefix, 64),
            linear_features(&folded, prefix, 64)
        );
    }
}

#[test]
fn prefixing_is_idempotent_and_skips_non_backbone_roots() {
    let mut weights = map(&[
        ("embed_tokens.weight", &[8, 4]),
        ("model.layers.0.mlp.up_proj.weight", &[8, 4]),
        ("dense.0.weight", &[8, 4]),
    ]);
    sanitize_decoder_embedding_weights(&mut weights);
    let once = sorted_keys(&weights);
    sanitize_decoder_embedding_weights(&mut weights);
    assert_eq!(sorted_keys(&weights), once);
    assert_eq!(
        once,
        vec![
            "dense.0.weight",
            "model.embed_tokens.weight",
            "model.layers.0.mlp.up_proj.weight",
        ]
    );
}

#[test]
fn unknown_dense_submodule_is_not_renamed() {
    let mut weights = map(&[("2_Dense.projection.weight", &[8, 4])]);
    assert_eq!(sanitize_decoder_embedding_weights(&mut weights), 0);
    assert_eq!(sorted_keys(&weights), vec!["2_Dense.projection.weight"]);
}

#[test]
fn embedding_norm_root_is_prefixed() {
    // LFM2 spells its final norm `embedding_norm.weight`; without this root
    // `Lfm2Model::from_weights` would fail looking for
    // `model.embedding_norm.weight`. A key already carrying the prefix is
    // untouched, so a second pass is a no-op.
    let mut weights = map(&[
        ("embedding_norm.weight", &[4]),
        ("layers.0.ffn_norm.weight", &[4]),
    ]);
    sanitize_decoder_embedding_weights(&mut weights);
    let once = sorted_keys(&weights);
    assert_eq!(
        once,
        vec![
            "model.embedding_norm.weight",
            "model.layers.0.ffn_norm.weight"
        ]
    );
    sanitize_decoder_embedding_weights(&mut weights);
    assert_eq!(sorted_keys(&weights), once);
}

#[test]
fn language_model_wrapper_is_stripped_exactly_once() {
    let mut weights = map(&[
        ("language_model.layers.0.mlp.up_proj.weight", &[8, 4]),
        ("language_model.model.norm.weight", &[4]),
        ("language_model.language_model.embed_tokens.weight", &[8, 4]),
        ("vision_model.language_model.weight", &[4]),
        ("mlp1.0.weight", &[4]),
    ]);
    strip_language_model_wrapper(&mut weights);
    assert_eq!(
        sorted_keys(&weights),
        vec![
            "language_model.embed_tokens.weight",
            "layers.0.mlp.up_proj.weight",
            "mlp1.0.weight",
            "model.norm.weight",
            "vision_model.language_model.weight",
        ],
        "only a leading wrapper goes, and only one level of it"
    );
}

#[test]
fn derived_buffers_are_dropped_at_any_depth() {
    let mut weights = map(&[
        ("model.layers.0.self_attn.rotary_emb.inv_freq", &[4]),
        ("rotary_emb.inv_freq", &[4]),
        ("position_ids", &[1, 4]),
        ("vision_model.embeddings.position_ids", &[1, 4]),
        ("vision_model.embeddings.position_embedding.weight", &[4, 4]),
        ("model.norm.weight", &[4]),
    ]);
    drop_derived_buffers(&mut weights);
    assert_eq!(
        sorted_keys(&weights),
        vec![
            "model.norm.weight",
            "vision_model.embeddings.position_embedding.weight",
        ]
    );
}
