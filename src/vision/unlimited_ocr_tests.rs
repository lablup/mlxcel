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

//! Capability-decline pins for `UnlimitedOcrVlModel`.
//!
//! The ring sliding decode cache is model-owned (kept in
//! `ModelOwnedSequenceState`, not the external `KVCache` slice), so the
//! wrapper must decline every `LanguageModel` capability that assumes an
//! externally managed, per-request KV cache. This mirrors how other
//! model-owned families pin their declines (e.g. the recurrent/hybrid-SSM
//! `cache_pool_model_owned_recurrent_sequence_is_never_detachable` test in
//! `mlxcel-core::cache::detach_tests`), applied here against the real
//! production wrapper instead of a synthetic stand-in.
//!
//! `super` is the `unlimited_ocr` module that includes this file via
//! `#[path]`, so `UnlimitedOcrVlModel` is reachable directly.

use super::UnlimitedOcrVlModel;
use crate::models::deepseek::{DeepSeekModel, ModelArgs};
use crate::vision::deepseekocr::DeepSeekOcrVlModel;
use crate::vision::encoders::deepseekocr_clip::{ClipConfig, ClipEncoder};
use crate::vision::encoders::deepseekocr_sam::{SamConfig, SamEncoder};
use crate::vision::processors::deepseekocr::DeepSeekOcrProcessor;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;

/// Build a minimal (0-layer text decoder, 0-block vision towers)
/// `UnlimitedOcrVlModel` fixture. Every tensor is a placeholder shape: this
/// fixture only exists to construct a real wrapper instance so the
/// `LanguageModel` capability declines below are pinned against the actual
/// production `impl`, not a hand-written stand-in; none of these tests run a
/// forward pass.
fn build_wrapper() -> UnlimitedOcrVlModel {
    let mut weights = WeightMap::new();

    // SAM ViT-B tower: depth 0 skips block loading, so only the patch embed /
    // neck / compressor tensors are needed.
    let sam_prefix = "sam_model";
    weights.insert(
        format!("{sam_prefix}.patch_embed.proj.weight"),
        mlxcel_core::from_slice_f32(&[0.0], &[1, 1, 1, 1]),
    );
    weights.insert(
        format!("{sam_prefix}.patch_embed.proj.bias"),
        mlxcel_core::from_slice_f32(&[0.0], &[1]),
    );
    weights.insert(
        format!("{sam_prefix}.pos_embed"),
        mlxcel_core::from_slice_f32(&[0.0], &[1, 1, 1, 1]),
    );
    weights.insert(
        format!("{sam_prefix}.neck.0.weight"),
        mlxcel_core::from_slice_f32(&[0.0], &[1, 1, 1, 1]),
    );
    weights.insert(
        format!("{sam_prefix}.neck.1.weight"),
        mlxcel_core::from_slice_f32(&[1.0], &[1]),
    );
    weights.insert(
        format!("{sam_prefix}.neck.2.weight"),
        mlxcel_core::from_slice_f32(&[0.0], &[1, 1, 1, 1]),
    );
    weights.insert(
        format!("{sam_prefix}.neck.3.weight"),
        mlxcel_core::from_slice_f32(&[1.0], &[1]),
    );
    weights.insert(
        format!("{sam_prefix}.net_2.weight"),
        mlxcel_core::from_slice_f32(&[0.0], &[1, 1, 1, 1]),
    );
    weights.insert(
        format!("{sam_prefix}.net_3.weight"),
        mlxcel_core::from_slice_f32(&[0.0], &[1, 1, 1, 1]),
    );
    let sam_config = SamConfig {
        embed_dim: 1,
        num_heads: 1,
        depth: 0,
        window_size: 0,
        global_attn_indexes: vec![],
        out_chans: 1,
        final_out_chans: 1,
        grid: 1,
    };
    let sam = SamEncoder::from_weights(&weights, sam_prefix, sam_config)
        .expect("minimal 0-block SAM fixture should load");

    // CLIP-L tower: num_layers 0 skips transformer-layer loading.
    let clip_prefix = "vision_model";
    weights.insert(
        format!("{clip_prefix}.embeddings.class_embedding"),
        mlxcel_core::from_slice_f32(&[0.0], &[1]),
    );
    weights.insert(
        format!("{clip_prefix}.embeddings.position_embedding.weight"),
        mlxcel_core::from_slice_f32(&[0.0], &[1, 1]),
    );
    weights.insert(
        format!("{clip_prefix}.pre_layrnorm.weight"),
        mlxcel_core::from_slice_f32(&[1.0], &[1]),
    );
    weights.insert(
        format!("{clip_prefix}.pre_layrnorm.bias"),
        mlxcel_core::from_slice_f32(&[0.0], &[1]),
    );
    let clip_config = ClipConfig {
        hidden_size: 1,
        num_heads: 1,
        num_layers: 0,
        layer_norm_eps: 1e-6,
        pos_grid: 1,
    };
    let clip = ClipEncoder::from_weights(&weights, clip_prefix, clip_config)
        .expect("minimal 0-layer CLIP fixture should load");

    // Linear projector (non-quantized: no `.scales` key present).
    weights.insert(
        "projector.layers.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0], &[1, 1]),
    );
    let projector = UnifiedLinear::from_weights(&weights, "projector.layers", 64, 4)
        .expect("minimal projector fixture should load");

    let image_newline = mlxcel_core::from_slice_f32(&[0.0], &[1]);
    let view_separator = mlxcel_core::from_slice_f32(&[0.0], &[1]);

    // 0-layer DeepSeek text decoder: only the embedding, final norm, and LM
    // head are needed since `num_hidden_layers: 0` skips per-layer loading.
    let mut text_weights = WeightMap::new();
    text_weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 4], &[2, 2]),
    );
    text_weights.insert(
        "model.norm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0, 1.0], &[2]),
    );
    text_weights.insert(
        "lm_head.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 4], &[2, 2]),
    );
    let args = ModelArgs {
        model_type: "deepseek".to_string(),
        vocab_size: 2,
        hidden_size: 2,
        intermediate_size: 2,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        max_position_embeddings: 16,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        moe_intermediate_size: None,
        n_shared_experts: None,
        n_routed_experts: None,
        num_experts_per_tok: None,
        moe_layer_freq: 1,
        first_k_dense_replace: 0,
        routed_scaling_factor: 1.0,
        attention_bias: false,
        group_size: None,
        bits: None,
    };
    let text_model = DeepSeekModel::from_weights(&text_weights, &args)
        .expect("minimal 0-layer DeepSeek text fixture should load");

    let inner = DeepSeekOcrVlModel {
        text_model,
        sam,
        clip,
        projector,
        image_newline,
        view_separator,
        processor: DeepSeekOcrProcessor::default(),
        image_token_id: 999,
        eos_token_id: 1,
        n_embed: 1,
    };

    UnlimitedOcrVlModel::new(inner, 128)
}

/// The ring cache infers the prefill -> decode boundary from the first
/// single-token forward (a192a81); a chunked prefill whose final chunk is one
/// token would misclassify that prompt token as the first decode token and
/// let the ring evict it. Pin the decline the same way other model-owned
/// families would (a real wrapper instance, not a hand-rolled stand-in).
#[test]
fn unlimited_ocr_wrapper_declines_chunked_prefill() {
    let wrapper = build_wrapper();
    assert!(
        !wrapper.supports_chunked_prefill(),
        "ring cache requires single-pass prefill to keep its boundary detection correct"
    );
}

/// Ring caches record the prefill boundary from the physical prefill length;
/// appending padding tokens would corrupt that boundary.
#[test]
fn unlimited_ocr_wrapper_declines_padded_prefill() {
    let wrapper = build_wrapper();
    assert!(!wrapper.supports_padded_prefill());
}

/// Ring caches are model-owned, not per-sequence `KVCache` slices, so
/// continuous batching (which relies on external per-sequence cache
/// isolation) is not available.
#[test]
fn unlimited_ocr_wrapper_declines_batching() {
    let wrapper = build_wrapper();
    assert!(!wrapper.supports_batching());
}
