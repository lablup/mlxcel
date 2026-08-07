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

//! Falcon-OCR synthetic parity checks.
//!
//! Builds a tiny decoder from a deterministic in-memory weight map (no
//! checkpoint) and exercises the three pieces that are unique to this family
//! and easiest to get subtly wrong:
//!
//! - the hybrid mask actually reaches SDPA, so image tokens see each other;
//! - the collapsed temporal positions plus the negative rope delta keep an
//!   incremental decode aligned with a single full-sequence pass;
//! - the per-head sinks are consumed and attenuate the attention output.
//!
//! Real-checkpoint numerics are validated end to end by OCR generation.

use mlxcel::models::falcon_ocr::{FalconOcrConfig, FalconOcrPrefillState, FalconOcrTextModel};
use mlxcel::models::falcon_ocr_rope::{
    build_hybrid_mask, rope_delta, spatial_positions, temporal_positions,
};
use mlxcel::vision::FalconOcrVlModel;
use mlxcel::vision::processors::falcon_ocr::FalconOcrProcessor;
use mlxcel_core::cache::SequenceStateBackend;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const DIM: i32 = 8;
const LAYERS: usize = 2;
const HEADS: i32 = 4;
const KV_HEADS: i32 = 2;
const HEAD_DIM: i32 = 4;
const VOCAB: i32 = 32;
const FFN: i32 = 6;
const PATCH: i32 = 2;
const CHANNELS: i32 = 3;

fn arr(shape: &[i32], seed: i32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (((i * 7 + seed) % 13) as f32 / 13.0 - 0.5) * 0.3)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn config() -> FalconOcrConfig {
    serde_json::from_str(&format!(
        r#"{{
          "model_type": "falcon_ocr",
          "dim": {DIM}, "n_layers": {LAYERS}, "n_heads": {HEADS},
          "head_dim": {HEAD_DIM}, "n_kv_heads": {KV_HEADS},
          "vocab_size": {VOCAB}, "ffn_dim": {FFN}, "norm_eps": 1e-05,
          "max_seq_len": 512, "rope_theta": 10000,
          "channel_size": {CHANNELS}, "spatial_patch_size": {PATCH},
          "temporal_patch_size": 1,
          "eos_id": 11, "img_id": 27, "img_end_id": 30,
          "image_cls_token_id": 24,
          "image_reg_1_token_id": 25, "image_reg_2_token_id": 26,
          "image_reg_3_token_id": 28, "image_reg_4_token_id": 29
        }}"#
    ))
    .expect("synthetic config parses")
}

fn weights(sinks: f32) -> WeightMap {
    let mut w = WeightMap::new();
    w.insert("tok_embeddings.weight".into(), arr(&[VOCAB, DIM], 1));
    w.insert(
        "img_projector.weight".into(),
        arr(&[DIM, PATCH * PATCH * CHANNELS], 2),
    );
    w.insert(
        "norm.weight".into(),
        mlxcel_core::ones(&[DIM], mlxcel_core::dtype::FLOAT32),
    );
    w.insert("output.weight".into(), arr(&[VOCAB, DIM], 3));
    // freqs_cis_golden is [n_heads, head_dim / 4, 2].
    w.insert("freqs_cis_golden".into(), arr(&[HEADS, HEAD_DIM / 4, 2], 4));

    let qkv_out = HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM;
    for layer in 0..LAYERS {
        let seed = 10 + layer as i32 * 5;
        w.insert(
            format!("layers.{layer}.attention.wqkv.weight"),
            arr(&[qkv_out, DIM], seed),
        );
        w.insert(
            format!("layers.{layer}.attention.wo.weight"),
            arr(&[DIM, HEADS * HEAD_DIM], seed + 1),
        );
        w.insert(
            format!("layers.{layer}.attention.sinks"),
            mlxcel_core::from_slice_f32(&vec![sinks; HEADS as usize], &[HEADS]),
        );
        w.insert(
            format!("layers.{layer}.feed_forward.w1.weight"),
            arr(&[FFN, DIM], seed + 2),
        );
        w.insert(
            format!("layers.{layer}.feed_forward.w3.weight"),
            arr(&[FFN, DIM], seed + 3),
        );
        w.insert(
            format!("layers.{layer}.feed_forward.w2.weight"),
            arr(&[DIM, FFN], seed + 4),
        );
    }
    w
}

fn model(sinks: f32) -> FalconOcrTextModel {
    FalconOcrTextModel::from_weights(&weights(sinks), &config()).expect("synthetic model builds")
}

/// `[cls, reg1..4, patch * 4, end, text, text]` for one 2x2-patch image.
fn image_prompt() -> Vec<i32> {
    vec![24, 25, 26, 28, 29, 27, 27, 27, 27, 30, 5, 6]
}

fn ids_array(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

/// Logits at one position of a `[1, L, V]` tensor.
fn logits_row(logits: &MlxArray, pos: usize) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(logits);
    let vocab = shape[2];
    let row = mlxcel_core::slice(logits, &[0, pos as i32, 0], &[1, pos as i32 + 1, vocab]);
    read(&row)
}

fn read(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::array_evaluated_bytes(a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn install_state(m: &FalconOcrTextModel, tokens: &[i32], grids: &[(i32, i32)]) {
    let ids = m.config.token_ids();
    let positions = temporal_positions(tokens, &ids);
    let delta = rope_delta(&positions);
    let pos_hw = spatial_positions(tokens, &ids, grids);
    m.state.set_current(FalconOcrPrefillState {
        positions,
        pos_hw: Some(mlxcel_core::from_slice_f32(
            &pos_hw,
            &[1, tokens.len() as i32, 2],
        )),
        rope_delta: delta,
    });
}

/// Last-position logits of one full-sequence forward over `tokens`.
fn full_pass(m: &FalconOcrTextModel, tokens: &[i32], grids: &[(i32, i32)]) -> Vec<f32> {
    install_state(m, tokens, grids);
    let mut caches = m.make_caches();
    let mask = build_hybrid_mask(tokens, &m.config.token_ids());
    let logits = m.forward_with_embeddings(
        &ids_array(tokens),
        None,
        &mut caches,
        Some(mask.as_ref().unwrap()),
    );
    let last = mlxcel_core::slice_last_logits(&logits);
    read(&last)
}

/// The bidirectional image block must change the answer. If the mask never
/// reached SDPA (or was silently replaced by the runtime's causal one), these
/// two runs would be identical.
#[test]
fn the_hybrid_mask_differs_from_a_plain_causal_prefill() {
    let m = model(0.0);
    let tokens = image_prompt();
    let grids = [(2, 2)];

    let hybrid = full_pass(&m, &tokens, &grids);

    install_state(&m, &tokens, &grids);
    let mut caches = m.make_caches();
    let causal = mlxcel_core::utils::create_causal_mask(tokens.len() as i32, 0);
    let logits = m.forward_with_embeddings(
        &ids_array(&tokens),
        None,
        &mut caches,
        Some(causal.as_ref().unwrap()),
    );
    let causal = read(&mlxcel_core::slice_last_logits(&logits));

    let max_delta = hybrid
        .iter()
        .zip(causal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-4,
        "hybrid and causal prefills agreed (max delta {max_delta}); the image block was not bidirectional"
    );
}

/// Prefill then decode must land on the same logits as a single pass over the
/// concatenated sequence. This is the check that catches a wrong rope delta:
/// the image block collapses ten sequence slots onto one temporal index, so a
/// decode step that used `cache_offset` directly would be off by nine.
#[test]
fn incremental_decode_matches_a_full_sequence_pass() {
    let m = model(0.25);
    let prompt = image_prompt();
    let grids = [(2, 2)];
    let generated = [7i32, 9, 3];

    // Reference: one pass over prompt ++ generated.
    let mut whole = prompt.clone();
    whole.extend_from_slice(&generated);
    let reference = full_pass(&m, &whole, &grids);

    // Incremental: prefill the prompt, then feed the generated tokens one by
    // one with no mask, the way the decode loop does.
    install_state(&m, &prompt, &grids);
    let mut caches = m.make_caches();
    let mask = build_hybrid_mask(&prompt, &m.config.token_ids());
    let _ = m.forward_with_embeddings(
        &ids_array(&prompt),
        None,
        &mut caches,
        Some(mask.as_ref().unwrap()),
    );
    let mut step_logits = None;
    for &token in &generated {
        let logits = m.forward(
            &mlxcel_core::from_slice_i32(&[token], &[1, 1]),
            &mut caches,
            None,
        );
        step_logits = Some(read(&mlxcel_core::slice_last_logits(&logits)));
    }
    let incremental = step_logits.expect("at least one decode step");

    assert_eq!(reference.len(), incremental.len());
    let max_delta = reference
        .iter()
        .zip(incremental.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = reference.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
    assert!(
        max_delta <= 2e-3 * scale,
        "incremental decode drifted from the full pass: max delta {max_delta} (scale {scale})\nfull: {reference:?}\nincremental: {incremental:?}"
    );
}

/// A sink is an extra logit in the softmax denominator, so raising it must
/// shrink the attention output and therefore move the logits. A sink tensor
/// that was loaded but never passed to SDPA would leave them untouched.
#[test]
fn the_per_head_sinks_change_the_attention_output() {
    let tokens = image_prompt();
    let grids = [(2, 2)];
    let neutral = full_pass(&model(-30.0), &tokens, &grids);
    let strong = full_pass(&model(4.0), &tokens, &grids);

    let max_delta = neutral
        .iter()
        .zip(strong.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-4,
        "sinks had no effect (max delta {max_delta}); the tensor is loaded but unused"
    );
}

/// A text-only prompt has no stashed state and no supplied mask, so the model
/// has to build its own causal mask rather than attending to the future.
#[test]
fn a_text_only_prefill_is_causal_without_being_handed_a_mask() {
    let m = model(0.0);
    let prefix = [5i32, 6, 7];
    let mut extended = prefix.to_vec();
    extended.push(9);

    let mut caches = m.make_caches();
    let logits = m.forward(&ids_array(&prefix), &mut caches, None);
    let short = read(&mlxcel_core::slice_last_logits(&logits));

    let m2 = model(0.0);
    let mut caches2 = m2.make_caches();
    let logits2 = m2.forward(&ids_array(&extended), &mut caches2, None);
    let long = logits_row(&logits2, prefix.len() - 1);

    let max_delta = short
        .iter()
        .zip(long.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta < 1e-4,
        "appending a token changed an earlier position's logits (max delta {max_delta}); the prefill leaked future context"
    );
}

/// The server allocates a sequence's KV set from `sequence_state_layout()`, and
/// the trait default infers a *model-owned* placeholder (an empty cache slice)
/// for any model that declines batching. Falcon-OCR declines batching because
/// its positional state is single-row, not because it owns its KV, so it must
/// declare the dense layout explicitly. Without this the scheduler hands
/// `forward` zero caches, `layers.iter().zip(caches)` runs zero times, and the
/// LM head reads the raw token embedding: the whole 22-layer decoder is skipped
/// with no error anywhere.
#[test]
fn the_declared_sequence_state_layout_is_a_dense_cache_per_layer() {
    let m = model(0.0);
    let layout = m.sequence_state_layout();
    assert_eq!(layout.backend, SequenceStateBackend::DenseKvCache);
    assert_eq!(layout.num_layers, LAYERS);
    assert_eq!(layout.num_layers, m.make_caches().len());

    // The loaded model the scheduler actually asks is the VLM wrapper, so the
    // delegation has to be there too.
    let vlm = FalconOcrVlModel {
        text_model: model(0.0),
        processor: FalconOcrProcessor::new(PATCH as u32, CHANNELS as usize),
        ocr_task_token_id: None,
        eos_token_ids: vec![11],
    };
    let vlm_layout = LanguageModel::sequence_state_layout(&vlm);
    assert_eq!(vlm_layout.backend, SequenceStateBackend::DenseKvCache);
    assert_eq!(vlm_layout.num_layers, LAYERS);
}

/// A forward pass that was handed no caches must not silently return logits
/// from the embedding table. This is the shape the empty-slice bug took: no
/// panic, no error, just a decoder that never ran.
#[test]
fn a_forward_pass_actually_runs_every_decoder_layer() {
    let m = model(0.0);
    let tokens = image_prompt();
    let grids = [(2, 2)];
    let through_layers = full_pass(&m, &tokens, &grids);

    install_state(&m, &tokens, &grids);
    let mut no_caches: Vec<mlxcel_core::layers::KVCache> = Vec::new();
    let mask = build_hybrid_mask(&tokens, &m.config.token_ids());
    let skipped = m.forward_with_embeddings(
        &ids_array(&tokens),
        None,
        &mut no_caches,
        Some(mask.as_ref().unwrap()),
    );
    let skipped = read(&mlxcel_core::slice_last_logits(&skipped));

    let max_delta = through_layers
        .iter()
        .zip(skipped.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-4,
        "an empty cache slice produced the same logits as a real forward (max delta {max_delta}); the layer loop is not observable"
    );
}
