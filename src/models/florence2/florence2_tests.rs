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

//! Unit tests for the Florence-2 seq2seq engine: config parsing,
//! `shift_tokens_right`, and dual-KV-cache decode correctness on a tiny
//! synthetic model (incremental decode must match teacher-forced full-seq
//! decode, cross-attention K/V must be computed once and reused).

use super::*;
use serde_json::json;

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

/// The shape of the real Florence-2-base config.json: text fields nested
/// under `text_config`, with 12 attention heads on both sides.
fn real_shaped_config() -> Value {
    json!({
        "model_type": "florence2",
        "is_encoder_decoder": true,
        "vocab_size": 51289,
        "text_config": {
            "model_type": "florence2_language",
            "d_model": 768,
            "encoder_layers": 6,
            "decoder_layers": 6,
            "encoder_attention_heads": 12,
            "decoder_attention_heads": 12,
            "encoder_ffn_dim": 3072,
            "decoder_ffn_dim": 3072,
            "vocab_size": 51289,
            "max_position_embeddings": 1024,
            "scale_embedding": false,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "decoder_start_token_id": 2
        },
        "vision_config": { "model_type": "" }
    })
}

#[test]
fn text_config_parses_nested_real_shape() {
    let cfg = Florence2TextConfig::from_model_config(&real_shaped_config()).unwrap();
    assert_eq!(cfg.d_model, 768);
    assert_eq!(cfg.encoder_layers, 6);
    assert_eq!(cfg.decoder_layers, 6);
    // The real checkpoint has 12 heads on both encoder and decoder.
    assert_eq!(cfg.encoder_attention_heads, 12);
    assert_eq!(cfg.decoder_attention_heads, 12);
    assert_eq!(cfg.encoder_ffn_dim, 3072);
    assert_eq!(cfg.decoder_ffn_dim, 3072);
    assert_eq!(cfg.vocab_size, 51289);
    assert_eq!(cfg.max_position_embeddings, 1024);
    assert!(!cfg.scale_embedding);
    assert_eq!(cfg.pad_token_id, 1);
    assert_eq!(cfg.bos_token_id, 0);
    assert_eq!(cfg.eos_token_id, 2);
    assert_eq!(cfg.decoder_start_token_id, 2);
}

#[test]
fn text_config_parses_bare_text_object() {
    let cfg =
        Florence2TextConfig::from_model_config(real_shaped_config().get("text_config").unwrap())
            .unwrap();
    assert_eq!(cfg.d_model, 768);
    assert_eq!(cfg.decoder_attention_heads, 12);
}

#[test]
fn text_config_missing_required_field_errors() {
    let cfg = json!({ "text_config": { "d_model": 768 } });
    assert!(Florence2TextConfig::from_model_config(&cfg).is_err());
}

#[test]
fn text_config_rejects_indivisible_heads() {
    let cfg = json!({
        "d_model": 10, "encoder_layers": 1, "decoder_layers": 1,
        "encoder_attention_heads": 3, "decoder_attention_heads": 2,
        "vocab_size": 16
    });
    assert!(Florence2TextConfig::from_model_config(&cfg).is_err());
}

// ---------------------------------------------------------------------------
// shift_tokens_right
// ---------------------------------------------------------------------------

#[test]
fn shift_tokens_right_prepends_decoder_start() {
    assert_eq!(shift_tokens_right(&[0, 7, 8, 2], 1, 2), vec![2, 0, 7, 8]);
}

#[test]
fn shift_tokens_right_replaces_label_padding() {
    assert_eq!(shift_tokens_right(&[0, 7, -100, 2], 1, 2), vec![2, 0, 7, 1]);
}

#[test]
fn shift_tokens_right_handles_empty_and_single() {
    assert_eq!(shift_tokens_right(&[], 1, 2), vec![2]);
    assert_eq!(shift_tokens_right(&[5], 1, 2), vec![2]);
}

// ---------------------------------------------------------------------------
// Synthetic tiny model
// ---------------------------------------------------------------------------

const D_MODEL: i32 = 8;
const HEADS: i32 = 2;
const LAYERS: i32 = 2;
const FFN: i32 = 16;
const VOCAB: i32 = 16;
const MAX_POS: i32 = 32;

fn tiny_config() -> Florence2TextConfig {
    Florence2TextConfig {
        d_model: D_MODEL,
        encoder_layers: LAYERS,
        decoder_layers: LAYERS,
        encoder_attention_heads: HEADS,
        decoder_attention_heads: HEADS,
        encoder_ffn_dim: FFN,
        decoder_ffn_dim: FFN,
        vocab_size: VOCAB,
        max_position_embeddings: MAX_POS,
        scale_embedding: false,
        pad_token_id: 1,
        bos_token_id: 0,
        eos_token_id: 2,
        decoder_start_token_id: 2,
        quantization: Florence2Quantization::DENSE,
    }
}

/// Deterministic pseudo-random tensor: small values so the f32 graph stays
/// numerically tame across the post-norm stack.
fn synth(map: &mut WeightMap, key: &str, shape: &[i32]) {
    let n: i32 = shape.iter().product();
    let seed: f32 = key.bytes().map(|b| b as f32).sum();
    let data: Vec<f32> = (0..n)
        .map(|i| 0.1 * ((seed + 0.7 * i as f32).sin()))
        .collect();
    map.insert(key.to_string(), mlxcel_core::from_slice_f32(&data, shape));
}

/// LayerNorm weights near identity so activations keep unit scale.
fn synth_ln(map: &mut WeightMap, prefix: &str) {
    let ones = vec![1.0f32; D_MODEL as usize];
    let zeros = vec![0.0f32; D_MODEL as usize];
    map.insert(
        format!("{prefix}.weight"),
        mlxcel_core::from_slice_f32(&ones, &[D_MODEL]),
    );
    map.insert(
        format!("{prefix}.bias"),
        mlxcel_core::from_slice_f32(&zeros, &[D_MODEL]),
    );
}

fn synth_attn(map: &mut WeightMap, prefix: &str) {
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        synth(map, &format!("{prefix}.{proj}.weight"), &[D_MODEL, D_MODEL]);
        synth(map, &format!("{prefix}.{proj}.bias"), &[D_MODEL]);
    }
}

fn tiny_weights() -> WeightMap {
    let mut map = WeightMap::new();
    synth(&mut map, "model.shared.weight", &[VOCAB, D_MODEL]);

    for side in ["encoder", "decoder"] {
        synth(
            &mut map,
            &format!("model.{side}.embed_positions.weight"),
            &[MAX_POS + 2, D_MODEL],
        );
        synth_ln(&mut map, &format!("model.{side}.layernorm_embedding"));
        for i in 0..LAYERS {
            let p = format!("model.{side}.layers.{i}");
            synth_attn(&mut map, &format!("{p}.self_attn"));
            synth_ln(&mut map, &format!("{p}.self_attn_layer_norm"));
            if side == "decoder" {
                synth_attn(&mut map, &format!("{p}.encoder_attn"));
                synth_ln(&mut map, &format!("{p}.encoder_attn_layer_norm"));
            }
            synth(&mut map, &format!("{p}.fc1.weight"), &[FFN, D_MODEL]);
            synth(&mut map, &format!("{p}.fc1.bias"), &[FFN]);
            synth(&mut map, &format!("{p}.fc2.weight"), &[D_MODEL, FFN]);
            synth(&mut map, &format!("{p}.fc2.bias"), &[D_MODEL]);
            synth_ln(&mut map, &format!("{p}.final_layer_norm"));
        }
    }
    // No lm_head.weight: exercises the tied-embedding fallback.
    map
}

fn tiny_model() -> Florence2TextModel {
    Florence2TextModel::from_weights(&tiny_weights(), tiny_config(), "").unwrap()
}

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Encoder behavior
// ---------------------------------------------------------------------------

#[test]
fn encoder_is_bidirectional() {
    // Changing the LAST input token must change the FIRST position's encoder
    // state; a causal encoder could not do that.
    let model = tiny_model();
    let a = mlxcel_core::from_slice_i32(&[0, 5, 6, 7], &[1, 4]);
    let b = mlxcel_core::from_slice_i32(&[0, 5, 6, 9], &[1, 4]);
    let ha = to_vec_f32(&model.encode_tokens(&a));
    let hb = to_vec_f32(&model.encode_tokens(&b));
    let d = D_MODEL as usize;
    let first_pos_diff: f32 = ha[..d]
        .iter()
        .zip(&hb[..d])
        .map(|(x, y)| (x - y).abs())
        .sum();
    assert!(
        first_pos_diff > 1e-6,
        "bidirectional encoder: position 0 must see the changed last token, diff {first_pos_diff}"
    );
}

#[test]
fn encode_embeds_matches_encode_tokens() {
    let model = tiny_model();
    let ids = mlxcel_core::from_slice_i32(&[0, 5, 6, 2], &[1, 4]);
    let via_tokens = to_vec_f32(&model.encode_tokens(&ids));
    let embeds = model.embed_tokens(&ids);
    let via_embeds = to_vec_f32(&model.encode_embeds(&embeds));
    assert_eq!(via_tokens, via_embeds);
}

// ---------------------------------------------------------------------------
// Dual KV cache decode
// ---------------------------------------------------------------------------

/// Incremental decode (one token per step, growing the self-attention cache
/// and reusing the one-shot cross-attention cache) must produce the same
/// logits as a single teacher-forced full-sequence pass.
#[test]
fn incremental_decode_matches_full_sequence() {
    let model = tiny_model();
    let src = mlxcel_core::from_slice_i32(&[0, 5, 6, 7, 2], &[1, 5]);
    let enc = model.encode_tokens(&src);

    let dec_ids = [2i32, 0, 5, 9, 3];

    // Teacher-forced full-sequence pass.
    let full_in = mlxcel_core::from_slice_i32(&dec_ids, &[1, dec_ids.len() as i32]);
    let mut full_cache = model.make_cache();
    let full_logits = to_vec_f32(&model.decode(&full_in, &enc, &mut full_cache));
    assert_eq!(full_cache.offset(), dec_ids.len() as i32);

    // Incremental pass, one token at a time.
    let mut cache = model.make_cache();
    let mut inc_logits = Vec::new();
    for (step, &tok) in dec_ids.iter().enumerate() {
        let t = mlxcel_core::from_slice_i32(&[tok], &[1, 1]);
        assert_eq!(cache.offset(), step as i32);
        inc_logits.extend(to_vec_f32(&model.decode(&t, &enc, &mut cache)));
    }

    assert_eq!(full_logits.len(), inc_logits.len());
    let max_diff = full_logits
        .iter()
        .zip(&inc_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "incremental decode diverged from full-sequence decode, max abs diff {max_diff}"
    );
}

/// A multi-token prefill followed by incremental steps (the production
/// decode shape) must also match the full-sequence pass, exercising the
/// offset-aware causal mask.
#[test]
fn prefill_then_steps_matches_full_sequence() {
    let model = tiny_model();
    let src = mlxcel_core::from_slice_i32(&[0, 5, 6, 2], &[1, 4]);
    let enc = model.encode_tokens(&src);

    let dec_ids = [2i32, 0, 5, 9, 3, 4];

    let full_in = mlxcel_core::from_slice_i32(&dec_ids, &[1, dec_ids.len() as i32]);
    let mut full_cache = model.make_cache();
    let full_logits = to_vec_f32(&model.decode(&full_in, &enc, &mut full_cache));

    // Prefill the first 3 tokens in one call, then step one by one. The
    // multi-token continuation would need an offset causal mask; production
    // only ever continues one token at a time, which is what we pin here.
    let mut cache = model.make_cache();
    let prefill = mlxcel_core::from_slice_i32(&dec_ids[..3], &[1, 3]);
    let mut got = to_vec_f32(&model.decode(&prefill, &enc, &mut cache));
    for &tok in &dec_ids[3..] {
        let t = mlxcel_core::from_slice_i32(&[tok], &[1, 1]);
        got.extend(to_vec_f32(&model.decode(&t, &enc, &mut cache)));
    }
    assert_eq!(cache.offset(), dec_ids.len() as i32);

    let max_diff = full_logits
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "prefill+steps decode diverged from full-sequence decode, max abs diff {max_diff}"
    );
}

/// The cross-attention K/V must be projected exactly once from the encoder
/// output and then reused: after the first decode call the cached cross K
/// length equals the encoder length and stays constant, while the
/// self-attention cache grows per step.
#[test]
fn cross_cache_is_one_shot_and_self_cache_grows() {
    let model = tiny_model();
    let enc_len = 5;
    let src = mlxcel_core::from_slice_i32(&[0, 5, 6, 7, 2], &[1, enc_len]);
    let enc = model.encode_tokens(&src);

    let mut cache = model.make_cache();
    for step in 0..3 {
        let t = mlxcel_core::from_slice_i32(&[2 + step], &[1, 1]);
        let _ = model.decode(&t, &enc, &mut cache);
        for layer in &cache.layers {
            let self_kv = layer.self_kv.as_ref().expect("self cache populated");
            let cross_kv = layer.cross_kv.as_ref().expect("cross cache populated");
            assert_eq!(
                mlxcel_core::array_shape(&self_kv.k)[1],
                step + 1,
                "self-attention cache must grow one entry per decode step"
            );
            assert_eq!(
                mlxcel_core::array_shape(&cross_kv.k)[1],
                enc_len,
                "cross-attention cache must stay pinned to the encoder length"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Greedy round trip
// ---------------------------------------------------------------------------

#[test]
fn generate_greedy_rejects_empty_and_overlong_prompts() {
    let model = tiny_model();
    assert!(model.generate_greedy(&[], 4).is_err());
    let overlong = vec![0i32; (MAX_POS + 1) as usize];
    assert!(model.generate_greedy(&overlong, 4).is_err());
}

#[test]
fn generate_greedy_round_trip_produces_valid_tokens() {
    let model = tiny_model();
    let out = model.generate_greedy(&[0, 5, 6, 2], 8).unwrap();
    assert!(out.len() <= 8);
    for &tok in &out {
        assert!((0..VOCAB).contains(&tok), "token {tok} outside vocab");
        assert_ne!(tok, model.config().eos_token_id);
    }
    // Deterministic: the same prompt greedy-decodes to the same ids.
    let again = model.generate_greedy(&[0, 5, 6, 2], 8).unwrap();
    assert_eq!(out, again);
}
