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

//! Unit tests for the EAR-TTS building blocks, on tiny random weights.
//! Real-checkpoint parity lives in `tests/nemotron_voicechat_tts_real.rs`.

use std::collections::HashMap;

use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::{CharEncoderConfig, MogConfig, TtsConfig};
use super::mog_head::{MogHead, top_p_logits};
use super::norm_mlp::OffsetRmsNorm;
use super::rvq::RvqCodebooks;
use super::subword::CharAwareSubwordEncoder;
use super::{SpeechDecoderAssets, TtsPrompt};

fn rand(key: &mut u64, shape: &[i32], scale: f32) -> UniquePtr<MlxArray> {
    *key += 1;
    let k = mlxcel_core::random_key(*key);
    let x = unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, &*k) };
    mlxcel_core::multiply_scalar(&x, scale)
}

fn to_vec(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

fn to_i32(a: &MlxArray) -> Vec<i32> {
    super::subword::to_host_i32(a)
}

#[test]
fn offset_rms_norm_equals_rms_norm_with_weight_plus_one() {
    let mut key = 1u64;
    let x = rand(&mut key, &[2, 3, 16], 2.0);
    let w = rand(&mut key, &[16], 0.2);
    let offset = OffsetRmsNorm::new(&w, 1e-6).forward(&x);
    let plus_one = mlxcel_core::add(&mlxcel_core::full_f32(&[], 1.0, dtype::FLOAT32), &w);
    let plain = mlxcel_core::fast_rms_norm(&x, &plus_one, 1e-6);
    assert_eq!(to_vec(&offset), to_vec(&plain));

    // bf16 input: normalized in f32, returned in bf16.
    let xb = mlxcel_core::astype(&x, dtype::BFLOAT16);
    let out = OffsetRmsNorm::new(&w, 1e-6).forward(&xb);
    assert_eq!(mlxcel_core::array_dtype(&out), dtype::BFLOAT16);
    let want = mlxcel_core::astype(
        &mlxcel_core::fast_rms_norm(&mlxcel_core::astype(&xb, dtype::FLOAT32), &plus_one, 1e-6),
        dtype::BFLOAT16,
    );
    assert_eq!(to_vec(&out), to_vec(&want));
}

#[test]
fn top_p_keeps_at_least_the_largest_logit() {
    let logits = mlxcel_core::from_slice_f32(&[0.1, 5.0, -1.0, 0.3, 4.0, -2.0], &[1, 1, 6]);
    let tiny = to_vec(&top_p_logits(&logits, 1e-6).unwrap());
    assert_eq!(tiny[1], 5.0);
    assert_eq!(tiny.iter().filter(|v| v.is_finite()).count(), 1);

    // 0.9 needs the two dominant logits (5.0 and 4.0 hold ~0.97 together).
    let two = to_vec(&top_p_logits(&logits, 0.9).unwrap());
    let kept: Vec<usize> = (0..6).filter(|&i| two[i].is_finite()).collect();
    assert_eq!(kept, vec![1, 4]);
    assert_eq!(two[4], 4.0);

    let all = to_vec(&top_p_logits(&logits, 1.0).unwrap());
    assert_eq!(all, to_vec(&logits));
    assert!(top_p_logits(&logits, 0.0).is_err());
}

#[test]
fn mask_schedule_counts_sum_to_31_for_exponent_3() {
    let cfg = TtsConfig::default();
    let counts = cfg.mask_schedule();
    assert_eq!(counts, vec![0, 0, 0, 1, 1, 3, 4, 22]);
    assert_eq!(counts.iter().sum::<usize>(), 31);
}

fn tiny_rvq(q: i32, c: i32, l: i32) -> (WeightMap, UniquePtr<MlxArray>) {
    let mut key = 11u64;
    let embs = mlxcel_core::astype(&rand(&mut key, &[q, c, l], 1.0), dtype::BFLOAT16);
    let mut w = WeightMap::new();
    w.insert("rvq".to_string(), mlxcel_core::copy(&embs));
    (w, embs)
}

#[test]
fn depthsum_ignores_mask_index() {
    let (weights, embs) = tiny_rvq(3, 4, 5);
    let rvq = RvqCodebooks::from_weights(&weights, "rvq", 3, 4, 5).unwrap();
    let masked = mlxcel_core::from_slice_i32(&[4, 4, 4], &[1, 1, 3]);
    let out = rvq.depthsum_embedding(&masked).unwrap();
    assert_eq!(mlxcel_core::array_dtype(&out), dtype::FLOAT32);
    assert!(to_vec(&out).iter().all(|&v| v == 0.0));

    // Codebook 1 masked: the sum is rows 2 (codebook 0) and 3 (codebook 2).
    let code = mlxcel_core::from_slice_i32(&[2, 4, 3], &[1, 1, 3]);
    let got = to_vec(&rvq.depthsum_embedding(&code).unwrap());
    let all = to_vec(&embs);
    for d in 0..5 {
        let want = all[(2 * 5) + d] + all[(2 * 4 * 5) + 3 * 5 + d];
        assert!((got[d] - want).abs() < 1e-6, "{d}: {} vs {want}", got[d]);
    }
}

#[test]
fn rvq_encode_step_recovers_exact_codewords() {
    let (weights, embs) = tiny_rvq(3, 4, 5);
    let rvq = RvqCodebooks::from_weights(&weights, "rvq", 3, 4, 5).unwrap();
    // residual = codebook0[1] + codebook1[2]; quantizing codebooks 0..2
    // greedily must pick exactly those rows and leave column 2 untouched.
    let e = mlxcel_core::astype(&embs, dtype::FLOAT32);
    let r0 = mlxcel_core::slice(&e, &[0, 1, 0], &[1, 2, 5]);
    let r1 = mlxcel_core::slice(&e, &[1, 2, 0], &[2, 3, 5]);
    let residual = mlxcel_core::add(&r0, &r1);
    let code = mlxcel_core::from_slice_i32(&[4, 4, 4], &[1, 1, 3]);
    let out = rvq.encode_step(&residual, &code, 0, 2).unwrap();
    assert_eq!(mlxcel_core::array_dtype(&out), dtype::INT32);
    assert_eq!(to_i32(&out), vec![1, 2, 4]);
}

fn tiny_char_cfg() -> CharEncoderConfig {
    CharEncoderConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 4,
        char_vocab_size: 4,
        ..CharEncoderConfig::default()
    }
}

/// Vocabulary: three single-char tokens (so a 4-row char table) plus words.
fn tiny_vocab() -> HashMap<String, u32> {
    [
        ("b", 1u32),
        ("a", 0),
        ("c", 2),
        ("ab", 3),
        ("cab", 4),
        ("<pad>", 5),
    ]
    .into_iter()
    .map(|(t, i)| (t.to_string(), i))
    .collect()
}

fn tiny_subword_weights(cfg: &CharEncoderConfig, out: i32, zero_flags: bool) -> WeightMap {
    let mut key = 21u64;
    let (h, i) = (cfg.hidden_size as i32, cfg.intermediate_size as i32);
    let mut w = WeightMap::new();
    let mut put = |k: &str, shape: &[i32], scale: f32| {
        let v = mlxcel_core::astype(&rand(&mut key, shape, scale), dtype::BFLOAT16);
        w.insert(format!("sw.{k}"), v);
    };
    let enc = "backbone.encoder";
    for n in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        put(
            &format!("{enc}.layers.0.self_attn.{n}.weight"),
            &[h, h],
            0.3,
        );
    }
    for n in ["gate_proj", "up_proj"] {
        put(&format!("{enc}.layers.0.mlp.{n}.weight"), &[i, h], 0.3);
    }
    put(
        &format!("{enc}.layers.0.mlp.down_proj.weight"),
        &[h, i],
        0.3,
    );
    for n in [
        "pre_self_attn_layernorm",
        "post_self_attn_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ] {
        put(&format!("{enc}.layers.0.{n}.weight"), &[h], 0.1);
    }
    put(&format!("{enc}.norm.weight"), &[h], 0.1);
    put("embed_tokens.weight", &[cfg.char_vocab_size as i32, h], 1.0);
    put("proj_embedding.weight", &[out, h], 0.3);
    let flag_scale = if zero_flags { 0.0 } else { 1.0 };
    put("subword_flag_emb.cont_emb.weight", &[2, out], flag_scale);
    put("bos_eos_emb.special_emb.weight", &[3, out], flag_scale);
    let vocab = 6;
    let ints = |v: Vec<i32>| {
        mlxcel_core::astype(
            &mlxcel_core::from_slice_i32(&v, &[v.len() as i32]),
            dtype::INT64,
        )
    };
    let mut cont = vec![0; vocab + 1];
    cont[3] = 1;
    w.insert("sw.subword_flag_emb.is_continuation".into(), ints(cont));
    w.insert(
        "sw.subword_flag_emb.pad_tensor".into(),
        mlxcel_core::full_f32(&[], vocab as f32, dtype::INT64),
    );
    let mut special = vec![0; vocab];
    special[5] = 2;
    w.insert("sw.bos_eos_emb.special_flags".into(), ints(special));
    w.insert(
        "sw.bos_eos_emb.pad_tensor".into(),
        mlxcel_core::full_f32(&[], (vocab - 1) as f32, dtype::INT64),
    );
    w
}

#[test]
fn set_vocabulary_rejects_a_wrong_char_table() {
    let cfg = CharEncoderConfig {
        char_vocab_size: 5,
        ..tiny_char_cfg()
    };
    let mut enc = CharAwareSubwordEncoder::from_weights(
        &tiny_subword_weights(&cfg, 6, true),
        "sw",
        &cfg,
        6,
        64,
        4,
    )
    .unwrap();
    let err = enc.set_vocabulary(&tiny_vocab()).unwrap_err();
    assert!(err.contains("4 entries, expected 5"), "{err}");
    assert!(
        enc.forward(&[0], &[true], 1, 1).is_err(),
        "forward before vocabulary must fail"
    );
}

#[test]
fn char_encoder_scatters_only_masked_positions() {
    let cfg = tiny_char_cfg();
    let mut enc = CharAwareSubwordEncoder::from_weights(
        &tiny_subword_weights(&cfg, 6, true),
        "sw",
        &cfg,
        6,
        64,
        4,
    )
    .unwrap();
    enc.set_vocabulary(&tiny_vocab()).unwrap();
    let ids = [3, 4, 3, 1];
    let out = enc
        .forward(&ids, &[true, false, true, false], 2, 2)
        .unwrap();
    assert_eq!(mlxcel_core::array_shape(&out), vec![2, 2, 6]);
    assert_eq!(mlxcel_core::array_dtype(&out), dtype::BFLOAT16);
    let v = to_vec(&out);
    let row = |p: usize| &v[p * 6..(p + 1) * 6];
    // Flag embeddings are zero, so unmasked positions stay exactly zero and
    // both masked "ab" positions carry the same non-zero encoding.
    assert!(row(1).iter().chain(row(3)).all(|&x| x == 0.0));
    assert!(row(0).iter().any(|&x| x != 0.0));
    assert_eq!(row(0), row(2));
    // Encoding is per token: "ab" alone gives the same vector.
    let alone = to_vec(&enc.forward(&[3], &[true], 1, 1).unwrap());
    assert_eq!(alone.as_slice(), row(0));
}

#[test]
fn flag_embeddings_follow_the_token_tables() {
    let cfg = tiny_char_cfg();
    let weights = tiny_subword_weights(&cfg, 6, false);
    let mut enc = CharAwareSubwordEncoder::from_weights(&weights, "sw", &cfg, 6, 64, 4).unwrap();
    enc.set_vocabulary(&tiny_vocab()).unwrap();
    // Unmasked ids: only flags. id 3 is a continuation, id 5 special flag 2,
    // id 99 is out of range and falls back to the pad rows.
    let out = to_vec(&enc.forward(&[3, 5, 99], &[false; 3], 1, 3).unwrap());
    let cont = to_vec(&weights["sw.subword_flag_emb.cont_emb.weight"]);
    let special = to_vec(&weights["sw.bos_eos_emb.special_emb.weight"]);
    let expect = |c: usize, s: usize| -> Vec<f32> {
        let sum = (0..6)
            .map(|d| cont[c * 6 + d] + special[s * 6 + d])
            .collect::<Vec<_>>();
        to_vec(&mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&sum, &[6]),
            dtype::BFLOAT16,
        ))
    };
    let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2e-2);
    assert!(close(&out[0..6], &expect(1, 0)));
    assert!(close(&out[6..12], &expect(0, 2)));
    // 99 -> is_continuation pad (id 6, flag 0), special pad (id 5, flag 2).
    assert!(close(&out[12..18], &expect(0, 2)));
}

#[test]
fn mog_infer_shapes_are_finite_with_guidance() {
    let cfg = MogConfig {
        intermediate_size: 16,
        low_rank: 3,
        num_layers: 2,
        num_predictions: 5,
        ..MogConfig::default()
    };
    let (h, out) = (8i32, 4i32);
    let mut key = 31u64;
    let mut w = WeightMap::new();
    let mut put = |k: String, shape: &[i32], scale: f32| {
        let v = mlxcel_core::astype(&rand(&mut key, shape, scale), dtype::BFLOAT16);
        w.insert(k, v);
    };
    for l in 0..2 {
        put(format!("mog_head.mlp_stack.{l}.pre_norm.weight"), &[h], 0.1);
        put(
            format!("mog_head.mlp_stack.{l}.post_norm.weight"),
            &[h],
            0.1,
        );
        put(
            format!("mog_head.mlp_stack.{l}.mlp.gate_proj.weight"),
            &[16, h],
            0.3,
        );
        put(
            format!("mog_head.mlp_stack.{l}.mlp.up_proj.weight"),
            &[16, h],
            0.3,
        );
        put(
            format!("mog_head.mlp_stack.{l}.mlp.down_proj.weight"),
            &[h, 16],
            0.3,
        );
    }
    put("mog_head.mlp_stack.2.weight".into(), &[h], 0.1);
    put("mog_head.proj_logits.weight".into(), &[5, h], 0.3);
    put("mog_head.proj_mus.weight".into(), &[15, h], 0.3);
    put("mog_head.proj_logs.weight".into(), &[1, h], 0.3);
    put("mog_head.proj_else.weight".into(), &[out, h], 0.3);
    put("mog_head.low_mat".into(), &[5, out, 3], 0.3);
    // Load the weights as `RvqEarTtsModel` does: the projections are held
    // as f32 and the gathered tables (`proj_mus`, `low_mat`) as stored, so
    // the f32 result also depends on the head casting the gathered slabs
    // (issue #2087: CUDA builds resolve bf16 + f32 to bf16).
    let w = crate::audio::f32_weights::promoted_subset(&w, "", super::model::promotes_to_f32);
    let head =
        MogHead::from_weights(&w, "mog_head", h as usize, out as usize, &cfg, 64, 4).unwrap();
    let x = rand(&mut key, &[2, 1, h], 1.0);
    let (mu, logs) = head.infer(&x, 0.2, 0.95).unwrap();
    assert_eq!(mlxcel_core::array_shape(&mu), vec![1, 1, out]);
    assert_eq!(mlxcel_core::array_shape(&logs), vec![1, 1, 1]);
    assert_eq!(mlxcel_core::array_dtype(&mu), dtype::FLOAT32);
    assert!(to_vec(&mu).iter().all(|v| v.is_finite()));
    assert!(to_vec(&logs).iter().all(|&v| v.is_finite() && v >= -4.0));
    let odd = rand(&mut key, &[3, 1, h], 1.0);
    assert!(head.infer(&odd, 0.2, 0.95).is_err());
}

fn tiny_assets() -> SpeechDecoderAssets {
    let cfg = TtsConfig {
        hidden_size: 4,
        num_quantizers: 3,
        ..TtsConfig::default()
    };
    let mut w = WeightMap::new();
    w.insert(
        "t.audio_prompt_latents.Aria".into(),
        mlxcel_core::zeros(&[1, 3, 4], dtype::BFLOAT16),
    );
    let i64s = |v: &[i32]| {
        mlxcel_core::astype(
            &mlxcel_core::from_slice_i32(v, &[v.len() as i32]),
            dtype::INT64,
        )
    };
    w.insert("t.codec_silence_tokens".into(), i64s(&[7, 8, 9]));
    w.insert("t._control_codes".into(), i64s(&[1000, 1001, 1002]));
    SpeechDecoderAssets::from_weights(&w, "t", &cfg).unwrap()
}

#[test]
fn control_codes_become_silence() {
    let assets = tiny_assets();
    assert_eq!(assets.prompt_frames(), 3);
    assert_eq!(assets.control_codes, vec![1000, 1001, 1002]);
    let codes = mlxcel_core::from_slice_i32(&[5, 1001, 6, 1000, 1002, 2], &[1, 2, 3]);
    let out = to_i32(&assets.replace_control_codes(&codes));
    // Frame 1: 1000 in codebook 0 -> 7, 1002 in codebook 1 -> 8.
    assert_eq!(out, vec![5, 8, 6, 7, 8, 2]);
}

#[test]
fn prompt_masks_first_and_second_to_last_frames() {
    let cfg = TtsConfig {
        num_quantizers: 2,
        codebook_size: 16,
        ..TtsConfig::default()
    };
    let codec = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7, 8], &[1, 4, 2]);
    let prompt = TtsPrompt::from_codec_codes(&codec, 3, &cfg, 12).unwrap();
    assert_eq!(to_i32(&prompt.codes), vec![16, 16, 3, 4, 16, 16]);
    assert_eq!(prompt.subword_ids, vec![12, 12, 12]);
    assert_eq!(prompt.subword_mask, vec![false, true, true]);
    assert_eq!(prompt.audio_mask, vec![false, false, true]);
    assert_eq!(to_i32(&prompt.last_frame()), vec![16, 16]);
    assert!(TtsPrompt::from_codec_codes(&codec, 4, &cfg, 12).is_err());
}

#[test]
fn native_sum_reduces_the_requested_axis() {
    let mut key = 41u64;
    let x = rand(&mut key, &[2, 3, 4], 1.0);
    for axis in 0..3usize {
        let got = super::norm_mlp::native_sum_axis(&x, axis);
        let want = mlxcel_core::sum_axis(&x, axis as i32, false);
        assert_eq!(
            mlxcel_core::array_shape(&got),
            mlxcel_core::array_shape(&want)
        );
        for (a, b) in to_vec(&got).iter().zip(to_vec(&want)) {
            assert!((a - b).abs() < 1e-5, "axis {axis}: {a} vs {b}");
        }
    }
    let xb = mlxcel_core::astype(&x, dtype::BFLOAT16);
    let got = super::norm_mlp::native_sum_axis(&xb, 1);
    assert_eq!(mlxcel_core::array_dtype(&got), dtype::BFLOAT16);
    assert_eq!(mlxcel_core::array_shape(&got), vec![2, 4]);
}

/// Issue #2045: only projections fed by f32 activations are held as f32.
#[test]
fn f32_promotion_keeps_norms_and_the_subword_path_as_stored() {
    use super::model::promotes_to_f32;
    for key in [
        "backbone.layers.3.self_attn.q_proj.weight",
        "backbone.layers.3.mlp.down_proj.weight",
        "mog_head.mlp_stack.1.mlp.gate_proj.weight",
        "mog_head.proj_logits.weight",
        "mog_head.proj_else.weight",
        "embed_code.weight",
        "gated_fusion_audio_text.audio_proj.bias",
    ] {
        assert!(promotes_to_f32(key), "{key}");
    }
    for key in [
        "backbone.layers.3.input_layernorm.weight",
        "backbone.layers.3.self_attn.q_norm.weight",
        "backbone.norm.weight",
        "mog_head.mlp_stack.1.pre_norm.weight",
        "mog_head.proj_mus.weight",
        "mog_head.low_mat",
        "embed_subword.backbone.encoder.layers.0.mlp.up_proj.weight",
        "gated_fusion_audio_text.text_proj.weight",
        "gated_fusion_audio_text.gate",
        "null_emb",
        "rvq_embs",
    ] {
        assert!(!promotes_to_f32(key), "{key}");
    }
}
