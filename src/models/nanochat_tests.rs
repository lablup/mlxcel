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

//! Unit tests for the nanochat model. Everything is checkpoint-free: the
//! numeric guards for the four non-standard choices use CPU references, and the
//! tiny-model tests use deterministic synthetic weights.

use super::{
    Attention, EosTokenId, ModelArgs, NANOCHAT_EOS_TOKEN_ID, NanoChatModel, apply_softcap,
    mirrored_rope_freqs, remap_key, sanitize_weights,
};
use mlxcel_core::MlxArray;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;

fn to_vec(arr: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Deterministic pseudo-random values in `[-scale, scale)`.
fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

// Config.

#[test]
fn config_defaults_match_d20() {
    let args: ModelArgs = serde_json::from_str(r#"{"model_type": "nanochat"}"#).unwrap();
    assert_eq!(args.hidden_size, 1280);
    assert_eq!(args.num_hidden_layers, 20);
    assert_eq!(args.num_attention_heads, 10);
    assert_eq!(args.num_key_value_heads, 10);
    assert_eq!(args.vocab_size, 65536);
    assert_eq!(args.max_position_embeddings, 2048);
    assert_eq!(args.intermediate_size(), 5120);
    assert_eq!(args.head_dim(), 128);
    assert!((args.rope_theta - 10000.0).abs() < 1e-3);
    assert!((args.rms_norm_eps - 1e-5).abs() < 1e-12);
    assert_eq!(args.soft_cap(), Some(15.0));
    assert!(!args.tie_word_embeddings);
    assert_eq!(args.eos_token_ids(), vec![NANOCHAT_EOS_TOKEN_ID]);
    args.validate().unwrap();
}

#[test]
fn config_accepts_the_real_mlx_conversion() {
    // `madmag77/nanochat-d20-q8-mlx` writes both spellings of the softcap and a
    // list-valued eos_token_id; serde's `alias` would reject the duplicate.
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "nanochat", "logits_soft_cap": 15.0, "logits_softcap": 15.0,
            "eos_token_id": [65531], "rms_norm_eps": 1e-06,
            "quantization": {"group_size": 64, "bits": 8, "mode": "affine"}}"#,
    )
    .unwrap();
    assert_eq!(args.soft_cap(), Some(15.0));
    assert!(matches!(args.eos_token_id, Some(EosTokenId::Multiple(_))));
    assert_eq!(args.eos_token_ids(), vec![65531]);
    assert_eq!((args.group_size(), args.bits()), (64, 8));
    assert!((args.rms_norm_eps - 1e-6).abs() < 1e-12);
}

#[test]
fn softcap_config_forms() {
    let parse = |s: &str| -> ModelArgs { serde_json::from_str(s).unwrap() };
    assert_eq!(parse(r#"{"logits_soft_cap": null}"#).soft_cap(), None);
    assert_eq!(parse(r#"{"logits_softcap": null}"#).soft_cap(), None);
    assert_eq!(parse(r#"{"logits_softcap": 30.0}"#).soft_cap(), Some(30.0));
    assert_eq!(parse(r#"{}"#).soft_cap(), Some(15.0));
}

#[test]
fn config_validation_rejects_hostile_scalars() {
    for (cfg, needle) in [
        (r#"{"num_attention_heads": 0}"#, "num_attention_heads"),
        (r#"{"hidden_size": 0}"#, "hidden_size"),
        (r#"{"hidden_size": 1281}"#, "divisible"),
        (r#"{"num_hidden_layers": 0}"#, "num_hidden_layers"),
        (r#"{"num_hidden_layers": 100000000}"#, "num_hidden_layers"),
        (r#"{"num_key_value_heads": 5}"#, "num_key_value_heads"),
        (
            r#"{"hidden_size": 10, "num_attention_heads": 10}"#,
            "head_dim",
        ),
        (r#"{"rope_theta": 0.0}"#, "rope_theta"),
        (r#"{"rms_norm_eps": -1.0}"#, "rms_norm_eps"),
        (r#"{"logits_soft_cap": 0.0}"#, "logits_soft_cap"),
    ] {
        let args: ModelArgs = serde_json::from_str(cfg).unwrap();
        let err = args.validate().unwrap_err();
        assert!(err.contains(needle), "{cfg}: {err}");
    }
}

// Weight-key sanitation.

fn mlx_layout_keys(layers: usize) -> Vec<String> {
    let mut keys = vec![
        "transformer.wte.weight".to_string(),
        "lm_head.weight".into(),
    ];
    for i in 0..layers {
        for k in [
            "attn.c_q",
            "attn.c_k",
            "attn.c_v",
            "attn.c_proj",
            "mlp.c_fc",
            "mlp.c_proj",
        ] {
            keys.push(format!("transformer.h.{i}.{k}.weight"));
            keys.push(format!("transformer.h.{i}.{k}.scales"));
        }
    }
    keys
}

fn transformers_layout(keys: &[String]) -> Vec<String> {
    keys.iter()
        .map(|k| {
            let mut k = k.replace("transformer.wte.", "model.embed_tokens.");
            if let Some(rest) = k.strip_prefix("transformer.h.") {
                let (idx, tail) = rest.split_once('.').unwrap();
                let tail = tail
                    .replace("attn.c_q.", "self_attn.q_proj.")
                    .replace("attn.c_k.", "self_attn.k_proj.")
                    .replace("attn.c_v.", "self_attn.v_proj.")
                    .replace("attn.c_proj.", "self_attn.o_proj.")
                    .replace("mlp.c_fc.", "mlp.fc1.")
                    .replace("mlp.c_proj.", "mlp.fc2.");
                k = format!("model.layers.{idx}.{tail}");
            }
            k
        })
        .collect()
}

fn map_of(keys: &[String]) -> WeightMap {
    keys.iter()
        .map(|k| (k.clone(), mlxcel_core::from_slice_f32(&[1.0], &[1])))
        .collect()
}

fn sorted_keys(w: &WeightMap) -> Vec<String> {
    let mut k: Vec<String> = w.keys().cloned().collect();
    k.sort();
    k
}

#[test]
fn sanitize_maps_transformers_layout_and_is_idempotent() {
    let mlx = mlx_layout_keys(3);
    let mut hf = transformers_layout(&mlx);
    assert_ne!(hf, mlx);
    hf.push("model.layers.0.self_attn.rotary_emb.inv_freq".into());
    hf.push("model.rotary_emb.inv_freq".into());

    let once = sanitize_weights(&map_of(&hf));
    let mut want = mlx.clone();
    want.sort();
    assert_eq!(sorted_keys(&once), want);

    let twice = sanitize_weights(&once);
    assert_eq!(sorted_keys(&twice), want);

    // The MLX layout passes through untouched.
    assert_eq!(sorted_keys(&sanitize_weights(&map_of(&mlx))), want);
    assert_eq!(remap_key("lm_head.scales"), "lm_head.scales");
}

// The four non-standard choices, against CPU references.

fn tiny_attention(head_dim: usize, base: f32) -> Attention {
    let dim = head_dim as i32;
    let lin = || {
        let mut w = WeightMap::new();
        w.insert(
            "l.weight".into(),
            mlxcel_core::from_slice_f32(&lcg(head_dim * head_dim, 3, 0.3), &[dim, dim]),
        );
        UnifiedLinear::from_weights(&w, "l", 64, 4).unwrap()
    };
    Attention {
        c_q: lin(),
        c_k: lin(),
        c_v: lin(),
        c_proj: lin(),
        freqs: mirrored_rope_freqs(head_dim, base),
        num_heads: 1,
        head_dim: dim,
        scale: (head_dim as f32).powf(-0.5),
        eps: 1e-5,
    }
}

/// `rope(x)[i] = x[i] cos(p/f) + x[i+half] sin(p/f)`,
/// `rope(x)[i+half] = -x[i] sin(p/f) + x[i+half] cos(p/f)`, `f = base^(i/half)`.
fn mirrored_rope_ref(x: &[f32], pos: f32, base: f32) -> Vec<f32> {
    let half = x.len() / 2;
    let mut out = vec![0.0; x.len()];
    for i in 0..half {
        let a = pos / base.powf(i as f32 / half as f32);
        out[i] = x[i] * a.cos() + x[i + half] * a.sin();
        out[i + half] = -x[i] * a.sin() + x[i + half] * a.cos();
    }
    out
}

#[test]
fn negative_freq_rope_rotates_backwards() {
    // One pair (head_dim 2) at position 1, base 10000: f_0 = 1, angle = 1 rad.
    let (q0, q1) = (0.7f32, -0.4f32);
    let x = mlxcel_core::from_slice_f32(&[q0, q1], &[1, 1, 1, 2]);
    let freqs = mirrored_rope_freqs(2, 10000.0);
    let out = to_vec(&mlxcel_core::fast_rope_with_freqs(
        &x, 2, false, 1.0, 1, &freqs,
    ));
    assert!((out[0] - (q0 * 1f32.cos() + q1 * 1f32.sin())).abs() < 1e-5);
    assert!((out[1] - (-q0 * 1f32.sin() + q1 * 1f32.cos())).abs() < 1e-5);

    // The ordinary rotation goes the other way; using it would be wrong.
    let ordinary = to_vec(&mlxcel_core::fast_rope(&x, 2, false, 10000.0, 1.0, 1));
    assert!((ordinary[0] - out[0]).abs() > 0.1);
}

#[test]
fn qk_norm_runs_after_rope() {
    let head_dim = 8;
    let attn = tiny_attention(head_dim, 10000.0);
    let qv = lcg(head_dim, 11, 2.0);
    let kv = lcg(head_dim, 12, 5.0);
    let q = mlxcel_core::from_slice_f32(&qv, &[1, 1, 1, head_dim as i32]);
    let k = mlxcel_core::from_slice_f32(&kv, &[1, 1, 1, head_dim as i32]);
    let offset = 3;
    let (qn, kn) = attn.rotate_and_norm(&q, &k, offset);

    let check = |got: &MlxArray, raw: &[f32]| {
        let rotated = mirrored_rope_ref(raw, offset as f32, 10000.0);
        let ms = rotated.iter().map(|v| v * v).sum::<f32>() / rotated.len() as f32;
        let inv = 1.0 / (ms + 1e-5).sqrt();
        for (g, r) in to_vec(got).iter().zip(&rotated) {
            assert!((g - r * inv).abs() < 1e-4, "{g} vs {}", r * inv);
        }
    };
    check(&qn, &qv);
    check(&kn, &kv);

    // Weightless: the per-head output has unit RMS (up to eps).
    let rms = (to_vec(&qn).iter().map(|v| v * v).sum::<f32>() / head_dim as f32).sqrt();
    assert!((rms - 1.0).abs() < 1e-3);
}

#[test]
fn softcap_bounds_logits() {
    let vals: Vec<f32> = lcg(64, 5, 1000.0);
    let x = mlxcel_core::from_slice_f32(&vals, &[1, 1, 64]);
    let capped = to_vec(&apply_softcap(mlxcel_core::copy(&x), Some(15.0)));
    for (c, v) in capped.iter().zip(&vals) {
        assert!(c.abs() <= 15.0, "{c} escapes the cap");
        assert!((c - 15.0 * (v / 15.0).tanh()).abs() < 1e-3);
    }
    // Disabled: passes through untouched.
    assert_eq!(to_vec(&apply_softcap(mlxcel_core::copy(&x), None)), vals);
}

// Tiny model.

fn tiny_args() -> ModelArgs {
    serde_json::from_str(
        r#"{"model_type": "nanochat", "hidden_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 32,
            "intermediate_size": 64, "max_position_embeddings": 256}"#,
    )
    .unwrap()
}

fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let (h, i, v) = (args.hidden_size, args.intermediate_size(), args.vocab_size);
    let mut w = WeightMap::new();
    let mut seed = 100u64;
    let mut put = |w: &mut WeightMap, name: String, rows: usize, cols: usize| {
        seed += 1;
        let data = lcg(rows * cols, seed, 0.25);
        w.insert(
            name,
            mlxcel_core::from_slice_f32(&data, &[rows as i32, cols as i32]),
        );
    };
    put(&mut w, "transformer.wte.weight".into(), v, h);
    put(&mut w, "lm_head.weight".into(), v, h);
    for l in 0..args.num_hidden_layers {
        let p = format!("transformer.h.{l}");
        for n in ["c_q", "c_k", "c_v", "c_proj"] {
            put(&mut w, format!("{p}.attn.{n}.weight"), h, h);
        }
        put(&mut w, format!("{p}.mlp.c_fc.weight"), i, h);
        put(&mut w, format!("{p}.mlp.c_proj.weight"), h, i);
    }
    w
}

fn tiny_model() -> NanoChatModel {
    let args = tiny_args();
    NanoChatModel::from_weights(&tiny_weights(&args), &args).unwrap()
}

fn prefill(model: &NanoChatModel, ids: &[i32]) -> Vec<f32> {
    let mut caches = LanguageModel::make_caches(model);
    let x = mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32]);
    to_vec(&LanguageModel::forward(model, &x, &mut caches, None))
}

#[test]
fn tiny_model_prefill_is_causal() {
    let model = tiny_model();
    let long: Vec<i32> = (0..96).map(|i| (i * 7 + 3) % 32).collect();
    let vocab = 32;
    let full = prefill(&model, &long);
    let short = prefill(&model, &long[..48]);
    assert_eq!(full.len(), 96 * vocab);
    for (a, b) in full[..48 * vocab].iter().zip(&short) {
        assert!((a - b).abs() < 1e-4, "prefix logits changed: {a} vs {b}");
    }
    assert!(full.iter().all(|v| v.is_finite() && v.abs() <= 15.0));
}

#[test]
fn tiny_model_decode_matches_prefill_and_last_logits() {
    let model = tiny_model();
    let ids: Vec<i32> = (0..10).map(|i| (i * 5 + 1) % 32).collect();
    let full = prefill(&model, &ids);
    let last_full = &full[9 * 32..];

    let mut caches = LanguageModel::make_caches(&model);
    let head = mlxcel_core::from_slice_i32(&ids[..9], &[1, 9]);
    LanguageModel::forward(&model, &head, &mut caches, None);
    let tok = mlxcel_core::from_slice_i32(&ids[9..], &[1, 1]);
    let step = to_vec(&LanguageModel::forward(&model, &tok, &mut caches, None));
    for (a, b) in last_full.iter().zip(&step) {
        assert!((a - b).abs() < 1e-3, "decode vs prefill: {a} vs {b}");
    }

    let mut caches = LanguageModel::make_caches(&model);
    let x = mlxcel_core::from_slice_i32(&ids, &[1, 10]);
    let row = to_vec(&LanguageModel::forward_last_logits(
        &model,
        &x,
        &mut caches,
        None,
        9,
    ));
    for (a, b) in last_full.iter().zip(&row) {
        assert!((a - b).abs() < 1e-4);
    }

    // The engine's prefill entry carries a sequence id and must take the same
    // last-row projection.
    let mut caches = LanguageModel::make_caches(&model);
    let row = LanguageModel::forward_last_logits_with_sequence_id(
        &model,
        &x,
        Some(mlxcel_core::cache::SequenceId::from_raw(7)),
        &mut caches,
        None,
        9,
    );
    assert_eq!(mlxcel_core::array_shape(&row), vec![1, 1, 32]);
    for (a, b) in last_full.iter().zip(&to_vec(&row)) {
        assert!((a - b).abs() < 1e-4);
    }
    assert_eq!(LanguageModel::num_layers(&model), 2);
    assert_eq!(
        LanguageModel::eos_token_ids(&model),
        vec![NANOCHAT_EOS_TOKEN_ID]
    );
}

#[test]
fn transformers_layout_loads_and_matches() {
    let args = tiny_args();
    let mlx = tiny_weights(&args);
    let keys: Vec<String> = mlx.keys().cloned().collect();
    let hf_keys = transformers_layout(&keys);
    let hf: WeightMap = keys
        .iter()
        .zip(&hf_keys)
        .map(|(k, hk)| (hk.clone(), mlxcel_core::copy(&mlx[k])))
        .collect();
    let a = NanoChatModel::from_weights(&mlx, &args).unwrap();
    let b = NanoChatModel::from_weights(&hf, &args).unwrap();
    let ids = [1, 2, 3, 4, 5];
    assert_eq!(prefill(&a, &ids), prefill(&b, &ids));
}
