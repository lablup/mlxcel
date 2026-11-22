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

use super::layers::{SparseLinear, Z1tLinear};
use super::{Z1t, Z1tConfig, make_states};
use crate::models::ModelType;
use crate::models::batched_decode_harness::{
    assert_model_owned_fresh_mixed_and_transition,
    assert_strict_model_owned_multistep_with_paged_lengths,
};
use crate::models::detection::get_model_type;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const DIM: usize = 8;
const HEADS: usize = 2;
const KSIZE: usize = 3;
const FAN: usize = 2;
const VOCAB: usize = 32;
const LAYERS: usize = 2;

fn config_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "z1t",
        "vocab_size": VOCAB,
        "hidden_size": DIM,
        "num_hidden_layers": LAYERS,
        "max_position_embeddings": 6,
        "aft_kind": "conv",
        "aft_heads": HEADS,
        "aft_ksize": KSIZE,
        "dyt_alpha": 0.5,
        "linear_fan_in": FAN,
        "tanh_linear": true,
        "tanh_mlp": true,
        "tie_word_embeddings": false,
        "eos_token_id": 31
    })
}

fn config() -> Z1tConfig {
    serde_json::from_value(config_json()).expect("Z1T test config parses")
}

fn varied(shape: &[i32], seed: usize, scale: f32) -> UniquePtr<MlxArray> {
    let len: usize = shape.iter().map(|d| *d as usize).product();
    let values: Vec<f32> = (0..len)
        .map(|i| {
            let magnitude = ((i * 7 + seed * 13) % 23 + 1) as f32 * scale;
            if (i + seed).is_multiple_of(2) {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

fn index_table(out: usize, in_features: usize, seed: usize) -> Vec<i32> {
    (0..out * FAN)
        .map(|i| ((i * 5 + seed * 3 + i / FAN) % in_features) as i32)
        .collect()
}

fn put_sparse(w: &mut WeightMap, prefix: &str, in_f: usize, out: usize, seed: usize) {
    let shape = [out as i32, FAN as i32];
    w.insert(format!("{prefix}.weight"), varied(&shape, seed, 0.03));
    w.insert(
        format!("{prefix}.bias"),
        varied(&[out as i32], seed + 1, 0.01),
    );
    let idx = index_table(out, in_f, seed);
    w.insert(
        format!("{prefix}.indices"),
        mlxcel_core::from_slice_i32(&idx, &shape),
    );
}

fn put_dyt(w: &mut WeightMap, prefix: &str, seed: usize) {
    w.insert(
        format!("{prefix}.alpha"),
        mlxcel_core::from_slice_f32(&[0.5], &[1]),
    );
    w.insert(
        format!("{prefix}.weight"),
        varied(&[DIM as i32], seed, 0.08),
    );
    w.insert(
        format!("{prefix}.bias"),
        varied(&[DIM as i32], seed + 2, 0.01),
    );
}

/// Raw JAX key names, as `mlx-community/Z1T-0-mlx` ships them.
fn weights() -> WeightMap {
    let mut w = WeightMap::new();
    w.insert(
        "embedding.weight".into(),
        varied(&[VOCAB as i32, DIM as i32], 1, 0.02),
    );
    w.insert("pe.pe".into(), varied(&[6, DIM as i32], 2, 0.01));
    for layer in 0..LAYERS {
        let p = format!("blocks.{layer}");
        let s = 10 * (layer + 1);
        put_dyt(&mut w, &format!("{p}.norm1"), s);
        put_dyt(&mut w, &format!("{p}.norm2"), s + 1);
        put_sparse(
            &mut w,
            &format!("{p}.attn.qkv_proj"),
            DIM,
            2 * DIM + HEADS,
            s + 2,
        );
        put_sparse(&mut w, &format!("{p}.attn.out_proj"), DIM, DIM, s + 3);
        w.insert(
            format!("{p}.attn.w_conv"),
            varied(&[HEADS as i32, KSIZE as i32], s + 4, 0.05),
        );
        put_sparse(&mut w, &format!("{p}.mlp.proj1"), DIM, 4 * DIM, s + 5);
        put_sparse(&mut w, &format!("{p}.mlp.proj2"), 4 * DIM, DIM, s + 6);
    }
    put_dyt(&mut w, "norm", 90);
    w.insert(
        "clf.weight".into(),
        varied(&[VOCAB as i32, DIM as i32], 91, 0.05),
    );
    w.insert("clf.bias".into(), varied(&[VOCAB as i32], 92, 0.01));
    w
}

fn model() -> Z1t {
    Z1t::from_weights(&weights(), config()).expect("Z1T test model loads")
}

fn values(array: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(array)
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    assert_eq!(mlxcel_core::array_shape(a), mlxcel_core::array_shape(b));
    values(a)
        .iter()
        .zip(values(b))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

const TOKENS: &[i32] = &[3, 1, 4, 1, 5, 9, 2, 6, 5, 3];

#[test]
fn z1t_streaming_state_matches_flat_forward() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    let len = TOKENS.len();
    let ids = mlxcel_core::from_slice_i32(TOKENS, &[1, len as i32]);

    let mut flat_states = make_states(LAYERS);
    let flat = model.model.forward_rows(&ids, &mut flat_states, &[0], None);
    mlxcel_core::eval(&flat);

    // Token by token, and in uneven multi-token pieces, through one state.
    for pieces in [vec![1; len], vec![4, 1, 3, 2]] {
        let mut states = make_states(LAYERS);
        let mut start = 0;
        let mut outputs = Vec::new();
        for piece in pieces {
            let chunk =
                mlxcel_core::from_slice_i32(&TOKENS[start..start + piece], &[1, piece as i32]);
            let offset = states[0].offset;
            let out = model
                .model
                .forward_rows(&chunk, &mut states, &[offset], None);
            mlxcel_core::eval(&out);
            outputs.push(out);
            start += piece;
        }
        assert_eq!(states[0].offset, len);
        let refs: Vec<&MlxArray> = outputs.iter().map(|o| &**o).collect();
        let streamed = mlxcel_core::concatenate_many(&refs, 1);
        let diff = max_abs_diff(&streamed, &flat);
        assert!(diff < 1e-5, "streamed hidden differs from flat by {diff}");
    }
}

/// Dense `[out, in]` matrix accumulating every `(o, indices[o, k])` weight.
fn dense_reference(
    x: &[f32],
    rows: usize,
    in_f: usize,
    out: usize,
    w: &[f32],
    b: &[f32],
    idx: &[i32],
) -> Vec<f32> {
    let mut dense = vec![0.0f32; out * in_f];
    for o in 0..out {
        for k in 0..FAN {
            dense[o * in_f + idx[o * FAN + k] as usize] += w[o * FAN + k];
        }
    }
    let mut y = vec![0.0f32; rows * out];
    for r in 0..rows {
        for o in 0..out {
            let mut acc = b[o];
            for i in 0..in_f {
                acc += dense[o * in_f + i] * x[r * in_f + i];
            }
            y[r * out + o] = acc;
        }
    }
    y
}

#[test]
fn z1t_sparse_linear_matches_dense_reference_including_chunked_gather() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let (in_f, out, len) = (DIM, 2 * DIM + HEADS, 7usize);
    let mut w = WeightMap::new();
    put_sparse(&mut w, "p", in_f, out, 4);
    let mut linear = SparseLinear::from_weights(&w, "p", in_f, out, FAN).expect("loads");
    let x = varied(&[1, len as i32, in_f as i32], 6, 0.1);
    let expected = dense_reference(
        &values(&x),
        len,
        in_f,
        out,
        &values(w.get("p.weight").unwrap()),
        &values(w.get("p.bias").unwrap()),
        &index_table(out, in_f, 4),
    );
    for cap in [super::layers::SPARSE_CHUNK_ELEMS, (out * 2) as i64, 1] {
        linear.chunk_elems = cap;
        let got = linear.forward(&x);
        assert_eq!(
            mlxcel_core::array_shape(&got),
            vec![1, len as i32, out as i32]
        );
        for (i, (g, e)) in values(&got).iter().zip(&expected).enumerate() {
            assert!((g - e).abs() < 1e-6, "cap {cap}: y[{i}] = {g}, dense {e}");
        }
    }
}

#[test]
fn z1t_chunked_gather_inside_the_model_is_exact() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let ids = mlxcel_core::from_slice_i32(TOKENS, &[1, TOKENS.len() as i32]);
    let reference = model();
    let mut caches = reference.make_caches();
    let want = reference.forward(&ids, &mut caches, None);
    mlxcel_core::eval(&want);

    let mut chunked = model();
    for block in &mut chunked.model.layers {
        for linear in [
            &mut block.attn.qkv_proj,
            &mut block.attn.out_proj,
            &mut block.mlp.proj1,
            &mut block.mlp.proj2,
        ] {
            linear.sparse_mut().expect("sparse at fan-in 2").chunk_elems = 40;
        }
    }
    let mut caches = chunked.make_caches();
    let got = chunked.forward(&ids, &mut caches, None);
    let diff = max_abs_diff(&got, &want);
    assert!(diff < 1e-6, "chunked gather moved logits by {diff}");
}

#[test]
fn z1t_accepts_upstream_sanitized_key_names() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut renamed = WeightMap::new();
    for (key, value) in weights() {
        let key = if key == "pe.pe" {
            "language_model.model.pe".to_string()
        } else if let Some(rest) = key.strip_prefix("embedding.") {
            format!("language_model.model.embed_tokens.{rest}")
        } else if let Some(rest) = key.strip_prefix("blocks.") {
            format!("language_model.model.layers.{rest}")
        } else if let Some(rest) = key.strip_prefix("norm.") {
            format!("language_model.model.norm.{rest}")
        } else if let Some(rest) = key.strip_prefix("clf.") {
            format!("language_model.lm_head.{rest}")
        } else {
            key
        };
        renamed.insert(key, value);
    }
    let a = model();
    let b = Z1t::from_weights(&renamed, config()).expect("sanitized names load");
    let ids = mlxcel_core::from_slice_i32(TOKENS, &[1, TOKENS.len() as i32]);
    let (mut ca, mut cb) = (a.make_caches(), b.make_caches());
    let diff = max_abs_diff(
        &a.forward(&ids, &mut ca, None),
        &b.forward(&ids, &mut cb, None),
    );
    assert_eq!(diff, 0.0);
}

#[test]
fn z1t_refuses_unsupported_configs_and_bad_index_tables() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut json = config_json();
    json["aft_kind"] = "linear".into();
    let cfg: Z1tConfig = serde_json::from_value(json).unwrap();
    let err = Z1t::from_weights(&weights(), cfg)
        .err()
        .expect("aft_kind linear refused");
    assert!(err.contains("aft_kind"), "{err}");

    let mut json = config_json();
    json["aft_heads"] = 3.into();
    let cfg: Z1tConfig = serde_json::from_value(json).unwrap();
    assert!(Z1t::from_weights(&weights(), cfg).is_err());

    let mut bad = weights();
    let out = 2 * DIM + HEADS;
    let mut idx = index_table(out, DIM, 12);
    idx[3] = DIM as i32;
    bad.insert(
        "blocks.0.attn.qkv_proj.indices".into(),
        mlxcel_core::from_slice_i32(&idx, &[out as i32, FAN as i32]),
    );
    let err = Z1t::from_weights(&bad, config())
        .err()
        .expect("out-of-range index refused");
    assert!(err.contains("outside the input width"), "{err}");
}

#[test]
fn z1t_dense_fan_in_uses_a_dense_linear() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut w = WeightMap::new();
    w.insert("d.weight".into(), varied(&[5, DIM as i32], 3, 0.05));
    w.insert("d.bias".into(), varied(&[5], 4, 0.01));
    let linear = Z1tLinear::from_weights(&w, "d", DIM, 5, None, &config()).expect("dense loads");
    assert!(matches!(linear, Z1tLinear::Dense(_)));
}

#[test]
fn z1t_detection_and_eos_come_from_config() {
    let dir = std::env::temp_dir().join(format!("mlxcel-z1t-detect-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json().to_string()).unwrap();
    assert_eq!(get_model_type(&dir).unwrap(), ModelType::Z1t);
    std::fs::remove_dir_all(&dir).unwrap();

    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    assert_eq!(model().eos_token_ids(), vec![31]);
    let policy: &dyn LanguageModel = &model();
    assert!(!policy.supports_chunked_prefill());
    assert!(!policy.supports_padded_prefill());
}

#[test]
fn z1t_model_owned_native_batched_decode() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    assert_strict_model_owned_multistep_with_paged_lengths(model, &[0; LAYERS]);
}

#[test]
fn z1t_fresh_mixed_and_native_scalar_native_transition() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    assert_model_owned_fresh_mixed_and_transition(model);
}

crate::models::last_row_logits_harness::last_row_case!(z1t_last_row_logits, model);
