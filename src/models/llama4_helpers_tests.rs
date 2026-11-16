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

use crate::models::llama4::{Llama4Cache, TextArgs};
use crate::models::llama4_helpers::{create_chunked_attention_mask, get_weight_copy};
use mlxcel_core::cache::KvAttention;
use mlxcel_core::layers::ChunkedKVCache;
use mlxcel_core::weights::WeightMap;
use std::sync::{Mutex, OnceLock};

fn test_guard() -> &'static Mutex<()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(()))
}

fn scalar_at(mask: &mlxcel_core::MlxArray, row: i32, col: i32) -> f32 {
    let cell = mlxcel_core::slice(mask, &[row, col], &[row + 1, col + 1]);
    let cell = mlxcel_core::reshape(&cell, &[1]);
    mlxcel_core::eval(&cell);
    mlxcel_core::item_f32(&cell)
}

#[test]
#[ignore = "requires serial MLX execution"]
fn chunked_attention_mask_blocks_other_chunks_and_future_tokens() {
    let _guard = test_guard().lock().unwrap();
    let mask = create_chunked_attention_mask(3, 0, 0, 2);

    assert_eq!(mlxcel_core::array_shape(&mask), vec![3, 3]);
    assert_eq!(scalar_at(&mask, 0, 0), 0.0);
    assert!(scalar_at(&mask, 0, 1).is_infinite() && scalar_at(&mask, 0, 1).is_sign_negative());
    assert_eq!(scalar_at(&mask, 1, 1), 0.0);
    assert!(scalar_at(&mask, 1, 2).is_infinite() && scalar_at(&mask, 1, 2).is_sign_negative());
    assert!(scalar_at(&mask, 2, 0).is_infinite() && scalar_at(&mask, 2, 0).is_sign_negative());
    assert_eq!(scalar_at(&mask, 2, 2), 0.0);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn get_weight_copy_returns_new_buffer_and_reports_missing_keys() {
    let _guard = test_guard().lock().unwrap();
    let original = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let mut weights = WeightMap::new();
    weights.insert("layers.0.test".to_string(), original);

    let copied = get_weight_copy(&weights, "layers.0.test").unwrap();
    assert_eq!(mlxcel_core::array_shape(&copied), vec![2, 2]);
    assert_ne!(
        copied.as_ref().unwrap() as *const _,
        weights["layers.0.test"].as_ref().unwrap() as *const _
    );

    let missing = get_weight_copy(&weights, "layers.1.test");
    assert!(missing.is_err());
}

#[test]
#[ignore = "requires serial MLX execution"]
fn text_args_quantization_defaults_are_available_to_helpers() {
    let _guard = test_guard().lock().unwrap();
    let args = TextArgs {
        model_type: "llama4".to_string(),
        hidden_size: 128,
        num_hidden_layers: 2,
        intermediate_size: 256,
        intermediate_size_mlp: 128,
        num_attention_heads: 4,
        num_key_value_heads: 1,
        rms_norm_eps: 1e-5,
        vocab_size: 32000,
        head_dim: 32,
        max_position_embeddings: 1024,
        attention_chunk_size: 8,
        interleave_moe_layer_step: 4,
        num_local_experts: 8,
        num_experts_per_tok: 2,
        attention_bias: false,
        use_qk_norm: true,
        rope_theta: 500000.0,
        attn_temperature_tuning: 0,
        floor_scale: 1,
        attn_scale: 1.0,
        group_size: None,
        bits: None,
    };

    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}

// Issue #3100: the chunked layers' maskless prefill must match the dense
// chunk-mask path it replaced.

const CHUNK: i32 = 8;
const Q_HEADS: i32 = 4;
const KV_HEADS: i32 = 2;
const HEAD_DIM: i32 = 16;

/// Deterministic pseudo-random `[1, heads, n, HEAD_DIM]` for absolute
/// positions `[start, start + n)`, distinct per `salt`.
fn tensor(
    heads: i32,
    start: i32,
    n: i32,
    salt: f32,
    dtype: i32,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut data = Vec::with_capacity((heads * n * HEAD_DIM) as usize);
    for h in 0..heads {
        for t in start..start + n {
            for d in 0..HEAD_DIM {
                let x = (h * 7919 + t * 104_729 + d * 31) as f32 * 0.001 + salt;
                data.push((x * 12.9898).sin() * 1.5);
            }
        }
    }
    let arr = mlxcel_core::from_slice_f32(&data, &[1, heads, n, HEAD_DIM]);
    mlxcel_core::astype(&arr, dtype)
}

/// Run one `len`-token step behind `history` cached tokens through a fresh
/// chunked cache, either maskless (the #3100 path) or with the dense
/// chunk mask (the path it replaced). The history arrives as one earlier
/// forward, then the front trim the model runs before every step.
fn chunked_step(history: i32, len: i32, dense: bool, dtype: i32) -> Vec<f32> {
    let mut cache = Llama4Cache::Chunked(ChunkedKVCache::new(CHUNK));
    if history > 0 {
        let q = tensor(Q_HEADS, 0, history, 0.0, dtype);
        let k = tensor(KV_HEADS, 0, history, 1.0, dtype);
        let v = tensor(KV_HEADS, 0, history, 2.0, dtype);
        let _ = cache.attend(&q, k, v, 0.25, None);
    }
    cache.maybe_trim_front();
    let (start, offset) = (cache.start_position(), cache.offset());
    let q = tensor(Q_HEADS, history, len, 0.0, dtype);
    let k = tensor(KV_HEADS, history, len, 1.0, dtype);
    let v = tensor(KV_HEADS, history, len, 2.0, dtype);
    let out = if dense {
        let mask = create_chunked_attention_mask(len, start, offset, CHUNK as usize);
        let mask = mlxcel_core::astype(&mask, dtype);
        cache.attend(&q, k, v, 0.25, Some(&mask))
    } else {
        cache.attend(&q, k, v, 0.25, None)
    };
    assert_eq!(
        mlxcel_core::array_shape(&out),
        vec![1, Q_HEADS, len, HEAD_DIM]
    );
    let out = mlxcel_core::astype(&out, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&out);
    mlxcel_core::array_to_raw_bytes(&out)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_maskless_matches_dense(history: i32, len: i32, dtype: i32, tol: f32) {
    let _guard = test_guard().lock().unwrap();
    let got = chunked_step(history, len, false, dtype);
    let want = chunked_step(history, len, true, dtype);
    assert_eq!(got.len(), want.len());
    let max = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    assert!(got.iter().all(|x| x.is_finite()), "non-finite output");
    assert!(
        max <= tol,
        "history {history}, len {len}: maskless chunked attention differs from the dense mask by {max}"
    );
}

#[test]
fn chunked_prefill_at_offset_zero_inside_one_chunk_matches_dense_mask() {
    assert_maskless_matches_dense(0, 6, mlxcel_core::dtype::FLOAT32, 1e-5);
}

#[test]
fn chunked_prefill_at_offset_zero_across_chunks_matches_dense_mask() {
    // 20 tokens over chunks [0, 8), [8, 16), [16, 20).
    assert_maskless_matches_dense(0, 20, mlxcel_core::dtype::FLOAT32, 1e-5);
}

#[test]
fn chunked_prefill_starting_mid_chunk_matches_dense_mask() {
    // Queries 3..7 attend to the cached keys 0..3 of their own chunk.
    assert_maskless_matches_dense(3, 4, mlxcel_core::dtype::FLOAT32, 1e-5);
}

#[test]
fn chunked_prefill_starting_mid_chunk_and_crossing_a_boundary_matches_dense_mask() {
    // Queries 5..14: 5..8 see cached 0..5; 8..14 see only chunk 1.
    assert_maskless_matches_dense(5, 9, mlxcel_core::dtype::FLOAT32, 1e-5);
}

#[test]
fn chunked_prefill_behind_a_trimmed_front_matches_dense_mask() {
    // 20 cached tokens trim to start_position 12, so the visible window
    // holds keys of chunk 1 that queries 20..27 (chunks 2 and 3) must skip.
    assert_maskless_matches_dense(20, 7, mlxcel_core::dtype::FLOAT32, 1e-5);
    // A step that starts exactly on a chunk boundary.
    assert_maskless_matches_dense(16, 8, mlxcel_core::dtype::FLOAT32, 1e-5);
}

#[test]
fn chunked_prefill_in_f16_matches_dense_mask() {
    assert_maskless_matches_dense(5, 19, mlxcel_core::dtype::FLOAT16, 5e-3);
}
