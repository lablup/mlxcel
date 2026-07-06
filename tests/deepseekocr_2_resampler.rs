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

//! DeepSeek-OCR 2 query resampler: synthetic forward-pass checks.
//!
//! Builds a small resampler from a deterministic in-memory weight map (no
//! checkpoint required) and exercises the parts unique to this module: query-
//! bank selection by input length, the GQA head repetition (kv_heads <
//! num_heads), the `[image | queries]` mixed mask, and the query-slice output.
//! The real-checkpoint numerics are validated by end-to-end OCR generation.

use mlxcel::vision::encoders::deepseekocr_qwen2::{
    Qwen2Resampler, Qwen2ResamplerConfig, mixed_attn_mask,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Deterministic small values in `[-0.05, 0.05]` for a weight of `shape`.
fn arr(shape: &[i32], seed: i32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (((i * 7 + seed) % 13) as f32 / 13.0 - 0.5) * 0.1)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn small_config() -> Qwen2ResamplerConfig {
    Qwen2ResamplerConfig {
        dim: 8,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 1, // GQA: one KV head repeated to two query heads.
        head_dim: 4,
        intermediate: 16,
        rms_eps: 1e-6,
        rope_theta: 10_000.0,
        query_len_global: 4,
        query_len_tile: 2,
    }
}

fn build_weights(cfg: &Qwen2ResamplerConfig, prefix: &str) -> WeightMap {
    let mut w = WeightMap::new();
    let mut put = |name: String, a: UniquePtr<MlxArray>| {
        w.insert(name, a);
    };
    let (dim, hd) = (cfg.dim, cfg.head_dim);
    let q_dim = cfg.num_heads * hd;
    let kv_dim = cfg.num_kv_heads * hd;
    for l in 0..cfg.num_layers {
        let lp = format!("{prefix}.layers.{l}");
        let s = (l as i32 + 1) * 11;
        put(format!("{lp}.input_layernorm.weight"), arr(&[dim], s + 1));
        put(
            format!("{lp}.post_attention_layernorm.weight"),
            arr(&[dim], s + 2),
        );
        put(
            format!("{lp}.self_attn.q_proj.weight"),
            arr(&[q_dim, dim], s + 3),
        );
        put(format!("{lp}.self_attn.q_proj.bias"), arr(&[q_dim], s + 4));
        put(
            format!("{lp}.self_attn.k_proj.weight"),
            arr(&[kv_dim, dim], s + 5),
        );
        put(format!("{lp}.self_attn.k_proj.bias"), arr(&[kv_dim], s + 6));
        put(
            format!("{lp}.self_attn.v_proj.weight"),
            arr(&[kv_dim, dim], s + 7),
        );
        put(format!("{lp}.self_attn.v_proj.bias"), arr(&[kv_dim], s + 8));
        put(
            format!("{lp}.self_attn.o_proj.weight"),
            arr(&[dim, q_dim], s + 9),
        );
        put(
            format!("{lp}.mlp.gate_proj.weight"),
            arr(&[cfg.intermediate, dim], s + 10),
        );
        put(
            format!("{lp}.mlp.up_proj.weight"),
            arr(&[cfg.intermediate, dim], s + 11),
        );
        put(
            format!("{lp}.mlp.down_proj.weight"),
            arr(&[dim, cfg.intermediate], s + 12),
        );
    }
    put(format!("{prefix}.norm.weight"), arr(&[dim], 99));
    put(
        format!("{prefix}.query_1024"),
        arr(&[cfg.query_len_global, dim], 100),
    );
    put(
        format!("{prefix}.query_768"),
        arr(&[cfg.query_len_tile, dim], 200),
    );
    w
}

fn absmean(a: &MlxArray) -> f32 {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    let flat = mlxcel_core::reshape(&f, &[-1]);
    let am = mlxcel_core::mean_axis(&mlxcel_core::abs(&flat), 0, false);
    mlxcel_core::eval(&am);
    mlxcel_core::item_f32(&am)
}

fn run(resampler: &Qwen2Resampler, seq: i32, dim: i32) -> UniquePtr<MlxArray> {
    let n = seq * dim;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32 - 0.5) * 0.2).collect();
    let x = mlxcel_core::from_slice_f32(&data, &[1, seq, dim]);
    resampler.forward(&x)
}

#[test]
fn resampler_selects_query_bank_and_slices_queries() {
    let cfg = small_config();
    let weights = build_weights(&cfg, "enc");
    let resampler = Qwen2Resampler::from_weights(&weights, "enc", cfg.clone()).expect("build");

    // Global-length input (S == query_len_global) -> query_1024, output Q = 4.
    let out_g = run(&resampler, cfg.query_len_global, cfg.dim);
    assert_eq!(
        mlxcel_core::array_shape(&out_g),
        vec![1, cfg.query_len_global, cfg.dim]
    );

    // Tile-length input (S == query_len_tile) -> query_768, output Q = 2.
    let out_t = run(&resampler, cfg.query_len_tile, cfg.dim);
    assert_eq!(
        mlxcel_core::array_shape(&out_t),
        vec![1, cfg.query_len_tile, cfg.dim]
    );

    // Finite, non-trivial output, and deterministic across runs.
    let m = absmean(&out_g);
    assert!(m.is_finite() && m > 0.0, "resampler output absmean={m}");
    let out_g2 = run(&resampler, cfg.query_len_global, cfg.dim);
    assert!(
        (absmean(&out_g2) - m).abs() < 1e-6,
        "resampler is nondeterministic"
    );
}

#[test]
fn mixed_mask_blocks_image_to_query_but_allows_query_to_image() {
    // S = 3 image tokens, Q = 2 queries, N = 5.
    let m = mixed_attn_mask(3, 2);
    let n = 5usize;
    let neg = -1e9f32;
    let cell = |row: usize, col: usize| m[row * n + col];
    // Image row 0 sees image keys, is blocked from both query keys (3, 4).
    assert_eq!(cell(0, 0), 0.0);
    assert_eq!(cell(0, 3), neg);
    assert_eq!(cell(0, 4), neg);
    // First query (row 3): all image keys allowed, self allowed, next query blocked.
    assert_eq!(cell(3, 0), 0.0);
    assert_eq!(cell(3, 3), 0.0);
    assert_eq!(cell(3, 4), neg);
    // Last query (row 4): everything allowed.
    assert!(m[4 * n..5 * n].iter().all(|&v| v == 0.0));
}
