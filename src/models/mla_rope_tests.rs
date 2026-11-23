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

//! Tests for the shared MLA rope and its DeepSeek-V3 wiring (#3295).
//!
//! The reference tables are computed here in float64 from upstream
//! `YarnRoPE.__init__` (`mlx_lm/layers/rope.py`), not from
//! `yarn_rope_setup`, so a drift in the shared builder shows up as a failure.

use std::collections::HashMap;

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::{MlxArray, UniquePtr};

use super::MlaRope;
use crate::models::deepseek_v3::tests::{
    ramp, test_config_dense_direct_q, tiny_dense_model_weights,
};
use crate::models::deepseek_v3::{DeepSeekV3Attention, DeepSeekV3Config, DeepSeekV3Model};

fn block(json: &str) -> HashMap<String, serde_json::Value> {
    serde_json::from_str(json).expect("rope_scaling block")
}

/// Upstream `YarnRoPE._freqs` in float64.
pub(crate) fn upstream_yarn_freqs(
    dims: usize,
    base: f64,
    factor: f64,
    original: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> Vec<f64> {
    let correction_dim = |rotations: f64| {
        (dims as f64 * (original / (rotations * 2.0 * std::f64::consts::PI)).ln())
            / (2.0 * base.ln())
    };
    let low = correction_dim(beta_fast).floor().max(0.0);
    let high = correction_dim(beta_slow).ceil().min(dims as f64 - 1.0);
    let high = if low == high { high + 0.001 } else { high };
    (0..dims / 2)
        .map(|i| {
            let extra = base.powf((2 * i) as f64 / dims as f64);
            let inter = factor * extra;
            let mask = 1.0 - ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            (inter * extra) / (inter * mask + extra * (1.0 - mask))
        })
        .collect()
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    array_to_vec_f32(a)
        .iter()
        .zip(array_to_vec_f32(b))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn pe_rows(heads: i32, len: i32, dims: i32) -> UniquePtr<MlxArray> {
    let n = (heads * len * dims) as usize;
    mlxcel_core::from_slice_f32(&ramp(n), &[1, heads, len, dims])
}

/// The tables the shared rope builds for the shipped DeepSeek-V3, GigaChat3
/// (`deepseek_v3`) and LongCat-Flash-Lite (`longcat_flash_ngram`) blocks match
/// upstream `YarnRoPE` at their real rope width.
#[test]
fn yarn_tables_match_upstream_at_checkpoint_geometry() {
    let _runtime = crate::initialize_runtime();
    let cases = [
        // DeepSeek-V3 / Kimi-K2: factor 40 over 4096 at rope_theta 1e4.
        (
            r#"{"type": "yarn", "factor": 40, "original_max_position_embeddings": 4096,
                "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0}"#,
            10_000.0,
            40.0,
            4096.0,
        ),
        // GigaChat3-10B-A1.8B: factor 64 over 4096 at rope_theta 1e5.
        (
            r#"{"type": "yarn", "factor": 64.0, "original_max_position_embeddings": 4096,
                "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0}"#,
            100_000.0,
            64.0,
            4096.0,
        ),
        // LongCat-Flash-Lite ngram: factor 10 over 32768 at rope_theta 5e6.
        (
            r#"{"rope_type": "yarn", "factor": 10, "original_max_position_embeddings": 32768,
                "beta_fast": 32, "beta_slow": 1, "mscale": 1, "mscale_all_dim": 1}"#,
            5_000_000.0,
            10.0,
            32768.0,
        ),
    ];
    for (json, base, factor, original) in cases {
        let block = block(json);
        let rope = MlaRope::from_block(Some(&block), 64, base as f32, None, "test");
        let got = array_to_vec_f32(rope.freqs().expect("a YaRN block builds a table"));
        let want = upstream_yarn_freqs(64, base, factor, original, 32.0, 1.0);
        assert_eq!(got.len(), want.len());
        for (pair, (g, w)) in got.iter().zip(&want).enumerate() {
            let g = f64::from(*g);
            assert!(
                ((g - w) / w).abs() < 1e-5,
                "base {base} pair {pair}: got {g}, upstream {w}"
            );
        }
        // The lowest-frequency pair is interpolated by the full factor.
        let last = want.len() - 1;
        let plain_last = base.powf((2 * last) as f64 / 64.0);
        assert!(((want[last] / plain_last) - factor).abs() < 1e-6);
        assert_eq!(rope.mscale(), 1.0, "mscale == mscale_all_dim resolves to 1");
    }
}

/// `deepseek_yarn` is the same table as `yarn`, upstream and here.
#[test]
fn deepseek_yarn_alias_builds_the_yarn_table() {
    let _runtime = crate::initialize_runtime();
    let yarn = block(r#"{"type": "yarn", "factor": 8, "original_max_position_embeddings": 64}"#);
    let alias =
        block(r#"{"type": "deepseek_yarn", "factor": 8, "original_max_position_embeddings": 64}"#);
    let a = MlaRope::from_block(Some(&yarn), 8, 10_000.0, None, "test");
    let b = MlaRope::from_block(Some(&alias), 8, 10_000.0, None, "test");
    let (a, b) = (a.freqs().expect("yarn"), b.freqs().expect("alias"));
    assert_eq!(array_to_vec_f32(a), array_to_vec_f32(b));
}

/// No block (and the explicit plain rope Youtu uses) keeps the exact
/// `fast_rope(.., base, 1.0, offset)` call, single and batched.
#[test]
fn absent_block_keeps_the_exact_plain_rope_call() {
    let _runtime = crate::initialize_runtime();
    let x = pe_rows(2, 3, 8);
    for rope in [
        MlaRope::from_block(None, 8, 50_000.0, None, "test"),
        MlaRope::plain(8, 50_000.0),
    ] {
        assert!(rope.freqs().is_none());
        for traditional in [true, false] {
            let got = rope.apply(&x, traditional, 9000);
            let want = mlxcel_core::fast_rope(&x, 8, traditional, 50_000.0, 1.0, 9000);
            assert_eq!(max_abs_diff(&got, &want), 0.0, "must stay byte-identical");
            let got = rope.apply_batched(&x, traditional, &[9000]);
            let want = mlxcel_core::fast_rope_batched(&x, 8, traditional, 50_000.0, 1.0, &[9000]);
            assert_eq!(
                max_abs_diff(&got, &want),
                0.0,
                "batched must stay byte-identical"
            );
        }
    }
}

/// A YaRN magnitude other than 1 scales only the rotated features.
#[test]
fn magnitude_scales_only_the_rotated_dims() {
    let _runtime = crate::initialize_runtime();
    let b = block(
        r#"{"type": "yarn", "factor": 16, "original_max_position_embeddings": 32,
            "mscale": 1.0, "mscale_all_dim": 0.0}"#,
    );
    let rope = MlaRope::from_block(Some(&b), 4, 10_000.0, None, "test");
    let mscale = rope.mscale();
    assert!((mscale - (0.1 * 16f32.ln() + 1.0)).abs() < 1e-6);

    let x = pe_rows(1, 1, 8);
    let got = array_to_vec_f32(&rope.apply(&x, true, 0));
    let src = array_to_vec_f32(&x);
    // Position 0 rotates by nothing, so the head is only scaled.
    for i in 0..4 {
        assert!((got[i] - src[i] * mscale).abs() < 1e-6, "rotated dim {i}");
    }
    for i in 4..8 {
        assert_eq!(got[i], src[i], "pass-through dim {i}");
    }
}

fn yarn_v3_config() -> DeepSeekV3Config {
    let mut cfg = test_config_dense_direct_q();
    cfg.rope_scaling = Some(block(
        r#"{"type": "yarn", "factor": 40, "original_max_position_embeddings": 4,
            "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0}"#,
    ));
    cfg
}

fn v3_weights(cfg: &DeepSeekV3Config) -> mlxcel_core::weights::WeightMap {
    DeepSeekV3Model::sanitize_weights_with_args(tiny_dense_model_weights(cfg), cfg)
        .expect("sanitize must succeed")
}

/// Before #3295 a `deepseek_v3` YaRN block reached only the softmax scale:
/// `q_pe` past the original context rotated with the plain frequencies. The
/// attention now rotates with the upstream table there, and stays on the
/// plain call without a block.
#[test]
fn deepseek_v3_yarn_block_changes_q_pe_past_the_original_context() {
    let _runtime = crate::initialize_runtime();
    let plain_cfg = test_config_dense_direct_q();
    let yarn_cfg = yarn_v3_config();
    let attention = |cfg: &DeepSeekV3Config| {
        DeepSeekV3Attention::from_weights(&v3_weights(cfg), cfg, "model.layers.0.self_attn")
            .expect("attention")
    };
    let plain = attention(&plain_cfg);
    let yarn = attention(&yarn_cfg);
    let dims = plain_cfg.qk_rope_head_dim as i32;
    let q_pe = pe_rows(plain_cfg.num_attention_heads as i32, 1, dims);

    // Position 16 is four times the original context of 4.
    let offset = 16;
    let plain_q = plain.rope.apply(&q_pe, plain.rope_traditional, offset);
    let yarn_q = yarn.rope.apply(&q_pe, yarn.rope_traditional, offset);
    assert!(
        max_abs_diff(&plain_q, &yarn_q) > 1e-3,
        "a YaRN block must change q_pe past the original context"
    );

    let want = mlxcel_core::fast_rope(&q_pe, dims, true, plain_cfg.rope_theta, 1.0, offset);
    assert_eq!(max_abs_diff(&plain_q, &want), 0.0, "no block: plain call");

    let table: Vec<f32> = upstream_yarn_freqs(4, 10_000.0, 40.0, 4.0, 32.0, 1.0)
        .into_iter()
        .map(|f| f as f32)
        .collect();
    let table = mlxcel_core::from_slice_f32(&table, &[2]);
    let want = mlxcel_core::fast_rope_with_freqs(&q_pe, dims, true, 1.0, offset, &table);
    assert!(
        max_abs_diff(&yarn_q, &want) < 1e-5,
        "upstream YarnRoPE rotation"
    );
}

/// The table reaches the model's forward (every layer, prefill and the
/// batched decode path alike).
#[test]
fn deepseek_v3_yarn_block_reaches_every_layer_and_the_batched_decode() {
    let _runtime = crate::initialize_runtime();
    let model = |cfg: &DeepSeekV3Config| {
        DeepSeekV3Model::from_weights(&v3_weights(cfg), cfg).expect("tiny model must load")
    };
    let plain = model(&test_config_dense_direct_q());
    let yarn = model(&yarn_v3_config());
    assert!(
        plain
            .layers
            .iter()
            .all(|l| l.self_attn.rope.freqs().is_none())
    );
    assert!(
        yarn.layers
            .iter()
            .all(|l| l.self_attn.rope.freqs().is_some())
    );

    let input = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], &[1, 10]);
    let a = plain.forward_impl(&input, &mut plain.make_caches_impl(), None);
    let b = yarn.forward_impl(&input, &mut yarn.make_caches_impl(), None);
    assert!(
        max_abs_diff(&a, &b) > 1e-4,
        "the YaRN table must change the logits"
    );

    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &yarn,
        &[&[1, 2, 3, 4, 5], &[2, 3], &[4, 5, 6, 7, 8, 9]],
        &[7, 8, 9],
        1e-5,
    );
}
