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

//! Unit tests for the Gemma 3 per-layer RoPE scale (#1340).
//!
//! Gemma 3 checkpoints from 4B up declare `{"rope_type": "linear", "factor":
//! 8.0}` and it applies to the global-attention layers only. Two things are
//! pinned here, and no shape assertion or smoke prompt reaches either.
//!
//! The first is that the block is read at all. It was parsed into
//! `ModelArgs::rope_scaling` and never consumed, so every layer rotated at
//! `scale = 1.0` and the global layers saw positions eight times larger than
//! the ones they were trained on. That produces fluent text, which is why it
//! survived, and the divergence grows with position, which is why only a long
//! prompt separates the two graphs at the output.
//!
//! The second is the sliding/global split. Handing the scale to every layer is
//! as wrong as handing it to none, and it is the mistake a fix is most likely
//! to make, because both mistakes still load, still decode and still read
//! fluently. Upstream reaches `initialize_rope` with a `scaling_config` on the
//! non-sliding branch only
//! ([`mlx_lm/models/gemma3_text.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma3_text.py)),
//! and mlx-vlm spells the same rule as `scaling_config=None if self.is_sliding
//! else config.rope_scaling`
//! ([`mlx_vlm/models/gemma3/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma3/language.py)).

use super::{ModelArgs, layer_rope_params};

/// Parse a config fragment the way a `config.json` (or a VLM `text_config`)
/// delivers it. `ModelArgs` is `#[serde(default)]`, so a fragment names only
/// the keys under test and every other field takes the Gemma 3 default.
fn args(json: &str) -> ModelArgs {
    serde_json::from_str(json).unwrap_or_else(|err| panic!("config must parse: {err}\n{json}"))
}

/// Read an array of any rank back into a flat `Vec<f32>`.
fn to_vec(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    let n = mlxcel_core::array_size(a);
    let flat = mlxcel_core::reshape(a, &[n as i32]);
    mlxcel_core::eval(&flat);
    (0..n)
        .map(|i| {
            let element = mlxcel_core::slice(&flat, &[i as i32], &[i as i32 + 1]);
            mlxcel_core::item_f32(&element)
        })
        .collect()
}

fn max_abs_diff(a: &mlxcel_core::MlxArray, b: &mlxcel_core::MlxArray) -> f32 {
    let (a, b) = (to_vec(a), to_vec(b));
    assert_eq!(a.len(), b.len(), "compared arrays must have equal size");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

// What the block resolves to.

#[test]
fn global_rope_scale_reads_linear_factor() {
    // The block every shipped Gemma 3 checkpoint from 4B up declares.
    // `mx.fast.rope` multiplies the position by `scale`, so `linear` is
    // `1 / factor`, matching upstream's `scale = 1 / scaling_config["factor"]`.
    assert_eq!(
        args(r#"{"rope_scaling": {"rope_type": "linear", "factor": 8.0}}"#).global_rope_scale(),
        Ok(0.125)
    );

    // Absent, empty, and an explicit "default" all stay unscaled. 1B declares
    // no block at all and must keep the graph it decodes with today.
    assert_eq!(args(r#"{}"#).global_rope_scale(), Ok(1.0));
    assert_eq!(args(r#"{"rope_scaling": {}}"#).global_rope_scale(), Ok(1.0));
    assert_eq!(
        args(r#"{"rope_scaling": {"rope_type": "default"}}"#).global_rope_scale(),
        Ok(1.0)
    );

    // A scheme this path does not implement is a named load error rather than
    // a silent 1.0. The name has to reach the operator: a `yarn` block decoding
    // on an unscaled table is exactly the failure this issue is about, one
    // scheme over.
    let err = args(r#"{"rope_scaling": {"rope_type": "yarn", "factor": 40.0}}"#)
        .global_rope_scale()
        .expect_err("an unimplemented scheme must not resolve to a scale");
    assert!(
        err.contains("yarn"),
        "the error must name the scheme: {err}"
    );

    // A `linear` block with no usable factor cannot select a scale, and
    // defaulting it to 1.0 would be the same silent no-op in a different place.
    assert!(
        args(r#"{"rope_scaling": {"rope_type": "linear"}}"#)
            .global_rope_scale()
            .is_err()
    );
    assert!(
        args(r#"{"rope_scaling": {"rope_type": "linear", "factor": 0.0}}"#)
            .global_rope_scale()
            .is_err()
    );
    assert!(
        args(r#"{"rope_scaling": {"rope_type": "linear", "factor": -8.0}}"#)
            .global_rope_scale()
            .is_err()
    );
}

#[test]
fn global_rope_scale_accepts_both_spellings_of_the_type_key() {
    // The legacy `type` key, which upstream reads first.
    assert_eq!(
        args(r#"{"rope_scaling": {"type": "linear", "factor": 4.0}}"#).global_rope_scale(),
        Ok(0.25)
    );

    // A block carrying BOTH keys must parse rather than fail. This is why the
    // reader goes through the shared `RopeScalingSpec` map lookup instead of a
    // derived `#[serde(rename = "type", alias = "rope_type")]` field: serde
    // rejects a repeated field with `duplicate field`, and five checkpoints in
    // the local model set spell both (#1355 found this the hard way).
    assert_eq!(
        args(r#"{"rope_scaling": {"type": "linear", "rope_type": "linear", "factor": 2.0}}"#)
            .global_rope_scale(),
        Ok(0.5)
    );

    // A JSON null under `type` reads as absent and falls through, matching
    // upstream's `get("type") or get("rope_type", "default")`.
    assert_eq!(
        args(r#"{"rope_scaling": {"type": null, "rope_type": "linear", "factor": 8.0}}"#)
            .global_rope_scale(),
        Ok(0.125)
    );
}

#[test]
fn the_shipped_4b_text_config_resolves_to_one_eighth() {
    // Verbatim `text_config` of `mlx-community/gemma-3-4b-it-4bit`, which is the
    // checkpoint in the recommended test set. It names no `rope_theta` and no
    // `sliding_window_pattern`, so the defaults have to carry them; if either
    // default drifts, the global layers stop being the layers this scale is
    // meant for.
    let args = args(
        r#"{"hidden_size": 2560, "intermediate_size": 10240, "model_type": "gemma3_text",
            "num_hidden_layers": 34, "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
            "sliding_window": 1024}"#,
    );
    assert_eq!(args.global_rope_scale(), Ok(0.125));
    assert_eq!(args.rope_theta, 1_000_000.0);
    assert_eq!(args.rope_local_base_freq, 10_000.0);
    assert_eq!(args.sliding_window_pattern, 6);
}

// Which layers the scale reaches.

#[test]
fn sliding_layers_keep_unit_scale() {
    let args = args(
        r#"{"num_hidden_layers": 12, "sliding_window_pattern": 6,
            "rope_scaling": {"rope_type": "linear", "factor": 8.0}}"#,
    );

    let mut global_layers = Vec::new();
    for layer_idx in 0..args.num_hidden_layers {
        let (is_sliding, base, scale) =
            layer_rope_params(&args, layer_idx).expect("a linear block must resolve");

        if (layer_idx + 1).is_multiple_of(args.sliding_window_pattern) {
            global_layers.push(layer_idx);
            assert!(!is_sliding, "layer {layer_idx} must be global");
            assert_eq!(base, args.rope_theta, "layer {layer_idx} base");
            assert_eq!(scale, 0.125, "layer {layer_idx} scale");
        } else {
            assert!(is_sliding, "layer {layer_idx} must be sliding");
            assert_eq!(base, args.rope_local_base_freq, "layer {layer_idx} base");
            assert_eq!(
                scale, 1.0,
                "sliding layer {layer_idx} must rotate at an unscaled position; upstream \
                 passes no scaling_config on this branch"
            );
        }
    }

    // Guard: a pattern that produced no global layer would make every
    // assertion above vacuous on the half this issue is about.
    assert_eq!(global_layers, vec![5, 11]);
}

#[test]
fn a_config_without_a_block_leaves_every_layer_unscaled() {
    // The `gemma-3-1b-it-4bit` shape. It is the control for the whole change:
    // if anything here moves, the fix is not confined to scaled configs.
    let args = args(r#"{"num_hidden_layers": 26, "sliding_window_pattern": 6}"#);
    for layer_idx in 0..args.num_hidden_layers {
        let (_, _, scale) = layer_rope_params(&args, layer_idx).expect("no block must resolve");
        assert_eq!(scale, 1.0, "layer {layer_idx}");
    }
}

#[test]
fn an_unsupported_scheme_fails_every_layer_not_just_the_global_ones() {
    // The scale is resolved before the sliding/global branch, so the load error
    // does not depend on which layer index happens to be global first. A
    // per-branch resolve would let a config whose `sliding_window_pattern`
    // exceeds `num_hidden_layers` load with an unimplemented scheme.
    let args = args(
        r#"{"num_hidden_layers": 8, "sliding_window_pattern": 6,
            "rope_scaling": {"rope_type": "llama3", "factor": 8.0}}"#,
    );
    for layer_idx in 0..args.num_hidden_layers {
        assert!(
            layer_rope_params(&args, layer_idx).is_err(),
            "layer {layer_idx} must refuse an unimplemented scheme"
        );
    }
}

// What the scale does to the rotation.

#[test]
fn global_layer_rope_matches_scaled_positions() {
    // `theta_{p,i} = (p * scale) * base^(-2i/head_dim)`, so with `factor = 8` a
    // global layer at cache offset 4096 must rotate exactly as an unscaled
    // layer at position 512. This is the arithmetic the whole issue reduces to,
    // and it is invisible at small offsets: at offset 8 the two graphs agree to
    // three decimals on every element, which is why a short prompt is not
    // evidence either way.
    let head_dim = 64_i32;
    let n_heads = 8_i32;
    let base = 1_000_000.0_f32;
    let scale = 0.125_f32;

    let vals: Vec<f32> = (0..(n_heads * head_dim))
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    // [B, H, T, D], the layout the attention block hands to `fast_rope`.
    let q = mlxcel_core::from_slice_f32(&vals, &[1, n_heads, 1, head_dim]);

    let scaled_at_4096 = mlxcel_core::fast_rope(&q, head_dim, false, base, scale, 4096);
    let unscaled_at_512 = mlxcel_core::fast_rope(&q, head_dim, false, base, 1.0, 512);
    let agreement = max_abs_diff(&scaled_at_4096, &unscaled_at_512);
    assert!(
        agreement < 1e-5,
        "a 1/8 scale at offset 4096 must equal an unscaled rotation at position 512, \
         max abs diff {agreement}"
    );

    // Guard: the comparison above would also pass if `fast_rope` ignored both
    // the scale and the offset. What the model did before this change is the
    // unscaled rotation at the true offset, and that must be materially
    // different, else the fix is a no-op on the family it targets.
    let unscaled_at_4096 = mlxcel_core::fast_rope(&q, head_dim, false, base, 1.0, 4096);
    let separation = max_abs_diff(&scaled_at_4096, &unscaled_at_4096);
    assert!(
        separation > 0.1,
        "the scaled and unscaled rotations at offset 4096 must differ materially, \
         max abs diff {separation}"
    );
}
