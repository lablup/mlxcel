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

//! Unit tests for the shared `rope_scaling` reader (#1355).
//!
//! Two things are pinned here and neither is visible from a shape assertion.
//!
//! The first is the key the scheme name is read from. Every Llama 3.x config
//! writes `rope_type`; the struct this replaced read `type` and nothing else, so
//! the field was `None` on exactly the checkpoints it described, and the table
//! it selected was never built. A parse test is the only place that catches
//! that, because a config that resolves to `Default` loads, caches, generates
//! and reads as fluent text.
//!
//! The second is the table itself. The scaled and unscaled tables agree closely
//! at low positions, so no short-prompt output separates them; the band
//! arithmetic is therefore checked against the closed form directly.

use super::{RopeScalingKind, RopeScalingSpec, llama3_rope_freqs};

/// Parse a `rope_scaling` block the way a `config.json` delivers it.
fn spec(json: &str) -> RopeScalingSpec {
    serde_json::from_str(json).unwrap_or_else(|err| panic!("block must parse: {err}\n{json}"))
}

/// Read a `[n]` float32 array back into a `Vec<f32>`.
fn to_vec(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::eval(a);
    let n = mlxcel_core::array_size(a);
    (0..n)
        .map(|i| {
            let element = mlxcel_core::slice(a, &[i as i32], &[i as i32 + 1]);
            mlxcel_core::item_f32(&element)
        })
        .collect()
}

// The parse contract.

#[test]
fn rope_scaling_reads_the_rope_type_key() {
    // The spelling every Llama 3.1 / 3.2 / 3.3 config actually uses. The struct
    // this replaced read only `type`, so this block resolved to `Default` and
    // the whole feature was inert on the checkpoints it was written for.
    let parsed = spec(
        r#"{"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3"}"#,
    );
    assert_eq!(parsed.rope_type(), "llama3");
    assert_eq!(parsed.factor, Some(8.0));
    assert_eq!(parsed.low_freq_factor, Some(1.0));
    assert_eq!(parsed.high_freq_factor, Some(4.0));
    assert_eq!(parsed.original_max_position_embeddings, Some(8192.0));
}

#[test]
fn rope_scaling_still_reads_the_legacy_type_key() {
    // `models/deepseek-coder-1.3b-4bit` is a `llama` checkpoint that spells it
    // this way and nothing else, so dropping the old spelling would have
    // silently taken linear scaling away from a checkpoint that reaches this
    // path today.
    let parsed = spec(r#"{"factor": 4.0, "type": "linear"}"#);
    assert_eq!(parsed.rope_type(), "linear");
    assert_eq!(parsed.factor, Some(4.0));
}

#[test]
fn a_block_carrying_both_spellings_parses_instead_of_erroring() {
    // `models/internvl3-1b`'s `text_config` writes both keys. This is why the
    // reader goes through a JSON map rather than a derived struct with
    // `#[serde(rename = "type", alias = "rope_type")]`: serde rejects a repeated
    // field, which would turn a checkpoint that loads today into a parse error.
    let parsed = spec(r#"{"factor": 2.0, "rope_type": "dynamic", "type": "dynamic"}"#);
    assert_eq!(parsed.rope_type(), "dynamic");
    assert_eq!(parsed.factor, Some(2.0));
}

#[test]
fn a_null_type_key_falls_through_to_rope_type() {
    // Upstream's `scaling_config.get("type") or scaling_config.get("rope_type",
    // "default")` treats a JSON null as absent because `None` is falsy. Reading
    // the key as a string does the same, so the two agree on this shape.
    let parsed = spec(r#"{"type": null, "rope_type": "llama3", "factor": 8.0}"#);
    assert_eq!(parsed.rope_type(), "llama3");
}

#[test]
fn a_block_naming_no_scheme_is_default() {
    assert_eq!(spec(r#"{}"#).rope_type(), "default");
    assert_eq!(spec(r#"{"factor": 8.0}"#).rope_type(), "default");
    assert_eq!(spec(r#"{"rope_type": "default"}"#).rope_type(), "default");
}

#[test]
fn unknown_keys_are_ignored_rather_than_rejected() {
    // YaRN and longrope blocks carry keys this reader has no field for, and
    // they still have to parse: an unimplemented scheme warns and decodes on
    // the plain table, it does not fail the load.
    let parsed = spec(
        r#"{"beta_fast": 32, "beta_slow": 1, "factor": 40, "mscale": 0.707,
            "original_max_position_embeddings": 4096, "type": "yarn"}"#,
    );
    assert_eq!(parsed.rope_type(), "yarn");
    assert_eq!(parsed.factor, Some(40.0));
}

// Which table each scheme selects.

#[test]
fn default_and_absent_blocks_select_the_plain_table() {
    let absent = RopeScalingKind::resolve(None, 128, 500_000.0, "llama");
    assert!(absent.freqs().is_none());
    assert_eq!(absent.scale(), 1.0);

    let declared = spec(r#"{"rope_type": "default"}"#);
    let kind = RopeScalingKind::resolve(Some(&declared), 128, 500_000.0, "llama");
    assert!(kind.freqs().is_none());
    assert_eq!(kind.scale(), 1.0);
}

#[test]
fn linear_divides_positions_by_factor() {
    let declared = spec(r#"{"factor": 4.0, "type": "linear"}"#);
    let kind = RopeScalingKind::resolve(Some(&declared), 128, 10_000.0, "llama");
    // MLX multiplies the position by `scale`, so dividing by `factor` is
    // `scale = 1 / factor`. The frequencies stay base-derived.
    assert_eq!(kind.scale(), 0.25);
    assert!(kind.freqs().is_none());
}

#[test]
fn llama3_builds_a_half_width_float32_table() {
    let declared = spec(
        r#"{"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3"}"#,
    );
    let kind = RopeScalingKind::resolve(Some(&declared), 128, 500_000.0, "llama");
    let freqs = kind.freqs().expect("llama3 must produce a table");
    assert_eq!(mlxcel_core::array_shape(freqs), vec![64]);
    assert_eq!(mlxcel_core::array_dtype(freqs), mlxcel_core::dtype::FLOAT32);
    // The table replaces the base, so the position scale stays neutral.
    assert_eq!(kind.scale(), 1.0);
}

#[test]
fn an_unimplemented_scheme_keeps_the_model_loading_on_the_plain_table() {
    // #1355 asked for a named load error here. It cannot ship: the shared Llama
    // args are what several VLM loaders parse a `text_config` into, and
    // `models/internvl3-1b` declares exactly this block in that position, so a
    // load error would take a working checkpoint offline over a scheme that is
    // unimplemented either way. The warning is the observability half; this is
    // the half that says the model still loads.
    for block in [
        r#"{"factor": 2.0, "rope_type": "dynamic", "type": "dynamic"}"#,
        r#"{"factor": 40.0, "type": "yarn"}"#,
        r#"{"type": "longrope", "long_factor": [1.0], "short_factor": [1.0]}"#,
        r#"{"type": "su"}"#,
        r#"{"type": "mrope", "mrope_section": [16, 24, 24]}"#,
    ] {
        let declared = spec(block);
        let kind = RopeScalingKind::resolve(Some(&declared), 128, 500_000.0, "qwen2");
        assert!(kind.freqs().is_none(), "{block} must not build a table");
        assert_eq!(kind.scale(), 1.0, "{block} must not scale positions");
    }
}

#[test]
fn a_linear_block_with_an_unusable_factor_falls_back_to_the_plain_table() {
    // A zero or negative factor would make `1 / factor` infinite or flip the
    // sign of every position. Upstream would index `scaling_config["factor"]`
    // and hand the result straight to `nn.RoPE`; falling back keeps the config
    // loading with the graph it already had.
    for block in [
        r#"{"factor": 0.0, "type": "linear"}"#,
        r#"{"factor": -2.0, "type": "linear"}"#,
    ] {
        let declared = spec(block);
        let kind = RopeScalingKind::resolve(Some(&declared), 128, 10_000.0, "llama");
        assert_eq!(kind.scale(), 1.0, "{block}");
        assert!(kind.freqs().is_none(), "{block}");
    }
}

// The llama3 band arithmetic.

#[test]
fn llama3_freqs_match_reference_bands() {
    // Llama 3.1 8B: head_dim 128, rope_theta 500000, factor 8, lf 1, hf 4,
    // original context 8192. Computed here in f64 from the closed form in
    // `Llama3RoPE.__init__` so the assertion does not restate the
    // implementation's own arithmetic.
    let dims = 128usize;
    let base = 500_000.0f32;
    let factor = 8.0f64;
    let low_freq_factor = 1.0f64;
    let high_freq_factor = 4.0f64;
    let old_context_len = 8192.0f64;
    let low_freq_wavelen = old_context_len / low_freq_factor;
    let high_freq_wavelen = old_context_len / high_freq_factor;

    let declared = spec(
        r#"{"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3"}"#,
    );
    let table = to_vec(&llama3_rope_freqs(&declared, dims, base));
    assert_eq!(table.len(), dims / 2);

    let mut low_band = 0usize;
    let mut smooth_band = 0usize;
    let mut high_band = 0usize;

    for (i, &got) in table.iter().enumerate() {
        let plain = (base as f64).powf((2 * i) as f64 / dims as f64);
        let wavelen = 2.0 * std::f64::consts::PI * plain;

        let want = if wavelen > low_freq_wavelen {
            low_band += 1;
            plain * factor
        } else if wavelen > high_freq_wavelen && wavelen < low_freq_wavelen {
            smooth_band += 1;
            let smooth = (old_context_len / wavelen - low_freq_factor)
                / (high_freq_factor - low_freq_factor);
            plain / ((1.0 - smooth) / factor + smooth)
        } else {
            high_band += 1;
            plain
        };

        let relative = ((got as f64) - want).abs() / want.abs().max(1e-30);
        assert!(
            relative < 1e-5,
            "pair {i}: got {got}, want {want} (relative {relative})"
        );
    }

    // All three bands must actually be exercised by this geometry, or the loop
    // above would pass while checking only one branch. Llama 3.1 8B leaves the
    // low pairs untouched, interpolates a handful, and multiplies the tail.
    assert!(high_band > 0 && smooth_band > 0 && low_band > 0);
    assert_eq!(high_band + smooth_band + low_band, dims / 2);
}

#[test]
fn llama3_leaves_the_high_band_alone_and_scales_the_low_band_by_factor() {
    // The two endpoints stated in the issue, checked directly: pair 0 is the
    // highest frequency (shortest wavelength) and must be untouched, and pair 63
    // is the lowest and must be exactly `base^(126/128) * factor`.
    let dims = 128usize;
    let base = 500_000.0f32;
    let declared = spec(
        r#"{"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3"}"#,
    );
    let table = to_vec(&llama3_rope_freqs(&declared, dims, base));

    assert_eq!(table[0], base.powf(0.0));
    assert_eq!(table[63], base.powf(126.0 / 128.0) * 8.0);
}

#[test]
fn a_larger_factor_moves_more_pairs_out_of_the_high_band() {
    // Llama 3.2 1B/3B declare `factor: 32`. The band boundaries do not depend on
    // the factor, so the same pairs change bands; what changes is how far the
    // low band is pushed. Asserting the ratio catches a table that silently
    // ignored `factor`.
    let dims = 128usize;
    let base = 500_000.0f32;
    let eight = to_vec(&llama3_rope_freqs(
        &spec(
            r#"{"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192, "rope_type": "llama3"}"#,
        ),
        dims,
        base,
    ));
    let thirty_two = to_vec(&llama3_rope_freqs(
        &spec(
            r#"{"factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192, "rope_type": "llama3"}"#,
        ),
        dims,
        base,
    ));

    assert_eq!(eight[0], thirty_two[0], "the high band ignores factor");
    let ratio = thirty_two[63] / eight[63];
    assert!(
        (ratio - 4.0).abs() < 1e-4,
        "the low band scales with factor: ratio {ratio}"
    );
}

#[test]
fn the_defaults_match_upstreams_when_the_block_omits_the_band_factors() {
    // `Llama3RoPE.__init__` indexes `factor` and `get`s the rest with 1.0, 4.0
    // and 8192. A block that names only the scheme and the factor must therefore
    // build the same table as one that spells all four out.
    let dims = 64usize;
    let base = 10_000.0f32;
    let terse = to_vec(&llama3_rope_freqs(
        &spec(r#"{"rope_type": "llama3", "factor": 8.0}"#),
        dims,
        base,
    ));
    let explicit = to_vec(&llama3_rope_freqs(
        &spec(
            r#"{"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}"#,
        ),
        dims,
        base,
    ));
    assert_eq!(terse, explicit);
}

#[test]
fn duplicate_hands_out_an_equal_table() {
    // Every decoder layer of a model gets a `duplicate` of one computation.
    // A duplicate that returned a different table would rotate one layer
    // differently from the rest, which produces correctly shaped activations
    // and fluent text.
    let declared = spec(
        r#"{"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3"}"#,
    );
    let kind = RopeScalingKind::resolve(Some(&declared), 128, 500_000.0, "llama");
    let copy = kind.duplicate();

    let original = to_vec(kind.freqs().expect("llama3 table"));
    let duplicated = to_vec(copy.freqs().expect("duplicated llama3 table"));
    assert_eq!(original, duplicated);

    let linear = RopeScalingKind::resolve(
        Some(&spec(r#"{"type": "linear", "factor": 4.0}"#)),
        128,
        10_000.0,
        "llama",
    );
    assert_eq!(linear.duplicate().scale(), linear.scale());
}
