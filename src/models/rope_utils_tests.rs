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

use super::{
    RopeScalingKind, RopeScalingSpec, llama3_rope_freqs, report_unusable_rope_scaling_once,
};

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
    let absent = RopeScalingKind::resolve(None, 128, 500_000.0, None, "llama");
    assert!(absent.freqs().is_none());
    assert_eq!(absent.scale(), 1.0);

    let declared = spec(r#"{"rope_type": "default"}"#);
    let kind = RopeScalingKind::resolve(Some(&declared), 128, 500_000.0, None, "llama");
    assert!(kind.freqs().is_none());
    assert_eq!(kind.scale(), 1.0);
}

#[test]
fn linear_divides_positions_by_factor() {
    let declared = spec(r#"{"factor": 4.0, "type": "linear"}"#);
    let kind = RopeScalingKind::resolve(Some(&declared), 128, 10_000.0, None, "llama");
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
    let kind = RopeScalingKind::resolve(Some(&declared), 128, 500_000.0, None, "llama");
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
        r#"{"type": "longrope", "long_factor": [1.0], "short_factor": [1.0]}"#,
        r#"{"type": "su"}"#,
        r#"{"type": "mrope", "mrope_section": [16, 24, 24]}"#,
    ] {
        let declared = spec(block);
        let kind = RopeScalingKind::resolve(Some(&declared), 128, 500_000.0, None, "qwen2");
        assert!(kind.freqs().is_none(), "{block} must not build a table");
        assert_eq!(kind.scale(), 1.0, "{block} must not scale positions");
    }
}

#[test]
fn an_unimplemented_scheme_warns_once_per_model_and_scheme() {
    let model = "qwen3-warning-dedup-test";
    assert!(report_unusable_rope_scaling_once(
        model,
        "longrope",
        "test-only unsupported scheme"
    ));
    assert!(!report_unusable_rope_scaling_once(
        model,
        "longrope",
        "test-only unsupported scheme"
    ));
    assert!(report_unusable_rope_scaling_once(
        model,
        "dynamic",
        "test-only distinct scheme"
    ));
    assert!(report_unusable_rope_scaling_once(
        "qwen3-moe-warning-dedup-test",
        "longrope",
        "test-only distinct model"
    ));
}

#[test]
fn a_linear_block_with_a_missing_or_unusable_factor_falls_back_to_the_plain_table() {
    // A missing factor would silently no-op, and a zero or negative factor
    // would make `1 / factor` infinite or flip the sign of every position.
    // Upstream would index `scaling_config["factor"]` and hand the result
    // straight to `nn.RoPE`; falling back keeps the config loading with the
    // graph it already had while emitting the once-per-model warning.
    for block in [
        r#"{"type": "linear"}"#,
        r#"{"factor": 0.0, "type": "linear"}"#,
        r#"{"factor": -2.0, "type": "linear"}"#,
    ] {
        let declared = spec(block);
        let kind = RopeScalingKind::resolve(Some(&declared), 128, 10_000.0, None, "llama");
        assert_eq!(kind.scale(), 1.0, "{block}");
        assert!(kind.freqs().is_none(), "{block}");
    }
}

/// Assert that a `llama3` block is rejected and leaves the plain table in place.
///
/// Both halves matter. The table builder must refuse the block, and `resolve`
/// must turn that refusal into the unscaled table rather than into a table with
/// zero, infinite or negative entries in it.
fn assert_llama3_block_falls_back(block: &str) {
    let declared = spec(block);
    let reason = llama3_rope_freqs(&declared, 128, 500_000.0)
        .err()
        .unwrap_or_else(|| panic!("{block} must not build a table"));
    assert!(!reason.is_empty(), "{block} must name what is wrong");

    let kind = RopeScalingKind::resolve(Some(&declared), 128, 500_000.0, None, "llama-3.1-8b-4bit");
    assert!(kind.freqs().is_none(), "{block}");
    assert_eq!(kind.scale(), 1.0, "{block}");
}

#[test]
fn a_llama3_block_with_a_zero_factor_falls_back_to_the_plain_table() {
    // The sharpest case. `freq * factor` is exactly 0.0 for every low-band pair
    // (35 of 64 at this geometry), `fast::rope` divides the position by the
    // table entry, and `reciprocal(0.0)` is `inf`, so every logit is NaN for
    // every token. Nothing on that path throws: the sampler orders logits with
    // `partial_cmp(..).unwrap_or(Equal)`.
    assert_llama3_block_falls_back(
        r#"{"factor": 0.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3"}"#,
    );
}

#[test]
fn a_llama3_block_with_no_numeric_factor_falls_back_instead_of_doing_nothing() {
    // An absent factor, or one written as a JSON string, used to default to 1.0
    // and build a table bit-identical to the plain one. That is the silent
    // no-op #1355 exists to remove, so it is reported rather than defaulted.
    // Upstream expresses the same judgement as a `KeyError` on
    // `scaling_config["factor"]`.
    assert_llama3_block_falls_back(r#"{"rope_type": "llama3"}"#);
    assert_llama3_block_falls_back(r#"{"rope_type": "llama3", "factor": "8.0"}"#);
}

#[test]
fn a_llama3_block_with_an_out_of_range_factor_falls_back_to_the_plain_table() {
    // A JSON number past f32 range saturates to `inf` through the `as` cast, so
    // `freq * factor` is `inf` on the whole low band and that band rotates not
    // at all.
    assert_llama3_block_falls_back(r#"{"rope_type": "llama3", "factor": 1e39}"#);
}

#[test]
fn a_llama3_block_with_a_negative_factor_falls_back_to_the_plain_table() {
    // A negative factor keeps every entry finite and reverses the rotation
    // direction on the low band, which is exactly the class of wrongness that
    // produces correctly shaped activations and fluent text.
    assert_llama3_block_falls_back(r#"{"rope_type": "llama3", "factor": -8.0}"#);
}

#[test]
fn a_llama3_block_with_the_band_factors_equal_still_builds_a_table() {
    // `Llama-4-Scout-17B-16E` declares `low_freq_factor` and `high_freq_factor`
    // both 1.0. That is well defined rather than degenerate: the medium band
    // `wavelen > L / hf && wavelen < L / lf` is empty, so the interpolation
    // whose denominator would be zero is never evaluated, and upstream reaches
    // the same table through `mx.where` discarding the unselected branch. The
    // screening must not reject it.
    let declared = spec(
        r#"{"factor": 16.0, "high_freq_factor": 1.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3"}"#,
    );
    let table = to_vec(&llama3_rope_freqs(&declared, 128, 500_000.0).expect("a valid block"));
    assert_eq!(table.len(), 64);
    assert!(
        table.iter().all(|f| f.is_finite() && *f > 0.0),
        "every entry must stay a positive finite frequency"
    );
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
    let table = to_vec(&llama3_rope_freqs(&declared, dims, base).expect("a valid block"));
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
    let table = to_vec(&llama3_rope_freqs(&declared, dims, base).expect("a valid block"));

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
    let eight = to_vec(
        &llama3_rope_freqs(
            &spec(
                r#"{"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192, "rope_type": "llama3"}"#,
            ),
            dims,
            base,
        )
        .expect("a valid block"),
    );
    let thirty_two = to_vec(
        &llama3_rope_freqs(
            &spec(
                r#"{"factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192, "rope_type": "llama3"}"#,
            ),
            dims,
            base,
        )
        .expect("a valid block"),
    );

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
    let terse = to_vec(
        &llama3_rope_freqs(
            &spec(r#"{"rope_type": "llama3", "factor": 8.0}"#),
            dims,
            base,
        )
        .expect("a valid block"),
    );
    let explicit = to_vec(
        &llama3_rope_freqs(
            &spec(
                r#"{"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}"#,
            ),
            dims,
            base,
        )
        .expect("a valid block"),
    );
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
    let kind = RopeScalingKind::resolve(Some(&declared), 128, 500_000.0, None, "llama");
    let copy = kind.duplicate();

    let original = to_vec(kind.freqs().expect("llama3 table"));
    let duplicated = to_vec(copy.freqs().expect("duplicated llama3 table"));
    assert_eq!(original, duplicated);

    let linear = RopeScalingKind::resolve(
        Some(&spec(r#"{"type": "linear", "factor": 4.0}"#)),
        128,
        10_000.0,
        None,
        "llama",
    );
    assert_eq!(linear.duplicate().scale(), linear.scale());
}

// The YaRN arm (#1472).

#[test]
fn a_yarn_block_builds_a_half_width_table_with_the_temperature_mscale() {
    let declared = spec(
        r#"{"rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 32768}"#,
    );
    let kind = RopeScalingKind::resolve(Some(&declared), 128, 500_000.0, None, "qwen2");
    let freqs = kind.freqs().expect("a yarn block builds a table");
    assert_eq!(mlxcel_core::array_shape(freqs), vec![64]);
    assert_eq!(kind.scale(), 1.0, "yarn rotates at unit position scale");

    // The default magnitude correction is YaRN's sqrt(t) = 0.1 ln(s) + 1.
    let expected = 0.1 * 4.0f32.ln() + 1.0;
    assert!(
        (kind.attn_scale() - expected).abs() < 1e-6,
        "attn_scale {} != {expected}",
        kind.attn_scale()
    );
}

#[test]
fn yarn_interpolates_the_low_band_and_extrapolates_the_high_band() {
    let factor = 8.0f32;
    let declared = spec(
        r#"{"rope_type": "yarn", "factor": 8.0,
            "original_max_position_embeddings": 4096}"#,
    );
    let kind = RopeScalingKind::resolve(Some(&declared), 128, 10_000.0, None, "qwen2");
    let table = to_vec(kind.freqs().expect("table"));
    let plain: Vec<f32> = (0..64)
        .map(|i| 10_000.0f32.powf(2.0 * i as f32 / 128.0))
        .collect();

    // Pair 0 rotates fastest: fully extrapolated, so the divisor is unchanged.
    assert!(
        (table[0] - plain[0]).abs() < 1e-6,
        "{} != {}",
        table[0],
        plain[0]
    );
    // The slowest pair is fully interpolated: divisor scaled by the factor.
    let last = table.len() - 1;
    assert!(
        (table[last] - plain[last] * factor).abs() / (plain[last] * factor) < 1e-5,
        "{} != {}",
        table[last],
        plain[last] * factor
    );
    // The table is monotonically non-decreasing relative to the plain one.
    for (t, p) in table.iter().zip(&plain) {
        assert!(*t >= *p * 0.999, "yarn must never speed a rotation up");
        assert!(
            *t <= *p * factor * 1.001,
            "yarn must never exceed full interpolation"
        );
    }
}

#[test]
fn a_yarn_block_with_deepseek_mscale_keys_uses_the_ratio() {
    let declared = spec(
        r#"{"rope_type": "yarn", "factor": 40.0, "mscale": 1.0, "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096}"#,
    );
    let kind = RopeScalingKind::resolve(Some(&declared), 128, 10_000.0, None, "probe");
    // mscale == mscale_all_dim makes the ratio exactly 1.0.
    assert_eq!(kind.attn_scale(), 1.0);
}

#[test]
fn a_yarn_block_missing_its_factor_falls_back_to_the_plain_table() {
    let declared = spec(r#"{"rope_type": "yarn"}"#);
    let kind = RopeScalingKind::resolve(Some(&declared), 128, 10_000.0, None, "probe");
    assert!(kind.freqs().is_none());
    assert_eq!(kind.attn_scale(), 1.0);
}

#[test]
fn yarn_duplicate_hands_out_an_equal_table_and_mscale() {
    let declared = spec(
        r#"{"rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 8192}"#,
    );
    let kind = RopeScalingKind::resolve(Some(&declared), 64, 500_000.0, None, "probe");
    let copy = kind.duplicate();
    assert_eq!(
        to_vec(kind.freqs().expect("table")),
        to_vec(copy.freqs().expect("copy"))
    );
    assert_eq!(kind.attn_scale(), copy.attn_scale());
}

#[test]
fn yarn_orig_ctx_falls_back_to_the_training_context_then_4096() {
    // The same factor over a longer original context moves the correction band
    // to higher pair indices, so more of the table stays extrapolated.
    let block = r#"{"rope_type": "yarn", "factor": 8.0}"#;
    let short = to_vec(
        RopeScalingKind::resolve(Some(&spec(block)), 128, 10_000.0, Some(4096.0), "probe")
            .freqs()
            .expect("table"),
    );
    let long = to_vec(
        RopeScalingKind::resolve(Some(&spec(block)), 128, 10_000.0, Some(131_072.0), "probe")
            .freqs()
            .expect("table"),
    );
    assert!(
        short.iter().zip(&long).any(|(s, l)| (s - l).abs() > 1e-3),
        "the original context must move the band"
    );
    // With no training context supplied, the 4096 default reproduces the
    // 4096-trained table, which is also upstream YarnRoPE's default.
    let defaulted = to_vec(
        RopeScalingKind::resolve(Some(&spec(block)), 128, 10_000.0, None, "probe")
            .freqs()
            .expect("table"),
    );
    assert_eq!(short, defaulted);
}

// YaRN `truncate` (#3215, ml-explore/mlx-lm#1931).

/// One YaRN geometry of the sweep the `truncate` tests share.
#[derive(Debug, Clone, Copy)]
struct YarnCase {
    dims: usize,
    factor: f32,
    base: f32,
    original_ctx: f32,
    beta_fast: f32,
    beta_slow: f32,
}

/// The Qwen 3.5 geometry (64 rotary dims at a base of 1e7).
const QWEN35_YARN_CASE: YarnCase = YarnCase {
    dims: 64,
    factor: 2.0,
    base: 1e7,
    original_ctx: 262144.0,
    beta_fast: 32.0,
    beta_slow: 1.0,
};

/// The dims, factors and bases the issue names, over three original
/// contexts (24 cases), then the Qwen 3.5 geometry, a zero-width band and a
/// band that starts below pair 0.
fn yarn_cases() -> Vec<YarnCase> {
    let mut cases = Vec::new();
    for dims in [64, 128] {
        for factor in [4.0, 32.0] {
            for base in [1e4, 1.5e5] {
                for original_ctx in [4096.0, 32768.0, 262144.0] {
                    cases.push(YarnCase {
                        dims,
                        factor,
                        base,
                        original_ctx,
                        beta_fast: 32.0,
                        beta_slow: 1.0,
                    });
                }
            }
        }
    }
    cases.push(QWEN35_YARN_CASE);
    cases.push(YarnCase {
        beta_fast: 8.0,
        beta_slow: 8.0,
        ..QWEN35_YARN_CASE
    });
    cases.push(YarnCase {
        original_ctx: 16.0,
        ..QWEN35_YARN_CASE
    });
    cases
}

/// Bands the two YaRN builders in the tree resolve differently, so only the
/// bit-identity test walks them: an inverted band (`beta_fast < beta_slow`)
/// and a band wholly above the table. This module steps at `low` in both,
/// as it did before `truncate` was read; `gpt_oss_common::yarn_rope_freqs`
/// and upstream's `clip` invert the ramp instead.
fn yarn_edge_cases() -> Vec<YarnCase> {
    vec![
        YarnCase {
            beta_fast: 1.0,
            beta_slow: 32.0,
            ..QWEN35_YARN_CASE
        },
        YarnCase {
            original_ctx: 1e30,
            ..QWEN35_YARN_CASE
        },
    ]
}

fn yarn_spec(case: YarnCase, truncate: Option<bool>) -> RopeScalingSpec {
    RopeScalingSpec {
        rope_type: Some("yarn".to_string()),
        factor: Some(case.factor),
        original_max_position_embeddings: Some(case.original_ctx),
        beta_fast: Some(case.beta_fast),
        beta_slow: Some(case.beta_slow),
        truncate,
        ..RopeScalingSpec::default()
    }
}

fn yarn_table(case: YarnCase, truncate: Option<bool>) -> Vec<f32> {
    let (freqs, _) = super::yarn_rope_setup(&yarn_spec(case, truncate), case.dims, case.base, None)
        .unwrap_or_else(|reason| panic!("{case:?} must build a table: {reason}"));
    to_vec(&freqs)
}

/// The table as this module built it before `truncate` was read: the
/// correction band floored and ceiled to `usize`, and an integer ramp. Kept
/// verbatim as the oracle for `yarn_truncate_default_is_bit_identical`.
fn yarn_table_before_truncate(case: YarnCase) -> Vec<f32> {
    let YarnCase {
        dims,
        factor,
        base,
        original_ctx,
        beta_fast,
        beta_slow,
    } = case;
    let low = super::yarn_find_correction_dim(beta_fast, dims, base, original_ctx).floor() as i64;
    let high = super::yarn_find_correction_dim(beta_slow, dims, base, original_ctx).ceil() as i64;
    let (low, high) = (
        low.max(0) as usize,
        (high.max(0) as usize).min(dims.saturating_sub(1)),
    );

    (0..dims / 2)
        .map(|i| {
            let exp = (2 * i) as f32 / dims as f32;
            let fe = base.powf(exp);
            let fi = factor * fe;
            let ramp = if i < low {
                1.0
            } else if i > high {
                0.0
            } else {
                let t = (i - low) as f32 / (high - low).max(1) as f32;
                1.0 - t.clamp(0.0, 1.0)
            };
            // The extrapolation mix is 1 with no `--yarn-ext-factor` installed.
            let mix = ramp;
            (fi * fe) / (fi * mix + fe * (1.0 - mix))
        })
        .collect()
}

#[test]
fn yarn_truncate_default_is_bit_identical() {
    for case in yarn_cases().into_iter().chain(yarn_edge_cases()) {
        let before: Vec<u32> = yarn_table_before_truncate(case)
            .into_iter()
            .map(f32::to_bits)
            .collect();
        for truncate in [None, Some(true)] {
            let after: Vec<u32> = yarn_table(case, truncate)
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_eq!(after, before, "{case:?} truncate {truncate:?}");
        }
    }
}

#[test]
fn yarn_truncate_false_matches_gpt_oss_builder() {
    use crate::models::gpt_oss_common::{YarnRopeParams, yarn_rope_freqs};

    let mut changed = 0;
    for case in yarn_cases() {
        let fractional = yarn_table(case, Some(false));
        let (reference, _) = yarn_rope_freqs(&YarnRopeParams {
            dims: case.dims,
            base: case.base,
            factor: case.factor,
            original_max_position_embeddings: case.original_ctx,
            beta_fast: case.beta_fast,
            beta_slow: case.beta_slow,
            mscale: 1.0,
            mscale_all_dim: 0.0,
            truncate: false,
        });
        assert_eq!(fractional.len(), reference.len(), "{case:?}");
        for (pair, (ours, theirs)) in fractional.iter().zip(&reference).enumerate() {
            assert!(
                (ours - theirs).abs() <= 1e-6 * theirs.abs(),
                "{case:?} pair {pair}: {ours} != {theirs}"
            );
        }
        if fractional != yarn_table(case, Some(true)) {
            changed += 1;
        }
    }
    // `truncate: false` is not a no-op: the fractional band moves the ramp on
    // every geometry the issue names (the 24 leading cases).
    assert!(changed >= 24, "only {changed} tables changed");
}

#[test]
fn reads_truncate_flag() {
    let block = |truncate: &str| {
        spec(&format!(
            r#"{{"rope_type": "yarn", "factor": 4.0, "truncate": {truncate}}}"#
        ))
    };
    assert_eq!(block("false").truncate, Some(false));
    assert_eq!(block("true").truncate, Some(true));
    // A string is not a flag: it reads as absent, which is upstream's `true`.
    assert_eq!(block(r#""false""#).truncate, None);
    assert_eq!(block("0").truncate, None);
    assert_eq!(block("null").truncate, None);
    assert_eq!(
        spec(r#"{"rope_type": "yarn", "factor": 4.0}"#).truncate,
        None
    );

    // The flag reaches the table through `resolve`, the path a checkpoint takes.
    let table = |truncate: &str| {
        let declared = spec(&format!(
            r#"{{"rope_type": "yarn", "factor": 4.0, "truncate": {truncate},
                "original_max_position_embeddings": 4096}}"#
        ));
        to_vec(
            RopeScalingKind::resolve(Some(&declared), 128, 10_000.0, None, "probe")
                .freqs()
                .expect("table"),
        )
    };
    assert_ne!(table("false"), table("true"));
    assert_eq!(table(r#""false""#), table("true"));
}

/// PLaMo 3's full-attention geometry (issue #2374): head_dim 128, base 1e6,
/// factor 256 over an original 4096. Its fractional band is where
/// `truncate: false` changes the table, while an absent flag still builds the
/// floored-and-ceiled table bit for bit.
#[test]
fn yarn_truncate_toggle_on_the_plamo3_geometry() {
    let case = YarnCase {
        dims: 128,
        factor: 256.0,
        base: 1e6,
        original_ctx: 4096.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let bits = |table: Vec<f32>| table.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
    let truncated = bits(yarn_table(case, Some(true)));
    assert_eq!(bits(yarn_table(case, None)), truncated);
    assert_eq!(bits(yarn_table_before_truncate(case)), truncated);
    let fractional = bits(yarn_table(case, Some(false)));
    assert_ne!(fractional, truncated);
}
