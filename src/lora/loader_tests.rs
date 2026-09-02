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

//! Unit tests for LoRA delta computation and weight fusion.
//!
//! The layout tests run against the same GPT-2 checkpoint fixtures the model
//! loader is tested with (`src/models/gpt2_tests.rs`), rather than against
//! hand-rolled shapes. That matters here: the defect these tests cover is that
//! a Conv1D-layout checkpoint has *plausible* shapes, so a fixture invented
//! locally would be assuming the very thing under test. Reusing
//! `raw_hf_weights` means the map is the one `Gpt2Layout::detect` is separately
//! proven to classify as a raw HuggingFace export, and each test re-asserts that
//! classification before exercising fusion.

use super::*;
use crate::lora::test_support::{adapter_pair_tensors, ones_tensor, temp_dir, write_adapter_dir};
use crate::models::gpt2::Gpt2Layout;
use crate::models::gpt2::tests::{mlx_converted_weights, raw_hf_weights, tiny_args};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build an mlx-lm-convention adapter pair for `layer`: `lora_a` is
/// `[in_features, rank]` and `lora_b` is `[rank, out_features]`, both all ones,
/// so the resulting delta is `scale * rank` in every position and a fused sum is
/// exactly predictable.
fn lora_pair(layer: &str, in_features: i32, rank: i32, out_features: i32) -> WeightMap {
    let mut adapter = WeightMap::new();
    adapter.insert(
        format!("{layer}.lora_a"),
        mlxcel_core::ones(&[in_features, rank], mlxcel_core::dtype::FLOAT32),
    );
    adapter.insert(
        format!("{layer}.lora_b"),
        mlxcel_core::ones(&[rank, out_features], mlxcel_core::dtype::FLOAT32),
    );
    adapter
}

/// The error text of a call that must fail.
///
/// `Result::expect_err` needs `T: Debug`, and a [`WeightMap`] holds
/// `UniquePtr<MlxArray>`, which is not `Debug`.
fn fusion_error<T>(result: Result<T>, what: &str) -> String {
    match result {
        Ok(_) => panic!("{what}"),
        Err(err) => err.to_string(),
    }
}

fn tensor_sum(weight: &MlxArray) -> f32 {
    mlxcel_core::eval(weight);
    let sum = mlxcel_core::sum_all(weight);
    mlxcel_core::eval(&sum);
    mlxcel_core::item_f32(&sum)
}

// ---------------------------------------------------------------------------
// Delta orientation
// ---------------------------------------------------------------------------

#[test]
fn test_compute_lora_delta_mlx_format() {
    // mlx-lm format: a=(in=4, rank=2), b=(rank=2, out=3)
    // Result should be (out=3, in=4)
    let lora_a = mlxcel_core::from_slice_f32(&[1.0f32; 8], &[4, 2]);
    let lora_b = mlxcel_core::from_slice_f32(&[1.0f32; 6], &[2, 3]);
    let scale = 1.0;

    let delta = compute_lora_delta(&lora_a, &lora_b, scale).unwrap();
    assert_eq!(mlxcel_core::array_shape(&delta), vec![3, 4]);
}

#[test]
fn test_compute_lora_delta_peft_format() {
    // PEFT format: a=(rank=2, in=4), b=(out=3, rank=2)
    // Result should be (out=3, in=4)
    let lora_a = mlxcel_core::from_slice_f32(&[1.0f32; 8], &[2, 4]);
    let lora_b = mlxcel_core::from_slice_f32(&[1.0f32; 6], &[3, 2]);
    let scale = 1.0;

    let delta = compute_lora_delta(&lora_a, &lora_b, scale).unwrap();
    assert_eq!(mlxcel_core::array_shape(&delta), vec![3, 4]);
}

#[test]
fn test_fuse_lora_weights_basic() {
    // Create base weights
    let mut base_weights = WeightMap::new();
    base_weights.insert(
        "layer.weight".to_string(),
        mlxcel_core::ones(&[3, 4], mlxcel_core::dtype::FLOAT32),
    );

    // Create adapter weights (mlx-lm format)
    let mut adapter_weights = WeightMap::new();
    adapter_weights.insert(
        "layer.lora_a".to_string(),
        mlxcel_core::ones(&[4, 2], mlxcel_core::dtype::FLOAT32),
    );
    adapter_weights.insert(
        "layer.lora_b".to_string(),
        mlxcel_core::ones(&[2, 3], mlxcel_core::dtype::FLOAT32),
    );

    let fused = fuse_lora_weights(&base_weights, &adapter_weights, 1.0).unwrap();

    // Should have the same key
    assert!(fused.contains_key("layer.weight"));
    let fused_weight = fused.get("layer.weight").unwrap();
    let shape = mlxcel_core::array_shape(fused_weight);
    assert_eq!(shape, vec![3, 4]);

    // Original was all 1s, delta should be scale * (lora_b.T @ lora_a.T) = 2s matrix
    // So fused should be > 1.0
    assert!(tensor_sum(fused_weight) > 12.0); // 3*4 = 12 base + delta > 0
}

// ---------------------------------------------------------------------------
// Conv1D projection recognition
// ---------------------------------------------------------------------------

#[test]
fn conv1d_projection_suffix_matches_only_transformer_block_projections() {
    for key in [
        "h.0.attn.c_attn.weight",
        "transformer.h.11.attn.c_proj.weight",
        "model.h.3.mlp.c_fc.weight",
        "model.transformer.h.7.mlp.c_proj.weight",
    ] {
        assert!(
            conv1d_projection_suffix(key).is_some(),
            "should match: {key}"
        );
    }

    for key in [
        // No block index at all.
        "attn.c_proj.weight",
        // `h.` is not a path segment here.
        "blah.0.attn.c_proj.weight",
        // A non-numeric block index.
        "h.x.attn.c_proj.weight",
        // Biases are 1-D and are never transposed at load.
        "h.0.attn.c_proj.bias",
        // A different family's projection.
        "model.layers.0.self_attn.q_proj.weight",
        "h.0.ln_1.weight",
    ] {
        assert!(
            conv1d_projection_suffix(key).is_none(),
            "should not match: {key}"
        );
    }
}

#[test]
fn conv1d_layout_detection_reads_the_fused_qkv_projection() {
    let args = tiny_args();

    let raw = raw_hf_weights(&args, "");
    let evidence = detect_conv1d_projection_layout(&raw).expect("raw export is Conv1D");
    assert_eq!(evidence.key, "h.0.attn.c_attn.weight");
    assert_eq!(
        evidence.shape,
        vec![args.n_embd as i32, 3 * args.n_embd as i32]
    );

    let converted = mlx_converted_weights(&args);
    assert!(
        detect_conv1d_projection_layout(&converted).is_none(),
        "an already-transposed conversion must not be flagged"
    );

    // A checkpoint from any other family has no `c_attn` at all.
    let mut other = WeightMap::new();
    other.insert(
        "model.layers.0.self_attn.q_proj.weight".into(),
        mlxcel_core::ones(&[4, 4], mlxcel_core::dtype::FLOAT32),
    );
    assert!(detect_conv1d_projection_layout(&other).is_none());
}

// ---------------------------------------------------------------------------
// Fusion against a real Conv1D-layout GPT-2 checkpoint
// ---------------------------------------------------------------------------

#[test]
fn conv1d_checkpoint_rejects_fusion_into_the_square_attention_c_proj() {
    // The case that produces no error today: base and delta are both
    // [n_embd, n_embd], the add broadcasts, and the model silently runs
    // `W + Dᵀ` after `conv1d_linear` transposes at construction time.
    let args = tiny_args();
    let mut base = raw_hf_weights(&args, "");
    assert!(
        Gpt2Layout::detect(&base, &args)
            .expect("layout detected")
            .conv1d,
        "fixture must be a genuine raw HuggingFace Conv1D export"
    );

    let h = args.n_embd as i32;
    let key = "h.0.attn.c_proj.weight";
    let before = tensor_sum(base.get(key).unwrap());

    // An mlxcel-convention adapter on the bare GPT-2 projection path, which is
    // exactly the artifact that reaches this code.
    let adapter = lora_pair("h.0.attn.c_proj", h, 2, h);
    assert_eq!(
        mlxcel_core::array_shape(base.get(key).unwrap()),
        mlxcel_core::array_shape(
            &compute_lora_delta(
                adapter.get("h.0.attn.c_proj.lora_a").unwrap(),
                adapter.get("h.0.attn.c_proj.lora_b").unwrap(),
                1.0,
            )
            .unwrap()
        ),
        "the shapes agree, which is why a shape check alone cannot catch this"
    );

    let err = fuse_lora_weights_into(&mut base, &adapter, 1.0)
        .expect_err("a Conv1D-layout square projection must not fuse")
        .to_string();
    assert!(err.contains("Conv1D"), "{err}");
    assert!(err.contains(key), "{err}");
    assert!(
        err.contains("h.0.attn.c_attn.weight"),
        "the error must name the evidence key: {err}"
    );

    let after = tensor_sum(base.get(key).unwrap());
    assert!(
        (before - after).abs() < 1e-6,
        "the base weight must be left untouched: {before} -> {after}"
    );
}

#[test]
fn conv1d_checkpoint_rejects_every_non_square_projection_without_aborting() {
    // On an unguarded build each of these reaches `mlxcel_core::add` with
    // non-broadcastable operands, and the MLX C++ throw crosses a cxx shim that
    // is not declared fallible, so the whole test binary dies with SIGABRT.
    // Reaching the assertions at all is the evidence.
    let args = tiny_args();
    let h = args.n_embd as i32;
    let ff = args.intermediate_size() as i32;

    // (adapter layer, in_features, out_features) in Linear terms.
    let cases = [
        ("h.0.attn.c_attn", h, 3 * h),
        ("h.0.mlp.c_fc", h, ff),
        ("h.0.mlp.c_proj", ff, h),
    ];

    for (layer, in_features, out_features) in cases {
        let mut base = raw_hf_weights(&args, "");
        let adapter = lora_pair(layer, in_features, 2, out_features);

        let err = match fuse_lora_weights_into(&mut base, &adapter, 0.5) {
            Ok(fused) => panic!("{layer} must not fuse ({fused} applied)"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("Conv1D"), "{layer}: {err}");
        assert!(err.contains(&format!("{layer}.weight")), "{layer}: {err}");
    }
}

#[test]
fn conv1d_checkpoint_reports_every_affected_projection_at_once() {
    // A whole-adapter report, sorted, so the diagnostic does not depend on
    // `WeightMap` iteration order.
    let args = tiny_args();
    let h = args.n_embd as i32;
    let mut base = raw_hf_weights(&args, "transformer.");

    let mut adapter = lora_pair("transformer.h.0.attn.c_proj", h, 2, h);
    adapter.extend(lora_pair("transformer.h.0.attn.c_attn", h, 2, 3 * h));

    let err = fuse_lora_weights_into(&mut base, &adapter, 1.0)
        .expect_err("must not fuse")
        .to_string();
    assert!(
        err.contains("transformer.h.0.attn.c_attn.weight, transformer.h.0.attn.c_proj.weight"),
        "{err}"
    );
}

#[test]
fn conv1d_checkpoint_still_fuses_an_adapter_that_targets_nothing_transposed() {
    // The verdict is whole-map, but the rejection is per-target: an adapter that
    // never lands on a Conv1D-stored projection is unaffected.
    let args = tiny_args();
    let mut base = raw_hf_weights(&args, "");
    let vocab = args.vocab_size as i32;
    let h = args.n_embd as i32;
    let before = tensor_sum(base.get("wte.weight").unwrap());

    let rank = 2;
    let adapter = lora_pair("wte", h, rank, vocab);
    fuse_lora_weights_into(&mut base, &adapter, 0.5).expect("embedding fusion is unaffected");

    let after = tensor_sum(base.get("wte.weight").unwrap());
    let expected = before + 0.5 * rank as f32 * (vocab * h) as f32;
    assert!(
        (after - expected).abs() < 1e-2,
        "after={after}, expected={expected}"
    );
}

#[test]
fn an_already_transposed_gpt2_checkpoint_still_fuses_correctly() {
    // The guard must not misfire on the layout adapters are actually published
    // against: an MLX conversion, whose projections are already [out, in].
    let args = tiny_args();
    let mut base = mlx_converted_weights(&args);
    assert!(
        !Gpt2Layout::detect(&base, &args)
            .expect("layout detected")
            .conv1d
    );

    let h = args.n_embd as i32;
    let key = "model.h.0.attn.c_proj.weight";
    let before = tensor_sum(base.get(key).unwrap());

    let rank = 2;
    let adapter = lora_pair("model.h.0.attn.c_proj", h, rank, h);
    fuse_lora_weights_into(&mut base, &adapter, 0.5).expect("fusion succeeds");

    // lora_a and lora_b are all ones, so every delta entry is scale * rank.
    let after = tensor_sum(base.get(key).unwrap());
    let expected = before + 0.5 * rank as f32 * (h * h) as f32;
    assert!(
        (after - expected).abs() < 1e-3,
        "after={after}, expected={expected}"
    );
}

// ---------------------------------------------------------------------------
// Generic shape guard (families with no Conv1D signal at all)
// ---------------------------------------------------------------------------

#[test]
fn a_transposed_base_weight_is_reported_instead_of_reaching_mlx_add() {
    // No `c_attn` anywhere, so only the generic shape guard can catch this.
    let mut base = WeightMap::new();
    base.insert(
        "model.layers.0.self_attn.q_proj.weight".into(),
        mlxcel_core::ones(&[8, 4], mlxcel_core::dtype::FLOAT32),
    );
    // a=[in=8, rank=2], b=[rank=2, out=4] gives a [4, 8] delta.
    let adapter = lora_pair("model.layers.0.self_attn.q_proj", 8, 2, 4);

    let err = fuse_lora_weights_into(&mut base, &adapter, 1.0)
        .expect_err("a transposed base weight must not fuse")
        .to_string();
    assert!(err.contains("[8, 4]") && err.contains("[4, 8]"), "{err}");
    assert!(err.contains("transposes"), "{err}");
    assert!(
        err.contains("model.layers.0.self_attn.q_proj.weight"),
        "{err}"
    );

    let unchanged = tensor_sum(base.get("model.layers.0.self_attn.q_proj.weight").unwrap());
    assert!((unchanged - 32.0).abs() < 1e-6, "base drifted: {unchanged}");
}

#[test]
fn a_mismatch_that_is_not_a_transpose_omits_the_transpose_hint() {
    let mut base = WeightMap::new();
    base.insert(
        "model.layers.0.mlp.up_proj.weight".into(),
        mlxcel_core::ones(&[8, 4], mlxcel_core::dtype::FLOAT32),
    );
    // a=[in=4, rank=2], b=[rank=2, out=4] gives a [4, 4] delta, which would
    // broadcast against [8, 4] and silently corrupt every row.
    let adapter = lora_pair("model.layers.0.mlp.up_proj", 4, 2, 4);

    let err = fuse_lora_weights_into(&mut base, &adapter, 1.0)
        .expect_err("a broadcastable mismatch must not fuse either")
        .to_string();
    assert!(err.contains("[8, 4]") && err.contains("[4, 4]"), "{err}");
    assert!(!err.contains("transposes"), "{err}");
}

// ---------------------------------------------------------------------------
// Adapter tensor validation (issue #1328)
//
// Every case below used to be a `warn!` and a `continue`: the load succeeded,
// the "fused into N layers" line counted the skipped tensors anyway, and the
// server answered from base weights under the adapter's name.
// ---------------------------------------------------------------------------

/// A base weight map holding one `[out, in]` projection per named layer.
fn base_map(layers: &[(&str, i32, i32)]) -> WeightMap {
    let mut base = WeightMap::new();
    for (layer, out_features, in_features) in layers {
        base.insert(
            format!("{layer}.weight"),
            mlxcel_core::ones(&[*out_features, *in_features], mlxcel_core::dtype::FLOAT32),
        );
    }
    base
}

#[test]
fn adapter_pair_without_base_weight_is_an_error() {
    let base = base_map(&[("layer", 3, 4)]);
    let adapter = lora_pair("other", 4, 2, 3);

    let err = fusion_error(
        fuse_lora_weights(&base, &adapter, 1.0),
        "an adapter layer the model does not have must not load",
    );
    assert!(err.contains("other.lora_a"), "{err}");
    assert!(err.contains("no base weight"), "{err}");
    assert!(err.contains("tried other.weight"), "{err}");
}

#[test]
fn adapter_with_unknown_leaf_is_an_error() {
    // A DoRA magnitude vector next to a well-formed pair: the pair alone would
    // fuse cleanly and produce weights that are neither base nor fine-tune.
    let base = base_map(&[("layer", 3, 4)]);
    let mut adapter = lora_pair("layer", 4, 2, 3);
    adapter.insert(
        "layer.m".to_string(),
        mlxcel_core::ones(&[3], mlxcel_core::dtype::FLOAT32),
    );

    let err = fusion_error(
        fuse_lora_weights(&base, &adapter, 1.0),
        "an unrecognised adapter tensor must not be dropped",
    );
    assert!(err.contains("layer.m"), "{err}");
    assert!(err.contains("not a LoRA tensor"), "{err}");
}

#[test]
fn incomplete_pair_is_an_error() {
    let base = base_map(&[("layer", 3, 4)]);
    let mut adapter = WeightMap::new();
    adapter.insert(
        "layer.lora_a".to_string(),
        mlxcel_core::ones(&[4, 2], mlxcel_core::dtype::FLOAT32),
    );

    let err = fusion_error(
        fuse_lora_weights(&base, &adapter, 1.0),
        "half a pair cannot be applied",
    );
    assert!(err.contains("layer.lora_a"), "{err}");
    assert!(err.contains("layer.lora_b"), "{err}");
    assert!(err.contains("incomplete pair"), "{err}");

    // The mirror case: a lone `lora_b` names the missing `lora_a`.
    let mut adapter = WeightMap::new();
    adapter.insert(
        "layer.lora_b".to_string(),
        mlxcel_core::ones(&[2, 3], mlxcel_core::dtype::FLOAT32),
    );
    let err = fusion_error(
        fuse_lora_weights(&base, &adapter, 1.0),
        "half a pair cannot be applied",
    );
    assert!(err.contains("layer.lora_b: incomplete pair"), "{err}");
    assert!(err.contains("missing layer.lora_a"), "{err}");
}

#[test]
fn all_violations_are_reported_together() {
    // An adapter built for another architecture fails on every layer it
    // carries. Reporting one at a time would need as many load attempts as the
    // model has layers, so the report is exhaustive and sorted (`WeightMap` is
    // a `HashMap`, so an unsorted report would reorder run to run).
    let base = base_map(&[("kept", 3, 4)]);
    let mut adapter = lora_pair("missing_one", 4, 2, 3);
    adapter.extend(lora_pair("missing_two", 4, 2, 3));
    adapter.insert(
        "kept.stray".to_string(),
        mlxcel_core::ones(&[3], mlxcel_core::dtype::FLOAT32),
    );

    let err = fusion_error(fuse_lora_weights(&base, &adapter, 1.0), "must not load");
    assert!(err.contains("3 adapter tensors cannot be applied"), "{err}");

    let kept = err.find("kept.stray").expect("stray leaf reported");
    let one = err
        .find("missing_one.lora_a")
        .expect("first layer reported");
    let two = err
        .find("missing_two.lora_a")
        .expect("second layer reported");
    assert!(kept < one && one < two, "report must be sorted: {err}");
}

#[test]
fn fused_count_reports_applied_pairs() {
    let mut base = base_map(&[("a", 3, 4), ("b", 5, 6)]);
    let mut adapter = lora_pair("a", 4, 2, 3);
    adapter.extend(lora_pair("b", 6, 2, 5));

    let fused = fuse_lora_weights_into(&mut base, &adapter, 1.0).expect("both pairs apply");
    assert_eq!(fused, 2);
}

#[test]
fn a_peft_base_layer_pair_resolves_to_the_wrapped_weight() {
    // HuggingFace PEFT wraps the frozen projection, so `<layer>.base_layer`
    // has to resolve to `<layer>.weight`. Dropping the no-match fallback must
    // not lose that.
    let mut base = base_map(&[("model.layers.0.self_attn.q_proj", 3, 4)]);
    let before = tensor_sum(base.get("model.layers.0.self_attn.q_proj.weight").unwrap());

    let adapter = lora_pair("model.layers.0.self_attn.q_proj.base_layer", 4, 2, 3);
    let fused = fuse_lora_weights_into(&mut base, &adapter, 0.5).expect("PEFT naming resolves");
    assert_eq!(fused, 1);

    let after = tensor_sum(base.get("model.layers.0.self_attn.q_proj.weight").unwrap());
    let expected = before + 0.5 * 2.0 * (3 * 4) as f32;
    assert!(
        (after - expected).abs() < 1e-3,
        "after={after}, expected={expected}"
    );
}

// ---------------------------------------------------------------------------
// Adapter directory acceptance (`apply_lora_adapters`)
// ---------------------------------------------------------------------------

#[test]
fn dora_adapter_is_refused() {
    // The pair itself is valid, so nothing but the declared type can catch
    // this: applying it would drop the magnitude vectors the checkpoint's own
    // tooling folds in, and the result matches neither base nor fine-tune.
    let dir = temp_dir("dora");
    write_adapter_dir(&dir, "dora", 2, adapter_pair_tensors("layer", 4, 2, 3));

    let base = base_map(&[("layer", 3, 4)]);
    let err = fusion_error(
        apply_lora_adapters(&base, &dir),
        "a DoRA adapter must not be applied as LoRA",
    );
    assert!(err.contains("DoRA"), "{err}");
    assert!(err.contains("dora"), "{err}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn full_fine_tune_adapter_is_refused() {
    let dir = temp_dir("full");
    write_adapter_dir(&dir, "full", 2, adapter_pair_tensors("layer", 4, 2, 3));

    let base = base_map(&[("layer", 3, 4)]);
    let err = fusion_error(
        apply_lora_adapters(&base, &dir),
        "a full fine-tune is not an adapter",
    );
    assert!(err.contains("not LoRA type"), "{err}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_adapter_that_applies_nothing_is_refused() {
    // Every tensor is either applied or reported, so an empty file is the one
    // way to reach zero. It is still a failed load: the model would serve base
    // weights while the startup log said an adapter was in place.
    let dir = temp_dir("empty");
    write_adapter_dir(&dir, "lora", 2, std::collections::HashMap::new());

    let base = base_map(&[("layer", 3, 4)]);
    let err = fusion_error(
        apply_lora_adapters(&base, &dir),
        "an adapter with no pairs must not load",
    );
    assert!(err.contains("applied no tensors"), "{err}");
    assert!(err.contains(&dir.display().to_string()), "{err}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_valid_adapter_directory_still_applies_and_names_its_directory_on_failure() {
    let dir = temp_dir("valid");
    let mut tensors = adapter_pair_tensors("layer", 4, 2, 3);
    tensors.insert("layer.m".to_string(), ones_tensor(3, 1));
    write_adapter_dir(&dir, "lora", 2, tensors);

    let base = base_map(&[("layer", 3, 4)]);
    let err = fusion_error(
        apply_lora_adapters(&base, &dir),
        "the stray magnitude tensor fails the load",
    );
    assert!(err.contains(&dir.display().to_string()), "{err}");
    assert!(err.contains("layer.m"), "{err}");

    // The same directory without the stray tensor loads and applies one pair.
    write_adapter_dir(&dir, "lora", 2, adapter_pair_tensors("layer", 4, 2, 3));
    let fused = apply_lora_adapters(&base, &dir).expect("a well-formed adapter applies");
    let after = tensor_sum(fused.get("layer.weight").unwrap());
    // 12 base entries of 1.0 plus a delta of scale (1.0) * rank (2) everywhere.
    assert!((after - 36.0).abs() < 1e-3, "unexpected fused sum: {after}");

    let _ = std::fs::remove_dir_all(&dir);
}
