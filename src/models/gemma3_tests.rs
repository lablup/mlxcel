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

// -----------------------------------------------------------------
// Exact-prefix snapshot prompt-cache support (issue #1335).
//
// The fixture is two layers under `sliding_window_pattern = 2`, so layer 0 is
// sliding and layer 1 is global. That is the smallest model that exercises
// both `Cache` arms in one snapshot, and with `sliding_window = 8` a 24-token
// prefill leaves the sliding ring wrapped while the global layer keeps every
// token. An exact restore has to survive both at once; a truncating restore
// has to decline once the ring wraps.
mod snapshot_prompt_cache {
    /// Sequence-id base for this module, so the ids stay readable as
    /// "issue 1335, sequence N" without tripping the inconsistent-digit-
    /// grouping lint that `1335_01` does.
    const SEQ_BASE: u64 = 1_335_000;

    use super::super::{Gemma3Model, Gemma3Wrapper, ModelArgs};
    use mlxcel_core::cache::{KVCacheMode, SequenceId};
    use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
    use mlxcel_core::weights::WeightMap;
    use mlxcel_core::{MlxArray, UniquePtr};

    const HIDDEN: i32 = 4;
    const VOCAB: i32 = 8;
    const INTERMEDIATE: i32 = 8;
    const HEAD_DIM: i32 = 2;
    const HEADS: i32 = 2;
    const KV_HEADS: i32 = 1;
    const LAYERS: usize = 2;

    /// Deterministic small-magnitude weights.
    ///
    /// Constant weights would make every logit identical and turn each
    /// comparison below vacuous, so the fill is seeded per tensor and varies
    /// along the flat index.
    fn tensor(shape: &[i32], seed: u64) -> UniquePtr<MlxArray> {
        let len = shape.iter().product::<i32>() as usize;
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let data: Vec<f32> = (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
                unit * 0.4 - 0.2
            })
            .collect();
        mlxcel_core::from_slice_f32(&data, shape)
    }

    fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
        let len = shape.iter().product::<i32>() as usize;
        mlxcel_core::from_slice_f32(&vec![1.0_f32; len], shape)
    }

    /// The fixture config with a caller-chosen sliding window.
    ///
    /// Most tests here want a window narrow enough that a 24-token prefill
    /// wraps the ring. One wants the opposite: a window wide enough that
    /// `RotatingKVCache`'s decode-time growth (`step = 256`) can run the
    /// physical buffer past `offset` while the ring is still linear, which is
    /// the state a restored snapshot is in when the next turn's appended
    /// tokens arrive.
    fn synthetic_args_with_window(sliding_window: i32) -> ModelArgs {
        super::args(&format!(
            r#"{{
                "model_type": "gemma3_text",
                "hidden_size": 4,
                "num_hidden_layers": 2,
                "intermediate_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 8,
                "rope_theta": 10000.0,
                "rope_local_base_freq": 10000.0,
                "query_pre_attn_scalar": 2.0,
                "sliding_window": {sliding_window},
                "sliding_window_pattern": 2,
                "max_position_embeddings": 4096
            }}"#
        ))
    }

    fn synthetic_weights() -> WeightMap {
        let mut w = WeightMap::new();
        w.insert(
            "model.embed_tokens.weight".into(),
            tensor(&[VOCAB, HIDDEN], 1),
        );
        w.insert("lm_head.weight".into(), tensor(&[VOCAB, HIDDEN], 2));
        w.insert("model.norm.weight".into(), ones(&[HIDDEN]));
        for layer in 0..LAYERS {
            let seed = 100 + layer as u64 * 10;
            let attn = format!("model.layers.{layer}.self_attn");
            w.insert(
                format!("{attn}.q_proj.weight"),
                tensor(&[HEADS * HEAD_DIM, HIDDEN], seed),
            );
            w.insert(
                format!("{attn}.k_proj.weight"),
                tensor(&[KV_HEADS * HEAD_DIM, HIDDEN], seed + 1),
            );
            w.insert(
                format!("{attn}.v_proj.weight"),
                tensor(&[KV_HEADS * HEAD_DIM, HIDDEN], seed + 2),
            );
            w.insert(
                format!("{attn}.o_proj.weight"),
                tensor(&[HIDDEN, HEADS * HEAD_DIM], seed + 3),
            );
            w.insert(format!("{attn}.q_norm.weight"), ones(&[HEAD_DIM]));
            w.insert(format!("{attn}.k_norm.weight"), ones(&[HEAD_DIM]));

            let mlp = format!("model.layers.{layer}.mlp");
            w.insert(
                format!("{mlp}.gate_proj.weight"),
                tensor(&[INTERMEDIATE, HIDDEN], seed + 4),
            );
            w.insert(
                format!("{mlp}.up_proj.weight"),
                tensor(&[INTERMEDIATE, HIDDEN], seed + 5),
            );
            w.insert(
                format!("{mlp}.down_proj.weight"),
                tensor(&[HIDDEN, INTERMEDIATE], seed + 6),
            );

            for norm in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                w.insert(
                    format!("model.layers.{layer}.{norm}.weight"),
                    ones(&[HIDDEN]),
                );
            }
        }
        w
    }

    fn build_wrapper() -> Gemma3Wrapper {
        build_wrapper_with_window(8)
    }

    fn build_wrapper_with_window(sliding_window: i32) -> Gemma3Wrapper {
        let args = synthetic_args_with_window(sliding_window);
        let weights = synthetic_weights();
        Gemma3Wrapper::new(
            Gemma3Model::from_weights(&weights, &args).expect("synthetic Gemma 3 must load"),
        )
    }

    fn to_vec_f32(arr: &MlxArray) -> Vec<f32> {
        let arr_f32 = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&arr_f32);
        mlxcel_core::array_to_raw_bytes(&arr_f32)
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn ids(range: std::ops::Range<i32>) -> Vec<i32> {
        range.map(|i| i.rem_euclid(VOCAB - 1) + 1).collect()
    }

    fn prefill(wrapper: &Gemma3Wrapper, seq: SequenceId, tokens: &[i32]) {
        wrapper.prepare_sequence_state(seq);
        let prompt = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
        let _ = wrapper.forward_with_sequence_id(&prompt, Some(seq), &mut [], None);
    }

    /// Feed a multi-token chunk to a sequence that already has state, the way
    /// the server prefills the tokens a prompt-cache hit did not cover.
    ///
    /// Deliberately not `prefill`: that one calls `prepare_sequence_state`,
    /// which throws the restored caches away and turns the append into a cold
    /// run.
    fn append_prefill(wrapper: &Gemma3Wrapper, seq: SequenceId, tokens: &[i32]) -> Vec<f32> {
        let chunk = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
        let logits = wrapper.forward_with_sequence_id(&chunk, Some(seq), &mut [], None);
        to_vec_f32(logits.as_ref().expect("logits"))
    }

    fn decode(wrapper: &Gemma3Wrapper, seq: SequenceId, token: i32) -> Vec<f32> {
        let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        let logits = wrapper.forward_with_sequence_id(&input, Some(seq), &mut [], None);
        to_vec_f32(logits.as_ref().expect("logits"))
    }

    fn assert_logits_agree(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: logit count");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let abs = (g - w).abs();
            let rel = abs / w.abs().max(1.0);
            assert!(
                abs < 1e-3 || rel < 1e-3,
                "{what}: logit[{i}] restored={g}, reference={w}, abs={abs}, rel={rel}"
            );
        }
    }

    #[test]
    fn gemma3_declares_snapshot_reuse() {
        assert!(build_wrapper().supports_snapshot_reuse());
    }

    #[test]
    #[ignore = "requires serial MLX execution"]
    fn snapshot_restore_matches_cold_decode() {
        // 24 prefill tokens plus 8 decode steps puts the sliding layer well
        // past its 8-token window, so the ring has wrapped and the restore
        // has to reproduce the ring geometry, not just the buffer contents.
        let cold = build_wrapper();
        let seq_cold = SequenceId::from_raw(SEQ_BASE + 1);
        let prompt = ids(0..24);
        prefill(&cold, seq_cold, &prompt);
        let decoded = ids(24..32);
        for &token in &decoded {
            let _ = decode(&cold, seq_cold, token);
        }

        let snapshot = cold
            .snapshot_sequence_state(seq_cold, 32)
            .expect("Gemma 3 must donate a non-empty snapshot");
        assert_eq!(snapshot.family(), "gemma3");
        assert_eq!(snapshot.token_len(), 32);

        let restored = build_wrapper();
        let seq_restored = SequenceId::from_raw(SEQ_BASE + 2);
        restored.prepare_sequence_state(seq_restored);
        restored
            .restore_sequence_state(seq_restored, &snapshot)
            .expect("Gemma 3 must restore its own snapshot");

        for token in ids(32..36) {
            let reference = decode(&cold, seq_cold, token);
            let got = decode(&restored, seq_restored, token);
            assert_logits_agree(&got, &reference, "exact snapshot restore");
        }
    }

    #[test]
    #[ignore = "requires serial MLX execution"]
    fn appended_prefill_after_a_restore_matches_a_cold_run() {
        // The server's own shape, which single-token decode after a restore
        // does not reach: turn one prefills and decodes, the snapshot is
        // donated at the end of it, and turn two restores that snapshot and
        // prefills the APPENDED tokens as one multi-token forward.
        //
        // Only the multi-token forward builds a prefill mask, and only a cache
        // that has decoded carries `physical > offset`, so this pairing is the
        // one that exposed a mask sized from the physical buffer length rather
        // than the visible window (issue #1335). The window is wide here on
        // purpose: under the 8-token window of the other tests, the physical
        // buffer cannot outgrow `min(offset, window - 1)` and the defect hides.
        let window = 512;
        let cold = build_wrapper_with_window(window);
        let seq_cold = SequenceId::from_raw(SEQ_BASE + 9);
        prefill(&cold, seq_cold, &ids(0..6));
        for token in ids(6..10) {
            let _ = decode(&cold, seq_cold, token);
        }

        let snapshot = cold
            .snapshot_sequence_state(seq_cold, 10)
            .expect("Gemma 3 must donate a non-empty snapshot");

        let restored = build_wrapper_with_window(window);
        let seq_restored = SequenceId::from_raw(SEQ_BASE + 10);
        restored.prepare_sequence_state(seq_restored);
        restored
            .restore_sequence_state(seq_restored, &snapshot)
            .expect("Gemma 3 must restore its own snapshot");

        let appended = ids(10..13);
        let reference = append_prefill(&cold, seq_cold, &appended);
        let got = append_prefill(&restored, seq_restored, &appended);
        assert_logits_agree(&got, &reference, "appended prefill after restore");
    }

    #[test]
    #[ignore = "requires serial MLX execution"]
    fn truncated_restore_matches_a_cold_prefill_of_the_same_prefix() {
        // Under the 8-token window a 6-token prefill leaves the sliding ring
        // unwrapped, which is the whole precondition for a truncating restore.
        let source = build_wrapper();
        let seq_source = SequenceId::from_raw(SEQ_BASE + 3);
        prefill(&source, seq_source, &ids(0..6));
        let snapshot = source
            .snapshot_sequence_state(seq_source, 6)
            .expect("snapshot");
        assert!(source.snapshot_truncatable_to(&snapshot, 4));
        assert!(
            !source.snapshot_truncatable_to(&snapshot, 7),
            "cannot invent tokens the snapshot never held"
        );

        let adopted = build_wrapper();
        let seq_adopted = SequenceId::from_raw(SEQ_BASE + 4);
        adopted.prepare_sequence_state(seq_adopted);
        adopted
            .restore_sequence_state_truncated(seq_adopted, &snapshot, 4)
            .expect("truncating restore must succeed while the ring is unwrapped");

        let control = build_wrapper();
        let seq_control = SequenceId::from_raw(SEQ_BASE + 5);
        prefill(&control, seq_control, &ids(0..4));

        let next = ids(4..5)[0];
        let got = decode(&adopted, seq_adopted, next);
        let want = decode(&control, seq_control, next);
        assert_logits_agree(&got, &want, "truncated restore versus cold prefill");
    }

    #[test]
    #[ignore = "requires serial MLX execution"]
    fn a_wrapped_sliding_layer_declines_a_truncating_restore() {
        let wrapper = build_wrapper();
        let seq = SequenceId::from_raw(SEQ_BASE + 6);
        prefill(&wrapper, seq, &ids(0..24));
        let snapshot = wrapper.snapshot_sequence_state(seq, 24).expect("snapshot");
        assert!(
            !wrapper.snapshot_truncatable_to(&snapshot, 10),
            "a wrapped sliding ring no longer keeps logical token t at slot t"
        );
        let err = wrapper
            .restore_sequence_state_truncated(SequenceId::from_raw(SEQ_BASE + 7), &snapshot, 10)
            .expect_err("a declined truncation must not install a partial state");
        assert!(
            err.contains("cannot be truncated"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[ignore = "requires serial MLX execution"]
    fn a_foreign_family_snapshot_is_refused() {
        let wrapper = build_wrapper();
        let foreign = ModelStateSnapshot::new("gemma4", 4);
        let err = wrapper
            .restore_sequence_state(SequenceId::from_raw(SEQ_BASE + 8), &foreign)
            .expect_err("a Gemma 4 snapshot must not land in Gemma 3");
        assert!(err.contains("gemma4"), "unexpected error: {err}");
        assert!(!wrapper.snapshot_truncatable_to(&foreign, 2));
    }

    #[test]
    #[ignore = "requires serial MLX execution"]
    fn a_quantized_cache_mode_declines_the_restore() {
        // The shared serializer refuses to install an Fp16 snapshot into a
        // cache configured for a quantized mode. `kv_snapshot_tests.rs` pins
        // that check directly; this pins that Gemma 3's own wiring surfaces
        // it too, through `set_kv_cache_layer_modes` rather than the raw
        // cache constructors.
        let cold = build_wrapper();
        let seq_cold = SequenceId::from_raw(SEQ_BASE + 11);
        prefill(&cold, seq_cold, &ids(0..6));
        let snapshot = cold
            .snapshot_sequence_state(seq_cold, 6)
            .expect("Gemma 3 must donate a non-empty snapshot");

        let quantized = build_wrapper();
        quantized.set_kv_cache_layer_modes(vec![KVCacheMode::Int8; LAYERS]);
        let err = quantized
            .restore_sequence_state(SequenceId::from_raw(SEQ_BASE + 12), &snapshot)
            .expect_err("an Fp16 snapshot must not land in an Int8-configured cache");
        assert!(
            err.contains("does not match configured cache mode"),
            "unexpected error: {err}"
        );
    }
}

// -----------------------------------------------------------------
// Prefill mask width versus the keys a rotating cache actually returns
// (issue #1335).
//
// `CacheInterface::live_len` sizes the sliding prefill mask, and the mask has
// to be exactly as wide as the K/V the same forward's `update_and_fetch`
// returns. The two agree for a cache built by prefill alone, because
// `update_concat` stores precisely what it returns. They part company after a
// single decode step: `update_in_place` grows the physical buffer by `step`
// (256) tokens ahead of `offset`, and `update_concat` still concatenates only
// the visible `min(physical, offset)` prior keys onto the new ones. Reading
// the physical length back as the live length therefore built a mask wider
// than the scores, which `broadcast_shapes` rejects.
//
// Prefill-only serving never reached that state, which is why it went
// unnoticed until a prompt-cache snapshot restore was followed by the
// appended-token prefill of the next turn.
mod rotating_live_len {
    use crate::models::gemma3::CacheInterface;
    use mlxcel_core::layers::RotatingKVCache;
    use mlxcel_core::utils::create_sliding_window_prefill_mask;
    use mlxcel_core::{MlxArray, UniquePtr};

    const WINDOW: i32 = 512;
    const KV_HEADS: i32 = 1;
    const HEAD_DIM: i32 = 2;

    fn kv(tokens: i32) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = [1, KV_HEADS, tokens, HEAD_DIM];
        let len = (KV_HEADS * tokens * HEAD_DIM) as usize;
        let keys: Vec<f32> = (0..len).map(|i| i as f32 * 0.01).collect();
        let values: Vec<f32> = (0..len).map(|i| 1.0 - i as f32 * 0.01).collect();
        (
            mlxcel_core::from_slice_f32(&keys, &shape),
            mlxcel_core::from_slice_f32(&values, &shape),
        )
    }

    fn physical_len(cache: &RotatingKVCache) -> i32 {
        cache
            .keys
            .as_ref()
            .and_then(|k| k.as_ref())
            .map(|k| mlxcel_core::array_shape(k)[2])
            .expect("fixture cache must hold keys")
    }

    #[test]
    fn live_len_matches_the_keys_a_multi_token_append_returns() {
        let mut cache = RotatingKVCache::new(WINDOW);

        // Turn one: a six-token prefill, then one decode step.
        let (k, v) = kv(6);
        cache.update_and_fetch(k, v);
        let (k, v) = kv(1);
        cache.update_and_fetch(k, v);

        // Fixture guard. Without the `step`-sized growth there is nothing for
        // this test to catch, and a future change to `RotatingKVCache`'s
        // growth policy should fail here rather than pass vacuously.
        assert!(
            physical_len(&cache) > cache.offset,
            "fixture must reach physical > offset; physical {}, offset {}",
            physical_len(&cache),
            cache.offset
        );

        // Turn two: the appended-token prefill. The mask is sized before the
        // forward, from the cache; the keys come out of it.
        let live_len = CacheInterface::live_len(&cache);
        let appended = 3;
        let mask = create_sliding_window_prefill_mask(appended, live_len, WINDOW);
        let mask_keys = *mlxcel_core::array_shape(&mask)
            .last()
            .expect("mask must be rank >= 1");

        let (k, v) = kv(appended);
        let (returned_keys, _) = cache.update_and_fetch(k, v);
        let returned = mlxcel_core::array_shape(&returned_keys)[2];

        assert_eq!(
            mask_keys, returned,
            "prefill mask key axis {mask_keys} must equal the {returned} keys the cache returned"
        );
    }
}
