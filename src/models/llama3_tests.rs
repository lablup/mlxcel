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

//! Unit tests for the shared Llama config surface, focused on
//! [`ModelArgs::rope_traditional`] (#931).
//!
//! Everything here is checkpoint-free: the parse tests use synthetic
//! `config.json` bodies shaped like the ones the real loaders hand to
//! `serde_json`, and the model tests build a tiny synthetic weight map.
//!
//! Why this file leans on values rather than shapes: the two RoPE conventions
//! rotate different channel pairs but produce identically shaped tensors from
//! identical weights. A checkpoint decoded with the wrong one loads, caches,
//! generates and reads as fluent text. No shape assertion, no cache assertion
//! and no logits-shape assertion can tell them apart, so every correctness
//! assertion below compares values against a model built with the convention
//! set programmatically.

use super::{
    Attention, FUSED_CAUSAL_PREFILL_ENV, FUSED_QKV_SPLIT_ROPE_ENV, FUSED_ROPE_ENV_VARS,
    Llama3Model, ModelArgs,
};
use crate::test_support::env_lock::env_lock;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// Test fixtures.

/// A deterministic fill, so every difference measured below is reproducible.
fn filled(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let mut state: u32 = 0x9E37_79B9;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape)
}

fn max_abs(a: &MlxArray) -> f32 {
    mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(a)))
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    max_abs(&mlxcel_core::subtract(a, b))
}

/// The required field set of a Llama `config.json`, as a JSON object body
/// without the surrounding braces so tests can append the key under test.
const TINY_LLAMA_FIELDS: &str = r#"
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "rms_norm_eps": 1e-5,
    "rope_theta": 100000.0,
    "vocab_size": 32
"#;

/// Parse a Llama config with `extra` appended to the required field set.
/// `extra` is a JSON fragment such as `"rope_traditional": true`.
fn parse_llama_config(extra: &str) -> ModelArgs {
    let body = if extra.is_empty() {
        format!("{{{TINY_LLAMA_FIELDS}}}")
    } else {
        format!("{{{TINY_LLAMA_FIELDS}, {extra}}}")
    };
    serde_json::from_str(&body).unwrap_or_else(|err| panic!("config must parse: {err}\n{body}"))
}

/// A float checkpoint matching [`TINY_LLAMA_FIELDS`].
fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let ff = args.intermediate_size as i32;
    let vocab = args.vocab_size as i32;
    let q_out = (args.num_attention_heads * args.head_dim()) as i32;
    let kv_out = (args.num_kv_heads() * args.head_dim()) as i32;

    let mut w = WeightMap::new();
    w.insert("model.embed_tokens.weight".into(), filled(&[vocab, hidden]));
    w.insert("model.norm.weight".into(), ones(&[hidden]));
    w.insert("lm_head.weight".into(), filled(&[vocab, hidden]));

    for layer in 0..args.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        w.insert(format!("{p}.input_layernorm.weight"), ones(&[hidden]));
        w.insert(
            format!("{p}.post_attention_layernorm.weight"),
            ones(&[hidden]),
        );
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            filled(&[q_out, hidden]),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            filled(&[kv_out, hidden]),
        );
        w.insert(
            format!("{p}.self_attn.v_proj.weight"),
            filled(&[kv_out, hidden]),
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            filled(&[hidden, q_out]),
        );
        w.insert(format!("{p}.mlp.gate_proj.weight"), filled(&[ff, hidden]));
        w.insert(format!("{p}.mlp.up_proj.weight"), filled(&[ff, hidden]));
        w.insert(format!("{p}.mlp.down_proj.weight"), filled(&[hidden, ff]));
    }
    w
}

// The parse contract.

#[test]
fn a_config_without_the_key_keeps_the_split_half_rotation() {
    // The no-regression half of #931. No mlx-community Llama, Qwen2 or Qwen2.5
    // checkpoint declares `rope_traditional`, so this is the case that covers
    // every checkpoint that exists today, and it must stay `false` forever.
    assert!(!parse_llama_config("").rope_traditional);
}

#[test]
fn a_config_that_declares_the_key_is_honored_in_both_directions() {
    // The behavior change. Until #931 the field was `#[serde(skip)]` and this
    // parsed to `false` no matter what the checkpoint said, which is the
    // divergence from the reference `ModelArgs` in mlx_lm/models/llama.py
    // (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/llama.py),
    // where `rope_traditional` is a plain field passed into `initialize_rope`.
    assert!(parse_llama_config(r#""rope_traditional": true"#).rope_traditional);
    assert!(!parse_llama_config(r#""rope_traditional": false"#).rope_traditional);
}

#[test]
fn an_explicitly_nulled_key_parses_as_false_rather_than_failing_to_load() {
    // `#[serde(skip)]` ignored whatever the key held, so a config carrying an
    // explicit null loaded fine before #931. Deserializing the field must not
    // turn a previously-ignored key into a load failure, and null is falsy in
    // the reference too.
    assert!(!parse_llama_config(r#""rope_traditional": null"#).rope_traditional);
}

#[test]
fn a_non_boolean_key_is_rejected_rather_than_guessed_at() {
    // The tolerance above is for null specifically. A string or a number is a
    // malformed config, and quietly picking a rotation for it is exactly the
    // silent-wrong-convention failure this issue is about.
    let body = format!(r#"{{{TINY_LLAMA_FIELDS}, "rope_traditional": "true"}}"#);
    serde_json::from_str::<ModelArgs>(&body)
        .expect_err("a string must not be coerced into a rotation convention");
}

#[test]
fn the_config_sanitizer_preserves_the_key() {
    // The generic Llama loader, both pipeline stage executors and the
    // tensor-parallel runtime all run `config.json` through
    // `sanitize_config_json` before parsing it. That pass rewrites `Infinity`
    // and `NaN` literals by string substitution, so it is worth pinning that it
    // does not disturb the key those four sites now depend on.
    let raw = format!(r#"{{{TINY_LLAMA_FIELDS}, "rope_traditional": true}}"#);
    let sanitized = crate::models::sanitize_config_json(&raw);
    let args: ModelArgs = serde_json::from_str(&sanitized).unwrap();
    assert!(args.rope_traditional);
}

#[test]
fn a_nested_text_config_is_honored_and_does_not_leak_from_the_top_level() {
    // Six VLM loaders (Pixtral, LLaVA, SmolVLM / Idefics3, Idefics2, InternVL)
    // hand `serde_json` the `text_config` sub-object, injecting only
    // `quantization` into it. FastVLM instead hands over the whole config
    // because it keeps its text fields at the root. Those are the two config
    // *shapes* the nine deserialization sites present, so both are pinned here.
    //
    // The negative half matters as much as the positive half: a top-level key
    // must not reach a nested text backbone, because in a VLM config the top
    // level describes the multimodal wrapper and not the decoder.
    let full: serde_json::Value = serde_json::from_str(&format!(
        r#"{{
            "model_type": "llava",
            "rope_traditional": true,
            "text_config": {{{TINY_LLAMA_FIELDS}}}
        }}"#
    ))
    .unwrap();

    let nested: ModelArgs = serde_json::from_value(full["text_config"].clone()).unwrap();
    assert!(
        !nested.rope_traditional,
        "a wrapper-level key must not silently re-rotate the text backbone"
    );

    // FastVLM's shape: the text fields, and the key, at the root.
    let root: ModelArgs = serde_json::from_value(
        serde_json::from_str(&format!(
            r#"{{{TINY_LLAMA_FIELDS}, "rope_traditional": true}}"#
        ))
        .unwrap(),
    )
    .unwrap();
    assert!(root.rope_traditional);
}

// The flag reaching the graph.

#[test]
fn the_parsed_flag_reaches_the_attention_block() {
    let args = parse_llama_config(r#""rope_traditional": true"#);
    let weights = tiny_weights(&args);
    let attention = Attention::from_weights(&weights, &args, "model.layers.0.self_attn").unwrap();
    assert!(attention.rope_traditional);

    let default_args = parse_llama_config("");
    let default_attention =
        Attention::from_weights(&weights, &default_args, "model.layers.0.self_attn").unwrap();
    assert!(!default_attention.rope_traditional);
}

#[test]
fn a_declared_flag_selects_exactly_the_interleaved_rotation() {
    // The strongest oracle available without a checkpoint that ships the key,
    // and the one that actually proves the fix. It is not enough that the JSON
    // key changes the output: it has to change it to the *interleaved*
    // rotation and to nothing else. So the model built from JSON is compared
    // against a model whose flag was set programmatically (the rotation Helium
    // has been validated on against mlx-lm), and separately against the
    // split-half model it must no longer equal.
    //
    // Real-checkpoint token-exactness against mlx-lm cannot be run here: no
    // public Llama, Qwen2 or Qwen2.5 checkpoint declares `rope_traditional`,
    // so there is nothing to generate from. What that comparison would test is
    // that mlxcel's interleaved rotation matches the reference's, and that
    // equivalence is `fast_rope(traditional = true)` itself, which the Helium
    // port validated token-exactly against mlx-lm on a real checkpoint.
    let from_json = parse_llama_config(r#""rope_traditional": true"#);
    let weights = tiny_weights(&from_json);

    let mut programmatic = parse_llama_config("");
    programmatic.rope_traditional = true;
    let mut split_half = parse_llama_config("");
    split_half.rope_traditional = false;

    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7, 8], &[1, 8]);
    let logits = |args: &ModelArgs| {
        let model = Llama3Model::from_weights(&weights, args).unwrap();
        let mut caches = model.make_caches();
        model.forward(&prompt, &mut caches, None)
    };

    let json_logits = logits(&from_json);
    let programmatic_logits = logits(&programmatic);
    let split_half_logits = logits(&split_half);

    assert_eq!(
        mlxcel_core::array_shape(&json_logits),
        mlxcel_core::array_shape(&split_half_logits),
        "the two conventions are shape-identical, which is why this test compares values"
    );

    // Same graph, so this is an equality check and not a tolerance check.
    let convention_gap = max_abs_diff(&json_logits, &split_half_logits);
    let identity_gap = max_abs_diff(&json_logits, &programmatic_logits);
    assert_eq!(
        identity_gap, 0.0,
        "the JSON key must select the same rotation the loader-set flag selects \
         (gap {identity_gap}, convention gap {convention_gap})"
    );

    // Two floors, absolute and relative, for the same reason the Helium suite
    // carries two: an absolute floor alone goes blind if a future change grows
    // the logits without growing the separation between the conventions.
    let logits_scale = max_abs(&json_logits);
    assert!(
        convention_gap > 1e-3,
        "declaring the key must actually change the logits \
         (gap {convention_gap}, logits scale {logits_scale})"
    );
    assert!(
        convention_gap > logits_scale * 1e-4,
        "the separation must be a meaningful fraction of the logits, not an epsilon \
         (gap {convention_gap}, logits scale {logits_scale})"
    );
}

#[test]
fn the_batched_route_honors_the_parsed_flag_too() {
    // The single-sequence path calls `fast_rope`, the batched decode and
    // batched/padded prefill path calls `fast_rope_batched`, and the paged
    // decode path rotates before its dispatch. Honoring the key on one and
    // dropping it on another would make a sequence decode differently depending
    // on whether the scheduler ran it alone or in a batch, which is invisible
    // from either result on its own.
    let args = parse_llama_config(r#""rope_traditional": true"#);
    let weights = tiny_weights(&args);
    let model = Llama3Model::from_weights(&weights, &args).unwrap();

    let single_ids = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7, 8], &[1, 8]);
    let mut single_caches = LanguageModel::make_caches(&model);
    let single = LanguageModel::forward(&model, &single_ids, &mut single_caches, None);

    let batched_ids =
        mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8], &[2, 8]);
    let mut row0 = LanguageModel::make_caches(&model);
    let mut row1 = LanguageModel::make_caches(&model);
    let mut batch: Vec<&mut [KVCache]> = vec![row0.as_mut_slice(), row1.as_mut_slice()];
    let batched = LanguageModel::forward_batched(&model, &batched_ids, &mut batch, None);
    let batched_row0 = mlxcel_core::slice(&batched, &[0, 0, 0], &[1, i32::MAX, i32::MAX]);

    let route_gap = max_abs_diff(&single, &batched_row0);

    // Scale the tolerance against the difference the two conventions make, so
    // the assertion cannot pass by both routes being wrong the same way.
    let mut split_half = parse_llama_config("");
    split_half.rope_traditional = false;
    let split_half_model = Llama3Model::from_weights(&weights, &split_half).unwrap();
    let mut split_half_caches = split_half_model.make_caches();
    let split_half_logits = split_half_model.forward(&single_ids, &mut split_half_caches, None);
    let convention_gap = max_abs_diff(&single, &split_half_logits);

    assert!(
        route_gap < convention_gap / 100.0,
        "the batched route must apply the rotation the config asked for \
         (route gap {route_gap}, convention gap {convention_gap})"
    );
}

// The fused-path decision (#931).

#[test]
fn the_fused_rope_env_vars_are_the_ones_the_code_actually_reads() {
    // The bypass, the one-time notice and `docs/environment-variables.md` all
    // name these two variables. `FUSED_CAUSAL_PREFILL_ENV` is read by the gate
    // in `Attention::forward`; `FUSED_QKV_SPLIT_ROPE_ENV` is read inside
    // `FusedQKVLinear::forward_split_rope`, in mlxcel-core, where this test
    // cannot see it. Pinning the spellings here means a rename on either side
    // fails a test instead of silently disabling the bypass or the notice.
    assert_eq!(
        FUSED_ROPE_ENV_VARS,
        [
            "MLXCEL_ENABLE_FUSED_CAUSAL_PREFILL_ATTENTION",
            "MLXCEL_ENABLE_FUSED_QKV_SPLIT_ROPE",
        ]
    );
    assert_eq!(FUSED_CAUSAL_PREFILL_ENV, FUSED_ROPE_ENV_VARS[0]);
    assert_eq!(FUSED_QKV_SPLIT_ROPE_ENV, FUSED_ROPE_ENV_VARS[1]);
}

/// A genuinely quantized attention block, which is the only shape that can
/// reach either fused launcher, built with the given rotation convention.
fn quantized_attention(rope_traditional: bool) -> Attention {
    const GROUP_SIZE: i32 = 32;
    const BITS: i32 = 4;
    let hidden = 64;
    let proj = 64; // num_attention_heads * head_dim

    let mut weights = WeightMap::new();
    for (name, rows, cols) in [
        ("q_proj", proj, hidden),
        ("k_proj", proj, hidden),
        ("v_proj", proj, hidden),
        ("o_proj", hidden, proj),
    ] {
        let quantized = mlxcel_core::quantize_weights(&filled(&[rows, cols]), GROUP_SIZE, BITS);
        let p = format!("model.layers.0.self_attn.{name}");
        weights.insert(
            format!("{p}.weight"),
            mlxcel_core::quantized_weights_w(&quantized),
        );
        weights.insert(
            format!("{p}.scales"),
            mlxcel_core::quantized_weights_scales(&quantized),
        );
        if mlxcel_core::quantized_weights_has_biases(&quantized) {
            weights.insert(
                format!("{p}.biases"),
                mlxcel_core::quantized_weights_biases(&quantized),
            );
        }
    }

    let mut args = parse_llama_config(r#""quantization": { "group_size": 32, "bits": 4 }"#);
    args.rope_traditional = rope_traditional;
    Attention::from_weights(&weights, &args, "model.layers.0.self_attn").unwrap()
}

#[test]
fn a_traditional_rope_block_is_routed_around_the_fused_prefill_launcher() {
    // #931 decided to keep bypassing the two fused launchers rather than extend
    // the cxx bridge signature; `Attention::forward` records the full reasoning.
    // This asserts the behavior that decision rests on, so the assumption fails
    // loudly if either launcher ever changes.
    //
    // Both halves are needed. The equality says the flag makes the launcher
    // request a no-op. The inequality says the launcher request is not a no-op
    // in general, which is what stops the equality from being vacuous: with the
    // same weights and the flag off, the same request produces a visibly
    // different rotation.
    let traditional = quantized_attention(true);
    let split_half = quantized_attention(false);

    // The gate is only reachable for quantized weights, so a float fixture
    // would make everything below pass for the wrong reason.
    assert!(
        traditional
            .qkv_proj
            .qkv_proj
            .as_quantized_weight()
            .is_some()
            && traditional.o_proj.as_quantized_weight().is_some(),
        "the fixture must be quantized, or the fused branch is unreachable and this test is blind"
    );

    let x = filled(&[1, 8, 64]);
    let run = |attention: &Attention| {
        let mut cache = KVCache::new();
        let out = attention.forward(&x, &mut cache, None);
        mlxcel_core::eval(&out);
        out
    };

    let baseline = run(&traditional);

    let _env_guard = env_lock();
    let previous = std::env::var_os(FUSED_CAUSAL_PREFILL_ENV);
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe { std::env::set_var(FUSED_CAUSAL_PREFILL_ENV, "1") };

    let gated_traditional = run(&traditional);
    let gated_split_half = run(&split_half);

    // SAFETY: serialized via the crate-wide ENV_LOCK, still held.
    match previous {
        Some(value) => unsafe { std::env::set_var(FUSED_CAUSAL_PREFILL_ENV, value) },
        None => unsafe { std::env::remove_var(FUSED_CAUSAL_PREFILL_ENV) },
    }

    assert_eq!(
        max_abs_diff(&baseline, &gated_traditional),
        0.0,
        "a rope_traditional block must take the graph path whether or not the fused prefill \
         launcher was requested, because the launcher hardcodes the split-half rotation"
    );
    let convention_gap = max_abs_diff(&gated_traditional, &gated_split_half);
    assert!(
        convention_gap > 1e-3,
        "the launcher path must be observably a different rotation, or the equality above \
         proves nothing (gap {convention_gap})"
    );
}

// Greedy token parity for the #905 fused decode kernels.

/// Argmax of the last position's logits.
///
/// The model returns `[1, L, vocab]`; greedy decoding only ever reads the last
/// row, which is the whole row for a decode step and the final prompt position
/// for the prefill call.
fn greedy_last_token(logits: &MlxArray) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let vocab = *shape.last().expect("logits have a trailing vocab axis") as usize;
    let f = mlxcel_core::astype(logits, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    let raw = mlxcel_core::array_to_raw_bytes(&f);
    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let row = &values[values.len() - vocab..];
    let mut best = 0usize;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best as i32
}

/// Greedy decode of a pinned prompt through the tiny synthetic Llama, which is
/// the same `TransformerBlock::forward` and `Attention::forward` the real
/// family uses.
fn greedy_sequence(steps: usize) -> Vec<i32> {
    let args = parse_llama_config("");
    let weights = tiny_weights(&args);
    let model = Llama3Model::from_weights(&weights, &args).unwrap();
    let mut caches = model.make_caches();

    let prompt = mlxcel_core::from_slice_i32(&[3, 1, 4, 1, 5, 9, 2, 6], &[1, 8]);
    let mut next = greedy_last_token(&model.forward(&prompt, &mut caches, None));
    let mut out = vec![next];
    for _ in 1..steps {
        let step = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
        next = greedy_last_token(&model.forward(&step, &mut caches, None));
        out.push(next);
    }
    out
}

/// The unfused baseline, captured from a run with both #905 kill switches set
/// to `0` and pinned here.
///
/// Both fusions sit on this exact path: `TransformerBlock::forward` takes the
/// fused residual join, and `Attention::forward` takes the fused q/k RoPE +
/// append-layout kernel ahead of the graph fallback. So the default build has
/// to reproduce this sequence token for token, and a run with the switches off
/// has to reproduce it trivially.
///
/// Why a pinned list rather than an in-process A/B: both gates are read once
/// and cached for the process lifetime (they are on the per-token hot path and
/// must not re-read the environment), so a single test process cannot exercise
/// both sides. Running this file in both modes is what closes that.
/// A tiny random-weight model settles into a short cycle after a few steps, so
/// the discriminating tokens are the leading ones; the tail is kept because a
/// fusion bug that only compounds over steps would show up there and nowhere
/// else.
const UNFUSED_GREEDY_BASELINE: &[i32] = &[
    13, 23, 20, 10, 20, 10, 20, 10, 20, 10, 20, 10, 20, 10, 20, 10, 20, 10, 20, 10, 20, 10, 20, 10,
];

#[test]
fn greedy_decode_is_token_identical_to_the_unfused_baseline() {
    let tokens = greedy_sequence(UNFUSED_GREEDY_BASELINE.len());
    assert_eq!(
        tokens,
        UNFUSED_GREEDY_BASELINE,
        "greedy decode diverged from the unfused baseline; \
         MLXCEL_FUSED_ADD_RMSNORM / MLXCEL_FUSED_ROPE_APPEND were \
         {:?} / {:?}",
        std::env::var("MLXCEL_FUSED_ADD_RMSNORM").ok(),
        std::env::var("MLXCEL_FUSED_ROPE_APPEND").ok()
    );
}
