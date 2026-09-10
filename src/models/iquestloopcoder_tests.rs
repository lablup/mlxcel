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

//! Tests for the IQuest-Coder Loop two-pass decoder.
//!
//! Every correctness risk in this family produces *fluent* output when it is
//! wrong: reading the wrong cache in pass 2, gating on the pre-RoPE query, or
//! forgetting the local window all give a model that still generates plausible
//! text. So the discriminating tests here are written as differential tests:
//! each one builds the wrong variant explicitly and asserts the implementation
//! does **not** match it, alongside asserting it *does* match the right one.
//! A test that only checked "output is finite" would pass on all three bugs.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::iquestloopcoder::{
    LayerCaches, LoopGate, ModelArgs, SUPPORTED_LOOP_NUM, TransformerBlock, mix_gated,
};
use super::iquestloopcoder_model::{
    IQuestLoopCoderModel, IQuestLoopCoderWrapper, sanitize_weights,
};

// Config.

/// `mlx-community/IQuest-Coder-V1-40B-Loop-Instruct-4bit`'s `config.json`,
/// field for field, including keys this loader ignores.
const LOOP_40B_CONFIG: &str = r#"{
    "architectures": ["IQuestLoopCoderForCausalLM"],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "eos_token_id": [2, 75864, 75869],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 27648,
    "loop_num": 2,
    "loop_window_size": 64,
    "max_position_embeddings": 131072,
    "mlp_bias": false,
    "model_type": "iquestloopcoder",
    "num_attention_heads": 40,
    "num_hidden_layers": 80,
    "num_key_value_heads": 8,
    "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "use_cache": true,
    "vocab_size": 76800
}"#;

#[test]
fn parses_the_published_40b_loop_config() {
    let args: ModelArgs = serde_json::from_str(LOOP_40B_CONFIG).expect("parse 40B-Loop config");

    assert_eq!(args.model_type, "iquestloopcoder");
    assert_eq!(args.num_hidden_layers, 80);
    assert_eq!(args.hidden_size, 5120);
    assert_eq!(args.num_attention_heads, 40);
    assert_eq!(args.num_kv_heads(), 8);
    assert_eq!(args.head_dim(), 128);
    assert_eq!(args.intermediate_size, 27648);
    assert_eq!(args.vocab_size, 76800);
    assert_eq!(args.loop_num, 2);
    assert_eq!(args.loop_window_size, 64);
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
    assert_eq!(args.quant_mode(), "affine");
    assert!(!args.tie_word_embeddings);
    // All three must stop generation, not just the first.
    assert_eq!(args.eos_token_ids(), vec![2, 75864, 75869]);
    args.validate().expect("the published config must load");
}

#[test]
fn rejects_loop_num_other_than_two() {
    for loop_num in [0usize, 1, 3, 4] {
        let mut args: ModelArgs = serde_json::from_str(LOOP_40B_CONFIG).unwrap();
        args.loop_num = loop_num;
        let err = args
            .validate()
            .expect_err("only loop_num 2 is implemented, so anything else must fail at load");
        assert!(
            err.contains("loop_num"),
            "the error must name the offending key, got: {err}"
        );
    }

    let mut args: ModelArgs = serde_json::from_str(LOOP_40B_CONFIG).unwrap();
    args.loop_num = SUPPORTED_LOOP_NUM;
    assert!(args.validate().is_ok());
}

#[test]
fn rejects_zero_loop_window_size() {
    let mut args: ModelArgs = serde_json::from_str(LOOP_40B_CONFIG).unwrap();
    args.loop_window_size = 0;
    let err = args
        .validate()
        .expect_err("a zero window stores no local keys");
    assert!(err.contains("loop_window_size"), "got: {err}");
}

#[test]
fn eos_token_id_accepts_both_spellings() {
    let list: ModelArgs =
        serde_json::from_str(&LOOP_40B_CONFIG.replace("[2, 75864, 75869]", "[7, 9]")).unwrap();
    assert_eq!(list.eos_token_ids(), vec![7, 9]);

    let scalar: ModelArgs =
        serde_json::from_str(&LOOP_40B_CONFIG.replace("[2, 75864, 75869]", "7")).unwrap();
    assert_eq!(scalar.eos_token_ids(), vec![7]);
}

#[test]
fn sanitize_weights_is_identity_and_idempotent() {
    let weights = synthetic_weights(&tiny_args(4, 8));
    let before: Vec<String> = sorted_keys(&weights);

    let once = sanitize_weights(weights);
    let after_once = sorted_keys(&once);
    assert_eq!(
        before, after_once,
        "this family needs no key renaming; sanitize must not add, drop or rewrite a key"
    );

    let twice = sanitize_weights(once);
    assert_eq!(
        after_once,
        sorted_keys(&twice),
        "sanitize must be idempotent"
    );
}

// Synthetic model fixtures.

/// A deterministic model small enough to run on CPU inside a unit test, with
/// the same structural shape as the real checkpoint (GQA, an explicit
/// `head_dim`, an untied `lm_head`).
fn tiny_args(loop_window_size: usize, num_hidden_layers: usize) -> ModelArgs {
    let json = format!(
        r#"{{
            "model_type": "iquestloopcoder",
            "hidden_size": 8,
            "num_hidden_layers": {num_hidden_layers},
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 1e-05,
            "vocab_size": 16,
            "rope_theta": 500000,
            "loop_num": 2,
            "loop_window_size": {loop_window_size},
            "tie_word_embeddings": false,
            "eos_token_id": [3]
        }}"#
    );
    serde_json::from_str(&json).expect("tiny config")
}

/// A small LCG so the fixtures are byte-reproducible across runs and hosts
/// without depending on MLX's RNG seeding.
fn pseudo_random(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 33) as u32) as f32 / (1u64 << 31) as f32;
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
}

fn rand_array(shape: &[i32], seed: u64, scale: f32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&pseudo_random(seed, n as usize, scale), shape)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(shape, 1.0, mlxcel_core::dtype::FLOAT32)
}

fn sorted_keys(weights: &WeightMap) -> Vec<String> {
    let mut keys: Vec<String> = weights.keys().cloned().collect();
    keys.sort();
    keys
}

fn synthetic_weights(args: &ModelArgs) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let head_dim = args.head_dim() as i32;
    let heads = args.num_attention_heads as i32;
    let kv_heads = args.num_kv_heads() as i32;
    let q_size = heads * head_dim;
    let kv_size = kv_heads * head_dim;
    let inter = args.intermediate_size as i32;
    let vocab = args.vocab_size as i32;

    let mut weights = WeightMap::new();
    let mut seed = 1u64;
    let mut next = |shape: &[i32], scale: f32| {
        seed += 7;
        rand_array(shape, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), scale)
    };

    weights.insert(
        "model.embed_tokens.weight".into(),
        next(&[vocab, hidden], 0.5),
    );
    weights.insert("model.norm.weight".into(), ones(&[hidden]));
    weights.insert("lm_head.weight".into(), next(&[vocab, hidden], 0.4));

    for layer in 0..args.num_hidden_layers {
        let attn = format!("model.layers.{layer}.self_attn");
        weights.insert(
            format!("{attn}.q_proj.weight"),
            next(&[q_size, hidden], 0.4),
        );
        weights.insert(
            format!("{attn}.k_proj.weight"),
            next(&[kv_size, hidden], 0.4),
        );
        weights.insert(
            format!("{attn}.v_proj.weight"),
            next(&[kv_size, hidden], 0.4),
        );
        weights.insert(
            format!("{attn}.o_proj.weight"),
            next(&[hidden, q_size], 0.4),
        );

        let mlp = format!("model.layers.{layer}.mlp");
        weights.insert(
            format!("{mlp}.gate_proj.weight"),
            next(&[inter, hidden], 0.3),
        );
        weights.insert(format!("{mlp}.up_proj.weight"), next(&[inter, hidden], 0.3));
        weights.insert(
            format!("{mlp}.down_proj.weight"),
            next(&[hidden, inter], 0.3),
        );

        weights.insert(
            format!("model.layers.{layer}.input_layernorm.weight"),
            ones(&[hidden]),
        );
        weights.insert(
            format!("model.layers.{layer}.post_attention_layernorm.weight"),
            ones(&[hidden]),
        );

        // Never quantized in any published checkpoint: plain [H, D] and [H].
        weights.insert(
            format!("model.gate_projections.{layer}.weight"),
            next(&[heads, head_dim], 0.5),
        );
        weights.insert(
            format!("model.gate_projections.{layer}.bias"),
            next(&[heads], 0.5),
        );
    }

    weights
}

fn tiny_model(args: &ModelArgs) -> IQuestLoopCoderModel {
    IQuestLoopCoderModel::from_weights(&synthetic_weights(args), args).expect("build tiny model")
}

fn token_ids(len: i32, vocab: i32) -> UniquePtr<MlxArray> {
    let ids: Vec<i32> = (0..len).map(|i| (i * 5 + 3) % vocab).collect();
    mlxcel_core::from_slice_i32(&ids, &[1, len])
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    let (a, b) = (array_to_vec_f32(a), array_to_vec_f32(b));
    assert_eq!(a.len(), b.len(), "shape mismatch in comparison");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Slice `[B, H, L, D]` down to one head.
fn head_slice(x: &MlxArray, head: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    mlxcel_core::slice(
        x,
        &[0, head, 0, 0],
        &[shape[0], head + 1, shape[2], shape[3]],
    )
}

/// Slice `[B, H, L, D]` down to one query position.
fn position_slice(x: &MlxArray, pos: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    mlxcel_core::slice(x, &[0, 0, pos, 0], &[shape[0], shape[1], pos + 1, shape[3]])
}

// Cache layout.

#[test]
fn make_caches_pairs_a_dense_and_a_rotating_cache_per_layer() {
    let args = tiny_args(4, 3);
    let model = tiny_model(&args);
    let caches = model.make_caches();

    assert_eq!(
        caches.len(),
        args.num_hidden_layers,
        "one LayerCaches per decoder layer, each holding both passes' caches"
    );
    for (i, layer) in caches.iter().enumerate() {
        assert_eq!(layer.pass1.offset, 0, "layer {i} pass-1 cache starts empty");
        assert_eq!(layer.pass2.offset, 0, "layer {i} pass-2 cache starts empty");
        assert_eq!(
            layer.pass2.max_size, 4,
            "layer {i} pass-2 cache is bounded at loop_window_size, not unbounded"
        );
    }
}

#[test]
fn both_caches_advance_together_and_the_local_one_stays_bounded() {
    let args = tiny_args(4, 1);
    let model = tiny_model(&args);
    let mut caches = model.make_caches();

    let _ = model.forward_with_caches(&token_ids(10, 16), &mut caches);
    assert_eq!(caches[0].pass1.offset, 10);
    assert_eq!(caches[0].pass2.offset, 10);

    for _ in 0..3 {
        let _ = model.forward_with_caches(&token_ids(1, 16), &mut caches);
    }
    assert_eq!(caches[0].pass1.offset, 13, "pass 1 keeps the whole history");
    assert_eq!(caches[0].pass2.offset, 13);
    assert_eq!(
        caches[0].pass2.visible_len(),
        4,
        "pass 2 holds at most loop_window_size keys once decode has trimmed the ring"
    );
}

// The gate.

#[test]
fn gate_is_per_head_sigmoid() {
    // gate_w = 0 and gate_b = [+10, -10] pin head 0's gate to ~1 (all global)
    // and head 1's to ~0 (all local), so the mixed output must equal the global
    // branch on head 0 and the local branch on head 1.
    let args = tiny_args(4, 1);
    let model = tiny_model(&args);
    let attn = model.attention(0);
    let (heads, head_dim) = (attn.num_heads, attn.head_dim);

    let gate = LoopGate::from_arrays(
        &mlxcel_core::full_f32(&[heads, head_dim], 0.0, mlxcel_core::dtype::FLOAT32),
        &mlxcel_core::from_slice_f32(&[10.0, -10.0], &[heads]),
        heads,
        head_dim,
    );

    let x = rand_array(&[1, 6, args.hidden_size as i32], 991, 1.0);
    let (q2, k2, v2) = attn.get_qkv(&x, 0);
    let g = gate.forward(&q2);

    let g_values = array_to_vec_f32(&g);
    assert_eq!(
        mlxcel_core::array_shape(&g),
        vec![1, heads, 6, 1],
        "the gate is one scalar per head per token"
    );
    let expected_hi = 1.0f32 / (1.0 + (-10.0f32).exp());
    let expected_lo = 1.0f32 / (1.0 + 10.0f32.exp());
    for t in 0..6 {
        assert!(
            (g_values[t] - expected_hi).abs() < 1e-5,
            "head 0 gate at t={t} is {} not ~{expected_hi}",
            g_values[t]
        );
        assert!(
            (g_values[6 + t] - expected_lo).abs() < 1e-5,
            "head 1 gate at t={t} is {} not ~{expected_lo}",
            g_values[6 + t]
        );
    }

    // Separate the two branches and check the mix routes each head correctly.
    let k1 = rand_array(&mlxcel_core::array_shape(&k2), 4242, 1.0);
    let v1 = rand_array(&mlxcel_core::array_shape(&v2), 8484, 1.0);
    let global = attn.attend(&q2, &k1, &v1, 0);
    let local = attn.attend(&q2, &k2, &v2, args.loop_window_size as i32);
    let mixed = mix_gated(&g, &global, &local);

    assert!(
        max_abs_diff(&global, &local) > 1e-2,
        "the fixture must make the two branches actually differ, or this test is vacuous"
    );
    assert!(
        max_abs_diff(&head_slice(&mixed, 0), &head_slice(&global, 0)) < 1e-4,
        "head 0 is gated fully to the global branch"
    );
    assert!(
        max_abs_diff(&head_slice(&mixed, 1), &head_slice(&local, 1)) < 1e-4,
        "head 1 is gated fully to the local branch"
    );
}

#[test]
fn gate_reads_the_post_rope_query() {
    // The gate must see the rotated query, which is the tensor that attends.
    // Rotation is identity only at position 0, so any later position separates
    // the two.
    let args = tiny_args(4, 1);
    let model = tiny_model(&args);
    let attn = model.attention(0);
    let gate = model.gate(0);

    let x = rand_array(&[1, 8, args.hidden_size as i32], 31_337, 1.0);
    let (q_post, _, _) = attn.get_qkv(&x, 0);

    // Same projection and head split, no rotation.
    let (q_raw, _, _) = attn.qkv_proj.forward(&x);
    let q_pre = mlxcel_core::transpose_axes(
        &mlxcel_core::reshape(&q_raw, &[1, 8, attn.num_heads, attn.head_dim]),
        &[0, 2, 1, 3],
    );

    assert!(
        max_abs_diff(&q_post, &q_pre) > 1e-3,
        "the fixture must actually rotate the query, or this test proves nothing"
    );
    assert!(
        max_abs_diff(&gate.forward(&q_post), &gate.forward(&q_pre)) > 1e-4,
        "a pre-RoPE gate would give different values here; the difference is what makes \
         `pass2_matches_only_the_correct_reference` able to tell them apart"
    );
}

// Differential tests for pass 2.

/// Which of the three pass-2 decisions to get deliberately wrong.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Variant {
    /// The implementation's own semantics.
    Correct,
    /// Gate computed from the query before RoPE.
    GateFromPreRope,
    /// Global branch reads the pass-2 K/V instead of pass 1's.
    GlobalReadsPass2Kv,
    /// Local branch attends the whole history instead of the window.
    LocalUnwindowed,
    /// The gate weights the local branch where it should weight the global one.
    BranchesSwapped,
}

/// An independent expression of one pass-2 layer, returning the same layer
/// output [`TransformerBlock::forward_pass2`] returns.
///
/// It reuses [`super::iquestloopcoder::Attention`]'s projection and SDPA helpers
/// and the block's own `feed_forward` tail (there is no point re-deriving RoPE
/// or SwiGLU here), but composes the four decisions itself: *which* query feeds
/// the gate, *which* cache feeds the global branch, whether the local branch is
/// windowed, and which way round the gate weights the two branches. Those four
/// are exactly what the implementation must get right, and each has a `Variant`
/// that gets it wrong.
fn reference_pass2_layer(
    block: &TransformerBlock,
    gate: &LoopGate,
    x: &MlxArray,
    pass1_keys: &MlxArray,
    pass1_values: &MlxArray,
    offset: i32,
    window: i32,
    variant: Variant,
) -> UniquePtr<MlxArray> {
    let attn = &block.self_attn;
    let normed = block.input_layernorm.forward(x);
    let (q2, k2, v2) = attn.get_qkv(&normed, offset);

    let gate_input = if variant == Variant::GateFromPreRope {
        let shape = mlxcel_core::array_shape(&normed);
        let (q_raw, _, _) = attn.qkv_proj.forward(&normed);
        mlxcel_core::transpose_axes(
            &mlxcel_core::reshape(&q_raw, &[shape[0], shape[1], attn.num_heads, attn.head_dim]),
            &[0, 2, 1, 3],
        )
    } else {
        mlxcel_core::copy(&q2)
    };
    let g = gate.forward(&gate_input);

    let global = if variant == Variant::GlobalReadsPass2Kv {
        attn.attend(&q2, &k2, &v2, 0)
    } else {
        attn.attend(&q2, pass1_keys, pass1_values, 0)
    };
    let local_window = if variant == Variant::LocalUnwindowed {
        0
    } else {
        window
    };
    let local = attn.attend(&q2, &k2, &v2, local_window);

    let mixed = if variant == Variant::BranchesSwapped {
        mix_gated(&g, &local, &global)
    } else {
        mix_gated(&g, &global, &local)
    };
    block.feed_forward(x, &attn.project_out(&mixed))
}

/// Run one layer's pass 1 and then compare pass 2 against each reference
/// variant. Returns `(diff_to_correct, diffs_to_each_wrong_variant)`.
fn pass2_variant_diffs(seq_len: i32, window: usize) -> (f32, Vec<(Variant, f32)>) {
    let args = tiny_args(window, 1);
    let model = tiny_model(&args);
    let block = model.layer(0);
    let gate = model.gate(0);
    let window = window as i32;

    let x0 = rand_array(&[1, seq_len, args.hidden_size as i32], 77_003, 1.0);

    // Pass 1 exactly as the model runs it, so pass 2 sees the real K/V.
    let mut pass1_cache = mlxcel_core::layers::KVCache::new();
    let (h, k1, v1) = block.forward_pass1(&x0, &mut pass1_cache, 0);

    // The real thing. Transcribing `forward_pass2`'s body here instead would
    // compare a copy of the code against variants of the same copy, and would
    // pass with the branches swapped or the gate on the pre-RoPE query. Both of
    // those were confirmed by mutation before this was changed to a real call.
    let mut pass2_cache = mlxcel_core::layers::RotatingKVCache::new(window);
    let implementation = block.forward_pass2(&h, &k1, &v1, gate, &mut pass2_cache, 0, window);

    let correct = reference_pass2_layer(block, gate, &h, &k1, &v1, 0, window, Variant::Correct);
    let diff_correct = max_abs_diff(&implementation, &correct);

    let wrong = [
        Variant::GateFromPreRope,
        Variant::GlobalReadsPass2Kv,
        Variant::LocalUnwindowed,
        Variant::BranchesSwapped,
    ]
    .into_iter()
    .map(|variant| {
        let other = reference_pass2_layer(block, gate, &h, &k1, &v1, 0, window, variant);
        (variant, max_abs_diff(&implementation, &other))
    })
    .collect();

    (diff_correct, wrong)
}

#[test]
fn pass2_matches_only_the_correct_reference() {
    // 12 tokens against a window of 4, so the local branch genuinely drops keys
    // and the unwindowed variant is separable. Below the window it would not be.
    let (diff_correct, wrong) = pass2_variant_diffs(12, 4);

    assert!(
        diff_correct < 1e-5,
        "pass 2 must match the correct reference; max abs diff {diff_correct}"
    );
    for (variant, diff) in wrong {
        assert!(
            diff > 1e-3,
            "{variant:?} is a real bug and must be distinguishable, but pass 2 differs from it \
             by only {diff}"
        );
    }
}

#[test]
fn pass2_uses_pass1_kv() {
    // The standalone form of the `GlobalReadsPass2Kv` case, kept separate
    // because it is the single most damaging way to get this architecture
    // wrong: the model still generates fluent text off the wrong context.
    let args = tiny_args(4, 1);
    let model = tiny_model(&args);
    let block = model.layer(0);
    let attn = &block.self_attn;

    let x0 = rand_array(&[1, 8, args.hidden_size as i32], 5_150, 1.0);
    let mut pass1_cache = mlxcel_core::layers::KVCache::new();
    let (h, k1, v1) = block.forward_pass1(&x0, &mut pass1_cache, 0);

    // What pass 1 hands forward is the cache's live window for the tokens just
    // written, not a stale read and not a doubled copy.
    assert_eq!(pass1_cache.offset, 8);
    assert_eq!(mlxcel_core::array_shape(&k1)[2], 8);
    let stored = pass1_cache.keys.as_ref().expect("pass-1 keys");
    let stored_shape = mlxcel_core::array_shape(stored);
    let live = mlxcel_core::slice(
        stored,
        &[0, 0, 0, 0],
        &[stored_shape[0], stored_shape[1], 8, stored_shape[3]],
    );
    assert_eq!(
        max_abs_diff(&k1, &live),
        0.0,
        "pass 2 must receive exactly the pass-1 cache contents"
    );

    let normed = block.input_layernorm.forward(&h);
    let (q2, k2, v2) = attn.get_qkv(&normed, 0);

    let global_from_pass1 = attn.attend(&q2, &k1, &v1, 0);
    let global_from_pass2 = attn.attend(&q2, &k2, &v2, 0);
    assert!(
        max_abs_diff(&global_from_pass1, &global_from_pass2) > 1e-3,
        "the two caches must hold materially different K/V, or this test is vacuous"
    );

    // Run the whole model and confirm each pass wrote only into its own cache.
    let mut caches = model.make_caches();
    let _ = model.forward_with_caches(&token_ids(8, 16), &mut caches);
    assert_eq!(
        caches[0].pass1.offset, 8,
        "pass 2 must not have appended its own K/V to the pass-1 cache"
    );
    assert_eq!(
        caches[0].pass2.offset, 8,
        "pass 2 wrote its own K/V into its own cache"
    );
}

#[test]
fn window_limits_local_branch() {
    // With window 4, the query at position 11 may attend only keys 8..11, so
    // rewriting keys 0..7 must leave its output untouched while changing the
    // output of a query whose window covers them.
    let args = tiny_args(4, 1);
    let model = tiny_model(&args);
    let attn = model.attention(0);
    let window = 4;

    let x = rand_array(&[1, 12, args.hidden_size as i32], 60_613, 1.0);
    let (q2, k2, v2) = attn.get_qkv(&x, 0);
    let baseline = attn.attend(&q2, &k2, &v2, window);

    // Replace keys and values 0..7 with different values, keeping 8..11.
    let kv_shape = mlxcel_core::array_shape(&k2);
    let tail_k = mlxcel_core::slice(
        &k2,
        &[0, 0, 8, 0],
        &[kv_shape[0], kv_shape[1], 12, kv_shape[3]],
    );
    let tail_v = mlxcel_core::slice(
        &v2,
        &[0, 0, 8, 0],
        &[kv_shape[0], kv_shape[1], 12, kv_shape[3]],
    );
    let head_shape = [kv_shape[0], kv_shape[1], 8, kv_shape[3]];
    let noisy_k = rand_array(&head_shape, 1_212, 3.0);
    let noisy_v = rand_array(&head_shape, 3_434, 3.0);
    let k_mod = mlxcel_core::concatenate(&noisy_k, &tail_k, 2);
    let v_mod = mlxcel_core::concatenate(&noisy_v, &tail_v, 2);

    let perturbed = attn.attend(&q2, &k_mod, &v_mod, window);

    let last = max_abs_diff(
        &position_slice(&baseline, 11),
        &position_slice(&perturbed, 11),
    );
    assert!(
        last < 1e-5,
        "position 11 is outside the rewritten range's window and must be unchanged, moved by {last}"
    );

    let inside = max_abs_diff(
        &position_slice(&baseline, 7),
        &position_slice(&perturbed, 7),
    );
    assert!(
        inside > 1e-3,
        "position 7 attends only rewritten keys and must change, moved by only {inside}"
    );

    // Without the window, position 11 would see the rewritten keys too.
    let unwindowed_baseline = attn.attend(&q2, &k2, &v2, 0);
    let unwindowed_perturbed = attn.attend(&q2, &k_mod, &v_mod, 0);
    assert!(
        max_abs_diff(
            &position_slice(&unwindowed_baseline, 11),
            &position_slice(&unwindowed_perturbed, 11)
        ) > 1e-3,
        "an unwindowed local branch would move at position 11; if it does not, this test cannot \
         distinguish the two implementations"
    );
}

// End-to-end cache consistency.

#[test]
fn prefill_matches_incremental_decode() {
    // 96 tokens in one call versus 64 + 32 in two. This exercises both caches'
    // offsets, the shared RoPE offset in pass 2, and the rotating cache's
    // multi-token continuation path (which keeps window - 1 prior keys).
    let args = tiny_args(64, 2);
    let model = tiny_model(&args);
    let all_ids = token_ids(96, 16);

    let mut single = model.make_caches();
    let single_logits = model.forward_last_logits_with_caches(&all_ids, &mut single, 95);

    let first = mlxcel_core::slice(&all_ids, &[0, 0], &[1, 64]);
    let second = mlxcel_core::slice(&all_ids, &[0, 64], &[1, 96]);
    let mut split = model.make_caches();
    let _ = model.forward_last_logits_with_caches(&first, &mut split, 63);
    let split_logits = model.forward_last_logits_with_caches(&second, &mut split, 31);

    assert_eq!(single[0].pass1.offset, split[0].pass1.offset);
    assert_eq!(single[0].pass2.offset, split[0].pass2.offset);

    let diff = max_abs_diff(&single_logits, &split_logits);
    assert!(
        diff < 1e-3,
        "a single-pass prefill and a chunked one must land on the same final logits; max abs \
         diff {diff}"
    );
}

#[test]
fn forward_last_logits_matches_the_sliced_full_logits() {
    let args = tiny_args(4, 1);
    let model = tiny_model(&args);
    let ids = token_ids(7, 16);

    let mut a = model.make_caches();
    let full = model.forward_with_caches(&ids, &mut a);
    let shape = mlxcel_core::array_shape(&full);
    let sliced = mlxcel_core::slice(&full, &[0, 6, 0], &[shape[0], 7, shape[2]]);

    let mut b = model.make_caches();
    let last = model.forward_last_logits_with_caches(&ids, &mut b, 6);

    assert_eq!(mlxcel_core::array_shape(&last), vec![1, 1, shape[2]]);
    assert!(max_abs_diff(&sliced, &last) < 1e-4);
}

// Wrapper and server surface.

#[test]
fn wrapper_declares_model_owned_state_and_no_dense_caches() {
    let args = tiny_args(4, 2);
    let wrapper = IQuestLoopCoderWrapper::new(tiny_model(&args));

    assert_eq!(wrapper.num_layers(), 2);
    assert!(
        wrapper.make_caches().is_empty(),
        "the generator's dense cache slice cannot express two caches per layer"
    );

    let layout = wrapper.sequence_state_layout();
    assert_eq!(
        layout.backend,
        mlxcel_core::cache::SequenceStateBackend::ModelOwned,
        "model-owned is what excludes this family from the prompt/prefix cache; the scheduler \
         keys its bail on exactly this value"
    );
    assert_eq!(layout.num_layers, 2);
    assert!(!wrapper.supports_batching());
    assert!(!wrapper.supports_padded_prefill());
    assert!(
        !wrapper.supports_snapshot_reuse(),
        "snapshot reuse is not wired for a dense + rotating cache pair"
    );
    assert_eq!(wrapper.eos_token_ids(), vec![3]);
}

#[test]
fn wrapper_isolates_concurrent_sequences() {
    use mlxcel_core::cache::SequenceId;

    let args = tiny_args(4, 1);
    let wrapper = IQuestLoopCoderWrapper::new(tiny_model(&args));
    let mut unused: Vec<mlxcel_core::layers::KVCache> = Vec::new();

    let a = SequenceId::from_raw(1360);
    let b = SequenceId::from_raw(1361);
    wrapper.prepare_sequence_state(a);
    wrapper.prepare_sequence_state(b);

    // Feed `a` five tokens and `b` two; each must keep its own offsets.
    let _ = wrapper.forward_with_sequence_id(&token_ids(5, 16), Some(a), &mut unused, None);
    let _ = wrapper.forward_with_sequence_id(&token_ids(2, 16), Some(b), &mut unused, None);

    // A sequence's own next step must continue from its own history: `b`'s
    // third token sees 2 past tokens, not `a`'s 5.
    let logits_b = wrapper.forward_with_sequence_id(&token_ids(1, 16), Some(b), &mut unused, None);
    assert!(array_to_vec_f32(&logits_b).iter().all(|v| v.is_finite()));

    // Rebuild `b` from scratch on the fallback slot and compare: same three
    // tokens, so the same logits. If the two sequences shared state, `b`'s
    // history would carry `a`'s tokens and this would not hold.
    let model = wrapper.model();
    let mut fresh = model.make_caches();
    let _ = model.forward_with_caches(&token_ids(2, 16), &mut fresh);
    let expected = model.forward_with_caches(&token_ids(1, 16), &mut fresh);

    assert!(
        max_abs_diff(&logits_b, &expected) < 1e-4,
        "sequence b's state was contaminated by sequence a"
    );

    wrapper.release_sequence_state_by_id(a);
    wrapper.release_sequence_state_by_id(b);
}

#[test]
fn reset_runtime_state_clears_the_fallback_caches() {
    let args = tiny_args(4, 1);
    let wrapper = IQuestLoopCoderWrapper::new(tiny_model(&args));
    let mut unused: Vec<mlxcel_core::layers::KVCache> = Vec::new();

    let _ = wrapper.forward(&token_ids(6, 16), &mut unused, None);
    wrapper.with_fallback_caches(|caches: &mut [LayerCaches]| {
        assert_eq!(caches[0].pass1.offset, 6);
        assert_eq!(caches[0].pass2.offset, 6);
    });

    wrapper.reset_runtime_state();
    wrapper.with_fallback_caches(|caches: &mut [LayerCaches]| {
        assert_eq!(
            caches[0].pass1.offset, 0,
            "a second generation must not continue the first"
        );
        assert_eq!(caches[0].pass2.offset, 0);
    });
}
