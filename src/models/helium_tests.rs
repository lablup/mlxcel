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

//! Unit tests for the Helium config surface, its untrusted-config guards, its
//! weight-shape contract, and above all its RoPE convention.
//!
//! Everything here is checkpoint-free: the config tests parse the real
//! `kyutai/helium-1-preview-2b` field set, and the model tests build tiny
//! synthetic weight maps whose key names and shapes mirror the checkpoint.
//!
//! The tests that matter most are the RoPE ones. Traditional and split-half RoPE
//! produce identically shaped tensors from identical weights, so no shape
//! assertion, no cache assertion and no logits-shape assertion can tell them
//! apart; only the values can. Dropping the flag on any one of the three routes
//! that can rotate Q and K would leave the model loading, generating, and
//! quietly wrong.

use super::{HeliumModel, ModelArgs, validate_weights};
use crate::models::llama3::{Attention, Llama3Model};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{FusedQKVLinear, KVCache};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// Config surface.

/// The `mlx-community/helium-1-preview-2b-4bit` config, field-for-field.
const HELIUM_1_PREVIEW_2B_CONFIG: &str = r#"{
    "architectures": ["HeliumForCausalLM"],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 7040,
    "max_position_embeddings": 4096,
    "mlp_bias": false,
    "model_type": "helium",
    "num_attention_heads": 20,
    "num_hidden_layers": 24,
    "num_key_value_heads": 20,
    "pretraining_tp": 1,
    "quantization": { "group_size": 64, "bits": 4 },
    "quantization_config": { "group_size": 64, "bits": 4 },
    "rms_norm_eps": 1e-08,
    "rope_theta": 100000.0,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.45.0.dev0",
    "use_cache": true,
    "vocab_size": 48000
}"#;

fn helium_1_preview_2b() -> ModelArgs {
    serde_json::from_str(HELIUM_1_PREVIEW_2B_CONFIG).expect("the real config must parse")
}

#[test]
fn the_real_config_parses_and_validates() {
    let args = helium_1_preview_2b();
    assert_eq!(args.model_type, "helium");
    assert_eq!(args.hidden_size, 2560);
    assert_eq!(args.num_hidden_layers, 24);
    assert_eq!(args.intermediate_size, 7040);
    assert_eq!(args.num_attention_heads, 20);
    assert_eq!(args.num_kv_heads(), 20);
    assert_eq!(args.vocab_size, 48000);
    assert_eq!(args.max_position_embeddings, 4096);
    assert!(!args.attention_bias);
    assert!(!args.mlp_bias);
    assert!(!args.tie_word_embeddings);
    assert_eq!(args.rope_theta, 100_000.0);
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);

    // 1e-08 is unusually small for this family (Llama uses 1e-05) and must not
    // be mistaken for an invalid value by the eps guard.
    assert_eq!(args.rms_norm_eps, 1e-8);
    args.validate().expect("the real config must validate");
}

#[test]
fn head_dim_is_derived_from_hidden_size_and_head_count() {
    let args = helium_1_preview_2b();
    // Upstream's attention computes hidden_size // n_heads and never reads the
    // declared field. On this checkpoint the two agree.
    assert_eq!(args.head_dim(), 128);
    assert_eq!(args.head_dim, Some(128));
}

#[test]
fn the_real_checkpoint_shape_signature_matches_what_loading_expects() {
    // The `mlx-community/helium-1-preview-2b-4bit` SafeTensors header, checked
    // against the widths `validate_weights` derives from `config.json`. Pinning
    // the arithmetic here costs nothing, where materializing 559 real tensors
    // in a unit test would cost hundreds of megabytes.
    let args = helium_1_preview_2b();
    let bits = args.bits();
    let group_size = args.group_size();

    // self_attn.{q,k,v,o}_proj.weight rows, and lm_head / embed_tokens rows.
    assert_eq!(args.num_attention_heads * args.head_dim(), 2560);
    assert_eq!(args.num_kv_heads() * args.head_dim(), 2560);
    assert_eq!(args.hidden_size, 2560);
    assert_eq!(args.vocab_size, 48000);
    // mlp.{gate,up}_proj.weight rows.
    assert_eq!(args.intermediate_size, 7040);

    // 4-bit affine packs eight values into each u32 word, so the stored input
    // axis is `in_features * bits / 32` and the scales axis is
    // `in_features / group_size`.
    assert_eq!(args.hidden_size as i32 * bits / 32, 320);
    assert_eq!(args.hidden_size as i32 / group_size, 40);
    assert_eq!(args.intermediate_size as i32 * bits / 32, 880);
    assert_eq!(args.intermediate_size as i32 / group_size, 110);

    // The input-width guard reconstructs the input axis the way MLX does,
    // `scales.shape(-1) * group_size`, so the real scales widths must land back
    // on the config's own values. This is what proves the guard accepts this
    // checkpoint rather than only rejecting broken ones.
    assert_eq!(40 * group_size, args.hidden_size as i32);
    assert_eq!(110 * group_size, args.intermediate_size as i32);
}

#[test]
fn config_validation_rejects_a_head_dim_that_disagrees_with_the_head_split() {
    let mut args = helium_1_preview_2b();
    args.head_dim = Some(64);
    let err = args
        .validate()
        .expect_err("a head_dim that disagrees with hidden_size / num_attention_heads is rejected");
    assert!(err.contains("head_dim"), "{err}");

    // Absent is fine: it resolves to the derived width.
    args.head_dim = None;
    args.validate().unwrap();
    assert_eq!(args.head_dim(), 128);
}

#[test]
fn config_validation_rejects_impossible_architecture_scalars() {
    // Zero heads must be rejected before the divisibility check, because
    // `0.is_multiple_of(0)` is true and `head_dim()` would then divide by zero.
    let mut zero_heads = helium_1_preview_2b();
    zero_heads.num_attention_heads = 0;
    zero_heads.hidden_size = 0;
    zero_heads.head_dim = None;
    assert!(
        zero_heads
            .validate()
            .expect_err("zero heads")
            .contains("num_attention_heads")
    );

    for mutate in [
        (|a: &mut ModelArgs| a.hidden_size = 0) as fn(&mut ModelArgs),
        |a: &mut ModelArgs| a.num_hidden_layers = 0,
        |a: &mut ModelArgs| a.num_hidden_layers = 1 << 20,
        |a: &mut ModelArgs| a.intermediate_size = 0,
        |a: &mut ModelArgs| a.vocab_size = 0,
        |a: &mut ModelArgs| a.vocab_size = 1 << 30,
        |a: &mut ModelArgs| a.max_position_embeddings = 0,
        // 2560 is not divisible by 7.
        |a: &mut ModelArgs| a.num_attention_heads = 7,
        // 20 query heads cannot be grouped into 3 KV heads.
        |a: &mut ModelArgs| a.num_key_value_heads = Some(3),
        |a: &mut ModelArgs| a.num_key_value_heads = Some(0),
    ] {
        let mut args = helium_1_preview_2b();
        args.head_dim = None;
        mutate(&mut args);
        assert!(
            args.validate().is_err(),
            "an impossible architecture scalar must be rejected: {args:?}"
        );
    }
}

#[test]
fn config_validation_rejects_a_rope_configuration_that_would_abort_an_mlx_kernel() {
    // MLX's `fast::rope` requires an even, positive `dims`. It enforces that by
    // throwing, and `fast_rope` crosses the cxx bridge as `UniquePtr` rather
    // than `Result`, so the throw is an uncatchable `std::terminate` at the
    // first forward pass rather than a load error. head_dim comes out odd here
    // (2560 / 512 = 5).
    let mut odd = helium_1_preview_2b();
    odd.num_attention_heads = 512;
    odd.num_key_value_heads = Some(512);
    odd.head_dim = None;
    assert!(
        odd.validate()
            .expect_err("odd head_dim")
            .contains("head_dim")
    );

    for theta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut args = helium_1_preview_2b();
        args.rope_theta = theta;
        assert!(
            args.validate().is_err(),
            "rope_theta {theta} must be rejected: RoPE exponentiates it, and a bad base NaNs \
             every rotated channel without throwing"
        );
    }

    // The real value, and the smallest legal head width, both stay accepted.
    let mut minimal = helium_1_preview_2b();
    minimal.num_attention_heads = 1280;
    minimal.num_key_value_heads = Some(1280);
    minimal.head_dim = None;
    minimal.validate().expect("a head_dim of 2 is legal");
}

#[test]
fn config_validation_rejects_an_rms_norm_eps_that_would_nan_every_hidden_state() {
    for eps in [0.0, -1e-5, f32::NAN, f32::INFINITY] {
        let mut args = helium_1_preview_2b();
        args.rms_norm_eps = eps;
        assert!(
            args.validate().is_err(),
            "rms_norm_eps {eps} must be rejected: `fast::rms_norm` never inspects it, so a bad \
             value produces NaN hidden states with no error at all"
        );
    }
    // The checkpoint's own unusually small eps must stay accepted.
    let mut small = helium_1_preview_2b();
    small.rms_norm_eps = 1e-8;
    small.validate().unwrap();
}

#[test]
fn config_validation_rejects_a_quantization_block_that_would_abort_an_mlx_kernel() {
    for (group_size, bits) in [(64, 0), (64, 33), (64, -4), (0, 4), (-1, 4)] {
        let mut args = helium_1_preview_2b();
        args.quantization = Some(super::Quantization { group_size, bits });
        assert!(
            args.validate().is_err(),
            "quantization group_size {group_size} / bits {bits} must be rejected"
        );
    }
    // A range check, not an allowlist: mixed-precision exports re-derive an
    // effective bit width from the tensor shapes, so unusual-but-legal values
    // stay accepted.
    for (group_size, bits) in [(64, 4), (32, 8), (128, 6), (64, 32), (1, 1)] {
        let mut args = helium_1_preview_2b();
        args.quantization = Some(super::Quantization { group_size, bits });
        args.validate().unwrap();
    }
}

#[test]
fn eos_token_ids_come_from_the_config_not_from_the_llama3_backbone() {
    let args = helium_1_preview_2b();
    assert_eq!(args.eos_token_ids(), vec![2]);
    // Llama 3's hardcoded stop tokens are outside Helium's 48000-entry
    // vocabulary, so delegating would mean generation never stops early.
    assert!(!args.eos_token_ids().contains(&128001));

    let list: ModelArgs = serde_json::from_str(
        &HELIUM_1_PREVIEW_2B_CONFIG.replace("\"eos_token_id\": 2", "\"eos_token_id\": [2, 5]"),
    )
    .unwrap();
    assert_eq!(list.eos_token_ids(), vec![2, 5]);
}

// The RoPE convention.

#[test]
fn to_llama3_args_asks_the_shared_decoder_for_traditional_rope() {
    let args = helium_1_preview_2b();
    let llama = args.to_llama3_args();
    assert!(
        llama.rope_traditional,
        "Helium's single architectural difference is `nn.RoPE(..., traditional=True)`"
    );
    assert_eq!(llama.head_dim, Some(128));
    assert_eq!(llama.num_key_value_heads, Some(20));
    assert_eq!(llama.rope_theta, 100_000.0);
    assert!(llama.rope_scaling.is_none());
}

#[test]
fn helium_still_needs_the_conversion_because_its_config_omits_the_key() {
    // #931 made `llama3::ModelArgs::rope_traditional` deserializable, which
    // could look like it makes `to_llama3_args` redundant. It does not: Helium's
    // convention is fixed in upstream code, so the published `config.json`
    // carries no `rope_traditional` key and parsing it directly yields `false`.
    // The conversion is what supplies the flag, and this pins that the two are
    // not interchangeable. The parse itself is covered in `llama3_tests.rs`.
    assert!(
        !HELIUM_1_PREVIEW_2B_CONFIG.contains("rope_traditional"),
        "the pinned upstream config must not declare the key, or this test proves nothing"
    );
    let direct: crate::models::llama3::ModelArgs =
        serde_json::from_str(HELIUM_1_PREVIEW_2B_CONFIG).unwrap();
    assert!(
        !direct.rope_traditional,
        "parsing a Helium config straight into the shared args cannot recover the convention"
    );
    assert!(helium_1_preview_2b().to_llama3_args().rope_traditional);
}

#[test]
fn traditional_and_split_half_rope_are_different_rotations() {
    // Why any of this matters: the two conventions rotate different channel
    // pairs, so they disagree on values while agreeing on every shape.
    let x = filled(&[1, 2, 4, 8]);
    let traditional = mlxcel_core::fast_rope(&x, 8, true, 100_000.0, 1.0, 0);
    let split_half = mlxcel_core::fast_rope(&x, 8, false, 100_000.0, 1.0, 0);
    assert_eq!(
        mlxcel_core::array_shape(&traditional),
        mlxcel_core::array_shape(&split_half),
        "the two conventions are shape-identical, which is exactly why a shape test cannot \
         distinguish them"
    );
    assert!(
        max_abs_diff(&traditional, &split_half) > 1e-2,
        "the two conventions must actually differ, or every test below is vacuous"
    );
}

#[test]
fn attention_carries_the_traditional_flag_from_its_args() {
    let args = tiny_args();
    let weights = tiny_weights(&args);

    let helium =
        Attention::from_weights(&weights, &args.to_llama3_args(), "model.layers.0.self_attn")
            .unwrap();
    assert!(helium.rope_traditional);

    let mut llama_args = args.to_llama3_args();
    llama_args.rope_traditional = false;
    let llama = Attention::from_weights(&weights, &llama_args, "model.layers.0.self_attn").unwrap();
    assert!(
        !llama.rope_traditional,
        "args that do not ask for the interleaved rotation must get the split-half one"
    );
}

#[test]
fn helium_logits_differ_from_the_same_weights_with_split_half_rope() {
    // THE regression test. If the flag is ever dropped between the config and
    // `fast_rope`, both models become the same graph and this fails.
    let args = tiny_args();
    let weights = tiny_weights(&args);

    let traditional = Llama3Model::from_weights(&weights, &args.to_llama3_args()).unwrap();
    let mut split_half_args = args.to_llama3_args();
    split_half_args.rope_traditional = false;
    let split_half = Llama3Model::from_weights(&weights, &split_half_args).unwrap();

    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7, 8], &[1, 8]);
    let mut a_caches = traditional.make_caches();
    let mut b_caches = split_half.make_caches();
    let a = traditional.forward(&prompt, &mut a_caches, None);
    let b = split_half.forward(&prompt, &mut b_caches, None);

    assert_eq!(mlxcel_core::array_shape(&a), mlxcel_core::array_shape(&b));

    // Two thresholds, because an absolute one alone can go blind.
    //
    // The gap this test measures is only meaningful next to the magnitude of
    // the logits it is measured on. An earlier version of `filled` made every
    // head nearly collinear and collapsed the gap to 2.4e-5 against a logits
    // scale of 5.8, a relative separation of 4e-6: green, and useless. That
    // particular collapse is now caught by the absolute floor, but its mirror
    // image is not. A future change that widens `tiny_args` or rescales
    // `filled` can grow the logits without growing the separation, and an
    // unanchored absolute floor would keep passing while the test drifts back
    // toward blindness.
    //
    // Measured on the pinned inputs below: gap 1.19e-2, logits scale 6.89,
    // relative separation 1.7e-3. That is ~12x the absolute floor and ~17x the
    // relative floor, and it is reproducible run to run because `filled` is a
    // fixed-seed LCG over fixed shapes. Both floors must keep real margin; if a
    // refactor pushes either one close, widen the model rather than lowering
    // the floor.
    let gap = max_abs_diff(&a, &b);
    let logits_scale = max_abs(&a);
    assert!(
        gap > 1e-3,
        "traditional and split-half RoPE must produce different logits from the same weights \
         (gap {gap}, logits scale {logits_scale})"
    );
    assert!(
        gap > logits_scale * 1e-4,
        "the two rotations must be separated by a meaningful fraction of the logits, not just by \
         an absolute epsilon (gap {gap}, logits scale {logits_scale})"
    );
}

#[test]
fn every_rope_route_honors_the_flag_consistently() {
    // Three code paths can rotate Q and K on this decoder: the single-sequence
    // graph path, the batched path (`fast_rope_batched`), and the two fused
    // quantized launchers, which cannot express the convention and are therefore
    // bypassed (see `the_fused_qkv_rope_launcher_cannot_express_traditional_rope`
    // and `Attention::forward`).
    //
    // Honoring the flag on one route and dropping it on another would make a
    // sequence decode differently depending on whether it was scheduled alone or
    // in a batch. This pins the batched route against the single-sequence route,
    // and scales the tolerance against the split-half difference so the
    // assertion cannot pass by both being wrong in the same way.
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let helium = HeliumModel::from_weights(&weights, &args).unwrap();

    let single_ids = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7, 8], &[1, 8]);
    let mut single_caches = LanguageModel::make_caches(&helium);
    let single = LanguageModel::forward(&helium, &single_ids, &mut single_caches, None);

    let batched_ids =
        mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8], &[2, 8]);
    let mut row0 = LanguageModel::make_caches(&helium);
    let mut row1 = LanguageModel::make_caches(&helium);
    let mut batch: Vec<&mut [KVCache]> = vec![row0.as_mut_slice(), row1.as_mut_slice()];
    let batched = LanguageModel::forward_batched(&helium, &batched_ids, &mut batch, None);

    let batched_row0 = mlxcel_core::slice(&batched, &[0, 0, 0], &[1, i32::MAX, i32::MAX]);
    let route_gap = max_abs_diff(&single, &batched_row0);

    let mut split_half_args = args.to_llama3_args();
    split_half_args.rope_traditional = false;
    let split_half = Llama3Model::from_weights(&weights, &split_half_args).unwrap();
    let mut split_half_caches = split_half.make_caches();
    let split_half_logits = split_half.forward(&single_ids, &mut split_half_caches, None);
    let convention_gap = max_abs_diff(&single, &split_half_logits);

    assert!(
        route_gap < convention_gap / 100.0,
        "the batched route must apply the same rotation as the single-sequence route \
         (route gap {route_gap}, convention gap {convention_gap})"
    );
}

#[test]
fn the_fused_qkv_rope_launcher_cannot_express_traditional_rope() {
    // This is the fact the bypass in `Attention::forward` rests on, so it is
    // asserted rather than assumed. `fused_qkv_project_split_rope` applies RoPE
    // inside C++ with `traditional` hardcoded to `false` and takes no flag, and
    // it is reachable only for *quantized* weights, which is exactly what the
    // Helium validation checkpoint is.
    //
    // If someone later teaches the launcher the flag, this test starts failing
    // and points at the gate that should then be removed.
    let hidden = 64;
    let head_dim = 32;
    let heads = 2;
    let group_size = 32;
    let bits = 4;

    let mut weights = WeightMap::new();
    for name in ["q_proj", "k_proj", "v_proj"] {
        let dense = filled(&[hidden, hidden]);
        let quantized = mlxcel_core::quantize_weights(&dense, group_size, bits);
        weights.insert(
            format!("attn.{name}.weight"),
            mlxcel_core::quantized_weights_w(&quantized),
        );
        weights.insert(
            format!("attn.{name}.scales"),
            mlxcel_core::quantized_weights_scales(&quantized),
        );
        if mlxcel_core::quantized_weights_has_biases(&quantized) {
            weights.insert(
                format!("attn.{name}.biases"),
                mlxcel_core::quantized_weights_biases(&quantized),
            );
        }
    }

    let fused = FusedQKVLinear::from_weights_separate(
        &weights, "attn", group_size, bits, heads, heads, head_dim,
    )
    .unwrap();

    let x = filled(&[1, 3, hidden]);
    let (fused_q, _, _) = fused
        .forward_split_rope_quantized(&x, head_dim, 100_000.0, 0)
        .expect("quantized weights take the fused launcher");

    // The same projection, rotated in Rust with each convention.
    let (raw_q, _, _) = fused.forward(&x);
    let raw_q = mlxcel_core::reshape(&raw_q, &[1, 3, heads, head_dim]);
    let raw_q = mlxcel_core::transpose_axes(&raw_q, &[0, 2, 1, 3]);
    let split_half = mlxcel_core::fast_rope(&raw_q, head_dim, false, 100_000.0, 1.0, 0);
    let traditional = mlxcel_core::fast_rope(&raw_q, head_dim, true, 100_000.0, 1.0, 0);

    assert!(
        max_abs_diff(&fused_q, &split_half) < 1e-4,
        "the fused launcher applies the split-half rotation"
    );
    assert!(
        max_abs_diff(&fused_q, &traditional) > 1e-3,
        "the fused launcher cannot produce the traditional rotation, which is why Helium must \
         not route through it"
    );
}

// Weight-shape contract.

#[test]
fn a_well_formed_checkpoint_loads_and_generates() {
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = HeliumModel::from_weights(&weights, &args).unwrap();

    assert_eq!(LanguageModel::num_layers(&model), args.num_hidden_layers);
    assert_eq!(LanguageModel::eos_token_ids(&model), vec![2]);

    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let mut caches = LanguageModel::make_caches(&model);
    let logits = LanguageModel::forward(&model, &prompt, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 3, args.vocab_size as i32]
    );

    let next = mlxcel_core::from_slice_i32(&[4], &[1, 1]);
    let step = LanguageModel::forward(&model, &next, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&step),
        vec![1, 1, args.vocab_size as i32]
    );
}

#[test]
fn loading_rejects_an_embedding_table_the_config_overstates() {
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    weights.insert(
        "model.embed_tokens.weight".into(),
        filled(&[args.vocab_size as i32 - 1, args.hidden_size as i32]),
    );
    let err = validate_weights(&weights, &args).expect_err("an undersized table is rejected");
    assert!(err.contains("vocab_size"), "{err}");
}

#[test]
fn loading_rejects_a_projection_whose_shape_disagrees_with_the_config() {
    let args = tiny_args();
    let head_dim = args.head_dim() as i32;

    // A q_proj sized for a different head count. Every offset stays in bounds,
    // so nothing downstream would fault; the reshape in `Attention::forward`
    // would throw inside MLX instead, which is an uncatchable abort.
    let mut wrong_rows = tiny_weights(&args);
    wrong_rows.insert(
        "model.layers.0.self_attn.q_proj.weight".into(),
        filled(&[head_dim, args.hidden_size as i32]),
    );
    assert!(validate_weights(&wrong_rows, &args).is_err());

    // A norm of the wrong width would broadcast-fail inside MLX.
    let mut wrong_norm = tiny_weights(&args);
    wrong_norm.insert(
        "model.layers.1.post_attention_layernorm.weight".into(),
        filled(&[args.hidden_size as i32 + 1]),
    );
    assert!(
        validate_weights(&wrong_norm, &args).is_err(),
        "every layer is checked, not just layer 0"
    );

    // A transposed projection is caught by name rather than silently loaded.
    let mut transposed = tiny_weights(&args);
    transposed.insert(
        "model.layers.0.mlp.gate_proj.weight".into(),
        filled(&[args.hidden_size as i32, args.intermediate_size as i32]),
    );
    assert!(validate_weights(&transposed, &args).is_err());
}

#[test]
fn loading_rejects_a_partially_quantized_attention_block() {
    // The fused QKV loader decides "is this quantized?" from `q_proj.scales`
    // alone and then concatenates all three projections. A checkpoint where they
    // disagree aborts inside `concatenate` rather than failing to load.
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    weights.insert(
        "model.layers.0.self_attn.q_proj.scales".into(),
        filled(&[
            (args.num_attention_heads * args.head_dim()) as i32,
            scale_cols(&args, args.hidden_size),
        ]),
    );
    let err = validate_weights(&weights, &args).expect_err("mixed quantization is rejected");
    assert!(err.contains("quantized"), "{err}");
}

#[test]
fn loading_rejects_an_attention_block_with_partial_quantization_biases() {
    // The fused loader keeps the affine `biases` only when all three
    // projections carry one, and silently drops the whole set otherwise, which
    // dequantizes the survivors without their zero points. Nothing throws.
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    let q_rows = (args.num_attention_heads * args.head_dim()) as i32;
    let kv_rows = (args.num_kv_heads() * args.head_dim()) as i32;
    let cols = scale_cols(&args, args.hidden_size);
    for (name, rows) in [("q_proj", q_rows), ("k_proj", kv_rows), ("v_proj", kv_rows)] {
        weights.insert(
            format!("model.layers.0.self_attn.{name}.scales"),
            filled(&[rows, cols]),
        );
    }
    weights.insert(
        "model.layers.0.self_attn.q_proj.biases".into(),
        filled(&[q_rows, cols]),
    );
    let err = validate_weights(&weights, &args).expect_err("partial biases are rejected");
    assert!(err.contains("quantization biases"), "{err}");
}

#[test]
fn a_consistently_quantized_attention_block_is_accepted() {
    // The positive control for the two rejections below: a quantized block whose
    // scales describe exactly `hidden_size` must still load, so the width check
    // cannot be passing by rejecting everything quantized.
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    quantize_attention_block(&args, &mut weights, 0, scale_cols(&args, args.hidden_size));
    validate_weights(&weights, &args).unwrap();
}

#[test]
fn loading_rejects_a_quantized_projection_packed_for_a_different_input_width() {
    // Packing compresses the input axis only, so a projection built for a
    // different `hidden_size` still has exactly the right number of rows and
    // survives every row check, and q/k/v agreeing with each other proves
    // nothing about whether any of them agrees with the config. MLX
    // reconstructs the input width as `scales.shape(-1) * group_size` and
    // throws from `extract_quantized_matmul_dims` when it disagrees with the
    // activation, which crosses the cxx bridge as an uncatchable abort at the
    // first forward pass rather than a load error.
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    let wrong = scale_cols(&args, args.hidden_size * 2);
    quantize_attention_block(&args, &mut weights, 0, wrong);
    let err = validate_weights(&weights, &args).expect_err("a mis-packed input width is rejected");
    assert!(err.contains("input width"), "{err}");
}

#[test]
fn loading_rejects_a_quantized_output_head_packed_for_a_different_input_width() {
    // Same defect on the untied head, where the throw comes from the final
    // `quantized_matmul` instead of an attention projection.
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    let vocab = args.vocab_size as i32;
    weights.insert(
        "lm_head.scales".into(),
        filled(&[vocab, scale_cols(&args, args.hidden_size * 2)]),
    );
    let err = validate_weights(&weights, &args).expect_err("a mis-packed head is rejected");
    assert!(err.contains("input width"), "{err}");
}

#[test]
fn loading_rejects_quantization_biases_that_disagree_with_the_scales() {
    // MLX requires the affine zero points to have the same shape as the scales
    // and throws otherwise. Presence alone is not enough to check.
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    let cols = scale_cols(&args, args.hidden_size);
    quantize_attention_block(&args, &mut weights, 0, cols);
    weights.insert(
        "model.layers.0.self_attn.q_proj.biases".into(),
        filled(&[
            (args.num_attention_heads * args.head_dim()) as i32,
            cols + 1,
        ]),
    );
    let err = validate_weights(&weights, &args).expect_err("mis-shaped zero points are rejected");
    assert!(err.contains("same shape"), "{err}");
}

#[test]
fn loading_rejects_an_attention_block_with_a_partial_dense_bias() {
    // The dense `.bias` set has the same all-or-nothing rule as the affine
    // `.biases` set: the fused loader concatenates q/k/v biases only when all
    // three are present and drops the whole set otherwise, so a checkpoint
    // carrying a bias on only some of them loads and silently runs those
    // projections unbiased.
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    weights.insert(
        "model.layers.0.self_attn.q_proj.bias".into(),
        filled(&[(args.num_attention_heads * args.head_dim()) as i32]),
    );
    let err = validate_weights(&weights, &args).expect_err("a partial bias set is rejected");
    assert!(err.contains("all carry a bias"), "{err}");
}

#[test]
fn loading_rejects_a_missing_untied_head() {
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    weights.remove("lm_head.weight");
    assert!(validate_weights(&weights, &args).is_err());

    // The same checkpoint is fine once the config says the head is tied.
    let mut tied = tiny_args();
    tied.tie_word_embeddings = true;
    validate_weights(&weights, &tied).unwrap();
}

// Helpers.

/// A Helium small enough to run in a unit test, keeping the checkpoint's own
/// head split (multi-head, `num_key_value_heads == num_attention_heads`) and its
/// `rope_theta`.
fn tiny_args() -> ModelArgs {
    serde_json::from_str(
        r#"{
            "model_type": "helium",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "rms_norm_eps": 1e-08,
            "rope_theta": 100000.0,
            "attention_bias": false,
            "mlp_bias": false,
            "tie_word_embeddings": false,
            "eos_token_id": 2,
            "max_position_embeddings": 128,
            "vocab_size": 32
        }"#,
    )
    .unwrap()
}

/// A float checkpoint with the Helium key layout: `model.embed_tokens`,
/// `model.layers.N.self_attn.{q,k,v,o}_proj`,
/// `model.layers.N.mlp.{gate,up,down}_proj`, the two RMSNorms, `model.norm`, and
/// an untied `lm_head`.
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

/// Scales columns for a quantized `[out, in_features]` weight: one group per
/// `group_size` input channels, which is the width MLX reconstructs the input
/// axis from.
fn scale_cols(args: &ModelArgs, in_features: usize) -> i32 {
    in_features as i32 / args.group_size()
}

/// Turn one layer's `q_proj` / `k_proj` / `v_proj` / `o_proj` into a quantized
/// block by adding `.scales` and `.biases` with `cols` groups each, leaving the
/// `.weight` tensors from [`tiny_weights`] in place (only the shapes are read).
fn quantize_attention_block(args: &ModelArgs, weights: &mut WeightMap, layer: usize, cols: i32) {
    let q_rows = (args.num_attention_heads * args.head_dim()) as i32;
    let kv_rows = (args.num_kv_heads() * args.head_dim()) as i32;
    let hidden = args.hidden_size as i32;
    for (name, rows) in [
        ("q_proj", q_rows),
        ("k_proj", kv_rows),
        ("v_proj", kv_rows),
        ("o_proj", hidden),
    ] {
        weights.insert(
            format!("model.layers.{layer}.self_attn.{name}.scales"),
            filled(&[rows, cols]),
        );
        weights.insert(
            format!("model.layers.{layer}.self_attn.{name}.biases"),
            filled(&[rows, cols]),
        );
    }
}

/// Deterministic pseudo-random filler in `[-0.5, 0.5)`.
///
/// A short repeating pattern (the obvious `(i % 7 - 3) * 0.1`) makes every head
/// nearly collinear, which flattens the attention scores and shrinks the
/// difference between two RoPE conventions to a few parts per million at the
/// logits. That would leave the convention tests technically green and
/// practically blind, so the filler is a fixed-seed LCG instead: still
/// bit-reproducible, but with enough spread for a rotation to matter.
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

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    max_abs(&mlxcel_core::subtract(a, b))
}

/// Largest absolute value in `a`, used to scale a difference against the
/// magnitude it was measured on.
fn max_abs(a: &MlxArray) -> f32 {
    mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(a)))
}
