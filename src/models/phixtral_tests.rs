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

//! Unit tests for the Phixtral loader.
//!
//! Everything here is checkpoint-free. Four groups carry more weight than the
//! rest, because they cover what a passing real-checkpoint run cannot tell apart
//! from a wrong implementation:
//!
//! 1. **Detection.** A phixtral checkpoint declares `phi-msft`, the same string
//!    dense Phi-2 declares, so the split is decided by `num_local_experts` and
//!    nothing else. Both directions are pinned.
//! 2. **The router's softmax domain.** Upstream softmaxes the `k` gathered
//!    logits; softmaxing the full expert row first and gathering afterwards
//!    normalizes by a different denominator and leaves the output finite. The
//!    two are computed side by side and asserted to differ before the
//!    implementation is checked against the correct one.
//! 3. **The per-expert bias.** `SwitchLinear` implements no bias, so it is
//!    gathered and added in this module. Dropping it entirely still produces
//!    fluent text, so the gather is pinned against hand-computed values.
//! 4. **Causal prefill without a supplied mask.** Generation calls `forward`
//!    with `mask == None`; a fully bidirectional prefill is fluent and wrong.

use super::{
    ModelArgs, PhixtralModel, PhixtralMoe, PhixtralSwitchMlp, Quantization, validate_weights,
};
use crate::models::detection::detect_phi_model_type;
use crate::models::{ModelType, switch_layers::SwitchLinear};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// The real checkpoint's config.

/// `mlabonne/phixtral-4x2_8`'s `config.json`, field for field, with the
/// `quantization` block an `mlx_lm convert -q` conversion adds.
const PHIXTRAL_CONFIG: &str = r#"{
    "activation_function": "gelu_new",
    "architectures": ["PhiForCausalLM"],
    "attn_pdrop": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_phi.PhiConfig",
        "AutoModelForCausalLM": "modeling_phi.PhiForCausalLM"
    },
    "embd_pdrop": 0.0,
    "flash_attn": false,
    "flash_rotary": false,
    "fused_dense": false,
    "img_processor": null,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "phi-msft",
    "n_embd": 2560,
    "n_head": 32,
    "n_head_kv": null,
    "n_inner": null,
    "n_layer": 32,
    "n_positions": 2048,
    "num_experts_per_tok": 2,
    "num_local_experts": 4,
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"},
    "resid_pdrop": 0.1,
    "rotary_dim": 32,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.35.2",
    "vocab_size": 51200
}"#;

fn phixtral_args() -> ModelArgs {
    serde_json::from_str(PHIXTRAL_CONFIG).expect("phixtral config parses")
}

// Config parsing.

#[test]
fn the_real_config_parses_and_validates() {
    let args = phixtral_args();
    assert_eq!(args.model_type, "phi-msft");
    assert_eq!(args.n_embd, 2560);
    assert_eq!(args.n_head, 32);
    assert_eq!(args.n_layer, 32);
    assert_eq!(args.vocab_size, 51200);
    assert_eq!(args.rotary_dim, 32);
    assert_eq!(args.num_local_experts, 4);
    assert_eq!(args.num_experts_per_tok, 2);
    assert_eq!(args.layer_norm_epsilon, 1e-5);
    assert_eq!(args.n_inner, None);
    assert!(!args.tie_word_embeddings);
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);

    // Derived geometry the checkpoint confirms: Wqkv is [7680, 2560], head 80,
    // experts [4, 10240, 2560].
    assert_eq!(args.head_dim(), 80);
    assert_eq!(args.qkv_out_features(), 7680);
    assert_eq!(args.intermediate_size(), 10240);
    args.validate().expect("the shipped config is accepted");
}

#[test]
fn the_dimensions_come_from_the_spellings_the_checkpoint_uses() {
    // Upstream's ModelArgs names `model_dim` / `num_heads` / `num_layers` /
    // `num_vocab`, none of which appear in a phixtral config.json, so upstream
    // silently takes its defaults for all four. This loader reads the config's
    // own `n_embd` / `n_head` / `n_layer` / `vocab_size`, which is why a
    // differently-sized phixtral loads here and not there.
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "phi-msft", "n_embd": 1280, "n_head": 16, "n_layer": 8,
            "vocab_size": 32000, "rotary_dim": 16, "num_local_experts": 2,
            "num_experts_per_tok": 1}"#,
    )
    .expect("a non-default-size phixtral config parses");
    assert_eq!(args.n_embd, 1280);
    assert_eq!(args.n_head, 16);
    assert_eq!(args.n_layer, 8);
    assert_eq!(args.vocab_size, 32000);
    assert_eq!(args.head_dim(), 80);
    assert_eq!(args.intermediate_size(), 5120);
    args.validate().expect("the shrunk config is valid");
}

#[test]
fn upstreams_own_field_names_are_accepted_as_aliases() {
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "phixtral", "model_dim": 1024, "num_heads": 16,
            "num_layers": 4, "num_vocab": 1000, "rotary_dim": 16}"#,
    )
    .expect("upstream's spellings parse");
    assert_eq!(args.n_embd, 1024);
    assert_eq!(args.n_head, 16);
    assert_eq!(args.n_layer, 4);
    assert_eq!(args.vocab_size, 1000);
}

#[test]
fn an_explicit_n_inner_overrides_the_four_x_default() {
    let mut args = phixtral_args();
    assert_eq!(args.intermediate_size(), 4 * 2560);
    args.n_inner = Some(7000);
    assert_eq!(args.intermediate_size(), 7000);
}

// Detection.

#[test]
fn the_phi_msft_arm_splits_on_num_local_experts() {
    // The whole point of the corrected issue: no checkpoint declares
    // `model_type: "phixtral"`, so the discriminator has to be the expert count.
    let sparse: serde_json::Value =
        serde_json::from_str(PHIXTRAL_CONFIG).expect("phixtral config parses");
    assert_eq!(detect_phi_model_type(&sparse), ModelType::Phixtral);

    // Dense Phi-2 declares the same model_type and no expert count.
    let dense: serde_json::Value = serde_json::from_str(
        r#"{"model_type": "phi-msft", "n_embd": 2560, "n_head": 32, "n_layer": 32}"#,
    )
    .expect("a dense phi config parses");
    assert_eq!(detect_phi_model_type(&dense), ModelType::Phi);

    // One expert is a dense MLP; routing it through the sparse block would be a
    // needless indirection over the same arithmetic.
    let single: serde_json::Value =
        serde_json::from_str(r#"{"model_type": "phi-msft", "num_local_experts": 1}"#)
            .expect("config parses");
    assert_eq!(detect_phi_model_type(&single), ModelType::Phi);
}

// Config guards.

#[test]
fn a_top_k_larger_than_the_expert_count_is_rejected() {
    // Upstream calls argpartition(kth = k - 1) over a row of num_local_experts
    // scores; MLX throws when that is out of range, and an MLX C++ exception
    // crossing the cxx bridge is an uncatchable abort at the first forward pass.
    let mut args = phixtral_args();
    args.num_experts_per_tok = 5;
    let err = args.validate().expect_err("5 of 4 experts is out of range");
    assert!(err.contains("num_experts_per_tok"), "{err}");

    args.num_experts_per_tok = 0;
    let err = args
        .validate()
        .expect_err("zero selected experts is invalid");
    assert!(err.contains("num_experts_per_tok"), "{err}");
}

#[test]
fn a_rotary_width_mlx_would_throw_on_is_rejected() {
    let mut args = phixtral_args();
    args.rotary_dim = 0;
    assert!(args.validate().is_err(), "a zero rotary width is rejected");

    let mut args = phixtral_args();
    args.rotary_dim = 81; // the head is 80 wide
    let err = args
        .validate()
        .expect_err("a rotary width wider than the head is rejected");
    assert!(err.contains("rotary_dim"), "{err}");

    let mut args = phixtral_args();
    args.rotary_dim = 33;
    let err = args
        .validate()
        .expect_err("an odd rotary width is rejected");
    assert!(err.contains("even"), "{err}");
}

#[test]
fn an_indivisible_n_embd_is_rejected_before_it_divides() {
    let mut args = phixtral_args();
    args.n_embd = 2561;
    let err = args
        .validate()
        .expect_err("2561 is not divisible by 32 heads");
    assert!(err.contains("divisible by n_head"), "{err}");
}

#[test]
fn a_zero_scalar_is_rejected_before_it_divides() {
    for mutate in [
        (|a: &mut ModelArgs| a.n_head = 0) as fn(&mut ModelArgs),
        |a: &mut ModelArgs| a.n_embd = 0,
        |a: &mut ModelArgs| a.n_layer = 0,
        |a: &mut ModelArgs| a.vocab_size = 0,
        |a: &mut ModelArgs| a.num_local_experts = 0,
        |a: &mut ModelArgs| a.layer_norm_epsilon = 0.0,
        |a: &mut ModelArgs| a.rope_theta = 0.0,
    ] {
        let mut args = phixtral_args();
        mutate(&mut args);
        assert!(
            args.validate().is_err(),
            "a zero scalar must be rejected at load"
        );
    }
}

// Helpers.

fn to_array(flat: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(flat, shape)
}

fn read_all(array: &MlxArray) -> Vec<f32> {
    let flat = mlxcel_core::reshape(array, &[-1]);
    let n = mlxcel_core::array_shape(&flat)[0];
    (0..n)
        .map(|i| mlxcel_core::item_f32(&mlxcel_core::slice(&flat, &[i], &[i + 1])))
        .collect()
}

fn softmax_host(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|v| v / sum).collect()
}

// Routing.

/// A router whose `gate` is the identity, so the input row *is* the logit row,
/// over experts that emit a one-hot vector naming themselves.
///
/// Identity experts would not work here: every expert would emit the same vector
/// and the routing weights sum to 1, so the block's output would be that vector
/// whatever the weights are. Making expert `e` emit `onehot(e)` instead turns
/// the output into the routing weight vector itself, which is exactly what the
/// test needs to observe.
///
/// `fc1` is zero with a zero bias, so `gelu(0) = 0` and `fc2` contributes only
/// its bias row.
fn onehot_expert_moe(num_experts: usize, top_k: usize) -> PhixtralMoe {
    let hidden = num_experts;
    let mut identity = vec![0.0f32; num_experts * hidden];
    for (i, row) in identity.chunks_mut(hidden).enumerate() {
        row[i] = 1.0;
    }
    let mut weights = WeightMap::new();
    weights.insert(
        "gate.weight".into(),
        to_array(&identity, &[num_experts as i32, hidden as i32]),
    );

    let zeros = vec![0.0f32; num_experts * hidden * hidden];
    let stacked = [num_experts as i32, hidden as i32, hidden as i32];
    for leaf in ["fc1", "fc2"] {
        weights.insert(
            format!("switch_mlp.{leaf}.weight"),
            to_array(&zeros, &stacked),
        );
    }

    PhixtralMoe {
        gate: UnifiedLinear::from_weights(&weights, "gate", 64, 4).expect("gate loads"),
        switch_mlp: PhixtralSwitchMlp {
            fc1: SwitchLinear::from_weights(&weights, "switch_mlp.fc1", 64, 4).expect("fc1 loads"),
            fc2: SwitchLinear::from_weights(&weights, "switch_mlp.fc2", 64, 4).expect("fc2 loads"),
            fc1_bias: to_array(
                &vec![0.0; num_experts * hidden],
                &[num_experts as i32, hidden as i32],
            ),
            // Expert e's bias row is onehot(e).
            fc2_bias: to_array(&identity, &[num_experts as i32, hidden as i32]),
        },
        top_k: top_k as i32,
    }
}

#[test]
fn the_router_softmaxes_the_selected_logits_not_the_full_row() {
    // The substitution that looks right: softmax the whole expert row, then
    // gather. That normalizes by the sum over all 4 experts instead of over the
    // selected 2, so every routed weight is smaller and the output stays finite.
    let logits = [3.0f32, 1.0, 0.5, -2.0];
    let top_k = 2;

    let correct = softmax_host(&logits[..top_k]);
    let full_row = softmax_host(&logits);
    let wrong = &full_row[..top_k];

    // The two really do differ, so this test is not vacuous.
    assert!(
        (correct[0] - wrong[0]).abs() > 1e-3,
        "correct {correct:?} vs gather-after-softmax {wrong:?}"
    );
    assert!((correct.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!((wrong.iter().sum::<f32>() - 1.0).abs() > 1e-3);

    // With one-hot experts the block's output IS the routing weight vector, so
    // channel e carries the weight expert e was combined at.
    let moe = onehot_expert_moe(4, top_k);
    let out = read_all(&moe.forward(&to_array(&logits, &[1, 4])));

    // Experts 0 and 1 hold the two largest logits.
    for (channel, weight) in correct.iter().enumerate().take(top_k) {
        assert!(
            (out[channel] - weight).abs() < 1e-3,
            "channel {channel}: got {}, expected {weight}; a full-row softmax \
             would have given {} (full output {out:?})",
            out[channel],
            wrong[channel]
        );
    }
    // The unselected experts contribute nothing.
    for (channel, value) in out.iter().enumerate().skip(top_k) {
        assert!(
            value.abs() < 1e-4,
            "channel {channel} was not selected but contributed {value}"
        );
    }
}

#[test]
fn the_expert_bias_is_gathered_per_selected_expert() {
    // `SwitchLinear` implements no bias, so the per-expert row is gathered and
    // added in this module. Dropping it produces fluent text, so the gather is
    // pinned here: with zero weights, each expert's output is exactly its own
    // fc2 bias row.
    let (num_experts, hidden, top_k) = (4usize, 3usize, 2usize);
    let mut weights = WeightMap::new();
    let zeros = vec![0.0f32; num_experts * hidden * hidden];
    let stacked = [num_experts as i32, hidden as i32, hidden as i32];
    for leaf in ["fc1", "fc2"] {
        weights.insert(format!("{leaf}.weight"), to_array(&zeros, &stacked));
    }

    // Expert e's fc2 bias row is [e, e, e]; fc1's is zero, so gelu(0) = 0 and
    // fc2 contributes only its bias.
    let fc1_bias = vec![0.0f32; num_experts * hidden];
    let mut fc2_bias = vec![0.0f32; num_experts * hidden];
    for e in 0..num_experts {
        for c in 0..hidden {
            fc2_bias[e * hidden + c] = e as f32;
        }
    }

    let mlp = PhixtralSwitchMlp {
        fc1: SwitchLinear::from_weights(&weights, "fc1", 64, 4).expect("fc1 loads"),
        fc2: SwitchLinear::from_weights(&weights, "fc2", 64, 4).expect("fc2 loads"),
        fc1_bias: to_array(&fc1_bias, &[num_experts as i32, hidden as i32]),
        fc2_bias: to_array(&fc2_bias, &[num_experts as i32, hidden as i32]),
    };

    // Two tokens selecting different expert pairs.
    let x = to_array(&[0.0; 6], &[2, hidden as i32]);
    let indices = mlxcel_core::from_slice_i32(&[3, 1, 0, 2], &[2, top_k as i32]);
    let out = read_all(&mlp.forward(&x, &indices));

    // `[n_tokens, top_k, hidden]` flattened: token 0 picks experts 3 and 1,
    // token 1 picks 0 and 2.
    let expected = [3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0];
    assert_eq!(out.len(), expected.len());
    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "slot {i}: {got} vs {want} (full output {out:?})"
        );
    }
}

// Weight-shape validation.

fn lazy(shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(shape, 0.0, mlxcel_core::dtype::FLOAT32)
}

/// A weight map matching the real export's names and shapes, unquantized.
fn synthetic_weights(args: &ModelArgs) -> WeightMap {
    let hidden = args.n_embd as i32;
    let inter = args.intermediate_size() as i32;
    let experts = args.num_local_experts as i32;
    let mut weights = WeightMap::new();

    weights.insert(
        "transformer.embd.wte.weight".into(),
        lazy(&[args.vocab_size as i32, hidden]),
    );
    weights.insert("lm_head.ln.weight".into(), lazy(&[hidden]));
    weights.insert("lm_head.ln.bias".into(), lazy(&[hidden]));
    weights.insert(
        "lm_head.linear.weight".into(),
        lazy(&[args.vocab_size as i32, hidden]),
    );
    weights.insert(
        "lm_head.linear.bias".into(),
        lazy(&[args.vocab_size as i32]),
    );

    for layer in 0..args.n_layer {
        let prefix = format!("transformer.h.{layer}");
        weights.insert(format!("{prefix}.ln.weight"), lazy(&[hidden]));
        weights.insert(format!("{prefix}.ln.bias"), lazy(&[hidden]));
        weights.insert(
            format!("{prefix}.mixer.Wqkv.weight"),
            lazy(&[args.qkv_out_features() as i32, hidden]),
        );
        weights.insert(
            format!("{prefix}.mixer.Wqkv.bias"),
            lazy(&[args.qkv_out_features() as i32]),
        );
        weights.insert(
            format!("{prefix}.mixer.out_proj.weight"),
            lazy(&[hidden, hidden]),
        );
        weights.insert(format!("{prefix}.mixer.out_proj.bias"), lazy(&[hidden]));
        weights.insert(
            format!("{prefix}.moe.gate.weight"),
            lazy(&[experts, hidden]),
        );
        weights.insert(
            format!("{prefix}.moe.switch_mlp.fc1.weight"),
            lazy(&[experts, inter, hidden]),
        );
        weights.insert(
            format!("{prefix}.moe.switch_mlp.fc1.bias"),
            lazy(&[experts, inter]),
        );
        weights.insert(
            format!("{prefix}.moe.switch_mlp.fc2.weight"),
            lazy(&[experts, hidden, inter]),
        );
        weights.insert(
            format!("{prefix}.moe.switch_mlp.fc2.bias"),
            lazy(&[experts, hidden]),
        );
    }
    weights
}

/// A shrunk phixtral: 2 layers, 4 heads of 8, 4 experts.
fn small_args() -> ModelArgs {
    let mut args = phixtral_args();
    args.n_embd = 32;
    args.n_head = 4;
    args.n_layer = 2;
    args.vocab_size = 64;
    args.rotary_dim = 8;
    args.n_inner = Some(64);
    args.quantization = Some(Quantization {
        group_size: 64,
        bits: 4,
    });
    args
}

#[test]
fn a_well_formed_checkpoint_passes_validation() {
    let args = small_args();
    args.validate().expect("the shrunk config is valid");
    let weights = synthetic_weights(&args);
    validate_weights(&weights, &args).expect("the synthetic export validates");
}

#[test]
fn a_missing_expert_bias_is_rejected() {
    // `SwitchMLP` is built with bias=True, so a checkpoint without the bias
    // plane is not a phixtral this loader can reproduce. Dropping it silently
    // would shift every expert's output while leaving the text fluent.
    let args = small_args();
    let mut weights = synthetic_weights(&args);
    weights.remove("transformer.h.0.moe.switch_mlp.fc1.bias");
    let err = validate_weights(&weights, &args).expect_err("the missing bias is rejected");
    assert!(err.contains("fc1.bias"), "{err}");
    assert!(err.contains("carry biases"), "{err}");
}

#[test]
fn a_short_expert_stack_is_rejected() {
    // The router emits indices below num_local_experts and the gather behind
    // gather_mm does not range-check a positive index.
    let args = small_args();
    let mut weights = synthetic_weights(&args);
    weights.insert(
        "transformer.h.0.moe.switch_mlp.fc1.weight".into(),
        lazy(&[
            args.num_local_experts as i32 - 1,
            args.intermediate_size() as i32,
            args.n_embd as i32,
        ]),
    );
    let err = validate_weights(&weights, &args).expect_err("a short stack is rejected");
    assert!(err.contains("num_local_experts"), "{err}");
}

#[test]
fn a_narrow_fused_qkv_is_rejected() {
    // MLX's `slice` clamps an out-of-range stop rather than throwing, so a
    // too-narrow Wqkv yields a short V block and the reshape aborts the process
    // rather than returning an error.
    let args = small_args();
    let mut weights = synthetic_weights(&args);
    weights.insert(
        "transformer.h.0.mixer.Wqkv.weight".into(),
        lazy(&[2 * args.n_embd as i32, args.n_embd as i32]),
    );
    let err = validate_weights(&weights, &args).expect_err("a 2/3-width Wqkv is rejected");
    assert!(err.contains("Wqkv"), "{err}");
    assert!(err.contains(&args.qkv_out_features().to_string()), "{err}");
}

#[test]
fn a_missing_output_head_norm_is_rejected() {
    // The final norm lives under `lm_head.ln`, not at the top of the
    // transformer, which is the layout trap a Llama-shaped loader would miss.
    let args = small_args();
    let mut weights = synthetic_weights(&args);
    weights.remove("lm_head.ln.weight");
    let err = validate_weights(&weights, &args).expect_err("lm_head.ln is required");
    assert!(err.contains("lm_head.ln"), "{err}");
}

// End-to-end construction and a forward pass.

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn filled_weights(args: &ModelArgs) -> WeightMap {
    let mut weights = synthetic_weights(args);
    let keys: Vec<String> = weights.keys().cloned().collect();
    let mut seed = 0x5EED_9876u32;
    for key in keys {
        let shape = mlxcel_core::array_shape(weights.get(&key).expect("key just listed"));
        let n: i32 = shape.iter().product();
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        weights.insert(
            key,
            mlxcel_core::from_slice_f32(&noise(n as usize, seed), &shape),
        );
    }
    weights
}

#[test]
fn a_synthetic_model_builds_and_produces_finite_logits() {
    let args = small_args();
    let weights = filled_weights(&args);
    let model = PhixtralModel::from_weights(&weights, &args).expect("the model builds");
    assert_eq!(model.num_layers(), args.n_layer);

    let tokens = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5], &[1, 5]);
    let mut caches = LanguageModel::make_caches(&model);
    let logits = LanguageModel::forward(&model, &tokens, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 5, args.vocab_size as i32]
    );
    for (i, value) in read_all(&mlxcel_core::slice(&logits, &[0, 4, 0], &[1, 5, 16]))
        .iter()
        .enumerate()
    {
        assert!(value.is_finite(), "logit[{i}] is {value}");
    }

    // A decode step must reuse the KV cache rather than start over.
    let next = mlxcel_core::from_slice_i32(&[6], &[1, 1]);
    let step = LanguageModel::forward(&model, &next, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&step),
        vec![1, 1, args.vocab_size as i32]
    );
}

#[test]
fn the_prefill_is_causal_without_being_handed_a_mask() {
    // Generation calls `forward` with `mask == None` and the model must build
    // its own causal mask. A fully bidirectional prefill still produces fluent
    // text, so it is checked directly: position 0 of a longer prefill must equal
    // a one-token prefill.
    let args = small_args();
    let weights = filled_weights(&args);
    let model = PhixtralModel::from_weights(&weights, &args).expect("the model builds");
    let vocab = args.vocab_size as i32;

    let one = mlxcel_core::from_slice_i32(&[7], &[1, 1]);
    let mut caches = LanguageModel::make_caches(&model);
    let single = LanguageModel::forward(&model, &one, &mut caches, None);
    let single_row = read_all(&mlxcel_core::slice(&single, &[0, 0, 0], &[1, 1, vocab]));

    let many = mlxcel_core::from_slice_i32(&[7, 11, 13, 17], &[1, 4]);
    let mut caches = LanguageModel::make_caches(&model);
    let prefill = LanguageModel::forward(&model, &many, &mut caches, None);
    let first_row = read_all(&mlxcel_core::slice(&prefill, &[0, 0, 0], &[1, 1, vocab]));

    for (i, (a, b)) in single_row.iter().zip(first_row.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "logit {i} at position 0 changed when later tokens were added: {a} vs {b}. \
             The prefill is not causal."
        );
    }
}
