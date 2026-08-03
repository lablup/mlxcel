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

//! Unit tests for the DeepSeek v1 per-expert `SwitchLinear` loader, covering
//! the `baidu/Unlimited-OCR` raw-checkpoint path (`experts.{idx}.{proj}`
//! stacked into `switch_mlp` via the shared
//! `switch_layers::stack_individual_experts` helper).

use super::{ModelArgs, SwitchLinear};
use crate::models::config::QuantizationArgs;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

fn test_args(n_routed_experts: Option<usize>) -> ModelArgs {
    ModelArgs {
        model_type: "deepseek".to_string(),
        vocab_size: 8,
        hidden_size: 8,
        intermediate_size: 8,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        max_position_embeddings: 16,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        moe_intermediate_size: None,
        n_shared_experts: None,
        n_routed_experts,
        num_experts_per_tok: None,
        moe_layer_freq: 1,
        first_k_dense_replace: 0,
        routed_scaling_factor: 1.0,
        attention_bias: false,
        quantization: None,
        group_size: None,
        bits: None,
    }
}

fn insert_expert(weights: &mut WeightMap, root: &str, idx: usize, out: i32, in_dim: i32) {
    weights.insert(
        format!("{root}.experts.{idx}.gate_proj.weight"),
        mlxcel_core::from_slice_f32(&vec![0.0; (out * in_dim) as usize], &[out, in_dim]),
    );
}

/// `baidu/Unlimited-OCR`-style truncated checkpoint: the config declares 4
/// routed experts but the shard only carries experts 0..2 contiguously (a gap
/// at index 3, e.g. a dropped middle/trailing shard). Before the cross-check,
/// `stack_individual_experts` silently stacked only the 3 experts it found,
/// which would let the router's top-k gather index expert 3 out of bounds at
/// inference instead of failing at load time.
#[test]
fn switch_linear_errors_when_stacked_experts_fall_short_of_config_count() {
    // (`SwitchLinear` holds non-Debug MlxArray handles, so match on the Result
    // rather than using `expect_err`.)
    let root = "model.layers.0.mlp";
    let mut weights = WeightMap::new();
    for e in 0..3 {
        insert_expert(&mut weights, root, e, 4, 4);
    }
    let args = test_args(Some(4));

    let err = match SwitchLinear::from_weights(
        &weights,
        &args,
        &format!("{root}.switch_mlp.gate_proj"),
    ) {
        Ok(_) => panic!("stacking fewer experts than n_routed_experts declares must error"),
        Err(e) => e,
    };
    assert!(
        err.contains('4') && err.contains('3'),
        "error should name both the declared (4) and found (3) counts: {err}"
    );
}

/// The same declared count with every expert present must still load
/// (the cross-check only rejects a shortfall, never an exact match).
#[test]
fn switch_linear_accepts_full_expert_count() {
    let root = "model.layers.0.mlp";
    let mut weights = WeightMap::new();
    for e in 0..4 {
        insert_expert(&mut weights, root, e, 4, 4);
    }
    let args = test_args(Some(4));

    let sl = SwitchLinear::from_weights(&weights, &args, &format!("{root}.switch_mlp.gate_proj"))
        .expect("stacking exactly n_routed_experts experts must succeed");
    assert_eq!(sl.num_experts(), 4);
}

/// The same minimal config with an explicit quantization pair in the flat
/// top-level spelling, which the accessors keep as a fallback.
fn quant_args(group_size: i32, bits: i32) -> ModelArgs {
    let mut args = test_args(None);
    args.group_size = Some(group_size);
    args.bits = Some(bits);
    args
}

/// The same pair in the nested `quantization` spelling that every conversion
/// tool actually emits (#975).
fn nested_quant_args(group_size: i32, bits: i32) -> ModelArgs {
    let mut args = test_args(None);
    args.quantization = Some(QuantizationArgs {
        group_size: Some(group_size),
        bits: Some(bits),
        mode: None,
    });
    args
}

/// `SwitchLinear::from_stacked_parts` stores the declared pair verbatim on the
/// quantized variant and hands it straight to `gather_qmm`, which crosses the
/// cxx bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++ throw there
/// is an uncatchable `std::terminate`,
/// so losing the bound turns a rejected load into an uncatchable abort at the
/// first routed forward pass in production. This test asserts on the load
/// result rather than running a forward pass, so a regression fails cleanly
/// here instead of aborting the test binary.
#[test]
fn deepseek_switch_linear_rejects_quantization_params_that_would_abort_gather_qmm() {
    // Honest 4-bit expert geometry: `packed_in * 32 == bits * num_groups *
    // group_size` (8 * 32 == 4 * 1 * 64), so the positive control below is a
    // plane MLX can actually describe.
    const EXPERTS: i32 = 3;
    const OUT: i32 = 4;
    const PACKED_IN: i32 = 8;
    const NUM_GROUPS: i32 = 1;
    const GROUP_SIZE: i32 = 64;
    const BITS: i32 = 4;
    const PREFIX: &str = "model.layers.0.mlp.switch_mlp.gate_proj";

    let mut weights = WeightMap::new();
    crate::models::switch_layers::insert_stacked_quantized_expert_plane(
        &mut weights,
        PREFIX,
        EXPERTS,
        OUT,
        PACKED_IN,
        NUM_GROUPS,
    );

    // Positive control first, so a guard that rejected every quantized plane
    // could not pass this test.
    SwitchLinear::from_weights(&weights, &quant_args(GROUP_SIZE, BITS), PREFIX)
        .expect("honest 4-bit expert plane must load");

    for (group_size, bits, field) in crate::models::switch_layers::HOSTILE_QUANT_PARAMS {
        let err = match SwitchLinear::from_weights(&weights, &quant_args(group_size, bits), PREFIX)
        {
            Ok(_) => panic!(
                "(group_size {group_size}, bits {bits}) must be refused at load, \
                 not stored for gather_qmm"
            ),
            Err(e) => e,
        };
        assert!(
            err.contains(field),
            "(group_size {group_size}, bits {bits}) must be blamed on {field}, got: {err}"
        );
    }

    // A bf16 expert plane carries no packing at all, so the declared pair is
    // irrelevant there and must not gate the non-quantized fallback.
    let mut regular = WeightMap::new();
    regular.insert(
        format!("{PREFIX}.weight"),
        mlxcel_core::ones(&[EXPERTS, OUT, PACKED_IN], mlxcel_core::dtype::BFLOAT16),
    );
    SwitchLinear::from_weights(&regular, &quant_args(0, 0), PREFIX)
        .expect("a bf16 expert plane must not be gated on quantization params");
}

/// The MoE half of issue #975. `SwitchLinear::from_stacked_parts` stores the
/// pair `args.group_size()` / `args.bits()` resolves to and hands it straight
/// to `gather_qmm` with no reconciliation, so a nested block that never reached
/// the accessors meant every DeepSeek-OCR expert plane was built with 64 / 4
/// regardless of the checkpoint. This drives the expert loader through the
/// nested spelling: the honest pair loads, and a pair MLX cannot describe is
/// refused at load rather than stored for an uncatchable abort.
#[test]
fn deepseek_switch_linear_reads_the_nested_quantization_block() {
    const EXPERTS: i32 = 3;
    const OUT: i32 = 4;
    const PACKED_IN: i32 = 8;
    const NUM_GROUPS: i32 = 1;
    const GROUP_SIZE: i32 = 64;
    const BITS: i32 = 4;
    const PREFIX: &str = "model.layers.0.mlp.switch_mlp.gate_proj";

    let mut weights = WeightMap::new();
    crate::models::switch_layers::insert_stacked_quantized_expert_plane(
        &mut weights,
        PREFIX,
        EXPERTS,
        OUT,
        PACKED_IN,
        NUM_GROUPS,
    );

    SwitchLinear::from_weights(&weights, &nested_quant_args(GROUP_SIZE, BITS), PREFIX)
        .expect("an honest nested-declared 4-bit expert plane must load");

    for (group_size, bits, field) in crate::models::switch_layers::HOSTILE_QUANT_PARAMS {
        let err = match SwitchLinear::from_weights(
            &weights,
            &nested_quant_args(group_size, bits),
            PREFIX,
        ) {
            Ok(_) => panic!(
                "nested (group_size {group_size}, bits {bits}) must reach the guard, \
                     not be dropped in favour of the 64 / 4 fallback"
            ),
            Err(e) => e,
        };
        assert!(
            err.contains(field),
            "nested (group_size {group_size}, bits {bits}) must be blamed on {field}, got: {err}"
        );
    }
}

/// A plain `deepseek` `config.json`, which `DeepSeekModel::load` deserializes
/// whole. The nested block is the only spelling any MLX conversion tool emits,
/// and before #975 this struct had no field for it.
#[test]
fn model_args_reads_a_nested_quantization_block() {
    let args: ModelArgs = serde_json::from_value(json!({
        "model_type": "deepseek",
        "vocab_size": 8,
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": 16,
        "quantization": {"group_size": 32, "bits": 8, "mode": "affine"},
    }))
    .expect("a plain deepseek config.json must parse");

    assert_eq!(args.group_size(), 32);
    assert_eq!(args.bits(), 8);
    assert_eq!(args.quantization_mode().unwrap(), "affine");
}

/// The flat spelling these two fields were originally added for keeps loading
/// with its declared values, so the new nested field is additive.
#[test]
fn model_args_still_reads_the_flat_quantization_keys() {
    let args: ModelArgs = serde_json::from_value(json!({
        "model_type": "deepseek",
        "vocab_size": 8,
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": 16,
        "group_size": 128,
        "bits": 6,
    }))
    .expect("the flat spelling must keep parsing");

    assert_eq!(args.group_size(), 128);
    assert_eq!(args.bits(), 6);
}

/// Documented precedence when a config carries both spellings.
#[test]
fn model_args_prefers_the_nested_block_over_the_flat_keys() {
    let mut args = nested_quant_args(32, 8);
    args.group_size = Some(128);
    args.bits = Some(4);

    assert_eq!(args.group_size(), 32);
    assert_eq!(args.bits(), 8);
}

/// Each key falls back independently, so a block naming only one of the two
/// resolves rather than failing to deserialize. `QuantizationArgs` keeps all
/// three fields optional for exactly this reason.
#[test]
fn model_args_falls_back_per_key_across_the_two_spellings() {
    let mut args = test_args(None);
    args.quantization = Some(QuantizationArgs {
        group_size: None,
        bits: Some(8),
        mode: None,
    });
    args.group_size = Some(32);

    assert_eq!(args.group_size(), 32, "the flat key fills the nested gap");
    assert_eq!(args.bits(), 8, "the nested key wins where it is declared");
}

/// No declaration at all keeps the family defaults, which is what the two
/// in-tree DeepSeek-OCR checkpoints and every bf16 checkpoint rely on.
#[test]
fn model_args_defaults_without_any_quantization_declaration() {
    let args = test_args(None);

    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
    assert_eq!(args.quantization_mode().unwrap(), "affine");
}

/// The declared mode is validated where it is read (#973). An absent block and
/// an absent `mode` key both resolve to `"affine"`; anything MLX's own parser
/// would reject is refused here instead of reaching a kernel.
#[test]
fn model_args_bounds_the_declared_quantization_mode() {
    for mode in mlxcel_core::layers::SUPPORTED_QUANTIZATION_MODES {
        let mut args = test_args(None);
        args.quantization = Some(QuantizationArgs {
            group_size: Some(64),
            bits: Some(4),
            mode: Some(mode.to_string()),
        });
        assert_eq!(
            args.quantization_mode().expect("a supported mode"),
            mode,
            "{mode} is one of MLX's own four modes"
        );
    }

    for mode in ["Affine", " affine", "", "int4"] {
        let mut args = test_args(None);
        args.quantization = Some(QuantizationArgs {
            group_size: Some(64),
            bits: Some(4),
            mode: Some(mode.to_string()),
        });
        let err = args
            .quantization_mode()
            .expect_err("a mode MLX cannot parse must not resolve");
        assert!(
            err.contains("quantization.mode"),
            "{mode:?} must be blamed on quantization.mode, got: {err}"
        );
    }
}

/// A pair that is individually in range can still describe a different tensor
/// than the one on disk. `validate_expert_quantization_params` cannot see that,
/// because it deliberately stays a range so mixed-precision exports keep
/// loading, and this family never reaches `reconcile_quantization_layout`. Now
/// that #975 lets a declared block actually arrive here, the mismatch has to be
/// a load error rather than an uncatchable `gather_qmm` abort at the first
/// routed forward pass.
#[test]
fn deepseek_switch_linear_rejects_a_pair_that_does_not_describe_the_expert_plane() {
    // packed_in * 32 == 256, so (64, 4) is the honest pair for this plane.
    const EXPERTS: i32 = 3;
    const OUT: i32 = 4;
    const PACKED_IN: i32 = 8;
    const NUM_GROUPS: i32 = 1;
    const PREFIX: &str = "model.layers.0.mlp.switch_mlp.gate_proj";

    let mut weights = WeightMap::new();
    crate::models::switch_layers::insert_stacked_quantized_expert_plane(
        &mut weights,
        PREFIX,
        EXPERTS,
        OUT,
        PACKED_IN,
        NUM_GROUPS,
    );

    // Positive control: the honest pair still loads through both spellings.
    SwitchLinear::from_weights(&weights, &quant_args(64, 4), PREFIX)
        .expect("the honest pair must load");
    SwitchLinear::from_weights(&weights, &nested_quant_args(64, 4), PREFIX)
        .expect("the honest pair must load through the nested spelling too");

    // Both of these pass the range bound (`1..=32` bits, `1..=MAX` group size)
    // and both describe a plane this checkpoint does not ship.
    for (group_size, bits) in [(64, 8), (32, 4), (128, 4)] {
        let err = match SwitchLinear::from_weights(
            &weights,
            &nested_quant_args(group_size, bits),
            PREFIX,
        ) {
            Ok(_) => panic!(
                "(group_size {group_size}, bits {bits}) describes a different plane and must be \
                 refused at load, not stored for gather_qmm"
            ),
            Err(e) => e,
        };
        assert!(
            err.contains("does not describe this expert plane"),
            "(group_size {group_size}, bits {bits}) must be blamed on the shape mismatch, \
             got: {err}"
        );
        assert!(
            err.contains(PREFIX),
            "the error must name the offending plane, got: {err}"
        );
    }
}
