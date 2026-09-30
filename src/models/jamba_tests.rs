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

//! Regression tests for Jamba conv_state contiguous fix.

use mlxcel_core::{dtype, generate::ModelStateSnapshot, layers::KVCache, utils::slice_axis};

/// Simulate 50 decode steps of conv-state update and assert the stored shape
/// stays at [B=1, k-1=3, channels=8] regardless of how many steps we run.
///
/// Before the fix, each step stored a lazy slice aliasing the growing
/// `padded_input` concat graph, leaking memory proportional to step count.
/// After the fix, contiguous() materializes a compact [1, 3, 8] buffer each time.
#[test]
#[ignore = "requires serial MLX execution"]
fn jamba_conv_state_shape_plateaus_after_50_steps() {
    let batch = 1i32;
    let channels = 8i32;
    let k = 4usize; // conv_kernel_size (mamba_d_conv)
    let n_keep = (k - 1) as i32; // = 3
    let expected_shape = vec![batch, n_keep, channels];

    let mut conv_state =
        mlxcel_core::zeros(&[batch, n_keep, channels], mlxcel_core::dtype::FLOAT32);

    for _step in 0..50 {
        let new_token = mlxcel_core::zeros(&[batch, 1, channels], mlxcel_core::dtype::FLOAT32);

        // Build padded_input = concat(conv_state, new_token, axis=1) -> [1, k, channels]
        let padded_input = mlxcel_core::concatenate(&conv_state, &new_token, 1);

        let padded_shape = mlxcel_core::array_shape(&padded_input);
        let len = padded_shape[1] as usize;

        // Apply the fixed conv-state update: slice then contiguous
        let tail = slice_axis(&padded_input, 1, (len - (k - 1)) as i32, len as i32);
        conv_state = mlxcel_core::contiguous(&tail, false);

        mlxcel_core::eval(&conv_state);

        let shape = mlxcel_core::array_shape(&conv_state);
        assert_eq!(
            shape, expected_shape,
            "step {_step}: conv_state shape {shape:?} != expected {expected_shape:?}"
        );
    }
}

/// Verify that JambaMambaCache::new() starts with no conv_state or ssm_state.
#[test]
fn jamba_cache_new_has_no_state() {
    let cache = super::JambaMambaCache::new();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
}

/// Verify that JambaMambaCache::default() starts with no state.
#[test]
fn jamba_cache_default_has_no_state() {
    let cache = super::JambaMambaCache::default();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
}

#[test]
fn jamba_mamba_cache_snapshot_restore_round_trips_state_shapes() {
    let mut cache = super::JambaMambaCache::new();
    cache.conv_state = Some(mlxcel_core::zeros(&[1, 3, 8], dtype::FLOAT32));
    cache.ssm_state = Some(mlxcel_core::zeros(&[1, 2, 4, 8], dtype::FLOAT32));

    let mut snapshot = ModelStateSnapshot::new("jamba", 11);
    cache.snapshot_into(&mut snapshot, "layer1.mamba");

    let mut restored = super::JambaMambaCache::new();
    restored.restore_from(&snapshot, "layer1.mamba");

    let conv = restored
        .conv_state
        .as_ref()
        .and_then(|a| a.as_ref())
        .expect("conv_state restored");
    let ssm = restored
        .ssm_state
        .as_ref()
        .and_then(|a| a.as_ref())
        .expect("ssm_state restored");
    assert_eq!(mlxcel_core::array_shape(conv), vec![1, 3, 8]);
    assert_eq!(mlxcel_core::array_shape(ssm), vec![1, 2, 4, 8]);
}

#[test]
fn jamba_attention_layer_snapshot_restore_round_trips_kv_state() {
    let mut kv = KVCache::new();
    kv.keys = Some(mlxcel_core::zeros(&[1, 2, 11, 4], dtype::FLOAT32));
    kv.values = Some(mlxcel_core::zeros(&[1, 2, 11, 4], dtype::FLOAT32));
    kv.offset = 11;
    let cache = super::JambaLayerCache::Attention(kv);

    let mut snapshot = ModelStateSnapshot::new("jamba", 11);
    cache.snapshot_into(&mut snapshot, "layer0");

    let mut restored = super::JambaLayerCache::Attention(KVCache::new());
    restored.restore_from(&snapshot, "layer0");

    match restored {
        super::JambaLayerCache::Attention(kv) => {
            let keys = kv
                .keys
                .as_ref()
                .and_then(|a| a.as_ref())
                .expect("keys restored");
            let values = kv
                .values
                .as_ref()
                .and_then(|a| a.as_ref())
                .expect("values restored");
            assert_eq!(kv.offset, 11);
            assert_eq!(mlxcel_core::array_shape(keys), vec![1, 2, 11, 4]);
            assert_eq!(mlxcel_core::array_shape(values), vec![1, 2, 11, 4]);
        }
        super::JambaLayerCache::Mamba(_) => panic!("restored wrong cache variant"),
    }
}

// Quantized MoE expert plane (issue #974).
//
// Jamba declared a family-local `SwitchLinear::Quantized` variant that nothing
// ever constructed, so a pre-stacked quantized `switch_mlp` plane was built as a
// dense plane from packed `uint32` weights and the `.scales` / `.biases` planes
// were dropped on the floor. `ncls-p/AI21-Jamba2-Mini-mlx-4Bit` ships exactly
// that layout: `num_experts: 16` over 32 layers, 48 `.weight` / 48 `.scales` /
// 48 `.biases` pre-stacked `switch_mlp` tensors, and 8-bit routers declared
// under a 4-bit top level.
//
// These drive `JambaModel::from_weights` itself rather than a loader helper in
// isolation, and assert on the returned `Result` without running a forward
// pass: `gather_qmm` and `quantized_matmul` cross the cxx bridge as
// `UniquePtr<MlxArray>` rather than `Result`, so an MLX C++ throw is an
// uncatchable `std::terminate` that would take the whole test binary down with
// SIGABRT instead of failing cleanly here.
//
// The hostile-parameter walk is also what proves the quantized branch is live at
// all: the dense `gather_mm` path ignores the declared `group_size` / `bits`
// pair entirely, so before the fix every hostile pair loaded without complaint.

use super::{JambaConfig, JambaModel};
use crate::models::switch_layers::{HOSTILE_QUANT_PARAMS, insert_stacked_quantized_expert_plane};
use mlxcel_core::weights::WeightMap;

/// Honest 4-bit geometry: `packed_in * 32 == bits * num_groups * group_size`
/// on both projection shapes, so the positive controls below are planes MLX can
/// actually describe.
const HIDDEN: i32 = 64;
const INTERMEDIATE: i32 = 128;
const VOCAB: i32 = 32;
const EXPERTS: i32 = 2;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

const MOE_PREFIX: &str = "model.layers.0.feed_forward.switch_mlp";
const GATE_PLANE: &str = "model.layers.0.feed_forward.switch_mlp.gate_proj";

/// One MoE layer: `expert_layer_period` 1 with `num_experts` 2 makes layer 0 an
/// expert layer, and `attn_layer_period` 1 makes it an attention layer, so the
/// fixture does not also have to carry a Mamba block.
fn moe_config(quantization: &str) -> JambaConfig {
    serde_json::from_str(&format!(
        r#"{{
            "model_type": "jamba",
            "hidden_size": {HIDDEN},
            "intermediate_size": {INTERMEDIATE},
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": {VOCAB},
            "attn_layer_period": 1,
            "attn_layer_offset": 0,
            "expert_layer_period": 1,
            "expert_layer_offset": 0,
            "num_experts": {EXPERTS},
            "num_experts_per_tok": 2,
            "tie_word_embeddings": true,
            "quantization": {quantization}
        }}"#
    ))
    .expect("jamba MoE test config")
}

fn quant_block(group_size: i32, bits: i32) -> String {
    format!(r#"{{"group_size": {group_size}, "bits": {bits}}}"#)
}

fn put_f32(weights: &mut WeightMap, name: &str, shape: &[i32], value: f32) {
    let n: usize = shape.iter().map(|d| *d as usize).product();
    weights.insert(
        name.to_string(),
        mlxcel_core::from_slice_f32(&vec![value; n], shape),
    );
}

/// Everything a one-layer attention-plus-MoE Jamba needs except the three
/// expert planes, which each test supplies in the layout it is exercising.
fn backbone_weights() -> WeightMap {
    let mut weights = WeightMap::new();
    put_f32(
        &mut weights,
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        0.02,
    );
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        put_f32(
            &mut weights,
            &format!("model.layers.0.self_attn.{proj}.weight"),
            &[HIDDEN, HIDDEN],
            0.02,
        );
    }
    put_f32(
        &mut weights,
        "model.layers.0.feed_forward.router.weight",
        &[EXPERTS, HIDDEN],
        0.02,
    );
    put_f32(
        &mut weights,
        "model.layers.0.input_layernorm.weight",
        &[HIDDEN],
        1.0,
    );
    put_f32(
        &mut weights,
        "model.layers.0.pre_ff_layernorm.weight",
        &[HIDDEN],
        1.0,
    );
    put_f32(&mut weights, "model.final_layernorm.weight", &[HIDDEN], 1.0);
    weights
}

/// Pre-stacked affine expert planes, the layout every mlx-community style
/// conversion (and `ncls-p/AI21-Jamba2-Mini-mlx-4Bit`) ships.
fn stacked_quantized_weights() -> WeightMap {
    let mut weights = backbone_weights();
    for (proj, out, in_features) in [
        ("gate_proj", INTERMEDIATE, HIDDEN),
        ("up_proj", INTERMEDIATE, HIDDEN),
        ("down_proj", HIDDEN, INTERMEDIATE),
    ] {
        insert_stacked_quantized_expert_plane(
            &mut weights,
            &format!("{MOE_PREFIX}.{proj}"),
            EXPERTS,
            out,
            in_features * BITS / 32,
            in_features / GROUP_SIZE,
        );
    }
    weights
}

/// The same three planes shipped unstacked, one tensor per expert, which is the
/// layout `sanitize_weights` used to stack by hand while carrying `.weight`
/// alone and dropping `.scales` / `.biases`.
fn per_expert_quantized_weights() -> WeightMap {
    let mut weights = backbone_weights();
    for (proj, out, in_features) in [
        ("gate_proj", INTERMEDIATE, HIDDEN),
        ("up_proj", INTERMEDIATE, HIDDEN),
        ("down_proj", HIDDEN, INTERMEDIATE),
    ] {
        let packed_in = in_features * BITS / 32;
        let num_groups = in_features / GROUP_SIZE;
        for e in 0..EXPERTS {
            let prefix = format!("model.layers.0.feed_forward.experts.{e}.{proj}");
            let n = (out * packed_in) as usize;
            let packed = mlxcel_core::from_slice_f32(&vec![0.0f32; n], &[out, packed_in]);
            weights.insert(
                format!("{prefix}.weight"),
                mlxcel_core::astype(&packed, mlxcel_core::dtype::UINT32),
            );
            put_f32(
                &mut weights,
                &format!("{prefix}.scales"),
                &[out, num_groups],
                1.0,
            );
            put_f32(
                &mut weights,
                &format!("{prefix}.biases"),
                &[out, num_groups],
                0.0,
            );
        }
    }
    weights
}

/// Dense bf16-style experts: one stacked `[experts, out, in]` plane per
/// projection and no `.scales`, so the declared pair is inert.
fn dense_expert_weights() -> WeightMap {
    let mut weights = backbone_weights();
    for (proj, out, in_features) in [
        ("gate_proj", INTERMEDIATE, HIDDEN),
        ("up_proj", INTERMEDIATE, HIDDEN),
        ("down_proj", HIDDEN, INTERMEDIATE),
    ] {
        put_f32(
            &mut weights,
            &format!("{MOE_PREFIX}.{proj}.weight"),
            &[EXPERTS, out, in_features],
            0.02,
        );
    }
    weights
}

/// A quantized expert plane must reach the `gather_qmm` path with its declared
/// `group_size` / `bits` bounded first.
///
/// This is also the discriminator for the bug itself. The dense `gather_mm`
/// path never reads the declared pair, so while Jamba built every expert plane
/// as `Regular` a `"bits": 0` loaded silently and then aborted the process
/// inside the kernel at the first routed token. A load that now refuses the
/// hostile pair is a load that took the quantized branch.
#[test]
fn jamba_rejects_expert_quantization_params_that_would_abort_gather_qmm() {
    // Positive control first, so a guard that rejected every quantized plane
    // could not pass this test.
    match JambaModel::from_weights(
        moe_config(&quant_block(GROUP_SIZE, BITS)),
        stacked_quantized_weights(),
    ) {
        Ok(_) => {}
        Err(e) => panic!("honest 4-bit pre-stacked Jamba expert planes must load: {e}"),
    }

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let err = match JambaModel::from_weights(
            moe_config(&quant_block(group_size, bits)),
            stacked_quantized_weights(),
        ) {
            Ok(_) => panic!(
                "(group_size {group_size}, bits {bits}) must be refused at load, \
                 not stored for gather_qmm"
            ),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains(field),
            "(group_size {group_size}, bits {bits}) must be blamed on {field}, got: {err}"
        );
        assert!(
            err.contains(GATE_PLANE),
            "the load error must name the offending tensor {GATE_PLANE}, got: {err}"
        );
    }
}

/// Unstacked per-expert quantized planes must be stacked with their `.scales`
/// and `.biases`, not with `.weight` alone.
///
/// The hostile pair is the probe: it can only be rejected if the stacked plane
/// carried scales into the quantized branch. When the per-expert remap copied
/// `.weight` alone, the scales stayed behind in the weight map and the layer was
/// built dense from packed `uint32`.
#[test]
fn jamba_stacks_per_expert_scales_and_biases_rather_than_dropping_them() {
    match JambaModel::from_weights(
        moe_config(&quant_block(GROUP_SIZE, BITS)),
        per_expert_quantized_weights(),
    ) {
        Ok(_) => {}
        Err(e) => panic!("honest 4-bit per-expert Jamba expert planes must load: {e}"),
    }

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let err = match JambaModel::from_weights(
            moe_config(&quant_block(group_size, bits)),
            per_expert_quantized_weights(),
        ) {
            Ok(_) => panic!(
                "(group_size {group_size}, bits {bits}) must be refused for the per-expert \
                 layout too; accepting it means the .scales plane was dropped"
            ),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains(field),
            "(group_size {group_size}, bits {bits}) must be blamed on {field}, got: {err}"
        );
    }
}

/// A dense expert plane carries no packing and no `.scales`, so the declared
/// pair is inert on the `gather_mm` path and must not gate the load.
#[test]
fn jamba_dense_expert_planes_load_with_an_unset_quantization_pair() {
    match JambaModel::from_weights(moe_config(&quant_block(0, 0)), dense_expert_weights()) {
        Ok(_) => {}
        Err(e) => panic!("a non-quantized Jamba expert plane must load with an unset pair: {e}"),
    }
}

/// Every Jamba loader builds affine planes, so a declared block-float or
/// unparseable mode has to fail at load rather than be silently reinterpreted as
/// affine and abort inside the kernel.
#[test]
fn jamba_rejects_a_declared_quantization_mode_its_loaders_cannot_honor() {
    for mode in ["mxfp4", "nvfp4", "mxfp8", "gptq", "", "Affine"] {
        let quantization =
            format!(r#"{{"group_size": {GROUP_SIZE}, "bits": {BITS}, "mode": "{mode}"}}"#);
        match JambaModel::from_weights(moe_config(&quantization), stacked_quantized_weights()) {
            Ok(_) => panic!("declared mode {mode:?} must be refused at load"),
            Err(e) => {
                let err = e.to_string();
                assert!(
                    err.contains("mode"),
                    "the load error for {mode:?} must name the mode, got: {err}"
                );
            }
        }
    }

    // An explicitly declared "affine" is the norm and must still load.
    let affine = format!(r#"{{"group_size": {GROUP_SIZE}, "bits": {BITS}, "mode": "affine"}}"#);
    match JambaModel::from_weights(moe_config(&affine), stacked_quantized_weights()) {
        Ok(_) => {}
        Err(e) => panic!("an explicitly declared affine mode must load: {e}"),
    }
}

/// Seeded small-magnitude values, so positions differ and a non-causal
/// attention would change the prefill rows.
fn varied(shape: &[i32], seed: u64) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let len: usize = shape.iter().map(|d| *d as usize).product();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let data: Vec<f32> = (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / ((1u32 << 24) as f32) * 0.4 - 0.2
        })
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn to_f32(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    let arr = mlxcel_core::astype(arr, dtype::FLOAT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Issue #1991: Jamba prefill passes no attention mask and the attention
/// layer takes MLX's maskless causal SDPA. Each prefill row must match the
/// same position decoded one token at a time, fresh and continued from an
/// offset. Varied embeddings and projections make a non-causal pass differ.
#[test]
fn jamba_prefill_rows_match_token_by_token_decode() {
    use mlxcel_core::cache::SequenceId;
    use mlxcel_core::generate::LanguageModel;

    let build = || {
        let mut weights = dense_expert_weights();
        weights.insert(
            "model.embed_tokens.weight".into(),
            varied(&[VOCAB, HIDDEN], 1),
        );
        for (i, proj) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
            weights.insert(
                format!("model.layers.0.self_attn.{proj}.weight"),
                varied(&[HIDDEN, HIDDEN], 10 + i as u64),
            );
        }
        JambaModel::from_weights(moe_config(&quant_block(0, 0)), weights)
            .expect("dense Jamba fixture must load")
    };
    let vocab = VOCAB as usize;
    for (case, (warm, chunk)) in [(0usize, 6usize), (3, 5)].into_iter().enumerate() {
        let tokens: Vec<i32> = (0..(warm + chunk) as i32)
            .map(|i| (i * 5 + 1) % VOCAB)
            .collect();

        let reference = build();
        let seq_ref = SequenceId::from_raw(1_991_000 + case as u64);
        reference.prepare_sequence_state(seq_ref);
        let want: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&t| {
                let x = mlxcel_core::from_slice_i32(&[t], &[1, 1]);
                to_f32(&reference.forward_with_sequence_id(&x, Some(seq_ref), &mut [], None))
            })
            .collect();

        let model = build();
        let seq = SequenceId::from_raw(1_991_100 + case as u64);
        model.prepare_sequence_state(seq);
        if warm > 0 {
            let x = mlxcel_core::from_slice_i32(&tokens[..warm], &[1, warm as i32]);
            let _ = to_f32(&model.forward_with_sequence_id(&x, Some(seq), &mut [], None));
        }
        let x = mlxcel_core::from_slice_i32(&tokens[warm..], &[1, chunk as i32]);
        let got = to_f32(&model.forward_with_sequence_id(&x, Some(seq), &mut [], None));
        assert_eq!(got.len(), chunk * vocab, "case {case}: logits shape");
        for row in 0..chunk {
            let max = got[row * vocab..(row + 1) * vocab]
                .iter()
                .zip(&want[warm + row])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max < 1e-4,
                "warm {warm} chunk {chunk} row {row}: max |diff| {max}"
            );
        }
    }
}

/// One Mamba layer (`attn_layer_period` 2, offset 1 puts attention at layer 1,
/// which a one-layer model never reaches) with a dense MoE feed-forward.
fn mamba_config() -> JambaConfig {
    serde_json::from_str(&format!(
        r#"{{
            "model_type": "jamba",
            "hidden_size": {HIDDEN},
            "intermediate_size": {INTERMEDIATE},
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": {VOCAB},
            "attn_layer_period": 2,
            "attn_layer_offset": 1,
            "expert_layer_period": 1,
            "expert_layer_offset": 0,
            "num_experts": {EXPERTS},
            "num_experts_per_tok": 2,
            "mamba_d_state": 8,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
            "mamba_dt_rank": 8,
            "mamba_conv_bias": true,
            "tie_word_embeddings": true,
            "quantization": {{"group_size": 0, "bits": 0}}
        }}"#
    ))
    .expect("jamba Mamba test config")
}

fn mamba_weights() -> WeightMap {
    let (h, i, n, r, k) = (HIDDEN, 2 * HIDDEN, 8, 8, 4);
    let mut weights = dense_expert_weights();
    weights.insert("model.embed_tokens.weight".into(), varied(&[VOCAB, h], 1));
    let m = "model.layers.0.mamba";
    for (name, shape, seed) in [
        ("in_proj.weight", vec![2 * i, h], 21u64),
        ("x_proj.weight", vec![r + 2 * n, i], 22),
        ("dt_proj.weight", vec![i, r], 23),
        ("dt_proj.bias", vec![i], 24),
        ("conv1d.weight", vec![i, k, 1], 25),
        ("conv1d.bias", vec![i], 26),
        ("A_log", vec![i, n], 27),
        ("D", vec![i], 28),
        ("out_proj.weight", vec![h, i], 29),
    ] {
        weights.insert(format!("{m}.{name}"), varied(&shape, seed));
    }
    for (name, dim) in [("dt_layernorm", r), ("b_layernorm", n), ("c_layernorm", n)] {
        put_f32(&mut weights, &format!("{m}.{name}.weight"), &[dim], 1.0);
    }
    weights
}

/// Issue #1999: the multi-token Mamba scan (the prefill branch of
/// `JambaMambaMixer::ssm_step`) must give, row for row, what the single-token
/// branch gives when the same tokens are decoded one at a time, both from an
/// empty state and continuing from an existing conv and SSM state.
#[test]
fn jamba_mamba_prefill_rows_match_token_by_token_decode() {
    use mlxcel_core::cache::SequenceId;
    use mlxcel_core::generate::LanguageModel;

    let build = || {
        JambaModel::from_weights(mamba_config(), mamba_weights())
            .expect("Mamba Jamba fixture must load")
    };
    let vocab = VOCAB as usize;
    for (case, (warm, chunk)) in [(0usize, 7usize), (3, 6)].into_iter().enumerate() {
        let tokens: Vec<i32> = (0..(warm + chunk) as i32)
            .map(|i| (i * 7 + 3) % VOCAB)
            .collect();

        let reference = build();
        let seq_ref = SequenceId::from_raw(1_999_000 + case as u64);
        reference.prepare_sequence_state(seq_ref);
        let want: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&t| {
                let x = mlxcel_core::from_slice_i32(&[t], &[1, 1]);
                to_f32(&reference.forward_with_sequence_id(&x, Some(seq_ref), &mut [], None))
            })
            .collect();

        let model = build();
        let seq = SequenceId::from_raw(1_999_100 + case as u64);
        model.prepare_sequence_state(seq);
        if warm > 0 {
            let x = mlxcel_core::from_slice_i32(&tokens[..warm], &[1, warm as i32]);
            let _ = to_f32(&model.forward_with_sequence_id(&x, Some(seq), &mut [], None));
        }
        let x = mlxcel_core::from_slice_i32(&tokens[warm..], &[1, chunk as i32]);
        let got = to_f32(&model.forward_with_sequence_id(&x, Some(seq), &mut [], None));
        assert_eq!(got.len(), chunk * vocab, "case {case}: logits shape");
        for row in 0..chunk {
            let max = got[row * vocab..(row + 1) * vocab]
                .iter()
                .zip(&want[warm + row])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max < 1e-4,
                "warm {warm} chunk {chunk} row {row}: max |diff| {max}"
            );
        }
    }
}

/// Issue #1981: on a backend with a fused Mamba1 scan port (Metal, CUDA), a
/// Jamba prefill must not walk the per-step graph scan. That loop issues
/// several small ops per timestep per Mamba layer, and on CUDA it was about
/// two thirds of a 3.3k-token prefill. Checked for f32 and bf16 weights (the
/// published checkpoints are bf16), for a prefill and the decode step after it.
#[test]
fn jamba_mamba_prefill_takes_the_fused_scan_where_a_port_exists() {
    use mlxcel_core::cache::SequenceId;
    use mlxcel_core::generate::LanguageModel;
    use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};

    // Not skipped under `MLXCEL_MAMBA1_SCAN_KERNEL=0`: that setting brings the
    // per-step loop back, which is exactly what this test must catch.
    if !matches!(
        gpu_backend_kind(),
        GpuBackendKind::Metal | GpuBackendKind::Cuda
    ) {
        return;
    }
    for weight_dtype in [dtype::FLOAT32, dtype::BFLOAT16] {
        let weights: WeightMap = mamba_weights()
            .into_iter()
            .map(|(k, v)| (k, mlxcel_core::astype(&v, weight_dtype)))
            .collect();
        let model = JambaModel::from_weights(mamba_config(), weights)
            .expect("Mamba Jamba fixture must load");
        let seq = SequenceId::from_raw(1_981_000 + weight_dtype as u64);
        model.prepare_sequence_state(seq);

        let before = super::GRAPH_SCAN_STEPS.with(|n| n.get());
        let prompt: Vec<i32> = (0..48).map(|i| (i * 7 + 3) % VOCAB).collect();
        let x = mlxcel_core::from_slice_i32(&prompt, &[1, prompt.len() as i32]);
        let _ = to_f32(&model.forward_with_sequence_id(&x, Some(seq), &mut [], None));
        let x = mlxcel_core::from_slice_i32(&[5], &[1, 1]);
        let _ = to_f32(&model.forward_with_sequence_id(&x, Some(seq), &mut [], None));
        let walked = super::GRAPH_SCAN_STEPS.with(|n| n.get()) - before;
        assert_eq!(
            walked, 0,
            "dtype {weight_dtype}: prefill and decode walked {walked} graph-scan timesteps instead of the fused kernel"
        );
    }
}
