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

//! Checkpoint-free tests for the Kimi K3 text backbone.
//!
//! Every mechanism the real-checkpoint gate cannot see is pinned against a
//! scalar reference here: the SiTU formula, the lower-bounded gate range, the
//! fused q/k/v projection and conv against the separate ones, the decode
//! conv step against the conv1d prefill, the absorbed q-LoRA MLA with its
//! output gate against the unabsorbed path, the latent MoE against a
//! per-expert loop, the Attention Residuals mix against a scalar softmax, the
//! mxfp4 repack against a scalar E2M1 decoder, the sanitizer's key rules and
//! idempotency, and prefill causality on a 96-token prompt.

use super::{
    KimiK3Config, KimiK3DeltaAttention, KimiK3DeltaCache, KimiK3LinearAttnConfig,
    KimiK3MLAAttention, KimiK3Model, KimiK3SparseMoE, KimiK3TextConfig, ResidualBlocks,
    attn_res_mix,
};
use crate::models::gated_delta::{compute_g, compute_g_lower_bounded};
use crate::models::kimi_linear::{Quantization, ShortConv1d};
use crate::models::switch_layers::situ_activation;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::utils::{array_to_vec_f32, create_causal_mask};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

// Fixture geometry.

const VOCAB: usize = 32;
const HIDDEN: usize = 16;
const LAYERS: usize = 4;
/// KDA heads and head width (`P = 16`).
const KDA_HEADS: usize = 2;
const KDA_HEAD_DIM: usize = 8;
const CONV_KERNEL: usize = 4;
/// MLA geometry.
const MLA_HEADS: usize = 2;
const Q_LORA: usize = 8;
const KV_LORA: usize = 8;
const QK_NOPE: usize = 8;
const QK_ROPE: usize = 4;
const V_HEAD: usize = 8;
/// MoE geometry.
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const MOE_INTER: usize = 8;
const LATENT: usize = 8;
const SHARED: usize = 1;
const DENSE_INTER: usize = 16;
const ATTN_RES_BLOCK: usize = 2;
const SITU_BETA: f32 = 4.0;
const SITU_LINEAR_BETA: f32 = 25.0;
const GATE_LOWER_BOUND: f32 = -5.0;
const RMS_EPS: f32 = 1e-5;

fn kda_p() -> usize {
    KDA_HEADS * KDA_HEAD_DIM
}

/// Layer `i` is KDA for `i` in {0, 2} (1-based `kda_layers = [1, 3]`), MLA
/// for {1, 3}; layer 0 is dense, 1..3 are MoE.
fn tiny_text_config() -> KimiK3TextConfig {
    KimiK3TextConfig {
        model_type: "kimi_linear".to_string(),
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        num_hidden_layers: LAYERS,
        num_attention_heads: MLA_HEADS,
        num_key_value_heads: Some(MLA_HEADS),
        intermediate_size: DENSE_INTER,
        rms_norm_eps: RMS_EPS,
        hidden_act: "situ".to_string(),
        activation_situ_beta: Some(SITU_BETA),
        activation_situ_linear_beta: Some(SITU_LINEAR_BETA),
        attn_res_block_size: Some(ATTN_RES_BLOCK),
        linear_attn_config: KimiK3LinearAttnConfig {
            kda_layers: vec![1, 3],
            full_attn_layers: vec![2, 4],
            num_heads: KDA_HEADS,
            head_dim: KDA_HEAD_DIM,
            short_conv_kernel_size: CONV_KERNEL,
            gate_lower_bound: Some(GATE_LOWER_BOUND),
            use_full_rank_gate: true,
        },
        q_lora_rank: Some(Q_LORA),
        kv_lora_rank: KV_LORA,
        qk_nope_head_dim: QK_NOPE,
        qk_rope_head_dim: QK_ROPE,
        v_head_dim: V_HEAD,
        mla_use_nope: true,
        mla_use_output_gate: true,
        num_experts: EXPERTS,
        num_experts_per_token: TOP_K,
        num_shared_experts: SHARED,
        moe_intermediate_size: MOE_INTER,
        routed_expert_hidden_size: Some(LATENT),
        latent_moe_use_norm: true,
        moe_router_activation_func: "sigmoid".to_string(),
        moe_renormalize: true,
        routed_scaling_factor: 1.0,
        first_k_dense_replace: 1,
        moe_layer_freq: 1,
        use_grouped_topk: true,
        num_expert_group: 1,
        topk_group: 1,
        num_nextn_predict_layers: 0,
        tie_word_embeddings: false,
        quantization: None,
        eos_token_id: Some(serde_json::json!(7)),
    }
}

// Deterministic data.

/// Pseudo-random filler in `[-0.5, 0.5)`.
fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

fn scaled_noise(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    noise(n, seed).into_iter().map(|v| v * scale).collect()
}

fn arr(data: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(data, shape)
}

fn noise_arr(shape: &[i32], seed: u32, scale: f32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    arr(&scaled_noise(n as usize, seed, scale), shape)
}

fn ones_plus_noise(n: usize, seed: u32) -> Vec<f32> {
    scaled_noise(n, seed, 0.2)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect()
}

fn u8_arr(values: &[u32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(&mlxcel_core::from_slice_u32(values, shape), dtype::UINT8)
}

fn read(a: &MlxArray) -> Vec<f32> {
    array_to_vec_f32(a)
}

fn shape(a: &MlxArray) -> Vec<i32> {
    mlxcel_core::array_shape(a)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn assert_close(actual: &MlxArray, expected: &[f32], tol: f32, what: &str) {
    let got = read(actual);
    let diff = max_abs_diff(&got, expected);
    assert!(
        diff <= tol,
        "{what}: max abs diff {diff} exceeds {tol}\n got: {got:?}\n exp: {expected:?}"
    );
}

fn arrays_equal(a: &MlxArray, b: &MlxArray) -> bool {
    let eq = mlxcel_core::array_equal(a, b, true);
    mlxcel_core::eval(&eq);
    mlxcel_core::item_bool(&eq)
}

// Scalar helpers.

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn situ_scalar(up: f32, gate: f32, beta: f32, linear_beta: Option<f32>) -> f32 {
    let a = beta * (gate / beta).tanh() * sigmoid(gate);
    let u = match linear_beta {
        Some(lb) => lb * (up / lb).tanh(),
        None => up,
    };
    a * u
}

/// `out = W x` for a row-major `[rows, cols]` weight.
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols);
    assert_eq!(x.len(), cols);
    (0..rows)
        .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
        .collect()
}

fn rms_norm_scalar(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

// Raw checkpoint fixture (the layout the sanitizer consumes).

struct RawLayout {
    /// Spell the residual weights the legacy way (`self_attention_res.proj_weight`).
    legacy_residual_keys: bool,
}

/// One KDA attention block under `prefix` (raw: split q/k/v, `[C, 1, K]` conv
/// weights, `A_log` two entries longer than the head count).
fn raw_kda_weights(weights: &mut WeightMap, prefix: &str, seed: u32) {
    let p = kda_p() as i32;
    let d = HIDDEN as i32;
    let hd = KDA_HEAD_DIM as i32;
    let h = KDA_HEADS as i32;
    let k = CONV_KERNEL as i32;
    for (i, part) in ["q", "k", "v"].iter().enumerate() {
        weights.insert(
            format!("{prefix}.{part}_proj.weight"),
            noise_arr(&[p, d], seed + i as u32, 0.5),
        );
        weights.insert(
            format!("{prefix}.{part}_conv1d.weight"),
            noise_arr(&[p, 1, k], seed + 10 + i as u32, 0.8),
        );
    }
    weights.insert(
        format!("{prefix}.f_a_proj.weight"),
        noise_arr(&[hd, d], seed + 20, 0.5),
    );
    weights.insert(
        format!("{prefix}.f_b_proj.weight"),
        noise_arr(&[p, hd], seed + 21, 0.5),
    );
    weights.insert(
        format!("{prefix}.b_proj.weight"),
        noise_arr(&[h, d], seed + 22, 0.5),
    );
    weights.insert(
        format!("{prefix}.g_proj.weight"),
        noise_arr(&[p, d], seed + 23, 0.5),
    );
    // `A_log` is longer than the head count, as on the published checkpoint
    // (`[128]` for 96 heads); only the first `num_heads` entries are real.
    let a_log: Vec<f32> = (0..KDA_HEADS + 2).map(|i| (1.0 + i as f32).ln()).collect();
    weights.insert(format!("{prefix}.A_log"), arr(&a_log, &[h + 2]));
    weights.insert(format!("{prefix}.dt_bias"), noise_arr(&[p], seed + 24, 0.5));
    weights.insert(
        format!("{prefix}.o_norm.weight"),
        arr(&ones_plus_noise(KDA_HEAD_DIM, seed + 25), &[hd]),
    );
    weights.insert(
        format!("{prefix}.o_proj.weight"),
        noise_arr(&[d, p], seed + 26, 0.5),
    );
}

/// One MLA attention block under `prefix` (raw: `kv_b_proj`, not yet decomposed).
fn raw_mla_weights(weights: &mut WeightMap, prefix: &str, seed: u32) {
    let d = HIDDEN as i32;
    let h = MLA_HEADS as i32;
    let qhd = (QK_NOPE + QK_ROPE) as i32;
    weights.insert(
        format!("{prefix}.q_a_proj.weight"),
        noise_arr(&[Q_LORA as i32, d], seed, 0.5),
    );
    weights.insert(
        format!("{prefix}.q_a_layernorm.weight"),
        arr(&ones_plus_noise(Q_LORA, seed + 1), &[Q_LORA as i32]),
    );
    weights.insert(
        format!("{prefix}.q_b_proj.weight"),
        noise_arr(&[h * qhd, Q_LORA as i32], seed + 2, 0.5),
    );
    weights.insert(
        format!("{prefix}.kv_a_proj_with_mqa.weight"),
        noise_arr(&[(KV_LORA + QK_ROPE) as i32, d], seed + 3, 0.5),
    );
    weights.insert(
        format!("{prefix}.kv_a_layernorm.weight"),
        arr(&ones_plus_noise(KV_LORA, seed + 4), &[KV_LORA as i32]),
    );
    weights.insert(
        format!("{prefix}.kv_b_proj.weight"),
        noise_arr(
            &[h * (QK_NOPE + V_HEAD) as i32, KV_LORA as i32],
            seed + 5,
            0.5,
        ),
    );
    weights.insert(
        format!("{prefix}.g_proj.weight"),
        noise_arr(&[h * V_HEAD as i32, d], seed + 6, 0.5),
    );
    weights.insert(
        format!("{prefix}.o_proj.weight"),
        noise_arr(&[d, h * V_HEAD as i32], seed + 7, 0.5),
    );
}

/// One MoE block under `prefix` (`block_sparse_moe`, dense MLX-style experts).
fn raw_moe_weights(weights: &mut WeightMap, prefix: &str, seed: u32, num_experts: usize) {
    let d = HIDDEN as i32;
    let e = num_experts as i32;
    let lat = LATENT as i32;
    let mi = MOE_INTER as i32;
    let shared = (SHARED * MOE_INTER) as i32;
    weights.insert(
        format!("{prefix}.gate.weight"),
        noise_arr(&[e, d], seed, 0.5),
    );
    weights.insert(
        format!("{prefix}.gate.e_score_correction_bias"),
        noise_arr(&[e], seed + 1, 0.2),
    );
    weights.insert(
        format!("{prefix}.routed_expert_down_proj.weight"),
        noise_arr(&[lat, d], seed + 2, 0.5),
    );
    weights.insert(
        format!("{prefix}.routed_expert_norm.weight"),
        arr(&ones_plus_noise(LATENT, seed + 3), &[lat]),
    );
    weights.insert(
        format!("{prefix}.routed_expert_up_proj.weight"),
        noise_arr(&[d, lat], seed + 4, 0.5),
    );
    weights.insert(
        format!("{prefix}.shared_experts.gate_proj.weight"),
        noise_arr(&[shared, d], seed + 5, 0.5),
    );
    weights.insert(
        format!("{prefix}.shared_experts.up_proj.weight"),
        noise_arr(&[shared, d], seed + 6, 0.5),
    );
    weights.insert(
        format!("{prefix}.shared_experts.down_proj.weight"),
        noise_arr(&[d, shared], seed + 7, 0.5),
    );
    for ex in 0..num_experts as u32 {
        weights.insert(
            format!("{prefix}.experts.{ex}.w1.weight"),
            noise_arr(&[mi, lat], seed + 100 + ex * 3, 0.5),
        );
        weights.insert(
            format!("{prefix}.experts.{ex}.w3.weight"),
            noise_arr(&[mi, lat], seed + 101 + ex * 3, 0.5),
        );
        weights.insert(
            format!("{prefix}.experts.{ex}.w2.weight"),
            noise_arr(&[lat, mi], seed + 102 + ex * 3, 0.5),
        );
    }
}

fn raw_dense_mlp_weights(weights: &mut WeightMap, prefix: &str, seed: u32) {
    let d = HIDDEN as i32;
    let i = DENSE_INTER as i32;
    weights.insert(
        format!("{prefix}.gate_proj.weight"),
        noise_arr(&[i, d], seed, 0.5),
    );
    weights.insert(
        format!("{prefix}.up_proj.weight"),
        noise_arr(&[i, d], seed + 1, 0.5),
    );
    weights.insert(
        format!("{prefix}.down_proj.weight"),
        noise_arr(&[d, i], seed + 2, 0.5),
    );
}

fn raw_residual_weights(
    weights: &mut WeightMap,
    prefix: &str,
    stem: &str,
    seed: u32,
    legacy: bool,
) {
    let d = HIDDEN as i32;
    let (proj_key, norm_key) = if legacy {
        (
            format!("{prefix}.{stem}.proj_weight"),
            format!("{prefix}.{stem}.norm_weight"),
        )
    } else {
        (
            format!("{prefix}.{stem}_proj.weight"),
            format!("{prefix}.{stem}_norm.weight"),
        )
    };
    weights.insert(proj_key, noise_arr(&[1, d], seed, 0.5));
    weights.insert(norm_key, arr(&ones_plus_noise(HIDDEN, seed + 1), &[d]));
}

/// The whole tiny checkpoint in the published key layout, `language_model.`
/// prefix included, plus the non-text tensors the sanitizer must drop.
fn raw_checkpoint(config: &KimiK3TextConfig, layout: RawLayout) -> WeightMap {
    let d = HIDDEN as i32;
    let v = VOCAB as i32;
    let mut w = WeightMap::new();
    let lm = |k: &str| format!("language_model.{k}");
    w.insert(lm("model.embed_tokens.weight"), noise_arr(&[v, d], 1, 1.0));
    w.insert(
        lm("model.norm.weight"),
        arr(&ones_plus_noise(HIDDEN, 2), &[d]),
    );
    w.insert(lm("lm_head.weight"), noise_arr(&[v, d], 3, 0.5));
    raw_residual_weights(
        &mut w,
        "language_model.model",
        "output_attn_res",
        4,
        layout.legacy_residual_keys,
    );

    for l in 0..config.num_hidden_layers {
        let prefix = lm(&format!("model.layers.{l}"));
        let seed = 1000 * (l as u32 + 1);
        w.insert(
            format!("{prefix}.input_layernorm.weight"),
            arr(&ones_plus_noise(HIDDEN, seed), &[d]),
        );
        w.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            arr(&ones_plus_noise(HIDDEN, seed + 1), &[d]),
        );
        raw_residual_weights(
            &mut w,
            &prefix,
            "self_attention_res",
            seed + 2,
            layout.legacy_residual_keys,
        );
        raw_residual_weights(
            &mut w,
            &prefix,
            "mlp_res",
            seed + 4,
            layout.legacy_residual_keys,
        );
        if config.is_linear_layer(l) {
            raw_kda_weights(&mut w, &format!("{prefix}.self_attn"), seed + 10);
        } else {
            raw_mla_weights(&mut w, &format!("{prefix}.self_attn"), seed + 10);
        }
        if config.is_moe_layer(l) {
            raw_moe_weights(
                &mut w,
                &format!("{prefix}.block_sparse_moe"),
                seed + 50,
                config.num_experts,
            );
        } else {
            raw_dense_mlp_weights(&mut w, &format!("{prefix}.mlp"), seed + 50);
        }
    }

    // Tensors that must not survive: the vision tower and projector, the MTP
    // head, and a layer past `num_hidden_layers` (a layer-truncated copy whose
    // shard still holds part of the next layer).
    w.insert(
        "vision_tower.blocks.0.attn.qkv.weight".to_string(),
        noise_arr(&[4, 4], 9, 1.0),
    );
    w.insert(
        "mm_projector.proj.weight".to_string(),
        noise_arr(&[4, 4], 10, 1.0),
    );
    w.insert(
        lm("model.mtp.0.embed_tokens.weight"),
        noise_arr(&[4, 4], 11, 1.0),
    );
    w.insert(
        lm(&format!(
            "model.layers.{}.input_layernorm.weight",
            config.num_hidden_layers
        )),
        noise_arr(&[d], 12, 1.0),
    );
    w
}

/// Sanitized weights of the tiny checkpoint.
fn sanitized(config: &KimiK3TextConfig) -> WeightMap {
    let raw = raw_checkpoint(
        config,
        RawLayout {
            legacy_residual_keys: false,
        },
    );
    KimiK3Model::sanitize_weights(raw, config).expect("tiny checkpoint sanitizes")
}

/// The raw map with the `language_model.` prefix stripped, for building
/// reference layers from the unfused planes.
fn stripped_raw(config: &KimiK3TextConfig) -> WeightMap {
    raw_checkpoint(
        config,
        RawLayout {
            legacy_residual_keys: false,
        },
    )
    .into_iter()
    .filter_map(|(k, v)| {
        k.strip_prefix("language_model.")
            .map(|s| (s.to_string(), v))
    })
    .collect()
}

// SiTU.

#[test]
fn situ_matches_formula() {
    let up = noise_arr(&[2, 3, 5], 7, 6.0);
    let gate = noise_arr(&[2, 3, 5], 8, 6.0);
    let up_v = read(&up);
    let gate_v = read(&gate);

    let expected: Vec<f32> = up_v
        .iter()
        .zip(&gate_v)
        .map(|(u, g)| situ_scalar(*u, *g, SITU_BETA, Some(SITU_LINEAR_BETA)))
        .collect();
    let actual = situ_activation(&gate, &up, SITU_BETA, Some(SITU_LINEAR_BETA));
    assert_eq!(mlxcel_core::array_dtype(&actual), dtype::FLOAT32);
    assert_close(&actual, &expected, 1e-6, "situ(beta 4, linear_beta 25)");

    // Without `linear_beta` the second factor is `up` itself: a * up.
    let expected_plain: Vec<f32> = up_v
        .iter()
        .zip(&gate_v)
        .map(|(u, g)| SITU_BETA * (g / SITU_BETA).tanh() * sigmoid(*g) * u)
        .collect();
    let plain = situ_activation(&gate, &up, SITU_BETA, None);
    assert_close(
        &plain,
        &expected_plain,
        1e-6,
        "situ(beta 4, no linear_beta)",
    );

    // Half-precision inputs come back in their own dtype.
    let up16 = mlxcel_core::astype(&up, dtype::FLOAT16);
    let gate16 = mlxcel_core::astype(&gate, dtype::FLOAT16);
    let half = situ_activation(&gate16, &up16, SITU_BETA, Some(SITU_LINEAR_BETA));
    assert_eq!(mlxcel_core::array_dtype(&half), dtype::FLOAT16);
}

// Lower-bounded gate.

#[test]
fn gated_delta_lower_bound_gate_range() {
    let h = KDA_HEADS as i32;
    let d = KDA_HEAD_DIM as i32;
    let a_log = arr(&[0.0, 1.0], &[h, 1]);
    let dt_bias = mlxcel_core::zeros(&[h, d], dtype::FLOAT32);
    // Extreme pre-activations in both directions plus zero.
    let mut a = Vec::new();
    for t in 0..3 {
        let v = [-50.0f32, 0.0, 50.0][t];
        a.extend(std::iter::repeat_n(v, (h * d) as usize));
    }
    let a = arr(&a, &[1, 3, h, d]);

    let g = compute_g_lower_bounded(&a_log, &a, &dt_bias, Some(GATE_LOWER_BOUND));
    assert_eq!(mlxcel_core::array_dtype(&g), dtype::FLOAT32);
    assert_eq!(shape(&g), vec![1, 3, h, d]);
    let g_v = read(&g);
    let floor = GATE_LOWER_BOUND.exp();
    for v in &g_v {
        assert!(
            *v >= floor - 1e-6 && *v <= 1.0 + 1e-6,
            "gate {v} outside (e^-5, 1)"
        );
    }
    let per_t = (h * d) as usize;
    // a = -50: sigmoid -> 0, g -> 1. a = +50: sigmoid -> 1, g -> e^-5.
    assert!(g_v[..per_t].iter().all(|v| (v - 1.0).abs() < 1e-5));
    assert!(g_v[2 * per_t..].iter().all(|v| (v - floor).abs() < 1e-5));
    // a = 0: g = exp(lb * sigmoid(0)) = exp(lb / 2).
    let mid = (GATE_LOWER_BOUND / 2.0).exp();
    assert!(g_v[per_t..2 * per_t].iter().all(|v| (v - mid).abs() < 1e-5));

    // The softplus gate has no floor: at a = +50 it is exp(-exp(A_log) * 50),
    // about 2e-22 for A_log = 0 and an f32 underflow to 0 for A_log = 1, both
    // far below the e^-5 floor the bounded form holds the +50 rows at.
    let plain = read(&compute_g(&a_log, &a, &dt_bias));
    assert!(plain[2 * per_t..].iter().all(|v| *v < floor));
    assert!(plain[2 * per_t..].contains(&0.0));

    // `None` reproduces `compute_g` bit for bit.
    let none = compute_g_lower_bounded(&a_log, &a, &dt_bias, None);
    assert!(arrays_equal(&none, &compute_g(&a_log, &a, &dt_bias)));
}

// KDA.

fn kda_attention(config: &KimiK3TextConfig, layer: usize) -> KimiK3DeltaAttention {
    let weights = sanitized(config);
    KimiK3DeltaAttention::from_weights(
        &weights,
        config,
        &format!("model.layers.{layer}.self_attn"),
        dtype::FLOAT32,
    )
    .expect("fused KDA layer loads")
}

#[test]
fn kda_fused_qkv_equals_separate_projections() {
    let config = tiny_text_config();
    let attn = kda_attention(&config, 0);
    let x = noise_arr(&[1, 16, HIDDEN as i32], 21, 2.0);

    let mut cache = KimiK3DeltaCache::new();
    let fused = attn.forward(&x, &mut cache);

    // Reference: three projections and three convs from the raw planes, then
    // the same post-conv math through `finish`.
    let raw = stripped_raw(&config);
    let prefix = "model.layers.0.self_attn";
    let p = kda_p();
    let per_head = [1, 16, KDA_HEADS as i32, KDA_HEAD_DIM as i32];
    let mut parts = Vec::new();
    for part in ["q", "k", "v"] {
        let proj =
            UnifiedLinear::from_weights(&raw, &format!("{prefix}.{part}_proj"), 64, 4).unwrap();
        let conv_w = raw.get(&format!("{prefix}.{part}_conv1d.weight")).unwrap();
        let conv = ShortConv1d::new(mlxcel_core::swap_axes(conv_w, 1, 2), CONV_KERNEL, p);
        let (out, _state) = conv.forward(&proj.forward(&x), None, None);
        parts.push(mlxcel_core::reshape(&out, &per_head));
    }
    let mut ref_cache = KimiK3DeltaCache::new();
    let separate = attn.finish(&x, &parts[0], &parts[1], &parts[2], &mut ref_cache);

    assert_eq!(shape(&fused), vec![1, 16, HIDDEN as i32]);
    assert_close(
        &fused,
        &read(&separate),
        1e-4,
        "fused qkv vs separate projections",
    );
    assert_eq!(cache.offset, 16);
    assert_eq!(
        shape(cache.conv.as_ref().unwrap()),
        vec![1, (CONV_KERNEL - 1) as i32, 3 * p as i32]
    );
    assert_eq!(
        shape(cache.ssm.as_ref().unwrap()),
        vec![
            1,
            KDA_HEADS as i32,
            KDA_HEAD_DIM as i32,
            KDA_HEAD_DIM as i32
        ]
    );
    assert_eq!(
        mlxcel_core::array_dtype(cache.ssm.as_ref().unwrap()),
        dtype::FLOAT32
    );
}

#[test]
fn decode_matches_prefill() {
    let config = tiny_text_config();
    let attn = kda_attention(&config, 0);
    let x = noise_arr(&[1, 16, HIDDEN as i32], 22, 2.0);

    let mut full_cache = KimiK3DeltaCache::new();
    let full = read(&attn.forward(&x, &mut full_cache));

    let mut step_cache = KimiK3DeltaCache::new();
    let head = mlxcel_core::slice(&x, &[0, 0, 0], &[1, 8, HIDDEN as i32]);
    let _ = attn.forward(&head, &mut step_cache);
    let mut decoded = Vec::new();
    for t in 8..16 {
        let tok = mlxcel_core::slice(&x, &[0, t, 0], &[1, t + 1, HIDDEN as i32]);
        let y = attn.forward(&tok, &mut step_cache);
        assert_eq!(shape(&y), vec![1, 1, HIDDEN as i32]);
        decoded.extend(read(&y));
    }
    let diff = max_abs_diff(&decoded, &full[8 * HIDDEN..]);
    assert!(diff <= 1e-3, "decode steps diverge from prefill: {diff}");
    assert_eq!(step_cache.offset, full_cache.offset);
    let ssm_diff = max_abs_diff(
        &read(step_cache.ssm.as_ref().unwrap()),
        &read(full_cache.ssm.as_ref().unwrap()),
    );
    assert!(ssm_diff <= 1e-3, "ssm state diverges: {ssm_diff}");
}

// MLA.

/// Unabsorbed reference: per-head k / v from `kv_b_proj`, softmax over the
/// full q (nope + pe) and k, output gate, `o_proj`. Returns `[1, L, D]`.
fn mla_reference(raw: &WeightMap, prefix: &str, x: &MlxArray) -> UniquePtr<MlxArray> {
    let l = shape(x)[1];
    let h = MLA_HEADS as i32;
    let qhd = (QK_NOPE + QK_ROPE) as i32;
    let lin =
        |name: &str| UnifiedLinear::from_weights(raw, &format!("{prefix}.{name}"), 64, 4).unwrap();
    let norm = |name: &str| {
        RMSNorm::new(
            mlxcel_core::copy(raw.get(&format!("{prefix}.{name}.weight")).unwrap()),
            1e-6,
        )
    };

    let q = lin("q_b_proj").forward(&norm("q_a_layernorm").forward(&lin("q_a_proj").forward(x)));
    let q = mlxcel_core::transpose_axes(&mlxcel_core::reshape(&q, &[1, l, h, qhd]), &[0, 2, 1, 3]);

    let c = lin("kv_a_proj_with_mqa").forward(x);
    let latent = mlxcel_core::slice(&c, &[0, 0, 0], &[1, l, KV_LORA as i32]);
    let latent = norm("kv_a_layernorm").forward(&latent);
    let k_pe = mlxcel_core::slice(
        &c,
        &[0, 0, KV_LORA as i32],
        &[1, l, (KV_LORA + QK_ROPE) as i32],
    );
    let k_pe = mlxcel_core::transpose_axes(
        &mlxcel_core::reshape(&k_pe, &[1, l, 1, QK_ROPE as i32]),
        &[0, 2, 1, 3],
    );
    let k_pe = mlxcel_core::broadcast_to(&k_pe, &[1, h, l, QK_ROPE as i32]);

    let kv = lin("kv_b_proj").forward(&latent);
    let kv = mlxcel_core::transpose_axes(
        &mlxcel_core::reshape(&kv, &[1, l, h, (QK_NOPE + V_HEAD) as i32]),
        &[0, 2, 1, 3],
    );
    let k_nope = mlxcel_core::slice(&kv, &[0, 0, 0, 0], &[1, h, l, QK_NOPE as i32]);
    let v = mlxcel_core::slice(
        &kv,
        &[0, 0, 0, QK_NOPE as i32],
        &[1, h, l, (QK_NOPE + V_HEAD) as i32],
    );
    let k = mlxcel_core::concatenate(&k_nope, &k_pe, -1);

    let scale = (qhd as f32).powf(-0.5);
    let scores = mlxcel_core::multiply_scalar(
        &mlxcel_core::matmul(&q, &mlxcel_core::swap_axes(&k, -1, -2)),
        scale,
    );
    let scores = mlxcel_core::add(&scores, &create_causal_mask(l, 0));
    let probs = mlxcel_core::softmax(&scores, -1);
    let o = mlxcel_core::matmul(&probs, &v);
    let o = mlxcel_core::reshape(&mlxcel_core::transpose_axes(&o, &[0, 2, 1, 3]), &[1, l, -1]);
    let gate = mlxcel_core::sigmoid(&lin("g_proj").forward(x));
    lin("o_proj").forward(&mlxcel_core::multiply(&o, &gate))
}

#[test]
fn mla_q_lora_and_output_gate() {
    let config = tiny_text_config();
    let weights = sanitized(&config);
    let prefix = "model.layers.1.self_attn";
    assert!(weights.contains_key(&format!("{prefix}.embed_q.weight")));
    assert!(!weights.contains_key(&format!("{prefix}.kv_b_proj.weight")));
    let attn =
        KimiK3MLAAttention::from_weights(&weights, &config, prefix).expect("MLA layer loads");
    let raw = stripped_raw(&config);

    let x = noise_arr(&[1, 5, HIDDEN as i32], 31, 2.0);
    let expected = read(&mla_reference(&raw, prefix, &x));

    // Prefill on the first four tokens against the reference's first four rows.
    let prefill = mlxcel_core::slice(&x, &[0, 0, 0], &[1, 4, HIDDEN as i32]);
    let mut cache = KVCache::new();
    let out = attn.forward(&prefill, Some(&create_causal_mask(4, 0)), &mut cache);
    assert_eq!(shape(&out), vec![1, 4, HIDDEN as i32]);
    assert_close(
        &out,
        &expected[..4 * HIDDEN],
        1e-4,
        "absorbed q-LoRA MLA prefill vs unabsorbed",
    );
    assert_eq!(cache.offset, 4);

    // Decode the fifth token through the latent path against the last row.
    let tok = mlxcel_core::slice(&x, &[0, 4, 0], &[1, 5, HIDDEN as i32]);
    let step = attn.forward(&tok, None, &mut cache);
    assert_eq!(shape(&step), vec![1, 1, HIDDEN as i32]);
    assert_close(
        &step,
        &expected[4 * HIDDEN..],
        1e-4,
        "absorbed MLA decode vs unabsorbed",
    );
    assert_eq!(cache.offset, 5);
}

// Latent MoE.

#[test]
fn latent_moe_forward_shapes_and_norm() {
    let config = tiny_text_config();
    let weights = sanitized(&config);
    let prefix = "model.layers.1.mlp";
    assert_eq!(
        shape(
            weights
                .get(&format!("{prefix}.switch_mlp.gate_proj.weight"))
                .unwrap()
        ),
        vec![EXPERTS as i32, MOE_INTER as i32, LATENT as i32]
    );
    let moe = KimiK3SparseMoE::from_weights(&weights, &config, prefix).expect("latent MoE loads");
    let raw = stripped_raw(&config);
    let plane = |name: &str| {
        read(
            raw.get(&format!("model.layers.1.block_sparse_moe.{name}"))
                .unwrap(),
        )
    };

    let x = noise_arr(&[1, 3, HIDDEN as i32], 41, 2.0);
    let out = moe.forward(&x);
    assert_eq!(shape(&out), vec![1, 3, HIDDEN as i32]);

    let gate_w = plane("gate.weight");
    let bias = plane("gate.e_score_correction_bias");
    let down_w = plane("routed_expert_down_proj.weight");
    let norm_w = plane("routed_expert_norm.weight");
    let up_w = plane("routed_expert_up_proj.weight");
    let sg = plane("shared_experts.gate_proj.weight");
    let su = plane("shared_experts.up_proj.weight");
    let sd = plane("shared_experts.down_proj.weight");
    let experts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..EXPERTS)
        .map(|e| {
            (
                plane(&format!("experts.{e}.w1.weight")),
                plane(&format!("experts.{e}.w3.weight")),
                plane(&format!("experts.{e}.w2.weight")),
            )
        })
        .collect();

    let x_v = read(&x);
    let mut expected = Vec::with_capacity(3 * HIDDEN);
    for t in 0..3 {
        let h = &x_v[t * HIDDEN..(t + 1) * HIDDEN];
        let logits = matvec(&gate_w, EXPERTS, HIDDEN, h);
        let scores: Vec<f32> = logits.iter().map(|v| sigmoid(*v)).collect();
        let selection: Vec<f32> = scores.iter().zip(&bias).map(|(s, b)| s + b).collect();
        let mut order: Vec<usize> = (0..EXPERTS).collect();
        order.sort_by(|a, b| selection[*b].partial_cmp(&selection[*a]).unwrap());
        let chosen = &order[..TOP_K];
        let denom: f32 = chosen.iter().map(|i| scores[*i]).sum::<f32>() + 1e-20;
        let z = matvec(&down_w, LATENT, HIDDEN, h);
        let mut routed = vec![0.0f32; LATENT];
        for &e in chosen {
            let (w1, w3, w2) = &experts[e];
            let gate = matvec(w1, MOE_INTER, LATENT, &z);
            let up = matvec(w3, MOE_INTER, LATENT, &z);
            let act: Vec<f32> = up
                .iter()
                .zip(&gate)
                .map(|(u, g)| situ_scalar(*u, *g, SITU_BETA, Some(SITU_LINEAR_BETA)))
                .collect();
            let y = matvec(w2, LATENT, MOE_INTER, &act);
            let w = scores[e] / denom;
            for (r, v) in routed.iter_mut().zip(y) {
                *r += w * v;
            }
        }
        let normed = rms_norm_scalar(&routed, &norm_w, RMS_EPS);
        let mut y = matvec(&up_w, HIDDEN, LATENT, &normed);
        let s_gate = matvec(&sg, SHARED * MOE_INTER, HIDDEN, h);
        let s_up = matvec(&su, SHARED * MOE_INTER, HIDDEN, h);
        let s_act: Vec<f32> = s_up
            .iter()
            .zip(&s_gate)
            .map(|(u, g)| situ_scalar(*u, *g, SITU_BETA, Some(SITU_LINEAR_BETA)))
            .collect();
        let shared = matvec(&sd, HIDDEN, SHARED * MOE_INTER, &s_act);
        for (a, b) in y.iter_mut().zip(shared) {
            *a += b;
        }
        expected.extend(y);
    }
    assert_close(&out, &expected, 1e-4, "latent MoE vs per-expert loop");
}

// Attention Residuals.

#[test]
fn attn_res_mix_matches_reference() {
    let d = HIDDEN as i32;
    let partial = noise_arr(&[1, 2, d], 51, 2.0);
    let w_eff = noise_arr(&[d, 1], 52, 1.0);
    let eps = RMS_EPS;

    // Empty blocks: the partial passes through unchanged.
    let empty = ResidualBlocks::new();
    assert!(arrays_equal(
        &attn_res_mix(&empty, &partial, &w_eff, eps),
        &partial
    ));

    let mut blocks = ResidualBlocks::new();
    let raws: Vec<UniquePtr<MlxArray>> =
        (0..3).map(|i| noise_arr(&[1, 2, d], 60 + i, 2.0)).collect();
    for r in &raws {
        blocks.push(r, eps);
    }
    assert_eq!(blocks.len(), 3);
    let mixed = attn_res_mix(&blocks, &partial, &w_eff, eps);
    assert_eq!(shape(&mixed), vec![1, 2, d]);

    // Scalar reference per token row.
    let w = read(&w_eff);
    let p_v = read(&partial);
    let r_v: Vec<Vec<f32>> = raws.iter().map(|r| read(r)).collect();
    let logit = |row: &[f32]| -> f32 {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        row.iter().zip(&w).map(|(v, g)| v * g).sum::<f32>() * inv
    };
    let mut expected = Vec::with_capacity(2 * HIDDEN);
    for t in 0..2 {
        let rows: Vec<&[f32]> = r_v
            .iter()
            .map(|r| &r[t * HIDDEN..(t + 1) * HIDDEN])
            .chain(std::iter::once(&p_v[t * HIDDEN..(t + 1) * HIDDEN]))
            .collect();
        let logits: Vec<f32> = rows.iter().map(|r| logit(r)).collect();
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for i in 0..HIDDEN {
            expected.push(rows.iter().zip(&exps).map(|(r, e)| r[i] * e / sum).sum());
        }
    }
    assert_close(&mixed, &expected, 1e-5, "attn_res_mix vs scalar softmax");

    // The result is cast to the partial's dtype.
    let partial16 = mlxcel_core::astype(&partial, dtype::BFLOAT16);
    let mixed16 = attn_res_mix(&blocks, &partial16, &w_eff, eps);
    assert_eq!(mlxcel_core::array_dtype(&mixed16), dtype::BFLOAT16);
}

#[test]
fn attn_res_block_size_two_over_four_layers_stores_two_blocks() {
    let config = tiny_text_config();
    let weights = sanitized(&config);
    let model = KimiK3Model::from_weights(&weights, &config).expect("tiny model loads");
    let ids = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5], &[1, 5]);
    let h = model.embed_tokens.forward(&ids);
    let mask = create_causal_mask(5, 0);
    let mut caches = model.make_layer_caches();
    let (h, blocks) = model.run_layers(h, Some(&mask), &mut caches);
    assert_eq!(blocks.len(), LAYERS / ATTN_RES_BLOCK);
    assert_eq!(shape(&h), vec![1, 5, HIDDEN as i32]);
    // Layer 0 froze the embeddings themselves.
    assert!(arrays_equal(
        &blocks.raw[0],
        &model.embed_tokens.forward(&ids)
    ));
}

// Sanitize.

#[test]
fn sanitize_mxfp4_experts_and_fused_conv() {
    let mut config = tiny_text_config();
    config.num_experts = 2;
    let mut raw = raw_checkpoint(
        &config,
        RawLayout {
            legacy_residual_keys: true,
        },
    );
    // Replace layer 1's dense experts with compressed-tensors mxfp4 planes:
    // uint8 `weight_packed [4, 16]` (32 codes per row) and one E8M0 scale
    // per 32-wide group.
    let moe = "language_model.model.layers.1.block_sparse_moe";
    for e in 0..config.num_experts {
        for leaf in ["w1", "w2", "w3"] {
            raw.remove(&format!("{moe}.experts.{e}.{leaf}.weight"));
        }
    }
    for e in 0..2u32 {
        for (li, leaf) in ["w1", "w3", "w2"].iter().enumerate() {
            let bytes: Vec<u32> = (0..64)
                .map(|i| (i as u32 * 7 + e * 13 + li as u32) % 256)
                .collect();
            raw.insert(
                format!("{moe}.experts.{e}.{leaf}.weight_packed"),
                u8_arr(&bytes, &[4, 16]),
            );
            raw.insert(
                format!("{moe}.experts.{e}.{leaf}.weight_scale"),
                u8_arr(&[127, 126, 128, 125], &[4, 1]),
            );
        }
    }

    let out = KimiK3Model::sanitize_weights(raw, &config).expect("sanitizes");

    // Scope: prefix stripped, non-text and out-of-range keys gone.
    assert!(
        out.keys()
            .all(|k| k.starts_with("model.") || k.starts_with("lm_head."))
    );
    assert!(
        !out.keys()
            .any(|k| k.contains("vision_tower") || k.contains("mm_projector"))
    );
    assert!(!out.keys().any(|k| k.starts_with("model.mtp")));
    assert!(!out.contains_key(&format!("model.layers.{LAYERS}.input_layernorm.weight")));
    assert!(out.contains_key("model.embed_tokens.weight"));

    // Legacy residual spellings renamed.
    assert!(out.contains_key("model.output_attn_res_proj.weight"));
    assert!(out.contains_key("model.output_attn_res_norm.weight"));
    assert!(out.contains_key("model.layers.0.self_attention_res_proj.weight"));
    assert!(out.contains_key("model.layers.0.mlp_res_norm.weight"));
    assert!(
        !out.keys()
            .any(|k| k.ends_with(".proj_weight") || k.ends_with(".norm_weight"))
    );

    // KDA: fused planes, `[C, K, 1]` conv, `A_log` sliced to the head count.
    let p = kda_p() as i32;
    let attn = "model.layers.0.self_attn";
    assert_eq!(
        shape(out.get(&format!("{attn}.qkv_proj.weight")).unwrap()),
        vec![3 * p, HIDDEN as i32]
    );
    assert_eq!(
        shape(out.get(&format!("{attn}.qkv_conv.conv.weight")).unwrap()),
        vec![3 * p, CONV_KERNEL as i32, 1]
    );
    assert_eq!(
        shape(out.get(&format!("{attn}.A_log")).unwrap()),
        vec![KDA_HEADS as i32]
    );
    assert_eq!(
        mlxcel_core::array_dtype(out.get(&format!("{attn}.A_log")).unwrap()),
        dtype::FLOAT32
    );
    assert_eq!(shape(out.get(&format!("{attn}.dt_bias")).unwrap()), vec![p]);
    for part in ["q", "k", "v"] {
        assert!(!out.contains_key(&format!("{attn}.{part}_proj.weight")));
        assert!(!out.contains_key(&format!("{attn}.{part}_conv1d.weight")));
    }
    // The fused conv is the three raw convs stacked in q, k, v order.
    let fused_conv = read(out.get(&format!("{attn}.qkv_conv.conv.weight")).unwrap());
    let raw_again = stripped_raw(&config);
    let q_conv = read(&mlxcel_core::swap_axes(
        raw_again.get(&format!("{attn}.q_conv1d.weight")).unwrap(),
        1,
        2,
    ));
    assert_eq!(&fused_conv[..q_conv.len()], &q_conv[..]);

    // MLA: `kv_b_proj` decomposed.
    assert_eq!(
        shape(out.get("model.layers.1.self_attn.embed_q.weight").unwrap()),
        vec![MLA_HEADS as i32, KV_LORA as i32, QK_NOPE as i32]
    );
    assert_eq!(
        shape(
            out.get("model.layers.1.self_attn.unembed_out.weight")
                .unwrap()
        ),
        vec![MLA_HEADS as i32, V_HEAD as i32, KV_LORA as i32]
    );

    // MoE: renamed, experts stacked and viewed as uint32.
    let mlp = "model.layers.1.mlp";
    assert!(!out.keys().any(|k| k.contains("block_sparse_moe")));
    assert!(out.contains_key(&format!("{mlp}.gate.weight")));
    assert!(out.contains_key(&format!("{mlp}.e_score_correction_bias")));
    assert!(out.contains_key(&format!("{mlp}.routed_expert_norm.weight")));
    assert!(out.contains_key(&format!("{mlp}.shared_experts.down_proj.weight")));
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        let w = out.get(&format!("{mlp}.switch_mlp.{proj}.weight")).unwrap();
        assert_eq!(shape(w), vec![2, 4, 4], "{proj} packed plane");
        assert_eq!(
            mlxcel_core::array_dtype(w),
            dtype::UINT32,
            "{proj} packed dtype"
        );
        let s = out.get(&format!("{mlp}.switch_mlp.{proj}.scales")).unwrap();
        assert_eq!(shape(s), vec![2, 4, 1], "{proj} scales");
        assert_eq!(
            mlxcel_core::array_dtype(s),
            dtype::UINT8,
            "{proj} scales dtype"
        );
        assert!(!out.contains_key(&format!("{mlp}.switch_mlp.{proj}.biases")));
    }
    assert!(!out.keys().any(|k| k.contains(".experts.")));
    // Layer 2's dense MLX-style experts stacked as well.
    assert_eq!(
        shape(
            out.get("model.layers.2.mlp.switch_mlp.down_proj.weight")
                .unwrap()
        ),
        vec![config.num_experts as i32, LATENT as i32, MOE_INTER as i32]
    );

    // Idempotent: a second pass is a no-op on every key and every tensor.
    let mut first: Vec<(String, UniquePtr<MlxArray>)> = out
        .iter()
        .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
        .collect();
    first.sort_by(|a, b| a.0.cmp(&b.0));
    let again = KimiK3Model::sanitize_weights(out, &config).expect("second pass sanitizes");
    assert_eq!(again.len(), first.len());
    for (k, v) in &first {
        let w = again
            .get(k)
            .unwrap_or_else(|| panic!("second pass lost {k}"));
        assert_eq!(shape(w), shape(v), "{k} shape changed on the second pass");
        assert!(arrays_equal(w, v), "{k} changed on the second pass");
    }
}

#[test]
fn sanitize_refuses_a_truncated_expert_set() {
    let config = tiny_text_config();
    let mut raw = raw_checkpoint(
        &config,
        RawLayout {
            legacy_residual_keys: false,
        },
    );
    raw.remove(&format!(
        "language_model.model.layers.1.block_sparse_moe.experts.{}.w1.weight",
        EXPERTS - 1
    ));
    let err = KimiK3Model::sanitize_weights(raw, &config)
        .err()
        .expect("missing expert must be refused");
    assert!(
        err.contains("model.layers.1.block_sparse_moe.experts"),
        "{err}"
    );
    assert!(
        err.contains(&format!("{} of the {EXPERTS} experts", EXPERTS - 1)),
        "{err}"
    );
}

/// OCP E2M1: bit 3 sign, bits 2..1 exponent, bit 0 mantissa.
fn e2m1(code: u32) -> f32 {
    let sign = if code & 0x8 != 0 { -1.0 } else { 1.0 };
    let exp = (code >> 1) & 0x3;
    let man = (code & 0x1) as f32;
    let mag = if exp == 0 {
        man * 0.5
    } else {
        2f32.powi(exp as i32 - 1) * (1.0 + man / 2.0)
    };
    sign * mag
}

#[test]
fn mxfp4_repack_matches_scalar_dequant() {
    // One expert, 4 rows of 32 codes: every nibble value appears, low nibble
    // first inside each byte, with a different E8M0 scale per row.
    let mut config = tiny_text_config();
    config.num_experts = 1;
    let bytes: Vec<u32> = (0..64).map(|i| (i as u32 * 37 + 11) % 256).collect();
    let scales: [u32; 4] = [127, 126, 128, 125];
    let moe = "language_model.model.layers.1.block_sparse_moe";
    let mut raw = WeightMap::new();
    for leaf in ["w1", "w3", "w2"] {
        raw.insert(
            format!("{moe}.experts.0.{leaf}.weight_packed"),
            u8_arr(&bytes, &[4, 16]),
        );
        raw.insert(
            format!("{moe}.experts.0.{leaf}.weight_scale"),
            u8_arr(&scales, &[4, 1]),
        );
    }
    let out = KimiK3Model::sanitize_weights(raw, &config).expect("sanitizes");

    let w = out
        .get("model.layers.1.mlp.switch_mlp.gate_proj.weight")
        .unwrap();
    let s = out
        .get("model.layers.1.mlp.switch_mlp.gate_proj.scales")
        .unwrap();
    assert_eq!(shape(w), vec![1, 4, 4]);
    let w = mlxcel_core::reshape(w, &[4, 4]);
    let s = mlxcel_core::reshape(s, &[4, 1]);
    let dequant = unsafe { mlxcel_core::dequantize(&w, &s, std::ptr::null(), 32, 4, "mxfp4") };
    assert_eq!(shape(&dequant), vec![4, 32]);
    let got = read(&dequant);

    let mut expected = Vec::with_capacity(128);
    for row in 0..4 {
        let scale = 2f32.powi(scales[row] as i32 - 127);
        for j in 0..32 {
            let byte = bytes[row * 16 + j / 2];
            let code = if j % 2 == 0 { byte & 0xF } else { byte >> 4 };
            expected.push(e2m1(code) * scale);
        }
    }
    let diff = max_abs_diff(&got, &expected);
    assert!(
        diff == 0.0,
        "mxfp4 repack disagrees with the scalar decoder by {diff}\n got: {got:?}\n exp: {expected:?}"
    );
}

// Whole model.

#[test]
fn prefill_is_causal() {
    let config = tiny_text_config();
    let weights = sanitized(&config);
    let model = KimiK3Model::from_weights(&weights, &config).expect("tiny model loads");

    let ids: Vec<i32> = noise(96, 71)
        .iter()
        .map(|v| (((v + 0.5) * VOCAB as f32) as i32).clamp(0, VOCAB as i32 - 1))
        .collect();
    let full_ids = mlxcel_core::from_slice_i32(&ids, &[1, 96]);
    let mut caches = model.make_layer_caches();
    let full = model.forward(&full_ids, &mut caches);
    assert_eq!(shape(&full), vec![1, 96, VOCAB as i32]);
    let full_v = read(&full);
    assert!(
        full_v.iter().all(|v| v.is_finite()),
        "prefill logits must be finite"
    );
    for cache in &caches {
        assert_eq!(cache.offset(), 96);
    }

    // A 40-token prefix must produce the same first 40 rows: no position may
    // see a later token through either attention type or the residual mix.
    let prefix_ids = mlxcel_core::from_slice_i32(&ids[..40], &[1, 40]);
    let mut prefix_caches = model.make_layer_caches();
    let prefix = read(&model.forward(&prefix_ids, &mut prefix_caches));
    let diff = max_abs_diff(&prefix, &full_v[..40 * VOCAB]);
    assert!(
        diff <= 1e-3,
        "prefix logits differ from the full prefill: {diff}"
    );

    // Decoding the 41st token from the prefix caches matches row 40.
    let next = mlxcel_core::from_slice_i32(&ids[40..41], &[1, 1]);
    let step = read(&model.forward(&next, &mut prefix_caches));
    let diff = max_abs_diff(&step, &full_v[40 * VOCAB..41 * VOCAB]);
    assert!(
        diff <= 1e-3,
        "decode step differs from the prefill row: {diff}"
    );
}

#[test]
fn model_without_attention_residuals_takes_the_plain_residual_form() {
    let mut config = tiny_text_config();
    config.attn_res_block_size = None;
    let weights = sanitized(&config);
    let model = KimiK3Model::from_weights(&weights, &config).expect("tiny model loads");
    let ids = mlxcel_core::from_slice_i32(&[3, 1, 4, 1, 5], &[1, 5]);
    let mut caches = model.make_layer_caches();
    let logits = read(&model.forward(&ids, &mut caches));
    assert_eq!(logits.len(), 5 * VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// Config.

#[test]
fn config_validation_names_the_offending_field() {
    let mut c = tiny_text_config();
    c.hidden_act = "silu".to_string();
    assert!(c.validate().unwrap_err().contains("hidden_act"));

    let mut c = tiny_text_config();
    c.moe_router_activation_func = "softmax".to_string();
    assert!(
        c.validate()
            .unwrap_err()
            .contains("moe_router_activation_func")
    );

    let mut c = tiny_text_config();
    c.mla_use_nope = false;
    assert!(c.validate().unwrap_err().contains("mla_use_nope"));

    let mut c = tiny_text_config();
    c.num_expert_group = 2;
    assert!(c.validate().unwrap_err().contains("num_expert_group"));

    let mut c = tiny_text_config();
    c.topk_group = 2;
    assert!(c.validate().unwrap_err().contains("topk_group"));

    let mut c = tiny_text_config();
    c.quantization = Some(Quantization {
        group_size: 64,
        bits: 0,
    });
    assert!(c.validate().unwrap_err().contains("bits"));

    assert!(tiny_text_config().validate().is_ok());
}

/// The router calls `argpartition(kth = k - 1)` and slices `[0, k)` off the
/// result, both of which index past the score axis for a `k` of zero or one
/// above `num_experts`. Config validation is the only place that sees the two
/// numbers together.
#[test]
fn config_validation_bounds_the_router_top_k() {
    let mut c = tiny_text_config();
    c.num_experts_per_token = 0;
    assert!(c.validate().unwrap_err().contains("num_experts_per_token"));

    let mut c = tiny_text_config();
    c.num_experts_per_token = EXPERTS + 1;
    let err = c.validate().unwrap_err();
    assert!(err.contains("num_experts_per_token"), "{err}");
    assert!(err.contains("exceeds"), "{err}");

    // Selecting every expert is legal, and so is a zero `k` on a config with
    // no experts at all, where the field never reaches the router.
    let mut c = tiny_text_config();
    c.num_experts_per_token = EXPERTS;
    assert!(c.validate().is_ok());

    let mut c = tiny_text_config();
    c.num_experts = 0;
    c.num_experts_per_token = 0;
    assert!(c.validate().is_ok());

    let mut c = tiny_text_config();
    c.routed_expert_hidden_size = Some(0);
    assert!(
        c.validate()
            .unwrap_err()
            .contains("routed_expert_hidden_size")
    );
}

/// A config edited away from its checkpoint (a layer-truncated debug copy, a
/// hand-written `text_config`, the wrong checkpoint entirely) must fail at
/// load naming the tensor and the field. Without the cross-checks the
/// disagreement reaches an MLX reshape or slice in the middle of a forward
/// pass, which aborts the process with no key to name, or broadcasts and
/// returns fluent nonsense.
#[test]
fn from_weights_names_a_config_checkpoint_shape_mismatch() {
    let config = tiny_text_config();
    let weights = sanitized(&config);
    assert!(KimiK3Model::from_weights(&weights, &config).is_ok());

    let mut c = config.clone();
    c.vocab_size = VOCAB + 1;
    let err = KimiK3Model::from_weights(&weights, &c)
        .err()
        .expect("the mismatch must be refused");
    assert!(err.contains("embed_tokens"), "{err}");
    assert!(err.contains("vocab_size"), "{err}");

    let mut c = config.clone();
    c.hidden_size = HIDDEN + 1;
    let err = KimiK3Model::from_weights(&weights, &c)
        .err()
        .expect("the mismatch must be refused");
    assert!(err.contains("model.norm.weight"), "{err}");
    assert!(err.contains("hidden_size"), "{err}");

    let mut c = config.clone();
    c.linear_attn_config.num_heads = KDA_HEADS + 1;
    let err = KimiK3Model::from_weights(&weights, &c)
        .err()
        .expect("the mismatch must be refused");
    assert!(err.contains("qkv_proj"), "{err}");

    let mut c = config.clone();
    c.kv_lora_rank = KV_LORA + 1;
    let err = KimiK3Model::from_weights(&weights, &c)
        .err()
        .expect("the mismatch must be refused");
    assert!(err.contains("kv_a_proj_with_mqa"), "{err}");
    assert!(err.contains("kv_lora_rank"), "{err}");

    let mut c = config.clone();
    c.num_attention_heads = MLA_HEADS + 1;
    let err = KimiK3Model::from_weights(&weights, &c)
        .err()
        .expect("the mismatch must be refused");
    assert!(err.contains("q_b_proj"), "{err}");

    let mut c = config.clone();
    c.num_experts = EXPERTS + 1;
    let err = KimiK3Model::from_weights(&weights, &c)
        .err()
        .expect("the mismatch must be refused");
    assert!(err.contains("num_experts"), "{err}");
}

/// `stack` aborts the process when the planes it is handed disagree in shape,
/// so the sanitizer has to reject a mismatched expert set itself rather than
/// let MLX see it.
#[test]
fn sanitize_refuses_mismatched_expert_plane_shapes() {
    let config = tiny_text_config();
    let mut raw = raw_checkpoint(
        &config,
        RawLayout {
            legacy_residual_keys: false,
        },
    );
    let key = format!(
        "language_model.model.layers.1.block_sparse_moe.experts.{}.w1.weight",
        EXPERTS - 1
    );
    raw.insert(
        key,
        noise_arr(&[MOE_INTER as i32 + 1, LATENT as i32], 77, 0.5),
    );
    let err = KimiK3Model::sanitize_weights(raw, &config)
        .err()
        .expect("a mismatched expert plane must be refused");
    assert!(err.contains(&format!("experts.{}", EXPERTS - 1)), "{err}");
    assert!(err.contains("differs from"), "{err}");
}

/// `concatenate` aborts the process on planes that disagree away from the
/// concatenated axis, so the q / k / v fuse has to reject them itself rather
/// than hand them to MLX.
#[test]
fn sanitize_refuses_mismatched_fused_qkv_plane_shapes() {
    let config = tiny_text_config();

    // Layer 0 is KDA. A `k_proj` one column wider than `q_proj` is what a
    // partially converted or mis-paired checkpoint looks like.
    let mut raw = raw_checkpoint(
        &config,
        RawLayout {
            legacy_residual_keys: false,
        },
    );
    raw.insert(
        "language_model.model.layers.0.self_attn.k_proj.weight".to_string(),
        noise_arr(&[kda_p() as i32, HIDDEN as i32 + 1], 91, 0.5),
    );
    let err = KimiK3Model::sanitize_weights(raw, &config)
        .err()
        .expect("a mismatched q/k/v plane must be refused");
    assert!(err.contains("k_proj"), "{err}");
    assert!(err.contains("concatenated on axis 0"), "{err}");

    // The fused conv weights go through the same path.
    let mut raw = raw_checkpoint(
        &config,
        RawLayout {
            legacy_residual_keys: false,
        },
    );
    raw.insert(
        "language_model.model.layers.0.self_attn.v_conv1d.weight".to_string(),
        noise_arr(&[kda_p() as i32, 1, CONV_KERNEL as i32 + 1], 92, 0.5),
    );
    let err = KimiK3Model::sanitize_weights(raw, &config)
        .err()
        .expect("a mismatched conv plane must be refused");
    assert!(err.contains("v_conv1d"), "{err}");
    assert!(err.contains("concatenated on axis 0"), "{err}");
}

/// `reshape` aborts the process on an element count the requested shape cannot
/// divide, and the MLA `kv_b_proj` decomposition reshapes checkpoint data with
/// four dimensions that come from `config.json`. A tensor that disagrees with
/// them has to be refused by name here: a count the head geometry cannot
/// describe aborts inside `reshape` during sanitize, and a width that divides
/// but is not `kv_lora_rank` survives sanitize and aborts inside the absorbed
/// MLA matmul on the first forward pass instead.
#[test]
fn sanitize_refuses_a_kv_b_proj_that_disagrees_with_the_config() {
    let config = tiny_text_config();

    // Positive control first, so a guard that refused every `kv_b_proj` could
    // not pass this test.
    let out = KimiK3Model::sanitize_weights(
        raw_checkpoint(
            &config,
            RawLayout {
                legacy_residual_keys: false,
            },
        ),
        &config,
    )
    .expect("an honest kv_b_proj must still decompose");
    assert!(out.contains_key("model.layers.1.self_attn.embed_q.weight"));

    // Layer 1 is MLA. One extra row is a count the head geometry cannot
    // describe; one extra column divides but describes a different latent.
    let rows = (MLA_HEADS * (QK_NOPE + V_HEAD)) as i32;
    for (rows, cols, seed) in [
        (rows + 1, KV_LORA as i32, 95u32),
        (rows, KV_LORA as i32 + 1, 96),
    ] {
        let mut raw = raw_checkpoint(
            &config,
            RawLayout {
                legacy_residual_keys: false,
            },
        );
        raw.insert(
            "language_model.model.layers.1.self_attn.kv_b_proj.weight".to_string(),
            noise_arr(&[rows, cols], seed, 0.5),
        );
        let err = KimiK3Model::sanitize_weights(raw, &config)
            .err()
            .expect("a kv_b_proj that disagrees with the config must be refused");
        assert!(err.contains("kv_b_proj"), "{err}");
        assert!(err.contains("kv_lora_rank"), "{err}");
    }
}

/// The Attention Residuals score weight is built by hand from two tensors and
/// then multiplied against the residual stream. A width that disagrees with
/// `hidden_size` aborts the process inside MLX (in `multiply` at load when the
/// pair disagrees with each other, in `matmul` at the first forward pass when
/// they agree with each other but not with the config), so both are named at
/// load instead.
#[test]
fn attn_res_weights_are_cross_checked_against_hidden_size() {
    let config = tiny_text_config();
    let base = sanitized(&config);
    assert!(KimiK3Model::from_weights(&base, &config).is_ok());

    for key in [
        "model.layers.0.self_attention_res_norm.weight",
        "model.layers.0.mlp_res_proj.weight",
        "model.output_attn_res_proj.weight",
    ] {
        let mut weights = sanitized(&config);
        weights.insert(key.to_string(), noise_arr(&[1, HIDDEN as i32 + 1], 93, 0.5));
        let err = KimiK3Model::from_weights(&weights, &config)
            .err()
            .unwrap_or_else(|| panic!("{key} must be refused"));
        assert!(err.contains(key), "{err}");
        assert!(err.contains("hidden_size"), "{err}");
    }
}

/// The two attention kinds gate their output with a `g_proj` whose width the
/// forward pass never re-derives, and the latent MoE returns to the residual
/// width through `routed_expert_up_proj`. A wrong width reaches `reshape` (KDA
/// gate, MoE up-projection) or a broadcast `multiply` (MLA gate) inside MLX,
/// which throws, and an MLX exception crossing the cxx bridge is an
/// uncatchable abort rather than an error this loader could name. Each is
/// checked at load instead.
#[test]
fn attention_gate_and_output_widths_are_cross_checked() {
    let config = tiny_text_config();
    assert!(KimiK3Model::from_weights(&sanitized(&config), &config).is_ok());

    let d = HIDDEN as i32;
    let p = kda_p() as i32;
    let mla_out = (MLA_HEADS * V_HEAD) as i32;
    let cases: Vec<(&str, Vec<i32>, &str)> = vec![
        // Layer 0 is KDA, layer 1 is MLA, and both are followed by their own
        // output projection back to `hidden_size`.
        (
            "model.layers.0.self_attn.g_proj.weight",
            vec![p + 1, d],
            "num_heads * head_dim",
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![d + 1, p],
            "hidden_size",
        ),
        (
            "model.layers.1.self_attn.g_proj.weight",
            vec![mla_out + 1, d],
            "num_attention_heads * v_head_dim",
        ),
        (
            "model.layers.1.self_attn.o_proj.weight",
            vec![d + 1, mla_out],
            "hidden_size",
        ),
        (
            "model.layers.1.mlp.routed_expert_up_proj.weight",
            vec![d + 1, LATENT as i32],
            "hidden_size",
        ),
    ];

    for (seed, (key, shape, field)) in cases.into_iter().enumerate() {
        let mut weights = sanitized(&config);
        weights.insert(key.to_string(), noise_arr(&shape, 140 + seed as u32, 0.5));
        let err = KimiK3Model::from_weights(&weights, &config)
            .err()
            .unwrap_or_else(|| panic!("{key} must be refused"));
        assert!(err.contains(key), "{err}");
        assert!(err.contains(field), "{err}");
    }
}

/// Kimi K3 keeps its KDA and MLA state on the model and resets it whenever a
/// multi-token forward arrives without a sequence id, which is how the
/// cache-less `forward` entry point stays safe across two unrelated prompts. A
/// chunked prefill sends exactly that shape once per chunk, so the family has
/// to opt out or a prompt longer than `MLXCEL_PREFILL_CHUNK` would be answered
/// from its last chunk alone, with nothing raised.
#[test]
fn kimi_k3_opts_out_of_chunked_prefill() {
    use mlxcel_core::generate::LanguageModel;

    let config = tiny_text_config();
    let model = KimiK3Model::from_weights(&sanitized(&config), &config).expect("tiny model loads");
    assert!(
        !model.supports_chunked_prefill(),
        "a chunked prefill would discard the recurrent state built by every chunk but the last"
    );
}

/// The published `config.json`, trimmed to the fields the loader reads.
const PUBLISHED_CONFIG: &str = r#"{
    "architectures": ["KimiK3ForConditionalGeneration"],
    "bos_token_id": 163584,
    "eos_token_id": 163586,
    "media_placeholder_token_id": 163605,
    "model_type": "kimi_k3",
    "pad_token_id": 163839,
    "tie_word_embeddings": false,
    "text_config": {
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 12,
        "eos_token_id": 163586,
        "first_k_dense_replace": 1,
        "hidden_act": "situ",
        "hidden_size": 7168,
        "intermediate_size": 33792,
        "kv_lora_rank": 512,
        "latent_moe_use_norm": true,
        "linear_attn_config": {
            "full_attn_layers": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93],
            "gate_lower_bound": -5.0,
            "head_dim": 128,
            "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53, 54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89, 90, 91],
            "num_heads": 96,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": true
        },
        "max_position_embeddings": 1048576,
        "mla_use_nope": true,
        "mla_use_output_gate": true,
        "model_type": "kimi_linear",
        "moe_intermediate_size": 3072,
        "moe_layer_freq": 1,
        "moe_renormalize": true,
        "moe_router_activation_func": "sigmoid",
        "num_attention_heads": 96,
        "num_expert_group": 1,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_hidden_layers": 93,
        "num_key_value_heads": 96,
        "num_nextn_predict_layers": 0,
        "num_shared_experts": 2,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "quantization_config": {
            "format": "mxfp4-pack-quantized",
            "quant_method": "compressed-tensors"
        },
        "rms_norm_eps": 1e-05,
        "routed_expert_hidden_size": 3584,
        "routed_scaling_factor": 1.0,
        "tie_word_embeddings": false,
        "topk_group": 1,
        "use_grouped_topk": true,
        "v_head_dim": 128,
        "vocab_size": 163840
    },
    "vision_config": { "patch_size": 14, "vt_hidden_size": 1152 }
}"#;

#[test]
fn published_config_parses_and_derives_the_layer_schedule() {
    let config = KimiK3Config::from_json_str(PUBLISHED_CONFIG).expect("published config parses");
    assert_eq!(config.model_type, "kimi_k3");
    assert_eq!(config.eos_token_ids(), vec![163586]);
    assert_eq!(config.media_placeholder_token_id, Some(163605));
    let text = &config.text_config;
    assert_eq!(text.delta_projection_dim(), 12288);
    assert_eq!(text.q_head_dim(), 192);
    assert!(text.is_linear_layer(0) && text.is_linear_layer(2) && !text.is_linear_layer(3));
    assert!(!text.is_linear_layer(92));
    assert!(!text.is_moe_layer(0) && text.is_moe_layer(1) && text.is_moe_layer(92));
    let n_kda = (0..93).filter(|&i| text.is_linear_layer(i)).count();
    assert_eq!(n_kda, 69);
    assert_eq!(text.linear_attn_config.gate_lower_bound, Some(-5.0));
    assert!(text.linear_attn_config.use_full_rank_gate);
    assert_eq!(text.situ_beta(), 4.0);
    assert_eq!(text.activation_situ_linear_beta, Some(25.0));
    assert!((text.qk_norm_eps() - 1e-6 / 128.0).abs() < 1e-12);
    assert!(
        text.quantization.is_none(),
        "compressed-tensors is not an MLX quantization block"
    );
    assert_eq!(text.group_size(), 64);
    assert_eq!(text.bits(), 4);

    // Rejections come through the same entry point.
    let rotated = PUBLISHED_CONFIG.replace("\"mla_use_nope\": true", "\"mla_use_nope\": false");
    let err = KimiK3Config::from_json_str(&rotated).unwrap_err();
    assert!(err.contains("mla_use_nope"), "{err}");
}

#[test]
fn eos_token_ids_fall_back_to_the_text_config() {
    let config = KimiK3Config::from_json_str(&PUBLISHED_CONFIG.replacen(
        "    \"eos_token_id\": 163586,\n",
        "",
        1,
    ))
    .expect("parses without a top-level eos");
    assert!(config.eos_token_id.is_none());
    assert_eq!(config.eos_token_ids(), vec![163586]);
}

#[test]
fn kimi_k3_sequence_state_isolates_prepared_sequences() {
    use crate::models::model_owned::ModelOwnedSequenceState;
    use mlxcel_core::cache::SequenceId;
    let make = || {
        vec![
            super::KimiK3LayerCache::Delta(KimiK3DeltaCache::new()),
            super::KimiK3LayerCache::Attn(KVCache::new()),
        ]
    };
    let state = ModelOwnedSequenceState::new(make());
    let seq = SequenceId::from_raw(1334);
    state.prepare_sequence_state(seq, make());
    state
        .with_existing_sequence_state(seq, |caches| match &mut caches[0] {
            super::KimiK3LayerCache::Delta(c) => {
                c.conv = Some(mlxcel_core::zeros(&[1, 3, 48], dtype::FLOAT32));
                c.advance(4);
            }
            super::KimiK3LayerCache::Attn(_) => panic!("expected a Delta cache"),
        })
        .expect("prepared sequence exists");
    state
        .with_existing_sequence_state(seq, |caches| {
            assert_eq!(caches[0].offset(), 4);
            match &caches[0] {
                super::KimiK3LayerCache::Delta(c) => assert!(c.conv.is_some()),
                super::KimiK3LayerCache::Attn(_) => panic!("expected a Delta cache"),
            }
        })
        .expect("prepared sequence still exists");
    assert!(
        state
            .with_existing_sequence_state(SequenceId::from_raw(1), |_| {})
            .is_err()
    );
}
