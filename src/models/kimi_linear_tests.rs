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

//! Regression tests for the two quantization bounds on KimiLinear's MLA path:
//! the bound on declared quantization params that
//! `kimi_linear::MultiLinear::from_weights` applies before it stores them on a
//! quantized per-head projection (issue #958), and the rejection of a
//! `kv_b_proj` plane that carries `.scales` with no `.biases` (issue #1026).

use super::{
    KimiDeltaCache, KimiLinearCache, KimiLinearConfig, KimiLinearModel, LinearAttnConfig,
    MultiLinear,
};
use crate::models::model_owned::ModelOwnedSequenceState;
use crate::models::switch_layers::{HOSTILE_QUANT_PARAMS, insert_stacked_quantized_expert_plane};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::weights::WeightMap;

/// Honest 4-bit geometry: `packed_in * 32 == bits * num_groups * group_size`
/// (8 * 32 == 4 * 1 * 64), so the positive control below is a projection MLX can
/// actually describe. `MultiLinear` is a per-head projection rather than an
/// expert plane, but its tensors carry the same `[heads, out, packed_in]` /
/// `[heads, out, num_groups]` shapes the shared fixture helper builds.
const HEADS: i32 = 3;
const OUT: i32 = 4;
const PACKED_IN: i32 = 8;
const NUM_GROUPS: i32 = 1;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

const PREFIX: &str = "model.layers.0.self_attn.embed_q";

/// The pair this loader stores is handed straight to `quantized_matmul`, which
/// crosses the cxx bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++
/// throw there is an uncatchable `std::terminate`,
/// so losing the bound turns a rejected load into an uncatchable abort at the
/// first routed forward pass in production. This test asserts on the load
/// result rather than running a forward pass, so a regression fails cleanly
/// here instead of aborting the test binary.
#[test]
fn kimi_linear_multi_linear_rejects_quantization_params_that_would_abort_quantized_matmul() {
    let mut weights = WeightMap::new();
    insert_stacked_quantized_expert_plane(&mut weights, PREFIX, HEADS, OUT, PACKED_IN, NUM_GROUPS);

    // Positive control first, so a guard that rejected every quantized
    // projection could not pass this test.
    match MultiLinear::from_weights(&weights, PREFIX, GROUP_SIZE, BITS) {
        Ok(_) => {}
        Err(e) => panic!("honest 4-bit per-head projection must load: {e}"),
    }

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let err = match MultiLinear::from_weights(&weights, PREFIX, group_size, bits) {
            Ok(_) => panic!(
                "(group_size {group_size}, bits {bits}) must be refused at load, \
                 not stored for quantized_matmul"
            ),
            Err(e) => e,
        };
        assert!(
            err.contains(field),
            "(group_size {group_size}, bits {bits}) must be blamed on {field}, got: {err}"
        );
        assert!(
            err.contains(PREFIX),
            "the load error must name the offending tensor {PREFIX}, got: {err}"
        );
    }

    // A bf16 projection carries no packing and no `.scales`, so the declared
    // pair is inert on the dense `matmul` path and must not gate the load.
    let mut dense = WeightMap::new();
    let n = (HEADS * OUT * PACKED_IN) as usize;
    dense.insert(
        format!("{PREFIX}.weight"),
        mlxcel_core::from_slice_f32(&vec![0.0f32; n], &[HEADS, OUT, PACKED_IN]),
    );
    match MultiLinear::from_weights(&dense, PREFIX, 0, 0) {
        Ok(_) => {}
        Err(e) => panic!("a non-quantized per-head projection must load with an unset pair: {e}"),
    }
}

/// A minimal config where every layer is an MLA layer (`kda_layers` empty, so
/// `is_linear_layer` is always false) and MoE stacking is inert (`num_experts`
/// 0), so `sanitize_weights` only exercises the `kv_b_proj` decomposition this
/// test targets.
fn mla_config(kv_lora_rank: usize) -> KimiLinearConfig {
    KimiLinearConfig {
        model_type: "kimi_linear".to_string(),
        vocab_size: 32,
        hidden_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        intermediate_size: 64,
        head_dim: 8,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        linear_attn_config: LinearAttnConfig {
            kda_layers: vec![],
            num_heads: 4,
            head_dim: 8,
            short_conv_kernel_size: 4,
        },
        num_experts: 0,
        moe_intermediate_size: 0,
        kv_lora_rank,
        tie_word_embeddings: true,
        qk_nope_head_dim: Some(4),
        qk_rope_head_dim: Some(4),
        v_head_dim: Some(4),
        mla_use_nope: false,
        num_experts_per_token: 1,
        num_shared_experts: 0,
        moe_router_activation_func: "sigmoid".to_string(),
        moe_renormalize: true,
        routed_scaling_factor: 1.0,
        first_k_dense_replace: 0,
        moe_layer_freq: 1,
        use_grouped_topk: true,
        num_expert_group: 1,
        topk_group: 1,
        quantization: None,
        eos_token_id: Some(serde_json::json!([50, 51])),
    }
}

/// An affine 4-bit `kv_b_proj` for every layer in `config`.
///
/// Honest geometry at `mla_config(16)`: packed_in * 32 == bits * num_groups *
/// group_size (2 * 32 == 4 * 1 * 16), with num_heads 4 and qk_nope_head_dim =
/// v_head_dim = 4 so rows = 4 * 8 = 32.
///
/// The packed plane is UINT32, not a float dtype: `dequantize` rejects any
/// other packed dtype by throwing, and that throw is the uncatchable abort
/// these guards exist to keep out of the forward path. A float fixture here
/// takes the test binary down on the positive control.
fn affine_kv_b_proj_weights(config: &KimiLinearConfig) -> WeightMap {
    let rows = (config.num_attention_heads * (config.qk_nope() + config.v_head())) as i32;
    let mut weights = WeightMap::new();
    for layer_idx in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer_idx}.self_attn.kv_b_proj");
        weights.insert(
            format!("{prefix}.weight"),
            mlxcel_core::zeros(&[rows, 2], mlxcel_core::dtype::UINT32),
        );
        weights.insert(
            format!("{prefix}.scales"),
            mlxcel_core::zeros(&[rows, 1], mlxcel_core::dtype::FLOAT32),
        );
        weights.insert(
            format!("{prefix}.biases"),
            mlxcel_core::zeros(&[rows, 1], mlxcel_core::dtype::FLOAT32),
        );
    }
    weights
}

/// The same plane an mxfp4 / nvfp4 / mxfp8 export ships: `.scales` present,
/// `.biases` absent, because the block-float modes carry no zero points.
fn block_float_kv_b_proj_weights(config: &KimiLinearConfig) -> WeightMap {
    let mut weights = affine_kv_b_proj_weights(config);
    for layer_idx in 0..config.num_hidden_layers {
        weights.remove(&format!(
            "model.layers.{layer_idx}.self_attn.kv_b_proj.biases"
        ));
    }
    weights
}

/// The MLA `kv_b_proj` pair `KimiLinearModel::sanitize_weights` decomposes is
/// solved from `kv_lora_rank` and two tensor axes rather than declared, and
/// every input is untrusted: `kv_lora_rank` comes from `config.json` and the
/// axes from the checkpoint. Before issue #958 the naive arithmetic divided by
/// both without checking them, so a `kv_lora_rank` of 0 panicked on integer
/// division and a solved pair outside anything MLX can describe reached
/// `dequantize`, which crosses the cxx bridge as `UniquePtr<MlxArray>` rather
/// than `Result` and therefore aborts during weight sanitization rather than
/// failing the load.
///
/// This drives the real `KimiLinearModel::sanitize_weights` rather than the
/// shared helper directly, so the guard is exercised where a checkpoint
/// reaches it. The assertions are on the returned `Result`, so a regression
/// fails cleanly here instead of aborting the test binary.
#[test]
fn kimi_linear_sanitize_rejects_a_kv_lora_rank_no_packing_can_describe() {
    // Positive control first, so a guard that rejected every quantized
    // kv_b_proj could not pass this test.
    let honest = mla_config(16);
    let sanitized =
        match KimiLinearModel::sanitize_weights(affine_kv_b_proj_weights(&honest), &honest) {
            Ok(w) => w,
            Err(e) => panic!("an honest quantized kv_b_proj must still sanitize: {e}"),
        };
    assert!(sanitized.contains_key("model.layers.0.self_attn.embed_q.weight"));

    // A zero `kv_lora_rank` is Rust integer division by zero on the very
    // first solve, so it has to be refused before the division rather than
    // after.
    let zero_rank = mla_config(0);
    let err = KimiLinearModel::sanitize_weights(affine_kv_b_proj_weights(&zero_rank), &zero_rank)
        .err()
        .unwrap_or_else(|| panic!("kv_lora_rank 0 must be refused, not divided by"));
    assert!(err.contains("kv_lora_rank"), "unhelpful error: {err}");

    // A large `kv_lora_rank` truncates the solved bit width to 0, which is
    // the divisor MLX would then divide by.
    let wide_rank = mla_config(4096);
    let err = KimiLinearModel::sanitize_weights(affine_kv_b_proj_weights(&wide_rank), &wide_rank)
        .err()
        .unwrap_or_else(|| panic!("a solved bit width of 0 must be refused"));
    assert!(err.contains("bits"), "unhelpful error: {err}");

    // A non-quantized kv_b_proj carries no packing, so the pair is never
    // solved and a `kv_lora_rank` that no packing could describe must not gate
    // it. The tensor is built at `wide_rank`'s own width so it still satisfies
    // the separate shape cross-check below the solve, which would otherwise
    // reject this for an unrelated reason.
    let mut float_only = WeightMap::new();
    let rows = (wide_rank.num_attention_heads * (wide_rank.qk_nope() + wide_rank.v_head())) as i32;
    float_only.insert(
        "model.layers.0.self_attn.kv_b_proj.weight".to_string(),
        mlxcel_core::zeros(
            &[rows, wide_rank.kv_lora_rank as i32],
            mlxcel_core::dtype::FLOAT32,
        ),
    );
    match KimiLinearModel::sanitize_weights(float_only, &wide_rank) {
        Ok(_) => {}
        Err(e) => panic!("a float kv_b_proj must not be gated on quantization params: {e}"),
    }
}

/// Issue #1026: `sanitize_weights` decides `kv_b_proj` is quantized on
/// `.scales` alone, then used to take `.biases` with `.unwrap()`.
///
/// Affine stores zero points; the block-float modes (mxfp4 / nvfp4 / mxfp8) do
/// not, which is exactly what `infer_quantization_mode` keys on. A block-float
/// `kv_b_proj` therefore satisfies the `.scales` gate and arrives at the
/// `.biases` removal with nothing to take, and the `.unwrap()` made that a
/// panic during weight sanitization. In the server that takes the process down
/// rather than rejecting one model load, which is strictly worse than the load
/// error it should be.
///
/// The decomposition below still dequantizes as `"affine"`, so a genuine
/// block-float `kv_b_proj` is not supported either way: what this pins is that
/// it is refused by name instead of unwinding. The assertion is on the returned
/// `Result` and no forward pass runs, so a regression fails cleanly here.
#[test]
fn kimi_linear_sanitize_rejects_a_kv_b_proj_with_scales_and_no_biases() {
    let config = mla_config(16);

    // Positive control first, so a check that rejected every quantized
    // kv_b_proj could not pass this test.
    let sanitized = KimiLinearModel::sanitize_weights(affine_kv_b_proj_weights(&config), &config)
        .expect("a kv_b_proj carrying both planes must still sanitize");
    assert!(sanitized.contains_key("model.layers.0.self_attn.embed_q.weight"));

    let err = KimiLinearModel::sanitize_weights(block_float_kv_b_proj_weights(&config), &config)
        .err()
        .expect("scales with no biases must be refused at load, not unwrapped");
    assert!(
        err.contains("model.layers.0.self_attn.kv_b_proj.biases"),
        "the error must name the key that is missing, got: {err}"
    );
    assert!(
        err.contains("scales but no biases"),
        "the wording must match the other four MLA sanitizers, got: {err}"
    );
}

/// Issue #1026: `MultiLinear::forward` used to pass a hardcoded `"affine"` to
/// `quantized_matmul` while `from_weights` accepted a `.biases`-less plane, so
/// a block-float projection would have reached MLX's `validate_mode_with_type`
/// as affine-with-null-biases and thrown `Biases must be provided for affine
/// quantization`. `quantized_matmul` crosses the cxx bridge as
/// `UniquePtr<MlxArray>` rather than `Result`, so that throw is an uncatchable
/// `std::terminate` at the first MLA forward.
///
/// This branch is reachable today, not merely defended against a future
/// sanitizer change: `sanitize_weights` only dequantizes `kv_b_proj` and
/// inserts `embed_q.weight` / `unembed_out.weight` dense inside its
/// `if weights.contains_key(&kv_b_key)` guard, so a checkpoint that ships no
/// `kv_b_proj.weight` at all skips that block outright and leaves its own
/// pre-decomposed, quantized `embed_q` / `unembed_out` planes untouched;
/// `MlaAttention::from_weights` then hands those planes straight to this
/// loader with `is_quantized == true`. This test drives the loader directly
/// rather than through a full checkpoint fixture, since that is the
/// cheapest way to pin the stored mode against every plane combination
/// without constructing that checkpoint shape end to end.
#[test]
fn kimi_linear_multi_linear_infers_its_quantization_mode_from_the_biases_plane() {
    let load = |group_size: i32, bits: i32, with_biases: bool| {
        let mut weights = WeightMap::new();
        insert_stacked_quantized_expert_plane(
            &mut weights,
            PREFIX,
            HEADS,
            OUT,
            PACKED_IN,
            NUM_GROUPS,
        );
        if !with_biases {
            weights.remove(&format!("{PREFIX}.biases"));
        }
        MultiLinear::from_weights(&weights, PREFIX, group_size, bits)
            .unwrap_or_else(|e| panic!("({group_size}, {bits}, {with_biases}) must load: {e}"))
    };

    // Zero points present is the only signal for affine, and it outranks the
    // pair: a `.biases`-carrying plane stays affine at every group_size / bits.
    assert_eq!(load(GROUP_SIZE, BITS, true).mode, "affine");
    assert_eq!(load(16, 4, true).mode, "affine");
    assert_eq!(load(64, 8, true).mode, "affine");

    // With no zero points the pair picks which block-float mode it is, on the
    // same rule the shared `UnifiedLinear` / `UnifiedEmbedding` loaders use.
    assert_eq!(load(GROUP_SIZE, BITS, false).mode, "mxfp4");
    assert_eq!(load(16, 4, false).mode, "nvfp4");
    assert_eq!(load(64, 8, false).mode, "mxfp8");

    // A dense projection never reaches `quantized_matmul`, so the mode is inert
    // there and must not be resolved from a pair the checkpoint does not carry.
    let mut dense = WeightMap::new();
    let n = (HEADS * OUT * PACKED_IN) as usize;
    dense.insert(
        format!("{PREFIX}.weight"),
        mlxcel_core::from_slice_f32(&vec![0.0f32; n], &[HEADS, OUT, PACKED_IN]),
    );
    let loaded = MultiLinear::from_weights(&dense, PREFIX, 0, 0)
        .expect("a non-quantized per-head projection must load with an unset pair");
    assert!(!loaded.is_quantized);
    assert_eq!(loaded.mode, "affine");
}

fn mixed_kimi_caches() -> Vec<KimiLinearCache> {
    vec![
        KimiLinearCache::Delta(KimiDeltaCache::new()),
        KimiLinearCache::MLA(mlxcel_core::layers::KVCache::new()),
    ]
}

#[test]
fn kimi_linear_eos_token_ids_come_from_config_metadata() {
    let config = mla_config(16);

    assert_eq!(
        crate::models::parse_optional_eos_token_ids(&config.eos_token_id),
        vec![50, 51]
    );
}

#[test]
fn kimi_linear_model_owned_sequence_state_preserves_greedy_decode_cache() {
    let state = ModelOwnedSequenceState::new(mixed_kimi_caches());
    let seq = SequenceId::from_raw(1220);

    state.prepare_sequence_state(seq, mixed_kimi_caches());
    state
        .with_existing_sequence_state(seq, |caches| match &mut caches[0] {
            KimiLinearCache::Delta(cache) => {
                cache.q_conv_state =
                    Some(mlxcel_core::zeros(&[1, 3, 8], mlxcel_core::dtype::FLOAT32));
                cache.advance(1);
            }
            KimiLinearCache::MLA(_) => panic!("expected Delta cache"),
        })
        .expect("prepared KimiLinear sequence must exist");

    state
        .with_existing_sequence_state(seq, |caches| match &mut caches[0] {
            KimiLinearCache::Delta(cache) => {
                assert!(
                    cache.q_conv_state.is_some(),
                    "decode must see the cache populated by the previous token"
                );
                assert_eq!(cache.offset, 1);
                cache.ssm_state = Some(mlxcel_core::zeros(
                    &[1, 2, 4, 8],
                    mlxcel_core::dtype::FLOAT32,
                ));
                cache.advance(1);
            }
            KimiLinearCache::MLA(_) => panic!("expected Delta cache"),
        })
        .expect("prepared KimiLinear sequence must still exist");

    state
        .with_existing_sequence_state(seq, |caches| match &caches[0] {
            KimiLinearCache::Delta(cache) => {
                assert!(
                    cache.ssm_state.is_some(),
                    "cache writes must survive across consecutive greedy steps"
                );
                assert_eq!(cache.offset, 2);
            }
            KimiLinearCache::MLA(_) => panic!("expected Delta cache"),
        })
        .expect("prepared KimiLinear sequence must still exist");
}

#[test]
fn kimi_linear_model_owned_sequence_state_refuses_missing_sequence_id() {
    let state = ModelOwnedSequenceState::new(mixed_kimi_caches());
    let seq = SequenceId::from_raw(1220);

    let err = state
        .with_existing_sequence_state(seq, |_| {})
        .expect_err("unprepared KimiLinear sequence must fail closed");

    assert!(err.contains("missing model-owned sequence state"));
}
