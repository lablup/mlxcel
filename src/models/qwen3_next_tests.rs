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

//! Regression test for the bound on declared quantization params that the
//! Qwen3-Next `SwitchLinear` loader applies before it stores them on a
//! quantized expert plane (issue #958).

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::utils::create_causal_mask;
use mlxcel_core::weights::WeightMap;

use super::{Qwen3NextCache, Qwen3NextConfig, Qwen3NextModel, SwitchLinear};
use crate::models::switch_layers::{HOSTILE_QUANT_PARAMS, insert_stacked_quantized_expert_plane};

/// Honest 4-bit expert geometry: `packed_in * 32 == bits * num_groups *
/// group_size` (8 * 32 == 4 * 1 * 64), so the positive control below is a plane
/// MLX can actually describe.
const EXPERTS: i32 = 3;
const OUT: i32 = 4;
const PACKED_IN: i32 = 8;
const NUM_GROUPS: i32 = 1;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

const PREFIX: &str = "model.layers.0.mlp.switch_mlp.gate_proj";

/// Smallest Qwen3-Next config that parses, varying only the declared
/// quantization pair. Qwen3-Next carries it in a nested `quantization` block.
/// Built through serde rather than a struct literal so every
/// `#[serde(default)]` field fills itself in.
fn config_with(group_size: i32, bits: i32) -> Qwen3NextConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "qwen3_next",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "num_experts": 3,
        "num_experts_per_tok": 1,
        "decoder_sparse_step": 1,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "vocab_size": 32,
        "quantization": { "group_size": group_size, "bits": bits },
    }))
    .expect("test config must parse")
}

fn recurrent_config() -> Qwen3NextConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "qwen3_next",
        "hidden_size": 8,
        "vocab_size": 32,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "num_experts": 0,
        "num_experts_per_tok": 1,
        "decoder_sparse_step": 1,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "tie_word_embeddings": true,
    }))
    .expect("test config must parse")
}

fn deterministic_values(seed: u32, count: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            let value = (unit * 2.0 - 1.0) * scale;
            if value.abs() < 1e-5 {
                scale / 17.0
            } else {
                value
            }
        })
        .collect()
}

fn recurrent_weights() -> WeightMap {
    let mut weights = WeightMap::new();
    let mut seed = 2285u32;
    let mut tensor = |name: &str, shape: &[i32], scale: f32| {
        seed += 1;
        let count = shape.iter().product::<i32>() as usize;
        weights.insert(
            name.to_string(),
            mlxcel_core::from_slice_f32(&deterministic_values(seed, count, scale), shape),
        );
    };

    tensor("model.embed_tokens.weight", &[32, 8], 0.2);
    tensor("model.layers.0.linear_attn.conv1d.weight", &[16, 4, 1], 0.2);
    tensor(
        "model.layers.0.linear_attn.in_proj_qkvz.weight",
        &[24, 8],
        0.2,
    );
    tensor("model.layers.0.linear_attn.in_proj_ba.weight", &[4, 8], 0.2);
    tensor("model.layers.0.linear_attn.dt_bias", &[2], 0.1);
    tensor("model.layers.0.linear_attn.A_log", &[2], 0.1);
    tensor("model.layers.0.linear_attn.norm.weight", &[4], 0.2);
    tensor("model.layers.0.linear_attn.out_proj.weight", &[8, 8], 0.2);
    tensor("model.layers.0.mlp.gate_proj.weight", &[8, 8], 0.2);
    tensor("model.layers.0.mlp.up_proj.weight", &[8, 8], 0.2);
    tensor("model.layers.0.mlp.down_proj.weight", &[8, 8], 0.2);
    tensor("model.layers.0.input_layernorm.weight", &[8], 0.2);
    tensor("model.layers.0.post_attention_layernorm.weight", &[8], 0.2);
    tensor("model.norm.weight", &[8], 0.2);

    weights
}

fn tokens(values: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_i32(values, &[1, values.len() as i32])
}

fn linear_offset(caches: &[Qwen3NextCache]) -> i32 {
    match caches.first().expect("one-layer fixture has one cache") {
        Qwen3NextCache::Linear(cache) => cache.offset,
        Qwen3NextCache::Attention(_) => panic!("layer 0 must be GatedDeltaNet"),
    }
}

/// The pair this loader stores is handed straight to `gather_qmm`, which
/// crosses the cxx bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++
/// throw there is an uncatchable `std::terminate`,
/// so losing the bound turns a rejected load into an uncatchable abort at the
/// first routed forward pass in production. This test asserts on the load
/// result rather than running a forward pass, so a regression fails cleanly
/// here instead of aborting the test binary.
///
/// Three config sources reach this one loader (Qwen3-Next, Qwen3.5 through
/// `to_qwen3next_config`, and the synthesized bridge config in the Qwen3-Omni
/// talker), which is why the bound lives at the storage point rather than at
/// any single producer.
#[test]
fn qwen3_next_switch_linear_rejects_quantization_params_that_would_abort_gather_qmm() {
    let mut weights = WeightMap::new();
    insert_stacked_quantized_expert_plane(
        &mut weights,
        PREFIX,
        EXPERTS,
        OUT,
        PACKED_IN,
        NUM_GROUPS,
    );

    // Positive control first, so a guard that rejected every quantized plane
    // could not pass this test.
    SwitchLinear::from_weights(&weights, &config_with(GROUP_SIZE, BITS), PREFIX)
        .expect("honest 4-bit expert plane must load");

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let err = match SwitchLinear::from_weights(&weights, &config_with(group_size, bits), PREFIX)
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
    SwitchLinear::from_weights(&regular, &config_with(0, 0), PREFIX)
        .expect("a bf16 expert plane must not be gated on quantization params");
}

/// The chunked-prefill contract after issue #2463. A no-ID multi-token forward
/// continues the model-owned GatedDelta/KV state (matching the Qwen3.5 model),
/// so `supports_chunked_prefill` is the trait default and two consecutive
/// no-ID forwards build on each other; a caller that starts a new prompt
/// resets through `reset_runtime_state`. Stable sequence IDs keep their
/// per-sequence isolation either way.
#[test]
fn qwen3_next_loaded_model_chunking_contract_is_safe() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = Qwen3NextModel::from_weights(&recurrent_weights(), &recurrent_config())
        .expect("one-layer GatedDeltaNet model must load");
    let language_model: &dyn LanguageModel = &model;
    assert!(
        language_model.supports_chunked_prefill(),
        "no-ID multi-token forwards continue the model-owned state, so no-ID \
         prompt chunks can retain the state built by earlier chunks"
    );

    let mut compatibility_caches = LanguageModel::make_caches(&model);
    let first_prompt =
        LanguageModel::forward(&model, &tokens(&[1, 2]), &mut compatibility_caches, None);
    mlxcel_core::eval(&first_prompt);
    let second_prompt =
        LanguageModel::forward(&model, &tokens(&[3, 4, 5]), &mut compatibility_caches, None);
    mlxcel_core::eval(&second_prompt);
    assert_eq!(
        model
            .sequence_state
            .with_sequence_state(None, |caches| linear_offset(caches)),
        5,
        "a second no-ID multi-token forward must continue the model-owned state"
    );

    // A new prompt resets through `reset_runtime_state`.
    model.reset_runtime_state();
    assert_eq!(
        model
            .sequence_state
            .with_sequence_state(None, |caches| linear_offset(caches)),
        0,
        "reset_runtime_state must start a new prompt from fresh model-owned state"
    );

    let seq_id = SequenceId::from_raw(2285);
    LanguageModel::prepare_sequence_state(&model, seq_id);
    let first_chunk = LanguageModel::forward_with_sequence_id(
        &model,
        &tokens(&[6, 7]),
        Some(seq_id),
        &mut compatibility_caches,
        None,
    );
    mlxcel_core::eval(&first_chunk);
    let second_chunk = LanguageModel::forward_with_sequence_id(
        &model,
        &tokens(&[8, 9, 10]),
        Some(seq_id),
        &mut compatibility_caches,
        None,
    );
    mlxcel_core::eval(&second_chunk);
    assert_eq!(
        model
            .sequence_state
            .with_sequence_state_ref(seq_id, linear_offset),
        Some(5),
        "stable sequence IDs must retain state across prefill chunks"
    );
}

/// Eight-layer Qwen3-Next config that exercises both layer kinds:
/// `full_attention_interval` 4 types layers 3 and 7 as full attention and the
/// rest as linear (GatedDeltaNet), with a dense MLP everywhere (`num_experts`
/// 0) and tied embeddings so the logits come from the embedding table.
fn hybrid_config() -> Qwen3NextConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "qwen3_next",
        "hidden_size": 16,
        "num_hidden_layers": 8,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "decoder_sparse_step": 1,
        "moe_intermediate_size": 0,
        "shared_expert_intermediate_size": 0,
        "vocab_size": 64,
        "tie_word_embeddings": true,
    }))
    .expect("hybrid qwen3-next config must parse")
}

/// Deterministic f32 weights for the eight-layer hybrid fixture. The conv1d
/// plane is stored post-swap `[conv_dim, kernel, 1]` (last dim 1, kept as-is
/// by the loader) and `in_proj_qkvz` is `[2*nk*dk + 2*nv*dv, hidden]` (q, k,
/// v, z per k-head group), matching the Qwen3-Next projection layout.
fn hybrid_weights() -> WeightMap {
    let mut weights = WeightMap::new();
    let mut seed = 7401u32;
    let mut tensor = |name: &str, shape: &[i32], scale: f32| {
        seed += 1;
        let count = shape.iter().product::<i32>() as usize;
        weights.insert(
            name.to_string(),
            mlxcel_core::from_slice_f32(&deterministic_values(seed, count, scale), shape),
        );
    };

    tensor("model.embed_tokens.weight", &[64, 16], 0.2);
    for l in 0..8 {
        let p = format!("model.layers.{l}");
        tensor(&format!("{p}.input_layernorm.weight"), &[16], 0.2);
        tensor(&format!("{p}.post_attention_layernorm.weight"), &[16], 0.2);
        if (l + 1) % 4 == 0 {
            // Full-attention layer: q carries a gate (2 * heads * head_dim).
            tensor(&format!("{p}.self_attn.q_proj.weight"), &[32, 16], 0.2);
            tensor(&format!("{p}.self_attn.k_proj.weight"), &[8, 16], 0.2);
            tensor(&format!("{p}.self_attn.v_proj.weight"), &[8, 16], 0.2);
            tensor(&format!("{p}.self_attn.o_proj.weight"), &[16, 16], 0.2);
            tensor(&format!("{p}.self_attn.q_norm.weight"), &[4], 0.2);
            tensor(&format!("{p}.self_attn.k_norm.weight"), &[4], 0.2);
        } else {
            // Linear GatedDeltaNet layer: key_dim 8, value_dim 16, conv_dim 32.
            tensor(&format!("{p}.linear_attn.conv1d.weight"), &[32, 4, 1], 0.2);
            tensor(
                &format!("{p}.linear_attn.in_proj_qkvz.weight"),
                &[48, 16],
                0.2,
            );
            tensor(&format!("{p}.linear_attn.in_proj_ba.weight"), &[8, 16], 0.2);
            tensor(&format!("{p}.linear_attn.dt_bias"), &[4], 0.1);
            tensor(&format!("{p}.linear_attn.A_log"), &[4], 0.1);
            tensor(&format!("{p}.linear_attn.norm.weight"), &[4], 0.2);
            tensor(&format!("{p}.linear_attn.out_proj.weight"), &[16, 16], 0.2);
        }
        // Dense MLP.
        tensor(&format!("{p}.mlp.gate_proj.weight"), &[32, 16], 0.2);
        tensor(&format!("{p}.mlp.up_proj.weight"), &[32, 16], 0.2);
        tensor(&format!("{p}.mlp.down_proj.weight"), &[16, 32], 0.2);
    }
    tensor("model.norm.weight", &[16], 0.2);

    weights
}

/// Read an evaluated array as f32, whatever it was stored as.
fn logits_values(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Logit tolerance for one T = 5 block versus five T = 1 steps behind the same
/// cache state. Both arms run f32 weights and float32 recurrent state; the
/// residual is reduction-order drift through the chunked gated-delta scan, the
/// conv, and masked-versus-unmasked SDPA across eight layers.
const BLOCK_PARITY_TOLERANCE: f32 = 2e-3;

/// Assert one T = 5 block's logits match five T = 1 steps' logits row by row,
/// within [`BLOCK_PARITY_TOLERANCE`]. `block` is `[1, 5, vocab]` and `steps`
/// holds one `[1, 1, vocab]` logits tensor per position.
fn assert_block_matches_steps(
    block: &mlxcel_core::MlxArray,
    steps: &[mlxcel_core::UniquePtr<mlxcel_core::MlxArray>],
) {
    let vocab = mlxcel_core::array_shape(block)[2];
    let block_all = logits_values(block);
    for (i, step) in steps.iter().enumerate() {
        let step_row = logits_values(step);
        let block_row = &block_all[i * vocab as usize..(i + 1) * vocab as usize];
        let max_diff = block_row
            .iter()
            .zip(step_row.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= BLOCK_PARITY_TOLERANCE,
            "position {i}: one T=5 forward diverged from the T=1 step by {max_diff}, \
             tolerance {BLOCK_PARITY_TOLERANCE}"
        );
    }
}

/// Issue #2463 acceptance test: a multi-token forward behind a populated cache
/// must read the state its predecessors built, exactly as the same tokens
/// forwarded one at a time behind it do. This is the shape speculative-decode
/// verification uses (a few tokens forwarded behind a cache), so a multi-token
/// forward that discards or misreads the cache rejects almost everything.
///
/// Both arms prefill the same 12-token context through the shared
/// no-sequence-id forward, then compare one T = 5 forward against five T = 1
/// forwards. The comparison goes through the `LanguageModel::forward` entry
/// the offline CLI and the logit-trace harness drive.
#[test]
fn qwen3_next_multi_token_forward_behind_cache_matches_stepwise() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let weights = hybrid_weights();
    let config = hybrid_config();
    let model_block = Qwen3NextModel::from_weights(&weights, &config)
        .expect("eight-layer hybrid model must load");
    let model_step = Qwen3NextModel::from_weights(&weights, &config)
        .expect("eight-layer hybrid model must load");

    let context: Vec<i32> = (1..=12).collect();
    let continuation: Vec<i32> = (13..=17).collect();

    // Prefill the same context on both instances so both arms start from an
    // identical populated cache.
    for model in [&model_block, &model_step] {
        let warm = LanguageModel::forward(model, &tokens(&context), &mut Vec::new(), None);
        mlxcel_core::eval(&warm);
    }

    // Arm A: one T = 5 forward behind the populated cache.
    let block = LanguageModel::forward(&model_block, &tokens(&continuation), &mut Vec::new(), None);
    mlxcel_core::eval(&block);

    // Arm B: the same five tokens forwarded one at a time from the same state.
    let mut steps = Vec::with_capacity(continuation.len());
    for t in continuation {
        let step = LanguageModel::forward(&model_step, &tokens(&[t]), &mut Vec::new(), None);
        mlxcel_core::eval(&step);
        steps.push(step);
    }

    assert_block_matches_steps(&block, &steps);
}

/// Pins the T > 1 numerical paths the shared forward reaches behind a
/// non-empty cache: the chunked gated-delta scan, the conv1d state carry, and
/// the causal mask / RoPE offset on the full-attention layers. Unlike the
/// acceptance test above, this drives the inherent
/// `Qwen3NextModel::forward` with an explicit causal mask and per-arm caches,
/// so the comparison isolates the arithmetic from the state lifecycle.
#[test]
fn qwen3_next_multi_token_state_update_behind_cache_matches_stepwise() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let weights = hybrid_weights();
    let config = hybrid_config();
    let model_block = Qwen3NextModel::from_weights(&weights, &config)
        .expect("eight-layer hybrid model must load");
    let model_step = Qwen3NextModel::from_weights(&weights, &config)
        .expect("eight-layer hybrid model must load");

    let context: Vec<i32> = (1..=12).collect();
    let continuation: Vec<i32> = (13..=17).collect();

    let mut caches_block = model_block.make_caches();
    let mut caches_step = model_step.make_caches();

    // Prefill both arms from an empty cache with the plain causal mask.
    let context_mask = create_causal_mask(context.len() as i32, 0);
    let warm_block = model_block.forward(&tokens(&context), &mut caches_block, Some(&context_mask));
    mlxcel_core::eval(&warm_block);
    let warm_step = model_step.forward(&tokens(&context), &mut caches_step, Some(&context_mask));
    mlxcel_core::eval(&warm_step);

    // Arm A: one T = 5 forward behind the populated cache, masked from the
    // cache offset the way `forward_for_sequence` builds it.
    let block_mask = create_causal_mask(continuation.len() as i32, context.len() as i32);
    let block = model_block.forward(&tokens(&continuation), &mut caches_block, Some(&block_mask));
    mlxcel_core::eval(&block);

    // Arm B: the same five tokens forwarded one at a time (mask None, T = 1).
    let mut steps = Vec::with_capacity(continuation.len());
    for t in continuation {
        let step = model_step.forward(&tokens(&[t]), &mut caches_step, None);
        mlxcel_core::eval(&step);
        steps.push(step);
    }

    assert_block_matches_steps(&block, &steps);
}

#[test]
fn issue_2572_qwen3_next_native_batched_decode() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let make = || {
        Qwen3NextModel::from_weights(&hybrid_weights(), &hybrid_config())
            .expect("eight-layer hybrid model must load")
    };
    crate::models::batched_decode_harness::assert_model_owned_multistep_with_paged_lengths(
        make,
        &[0, 0, 0, 3, 0, 0, 0, 3],
    );
    crate::models::batched_decode_harness::assert_model_owned_fresh_and_mixed(make);
}
