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

//! Split-boundary parity for [`Gemma4StageModel`] (issue #2654).
//!
//! A two-stage pipeline of the same weights must produce the logits the
//! single-process [`Gemma4Model`] produces, bit for bit. The stage model used
//! to build explicit causal and sliding masks for every prefill while the
//! dense text forward leaves both `None` inside the window, so the two ran
//! different attention kernels and the remote pipeline parity tests failed on
//! real checkpoints. The fixture is bf16 with a head dim the fused SDPA
//! kernels serve, alternating sliding and full attention, and the split puts
//! stage 1 on a full-attention layer.

use std::collections::HashMap;

use super::{Cache, Gemma4Model, Gemma4StageModel, ModelArgs};
use crate::distributed::pipeline::{LayerFilter, StageExecutionOutput};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const HIDDEN: i32 = 256;
const VOCAB: i32 = 64;
const INTERMEDIATE: i32 = 512;
const HEAD_DIM: i32 = 256;
const HEADS: i32 = 8;
const KV_HEADS: i32 = 1;
const LAYERS: usize = 4;
const WINDOW: i32 = 16;
/// The activation dtype the real checkpoint runs in on CUDA.
const DTYPE: i32 = mlxcel_core::dtype::BFLOAT16;

fn tensor(shape: &[i32], seed: u64) -> UniquePtr<MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let data: Vec<f32> = (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            unit * 0.2 - 0.1
        })
        .collect();
    let f32_arr = mlxcel_core::from_slice_f32(&data, shape);
    mlxcel_core::astype(&f32_arr, DTYPE)
}

fn filled(shape: &[i32], value: f32) -> UniquePtr<MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    let f32_arr = mlxcel_core::from_slice_f32(&vec![value; len], shape);
    mlxcel_core::astype(&f32_arr, DTYPE)
}

fn layer_types() -> Vec<&'static str> {
    (0..LAYERS)
        .map(|layer| {
            if layer % 2 == 1 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect()
}

fn args() -> ModelArgs {
    let rope = serde_json::json!({
        "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0,
        "rope_type": "default"
    });
    ModelArgs {
        model_type: "gemma4".to_string(),
        text_config: serde_json::json!({
            "model_type": "gemma4_text",
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "intermediate_size": INTERMEDIATE,
            "num_attention_heads": HEADS,
            "head_dim": HEAD_DIM,
            "rms_norm_eps": 1e-6,
            "vocab_size": VOCAB,
            "vocab_size_per_layer_input": 0,
            "num_key_value_heads": KV_HEADS,
            "num_global_key_value_heads": null,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "rope_traditional": false,
            "rope_parameters": {
                "sliding_attention": rope.clone(),
                "full_attention": rope
            },
            "sliding_window": WINDOW,
            "sliding_window_pattern": 2,
            "max_position_embeddings": 4096,
            "attention_k_eq_v": false,
            "final_logit_softcapping": 30.0,
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "num_experts": null,
            "top_k_experts": null,
            "moe_intermediate_size": null,
            "layer_types": layer_types(),
            "quantization": null
        }),
        eos_token_id: Some(serde_json::json!([1])),
        quantization: None,
        quantization_config: None,
    }
}

fn weights() -> WeightMap {
    let prefix = "language_model.model";
    let mut w: WeightMap = HashMap::new();
    w.insert(
        format!("{prefix}.embed_tokens.weight"),
        tensor(&[VOCAB, HIDDEN], 1),
    );
    w.insert(format!("{prefix}.norm.weight"), filled(&[HIDDEN], 1.0));
    for layer in 0..LAYERS {
        let seed = 100 + layer as u64 * 10;
        let attn = format!("{prefix}.layers.{layer}.self_attn");
        for (name, shape, offset) in [
            ("q_proj", [HEADS * HEAD_DIM, HIDDEN], 0),
            ("k_proj", [KV_HEADS * HEAD_DIM, HIDDEN], 1),
            ("v_proj", [KV_HEADS * HEAD_DIM, HIDDEN], 2),
            ("o_proj", [HIDDEN, HEADS * HEAD_DIM], 3),
        ] {
            w.insert(
                format!("{attn}.{name}.weight"),
                tensor(&shape, seed + offset),
            );
        }
        w.insert(format!("{attn}.q_norm.weight"), filled(&[HEAD_DIM], 1.0));
        w.insert(format!("{attn}.k_norm.weight"), filled(&[HEAD_DIM], 1.0));

        let mlp = format!("{prefix}.layers.{layer}.mlp");
        for (name, shape, offset) in [
            ("gate_proj", [INTERMEDIATE, HIDDEN], 4),
            ("up_proj", [INTERMEDIATE, HIDDEN], 5),
            ("down_proj", [HIDDEN, INTERMEDIATE], 6),
        ] {
            w.insert(
                format!("{mlp}.{name}.weight"),
                tensor(&shape, seed + offset),
            );
        }
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            w.insert(
                format!("{prefix}.layers.{layer}.{norm}.weight"),
                filled(&[HIDDEN], 1.0),
            );
        }
        w.insert(
            format!("{prefix}.layers.{layer}.layer_scalar"),
            filled(&[1], 1.0),
        );
    }
    w
}

fn stage(weights: &WeightMap, args: &ModelArgs, range: std::ops::Range<usize>) -> Gemma4StageModel {
    let filter = LayerFilter {
        has_embedding: range.start == 0,
        has_lm_head: range.end == LAYERS,
        layer_range: range,
    };
    Gemma4StageModel::from_filtered_weights(weights, args, &filter, 0, Default::default())
        .expect("fixture stage must load")
}

fn dense_logits(dense: &Gemma4Model, ids: &MlxArray, caches: &mut [Cache]) -> UniquePtr<MlxArray> {
    dense.forward_with_caches_and_embeddings(ids, None, caches, None, None)
}

fn two_stage_logits(
    stage0: &Gemma4StageModel,
    stage1: &Gemma4StageModel,
    ids: &MlxArray,
    caches0: &mut [Cache],
    caches1: &mut [Cache],
) -> UniquePtr<MlxArray> {
    let StageExecutionOutput::HiddenStates(hidden) = stage0
        .execute_from_token_ids(ids, caches0)
        .expect("stage 0 must run")
    else {
        panic!("stage 0 must hand hidden states downstream");
    };
    let StageExecutionOutput::Logits(logits) = stage1
        .execute_from_hidden_states(hidden, caches1)
        .expect("stage 1 must run")
    else {
        panic!("the last stage must produce logits");
    };
    logits
}

fn assert_bitwise_equal(got: &MlxArray, want: &MlxArray, what: &str) {
    assert_eq!(
        mlxcel_core::array_dtype(got),
        mlxcel_core::array_dtype(want),
        "{what}: dtype"
    );
    assert_eq!(
        mlxcel_core::array_shape(got),
        mlxcel_core::array_shape(want),
        "{what}: shape"
    );
    let got_bytes = mlxcel_core::array_to_raw_bytes(got);
    let want_bytes = mlxcel_core::array_to_raw_bytes(want);
    if got_bytes != want_bytes {
        let diff = mlxcel_core::abs(&mlxcel_core::subtract(
            &mlxcel_core::astype(got, mlxcel_core::dtype::FLOAT32),
            &mlxcel_core::astype(want, mlxcel_core::dtype::FLOAT32),
        ));
        let max = mlxcel_core::item_f32(&mlxcel_core::max_all(&diff));
        panic!("{what}: stage logits differ from the dense logits (max abs diff {max:e})");
    }
}

/// Prefills inside the window, each followed by one decode step, split
/// before a full-attention layer.
#[test]
fn two_stage_split_matches_dense_prefill_and_decode_bitwise() {
    let args = args();
    let weights = weights();
    let dense = Gemma4Model::from_weights(&weights, &args).expect("fixture model must load");
    let stage0 = stage(&weights, &args, 0..1);
    let stage1 = stage(&weights, &args, 1..LAYERS);

    // The two-token prompt is the shape of the real-model parity tests and
    // the one where the masked and maskless SDPA kernels disagree.
    for ids in [&[3, 17][..], &[3, 17, 42, 5, 9, 60, 33, 1][..]] {
        let prompt = mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32]);
        let decode = mlxcel_core::from_slice_i32(&[11], &[1, 1]);

        let mut dense_caches = dense.make_caches();
        let dense_prefill = dense_logits(&dense, &prompt, &mut dense_caches);
        let dense_decode = dense_logits(&dense, &decode, &mut dense_caches);

        let mut caches0 = stage0.make_caches();
        let mut caches1 = stage1.make_caches();
        let stage_prefill = two_stage_logits(&stage0, &stage1, &prompt, &mut caches0, &mut caches1);
        let stage_decode = two_stage_logits(&stage0, &stage1, &decode, &mut caches0, &mut caches1);

        let len = ids.len();
        assert_bitwise_equal(
            &stage_prefill,
            &dense_prefill,
            &format!("{len}-token prefill"),
        );
        assert_bitwise_equal(&stage_decode, &dense_decode, &format!("decode after {len}"));
    }
}

/// A prefill longer than the sliding window takes the over-window branch of
/// the shared mask decision in both models.
#[test]
fn two_stage_split_matches_dense_over_window_prefill_bitwise() {
    let args = args();
    let weights = weights();
    let dense = Gemma4Model::from_weights(&weights, &args).expect("fixture model must load");
    let stage0 = stage(&weights, &args, 0..1);
    let stage1 = stage(&weights, &args, 1..LAYERS);

    let ids: Vec<i32> = (0..(WINDOW + 8)).map(|i| (i * 7 + 3) % VOCAB).collect();
    let prompt = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);

    let mut dense_caches = dense.make_caches();
    let dense_prefill = dense_logits(&dense, &prompt, &mut dense_caches);

    let mut caches0 = stage0.make_caches();
    let mut caches1 = stage1.make_caches();
    let stage_prefill = two_stage_logits(&stage0, &stage1, &prompt, &mut caches0, &mut caches1);

    assert_bitwise_equal(&stage_prefill, &dense_prefill, "over-window prefill");
}

/// A second multi-token prefill on a non-empty cache (chunked prefill). The
/// second chunk starts at a nonzero offset, with and without crossing the
/// sliding window, and must match the dense forward bit for bit.
#[test]
fn two_stage_split_matches_dense_continuation_prefill_bitwise() {
    let args = args();
    let weights = weights();
    let dense = Gemma4Model::from_weights(&weights, &args).expect("fixture model must load");
    let stage0 = stage(&weights, &args, 0..1);
    let stage1 = stage(&weights, &args, 1..LAYERS);

    // (first chunk, second chunk): the second stays inside the window, then
    // ends past it.
    for (first, second) in [(4usize, 3usize), (12, 8)] {
        let ids: Vec<i32> = (0..(first + second) as i32)
            .map(|i| (i * 7 + 3) % VOCAB)
            .collect();
        let chunk_a = mlxcel_core::from_slice_i32(&ids[..first], &[1, first as i32]);
        let chunk_b = mlxcel_core::from_slice_i32(&ids[first..], &[1, second as i32]);
        let decode = mlxcel_core::from_slice_i32(&[11], &[1, 1]);

        let mut dense_caches = dense.make_caches();
        dense_logits(&dense, &chunk_a, &mut dense_caches);
        let dense_b = dense_logits(&dense, &chunk_b, &mut dense_caches);
        let dense_d = dense_logits(&dense, &decode, &mut dense_caches);

        let mut caches0 = stage0.make_caches();
        let mut caches1 = stage1.make_caches();
        two_stage_logits(&stage0, &stage1, &chunk_a, &mut caches0, &mut caches1);
        let stage_b = two_stage_logits(&stage0, &stage1, &chunk_b, &mut caches0, &mut caches1);
        let stage_d = two_stage_logits(&stage0, &stage1, &decode, &mut caches0, &mut caches1);

        assert_bitwise_equal(
            &stage_b,
            &dense_b,
            &format!("{first}+{second} continuation prefill"),
        );
        assert_bitwise_equal(
            &stage_d,
            &dense_d,
            &format!("decode after {first}+{second}"),
        );
    }
}
