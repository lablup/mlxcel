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

//! Inbound activation width of a [`Gemma4StageModel`] (issue #2716).
//!
//! A remote stage refuses hidden states whose last axis is not the width it
//! expects. Per-layer-input checkpoints (E2B / E4B) append the per-layer
//! inputs of every remaining layer to the stage-boundary activation, so the
//! expected width must be the one `pack_hidden_for_downstream` emits
//! upstream, or every honest frame past the entry stage is rejected.

use super::gemma4_stage_tests::{HIDDEN, LAYERS, VOCAB, args, filled, stage, tensor, weights};
use super::{Cache, Gemma4StageModel, ModelArgs};
use crate::distributed::pipeline::StageExecutionOutput;
use crate::distributed::pipeline::wire_tensor::{
    StageWireLimits, deserialize_wire_tensor_with_limits, serialize_mlx_array,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const PER_LAYER: i32 = 16;

fn per_layer_args() -> ModelArgs {
    let mut args = args();
    args.text_config["hidden_size_per_layer_input"] = PER_LAYER.into();
    args.text_config["vocab_size_per_layer_input"] = VOCAB.into();
    args
}

fn per_layer_weights() -> WeightMap {
    let prefix = "language_model.model";
    let mut w = weights();
    let all_layers = LAYERS as i32 * PER_LAYER;
    w.insert(
        format!("{prefix}.embed_tokens_per_layer.weight"),
        tensor(&[VOCAB, all_layers], 7),
    );
    w.insert(
        format!("{prefix}.per_layer_model_projection.weight"),
        tensor(&[all_layers, HIDDEN], 8),
    );
    w.insert(
        format!("{prefix}.per_layer_projection_norm.weight"),
        filled(&[PER_LAYER], 1.0),
    );
    for layer in 0..LAYERS {
        let base = format!("{prefix}.layers.{layer}");
        let seed = 500 + layer as u64 * 10;
        w.insert(
            format!("{base}.per_layer_input_gate.weight"),
            tensor(&[PER_LAYER, HIDDEN], seed),
        );
        w.insert(
            format!("{base}.per_layer_projection.weight"),
            tensor(&[HIDDEN, PER_LAYER], seed + 1),
        );
        w.insert(
            format!("{base}.post_per_layer_input_norm.weight"),
            filled(&[HIDDEN], 1.0),
        );
    }
    w
}

fn hidden_out(output: Result<StageExecutionOutput, String>) -> UniquePtr<MlxArray> {
    match output.expect("stage must run") {
        StageExecutionOutput::HiddenStates(hidden) => hidden,
        StageExecutionOutput::Logits(_) => panic!("a non-final stage must hand hidden states on"),
    }
}

/// Decode `hidden` as a remote stage with `next`'s limits would.
fn decode_for(next: &Gemma4StageModel, hidden: &MlxArray) -> UniquePtr<MlxArray> {
    let limits = StageWireLimits {
        activation_width: Some(next.inbound_activation_width()),
        vocab_size: None,
    };
    let wire = serialize_mlx_array(hidden).expect("serialize");
    deserialize_wire_tensor_with_limits(&wire, &limits.hidden_states())
        .expect("an honest stage-boundary frame must pass the next stage's limits")
}

fn last_axis(arr: &MlxArray) -> u64 {
    *mlxcel_core::array_shape(arr).last().expect("rank >= 1") as u64
}

/// Three stages over four per-layer-input layers: each non-entry stage
/// expects exactly the width the stage before it emits, and the frame passes
/// that stage's wire limits.
#[test]
fn per_layer_input_inbound_width_matches_upstream_packing() {
    let args = per_layer_args();
    let weights = per_layer_weights();
    let stage0 = stage(&weights, &args, 0..1);
    let stage1 = stage(&weights, &args, 1..3);
    let stage2 = stage(&weights, &args, 3..LAYERS);
    let hidden = HIDDEN as u64;
    let per_layer = PER_LAYER as u64;
    assert_eq!(stage1.inbound_activation_width(), hidden + 3 * per_layer);
    assert_eq!(stage2.inbound_activation_width(), hidden + per_layer);

    let ids = mlxcel_core::from_slice_i32(&[3, 17, 42, 5], &[1, 4]);
    let mut caches0: Vec<Cache> = stage0.make_caches();
    let mut caches1: Vec<Cache> = stage1.make_caches();
    let mut caches2: Vec<Cache> = stage2.make_caches();

    let packed0 = hidden_out(stage0.execute_from_token_ids(&ids, &mut caches0));
    assert_eq!(last_axis(&packed0), stage1.inbound_activation_width());
    let packed0 = decode_for(&stage1, &packed0);

    let packed1 = hidden_out(stage1.execute_from_hidden_states(packed0, &mut caches1));
    assert_eq!(last_axis(&packed1), stage2.inbound_activation_width());
    let packed1 = decode_for(&stage2, &packed1);

    match stage2
        .execute_from_hidden_states(packed1, &mut caches2)
        .expect("last stage must run")
    {
        StageExecutionOutput::Logits(logits) => {
            assert_eq!(last_axis(&logits), VOCAB as u64);
        }
        StageExecutionOutput::HiddenStates(_) => panic!("the last stage must produce logits"),
    }
}

/// Without per-layer inputs the boundary activation is plain hidden states.
#[test]
fn inbound_width_without_per_layer_inputs_is_hidden_size() {
    let stage1 = stage(&weights(), &args(), 1..LAYERS);
    assert_eq!(stage1.inbound_activation_width(), HIDDEN as u64);
}
