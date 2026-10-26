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

//! Native batched-decode runtime contracts for the six issue #2563 towers.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::embedding_test_support::mlx_test_guard;
use crate::models::qwen_vl_batched_decode_test_support::RowPositionState;
use crate::models::vlm_family_batched_decode_test_support::{
    tiny_cohere_compass_full_mrope_model, tiny_cohere_compass_model,
    tiny_cohere_compass_sliding_nope_model, tiny_ernie45_moe_vl_model, tiny_glm4v_model,
    tiny_glm4v_moe_model, tiny_hunyuan_vl_model, tiny_paddleocr_model,
};

const HIDDEN: i32 = 16;

fn state_values(
    axes: i32,
    state: RowPositionState,
    prompt_len: usize,
) -> Option<(UniquePtr<MlxArray>, i32)> {
    match state {
        RowPositionState::Text => None,
        RowPositionState::Delta(delta) => Some((
            mlxcel_core::from_slice_i32(&vec![0; axes as usize], &[axes, 1, 1]),
            delta,
        )),
        RowPositionState::Stored { delta } => {
            let len = prompt_len + 1;
            let mut values = Vec::with_capacity(axes as usize * len);
            for axis in 0..axes {
                values.extend((0..len).map(|position| position as i32 + axis * 13));
            }
            Some((
                mlxcel_core::from_slice_i32(&values, &[axes, 1, len as i32]),
                delta,
            ))
        }
    }
}

fn bind_state(
    seq_id: SequenceId,
    state: RowPositionState,
    prompt_len: usize,
    axes: i32,
    mut set_fallback: impl FnMut(UniquePtr<MlxArray>, i32),
    mut bind: impl FnMut(SequenceId),
) {
    if let Some((positions, delta)) = state_values(axes, state, prompt_len) {
        set_fallback(positions, delta);
    }
    bind(seq_id);
}

macro_rules! native_contract {
    ($test:ident, $fixture:ident, $axes:expr) => {
        #[test]
        fn $test() {
            let _guard = mlx_test_guard();
            let model = $fixture();
            crate::models::qwen_vl_batched_decode_test_support::assert_native_qwen_vl_contract(
                &model,
                |id, state, prompt_len| {
                    bind_state(
                        id,
                        state,
                        prompt_len,
                        $axes,
                        |positions, delta| model.set_mrope_state(positions, delta),
                        |seq_id| model.bind_mrope_state_to_sequence(seq_id),
                    )
                },
            );
        }
    };
}

native_contract!(glm4v_native_batched_decode_contract, tiny_glm4v_model, 3);
native_contract!(
    glm4v_moe_native_batched_decode_contract,
    tiny_glm4v_moe_model,
    3
);
native_contract!(
    hunyuan_vl_native_batched_decode_contract,
    tiny_hunyuan_vl_model,
    4
);
native_contract!(
    ernie45_moe_vl_native_batched_decode_contract,
    tiny_ernie45_moe_vl_model,
    3
);
native_contract!(
    paddleocr_vl_native_batched_decode_contract,
    tiny_paddleocr_model,
    3
);
native_contract!(
    cohere_compass_native_batched_decode_contract,
    tiny_cohere_compass_model,
    3
);
native_contract!(
    cohere_compass_sliding_nope_native_batched_decode_contract,
    tiny_cohere_compass_sliding_nope_model,
    3
);
native_contract!(
    cohere_compass_full_mrope_native_batched_decode_contract,
    tiny_cohere_compass_full_mrope_model,
    3
);

/// Fallback-slot DeepStack state (CLI, unbound rows) must not change decode
/// numerics: injection happens at offset 0 only.
#[test]
fn cohere_compass_fallback_deepstack_state_keeps_exact_per_row_decode() {
    let _guard = mlx_test_guard();
    let model = tiny_cohere_compass_model();
    crate::models::qwen_vl_batched_decode_test_support::assert_decode_matches_rows_with_global_state(
        &model,
        || {
            model.set_deepstack_state(
                mlxcel_core::from_slice_i32(&[0], &[1, 1]),
                vec![mlxcel_core::zeros(
                    &[1, 1, HIDDEN],
                    mlxcel_core::dtype::FLOAT32,
                )],
            );
        },
    );
}

/// Two image requests admitted in one drain tick: each sequence's prefill
/// injects its own DeepStack entry (#2567).
#[test]
fn cohere_compass_bound_deepstack_entries_stay_with_their_sequences() {
    let _guard = mlx_test_guard();
    let model = tiny_cohere_compass_model();
    crate::models::deepstack_state_test_support::assert_bound_deepstack_entries_stay_with_their_sequences(
        &model,
        HIDDEN,
        2,
        crate::models::deepstack_state_test_support::DeepStackHooks {
            set_state: &|masks, embeds| model.set_deepstack_state(masks, embeds),
            bind: &|seq_id| model.bind_mrope_state_to_sequence(seq_id),
            embed: &|ids| model.get_embed_tokens(ids),
        },
    );
}

/// A text-only offset-0 prefill on another sequence must not clear a bound
/// DeepStack entry (#2567).
#[test]
fn cohere_compass_text_only_forward_keeps_bound_deepstack_entry() {
    let _guard = mlx_test_guard();
    let model = tiny_cohere_compass_model();
    crate::models::deepstack_state_test_support::assert_text_only_forward_keeps_bound_deepstack_entry(
        &model,
        HIDDEN,
        2,
        crate::models::deepstack_state_test_support::DeepStackHooks {
            set_state: &|masks, embeds| model.set_deepstack_state(masks, embeds),
            bind: &|seq_id| model.bind_mrope_state_to_sequence(seq_id),
            embed: &|ids| model.get_embed_tokens(ids),
        },
    );
}
