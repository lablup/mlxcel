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

//! Shared check for the per-family `execute_last_row` overrides (#3081).
//!
//! Each family builds one executor over a synthetic checkpoint and hands
//! [`assert_last_row_matches_full`] a builder from a [`LayerFilter`] to its
//! executor. The check runs the full forward, then every sampled position
//! through `execute_last_row`, and compares the row with the full forward's;
//! a non-final stage must return its whole hidden state regardless of
//! `last_pos`.

use mlxcel_core::MlxArray;

use super::{FamilyStageExecutor, LayerFilter, StageExecutionInput, StageExecutionOutput};

/// Prompt length of the check; long enough to give three distinct positions.
const SEQ_LEN: i32 = 6;

fn close(got: &MlxArray, want: &MlxArray, tol: f32, what: &str) {
    assert_eq!(
        mlxcel_core::array_shape(got),
        mlxcel_core::array_shape(want),
        "{what}: shape"
    );
    let got = mlxcel_core::astype(got, mlxcel_core::dtype::FLOAT32);
    let want = mlxcel_core::astype(want, mlxcel_core::dtype::FLOAT32);
    assert!(
        mlxcel_core::item_bool(&mlxcel_core::allclose(
            &got,
            &want,
            f64::from(tol),
            f64::from(tol)
        )),
        "{what}: values differ beyond {tol}"
    );
}

fn run(
    exec: &dyn FamilyStageExecutor,
    ids: &MlxArray,
    last_pos: Option<usize>,
) -> StageExecutionOutput {
    let mut caches = exec.make_caches();
    let out = exec
        .execute_last_row(
            StageExecutionInput::TokenIds(ids),
            &mut caches,
            None,
            last_pos,
        )
        .expect("stage must run");
    mlxcel_core::eval(match &out {
        StageExecutionOutput::HiddenStates(a) | StageExecutionOutput::Logits(a) => a,
    });
    exec.release_caches(&caches);
    out
}

/// `build` makes an executor for a filter over the family's `layers` layers
/// and `vocab`-wide head; `tol` is the row comparison tolerance.
pub(super) fn assert_last_row_matches_full(
    build: impl Fn(&LayerFilter) -> Box<dyn FamilyStageExecutor>,
    layers: usize,
    vocab: i32,
    tol: f32,
) {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let tokens: Vec<i32> = (0..SEQ_LEN).map(|i| (i * 5 + 3) % vocab).collect();
    let ids = mlxcel_core::from_slice_i32(&tokens, &[1, SEQ_LEN]);
    let ids = ids.as_ref().unwrap();

    let final_stage = build(&LayerFilter {
        layer_range: 0..layers,
        has_embedding: true,
        has_lm_head: true,
    });
    let full = run(final_stage.as_ref(), ids, None)
        .into_logits()
        .expect("final stage must produce logits");
    assert_eq!(
        mlxcel_core::array_shape(&full)[..2],
        [1, SEQ_LEN],
        "`None` keeps every position"
    );
    for pos in [0, SEQ_LEN as usize / 2, SEQ_LEN as usize - 1] {
        let row = run(final_stage.as_ref(), ids, Some(pos))
            .into_logits()
            .expect("final stage must produce logits");
        assert_eq!(mlxcel_core::array_shape(&row)[..2], [1, 1], "row {pos}");
        let want = mlxcel_core::generate::logits_at_position(&full, pos);
        close(&row, &want, tol, &format!("row {pos}"));
    }

    let first_stage = build(&LayerFilter {
        layer_range: 0..layers,
        has_embedding: true,
        has_lm_head: false,
    });
    let whole = run(first_stage.as_ref(), ids, None)
        .into_hidden_states()
        .expect("non-final stage must produce hidden states");
    let ignored = run(first_stage.as_ref(), ids, Some(SEQ_LEN as usize - 1))
        .into_hidden_states()
        .expect("non-final stage must produce hidden states");
    close(&ignored, &whole, tol, "non-final stage ignores last_pos");
}
