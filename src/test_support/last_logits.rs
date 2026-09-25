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

//! Contract check for `LanguageModel::forward_last_logits*` overrides that
//! slice the hidden state before the LM head (#1968 and its follow-up).

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::utils::array_to_vec_f32;

/// Asserts that `forward_last_logits`, `forward_last_logits_with_sequence_id`
/// and (when `with_embeddings`) `forward_last_logits_with_embeddings_and_sequence_id`
/// return the full forward's row at the last and at an interior position,
/// shaped `[1, 1, vocab]`, within `tol`.
pub(crate) fn assert_last_logits_match_full_forward<M: LanguageModel>(
    model: &M,
    seq: &[i32],
    with_embeddings: bool,
    tol: f32,
) {
    let ids = mlxcel_core::from_slice_i32(seq, &[1, seq.len() as i32]);
    let mut caches = model.make_caches();
    let full = model.forward(&ids, &mut caches, None);
    let vocab = mlxcel_core::array_shape(&full)[2];

    let max_diff = |a: &mlxcel_core::MlxArray, b: &mlxcel_core::MlxArray| {
        let (a, b) = (array_to_vec_f32(a), array_to_vec_f32(b));
        assert_eq!(a.len(), b.len(), "shape mismatch");
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    for last_pos in [seq.len() - 1, seq.len() / 2] {
        let p = last_pos as i32;
        let expected = mlxcel_core::slice(&full, &[0, p, 0], &[1, p + 1, vocab]);

        let mut c = model.make_caches();
        let mut got = vec![(
            "plain",
            model.forward_last_logits(&ids, &mut c, None, last_pos),
        )];
        let mut c = model.make_caches();
        got.push((
            "sequence_id",
            model.forward_last_logits_with_sequence_id(
                &ids,
                Some(SequenceId::from_raw(11 + last_pos as u64)),
                &mut c,
                None,
                last_pos,
            ),
        ));
        if with_embeddings {
            let embeds = model.embed_tokens(&ids).expect("model exposes embeddings");
            let mut c = model.make_caches();
            got.push((
                "embeddings",
                model.forward_last_logits_with_embeddings_and_sequence_id(
                    &ids,
                    Some(&embeds),
                    None,
                    &mut c,
                    None,
                    last_pos,
                ),
            ));
        }
        for (label, logits) in got {
            assert_eq!(
                mlxcel_core::array_shape(&logits),
                vec![1, 1, vocab],
                "{label} shape at {last_pos}"
            );
            let gap = max_diff(&expected, &logits);
            assert!(gap < tol, "{label} at {last_pos}: max |diff| {gap}");
        }
    }
}
