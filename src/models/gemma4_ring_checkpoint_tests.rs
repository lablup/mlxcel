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

//! Gemma 4 ring-only state checkpoints decode like a cold run (issue #3255).
//!
//! The fixture has four layers (sliding, full, and a KV-shared sliding and
//! full pair) with `sliding_window = 8`, so a 13-token history has wrapped
//! every ring and the donor of 18 tokens cannot be cut back on its own.

use crate::models::gemma4::Gemma4Wrapper;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};

const FIRST: [i32; 10] = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3];
const SECOND: [i32; 3] = [4, 6, 2];
const DONOR_TAIL: [i32; 5] = [5, 1, 7, 3, 6];
const BOUNDARY: usize = FIRST.len() + SECOND.len();
const DECODE: i32 = 5;

/// The scalar fields `snapshot_rotating` writes besides the buffers.
const RING_SCALARS: [&str; 10] = [
    "max_size",
    "buffer_size",
    "offset",
    "ring_origin_offset",
    "ring_origin_cursor",
    "start_position",
    "idx",
    "step",
    "mode",
    "turbo_seed",
];

fn forward(wrapper: &Gemma4Wrapper, seq: SequenceId, ids: &[i32]) -> Vec<f32> {
    let input = mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32]);
    let logits = wrapper.forward_with_sequence_id(&input, Some(seq), &mut [], None);
    let logits = mlxcel_core::astype(logits.as_ref().unwrap(), mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&logits);
    mlxcel_core::array_to_raw_bytes(&logits)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The history, forwarded as the cold run of the same prompt partitions it:
/// the first piece, then the piece that ends at the history boundary.
fn prefill_history(wrapper: &Gemma4Wrapper, seq: SequenceId) {
    wrapper.prepare_sequence_state(seq);
    forward(wrapper, seq, &FIRST);
    forward(wrapper, seq, &SECOND);
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The checkpoint's scalars over the donor's ring buffers: right metadata,
/// rows from other tokens.
fn with_donor_rings(
    checkpoint: &ModelStateSnapshot,
    donor: &ModelStateSnapshot,
    layers: usize,
) -> ModelStateSnapshot {
    let mut forged = ModelStateSnapshot::new(checkpoint.family(), checkpoint.token_len());
    for idx in 0..layers {
        let ring = format!("layer{idx}.rotating");
        for buffer in ["keys", "values"] {
            if let Some(array) = donor.tensor(&format!("{ring}.{buffer}")) {
                forged.push_tensor(format!("{ring}.{buffer}"), array);
            }
        }
        for scalar in RING_SCALARS {
            if let Some(array) = checkpoint.tensor(&format!("{ring}.{scalar}")) {
                forged.push_tensor(format!("{ring}.{scalar}"), array);
            }
        }
    }
    forged
}

#[test]
#[ignore = "requires serial MLX execution"]
fn a_ring_checkpoint_over_a_longer_donor_decodes_like_a_cold_run() {
    let wrapper = crate::models::gemma4_tests::build_synthetic_kv_shared_wrapper();
    let layers = wrapper.num_layers();

    // The conversation: history to the boundary, its snapshot and checkpoint,
    // then a later turn whose snapshot is the donor.
    let source = SequenceId::from_raw(3255);
    prefill_history(&wrapper, source);
    let boundary = wrapper
        .snapshot_sequence_state(source, BOUNDARY)
        .expect("boundary snapshot");
    assert!(
        !wrapper.snapshot_truncatable_to(&boundary, BOUNDARY - 1),
        "the fixture must wrap its rings before the boundary"
    );
    let checkpoint = wrapper
        .state_checkpoint_from_snapshot(&boundary)
        .expect("a wrapped ring yields a checkpoint");
    assert!(checkpoint.nbytes() < boundary.nbytes());
    assert!(
        checkpoint.tensor("layer1.standard.keys").is_none(),
        "the full-attention layer comes from the donor"
    );
    drop(boundary);
    forward(&wrapper, source, &DONOR_TAIL);
    let donor = wrapper
        .snapshot_sequence_state(source, BOUNDARY + DONOR_TAIL.len())
        .expect("donor snapshot");
    assert!(
        !wrapper.snapshot_truncatable_to(&donor, BOUNDARY),
        "without the checkpoint the donor cannot serve the boundary"
    );

    let cold = SequenceId::from_raw(3256);
    prefill_history(&wrapper, cold);
    let want = forward(&wrapper, cold, &[DECODE]);

    let restored = SequenceId::from_raw(3257);
    wrapper.prepare_sequence_state(restored);
    wrapper
        .restore_sequence_state_from_checkpoint(restored, &checkpoint, &donor)
        .expect("checkpoint plus donor restores");
    let got = forward(&wrapper, restored, &[DECODE]);
    // Same pieces from the same state: bit for bit, not within a tolerance.
    assert_eq!(
        got, want,
        "a restored checkpoint must decode like the cold run of the same partition"
    );

    // Negative control: the same metadata over the donor's rings must not
    // pass, or the comparison above could not tell the rings apart.
    let forged = with_donor_rings(&checkpoint, &donor, layers);
    let wrong = SequenceId::from_raw(3258);
    wrapper.prepare_sequence_state(wrong);
    wrapper
        .restore_sequence_state_from_checkpoint(wrong, &forged, &donor)
        .expect("the forged ring has a valid layout");
    let bad = forward(&wrapper, wrong, &[DECODE]);
    // The fixture's weights are tiny, so the logits move by about 1e-4; the
    // positive case above is exact, so any difference separates the two.
    let diff = max_abs_diff(&bad, &want);
    assert!(
        diff > 1e-5,
        "restoring the donor's ring must change the logits: max |diff| = {diff}"
    );

    // A checkpoint is refused over a donor of another family, and a whole
    // snapshot's own family check still applies.
    let other = ModelStateSnapshot::new("gemma3", BOUNDARY + DONOR_TAIL.len());
    assert!(
        wrapper
            .restore_sequence_state_from_checkpoint(SequenceId::from_raw(3259), &checkpoint, &other)
            .is_err()
    );
}
