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

//! Gemma 3 on window-bounded paged sliding storage (issue #3256).
//!
//! The two-layer fixture of `snapshot_prompt_cache` with a 32-token window
//! (layer 0 slides, layer 1 is global) on 16-token blocks. The paged layers
//! must attend the same keys the ring does, keep the live table within
//! `ceil(window / block) + 1` blocks, and serve a batched decode step like
//! each row alone.

use super::super::Gemma3Wrapper;
use super::snapshot_prompt_cache::{bare_wrapper, ids, to_vec_f32};
use crate::models::pooled_model::Pooled;
use mlxcel_core::cache::{LayerState, SequenceId};
use mlxcel_core::generate::LanguageModel;

const WINDOW: i32 = 32;
const BLOCK: usize = 16;
const SEQ_BASE: u64 = 3_256_000;

fn paged() -> Pooled<Gemma3Wrapper> {
    Pooled::with_paged_blocks(bare_wrapper(WINDOW), BLOCK)
}

fn ring() -> Pooled<Gemma3Wrapper> {
    Pooled::new(bare_wrapper(WINDOW).without_paged_sliding())
}

fn forward(model: &Pooled<Gemma3Wrapper>, seq: SequenceId, tokens: &[i32]) -> Vec<f32> {
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    let logits = model.forward_with_sequence_id(&input, Some(seq), &mut [], None);
    to_vec_f32(&logits)
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: logit count");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let abs = (g - w).abs();
        assert!(
            abs < 2e-3 || abs / w.abs().max(1.0) < 2e-3,
            "{what}: logit[{i}] paged={g}, ring={w}"
        );
    }
}

/// Live and retired blocks of each paged layer of `seq`.
fn blocks(model: &Pooled<Gemma3Wrapper>, seq: SequenceId) -> Vec<(usize, usize)> {
    let id = model.pool_sequence(seq);
    model.with_pool(|pool| {
        let state = pool.get_paged_state(id).expect("paged sequence");
        state
            .layers
            .iter()
            .map(|layer| (layer.block_ids.len(), layer.retired.len()))
            .collect()
    })
}

#[test]
fn an_fp16_sequence_is_allocated_on_paged_blocks() {
    let model = paged();
    let seq = SequenceId::from_raw(SEQ_BASE);
    model.prepare_sequence_state(seq);
    model.with_layer_state(seq, |state| {
        assert!(matches!(
            state,
            [LayerState::PagedSliding(_), LayerState::Kv(kv)] if kv.is_paged_backed()
        ));
    });
    let id = model.pool_sequence(seq);
    model.with_pool(|pool| {
        let state = pool.get_paged_state(id).expect("paged sequence");
        assert_eq!(state.layers[0].window, Some(WINDOW as usize));
        assert_eq!(state.layers[1].window, None);
    });
}

#[test]
fn paged_storage_decodes_like_the_ring_past_the_window() {
    let (paged, ring) = (paged(), ring());
    let seq = SequenceId::from_raw(SEQ_BASE + 1);
    paged.prepare_sequence_state(seq);
    ring.prepare_sequence_state(seq);
    let prompt = ids(0..72);
    // Two prefill pieces, the second one crossing the window.
    for piece in [&prompt[..40], &prompt[40..]] {
        assert_close(
            &forward(&paged, seq, piece),
            &forward(&ring, seq, piece),
            "prefill",
        );
    }
    for (step, token) in ids(72..120).into_iter().enumerate() {
        assert_close(
            &forward(&paged, seq, &[token]),
            &forward(&ring, seq, &[token]),
            &format!("decode step {step}"),
        );
    }
}

#[test]
fn live_sliding_blocks_stay_within_the_window_bound() {
    let model = paged();
    let seq = SequenceId::from_raw(SEQ_BASE + 2);
    model.prepare_sequence_state(seq);
    let prompt = ids(0..300);
    for piece in prompt.chunks(64) {
        let _ = forward(&model, seq, piece);
    }
    let bound = (WINDOW as usize).div_ceil(BLOCK) + 1;
    for (k, token) in ids(300..340).into_iter().enumerate() {
        let _ = forward(&model, seq, &[token]);
        let layers = blocks(&model, seq);
        let (sliding_live, sliding_retired) = layers[0];
        assert!(
            sliding_live <= bound,
            "{sliding_live} live sliding blocks exceed {bound}"
        );
        // Retention is off in this pool: only the rewind reserve stays.
        assert!(sliding_retired <= mlxcel_core::cache::REWIND_RESERVE_BLOCKS);
        // The global layer keeps every block.
        assert_eq!(layers[1], ((301 + k).div_ceil(BLOCK), 0));
    }
}

#[test]
fn a_batched_paged_decode_matches_each_row_alone() {
    let (batched, single) = (paged(), paged());
    let seqs = [
        SequenceId::from_raw(SEQ_BASE + 10),
        SequenceId::from_raw(SEQ_BASE + 11),
    ];
    let prompts = [ids(0..70), ids(5..53)];
    for (seq, prompt) in seqs.iter().zip(&prompts) {
        for model in [&batched, &single] {
            model.prepare_sequence_state(*seq);
            let _ = forward(model, *seq, prompt);
        }
    }
    for step in 0..24 {
        let tokens = [(step % 50) + 1, (step % 37) + 2];
        let input = mlxcel_core::from_slice_i32(&tokens, &[2, 1]);
        let logits = batched
            .try_forward_batched_with_ids(&input, Some(&seqs), &mut [], None)
            .expect("batched paged decode");
        let rows = to_vec_f32(&logits);
        let half = rows.len() / 2;
        for (row, (seq, token)) in seqs.iter().zip(tokens).enumerate() {
            let alone = forward(&single, *seq, &[token]);
            assert_close(
                &rows[row * half..(row + 1) * half],
                &alone,
                &format!("step {step} row {row}"),
            );
        }
    }
}
