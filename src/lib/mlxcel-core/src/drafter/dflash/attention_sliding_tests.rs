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

//! Sliding-window and causal draft attention (issue #3261).
//!
//! Every case is checked against a reference built from the PLAIN attention
//! (no window, no mask) with the same weights: block row `j` of a causal
//! sliding layer must equal the last row of a plain forward that is handed
//! exactly the context rows and block rows that row may see. RoPE is
//! relative, so the reference may place those rows at other absolute
//! positions; the comparison tolerance covers the rotation's rounding.

use super::*;
use crate::dtype;

const HIDDEN: i32 = 8;
const HEADS: i32 = 2;
const HEAD_DIM: i32 = 4;

/// Varied, bounded weights: a salted ramp, so no projection is the identity
/// and every context row contributes a different value.
fn ramp(shape: &[i32], salt: usize) -> UniquePtr<MlxArray> {
    let n: usize = shape.iter().product::<i32>() as usize;
    let data: Vec<f32> = (0..n)
        .map(|i| ((((i + salt * 131) * 37 + 11) % 97) as f32 - 48.0) * 0.02)
        .collect();
    ffi::from_slice_f32(&data, shape)
}

fn attention(sliding_window: Option<usize>, is_causal: Option<bool>) -> DFlashAttention {
    let mut weights: WeightMap = std::collections::HashMap::new();
    weights.insert(
        "self_attn.q_proj.weight".to_string(),
        ramp(&[HEADS * HEAD_DIM, HIDDEN], 1),
    );
    weights.insert(
        "self_attn.k_proj.weight".to_string(),
        ramp(&[HEAD_DIM, HIDDEN], 2),
    );
    weights.insert(
        "self_attn.v_proj.weight".to_string(),
        ramp(&[HEAD_DIM, HIDDEN], 3),
    );
    weights.insert(
        "self_attn.o_proj.weight".to_string(),
        ramp(&[HIDDEN, HEADS * HEAD_DIM], 4),
    );
    weights.insert(
        "self_attn.q_norm.weight".to_string(),
        ffi::ones(&[HEAD_DIM], dtype::FLOAT32),
    );
    weights.insert(
        "self_attn.k_norm.weight".to_string(),
        ffi::ones(&[HEAD_DIM], dtype::FLOAT32),
    );
    let cfg = DFlashConfig {
        hidden_size: HIDDEN as usize,
        num_hidden_layers: 1,
        num_attention_heads: HEADS as usize,
        num_key_value_heads: 1,
        head_dim: HEAD_DIM as usize,
        vocab_size: 32,
        rope_theta: 10000.0,
        mask_token_id: 31,
        target_layer_ids: vec![0],
        num_target_layers: Some(1),
        // The sliding fields are read for a DSpark drafter only.
        markov_rank: 2,
        layer_types: vec![
            if sliding_window.is_some() {
                "sliding_attention"
            } else {
                "full_attention"
            }
            .to_string(),
        ],
        sliding_window,
        is_causal,
        ..DFlashConfig::default()
    };
    DFlashAttention::from_weights(&weights, "self_attn", &cfg, 0, 64, 4).unwrap()
}

/// `[1, rows, HIDDEN]` rows numbered from `first`: row `r` is a distinct,
/// bounded pattern, so any row that is wrongly included or dropped moves the
/// output.
fn rows(first: i32, count: i32) -> UniquePtr<MlxArray> {
    let mut data = Vec::with_capacity((count * HIDDEN) as usize);
    for r in first..first + count {
        for c in 0..HIDDEN {
            data.push((((r * 7 + c * 3 + 5) % 23) as f32 - 11.0) * 0.1 + (r % 101) as f32 * 0.003);
        }
    }
    ffi::from_slice_f32(&data, &[1, count, HIDDEN])
}

/// Proposal rows `0..count`, distinct from every context row.
fn block(count: i32) -> UniquePtr<MlxArray> {
    rows(1000, count)
}

fn to_vec(a: &MlxArray) -> Vec<f32> {
    ffi::eval(a);
    crate::utils::array_to_vec_f32(a)
}

fn row(a: &MlxArray, index: i32) -> Vec<f32> {
    let shape = ffi::array_shape(a);
    to_vec(&ffi::slice(a, &[0, index, 0], &[1, index + 1, shape[2]]))
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_close_within(got, want, 2e-4, what);
}

/// [`assert_close`] with an explicit relative tolerance, for rows far from
/// position zero: the reference rotates the same rows near zero, and f32
/// rounds a rotation angle of a few thousand radians to about 1e-4.
fn assert_close_within(got: &[f32], want: &[f32], tolerance: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tolerance * (1.0 + w.abs()),
            "{what}: element {i} got {g}, want {w}"
        );
    }
}

/// The reference for block row `j`: a plain forward over the context rows
/// `ctx_first..ctx_first + ctx_count` and block rows `0..=j` (or all of them
/// for a non-causal layer), last or `j`-th row.
fn reference_row(ctx_first: i32, ctx_count: i32, block_len: i32, j: i32, causal: bool) -> Vec<f32> {
    let plain = attention(None, Some(false));
    let mut cache = KVCache::new();
    let visible = if causal { j + 1 } else { block_len };
    let x = ffi::slice(&block(block_len), &[0, 0, 0], &[1, visible, HIDDEN]);
    let out = plain.forward(&x, &rows(ctx_first, ctx_count), &mut cache);
    row(&out, j)
}

/// The reference for a block row whose window no longer reaches the context:
/// row `j` of a `block_len`-row block sees block rows `first..=j` only. A
/// forward needs at least one context row, and context rows go through
/// `k_proj` / `v_proj` exactly as block rows do, so rows `first..j` stand in
/// the context slot and row `j` is the block.
fn reference_block_only(block_len: i32, first: i32, j: i32) -> Vec<f32> {
    assert!(first < j, "the row needs at least one earlier block row");
    let plain = attention(None, Some(false));
    let mut cache = KVCache::new();
    let full = block(block_len);
    let stand_in_ctx = ffi::slice(&full, &[0, first, 0], &[1, j, HIDDEN]);
    let last = ffi::slice(&full, &[0, j, 0], &[1, j + 1, HIDDEN]);
    row(&plain.forward(&last, &stand_in_ctx, &mut cache), 0)
}

/// A context shorter than the window: nothing slides, the block is causal.
#[test]
fn a_prompt_shorter_than_the_window_is_plain_causal_attention() {
    let attn = attention(Some(8), None);
    assert_eq!(attn.sliding_window, Some(8));
    assert!(
        attn.is_causal,
        "a sliding layer is causal when is_causal is unset"
    );
    let mut cache = KVCache::new();
    let out = attn.forward(&block(3), &rows(0, 3), &mut cache);
    assert_eq!(ffi::array_shape(&out), vec![1, 3, HIDDEN]);
    for j in 0..3 {
        // Position of row j is 3 + j; the window of 8 reaches back past the
        // first context row for every j, so all three context rows count.
        assert_close(
            &row(&out, j),
            &reference_row(0, 3, 3, j, true),
            "short prompt",
        );
    }
    assert_eq!(cache.offset, 3);
    assert_eq!(cache.live_len(), 3);
}

/// A context longer than the window, in one chunk: only the newest
/// `window - 1` rows are kept, and each later block row sees one fewer.
#[test]
fn a_prompt_longer_than_the_window_keeps_the_newest_rows() {
    let window = 5;
    let attn = attention(Some(window as usize), None);
    let mut cache = KVCache::new();
    let ctx = 20;
    let len = 4;
    let out = attn.forward(&block(len), &rows(0, ctx), &mut cache);
    for j in 0..len {
        // Row j at position ctx + j attends positions [ctx + j - 4, ctx + j]:
        // the newest `window - 1 - j` context rows and block rows 0..=j.
        let ctx_count = (window - 1 - j).max(0);
        assert_close(
            &row(&out, j),
            &reference_row(ctx - ctx_count, ctx_count, len, j, true),
            &format!("long prompt row {j}"),
        );
    }
    // The cache is bounded by the window and still counts every position.
    assert_eq!(cache.live_len(), window - 1);
    assert_eq!(cache.offset, ctx);
}

/// The same context fed over several rounds, crossing the window: each
/// round's block must see exactly the newest rows, whatever the chunking.
#[test]
fn the_window_slides_across_rounds() {
    let window = 6;
    let attn = attention(Some(window as usize), None);
    let mut cache = KVCache::new();
    let mut total = 0;
    for (round, chunk) in [2, 1, 4, 1, 1, 7, 3].into_iter().enumerate() {
        let len = 3;
        let out = attn.forward(&block(len), &rows(total, chunk), &mut cache);
        total += chunk;
        for j in 0..len {
            let ctx_count = (window - 1 - j).min(total).max(0);
            assert_close(
                &row(&out, j),
                &reference_row(total - ctx_count, ctx_count, len, j, true),
                &format!("round {round} row {j}"),
            );
        }
        assert!(
            cache.live_len() < window,
            "round {round}: cache grew past the window"
        );
        assert_eq!(cache.live_len(), total.min(window - 1));
        assert_eq!(
            cache.offset, total,
            "round {round}: offset must stay absolute"
        );
    }
}

/// The smallest window, the widest block the round loop sends, and a block
/// of one row.
#[test]
fn window_and_block_size_boundaries() {
    // Window 2: one context row plus the row itself.
    let attn = attention(Some(2), None);
    let mut cache = KVCache::new();
    let out = attn.forward(&block(3), &rows(0, 9), &mut cache);
    assert_close(
        &row(&out, 0),
        &reference_row(8, 1, 3, 0, true),
        "window 2 row 0",
    );
    // Row 1 sees block rows 0 and 1 only: its window no longer reaches the
    // context. Row 2 sees block rows 1 and 2.
    assert_close(
        &row(&out, 1),
        &reference_block_only(3, 0, 1),
        "window 2 row 1",
    );
    assert_close(
        &row(&out, 2),
        &reference_block_only(3, 1, 2),
        "window 2 row 2",
    );
    assert_eq!(cache.live_len(), 1);

    // Block of one row (verify width 2 with `sample_from_anchor`).
    let attn = attention(Some(4), None);
    let mut cache = KVCache::new();
    let out = attn.forward(&block(1), &rows(0, 10), &mut cache);
    assert_eq!(ffi::array_shape(&out), vec![1, 1, HIDDEN]);
    assert_close(
        &row(&out, 0),
        &reference_row(7, 3, 1, 0, true),
        "block of one",
    );

    // A block wider than the window: late rows see only block rows.
    let attn = attention(Some(3), None);
    let mut cache = KVCache::new();
    let out = attn.forward(&block(6), &rows(0, 5), &mut cache);
    assert_close(
        &row(&out, 0),
        &reference_row(3, 2, 6, 0, true),
        "wide block row 0",
    );
    // Row 5 attends positions [8, 10]: block rows 3, 4 and 5, no context.
    assert_close(
        &row(&out, 5),
        &reference_block_only(6, 3, 5),
        "wide block row 5",
    );
}

/// `sliding_window_non_causal`: the context still slides, the block attends
/// itself bidirectionally.
#[test]
fn a_non_causal_sliding_layer_drops_old_context_but_not_the_mask() {
    let window = 4;
    let attn = attention(Some(window as usize), Some(false));
    assert!(!attn.is_causal);
    let mut cache = KVCache::new();
    let out = attn.forward(&block(3), &rows(0, 12), &mut cache);
    for j in 0..3 {
        assert_close(
            &row(&out, j),
            &reference_row(9, 3, 3, j, false),
            &format!("non-causal sliding row {j}"),
        );
    }
    assert_eq!(cache.live_len(), window - 1);
}

/// An explicit `is_causal` on a full-attention layer: causal, no window, the
/// cache keeps everything.
#[test]
fn an_explicitly_causal_full_layer_masks_without_sliding() {
    let attn = attention(None, Some(true));
    assert!(attn.is_causal);
    assert_eq!(attn.sliding_window, None);
    let mut cache = KVCache::new();
    let out = attn.forward(&block(4), &rows(0, 30), &mut cache);
    for j in 0..4 {
        assert_close(
            &row(&out, j),
            &reference_row(0, 30, 4, j, true),
            &format!("causal full row {j}"),
        );
    }
    assert_eq!(cache.live_len(), 30);
}

/// The plain DFlash and LFM2 DSpark configuration is the pre-#3261 forward:
/// no window, no mask, every block row sees the whole block.
#[test]
fn a_full_non_causal_layer_is_unchanged() {
    let attn = attention(None, Some(false));
    assert!(!attn.is_causal);
    assert_eq!(attn.sliding_window, None);
    let mut cache = KVCache::new();
    let out = attn.forward(&block(3), &rows(0, 6), &mut cache);
    for j in 0..3 {
        assert_close(
            &row(&out, j),
            &reference_row(0, 6, 3, j, false),
            &format!("plain row {j}"),
        );
    }
    assert_eq!(cache.live_len(), 6);

    // A plain DFlash config (no Markov head) that declares sliding layers
    // builds this same attention.
    let mut weights: WeightMap = std::collections::HashMap::new();
    for (name, shape, salt) in [
        ("q_proj", [HEADS * HEAD_DIM, HIDDEN], 1),
        ("k_proj", [HEAD_DIM, HIDDEN], 2),
        ("v_proj", [HEAD_DIM, HIDDEN], 3),
        ("o_proj", [HIDDEN, HEADS * HEAD_DIM], 4),
    ] {
        weights.insert(format!("self_attn.{name}.weight"), ramp(&shape, salt));
    }
    for name in ["q_norm", "k_norm"] {
        weights.insert(
            format!("self_attn.{name}.weight"),
            ffi::ones(&[HEAD_DIM], dtype::FLOAT32),
        );
    }
    let plain_dflash = DFlashConfig {
        hidden_size: HIDDEN as usize,
        num_hidden_layers: 1,
        num_attention_heads: HEADS as usize,
        num_key_value_heads: 1,
        head_dim: HEAD_DIM as usize,
        layer_types: vec!["sliding_attention".to_string()],
        sliding_window: Some(2),
        is_causal: None,
        ..DFlashConfig::default()
    };
    let attn = DFlashAttention::from_weights(&weights, "self_attn", &plain_dflash, 0, 64, 4)
        .expect("plain DFlash attention");
    assert_eq!(attn.sliding_window, None);
    assert!(!attn.is_causal);
}

/// The cache of a sliding layer is bounded by the window however long the
/// context is and however many rounds run: the rows a round touches are the
/// window plus the block, never the history.
#[test]
fn a_long_run_keeps_the_sliding_cache_bounded() {
    let window = 16;
    let attn = attention(Some(window as usize), None);
    let mut cache = KVCache::new();
    // One prompt-sized chunk, far longer than the window.
    let prompt = 3_000;
    let out = attn.forward(&block(4), &rows(0, prompt), &mut cache);
    assert_close_within(
        &row(&out, 0),
        &reference_row(prompt - (window - 1), window - 1, 4, 0, true),
        5e-3,
        "3k prompt row 0",
    );
    assert_eq!(cache.live_len(), window - 1);
    assert_eq!(cache.offset, prompt);

    let mut total = prompt;
    let mut last = out;
    for _ in 0..400 {
        last = attn.forward(&block(4), &rows(total, 3), &mut cache);
        total += 3;
        assert_eq!(cache.live_len(), window - 1);
    }
    assert_eq!(cache.offset, total);
    assert_close_within(
        &row(&last, 2),
        &reference_row(total - (window - 3), window - 3, 4, 2, true),
        5e-3,
        "after 400 rounds",
    );

    // A 200 000-row context in one chunk: the cache still ends at
    // `window - 1` rows and the position count is exact.
    let mut cache = KVCache::new();
    let huge = 200_000;
    let out = attn.forward(&block(2), &rows(0, huge), &mut cache);
    ffi::eval(&out);
    assert_eq!(ffi::array_shape(&out), vec![1, 2, HIDDEN]);
    assert_eq!(cache.live_len(), window - 1);
    assert_eq!(cache.offset, huge);
}
