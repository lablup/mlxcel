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

//! The fused batched decode over window-bounded sliding layers (issue #3256).
//!
//! Each row's sliding layer has dropped head blocks (`block_base > 0`) and its
//! window starts mid-page (`first_page_offset > 0`). The batch is sized past
//! the batched v2 floor, so the fused kernel serves it. Every row is checked
//! against attention computed on the host from the keys and values the test
//! wrote: the last `window` positions of that row.

use std::cell::RefCell;

use super::*;
use crate::cache::{PagedBlockPool, PagedKvLayout, PagedLayerState, PagedSequenceState};
use crate::dtype;

const PAGE: usize = 32;
const WINDOW: usize = 512;

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

/// Random fp16 values, returned as the array and as the f32 values it holds.
fn random(rng: &mut Rng, shape: &[i32]) -> (UniquePtr<MlxArray>, Vec<f32>) {
    let n: usize = shape.iter().map(|d| *d as usize).product();
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let arr = ffi::astype(&ffi::from_slice_f32(&data, shape), dtype::FLOAT16);
    let held = to_vec_f32(&arr);
    (arr, held)
}

fn to_vec_f32(arr: &MlxArray) -> Vec<f32> {
    let as_f32 = ffi::astype(arr, dtype::FLOAT32);
    ffi::eval(&as_f32);
    ffi::array_to_raw_bytes(&as_f32)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
        .collect()
}

/// Host attention of one row: `q` is `[q_heads, d]`, `k`/`v` are
/// `[kv_heads, n, d]`, over positions `start..n`.
fn host_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_heads: usize,
    kv_heads: usize,
    n: usize,
    d: usize,
    start: usize,
    scale: f32,
) -> Vec<f32> {
    let group = q_heads / kv_heads;
    let mut out = vec![0.0f32; q_heads * d];
    for h in 0..q_heads {
        let kh = h / group;
        let qh = &q[h * d..(h + 1) * d];
        let scores: Vec<f32> = (start..n)
            .map(|t| {
                let kt = &k[(kh * n + t) * d..(kh * n + t + 1) * d];
                qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * scale
            })
            .collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f32 = weights.iter().sum();
        for (w, t) in weights.iter().zip(start..n) {
            let vt = &v[(kh * n + t) * d..(kh * n + t + 1) * d];
            for (o, x) in out[h * d..(h + 1) * d].iter_mut().zip(vt) {
                *o += w / total * x;
            }
        }
    }
    out
}

/// Append `[kv_heads, extra, d]` rows to `[kv_heads, n, d]` host values.
fn append_rows(
    base: &[f32],
    extra: &[f32],
    kv_heads: usize,
    n: usize,
    add: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(base.len() + extra.len());
    for h in 0..kv_heads {
        out.extend_from_slice(&base[h * n * d..(h + 1) * n * d]);
        out.extend_from_slice(&extra[h * add * d..(h + 1) * add * d]);
    }
    out
}

#[test]
fn a_fused_sliding_batch_attends_each_rows_own_window() {
    crate::test_support::kernel_ports::require_paged_attention_port!();
    run_sliding_batch(8, 2, 64, &[700, 1100, 937, 1500], 40);
}

/// A lone row reads its window in place through the single-row entry, though
/// 512 visible tokens sit far below the batched floor.
#[test]
fn a_single_sliding_row_takes_the_fused_read_below_the_batch_floor() {
    crate::test_support::kernel_ports::require_paged_attention_port!();
    run_sliding_batch(4, 1, 256, &[777], 70);
}

/// The Gemma 3 1B attention shape: 4 query heads over one KV head of 256.
#[test]
fn a_fused_sliding_batch_at_gemma3_1b_geometry() {
    crate::test_support::kernel_ports::require_paged_attention_port!();
    run_sliding_batch(4, 1, 256, &[520, 515, 600, 513], 80);
}

fn run_sliding_batch(q_heads: usize, kv_heads: usize, d: usize, prompts: &[usize], steps: usize) {
    let batch = prompts.len();
    let layout = PagedKvLayout::uniform(1, PAGE, PAGE * kv_heads * d * 2).expect("layout");
    let mut pool = PagedBlockPool::new(layout.clone());
    pool.set_slab_blocks(1024).expect("fresh pool");
    let pool = Rc::new(RefCell::new(pool));
    let states: Vec<_> = (0..batch)
        .map(|_| {
            let mut state = PagedSequenceState::new(&layout);
            state.layers[0] = PagedLayerState::sliding(WINDOW);
            Rc::new(RefCell::new(state))
        })
        .collect();
    let mut caches: Vec<KVCache> = states
        .iter()
        .map(|s| KVCache::new_paged(pool.clone(), s.clone(), 0))
        .collect();

    let mut rng = Rng(0x3256 | 1);
    let mut host: Vec<(Vec<f32>, Vec<f32>, usize)> = Vec::new();
    for (b, &len) in prompts.iter().enumerate() {
        let shape = [1, kv_heads as i32, len as i32, d as i32];
        let (k, hk) = random(&mut rng, &shape);
        let (v, hv) = random(&mut rng, &shape);
        caches[b].update(k, v);
        host.push((hk, hv, len));
    }

    let scale = 1.0 / (d as f32).sqrt();
    let before = paged_batch_decode_test_stats();
    for step in 0..steps {
        let (q, hq) = random(&mut rng, &[batch as i32, q_heads as i32, 1, d as i32]);
        let (k, hk) = random(&mut rng, &[batch as i32, kv_heads as i32, 1, d as i32]);
        let (v, hv) = random(&mut rng, &[batch as i32, kv_heads as i32, 1, d as i32]);
        let out = if batch == 1 {
            paged_single_decode_attention(&q, &k, &v, &mut caches[0], scale).expect("servable")
        } else {
            let mut refs: Vec<&mut KVCache> = caches.iter_mut().collect();
            paged_batch_decode_attention(&q, &k, &v, &mut refs, scale, 0.0).expect("servable")
        };
        let got = to_vec_f32(&out);
        for b in 0..batch {
            let (hk_b, hv_b, n) = &host[b];
            let row_k = &hk[b * kv_heads * d..(b + 1) * kv_heads * d];
            let row_v = &hv[b * kv_heads * d..(b + 1) * kv_heads * d];
            let k_all = append_rows(hk_b, row_k, kv_heads, *n, 1, d);
            let v_all = append_rows(hv_b, row_v, kv_heads, *n, 1, d);
            let n1 = n + 1;
            let want = host_attention(
                &hq[b * q_heads * d..(b + 1) * q_heads * d],
                &k_all,
                &v_all,
                q_heads,
                kv_heads,
                n1,
                d,
                n1.saturating_sub(WINDOW),
                scale,
            );
            let row = &got[b * q_heads * d..(b + 1) * q_heads * d];
            let err = row
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(err < 2e-2, "step {step} row {b}: max abs error {err}");
            host[b] = (k_all, v_all, n1);
        }
    }
    let after = paged_batch_decode_test_stats();
    assert!(
        after.v2_launches - before.v2_launches == steps as u64,
        "every step should take the fused path"
    );
    for state in &states {
        let state = state.borrow();
        let layer = &state.layers[0];
        assert!(layer.block_base > 0, "head blocks should have retired");
        assert_eq!(layer.visible_len(), WINDOW);
    }
}
