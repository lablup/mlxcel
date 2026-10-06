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

use super::*;

#[test]
fn full_head_verify_rows_are_byte_equal_to_decode_queries() {
    const HISTORY: i32 = 257;
    const WIDTH: i32 = 4;
    const DIM: i32 = 512;
    let make = |shape: &[i32], salt: usize| {
        let n = shape.iter().product::<i32>() as usize;
        let data: Vec<f32> = (0..n)
            .map(|i| (((i * 131 + salt) % 257) as f32 - 128.0) / 512.0)
            .collect();
        mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&data, shape),
            mlxcel_core::dtype::FLOAT16,
        )
    };
    let q = make(&[1, 2, WIDTH, DIM], 3);
    let k = make(&[1, 1, HISTORY + WIDTH, DIM], 7);
    let v = make(&[1, 1, HISTORY + WIDTH, DIM], 13);
    let mask = mlxcel_core::utils::create_causal_mask(WIDTH, HISTORY);
    let block = attend_query_rows(&q, &k, &v, Some(&mask), 1.0, 0);
    for row in 0..WIDTH {
        let query = slice_axis(&q, 2, row, row + 1);
        let key = slice_axis(&k, 2, 0, HISTORY + row + 1);
        let value = slice_axis(&v, 2, 0, HISTORY + row + 1);
        let reference = mlxcel_core::causal_attention(&query, &key, &value, 1.0, 0.0, 0);
        assert_eq!(
            mlxcel_core::array_to_raw_bytes(&slice_axis(&block, 2, row, row + 1)),
            mlxcel_core::array_to_raw_bytes(&reference)
        );
    }
}

#[test]
fn query_rows_preserve_tree_sibling_and_batch_padding_masks() {
    const WIDTH: i32 = 4;
    const HISTORY: i32 = 3;
    const TOTAL: i32 = HISTORY + WIDTH;
    const DIM: i32 = 512;
    let q = mlxcel_core::zeros(&[2, 1, WIDTH, DIM], mlxcel_core::dtype::FLOAT32);
    let k = mlxcel_core::zeros(&[2, 1, TOTAL, DIM], mlxcel_core::dtype::FLOAT32);
    let mut identity = vec![0.0f32; (2 * TOTAL * DIM) as usize];
    for batch in 0..2 {
        for key in 0..TOTAL {
            identity[((batch * TOTAL + key) * DIM + key) as usize] = 1.0;
        }
    }
    let v = mlxcel_core::from_slice_f32(&identity, &[2, 1, TOTAL, DIM]);
    let ancestors = [vec![0], vec![1], vec![1, 2], vec![0, 3]];
    let mut masks = vec![f32::NEG_INFINITY; (2 * WIDTH * TOTAL) as usize];
    for batch in 0..2 {
        for row in 0..WIDTH {
            for key in 0..TOTAL {
                // Batch row 1 has one leading padding key. Each tree branch must
                // exclude its sibling even when that sibling is physically earlier.
                if (key >= batch && key < HISTORY)
                    || ancestors[row as usize].contains(&(key - HISTORY))
                {
                    masks[((batch * WIDTH + row) * TOTAL + key) as usize] = 0.0;
                }
            }
        }
    }
    let mask = mlxcel_core::from_slice_f32(&masks, &[2, 1, WIDTH, TOTAL]);
    let output = attend_query_rows(&q, &k, &v, Some(&mask), 1.0, 0);
    let raw = mlxcel_core::array_to_raw_bytes(&output);
    let output: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    for batch in 0..2 {
        for row in 0..WIDTH {
            for key in 0..TOTAL {
                assert_eq!(
                    output[((batch * WIDTH + row) * DIM + key) as usize] > 0.0,
                    masks[((batch * WIDTH + row) * TOTAL + key) as usize] == 0.0,
                    "batch={batch} query={row} key={key}"
                );
            }
        }
    }
}

#[test]
fn exact_query_rows_are_limited_to_a_single_linear_unpadded_stream() {
    assert!(linear_verify_layout(1, None, false));
    assert!(linear_verify_layout(1, Some(&[0, 1, 2, 3]), false));
    assert!(!linear_verify_layout(1, Some(&[0, 1, 1, 2]), false));
    assert!(!linear_verify_layout(2, None, false));
    assert!(!linear_verify_layout(1, None, true));
}

#[test]
fn buffered_ring_attention_matches_decode_after_wrap_and_partial_accept() {
    const WINDOW: i32 = 32;
    const WIDTH: i32 = 4;
    const DIM: i32 = 256;
    let make = |count: i32, salt: i32| {
        let data: Vec<f32> = (0..count * DIM)
            .map(|i| (((i * 31 + salt * 17) % 127) as f32 - 63.0) / 128.0)
            .collect();
        mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&data, &[1, 1, count, DIM]),
            mlxcel_core::dtype::FLOAT16,
        )
    };
    for prefill in [7, WINDOW + 9] {
        let mut chain = mlxcel_core::cache::RotatingKVCache::new(WINDOW);
        let mut block = mlxcel_core::cache::RotatingKVCache::new(WINDOW);
        chain.update_and_fetch(make(prefill, 1), make(prefill, 2));
        block.update_and_fetch(make(prefill, 1), make(prefill, 2));
        block.enable_speculative_buffer(8).unwrap();
        for round in 0..20 {
            let cursor = block.speculative_ring_cursor().unwrap();
            let q = make(WIDTH, round + 3);
            let k = make(WIDTH, round + 7);
            let v = make(WIDTH, round + 11);
            let (keys, values) =
                block.update_and_fetch(mlxcel_core::copy(&k), mlxcel_core::copy(&v));
            let actual = attend_ring_rows(&q, &keys, &values, 1.0, WINDOW, cursor);
            // Keep only the first two verify rows; next round overwrites the
            // rejected tail, while the reference advances exactly two steps.
            for row in 0..2 {
                let (keys, values) = chain.update_and_fetch(
                    slice_axis(&k, 2, row, row + 1),
                    slice_axis(&v, 2, row, row + 1),
                );
                let expected = mlxcel_core::causal_attention(
                    &slice_axis(&q, 2, row, row + 1),
                    &keys,
                    &values,
                    1.0,
                    0.0,
                    WINDOW,
                );
                assert_eq!(
                    mlxcel_core::array_to_raw_bytes(&slice_axis(&actual, 2, row, row + 1)),
                    mlxcel_core::array_to_raw_bytes(&expected),
                    "prefill={prefill}, round={round}, row={row}"
                );
            }
            block.trim(WIDTH - 2);
        }
    }
}

/// #2160: the 12B geometry (16 query heads over 8 KV heads at head_dim 256,
/// windowed) verified row-wise must reproduce classic M=1 decode byte for byte,
/// with and without the MTP rotating buffer, before and after the window wraps.
/// On CUDA a batched M=4 block leaves the single-query `sdpa_vector` kernel
/// classic decode takes at this head_dim, which is what the row-wise path fixes.
#[test]
fn gemma4_12b_sliding_verify_rows_match_decode_byte_for_byte() {
    const WINDOW: i32 = 32;
    const WIDTH: i32 = 4;
    const DIM: i32 = 256;
    const HEADS: i32 = 16;
    const KV_HEADS: i32 = 8;
    let make = |heads: i32, count: i32, salt: i32| {
        let data: Vec<f32> = (0..heads * count * DIM)
            .map(|i| (((i * 31 + salt * 17) % 127) as f32 - 63.0) / 128.0)
            .collect();
        mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&data, &[1, heads, count, DIM]),
            mlxcel_core::dtype::BFLOAT16,
        )
    };
    for buffered in [false, true] {
        for prefill in [7, WINDOW + 9] {
            // Unbuffered caches only stay in decode order while the block fits
            // before the wrap; the buffered ring is the serving layout.
            if !buffered && prefill + WIDTH > WINDOW {
                continue;
            }
            let mut chain = mlxcel_core::cache::RotatingKVCache::new(WINDOW);
            let mut block = mlxcel_core::cache::RotatingKVCache::new(WINDOW);
            chain.update_and_fetch(make(KV_HEADS, prefill, 1), make(KV_HEADS, prefill, 2));
            block.update_and_fetch(make(KV_HEADS, prefill, 1), make(KV_HEADS, prefill, 2));
            if buffered {
                block.enable_speculative_buffer(8).unwrap();
            }
            let rounds = if buffered { 12 } else { 1 };
            for round in 0..rounds {
                let cursor = block.speculative_ring_cursor();
                assert_eq!(cursor.is_some(), buffered);
                let q = make(HEADS, WIDTH, round + 3);
                let k = make(KV_HEADS, WIDTH, round + 7);
                let v = make(KV_HEADS, WIDTH, round + 11);
                let (keys, values) =
                    block.update_and_fetch(mlxcel_core::copy(&k), mlxcel_core::copy(&v));
                let actual = attend_verify_rows(&q, &keys, &values, 0.0625, WINDOW, cursor)
                    .expect("a windowed verify block takes the row-wise path");
                let kept = if buffered { 2 } else { WIDTH };
                for row in 0..kept {
                    let (keys, values) = chain.update_and_fetch(
                        slice_axis(&k, 2, row, row + 1),
                        slice_axis(&v, 2, row, row + 1),
                    );
                    let expected = mlxcel_core::causal_attention(
                        &slice_axis(&q, 2, row, row + 1),
                        &keys,
                        &values,
                        0.0625,
                        0.0,
                        WINDOW,
                    );
                    assert_eq!(
                        mlxcel_core::array_to_raw_bytes(&slice_axis(&actual, 2, row, row + 1)),
                        mlxcel_core::array_to_raw_bytes(&expected),
                        "buffered={buffered}, prefill={prefill}, round={round}, row={row}"
                    );
                }
                if buffered {
                    block.trim(WIDTH - 2);
                }
            }
        }
    }
}

#[test]
fn single_full_attention_query_keeps_ordinary_dispatch() {
    let q = mlxcel_core::zeros(&[1, 2, 1, 512], mlxcel_core::dtype::FLOAT16);
    let k = mlxcel_core::zeros(&[1, 1, 5, 512], mlxcel_core::dtype::FLOAT16);
    assert!(attend_verify_rows(&q, &k, &k, 1.0, 0, None).is_none());
    assert!(attend_verify_rows(&q, &k, &k, 1.0, 0, Some(3)).is_none());
}
