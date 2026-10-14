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
use mlxcel_core::cache::{KVCache, RotatingKVCache};

/// `[1, 1, len, 2]` keys whose values encode `(row, position)`.
fn row_tokens(row: usize, len: i32) -> UniquePtr<MlxArray> {
    let data: Vec<f32> = (0..len)
        .flat_map(|t| [row as f32 * 100.0 + t as f32, -(t as f32)])
        .collect();
    mlxcel_core::from_slice_f32(&data, &[1, 1, len, 2])
}

fn read(array: &MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&array);
    mlxcel_core::array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn standard(row: usize, len: i32) -> Cache {
    let mut cache = KVCache::new();
    let _ = cache.update_and_fetch(row_tokens(row, len), row_tokens(row, len));
    Cache::Standard(cache)
}

fn rotating(row: usize, len: i32, window: i32) -> Cache {
    let mut cache = RotatingKVCache::new(window);
    let _ = cache.update_and_fetch(row_tokens(row, len), row_tokens(row, len));
    Cache::Rotating(cache)
}

fn stacked_keys(cache: &Cache) -> (Vec<i32>, Vec<f32>, i32) {
    let (keys, offset) = match cache {
        Cache::Standard(c) => (c.keys.as_ref().unwrap(), c.offset),
        Cache::Rotating(c) => (c.keys.as_ref().unwrap(), c.offset),
    };
    (mlxcel_core::array_shape(keys), read(keys), offset)
}

#[test]
fn ragged_rows_stack_with_history_first_and_a_zero_tail() {
    let stacked = stack_prefilled_rows(vec![
        vec![standard(0, 3), rotating(0, 3, 8)],
        vec![standard(1, 5), rotating(1, 5, 8)],
    ])
    .expect("unwrapped ragged rows stack");
    for cache in &stacked {
        let (shape, keys, offset) = stacked_keys(cache);
        assert_eq!(offset, 5, "the shared offset is the longest row's");
        assert_eq!(shape[0], 2);
        let width = shape[2] as usize;
        let at = |row: usize, t: usize| keys[(row * width + t) * 2];
        // Row 0 keeps slot t == token t, then zeros: never left padding.
        assert_eq!(
            (0..3).map(|t| at(0, t)).collect::<Vec<_>>(),
            [0.0, 1.0, 2.0]
        );
        assert!((3..width).all(|t| at(0, t) == 0.0 && keys[(t) * 2 + 1] == 0.0));
        assert_eq!(
            (0..5).map(|t| at(1, t)).collect::<Vec<_>>(),
            [100.0, 101.0, 102.0, 103.0, 104.0]
        );
    }
}

#[test]
fn equal_rows_stack_unchanged_even_past_the_window() {
    let stacked = stack_prefilled_rows(vec![vec![rotating(0, 10, 4)], vec![rotating(1, 10, 4)]])
        .expect("equal-length rows always stack");
    let (shape, keys, offset) = stacked_keys(&stacked[0]);
    assert_eq!(offset, 10);
    let width = shape[2] as usize;
    let row1: Vec<f32> = (0..width).map(|t| keys[(width + t) * 2]).collect();
    let row0: Vec<f32> = (0..width).map(|t| keys[t * 2]).collect();
    assert!(row1.iter().zip(&row0).all(|(a, b)| a - b == 100.0));
}

#[test]
fn ragged_rows_past_the_window_are_refused() {
    let Err(error) = stack_prefilled_rows(vec![vec![rotating(0, 3, 4)], vec![rotating(1, 10, 4)]])
    else {
        panic!("a wrapped ring cannot keep slot == position");
    };
    assert!(error.contains("sliding window"), "{error}");
}

#[test]
fn quantized_rows_are_refused() {
    let mut cache = KVCache::new_with_mode(mlxcel_core::cache::KVCacheMode::Int8);
    let _ = cache.update_and_fetch(row_tokens(0, 3), row_tokens(0, 3));
    let Err(error) = stack_prefilled_rows(vec![vec![Cache::Standard(cache)], vec![standard(1, 3)]])
    else {
        panic!("only FP16 caches stack");
    };
    assert!(error.contains("FP16"), "{error}");
}

#[test]
fn per_row_concatenates_row_results_in_order() {
    let x = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let out = per_row(&x, |row| mlxcel_core::multiply_scalar(row, 2.0));
    assert_eq!(mlxcel_core::array_shape(&out), [3, 2]);
    assert_eq!(read(&out), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}
