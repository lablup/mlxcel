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

fn rand(shape: &[i32], seed: u64) -> UniquePtr<MlxArray> {
    let key = mlxcel_core::random_key(seed);
    // SAFETY: `key` is live for the call.
    unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, &*key) }
}

fn rows(x: &MlxArray, start: i32, stop: i32) -> Vec<f32> {
    let s = mlxcel_core::array_shape(x);
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::slice(
        x,
        &[0, start, 0],
        &[s[0], stop, s[2]],
    ))
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn kv_cache_extract_then_cached_matches_full_attention() {
    // [txt 3, ref 4, img 5] tokens, 2 heads of 8.
    let (txt, nref, img) = (3, 4, 5);
    let total = txt + nref + img;
    let q = rand(&[1, 2, total, 8], 1);
    let k = rand(&[1, 2, total, 8], 2);
    let v = rand(&[1, 2, total, 8], 3);
    let mut cache = KvLayerCache::default();
    let extract = attention_with_cache(
        &q,
        &k,
        &v,
        txt,
        Some(KvCacheMode::Extract {
            num_ref_tokens: nref,
        }),
        Some(&mut cache),
    )
    .unwrap();
    assert!(cache.is_populated());
    let full = attention_with_cache(&q, &k, &v, txt, None, None).unwrap();
    // Text and image queries see every key, exactly as without the cache.
    assert!(max_diff(&rows(&extract, 0, txt), &rows(&full, 0, txt)) < 1e-5);
    assert!(
        max_diff(
            &rows(&extract, txt + nref, total),
            &rows(&full, txt + nref, total)
        ) < 1e-5
    );
    // Reference queries see only reference keys.
    let r = |x: &MlxArray| seq(x, txt, txt + nref);
    let ref_only = from_heads(&attend(&r(&q), &r(&k), &r(&v)));
    assert!(
        max_diff(
            &rows(&extract, txt, txt + nref),
            &mlxcel_core::utils::array_to_vec_f32(&ref_only)
        ) < 1e-5
    );

    // A cached step drops the reference tokens from the sequence.
    let drop_ref = |x: &MlxArray| cat_seq(&[&seq(x, 0, txt), &seq(x, txt + nref, total)]);
    let cached = attention_with_cache(
        &drop_ref(&q),
        &drop_ref(&k),
        &drop_ref(&v),
        txt,
        Some(KvCacheMode::Cached),
        Some(&mut cache),
    )
    .unwrap();
    assert!(max_diff(&rows(&cached, 0, txt), &rows(&full, 0, txt)) < 1e-5);
    assert!(
        max_diff(
            &rows(&cached, txt, txt + img),
            &rows(&full, txt + nref, total)
        ) < 1e-5
    );
}

#[test]
fn cached_mode_without_an_entry_errors() {
    let q = rand(&[1, 1, 4, 8], 4);
    let mut cache = KvLayerCache::default();
    assert!(
        attention_with_cache(&q, &q, &q, 2, Some(KvCacheMode::Cached), Some(&mut cache)).is_err()
    );
}

#[test]
fn rope_rotates_adjacent_pairs_in_f32() {
    // One head, two positions, D = 4: pairs (x0, x1), (x2, x3).
    let x = mlxcel_core::from_slice_f32(&[1., 2., 3., 4., 5., 6., 7., 8.], &[1, 1, 2, 4]);
    let x = mlxcel_core::astype(&x, dtype::BFLOAT16);
    let angles = [0.3_f32, -1.1, 0.7, 2.0];
    let cos: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    let sin: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
    let cos_a = mlxcel_core::from_slice_f32(&cos, &[2, 2]);
    let sin_a = mlxcel_core::from_slice_f32(&sin, &[2, 2]);
    let out = apply_rope(&x, &cos_a, &sin_a);
    assert_eq!(mlxcel_core::array_dtype(&out), dtype::BFLOAT16);
    let got = mlxcel_core::utils::array_to_vec_f32(&out);
    let xs = [1., 2., 3., 4., 5., 6., 7., 8.];
    for p in 0..4 {
        let (re, im) = (xs[2 * p], xs[2 * p + 1]);
        let want = [re * cos[p] - im * sin[p], im * cos[p] + re * sin[p]];
        for j in 0..2 {
            assert!((got[2 * p + j] - want[j]).abs() < 0.05, "pair {p}: {got:?}");
        }
    }
}
