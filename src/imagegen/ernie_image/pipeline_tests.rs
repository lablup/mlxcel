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
use crate::imagegen::testing::values;

#[test]
fn text_batches_pad_with_zeros_and_keep_lengths() {
    let short = mlxcel_core::from_slice_f32(&[1.0, 2.0], &[1, 1, 2]);
    let long = mlxcel_core::from_slice_f32(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 3, 2]);
    let batch = TextBatch::new(&[&short, &long], 4.0).unwrap();
    assert!(batch.cfg());
    assert_eq!(batch.lengths, [1, 3]);
    assert_eq!(mlxcel_core::array_shape(&batch.hidden), [2, 3, 2]);
    assert_eq!(
        values(&batch.hidden),
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
    let single = TextBatch::new(&[&long], 1.0).unwrap();
    assert!(!single.cfg());
    assert!(TextBatch::new(&[], 1.0).is_err());
}

#[test]
fn initial_noise_is_a_direct_bf16_draw_from_the_seed_key() {
    let noise = initial_noise(42, 128, 64, 32);
    assert_eq!(mlxcel_core::array_shape(&noise), [1, 128, 2, 4]);
    assert_eq!(mlxcel_core::array_dtype(&noise), dtype::BFLOAT16);
    let key = mlxcel_core::random_key(42);
    // SAFETY: `key` is alive for the call.
    let direct = unsafe { mlxcel_core::random_normal(&[1, 128, 2, 4], dtype::BFLOAT16, &*key) };
    assert_eq!(
        mlxcel_core::array_to_raw_bytes(&noise),
        mlxcel_core::array_to_raw_bytes(&direct)
    );
    assert_ne!(
        mlxcel_core::array_to_raw_bytes(&noise),
        mlxcel_core::array_to_raw_bytes(&initial_noise(43, 128, 64, 32))
    );
}
