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
fn sequence_positions_count_from_start_and_pad_with_zeros() {
    let p = sequence_positions(3, 1);
    assert_eq!(p.len(), 32 * 3);
    assert_eq!(&p[..9], &[1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]);
    assert!(p[9..].iter().all(|v| *v == 0.0));
    assert_eq!(sequence_positions(64, 5).len(), 64 * 3);
    assert!(sequence_positions(0, 5).is_empty());
}

#[test]
fn image_positions_are_start_row_column() {
    let p = image_positions(2, 3, 33);
    assert_eq!(p.len(), 32 * 3);
    #[rustfmt::skip]
    let want = [
        33.0, 0.0, 0.0, 33.0, 0.0, 1.0, 33.0, 0.0, 2.0,
        33.0, 1.0, 0.0, 33.0, 1.0, 1.0, 33.0, 1.0, 2.0,
    ];
    assert_eq!(&p[..18], &want);
    assert!(p[18..].iter().all(|v| *v == 0.0));
}

#[test]
fn padding_appends_the_pad_token() {
    let x = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let token = mlxcel_core::from_slice_f32(&[9.0, 8.0], &[1, 2]);
    let out = pad_tokens(&x, &token);
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 32, 2]);
    let values = crate::imagegen::llada_image::conditioning::tests::host(&out);
    assert_eq!(&values[..6], &[1.0, 2.0, 3.0, 4.0, 9.0, 8.0]);
    assert_eq!(&values[62..], &[9.0, 8.0]);
}
