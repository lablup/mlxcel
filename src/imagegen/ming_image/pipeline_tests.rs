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
fn the_noise_is_a_seeded_single_frame_latent() {
    let a = initial_noise(0, 16, 512, 256);
    assert_eq!(mlxcel_core::array_shape(&a), [1, 16, 1, 32, 64]);
    assert_eq!(mlxcel_core::array_dtype(&a), dtype::FLOAT32);
    let same = crate::imagegen::testing::values(&initial_noise(0, 16, 512, 256));
    let other = crate::imagegen::testing::values(&initial_noise(1, 16, 512, 256));
    let a = crate::imagegen::testing::values(&a);
    assert_eq!(a, same);
    assert_ne!(a, other);
}
