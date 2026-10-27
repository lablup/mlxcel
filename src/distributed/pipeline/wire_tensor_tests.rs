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

const FLOAT16: i32 = 9;
const FLOAT32: i32 = 10;
const BFLOAT16: i32 = 12;

/// A `[1, 3, 4]` activation-shaped tensor in `dtype`, with values that are
/// not exactly representable in half precision so a widening or narrowing
/// cast anywhere on the wire path would change the bytes.
fn activation(dtype: i32) -> UniquePtr<MlxArray> {
    let values: Vec<f32> = (0..12).map(|i| (i as f32 - 5.3) * 0.731).collect();
    let base = mlxcel_core::from_slice_f32(&values, &[1, 3, 4]);
    let arr = mlxcel_core::astype(&base, dtype);
    mlxcel_core::eval(&arr);
    arr
}

fn assert_round_trip_is_exact(original: &MlxArray) {
    let wire = serialize_mlx_array(original).expect("serialize");
    let restored = deserialize_wire_tensor(&wire).expect("deserialize");
    assert_eq!(
        mlxcel_core::array_dtype(&restored),
        mlxcel_core::array_dtype(original),
        "a stage boundary must not change the activation dtype"
    );
    assert_eq!(
        mlxcel_core::array_shape(&restored),
        mlxcel_core::array_shape(original)
    );
    assert_eq!(
        mlxcel_core::array_to_raw_bytes(&restored),
        mlxcel_core::array_to_raw_bytes(original),
        "a stage boundary must hand the next stage the exact bytes"
    );
}

/// Issue #2654: on CUDA the float16 branch went through the checkpoint
/// loader `from_bytes_f16`, which widens to float32, so a float16 model's
/// downstream stage computed in float32 and its logits left the dense ones.
#[test]
fn float16_activation_round_trips_in_float16() {
    assert_round_trip_is_exact(&activation(FLOAT16));
}

#[test]
fn bfloat16_activation_round_trips_in_bfloat16() {
    assert_round_trip_is_exact(&activation(BFLOAT16));
}

#[test]
fn float32_activation_round_trips_in_float32() {
    assert_round_trip_is_exact(&activation(FLOAT32));
}

#[test]
fn token_ids_round_trip_in_int32() {
    let ids = mlxcel_core::from_slice_i32(&[128000, 9906, 13], &[1, 3]);
    assert_round_trip_is_exact(&ids);
}
