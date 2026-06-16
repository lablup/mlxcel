// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Dtype constants matching MLX's C++ enum layout.

pub const BOOL: i32 = 0;
pub const UINT8: i32 = 1;
pub const UINT16: i32 = 2;
pub const UINT32: i32 = 3;
pub const UINT64: i32 = 4;
pub const INT8: i32 = 5;
pub const INT16: i32 = 6;
pub const INT32: i32 = 7;
pub const INT64: i32 = 8;
pub const FLOAT16: i32 = 9;
pub const FLOAT32: i32 = 10;
pub const FLOAT64: i32 = 11;
pub const BFLOAT16: i32 = 12;
pub const COMPLEX64: i32 = 13;

/// Bytes per element for an MLX dtype code, or `None` for unknown codes.
pub fn size_bytes(dtype: i32) -> Option<usize> {
    match dtype {
        BOOL | UINT8 | INT8 => Some(1),
        UINT16 | INT16 | FLOAT16 | BFLOAT16 => Some(2),
        UINT32 | INT32 | FLOAT32 => Some(4),
        UINT64 | INT64 | FLOAT64 | COMPLEX64 => Some(8),
        _ => None,
    }
}
