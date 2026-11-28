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

//! Rotary-lane scaling for YaRN's attention factor under partial RoPE.
//!
//! transformers' YaRN multiplies the cos/sin tables by the attention factor,
//! so only the rotated lanes of Q and K are scaled; the pass-through lanes of
//! a partial-rotary head keep their magnitude. `fast_rope_with_freqs` has no
//! such multiplier, so callers scale the rotated lanes before rotating.
//!
//! Shared by the Qwen 3.5 target attention (`Qwen3NextAttention` in the
//! binary crate) and the Qwen 3.5 MTP drafter layer
//! ([`crate::drafter::qwen3_5_mtp::layer`]), so both apply the identical
//! operation (issue #3479).

use crate::{MlxArray, UniquePtr, array_shape, concatenate, copy, multiply_scalar, slice};

/// Multiply the first `rope_dims` lanes of a `[.., D]` Q or K block by
/// `attention_factor` and leave the pass-through lanes as they are. The
/// scalar is materialized in the block's own dtype. A factor of exactly 1.0
/// returns a copy without arithmetic.
pub fn scale_rotary_lanes(
    x: &MlxArray,
    rope_dims: i32,
    attention_factor: f32,
) -> UniquePtr<MlxArray> {
    if attention_factor == 1.0 {
        return copy(x);
    }
    let shape = array_shape(x);
    let last = shape.len() - 1;
    if rope_dims >= shape[last] {
        return multiply_scalar(x, attention_factor);
    }
    let start = vec![0; shape.len()];
    let mut rotary_end = shape.clone();
    rotary_end[last] = rope_dims;
    let mut pass_start = start.clone();
    pass_start[last] = rope_dims;
    let rotary = slice(x, &start, &rotary_end);
    let pass = slice(x, &pass_start, &shape);
    let rotary = multiply_scalar(&rotary, attention_factor);
    concatenate(&rotary, &pass, -1)
}
