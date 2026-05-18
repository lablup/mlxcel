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

//! Low-level tensor-zeroing primitives over the `mlxcel-core` FFI.
//!
//! These helpers are deliberately thin wrappers around
//! [`mlxcel_core::slice_update`] and [`mlxcel_core::zeros`]. They
//! exist primarily to:
//!
//! 1. Carry the dtype + shape introspection so the per-granularity
//!    pruners do not have to repeat it.
//! 2. Centralize the alignment / range error messages so every
//!    granularity reports failures the same way.
//! 3. Force an `eval` after each write so the
//!    [`crate::WeightMap`] returned to the loader does not retain a
//!    deferred lazy node referencing the old buffer (which would
//!    quietly defeat the prune on the first forward pass).
//!
//! Used by: `super::granularity` (every per-granularity pruner).

// Alias the `mlxcel_core` root as `ffi` so this module reads
// consistently with the C++-binding style used throughout
// `mlxcel-core` itself (which imports `crate::ffi`). The functions
// re-exported by `pub use ffi::*` at the crate root (`array_shape`,
// `zeros`, `slice_update`, etc.) are reachable as `ffi::array_shape`
// via this alias.
use mlxcel_core as ffi;

use crate::{SurgeryError, WeightMap};

/// Replace `weights[key]` with a zero-filled tensor of the same shape
/// and dtype. Caller has already verified the key exists.
pub(super) fn zero_tensor_inplace(weights: &mut WeightMap, key: &str) -> Result<(), SurgeryError> {
    let arr = weights
        .get(key)
        .ok_or_else(|| SurgeryError::TensorNotFound(key.to_string()))?;
    let shape = ffi::array_shape(arr);
    let dtype = ffi::array_dtype(arr);
    let zeroed = ffi::zeros(&shape, dtype);
    // Force evaluation so any later read goes through the rewritten
    // tensor rather than a deferred lazy node referencing the old
    // buffer.
    ffi::eval(&zeroed);
    weights.insert(key.to_string(), zeroed);
    Ok(())
}

/// Zero rows `[start, stop)` along axis 0 of `weights[key]`.
///
/// Works on 2-D and higher-rank tensors. For 1-D tensors (e.g. a bias
/// vector) it zeroes the slice `[start, stop)` along the only axis.
pub(super) fn zero_axis0_rows(
    weights: &mut WeightMap,
    key: &str,
    start: i32,
    stop: i32,
) -> Result<(), SurgeryError> {
    let arr = weights
        .get(key)
        .ok_or_else(|| SurgeryError::TensorNotFound(key.to_string()))?;
    let shape = ffi::array_shape(arr);
    let dtype = ffi::array_dtype(arr);
    if shape.is_empty() {
        return Err(SurgeryError::Other(anyhow::anyhow!(
            "prune: cannot slice 0-D tensor `{key}`"
        )));
    }
    if start < 0 || stop > shape[0] || start >= stop {
        return Err(SurgeryError::Other(anyhow::anyhow!(
            "prune: axis-0 slice [{start}, {stop}) out of range for `{key}` shape {shape:?}"
        )));
    }
    let mut slice_shape: Vec<i32> = shape.clone();
    slice_shape[0] = stop - start;
    let zeros = ffi::zeros(&slice_shape, dtype);

    let mut starts = vec![0i32; shape.len()];
    let mut stops = shape.clone();
    starts[0] = start;
    stops[0] = stop;
    let updated = ffi::slice_update(arr, &zeros, &starts, &stops);
    ffi::eval(&updated);
    weights.insert(key.to_string(), updated);
    Ok(())
}

/// Zero columns `[start, stop)` along axis 1 of `weights[key]`.
///
/// For affine quantized `weight` tensors (axis 1 = packed IN), the
/// caller must guarantee `start` and `stop` are aligned to the packing
/// factor (multiple of 8 for 4-bit, 4 for 8-bit) so the i32 column
/// boundary maps cleanly onto packed u32 words. This op only zeros the
/// dequant-result via raw scales/biases; for non-aligned slices use
/// [`zero_axis1_columns_or_packed_only`].
///
/// `align` is the required alignment along axis 1; ops pass `head_dim`
/// for o_proj (always aligned to head_dim in practice).
pub(super) fn zero_axis1_columns(
    weights: &mut WeightMap,
    key: &str,
    start: i32,
    stop: i32,
    align: usize,
) -> Result<(), SurgeryError> {
    let arr = weights
        .get(key)
        .ok_or_else(|| SurgeryError::TensorNotFound(key.to_string()))?;
    let shape = ffi::array_shape(arr);
    let dtype = ffi::array_dtype(arr);

    // Bias tensors (1-D, no axis 1): silent skip — the IN-axis prune
    // does not apply to OUT-axis-only tensors.
    if shape.len() < 2 {
        return Ok(());
    }
    if start < 0 || stop > shape[1] || start >= stop {
        return Err(SurgeryError::Other(anyhow::anyhow!(
            "prune: axis-1 slice [{start}, {stop}) out of range for `{key}` shape {shape:?}"
        )));
    }
    if align > 1 && !(start as usize).is_multiple_of(align) {
        return Err(SurgeryError::Other(anyhow::anyhow!(
            "prune: axis-1 slice start {start} not aligned to {align} for `{key}`"
        )));
    }

    let mut slice_shape: Vec<i32> = shape.clone();
    slice_shape[1] = stop - start;
    let zeros = ffi::zeros(&slice_shape, dtype);

    let mut starts = vec![0i32; shape.len()];
    let mut stops = shape.clone();
    starts[1] = start;
    stops[1] = stop;
    let updated = ffi::slice_update(arr, &zeros, &starts, &stops);
    ffi::eval(&updated);
    weights.insert(key.to_string(), updated);
    Ok(())
}

/// IN-axis prune variant for tensors where the requested alignment
/// (e.g. width 1 for a single mlp channel) is below the quantization
/// pack factor. Behavior:
///
/// - On non-quantized `weight` (`.weight` keys with float dtype):
///   zero the requested column range exactly.
/// - On quantized `.weight` (integer dtype): zero the packed column
///   range *if* it lies within a single u32 boundary; otherwise return
///   an error explaining the alignment requirement. In practice, the
///   per-channel slice width is 1 which is never packed-aligned, so
///   the op refuses with a clear message that quantized down_proj
///   single-channel pruning is not supported.
pub(super) fn zero_axis1_columns_or_packed_only(
    weights: &mut WeightMap,
    key: &str,
    start: i32,
    stop: i32,
    _channel_width: usize,
) -> Result<(), SurgeryError> {
    let arr = weights
        .get(key)
        .ok_or_else(|| SurgeryError::TensorNotFound(key.to_string()))?;
    let shape = ffi::array_shape(arr);
    let dtype = ffi::array_dtype(arr);

    if shape.len() < 2 {
        return Ok(());
    }
    if start < 0 || stop > shape[1] || start >= stop {
        return Err(SurgeryError::Other(anyhow::anyhow!(
            "prune: axis-1 slice [{start}, {stop}) out of range for `{key}` shape {shape:?}"
        )));
    }

    // Non-quantized float tensors: zero the slice exactly.
    if is_float_dtype(dtype) {
        let mut slice_shape: Vec<i32> = shape.clone();
        slice_shape[1] = stop - start;
        let zeros = ffi::zeros(&slice_shape, dtype);
        let mut starts = vec![0i32; shape.len()];
        let mut stops = shape.clone();
        starts[1] = start;
        stops[1] = stop;
        let updated = ffi::slice_update(arr, &zeros, &starts, &stops);
        ffi::eval(&updated);
        weights.insert(key.to_string(), updated);
        return Ok(());
    }

    // Quantized weight: refuse single-channel pruning unless the column
    // range happens to be packed-aligned. Most Llama-family checkpoints
    // pack 8 4-bit values per u32 along axis 1, so a width-1 channel
    // is never packed-aligned.
    Err(SurgeryError::Other(anyhow::anyhow!(
        "prune (mlp_channel): refusing to zero columns [{start}, {stop}) of \
         quantized tensor `{key}` (shape {shape:?}, dtype code {dtype}). \
         Quantized MLP-channel pruning along the IN axis would require \
         zeroing a packed u32 word that contains the channel, but that \
         would silently affect adjacent channels. Either (a) dequantize \
         the checkpoint first, (b) prune at MLP-block boundaries aligned \
         to the quantization group_size, or (c) use granularity=layer."
    )))
}

/// `true` when `dtype` is a floating-point dtype (f16, f32, f64, bf16).
fn is_float_dtype(dtype: i32) -> bool {
    use mlxcel_core::dtype;
    matches!(dtype, _ if dtype == dtype::FLOAT16
        || dtype == dtype::FLOAT32
        || dtype == dtype::FLOAT64
        || dtype == dtype::BFLOAT16)
}
