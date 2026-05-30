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

//! Pure-Rust wrappers around frequently reused MLX FFI entry points.
//!
//! These helpers keep small pointer-shaping and scalar-materialization logic
//! outside the `cxx::bridge` root so the FFI boundary stays focused on raw
//! bindings instead of policy code.

use crate::ffi;
use cxx::UniquePtr;

/// Safe wrapper to concatenate two arrays along an axis.
///
/// Used by: cache state machines, Mamba/Jamba/Nemotron-H SSM paths, Qwen VL,
/// Gemma3n multimodal helpers, Phi3V vision path, assorted shared model code.
pub fn concatenate(a: &ffi::MlxArray, b: &ffi::MlxArray, axis: i32) -> UniquePtr<ffi::MlxArray> {
    let ptrs: [*const ffi::MlxArray; 2] = [a as *const ffi::MlxArray, b as *const ffi::MlxArray];
    unsafe { ffi::concatenate(&ptrs, axis) }
}

/// Stack arrays along a new axis from raw MLX pointers.
///
/// Used by: shared MoE helpers and model implementations that already manage
/// their own pointer collections.
pub fn stack(ptrs: &[*const ffi::MlxArray], axis: i32) -> UniquePtr<ffi::MlxArray> {
    unsafe { ffi::stack(ptrs, axis) }
}

/// Stack arrays along a new axis from owned MLX buffers.
///
/// Used by: quantized MoE expert packing and vision position helper paths.
pub fn stack_owned(arrays: &[UniquePtr<ffi::MlxArray>], axis: i32) -> UniquePtr<ffi::MlxArray> {
    let ptrs: Vec<*const ffi::MlxArray> = arrays
        .iter()
        .map(|array| array.as_ref().unwrap() as *const ffi::MlxArray)
        .collect();
    unsafe { ffi::stack(&ptrs, axis) }
}

/// Multiply an array by a scalar by materializing the scalar once as MLX data.
///
/// Used by: shared softcap helpers, Gemma/DeepSeek/MiniCPM scaling, vision
/// merge paths, and other policy code that needs scalar broadcasting.
pub fn multiply_scalar(a: &ffi::MlxArray, scalar: f32) -> UniquePtr<ffi::MlxArray> {
    // Use input dtype for scalar to avoid float32 type promotion
    let a_dtype = ffi::array_dtype(a);
    let scalar_array = ffi::full_f32(&[1], scalar, a_dtype);
    ffi::multiply(a, &scalar_array)
}

/// Divide an array by a scalar by materializing the scalar once as MLX data.
///
/// Used by: shared softcap helpers and model families with output scaling.
pub fn divide_scalar(a: &ffi::MlxArray, scalar: f32) -> UniquePtr<ffi::MlxArray> {
    let a_dtype = ffi::array_dtype(a);
    let scalar_array = ffi::full_f32(&[1], scalar, a_dtype);
    ffi::divide(a, &scalar_array)
}

/// Walsh–Hadamard Transform along the last axis (B0 spike — TurboQuant KV cache compression).
///
/// Applies an orthonormal Hadamard transform to the last axis of `x`, scaled by
/// `1/sqrt(N)` where `N` is the size of the last axis. This is the "graph-only"
/// path the spike validates: it delegates to the MLX-native `hadamard_transform`
/// op which dispatches to the platform's optimized backend (Metal kernel on
/// Apple Silicon, the same one used by community Hadamard-quantization codepaths
/// upstream).
///
/// **Supported head_dim**: powers of 2 only — `{64, 128, 256}` covers the
/// production model matrix in mlxcel. The MLX op also accepts `m * 2^k` for
/// `m ∈ {12, 20, 28}` (i.e. `{80, 96, 192, …}`) but uses a different
/// normalization for those, so the involution `wht(wht(x)) ≈ x` does not hold
/// — they are intentionally out of scope per the issue body's "Out of scope"
/// statement and current model coverage.
///
/// This is the foundational op that the asymmetric KV compression sub-issues
/// (B2 onward) compose with random sign flips to form the structured rotation
/// `D2 · H · D1 · x`. The sign-flip wiring is intentionally out of scope here.
///
/// # Panics
///
/// Panics if the last axis size is 0 or not a power of 2. The MLX C++ side
/// throws `std::invalid_argument` for unsupported sizes, but the cxx-generated
/// extern is `noexcept` so an unguarded thrown exception aborts the process via
/// `std::terminate`. Validating in Rust gives callers a recoverable panic with
/// a clear message instead. (Production models all use power-of-2 head_dims so
/// this path is unreachable in normal usage; the guard exists so future B2
/// integration can rely on a sane error contract.)
///
/// Used by: TurboQuant cache compression (planned: cache::turbo, sub-– under).
pub fn wht(x: &ffi::MlxArray) -> UniquePtr<ffi::MlxArray> {
    let shape = ffi::array_shape(x);
    let last = shape.last().copied().unwrap_or(0);
    assert!(
        last > 0 && (last as u32).is_power_of_two(),
        "wht: last axis must be a non-zero power of 2; got shape={shape:?}"
    );
    ffi::hadamard_transform(x)
}
