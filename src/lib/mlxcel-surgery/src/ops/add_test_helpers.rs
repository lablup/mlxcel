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

//! Shared fixtures for `AddOp` unit tests.
//!
//! Split out so `add_tests.rs` (error-case unit tests) and
//! `add_apply_tests.rs` (correctness unit tests) can both reuse
//! `OwnedTensor`, the safetensors writer, and the MLX-array
//! conversion helpers without exceeding the project's
//! 500-line-per-file budget.
//!
//! Used by: `add_tests`, `add_apply_tests`.

use std::path::Path;

use mlxcel_core::dtype as mlx_dtype;
use mlxcel_core::{array_to_raw_bytes, eval, from_bytes, MlxArray, UniquePtr};
use safetensors::tensor::Dtype as SafeTensorDtype;
use safetensors::View;

/// A `safetensors::View` impl over owned bytes — copied from the
/// pattern used in `src/distributed/pipeline/partial_loading_adapter_tests.rs`.
#[derive(Clone)]
pub(crate) struct OwnedTensor {
    pub dtype: SafeTensorDtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl View for &OwnedTensor {
    fn dtype(&self) -> SafeTensorDtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        self.data.as_slice().into()
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Build an `OwnedTensor` of f32 from a flat slice.
pub(crate) fn f32_tensor(values: &[f32], shape: &[usize]) -> OwnedTensor {
    let mut data = Vec::with_capacity(values.len() * 4);
    for v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    OwnedTensor {
        dtype: SafeTensorDtype::F32,
        shape: shape.to_vec(),
        data,
    }
}

/// Write a single-tensor safetensors file under `dir` named
/// `donor.safetensors`. Returns the file path.
pub(crate) fn write_single_donor(dir: &Path, key: &str, tensor: OwnedTensor) -> std::path::PathBuf {
    let mut tensors = std::collections::HashMap::new();
    tensors.insert(key.to_string(), tensor);
    let path = dir.join("donor.safetensors");
    safetensors::serialize_to_file(&tensors, None, &path).expect("serialize donor safetensors");
    path
}

/// Build a fresh f32 MLX array from a flat slice plus its shape.
pub(crate) fn mlx_f32(values: &[f32], shape: &[usize]) -> UniquePtr<MlxArray> {
    let shape_i32: Vec<i32> = shape.iter().map(|d| *d as i32).collect();
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    from_bytes(&bytes, &shape_i32, mlx_dtype::FLOAT32)
}

/// Read an f32 MLX array back to a `Vec<f32>` so tests can assert
/// numerical content.
pub(crate) fn extract_f32(arr: &UniquePtr<MlxArray>) -> Vec<f32> {
    eval(arr);
    let bytes = array_to_raw_bytes(arr);
    assert_eq!(bytes.len() % 4, 0, "f32 must be 4-byte aligned");
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(chunk);
        out.push(f32::from_le_bytes(buf));
    }
    out
}
