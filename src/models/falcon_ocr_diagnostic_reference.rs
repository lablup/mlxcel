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

use std::path::Path;

use safetensors::{Dtype, SafeTensors};

pub(super) fn reference_tensor_f32(path: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).expect("read Python Falcon reference");
    let tensors = SafeTensors::deserialize(&bytes).expect("parse Python Falcon reference");
    let tensor = tensors.tensor(name).expect("reference tensor exists");
    assert_eq!(tensor.dtype(), Dtype::F32);
    let values = tensor
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four F32 bytes")))
        .collect();
    (tensor.shape().to_vec(), values)
}

pub(super) fn reference_tensor_i64(path: &Path, name: &str) -> Vec<i32> {
    let bytes = std::fs::read(path).expect("read Python Falcon reference");
    let tensors = SafeTensors::deserialize(&bytes).expect("parse Python Falcon reference");
    let tensor = tensors.tensor(name).expect("reference tensor exists");
    assert_eq!(tensor.dtype(), Dtype::I64);
    tensor
        .data()
        .chunks_exact(8)
        .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("eight I64 bytes")) as i32)
        .collect()
}

pub(super) fn patchify_nhwc(values: &[f32], height: usize, width: usize) -> Vec<f32> {
    let patch = 16;
    let mut patched = Vec::with_capacity(values.len());
    for patch_y in 0..height / patch {
        for patch_x in 0..width / patch {
            for y in 0..patch {
                for x in 0..patch {
                    let pixel = ((patch_y * patch + y) * width + patch_x * patch + x) * 3;
                    patched.extend_from_slice(&values[pixel..pixel + 3]);
                }
            }
        }
    }
    patched
}
