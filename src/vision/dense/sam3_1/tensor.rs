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

use mlxcel_core::{MlxArray, UniquePtr};

use super::super::interpolate::resize_bilinear_nhwc;

pub(super) fn add_scalar(array: &MlxArray, value: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::add(
        array,
        &mlxcel_core::full_f32(&[1], value, mlxcel_core::array_dtype(array)),
    )
}

pub(super) fn resize_masks_nchw(
    masks: &MlxArray,
    size: (usize, usize),
    antialias: bool,
) -> Result<UniquePtr<MlxArray>, String> {
    let shape = mlxcel_core::array_shape(masks);
    let flattened = mlxcel_core::reshape(masks, &[shape[0] * shape[1], shape[2], shape[3], 1]);
    let resized = resize_bilinear_nhwc(&flattened, size, false, antialias)
        .map_err(|error| error.to_string())?;
    Ok(mlxcel_core::reshape(
        &resized,
        &[shape[0], shape[1], size.0 as i32, size.1 as i32],
    ))
}

pub(super) fn gather_mask_candidate(masks: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(masks);
    let indices = mlxcel_core::broadcast_to(
        &mlxcel_core::expand_dims_multi(indices, &[1, 2, 3]),
        &[shape[0], 1, shape[2], shape[3]],
    );
    mlxcel_core::take_along_axis(masks, &indices, 1)
}

pub(super) fn mask_presence(masks: &MlxArray) -> UniquePtr<MlxArray> {
    let present = mlxcel_core::greater(masks, &mlxcel_core::zeros_like(masks));
    let present = mlxcel_core::any_axis(&present, 3, false);
    let present = mlxcel_core::any_axis(&present, 2, false);
    mlxcel_core::astype(
        &mlxcel_core::any_axis(&present, 1, true),
        mlxcel_core::dtype::FLOAT32,
    )
}
