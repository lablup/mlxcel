// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use mlxcel_core::{MlxArray, UniquePtr, layers::LayerNorm};
/// SAM 3D normalizes F32 activations with checkpoint affine parameters, then restores the activation dtype.
pub(crate) fn norm_with_policy(
    norm: &LayerNorm,
    x: &MlxArray,
    upcast: bool,
) -> UniquePtr<MlxArray> {
    if !upcast {
        return norm.forward(x);
    }
    let y = norm.forward(&mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32));
    mlxcel_core::astype(&y, mlxcel_core::array_dtype(x))
}
