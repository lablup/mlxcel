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

/// Match upstream `nn.Linear` only for immutable Objects checkpoint layers.
/// Other consumers retain core Linear's adapter and separate-bias semantics.
pub(super) fn linear_with_policy(
    linear: &mlxcel_core::layers::Linear,
    x: &MlxArray,
    objects: bool,
) -> UniquePtr<MlxArray> {
    if !objects {
        return linear.forward(x);
    }
    let weight = mlxcel_core::transpose(&linear.weight);
    match &linear.bias {
        Some(bias) => mlxcel_core::addmm(bias, x, &weight, 1.0, 1.0),
        None => mlxcel_core::matmul(x, &weight),
    }
}

impl super::Attention {
    /// Preserve upstream BF16 rounding in both QKV and output projections.
    pub fn forward_objects(&self, x: &MlxArray) -> anyhow::Result<UniquePtr<MlxArray>> {
        self.forward_with_policy(x, true)
    }
}

impl super::Mlp {
    /// Upstream biased projections and compiled weak-scalar GELU for Objects.
    pub fn forward_objects(&self, x: &MlxArray) -> anyhow::Result<UniquePtr<MlxArray>> {
        let projected = linear_with_policy(&self.fc1, x, true);
        let hidden = mlxcel_core::compiled_objects_gelu(&projected)?;
        Ok(linear_with_policy(&self.fc2, &hidden, true))
    }
}

#[cfg(test)]
#[path = "dinov2_policy_tests.rs"]
mod tests;

#[cfg(test)]
impl super::Block {
    pub(crate) fn objects_stage_diagnostics(
        &self,
        tokens: &MlxArray,
    ) -> anyhow::Result<std::collections::BTreeMap<String, UniquePtr<MlxArray>>> {
        let normalized = norm_with_policy(&self.norm1, tokens, true);
        let fused = linear_with_policy(&self.attention.qkv, &normalized, true);
        let separate = linear_with_policy(&self.attention.qkv, &normalized, false);
        Ok(std::collections::BTreeMap::from([
            ("norm1".into(), normalized),
            ("qkv_fused".into(), fused),
            ("qkv_separate".into(), separate),
            (
                "block0".into(),
                self.forward_with_norm_policy(tokens, true)?,
            ),
        ]))
    }
}
