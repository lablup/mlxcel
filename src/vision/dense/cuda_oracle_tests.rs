// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Upstream MLX CUDA values for pinned oracles that CUDA's fused kernels round
//! differently (issue #2739).
//!
//! The Objects layer, tanh GELU and depth bicubic oracles were captured on
//! ROCm and x86_64. On CUDA, upstream `mlx` 0.32.4 itself returns other values
//! for some elements: its fused bf16 kernels and f32 resize arithmetic round a
//! few results differently (the elements are listed in
//! `tests/fixtures/sam3d_objects/cuda-oracle.json`, written by
//! `generate_cuda_oracle.py` on an NVIDIA GB10, sm_121, aarch64, with
//! `MLX_ENABLE_TF32=0`). Forcing f32 intermediates does not reproduce the
//! pinned values either (13 of the 32 modulate elements differ that way, 11
//! with the compiled graph), so the oracle is keyed on CUDA: mlxcel must match
//! upstream CUDA bit for bit there, and the pinned oracle everywhere else.
use mlxcel_core::{self as mx, MlxArray, UniquePtr};

/// Whether the default device is a CUDA GPU.
pub(crate) fn on_cuda() -> bool {
    mx::default_device_is_gpu()
        && mx::hardware::gpu_backend_kind() == mx::hardware::GpuBackendKind::Cuda
}

/// `pinned` with the CUDA overrides recorded under `key`, on CUDA; a copy of
/// `pinned` on every other device. Dtype and shape follow `pinned`.
pub(crate) fn keyed(key: &str, pinned: &MlxArray) -> UniquePtr<MlxArray> {
    if !on_cuda() {
        return mx::copy(pinned);
    }
    let document: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/sam3d_objects/cuda-oracle.json"
    ))
    .expect("CUDA oracle is valid JSON");
    let entry = &document[key];
    let mut values = mx::utils::array_to_vec_f32(pinned);
    assert_eq!(
        entry["count"].as_u64(),
        Some(values.len() as u64),
        "CUDA oracle {key} covers a different element count"
    );
    let index = entry["index"].as_array().expect("override indices");
    let value = entry["value"].as_array().expect("override values");
    assert_eq!(index.len(), value.len(), "CUDA oracle {key}");
    for (i, v) in index.iter().zip(value) {
        values[i.as_u64().expect("index") as usize] = v.as_f64().expect("value") as f32;
    }
    let shape = mx::array_shape(pinned);
    mx::astype(
        &mx::from_slice_f32(&values, &shape),
        mx::array_dtype(pinned),
    )
}
