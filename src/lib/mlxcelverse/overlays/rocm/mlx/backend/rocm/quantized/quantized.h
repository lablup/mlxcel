// Copyright © 2025 Apple Inc.

#pragma once

#include <optional>
#include "mlx/array.h"
#include "mlx/backend/rocm/device.h"

namespace mlx::core {

// Affine quantization functions
void affine_quantize(
    const array& w,
    array& wq,
    array& scales,
    array& biases,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s);

void affine_dequantize(
    const array& wq,
    const array& scales,
    const std::optional<array>& biases,
    array& w,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s);

// Floating-point quantization functions
void fp_quantize(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s);

void fp_dequantize(
    const array& wq,
    const array& scales,
    array& w,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s);

// NVFP4 (E2M1 values, E4M3 scales per 16) dequantize into w's dtype, exact
// (mlxcelverse, lablup/mlxcel#2698). Defined in qmm.hip next to the NVFP4
// matmul kernels.
void nvfp4_dequantize(
    const array& wq,
    const array& scales,
    array& w,
    rocm::CommandEncoder& enc);

} // namespace mlx::core
