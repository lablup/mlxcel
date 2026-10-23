// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
// Exact compile boundaries from pinned sam3d_objects/layers.py.
#include "mlx_cxx_internal.h"
#include <cmath>
#include <functional>

namespace mlx_cxx {
namespace {
using ObjectsFn = std::function<std::vector<array>(const std::vector<array>&)>;
void floating(const array& x) {
    if (x.dtype() != float32 && x.dtype() != float16 && x.dtype() != bfloat16) {
        throw std::invalid_argument("Objects compiled layers require floating inputs");
    }
}
void broadcast_into(const array& x, const array& other) {
    floating(other);
    if (other.ndim() > x.ndim() || other.dtype() != x.dtype()) {
        throw std::invalid_argument("Objects modulation requires matching floating dtypes and broadcastable shapes");
    }
    for (int i = 1; i <= other.ndim(); ++i) {
        if (other.shape(-i) != 1 && other.shape(-i) != x.shape(-i)) {
            throw std::invalid_argument("Objects modulation operand cannot broadcast into input");
        }
    }
}
ObjectsFn& modulate_fn() {
    // MLX's process-wide CompilerCache outlives model instances but may be
    // destroyed before ordinary function statics; keep compiled handles alive.
    static auto& fn = *new ObjectsFn(mlx::core::compile(
        [](const std::vector<array>& a) -> std::vector<array> {
            auto scale = mlx::core::add(array(1, a[1].dtype()), a[1]);
            return {mlx::core::add(mlx::core::multiply(a[0], scale), a[2])};
        }, /*shapeless=*/true));
    return fn;
}
ObjectsFn& residual_fn() {
    static auto& fn = *new ObjectsFn(mlx::core::compile(
        [](const std::vector<array>& a) -> std::vector<array> {
            return {mlx::core::add(a[0], mlx::core::multiply(a[1], a[2]))};
        }, /*shapeless=*/true));
    return fn;
}
ObjectsFn& gelu_fn() {
    static auto& fn = *new ObjectsFn(mlx::core::compile(
        [](const std::vector<array>& a) -> std::vector<array> {
            const auto& x = a[0];
            // Python weak scalars adopt x.dtype before the compiled arithmetic.
            auto root_two = array(static_cast<float>(std::sqrt(2.0)), x.dtype());
            auto one = array(1, x.dtype());
            auto two = array(2, x.dtype());
            auto erf = mlx::core::erf(mlx::core::divide(x, root_two));
            auto scaled = mlx::core::multiply(x, mlx::core::add(one, erf));
            return {mlx::core::divide(scaled, two)};
        }, /*shapeless=*/true));
    return fn;
}
ObjectsFn& gelu_tanh_fn() {
    static auto& fn = *new ObjectsFn(mlx::core::compile(
        [](const std::vector<array>& a) -> std::vector<array> {
            const auto& x = a[0];
            // Preserve nn.gelu_approx's Power primitive: multiplying x twice
            // introduces an extra BF16 rounding even inside a compiled graph.
            auto cube = mlx::core::power(x, array(3, x.dtype()));
            auto cubic = mlx::core::multiply(array(0.044715f, x.dtype()), cube);
            auto inner = mlx::core::multiply(
                array(0.7978845608028654f, x.dtype()), mlx::core::add(x, cubic));
            auto cdf = mlx::core::add(array(1, x.dtype()), mlx::core::tanh(inner));
            auto half_x = mlx::core::multiply(array(0.5f, x.dtype()), x);
            return {mlx::core::multiply(half_x, cdf)};
        }, /*shapeless=*/true));
    return fn;
}
ObjectsFn& head_norm_fn() {
    static auto& fn = *new ObjectsFn(mlx::core::compile(
        [](const std::vector<array>& a) -> std::vector<array> {
            auto xf = mlx::core::astype(a[0], float32);
            auto squared = mlx::core::sum(mlx::core::multiply(xf, xf), -1, true);
            auto inverse = mlx::core::rsqrt(mlx::core::maximum(squared, array(1e-24f)));
            auto y = mlx::core::multiply(xf, inverse);
            y = mlx::core::multiply(y, a[1]);
            y = mlx::core::multiply(y, array(static_cast<float>(std::sqrt(a[0].shape(-1)))));
            return {mlx::core::astype(y, a[0].dtype())};
        }, /*shapeless=*/false));
    return fn;
}
}  // namespace

std::unique_ptr<MlxArray> compiled_objects_modulate(
    const MlxArray& x, const MlxArray& scale, const MlxArray& shift) {
    floating(x.inner);
    broadcast_into(x.inner, scale.inner);
    broadcast_into(x.inner, shift.inner);
    auto out = modulate_fn()({x.inner, scale.inner, shift.inner});
    return std::make_unique<MlxArray>(std::move(out[0]));
}
std::unique_ptr<MlxArray> compiled_objects_gated_residual(
    const MlxArray& x, const MlxArray& h, const MlxArray& gate) {
    floating(x.inner);
    broadcast_into(x.inner, h.inner);
    broadcast_into(x.inner, gate.inner);
    auto out = residual_fn()({x.inner, h.inner, gate.inner});
    return std::make_unique<MlxArray>(std::move(out[0]));
}
std::unique_ptr<MlxArray> compiled_objects_gelu(const MlxArray& x) {
    floating(x.inner);
    auto out = gelu_fn()({x.inner});
    return std::make_unique<MlxArray>(std::move(out[0]));
}
std::unique_ptr<MlxArray> compiled_objects_gelu_tanh(const MlxArray& x) {
    floating(x.inner);
    auto out = gelu_tanh_fn()({x.inner});
    return std::make_unique<MlxArray>(std::move(out[0]));
}
std::unique_ptr<MlxArray> compiled_objects_head_norm(
    const MlxArray& x, const MlxArray& gamma) {
    floating(x.inner);
    floating(gamma.inner);
    if (x.inner.ndim() != 4 || x.inner.shape(-1) <= 0 ||
        gamma.inner.shape() != Shape({x.inner.shape(-2), x.inner.shape(-1)}) ||
        gamma.inner.dtype() != x.inner.dtype()) {
        throw std::invalid_argument("Objects head norm expects BNHW input and matching HW gamma");
    }
    auto out = head_norm_fn()({x.inner, gamma.inner});
    return std::make_unique<MlxArray>(std::move(out[0]));
}
}  // namespace mlx_cxx
