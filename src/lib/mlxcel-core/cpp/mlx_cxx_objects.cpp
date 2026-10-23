// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
// Exact compile boundaries from pinned Objects layers and bundled MoGe geometry.
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
    for (size_t i = 1; i <= other.ndim(); ++i) {
        const auto axis = -static_cast<int>(i);
        if (other.shape(axis) != 1 && other.shape(axis) != x.shape(axis)) {
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
// Pinned moge3/geometry.py: evaluate eagerly once, then compile only _lm_step.
// State order: shift, lambda, done, cost, gradient, Hessian, focal.
std::vector<array> focal_evaluate(
    const array& shift, const array& uv, const array& xy, const array& z) {
    using namespace mlx::core;
    const std::vector<int> axes{-2, -1};
    auto inv = divide(array(1.0f), add(z, shift));
    auto inv_last = expand_dims(inv, -1);
    auto p = multiply(xy, inv_last);
    auto q = multiply(negative(p), inv_last);
    auto a = sum(multiply(p, uv), axes, true);
    auto b = maximum(sum(multiply(p, p), axes, true), array(1e-12f));
    auto f = divide(a, b);
    auto da = sum(multiply(q, uv), axes, true);
    auto db = multiply(array(2.0f), sum(multiply(p, q), axes, true));
    auto df = divide(subtract(multiply(da, b), multiply(a, db)), multiply(b, b));
    auto r = subtract(multiply(f, p), uv);
    auto j = add(multiply(df, p), multiply(f, q));
    return {sum(multiply(r, r), axes), sum(multiply(j, r), axes),
            sum(multiply(j, j), axes), reshape(f, {-1})};
}
ObjectsFn& focal_step_fn() {
    static auto& fn = *new ObjectsFn(mlx::core::compile(
        [](const std::vector<array>& a) -> std::vector<array> {
            using namespace mlx::core;
            const auto& shift = a[0];
            const auto& lam = a[1];
            const auto& done = a[2];
            const auto& cost = a[3];
            auto step = divide(negative(a[4]),
                maximum(multiply(a[5], add(array(1.0f), lam)), array(1e-30f)));
            auto trial = add(shift, expand_dims(step, 1));
            auto next = focal_evaluate(trial, a[7], a[8], a[9]);
            auto accept = bitwise_and(less(next[0], cost), bitwise_invert(done));
            auto converged = bitwise_and(accept,
                less_equal(subtract(cost, next[0]), multiply(a[10], next[0])));
            auto tiny = less(abs(step), multiply(array(1e-8f),
                maximum(abs(squeeze(shift, 1)), array(1.0f))));
            auto new_shift = where(expand_dims(accept, 1), trial, shift);
            auto new_lam = where(done, lam,
                where(accept, divide(lam, array(3.0f)), multiply(lam, array(5.0f))));
            auto new_done = bitwise_or(bitwise_or(done, converged), tiny);
            return {new_shift, new_lam, new_done,
                    where(accept, next[0], cost), where(accept, next[1], a[4]),
                    where(accept, next[2], a[5]), where(accept, next[3], a[6])};
        }, /*shapeless=*/false));
    return fn;
}
array focal_solve(const array& uv, const array& xyz, bool trace) {
    using namespace mlx::core;
    if (uv.ndim() != 3 || xyz.ndim() != 3 || uv.dtype() != float32 ||
        xyz.dtype() != float32 || uv.shape(0) <= 0 || uv.shape(1) <= 0 ||
        uv.shape(0) != xyz.shape(0) || uv.shape(1) != xyz.shape(1) ||
        uv.shape(2) != 2 || xyz.shape(2) != 3) {
        throw std::invalid_argument("Objects focal solver requires F32 UV[B,N,2] and XYZ[B,N,3] with positive matching B,N");
    }
    const int batch = xyz.shape(0), count = xyz.shape(1);
    auto xy = slice(xyz, {0, 0, 0}, {batch, count, 2});
    auto z = squeeze(slice(xyz, {0, 0, 2}, {batch, count, 3}), -1);
    auto shift = zeros({batch, 1}, float32);
    auto initial = focal_evaluate(shift, uv, xy, z);
    std::vector<array> state{shift, full({batch}, 1e-3f, float32), zeros({batch}, bool_),
                            initial[0], initial[1], initial[2], initial[3]};
    const array ftol(1e-3f);
    std::vector<array> snapshots;
    auto record = [&]() {
        snapshots.push_back(stack({reshape(state[0], {-1}), state[1], astype(state[2], float32),
                                   state[3], state[4], state[5], state[6]}, -1));
    };
    if (trace) record();
    for (int iteration = 0; iteration < 30; ++iteration) {
        auto inputs = state;
        inputs.insert(inputs.end(), {uv, xy, z, ftol});
        state = focal_step_fn()(inputs);
        if (trace) record();
    }
    return trace ? stack(snapshots, 0) : stack({state[6], reshape(state[0], {-1})}, -1);
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
std::unique_ptr<MlxArray> compiled_objects_focal_shift(
    const MlxArray& uv, const MlxArray& xyz) {
    return std::make_unique<MlxArray>(focal_solve(uv.inner, xyz.inner, false));
}
std::unique_ptr<MlxArray> objects_focal_shift_states_for_test(
    const MlxArray& uv, const MlxArray& xyz) {
    return std::make_unique<MlxArray>(focal_solve(uv.inner, xyz.inner, true));
}
}  // namespace mlx_cxx
