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

use super::builder::{Builder, Val};

pub(super) fn scalar_broadcast(builder: &mut Builder, value: f32, shape: Vec<usize>) -> Val {
    let scalar = builder.const_f32(value);
    builder.broadcast(&scalar, &[], shape)
}

pub(super) fn softmax_last(builder: &mut Builder, scores: &Val) -> Val {
    let last = scores.ty.shape.len() - 1;
    let negative_infinity = builder.const_f32(f32::NEG_INFINITY);
    let maximum = builder.reduce_max(scores, last, &negative_infinity);
    let mut broadcast_shape = maximum.ty.shape.clone();
    broadcast_shape.push(scores.ty.shape[last]);
    let dimensions = (0..maximum.ty.shape.len()).collect::<Vec<_>>();
    let maximum = builder.broadcast(&maximum, &dimensions, broadcast_shape);
    let shifted = builder.subtract(scores, &maximum);
    let exponentials = builder.exponential(&shifted);
    let zero = builder.const_f32(0.0);
    let denominator = builder.reduce_add(&exponentials, last, &zero);
    let mut broadcast_shape = denominator.ty.shape.clone();
    broadcast_shape.push(scores.ty.shape[last]);
    let dimensions = (0..denominator.ty.shape.len()).collect::<Vec<_>>();
    let denominator = builder.broadcast(&denominator, &dimensions, broadcast_shape);
    builder.divide(&exponentials, &denominator)
}

pub(super) fn silu(builder: &mut Builder, value: &Val) -> Val {
    let negated = builder.negate(value);
    let exponential = builder.exponential(&negated);
    let one = scalar_broadcast(builder, 1.0, value.ty.shape.clone());
    let denominator = builder.add(&one, &exponential);
    builder.divide(value, &denominator)
}
