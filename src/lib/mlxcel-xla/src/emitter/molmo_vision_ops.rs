use super::builder::{Builder, Val};

pub(super) fn scalar_broadcast(builder: &mut Builder, value: f32, shape: Vec<usize>) -> Val {
    let scalar = builder.const_f32(value);
    builder.broadcast(&scalar, &[], shape)
}

pub(super) fn tanh_gelu(builder: &mut Builder, value: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let half = scalar_broadcast(builder, 0.5, shape.clone());
    let one = scalar_broadcast(builder, 1.0, shape.clone());
    let coefficient = scalar_broadcast(builder, 0.044_715, shape.clone());
    let scale = scalar_broadcast(builder, 0.797_884_6, shape);
    let squared = builder.multiply(value, value);
    let cubed = builder.multiply(&squared, value);
    let nonlinear = builder.multiply(&coefficient, &cubed);
    let inner = builder.add(value, &nonlinear);
    let scaled = builder.multiply(&scale, &inner);
    let tanh = builder.tanh(&scaled);
    let cdf = builder.add(&one, &tanh);
    let half_value = builder.multiply(value, &half);
    builder.multiply(&half_value, &cdf)
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
