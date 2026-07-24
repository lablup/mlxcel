//! Shared exact math for the Gemma3n audio StableHLO graph.

use super::builder::{Builder, Val};

pub(super) fn scalar_like(b: &mut Builder, value: f32, x: &Val) -> Val {
    let scalar = b.const_f32(value);
    b.broadcast(&scalar, &[], x.ty.shape.clone())
}

pub(super) fn zeros_like(b: &mut Builder, x: &Val) -> Val {
    scalar_like(b, 0.0, x)
}

pub(super) fn round_bf16(b: &mut Builder, x: &Val) -> Val {
    let narrow = b.convert(x, "bf16");
    b.convert(&narrow, "f32")
}

pub(super) fn clip(b: &mut Builder, x: &Val, limit: f32) -> Val {
    let low = scalar_like(b, -limit, x);
    let high = scalar_like(b, limit, x);
    let x = b.maximum(x, &low);
    b.minimum(&x, &high)
}

pub(super) fn relu(b: &mut Builder, x: &Val) -> Val {
    let zero = zeros_like(b, x);
    b.maximum(x, &zero)
}

pub(super) fn sigmoid(b: &mut Builder, x: &Val) -> Val {
    let negative = b.negate(x);
    let exponential = b.exponential(&negative);
    let one = scalar_like(b, 1.0, x);
    let denominator = b.add(&one, &exponential);
    b.divide(&one, &denominator)
}

pub(super) fn silu(b: &mut Builder, x: &Val) -> Val {
    let gate = sigmoid(b, x);
    b.multiply(x, &gate)
}

pub(super) fn softplus(b: &mut Builder, x: &Val) -> Val {
    let exponential = b.exponential(x);
    let one = scalar_like(b, 1.0, x);
    let sum = b.add(&one, &exponential);
    b.logarithm(&sum)
}

pub(super) fn rms_norm(b: &mut Builder, x: &Val, weight: Option<&Val>, eps: f32) -> Val {
    let last = x.ty.shape.len() - 1;
    let width = x.ty.shape[last];
    let square = b.multiply(x, x);
    let zero = b.const_f32(0.0);
    let sum = b.reduce_add(&square, last, &zero);
    let divisor = scalar_like(b, width as f32, &sum);
    let mean = b.divide(&sum, &divisor);
    let epsilon = scalar_like(b, eps, &mean);
    let stabilized = b.add(&mean, &epsilon);
    let inverse = b.rsqrt(&stabilized);
    let dims: Vec<usize> = (0..last).collect();
    let inverse = b.broadcast(&inverse, &dims, x.ty.shape.clone());
    let normalized = b.multiply(x, &inverse);
    match weight {
        Some(weight) => {
            let weight = b.broadcast(weight, &[last], x.ty.shape.clone());
            b.multiply(&normalized, &weight)
        }
        None => normalized,
    }
}

pub(super) fn cumsum_time_f32(b: &mut Builder, x: &Val) -> Val {
    assert_eq!(x.ty.shape.len(), 4);
    b.reduce_window_prefix_add(x, 1)
}

pub(super) fn softmax_last(b: &mut Builder, x: &Val) -> Val {
    let last = x.ty.shape.len() - 1;
    let negative_infinity = b.const_f32(f32::NEG_INFINITY);
    let maximum = b.reduce_max(x, last, &negative_infinity);
    let dims: Vec<usize> = (0..last).collect();
    let maximum = b.broadcast(&maximum, &dims, x.ty.shape.clone());
    let centered = b.subtract(x, &maximum);
    let exponential = b.exponential(&centered);
    let zero = b.const_f32(0.0);
    let denominator = b.reduce_add(&exponential, last, &zero);
    let denominator = b.broadcast(&denominator, &dims, x.ty.shape.clone());
    b.divide(&exponential, &denominator)
}

pub(super) fn stride_time(b: &mut Builder, x: &Val, stride: usize) -> Val {
    let ranges =
        x.ty.shape
            .iter()
            .enumerate()
            .map(|(axis, &limit)| {
                if axis == 1 {
                    (0, limit, stride)
                } else {
                    (0, limit, 1)
                }
            })
            .collect::<Vec<_>>();
    b.slice_strided(x, &ranges)
}

pub(super) fn concat_axis(b: &mut Builder, values: &[Val], axis: usize) -> Val {
    let mut result = values[0].clone();
    for value in &values[1..] {
        result = b.concatenate(&result, value, axis);
    }
    result
}

pub(super) fn zero_invalid(b: &mut Builder, x: &Val, valid: &Val) -> Val {
    let mut shape = valid.ty.shape.clone();
    while shape.len() < x.ty.shape.len() {
        shape.push(1);
    }
    let valid = b.reshape(valid, shape);
    let dims = (0..valid.ty.shape.len()).collect::<Vec<_>>();
    let valid = b.broadcast(&valid, &dims, x.ty.shape.clone());
    let zero = zeros_like(b, x);
    b.select(&valid, x, &zero)
}
