//! Shared exact math for the Gemma3n audio StableHLO graph.

use super::builder::{Builder, Val};
use super::numeric_ops::mlx_cuda_prefix_sum_2d;

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

fn slice_axis(b: &mut Builder, x: &Val, axis: usize, start: usize, limit: usize) -> Val {
    let ranges =
        x.ty.shape
            .iter()
            .enumerate()
            .map(|(current, &size)| {
                if current == axis {
                    (start, limit)
                } else {
                    (0, size)
                }
            })
            .collect::<Vec<_>>();
    b.slice(x, &ranges)
}

fn gather_axis(b: &mut Builder, x: &Val, axis: usize, indices: &[usize]) -> Val {
    assert_eq!(indices.len(), x.ty.shape[axis]);
    let mut permutation = vec![axis];
    permutation.extend((0..x.ty.shape.len()).filter(|&current| current != axis));
    let transposed = b.transpose(x, &permutation);
    let width = transposed.ty.shape[0];
    let remainder = transposed.ty.shape[1..].iter().product::<usize>();
    let flattened = b.reshape(&transposed, vec![width, remainder]);
    let indices = indices
        .iter()
        .map(|&index| i32::try_from(index).expect("static axis index must fit i32"))
        .collect::<Vec<_>>();
    let indices = b.const_tensor_i32(&indices, vec![width, 1]);
    let gathered = b.gather(&flattened, &indices);
    let gathered = b.reshape(&gathered, transposed.ty.shape);
    let mut inverse = vec![0; permutation.len()];
    for (transposed_axis, original_axis) in permutation.into_iter().enumerate() {
        inverse[original_axis] = transposed_axis;
    }
    b.transpose(&gathered, &inverse)
}

/// Match MLX CUDA's f32 `row_reduce_simple` schedule for the SSCP normalization
/// widths. Each CUDA lane accumulates four adjacent values serially, then the
/// 32-lane tile combines its partials through XOR shuffles (16, 8, 4, 2, 1).
///
/// StableHLO reductions deliberately leave the tree to the backend. That is a
/// poor fit for the CUDA-produced MLX oracle: cancellation in cumulative group
/// normalization can move a BF16-rounded result when the CPU backend selects a
/// different tree. Keep the CUDA schedule local to this normalization instead
/// of changing graph-wide reductions.
pub(super) fn mlx_cuda_row_sum_f32(b: &mut Builder, x: &Val, axis: usize) -> Val {
    assert_eq!(x.ty.elt, "f32");
    assert!(axis < x.ty.shape.len());
    let width = x.ty.shape[axis];
    assert!(
        (1..=128).contains(&width),
        "MLX CUDA row-sum emulation supports one 32-lane block"
    );

    let padded = if width == 128 {
        x.clone()
    } else {
        let zero = b.const_f32(0.0);
        let low = vec![0; x.ty.shape.len()];
        let mut high = low.clone();
        high[axis] = 128 - width;
        b.pad(x, &zero, &low, &high)
    };
    let mut grouped_shape = padded.ty.shape.clone();
    grouped_shape[axis] = 32;
    grouped_shape.insert(axis + 1, 4);
    let grouped = b.reshape(&padded, grouped_shape);
    let lane0 = slice_axis(b, &grouped, axis + 1, 0, 1);
    let lane1 = slice_axis(b, &grouped, axis + 1, 1, 2);
    let lane2 = slice_axis(b, &grouped, axis + 1, 2, 3);
    let lane3 = slice_axis(b, &grouped, axis + 1, 3, 4);
    let zero = zeros_like(b, &lane0);
    let local0 = b.add(&zero, &lane0);
    let local1 = b.add(&local0, &lane1);
    let local2 = b.add(&local1, &lane2);
    let local3 = b.add(&local2, &lane3);

    let mut partial_shape = padded.ty.shape.clone();
    partial_shape[axis] = 32;
    let mut partial = b.reshape(&local3, partial_shape);
    for mask in [16, 8, 4, 2, 1] {
        let shuffled_indices = (0..32).map(|lane| lane ^ mask).collect::<Vec<_>>();
        let shuffled = gather_axis(b, &partial, axis, &shuffled_indices);
        partial = b.add(&partial, &shuffled);
    }

    let reduced = slice_axis(b, &partial, axis, 0, 1);
    let mut output_shape = x.ty.shape.clone();
    output_shape.remove(axis);
    b.reshape(&reduced, output_shape)
}

/// Match the one-block MLX CUDA contiguous inclusive-scan schedule used by the
/// maintained Gemma3n oracle. Released frame buckets produce at most 1,499
/// values after the first SSCP convolution, below CUDA's 4,096-value block
/// limit, so no inter-block prefix is needed.
pub(super) fn mlx_cuda_cumsum_time_f32(b: &mut Builder, x: &Val) -> Val {
    assert_eq!(x.ty.elt, "f32");
    assert_eq!(x.ty.shape.len(), 4);
    let batch = x.ty.shape[0];
    let time = x.ty.shape[1];
    assert_eq!(
        x.ty.shape[2..],
        [1, 1],
        "Gemma3n cumulative sum expects singleton feature axes"
    );
    let flattened = b.reshape(x, vec![batch, time]);
    let scanned = mlx_cuda_prefix_sum_2d(b, &flattened);
    b.reshape(&scanned, x.ty.shape.clone())
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
