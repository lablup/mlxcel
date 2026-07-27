// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Shared contract-sensitive numeric decompositions and bounded probes.
//!
//! Production emitters and the probe harness call the same helpers so their
//! materialization order cannot drift independently. Probe entry points keep
//! each contract boundary explicit and avoid graph-wide precision rewrites.

use super::builder::{Builder, Ty, Val};

const MAX_EXACT_F32_INTEGER: usize = 1 << 24;

pub(crate) const DENSE_MATMUL_MODULE: &str = "numeric_dense_matmul";
pub(crate) const DENSE_MATMUL_ENTRY: &str = "numeric_dense_matmul.main";
pub(crate) const RESIDUAL_ADD_MODULE: &str = "numeric_residual_add";
pub(crate) const RESIDUAL_ADD_ENTRY: &str = "numeric_residual_add.main";
pub(crate) const RMS_NORM_MODULE: &str = "numeric_rms_norm";
pub(crate) const RMS_NORM_ENTRY: &str = "numeric_rms_norm.main";
pub(crate) const SOFTMAX_MODULE: &str = "numeric_attention_softmax";
pub(crate) const SOFTMAX_ENTRY: &str = "numeric_attention_softmax.main";
pub(crate) const LAYER_NORM_MODULE: &str = "numeric_layer_norm";
pub(crate) const LAYER_NORM_ENTRY: &str = "numeric_layer_norm.main";
pub(crate) const SILU_PROJECTION_MODULE: &str = "numeric_silu_projection";
pub(crate) const SILU_PROJECTION_ENTRY: &str = "numeric_silu_projection.main";
pub(crate) const GELU_PROJECTION_MODULE: &str = "numeric_gelu_projection";
pub(crate) const GELU_PROJECTION_ENTRY: &str = "numeric_gelu_projection.main";
pub(crate) const PREFIX_SUM_MODULE: &str = "numeric_cuda_prefix_sum";
pub(crate) const PREFIX_SUM_ENTRY: &str = "numeric_cuda_prefix_sum.main";
pub(crate) const AFFINE_Q4_DEQUANT_MODULE: &str = "numeric_affine_q4_dequant";
pub(crate) const AFFINE_Q4_DEQUANT_ENTRY: &str = "numeric_affine_q4_dequant.main";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DenseMatmulProbeSpec {
    pub(crate) rows: usize,
    pub(crate) outputs: usize,
    pub(crate) contraction: usize,
}

impl DenseMatmulProbeSpec {
    pub(crate) fn validate(self) -> Result<Self, String> {
        if self.rows == 0 || self.outputs == 0 || self.contraction == 0 {
            return Err("dense matmul probe dimensions must be non-zero".to_string());
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AffineQ4ProbeSpec {
    pub(crate) outputs: usize,
    pub(crate) inputs: usize,
    pub(crate) group_size: usize,
}

impl AffineQ4ProbeSpec {
    fn validate(self) -> Result<Self, String> {
        if self.outputs == 0 || self.inputs == 0 || self.group_size == 0 {
            return Err("affine Q4 probe dimensions must be non-zero".to_string());
        }
        if !self.inputs.is_multiple_of(8) {
            return Err(format!(
                "affine Q4 probe input width {} must be divisible by 8 packed lanes",
                self.inputs
            ));
        }
        if !self.inputs.is_multiple_of(self.group_size) {
            return Err(format!(
                "affine Q4 probe input width {} must be divisible by group size {}",
                self.inputs, self.group_size
            ));
        }
        if !self.group_size.is_multiple_of(8) {
            return Err(format!(
                "affine Q4 probe group size {} must align to an eight-lane u32 carrier",
                self.group_size
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowWiseProbeSpec {
    pub(crate) rows: usize,
    pub(crate) features: usize,
}

impl RowWiseProbeSpec {
    pub(crate) fn validate(self, operation: &str) -> Result<Self, String> {
        if self.rows == 0 || self.features == 0 {
            return Err(format!("{operation} probe dimensions must be non-zero"));
        }
        Ok(self)
    }
}

fn validate_row_wise_inputs(operation: &str, value: &Val, parameters: &[&Val]) -> (usize, usize) {
    assert_eq!(value.ty.shape.len(), 2, "{operation} input must be rank-2");
    let rows = value.ty.shape[0];
    let features = value.ty.shape[1];
    assert!(
        features <= MAX_EXACT_F32_INTEGER,
        "{operation} feature count must be exactly representable as f32"
    );
    for parameter in parameters {
        assert_eq!(
            parameter.ty.shape,
            [features],
            "{operation} parameter must match the feature axis"
        );
        assert_eq!(
            parameter.ty.elt, value.ty.elt,
            "{operation} parameter dtype must match the input"
        );
    }
    (rows, features)
}

/// Explicit row-wise LayerNorm decomposition shared by production vision
/// emitters and bounded numeric probes.
pub(crate) fn layer_norm_2d(
    builder: &mut Builder,
    value: &Val,
    weight: &Val,
    bias: &Val,
    epsilon: f32,
) -> Val {
    let (rows, features) = validate_row_wise_inputs("LayerNorm", value, &[weight, bias]);
    assert!(
        epsilon.is_finite() && epsilon > 0.0,
        "LayerNorm epsilon must be finite and positive"
    );
    let zero = builder.const_f32(0.0);
    let width_scalar = builder.const_f32(features as f32);
    let width_rows = builder.broadcast(&width_scalar, &[], vec![rows]);
    let sum = builder.reduce_add(value, 1, &zero);
    let mean = builder.divide(&sum, &width_rows);
    let mean = builder.broadcast(&mean, &[0], vec![rows, features]);
    let centered = builder.subtract(value, &mean);
    let squared = builder.multiply(&centered, &centered);
    let squared_sum = builder.reduce_add(&squared, 1, &zero);
    let variance = builder.divide(&squared_sum, &width_rows);
    let epsilon = builder.const_f32(epsilon);
    let epsilon = builder.broadcast(&epsilon, &[], vec![rows]);
    let variance = builder.add(&variance, &epsilon);
    let inv_std = builder.rsqrt(&variance);
    let inv_std = builder.broadcast(&inv_std, &[0], vec![rows, features]);
    let normalized = builder.multiply(&centered, &inv_std);
    let weight = builder.broadcast(weight, &[1], vec![rows, features]);
    let bias = builder.broadcast(bias, &[1], vec![rows, features]);
    let normalized = builder.multiply(&normalized, &weight);
    builder.add(&normalized, &bias)
}

/// Explicit row-wise RMSNorm decomposition.
pub(crate) fn rms_norm_2d(builder: &mut Builder, value: &Val, weight: &Val, epsilon: f32) -> Val {
    let (rows, features) = validate_row_wise_inputs("RMSNorm", value, &[weight]);
    assert!(
        epsilon.is_finite() && epsilon > 0.0,
        "RMSNorm epsilon must be finite and positive"
    );
    let zero = builder.const_f32(0.0);
    let squared = builder.multiply(value, value);
    let sum = builder.reduce_add(&squared, 1, &zero);
    let count = builder.const_f32(features as f32);
    let count = builder.broadcast(&count, &[], vec![rows]);
    let mean = builder.divide(&sum, &count);
    let epsilon = builder.const_f32(epsilon);
    let epsilon = builder.broadcast(&epsilon, &[], vec![rows]);
    let mean = builder.add(&mean, &epsilon);
    let inverse_rms = builder.rsqrt(&mean);
    let inverse_rms = builder.broadcast(&inverse_rms, &[0], value.ty.shape.clone());
    let normalized = builder.multiply(value, &inverse_rms);
    let weight = builder.broadcast(weight, &[1], value.ty.shape.clone());
    builder.multiply(&normalized, &weight)
}

/// Exact-erf GELU decomposition used by vision projector paths.
pub(crate) fn exact_gelu(builder: &mut Builder, value: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let half = builder.const_f32(0.5);
    let half = builder.broadcast(&half, &[], shape.clone());
    let one = builder.const_f32(1.0);
    let one = builder.broadcast(&one, &[], shape.clone());
    let inv_sqrt_two = builder.const_f32(std::f32::consts::FRAC_1_SQRT_2);
    let inv_sqrt_two = builder.broadcast(&inv_sqrt_two, &[], shape);
    let scaled = builder.multiply(value, &inv_sqrt_two);
    let erf = builder.erf(&scaled);
    let cdf = builder.add(&one, &erf);
    let half_value = builder.multiply(value, &half);
    builder.multiply(&half_value, &cdf)
}

/// PyTorch tanh-approximation GELU decomposition.
pub(crate) fn tanh_gelu(builder: &mut Builder, value: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let half = builder.const_f32(0.5);
    let half = builder.broadcast(&half, &[], shape.clone());
    let one = builder.const_f32(1.0);
    let one = builder.broadcast(&one, &[], shape.clone());
    let coefficient = builder.const_f32(0.044_715);
    let coefficient = builder.broadcast(&coefficient, &[], shape.clone());
    let scale = builder.const_f32(0.797_884_6);
    let scale = builder.broadcast(&scale, &[], shape);
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

/// SiLU decomposition with an explicit activation materialization boundary.
pub(crate) fn silu(builder: &mut Builder, value: &Val) -> Val {
    let one = builder.const_f32(1.0);
    let one = builder.broadcast(&one, &[], value.ty.shape.clone());
    let negative = builder.negate(value);
    let exponential = builder.exponential(&negative);
    let denominator = builder.add(&one, &exponential);
    let sigmoid = builder.divide(&one, &denominator);
    builder.multiply(value, &sigmoid)
}

/// Numerically stable softmax with an explicit reduction axis.
pub(crate) fn stable_softmax(builder: &mut Builder, value: &Val, axis: usize) -> Val {
    assert!(
        axis < value.ty.shape.len(),
        "softmax axis must be within the input rank"
    );
    let negative_infinity = builder.const_f32(f32::NEG_INFINITY);
    let maximum = builder.reduce_max(value, axis, &negative_infinity);
    let kept_axes = (0..value.ty.shape.len())
        .filter(|candidate| *candidate != axis)
        .collect::<Vec<_>>();
    let maximum = builder.broadcast(&maximum, &kept_axes, value.ty.shape.clone());
    let shifted = builder.subtract(value, &maximum);
    let exponentials = builder.exponential(&shifted);
    let zero = builder.const_f32(0.0);
    let sum = builder.reduce_add(&exponentials, axis, &zero);
    let sum = builder.broadcast(&sum, &kept_axes, value.ty.shape.clone());
    builder.divide(&exponentials, &sum)
}

fn slice_axis(builder: &mut Builder, value: &Val, axis: usize, start: usize, limit: usize) -> Val {
    let ranges = value
        .ty
        .shape
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
    builder.slice(value, &ranges)
}

fn zeros_like(builder: &mut Builder, value: &Val) -> Val {
    let zero = builder.const_f32(0.0);
    builder.broadcast(&zero, &[], value.ty.shape.clone())
}

fn inclusive_scan_32_f32(builder: &mut Builder, value: &Val, axis: usize) -> Val {
    assert_eq!(value.ty.elt, "f32");
    assert_eq!(value.ty.shape[axis], 32);
    let zero = builder.const_f32(0.0);
    let mut result = value.clone();
    for offset in [1, 2, 4, 8, 16] {
        let prior = slice_axis(builder, &result, axis, 0, 32 - offset);
        let mut low = vec![0; result.ty.shape.len()];
        let high = low.clone();
        low[axis] = offset;
        let prior = builder.pad(&prior, &zero, &low, &high);
        result = builder.add(&result, &prior);
    }
    result
}

fn exclusive_from_inclusive_32_f32(builder: &mut Builder, inclusive: &Val, axis: usize) -> Val {
    let prior = slice_axis(builder, inclusive, axis, 0, 31);
    let zero = builder.const_f32(0.0);
    let mut low = vec![0; inclusive.ty.shape.len()];
    let high = low.clone();
    low[axis] = 1;
    builder.pad(&prior, &zero, &low, &high)
}

/// Match MLX CUDA's one-block contiguous inclusive-scan association schedule.
///
/// Each logical thread scans four adjacent values serially, each 32-thread
/// warp uses staged shuffle-up additions, and the warp totals are scanned by
/// the first warp. The helper accepts `[rows, width]` f32 input with a static
/// width no larger than the kernel's 4,096-value one-block limit.
pub(crate) fn mlx_cuda_prefix_sum_2d(builder: &mut Builder, value: &Val) -> Val {
    assert_eq!(value.ty.elt, "f32");
    assert_eq!(
        value.ty.shape.len(),
        2,
        "CUDA prefix sum input must be rank-2"
    );
    let rows = value.ty.shape[0];
    let width = value.ty.shape[1];
    assert!(
        (1..=4096).contains(&width),
        "CUDA prefix sum supports one scan block"
    );
    let padded_width = width.div_ceil(128) * 128;
    let padded = if padded_width == width {
        value.clone()
    } else {
        let zero = builder.const_f32(0.0);
        builder.pad(value, &zero, &[0, 0], &[0, padded_width - width])
    };
    let threads = padded_width / 4;
    let warps = threads / 32;
    let grouped = builder.reshape(&padded, vec![rows, threads, 4]);
    let value0 = slice_axis(builder, &grouped, 2, 0, 1);
    let value1 = slice_axis(builder, &grouped, 2, 1, 2);
    let value2 = slice_axis(builder, &grouped, 2, 2, 3);
    let value3 = slice_axis(builder, &grouped, 2, 3, 4);
    let local0 = value0;
    let local1 = builder.add(&value1, &local0);
    let local2 = builder.add(&value2, &local1);
    let local3 = builder.add(&value3, &local2);
    let first_half = builder.concatenate(&local0, &local1, 2);
    let second_half = builder.concatenate(&local2, &local3, 2);
    let local = builder.concatenate(&first_half, &second_half, 2);

    let thread_sums = builder.reshape(&local3, vec![rows, warps, 32]);
    let thread_inclusive = inclusive_scan_32_f32(builder, &thread_sums, 2);
    let thread_exclusive = exclusive_from_inclusive_32_f32(builder, &thread_inclusive, 2);

    // The MLX kernel forms a warp total as the exclusive last-lane prefix plus
    // the current last-lane value. Keep that distinct association boundary.
    let prior_last = slice_axis(builder, &thread_exclusive, 2, 31, 32);
    let current_last = slice_axis(builder, &thread_sums, 2, 31, 32);
    let warp_totals = builder.add(&prior_last, &current_last);
    let warp_totals = builder.reshape(&warp_totals, vec![rows, warps]);
    let zero = builder.const_f32(0.0);
    let warp_totals = builder.pad(&warp_totals, &zero, &[0, 0], &[0, 32 - warps]);
    let warp_inclusive = inclusive_scan_32_f32(builder, &warp_totals, 1);
    let warp_exclusive = exclusive_from_inclusive_32_f32(builder, &warp_inclusive, 1);
    let warp_exclusive = slice_axis(builder, &warp_exclusive, 1, 0, warps);

    let local = builder.reshape(&local, vec![rows, warps, 32, 4]);
    let explicit_zero = zeros_like(builder, &local);
    let local = builder.add(&local, &explicit_zero);
    let warp_exclusive = builder.reshape(&warp_exclusive, vec![rows, warps, 1, 1]);
    let warp_exclusive = builder.broadcast(&warp_exclusive, &[0, 1, 2, 3], local.ty.shape.clone());
    let with_warp_prefix = builder.add(&local, &warp_exclusive);
    let thread_exclusive = builder.reshape(&thread_exclusive, vec![rows, warps, 32, 1]);
    let thread_exclusive = builder.broadcast(
        &thread_exclusive,
        &[0, 1, 2, 3],
        with_warp_prefix.ty.shape.clone(),
    );
    let scanned = builder.add(&with_warp_prefix, &thread_exclusive);
    let scanned = builder.reshape(&scanned, vec![rows, padded_width]);
    slice_axis(builder, &scanned, 1, 0, width)
}

/// Emit `output[rows, outputs] = input[rows, contraction] @
/// weight[outputs, contraction]^T` with f32 inputs, accumulation, and result.
pub(crate) fn emit_dense_matmul_probe(spec: DenseMatmulProbeSpec) -> Result<String, String> {
    let spec = spec.validate()?;
    let weight_ty = Ty::f32(vec![spec.outputs, spec.contraction]);
    let input_ty = Ty::f32(vec![spec.rows, spec.contraction]);
    let output_ty = Ty::f32(vec![spec.rows, spec.outputs]);
    let weight = Builder::arg(0, weight_ty.clone());
    let input = Builder::arg(1, input_ty.clone());
    let mut builder = Builder::new();
    let output = builder.linear_seq(&input, &weight);
    if output.ty.shape != output_ty.shape || output.ty.elt != output_ty.elt {
        return Err(format!(
            "dense matmul probe emitted {}, expected {}",
            output.ty.render(),
            output_ty.render()
        ));
    }
    Ok(format!(
        "module @{DENSE_MATMUL_MODULE} {{\n  \
         func.func public @main(%arg0: {weight_ty}, %arg1: {input_ty}) -> {output_ty} {{\n\
         {body}    return {output} : {output_ty}\n  }}\n}}\n",
        weight_ty = weight_ty.render(),
        input_ty = input_ty.render(),
        output_ty = output_ty.render(),
        body = builder.body(),
        output = output.name,
    ))
}

/// Emit a binary f32 residual addition.
///
/// The leading one-element resident weight is deliberately unused. The generic
/// auxiliary ABI requires at least one resident weight, while the numeric
/// contract and oracle operands cover only the actual residual inputs.
pub(crate) fn emit_residual_add_probe(spec: RowWiseProbeSpec) -> Result<String, String> {
    let spec = spec.validate("residual add")?;
    let dummy_ty = Ty::f32(vec![1]);
    let input_ty = Ty::f32(vec![spec.rows, spec.features]);
    let lhs = Builder::arg(1, input_ty.clone());
    let rhs = Builder::arg(2, input_ty.clone());
    let mut builder = Builder::new();
    let output = builder.add(&lhs, &rhs);
    Ok(format!(
        "module @{RESIDUAL_ADD_MODULE} {{\n  \
         func.func public @main(%arg0: {dummy_ty}, %arg1: {input_ty}, %arg2: {input_ty}) -> \
         {input_ty} {{\n\
         {body}    return {output} : {input_ty}\n  }}\n}}\n",
        dummy_ty = dummy_ty.render(),
        input_ty = input_ty.render(),
        body = builder.body(),
        output = output.name,
    ))
}

/// Emit row-wise f32 RMSNorm with a resident affine weight.
pub(crate) fn emit_rms_norm_probe(spec: RowWiseProbeSpec, epsilon: f32) -> Result<String, String> {
    let spec = spec.validate("RMSNorm")?;
    if spec.features > MAX_EXACT_F32_INTEGER {
        return Err(format!(
            "RMSNorm probe feature count {} exceeds the largest consecutive integer exactly \
             representable as f32 ({MAX_EXACT_F32_INTEGER})",
            spec.features
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("RMSNorm probe epsilon must be finite and positive".to_string());
    }
    let weight_ty = Ty::f32(vec![spec.features]);
    let input_ty = Ty::f32(vec![spec.rows, spec.features]);
    let weight = Builder::arg(0, weight_ty.clone());
    let input = Builder::arg(1, input_ty.clone());
    let mut builder = Builder::new();
    let output = rms_norm_2d(&mut builder, &input, &weight, epsilon);
    Ok(format!(
        "module @{RMS_NORM_MODULE} {{\n  \
         func.func public @main(%arg0: {weight_ty}, %arg1: {input_ty}) -> {input_ty} {{\n\
         {body}    return {output} : {input_ty}\n  }}\n}}\n",
        weight_ty = weight_ty.render(),
        input_ty = input_ty.render(),
        body = builder.body(),
        output = output.name,
    ))
}

/// Emit numerically stable row-wise f32 softmax.
pub(crate) fn emit_softmax_probe(spec: RowWiseProbeSpec) -> Result<String, String> {
    let spec = spec.validate("attention softmax")?;
    let dummy_ty = Ty::f32(vec![1]);
    let input_ty = Ty::f32(vec![spec.rows, spec.features]);
    let input = Builder::arg(1, input_ty.clone());
    let mut builder = Builder::new();
    let output = stable_softmax(&mut builder, &input, 1);
    Ok(format!(
        "module @{SOFTMAX_MODULE} {{\n  \
         func.func public @main(%arg0: {dummy_ty}, %arg1: {input_ty}) -> {input_ty} {{\n\
         {body}    return {output} : {input_ty}\n  }}\n}}\n",
        dummy_ty = dummy_ty.render(),
        input_ty = input_ty.render(),
        body = builder.body(),
        output = output.name,
    ))
}

/// Emit row-wise f32 LayerNorm with resident affine weight and bias.
pub(crate) fn emit_layer_norm_probe(
    spec: RowWiseProbeSpec,
    epsilon: f32,
) -> Result<String, String> {
    let spec = spec.validate("LayerNorm")?;
    if spec.features > MAX_EXACT_F32_INTEGER {
        return Err(format!(
            "LayerNorm probe feature count {} exceeds the largest consecutive integer exactly \
             representable as f32 ({MAX_EXACT_F32_INTEGER})",
            spec.features
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("LayerNorm probe epsilon must be finite and positive".to_string());
    }
    let parameter_ty = Ty::f32(vec![spec.features]);
    let input_ty = Ty::f32(vec![spec.rows, spec.features]);
    let weight = Builder::arg(0, parameter_ty.clone());
    let bias = Builder::arg(1, parameter_ty.clone());
    let input = Builder::arg(2, input_ty.clone());
    let mut builder = Builder::new();
    let output = layer_norm_2d(&mut builder, &input, &weight, &bias, epsilon);
    Ok(format!(
        "module @{LAYER_NORM_MODULE} {{\n  \
         func.func public @main(%arg0: {parameter_ty}, %arg1: {parameter_ty}, %arg2: {input_ty}) \
         -> {input_ty} {{\n\
         {body}    return {output} : {input_ty}\n  }}\n}}\n",
        parameter_ty = parameter_ty.render(),
        input_ty = input_ty.render(),
        body = builder.body(),
        output = output.name,
    ))
}

fn emit_activation_projection_probe(
    spec: DenseMatmulProbeSpec,
    module: &str,
    activation: fn(&mut Builder, &Val) -> Val,
) -> Result<String, String> {
    let spec = spec.validate()?;
    let weight_ty = Ty::f32(vec![spec.outputs, spec.contraction]);
    let input_ty = Ty::f32(vec![spec.rows, spec.contraction]);
    let output_ty = Ty::f32(vec![spec.rows, spec.outputs]);
    let weight = Builder::arg(0, weight_ty.clone());
    let input = Builder::arg(1, input_ty.clone());
    let mut builder = Builder::new();
    let projected = builder.linear_seq(&input, &weight);
    let output = activation(&mut builder, &projected);
    Ok(format!(
        "module @{module} {{\n  \
         func.func public @main(%arg0: {weight_ty}, %arg1: {input_ty}) -> {output_ty} {{\n\
         {body}    return {output} : {output_ty}\n  }}\n}}\n",
        weight_ty = weight_ty.render(),
        input_ty = input_ty.render(),
        output_ty = output_ty.render(),
        body = builder.body(),
        output = output.name,
    ))
}

/// Emit dense projection followed by explicit SiLU materialization.
pub(crate) fn emit_silu_projection_probe(spec: DenseMatmulProbeSpec) -> Result<String, String> {
    emit_activation_projection_probe(spec, SILU_PROJECTION_MODULE, silu)
}

/// Emit dense projection followed by PyTorch tanh GELU materialization.
pub(crate) fn emit_gelu_projection_probe(spec: DenseMatmulProbeSpec) -> Result<String, String> {
    emit_activation_projection_probe(spec, GELU_PROJECTION_MODULE, tanh_gelu)
}

/// Emit the explicit MLX CUDA one-block contiguous inclusive-scan schedule.
pub(crate) fn emit_prefix_sum_probe(spec: RowWiseProbeSpec) -> Result<String, String> {
    let spec = spec.validate("prefix sum")?;
    if spec.features > 4096 {
        return Err(format!(
            "prefix sum probe width {} exceeds the one-block CUDA limit 4096",
            spec.features
        ));
    }
    let dummy_ty = Ty::f32(vec![1]);
    let input_ty = Ty::f32(vec![spec.rows, spec.features]);
    let input = Builder::arg(1, input_ty.clone());
    let mut builder = Builder::new();
    let output = mlx_cuda_prefix_sum_2d(&mut builder, &input);
    Ok(format!(
        "module @{PREFIX_SUM_MODULE} {{\n  \
         func.func public @main(%arg0: {dummy_ty}, %arg1: {input_ty}) -> {input_ty} {{\n\
         {body}    return {output} : {input_ty}\n  }}\n}}\n",
        dummy_ty = dummy_ty.render(),
        input_ty = input_ty.render(),
        body = builder.body(),
        output = output.name,
    ))
}

/// Emit least-significant-lane-first affine Q4 unpack and dequantization.
pub(crate) fn emit_affine_q4_dequant_probe(spec: AffineQ4ProbeSpec) -> Result<String, String> {
    let spec = spec.validate()?;
    let packed_ty = Ty::new(vec![spec.outputs, spec.inputs / 8], "ui32");
    let metadata_ty = Ty::new(vec![spec.outputs, spec.inputs / spec.group_size], "f16");
    let dummy_ty = Ty::f32(vec![1]);
    let output_ty = Ty::f32(vec![spec.outputs, spec.inputs]);
    let packed = Builder::arg(0, packed_ty.clone());
    let scales = Builder::arg(1, metadata_ty.clone());
    let biases = Builder::arg(2, metadata_ty.clone());
    let mut builder = Builder::new();
    let output = builder.dequant_affine(&packed, &scales, &biases, 4, spec.group_size);
    Ok(format!(
        "module @{AFFINE_Q4_DEQUANT_MODULE} {{\n  \
         func.func public @main(%arg0: {packed_ty}, %arg1: {metadata_ty}, \
         %arg2: {metadata_ty}, %arg3: {dummy_ty}) -> {output_ty} {{\n\
         {body}    return {output} : {output_ty}\n  }}\n}}\n",
        packed_ty = packed_ty.render(),
        metadata_ty = metadata_ty.render(),
        dummy_ty = dummy_ty.render(),
        output_ty = output_ty.render(),
        body = builder.body(),
        output = output.name,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_matmul_probe_has_explicit_f32_contract() {
        let graph = emit_dense_matmul_probe(DenseMatmulProbeSpec {
            rows: 2,
            outputs: 2,
            contraction: 3,
        })
        .unwrap();
        assert!(graph.starts_with("module @numeric_dense_matmul {"));
        assert!(graph.contains(
            "func.func public @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) \
             -> tensor<2x2xf32>"
        ));
        assert!(graph.contains(
            "stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1] : \
             (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x2xf32>"
        ));
        assert!(!graph.contains("stablehlo.convert"));
    }

    #[test]
    fn dense_matmul_probe_rejects_zero_dimensions() {
        assert!(
            emit_dense_matmul_probe(DenseMatmulProbeSpec {
                rows: 0,
                outputs: 2,
                contraction: 3,
            })
            .is_err()
        );
    }

    #[test]
    fn residual_add_probe_is_one_explicit_f32_add() {
        let graph = emit_residual_add_probe(RowWiseProbeSpec {
            rows: 2,
            features: 4,
        })
        .unwrap();
        assert!(graph.starts_with("module @numeric_residual_add {"));
        assert!(graph.contains(
            "func.func public @main(%arg0: tensor<1xf32>, %arg1: tensor<2x4xf32>, \
             %arg2: tensor<2x4xf32>) -> tensor<2x4xf32>"
        ));
        assert_eq!(graph.matches("stablehlo.add").count(), 1);
        assert!(!graph.contains("stablehlo.convert"));
    }

    #[test]
    fn rms_norm_probe_exposes_reduction_and_rsqrt_boundaries() {
        let graph = emit_rms_norm_probe(
            RowWiseProbeSpec {
                rows: 2,
                features: 4,
            },
            1e-5,
        )
        .unwrap();
        assert!(graph.starts_with("module @numeric_rms_norm {"));
        assert!(graph.contains("applies stablehlo.add across dimensions = [1]"));
        assert!(graph.contains("stablehlo.rsqrt"));
        assert!(!graph.contains("stablehlo.convert"));
    }

    #[test]
    fn softmax_probe_uses_stable_row_wise_decomposition() {
        let graph = emit_softmax_probe(RowWiseProbeSpec {
            rows: 2,
            features: 4,
        })
        .unwrap();
        assert!(graph.starts_with("module @numeric_attention_softmax {"));
        assert!(graph.contains("applies stablehlo.maximum across dimensions = [1]"));
        assert!(graph.contains("stablehlo.exponential"));
        assert!(graph.contains("applies stablehlo.add across dimensions = [1]"));
    }

    #[test]
    fn row_wise_probes_reject_invalid_parameters() {
        let empty = RowWiseProbeSpec {
            rows: 0,
            features: 4,
        };
        assert!(emit_residual_add_probe(empty).is_err());
        assert!(emit_softmax_probe(empty).is_err());
        assert!(
            emit_rms_norm_probe(
                RowWiseProbeSpec {
                    rows: 2,
                    features: 4,
                },
                0.0,
            )
            .is_err()
        );
        assert!(
            emit_rms_norm_probe(
                RowWiseProbeSpec {
                    rows: 1,
                    features: MAX_EXACT_F32_INTEGER + 1,
                },
                1e-5,
            )
            .is_err()
        );
    }

    #[test]
    fn layer_norm_probe_pins_center_variance_affine_order() {
        let graph = emit_layer_norm_probe(
            RowWiseProbeSpec {
                rows: 2,
                features: 4,
            },
            1e-5,
        )
        .unwrap();
        assert!(graph.starts_with("module @numeric_layer_norm {"));
        assert_eq!(
            graph
                .matches("applies stablehlo.add across dimensions = [1]")
                .count(),
            2
        );
        assert!(graph.contains("stablehlo.subtract"));
        assert!(graph.contains("stablehlo.rsqrt"));
        assert!(!graph.contains("stablehlo.convert"));
    }

    #[test]
    fn activation_projection_probes_pin_matmul_before_activation() {
        let spec = DenseMatmulProbeSpec {
            rows: 2,
            outputs: 3,
            contraction: 4,
        };
        let silu_graph = emit_silu_projection_probe(spec).unwrap();
        let silu_dot = silu_graph.find("stablehlo.dot_general").unwrap();
        let silu_exp = silu_graph.find("stablehlo.exponential").unwrap();
        assert!(silu_dot < silu_exp);
        assert!(silu_graph.contains("stablehlo.negate"));

        let gelu_graph = emit_gelu_projection_probe(spec).unwrap();
        let gelu_dot = gelu_graph.find("stablehlo.dot_general").unwrap();
        let gelu_tanh = gelu_graph.find("stablehlo.tanh").unwrap();
        assert!(gelu_dot < gelu_tanh);
        assert!(gelu_graph.contains("0x3D372713"));
        assert!(!gelu_graph.contains("chlo.erf"));
    }

    #[test]
    fn exact_gelu_uses_erf_not_tanh() {
        let input = Builder::arg(0, Ty::f32(vec![2, 3]));
        let mut builder = Builder::new();
        let _ = exact_gelu(&mut builder, &input);
        let body = builder.body();
        assert!(body.contains("chlo.erf"));
        assert!(!body.contains("stablehlo.tanh"));
    }

    #[test]
    fn prefix_sum_probe_pins_cuda_hierarchical_association() {
        let graph = emit_prefix_sum_probe(RowWiseProbeSpec {
            rows: 2,
            features: 130,
        })
        .unwrap();
        assert!(graph.starts_with("module @numeric_cuda_prefix_sum {"));
        assert_eq!(graph.matches("stablehlo.add").count(), 17);
        assert_eq!(graph.matches("stablehlo.concatenate").count(), 3);
        assert!(!graph.contains("stablehlo.reduce"));
        assert!(!graph.contains("stablehlo.reduce_window"));
        assert!(!graph.contains("stablehlo.convert"));
    }

    #[test]
    fn prefix_sum_probe_rejects_multi_block_width() {
        assert!(
            emit_prefix_sum_probe(RowWiseProbeSpec {
                rows: 1,
                features: 4097,
            })
            .is_err()
        );
    }

    #[test]
    fn affine_q4_probe_pins_lane_and_evaluation_order() {
        let graph = emit_affine_q4_dequant_probe(AffineQ4ProbeSpec {
            outputs: 2,
            inputs: 16,
            group_size: 8,
        })
        .unwrap();
        assert!(graph.starts_with("module @numeric_affine_q4_dequant {"));
        assert!(graph.contains("dense<[0, 4, 8, 12, 16, 20, 24, 28]>"));
        assert!(graph.contains("stablehlo.shift_right_logical"));
        assert!(graph.contains("dense<15> : tensor<2x2x8xui32>"));
        let multiply = graph.find("stablehlo.multiply").unwrap();
        let add = graph.find("stablehlo.add").unwrap();
        assert!(multiply < add);
        assert_eq!(graph.matches("stablehlo.convert").count(), 3);
    }

    #[test]
    fn affine_q4_probe_rejects_unaligned_shapes() {
        for spec in [
            AffineQ4ProbeSpec {
                outputs: 0,
                inputs: 16,
                group_size: 8,
            },
            AffineQ4ProbeSpec {
                outputs: 1,
                inputs: 15,
                group_size: 8,
            },
            AffineQ4ProbeSpec {
                outputs: 1,
                inputs: 16,
                group_size: 6,
            },
        ] {
            assert!(emit_affine_q4_dequant_probe(spec).is_err());
        }
    }
}
