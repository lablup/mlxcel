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

//! Semantically narrow emitters for operator-level numeric probes.
//!
//! These entry points deliberately expose every materialization boundary in
//! their signature. They reuse the same [`Builder`] operations as production
//! model emitters without applying graph-wide precision rewrites.

use super::builder::{Builder, Ty};

const MAX_EXACT_F32_INTEGER: usize = 1 << 24;

pub(crate) const DENSE_MATMUL_MODULE: &str = "numeric_dense_matmul";
pub(crate) const DENSE_MATMUL_ENTRY: &str = "numeric_dense_matmul.main";
pub(crate) const RESIDUAL_ADD_MODULE: &str = "numeric_residual_add";
pub(crate) const RESIDUAL_ADD_ENTRY: &str = "numeric_residual_add.main";
pub(crate) const RMS_NORM_MODULE: &str = "numeric_rms_norm";
pub(crate) const RMS_NORM_ENTRY: &str = "numeric_rms_norm.main";
pub(crate) const SOFTMAX_MODULE: &str = "numeric_attention_softmax";
pub(crate) const SOFTMAX_ENTRY: &str = "numeric_attention_softmax.main";

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
    let zero = builder.const_f32(0.0);
    let squared = builder.multiply(&input, &input);
    let sum = builder.reduce_add(&squared, 1, &zero);
    let count = builder.const_f32(spec.features as f32);
    let count = builder.broadcast(&count, &[], vec![spec.rows]);
    let mean = builder.divide(&sum, &count);
    let epsilon = builder.const_f32(epsilon);
    let epsilon = builder.broadcast(&epsilon, &[], vec![spec.rows]);
    let mean = builder.add(&mean, &epsilon);
    let inverse_rms = builder.rsqrt(&mean);
    let inverse_rms = builder.broadcast(&inverse_rms, &[0], input_ty.shape.clone());
    let normalized = builder.multiply(&input, &inverse_rms);
    let weight = builder.broadcast(&weight, &[1], input_ty.shape.clone());
    let output = builder.multiply(&normalized, &weight);
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
    let negative_infinity = builder.const_f32(f32::NEG_INFINITY);
    let maximum = builder.reduce_max(&input, 1, &negative_infinity);
    let maximum = builder.broadcast(&maximum, &[0], input_ty.shape.clone());
    let shifted = builder.subtract(&input, &maximum);
    let exponentials = builder.exponential(&shifted);
    let zero = builder.const_f32(0.0);
    let sum = builder.reduce_add(&exponentials, 1, &zero);
    let sum = builder.broadcast(&sum, &[0], input_ty.shape.clone());
    let output = builder.divide(&exponentials, &sum);
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
}
