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

pub(crate) const DENSE_MATMUL_MODULE: &str = "numeric_dense_matmul";
pub(crate) const DENSE_MATMUL_ENTRY: &str = "numeric_dense_matmul.main";

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
}
