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

//! Emitted operations for the production-MLX micro-oracles (issue #3212).
//!
//! Each probe is built from the same builder calls the production graphs use:
//! `Builder::dequant_affine` plus the precision-demoting `linear_seq` for the
//! affine Q4 projections, and the Gemma3n subsampler's own operand layout and
//! BF16 result helpers for the first SSCP convolution. Nothing here rewrites
//! precision beyond what the production emitter applies.

use super::builder::{Builder, Precision, Ty};
use super::gemma3n_audio_math::round_bf16;
use super::gemma3n_audio_ops::{sscp_conv_operands, sscp_conv_result};

pub(crate) const AFFINE_Q4_MATMUL_MODULE: &str = "numeric_affine_q4_matmul";
pub(crate) const AFFINE_Q4_MATMUL_ENTRY: &str = "numeric_affine_q4_matmul.main";
pub(crate) const SSCP_CONVOLUTION_MODULE: &str = "numeric_gemma3n_sscp_convolution";
pub(crate) const SSCP_CONVOLUTION_ENTRY: &str = "numeric_gemma3n_sscp_convolution.main";

/// `output[rows, outputs] = input[rows, inputs] @ dequant(packed)^T`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AffineQ4MatmulProbeSpec {
    pub(crate) rows: usize,
    pub(crate) outputs: usize,
    pub(crate) inputs: usize,
    pub(crate) group_size: usize,
}

impl AffineQ4MatmulProbeSpec {
    fn validate(self) -> Result<Self, String> {
        if self.rows == 0 || self.outputs == 0 || self.inputs == 0 || self.group_size == 0 {
            return Err("affine Q4 matmul probe dimensions must be non-zero".to_string());
        }
        if !self.group_size.is_multiple_of(8) || !self.inputs.is_multiple_of(self.group_size) {
            return Err(format!(
                "affine Q4 matmul probe input width {} must be a multiple of group size {}, \
                 which must align to an eight-lane u32 carrier",
                self.inputs, self.group_size
            ));
        }
        Ok(self)
    }
}

/// Emit packed affine Q4 dequantization followed by the F16-precision
/// projection and an F16 output materialization.
///
/// `Precision::F16` makes `linear_seq` demote the F32 activations and the F32
/// dequantized weight to F16 and accumulate in F32, which is the production
/// rule in `Builder::dot_general`. The F16 output rounding mirrors the F16
/// tensor MLX `quantized_matmul` returns for F16 activations.
pub(crate) fn emit_affine_q4_matmul_probe(spec: AffineQ4MatmulProbeSpec) -> Result<String, String> {
    let spec = spec.validate()?;
    let packed_ty = Ty::new(vec![spec.outputs, spec.inputs / 8], "ui32");
    let metadata_ty = Ty::new(vec![spec.outputs, spec.inputs / spec.group_size], "f16");
    let input_ty = Ty::f32(vec![spec.rows, spec.inputs]);
    let output_ty = Ty::f32(vec![spec.rows, spec.outputs]);
    let packed = Builder::arg(0, packed_ty.clone());
    let scales = Builder::arg(1, metadata_ty.clone());
    let biases = Builder::arg(2, metadata_ty.clone());
    let input = Builder::arg(3, input_ty.clone());
    let mut builder = Builder::new().with_precision(Precision::F16);
    let weight = builder.dequant_affine(&packed, &scales, &biases, 4, spec.group_size);
    let projected = builder.linear_seq(&input, &weight);
    let narrowed = builder.convert(&projected, "f16");
    let output = builder.convert(&narrowed, "f32");
    Ok(format!(
        "module @{AFFINE_Q4_MATMUL_MODULE} {{\n  \
         func.func public @main(%arg0: {packed_ty}, %arg1: {metadata_ty}, \
         %arg2: {metadata_ty}, %arg3: {input_ty}) -> {output_ty} {{\n\
         {body}    return {output} : {output_ty}\n  }}\n}}\n",
        packed_ty = packed_ty.render(),
        metadata_ty = metadata_ty.render(),
        input_ty = input_ty.render(),
        output_ty = output_ty.render(),
        body = builder.body(),
        output = output.name,
    ))
}

/// Geometry of the first Gemma3n SSCP convolution: a single-channel
/// `[1, time, frequency, 1]` mel input and an `[channels, kh, kw, 1]` weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SscpConvolutionProbeSpec {
    pub(crate) time: usize,
    pub(crate) frequency: usize,
    pub(crate) channels: usize,
    pub(crate) kernel: [usize; 2],
    pub(crate) stride: [usize; 2],
}

impl SscpConvolutionProbeSpec {
    fn validate(self) -> Result<Self, String> {
        if self.time == 0
            || self.frequency == 0
            || self.channels == 0
            || self.kernel.contains(&0)
            || self.stride.contains(&0)
        {
            return Err("SSCP convolution probe dimensions must be non-zero".to_string());
        }
        if self.frequency + 2 < self.kernel[1] {
            return Err("SSCP convolution kernel exceeds the padded frequency axis".to_string());
        }
        Ok(self)
    }

    /// `[1, time_out, frequency_out, channels]` after `(0, kh - 1)` time and
    /// `(1, 1)` frequency padding.
    pub(crate) fn output_shape(self) -> Vec<usize> {
        let time = (self.time - 1) / self.stride[0] + 1;
        let frequency = (self.frequency + 2 - self.kernel[1]) / self.stride[1] + 1;
        vec![1, time, frequency, self.channels]
    }
}

/// Emit the production first SSCP convolution: BF16-rounded F32 mel input,
/// BF16 operands under `Precision::Bf16`, F32 accumulation, BF16 result.
pub(crate) fn emit_sscp_convolution_probe(
    spec: SscpConvolutionProbeSpec,
) -> Result<String, String> {
    let spec = spec.validate()?;
    let weight_ty = Ty::f32(vec![spec.channels, spec.kernel[0], spec.kernel[1], 1]);
    let input_ty = Ty::f32(vec![1, spec.time, spec.frequency, 1]);
    let output_ty = Ty::f32(spec.output_shape());
    let weight = Builder::arg(0, weight_ty.clone());
    let input = Builder::arg(1, input_ty.clone());
    let mut builder = Builder::new().with_precision(Precision::Bf16);
    // The maintained MLX path casts processor F32 mel features to the BF16
    // convolution weight dtype before the first convolution.
    let input = round_bf16(&mut builder, &input);
    let (padded, weight) = sscp_conv_operands(&mut builder, &input, &weight, spec.kernel);
    let output = sscp_conv_result(&mut builder, &padded, &weight, spec.stride, 1);
    if output.ty.shape != output_ty.shape {
        return Err(format!(
            "SSCP convolution probe emitted {}, expected {}",
            output.ty.render(),
            output_ty.render()
        ));
    }
    Ok(format!(
        "module @{SSCP_CONVOLUTION_MODULE} {{\n  \
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
    fn affine_q4_matmul_probe_demotes_contraction_inputs_and_rounds_output_to_f16() {
        let graph = emit_affine_q4_matmul_probe(AffineQ4MatmulProbeSpec {
            rows: 2,
            outputs: 4,
            inputs: 16,
            group_size: 8,
        })
        .unwrap();
        assert!(graph.starts_with("module @numeric_affine_q4_matmul {"));
        assert!(graph.contains(
            "func.func public @main(%arg0: tensor<4x2xui32>, %arg1: tensor<4x2xf16>, \
             %arg2: tensor<4x2xf16>, %arg3: tensor<2x16xf32>) -> tensor<2x4xf32>"
        ));
        assert!(graph.contains("stablehlo.shift_right_logical"));
        assert!(graph.contains("(tensor<2x16xf16>, tensor<4x16xf16>) -> tensor<2x4xf32>"));
        assert!(graph.contains("(tensor<2x4xf32>) -> tensor<2x4xf16>"));
    }

    #[test]
    fn affine_q4_matmul_probe_rejects_misaligned_groups() {
        let spec = AffineQ4MatmulProbeSpec {
            rows: 1,
            outputs: 2,
            inputs: 12,
            group_size: 8,
        };
        assert!(emit_affine_q4_matmul_probe(spec).is_err());
    }

    #[test]
    fn sscp_convolution_probe_matches_the_production_first_convolution_boundary() {
        let spec = SscpConvolutionProbeSpec {
            time: 8,
            frequency: 128,
            channels: 128,
            kernel: [3, 3],
            stride: [2, 2],
        };
        assert_eq!(spec.output_shape(), [1, 4, 64, 128]);
        let graph = emit_sscp_convolution_probe(spec).unwrap();
        assert!(graph.starts_with("module @numeric_gemma3n_sscp_convolution {"));
        assert!(graph.contains("edge_padding_low = array<i64: 0, 0, 1, 0>"));
        assert!(graph.contains("edge_padding_high = array<i64: 0, 2, 1, 0>"));
        assert!(graph.contains("window_strides = array<i64: 2, 2>"));
        assert!(graph.contains(
            "(tensor<1x10x130x1xbf16>, tensor<3x3x1x128xbf16>) -> tensor<1x4x64x128xf32>"
        ));
        assert!(graph.contains("(tensor<1x4x64x128xf32>) -> tensor<1x4x64x128xbf16>"));
    }
}
