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

//! The production MLX operation behind every micro-oracle, on the MLX CUDA
//! default stream. Each reference receives the probe's operands verbatim and
//! returns the MLX result widened to F32 (a lossless step for the F16 and
//! BF16 results).

use mlxcel_core::{MlxArray, UniquePtr, dtype};
use mlxcel_xla::{NumericTensor, ProductionProbeOperator, ProductionProbeSpec};

/// The MLX primitive each reference runs, recorded as its algorithm identity.
pub fn mlx_operation(operator: ProductionProbeOperator) -> &'static str {
    use ProductionProbeOperator as Op;
    match operator {
        Op::DenseMatmul => "mlx::core::matmul(x, transpose(w)) f32",
        Op::AffineQ4Qmv | Op::AffineQ4Qmm => {
            "mlx::core::quantized_matmul(transpose=true, bits=4, group=64, affine) f16"
        }
        Op::RmsNorm => "mlx::core::fast::rms_norm f32",
        Op::LayerNorm => "mlx::core::fast::layer_norm f32",
        Op::AttentionSoftmax => "mlx::core::softmax(axis=-1) f32",
        Op::SiluProjection => "mlx::core::matmul then silu f32",
        Op::GeluProjection => "mlx::core::matmul then gelu_tanh f32",
        Op::ResidualAdd => "mlx::core::add f32",
        Op::PrefixSum => "mlx::core::cumsum(axis=1, inclusive) f32",
        Op::Gemma3nSscpConvolution => "mlx::core::pad then conv2d(stride=2x2) bf16",
        _ => "unknown",
    }
}

fn shape_i32(shape: &[usize]) -> Result<Vec<i32>, String> {
    shape
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension).map_err(|_| format!("dimension {dimension} exceeds i32"))
        })
        .collect()
}

fn array(tensor: &NumericTensor) -> Result<UniquePtr<MlxArray>, String> {
    Ok(mlxcel_core::from_slice_f32(
        tensor.values(),
        &shape_i32(tensor.shape())?,
    ))
}

fn narrow(tensor: &NumericTensor, dtype: i32) -> Result<UniquePtr<MlxArray>, String> {
    Ok(mlxcel_core::astype(&*array(tensor)?, dtype))
}

/// Evaluate `output`, widen it to F32, and check it has `shape`.
fn read(output: &MlxArray, shape: Vec<usize>) -> Result<NumericTensor, String> {
    let widened = mlxcel_core::astype(output, dtype::FLOAT32);
    mlxcel_core::try_eval(&widened).map_err(|error| format!("MLX evaluation failed: {error}"))?;
    let observed = mlxcel_core::array_shape(&widened)
        .into_iter()
        .map(|dimension| usize::try_from(dimension).unwrap_or(0))
        .collect::<Vec<_>>();
    if observed != shape {
        return Err(format!(
            "MLX produced shape {observed:?}, expected {shape:?}"
        ));
    }
    let values = mlxcel_core::array_to_raw_bytes(&widened)
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte f32")))
        .collect();
    NumericTensor::new("output", shape, values)
}

fn matmul_transposed(
    input: &NumericTensor,
    weight: &NumericTensor,
) -> Result<UniquePtr<MlxArray>, String> {
    let weight = array(weight)?;
    Ok(mlxcel_core::matmul(
        &*array(input)?,
        &mlxcel_core::transpose(&weight),
    ))
}

fn rows_by(input: &NumericTensor, columns: usize) -> Vec<usize> {
    vec![input.shape()[0], columns]
}

fn affine_q4(
    spec: &ProductionProbeSpec,
    operands: &[NumericTensor],
) -> Result<NumericTensor, String> {
    let [input, lanes, scales, biases] = operands else {
        return Err("affine Q4 reference needs four operands".to_string());
    };
    let group_size = spec
        .group_size()
        .ok_or_else(|| "affine Q4 spec carries no group size".to_string())?;
    let packed = mlxcel_xla::pack_affine_q4_lanes(lanes)?;
    let outputs = lanes.shape()[0];
    let packed =
        mlxcel_core::from_slice_u32(&packed, &shape_i32(&[outputs, lanes.shape()[1] / 8])?);
    let scales = narrow(scales, dtype::FLOAT16)?;
    let biases = narrow(biases, dtype::FLOAT16)?;
    let input_f16 = narrow(input, dtype::FLOAT16)?;
    let group_size = i32::try_from(group_size).map_err(|_| "group size exceeds i32".to_string())?;
    // SAFETY: `biases` outlives the call and is a valid MLX array.
    let output = unsafe {
        mlxcel_core::quantized_matmul(
            &input_f16,
            &packed,
            &scales,
            &*biases as *const MlxArray,
            true,
            group_size,
            4,
            "affine",
        )
    };
    if mlxcel_core::array_dtype(&output) != dtype::FLOAT16 {
        return Err("MLX quantized_matmul did not return F16 for F16 activations".to_string());
    }
    read(&output, rows_by(input, outputs))
}

fn sscp_convolution(
    spec: &ProductionProbeSpec,
    operands: &[NumericTensor],
) -> Result<NumericTensor, String> {
    let [mel, weight] = operands else {
        return Err("SSCP convolution reference needs two operands".to_string());
    };
    let geometry = spec
        .convolution()
        .ok_or_else(|| "SSCP spec carries no convolution geometry".to_string())?;
    let to_i32 = |value: usize| i32::try_from(value).map_err(|_| format!("{value} exceeds i32"));
    let mel = narrow(mel, dtype::BFLOAT16)?;
    let weight_bf16 = narrow(weight, dtype::BFLOAT16)?;
    let padding = [
        0,
        0,
        to_i32(geometry.padding_low[0])?,
        to_i32(geometry.padding_high[0])?,
        to_i32(geometry.padding_low[1])?,
        to_i32(geometry.padding_high[1])?,
        0,
        0,
    ];
    let padded = mlxcel_core::pad(&mel, &padding, 0.0);
    let output = mlxcel_core::try_conv2d(
        &padded,
        &weight_bf16,
        to_i32(geometry.stride[0])?,
        to_i32(geometry.stride[1])?,
        0,
        0,
        1,
        1,
        1,
    )
    .map_err(|error| format!("MLX conv2d failed: {error}"))?;
    if mlxcel_core::array_dtype(&output) != dtype::BFLOAT16 {
        return Err("MLX conv2d did not materialize a BF16 result".to_string());
    }
    let shape = mlxcel_core::array_shape(&output)
        .into_iter()
        .map(|dimension| usize::try_from(dimension).unwrap_or(0))
        .collect();
    read(&output, shape)
}

/// Run the production MLX operation for `spec` on `operands`.
pub fn run_mlx_reference(
    spec: &ProductionProbeSpec,
    operands: &[NumericTensor],
) -> Result<NumericTensor, String> {
    use ProductionProbeOperator as Op;
    let operand = |index: usize| {
        operands.get(index).ok_or_else(|| {
            format!(
                "{} reference is missing operand {index}",
                spec.operator().name()
            )
        })
    };
    let epsilon = || {
        spec.epsilon()
            .ok_or_else(|| "normalization spec carries no epsilon".to_string())
    };
    match spec.operator() {
        Op::DenseMatmul => {
            let output = matmul_transposed(operand(0)?, operand(1)?)?;
            read(&output, rows_by(operand(0)?, operand(1)?.shape()[0]))
        }
        Op::AffineQ4Qmv | Op::AffineQ4Qmm => affine_q4(spec, operands),
        Op::RmsNorm => {
            let output = mlxcel_core::fast_rms_norm(
                &*array(operand(0)?)?,
                &*array(operand(1)?)?,
                epsilon()?,
            );
            read(&output, operand(0)?.shape().to_vec())
        }
        Op::LayerNorm => {
            let weight = array(operand(1)?)?;
            let bias = array(operand(2)?)?;
            // SAFETY: `weight` and `bias` outlive the call.
            let output = unsafe {
                mlxcel_core::fast_layer_norm(
                    &*array(operand(0)?)?,
                    &*weight as *const MlxArray,
                    &*bias as *const MlxArray,
                    epsilon()?,
                )
            };
            read(&output, operand(0)?.shape().to_vec())
        }
        Op::AttentionSoftmax => {
            let output = mlxcel_core::softmax(&*array(operand(0)?)?, -1);
            read(&output, operand(0)?.shape().to_vec())
        }
        Op::SiluProjection => {
            let projected = matmul_transposed(operand(0)?, operand(1)?)?;
            let output = mlxcel_core::silu(&projected);
            read(&output, rows_by(operand(0)?, operand(1)?.shape()[0]))
        }
        Op::GeluProjection => {
            let projected = matmul_transposed(operand(0)?, operand(1)?)?;
            let output = mlxcel_core::gelu_tanh(&projected);
            read(&output, rows_by(operand(0)?, operand(1)?.shape()[0]))
        }
        Op::ResidualAdd => {
            let output = mlxcel_core::add(&*array(operand(0)?)?, &*array(operand(1)?)?);
            read(&output, operand(0)?.shape().to_vec())
        }
        Op::PrefixSum => {
            let output = mlxcel_core::cumsum(&*array(operand(0)?)?, 1, false, true);
            read(&output, operand(0)?.shape().to_vec())
        }
        Op::Gemma3nSscpConvolution => sscp_convolution(spec, operands),
        other => Err(format!("no MLX reference for {}", other.name())),
    }
}
