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

//! Small, deterministic MLX↔IREE oracle for the Gemma3n first SSCP convolution.
//!
//! This deliberately excludes the cumulative group norm and every later audio
//! operation. Gemma3n's first SSCP convolution has no bias; its exact dtype
//! boundary is F32 processor features -> BF16 input, BF16 checkpoint weights,
//! a 3x3 contraction, and a materialized BF16 output.

use std::path::{Path, PathBuf};
use std::process::Command;

const INPUT_SHAPE: [usize; 4] = [1, 3, 3, 1];
const OUTPUT_CHANNELS: usize = 4;
const OUTPUT_SHAPE: [usize; 4] = [1, 1, 1, OUTPUT_CHANNELS];
const INPUT: [f32; 9] = [16384.0, 1.0, -16384.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

// MLX stores convolution weights as [O, H, W, I]. Each row makes the three
// non-zero products a cancellation-sensitive permutation of [2^24, 1, -2^24].
const MLX_WEIGHTS: [[f32; 9]; OUTPUT_CHANNELS] = [
    [1024.0, 1.0, 1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [
        1024.0,
        -16777216.0,
        -0.00006103515625,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.00006103515625,
        16777216.0,
        1024.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [-1024.0, 1.0, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
];

const MLIR: &str = r#"module @gemma3n_sscp_conv_oracle {
  func.func public @main(
      %input: tensor<1x3x3x1xf32>,
      %kernel: tensor<3x3x1x4xf32>
  ) -> tensor<1x1x1x4xf32> {
    %input_bf16 = stablehlo.convert %input : (tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xbf16>
    %kernel_bf16 = stablehlo.convert %kernel : (tensor<3x3x1x4xf32>) -> tensor<3x3x1x4xbf16>
    %convolved = "stablehlo.convolution"(%input_bf16, %kernel_bf16) {
      window_strides = array<i64: 1, 1>,
      padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
      lhs_dilation = array<i64: 1, 1>,
      rhs_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<1x3x3x1xbf16>, tensor<3x3x1x4xbf16>) -> tensor<1x1x1x4xbf16>
    %output = stablehlo.convert %convolved : (tensor<1x1x1x4xbf16>) -> tensor<1x1x1x4xf32>
    return %output : tensor<1x1x1x4xf32>
  }
}
"#;

fn round_bf16(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
}

fn bf16_bits(value: f32) -> Option<u16> {
    let bits = value.to_bits();
    (bits & 0xffff == 0).then_some((bits >> 16) as u16)
}

fn ordered_bf16(bits: u16) -> i32 {
    if bits & 0x8000 == 0 {
        0x8000 + i32::from(bits)
    } else {
        0x8000 - i32::from(bits & 0x7fff)
    }
}

fn bf16_ulp_distance(left: f32, right: f32) -> Option<u32> {
    Some(ordered_bf16(bf16_bits(left)?).abs_diff(ordered_bf16(bf16_bits(right)?)))
}

fn mlx_output() -> Result<Vec<f32>, String> {
    let input_shape = INPUT_SHAPE.map(|dimension| dimension as i32);
    let input = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&INPUT, &input_shape),
        mlxcel_core::dtype::BFLOAT16,
    );
    let weights = MLX_WEIGHTS
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let weights = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&weights, &[OUTPUT_CHANNELS as i32, 3, 3, 1]),
        mlxcel_core::dtype::BFLOAT16,
    );
    let output = mlxcel_core::try_conv2d(&input, &weights, 1, 1, 0, 0, 1, 1, 1)
        .map_err(|error| format!("MLX first-conv probe failed: {error}"))?;
    if mlxcel_core::array_dtype(&output) != mlxcel_core::dtype::BFLOAT16 {
        return Err("MLX first-conv probe did not materialize BF16 output".to_string());
    }
    let output = mlxcel_core::astype(&output, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&output);
    Ok(mlxcel_core::array_to_raw_bytes(&output)
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32 byte width")))
        .collect())
}

fn iree_kernel() -> Vec<f32> {
    let mut kernel = vec![0.0; 3 * 3 * OUTPUT_CHANNELS];
    for output in 0..OUTPUT_CHANNELS {
        for spatial in 0..9 {
            kernel[spatial * OUTPUT_CHANNELS + output] = MLX_WEIGHTS[output][spatial];
        }
    }
    kernel
}

fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

fn required_path(
    name: &str,
    fallback: impl FnOnce() -> Option<PathBuf>,
) -> Result<PathBuf, String> {
    let path = std::env::var_os(name)
        .map(PathBuf::from)
        .or_else(fallback)
        .ok_or_else(|| format!("set {name} for the first-conv oracle"))?;
    if !path.is_file() {
        return Err(format!("{name} is not a file: {}", path.display()));
    }
    Ok(path)
}

fn iree_output() -> Result<Vec<f32>, String> {
    let compiler = required_path("MLXCEL_XLA_IREE_COMPILE", || {
        std::env::var_os("IREE_CUDA_HOME")
            .map(PathBuf::from)
            .map(|home| home.join("venv/bin/iree-compile"))
    })?;
    let runner = required_path("MLXCEL_XLA_IREE_RUN_MODULE", || {
        std::env::var_os("IREE_CUDA_HOME")
            .map(PathBuf::from)
            .map(|home| home.join("build/tools/iree-run-module"))
    })?;
    let stem = format!("mlxcel-gemma3n-sscp-conv-oracle-{}", std::process::id());
    let temporary = std::env::temp_dir();
    let mlir_path = temporary.join(format!("{stem}.mlir"));
    let vmfb_path = temporary.join(format!("{stem}.vmfb"));
    let input_path = temporary.join(format!("{stem}-input.bin"));
    let kernel_path = temporary.join(format!("{stem}-kernel.bin"));
    let output_path = temporary.join(format!("{stem}-output.bin"));
    let paths = [
        mlir_path.as_path(),
        vmfb_path.as_path(),
        input_path.as_path(),
        kernel_path.as_path(),
        output_path.as_path(),
    ];
    let result = (|| {
        std::fs::write(&mlir_path, MLIR)
            .map_err(|error| format!("write {}: {error}", mlir_path.display()))?;
        write_f32(&input_path, &INPUT)?;
        write_f32(&kernel_path, &iree_kernel())?;
        let compiled = Command::new(&compiler)
            .arg("--iree-input-type=stablehlo")
            .arg("--iree-hal-target-device=local")
            .arg("--iree-hal-local-target-device-backends=llvm-cpu")
            .arg(&mlir_path)
            .arg("-o")
            .arg(&vmfb_path)
            .output()
            .map_err(|error| format!("run {}: {error}", compiler.display()))?;
        if !compiled.status.success() {
            return Err(format!(
                "iree-compile failed:\n{}",
                String::from_utf8_lossy(&compiled.stderr)
            ));
        }
        let ran = Command::new(&runner)
            .arg("--device=local-task")
            .arg(format!("--module={}", vmfb_path.display()))
            .arg("--function=main")
            .arg(format!("--input=1x3x3x1xf32=@{}", input_path.display()))
            .arg(format!(
                "--input=3x3x1x{OUTPUT_CHANNELS}xf32=@{}",
                kernel_path.display()
            ))
            .arg(format!("--output=@{}", output_path.display()))
            .output()
            .map_err(|error| format!("run {}: {error}", runner.display()))?;
        if !ran.status.success() {
            return Err(format!(
                "iree-run-module failed:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&ran.stdout),
                String::from_utf8_lossy(&ran.stderr)
            ));
        }
        let bytes = std::fs::read(&output_path)
            .map_err(|error| format!("read {}: {error}", output_path.display()))?;
        if bytes.len() != OUTPUT_CHANNELS * 4 {
            return Err(format!(
                "IREE first-conv probe returned {} bytes, expected {}",
                bytes.len(),
                OUTPUT_CHANNELS * 4
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 byte width")))
            .collect())
    })();
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
    result
}

fn schedule_outcomes(input: &[f32], weights: &[f32]) -> Vec<f32> {
    fn permutations(values: &mut [f32], start: usize, outputs: &mut Vec<f32>) {
        if start == values.len() {
            let left = round_bf16(values.iter().copied().fold(0.0, |sum, value| sum + value));
            let right = round_bf16(
                values
                    .iter()
                    .rev()
                    .copied()
                    .fold(0.0, |sum, value| sum + value),
            );
            outputs.extend([left, right]);
            return;
        }
        for index in start..values.len() {
            values.swap(start, index);
            permutations(values, start + 1, outputs);
            values.swap(start, index);
        }
    }

    let mut products = input
        .iter()
        .zip(weights)
        .map(|(&input, &weight)| round_bf16(input) * round_bf16(weight))
        .filter(|value| *value != 0.0)
        .collect::<Vec<_>>();
    let mut outcomes = Vec::new();
    permutations(&mut products, 0, &mut outcomes);
    outcomes.sort_by_key(|value| value.to_bits());
    outcomes.dedup_by_key(|value| value.to_bits());
    outcomes
}

fn main() -> Result<(), String> {
    let mlx = mlx_output()?;
    let iree = iree_output()?;
    if mlx.len() != OUTPUT_SHAPE.iter().product::<usize>()
        || iree.len() != OUTPUT_SHAPE.iter().product::<usize>()
    {
        return Err(format!(
            "first-conv output shape mismatch: MLX={} IREE={} expected={OUTPUT_SHAPE:?}",
            mlx.len(),
            iree.len()
        ));
    }

    let mut exact = true;
    for output in 0..OUTPUT_CHANNELS {
        let mlx_value = mlx[output];
        let iree_value = iree[output];
        let outcomes = schedule_outcomes(&INPUT, &MLX_WEIGHTS[output]);
        if !outcomes
            .iter()
            .any(|value| value.to_bits() == mlx_value.to_bits())
            || !outcomes
                .iter()
                .any(|value| value.to_bits() == iree_value.to_bits())
        {
            return Err(format!(
                "channel {output} escaped the BF16-input/F32-accumulator schedule envelope: \
                 MLX={mlx_value:?} IREE={iree_value:?} outcomes={outcomes:?}"
            ));
        }
        let distance = bf16_ulp_distance(mlx_value, iree_value)
            .ok_or_else(|| format!("channel {output} did not materialize exact BF16 values"))?;
        eprintln!(
            "gemma3n-sscp-conv-oracle: channel={output} mlx={mlx_value:.9e} \
             iree={iree_value:.9e} mlx_bf16=0x{:04x} iree_bf16=0x{:04x} \
             bf16_ulp={distance} schedule_outcomes={outcomes:?}",
            bf16_bits(mlx_value).expect("checked BF16"),
            bf16_bits(iree_value).expect("checked BF16"),
        );
        exact &= mlx_value.to_bits() == iree_value.to_bits();
    }

    println!(
        "Gemma3n first SSCP convolution MLX↔IREE CPU oracle PASS ({})",
        if exact {
            "exact schedule match"
        } else {
            "backend reduction schedules differ within the enumerated F32 accumulation envelope"
        }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_pins_the_production_first_convolution_dtype_boundary() {
        assert!(MLIR.contains("tensor<1x3x3x1xbf16>"));
        assert!(MLIR.contains("tensor<3x3x1x4xbf16>"));
        assert!(MLIR.contains("-> tensor<1x1x1x4xbf16>"));
        assert!(MLIR.contains("precision DEFAULT"));
        assert!(!MLIR.contains("stablehlo.add"));
    }

    #[test]
    fn cancellation_fixture_distinguishes_valid_f32_reduction_schedules() {
        let outcomes = schedule_outcomes(&INPUT, &MLX_WEIGHTS[0]);
        assert_eq!(outcomes, [0.0, 1.0]);
        assert_eq!(round_bf16(128.0), 128.0);
        assert_eq!(bf16_ulp_distance(128.0, 129.0), Some(1));
    }

    #[test]
    fn iree_kernel_transposes_mlx_output_major_weights() {
        let kernel = iree_kernel();
        for output in 0..OUTPUT_CHANNELS {
            for spatial in 0..9 {
                assert_eq!(
                    kernel[spatial * OUTPUT_CHANNELS + output],
                    MLX_WEIGHTS[output][spatial]
                );
            }
        }
        assert_eq!(INPUT.len(), INPUT_SHAPE.iter().product::<usize>());
    }
}
