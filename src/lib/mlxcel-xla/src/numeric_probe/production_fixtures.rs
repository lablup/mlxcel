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

//! Deterministic operands for the production-MLX probes (issue #3212).
//!
//! Every value is a small dyadic rational, so it is exact in the narrow dtype
//! the operator materializes (F16 for the Q4 projections, BF16 for the SSCP
//! convolution). Both backends therefore start from bit-identical operands and
//! any difference in the report comes from the operation itself.

use crate::emitter::production_probe_ops::SscpConvolutionProbeSpec;
use crate::numeric_oracle::NumericTensor;

/// SplitMix64: a fixed, dependency-free stream for reproducible operands.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Uniform integer in `low..=high`.
    fn integer(&mut self, low: i64, high: i64) -> i64 {
        let span = u64::try_from(high - low + 1).expect("non-empty fixture range");
        low + i64::try_from(self.next() % span).expect("fixture range fits i64")
    }

    /// `k / denominator` for uniform `k` in `low..=high`.
    fn dyadic(&mut self, low: i64, high: i64, denominator: f32) -> f32 {
        self.integer(low, high) as f32 / denominator
    }
}

/// `[input [rows, inputs], lanes [outputs, inputs], scales, biases]`.
///
/// Activations are `k / 1024` in `[-1, 1]`, scales `(16 + k) / 2048` in
/// `[2^-7, 2^-5)`, and biases `-k / 256` in `(-2^-4, 0]`, all exact in F16 and
/// shaped like released affine Q4 metadata.
pub(super) fn affine_q4_operands(
    rows: usize,
    outputs: usize,
    inputs: usize,
    group_size: usize,
) -> Result<Vec<NumericTensor>, String> {
    let groups = inputs / group_size;
    let mut stream = Stream(0x3212_0004 ^ rows as u64);
    let input = (0..rows * inputs)
        .map(|_| stream.dyadic(-1024, 1024, 1024.0))
        .collect();
    let lanes = (0..outputs * inputs)
        .map(|_| stream.integer(0, 15) as f32)
        .collect();
    let scales = (0..outputs * groups)
        .map(|_| (16.0 + stream.integer(0, 47) as f32) / 2048.0)
        .collect();
    let biases = (0..outputs * groups)
        .map(|_| -(stream.integer(0, 15) as f32) / 256.0)
        .collect();
    Ok(vec![
        NumericTensor::new("input", vec![rows, inputs], input)?,
        NumericTensor::new("quantized_lanes", vec![outputs, inputs], lanes)?,
        NumericTensor::new("f16_scales", vec![outputs, groups], scales)?,
        NumericTensor::new("f16_biases", vec![outputs, groups], biases)?,
    ])
}

/// `[mel [1, time, frequency, 1], weight [channels, kh, kw, 1]]`.
///
/// Mel values are `k / 32` in `[-8, 4]` (log-mel range) and weights `k / 512`
/// in `[-0.25, 0.25)`; both carry at most eight significant bits, so they are
/// exact in BF16.
pub(super) fn sscp_convolution_operands(
    spec: SscpConvolutionProbeSpec,
) -> Result<Vec<NumericTensor>, String> {
    let mut stream = Stream(0x3212_5c5c);
    let mel = (0..spec.time * spec.frequency)
        .map(|_| stream.dyadic(-256, 128, 32.0))
        .collect();
    let weight = (0..spec.channels * spec.kernel[0] * spec.kernel[1])
        .map(|_| stream.dyadic(-128, 127, 512.0))
        .collect();
    Ok(vec![
        NumericTensor::new("mel", vec![1, spec.time, spec.frequency, 1], mel)?,
        NumericTensor::new(
            "weight",
            vec![spec.channels, spec.kernel[0], spec.kernel[1], 1],
            weight,
        )?,
    ])
}
