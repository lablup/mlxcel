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

//! Shared tensor wire-format helpers for pipeline runtime paths.
//!
//! Used by: `stage_worker`, remote pipeline runtime, remote stage services

use anyhow::{Result, anyhow, bail};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::distributed::kv_cache_serde::types::mlx_dtype_to_tensor_dtype;
use crate::distributed::kv_cache_serde::{RawTensorData, reconstruct_mlx_array};
use crate::distributed::tensor_protocol::TensorDtype;
use crate::distributed::tensor_serialize::{DimBound, ShapeBound, TensorDecodeLimits};

use super::activation_transfer::ActivationMessage;

/// Most tokens one stage-boundary frame may carry: the sequence axis bound of
/// every tensor a remote stage or coordinator decodes. The same ceiling as
/// `MAX_HANDOFF_PROMPT_TOKENS`. The model's context window is not used,
/// because rope-scaled checkpoints legitimately run past
/// `max_position_embeddings`.
pub const MAX_STAGE_TOKENS_PER_FRAME: u64 = 1 << 20;

/// Floating-point dtypes an activation or logits tensor may arrive in.
const FLOAT_DTYPES: [TensorDtype; 3] = [
    TensorDtype::Float16,
    TensorDtype::BFloat16,
    TensorDtype::Float32,
];

/// What a pipeline peer expects of the tensors it decodes, built once from
/// the loaded model. A `None` field leaves that axis unbounded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageWireLimits {
    /// Last-axis width of the hidden states this stage receives
    /// ([`super::StageExecutor::inbound_activation_width`]).
    pub activation_width: Option<u64>,
    /// Vocabulary size, the last axis of the logits.
    pub vocab_size: Option<u64>,
}

impl StageWireLimits {
    /// Upstream hidden states: `[1, 1..=MAX_STAGE_TOKENS_PER_FRAME, width]`.
    pub fn hidden_states(&self) -> TensorDecodeLimits {
        float_limits(vec![
            DimBound::Exact(1),
            sequence_axis(MAX_STAGE_TOKENS_PER_FRAME),
            exact_or_unbounded(self.activation_width),
        ])
    }

    /// Entry-stage token ids: int32 `[1, 1..=MAX_STAGE_TOKENS_PER_FRAME]`.
    pub fn input_ids(&self) -> TensorDecodeLimits {
        TensorDecodeLimits {
            allow_compressed: false,
            allow_quantized: false,
            dtypes: vec![TensorDtype::Int32],
            shape: ShapeBound::Axes(vec![
                DimBound::Exact(1),
                sequence_axis(MAX_STAGE_TOKENS_PER_FRAME),
            ]),
        }
    }

    /// Attention mask: rank 2 to 4, every axis between 1 and
    /// `MAX_STAGE_TOKENS_PER_FRAME`, boolean or floating point.
    pub fn attention_mask(&self) -> TensorDecodeLimits {
        let mut dtypes = vec![TensorDtype::Bool];
        dtypes.extend(FLOAT_DTYPES);
        TensorDecodeLimits {
            allow_compressed: false,
            allow_quantized: false,
            dtypes,
            shape: ShapeBound::Uniform {
                rank: 2..=4,
                min_dim: 1,
                max_dim: MAX_STAGE_TOKENS_PER_FRAME,
            },
        }
    }

    /// Final-stage logits for an `input_seq_len`-token request:
    /// `[1, 1..=input_seq_len, vocab]`.
    pub fn logits(&self, input_seq_len: u64) -> TensorDecodeLimits {
        float_limits(vec![
            DimBound::Exact(1),
            sequence_axis(input_seq_len),
            exact_or_unbounded(self.vocab_size),
        ])
    }
}

/// A sequence axis of 1 to `max` tokens: an empty tensor is never a valid
/// stage input or output (#2726).
fn sequence_axis(max: u64) -> DimBound {
    DimBound::Within { min: 1, max }
}

fn exact_or_unbounded(dim: Option<u64>) -> DimBound {
    dim.map_or(
        DimBound::Within {
            min: 1,
            max: u64::MAX,
        },
        DimBound::Exact,
    )
}

fn float_limits(axes: Vec<DimBound>) -> TensorDecodeLimits {
    TensorDecodeLimits {
        allow_compressed: false,
        allow_quantized: false,
        dtypes: FLOAT_DTYPES.to_vec(),
        shape: ShapeBound::Axes(axes),
    }
}

pub fn serialize_mlx_array(arr: &MlxArray) -> Result<Vec<u8>> {
    let contiguous = mlxcel_core::contiguous(arr, false);
    let contiguous = contiguous.as_ref().unwrap();
    let shape: Vec<u64> = mlxcel_core::array_shape(contiguous)
        .into_iter()
        .map(|dim| {
            u64::try_from(dim).map_err(|_| anyhow!("negative tensor shape dimension: {}", dim))
        })
        .collect::<Result<_>>()?;
    let dtype = mlx_dtype_to_tensor_dtype(mlxcel_core::array_dtype(contiguous))?;
    let data = mlxcel_core::array_to_raw_bytes(contiguous);
    ActivationMessage::serialize_activation(dtype, &shape, &data)
}

/// Decode a stage-boundary tensor with no shape expectation (in-process
/// paths, whose tensors never left this process).
pub fn deserialize_wire_tensor(wire_bytes: &[u8]) -> Result<UniquePtr<MlxArray>> {
    deserialize_wire_tensor_with_limits(wire_bytes, &TensorDecodeLimits::unrestricted())
}

/// Decode a stage-boundary tensor from a peer, refusing a header outside
/// `limits` (usually one of the [`StageWireLimits`] builders) before any
/// payload byte is copied.
pub fn deserialize_wire_tensor_with_limits(
    wire_bytes: &[u8],
    limits: &TensorDecodeLimits,
) -> Result<UniquePtr<MlxArray>> {
    let tensor = ActivationMessage::deserialize_activation_with_limits(wire_bytes, limits)?;
    let shape: Vec<i32> = tensor
        .shape
        .iter()
        .copied()
        .map(|dim| {
            i32::try_from(dim).map_err(|_| anyhow!("tensor shape dimension too large: {}", dim))
        })
        .collect::<Result<_>>()?;
    // Every dtype, half precision included, is rebuilt in the dtype it was
    // sent in. `from_bytes_f16` is a checkpoint loader: on CUDA it widens
    // float16 to float32 (and bfloat16 too under `MLX_BF16_NATIVE=0`), which
    // turned a float16 activation into a float32 one at every stage boundary.
    // The downstream stage then ran its layers and LM head in float32 while the
    // single-process model stays in float16, so remote pipeline logits drifted
    // from dense ones (issue #2654). A stage boundary must hand the next stage
    // exactly the tensor the previous one produced.
    let dtype = tensor_dtype_to_mlx(tensor.dtype)?;
    // The wire buffer comes from a peer, and `from_bytes` copies
    // `product(shape)` elements out of a raw pointer without looking at the
    // payload length. `reconstruct_mlx_array` checks the payload is exactly
    // `product(shape) * element_size` bytes (an empty shape is a 1-element
    // scalar) before the FFI call; that check is what keeps a short payload
    // from turning into a heap over-read.
    let raw = RawTensorData {
        data: tensor.data,
        shape,
        dtype,
    };
    reconstruct_mlx_array(&raw)
}

pub fn sequence_length(arr: &MlxArray) -> Result<u32> {
    let shape = mlxcel_core::array_shape(arr);
    let seq_len = *shape
        .get(1)
        .ok_or_else(|| anyhow!("activation tensor must have at least 2 dimensions"))?;
    u32::try_from(seq_len).map_err(|_| anyhow!("negative sequence length: {}", seq_len))
}

fn tensor_dtype_to_mlx(dtype: TensorDtype) -> Result<i32> {
    match dtype {
        TensorDtype::Bool => Ok(0),
        TensorDtype::UInt8 => Ok(1),
        TensorDtype::Int8 => Ok(5),
        TensorDtype::Int16 => Ok(6),
        TensorDtype::Int32 => Ok(7),
        TensorDtype::Float16 => Ok(9),
        TensorDtype::Float32 => Ok(10),
        TensorDtype::BFloat16 => Ok(12),
        TensorDtype::Int4 => bail!("int4 wire tensors are not supported for activation payloads"),
    }
}

#[cfg(test)]
#[path = "wire_tensor_tests.rs"]
mod tests;
