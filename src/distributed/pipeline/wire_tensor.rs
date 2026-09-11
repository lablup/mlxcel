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
use crate::distributed::tensor_protocol::TensorDtype;

use super::activation_transfer::ActivationMessage;

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

pub fn deserialize_wire_tensor(wire_bytes: &[u8]) -> Result<UniquePtr<MlxArray>> {
    let tensor = ActivationMessage::deserialize_activation(wire_bytes)?;
    let shape: Vec<i32> = tensor
        .shape
        .iter()
        .copied()
        .map(|dim| {
            i32::try_from(dim).map_err(|_| anyhow!("tensor shape dimension too large: {}", dim))
        })
        .collect::<Result<_>>()?;
    match tensor.dtype {
        TensorDtype::Float16 => Ok(mlxcel_core::from_bytes_f16(&tensor.data, &shape, false)),
        TensorDtype::BFloat16 => Ok(mlxcel_core::from_bytes_f16(&tensor.data, &shape, true)),
        other => {
            let dtype = tensor_dtype_to_mlx(other)?;
            Ok(mlxcel_core::from_bytes(&tensor.data, &shape, dtype))
        }
    }
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
