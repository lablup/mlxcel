// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! GlobalPointer reading-order head.

use mlxcel_core::layers::Linear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::PpDocLayoutV3Config;

pub struct GlobalPointer {
    dense: Linear,
    head_size: i32,
}

impl GlobalPointer {
    pub fn from_weights(weights: &WeightMap, config: &PpDocLayoutV3Config) -> Result<Self, String> {
        Ok(Self {
            dense: Linear::from_weights(weights, "decoder_global_pointer.dense")?,
            head_size: config.global_pointer_head_size as i32,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (batch, queries) = (shape[0], shape[1]);
        let qk = mlxcel_core::reshape(&self.dense.forward(x), &[batch, queries, 2, self.head_size]);
        let queries_tensor =
            mlxcel_core::slice(&qk, &[0, 0, 0, 0], &[batch, queries, 1, self.head_size]);
        let keys_tensor =
            mlxcel_core::slice(&qk, &[0, 0, 1, 0], &[batch, queries, 2, self.head_size]);
        let queries_tensor =
            mlxcel_core::reshape(&queries_tensor, &[batch, queries, self.head_size]);
        let keys_tensor = mlxcel_core::reshape(&keys_tensor, &[batch, queries, self.head_size]);
        let keys_tensor = mlxcel_core::transpose_axes(&keys_tensor, &[0, 2, 1]);
        let scores = mlxcel_core::multiply_scalar(
            &mlxcel_core::matmul(&queries_tensor, &keys_tensor),
            (self.head_size as f32).powf(-0.5),
        );
        let lower = mlxcel_core::tril(
            &mlxcel_core::ones(&[queries, queries], mlxcel_core::dtype::BOOL),
            0,
        );
        let lower = mlxcel_core::expand_dims(&lower, 0);
        let masked = mlxcel_core::full_f32(&[1], -1.0e4, mlxcel_core::array_dtype(&scores));
        mlxcel_core::where_cond(&lower, &masked, &scores)
    }
}
