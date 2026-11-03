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

//! Exact YOLO11 tensor-schema loading.
//!
//! Every expected tensor is taken exactly once with its expected shape, and
//! [`Loader::finish`] reports any key the graph never consumed. That makes a
//! weight for a concat or upsample layer, or a layer index past the head, fail
//! at load time with the offending key instead of silently going unused.

use std::collections::HashSet;

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const MAX_PARAMETER_ELEMENTS: usize = 256 * 1024 * 1024;

pub(crate) struct Loader {
    weights: WeightMap,
    used: HashSet<String>,
    dtype: i32,
}

impl Loader {
    pub(crate) fn new(weights: WeightMap) -> Result<Self, String> {
        if weights.is_empty() {
            return Err("YOLO11 checkpoint has no tensors".into());
        }
        let dtype = weights
            .values()
            .next()
            .map(|value| mlxcel_core::array_dtype(value))
            .ok_or_else(|| "YOLO11 checkpoint has no tensors".to_string())?;
        if !matches!(
            dtype,
            mlxcel_core::dtype::FLOAT16
                | mlxcel_core::dtype::BFLOAT16
                | mlxcel_core::dtype::FLOAT32
        ) {
            return Err(format!(
                "YOLO11 checkpoint tensors must be floating point, got dtype {dtype}"
            ));
        }
        let mut elements = 0usize;
        for (key, value) in &weights {
            let actual = mlxcel_core::array_dtype(value);
            if actual != dtype {
                return Err(format!(
                    "YOLO11 weight {key} has dtype {actual}; expected checkpoint dtype {dtype}"
                ));
            }
            let count: usize = mlxcel_core::array_shape(value)
                .iter()
                .map(|&dimension| dimension.max(0) as usize)
                .product();
            elements = elements
                .saturating_add(count)
                .min(MAX_PARAMETER_ELEMENTS + 1);
        }
        if elements > MAX_PARAMETER_ELEMENTS {
            return Err(format!(
                "YOLO11 checkpoint exceeds the {MAX_PARAMETER_ELEMENTS} element limit"
            ));
        }
        let capacity = weights.len();
        Ok(Self {
            weights,
            used: HashSet::with_capacity(capacity),
            dtype,
        })
    }

    pub(crate) fn dtype(&self) -> i32 {
        self.dtype
    }

    /// Take one tensor, rejecting a missing key, a mismatched shape or a
    /// second take of the same key.
    pub(crate) fn take(
        &mut self,
        key: &str,
        shape: &[usize],
    ) -> Result<UniquePtr<MlxArray>, String> {
        if !self.used.insert(key.to_string()) {
            return Err(format!("YOLO11 weight {key} is loaded twice"));
        }
        let value = self
            .weights
            .remove(key)
            .ok_or_else(|| format!("YOLO11 weight not found: {key}"))?;
        let expected: Vec<i32> = shape
            .iter()
            .map(|&dimension| {
                i32::try_from(dimension).map_err(|_| {
                    format!("YOLO11 expected shape for {key} exceeds the i32 backend limit")
                })
            })
            .collect::<Result<_, _>>()?;
        let actual = mlxcel_core::array_shape(&value);
        if actual != expected {
            return Err(format!(
                "YOLO11 weight {key} has shape {actual:?}; expected {expected:?}"
            ));
        }
        Ok(value)
    }

    pub(crate) fn finish(self) -> Result<(), String> {
        if !self.weights.is_empty() {
            let mut unexpected: Vec<&str> = self.weights.keys().map(String::as_str).collect();
            unexpected.sort_unstable();
            return Err(format!(
                "YOLO11 checkpoint has {} unexpected tensor(s): {}",
                unexpected.len(),
                unexpected
                    .into_iter()
                    .take(8)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        Ok(())
    }
}
