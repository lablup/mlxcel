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

//! The packed ternary linears of `transformer-packed-mflux/`.
//!
//! Upstream `_load_packed_or_dense_linear`
//! (<https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/bonsai/klein_fast/loader.py>):
//! a linear with sibling `.scales` and `.biases` is a `uint32` packed affine
//! weight run through `mx.quantized_matmul`, everything else is dense bf16.
//! The scales and biases ship as bf16, the compute dtype, so they are kept as
//! stored. This source loads the packed ones as affine [`UnifiedLinear`]s
//! with the file's `bits` and `group_size`, which is the same
//! `quantized_matmul` upstream's megakernel calls, and the FLUX.2 transformer
//! is reused unchanged.

use std::collections::HashSet;

use anyhow::{Context, Result, anyhow, bail};
use mlxcel_core::dtype;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;

use super::config::{PackedQuantConfig, QUANT_CONFIG_FILE};
use crate::imagegen::flux2::embeddings::{Flux2Linear, LinearSource};
use crate::imagegen::flux2::transformer::rename_key;

/// [`LinearSource`] for Bonsai: the modules `quantized_modules` lists are
/// packed affine, every other linear is dense, and the file must agree.
#[derive(Debug, Clone)]
pub struct PackedLinears {
    config: PackedQuantConfig,
    /// `config.modules` under the names the transformer loads them by.
    renamed: HashSet<String>,
}

impl PackedLinears {
    #[must_use]
    pub fn new(config: PackedQuantConfig) -> Self {
        let renamed = config
            .modules
            .iter()
            .map(|m| {
                let key = rename_key(&format!("{m}.weight"));
                key.strip_suffix(".weight").unwrap_or(&key).to_string()
            })
            .collect();
        Self { config, renamed }
    }

    #[must_use]
    pub fn config(&self) -> &PackedQuantConfig {
        &self.config
    }

    /// Load the linear at `prefix` (a renamed key, as the transformer asks).
    pub fn load_linear(&self, weights: &WeightMap, prefix: &str) -> Result<UnifiedLinear> {
        let has_scales = weights.contains_key(&format!("{prefix}.scales"));
        let q = self.config.quant;
        if !self.renamed.contains(prefix) {
            if has_scales {
                bail!(
                    "Bonsai module {prefix} has .scales but is not in {QUANT_CONFIG_FILE} \
                     quantized_modules"
                );
            }
            return UnifiedLinear::from_weights_with_mode(weights, prefix, q.group_size, q.bits, "affine")
                .map_err(|e| anyhow!(e))
                .with_context(|| format!("Bonsai transformer weight {prefix}"));
        }
        for suffix in ["scales", "biases"] {
            if !weights.contains_key(&format!("{prefix}.{suffix}")) {
                bail!(
                    "Bonsai module {prefix} is listed in {QUANT_CONFIG_FILE} quantized_modules \
                     but has no .{suffix}"
                );
            }
        }
        let layer =
            UnifiedLinear::from_weights_with_mode(weights, prefix, q.group_size, q.bits, "affine")
                .map_err(|e| anyhow!(e))
                .with_context(|| format!("Bonsai transformer weight {prefix}"))?;
        let qw = layer
            .quantized_weight()
            .with_context(|| format!("Bonsai module {prefix} did not load as quantized"))?;
        if (qw.bits, qw.group_size) != (q.bits, q.group_size) {
            bail!(
                "Bonsai module {prefix} has the shapes of {}-bit group {} weights, but \
                 {QUANT_CONFIG_FILE} declares {}-bit group {}",
                qw.bits,
                qw.group_size,
                q.bits,
                q.group_size
            );
        }
        Ok(layer)
    }
}

impl LinearSource for PackedLinears {
    fn load(&self, weights: &WeightMap, prefix: &str) -> Result<Box<dyn Flux2Linear>> {
        Ok(Box::new(self.load_linear(weights, prefix)?))
    }

    /// Every listed module has a `uint32` weight, `.scales` and `.biases`, and
    /// no unlisted tensor has `.scales`, so the list and the file agree.
    fn check_weights(&self, raw: &WeightMap) -> Result<()> {
        for module in &self.config.modules {
            for suffix in ["weight", "scales", "biases"] {
                if !raw.contains_key(&format!("{module}.{suffix}")) {
                    bail!(
                        "Bonsai module {module} is listed in {QUANT_CONFIG_FILE} \
                         quantized_modules but has no .{suffix}"
                    );
                }
            }
            let weight = &raw[&format!("{module}.weight")];
            if mlxcel_core::array_dtype(weight) != dtype::UINT32 {
                bail!("Bonsai module {module} is listed as packed but its weight is not uint32");
            }
        }
        let mut unlisted: Vec<&str> = raw
            .keys()
            .filter_map(|k| k.strip_suffix(".scales"))
            .filter(|m| !self.config.modules.contains(*m))
            .collect();
        unlisted.sort_unstable();
        if let Some(first) = unlisted.first() {
            bail!(
                "Bonsai module {first} has .scales but is not in {QUANT_CONFIG_FILE} \
                 quantized_modules ({} such modules)",
                unlisted.len()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;
