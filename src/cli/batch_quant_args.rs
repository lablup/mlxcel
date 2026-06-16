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

//! Continuous-batching KV quantization CLI flag group.
//!
//! This module owns the canonical clap definition for the `--kv-bits`,
//! `--kv-group-size`, `--kv-quant-scheme`, and `--kv-skip-last-layer` flags
//! that drive the batch KV quantization path inside the server worker.
//!
//! Both server binaries (`mlxcel serve` and `mlxcel-server`) flatten
//! [`BatchKvQuantArgs`] via `#[command(flatten)]`, which means the
//! user-visible `--help` text is identical across them and adding, renaming,
//! or extending one of these flags only requires editing this file.
//!
//! The single-shot `mlxcel generate` surface intentionally does NOT flatten
//! this group because the continuous-batching scheduler is the only consumer
//! of [`mlxcel_core::cache::BatchKvQuantConfig`]; surfacing the flag would
//! be confusing on the offline path.
//!
//! Used by: mlxcel serve, mlxcel-server.
//!
//! Resolution helpers ([`crate::server::resolve_batch_kv_quant_config`] and
//! the `env_fallback_kv_*` family) live next to [`crate::server::ServerStartupInput`]
//! because they only make sense on the server pipeline.

use clap::Args;

/// Shared continuous-batching KV quantization flag group.
///
/// Flattened into the clap `Args` struct of both server binaries. The four
/// flags below MUST stay in sync across both callers.
#[derive(Args, Debug, Clone)]
#[command(next_help_heading = "Batch KV Quantization Options")]
pub struct BatchKvQuantArgs {
    /// KV cache quantization bit count for the continuous-batching path.
    ///
    /// `0` (default) disables KV cache quantization on the batched decode
    /// path. Valid values:
    ///   `8` for `--kv-quant-scheme uniform`;
    ///   `4` only for `--kv-quant-scheme turboquant`.
    ///
    /// The `(uniform, 4)` combination is rejected today because mlxcel-core
    /// has no `Int4Affine` single-stream cache mode; use
    /// `--kv-quant-scheme turboquant` for 4-bit batched KV cache or
    /// `--kv-bits 8` for uniform 8-bit.
    ///
    /// Also read from `LLAMA_ARG_KV_BITS`.
    #[arg(
        long = "kv-bits",
        default_value_t = 0,
        env = "LLAMA_ARG_KV_BITS",
        value_name = "BITS"
    )]
    pub kv_bits: i32,

    /// Channel-group size for the uniform `mx.quantize`-style KV cache.
    ///
    /// Default: 64 (`mlxcel_core::cache::DEFAULT_KV_GROUP_SIZE`). Ignored
    /// when `--kv-quant-scheme turboquant` is in effect. Must divide every
    /// model's V `head_dim` at runtime.
    ///
    /// Also read from `LLAMA_ARG_KV_GROUP_SIZE`.
    #[arg(
        long = "kv-group-size",
        default_value_t = mlxcel_core::cache::DEFAULT_KV_GROUP_SIZE,
        env = "LLAMA_ARG_KV_GROUP_SIZE",
        value_name = "N"
    )]
    pub kv_group_size: i32,

    /// KV cache quantization scheme for the continuous-batching path.
    ///
    /// Accepted values:
    ///   `uniform`    : `mx.quantize`-style affine codec (default).
    ///   `turboquant` : 4-bit V on FP16 K via TurboQuant.
    ///
    /// Inert when `--kv-bits 0`.
    ///
    /// Also read from `LLAMA_ARG_KV_QUANT_SCHEME`.
    #[arg(
        long = "kv-quant-scheme",
        env = "LLAMA_ARG_KV_QUANT_SCHEME",
        value_name = "SCHEME"
    )]
    pub kv_quant_scheme: Option<String>,

    /// Skip the final transformer layer when KV cache quantization is
    /// enabled.
    ///
    /// Default: `true` (preserves quality on deep models such as
    /// gemma-4-31b). Set to `false` to opt out for benchmarking.
    ///
    /// Also read from `LLAMA_ARG_KV_SKIP_LAST_LAYER` and
    /// `MLXCEL_KV_SKIP_LAST_LAYER`.
    #[arg(
        long = "kv-skip-last-layer",
        default_value_t = true,
        env = "LLAMA_ARG_KV_SKIP_LAST_LAYER",
        action = clap::ArgAction::Set,
        value_name = "BOOL"
    )]
    pub kv_skip_last_layer: bool,
}

impl Default for BatchKvQuantArgs {
    fn default() -> Self {
        Self {
            kv_bits: 0,
            kv_group_size: mlxcel_core::cache::DEFAULT_KV_GROUP_SIZE,
            kv_quant_scheme: None,
            kv_skip_last_layer: true,
        }
    }
}
