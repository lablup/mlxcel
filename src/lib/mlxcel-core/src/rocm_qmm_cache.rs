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

//! Counters of the ROCm QuantizedMatmul dequantized-weight cache (issue
//! #2151).
//!
//! On ROCm, a quantized GEMM that takes the dequantize + GEMM route keeps the
//! dequantized weight in a process-wide LRU (8 matrices or 256 MB by default,
//! `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE` and `MLX_ROCM_QMM_DEQUANT_CACHE_MAX_BYTES`).
//! bf16 GEMMs at or above the fused WMMA kernel's row ceiling take the same
//! GEMM without the cache, and count as bypasses. These counters say whether
//! the cache earns the memory it holds. `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1`
//! prints the same numbers on stderr when the process exits, which is how a
//! benchmark run reads them.

/// A snapshot of the cache's process-wide counters and its current size.
/// All zero on every backend other than ROCm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DequantCacheStats {
    /// Lookups that found the dequantized weight.
    pub hits: u64,
    /// Lookups that did not, after which the weight was dequantized.
    pub misses: u64,
    /// Entries stored, including a refresh of an entry under the same key.
    pub inserts: u64,
    /// Entries dropped to keep the cache within its limits.
    pub evictions: u64,
    /// bf16 GEMMs above the WMMA row ceiling, dequantized without the cache.
    pub bypasses: u64,
    /// Entries the cache holds now.
    pub entries: u64,
    /// Bytes of dequantized weights the cache holds now.
    pub bytes: u64,
}

/// Reads the counters. Cheap: seven relaxed atomic loads and the cache's
/// size under its lock.
pub fn dequant_cache_stats() -> DequantCacheStats {
    let raw = crate::ffi::rocm_dequant_cache_stats();
    let at = |i: usize| raw.get(i).copied().unwrap_or(0);
    DequantCacheStats {
        hits: at(0),
        misses: at(1),
        inserts: at(2),
        evictions: at(3),
        bypasses: at(4),
        entries: at(5),
        bytes: at(6),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{GpuBackendKind, gpu_backend_kind};

    #[test]
    fn stats_are_zero_off_rocm() {
        if gpu_backend_kind() == GpuBackendKind::Rocm {
            eprintln!("skipping: ROCm counts real GEMMs");
            return;
        }
        assert_eq!(dequant_cache_stats(), DequantCacheStats::default());
    }
}
