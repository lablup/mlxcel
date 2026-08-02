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

//! Matrix-absorbed MLA decode over a compressed-latent KV cache (issue #907).
//!
//! ## The identity
//!
//! DeepSeek-family MLA compresses K and V into one latent `c` of width
//! `kv_lora_rank` (512 in every shipping checkpoint) plus a shared rope stream
//! `k_pe` of width `qk_rope_head_dim` (64). The porting-style implementation
//! up-projects `c` through `kv_b_proj` into full per-head K and V and caches
//! *those*, so the cache costs `num_heads * (qk_head_dim + v_head_dim)` per
//! token. Absorption removes the up-projection from the decode graph instead of
//! removing it from the math:
//!
//! ```text
//!   score:  q_nope . (W_UK c)  ==  (W_UK^T q_nope) . c
//!   output: attn @ (W_UV c)    ==  (attn @ c) @ W_UV
//! ```
//!
//! Folding `W_UK` into the query and `W_UV` after the attention lets decode run
//! directly against `c`, so the cache holds `kv_lora_rank + qk_rope_head_dim`
//! floats per token (576) for **one** latent head rather than the decompressed
//! per-head K/V. For DeepSeek-V2-Lite (16 heads, 192 + 128 per head) that is
//! 5120 -> 576 floats per token per layer, an 8.9x reduction; see
//! [`cache::latent_bytes_per_token`] and [`cache::decompressed_bytes_per_token`],
//! which are the functions the issue's KV-memory table is computed from.
//!
//! The absorbed score is the sum of two dot products, `q_absorbed . ckv` over
//! `kv_lora_rank` and `q_pe . kpe` over `qk_rope_head_dim`, and the attention
//! output lives in the latent space with `num_heads` query heads sharing the
//! single latent "KV head" (an MQA shape with `head_dim = kv_lora_rank`).
//!
//! ## Stages
//!
//! * **Stage 1** ([`decode::absorbed_decode`]) composes the two contractions
//!   out of MLX ops. No custom kernel; the win is cache bandwidth and cache
//!   memory, not arithmetic.
//! * **Stage 2** ([`split_kv`]) cuts the latent range into chunks that produce
//!   `(partial_v, lse)` states and merges them with issue #898's
//!   `paged_attention_merge_states`, reused unchanged. Small batch times long
//!   context is exactly the shape a single-CTA-per-request decode cannot fill,
//!   which is what the split buys.
//!
//! Prefill deliberately stays on the decompressed path: absorption trades a
//! `[L, r] -> [L, H*(nope+v)]` up-projection done once for an
//! `[H, nope, r]` fold applied to every query, which is a win only when `L` is
//! large relative to the number of queries. Prefill also still *writes* the
//! latent cache, so a prefill and a decode step see the same cache.
//!
//! ## Gate
//!
//! Off unless `MLXCEL_MLA_ABSORBED` is set to one of the tree's usual on
//! spellings ([`absorbed_enabled`]). With it unset nothing in this module is
//! constructed, no weight is dequantized, and the families keep the byte-identical
//! decompressed path.
//!
//! ## Cost of the fold
//!
//! `W_UK` and `W_UV` are slices of `kv_b_proj`, which ships quantized in every
//! mlx-community MLA checkpoint. The absorbed query and output contractions are
//! dense batched matmuls, so the fold dequantizes `kv_b_proj` at load time and
//! keeps it dense: `num_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank`
//! elements per layer. On DeepSeek-V2-Lite that is 2.1 M elements per layer, so
//! ~113 MB in f16 across 27 layers against ~28 MB for the 4-bit original. The
//! trade is fixed weight memory for per-token cache memory, and it only pays off
//! past a context length where the cache saving exceeds it; the benchmark
//! harness reports both numbers so the crossover is visible rather than assumed.

use std::sync::OnceLock;

pub mod absorb;
pub mod cache;
pub mod decode;
pub mod split_kv;
pub mod stats;

#[cfg(test)]
pub(crate) mod testkit;

pub use absorb::MlaAbsorbedProjections;
pub use cache::{MlaLatentCache, decompressed_bytes_per_token, latent_bytes_per_token};
pub use decode::{absorb_queries, absorbed_decode};
pub use split_kv::{MlaSplitPlan, absorbed_decode_split_kv};
pub use stats::{MlaDecodePath, MlaDispatchCounts};

/// Environment variable selecting the absorbed MLA decode path. Default off.
pub const ABSORBED_ENV: &str = "MLXCEL_MLA_ABSORBED";

/// Environment variable selecting the Stage 2 split-KV decode path.
///
/// Only consulted when [`ABSORBED_ENV`] is already on, since split-KV operates
/// on the latent cache that only the absorbed path maintains.
pub const SPLIT_KV_ENV: &str = "MLXCEL_MLA_SPLIT_KV";

/// Pure parse of an on/off environment value.
///
/// Accepts the tree's usual on spellings; anything else, including an unset
/// variable, is off, so a typo degrades to the decompressed path rather than
/// silently enabling one the operator did not ask for. Same policy as
/// [`crate::paged_v2::parse_v2_enabled`].
#[must_use]
pub fn parse_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "on" | "yes")
    )
}

/// Whether the absorbed MLA decode path is selected, read once per process.
#[must_use]
pub fn absorbed_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_enabled(std::env::var(ABSORBED_ENV).ok().as_deref()))
}

/// Whether the Stage 2 split-KV decode path is selected, read once per process.
///
/// Implies [`absorbed_enabled`]: split-KV reads the `(ckv, kpe)` cache, which
/// only exists when absorption is on.
#[must_use]
pub fn split_kv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_enabled(std::env::var(SPLIT_KV_ENV).ok().as_deref()))
        && absorbed_enabled()
}

/// The per-layer MLA shape the absorbed path needs.
///
/// Every field is read straight off the family `ModelArgs`; the type exists so
/// the fold, the cache accounting, and the decode graph cannot disagree about
/// which of the four head dimensions is which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaGeometry {
    /// Query heads. All of them share the single latent KV head.
    pub num_heads: usize,
    /// Latent width, `kv_lora_rank`. 512 in every shipping checkpoint.
    pub kv_lora_rank: usize,
    /// Non-rope part of the query/key head dim, the part absorption folds.
    pub qk_nope_head_dim: usize,
    /// Rope part of the query/key head dim, carried uncompressed beside the
    /// latent because RoPE does not commute with the up-projection.
    pub qk_rope_head_dim: usize,
    /// Value head dim, the width the output fold restores.
    pub v_head_dim: usize,
}

impl MlaGeometry {
    /// Rows of `kv_b_proj` that belong to one head: `qk_nope + v`.
    #[must_use]
    pub const fn kv_b_rows_per_head(&self) -> usize {
        self.qk_nope_head_dim + self.v_head_dim
    }

    /// Full query/key head dim, `qk_nope + qk_rope`. The scale denominator.
    #[must_use]
    pub const fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Reject a geometry the absorbed path cannot serve.
    ///
    /// Returned as `Err` rather than asserted so a loader can decline
    /// absorption for one family and keep the decompressed path, which is the
    /// behaviour the issue asks for while the flag is still opt-in.
    pub fn check(&self) -> Result<(), String> {
        for (name, value) in [
            ("num_heads", self.num_heads),
            ("kv_lora_rank", self.kv_lora_rank),
            ("qk_nope_head_dim", self.qk_nope_head_dim),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("v_head_dim", self.v_head_dim),
        ] {
            if value == 0 {
                return Err(format!("mla: {name} must be non-zero, got 0"));
            }
        }
        if self.kv_lora_rank > i32::MAX as usize {
            return Err(format!(
                "mla: kv_lora_rank {} overflows i32",
                self.kv_lora_rank
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LITE: MlaGeometry = MlaGeometry {
        num_heads: 16,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
    };

    #[test]
    fn absorption_is_off_unless_explicitly_enabled() {
        assert!(!parse_enabled(None));
        assert!(!parse_enabled(Some("")));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("off")));
        assert!(!parse_enabled(Some("2")));
        assert!(!parse_enabled(Some("absorbed")));
    }

    #[test]
    fn absorption_accepts_the_usual_on_spellings() {
        for v in ["1", "true", "TRUE", "on", "ON", "yes", " yes "] {
            assert!(parse_enabled(Some(v)), "{v:?} should enable absorption");
        }
    }

    #[test]
    fn geometry_derives_the_two_composite_widths() {
        assert_eq!(LITE.kv_b_rows_per_head(), 256);
        assert_eq!(LITE.q_head_dim(), 192);
        LITE.check().unwrap();
    }

    #[test]
    fn geometry_rejects_a_zero_dimension() {
        let mut g = LITE;
        g.kv_lora_rank = 0;
        let err = g.check().unwrap_err();
        assert!(err.contains("kv_lora_rank"), "{err}");
    }
}
