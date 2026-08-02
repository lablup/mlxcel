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

//! Compressed-latent MLA KV cache (issue #907).
//!
//! ## Why this is a view over [`KVCache`], not a new cache type
//!
//! `LanguageModel::make_caches` returns `Vec<KVCache>` and `forward` takes
//! `&mut [KVCache]`, so a genuinely new cache type forces every family that
//! uses it into the `ModelOwnedSequenceState` escape hatch that the SSM hybrids
//! use, which in turn costs `supports_batching`, the paged decode backend, and
//! the generic prompt-cache donation path. That is a large regression to pay
//! for a change that only alters *what* the two per-layer tensors contain.
//!
//! The FP16 update path stores keys and values as independent `[B, H, L, D]`
//! buffers and reads each one's head dimension from its own shape (see
//! `KVCache::update_fp16`), so a cache whose "keys" are `[B, 1, L,
//! kv_lora_rank]` and whose "values" are `[B, 1, L, qk_rope_head_dim]` is
//! already expressible. This type is the named, checked view of that packing:
//!
//! | slot     | holds | shape                              |
//! |----------|-------|------------------------------------|
//! | `keys`   | `ckv` | `[B, 1, L, kv_lora_rank]`          |
//! | `values` | `kpe` | `[B, 1, L, qk_rope_head_dim]`      |
//!
//! The slot assignment matches what `src/models/deepseek_v3.rs` has always
//! done (`cache.update_and_fetch(kv_latent, k_pe)`), so V2 folded onto this
//! type and V3's existing code describe the same buffer layout.
//!
//! Everything `KVCache` provides keeps working unchanged: `offset` and
//! `live_start` stay the RoPE position and live-window start, `trim` and
//! `trim_front` operate on the latent rows exactly as they did on the
//! decompressed rows, `nbytes` measures the real (now much smaller) buffers,
//! and speculative decode's `can_trim_prompt_cache` still reports trimmable.
//!
//! ## Modes this declines
//!
//! Only `KVCacheMode::Fp16` and non-paged backing. The INT8 and Turbo3/Turbo4
//! modes quantize along the head dimension with a per-token scale, which is
//! calibrated for a per-head K/V row, not for a 512-wide shared latent whose
//! reconstruction error is amplified by every one of the `num_heads` query
//! heads that read it. The paged pool allocates `[num_blocks, page_size, Hkv,
//! head_dim]` tensors with one head dim for both sides, which the asymmetric
//! `(512, 64)` split cannot fill. Both are declined at wrap time with a
//! message rather than silently mis-served.

use cxx::UniquePtr;

use crate::cache::{KVCache, KVCacheMode};
use crate::ffi::MlxArray;
use crate::mla::MlaGeometry;

/// Bytes one token of the compressed-latent cache costs, per layer.
///
/// One latent "head" of `kv_lora_rank` plus the shared rope stream of
/// `qk_rope_head_dim`, independent of `num_heads`.
#[must_use]
pub const fn latent_bytes_per_token(geometry: &MlaGeometry, bytes_per_element: usize) -> usize {
    (geometry.kv_lora_rank + geometry.qk_rope_head_dim) * bytes_per_element
}

/// Bytes one token of the decompressed cache costs, per layer.
///
/// The pre-absorption layout: `num_heads` keys of `qk_nope + qk_rope` (the
/// shared rope stream is repeated into every head) and `num_heads` values of
/// `v_head_dim`.
#[must_use]
pub const fn decompressed_bytes_per_token(
    geometry: &MlaGeometry,
    bytes_per_element: usize,
) -> usize {
    geometry.num_heads * (geometry.q_head_dim() + geometry.v_head_dim) * bytes_per_element
}

/// A [`KVCache`] used as a compressed-latent MLA cache.
///
/// Borrowed rather than owned so a family keeps handing `&mut [KVCache]`
/// around and wraps per layer at the point of use.
pub struct MlaLatentCache<'a> {
    inner: &'a mut KVCache,
    geometry: MlaGeometry,
}

impl<'a> MlaLatentCache<'a> {
    /// View `cache` as a latent cache, or explain why it cannot be.
    pub fn wrap(cache: &'a mut KVCache, geometry: MlaGeometry) -> Result<Self, String> {
        geometry.check()?;
        if cache.mode != KVCacheMode::Fp16 {
            return Err(format!(
                "mla: absorbed decode needs an FP16 KV cache; this cache is {} and its per-token \
                 quantization is calibrated for per-head K/V rows, not a shared latent",
                cache.mode
            ));
        }
        if cache.is_paged_backed() {
            return Err(
                "mla: absorbed decode cannot use a paged-backed cache; the pool's \
                 [blocks, page_size, Hkv, head_dim] layout has one head dim for both sides and \
                 the latent split is asymmetric"
                    .to_string(),
            );
        }
        Ok(Self {
            inner: cache,
            geometry,
        })
    }

    /// Live token count, i.e. `offset - live_start`.
    #[must_use]
    pub fn seq_len(&self) -> i32 {
        self.inner.seq_len()
    }

    /// Monotonic write position, the RoPE position for the next token.
    #[must_use]
    pub fn offset(&self) -> i32 {
        self.inner.offset
    }

    /// The geometry this view was wrapped for.
    #[must_use]
    pub const fn geometry(&self) -> MlaGeometry {
        self.geometry
    }

    /// Append one step's latent and rope rows, and read the whole live window
    /// back.
    ///
    /// `ckv` is `[B, 1, L, kv_lora_rank]` and `kpe` is `[B, 1, L,
    /// qk_rope_head_dim]`, both already normalized and rotated respectively by
    /// the caller. Returns `(ckv_all, kpe_all)` over the live window.
    pub fn update_and_fetch(
        &mut self,
        ckv: UniquePtr<MlxArray>,
        kpe: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.inner.update_and_fetch(ckv, kpe)
    }

    /// Bytes per token this cache costs, at the element size it stores.
    #[must_use]
    pub const fn bytes_per_token(&self, bytes_per_element: usize) -> usize {
        latent_bytes_per_token(&self.geometry, bytes_per_element)
    }
}

impl std::fmt::Debug for MlaLatentCache<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlaLatentCache")
            .field("geometry", &self.geometry)
            .field("seq_len", &self.seq_len())
            .finish()
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

    /// DeepSeek-V3 geometry: 128 heads make the decompressed layout eight times
    /// worse again while the latent side does not move at all, which is the
    /// whole point of the accounting.
    const V3: MlaGeometry = MlaGeometry {
        num_heads: 128,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
    };

    #[test]
    fn latent_bytes_per_token_is_the_issue_s_576_floats() {
        assert_eq!(latent_bytes_per_token(&LITE, 1), 576);
        assert_eq!(latent_bytes_per_token(&LITE, 2), 1152);
        // Independent of head count, unlike the decompressed layout.
        assert_eq!(
            latent_bytes_per_token(&V3, 2),
            latent_bytes_per_token(&LITE, 2)
        );
    }

    #[test]
    fn decompressed_bytes_per_token_scales_with_head_count() {
        // 16 heads * (192 key + 128 value) = 5120 elements.
        assert_eq!(decompressed_bytes_per_token(&LITE, 1), 5120);
        assert_eq!(decompressed_bytes_per_token(&V3, 1), 40960);
    }

    #[test]
    fn absorption_reduces_the_per_token_cache_by_the_head_count_ratio() {
        let before = decompressed_bytes_per_token(&LITE, 2) as f64;
        let after = latent_bytes_per_token(&LITE, 2) as f64;
        assert!(
            (before / after - 8.888).abs() < 0.01,
            "DeepSeek-V2-Lite ratio was {}",
            before / after
        );
        let before_v3 = decompressed_bytes_per_token(&V3, 2) as f64;
        let after_v3 = latent_bytes_per_token(&V3, 2) as f64;
        assert!(
            (before_v3 / after_v3 - 71.1).abs() < 0.1,
            "DeepSeek-V3 ratio was {}",
            before_v3 / after_v3
        );
    }

    #[test]
    fn wrap_declines_a_quantized_kv_cache() {
        let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
        let err = MlaLatentCache::wrap(&mut cache, LITE).unwrap_err();
        assert!(err.contains("FP16"), "{err}");
    }

    #[test]
    fn wrap_accepts_the_default_fp16_cache() {
        let mut cache = KVCache::new();
        let view = MlaLatentCache::wrap(&mut cache, LITE).unwrap();
        assert_eq!(view.seq_len(), 0);
        assert_eq!(view.bytes_per_token(2), 1152);
    }
}
