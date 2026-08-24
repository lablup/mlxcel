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
//! calibrated for a per-head K/V row, not for a shared latent hundreds of
//! elements wide whose reconstruction error is amplified by every one of the
//! `num_heads` query heads that read it. The paged pool allocates
//! `[num_blocks, page_size, Hkv, head_dim]` tensors with one head dim for both
//! sides, which the asymmetric `(512, 64)` split cannot fill. Both are declined
//! at wrap time with a message rather than silently mis-served.

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

/// `model_type` values whose attention stores an MLA `(kv_latent, k_pe)` pair
/// in one shared [`KVCache`] on **every** step, with no decompressed fallback.
///
/// Each of these families calls `cache.update_and_fetch(kv_latent, k_pe)`
/// unconditionally, so the cache's "K" slot is at least `kv_lora_rank` wide and
/// its "V" slot is `qk_rope_head_dim` wide (64). Neither slot is a per-head K/V
/// row. That is the packing [`MlaLatentCache`] names, and the reason
/// [`latent_layout_supports_mode`] is the rule for all of them.
///
/// The "K" slot is not one fixed width across the list. `kv_lora_rank` is 512
/// in the shipping checkpoints of every family here, and it is the serde
/// default wherever one is declared. On top of that, `deepseek_v32`,
/// `deepseek_v3.2` and `glm_moe_dsa` concatenate the `index_head_dim`-wide DSA
/// indexer key onto the same slot before caching it
/// (`src/models/deepseek_v32.rs`), which makes it 640 wide at the default
/// `index_head_dim` of 128. A codebook and a sign vector calibrated from the
/// 64-wide "V" slot are wrong for any of those widths.
///
/// `deepseek_v2` is deliberately **absent**. It asks
/// [`MlaLatentCache::supports`] first and falls back to the decompressed
/// per-head K/V layout when the cache is not FP16 (`src/models/deepseek_v2.rs`),
/// so a quantized mode there is applied to ordinary per-head rows and needs no
/// family-level override. Adding it here would take away a working
/// configuration rather than fixing a broken one.
///
/// Used by: [`caches_mla_latent_pair`].
pub static MLA_LATENT_CACHE_FAMILIES: &[&str] = &[
    "glm4_moe_lite",
    "deepseek_v3",
    // `deepseek_v32` and the `deepseek_v3.2` spelling of the same
    // `config.json` value both resolve to `DeepSeekV32Model`
    // (`src/models/detection.rs`), and `glm_moe_dsa` wraps that model
    // wholesale (`src/models/glm_moe_dsa.rs`). All three share the single
    // unconditional `cache.update_and_fetch(cache_keys, k_pe)` in
    // `src/models/deepseek_v32.rs` and consult neither
    // [`MlaLatentCache::supports`] nor the cache's mode, so they have no
    // decompressed fallback to land on.
    "deepseek_v32",
    "deepseek_v3.2",
    "glm_moe_dsa",
    "kimi_linear",
    "longcat_flash_ngram",
];

/// Whether `model_type` names a family that always caches an MLA latent pair.
///
/// `model_type` is the lowercase `config.json` string, matching the canonical
/// key used by the model detection path. The comparison is exact rather than a
/// prefix walk: the latent packing is a property of one concrete attention
/// implementation, so a neighbouring `model_type` that merely shares a prefix
/// must not inherit the answer.
///
/// Used by: `cache::turbo::allowlist::resolve_kv_cache_mode_for_model`.
#[must_use]
pub fn caches_mla_latent_pair(model_type: &str) -> bool {
    let needle = model_type.trim().to_ascii_lowercase();
    MLA_LATENT_CACHE_FAMILIES
        .iter()
        .any(|&family| needle == family)
}

/// The rule: an MLA latent cache holds FP16 and nothing else.
///
/// Every quantized [`KVCacheMode`] calibrates a per-token codebook from one
/// head dimension and applies it to both stored streams. A latent cache has two
/// different widths in those two slots (`kv_lora_rank` and
/// `qk_rope_head_dim`), and neither slot is a per-head K/V row:
///
/// * Symmetric `Turbo4` derives its sign vectors and codebook from the "V"
///   width and then applies them to the "K" slot as well, which is
///   `kv_lora_rank` wide. The sign vectors are read past their end.
/// * The asymmetric modes quantize only the "V" slot, but here that slot holds
///   `k_pe`, the RoPE **key** stream. They therefore compress a key and leave
///   the latent (which is both K and V after absorption) exact, which is the
///   opposite of the "FP16 K, quantized V" contract those modes document.
///
/// Stated once here so [`MlaLatentCache::supports`] and the model-level mode
/// resolver cannot drift apart. Returns the operator-facing reason on refusal.
///
/// Used by: [`MlaLatentCache::supports`],
/// `cache::turbo::allowlist::resolve_kv_cache_mode_for_model`.
pub fn latent_layout_supports_mode(mode: KVCacheMode) -> Result<(), String> {
    if mode == KVCacheMode::Fp16 {
        return Ok(());
    }
    Err(format!(
        "mla: absorbed decode needs an FP16 KV cache; this cache is {mode} and its per-token \
         quantization is calibrated for per-head K/V rows, not a shared latent"
    ))
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
    /// Whether `cache` can hold a latent, without borrowing it mutably.
    ///
    /// Separate from [`Self::wrap`] because a caller that wants to fall back to
    /// the decompressed path on failure cannot ask `wrap` first: the borrow
    /// checker keeps the `&mut` alive through the `Err` arm of a `Result`
    /// carrying the borrow's lifetime, so the fallback could not touch the
    /// cache. Asking this first keeps the fallback expressible.
    ///
    /// The answer is stable for the life of a cache (mode and backing are set
    /// at construction), so a family that takes the fallback on step one takes
    /// it on every step, and the cache never ends up holding a mix of latent
    /// and decompressed rows.
    pub fn supports(cache: &KVCache, geometry: MlaGeometry) -> Result<(), String> {
        geometry.check()?;
        latent_layout_supports_mode(cache.mode)?;
        if cache.is_paged_backed() {
            return Err(
                "mla: absorbed decode cannot use a paged-backed cache; the pool's \
                 [blocks, page_size, Hkv, head_dim] layout has one head dim for both sides and \
                 the latent split is asymmetric"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// View `cache` as a latent cache, or explain why it cannot be.
    pub fn wrap(cache: &'a mut KVCache, geometry: MlaGeometry) -> Result<Self, String> {
        Self::supports(cache, geometry)?;
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
        // `supports` must give the same answer, since the fallback branch in a
        // family reads it instead of `wrap`.
        assert!(MlaLatentCache::supports(&cache, LITE).is_err());
    }

    #[test]
    fn a_latent_cache_stays_trimmable_for_speculative_decode() {
        // Speculative decode gates on `can_trim_prompt_cache` and then rejects
        // the tokens the draft got wrong with `trim`. Both operate on the
        // sequence axis, which is axis 2 in the latent layout exactly as in the
        // decompressed one, so a rejected draft token removes one latent row.
        // If that were not true, a rejection would silently corrupt the cache.
        const G: MlaGeometry = MlaGeometry {
            num_heads: 4,
            kv_lora_rank: 32,
            qk_nope_head_dim: 16,
            qk_rope_head_dim: 8,
            v_head_dim: 12,
        };
        let mut cache = KVCache::new();
        {
            let mut view = MlaLatentCache::wrap(&mut cache, G).unwrap();
            let ckv = crate::ffi::zeros(&[1, 1, 6, 32], crate::dtype::FLOAT16);
            let kpe = crate::ffi::zeros(&[1, 1, 6, 8], crate::dtype::FLOAT16);
            view.update_and_fetch(ckv, kpe);
            assert_eq!(view.seq_len(), 6);
        }
        assert!(crate::cache::can_trim_prompt_cache(std::slice::from_ref(
            &cache
        )));
        assert_eq!(cache.trim(2), 2);
        let view = MlaLatentCache::wrap(&mut cache, G).unwrap();
        assert_eq!(view.seq_len(), 4);
    }

    /// The four families that always cache a latent pair, and the one that
    /// does not. `deepseek_v2` has a decompressed fallback, so a quantized
    /// mode there lands on ordinary per-head K/V rows and must keep working.
    #[test]
    fn only_the_fallback_less_families_are_latent_families() {
        for family in [
            "glm4_moe_lite",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_v3.2",
            "glm_moe_dsa",
            "kimi_linear",
            "longcat_flash_ngram",
        ] {
            assert!(caches_mla_latent_pair(family), "{family}");
        }
        assert!(!caches_mla_latent_pair("deepseek_v2"));
        assert!(!caches_mla_latent_pair("llama"));
        assert!(!caches_mla_latent_pair(""));
    }

    /// The DSA trio needs its own entries, not coverage inherited from
    /// `deepseek_v3`. `caches_mla_latent_pair` matches exactly, so a family
    /// whose `model_type` merely extends an entry is not covered by it, and
    /// `glm_moe_dsa` shares nothing textual with the model it wraps.
    #[test]
    fn the_dsa_families_are_listed_in_their_own_right() {
        for family in ["deepseek_v32", "deepseek_v3.2", "glm_moe_dsa"] {
            assert!(
                MLA_LATENT_CACHE_FAMILIES.contains(&family),
                "{family} must be listed explicitly; `deepseek_v3` does not cover it"
            );
        }
    }

    /// Exact match, not a prefix walk: the latent packing belongs to one
    /// concrete attention implementation, and a neighbouring `model_type` that
    /// happens to share a prefix must not inherit the answer.
    #[test]
    fn latent_family_lookup_is_exact_and_case_insensitive() {
        assert!(caches_mla_latent_pair("  GLM4_MoE_Lite  "));
        assert!(caches_mla_latent_pair("  DeepSeek_V3.2  "));
        assert!(!caches_mla_latent_pair("glm4_moe_lite_vl"));
        assert!(!caches_mla_latent_pair("glm4_moe"));
        assert!(!caches_mla_latent_pair("glm_moe"));
        assert!(!caches_mla_latent_pair("deepseek_v32_vl"));
    }

    /// The rule `MlaLatentCache::supports` has always applied, now callable on
    /// its own so the model-level resolver states it once rather than twice.
    #[test]
    fn latent_layout_accepts_fp16_and_nothing_else() {
        assert!(latent_layout_supports_mode(KVCacheMode::Fp16).is_ok());
        for mode in [
            KVCacheMode::Int8,
            KVCacheMode::Turbo4Asym,
            KVCacheMode::Turbo3Asym,
            KVCacheMode::Turbo4,
            KVCacheMode::Turbo4Delegated,
        ] {
            let err = latent_layout_supports_mode(mode).unwrap_err();
            assert!(err.contains("FP16"), "{err}");
            assert!(err.contains(&mode.to_string()), "{err}");
        }
    }

    #[test]
    fn wrap_accepts_the_default_fp16_cache() {
        let mut cache = KVCache::new();
        let view = MlaLatentCache::wrap(&mut cache, LITE).unwrap();
        assert_eq!(view.seq_len(), 0);
        assert_eq!(view.bytes_per_token(2), 1152);
    }
}
