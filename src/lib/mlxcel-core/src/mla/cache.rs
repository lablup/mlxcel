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
//! ## Quantized latent (issue #3155)
//!
//! Under `KVCacheMode::Int8` the latent is stored as an MLX affine group-64
//! triple and `k_pe` stays FP16; the slot layout is in
//! [`crate::cache::latent_quant`]. This is a different codec from per-head
//! INT8, whose single absmax scale per row is too coarse for a row hundreds of
//! elements wide that every query head reads. [`MlaLatentCache::update_and_fetch`]
//! returns a [`LatentWindow`] so a reader sees which form it was handed.
//!
//! The width is 8 bits unless `--mla-latent-quant int4` is in force
//! ([`crate::mla::latent_option`], issue #3179), which packs the latent at 4
//! in the same `Int8` storage; a cache that already holds rows keeps their
//! width. A DSA key slot then stores its indexer key at 8 bits beside the
//! 4-bit latent (`mla::latent_keys`, "Split slot").
//!
//! ## Modes this declines
//!
//! Every Turbo mode, and paged backing. The Turbo3/Turbo4 modes calibrate a
//! per-token codebook from one head dimension, which fits a per-head K/V row
//! and not a shared latent (see [`latent_layout_supports_mode`]). A Turbo-4
//! codec applied to the rotated latent itself avoids that mismatch, and was
//! measured and not adopted for quality (#3157, `mla::turbo_latent`). The paged
//! pool allocates `[num_blocks, page_size, Hkv, head_dim]` tensors with one
//! head dim for both sides, which the asymmetric `(512, 64)` split cannot
//! fill. Both are declined at wrap time with a message rather than silently
//! mis-served.

use cxx::UniquePtr;

use crate::cache::{KVCache, KVCacheMode};
use crate::ffi::MlxArray;
use crate::mla::MlaGeometry;
use crate::mla::MlaSplitPlan;
use crate::mla::absorb::MlaAbsorbedProjections;
use crate::mla::decode::{absorb_queries, absorbed_attention, unabsorb_output};
use crate::mla::latent_keys::{
    check_stored_key_width, quantize_key_slot_rows, quantized_key_slot_bytes_per_token,
};
use crate::mla::latent_option::latent_cache_bits;
use crate::mla::quant_latent::{LatentWindow, check_quantizable, quantized_absorbed_attention};
use crate::mla::split_kv::absorbed_attention_split_kv;

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
/// default wherever one is declared. On top of that, the DSA families
/// ([`crate::mla::DSA_INDEXER_KEY_FAMILIES`]) concatenate the
/// `index_head_dim`-wide indexer key onto the same slot before caching it,
/// which makes it 640 wide at the default `index_head_dim` of 128. A codebook and a sign vector calibrated from the
/// 64-wide "V" slot are wrong for any of those widths.
///
/// `deepseek_v2` is deliberately **absent**. It goes through
/// [`MlaLatentCache`], which stores the latent quantized under `int8`
/// (issue #3155), and it falls back to the decompressed per-head K/V layout
/// when [`MlaLatentCache::supports`] declines (`src/models/deepseek_v2.rs`).
/// The families in [`crate::mla::QUANTIZED_LATENT_FAMILIES`] write their
/// pair through the view and hold FP16 or the affine `int8` latent (#3160);
/// the rest call `KVCache::update_and_fetch` directly and hold FP16 only
/// until each is moved onto it (#3161-#3163).
///
/// Used by: [`caches_mla_latent_pair`].
pub static MLA_LATENT_CACHE_FAMILIES: &[&str] = &[
    "glm4_moe_lite",
    "deepseek_v3",
    // `kimi_k2` and `joyai_llm_flash` are `DeepseekV3ForCausalLM` checkpoints
    // that detection routes to `DeepSeekV3Model` (#2384).
    "kimi_k2",
    "joyai_llm_flash",
    // `kimi_vl` and `kimi_k25` wrap `DeepSeekV3Model` as the text decoder
    // (`src/vision/kimi_vl.rs`), so every layer runs `DeepSeekV3Attention`,
    // which always caches `(kv_latent, k_pe)` (#3159).
    "kimi_vl",
    "kimi_k25",
    // `youtu` and `youtu_llm` load `YoutuLanguageModel`
    // (`src/loading/nonstandard.rs`) and `youtu_vl` loads the Youtu VLM text
    // decoder; both hold `self_attn: DeepSeekV3Attention` per layer
    // (`src/models/youtu_vl_lm.rs`), so the same latent pair is cached (#3159).
    "youtu",
    "youtu_llm",
    "youtu_vl",
    // `deepseek_v32` and the `deepseek_v3.2` spelling of the same
    // `config.json` value both resolve to `DeepSeekV32Model`
    // (`src/models/detection.rs`), and `glm_moe_dsa` wraps that model
    // wholesale (`src/models/glm_moe_dsa.rs`). All three share the single
    // unconditional `[kv_latent | indexer key]` write in
    // `src/models/deepseek_v32.rs`, with no decompressed fallback.
    "deepseek_v32",
    "deepseek_v3.2",
    "glm_moe_dsa",
    // Kimi Linear and Kimi K3 (which reuses Kimi Linear's absorbed MLA)
    // cache the pair in model-owned MLA layers and write it through the
    // latent view (#3162).
    "kimi_linear",
    "kimi_k3",
    // Both LongCat `model_type`s load `LongcatFlashNgramModel`, whose MLA
    // stores `update_and_fetch(kv_latent, k_pe)` on every step
    // (`src/models/longcat_flash_ngram.rs`); the plain variant only swaps
    // the n-gram embedding for a token embedding.
    "longcat_flash",
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

/// Whether the decoder `model_type` names a family that stores its latent
/// through [`MlaLatentCache`], and can therefore hold an affine-quantized
/// latent.
///
/// `deepseek_v2` only. A wrapper whose decoder is that model (DeepSeek-VL2) is
/// resolved to its decoder type by the caller
/// (`mlxcel::execution::config_fields::kv_decoder_model_type`), not listed
/// here. Exact match, for the reason [`caches_mla_latent_pair`] gives.
///
/// Used by: `mlxcel::cli::turbo_args` (mode resolution, the `--kv-group-size`
/// check and the `generate` banner).
#[must_use]
pub fn stores_latent_through_view(model_type: &str) -> bool {
    model_type.trim().eq_ignore_ascii_case("deepseek_v2")
}

/// The rule: an MLA latent cache holds FP16, or the latent affine-quantized
/// at 8 bits, and no Turbo mode.
///
/// `Int8` stores the latent as an MLX affine group-64 triple and leaves `k_pe`
/// FP16 (issue #3155). Every Turbo [`KVCacheMode`] calibrates a
/// per-token codebook from one head dimension and applies it to both stored
/// streams. A latent cache has two different widths in those two slots
/// (`kv_lora_rank` and `qk_rope_head_dim`), and neither slot is a per-head K/V
/// row:
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
/// This is the rule for a cache read through [`MlaLatentCache`]. A
/// latent-pair family that does not use the view yet holds FP16 only;
/// [`crate::mla::latent_pair_supports_mode`] applies that narrower rule.
///
/// Used by: [`MlaLatentCache::supports`],
/// `cache::turbo::allowlist::resolve_kv_cache_mode_for_model`.
pub fn latent_layout_supports_mode(mode: KVCacheMode) -> Result<(), String> {
    if mode == KVCacheMode::Fp16 || mode.latent_affine_bits().is_some() {
        return Ok(());
    }
    Err(format!(
        "mla: absorbed decode needs an fp16 or int8 latent cache; this cache is {mode} and \
         its per-token quantization is calibrated for per-head K/V rows, not a shared latent"
    ))
}

/// A [`KVCache`] used as a compressed-latent MLA cache.
///
/// Borrowed rather than owned so a family keeps handing `&mut [KVCache]`
/// around and wraps per layer at the point of use.
pub struct MlaLatentCache<'a> {
    inner: &'a mut KVCache,
    geometry: MlaGeometry,
    /// "K" slot width: `kv_lora_rank` plus any DSA sidecar (issue #3163).
    key_width: usize,
    /// Width the latent is packed at, `None` on an fp16 cache: the width of
    /// the rows the cache already holds, else
    /// [`latent_cache_bits`] for its mode (issue #3179).
    latent_bits: Option<i32>,
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
        Self::supports_with_key_width(cache, geometry, geometry.kv_lora_rank)
    }

    /// [`Self::supports`] for a "K" slot `stored_key_width` wide whose first
    /// `kv_lora_rank` elements are the latent: the DSA families cache the
    /// indexer key beside it (`mla::latent_keys`, issue #3163), and an `int8`
    /// cache needs that width to be whole groups too.
    ///
    /// Used by: `mlxcel::models::mla_batched::store_latent_step`.
    pub fn supports_with_key_width(
        cache: &KVCache,
        geometry: MlaGeometry,
        stored_key_width: usize,
    ) -> Result<(), String> {
        geometry.check()?;
        latent_layout_supports_mode(cache.mode)?;
        check_quantizable(&geometry, cache.mode)?;
        check_stored_key_width(geometry.kv_lora_rank, stored_key_width, cache.mode)?;
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
        Self::wrap_with_key_width(cache, geometry, geometry.kv_lora_rank)
    }

    /// [`Self::wrap`] for a "K" slot `stored_key_width` wide
    /// ([`Self::supports_with_key_width`]). [`Self::update_and_fetch`] takes
    /// and returns rows of that width; the `attend` entries read the whole
    /// slot as the latent, so a caller with a sidecar reads the window itself.
    pub fn wrap_with_key_width(
        cache: &'a mut KVCache,
        geometry: MlaGeometry,
        stored_key_width: usize,
    ) -> Result<Self, String> {
        Self::supports_with_key_width(cache, geometry, stored_key_width)?;
        let latent_bits = cache
            .stored_latent_bits()
            .or_else(|| latent_cache_bits(cache.mode));
        Ok(Self {
            inner: cache,
            geometry,
            key_width: stored_key_width,
            latent_bits,
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
    /// `ckv` is `[B, 1, L, kv_lora_rank]` (the wrapped key width, with a
    /// sidecar) and `kpe` is `[B, 1, L, qk_rope_head_dim]`, both already
    /// normalized and rotated respectively by the caller. On an `int8` cache the fresh rows are quantized
    /// before anything reads them, so a prefill and the decode steps after it
    /// see the same stored values.
    pub fn update_and_fetch(
        &mut self,
        ckv: UniquePtr<MlxArray>,
        kpe: UniquePtr<MlxArray>,
    ) -> LatentWindow {
        let Some(bits) = self.latent_bits else {
            let (ckv, kpe) = self.inner.update_and_fetch(ckv, kpe);
            return LatentWindow::Dense { ckv, kpe };
        };
        let rank = i32::try_from(self.geometry.kv_lora_rank).unwrap_or(i32::MAX);
        self.inner
            .update_latent_quantized(quantize_key_slot_rows(&ckv, kpe, rank, bits));
        let rows = self
            .inner
            .latent_quantized_window()
            .expect("the update above stored quantized latent rows");
        LatentWindow::from_rows(rows, bits)
    }

    /// Append one latent step and attend after the query-side folded
    /// projection, returning the result in latent value space.
    ///
    /// This is the ragged-cache half of [`Self::attend`]. It lets a model batch
    /// [`crate::mla::absorb_queries`] across compatible request rows before
    /// entering the per-row cache loop, concatenate the returned latent rows,
    /// and batch [`crate::mla::unabsorb_output`] afterward. Stage 2 split-KV
    /// selection remains identical to [`Self::attend`].
    ///
    /// A quantized window takes [`quantized_absorbed_attention`] for one
    /// query (the fused kernel's quantized variant where it applies, #3156)
    /// and is dequantized once for a multi-token step. The dense fused kernel
    /// and split-KV read an FP16 latent, so the match below keeps both away
    /// from a quantized window before either looks at it.
    ///
    /// `q_absorbed` is `[B, H, 1, kv_lora_rank]`; the returned tensor has the
    /// same shape.
    ///
    /// Used by: DeepSeek V2 native batched decode.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn attend_preprojected(
        &mut self,
        q_absorbed: &MlxArray,
        q_pe: &MlxArray,
        ckv: UniquePtr<MlxArray>,
        kpe: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let q_shape = crate::ffi::array_shape(q_absorbed);
        let single_query = q_shape.len() == 4 && q_shape[2] == 1;
        let (ckv_all, kpe_all) = match self.update_and_fetch(ckv, kpe) {
            LatentWindow::Dense { ckv, kpe } => (ckv, kpe),
            LatentWindow::Quantized {
                ckv_q,
                scales,
                biases,
                kpe,
                bits,
                group_size,
            } if single_query => {
                return quantized_absorbed_attention(
                    q_absorbed, q_pe, &ckv_q, &scales, &biases, &kpe, bits, group_size, scale, mask,
                );
            }
            // More than one query: dequantize once and run the dense folded
            // attention, which never takes the fused kernel. The dequantized
            // window is FP16; it is cast to the query dtype so a bf16
            // checkpoint's step is not promoted to float32 (#3177).
            window @ LatentWindow::Quantized { .. } => {
                let (ckv_all, kpe_all) = window.into_dense();
                let dtype = crate::ffi::array_dtype(q_pe);
                let cast = |a: UniquePtr<MlxArray>| {
                    if crate::ffi::array_dtype(&a) == dtype {
                        a
                    } else {
                        crate::ffi::astype(&a, dtype)
                    }
                };
                let (ckv_all, kpe_all) = (cast(ckv_all), cast(kpe_all));
                return absorbed_attention(q_absorbed, q_pe, &ckv_all, &kpe_all, scale, mask);
            }
        };
        if mask.is_none() && single_query && crate::mla::split_kv_enabled() {
            let plan = MlaSplitPlan::heuristic(
                q_shape[0].max(0) as usize,
                q_shape[1].max(0) as usize,
                self.seq_len(),
                crate::paged_v2::device_target_ctas(),
            );
            if let Ok(out) =
                absorbed_attention_split_kv(q_absorbed, q_pe, &ckv_all, &kpe_all, scale, &plan)
            {
                return out;
            }
        }
        absorbed_attention(q_absorbed, q_pe, &ckv_all, &kpe_all, scale, mask)
    }

    /// Append one step's latent and rope rows and run absorbed attention over
    /// the live window: the MLA analogue of [`KVCache::attend`] (#2171).
    ///
    /// A latent cache keeps its own attention (`absorbed_decode`: the score is
    /// split into the absorbed latent term and the rope term, and the kernel
    /// sees the latent as both K and V), so it cannot share the dense
    /// dispatch. It shares the entry: the caller hands over this step's rows
    /// and the query halves, and never reads the cache back to choose a
    /// kernel.
    ///
    /// `q_nope` is `[B, H, L, qk_nope_head_dim]` (not yet absorbed), `q_pe` is
    /// `[B, H, L, qk_rope_head_dim]` (already rotated), `ckv` and `kpe` are
    /// this step's `[B, 1, L, kv_lora_rank]` and `[B, 1, L, qk_rope_head_dim]`
    /// rows, `mask` is the optional additive causal or padding mask. Returns
    /// `[B, H, L, v_head_dim]`.
    ///
    /// A single-token unmasked step takes the Stage 2 split-KV decode when
    /// [`crate::mla::split_kv_enabled`] is on and its plan accepts the shape,
    /// and Stage 1 [`crate::mla::absorbed_decode`] otherwise, the same order
    /// the model forward ran before this entry existed.
    ///
    /// Used by: `models::deepseek_v2::Attention::forward_absorbed` (decode
    /// steps; prefill up-projects the window through the model's `kv_b_proj`
    /// and stays in the model); `mla::decode_tests` pins it against the
    /// decompressed reference.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn attend(
        &mut self,
        q_nope: &MlxArray,
        q_pe: &MlxArray,
        ckv: UniquePtr<MlxArray>,
        kpe: UniquePtr<MlxArray>,
        proj: &MlaAbsorbedProjections,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let q_absorbed = absorb_queries(q_nope, proj);
        let o_latent = self.attend_preprojected(&q_absorbed, q_pe, ckv, kpe, scale, mask);
        unabsorb_output(&o_latent, proj)
    }

    /// The width this view packs the latent at, `None` on an fp16 cache.
    #[must_use]
    pub const fn latent_bits(&self) -> Option<i32> {
        self.latent_bits
    }

    /// Bytes per token this cache costs: the affine-quantized rate on an
    /// `int8` cache (at the width it packs the latent, with any sidecar at 8
    /// bits), else `bytes_per_element` per latent and rope element.
    #[must_use]
    pub const fn bytes_per_token(&self, bytes_per_element: usize) -> usize {
        match self.latent_bits {
            Some(bits) => quantized_key_slot_bytes_per_token(
                self.geometry.kv_lora_rank,
                self.key_width.saturating_sub(self.geometry.kv_lora_rank),
                self.geometry.qk_rope_head_dim,
                bits as usize,
                crate::cache::latent_quant::LATENT_QUANT_GROUP_SIZE as usize,
            ),
            None => (self.key_width + self.geometry.qk_rope_head_dim) * bytes_per_element,
        }
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
#[path = "cache_tests.rs"]
mod tests;
