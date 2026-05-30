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

//! DFlash drafter K/V cache type alias.
//!
//! Upstream `mlx-vlm/speculative/drafters/qwen3_dflash/dflash.py` declares
//! `DFlashKVCache = KVCache` at the module bottom. The DFlash drafter
//! reuses MLX-LM's stock `KVCache` shape and `update_and_fetch` semantics
//! verbatim — no new cache layout is required.
//!
//! Two things make this alias important enough to keep as its own module:
//!
//! 1. It matches the upstream symbol name so other parts of the port that
//!    follow the upstream wiring (`from .dflash import DFlashKVCache`)
//!    keep using a recognisable name when ported.
//! 2. The drafter's `update_and_fetch` writes ONLY context K/V into the
//!    cache; proposal K/V is concatenated post-hoc in the attention
//!    forward. Encoding this constraint in the alias' documentation is
//!    load-bearing acceptance.

use crate::layers::KVCache;

/// Drafter-side K/V cache for the Qwen 3.5 DFlash drafter.
///
/// This is a transparent alias for [`KVCache`]. The DFlash drafter calls
/// `update_and_fetch(ctx_keys, ctx_values)` with **only** the context-side
/// projections per round; the proposal-side K/V is concatenated onto the
/// fetched tensors post-hoc and never enters this cache.
///
/// One cache instance is created per drafter layer (5 layers on the
/// published `z-lab/Qwen3.5-4B-DFlash` checkpoint).
pub type DFlashKVCache = KVCache;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_constructs_default_cache_with_zero_offset() {
        // Verify the alias produces a bog-standard, empty KVCache with the
        // FP16 default mode (DFlash never selects a Turbo mode at construction
        // time — quantization is a separate decision).
        let cache: DFlashKVCache = KVCache::new();
        assert_eq!(cache.offset, 0, "fresh cache must start at offset 0");
        assert!(
            cache.keys.is_none(),
            "fresh cache must have no key buffer yet"
        );
        assert!(
            cache.values.is_none(),
            "fresh cache must have no value buffer yet"
        );
    }

    #[test]
    fn alias_is_kvcache_type_at_compile_time() {
        // Compile-time assertion: the alias is *exactly* KVCache, not a
        // newtype wrapper. This is load-bearing because callers in
        // `mlxcel-core::drafter::dflash::model::DFlashDraftModel::make_cache`
        // return `Vec<KVCache>` to satisfy the `Drafter::make_cache` trait
        // contract — they cannot do that if `DFlashKVCache` is a newtype.
        fn assert_same_type<A: 'static, B: 'static>() -> bool {
            std::any::TypeId::of::<A>() == std::any::TypeId::of::<B>()
        }
        assert!(
            assert_same_type::<DFlashKVCache, KVCache>(),
            "DFlashKVCache must be a type alias for KVCache"
        );
    }
}
