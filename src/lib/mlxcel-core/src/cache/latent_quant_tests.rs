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

//! Mode-level contract of the affine latent mode (issue #3155). The storage
//! and decode behaviour is pinned through the latent view in
//! `mla/quant_latent_tests.rs`.

use super::*;

#[test]
fn only_int8_is_an_affine_latent_mode() {
    assert_eq!(KVCacheMode::Int8.latent_affine_bits(), Some(8));
    assert!(KVCacheMode::Int8.uses_scale_sidecars());
    for mode in [
        KVCacheMode::Fp16,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo4Delegated,
        KVCacheMode::Turbo3Asym,
    ] {
        assert_eq!(mode.latent_affine_bits(), None, "{mode}");
        assert!(!mode.uses_scale_sidecars(), "{mode}");
    }
}

/// 4-bit failed the #3155 quality gate, so no mode selects it and the
/// spelling stays unknown rather than parsing to something that would be
/// substituted later.
#[test]
fn int4_is_not_a_kv_cache_mode() {
    for spelling in ["int4", "i4"] {
        let err = spelling.parse::<KVCacheMode>().unwrap_err();
        assert!(err.contains("unknown kv-cache-mode"), "{err}");
    }
}

#[test]
fn an_empty_or_per_head_int8_cache_holds_no_quantized_latent() {
    let cache = KVCache::new_with_mode(KVCacheMode::Int8);
    assert!(!cache.holds_quantized_latent());
    assert!(cache.latent_quantized_window().is_none());

    // Per-head rows on an INT8 cache keep the per-token absmax codec: the
    // int8 `keys` buffer is what tells the two layouts apart.
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let k = ffi::full_f32(&[1, 2, 3, 8], 0.25, dtype::FLOAT16);
    let v = ffi::full_f32(&[1, 2, 3, 8], -0.5, dtype::FLOAT16);
    let (k_out, _) = cache.update_and_fetch(k, v);
    assert_eq!(ffi::array_shape(&k_out), vec![1, 2, 3, 8]);
    assert!(!cache.holds_quantized_latent());
    assert!(cache.latent_quantized_window().is_none());
}
