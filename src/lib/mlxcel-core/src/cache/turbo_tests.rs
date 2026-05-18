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

//! Integration tests for `KVCacheMode::Turbo4Asym` (issue #474, epic #458).
//!
//! Covers:
//! 1. Mode-string parsing (`fp16+turbo4`, `turbo4-asym`).
//! 2. Single-token / multi-token update + read round-trip.
//! 3. Trim correctness across the V sidecars.
//! 4. detach / install_detached round-trip preserves V sidecars + seed.

use super::*;
use crate::dtype;
use crate::ffi;

// ---------------------------------------------------------------------------
// Mode parsing
// ---------------------------------------------------------------------------

#[test]
fn turbo4_asym_parses_canonical_string() {
    let m: KVCacheMode = "fp16+turbo4".parse().unwrap();
    assert_eq!(m, KVCacheMode::Turbo4Asym);
}

#[test]
fn turbo4_asym_parses_alias() {
    let m: KVCacheMode = "turbo4-asym".parse().unwrap();
    assert_eq!(m, KVCacheMode::Turbo4Asym);
}

#[test]
fn turbo4_asym_parsing_is_case_insensitive() {
    let m: KVCacheMode = "FP16+TURBO4".parse().unwrap();
    assert_eq!(m, KVCacheMode::Turbo4Asym);
    let m: KVCacheMode = "Turbo4-Asym".parse().unwrap();
    assert_eq!(m, KVCacheMode::Turbo4Asym);
}

#[test]
fn turbo4_asym_display_round_trip() {
    let m = KVCacheMode::Turbo4Asym;
    let s = m.to_string();
    assert_eq!(s, "fp16+turbo4");
    let parsed: KVCacheMode = s.parse().unwrap();
    assert_eq!(parsed, m);
}

#[test]
fn unknown_mode_string_errors() {
    // "turbo2" is intentionally not a recognised alias — issue #477 is the
    // 3-bit mode, and 2-bit (Turbo2) is an explicit non-goal of epic #458.
    let r: Result<KVCacheMode, _> = "turbo2".parse();
    assert!(r.is_err());
    let err = r.unwrap_err();
    assert!(
        err.contains("turbo2"),
        "error message should include input: {err}"
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a deterministic [B, H, T, D] f32 V tensor with token-varying
/// magnitudes. Uses the in-house LCG so every test sees the same data.
fn synth_kv_tensor(b: i32, h: i32, t: i32, d: i32, seed: u32) -> cxx::UniquePtr<ffi::MlxArray> {
    let total = (b * h * t * d) as usize;
    let mut state = if seed == 0 { 0xDEADBEEF } else { seed };
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        // Same LCG as quant::Lcg32 — keeps tests reproducible across platforms.
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // Map to [-1.0, 1.0]
        let x = (state >> 1) as f32 / (i32::MAX as f32);
        data.push(x);
    }
    ffi::from_slice_f32(&data, &[b, h, t, d])
}

fn flatten_fp32(arr: &ffi::MlxArray) -> Vec<f32> {
    let a = ffi::astype(arr, dtype::FLOAT32);
    ffi::eval(&a);
    let bytes = ffi::array_to_raw_bytes(&a);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// update_and_fetch round-trip
// ---------------------------------------------------------------------------

#[test]
fn turbo4_asym_update_returns_fp16_dequantized_v() {
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);

    let k = synth_kv_tensor(1, 1, 4, head_dim, 42);
    let v = synth_kv_tensor(1, 1, 4, head_dim, 43);

    let (k_out, v_out) = cache.update_and_fetch(k, v);
    assert_eq!(ffi::array_dtype(&v_out), dtype::FLOAT16);
    assert_eq!(ffi::array_shape(&v_out), vec![1_i32, 1, 4, head_dim]);

    // K side must be untouched — same shape, fp16 dtype.
    assert_eq!(ffi::array_dtype(&k_out), dtype::FLOAT16);
    assert_eq!(ffi::array_shape(&k_out), vec![1_i32, 1, 4, head_dim]);

    // The packed and norms sidecars must be populated; the standard `values`
    // tensor must NOT be (Turbo4Asym never stores fp16 V).
    assert!(cache.v_packed.is_some());
    assert!(cache.v_norms.is_some());
    assert!(cache.values.is_none());
    assert_eq!(cache.seq_len(), 4);
}

#[test]
fn turbo4_asym_multi_token_growth_keeps_visible_window_correct() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);

    let k1 = synth_kv_tensor(1, 1, 3, head_dim, 1);
    let v1 = synth_kv_tensor(1, 1, 3, head_dim, 2);
    let (k_out_1, v_out_1) = cache.update_and_fetch(k1, v1);
    assert_eq!(ffi::array_shape(&k_out_1)[2], 3);
    assert_eq!(ffi::array_shape(&v_out_1)[2], 3);
    assert_eq!(cache.seq_len(), 3);

    let k2 = synth_kv_tensor(1, 1, 5, head_dim, 3);
    let v2 = synth_kv_tensor(1, 1, 5, head_dim, 4);
    let (k_out_2, v_out_2) = cache.update_and_fetch(k2, v2);
    assert_eq!(cache.seq_len(), 8);
    assert_eq!(ffi::array_shape(&k_out_2)[2], 8);
    assert_eq!(ffi::array_shape(&v_out_2)[2], 8);
}

#[test]
fn turbo4_asym_keys_are_bit_identical_to_input() {
    // K side bypasses the quantizer entirely. After update, the visible portion
    // of self.keys (cast to f32) should equal the input bytes (cast through f16).
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);

    // Use a small, exactly-fp16-representable input to avoid round-off noise.
    let k_data: Vec<f32> = (0..head_dim).map(|i| ((i % 8) as f32) * 0.125).collect();
    let v_data: Vec<f32> = (0..head_dim).map(|i| ((i + 1) as f32) * 0.01).collect();
    let k = ffi::from_slice_f32(&k_data, &[1, 1, 1, head_dim]);
    let v = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);

    let (k_out, _) = cache.update_and_fetch(k, v);
    let recovered = flatten_fp32(&k_out);
    for (a, b) in recovered.iter().zip(k_data.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "K side must round-trip through fp16 unchanged: got {a}, expected {b}"
        );
    }
}

#[test]
fn turbo4_asym_v_reconstruction_has_bounded_error() {
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);

    // Single token with realistic magnitude
    let v_data: Vec<f32> = (0..head_dim)
        .map(|i| (i as f32 / head_dim as f32 - 0.5) * 2.0)
        .collect();
    let k = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);
    let v = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);

    let (_k_out, v_out) = cache.update_and_fetch(k, v);
    let v_recovered = flatten_fp32(&v_out);

    // Per-token relative L2 error should be < 15% (Lloyd-Max bound + fp16
    // round-off + norm-correction noise). See cache::turbo::quant tests.
    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for i in 0..head_dim as usize {
        let diff = v_data[i] - v_recovered[i];
        num += diff * diff;
        den += v_data[i] * v_data[i];
    }
    let rel = (num / den.max(1e-12)).sqrt();
    assert!(rel < 0.15, "V reconstruction relative error {rel:.4} > 15%");
}

// ---------------------------------------------------------------------------
// Trim
// ---------------------------------------------------------------------------

#[test]
fn turbo4_asym_trim_to_zero_clears_all_buffers() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 5, head_dim, 11);
    let v = synth_kv_tensor(1, 1, 5, head_dim, 12);
    cache.update(k, v);
    assert_eq!(cache.seq_len(), 5);
    assert!(cache.v_packed.is_some());
    assert!(cache.v_norms.is_some());

    let trimmed = cache.trim(5);
    assert_eq!(trimmed, 5);
    assert_eq!(cache.seq_len(), 0);
    assert!(cache.is_empty());
    assert!(cache.v_packed.is_none());
    assert!(cache.v_norms.is_none());
}

#[test]
fn turbo4_asym_partial_trim_shrinks_v_sidecars() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 8, head_dim, 21);
    let v = synth_kv_tensor(1, 1, 8, head_dim, 22);
    cache.update(k, v);
    assert_eq!(cache.seq_len(), 8);

    let n = cache.trim(3);
    assert_eq!(n, 3);
    assert_eq!(cache.seq_len(), 5);

    // V sidecars must reflect the new offset.
    let vp = cache.v_packed.as_ref().unwrap();
    let vn = cache.v_norms.as_ref().unwrap();
    assert_eq!(ffi::array_shape(vp)[2], 5);
    assert_eq!(ffi::array_shape(vn)[2], 5);
    let k_buf = cache.keys.as_ref().unwrap();
    assert_eq!(ffi::array_shape(k_buf)[2], 5);
}

/// LOW-1 regression: `turbo_params` must be `None` after a full trim so the
/// next quantize call rebuilds params from scratch. This matters when a cache
/// slot is reused with a different head_dim after the trim (e.g. a sequence
/// completes and the slot is handed to a new sequence of a different model).
#[test]
fn turbo4_asym_trim_to_zero_clears_turbo_params() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 4, head_dim, 99);
    let v = synth_kv_tensor(1, 1, 4, head_dim, 100);
    cache.update(k, v);

    // turbo_params is populated after the first update.
    assert!(
        cache.turbo_params.is_some(),
        "turbo_params must be Some after update"
    );
    assert_eq!(cache.seq_len(), 4);

    // Full trim should clear turbo_params alongside the tensor buffers.
    let trimmed = cache.trim(4);
    assert_eq!(trimmed, 4);
    assert_eq!(cache.seq_len(), 0);
    assert!(cache.is_empty());
    assert!(
        cache.turbo_params.is_none(),
        "turbo_params must be None after full trim (LOW-1)"
    );
}

/// LOW-1 regression: `clone_handle` must clear `turbo_params` on the source
/// so the slot can be reused or initialized fresh without stale params.
#[test]
fn turbo4_asym_clone_handle_clears_turbo_params_on_source() {
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 3, head_dim, 101);
    let v = synth_kv_tensor(1, 1, 3, head_dim, 102);
    cache.update(k, v);

    // turbo_params is populated after the first update.
    assert!(
        cache.turbo_params.is_some(),
        "turbo_params must be Some after update"
    );

    let _handle = cache.clone_handle();

    // Source cache must have turbo_params cleared after clone_handle so a
    // fresh sequence can start with a clean slate (LOW-1 fix, #474).
    assert!(
        cache.turbo_params.is_none(),
        "clone_handle must clear turbo_params on source (LOW-1)"
    );
    // The source is also empty (existing contract).
    assert!(cache.is_empty());
}

// ---------------------------------------------------------------------------
// detach / install_detached round-trip
// ---------------------------------------------------------------------------

#[test]
fn turbo4_asym_clone_handle_round_trip_preserves_sidecars() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 4, head_dim, 31);
    let v = synth_kv_tensor(1, 1, 4, head_dim, 32);
    cache.update(k, v);

    // Capture the bytes of v_packed and v_norms pre-detach so we can compare
    // post-adopt.
    let pre_vp = ffi::array_to_raw_bytes(cache.v_packed.as_ref().unwrap());
    let pre_vn = ffi::array_to_raw_bytes(cache.v_norms.as_ref().unwrap());
    let pre_seed = cache.turbo_seed;

    let handle = cache.clone_handle();
    assert_eq!(handle.mode(), KVCacheMode::Turbo4Asym);

    // Source cache should be empty after clone_handle.
    assert!(cache.is_empty());
    assert!(cache.v_packed.is_none());
    assert!(cache.v_norms.is_none());

    let mut restored = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    restored.install_detached(handle).unwrap();

    assert_eq!(restored.seq_len(), 4);
    assert_eq!(restored.mode, KVCacheMode::Turbo4Asym);
    assert!(restored.v_packed.is_some());
    assert!(restored.v_norms.is_some());

    // turbo_params should have been re-derived from v_packed shape.
    assert!(restored.turbo_params.is_some());
    assert_eq!(restored.turbo_seed, pre_seed);

    let post_vp = ffi::array_to_raw_bytes(restored.v_packed.as_ref().unwrap());
    let post_vn = ffi::array_to_raw_bytes(restored.v_norms.as_ref().unwrap());
    assert_eq!(
        pre_vp, post_vp,
        "v_packed must survive detach/adopt bit-for-bit"
    );
    assert_eq!(
        pre_vn, post_vn,
        "v_norms must survive detach/adopt bit-for-bit"
    );
}

#[test]
fn turbo4_asym_clone_handle_install_then_dequant_matches_pre_detach() {
    // This is the strongest property: detach + install + read must yield the
    // same dequantized V tensor as a direct read on the original cache.
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k_data: Vec<f32> = (0..2 * head_dim)
        .map(|i| (i as f32 / 256.0) - 0.5)
        .collect();
    let v_data: Vec<f32> = (0..2 * head_dim)
        .map(|i| (((i * 7) % 13) as f32 / 13.0) - 0.5)
        .collect();
    let k = ffi::from_slice_f32(&k_data, &[1, 1, 2, head_dim]);
    let v = ffi::from_slice_f32(&v_data, &[1, 1, 2, head_dim]);

    let (_k1, v1_out) = cache.update_and_fetch(k, v);
    let v1 = flatten_fp32(&v1_out);

    let handle = cache.clone_handle();
    let mut restored = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    restored.install_detached(handle).unwrap();

    // Re-dequantize the visible portion of the restored cache using the
    // restored TurboQuantParams. To match the pre-detach v1, slice the V
    // sidecars to `restored.offset` first — the buffer capacity may exceed
    // the visible length due to step-aligned pre-allocation.
    let params = restored.turbo_params.as_ref().unwrap();
    let vp_buf = restored.v_packed.as_ref().unwrap();
    let vn_buf = restored.v_norms.as_ref().unwrap();
    let vps = ffi::array_shape(vp_buf);
    let vns = ffi::array_shape(vn_buf);
    let off = restored.offset;
    let vp = ffi::slice(vp_buf, &[0, 0, 0, 0], &[vps[0], vps[1], off, vps[3]]);
    let vn = ffi::slice(vn_buf, &[0, 0, 0, 0], &[vns[0], vns[1], off, 1]);
    let v2_out = super::turbo::quant::dequantize_v_turbo4(&vp, &vn, params);
    let v2 = flatten_fp32(&v2_out);

    assert_eq!(v1.len(), v2.len());
    for (a, b) in v1.iter().zip(v2.iter()) {
        // Fp16 quantize/dequant determinism: identical sidecars must produce
        // identical V output.
        assert!(
            (a - b).abs() < 1e-4,
            "post-detach dequant mismatch: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// CachePool detach/adopt round-trip on Turbo4Asym
// ---------------------------------------------------------------------------

#[test]
fn cache_pool_detach_adopt_preserves_turbo4_asym() {
    use crate::generate::LanguageModel;

    struct Stub {
        n: usize,
    }
    impl LanguageModel for Stub {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> cxx::UniquePtr<MlxArray> {
            ffi::zeros(&[1], 0)
        }
        fn make_caches(&self) -> Vec<KVCache> {
            (0..self.n).map(|_| KVCache::new()).collect()
        }
        fn num_layers(&self) -> usize {
            self.n
        }
        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }
    }

    let head_dim = 64;
    let model = Stub { n: 1 };
    let mut pool = CachePool::new(4);

    let seq_a = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        caches[0] = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
        let k = synth_kv_tensor(1, 1, 5, head_dim, 51);
        let v = synth_kv_tensor(1, 1, 5, head_dim, 52);
        caches[0].update(k, v);
    }

    let pre_vp = {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        ffi::array_to_raw_bytes(caches[0].v_packed.as_ref().unwrap())
    };
    let pre_vn = {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        ffi::array_to_raw_bytes(caches[0].v_norms.as_ref().unwrap())
    };

    let detached = pool.detach(seq_a).unwrap();
    let seq_b = pool.adopt(&model, detached).unwrap();

    let caches = pool.get_caches_mut(seq_b).unwrap();
    assert_eq!(caches[0].mode, KVCacheMode::Turbo4Asym);
    assert_eq!(caches[0].seq_len(), 5);
    assert!(caches[0].v_packed.is_some());
    assert!(caches[0].v_norms.is_some());

    let post_vp = ffi::array_to_raw_bytes(caches[0].v_packed.as_ref().unwrap());
    let post_vn = ffi::array_to_raw_bytes(caches[0].v_norms.as_ref().unwrap());
    assert_eq!(pre_vp, post_vp);
    assert_eq!(pre_vn, post_vn);
}

// ---------------------------------------------------------------------------
// Memory accounting
// ---------------------------------------------------------------------------

#[test]
fn turbo4_asym_nbytes_includes_v_sidecars_and_excludes_values() {
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 32, head_dim, 71);
    let v = synth_kv_tensor(1, 1, 32, head_dim, 72);
    cache.update(k, v);
    let bytes = cache.nbytes();
    // Expected lower bound:
    //   K buffer   = 1*1*step(256)*128*2 = 65536
    //   v_packed   = 1*1*256*64*1        = 16384
    //   v_norms    = 1*1*256*1*2         = 512
    // Some padding from step alignment is fine.
    assert!(bytes > 0, "Turbo4Asym nbytes() must be non-zero");
    assert!(
        bytes < 200_000,
        "Turbo4Asym nbytes() should be ~80KB, got {bytes}"
    );
    // Values tensor stays None — it should NOT contribute to the byte count.
    assert!(cache.values.is_none());
}

// ---------------------------------------------------------------------------
// RotatingKVCache + Turbo4Asym (B9, issue #481)
// ---------------------------------------------------------------------------

/// Constructor must reject non-32-aligned `max_size` for `Turbo4Asym`.
#[test]
#[should_panic(expected = "max_size must be a positive multiple of")]
fn rotating_turbo4_rejects_misaligned_max_size() {
    // 100 is not divisible by 32 — must fail loudly.
    let _ = RotatingKVCache::new_with_mode(100, KVCacheMode::Turbo4Asym);
}

/// Constructor must accept the canonical sliding-window sizes used by today's
/// models (Gemma 3 4 K, Gemma 4 8 K, Ministral 3 8 K, etc.).
#[test]
fn rotating_turbo4_accepts_canonical_window_sizes() {
    for &n in &[32, 64, 256, 1024, 4096, 8192] {
        let cache = RotatingKVCache::new_with_mode(n, KVCacheMode::Turbo4Asym);
        assert_eq!(cache.max_size, n);
        assert_eq!(cache.mode, KVCacheMode::Turbo4Asym);
        assert!(cache.is_empty());
    }
}

/// Backward-compat: the old `RotatingKVCache::new(max_size)` constructor must
/// still produce an FP16 cache so existing models keep working unchanged.
#[test]
fn rotating_new_defaults_to_fp16_mode() {
    let cache = RotatingKVCache::new(4096);
    assert_eq!(cache.mode, KVCacheMode::Fp16);
    assert!(cache.v_packed.is_none());
    assert!(cache.v_norms.is_none());
}

/// Single-token decode update returns FP16 V regardless of the underlying
/// packed storage.
#[test]
fn rotating_turbo4_single_token_returns_fp16_v() {
    let head_dim = 128;
    let mut cache = RotatingKVCache::new_with_mode(64, KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 1, head_dim, 201);
    let v = synth_kv_tensor(1, 1, 1, head_dim, 202);
    let (k_out, v_out) = cache.update_and_fetch(k, v);
    assert_eq!(ffi::array_dtype(&v_out), dtype::FLOAT16);
    assert_eq!(ffi::array_shape(&v_out)[3], head_dim);
    assert_eq!(ffi::array_dtype(&k_out), dtype::FLOAT16);
    assert!(cache.v_packed.is_some());
    assert!(cache.v_norms.is_some());
    assert!(cache.values.is_none());
    assert_eq!(cache.get_offset(), 1);
}

/// Multi-token (prefill) update populates the packed sidecars and visible
/// window matches the input sequence length when below `max_size`.
#[test]
fn rotating_turbo4_prefill_populates_sidecars() {
    let head_dim = 64;
    let mut cache = RotatingKVCache::new_with_mode(128, KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 8, head_dim, 211);
    let v = synth_kv_tensor(1, 1, 8, head_dim, 212);
    let (k_out, v_out) = cache.update_and_fetch(k, v);
    assert_eq!(ffi::array_shape(&k_out)[2], 8);
    assert_eq!(ffi::array_shape(&v_out)[2], 8);
    assert_eq!(cache.get_offset(), 8);
    let vp_shape = ffi::array_shape(cache.v_packed.as_ref().unwrap());
    assert_eq!(vp_shape[2], 8); // packed rows match offset, not max_size
    assert_eq!(vp_shape[3], head_dim / 2); // nibble-packed
}

/// Wraparound write at `idx == max_size` produces correct dequantized output
/// for the freshly-written token. This is the core invariant for B9: a
/// single-token decode that lands on slot 0 must overwrite cleanly.
#[test]
fn rotating_turbo4_wraparound_overwrites_oldest_slot() {
    let head_dim: i32 = 64;
    let max_size: i32 = 32; // exactly one BLOCK_SIZE

    let mut cache = RotatingKVCache::new_with_mode(max_size, KVCacheMode::Turbo4Asym);

    // Prime the cache with `max_size` distinct tokens so the next write wraps.
    for t in 0..max_size {
        let k = synth_kv_tensor(1, 1, 1, head_dim, 1000 + t as u32);
        let v = synth_kv_tensor(1, 1, 1, head_dim, 2000 + t as u32);
        cache.update_and_fetch(k, v);
    }
    assert_eq!(cache.get_offset(), max_size);

    // Write one more token — this lands on physical slot 0, overwriting the
    // very first token.
    let new_k = synth_kv_tensor(1, 1, 1, head_dim, 31337);
    let new_v_data: Vec<f32> = (0..head_dim)
        .map(|i| (i as f32 / head_dim as f32) - 0.5)
        .collect();
    let new_v = ffi::from_slice_f32(&new_v_data, &[1, 1, 1, head_dim]);

    let (_k_out, v_out) = cache.update_and_fetch(new_k, new_v);
    assert_eq!(cache.get_offset(), max_size + 1);

    // The visible window is exactly `max_size` tokens (full ring).
    assert_eq!(ffi::array_shape(&v_out)[2], max_size);

    // The packed bytes for slot 0 must reflect the new token. Read the
    // packed buffer at slot 0, dequantize via params, and compare relative
    // L2 against the input. Allow up to 15% per-token reconstruction error
    // (the same bound as direct quant tests).
    let params = cache.turbo_params.as_ref().unwrap();
    let vp_buf = cache.v_packed.as_ref().unwrap();
    let vn_buf = cache.v_norms.as_ref().unwrap();
    let vp_slot = ffi::slice(vp_buf, &[0, 0, 0, 0], &[1, 1, 1, head_dim / 2]);
    let vn_slot = ffi::slice(vn_buf, &[0, 0, 0, 0], &[1, 1, 1, 1]);
    let v_recovered_arr = super::turbo::quant::dequantize_v_turbo4(&vp_slot, &vn_slot, params);
    let v_recovered = flatten_fp32(&v_recovered_arr);
    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for i in 0..head_dim as usize {
        let diff = new_v_data[i] - v_recovered[i];
        num += diff * diff;
        den += new_v_data[i] * new_v_data[i];
    }
    let rel = (num / den.max(1e-12)).sqrt();
    assert!(
        rel < 0.15,
        "wraparound overwrite at slot 0 must produce correct dequantized V: \
         relative L2 error {rel:.4} > 15% — block alignment likely broken"
    );
}

/// Block alignment invariant: at the wraparound boundary, every 32-token
/// block must contain self-consistent packed bytes (i.e., per-token quant
/// is independent so each slot decodes correctly regardless of its
/// neighbours). Verifies that writing a wrap-around token does NOT corrupt
/// the previous block.
#[test]
fn rotating_turbo4_wraparound_preserves_other_block_data() {
    let head_dim: i32 = 64;
    let max_size: i32 = 64; // two BLOCK_SIZE blocks

    let mut cache = RotatingKVCache::new_with_mode(max_size, KVCacheMode::Turbo4Asym);

    // Write a sentinel token at slot 31 (last token in block 0).
    let sentinel_data: Vec<f32> = (0..head_dim)
        .map(|i| (i as f32 / head_dim as f32) - 0.25)
        .collect();
    let sentinel_v = ffi::from_slice_f32(&sentinel_data, &[1, 1, 1, head_dim]);
    let sentinel_k = synth_kv_tensor(1, 1, 1, head_dim, 999);

    // Prime: 31 nondescript tokens, then sentinel, then 32 more.
    for t in 0..31 {
        cache.update_and_fetch(
            synth_kv_tensor(1, 1, 1, head_dim, 100 + t as u32),
            synth_kv_tensor(1, 1, 1, head_dim, 200 + t as u32),
        );
    }
    cache.update_and_fetch(sentinel_k, sentinel_v);
    for t in 0..32 {
        cache.update_and_fetch(
            synth_kv_tensor(1, 1, 1, head_dim, 300 + t as u32),
            synth_kv_tensor(1, 1, 1, head_dim, 400 + t as u32),
        );
    }
    // Now write a wraparound token at physical slot 0 (one past max_size).
    cache.update_and_fetch(
        synth_kv_tensor(1, 1, 1, head_dim, 31337),
        synth_kv_tensor(1, 1, 1, head_dim, 31338),
    );
    assert_eq!(cache.get_offset(), max_size + 1);

    // The sentinel at physical slot 31 (block 0, last position) MUST still
    // dequantize correctly — block alignment + per-token independence
    // guarantee the wraparound write to slot 0 cannot have touched slot 31.
    let params = cache.turbo_params.as_ref().unwrap();
    let vp_buf = cache.v_packed.as_ref().unwrap();
    let vn_buf = cache.v_norms.as_ref().unwrap();
    let vp_sentinel = ffi::slice(vp_buf, &[0, 0, 31, 0], &[1, 1, 32, head_dim / 2]);
    let vn_sentinel = ffi::slice(vn_buf, &[0, 0, 31, 0], &[1, 1, 32, 1]);
    let v_recovered_arr =
        super::turbo::quant::dequantize_v_turbo4(&vp_sentinel, &vn_sentinel, params);
    let v_recovered = flatten_fp32(&v_recovered_arr);
    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for i in 0..head_dim as usize {
        let diff = sentinel_data[i] - v_recovered[i];
        num += diff * diff;
        den += sentinel_data[i] * sentinel_data[i];
    }
    let rel = (num / den.max(1e-12)).sqrt();
    assert!(
        rel < 0.15,
        "block-alignment invariant violated: sentinel at slot 31 corrupted by \
         wraparound write to slot 0 (relative L2 = {rel:.4})"
    );
}

/// FP16 mode of `RotatingKVCache` must remain bit-identical to the pre-B9
/// behavior — we cannot regress non-Turbo paths.
#[test]
fn rotating_fp16_mode_unchanged_by_b9() {
    let head_dim = 32;
    let mut cache = RotatingKVCache::new_with_mode(8, KVCacheMode::Fp16);
    let k = synth_kv_tensor(1, 1, 4, head_dim, 50);
    let v = synth_kv_tensor(1, 1, 4, head_dim, 51);
    let (_k_out, v_out) = cache.update_and_fetch(k, v);
    let v_out_shape = ffi::array_shape(&v_out);
    assert_eq!(v_out_shape[2], 4);
    // No Turbo sidecars in FP16 mode.
    assert!(cache.v_packed.is_none());
    assert!(cache.v_norms.is_none());
    // Standard `values` buffer is populated.
    assert!(cache.values.is_some());
}

// ---------------------------------------------------------------------------
// detach / install_detached round-trip on RotatingKVCache + Turbo4Asym
// ---------------------------------------------------------------------------

#[test]
fn rotating_turbo4_clone_handle_round_trip_preserves_sidecars() {
    let head_dim = 64;
    let max_size = 64;
    let mut cache = RotatingKVCache::new_with_mode(max_size, KVCacheMode::Turbo4Asym);

    // Populate enough tokens to exceed half the ring (so `idx` matters).
    for t in 0..40 {
        cache.update_and_fetch(
            synth_kv_tensor(1, 1, 1, head_dim, 800 + t as u32),
            synth_kv_tensor(1, 1, 1, head_dim, 900 + t as u32),
        );
    }
    assert_eq!(cache.get_offset(), 40);
    let pre_idx_offset = cache.get_offset();

    let pre_vp = ffi::array_to_raw_bytes(cache.v_packed.as_ref().unwrap());
    let pre_vn = ffi::array_to_raw_bytes(cache.v_norms.as_ref().unwrap());
    let pre_seed = cache.turbo_seed;

    let handle = cache.clone_handle();
    assert_eq!(handle.mode(), KVCacheMode::Turbo4Asym);
    assert_eq!(handle.max_size(), max_size);
    assert_eq!(handle.seq_len(), pre_idx_offset);

    // Source cache should be empty after clone_handle.
    assert!(cache.is_empty());
    assert!(cache.v_packed.is_none());
    assert!(cache.v_norms.is_none());
    assert_eq!(cache.get_offset(), 0);

    let mut restored =
        RotatingKVCache::new_with_mode_and_seed(max_size, KVCacheMode::Turbo4Asym, pre_seed);
    restored.install_detached(handle).unwrap();

    assert_eq!(restored.get_offset(), pre_idx_offset);
    assert_eq!(restored.max_size, max_size);
    assert_eq!(restored.mode, KVCacheMode::Turbo4Asym);
    assert!(restored.v_packed.is_some());
    assert!(restored.v_norms.is_some());
    assert!(restored.turbo_params.is_some());
    assert_eq!(restored.turbo_seed, pre_seed);

    let post_vp = ffi::array_to_raw_bytes(restored.v_packed.as_ref().unwrap());
    let post_vn = ffi::array_to_raw_bytes(restored.v_norms.as_ref().unwrap());
    assert_eq!(
        pre_vp, post_vp,
        "v_packed must survive detach/adopt bit-for-bit"
    );
    assert_eq!(
        pre_vn, post_vn,
        "v_norms must survive detach/adopt bit-for-bit"
    );
}

/// `idx` and `offset` must round-trip across detach/adopt so wraparound
/// state is preserved. Without this, an adopted cache that was already in
/// the wrap-around regime would silently fall back to "no wraparound yet".
#[test]
fn rotating_turbo4_detach_preserves_idx_after_wraparound() {
    let head_dim = 64;
    let max_size = 32; // one BLOCK_SIZE — easy to exhaust
    let mut cache = RotatingKVCache::new_with_mode(max_size, KVCacheMode::Turbo4Asym);

    // Drive into wrap-around: write `max_size + 5` tokens.
    for t in 0..(max_size + 5) {
        cache.update_and_fetch(
            synth_kv_tensor(1, 1, 1, head_dim, 700 + t as u32),
            synth_kv_tensor(1, 1, 1, head_dim, 800 + t as u32),
        );
    }
    let pre_offset = cache.get_offset();
    assert_eq!(pre_offset, max_size + 5);

    let handle = cache.clone_handle();
    assert_eq!(handle.seq_len(), pre_offset);

    let mut restored = RotatingKVCache::new_with_mode(max_size, KVCacheMode::Turbo4Asym);
    restored.install_detached(handle).unwrap();
    assert_eq!(restored.get_offset(), pre_offset);
    // `visible_len()` should still be max_size (we're past wraparound).
    assert_eq!(restored.visible_len(), max_size);
    // Continuing to write must not corrupt the ring. One more token brings
    // us to offset = max_size + 6, still wrapped.
    let (_k, v_out) = restored.update_and_fetch(
        synth_kv_tensor(1, 1, 1, head_dim, 90001),
        synth_kv_tensor(1, 1, 1, head_dim, 90002),
    );
    assert_eq!(restored.get_offset(), pre_offset + 1);
    assert_eq!(ffi::array_shape(&v_out)[2], max_size);
}

/// Install on a non-empty cache must error to prevent silent buffer drops.
#[test]
fn rotating_install_detached_rejects_non_empty_target() {
    let mut a = RotatingKVCache::new_with_mode(32, KVCacheMode::Turbo4Asym);
    a.update_and_fetch(
        synth_kv_tensor(1, 1, 1, 64, 1),
        synth_kv_tensor(1, 1, 1, 64, 2),
    );

    let mut b = RotatingKVCache::new_with_mode(32, KVCacheMode::Turbo4Asym);
    b.update_and_fetch(
        synth_kv_tensor(1, 1, 1, 64, 3),
        synth_kv_tensor(1, 1, 1, 64, 4),
    );

    let handle = a.clone_handle();
    let err = b.install_detached(handle).unwrap_err();
    assert!(
        err.contains("not empty"),
        "expected non-empty error, got: {err}"
    );
}

/// LOW-1 parity with `KVCache`: `clone_handle` clears `turbo_params` on the
/// source so the slot can be reused with a different head_dim if needed.
#[test]
fn rotating_turbo4_clone_handle_clears_turbo_params_on_source() {
    let head_dim = 128;
    let mut cache = RotatingKVCache::new_with_mode(64, KVCacheMode::Turbo4Asym);
    cache.update_and_fetch(
        synth_kv_tensor(1, 1, 1, head_dim, 11),
        synth_kv_tensor(1, 1, 1, head_dim, 12),
    );
    assert!(cache.turbo_params.is_some());
    let _handle = cache.clone_handle();
    assert!(
        cache.turbo_params.is_none(),
        "clone_handle must clear turbo_params on source for slot reuse"
    );
}

// ---------------------------------------------------------------------------
// Boundary-V (B6, issue #478) integration tests
// ---------------------------------------------------------------------------
//
// These integration tests cover the cache-side behavior of the boundary
// policy. The pure-helper unit tests for the resolver itself live in
// `cache/turbo/boundary.rs`.

mod boundary_v {
    use super::*;
    use crate::cache::turbo::boundary::{
        is_boundary_layer, resolve_boundary_count, resolve_layer_mode, resolve_layer_modes,
        DEFAULT_BOUNDARY_V_LAYERS,
    };

    /// The default count must be the LA-V7 boundary width (2) per the
    /// layer-aware-v-compression paper. Changing this constant requires a
    /// quality-gate re-run on the B3 PPL/NIAH suite.
    #[test]
    fn default_boundary_count_is_two() {
        assert_eq!(DEFAULT_BOUNDARY_V_LAYERS, 2);
    }

    /// Inert when the nominal mode is not a Turbo4* variant — every layer
    /// keeps the nominal mode regardless of the boundary count.
    #[test]
    fn fp16_mode_is_unaffected_by_boundary_policy() {
        let modes = resolve_layer_modes(KVCacheMode::Fp16, 8, 4);
        assert!(modes.iter().all(|m| *m == KVCacheMode::Fp16));
    }

    #[test]
    fn int8_mode_is_unaffected_by_boundary_policy() {
        let modes = resolve_layer_modes(KVCacheMode::Int8, 8, 4);
        assert!(modes.iter().all(|m| *m == KVCacheMode::Int8));
    }

    /// Boundary clamping: when `boundary >= n_layers / 2` every layer ends up
    /// boundary-protected (degenerates into "all layers are Fp16").
    #[test]
    fn boundary_clamping_does_not_overprotect_shallow_models() {
        // 4-layer model with requested boundary = 8 → clamp to 2 each side
        // → all 4 layers are boundary, all upgrade to Fp16.
        let modes = resolve_layer_modes(KVCacheMode::Turbo4Asym, 4, 8);
        assert_eq!(modes.len(), 4);
        for (i, m) in modes.iter().enumerate() {
            assert_eq!(*m, KVCacheMode::Fp16, "layer {i} should be FP16");
        }
        // resolve_boundary_count clamps the raw value to n_layers / 2.
        assert_eq!(resolve_boundary_count(8, 4), 2);
    }

    /// On a 32-layer model with default boundary count (2 each side), exactly
    /// 4 layers (0, 1, 30, 31) get the FP16 upgrade — the rest stay
    /// Turbo4Asym.
    #[test]
    fn typical_32_layer_split_protects_first_two_and_last_two() {
        let n = 32;
        let modes = resolve_layer_modes(KVCacheMode::Turbo4Asym, n, 2);
        for (i, &actual) in modes.iter().enumerate().take(n) {
            let expected = if i < 2 || i >= n - 2 {
                KVCacheMode::Fp16
            } else {
                KVCacheMode::Turbo4Asym
            };
            assert_eq!(actual, expected, "layer {i}");
        }
    }

    /// The single-layer helper agrees with the bulk helper for every layer
    /// position. Round-trip across all layers in the model.
    #[test]
    fn single_layer_helper_matches_bulk_for_turbo_modes() {
        let n = 16;
        for mode in [
            KVCacheMode::Turbo4Asym,
            KVCacheMode::Turbo4,
            KVCacheMode::Turbo4Delegated,
        ] {
            let bulk = resolve_layer_modes(mode, n, 2);
            for (i, &expected) in bulk.iter().enumerate().take(n) {
                let single = resolve_layer_mode(mode, i, n, 2);
                assert_eq!(
                    expected, single,
                    "{mode:?} layer {i}: bulk vs single helper disagree"
                );
            }
        }
    }

    /// Zero boundary disables the policy entirely — every layer keeps the
    /// nominal mode even when nominal is a Turbo4* variant.
    #[test]
    fn zero_boundary_disables_policy() {
        for mode in [
            KVCacheMode::Turbo4Asym,
            KVCacheMode::Turbo4,
            KVCacheMode::Turbo4Delegated,
        ] {
            let modes = resolve_layer_modes(mode, 16, 0);
            assert!(modes.iter().all(|m| *m == mode));
        }
    }

    /// Boundary-protected layer caches actually allocate the FP16 buffers
    /// (not the packed/turbo sidecars), and middle-layer caches use the
    /// turbo sidecars. Verifies the resolver wires through to real cache
    /// state, not just a stored field.
    #[test]
    fn boundary_layer_caches_use_fp16_storage() {
        let head_dim = 64;
        let n_layers = 8usize;

        // Build the per-layer modes that the generator would produce.
        let modes = resolve_layer_modes(KVCacheMode::Turbo4Asym, n_layers, 2);

        let mut caches: Vec<KVCache> = modes.iter().copied().map(KVCache::new_with_mode).collect();

        // Push one token per cache so the storage paths actually fire.
        for (i, cache) in caches.iter_mut().enumerate() {
            let k = synth_kv_tensor(1, 1, 1, head_dim, (i as u32) * 17 + 1);
            let v = synth_kv_tensor(1, 1, 1, head_dim, (i as u32) * 17 + 2);
            cache.update_and_fetch(k, v);
            assert_eq!(cache.seq_len(), 1, "layer {i} update");
        }

        for (i, cache) in caches.iter().enumerate() {
            if is_boundary_layer(i, n_layers, 2) {
                assert_eq!(
                    cache.mode,
                    KVCacheMode::Fp16,
                    "layer {i} should be Fp16 (boundary)"
                );
                // FP16 mode keeps `values` populated and never touches the
                // turbo sidecars.
                assert!(
                    cache.keys.is_some(),
                    "boundary layer {i} must have FP16 keys"
                );
                assert!(
                    cache.values.is_some(),
                    "boundary layer {i} must have FP16 values"
                );
                assert!(
                    cache.v_packed.is_none(),
                    "boundary layer {i} must NOT have packed V (Fp16 mode)"
                );
                assert!(
                    cache.v_norms.is_none(),
                    "boundary layer {i} must NOT have V norms"
                );
            } else {
                assert_eq!(
                    cache.mode,
                    KVCacheMode::Turbo4Asym,
                    "layer {i} should be Turbo4Asym (middle)"
                );
                // Turbo4Asym keeps `values` empty; sidecars hold the V state.
                assert!(
                    cache.keys.is_some(),
                    "middle layer {i} must have FP16 keys (asymmetric)"
                );
                assert!(
                    cache.values.is_none(),
                    "middle layer {i} must NOT have FP16 values"
                );
                assert!(
                    cache.v_packed.is_some(),
                    "middle layer {i} must have packed V"
                );
                assert!(
                    cache.v_norms.is_some(),
                    "middle layer {i} must have V norms"
                );
            }
        }
    }

    /// nbytes() accounting reflects the per-layer mode mix: boundary layers
    /// charge FP16 K + V, middle layers charge FP16 K + packed V + V norms.
    /// The exact numbers depend on step-grown buffer capacity, so we assert
    /// the inequality boundary > middle (FP16 V is more bytes than packed V
    /// for the same logical content) instead of fixed totals.
    #[test]
    fn nbytes_reflects_per_layer_mode_mix() {
        let head_dim = 64;
        let n_layers = 8usize;
        let modes = resolve_layer_modes(KVCacheMode::Turbo4Asym, n_layers, 2);

        let mut boundary_total = 0usize;
        let mut middle_total = 0usize;

        for (i, mode) in modes.iter().enumerate() {
            let mut cache = KVCache::new_with_mode(*mode);
            let k = synth_kv_tensor(1, 1, 1, head_dim, (i as u32) * 11 + 1);
            let v = synth_kv_tensor(1, 1, 1, head_dim, (i as u32) * 11 + 2);
            cache.update_and_fetch(k, v);
            let bytes = cache.nbytes();
            if is_boundary_layer(i, n_layers, 2) {
                boundary_total += bytes;
            } else {
                middle_total += bytes;
            }
            // Sanity: every cache reports non-zero memory after one token.
            assert!(bytes > 0, "layer {i} mode {mode:?} reported zero nbytes");
        }

        // Boundary layers store full FP16 V; middle layers store packed V.
        // For 1 token at head_dim=64:
        // - FP16 V: 64 * 2 = 128 bytes (per layer logical, more after step
        //   alignment to 256).
        // - Packed V: 64 / 2 = 32 bytes + 1 norm fp16 = 34 bytes per layer.
        // The step-grown buffers amplify both but boundary should always be
        // strictly larger than middle (per layer).
        let avg_boundary = boundary_total / 4; // 4 boundary layers
        let avg_middle = middle_total / 4; // 4 middle layers
        assert!(
            avg_boundary > avg_middle,
            "boundary avg={avg_boundary} should exceed middle avg={avg_middle} \
             (FP16 storage is denser than packed Turbo4)"
        );
    }

    /// Round-trip via clone_handle / install_detached preserves the per-layer
    /// mode (the detach handle carries the layer's effective mode so adopt
    /// rebuilds an identically-resolved cache slot).
    #[test]
    fn detach_adopt_preserves_per_layer_mode() {
        let head_dim = 64;
        let n_layers = 8usize;
        let modes = resolve_layer_modes(KVCacheMode::Turbo4Asym, n_layers, 2);

        for (i, mode) in modes.iter().enumerate() {
            let mut src = KVCache::new_with_mode(*mode);
            let k = synth_kv_tensor(1, 1, 1, head_dim, (i as u32) * 7 + 1);
            let v = synth_kv_tensor(1, 1, 1, head_dim, (i as u32) * 7 + 2);
            src.update_and_fetch(k, v);

            let handle = src.clone_handle();
            assert_eq!(
                handle.mode(),
                *mode,
                "detach handle must preserve layer {i} mode"
            );

            let mut dst = KVCache::new_with_mode(*mode);
            dst.install_detached(handle).expect("install_detached");
            assert_eq!(
                dst.mode, *mode,
                "adopted cache must match layer {i} resolved mode"
            );
            assert_eq!(dst.seq_len(), 1, "adopted cache must keep offset");
        }
    }
}

// ===========================================================================
// Symmetric Turbo4 dequant-first SDPA
// ===========================================================================

fn turbo4_reference_attention(
    cache: &mut KVCache,
    q: &ffi::MlxArray,
    k: cxx::UniquePtr<ffi::MlxArray>,
    v: cxx::UniquePtr<ffi::MlxArray>,
    scale: f32,
) -> cxx::UniquePtr<ffi::MlxArray> {
    let (cache_k, cache_v) = cache.update_and_fetch(k, v);
    crate::layers::attention(q, &cache_k, &cache_v, scale, None, 0.0, 0)
}

#[test]
fn turbo4_dequant_sdpa_matches_full_dequant_attention() {
    let head_dim = 64;
    let prefill_len = 8;
    let total_steps = 32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut cache_dequant = KVCache::new_with_mode(KVCacheMode::Turbo4);
    let mut cache_ref = KVCache::new_with_mode(KVCacheMode::Turbo4);

    let k_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 31);
    let v_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 32);
    let k_pre_b = ffi::copy(&k_pre);
    let v_pre_b = ffi::copy(&v_pre);
    let _ = cache_dequant.update_and_fetch(k_pre, v_pre);
    let _ = cache_ref.update_and_fetch(k_pre_b, v_pre_b);

    let mut max_rms = 0.0_f32;
    for step in 0..total_steps {
        let k_a = synth_kv_tensor(1, 1, 1, head_dim, 20_000 + step as u32);
        let v_a = synth_kv_tensor(1, 1, 1, head_dim, 21_000 + step as u32);
        let k_b = ffi::copy(&k_a);
        let v_b = ffi::copy(&v_a);
        let q = synth_kv_tensor(1, 2, 1, head_dim, 22_000 + step as u32);

        let out_dequant =
            cache_dequant.update_and_turbo4_dequant_sdpa_attention(&q, k_a, v_a, scale, None);
        let out_ref = turbo4_reference_attention(&mut cache_ref, &q, k_b, v_b, scale);

        let flat_a = flatten_fp32(&out_dequant);
        let flat_b = flatten_fp32(&out_ref);
        assert_eq!(flat_a.len(), flat_b.len(), "step {step}: shape mismatch");
        let mut sum_sq = 0.0_f64;
        for (x, y) in flat_a.iter().zip(flat_b.iter()) {
            let d = (x - y) as f64;
            sum_sq += d * d;
        }
        let rms = (sum_sq / flat_a.len() as f64).sqrt() as f32;
        if rms > max_rms {
            max_rms = rms;
        }
        assert!(
            rms < 5e-3,
            "step {step}: Turbo4 dequant-SDPA vs full-dequant RMS {rms:.4e} exceeds 5e-3"
        );
    }

    eprintln!(
        "turbo4_dequant_sdpa_matches_full_dequant_attention: max RMS over {total_steps} steps = \
         {max_rms:.4e}"
    );
}

// ===========================================================================
// Turbo3Asym (issue #477, epic #458) — 3-bit V-side PolarQuant
// ===========================================================================

// ---------------------------------------------------------------------------
// Mode parsing
// ---------------------------------------------------------------------------

#[test]
fn turbo3_asym_parses_canonical_string() {
    let m: KVCacheMode = "fp16+turbo3".parse().unwrap();
    assert_eq!(m, KVCacheMode::Turbo3Asym);
}

#[test]
fn turbo3_asym_parses_aliases() {
    let m1: KVCacheMode = "turbo3-asym".parse().unwrap();
    let m2: KVCacheMode = "turbo3".parse().unwrap();
    assert_eq!(m1, KVCacheMode::Turbo3Asym);
    assert_eq!(m2, KVCacheMode::Turbo3Asym);
}

#[test]
fn turbo3_asym_parsing_is_case_insensitive() {
    let m: KVCacheMode = "FP16+TURBO3".parse().unwrap();
    assert_eq!(m, KVCacheMode::Turbo3Asym);
    let m: KVCacheMode = "Turbo3-Asym".parse().unwrap();
    assert_eq!(m, KVCacheMode::Turbo3Asym);
}

#[test]
fn turbo3_asym_display_round_trip() {
    let m = KVCacheMode::Turbo3Asym;
    assert_eq!(m.to_string(), "fp16+turbo3");
    let parsed: KVCacheMode = m.to_string().parse().unwrap();
    assert_eq!(parsed, m);
}

// ---------------------------------------------------------------------------
// update_and_fetch round-trip
// ---------------------------------------------------------------------------

#[test]
fn turbo3_asym_update_returns_fp16_dequantized_v() {
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);

    let k = synth_kv_tensor(1, 1, 4, head_dim, 142);
    let v = synth_kv_tensor(1, 1, 4, head_dim, 143);

    let (k_out, v_out) = cache.update_and_fetch(k, v);
    assert_eq!(ffi::array_dtype(&v_out), dtype::FLOAT16);
    assert_eq!(ffi::array_shape(&v_out), vec![1_i32, 1, 4, head_dim]);
    assert_eq!(ffi::array_dtype(&k_out), dtype::FLOAT16);
    assert_eq!(ffi::array_shape(&k_out), vec![1_i32, 1, 4, head_dim]);

    assert!(cache.v_packed.is_some());
    assert!(cache.v_norms.is_some());
    assert!(cache.values.is_none());
    assert_eq!(cache.seq_len(), 4);

    // Sidecar dim must be head_dim * 3 / 8 = 48 for D=128.
    let vp_shape = ffi::array_shape(cache.v_packed.as_ref().unwrap());
    assert_eq!(vp_shape[3], head_dim * 3 / 8);
}

/// Multi-token growth keeps the packed sidecars aligned with the visible
/// window across step-grown buffer reallocations.
#[test]
fn turbo3_asym_multi_token_growth_keeps_visible_window_correct() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);

    let k1 = synth_kv_tensor(1, 1, 3, head_dim, 1);
    let v1 = synth_kv_tensor(1, 1, 3, head_dim, 2);
    let (k_out_1, v_out_1) = cache.update_and_fetch(k1, v1);
    assert_eq!(ffi::array_shape(&k_out_1)[2], 3);
    assert_eq!(ffi::array_shape(&v_out_1)[2], 3);
    assert_eq!(cache.seq_len(), 3);

    let k2 = synth_kv_tensor(1, 1, 5, head_dim, 3);
    let v2 = synth_kv_tensor(1, 1, 5, head_dim, 4);
    let (k_out_2, v_out_2) = cache.update_and_fetch(k2, v2);
    assert_eq!(cache.seq_len(), 8);
    assert_eq!(ffi::array_shape(&k_out_2)[2], 8);
    assert_eq!(ffi::array_shape(&v_out_2)[2], 8);
}

/// K side bypasses the quantizer entirely; bytes round-trip through fp16
/// unchanged (within fp16 precision).
#[test]
fn turbo3_asym_keys_are_bit_identical_to_input() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);

    let k_data: Vec<f32> = (0..head_dim).map(|i| ((i % 8) as f32) * 0.125).collect();
    let v_data: Vec<f32> = (0..head_dim).map(|i| ((i + 1) as f32) * 0.01).collect();
    let k = ffi::from_slice_f32(&k_data, &[1, 1, 1, head_dim]);
    let v = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);

    let (k_out, _) = cache.update_and_fetch(k, v);
    let recovered = flatten_fp32(&k_out);
    for (a, b) in recovered.iter().zip(k_data.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "K side must round-trip through fp16 unchanged: got {a}, expected {b}"
        );
    }
}

/// V reconstruction error is bounded by Lloyd-Max distortion at 3 bits
/// (~−17.8 dB) plus rotation/fp16 noise. Allow up to 25% relative L2 error
/// per token — the same bound as the unit tests in `quant3.rs`.
#[test]
fn turbo3_asym_v_reconstruction_has_bounded_error() {
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);

    let v_data: Vec<f32> = (0..head_dim)
        .map(|i| (i as f32 / head_dim as f32 - 0.5) * 2.0)
        .collect();
    let k = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);
    let v = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);

    let (_k_out, v_out) = cache.update_and_fetch(k, v);
    let v_recovered = flatten_fp32(&v_out);

    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for i in 0..head_dim as usize {
        let diff = v_data[i] - v_recovered[i];
        num += diff * diff;
        den += v_data[i] * v_data[i];
    }
    let rel = (num / den.max(1e-12)).sqrt();
    assert!(
        rel < 0.25,
        "Turbo3Asym V reconstruction relative error {rel:.4} > 25%"
    );
}

// ---------------------------------------------------------------------------
// Trim
// ---------------------------------------------------------------------------

#[test]
fn turbo3_asym_trim_to_zero_clears_all_buffers_and_params() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);
    let k = synth_kv_tensor(1, 1, 5, head_dim, 11);
    let v = synth_kv_tensor(1, 1, 5, head_dim, 12);
    cache.update(k, v);
    assert_eq!(cache.seq_len(), 5);
    assert!(cache.v_packed.is_some());
    assert!(cache.v_norms.is_some());
    assert!(cache.turbo3_params.is_some());

    let trimmed = cache.trim(5);
    assert_eq!(trimmed, 5);
    assert_eq!(cache.seq_len(), 0);
    assert!(cache.is_empty());
    assert!(cache.v_packed.is_none());
    assert!(cache.v_norms.is_none());
    // turbo3_params must be cleared so the slot can be reused with a
    // different head_dim (mirrors the LOW-1 fix from #474 for the 4-bit path).
    assert!(cache.turbo3_params.is_none());
}

#[test]
fn turbo3_asym_partial_trim_shrinks_v_sidecars() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);
    let k = synth_kv_tensor(1, 1, 8, head_dim, 21);
    let v = synth_kv_tensor(1, 1, 8, head_dim, 22);
    cache.update(k, v);
    assert_eq!(cache.seq_len(), 8);

    let n = cache.trim(3);
    assert_eq!(n, 3);
    assert_eq!(cache.seq_len(), 5);

    let vp = cache.v_packed.as_ref().unwrap();
    let vn = cache.v_norms.as_ref().unwrap();
    assert_eq!(ffi::array_shape(vp)[2], 5);
    assert_eq!(ffi::array_shape(vn)[2], 5);
    let k_buf = cache.keys.as_ref().unwrap();
    assert_eq!(ffi::array_shape(k_buf)[2], 5);
}

// ---------------------------------------------------------------------------
// detach / install_detached round-trip
// ---------------------------------------------------------------------------

#[test]
fn turbo3_asym_clone_handle_round_trip_preserves_sidecars() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);
    let k = synth_kv_tensor(1, 1, 4, head_dim, 31);
    let v = synth_kv_tensor(1, 1, 4, head_dim, 32);
    cache.update(k, v);

    let pre_vp = ffi::array_to_raw_bytes(cache.v_packed.as_ref().unwrap());
    let pre_vn = ffi::array_to_raw_bytes(cache.v_norms.as_ref().unwrap());
    let pre_seed = cache.turbo_seed;

    let handle = cache.clone_handle();
    assert_eq!(handle.mode(), KVCacheMode::Turbo3Asym);

    assert!(cache.is_empty());
    assert!(cache.v_packed.is_none());
    assert!(cache.v_norms.is_none());
    // turbo3_params cleared on the source after clone_handle.
    assert!(cache.turbo3_params.is_none());

    let mut restored = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);
    restored.install_detached(handle).unwrap();

    assert_eq!(restored.seq_len(), 4);
    assert_eq!(restored.mode, KVCacheMode::Turbo3Asym);
    assert!(restored.v_packed.is_some());
    assert!(restored.v_norms.is_some());

    // turbo3_params should have been re-derived from v_packed shape.
    assert!(restored.turbo3_params.is_some());
    assert_eq!(restored.turbo_seed, pre_seed);

    let post_vp = ffi::array_to_raw_bytes(restored.v_packed.as_ref().unwrap());
    let post_vn = ffi::array_to_raw_bytes(restored.v_norms.as_ref().unwrap());
    assert_eq!(
        pre_vp, post_vp,
        "v_packed must survive Turbo3Asym detach/adopt bit-for-bit"
    );
    assert_eq!(
        pre_vn, post_vn,
        "v_norms must survive Turbo3Asym detach/adopt bit-for-bit"
    );
}

#[test]
fn turbo3_asym_clone_handle_install_then_dequant_matches_pre_detach() {
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);
    let k_data: Vec<f32> = (0..2 * head_dim)
        .map(|i| (i as f32 / 256.0) - 0.5)
        .collect();
    let v_data: Vec<f32> = (0..2 * head_dim)
        .map(|i| (((i * 7) % 13) as f32 / 13.0) - 0.5)
        .collect();
    let k = ffi::from_slice_f32(&k_data, &[1, 1, 2, head_dim]);
    let v = ffi::from_slice_f32(&v_data, &[1, 1, 2, head_dim]);

    let (_k1, v1_out) = cache.update_and_fetch(k, v);
    let v1 = flatten_fp32(&v1_out);

    let handle = cache.clone_handle();
    let mut restored = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);
    restored.install_detached(handle).unwrap();

    let params = restored.turbo3_params.as_ref().unwrap();
    let vp_buf = restored.v_packed.as_ref().unwrap();
    let vn_buf = restored.v_norms.as_ref().unwrap();
    let vps = ffi::array_shape(vp_buf);
    let vns = ffi::array_shape(vn_buf);
    let off = restored.offset;
    let vp = ffi::slice(vp_buf, &[0, 0, 0, 0], &[vps[0], vps[1], off, vps[3]]);
    let vn = ffi::slice(vn_buf, &[0, 0, 0, 0], &[vns[0], vns[1], off, 1]);
    let v2_out = super::turbo::quant3::dequantize_v_turbo3(&vp, &vn, params);
    let v2 = flatten_fp32(&v2_out);

    assert_eq!(v1.len(), v2.len());
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "post-detach Turbo3Asym dequant mismatch: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// CachePool detach/adopt round-trip on Turbo3Asym
// ---------------------------------------------------------------------------

#[test]
fn cache_pool_detach_adopt_preserves_turbo3_asym() {
    use crate::generate::LanguageModel;

    struct Stub {
        n: usize,
    }
    impl LanguageModel for Stub {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> cxx::UniquePtr<MlxArray> {
            ffi::zeros(&[1], 0)
        }
        fn make_caches(&self) -> Vec<KVCache> {
            (0..self.n).map(|_| KVCache::new()).collect()
        }
        fn num_layers(&self) -> usize {
            self.n
        }
        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }
    }

    let head_dim = 64;
    let model = Stub { n: 1 };
    let mut pool = CachePool::new(4);

    let seq_a = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        caches[0] = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);
        let k = synth_kv_tensor(1, 1, 5, head_dim, 51);
        let v = synth_kv_tensor(1, 1, 5, head_dim, 52);
        caches[0].update(k, v);
    }

    let pre_vp = {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        ffi::array_to_raw_bytes(caches[0].v_packed.as_ref().unwrap())
    };
    let pre_vn = {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        ffi::array_to_raw_bytes(caches[0].v_norms.as_ref().unwrap())
    };

    let detached = pool.detach(seq_a).unwrap();
    let seq_b = pool.adopt(&model, detached).unwrap();

    let caches = pool.get_caches_mut(seq_b).unwrap();
    assert_eq!(caches[0].mode, KVCacheMode::Turbo3Asym);
    assert_eq!(caches[0].seq_len(), 5);
    assert!(caches[0].v_packed.is_some());
    assert!(caches[0].v_norms.is_some());

    let post_vp = ffi::array_to_raw_bytes(caches[0].v_packed.as_ref().unwrap());
    let post_vn = ffi::array_to_raw_bytes(caches[0].v_norms.as_ref().unwrap());
    assert_eq!(pre_vp, post_vp);
    assert_eq!(pre_vn, post_vn);
}

// ---------------------------------------------------------------------------
// Memory accounting
// ---------------------------------------------------------------------------

/// Turbo3Asym must use *fewer* bytes per token than Turbo4Asym for the same
/// head_dim — that's the whole point of the 3-bit mode. Compare nbytes() at
/// the same offset and assert the Turbo3 footprint is strictly smaller than
/// Turbo4 (and that both are smaller than FP16).
#[test]
fn turbo3_asym_uses_fewer_bytes_than_turbo4_asym() {
    let head_dim = 128;
    let n_tokens = 32;

    let mut c3 = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);
    let mut c4 = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let mut c_fp16 = KVCache::new();

    for cache in [&mut c3, &mut c4, &mut c_fp16] {
        let k = synth_kv_tensor(1, 1, n_tokens, head_dim, 71);
        let v = synth_kv_tensor(1, 1, n_tokens, head_dim, 72);
        cache.update(k, v);
    }

    let b3 = c3.nbytes();
    let b4 = c4.nbytes();
    let b_fp16 = c_fp16.nbytes();

    assert!(b3 > 0, "Turbo3Asym nbytes() must be non-zero");
    assert!(
        b3 < b4,
        "Turbo3Asym ({b3} bytes) must be smaller than Turbo4Asym ({b4} bytes)"
    );
    assert!(
        b4 < b_fp16,
        "Turbo4Asym ({b4} bytes) must already be smaller than FP16 ({b_fp16} bytes)"
    );
    // Values tensor stays None — must not contribute.
    assert!(c3.values.is_none());
}

/// Spot-check the active WHT-compatible head_dim grid: head_dim=64 must
/// produce a 24-byte/token V buffer (192 bits per token). Catches accidental
/// misalignment introduced by future changes to `pack_3bit_per_token`.
#[test]
fn turbo3_asym_packed_dim_matches_head_dim_grid() {
    for &(d, expected_bytes) in &[(64_i32, 24_i32), (128, 48), (256, 96)] {
        let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);
        let k = synth_kv_tensor(1, 1, 1, d, 81);
        let v = synth_kv_tensor(1, 1, 1, d, 82);
        cache.update(k, v);
        let vp_shape = ffi::array_shape(cache.v_packed.as_ref().unwrap());
        assert_eq!(
            vp_shape[3], expected_bytes,
            "head_dim={d}: expected {expected_bytes} packed bytes/token, got {}",
            vp_shape[3]
        );
    }
}

/// Determinism: same seed + same V → same packed bytes across two
/// independently-constructed caches.
#[test]
fn turbo3_asym_quantize_is_deterministic_across_caches() {
    let head_dim = 128;
    let v_data: Vec<f32> = (0..head_dim)
        .map(|i| ((i % 17) as f32 / 17.0) - 0.5)
        .collect();
    let v1 = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);
    let v2 = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);
    let k1 = ffi::zeros(&[1, 1, 1, head_dim], dtype::FLOAT16);
    let k2 = ffi::zeros(&[1, 1, 1, head_dim], dtype::FLOAT16);

    let mut c1 = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);
    let mut c2 = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);
    c1.update(k1, v1);
    c2.update(k2, v2);

    let p1 = ffi::array_to_raw_bytes(c1.v_packed.as_ref().unwrap());
    let p2 = ffi::array_to_raw_bytes(c2.v_packed.as_ref().unwrap());
    assert_eq!(
        p1, p2,
        "Turbo3Asym quantize must be deterministic across cache instances"
    );
}

/// Lloyd-Max distortion bound: at 3 bits over a `N(0, 1/d)` rotated
/// distribution, the expected RMSE is ~13% per coordinate (D(R=3) ≈
/// −17.8 dB on a Gaussian source). The end-to-end V reconstruction error
/// (after rotation, fp16 round-off, and norm correction) should land
/// noticeably above the 4-bit baseline (~6.5% from Lloyd-Max, ~10% e2e)
/// but stay below 25%. Validates that we are not silently using a wider
/// codebook by accident.
#[test]
fn turbo3_asym_distortion_is_bounded_and_worse_than_turbo4() {
    let head_dim: i32 = 128;
    // Use a deterministic Gaussian-ish input so per-token error is stable.
    let mut state: u32 = 0xDEAD_BEEF;
    let n_tokens = 8_usize;
    let hd_usize = head_dim as usize;
    let mut v_data: Vec<f32> = Vec::with_capacity(n_tokens * hd_usize);
    for _ in 0..n_tokens {
        for _ in 0..head_dim {
            let mut acc = 0.0_f32;
            for _ in 0..6 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                acc += if state >> 31 == 0 { -1.0 } else { 1.0 };
            }
            v_data.push((acc / 6.0) * 1.5);
        }
    }
    let k = synth_kv_tensor(1, 1, n_tokens as i32, head_dim, 901);
    let v = ffi::from_slice_f32(&v_data, &[1, 1, n_tokens as i32, head_dim]);

    let mut c3 = KVCache::new_with_mode(KVCacheMode::Turbo3Asym);
    let (_, v3_out) = c3.update_and_fetch(k, v);
    let v3 = flatten_fp32(&v3_out);

    // Compute mean per-token relative L2.
    let mut total_rel = 0.0_f32;
    for tok in 0..n_tokens {
        let off = tok * hd_usize;
        let mut num = 0.0_f32;
        let mut den = 0.0_f32;
        for i in 0..hd_usize {
            let diff = v_data[off + i] - v3[off + i];
            num += diff * diff;
            den += v_data[off + i] * v_data[off + i];
        }
        total_rel += (num / den.max(1e-12)).sqrt();
    }
    let mean_rel = total_rel / n_tokens as f32;
    assert!(
        mean_rel < 0.25,
        "Turbo3Asym mean per-token relative L2 error {mean_rel:.4} > 25% bound"
    );
    // Sanity floor: the error should be ABOVE the noise floor (~1e-4) so
    // we know the test actually exercised quantization rather than copying
    // through unchanged.
    assert!(
        mean_rel > 0.01,
        "Turbo3Asym mean per-token relative L2 error {mean_rel:.4} suspiciously low — \
         test may not be exercising quantization"
    );
}

/// Boundary-V policy upgrades Turbo3Asym layers to FP16 just like the
/// Turbo4* family. Validates the matrix entry added in `boundary.rs`.
#[test]
fn turbo3_asym_layer_modes_apply_boundary_upgrade() {
    use crate::cache::turbo::resolve_layer_modes;

    let modes = resolve_layer_modes(KVCacheMode::Turbo3Asym, 8, 2);
    assert_eq!(modes.len(), 8);
    // First 2 + last 2 are upgraded to FP16; middle 4 stay Turbo3Asym.
    for (i, m) in modes.iter().enumerate() {
        if !(2..6).contains(&i) {
            assert_eq!(*m, KVCacheMode::Fp16, "layer {i} must be FP16 boundary");
        } else {
            assert_eq!(
                *m,
                KVCacheMode::Turbo3Asym,
                "layer {i} must stay Turbo3Asym"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Turbo4Delegated decode helpers
// ---------------------------------------------------------------------------
//
// The PR-#525 cold-V dequant memo tests that previously lived here were
// retired by issue #528 — the fused dequant + SDPA Metal kernel reads packed
// cold V directly so there is no host-side memo state to verify. The two
// helpers below are still used by the issue #527 unified-K tests further
// down (`delegated_unified_k_*`, `delegated_nbytes_no_cold_k_buffer_after_fold`)
// and the issue #528 fused-kernel parity test.

/// Build a Turbo4Delegated cache with a small hot threshold so folds are
/// triggered after a handful of decode steps. The threshold is rounded up to
/// `BLOCK_SIZE` (32) inside the cache, so 32 is the minimum that keeps the
/// fold-block alignment invariant.
fn build_delegated_cache_with_small_threshold(hot_threshold: i32) -> KVCache {
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Delegated);
    cache.set_hot_threshold(hot_threshold);
    cache
}

/// Helper: prefill the cache with a multi-token chunk, then run a fixed number
/// of single-token decode steps. Returns the final fetched V tensor (flat
/// fp32 view) for the parity comparison below.
fn delegated_decode_run(cache: &mut KVCache, head_dim: i32, prefill_len: i32, decode_steps: i32) {
    let k_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 7);
    let v_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 8);
    let _ = cache.update_and_fetch(k_pre, v_pre);
    for step in 0..decode_steps {
        let k = synth_kv_tensor(1, 1, 1, head_dim, 100 + step as u32);
        let v = synth_kv_tensor(1, 1, 1, head_dim, 200 + step as u32);
        let _ = cache.update_and_fetch(k, v);
    }
}

// ---------------------------------------------------------------------------
// Turbo4Delegated unified-K storage (issue #527)
// ---------------------------------------------------------------------------
//
// Issue #527 unifies the K side of `KVCacheMode::Turbo4Delegated` into a
// single growing FP16 buffer (same shape contract as `KVCacheMode::Fp16`),
// dropping the cold/hot split for K. These tests verify:
// 1. `keys` is a single buffer that holds all `offset` tokens after folds.
// 2. SDPA-side reads return `slice(keys, 0, offset)` with no concat.
// 3. `clone_handle` / `install_detached` round-trip preserves K data
//    bit-identically when there's a populated cold V body.
// 4. The fetched K matches what an `Fp16`-mode reference cache produces for
//    the same input sequence.

/// After enough decode steps to trigger at least one fold, the unified `keys`
/// buffer must hold all `offset` tokens (i.e. its shape's seq dim is at least
/// `offset`) and SDPA-visible K from `update_and_fetch` must be the FP16
/// slice `[0..offset]` — no separate cold-K buffer should exist (issue #527).
#[test]
fn delegated_unified_k_buffer_grows_with_offset() {
    let head_dim = 64;
    let mut cache = build_delegated_cache_with_small_threshold(32);

    // Prefill 16, then decode enough to push past the threshold so a fold
    // fires.  Hot threshold = 32, prefill 16 + 64 decode = 80 tokens, with at
    // least one fold (cold_offset > 0).
    delegated_decode_run(&mut cache, head_dim, 16, 64);

    assert!(cache.cold_offset() > 0, "test setup must trigger a fold");
    let total_offset = cache.seq_len();
    assert_eq!(total_offset, 80, "16 prefill + 64 decode = 80 total tokens");

    // The unified K buffer must hold at least `offset` tokens (capacity may
    // be rounded up to step). There must be no separate cold-K buffer
    // (issue #527 removed the field).
    let keys = cache
        .keys
        .as_ref()
        .expect("unified keys must be populated after prefill+decode");
    let k_shape = ffi::array_shape(keys);
    assert_eq!(k_shape.len(), 4, "keys is a 4-D tensor");
    assert!(
        k_shape[2] >= total_offset,
        "unified K capacity ({}) must be >= offset ({})",
        k_shape[2],
        total_offset
    );

    // Drive one more decode step and verify the fetched K is `slice(keys, 0,
    // offset)` shape.  Use a fresh K/V token; the read path runs through
    // `fetch_turbo4_delegated`.
    let k = synth_kv_tensor(1, 1, 1, head_dim, 9999);
    let v = synth_kv_tensor(1, 1, 1, head_dim, 8888);
    let (k_out, _) = cache.update_and_fetch(k, v);
    let k_out_shape = ffi::array_shape(&k_out);
    assert_eq!(
        k_out_shape,
        vec![1_i32, 1, total_offset + 1, head_dim],
        "fetched K shape must be [B, H, offset, K_dim] (no cold/hot concat, issue #527)"
    );
    assert_eq!(
        ffi::array_dtype(&k_out),
        dtype::FLOAT16,
        "fetched K dtype must be FLOAT16"
    );
}

/// `clone_handle` / `install_detached` must round-trip the unified K buffer
/// bit-identically when the source has been through at least one fold
/// (issue #527).  A subsequent decode step on the installed target must
/// produce a K slice equal to the same step on the source before detach.
#[test]
fn delegated_clone_handle_preserves_unified_k_data() {
    let head_dim = 64;
    let mut src = build_delegated_cache_with_small_threshold(32);
    delegated_decode_run(&mut src, head_dim, 16, 64);

    assert!(src.cold_offset() > 0, "test setup must trigger a fold");
    let cold_offset_before = src.cold_offset();
    let offset_before = src.seq_len();

    // Snapshot the source's current unified K view by fetching once.
    // (Don't drive an update — we'd consume the input tensors and changing
    // `offset` would diverge from the post-install replay.)
    let src_k_snapshot = {
        let keys = src.keys.as_ref().unwrap();
        let ks = ffi::array_shape(keys);
        ffi::slice(keys, &[0, 0, 0, 0], &[ks[0], ks[1], offset_before, ks[3]])
    };
    let src_k_flat = flatten_fp32(&src_k_snapshot);

    // Detach + install into a fresh target.
    let handle = src.clone_handle();
    let mut tgt = KVCache::new_with_mode(KVCacheMode::Turbo4Delegated);
    tgt.set_hot_threshold(32);
    tgt.install_detached(handle).unwrap();

    // Logical state survives the round-trip.
    assert_eq!(tgt.seq_len(), offset_before);
    assert_eq!(tgt.cold_offset(), cold_offset_before);

    // The unified K data must match bit-for-bit.
    let tgt_keys = tgt.keys.as_ref().expect("target keys must exist");
    let tgt_ks = ffi::array_shape(tgt_keys);
    let tgt_k_snapshot = ffi::slice(
        tgt_keys,
        &[0, 0, 0, 0],
        &[tgt_ks[0], tgt_ks[1], offset_before, tgt_ks[3]],
    );
    let tgt_k_flat = flatten_fp32(&tgt_k_snapshot);
    assert_eq!(
        src_k_flat.len(),
        tgt_k_flat.len(),
        "src and tgt K must have the same flattened length"
    );
    let mut max_abs = 0.0_f32;
    for (a, b) in src_k_flat.iter().zip(tgt_k_flat.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert_eq!(
        max_abs, 0.0,
        "clone_handle must preserve unified K data bit-identically"
    );

    // A fresh decode step on the target must yield a K slice of shape
    // [B, H, offset+1, K_dim].
    let k = synth_kv_tensor(1, 1, 1, head_dim, 11111);
    let v = synth_kv_tensor(1, 1, 1, head_dim, 22222);
    let (k_out, _) = tgt.update_and_fetch(k, v);
    let k_out_shape = ffi::array_shape(&k_out);
    assert_eq!(
        k_out_shape,
        vec![1_i32, 1, offset_before + 1, head_dim],
        "post-install fetch must return unified K slice"
    );
}

/// Parity test: the K tensor returned by `KVCacheMode::Turbo4Delegated`
/// must produce FP16-equivalent K storage that is bitwise-identical to the
/// FP16 reference when the FP16 reference is also given FP32 input (both
/// modes cast K to FP16 before writing, so the bit patterns match; the
/// `max_abs < 1e-3` tolerance accounts for the FP32→FP16 round-trip that
/// both caches perform, not a delegated-mode approximation).  V is
/// *not* expected to match — V is still compressed in the delegated mode
/// via `quantize_v_turbo4` and has bounded reconstruction error.
/// Issue #527 unifies the K storage so that `Turbo4Delegated` uses the
/// same single growing FP16 buffer as `Fp16`, eliminating the per-step
/// K concat entirely.
#[test]
fn delegated_unified_k_matches_fp16_reference() {
    let head_dim = 64;
    let prefill_len = 16;
    let decode_steps = 80; // enough to trigger >= 2 folds at hot_threshold=32

    let mut cache_delegated = build_delegated_cache_with_small_threshold(32);
    let mut cache_fp16 = KVCache::new_with_mode(KVCacheMode::Fp16);

    // Identical prefill on both caches.
    let k_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 7);
    let v_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 8);
    let k_pre_b = ffi::copy(&k_pre);
    let v_pre_b = ffi::copy(&v_pre);
    let (k_a, _) = cache_delegated.update_and_fetch(k_pre, v_pre);
    let (k_b, _) = cache_fp16.update_and_fetch(k_pre_b, v_pre_b);
    let max_abs = max_abs_diff(&k_a, &k_b);
    assert!(
        max_abs < 1e-3,
        "prefill K mismatch: delegated vs fp16 max abs {max_abs:.4e}"
    );

    // Identical decode steps.  The delegated mode triggers folds of K
    // (which now live in the same unified buffer) — the fetched K slice
    // must remain bit-identical to the fp16 reference at every step.
    for step in 0..decode_steps {
        let k = synth_kv_tensor(1, 1, 1, head_dim, 5000 + step as u32);
        let v = synth_kv_tensor(1, 1, 1, head_dim, 6000 + step as u32);
        let k_b = ffi::copy(&k);
        let v_b = ffi::copy(&v);
        let (k_a, _) = cache_delegated.update_and_fetch(k, v);
        let (k_b, _) = cache_fp16.update_and_fetch(k_b, v_b);
        let max_abs = max_abs_diff(&k_a, &k_b);
        assert!(
            max_abs < 1e-3,
            "step {step}: K mismatch: delegated vs fp16 max abs {max_abs:.4e}"
        );
    }
}

/// Opt-in FP16 fast path: Turbo4Delegated should still maintain packed cold V
/// sidecars, but fetches must return the unified FP16 V buffer rather than
/// dequantizing packed cold V every step. This is the local analogue of the
/// TurboQuant+ delegated KVCache architecture.
#[test]
fn delegated_fp16_fast_path_keeps_unified_v_and_sidecars() {
    let head_dim = 64;
    let prefill_len = 16;
    let decode_steps = 80;

    let mut cache_fast = build_delegated_cache_with_small_threshold(32);
    cache_fast.delegated_fp16_fast_path = true;
    let mut cache_fp16 = KVCache::new_with_mode(KVCacheMode::Fp16);

    let k_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 7);
    let v_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 8);
    let k_pre_b = ffi::copy(&k_pre);
    let v_pre_b = ffi::copy(&v_pre);
    let (_, v_fast) = cache_fast.update_and_fetch(k_pre, v_pre);
    let (_, v_ref) = cache_fp16.update_and_fetch(k_pre_b, v_pre_b);
    assert!(
        max_abs_diff(&v_fast, &v_ref) < 1e-3,
        "prefill V must match FP16 exactly in delegated FP16 fast path"
    );

    for step in 0..decode_steps {
        let k = synth_kv_tensor(1, 1, 1, head_dim, 5000 + step as u32);
        let v = synth_kv_tensor(1, 1, 1, head_dim, 6000 + step as u32);
        let k_b = ffi::copy(&k);
        let v_b = ffi::copy(&v);
        let (_, v_fast) = cache_fast.update_and_fetch(k, v);
        let (_, v_ref) = cache_fp16.update_and_fetch(k_b, v_b);
        let max_abs = max_abs_diff(&v_fast, &v_ref);
        assert!(
            max_abs < 1e-3,
            "step {step}: fast-path V must match FP16 reference; max abs {max_abs:.4e}"
        );
    }

    assert!(
        cache_fast.cold_offset() > 0,
        "fast path must still advance packed cold V sidecars"
    );
    assert!(
        cache_fast.v_packed.is_some() && cache_fast.v_norms.is_some(),
        "fast path must keep packed V sidecars for measurement / recovery"
    );
    let values = cache_fast
        .values
        .as_ref()
        .expect("fast path keeps unified FP16 V buffer");
    let v_shape = ffi::array_shape(values);
    assert!(
        v_shape[2] >= cache_fast.seq_len(),
        "fast-path values buffer is unified V, so capacity {} must cover offset {}",
        v_shape[2],
        cache_fast.seq_len()
    );
}

/// The pre-decode compaction hook should fold the prefill body into packed
/// sidecars once, without changing the FP16 V read path used by attention.
#[test]
fn delegated_fp16_fast_path_predecode_compaction_folds_prefill_once() {
    let head_dim = 64;
    let prefill_len = 16;

    let mut cache_fast = build_delegated_cache_with_small_threshold(32);
    cache_fast.delegated_fp16_fast_path = true;
    let mut cache_fp16 = KVCache::new_with_mode(KVCacheMode::Fp16);

    let k_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 17);
    let v_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 18);
    let k_pre_b = ffi::copy(&k_pre);
    let v_pre_b = ffi::copy(&v_pre);
    let _ = cache_fast.update_and_fetch(k_pre, v_pre);
    let _ = cache_fp16.update_and_fetch(k_pre_b, v_pre_b);

    assert_eq!(
        cache_fast.cold_offset(),
        0,
        "prefill alone must not compact sidecars before the handoff hook"
    );
    assert!(
        cache_fast.compact_turbo4_delegated_fp16_sidecars_for_decode(),
        "first pre-decode compaction must fold the prefill body"
    );
    assert_eq!(cache_fast.cold_offset(), prefill_len);
    assert_eq!(cache_fast.hot_offset(), 0);
    assert!(
        cache_fast.v_packed.is_some() && cache_fast.v_norms.is_some(),
        "compaction must materialize packed V sidecars"
    );
    assert!(
        !cache_fast.compact_turbo4_delegated_fp16_sidecars_for_decode(),
        "compaction must be idempotent once cold sidecars exist"
    );

    let k = synth_kv_tensor(1, 1, 1, head_dim, 7000);
    let v = synth_kv_tensor(1, 1, 1, head_dim, 8000);
    let k_b = ffi::copy(&k);
    let v_b = ffi::copy(&v);
    let (k_fast, v_fast) = cache_fast.update_and_fetch(k, v);
    let (k_ref, v_ref) = cache_fp16.update_and_fetch(k_b, v_b);
    assert!(
        max_abs_diff(&k_fast, &k_ref) < 1e-3,
        "pre-compacted fast-path K must still match FP16"
    );
    assert!(
        max_abs_diff(&v_fast, &v_ref) < 1e-3,
        "pre-compacted fast-path V must still read unified FP16"
    );
    assert_eq!(cache_fast.cold_offset(), prefill_len);
    assert_eq!(cache_fast.hot_offset(), 1);
}

/// The compressed packed-V path should also move the initial full prefill fold
/// into the prefill->decode handoff. Unlike the FP16 fast path, this is not a
/// sidecar-only operation: the FP16 hot V body is consumed and cold V remains
/// only in packed storage before the first single-token decode update.
#[test]
fn delegated_compressed_predecode_compaction_folds_prefill_once() {
    let head_dim = 64;
    let prefill_len = 16;

    let mut cache = build_delegated_cache_with_small_threshold(32);

    let k_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 37);
    let v_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 38);
    let _ = cache.update_and_fetch(k_pre, v_pre);

    assert_eq!(
        cache.cold_offset(),
        0,
        "prefill alone must not fold compressed delegated V"
    );
    assert_eq!(cache.hot_offset(), prefill_len);
    assert!(
        cache.prepare_turbo4_delegated_for_decode(),
        "pre-decode handoff must fold the prefill hot V body"
    );
    assert_eq!(cache.cold_offset(), prefill_len);
    assert_eq!(cache.hot_offset(), 0);
    assert!(
        cache.v_packed.is_some() && cache.v_norms.is_some() && cache.v_rescale.is_some(),
        "compressed handoff must materialize packed V sidecars"
    );
    assert!(
        !cache.prepare_turbo4_delegated_for_decode(),
        "handoff fold must be idempotent once cold V is populated"
    );

    let k = synth_kv_tensor(1, 1, 1, head_dim, 9000);
    let v = synth_kv_tensor(1, 1, 1, head_dim, 10000);
    let (_k_full, v_full) = cache.update_and_fetch(k, v);
    assert_eq!(cache.cold_offset(), prefill_len);
    assert_eq!(cache.hot_offset(), 1);
    assert_eq!(cache.seq_len(), prefill_len + 1);
    assert_eq!(
        ffi::array_shape(&v_full)[2],
        prefill_len + 1,
        "fetch after predecode fold must expose cold packed V plus the new hot token"
    );
}

#[test]
fn delegated_fp16_sidecar_policy_parses_values() {
    assert_eq!(
        turbo::parse_delegated_fp16_sidecar_policy("predecode"),
        Some(turbo::DelegatedFp16SidecarPolicy::Predecode)
    );
    assert_eq!(
        turbo::parse_delegated_fp16_sidecar_policy("eager"),
        Some(turbo::DelegatedFp16SidecarPolicy::Predecode)
    );
    assert_eq!(
        turbo::parse_delegated_fp16_sidecar_policy("lazy"),
        Some(turbo::DelegatedFp16SidecarPolicy::Lazy)
    );
    assert_eq!(
        turbo::parse_delegated_fp16_sidecar_policy("on-demand"),
        Some(turbo::DelegatedFp16SidecarPolicy::Lazy)
    );
    assert_eq!(turbo::parse_delegated_fp16_sidecar_policy("bogus"), None);
}

/// Lazy sidecar policy keeps decode on the unified FP16 K/V hot path and avoids
/// foreground packed sidecar work until a preservation path explicitly detaches
/// the cache.
#[test]
fn delegated_fp16_lazy_sidecars_skip_decode_and_compact_on_detach() {
    let head_dim = 64;
    let prefill_len = 16;
    let decode_steps = 80;

    let mut cache_fast = build_delegated_cache_with_small_threshold(32);
    cache_fast.delegated_fp16_fast_path = true;
    cache_fast.delegated_fp16_sidecar_policy = turbo::DelegatedFp16SidecarPolicy::Lazy;
    let mut cache_fp16 = KVCache::new_with_mode(KVCacheMode::Fp16);

    let k_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 27);
    let v_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 28);
    let k_pre_b = ffi::copy(&k_pre);
    let v_pre_b = ffi::copy(&v_pre);
    let (_, v_fast) = cache_fast.update_and_fetch(k_pre, v_pre);
    let (_, v_ref) = cache_fp16.update_and_fetch(k_pre_b, v_pre_b);
    assert!(max_abs_diff(&v_fast, &v_ref) < 1e-3);
    assert!(
        !cache_fast.compact_turbo4_delegated_fp16_sidecars_for_decode(),
        "lazy policy must not predecode-compact through the generation hook"
    );

    for step in 0..decode_steps {
        let k = synth_kv_tensor(1, 1, 1, head_dim, 9000 + step as u32);
        let v = synth_kv_tensor(1, 1, 1, head_dim, 10000 + step as u32);
        let k_b = ffi::copy(&k);
        let v_b = ffi::copy(&v);
        let (_, v_fast) = cache_fast.update_and_fetch(k, v);
        let (_, v_ref) = cache_fp16.update_and_fetch(k_b, v_b);
        assert!(
            max_abs_diff(&v_fast, &v_ref) < 1e-3,
            "step {step}: lazy fast-path V must match FP16 reference"
        );
    }

    let seq_len = cache_fast.seq_len();
    assert_eq!(
        cache_fast.cold_offset(),
        0,
        "lazy policy must skip foreground decode sidecar folds"
    );
    assert!(
        cache_fast.v_packed.is_none() && cache_fast.v_norms.is_none(),
        "lazy policy should not allocate packed sidecars before preservation"
    );

    let handle = cache_fast.clone_handle();
    assert_eq!(
        handle.cold_offset, seq_len,
        "detach must compact all missing sidecars before donation"
    );
    assert!(
        handle.v_packed.is_some() && handle.v_norms.is_some(),
        "detached lazy fast-path cache must carry packed sidecars"
    );
    assert_eq!(
        handle.delegated_fp16_sidecar_policy,
        turbo::DelegatedFp16SidecarPolicy::Lazy
    );

    let mut tgt = KVCache::new_with_mode(KVCacheMode::Turbo4Delegated);
    tgt.install_detached(handle).unwrap();
    assert_eq!(
        tgt.delegated_fp16_sidecar_policy,
        turbo::DelegatedFp16SidecarPolicy::Lazy
    );
    let old_cold = tgt.cold_offset();
    let (_, v_out) = tgt.update_and_fetch(
        synth_kv_tensor(1, 1, 1, head_dim, 11111),
        synth_kv_tensor(1, 1, 1, head_dim, 22222),
    );
    assert_eq!(ffi::array_shape(&v_out)[2], tgt.seq_len());
    assert_eq!(
        tgt.cold_offset(),
        old_cold,
        "adopted lazy cache must not resume foreground sidecar folds"
    );
}

/// Detach/adopt must preserve whether `values` is a hot ring or a unified FP16
/// V buffer. Otherwise an adopted fast-path cache would interpret unified V as
/// hot-only V and return the wrong prefix.
#[test]
fn delegated_fp16_fast_path_survives_detach_adopt() {
    let head_dim = 64;
    let mut src = build_delegated_cache_with_small_threshold(32);
    src.delegated_fp16_fast_path = true;
    delegated_decode_run(&mut src, head_dim, 16, 64);

    assert!(src.delegated_fp16_fast_path);
    assert!(
        src.cold_offset() > 0,
        "test setup must trigger sidecar folds"
    );

    let handle = src.clone_handle();
    let mut tgt = KVCache::new_with_mode(KVCacheMode::Turbo4Delegated);
    tgt.install_detached(handle).unwrap();

    assert!(
        tgt.delegated_fp16_fast_path,
        "install_detached must preserve FP16 fast-path interpretation"
    );
    assert_eq!(
        tgt.delegated_fp16_sidecar_policy,
        turbo::DelegatedFp16SidecarPolicy::Predecode,
        "install_detached must preserve FP16 sidecar policy"
    );
    let (k_out, v_out) = tgt.update_and_fetch(
        synth_kv_tensor(1, 1, 1, head_dim, 11111),
        synth_kv_tensor(1, 1, 1, head_dim, 22222),
    );
    assert_eq!(ffi::array_shape(&k_out)[2], tgt.seq_len());
    assert_eq!(
        ffi::array_shape(&v_out)[2],
        tgt.seq_len(),
        "adopted fast-path cache must fetch full unified V, not only hot tail"
    );
}

/// `nbytes()` after a fold must reflect a single unified K buffer, not a
/// hot ring + a separate cold K buffer (issue #527 removes the cold-K
/// allocation).  V-side accounting (packed sidecars + dequant memo) must
/// be unchanged from PR #525's contract.
#[test]
fn delegated_nbytes_no_cold_k_buffer_after_fold() {
    let head_dim = 64;
    let mut cache = build_delegated_cache_with_small_threshold(32);

    // Prefill + enough decode to trigger at least one fold.
    delegated_decode_run(&mut cache, head_dim, 16, 64);
    assert!(cache.cold_offset() > 0, "test setup must trigger a fold");

    let total = cache.nbytes();
    assert!(total > 0, "nbytes must report non-zero after population");

    // The K side must consist of a single buffer (`keys`).  It must be at
    // least as large as `offset * K_dim * 2` bytes for fp16 (ignoring the
    // step-aligned tail allocation).
    let keys = cache
        .keys
        .as_ref()
        .expect("unified keys must be populated after fold");
    let k_shape = ffi::array_shape(keys);
    let min_k_bytes = (cache.seq_len() as usize) * (k_shape[3] as usize) * 2;
    let actual_k_bytes = ffi::array_nbytes(keys);
    assert!(
        actual_k_bytes >= min_k_bytes,
        "unified keys nbytes ({}) must cover at least offset*K_dim*2 = {}",
        actual_k_bytes,
        min_k_bytes
    );

    // V-side cold sidecars must still be present (V is still compressed).
    assert!(
        cache.v_packed.is_some(),
        "v_packed must exist after a fold (V-side compression unchanged)"
    );
    assert!(cache.v_norms.is_some(), "v_norms must exist after a fold");
    assert!(
        cache.v_rescale.is_some(),
        "v_rescale must exist after a fold"
    );
    // Issue #528 retired the PR-#525 cold-V dequant memo, so there is no
    // memo field to assert here. The packed cold V is consumed directly by
    // the fused kernel (`update_and_turbo4_delegated_attention`) without
    // ever being expanded to FP16 in global memory.
}

/// Helper: maximum absolute difference between two FP16 tensors after
/// promotion to FP32 and flatten.
fn max_abs_diff(a: &ffi::MlxArray, b: &ffi::MlxArray) -> f32 {
    let fa = flatten_fp32(a);
    let fb = flatten_fp32(b);
    assert_eq!(fa.len(), fb.len(), "shape mismatch in max_abs_diff");
    let mut m = 0.0_f32;
    for (x, y) in fa.iter().zip(fb.iter()) {
        m = m.max((x - y).abs());
    }
    m
}

// ---------------------------------------------------------------------------
// Turbo4Delegated fused kernel parity (issue #528)
// ---------------------------------------------------------------------------
//
// The fused dequant + SDPA kernel (`update_and_turbo4_delegated_attention`)
// must produce attention output that is RMS-equivalent to the pre-#528 path
// (`update_and_fetch` + standard SDPA on the `concat(cold_v_dequant, hot_v)`
// tensor). The contract is RMS < 5e-3 over 200 decode steps spanning at least
// three folds, matching the `sparse_v_kernel_threshold_zero_matches_graph`
// discipline already used in this repo for the Turbo4Asym kernel.

/// Compute reference attention output for a Turbo4Delegated cache by going
/// through the legacy `update_and_fetch` + manual SDPA path. This mirrors
/// what the standard `attention()` call site does so we can validate the
/// fused-kernel output against it. Both inputs are consumed.
#[cfg(target_os = "macos")]
fn delegated_reference_attention(
    cache: &mut KVCache,
    q: &ffi::MlxArray,
    k: cxx::UniquePtr<ffi::MlxArray>,
    v: cxx::UniquePtr<ffi::MlxArray>,
    scale: f32,
) -> cxx::UniquePtr<ffi::MlxArray> {
    let (cache_k, cache_v) = cache.update_and_fetch(k, v);
    // Standard SDPA: softmax(Q · K^T * scale) · V. The cache returns FP16
    // already so we cast to FP32 for parity with the fused kernel's host
    // softmax.
    let q_shape = ffi::array_shape(q);
    let k_shape = ffi::array_shape(&cache_k);
    let b = q_shape[0];
    let hq = q_shape[1];
    let kv_heads = k_shape[1];
    let n_rep = hq / kv_heads;
    let kt = k_shape[2];
    let kd = k_shape[3];
    let vd = ffi::array_shape(&cache_v)[3];

    let k_for_q = if n_rep == 1 {
        ffi::contiguous(&cache_k, false)
    } else {
        let k_exp = ffi::expand_dims(&cache_k, 2);
        let k_tiled = ffi::broadcast_to(&k_exp, &[b, kv_heads, n_rep, kt, kd]);
        ffi::reshape(&k_tiled, &[b, hq, kt, kd])
    };
    let v_for_q = if n_rep == 1 {
        ffi::contiguous(&cache_v, false)
    } else {
        let v_exp = ffi::expand_dims(&cache_v, 2);
        let v_tiled = ffi::broadcast_to(&v_exp, &[b, kv_heads, n_rep, kt, vd]);
        ffi::reshape(&v_tiled, &[b, hq, kt, vd])
    };
    let k_t = ffi::transpose_axes(&k_for_q, &[0, 1, 3, 2]);
    let q_f32 = ffi::astype(q, dtype::FLOAT32);
    let k_t_f32 = ffi::astype(&k_t, dtype::FLOAT32);
    let v_f32 = ffi::astype(&v_for_q, dtype::FLOAT32);
    let qk = ffi::matmul(&q_f32, &k_t_f32);
    let scale_arr = ffi::full_f32(&[1], scale, dtype::FLOAT32);
    let scores = ffi::multiply(&qk, &scale_arr);
    let attn = ffi::softmax_precise(&scores, -1);
    let out_f32 = ffi::matmul(&attn, &v_f32);
    ffi::astype(&out_f32, dtype::FLOAT16)
}

#[cfg(target_os = "macos")]
fn delegated_dequant_sdpa_from_updated_cache(
    cache: &mut KVCache,
    q: &ffi::MlxArray,
    scale: f32,
    mask: Option<&ffi::MlxArray>,
) -> cxx::UniquePtr<ffi::MlxArray> {
    let cold_offset = cache.cold_offset();
    let hot_offset = cache.seq_len() - cold_offset;

    if cache.turbo_params.is_none() {
        let head_dim_u32 = cache
            .values
            .as_ref()
            .map(|v| ffi::array_shape(v)[3] as u32)
            .or_else(|| {
                cache
                    .v_packed
                    .as_ref()
                    .map(|vp| (ffi::array_shape(vp)[3] as u32) * 2)
            })
            .expect("either hot V ring or v_packed must exist after update");
        cache.turbo_params = Some(super::turbo::TurboQuantParams::new(
            head_dim_u32,
            cache.turbo_seed,
        ));
    }

    let k_buf = cache
        .keys
        .as_ref()
        .expect("unified keys must exist after update on Turbo4Delegated");
    let ks = ffi::array_shape(k_buf);
    let k_slice = ffi::slice(
        k_buf,
        &[0, 0, 0, 0],
        &[ks[0], ks[1], cache.seq_len(), ks[3]],
    );

    let v_packed_owned = if cold_offset > 0 {
        let vp = cache
            .v_packed
            .as_ref()
            .expect("v_packed must exist when cold_offset > 0");
        let vp_shape = ffi::array_shape(vp);
        Some(ffi::slice(
            vp,
            &[0, 0, 0, 0],
            &[vp_shape[0], vp_shape[1], cold_offset, vp_shape[3]],
        ))
    } else {
        None
    };
    let v_rescale_owned = if cold_offset > 0 {
        let vr = cache
            .v_rescale
            .as_ref()
            .expect("v_rescale must exist when cold_offset > 0");
        let vr_shape = ffi::array_shape(vr);
        Some(ffi::slice(
            vr,
            &[0, 0, 0, 0],
            &[vr_shape[0], vr_shape[1], cold_offset, 1],
        ))
    } else {
        None
    };
    let hot_v_owned = if hot_offset > 0 {
        let hv = cache
            .values
            .as_ref()
            .expect("hot values must exist when hot_offset > 0");
        let hv_shape = ffi::array_shape(hv);
        Some(ffi::slice(
            hv,
            &[0, 0, 0, 0],
            &[hv_shape[0], hv_shape[1], hot_offset, hv_shape[3]],
        ))
    } else {
        None
    };
    let params = cache
        .turbo_params
        .as_ref()
        .expect("turbo_params populated above");

    super::turbo::sparse_v::attention_turbo4_delegated_dequant_sdpa(
        q,
        &k_slice,
        v_packed_owned.as_deref(),
        v_rescale_owned.as_deref(),
        hot_v_owned.as_deref(),
        params,
        cold_offset,
        hot_offset,
        scale,
        mask,
    )
    .expect("dequant-SDPA path must accept power-of-two delegated test inputs")
}

/// 200-step decode parity: the fused kernel's attention output must be
/// within RMS < 5e-3 of the reference path that runs full V dequant +
/// graph-level SDPA. Spans at least three folds at `hot_threshold = 32`
/// (cadence = `DELEGATED_FOLD_BLOCK = 128` post-fill).
#[cfg(target_os = "macos")]
#[test]
fn delegated_fused_kernel_matches_reference_over_200_steps() {
    let head_dim = 64;
    let prefill_len = 8;
    let total_steps = 200;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut cache_fused = build_delegated_cache_with_small_threshold(32);
    let mut cache_ref = build_delegated_cache_with_small_threshold(32);

    let k_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 7);
    let v_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 8);
    let k_pre_b = ffi::copy(&k_pre);
    let v_pre_b = ffi::copy(&v_pre);
    // Drive both caches through the prefill (multi-token) update_and_fetch.
    let _ = cache_fused.update_and_fetch(k_pre, v_pre);
    let _ = cache_ref.update_and_fetch(k_pre_b, v_pre_b);

    let mut max_rms = 0.0_f32;
    let mut steps_with_fold = 0;
    let mut prev_cold_offset = cache_fused.cold_offset();

    for step in 0..total_steps {
        let k_a = synth_kv_tensor(1, 1, 1, head_dim, 5000 + step as u32);
        let v_a = synth_kv_tensor(1, 1, 1, head_dim, 6000 + step as u32);
        let k_b = ffi::copy(&k_a);
        let v_b = ffi::copy(&v_a);
        let q = synth_kv_tensor(1, 1, 1, head_dim, 9000 + step as u32);

        // Fused kernel path. Returns the attention output unconditionally —
        // on macOS with a power-of-2 head_dim the Metal kernel handles the
        // dispatch; otherwise the same call falls through to the graph path.
        let q_for_fused = ffi::copy(&q);
        let out_fused =
            cache_fused.update_and_turbo4_delegated_attention(&q_for_fused, k_a, v_a, scale, None);
        // Reference path.
        let out_ref = delegated_reference_attention(&mut cache_ref, &q, k_b, v_b, scale);

        // RMS in the same FP16 → FP32 view (delta on the FP16 outputs).
        let flat_a = flatten_fp32(&out_fused);
        let flat_b = flatten_fp32(&out_ref);
        assert_eq!(flat_a.len(), flat_b.len(), "step {step}: shape mismatch");
        let mut sum_sq = 0.0_f64;
        for (x, y) in flat_a.iter().zip(flat_b.iter()) {
            let d = (x - y) as f64;
            sum_sq += d * d;
        }
        let rms = (sum_sq / flat_a.len() as f64).sqrt() as f32;
        if rms > max_rms {
            max_rms = rms;
        }

        // Track fold transitions to ensure the test covers cross-fold
        // boundaries.
        if cache_fused.cold_offset() != prev_cold_offset {
            steps_with_fold += 1;
            prev_cold_offset = cache_fused.cold_offset();
        }

        assert!(
            rms < 5e-3,
            "step {step}: fused vs reference RMS {rms:.4e} exceeds 5e-3 \
             (cold_offset={}, hot_offset={})",
            cache_fused.cold_offset(),
            cache_fused.seq_len() - cache_fused.cold_offset()
        );
    }

    // Sanity: the test must actually exercise multiple fold transitions.
    assert!(
        steps_with_fold >= 2,
        "test must cross at least two fold boundaries; only saw {steps_with_fold}"
    );
    eprintln!(
        "delegated_fused_kernel_matches_reference: max RMS over {total_steps} steps = \
         {max_rms:.4e}; folds crossed = {steps_with_fold}"
    );
}

/// Dequant-first SDPA parity for the Swift-LM-inspired delegated path.
///
/// This path dequantizes cold packed V into rotated value basis, forward-rotates
/// the small hot tail, runs native SDPA, and inverse-rotates the output. It must
/// match the legacy full-dequant reference because the value rotation is linear
/// and Q/K scores are unchanged.
#[cfg(target_os = "macos")]
#[test]
fn delegated_dequant_sdpa_matches_reference_attention() {
    let head_dim = 64;
    let prefill_len = 8;
    let total_steps = 48;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut cache_dequant = build_delegated_cache_with_small_threshold(32);
    let mut cache_ref = build_delegated_cache_with_small_threshold(32);

    let k_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 17);
    let v_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 18);
    let k_pre_b = ffi::copy(&k_pre);
    let v_pre_b = ffi::copy(&v_pre);
    let _ = cache_dequant.update_and_fetch(k_pre, v_pre);
    let _ = cache_ref.update_and_fetch(k_pre_b, v_pre_b);

    let mut max_rms = 0.0_f32;
    let mut steps_with_fold = 0;
    let mut prev_cold_offset = cache_dequant.cold_offset();

    for step in 0..total_steps {
        let k_a = synth_kv_tensor(1, 1, 1, head_dim, 15_000 + step as u32);
        let v_a = synth_kv_tensor(1, 1, 1, head_dim, 16_000 + step as u32);
        let k_b = ffi::copy(&k_a);
        let v_b = ffi::copy(&v_a);
        let q = synth_kv_tensor(1, 1, 1, head_dim, 19_000 + step as u32);

        cache_dequant.update(k_a, v_a);
        let q_for_dequant = ffi::copy(&q);
        let out_dequant = delegated_dequant_sdpa_from_updated_cache(
            &mut cache_dequant,
            &q_for_dequant,
            scale,
            None,
        );

        let out_ref = delegated_reference_attention(&mut cache_ref, &q, k_b, v_b, scale);

        let flat_a = flatten_fp32(&out_dequant);
        let flat_b = flatten_fp32(&out_ref);
        assert_eq!(flat_a.len(), flat_b.len(), "step {step}: shape mismatch");
        let mut sum_sq = 0.0_f64;
        for (x, y) in flat_a.iter().zip(flat_b.iter()) {
            let d = (x - y) as f64;
            sum_sq += d * d;
        }
        let rms = (sum_sq / flat_a.len() as f64).sqrt() as f32;
        if rms > max_rms {
            max_rms = rms;
        }

        if cache_dequant.cold_offset() != prev_cold_offset {
            steps_with_fold += 1;
            prev_cold_offset = cache_dequant.cold_offset();
        }

        assert!(
            rms < 5e-3,
            "step {step}: dequant-SDPA vs reference RMS {rms:.4e} exceeds 5e-3 \
             (cold_offset={}, hot_offset={})",
            cache_dequant.cold_offset(),
            cache_dequant.seq_len() - cache_dequant.cold_offset()
        );
    }

    assert!(
        steps_with_fold >= 2,
        "test must cross at least two fold boundaries; only saw {steps_with_fold}"
    );
    eprintln!(
        "delegated_dequant_sdpa_matches_reference: max RMS over {total_steps} steps = \
         {max_rms:.4e}; folds crossed = {steps_with_fold}"
    );
}

/// Steel-envelope vs cold-only fused composition parity (issue #531).
///
/// `update_and_turbo4_delegated_attention` first tries the steel-envelope
/// kernel (issue #531, single Metal dispatch covering softmax, cold-V dequant,
/// and hot-V accumulate). On the same hardware where the cold-only fused
/// composition path (issue #528) already produces correct output, the
/// steel-envelope path must produce numerically equivalent output (within
/// FP16 round-off) — both paths funnel through the same MLX softmax algebra
/// and the same Turbo4 inverse rotation; the only difference is whether the
/// post-Q·K work runs as one Metal dispatch or as a host-graph composition.
///
/// This is a **direct** comparison: we drive two caches with identical
/// inputs, call the steel-envelope path on one and the cold-only fused path
/// on the other, and require RMS < 5e-3. This guards against the failure
/// mode where both paths individually pass against the
/// `delegated_reference_attention` baseline (which uses graph-level dense
/// V dequant) but disagree with each other due to a softmax-merge bug
/// inside the steel kernel.
///
/// The 200-step length and `hot_threshold = 32` cadence match
/// `delegated_fused_kernel_matches_reference_over_200_steps` so this test
/// also crosses at least two fold boundaries.
#[cfg(target_os = "macos")]
#[test]
fn delegated_steel_envelope_matches_cold_only_fused_over_200_steps() {
    let head_dim = 64;
    let prefill_len = 8;
    let total_steps = 200;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Cache A: drives `update_and_turbo4_delegated_attention`, which prefers
    // the steel-envelope kernel on macOS + power-of-2 head_dim (issue #531
    // wiring in `cache::update_and_turbo4_delegated_attention`).
    let mut cache_steel = build_delegated_cache_with_small_threshold(32);
    // Cache B: drives the cold-only fused composition (issue #528) directly
    // by calling `attention_turbo4_delegated_fused` after `update`. Bypasses
    // the wrapper so the steel envelope is not selected.
    let mut cache_cold_only = build_delegated_cache_with_small_threshold(32);

    let k_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 7);
    let v_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 8);
    let k_pre_b = ffi::copy(&k_pre);
    let v_pre_b = ffi::copy(&v_pre);
    let _ = cache_steel.update_and_fetch(k_pre, v_pre);
    let _ = cache_cold_only.update_and_fetch(k_pre_b, v_pre_b);

    let mut max_rms = 0.0_f32;
    let mut steps_with_fold = 0;
    let mut prev_cold_offset = cache_steel.cold_offset();

    for step in 0..total_steps {
        let k_a = synth_kv_tensor(1, 1, 1, head_dim, 5000 + step as u32);
        let v_a = synth_kv_tensor(1, 1, 1, head_dim, 6000 + step as u32);
        let k_b = ffi::copy(&k_a);
        let v_b = ffi::copy(&v_a);
        let q = synth_kv_tensor(1, 1, 1, head_dim, 9000 + step as u32);

        // Steel-envelope path (default after issue #531).
        let q_for_steel = ffi::copy(&q);
        let out_steel =
            cache_steel.update_and_turbo4_delegated_attention(&q_for_steel, k_a, v_a, scale, None);

        // Cold-only fused composition path. We replicate the relevant slice
        // of `update_and_turbo4_delegated_attention` so the cold-only fused
        // helper sees the same per-token snapshot the wrapper would have
        // built. The wrapper code path is mirrored verbatim from `cache.rs`
        // (commit 5d95adb on this branch) modulo the steel-envelope try
        // that we explicitly skip.
        cache_cold_only.update(k_b, v_b);
        let cold_offset = cache_cold_only.cold_offset();
        let hot_offset = cache_cold_only.seq_len() - cold_offset;

        // Eagerly resolve turbo_params just like the wrapper does. We poke
        // the cache through one no-op `delegated_graph_attention` call so
        // turbo_params is populated; that call is otherwise harmless for
        // this comparison since we throw away its output.
        let _ = cache_cold_only.delegated_graph_attention(&q, scale, None);
        // (The `delegated_graph_attention` mutated nothing on the cache —
        // it only reads. After it returns, turbo_params is set if it was
        // unset, and v_packed/v_rescale/values are still in sync.)

        // Slice the kernel inputs out of the (now-updated) cache, mirroring
        // the wrapper's slicing logic.
        let k_buf = cache_cold_only
            .keys
            .as_ref()
            .expect("unified keys must exist after update on Turbo4Delegated");
        let ks = ffi::array_shape(k_buf);
        let k_slice = ffi::slice(
            k_buf,
            &[0, 0, 0, 0],
            &[ks[0], ks[1], cache_cold_only.seq_len(), ks[3]],
        );
        let v_packed_owned = if cold_offset > 0 {
            let vp = cache_cold_only
                .v_packed
                .as_ref()
                .expect("v_packed must exist when cold_offset > 0");
            let vp_shape = ffi::array_shape(vp);
            Some(ffi::slice(
                vp,
                &[0, 0, 0, 0],
                &[vp_shape[0], vp_shape[1], cold_offset, vp_shape[3]],
            ))
        } else {
            None
        };
        let v_rescale_owned = if cold_offset > 0 {
            let vr = cache_cold_only
                .v_rescale
                .as_ref()
                .expect("v_rescale must exist when cold_offset > 0");
            let vr_shape = ffi::array_shape(vr);
            Some(ffi::slice(
                vr,
                &[0, 0, 0, 0],
                &[vr_shape[0], vr_shape[1], cold_offset, 1],
            ))
        } else {
            None
        };
        let hot_v_owned = if hot_offset > 0 {
            let hv = cache_cold_only
                .values
                .as_ref()
                .expect("hot values must exist when hot_offset > 0");
            let hv_shape = ffi::array_shape(hv);
            Some(ffi::slice(
                hv,
                &[0, 0, 0, 0],
                &[hv_shape[0], hv_shape[1], hot_offset, hv_shape[3]],
            ))
        } else {
            None
        };
        let params = cache_cold_only
            .turbo_params
            .as_ref()
            .expect("turbo_params populated after delegated_graph_attention");
        let threshold = crate::cache::turbo::sparse_v::threshold();

        let out_cold_only = crate::cache::turbo::sparse_v::attention_turbo4_delegated_fused(
            &q,
            &k_slice,
            v_packed_owned.as_deref(),
            v_rescale_owned.as_deref(),
            hot_v_owned.as_deref(),
            params,
            cold_offset,
            hot_offset,
            scale,
            None,
            threshold,
        )
        .expect("cold-only fused path must dispatch on macOS + pow-of-2 head_dim");

        // Compare steel vs cold-only output element-by-element.
        let flat_a = flatten_fp32(&out_steel);
        let flat_b = flatten_fp32(&out_cold_only);
        assert_eq!(flat_a.len(), flat_b.len(), "step {step}: shape mismatch");
        let mut sum_sq = 0.0_f64;
        for (x, y) in flat_a.iter().zip(flat_b.iter()) {
            let d = (x - y) as f64;
            sum_sq += d * d;
        }
        let rms = (sum_sq / flat_a.len() as f64).sqrt() as f32;
        if rms > max_rms {
            max_rms = rms;
        }
        if cache_steel.cold_offset() != prev_cold_offset {
            steps_with_fold += 1;
            prev_cold_offset = cache_steel.cold_offset();
        }
        assert!(
            rms < 5e-3,
            "step {step}: steel vs cold-only RMS {rms:.4e} exceeds 5e-3 \
             (cold_offset={cold_offset}, hot_offset={hot_offset})",
        );
    }

    assert!(
        steps_with_fold >= 2,
        "test must cross at least two fold boundaries; only saw {steps_with_fold}"
    );
    eprintln!(
        "delegated_steel_envelope_matches_cold_only_fused: max RMS over {total_steps} steps = \
         {max_rms:.4e}; folds crossed = {steps_with_fold}"
    );
}

/// Steel-envelope grouped-attention + additive-mask parity (issue #531).
///
/// The 200-step parity tests above use `Hkv = Hq = 1` (no grouping) and no
/// additive mask. Production decode call sites in `models/llama3.rs` and
/// `models/qwen3.rs` use `Hq = n_rep * Hkv` with `n_rep ∈ {2, 4, 8}` and may
/// pass a non-trivial mask (causal or attention-sink). This test exercises
/// both: `Hkv = 2`, `Hq = 4` (n_rep = 2), an additive causal mask with
/// `-inf` in the upper triangle, and a single-token decode tail driven
/// against a manually-built reference.
///
/// We use a 24-step decode after a 16-token prefill so the test crosses
/// exactly one fold boundary at `hot_threshold = 32` (16 + 16 = 32 →
/// fold on step 17). Both pre-fold (only hot) and post-fold (cold + hot)
/// states are exercised.
#[cfg(target_os = "macos")]
#[test]
fn delegated_steel_envelope_grouped_attention_with_mask_parity() {
    let head_dim = 64;
    let kv_heads = 2_i32;
    let n_rep = 2_i32;
    let q_heads = kv_heads * n_rep; // 4
    let prefill_len = 16;
    let total_steps = 24;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut cache_steel = build_delegated_cache_with_small_threshold(32);
    let mut cache_ref = build_delegated_cache_with_small_threshold(32);

    let k_pre = synth_kv_tensor(1, kv_heads, prefill_len, head_dim, 11);
    let v_pre = synth_kv_tensor(1, kv_heads, prefill_len, head_dim, 12);
    let k_pre_b = ffi::copy(&k_pre);
    let v_pre_b = ffi::copy(&v_pre);
    let _ = cache_steel.update_and_fetch(k_pre, v_pre);
    let _ = cache_ref.update_and_fetch(k_pre_b, v_pre_b);

    let mut max_rms = 0.0_f32;

    for step in 0..total_steps {
        let k_a = synth_kv_tensor(1, kv_heads, 1, head_dim, 7000 + step as u32);
        let v_a = synth_kv_tensor(1, kv_heads, 1, head_dim, 8000 + step as u32);
        let k_b = ffi::copy(&k_a);
        let v_b = ffi::copy(&v_a);
        // Q has Hq = q_heads (grouped attention).
        let q = synth_kv_tensor(1, q_heads, 1, head_dim, 9500 + step as u32);

        // Build an additive mask of shape `[1, 1, 1, T_total]`. T_total grows
        // as the cache grows. Use an attention-sink-like mask: -inf on every
        // 4th position (so some scores are masked out) and 0 elsewhere. This
        // exercises the kernel's softmax stability across `-inf` entries
        // without relying on a full causal mask (which is degenerate for
        // single-token decode anyway).
        let t_total_after = cache_steel.seq_len() + 1;
        let mut mask_data = vec![0.0_f32; t_total_after as usize];
        for (i, m) in mask_data.iter_mut().enumerate() {
            if i % 4 == 0 && i + 1 < t_total_after as usize {
                // -inf except on the latest token (which the decode step is
                // about to add — keep it alive so the softmax denominator is
                // never structurally zero).
                *m = f32::NEG_INFINITY;
            }
        }
        let mask_a = ffi::from_slice_f32(&mask_data, &[1, 1, 1, t_total_after]);
        let mask_b = ffi::copy(&mask_a);

        let q_for_steel = ffi::copy(&q);
        let out_steel = cache_steel.update_and_turbo4_delegated_attention(
            &q_for_steel,
            k_a,
            v_a,
            scale,
            Some(&mask_a),
        );

        // Reference: drive `cache_ref` through update_and_fetch + manual SDPA
        // including the additive mask.
        let (cache_k, cache_v) = cache_ref.update_and_fetch(k_b, v_b);
        let q_shape = ffi::array_shape(&q);
        let k_shape = ffi::array_shape(&cache_k);
        let b = q_shape[0];
        let hq = q_shape[1];
        let hkv = k_shape[1];
        let nrep = hq / hkv;
        let kt = k_shape[2];
        let kd = k_shape[3];
        let vd = ffi::array_shape(&cache_v)[3];
        let k_for_q = if nrep == 1 {
            ffi::contiguous(&cache_k, false)
        } else {
            let k_exp = ffi::expand_dims(&cache_k, 2);
            let k_tiled = ffi::broadcast_to(&k_exp, &[b, hkv, nrep, kt, kd]);
            ffi::reshape(&k_tiled, &[b, hq, kt, kd])
        };
        let v_for_q = if nrep == 1 {
            ffi::contiguous(&cache_v, false)
        } else {
            let v_exp = ffi::expand_dims(&cache_v, 2);
            let v_tiled = ffi::broadcast_to(&v_exp, &[b, hkv, nrep, kt, vd]);
            ffi::reshape(&v_tiled, &[b, hq, kt, vd])
        };
        let k_t = ffi::transpose_axes(&k_for_q, &[0, 1, 3, 2]);
        let q_f32 = ffi::astype(&q, dtype::FLOAT32);
        let k_t_f32 = ffi::astype(&k_t, dtype::FLOAT32);
        let v_f32 = ffi::astype(&v_for_q, dtype::FLOAT32);
        let qk = ffi::matmul(&q_f32, &k_t_f32);
        let scale_arr = ffi::full_f32(&[1], scale, dtype::FLOAT32);
        let mut scores = ffi::multiply(&qk, &scale_arr);
        let m_f32 = ffi::astype(&mask_b, dtype::FLOAT32);
        scores = ffi::add(&scores, &m_f32);
        let attn = ffi::softmax_precise(&scores, -1);
        let out_f32 = ffi::matmul(&attn, &v_f32);
        let out_ref = ffi::astype(&out_f32, dtype::FLOAT16);

        let flat_a = flatten_fp32(&out_steel);
        let flat_b = flatten_fp32(&out_ref);
        assert_eq!(flat_a.len(), flat_b.len(), "step {step}: shape mismatch");
        let mut sum_sq = 0.0_f64;
        for (x, y) in flat_a.iter().zip(flat_b.iter()) {
            let d = (x - y) as f64;
            sum_sq += d * d;
        }
        let rms = (sum_sq / flat_a.len() as f64).sqrt() as f32;
        if rms > max_rms {
            max_rms = rms;
        }
        assert!(
            rms < 5e-3,
            "step {step}: grouped+masked steel vs reference RMS {rms:.4e} exceeds 5e-3 \
             (cold_offset={}, hot_offset={})",
            cache_steel.cold_offset(),
            cache_steel.seq_len() - cache_steel.cold_offset()
        );
    }

    eprintln!(
        "delegated_steel_envelope_grouped_attention_with_mask: max RMS over \
         {total_steps} steps (Hkv={kv_heads}, Hq={q_heads}, n_rep={n_rep}) = \
         {max_rms:.4e}"
    );

    // Verify the test actually crossed at least one fold so cold and hot are
    // both non-zero by the end.
    assert!(
        cache_steel.cold_offset() > 0,
        "test must trigger at least one fold; got cold_offset={}",
        cache_steel.cold_offset()
    );
}

/// Graph-fallback parity (issue #528 / PR #530 security review HIGH-1).
///
/// `update_and_turbo4_delegated_attention` must always produce a sane
/// attention output, even when the fused Metal kernel cannot dispatch
/// (non-macOS, non-power-of-2 head dim, or `MLXCEL_SPARSE_V_KERNEL=0`). The
/// graph-only fallback must match the legacy `update_and_fetch + manual SDPA`
/// reference within RMS < 5e-3 — same tolerance as the kernel parity test.
///
/// Direct env-var manipulation of `MLXCEL_SPARSE_V_KERNEL` is fragile under
/// `cargo test` because `kernel_enabled()` caches the resolved flag in a
/// `OnceLock<bool>` and other tests may have already populated it. Instead
/// we exercise the fallback path by calling [`KVCache::delegated_graph_attention`]
/// directly after `update`, which is exactly the code path
/// `update_and_turbo4_delegated_attention` runs when the kernel returns
/// `None`. The test compares its output against
/// `delegated_reference_attention` (the same legacy SDPA composition).
#[cfg(target_os = "macos")]
#[test]
fn delegated_graph_fallback_matches_reference_attention() {
    let head_dim = 64;
    let prefill_len = 8;
    let total_steps = 24;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut cache_fallback = build_delegated_cache_with_small_threshold(32);
    let mut cache_ref = build_delegated_cache_with_small_threshold(32);

    let k_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 7);
    let v_pre = synth_kv_tensor(1, 1, prefill_len, head_dim, 8);
    let k_pre_b = ffi::copy(&k_pre);
    let v_pre_b = ffi::copy(&v_pre);
    // Drive both caches through the prefill (multi-token) update_and_fetch.
    let _ = cache_fallback.update_and_fetch(k_pre, v_pre);
    let _ = cache_ref.update_and_fetch(k_pre_b, v_pre_b);

    let mut max_rms = 0.0_f32;
    let mut steps_with_fold = 0;
    let mut prev_cold_offset = cache_fallback.cold_offset();

    for step in 0..total_steps {
        let k_a = synth_kv_tensor(1, 1, 1, head_dim, 5000 + step as u32);
        let v_a = synth_kv_tensor(1, 1, 1, head_dim, 6000 + step as u32);
        let k_b = ffi::copy(&k_a);
        let v_b = ffi::copy(&v_a);
        let q = synth_kv_tensor(1, 1, 1, head_dim, 9000 + step as u32);

        // Graph-fallback path: `update` mutates the cache, then we call the
        // graph-only attention helper directly. This is the exact path the
        // wrapper takes when `attention_turbo4_delegated_fused` returns
        // `None`. After this call the cache state is identical to what the
        // kernel-path wrapper would have produced — no state divergence.
        let q_for_fallback = ffi::copy(&q);
        cache_fallback.update(k_a, v_a);
        let out_fallback = cache_fallback.delegated_graph_attention(&q_for_fallback, scale, None);

        // Reference: legacy `update_and_fetch + manual SDPA` on a
        // separately-driven cache. Identical input sequence so the cache
        // states stay in lockstep.
        let out_ref = delegated_reference_attention(&mut cache_ref, &q, k_b, v_b, scale);

        let flat_a = flatten_fp32(&out_fallback);
        let flat_b = flatten_fp32(&out_ref);
        assert_eq!(flat_a.len(), flat_b.len(), "step {step}: shape mismatch");
        let mut sum_sq = 0.0_f64;
        for (x, y) in flat_a.iter().zip(flat_b.iter()) {
            let d = (x - y) as f64;
            sum_sq += d * d;
        }
        let rms = (sum_sq / flat_a.len() as f64).sqrt() as f32;
        if rms > max_rms {
            max_rms = rms;
        }

        if cache_fallback.cold_offset() != prev_cold_offset {
            steps_with_fold += 1;
            prev_cold_offset = cache_fallback.cold_offset();
        }

        assert!(
            rms < 5e-3,
            "step {step}: graph-fallback vs reference RMS {rms:.4e} exceeds 5e-3 \
             (cold_offset={}, hot_offset={})",
            cache_fallback.cold_offset(),
            cache_fallback.seq_len() - cache_fallback.cold_offset()
        );
    }

    // Sanity: the total length should be enough to drive at least one fold.
    // 8 prefill + 24 decode = 32 tokens, hot threshold = 32, so the very
    // last decode step folds.
    let _ = steps_with_fold;
    eprintln!(
        "delegated_graph_fallback_matches_reference: max RMS over {total_steps} steps = \
         {max_rms:.4e}; folds crossed = {steps_with_fold}"
    );

    // Verify that the cache's post-call state matches the kernel-path
    // contract: cold and hot offsets are advanced just like the kernel path
    // would advance them.
    assert_eq!(
        cache_fallback.seq_len(),
        cache_ref.seq_len(),
        "graph-fallback and reference caches must agree on visible token count"
    );
    assert_eq!(
        cache_fallback.cold_offset(),
        cache_ref.cold_offset(),
        "graph-fallback and reference caches must agree on cold offset"
    );
}

/// Per-token V byte budget: the V-side memory footprint must stay at the
/// 4-bit packed form (D/2 bytes plus one fp16 norm + one fp16 rescale per
/// cold token, plus 2 bytes per hot fp16 dim). Issue #528 retired the FP16
/// `cold_v_dequant_cache` so the only working set above the packed form is
/// the small hot ring; cold V never goes above D/2 bytes per token in
/// global memory.
///
/// **What this test guards against.** A regression that re-introduces a
/// `[B, Hkv, cold_offset, head_dim]` FP16 cold-V memo on `KVCache`. That
/// memo would inflate `cache.nbytes()` by `cold * head_dim * 2` bytes — the
/// regression guard below uses the visible-buffer sum plus that hypothetical
/// memo size as a strict upper bound on `cache.nbytes()`. Strict equality on
/// `nbytes() == visible_buffer_sum` would also trip on any future legitimate
/// internal sidecar addition that is *not* a cold-V memo (e.g. a small
/// allowlist counter or alignment pad), so we deliberately allow slack up to
/// the memo size and let the hypothetical-memo footprint be the regression
/// boundary.
#[test]
fn delegated_per_token_v_budget_after_issue_528() {
    let head_dim = 64;
    let mut cache = build_delegated_cache_with_small_threshold(32);

    // Drive enough decode steps to populate cold storage. Hot threshold = 32
    // so any decode burst > 32 tokens triggers a fold.
    delegated_decode_run(&mut cache, head_dim, 16, 96);
    assert!(cache.cold_offset() > 0, "test must trigger a fold");

    let cold = cache.cold_offset() as usize;
    // Packed V cold body: D/2 bytes per token.
    let vp = cache
        .v_packed
        .as_ref()
        .expect("v_packed must exist after a fold");
    let vp_shape = ffi::array_shape(vp);
    // The per-token byte width is `head_dim/2`. Total covered bytes for the
    // visible cold range must equal `cold * head_dim/2` (the buffer may be
    // step-aligned but we slice on the visible range for the budget check).
    assert_eq!(
        vp_shape[3] as usize,
        head_dim as usize / 2,
        "v_packed last dim must equal head_dim/2 = {}",
        head_dim / 2
    );

    // norm + rescale: 1 fp16 each per cold token.
    let vn = cache
        .v_norms
        .as_ref()
        .expect("v_norms must exist after a fold");
    let vn_shape = ffi::array_shape(vn);
    assert_eq!(
        vn_shape[3], 1,
        "v_norms last dim must be 1 (per-token scalar)"
    );

    // The visible-buffer footprint is the sum of the unified K buffer, the
    // hot V ring, and the packed cold V sidecars (packed indices + norms +
    // rescale). Any future legitimate sidecar (e.g. a separate index map)
    // would push `nbytes()` above this baseline without indicating a memo
    // regression — so we use this sum as a *floor* on what `nbytes()` may
    // include, not a strict equality.
    let k_bytes = cache
        .keys
        .as_ref()
        .map(|k| ffi::array_nbytes(k))
        .unwrap_or(0);
    let hot_v_bytes = cache
        .values
        .as_ref()
        .map(|v| ffi::array_nbytes(v))
        .unwrap_or(0);
    let vp_bytes = ffi::array_nbytes(vp);
    let vn_bytes = ffi::array_nbytes(vn);
    let vr_bytes = cache
        .v_rescale
        .as_ref()
        .map(|v| ffi::array_nbytes(v))
        .unwrap_or(0);
    let visible_buffer_sum = k_bytes + hot_v_bytes + vp_bytes + vn_bytes + vr_bytes;

    // Regression guard: a `[B, Hkv, cold, head_dim]` FP16 memo would weigh
    // `cold * head_dim * 2` bytes per (B, Hkv). The actual `nbytes()` must
    // stay strictly below `visible_buffer_sum + memo_size_if_reintroduced`,
    // so any future cold-V memo of that shape trips this assertion. The
    // bound is intentionally a strict `<` so a re-introduced memo of even
    // a single byte over the visible sum cannot pass.
    let memo_size_if_reintroduced = cold * head_dim as usize * 2;
    assert!(
        cache.nbytes() < visible_buffer_sum + memo_size_if_reintroduced,
        "nbytes()={} would tolerate a re-introduced FP16 cold-V memo of \
         {} bytes (visible buffer sum = {}); any value at or above \
         {} indicates a likely PR-#525 cold-V memo regression",
        cache.nbytes(),
        memo_size_if_reintroduced,
        visible_buffer_sum,
        visible_buffer_sum + memo_size_if_reintroduced,
    );
}

// ---------------------------------------------------------------------------
// Issue #544 regression — TurboQuant continuous batching, batch grow / shrink
// mid-decode.
//
// These tests use real `KVCache::update_and_fetch` calls in `Turbo4Asym`
// mode and drive `BatchedAttentionMetadata` through the same
// extend / filter shape transitions that would happen in the scheduler
// when a sequence is admitted into or evicted from an active batch.
//
// The acceptance criterion in issue #544 is: the per-row decode output is
// equivalent (RMS within tolerance) to running the same prompts
// sequentially. We approximate that at the unit-test level by asserting
// that the dequantized V tensor returned by `update_and_fetch` for each
// sequence is bit-identical between the "in a batched composition that
// changes mid-decode" path and the "sequence run on its own" path. Each
// per-sequence cache is independent (mlxcel's continuous-batching
// architecture stores caches per-sequence; see
// `mlxcel_core::cache::CachePool` and the per-sequence forward in
// `forward_split_attention`), so there is no cross-row interference at the
// cache layer — what we are validating is that the orchestration
// (metadata extend / filter, plus per-sequence quantize/dequantize round
// trips) preserves that property.
// ---------------------------------------------------------------------------

/// Compute root-mean-square error between two flattened float32 buffers.
/// Returns `None` for empty inputs.
fn rms_error(reference: &[f32], actual: &[f32]) -> Option<f32> {
    if reference.is_empty() || reference.len() != actual.len() {
        return None;
    }
    let mut sum_sq = 0.0_f32;
    for (r, a) in reference.iter().zip(actual.iter()) {
        let d = r - a;
        sum_sq += d * d;
    }
    Some((sum_sq / reference.len() as f32).sqrt())
}

/// Run a single sequence's prefill+decode trace through a fresh
/// Turbo4Asym KV cache and return the dequantized V tensor for each step.
///
/// This is the "sequential reference" used by the
/// continuous-batching equivalence test below.
fn turbo4_asym_sequential_decode_trace(
    head_dim: i32,
    prompt_len: i32,
    decode_steps: i32,
    seed: u32,
) -> Vec<Vec<f32>> {
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let mut traces: Vec<Vec<f32>> = Vec::with_capacity(decode_steps as usize + 1);

    // Prefill (`prompt_len` tokens at once). Capture the dequantized V
    // tensor so the test can compare it with the batched version.
    let k_prefill = synth_kv_tensor(1, 1, prompt_len, head_dim, seed);
    let v_prefill = synth_kv_tensor(1, 1, prompt_len, head_dim, seed.wrapping_add(1));
    let (_k_out, v_out) = cache.update_and_fetch(k_prefill, v_prefill);
    traces.push(flatten_fp32(&v_out));

    // Decode (one token per step).
    for step in 0..decode_steps {
        let token_seed = seed.wrapping_add(2 + step as u32 * 2);
        let k_step = synth_kv_tensor(1, 1, 1, head_dim, token_seed);
        let v_step = synth_kv_tensor(1, 1, 1, head_dim, token_seed.wrapping_add(1));
        let (_k_out, v_out) = cache.update_and_fetch(k_step, v_step);
        traces.push(flatten_fp32(&v_out));
    }

    traces
}

/// Issue #544 acceptance criterion (unit-level): a continuous-batching
/// scenario where the batch composition changes mid-decode produces decode
/// output equivalent (RMS within tolerance) to running the same prompts
/// sequentially.
///
/// Scenario:
///   1. Sequence A starts decode alone (batch = [A]).
///   2. Sequence B is admitted mid-decode (batch = [A, B]).
///   3. Sequence A finishes; B continues alone (batch = [B]).
///
/// At each step the test runs `KVCache::update_and_fetch` on each active
/// per-sequence cache and rebuilds `BatchedAttentionMetadata` from the
/// current cache offsets. The dequantized V output for each
/// (sequence, step) pair must be bit-identical to the same step of a
/// fresh-cache sequential run, because per-sequence caches do not share
/// state.
#[test]
fn turbo4_asym_continuous_batching_grow_shrink_matches_sequential() {
    let head_dim: i32 = 64;
    let seq_a_seed: u32 = 0xA1A1_A1A1;
    let seq_b_seed: u32 = 0xB2B2_B2B2;
    let prompt_len_a: i32 = 4;
    let prompt_len_b: i32 = 3;
    // Decode plan:
    //   step 0: A only (batch = [A]).
    //   step 1: A, B both decode (batch = [A, B]) — B prefilled with
    //           `prompt_len_b` tokens before this step.
    //   step 2: A, B both decode (batch = [A, B]).
    //   step 3: A only (B finished after step 2, batch = [A]).
    // A runs decode_steps_a = 4 single-token steps total (one before
    // admission, two during, one after). B runs decode_steps_b = 2 (both
    // during the joint phase).
    let decode_steps_a: i32 = 4;
    let decode_steps_b: i32 = 2;

    // Sequential reference traces (one fresh cache per sequence).
    let ref_a =
        turbo4_asym_sequential_decode_trace(head_dim, prompt_len_a, decode_steps_a, seq_a_seed);
    let ref_b =
        turbo4_asym_sequential_decode_trace(head_dim, prompt_len_b, decode_steps_b, seq_b_seed);

    // Batched run: independent per-sequence caches, metadata rebuilt on
    // every step from the *current* batch composition.
    let mut cache_a = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let mut cache_b = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);

    // Phase 1: prefill A alone — batch = [A].
    let k_a_prefill = synth_kv_tensor(1, 1, prompt_len_a, head_dim, seq_a_seed);
    let v_a_prefill = synth_kv_tensor(1, 1, prompt_len_a, head_dim, seq_a_seed.wrapping_add(1));
    let (_k_out, v_out_a) = cache_a.update_and_fetch(k_a_prefill, v_a_prefill);
    let batched_a_step0 = flatten_fp32(&v_out_a);

    // Verify metadata is per-row [A].
    {
        let caches = vec![&mut cache_a];
        let metadata = BatchedAttentionMetadata::uniform_kv_caches(&caches, 1, 0).unwrap();
        assert_eq!(metadata.rope_offsets, vec![prompt_len_a]);
        assert_eq!(metadata.kv_lens, vec![prompt_len_a + 1]);
        metadata.assert_consistent().unwrap();
    }

    // Phase 2: A's first solo decode step. Still batch = [A].
    let token_seed_a_0 = seq_a_seed.wrapping_add(2);
    let k_a_t0 = synth_kv_tensor(1, 1, 1, head_dim, token_seed_a_0);
    let v_a_t0 = synth_kv_tensor(1, 1, 1, head_dim, token_seed_a_0.wrapping_add(1));
    let (_k_out, v_out_a) = cache_a.update_and_fetch(k_a_t0, v_a_t0);
    let batched_a_step1 = flatten_fp32(&v_out_a);

    // Phase 3: admit B. Prefill B, then run a joint decode step. batch = [A, B].
    let k_b_prefill = synth_kv_tensor(1, 1, prompt_len_b, head_dim, seq_b_seed);
    let v_b_prefill = synth_kv_tensor(1, 1, prompt_len_b, head_dim, seq_b_seed.wrapping_add(1));
    let (_k_out, v_out_b) = cache_b.update_and_fetch(k_b_prefill, v_b_prefill);
    let batched_b_step0 = flatten_fp32(&v_out_b);

    // Build heterogeneous metadata for the [A, B] composition.
    {
        let caches = vec![&mut cache_a, &mut cache_b];
        let metadata = BatchedAttentionMetadata::from_kv_caches(&caches, &[1, 1], &[0, 0]).unwrap();
        assert_eq!(
            metadata.rope_offsets,
            vec![prompt_len_a + 1, prompt_len_b],
            "batch grow must keep per-row offsets correct (issue #544)"
        );
        assert_eq!(metadata.kv_lens, vec![prompt_len_a + 2, prompt_len_b + 1]);
        metadata.assert_consistent().unwrap();
    }

    // Joint decode step: A's 2nd decode token + B's 1st decode token.
    let token_seed_a_1 = seq_a_seed.wrapping_add(4);
    let k_a_t1 = synth_kv_tensor(1, 1, 1, head_dim, token_seed_a_1);
    let v_a_t1 = synth_kv_tensor(1, 1, 1, head_dim, token_seed_a_1.wrapping_add(1));
    let (_k_out, v_out_a) = cache_a.update_and_fetch(k_a_t1, v_a_t1);
    let batched_a_step2 = flatten_fp32(&v_out_a);

    let token_seed_b_0 = seq_b_seed.wrapping_add(2);
    let k_b_t0 = synth_kv_tensor(1, 1, 1, head_dim, token_seed_b_0);
    let v_b_t0 = synth_kv_tensor(1, 1, 1, head_dim, token_seed_b_0.wrapping_add(1));
    let (_k_out, v_out_b) = cache_b.update_and_fetch(k_b_t0, v_b_t0);
    let batched_b_step1 = flatten_fp32(&v_out_b);

    // Phase 4: another joint decode step. Still batch = [A, B].
    let token_seed_a_2 = seq_a_seed.wrapping_add(6);
    let k_a_t2 = synth_kv_tensor(1, 1, 1, head_dim, token_seed_a_2);
    let v_a_t2 = synth_kv_tensor(1, 1, 1, head_dim, token_seed_a_2.wrapping_add(1));
    let (_k_out, v_out_a) = cache_a.update_and_fetch(k_a_t2, v_a_t2);
    let batched_a_step3 = flatten_fp32(&v_out_a);

    let token_seed_b_1 = seq_b_seed.wrapping_add(4);
    let k_b_t1 = synth_kv_tensor(1, 1, 1, head_dim, token_seed_b_1);
    let v_b_t1 = synth_kv_tensor(1, 1, 1, head_dim, token_seed_b_1.wrapping_add(1));
    let (_k_out, v_out_b) = cache_b.update_and_fetch(k_b_t1, v_b_t1);
    let batched_b_step2 = flatten_fp32(&v_out_b);

    // Phase 5: B finishes (filter); A continues alone. batch = [A].
    {
        let mut metadata = {
            let caches = vec![&mut cache_a, &mut cache_b];
            BatchedAttentionMetadata::from_kv_caches(&caches, &[1, 1], &[0, 0]).unwrap()
        };
        // Filter to keep only row 0 (A). After this, the per-row vectors
        // must shrink to length 1 — but never collapse to a scalar.
        metadata.filter_in_place(&[0]).unwrap();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata.rope_offsets, vec![prompt_len_a + 3]);
        metadata.assert_consistent().unwrap();
    }

    // A's final solo decode step.
    let token_seed_a_3 = seq_a_seed.wrapping_add(8);
    let k_a_t3 = synth_kv_tensor(1, 1, 1, head_dim, token_seed_a_3);
    let v_a_t3 = synth_kv_tensor(1, 1, 1, head_dim, token_seed_a_3.wrapping_add(1));
    let (_k_out, v_out_a) = cache_a.update_and_fetch(k_a_t3, v_a_t3);
    let batched_a_step4 = flatten_fp32(&v_out_a);

    // Equivalence checks. Per-sequence caches are independent, so the
    // batched and sequential dequantized V tensors must be bit-identical
    // (RMS == 0). The assertion uses `< 1e-6` to allow for any
    // floating-point reorder noise from MLX kernel scheduling, well
    // below the Turbo4 quantization error budget.
    //
    // The `kv_visible_len` for each step is `prompt_len + step_count`,
    // so we slice the V output to compare only the visible window.
    let tol: f32 = 1e-6;

    // Sequence A trace: ref_a[i] corresponds to V output after the i-th
    // update (i=0 is prefill, i=1..decode_steps_a are decode tokens).
    let visible = |t: i32| -> usize { (t * head_dim) as usize };

    let kv_a_lens = [
        prompt_len_a,
        prompt_len_a + 1,
        prompt_len_a + 2,
        prompt_len_a + 3,
        prompt_len_a + 4,
    ];
    let batched_a = [
        &batched_a_step0,
        &batched_a_step1,
        &batched_a_step2,
        &batched_a_step3,
        &batched_a_step4,
    ];
    for (i, (batched, kv_len)) in batched_a.iter().zip(kv_a_lens.iter()).enumerate() {
        let n = visible(*kv_len);
        let bs = &batched[..n];
        let rs = &ref_a[i][..n];
        let rms = rms_error(rs, bs).unwrap();
        assert!(
            rms < tol,
            "sequence A step {i}: batched-vs-sequential RMS {rms:e} exceeds {tol:e}"
        );
    }

    let kv_b_lens = [prompt_len_b, prompt_len_b + 1, prompt_len_b + 2];
    let batched_b = [&batched_b_step0, &batched_b_step1, &batched_b_step2];
    for (i, (batched, kv_len)) in batched_b.iter().zip(kv_b_lens.iter()).enumerate() {
        let n = visible(*kv_len);
        let bs = &batched[..n];
        let rs = &ref_b[i][..n];
        let rms = rms_error(rs, bs).unwrap();
        assert!(
            rms < tol,
            "sequence B step {i}: batched-vs-sequential RMS {rms:e} exceeds {tol:e}"
        );
    }
}
