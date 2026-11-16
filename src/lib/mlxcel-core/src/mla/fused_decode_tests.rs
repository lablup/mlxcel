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

//! Parity tests for the fused absorbed decode kernel (issue #3121).
//!
//! The kernel is a second implementation of the composed single-query path, so
//! it is held to the same host f64 reference ([`MlaFixture::decompressed_reference`])
//! and every case also asserts, through the dispatch counters, that the kernel
//! and not the composed fallback produced the number. Skipped where the kernel
//! has no port (off Metal), since there is nothing to compare.

use crate::cache::KVCache;
use crate::dtype;
use crate::ffi;
use crate::mla::absorb::MlaAbsorbedProjections;
use crate::mla::cache::MlaLatentCache;
use crate::mla::decode::{FUSED_DECODE_MAX_KEYS, absorbed_decode};
use crate::mla::testkit::{MlaFixture, max_rel_error, serial, to_vec_f32};
use crate::mla::{MlaGeometry, fused_decode_enabled, stats};

/// The smallest geometry the kernel takes: both latent widths multiples of 32.
const FUSED: MlaGeometry = MlaGeometry {
    num_heads: 4,
    kv_lora_rank: 64,
    qk_nope_head_dim: 16,
    qk_rope_head_dim: 32,
    v_head_dim: 12,
};

const F32_TOL: f32 = 5e-5;
const F16_TOL: f32 = 6e-3;

/// Whether the kernel runs here. On Apple Silicon it must, unless the
/// operator turned it off: a test that skips there would pass while the fused
/// path never ran, the #899 shape. Elsewhere (no Metal port) the tests skip.
fn kernel_live() -> bool {
    let switched_off = std::env::var_os(crate::mla::FUSED_DECODE_ENV).is_some();
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) && !switched_off {
        assert!(
            fused_decode_enabled(),
            "the fused MLA decode kernel is not live on Apple Silicon"
        );
    }
    fused_decode_enabled()
}

fn decode_once(fx: &MlaFixture, dt: i32) -> Vec<f32> {
    let proj = MlaAbsorbedProjections::from_dense(&fx.kv_b_array(dt), fx.geometry).unwrap();
    let out = absorbed_decode(
        &fx.q_nope_array(dt),
        &fx.q_pe_array(dt),
        &fx.ckv_array(dt),
        &fx.kpe_array(dt),
        &proj,
        fx.scale,
        None,
    );
    to_vec_f32(&out)
}

#[test]
fn fused_kernel_matches_the_decompressed_reference_across_windows() {
    let _guard = serial();
    if !kernel_live() {
        return;
    }
    // 1 and 5 leave the second half of the split empty, 31/33 straddle one
    // simdgroup stride, 257 is uneven across both halves.
    for (i, &kv_len) in [1usize, 5, 31, 33, 64, 257].iter().enumerate() {
        for (dt, tol) in [(dtype::FLOAT32, F32_TOL), (dtype::FLOAT16, F16_TOL)] {
            let _ = stats::take();
            let fx = MlaFixture::new(FUSED, 2, 1, kv_len, 0x3121 + i as u64);
            let got = decode_once(&fx, dt);
            let counts = stats::take();
            assert_eq!(
                counts.absorbed_fused,
                1,
                "kv_len {kv_len}: {}",
                counts.summary()
            );
            assert_eq!(
                counts.absorbed_composed,
                0,
                "kv_len {kv_len}: {}",
                counts.summary()
            );
            let err = max_rel_error(&got, &fx.decompressed_reference(false));
            assert!(
                err < tol,
                "kv_len {kv_len} dtype {dt}: fused decode drifted by {err}"
            );
        }
    }
}

#[test]
fn fused_kernel_reads_a_strided_cache_window() {
    let _guard = serial();
    if !kernel_live() {
        return;
    }
    // Through the latent cache: the live window is a slice of the cache's
    // preallocated buffer, so with two batch rows it is not row-contiguous and
    // the kernel has to follow its strides rather than assume a packed tensor.
    let fx = MlaFixture::new(FUSED, 2, 1, 41, 0x51DE);
    let dt = dtype::FLOAT32;
    let proj = MlaAbsorbedProjections::from_dense(&fx.kv_b_array(dt), fx.geometry).unwrap();
    let ckv = fx.ckv_array(dt);
    let kpe = fx.kpe_array(dt);
    let (r, p) = (FUSED.kv_lora_rank as i32, FUSED.qk_rope_head_dim as i32);
    let mut inner = KVCache::new();
    let mut cache = MlaLatentCache::wrap(&mut inner, fx.geometry).unwrap();
    let _ = cache.update_and_fetch(
        ffi::slice(&ckv, &[0, 0, 0, 0], &[2, 1, 40, r]),
        ffi::slice(&kpe, &[0, 0, 0, 0], &[2, 1, 40, p]),
    );
    let _ = stats::take();
    let out = cache.attend(
        &fx.q_nope_array(dt),
        &fx.q_pe_array(dt),
        ffi::slice(&ckv, &[0, 0, 40, 0], &[2, 1, 41, r]),
        ffi::slice(&kpe, &[0, 0, 40, 0], &[2, 1, 41, p]),
        &proj,
        fx.scale,
        None,
    );
    let counts = stats::take();
    assert_eq!(counts.absorbed_fused, 1, "{}", counts.summary());
    let err = max_rel_error(&to_vec_f32(&out), &fx.decompressed_reference(false));
    assert!(
        err < F32_TOL,
        "fused decode over a cache window drifted by {err}"
    );
}

#[test]
fn windows_past_the_fused_limit_and_unfit_shapes_keep_the_composed_path() {
    let _guard = serial();
    if !kernel_live() {
        return;
    }
    // Longer than the kernel's measured win: composed.
    let fx = MlaFixture::new(FUSED, 1, 1, FUSED_DECODE_MAX_KEYS as usize + 1, 0xB16);
    let _ = stats::take();
    let _ = decode_once(&fx, dtype::FLOAT16);
    let counts = stats::take();
    assert_eq!(counts.absorbed_composed, 1, "{}", counts.summary());
    assert_eq!(counts.absorbed_fused, 0, "{}", counts.summary());

    // A rope width that is not a multiple of 32: composed, still correct.
    let tiny = crate::mla::testkit::TINY;
    let fx = MlaFixture::new(tiny, 1, 1, 9, 0xB17);
    let got = decode_once(&fx, dtype::FLOAT32);
    let counts = stats::take();
    assert_eq!(counts.absorbed_composed, 1, "{}", counts.summary());
    assert_eq!(counts.absorbed_fused, 0, "{}", counts.summary());
    assert!(max_rel_error(&got, &fx.decompressed_reference(false)) < 2e-5);
}
