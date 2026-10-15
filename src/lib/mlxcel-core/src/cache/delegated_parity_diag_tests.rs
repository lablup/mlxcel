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

//! Tests for the #1712 delegated parity diagnostics.
//!
//! The parity tests that call `diagnose_delegated_mismatch` are macOS-only and
//! only reach it on failure, so without these the report would first run on
//! the day it is needed. These run on every backend: off macOS the Metal
//! launchers report "not dispatched" and the graph-level parts still execute.

use super::delegated_parity_diag::{
    DelegatedKernelInputs, MismatchContext, diagnose_delegated_mismatch, flatten_fp32, parity_diff,
    rms_of,
};
use super::{KVCache, KVCacheMode};
use crate::ffi;

fn synth(h: i32, t: i32, d: i32, seed: u32) -> cxx::UniquePtr<ffi::MlxArray> {
    let mut state = seed;
    let data: Vec<f32> = (0..(h * t * d))
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 1) as f32 / (i32::MAX as f32)
        })
        .collect();
    ffi::from_slice_f32(&data, &[1, h, t, d])
}

fn delegated_cache_after_one_decode_step(head_dim: i32) -> KVCache {
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Delegated);
    cache.set_hot_threshold(32);
    let _ = cache.update_and_fetch(synth(1, 8, head_dim, 7), synth(1, 8, head_dim, 8));
    cache.update(synth(1, 1, head_dim, 5000), synth(1, 1, head_dim, 6000));
    cache
}

#[test]
fn parity_diff_locates_the_largest_deviation() {
    let lhs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0];
    let rhs = [0.0; 8];
    let diff = parity_diff(&lhs, &rhs, 4);
    assert_eq!(diff.max_idx, 5);
    assert_eq!(diff.max_abs, 0.5);
    assert!((diff.rms - (0.25_f32 / 8.0).sqrt()).abs() < 1e-7);
    assert!(!diff.non_finite);
    let shown = diff.to_string();
    assert!(shown.contains("row 1, d 1"), "{shown}");

    let nan = parity_diff(&[f32::NAN, 0.0], &[0.0, 0.0], 2);
    assert!(nan.non_finite);
    assert!(nan.to_string().contains("NON-FINITE"));
    assert_eq!(rms_of(&[3.0, 4.0]), (12.5_f32).sqrt());
}

#[test]
fn diagnose_reports_state_reference_signatures_and_reruns() {
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut cache_a = delegated_cache_after_one_decode_step(head_dim);
    let mut cache_b = delegated_cache_after_one_decode_step(head_dim);
    let q = synth(1, 1, head_dim, 9000);

    let inputs = DelegatedKernelInputs::from_cache(&cache_a);
    assert_eq!(inputs.cold_offset + inputs.hot_offset, cache_a.seq_len());
    assert!(inputs.cold_offset > 0, "the prefill must fold into cold");
    assert!(
        inputs.hot_offset > 0,
        "the decode token must sit in the hot ring"
    );

    let reference = flatten_fp32(&cache_b.delegated_graph_attention(&q, scale, None));
    let zeros = vec![0.0_f32; reference.len()];
    let observed: [(&str, &[f32]); 2] = [("correct", &reference), ("zeroed", &zeros)];
    let report = diagnose_delegated_mismatch(
        &MismatchContext {
            step: 0,
            head_dim: head_dim as usize,
            q: &q,
            scale,
            mask: None,
            observed: &observed,
        },
        &mut cache_a,
        &mut cache_b,
    );

    // Identical inputs leave identical state.
    assert!(
        report.contains("K=0.000e0 v_packed=0.000e0 v_rescale=0.000e0 hot_v=0.000e0"),
        "{report}"
    );
    // The zeroed output sits exactly rms(reference) from the reference.
    let zero_line = format!(
        "observed zeroed vs graph reference: rms={:.4e}",
        rms_of(&reference)
    );
    assert!(report.contains(&zero_line), "{report}");
    assert!(
        report.contains("observed correct vs graph reference: rms=0.0000e0"),
        "{report}"
    );
    assert!(report.contains("signature hot V dropped"), "{report}");
    assert!(report.contains("signature cold V dropped"), "{report}");
    // Every route is either re-run or reported as not dispatched, on both caches.
    for path in ["dequant-sdpa", "steel", "cold-only"] {
        for side in ["a", "b"] {
            assert!(
                report.contains(&format!("rerun {path} on cache {side}:")),
                "missing rerun {path}/{side}:\n{report}"
            );
        }
    }
    eprintln!("{report}");
}
