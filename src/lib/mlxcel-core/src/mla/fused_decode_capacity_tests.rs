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

//! The fused absorbed-MLA decode kernel is only taken where it evaluates.
//!
//! The partial pass always launches 1024 threads, and Metal fixes each
//! pipeline's thread limit per chip from the kernel's register use. At
//! DeepSeek-V2-Lite's geometry (16 heads, latent rank 512, rope width 64) an
//! M1 Ultra caps the pipeline at 768, so before the per-geometry probe the
//! gate took the kernel and the step failed at eval with "Thread group size
//! (1024) is greater than the maximum allowed threads per threadgroup (768)".
//! These tests hold the contract on every host: when a gate returns an array,
//! that array evaluates; when it declines, the probe said the kernel does not
//! run there. On a chip where the pipeline fits they check the kernel at the
//! production geometry instead.

use cxx::UniquePtr;

use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::mla::decode::{fused_absorbed_decode, fused_absorbed_decode_quantized};
use crate::mla::quant_latent::{LatentWindow, quantize_latent_rows};
use crate::mla::testkit::{Rng, serial, to_vec_f32};
use crate::mla::{DEFAULT_PROBE_RANK, DEFAULT_PROBE_ROPE, fused_decode_runs_by_default};

/// DeepSeek-V2-Lite's decode geometry.
const HEADS: i32 = 16;
const RANK: i32 = 512;
const ROPE: i32 = 64;
const KEYS: i32 = 24;

fn random(rng: &mut Rng, shape: &[i32], dt: i32) -> UniquePtr<MlxArray> {
    let n = shape.iter().product::<i32>() as usize;
    ffi::astype(&ffi::from_slice_f32(&rng.vec(n, 0.5), shape), dt)
}

fn assert_evaluates(out: &MlxArray, what: &str) {
    ffi::try_eval(out)
        .unwrap_or_else(|error| panic!("{what}: taken but does not evaluate: {error}"));
    let values = to_vec_f32(out);
    assert!(
        values.iter().all(|v| v.is_finite()),
        "{what}: non-finite output"
    );
}

#[test]
fn the_dense_gate_takes_the_kernel_only_where_it_evaluates() {
    let _guard = serial();
    for dt in [dtype::FLOAT16, dtype::BFLOAT16, dtype::FLOAT32] {
        let mut rng = Rng::new(0x3121 ^ dt as u64);
        let q = random(&mut rng, &[1, HEADS, 1, RANK], dt);
        let q_pe = random(&mut rng, &[1, HEADS, 1, ROPE], dt);
        let ckv = random(&mut rng, &[1, 1, KEYS, RANK], dt);
        let kpe = random(&mut rng, &[1, 1, KEYS, ROPE], dt);
        let supported = ffi::mla_absorbed_decode_fused_supported(dt, RANK, ROPE);
        match fused_absorbed_decode(&q, &q_pe, &ckv, &kpe, 0.07) {
            Some(out) => {
                assert_eq!(ffi::array_shape(&out), vec![1, HEADS, 1, RANK]);
                assert_evaluates(&out, &format!("dense dtype {dt}"));
                assert!(supported, "dtype {dt}: taken although the probe failed");
            }
            None => assert!(
                !supported || !crate::mla::fused_decode_enabled(),
                "dtype {dt}: declined although the kernel evaluates here"
            ),
        }
    }
}

#[test]
fn the_quantized_gate_takes_the_kernel_only_where_it_evaluates() {
    let _guard = serial();
    let mut rng = Rng::new(0x3156);
    let ckv = random(&mut rng, &[1, 1, KEYS, RANK], dtype::FLOAT32);
    let kpe = random(&mut rng, &[1, 1, KEYS, ROPE], dtype::FLOAT16);
    let LatentWindow::Quantized {
        ckv_q,
        scales,
        biases,
        kpe,
        bits,
        group_size,
    } = LatentWindow::from_rows(quantize_latent_rows(&ckv, kpe, 8), 8)
    else {
        unreachable!("from_rows builds a quantized window");
    };
    for dt in [dtype::FLOAT16, dtype::BFLOAT16] {
        let q = random(&mut rng, &[1, HEADS, 1, RANK], dt);
        let q_pe = random(&mut rng, &[1, HEADS, 1, ROPE], dt);
        let supported = ffi::mla_absorbed_decode_fused_quantized_supported(
            dt,
            ffi::array_dtype(&kpe),
            ffi::array_dtype(&scales),
            RANK,
            ROPE,
            group_size,
            bits,
        );
        match fused_absorbed_decode_quantized(
            &q, &q_pe, &ckv_q, &scales, &biases, &kpe, bits, group_size, 0.07,
        ) {
            Some(out) => {
                assert_evaluates(&out, &format!("quantized dtype {dt}"));
                assert!(supported, "dtype {dt}: taken although the probe failed");
            }
            None => assert!(
                !supported || !crate::mla::fused_decode_enabled(),
                "dtype {dt}: declined although the kernel evaluates here"
            ),
        }
    }
}

#[test]
fn a_failed_probe_leaves_the_stream_usable_and_is_remembered() {
    let _guard = serial();
    let first = ffi::mla_absorbed_decode_fused_supported(dtype::FLOAT16, RANK, ROPE);
    // Whatever the probe found, the next graph on the default stream runs.
    let x = ffi::from_slice_f32(&[1.0, 2.0, 3.0], &[3]);
    let y = ffi::add(&x, &x);
    ffi::try_eval(&y).expect("an eval after the probe failed");
    assert_eq!(to_vec_f32(&y), vec![2.0, 4.0, 6.0]);
    // The answer is cached per geometry, and the layout default reads the same one.
    assert_eq!(
        ffi::mla_absorbed_decode_fused_supported(dtype::FLOAT16, RANK, ROPE),
        first
    );
    assert_eq!((DEFAULT_PROBE_RANK, DEFAULT_PROBE_ROPE), (RANK, ROPE));
    assert_eq!(
        fused_decode_runs_by_default(),
        crate::mla::fused_decode_enabled() && first
    );
}

#[test]
fn geometries_outside_the_kernel_contract_are_unsupported_without_a_launch() {
    let _guard = serial();
    for (rank, rope) in [(0, 64), (500, 64), (512, 0), (512, 48), (2048, 64)] {
        assert!(
            !ffi::mla_absorbed_decode_fused_supported(dtype::FLOAT16, rank, rope),
            "rank {rank}, rope {rope}"
        );
    }
    assert!(!ffi::mla_absorbed_decode_fused_supported(
        dtype::INT32,
        RANK,
        ROPE
    ));
}
