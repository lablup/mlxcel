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

//! Parity tests for the fused decode kernel's quantized variant (issue #3156).
//!
//! The reference is the composed quantized path over the same
//! `(ckv_q, scales, biases, kpe)`: both read identical stored values, so the
//! codec's own error cancels and what is left is arithmetic order. The two are
//! not bitwise equal. The kernel sums each score and each value column in f32
//! across 32 lanes and 32 simdgroups in two halves, where the composed path
//! runs `quantized_matmul` and computes the rope score in the query's dtype
//! (fp16 for an fp16 model) before widening it. [`PARITY`] states the bound
//! that leaves per query dtype.
//!
//! Skipped where the kernel has no port (off Metal).

use cxx::UniquePtr;

use crate::cache::{KVCache, KVCacheMode};
use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::mla::cache::MlaLatentCache;
use crate::mla::decode::{FUSED_DECODE_QUANTIZED_MAX_KEYS, fused_absorbed_decode_quantized};
use crate::mla::quant_latent::{LatentWindow, composed_quantized_attention, quantize_latent_rows};
use crate::mla::testkit::{MlaFixture, Rng, max_rel_error, serial, to_vec_f32};
use crate::mla::{MlaGeometry, fused_decode_enabled, stats};

/// Per query dtype: the bound on fused against composed, relative to the
/// largest output element, and what sets it.
///
/// With f32 queries both paths compute and return f32, so the bound is the
/// summation-order difference alone. With fp16 or bf16 queries each path
/// rounds its f32 result to the query dtype, and results that differ by less
/// than that land on neighbouring representable values: one unit in the last
/// place is up to 2^-10 of the value in fp16 and 2^-7 in bf16. The bounds are
/// two such units. Measured worst case on an M3 Ultra over the grid below:
/// 5.1e-7 (f32), 5.6e-4 (fp16), 6.25e-3 (bf16).
const PARITY: [(i32, f32); 3] = [
    (dtype::FLOAT32, 1e-5),
    (dtype::FLOAT16, 2e-3),
    (dtype::BFLOAT16, 1.6e-2),
];

fn parity_tol(q_dt: i32) -> f32 {
    PARITY.iter().find(|(dt, _)| *dt == q_dt).unwrap().1
}

/// Two affine groups, four elements per lane.
const WIDE: MlaGeometry = MlaGeometry {
    num_heads: 4,
    kv_lora_rank: 128,
    qk_nope_head_dim: 16,
    qk_rope_head_dim: 32,
    v_head_dim: 12,
};

/// One group, two elements per lane: a lane's slice is half a packed word.
const NARROW: MlaGeometry = MlaGeometry {
    num_heads: 4,
    kv_lora_rank: 64,
    qk_nope_head_dim: 16,
    qk_rope_head_dim: 32,
    v_head_dim: 12,
};

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

/// A quantized window plus one decode step's queries.
struct Case {
    q: UniquePtr<MlxArray>,
    q_pe: UniquePtr<MlxArray>,
    ckv_q: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: UniquePtr<MlxArray>,
    kpe: UniquePtr<MlxArray>,
    bits: i32,
    group_size: i32,
    scale: f32,
}

impl Case {
    /// `zero_row` clears one latent row before quantizing, so its groups have
    /// zero range.
    fn new(
        g: MlaGeometry,
        batch: usize,
        keys: usize,
        q_dt: i32,
        bits: i32,
        seed: u64,
        zero_row: Option<usize>,
    ) -> Self {
        let fx = MlaFixture::new(g, batch, 1, keys, seed);
        let (b, h, r) = (batch as i32, g.num_heads as i32, g.kv_lora_rank);
        let mut ckv = fx.ckv.clone();
        if let Some(row) = zero_row {
            ckv[row * r..(row + 1) * r].fill(0.0);
        }
        let ckv = ffi::from_slice_f32(&ckv, &[b, 1, keys as i32, r as i32]);
        let rows = quantize_latent_rows(&ckv, fx.kpe_array(dtype::FLOAT16), bits);
        let LatentWindow::Quantized {
            ckv_q,
            scales,
            biases,
            kpe,
            bits,
            group_size,
        } = LatentWindow::from_rows(rows, bits)
        else {
            unreachable!("from_rows builds a quantized window");
        };
        let q = Rng::new(seed ^ 0x51).vec((b * h) as usize * r, 0.5);
        Self {
            q: ffi::astype(&ffi::from_slice_f32(&q, &[b, h, 1, r as i32]), q_dt),
            q_pe: fx.q_pe_array(q_dt),
            ckv_q,
            scales,
            biases,
            kpe,
            bits,
            group_size,
            scale: fx.scale,
        }
    }

    fn fused(&self) -> Option<UniquePtr<MlxArray>> {
        fused_absorbed_decode_quantized(
            &self.q,
            &self.q_pe,
            &self.ckv_q,
            &self.scales,
            &self.biases,
            &self.kpe,
            self.bits,
            self.group_size,
            self.scale,
        )
    }

    fn composed(&self) -> UniquePtr<MlxArray> {
        composed_quantized_attention(
            &self.q,
            &self.q_pe,
            &self.ckv_q,
            &self.scales,
            &self.biases,
            &self.kpe,
            self.bits,
            self.group_size,
            self.scale,
            None,
        )
    }

    /// The kernel through the bridge, with none of the Rust gate's limits.
    fn kernel(&self) -> UniquePtr<MlxArray> {
        ffi::mla_absorbed_decode_fused_quantized(
            &self.q,
            &self.q_pe,
            &self.ckv_q,
            &self.scales,
            &self.biases,
            &self.kpe,
            self.scale,
            self.group_size,
            self.bits,
        )
        .expect("the launcher refused a window inside its contract")
    }

    /// Run both paths and return the kernel output's error against the
    /// composed one. A window the decode path gives the kernel goes through
    /// the gate, asserting it took the call; a longer one (the kernel is
    /// still correct there, only slower than composed) goes to the bridge.
    fn parity(&self, what: &str) -> f32 {
        let keys = ffi::array_shape(&self.ckv_q)[2];
        let _ = stats::take();
        let fused = if keys <= FUSED_DECODE_QUANTIZED_MAX_KEYS {
            let out = self.fused().unwrap_or_else(|| panic!("{what}: declined"));
            let counts = stats::take();
            assert_eq!(counts.absorbed_fused, 1, "{what}: {}", counts.summary());
            assert_eq!(counts.total(), 1, "{what}: {}", counts.summary());
            out
        } else {
            assert!(self.fused().is_none(), "{what}: past the window limit");
            self.kernel()
        };
        assert_eq!(ffi::array_dtype(&fused), ffi::array_dtype(&self.q));
        assert_eq!(ffi::array_shape(&fused), ffi::array_shape(&self.q));
        let want = to_vec_f32(&self.composed());
        let _ = stats::take();
        max_rel_error(&to_vec_f32(&fused), &want)
    }
}

#[test]
fn fused_quantized_matches_composed_quantized_across_windows() {
    let _guard = serial();
    if !kernel_live() {
        return;
    }
    // 1 leaves the second half empty, 255 is uneven across halves and
    // simdgroup strides, 2048 is the longest window the decode path gives
    // the kernel, and 8192 is past that limit.
    assert_eq!(
        FUSED_DECODE_QUANTIZED_MAX_KEYS, 2048,
        "the grid's third window"
    );
    for (q_dt, tol) in PARITY {
        let mut worst = 0.0f32;
        for (i, &keys) in [1usize, 255, 2048, 8192].iter().enumerate() {
            for g in [WIDE, NARROW] {
                let case = Case::new(g, 1, keys, q_dt, 8, 0x3156 + i as u64, None);
                let what = format!("rank {} keys {keys} dtype {q_dt}", g.kv_lora_rank);
                let err = case.parity(&what);
                assert!(err < tol, "{what}: fused off composed by {err}");
                worst = worst.max(err);
            }
        }
        println!("dtype {q_dt}: fused quantized off composed quantized by at most {worst:e}");
    }
}

#[test]
fn fused_quantized_reads_sidecars_of_either_half_dtype_and_zero_groups() {
    let _guard = serial();
    if !kernel_live() {
        return;
    }
    // bf16 scales and biases under fp16 queries, with row 3 all zero.
    let mut case = Case::new(WIDE, 2, 37, dtype::FLOAT16, 8, 0x5CA1, Some(3));
    case.scales = ffi::astype(&case.scales, dtype::BFLOAT16);
    case.biases = ffi::astype(&case.biases, dtype::BFLOAT16);
    let err = case.parity("bf16 sidecars");
    assert!(
        err < parity_tol(dtype::FLOAT16),
        "bf16 sidecars: off composed by {err}"
    );

    // A one-key window whose only row is zero: every group has scale 0 and
    // bias 0, the softmax weight is 1, and the output is exactly zero.
    let case = Case::new(NARROW, 1, 1, dtype::FLOAT16, 8, 0x5CA2, Some(0));
    let out = to_vec_f32(&case.fused().expect("one zero key"));
    let _ = stats::take();
    assert!(
        out.iter().all(|v| *v == 0.0),
        "a zero latent read back {out:?}"
    );
}

#[test]
fn fused_quantized_follows_a_strided_cache_window_through_the_real_route() {
    let _guard = serial();
    if !kernel_live() {
        return;
    }
    // 300 rows cross one 256-row growth step, so the live window ends inside
    // a step-padded buffer, and two batch rows make it a strided slice.
    let (batch, keys) = (2usize, 301usize);
    let fx = MlaFixture::new(WIDE, batch, 1, keys, 0x57E9);
    let dt = dtype::FLOAT16;
    let (ckv, kpe) = (fx.ckv_array(dt), fx.kpe_array(dt));
    let rows = |a: &MlxArray, from: i32, to: i32| {
        let s = ffi::array_shape(a);
        ffi::slice(a, &[0, 0, from, 0], &[s[0], s[1], to, s[3]])
    };
    let case = Case::new(WIDE, batch, keys, dt, 8, 0x57E9, None);
    let mut inner = KVCache::new_with_mode(KVCacheMode::Int8);
    let mut cache = MlaLatentCache::wrap(&mut inner, WIDE).unwrap();
    let _ = cache.update_and_fetch(rows(&ckv, 0, 300), rows(&kpe, 0, 300));
    let _ = stats::take();
    let out = cache.attend_preprojected(
        &case.q,
        &case.q_pe,
        rows(&ckv, 300, 301),
        rows(&kpe, 300, 301),
        fx.scale,
        None,
    );
    let counts = stats::take();
    assert_eq!(counts.absorbed_fused, 1, "{}", counts.summary());
    assert_eq!(counts.total(), 1, "{}", counts.summary());

    let w = inner.latent_quantized_window().unwrap();
    let want = composed_quantized_attention(
        &case.q, &case.q_pe, &w.packed, &w.scales, &w.biases, &w.kpe, 8, 64, fx.scale, None,
    );
    let _ = stats::take();
    let err = max_rel_error(&to_vec_f32(&out), &to_vec_f32(&want));
    assert!(
        err < parity_tol(dt),
        "cache window: fused off composed by {err}"
    );
}

#[test]
fn windows_outside_the_quantized_contract_keep_the_composed_path() {
    let _guard = serial();
    if !kernel_live() {
        return;
    }
    let declined = |case: &Case, what: &str| {
        let _ = stats::take();
        assert!(case.fused().is_none(), "{what}: the kernel took it");
        assert_eq!(stats::take().total(), 0, "{what}: a declined call recorded");
    };
    // A 4-bit window: not a cache mode, refused by the Rust gate and, behind
    // it, by the launcher's own width dispatch (an error, not an abort).
    let four = Case::new(WIDE, 1, 9, dtype::FLOAT16, 4, 0xD0, None);
    declined(&four, "4-bit");
    assert!(
        ffi::mla_absorbed_decode_fused_quantized(
            &four.q,
            &four.q_pe,
            &four.ckv_q,
            &four.scales,
            &four.biases,
            &four.kpe,
            four.scale,
            four.group_size,
            four.bits,
        )
        .is_err(),
        "the launcher ran a 4-bit latent"
    );
    // One key past the limit.
    let long = FUSED_DECODE_QUANTIZED_MAX_KEYS as usize + 1;
    declined(
        &Case::new(NARROW, 1, long, dtype::FLOAT16, 8, 0xD1, None),
        "long",
    );
    // A rank whose lanes would straddle groups (192 / 32 = 6 elements each).
    let straddling = MlaGeometry {
        kv_lora_rank: 192,
        ..WIDE
    };
    let c = Case::new(straddling, 1, 9, dtype::FLOAT16, 8, 0xD4, None);
    declined(&c, "rank 192");
    // A group size other than the cache's.
    let mut odd = Case::new(WIDE, 1, 9, dtype::FLOAT16, 8, 0xD2, None);
    odd.group_size = 32;
    declined(&odd, "group size 32");
    // More than one query.
    let mut multi = Case::new(WIDE, 1, 9, dtype::FLOAT16, 8, 0xD3, None);
    multi.q = crate::concatenate(&multi.q, &multi.q, 2);
    multi.q_pe = crate::concatenate(&multi.q_pe, &multi.q_pe, 2);
    declined(&multi, "two queries");
}

/// Window crossover of the quantized kernel against the composed quantized
/// path, at DeepSeek-V2-Lite's geometry. Not a test: it prints the table
/// `FUSED_DECODE_QUANTIZED_MAX_KEYS` is set from. The kernel is launched
/// through the bridge so windows past the limit can be timed.
///
/// ```text
/// cargo test --profile test-fast --features metal,accelerate -p mlxcel-core --lib \
///     quantized_kernel_window_crossover -- --ignored --nocapture --test-threads=1
/// ```
#[test]
#[ignore = "microbenchmark; run by hand under the GPU lock"]
fn quantized_kernel_window_crossover() {
    use std::time::Instant;
    let _guard = serial();
    if !kernel_live() {
        return;
    }
    const LITE: MlaGeometry = MlaGeometry {
        num_heads: 16,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
    };
    const STEPS: usize = 200;
    const ROUNDS: usize = 5;
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    println!(
        "keys  int8_fused_us  int8_composed_us  ratio  fp16_fused_us  fp16_composed_us  ratio"
    );
    for keys in [512usize, 2048, 4096, 8192, 12288, 16384, 32768] {
        let c = Case::new(LITE, 1, keys, dtype::FLOAT16, 8, 0xC0 + keys as u64, None);
        // The fp16 latent the same rows stand for, for the dense kernel's own
        // crossover on this host.
        let dense = ffi::astype(
            &crate::mla::quant_latent::dequantize_latent(&c.ckv_q, &c.scales, &c.biases, 8, 64),
            dtype::FLOAT16,
        );
        ffi::eval(&dense);
        let arms: [&dyn Fn() -> UniquePtr<MlxArray>; 4] = [
            &|| c.kernel(),
            &|| c.composed(),
            &|| {
                ffi::mla_absorbed_decode_fused(&c.q, &c.q_pe, &dense, &c.kpe, c.scale)
                    .expect("the kernel has no key limit of its own")
            },
            &|| {
                let mask = crate::mla::decode::rope_score_mask(&c.q_pe, &c.kpe, c.scale, None);
                crate::layers::attention(&c.q, &dense, &dense, c.scale, Some(&mask), 0.0, 0)
            },
        ];
        let mut times = [const { Vec::new() }; 4];
        for _ in 0..ROUNDS {
            for (arm, t) in arms.iter().zip(&mut times) {
                let start = Instant::now();
                for _ in 0..STEPS {
                    ffi::eval(&arm());
                }
                t.push(start.elapsed().as_secs_f64() * 1e6 / STEPS as f64);
            }
        }
        let [qf, qc, df, dc] = times.map(median);
        println!(
            "{keys:5}  {qf:13.1}  {qc:16.1}  {:5.3}  {df:13.1}  {dc:16.1}  {:5.3}",
            qf / qc,
            df / dc
        );
    }
    let _ = stats::take();
}
