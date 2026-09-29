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

//! Affine 1-bit quantization (issue #1370): kernel parity, the dequantize
//! reference, load-time validation, and routing through the layer types.
//!
//! The oracle is a host f64 computation over the exact values the device sees
//! (activations, scales and biases are read back after the dtype cast), so the
//! fused kernels and the dequantize fallback are each checked against ground
//! truth rather than against one another.
//!
//! Kernel cases skip on a backend without a 1-bit port (anything but Metal
//! today); the dequantize, validation and routing cases run everywhere.
//!
//!   cargo test --release -p mlxcel-core --lib one_bit_tests

use super::*;
use crate::layers::{
    FusedQKVLinear, QuantizedEmbedding, QuantizedMultiLinear, UnifiedLinear,
    reconcile_quantization_layout, validate_one_bit_layout,
};
use crate::weights::WeightMap;

/// Deterministic pseudo-random stream (64-bit LCG), so the host reference and
/// the device inputs are built from the same numbers without an RNG crate.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    /// Uniform in [-1, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn flatten_f32(arr: &MlxArray) -> Vec<f32> {
    let a = astype(arr, dtype::FLOAT32);
    eval(&a);
    array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A packed 1-bit triple plus the host copies of the values the device holds.
struct Triple {
    w: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: UniquePtr<MlxArray>,
    words: Vec<u32>,
    scales_host: Vec<f32>,
    biases_host: Vec<f32>,
    n: usize,
    k: usize,
    g: usize,
}

impl Triple {
    fn new(rng: &mut Lcg, n: usize, k: usize, g: usize, scale_dtype: i32) -> Self {
        let words: Vec<u32> = (0..n * k / 32).map(|_| rng.next_u32()).collect();
        let groups = k / g;
        // Bonsai-like magnitudes: a positive scale and a bias around -scale / 2,
        // so each weight is roughly +-scale / 2.
        let s: Vec<f32> = (0..n * groups)
            .map(|_| 0.02 + 0.03 * rng.next_f32().abs())
            .collect();
        let b: Vec<f32> = s
            .iter()
            .map(|v| -0.5 * v + 0.002 * rng.next_f32())
            .collect();
        let w = from_slice_u32(&words, &[n as i32, (k / 32) as i32]);
        let scales = astype(&from_slice_f32(&s, &[n as i32, groups as i32]), scale_dtype);
        let biases = astype(&from_slice_f32(&b, &[n as i32, groups as i32]), scale_dtype);
        let scales_host = flatten_f32(&scales);
        let biases_host = flatten_f32(&biases);
        Self {
            w,
            scales,
            biases,
            words,
            scales_host,
            biases_host,
            n,
            k,
            g,
        }
    }

    fn weight(&self, o: usize, i: usize) -> f64 {
        let bit = (self.words[o * self.k / 32 + i / 32] >> (i % 32)) & 1;
        let gi = o * (self.k / self.g) + i / self.g;
        f64::from(bit) * f64::from(self.scales_host[gi]) + f64::from(self.biases_host[gi])
    }

    /// Host `x @ W^T` in f64 over `x` read back from the device.
    fn reference(&self, x_host: &[f32], m: usize) -> Vec<f64> {
        let mut y = vec![0f64; m * self.n];
        for o in 0..self.n {
            let row: Vec<f64> = (0..self.k).map(|i| self.weight(o, i)).collect();
            for r in 0..m {
                let xr = &x_host[r * self.k..(r + 1) * self.k];
                y[r * self.n + o] = row.iter().zip(xr).map(|(w, x)| w * f64::from(*x)).sum();
            }
        }
        y
    }
}

fn activations(rng: &mut Lcg, m: usize, k: usize, dtype: i32) -> (UniquePtr<MlxArray>, Vec<f32>) {
    let v: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
    let x = astype(&from_slice_f32(&v, &[m as i32, k as i32]), dtype);
    let host = flatten_f32(&x);
    (x, host)
}

/// Asserts `max |got - want| <= tol * max |want|`.
fn assert_close(got: &MlxArray, want: &[f64], tol: f64, what: &str) {
    let got = flatten_f32(got);
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f64, |a, v| a.max(v.abs())).max(1e-6);
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).abs())
        .fold(0f64, f64::max);
    assert!(
        worst <= tol * scale,
        "{what}: max deviation {worst} exceeds {tol} * {scale}"
    );
}

fn tolerance(dtype: i32) -> f64 {
    if dtype == dtype::BFLOAT16 { 2e-2 } else { 5e-3 }
}

fn matmul_case(m: usize, k: usize, n: usize, g: usize, x_dtype: i32, seed: u64) {
    let mut rng = Lcg(seed);
    let t = Triple::new(&mut rng, n, k, g, dtype::FLOAT16);
    let (x, x_host) = activations(&mut rng, m, k, x_dtype);
    let want = t.reference(&x_host, m);
    let what = format!("m={m} k={k} n={n} g={g} dtype={x_dtype}");

    let fallback =
        one_bit_quantized_matmul(&x, &t.w, &t.scales, &t.biases, g as i32, true).expect("fallback");
    assert_eq!(array_shape(&fallback), vec![m as i32, n as i32]);
    assert_eq!(array_dtype(&fallback), x_dtype, "{what}: output dtype");
    assert_close(
        &fallback,
        &want,
        tolerance(x_dtype),
        &format!("fallback {what}"),
    );

    if one_bit_kernel_available() {
        let fused = one_bit_quantized_matmul(&x, &t.w, &t.scales, &t.biases, g as i32, false)
            .expect("kernel");
        assert_eq!(array_dtype(&fused), x_dtype, "{what}: output dtype");
        assert_close(&fused, &want, tolerance(x_dtype), &format!("kernel {what}"));
    }
}

/// On Metal the parity tests below must exercise the fused kernels, not
/// silently fall back to the graph because the port table went missing.
#[test]
fn one_bit_kernel_available_on_metal() {
    if metal_is_available() {
        assert!(one_bit_kernel_available());
    }
}

#[test]
fn one_bit_dequantize_matches_host_formula() {
    for scale_dtype in [dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
        let mut rng = Lcg(7);
        let t = Triple::new(&mut rng, 8, 128, 128, scale_dtype);
        let dense = one_bit_dequantize(&t.w, &t.scales, &t.biases, 128).expect("dequantize");
        assert_eq!(array_shape(&dense), vec![8, 128]);
        assert_eq!(array_dtype(&dense), scale_dtype, "keeps the shipped dtype");
        let got = flatten_f32(&dense);
        for o in 0..8 {
            for i in 0..128 {
                // One rounding of `scale + bias` (or `bias` alone) to the
                // shipped dtype, formed in f32.
                let want = astype(&from_slice_f32(&[t.weight(o, i) as f32], &[1]), scale_dtype);
                let want = flatten_f32(&want)[0];
                assert_eq!(got[o * 128 + i], want, "dtype {scale_dtype} at ({o}, {i})");
            }
        }
    }
}

#[test]
fn one_bit_qmv_matches_reference_m1_m7() {
    for m in [1, 7] {
        for k in [512, 2048, 4096] {
            for n in [64, 96, 1000] {
                for x_dtype in [dtype::FLOAT16, dtype::BFLOAT16] {
                    matmul_case(m, k, n, 128, x_dtype, (m * 31 + k + n) as u64);
                }
            }
        }
    }
}

#[test]
fn one_bit_qmv_unaligned_rows_and_short_k() {
    // N not a multiple of 2R (the tied lm_head has N = 151669), and K below
    // one 512-column lane stride so most lanes see no work.
    matmul_case(1, 256, 1001, 128, dtype::FLOAT16, 11);
    matmul_case(3, 2048, 151, 64, dtype::FLOAT32, 12);
}

#[test]
fn one_bit_qmm_matches_reference_m16_m33() {
    for m in [16, 33] {
        for n in [96, 100] {
            matmul_case(m, 1024, n, 128, dtype::FLOAT16, (m + n) as u64);
        }
    }
    matmul_case(40, 2048, 64, 64, dtype::BFLOAT16, 5);
}

#[test]
fn one_bit_qmv_group_sizes_32_64_128() {
    for g in [32, 64, 128] {
        matmul_case(1, 1024, 96, g, dtype::FLOAT16, g as u64);
        matmul_case(17, 1024, 96, g, dtype::FLOAT16, g as u64 + 1);
    }
}

#[test]
fn one_bit_kill_switch_routes_to_dequant() {
    let mut rng = Lcg(3);
    let t = Triple::new(&mut rng, 96, 1024, 128, dtype::FLOAT16);
    let (x, _) = activations(&mut rng, 2, 1024, dtype::FLOAT16);
    let forced =
        one_bit_quantized_matmul(&x, &t.w, &t.scales, &t.biases, 128, true).expect("fallback");
    let dense = one_bit_dequantize(&t.w, &t.scales, &t.biases, 128).expect("dequantize");
    let graph = matmul(&x, &transpose(&astype(&dense, dtype::FLOAT16)));
    eval(&forced);
    eval(&graph);
    assert_eq!(array_to_raw_bytes(&forced), array_to_raw_bytes(&graph));
}

#[test]
fn one_bit_matmul_rejects_inconsistent_triple() {
    let mut rng = Lcg(4);
    let t = Triple::new(&mut rng, 8, 256, 128, dtype::FLOAT16);
    let (x, _) = activations(&mut rng, 1, 256, dtype::FLOAT16);
    assert!(one_bit_quantized_matmul(&x, &t.w, &t.scales, &t.biases, 64, false).is_err());
    assert!(one_bit_dequantize(&t.w, &t.scales, &t.biases, 16).is_err());
}

#[test]
fn bridge_quantized_matmul_routes_one_bit_both_layouts() {
    let mut rng = Lcg(5);
    let t = Triple::new(&mut rng, 64, 512, 64, dtype::FLOAT16);
    let (x, x_host) = activations(&mut rng, 3, 512, dtype::FLOAT16);
    let y = unsafe {
        quantized_matmul(
            &x,
            &t.w,
            &t.scales,
            t.biases.as_ref().unwrap(),
            true,
            64,
            1,
            "affine",
        )
    };
    assert_close(&y, &t.reference(&x_host, 3), 5e-3, "transpose=true");

    // transpose=false is `x @ W` with W's packed rows as the reduction axis.
    let (x2, x2_host) = activations(&mut rng, 2, 64, dtype::FLOAT16);
    let y2 = unsafe {
        quantized_matmul(
            &x2,
            &t.w,
            &t.scales,
            t.biases.as_ref().unwrap(),
            false,
            64,
            1,
            "affine",
        )
    };
    let mut want = vec![0f64; 2 * 512];
    for r in 0..2 {
        for i in 0..512 {
            want[r * 512 + i] = (0..64)
                .map(|o| f64::from(x2_host[r * 64 + o]) * t.weight(o, i))
                .sum();
        }
    }
    assert_close(&y2, &want, 5e-3, "transpose=false");
}

fn one_bit_weight_map(prefix: &str, t: &Triple) -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert(format!("{prefix}.weight"), copy(&t.w));
    weights.insert(format!("{prefix}.scales"), copy(&t.scales));
    weights.insert(format!("{prefix}.biases"), copy(&t.biases));
    weights
}

#[test]
fn unified_linear_routes_one_bit() {
    let mut rng = Lcg(6);
    let t = Triple::new(&mut rng, 96, 1024, 128, dtype::FLOAT16);
    let weights = one_bit_weight_map("proj", &t);
    let layer = UnifiedLinear::from_weights(&weights, "proj", 128, 1).expect("loads");
    match &layer {
        UnifiedLinear::Quantized { weight, .. } => {
            assert_eq!(weight.bits, 1);
            assert_eq!(weight.mode, "affine");
            assert!(weight.is_one_bit());
        }
        UnifiedLinear::Regular(_) => panic!("1-bit triple loaded as a dense linear"),
    }
    // Fused consumers must not see the packed weight.
    assert!(layer.quantized_weight().is_none());
    assert!(layer.as_quantized_weight().is_none());

    for m in [1, 20] {
        let (x, x_host) = activations(&mut rng, m, 1024, dtype::FLOAT16);
        let x3 = reshape(&x, &[1, m as i32, 1024]);
        let y = layer.forward(&x3);
        assert_eq!(array_shape(&y), vec![1, m as i32, 96]);
        assert_close(
            &y,
            &t.reference(&x_host, m),
            5e-3,
            &format!("forward m={m}"),
        );
    }

    let dense = layer.dequantized_weight();
    let reference = one_bit_dequantize(&t.w, &t.scales, &t.biases, 128).expect("dequantize");
    eval(&dense);
    eval(&reference);
    assert_eq!(array_to_raw_bytes(&dense), array_to_raw_bytes(&reference));
}

#[test]
fn one_bit_declared_under_another_width_is_reconciled() {
    // Shapes of a 1-bit plane at group_size 128: packed_in = 64, 16 groups.
    let layout = reconcile_quantization_layout(&[32, 64], &[32, 16], 128, 1, "affine")
        .expect("1 bit is a loadable width");
    assert_eq!((layout.bits, layout.reconciled), (1, false));
    let layout = reconcile_quantization_layout(&[32, 64], &[32, 16], 128, 4, "affine")
        .expect("declared 4, shapes say 1");
    assert_eq!((layout.bits, layout.reconciled), (1, true));
}

#[test]
fn one_bit_without_biases_is_rejected() {
    let mut rng = Lcg(8);
    let t = Triple::new(&mut rng, 32, 512, 128, dtype::FLOAT16);
    let mut weights = WeightMap::new();
    weights.insert("proj.weight".into(), copy(&t.w));
    weights.insert("proj.scales".into(), copy(&t.scales));
    let err = UnifiedLinear::from_weights(&weights, "proj", 128, 1)
        .err()
        .expect("block-float 1-bit must be refused");
    assert!(err.contains("affine-only"), "{err}");
    assert!(err.contains("proj"), "{err}");

    let err = QuantizedEmbedding::from_weights(&weights, "proj", 128, 1)
        .err()
        .expect("block-float 1-bit embedding must be refused");
    assert!(err.contains("affine-only"), "{err}");
}

#[test]
fn one_bit_layout_validator() {
    assert!(validate_one_bit_layout(4, 16, "affine", true).is_ok());
    assert!(validate_one_bit_layout(1, 128, "affine", true).is_ok());
    assert!(validate_one_bit_layout(1, 128, "mxfp4", false).is_err());
    assert!(validate_one_bit_layout(1, 16, "affine", true).is_err());
}

#[test]
fn quantized_multi_linear_rejects_one_bit_with_prefix() {
    let mut rng = Lcg(9);
    let t = Triple::new(&mut rng, 32, 512, 128, dtype::FLOAT16);
    let weights = one_bit_weight_map("embed_q", &t);
    let err = QuantizedMultiLinear::from_weights(&weights, "embed_q", 128, 1)
        .err()
        .expect("1-bit MLA must be refused");
    assert!(err.contains("embed_q"), "{err}");
    assert!(err.contains("1-bit"), "{err}");
}

#[test]
fn one_bit_embedding_gather_matches_dequant_rows() {
    let mut rng = Lcg(10);
    let t = Triple::new(&mut rng, 50, 512, 128, dtype::FLOAT16);
    let weights = one_bit_weight_map("embed", &t);
    let emb = QuantizedEmbedding::from_weights(&weights, "embed", 128, 1).expect("loads");
    assert_eq!(emb.bits, 1);

    let ids = [3_i32, 0, 49, 3, 17, 8];
    let indices = from_slice_i32(&ids, &[2, 3]);
    let rows = emb.forward(&indices);
    assert_eq!(array_shape(&rows), vec![2, 3, 512]);
    assert_eq!(array_dtype(&rows), dtype::FLOAT16);

    let dense = one_bit_dequantize(&t.w, &t.scales, &t.biases, 128).expect("dequantize");
    let table = flatten_f32(&dense);
    let got = flatten_f32(&rows);
    for (pos, id) in ids.iter().enumerate() {
        let id = *id as usize;
        assert_eq!(
            &got[pos * 512..(pos + 1) * 512],
            &table[id * 512..(id + 1) * 512],
            "row for token {id}"
        );
    }
}

#[test]
fn one_bit_embedding_as_linear_matches_reference() {
    let mut rng = Lcg(12);
    // An unaligned vocabulary, as a tied lm_head has.
    let t = Triple::new(&mut rng, 1003, 1024, 128, dtype::FLOAT16);
    let weights = one_bit_weight_map("embed", &t);
    let emb = QuantizedEmbedding::from_weights(&weights, "embed", 128, 1).expect("loads");
    for m in [1, 18] {
        let (x, x_host) = activations(&mut rng, m, 1024, dtype::FLOAT16);
        let logits = emb.as_linear(&x);
        assert_close(
            &logits,
            &t.reference(&x_host, m),
            5e-3,
            &format!("as_linear m={m}"),
        );
    }
}

#[test]
fn fused_qkv_declines_one_bit() {
    let mut rng = Lcg(13);
    let (n_heads, n_kv_heads, head_dim) = (4, 2, 64);
    let k = 512;
    let q = Triple::new(&mut rng, n_heads * head_dim, k, 128, dtype::FLOAT16);
    let kk = Triple::new(&mut rng, n_kv_heads * head_dim, k, 128, dtype::FLOAT16);
    let v = Triple::new(&mut rng, n_kv_heads * head_dim, k, 128, dtype::FLOAT16);
    let mut weights = WeightMap::new();
    for (name, t) in [
        ("attn.q_proj", &q),
        ("attn.k_proj", &kk),
        ("attn.v_proj", &v),
    ] {
        weights.extend(one_bit_weight_map(name, t));
    }
    let qkv = FusedQKVLinear::from_weights_separate(
        &weights,
        "attn",
        128,
        1,
        n_heads as i32,
        n_kv_heads as i32,
        head_dim as i32,
    )
    .expect("loads");
    assert!(
        qkv.fused_quantized_weight().is_none(),
        "fused launchers must decline 1-bit"
    );
    assert!(
        qkv.forward_split_rope_quantized(&ones(&[1, 1, k as i32], dtype::FLOAT16), 64, 10000.0, 0)
            .is_none()
    );

    let (x, x_host) = activations(&mut rng, 1, k, dtype::FLOAT16);
    let x3 = reshape(&x, &[1, 1, k as i32]);
    let (yq, yk, yv) = qkv.forward(&x3);
    assert_close(&yq, &q.reference(&x_host, 1), 5e-3, "q");
    assert_close(&yk, &kk.reference(&x_host, 1), 5e-3, "k");
    assert_close(&yv, &v.reference(&x_host, 1), 5e-3, "v");
}
