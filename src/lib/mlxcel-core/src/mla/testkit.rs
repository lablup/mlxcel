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

//! Synthetic MLA fixtures and a host-side f64 reference (issue #907).
//!
//! MLA is a DeepSeek-family feature and no MLA checkpoint fits on the
//! development host, so every test in this module family runs on synthetic
//! tensors with MLA geometry. That is a real limitation and it is stated here
//! rather than implied: these tests prove the absorption identity, the cache
//! packing, and the split-and-merge decomposition, and they do not prove that a
//! real DeepSeek checkpoint produces identical tokens.
//!
//! The reference is computed on the host in f64 from the same values that were
//! uploaded, following `paged_v2::launch_tests`. A mismatch is then attributable
//! to the path under test rather than to a disagreement between two GPU paths
//! that could both be wrong.

use cxx::UniquePtr;

use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::mla::MlaGeometry;

/// Serializes MLA tests that inspect [`crate::mla::stats`].
///
/// The dispatch counters are process-global by design (a benchmark reads them
/// after a measured region), so a test that asserts an exact count has to be
/// the only MLA test running. Every helper below that executes a decode path
/// takes this, which costs nothing: these tests are milliseconds each.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire the MLA test serialization lock, ignoring poisoning so one failing
/// test does not cascade into every other one.
pub fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// xorshift64* in [-1, 1). Deterministic so a failure reproduces exactly.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = ((self.0 >> 40) as f32) / (1u32 << 24) as f32;
        unit * 2.0 - 1.0
    }

    pub fn vec(&mut self, n: usize, amplitude: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * amplitude).collect()
    }
}

/// Read an array back to the host as f32.
pub fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let f = ffi::astype(a, dtype::FLOAT32);
    ffi::eval(&f);
    ffi::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

/// Max absolute deviation relative to the reference's own scale.
pub fn max_rel_error(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "output length mismatch");
    let scale = want
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        .max(1e-6);
    got.iter()
        .zip(want)
        .fold(0.0f32, |acc, (g, w)| acc.max((g - w).abs() / scale))
}

/// One synthetic MLA layer plus a decode (or short prefill) step, held on the
/// host so the reference needs no device readback.
pub struct MlaFixture {
    pub geometry: MlaGeometry,
    pub batch: usize,
    pub q_len: usize,
    pub kv_len: usize,
    pub scale: f32,
    /// `[H * (qk_nope + v), kv_lora_rank]`, row-major.
    pub kv_b: Vec<f32>,
    /// `[B, kv_len, kv_lora_rank]`.
    pub ckv: Vec<f32>,
    /// `[B, kv_len, qk_rope_head_dim]`.
    pub kpe: Vec<f32>,
    /// `[B, H, q_len, qk_nope_head_dim]`.
    pub q_nope: Vec<f32>,
    /// `[B, H, q_len, qk_rope_head_dim]`.
    pub q_pe: Vec<f32>,
}

impl MlaFixture {
    pub fn new(
        geometry: MlaGeometry,
        batch: usize,
        q_len: usize,
        kv_len: usize,
        seed: u64,
    ) -> Self {
        let mut rng = Rng::new(seed);
        let h = geometry.num_heads;
        let r = geometry.kv_lora_rank;
        let nope = geometry.qk_nope_head_dim;
        let p = geometry.qk_rope_head_dim;
        // Amplitudes kept modest so an f16 fixture does not saturate: the
        // absorbed score sums `r` products where the decompressed one sums
        // `nope`, so an aggressive amplitude would test float range, not the
        // identity.
        Self {
            geometry,
            batch,
            q_len,
            kv_len,
            scale: (geometry.q_head_dim() as f32).powf(-0.5),
            kv_b: rng.vec(h * geometry.kv_b_rows_per_head() * r, 0.5),
            ckv: rng.vec(batch * kv_len * r, 0.5),
            kpe: rng.vec(batch * kv_len * p, 0.5),
            q_nope: rng.vec(batch * h * q_len * nope, 0.5),
            q_pe: rng.vec(batch * h * q_len * p, 0.5),
        }
    }

    /// `kv_b_proj` as `[H*(qk_nope+v), kv_lora_rank]` in `dt`.
    pub fn kv_b_array(&self, dt: i32) -> UniquePtr<MlxArray> {
        let rows = (self.geometry.num_heads * self.geometry.kv_b_rows_per_head()) as i32;
        let a = ffi::from_slice_f32(&self.kv_b, &[rows, self.geometry.kv_lora_rank as i32]);
        ffi::astype(&a, dt)
    }

    /// `ckv` as `[B, 1, kv_len, kv_lora_rank]` in `dt`, the cache's key slot.
    pub fn ckv_array(&self, dt: i32) -> UniquePtr<MlxArray> {
        let a = ffi::from_slice_f32(
            &self.ckv,
            &[
                self.batch as i32,
                1,
                self.kv_len as i32,
                self.geometry.kv_lora_rank as i32,
            ],
        );
        ffi::astype(&a, dt)
    }

    /// `kpe` as `[B, 1, kv_len, qk_rope_head_dim]` in `dt`, the value slot.
    pub fn kpe_array(&self, dt: i32) -> UniquePtr<MlxArray> {
        let a = ffi::from_slice_f32(
            &self.kpe,
            &[
                self.batch as i32,
                1,
                self.kv_len as i32,
                self.geometry.qk_rope_head_dim as i32,
            ],
        );
        ffi::astype(&a, dt)
    }

    /// `q_nope` as `[B, H, q_len, qk_nope_head_dim]` in `dt`.
    pub fn q_nope_array(&self, dt: i32) -> UniquePtr<MlxArray> {
        let a = ffi::from_slice_f32(
            &self.q_nope,
            &[
                self.batch as i32,
                self.geometry.num_heads as i32,
                self.q_len as i32,
                self.geometry.qk_nope_head_dim as i32,
            ],
        );
        ffi::astype(&a, dt)
    }

    /// `q_pe` as `[B, H, q_len, qk_rope_head_dim]` in `dt`.
    pub fn q_pe_array(&self, dt: i32) -> UniquePtr<MlxArray> {
        let a = ffi::from_slice_f32(
            &self.q_pe,
            &[
                self.batch as i32,
                self.geometry.num_heads as i32,
                self.q_len as i32,
                self.geometry.qk_rope_head_dim as i32,
            ],
        );
        ffi::astype(&a, dt)
    }

    /// The pre-absorption answer, computed on the host in f64.
    ///
    /// Up-projects the latent into per-head K and V exactly as
    /// `deepseek_v2::MLAAttention::forward` does today, concatenates the rope
    /// stream into every head, and runs dense attention. Returns
    /// `[B, H, q_len, v_head_dim]` flattened row-major.
    ///
    /// `causal` applies the prefill mask, where query `l` sees cache rows
    /// `0 ..= kv_len - q_len + l`.
    pub fn decompressed_reference(&self, causal: bool) -> Vec<f32> {
        let h = self.geometry.num_heads;
        let r = self.geometry.kv_lora_rank;
        let nope = self.geometry.qk_nope_head_dim;
        let p = self.geometry.qk_rope_head_dim;
        let v = self.geometry.v_head_dim;
        let rows = self.geometry.kv_b_rows_per_head();
        let scale = self.scale as f64;
        let mut out = vec![0.0f32; self.batch * h * self.q_len * v];

        for b in 0..self.batch {
            for head in 0..h {
                // Up-project this head's K_nope and V from the latent.
                let mut k_nope = vec![0.0f64; self.kv_len * nope];
                let mut val = vec![0.0f64; self.kv_len * v];
                for t in 0..self.kv_len {
                    let c = &self.ckv[(b * self.kv_len + t) * r..][..r];
                    for d in 0..nope {
                        let w = &self.kv_b[(head * rows + d) * r..][..r];
                        k_nope[t * nope + d] =
                            (0..r).map(|i| w[i] as f64 * c[i] as f64).sum::<f64>();
                    }
                    for e in 0..v {
                        let w = &self.kv_b[(head * rows + nope + e) * r..][..r];
                        val[t * v + e] = (0..r).map(|i| w[i] as f64 * c[i] as f64).sum::<f64>();
                    }
                }

                for l in 0..self.q_len {
                    let qn = &self.q_nope[((b * h + head) * self.q_len + l) * nope..][..nope];
                    let qp = &self.q_pe[((b * h + head) * self.q_len + l) * p..][..p];
                    let visible = if causal {
                        self.kv_len - self.q_len + l + 1
                    } else {
                        self.kv_len
                    };

                    let mut scores = vec![f64::NEG_INFINITY; self.kv_len];
                    for (t, score) in scores.iter_mut().enumerate().take(visible) {
                        let nope_dot: f64 =
                            (0..nope).map(|d| qn[d] as f64 * k_nope[t * nope + d]).sum();
                        let kp = &self.kpe[(b * self.kv_len + t) * p..][..p];
                        let pe_dot: f64 = (0..p).map(|d| qp[d] as f64 * kp[d] as f64).sum();
                        *score = scale * (nope_dot + pe_dot);
                    }

                    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                    let denom: f64 = exps.iter().sum();
                    for e in 0..v {
                        let acc: f64 = (0..visible).map(|t| exps[t] * val[t * v + e]).sum();
                        out[((b * h + head) * self.q_len + l) * v + e] = (acc / denom) as f32;
                    }
                }
            }
        }
        out
    }

    /// Additive causal mask `[1, 1, q_len, kv_len]` matching
    /// [`Self::decompressed_reference`]`(true)`.
    pub fn causal_mask(&self, dt: i32) -> UniquePtr<MlxArray> {
        let mut data = vec![0.0f32; self.q_len * self.kv_len];
        for l in 0..self.q_len {
            let visible = self.kv_len - self.q_len + l + 1;
            for (t, cell) in data[l * self.kv_len..(l + 1) * self.kv_len]
                .iter_mut()
                .enumerate()
            {
                if t >= visible {
                    *cell = f32::NEG_INFINITY;
                }
            }
        }
        let a = ffi::from_slice_f32(&data, &[1, 1, self.q_len as i32, self.kv_len as i32]);
        ffi::astype(&a, dt)
    }
}

/// A small geometry with every dimension distinct, so a transposed or swapped
/// operand cannot pass by coincidence.
pub const TINY: MlaGeometry = MlaGeometry {
    num_heads: 4,
    kv_lora_rank: 32,
    qk_nope_head_dim: 16,
    qk_rope_head_dim: 8,
    v_head_dim: 12,
};
