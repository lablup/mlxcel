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

//! Kokoro prosody predictor: durations, F0 (pitch) and N (energy/noise).
//!
//! Pipeline (all conditioned on the 128-d predictor style `s = ref_s[:,128:]`):
//! 1. `DurationEncoder`: three bi-LSTM + AdaLayerNorm blocks over the BERT
//!    features (with the style concatenated each block) -> `(L, 640)`.
//! 2. Duration head: a bi-LSTM (`640 -> 512`) then `duration_proj` (`512 -> 50`);
//!    `sigmoid(.).sum(-1) / speed` gives per-token frame counts.
//! 3. `F0Ntrain`: a shared bi-LSTM (`640 -> 512`) then two `AdainResBlk1d`
//!    stacks (the middle block upsamples `2x`) and a `1x1` projection, giving
//!    the F0 and N curves at twice the frame resolution.

use mlxcel_core::{MlxArray, UniquePtr};

use super::blocks::{AdainResBlk1d, align_time};
use super::lstm::BiLstm;
use super::ops;
use super::weights::{ConvWn, Weights};

const STYLE_DIM: i32 = 128;
const HIDDEN: i32 = 512;
const LN_EPS: f32 = 1e-5;

/// AdaLayerNorm: LayerNorm over the feature axis with style-derived affine.
struct AdaLayerNorm {
    fc_w: UniquePtr<MlxArray>,
    fc_b: UniquePtr<MlxArray>,
    channels: i32,
}

impl AdaLayerNorm {
    fn load(w: &Weights, prefix: &str, channels: i32) -> Result<Self, String> {
        let (fc_w, fc_b) = w.linear(&format!("{prefix}.fc"))?;
        let fc_b = fc_b.ok_or_else(|| format!("kokoro: {prefix}.fc.bias missing"))?;
        Ok(Self {
            fc_w,
            fc_b,
            channels,
        })
    }

    /// Apply to `(L, C)` with style `(1, 128)`.
    fn forward(&self, x: &UniquePtr<MlxArray>, style: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
        let h = ops::linear(style, &self.fc_w, Some(&self.fc_b)); // (1, 2C)
        let h = ops::reshape(&h, &[2 * self.channels]);
        let gamma = ops::slice(&h, &[0], &[self.channels]); // (C,)
        let beta = ops::slice(&h, &[self.channels], &[2 * self.channels]);
        // LayerNorm over last axis, no built-in affine, then (1+gamma)*x+beta.
        let mu = ops::mean_axis(x, -1);
        let xc = ops::sub(x, &mu);
        let var = ops::var_axis(x, -1);
        let denom = mlxcel_core::rsqrt(ops::r(&ops::add_scalar(&var, LN_EPS)));
        let normed = ops::mul(&xc, &denom);
        let scaled = ops::mul(&normed, &ops::add_scalar(&gamma, 1.0));
        ops::add(&scaled, &beta)
    }
}

/// DurationEncoder: three bi-LSTM + AdaLayerNorm blocks.
struct DurationEncoder {
    lstms: Vec<BiLstm>,
    norms: Vec<AdaLayerNorm>,
}

impl DurationEncoder {
    fn load(w: &Weights) -> Result<Self, String> {
        // lstms.0/2/4 are bi-LSTMs, lstms.1/3/5 carry the AdaLayerNorm fc.
        let mut lstms = Vec::new();
        let mut norms = Vec::new();
        for &li in &[0, 2, 4] {
            lstms.push(BiLstm::load(
                w,
                &format!("predictor.text_encoder.lstms.{li}"),
                HIDDEN / 2,
            )?);
        }
        for &ni in &[1, 3, 5] {
            norms.push(AdaLayerNorm::load(
                w,
                &format!("predictor.text_encoder.lstms.{ni}"),
                HIDDEN,
            )?);
        }
        Ok(Self { lstms, norms })
    }

    /// `d_en` is `(512, L)`; returns `(L, 640)` (style re-concatenated).
    fn forward(
        &self,
        d_en: &UniquePtr<MlxArray>,
        style: &UniquePtr<MlxArray>,
        l: usize,
    ) -> UniquePtr<MlxArray> {
        let mut x = ops::swap_axes(d_en, 0, 1); // (L, 512)
        let sty = broadcast_style(style, l); // (L, 128)
        x = ops::concat2(&x, &sty, 1); // (L, 640)
        for (lstm, norm) in self.lstms.iter().zip(self.norms.iter()) {
            x = lstm.forward(&x, l); // (L, 512)
            x = norm.forward(&x, style); // (L, 512)
            x = ops::concat2(&x, &sty, 1); // (L, 640)
        }
        x
    }
}

/// The full prosody predictor.
pub(crate) struct Predictor {
    duration_encoder: DurationEncoder,
    lstm: BiLstm,
    duration_proj_w: UniquePtr<MlxArray>,
    duration_proj_b: UniquePtr<MlxArray>,
    shared: BiLstm,
    f0_blocks: Vec<AdainResBlk1d>,
    n_blocks: Vec<AdainResBlk1d>,
    f0_proj: ConvWn,
    n_proj: ConvWn,
}

impl Predictor {
    pub(crate) fn load(w: &Weights) -> Result<Self, String> {
        let duration_encoder = DurationEncoder::load(w)?;
        let lstm = BiLstm::load(w, "predictor.lstm", HIDDEN / 2)?;
        let (duration_proj_w, duration_proj_b) =
            w.linear("predictor.duration_proj.linear_layer")?;
        let duration_proj_b =
            duration_proj_b.ok_or_else(|| "kokoro: duration_proj bias missing".to_string())?;
        let shared = BiLstm::load(w, "predictor.shared", HIDDEN / 2)?;

        let f0_blocks = vec![
            AdainResBlk1d::load(w, "predictor.F0.0", 512, 512, false, false)?,
            AdainResBlk1d::load(w, "predictor.F0.1", 512, 256, true, false)?,
            AdainResBlk1d::load(w, "predictor.F0.2", 256, 256, false, false)?,
        ];
        let n_blocks = vec![
            AdainResBlk1d::load(w, "predictor.N.0", 512, 512, false, false)?,
            AdainResBlk1d::load(w, "predictor.N.1", 512, 256, true, false)?,
            AdainResBlk1d::load(w, "predictor.N.2", 256, 256, false, false)?,
        ];
        // F0_proj / N_proj are plain Conv1d(256 -> 1, k=1) (no weight-norm).
        let f0_proj = ConvWn::load_plain(w, "predictor.F0_proj", 1, 0)?;
        let n_proj = ConvWn::load_plain(w, "predictor.N_proj", 1, 0)?;

        Ok(Self {
            duration_encoder,
            lstm,
            duration_proj_w,
            duration_proj_b,
            shared,
            f0_blocks,
            n_blocks,
            f0_proj,
            n_proj,
        })
    }

    /// Run the duration encoder and head.
    ///
    /// Returns `(d, durations)` where `d` is `(L, 640)` (reused for prosody
    /// expansion) and `durations` is the `(L, 50)` logits.
    pub(crate) fn durations(
        &self,
        d_en: &UniquePtr<MlxArray>,
        style: &UniquePtr<MlxArray>,
        l: usize,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let d = self.duration_encoder.forward(d_en, style, l); // (L, 640)
        let x = self.lstm.forward(&d, l); // (L, 512)
        let dur = ops::linear(&x, &self.duration_proj_w, Some(&self.duration_proj_b)); // (L, 50)
        (d, dur)
    }

    /// F0 and N curves from the expanded prosody features `en` `(640, T)`.
    /// Returns `(f0, n)` each `(2T,)`.
    pub(crate) fn f0n(
        &self,
        en: &UniquePtr<MlxArray>,
        style: &UniquePtr<MlxArray>,
        t: usize,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shared = self.shared.forward(&ops::swap_axes(en, 0, 1), t); // (T, 512)
        let base = ops::swap_axes(&shared, 0, 1); // (512, T)

        let mut f0 = mlxcel_core::copy(base.as_ref().expect("kokoro: f0 base"));
        for blk in &self.f0_blocks {
            f0 = blk.forward(&f0, style);
        }
        let f0 = self.f0_proj.forward(&f0); // (1, 2T)

        let mut n = mlxcel_core::copy(base.as_ref().expect("kokoro: n base"));
        for blk in &self.n_blocks {
            n = blk.forward(&n, style);
        }
        let n = self.n_proj.forward(&n); // (1, 2T)

        // Trim to the common length and squeeze the channel axis.
        let (f0, n) = align_time(&f0, &n);
        let f0 = ops::reshape(&f0, &[ops::shape(&f0)[1]]);
        let n = ops::reshape(&n, &[ops::shape(&n)[1]]);
        (f0, n)
    }
}

/// Broadcast a `(1, 128)` style row to `(L, 128)`.
fn broadcast_style(style: &UniquePtr<MlxArray>, l: usize) -> UniquePtr<MlxArray> {
    mlxcel_core::broadcast_to(ops::r(style), &[l as i32, STYLE_DIM])
}
