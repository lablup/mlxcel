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

//! iSTFTNet decoder: aligned acoustic features + prosody -> waveform.
//!
//! Two stages. The `Decoder` mixes the asr features with the F0 and N curves
//! (each downsampled `2x` by a strided conv so they realign with the asr frame
//! rate), runs them through an `encode` block and four `decode` blocks (the last
//! upsamples `2x`), then hands off to the `Generator`. The `Generator` is the
//! HiFi-GAN/iSTFTNet vocoder: two transposed-conv upsamples, an NSF harmonic
//! source injected at each stage, a multi-receptive-field (MRF) sum of
//! Snake-activated residual blocks, a final `conv_post` producing `n_fft + 2`
//! channels, and an inverse STFT (`magnitude = exp(.)`, `phase = sin(.)`).

use mlxcel_core::{MlxArray, UniquePtr};

use super::blocks::{AdaIn1d, AdainResBlk1d, align_time};
use super::ops;
use super::stft::{self, NsfSource};
use super::weights::{ConvWn, Weights};

const STYLE_DIM: i32 = 128;
const N_FFT: i32 = 20;
const HARMONIC_NUM: usize = 8;
/// Total audio-rate upsample factor for the F0 curve feeding the NSF source:
/// product of the upsample rates (10 * 6) times the iSTFT hop (5).
const F0_UPSAMPLE: i32 = 300;

/// Snake-activated MRF residual block (`AdaINResBlock1`).
///
/// Three dilated conv pairs, each preceded by an AdaIN norm and a Snake
/// activation `x + (1/alpha) * sin(alpha*x)^2` with a learnable per-channel
/// `alpha`.
struct SnakeResBlock {
    convs1: Vec<ConvWn>,
    convs2: Vec<ConvWn>,
    adain1: Vec<AdaIn1d>,
    adain2: Vec<AdaIn1d>,
    alpha1: Vec<UniquePtr<MlxArray>>,
    alpha2: Vec<UniquePtr<MlxArray>>,
}

impl SnakeResBlock {
    fn load(
        w: &Weights,
        prefix: &str,
        channels: i32,
        kernel: i32,
        dilations: [i32; 3],
    ) -> Result<Self, String> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut adain1 = Vec::new();
        let mut adain2 = Vec::new();
        let mut alpha1 = Vec::new();
        let mut alpha2 = Vec::new();
        for (j, &d) in dilations.iter().enumerate() {
            convs1.push(ConvWn::load(
                w,
                &format!("{prefix}.convs1.{j}"),
                1,
                get_padding(kernel, d),
                d,
                1,
                true,
            )?);
            convs2.push(ConvWn::load(
                w,
                &format!("{prefix}.convs2.{j}"),
                1,
                get_padding(kernel, 1),
                1,
                1,
                true,
            )?);
            adain1.push(AdaIn1d::load(w, &format!("{prefix}.adain1.{j}"), channels)?);
            adain2.push(AdaIn1d::load(w, &format!("{prefix}.adain2.{j}"), channels)?);
            // alpha stored (1, C, 1); reshape to (C, 1) for the (C, T) activation.
            let a1 = ops::reshape(&w.get(&format!("{prefix}.alpha1.{j}"))?, &[channels, 1]);
            let a2 = ops::reshape(&w.get(&format!("{prefix}.alpha2.{j}"))?, &[channels, 1]);
            alpha1.push(a1);
            alpha2.push(a2);
        }
        Ok(Self {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
        })
    }

    fn forward(&self, x: &UniquePtr<MlxArray>, style: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
        let mut x = mlxcel_core::copy(x.as_ref().expect("kokoro: snake input"));
        for j in 0..3 {
            let mut xt = self.adain1[j].forward(&x, style);
            xt = snake(&xt, &self.alpha1[j]);
            xt = self.convs1[j].forward(&xt);
            xt = self.adain2[j].forward(&xt, style);
            xt = snake(&xt, &self.alpha2[j]);
            xt = self.convs2[j].forward(&xt);
            let (xt, xs) = align_time(&xt, &x);
            x = ops::add(&xt, &xs);
        }
        x
    }
}

/// Snake activation `x + (1/alpha) * sin(alpha*x)^2` (alpha broadcast `(C,1)`).
///
/// A small epsilon guards the `1/alpha` term against a zero scale.
fn snake(x: &UniquePtr<MlxArray>, alpha: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    let ax = ops::mul(x, alpha);
    let s = ops::sin(&ax);
    let s2 = ops::mul(&s, &s);
    let one = ops::scalar(1.0);
    let inv_alpha = ops::div(&one, &ops::add_scalar(alpha, 1e-9));
    ops::add(x, &ops::mul(&inv_alpha, &s2))
}

/// The iSTFTNet generator (vocoder).
struct Generator {
    ups: Vec<ConvWn>,
    noise_convs: Vec<ConvWn>,
    noise_res: Vec<SnakeResBlock>,
    resblocks: Vec<SnakeResBlock>,
    conv_post: ConvWn,
    source: NsfSource,
}

impl Generator {
    fn load(w: &Weights) -> Result<Self, String> {
        let g = "decoder.generator";
        // ups: ConvTranspose1d(512->256, k=20, s=10, pad=5); (256->128, k=12, s=6, pad=3)
        let ups = vec![
            ConvWn::load_transposed(w, &format!("{g}.ups.0"), 10, 5, 0, 1)?,
            ConvWn::load_transposed(w, &format!("{g}.ups.1"), 6, 3, 0, 1)?,
        ];
        // noise_convs: plain Conv1d. 0: (22->256, k=12, s=6, pad=(6+1)/2=3). 1: (22->128, k=1).
        let noise_convs = vec![
            ConvWn::load_plain(w, &format!("{g}.noise_convs.0"), 6, 3)?,
            ConvWn::load_plain(w, &format!("{g}.noise_convs.1"), 1, 0)?,
        ];
        let noise_res = vec![
            SnakeResBlock::load(w, &format!("{g}.noise_res.0"), 256, 7, [1, 3, 5])?,
            SnakeResBlock::load(w, &format!("{g}.noise_res.1"), 128, 11, [1, 3, 5])?,
        ];
        // MRF resblocks: stage 0 (ch=256): k=3,7,11; stage 1 (ch=128): k=3,7,11.
        let res_k = [3, 7, 11];
        let mut resblocks = Vec::new();
        for (j, &k) in res_k.iter().enumerate() {
            resblocks.push(SnakeResBlock::load(
                w,
                &format!("{g}.resblocks.{j}"),
                256,
                k,
                [1, 3, 5],
            )?);
        }
        for (j, &k) in res_k.iter().enumerate() {
            resblocks.push(SnakeResBlock::load(
                w,
                &format!("{g}.resblocks.{}", 3 + j),
                128,
                k,
                [1, 3, 5],
            )?);
        }
        let conv_post = ConvWn::load(w, &format!("{g}.conv_post"), 1, 3, 1, 1, true)?;

        let (lw, lb) = w.linear(&format!("{g}.m_source.l_linear"))?;
        let lb = lb.ok_or_else(|| "kokoro: m_source.l_linear.bias missing".to_string())?;
        let source = NsfSource::new(lw, lb, HARMONIC_NUM);

        Ok(Self {
            ups,
            noise_convs,
            noise_res,
            resblocks,
            conv_post,
            source,
        })
    }

    /// `x`: `(512, T)` decoder features; `f0_curve`: host F0 at the decoder
    /// frame rate. Returns the host waveform.
    fn forward(
        &self,
        x: &UniquePtr<MlxArray>,
        style: &UniquePtr<MlxArray>,
        f0_curve: &[f32],
    ) -> Result<Vec<f32>, String> {
        // NSF source: upsample F0 to audio rate, generate harmonic excitation,
        // STFT it into the (22, frames) magnitude+phase feature `har`.
        let mut f0_up = Vec::with_capacity(f0_curve.len() * F0_UPSAMPLE as usize);
        for &v in f0_curve {
            for _ in 0..F0_UPSAMPLE {
                f0_up.push(v);
            }
        }
        let har_source = self.source.forward(&f0_up)?;
        let (har_mag, har_phase) = stft::stft(&har_source);
        let har = ops::concat2(&har_mag, &har_phase, 0); // (22, frames)

        let mut x = mlxcel_core::copy(x.as_ref().expect("kokoro: generator input"));
        for i in 0..2 {
            x = ops::leaky_relu(&x, 0.1);
            // Source branch.
            let x_source = self.noise_convs[i].forward(&har);
            let x_source = self.noise_res[i].forward(&x_source, style);
            // Upsample main path.
            x = self.ups[i].forward(&x);
            if i == 1 {
                // Reflection pad (1,0) on the last stage before adding source.
                x = reflect_pad_left(&x);
            }
            let (xa, xb) = align_time(&x, &x_source);
            x = ops::add(&xa, &xb);
            // MRF: average of three resblocks for this stage.
            let base = i * 3;
            let mut acc: Option<UniquePtr<MlxArray>> = None;
            for j in 0..3 {
                let r = self.resblocks[base + j].forward(&x, style);
                acc = Some(match acc {
                    None => r,
                    Some(a) => {
                        let (a, r) = align_time(&a, &r);
                        ops::add(&a, &r)
                    }
                });
            }
            x = ops::div_scalar(&acc.expect("kokoro: empty MRF"), 3.0);
        }
        x = ops::leaky_relu(&x, 0.01);
        x = self.conv_post.forward(&x); // (22, T_out)

        let bins = N_FFT / 2 + 1; // 11
        let mag = ops::exp(&ops::slice(&x, &[0, 0], &[bins, i32::MAX]));
        let phase = ops::sin(&ops::slice(&x, &[bins, 0], &[2 * bins, i32::MAX]));
        stft::istft(&mag, &phase)
    }
}

/// Reflection pad a `(C, T)` activation by one frame on the left.
fn reflect_pad_left(x: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    let shape = ops::shape(x);
    let t = shape[1];
    if t < 2 {
        return mlxcel_core::copy(x.as_ref().expect("kokoro: pad input"));
    }
    // reflect: prepend x[:,1].
    let col = ops::slice(x, &[0, 1], &[shape[0], 2]); // (C,1)
    ops::concat2(&col, x, 1)
}

/// The full decoder.
pub(crate) struct Decoder {
    encode: AdainResBlk1d,
    decode: Vec<AdainResBlk1d>,
    f0_conv: ConvWn,
    n_conv: ConvWn,
    asr_res: ConvWn,
    generator: Generator,
}

impl Decoder {
    pub(crate) fn load(w: &Weights) -> Result<Self, String> {
        let encode = AdainResBlk1d::load(w, "decoder.encode", 514, 1024, false, false)?;
        let decode = vec![
            AdainResBlk1d::load(w, "decoder.decode.0", 1090, 1024, false, true)?,
            AdainResBlk1d::load(w, "decoder.decode.1", 1090, 1024, false, true)?,
            AdainResBlk1d::load(w, "decoder.decode.2", 1090, 1024, false, true)?,
            AdainResBlk1d::load(w, "decoder.decode.3", 1090, 512, true, true)?,
        ];
        let f0_conv = ConvWn::load(w, "decoder.F0_conv", 2, 1, 1, 1, true)?;
        let n_conv = ConvWn::load(w, "decoder.N_conv", 2, 1, 1, 1, true)?;
        let asr_res = ConvWn::load(w, "decoder.asr_res.0", 1, 0, 1, 1, true)?;
        let generator = Generator::load(w)?;
        Ok(Self {
            encode,
            decode,
            f0_conv,
            n_conv,
            asr_res,
            generator,
        })
    }

    /// Decode to a waveform.
    ///
    /// `asr`: `(512, T)`. `f0_pred`/`n_pred`: `(2T,)`. `style`: decoder style
    /// `(1, 128)`.
    pub(crate) fn forward(
        &self,
        asr: &UniquePtr<MlxArray>,
        f0_pred: &UniquePtr<MlxArray>,
        n_pred: &UniquePtr<MlxArray>,
        style: &UniquePtr<MlxArray>,
    ) -> Result<Vec<f32>, String> {
        // Downsample F0/N (2x stride) so they realign with the asr frame rate.
        let f0_in = ops::expand_dims(f0_pred, 0); // (1, 2T)
        let n_in = ops::expand_dims(n_pred, 0);
        let f0 = self.f0_conv.forward(&f0_in); // (1, T)
        let n = self.n_conv.forward(&n_in);

        // Align asr / F0 / N to a common length.
        let (asr_a, f0_a) = align_time(asr, &f0);
        let (asr_a, n_a) = align_time(&asr_a, &n);
        let (f0_a, n_a) = align_time(&f0_a, &n_a);
        let (asr_a, f0_a) = align_time(&asr_a, &f0_a);

        let x = ops::concat(&[&asr_a, &f0_a, &n_a], 0); // (514, T)
        let mut x = self.encode.forward(&x, style); // (1024, T)
        let asr_res = self.asr_res.forward(&asr_a); // (64, T)

        let mut res = true;
        for (bi, blk) in self.decode.iter().enumerate() {
            if res {
                let (x_a, ar) = align_time(&x, &asr_res);
                let (x_a, f0_b) = align_time(&x_a, &f0_a);
                let (x_a, n_b) = align_time(&x_a, &n_a);
                x = ops::concat(&[&x_a, &ar, &f0_b, &n_b], 0); // (1090, T)
            }
            x = blk.forward(&x, style);
            if bi == 3 {
                res = false;
            }
        }

        // NSF harmonic source uses the RAW predictor F0 at full (2T) resolution,
        // matching the upstream reference (the generator upsamples it by 300).
        // `f0_a` above is the conv-downsampled curve used only for the
        // encode/decode feature concat, not the harmonic source.
        let f0_curve = ops::to_vec_f32(f0_pred)?;
        let _ = STYLE_DIM;
        self.generator.forward(&x, style, &f0_curve)
    }
}

/// Symmetric padding `(kernel*dilation - dilation) / 2` for a dilated conv.
fn get_padding(kernel: i32, dilation: i32) -> i32 {
    (kernel * dilation - dilation) / 2
}
