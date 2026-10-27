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

//! Mage-VAE (`MageVAE` in mlx-vlm `73d366ec` `vae.py`): a 16x-downsampling,
//! 128-channel image autoencoder whose decoder is itself a one-step
//! denoiser. Activations are NHWC; the encoder loads only for edit variants.

mod blocks;
mod decoder;
mod encoder;

use anyhow::{Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use self::decoder::{CoDDecoder, DConvDenoiser, PATCH};
use self::encoder::DConvEncoder;
use super::layers::scalar_like;

/// Latent channels.
pub const LATENT_CHANNELS: i32 = 128;

pub struct MageVae {
    encoder: Option<DConvEncoder>,
    cod_decoder: CoDDecoder,
    denoiser: DConvDenoiser,
}

impl MageVae {
    /// Build from sanitized weights (see
    /// [`super::weights::sanitize_vae_weights`]).
    pub fn from_weights(w: &WeightMap, include_encoder: bool) -> Result<Self> {
        let encoder = if include_encoder {
            Some(DConvEncoder::load(w, "dconv_encoder")?)
        } else {
            None
        };
        Ok(Self {
            encoder,
            cod_decoder: CoDDecoder::load(w, "decoder_model.y_embedder.decoder")?,
            denoiser: DConvDenoiser::load(w, "decoder_model")?,
        })
    }

    /// Latent `[B, h, w, 128]` to pixels `[B, 16h, 16w, 3]` (about `[-1, 1]`).
    pub fn decode(&self, latent: &MlxArray) -> UniquePtr<MlxArray> {
        self.denoiser.forward(&self.cod_decoder.forward(latent))
    }

    /// The posterior mean and clipped log-variance of `image`
    /// (`[B, H, W, 3]`, `H` and `W` multiples of 16).
    pub fn encode_moments(
        &self,
        image: &MlxArray,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let Some(encoder) = &self.encoder else {
            bail!("MageVAE was loaded without encoder weights");
        };
        let s = mlxcel_core::array_shape(image);
        if s.len() != 4 || s[1] % PATCH != 0 || s[2] % PATCH != 0 {
            bail!("image height and width must be multiples of 16, got {s:?}");
        }
        Ok(encoder.forward(image))
    }

    /// `MageVAE.encode`: the mean, or with `sample_posterior` a draw
    /// `mean + exp(0.5 * logvar) * normal(key)` in the mean's dtype.
    pub fn encode(
        &self,
        image: &MlxArray,
        sample_posterior: bool,
        key: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>> {
        let (mean, logvar) = self.encode_moments(image)?;
        if !sample_posterior {
            return Ok(mean);
        }
        let shape = mlxcel_core::array_shape(&mean);
        let dt = mlxcel_core::array_dtype(&mean);
        // SAFETY: `key` is a live array borrowed for the call.
        let noise = unsafe { mlxcel_core::random_normal(&shape, dt, key) };
        let std = mlxcel_core::exp(&mlxcel_core::multiply(&scalar_like(0.5, &logvar), &logvar));
        Ok(mlxcel_core::add(
            &mean,
            &mlxcel_core::multiply(&std, &noise),
        ))
    }
}

#[cfg(test)]
#[path = "vae_tests.rs"]
mod tests;
