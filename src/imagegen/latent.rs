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

//! Latent patching and packing shared by the FLUX.2-family pipelines.
//!
//! A port of `patchify_latents`, `pack_latents` and `unpack_latents` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/flux2/latent.py>,
//! plus the inverse patch step the VAE applies before decoding.
//!
//! Layouts: unpacked latents are `[B, C, H, W]`; patchified latents fold each
//! 2x2 spatial block into channels, `[B, 4C, H/2, W/2]`; packed latents are the
//! patchified grid flattened to a token sequence, `[B, (H/2)*(W/2), 4C]`.
//!
//! Used by: ErnieImage, LLaDAImage (and FLUX.2 itself).

use anyhow::{Result, bail};
use mlxcel_core::{MlxArray, UniquePtr};

/// Drop a singleton frame axis: `[B, C, 1, H, W]` -> `[B, C, H, W]`.
fn squeeze_frame_axis(latents: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(latents);
    mlxcel_core::reshape(latents, &[shape[0], shape[1], shape[3], shape[4]])
}

/// Fold each 2x2 spatial block into channels: `[B, C, H, W]` -> `[B, 4C, H/2, W/2]`.
///
/// A 5-D input with a singleton frame axis (`[B, C, 1, H, W]`) is accepted.
pub fn patchify_latents(latents: &MlxArray) -> Result<UniquePtr<MlxArray>> {
    let mut shape = mlxcel_core::array_shape(latents);
    let squeezed;
    let latents = if shape.len() == 5 && shape[2] == 1 {
        squeezed = squeeze_frame_axis(latents);
        shape = mlxcel_core::array_shape(&squeezed);
        &*squeezed
    } else {
        latents
    };
    if shape.len() != 4 {
        bail!("Expected latents with ndim=4, got shape={shape:?}");
    }
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    if h % 2 != 0 || w % 2 != 0 {
        bail!("Latent height and width must be even to patchify, got {h}x{w}");
    }
    let x = mlxcel_core::reshape(latents, &[b, c, h / 2, 2, w / 2, 2]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 5, 2, 4]);
    Ok(mlxcel_core::reshape(&x, &[b, c * 4, h / 2, w / 2]))
}

/// The inverse of [`patchify_latents`]: `[B, 4C, H, W]` -> `[B, C, 2H, 2W]`.
pub fn unpatchify_latents(latents: &MlxArray) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(latents);
    if shape.len() != 4 {
        bail!("Expected latents with ndim=4, got shape={shape:?}");
    }
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    if c % 4 != 0 {
        bail!("Patchified latents need a channel count divisible by 4, got {c}");
    }
    let x = mlxcel_core::reshape(latents, &[b, c / 4, 2, 2, h, w]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 1, 4, 2, 5, 3]);
    Ok(mlxcel_core::reshape(&x, &[b, c / 4, h * 2, w * 2]))
}

/// Flatten a `[B, C, H, W]` grid to tokens: `[B, H*W, C]`.
pub fn pack_latents(latents: &MlxArray) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(latents);
    if shape.len() != 4 {
        bail!("Expected latents with ndim=4, got shape={shape:?}");
    }
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let x = mlxcel_core::reshape(latents, &[b, c, h * w]);
    Ok(mlxcel_core::transpose_axes(&x, &[0, 2, 1]))
}

/// Restore the grid from tokens: `[B, H*W, C]` -> `[B, C, H, W]`.
///
/// A 4-D input is already unpacked and is returned as is.
pub fn unpack_latents(
    latents: &MlxArray,
    latent_height: i32,
    latent_width: i32,
) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(latents);
    match shape.len() {
        4 => Ok(mlxcel_core::copy(latents)),
        3 => {
            if shape[1] != latent_height * latent_width {
                bail!(
                    "Packed latents hold {} tokens but a {latent_height}x{latent_width} grid needs {}",
                    shape[1],
                    latent_height * latent_width
                );
            }
            let x =
                mlxcel_core::reshape(latents, &[shape[0], latent_height, latent_width, shape[2]]);
            Ok(mlxcel_core::transpose_axes(&x, &[0, 3, 1, 2]))
        }
        _ => bail!("Expected packed latents with ndim=3, got shape={shape:?}"),
    }
}

#[cfg(test)]
#[path = "latent_tests.rs"]
mod tests;
