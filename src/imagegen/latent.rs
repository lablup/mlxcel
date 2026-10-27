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
//! [`prepare_packed_latents`] and [`prepare_grid_ids`] port the noise and
//! position-id helpers of the same file (`#L38-L71`).
//!
//! Used by: ErnieImage, LLaDAImage, Flux2, Bonsai.

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

/// Initial noise and its grid ids, as upstream `prepare_packed_latents`.
pub struct PackedLatents {
    /// `[batch, (H/16) * (W/16), 128]` noise in the requested dtype.
    pub latents: UniquePtr<MlxArray>,
    /// `[batch, (H/16) * (W/16), 4]` int32 `(t, h, w, layer)` coordinates.
    pub latent_ids: UniquePtr<MlxArray>,
    pub latent_height: i32,
    pub latent_width: i32,
}

/// Draw the initial packed noise for a `height x width` image.
///
/// Matches upstream call for call: one `normal` draw of shape
/// `[batch, num_latent_channels * 4, H/16, W/16]` in f32 from
/// `random.key(seed)`, cast to `dtype`, with grid ids at `t = 0`, then packed.
/// One tensor holds every image of a batch, so image `i` of a batch is not the
/// image `seed + i` would give alone.
///
/// Used by: Flux2, Bonsai.
pub fn prepare_packed_latents(
    seed: u64,
    height: u32,
    width: u32,
    batch: i32,
    num_latent_channels: i32,
    vae_scale_factor: u32,
    dtype: i32,
) -> Result<PackedLatents> {
    if batch < 1 || num_latent_channels < 1 || vae_scale_factor == 0 {
        bail!("prepare_packed_latents needs a positive batch, channel count and VAE scale");
    }
    let height = 2 * (height / (vae_scale_factor * 2));
    let width = 2 * (width / (vae_scale_factor * 2));
    let latent_height = i32::try_from(height / 2)?;
    let latent_width = i32::try_from(width / 2)?;
    if latent_height < 1 || latent_width < 1 {
        bail!("image is too small for a {vae_scale_factor}x VAE with 2x2 patches");
    }
    let key = mlxcel_core::random_key(seed);
    let shape = [batch, num_latent_channels * 4, latent_height, latent_width];
    // SAFETY: `key` is a live array for the duration of the call.
    let noise = unsafe { mlxcel_core::random_normal(&shape, mlxcel_core::dtype::FLOAT32, &*key) };
    let noise = mlxcel_core::astype(&noise, dtype);
    let latent_ids = prepare_grid_ids(&noise, 0)?;
    Ok(PackedLatents {
        latents: pack_latents(&noise)?,
        latent_ids,
        latent_height,
        latent_width,
    })
}

/// `(t_coord, h, w, 0)` int32 coordinates of every cell of a `[B, C, H, W]`
/// grid, row-major: `[B, H*W, 4]`.
///
/// Used by: Flux2, Bonsai.
pub fn prepare_grid_ids(latents: &MlxArray, t_coord: i32) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(latents);
    if shape.len() != 4 {
        bail!("Expected latents with ndim=4, got shape={shape:?}");
    }
    let (batch, height, width) = (shape[0], shape[2], shape[3]);
    let mut coords = Vec::with_capacity((height * width * 4).max(0) as usize);
    for h in 0..height {
        for w in 0..width {
            coords.extend_from_slice(&[t_coord, h, w, 0]);
        }
    }
    let coords = mlxcel_core::from_slice_i32(&coords, &[1, height * width, 4]);
    Ok(mlxcel_core::broadcast_to(
        &coords,
        &[batch, height * width, 4],
    ))
}

#[cfg(test)]
#[path = "latent_tests.rs"]
mod tests;
