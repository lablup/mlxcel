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

//! MoonViT3D vision encoder (Kimi K3, `vision_config.model_type: "moonvit3d"`).
//!
//! Reference: https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_k3.py
//! (the vision tower) and the checkpoint's `vision_config` (issue #1342).
//!
//! Native-resolution ViT over `[N, 3, p, p]` patches (`p = 14`) pre-cut by
//! `processors::kimi_k3`. Per image, in order:
//!
//! 1. `conv2d(patches -> NHWC, proj.weight [1024, 14, 14, 3], stride 14, no
//!    bias)` flattened to `[t*h*w, 1024]`.
//! 2. Plus the position term `pos3d` from `moonvit3d_pos_emb.rs`: the 64x64
//!    learnable grid bilinearly resampled to `(h, w)` with half-pixel sampling,
//!    and for `t > 1` the sincos time table added per frame.
//! 3. 27 pre-norm blocks: `RMSNorm(eps 2^-7)` -> fused `wqkv` ->
//!    `[L, 3, 12, 128]` -> 2D RoPE on q/k (the MoonViT `rope2d` table, 32
//!    frequencies alternating x/y, applied as interleaved pairs) -> SDPA at
//!    scale `128^-0.5` over all `t*h*w` tokens of the image (no mask) -> `wo`
//!    -> residual -> `RMSNorm` -> `fc1(gelu_tanh(fc0(.)))` -> residual.
//! 4. `RMSNorm` (`final_layernorm`).
//! 5. `sd2_tpool` merge: `[t, h/2, 2, w/2, 2, D]` -> `[t, h/2, w/2, 2, 2, D]`
//!    -> mean over `t` -> `[h/2 * w/2, 4, D]`.
//!
//! Where this differs from MoonViT (`kimi_vl.rs`): RMSNorm at `2^-7` instead
//! of LayerNorm, `qkv_hidden_size` (1536) decoupled from the hidden size
//! (1024) so `head_dim = 1536 / 12 = 128`, bilinear instead of bicubic
//! position resampling, the fixed sincos time term, the temporal mean before
//! the 2x2 merge, and the weight keys (`encoder.blocks.N.{wqkv,wo}` flat
//! under the block, `encoder.final_layernorm`). The 2D RoPE table and the
//! interleaved-pair rotation are the same as MoonViT's and are reused from
//! `kimi_vl::rope`.
//!
//! Attention is per image: each image runs through the whole stack on its
//! own, so images never attend to each other and no block-diagonal mask is
//! needed.
//!
//! Weight layout (rooted at `vision_tower.`):
//! - `patch_embed.proj.weight` `[1024, 14, 14, 3]` (channel-last at load).
//! - `patch_embed.pos_emb.weight` `[64, 64, 1024]`.
//! - `encoder.blocks.{i}.{norm0,norm1}.weight` `[1024]`.
//! - `encoder.blocks.{i}.wqkv.weight` `[4608, 1024]`, `.wo.weight` `[1024, 1536]`.
//! - `encoder.blocks.{i}.mlp.fc0.weight` `[4096, 1024]`, `.fc1.weight` `[1024, 4096]`.
//! - `encoder.final_layernorm.weight` `[1024]`.
//!
//! Used by: `vision::kimi_k3_vl::KimiK3VLModel`.

use super::kimi_vl::rope::{Rope2DPosEmb, apply_rope};
use super::kimi_vl::{KimiMediaGrid, gelu_tanh};
use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

#[path = "moonvit3d_config.rs"]
mod config;
#[path = "moonvit3d_pos_emb.rs"]
pub(crate) mod pos_emb;

pub use config::MoonViT3DConfig;
use pos_emb::DividedFixedPosEmb;
pub use pos_emb::MoonViT3DGrid;

/// `2^-7`, the RMSNorm epsilon of every norm in the tower.
pub const MOONVIT3D_NORM_EPS: f32 = 0.007_812_5;

fn take_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {key}"))
}

fn load_rms_norm(weights: &WeightMap, prefix: &str) -> Result<RMSNorm, String> {
    Ok(RMSNorm::new(
        take_weight(weights, &format!("{prefix}.weight"))?,
        MOONVIT3D_NORM_EPS,
    ))
}

/// Conv2d patch projection plus the 3D position term.
struct PatchEmbed {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    pos_emb: DividedFixedPosEmb,
    patch_size: i32,
    hidden: i32,
}

impl PatchEmbed {
    fn from_weights(
        weights: &WeightMap,
        config: &MoonViT3DConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let wkey = format!("{prefix}.proj.weight");
        let weight = take_weight(weights, &wkey)?;
        let shape = mlxcel_core::array_shape(&weight);
        let p = config.patch_size as i32;
        let hidden = config.vt_hidden_size as i32;
        if shape != [hidden, p, p, 3] {
            return Err(format!(
                "{wkey}: expected the channel-last conv kernel [{hidden}, {p}, {p}, 3], got \
                 {shape:?} (the loader transposes the PyTorch [out, in, kh, kw] layout)"
            ));
        }
        let bias = weights
            .get(&format!("{prefix}.proj.bias"))
            .map(|b| mlxcel_core::copy(b));
        let pos_emb = DividedFixedPosEmb::from_weights(
            weights,
            &format!("{prefix}.pos_emb"),
            config.init_pos_emb_height as i32,
            config.init_pos_emb_width as i32,
            hidden,
        )?;
        Ok(Self {
            weight,
            bias,
            pos_emb,
            patch_size: p,
            hidden,
        })
    }

    /// `patches`: `[t*h*w, p, p, 3]` (channels-last) of one media item.
    /// Returns `[t*h*w, hidden]` with the position term added.
    fn forward(&self, patches: &MlxArray, grid: MoonViT3DGrid) -> UniquePtr<MlxArray> {
        let conv = mlxcel_core::conv2d(
            patches,
            &self.weight,
            self.patch_size,
            self.patch_size,
            0,
            0,
            1,
            1,
            1,
        );
        let n = mlxcel_core::array_shape(patches)[0];
        let mut x = mlxcel_core::reshape(&conv, &[n, self.hidden]);
        if let Some(bias) = &self.bias {
            x = mlxcel_core::add(&x, bias);
        }
        let pos = self.pos_emb.pos_for(grid);
        let pos = mlxcel_core::astype(&pos, mlxcel_core::array_dtype(&x));
        mlxcel_core::add(&x, &pos)
    }
}

struct Attention {
    wqkv: UnifiedLinear,
    wo: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    fn from_weights(
        weights: &WeightMap,
        config: &MoonViT3DConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let wqkv = UnifiedLinear::from_weights(weights, &format!("{prefix}.wqkv"), 64, 4)?;
        let wo = UnifiedLinear::from_weights(weights, &format!("{prefix}.wo"), 64, 4)?;
        let head_dim = config.head_dim() as i32;
        Ok(Self {
            wqkv,
            wo,
            num_heads: config.vt_num_attention_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[L, hidden]` of one image; `cos`/`sin`: `[L, head_dim/2]`.
    fn forward(&self, x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
        let l = mlxcel_core::array_shape(x)[0];
        let dtype = mlxcel_core::array_dtype(x);
        let qkv = self.wqkv.forward(x);
        let qkv = mlxcel_core::reshape(&qkv, &[l, 3, self.num_heads, self.head_dim]);
        let pick = |i: i32| {
            let s = mlxcel_core::slice(
                &qkv,
                &[0, i, 0, 0],
                &[l, i + 1, self.num_heads, self.head_dim],
            );
            mlxcel_core::reshape(&s, &[l, self.num_heads, self.head_dim])
        };
        let (q, k) = apply_rope(&pick(0), &pick(1), cos, sin);
        let v = pick(2);

        // The rotation runs against the f32 angle tables, so it promotes q and
        // k. Cast them back: without this the attention output is f32, `wo`
        // returns f32, the residual add promotes the stream, and every one of
        // the 27 blocks runs its bf16 weights against an f32 activation
        // (`docs/code-guidelines.md`, "Bridge Helpers Return the Input Dtype").
        let q = mlxcel_core::astype(&q, dtype);
        let k = mlxcel_core::astype(&k, dtype);

        // [1, heads, L, head_dim]; full bidirectional attention over the image.
        let to_heads = |a: &MlxArray| {
            let a = mlxcel_core::transpose_axes(a, &[1, 0, 2]);
            mlxcel_core::expand_dims(&a, 0)
        };
        let q = to_heads(&q);
        let k = to_heads(&k);
        let v = to_heads(&v);
        // The fast kernel, not the graph path: the graph form materializes the
        // `[1, heads, L, L]` score matrix, and `L` here is the whole image's
        // patch count. At the navit ceiling of about 67,000 patches that is
        // hundreds of gigabytes for one image, so the graph path is not an
        // option at this width even though the image budget bounds how many
        // such images one request may send.
        let attn = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(&q, &k, &v, self.scale, std::ptr::null())
        };
        let attn = mlxcel_core::reshape(&attn, &[self.num_heads, l, self.head_dim]);
        let attn = mlxcel_core::transpose_axes(&attn, &[1, 0, 2]);
        let attn = mlxcel_core::reshape(&attn, &[l, self.num_heads * self.head_dim]);
        self.wo.forward(&attn)
    }
}

struct Mlp {
    fc0: UnifiedLinear,
    fc1: UnifiedLinear,
}

impl Mlp {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            fc0: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc0"), 64, 4)?,
            fc1: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc1"), 64, 4)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.fc1.forward(&gelu_tanh(&self.fc0.forward(x)))
    }
}

/// `norm0 -> attn -> +resid -> norm1 -> mlp -> +resid`.
struct Block {
    norm0: RMSNorm,
    norm1: RMSNorm,
    attn: Attention,
    mlp: Mlp,
}

impl Block {
    fn from_weights(
        weights: &WeightMap,
        config: &MoonViT3DConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            norm0: load_rms_norm(weights, &format!("{prefix}.norm0"))?,
            norm1: load_rms_norm(weights, &format!("{prefix}.norm1"))?,
            attn: Attention::from_weights(weights, config, prefix)?,
            mlp: Mlp::from_weights(weights, &format!("{prefix}.mlp"))?,
        })
    }

    fn forward(&self, x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
        let h = mlxcel_core::add(x, &self.attn.forward(&self.norm0.forward(x), cos, sin));
        mlxcel_core::add(&h, &self.mlp.forward(&self.norm1.forward(&h)))
    }
}

/// `sd2_tpool`: group each `(kh, kw)` spatial block of one item and average
/// over its frames.
///
/// `x`: `[t*h*w, dim]` (frame-major, row-major within a frame). Returns
/// `[(h/kh) * (w/kw), kh*kw, dim]`; the four channel-stacked vectors of a
/// merged token are ordered `(dy, dx)` row-major within the block.
pub fn tpool_merge(
    x: &MlxArray,
    grid: MoonViT3DGrid,
    merge: (i32, i32),
) -> Result<UniquePtr<MlxArray>, String> {
    let MoonViT3DGrid { t, h, w } = grid;
    let (kh, kw) = merge;
    let shape = mlxcel_core::array_shape(x);
    if shape.len() != 2 || shape[0] != grid.token_count() {
        return Err(format!(
            "tpool_merge: expected [{}, dim] for grid {grid:?}, got {shape:?}",
            grid.token_count()
        ));
    }
    if kh <= 0 || kw <= 0 || h % kh != 0 || w % kw != 0 {
        return Err(format!(
            "tpool_merge: grid ({h}, {w}) is not divisible by the merge kernel ({kh}, {kw})"
        ));
    }
    let dim = shape[1];
    let (mh, mw) = (h / kh, w / kw);
    let x = mlxcel_core::reshape(x, &[t, mh, kh, mw, kw, dim]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]); // [t, mh, mw, kh, kw, dim]
    let x = mlxcel_core::mean_axis(&x, 0, false); // [mh, mw, kh, kw, dim]
    Ok(mlxcel_core::reshape(&x, &[mh * mw, kh * kw, dim]))
}

/// The MoonViT3D tower: patch embedding, 27 blocks, final norm.
pub struct MoonViT3DVisionModel {
    patch_embed: PatchEmbed,
    blocks: Vec<Block>,
    final_layernorm: RMSNorm,
    rope: Rope2DPosEmb,
    patch_size: i32,
    merge: (i32, i32),
}

impl MoonViT3DVisionModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: &MoonViT3DConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        config.validate()?;
        let patch_embed =
            PatchEmbed::from_weights(weights, config, &format!("{prefix}.patch_embed"))?;
        let mut blocks = Vec::with_capacity(config.vt_num_hidden_layers);
        for i in 0..config.vt_num_hidden_layers {
            blocks.push(Block::from_weights(
                weights,
                config,
                &format!("{prefix}.encoder.blocks.{i}"),
            )?);
        }
        let final_layernorm = load_rms_norm(weights, &format!("{prefix}.encoder.final_layernorm"))?;
        Ok(Self {
            patch_embed,
            blocks,
            final_layernorm,
            rope: Rope2DPosEmb::new(config.head_dim() as i32),
            patch_size: config.patch_size as i32,
            merge: config.merge(),
        })
    }

    /// `(kh, kw)` of the spatial merge.
    pub fn merge_kernel(&self) -> (i32, i32) {
        self.merge
    }

    /// Run the tower over the media items, one at a time.
    ///
    /// `patches`: `[N, p, p, 3]` (channels-last), the items' patches
    /// concatenated in order with `N = sum(t*h*w)`. Returns one
    /// `[t*h*w, hidden]` array per item, after the final norm and before the
    /// merge.
    pub fn forward(
        &self,
        patches: &MlxArray,
        grids: &[MoonViT3DGrid],
    ) -> Result<Vec<UniquePtr<MlxArray>>, String> {
        let shape = mlxcel_core::array_shape(patches);
        let total: i32 = grids.iter().map(MoonViT3DGrid::token_count).sum();
        if shape.len() != 4
            || shape[0] != total
            || shape[1] != self.patch_size
            || shape[2] != self.patch_size
            || shape[3] != 3
        {
            return Err(format!(
                "moonvit3d: expected patches [{total}, {p}, {p}, 3] for grids {grids:?}, got {shape:?}",
                p = self.patch_size
            ));
        }
        let mut outputs = Vec::with_capacity(grids.len());
        let mut offset = 0i32;
        for &grid in grids {
            let n = grid.token_count();
            let item = mlxcel_core::slice(
                patches,
                &[offset, 0, 0, 0],
                &[offset + n, self.patch_size, self.patch_size, 3],
            );
            offset += n;

            let rope_grid = if grid.t > 1 {
                KimiMediaGrid::Video {
                    t: grid.t,
                    h: grid.h,
                    w: grid.w,
                }
            } else {
                KimiMediaGrid::Image {
                    h: grid.h,
                    w: grid.w,
                }
            };
            let (cos, sin) = self.rope.cos_sin(&[rope_grid]);
            let mut x = self.patch_embed.forward(&item, grid);
            for block in &self.blocks {
                x = block.forward(&x, &cos, &sin);
            }
            outputs.push(self.final_layernorm.forward(&x));
        }
        Ok(outputs)
    }

    /// [`Self::forward`] followed by [`tpool_merge`] per item, concatenated:
    /// `[sum((h/kh)*(w/kw)), kh*kw, hidden]`.
    pub fn forward_merged(
        &self,
        patches: &MlxArray,
        grids: &[MoonViT3DGrid],
    ) -> Result<UniquePtr<MlxArray>, String> {
        let features = self.forward(patches, grids)?;
        let mut merged = Vec::with_capacity(features.len());
        for (x, &grid) in features.iter().zip(grids) {
            merged.push(tpool_merge(x, grid, self.merge)?);
        }
        let refs: Vec<&MlxArray> = merged.iter().map(|m| m.as_ref().unwrap()).collect();
        Ok(mlxcel_core::concatenate_many(&refs, 0))
    }
}

#[cfg(test)]
#[path = "moonvit3d_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "moonvit3d_real_weights_tests.rs"]
mod real_weights_tests;
