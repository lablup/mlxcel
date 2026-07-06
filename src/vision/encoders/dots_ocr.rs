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

//! `dots_vit` vision tower for dots.ocr (`vision_tower.*`).
//!
//! A 42-block dynamic-resolution ViT with packed variable-length attention
//! (`cu_seqlens`, block-diagonal over images) and 2D vision RoPE, sharing the
//! helpers in [`super::qwen2_vl`]. It differs from Qwen2-VL in the norm family
//! (RMSNorm blocks + patchifier norm + `post_trunk_norm`), the vision MLP
//! (SwiGLU), and the patch-embed prefix (`patch_embed.patchifier.*`, with a
//! bias and a trailing RMSNorm). The merger projects each 2x2 patch block to
//! the text hidden width.
//!
//! Reference: mlx-vlm `mlx_vlm/models/dots_ocr/` (`DotsVisionTransformer`).
//! Layout convention: activations are `(tokens, C)`; linear weights are
//! `(out_features, in_features)`.

use super::VisionEncoderOutput;
use super::qwen2_vl::{VisionRotaryEmbedding, apply_rotary_pos_emb_vision, concat_many};
use mlxcel_core::layers::{LayerNorm, RMSNorm, UnifiedLinear, attention_from_ptr};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

fn default_embed_dim() -> i32 {
    1536
}
fn default_intermediate() -> i32 {
    4224
}
fn default_layers() -> usize {
    42
}
fn default_heads() -> i32 {
    12
}
fn default_patch_size() -> i32 {
    14
}
fn default_temporal_patch_size() -> i32 {
    1
}
fn default_merge() -> i32 {
    2
}
fn default_rms_eps() -> f32 {
    1e-5
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct DotsVisionConfig {
    #[serde(default = "default_embed_dim")]
    pub embed_dim: i32,
    #[serde(default = "default_intermediate")]
    pub intermediate_size: i32,
    #[serde(default = "default_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_patch_size")]
    pub patch_size: i32,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: i32,
    #[serde(default = "default_merge")]
    pub spatial_merge_size: i32,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_true")]
    pub post_norm: bool,
    /// Group size / bits for quantized exports; ignored for bf16 towers
    /// (`UnifiedLinear` falls back to a plain linear when no `.scales` exist).
    #[serde(default)]
    pub quant_group_size: i32,
    #[serde(default)]
    pub quant_bits: i32,
}

fn get(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("dots_vit weight missing: {name}"))
}

fn load_rms(weights: &WeightMap, prefix: &str, eps: f32) -> Result<RMSNorm, String> {
    Ok(RMSNorm::new(
        get(weights, &format!("{prefix}.weight"))?,
        eps,
    ))
}

// Patch embed: 14x14 conv-as-linear (with bias) + trailing RMSNorm.
struct PatchEmbed {
    proj: UnifiedLinear,
    norm: RMSNorm,
}

impl PatchEmbed {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &DotsVisionConfig,
    ) -> Result<Self, String> {
        // The checkpoint stores the projection as a 4-D conv weight. Normalize it
        // to a 2-D `(out, C*pH*pW)` linear whose column order is `(c, dy, dx)`,
        // matching the processor's per-row feature layout. Layouts seen in the
        // wild: pre-flattened `(out, 588)`; OIHW `(out, C, pH, pW)`; and the
        // channels-last OHWI `(out, pH, pW, C)` that the 4-bit export ships.
        let key = format!("{prefix}.patchifier.proj.weight");
        let raw = get(weights, &key)?;
        let shape = mlxcel_core::array_shape(&raw);
        let out = config.embed_dim;
        let (c, p) = (3, config.patch_size);
        let weight = match shape.as_slice() {
            [_, _] => raw,
            [_, a, _, _] if *a == c => mlxcel_core::reshape(&raw, &[out, c * p * p]),
            [_, _, _, a] if *a == c => {
                let t = mlxcel_core::transpose_axes(&raw, &[0, 3, 1, 2]);
                mlxcel_core::reshape(&t, &[out, c * p * p])
            }
            _ => return Err(format!("unexpected patch proj shape {shape:?}")),
        };
        let bias = weights
            .get(&format!("{prefix}.patchifier.proj.bias"))
            .map(|w| mlxcel_core::copy(w));
        let proj = UnifiedLinear::Regular(mlxcel_core::layers::Linear::new(weight, bias));
        let norm = load_rms(
            weights,
            &format!("{prefix}.patchifier.norm"),
            config.rms_norm_eps,
        )?;
        Ok(Self { proj, norm })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.norm.forward(&self.proj.forward(x))
    }
}

struct DotsVisionAttention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl DotsVisionAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &DotsVisionConfig,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = config.embed_dim / config.num_attention_heads;
        Ok(Self {
            qkv: UnifiedLinear::from_weights(weights, &format!("{prefix}.qkv"), gs, bits)?,
            proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.proj"), gs, bits)?,
            num_heads: config.num_attention_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `(S, dim)`; block-diagonal attention over `cu_seqlens` segments.
    fn forward(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let seq_length = mlxcel_core::array_shape(x)[0];
        let (heads, hd) = (self.num_heads, self.head_dim);

        let qkv = self.qkv.forward(x); // (S, 3*dim)
        let qkv = mlxcel_core::reshape(&qkv, &[seq_length, 3, heads, hd]);
        let qkv = mlxcel_core::transpose_axes(&qkv, &[1, 0, 2, 3]); // (3, S, heads, hd)
        let pick = |i: i32| {
            let sl = mlxcel_core::slice(&qkv, &[i, 0, 0, 0], &[i + 1, seq_length, heads, hd]);
            mlxcel_core::squeeze_axis(&sl, 0) // (S, heads, hd)
        };
        let q = apply_rotary_pos_emb_vision(&pick(0), rotary_pos_emb);
        let k = apply_rotary_pos_emb_vision(&pick(1), rotary_pos_emb);
        let v = pick(2);

        // (S, heads, hd) -> (1, heads, S, hd)
        let to_attn = |a: &MlxArray| {
            let a = mlxcel_core::transpose_axes(a, &[1, 0, 2]);
            mlxcel_core::expand_dims(&a, 0)
        };
        let q = to_attn(&q);
        let k = to_attn(&k);
        let v = to_attn(&v);

        let num_segments = cu_seqlens.len() - 1;
        let mut outs = Vec::with_capacity(num_segments);
        for seg in 0..num_segments {
            let (start, end) = (cu_seqlens[seg], cu_seqlens[seg + 1]);
            let sl = |a: &MlxArray| mlxcel_core::slice(a, &[0, 0, start, 0], &[1, heads, end, hd]);
            let (qs, ks, vs) = (sl(&q), sl(&k), sl(&v));
            // SAFETY: q/k/v segments valid; null mask (full attention within segment).
            let attn =
                unsafe { attention_from_ptr(&qs, &ks, &vs, self.scale, std::ptr::null(), 0.0, 0) };
            outs.push(attn);
        }
        let out = if outs.len() == 1 {
            outs.into_iter().next().unwrap()
        } else {
            concat_many(&outs, 2)
        };
        let out = mlxcel_core::squeeze_axis(&out, 0); // (heads, S, hd)
        let out = mlxcel_core::transpose_axes(&out, &[1, 0, 2]); // (S, heads, hd)
        let out = mlxcel_core::reshape(&out, &[seq_length, -1]);
        self.proj.forward(&out)
    }
}

// SwiGLU vision MLP: fc2(silu(fc1(x)) * fc3(x)).
struct DotsVisionMlp {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
    fc3: UnifiedLinear,
}

impl DotsVisionMlp {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        Ok(Self {
            fc1: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc1"), gs, bits)?,
            fc2: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc2"), gs, bits)?,
            fc3: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc3"), gs, bits)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let g = mlxcel_core::utils::silu(&self.fc1.forward(x));
        let u = self.fc3.forward(x);
        self.fc2.forward(&mlxcel_core::multiply(&g, &u))
    }
}

struct VisionBlock {
    norm1: RMSNorm,
    attn: DotsVisionAttention,
    norm2: RMSNorm,
    mlp: DotsVisionMlp,
}

impl VisionBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &DotsVisionConfig,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            norm1: load_rms(weights, &format!("{prefix}.norm1"), config.rms_norm_eps)?,
            attn: DotsVisionAttention::from_weights(
                weights,
                &format!("{prefix}.attn"),
                config,
                gs,
                bits,
            )?,
            norm2: load_rms(weights, &format!("{prefix}.norm2"), config.rms_norm_eps)?,
            mlp: DotsVisionMlp::from_weights(weights, &format!("{prefix}.mlp"), gs, bits)?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let y = self
            .attn
            .forward(&self.norm1.forward(x), cu_seqlens, rotary_pos_emb);
        let x = mlxcel_core::add(x, &y);
        let y = self.mlp.forward(&self.norm2.forward(&x));
        mlxcel_core::add(&x, &y)
    }
}

// Merger: LayerNorm -> group 2x2 patches -> Linear -> GELU -> Linear.
struct PatchMerger {
    ln_q: LayerNorm,
    mlp_0: UnifiedLinear,
    mlp_2: UnifiedLinear,
    hidden_size: i32,
}

impl PatchMerger {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &DotsVisionConfig,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let ln_q = LayerNorm::new(
            get(weights, &format!("{prefix}.ln_q.weight"))?,
            weights
                .get(&format!("{prefix}.ln_q.bias"))
                .map(|w| mlxcel_core::copy(w)),
            1e-6,
        );
        let merge = config.spatial_merge_size;
        Ok(Self {
            ln_q,
            mlp_0: UnifiedLinear::from_weights(weights, &format!("{prefix}.mlp.0"), gs, bits)?,
            mlp_2: UnifiedLinear::from_weights(weights, &format!("{prefix}.mlp.2"), gs, bits)?,
            hidden_size: config.embed_dim * merge * merge,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.ln_q.forward(x);
        let h = mlxcel_core::reshape(&h, &[-1, self.hidden_size]);
        let h = self.mlp_0.forward(&h);
        let h = mlxcel_core::gelu(&h);
        self.mlp_2.forward(&h)
    }
}

pub struct DotsVisionEncoder {
    patch_embed: PatchEmbed,
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: Vec<VisionBlock>,
    post_trunk_norm: Option<RMSNorm>,
    merger: PatchMerger,
    spatial_merge_size: i32,
}

impl DotsVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &DotsVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let (gs, bits) = (config.quant_group_size, config.quant_bits);
        let head_dim = config.embed_dim / config.num_attention_heads;
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            blocks.push(VisionBlock::from_weights(
                weights,
                &format!("{prefix}.blocks.{i}"),
                config,
                gs,
                bits,
            )?);
        }
        let post_trunk_norm = if config.post_norm {
            Some(load_rms(
                weights,
                &format!("{prefix}.post_trunk_norm"),
                config.rms_norm_eps,
            )?)
        } else {
            None
        };
        Ok(Self {
            patch_embed: PatchEmbed::from_weights(
                weights,
                &format!("{prefix}.patch_embed"),
                config,
            )?,
            rotary_pos_emb: VisionRotaryEmbedding::new((head_dim / 2) as usize),
            blocks,
            post_trunk_norm,
            merger: PatchMerger::from_weights(
                weights,
                &format!("{prefix}.merger"),
                config,
                gs,
                bits,
            )?,
            spatial_merge_size: config.spatial_merge_size,
        })
    }

    /// 2D rotary position ids in merge-block-grouped order (matches the
    /// processor emission and the merger's consecutive-4 grouping).
    fn rot_pos_emb(&self, grid_thw: &[(i32, i32, i32)]) -> UniquePtr<MlxArray> {
        let merge = self.spatial_merge_size;
        let mut all_pos_ids: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut max_grid_dim = 0i32;
        for &(t, h, w) in grid_thw {
            max_grid_dim = max_grid_dim.max(h).max(w);
            let h_arange = mlxcel_core::arange_i32(0, h, 1);
            let h_col = mlxcel_core::reshape(&h_arange, &[h, 1]);
            let hpos = mlxcel_core::repeat(&h_col, w, 1);
            let hpos = mlxcel_core::reshape(&hpos, &[h / merge, merge, w / merge, merge]);
            let hpos = mlxcel_core::transpose_axes(&hpos, &[0, 2, 1, 3]);
            let hpos = mlxcel_core::flatten(&hpos);

            let w_arange = mlxcel_core::arange_i32(0, w, 1);
            let w_row = mlxcel_core::reshape(&w_arange, &[1, w]);
            let wpos = mlxcel_core::repeat(&w_row, h, 0);
            let wpos = mlxcel_core::reshape(&wpos, &[h / merge, merge, w / merge, merge]);
            let wpos = mlxcel_core::transpose_axes(&wpos, &[0, 2, 1, 3]);
            let wpos = mlxcel_core::flatten(&wpos);

            let stacked = mlxcel_core::stack_owned(&[hpos, wpos], -1);
            all_pos_ids.push(mlxcel_core::tile(&stacked, &[t, 1]));
        }
        let pos_ids = if all_pos_ids.len() == 1 {
            all_pos_ids.into_iter().next().unwrap()
        } else {
            concat_many(&all_pos_ids, 0)
        };

        let rotary_table = self.rotary_pos_emb.forward(max_grid_dim);
        let total_tokens = mlxcel_core::array_shape(&pos_ids)[0];
        let pos_ids_flat = mlxcel_core::flatten(&pos_ids);
        let all_freqs = mlxcel_core::take(&rotary_table, &pos_ids_flat, 0);
        let half_dim = mlxcel_core::array_shape(&all_freqs)[1];
        let all_freqs = mlxcel_core::reshape(&all_freqs, &[total_tokens, 2, half_dim]);
        mlxcel_core::reshape(&all_freqs, &[total_tokens, 2 * half_dim])
    }

    fn compute_cu_seqlens(grid_thw: &[(i32, i32, i32)]) -> Vec<i32> {
        let mut cu = vec![0i32];
        let mut acc = 0i32;
        for &(t, h, w) in grid_thw {
            for _ in 0..t {
                acc += h * w;
                cu.push(acc);
            }
        }
        cu
    }

    /// `hidden_states`: `(total_patch_rows, C*pH*pW)`; returns merged features
    /// `(sum t*(h/2)*(w/2), text_hidden)`.
    pub fn forward_with_grid(
        &self,
        hidden_states: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> VisionEncoderOutput {
        let mut h = self.patch_embed.forward(hidden_states);
        let rotary_pos_emb = self.rot_pos_emb(grid_thw);
        let cu_seqlens = Self::compute_cu_seqlens(grid_thw);
        for block in &self.blocks {
            h = block.forward(&h, &cu_seqlens, &rotary_pos_emb);
        }
        if let Some(norm) = &self.post_trunk_norm {
            h = norm.forward(&h);
        }
        h = self.merger.forward(&h);
        VisionEncoderOutput { hidden_states: h }
    }
}
