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

//! `QwenImageTransformer`, the single-stream DiT of Qwen-Image-2.1, a port of
//! `transformer.py` (mlx-vlm `73d366ec`).
//!
//! Text, reference and target tokens run through one stack. Each
//! `<|image_pad|>` run of the prompt is replaced by its reference's VAE
//! tokens (four latent tokens per vision slot) and the target latents follow
//! the prompt. A block-causal mask lets every token see what precedes it and
//! its own image block. With `causal_condition` the target tokens take the
//! real timestep's modulation and every other token the zero timestep's, so
//! nothing before the target changes between steps: an edit runs the whole
//! sequence once and later steps only the target against the stored keys and
//! values ([`super::kv_cache`]).

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::Value;

use super::blocks::{
    AdaLayerNormContinuous, Block, CacheUse, TextProjection, TimestepEmbed, rope_positions,
    rope_tables,
};
use super::config::TransformerConfig;
use super::kv_cache::{ImageShape, KvCache, KvMode};
use crate::imagegen::mage_flow::layers::{Dense, LinearQuant, silu};
use crate::imagegen::quant::{QuantSpec, quantized_prefixes, validate_quantized_weights};

/// The reference images of an edit as the transformer sees them.
pub struct EditLayout<'a> {
    /// `(1, h, w)` latent grid of each reference, in prompt order.
    pub reference_shapes: &'a [ImageShape],
    /// Per prompt token (after `drop_idx`): whether it is an `<|image_pad|>`.
    pub image_pad_mask: &'a [bool],
}

/// The joint sequence before the blocks.
struct Joint {
    hidden: UniquePtr<MlxArray>,
    /// Image block of each token, `-1` for text.
    image_ids: Vec<i32>,
    /// `(start, height, width)` of each image block.
    blocks: Vec<(i32, i32, i32)>,
}

/// Upstream `_remap_transformer`: diffusers names to the module names.
/// Idempotent, so an `mlx_format` checkpoint passes through unchanged.
#[must_use]
pub fn remap_transformer_key(key: &str) -> String {
    key.replace("time_text_embed.timestep_embedder.", "time_text_embed.")
        .replace("modulation.1.", "modulation.0.")
}

/// Upstream `_block_causal_mask` with every key valid: `[1, 1, S, S]` bool.
fn block_causal_mask(image_ids: &[i32]) -> UniquePtr<MlxArray> {
    let s = image_ids.len() as i32;
    let idx = mlxcel_core::arange_i32(0, s, 1);
    let rows = mlxcel_core::reshape(&idx, &[-1, 1]);
    let cols = mlxcel_core::reshape(&idx, &[1, -1]);
    let causal = mlxcel_core::greater_equal(&rows, &cols);
    let ids = mlxcel_core::from_slice_i32(image_ids, &[s]);
    let id_rows = mlxcel_core::reshape(&ids, &[-1, 1]);
    let id_cols = mlxcel_core::reshape(&ids, &[1, -1]);
    let zero = mlxcel_core::from_slice_i32(&[0], &[1]);
    let same_image = mlxcel_core::logical_and(
        &mlxcel_core::equal(&id_rows, &id_cols),
        &mlxcel_core::greater_equal(&id_rows, &zero),
    );
    mlxcel_core::reshape(
        &mlxcel_core::logical_or(&causal, &same_image),
        &[1, 1, s, s],
    )
}

fn bool_mask(values: &[bool]) -> UniquePtr<MlxArray> {
    let ints: Vec<i32> = values.iter().map(|&v| i32::from(v)).collect();
    let ints = mlxcel_core::from_slice_i32(&ints, &[ints.len() as i32]);
    mlxcel_core::astype(&ints, dtype::BOOL)
}

fn take_tokens(x: &MlxArray, start: i32, end: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(x);
    mlxcel_core::slice(x, &[0, start, 0], &[s[0], end, s[2]])
}

pub struct QwenImageTransformer {
    config: TransformerConfig,
    /// Activation dtype: the dtype of the (unquantized) weights.
    dtype: i32,
    time_embed: TimestepEmbed,
    txt_in: TextProjection,
    img_in: Dense,
    modulation: Dense,
    blocks: Vec<Block>,
    norm_out: AdaLayerNormContinuous,
    proj_out: Dense,
}

impl QwenImageTransformer {
    /// Build from remapped weights; `quant` applies to every projection that
    /// carries `.scales` (upstream's `nn.quantize` predicate).
    pub fn from_weights(
        config: TransformerConfig,
        w: &WeightMap,
        quant: Option<LinearQuant>,
    ) -> Result<Self> {
        let eps = config.eps;
        let blocks = (0..config.num_layers)
            .map(|i| {
                Block::load(
                    w,
                    &format!("transformer_blocks.{i}"),
                    config.num_heads,
                    config.head_dim,
                    eps,
                    quant,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        if w.contains_key(&format!(
            "transformer_blocks.{}.attn.to_q.weight",
            config.num_layers
        )) {
            bail!(
                "the Qwen-Image transformer checkpoint has more blocks than num_layers {}",
                config.num_layers
            );
        }
        let dtype = mlxcel_core::array_dtype(
            w.get("txt_in.text_norm.weight")
                .context("Qwen-Image weight not found: txt_in.text_norm.weight")?,
        );
        Ok(Self {
            dtype,
            time_embed: TimestepEmbed::load(w, "time_text_embed", quant)?,
            txt_in: TextProjection::load(w, "txt_in", eps, quant)?,
            img_in: Dense::load(w, "img_in", quant)?,
            modulation: Dense::load(w, "modulation.0", quant)?,
            blocks,
            norm_out: AdaLayerNormContinuous::load(w, "norm_out", eps, quant)?,
            proj_out: Dense::load(w, "proj_out", quant)?,
            config,
        })
    }

    /// Read `<root>/transformer`.
    pub fn load(root: &Path) -> Result<Self> {
        let dir = root.join("transformer");
        let config = TransformerConfig::from_dir(&dir)?;
        let raw = mlxcel_core::weights::load_weights_from_dir(&dir)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("cannot load Qwen-Image weights from {}", dir.display()))?;
        let weights: WeightMap = raw
            .into_iter()
            .map(|(k, v)| (remap_transformer_key(&k), v))
            .collect();
        let quant = read_quantization(&dir, &weights)?;
        // The project's precision rule (`bf16_to_f16_at_load`): f16 on Metal,
        // the checkpoint's bf16 on CUDA unless MLXCEL_CUDA_F16_NORMALIZE opts
        // in. Both pass the seed-fixed parity cases (docs/image-generation.md,
        // "Qwen-Image precision").
        let weights = if crate::models::bf16_to_f16_at_load(quant.is_some(), None) {
            weights
                .into_iter()
                .map(|(k, v)| {
                    let v = if mlxcel_core::array_dtype(&v) == dtype::BFLOAT16 {
                        mlxcel_core::astype(&v, dtype::FLOAT16)
                    } else {
                        v
                    };
                    (k, v)
                })
                .collect()
        } else {
            weights
        };
        Self::from_weights(config, &weights, quant)
    }

    #[must_use]
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// The activation dtype the latents and conditioning are cast to.
    #[must_use]
    pub fn dtype(&self) -> i32 {
        self.dtype
    }

    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Upstream `_joint_inputs`.
    fn joint_inputs(
        &self,
        hidden: &MlxArray,
        encoder: &MlxArray,
        img_shape: ImageShape,
        edit: Option<&EditLayout<'_>>,
    ) -> Result<Joint> {
        let txt_len = mlxcel_core::array_shape(encoder)[1];
        let no_slots;
        let (references, slots): (&[ImageShape], &[bool]) = match edit {
            Some(e) => (e.reference_shapes, e.image_pad_mask),
            None => {
                no_slots = vec![false; txt_len as usize];
                (&[], &no_slots)
            }
        };
        ensure!(
            slots.len() == txt_len as usize,
            "image_pad_mask must describe one prompt's image slots"
        );
        let shapes: Vec<ImageShape> = references.iter().copied().chain([img_shape]).collect();
        ensure!(
            shapes.iter().all(|&(f, _, _)| f == 1),
            "Qwen-Image supports single-frame images only"
        );
        let total: i32 = shapes.iter().map(|&(f, h, w)| f * h * w).sum();
        ensure!(
            mlxcel_core::array_shape(hidden)[1] == total,
            "Image latent count does not match image shapes"
        );
        let text = self.txt_in.forward(encoder);
        let images = self.img_in.forward(hidden);
        let mut pieces = Vec::new();
        let (mut image_ids, mut blocks) = (Vec::new(), Vec::new());
        let (mut cursor, mut image_cursor, mut seq_len) = (0usize, 0i32, 0i32);
        for (image_id, &(_, height, width)) in shapes.iter().enumerate() {
            let count = height * width;
            let (start, end) = if image_id == references.len() {
                ensure!(
                    !slots[cursor..].iter().any(|&s| s),
                    "Extra image slots without reference latents"
                );
                (txt_len as usize, txt_len as usize)
            } else {
                ensure!(
                    count % 4 == 0,
                    "Reference images must have dimensions divisible by 32"
                );
                let start = slots[cursor..]
                    .iter()
                    .position(|&s| s)
                    .map(|p| p + cursor)
                    .context("Missing reference-image slots in prompt")?;
                let end = start + (count / 4) as usize;
                ensure!(
                    end <= txt_len as usize && slots[start..end].iter().all(|&s| s),
                    "Reference-image slots do not match VAE dimensions"
                );
                (start, end)
            };
            if start > cursor {
                pieces.push(take_tokens(&text, cursor as i32, start as i32));
                image_ids.extend(std::iter::repeat_n(-1, start - cursor));
                seq_len += (start - cursor) as i32;
            }
            pieces.push(take_tokens(&images, image_cursor, image_cursor + count));
            image_ids.extend(std::iter::repeat_n(image_id as i32, count as usize));
            blocks.push((seq_len, height, width));
            seq_len += count;
            image_cursor += count;
            cursor = end;
        }
        let refs: Vec<&MlxArray> = pieces.iter().map(|p| &**p).collect();
        Ok(Joint {
            hidden: mlxcel_core::concatenate_many(&refs, 1),
            image_ids,
            blocks,
        })
    }

    /// Velocity of the target latents `hidden` (`[1, L, in_channels]`; in an
    /// uncached edit the reference latents come first) at timestep `t`
    /// (`[1]`, already divided by 1000 as upstream's pipeline does).
    ///
    /// `cache` is `(cache, Extract)` on the first step of a cached edit and
    /// `(cache, Cached)` after it, when `hidden` holds only the target.
    pub fn forward(
        &self,
        hidden: &MlxArray,
        encoder: &MlxArray,
        t: &MlxArray,
        img_shape: ImageShape,
        edit: Option<&EditLayout<'_>>,
        mut cache: Option<(&mut KvCache, KvMode)>,
    ) -> Result<UniquePtr<MlxArray>> {
        let target_len = img_shape.0 * img_shape.1 * img_shape.2;
        if let Some((kv, _)) = &cache {
            ensure!(
                self.config.causal_condition,
                "KV caching requires causal_condition=True"
            );
            ensure!(
                kv.layers.len() == self.blocks.len(),
                "KV cache layer count does not match the transformer"
            );
        }
        let mode = cache.as_ref().map(|(_, m)| *m);
        let mut prefix_len = 0;
        let (mut h, cos, sin, mask, target_mask);
        if mode == Some(KvMode::Cached) {
            let (kv, _) = cache
                .as_ref()
                .unwrap_or_else(|| unreachable!("mode is set"));
            let Some(shape) = kv.target_shape else {
                bail!("Qwen-Image KV cache has not been populated");
            };
            ensure!(
                shape == img_shape
                    && kv.input_shape.as_deref() == Some(&mlxcel_core::array_shape(hidden)[..])
                    && kv.dtype == Some(mlxcel_core::array_dtype(hidden)),
                "Cached calls require the original target shape and dtype"
            );
            h = self.img_in.forward(hidden);
            cos = mlxcel_core::copy(kv.cos.as_deref().context("KV cache has no RoPE rows")?);
            sin = mlxcel_core::copy(kv.sin.as_deref().context("KV cache has no RoPE rows")?);
            mask = None;
            target_mask = Some(mlxcel_core::ones(&[target_len], dtype::BOOL));
        } else {
            let joint = self.joint_inputs(hidden, encoder, img_shape, edit)?;
            let seq_len = joint.image_ids.len() as i32;
            let positions = rope_positions(seq_len, &joint.blocks);
            (cos, sin) = rope_tables(&positions, self.config.axes_dims_rope);
            mask = Some(block_causal_mask(&joint.image_ids));
            let target_id = edit.map_or(0, |e| e.reference_shapes.len() as i32);
            target_mask = self.config.causal_condition.then(|| {
                bool_mask(
                    &joint
                        .image_ids
                        .iter()
                        .map(|&i| i == target_id)
                        .collect::<Vec<_>>(),
                )
            });
            h = joint.hidden;
            if let Some((kv, KvMode::Extract)) = cache.as_mut() {
                kv.clear();
                prefix_len = seq_len - target_len;
                kv.prefix_len = prefix_len;
                kv.target_shape = Some(img_shape);
                let hs = mlxcel_core::array_shape(hidden);
                kv.input_shape = Some(vec![hs[0], target_len, hs[2]]);
                kv.dtype = Some(mlxcel_core::array_dtype(hidden));
                let rows = |a: &MlxArray| {
                    let s = mlxcel_core::array_shape(a);
                    mlxcel_core::contiguous(
                        &mlxcel_core::slice(a, &[seq_len - target_len, 0], &[s[0], s[1]]),
                        false,
                    )
                };
                kv.cos = Some(rows(&cos));
                kv.sin = Some(rows(&sin));
            }
        }
        let dt = mlxcel_core::array_dtype(&h);
        let mut timestep = mlxcel_core::astype(t, dt);
        if self.config.causal_condition {
            let zero = mlxcel_core::astype(&mlxcel_core::from_slice_f32(&[0.0], &[1]), dt);
            timestep = mlxcel_core::concatenate(&timestep, &zero, 0);
        }
        let temb = self.time_embed.forward(&timestep, dt);
        let modulation = self.modulation.forward(&silu(&temb));
        for (index, block) in self.blocks.iter().enumerate() {
            let layer = cache.as_mut().map(|(kv, mode)| CacheUse {
                layer: &mut kv.layers[index],
                mode: *mode,
                prefix_len,
            });
            h = block.forward(
                &h,
                &modulation,
                &cos,
                &sin,
                mask.as_deref(),
                target_mask.as_deref(),
                layer,
            )?;
        }
        let s = mlxcel_core::array_shape(&h);
        let h = take_tokens(&h, s[1] - target_len, s[1]);
        let target_mask = target_mask.map(|m| {
            let n = mlxcel_core::array_shape(&m)[0];
            mlxcel_core::slice(&m, &[n - target_len], &[n])
        });
        Ok(self
            .proj_out
            .forward(&self.norm_out.forward(&h, &temb, target_mask.as_deref())))
    }
}

/// Upstream `_read_quant`: the `quantization` object of `config.json`, and
/// a `.scales` without one is an error.
fn read_quantization(dir: &Path, weights: &WeightMap) -> Result<Option<LinearQuant>> {
    let path = dir.join("config.json");
    let config: Value = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?,
    )
    .with_context(|| format!("cannot parse {}", path.display()))?;
    let quantized = !quantized_prefixes(weights).is_empty();
    let Some(q) = config.get("quantization").and_then(Value::as_object) else {
        ensure!(
            !quantized,
            "the Qwen-Image transformer has quantized weights but no `quantization` in config.json"
        );
        return Ok(None);
    };
    let int = |key: &str| -> Result<i32> {
        q.get(key)
            .and_then(Value::as_i64)
            .and_then(|v| i32::try_from(v).ok())
            .with_context(|| format!("Qwen-Image transformer quantization is missing `{key}`"))
    };
    let mode = q.get("mode").and_then(Value::as_str).unwrap_or("affine");
    let spec = QuantSpec::validated("Qwen-Image", mode, int("bits")?, int("group_size")?)?;
    if !quantized {
        return Ok(None);
    }
    validate_quantized_weights(weights, &spec)?;
    LinearQuant::from_spec(&spec).map(Some)
}

#[cfg(test)]
#[path = "transformer_tests.rs"]
mod tests;
