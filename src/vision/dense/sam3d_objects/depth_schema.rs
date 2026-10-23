// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Configuration-derived on-disk schema for the bundled F32 MoGe-3 model.
use super::super::moge3::Moge3Config;
use anyhow::Result;
use std::collections::HashMap;

type Shapes = HashMap<String, Vec<usize>>;
#[derive(Default)]
struct Schema {
    shapes: Shapes,
}
impl Schema {
    fn tensor(&mut self, p: &str, shape: &[usize]) {
        self.shapes
            .insert(format!("depth_model.{p}"), shape.to_vec());
    }
    fn norm(&mut self, p: &str, channels: usize) {
        self.tensor(&format!("{p}.weight"), &[channels]);
        self.tensor(&format!("{p}.bias"), &[channels]);
    }
    fn linear(&mut self, p: &str, input: usize, output: usize) {
        self.tensor(&format!("{p}.weight"), &[output, input]);
        self.tensor(&format!("{p}.bias"), &[output]);
    }
    fn conv2(&mut self, p: &str, input: usize, output: usize, kernel: usize) {
        self.tensor(&format!("{p}.weight"), &[output, kernel, kernel, input]);
        self.tensor(&format!("{p}.bias"), &[output]);
    }
    fn sparse_res(&mut self, p: &str, channels: usize) {
        self.norm(&format!("{p}.norm1"), channels);
        for name in ["conv1", "conv2"] {
            self.tensor(
                &format!("{p}.{name}.weight"),
                &[channels, 3, 3, 3, channels],
            );
            self.tensor(&format!("{p}.{name}.bias"), &[channels]);
        }
    }
}

/// Exact names/shapes before any MoGe weight sanitizer runs.
///
/// The released Objects bundle already stores ConvTranspose2d weights in MLX
/// `[out, 2, 2, in]` layout. Standalone MoGe also accepts Torch `[in,out,2,2]`
/// through its sanitizer, but that alternate layout is not the bundle format.
pub fn expected(c: &Moge3Config) -> Result<Shapes> {
    c.validate()?;
    let dino = c.encoder.dinov2()?;
    let mut s = Schema::default();
    let (p, d) = ("encoder.backbone", dino.hidden_size);
    s.tensor(&format!("{p}.cls_token"), &[1, 1, d]);
    s.tensor(&format!("{p}.mask_token"), &[1, d]);
    s.tensor(
        &format!("{p}.pos_embed"),
        &[1, (dino.image_size / dino.patch_size).pow(2) + 1, d],
    );
    if dino.num_register_tokens > 0 {
        s.tensor(
            &format!("{p}.register_tokens"),
            &[1, dino.num_register_tokens, d],
        );
    }
    s.conv2(
        &format!("{p}.patch_embed.proj"),
        dino.num_channels,
        d,
        dino.patch_size,
    );
    s.norm(&format!("{p}.norm"), d);
    let intermediate = (d as f32 * dino.mlp_ratio) as usize;
    for i in 0..dino.num_hidden_layers {
        let p = format!("{p}.blocks.{i}");
        s.norm(&format!("{p}.norm1"), d);
        s.norm(&format!("{p}.norm2"), d);
        s.linear(&format!("{p}.attn.qkv"), d, d * 3);
        if !dino.qkv_bias {
            s.shapes.remove(&format!("depth_model.{p}.attn.qkv.bias"));
        }
        s.linear(&format!("{p}.attn.proj"), d, d);
        for name in ["ls1", "ls2"] {
            s.tensor(&format!("{p}.{name}.gamma"), &[d]);
        }
        if dino.use_swiglu_ffn {
            let reduced = (intermediate * 2 / 3).div_ceil(8) * 8;
            s.linear(&format!("{p}.mlp.w12"), d, reduced * 2);
            s.linear(&format!("{p}.mlp.w3"), reduced, d);
        } else {
            s.linear(&format!("{p}.mlp.fc1"), d, intermediate);
            s.linear(&format!("{p}.mlp.fc2"), intermediate, d);
        }
    }
    for name in ["image_mean", "image_std"] {
        s.tensor(&format!("encoder.{name}"), &[1, 1, 1, 3]);
    }
    for i in 0..c.encoder.intermediate_layers.len() {
        s.conv2(
            &format!("encoder.output_projections.{i}"),
            d,
            c.encoder.dim_out,
            1,
        );
    }
    for (p, stack) in [
        ("neck", Some(&c.neck)),
        ("points_head", c.points_head.as_ref()),
        ("mask_head", c.mask_head.as_ref()),
        ("normal_head", c.normal_head.as_ref()),
    ] {
        let Some(stack) = stack else { continue };
        let channels = &stack.dim_res_blocks;
        let levels = channels.len();
        let counts = stack.num_res_blocks.resolve(levels, "num_res_blocks")?;
        let kinds = stack.resamplers.resolve(levels - 1, "resamplers")?;
        let outputs = stack.output_dims();
        for i in 0..levels {
            if let Some(input) = stack.dim_in[i] {
                s.conv2(&format!("{p}.input_blocks.{i}"), input, channels[i], 1);
            }
            if let Some(output) = outputs[i] {
                s.conv2(&format!("{p}.output_blocks.{i}"), channels[i], output, 1);
            }
            let hidden = channels[i] * stack.dim_times_res_block_hidden;
            for j in 0..counts[i] {
                let p = format!("{p}.res_blocks.{i}.{j}");
                if matches!(
                    stack.res_block_in_norm.as_str(),
                    "group_norm" | "layer_norm"
                ) {
                    s.norm(&format!("{p}.layers.0"), channels[i]);
                }
                s.conv2(&format!("{p}.layers.2"), channels[i], hidden, 3);
                if matches!(
                    stack.res_block_hidden_norm.as_str(),
                    "group_norm" | "layer_norm"
                ) {
                    s.norm(&format!("{p}.layers.3"), hidden);
                }
                s.conv2(&format!("{p}.layers.5"), hidden, channels[i], 3);
            }
            if i + 1 < levels {
                let p = format!("{p}.resamplers.{i}");
                if kinds[i] == "conv_transpose" {
                    s.conv2(&format!("{p}.0"), channels[i], channels[i + 1], 2);
                    s.conv2(&format!("{p}.1"), channels[i + 1], channels[i + 1], 3);
                } else {
                    s.conv2(&format!("{p}.1"), channels[i], channels[i + 1], 3);
                }
            }
        }
    }
    if let Some(scale) = &c.scale_head {
        for (i, pair) in scale.dims.windows(2).enumerate() {
            s.linear(&format!("scale_head.{}", i * 2), pair[0], pair[1]);
        }
    }
    if let Some(r) = &c.refiner {
        let channels = &r.model_channels;
        let levels = channels.len();
        let last = channels[levels - 1];
        let encoder = r
            .encoder_blocks_per_level
            .resolve(levels, "encoder_blocks_per_level")?;
        let decoder = r
            .decoder_blocks_per_level
            .resolve(levels - 1, "decoder_blocks_per_level")?;
        for (i, &ch) in channels.iter().enumerate() {
            for j in 0..encoder[i] {
                s.sparse_res(&format!("refiner.down_stages.{i}.{j}"), ch);
            }
            if i + 1 < levels {
                s.linear(
                    &format!("refiner.downsample_blocks.{i}.linear"),
                    ch,
                    channels[i + 1],
                );
            }
        }
        for i in 0..r.bottleneck_blocks {
            s.sparse_res(&format!("refiner.bottleneck_stage.{i}"), last);
        }
        for (i, level) in (1..levels).rev().enumerate() {
            s.linear(
                &format!("refiner.upsample_blocks.{i}.linear"),
                channels[level],
                channels[level - 1],
            );
            for j in 0..decoder[i] {
                s.sparse_res(&format!("refiner.up_stages.{i}.{j}"), channels[level - 1]);
            }
        }
        s.linear("refiner.input_proj", r.in_channels, channels[0]);
        s.linear("refiner.encoder_fuse", r.encoder_channels, last);
        s.linear("refiner.fuse_proj.0", last * 2, last);
        s.linear("refiner.fuse_proj.2", last, last);
        s.linear("refiner.out_proj", channels[0], r.out_channels);
    }
    Ok(s.shapes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bundled_depth_schema_covers607_f32_tensors() -> Result<()> {
        let s = expected(&Moge3Config::default())?;
        assert_eq!(s.len(), 607);
        assert_eq!(s["depth_model.encoder.backbone.pos_embed"], [1, 1370, 1024]);
        assert_eq!(
            s["depth_model.neck.resamplers.0.0.weight"],
            [256, 2, 2, 1024]
        );
        assert_eq!(s["depth_model.refiner.encoder_fuse.weight"], [512, 1026]);
        assert_eq!(
            s["depth_model.refiner.up_stages.3.0.conv1.weight"],
            [32, 3, 3, 3, 32]
        );
        assert_eq!(s["depth_model.encoder.image_mean"], [1, 1, 1, 3]);
        Ok(())
    }
    #[test]
    fn omitted_refiner_has_no_schema_keys() -> Result<()> {
        let c = Moge3Config {
            refiner: None,
            ..Moge3Config::default()
        };
        let s = expected(&c)?;
        assert!(!s.keys().any(|key| key.starts_with("depth_model.refiner.")));
        Ok(())
    }
}
