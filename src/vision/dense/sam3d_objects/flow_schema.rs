// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Exact checkpoint names and dimensions for both flows and all decoder heads.
use super::{config::Config, flow::LATENTS};
use anyhow::{Result, ensure};
use std::collections::HashMap;

type Shapes = HashMap<String, Vec<usize>>;
#[derive(Default)]
struct Schema {
    shapes: Shapes,
}
impl Schema {
    fn tensor(&mut self, key: String, shape: &[usize]) {
        self.shapes.insert(key, shape.to_vec());
    }
    fn linear(&mut self, p: &str, input: usize, output: usize) {
        self.tensor(format!("{p}.weight"), &[output, input]);
        self.tensor(format!("{p}.bias"), &[output]);
    }
    fn norm(&mut self, p: &str, channels: usize) {
        self.tensor(format!("{p}.weight"), &[channels]);
        self.tensor(format!("{p}.bias"), &[channels]);
    }
    fn conv(&mut self, p: &str, input: usize, output: usize, kernel: usize) {
        self.tensor(
            format!("{p}.weight"),
            &[output, kernel, kernel, kernel, input],
        );
        self.tensor(format!("{p}.bias"), &[output]);
    }
    fn mlp(&mut self, p: &str, channels: usize) {
        self.linear(&format!("{p}.mlp.0"), channels, channels * 4);
        self.linear(&format!("{p}.mlp.2"), channels * 4, channels);
    }
    fn time(&mut self, p: &str, channels: usize) {
        self.linear(&format!("{p}.mlp.0"), 256, channels);
        self.linear(&format!("{p}.mlp.2"), channels, channels);
    }
    fn cross(&mut self, p: &str, channels: usize, context: usize) {
        self.linear(&format!("{p}.to_q"), channels, channels);
        self.linear(&format!("{p}.to_kv"), context, channels * 2);
        self.linear(&format!("{p}.to_out"), channels, channels);
    }
    fn block(&mut self, p: &str, c: &Config, multi: bool) {
        let (channels, heads) = (c.hidden_size, c.num_heads);
        self.linear(&format!("{p}.adaLN_modulation.1"), channels, channels * 6);
        for name in if multi {
            vec![".shape", ".6drotation_normalized"]
        } else {
            vec![""]
        } {
            self.norm(&format!("{p}.norm2{name}"), channels);
            self.cross(&format!("{p}.cross_attn{name}"), channels, c.cond_channels);
            self.mlp(&format!("{p}.mlp{name}"), channels);
            self.linear(
                &format!("{p}.self_attn.to_qkv{name}"),
                channels,
                channels * 3,
            );
            self.linear(&format!("{p}.self_attn.to_out{name}"), channels, channels);
            for norm in ["q_rms_norm", "k_rms_norm"] {
                self.tensor(
                    format!("{p}.self_attn.{norm}{name}.gamma"),
                    &[heads, channels / heads],
                );
            }
        }
    }
    fn sparse_res(&mut self, p: &str, input: usize, output: usize, time: usize) {
        self.norm(&format!("{p}.norm1"), input);
        self.conv(&format!("{p}.conv1.conv"), input, output, 3);
        self.conv(&format!("{p}.conv2.conv"), output, output, 3);
        self.linear(&format!("{p}.emb_layers.1"), time, output * 2);
        if input != output {
            self.linear(&format!("{p}.skip_connection"), input, output);
        }
    }
    fn dense_res(&mut self, p: &str, channels: usize) {
        self.norm(&format!("{p}.norm1"), channels);
        self.norm(&format!("{p}.norm2"), channels);
        self.conv(&format!("{p}.conv1"), channels, channels, 3);
        self.conv(&format!("{p}.conv2"), channels, channels, 3);
    }
    fn structure(&mut self, c: &Config, tokens: usize) {
        let p = "ss_generator";
        for name in ["t_embedder", "d_embedder"] {
            self.time(&format!("{p}.{name}"), c.hidden_size);
        }
        for (name, channels) in LATENTS {
            let p = format!("{p}.latent_mapping.{name}");
            self.linear(&format!("{p}.input_layer"), channels, c.hidden_size);
            self.linear(&format!("{p}.out_layer"), c.hidden_size, channels);
            self.tensor(
                format!("{p}.pos_emb"),
                &[if name == "shape" { tokens } else { 1 }, c.hidden_size],
            );
        }
        for i in 0..c.num_blocks {
            self.block(&format!("{p}.blocks.{i}"), c, true);
        }
    }
    fn latent(&mut self, c: &Config) {
        let p = "slat_generator";
        self.time(&format!("{p}.t_embedder"), c.hidden_size);
        self.linear(
            &format!("{p}.input_layer"),
            c.latent_channels,
            c.io_channels,
        );
        self.linear(&format!("{p}.out_layer"), c.io_channels, c.latent_channels);
        for (name, input, output) in [
            ("input_blocks.0", c.io_channels, c.io_channels),
            ("input_blocks.1", c.io_channels, c.hidden_size),
            ("out_blocks.0", 2 * c.hidden_size, c.io_channels),
            ("out_blocks.1", 2 * c.io_channels, c.io_channels),
        ] {
            self.sparse_res(&format!("{p}.{name}"), input, output, c.hidden_size);
        }
        for i in 0..c.num_blocks {
            self.block(&format!("{p}.blocks.{i}"), c, false);
        }
    }
    fn occupancy(&mut self, c: &Config) {
        let p = "ss_decoder";
        self.conv(
            &format!("{p}.input_layer"),
            c.latent_channels,
            c.structure_channels[0],
            3,
        );
        for i in 0..c.structure_res_blocks {
            self.dense_res(&format!("{p}.middle_block.{i}"), c.structure_channels[0]);
        }
        let mut index = 0;
        for (level, &channels) in c.structure_channels.iter().enumerate() {
            for _ in 0..c.structure_res_blocks {
                self.dense_res(&format!("{p}.blocks.{index}"), channels);
                index += 1;
            }
            if let Some(&next) = c.structure_channels.get(level + 1) {
                self.conv(&format!("{p}.blocks.{index}.conv"), channels, next * 8, 3);
                index += 1;
            }
        }
        let last = c.structure_channels[c.structure_channels.len() - 1];
        self.norm(&format!("{p}.out_layer.0"), last);
        self.conv(&format!("{p}.out_layer.2"), last, 1, 3);
    }
    fn decoder(&mut self, p: &str, c: &Config, count: Option<usize>) {
        let channels = c.decoder_channels;
        self.linear(&format!("{p}.input_layer"), c.latent_channels, channels);
        for i in 0..c.decoder_blocks {
            let p = format!("{p}.blocks.{i}");
            self.linear(&format!("{p}.attn.to_qkv"), channels, channels * 3);
            self.linear(&format!("{p}.attn.to_out"), channels, channels);
            self.mlp(&format!("{p}.mlp"), channels);
        }
        if let Some(count) = count {
            self.linear(&format!("{p}.out_layer"), channels, 14 * count);
            self.tensor(format!("{p}.offset_perturbation"), &[count, 3]);
        } else {
            for (i, (input, output)) in [(channels, channels / 4), (channels / 4, channels / 8)]
                .into_iter()
                .enumerate()
            {
                let p = format!("{p}.upsample.{i}");
                self.norm(&format!("{p}.act_layers.0"), input);
                self.conv(&format!("{p}.out_layers.0.conv"), input, output, 3);
                self.norm(&format!("{p}.out_layers.1"), output);
                self.conv(&format!("{p}.out_layers.3.conv"), output, output, 3);
                self.conv(&format!("{p}.skip_connection.conv"), input, output, 1);
            }
            self.linear(&format!("{p}.out_layer"), channels / 8, 101);
        }
    }
}

/// Names and exact shapes of every parameter consumed by the flow/decoder port.
/// Callers merge the conditioner schema and reject missing or unexpected keys.
pub fn expected(c: &Config) -> Result<Shapes> {
    c.validate()?;
    ensure!(
        c.hidden_size <= i32::MAX as usize / 6
            && c.decoder_channels <= i32::MAX as usize / 4
            && c.io_channels <= i32::MAX as usize / 2,
        "flow dimensions exceed array shape limits"
    );
    let tokens = c
        .latent_resolution
        .checked_pow(3)
        .ok_or_else(|| anyhow::anyhow!("latent token count overflow"))?;
    ensure!(
        tokens <= i32::MAX as usize && c.decoder_channels >= 8,
        "invalid latent or mesh dimensions"
    );
    let mut schema = Schema::default();
    schema.structure(c, tokens);
    schema.latent(c);
    schema.occupancy(c);
    schema.decoder("slat_decoder_gs", c, Some(32));
    schema.decoder("slat_decoder_gs_4", c, Some(4));
    schema.decoder("slat_decoder_mesh", c, None);
    Ok(schema.shapes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_schema_covers_all_released_flow_decoder_tensors() -> Result<()> {
        let shapes = expected(&Config::default())?;
        assert_eq!(shapes.len(), 1867);
        assert_eq!(
            shapes["ss_generator.latent_mapping.shape.pos_emb"],
            [4096, 1024]
        );
        assert_eq!(
            shapes["ss_generator.blocks.0.self_attn.q_rms_norm.shape.gamma"],
            [16, 64]
        );
        assert_eq!(shapes["slat_decoder_gs.out_layer.weight"], [448, 768]);
        assert_eq!(shapes["slat_decoder_gs_4.offset_perturbation"], [4, 3]);
        assert_eq!(
            shapes["slat_decoder_mesh.upsample.1.out_layers.3.conv.weight"],
            [96, 3, 3, 3, 96]
        );
        Ok(())
    }
    #[test]
    fn schema_uses_configured_widths_not_checkpoint_inference() -> Result<()> {
        let c = Config {
            hidden_size: 512,
            cond_channels: 256,
            decoder_channels: 1536,
            ..Config::default()
        };
        let shapes = expected(&c)?;
        assert_eq!(
            shapes["ss_generator.latent_mapping.shape.pos_emb"],
            [4096, 512]
        );
        assert_eq!(
            shapes["slat_generator.blocks.0.cross_attn.to_kv.weight"],
            [1024, 256]
        );
        assert_eq!(
            shapes["slat_generator.blocks.0.self_attn.q_rms_norm.gamma"],
            [16, 32]
        );
        assert_eq!(shapes["slat_decoder_mesh.out_layer.weight"], [101, 192]);
        Ok(())
    }
}
