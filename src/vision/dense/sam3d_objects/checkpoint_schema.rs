// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Exact configuration-derived main-model checkpoint schema.
use super::config::Config;
use anyhow::{Context, Result, ensure};
use mlxcel_core::{self as mx, weights::WeightMap};
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    io::Read,
    path::{Component, Path},
};
pub type Schema = HashMap<String, Vec<usize>>;
fn linear(s: &mut Schema, p: &str, output: usize, input: usize, bias: bool) {
    s.insert(format!("{p}.weight"), vec![output, input]);
    if bias {
        s.insert(format!("{p}.bias"), vec![output]);
    }
}
fn norm(s: &mut Schema, p: &str, channels: usize) {
    for suffix in ["weight", "bias"] {
        s.insert(format!("{p}.{suffix}"), vec![channels]);
    }
}
fn block(s: &mut Schema, p: &str, d: usize, expansion: usize, scale: bool) {
    for n in ["norm1", "norm2"] {
        norm(s, &format!("{p}.{n}"), d);
    }
    linear(s, &format!("{p}.attn.qkv"), 3 * d, d, true);
    linear(s, &format!("{p}.attn.proj"), d, d, true);
    linear(s, &format!("{p}.mlp.fc1"), expansion * d, d, true);
    linear(s, &format!("{p}.mlp.fc2"), d, expansion * d, true);
    if scale {
        for n in ["ls1", "ls2"] {
            s.insert(format!("{p}.{n}.gamma"), vec![d]);
        }
    }
}
pub fn expected(c: &Config) -> Result<Schema> {
    c.validate()?;
    let mut s = super::flow_schema::expected(c)?;
    let d = c.dino_hidden_size;
    for (p, pointmap) in [
        ("ss_condition_embedder", true),
        ("slat_condition_embedder", false),
    ] {
        s.insert(format!("{p}.idx_emb"), vec![3, c.cond_channels]);
        for i in 0..2 {
            let b = format!("{p}.module_list.{i}.backbone");
            s.insert(format!("{b}.cls_token"), vec![1, 1, d]);
            s.insert(format!("{b}.mask_token"), vec![1, d]);
            s.insert(format!("{b}.register_tokens"), vec![1, 4, d]);
            s.insert(
                format!("{b}.pos_embed"),
                vec![1, (c.dino_image_size / 14).pow(2) + 1, d],
            );
            s.insert(format!("{b}.patch_embed.proj.weight"), vec![d, 14, 14, 3]);
            s.insert(format!("{b}.patch_embed.proj.bias"), vec![d]);
            norm(&mut s, &format!("{b}.norm"), d);
            for j in 0..c.dino_layers {
                block(&mut s, &format!("{b}.blocks.{j}"), d, 4, true);
            }
        }
        for i in 0..if pointmap { 3 } else { 2 } {
            let input = if i == 2 { c.point_channels } else { d };
            let p = format!("{p}.projection_nets.{i}");
            let hidden = (8 * c.cond_channels / 3).div_ceil(256) * 256;
            norm(&mut s, &format!("{p}.0"), input);
            linear(&mut s, &format!("{p}.1.w1"), hidden, input, false);
            linear(&mut s, &format!("{p}.1.w2"), c.cond_channels, hidden, false);
            linear(&mut s, &format!("{p}.1.w3"), hidden, input, false);
        }
    }
    let p = "ss_condition_embedder.module_list.2";
    let d = c.point_channels;
    let grid = c.point_size / c.point_patch_size;
    linear(&mut s, &format!("{p}.point_proj"), d, 3, true);
    for (name, shape) in [
        ("invalid_xyz_token", vec![d]),
        ("pos_embed", vec![1, d, grid, grid]),
        (
            "pos_embed_window",
            vec![1, 1 + c.point_patch_size.pow(2), d],
        ),
        ("cls_token", vec![1, 1, d]),
    ] {
        s.insert(format!("{p}.{name}"), shape);
    }
    block(&mut s, &format!("{p}.blocks.0"), d, 2, false);
    if let Some(depth) = &c.depth_model {
        s.extend(super::depth_schema::expected(depth)?);
    }
    Ok(s)
}
#[derive(Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}
// Match the shared loader, which includes unindexed safetensors sidecars.
fn shard_names(path: &Path) -> Result<Vec<String>> {
    let mut shards = mx::weights::parse_shard_index(path)
        .map_err(anyhow::Error::msg)?
        .unwrap_or_default();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|v| v == "safetensors") && entry.path().is_file() {
            shards.push(
                entry
                    .file_name()
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("Objects shard name must be UTF-8"))?,
            );
        }
    }
    shards.sort();
    shards.dedup();
    ensure!(
        !shards.is_empty(),
        "Objects checkpoint contains no safetensors shards"
    );
    Ok(shards)
}
/// Check headers before auto-depth or any main-model tensor allocation.
pub fn validate_headers(path: &Path, c: &Config) -> Result<()> {
    let expected = expected(c)?;
    let shards = shard_names(path)?;
    let mut seen = HashSet::new();
    for shard in shards {
        ensure!(
            Path::new(&shard)
                .components()
                .all(|v| matches!(v, Component::Normal(_))),
            "invalid Objects shard path"
        );
        let mut file = std::fs::File::open(path.join(&shard))?;
        let file_len = file.metadata()?.len();
        let mut length = [0; 8];
        file.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length);
        ensure!(
            length <= 16 * 1024 * 1024 && length.checked_add(8).is_some_and(|v| v <= file_len),
            "invalid Objects safetensors header length"
        );
        let mut bytes = vec![0; usize::try_from(length)?];
        file.read_exact(&mut bytes)?;
        let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&bytes)?;
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let h: TensorHeader = serde_json::from_value(value)
                .with_context(|| format!("invalid Objects header {name}"))?;
            ensure!(seen.insert(name.clone()), "duplicate Objects tensor {name}");
            validate_shape(&expected, &name, &h.shape)?;
            let depth = name.starts_with("depth_model.");
            let dtype = if depth { "F32" } else { "BF16" };
            ensure!(
                h.dtype == dtype,
                "Objects weight {name} must remain {dtype}"
            );
            let size = h
                .shape
                .iter()
                .try_fold(if depth { 4u64 } else { 2u64 }, |v, d| {
                    v.checked_mul(*d as u64)
                })
                .context("Objects tensor size overflow")?;
            ensure!(
                h.data_offsets[1].checked_sub(h.data_offsets[0]) == Some(size)
                    && h.data_offsets[1] <= file_len - length - 8,
                "invalid Objects tensor offsets {name}"
            );
        }
    }
    ensure!(
        seen.len() == expected.len(),
        "Objects checkpoint is missing required tensors"
    );
    Ok(())
}
fn validate_shape(expected: &Schema, name: &str, shape: &[usize]) -> Result<()> {
    let want = expected
        .get(name)
        .with_context(|| format!("unexpected Objects tensor {name}"))?;
    ensure!(
        shape == want,
        "Objects weight {name}: expected {want:?}, got {shape:?}"
    );
    Ok(())
}
/// Recheck loaded arrays, so a replaced shard cannot bypass header validation.
pub fn validate_loaded(w: &WeightMap, expected: &Schema, prefix: &str) -> Result<()> {
    let required = expected
        .keys()
        .filter(|name| {
            if prefix.is_empty() {
                !name.contains(".backbone.") && !name.starts_with("depth_model.")
            } else {
                name.starts_with(prefix)
            }
        })
        .collect::<Vec<_>>();
    ensure!(
        w.len() == required.len()
            && required
                .iter()
                .all(|name| w.contains_key(&name[prefix.len()..])),
        "Objects loaded tensor set differs from expected scope {prefix}"
    );
    for (name, value) in w {
        let shape = mx::array_shape(value)
            .iter()
            .map(|v| usize::try_from(*v))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        validate_shape(expected, &format!("{prefix}{name}"), &shape)?;
        let dtype = if prefix.starts_with("depth_model.") {
            mx::dtype::FLOAT32
        } else {
            mx::dtype::BFLOAT16
        };
        ensure!(
            mx::array_dtype(value) == dtype,
            "Objects tensor {prefix}{name} has incorrect dtype"
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sam3d_objects_checkpoint_includes_unindexed_shards() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"known":"model.safetensors"}}"#,
        )?;
        std::fs::write(dir.path().join("model.safetensors"), [])?;
        std::fs::write(dir.path().join("extra.safetensors"), [])?;
        assert_eq!(
            shard_names(dir.path())?,
            ["extra.safetensors", "model.safetensors"]
        );
        std::fs::remove_file(dir.path().join("model.safetensors.index.json"))?;
        assert_eq!(
            shard_names(dir.path())?,
            ["extra.safetensors", "model.safetensors"]
        );
        Ok(())
    }
    #[test]
    fn sam3d_objects_checkpoint_rejects_wrong_rank_and_unused_keys() {
        let schema = expected(&Config::default()).unwrap();
        assert_eq!(schema.len(), 3288);
        assert!(
            validate_loaded(
                &WeightMap::new(),
                &schema,
                "ss_condition_embedder.module_list.0.backbone."
            )
            .is_err()
        );
        assert!(validate_loaded(&WeightMap::new(), &schema, "").is_err());
        let name = "ss_condition_embedder.idx_emb";
        assert!(validate_shape(&schema, name, &[]).is_err());
        assert!(validate_shape(&schema, name, &[3, 512]).is_err());
        assert!(validate_shape(&schema, "ss_generator.unused.weight", &[1]).is_err());
        assert!(validate_shape(&schema, name, &[3, 1024]).is_ok());
        assert!(
            validate_shape(
                &schema,
                "ss_condition_embedder.module_list.2.pos_embed",
                &[1, 32, 32, 512]
            )
            .is_err()
        );
    }
}
