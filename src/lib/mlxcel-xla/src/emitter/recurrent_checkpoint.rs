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

//! Cross-check a recurrent config against its checkpoint tensors (issue #3338).
//!
//! `SsmConfig::validate` only rules out overflow and non-finite parameters. An
//! untrusted `config.json` could still ask for state and graph shapes far
//! larger than the model it ships with, so every SSM dim is checked against
//! the safetensors headers before any graph is emitted: the checkpoint is the
//! real upper bound on what the config may request.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::config::{Config, QuantConfig, WeightScheme};

/// Largest safetensors header this check reads (the format caps it at 100 MB).
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// Read the tensor shapes of one safetensors file from its JSON header.
fn header_shapes(path: &Path) -> Result<HashMap<String, Vec<usize>>, String> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut size = [0u8; 8];
    file.read_exact(&mut size)
        .map_err(|e| format!("read {} header size: {e}", path.display()))?;
    let size = u64::from_le_bytes(size);
    if size == 0 || size > MAX_HEADER_BYTES {
        return Err(format!(
            "{} has an invalid safetensors header size {size}",
            path.display()
        ));
    }
    let mut bytes = Vec::new();
    file.take(size)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("read {} header: {e}", path.display()))?;
    let header: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse {} header: {e}", path.display()))?;
    let object = header
        .as_object()
        .ok_or_else(|| format!("{} header is not an object", path.display()))?;
    let mut shapes = HashMap::new();
    for (name, entry) in object {
        if name == "__metadata__" {
            continue;
        }
        let shape = entry
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("{} tensor {name} has no shape", path.display()))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .and_then(|dim| usize::try_from(dim).ok())
                    .ok_or_else(|| format!("{} tensor {name} has a bad dim", path.display()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        shapes.insert(name.clone(), shape);
    }
    Ok(shapes)
}

/// The checkpoint files of `model_dir`: the single `model.safetensors`, or the
/// shards an index names (each kept inside `model_dir`).
fn checkpoint_files(model_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let single = model_dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index_path).map_err(|e| {
        format!(
            "{}: no model.safetensors and no readable index ({e})",
            model_dir.display()
        )
    })?;
    let index: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", index_path.display()))?;
    let map = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("{} has no weight_map", index_path.display()))?;
    let mut files: Vec<PathBuf> = Vec::new();
    for shard in map.values() {
        let shard = shard
            .as_str()
            .ok_or_else(|| format!("{} maps a tensor to a non-string", index_path.display()))?;
        let file = Path::new(shard);
        if file.components().count() != 1 {
            return Err(format!(
                "{} names a shard outside the model directory: {shard}",
                index_path.display()
            ));
        }
        let path = model_dir.join(file);
        if !files.contains(&path) {
            files.push(path);
        }
    }
    Ok(files)
}

/// Whether `shape` is a `[out, in]` projection, plain or MLX affine-packed.
fn is_projection(shape: &[usize], out: usize, in_: usize, quant: Option<QuantConfig>) -> bool {
    match shape {
        [o, i] if *o == out && *i == in_ => true,
        [o, i] => quant
            .is_some_and(|q| *o == out && q.bits > 0 && in_.checked_mul(q.bits) == Some(i * 32)),
        _ => false,
    }
}

impl Config {
    /// Check every SSM dim against the checkpoint in `model_dir` (a no-op for
    /// a config without recurrent state). Called by [`Config::from_json`].
    pub(crate) fn validate_recurrent_checkpoint(&self, model_dir: &Path) -> Result<(), String> {
        let Some(ssm) = self.ssm.as_ref().filter(|_| self.has_recurrent_state()) else {
            return Ok(());
        };
        if self.weight_scheme != WeightScheme::Mamba2 {
            return Err("a recurrent config must use the Mamba2 weight scheme".into());
        }
        let mut shapes = HashMap::new();
        for file in checkpoint_files(model_dir)? {
            shapes.extend(header_shapes(&file)?);
        }
        let quant = self.quantization;
        let (h, di, cd, nh) = (self.hidden, ssm.d_inner(), ssm.conv_dim, ssm.n_heads);
        let k = ssm.conv_kernel;
        let shape = |name: &str| -> Result<&Vec<usize>, String> {
            shapes
                .get(name)
                .ok_or_else(|| format!("checkpoint has no tensor {name}"))
        };
        let expect = |name: &str, ok: bool, want: String| -> Result<(), String> {
            if ok {
                Ok(())
            } else {
                Err(format!(
                    "checkpoint tensor {name} has shape {:?}, config.json implies {want}",
                    shapes.get(name)
                ))
            }
        };
        let proj = |name: &str, out: usize, in_: usize| -> Result<(), String> {
            let ok = is_projection(shape(name)?, out, in_, quant);
            expect(name, ok, format!("[{out}, {in_}]"))
        };
        let vector = |name: &str, len: usize| -> Result<(), String> {
            let ok = shape(name)?.as_slice() == [len];
            expect(name, ok, format!("[{len}]"))
        };
        proj("backbone.embeddings.weight", self.vocab, h)?;
        vector("backbone.norm_f.weight", h)?;
        if !self.tie_word_embeddings {
            proj("lm_head.weight", self.vocab, h)?;
        }
        for li in 0..self.n_layers {
            let p = format!("backbone.layers.{li}.");
            vector(&format!("{p}norm.weight"), h)?;
            let m = format!("{p}mixer.");
            proj(&format!("{m}in_proj.weight"), ssm.in_proj_dim(), h)?;
            proj(&format!("{m}out_proj.weight"), h, di)?;
            if ssm.use_bias {
                vector(&format!("{m}in_proj.bias"), ssm.in_proj_dim())?;
                vector(&format!("{m}out_proj.bias"), h)?;
            }
            // torch `[C, 1, K]`, or the MLX conversion's `[C, K, 1]` (same bytes).
            let conv = format!("{m}conv1d.weight");
            let conv_shape = shape(&conv)?.as_slice();
            let ok = conv_shape == [cd, 1, k] || conv_shape == [cd, k, 1];
            expect(&conv, ok, format!("[{cd}, 1, {k}]"))?;
            if ssm.use_conv_bias {
                vector(&format!("{m}conv1d.bias"), cd)?;
            }
            for name in ["dt_bias", "A_log", "D"] {
                vector(&format!("{m}{name}"), nh)?;
            }
            vector(&format!("{m}norm.weight"), di)?;
        }
        Ok(())
    }
}
