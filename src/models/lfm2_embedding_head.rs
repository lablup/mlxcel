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

//! Head selection for the LFM2 bidirectional checkpoints.
//!
//! LFM2.5-Embedding, LFM2.5-Encoder and LFM2.5-ColBERT share one backbone and
//! one `model_type`; what tells them apart is whether a token projection ships
//! with the weights. Two layouts carry the ColBERT projection:
//!
//! - `LiquidAI/LFM2.5-ColBERT-350M` keeps it in a sentence-transformers
//!   `1_Dense/` folder, which [`sanitize_decoder_embedding_weights`] folds to
//!   `dense.0.*`.
//! - The mlx-community conversions store it in the main shard as a bare
//!   `dense.*` linear, quantized alongside the backbone in the `-4bit` and
//!   `-8bit` builds.
//! - A checkpoint re-exported by mlx-vlm itself stores it under the module
//!   name its `lfm2_colbert` sanitize produces, `projection.*` (or
//!   `model.projection.*` before that sanitize), quantized or not.
//!
//! Exactly one projection is accepted. Two or more Dense modules are a layout
//! this family has never published, and stacking them silently would serve the
//! wrong vectors, so they are a load error.
//!
//! [`sanitize_decoder_embedding_weights`]: crate::models::embedding_sanitize::sanitize_decoder_embedding_weights

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use serde::Deserialize;
use serde_json::Value;

use crate::models::embedding_sanitize::linear_features;

/// Every prefix a ColBERT projection can be stored under, in the order the
/// layouts are documented above: the folded `1_Dense/` module, the
/// mlx-community `dense.*`, and the two spellings of mlx-vlm's own
/// `projection` module.
pub(super) const PROJECTION_PREFIXES: &[&str] =
    &["dense.0", "dense", "projection", "model.projection"];

/// Tensor suffixes a projection can carry, dense or quantized.
pub(super) const PROJECTION_SUFFIXES: &[&str] = &["weight", "bias", "scales", "biases"];

/// The projection prefix present in `weights`, or `None` for a single-vector
/// export.
///
/// `dense_folders` is the count [`fold_dense_modules`] returned; a second
/// folded module (`dense.1.*`) is caught from the tensors as well, so a map that
/// was sanitized earlier is judged the same way.
///
/// [`fold_dense_modules`]: crate::models::embedding_sanitize::fold_dense_modules
pub(super) fn find_projection(
    weights: &WeightMap,
    dense_folders: usize,
) -> Result<Option<&'static str>> {
    if dense_folders >= 2 || weights.contains_key("dense.1.weight") {
        bail!(
            "the checkpoint carries {} sentence-transformers Dense modules; the LFM2 ColBERT \
             head is exactly one token projection, so a stacked Dense head is not supported",
            dense_folders.max(2)
        );
    }
    let present: Vec<&'static str> = PROJECTION_PREFIXES
        .iter()
        .copied()
        .filter(|prefix| weights.contains_key(&format!("{prefix}.weight")))
        .collect();
    match present.as_slice() {
        [] => Ok(None),
        [prefix] => Ok(Some(prefix)),
        several => bail!(
            "the checkpoint carries {} projections ({}); exactly one LFM2 ColBERT projection \
             is expected",
            several.len(),
            several
                .iter()
                .map(|p| format!("`{p}.weight`"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Remove every tensor of the projection at `prefix` so the backbone
/// constructor never sees it.
pub(super) fn take_projection_keys(weights: &mut WeightMap, prefix: &str) {
    for suffix in PROJECTION_SUFFIXES {
        weights.remove(&format!("{prefix}.{suffix}"));
    }
}

/// The `(out_features, in_features)` of a built projection, after checking
/// every tensor it carries agrees with them.
///
/// MLX checks operand shapes when the op is built, and those ops cross the
/// FFI without a `Result`, so a `bias` of the wrong length or a `scales`
/// plane whose rows disagree with the packed weight would abort the process
/// at the first embed request instead of failing the load. The widths come
/// from the built linear, whose quantization layout is already reconciled
/// against the tensor shapes, not from the config's declared `group_size`.
pub(super) fn projection_widths(linear: &UnifiedLinear, prefix: &str) -> Result<(usize, usize)> {
    let (weight_shape, bias, quantized) = match linear {
        UnifiedLinear::Regular(dense) => (
            mlxcel_core::array_shape(&dense.weight),
            dense.bias.as_ref(),
            None,
        ),
        UnifiedLinear::Quantized { weight, bias } => (
            mlxcel_core::array_shape(&weight.weight),
            bias.as_ref(),
            Some(weight),
        ),
    };
    let [out_features, stored_in] = weight_shape[..] else {
        bail!("LFM2 ColBERT projection `{prefix}.weight` has shape {weight_shape:?}, not rank 2");
    };
    let in_features = match quantized {
        None => stored_in,
        Some(q) => {
            let scales = mlxcel_core::array_shape(&q.scales);
            let groups = match scales[..] {
                [rows, groups] if rows == out_features => groups,
                _ => bail!(
                    "LFM2 ColBERT projection `{prefix}.scales` has shape {scales:?}, which does \
                     not match the {out_features}-row packed weight"
                ),
            };
            if let Some(biases) = &q.biases {
                let biases = mlxcel_core::array_shape(biases);
                if biases != scales {
                    bail!(
                        "LFM2 ColBERT projection `{prefix}.biases` has shape {biases:?} but \
                         `{prefix}.scales` has {scales:?}"
                    );
                }
            }
            let unpacked = (q.bits > 0)
                .then(|| i64::from(stored_in) * 32)
                .filter(|bits_total| bits_total % i64::from(q.bits) == 0)
                .map(|bits_total| bits_total / i64::from(q.bits));
            let grouped = i64::from(groups) * i64::from(q.group_size);
            match unpacked {
                Some(width) if width == grouped && width <= i64::from(i32::MAX) => width as i32,
                _ => bail!(
                    "LFM2 ColBERT projection `{prefix}` packs {stored_in} words at {} bits, which \
                     does not match {groups} groups of {}",
                    q.bits,
                    q.group_size
                ),
            }
        }
    };
    if out_features <= 0 || in_features <= 0 {
        bail!(
            "LFM2 ColBERT projection `{prefix}` is {out_features}x{in_features}; both widths \
             must be positive"
        );
    }
    if let Some(bias) = bias {
        let bias = mlxcel_core::array_shape(bias);
        if bias != [out_features] {
            bail!(
                "LFM2 ColBERT projection `{prefix}.bias` has shape {bias:?}, but the projection \
                 has {out_features} output features"
            );
        }
    }
    Ok((out_features as usize, in_features as usize))
}

/// `{N}_Dense` module folders in `model_dir`, sorted.
fn dense_folders_on_disk(model_dir: &Path) -> Vec<PathBuf> {
    let mut folders: Vec<PathBuf> = std::fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| {
            entry.file_name().to_str().is_some_and(|name| {
                name.strip_suffix("_Dense")
                    .is_some_and(|index| index.parse::<u32>().is_ok())
            })
        })
        .map(|entry| entry.path())
        .collect();
    folders.sort();
    folders
}

/// The fields of a sentence-transformers (or PyLate) `Dense/config.json` that
/// decide whether the module is a plain projection.
#[derive(Debug, Deserialize)]
struct DenseConfig {
    out_features: Option<i64>,
    in_features: Option<i64>,
    activation_function: Option<String>,
    #[serde(default)]
    use_residual: bool,
}

/// Check the projection against what the checkpoint declares about it, and
/// return its output width.
///
/// With a `{N}_Dense/config.json` on disk the module must be a plain linear:
/// `activation_function` Identity (sentence-transformers defaults to Tanh when
/// the key is missing, so a missing key is refused too) and `use_residual`
/// false, and its declared widths must match the tensors. The mlx-community
/// conversions declare `mlx.proj_dim` in `config.json`, which is cross-checked
/// the same way. Without either, the width is read from the weight.
pub(super) fn validate_projection(
    model_dir: &Path,
    config: &Value,
    weights: &WeightMap,
    prefix: Option<&str>,
    group_size: i32,
    hidden_size: usize,
) -> Result<Option<usize>> {
    let folders = dense_folders_on_disk(model_dir);
    if folders.len() >= 2 {
        bail!(
            "{} carries {} sentence-transformers Dense folders; the LFM2 ColBERT head is exactly \
             one token projection",
            model_dir.display(),
            folders.len()
        );
    }
    let Some(prefix) = prefix else {
        if let Some(folder) = folders.first() {
            bail!(
                "{} has no `linear.weight` projection tensor, so the LFM2 ColBERT head cannot be \
                 built",
                folder.display()
            );
        }
        return Ok(None);
    };

    let (out_features, in_features) = linear_features(weights, prefix, group_size)
        .with_context(|| format!("LFM2 ColBERT projection `{prefix}` is not a rank-2 linear"))?;
    if in_features as usize != hidden_size {
        bail!(
            "LFM2 ColBERT projection `{prefix}` takes {in_features} input features but the \
             backbone hidden size is {hidden_size}"
        );
    }

    if let Some(folder) = folders.first() {
        let path = folder.join("config.json");
        if path.is_file() {
            let raw = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let dense: DenseConfig = serde_json::from_str(&raw)
                .with_context(|| format!("failed to parse {}", path.display()))?;
            check_dense_config(&dense, &path, out_features, in_features)?;
        }
    }

    if let Some(proj_dim) = config.pointer("/mlx/proj_dim").and_then(Value::as_i64)
        && proj_dim != i64::from(out_features)
    {
        bail!(
            "config.json declares mlx.proj_dim {proj_dim} but the LFM2 ColBERT projection \
             `{prefix}` has {out_features} output features"
        );
    }
    Ok(Some(out_features as usize))
}

fn check_dense_config(
    dense: &DenseConfig,
    path: &Path,
    out_features: i32,
    in_features: i32,
) -> Result<()> {
    let activation = dense.activation_function.as_deref().unwrap_or("<missing>");
    if !activation.ends_with("Identity") {
        bail!(
            "{} declares activation_function {activation}; the LFM2 ColBERT head is a plain \
             projection, and only Identity is supported",
            path.display()
        );
    }
    if dense.use_residual {
        bail!(
            "{} declares use_residual true; a residual Dense head is not supported",
            path.display()
        );
    }
    if let Some(declared) = dense.out_features
        && declared != i64::from(out_features)
    {
        bail!(
            "{} declares out_features {declared} but the projection weight has {out_features} rows",
            path.display()
        );
    }
    if let Some(declared) = dense.in_features
        && declared != i64::from(in_features)
    {
        bail!(
            "{} declares in_features {declared} but the projection weight reads {in_features}",
            path.display()
        );
    }
    Ok(())
}
