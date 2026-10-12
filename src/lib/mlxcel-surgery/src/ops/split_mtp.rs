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

//! `split-mtp`: extract a GLM-4.7-Flash next-token-prediction block into a
//! standalone `glm4_moe_lite_mtp` drafter directory (issue #1326).
//!
//! `zai-org/GLM-4.7-Flash` stores its one MTP layer
//! (`num_nextn_predict_layers: 1`) as `model.layers.{num_hidden_layers}.*`
//! beyond the decoder layers: a DeepSeek-V3-style nextn block with its own
//! `embed_tokens`, `enorm`, `hnorm`, `eh_proj`, one MLA + MoE decoder block,
//! `shared_head.norm` and `shared_head.head`. The community 4-bit
//! conversions drop those tensors entirely, so the only source is the raw
//! checkpoint, and the drafter must be a separate, independently quantizable
//! directory the target loader never reads.
//!
//! [`split_mtp`] is the pure transform over an in-memory [`WeightMap`] that
//! already holds the nextn tensors; [`split_mtp_dir`] is the directory driver
//! the `mlxcel split-mtp` subcommand calls. The driver opens only the shards
//! the safetensors index names for the nextn layer (three of the raw
//! checkpoint's 48), so a 62 GB checkpoint costs a few GB of address space
//! and never has to be loaded whole.
//!
//! ## Output layout
//!
//! ```text
//! model.layers.N.embed_tokens.weight      -> model.embed_tokens.weight
//! model.layers.N.enorm.weight             -> model.enorm.weight
//! model.layers.N.hnorm.weight             -> model.hnorm.weight
//! model.layers.N.eh_proj.weight           -> model.eh_proj.weight
//! model.layers.N.shared_head.norm.weight  -> model.shared_head_norm.weight
//! model.layers.N.shared_head.head.weight  -> lm_head.weight
//! model.layers.N.<rest>                   -> model.mtp_block.<rest>
//! *.rotary_emb.inv_freq                   -> dropped
//! ```
//!
//! followed by the same post-processing the `glm4_moe_lite` target loader
//! applies to its decoder layers: `kv_b_proj` is decomposed into the
//! per-head `embed_q` / `unembed_out` pair through the shared
//! [`mlxcel_core::mla::decompose_kv_b_proj`], the per-expert
//! `mlp.experts.{e}.{proj}` tensors are stacked into
//! `mlp.switch_mlp.{proj}` `[num_experts, out, in]`, every tensor is cast to
//! bf16 except `mlp.gate.e_score_correction_bias` (kept float32, the router
//! selection is sensitive to it), and with `q_bits` set every `.weight` of
//! rank two or more whose last axis is a multiple of the group size is
//! affine-quantized to `.weight` / `.scales` / `.biases`, except the router
//! `mlp.gate.weight`. That is the layout the 4-bit community target ships
//! for its own layers, so `TransformerBlock::from_weights` loads the block
//! unchanged.

use std::path::PathBuf;

use anyhow::anyhow;
use mlxcel_core::mla::{KvBProjGeometry, decompose_kv_b_proj};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Value, json};

use crate::SurgeryError;

/// `model_type` the source checkpoint must declare.
pub const SOURCE_MODEL_TYPE: &str = "glm4_moe_lite";
/// `model_type` written into the drafter's `config.json`.
pub const DRAFTER_MODEL_TYPE: &str = "glm4_moe_lite_mtp";
/// Prefix every tensor of the extracted decoder block lands under.
pub const BLOCK_PREFIX: &str = "model.mtp_block";

/// Files copied from the source directory beside the drafter weights when
/// present. `tokenizer.json` is the one a missing copy is warned about.
const COMPANION_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
];

/// Options for [`split_mtp`] / [`split_mtp_dir`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitMtpOptions {
    /// Verify block size written to the drafter config (bonus token
    /// included). `None` resolves to `num_nextn_predict_layers + 1`, with a
    /// floor of 2 since a block of 1 drafts nothing and a ceiling of
    /// [`MAX_BLOCK_SIZE`] since the value becomes the served verify width.
    pub block_size: Option<usize>,
    /// Affine quantization bit width. `None` keeps the drafter bf16.
    pub q_bits: Option<i32>,
    /// Affine quantization group size. Only read when `q_bits` is set.
    pub q_group_size: i32,
    /// Replace drafter files in a non-empty output directory. Foreign
    /// safetensors files and symlinked drafter paths are always refused.
    pub force: bool,
}

impl Default for SplitMtpOptions {
    fn default() -> Self {
        Self {
            block_size: None,
            q_bits: None,
            q_group_size: 64,
            force: false,
        }
    }
}

/// What [`split_mtp`] produced, before anything is written.
pub struct SplitMtpResult {
    /// The drafter weight map in its final on-disk key layout.
    pub weights: WeightMap,
    /// The drafter `config.json`.
    pub config: Value,
    /// The decoder-layer index the block was read from (`num_hidden_layers`).
    pub source_layer: usize,
    /// Number of `.weight` tensors that were affine-quantized.
    pub quantized_tensors: usize,
}

/// What [`split_mtp_dir`] wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitMtpReport {
    pub output_dir: PathBuf,
    pub source_layer: usize,
    pub tensors: usize,
    pub quantized_tensors: usize,
    pub bytes_written: u64,
    /// Companion files (tokenizer, chat template) copied beside the weights.
    pub copied_files: Vec<String>,
    /// Companion files that were expected and absent from the source.
    pub missing_files: Vec<String>,
}

/// The text config: `text_config` when the checkpoint nests one, else the
/// root object.
pub fn text_config(config: &Value) -> &Value {
    config
        .get("text_config")
        .filter(|v| v.is_object())
        .unwrap_or(config)
}

fn cfg_usize(cfg: &Value, key: &str) -> Result<usize, SurgeryError> {
    cfg.get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| anyhow!("config.json: missing or non-integer `{key}`").into())
}

/// Decoder-layer index that holds the nextn block: `num_hidden_layers`.
///
/// Also validates that the checkpoint is a `glm4_moe_lite` one, because the
/// rename table and the block layout below are specific to that family.
pub fn nextn_layer_index(config: &Value) -> Result<usize, SurgeryError> {
    let text = text_config(config);
    let model_type = text
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    if model_type != SOURCE_MODEL_TYPE {
        return Err(anyhow!(
            "split-mtp: config.json declares model_type {model_type:?}; this tool splits the \
             next-token-prediction block of {SOURCE_MODEL_TYPE} checkpoints only"
        )
        .into());
    }
    cfg_usize(text, "num_hidden_layers")
}

/// Largest `block_size` this tool will record in a drafter config.
///
/// The recorded value becomes the server's default verify width
/// (`peek_glm4_moe_lite_mtp_configured_block_size`), and the GLM verify
/// attention materializes one query row at a time, so the width multiplies
/// both the per-round graph size and the exactness probe's chain arm. A typo
/// (`--block-size 20000`, or a `--q-group-size` value pasted into the wrong
/// flag) would otherwise produce a directory that loads fine and then wedges
/// the scheduler tick for every tenant on the first request. Drafting past
/// the trained `num_nextn_predict_layers` already loses acceptance, so a
/// ceiling well above any useful width costs nothing (issue #1326).
const MAX_BLOCK_SIZE: usize = 16;

/// The block size the drafter config records for `opts`.
fn resolve_block_size(text: &Value, opts: &SplitMtpOptions) -> Result<usize, SurgeryError> {
    let nextn = text
        .get("num_nextn_predict_layers")
        .and_then(Value::as_u64)
        .unwrap_or(1) as usize;
    let requested = opts.block_size.unwrap_or_else(|| nextn.saturating_add(1));
    if requested > MAX_BLOCK_SIZE {
        return Err(anyhow!(
            "split-mtp: --block-size {requested} exceeds the maximum of {MAX_BLOCK_SIZE}; the \
             value becomes the served verify width, and this family verifies one query row at a \
             time"
        )
        .into());
    }
    Ok(requested.max(2))
}

/// Map one `model.layers.N.<rest>` suffix to its drafter key. `None` drops
/// the tensor.
fn rename_nextn_key(rest: &str) -> Option<String> {
    if rest.ends_with("rotary_emb.inv_freq") {
        return None;
    }
    const RENAMES: &[(&str, &str)] = &[
        ("embed_tokens.", "model.embed_tokens."),
        ("enorm.", "model.enorm."),
        ("hnorm.", "model.hnorm."),
        ("eh_proj.", "model.eh_proj."),
        ("shared_head.norm.", "model.shared_head_norm."),
        ("shared_head.head.", "lm_head."),
    ];
    for (from, to) in RENAMES {
        if let Some(tail) = rest.strip_prefix(from) {
            return Some(format!("{to}{tail}"));
        }
    }
    Some(format!("{BLOCK_PREFIX}.{rest}"))
}

/// Stack `mlp.experts.{e}.{proj}.weight` for `e in 0..num_experts` into
/// `mlp.switch_mlp.{proj}.weight` `[num_experts, out, in]`.
fn stack_experts(weights: &mut WeightMap, num_experts: usize) -> Result<(), SurgeryError> {
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        let stacked_key = format!("{BLOCK_PREFIX}.mlp.switch_mlp.{proj}.weight");
        if weights.contains_key(&stacked_key) {
            // Already stacked (an mlx-lm style conversion that kept the
            // nextn layer): nothing to do for this projection.
            continue;
        }
        let mut planes: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let key = format!("{BLOCK_PREFIX}.mlp.experts.{e}.{proj}.weight");
            let scales_key = format!("{BLOCK_PREFIX}.mlp.experts.{e}.{proj}.scales");
            if weights.contains_key(&scales_key) {
                return Err(anyhow!(
                    "split-mtp: `{key}` is already quantized; the tool stacks bf16 expert \
                     planes and quantizes the stack itself, so use the raw zai-org checkpoint"
                )
                .into());
            }
            let plane = weights
                .remove(&key)
                .ok_or_else(|| SurgeryError::TensorNotFound(key.clone()))?;
            planes.push(plane);
        }
        let stacked = mlxcel_core::utils::stack_arrays(&planes, 0);
        weights.insert(stacked_key, stacked);
    }
    Ok(())
}

/// The float32 exception to the bf16 cast: casting the router's selection
/// bias changes which experts are chosen.
fn keeps_float32(key: &str) -> bool {
    key.ends_with("mlp.gate.e_score_correction_bias")
}

/// Refuse a nextn layer that arrives already quantized.
///
/// Everything past this point assumes dense planes: the bf16 pass would
/// rewrite a packed `uint32` payload as floats, and `--q-bits` would then
/// quantize that result, both without an error. `decompose_kv_b_proj` and
/// [`stack_experts`] each handle or refuse their own quantized shape, so a
/// `.scales` plane still present here belongs to a tensor this tool has no
/// dequantization path for (a converted checkpoint that kept the nextn layer
/// with its experts already stacked and packed is the reachable case).
fn reject_packed_tensors(weights: &WeightMap) -> Result<(), SurgeryError> {
    let mut packed: Vec<&String> = weights.keys().filter(|k| k.ends_with(".scales")).collect();
    if packed.is_empty() {
        return Ok(());
    }
    packed.sort();
    let shown: Vec<&str> = packed.iter().take(5).map(|s| s.as_str()).collect();
    Err(anyhow!(
        "split-mtp: {} quantized tensor(s) in the next-token-prediction layer (first: {}); \
         this tool casts dense planes to bf16 and quantizes the result, so a packed input \
         would be silently corrupted; use the raw zai-org checkpoint",
        packed.len(),
        shown.join(", ")
    )
    .into())
}

/// Whether `key` is a candidate for affine quantization under `group_size`.
fn quantizable(key: &str, shape: &[i32], group_size: i32) -> bool {
    key.ends_with(".weight")
        && !key.ends_with("mlp.gate.weight")
        && shape.len() >= 2
        && group_size > 0
        && shape[shape.len() - 1] % group_size == 0
}

/// Extract the nextn block from `weights` (which may hold other tensors,
/// ignored) into the drafter layout described in the module docs.
pub fn split_mtp(
    mut weights: WeightMap,
    config: &Value,
    opts: &SplitMtpOptions,
) -> Result<SplitMtpResult, SurgeryError> {
    let text = text_config(config);
    let source_layer = nextn_layer_index(config)?;
    // Resolved up front: a rejected `--block-size` should fail before the
    // tensor work, not after it. `--q-bits` / `--q-group-size` get the same
    // treatment: MLX's affine quantize kernel only implements
    // `SUPPORTED_AFFINE_BITS` / `SUPPORTED_AFFINE_GROUP_SIZES`, and a wider
    // bounds check (`validate_quantization_params`) still runs later at the
    // quantize call site because it is the shared load-time guard, not a
    // producer-specific one.
    let block_size = resolve_block_size(text, opts)?;
    if let Some(bits) = opts.q_bits {
        mlxcel_core::layers::validate_affine_quantization_bits(bits)
            .map_err(|e| anyhow!("split-mtp: --q-bits: {e}"))?;
        mlxcel_core::layers::validate_affine_quantization_group_size(opts.q_group_size)
            .map_err(|e| anyhow!("split-mtp: --q-group-size: {e}"))?;
    }
    let geometry = KvBProjGeometry {
        num_heads: cfg_usize(text, "num_attention_heads")?,
        qk_nope_head_dim: cfg_usize(text, "qk_nope_head_dim")?,
        v_head_dim: cfg_usize(text, "v_head_dim")?,
        kv_lora_rank: cfg_usize(text, "kv_lora_rank")?,
    };
    let num_experts = text
        .get("n_routed_experts")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;

    // Rename.
    let layer_prefix = format!("model.layers.{source_layer}.");
    let source_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.starts_with(&layer_prefix))
        .cloned()
        .collect();
    if source_keys.is_empty() {
        return Err(anyhow!(
            "split-mtp: no `model.layers.{source_layer}.*` tensor in the checkpoint; this \
             checkpoint was converted without its MTP layer; use the raw zai-org checkpoint"
        )
        .into());
    }
    let mut out = WeightMap::with_capacity(source_keys.len());
    for key in source_keys {
        let value = weights
            .remove(&key)
            .ok_or_else(|| SurgeryError::TensorNotFound(key.clone()))?;
        if let Some(new_key) = rename_nextn_key(&key[layer_prefix.len()..]) {
            out.insert(new_key, value);
        }
    }
    drop(weights);

    // The same per-prefix decomposition the target loader runs per layer.
    decompose_kv_b_proj(
        &mut out,
        &format!("{BLOCK_PREFIX}.self_attn"),
        geometry,
        "MTP block",
    )
    .map_err(|e| anyhow!("split-mtp: {e}"))?;

    if num_experts > 0 {
        stack_experts(&mut out, num_experts)?;
    }
    reject_packed_tensors(&out)?;

    // dtype pass: bf16 everywhere except the router's selection bias.
    let keys: Vec<String> = out.keys().cloned().collect();
    for key in &keys {
        let target = if keeps_float32(key) {
            mlxcel_core::dtype::FLOAT32
        } else {
            mlxcel_core::dtype::BFLOAT16
        };
        let Some(value) = out.get(key) else { continue };
        if mlxcel_core::array_dtype(value) != target {
            let cast = mlxcel_core::astype(value, target);
            out.insert(key.clone(), cast);
        }
    }

    // Optional affine quantization, matching the 4-bit target's own layout.
    let mut quantized_tensors = 0usize;
    if let Some(bits) = opts.q_bits {
        let group_size = opts.q_group_size;
        mlxcel_core::layers::validate_quantization_params(group_size, bits)
            .map_err(|e| anyhow!("split-mtp: {e}"))?;
        for key in &keys {
            let Some(value) = out.get(key) else { continue };
            let shape = mlxcel_core::array_shape(value);
            if !quantizable(key, &shape, group_size) {
                continue;
            }
            let quantized =
                mlxcel_core::quantize_weights_with_mode(value, group_size, bits, "affine");
            let prefix = key.trim_end_matches(".weight");
            out.insert(key.clone(), mlxcel_core::quantized_weights_w(&quantized));
            out.insert(
                format!("{prefix}.scales"),
                mlxcel_core::quantized_weights_scales(&quantized),
            );
            if mlxcel_core::quantized_weights_has_biases(&quantized) {
                out.insert(
                    format!("{prefix}.biases"),
                    mlxcel_core::quantized_weights_biases(&quantized),
                );
            }
            quantized_tensors += 1;
        }
    }

    // Drafter config.
    let mut text_out = text.clone();
    if let Some(obj) = text_out.as_object_mut() {
        obj.remove("quantization");
        obj.remove("quantization_config");
    }
    let mut drafter_config = json!({
        "model_type": DRAFTER_MODEL_TYPE,
        "block_size": block_size,
        "tie_word_embeddings": false,
        "text_config": text_out,
    });
    if let Some(bits) = opts.q_bits {
        let quant = json!({
            "group_size": opts.q_group_size,
            "bits": bits,
            "mode": "affine",
        });
        drafter_config["quantization"] = quant.clone();
        drafter_config["quantization_config"] = quant;
    }

    Ok(SplitMtpResult {
        weights: out,
        config: drafter_config,
        source_layer,
        quantized_tensors,
    })
}

#[path = "split_mtp_dir.rs"]
mod directory;
pub use directory::{refuse_output_is_source, split_mtp_dir};
#[path = "split_mtp_output.rs"]
mod output;
pub use output::write_safetensors;

#[cfg(test)]
#[path = "split_mtp_tests.rs"]
mod split_mtp_tests;
