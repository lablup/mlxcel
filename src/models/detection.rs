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

//! Model-type detection helpers.
//!
//! This module owns config-driven architecture classification and related
//! detection helpers so `models/mod.rs` can stay focused on the registry of
//! model implementations and exported types.

use anyhow::Result;
use mlxcel_core::drafter::dflash::is_dflash_drafter_config;
use serde_json::Value;
use std::io::Read;
use std::path::Path;

use super::ModelType;
use super::sanitize::sanitize_config_json;

/// Canonical error for "this directory is a DFlash speculative drafter, not a
/// model you can load standalone" (#1168).
///
/// Every entry point that resolves a checkpoint directory goes through
/// [`get_model_type`], so raising the rejection there gives the offline
/// `mlxcel generate -m`, server startup, and the distributed stage loaders one
/// shared message instead of each one surfacing a different weight-map symptom.
///
/// Before this arm existed, a DFlash drafter passed to `-m` (or reached by the
/// offline `--draft-model` path, which loads the drafter as a full model) was
/// classified `ModelType::Qwen3` from its `"model_type": "qwen3"`, routed to
/// `Qwen3Model::load`, and died on its first weight lookup with
/// `Weight not found: model.embed_tokens.weight`. That message names a tensor,
/// not the problem.
fn dflash_drafter_not_standalone_error(model_path: &Path) -> anyhow::Error {
    anyhow::anyhow!(
        "{path} is a DFlash-family speculative drafter checkpoint (Qwen 3.5 DFlash, \
         LFM2 DSpark, Muse Glimmer assistant or Laguna DFlash), not a standalone model. \
         Its config.json declares the DFlashDraftModel, Lfm2DSparkDraftModel, \
         MuseGlimmerAssistantModel or DFlashLagunaForCausalLM architecture and/or a \
         dflash_config block, and \
         its weights carry no embed_tokens and no lm_head because such a drafter \
         borrows both from the target model when it binds. Pass a full model to -m, \
         and pass this directory to --draft-model on `mlxcel-server` to use it as a \
         drafter.",
        path = model_path.display(),
    )
}

pub(crate) fn has_vision_config(config: &serde_json::Value) -> bool {
    config.get("vision_config").is_some()
}

/// Classify an `iquestcoder` config, refusing the two switches that would make
/// it stop being a plain Llama decoder (#1357).
///
/// IQuest-Coder V1 (7B / 14B / 40B, Base / Instruct / Thinking) is RMSNorm plus
/// GQA with an explicit `head_dim`, SwiGLU, an untied `lm_head` and default
/// RoPE, which is exactly what `models::llama3` runs. Its config schema is
/// Llama's plus three keys that are dormant in every published checkpoint, and
/// the whole equivalence rests on them staying dormant, so this refuses at load
/// rather than ignoring them:
///
/// * `clip_qkv` clamps Q, K and V to `[-clip_qkv, clip_qkv]` before attention.
///   The vendor decoder applies it whenever the value is not null
///   (`IQuestCoderAttention.forward` in
///   <https://huggingface.co/mlx-community/IQuest-Coder-V1-7B-Instruct-8bit/blob/main/modeling_iquestcoder.py>),
///   and the shared Llama attention has no such clamp. Every published
///   checkpoint sets it to `null`.
/// * `use_sliding_window` with a non-null `sliding_window` makes every layer
///   from `max_window_layers` on attend over a window instead of the full
///   prefix. The Llama route builds a plain causal mask, so a windowed
///   checkpoint would decode with the wrong receptive field. Every published
///   checkpoint ships `use_sliding_window: false`, `sliding_window: null`,
///   `max_window_layers: 0`.
///
/// Both refusals reproduce the vendor's own activation conditions, so a config
/// that merely carries the keys at their inert values still routes. The vendor
/// window condition reads, verbatim:
///
/// ```text
/// self.config.use_sliding_window
/// and getattr(self.config, "sliding_window", None) is not None
/// and self.layer_idx >= self.config.max_window_layers
/// ```
///
/// so the layer comparison is `>=`, and a `max_window_layers` at or past the
/// last layer index leaves every layer unwindowed.
///
/// Both tests are deliberately written against JSON *presence*, not JSON type.
/// The vendor tests `is not None` and Python truthiness, so a `sliding_window`
/// written as `4096.0` or `"4096"` is just as live as `4096`, and reading it
/// through `as_u64` would answer `None` and let the checkpoint through into a
/// full-attention decode. A guard that exists to stop silently wrong output
/// must fail closed on a spelling it does not recognise.
fn iquest_coder_model_type(config: &Value) -> Result<ModelType> {
    if let Some(clip) = config.get("clip_qkv")
        && !clip.is_null()
    {
        return Err(anyhow::anyhow!(
            "IQuest-Coder checkpoint declares clip_qkv = {clip}, which clamps the Q/K/V \
             projections before attention. mlxcel routes this family through the shared Llama \
             decoder, which applies no such clamp, so loading it would silently change every \
             attention score. Only checkpoints with clip_qkv null are supported."
        ));
    }

    // Python truthiness: absent, null and `false` are off, anything else is on.
    let uses_sliding_window = config
        .get("use_sliding_window")
        .is_some_and(|value| !matches!(value, Value::Null | Value::Bool(false)));
    let window = config.get("sliding_window").filter(|v| !v.is_null());
    let first_windowed_layer = config
        .get("max_window_layers")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    // An unreadable layer count has to mean "assume some layer reaches the
    // window", or an absent `num_hidden_layers` would open the guard.
    let num_layers = config
        .get("num_hidden_layers")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX);
    if let Some(window) = window
        && uses_sliding_window
        && first_windowed_layer < num_layers
    {
        return Err(anyhow::anyhow!(
            "IQuest-Coder checkpoint enables sliding-window attention (use_sliding_window is set, \
             sliding_window = {window}, max_window_layers = {first_windowed_layer}). mlxcel routes \
             this family through the shared Llama decoder, which attends over the full prefix, so \
             loading it would decode with the wrong receptive field. Only checkpoints with the \
             sliding window disabled are supported."
        ));
    }

    Ok(ModelType::IQuestCoder)
}

/// `architectures[0]` of the IQuest-Coder causal-LM checkpoints.
const IQUEST_CODER_ARCHITECTURE: &str = "IQuestCoderForCausalLM";

/// Whether a config declares the IQuest-Coder architecture regardless of the
/// `model_type` it carries.
///
/// A checkpoint relabelled `"model_type": "llama"` loads on the same decoder
/// either way, so this is not about routing. It is about the guards: without
/// it, relabelling is a way to reach the Llama route with a live `clip_qkv` or
/// sliding window and get exactly the silently wrong decode
/// [`iquest_coder_model_type`] exists to refuse. Relabelling to `llama` is a
/// real practice for this family, because it is how a checkpoint is made
/// loadable by reference stacks that do not run its `auto_map` code.
fn declares_iquest_coder_architecture(config: &Value) -> bool {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|entries| {
            entries
                .iter()
                .any(|entry| entry.as_str() == Some(IQUEST_CODER_ARCHITECTURE))
        })
}

/// `architectures[0]` of the IQuest-Coder Loop causal-LM checkpoints.
const IQUEST_LOOP_CODER_ARCHITECTURE: &str = "IQuestLoopCoderForCausalLM";

/// Whether a config declares the IQuest-Coder **Loop** architecture regardless
/// of the `model_type` it carries.
///
/// Same relabelling practice as [`declares_iquest_coder_architecture`], but a
/// worse failure if it goes unnoticed: the Loop decoder runs its layer stack
/// twice and mixes each pass-2 layer's attention through a per-head gate, so a
/// Loop checkpoint that reached the plain Llama route would run half the
/// computation and never read a single `model.gate_projections.*` tensor. The
/// output would still be fluent, which is exactly why this has to be refused at
/// detection rather than discovered later.
fn declares_iquest_loop_coder_architecture(config: &Value) -> bool {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|entries| {
            entries
                .iter()
                .any(|entry| entry.as_str() == Some(IQUEST_LOOP_CODER_ARCHITECTURE))
        })
}

/// Classify an `iquestloopcoder` config, refusing a `loop_num` this decoder
/// does not implement.
///
/// `loop_num` decides how many times the layer stack is run and therefore how
/// many cache sets each layer owns. `crate::models::iquestloopcoder` implements
/// exactly two, which is what every published checkpoint declares; anything
/// else would need a third cache set per layer and a second gate application.
/// Refusing here rather than at `from_weights` keeps a 20+ GB read from
/// happening first.
///
/// Written against JSON *presence* rather than JSON type, for the same reason
/// [`iquest_coder_model_type`] is: the vendor config compares the value
/// directly, so a `loop_num` written as `3.0` or `"3"` is just as live as `3`,
/// and reading it through `as_u64` alone would answer `None` and let it through
/// as the default 2.
fn iquest_loop_coder_model_type(config: &Value) -> Result<ModelType> {
    let loop_num = match config.get("loop_num") {
        // Absent or explicitly null: the vendor config class defaults to 2.
        None | Some(Value::Null) => 2,
        Some(value) => value.as_u64().ok_or_else(|| {
            anyhow::anyhow!(
                "IQuest-Coder Loop checkpoint declares loop_num = {value}, which is not a whole \
                 number. mlxcel implements loop_num 2 only and cannot tell whether this \
                 checkpoint asks for it, so it refuses rather than guessing."
            )
        })?,
    };
    if loop_num != 2 {
        return Err(anyhow::anyhow!(
            "IQuest-Coder Loop checkpoint declares loop_num = {loop_num}. mlxcel implements the \
             two-pass loop only: each layer owns one full KV cache for pass 1 and one rotating \
             cache for pass 2, and a third pass would need a third set. Only loop_num 2 is \
             supported."
        ));
    }
    Ok(ModelType::IQuestLoopCoder)
}

fn gemma4_has_vision_weights(model_path: &Path) -> bool {
    let index_path = model_path.join("model.safetensors.index.json");
    if let Ok(index_str) = std::fs::read_to_string(&index_path)
        && let Ok(index) = serde_json::from_str::<Value>(&index_str)
        && let Some(weight_map) = index.get("weight_map").and_then(Value::as_object)
    {
        // MLX-community checkpoints expose the vision front-end unprefixed
        // (`vision_tower.` / `embed_vision.`); ModelOpt NVFP4 exports nest it
        // under a leading `model.` (`model.vision_tower.` /
        // `model.embed_vision.`). Recognize both so an NVFP4 multimodal
        // checkpoint routes to Gemma4VLM instead of the text-only path, where
        // `normalize_nvfp4_keys` then strips the `model.` prefix (issue #749).
        return weight_map.keys().any(|key| {
            key.starts_with("vision_tower.")
                || key.starts_with("embed_vision.")
                || key.starts_with("model.vision_tower.")
                || key.starts_with("model.embed_vision.")
        });
    }

    model_path.join("processor_config.json").exists()
}

fn inkling_dir_is_mtp_only(model_path: &Path) -> bool {
    let (mut has_mtp, mut has_target) =
        std::fs::read_to_string(model_path.join("model.safetensors.index.json"))
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .and_then(|index| index.get("weight_map").and_then(Value::as_object).cloned())
            .map_or((false, false), |weight_map| {
                let has_mtp = weight_map.keys().any(|key| {
                    key.starts_with("model.mtp.layers.") || key.starts_with("mtp.layers.")
                });
                let has_target = weight_map.keys().any(|key| {
                    key == "model.llm.embed.weight"
                        || key == "model.embed_tokens.weight"
                        || key.starts_with("model.llm.layers.")
                        || key.starts_with("model.layers.")
                });
                (has_mtp, has_target)
            });
    if !has_mtp {
        has_mtp = mlxcel_core::weights::dir_has_tensor_name(model_path, |key| {
            key.starts_with("model.mtp.layers.") || key.starts_with("mtp.layers.")
        })
        .unwrap_or(false);
    }
    if !has_target {
        has_target = mlxcel_core::weights::dir_has_tensor_name(model_path, |key| {
            key == "model.llm.embed.weight"
                || key == "model.embed_tokens.weight"
                || key.starts_with("model.llm.layers.")
                || key.starts_with("model.layers.")
        })
        .unwrap_or(false);
    }
    has_mtp && !has_target
}

const MAX_SAFETENSORS_HEADER_BYTES: u64 = 128 * 1024 * 1024;

/// `true` when one safetensors shard's header names a tensor under `prefix`.
fn safetensors_header_has_key_prefix(path: &Path, prefix: &str) -> bool {
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut length = [0_u8; 8];
    if file.read_exact(&mut length).is_err() {
        return false;
    }
    let length = u64::from_le_bytes(length);
    if length == 0 || length > MAX_SAFETENSORS_HEADER_BYTES {
        return false;
    }
    let Ok(length) = usize::try_from(length) else {
        return false;
    };
    let mut header = vec![0_u8; length];
    if file.read_exact(&mut header).is_err() {
        return false;
    }
    serde_json::from_slice::<Value>(&header)
        .ok()
        .and_then(|value| {
            value
                .as_object()
                .map(|entries| entries.keys().any(|key| key.starts_with(prefix)))
        })
        .unwrap_or(false)
}

/// `true` when the checkpoint stores a tensor under `prefix`, read from the
/// safetensors index when there is one and from the shard headers otherwise.
///
/// Used by: Inkling (`model.visual.`), Kimi K3 (`vision_tower.`).
pub(crate) fn checkpoint_has_weight_prefix(model_path: &Path, prefix: &str) -> bool {
    let index_path = model_path.join("model.safetensors.index.json");
    if let Ok(index) = std::fs::read_to_string(index_path)
        && let Ok(index) = serde_json::from_str::<Value>(&index)
        && let Some(weights) = index.get("weight_map").and_then(Value::as_object)
    {
        return weights.keys().any(|key| key.starts_with(prefix));
    }

    let Ok(entries) = std::fs::read_dir(model_path) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        entry.path().extension().and_then(|value| value.to_str()) == Some("safetensors")
            && safetensors_header_has_key_prefix(&entry.path(), prefix)
    })
}

pub(crate) fn inkling_has_vision_weights(model_path: &Path) -> bool {
    checkpoint_has_weight_prefix(model_path, "model.visual.")
}

/// Kimi K3 ships its MoonViT3D tower under `vision_tower.` (#1342).
pub(crate) fn kimi_k3_has_vision_weights(model_path: &Path) -> bool {
    checkpoint_has_weight_prefix(model_path, "vision_tower.")
}

pub(crate) fn detect_text_or_vlm(
    config: &serde_json::Value,
    text_model: ModelType,
    vlm_model: ModelType,
) -> ModelType {
    if has_vision_config(config) {
        vlm_model
    } else {
        text_model
    }
}

/// Split the `phi` / `phi-msft` arm between the dense Phi decoder and Phixtral.
///
/// No phixtral checkpoint declares `model_type: "phixtral"`.
/// `mlabonne/phixtral-4x2_8` declares `phi-msft`, and upstream mlx-lm reaches
/// its phixtral implementation through `MODEL_REMAPPING` rather than the config
/// value, so an arm keyed on the string `"phixtral"` could never fire. The
/// discriminator is `num_local_experts`, which the sparse config carries and
/// the dense Phi-2 config does not.
///
/// A value of 1 is treated as dense: it describes one expert, which is a dense
/// MLP, and the phixtral block would be a needless indirection over it.
pub(crate) fn detect_phi_model_type(config: &serde_json::Value) -> ModelType {
    let num_local_experts = config["num_local_experts"].as_i64().unwrap_or(0);
    if num_local_experts > 1 {
        ModelType::Phixtral
    } else {
        ModelType::Phi
    }
}

pub(crate) fn detect_hunyuan_model_type(config: &serde_json::Value) -> ModelType {
    let num_experts = config["num_experts"].as_i64().unwrap_or(1);
    if num_experts > 1 {
        ModelType::HunyuanMoe
    } else {
        ModelType::HunyuanV1Dense
    }
}

/// `model_type` values that are encoder-only and never generate text. A
/// `BertForMaskedLM` / `ModernBertForMaskedLM` checkpoint loads as an
/// embedder with its MLM head dropped.
const ENCODER_ONLY_MODEL_TYPES: &[&str] = &["bert", "xlm-roberta", "modernbert", "siglip"];

/// `architectures[0]` values that mark an embedding export outright.
const EMBEDDING_ARCHITECTURES: &[&str] = &[
    "BertModel",
    "XLMRobertaModel",
    "ModernBertModel",
    "SiglipModel",
    "SiglipTextModel",
    "LlamaBidirectionalModel",
    "LlamaNemotronVLModel",
    "Lfm2BidirectionalModel",
    "ColIdefics3",
    "ColQwen2_5",
    "ColQwen2ForRetrieval",
];

/// `model_type` values whose `ForSequenceClassification` export is a
/// cross-encoder reranker mlxcel can serve on `/v1/rerank`.
///
/// Every entry has an encoder trunk and a one-label head sitting on it
/// (`BertForSequenceClassification`, `XLMRobertaForSequenceClassification`,
/// `ModernBertForSequenceClassification`). A classifier on any other family
/// (`DebertaV2ForSequenceClassification`, for instance) has no port, so it
/// falls through to the generation dispatch and is rejected there rather than
/// being silently mistaken for a reranker.
const RERANKER_CLASSIFIER_MODEL_TYPES: &[&str] =
    &["bert", "xlm-roberta", "xlm_roberta", "modernbert"];

fn first_architecture(config: &Value) -> Option<&str> {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(Value::as_str)
}

/// `config.architectures[0]` ends with `ForSequenceClassification`.
pub(crate) fn is_sequence_classification_architecture(config: &Value) -> bool {
    first_architecture(config).is_some_and(|arch| arch.ends_with("ForSequenceClassification"))
}

/// Recognize a cross-encoder reranker checkpoint.
///
/// `Some(ModelType::SequenceClassifier)` when `architectures[0]` ends with
/// `ForSequenceClassification` and `model_type` names one of the encoder
/// families the `/v1/rerank` sequence-classifier path implements. Everything
/// else is `None`, which keeps a classifier on an unported family on the
/// existing dispatch path.
pub(crate) fn is_sequence_classifier_checkpoint(config: &Value) -> Option<ModelType> {
    if !is_sequence_classification_architecture(config) {
        return None;
    }
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)?
        .to_ascii_lowercase();
    RERANKER_CLASSIFIER_MODEL_TYPES
        .contains(&model_type.as_str())
        .then_some(ModelType::SequenceClassifier)
}

/// `config.architectures[0]` names an embedding export (including the two
/// flag-gated decoders: `Gemma3TextModel` with `use_bidirectional_attention`
/// and `Ministral3Model` with `is_causal: false`).
fn has_embedding_architecture(config: &Value) -> bool {
    let Some(arch) = first_architecture(config) else {
        return false;
    };
    if EMBEDDING_ARCHITECTURES.contains(&arch) {
        return true;
    }
    let flag = |key: &str| config.get(key).and_then(Value::as_bool);
    match arch {
        "Gemma3TextModel" => flag("use_bidirectional_attention") == Some(true),
        "Ministral3Model" => flag("is_causal") == Some(false),
        _ => false,
    }
}

/// `<model_dir>/modules.json` lists a sentence-transformers module whose
/// `type` ends with `.Pooling`. A file whose only extra module is
/// `1_LogitScore` (Qwen3-VL-Reranker) does not qualify.
fn modules_json_has_pooling(model_path: &Path) -> bool {
    let Ok(raw) = std::fs::read_to_string(model_path.join("modules.json")) else {
        return false;
    };
    let Ok(modules) = serde_json::from_str::<Value>(&raw) else {
        return false;
    };
    modules
        .as_array()
        .map(|entries| {
            entries.iter().any(|entry| {
                entry
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|ty| ty.ends_with(".Pooling"))
            })
        })
        .unwrap_or(false)
}

/// Map the `model_type` of a detected embedding layout to its family.
fn embedding_variant_for_model_type(model_type: &str) -> Option<ModelType> {
    Some(match model_type {
        "bert" => ModelType::Bert,
        "xlm-roberta" | "xlm_roberta" => ModelType::XlmRoberta,
        "modernbert" => ModelType::ModernBert,
        "siglip" | "siglip_text_model" => ModelType::SiglipText,
        "gemma3_text" | "gemma3" => ModelType::Gemma3Embedding,
        "qwen3" => ModelType::Qwen3Embedding,
        "qwen3_vl" => ModelType::Qwen3VLEmbedding,
        "lfm2" => ModelType::Lfm2Embedding,
        "ministral3" => ModelType::Ministral3Embedding,
        "llama" | "llama_bidirec" => ModelType::LlamaBidirec,
        "llama_nemotron_vl" => ModelType::LlamaNemotronVLEmbedding,
        "idefics3" => ModelType::ColIdefics3,
        "qwen2_5_vl" | "colqwen2" => ModelType::ColQwen25,
        _ => return None,
    })
}

/// Recognize an embedding checkpoint before the `model_type` dispatch.
///
/// Returns `Ok(Some(variant))` when the checkpoint is an embedding export:
/// its `model_type` is an encoder-only family, `config.architectures[0]` is
/// an embedding architecture, `modules.json` carries a `.Pooling` module, or
/// `1_Pooling/config.json` exists. A checkpoint whose `architectures[0]`
/// ends with `ForSequenceClassification` is a reranker, never an embedder.
/// `Ok(None)` means "not an embedding checkpoint, continue with the
/// generation dispatch"; `Err` means the layout says embedding but the
/// `model_type` has no embedding family, which is reported rather than
/// misrouted to a causal generator.
pub(crate) fn is_embedding_checkpoint(
    model_path: &Path,
    config: &Value,
) -> Result<Option<ModelType>> {
    let Some(model_type_raw) = config.get("model_type").and_then(Value::as_str) else {
        return Ok(None);
    };
    let model_type = model_type_raw.to_ascii_lowercase();

    if is_sequence_classification_architecture(config) {
        return Ok(None);
    }

    let encoder_only = ENCODER_ONLY_MODEL_TYPES.contains(&model_type.as_str());
    let layout_says_embedding = encoder_only
        || has_embedding_architecture(config)
        || modules_json_has_pooling(model_path)
        || model_path.join("1_Pooling").join("config.json").exists();
    if !layout_says_embedding {
        return Ok(None);
    }

    match embedding_variant_for_model_type(&model_type) {
        Some(variant) => Ok(Some(variant)),
        None => Err(anyhow::anyhow!(
            "{} is an embedding checkpoint (sentence-transformers pooling layout or embedding \
             architecture), but model_type `{model_type_raw}` has no embedding family in \
             mlxcel; see docs/embeddings.md for the supported families",
            model_path.display()
        )),
    }
}

/// `true` when the family consumes a `video_url` / `--video` clip through a
/// temporal path of its own: a 3D encoder, adjacent-frame planes, or a
/// per-frame scatter into video placeholders.
///
/// The single list both fronts read. `server::startup::detect_model_media_support`
/// turns it into `ModelMediaSupport::video_native` and
/// `commands::generate` uses it to decide whether `--video` needs the
/// frames fallback, so the HTTP boundary and the CLI cannot drift apart.
/// Mirror the dispatch in `commands/generate_vlm::compute_vlm_embeddings` and
/// `server::model_worker::prepare_request_video_embeddings` when a family
/// gains a native path.
#[must_use]
pub fn model_type_has_native_video(model_type: ModelType) -> bool {
    matches!(
        model_type,
        ModelType::Gemma4VLM
            | ModelType::Gemma4Unified
            | ModelType::InklingVLM
            | ModelType::KimiVL
            | ModelType::KimiK25
            | ModelType::Qwen2VL
            | ModelType::Qwen25VL
            | ModelType::Qwen3VL
            | ModelType::Qwen3VLMoe
            | ModelType::Qwen35VLM
            | ModelType::Qwen35MoeVLM
    )
}

/// `true` when the family loads as a vision-language model, so image content
/// can reach a vision tower.
///
/// The config.json-only predicate the model registry itself uses, re-exported
/// because the binary crate's `--video` handling needs it too.
#[must_use]
pub fn model_type_is_vision_capable(model_type: ModelType) -> bool {
    crate::model_metadata::is_vlm_model_type(model_type)
}

/// Detect model type from config.json
pub fn get_model_type(model_path: &Path) -> Result<ModelType> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(config_path)?;
    let config_str = sanitize_config_json(&config_str);
    let v: serde_json::Value = serde_json::from_str(&config_str)?;

    // A DFlash speculative drafter is structurally not a standalone model, but
    // it declares an ordinary `"model_type": "qwen3"`, so it would otherwise
    // fall through to the Qwen 3 arm below (#1168). Reject it before any
    // `model_type` dispatch runs. The discriminator is structural (a
    // `dflash_config` block and/or the `DFlashDraftModel` architecture) rather
    // than the resolved `DrafterKind`, because `DEFAULT_DRAFTER_KIND` is
    // `Dflash`: keying on the resolved kind would also reject an ordinary small
    // full model used as a classic drafter, which loads and runs fine.
    if is_dflash_drafter_config(&v) {
        return Err(dflash_drafter_not_standalone_error(model_path));
    }

    // Kokoro TTS checkpoints carry no top-level `model_type`, so detect them by
    // architecture signal (the `istftnet` config block or the canonical weight
    // filename) before the `model_type`-based dispatch below would error.
    if super::kokoro::is_kokoro_checkpoint(model_path, &v) {
        return Ok(ModelType::Kokoro);
    }

    // A cross-encoder reranker reuses an encoder `model_type` (`bert`,
    // `xlm-roberta`, `modernbert`) that the embedding rules would otherwise
    // claim, so the classifier check runs first (#1356). `is_embedding_checkpoint`
    // refuses every `ForSequenceClassification` export for the same reason: a
    // reranker scores a pair, it does not produce a vector.
    if let Some(reranker) = is_sequence_classifier_checkpoint(&v) {
        return Ok(reranker);
    }

    // Embedding exports reuse generator `model_type`s (`qwen3` for
    // Qwen3-Embedding, `gemma3_text` for EmbeddingGemma), so the layout and
    // architecture rules must run before the `model_type` match below.
    if let Some(embedding) = is_embedding_checkpoint(model_path, &v)? {
        return Ok(embedding);
    }

    let model_type_raw = v["model_type"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("model_type not found"))?;
    // Normalize to lowercase so HuggingFace checkpoints that preserve the
    // upstream casing (e.g. `NemotronH_Nano_Omni_Reasoning_V3`) match the
    // same arm as their canonical lowercase form.
    let model_type = model_type_raw.to_ascii_lowercase();

    if matches!(model_type.as_str(), "inkling_mm_model" | "inkling")
        && inkling_dir_is_mtp_only(model_path)
    {
        return Err(anyhow::anyhow!(
            "{} is an isolated Inkling MTP drafter checkpoint, not a standalone target model. Pass the full Inkling checkpoint to -m and this directory to --draft-model with --draft-kind mtp.",
            model_path.display()
        ));
    }

    match model_type.as_str() {
        // An IQuest-Coder checkpoint relabelled `llama` (the usual way to make
        // it loadable by a stack that will not run its `auto_map` code) routes
        // to the same decoder either way, but it has to pass the same config
        // guards; see `declares_iquest_coder_architecture`.
        // The IQuest-Coder Loop architecture string beats every `model_type`
        // spelling, not just `llama`. Guarding only the `llama` relabel left the
        // likelier mistake open: `"model_type": "iquestcoder"` with
        // `IQuestLoopCoderForCausalLM` fell through to the `iquestcoder` arm
        // below, which routes to the shared Llama decoder and would run the
        // stack once instead of twice, never reading a `gate_projections`
        // tensor. That output is fluent, so it has to be refused here.
        _ if declares_iquest_loop_coder_architecture(&v) => iquest_loop_coder_model_type(&v),
        "llama" | "mistral" if declares_iquest_coder_architecture(&v) => {
            iquest_coder_model_type(&v)
        }
        "llama" | "mistral" => Ok(ModelType::Llama),
        "iquestcoder" => iquest_coder_model_type(&v),
        "iquestloopcoder" => iquest_loop_coder_model_type(&v),
        "llama4" => Ok(detect_text_or_vlm(
            &v,
            ModelType::Llama4,
            ModelType::Llama4VLM,
        )),
        // Llama 3.2 Vision. Always multimodal (Llama-3 text backbone with gated
        // cross-attention adapters attending to a tiled ViT tower).
        "mllama" => Ok(ModelType::MllamaVLM),
        "qwen2" => Ok(ModelType::Qwen2),
        "qwen3" => Ok(ModelType::Qwen3),
        "qwen3_moe" => Ok(ModelType::Qwen3Moe),
        "qwen3_next" | "qwen3next" => Ok(ModelType::Qwen3Next),
        "qwen3_5" => Ok(detect_text_or_vlm(
            &v,
            ModelType::Qwen35,
            ModelType::Qwen35VLM,
        )),
        "qwen3_5_moe" => Ok(detect_text_or_vlm(
            &v,
            ModelType::Qwen35Moe,
            ModelType::Qwen35MoeVLM,
        )),
        "qwen2_moe" => Ok(ModelType::Qwen2Moe),
        "gemma" => Ok(ModelType::Gemma),
        "gemma2" => Ok(ModelType::Gemma2),
        "gemma3" | "gemma3_text" => Ok(detect_text_or_vlm(
            &v,
            ModelType::Gemma3,
            ModelType::Gemma3VLM,
        )),
        "gemma4" | "gemma4_text" => Ok(if gemma4_has_vision_weights(model_path) {
            ModelType::Gemma4VLM
        } else {
            ModelType::Gemma4
        }),
        // LLaDA-2 MoE (masked-diffusion LM with a DeepSeek-V3-style MoE FFN).
        // Generates by iterative block-wise unmasking rather than autoregressive
        // decode; served on the shared diffusion worker loop.
        "llada2_moe" => Ok(ModelType::Llada2Moe),
        // DiffusionGemma (block-diffusion on the Gemma 4 MoE backbone). The
        // checkpoint always ships a vision tower, but phase 1 is text-only:
        // the loader skips the vision weights, so detection is by model_type
        // alone. `diffusion_gemma_text` is accepted for text-only exports.
        "diffusion_gemma" | "diffusion_gemma_text" => Ok(ModelType::DiffusionGemma),
        // Gemma 4 Unified is always multimodal (text + vision [+ audio]); it
        // carries `vision_embedder.*` patch-projector weights rather than the
        // `vision_tower.*` ViT used by `gemma4`/Gemma4VLM, so it is detected by
        // model_type alone and never misrouted to Gemma4VLM.
        "gemma4_unified" => Ok(ModelType::Gemma4Unified),
        "gemma3n" | "gemma3n_text" => Ok(detect_text_or_vlm(
            &v,
            ModelType::Gemma3n,
            ModelType::Gemma3nVLM,
        )),
        "phi" | "phi-msft" => Ok(detect_phi_model_type(&v)),
        "phi3" => Ok(ModelType::Phi3),
        "phi4mm" => Ok(ModelType::Phi4MMVLM),
        "phi4-siglip" => Ok(ModelType::Phi4SigLipVLM),
        "phi3_v" => Ok(ModelType::Phi3VLM),
        "phi3small" => Ok(ModelType::Phi3Small),
        "phimoe" => Ok(ModelType::PhiMoe),
        "minimax" => Ok(ModelType::MiniMax),
        "minimax_m3" => Ok(ModelType::MiniMaxM3),
        "minimax_m3_vl" => Ok(ModelType::MiniMaxM3VL),
        "muse_glimmer" => Ok(ModelType::MuseGlimmerVLM),
        "gpt_oss" => Ok(ModelType::GptOss),
        "mixtral" => Ok(ModelType::Mixtral),
        "dbrx" => Ok(ModelType::Dbrx),
        "olmoe" => Ok(ModelType::OLMoE),
        "deepseek" => Ok(ModelType::DeepSeek),
        // `mlx-community/Youtu-LLM-2B-4bit` relabels a Youtu-LLM export
        // `deepseek_v2` so mlx-lm can load it, keeping
        // `architectures: ["YoutuForCausalLM"]`. It stays on this arm on
        // purpose: greedy decode of that checkpoint through the DeepSeek-V2
        // decoder was measured against an mlx-lm oracle on the same weights and
        // matches it, so there is nothing for a detection split to fix and one
        // would only move working checkpoints onto a different decoder (#1371).
        "deepseek_v2" => Ok(ModelType::DeepSeekV2),
        "deepseek_v3" => Ok(ModelType::DeepSeekV3),
        "deepseek_v32" | "deepseek_v3.2" => Ok(ModelType::DeepSeekV32),
        "deepseek_v4" => Ok(ModelType::DeepSeekV4),
        "dots1" => Ok(ModelType::Dots1),
        "cohere" => Ok(ModelType::Cohere),
        "cohere2" => Ok(ModelType::Cohere2),
        "cohere2_moe" => Ok(ModelType::Cohere2Moe),
        // Cohere Compass / North-Micro-Vision: a Qwen3-VL deepstack vision
        // tower in front of a Command-style parallel decoder. The text
        // sub-config says `cohere_compass_text`, so this top-level arm has to
        // win or the VLM would load as a text-only Command model.
        "cohere_compass" => Ok(ModelType::CohereCompassVLM),
        "internlm2" => Ok(ModelType::InternLM2),
        "internlm3" => Ok(ModelType::InternLM3),
        "baichuan_m1" => Ok(ModelType::Baichuan),
        "bitnet" => Ok(ModelType::BitNet),
        "glm4" => Ok(ModelType::Glm4),
        "glm4_moe" => Ok(ModelType::Glm4Moe),
        "solar_open" => Ok(ModelType::SolarOpen),
        "glm4_moe_lite" => Ok(ModelType::Glm4MoeLite),
        "glm_moe_dsa" => Ok(ModelType::GlmMoeDsa),
        "ernie4_5" | "ernie4.5" => Ok(ModelType::Ernie45),
        "ernie4_5_moe" | "ernie4.5_moe" => Ok(ModelType::Ernie45Moe),
        "ernie4_5_moe_vl" | "ernie4.5_moe_vl" => Ok(ModelType::Ernie45MoeVLM),
        "hunyuan_v1_dense" | "hunyuan_dense" => Ok(ModelType::HunyuanV1Dense),
        "hunyuan_vl" => Ok(ModelType::HunyuanVLM),
        "hunyuan" => Ok(detect_hunyuan_model_type(&v)),
        "mimo" => Ok(ModelType::MiMo),
        "bailing_moe" => Ok(ModelType::BailingMoe),
        "bailing_moe_linear" => Ok(ModelType::BailingMoeLinear),
        "afmoe" => Ok(ModelType::Afmoe),
        // `Kwai-Klear/Klear-46B-A2.5B-Instruct` declares the CAPITALIZED
        // `"Klear"`. It matches this lowercase arm only because
        // `model_type_raw` is lowercased above; mlx-lm, which does not
        // normalize, has to ship `Klear.py` and a byte-identical `klear.py` to
        // cover both spellings. Without that normalization this arm would miss
        // every published checkpoint.
        "klear" => Ok(ModelType::Klear),
        "apertus" => Ok(ModelType::Apertus),
        "seed_oss" => Ok(ModelType::SeedOss),
        "granite" => Ok(ModelType::Granite),
        "exaone" => Ok(ModelType::ExaOne),
        "exaone4" => Ok(ModelType::ExaOne4),
        "exaone_moe" => Ok(ModelType::ExaOneMoe),
        "olmo" => Ok(ModelType::Olmo),
        "olmo2" => Ok(ModelType::Olmo2),
        "olmo3" => Ok(ModelType::Olmo3),
        "openelm" => Ok(ModelType::OpenElm),
        "gpt2" => Ok(ModelType::Gpt2),
        "gpt_bigcode" => Ok(ModelType::GptBigCode),
        "gpt_neox" => Ok(ModelType::GptNeoX),
        "helium" => Ok(ModelType::Helium),
        "telechat3" => Ok(ModelType::TeleChat3),
        "starcoder2" => Ok(ModelType::StarCoder2),
        "mellum" => Ok(ModelType::Mellum),
        "laguna" => Ok(ModelType::Laguna),
        "minicpm" => Ok(ModelType::MiniCPM),
        "minicpm3" => Ok(ModelType::MiniCPM3),
        "stablelm" => Ok(ModelType::StableLM),
        "smollm3" => Ok(ModelType::SmolLM3),
        "ministral3" => Ok(ModelType::Ministral3),
        "mistral3" => Ok(detect_text_or_vlm(
            &v,
            ModelType::Mistral3,
            ModelType::Mistral3VLM,
        )),
        "mistral4" => Ok(ModelType::Mistral4),
        "nemotron" => Ok(ModelType::Nemotron),
        "mamba" | "falcon_mamba" => Ok(ModelType::Mamba),
        "mamba2" => Ok(ModelType::Mamba2),
        "jamba" => Ok(ModelType::Jamba),
        "falcon_h1" => Ok(ModelType::FalconH1),
        "lfm2" => Ok(ModelType::Lfm2),
        "lfm2_vl" | "lfm2-vl" => Ok(ModelType::Lfm2VL),
        "lfm2_moe" => Ok(ModelType::Lfm2Moe),
        "inkling_mm_model" | "inkling" => {
            if has_vision_config(&v) && inkling_has_vision_weights(model_path) {
                Ok(ModelType::InklingVLM)
            } else {
                Ok(ModelType::Inkling)
            }
        }
        "plamo2" => Ok(ModelType::Plamo2),
        "granitemoehybrid" => Ok(ModelType::GraniteMoeHybrid),
        "nemotron_h" => Ok(ModelType::NemotronH),
        "nemotron_h_nano_omni" | "nemotronh_nano_omni_reasoning_v3" => {
            Ok(ModelType::NemotronHNanoOmniVLM)
        }
        "nemotron-nas" => Ok(ModelType::NemotronNAS),
        "rwkv7" => Ok(ModelType::Rwkv7),
        "kimi_linear" => Ok(ModelType::KimiLinear),
        // Kimi K3 loads as the VLM when `config.json` carries `vision_config`
        // and the checkpoint ships `vision_tower.*` tensors (#1342); a copy
        // without either is the text backbone, whose sanitizer drops any
        // stray vision keys.
        "kimi_k3" => {
            if has_vision_config(&v) && kimi_k3_has_vision_weights(model_path) {
                Ok(ModelType::KimiK3VLM)
            } else {
                Ok(ModelType::KimiK3)
            }
        }
        "kimi_vl" => Ok(ModelType::KimiVL),
        "kimi_k25" => Ok(ModelType::KimiK25),
        // LocateAnything: MoonViT tower + MLP connector + Qwen2 text decoder.
        // The text sub-config also says "qwen2", so this arm must win at the
        // top level or the grounding VLM would load as a text-only Qwen2.
        "locateanything" => Ok(ModelType::LocateAnythingVLM),
        "longcat_flash" => Ok(ModelType::LongcatFlash),
        "longcat_flash_ngram" => Ok(ModelType::LongcatFlashNgram),
        "step3p5" => Ok(ModelType::Step3p5),
        "step3p7" => Ok(ModelType::Step3p7),
        "recurrent_gemma" | "griffin" => Ok(ModelType::RecurrentGemma),
        "qwen2_vl" => Ok(ModelType::Qwen2VL),
        "qwen2_5_vl" => Ok(ModelType::Qwen25VL),
        "qwen3_vl" => Ok(ModelType::Qwen3VL),
        "qwen3_vl_moe" => Ok(ModelType::Qwen3VLMoe),
        "qwen3_omni_moe" => Ok(ModelType::Qwen3OmniMoe),
        "paddleocr_vl" => Ok(ModelType::PaddleOcrVL),
        "dots_ocr" => Ok(ModelType::DotsOcrVL),
        "falcon_ocr" => Ok(ModelType::FalconOcrVL),
        "glm4v" => Ok(ModelType::Glm4v),
        "glm4v_moe" => Ok(ModelType::Glm4vMoe),
        "glm_ocr" => Ok(ModelType::GlmOcr),
        // Text-only Youtu-LLM. The vendor checkpoint labels itself `youtu` and
        // a community conversion `youtu_llm`; both were rejected outright
        // before #1371. A third conversion relabels itself `deepseek_v2` for
        // mlx-lm compatibility and keeps the DeepSeek-V2 arm above.
        "youtu" | "youtu_llm" => Ok(ModelType::YoutuLLM),
        "youtu_vl" => Ok(ModelType::YoutuVLM),
        "internvl_chat" => Ok(ModelType::InternVLChatVLM),
        // LLM-jp-VL: one architecture, two released text backbones
        // (`llm_config.model_type` is `llama` for llm-jp-4-vl-9B-beta and
        // `qwen3` for Jagle-VL-2.2B). Both label themselves `llmjpvl`.
        "llmjpvl" => Ok(ModelType::LlmJpVLM),
        // GOT-OCR 2.0 declares `model_type: "GOT"` in upper case; the
        // normalization above lowercases it before this match.
        "got" => Ok(ModelType::GotOcrVLM),
        // SmolVLM2 ships as `smolvlm`/`smolvlm2`. SmolVLM-Instruct ships as an
        // Idefics3 checkpoint (`idefics3`, `Idefics3ForConditionalGeneration`):
        // a SigLIP vision tower + pixel-shuffle connector + Llama text backbone,
        // which is exactly what the SmolVLM runtime implements.
        "smolvlm" | "smolvlm2" | "idefics3" => Ok(ModelType::SmolVLM),
        // Idefics2 shares SmolVLM's SigLIP tower but uses a perceiver-resampler
        // connector and a Mistral text backbone, so it gets its own runtime.
        "idefics2" => Ok(ModelType::Idefics2),
        "minicpmo" => Ok(ModelType::MiniCPMOVLM),
        "minicpmv4_6" => Ok(ModelType::MiniCPMV46VLM),
        "moondream3" => Ok(ModelType::Moondream3VLM),
        "moondream2" | "moondream1" => Ok(ModelType::Moondream2VLM),
        "granite_vision" => Ok(ModelType::GraniteVisionVLM),
        "granite4_vision" => Ok(ModelType::Granite4VisionVLM),
        "deepseekocr" => Ok(ModelType::DeepSeekOcrVLM),
        "deepseekocr_2" => Ok(ModelType::DeepSeekOcr2VLM),
        "unlimited-ocr" | "unlimited_ocr" => Ok(ModelType::UnlimitedOcrVLM),
        "deepseek_vl_v2" | "deepseek_vl2" => Ok(ModelType::DeepSeekVL2),
        "llava" | "llava_next" => {
            // The original IBM Granite Vision checkpoint ships as `llava_next`
            // with a `granite` text backbone; route it to the Granite VLM.
            let text_model_type = v
                .get("text_config")
                .and_then(|t| t.get("model_type"))
                .and_then(|m| m.as_str())
                .unwrap_or("");
            if text_model_type == "granite" {
                Ok(ModelType::GraniteVisionVLM)
            } else {
                Ok(ModelType::LlavaVLM)
            }
        }
        "llava_bunny" | "bunny-llama" | "llava-qwen2" => Ok(ModelType::LlavaBunnyVLM),
        "fastvlm" | "llava_qwen2" => Ok(ModelType::FastVLM),
        "aya_vision" => Ok(ModelType::AyaVisionVLM),
        "paligemma" => Ok(ModelType::PaliGemmaVLM),
        "pixtral" => Ok(ModelType::PixtralVLM),
        // The released checkpoints (and both sub-configs) spell this `jvlm`;
        // `jina_vlm` is only the upstream mlx-vlm module name and is accepted
        // as an alias so a hand-edited config still routes.
        "jvlm" | "jina_vlm" => Ok(ModelType::JinaVLM),
        "molmo" => Ok(ModelType::MolmoVLM),
        "molmo2" => Ok(ModelType::Molmo2VLM),
        "molmo_point" => Ok(ModelType::MolmoPointVLM),
        // Florence-2 (DaViT tower + BART encoder-decoder text stack).
        "florence2" => Ok(ModelType::Florence2VLM),
        // Speech-to-text (encoder-decoder ASR).
        "whisper" => Ok(ModelType::Whisper),
        _ => Err(anyhow::anyhow!(
            "Unsupported model type: {}",
            model_type_raw
        )),
    }
}
