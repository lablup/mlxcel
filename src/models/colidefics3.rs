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

//! ColIdefics3 (`model_type: idefics3`, `architectures: ["ColIdefics3"]`):
//! the SmolVLM / Idefics3 stack turned into a late-interaction visual
//! document retriever.
//!
//! The backbone is exactly the one `mlxcel generate` runs for SmolVLM: a
//! SigLIP vision tower, the `pixel_shuffle(scale_factor)` plus bias-free
//! `modality_projection` connector, and a plain Llama (SmolLM2) decoder.
//! Retrieval changes only the ends of that stack:
//!
//! - the decoder stops at its final norm, and the checkpoint ships no
//!   `lm_head` at all, so it runs through
//!   [`crate::models::headless_llama::HeadlessLlama`] rather than through
//!   [`crate::models::Llama3Model`], which always loads a head;
//! - a single `Linear` (`linear.{weight,bias}`, `[128, 576]`) projects every
//!   token's hidden state to 128 dimensions, each token vector is
//!   L2-normalized and padding rows are zeroed;
//! - similarity is MaxSim over the token sets
//!   ([`crate::embeddings::maxsim`]), not cosine over one pooled vector, so
//!   [`EmbeddingModel::multi_vector`] is `true` and the engine returns
//!   `[num_real_tokens, 128]` per input.
//!
//! Prompt formats follow the reference `ColIdefics3Processor`: an image
//! document is `<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:`
//! and a query is `Query: {text}` followed by ten `<end_of_utterance>`
//! augmentation tokens. The engine tokenizes the formatted string, then
//! asks [`EmbeddingModel::expand_image_tokens`] to replace the single
//! `<image>` placeholder with the framed row / global tile runs the
//! processor emits, so the token count the response reports is the one the
//! forward pass consumed.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, bail};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::utils::create_causal_padding_mask;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::embeddings::loader::{load_embedding_weights, quantization_params};
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput, ImageInput};
use crate::embeddings::pooling::PoolingMode;
use crate::models::col_late_interaction::{
    apply_dense_projection_override, embedding_dim, format_query, project_and_normalize,
    reject_lora_only_checkpoint,
};
use crate::models::headless_llama::HeadlessLlama;
use crate::models::llama3::ModelArgs;
use crate::multimodal::smolvlm_prompt::insert_smolvlm_image_tokens;
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};
use crate::vision::config::VisionConfig;
use crate::vision::encoders::VisionEncoder;
use crate::vision::encoders::siglip::SigLipVisionModel;
use crate::vision::merge;
use crate::vision::processors::smolvlm::{
    DEFAULT_SIGLIP_MEAN, DEFAULT_SIGLIP_STD, SmolVLMProcessor, TileLayout,
};
use crate::vision::smolvlm::SmolVLMConnector;

/// Checkpoint key of the 128-dimension projection.
const PROJECTION_PREFIX: &str = "linear";

/// Pixel-shuffle compression factor when `config.json` omits
/// `scale_factor`. Matches the generation loader's default.
const DEFAULT_SCALE_FACTOR: i32 = 2;

/// `<image>` id when `config.json` omits `image_token_id`.
const DEFAULT_IMAGE_TOKEN_ID: i32 = 49153;

/// Turn terminator that also serves as the query augmentation token.
const END_OF_UTTERANCE: &str = "<end_of_utterance>";

/// The document prompt the reference processor renders for an image item
/// (`ColIdefics3Processor.visual_prompt_prefix`), which is also what the
/// checkpoint's own `chat_template.json` renders for one image message with
/// `add_generation_prompt = true`.
///
/// The checkpoint additionally ships an `additional_chat_templates/sentence_transformers.jinja`
/// that orders the same pieces differently
/// (`<|im_start|>User: Describe the image.<image><end_of_utterance>`). Both
/// were measured on `vidore/colSmol-256M` merged into its base: the
/// relevant-page MaxSim margin was 53.64 percent for this form and 53.77
/// percent for the other, so the two are equivalent in practice and the
/// processor's form is kept.
const IMAGE_DOCUMENT_PROMPT: &str =
    "<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:";

/// ColIdefics3: SmolVLM without a head plus a 128-dimension projection.
pub struct ColIdefics3Model {
    text: HeadlessLlama,
    vision_model: SigLipVisionModel,
    connector: SmolVLMConnector,
    processor: SmolVLMProcessor,
    linear: UnifiedLinear,
    /// Tokenizer used only to encode the processor's tile markers
    /// (`<fake_token_around_image><row_2_col_3>`), which are plain text and
    /// not always added tokens.
    tokenizer: MlxcelTokenizer,
    marker_cache: Mutex<HashMap<String, Vec<i32>>>,
    image_token_id: i32,
    /// Image feature vectors per tile after pixel shuffle,
    /// `(image_size / patch_size / scale_factor)^2`.
    num_image_token: usize,
    embedding_dim: usize,
}

impl ColIdefics3Model {
    /// Load a ColIdefics3 checkpoint from `model_dir`.
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        reject_lora_only_checkpoint(model_dir)?;
        if config
            .get("mask_non_image_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            bail!(
                "this ColIdefics3 checkpoint sets `mask_non_image_embeddings: true`, which drops \
                 every text token's vector from an image document; mlxcel does not implement that \
                 variant"
            );
        }

        let text_args = text_args(config)?;
        let vision_config: VisionConfig = serde_json::from_value(
            config
                .get("vision_config")
                .cloned()
                .ok_or_else(|| anyhow!("ColIdefics3 config.json has no `vision_config`"))?,
        )
        .context("failed to parse the ColIdefics3 `vision_config` block")?;

        let (group_size, bits) = quantization_params(config)
            .map_or((text_args.group_size(), text_args.bits()), |quant| {
                (quant.group_size, quant.bits)
            });

        let mut weights = load_embedding_weights(model_dir, config)?;
        apply_dense_projection_override(&mut weights, PROJECTION_PREFIX);
        weights.retain(|key, _| !key.starts_with("lm_head."));

        let text_weights = text_backbone_weights(&weights);
        let text = HeadlessLlama::from_weights(&text_weights, &text_args)
            .map_err(|e| anyhow!("ColIdefics3 text backbone: {e}"))?;

        let vision_prefix = resolve_prefix(&weights, "model.vision_model", "vision_model");
        let vision_model = SigLipVisionModel::from_weights_with_quant(
            &weights,
            &vision_config,
            vision_prefix,
            group_size,
            bits,
        )
        .map_err(|e| anyhow!("ColIdefics3 SigLIP vision tower: {e}"))?;

        let scale_factor = config
            .get("scale_factor")
            .and_then(Value::as_i64)
            .unwrap_or(DEFAULT_SCALE_FACTOR as i64) as i32;
        let connector_prefix = resolve_prefix(&weights, "model.connector", "connector");
        let connector = SmolVLMConnector::from_weights(
            &weights,
            connector_prefix,
            scale_factor,
            group_size,
            bits,
        )
        .map_err(|e| anyhow!("ColIdefics3 connector: {e}"))?;

        let linear = UnifiedLinear::from_weights(&weights, PROJECTION_PREFIX, group_size, bits)
            .map_err(|e| {
                anyhow!("ColIdefics3 projection `{PROJECTION_PREFIX}` is missing or malformed: {e}")
            })?;

        let processor = image_processor(model_dir, vision_config.image_size);
        // Derived from the vision geometry rather than read from
        // `processor_config.json` so the merge invariant (one `<image>`
        // placeholder per projected feature row) cannot drift.
        let side = (vision_config.image_size / vision_config.patch_size.max(1))
            / (scale_factor.max(1) as usize);
        let num_image_token = (side * side).max(1);

        let image_token_id = config
            .get("image_token_id")
            .or_else(|| config.get("image_token_index"))
            .and_then(Value::as_i64)
            .unwrap_or(DEFAULT_IMAGE_TOKEN_ID as i64) as i32;

        Ok(Self {
            text,
            vision_model,
            connector,
            processor,
            linear,
            tokenizer: load_tokenizer(model_dir)?,
            marker_cache: Mutex::new(HashMap::new()),
            image_token_id,
            num_image_token,
            embedding_dim: embedding_dim(config),
        })
    }

    /// Tokenize one tile marker without special tokens, memoizing the
    /// result: a split image asks for the same handful of marker strings
    /// once per tile.
    fn encode_marker(&self, marker: &str) -> Vec<i32> {
        if let Ok(cache) = self.marker_cache.lock()
            && let Some(tokens) = cache.get(marker)
        {
            return tokens.clone();
        }
        let tokens: Vec<i32> = self
            .tokenizer
            .encode(marker, false)
            .map(|ids| ids.into_iter().map(|id| id as i32).collect())
            .unwrap_or_default();
        if let Ok(mut cache) = self.marker_cache.lock() {
            cache.insert(marker.to_string(), tokens.clone());
        }
        tokens
    }

    /// Tile layout of each image, which fixes how many image tokens its
    /// placeholder expands into.
    fn tile_layouts(&self, images: &[ImageInput]) -> Vec<TileLayout> {
        images
            .iter()
            .map(|input| {
                self.processor
                    .tile_layout(input.image.width(), input.image.height())
            })
            .collect()
    }

    /// Token embeddings with the projected image features written over the
    /// `<image>` positions, mirroring `SmolVLMModel::get_input_embeddings`.
    fn image_embeddings(&self, input_ids: &MlxArray, images: &[ImageInput]) -> UniquePtr<MlxArray> {
        let decoded: Vec<image::DynamicImage> =
            images.iter().map(|input| input.image.clone()).collect();
        let (pixel_values, _layouts) = self.processor.preprocess_with_tiles(&decoded);

        let inputs_embeds = self.text.embed_tokens(input_ids);
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pixels = mlxcel_core::astype(&pixel_values, embed_dtype);
        // [tiles, C, H, W] -> [tiles, H, W, C] for the conv patch embed.
        let pixels = mlxcel_core::transpose_axes(&pixels, &[0, 2, 3, 1]);
        let vision_output = self.vision_model.forward(&pixels);
        let image_features = self.connector.forward(&vision_output.hidden_states);

        merge::merge_llava(
            self.image_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        )
        .inputs_embeds
    }

    /// `[B, L, 128]` token vectors for one padded micro-batch.
    fn forward(&self, batch: &EmbeddingBatch) -> UniquePtr<MlxArray> {
        let images = batch.images.unwrap_or(&[]);
        let embeddings = if images.is_empty() {
            self.text.embed_tokens(batch.input_ids)
        } else {
            self.image_embeddings(batch.input_ids, images)
        };

        let mask = create_causal_padding_mask(batch.attention_mask, 0);
        let mut caches = self.text.make_caches();
        let hidden =
            self.text
                .forward_hidden(batch.input_ids, Some(&embeddings), &mut caches, Some(&mask));
        project_and_normalize(&hidden, &self.linear, batch.attention_mask)
    }
}

impl EmbeddingModel for ColIdefics3Model {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        Ok(EmbeddingOutput {
            embeddings: self.forward(batch),
            last_hidden_state: None,
        })
    }

    /// Reported for the startup log only: a late-interaction family keeps
    /// every token vector and never pools.
    fn default_pooling(&self) -> PoolingMode {
        PoolingMode::LastToken
    }

    fn multi_vector(&self) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        true
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// An empty text is the engine's image path (`embed_image` formats the
    /// empty string), and a non-empty one is a query. Text items can never
    /// be empty: the route and the engine both reject an empty string
    /// before it reaches here.
    fn format_text(&self, text: &str, _instruction: Option<&str>) -> String {
        if text.is_empty() {
            IMAGE_DOCUMENT_PROMPT.to_string()
        } else {
            format_query(text, END_OF_UTTERANCE)
        }
    }

    fn expand_image_tokens(&self, ids: &[u32], images: &[ImageInput]) -> Result<Vec<u32>> {
        if images.is_empty() {
            return Ok(ids.to_vec());
        }
        let layouts = self.tile_layouts(images);
        let mut tokens: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        insert_smolvlm_image_tokens(
            &mut tokens,
            &layouts,
            self.num_image_token,
            self.image_token_id,
            |marker: &str| self.encode_marker(marker),
        )
        .ok_or_else(|| {
            anyhow!(
                "ColIdefics3 could not expand the `<image>` placeholder for {} image(s); the \
                 rendered prompt carried a placeholder count that does not match",
                images.len()
            )
        })?;
        Ok(tokens.into_iter().map(|token| token as u32).collect())
    }
}

/// Parse `text_config` as Llama args, inheriting a top-level `quantization`
/// block when the text config carries none (the layout mlx conversions of
/// SmolVLM use).
fn text_args(config: &Value) -> Result<ModelArgs> {
    let mut text_config = config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow!("ColIdefics3 config.json has no `text_config`"))?;
    if text_config.get("quantization").is_none()
        && let Some(quant) = config.get("quantization")
        && let Some(object) = text_config.as_object_mut()
    {
        object.insert("quantization".to_string(), quant.clone());
    }
    serde_json::from_value(text_config).context("failed to parse the ColIdefics3 `text_config`")
}

/// Collect the decoder tensors under the plain `model.*` roots the Llama
/// blocks expect.
///
/// Both published layouts are accepted, matching the generation loader:
/// `model.text_model.*` (SmolVLM / ColIdefics3 exports) and
/// `language_model.*` (Idefics3 exports). The head is dropped rather than
/// remapped; this backbone has none.
fn text_backbone_weights(weights: &WeightMap) -> WeightMap {
    let mut out = WeightMap::new();
    for (key, value) in weights.iter() {
        let dest = if let Some(rest) = key.strip_prefix("model.text_model.") {
            format!("model.{rest}")
        } else if let Some(rest) = key.strip_prefix("language_model.") {
            if rest.starts_with("lm_head.") {
                continue;
            }
            format!("model.{rest}")
        } else {
            continue;
        };
        out.insert(dest, mlxcel_core::copy(value));
    }
    out
}

/// Pick whichever sub-module prefix the checkpoint actually uses.
fn resolve_prefix<'a>(weights: &WeightMap, hf_prefix: &'a str, bare_prefix: &'a str) -> &'a str {
    if weights.keys().any(|key| key.starts_with(hf_prefix)) {
        hf_prefix
    } else {
        bare_prefix
    }
}

fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

fn nested_longest_edge(config: Option<&Value>, key: &str) -> Option<usize> {
    config?
        .get(key)?
        .get("longest_edge")?
        .as_u64()
        .map(|v| v.max(1) as usize)
}

fn rgb_triplet(config: Option<&Value>, key: &str) -> Option<[f32; 3]> {
    let values = config?.get(key)?.as_array()?;
    if values.len() != 3 {
        return None;
    }
    Some([
        values[0].as_f64()? as f32,
        values[1].as_f64()? as f32,
        values[2].as_f64()? as f32,
    ])
}

/// Build the tile processor from `preprocessor_config.json`, falling back
/// to the vision geometry. `do_image_splitting` defaults to `true`: every
/// published ColIdefics3 checkpoint enables it, and disabling it silently
/// would drop most of a page.
fn image_processor(model_dir: &Path, vision_image_size: usize) -> SmolVLMProcessor {
    let preprocessor = read_json(&model_dir.join("preprocessor_config.json"));
    let tile_size = nested_longest_edge(preprocessor.as_ref(), "max_image_size")
        .unwrap_or(vision_image_size.max(1));
    let longest_edge =
        nested_longest_edge(preprocessor.as_ref(), "size").unwrap_or(tile_size.max(1) * 4);
    let do_image_splitting = preprocessor
        .as_ref()
        .and_then(|config| config.get("do_image_splitting"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let mean = rgb_triplet(preprocessor.as_ref(), "image_mean").unwrap_or(DEFAULT_SIGLIP_MEAN);
    let std = rgb_triplet(preprocessor.as_ref(), "image_std").unwrap_or(DEFAULT_SIGLIP_STD);
    SmolVLMProcessor::new(tile_size, do_image_splitting, longest_edge, mean, std)
}

#[cfg(test)]
#[path = "colidefics3_tests.rs"]
mod colidefics3_tests;
