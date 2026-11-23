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

//! Clef and Clef-Flash (`model_type: clef`, `nativ-community/clef-*`).
//!
//! A Qwen3.5 vision-language backbone with a joint schema head that scores
//! every option of every question in one forward pass, reading text, images
//! and videos. Port of mlx-vlm's `mlx_vlm/models/clef/clef.py`:
//!
//! - The backbone loads through the Qwen3.5 VLM loader (`language_model.*`,
//!   `vision_tower.*`); the head (`head.*`) loads separately in its stored
//!   dtype and is never quantized ([`super::clef_head`]).
//! - The sequence is upstream's `_sequence` ([`super::clef_sequence`]): the
//!   fixed system prompt, the media, the state truncated so the whole
//!   sequence stays within 16384 tokens, the schema and the answer suffix.
//! - One forward over the whole sequence (batch 1) with the media merged at
//!   their placeholder tokens and MRoPE positions, then the head reads the
//!   post-norm hidden, the question and option span means, and the option
//!   lexical vectors (mean `lm_head` rows of each option span, dequantized).
//!
//! Upstream's `fps` / `num_frames` keyword overrides are Python arguments
//! only and are not ported.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use image::DynamicImage;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::{Map, Value, json};

use super::clef_head::{
    HeadConfig, HeadInputs, JointSchemaHead, sanitize_head_weights, span_means,
};
use super::clef_media::{
    CLIP_FPS, MAX_DECODED_VIDEO_BYTES, VideoSettings, frame_count, linspace_indices, resize_frames,
    video_frame_size,
};
use super::clef_sequence::{
    ClefRow, ClefTokenizer, check_placeholders, frame, gated_media_block, place_state,
    placeholder_count, type_id,
};
use super::decider2_rows::{calibrated_softmax, round_to};
use super::{
    Answer, DecisionError, DecisionMedia, DecisionModel, DecisionResult, DecisionUsage, MediaKind,
    Question, QuestionKind,
};
use crate::multimodal::video::{VideoError, VideoLimits, VideoSource, load_video_source_frames_at};
use crate::vision::Qwen35VLModel;
use crate::vision::processors::qwen2_vl::{
    Qwen2VLMediaInput, Qwen2VLProcessor, QwenVideoProcessorConfig,
};

/// `config.json` `model_type` the loader detects.
pub(crate) const CLEF_MODEL_TYPE: &str = "clef";

const CLEF_TYPES: &[QuestionKind] = &[
    QuestionKind::Choice,
    QuestionKind::Score,
    QuestionKind::Bool,
];
const CLEF_MEDIA: &[MediaKind] = &[MediaKind::Image, MediaKind::Video];

fn internal(error: impl std::fmt::Display) -> DecisionError {
    DecisionError::Internal(format!("clef: {error}"))
}

/// Whether a checkpoint key belongs to the backbone (upstream `sanitize`);
/// every other key is a head tensor.
fn is_backbone_key(key: &str) -> bool {
    ["model.", "lm_head", "language_model.", "vision_tower."]
        .iter()
        .any(|prefix| key.starts_with(prefix))
}

/// One video, sampled, resized and timed.
struct PreparedVideo {
    frames: Vec<DynamicImage>,
    group_seconds: Vec<f64>,
    /// Bytes the source-resolution frames took when decoded (four per
    /// pixel), charged against the request's [`MAX_DECODED_VIDEO_BYTES`].
    decoded_bytes: usize,
}

/// A video error as the client sees it: the server-side path (an allowlisted
/// file or a download's temp file) replaced by "the video".
pub(crate) fn video_error_message(error: &VideoError) -> String {
    let text = error.to_string();
    let path = match error {
        VideoError::FileNotFound(path) | VideoError::EmptyVideo(path) => Some(path),
        VideoError::Probe { path, .. }
        | VideoError::Extract { path, .. }
        | VideoError::Io { path, .. }
        | VideoError::Decode { path, .. } => Some(path),
        _ => None,
    };
    match path.map(|path| path.display().to_string()) {
        Some(path) if !path.is_empty() => text.replace(&path, "the video"),
        _ => text,
    }
}

/// The loaded Clef scorer.
pub(crate) struct ClefModel {
    vlm: Qwen35VLModel,
    head: JointSchemaHead,
    tokenizer: ClefTokenizer,
    /// The video frames' processor: same patching and normalization as the
    /// image processor, but frames arrive already resized to the clip size.
    video_processor: Qwen2VLProcessor,
    video_settings: VideoSettings,
}

// SAFETY: the decision worker and `mlxcel decide` both build the scorer and
// call `predict` on one thread (the worker runs its loader closure on its own
// thread), so the MLX handles and the backbone's single-threaded MRoPE state
// inside never move between threads while in use. `DecisionModel` requires
// `Send` only so the boxed scorer can be returned from that loader.
unsafe impl Send for ClefModel {}

fn read_json(path: &Path) -> Result<Option<Value>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Some(
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?,
    ))
}

/// `image_processor` `min_pixels` / `max_pixels` (or `size` edges).
fn image_pixel_bounds(processor_config: Option<&Value>) -> Option<(usize, usize)> {
    let section = processor_config?.get("image_processor")?;
    let get = |size_key: &str, key: &str| {
        section
            .get("size")
            .and_then(|size| size.get(size_key))
            .or_else(|| section.get(key))
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
    };
    Some((
        get("shortest_edge", "min_pixels")?,
        get("longest_edge", "max_pixels")?,
    ))
}

impl ClefModel {
    /// Load the checkpoint at `model_dir`.
    pub(crate) fn load(model_dir: &Path) -> Result<Self> {
        let config = read_json(&model_dir.join("config.json"))?
            .ok_or_else(|| anyhow!("{} has no config.json", model_dir.display()))?;
        let head_config = HeadConfig::from_config(&config)?;
        let loaded = crate::loading::load_qwen3_5_vlm(model_dir)
            .with_context(|| format!("loading the Clef backbone from {}", model_dir.display()))?;
        let crate::LoadedModel::Qwen35VLM(mut vlm) = loaded else {
            bail!("{} did not load as a Qwen3.5 VLM", model_dir.display());
        };
        let text_hidden = vlm.text_model.config.hidden_size;
        if head_config.hidden_size != text_hidden {
            bail!(
                "clef head_config.hidden_size {} does not match the backbone hidden size {text_hidden}",
                head_config.hidden_size
            );
        }
        let raw = mlxcel_core::weights::load_weights_from_dir_filtered(model_dir, |key| {
            !is_backbone_key(key)
        })
        .map_err(|error| anyhow!("loading the Clef head tensors: {error}"))?;
        let head = JointSchemaHead::from_weights(&sanitize_head_weights(raw)?, &head_config)?;
        let tokenizer = ClefTokenizer::load(model_dir)?;

        let processor_config = read_json(&model_dir.join("processor_config.json"))?;
        if let Some((min_pixels, max_pixels)) = image_pixel_bounds(processor_config.as_ref()) {
            vlm.processor.min_pixels = min_pixels;
            vlm.processor.max_pixels = max_pixels;
        }
        let video_settings = VideoSettings::from_processor_config(processor_config.as_ref());
        let image = &vlm.processor;
        let video_processor = Qwen2VLProcessor::new_with_norm(
            image.patch_size,
            image.temporal_patch_size,
            image.spatial_merge_size,
            image.mean,
            image.std,
        )
        .with_pixel_bounds(1, usize::MAX)
        .with_video_config(QwenVideoProcessorConfig {
            fps: video_settings.fps,
            min_frames: 1,
            max_frames: video_settings.max_frames,
        });
        Ok(Self {
            vlm,
            head,
            tokenizer,
            video_processor,
            video_settings,
        })
    }

    fn merged_tokens(&self, (t, h, w): (i32, i32, i32)) -> usize {
        let merge = self.vlm.spatial_merge_size as i32;
        (t * (h / merge) * (w / merge)).max(0) as usize
    }

    /// Sample, decode, resize and time one video within `budget` bytes.
    fn prepare_video(
        &self,
        source: &VideoSource,
        budget: usize,
    ) -> Result<PreparedVideo, DecisionError> {
        let settings = self.video_settings;
        let sampled =
            load_video_source_frames_at(source, &VideoLimits::from_env(), Some(budget), |total| {
                Ok(linspace_indices(total, frame_count(total, &settings)))
            })
            .map_err(|error| {
                DecisionError::InvalidInput(format!(
                    "could not read the video: {}",
                    video_error_message(&error)
                ))
            })?;
        let temporal = self.video_processor.temporal_patch_size;
        let factor = self.video_processor.patch_size * self.video_processor.spatial_merge_size;
        let first = sampled
            .frames
            .first()
            .ok_or_else(|| DecisionError::InvalidInput("the video has no frames".to_string()))?;
        let decoded_bytes = sampled
            .frames
            .iter()
            .map(|frame| frame.width() as usize * frame.height() as usize * 4)
            .sum();
        let (height, width) = video_frame_size(
            sampled.frames.len(),
            first.height(),
            first.width(),
            temporal,
            factor,
            &settings,
        )?;
        let frames = resize_frames(&sampled.frames, height, width);
        let groups = frames.len().div_ceil(temporal);
        let group_seconds = crate::qwen_vl::qwen_video_group_timestamps(
            &sampled.frame_indices,
            CLIP_FPS,
            temporal,
            groups,
        )
        .map_err(internal)?;
        Ok(PreparedVideo {
            frames,
            group_seconds,
            decoded_bytes,
        })
    }

    /// Merged tokens of one temporal group of a prepared video.
    fn video_group_tokens(&self, video: &PreparedVideo) -> usize {
        let patch = self.video_processor.patch_size as i32;
        video.frames.first().map_or(0, |first| {
            self.merged_tokens((
                1,
                first.height() as i32 / patch,
                first.width() as i32 / patch,
            ))
        })
    }

    /// The media block and the prepared videos of one request.
    ///
    /// Videos are decoded one at a time, sharing [`MAX_DECODED_VIDEO_BYTES`]
    /// of source-resolution frames, and only while the sequence so far
    /// (`text_tokens` of text, the images and the videos already decoded)
    /// still fits [`super::clef_sequence::MAX_LENGTH`], so a request with
    /// more video than the model can read is refused without decoding it.
    fn media(
        &self,
        media: &DecisionMedia,
        text_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<PreparedVideo>), DecisionError> {
        let image_tokens: Vec<usize> = self
            .vlm
            .processor
            .compute_grid_thw(&media.images)
            .into_iter()
            .map(|grid| self.merged_tokens(grid))
            .collect();
        let mut remaining = MAX_DECODED_VIDEO_BYTES;
        gated_media_block(
            &self.tokenizer,
            &image_tokens,
            media.videos.len(),
            text_tokens,
            |index| {
                let video = self.prepare_video(&media.videos[index], remaining)?;
                remaining = remaining.saturating_sub(video.decoded_bytes);
                let tokens = self.video_group_tokens(&video);
                let seconds = video.group_seconds.clone();
                Ok((video, tokens, seconds))
            },
        )
    }

    /// Pixel rows and grids of the images then the videos, in prompt order.
    fn pixels(
        &self,
        images: &[DynamicImage],
        videos: &[PreparedVideo],
    ) -> Result<(UniquePtr<MlxArray>, Vec<(i32, i32, i32)>), DecisionError> {
        let mut parts = Vec::new();
        let mut grids = Vec::new();
        if !images.is_empty() {
            let inputs: Vec<_> = images.iter().map(Qwen2VLMediaInput::Image).collect();
            let (values, grid) = self
                .vlm
                .processor
                .preprocess_media_with_grid(&inputs)
                .map_err(DecisionError::InvalidInput)?;
            parts.push(values);
            grids.extend(grid);
        }
        if !videos.is_empty() {
            let inputs: Vec<_> = videos
                .iter()
                .map(|video| Qwen2VLMediaInput::Video(&video.frames))
                .collect();
            let (values, grid) = self
                .video_processor
                .preprocess_media_with_grid(&inputs)
                .map_err(DecisionError::InvalidInput)?;
            parts.push(values);
            grids.extend(grid);
        }
        let refs: Vec<&MlxArray> = parts.iter().map(|part| &**part).collect();
        Ok((mlxcel_core::concatenate_many(&refs, 0), grids))
    }

    /// Mean `lm_head` row of each option span, `[options, hidden]` in f32.
    fn lexical(&self, ids: &[u32], rows: &[ClefRow]) -> Result<UniquePtr<MlxArray>, DecisionError> {
        let mut gathered = Vec::new();
        let mut spans = Vec::new();
        for &(start, end) in rows.iter().flat_map(|row| &row.option_spans) {
            let first = gathered.len();
            for &id in &ids[start..end] {
                gathered.push(i32::try_from(id).map_err(|_| internal("token id out of range"))?);
            }
            spans.push((first, gathered.len()));
        }
        let index = mlxcel_core::from_slice_i32(&gathered, &[gathered.len() as i32]);
        let text = &self.vlm.text_model;
        let embeddings = match &text.lm_head {
            Some(UnifiedLinear::Quantized { weight, .. }) => {
                let packed = mlxcel_core::take(&weight.weight, &index, 0);
                let scales = mlxcel_core::take(&weight.scales, &index, 0);
                let biases = weight
                    .biases
                    .as_ref()
                    .map(|biases| mlxcel_core::take(biases, &index, 0));
                let biases_ptr = biases
                    .as_ref()
                    .map_or(std::ptr::null(), |b| &**b as *const MlxArray);
                // SAFETY: every pointer is borrowed from an array alive for
                // the call; a null `biases` is the non-affine modes' value.
                let rows = unsafe {
                    mlxcel_core::dequantize(
                        &packed,
                        &scales,
                        biases_ptr,
                        weight.group_size,
                        weight.bits,
                        &weight.mode,
                    )
                };
                match &weight.global_scale {
                    Some(scale) => mlxcel_core::multiply(&rows, scale),
                    None => rows,
                }
            }
            Some(UnifiedLinear::Regular(linear)) => mlxcel_core::take(&linear.weight, &index, 0),
            None => text.embed_tokens.forward(&index),
        };
        Ok(span_means(&embeddings, &spans, dtype::FLOAT32))
    }

    /// The option logits of every row, in row order.
    fn logits(
        &self,
        ids: &[u32],
        rows: &[ClefRow],
        types: Vec<i32>,
        media: Option<(UniquePtr<MlxArray>, Vec<(i32, i32, i32)>)>,
    ) -> Result<Vec<f32>, DecisionError> {
        let ids_i32: Vec<i32> = ids
            .iter()
            .map(|&id| i32::try_from(id).map_err(|_| internal("token id out of range")))
            .collect::<Result<_, _>>()?;
        let input = mlxcel_core::from_slice_i32(&ids_i32, &[1, ids_i32.len() as i32]);
        let text = &self.vlm.text_model;
        let embeds = match &media {
            Some((pixels, grids)) => {
                self.vlm
                    .get_input_embeddings(&input, pixels, grids)
                    .inputs_embeds
            }
            None => text.get_embed_tokens(&input),
        };
        let hidden = text.forward_final_hidden_with_embeddings(&embeds);
        let hidden = self
            .head
            .normalize_hidden(&mlxcel_core::squeeze_axis(&hidden, 0));
        let lexical = self.lexical(ids, rows)?;
        let lexical = mlxcel_core::astype(&lexical, mlxcel_core::array_dtype(&hidden));
        let inputs = HeadInputs {
            question_spans: rows.iter().map(|row| row.question_span).collect(),
            option_spans: rows
                .iter()
                .flat_map(|row| row.option_spans.clone())
                .collect(),
            counts: rows.iter().map(|row| row.option_spans.len()).collect(),
            types,
        };
        let logits = mlxcel_core::astype(
            &self.head.forward(&hidden, &inputs, &lexical),
            dtype::FLOAT32,
        );
        mlxcel_core::try_eval(&logits).map_err(|error| internal(format!("forward: {error}")))?;
        Ok(mlxcel_core::utils::array_to_vec_f32(&logits))
    }
}

/// Upstream's answer for one question from its option probabilities in
/// schema order (`labels`).
pub(crate) fn clef_answer(
    question: &Question,
    labels: &[String],
    p: &[f64],
) -> Result<Answer, DecisionError> {
    let lookup = |label: &str| {
        labels
            .iter()
            .position(|candidate| candidate == label)
            .map(|index| p[index])
            .ok_or_else(|| internal(format!("option `{label}` was not scored")))
    };
    let round4 = |value: f64| round_to(value, 4);
    if question.kind == QuestionKind::Bool {
        return Ok(Answer::bool(round4(lookup("true")?), Map::new()));
    }
    // Request order: choice labels as sent, score levels by index.
    let ordered: Vec<f64> = match question.kind {
        QuestionKind::Choice => question
            .labels()
            .iter()
            .map(|label| lookup(&label.name))
            .collect::<Result<_, _>>()?,
        _ => (0..question.levels().len())
            .map(|index| lookup(&index.to_string()))
            .collect::<Result<_, _>>()?,
    };
    // Python's `max` keeps the first maximum.
    let best =
        ordered.iter().enumerate().fold(
            0,
            |best, (index, value)| if *value > ordered[best] { index } else { best },
        );
    let mut metadata = Map::new();
    metadata.insert("confidence".to_string(), json!(round4(ordered[best])));
    let mut answer = match question.kind {
        QuestionKind::Choice => Answer::choice(
            question,
            ordered.iter().copied().map(round4).collect(),
            metadata,
        )?,
        _ => Answer::score(question, ordered.clone(), metadata)?,
    };
    match &mut answer {
        Answer::Choice { value, .. } => value.clone_from(&question.labels()[best].name),
        Answer::Score {
            value,
            probabilities,
            ..
        } => {
            *value = round4(*value);
            probabilities.iter_mut().for_each(|p| *p = round4(*p));
        }
        _ => {}
    }
    Ok(answer)
}

impl DecisionModel for ClefModel {
    fn decision_types(&self) -> &[QuestionKind] {
        CLEF_TYPES
    }

    fn decision_media(&self) -> &[MediaKind] {
        CLEF_MEDIA
    }

    fn predict(
        &mut self,
        state: Option<&Value>,
        questions: &[(String, Question)],
        media: DecisionMedia,
    ) -> Result<DecisionResult, DecisionError> {
        self.vlm.text_model.clear_mrope_state();
        // Text first, then the media in order, each checked against the
        // cap: a schema that cannot fit is refused before any video is
        // decoded or any pixel is processed.
        let frame = frame(&self.tokenizer, questions, &[])?;
        let has_media = !media.images.is_empty() || !media.videos.is_empty();
        let (frame, videos, pads) = if has_media {
            let (block, videos) = self.media(&media, frame.fixed_len())?;
            let pads = placeholder_count(&self.tokenizer, &block);
            (frame.with_media(&block)?, videos, pads)
        } else {
            (frame, Vec::new(), 0)
        };
        let sequence = place_state(&self.tokenizer, frame, state)?;
        if has_media {
            check_placeholders(&self.tokenizer, &sequence.ids, pads)?;
        }
        let pixels = if has_media {
            Some(self.pixels(&media.images, &videos)?)
        } else {
            None
        };
        drop(videos);
        let types = questions
            .iter()
            .map(|(_, question)| type_id(question.kind))
            .collect();
        let logits = self.logits(&sequence.ids, &sequence.rows, types, pixels)?;
        let mut answers = Vec::with_capacity(questions.len());
        let mut start = 0usize;
        for ((name, question), row) in questions.iter().zip(&sequence.rows) {
            let count = row.option_spans.len();
            let values = logits
                .get(start..start + count)
                .ok_or_else(|| internal("the head returned too few logits"))?;
            start += count;
            let p = calibrated_softmax(values, 1.0);
            answers.push((name.clone(), clef_answer(question, &row.labels, &p)?));
        }
        Ok(DecisionResult {
            answers,
            usage: DecisionUsage {
                input_tokens: sequence.ids.len(),
            },
        })
    }
}
