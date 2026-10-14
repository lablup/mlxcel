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

//! Multimodal embeddings for media named by local paths, through the server's
//! own preparation (issue #2173).
//!
//! `mlxcel generate` takes `--image`, `--audio` and `--video` as file paths.
//! It used to dispatch them to the vision and audio towers with a CLI copy of
//! the server's per-family dispatch; [`prepare_local_vlm_embeddings`] reads the
//! files and hands them to
//! [`prepare_request_vlm_embeddings`](crate::server::model_provider::model_worker::prepare_request_vlm_embeddings),
//! the function the model worker runs for every chat request, so the two
//! front ends prepare the same media the same way. Paths named on the command
//! line are trusted input, so videos are opened by path rather than through
//! the request route's fd-backed resolver.
//!
//! One family keeps a CLI-specific step. Inkling's server path expects the
//! ordered media markers the chat route renders into the prompt, which
//! `generate`'s own render does not produce, so Inkling audio and video take
//! the server's Inkling functions with the prompt layout `generate` always
//! used: `Structured` after the chat template and `Plain` under
//! `--no-chat-template` ([`inkling_cli_layouts`]).

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};

use crate::LoadedModel;
use crate::multimodal::video::VideoSource;
use crate::server::batch::BatchObservability;
use crate::server::media::ResolvedVideo;
use crate::server::model_provider::model_worker::{
    prepare_inkling_audio_embeddings, prepare_inkling_video_embeddings,
    prepare_request_vlm_embeddings,
};
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::merge::InputEmbeddings;
use crate::vlm_runtime::{
    InklingAudioPromptLayout, InklingPromptTokenIds, InklingVideoPromptLayout,
};

/// Media named on the command line for one prompt.
#[derive(Debug, Clone, Copy)]
pub struct LocalMedia<'a> {
    /// Image files, in prompt order.
    pub images: &'a [PathBuf],
    /// One audio clip.
    pub audio: Option<&'a Path>,
    /// Video files a native video family decodes itself.
    pub videos: &'a [PathBuf],
    /// Sampling rate for the videos, in frames per second.
    pub fps: f64,
    /// The Gemma 4 image soft-token budget, already validated.
    pub image_soft_tokens: Option<usize>,
    /// `--no-chat-template`: the prompt is raw text with no message parts.
    pub no_chat_template: bool,
}

/// The Inkling prompt layouts for a prompt `generate` rendered itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InklingCliLayouts {
    /// Layout for `--video` clips.
    pub video: InklingVideoPromptLayout,
    /// Layout for the `--audio` clip.
    pub audio: InklingAudioPromptLayout,
}

/// The layouts `generate` uses for Inkling media: `Plain` for a raw
/// `--no-chat-template` prompt (no message parts to insert into), otherwise
/// `Structured` over the chat template's part markers. `prompt_ids` resolves
/// those markers and is only called on the templated path, so a raw prompt
/// never needs them.
pub fn inkling_cli_layouts(
    no_chat_template: bool,
    prompt_ids: impl FnOnce() -> Result<InklingPromptTokenIds>,
) -> Result<InklingCliLayouts> {
    if no_chat_template {
        return Ok(InklingCliLayouts {
            video: InklingVideoPromptLayout::Plain,
            audio: InklingAudioPromptLayout::Plain,
        });
    }
    let ids = prompt_ids()?;
    Ok(InklingCliLayouts {
        video: InklingVideoPromptLayout::Structured(ids),
        audio: InklingAudioPromptLayout::Structured(ids),
    })
}

/// Which Inkling-specific path a command-line request takes, if any. Video
/// and audio together fall through to the server dispatch, which refuses the
/// combination for every family but Gemma 4 Unified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InklingCliMedia {
    Video,
    Audio,
}

fn inkling_cli_media(
    is_inkling: bool,
    has_audio: bool,
    has_video: bool,
) -> Option<InklingCliMedia> {
    match (is_inkling, has_audio, has_video) {
        (true, false, true) => Some(InklingCliMedia::Video),
        (true, true, false) => Some(InklingCliMedia::Audio),
        _ => None,
    }
}

/// Read every file in `paths`, naming the one that fails.
fn read_files(paths: &[PathBuf], kind: &str) -> Result<Vec<Vec<u8>>> {
    paths
        .iter()
        .map(|path| {
            std::fs::read(path).with_context(|| format!("failed to read {kind} {}", path.display()))
        })
        .collect()
}

/// The videos in the worker's resolved form, opened by path at `fps`.
fn resolved_videos(paths: &[PathBuf], fps: f64) -> Vec<ResolvedVideo> {
    paths
        .iter()
        .map(|path| ResolvedVideo {
            source: VideoSource::Path(path.clone()),
            fps: Some(fps),
            temp_guard: None,
        })
        .collect()
}

/// Prepare `prompt_tokens` and the input embeddings for `media`, exactly as
/// the model worker prepares a chat request carrying the same bytes.
/// Returns `None` when the model consumes the prompt as text.
pub fn prepare_local_vlm_embeddings(
    model: &LoadedModel,
    tokenizer: &MlxcelTokenizer,
    prompt: &str,
    prompt_tokens: &mut Vec<i32>,
    media: LocalMedia<'_>,
) -> Result<Option<InputEmbeddings>> {
    let images = read_files(media.images, "image")?;
    let audio = match media.audio {
        Some(path) => read_files(&[path.to_path_buf()], "audio")?,
        None => Vec::new(),
    };
    let videos = resolved_videos(media.videos, media.fps);
    let cancelled = AtomicBool::new(false);
    let observability = BatchObservability::new();
    if let LoadedModel::InklingVLM(inkling) = model
        && let Some(kind) = inkling_cli_media(true, !audio.is_empty(), !videos.is_empty())
    {
        let layouts = inkling_cli_layouts(media.no_chat_template, || {
            crate::vlm_runtime::resolve_inkling_prompt_token_ids(tokenizer)
        })?;
        return match kind {
            InklingCliMedia::Video => prepare_inkling_video_embeddings(
                inkling,
                tokenizer,
                prompt_tokens,
                &images,
                &videos,
                layouts.video,
            ),
            InklingCliMedia::Audio => prepare_inkling_audio_embeddings(
                model,
                tokenizer,
                prompt,
                prompt_tokens,
                &images,
                &audio,
                Some(layouts.audio),
                &cancelled,
                &observability,
            ),
        };
    }
    prepare_request_vlm_embeddings(
        model,
        tokenizer,
        prompt,
        prompt_tokens,
        &images,
        &audio,
        &videos,
        None,
        media.image_soft_tokens,
        &cancelled,
        &observability,
    )
}

#[cfg(test)]
#[path = "local_media_tests.rs"]
mod tests;
