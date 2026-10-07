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

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};

use crate::LoadedModel;
use crate::multimodal::video::VideoSource;
use crate::server::batch::BatchObservability;
use crate::server::media::ResolvedVideo;
use crate::server::model_provider::model_worker::prepare_request_vlm_embeddings;
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::merge::InputEmbeddings;

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
    let images = media
        .images
        .iter()
        .map(|path| {
            std::fs::read(path).with_context(|| format!("failed to read image {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let audio = match media.audio {
        Some(path) => vec![
            std::fs::read(path)
                .with_context(|| format!("failed to read audio {}", path.display()))?,
        ],
        None => Vec::new(),
    };
    let videos: Vec<ResolvedVideo> = media
        .videos
        .iter()
        .map(|path| ResolvedVideo {
            source: VideoSource::Path(path.clone()),
            fps: Some(media.fps),
            temp_guard: None,
        })
        .collect();
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
        &AtomicBool::new(false),
        &BatchObservability::new(),
    )
}
