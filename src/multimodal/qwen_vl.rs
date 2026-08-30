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

//! Qwen-VL prompt token insertion rules.
//!
//! Qwen2/2.5/3/3.5-VL families reserve image-token blocks based on the image
//! grid and spatial merge size. This module keeps that token arithmetic out of
//! CLI/server callers so Qwen-VL prompt preparation stays consistent.

use crate::models::qwen_mrope_state::MRopeEntry;
use crate::vision;
use crate::vision::feature_cache::{CacheKey, ModelVisionCaches};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::{MlxArray, UniquePtr};

#[derive(Clone, Copy)]
pub struct QwenVlmPromptInfo<'a> {
    pub processor: &'a vision::processors::qwen2_vl::Qwen2VLProcessor,
    pub spatial_merge_size: usize,
    pub vision_start_token_id: i32,
    pub image_token_id: i32,
    pub video_token_id: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedQwenVlmTokens {
    pub image_blocks: usize,
    pub video_blocks: usize,
    pub total_image_tokens: i32,
    pub total_video_tokens: i32,
}

impl InsertedQwenVlmTokens {
    #[must_use]
    pub fn total_visual_tokens(self) -> i32 {
        self.total_image_tokens + self.total_video_tokens
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenVisualKind {
    Image,
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenVisualGrid {
    pub kind: QwenVisualKind,
    pub grid_thw: (i32, i32, i32),
}

/// Opaque container for an MRoPE entry that has been removed from the
/// per-sequence map. Used so callers (e.g. the server preemption path)
/// can carry the entry across operations that release the original
/// sequence id without leaking the underlying `MRopeEntry` type.
pub struct QwenVlMRopeSnapshot(pub(crate) Option<MRopeEntry>);

impl QwenVlMRopeSnapshot {
    /// True when the snapshot holds no entry (text-only or already-released).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }
}

pub trait QwenVlRuntime {
    fn prompt_info(&self) -> QwenVlmPromptInfo<'_>;
    fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> vision::merge::InputEmbeddings;

    /// Variant of [`input_embeddings`] that consults a shared vision feature
    /// cache. Implementors that do not support caching (e.g. older Qwen-VL
    /// variants not yet wired for the cache) should fall through to the plain
    /// [`input_embeddings`] path. The default implementation here matches that
    /// pass-through behavior so trait users always get *something* compiled.
    ///
    /// `caches` is shared per model instance. Runtimes whose vision output
    /// shape matches [`super::feature_cache::SingleArrayFeatures`] use
    /// `caches.single`; Qwen3-VL uses `caches.deepstack`.
    fn input_embeddings_with_cache(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
        _cache_key: Option<&CacheKey>,
        _caches: Option<&ModelVisionCaches>,
    ) -> vision::merge::InputEmbeddings {
        self.input_embeddings(input_ids, pixel_values, grid_thw)
    }

    /// Bind the MRoPE state computed during embedding preparation to a
    /// specific `SequenceId` so the per-row delta cannot leak into other
    /// requests' decode steps (mlx-vlm PR #1095).
    ///
    /// The default implementation is a no-op for runtimes that do not use
    /// MRoPE (in mlxcel today this trait only covers Qwen VL families that
    /// always use MRoPE, but the default keeps the trait additive).
    fn bind_mrope_state_to_sequence(&self, _seq_id: SequenceId) {}

    /// Take the per-sequence MRoPE entry under `seq_id` out of the
    /// language model's per-sequence map. Used by the server preemption
    /// path so the entry survives the eviction (which releases the old
    /// sequence id) and can be reinstalled under the freshly allocated id.
    ///
    /// The default returns an empty snapshot so non-Qwen runtimes are
    /// unaffected.
    fn take_mrope_entry_for_sequence(&self, _seq_id: SequenceId) -> QwenVlMRopeSnapshot {
        QwenVlMRopeSnapshot(None)
    }

    /// Re-install a previously taken MRoPE entry under `seq_id`. The
    /// default is a no-op so non-Qwen runtimes are unaffected.
    fn install_mrope_entry_for_sequence(
        &self,
        _seq_id: SequenceId,
        _snapshot: QwenVlMRopeSnapshot,
    ) {
    }

    // NOTE: per-row batched dispatch lives directly on each
    // `vision::Qwen*VLModel`'s `LanguageModel::forward_batched_with_context_and_ids`
    // override, not on this trait. Most wrappers delegate to the free
    // helper [`forward_batched_with_seq_ids_dispatch`]; Qwen3.5 forwards
    // straight to its text model's batched-with-ids method.
}

/// Per-row batched dispatch helper. Re-exported for backwards
/// compatibility with the Qwen VL wrappers that imported this symbol
/// when the helper lived in this module. The implementation
/// now lives in [`super::batched_dispatch`] so Gemma 4 and
/// the Qwen VL families share a single source of truth — see the
/// duplication report flagged on (M-2).
///
/// Used by: Qwen2VLModel, Qwen25VLModel, Qwen3VLModel, Qwen3VLMoeModel.
pub use super::batched_dispatch::forward_batched_with_seq_ids_dispatch;

/// Per-row dispatch shared by every Qwen VL wrapper whose text model
/// uses the default `forward_batched_with_context_and_ids` trait impl
/// (i.e. all of them except Qwen3.5). Calls the shared helper.
macro_rules! impl_qwen_vl_runtime_loop_dispatch {
    ($ty:ty) => {
        impl QwenVlRuntime for $ty {
            fn prompt_info(&self) -> QwenVlmPromptInfo<'_> {
                QwenVlmPromptInfo {
                    processor: &self.processor,
                    spatial_merge_size: self.spatial_merge_size,
                    vision_start_token_id: self.vision_start_token_id,
                    image_token_id: self.image_token_id,
                    video_token_id: self.video_token_id,
                }
            }

            fn input_embeddings(
                &self,
                input_ids: &MlxArray,
                pixel_values: &MlxArray,
                grid_thw: &[(i32, i32, i32)],
            ) -> vision::merge::InputEmbeddings {
                self.get_input_embeddings(input_ids, pixel_values, grid_thw)
            }

            fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
                self.text_model.bind_mrope_state_to_sequence(seq_id);
            }

            fn take_mrope_entry_for_sequence(&self, seq_id: SequenceId) -> QwenVlMRopeSnapshot {
                QwenVlMRopeSnapshot(self.text_model.take_mrope_entry(seq_id))
            }

            fn install_mrope_entry_for_sequence(
                &self,
                seq_id: SequenceId,
                snapshot: QwenVlMRopeSnapshot,
            ) {
                if let Some(entry) = snapshot.0 {
                    self.text_model.install_mrope_entry(seq_id, entry);
                }
            }
        }
    };
}

// Runtimes without cache wiring (yet) — they fall back to the default
// trait method which just routes through `input_embeddings`.
impl_qwen_vl_runtime_loop_dispatch!(vision::Qwen2VLModel);
impl_qwen_vl_runtime_loop_dispatch!(vision::Qwen3VLMoeModel);
impl_qwen_vl_runtime_loop_dispatch!(vision::Glm4vModel);
impl_qwen_vl_runtime_loop_dispatch!(vision::Glm4vMoeModel);
impl_qwen_vl_runtime_loop_dispatch!(vision::GlmOcrModel);

// Qwen2.5-VL: single-array cache path.
impl QwenVlRuntime for vision::Qwen25VLModel {
    fn prompt_info(&self) -> QwenVlmPromptInfo<'_> {
        QwenVlmPromptInfo {
            processor: &self.processor,
            spatial_merge_size: self.spatial_merge_size,
            vision_start_token_id: self.vision_start_token_id,
            image_token_id: self.image_token_id,
            video_token_id: self.video_token_id,
        }
    }

    fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> vision::merge::InputEmbeddings {
        self.get_input_embeddings(input_ids, pixel_values, grid_thw)
    }

    fn input_embeddings_with_cache(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
        cache_key: Option<&CacheKey>,
        caches: Option<&ModelVisionCaches>,
    ) -> vision::merge::InputEmbeddings {
        self.get_input_embeddings_with_cache(
            input_ids,
            pixel_values,
            grid_thw,
            cache_key,
            caches.map(|c| &c.single),
        )
    }

    fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.text_model.bind_mrope_state_to_sequence(seq_id);
    }

    fn take_mrope_entry_for_sequence(&self, seq_id: SequenceId) -> QwenVlMRopeSnapshot {
        QwenVlMRopeSnapshot(self.text_model.take_mrope_entry(seq_id))
    }

    fn install_mrope_entry_for_sequence(&self, seq_id: SequenceId, snapshot: QwenVlMRopeSnapshot) {
        if let Some(entry) = snapshot.0 {
            self.text_model.install_mrope_entry(seq_id, entry);
        }
    }
}

// Qwen3-VL: DeepStack-shaped cache path.
impl QwenVlRuntime for vision::Qwen3VLModel {
    fn prompt_info(&self) -> QwenVlmPromptInfo<'_> {
        QwenVlmPromptInfo {
            processor: &self.processor,
            spatial_merge_size: self.spatial_merge_size,
            vision_start_token_id: self.vision_start_token_id,
            image_token_id: self.image_token_id,
            video_token_id: self.video_token_id,
        }
    }

    fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> vision::merge::InputEmbeddings {
        self.get_input_embeddings(input_ids, pixel_values, grid_thw)
    }

    fn input_embeddings_with_cache(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
        cache_key: Option<&CacheKey>,
        caches: Option<&ModelVisionCaches>,
    ) -> vision::merge::InputEmbeddings {
        self.get_input_embeddings_with_cache(
            input_ids,
            pixel_values,
            grid_thw,
            cache_key,
            caches.map(|c| &c.deepstack),
        )
    }

    fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.text_model.bind_mrope_state_to_sequence(seq_id);
    }

    fn take_mrope_entry_for_sequence(&self, seq_id: SequenceId) -> QwenVlMRopeSnapshot {
        QwenVlMRopeSnapshot(self.text_model.take_mrope_entry(seq_id))
    }

    fn install_mrope_entry_for_sequence(&self, seq_id: SequenceId, snapshot: QwenVlMRopeSnapshot) {
        if let Some(entry) = snapshot.0 {
            self.text_model.install_mrope_entry(seq_id, entry);
        }
    }
}

// Qwen3.5-VL: text model already implements
// `forward_batched_with_context_and_ids` natively (per-row dispatch and
// batched-prefill fast path), so the wrapper forwards directly to it.
impl QwenVlRuntime for vision::Qwen35VLModel {
    fn prompt_info(&self) -> QwenVlmPromptInfo<'_> {
        QwenVlmPromptInfo {
            processor: &self.processor,
            spatial_merge_size: self.spatial_merge_size,
            vision_start_token_id: self.vision_start_token_id,
            image_token_id: self.image_token_id,
            video_token_id: self.video_token_id,
        }
    }

    fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> vision::merge::InputEmbeddings {
        self.get_input_embeddings(input_ids, pixel_values, grid_thw)
    }

    fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.text_model.bind_mrope_state_to_sequence(seq_id);
    }

    fn take_mrope_entry_for_sequence(&self, seq_id: SequenceId) -> QwenVlMRopeSnapshot {
        QwenVlMRopeSnapshot(self.text_model.take_mrope_entry(seq_id))
    }

    fn install_mrope_entry_for_sequence(&self, seq_id: SequenceId, snapshot: QwenVlMRopeSnapshot) {
        if let Some(entry) = snapshot.0 {
            self.text_model.install_mrope_entry(seq_id, entry);
        }
    }
}

/// Reserve the per-image `<|image_pad|>` runs for a Qwen-VL prompt.
///
/// Qwen-VL prompts carry exactly one `<|image_pad|>` (`image_token_id`) per
/// image, framed by `<|vision_start|>`/`<|vision_end|>`. HF's processor then
/// expands each single placeholder into `t * (h/merge) * (w/merge)` copies so
/// the count matches the vision tower's per-image feature count. mlxcel must do
/// the same, but two prompt shapes reach this function depending on how the
/// prompt was templated upstream:
///
/// 1. **Expand** — the chat template already rendered the canonical
///    `<|vision_start|><|image_pad|><|vision_end|>` framing (one `<|image_pad|>`
///    per image). This happens on the CLI image path when the model template
///    advertises image content (`supports_image_content() == true`, e.g.
///    `qwen2_vl`). Here we expand each single placeholder in place; the framing
///    is already present. (Previously this case was skipped because
///    the prompt "contains" the image token, leaving a single placeholder to
///    face N vision features — a count mismatch that produced zero generated
///    tokens.)
/// 2. **Insert** — the prompt is text-only (no placeholder), e.g. when the
///    template has no image branch and the CLI/server fall back to text
///    rendering (`qwen2_5_vl`, `qwen3_vl`, ...). Here we splice the full framed
///    run (`<|vision_start|>` + `image_token`×N + `<|vision_end|>`) after the
///    first token.
///
/// A placeholder count that is neither `0` nor `grid_thw.len()` (e.g. already
/// expanded) returns `None` so we never double-expand.
///
/// Used by: `multimodal::vlm_runtime` (Qwen2VL, Qwen2.5VL, Qwen3VL, Qwen3VLMoe,
/// Qwen3.5VL image prompt preparation).
/// Qwen3-Omni variant of [`insert_qwen_vl_image_tokens`]: same per-image
/// expansion when placeholders are present, but the no-placeholder fallback
/// splices the framed runs right BEFORE the last `im_end_token_id` (the close
/// of the user turn) instead of after the leading token, keeping the
/// `<|im_start|>system` header intact.
///
/// Used by: Qwen3OmniMoeModel.
pub fn insert_qwen3_omni_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    grid_thw: &[(i32, i32, i32)],
    spatial_merge_size: usize,
    vision_start_token_id: i32,
    image_token_id: i32,
    im_end_token_id: i32,
) -> Option<InsertedQwenVlmTokens> {
    if prompt_tokens.is_empty() || grid_thw.is_empty() || spatial_merge_size == 0 {
        return None;
    }

    let merge = spatial_merge_size as i32;
    let per_image_counts: Vec<i32> = grid_thw
        .iter()
        .map(|&(t, h, w)| t * (h / merge) * (w / merge))
        .collect();
    let total_image_tokens: i32 = per_image_counts.iter().sum();

    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();

    if placeholder_count == grid_thw.len() {
        // Delegate the canonical expand-per-placeholder path.
        return insert_qwen_vl_image_tokens(
            prompt_tokens,
            grid_thw,
            spatial_merge_size,
            vision_start_token_id,
            image_token_id,
        );
    }
    if placeholder_count != 0 {
        return None;
    }

    let vision_end_token_id = vision_start_token_id + 1;
    let mut image_tokens = Vec::with_capacity(total_image_tokens as usize + 2 * grid_thw.len());
    for &count in &per_image_counts {
        image_tokens.push(vision_start_token_id);
        for _ in 0..count {
            image_tokens.push(image_token_id);
        }
        image_tokens.push(vision_end_token_id);
    }

    // End of the last user turn, else after the leading token.
    let insert_at = prompt_tokens
        .iter()
        .rposition(|&t| t == im_end_token_id)
        .unwrap_or(1.min(prompt_tokens.len()));
    let mut spliced = Vec::with_capacity(prompt_tokens.len() + image_tokens.len());
    spliced.extend_from_slice(&prompt_tokens[..insert_at]);
    spliced.extend(image_tokens);
    spliced.extend_from_slice(&prompt_tokens[insert_at..]);
    *prompt_tokens = spliced;

    Some(InsertedQwenVlmTokens {
        image_blocks: grid_thw.len(),
        video_blocks: 0,
        total_image_tokens,
        total_video_tokens: 0,
    })
}

/// Expand or insert Qwen-VL image/video placeholders with exact cardinality.
///
/// The prompt may arrive in three states:
///
/// - no visual placeholders: insert framed runs after BOS in the caller's media
///   order;
/// - one placeholder per image/video item: expand each run in place, preserving
///   prompt order;
/// - already-expanded runs: validate that each run length equals the matching
///   visual grid's feature-row count and leave it untouched.
///
/// Any other shape is rejected before generation, because silently dropping
/// media or scattering feature rows into the wrong modality changes the model's
/// input without a diagnostic.
pub fn insert_qwen_vl_media_tokens(
    prompt_tokens: &mut Vec<i32>,
    media: &[QwenVisualGrid],
    spatial_merge_size: usize,
    vision_start_token_id: i32,
    image_token_id: i32,
    video_token_id: i32,
) -> anyhow::Result<Option<InsertedQwenVlmTokens>> {
    if prompt_tokens.is_empty() || media.is_empty() || spatial_merge_size == 0 {
        return Ok(None);
    }

    let per_media_counts = qwen_media_token_counts(media, spatial_merge_size)?;
    let placeholder_runs = qwen_visual_token_runs(prompt_tokens, image_token_id, video_token_id);

    if placeholder_runs.is_empty() {
        let vision_end_token_id = vision_start_token_id + 1;
        let total_visual_tokens = per_media_counts
            .iter()
            .map(|(_, count)| *count)
            .sum::<i32>();
        let mut visual_tokens =
            Vec::with_capacity(total_visual_tokens as usize + 2usize.saturating_mul(media.len()));
        for ((kind, count), _) in per_media_counts.iter().zip(media) {
            visual_tokens.push(vision_start_token_id);
            let token_id = match kind {
                QwenVisualKind::Image => image_token_id,
                QwenVisualKind::Video => video_token_id,
            };
            visual_tokens.extend(std::iter::repeat_n(token_id, *count as usize));
            visual_tokens.push(vision_end_token_id);
        }

        let bos = prompt_tokens[0];
        let rest = prompt_tokens[1..].to_vec();
        *prompt_tokens = vec![bos];
        prompt_tokens.extend(visual_tokens);
        prompt_tokens.extend(rest);
        return Ok(Some(inserted_qwen_stats(media, &per_media_counts)));
    }

    for pair in placeholder_runs.windows(2) {
        let prev = pair[0];
        let next = pair[1];
        if prev.end == next.start {
            anyhow::bail!(
                "Qwen-VL prompt has adjacent {:?}/{:?} visual token runs with no framing token \
                 between them; refusing to pass ambiguous spans to MRoPE position assignment",
                prev.kind,
                next.kind
            );
        }
    }

    let run_image_count = placeholder_runs
        .iter()
        .filter(|run| run.kind == QwenVisualKind::Image)
        .count();
    let run_video_count = placeholder_runs
        .iter()
        .filter(|run| run.kind == QwenVisualKind::Video)
        .count();
    let expected_image_count = media
        .iter()
        .filter(|m| m.kind == QwenVisualKind::Image)
        .count();
    let expected_video_count = media
        .iter()
        .filter(|m| m.kind == QwenVisualKind::Video)
        .count();
    if run_image_count != expected_image_count || run_video_count != expected_video_count {
        anyhow::bail!(
            "Qwen-VL prompt/media mismatch: prompt has {} image block(s) and {} video block(s), \
             but request provided {} image(s) and {} video(s)",
            run_image_count,
            run_video_count,
            expected_image_count,
            expected_video_count
        );
    }

    for (run, (expected_kind, expected_count)) in placeholder_runs.iter().zip(&per_media_counts) {
        if run.kind != *expected_kind {
            anyhow::bail!(
                "Qwen-VL prompt/media order mismatch: prompt block {} is {:?}, but decoded media \
                 item {} is {:?}",
                run.index,
                run.kind,
                run.index,
                expected_kind
            );
        }
        if run.len != 1 && run.len as i32 != *expected_count {
            anyhow::bail!(
                "Qwen-VL prompt/media cardinality mismatch: {:?} block {} has {} placeholder \
                 token(s), expected either 1 template token or {} expanded feature token(s)",
                run.kind,
                run.index,
                run.len,
                expected_count
            );
        }
    }

    if placeholder_runs
        .iter()
        .all(|run| run.len as i32 == per_media_counts[run.index].1)
    {
        return Ok(Some(inserted_qwen_stats(media, &per_media_counts)));
    }

    if placeholder_runs.iter().any(|run| run.len != 1) {
        anyhow::bail!(
            "Qwen-VL prompt mixes single placeholders and already-expanded visual runs; refusing \
             to guess how decoded feature rows should align"
        );
    }

    let mut expanded = Vec::with_capacity(
        prompt_tokens.len()
            + per_media_counts
                .iter()
                .map(|(_, count)| (*count as usize).saturating_sub(1))
                .sum::<usize>(),
    );
    let mut media_idx = 0usize;
    for &token in prompt_tokens.iter() {
        let kind = if token == image_token_id {
            Some(QwenVisualKind::Image)
        } else if token == video_token_id {
            Some(QwenVisualKind::Video)
        } else {
            None
        };
        if let Some(kind) = kind {
            let (expected_kind, count) = per_media_counts
                .get(media_idx)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("Qwen-VL visual placeholder iterator exhausted"))?;
            if kind != expected_kind {
                anyhow::bail!(
                    "Qwen-VL prompt/media order mismatch: prompt token {} is {:?}, but decoded \
                     media item {} is {:?}",
                    media_idx,
                    kind,
                    media_idx,
                    expected_kind
                );
            }
            expanded.extend(std::iter::repeat_n(token, count as usize));
            media_idx += 1;
        } else {
            expanded.push(token);
        }
    }
    if media_idx != media.len() {
        anyhow::bail!(
            "Qwen-VL expanded {} visual placeholder(s), but {} media item(s) were decoded",
            media_idx,
            media.len()
        );
    }

    *prompt_tokens = expanded;
    Ok(Some(inserted_qwen_stats(media, &per_media_counts)))
}

pub fn insert_qwen_vl_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    grid_thw: &[(i32, i32, i32)],
    spatial_merge_size: usize,
    vision_start_token_id: i32,
    image_token_id: i32,
) -> Option<InsertedQwenVlmTokens> {
    if prompt_tokens.is_empty() || grid_thw.is_empty() || spatial_merge_size == 0 {
        return None;
    }

    let merge = spatial_merge_size as i32;
    let per_image_counts: Vec<i32> = grid_thw
        .iter()
        .map(|&(t, h, w)| t * (h / merge) * (w / merge))
        .collect();
    let total_image_tokens: i32 = per_image_counts.iter().sum();

    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();

    // Case 1: expand one-placeholder-per-image (canonical templated prompt).
    if placeholder_count == grid_thw.len() {
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + total_image_tokens as usize);
        let mut image_idx = 0usize;
        for &token in prompt_tokens.iter() {
            if token == image_token_id {
                let count = per_image_counts[image_idx];
                for _ in 0..count {
                    expanded.push(image_token_id);
                }
                image_idx += 1;
            } else {
                expanded.push(token);
            }
        }
        *prompt_tokens = expanded;
        return Some(InsertedQwenVlmTokens {
            image_blocks: grid_thw.len(),
            video_blocks: 0,
            total_image_tokens,
            total_video_tokens: 0,
        });
    }

    // A non-zero placeholder count that does not match the image count means the
    // prompt was already expanded (or is malformed) — do not touch it.
    if placeholder_count != 0 {
        return None;
    }

    // Case 2: insert framed runs after the first token (text-only prompt).
    let vision_end_token_id = vision_start_token_id + 1;
    let mut image_tokens = Vec::new();
    for &count in &per_image_counts {
        image_tokens.push(vision_start_token_id);
        for _ in 0..count {
            image_tokens.push(image_token_id);
        }
        image_tokens.push(vision_end_token_id);
    }

    let bos = prompt_tokens[0];
    let rest = prompt_tokens[1..].to_vec();
    *prompt_tokens = vec![bos];
    prompt_tokens.extend(image_tokens);
    prompt_tokens.extend(rest);

    Some(InsertedQwenVlmTokens {
        image_blocks: grid_thw.len(),
        video_blocks: 0,
        total_image_tokens,
        total_video_tokens: 0,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QwenVisualTokenRun {
    kind: QwenVisualKind,
    index: usize,
    start: usize,
    end: usize,
    len: usize,
}

fn qwen_visual_token_runs(
    prompt_tokens: &[i32],
    image_token_id: i32,
    video_token_id: i32,
) -> Vec<QwenVisualTokenRun> {
    let mut runs = Vec::new();
    let mut i = 0usize;
    while i < prompt_tokens.len() {
        let kind = if prompt_tokens[i] == image_token_id {
            Some(QwenVisualKind::Image)
        } else if prompt_tokens[i] == video_token_id {
            Some(QwenVisualKind::Video)
        } else {
            None
        };
        let Some(kind) = kind else {
            i += 1;
            continue;
        };
        let start = i;
        let token_id = prompt_tokens[i];
        while i < prompt_tokens.len() && prompt_tokens[i] == token_id {
            i += 1;
        }
        runs.push(QwenVisualTokenRun {
            kind,
            index: runs.len(),
            start,
            end: i,
            len: i - start,
        });
    }
    runs
}

fn qwen_media_token_counts(
    media: &[QwenVisualGrid],
    spatial_merge_size: usize,
) -> anyhow::Result<Vec<(QwenVisualKind, i32)>> {
    let merge = i32::try_from(spatial_merge_size)
        .map_err(|_| anyhow::anyhow!("Qwen-VL spatial_merge_size is too large"))?;
    if merge <= 0 {
        anyhow::bail!("Qwen-VL spatial_merge_size must be positive");
    }
    media
        .iter()
        .map(|item| {
            let (t, h, w) = item.grid_thw;
            if t <= 0 || h <= 0 || w <= 0 {
                anyhow::bail!(
                    "Qwen-VL {:?} grid must be positive, got ({t}, {h}, {w})",
                    item.kind
                );
            }
            if h % merge != 0 || w % merge != 0 {
                anyhow::bail!(
                    "Qwen-VL {:?} grid ({t}, {h}, {w}) is not divisible by spatial_merge_size={}",
                    item.kind,
                    spatial_merge_size
                );
            }
            let count = t
                .checked_mul(h / merge)
                .and_then(|value| value.checked_mul(w / merge))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Qwen-VL {:?} grid ({t}, {h}, {w}) expands past i32 token capacity",
                        item.kind
                    )
                })?;
            Ok((item.kind, count))
        })
        .collect()
}

fn inserted_qwen_stats(
    media: &[QwenVisualGrid],
    counts: &[(QwenVisualKind, i32)],
) -> InsertedQwenVlmTokens {
    let mut stats = InsertedQwenVlmTokens {
        image_blocks: 0,
        video_blocks: 0,
        total_image_tokens: 0,
        total_video_tokens: 0,
    };
    for (item, (_, count)) in media.iter().zip(counts) {
        match item.kind {
            QwenVisualKind::Image => {
                stats.image_blocks += 1;
                stats.total_image_tokens += *count;
            }
            QwenVisualKind::Video => {
                stats.video_blocks += 1;
                stats.total_video_tokens += *count;
            }
        }
    }
    stats
}

/// Build Qwen-VL interleaved MRoPE position IDs for image and video runs.
///
/// The production wrappers call this after prompt placeholder expansion, so a
/// visual token run must already have exactly
/// `grid_t * (grid_h / spatial_merge_size) * (grid_w / spatial_merge_size)`
/// tokens. Videos keep `grid_t` as the temporal axis; only the H/W axes are
/// divided by spatial merge.
pub(crate) fn compute_qwen_vl_mrope_position_ids(
    input_ids: &MlxArray,
    grid_thw: &[(i32, i32, i32)],
    spatial_merge_size: usize,
    image_token_id: i32,
    video_token_id: i32,
) -> UniquePtr<MlxArray> {
    mlxcel_core::eval(input_ids);
    let ids_shape = mlxcel_core::array_shape(input_ids);
    let seq_len = ids_shape[1] as usize;

    let mut tokens = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let tok = mlxcel_core::slice(input_ids, &[0, i as i32], &[1, i as i32 + 1]);
        mlxcel_core::eval(&tok);
        tokens.push(mlxcel_core::item_i32(&tok));
    }

    let pos_ids = qwen_vl_mrope_positions_from_tokens(
        &tokens,
        grid_thw,
        spatial_merge_size,
        image_token_id,
        video_token_id,
    )
    .expect("Qwen-VL visual placeholders must be cardinality-validated before MRoPE");

    qwen_vl_mrope_positions_to_array(pos_ids)
}

fn qwen_vl_mrope_positions_from_tokens(
    tokens: &[i32],
    grid_thw: &[(i32, i32, i32)],
    spatial_merge_size: usize,
    image_token_id: i32,
    video_token_id: i32,
) -> anyhow::Result<[Vec<i32>; 3]> {
    let merge = i32::try_from(spatial_merge_size)
        .map_err(|_| anyhow::anyhow!("Qwen-VL spatial_merge_size is too large"))?;
    if merge <= 0 {
        anyhow::bail!("Qwen-VL spatial_merge_size must be positive");
    }

    let mut pos_ids = [Vec::new(), Vec::new(), Vec::new()];
    let mut visual_idx = 0usize;
    let mut st = 0usize;
    let mut current_pos = 0i32;
    let mut i = 0usize;

    while i < tokens.len() {
        let kind = if tokens[i] == image_token_id {
            Some(QwenVisualKind::Image)
        } else if tokens[i] == video_token_id {
            Some(QwenVisualKind::Video)
        } else {
            None
        };
        let Some(kind) = kind else {
            i += 1;
            continue;
        };

        let vision_start = i;
        let token_id = tokens[i];
        while i < tokens.len() && tokens[i] == token_id {
            i += 1;
        }
        if i < tokens.len() && (tokens[i] == image_token_id || tokens[i] == video_token_id) {
            anyhow::bail!(
                "Qwen-VL MRoPE saw adjacent {:?} visual token run without framing",
                kind
            );
        }

        if vision_start > st {
            let text_len = vision_start - st;
            for p in current_pos..current_pos + text_len as i32 {
                pos_ids[0].push(p);
                pos_ids[1].push(p);
                pos_ids[2].push(p);
            }
            current_pos += text_len as i32;
        }

        let Some(&(t, h, w)) = grid_thw.get(visual_idx) else {
            anyhow::bail!(
                "Qwen-VL MRoPE saw more visual token runs than visual grids: run {}",
                visual_idx
            );
        };
        if t <= 0 || h <= 0 || w <= 0 {
            anyhow::bail!("Qwen-VL MRoPE grid must be positive, got ({t}, {h}, {w})");
        }
        if h % merge != 0 || w % merge != 0 {
            anyhow::bail!(
                "Qwen-VL MRoPE grid ({t}, {h}, {w}) is not divisible by spatial_merge_size={}",
                spatial_merge_size
            );
        }

        let llm_t = t;
        let llm_h = h / merge;
        let llm_w = w / merge;
        let expected_len = llm_t
            .checked_mul(llm_h)
            .and_then(|value| value.checked_mul(llm_w))
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen-VL MRoPE grid ({t}, {h}, {w}) expands past platform token capacity"
                )
            })?;
        let observed_len = i - vision_start;
        if observed_len != expected_len {
            anyhow::bail!(
                "Qwen-VL MRoPE {:?} run {} has {} token(s), expected {} from grid ({t}, {h}, {w})",
                kind,
                visual_idx,
                observed_len,
                expected_len
            );
        }

        for ti in 0..llm_t {
            for hi in 0..llm_h {
                for wi in 0..llm_w {
                    pos_ids[0].push(current_pos + ti);
                    pos_ids[1].push(current_pos + hi);
                    pos_ids[2].push(current_pos + wi);
                }
            }
        }
        current_pos += llm_t.max(llm_h).max(llm_w);
        visual_idx += 1;
        st = i;
    }

    if st < tokens.len() {
        let text_len = tokens.len() - st;
        for p in current_pos..current_pos + text_len as i32 {
            pos_ids[0].push(p);
            pos_ids[1].push(p);
            pos_ids[2].push(p);
        }
    }
    if visual_idx != grid_thw.len() {
        anyhow::bail!(
            "Qwen-VL MRoPE saw {} visual token run(s), but {} visual grid(s) were provided",
            visual_idx,
            grid_thw.len()
        );
    }
    debug_assert_eq!(pos_ids[0].len(), tokens.len());
    Ok(pos_ids)
}

fn qwen_vl_mrope_positions_to_array(pos_ids: [Vec<i32>; 3]) -> UniquePtr<MlxArray> {
    let total_len = pos_ids[0].len() as i32;
    let t_arr = mlxcel_core::from_slice_i32(&pos_ids[0], &[1, 1, total_len]);
    let h_arr = mlxcel_core::from_slice_i32(&pos_ids[1], &[1, 1, total_len]);
    let w_arr = mlxcel_core::from_slice_i32(&pos_ids[2], &[1, 1, total_len]);
    let th = mlxcel_core::concatenate(t_arr.as_ref().unwrap(), h_arr.as_ref().unwrap(), 0);
    mlxcel_core::concatenate(th.as_ref().unwrap(), w_arr.as_ref().unwrap(), 0)
}

#[cfg(test)]
#[path = "qwen_vl_tests.rs"]
mod tests;
