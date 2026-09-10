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

//! Shared chat-request preparation helpers.
//!
//! Both streaming and non-streaming chat routes should apply the same message
//! flattening, template rendering, and image extraction rules.
//!
//! Used by: routes/chat
//!
//! # Prefix stability guarantees
//!
//! When the prompt prefix cache is enabled, [`prepare_chat_request`]
//! guarantees that rendering the same conversation across turns yields a
//! prompt that is a prefix of the next turn's prompt, so the KV cache is
//! reusable:
//!
//! 1. **`preserve_thinking` defaulting.** With the prompt cache on, unset
//!    `preserve_thinking` defaults to `true`. This disables the rolling
//!    checkpoint stripper (see [`super::chat_template_kwargs::
//!    strip_rolling_checkpoint`]) whose "strip every thinking block before
//!    the latest user turn" rule is the primary source of prefix drift. Opt
//!    out explicitly via `chat_template_kwargs.preserve_thinking = false`
//!    when prefix instability is acceptable.
//! 2. **Template signature in the cache key.** The
//!    [`super::prompt_cache::key::template_sig`] hash is derived from
//!    `(chat_template_source, chat_template_kwargs, tool_choice,
//!    tools_digest)`. Any change in rendering inputs — including a
//!    non-deterministic template tweak or a tool-list reorder — drops the
//!    affected entries cleanly by producing a new `template_sig`.
//! 3. **Known non-determinism sources and how they are handled.**
//!    * `strftime_now` in Jinja (templates like Llama 3.x emit "today's
//!      date"). This is an intentional per-render value; the cache keys the
//!      prompt tokens rather than the template output, so a date-of-day
//!      difference surfaces as a different token prefix and creates a new
//!      bucket that naturally ages out. Documented, not silently masked.
//!    * `<think>` block stripping across turns — invalidated by the
//!      `preserve_thinking=true` default.
//!    * Tool-schema hashing: [`super::prompt_cache::key::tools_digest`] is
//!      order-preserving, so reordering tools invalidates the cache. This
//!      is intentional: HuggingFace templates iterate tools in order and
//!      some models key their protocol to the index.
//!    * Kwargs-key reordering: kwargs are canonicalized with sorted object
//!      keys before hashing, so map insertion-order drift is absorbed.
//!    * `enable_thinking` / future `chat_template_kwargs` additions: any
//!      new kwarg automatically propagates into `template_sig` because it
//!      participates in the canonicalized map hash.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;

use super::chat_template::{ChatMessage, ChatTemplateProcessor, template_rejection_message};
use super::chat_template_kwargs::{
    ChatTemplateKwargs, extract_request_kwargs, merge_server_and_request, strip_rolling_checkpoint,
    strip_think_block,
};
use super::media::{
    MediaRequestMetadata, ResolvedVideo, extract_chat_video_paths, try_extract_chat_audio_data,
    try_extract_chat_image_data,
};
use super::prompt_cache::key::resolve_session_key;
use super::types::request::{
    ContentPart, Message, MessageContent, Tool, ToolChoice, ordered_audio_sentinel,
    ordered_image_sentinel,
};
use super::types::{ChatCompletionRequest, Role};

#[cfg(test)]
thread_local! {
    static HISTORY_BOUNDARY_RENDER_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_history_boundary_render_attempts_for_test() {
    HISTORY_BOUNDARY_RENDER_ATTEMPTS.with(|attempts| attempts.set(0));
}

#[cfg(test)]
pub(crate) fn history_boundary_render_attempts_for_test() -> usize {
    HISTORY_BOUNDARY_RENDER_ATTEMPTS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn history_boundary_render_attempted() {
    HISTORY_BOUNDARY_RENDER_ATTEMPTS.with(|attempts| attempts.set(attempts.get() + 1));
}

#[cfg(not(test))]
fn history_boundary_render_attempted() {}

pub(crate) struct PreparedChatRequest {
    pub(crate) prompt: String,
    /// Token ids a native (non-Jinja) chat renderer produced for this prompt
    /// (#1338).
    ///
    /// `Some` only for Kimi K3, whose XTML renderer emits control-token ids
    /// directly; `prompt` then holds the equivalent text for diagnostics, the
    /// prompt-cache key and the primed-thinking check, but is never the thing
    /// that gets tokenized. `None` for every template-rendered request, which
    /// keeps the existing tokenize-the-string path exactly as it was.
    pub(crate) prompt_token_ids: Option<Vec<i32>>,
    /// b10621 `--prefill-assistant` (#1470): the trailing assistant text this
    /// prompt continues from, when the request had one and prefill is on.
    ///
    /// The prompt already carries it; this copy exists so the route can put it
    /// back at the head of the response when the request sets `echo`, which is
    /// upstream's only consumer of that field
    /// (`task_result_state`: an `is_continuation` request with `echo == false`
    /// primes the parser so the prefill is not re-emitted).
    pub(crate) assistant_prefill: Option<String>,
    /// The same conversation re-rendered with `add_generation_prompt = false`
    /// (issue #1143): the history-boundary form of this request's prompt.
    ///
    /// `Some` only when the prompt cache is enabled, the loaded model supports
    /// snapshot reuse, and the request is text-only; `None` otherwise, and
    /// also `None` when the history render failed or did not come out as a text
    /// prefix of `prompt` (a template that reorders or rewrites history rather
    /// than appending to it).
    ///
    /// Everything the generation prompt adds beyond this point is exactly the
    /// part that cannot survive to the next turn: thought scaffolds, primed
    /// `<think>` markers, and any tail whose retokenization differs from the
    /// sampled ids. Tokenizing this string is therefore the only way to obtain
    /// a token vector that is guaranteed to prefix the next turn's prompt.
    pub(crate) history_prompt: Option<String>,
    pub(crate) image_data: Vec<Vec<u8>>,
    /// Cardinalities before and after the tolerant media resolver.
    ///
    /// Image declaration/resolution mismatches are rejected during request
    /// preparation. XLA also carries this metadata to its worker for backend
    /// capability checks.
    pub(crate) media: MediaRequestMetadata,
    /// Request-scoped Gemma 4 image soft-token budget, resolved from the
    /// `detail` / `max_soft_tokens` fields on the `image_url` content parts
    /// (see [`crate::server::types::request::ImageUrl`]).
    ///
    /// `None` means "no override" and leaves the checkpoint's configured
    /// budget in place, which is the behavior for every request that does not
    /// set either field. Validation happens here, at the request boundary, so
    /// an unsupported value fails the whole request with a 400 before any
    /// image is decoded.
    pub(crate) image_soft_tokens: Option<usize>,
    pub(crate) audio_data: Vec<Vec<u8>>,
    /// Resolved video items (hardened).
    ///
    /// Each entry holds:
    /// * a [`crate::multimodal::video::VideoSource`] handle that the
    ///   model worker passes to
    ///   [`crate::multimodal::video::load_video_source`]. On Unix this is
    ///   the fd-backed variant — the resolver opened the file once after
    ///   canonicalise + allowlist + regular-file + extension checks, and
    ///   ffmpeg reads from that open file description (via `/dev/fd/N`),
    ///   so the canonicalise → ffmpeg-open TOCTOU race is closed at the
    ///   kernel level.
    /// * the optional per-video FPS override from
    ///   [`crate::server::types::request::VideoUrl::fps`].
    /// * an internal RAII guard for any server-owned temp file (data-URI
    ///   decode / HTTP fetch). The guard fires `fs::remove_file` when the
    ///   `PreparedChatRequest` drops, so the temp file lifecycle equals
    ///   the request handler lifecycle.
    pub(crate) videos: Vec<ResolvedVideo>,
}

/// Dedup set for the "defaulted `preserve_thinking=true`" info log.
///
/// Keyed by the resolved `session_key` (see [`resolve_session_key`]) so we
/// log exactly once per logical session per server lifetime. Process-wide
/// state — the dedup tracks the *runtime identity* of the server, not a
/// persistent record. A restart resets the set, which is fine: operators
/// should see the log after each restart confirming the defaulting is live.
pub(super) fn log_once_sessions() -> &'static Mutex<HashSet<String>> {
    static LOGGED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    LOGGED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Turn a template-raised refusal into a request error; leave every other
/// render failure to the caller's fallback (issue #1164).
///
/// [`render_simple_fallback`] exists for checkpoints whose template mlxcel
/// cannot render at all, and dropping to a bare prompt is the right call there:
/// the alternative is refusing to serve a model that otherwise works. It is the
/// wrong call when the template rendered fine and deliberately rejected a
/// caller-supplied value. That produced an HTTP 200 whose prompt had no chat
/// framing, no system message, and no tool declarations, so the client got a
/// plausible answer it had no way to tell apart from a real one, and the only
/// record was a server-side `WARN`.
///
/// The discriminator is [`template_rejection_message`], which downcasts to the
/// sentinel that `raise_exception` attaches as the error's source. It is exact:
/// it cannot fire on an engine failure the way `ErrorKind::InvalidOperation`
/// would, and it does not depend on how a template author worded the message.
///
/// The returned error propagates out of [`prepare_chat_request_with_cache`]
/// with the original render error as its source. That keeps
/// [`template_rejection_message`] usable at outer route boundaries that need to
/// distinguish template rejections from generic render failures.
fn reject_or_return_render_error(err: anyhow::Error) -> Result<anyhow::Error> {
    let Some(message) = template_rejection_message(&err).map(str::to_string) else {
        return Ok(err);
    };
    tracing::info!("Chat template rejected the request: {message}");
    Err(err.context(format!(
        "the model's chat template rejected this request: {message}"
    )))
}

/// Resolve the chat-template kwargs a request renders with.
///
/// Single source of truth for the kwargs map, shared by the rendering pipeline
/// in [`prepare_chat_request_with_cache`], by the prompt-cache context builder
/// in [`super::routes::chat::build_prompt_cache_request_context`], and by the
/// next-turn warm-up render in [`render_next_turn_history`], so the
/// `template_sig` hash, the served prompt, and the warmed vector are all taken
/// over the same map. A caller that derives the merge by hand instead warms or
/// keys a prompt the other two never produce.
///
/// Precedence, highest first:
///
/// 1. Per-request `chat_template_kwargs` (top-level, then nested/flattened
///    `extra_body.chat_template_kwargs`, then the DashScope/OpenAI-SDK
///    `preserve_thinking` alias); see [`extract_request_kwargs`].
/// 2. Per-request reasoning controls, resolved from `reasoning_effort` or the
///    compatible `reasoning` shapes and mapped by
///    [`map_reasoning_control_kwargs`]. They fill `enable_thinking` and the
///    template's `reasoning_effort` / `reasoning_strength` level key only when
///    the explicit per-request kwargs did not already set that key.
/// 3. Server-default `--chat-template-kwargs` / `LLAMA_ARG_CHAT_TEMPLATE_KWARGS`.
///
/// Steps 1 and 2 are both per-request, so they are resolved into the
/// per-request map before the server-default merge. That keeps the module's
/// "per-request wins per-key, unrelated server-default keys persist" rule
/// (see [`super::chat_template_kwargs`]) intact for the new field.
///
/// The prompt-cache `preserve_thinking=true` defaulting is deliberately **not**
/// part of this helper: it is applied by the caller after the merge, and only
/// the rendering pipeline applies it today. Folding it in here would change
/// every existing deployment's `template_sig` on upgrade, which is a cache
/// invalidation this change has no reason to cause. [`render_next_turn_history`]
/// applies the same default itself, for the same reason and with the same
/// value, so the warm-up and the render it mirrors stay in step.
// Used by: chat_request (prepare_chat_request_with_cache, render_next_turn_history), routes/chat
pub(crate) fn resolve_effective_kwargs(
    processor: &ChatTemplateProcessor,
    request: &ChatCompletionRequest,
    server_default_kwargs: Option<&ChatTemplateKwargs>,
    merged_extra_body: &Option<serde_json::Map<String, serde_json::Value>>,
) -> ChatTemplateKwargs {
    let mut per_request_kwargs = extract_request_kwargs(
        request.chat_template_kwargs.as_ref(),
        merged_extra_body.as_ref(),
    );
    map_reasoning_control_kwargs(processor, request, &mut per_request_kwargs);
    merge_server_and_request(server_default_kwargs, &per_request_kwargs)
}

/// Map portable request reasoning controls onto chat-template kwargs.
///
/// Three guards, each deliberate:
///
/// * **An explicit per-request kwarg wins per key.** Derived
///   `enable_thinking`, `reasoning_effort`, and `reasoning_strength` values
///   only fill missing keys. The checkpoint-specific channel therefore keeps
///   precedence over the portable request field.
/// * **The template must mention the level name.** Enabled effort is mapped to
///   `reasoning_effort` when that identifier is present, otherwise to
///   `reasoning_strength` when that alias is present. Templates that read
///   neither do not silently acquire a level kwarg.
/// * **No value translation.** OpenAI's vocabulary is
///   `{minimal, low, medium, high}` and Qwen3.8's is `{xhigh, medium, low}`, so
///   `high` is valid OpenAI and invalid Qwen3.8 while `xhigh` is the reverse.
///   Remapping `high` to `xhigh` would silently change the model's reasoning
///   budget to something the caller did not ask for. The value goes through
///   verbatim, the template decides, and a refusal surfaces as a 400 carrying
///   the template's own message and its accepted set. Disabled values never
///   become a level kwarg because templates commonly reject them. Same
///   reasoning as
///   `resolve_drafter_kind`'s refusal to guess a drafter
///   (`src/lib/mlxcel-core/src/drafter/mod.rs`).
fn map_reasoning_control_kwargs(
    processor: &ChatTemplateProcessor,
    request: &ChatCompletionRequest,
    per_request_kwargs: &mut ChatTemplateKwargs,
) {
    const ENABLE_THINKING: &str = "enable_thinking";
    const REASONING_EFFORT: &str = "reasoning_effort";
    const REASONING_STRENGTH: &str = "reasoning_strength";

    let Some(control) = request.resolve_reasoning_control() else {
        return;
    };

    if !per_request_kwargs.as_map().contains_key(ENABLE_THINKING) {
        per_request_kwargs.set(ENABLE_THINKING, serde_json::Value::Bool(control.enabled));
    }

    if !control.enabled {
        return;
    }
    let Some(effort) = control.effort else {
        return;
    };

    let target = if processor.template_mentions(REASONING_EFFORT) {
        REASONING_EFFORT
    } else if processor.template_mentions(REASONING_STRENGTH) {
        REASONING_STRENGTH
    } else {
        tracing::debug!(
            "request set a reasoning effort but the loaded chat template does not \
             reference reasoning_effort or reasoning_strength; omitting the level kwarg"
        );
        return;
    };

    if per_request_kwargs.as_map().contains_key(target) {
        return;
    }
    per_request_kwargs.set(target, serde_json::Value::String(effort));
}

/// Operator-tunable knobs for the video-to-frames fallback (issue #1322).
///
/// Carried explicitly rather than read from a process global so a test can pin
/// them without mutating the environment, and so the router's per-model
/// `AppState`s each answer with the config they were built from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VideoFramesFallback {
    /// Cap on frames kept per clip. Values below
    /// [`MIN_FALLBACK_MAX_FRAMES`](crate::multimodal::video::MIN_FALLBACK_MAX_FRAMES)
    /// are raised to it here, so a caller cannot ask for a single frame.
    pub(crate) max_frames: usize,
    /// Decode rate used when the `video_url` part carries no `fps` of its own.
    pub(crate) default_fps: f64,
}

impl VideoFramesFallback {
    /// Read the operator settings off a live [`ServerConfig`].
    pub(crate) fn from_config(config: &super::config::ServerConfig) -> Self {
        Self {
            max_frames: config.video_max_frames,
            default_fps: config.video_fps,
        }
    }

    fn effective_max_frames(self) -> usize {
        self.max_frames
            .max(crate::multimodal::video::MIN_FALLBACK_MAX_FRAMES)
    }
}

/// The sentence inserted ahead of a clip's frames so the model reads them as
/// one video rather than as unrelated pictures.
fn video_frames_lead_text(frames: usize) -> String {
    format!("Here is a video as a sequence of {frames} frames in chronological order.")
}

/// Rewrite every `video_url` content part into the sampled frames of that clip,
/// as ordered `image_url` parts (issue #1322).
///
/// Runs after the HTTP boundary's [`media_capability_rejection`] and before
/// [`prepare_chat_request_with_cache`], so the template sees one image
/// placeholder per frame and the unchanged image pipeline carries the bytes to
/// the vision tower. The request is left untouched, and `Ok(0)` returned, for a
/// checkpoint with a native video path: those keep their own temporal
/// processor and `prepared.videos`.
///
/// [`media_capability_rejection`]: crate::server::media_capability_rejection
///
/// # Errors
/// Returns a client-facing message when `ffmpeg` is missing, a referenced clip
/// cannot be resolved or decoded, or the substituted frames would push the
/// request past the per-request image limit.
pub(crate) async fn expand_video_parts_to_frames(
    request: &mut ChatCompletionRequest,
    support: super::state::ModelMediaSupport,
    settings: VideoFramesFallback,
    model_id: &str,
) -> std::result::Result<usize, String> {
    let allowlist = super::media::video_dir_allowlist_from_env();
    expand_video_parts_to_frames_with_allowlist(request, support, settings, model_id, &allowlist)
        .await
}

/// Test-friendly variant of [`expand_video_parts_to_frames`] that takes the
/// directory allowlist directly.
///
/// Same split `extract_chat_video_paths_with_allowlist` uses, and for the same
/// reason: `MLXCEL_VIDEO_DIR_ALLOWLIST` is process-global state, so a test that
/// set it would have to hold the crate env lock across this function's awaits.
pub(crate) async fn expand_video_parts_to_frames_with_allowlist(
    request: &mut ChatCompletionRequest,
    support: super::state::ModelMediaSupport,
    settings: VideoFramesFallback,
    model_id: &str,
    allowlist: &[std::path::PathBuf],
) -> std::result::Result<usize, String> {
    if !support.video_frames_fallback {
        return Ok(0);
    }
    // Collect the positions first: the decode is async and the rewrite shifts
    // every index after it, so doing both in one pass would walk a moving list.
    let targets: Vec<(usize, usize, crate::server::types::request::VideoUrl)> = request
        .messages
        .iter()
        .enumerate()
        .flat_map(|(message_index, message)| match &message.content {
            MessageContent::Parts(parts) => parts
                .iter()
                .enumerate()
                .filter_map(move |(part_index, part)| match part {
                    ContentPart::VideoUrl { video_url } => {
                        Some((message_index, part_index, video_url.clone()))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            MessageContent::Text(_) => Vec::new(),
        })
        .collect();
    if targets.is_empty() {
        return Ok(0);
    }

    if !crate::multimodal::video::ffmpeg_available() {
        return Err(
            "Video input requires `ffmpeg` on PATH. Install ffmpeg (e.g. `brew install ffmpeg` \
             on macOS or `apt install ffmpeg` on Linux) and retry."
                .to_string(),
        );
    }

    let max_frames = settings.effective_max_frames();
    let mut expansions: Vec<(usize, usize, Vec<Vec<u8>>)> = Vec::with_capacity(targets.len());
    for (message_index, part_index, video_url) in targets {
        let resolved = super::media::resolve_video_url(&video_url, allowlist)
            .await
            .ok_or_else(|| {
                format!(
                    "Could not read the video referenced by {:?}. Local paths must sit inside a \
                     directory listed in {}.",
                    video_url.url,
                    super::media::VIDEO_DIR_ALLOWLIST_ENV
                )
            })?;
        let fps = video_url.fps.unwrap_or(settings.default_fps);
        let label = resolved.canonical_path().display().to_string();
        // ffmpeg decode plus PNG encode is seconds of CPU on a long clip, so it
        // runs on the blocking pool like the image decode path does rather than
        // parking a Tokio worker.
        let (kept, sampled) = tokio::task::spawn_blocking(move || {
            let frames =
                crate::multimodal::video::load_video_source(&resolved.source, Some(fps), None)?;
            let sampled = frames.len();
            let kept = crate::multimodal::video::subsample_evenly(frames, max_frames);
            crate::multimodal::video::frames_to_png(&kept).map(|png| (png, sampled))
        })
        .await
        .map_err(|err| format!("Video frame extraction task failed: {err}"))?
        .map_err(|err| format!("Failed to load video {label:?}: {err}"))?;

        tracing::info!(
            "model {model_id} has no native video path; sending {} of {sampled} sampled frames \
             from {label} as ordered images",
            kept.len()
        );
        expansions.push((message_index, part_index, kept));
    }

    // The frames become ordinary image parts, so they spend the same per-request
    // image budget. Refuse here, naming the frames, rather than letting
    // `validate_image_count` report a count the caller never sent.
    let injected: usize = expansions.iter().map(|(_, _, frames)| frames.len()).sum();
    if let Some(message) = video_frame_budget_rejection(
        request.image_urls().len(),
        injected,
        super::media::current_image_input_limits().max_images_per_request,
    ) {
        return Err(message);
    }

    apply_video_frame_expansion(request, expansions);
    Ok(injected)
}

/// Refuse a request whose substituted frames would not fit the per-request
/// image budget, naming the frames.
///
/// `validate_image_count` would refuse it a moment later, but its message
/// reports an image count the caller never sent, which reads as a server bug
/// rather than as a `--video-max-frames` that is too high for this deployment.
fn video_frame_budget_rejection(
    existing_images: usize,
    injected: usize,
    limit: usize,
) -> Option<String> {
    (existing_images + injected > limit).then(|| {
        format!(
            "The video expanded to {injected} frame image(s) alongside {existing_images} image \
             input(s), over the per-request limit of {limit}. Lower --video-max-frames or raise \
             --max-images."
        )
    })
}

/// Splice each clip's frames into the request in place of its `video_url` part.
///
/// Split out from [`expand_video_parts_to_frames`] because everything above it
/// is I/O (allowlist resolution, ffmpeg, PNG encoding) and everything here is
/// the ordering contract: the lead sentence first, then the frames in
/// chronological order, with the parts that surrounded the clip keeping their
/// positions relative to it. Tests drive this half directly with synthetic
/// bytes so the contract is covered on a host with no ffmpeg.
///
/// `expansions` is `(message index, part index, frame PNG bytes)`. Applied back
/// to front so an earlier splice cannot move a later part's index.
fn apply_video_frame_expansion(
    request: &mut ChatCompletionRequest,
    expansions: Vec<(usize, usize, Vec<Vec<u8>>)>,
) {
    use base64::Engine as _;

    for (message_index, part_index, frames) in expansions.into_iter().rev() {
        let Some(message) = request.messages.get_mut(message_index) else {
            continue;
        };
        let MessageContent::Parts(parts) = &mut message.content else {
            continue;
        };
        if part_index >= parts.len() {
            continue;
        }
        let mut replacement: Vec<ContentPart> = Vec::with_capacity(frames.len() + 1);
        replacement.push(ContentPart::Text {
            text: video_frames_lead_text(frames.len()),
        });
        replacement.extend(frames.into_iter().map(|png| ContentPart::ImageUrl {
            image_url: crate::server::types::request::ImageUrl::new(format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(png)
            )),
        }));
        parts.splice(part_index..=part_index, replacement);
    }
}

/// Legacy wrapper preserved for tests and any callers outside the hot
/// route path. Delegates to [`prepare_chat_request_with_cache`] with
/// the cache-enabled flag set to `false`, matching earlier behavior.
///
/// Production HTTP handlers (see `src/server/routes/chat.rs`) use the
/// `_with_cache` variant directly so they can honor the installed
/// [`super::prompt_cache::PromptCacheStore`].
#[allow(dead_code)]
pub(crate) async fn prepare_chat_request(
    processor: &ChatTemplateProcessor,
    request: &ChatCompletionRequest,
    server_default_kwargs: Option<&ChatTemplateKwargs>,
) -> Result<PreparedChatRequest> {
    prepare_chat_request_with_cache(
        processor,
        request,
        server_default_kwargs,
        false,
        false,
        false,
        &crate::tokenizer::ThinkingMarkers::default(),
    )
    .await
}

/// Full variant of [`prepare_chat_request`] with explicit prompt-cache
/// awareness.
///
/// When `prompt_cache_enabled` is `true` and the caller has not explicitly
/// set `preserve_thinking` anywhere in the request, this function defaults
/// it to `true` before running the rendering pipeline. Explicit overrides
/// (`chat_template_kwargs.preserve_thinking = false`, flattened OpenAI-SDK
/// field, nested `extra_body`, or server-default kwargs) are respected
/// unchanged.
///
/// When `prompt_cache_enabled` is `false` this is identical to the legacy
/// [`prepare_chat_request`].
pub(crate) async fn prepare_chat_request_with_cache(
    processor: &ChatTemplateProcessor,
    request: &ChatCompletionRequest,
    server_default_kwargs: Option<&ChatTemplateKwargs>,
    prompt_cache_enabled: bool,
    snapshot_reuse_capable: bool,
    prefill_assistant: bool,
    thinking_markers: &crate::tokenizer::ThinkingMarkers,
) -> Result<PreparedChatRequest> {
    // Refuse rather than fall back to the generic template when the loaded
    // tokenizer's vocabulary family is Kimi K3 but no native renderer could be
    // attached (#1743 security review). The fallback below would render user
    // text as a plain string and pass it to `MlxcelTokenizer::encode`, which
    // still recognizes the checkpoint's live control-token spellings with
    // special parsing on; see `ChatTemplateProcessor::kimi_k3_family_unrenderable`
    // for why this cannot be gated on `kimi_k3_control_ids()` instead.
    if processor.kimi_k3_family_unrenderable() {
        anyhow::bail!(
            "the loaded tokenizer's Kimi K3 control-token vocabulary is incomplete: refusing \
             to render through the generic chat template, which would let message text \
             re-tokenize as control ids"
        );
    }

    let declared_images = request.image_urls().len();
    let declared_audio = request.audio_inputs().len();
    let declared_videos = request.video_urls().len();
    // No video+audio refusal here: preparation cannot see the loaded model, and
    // the Gemma 4 Unified merge path takes both in one prompt (issue #1349).
    // The capability check runs at the HTTP boundary in
    // `media_capability_rejection` (which reads `ModelMediaSupport`). Every
    // caller of this function reaches a refusal before it: the generating
    // routes and the prompt-inspection routes call the boundary check, and
    // `router_front` refuses any declared media itself because it is text-only.
    // `prepare_request_vlm_embeddings` repeats the refusal on the worker, but
    // it is a backstop rather than the last line of defence for any route: the
    // fetch this function performs happens before a worker ever sees the
    // request, so a guard that only lives there would arrive too late.
    if declared_audio > 0 {
        validate_no_reserved_media_sentinels(request)?;
    }
    // Forced tool choice (#1319): from here on the renderer works on a copy
    // that carries the injected instruction. Shadowing the parameter keeps the
    // primary render, the history-boundary render and the media resolution
    // reading one message list, so `history_prompt` stays a prefix of `prompt`.
    //
    // Kimi K3 renders `tool_choice=required` (and `none`) as a system message
    // of its own, so the generic textual injection would say the same thing
    // twice. A named function has no native K3 form and keeps the injection.
    let request_for_render = if processor.kimi_k3().is_some()
        && request
            .tool_choice
            .as_ref()
            .is_some_and(|choice| choice.is_required())
    {
        std::borrow::Cow::Borrowed(request)
    } else {
        with_tool_choice_instruction(request)
    };
    let request: &ChatCompletionRequest = &request_for_render;
    // Determine effective tools based on tool_choice
    let effective_tools = effective_tools(request);
    let merged_extra_body = request.merged_extra_body();

    // resolve merged kwargs once up-front. See
    // `resolve_effective_kwargs` for the precedence chain; the prompt-cache
    // context builder calls the same helper so the cache key is derived from
    // the same map this render uses.
    let mut merged_kwargs = resolve_effective_kwargs(
        processor,
        request,
        server_default_kwargs,
        &merged_extra_body,
    );

    // default preserve_thinking=true when the prompt cache is on
    // and no layer of the precedence chain set it. The per-request-kwargs
    // object already reflects top-level + flattened-SDK + nested-extra_body
    // + DashScope flat shape, and `merged_kwargs` folds in the server
    // default. So "not set anywhere" is precisely "the merged map lacks the
    // key."
    if prompt_cache_enabled && !merged_kwargs.as_map().contains_key("preserve_thinking") {
        merged_kwargs.set_preserve_thinking(true);
        maybe_log_defaulting_once(request);
    }

    let preserve_thinking = merged_kwargs.preserve_thinking();

    // Whether to produce the extra history-boundary render (issue #1143).
    // Only worth doing when the prompt cache is live, the loaded model can
    // reuse model-owned snapshots, and the request is text-only: a multimodal
    // prompt's token stream is rewritten by placeholder expansion after
    // rendering, so a text-level prefix says nothing about the token-level one.
    // The operator kill switch is checked here as well as in the scheduler, so
    // disabling the feature really does remove this render and the tokenization
    // that follows it, rather than paying for both and discarding the result on
    // the worker thread.
    //
    // b10621 `--prefill-assistant` (#1470). Resolved before the render because
    // it decides BOTH which messages the template sees (the continuation
    // message is dropped) and how the prompt ends (the generation prompt, then
    // the continuation text, with no closing tag).
    let prefill = super::assistant_prefill::resolve(request, prefill_assistant)
        .map_err(|msg| anyhow::anyhow!("{msg}"))?;

    // Kimi K3 renders natively, in code, straight to token ids (#1338). Its
    // prompt has no Jinja template behind it, so everything below this point
    // (the raw-vs-typed render split, the history-boundary render, the
    // rolling-checkpoint `<think>` stripping) does not apply to it.
    if let Some(renderer) = processor.kimi_k3() {
        return prepare_kimi_k3_chat_request(
            processor,
            renderer,
            request,
            kimi_k3_tools(request, effective_tools),
            &merged_kwargs,
            prefill.is_some(),
            declared_images,
            declared_audio,
            declared_videos,
        );
    }

    let render_history_prefix = prompt_cache_enabled
        && snapshot_reuse_capable
        && !super::prompt_cache::boundary_snapshot_disabled()
        && declared_images == 0
        && declared_audio == 0
        && declared_videos == 0
        // Under a prefill the primary prompt is `messages[:-1]` plus the
        // continuation text, so a history render of the full list is not a
        // prefix of it and would produce a snapshot the next turn cannot use.
        && prefill.is_none();

    // Render the prompt, and — when the prompt cache is on — the same
    // conversation as history (`add_generation_prompt = false`). Both renders
    // run over the identical message list and kwargs so the only difference
    // between them is the template's own generation-prompt tail (issue #1143).
    //
    // The history render is skipped when the primary render fell back to
    // `render_simple_fallback`: the fallback is not the template, so its
    // "history" form carries no relationship to what the model was actually
    // prompted with.

    // Typed media parts only belong on the raw-JSON render when the template
    // actually inspects them. A template that never mentions an `image` /
    // `video` / `audio` content type expects `content` to be a plain string,
    // and minijinja renders a list handed to `{{ message['content'] }}` by
    // printing the list itself into the prompt. Those families carry the image
    // through token ids inserted after rendering (LLM-jp-VL's
    // `<|image_start|>` block, InternVL's `<img>` block, ...), so the flattened
    // text render is both what the template expects and what the token-level
    // expansion assumes. Audio requests stay on the raw path regardless,
    // because that is where the ordered `<|audio_i|>` sentinels are emitted.
    let media_parts_need_raw_render = has_template_media_parts(request)
        && (processor.supports_image_content()
            || processor.supports_video_content()
            || processor.supports_audio_content()
            || !request.audio_inputs().is_empty());

    let (prompt, history_render) = if has_tool_fields(request)
        || has_reasoning_fields(request)
        || media_parts_need_raw_render
    {
        // When messages contain tool_calls / tool_call_id, a parallel
        // `reasoning` field (issue #362), or typed media content, use raw JSON
        // rendering so the Jinja2 template can access the complete message
        // shape. The typed `ChatMessage` path only carries role + flattened
        // text, which would otherwise drop image/audio/video positions before
        // processor-aware templates can render their placeholder tokens.
        let mut raw_messages = build_raw_json_messages_with_thinking(request, preserve_thinking);
        // Under a prefill the continuation message is not rendered by the
        // template: it is appended after the generation prompt below.
        if prefill.is_some()
            && let Some(array) = raw_messages.as_array_mut()
        {
            array.pop();
        }
        // Build the stripped ChatMessages in parallel so the fallback path can
        // use them without re-running strip_rolling_checkpoint.
        let stripped = build_chat_messages_with_thinking(request, preserve_thinking);
        match processor.apply_raw_with_kwargs(&raw_messages, effective_tools, &merged_kwargs) {
            Ok(rendered) => {
                let history = render_history_prefix.then(|| {
                    history_boundary_render_attempted();
                    processor.apply_raw_history_with_kwargs(
                        &raw_messages,
                        effective_tools,
                        &merged_kwargs,
                    )
                });
                (rendered, history)
            }
            Err(err) => {
                // A deliberate template refusal is a client error, not a render
                // failure: fail the request instead of answering from a
                // stripped prompt. See `reject_or_return_render_error`.
                let err = reject_or_return_render_error(err)?;
                tracing::warn!(
                    "Chat template render (raw) failed, using fallback: {:#}",
                    err
                );
                // Security (H-1): use pre-stripped messages so that a
                // template-breaking payload cannot bypass rolling-checkpoint
                // stripping and leak prior <think> blocks to the model prompt.
                (render_simple_fallback(&stripped), None)
            }
        }
    } else {
        let mut messages = build_chat_messages_with_thinking(request, preserve_thinking);
        if prefill.is_some() {
            messages.pop();
        }
        match processor.apply_with_kwargs(&messages, effective_tools, &merged_kwargs) {
            Ok(rendered) => {
                let history = render_history_prefix.then(|| {
                    history_boundary_render_attempted();
                    processor.apply_history_with_kwargs(&messages, effective_tools, &merged_kwargs)
                });
                (rendered, history)
            }
            Err(err) => {
                // Same split as the raw/multimodal arm above: a template that
                // refused a caller-supplied value must not degrade to a
                // fallback prompt.
                let err = reject_or_return_render_error(err)?;
                tracing::warn!("Chat template render failed, using fallback: {:#}", err);
                // Security (H-1): use pre-stripped messages for the same reason.
                (render_simple_fallback(&messages), None)
            }
        }
    };

    // Keep the history render only when it really is a text prefix of the
    // prompt. That is the whole invariant the boundary snapshot rests on: the
    // snapshot is model state for `prompt[..history.len()]`, so a history
    // render that is not a prefix would key a snapshot against state it does
    // not describe. Templates that rewrite rather than append (or a render
    // that failed) simply opt out.
    let history_prompt = history_render.and_then(|rendered| match rendered {
        Ok(history) if !history.is_empty() && prompt.starts_with(&history) => Some(history),
        Ok(_) => {
            tracing::debug!(
                "history-boundary render is not a prefix of the generation prompt; \
                 skipping the boundary snapshot for this request"
            );
            None
        }
        Err(err) => {
            tracing::debug!("history-boundary render failed: {:#}", err);
            None
        }
    });

    // Validate the per-request soft-token budget before any image is fetched or
    // decoded: an unsupported `detail` / `max_soft_tokens` is a client error, so
    // there is no reason to spend a download or a decode on it first.
    let image_soft_tokens = request
        .image_soft_tokens()
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    // Audio acquisition is owned by this request-preparation future. Before
    // scheduler admission there is no streaming disconnect token to pass, so
    // handler cancellation drops this future (and its URL stream) as a unit.
    // The audio resolver exposes a separate cooperative-token entry point for
    // the future XLA audio admission path; XLA audio remains capability-false.
    let (image_data, audio_data, videos) = tokio::join!(
        try_extract_chat_image_data(request),
        try_extract_chat_audio_data(request),
        extract_chat_video_paths(request),
    );
    let image_data = image_data?;
    let audio_data = audio_data?;
    let media = MediaRequestMetadata::new(
        declared_images,
        declared_audio,
        declared_videos,
        image_data.len(),
        audio_data.len(),
        videos.len(),
    );
    media
        .validate_resolved_image_count()
        .map_err(anyhow::Error::from)?;

    // Append the continuation text after the template's own generation prompt.
    // A reasoning-only continuation is representable only when that prompt
    // primed an open thinking block, because the open marker is the template's.
    let (prompt, assistant_prefill) = match prefill.as_ref() {
        None => (prompt, None),
        Some(prefill) => {
            let primed =
                super::routes::chat::primed_open_thinking_close_marker(thinking_markers, &prompt);
            let continued =
                super::assistant_prefill::append_to_prompt(&prompt, prefill, primed.as_deref())
                    .map_err(|msg| anyhow::anyhow!("{msg}"))?;
            let echoed = (!prefill.is_reasoning).then(|| prefill.text.clone());
            (continued, echoed)
        }
    };

    Ok(PreparedChatRequest {
        prompt,
        prompt_token_ids: None,
        assistant_prefill,
        history_prompt,
        image_data,
        media,
        image_soft_tokens,
        audio_data,
        videos,
    })
}

/// Prepare a request for Kimi K3's native XTML chat format (#1338).
///
/// The renderer emits token ids directly, so the returned
/// [`PreparedChatRequest`] carries both the ids (authoritative) and their text
/// form (diagnostics, prompt-cache key material, and the primed-thinking
/// check). Four behaviors differ from the template path and are deliberate:
///
/// * **No history-boundary snapshot** (#1143). That optimization rests on the
///   history render being a text prefix of the generation prompt; K3's
///   generation prompt is a structural tag stream, and the id vector is what
///   would have to prefix-match, not the string. Opting out costs a cache
///   entry, never correctness.
/// * **No rolling-checkpoint `<think>` stripping.** K3's history form is
///   structural: in thinking mode every prior assistant turn carries its
///   `think` channel, empty or not. `preserve_thinking` therefore has nothing
///   to strip and is ignored, and a prior turn's `reasoning` renders exactly
///   as the reference renders it.
/// * **No assistant prefill.** b10621's `--prefill-assistant` appends
///   continuation text after the generation prompt; K3's generation prompt
///   ends inside an open XTML tag, so there is no text position to append to
///   without corrupting the structure.
/// * **No media.** Image prompts are #1342; until then a declared image,
///   audio or video input is refused rather than silently dropped.
#[allow(clippy::too_many_arguments)]
fn prepare_kimi_k3_chat_request(
    processor: &ChatTemplateProcessor,
    renderer: &super::kimi_k3_chat::KimiK3Renderer,
    request: &ChatCompletionRequest,
    effective_tools: Option<&[Tool]>,
    merged_kwargs: &ChatTemplateKwargs,
    has_prefill: bool,
    declared_images: usize,
    declared_audio: usize,
    declared_videos: usize,
) -> Result<PreparedChatRequest> {
    if has_prefill {
        anyhow::bail!(
            "Kimi K3 does not support prefilling an assistant message: its generation prompt \
             ends inside an open XTML tag, so there is no text position to continue from"
        );
    }
    if declared_images > 0 || declared_audio > 0 || declared_videos > 0 {
        anyhow::bail!(
            "Kimi K3 chat rendering does not accept image, audio or video inputs yet; \
             image prompts land with the vision path"
        );
    }

    // `thinking` is K3's own kwarg name; `enable_thinking` is mlxcel's
    // cross-family one and is also what `map_reasoning_control_kwargs` fills
    // in from a portable request reasoning control. The reference default is
    // on.
    let thinking = kwarg_bool(merged_kwargs, "thinking")?
        .or(kwarg_bool(merged_kwargs, "enable_thinking")?)
        .unwrap_or(true);
    // Same two-name treatment for the level. An explicit JSON `null` means
    // "omit the thinking-effort message", which is distinct from an absent key
    // (the reference's `kwargs.setdefault("thinking_effort", "max")`).
    let thinking_effort = match kwarg_effort(merged_kwargs, "thinking_effort")? {
        Some(effort) => effort,
        None => match kwarg_effort(merged_kwargs, "reasoning_effort")? {
            // The portable name is clamped onto K3's three levels rather than
            // passed through, so an OpenAI-shaped request asking for the
            // portable default `medium` renders instead of failing. An
            // unrecognized level still reaches the renderer and still errors.
            Some(Some(effort)) => Some(
                super::kimi_k3_chat::clamp_portable_reasoning_effort(&effort)
                    .map(str::to_string)
                    .unwrap_or(effort),
            ),
            Some(None) => None,
            None => Some(super::kimi_k3_chat::DEFAULT_THINKING_EFFORT.to_string()),
        },
    };

    // `required` and `none` are the two modes K3 renders itself. `auto`, a
    // named function, and an absent field render nothing here; a named
    // function is still narrowed by `effective_tools` and still carries the
    // generic textual instruction.
    let tool_choice = request.tool_choice.as_ref().and_then(|choice| {
        matches!(choice.mode(), "required" | "none").then(|| choice.mode().to_string())
    });

    let options = super::kimi_k3_chat::K3RenderOptions {
        add_generation_prompt: processor.add_generation_prompt(),
        thinking,
        thinking_effort,
        tool_choice,
        response_format: request.response_format.as_ref(),
        image_prompts: None,
    };
    let rendered = renderer.render(&request.messages, effective_tools, &options)?;

    Ok(PreparedChatRequest {
        prompt: rendered.text,
        prompt_token_ids: Some(rendered.ids),
        assistant_prefill: None,
        history_prompt: None,
        image_data: Vec::new(),
        media: MediaRequestMetadata::new(0, 0, 0, 0, 0, 0),
        image_soft_tokens: None,
        audio_data: Vec::new(),
        videos: Vec::new(),
    })
}

/// Read a boolean chat-template kwarg, rejecting a value of the wrong type
/// rather than silently falling back to the default.
fn kwarg_bool(kwargs: &ChatTemplateKwargs, key: &str) -> Result<Option<bool>> {
    match kwargs.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Bool(value)) => Ok(Some(*value)),
        Some(other) => {
            anyhow::bail!("chat_template_kwargs.{key} must be a boolean, got {other}")
        }
    }
}

/// Read a thinking-effort kwarg.
///
/// `Ok(None)` means the key is absent (fall through to the next name, then to
/// the default); `Ok(Some(None))` means it was explicitly `null`, which omits
/// the thinking-effort message entirely.
fn kwarg_effort(kwargs: &ChatTemplateKwargs, key: &str) -> Result<Option<Option<String>>> {
    match kwargs.get(key) {
        None => Ok(None),
        Some(serde_json::Value::Null) => Ok(Some(None)),
        Some(serde_json::Value::String(value)) => Ok(Some(Some(value.clone()))),
        Some(other) => {
            anyhow::bail!("chat_template_kwargs.{key} must be a string or null, got {other}")
        }
    }
}

/// Render the history prefix the NEXT turn of this conversation will start
/// with: the request's messages plus the assistant reply that just finished,
/// rendered with `add_generation_prompt = false` (issue #1144).
///
/// This is what the background warm-up prefills toward. It must be produced by
/// re-rendering rather than by appending the generated token ids, because the
/// history form of a reply differs from the generated form in all three ways
/// epic #1148 documented: the generation prompt's scaffold is absent, a
/// `<think>` block may be dropped, and the same text re-tokenizes differently
/// from the sampled sequence.
///
/// `reply` is the assistant `content` as the response carried it. That is what
/// an OpenAI-shaped client echoes back on the following turn, so it is the
/// right guess at the next prompt. A client that sends something else simply
/// misses the warm-up entry and falls back to the #1143 boundary snapshot, so a
/// wrong guess costs one background prefill and never correctness.
///
/// Returns `None` whenever the render is not usable: no reply text, a template
/// that failed, or a result that is not an extension of this turn's own history
/// prefix. That last check is what keeps a warm-up from storing a snapshot
/// under a vector the next turn cannot match.
///
/// Used by: `server::routes::chat` (streaming and non-streaming)
/// Two probe renders of the next turn, differing only in the placeholder user
/// text they append (issue #1144).
///
/// Both are real generation prompts for a hypothetical next turn, so both
/// render the just-finished reply in the position it will actually occupy: an
/// earlier assistant message, not the last one. That distinction is the whole
/// reason two probes exist rather than one render of
/// `messages + reply` with `add_generation_prompt = false`.
///
/// Measured on qwen3.5-0.8b-4bit: the `add_generation_prompt = false` form
/// renders the final assistant message differently from an earlier one, so
/// clipping it against a probe kept only three tokens past the history
/// boundary, making the warm-up worth +3 cached tokens. Taking the common
/// prefix of two probes instead keeps the reply itself, because the only thing
/// that differs between them is the placeholder text at the very end.
pub(crate) struct NextTurnHistory {
    pub(crate) probe_a: String,
    pub(crate) probe_b: String,
}

pub(crate) fn render_next_turn_history(
    processor: &ChatTemplateProcessor,
    request: &ChatCompletionRequest,
    server_default_kwargs: Option<&ChatTemplateKwargs>,
    reply: &str,
) -> Option<NextTurnHistory> {
    if reply.trim().is_empty() {
        tracing::debug!("warmup: empty reply");
        return None;
    }
    // Multimodal turns are excluded for the same reason the boundary render is:
    // placeholder expansion rewrites the token stream after rendering.
    if !request.image_urls().is_empty()
        || !request.audio_inputs().is_empty()
        || !request.video_urls().is_empty()
    {
        tracing::debug!("warmup: multimodal");
        return None;
    }
    // Same forced-tool-choice injection as `prepare_chat_request_with_cache`
    // (#1319), so the warmed prefix matches what a next turn carrying the same
    // `tool_choice` will render.
    let request_for_render = with_tool_choice_instruction(request);
    let request: &ChatCompletionRequest = &request_for_render;

    // Resolve the kwargs through the same helper the render pipeline and the
    // prompt-cache context builder use, so a mapped top-level
    // `reasoning_effort` (issue #1164) is present in the probe renders too.
    // Deriving the merge separately here would warm a vector rendered with a
    // different effort than the bucket's `template_sig` describes, and the next
    // turn would miss it.
    let mut merged_kwargs = resolve_effective_kwargs(
        processor,
        request,
        server_default_kwargs,
        &request.merged_extra_body(),
    );
    // Mirror the prompt-cache defaulting that `prepare_chat_request_with_cache`
    // applied to the request this reply answered, or the two renders would
    // disagree about thinking retention and the warm-up would key a vector the
    // next turn never produces.
    if !merged_kwargs.as_map().contains_key("preserve_thinking") {
        merged_kwargs.set_preserve_thinking(true);
    }
    let preserve_thinking = merged_kwargs.preserve_thinking();
    let effective_tools = effective_tools(request);

    // This turn's own history prefix, for the extension check below.
    let this_turn = {
        let messages = build_chat_messages_with_thinking(request, preserve_thinking);
        match processor.apply_history_with_kwargs(&messages, effective_tools, &merged_kwargs) {
            Ok(v) => v,
            Err(err) => {
                tracing::debug!("warmup: this-turn history render failed: {err:#}");
                return None;
            }
        }
    };

    let mut with_reply = request.clone();
    with_reply.messages.push(Message {
        role: Role::Assistant,
        content: MessageContent::Text(reply.to_string()),
        name: None,
        tool_call_id: None,
        reasoning: None,
        tool_calls: None,
    });
    // Render two probe turns. Both put the reply where the next turn will put
    // it (an earlier assistant message), and they differ only in the trailing
    // placeholder user text, so their common prefix is exactly the part of the
    // next turn's prompt that does not depend on what the user says next.
    //
    // Rendering `messages + reply` with `add_generation_prompt = false` and
    // clipping that against a single probe does NOT work: templates routinely
    // render the final assistant message differently from an earlier one, so
    // the two disagree immediately after the assistant header and the reply
    // itself is clipped away. Measured on qwen3.5-0.8b-4bit, that construction
    // was worth +3 cached tokens over the #1143 boundary alone.
    let render_probe = |text: &str| {
        let mut probed = with_reply.clone();
        probed.messages.push(Message {
            role: Role::User,
            content: MessageContent::Text(text.to_string()),
            name: None,
            tool_call_id: None,
            reasoning: None,
            tool_calls: None,
        });
        let msgs = build_chat_messages_with_thinking(&probed, preserve_thinking);
        processor.apply_with_kwargs(&msgs, effective_tools, &merged_kwargs)
    };
    let (Ok(probe_a), Ok(probe_b)) = (
        render_probe(NEXT_TURN_PROBE_A),
        render_probe(NEXT_TURN_PROBE_B),
    ) else {
        tracing::debug!("warmup: probe render failed");
        return None;
    };
    // Both probes must still start with this turn's history, or the template is
    // rewriting rather than appending and nothing can be warmed safely.
    if !probe_a.starts_with(&this_turn) {
        tracing::debug!("warmup: probe does not extend this turn's history");
        return None;
    }

    Some(NextTurnHistory { probe_a, probe_b })
}

/// Placeholder user texts for the two next-turn probe renders.
///
/// Their content is irrelevant and only their *difference* is load-bearing:
/// the clip keeps the head the two agree on, which ends where the placeholder
/// text begins. They are deliberately ordinary and unequal in both wording and
/// length so no template branch keyed on content or size can make them agree
/// further than they should.
/// Length of the head two next-turn probe renders agree on (issue #1144).
///
/// The probes differ only in the placeholder user text they append, so their
/// common prefix ends exactly where the next turn's own words begin. Everything
/// up to that point is what any next turn reproduces: the whole conversation
/// including the just-finished reply, rendered in the position it will occupy.
///
/// This clip is load-bearing, not defensive. A warm-up snapshot supersedes the
/// history-boundary snapshot it chains from, so storing a vector the next turn
/// cannot match does not merely waste a background prefill, it destroys a
/// working hit. Measured on qwen3.5-0.8b-4bit with an unclipped target,
/// `cached_tokens` went from 150 to 0.
///
/// Returns `None` when nothing survives, which means the template's rendering
/// depends on the next turn's content from the very start and this conversation
/// cannot be warmed.
///
/// Used by: `server::routes::chat::submit_next_turn_warmup`
pub(crate) fn clip_warmup_target(probe_a: &[i32], probe_b: &[i32]) -> Option<usize> {
    let keep = probe_a
        .iter()
        .zip(probe_b)
        .take_while(|(a, b)| a == b)
        .count();
    (keep > 0).then_some(keep)
}

const NEXT_TURN_PROBE_A: &str = "ok";
const NEXT_TURN_PROBE_B: &str = "could you expand on that a little further please";

fn validate_no_reserved_media_sentinels(request: &ChatCompletionRequest) -> Result<()> {
    for message in &request.messages {
        match &message.content {
            MessageContent::Text(text)
                if text.contains(super::types::request::ORDERED_MEDIA_PREFIX) =>
            {
                anyhow::bail!("message text contains a reserved ordered-media sentinel");
            }
            MessageContent::Parts(parts)
                if parts.iter().any(|part| {
                    matches!(
                        part,
                        ContentPart::Text { text }
                            if text.contains(super::types::request::ORDERED_MEDIA_PREFIX)
                    )
                }) =>
            {
                anyhow::bail!("message text contains a reserved ordered-media sentinel");
            }
            _ => {}
        }
    }
    Ok(())
}

/// Emit an INFO log exactly once per resolved session when the
/// prompt-cache-driven `preserve_thinking=true` default kicks in.
///
/// The dedup key is the same `session_key` the cache uses, so distinct
/// users (OpenAI-standard `user` field or `prompt_cache_key`) each get
/// their own one-shot log line; anonymous traffic shares the log entry
/// for [`super::prompt_cache::key::ANONYMOUS_SESSION_SENTINEL`].
///
/// The log is purely informational; the defaulting decision itself runs
/// regardless.
fn maybe_log_defaulting_once(request: &ChatCompletionRequest) {
    let pck = request.resolve_prompt_cache_key();
    let user = request.resolve_user();
    let session = resolve_session_key(pck, user).to_string();
    let Ok(mut set) = log_once_sessions().lock() else {
        return;
    };
    if set.insert(session.clone()) {
        tracing::info!(
            session = %session,
            "prompt cache on + preserve_thinking unset: defaulting preserve_thinking=true \
             for prefix stability (override via chat_template_kwargs.preserve_thinking=false)"
        );
    }
}

/// Determine the effective tools slice to pass to the template.
///
/// Returns `None` when tool_choice is "none" or no tools are provided.
///
/// This is load-bearing beyond template rendering: since issue #967 it is also
/// part of the Gemma 4 loop-detection activation signal, read through
/// [`crate::server::request_options::chat_carries_loop_amplifier`]. The gate's
/// premise is that tool declarations amplify the repetition collapse only when
/// the model actually sees them, so the gate and the template deliberately read
/// the same helper. A change here moves both, which is the intent: they must not
/// drift apart.
///
/// Note the precise claim: this reports what is *handed to* the template, not
/// what the template does with it. A checkpoint whose chat template ignores its
/// `tools` argument renders no declarations, and when `apply_raw_with_kwargs` /
/// `apply_with_kwargs` fails, `render_simple_fallback` drops tools entirely and
/// emits a plain-chat prompt. In both cases the gate reports amplified for a
/// prompt that is not. That is accepted as a conservative over-approximation:
/// erring toward keeping issue #432's protection on is the safe direction, and
/// the alternative would mean resolving the gate after rendering.
pub(crate) fn effective_tools(request: &ChatCompletionRequest) -> Option<&[Tool]> {
    let tools = request.tools.as_deref();
    match request.tool_choice.as_ref() {
        // "none": do not pass tools to the template.
        Some(tc) if tc.is_none() => None,
        // Named function (#1319): the template sees only that tool, so the
        // model cannot pick another one. A name that is not declared renders
        // no tools at all; the routes reject that shape before rendering
        // through `ToolChoice::validate`, so this branch is defensive.
        Some(ToolChoice::Specific(choice)) => tools
            .and_then(|tools| {
                tools
                    .iter()
                    .find(|tool| tool.function.name == choice.function.name)
            })
            .map(std::slice::from_ref),
        _ => tools,
    }
}

/// The tools Kimi K3 declares, which differ from [`effective_tools`] in the
/// `tool_choice: "none"` case alone (#1338).
///
/// `effective_tools` hides the tools from a Jinja template under `none`,
/// because for a template the tools block *is* the offer: showing it and then
/// saying "do not call these" is a contradiction the template cannot express.
/// K3 can. It renders the refusal as its own system message, and the reference
/// emits that message after the declaration, so the model is told exactly
/// which tools it must not call. The `xtml/tool_choice_none.json` fixture
/// carries both blocks in that order.
///
/// Every other mode keeps the shared narrowing, the named-function case
/// included, so this only ever widens `none` back to the declared list.
///
/// Used by: prepare_chat_request_with_cache
fn kimi_k3_tools<'a>(
    request: &'a ChatCompletionRequest,
    effective: Option<&'a [Tool]>,
) -> Option<&'a [Tool]> {
    match request.tool_choice.as_ref() {
        Some(choice) if choice.is_none() => request.tools.as_deref(),
        _ => effective,
    }
}

/// The instruction a forced `tool_choice` adds to the prompt (#1319), or `None`
/// for `auto` / `none` / absent.
///
/// The wording is fixed so the rendered prompt, and with it the prompt-cache
/// key, is a pure function of the request. It is injected by
/// [`inject_tool_choice_instruction`] on the message list handed to the
/// renderer, never on the request itself.
pub(crate) fn tool_choice_instruction(choice: &ToolChoice) -> Option<String> {
    match choice {
        ToolChoice::Mode(mode) if mode == "required" => Some(
            "You must call one or more of the available functions to answer the user's \
             request. Do not answer directly without calling a function."
                .to_string(),
        ),
        ToolChoice::Specific(choice) => Some(format!(
            "You must call the '{}' function to answer the user's request. Do not call \
             any other function and do not answer directly.",
            choice.function.name
        )),
        _ => None,
    }
}

/// Place a forced-tool-choice instruction into `messages` (#1319).
///
/// Placement, in order: appended to the first `system` message as
/// `"\n\n" + instruction`; otherwise appended to the last `user` message
/// (plain text gets the same suffix, a content-parts message gets a trailing
/// text part); otherwise inserted as a new leading `system` message. The
/// message list is the renderer's copy, so the original request stays
/// untouched for response echoing.
pub(crate) fn inject_tool_choice_instruction(messages: &mut Vec<Message>, instruction: &str) {
    let suffix = format!("\n\n{instruction}");
    if let Some(system) = messages.iter_mut().find(|m| m.role == Role::System) {
        append_text(&mut system.content, &suffix);
        return;
    }
    if let Some(user) = messages.iter_mut().rev().find(|m| m.role == Role::User) {
        append_text(&mut user.content, &suffix);
        return;
    }
    messages.insert(
        0,
        Message {
            role: Role::System,
            content: MessageContent::Text(instruction.to_string()),
            name: None,
            tool_call_id: None,
            reasoning: None,
            tool_calls: None,
        },
    );
}

/// Append `suffix` to a message body without disturbing its media parts.
fn append_text(content: &mut MessageContent, suffix: &str) {
    match content {
        MessageContent::Text(text) => text.push_str(suffix),
        MessageContent::Parts(parts) => parts.push(ContentPart::Text {
            text: suffix.to_string(),
        }),
    }
}

/// The request as the renderer should see it: unchanged unless `tool_choice`
/// is forced, in which case a clone carries the injected instruction (#1319).
///
/// Every render of a request goes through this, including the history-boundary
/// render and the next-turn warm-up probes, so the prompt-cache prefix those
/// derive is taken from the prompt the model was actually shown.
pub(crate) fn with_tool_choice_instruction(
    request: &ChatCompletionRequest,
) -> std::borrow::Cow<'_, ChatCompletionRequest> {
    let Some(instruction) = request
        .tool_choice
        .as_ref()
        .and_then(tool_choice_instruction)
    else {
        return std::borrow::Cow::Borrowed(request);
    };
    let mut injected = request.clone();
    inject_tool_choice_instruction(&mut injected.messages, &instruction);
    std::borrow::Cow::Owned(injected)
}

/// Check if any message in the request has tool-related fields that
/// require raw JSON rendering (tool_calls, tool_call_id).
///
/// Also the second half of the tools signal for the issue #967 loop-detection
/// gate, via [`crate::server::request_options::chat_carries_loop_amplifier`].
/// The raw-JSON path writes `tool_calls` and `tool_call_id` into the rendered
/// prompt independently of [`effective_tools`], so an agent loop replaying prior
/// tool calls produces a thoroughly tool-shaped prompt even when the follow-up
/// turn sends no top-level `tools` array (or sends `tool_choice: "none"`). Those
/// turns are amplified and must keep detection on.
pub(crate) fn has_tool_fields(request: &ChatCompletionRequest) -> bool {
    request
        .messages
        .iter()
        .any(|m| m.tool_call_id.is_some() || m.tool_calls.is_some())
}

/// Check if any message carries a non-empty `reasoning` field (issue #362).
///
/// Such requests must take the raw-JSON render path so the parallel reasoning
/// reaches templates that read `message.get('reasoning')`; the typed
/// [`ChatMessage`] path drops everything except role and content.
fn has_reasoning_fields(request: &ChatCompletionRequest) -> bool {
    request
        .messages
        .iter()
        .any(|m| m.reasoning.as_ref().is_some_and(|r| !r.is_empty()))
}

/// Returns `true` when the request carries at least one message with
/// user-meaningful ("effective") input, per issue #773.
///
/// A message counts as effective input when any of the following holds:
///
/// * its flattened text content (string form, or a content-list `text`
///   part) is non-empty after trimming whitespace
/// * a content-list `image_url` part carries a non-empty `url`
/// * a content-list `video_url` part carries a non-empty `url` (mlxcel's
///   media-input surface extends the OpenAI shape with video; treated the
///   same as an image for this check)
/// * a content-list `input_audio` part carries non-empty `data`
/// * the message carries a non-empty `tool_calls` array
/// * the message carries a non-empty `reasoning` field
///
/// An empty `messages` array (or a request where every message fails all
/// of the above) has no effective input and should be rejected with a 400
/// before any model dispatch — see [`super::routes::chat::chat_completions`]
/// and [`super::routes::responses::create_response`].
pub(crate) fn request_has_effective_input(request: &ChatCompletionRequest) -> bool {
    request.messages.iter().any(message_has_effective_input)
}

/// Single-message half of [`request_has_effective_input`].
fn message_has_effective_input(message: &Message) -> bool {
    if message.content.has_effective_text() {
        return true;
    }

    if let MessageContent::Parts(parts) = &message.content {
        for part in parts {
            match part {
                // Text parts are already covered by
                // `content.has_effective_text()` above.
                ContentPart::Text { .. } => {}
                ContentPart::ImageUrl { image_url } => {
                    if !image_url.url.trim().is_empty() {
                        return true;
                    }
                }
                ContentPart::VideoUrl { video_url } => {
                    if !video_url.url.trim().is_empty() {
                        return true;
                    }
                }
                ContentPart::InputAudio { input_audio } => {
                    if !input_audio.data.trim().is_empty() {
                        return true;
                    }
                }
            }
        }
    }

    if message.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty()) {
        return true;
    }

    if message
        .reasoning
        .as_ref()
        .is_some_and(|r| !r.trim().is_empty())
    {
        return true;
    }

    false
}

/// Build raw JSON messages for template rendering, preserving all fields
/// (including tool_calls, tool_call_id) so Jinja2 templates can iterate over
/// multi-turn tool-use conversations.
///
/// Thin wrapper with `preserve_thinking=true` — used by tests that predate
/// and by any caller that does not want rolling-checkpoint
/// stripping.
#[cfg(test)]
pub(super) fn build_raw_json_messages(request: &ChatCompletionRequest) -> serde_json::Value {
    build_raw_json_messages_with_thinking(request, true)
}

/// build raw JSON messages with optional rolling-checkpoint
/// stripping of `<think>` blocks.
///
/// When `preserve_thinking` is `true`, all `<think>...</think>` blocks reach
/// the template unchanged (Qwen3.6 multi-turn retention). When `false` (the
/// default), the rolling-checkpoint rule strips thinking from every assistant
/// message **before** the most recent non-tool-call user turn — matching the
/// Qwen3/Qwen3.5 convention. The most recent assistant reply keeps its
/// reasoning regardless.
///
/// This Rust-side stripping is the fallback for templates that don't
/// understand the `preserve_thinking` kwarg. Templates that do understand it
/// (like the official Qwen3.6 chat template) will still see the stripped
/// strings; because the stripped text contains no `<think>` markers, the
/// template's own preserve-logic is a no-op there — we reach the same
/// effective prompt either way.
fn build_raw_json_messages_with_thinking(
    request: &ChatCompletionRequest,
    preserve_thinking: bool,
) -> serde_json::Value {
    let preserve_media_order = !request.audio_inputs().is_empty();
    let mut ordered_media = preserve_media_order.then(OrderedMediaOrdinals::default);
    // Decide which assistant messages (by index) need their think blocks
    // stripped. Empty set means "keep everything."
    let strip_indices: std::collections::HashSet<usize> = if preserve_thinking {
        std::collections::HashSet::new()
    } else {
        strip_rolling_checkpoint(&request.messages, |m| m.role.as_str(), |m| m.content.text())
            .into_iter()
            .collect()
    };

    let messages: Vec<serde_json::Value> = request
        .messages
        .iter()
        .enumerate()
        .map(|(idx, m)| {
            // Strip think blocks from assistant messages before the checkpoint.
            let stripped = strip_indices.contains(&idx);
            let normalized_content = template_content(&m.content, ordered_media.as_mut());
            let raw_content = template_text_content(&normalized_content);
            let content = if stripped {
                serde_json::Value::String(strip_think_block(&raw_content).into_owned())
            } else {
                normalized_content
            };

            let mut msg = serde_json::json!({
                "role": m.role.as_str(),
                "content": content,
            });

            if let Some(ref name) = m.name {
                msg["name"] = serde_json::Value::String(name.clone());
            }
            if let Some(ref tool_call_id) = m.tool_call_id {
                msg["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
            }
            if let Some(ref tool_calls) = m.tool_calls {
                let mut tc_value =
                    serde_json::to_value(tool_calls).unwrap_or(serde_json::Value::Null);
                normalize_tool_call_arguments(&mut tc_value);
                msg["tool_calls"] = tc_value;
            }

            // Forward the parallel `reasoning` field (issue #362) so templates
            // that render `message.get('reasoning')` (e.g. Gemma 4) see prior
            // assistant thinking across turns. The decision mirrors the inline
            // `<think>` handling exactly so the two channels stay consistent:
            //
            // - When this message is being stripped (preserve_thinking=false and
            //   it sits before the rolling checkpoint), drop the reasoning field
            //   too. Stripping the inline block while leaking the parallel field
            //   would still feed prior thinking back into the prompt.
            // - When preserve_thinking=true (or this is the retained latest
            //   reply), forward the reasoning field, unless the content already
            //   carries an inline `<think>` block. Forwarding it on top of an
            //   inline block would double-inject the same reasoning into
            //   templates that render both channels.
            if !stripped
                && let Some(reasoning) = m.reasoning.as_ref()
                && !reasoning.is_empty()
                && !raw_content.contains("<think>")
            {
                msg["reasoning"] = serde_json::Value::String(reasoning.clone());
            }

            msg
        })
        .collect();

    serde_json::Value::Array(messages)
}

/// Return whether any message carries a media part whose position must reach
/// the processor-aware chat template.
///
/// OpenAI's wire types (`image_url`, `video_url`, `input_audio`) are transport
/// objects, while Hugging Face processor templates branch on semantic types
/// (`image`, `video`, `audio`). Routing only these requests through the raw
/// renderer keeps the established string-content path unchanged for ordinary
/// text requests and text-only content arrays.
fn has_template_media_parts(request: &ChatCompletionRequest) -> bool {
    request.messages.iter().any(|message| {
        matches!(
            &message.content,
            MessageContent::Parts(parts)
                if parts.iter().any(|part| !matches!(part, ContentPart::Text { .. }))
        )
    })
}

/// Normalize OpenAI transport content into the typed content shape consumed by
/// Hugging Face processor templates.
///
/// Media bytes/URLs are resolved by the server's media pipeline, not by Jinja.
/// Exposing them to a template would both leak unnecessarily large payloads
/// into rendering and fail templates such as LLaVA's, which select
/// `content.type == "image"`. Keep only an ordered semantic marker at this
/// boundary.
#[derive(Default)]
struct OrderedMediaOrdinals {
    image: usize,
    audio: usize,
}

fn template_content(
    content: &MessageContent,
    ordered_media: Option<&mut OrderedMediaOrdinals>,
) -> serde_json::Value {
    // Audio-capable families consume private image/audio ordinals after chat
    // rendering. Combined audio/video requests are rejected before this
    // helper, so the ordered representation deliberately has no video
    // sentinel.
    if let Some(ordinals) = ordered_media {
        return match content {
            MessageContent::Text(text) => serde_json::Value::String(text.clone()),
            MessageContent::Parts(parts) => {
                let mut flattened = String::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => flattened.push_str(text),
                        ContentPart::ImageUrl { .. } => {
                            ordinals.image += 1;
                            flattened.push_str(&ordered_image_sentinel(ordinals.image));
                        }
                        ContentPart::InputAudio { .. } => {
                            ordinals.audio += 1;
                            flattened.push_str(&ordered_audio_sentinel(ordinals.audio));
                        }
                        ContentPart::VideoUrl { .. } => {}
                    }
                }
                serde_json::Value::String(flattened)
            }
        };
    }

    match content {
        MessageContent::Text(text) => serde_json::Value::String(text.clone()),
        MessageContent::Parts(parts) => serde_json::Value::Array(
            parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => serde_json::json!({
                        "type": "text",
                        "text": text,
                    }),
                    ContentPart::ImageUrl { .. } => serde_json::json!({
                        "type": "image",
                    }),
                    ContentPart::VideoUrl { .. } => serde_json::json!({
                        "type": "video",
                    }),
                    ContentPart::InputAudio { .. } => serde_json::json!({
                        "type": "audio",
                    }),
                })
                .collect(),
        ),
    }
}

fn template_text_content(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
            .collect(),
        _ => String::new(),
    }
}

/// Normalize each tool call's `function.arguments` from a JSON-encoded string
/// into a parsed object so dict-iterating chat templates can consume it.
///
/// The OpenAI wire format carries `arguments` as a JSON *string* (e.g.
/// `"{\"path\":\"/foo\"}"`), and that's what agentic clients echo back on
/// later turns. But the Qwen3-Coder / Qwen3.5 / Qwen3.6 chat templates iterate
/// it with `tool_call.arguments|items`, which requires a mapping. Passing the
/// raw string makes minijinja's `|items` fail ("cannot convert value into
/// pairs"), the whole render falls back to a default template, and the
/// resulting prompt diverges from the prior turn from the first token,
/// silently degrading the model's multi-turn context and destroying
/// prompt-cache prefix reuse (every tool turn re-prefills cold).
///
/// We replace `arguments` only when the string parses to a JSON **object**;
/// templates that serialize arguments via `tojson` then emit the original
/// object shape, while malformed/scalar arguments stay strings for templates
/// that treat them as text. Mirrors mlx-serve's `chat.zig` workaround.
///
/// Two zero-argument spellings get the same treatment even though neither
/// parses to an object: an empty (or whitespace-only) string, the common
/// spelling agentic clients echo back for a call that takes no parameters,
/// and the literal string `"null"`, the same intent from a client that
/// `JSON.stringify()`s a `null` arguments value. Both are unambiguous "there
/// were no arguments" signals, unlike `"[1,2]"` or a truncated payload, which
/// stay strings because there is no safe reading of them as "no arguments".
/// Before this, a template macro that requires `arguments` to be a mapping
/// (e.g. Onyx ATEM's `render_atem`, see
/// `tests/fixtures/muse_glimmer/chat_template.jinja`) raised on either
/// spelling. That turned a routine zero-argument tool call into an HTTP 400
/// the caller could not act on, and the failure was sticky: the offending
/// `tool_calls` entry lives on in conversation history and replays on every
/// later turn.
fn normalize_tool_call_arguments(tool_calls: &mut serde_json::Value) {
    let serde_json::Value::Array(calls) = tool_calls else {
        return;
    };
    for call in calls {
        let Some(args) = call.pointer_mut("/function/arguments") else {
            continue;
        };
        let serde_json::Value::String(s) = args else {
            continue;
        };
        let trimmed = s.trim();
        if trimmed.is_empty() || trimmed == "null" {
            *args = serde_json::json!({});
            continue;
        }
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s)
            && parsed.is_object()
        {
            *args = parsed;
        }
    }
}

/// Flatten request messages into [`ChatMessage`], preserving all `<think>`
/// blocks.
///
/// Thin wrapper around [`build_chat_messages_with_thinking`] with
/// `preserve_thinking=true`. Only exercised by tests today — the production
/// code path in [`prepare_chat_request`] always calls
/// `build_chat_messages_with_thinking` directly so it can honor the merged
/// kwargs.
#[cfg(test)]
pub(super) fn build_chat_messages(request: &ChatCompletionRequest) -> Vec<ChatMessage> {
    build_chat_messages_with_thinking(request, true)
}

/// Security (H-1): produce the same "System: … User: … Assistant: …" fallback
/// prompt that `ChatCompletionRequest::to_prompt()` emits, but operating on
/// messages that have **already been stripped** by either
/// [`build_chat_messages_with_thinking`] or equivalent pre-processing.
///
/// This is the single fallback renderer used by both the raw-JSON path and the
/// typed-message path when Jinja template rendering fails (parse error, `raise`
/// in template, minijinja internal error).  Centralising the fallback here
/// ensures the `preserve_thinking` stripping decision made before the Jinja
/// call is never bypassed by a deliberately template-breaking request payload.
fn render_simple_fallback(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => prompt.push_str(&format!("System: {}\n\n", msg.content)),
            "user" => prompt.push_str(&format!("User: {}\n\n", msg.content)),
            "assistant" => prompt.push_str(&format!("Assistant: {}\n\n", msg.content)),
            "tool" => prompt.push_str(&format!("Tool: {}\n\n", msg.content)),
            other => prompt.push_str(&format!("{}: {}\n\n", other, msg.content)),
        }
    }
    prompt.push_str("Assistant: ");
    prompt
}

/// flatten request messages into [`ChatMessage`] with optional
/// rolling-checkpoint stripping.
///
/// See [`build_raw_json_messages_with_thinking`] for the stripping rules. The
/// `ChatMessage` path is used for the common non-tool-call case; the typed
/// struct doesn't carry `tool_calls`/`tool_call_id`, which is fine because
/// `has_tool_fields` routes those cases to the raw-JSON path.
fn build_chat_messages_with_thinking(
    request: &ChatCompletionRequest,
    preserve_thinking: bool,
) -> Vec<ChatMessage> {
    let preserve_media_order = !request.audio_inputs().is_empty();
    let mut ordered_media = preserve_media_order.then(OrderedMediaOrdinals::default);
    let strip_indices: std::collections::HashSet<usize> = if preserve_thinking {
        std::collections::HashSet::new()
    } else {
        strip_rolling_checkpoint(&request.messages, |m| m.role.as_str(), |m| m.content.text())
            .into_iter()
            .collect()
    };

    request
        .messages
        .iter()
        .enumerate()
        .map(|(idx, message)| {
            let normalized_content = template_content(&message.content, ordered_media.as_mut());
            let raw = template_text_content(&normalized_content);
            let content = if strip_indices.contains(&idx) {
                strip_think_block(&raw).into_owned()
            } else {
                raw
            };
            ChatMessage {
                role: message.role.as_str().to_string(),
                content,
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "chat_request_tests.rs"]
mod tests;
