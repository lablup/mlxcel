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

//! Cache-aware online VoiceChat session (issue #1378).
//!
//! Port of `VoiceChatStreamingSession` in upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_voicechat/streaming.py.
//! PCM chunks of any size are buffered into 1280-sample (80 ms) frames; each
//! frame advances every network exactly once from persistent state (the
//! streaming log-mel, the FastConformer attention/conv/subsampling caches,
//! the RNNT prediction state, the Nemotron-H caches, the EAR-TTS KV caches
//! and the codec causal-conv / iSTFT overlap caches) and yields events tagged
//! with the frame index.
//!
//! MLX evaluation is thread-affine, so a session must be created and driven
//! on one thread (the realtime server gives each model a dedicated worker).
//!
//! Used by: `mlxcel generate --stream`, the `/v1/realtime` engine

use std::time::Instant;

use mlxcel_core::{MlxArray, UniquePtr};

use super::buffer::FrameBuffer;
use super::events::{StreamingOptions, TokenAccumulator, VoiceChatError, VoiceChatEvent};
use super::perception::StreamingPerception;
use super::profile::{FrameTiming, VoiceChatProfile};
use crate::audio::nemotron_codec::CausalConv1dCache;
use crate::audio::rnnt::RnntStreamState;
use crate::models::nemotron_h::NemotronLayerCache;
use crate::models::nemotron_voicechat::model::NemotronVoiceChatModel;
use crate::models::nemotron_voicechat::session::host_i32;
use crate::models::nemotron_voicechat::tts::TtsCaches;

type Result<T> = std::result::Result<T, VoiceChatError>;

/// Language-model state: persistent caches, or the fused-input history
/// recomputed every step (diagnostic mode).
enum LanguageState {
    Cached(Vec<NemotronLayerCache>),
    History(Vec<UniquePtr<MlxArray>>),
}

struct StepTiming {
    language_ms: f64,
    tts_ms: f64,
    codec_ms: f64,
}

/// One stateful, full-duplex VoiceChat timeline.
pub struct VoiceChatStreamingSession<'m> {
    model: &'m NemotronVoiceChatModel,
    frame_samples: usize,
    max_frames: Option<u64>,
    buffer: FrameBuffer,
    perception: StreamingPerception<'m>,
    rnnt: RnntStreamState,
    language: LanguageState,
    tts_caches: TtsCaches,
    previous_code: UniquePtr<MlxArray>,
    codec_cache: CausalConv1dCache,
    text: TokenAccumulator,
    function: TokenAccumulator,
    last_text: i32,
    last_function: i32,
    /// Greedy ids of every audio frame (the prompt prefix is excluded).
    text_ids: Vec<i32>,
    function_ids: Vec<i32>,
    timeline_index: u64,
    frame_index: u64,
    closed: bool,
    profiling: bool,
    profile: VoiceChatProfile,
}

fn ms_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

impl NemotronVoiceChatModel {
    /// Create an independent cache-aware online session. Warms EAR-TTS with
    /// the `Aria` prompt and prefills the system prompt.
    pub fn create_streaming_session(
        &self,
        options: StreamingOptions,
    ) -> Result<VoiceChatStreamingSession<'_>> {
        VoiceChatStreamingSession::new(self, options)
    }
}

impl<'m> VoiceChatStreamingSession<'m> {
    fn new(model: &'m NemotronVoiceChatModel, options: StreamingOptions) -> Result<Self> {
        let frame_duration = model.config.frame_duration;
        let max_frames = match options.max_streaming_seconds {
            Some(s) if !(s.is_finite() && s > 0.0) => {
                return Err(VoiceChatError::InvalidInput(
                    "max_streaming_seconds must be positive".to_string(),
                ));
            }
            Some(s) => Some((s / frame_duration) as u64),
            None => None,
        };
        let frame_samples = model.config.frame_samples();
        let perception =
            StreamingPerception::new(model, frame_samples, options.use_perception_cache)?;
        let language = if options.use_language_cache {
            LanguageState::Cached(model.lm.make_caches())
        } else {
            LanguageState::History(Vec::new())
        };
        let special = model.config.special_text_ids();
        mlxcel_core::random_seed(options.seed);
        let (tts_caches, previous_code) = model.warm_tts()?;
        let pad = model.config.pad_token_id;
        let mut session = Self {
            model,
            frame_samples,
            max_frames,
            buffer: FrameBuffer::new(frame_samples),
            perception,
            rnnt: RnntStreamState::new(&model.rnnt),
            language,
            tts_caches,
            previous_code,
            codec_cache: CausalConv1dCache::new(),
            text: TokenAccumulator::new(&special),
            function: TokenAccumulator::new(&special),
            last_text: pad,
            last_function: pad,
            text_ids: Vec::new(),
            function_ids: Vec::new(),
            timeline_index: 0,
            frame_index: 0,
            closed: false,
            profiling: options.profile,
            profile: VoiceChatProfile::new(f64::from(frame_duration) * 1000.0),
        };
        session.prefill_prompt(options.system_prompt.as_deref())?;
        Ok(session)
    }

    /// Audio frames processed so far.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Timeline positions (prompt prefix plus audio frames) so far.
    pub fn timeline_index(&self) -> u64 {
        self.timeline_index
    }

    /// Whether `flush` or `cancel` closed the session.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Samples per frame (1280 at 16 kHz).
    pub fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    /// Greedy text and function ids of every audio frame so far, including
    /// the special ids the text events skip.
    pub fn channel_ids(&self) -> (&[i32], &[i32]) {
        (&self.text_ids, &self.function_ids)
    }

    /// Collected per-frame timings (empty unless `profile` was set).
    pub fn profile(&self) -> &VoiceChatProfile {
        &self.profile
    }

    fn prefill_prompt(&mut self, prompt: Option<&str>) -> Result<()> {
        let ids = self.model.system_prompt_ids(prompt)?;
        if ids.is_empty() {
            return Ok(());
        }
        let embeds = self.model.lm.embed(&ids);
        for index in 0..ids.len() {
            let row = crate::models::nemotron_voicechat::session::frame_row(&embeds, index as i32);
            self.timeline_step(&row, false, false)?;
        }
        Ok(())
    }

    fn language_step(
        &mut self,
        fused: UniquePtr<MlxArray>,
    ) -> crate::models::nemotron_voicechat::llm::DuplexTokens {
        match &mut self.language {
            LanguageState::Cached(caches) => self.model.lm.step(&fused, caches),
            LanguageState::History(history) => {
                history.push(fused);
                let mut all = mlxcel_core::copy(&history[0]);
                for item in &history[1..] {
                    all = mlxcel_core::concatenate(&all, item, 1);
                }
                let mut caches = self.model.lm.make_caches();
                self.model.lm.step(&all, &mut caches)
            }
        }
    }

    /// One timeline position: language model, EAR-TTS, and (for audio
    /// frames) the text events and one codec step.
    fn timeline_step(
        &mut self,
        audio_embedding: &MlxArray,
        generate_channels: bool,
        decode_audio: bool,
    ) -> Result<(Vec<VoiceChatEvent>, StepTiming)> {
        let pad = self.model.config.pad_token_id;
        let fused = self
            .model
            .lm
            .fused_input(self.last_text, audio_embedding, self.last_function);
        let start = Instant::now();
        let out = self.language_step(fused);
        let (text_id, function_id) = if generate_channels {
            (out.text, out.function)
        } else {
            (pad, pad)
        };
        let language_ms = ms_since(start);
        self.last_text = text_id;
        self.last_function = function_id;

        let start = Instant::now();
        let code = if self.timeline_index == 0 {
            self.model.assets.silence_frame()
        } else {
            if text_id == self.model.config.eos_token_id {
                self.previous_code = self.model.assets.silence_frame();
            }
            let step = self
                .model
                .tts
                .step(&self.previous_code, text_id, &mut self.tts_caches)?;
            self.previous_code = step.codes;
            mlxcel_core::copy(&self.previous_code)
        };
        self.timeline_index += 1;
        mlxcel_core::eval(&code);
        let tts_ms = ms_since(start);
        if !generate_channels {
            return Ok((
                Vec::new(),
                StepTiming {
                    language_ms,
                    tts_ms,
                    codec_ms: 0.0,
                },
            ));
        }

        self.text_ids.push(text_id);
        self.function_ids.push(function_id);
        let mut events = Vec::new();
        let frame_index = self.frame_index;
        let tokenizer = &self.model.tokenizer;
        let decode = |ids: &[i32]| -> std::result::Result<String, String> {
            let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
            tokenizer.decode(&ids, false).map_err(|e| e.to_string())
        };
        if let Some((delta, text)) = self.text.append(text_id, decode)? {
            events.push(VoiceChatEvent::AssistantTextDelta {
                frame_index,
                token_id: text_id,
                delta,
                text,
            });
        }
        if let Some((delta, text)) = self.function.append(function_id, decode)? {
            events.push(VoiceChatEvent::FunctionDelta {
                frame_index,
                token_id: function_id,
                delta,
                text,
            });
        }

        let mut codec_ms = 0.0;
        if decode_audio {
            let start = Instant::now();
            let clean = self.model.assets.replace_control_codes(&code);
            let codes_t = mlxcel_core::transpose_axes(&clean, &[0, 2, 1]);
            let samples = self
                .model
                .codec
                .decode_step(&codes_t, &mut self.codec_cache, false)?;
            let samples = crate::models::nemotron_voicechat::session::host_f32(&samples);
            codec_ms = ms_since(start);
            let expected = self.model.codec.waveform_to_token_ratio();
            if samples.len() != expected {
                return Err(VoiceChatError::Inference(format!(
                    "streaming codec returned {} samples, expected {expected}",
                    samples.len()
                )));
            }
            events.push(VoiceChatEvent::Audio {
                frame_index,
                samples,
                sample_rate: self.model.config.output_sample_rate,
                audio_codes: host_i32(&clean),
            });
        }
        Ok((
            events,
            StepTiming {
                language_ms,
                tts_ms,
                codec_ms,
            },
        ))
    }

    fn step_audio_frame(&mut self, frame: &[f32]) -> Result<Vec<VoiceChatEvent>> {
        if let Some(max_frames) = self.max_frames
            && self.frame_index >= max_frames
        {
            return Err(VoiceChatError::ContextLimit { max_frames });
        }
        let frame_start = Instant::now();
        let perception = self.perception.step(frame)?;
        let perception_ms = ms_since(frame_start);

        let mut events = Vec::new();
        let start = Instant::now();
        let update = self.rnnt.step(
            &self.model.rnnt,
            &perception.encoded,
            self.model.max_symbols,
            &self.model.rnnt_vocabulary,
        )?;
        let rnnt_ms = ms_since(start);
        if let Some((delta, text)) = update {
            events.push(VoiceChatEvent::UserTranscriptDelta {
                frame_index: self.frame_index,
                delta,
                text,
            });
        }
        let (timeline_events, timing) = self.timeline_step(&perception.projected, true, true)?;
        events.extend(timeline_events);
        if self.profiling {
            self.profile.frames.push(FrameTiming {
                frame_index: self.frame_index,
                perception_ms,
                rnnt_ms,
                language_ms: timing.language_ms,
                tts_ms: timing.tts_ms,
                codec_ms: timing.codec_ms,
                total_ms: ms_since(frame_start),
            });
        }
        self.frame_index += 1;
        Ok(events)
    }

    /// Buffer mono 16 kHz PCM and run one timeline step per complete
    /// 1280-sample frame, returning the events of those frames.
    pub fn push_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<VoiceChatEvent>> {
        if self.closed {
            return Err(VoiceChatError::Closed);
        }
        if sample_rate != self.model.config.input_sample_rate {
            return Err(VoiceChatError::InvalidInput(format!(
                "expected {} Hz PCM, received {sample_rate} Hz",
                self.model.config.input_sample_rate
            )));
        }
        if samples.iter().any(|x| !x.is_finite()) {
            return Err(VoiceChatError::InvalidInput(
                "audio contains a non-finite sample".to_string(),
            ));
        }
        let mut events = Vec::new();
        for frame in self.buffer.push(samples) {
            events.extend(self.step_audio_frame(&frame)?);
        }
        Ok(events)
    }

    /// Finish the input: optionally zero-pad and run the partial frame,
    /// clear the codec overlap state, close, and emit `Done`.
    pub fn flush(&mut self, pad_partial: bool) -> Result<Vec<VoiceChatEvent>> {
        if self.closed {
            return Ok(Vec::new());
        }
        let mut events = Vec::new();
        if let Some(frame) = self.buffer.take_partial(pad_partial) {
            events.extend(self.step_audio_frame(&frame)?);
        }
        self.closed = true;
        self.codec_cache.clear();
        events.push(VoiceChatEvent::Done {
            frame_index: self.frame_index,
        });
        Ok(events)
    }

    /// Stop without finishing the input; emits `Cancelled`.
    pub fn cancel(&mut self) -> Vec<VoiceChatEvent> {
        if self.closed {
            return Vec::new();
        }
        self.closed = true;
        self.buffer.clear();
        self.codec_cache.clear();
        vec![VoiceChatEvent::Cancelled {
            frame_index: self.frame_index,
        }]
    }
}
