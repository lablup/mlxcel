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

//! Per-frame perception of the streaming VoiceChat session.
//!
//! Port of `VoiceChatStreamingSession._perception_step` in upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_voicechat/streaming.py.
//! The cached path runs the streaming log-mel and the cache-aware
//! FastConformer so each 1280-sample frame costs one encoder frame; the
//! uncached fallback recomputes log-mel and the encoder over a sliding
//! sample window (the reference's diagnostic mode).
//!
//! Used by: [`super::VoiceChatStreamingSession`]

use mlxcel_core::{MlxArray, UniquePtr};

use crate::audio::fastconformer::ConformerStreamingState;
use crate::audio::nemotron_mel::{StreamingLogMel, log_mel_array};
use crate::models::nemotron_voicechat::model::NemotronVoiceChatModel;

/// Projected (`[1, 1, 4480]`) and raw encoder (`[1, 1, 1024]`) output of
/// one audio frame.
pub(crate) struct PerceptionFrame {
    pub projected: UniquePtr<MlxArray>,
    pub encoded: UniquePtr<MlxArray>,
}

pub(crate) enum StreamingPerception<'m> {
    Cached {
        model: &'m NemotronVoiceChatModel,
        mel: StreamingLogMel,
        conformer: ConformerStreamingState<'m>,
    },
    Window {
        model: &'m NemotronVoiceChatModel,
        window: Vec<f32>,
        max_samples: usize,
    },
}

impl<'m> StreamingPerception<'m> {
    pub(crate) fn new(
        model: &'m NemotronVoiceChatModel,
        frame_samples: usize,
        cached: bool,
    ) -> Result<Self, String> {
        let context = model.perception.encoder().args().default_att_context();
        if cached {
            Ok(Self::Cached {
                model,
                mel: StreamingLogMel::new(&model.mel_args, Some(frame_samples))?,
                conformer: ConformerStreamingState::new(model.perception.encoder(), 1, context)?,
            })
        } else {
            let frames = (context[0] + context[1] + 1).max(2) as usize;
            Ok(Self::Window {
                model,
                window: Vec::new(),
                max_samples: frames * frame_samples,
            })
        }
    }

    /// Advance by one frame of 16 kHz samples.
    pub(crate) fn step(&mut self, frame: &[f32]) -> Result<PerceptionFrame, String> {
        match self {
            Self::Cached {
                model,
                mel,
                conformer,
            } => {
                let mel_frames = mel.push(frame)?;
                let mut chunks = conformer.push(&mel_frames, false, true)?;
                let shape = chunks.first().map(|c| mlxcel_core::array_shape(c));
                if chunks.len() != 1 || shape.as_ref().is_none_or(|s| s.len() != 3 || s[1] != 1) {
                    return Err(format!(
                        "cached perception did not emit exactly one encoder frame: mel {:?}, \
                         encoded {:?}",
                        mlxcel_core::array_shape(&mel_frames),
                        chunks
                            .iter()
                            .map(|c| mlxcel_core::array_shape(c))
                            .collect::<Vec<_>>()
                    ));
                }
                let encoded = chunks.remove(0);
                let projected = model.perception.project(&encoded);
                conformer.materialize(&[&projected, &encoded]);
                Ok(PerceptionFrame { projected, encoded })
            }
            Self::Window {
                model,
                window,
                max_samples,
            } => {
                window.extend_from_slice(frame);
                if window.len() > *max_samples {
                    let excess = window.len() - *max_samples;
                    window.drain(..excess);
                }
                let mel = log_mel_array(window, &model.mel_args)?;
                let frames = mlxcel_core::array_shape(&mel)[1] as usize;
                let out = model.perception.forward(&mel, frames)?;
                let t = mlxcel_core::array_shape(&out.projected)[1];
                if t < 2 {
                    return Err(
                        "perception encoder returned fewer than two frames for an 80 ms input"
                            .to_string(),
                    );
                }
                // The last embedding contains the preprocessor's right-edge
                // padding; the reference takes the second-to-last one.
                let pick = |a: &MlxArray| {
                    let s = mlxcel_core::array_shape(a);
                    mlxcel_core::slice(a, &[0, t - 2, 0], &[1, t - 1, s[2]])
                };
                let projected = pick(&out.projected);
                let encoded = pick(&out.encoded);
                mlxcel_core::eval(&projected);
                mlxcel_core::eval(&encoded);
                Ok(PerceptionFrame { projected, encoded })
            }
        }
    }
}
