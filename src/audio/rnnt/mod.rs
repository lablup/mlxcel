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

//! RNN-T prediction and joint networks with greedy decoding.
//!
//! Ports `PredictNetwork` / `JointNetwork` from
//! `mlx_audio/stt/models/nemotron_asr/rnnt.py` and the greedy loop of
//! `VoiceChatSession._rnnt_decode` in
//! `mlx_vlm/models/nemotron_voicechat/session.py`. Per encoder frame, up to
//! `max_symbols` symbols are emitted: the predictor runs on the last emitted
//! token (a zero vector while it is still blank), the joint scores
//! `relu(enc(frame) + pred(prediction))`, and the argmax either is blank (next
//! frame) or becomes the new last token, in which case the predictor state
//! advances. The state only changes on a non-blank symbol.
//!
//! The per-frame work is [`RnntDecoder::step_frame`] over an explicit
//! [`RnntState`], so a streaming session can keep the state across chunks.
//!
//! Used by: NemotronLabs VoiceChat user transcript.

mod lstm;
pub mod vocab;

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

pub use lstm::{LstmState, StackedLstm};
pub use vocab::{
    decode_pieces, is_lang_tag, is_special_piece, is_special_token, load_vocabulary_json,
};

/// RNN-T prediction network settings (`audio_config.decoder`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct PredictArgs {
    pub pred_hidden: usize,
    pub pred_rnn_layers: usize,
    pub vocab_size: usize,
    pub blank_as_pad: bool,
}

impl Default for PredictArgs {
    fn default() -> Self {
        Self {
            pred_hidden: 640,
            pred_rnn_layers: 2,
            vocab_size: 1024,
            blank_as_pad: true,
        }
    }
}

/// RNN-T joint network settings (`audio_config.joint`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct JointArgs {
    pub joint_hidden: usize,
    pub activation: String,
    pub encoder_hidden: usize,
    pub pred_hidden: usize,
    pub num_classes: usize,
}

impl Default for JointArgs {
    fn default() -> Self {
        Self {
            joint_hidden: 640,
            activation: "relu".to_string(),
            encoder_hidden: 1024,
            pred_hidden: 640,
            num_classes: 1024,
        }
    }
}

/// Predictor output for the current `(last_token, hidden)`, reused across
/// frames until a non-blank symbol changes the state.
struct CachedPrediction {
    /// `joint.pred(prediction)`, `[1, 1, joint_hidden]`.
    pred_proj: UniquePtr<MlxArray>,
    proposed: LstmState,
}

/// Greedy decoding state carried across frames (and across streaming chunks).
pub struct RnntState {
    last_token: i32,
    hidden: Option<LstmState>,
    cached: Option<CachedPrediction>,
}

impl RnntState {
    pub fn new(blank_id: i32) -> Self {
        Self {
            last_token: blank_id,
            hidden: None,
            cached: None,
        }
    }

    /// Last emitted non-blank token, or blank before the first emission.
    pub fn last_token(&self) -> i32 {
        self.last_token
    }
}

#[derive(Clone, Copy)]
enum Activation {
    Relu,
    Sigmoid,
    Tanh,
}

pub struct RnntDecoder {
    embed: UniquePtr<MlxArray>,
    lstm: StackedLstm,
    enc: UnifiedLinear,
    pred: UnifiedLinear,
    out: UnifiedLinear,
    activation: Activation,
    pred_hidden: i32,
    blank_id: i32,
}

impl RnntDecoder {
    /// Load `{decoder_prefix}.prediction.{embed, dec_rnn.lstm}` and
    /// `{joint_prefix}.{enc, pred, joint_net.2}` (e.g.
    /// `stt_model.rnnt_decoder` / `stt_model.rnnt_joint`).
    pub fn from_weights(
        weights: &WeightMap,
        decoder_prefix: &str,
        joint_prefix: &str,
        predict: &PredictArgs,
        joint: &JointArgs,
    ) -> Result<Self, String> {
        let activation = match joint.activation.to_ascii_lowercase().as_str() {
            "relu" => Activation::Relu,
            "sigmoid" => Activation::Sigmoid,
            "tanh" => Activation::Tanh,
            other => return Err(format!("unsupported RNNT joint activation {other:?}")),
        };
        let embed_key = format!("{decoder_prefix}.prediction.embed.weight");
        let embed = weights
            .get(&embed_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("RNNT weight not found: {embed_key}"))?;
        let rows = predict.vocab_size + usize::from(predict.blank_as_pad);
        let embed_shape = mlxcel_core::array_shape(&embed);
        if embed_shape != [rows as i32, predict.pred_hidden as i32] {
            return Err(format!(
                "{embed_key} has shape {embed_shape:?}, expected [{rows}, {}]",
                predict.pred_hidden
            ));
        }
        let lstm = StackedLstm::from_weights(
            weights,
            &format!("{decoder_prefix}.prediction.dec_rnn.lstm"),
            predict.pred_hidden,
            predict.pred_hidden,
            predict.pred_rnn_layers,
        )?;
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{joint_prefix}.{name}"), 64, 4)
        };
        Ok(Self {
            embed,
            lstm,
            enc: linear("enc")?,
            pred: linear("pred")?,
            out: linear("joint_net.2")?,
            activation,
            pred_hidden: predict.pred_hidden as i32,
            // Blank is the last joint class (`num_classes`), which is also the
            // embedding padding row when `blank_as_pad` is set.
            blank_id: joint.num_classes as i32,
        })
    }

    pub fn blank_id(&self) -> i32 {
        self.blank_id
    }

    pub fn initial_state(&self) -> RnntState {
        RnntState::new(self.blank_id)
    }

    /// Run the predictor for `state` (or reuse the cached result).
    fn prediction<'a>(&self, state: &'a mut RnntState, dtype: i32) -> &'a CachedPrediction {
        state.cached.get_or_insert_with(|| {
            let input = if state.last_token != self.blank_id {
                let ids = mlxcel_core::from_slice_i32(&[state.last_token], &[1]);
                mlxcel_core::take(&self.embed, &ids, 0)
            } else {
                mlxcel_core::zeros(&[1, self.pred_hidden], mlxcel_core::dtype::FLOAT32)
            };
            let (out, proposed) = self.lstm.step(&input, state.hidden.as_ref());
            let out = mlxcel_core::astype(&out, dtype);
            let out = mlxcel_core::reshape(&out, &[1, 1, self.pred_hidden]);
            CachedPrediction {
                pred_proj: self.pred.forward(&out),
                proposed,
            }
        })
    }

    /// Greedy symbols for one encoder frame `enc_frame: [1, 1, encoder_hidden]`.
    ///
    /// Returns every non-blank token emitted for this frame (special pieces
    /// included; they count toward `max_symbols` like in the reference).
    pub fn step_frame(
        &self,
        enc_frame: &MlxArray,
        state: &mut RnntState,
        max_symbols: usize,
    ) -> Result<Vec<i32>, String> {
        let dtype = mlxcel_core::array_dtype(enc_frame);
        let enc_proj = self.enc.forward(enc_frame);
        let max_symbols = max_symbols.max(1);
        let mut emitted = Vec::new();
        while emitted.len() < max_symbols {
            let cached = self.prediction(state, dtype);
            let joint = mlxcel_core::add(&enc_proj, &cached.pred_proj);
            let joint = match self.activation {
                Activation::Relu => mlxcel_core::relu(&joint),
                Activation::Sigmoid => mlxcel_core::sigmoid(&joint),
                Activation::Tanh => mlxcel_core::tanh(&joint),
            };
            let logits = mlxcel_core::reshape(&self.out.forward(&joint), &[-1]);
            let token_arr = mlxcel_core::argmax(&logits, 0, false);
            mlxcel_core::try_eval(&token_arr).map_err(|e| format!("RNNT joint failed: {e}"))?;
            let token = mlxcel_core::item_i32(&token_arr);
            if token == self.blank_id {
                break;
            }
            let Some(cached) = state.cached.take() else {
                return Err("RNNT predictor cache missing".to_string());
            };
            state.hidden = Some(
                cached
                    .proposed
                    .into_iter()
                    .map(|(h, c)| {
                        (
                            mlxcel_core::astype(&h, dtype),
                            mlxcel_core::astype(&c, dtype),
                        )
                    })
                    .collect(),
            );
            state.last_token = token;
            emitted.push(token);
        }
        Ok(emitted)
    }

    /// Greedy decode `encoded: [1, T, encoder_hidden]` over its first `length`
    /// frames. Returns every non-blank token (special pieces included).
    pub fn greedy_decode(
        &self,
        encoded: &MlxArray,
        length: usize,
        max_symbols: usize,
    ) -> Result<Vec<i32>, String> {
        let shape = mlxcel_core::array_shape(encoded);
        if shape.len() != 3 || shape[0] != 1 {
            return Err(format!(
                "RNNT expects encoder output [1, T, D], got {shape:?}"
            ));
        }
        let frames = length.min(shape[1].max(0) as usize) as i32;
        let mut state = self.initial_state();
        let mut tokens = Vec::new();
        for t in 0..frames {
            let frame = mlxcel_core::slice(encoded, &[0, t, 0], &[1, t + 1, shape[2]]);
            tokens.extend(self.step_frame(&frame, &mut state, max_symbols)?);
        }
        Ok(tokens)
    }

    /// Greedy transcript: decoded pieces with specials dropped, trimmed.
    pub fn transcribe(
        &self,
        encoded: &MlxArray,
        length: usize,
        max_symbols: usize,
        vocabulary: &[String],
    ) -> Result<String, String> {
        let tokens = self.greedy_decode(encoded, length, max_symbols)?;
        Ok(decode_pieces(&tokens, vocabulary).trim().to_string())
    }
}

#[cfg(test)]
mod tests;
