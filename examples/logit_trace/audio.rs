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

//! Opt-in audio-prompted arm for Gemma 4 (#2379).
//!
//! `MLXCEL_TRACE_AUDIO=CLIP.wav` together with `MLXCEL_TRACE_AUDIO_PROMPT=FILE`
//! traces a teacher-forced transcript behind an audio prompt, so a change to
//! the Conformer audio tower is judged by what it does to the decoder's
//! logits. `FILE` holds the rendered chat-template prompt, with the
//! `<|audio|>` placeholder where the clip goes and the model's turn opened at
//! the end; the corpus argument is the transcript that follows it.
//!
//! The clip goes through the server's own path: decode, resample to 16 kHz,
//! the duration-based placeholder count, the checkpoint's mel extractor, the
//! audio tower and the embedder. Prompt and transcript then run as one
//! single-pass prefill, the width a real audio request runs at, and only the
//! transcript rows are traced.
use anyhow::{Context, Result, bail, ensure};
use mlxcel::{LanguageModel, LoadedModel, audio, tokenizer::MlxcelTokenizer};
use mlxcel_core::{MlxArray, UniquePtr};
use std::path::PathBuf;

pub struct AudioArm {
    clip: PathBuf,
    prompt: PathBuf,
}

impl AudioArm {
    pub fn from_env() -> Result<Option<Self>> {
        let Ok(clip) = std::env::var("MLXCEL_TRACE_AUDIO") else {
            return Ok(None);
        };
        let prompt = std::env::var("MLXCEL_TRACE_AUDIO_PROMPT")
            .context("MLXCEL_TRACE_AUDIO needs MLXCEL_TRACE_AUDIO_PROMPT (the rendered prompt)")?;
        Ok(Some(Self {
            clip: clip.into(),
            prompt: prompt.into(),
        }))
    }

    /// Token ids of the expanded prompt, the transcript, and the logits of one
    /// forward over both (`[1, prompt + transcript - 1, vocab]`).
    pub fn forward(
        &self,
        model: &LoadedModel,
        tokenizer: &MlxcelTokenizer,
        transcript: &str,
    ) -> Result<(usize, Vec<i32>, UniquePtr<MlxArray>)> {
        let LoadedModel::Gemma4VLM(vl) = model else {
            bail!("MLXCEL_TRACE_AUDIO supports Gemma 4 checkpoints with an audio tower");
        };
        ensure!(
            vl.audio_tower.is_some(),
            "this checkpoint has no audio tower"
        );

        let (samples, rate) = audio::load_wav_file(&self.clip)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", self.clip.display()))?;
        let num_audio_tokens = audio::compute_audio_num_tokens(samples.len(), rate, 40, 750);
        let samples = if rate == 16_000 {
            samples
        } else {
            audio::whisper_mel::resample_to_16k(&samples, rate)
        };

        let prompt_text = std::fs::read_to_string(&self.prompt)
            .with_context(|| format!("reading {}", self.prompt.display()))?;
        let mut prompt: Vec<i32> = encode(tokenizer, &prompt_text)?;
        ensure!(
            prompt.contains(&vl.audio_token_id),
            "the prompt must carry the <|audio|> placeholder (id {})",
            vl.audio_token_id
        );
        mlxcel::vlm_runtime::expand_gemma4_audio_tokens_for_server(
            &mut prompt,
            vl.audio_token_id,
            vl.boa_token_id,
            vl.eoa_token_id,
            num_audio_tokens,
            None,
        );
        let target = encode(tokenizer, transcript)?;
        ensure!(target.len() >= 2, "transcript too short");

        let extractor = audio::AudioFeatureExtractor::new(vl.audio_extractor_config.clone());
        let (features, mask) = extractor.extract(&samples, None);
        let frames = mask.len() as i32;
        let features =
            mlxcel_core::from_slice_f32(&features, &[1, frames, extractor.feature_size() as i32]);
        let mask: Vec<i32> = mask.iter().map(|&valid| i32::from(valid)).collect();
        let mask = mlxcel_core::astype(
            &mlxcel_core::from_slice_i32(&mask, &[1, frames]),
            mlxcel_core::dtype::BOOL,
        );

        // Row `prompt.len() - 1` predicts the transcript's first token, so the
        // last transcript token is the only one not fed.
        let mut ids = prompt.clone();
        ids.extend_from_slice(&target[..target.len() - 1]);
        let input = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);
        let embeddings = vl
            .get_input_embeddings_with_audio(&input, &[], Some(&features), Some(&mask))
            .map_err(|e| anyhow::anyhow!("audio embeddings: {e}"))?;

        model.reset_runtime_state();
        let mut caches = model.make_caches();
        let logits = model.forward_with_embeddings(
            &input,
            Some(&embeddings.inputs_embeds),
            &mut caches,
            None,
        );
        eprintln!(
            "  audio arm: {num_audio_tokens} audio tokens, prompt {} tokens, transcript {} tokens",
            prompt.len(),
            target.len()
        );
        Ok((prompt.len() - 1, target, logits))
    }
}

fn encode(tokenizer: &MlxcelTokenizer, text: &str) -> Result<Vec<i32>> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .into_iter()
        .map(|t| t as i32)
        .collect())
}
