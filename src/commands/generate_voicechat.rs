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

//! `mlxcel generate` driver for Nemotron VoiceChat (issue #1374).
//!
//! VoiceChat is a full-duplex speech model: its output length is the input
//! timeline length, not a token budget, and `-p` is the system prompt rather
//! than a user turn. `run_generate_once` routes the checkpoint here before
//! any chat-template, tokenizer, or decode-loop setup, right after `-m` is
//! resolved to a local directory.

use std::path::Path;

use anyhow::{Context, Result, ensure};

use crate::GenerateArgs;

/// True when `model_dir` holds a Nemotron VoiceChat checkpoint.
pub(crate) fn is_voicechat_checkpoint(model_dir: &Path) -> bool {
    matches!(
        mlxcel::models::get_model_type(model_dir),
        Ok(mlxcel::models::ModelType::NemotronVoiceChat)
    )
}

/// Run one offline VoiceChat turn: read `--audio`, run the duplex timeline,
/// print the transcript and answer, and write `--output-audio`.
pub(crate) fn run_voicechat_generation(args: &GenerateArgs) -> Result<()> {
    let generation = &args.generation;
    ensure!(
        generation.image.is_empty() && generation.video.is_empty(),
        "Nemotron VoiceChat takes speech only; --image and --video are not supported"
    );
    let audio_path = generation
        .audio
        .as_ref()
        .context("Nemotron VoiceChat requires --audio <input.wav> (16 kHz mono speech)")?;
    ensure!(
        generation.extra_decoding_seconds.is_finite()
            && (0.0..=mlxcel::models::nemotron_voicechat::session::MAX_EXTRA_DECODING_SECONDS)
                .contains(&generation.extra_decoding_seconds),
        "--extra-decoding-seconds must be between 0 and {} seconds",
        mlxcel::models::nemotron_voicechat::session::MAX_EXTRA_DECODING_SECONDS
    );
    // `-p` is the system prompt as given; an empty `-p` (or none, which the
    // router maps to an empty prompt) means no system prompt.
    let system_prompt = generation.prompt.as_deref();

    let (samples, rate) =
        mlxcel::audio::feature_extractor::load_wav_file(audio_path).map_err(anyhow::Error::msg)?;
    let samples = mlxcel::audio::whisper_mel::resample_to_16k(&samples, rate);

    let model_dir = Path::new(&args.model.model);
    let load_started = std::time::Instant::now();
    let model = mlxcel::models::NemotronVoiceChatModel::load(model_dir).with_context(|| {
        format!(
            "failed to load Nemotron VoiceChat from {}",
            model_dir.display()
        )
    })?;
    eprintln!(
        "Loaded Nemotron VoiceChat in {:.1}s",
        load_started.elapsed().as_secs_f64()
    );

    let seed = args.sampling.seed.unwrap_or(0);
    let run_started = std::time::Instant::now();
    let result = model
        .generate_offline(
            &samples,
            system_prompt,
            generation.extra_decoding_seconds,
            seed,
        )
        .map_err(anyhow::Error::msg)?;
    let elapsed = run_started.elapsed().as_secs_f64();

    if let Some(transcript) = result.user_transcript.as_deref() {
        println!("[user] {transcript}");
    }
    println!("{}", result.text);
    if !result.function_text.is_empty() {
        println!("[function] {}", result.function_text);
    }

    if let Some(out) = generation.output_audio.as_ref() {
        let wav = mlxcel::audio::wav_writer::encode_wav_pcm16(&result.audio, result.sample_rate, 1);
        std::fs::write(out, wav).with_context(|| format!("failed to write {}", out.display()))?;
        eprintln!(
            "Wrote {} ({} samples, {:.2}s at {} Hz)",
            out.display(),
            result.audio.len(),
            result.audio.len() as f64 / f64::from(result.sample_rate),
            result.sample_rate
        );
    }
    eprintln!(
        "{} timeline frames in {elapsed:.2}s",
        result.text_tokens.len()
    );
    Ok(())
}
