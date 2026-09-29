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
    if generation.stream {
        return run_streaming(&model, args, &samples, system_prompt, seed);
    }
    ensure!(
        generation.max_streaming_seconds.is_none(),
        "--max-streaming-seconds applies to --stream runs only"
    );
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

/// Number of cold frames the `--profile` summary drops (graph compilation
/// and cache allocation land on the first frames).
const PROFILE_COLD_FRAMES: usize = 5;

/// `MLXCEL_VOICECHAT_PROFILE_STAGES=1` adds sub-stage attribution to
/// `--stream --profile` (extra forced evaluations; attribution only).
const PROFILE_STAGES_ENV: &str = "MLXCEL_VOICECHAT_PROFILE_STAGES";

/// `MLXCEL_VOICECHAT_PROFILE_FRAMES=<path>` also writes every frame's
/// timings (cold frames included) as a JSON array, for steady-state
/// analysis of long inputs.
const PROFILE_FRAMES_ENV: &str = "MLXCEL_VOICECHAT_PROFILE_FRAMES";

fn profile_stages_requested() -> bool {
    std::env::var(PROFILE_STAGES_ENV).is_ok_and(|v| !v.is_empty() && v != "0")
}

/// `--stream`: push the input (plus `--extra-decoding-seconds` of silence)
/// through the cache-aware online session in 80 ms frames, printing deltas
/// as frames produce them, then flush.
fn run_streaming(
    model: &mlxcel::models::NemotronVoiceChatModel,
    args: &GenerateArgs,
    samples: &[f32],
    system_prompt: Option<&str>,
    seed: u64,
) -> Result<()> {
    use mlxcel::models::nemotron_voicechat::{StreamingOptions, VoiceChatEvent};
    use std::io::Write;

    let generation = &args.generation;
    let options = StreamingOptions {
        system_prompt: Some(system_prompt.unwrap_or_default().to_string()),
        seed,
        max_streaming_seconds: generation.max_streaming_seconds,
        use_language_cache: true,
        use_perception_cache: true,
        profile: generation.profile,
        profile_stages: generation.profile && profile_stages_requested(),
    };
    let mut session = model.create_streaming_session(options)?;
    let rate = model.config().input_sample_rate;
    let mut input = samples.to_vec();
    input.resize(
        input.len() + (generation.extra_decoding_seconds * rate as f32).round() as usize,
        0.0,
    );

    let mut audio: Vec<f32> = Vec::new();
    let mut sample_rate = model.config().output_sample_rate;
    let mut transcript = String::new();
    let mut answer = String::new();
    let mut function = String::new();
    let mut handle = |events: Vec<VoiceChatEvent>| {
        for event in events {
            match event {
                VoiceChatEvent::AssistantTextDelta { delta, text, .. } => {
                    print!("{delta}");
                    let _ = std::io::stdout().flush();
                    answer = text;
                }
                VoiceChatEvent::FunctionDelta { text, .. } => function = text,
                VoiceChatEvent::UserTranscriptDelta { text, .. } => transcript = text,
                VoiceChatEvent::Audio {
                    samples,
                    sample_rate: rate,
                    ..
                } => {
                    sample_rate = rate;
                    audio.extend(samples);
                }
                VoiceChatEvent::Done { .. } | VoiceChatEvent::Cancelled { .. } => {}
            }
        }
    };

    let started = std::time::Instant::now();
    let frame = session.frame_samples();
    for chunk in input.chunks(frame) {
        handle(session.push_audio(chunk, rate)?);
    }
    handle(session.flush(true)?);
    let elapsed = started.elapsed().as_secs_f64();
    println!();
    println!("[user] {transcript}");
    if !function.is_empty() {
        println!("[function] {function}");
    }
    if answer.is_empty() {
        eprintln!("(no assistant text)");
    }

    if let Some(out) = generation.output_audio.as_ref() {
        let wav = mlxcel::audio::wav_writer::encode_wav_pcm16(&audio, sample_rate, 1);
        std::fs::write(out, wav).with_context(|| format!("failed to write {}", out.display()))?;
        eprintln!(
            "Wrote {} ({} samples, {:.2}s at {sample_rate} Hz)",
            out.display(),
            audio.len(),
            audio.len() as f64 / f64::from(sample_rate)
        );
    }
    eprintln!(
        "{} audio frames streamed in {elapsed:.2}s",
        session.frame_index()
    );
    if generation.profile {
        let summary = session.profile().summary(PROFILE_COLD_FRAMES);
        eprintln!("{}", serde_json::to_string_pretty(&summary)?);
        if let Some(path) = std::env::var_os(PROFILE_FRAMES_ENV) {
            let frames = serde_json::to_string(&session.profile().frames)?;
            std::fs::write(&path, frames)
                .with_context(|| format!("failed to write {}", path.to_string_lossy()))?;
        }
    }
    Ok(())
}
