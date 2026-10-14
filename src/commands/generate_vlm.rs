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

//! Media helpers left in the CLI after its multimodal dispatch moved onto the
//! server's preparation (`mlxcel::server::local_media`, issue #2173): image
//! opening for the diffusion path and Qwen3-Omni speech synthesis.

use std::path::Path;

use anyhow::Result;

use mlxcel::LoadedModel;

/// Open an image file for a CLI command, naming the path in the error.
///
/// Used by `generate_diffusion::prepare_diffusion_vision`.
pub(crate) fn open_image(path: &Path) -> Result<image::DynamicImage> {
    image::open(path).map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
}

/// `--output-audio` (issue #665): synthesize speech for a completed
/// Qwen3-Omni generation and write it as a WAV file.
///
/// Runs AFTER text generation: loads the talker + code2wav stack lazily (the
/// default model load drops those weights, so text-only use pays nothing),
/// re-conditions on [prompt + generated] token embeddings, generates codec
/// frames, vocodes, and writes 24 kHz mono PCM16.
pub(crate) fn run_speech_synthesis(
    model: &LoadedModel,
    args: &crate::GenerateArgs,
    wav_path: &Path,
    prompt_tokens: &[i32],
    generated_tokens: &[i32],
) -> Result<()> {
    let LoadedModel::Qwen3OmniMoe(omni) = model else {
        anyhow::bail!(
            "--output-audio is only supported for Qwen3-Omni models (this model has no \
             talker/code2wav speech stack)"
        );
    };

    println!("Loading Qwen3-Omni speech stack (talker + code2wav)...");
    let load_start = std::time::Instant::now();
    let speech = mlxcel::load_qwen3_omni_speech(Path::new(&args.model.model))?;
    println!(
        "Speech stack loaded in {:.2}s. Synthesizing speech (speaker: {})...",
        load_start.elapsed().as_secs_f64(),
        args.generation.speaker
    );

    let synth_start = std::time::Instant::now();
    let output = speech
        .synthesize(
            omni,
            prompt_tokens,
            generated_tokens,
            &args.generation.speaker,
            None,
        )
        .map_err(|e| anyhow::anyhow!("Speech synthesis failed: {e}"))?;

    let wav = mlxcel::audio::encode_wav_pcm16(&output.samples, output.sample_rate, 1);
    std::fs::write(wav_path, wav)
        .map_err(|e| anyhow::anyhow!("Failed to write WAV file {}: {e}", wav_path.display()))?;

    println!(
        "[Speech] {} codec frames = {:.2}s of audio (synthesized in {:.2}s) -> {}",
        output.frames,
        output.samples.len() as f64 / f64::from(output.sample_rate),
        synth_start.elapsed().as_secs_f64(),
        wav_path.display()
    );
    Ok(())
}
