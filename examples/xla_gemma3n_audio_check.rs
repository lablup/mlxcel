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

//! Real-IREE #875 fixture gate for the split Gemma3n audio runtime.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use mlxcel::audio::gemma3n::Gemma3nAudioFeatureExtractor;
use mlxcel::audio::{AudioFamilyPolicy, preprocess_wav_file};
use mlxcel::tokenizer::load_tokenizer;
use mlxcel_xla::{
    GEMMA3N_AUDIO_MEL_BINS, GEMMA3N_AUDIO_SOFT_TOKENS, Gemma3nAudioInput, XlaInferenceSession,
    select_gemma3n_audio_frame_bucket,
};

const PROMPT: &str = "Transcribe the following speech segment in English:";
const AUDIO_TOKEN_ID: i32 = 262_273;
const BOA_TOKEN_ID: i32 = 256_000;
const EOA_TOKEN_ID: i32 = 262_272;
const CONTEXT_CAPACITY: usize = 256;
const EXPECTED_TRANSCRIPT: &str = "Chapter 1. Mrs. Rachel Lind is surprised.\n\nMrs. Rachel Lind";

fn path_env(name: &str, default: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn language_layer_count(model_dir: &Path) -> Result<usize, String> {
    let path = model_dir.join("config.json");
    let bytes =
        std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let config: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let layers = config
        .get("text_config")
        .and_then(|text| text.get("num_hidden_layers"))
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("{} has no text_config.num_hidden_layers", path.display()))?;
    usize::try_from(layers)
        .map_err(|_| format!("Gemma3n language layer count {layers} does not fit usize"))
}

fn expanded_prompt(
    model_dir: &Path,
) -> Result<(mlxcel::tokenizer::MlxcelTokenizer, Vec<i32>), String> {
    let tokenizer = load_tokenizer(model_dir).map_err(|error| error.to_string())?;
    let rendered = format!(
        "<bos><start_of_turn>user\n{PROMPT}<audio_soft_token><end_of_turn>\n\
         <start_of_turn>model\n"
    );
    let mut tokens = tokenizer
        .encode(&rendered, false)
        .map_err(|error| error.to_string())?
        .into_iter()
        .map(|token| token as i32)
        .collect::<Vec<_>>();
    let wrapper = tokenizer
        .encode("\n\n", false)
        .map_err(|error| error.to_string())?
        .into_iter()
        .map(|token| token as i32)
        .collect::<Vec<_>>();
    mlxcel::vlm_runtime::expand_gemma3n_audio_tokens(
        &mut tokens,
        AUDIO_TOKEN_ID,
        BOA_TOKEN_ID,
        EOA_TOKEN_ID,
        1,
        GEMMA3N_AUDIO_SOFT_TOKENS,
        &wrapper,
        None,
    )
    .map_err(|error| error.to_string())?;
    if tokens.len() != 210 {
        return Err(format!(
            "#875 expanded prompt has {} tokens, expected 210",
            tokens.len()
        ));
    }
    Ok((tokenizer, tokens))
}

fn audio_input(audio_path: &Path) -> Result<Gemma3nAudioInput, String> {
    let waveforms = preprocess_wav_file(
        audio_path,
        AudioFamilyPolicy::gemma3n(),
        &AtomicBool::new(false),
    )
    .map_err(|error| error.to_string())?;
    let clips = waveforms
        .clips
        .into_iter()
        .map(|clip| clip.samples)
        .collect::<Vec<_>>();
    let features = Gemma3nAudioFeatureExtractor::new().extract_batch(&clips)?;
    if features.frames != 1_406 {
        return Err(format!(
            "#875 mel frontend emitted {} frames, expected 1406",
            features.frames
        ));
    }
    let bucket =
        select_gemma3n_audio_frame_bucket(features.frames).map_err(|error| error.to_string())?;
    if bucket != 2_048 {
        return Err(format!(
            "#875 selected frame bucket {bucket}, expected 2048"
        ));
    }
    let mut mel = vec![0.0; bucket * GEMMA3N_AUDIO_MEL_BINS];
    for frame in 0..features.frames {
        let start = frame * GEMMA3N_AUDIO_MEL_BINS;
        mel[start..start + GEMMA3N_AUDIO_MEL_BINS]
            .copy_from_slice(&features.features[start..start + GEMMA3N_AUDIO_MEL_BINS]);
    }
    let mut valid_mask = vec![0; bucket];
    for (output, valid) in valid_mask
        .iter_mut()
        .zip(features.valid_mask.iter().copied())
    {
        *output = u8::from(valid);
    }
    let valid_frames = features.valid_mask.iter().filter(|&&valid| valid).count();
    Gemma3nAudioInput::new(mel, valid_mask, vec![valid_frames], bucket)
        .map_err(|error| error.to_string())
}

fn main() -> Result<(), String> {
    let model_dir = path_env(
        "MLXCEL_GEMMA3N_AUDIO_MODEL",
        "/home/inureyes/models/gemma3n-e4b-4bit",
    );
    let audio_path = path_env("MLXCEL_GEMMA3N_AUDIO_WAV", "/tmp/gemma3n-103-1240-0000.wav");
    let (tokenizer, tokens) = expanded_prompt(&model_dir)?;
    let input = audio_input(&audio_path)?;
    eprintln!(
        "gemma3n-audio-check: frontend frames=1406 bucket={} expanded={}",
        input.frame_bucket(),
        tokens.len()
    );

    let layers = language_layer_count(&model_dir)?;
    let mut session =
        XlaInferenceSession::load_with_context_capacity(&model_dir, layers, CONTEXT_CAPACITY)?;
    eprintln!("gemma3n-audio-check: #876 language bundle loaded");
    let mut audio = session.load_gemma3n_audio(input.frame_bucket(), input.clips())?;
    if !audio.has_capability() {
        return Err("Gemma3n audio capability stayed false after verified bundle load".to_string());
    }
    eprintln!("gemma3n-audio-check: split audio bundle loaded");
    let prepared = audio.invoke_prepared(&input, tokens, AUDIO_TOKEN_ID)?;
    eprintln!(
        "gemma3n-audio-check: encode->merge complete, projected_lengths={:?}",
        prepared.projected_lengths()
    );
    let eos_token_ids = session.eos_token_ids().to_vec();
    let output =
        session.generate_gemma3n_prepared_greedy(prepared.request(), 16, &eos_token_ids)?;
    let text = tokenizer
        .decode(
            &output.iter().map(|token| *token as u32).collect::<Vec<_>>(),
            true,
        )
        .map_err(|error| error.to_string())?;
    eprintln!("gemma3n-audio-check: generated tokens={output:?}");
    if text != EXPECTED_TRANSCRIPT {
        return Err(format!(
            "#875 token-exact transcript mismatch:\nexpected: {EXPECTED_TRANSCRIPT:?}\nactual:   {text:?}"
        ));
    }
    println!("{text}");
    Ok(())
}
