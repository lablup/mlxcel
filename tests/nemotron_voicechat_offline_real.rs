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

//! End-to-end offline Nemotron VoiceChat parity on the real checkpoint.
//!
//! Runs `NemotronVoiceChatModel::generate_offline` on the reference WAV
//! ("What is the capital of France?", system prompt "Be concise and answer
//! in one sentence.", 3 s extra decoding, seed 0) and compares against the
//! mlx-vlm reference dump: exact text and function ids, the transcript, the
//! answer text, the audio length, and the codec codes and audio.
//!
//! Gated on `MLXCEL_VOICECHAT_MODEL` (checkpoint dir) and
//! `MLXCEL_VOICECHAT_REF` (reference.safetensors); skipped when unset.

use mlxcel::models::NemotronVoiceChatModel;
use mlxcel_core::weights::load_safetensors;

const PROMPT: &str = "Be concise and answer in one sentence.";

fn host_i32(arr: &mlxcel_core::MlxArray) -> Vec<i32> {
    let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::INT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn host_f32(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn offline_turn_matches_reference() {
    let (Ok(model_dir), Ok(ref_path)) = (
        std::env::var("MLXCEL_VOICECHAT_MODEL"),
        std::env::var("MLXCEL_VOICECHAT_REF"),
    ) else {
        eprintln!("skip: MLXCEL_VOICECHAT_MODEL / MLXCEL_VOICECHAT_REF not set");
        return;
    };
    let reference = load_safetensors(&ref_path).unwrap();
    let wav = host_f32(&reference["wav"]);
    let model = NemotronVoiceChatModel::load(std::path::Path::new(&model_dir)).unwrap();

    let result = model.generate_offline(&wav, Some(PROMPT), 3.0, 0).unwrap();
    eprintln!("[user] {:?}", result.user_transcript);
    eprintln!("[assistant] {}", result.text);
    eprintln!("[function] {:?}", result.function_text);

    let want_text = host_i32(&reference["cached_text_tokens"]);
    let want_function = host_i32(&reference["cached_function_tokens"]);
    assert_eq!(result.text_tokens, want_text, "text ids");
    assert_eq!(result.function_tokens, want_function, "function ids");
    assert_eq!(result.text, "The capital of France is Paris.");
    assert_eq!(
        result.user_transcript.as_deref(),
        Some("What is the capital of France?")
    );

    let frames = want_text.len();
    assert_eq!(result.audio.len(), frames * 1764, "1764 samples per frame");
    assert!(result.audio.iter().all(|x| x.is_finite()));

    let want_codes = host_i32(&reference["cached_codes"]);
    let got_codes: Vec<i32> = result.audio_codes.iter().flatten().copied().collect();
    let code_diff = got_codes
        .iter()
        .zip(&want_codes)
        .filter(|(a, b)| a != b)
        .count();
    let first_diff_frame = got_codes
        .iter()
        .zip(&want_codes)
        .position(|(a, b)| a != b)
        .map(|i| i / 31);
    eprintln!(
        "codes differing: {code_diff} of {} (first differing frame {first_diff_frame:?})",
        want_codes.len()
    );

    let want_audio = host_f32(&reference["cached_audio"]);
    let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    eprintln!(
        "audio rms ours {:.4} reference {:.4}",
        rms(&result.audio),
        rms(&want_audio)
    );
    assert!(rms(&result.audio) > 1e-3, "answer audio must not be silent");
    if code_diff == 0 {
        let max_abs = result
            .audio
            .iter()
            .zip(&want_audio)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("audio max abs diff {max_abs:.3e}");
        assert!(max_abs < 1e-3);
    }

    let again = model.generate_offline(&wav, Some(PROMPT), 3.0, 0).unwrap();
    assert_eq!(
        again.audio_codes, result.audio_codes,
        "seeded runs are deterministic"
    );
    assert_eq!(again.audio, result.audio, "seeded runs are byte-identical");
}
