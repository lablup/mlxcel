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

//! Real-checkpoint parity of the cache-aware online VoiceChat session
//! against the offline timeline (issue #1378).
//!
//! The reference WAV plus 3 s of silence is pushed in 1280-sample frames
//! with the same system prompt and seed as an offline run. The first audio
//! frame's text id, function id and 31 codes must equal the offline ones;
//! the streamed assistant text must equal the offline text; every audio
//! event carries exactly 1764 samples with increasing frame indices. Also
//! checks chunk-size independence (odd chunk sizes give identical events)
//! and that two sessions created back to back on one model do not share
//! state.
//!
//! Gated on `MLXCEL_VOICECHAT_MODEL` and `MLXCEL_VOICECHAT_REF`.

use mlxcel::models::NemotronVoiceChatModel;
use mlxcel::models::nemotron_voicechat::{StreamingOptions, VoiceChatEvent};
use mlxcel_core::weights::load_safetensors;

const PROMPT: &str = "Be concise and answer in one sentence.";

fn host_f32(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

struct StreamRun {
    events: Vec<VoiceChatEvent>,
    text_ids: Vec<i32>,
    function_ids: Vec<i32>,
}

fn stream(model: &NemotronVoiceChatModel, input: &[f32], chunk: usize) -> StreamRun {
    let options = StreamingOptions {
        system_prompt: Some(PROMPT.to_string()),
        seed: 0,
        ..StreamingOptions::default()
    };
    let mut session = model.create_streaming_session(options).unwrap();
    let mut events = Vec::new();
    for piece in input.chunks(chunk) {
        events.extend(session.push_audio(piece, 16_000).unwrap());
    }
    events.extend(session.flush(true).unwrap());
    assert!(session.is_closed());
    assert!(
        session.push_audio(&[0.0; 10], 16_000).is_err(),
        "closed session rejects push"
    );
    let (text, function) = session.channel_ids();
    StreamRun {
        events,
        text_ids: text.to_vec(),
        function_ids: function.to_vec(),
    }
}

#[test]
fn streaming_first_frame_matches_offline() {
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
    let offline = model.generate_offline(&wav, Some(PROMPT), 3.0, 0).unwrap();

    let mut input = wav.clone();
    input.resize(input.len() + 3 * 16_000, 0.0);
    let run = stream(&model, &input, 1280);

    // Event schema: audio frames are 1764 samples with increasing indices,
    // and the stream ends with Done.
    let audio: Vec<(u64, &Vec<f32>, &Vec<i32>)> = run
        .events
        .iter()
        .filter_map(|e| match e {
            VoiceChatEvent::Audio {
                frame_index,
                samples,
                audio_codes,
                ..
            } => Some((*frame_index, samples, audio_codes)),
            _ => None,
        })
        .collect();
    assert!(
        audio
            .iter()
            .all(|(_, s, c)| s.len() == 1764 && c.len() == 31)
    );
    assert!(audio.windows(2).all(|w| w[1].0 == w[0].0 + 1));
    assert_eq!(audio.first().map(|a| a.0), Some(0));
    assert!(matches!(
        run.events.last(),
        Some(VoiceChatEvent::Done { .. })
    ));

    // First-frame parity with the offline path.
    assert_eq!(run.text_ids[0], offline.text_tokens[0], "first text id");
    assert_eq!(
        run.function_ids[0], offline.function_tokens[0],
        "first function id"
    );
    assert_eq!(*audio[0].2, offline.audio_codes[0], "first-frame codes");
    let same_codes = audio
        .iter()
        .zip(&offline.audio_codes)
        .take_while(|(a, b)| *a.2 == **b)
        .count();
    let n = audio.len().min(offline.audio_codes.len());
    let same_text = run
        .text_ids
        .iter()
        .zip(&offline.text_tokens)
        .filter(|(a, b)| a == b)
        .count();
    eprintln!(
        "frames: stream {} offline {}; leading identical code frames {same_codes}/{n}; \
         identical text ids {same_text}/{n}",
        audio.len(),
        offline.audio_codes.len()
    );

    // The streamed text equals the offline text.
    let streamed_text = run
        .events
        .iter()
        .rev()
        .find_map(|e| match e {
            VoiceChatEvent::AssistantTextDelta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default();
    let transcript = run
        .events
        .iter()
        .rev()
        .find_map(|e| match e {
            VoiceChatEvent::UserTranscriptDelta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default();
    eprintln!("[user] {transcript}\n[assistant] {streamed_text}");
    assert_eq!(streamed_text, offline.text);
    assert!(transcript.contains("capital of France"));

    // Chunk boundaries do not change the output.
    let odd = stream(&model, &input, 777);
    assert_eq!(odd.text_ids, run.text_ids, "chunking changed text ids");
    let odd_audio: Vec<&Vec<i32>> = odd
        .events
        .iter()
        .filter_map(|e| match e {
            VoiceChatEvent::Audio { audio_codes, .. } => Some(audio_codes),
            _ => None,
        })
        .collect();
    assert_eq!(odd_audio, audio.iter().map(|a| a.2).collect::<Vec<_>>());
}
