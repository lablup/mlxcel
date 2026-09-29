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

//! Real-checkpoint parity for the NemotronLabs VoiceChat cached streaming
//! perception: streaming log-mel (`lookahead_samples = 1280`), the cache-aware
//! FastConformer and the streamed RNNT transcript, frame by frame against the
//! mlx-audio / mlx-vlm streaming reference.
//!
//! ```text
//! MLXCEL_VOICECHAT_MODEL=/path/to/nemotronlabs-voicechat-11b-4bit \
//! MLXCEL_VOICECHAT_REF=/path/to/reference.safetensors \
//! MLXCEL_VOICECHAT_STREAM_REF=/path/to/stream_ref.safetensors \
//! cargo test --release --test nemotron_voicechat_stream_real -- --nocapture
//! ```
//!
//! The input is `wav` from `reference.safetensors` plus 3 s of silence, pushed
//! in 1280-sample (80 ms) frames. `stream_ref.safetensors` holds the reference
//! `stream_mel [1, M, 128]`, `stream_encoded [1, N, 1024]` and
//! `stream_projected [1, N, 4480]` (one encoder frame per push), and the
//! `stream_summary.json` next to it holds the streamed transcript.

use std::path::Path;

use mlxcel::audio::fastconformer::{ConformerArgs, ConformerStreamingState, VoiceChatPerception};
use mlxcel::audio::nemotron_mel::{MelArgs, StreamingLogMel, log_mel_array};
use mlxcel::audio::rnnt::{JointArgs, PredictArgs, RnntDecoder, RnntStreamState};
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::{WeightMap, load_safetensors, load_weights_from_dir_index_filtered};
use serde_json::Value;

const FRAME: usize = 1280;

struct Diff {
    max_abs: f32,
    rms: f32,
    ref_rms: f32,
}

fn diff(actual: &[f32], expected: &[f32]) -> Diff {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    let n = actual.len() as f64;
    let mut max_abs = 0f32;
    let (mut sq, mut ref_sq) = (0f64, 0f64);
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(a.is_finite(), "non-finite output");
        max_abs = max_abs.max((a - e).abs());
        sq += ((a - e) as f64).powi(2);
        ref_sq += (e as f64).powi(2);
    }
    Diff {
        max_abs,
        rms: (sq / n).sqrt() as f32,
        ref_rms: (ref_sq / n).sqrt() as f32,
    }
}

fn report(label: &str, d: &Diff) {
    eprintln!(
        "{label}: max_abs {:.3e}, rms {:.3e} (reference rms {:.3e})",
        d.max_abs, d.rms, d.ref_rms
    );
}

fn concat(parts: &[mlxcel_core::UniquePtr<mlxcel_core::MlxArray>]) -> Vec<f32> {
    parts.iter().flat_map(|p| array_to_vec_f32(p)).collect()
}

#[test]
fn voicechat_stream_perception_matches_reference() {
    let (Ok(model_dir), Ok(ref_path), Ok(stream_path)) = (
        std::env::var("MLXCEL_VOICECHAT_MODEL"),
        std::env::var("MLXCEL_VOICECHAT_REF"),
        std::env::var("MLXCEL_VOICECHAT_STREAM_REF"),
    ) else {
        eprintln!(
            "Skipping VoiceChat streaming parity: MLXCEL_VOICECHAT_MODEL / MLXCEL_VOICECHAT_REF / MLXCEL_VOICECHAT_STREAM_REF not set"
        );
        return;
    };
    let model_dir = Path::new(&model_dir);
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap())
            .unwrap();
    let audio = &config["audio_config"];
    let mel_args: MelArgs = serde_json::from_value(audio["preprocessor"].clone()).unwrap();
    let enc_args: ConformerArgs = serde_json::from_value(audio["encoder"].clone()).unwrap();
    let predict: PredictArgs = serde_json::from_value(audio["decoder"].clone()).unwrap();
    let joint: JointArgs = serde_json::from_value(audio["joint"].clone()).unwrap();
    let vocabulary: Vec<String> =
        serde_json::from_value(config["rnnt_vocabulary"].clone()).unwrap();
    let max_symbols = audio["max_symbols"].as_u64().unwrap_or(10) as usize;
    let weights: WeightMap = load_weights_from_dir_index_filtered(model_dir, |k| {
        (k.starts_with("stt_model.perception.")
            && !k.starts_with("stt_model.perception.preprocessor."))
            || k.starts_with("stt_model.rnnt_")
    })
    .unwrap();
    let perception =
        VoiceChatPerception::from_weights(&weights, "stt_model.perception", &enc_args).unwrap();
    let rnnt = RnntDecoder::from_weights(
        &weights,
        "stt_model.rnnt_decoder",
        "stt_model.rnnt_joint",
        &predict,
        &joint,
    )
    .unwrap();

    let reference = load_safetensors(&ref_path).unwrap();
    let stream_ref = load_safetensors(&stream_path).unwrap();
    let summary: Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(&stream_path).with_file_name("stream_summary.json"))
            .unwrap(),
    )
    .unwrap();
    let expected_transcript = summary["transcript"].as_str().unwrap().to_string();

    let mut signal = array_to_vec_f32(reference.get("wav").unwrap());
    signal.extend(std::iter::repeat_n(0.0f32, 48_000));
    let pushes = signal.len() / FRAME;

    let mut mel_stream = StreamingLogMel::new(&mel_args, Some(FRAME)).unwrap();
    let att_context = enc_args.default_att_context();
    let mut conformer = ConformerStreamingState::new(perception.encoder(), 1, att_context).unwrap();
    let mut transcript = RnntStreamState::new(&rnnt);
    let (mut mels, mut encs, mut projs) = (Vec::new(), Vec::new(), Vec::new());
    let mut updates = Vec::new();
    let start = std::time::Instant::now();
    for i in 0..pushes {
        let mel = mel_stream
            .push(&signal[i * FRAME..(i + 1) * FRAME])
            .unwrap();
        let mut chunks = conformer.push(&mel, false, true).unwrap();
        assert_eq!(chunks.len(), 1, "push {i}: expected one encoder chunk");
        let encoded = chunks.remove(0);
        assert_eq!(mlxcel_core::array_shape(&encoded), vec![1, 1, 1024]);
        let projected = perception.project(&encoded);
        conformer.materialize(&[&projected, &encoded]);
        if let Some((delta, text)) = transcript
            .step(&rnnt, &encoded, max_symbols, &vocabulary)
            .unwrap()
        {
            updates.push((i, delta, text));
        }
        mels.push(mel);
        encs.push(encoded);
        projs.push(projected);
    }
    let elapsed = start.elapsed();
    eprintln!(
        "{pushes} pushes in {:.1} ms ({:.2} ms per 80 ms frame, perception + RNNT)",
        elapsed.as_secs_f64() * 1e3,
        elapsed.as_secs_f64() * 1e3 / pushes as f64
    );
    assert!(mel_stream.buffered_samples() < 2 * FRAME, "mel buffer grew");

    let get = |k: &str| {
        stream_ref
            .get(k)
            .unwrap_or_else(|| panic!("stream key {k}"))
    };
    let ref_mel = get("stream_mel");
    let ours_mel = concat(&mels);
    assert_eq!(ours_mel.len(), mlxcel_core::array_size(ref_mel));
    let d = diff(&ours_mel, &array_to_vec_f32(ref_mel));
    report("streamed log-mel", &d);
    assert!(d.rms < 1e-4 && d.max_abs < 5e-3, "streamed log-mel drifted");

    let ref_enc = get("stream_encoded");
    assert_eq!(
        mlxcel_core::array_shape(ref_enc),
        vec![1, pushes as i32, 1024]
    );
    let ours_enc = concat(&encs);
    let d = diff(&ours_enc, &array_to_vec_f32(ref_enc));
    report("streamed encoder vs reference stream", &d);
    assert!(d.rms < 2e-3 * d.ref_rms, "streamed encoder drifted");
    let d = diff(&concat(&projs), &array_to_vec_f32(get("stream_projected")));
    report("streamed projection vs reference stream", &d);
    assert!(d.rms < 2e-3 * d.ref_rms, "streamed projection drifted");

    // Streamed vs offline encoder over the same pushed samples (informational:
    // the reference streaming subsampler is close to, not equal to, offline).
    let used = &signal[..pushes * FRAME];
    let offline_mel = log_mel_array(used, &mel_args).unwrap();
    let frames = mlxcel_core::array_shape(&offline_mel)[1] as usize;
    let offline = perception.forward(&offline_mel, frames).unwrap();
    let offline_enc = mlxcel_core::slice(&offline.encoded, &[0, 0, 0], &[1, pushes as i32, 1024]);
    let d = diff(&ours_enc, &array_to_vec_f32(&offline_enc));
    report("streamed encoder vs offline encoder", &d);
    let head = diff(
        &ours_enc[..3 * 1024],
        &array_to_vec_f32(&offline_enc)[..3 * 1024],
    );
    report("streamed vs offline, frames 0..3", &head);
    assert!(
        head.rms < 1e-3 * head.ref_rms.max(1e-6),
        "aligned frames differ from offline"
    );

    for (i, delta, text) in &updates {
        eprintln!("frame {i}: delta {delta:?} -> {text:?}");
    }
    eprintln!("streamed transcript: {:?}", transcript.text);
    assert_eq!(transcript.text, expected_transcript);
    let expected_tokens: Vec<i64> = summary["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let ours_tokens: Vec<i64> = transcript.tokens.iter().map(|&t| t as i64).collect();
    assert_eq!(ours_tokens, expected_tokens);
    let expected_events = summary["events"].as_array().unwrap();
    assert_eq!(updates.len(), expected_events.len());
    for ((i, delta, text), ev) in updates.iter().zip(expected_events) {
        assert_eq!(*i as i64, ev["frame"].as_i64().unwrap());
        assert_eq!(delta, ev["delta"].as_str().unwrap());
        assert_eq!(text, ev["text"].as_str().unwrap());
    }
}
