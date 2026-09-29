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

//! Real-checkpoint parity for the NemotronLabs VoiceChat speech front end:
//! log-mel, FastConformer encoder, perception projection and the RNNT
//! transcript branch, stage by stage against tensors dumped from the
//! mlx-vlm / mlx-audio reference.
//!
//! ```text
//! MLXCEL_VOICECHAT_MODEL=/path/to/nemotronlabs-voicechat-11b-4bit \
//! MLXCEL_VOICECHAT_REF=/path/to/reference.safetensors \
//! [MLXCEL_VOICECHAT_PADDED_REF=/path/to/padded.safetensors] \
//! cargo test --release --test nemotron_voicechat_front_real -- --nocapture
//! ```
//!
//! `reference.safetensors` holds `wav`, `mel [1, T, 128]`, `encoder [1, T', 1024]`
//! and `projected [1, T', 4480]` for a 16 kHz "What is the capital of France?"
//! clip; the optional padded file holds the same clip with 3 s of trailing
//! silence (`padded_wav`, `padded_mel`, `padded_encoder`, `padded_projected`).

use std::path::Path;

use mlxcel::audio::fastconformer::{ConformerArgs, VoiceChatPerception};
use mlxcel::audio::nemotron_mel::{MelArgs, log_mel_spectrogram};
use mlxcel::audio::rnnt::{JointArgs, PredictArgs, RnntDecoder};
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::{WeightMap, load_safetensors, load_weights_from_dir_index_filtered};
use serde_json::Value;

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

struct Front {
    mel: MelArgs,
    perception: VoiceChatPerception,
    rnnt: RnntDecoder,
    vocabulary: Vec<String>,
    max_symbols: usize,
}

fn load_front(model_dir: &Path) -> Front {
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap())
            .unwrap();
    let audio = &config["audio_config"];
    let mel: MelArgs = serde_json::from_value(audio["preprocessor"].clone()).unwrap();
    let encoder: ConformerArgs = serde_json::from_value(audio["encoder"].clone()).unwrap();
    let predict: PredictArgs = serde_json::from_value(audio["decoder"].clone()).unwrap();
    let joint: JointArgs = serde_json::from_value(audio["joint"].clone()).unwrap();
    let vocabulary: Vec<String> =
        serde_json::from_value(config["rnnt_vocabulary"].clone()).unwrap();
    assert_eq!(
        config["rnnt_blank_id"].as_i64(),
        Some(joint.num_classes as i64)
    );

    let weights: WeightMap = load_weights_from_dir_index_filtered(model_dir, |k| {
        (k.starts_with("stt_model.perception.")
            && !k.starts_with("stt_model.perception.preprocessor."))
            || k.starts_with("stt_model.rnnt_")
    })
    .unwrap();
    let perception =
        VoiceChatPerception::from_weights(&weights, "stt_model.perception", &encoder, (64, 4))
            .unwrap();
    let rnnt = RnntDecoder::from_weights(
        &weights,
        "stt_model.rnnt_decoder",
        "stt_model.rnnt_joint",
        &predict,
        &joint,
        (64, 4),
    )
    .unwrap();
    Front {
        mel,
        perception,
        rnnt,
        vocabulary,
        max_symbols: audio["max_symbols"].as_u64().unwrap_or(10) as usize,
    }
}

/// Check one reference clip; `keys` = (wav, mel, encoder, projected).
fn check_clip(front: &Front, reference: &WeightMap, keys: [&str; 4], transcript: &str) {
    let get = |k: &str| {
        reference
            .get(k)
            .unwrap_or_else(|| panic!("reference key {k}"))
    };
    let [wav_key, mel_key, enc_key, proj_key] = keys;

    // 1. Log-mel on the reference waveform.
    let wav = array_to_vec_f32(get(wav_key));
    let (mel, frames) = log_mel_spectrogram(&wav, &front.mel).unwrap();
    let ref_mel = get(mel_key);
    let ref_shape = mlxcel_core::array_shape(ref_mel);
    assert_eq!(ref_shape, vec![1, frames as i32, 128]);
    let d = diff(&mel, &array_to_vec_f32(ref_mel));
    report(&format!("{mel_key} (log-mel)"), &d);
    assert!(d.rms < 1e-4 && d.max_abs < 5e-3, "log-mel drifted");

    // 2. Encoder + projection on the REFERENCE mel (stage isolation).
    let out = front.perception.forward(ref_mel, frames).unwrap();
    let ref_enc = get(enc_key);
    assert_eq!(
        mlxcel_core::array_shape(&out.encoded),
        mlxcel_core::array_shape(ref_enc)
    );
    assert_eq!(out.length as i32, mlxcel_core::array_shape(ref_enc)[1]);
    assert_eq!(
        mlxcel_core::array_dtype(&out.encoded),
        mlxcel_core::dtype::FLOAT32
    );
    let d = diff(&array_to_vec_f32(&out.encoded), &array_to_vec_f32(ref_enc));
    report(&format!("{enc_key} (encoder)"), &d);
    assert!(d.rms < 1e-3 * d.ref_rms.max(1e-6), "encoder drifted");
    let d = diff(
        &array_to_vec_f32(&out.projected),
        &array_to_vec_f32(get(proj_key)),
    );
    report(&format!("{proj_key} (perception proj)"), &d);
    assert!(d.rms < 1e-3 * d.ref_rms.max(1e-6), "projection drifted");

    // 3. RNNT greedy on the reference encoder output and on ours.
    let length = out.length;
    let from_ref = front
        .rnnt
        .transcribe(ref_enc, length, front.max_symbols, &front.vocabulary)
        .unwrap();
    eprintln!("{enc_key}: transcript from reference encoder: {from_ref:?}");
    assert_eq!(from_ref, transcript);
    let from_ours = front
        .rnnt
        .transcribe(&out.encoded, length, front.max_symbols, &front.vocabulary)
        .unwrap();
    eprintln!("{enc_key}: transcript from our encoder: {from_ours:?}");
    assert_eq!(from_ours, transcript);

    // 4. End to end from our own mel.
    let own_mel = mlxcel_core::from_slice_f32(&mel, &[1, frames as i32, 128]);
    let own = front.perception.forward(&own_mel, frames).unwrap();
    let d = diff(&array_to_vec_f32(&own.encoded), &array_to_vec_f32(ref_enc));
    report(&format!("{enc_key} (encoder from our mel)"), &d);
    let e2e = front
        .rnnt
        .transcribe(
            &own.encoded,
            own.length,
            front.max_symbols,
            &front.vocabulary,
        )
        .unwrap();
    eprintln!("{enc_key}: end-to-end transcript: {e2e:?}");
    assert_eq!(e2e, transcript);
}

#[test]
fn voicechat_front_matches_reference() {
    let (Ok(model_dir), Ok(ref_path)) = (
        std::env::var("MLXCEL_VOICECHAT_MODEL"),
        std::env::var("MLXCEL_VOICECHAT_REF"),
    ) else {
        eprintln!(
            "Skipping VoiceChat front parity: MLXCEL_VOICECHAT_MODEL / MLXCEL_VOICECHAT_REF not set"
        );
        return;
    };
    let front = load_front(Path::new(&model_dir));
    let reference = load_safetensors(&ref_path).unwrap();
    check_clip(
        &front,
        &reference,
        ["wav", "mel", "encoder", "projected"],
        "What is the capital of France",
    );

    if let Ok(padded_path) = std::env::var("MLXCEL_VOICECHAT_PADDED_REF") {
        let padded = load_safetensors(&padded_path).unwrap();
        check_clip(
            &front,
            &padded,
            [
                "padded_wav",
                "padded_mel",
                "padded_encoder",
                "padded_projected",
            ],
            "What is the capital of France?",
        );
    }
}
