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

//! Real-checkpoint parity for the NemotronLabs VoiceChat codec.
//!
//! Needs `MLXCEL_VOICECHAT_MODEL` (converted checkpoint dir) and
//! `MLXCEL_VOICECHAT_REF` (`reference.safetensors` dumped from the mlx-vlm /
//! mlx-audio reference). Skips when either is unset.
//!
//! Checks: tone encode codes (exact), tone decode and offline answer decode
//! against the reference reconstructions, the silent TTS prompt codes
//! (frames 1..=35, exact), and streaming `decode_step` against the full
//! decode after the reference's 8-sample stream delay.

use std::path::PathBuf;

use mlxcel::audio::nemotron_codec::{CausalConv1dCache, CodecConfig, NemotronCodec};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const PREFIX: &str = "tts_model.audio_codec";

fn env_paths() -> Option<(PathBuf, PathBuf)> {
    let model = std::env::var_os("MLXCEL_VOICECHAT_MODEL")?;
    let reference = std::env::var_os("MLXCEL_VOICECHAT_REF")?;
    Some((PathBuf::from(model), PathBuf::from(reference)))
}

fn get<'a>(map: &'a WeightMap, key: &str) -> &'a MlxArray {
    map.get(key)
        .unwrap_or_else(|| panic!("reference tensor {key} missing"))
}

fn f32s(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

fn stats(label: &str, ours: &[f32], reference: &[f32]) -> (f32, f32) {
    assert_eq!(ours.len(), reference.len(), "{label}: length mismatch");
    let mut max = 0.0f32;
    let mut sq = 0.0f64;
    for (a, b) in ours.iter().zip(reference) {
        let d = (a - b).abs();
        max = max.max(d);
        sq += f64::from(d) * f64::from(d);
    }
    let rms = (sq / ours.len() as f64).sqrt() as f32;
    let ref_rms = (reference.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>()
        / reference.len() as f64)
        .sqrt();
    println!(
        "{label}: n={} max_abs={max:.3e} rms={rms:.3e} (reference rms {ref_rms:.3e})",
        ours.len()
    );
    (max, rms)
}

fn code_diffs(label: &str, ours: &[f32], reference: &[f32]) -> usize {
    assert_eq!(ours.len(), reference.len(), "{label}: length mismatch");
    let diffs = ours.iter().zip(reference).filter(|(a, b)| a != b).count();
    println!("{label}: {diffs} of {} codes differ", ours.len());
    diffs
}

fn load_codec(model: &PathBuf) -> NemotronCodec {
    let raw = std::fs::read_to_string(model.join("config.json")).expect("read config.json");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse config.json");
    let config: CodecConfig =
        serde_json::from_value(json["codec_config"].clone()).expect("codec_config");
    let weights = mlxcel_core::weights::load_weights_from_dir_index_filtered(model, |k| {
        k.starts_with("tts_model.audio_codec.")
    })
    .expect("load codec weights");
    NemotronCodec::from_weights(&weights, PREFIX, &config).expect("build codec")
}

fn shaped(a: &MlxArray, shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::reshape(a, shape)
}

#[test]
fn nemotron_voicechat_codec_matches_reference() {
    let Some((model, ref_path)) = env_paths() else {
        println!("skip: set MLXCEL_VOICECHAT_MODEL and MLXCEL_VOICECHAT_REF to run");
        return;
    };
    let codec = load_codec(&model);
    let reference = mlxcel_core::weights::load_safetensors(&ref_path).expect("load reference");
    assert_eq!(codec.waveform_to_token_ratio(), 1764);

    // 1) Tone encode must reproduce the reference codes exactly.
    let tone = get(&reference, "tone");
    let tone_len = mlxcel_core::array_shape(tone)[0];
    let codes = codec
        .encode(&shaped(tone, &[1, 1, tone_len]))
        .expect("encode tone");
    let ref_codes = get(&reference, "tone_codes");
    assert_eq!(
        mlxcel_core::array_shape(&codes),
        mlxcel_core::array_shape(ref_codes)
    );
    assert_eq!(code_diffs("tone codes", &f32s(&codes), &f32s(ref_codes)), 0);

    // 2) Tone decode vs the reference reconstruction (bf16 decoder; the port
    //    mirrors the reference dtypes, so this is near bit-exact).
    let recon = codec.decode(ref_codes).expect("decode tone");
    let (max, rms) = stats(
        "tone recon",
        &f32s(&recon),
        &f32s(get(&reference, "tone_recon")),
    );
    assert!(
        max < 1e-4 && rms < 1e-5,
        "tone recon drifted: max {max} rms {rms}"
    );

    // 3) Offline decode of the reference answer (already control-code replaced).
    let cached = get(&reference, "cached_codes");
    let cached_t = mlxcel_core::transpose_axes(cached, &[1, 0]);
    let cshape = mlxcel_core::array_shape(&cached_t);
    let answer_codes = shaped(&cached_t, &[1, cshape[0], cshape[1]]);
    let full = f32s(&codec.decode(&answer_codes).expect("decode answer"));
    let (max, rms) = stats(
        "answer decode",
        &full,
        &f32s(get(&reference, "cached_audio")),
    );
    assert!(
        max < 1e-4 && rms < 1e-5,
        "answer decode drifted: max {max} rms {rms}"
    );

    // 4) Silent TTS prompt: encode (37 + 1) frames of zeros; frames 1..=35
    //    must match (0 and 36 are overwritten with the mask code).
    let ratio = codec.waveform_to_token_ratio() as i32;
    let silence = mlxcel_core::zeros(&[1, 1, 38 * ratio], mlxcel_core::dtype::FLOAT32);
    let prompt = codec.encode(&silence).expect("encode silence");
    let prompt = mlxcel_core::transpose_axes(&prompt, &[0, 2, 1]);
    let ours = mlxcel_core::slice(&prompt, &[0, 1, 0], &[1, 36, 31]);
    let theirs = mlxcel_core::slice(
        get(&reference, "tts_prompt_codes"),
        &[0, 1, 0],
        &[1, 36, 31],
    );
    assert_eq!(
        code_diffs("tts prompt codes 1..=35", &f32s(&ours), &f32s(&theirs)),
        0
    );

    // 5) Streaming decode_step over the answer vs the full decode. The bf16
    //    decoder sees different conv batch shapes per step, so the reference
    //    itself differs from its full decode by max 1.8e-3 / RMS 8.5e-5 here.
    let frames = cshape[1];
    let delay = codec.stream_delay_samples();
    let mut cache = CausalConv1dCache::new();
    let mut stream = Vec::with_capacity(full.len() + delay);
    for t in 0..frames {
        let step = mlxcel_core::slice(&answer_codes, &[0, 0, t], &[1, cshape[0], t + 1]);
        let out = codec
            .decode_step(&step, &mut cache, t == frames - 1)
            .expect("decode_step");
        stream.extend(f32s(&out));
    }
    assert_eq!(stream.len(), full.len() + delay);
    let (max, rms) = stats("stream[8..] vs full decode", &stream[delay..], &full);
    assert!(
        max < 1e-2 && rms < 5e-4,
        "stream drifted: max {max} rms {rms}"
    );
}
