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

//! Nemotron VoiceChat EAR-TTS parity against the mlx-vlm reference.
//!
//! Gated on two environment variables; without them the test prints a skip
//! line and passes:
//!
//! - `MLXCEL_VOICECHAT_MODEL`: the converted checkpoint directory.
//! - `MLXCEL_VOICECHAT_REF`: `reference.safetensors` from the reference dump
//!   (`tts_prompt_codes`, `cached_text_tokens`, `cached_codes`), with its
//!   `summary.json` sibling (`prompt_ids`).
//!
//! Optional `MLXCEL_VOICECHAT_TTS_REF` points at the TTS-only replay dump
//! (`warmup_hidden`, `step_hidden`, untrimmed `generated_codes`, `char_ids`,
//! `char_out`); with it the test also reports continuous diffs per step and
//! runs a teacher-forced pass that separates numeric mismatches from
//! cascaded ones.
//!
//! The replay mirrors `VoiceChatSession.generate`: `random_seed(0)`, warmup
//! on the cached Aria prompt, then one TTS step per timeline position from
//! 1, with PAD as the text over the system-prompt prefix, silence fed back
//! after EOS, and the prefix trimmed plus control codes replaced at the end.

use std::path::{Path, PathBuf};

use mlxcel::models::nemotron_voicechat::tts::{
    RvqEarTtsModel, SpeechDecoderAssets, TtsConfig, TtsPrompt,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

const PAD: i32 = 12;
const EOS: i32 = 2;

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

fn f32s(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

fn i32s(a: &MlxArray) -> Vec<i32> {
    let a = mlxcel_core::astype(a, dtype::INT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `(max_abs, rms)` of `a - b`.
fn diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let mut max = 0f32;
    let mut sq = 0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        max = max.max(d);
        sq += f64::from(d) * f64::from(d);
    }
    (max, (sq / a.len().max(1) as f64).sqrt() as f32)
}

fn get<'a>(map: &'a WeightMap, key: &str) -> &'a MlxArray {
    map.get(key)
        .unwrap_or_else(|| panic!("reference tensor {key} missing"))
}

fn slice_step(a: &MlxArray, t: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(a);
    mlxcel_core::slice(a, &[0, t, 0], &[s[0], t + 1, s[2]])
}

struct Setup {
    model: RvqEarTtsModel,
    assets: SpeechDecoderAssets,
    prompt: TtsPrompt,
    timeline: Vec<i32>,
    prompt_frames: usize,
}

fn setup(model_dir: &Path, reference: &WeightMap, ref_path: &Path) -> Setup {
    let raw = std::fs::read_to_string(model_dir.join("config.json")).expect("config.json");
    let root: serde_json::Value = serde_json::from_str(&raw).expect("parse config.json");
    let mut config: TtsConfig =
        serde_json::from_value(root["tts_config"].clone()).expect("parse tts_config");
    config.quantization = serde_json::from_value(root["quantization"].clone()).ok();

    let weights = mlxcel_core::weights::load_weights_from_dir_index_filtered(model_dir, |k| {
        k.starts_with("tts_model.") && !k.starts_with("tts_model.audio_codec.")
    })
    .expect("load TTS weights");
    let mut model = RvqEarTtsModel::from_weights(&weights, "tts_model.tts_model", &config)
        .expect("build EAR-TTS");
    let assets =
        SpeechDecoderAssets::from_weights(&weights, "tts_model", &config).expect("TTS assets");
    drop(weights);

    let tokenizer =
        tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("tokenizer.json");
    model
        .set_vocabulary(&tokenizer.get_vocab(true))
        .expect("character vocabulary");

    // The reference prompt codes are `_tts_prompt()`'s output; append a
    // dummy trailing frame so the codec-shaped builder can be checked too.
    let prompt_codes = get(reference, "tts_prompt_codes");
    let frames = assets.prompt_frames();
    let q = config.num_quantizers as i32;
    let tail = mlxcel_core::zeros(&[1, 1, q], dtype::INT32);
    let codec_like =
        mlxcel_core::concatenate(&mlxcel_core::astype(prompt_codes, dtype::INT32), &tail, 1);
    let prompt = TtsPrompt::from_codec_codes(&codec_like, frames, &config, PAD).expect("prompt");
    assert_eq!(i32s(&prompt.codes), i32s(prompt_codes), "TtsPrompt codes");

    let summary_path = ref_path.with_file_name("summary.json");
    let prompt_frames = std::fs::read_to_string(&summary_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v["prompt_ids"].as_array().map(Vec::len))
        .unwrap_or(10);
    let mut timeline = vec![PAD; prompt_frames];
    timeline.extend(i32s(get(reference, "cached_text_tokens")));

    Setup {
        model,
        assets,
        prompt,
        timeline,
        prompt_frames,
    }
}

/// Run warmup plus every step. `teacher` supplies the previous frame's codes
/// per position instead of feeding back the model's own.
fn run(
    s: &Setup,
    teacher: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, Vec<UniquePtr<MlxArray>>, Vec<i32>) {
    mlxcel_core::random_seed(0);
    let mut caches = s.model.make_caches();
    let latent = &s.assets.aria_latent;
    let warm = s
        .model
        .warmup(
            &s.prompt.codes,
            &s.prompt.subword_ids,
            &s.prompt.subword_mask,
            &s.prompt.audio_mask,
            Some(latent),
            &mut caches,
        )
        .expect("warmup");
    mlxcel_core::eval(&warm);

    let q = s.model.config().num_quantizers;
    let mut previous = s.prompt.last_frame();
    let mut hidden = Vec::new();
    let mut codes = vec![0i32; s.timeline.len() * q];
    for t in 1..s.timeline.len() {
        if let Some(teacher) = teacher
            && t >= 2
        {
            previous = slice_step(teacher, t as i32 - 1);
        }
        if s.timeline[t] == EOS {
            previous = s.assets.silence_frame();
        }
        let out = s
            .model
            .step(&previous, s.timeline[t], &mut caches)
            .expect("step");
        mlxcel_core::eval(&out.codes);
        codes[t * q..(t + 1) * q].copy_from_slice(&i32s(&out.codes));
        hidden.push(out.hidden_states);
        previous = out.codes;
    }
    (warm, hidden, codes)
}

#[test]
fn ear_tts_matches_reference() {
    let (Some(model_dir), Some(ref_path)) = (
        env_path("MLXCEL_VOICECHAT_MODEL"),
        env_path("MLXCEL_VOICECHAT_REF"),
    ) else {
        eprintln!("Skipping EAR-TTS parity: set MLXCEL_VOICECHAT_MODEL and MLXCEL_VOICECHAT_REF");
        return;
    };
    let reference = mlxcel_core::weights::load_safetensors(&ref_path).expect("reference");
    let tts_ref = env_path("MLXCEL_VOICECHAT_TTS_REF")
        .map(|p| mlxcel_core::weights::load_safetensors(p).expect("TTS reference"));
    let s = setup(&model_dir, &reference, &ref_path);
    let q = s.model.config().num_quantizers;
    let total = s.timeline.len();
    println!(
        "timeline {total} frames ({} prompt + {} audio), {q} codebooks",
        s.prompt_frames,
        total - s.prompt_frames
    );

    if let Some(r) = &tts_ref {
        let ids = i32s(get(r, "char_ids"));
        let out = s
            .model
            .subword_embedding(&ids, &vec![true; ids.len()], 1, ids.len())
            .expect("subword embedding");
        let (max, rms) = diff(&f32s(&out), &f32s(get(r, "char_out")));
        println!(
            "char-aware subword embedding ({} tokens): max_abs {max:.3e} rms {rms:.3e}",
            ids.len()
        );
    }

    let (warm, hidden, codes) = run(&s, None);
    if let Some(r) = &tts_ref {
        let (max, rms) = diff(&f32s(&warm), &f32s(get(r, "warmup_hidden")));
        println!(
            "warmup hidden [2,{},H]: max_abs {max:.3e} rms {rms:.3e}",
            s.prompt.subword_ids.len()
        );
        let step_ref = get(r, "step_hidden");
        for t in [0usize, 1, 9, 10, 18, 22, 30, total - 2] {
            let (max, rms) = diff(&f32s(&hidden[t]), &f32s(&slice_step(step_ref, t as i32)));
            println!("step {:>2} hidden: max_abs {max:.3e} rms {rms:.3e}", t + 1);
        }
        let want = i32s(get(r, "generated_codes"));
        let per_step: Vec<usize> = (1..total)
            .map(|t| {
                (0..q)
                    .filter(|&k| codes[t * q + k] == want[t * q + k])
                    .count()
            })
            .collect();
        let exact = per_step.iter().sum::<usize>();
        let first_bad = per_step.iter().position(|&n| n != q).map(|i| i + 1);
        println!(
            "free-running full-timeline codes: {exact}/{} exact, first diverging position {first_bad:?}",
            (total - 1) * q
        );
        assert_eq!(
            first_bad, None,
            "full-timeline codes diverge from the TTS replay dump"
        );

        let (_, _, forced) = run(&s, Some(get(r, "generated_codes")));
        let forced_exact: Vec<usize> = (1..total)
            .map(|t| {
                (0..q)
                    .filter(|&k| forced[t * q + k] == want[t * q + k])
                    .count()
            })
            .collect();
        let steps_exact = forced_exact.iter().filter(|&&n| n == q).count();
        println!(
            "teacher-forced codes: {}/{} exact, {steps_exact}/{} steps fully exact",
            forced_exact.iter().sum::<usize>(),
            (total - 1) * q,
            total - 1
        );
    }

    // Trim the prefix, replace control codes, compare with the reference.
    let audio = total - s.prompt_frames;
    let trimmed =
        mlxcel_core::from_slice_i32(&codes[s.prompt_frames * q..], &[1, audio as i32, q as i32]);
    let replaced = i32s(&s.assets.replace_control_codes(&trimmed));
    let want = i32s(get(&reference, "cached_codes"));
    let exact = replaced.iter().zip(&want).filter(|(a, b)| a == b).count();
    let frames_exact = (0..audio)
        .filter(|&f| replaced[f * q..(f + 1) * q] == want[f * q..(f + 1) * q])
        .count();
    println!(
        "session codes (trimmed, control-replaced) vs reference cached_codes: {exact}/{} exact, \
         {frames_exact}/{audio} frames exact",
        want.len()
    );
    assert_eq!(replaced.len(), want.len());
    // The run is seeded and mirrors the reference's op order and global-RNG
    // call sequence, so the sampled codes are reproducible bit for bit.
    assert_eq!(exact, want.len(), "session codes differ from the reference");
    assert!(
        replaced
            .iter()
            .all(|&c| (0..s.model.config().codebook_size as i32).contains(&c)),
        "generated codes out of codebook range"
    );
}
