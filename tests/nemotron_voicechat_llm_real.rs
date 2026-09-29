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

//! Real-checkpoint parity for the Nemotron VoiceChat duplex language model.
//!
//! Feeds the reference perception output (`padded_projected`, 62 frames of
//! "What is the capital of France?" plus 3 s of silence) and the reference
//! system-prompt ids through [`VoiceChatLanguageModel`] and requires the
//! greedy text and function ids to equal the mlx-vlm reference exactly.
//!
//! Gated on `MLXCEL_VOICECHAT_MODEL` (checkpoint dir), `MLXCEL_VOICECHAT_REF`
//! (reference.safetensors) and `MLXCEL_VOICECHAT_PADDED` (padded perception
//! dump); skipped when any is unset.

use mlxcel::models::nemotron_h::NemotronHConfig;
use mlxcel::models::nemotron_voicechat::llm::VoiceChatLanguageModel;
use mlxcel_core::weights::{load_safetensors, load_weights_from_dir};

const PROMPT_IDS: [i32; 10] = [1, 5934, 104335, 1321, 4832, 1294, 1925, 19286, 1046, 2];
const PAD: i32 = 12;

fn read_i32(arr: &mlxcel_core::MlxArray) -> Vec<i32> {
    let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::INT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn duplex_llm_matches_reference_tokens() {
    let (Ok(model_dir), Ok(ref_path), Ok(padded_path)) = (
        std::env::var("MLXCEL_VOICECHAT_MODEL"),
        std::env::var("MLXCEL_VOICECHAT_REF"),
        std::env::var("MLXCEL_VOICECHAT_PADDED"),
    ) else {
        eprintln!("skip: MLXCEL_VOICECHAT_MODEL/REF/PADDED not set");
        return;
    };
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{model_dir}/config.json")).unwrap())
            .unwrap();
    let text_config: NemotronHConfig =
        serde_json::from_value(config["text_config"].clone()).unwrap();
    let quant = config["quantization"].as_object().map(|q| {
        (
            q["group_size"].as_i64().unwrap() as i32,
            q["bits"].as_i64().unwrap() as i32,
        )
    });
    let mut weights = load_weights_from_dir(&model_dir).unwrap();
    weights.retain(|k, _| k.starts_with("stt_model.") && !k.starts_with("stt_model.perception"));
    let lm = VoiceChatLanguageModel::from_weights(text_config, quant, &mut weights, 2.0).unwrap();

    let reference = load_safetensors(&ref_path).unwrap();
    let padded = load_safetensors(&padded_path).unwrap();
    let projected = &padded["padded_projected"];
    let audio_frames = mlxcel_core::array_shape(projected)[1];
    let prompt_embeds = lm.embed(&PROMPT_IDS);
    let prompt_embeds = mlxcel_core::astype(&prompt_embeds, mlxcel_core::array_dtype(projected));

    let prompt_frames = PROMPT_IDS.len() as i32;
    let timeline = prompt_frames + audio_frames;
    let mut caches = lm.make_caches();
    let mut text = vec![PAD; timeline as usize];
    let mut function = vec![PAD; timeline as usize];
    for t in 0..timeline {
        let audio = if t < prompt_frames {
            mlxcel_core::slice(&prompt_embeds, &[0, t, 0], &[1, t + 1, 4480])
        } else {
            let a = t - prompt_frames;
            mlxcel_core::slice(projected, &[0, a, 0], &[1, a + 1, 4480])
        };
        let (prev_t, prev_f) = if t == 0 {
            (PAD, PAD)
        } else {
            (text[t as usize - 1], function[t as usize - 1])
        };
        let fused = lm.fused_input(prev_t, &audio, prev_f);
        let out = lm.step(&fused, &mut caches);
        if t >= prompt_frames {
            text[t as usize] = out.text;
            function[t as usize] = out.function;
        }
    }
    let got_text = &text[prompt_frames as usize..];
    let got_function = &function[prompt_frames as usize..];
    let want_text = read_i32(&reference["cached_text_tokens"]);
    let want_function = read_i32(&reference["cached_function_tokens"]);
    eprintln!("text     {got_text:?}");
    eprintln!("function {got_function:?}");
    assert_eq!(
        got_text,
        want_text.as_slice(),
        "text ids differ from reference"
    );
    assert_eq!(
        got_function,
        want_function.as_slice(),
        "function ids differ from reference"
    );
}
