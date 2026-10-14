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

//! Runtime dtype of the EAR-TTS backbone stream on a tiny bf16 checkpoint
//! (issue #2109).
//!
//! The reference runs the backbone in `f32` against bf16 weights. On CUDA
//! builds mlxcel's promotion overlay resolves bf16 with f32 to bf16, so any
//! bf16 tensor that meets the stream unconverted demotes it. These tests
//! load every weight as bf16 and check the stream at each place where a
//! stored tensor joins it.

use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::{MogConfig, TtsConfig};
use super::model::RvqEarTtsModel;
use super::tts_tests::{rand, tiny_char_cfg, tiny_subword_weights, tiny_vocab};

const H: i32 = 8;
const LATENT: i32 = 4;
const Q: i32 = 3;
const CODEBOOK: i32 = 4;

fn tiny_config() -> TtsConfig {
    TtsConfig {
        hidden_size: H as usize,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 4,
        sliding_window: 4,
        sliding_window_pattern: 2,
        latent_size: LATENT as usize,
        num_quantizers: Q as usize,
        codebook_size: CODEBOOK as usize,
        character_encoder: tiny_char_cfg(),
        mog_head: MogConfig {
            intermediate_size: 16,
            low_rank: 3,
            num_layers: 2,
            num_predictions: 5,
            ..MogConfig::default()
        },
        ..TtsConfig::default()
    }
}

/// Every EAR-TTS weight under `tts.`, all bf16 as in the checkpoint.
fn tiny_bf16_weights(cfg: &TtsConfig) -> WeightMap {
    let mut key = 61u64;
    let mut w = WeightMap::new();
    let mut put = |k: String, shape: &[i32], scale: f32| {
        let v = mlxcel_core::astype(&rand(&mut key, shape, scale), dtype::BFLOAT16);
        w.insert(format!("tts.{k}"), v);
    };
    let (i, hd) = (cfg.intermediate_size as i32, cfg.head_dim as i32);
    let qd = (cfg.num_attention_heads * cfg.head_dim) as i32;
    let kd = (cfg.num_key_value_heads * cfg.head_dim) as i32;
    for l in 0..cfg.num_hidden_layers {
        let p = format!("backbone.layers.{l}");
        for (name, shape) in [
            ("self_attn.q_proj.weight", [qd, H]),
            ("self_attn.k_proj.weight", [kd, H]),
            ("self_attn.v_proj.weight", [kd, H]),
            ("self_attn.o_proj.weight", [H, qd]),
            ("mlp.gate_proj.weight", [i, H]),
            ("mlp.up_proj.weight", [i, H]),
            ("mlp.down_proj.weight", [H, i]),
        ] {
            put(format!("{p}.{name}"), &shape, 0.3);
        }
        put(format!("{p}.self_attn.q_norm.weight"), &[hd], 0.1);
        put(format!("{p}.self_attn.k_norm.weight"), &[hd], 0.1);
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            put(format!("{p}.{norm}.weight"), &[H], 0.1);
        }
    }
    put("backbone.norm.weight".into(), &[H], 0.1);
    put("bos_emb".into(), &[H], 1.0);
    put("null_emb".into(), &[H], 1.0);
    put("audio_prompt_projection_W".into(), &[H, H], 0.3);
    put("embed_code.weight".into(), &[H, LATENT], 0.3);
    put("rvq_embs".into(), &[Q, CODEBOOK, LATENT], 1.0);
    let fusion = "gated_fusion_audio_text";
    put(format!("{fusion}.audio_proj.weight"), &[H, H], 0.3);
    put(format!("{fusion}.text_proj.weight"), &[H, H], 0.3);
    put(format!("{fusion}.gate"), &[H], 1.0);
    put(format!("{fusion}.final_norm.weight"), &[H], 0.1);
    let mog = &cfg.mog_head;
    let mi = mog.intermediate_size as i32;
    for l in 0..mog.num_layers {
        let p = format!("mog_head.mlp_stack.{l}");
        put(format!("{p}.pre_norm.weight"), &[H], 0.1);
        put(format!("{p}.post_norm.weight"), &[H], 0.1);
        put(format!("{p}.mlp.gate_proj.weight"), &[mi, H], 0.3);
        put(format!("{p}.mlp.up_proj.weight"), &[mi, H], 0.3);
        put(format!("{p}.mlp.down_proj.weight"), &[H, mi], 0.3);
    }
    let (np, rank) = (mog.num_predictions as i32, mog.low_rank as i32);
    put(
        format!("mog_head.mlp_stack.{}.weight", mog.num_layers),
        &[H],
        0.1,
    );
    put("mog_head.proj_logits.weight".into(), &[np, H], 0.3);
    put("mog_head.proj_mus.weight".into(), &[np * rank, H], 0.3);
    put("mog_head.proj_logs.weight".into(), &[1, H], 0.3);
    put("mog_head.proj_else.weight".into(), &[LATENT, H], 0.3);
    put("mog_head.low_mat".into(), &[np, LATENT, rank], 0.3);
    w.insert(
        format!("tts.{fusion}.residual_scale"),
        mlxcel_core::full_f32(&[], 0.5, dtype::BFLOAT16),
    );
    for (k, v) in tiny_subword_weights(&cfg.character_encoder, H, false) {
        let k = k.strip_prefix("sw.").unwrap_or(&k).to_string();
        w.insert(format!("tts.embed_subword.{k}"), v);
    }
    w
}

fn tiny_model() -> RvqEarTtsModel {
    let cfg = tiny_config();
    let mut model = RvqEarTtsModel::from_weights(&tiny_bf16_weights(&cfg), "tts", &cfg).unwrap();
    model.set_vocabulary(&tiny_vocab()).unwrap();
    model
}

fn assert_f32(stage: &str, a: &MlxArray) {
    assert_eq!(
        mlxcel_core::array_dtype(a),
        dtype::FLOAT32,
        "{stage}: the backbone stream must stay f32"
    );
}

fn prompt_codes() -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(&[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], &[1, 4, Q])
}

/// Issue #2109: with every weight stored as bf16, the hidden stream is f32
/// after the prompt embedding assembly (frozen projection and `bos_emb`),
/// after the gated fusion, and after the backbone's norms. Under
/// `--features cuda` each of these demoted the stream to bf16 before the
/// fix.
#[test]
fn backbone_stream_is_f32_with_bf16_weights() {
    let model = tiny_model();
    let code = prompt_codes();
    let audio_mask = [false, false, true, true];
    let (ids, subword_mask) = ([3, 3, 3, 3], [false, true, true, true]);

    // Frames 0-1 are pre-BOS (projection), frame 2 is the BOS frame.
    let embeds = model.prompt_embeds(&code, &audio_mask, None).unwrap();
    assert_f32("prompt embeddings (projection + bos_emb)", &embeds);

    // The guided condition is the reference's bf16 subword path, `null_emb`
    // included; the fusion's text branch is what joins it to the stream.
    let cond = model.guided_condition(&ids, &subword_mask).unwrap();
    assert_eq!(mlxcel_core::array_dtype(&cond), dtype::BFLOAT16);
    assert_f32("gated fusion", &model.backbone_inputs(&embeds, &cond));

    let mut caches = model.make_caches();
    let hidden = model
        .warmup(&code, &ids, &subword_mask, &audio_mask, None, &mut caches)
        .unwrap();
    assert_eq!(mlxcel_core::array_shape(&hidden), vec![2, 4, H]);
    assert_f32("backbone output (final norm)", &hidden);

    let last = mlxcel_core::slice(&code, &[0, 3, 0], &[1, 4, Q]);
    let out = model.step(&last, 3, &mut caches).unwrap();
    assert_f32("step backbone output", &out.hidden_states);
    assert_eq!(mlxcel_core::array_shape(&out.codes), vec![1, 1, Q]);
}
