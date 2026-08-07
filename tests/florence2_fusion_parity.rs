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

//! Florence-2 end-to-end vision-language parity against the mlx-vlm reference.
//!
//! Pins the whole fused path: DaViT tower -> learned 2-D image position
//! embedding -> cosine temporal embedding -> `image_feature_source` pooling ->
//! `image_projection` / `image_proj_norm` -> concatenation with the embedded
//! task prompt -> BART encoder -> cross-attending decoder logits -> greedy
//! token ids. The reference is the mlx-vlm florence2 `Model`
//! (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/florence2.py)
//! running the same `models/Florence-2-base-ft-bf16` checkpoint with weights
//! cast bf16 -> f16, matching mlxcel's Apple Silicon precision policy. Skips
//! when the checkpoint is absent (CI has no Metal and no weights).
//!
//! The input is the same deterministic synthetic pixel tensor the DaViT
//! backbone parity test uses (`tests/florence2_vision_parity.rs`): Florence-2
//! image preprocessing belongs to the processor sub-issue, and a closed-form
//! tensor is the only way to guarantee both runtimes see bit-identical input
//! today.
//!
//! One deliberate difference on the reference side: upstream's `ModelConfig`
//! reads `image_feature_source`, `image_pos_embed`, and
//! `visual_temporal_embedding` from the *top level* of `config.json`, but
//! every real Florence-2 checkpoint stores them inside `vision_config`. Left
//! alone, upstream therefore falls back to its dataclass defaults, whose
//! `image_feature_source` is `["temporal_avg_pool", "spatial_avg_pool"]`, the
//! reverse of what the checkpoint asks for. The reference run overrides those
//! three fields from `vision_config` so both sides execute the checkpoint's
//! own recipe, which is also what the HuggingFace Florence-2 implementation
//! does.
//!
//! To regenerate the pins, in a virtualenv holding `mlx`, `numpy`, and
//! `Pillow`, with a local checkout of https://github.com/Blaizzy/mlx-vlm on
//! `sys.path` (the `mlx_vlm` package entry can be stubbed so `transformers`
//! is not required): build
//! `ModelConfig` as described above, instantiate `florence2.Model`, load
//! `models/Florence-2-base-ft-bf16/model.safetensors`, pass it through
//! `VisionModel.sanitize` then `Model.sanitize`, cast every tensor to
//! `mx.float16`, and `load_weights`. Then call `_encode_image` on the
//! synthetic pixels, `_merge_input_ids_with_image_features` with
//! `language_model.model.shared(prompt_ids)`, `language_model.model.encoder`,
//! and finally `language_model(inputs=token, encoder_outputs=..., cache=...)`
//! in a greedy loop. Print each tensor's shape, first 16 values, mean, and
//! standard deviation.

use std::path::Path;

use mlxcel::models::Florence2Model;

const MODEL_DIR: &str = "models/Florence-2-base-ft-bf16";
const IMAGE_SIDE: i32 = 768;

/// `<s>What does the image describe?</s>` under the checkpoint tokenizer, the
/// task prompt Florence-2 uses for `<CAPTION>`-style captioning.
const PROMPT_IDS: &[i32] = &[0, 2264, 473, 5, 2274, 6190, 116, 2];

/// 1 spatially averaged token + 576 temporally averaged grid tokens.
const IMAGE_TOKENS: i32 = 577;
const D_MODEL: i32 = 768;

// Reference activations from the mlx-vlm florence2 Model (f16 weights, f32
// readout). The full digit strings are the exact f16-representable reference
// values; truncating them would move the pins off what the reference produced.
#[allow(clippy::excessive_precision)]
const REF_IMAGE_FEATURES_FIRST16: &[f32] = &[
    -0.681640625,
    0.7470703125,
    -0.0171966552734375,
    -1.7763671875,
    1.1005859375,
    0.38134765625,
    -0.1007080078125,
    -0.57373046875,
    0.09912109375,
    -1.4189453125,
    -0.71826171875,
    -0.18505859375,
    -0.36279296875,
    0.1702880859375,
    1.26953125,
    -0.4267578125,
];
const REF_IMAGE_FEATURES_STATS: (f32, f32) = (-0.004598, 0.813462);

#[allow(clippy::excessive_precision)]
const REF_ENCODER_FIRST16: &[f32] = &[
    5.3984375,
    1.9208984375,
    0.93701171875,
    -5.44921875,
    3.546875,
    3.794921875,
    1.384765625,
    -0.7529296875,
    -1.4326171875,
    -1.91015625,
    -1.5458984375,
    -0.94091796875,
    -5.51171875,
    1.9794921875,
    -1.451171875,
    -0.517578125,
];
const REF_ENCODER_STATS: (f32, f32) = (0.064738, 3.541291);

#[allow(clippy::excessive_precision)]
const REF_STEP0_LOGITS_FIRST16: &[f32] = &[
    21.1875,
    -6.484375,
    4.98828125,
    -6.48828125,
    -2.791015625,
    -3.642578125,
    -0.409423828125,
    -0.9736328125,
    -1.2841796875,
    -0.8759765625,
    -3.626953125,
    -1.7275390625,
    -1.4326171875,
    -0.0731201171875,
    -3.181640625,
    -1.7861328125,
];
const REF_STEP0_LOGITS_STATS: (f32, f32) = (-4.34751, 1.661954);

/// Greedy ids after the `decoder_start_token_id` seed, EOS excluded. These
/// decode to `<s>unanswerable`, which is the sensible caption for a
/// procedurally generated noise field.
const REF_GENERATED: &[i32] = &[0, 879, 27740, 868];

/// `x[0, c, h, w] = ((h * side + w) * 3 + c) % 251 / 251.0 - 0.5`, NCHW.
///
/// Identical to `tests/florence2_vision_parity.rs`; every operation is a
/// single IEEE-754 f32 step over exactly represented integers, so the Python
/// reference reproduces it bit for bit.
fn synthetic_pixels(side: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; (3 * side * side) as usize];
    for c in 0..3i64 {
        for h in 0..side as i64 {
            for w in 0..side as i64 {
                let raw = ((h * side as i64 + w) * 3 + c) % 251;
                let value = raw as f32 / 251.0f32 - 0.5f32;
                out[(c * side as i64 * side as i64 + h * side as i64 + w) as usize] = value;
            }
        }
    }
    out
}

fn to_vec_f32(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let dev = (g - w).abs();
        if dev > worst {
            worst = dev;
            worst_at = i;
        }
    }
    eprintln!("{what}: max abs deviation {worst} at index {worst_at} (tol {tol})");
    assert!(
        worst <= tol,
        "{what}[{worst_at}]: got {}, reference {} (deviation {worst}, tol {tol})",
        got[worst_at],
        want[worst_at]
    );
}

fn assert_stats(values: &[f32], want: (f32, f32), tol: f32, what: &str) {
    let n = values.len() as f64;
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = values
        .iter()
        .map(|v| {
            let d = *v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = var.sqrt();
    let (ref_mean, ref_std) = want;
    eprintln!("{what}: mean {mean:.6} (ref {ref_mean:.6}), std {std:.6} (ref {ref_std:.6})");
    assert!(
        (mean as f32 - ref_mean).abs() <= tol,
        "{what} mean: got {mean}, reference {ref_mean} (tol {tol})"
    );
    assert!(
        (std as f32 - ref_std).abs() <= tol,
        "{what} std: got {std}, reference {ref_std} (tol {tol})"
    );
}

#[test]
fn florence2_fusion_matches_mlx_vlm_reference() {
    if !Path::new(MODEL_DIR).exists() {
        eprintln!("skipping florence2_fusion_parity: {MODEL_DIR} not present");
        return;
    }

    let model = Florence2Model::load(Path::new(MODEL_DIR)).expect("load florence2 model");
    let config = model.config();
    assert_eq!(config.text.d_model, D_MODEL);
    assert_eq!(config.text.encoder_attention_heads, 12);
    assert_eq!(config.text.decoder_attention_heads, 12);
    assert_eq!(config.vision.output_dim(), 1024);
    assert_eq!(config.vision.projection_dim, D_MODEL);
    assert_eq!(
        config.vision.image_feature_source,
        vec![
            "spatial_avg_pool".to_string(),
            "temporal_avg_pool".to_string()
        ]
    );

    let pixels = synthetic_pixels(IMAGE_SIDE);
    let pixel_values = mlxcel_core::from_slice_f32(&pixels, &[1, 3, IMAGE_SIDE, IMAGE_SIDE]);

    // 1. Vision tower + fusion-stage projection.
    let image_features = model.encode_image(&pixel_values).expect("encode image");
    assert_eq!(
        mlxcel_core::array_shape(&image_features),
        vec![1, IMAGE_TOKENS, D_MODEL],
        "image feature shape"
    );
    let image_values = to_vec_f32(&image_features);
    // Both sides run f16 with different op ordering. absmax here is 8.6, one
    // f16 ulp at that magnitude is 8e-3, and the observed deviation on Apple
    // Silicon was 2.2e-3; the bound is a small multiple of that and 80x below
    // the tensor's own standard deviation.
    assert_close(
        &image_values[..16],
        REF_IMAGE_FEATURES_FIRST16,
        1e-2,
        "image_features first16",
    );
    assert_stats(
        &image_values,
        REF_IMAGE_FEATURES_STATS,
        2e-3,
        "image_features",
    );

    // 2. Concatenation with the embedded task prompt + joint attention mask.
    let prompt_embeds = model.embed_prompt(PROMPT_IDS).expect("embed prompt");
    let (fused, attention_mask) = model
        .merge_input_ids_with_image_features(&image_features, Some(&prompt_embeds))
        .expect("merge image features");
    let fused_len = IMAGE_TOKENS + PROMPT_IDS.len() as i32;
    assert_eq!(
        mlxcel_core::array_shape(&fused),
        vec![1, fused_len, D_MODEL],
        "fused encoder input shape"
    );
    assert_eq!(
        mlxcel_core::array_shape(&attention_mask),
        vec![1, fused_len],
        "joint attention mask shape"
    );
    // Concatenation, not scatter: the image half of the fused sequence must be
    // the image features unchanged.
    assert_close(
        &to_vec_f32(&fused)[..16],
        REF_IMAGE_FEATURES_FIRST16,
        1e-2,
        "fused first16 (image half)",
    );
    let mask_values = to_vec_f32(&attention_mask);
    assert!(
        mask_values.iter().all(|v| *v == 1.0),
        "joint attention mask must be all ones for unpadded input"
    );

    // 3. BART encoder over the fused sequence.
    let encoder_hidden = model
        .encode(&pixel_values, PROMPT_IDS)
        .expect("fused encode");
    assert_eq!(
        mlxcel_core::array_shape(&encoder_hidden),
        vec![1, fused_len, D_MODEL],
        "encoder hidden state shape"
    );
    let encoder_values = to_vec_f32(&encoder_hidden);
    // absmax 45.8 here, so one f16 ulp is 3.1e-2. Observed deviation 1.5e-2.
    assert_close(
        &encoder_values[..16],
        REF_ENCODER_FIRST16,
        8e-2,
        "encoder_hidden_states first16",
    );
    assert_stats(&encoder_values, REF_ENCODER_STATS, 5e-3, "encoder_hidden");

    // 4. First decoder step against the fused encoder output.
    let mut cache = model.make_cache();
    let start = mlxcel_core::from_slice_i32(&[config.text.decoder_start_token_id], &[1, 1]);
    let logits = model.decode(&start, &encoder_hidden, &mut cache);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 1, config.text.vocab_size],
        "decoder logits shape"
    );
    let logit_values = to_vec_f32(&logits);
    // absmax 21.2, one f16 ulp 1.6e-2. Observed deviation 6.6e-3.
    assert_close(
        &logit_values[..16],
        REF_STEP0_LOGITS_FIRST16,
        5e-2,
        "step0 logits first16",
    );
    assert_stats(&logit_values, REF_STEP0_LOGITS_STATS, 5e-3, "step0 logits");

    // 5. Whole greedy loop.
    let generated = model
        .generate_greedy(&pixel_values, PROMPT_IDS, 8)
        .expect("greedy generation");
    assert_eq!(
        generated, REF_GENERATED,
        "greedy token ids must match the reference"
    );
}

#[test]
fn florence2_fusion_config_parses_real_checkpoint() {
    if !Path::new(MODEL_DIR).exists() {
        eprintln!("skipping florence2_fusion_parity config check: {MODEL_DIR} not present");
        return;
    }
    let raw = std::fs::read_to_string(Path::new(MODEL_DIR).join("config.json"))
        .expect("read florence2 config.json");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("parse florence2 config.json");
    let config =
        mlxcel::models::Florence2Config::from_model_config(&value).expect("parse florence2 config");

    assert_eq!(config.text.vocab_size, 51289);
    assert_eq!(config.text.decoder_start_token_id, 2);
    assert_eq!(config.text.eos_token_id, 2);
    assert!(!config.text.scale_embedding);
    // No `image_token_id` in the checkpoint; upstream's default is one past
    // the last real vocabulary id.
    assert_eq!(config.image_token_id, config.text.vocab_size);
    assert_eq!(config.vision.dim_embed, vec![128, 256, 512, 1024]);
}
