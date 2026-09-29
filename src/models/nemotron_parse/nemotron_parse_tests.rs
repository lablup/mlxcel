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

//! Unit tests for the Nemotron-Parse port (issue #1369): geometry helpers,
//! key canonicalization, config parsing, seeding, the repetition penalty,
//! the page processor, and decode-cache behavior on a tiny synthetic model.
//! Numerical parity with the reference runs against the real checkpoint.

use mlxcel_core::MlxArray;
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

use super::checkpoint::{
    canonicalize_keys, conv1_is_torch_layout, conv2_is_torch_layout,
    reject_unsupported_quantized_tensors,
};
use super::config::NemotronParseConfig;
use super::encoder::{align_corners_weights, bilinear_align_corners, patchify, pos_embed_for_grid};
use super::model::{apply_repetition_penalty, argmax_f32};
use super::neck::row_window_conv;
use super::processor::{DEFAULT_TASK_PROMPT, NemotronParseImageProcessor, seed_from_prompt_ids};

fn host(a: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    array_to_vec_f32(&a)
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "index {i}: {x} vs {y}");
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

#[test]
fn bilinear_align_corners_2x2_to_3x3() {
    let grid = mlxcel_core::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[2, 2, 1]);
    let out = bilinear_align_corners(&grid, 3, 3);
    assert_eq!(mlxcel_core::array_shape(&out), vec![3, 3, 1]);
    close(
        &host(&out),
        &[0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0],
        1e-6,
    );
    // An identity resize is the identity matrix.
    let w = align_corners_weights(5, 5);
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(w[i * 5 + j], if i == j { 1.0 } else { 0.0 });
        }
    }
}

#[test]
fn patchify_order_matches_row_major_grid() {
    // Pixel value encodes (c, y, x) so each patch row can be checked.
    let (c, h, w, p) = (3, 4, 6, 2);
    let data: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
    let px = mlxcel_core::from_slice_f32(&data, &[1, c, h, w]);
    let patches = patchify(&px, p);
    let (hp, wp) = (h / p, w / p);
    assert_eq!(
        mlxcel_core::array_shape(&patches),
        vec![1, hp * wp, c * p * p]
    );
    let got = host(&patches);
    for py in 0..hp {
        for pxi in 0..wp {
            let row = (py * wp + pxi) as usize;
            for ch in 0..c {
                for yy in 0..p {
                    for xx in 0..p {
                        let inner = (ch * p * p + yy * p + xx) as usize;
                        let y = py * p + yy;
                        let x = pxi * p + xx;
                        let expect = (ch * h * w + y * w + x) as f32;
                        assert_eq!(got[row * (c * p * p) as usize + inner], expect);
                    }
                }
            }
        }
    }
}

#[test]
fn pos_embed_native_page_is_identity_crop() {
    // A 4x4 grid read at a 4x3 patch grid: max(4, 3) = 4 equals the grid, so
    // no resampling happens and the first 3 columns of every row are kept
    // (the 2048x1664 page reads the 128x128 grid as 128x104 the same way).
    let (g, ch) = (4, 2);
    let data: Vec<f32> = (0..g * g * ch).map(|i| i as f32).collect();
    let grid = mlxcel_core::from_slice_f32(&data, &[g, g, ch]);
    let pos = pos_embed_for_grid(&grid, 4, 3);
    assert_eq!(mlxcel_core::array_shape(&pos), vec![12, ch]);
    let mut expect = Vec::new();
    for r in 0..4 {
        for col in 0..3 {
            for k in 0..ch {
                expect.push(((r * g + col) * ch + k) as f32);
            }
        }
    }
    assert_eq!(host(&pos), expect);
    assert_eq!(host(&pos_embed_for_grid(&grid, 4, 4)), data);
}

#[test]
fn pos_embed_smaller_page_resamples_then_crops() {
    // 2x2 grid read at a 3x2 patch grid: resample to 3x3, keep 3 rows x 2 cols.
    let grid = mlxcel_core::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[2, 2, 1]);
    let pos = pos_embed_for_grid(&grid, 3, 2);
    close(&host(&pos), &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5], 1e-6);
}

#[test]
fn row_window_conv_matches_a_direct_convolution() {
    let (b, hp, wp, cin, cout, kw) = (1, 2, 9, 3, 2, 4);
    let x: Vec<f32> = (0..b * hp * wp * cin)
        .map(|i| (i % 7) as f32 - 3.0)
        .collect();
    // MLX kernel layout [out, 1, kw, in], flattened to [out, kw * in].
    let w: Vec<f32> = (0..cout * kw * cin).map(|i| (i % 5) as f32 * 0.5).collect();
    let xa = mlxcel_core::from_slice_f32(&x, &[b, hp, wp, cin]);
    let wa = mlxcel_core::from_slice_f32(&w, &[cout, kw * cin]);
    let wt = mlxcel_core::transpose_axes(&wa, &[1, 0]);
    let out = host(&row_window_conv(&xa, &wt, kw));
    let wo = wp / kw; // 2: the trailing partial window is dropped
    assert_eq!(out.len() as i32, b * hp * wo * cout);
    for h in 0..hp {
        for j in 0..wo {
            for o in 0..cout {
                let mut acc = 0.0f32;
                for k in 0..kw {
                    for i in 0..cin {
                        let xv = x[((h * wp + j * kw + k) * cin + i) as usize];
                        let wv = w[((o * kw + k) * cin + i) as usize];
                        acc += xv * wv;
                    }
                }
                let got = out[((h * wo + j) * cout + o) as usize];
                assert!((got - acc).abs() < 1e-4, "{got} vs {acc}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

fn real_shaped_config() -> serde_json::Value {
    json!({
        "model_type": "nemotron_parse",
        "image_size": [2048, 1664],
        "vocab_size": 72256,
        "max_sequence_length": 9000,
        "pad_token_id": 1, "bos_token_id": 0, "eos_token_id": 2,
        "decoder_start_token_id": 2,
        "tie_word_embeddings": true,
        "decoder": {
            "model_type": "nemotron_parse_text", "d_model": 1024, "decoder_layers": 10,
            "decoder_attention_heads": 16, "decoder_ffn_dim": 4096,
            "activation_function": "gelu", "scale_embedding": true, "vocab_size": 72256,
            "decoder_start_token_id": null, "encoder_layers": 12
        },
        "encoder": {
            "patch_size": 16, "max_resolution": 2048,
            "args": {
                "model": "vit_huge_patch16_224", "register_multiple": 8,
                "cls_token_per_teacher": true,
                "teachers": [
                    {"name": "clip", "use_summary": true},
                    {"name": "siglip", "use_summary": true},
                    {"name": "dino_v2", "use_summary": true},
                    {"name": "sam", "use_summary": false}
                ]
            }
        },
        "quantization": {"group_size": 64, "bits": 8, "mode": "affine"}
    })
}

#[test]
fn config_parses_the_real_shape() {
    let cfg = NemotronParseConfig::from_model_config(&real_shaped_config()).unwrap();
    let v = &cfg.vision;
    assert_eq!((v.hidden_size, v.num_heads, v.num_layers), (1280, 16, 32));
    assert_eq!(v.mlp_dim, 5120);
    assert_eq!((v.patch_size, v.pos_grid), (16, 128));
    assert_eq!((v.num_cls_tokens, v.num_register_tokens), (4, 4));
    assert_eq!(v.summary_idxs, vec![0, 1, 2]);
    assert_eq!(v.image_size, (2048, 1664));
    assert_eq!(v.neck_dim, 1024);
    let t = &cfg.text;
    assert_eq!(
        (t.d_model, t.decoder_layers, t.decoder_attention_heads),
        (1024, 10, 16)
    );
    assert_eq!((t.decoder_ffn_dim, t.vocab_size), (4096, 72256));
    assert!(t.scale_embedding);
    assert_eq!(cfg.decoder_start_token_id, 2);
    assert_eq!(cfg.eos_token_id, 2);
    assert_eq!(cfg.max_sequence_length, 9000);
    assert_eq!(cfg.quantization.bits, 8);
}

#[test]
fn neck_output_has_3329_rows_for_native_page() {
    let cfg = NemotronParseConfig::from_model_config(&real_shaped_config()).unwrap();
    assert_eq!(cfg.vision.patch_grid(2048, 1664), (128, 104));
    assert_eq!(cfg.vision.encoder_seq_len(2048, 1664), 3329);
}

#[test]
fn config_rejects_a_non_gelu_decoder_and_unknown_vit() {
    let mut c = real_shaped_config();
    c["decoder"]["activation_function"] = json!("relu");
    assert!(NemotronParseConfig::from_model_config(&c).is_err());
    let mut c = real_shaped_config();
    c["encoder"]["args"]["model"] = json!("vit_giant_patch14");
    assert!(NemotronParseConfig::from_model_config(&c).is_err());
}

// ---------------------------------------------------------------------------
// Seeding, penalty, argmax
// ---------------------------------------------------------------------------

#[test]
fn seed_ids_default_prompt_is_expected_sequence() {
    // The tokenized default prompt already starts with `</s>` (2), so it is
    // used as-is: [2, 0, 50004, 50008, 50001, 50010].
    let ids = vec![2, 0, 50004, 50008, 50001, 50010];
    assert_eq!(seed_from_prompt_ids(ids.clone(), 2), ids);
    assert!(DEFAULT_TASK_PROMPT.starts_with("</s><s>"));
}

#[test]
fn seed_ids_prepend_decoder_start_like_transformers_generate() {
    // A prompt that does not begin with decoder_start gets it prepended
    // (the reference turns [50004, 50008, 50001] into [2, 50004, ...]).
    assert_eq!(
        seed_from_prompt_ids(vec![50004, 50008, 50001], 2),
        vec![2, 50004, 50008, 50001]
    );
    // An empty prompt seeds with decoder_start alone.
    assert_eq!(seed_from_prompt_ids(vec![], 2), vec![2]);
}

#[test]
fn repetition_penalty_follows_transformers_semantics() {
    let mut logits = vec![2.0, -2.0, 4.0, 1.0];
    apply_repetition_penalty(&mut logits, &[0, 1, 1, 7], 2.0);
    // Positive divided, negative multiplied, a repeated id penalized once,
    // an out-of-range id ignored.
    assert_eq!(logits, vec![1.0, -4.0, 4.0, 1.0]);
    let mut same = vec![1.0, -1.0];
    apply_repetition_penalty(&mut same, &[0, 1], 1.0);
    assert_eq!(same, vec![1.0, -1.0]);
}

#[test]
fn argmax_prefers_the_first_maximum() {
    assert_eq!(argmax_f32(&[1.0, 3.0, 3.0, 2.0]), Some(1));
    assert_eq!(argmax_f32(&[f32::NAN, 0.5]), Some(1));
    assert_eq!(argmax_f32(&[]), None);
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

#[test]
fn resize_rule_clamps_height_then_width_with_truncation() {
    let p = NemotronParseImageProcessor::default();
    // Already inside the box: unchanged.
    assert_eq!(p.resized_size(1240, 1754), (1240, 1754));
    // Tall page: height clamps to 2048, width follows.
    assert_eq!(p.resized_size(1000, 4096), (500, 2048));
    // Wide page: width clamps to 1664, height follows.
    assert_eq!(p.resized_size(3328, 1000), (1664, 500));
    // Both: height first (2048 * 2560 / 3000 -> 1747 > 1664), then width.
    // `1664 / (2560 / 3000)` is 1949.999... in f64, and Python `int()`
    // truncates it to 1949, so the port must too.
    assert_eq!(p.resized_size(2560, 3000), (1664, 1949));
}

#[test]
fn preprocess_centre_pads_with_normalized_white() {
    let p = NemotronParseImageProcessor {
        final_size: (4, 6),
        ..Default::default()
    };
    let img = image::RgbImage::from_pixel(2, 2, image::Rgb([0, 0, 0]));
    let (data, h, w) = p.preprocess_to_vec(&image::DynamicImage::ImageRgb8(img));
    assert_eq!((h, w), (4, 6));
    assert_eq!(data.len(), 3 * 24);
    let white0 = (1.0 - p.image_mean[0]) / p.image_std[0];
    let black0 = (0.0 - p.image_mean[0]) / p.image_std[0];
    // pad_top = 1, pad_left = 2: pixels (1..3, 2..4) are the black image.
    for y in 0..4 {
        for x in 0..6 {
            let v = data[y * 6 + x];
            let inside = (1..3).contains(&y) && (2..4).contains(&x);
            let expect = if inside { black0 } else { white0 };
            assert!((v - expect).abs() < 1e-6, "({y},{x}) {v}");
        }
    }
}

// ---------------------------------------------------------------------------
// Key canonicalization
// ---------------------------------------------------------------------------

#[test]
fn conv_transpose_gates_are_idempotent() {
    assert!(conv1_is_torch_layout(&[1024, 1280, 1]));
    assert!(!conv1_is_torch_layout(&[1024, 1, 1280]));
    assert!(!conv1_is_torch_layout(&[1024, 1280]));
    assert!(conv2_is_torch_layout(&[1024, 1024, 1, 4]));
    assert!(!conv2_is_torch_layout(&[1024, 1, 4, 1024]));

    let mut w = WeightMap::new();
    w.insert(
        "encoder.conv1.weight".into(),
        mlxcel_core::zeros(&[6, 5, 1], mlxcel_core::dtype::FLOAT32),
    );
    w.insert(
        "encoder.conv2.weight".into(),
        mlxcel_core::zeros(&[6, 6, 1, 4], mlxcel_core::dtype::FLOAT32),
    );
    let once = canonicalize_keys(w);
    let c1 = mlxcel_core::array_shape(&once["vision_tower.neck.conv1.weight"]);
    let c2 = mlxcel_core::array_shape(&once["vision_tower.neck.conv2.weight"]);
    assert_eq!(c1, vec![6, 5]);
    assert_eq!(c2, vec![6, 1, 4, 6]);
    let twice = canonicalize_keys(once);
    assert_eq!(
        mlxcel_core::array_shape(&twice["vision_tower.neck.conv1.weight"]),
        c1
    );
    assert_eq!(
        mlxcel_core::array_shape(&twice["vision_tower.neck.conv2.weight"]),
        c2
    );
}

#[test]
fn conv2_torch_kernel_is_transposed_to_out_kh_kw_in() {
    // torch [out=2, in=2, kh=1, kw=3]; value = 100 * out + 10 * in + kw.
    let mut w = WeightMap::new();
    let data2: Vec<f32> = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
        .iter()
        .chain([100.0, 101.0, 102.0, 110.0, 111.0, 112.0].iter())
        .copied()
        .collect();
    w.insert(
        "encoder.conv2.weight".into(),
        mlxcel_core::from_slice_f32(&data2, &[2, 2, 1, 3]),
    );
    let out = canonicalize_keys(w);
    let t = &out["vision_tower.neck.conv2.weight"];
    assert_eq!(mlxcel_core::array_shape(t), vec![2, 1, 3, 2]);
    // MLX [out, kh, kw, in]: for out 0, kw 0..3, in 0..2.
    assert_eq!(
        host(t),
        vec![
            0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 100.0, 110.0, 101.0, 111.0, 102.0, 112.0
        ]
    );
}

#[test]
fn hub_keys_map_onto_the_converted_layout() {
    let z = || mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32);
    let mut w = WeightMap::new();
    let radio = "encoder.model_encoder.radio_model";
    for k in [
        format!("{radio}.model.patch_generator.embedder.weight"),
        format!("{radio}.model.patch_generator.pos_embed"),
        format!("{radio}.model.patch_generator.cls_token.token"),
        format!("{radio}.model.blocks.3.attn.qkv.bias"),
        format!("{radio}.summary_idxs"),
        format!("{radio}.input_conditioner.norm_mean"),
        "encoder.sum_proj.weight".to_string(),
        "decoder.embed_tokens.weight".to_string(),
        "decoder.layers.2.fc1.weight".to_string(),
        "decoder.layer_norm.bias".to_string(),
        "decoder.extra_heads.0.weight".to_string(),
        "lm_head.weight".to_string(),
    ] {
        w.insert(k, z());
    }
    let out = canonicalize_keys(w);
    let mut keys: Vec<&str> = out.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "language_model.lm_head.weight",
            "language_model.model.decoder.layer_norm.bias",
            "language_model.model.decoder.layers.2.fc1.weight",
            "language_model.model.shared.weight",
            "vision_tower.blocks.3.attn.qkv.bias",
            "vision_tower.cls_token",
            "vision_tower.neck.sum_proj.weight",
            "vision_tower.patch_embed.weight",
            "vision_tower.pos_embed",
        ]
    );
}

#[test]
fn quantized_dense_only_tensors_are_refused() {
    let z = || mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32);
    let mut w = WeightMap::new();
    w.insert(
        "language_model.model.decoder.layers.0.fc1.scales".into(),
        z(),
    );
    assert!(reject_unsupported_quantized_tensors(&w).is_ok());
    w.insert("vision_tower.neck.conv2.scales".into(), z());
    assert!(reject_unsupported_quantized_tensors(&w).is_err());
}
