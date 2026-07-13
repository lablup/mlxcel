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

//! Step-3.7 (step3p7) synthetic parity and integration checks.
//!
//! No real checkpoint is needed (deferred to the orchestrator's real-model
//! validation). Three layers:
//!
//! 1. Geometry identities of the downsampler connector (`52x52 -> 13x13 = 169`,
//!    `36x36 -> 9x9 = 81`) on synthetic encoder output.
//! 2. Full-wrapper construct-and-run from synthetic weights: processor ->
//!    encoder -> downsamplers -> projector -> patches-first scatter -> text
//!    forward, plus the hard-error on a token/feature count mismatch.
//! 3. Fixed-value checks for the Step-3.5 clamped-SwiGLU and sigmoid MoE gate
//!    math the text backbone implements, against precomputed expectations.

use image::{DynamicImage, RgbImage};

use mlxcel::models::Step3p5Model;
use mlxcel::models::step3p5::Step3p5Config;
use mlxcel::multimodal::step3p7_prompt::insert_step3p7_image_tokens;
use mlxcel::vision::connectors::step3p7::Step3p7Connector;
use mlxcel::vision::encoders::step3p7::{Step3p7VisionConfig, Step3p7VisionEncoder};
use mlxcel::vision::processors::step3p7::Step3p7Processor;
use mlxcel::vision::step3p7::{Step3p7TokenIds, Step3p7VlModel};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// ---------- synthetic-weight helpers ----------

fn varied(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (((i % 13) as f32) - 6.0) * 0.05).collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn constant(shape: &[i32], val: f32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![val; n as usize], shape)
}

fn put(map: &mut WeightMap, key: &str, shape: &[i32]) {
    map.insert(key.to_string(), varied(shape));
}

fn put_linear(map: &mut WeightMap, prefix: &str, out: i32, inp: i32, bias: bool) {
    put(map, &format!("{prefix}.weight"), &[out, inp]);
    if bias {
        map.insert(format!("{prefix}.bias"), constant(&[out], 0.0));
    }
}

fn put_layer_norm(map: &mut WeightMap, prefix: &str, dim: i32) {
    map.insert(format!("{prefix}.weight"), constant(&[dim], 1.0));
    map.insert(format!("{prefix}.bias"), constant(&[dim], 0.0));
}

const VW: i32 = 16; // vision width
const HEADS: i32 = 2; // vision heads -> head_dim 8
const VMLP: i32 = 32; // vision MLP hidden
const TH: i32 = 16; // text hidden (== projector output)
const VOCAB: i32 = 24;

fn vision_config() -> Step3p7VisionConfig {
    Step3p7VisionConfig {
        width: VW as usize,
        layers: 1,
        heads: HEADS as usize,
        num_channels: 3,
        image_size: 728,
        patch_size: 14,
        mlp_ratio: VMLP as f64 / VW as f64,
        layer_norm_eps: 1e-5,
        use_cls_token: false,
        use_ln_pre: true,
        use_ln_post: false,
        use_abs_posemb: true,
        use_rope2d: true,
        ls_init_value: 0.1,
        rope_theta: 10000.0,
        quant_group_size: 0,
        quant_bits: 0,
    }
}

fn vision_weights(map: &mut WeightMap) {
    // conv1 in channels-last (out, kH, kW, in) = (VW, 14, 14, 3).
    put(map, "vision_model.conv1.weight", &[VW, 14, 14, 3]);
    // learned position table for the 52x52 base grid.
    put(map, "vision_model.positional_embedding", &[2704, VW]);
    put_layer_norm(map, "vision_model.ln_pre", VW);

    put_layer_norm(map, "vision_model.transformer.0.ln_1", VW);
    put_layer_norm(map, "vision_model.transformer.0.ln_2", VW);
    put_linear(
        map,
        "vision_model.transformer.0.attn.in_proj",
        3 * VW,
        VW,
        true,
    );
    put_linear(
        map,
        "vision_model.transformer.0.attn.out_proj",
        VW,
        VW,
        true,
    );
    map.insert(
        "vision_model.transformer.0.ls_1.gamma".into(),
        constant(&[VW], 0.1),
    );
    map.insert(
        "vision_model.transformer.0.ls_2.gamma".into(),
        constant(&[VW], 0.1),
    );
    put_linear(map, "vision_model.transformer.0.mlp.c_fc", VMLP, VW, true);
    put_linear(map, "vision_model.transformer.0.mlp.c_proj", VW, VMLP, true);

    // Downsampler convs in channels-last: (out, kH, kW, in).
    put(
        map,
        "vision_model.vit_downsampler1.weight",
        &[2 * VW, 3, 3, VW],
    );
    map.insert(
        "vision_model.vit_downsampler1.bias".into(),
        constant(&[2 * VW], 0.0),
    );
    put(
        map,
        "vision_model.vit_downsampler2.weight",
        &[4 * VW, 3, 3, 2 * VW],
    );
    map.insert(
        "vision_model.vit_downsampler2.bias".into(),
        constant(&[4 * VW], 0.0),
    );

    // Projector: 4*width -> text hidden.
    put_linear(map, "vit_large_projector", TH, 4 * VW, false);
}

fn text_config() -> Step3p5Config {
    Step3p5Config::from_nested_text_config(&serde_json::json!({
        "hidden_size": TH,
        "num_hidden_layers": 1,
        "vocab_size": VOCAB,
        "num_attention_heads": 2,
        "num_attention_groups": 1,
        "head_dim": 8,
        "intermediate_size": 32,
        "layer_types": ["full_attention"]
    }))
    .expect("nested text_config parses")
}

fn text_weights(map: &mut WeightMap) {
    put(map, "model.embed_tokens.weight", &[VOCAB, TH]);
    // head_dim 8: q = heads*hd = 16, k/v = groups*hd = 8, o -> hidden.
    put_linear(map, "model.layers.0.self_attn.q_proj", 16, TH, false);
    put_linear(map, "model.layers.0.self_attn.k_proj", 8, TH, false);
    put_linear(map, "model.layers.0.self_attn.v_proj", 8, TH, false);
    put_linear(map, "model.layers.0.self_attn.o_proj", TH, 16, false);
    map.insert(
        "model.layers.0.self_attn.q_norm.weight".into(),
        constant(&[8], 1.0),
    );
    map.insert(
        "model.layers.0.self_attn.k_norm.weight".into(),
        constant(&[8], 1.0),
    );
    put_linear(map, "model.layers.0.mlp.gate_proj", 32, TH, false);
    put_linear(map, "model.layers.0.mlp.up_proj", 32, TH, false);
    put_linear(map, "model.layers.0.mlp.down_proj", TH, 32, false);
    map.insert(
        "model.layers.0.input_layernorm.weight".into(),
        constant(&[TH], 1.0),
    );
    map.insert(
        "model.layers.0.post_attention_layernorm.weight".into(),
        constant(&[TH], 1.0),
    );
    map.insert("model.norm.weight".into(), constant(&[TH], 1.0));
    put(map, "lm_head.weight", &[VOCAB, TH]);
}

fn token_ids() -> Step3p7TokenIds {
    Step3p7TokenIds {
        image_token_index: 5,
        im_start: 6,
        im_end: 7,
        patch_start: 8,
        patch_end: 9,
        patch_newline: 10,
    }
}

fn build_wrapper() -> Step3p7VlModel {
    let mut weights = WeightMap::new();
    vision_weights(&mut weights);
    text_weights(&mut weights);

    let vcfg = vision_config();
    let tcfg = text_config();

    let backbone = Step3p5Model::from_weights(&weights, &tcfg).expect("build text backbone");
    let encoder =
        Step3p7VisionEncoder::from_weights(&weights, &vcfg, "vision_model").expect("build encoder");
    let connector = Step3p7Connector::from_weights(
        &weights,
        "vision_model",
        "vit_large_projector",
        VW as usize,
        2,
        0,
        0,
    )
    .expect("build connector");

    Step3p7VlModel {
        backbone,
        encoder,
        connector,
        processor: Step3p7Processor::new(),
        tokens: token_ids(),
        base_grid: 52,
        patch_grid: 36,
        eos_token_ids: vec![2],
    }
}

// ---------- geometry identities ----------

#[test]
fn connector_collapses_base_grid_52_to_13_giving_169_tokens() {
    let mut weights = WeightMap::new();
    vision_weights(&mut weights);
    let connector = Step3p7Connector::from_weights(
        &weights,
        "vision_model",
        "vit_large_projector",
        VW as usize,
        2,
        0,
        0,
    )
    .expect("build connector");

    // Synthetic encoder output for a 52x52 base grid.
    let hidden = varied(&[1, 2704, VW]);
    let out = connector.forward(hidden.as_ref().unwrap(), 52, 52);
    assert_eq!(
        mlxcel_core::array_shape(out.as_ref().unwrap()),
        vec![1, 169, TH],
        "52x52 -> 13x13 = 169 base tokens"
    );
}

#[test]
fn connector_collapses_patch_grid_36_to_9_giving_81_tokens() {
    let mut weights = WeightMap::new();
    vision_weights(&mut weights);
    let connector = Step3p7Connector::from_weights(
        &weights,
        "vision_model",
        "vit_large_projector",
        VW as usize,
        2,
        0,
        0,
    )
    .expect("build connector");

    // Synthetic encoder output for four 36x36 patch grids.
    let hidden = varied(&[4, 1296, VW]);
    let out = connector.forward(hidden.as_ref().unwrap(), 36, 36);
    assert_eq!(
        mlxcel_core::array_shape(out.as_ref().unwrap()),
        vec![4, 81, TH],
        "36x36 -> 9x9 = 81 patch tokens"
    );
}

#[test]
fn encoder_base_pass_produces_2704_grid_tokens() {
    let mut weights = WeightMap::new();
    vision_weights(&mut weights);
    let encoder = Step3p7VisionEncoder::from_weights(&weights, &vision_config(), "vision_model")
        .expect("build encoder");

    // One 728x728 image -> conv1 (k14 s14) -> 52x52 = 2704 tokens.
    let pixel = varied(&[1, 3, 728, 728]);
    let out = encoder.forward(pixel.as_ref().unwrap());
    assert_eq!(
        mlxcel_core::array_shape(out.as_ref().unwrap()),
        vec![1, 2704, VW],
        "base pass keeps 2704 per-token hidden states"
    );
}

// ---------- full-wrapper construct-and-run ----------

#[test]
fn full_wrapper_orders_patches_first_and_forwards_to_logits() {
    let wrapper = build_wrapper();
    let ids = token_ids();

    // One windowed image (800x800 -> k=2 -> 4 patches).
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(800, 800, image::Rgb([90, 140, 60])));
    let preprocessed = wrapper.processor.preprocess(std::slice::from_ref(&image));
    assert_eq!(preprocessed.layouts[0].num_patches, 4);

    // Expand the single <im_patch> placeholder into the full block.
    let mut prompt = vec![1, ids.image_token_index, 2];
    let stats = insert_step3p7_image_tokens(&mut prompt, &preprocessed.layouts, &ids)
        .expect("placeholder expands");
    // 169 base + 81*4 patch scatter targets.
    assert_eq!(stats.total_image_tokens, 169 + 81 * 4);

    // Patches-first ordering: the first framing token after BOS is <patch_start>.
    let first_special = prompt
        .iter()
        .find(|&&t| t == ids.patch_start || t == ids.im_start)
        .copied();
    assert_eq!(
        first_special,
        Some(ids.patch_start),
        "patch block precedes the base block"
    );

    let im_patch_count = prompt
        .iter()
        .filter(|&&t| t == ids.image_token_index)
        .count();
    assert_eq!(im_patch_count as i32, stats.total_image_tokens);

    let input_ids = mlxcel_core::from_slice_i32(&prompt, &[1, prompt.len() as i32]);
    let embeddings = wrapper
        .input_embeddings(input_ids.as_ref().unwrap(), &preprocessed)
        .expect("input_embeddings succeeds when counts match");
    let embeds = embeddings.inputs_embeds.as_ref().unwrap();
    assert_eq!(
        mlxcel_core::array_shape(embeds),
        vec![1, prompt.len() as i32, TH],
        "merged embeddings keep (batch, seq, text_hidden)"
    );

    // Text forward through the wrapper from the merged embeddings.
    let mut caches = LanguageModel::make_caches(&wrapper);
    let logits = wrapper.forward_with_embeddings(
        input_ids.as_ref().unwrap(),
        Some(embeds),
        &mut caches,
        None,
    );
    assert_eq!(
        mlxcel_core::array_shape(logits.as_ref().unwrap()),
        vec![1, prompt.len() as i32, VOCAB],
        "logits are (batch, seq, vocab)"
    );
}

#[test]
fn token_feature_count_mismatch_is_a_hard_error() {
    let wrapper = build_wrapper();
    let ids = token_ids();

    // Windowed image -> 4 patches -> 493 feature rows expected.
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(800, 800, image::Rgb([30, 30, 30])));
    let preprocessed = wrapper.processor.preprocess(std::slice::from_ref(&image));

    // Deliberately under-count placeholders (only 169, missing the patches).
    let mut prompt = vec![1i32];
    prompt.extend(std::iter::repeat_n(ids.image_token_index, 169));
    prompt.push(2);
    let input_ids = mlxcel_core::from_slice_i32(&prompt, &[1, prompt.len() as i32]);

    let result = wrapper.input_embeddings(input_ids.as_ref().unwrap(), &preprocessed);
    assert!(
        result.is_err(),
        "count mismatch must hard-error, never silently truncate"
    );
}

// ---------- fixed-value Step-3.5 math checks ----------

fn slice_scalar_f32(a: &MlxArray, index: &[i32]) -> f32 {
    let stops: Vec<i32> = index.iter().map(|&x| x + 1).collect();
    let s = mlxcel_core::slice(a, index, &stops);
    mlxcel_core::eval(&s);
    mlxcel_core::item_f32(&s)
}

fn slice_scalar_i32(a: &MlxArray, index: &[i32]) -> i32 {
    let stops: Vec<i32> = index.iter().map(|&x| x + 1).collect();
    let s = mlxcel_core::slice(a, index, &stops);
    mlxcel_core::eval(&s);
    mlxcel_core::item_i32(&s)
}

#[test]
fn clamped_swiglu_matches_precomputed_formula() {
    // out = min(silu(gate), limit) * clip(x, -limit, limit).
    let limit = 1.0f32;
    let gate_host = [-2.0f32, 0.0, 0.5, 3.0];
    let x_host = [2.0f32, -3.0, 0.5, 1.0];

    let silu = |v: f32| v / (1.0 + (-v).exp());
    let expected: Vec<f32> = gate_host
        .iter()
        .zip(&x_host)
        .map(|(&g, &x)| silu(g).min(limit) * x.clamp(-limit, limit))
        .collect();

    let gate = mlxcel_core::from_slice_f32(&gate_host, &[4]);
    let x = mlxcel_core::from_slice_f32(&x_host, &[4]);
    let limit_arr = constant(&[1], limit);
    let neg_limit = constant(&[1], -limit);

    let silu_gate = mlxcel_core::silu(gate.as_ref().unwrap());
    let clamped_gate = mlxcel_core::minimum(&silu_gate, &limit_arr);
    let clamped_x = mlxcel_core::clip(x.as_ref().unwrap(), &neg_limit, &limit_arr);
    let out = mlxcel_core::multiply(&clamped_gate, &clamped_x);

    let expected_arr = mlxcel_core::from_slice_f32(&expected, &[4]);
    let close = mlxcel_core::allclose(
        out.as_ref().unwrap(),
        expected_arr.as_ref().unwrap(),
        1e-5,
        1e-5,
    );
    assert!(
        mlxcel_core::item_bool(&close),
        "clamped SwiGLU op path matches the documented formula"
    );
}

#[test]
fn sigmoid_moe_gate_weights_come_from_uncorrected_scores() {
    // Two experts: bias flips the top-1 selection so the corrected argmax (1)
    // differs from the raw argmax (0). The selection uses corrected scores but
    // the weight is taken from the UNCORRECTED sigmoid score.
    let logits_host = [2.0f32, 1.0];
    let bias_host = [0.0f32, 0.5];
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let scores_host: Vec<f32> = logits_host.iter().map(|&l| sigmoid(l)).collect();
    let sum = scores_host[0] + scores_host[1];

    let logits = mlxcel_core::from_slice_f32(&logits_host, &[1, 1, 2]);
    let bias = mlxcel_core::from_slice_f32(&bias_host, &[2]);

    let scores = mlxcel_core::sigmoid(logits.as_ref().unwrap());
    let corrected = mlxcel_core::add(&scores, &bias);
    let neg = mlxcel_core::negative(&corrected);
    let all_idx = mlxcel_core::argpartition(&neg, 1, -1); // top_k = 2
    let topk_idx = mlxcel_core::slice(&all_idx, &[0, 0, 0], &[1, 1, 2]);
    let topk_w = mlxcel_core::take_along_axis(&scores, &topk_idx, -1);

    // Normalize over the top-k axis (norm_expert_weight), scale 1.0.
    let eps = constant(&[1], 1e-20);
    let wsum = mlxcel_core::sum_axis(&topk_w, -1, true);
    let wsum = mlxcel_core::add(&wsum, &eps);
    let topk_w = mlxcel_core::divide(&topk_w, &wsum);

    // Locate the slot holding expert 1 and confirm its weight uses the raw score.
    let idx0 = slice_scalar_i32(topk_idx.as_ref().unwrap(), &[0, 0, 0]);
    let expert1_slot = if idx0 == 1 { 0 } else { 1 };
    let w_expert1 = slice_scalar_f32(topk_w.as_ref().unwrap(), &[0, 0, expert1_slot]);

    let expected_uncorrected = scores_host[1] / sum;
    let expected_corrected = (scores_host[1] + bias_host[1])
        / (scores_host[0] + bias_host[0] + scores_host[1] + bias_host[1]);
    assert!(
        (w_expert1 - expected_uncorrected).abs() < 1e-4,
        "gate weight uses uncorrected score ({w_expert1} vs {expected_uncorrected})"
    );
    assert!(
        (w_expert1 - expected_corrected).abs() > 1e-3,
        "gate weight is NOT the bias-corrected score"
    );
}
