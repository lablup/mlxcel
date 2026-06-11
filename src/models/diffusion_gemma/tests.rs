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

//! Unit tests for the DiffusionGemma module: pure host-side helper
//! functions, generation-config parsing (against a literal snippet of the
//! real checkpoint config), the fused gate_up split arithmetic, and the
//! dense-axis sliding-window mask.

use super::generate::{
    StabilityTracker, canvas_length_for, confidence_transfer_mask, debug_canvas_pattern,
    entropy_bound_accept_count, entropy_bound_acceptance_mask, linear_schedule_temperature,
};
use super::*;

// -------------------------------------------------------------------------
// Entropy-bound sampler
// -------------------------------------------------------------------------

#[test]
fn entropy_bound_accept_count_all_below_bound() {
    // prefix sums before each rank: 0, 0.01, 0.03, 0.06 — all <= 0.1.
    let sorted = [0.01, 0.02, 0.03, 0.02];
    assert_eq!(entropy_bound_accept_count(&sorted, 0.1), 4);
}

#[test]
fn entropy_bound_accept_count_none_below_except_forced_first() {
    // Rank 0 always qualifies (prefix sum 0 <= bound); rank 1 has prefix
    // 5.0 > 0.1, so exactly one position is accepted.
    let sorted = [5.0, 6.0, 7.0];
    assert_eq!(entropy_bound_accept_count(&sorted, 0.1), 1);
}

#[test]
fn entropy_bound_accept_count_negative_bound_forces_one() {
    let sorted = [0.5, 0.6];
    assert_eq!(entropy_bound_accept_count(&sorted, -1.0), 1);
}

#[test]
fn entropy_bound_accept_count_partial_prefix() {
    // prefixes: 0, 0.05, 0.15 -> ranks 0 and 1 qualify (0 <= 0.1,
    // 0.05 <= 0.1), rank 2 fails (0.15 > 0.1).
    let sorted = [0.05, 0.10, 0.20];
    assert_eq!(entropy_bound_accept_count(&sorted, 0.1), 2);
}

#[test]
fn entropy_bound_accept_count_empty_is_zero() {
    assert_eq!(entropy_bound_accept_count(&[], 0.1), 0);
}

#[test]
fn entropy_bound_acceptance_mask_picks_lowest_entropy_positions() {
    // Entropies (unsorted): positions 1 and 3 are the two lowest.
    // sorted: [0.01 (pos 3), 0.02 (pos 1), 0.5 (pos 0), 0.9 (pos 2)]
    // prefixes: 0, 0.01, 0.03 (> bound 0.02 at rank 2 -> accept 2).
    let entropies = [0.5, 0.02, 0.9, 0.01];
    let mask = entropy_bound_acceptance_mask(&entropies, 0.02);
    assert_eq!(mask, vec![false, true, false, true]);
}

#[test]
fn entropy_bound_acceptance_mask_ties_prefer_lower_index() {
    // Exact ties: stable sort keeps position order, so with bound 0.0 only
    // the forced first (lowest index among the minimum) is accepted.
    let entropies = [0.3, 0.3, 0.3];
    let mask = entropy_bound_acceptance_mask(&entropies, 0.0);
    assert_eq!(mask, vec![true, false, false]);
}

// -------------------------------------------------------------------------
// Confidence-threshold sampler
// -------------------------------------------------------------------------

#[test]
fn confidence_transfer_mask_accepts_above_threshold() {
    let confidence = [0.95, 0.5, 0.91];
    let unrevealed = [true, true, true];
    let mask = confidence_transfer_mask(&confidence, &unrevealed, 0.9, false);
    assert_eq!(mask, vec![true, false, true]);
}

#[test]
fn confidence_transfer_mask_forces_best_unrevealed_when_none_clear() {
    let confidence = [0.2, 0.8, 0.5];
    let unrevealed = [true, true, true];
    let mask = confidence_transfer_mask(&confidence, &unrevealed, 0.9, false);
    assert_eq!(mask, vec![false, true, false]);
}

#[test]
fn confidence_transfer_mask_force_ignores_revealed_positions() {
    // Highest raw confidence is revealed; force must pick the best among
    // the unrevealed ones only.
    let confidence = [0.99, 0.3, 0.6];
    let unrevealed = [false, true, true];
    let mask = confidence_transfer_mask(&confidence, &unrevealed, 0.9, false);
    assert_eq!(mask, vec![false, false, true]);
}

#[test]
fn confidence_transfer_mask_no_unrevealed_yields_empty_mask() {
    let confidence = [0.1, 0.2];
    let unrevealed = [false, false];
    let mask = confidence_transfer_mask(&confidence, &unrevealed, 0.9, false);
    assert_eq!(mask, vec![false, false]);
}

#[test]
fn confidence_transfer_mask_force_all_returns_unrevealed() {
    let confidence = [0.0, 0.0, 0.0];
    let unrevealed = [true, false, true];
    let mask = confidence_transfer_mask(&confidence, &unrevealed, 0.9, true);
    assert_eq!(mask, vec![true, false, true]);
}

// -------------------------------------------------------------------------
// Temperature schedule and stopping
// -------------------------------------------------------------------------

#[test]
fn linear_schedule_first_and_last_step() {
    // First iteration: cur_step == max_steps -> tau == t_max.
    let first = linear_schedule_temperature(48, 48, 0.4, 0.8);
    assert!((first - 0.8).abs() < 1e-6, "first tau {first}");
    // Last executed step: cur_step == 1 -> t_min + range / max_steps.
    let last = linear_schedule_temperature(1, 48, 0.4, 0.8);
    let expected = 0.4 + (0.8 - 0.4) * (1.0 / 48.0);
    assert!((last - expected).abs() < 1e-6, "last tau {last}");
}

#[test]
fn stability_tracker_checks_history_before_pushing() {
    // stability_threshold = 1: the FIRST observation can never be stable
    // (history empty), the second identical one is stable.
    let mut tracker = StabilityTracker::new(1);
    assert!(!tracker.observe(&[1, 2, 3]));
    assert!(tracker.observe(&[1, 2, 3]));
    // A change resets stability.
    assert!(!tracker.observe(&[1, 2, 4]));
    assert!(tracker.observe(&[1, 2, 4]));
}

#[test]
fn stability_tracker_threshold_two_needs_two_prior_matches() {
    let mut tracker = StabilityTracker::new(2);
    assert!(!tracker.observe(&[7]));
    assert!(!tracker.observe(&[7]));
    // History is now [7, 7] (full): the third identical canvas is stable.
    assert!(tracker.observe(&[7]));
}

// -------------------------------------------------------------------------
// Canvas length rule and debug pattern
// -------------------------------------------------------------------------

#[test]
fn canvas_length_rule_matches_reference() {
    // min(max_canvas, max(remaining, min_canvas))
    assert_eq!(canvas_length_for(300, 64, 256, 256, false), 256);
    assert_eq!(canvas_length_for(100, 64, 256, 256, false), 100);
    assert_eq!(canvas_length_for(10, 64, 256, 256, false), 64);
    // full_canvas always allocates the model canvas.
    assert_eq!(canvas_length_for(10, 64, 128, 256, true), 256);
}

#[test]
fn debug_canvas_pattern_matches_formula() {
    let vocab = 262_144i64;
    let block = debug_canvas_pattern(4, vocab, 0);
    assert_eq!(block, vec![7919, 15838, 23757, 31676]);
    let block_k2 = debug_canvas_pattern(2, vocab, 2);
    let expected: Vec<i32> = (0..2i64)
        .map(|i| (((i + 1) * 7919 + 2 * 104_729) % vocab) as i32)
        .collect();
    assert_eq!(block_k2, expected);
}

// -------------------------------------------------------------------------
// Config parsing (literal snippet of the real checkpoint config.json)
// -------------------------------------------------------------------------

const REAL_CONFIG_SNIPPET: &str = r#"{
  "architectures": ["DiffusionGemmaForBlockDiffusion"],
  "model_type": "diffusion_gemma",
  "canvas_length": 256,
  "eos_token_id": [1, 106, 50],
  "boi_token_id": 255999,
  "eoi_token_id": 258882,
  "image_token_id": 258880,
  "tie_word_embeddings": true,
  "vision_soft_tokens_per_image": 280,
  "vision_config": {
    "model_type": "gemma4_vision",
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 72,
    "global_head_dim": 72,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "rms_norm_eps": 1e-06,
    "standardize": true,
    "use_clipped_linears": false,
    "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"}
  },
  "generation_config": {
    "confidence_threshold": 0.005,
    "eos_token_id": [1, 106, 50],
    "max_denoising_steps": 48,
    "max_new_tokens": 256,
    "pad_token_id": 0,
    "sampler_config": {
      "_cls_name": "EntropyBoundSamplerConfig",
      "entropy_bound": 0.1
    },
    "stability_threshold": 1,
    "t_max": 0.8,
    "t_min": 0.4
  },
  "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 2,
    "dtype": "bfloat16",
    "eos_token_id": 1,
    "final_logit_softcapping": 30.0,
    "global_head_dim": 512,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 2816,
    "initializer_range": 0.02,
    "intermediate_size": 2112,
    "layer_types": [
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention"
    ],
    "max_position_embeddings": 262144,
    "model_type": "diffusion_gemma_text",
    "moe_intermediate_size": 704,
    "num_attention_heads": 16,
    "num_experts": 128,
    "num_global_key_value_heads": 2,
    "num_hidden_layers": 30,
    "num_key_value_heads": 8,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional"
      },
      "sliding_attention": {
        "rope_theta": 10000.0,
        "rope_type": "default"
      }
    },
    "sliding_window": 1024,
    "tie_word_embeddings": true,
    "top_k_experts": 8,
    "use_bidirectional_attention": "vision",
    "vocab_size": 262144
  }
}"#;

#[test]
fn parses_real_config_snippet() {
    let args: ModelArgs = serde_json::from_str(REAL_CONFIG_SNIPPET).expect("config parses");
    assert_eq!(args.model_type, "diffusion_gemma");
    assert_eq!(args.canvas_length(), 256);

    let generation = args.generation_config().expect("generation config");
    assert_eq!(generation.max_denoising_steps, 48);
    assert_eq!(generation.max_new_tokens, 256);
    assert!((generation.t_min - 0.4).abs() < 1e-6);
    assert!((generation.t_max - 0.8).abs() < 1e-6);
    assert!((generation.entropy_bound - 0.1).abs() < 1e-6);
    let stopping = generation.stopping.expect("stopping config present");
    assert!((stopping.confidence_threshold - 0.005).abs() < 1e-6);
    assert_eq!(stopping.stability_threshold, 1);

    let eos = args.eos_token_ids(&generation);
    assert_eq!(eos, vec![1, 106, 50]);
}

#[test]
fn text_args_forces_structural_flags() {
    let args: ModelArgs = serde_json::from_str(REAL_CONFIG_SNIPPET).expect("config parses");
    let config = args.text_args().expect("text config parses");

    // Shape parameters straight from the checkpoint.
    assert_eq!(config.hidden_size, 2816);
    assert_eq!(config.num_hidden_layers, 30);
    assert_eq!(config.vocab_size, 262_144);
    assert_eq!(config.sliding_window, 1024);
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.global_head_dim, Some(512));
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.num_global_key_value_heads, Some(2));
    assert_eq!(config.num_experts, Some(128));
    assert_eq!(config.top_k_experts, Some(8));
    assert_eq!(config.moe_intermediate_size, Some(704));
    assert_eq!(config.intermediate_size, 2112);
    assert_eq!(config.final_logit_softcapping, Some(30.0));
    assert_eq!(config.layer_types.len(), 30);

    // The checkpoint text_config OMITS these presence-only flags; the
    // parser must force them because the weights require both: the 5
    // full-attention layers ship without v_proj (k-eq-v values) and every
    // layer carries the MoE branch.
    assert!(config.attention_k_eq_v, "k-eq-v must be forced on");
    assert!(config.enable_moe_block, "MoE branch must be forced on");
    assert_eq!(config.num_kv_shared_layers, 0);
    assert_eq!(config.hidden_size_per_layer_input, 0);

    // Root quantization is inherited when text_config has none.
    let quant = config.quantization.expect("quantization inherited");
    assert_eq!(quant.group_size, 64);
    assert_eq!(quant.bits, 4);
}

#[test]
fn rejects_unknown_sampler_class() {
    let mut value: serde_json::Value =
        serde_json::from_str(REAL_CONFIG_SNIPPET).expect("config parses");
    value["generation_config"]["sampler_config"]["_cls_name"] =
        serde_json::Value::String("SomeOtherSamplerConfig".to_string());
    let args: ModelArgs =
        serde_json::from_value(value).expect("config with foreign sampler parses");
    let err = args.generation_config().expect_err("must reject");
    assert!(err.contains("SomeOtherSamplerConfig"), "error: {err}");
}

// -------------------------------------------------------------------------
// Fused gate_up split arithmetic (synthetic shapes, no real tensors)
// -------------------------------------------------------------------------

#[test]
fn split_gate_up_tensor_splits_output_axis() {
    // [experts = 2, out = 4, k = 3]: rows 0..2 are the gate half, rows
    // 2..4 the up half (gate first, matching the reference slicing).
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let fused = mlxcel_core::from_slice_f32(&data, &[2, 4, 3]);
    let (gate, up) = split_gate_up_tensor(&fused, 2).expect("split succeeds");
    mlxcel_core::eval(&gate);
    mlxcel_core::eval(&up);

    assert_eq!(mlxcel_core::array_shape(&gate), vec![2, 2, 3]);
    assert_eq!(mlxcel_core::array_shape(&up), vec![2, 2, 3]);

    let at = |arr: &MlxArray, e: i32, o: i32, k: i32| -> f32 {
        let scalar = mlxcel_core::slice(arr, &[e, o, k], &[e + 1, o + 1, k + 1]);
        mlxcel_core::item_f32(&scalar)
    };
    // Expert 0: fused rows 0..4 hold values 0..12 (3 per row).
    assert_eq!(at(&gate, 0, 0, 0), 0.0);
    assert_eq!(at(&gate, 0, 1, 2), 5.0);
    assert_eq!(at(&up, 0, 0, 0), 6.0);
    assert_eq!(at(&up, 0, 1, 2), 11.0);
    // Expert 1: fused rows hold values 12..24.
    assert_eq!(at(&gate, 1, 0, 0), 12.0);
    assert_eq!(at(&up, 1, 1, 2), 23.0);
}

#[test]
fn split_gate_up_tensor_rejects_bad_shapes() {
    let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let rank2 = mlxcel_core::from_slice_f32(&data, &[4, 3]);
    assert!(split_gate_up_tensor(&rank2, 2).is_err());

    let wrong_out = mlxcel_core::from_slice_f32(&data, &[2, 2, 3]);
    assert!(split_gate_up_tensor(&wrong_out, 2).is_err());
}

// -------------------------------------------------------------------------
// Dense-axis sliding-window mask (multi-token continuation at offset > 0)
// -------------------------------------------------------------------------

#[test]
fn dense_windowed_mask_is_correct_for_multi_token_offset_forward() {
    // 3 query tokens appended at offset 2000 with window 1024 over the FULL
    // dense key axis [0, 2003): query j (logical position 2000 + j) may
    // attend keys k with k <= 2000 + j and k > 976 + j.
    let mask = dense_windowed_causal_mask(3, 2000, 1024);
    mlxcel_core::eval(&mask);
    assert_eq!(mlxcel_core::array_shape(&mask), vec![3, 2003]);

    let at = |q: i32, k: i32| -> f32 {
        let scalar = mlxcel_core::slice(&mask, &[q, k], &[q + 1, k + 1]);
        mlxcel_core::item_f32(&scalar)
    };
    let blocked = |v: f32| v.is_infinite() && v < 0.0;

    // Row 0 (logical 2000): window lower edge between k = 976 and 977.
    assert!(blocked(at(0, 976)), "k 976 outside window for q 2000");
    assert_eq!(at(0, 977), 0.0, "k 977 inside window for q 2000");
    assert_eq!(at(0, 2000), 0.0, "self-attention allowed");
    assert!(blocked(at(0, 2001)), "future key blocked (causality)");

    // Row 2 (logical 2002): window shifts with the query position.
    assert!(blocked(at(2, 978)), "k 978 outside window for q 2002");
    assert_eq!(at(2, 979), 0.0, "k 979 inside window for q 2002");
    assert_eq!(at(2, 2002), 0.0, "self-attention allowed");
    assert!(blocked(at(1, 2002)), "row 1 cannot see row 2's key");
}

// -------------------------------------------------------------------------
// Real-model regression tests (issue #217; #[ignore], need the checkpoint)
// -------------------------------------------------------------------------

/// Determinism regression for the steel GEMM safe-load overlay
/// (`src/lib/mlx-cpp/patches/mlx/backend/metal/kernels/steel/gemm/mma.h`,
/// upstream MLX PRs #3560/#3565).
///
/// The DiffusionGemma canvas forward runs head_dim 256/512 SDPA through the
/// unfused steel-GEMM path with non-tile-aligned key lengths, so the broken
/// edge-tile loader read junk through strided views and the SAME 30-layer
/// lazy graph produced different bytes run to run. Both forward modes must
/// be byte-deterministic.
///
/// `cargo test --release --lib models::diffusion_gemma::tests::real_model_forward_determinism -- --ignored --nocapture`
#[test]
#[ignore]
fn real_model_forward_determinism() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models/diffusiongemma-26B-A4B-it-4bit");
    if !dir.exists() {
        eprintln!("skip: checkpoint not present");
        return;
    }
    let model = DiffusionGemmaModel::load(&dir).expect("load");
    // "Why is the sky blue?" through chat_template.jinja.
    let prompt: Vec<i32> = vec![
        2, 105, 2364, 107, 11355, 563, 506, 7217, 3730, 236881, 106, 107, 105, 4368, 107, 100,
        45518, 107, 101,
    ];
    let ids = mlxcel_core::from_slice_i32(&prompt, &[1, prompt.len() as i32]);

    // Warm the Metal buffer cache first: the historical failure modes (the
    // steel GEMM safe-load typo and the strided split-weight gather_qmm
    // reads) only turned nondeterministic once recycled buffers carried
    // changing junk, so a fresh-process run could pass with the bug present.
    {
        let warm_ids = super::generate::debug_canvas_pattern(64, 262144, 1);
        let warm = mlxcel_core::from_slice_i32(&warm_ids, &[1, 64]);
        for _ in 0..3 {
            let mut caches = model.make_diffusion_caches();
            let h = model.forward_encoder(&warm, &mut caches, None);
            mlxcel_core::eval(&h);
        }
    }
    let canvas_ids = super::generate::debug_canvas_pattern(64, 262144, 0);
    let canvas = mlxcel_core::from_slice_i32(&canvas_ids, &[1, 64]);

    let run = || -> (Vec<u8>, Vec<u8>) {
        let mut caches = model.make_diffusion_caches();
        let hidden = model.forward_encoder(&ids, &mut caches, None);
        let hidden = mlxcel_core::astype(&hidden, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&hidden);
        let encoder_bytes = mlxcel_core::array_to_raw_bytes(&hidden);

        let logits = model.forward_canvas(&canvas, &caches, None);
        let logits = mlxcel_core::astype(&logits, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&logits);
        let canvas_bytes = mlxcel_core::array_to_raw_bytes(&logits);
        (encoder_bytes, canvas_bytes)
    };

    let (encoder_a, canvas_a) = run();
    let (encoder_b, canvas_b) = run();
    assert_eq!(
        encoder_a, encoder_b,
        "encoder forward must be byte-deterministic across runs"
    );
    assert_eq!(
        canvas_a, canvas_b,
        "canvas forward must be byte-deterministic across runs"
    );
}

// -------------------------------------------------------------------------
// Vision config parsing (phase 2)
// -------------------------------------------------------------------------

#[test]
fn parses_vision_config_fields() {
    let args: ModelArgs = serde_json::from_str(REAL_CONFIG_SNIPPET).expect("config parses");
    assert_eq!(args.image_token_id(), 258_880);
    assert_eq!(args.boi_token_id(), 255_999);
    assert_eq!(args.eoi_token_id(), 258_882);
    // The chat checkpoint has no video token; the accessor must fall back to
    // -1 so the video branch is cleanly disabled (never matches a real id).
    assert_eq!(args.video_token_id(), -1);
    assert_eq!(args.vision_soft_tokens_per_image(), 280);

    let vision = args
        .vision_config()
        .expect("vision_config parses")
        .expect("vision_config present");
    assert_eq!(vision.hidden_size, 1152);
    assert_eq!(vision.num_hidden_layers, 27);
    assert_eq!(vision.patch_size, 16);
    assert_eq!(vision.pooling_kernel_size, 3);
    assert!(!vision.use_clipped_linears);
    assert!((vision.rms_norm_eps() - 1e-6).abs() < 1e-9);
}

#[test]
fn vision_config_absent_for_text_only_export() {
    // A text-only export drops the vision fields entirely.
    let mut value: serde_json::Value =
        serde_json::from_str(REAL_CONFIG_SNIPPET).expect("config parses");
    let obj = value.as_object_mut().expect("object");
    obj.remove("vision_config");
    obj.remove("image_token_id");
    let args: ModelArgs = serde_json::from_value(value).expect("text-only config parses");
    assert!(
        args.vision_config().expect("ok").is_none(),
        "no vision_config => None"
    );
    // The image token id falls back to the family default even when absent.
    assert_eq!(args.image_token_id(), 258_880);
}

// -------------------------------------------------------------------------
// Prompt image-token expansion (phase 2)
// -------------------------------------------------------------------------

const IMAGE_TOKEN: i32 = 258_880;
const BOI_TOKEN: i32 = 255_999;
const EOI_TOKEN: i32 = 258_882;

#[test]
fn expands_single_image_placeholder() {
    // BOS, <start_of_image> placeholder, then a text token.
    let mut tokens = vec![2, BOI_TOKEN, 9999];
    crate::vlm_runtime::expand_gemma4_image_tokens_pub(
        &mut tokens,
        IMAGE_TOKEN,
        BOI_TOKEN,
        EOI_TOKEN,
        &[4],
    )
    .expect("expansion succeeds");
    // The placeholder becomes boi + image*4 + eoi.
    let mut expected = vec![2, BOI_TOKEN];
    expected.extend(std::iter::repeat_n(IMAGE_TOKEN, 4));
    expected.push(EOI_TOKEN);
    expected.push(9999);
    assert_eq!(tokens, expected);
    assert_eq!(
        tokens.iter().filter(|&&t| t == IMAGE_TOKEN).count(),
        4,
        "exactly 4 image soft tokens"
    );
}

#[test]
fn expands_two_image_placeholders_into_separate_blocks() {
    let mut tokens = vec![2, BOI_TOKEN, 7, BOI_TOKEN, 8];
    crate::vlm_runtime::expand_gemma4_image_tokens_pub(
        &mut tokens,
        IMAGE_TOKEN,
        BOI_TOKEN,
        EOI_TOKEN,
        &[2, 3],
    )
    .expect("expansion succeeds");
    // Two contiguous blocks, the first with 2 soft tokens, the second with 3.
    let mut expected = vec![
        2,
        BOI_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        EOI_TOKEN,
        7,
        BOI_TOKEN,
    ];
    expected.extend(std::iter::repeat_n(IMAGE_TOKEN, 3));
    expected.push(EOI_TOKEN);
    expected.push(8);
    assert_eq!(tokens, expected);
    // boi count equals the number of blocks (2).
    assert_eq!(tokens.iter().filter(|&&t| t == BOI_TOKEN).count(), 2);
}

// -------------------------------------------------------------------------
// mm_token_type_ids + vision block ids (phase 2 overlay inputs)
// -------------------------------------------------------------------------

#[test]
fn derives_mm_token_type_ids_for_image_block() {
    use crate::vision::gemma4_unified_mask::{
        UnifiedTokenIds, derive_mm_token_type_ids, token_type,
    };
    let ids = UnifiedTokenIds {
        image: IMAGE_TOKEN,
        video: -1,
        audio: -1,
    };
    // text, boi(text), image, image, eoi(text), text
    let input = vec![2, BOI_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, EOI_TOKEN, 7];
    let types = derive_mm_token_type_ids(&input, ids);
    assert_eq!(
        types,
        vec![
            token_type::TEXT,
            token_type::TEXT,
            token_type::IMAGE,
            token_type::IMAGE,
            token_type::TEXT,
            token_type::TEXT,
        ]
    );
}

#[test]
fn computes_contiguous_vision_block_ids() {
    use crate::vision::gemma4_unified_mask::{UnifiedTokenIds, compute_vision_block_ids};
    let ids = UnifiedTokenIds {
        image: IMAGE_TOKEN,
        video: -1,
        audio: -1,
    };
    // Two separate image blocks: block 0 = positions 2..4, block 1 = 6..9.
    let input = vec![
        2,
        BOI_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        EOI_TOKEN,
        BOI_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        EOI_TOKEN,
        7,
    ];
    let block_ids = compute_vision_block_ids(&input, ids, true).expect("blocks present");
    assert_eq!(block_ids, vec![-1, -1, 0, 0, -1, -1, 1, 1, 1, -1, -1]);
}

// -------------------------------------------------------------------------
// Bidirectional vision-block overlay on the dense masks (phase 2)
// -------------------------------------------------------------------------

#[test]
fn overlay_makes_image_block_bidirectional_keeps_text_causal() {
    // 5-token prompt at offset 0: positions 1..3 are one image block, 0/3/4
    // are text. The overlay must let the image block attend to itself in
    // BOTH directions while text stays causal and the sliding window is
    // preserved.
    let block_ids = [-1i32, 0, 0, -1, -1];
    let l = block_ids.len() as i32;

    // Build the same causal base mask forward_encoder_embeds uses (offset 0).
    let base = mlxcel_core::utils::create_causal_mask(l, 0);
    let ids = mlxcel_core::from_slice_i32(&block_ids, &[l]);
    let overlaid = crate::models::gemma4::overlay_block_bidirectional(&base, &ids);
    mlxcel_core::eval(&overlaid);
    assert_eq!(mlxcel_core::array_shape(&overlaid), vec![l, l]);

    let at = |q: i32, k: i32| -> f32 {
        let scalar = mlxcel_core::slice(&overlaid, &[q, k], &[q + 1, k + 1]);
        mlxcel_core::item_f32(&scalar)
    };
    let blocked = |v: f32| v.is_infinite() && v < 0.0;

    // Image block (positions 1,2) is now bidirectional: 1 can see 2.
    assert_eq!(at(1, 2), 0.0, "image position 1 attends forward to 2");
    assert_eq!(at(2, 1), 0.0, "image position 2 attends back to 1");
    // Text positions stay strictly causal.
    assert!(blocked(at(0, 1)), "text 0 cannot see future token 1");
    assert!(blocked(at(3, 4)), "text 3 cannot see future token 4");
    // Cross text<->image pairs keep the causal base (text 3 may attend
    // back to image 1, but image 1 cannot attend forward to text 3).
    assert_eq!(at(3, 1), 0.0, "later text sees earlier image (causal)");
    assert!(
        blocked(at(1, 3)),
        "image cannot attend forward to later text"
    );
}

// -------------------------------------------------------------------------
// Sanitize keep/skip rules (phase 2 vision weights)
// -------------------------------------------------------------------------

#[test]
fn sanitize_keeps_text_and_vision_weights() {
    use crate::models::sanitize::keep_diffusion_gemma_weight;
    // Text backbone + encoder scalars are always kept.
    assert!(keep_diffusion_gemma_weight(
        "model.decoder.layers.0.self_attn.q_proj.weight",
        false
    ));
    assert!(keep_diffusion_gemma_weight(
        "model.encoder.language_model.layers.5.layer_scalar",
        false
    ));
    // Vision tower + embedder are kept (phase 2).
    assert!(keep_diffusion_gemma_weight(
        "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight",
        false
    ));
    assert!(keep_diffusion_gemma_weight(
        "model.encoder.embed_vision.embedding_projection.weight",
        false
    ));
    // Unrelated keys are dropped.
    assert!(!keep_diffusion_gemma_weight("lm_head.weight", false));
}

#[test]
fn sanitize_drops_clip_calibration_when_unclipped() {
    use crate::models::sanitize::keep_diffusion_gemma_weight;
    let clip_key = "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.input_max";
    // use_clipped_linears == false: drop the calibration tensors.
    assert!(!keep_diffusion_gemma_weight(clip_key, false));
    assert!(!keep_diffusion_gemma_weight(
        "model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.output_min",
        false
    ));
    // use_clipped_linears == true: keep them.
    assert!(keep_diffusion_gemma_weight(clip_key, true));
}

/// Real-model vision load smoke test (issue #217, phase 2): assert the vision
/// tower and embedder load alongside the text backbone. No forward pass.
///
/// `cargo test --release --lib models::diffusion_gemma::tests::real_model_loads_vision_front_end -- --ignored --nocapture`
#[test]
#[ignore]
fn real_model_loads_vision_front_end() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models/diffusiongemma-26B-A4B-it-4bit");
    if !dir.exists() {
        eprintln!("skip: checkpoint not present");
        return;
    }
    let model = DiffusionGemmaModel::load(&dir).expect("load");
    assert!(
        model.supports_images(),
        "the chat checkpoint ships a vision tower; vision must load"
    );
    let vision = model.vision().expect("vision front-end present");
    assert_eq!(vision.image_token_id, 258_880);
    assert_eq!(vision.boi_token_id, 255_999);
    assert_eq!(vision.eoi_token_id, 258_882);
    assert_eq!(vision.soft_tokens_per_image, 280);
}
