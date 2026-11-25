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

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::sampling::LogprobsConfig;
use mlxcel_core::speculative::mtp::target::MtpTarget;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::json;

use super::{InklingConfig, InklingModel};

fn matrix(rows: i32, columns: i32, seed: usize) -> UniquePtr<MlxArray> {
    let values: Vec<f32> = (0..rows as usize * columns as usize)
        .map(|index| (((index + seed) % 17) as f32 - 8.0) / 50.0)
        .collect();
    mlxcel_core::from_slice_f32(&values, &[rows, columns])
}

fn conv(channels: i32, kernel: i32, seed: usize) -> UniquePtr<MlxArray> {
    let values: Vec<f32> = (0..channels as usize * kernel as usize)
        .map(|index| (((index + seed) % 7) as f32 - 3.0) / 40.0)
        .collect();
    mlxcel_core::from_slice_f32(&values, &[channels, kernel, 1])
}

pub(crate) fn tiny_model() -> InklingModel {
    tiny_model_impl(false, 4, 2, 8)
}

pub(crate) fn tiny_audio_model() -> InklingModel {
    tiny_model_impl(true, 4, 2, 8)
}

/// The tiny model at the attention geometry real checkpoints run: MLX's
/// fused SDPA does not take a 2-wide head and falls back to an unfused path
/// whose arithmetic depends on the block length, which no checkpoint takes
/// (the Inkling 2 tiny model needed the same, issue #3280).
pub(crate) fn tiny_model_with_real_heads() -> InklingModel {
    tiny_model_impl(false, 128, 64, 32)
}

#[allow(non_snake_case)]
fn tiny_model_impl(with_audio: bool, HIDDEN: i32, HEAD_DIM: i32, VOCAB: i32) -> InklingModel {
    const HEADS: i32 = 2;
    const KV_HEADS: i32 = 1;
    const D_REL: i32 = 2;
    const DENSE: i32 = 6;
    const KERNEL: i32 = 2;
    let mut config = json!({
        "vocab_size": VOCAB,
        "text_config": {
            "hidden_size": HIDDEN,
            "num_hidden_layers": 2,
            "vocab_size": VOCAB,
            "unpadded_vocab_size": VOCAB,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "swa_num_attention_heads": HEADS,
            "swa_num_key_value_heads": KV_HEADS,
            "swa_head_dim": HEAD_DIM,
            "sliding_window_size": 3,
            "layer_types": ["hybrid_sliding", "hybrid"],
            "d_rel": D_REL,
            "rel_extent": 4,
            "log_scaling_n_floor": 2,
            "log_scaling_alpha": 0.1,
            "sconv_kernel_size": KERNEL,
            "dense_mlp_idx": 2,
            "mlp_layer_types": ["dense", "dense"],
            "intermediate_size": 2,
            "dense_intermediate_size": DENSE,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1
        }
    });
    if with_audio {
        config["audio_token_id"] = json!(6);
        config["audio_config"] = json!({
            "model_type": "inkling_audio",
            "n_mel_bins": 80,
            "mel_vocab_size": 16,
            "text_hidden_size": HIDDEN,
            "max_frames_per_chunk": 2
        });
    }
    let config: InklingConfig = serde_json::from_value(config).unwrap();
    let mut weights = WeightMap::new();
    weights.insert("model.embed_tokens.weight".into(), matrix(VOCAB, HIDDEN, 1));
    weights.insert(
        "model.embed_norm.weight".into(),
        mlxcel_core::ones(&[HIDDEN], dtype::FLOAT32),
    );
    weights.insert(
        "model.norm.weight".into(),
        mlxcel_core::ones(&[HIDDEN], dtype::FLOAT32),
    );
    weights.insert("lm_head.weight".into(), matrix(VOCAB, HIDDEN, 5));
    if with_audio {
        weights.insert(
            "audio_tower.embed_audio_tokens.weight".into(),
            matrix(80 * 16, HIDDEN, 13),
        );
        weights.insert(
            "audio_tower.norm.weight".into(),
            mlxcel_core::ones(&[HIDDEN], dtype::FLOAT32),
        );
    }
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            weights.insert(
                format!("{prefix}.{name}"),
                mlxcel_core::ones(&[HIDDEN], dtype::FLOAT32),
            );
        }
        weights.insert(
            format!("{prefix}.attn_sconv.conv.weight"),
            conv(HIDDEN, KERNEL, layer + 2),
        );
        weights.insert(
            format!("{prefix}.mlp_sconv.conv.weight"),
            conv(HIDDEN, KERNEL, layer + 4),
        );
        let attn = format!("{prefix}.self_attn");
        weights.insert(
            format!("{attn}.q_proj.weight"),
            matrix(HEADS * HEAD_DIM, HIDDEN, layer + 1),
        );
        weights.insert(
            format!("{attn}.k_proj.weight"),
            matrix(KV_HEADS * HEAD_DIM, HIDDEN, layer + 2),
        );
        weights.insert(
            format!("{attn}.v_proj.weight"),
            matrix(KV_HEADS * HEAD_DIM, HIDDEN, layer + 3),
        );
        weights.insert(
            format!("{attn}.r_proj.weight"),
            matrix(HEADS * D_REL, HIDDEN, layer + 4),
        );
        weights.insert(
            format!("{attn}.o_proj.weight"),
            matrix(HIDDEN, HEADS * HEAD_DIM, layer + 5),
        );
        weights.insert(
            format!("{attn}.q_norm.weight"),
            mlxcel_core::ones(&[HEAD_DIM], dtype::FLOAT32),
        );
        weights.insert(
            format!("{attn}.k_norm.weight"),
            mlxcel_core::ones(&[HEAD_DIM], dtype::FLOAT32),
        );
        weights.insert(
            format!("{attn}.k_sconv.conv.weight"),
            conv(KV_HEADS * HEAD_DIM, KERNEL, layer + 6),
        );
        weights.insert(
            format!("{attn}.v_sconv.conv.weight"),
            conv(KV_HEADS * HEAD_DIM, KERNEL, layer + 7),
        );
        let extent = if layer == 0 { 3 } else { 4 };
        weights.insert(format!("{attn}.rel_proj"), matrix(D_REL, extent, layer + 8));
        let mlp = format!("{prefix}.mlp");
        weights.insert(
            format!("{mlp}.gate_proj.weight"),
            matrix(DENSE, HIDDEN, layer + 9),
        );
        weights.insert(
            format!("{mlp}.up_proj.weight"),
            matrix(DENSE, HIDDEN, layer + 10),
        );
        weights.insert(
            format!("{mlp}.down_proj.weight"),
            matrix(HIDDEN, DENSE, layer + 11),
        );
        weights.insert(
            format!("{mlp}.global_scale"),
            mlxcel_core::ones(&[1], dtype::FLOAT32),
        );
    }
    InklingModel::from_weights(config, weights).unwrap()
}

fn forward(model: &InklingModel, tokens: &[i32]) -> UniquePtr<MlxArray> {
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    let mut caches = model.make_caches();
    model.forward(&input, &mut caches, None)
}

#[test]
fn tiny_hybrid_prefill_is_causal_and_matches_incremental_decode() {
    let tokens = [1, 2, 3, 4, 5];
    let full = forward(&tiny_model(), &tokens);

    let changed = forward(&tiny_model(), &[1, 2, 6, 7, 0]);
    let prefix_full = mlxcel_core::utils::slice_axis(&full, 1, 0, 2);
    let prefix_changed = mlxcel_core::utils::slice_axis(&changed, 1, 0, 2);
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &prefix_full,
        &prefix_changed,
        1e-5,
        1e-5,
    )));

    let incremental_model = tiny_model();
    let mut caches = incremental_model.make_caches();
    let mut incremental: Option<UniquePtr<MlxArray>> = None;
    for token in tokens {
        let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        let output = incremental_model.forward(&input, &mut caches, None);
        incremental = Some(match incremental {
            Some(previous) => mlxcel_core::concatenate(&previous, &output, 1),
            None => output,
        });
    }
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full,
        incremental.as_deref().unwrap(),
        2e-4,
        2e-4,
    )));
}

#[test]
fn mtp_verify_argmax_matches_the_incremental_target_chain() {
    let prompt = [1, 2, 3];
    let verify_input = [4, 5, 6];
    let model = tiny_model();
    let _ = model.make_caches();
    let adapter = crate::models::inkling_mtp_target::InklingMtpTargetAdapter::new(&model, None);
    let sampler = mlxcel_core::generate::SamplingConfig::greedy();
    let logprobs = LogprobsConfig::default();
    let _ = adapter.prefill_and_seed(&prompt, &sampler, &prompt, &logprobs);
    let verify = adapter.verify_forward(&verify_input, &sampler, &logprobs);

    let classic = tiny_model();
    let _ = classic.make_caches();
    let prompt_arr = mlxcel_core::from_slice_i32(&prompt, &[1, prompt.len() as i32]);
    let _ = classic.forward_prefill_with_hidden_for_sequence(&prompt_arr, None);
    let mut expected = Vec::new();
    for token in verify_input {
        let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        let (logits, _) = classic.forward_prefill_with_hidden_for_sequence(&input, None);
        let argmax = mlxcel_core::argmax(&logits, -1, false);
        mlxcel_core::eval(&argmax);
        expected.push(mlxcel_core::item_i32(&argmax));
    }
    assert_eq!(verify.target_tokens, expected);
}

/// Without the row-invariant products a verify block keeps a pre-verify
/// snapshot, and a partial accept restores it and replays the kept prefix.
#[test]
fn mtp_partial_accept_restores_then_replays_kv_and_conv_state_exactly() {
    // Exact comparison of two computations: both must run on one default
    // device while sibling tests move it (#1622).
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let prompt = [1, 2, 3];
    let verify_input = [4, 5, 6];
    let accepted = 1;
    let mut model = tiny_model();
    model.set_row_invariant_for_test(false);
    let _ = model.make_caches();
    let adapter = crate::models::inkling_mtp_target::InklingMtpTargetAdapter::new(&model, None);
    let sampler = mlxcel_core::generate::SamplingConfig::greedy();
    let logprobs = LogprobsConfig::default();
    let _ = adapter.prefill_and_seed(&prompt, &sampler, &prompt, &logprobs);
    let verify = adapter.verify_forward(&verify_input, &sampler, &logprobs);
    let finalized = adapter.verify_finalize(accepted, verify_input.len(), verify.captured);
    assert_eq!(finalized.kv_offset, prompt.len() + accepted + 1);

    let mut reference = tiny_model();
    reference.set_row_invariant_for_test(false);
    let _ = reference.make_caches();
    let prompt_arr = mlxcel_core::from_slice_i32(&prompt, &[1, prompt.len() as i32]);
    let _ = reference.forward_prefill_with_hidden_for_sequence(&prompt_arr, None);
    let kept = mlxcel_core::from_slice_i32(&verify_input[..=accepted], &[1, accepted as i32 + 1]);
    let _ = reference.forward_prefill_with_hidden_for_sequence(&kept, None);

    let next = mlxcel_core::from_slice_i32(&[7], &[1, 1]);
    let (actual_logits, actual_hidden) =
        model.forward_prefill_with_hidden_for_sequence(&next, None);
    let (expected_logits, expected_hidden) =
        reference.forward_prefill_with_hidden_for_sequence(&next, None);
    let logits_equal = mlxcel_core::allclose(&actual_logits, &expected_logits, 0.0, 0.0);
    let hidden_equal = mlxcel_core::allclose(&actual_hidden, &expected_hidden, 0.0, 0.0);
    mlxcel_core::eval(&logits_equal);
    mlxcel_core::eval(&hidden_equal);
    assert!(mlxcel_core::item_bool(&logits_equal));
    assert!(mlxcel_core::item_bool(&hidden_equal));
}

/// Issue #3280: a partial accept keeps the verify block's own state in
/// place (KV tail trim, convolution windows from the block), so the next
/// step equals classic decode, which fed every kept token alone. The replay
/// it replaces ran the kept prefix as one forward outside the verify scope,
/// whose products and convolution are not the single-token step's, and was
/// a second target forward per round. Widths 3 and 10 (the Inkling-Small
/// MTP default, past the 8 queries MLX's SDPA runs per query).
#[test]
fn mtp_partial_accept_keeps_the_single_token_chain_state_on_metal() {
    if !mlxcel_core::row_invariant::available() {
        eprintln!("skipping: the row-invariant kernel needs the Metal device");
        return;
    }
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let sampler = mlxcel_core::generate::SamplingConfig::greedy();
    let logprobs = LogprobsConfig::default();
    let prompt: Vec<i32> = (0..8).map(|i| (i * 5 + 1) % 32).collect();
    let step = |model: &InklingModel, tokens: &[i32]| {
        let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
        model.forward_prefill_with_hidden_for_sequence(&input, None)
    };
    for width in [3_usize, 10] {
        let verify_input: Vec<i32> = (0..width as i32).map(|i| (i * 7 + 3) % 32).collect();
        for accepted in 0..width {
            let model = tiny_model_with_real_heads();
            let adapter =
                crate::models::inkling_mtp_target::InklingMtpTargetAdapter::new(&model, None);
            let _ = adapter.prefill_and_seed(&prompt, &sampler, &prompt, &logprobs);
            let verify = adapter.verify_forward(&verify_input, &sampler, &logprobs);
            let finalized = adapter.verify_finalize(accepted, width, verify.captured);
            assert_eq!(finalized.kv_offset, prompt.len() + accepted + 1);

            let chain = tiny_model_with_real_heads();
            let _ = step(&chain, &prompt);
            for token in &verify_input[..=accepted] {
                let _ = step(&chain, &[*token]);
            }
            for next in [9, 17] {
                let (actual_logits, actual_hidden) = step(&model, &[next]);
                let (expected_logits, expected_hidden) = step(&chain, &[next]);
                let context = format!("width {width}, accepted {accepted}, next {next}");
                assert_eq!(
                    crate::models::speculative_exactness_draws::position_bytes(&actual_logits, 0),
                    crate::models::speculative_exactness_draws::position_bytes(&expected_logits, 0),
                    "logits, {context}"
                );
                assert_eq!(
                    crate::models::speculative_exactness_draws::position_bytes(&actual_hidden, 0),
                    crate::models::speculative_exactness_draws::position_bytes(&expected_hidden, 0),
                    "hidden, {context}"
                );
            }
        }
    }
}

#[test]
fn model_owned_batched_multistep() {
    crate::models::batched_decode_harness::assert_model_owned_multistep(tiny_model);
}

#[test]
fn batched_decode_initializes_absent_convolution_tails() {
    use mlxcel_core::cache::SequenceId;
    let model = tiny_model();
    let ids = [SequenceId::from_raw(71), SequenceId::from_raw(72)];
    for id in ids {
        model.prepare_sequence_state(id);
    }
    let mut caches: Vec<_> = ids.iter().map(|_| model.make_caches()).collect();
    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let _ = model.forward_with_sequence_id(&prompt, Some(ids[0]), &mut caches[0], None);
    let input = mlxcel_core::from_slice_i32(&[4, 5], &[2, 1]);
    let mut rows: Vec<_> = caches.iter_mut().map(Vec::as_mut_slice).collect();
    let actual = model.forward_batched_with_ids(&input, Some(&ids), &mut rows, None);
    for (row, token) in [4, 5].into_iter().enumerate() {
        let solo = tiny_model();
        let mut cache = solo.make_caches();
        if row == 0 {
            let _ = solo.forward(&prompt, &mut cache, None);
        }
        let token = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        let expected = solo.forward(&token, &mut cache, None);
        let actual_row = mlxcel_core::utils::slice_axis(&actual, 0, row as i32, row as i32 + 1);
        let difference = mlxcel_core::subtract(&actual_row, &expected);
        let values = mlxcel_core::utils::array_to_vec_f32(&difference);
        assert!(values.iter().all(|v| v.abs() <= 1e-5));
    }
}

/// The Inkling block-vs-chain exactness probe runs on a real model graph and
/// refuses widths that draft nothing (issue #1549).
#[test]
fn exactness_probe_runs_on_the_tiny_model() {
    let model = tiny_model();
    let verdict = model.probe_block_chain_exactness(3);
    assert!(
        !matches!(
            verdict,
            crate::models::speculative_exactness::BlockChainExactness::NotRun(_)
        ),
        "a width-3 probe must compare both arms, got {verdict:?}"
    );
    assert!(matches!(
        model.probe_block_chain_exactness(1),
        crate::models::speculative_exactness::BlockChainExactness::NotRun(_)
    ));
}

/// Issue #3409: with the row-invariant products a verify block of the tiny
/// model equals the single-token chain at every surveyed width, short draws
/// and the long draw past the 3-token sliding window alike. Without them
/// (the code before this change) it differs from layer 0 on.
#[test]
fn the_verify_block_matches_the_chain_on_metal() {
    use crate::models::speculative_exactness::BlockChainExactness;
    if !mlxcel_core::row_invariant::available() {
        eprintln!("skipping: the row-invariant kernel needs the Metal device");
        return;
    }
    let model = tiny_model_with_real_heads();
    for width in [2, 4, 8, 10, 16] {
        assert_eq!(
            model.probe_block_chain_exactness(width),
            BlockChainExactness::Equal,
            "width {width}"
        );
        assert_eq!(
            model.probe_block_chain_exactness_at(width, &[40]),
            BlockChainExactness::Equal,
            "width {width}, 40-token prompt"
        );
    }
    let mut plain = tiny_model_with_real_heads();
    plain.set_row_invariant_for_test(false);
    let before = plain.probe_block_chain_exactness(8);
    assert!(
        matches!(&before, BlockChainExactness::Localized { location, .. }
            if location.contains("layer 0")),
        "without the row-invariant products the block must differ, got {before:?}"
    );
}

/// Issue #3409 on a real checkpoint: without the row-invariant products the
/// probe declines Inkling at every width (the state before the change) and
/// names layer 0; with them the block is byte-identical, also after prompts
/// past the 512-token sliding window and the 2048-token prefill chunk, and
/// at width 10, the Inkling-Small MTP default (issue #3280; passes on
/// `models/inkling-small-mlx-4bit` too).
/// Ignored; run under the GPU lock. `INKLING_PROBE_MODEL` picks the
/// checkpoint (default `models/inkling-0.6b-a0.6b`).
#[test]
#[ignore = "needs a real Inkling checkpoint and the GPU lock"]
fn real_checkpoint_probe_fails_without_row_invariant_and_passes_with_it() {
    use crate::models::speculative_exactness::BlockChainExactness;
    let path =
        std::env::var("INKLING_PROBE_MODEL").unwrap_or_else(|_| "models/inkling-0.6b-a0.6b".into());
    if !std::path::Path::new(&path).join("config.json").exists() {
        eprintln!("skipping: {path} is not present");
        return;
    }
    let (mut model, _) = InklingModel::load(&path).unwrap();
    for width in [2, 4, 8, 10] {
        model.set_row_invariant_for_test(false);
        let before = model.probe_block_chain_exactness(width);
        assert!(
            matches!(&before, BlockChainExactness::Localized { location, .. }
                if location.contains("layer 0")),
            "width {width} without row_invariant: {before:?}"
        );
        model.set_row_invariant_for_test(true);
        assert_eq!(
            model.probe_block_chain_exactness(width),
            BlockChainExactness::Equal,
            "width {width} with row_invariant"
        );
        assert_eq!(
            model.probe_block_chain_exactness_at(width, &[600, 2100]),
            BlockChainExactness::Equal,
            "width {width} with row_invariant, long prompts"
        );
    }
}

/// Issue #3422: prompt lookup takes Inkling only with the row kernel on.
#[test]
fn prompt_lookup_gate_follows_the_row_kernel() {
    use mlxcel_core::speculative::prompt_lookup::prompt_lookup_unsupported_reason as reason;
    let mut model = tiny_model();
    model.set_row_invariant_for_test(true);
    assert_eq!(reason(&model), None);
    model.set_row_invariant_for_test(false);
    assert!(reason(&model).is_some());
}
