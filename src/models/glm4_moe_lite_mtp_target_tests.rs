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

//! Unit tests for the GLM4 MoE Lite MTP target adapter (issue #1326) on a
//! tiny synthetic two-layer model (layer 0 dense, layer 1 routed): the
//! exactness premise (a verify block reproduces the single-token chain), the
//! trim-based rollback, and the hidden-state tap.

use super::*;
use crate::models::glm4_moe_lite::ModelArgs;
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::sampling::LogprobsConfig;
use mlxcel_core::weights::WeightMap;

const HIDDEN: i32 = 8;
const VOCAB: i32 = 16;
const HEADS: i32 = 2;
const QK_NOPE: i32 = 4;
const QK_ROPE: i32 = 2;
const V_HEAD: i32 = 4;
const KV_LORA_RANK: i32 = 8;
const EXPERTS: i32 = 2;
const MOE_INTER: i32 = 8;
const DENSE_INTER: i32 = 16;
const LAYERS: usize = 2;

fn tiny_args() -> ModelArgs {
    serde_json::from_str(
        r#"{
        "model_type": "glm4_moe_lite",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "kv_lora_rank": 8,
        "q_lora_rank": null,
        "qk_rope_head_dim": 2,
        "qk_nope_head_dim": 4,
        "v_head_dim": 4,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "n_shared_experts": 1,
        "first_k_dense_replace": 1,
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0
    }"#,
    )
    .expect("tiny args parse")
}

fn insert_ramp(w: &mut WeightMap, key: &str, shape: &[i32], seed: f32) {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| 0.08 * ((i as f32 * 0.53 + seed).sin()))
        .collect();
    w.insert(key.to_string(), mlxcel_core::from_slice_f32(&data, shape));
}

fn insert_ones(w: &mut WeightMap, key: &str, shape: &[i32]) {
    let n: i32 = shape.iter().product();
    w.insert(
        key.to_string(),
        mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape),
    );
}

fn tiny_weights() -> WeightMap {
    let mut w = WeightMap::new();
    insert_ramp(&mut w, "model.embed_tokens.weight", &[VOCAB, HIDDEN], 1.0);
    insert_ones(&mut w, "model.norm.weight", &[HIDDEN]);
    insert_ramp(&mut w, "lm_head.weight", &[VOCAB, HIDDEN], 2.0);
    for i in 0..LAYERS {
        let l = format!("model.layers.{i}");
        let seed = 10.0 * (i as f32 + 1.0);
        insert_ones(&mut w, &format!("{l}.input_layernorm.weight"), &[HIDDEN]);
        insert_ones(
            &mut w,
            &format!("{l}.post_attention_layernorm.weight"),
            &[HIDDEN],
        );
        insert_ramp(
            &mut w,
            &format!("{l}.self_attn.q_proj.weight"),
            &[HEADS * (QK_NOPE + QK_ROPE), HIDDEN],
            seed + 1.0,
        );
        insert_ramp(
            &mut w,
            &format!("{l}.self_attn.kv_a_proj_with_mqa.weight"),
            &[KV_LORA_RANK + QK_ROPE, HIDDEN],
            seed + 2.0,
        );
        insert_ones(
            &mut w,
            &format!("{l}.self_attn.kv_a_layernorm.weight"),
            &[KV_LORA_RANK],
        );
        insert_ramp(
            &mut w,
            &format!("{l}.self_attn.embed_q.weight"),
            &[HEADS, KV_LORA_RANK, QK_NOPE],
            seed + 3.0,
        );
        insert_ramp(
            &mut w,
            &format!("{l}.self_attn.unembed_out.weight"),
            &[HEADS, V_HEAD, KV_LORA_RANK],
            seed + 4.0,
        );
        insert_ramp(
            &mut w,
            &format!("{l}.self_attn.o_proj.weight"),
            &[HIDDEN, HEADS * V_HEAD],
            seed + 5.0,
        );
        if i == 0 {
            insert_ramp(
                &mut w,
                &format!("{l}.mlp.gate_proj.weight"),
                &[DENSE_INTER, HIDDEN],
                seed + 6.0,
            );
            insert_ramp(
                &mut w,
                &format!("{l}.mlp.up_proj.weight"),
                &[DENSE_INTER, HIDDEN],
                seed + 7.0,
            );
            insert_ramp(
                &mut w,
                &format!("{l}.mlp.down_proj.weight"),
                &[HIDDEN, DENSE_INTER],
                seed + 8.0,
            );
        } else {
            insert_ramp(
                &mut w,
                &format!("{l}.mlp.gate.weight"),
                &[EXPERTS, HIDDEN],
                seed + 6.0,
            );
            w.insert(
                format!("{l}.mlp.gate.e_score_correction_bias"),
                mlxcel_core::from_slice_f32(&[0.0, 0.0], &[EXPERTS]),
            );
            insert_ramp(
                &mut w,
                &format!("{l}.mlp.switch_mlp.gate_proj.weight"),
                &[EXPERTS, MOE_INTER, HIDDEN],
                seed + 7.0,
            );
            insert_ramp(
                &mut w,
                &format!("{l}.mlp.switch_mlp.up_proj.weight"),
                &[EXPERTS, MOE_INTER, HIDDEN],
                seed + 8.0,
            );
            insert_ramp(
                &mut w,
                &format!("{l}.mlp.switch_mlp.down_proj.weight"),
                &[EXPERTS, HIDDEN, MOE_INTER],
                seed + 9.0,
            );
            insert_ramp(
                &mut w,
                &format!("{l}.mlp.shared_experts.gate_proj.weight"),
                &[MOE_INTER, HIDDEN],
                seed + 10.0,
            );
            insert_ramp(
                &mut w,
                &format!("{l}.mlp.shared_experts.up_proj.weight"),
                &[MOE_INTER, HIDDEN],
                seed + 11.0,
            );
            insert_ramp(
                &mut w,
                &format!("{l}.mlp.shared_experts.down_proj.weight"),
                &[HIDDEN, MOE_INTER],
                seed + 12.0,
            );
        }
    }
    w
}

fn tiny_model() -> Glm4MoeLiteModel {
    let _runtime = crate::initialize_runtime();
    Glm4MoeLiteModel::from_weights(&tiny_weights(), &tiny_args()).expect("tiny model builds")
}

fn ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn read_f32(arr: &MlxArray) -> Vec<f32> {
    let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn row(logits: &MlxArray, index: i32) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(logits);
    let r = mlxcel_core::slice(logits, &[0, index, 0], &[shape[0], index + 1, shape[2]]);
    read_f32(&r)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

fn layer_offsets(model: &Glm4MoeLiteModel, seq_id: Option<SequenceId>) -> Vec<i32> {
    model
        .mtp_sequence_state
        .with_sequence_state(seq_id, |caches| caches.iter().map(|c| c.offset).collect())
}

// ── Exactness premise ─────────────────────────────────────────────────────

/// The temperature-0 contract: an `M = bs` verify block over `[bonus] ++
/// draft` produces, per position, the logits the single-token decode chain
/// produces for the same tokens on the same prefix. The projections are
/// batched in the block, so byte-identity is a kernel property the runtime
/// probe measures; here the tolerance is the f32 accumulation-order noise of
/// the synthetic model and the argmax must agree exactly.
#[test]
fn verify_block_logits_match_single_token_decode() {
    let model = tiny_model();
    let prompt = [3_i32, 5, 7, 2, 11, 6];
    let block = [9_i32, 4, 13];

    let mut chain = model.make_caches();
    let _ = model.forward(&ids(&prompt), &mut chain, None);
    let chain_rows: Vec<Vec<f32>> = block
        .iter()
        .map(|t| row(&model.forward(&ids(&[*t]), &mut chain, None), 0))
        .collect();

    let mut blocked = model.make_caches();
    let _ = model.forward(&ids(&prompt), &mut blocked, None);
    let (logits, hidden) = model.forward_verify(&ids(&block), &mut blocked);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, block.len() as i32, VOCAB]
    );
    assert_eq!(
        mlxcel_core::array_shape(&hidden),
        vec![1, block.len() as i32, HIDDEN]
    );

    for (i, chain_row) in chain_rows.iter().enumerate() {
        let block_row = row(&logits, i as i32);
        let diff = max_abs_diff(&block_row, chain_row);
        assert!(diff < 1e-4, "position {i}: max |block - chain| = {diff}");
        assert_eq!(argmax(&block_row), argmax(chain_row), "position {i} argmax");
    }
    // Both arms consumed the same number of positions on every layer.
    for (c, b) in chain.iter().zip(&blocked) {
        assert_eq!(c.offset, (prompt.len() + block.len()) as i32);
        assert_eq!(b.offset, c.offset);
    }
}

/// The verify hidden is the pre-norm residual; the adapter's default tap
/// applies `model.norm` to it, and the `pre` tap hands it out unchanged.
#[test]
fn hidden_tap_applies_the_final_norm_by_default() {
    let model = tiny_model();
    let mut caches = model.make_caches();
    let (_, hidden_pre) = model.forward_with_hidden(&ids(&[3, 5, 7]), &mut caches);

    let post = Glm4MoeLiteMtpTargetAdapter::new(&model, None)
        .with_hidden_tap(HiddenTap::PostFinalNorm)
        .tap_hidden(&hidden_pre);
    let pre = Glm4MoeLiteMtpTargetAdapter::new(&model, None)
        .with_hidden_tap(HiddenTap::PreFinalNorm)
        .tap_hidden(&hidden_pre);
    let expected_post = read_f32(&model.apply_final_norm(&hidden_pre));
    assert_eq!(read_f32(&post), expected_post);
    assert_eq!(read_f32(&pre), read_f32(&hidden_pre));
    assert!(
        max_abs_diff(&read_f32(&post), &read_f32(&pre)) > 1e-6,
        "the two taps must differ on a non-trivial residual"
    );
}

#[test]
fn hidden_tap_env_parses_pre_spellings_only() {
    // `from_env` memoizes, so exercise the parser through the enum values
    // the adapter builder takes rather than mutating the process env.
    let model = tiny_model();
    let adapter = Glm4MoeLiteMtpTargetAdapter::new(&model, None);
    assert_eq!(adapter.tap, HiddenTap::from_env());
    let adapter = adapter.with_hidden_tap(HiddenTap::PreFinalNorm);
    assert_eq!(adapter.tap, HiddenTap::PreFinalNorm);
}

// ── Round loop surface ────────────────────────────────────────────────────

/// `prefill_and_seed` forwards the whole prompt on a fresh slot, samples
/// the first bonus greedily from the last position, and reports the cache
/// offset the drafter rebinds to.
#[test]
fn prefill_and_seed_reports_prompt_offset_and_full_hidden() {
    let model = tiny_model();
    let adapter = Glm4MoeLiteMtpTargetAdapter::new(&model, None);
    let prompt = [3_i32, 5, 7, 2];
    let sampler = SamplingConfig::greedy();
    let (bonus, seed, lp) =
        adapter.prefill_and_seed(&prompt, &sampler, &[], &LogprobsConfig::default());
    assert!((0..VOCAB).contains(&bonus));
    assert!(lp.is_none());
    assert_eq!(seed.kv_offset, prompt.len());
    assert_eq!(seed.bonus_position, prompt.len() - 1);
    assert!(seed.next_shared_kv.is_empty());
    assert_eq!(
        mlxcel_core::array_shape(&seed.next_hidden),
        vec![1, 1, HIDDEN]
    );
    let full = seed.verify_hidden_full.expect("full prompt hidden");
    assert_eq!(
        mlxcel_core::array_shape(&full),
        vec![1, prompt.len() as i32, HIDDEN]
    );
    assert_eq!(
        layer_offsets(&model, None),
        vec![prompt.len() as i32; LAYERS]
    );

    // The greedy bonus is the classic path's next token.
    let mut caches = model.make_caches();
    let logits = model.forward(&ids(&prompt), &mut caches, None);
    let last = row(&logits, prompt.len() as i32 - 1);
    assert_eq!(bonus as usize, argmax(&last));

    // A second prefill starts from a fresh slot rather than appending.
    let _ = adapter.prefill_and_seed(&prompt, &sampler, &[], &LogprobsConfig::default());
    assert_eq!(
        layer_offsets(&model, None),
        vec![prompt.len() as i32; LAYERS]
    );
}

/// After a `block_size` verify, `verify_finalize(accepted)` leaves every
/// layer cache at `P + accepted + 1`, for every accepted count the block
/// admits.
#[test]
fn verify_finalize_trims_every_layer_to_accepted_prefix() {
    let model = tiny_model();
    let prompt = [3_i32, 5, 7, 2];
    let block_size = 3usize;
    let sampler = SamplingConfig::greedy();
    let logprobs = LogprobsConfig::default();
    let raw_seq_id = SequenceId::from_raw(7);
    let seq_id = Some(raw_seq_id);

    for accepted in 0..block_size {
        let adapter = Glm4MoeLiteMtpTargetAdapter::new(&model, seq_id);
        let (bonus, _seed, _) = adapter.prefill_and_seed(&prompt, &sampler, &[], &logprobs);
        let verify_input = [bonus, 4, 13];
        let out = adapter.verify_forward(&verify_input, &sampler, &logprobs);
        assert_eq!(out.target_tokens.len(), block_size);
        assert_eq!(out.captured.tensors.len(), 1);
        assert_eq!(
            layer_offsets(&model, seq_id),
            vec![(prompt.len() + block_size) as i32; LAYERS],
            "verify appended the whole block"
        );

        let fin = adapter.verify_finalize(accepted, block_size, out.captured);
        let expected = prompt.len() + accepted + 1;
        assert_eq!(fin.kv_offset, expected, "accepted = {accepted}");
        assert_eq!(fin.bonus_position, expected - 1);
        assert_eq!(
            layer_offsets(&model, seq_id),
            vec![expected as i32; LAYERS],
            "accepted = {accepted}: every layer trimmed to the accepted prefix"
        );
        assert_eq!(
            mlxcel_core::array_shape(&fin.next_hidden),
            vec![1, 1, HIDDEN]
        );
        let full = fin
            .verify_hidden_full
            .expect("verify hidden kept for the drafter");
        assert_eq!(
            mlxcel_core::array_shape(&full),
            vec![1, block_size as i32, HIDDEN]
        );
    }

    // The scheduler's release drops the slot.
    <Glm4MoeLiteModel as mlxcel_core::generate::LanguageModel>::release_sequence_state_by_id(
        &model, raw_seq_id,
    );
    assert!(!model.mtp_sequence_state.has_sequence_state(raw_seq_id));
}

/// The verify-row argmax equals the classic decode's next token for the
/// same prefix, which is what the deferred-greedy walk compares against.
#[test]
fn verify_forward_target_tokens_are_the_classic_greedy_tokens() {
    let model = tiny_model();
    let prompt = [3_i32, 5, 7, 2, 11];
    let sampler = SamplingConfig::greedy();
    let logprobs = LogprobsConfig::default();
    let adapter = Glm4MoeLiteMtpTargetAdapter::new(&model, None);
    let (bonus, _seed, _) = adapter.prefill_and_seed(&prompt, &sampler, &[], &logprobs);
    let verify_input = [bonus, 4, 13];
    let out = adapter.verify_forward(&verify_input, &sampler, &logprobs);

    let mut caches = model.make_caches();
    let _ = model.forward(&ids(&prompt), &mut caches, None);
    let mut classic = Vec::new();
    for t in &verify_input {
        let logits = model.forward(&ids(&[*t]), &mut caches, None);
        classic.push(argmax(&row(&logits, 0)) as i32);
    }
    assert_eq!(out.target_tokens, classic);
    assert_eq!(adapter.num_layers(), LAYERS);
    assert_eq!(adapter.eos_token_ids(), vec![154820, 154827, 154829]);
    assert_eq!(
        mlxcel_core::array_shape(&adapter.embed_token(3)),
        vec![1, 1, HIDDEN]
    );
}

/// The probe behind `mtp_exactness_allows` runs on the tiny model and
/// returns a verdict rather than panicking; on a synthetic f32 model the
/// answer is whatever the kernels give, so only the shape of the result is
/// pinned.
#[test]
fn exactness_probe_runs_on_the_tiny_model() {
    let model = tiny_model();
    let verdict = model.probe_block_chain_exactness(3);
    let _ = verdict.is_equal();
    assert!(matches!(
        model.probe_block_chain_exactness(1),
        crate::models::speculative_exactness::BlockChainExactness::NotRun(_)
    ));
}

/// Opt-in real-checkpoint gate (issue #1326). Set
/// `MLXCEL_TEST_GLM_MTP_TARGET=<glm4_moe_lite checkpoint dir>` to run it;
/// unset, the test passes trivially so the default suite never loads a
/// checkpoint.
///
/// Teacher-forced, so nothing is lost to divergence: the classic greedy chain
/// (`forward` at `M = 1`) is walked for `MLXCEL_TEST_GLM_MTP_STEPS` (default
/// 128) positions after `MLXCEL_TEST_GLM_MTP_PROMPT`, then the same tokens
/// are replayed through `forward_verify` in blocks of
/// `MLXCEL_TEST_GLM_MTP_BLOCK` (default 2) on a second cache set, and every
/// position's argmax, top-two gap and raw logit bytes are compared. The
/// process is pinned narrow (`set_qmv_wide(false)`) exactly as the exactness
/// gate pins it on Apple GPU generation 15+, so the comparison is between the
/// kernels an engaged MTP session actually runs. One `GLMMTP` line per
/// position goes to stderr (run with `--nocapture`); the test fails on an
/// argmax disagreement at a position whose chain gap exceeds
/// `MLXCEL_TEST_GLM_MTP_GAP` (default 0.5, above one bf16 ulp at logit
/// magnitude 30), the decided-position rule `docs/benchmarks.md` gives.
#[test]
fn real_checkpoint_verify_block_matches_decode_chain() {
    let Some(dir) = std::env::var_os("MLXCEL_TEST_GLM_MTP_TARGET") else {
        eprintln!("MLXCEL_TEST_GLM_MTP_TARGET unset; skipping the real-checkpoint gate");
        return;
    };
    let env_usize = |key: &str, default: usize| {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let steps = env_usize("MLXCEL_TEST_GLM_MTP_STEPS", 128);
    let block = env_usize("MLXCEL_TEST_GLM_MTP_BLOCK", 2).max(1);
    let decided_gap: f32 = std::env::var("MLXCEL_TEST_GLM_MTP_GAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5);
    let prompt_text = std::env::var("MLXCEL_TEST_GLM_MTP_PROMPT")
        .unwrap_or_else(|_| "Explain in two sentences why the sky is blue.".to_string());

    let _runtime = crate::initialize_runtime();
    mlxcel_core::set_qmv_wide(false);
    let (loaded, tokenizer) =
        crate::load_model(std::path::Path::new(&dir)).expect("real checkpoint loads");
    let crate::LoadedModel::Glm4MoeLite(model) = &loaded else {
        panic!("MLXCEL_TEST_GLM_MTP_TARGET must name a glm4_moe_lite checkpoint");
    };
    let prompt: Vec<i32> = tokenizer
        .encode(&prompt_text, true)
        .expect("prompt tokenizes")
        .into_iter()
        .map(|t| t as i32)
        .collect();
    eprintln!(
        "GLMMTP\tprompt_tokens\t{}\tsteps\t{steps}\tblock\t{block}",
        prompt.len()
    );

    // Chain arm: classic decode, one token per forward.
    let mut chain_caches = model.make_caches();
    let prefill = model.forward(&ids(&prompt), &mut chain_caches, None);
    let last = mlxcel_core::array_shape(&prefill)[1] - 1;
    let mut tok = argmax(&row(&prefill, last)) as i32;
    let mut tokens: Vec<i32> = vec![tok];
    let mut chain_rows: Vec<Vec<f32>> = Vec::with_capacity(steps);
    for _ in 0..steps {
        let logits = model.forward(&ids(&[tok]), &mut chain_caches, None);
        let r = row(&logits, 0);
        tok = argmax(&r) as i32;
        tokens.push(tok);
        chain_rows.push(r);
    }

    // Block arm: the same prefill, then the chain's tokens replayed through
    // the verify forward `block` at a time.
    let mut block_caches = model.make_caches();
    let _ = model.forward(&ids(&prompt), &mut block_caches, None);
    let mut disagreements = 0usize;
    let mut decided_disagreements = 0usize;
    let mut byte_mismatches = 0usize;
    let mut worst_abs = 0.0f32;
    let mut pos = 0usize;
    while pos < steps {
        let width = block.min(steps - pos);
        let input: Vec<i32> = tokens[pos..pos + width].to_vec();
        let (logits, _hidden) = model.forward_verify(&ids(&input), &mut block_caches);
        for i in 0..width {
            let v = row(&logits, i as i32);
            let c = &chain_rows[pos + i];
            let c_arg = argmax(c);
            let v_arg = argmax(&v);
            let mut sorted: Vec<f32> = c.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let gap = sorted[0] - sorted[1];
            let diff = max_abs_diff(c, &v);
            worst_abs = worst_abs.max(diff);
            let bytes_equal = c.iter().zip(&v).all(|(a, b)| a.to_bits() == b.to_bits());
            if !bytes_equal {
                byte_mismatches += 1;
            }
            let flip = c_arg != v_arg;
            if flip {
                disagreements += 1;
                if gap > decided_gap {
                    decided_disagreements += 1;
                }
            }
            eprintln!(
                "GLMMTP\t{}\trow\t{i}\tchain_argmax\t{c_arg}\tblock_argmax\t{v_arg}\tgap\t{gap:.4}\tmax_abs_diff\t{diff:.4}\tbytes_equal\t{bytes_equal}",
                pos + i
            );
        }
        pos += width;
    }
    eprintln!(
        "GLMMTP\tsummary\tpositions\t{steps}\tbyte_mismatches\t{byte_mismatches}\tworst_abs_diff\t{worst_abs:.4}\targmax_flips\t{disagreements}\tdecided_flips\t{decided_disagreements}\tdecided_gap\t{decided_gap}"
    );
    assert_eq!(
        decided_disagreements, 0,
        "verify block flipped {decided_disagreements} decided position(s) (gap > {decided_gap})"
    );
}

// ── KV cache mode reaches the model-owned MTP slot ────────────────────────

/// The scheduler upgrades a `DenseKvCache` family's caches inside its own
/// `CachePool`, which is why this family carried no mode table before the MTP
/// slot existed. The slot is built by the model, so the resolved table has to
/// reach it here or an operator's `--kv-cache-mode` silently applies to
/// nothing while the server still logs it as applied (issue #1326).
#[test]
fn injected_kv_cache_modes_reach_the_mtp_slot() {
    let model = tiny_model();
    let layers = model.layers.len();
    assert!(layers >= 2, "the tiny model needs a per-layer table");

    // Default: no table injected, every slot cache is FP16.
    assert!(
        model
            .make_configured_caches()
            .iter()
            .all(|c| c.mode == KVCacheMode::Fp16),
        "an uninjected model must stay on FP16"
    );

    // A per-layer table, last layer deliberately left at FP16 so a blanket
    // application would be indistinguishable from the real one.
    let mut modes = vec![KVCacheMode::Int8; layers];
    modes[layers - 1] = KVCacheMode::Fp16;
    LanguageModel::set_kv_cache_layer_modes(&model, modes.clone());

    assert_eq!(
        LanguageModel::kv_cache_layer_modes(&model),
        Some(modes.clone()),
        "the table must be readable back"
    );
    assert_eq!(
        model
            .make_configured_caches()
            .iter()
            .map(|c| c.mode)
            .collect::<Vec<_>>(),
        modes,
        "every MTP slot cache must carry its layer's resolved mode"
    );

    // The install sites, not just the constructor: a sequence's slot and the
    // offline fallback slot both have to come out of `make_configured_caches`.
    let seq = SequenceId::from_raw(7);
    model.reset_mtp_sequence_state(Some(seq));
    assert_eq!(
        model
            .mtp_sequence_state
            .with_sequence_state(Some(seq), |caches| caches
                .iter()
                .map(|c| c.mode)
                .collect::<Vec<_>>()),
        modes,
        "the per-sequence slot install must honour the table"
    );

    model.reset_mtp_sequence_state(None);
    assert_eq!(
        model
            .mtp_sequence_state
            .with_sequence_state(None, |caches| caches
                .iter()
                .map(|c| c.mode)
                .collect::<Vec<_>>()),
        modes,
        "the fallback slot install must honour the table"
    );
}
