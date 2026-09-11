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

//! Greedy-invariant tests for the Laguna DFlash target (#1351).
//!
//! Both tests compare the `DFlashGenerator` round loop against a plain
//! token-at-a-time greedy decode of the same tiny Laguna target
//! ([`super::laguna_tests::tiny_model`], mixed full / sliding layers with a
//! window of 6, so the sliding caches wrap several times inside a run):
//!
//! - [`greedy_invariant_with_forced_rejections`] drives an oracle drafter
//!   that proposes the target's own continuation but corrupts one scheduled
//!   position per round, so every round is a partial accept at a chosen
//!   length (0, 1, 2, and full). That is what exercises
//!   `rollback_partial` on the dense `KVCache`s and the buffered
//!   `RotatingKVCache`s at every trim length across the window wrap.
//! - [`greedy_invariant_with_real_drafter`] runs the real
//!   `LagunaDFlashDrafter` (random weights) through `load_drafter`'s
//!   construction path, `bind`, `reset`, the multi-row first context and the
//!   per-round context appends.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;

use mlxcel_core::drafter::dflash::{DFlashGenerator, SpeculativeTarget};
use mlxcel_core::drafter::laguna_dflash::{LagunaDFlashConfig, LagunaDFlashDrafter};
use mlxcel_core::drafter::{Drafter, DrafterError, DrafterKind};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::sampling::LogprobsConfig;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::laguna::{LagunaModel, LagunaWrapper};
use super::laguna_tests::tiny_model;

const TARGET_LAYER_IDS: [usize; 2] = [1, 3];
const MASK_TOKEN_ID: i32 = 31;
const BLOCK_SIZE: usize = 4;

/// Token-at-a-time greedy continuation of `prompt` for `n` tokens on fresh
/// plain caches (the classic decode path this family ships with).
fn greedy_reference(model: &LagunaModel, prompt: &[i32], n: usize) -> Vec<i32> {
    let mut caches = model.make_caches();
    let ids = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let logits = model.forward_with_caches(&ids, &mut caches);
    let mut next = argmax_last(&logits);
    let mut out = vec![next];
    while out.len() < n {
        let ids = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
        let logits = model.forward_with_caches(&ids, &mut caches);
        next = argmax_last(&logits);
        out.push(next);
    }
    out
}

fn argmax_last(logits: &MlxArray) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let last = shape[1] - 1;
    let row = mlxcel_core::slice(logits, &[0, last, 0], &[1, last + 1, shape[2]]);
    let arg = mlxcel_core::argmax_last_axis(&row);
    let arg = mlxcel_core::reshape(&arg, &[]);
    mlxcel_core::eval(&arg);
    mlxcel_core::item_i32(&arg)
}

/// Drive the round loop on `wrapper` with `drafter`, returning every emitted
/// token (first bonus included) and the per-round accept lengths.
fn run_speculative(
    wrapper: &LagunaWrapper,
    drafter: Box<dyn Drafter>,
    prompt: &[i32],
    max_tokens: usize,
) -> (Vec<i32>, Vec<u32>) {
    run_speculative_with(wrapper, drafter, prompt, max_tokens, BLOCK_SIZE)
}

fn run_speculative_with(
    wrapper: &LagunaWrapper,
    drafter: Box<dyn Drafter>,
    prompt: &[i32],
    max_tokens: usize,
    block_size: usize,
) -> (Vec<i32>, Vec<u32>) {
    let capture: Vec<usize> = drafter
        .dflash_target_layer_ids()
        .expect("drafter declares target layers")
        .to_vec();
    let mut caches = wrapper.model.make_speculative_caches(block_size);
    let ids = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let verify_out = wrapper.verify_forward_with_capture_layers(&ids, &mut caches, &capture);
    let first_bonus = argmax_last(wrapper.verify_logits(&verify_out));
    let first_hidden = wrapper.concat_hidden_for_drafter(&verify_out);
    assert_eq!(
        mlxcel_core::array_shape(&first_hidden),
        vec![
            1,
            prompt.len() as i32,
            capture.len() as i32 * wrapper.model.embed_tokens.weight_shape_hidden()
        ],
        "the whole prompt seeds the drafter context"
    );
    let sampler = SamplingConfig {
        temperature: 0.0,
        ..SamplingConfig::default()
    };
    let mut generator = DFlashGenerator::new(drafter, sampler, block_size as u32, MASK_TOKEN_ID);
    let out = generator
        .run(
            wrapper,
            wrapper as &dyn LanguageModel,
            &mut caches,
            first_bonus,
            first_hidden,
            &[],
            max_tokens,
            &AtomicBool::new(false),
            &LogprobsConfig::default(),
        )
        .expect("round loop runs");
    let mut tokens = vec![first_bonus];
    tokens.extend(out.tokens);
    (tokens, out.accept_lens)
}

/// Proposes the reference continuation, corrupting position
/// `schedule[round]` of each block (when it is inside the block) so the
/// target accepts exactly that many proposals.
struct OracleDrafter {
    reference: Vec<i32>,
    schedule: Vec<usize>,
    target_layer_ids: Vec<usize>,
    round: usize,
    /// Reference index of the bonus the next round must start from; advanced
    /// by the accept length the schedule forces, so a wrong bonus means the
    /// previous round did not accept exactly that many tokens.
    expected_bonus_index: usize,
    vocab: i32,
    bound: bool,
}

impl Drafter for OracleDrafter {
    fn bind(&mut self, _target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.bound = true;
        Ok(())
    }

    fn dflash_target_layer_ids(&self) -> Option<&[usize]> {
        Some(&self.target_layer_ids)
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        assert!(self.bound, "round loop binds before drafting");
        let hidden = hidden.expect("round loop always passes the captured hidden");
        let shape = mlxcel_core::array_shape(hidden);
        assert_eq!(shape.len(), 3);
        // The bonus must be the reference token the forced accept lengths
        // lead to; propose the tokens after it and corrupt the scheduled
        // position so the target accepts exactly that many.
        let k = self.expected_bonus_index;
        assert_eq!(
            self.reference.get(k).copied(),
            Some(last_bonus),
            "round {}: bonus is off the forced accept path",
            self.round
        );
        let mut proposals: Vec<i32> = (1..block_size)
            .map(|j| self.reference.get(k + j).copied().unwrap_or(0))
            .collect();
        let corrupt_at = self.schedule[self.round % self.schedule.len()];
        self.round += 1;
        let accepted = corrupt_at.min(proposals.len());
        if corrupt_at < proposals.len() {
            proposals[corrupt_at] = (proposals[corrupt_at] + 1) % self.vocab;
        }
        self.expected_bonus_index = k + accepted + 1;
        Ok(proposals)
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Dflash
    }
}

trait EmbeddingHidden {
    fn weight_shape_hidden(&self) -> i32;
}

impl EmbeddingHidden for mlxcel_core::layers::UnifiedEmbedding {
    fn weight_shape_hidden(&self) -> i32 {
        mlxcel_core::array_shape(self.weight())[1]
    }
}

fn prompt() -> Vec<i32> {
    vec![3, 17, 9, 25, 4, 12, 30, 7, 18, 2, 21, 11]
}

#[test]
fn greedy_invariant_with_forced_rejections() {
    let (model, args) = tiny_model();
    let n = 40usize;
    let prompt = prompt();
    let reference = greedy_reference(&model, &prompt, n);
    assert!(
        reference.windows(2).any(|w| w[0] != w[1]),
        "the reference must not be a degenerate repeat"
    );
    let wrapper = LagunaWrapper::new(model);

    // Accept lengths 0, 1, 2 and a full accept (schedule index past the
    // proposals), cycling so every trim length hits the caches after the
    // sliding window (6) has wrapped.
    let schedule = vec![0, 2, 1, usize::MAX, 0, 1, 2];
    let drafter = OracleDrafter {
        reference: reference.clone(),
        schedule: schedule.clone(),
        target_layer_ids: TARGET_LAYER_IDS.to_vec(),
        round: 0,
        expected_bonus_index: 0,
        vocab: args.vocab_size as i32,
        bound: false,
    };
    let (tokens, accept_lens) = run_speculative(&wrapper, Box::new(drafter), &prompt, n);
    assert_eq!(
        tokens, reference,
        "speculative output must equal greedy decode"
    );
    // Every round accepted exactly what the schedule forced (the last
    // round may be shortened by the token budget).
    for (round, accepted) in accept_lens.iter().enumerate() {
        let forced = schedule[round % schedule.len()].min(BLOCK_SIZE - 1) as u32;
        if round + 1 < accept_lens.len() {
            assert_eq!(*accepted, forced, "round {round} accept length");
        }
    }
    assert!(accept_lens.contains(&0) && accept_lens.contains(&1) && accept_lens.contains(&2));
    assert!(accept_lens.contains(&((BLOCK_SIZE - 1) as u32)));
}

/// Random drafter weights in the published layout, sized for the tiny
/// target (hidden 16, vocab 32) with `TARGET_LAYER_IDS` captured.
fn tiny_drafter_weights(cfg: &LagunaDFlashConfig, seed: u64) -> WeightMap {
    let mut state = seed;
    let mut next = |scale: f32| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32);
        (unit * 2.0 - 1.0) * scale
    };
    let mut rand = |shape: &[i32], scale: f32| -> UniquePtr<MlxArray> {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| next(scale)).collect();
        mlxcel_core::from_slice_f32(&data, shape)
    };
    let ones = |n: i32| mlxcel_core::from_slice_f32(&vec![1.0; n as usize], &[n]);
    let h = cfg.hidden_size as i32;
    let d = cfg.head_dim as i32;
    let nh = cfg.num_attention_heads as i32;
    let kv = cfg.num_key_value_heads as i32;
    let ii = cfg.intermediate_size as i32;
    let n = cfg.target_layer_ids.len() as i32;
    let mut w: WeightMap = std::collections::HashMap::new();
    w.insert("fc.weight".into(), rand(&[h, n * h], 0.3));
    w.insert("hidden_norm.weight".into(), ones(h));
    w.insert("norm.weight".into(), ones(h));
    for i in 0..n {
        w.insert(format!("aux_hidden_norms.{i}.weight"), ones(h));
    }
    for i in 0..cfg.num_hidden_layers {
        let p = format!("layers.{i}");
        w.insert(format!("{p}.input_layernorm.weight"), ones(h));
        w.insert(format!("{p}.post_attention_layernorm.weight"), ones(h));
        w.insert(
            format!("{p}.self_attn.qkv_proj.weight"),
            rand(&[(nh + 2 * kv) * d, h], 0.3),
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand(&[h, nh * d], 0.3),
        );
        w.insert(format!("{p}.self_attn.g_proj.weight"), rand(&[nh, h], 0.3));
        w.insert(format!("{p}.self_attn.q_norm.weight"), ones(d));
        w.insert(format!("{p}.self_attn.k_norm.weight"), ones(d));
        w.insert(format!("{p}.mlp.gate_proj.weight"), rand(&[ii, h], 0.3));
        w.insert(format!("{p}.mlp.up_proj.weight"), rand(&[ii, h], 0.3));
        w.insert(format!("{p}.mlp.down_proj.weight"), rand(&[h, ii], 0.3));
    }
    w
}

fn tiny_drafter_config() -> LagunaDFlashConfig {
    LagunaDFlashConfig::from_json(&serde_json::json!({
        "model_type": "laguna", "architectures": ["DFlashLagunaForCausalLM"],
        "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        "rms_norm_eps": 1e-6, "vocab_size": 32, "draft_vocab_size": 32,
        "rope_theta": 10000.0, "sliding_window": 6,
        "layer_types": ["sliding_attention", "sliding_attention"], "gating": "per-head",
        "dflash_config": {"block_size": BLOCK_SIZE, "mask_token_id": MASK_TOKEN_ID,
                          "num_target_layers": 4, "target_layer_ids": TARGET_LAYER_IDS,
                          "causal": true},
        "num_experts": 0
    }))
    .expect("tiny drafter config")
}

#[test]
fn greedy_invariant_with_real_drafter() {
    let (model, _args) = tiny_model();
    let n = 40usize;
    let prompt = prompt();
    let reference = greedy_reference(&model, &prompt, n);
    let wrapper = LagunaWrapper::new(model);

    let cfg = tiny_drafter_config();
    let weights = tiny_drafter_weights(&cfg, 0x1351);
    let drafter = LagunaDFlashDrafter::from_weights(&weights, cfg).expect("drafter builds");
    // `validate_target_compat` reads the target depth and vocabulary.
    drafter
        .validate_target_compat(&wrapper as &dyn LanguageModel)
        .expect("tiny drafter matches the tiny target");
    let mut mismatched = LagunaDFlashDrafter::from_weights(&weights, {
        let mut c = tiny_drafter_config();
        c.num_target_layers = 5;
        c
    })
    .expect("builds");
    assert!(
        mismatched
            .validate_target_compat(&wrapper as &dyn LanguageModel)
            .is_err(),
        "a depth mismatch is refused before bind"
    );
    assert!(mismatched.bind(&wrapper as &dyn LanguageModel).is_ok());

    let (tokens, accept_lens) = run_speculative(&wrapper, Box::new(drafter), &prompt, n);
    assert_eq!(
        tokens, reference,
        "speculative output must equal greedy decode"
    );
    assert!(!accept_lens.is_empty());
    // A random drafter rejects almost everything, so rollback runs on
    // nearly every round.
    assert!(accept_lens.iter().any(|a| (*a as usize) < BLOCK_SIZE - 1));
}

/// Per-round record of the real drafter's proposals: `(reference index of
/// the bonus, proposals)`.
type ProposalLog = Rc<RefCell<Vec<(usize, Vec<i32>)>>>;

/// Records the real drafter's proposals every round while answering the
/// round loop with the reference continuation, so the loop stays on the
/// reference path (full accepts) and the real drafter's per-position
/// accuracy can be measured without rejection feedback.
struct ShadowDrafter {
    real: Box<dyn Drafter>,
    oracle: OracleDrafter,
    log: ProposalLog,
}

impl Drafter for ShadowDrafter {
    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.oracle.bind(target)?;
        self.real.bind(target)
    }

    fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.oracle.reset(target)?;
        self.real.reset(target)
    }

    fn dflash_target_layer_ids(&self) -> Option<&[usize]> {
        // The oracle's list (an optional `MLXCEL_LAGUNA_PROBE_CAPTURE`
        // override of the drafter's own `target_layer_ids`).
        Some(&self.oracle.target_layer_ids)
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        let k = self.oracle.expected_bonus_index;
        let real = self
            .real
            .draft_block(last_bonus, hidden, block_size, sampler)?;
        self.log.borrow_mut().push((k, real));
        self.oracle
            .draft_block(last_bonus, hidden, block_size, sampler)
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Dflash
    }
}

fn env_ids(name: &str) -> Option<Vec<i32>> {
    let raw = std::env::var(name).ok()?;
    let ids: Vec<i32> = raw
        .trim()
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<i32>().expect("integer token id"))
        .collect();
    Some(ids)
}

/// Real-checkpoint probe. Needs the target and drafter checkpoints and the
/// classic decode's ids (`MLXCEL_PRINT_TOKEN_IDS=1 mlxcel generate ...`):
///
/// ```text
/// MLXCEL_LAGUNA_PROBE_PROMPT="[...]" MLXCEL_LAGUNA_PROBE_REFERENCE="[...]" \
///   cargo test --profile test-fast --features cuda --lib -- \
///   models::laguna_dflash_tests::laguna_real_checkpoint_probe --ignored --nocapture
/// ```
///
/// Reports (a) whether the target verify path reproduces the reference with
/// an oracle drafter (block-versus-chain agreement on this checkpoint), and
/// (b) the real drafter's per-position accuracy along the reference path.
#[test]
#[ignore = "needs the real Laguna XS 2.1 checkpoints and the classic decode ids"]
fn laguna_real_checkpoint_probe() {
    use std::path::Path;

    let target_dir = std::env::var("MLXCEL_LAGUNA_PROBE_TARGET")
        .unwrap_or_else(|_| "models/mlx/laguna-xs-2.1-nvfp4".to_string());
    let drafter_dir = std::env::var("MLXCEL_LAGUNA_PROBE_DRAFTER")
        .unwrap_or_else(|_| "models/mlx/laguna-xs-2.1-dflash".to_string());
    let prompt = env_ids("MLXCEL_LAGUNA_PROBE_PROMPT").expect("MLXCEL_LAGUNA_PROBE_PROMPT");
    let reference =
        env_ids("MLXCEL_LAGUNA_PROBE_REFERENCE").expect("MLXCEL_LAGUNA_PROBE_REFERENCE");
    let block_size: usize = std::env::var("MLXCEL_LAGUNA_PROBE_BLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let (model, _tokenizer) = crate::backend::select_backend()
        .load_model(Path::new(&target_dir))
        .expect("target loads");
    let crate::LoadedModel::Laguna(wrapper) = &model else {
        panic!("target is not Laguna");
    };
    let (drafter, _kind) =
        mlxcel_core::drafter::load_drafter(Path::new(&drafter_dir), Some(DrafterKind::Dflash))
            .expect("drafter loads");
    let target_layer_ids: Vec<usize> = match std::env::var("MLXCEL_LAGUNA_PROBE_CAPTURE") {
        Ok(v) => v
            .split(',')
            .map(|s| s.trim().parse::<usize>().expect("layer id"))
            .collect(),
        Err(_) => drafter
            .dflash_target_layer_ids()
            .expect("target layer ids")
            .to_vec(),
    };
    println!("[probe] capture layers {target_layer_ids:?}");
    let n = reference.len();
    let oracle = |target_layer_ids: Vec<usize>| OracleDrafter {
        reference: reference.clone(),
        schedule: vec![usize::MAX],
        target_layer_ids,
        round: 0,
        expected_bonus_index: 0,
        vocab: 100_352,
        bound: false,
    };

    let mode = std::env::var("MLXCEL_LAGUNA_PROBE_MODE").unwrap_or_else(|_| "ab".to_string());
    // (a) Block versus chain on the target alone: the same prompt prefill,
    // then the reference tokens fed as single-token decode steps (the
    // classic path) and as `block_size`-wide verify blocks (the speculative
    // path). At the first argmax disagreement, report both arms' top-2
    // tokens and margins.
    if mode.contains('a') {
        probe_block_versus_chain(wrapper, &prompt, &reference, block_size);
    }
    if !mode.contains('b') {
        return;
    }

    // (b) Real drafter along the reference path.
    let log: ProposalLog = Rc::new(RefCell::new(Vec::new()));
    let shadow = ShadowDrafter {
        real: drafter,
        oracle: oracle(target_layer_ids),
        log: Rc::clone(&log),
    };
    let (tokens_b, _) = run_speculative_with(wrapper, Box::new(shadow), &prompt, n, block_size);
    let first_diff_b = tokens_b.iter().zip(&reference).position(|(a, b)| a != b);
    println!(
        "[probe b] shadow round loop: first divergence from reference at {:?}",
        first_diff_b
    );
    let mut hits = vec![0usize; block_size - 1];
    let mut counts = vec![0usize; block_size - 1];
    let mut prefix_hist = vec![0usize; block_size];
    for (k, proposals) in log.borrow().iter() {
        let mut prefix = 0usize;
        let mut prefix_ok = true;
        for (j, d) in proposals.iter().enumerate() {
            let Some(want) = reference.get(k + 1 + j) else {
                break;
            };
            counts[j] += 1;
            if d == want {
                hits[j] += 1;
                if prefix_ok {
                    prefix += 1;
                }
            } else {
                prefix_ok = false;
            }
        }
        prefix_hist[prefix] += 1;
        println!(
            "[probe b] round at reference {k}: proposals {:?} reference {:?}",
            proposals,
            &reference[(k + 1).min(n)..(k + 1 + proposals.len()).min(n)]
        );
    }
    let acc: Vec<String> = hits
        .iter()
        .zip(&counts)
        .map(|(h, c)| {
            if *c == 0 {
                "-".to_string()
            } else {
                format!("{:.2}", *h as f64 / *c as f64)
            }
        })
        .collect();
    let rounds = log.borrow().len();
    let mean_prefix: f64 = prefix_hist
        .iter()
        .enumerate()
        .map(|(len, c)| len as f64 * *c as f64)
        .sum::<f64>()
        / rounds.max(1) as f64;
    println!(
        "[probe b] per-position accuracy d_0..d_{}: {:?}; mean accepted prefix on the \
         reference path = {mean_prefix:.2} over {rounds} rounds",
        block_size - 2,
        acc
    );
}

fn top2(logits: &MlxArray, pos: i32) -> ((i32, f32), (i32, f32)) {
    let shape = mlxcel_core::array_shape(logits);
    let row = mlxcel_core::slice(logits, &[0, pos, 0], &[1, pos + 1, shape[2]]);
    let row = mlxcel_core::astype(&row, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&row);
    let data = mlxcel_core::utils::array_to_vec_f32(&row);
    let mut best = (0usize, f32::NEG_INFINITY);
    let mut second = (0usize, f32::NEG_INFINITY);
    for (i, v) in data.iter().enumerate() {
        if *v > best.1 {
            second = best;
            best = (i, *v);
        } else if *v > second.1 {
            second = (i, *v);
        }
    }
    ((best.0 as i32, best.1), (second.0 as i32, second.1))
}

fn probe_block_versus_chain(
    wrapper: &LagunaWrapper,
    prompt: &[i32],
    reference: &[i32],
    block_size: usize,
) {
    let model = &wrapper.model;
    // Chain arm: classic prefill then one token per forward.
    let mut chain_caches = model.make_caches();
    let ids = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let prefill_logits = model.forward_with_caches(&ids, &mut chain_caches);
    let chain_first = top2(&prefill_logits, prompt.len() as i32 - 1);
    let mut chain: Vec<((i32, f32), (i32, f32))> = Vec::with_capacity(reference.len());
    for tok in reference {
        let ids = mlxcel_core::from_slice_i32(&[*tok], &[1, 1]);
        let logits = model.forward_with_caches(&ids, &mut chain_caches);
        chain.push(top2(&logits, 0));
    }
    // Block arm: speculative caches, prefill through the verify hook, then
    // the reference in `block_size` blocks (each block plays the role of
    // `[bonus, d_0, ...]` with every proposal correct).
    let mut block_caches = model.make_speculative_caches(block_size);
    let out = model.forward_speculative(&ids_of(prompt), &mut block_caches, &[]);
    let block_first = top2(&out.logits, prompt.len() as i32 - 1);
    let mut block: Vec<((i32, f32), (i32, f32))> = Vec::with_capacity(reference.len());
    for chunk in reference.chunks(block_size) {
        let out = model.forward_speculative(&ids_of(chunk), &mut block_caches, &[]);
        for pos in 0..chunk.len() {
            block.push(top2(&out.logits, pos as i32));
        }
    }
    println!(
        "[probe a] first token: chain {:?} block {:?}",
        chain_first, block_first
    );
    let mut mismatches = 0usize;
    let mut min_margin = f32::INFINITY;
    for (i, (c, b)) in chain.iter().zip(&block).enumerate() {
        let margin = c.0.1 - c.1.1;
        min_margin = min_margin.min(margin);
        if c.0.0 != b.0.0 {
            mismatches += 1;
            if mismatches <= 5 {
                println!(
                    "[probe a] argmax disagreement after reference[{i}] (predicting \
                     reference[{}] = {:?}): chain top2 {:?} block top2 {:?}",
                    i + 1,
                    reference.get(i + 1),
                    c,
                    b
                );
            }
        }
    }
    println!(
        "[probe a] {} of {} positions disagree between chain and block; smallest chain \
         top-2 margin {min_margin:.4}",
        mismatches,
        chain.len()
    );
}

fn ids_of(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}
