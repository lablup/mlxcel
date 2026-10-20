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

//! `LanguageModel::after_prefill` ordering (#2177): an embedding prefill
//! evaluates its logits before the hook runs, the hook runs exactly once per
//! embedding prefill, after the prefill forward and before the first decode
//! forward, and a text prefill never runs it.

use std::cell::RefCell;

use super::*;
use crate::cache::SequenceId;
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::KVCache;
use crate::session::MlxInferenceSession;
use crate::utils::array_to_vec_f32;
use crate::{ffi, from_slice_i32};

const VOCAB: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    /// A token forward: a text prefill piece or a decode step.
    Forward,
    /// The embedding-prefill forward.
    EmbeddingPrefill,
    AfterPrefill,
}

/// Records every forward and `after_prefill` call in order. Logits put the
/// input token at the top, so greedy decode echoes the last prompt token.
/// With `failed_prefill_logits`, both prefill forwards return an array whose
/// event carries a synthetic backend error: evaluating it fails, so whether
/// the engine forced an eval is visible in `PrefillOutcome::eval`.
#[derive(Default)]
struct HookModel {
    events: RefCell<Vec<Event>>,
    failed_prefill_logits: bool,
}

impl HookModel {
    fn failing() -> Self {
        Self {
            failed_prefill_logits: true,
            ..Self::default()
        }
    }

    fn events(&self) -> Vec<Event> {
        self.events.borrow().clone()
    }

    fn record(&self, event: Event) {
        self.events.borrow_mut().push(event);
    }

    fn append(caches: &mut [KVCache], n: i32) {
        for cache in caches.iter_mut() {
            let k = ffi::zeros(&[1, 1, n, 4], crate::dtype::FLOAT32);
            let v = ffi::zeros(&[1, 1, n, 4], crate::dtype::FLOAT32);
            cache.update(k, v);
        }
    }

    /// `[B, L, VOCAB]` logits, or `[1, 1, VOCAB]` at `last_pos` of row 0.
    fn echo_logits(input_ids: &MlxArray, last_pos: Option<usize>) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(input_ids);
        let (b, l) = (shape[0] as usize, shape[1] as usize);
        let toks: Vec<i32> = array_to_vec_f32(&ffi::astype(input_ids, crate::dtype::FLOAT32))
            .into_iter()
            .map(|t| t as i32)
            .collect();
        let rows: Vec<i32> = match last_pos {
            Some(pos) => vec![toks[pos]],
            None => toks,
        };
        let mut logits = vec![0.0f32; rows.len() * VOCAB];
        for (i, &tok) in rows.iter().enumerate() {
            if (0..VOCAB as i32).contains(&tok) {
                logits[i * VOCAB + tok as usize] = 10.0;
            }
        }
        let (b, l) = if last_pos.is_some() { (1, 1) } else { (b, l) };
        ffi::from_slice_f32(&logits, &[b as i32, l as i32, VOCAB as i32])
    }

    fn prefill_logits(&self, input_ids: &MlxArray, last_pos: usize) -> UniquePtr<MlxArray> {
        if self.failed_prefill_logits {
            crate::array_with_failed_event_for_test()
        } else {
            Self::echo_logits(input_ids, Some(last_pos))
        }
    }
}

impl LanguageModel for HookModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.record(Event::Forward);
        Self::append(caches, ffi::array_shape(input_ids)[1]);
        Self::echo_logits(input_ids, None)
    }

    fn forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.record(Event::Forward);
        Self::append(caches, ffi::array_shape(input_ids)[1]);
        self.prefill_logits(input_ids, last_pos)
    }

    fn forward_last_logits_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        assert!(
            input_embeddings.is_some(),
            "the embedding forward gets them"
        );
        self.record(Event::EmbeddingPrefill);
        Self::append(caches, ffi::array_shape(input_ids)[1]);
        self.prefill_logits(input_ids, last_pos)
    }

    fn after_prefill(&self) {
        self.record(Event::AfterPrefill);
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new(), KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        2
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![7]
    }
}

fn embeddings(len: i32) -> UniquePtr<MlxArray> {
    ffi::zeros(&[1, len, 4], crate::dtype::FLOAT32)
}

/// Asserts `after_prefill` ran once, right after the embedding prefill
/// forward and before any decode forward, and that decode did run.
fn assert_hook_between_prefill_and_decode(events: &[Event]) {
    assert_eq!(
        events.first(),
        Some(&Event::EmbeddingPrefill),
        "the run starts with the embedding prefill: {events:?}"
    );
    assert_eq!(
        events.get(1),
        Some(&Event::AfterPrefill),
        "the hook follows the prefill forward: {events:?}"
    );
    let rest = &events[2..];
    assert!(
        !rest.is_empty() && rest.iter().all(|e| *e == Event::Forward),
        "only decode forwards follow the hook, at least one: {events:?}"
    );
}

/// The eval is forced for an embedding piece even when the caller leaves
/// `eval` off: the failing logits surface in `outcome.eval`. Before #2177
/// the hook ran over the unevaluated graph and `outcome.eval` was `Ok`.
#[test]
fn embedding_prefill_evaluates_its_logits_even_with_eval_off() {
    let mut engine = Engine::with_capacity(HookModel::failing(), 2);
    let id = engine.open(SequenceSpec::default()).unwrap();
    let input = from_slice_i32(&[3, 5, 1], &[1, 3]);
    let embeds = embeddings(3);
    let outcome = engine
        .prefill(&PrefillStep {
            seq_id: id,
            input: &input,
            embeddings: Some(&embeds),
            mask: None,
            last_pos: 2,
            trim_excess: 0,
            eval: false,
        })
        .unwrap();
    let err = outcome
        .eval
        .expect_err("an embedding prefill must evaluate its logits");
    assert!(
        err.contains("synthetic array read failure"),
        "unexpected eval error: {err}"
    );
    assert_eq!(
        engine.model().events(),
        vec![Event::EmbeddingPrefill, Event::AfterPrefill]
    );
}

/// A text piece keeps `eval` as the caller set it (the terminal piece's eval
/// stays with the first sample) and never runs the hook.
#[test]
fn text_prefill_keeps_eval_deferred_and_skips_after_prefill() {
    let mut engine = Engine::with_capacity(HookModel::failing(), 2);
    let id = engine.open(SequenceSpec::default()).unwrap();
    let input = from_slice_i32(&[3, 5, 1], &[1, 3]);
    let outcome = engine
        .prefill(&PrefillStep {
            seq_id: id,
            input: &input,
            embeddings: None,
            mask: None,
            last_pos: 2,
            trim_excess: 0,
            eval: false,
        })
        .unwrap();
    assert!(outcome.eval.is_ok(), "a deferred text eval is not forced");
    assert_eq!(engine.model().events(), vec![Event::Forward]);
}

/// Both session embedding variants (`generate_streaming_with_embeddings`,
/// the plain `mlxcel generate --image` path, and
/// `generate_with_stats_and_embeddings`, its `--profile` twin) run the hook
/// exactly once, between the prefill forward and the first decode forward,
/// and decode the same tokens.
#[test]
fn both_session_embedding_variants_run_after_prefill_once_before_decode() {
    let prompt = [3, 5, 1];
    let greedy = SamplingConfig::greedy();
    let embeds = embeddings(prompt.len() as i32);

    let streaming_model = HookModel::default();
    let mut session = MlxInferenceSession::new(2);
    let streamed = session.generate_streaming_with_embeddings(
        &streaming_model,
        &prompt,
        Some(&embeds),
        None,
        4,
        &greedy,
        |_| true,
    );
    assert_eq!(streamed, vec![1, 1, 1, 1]);
    assert_hook_between_prefill_and_decode(&streaming_model.events());

    let stats_model = HookModel::default();
    let mut session = MlxInferenceSession::new(2);
    let (with_stats, stats) = session.generate_with_stats_and_embeddings(
        &stats_model,
        &prompt,
        Some(&embeds),
        None,
        4,
        &greedy,
    );
    assert_eq!(
        with_stats, streamed,
        "the profile path decodes the same tokens"
    );
    assert_eq!(stats.generated_tokens, 4);
    assert_hook_between_prefill_and_decode(&stats_model.events());
}

/// A text run through the same client never calls the hook.
#[test]
fn text_generation_never_runs_after_prefill() {
    let model = HookModel::default();
    let mut session = MlxInferenceSession::new(2);
    let tokens = session.generate(&model, &[3, 5, 1], 3, &SamplingConfig::greedy());
    assert_eq!(tokens, vec![1, 1, 1]);
    assert!(
        !model.events().contains(&Event::AfterPrefill),
        "{:?}",
        model.events()
    );
}
