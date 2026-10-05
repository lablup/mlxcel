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

//! Cache-dependent toy targets and a sequential reference decoder, shared by
//! the decode-loop tests (`generate_history_tests.rs`, `speculative/prompt_lookup_tests.rs`).
//!
//! A decode loop is right when it emits exactly what [`sequential_reference`]
//! emits for the same model and sampler: one forward per token, every sample
//! taken against the full emitted history. [`InductionModel`] makes that a
//! sharp test, because its prediction depends on everything in its KV cache.

use crate::ffi;
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::KVCache;
use crate::sampling::sample_token_optimized;
use cxx::UniquePtr;

/// Deterministic pseudo-random tokens from a small alphabet, so n-grams recur.
pub(crate) fn lcg_tokens(seed: u64, len: usize, alphabet: u64) -> Vec<i32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % alphabet) as i32
        })
        .collect()
}

/// Vocabulary of [`InductionModel`].
pub(crate) const INDUCTION_VOCAB: usize = 6;

/// Target whose prediction depends on everything in its KV cache, so a
/// rollback that leaves a rejected proposal behind, or trims one token too
/// many, changes what it predicts from then on.
///
/// Each forward appends its input tokens to the cache as key values and, at
/// every new position, predicts the token that followed an earlier occurrence
/// of the token at that position (a one-token induction head over the cached
/// sequence), or `token + 1` when there is none. The following token gets a
/// runner-up logit, so a repetition penalty can flip the choice.
///
/// With `earliest: false` it copies from the most recent occurrence, the one
/// lookup also prefers, so most proposals land and the governor keeps full
/// blocks. With `earliest: true` it copies from the first occurrence, so
/// proposals keep missing, the governor pauses, and the decode loop pipelines
/// runs of plain rounds and switches back to verifying when a pause ends.
pub(crate) struct InductionModel {
    pub(crate) earliest: bool,
}

pub(crate) const INDUCTION_MODELS: [InductionModel; 2] = [
    InductionModel { earliest: false },
    InductionModel { earliest: true },
];

impl LanguageModel for InductionModel {
    fn forward(
        &self,
        input_ids: &ffi::MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&ffi::MlxArray>,
    ) -> UniquePtr<ffi::MlxArray> {
        let seq_len = ffi::array_shape(input_ids)[1];
        let as_f16 = ffi::astype(input_ids, crate::dtype::FLOAT16);
        let keys = ffi::reshape(&as_f16, &[1, 1, seq_len, 1]);
        let values = ffi::reshape(&as_f16, &[1, 1, seq_len, 1]);
        let (window, _) = caches[0].update_and_fetch(keys, values);
        let sequence: Vec<i32> = crate::utils::array_to_vec_f32(&window)
            .into_iter()
            .map(|v| v as i32)
            .collect();
        let new_len = seq_len as usize;
        let mut logits = vec![0.0f32; new_len * INDUCTION_VOCAB];
        for (row, pos) in (sequence.len() - new_len..sequence.len()).enumerate() {
            let token = sequence[pos];
            let mut earlier = (0..pos).filter(|&i| sequence[i] == token);
            let source = if self.earliest {
                earlier.next()
            } else {
                earlier.next_back()
            };
            let next =
                source.map_or((token + 1) % INDUCTION_VOCAB as i32, |i| sequence[i + 1]) as usize;
            logits[row * INDUCTION_VOCAB + next] = 8.0;
            logits[row * INDUCTION_VOCAB + (next + 1) % INDUCTION_VOCAB] = 7.0;
        }
        ffi::from_slice_f32(&logits, &[1, seq_len, INDUCTION_VOCAB as i32])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }
}

/// Plain decoding without pipelining: one forward per token, each sampled
/// against a history that already holds every emitted token, which is what a
/// history-reading sampler is defined against (mlx-lm's `generate_step` hands
/// its logits processors the token it just consumed).
///
/// Every decode loop has to match it, `CxxGenerator`'s pipelined ones included:
/// until #2090 they sampled each step against a history missing the token
/// they had just read.
pub(crate) fn sequential_reference<M: LanguageModel>(
    model: &M,
    prompt: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
) -> Vec<i32> {
    let mut caches = model.make_caches();
    let mut history = prompt.to_vec();
    let mut input = prompt.to_vec();
    let mut emitted = Vec::with_capacity(max_tokens);
    while emitted.len() < max_tokens {
        let ids = ffi::from_slice_i32(&input, &[1, input.len() as i32]);
        let logits = model.forward(&ids, &mut caches, None);
        let (token, _) = sample_token_optimized(&logits, sampling, &history);
        let token = ffi::item_i32(&token);
        emitted.push(token);
        history.push(token);
        input = vec![token];
    }
    emitted
}
