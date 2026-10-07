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

//! One token-bias composition for the CLI and the server (issue #2169).
//!
//! A request's effective [`TokenBiasMap`] comes from three sources, and
//! [`compose_token_bias`] is the only place they are merged. Both the CLI and
//! the server call it once per request.
//!
//! ## Precedence
//!
//! 1. **Request bias.** The request's own map: `logit_bias` entries, with
//!    string keys already tokenized, accumulated additively
//!    ([`TokenBiasMap::accumulate`]: a token named twice gets both biases,
//!    and `-inf` saturates). On the server this map starts from the bias the
//!    request resolution placed on the request's sampling config (the live
//!    settings' resolved language bias), and the request's `logit_bias`
//!    accumulates on top of it. The CLI has no per-request bias and passes an
//!    empty map.
//! 2. **Language bias.** The worker- or CLI-level `--lang-bias` map is a
//!    fallback policy: it applies only when the request bias is empty. A
//!    biased request carries its own policy, so the two are not merged. The
//!    server passes an empty map here when the request runs under a per-request
//!    runtime override (`use_worker_token_bias == false`).
//! 3. **Output suppression.** The model's output-illegal ids (multimodal
//!    placeholder markers, #350) are forced to `-inf` last, so a suppressed id
//!    stays at `-inf` whatever the other two maps say.
//!
//! `ignore_eos` is not a source here: the server suppresses the request's
//! merged EOS set right after composition, because that set depends on the
//! request's stop tokens, and suppression is idempotent, so applying it after
//! composition cannot lose to any bias either. The CLI has no `ignore_eos`.
//!
//! When every source is empty the result is an empty map, which the sampler
//! short-circuits with no graph nodes: an unbiased request costs nothing.

use crate::sampling::TokenBiasMap;

/// Compose a request's effective token bias (see the module doc for the
/// precedence).
///
/// `request` is consumed so the common case (no request bias, or no language
/// bias) moves a map instead of cloning one; `language` is cloned only when it
/// is the map that applies.
///
/// Used by: CLI `run_generation_mode`, the chat REPL, the CLI pipeline path,
/// `BatchScheduler::enqueue_request`
pub fn compose_token_bias(
    request: TokenBiasMap,
    language: &TokenBiasMap,
    output_suppression: &[i32],
) -> TokenBiasMap {
    let mut composed = if request.is_empty() && !language.is_empty() {
        language.clone()
    } else {
        request
    };
    composed.suppress_tokens(output_suppression);
    composed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(i32, f32)]) -> TokenBiasMap {
        let mut map = TokenBiasMap::new();
        for &(id, bias) in entries {
            map.insert(id, bias);
        }
        map
    }

    #[test]
    fn empty_sources_compose_to_an_empty_map() {
        let composed = compose_token_bias(TokenBiasMap::new(), &TokenBiasMap::new(), &[]);
        assert!(composed.is_empty());
    }

    #[test]
    fn language_bias_applies_only_without_a_request_bias() {
        let language = map(&[(1, -5.0), (2, f32::NEG_INFINITY)]);
        let composed = compose_token_bias(TokenBiasMap::new(), &language, &[]);
        assert_eq!(composed.len(), 2);
        assert_eq!(composed.get(&1), Some(&-5.0));

        let request = map(&[(3, 1.5)]);
        let composed = compose_token_bias(request, &language, &[]);
        assert_eq!(
            composed.len(),
            1,
            "a biased request replaces the language policy"
        );
        assert_eq!(composed.get(&3), Some(&1.5));
        assert!(!composed.contains(1));
    }

    #[test]
    fn output_suppression_always_wins() {
        let language = map(&[(7, 4.0)]);
        let request = map(&[(8, 9.0)]);
        let composed = compose_token_bias(request, &language, &[8, 9]);
        assert_eq!(composed.get(&8), Some(&f32::NEG_INFINITY));
        assert_eq!(composed.get(&9), Some(&f32::NEG_INFINITY));

        let composed = compose_token_bias(TokenBiasMap::new(), &language, &[7]);
        assert_eq!(composed.get(&7), Some(&f32::NEG_INFINITY));
    }

    #[test]
    fn byte_fragment_tags_survive_the_language_fallback() {
        let mut language = TokenBiasMap::new();
        language.insert_byte_fragment(11, f32::NEG_INFINITY);
        let composed = compose_token_bias(TokenBiasMap::new(), &language, &[]);
        assert!(composed.is_byte_fragment(11));
    }
}
