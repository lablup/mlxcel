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

//! Focused tests for the core inference-session contract. These do not load a
//! real checkpoint (token-parity is owned by the CLI parity gate); they assert
//! the capability advertisement, token-bias wiring, the object-safe trait
//! contract, and that the conceptual token-level primitives report they are
//! reserved for the compiler-family backend on MLX.

use super::*;
use crate::sampling::TokenBiasMap;

/// Compile-time proof that the MLX session satisfies the object-safe contract,
/// so a future backend can be held behind `Box<dyn InferenceSession>`.
const _: () = {
    fn _requires_session<T: InferenceSession>() {}
    fn _check() {
        _requires_session::<MlxInferenceSession>();
    }
    fn _object_safe(_s: &dyn InferenceSession) {}
};

#[test]
fn single_sequence_caps_are_the_conservative_floor() {
    let caps = SessionCapabilities::single_sequence();
    assert!(!caps.batched_serving);
    assert!(!caps.paged_kv);
    assert!(!caps.speculative_decode);
    assert!(!caps.multimodal);
}

#[test]
fn mlx_session_advertises_single_sequence_plus_multimodal() {
    let session = MlxInferenceSession::new(4);
    let caps = session.capabilities();
    // The single-sequence MLX session does not batch, page, or speculate
    // internally, but it accepts multimodal embedding prefill.
    assert!(!caps.batched_serving);
    assert!(!caps.paged_kv);
    assert!(!caps.speculative_decode);
    assert!(caps.multimodal);
}

#[test]
fn mlx_session_default_bias_is_empty() {
    let session = MlxInferenceSession::new(4);
    assert!(session.token_bias().is_empty());
}

#[test]
fn mlx_session_with_token_bias_caches_map() {
    let mut bias = TokenBiasMap::new();
    bias.insert(3, -1.5);
    let session = MlxInferenceSession::new_with_kv_mode(4, KVCacheMode::Fp16).with_token_bias(bias);
    assert_eq!(session.token_bias().len(), 1);
    assert!(session.token_bias().contains(3));
}

#[test]
fn mlx_session_step_primitives_are_reserved_for_compiler_backend() {
    let mut session = MlxInferenceSession::new(4);
    let prefill = session.prefill(&[1, 2, 3]);
    assert!(
        prefill.is_err(),
        "MLX prefill must report it is the reserved token-level contract"
    );
    let step = session.decode_step(7);
    assert!(
        step.is_err(),
        "MLX decode_step must report it is the reserved token-level contract"
    );
    // The error names the fused entry points the CLI actually uses.
    let msg = prefill.unwrap_err().to_string();
    assert!(msg.contains("generate_streaming"), "error message: {msg}");
}
