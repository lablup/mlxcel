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

//! Model-owned-state family (second half) coverage of the `last_row_logits` hook (#3070): each case builds
//! a tiny random-weight model and runs the shared harness
//! (`last_row_logits_harness`), which asserts the hook is taken and returns
//! `[1, 1, vocab]` logits equal to the full forward's row.

use crate::models::last_row_logits_harness::last_row_case;
use crate::models::mixed_cache_batched_decode_tests as mixed;
use crate::vision::llmjp_vl_test_support as llmjp;

last_row_case!(ministral3_last_row_logits, mixed::ministral);

/// The Mellum fixture has an eight-entry vocabulary, so the shared prompt
/// (ids up to 9) does not fit; the window of 4 is shorter than the prompt.
#[test]
fn mellum_last_row_logits() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    crate::models::last_row_logits_harness::assert_last_row_logits(
        mixed::mellum,
        &[3, 1, 4, 1, 5, 2, 6, 5, 3, 7],
    );
}

last_row_case!(laguna_last_row_logits, mixed::laguna);

// The LLM-jp text arm is an enum over two dense backbones, and the VLM
// wrapper drops the sequence id before delegating to it.
last_row_case!(
    llmjp_llama_text_last_row_logits,
    || llmjp::build_text_model("llama"),
    embeddings
);
last_row_case!(
    llmjp_qwen3_text_last_row_logits,
    || llmjp::build_text_model("qwen3"),
    embeddings
);
last_row_case!(
    llmjp_vl_llama_last_row_logits,
    || llmjp::build_model("llama", 1e-6),
    embeddings
);
last_row_case!(
    llmjp_vl_qwen3_last_row_logits,
    || llmjp::build_model("qwen3", 1e-6),
    embeddings
);
