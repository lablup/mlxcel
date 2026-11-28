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

//! MoE-family (second half) coverage of the `last_row_logits` hook (#3070): each case builds
//! a tiny random-weight model and runs the shared harness
//! (`last_row_logits_harness`), which asserts the hook is taken and returns
//! `[1, 1, vocab]` logits equal to the full forward's row.

use crate::models::last_row_logits_harness::last_row_case;
use crate::models::mixtral_style_batched_decode_tests as mixtral_style;

last_row_case!(olmoe_last_row_logits, mixtral_style::tiny_olmoe_model);
last_row_case!(phimoe_last_row_logits, mixtral_style::tiny_phimoe_model);
last_row_case!(
    qwen2_moe_last_row_logits,
    mixtral_style::tiny_qwen2_moe_model
);
last_row_case!(
    step3p5_last_row_logits,
    crate::models::mixed_cache_batched_decode_tests::step,
    embeddings
);
