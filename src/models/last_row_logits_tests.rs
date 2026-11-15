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

//! Dense-family coverage of the `last_row_logits` hook (#3070): each case builds
//! a tiny random-weight model and runs the shared harness
//! (`last_row_logits_harness`), which asserts the hook is taken and returns
//! `[1, 1, vocab]` logits equal to the full forward's row.

use crate::models::last_row_logits_harness::last_row_case;
use crate::models::popular_dense_batched_decode_tests as dense;

last_row_case!(apertus_last_row_logits, dense::tiny_apertus_model);
last_row_case!(cohere_last_row_logits, dense::tiny_cohere_model);
last_row_case!(ernie45_last_row_logits, dense::tiny_ernie45_model);
last_row_case!(exaone_last_row_logits, dense::tiny_exaone_model);
last_row_case!(gemma_last_row_logits, dense::tiny_gemma_model, embeddings);
last_row_case!(glm4_last_row_logits, dense::tiny_glm4_model);
last_row_case!(
    granite_last_row_logits,
    dense::tiny_granite_model,
    embeddings
);
last_row_case!(hunyuan_v1_dense_last_row_logits, dense::tiny_hunyuan_model);
last_row_case!(internlm2_last_row_logits, dense::tiny_internlm2_model);
last_row_case!(internlm3_last_row_logits, dense::tiny_internlm3_model);
last_row_case!(mimo_last_row_logits, dense::tiny_mimo_model);
last_row_case!(olmo2_last_row_logits, dense::tiny_olmo2_model);
last_row_case!(seed_oss_last_row_logits, dense::tiny_seed_oss_model);
last_row_case!(smollm3_last_row_logits, dense::tiny_smollm3_model);
last_row_case!(starcoder2_last_row_logits, dense::tiny_starcoder2_model);
