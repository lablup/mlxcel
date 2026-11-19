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

//! Source roster for the last-row prefill projection (#3070).
//!
//! The engine prefill samples one position. A `LanguageModel` that neither
//! implements `last_row_logits` nor overrides `forward_last_logits` projects
//! every prompt row through its LM head, which at a 262K vocabulary is about
//! 1 GB per 2048-token piece. This test lists every implementation that still
//! does, with the reason, so a new model has to either take the hook or be
//! added here on purpose.

use std::fs;
use std::path::Path;

use super::batched_decode_coverage_tests::{matching_brace, relative_source_path, visit_rs_files};

/// Implementations that keep the full-logits default, `Type@path`, sorted.
const EXPECTED_FULL_LOGITS: &[&str] = &[
    // Masked-diffusion decoder: generation runs its own denoising loop over
    // the whole canvas, which reads every position's logits.
    "DiffusionGemmaModel@src/models/diffusion_gemma/mod.rs",
    // Encoder-decoder runtime: the trait forward exists for warmup and
    // tooling, generation goes through `Florence2VlmModel::run_task`.
    "Florence2VlmModel@src/models/florence2/runtime.rs",
    // Masked-diffusion decoder, as DiffusionGemma.
    "Llada2MoeModel@src/models/llada2_moe/mod.rs",
    // Encoder-decoder runtime: generation goes through
    // `NemotronParseVlmModel::run`.
    "NemotronParseVlmModel@src/models/nemotron_parse/runtime.rs",
];

fn full_logits_impls(root: &Path) -> Vec<String> {
    let needles = [
        "impl LanguageModel for ",
        "impl mlxcel_core::generate::LanguageModel for ",
    ];
    let mut found = Vec::new();
    for relative in ["src/models", "src/vision", "src/distributed"] {
        visit_rs_files(&root.join(relative), &mut |path| {
            let source = fs::read_to_string(path)
                .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
            let relative = relative_source_path(root, path);
            let mut cursor = 0;
            while let Some((start, needle)) = needles
                .iter()
                .filter_map(|needle| {
                    source[cursor..]
                        .find(needle)
                        .map(|offset| (cursor + offset, *needle))
                })
                .min_by_key(|(offset, _)| *offset)
            {
                let type_start = start + needle.len();
                let open = source[type_start..]
                    .find('{')
                    .map(|offset| type_start + offset)
                    .expect("impl body");
                let type_name = source[type_start..open]
                    .split_whitespace()
                    .next()
                    .expect("impl type");
                let close = matching_brace(&source, open);
                let block = &source[open + 1..close];
                cursor = close + 1;
                // A test double inside a production file is not a model.
                let in_test_module = source[..start].contains("#[cfg(test)]\nmod ");
                if !in_test_module
                    && !block.contains("fn last_row_logits(")
                    && !block.contains("fn forward_last_logits")
                {
                    found.push(format!("{type_name}@{relative}"));
                }
            }
        });
    }
    found.sort();
    found
}

#[test]
fn every_language_model_projects_only_the_sampled_prefill_row_or_is_listed() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert_eq!(
        full_logits_impls(root),
        EXPECTED_FULL_LOGITS,
        "a LanguageModel without last_row_logits projects every prefill row \
         through its LM head; implement the hook or list it with a reason"
    );
}
