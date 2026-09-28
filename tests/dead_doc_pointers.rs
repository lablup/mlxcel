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

//! Issue #1658: `docs/model_tests.md` and `docs/testing.md` never existed in
//! this tree, yet six source comments and one assertion message pointed
//! readers at them. `src/main_tests.rs:252` (`supported_models_output_has_no_dead_doc_link`,
//! added for issue #26) guards the same class of defect for rendered CLI
//! output; this file guards it for the raw source text of the files that
//! carried the dead pointers, which is a check that file cannot perform.
//!
//! This is deliberately narrow: it re-checks only the two paths this issue
//! fixed, in only the files that referenced them. It is not a general
//! "every `docs/*.md` path mentioned anywhere in source must exist"
//! scanner; the tree has other, pre-existing dead doc references outside
//! this issue's scope (for example `docs/USAGE.md`,
//! `docs/bridge-overhead-microbench.md`) that a generic scanner would flag
//! and that belong to separate cleanup, not this regression guard.

const CARGO_TOML: &str = include_str!("../Cargo.toml");
const SPECULATIVE_BENCH: &str = include_str!("../src/bin/speculative_bench.rs");
const SPECULATIVE_PARITY: &str = include_str!("speculative_parity.rs");
const PROMPT_CACHE_PREFILL_BENCH: &str = include_str!("prompt_cache_prefill_bench.rs");
const TOKENIZER_MOD: &str = include_str!("../src/tokenizer/mod.rs");

const CHECKED_SOURCES: &[(&str, &str)] = &[
    ("Cargo.toml", CARGO_TOML),
    ("src/bin/speculative_bench.rs", SPECULATIVE_BENCH),
    ("tests/speculative_parity.rs", SPECULATIVE_PARITY),
    (
        "tests/prompt_cache_prefill_bench.rs",
        PROMPT_CACHE_PREFILL_BENCH,
    ),
    ("src/tokenizer/mod.rs", TOKENIZER_MOD),
];

#[test]
fn no_reference_to_the_nonexistent_docs_model_tests_md() {
    for (path, contents) in CHECKED_SOURCES {
        assert!(
            !contents.contains("docs/model_tests.md"),
            "{path} must not reference the nonexistent doc `docs/model_tests.md` \
             (issue #1658); the real file is `docs/benchmark_results/model_tests.md`"
        );
    }
}

#[test]
fn no_reference_to_the_nonexistent_docs_testing_md() {
    for (path, contents) in CHECKED_SOURCES {
        assert!(
            !contents.contains("docs/testing.md"),
            "{path} must not reference the nonexistent doc `docs/testing.md` (issue #1658)"
        );
    }
}
