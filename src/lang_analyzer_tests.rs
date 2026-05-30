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

//! Integration tests for `mlxcel_core::lang_analyzer`.
//!
//! These tests verify the public API of the lang_analyzer module from the
//! perspective of a downstream consumer (the main mlxcel crate). Unit-level
//! tests live in `mlxcel-core/src/lang_analyzer/mod.rs`; this file validates
//! the same invariants via the public API surface.

use mlxcel_core::lang_analyzer::{
    CURRENT_VERSION, Script, TokenLanguageIndex, cache, classify_token, is_numeric, is_punctuation,
    is_whitespace,
};
// Env-var-sensitive tests must serialize through the crate-wide `ENV_LOCK`
// per-module locks race with env mutations in unrelated
// modules of the same test binary.
use crate::test_support::env_lock::env_lock;

// ============================================================================
// B2 — classify_token — §10.1 acceptance criteria strings
// ============================================================================

#[test]
fn lang_analyze_pure_hangul() {
    let result = classify_token("한국어");
    assert_eq!(result.as_slice(), &[Script::Hangul]);
}

#[test]
fn lang_analyze_pure_han() {
    let result = classify_token("中文");
    assert_eq!(result.as_slice(), &[Script::Han]);
}

#[test]
fn lang_analyze_hangul_and_han() {
    let result = classify_token("韓국어");
    assert!(result.contains(&Script::Hangul));
    assert!(result.contains(&Script::Han));
    assert_eq!(result.len(), 2);
}

#[test]
fn lang_analyze_pure_hiragana() {
    let result = classify_token("ひらがな");
    assert_eq!(result.as_slice(), &[Script::Hiragana]);
}

#[test]
fn lang_analyze_pure_katakana() {
    let result = classify_token("カタカナ");
    assert_eq!(result.as_slice(), &[Script::Katakana]);
}

#[test]
fn lang_analyze_hiragana_and_han() {
    let result = classify_token("日本語のひらがな");
    assert!(result.contains(&Script::Hiragana));
    assert!(result.contains(&Script::Han));
    assert_eq!(result.len(), 2);
}

#[test]
fn lang_analyze_pure_latin() {
    let result = classify_token("hello");
    assert_eq!(result.as_slice(), &[Script::Latin]);
}

#[test]
fn lang_analyze_pure_cyrillic() {
    let result = classify_token("Привет");
    assert_eq!(result.as_slice(), &[Script::Cyrillic]);
}

#[test]
fn lang_analyze_numeric_returns_empty_scripts() {
    let result = classify_token("12345");
    assert!(
        result.is_empty(),
        "numeric string should yield no scripts; got {result:?}"
    );
}

#[test]
fn lang_analyze_numeric_is_numeric_flag() {
    assert!(is_numeric("12345"));
    assert!(!is_numeric("hello"));
}

#[test]
fn lang_analyze_punctuation_returns_empty_scripts() {
    let result = classify_token(",.!?");
    assert!(
        result.is_empty(),
        "punctuation string should yield no scripts; got {result:?}"
    );
}

#[test]
fn lang_analyze_punctuation_is_punctuation_flag() {
    assert!(is_punctuation(",.!?"));
    assert!(!is_punctuation("hello"));
}

#[test]
fn lang_analyze_whitespace_is_whitespace_flag() {
    assert!(is_whitespace("   "));
    assert!(!is_whitespace("hello"));
}

#[test]
fn lang_analyze_bpe_prefix_strips_prefix() {
    // ▁hello — ▁ is a SentencePiece BPE prefix, should be ignored.
    let result = classify_token("▁hello");
    assert_eq!(result.as_slice(), &[Script::Latin]);
}

#[test]
fn lang_analyze_bpe_prefix_alone_returns_empty() {
    let result = classify_token("▁");
    assert!(result.is_empty());
    assert!(is_whitespace("▁"), "BPE prefix should count as whitespace");
}

// ============================================================================
// B3 — TokenLanguageIndex helper
// ============================================================================

/// Minimal 100-token WordLevel tokenizer JSON for unit tests.
/// Contains 3 special tokens and 97 regular tokens, including some Latin
/// words and numeric strings for flag coverage.
const MOCK_TOKENIZER_100_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<s>",   "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "</s>",  "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "hello": 3, "world": 4, "test": 5, "the": 6, "and": 7,
      "123": 8, "456": 9, "789": 10,
      "token11": 11, "token12": 12, "token13": 13, "token14": 14,
      "token15": 15, "token16": 16, "token17": 17, "token18": 18,
      "token19": 19, "token20": 20, "token21": 21, "token22": 22,
      "token23": 23, "token24": 24, "token25": 25, "token26": 26,
      "token27": 27, "token28": 28, "token29": 29, "token30": 30,
      "token31": 31, "token32": 32, "token33": 33, "token34": 34,
      "token35": 35, "token36": 36, "token37": 37, "token38": 38,
      "token39": 39, "token40": 40, "token41": 41, "token42": 42,
      "token43": 43, "token44": 44, "token45": 45, "token46": 46,
      "token47": 47, "token48": 48, "token49": 49, "token50": 50,
      "token51": 51, "token52": 52, "token53": 53, "token54": 54,
      "token55": 55, "token56": 56, "token57": 57, "token58": 58,
      "token59": 59, "token60": 60, "token61": 61, "token62": 62,
      "token63": 63, "token64": 64, "token65": 65, "token66": 66,
      "token67": 67, "token68": 68, "token69": 69, "token70": 70,
      "token71": 71, "token72": 72, "token73": 73, "token74": 74,
      "token75": 75, "token76": 76, "token77": 77, "token78": 78,
      "token79": 79, "token80": 80, "token81": 81, "token82": 82,
      "token83": 83, "token84": 84, "token85": 85, "token86": 86,
      "token87": 87, "token88": 88, "token89": 89, "token90": 90,
      "token91": 91, "token92": 92, "token93": 93, "token94": 94,
      "token95": 95, "token96": 96, "token97": 97, "token98": 98,
      "token99": 99
    },
    "unk_token": "<unk>"
  }
}"#;

/// Slight variation of the mock tokenizer — different vocab hash.
const MOCK_TOKENIZER_ALT_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<pad>": 0, "alt_token": 1
    },
    "unk_token": "<pad>"
  }
}"#;

fn make_tokenizer(json: &str) -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("failed to parse mock tokenizer JSON")
}

// ============================================================================
// B3 — TokenLanguageIndex unit tests (§ acceptance criteria)
// ============================================================================

#[test]
fn build_populates_all_tokens() {
    let tok = make_tokenizer(MOCK_TOKENIZER_100_JSON);
    let idx = TokenLanguageIndex::build(&tok, MOCK_TOKENIZER_100_JSON.as_bytes())
        .expect("build should succeed");
    assert_eq!(idx.tokens.len(), 100, "expected 100 token entries");
}

#[test]
fn build_inverts_by_script() {
    let tok = make_tokenizer(MOCK_TOKENIZER_100_JSON);
    let idx = TokenLanguageIndex::build(&tok, MOCK_TOKENIZER_100_JSON.as_bytes())
        .expect("build should succeed");

    // For every script in by_script, verify each listed token id actually
    // has that script in its TokenScriptInfo.
    for (script, ids) in &idx.by_script {
        for &token_id in ids {
            let info = &idx.tokens[token_id as usize];
            assert!(
                info.scripts.contains(script),
                "token id {token_id} in by_script[{script:?}] but missing script in info.scripts"
            );
        }
    }
}

#[test]
fn build_flags_special_tokens() {
    let tok = make_tokenizer(MOCK_TOKENIZER_100_JSON);
    let idx = TokenLanguageIndex::build(&tok, MOCK_TOKENIZER_100_JSON.as_bytes())
        .expect("build should succeed");

    // Ids 0, 1, 2 are special tokens in the mock tokenizer.
    for special_id in [0u32, 1, 2] {
        assert!(
            idx.tokens[special_id as usize].is_special,
            "token id {special_id} should be flagged as special"
        );
    }
    // A regular token (id 3, "hello") should NOT be special.
    assert!(
        !idx.tokens[3].is_special,
        "token id 3 ('hello') should not be flagged as special"
    );
}

#[test]
fn build_vocab_hash_deterministic() {
    let tok = make_tokenizer(MOCK_TOKENIZER_100_JSON);
    let bytes = MOCK_TOKENIZER_100_JSON.as_bytes();
    let idx1 = TokenLanguageIndex::build(&tok, bytes).expect("first build failed");
    let idx2 = TokenLanguageIndex::build(&tok, bytes).expect("second build failed");
    assert_eq!(
        idx1.vocab_hash, idx2.vocab_hash,
        "vocab_hash must be deterministic for the same tokenizer"
    );
}

#[test]
fn build_vocab_hash_differs_across_tokenizers() {
    let tok1 = make_tokenizer(MOCK_TOKENIZER_100_JSON);
    let tok2 = make_tokenizer(MOCK_TOKENIZER_ALT_JSON);

    let idx1 = TokenLanguageIndex::build(&tok1, MOCK_TOKENIZER_100_JSON.as_bytes())
        .expect("first build failed");
    let idx2 = TokenLanguageIndex::build(&tok2, MOCK_TOKENIZER_ALT_JSON.as_bytes())
        .expect("second build failed");

    assert_ne!(
        idx1.vocab_hash, idx2.vocab_hash,
        "vocab_hash must differ for different tokenizer.json bytes"
    );
}

#[test]
fn version_constant_is_three() {
    // v1 → v2 bumped when classify switched from id_to_token to decode so
    // byte-level BPE tokenizers produce correct script assignments.
    // v2 → v3 bumped when TokenScriptInfo gained the
    // `is_byte_fragment` field and the opt-in byte-fragment classifier.
    assert_eq!(CURRENT_VERSION, 3, "CURRENT_VERSION must be 3");

    // Also verify the built index carries the correct version.
    let tok = make_tokenizer(MOCK_TOKENIZER_100_JSON);
    let idx =
        TokenLanguageIndex::build(&tok, MOCK_TOKENIZER_100_JSON.as_bytes()).expect("build failed");
    assert_eq!(idx.version, 3);
}

// ============================================================================
// B3 — Integration criterion: real tokenizer smoke test
// ============================================================================

/// Smoke test that loads a real tokenizer.json from a model in models/ and
/// builds a TokenLanguageIndex from it. This validates the full end-to-end
/// path that B4 (disk cache) and B5 (LangBiasSet conversion) will consume.
///
/// The test is skipped gracefully when no model directory is present so that
/// it does not break CI environments without downloaded models.
#[test]
fn build_index_from_real_tokenizer_smoke() {
    // Try multiple candidate tokenizer paths — use the first one found.
    let candidates = [
        "models/smollm-135m-4bit/tokenizer.json",
        "models/Qwen2.5-7B-Instruct-4bit/tokenizer.json",
        "models/Meta-Llama-3.1-8B-Instruct-4bit/tokenizer.json",
    ];

    let found = candidates.iter().find(|p| std::path::Path::new(p).exists());
    let Some(path) = found else {
        // No model downloaded — skip gracefully.
        eprintln!("[skip] no model tokenizer.json found; skipping real-tokenizer smoke test");
        return;
    };

    let bytes = std::fs::read(path).expect("failed to read tokenizer.json");
    let tok = tokenizers::Tokenizer::from_bytes(&bytes).expect("failed to parse tokenizer.json");

    let vocab_size = tok.get_vocab_size(true);
    assert!(vocab_size > 0, "tokenizer must have a non-empty vocabulary");

    let idx =
        TokenLanguageIndex::build(&tok, &bytes).expect("build should succeed on real tokenizer");

    assert_eq!(
        idx.tokens.len(),
        vocab_size,
        "index must have one entry per vocab token"
    );
    assert_eq!(idx.version, CURRENT_VERSION);
    assert_eq!(
        idx.vocab_hash.len(),
        16,
        "vocab_hash must be exactly 16 hex chars"
    );

    // Verify the by_script inversion is consistent.
    for (script, ids) in &idx.by_script {
        for &token_id in ids {
            let info = &idx.tokens[token_id as usize];
            assert!(
                info.scripts.contains(script),
                "real tokenizer: token id {token_id} in by_script[{script:?}] but missing from info.scripts"
            );
        }
    }

    eprintln!(
        "[ok] built index from {path}: vocab={vocab_size}, by_script entries={}",
        idx.by_script.len()
    );
}

/// Regression test: verifies the classifier correctly handles byte-level BPE.
///
/// Before v2, `build` called `id_to_token(id)` which returns the byte-mapped
/// pre-image for byte-level tokenizers (Qwen, GPT-2, LLaMA). Non-ASCII bytes
/// (0x80..0xFF) get mapped into Unicode Latin Extended-A codepoints, so every
/// non-ASCII token was misclassified as `Latin` and `by_script[Hangul]` was
/// empty for multilingual models. The feature silently did nothing.
///
/// v2 uses `decode(&[id], false)` which returns the logical UTF-8 string, so a
/// multilingual tokenizer like Qwen populates Hangul/Han/Hiragana correctly.
#[test]
fn real_multilingual_tokenizer_populates_non_latin_scripts() {
    let candidates = [
        "models/qwen2.5-0.5b-bf16/tokenizer.json",
        "models/qwen2.5-7b-4bit/tokenizer.json",
        "models/qwen3-0.6b/tokenizer.json",
        "models/Qwen2.5-7B-Instruct-4bit/tokenizer.json",
    ];
    let found = candidates.iter().find(|p| std::path::Path::new(p).exists());
    let Some(path) = found else {
        eprintln!(
            "[skip] no multilingual byte-level BPE tokenizer found; \
             skipping classifier regression"
        );
        return;
    };

    let bytes = std::fs::read(path).expect("read tokenizer.json");
    let tok = tokenizers::Tokenizer::from_bytes(&bytes).expect("parse tokenizer.json");
    let idx = TokenLanguageIndex::build(&tok, &bytes).expect("build index");

    let count = |s: mlxcel_core::lang_analyzer::Script| -> usize {
        idx.by_script.get(&s).map(Vec::len).unwrap_or(0)
    };
    let hangul = count(mlxcel_core::lang_analyzer::Script::Hangul);
    let han = count(mlxcel_core::lang_analyzer::Script::Han);
    let latin = count(mlxcel_core::lang_analyzer::Script::Latin);

    eprintln!(
        "[{path}] by_script: Hangul={hangul}, Han={han}, Latin={latin}, \
         total_tokens={}",
        idx.tokens.len()
    );

    // Qwen has ~150k vocab; multilingual support demands at least hundreds of
    // Hangul and Han tokens. If the classifier regresses to `id_to_token` we
    // would see 0.
    assert!(
        hangul >= 100,
        "expected ≥100 Hangul tokens on multilingual tokenizer; got {hangul} \
         (byte-level BPE classifier regression?)"
    );
    assert!(
        han >= 100,
        "expected ≥100 Han tokens on multilingual tokenizer; got {han} \
         (byte-level BPE classifier regression?)"
    );
}

// ============================================================================
// B4 — Disk cache integration tests
// ============================================================================

/// Integration test for B4 cache API exposed via the public `lang_analyzer`
/// surface. Exercises `cache::cache_path`, `cache::load_or_build`, and
/// `TokenLanguageIndex::compute_vocab_hash`.
///
/// Uses the mock tokenizer JSON to avoid requiring a downloaded model.
#[test]
fn b4_cache_integration_roundtrip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let _guard = env_lock();
    // SAFETY: serialized via ENV_LOCK; no other thread mutates MLXCEL_CACHE_DIR concurrently.
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());
    }

    let json = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0, "integration_test_b4": 1, "hello": 2
    },
    "unk_token": "<unk>"
  }
}"#;

    let json_bytes = json.as_bytes();
    let tok = tokenizers::Tokenizer::from_bytes(json_bytes).expect("parse tokenizer");

    // Verify compute_vocab_hash is consistent with the stored hash.
    let hash = TokenLanguageIndex::compute_vocab_hash(json_bytes);
    assert_eq!(hash.len(), 16, "vocab_hash must be 16 hex chars");

    // cache_path must resolve under the overridden directory.
    let expected_path = cache_path_relative(&tmp, &hash);
    let resolved = cache::cache_path(&hash);
    assert_eq!(resolved, expected_path);

    // load_or_build: first call must build and persist.
    let idx1 =
        cache::load_or_build(&tok, json_bytes, false).expect("first load_or_build should succeed");
    assert_eq!(idx1.version, CURRENT_VERSION);
    assert_eq!(idx1.vocab_hash, hash);
    assert!(
        resolved.exists(),
        "cache file must exist after first load_or_build"
    );

    let mtime1 = std::fs::metadata(&resolved).unwrap().modified().unwrap();

    // load_or_build: second call must load from disk (mtime unchanged).
    let idx2 =
        cache::load_or_build(&tok, json_bytes, false).expect("second load_or_build should succeed");
    let mtime2 = std::fs::metadata(&resolved).unwrap().modified().unwrap();

    // SAFETY: serialized via ENV_LOCK; no other thread mutates MLXCEL_CACHE_DIR concurrently.
    unsafe {
        std::env::remove_var("MLXCEL_CACHE_DIR");
    }
    drop(_guard);

    assert_eq!(idx1.vocab_hash, idx2.vocab_hash);
    assert_eq!(mtime1, mtime2, "second call must hit disk cache");

    eprintln!(
        "[ok] B4 cache integration: hash={hash}, path={}",
        resolved.display()
    );
}

/// Helper to build the expected cache path relative to a tempdir.
fn cache_path_relative(tmp: &tempfile::TempDir, hash: &str) -> std::path::PathBuf {
    tmp.path()
        .join("tokenizer-scripts")
        .join(format!("{hash}.bin"))
}

/// Integration smoke test: exercises the B4 cache with a real tokenizer.json.
/// Skipped when no model is available.
#[test]
fn b4_cache_real_tokenizer_integration_smoke() {
    let candidates = [
        "models/smollm-135m-4bit/tokenizer.json",
        "models/Qwen2.5-7B-Instruct-4bit/tokenizer.json",
        "models/Meta-Llama-3.1-8B-Instruct-4bit/tokenizer.json",
    ];

    let found = candidates.iter().find(|p| std::path::Path::new(p).exists());
    let Some(tok_path) = found else {
        eprintln!(
            "[skip] no model tokenizer.json found; skipping B4 real-tokenizer integration smoke test"
        );
        return;
    };

    let tmp = tempfile::tempdir().expect("tempdir");
    let json_bytes = std::fs::read(tok_path).expect("read tokenizer.json");
    let tok = tokenizers::Tokenizer::from_bytes(&json_bytes).expect("parse tokenizer");

    let _guard = env_lock();
    // SAFETY: serialized via ENV_LOCK; no other thread mutates MLXCEL_CACHE_DIR concurrently.
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());
    }

    let hash = TokenLanguageIndex::compute_vocab_hash(&json_bytes);
    let idx1 = cache::load_or_build(&tok, &json_bytes, false)
        .expect("first load_or_build on real tokenizer");

    assert_eq!(idx1.vocab_hash, hash);
    assert_eq!(idx1.version, CURRENT_VERSION);

    let path = cache::cache_path(&hash);
    assert!(path.exists(), "cache file must exist");
    let mtime1 = std::fs::metadata(&path).unwrap().modified().unwrap();

    let idx2 = cache::load_or_build(&tok, &json_bytes, false)
        .expect("second load_or_build on real tokenizer");
    let mtime2 = std::fs::metadata(&path).unwrap().modified().unwrap();

    // SAFETY: serialized via ENV_LOCK; no other thread mutates MLXCEL_CACHE_DIR concurrently.
    unsafe {
        std::env::remove_var("MLXCEL_CACHE_DIR");
    }
    drop(_guard);

    assert_eq!(idx1.vocab_hash, idx2.vocab_hash);
    assert_eq!(
        mtime1, mtime2,
        "real tokenizer: second call must hit disk cache"
    );

    eprintln!(
        "[ok] B4 real-tokenizer integration smoke: vocab={}, hash={}",
        idx1.tokens.len(),
        hash
    );
}

// ============================================================================
// B8 — LangBiasConfig::resolve_token_bias (generation-loop entry point)
// ============================================================================

use mlxcel_core::lang_analyzer::{
    ExceptionConfig, InclusionPolicy, LangBiasConfig, LangBiasSet, LanguageCode,
};

fn b8_mock_tokenizer_json(marker: &str) -> String {
    format!(
        r#"{{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {{"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
    {{"id": 1, "content": "<s>",   "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {{
    "type": "WordLevel",
    "vocab": {{
      "<unk>": 0, "<s>": 1,
      "hello": 2, "world": 3, "한국어": 4, "中文": 5,
      "{marker}": 6
    }},
    "unk_token": "<unk>"
  }}
}}"#
    )
}

/// Empty `LangBiasSet::ordered` must return an empty map and MUST NOT touch
/// the disk cache. Proof: point `MLXCEL_CACHE_DIR` at an empty tempdir and
/// assert the cache sub-directory is never created.
#[test]
fn b8_resolve_token_bias_empty_returns_empty() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let json = b8_mock_tokenizer_json("marker_b8_empty");
    let tok = tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("parse tokenizer");

    let config = LangBiasConfig::default();
    assert!(config.bias_set.ordered.is_empty());

    let _guard = env_lock();
    // SAFETY: serialized via ENV_LOCK.
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());
    }

    let map = config
        .resolve_token_bias(&tok, json.as_bytes())
        .expect("empty resolve must succeed without disk I/O");

    // Inspect the cache state before releasing the env override.
    let cache_subdir = tmp.path().join("tokenizer-scripts");
    let subdir_exists = cache_subdir.exists();
    let entry_count = std::fs::read_dir(&cache_subdir)
        .map(|rd| rd.count())
        .unwrap_or(0);

    // SAFETY: serialized via ENV_LOCK.
    unsafe {
        std::env::remove_var("MLXCEL_CACHE_DIR");
    }
    drop(_guard);

    assert!(
        map.is_empty(),
        "empty bias_set must yield empty TokenBiasMap"
    );
    assert!(
        !subdir_exists,
        "empty-bias resolve must NOT create the tokenizer-scripts cache dir"
    );
    assert_eq!(
        entry_count, 0,
        "empty-bias resolve must NOT write any cache files"
    );
}

/// Non-empty bias resolves from (and populates) the on-disk cache. Second
/// invocation hits the cache (mtime unchanged).
#[test]
fn b8_resolve_token_bias_populates_from_cache() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let json = b8_mock_tokenizer_json("marker_b8_populate");
    let tok = tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("parse tokenizer");

    let config = LangBiasConfig {
        bias_set: LangBiasSet {
            ordered: vec![
                (LanguageCode::Ja, f32::NEG_INFINITY),
                (LanguageCode::En, 1.5),
            ],
        },
        policy: InclusionPolicy::Conservative,
        exceptions: ExceptionConfig::default(),
        rebuild_cache: false,
    };

    let _guard = env_lock();
    // SAFETY: serialized via ENV_LOCK.
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());
    }

    // First call: cache miss -> build + persist.
    let map1 = config
        .resolve_token_bias(&tok, json.as_bytes())
        .expect("first resolve must succeed");

    let hash = TokenLanguageIndex::compute_vocab_hash(json.as_bytes());
    let cache_file = cache_path_relative(&tmp, &hash);
    let exists_after_first = cache_file.exists();
    let mtime1 = std::fs::metadata(&cache_file)
        .ok()
        .and_then(|m| m.modified().ok());

    // Second call: cache hit -> no rewrite.
    let map2 = config
        .resolve_token_bias(&tok, json.as_bytes())
        .expect("second resolve must succeed");
    let mtime2 = std::fs::metadata(&cache_file)
        .ok()
        .and_then(|m| m.modified().ok());

    // SAFETY: serialized via ENV_LOCK.
    unsafe {
        std::env::remove_var("MLXCEL_CACHE_DIR");
    }
    drop(_guard);

    assert!(
        exists_after_first,
        "first resolve must persist the cache file"
    );
    assert!(!map1.is_empty(), "non-empty bias must populate the map");
    assert_eq!(
        map1.len(),
        map2.len(),
        "cache-hit map must match cache-build map"
    );
    assert!(
        map1.iter().any(|(_id, &b)| b == f32::NEG_INFINITY),
        "ja=-inf entry must populate at least one Japanese-script token"
    );
    assert!(
        map1.iter().any(|(_id, &b)| b == 1.5),
        "en=+1.5 entry must populate at least one Latin-script token"
    );
    assert_eq!(
        mtime1, mtime2,
        "second call must not rewrite the cache file (disk hit expected)"
    );
}

// ============================================================================
// B8 — SpeculativeGenerator::with_token_bias (target-only)
// ============================================================================

use crate::SpeculativeGenerator;
use mlxcel_core::TokenBiasMap;

/// Default `SpeculativeGenerator` has no cached bias.
#[test]
fn b8_speculative_generator_default_bias_is_empty() {
    let g = SpeculativeGenerator::new(4, 2);
    assert!(g.token_bias().is_empty());
}

/// `with_token_bias` caches a map that is exposed via the inspector and
/// therefore reaches the target-only sampler path (see core unit tests in
/// `speculative.rs::tests::speculative_generator_passes_bias_to_target_only`
/// for the composition-level invariant).
#[test]
fn b8_speculative_generator_with_token_bias_caches_map() {
    let mut bias = TokenBiasMap::new();
    bias.insert(7, f32::NEG_INFINITY);
    bias.insert(11, 2.0);

    let g = SpeculativeGenerator::new(4, 2).with_token_bias(bias);
    assert_eq!(g.token_bias().len(), 2);
    assert!(g.token_bias().contains(7));
    assert!(g.token_bias().contains(11));
}

// ============================================================================
// B8 — CxxGenerator::with_token_bias
// ============================================================================

use crate::CxxGenerator;

#[test]
fn b8_cxx_generator_default_bias_is_empty() {
    let g = CxxGenerator::new(4);
    assert!(g.token_bias().is_empty());
}

#[test]
fn b8_cxx_generator_with_token_bias_caches_map() {
    let mut bias = TokenBiasMap::new();
    bias.insert(3, -1.5);
    let g = CxxGenerator::new(4).with_token_bias(bias);
    assert_eq!(g.token_bias().len(), 1);
    assert!(g.token_bias().contains(3));
}

// ============================================================================
// trimmable cache validation and last-token reservation
// ============================================================================

use mlxcel_core::cache::can_trim_prompt_cache;
use mlxcel_core::layers::KVCache;

/// All freshly-constructed KVCache entries report `is_trimmable() == true`.
/// This is the per-entry predicate consumed by `can_trim_prompt_cache`.
#[test]
fn issue589_kv_cache_is_trimmable_always_true() {
    // Empty cache
    let c = KVCache::new();
    assert!(c.is_trimmable());
}

/// `can_trim_prompt_cache` returns `true` for a standard slice of KVCaches.
#[test]
fn issue589_can_trim_prompt_cache_standard_caches() {
    let caches: Vec<KVCache> = (0..4).map(|_| KVCache::new()).collect();
    assert!(
        can_trim_prompt_cache(&caches),
        "standard KVCache slice must report trimmable"
    );
}

/// `can_trim_prompt_cache` returns `true` for an empty slice (vacuously true).
#[test]
fn issue589_can_trim_prompt_cache_empty_slice() {
    let caches: Vec<KVCache> = Vec::new();
    assert!(can_trim_prompt_cache(&caches));
}
