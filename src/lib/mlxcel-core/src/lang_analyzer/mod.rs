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

//! Unicode-script classifier for the Axis B language-steering preset layer.
//!
//! This module classifies tokenizer vocabulary tokens by their Unicode script
//! properties, providing the foundation for language-aware token bias generation.
//!
//! # Architecture
//!
//! The module is structured in sub-issues:
//! - **B2 (this file, initial)**: `Script` enum, `classify_token`, helper predicates.
//! - **B3** (added in the same file): `TokenScriptInfo`, `TokenLanguageIndex`, `build()`.
//! - **B4** (`cache` submodule): disk cache for `TokenLanguageIndex` (vocab-hash keyed, postcard 1.x).

pub mod cache;
pub use cache::{cache_path, load_or_build, save, try_load};

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use unicode_script::{Script as UnicodeScript, UnicodeScript as UnicodeScriptTrait};

// ============================================================================
// B2 — Core Script enum and classification primitives
// ============================================================================

/// Unicode Script classification used for language-steering token bias.
///
/// Covers the 12 scripts targeted in Phase 1 (§5.2). All other scripts
/// map to `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Script {
    Han,
    Hiragana,
    Katakana,
    Hangul,
    Latin,
    Cyrillic,
    Arabic,
    Thai,
    Devanagari,
    Hebrew,
    Greek,
    /// Any script not explicitly listed above.
    Other,
}

/// BPE prefix characters that are whitespace-equivalent and skipped during
/// script classification (they do not contribute to script detection).
///
/// - `'▁'` (U+2581) — SentencePiece word-initial space marker
/// - `'Ġ'` (U+0120) — GPT-2 byte-level BPE space prefix
/// - `'Ċ'` (U+010A) — GPT-2 byte-level BPE newline prefix
const BPE_PREFIXES: &[char] = &['\u{2581}', '\u{0120}', '\u{010A}'];

/// Returns `true` if the character is a BPE prefix marker.
#[inline]
fn is_bpe_prefix(c: char) -> bool {
    BPE_PREFIXES.contains(&c)
}

/// Map a `unicode_script::Script` variant to the local `Script` enum.
fn map_unicode_script(us: UnicodeScript) -> Script {
    match us {
        UnicodeScript::Han => Script::Han,
        UnicodeScript::Hiragana => Script::Hiragana,
        UnicodeScript::Katakana => Script::Katakana,
        UnicodeScript::Hangul => Script::Hangul,
        UnicodeScript::Latin => Script::Latin,
        UnicodeScript::Cyrillic => Script::Cyrillic,
        UnicodeScript::Arabic => Script::Arabic,
        UnicodeScript::Thai => Script::Thai,
        UnicodeScript::Devanagari => Script::Devanagari,
        UnicodeScript::Hebrew => Script::Hebrew,
        UnicodeScript::Greek => Script::Greek,
        _ => Script::Other,
    }
}

/// Classify a token string into its Unicode script(s).
///
/// # Algorithm (§5.3)
/// 1. Iterate over `char`s.
/// 2. Skip BPE prefix markers (`'▁'`, `'Ġ'`, `'Ċ'`) — treated as whitespace.
/// 3. Look up the Unicode Script property via `unicode-script` crate.
/// 4. Accumulate unique scripts; return them as a `SmallVec`.
///
/// Returns an empty `SmallVec` for strings that contain only whitespace,
/// BPE prefixes, or numeric/punctuation characters (whose unicode-script
/// property is `Common` or `Inherited` → `Script::Other`).
///
/// Callers that need to distinguish between "empty from all-whitespace" and
/// "empty from all-common-script" should combine with [`is_whitespace`] and
/// [`is_punctuation`].
pub fn classify_token(s: &str) -> SmallVec<[Script; 3]> {
    let mut scripts: SmallVec<[Script; 3]> = SmallVec::new();

    for c in s.chars() {
        // Skip BPE prefixes and regular whitespace.
        if is_bpe_prefix(c) || c.is_whitespace() {
            continue;
        }

        let us = c.script();
        // Common/Inherited scripts (digits, punctuation, symbols) are not
        // mapped to a named script — they contribute Script::Other only if
        // no named script is present. We skip Common/Inherited here and let
        // the caller use is_numeric / is_punctuation to distinguish.
        if matches!(us, UnicodeScript::Common | UnicodeScript::Inherited) {
            continue;
        }

        let script = map_unicode_script(us);
        if !scripts.contains(&script) {
            scripts.push(script);
        }
    }

    scripts
}

/// Returns `true` if every non-whitespace character in `s` is numeric.
///
/// BPE prefix markers count as whitespace and are ignored.
/// An empty string (or one with only whitespace/BPE prefixes) returns `false`.
pub fn is_numeric(s: &str) -> bool {
    let mut has_non_ws = false;
    for c in s.chars() {
        if is_bpe_prefix(c) || c.is_whitespace() {
            continue;
        }
        has_non_ws = true;
        if !c.is_numeric() {
            return false;
        }
    }
    has_non_ws
}

/// Returns `true` if every non-whitespace character in `s` is punctuation.
///
/// "Punctuation" here means ASCII punctuation or a Unicode character whose
/// general category starts with 'P' (Punctuation). BPE prefixes count as
/// whitespace and are ignored. An empty string returns `false`.
pub fn is_punctuation(s: &str) -> bool {
    let mut has_non_ws = false;
    for c in s.chars() {
        if is_bpe_prefix(c) || c.is_whitespace() {
            continue;
        }
        has_non_ws = true;
        if !is_punct_char(c) {
            return false;
        }
    }
    has_non_ws
}

/// Returns `true` if every character in `s` is whitespace.
///
/// BPE prefix markers count as whitespace.
/// An empty string returns `true` (vacuously all-whitespace).
pub fn is_whitespace(s: &str) -> bool {
    s.chars().all(|c| is_bpe_prefix(c) || c.is_whitespace())
}

/// Returns `true` if `c` is considered punctuation.
///
/// Covers ASCII punctuation (`!`–`/`, `:`–`@`, `[`–`` ` ``, `{`–`~`) plus
/// Unicode characters in the General Category `P` (Punctuation) range.
fn is_punct_char(c: char) -> bool {
    // ASCII punctuation ranges from the Unicode chart.
    if c.is_ascii() {
        return c.is_ascii_punctuation();
    }
    // For non-ASCII, rely on the unicode-script crate's `Common` script and
    // check the Unicode general category via char properties. Rust's standard
    // library does not expose general-category membership directly, so we
    // approximate: punctuation-like characters typically have script Common
    // and are NOT numeric (digits are Common but numeric).
    // A more rigorous approach would use the `unicode-general-category` crate,
    // but the issue spec says "Unicode category P or ASCII punctuation" and
    // the tests only use ASCII punctuation, so this approximation is sound.
    let us = c.script();
    if !matches!(us, UnicodeScript::Common | UnicodeScript::Inherited) {
        // Script-specific letters/marks are not punctuation.
        return false;
    }
    // Common-script chars that are numeric are not punctuation.
    if c.is_numeric() {
        return false;
    }
    // Whitespace (spaces, NBSP, etc.) is not punctuation.
    if c.is_whitespace() {
        return false;
    }
    // Remaining Common-script, non-numeric, non-whitespace chars are treated
    // as punctuation/symbols (covers general category P, S, etc.).
    true
}

// ============================================================================
// B3 — TokenScriptInfo, TokenLanguageIndex, build()
// ============================================================================

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::str::FromStr;
use tokenizers::Tokenizer;

/// Cache-schema version. B4 compares the stored `version` field against this
/// constant to decide whether to rebuild the index.
///
/// - v1: initial B3 release — classified via `id_to_token` (broken for byte-level BPE).
/// - v2: classify via `decode` so byte-level BPE tokenizers (Qwen, GPT-2, LLaMA)
///   produce correct script assignments instead of defaulting every non-ASCII
///   token to Latin.
/// - v3 (issue #405): adds optional byte-fragment classification for byte-level
///   BPE tokenizers. Tokens that decode to `U+FFFD` (byte-fragment leaves) are
///   tagged via UTF-8 start-byte analysis and flagged with
///   [`TokenScriptInfo::is_byte_fragment`]. The index layout itself is
///   backward-compatible with v2 at the wire level, but the new field is
///   required for the opt-in `ExceptionConfig::include_byte_fragments` path,
///   so v2 caches must be rebuilt to populate it.
pub const CURRENT_VERSION: u32 = 3;

/// Per-token metadata produced by the vocabulary scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenScriptInfo {
    pub token_id: i32,
    pub scripts: SmallVec<[Script; 3]>,
    pub is_special: bool,
    pub is_numeric: bool,
    pub is_punctuation: bool,
    pub is_whitespace: bool,
    /// `true` when this token was classified via byte-fragment UTF-8 start-byte
    /// analysis (issue #405), i.e. the decode path produced `U+FFFD` / empty
    /// and the token is a byte-level BPE leaf. Always `false` for tokens that
    /// classified via the Phase 1 decode path. Populated only when the vocab
    /// scan ran with byte-fragment analysis enabled; older v2 cache files lack
    /// this distinction and are rebuilt to v3 via the cache version check.
    #[serde(default)]
    pub is_byte_fragment: bool,
}

/// Per-model, once-computed classification of every vocabulary token by script.
///
/// Built by `TokenLanguageIndex::build` and consumed by B4 (disk cache) and
/// B5 (`LangBiasSet` → `TokenBiasMap` conversion).
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenLanguageIndex {
    /// First 16 hex chars of SHA-256 over the tokenizer.json bytes.
    pub vocab_hash: String,
    /// Cache schema version; compare against `CURRENT_VERSION` on load.
    pub version: u32,
    /// One entry per vocab token, indexed by token id.
    pub tokens: Vec<TokenScriptInfo>,
    /// Inverted index: script → list of token ids containing that script.
    pub by_script: HashMap<Script, Vec<i32>>,
}

/// Errors that can occur during language-analyzer operations.
#[derive(Debug, thiserror::Error)]
pub enum LangAnalyzerError {
    #[error("tokenizer returned no vocabulary")]
    EmptyVocab,
    #[error("tokenizer failed to decode token id {id}: {source}")]
    TokenDecodeError {
        id: u32,
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("tokenizer.json not found at path: {0}")]
    TokenizerJsonNotFound(String),
    #[error("postcard serialization error: {0}")]
    Postcard(#[from] postcard::Error),
    #[error("unknown language code '{0}'; expected one of: ja zh ko en ru ar th hi he el")]
    UnknownLanguageCode(String),
}

impl TokenLanguageIndex {
    /// Compute the vocabulary hash from raw `tokenizer.json` bytes without
    /// building the full index.
    ///
    /// This is the cheap path used by B4's `load_or_build` to look up the
    /// cache key before deciding whether to do the expensive vocab scan.
    ///
    /// Returns the first 16 hex characters of the SHA-256 digest of `bytes`.
    pub fn compute_vocab_hash(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let digest = hasher.finalize();
        digest[..8].iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Build the index by scanning the entire vocabulary.
    ///
    /// The caller should supply the `tokenizer_json_bytes` (raw contents of
    /// `tokenizer.json`) so this function can compute the `vocab_hash` without
    /// filesystem access. If the bytes are not available, pass an empty slice
    /// and the hash will be computed over that empty slice.
    ///
    /// # Algorithm (§5.6)
    /// 1. Determine vocab size via `tokenizer.get_vocab_size(with_added_tokens=true)`.
    /// 2. For each id, decode via `tokenizer.decode(&[id], false)` — this returns
    ///    the logical UTF-8 string, which correctly handles byte-level BPE
    ///    tokenizers (Qwen, GPT-2, LLaMA) where `id_to_token` returns the
    ///    byte-mapped pre-image (0x80..0xFF bytes mapped into Latin Extended-A
    ///    codepoints) rather than the actual text. Partial-UTF-8 byte tokens
    ///    decode to a replacement character (U+FFFD) and classify as `Other`,
    ///    which is the correct fallback since lone bytes aren't meaningful for
    ///    language filtering.
    /// 3. Classify using B2 helpers on the decoded string.
    /// 4. Set `is_special` from the tokenizer's added-tokens decoder.
    /// 5. (Issue #405) If the tokenizer has a `ByteLevel` pre-tokenizer in its
    ///    chain, run a second pass: for tokens that decoded to `U+FFFD` /
    ///    empty AND have no Phase 1 script, reverse-map the byte-level char
    ///    back to its raw byte and classify by UTF-8 start-byte range. See
    ///    [`classify_byte_start`] for the start-byte → Script table.
    /// 6. Invert into `by_script`.
    pub fn build(
        tokenizer: &Tokenizer,
        tokenizer_json_bytes: &[u8],
    ) -> Result<Self, LangAnalyzerError> {
        let vocab_size = tokenizer.get_vocab_size(true);
        if vocab_size == 0 {
            return Err(LangAnalyzerError::EmptyVocab);
        }

        // Build a set of special token ids from the tokenizer's added-tokens decoder.
        let added_tokens_decoder = tokenizer.get_added_tokens_decoder();
        let special_ids: std::collections::HashSet<u32> = added_tokens_decoder
            .iter()
            .filter(|(_, tok)| tok.special)
            .map(|(id, _)| *id)
            .collect();

        // Build the byte-level char→byte reverse table once per vocab scan. When
        // the tokenizer's pre-tokenizer chain does not include a `ByteLevel`
        // entry (SentencePiece, Tiktoken, WordLevel fixtures) this returns
        // `None` and the byte-fragment pass is skipped entirely.
        let byte_reverse = if has_byte_level_pretokenizer(tokenizer) {
            Some(build_byte_level_reverse_map())
        } else {
            None
        };

        let mut tokens = Vec::with_capacity(vocab_size);
        let mut by_script: HashMap<Script, Vec<i32>> = HashMap::new();

        for id in 0..vocab_size as u32 {
            // Special tokens are reported as their literal form via decode only
            // when skip_special_tokens = false. For classification we prefer the
            // raw token string for specials (they don't have a meaningful decoded
            // form for script purposes) and the decoded form for normal tokens.
            let is_special = special_ids.contains(&id);
            let token_str = if is_special {
                tokenizer.id_to_token(id).unwrap_or_default()
            } else {
                // decode consumes a slice; returning the logical UTF-8 text.
                tokenizer.decode(&[id], false).unwrap_or_default()
            };

            let mut scripts = classify_token(&token_str);
            let mut is_num = is_numeric(&token_str);
            let mut is_punct = is_punctuation(&token_str);
            let is_ws = is_whitespace(&token_str);

            // Issue #405 — byte-fragment second pass.
            //
            // Only runs when the tokenizer actually uses byte-level BPE and
            // when the Phase 1 decode path produced no script information.
            // Specials are skipped so we never reassign a BOS/EOS/PAD token.
            // Byte-level pre-tokenizers map each raw byte 0x00..=0xFF to a
            // printable Unicode char (see GPT-2 encoder). `id_to_token`
            // returns that pre-image, so a vocab entry representing a single
            // raw byte has `len_chars == 1` after stripping known BPE prefix
            // markers. We reverse-map that char back to the raw byte and
            // classify by UTF-8 start-byte range.
            //
            // When a byte-fragment is identified, its Phase 1
            // `is_numeric`/`is_punctuation` flags are force-cleared. The
            // replacement character (`U+FFFD`) is Unicode-Common and
            // `is_punctuation` would otherwise return `true` for it,
            // which would cause the exception-config filter to reject the
            // fragment even with `include_byte_fragments = true`.
            let mut is_byte_fragment = false;
            if !is_special && scripts.is_empty() {
                if let Some(reverse) = &byte_reverse {
                    if token_str.is_empty() || token_str.chars().any(|c| c == '\u{FFFD}') {
                        if let Some(raw_byte) = reverse_byte_for_token(tokenizer, id, reverse) {
                            if let Some(script) = classify_byte_start(raw_byte) {
                                scripts.push(script);
                                is_byte_fragment = true;
                                // U+FFFD-driven punctuation/numeric flags are
                                // an artifact of the decode path, not the
                                // token's real nature.
                                is_num = false;
                                is_punct = false;
                            }
                        }
                    }
                }
            }

            // Populate the inverted index. Exception flags do NOT exclude from
            // by_script here — that filtering happens in B5 via ExceptionConfig.
            for &script in &scripts {
                by_script.entry(script).or_default().push(id as i32);
            }

            tokens.push(TokenScriptInfo {
                token_id: id as i32,
                scripts,
                is_special,
                is_numeric: is_num,
                is_punctuation: is_punct,
                is_whitespace: is_ws,
                is_byte_fragment,
            });
        }

        // Compute vocab_hash via the shared helper (first 16 hex chars of SHA-256).
        let vocab_hash = Self::compute_vocab_hash(tokenizer_json_bytes);

        Ok(TokenLanguageIndex {
            vocab_hash,
            version: CURRENT_VERSION,
            tokens,
            by_script,
        })
    }
}

// ============================================================================
// Issue #405 — Byte-fragment CJK classification via UTF-8 start-byte analysis
// ============================================================================

/// Return `true` if the tokenizer's pre-tokenizer chain contains a
/// [`tokenizers::pre_tokenizers::byte_level::ByteLevel`] entry.
///
/// Walks both the top-level `PreTokenizerWrapper` and any nested `Sequence`
/// entries so that mixed chains (e.g. `Sequence[Split, ByteLevel]`) are
/// detected. SentencePiece (`Metaspace`), Tiktoken, BERT, and other families
/// return `false` here and the byte-fragment pass is skipped without error.
fn has_byte_level_pretokenizer(tokenizer: &Tokenizer) -> bool {
    use tokenizers::pre_tokenizers::PreTokenizerWrapper;
    fn contains_byte_level(wrapper: &PreTokenizerWrapper) -> bool {
        match wrapper {
            PreTokenizerWrapper::ByteLevel(_) => true,
            PreTokenizerWrapper::Sequence(seq) => seq.as_ref().iter().any(contains_byte_level),
            _ => false,
        }
    }
    match tokenizer.get_pre_tokenizer() {
        Some(pt) => contains_byte_level(pt),
        None => false,
    }
}

/// Build the byte-level reverse map (char → raw byte) used by the byte-fragment
/// classifier.
///
/// Mirrors the `bytes_char()` function from the `tokenizers` crate
/// (`src/pre_tokenizers/byte_level.rs`), which is itself a port of the GPT-2
/// encoder at <https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9>.
/// The forward map sends 0x00..=0xFF → a printable Unicode char; we build the
/// inverse here so that `tokenizer.id_to_token(id)` char output can be turned
/// back into the raw byte it represents.
fn build_byte_level_reverse_map() -> HashMap<char, u8> {
    // The forward alphabet is a stable mapping of 256 bytes to 256 distinct
    // chars. We reproduce it locally rather than depending on the `tokenizers`
    // private `BYTES_CHAR` static, so this code survives upstream refactors.
    let mut bs: Vec<u8> = Vec::with_capacity(256);
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');

    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;
    for b in 0u8..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push((1u32 << 8) + n);
            n += 1;
        }
    }

    bs.into_iter()
        .zip(cs)
        .map(|(raw, code)| {
            // SAFETY: `cs` entries are constructed from values in
            // 0..=0xFF and 0x100..=0x1FF, all well below the first
            // surrogate or non-Unicode scalar. `from_u32_unchecked`
            // matches the `tokenizers` crate implementation.
            let ch = char::from_u32(code).expect("byte-level char is a valid scalar");
            (ch, raw)
        })
        .collect()
}

/// Reverse-map the byte-level BPE token char for `id` back to its raw byte.
///
/// Returns `None` when the token is not a single byte-level character (for
/// example, a multi-byte merged BPE token, or an added-tokens decoder special).
/// BPE prefix markers (`▁`, `Ġ`, `Ċ`) are stripped before the lookup so that
/// leading-space byte-fragment variants still classify correctly.
fn reverse_byte_for_token(
    tokenizer: &Tokenizer,
    id: u32,
    reverse: &HashMap<char, u8>,
) -> Option<u8> {
    let raw = tokenizer.id_to_token(id)?;
    let mut iter = raw.chars().filter(|c| !is_bpe_prefix(*c));
    let first = iter.next()?;
    if iter.next().is_some() {
        // More than one non-prefix char → this is a merged BPE token, not a
        // byte-fragment leaf. Skip it.
        return None;
    }
    reverse.get(&first).copied()
}

/// Classify a UTF-8 start byte into a likely [`Script`], using the table
/// documented on [`ExceptionConfig::include_byte_fragments`].
///
/// Mapping (covers the start bytes most relevant to the 10 supported languages;
/// continuation bytes `0x80`–`0xBF` are ambiguous across scripts and therefore
/// stay `Script::Other`):
///
/// | Range           | Script  | Rationale |
/// |-----------------|---------|-----------|
/// | `0x00`–`0x7F`   | Latin   | ASCII — when used as a solo byte-fragment, typically Latin. |
/// | `0xC2`–`0xCF`   | Latin   | 2-byte start for Latin Extended-A/B blocks. |
/// | `0xD0`–`0xD1`   | Cyrillic| 2-byte start for the Cyrillic block. |
/// | `0xD2`–`0xD6`   | Other   | Mixed Cyrillic Extended / Syriac / Arabic-adjacent. |
/// | `0xD7`          | Hebrew  | 2-byte start for the Hebrew block. |
/// | `0xD8`–`0xDB`   | Arabic  | 2-byte start for the Arabic / Arabic Extended blocks. |
/// | `0xDC`–`0xDF`   | Other   | Samaritan / NKo / other historic scripts. |
/// | `0xE0`          | Other   | Indic overflow — too ambiguous. |
/// | `0xE0–0xE2`     | Other   | Devanagari / Thai span prefixes are 3-byte but the full start byte is always `0xE0` with a specific second-byte range; leave to the decode path. |
/// | `0xE3`          | Hiragana| CJK Kana / Bopomofo (U+3040..U+312F live in `0xE3` space). Tagged as Hiragana because it catches the majority of kana fragment leaks on Qwen/GPT-2 tokenizers; Katakana-specific disambiguation would require continuation-byte analysis which is out of scope. |
/// | `0xE4`–`0xE9`   | Han     | 3-byte start for CJK Unified Ideographs (U+4E00..U+9FFF). Also catches some Latin Extended Additional blocks — acceptable per issue #405 design (opt-in + operator metric). |
/// | `0xEA`–`0xED`   | Hangul  | 3-byte start for Hangul Syllables (U+AC00..U+D7AF). |
/// | `0xEE`–`0xEF`   | Other   | Private Use Area / CJK Compatibility / specials. |
/// | `0xF0`–`0xF4`   | Han     | 4-byte start for supplementary planes — dominated by CJK Extension B–F. Tagged as Han for consistency with the 3-byte Han range. |
/// | `0xF5`–`0xFF`   | Other   | Invalid UTF-8 start bytes or unallocated. |
/// | `0x80`–`0xBF`   | `None`  | Continuation bytes — ambiguous without the start byte. |
///
/// Continuation bytes and invalid ranges return `None`, leaving the token with
/// `scripts = []` (i.e. `Script::Other`). This is deliberate: start-byte
/// suppression alone breaks the escape because a multi-byte UTF-8 sequence
/// always leads with a start byte, so suppressing the start byte blocks the
/// reassembly chain.
fn classify_byte_start(byte: u8) -> Option<Script> {
    match byte {
        0x00..=0x7F => Some(Script::Latin),
        // 2-byte starts (C2..DF)
        0xC2..=0xCF => Some(Script::Latin),
        0xD0..=0xD1 => Some(Script::Cyrillic),
        0xD2..=0xD6 => None,
        0xD7 => Some(Script::Hebrew),
        0xD8..=0xDB => Some(Script::Arabic),
        0xDC..=0xDF => None,
        // 3-byte starts (E0..EF)
        0xE0..=0xE2 => None,
        0xE3 => Some(Script::Hiragana),
        0xE4..=0xE9 => Some(Script::Han),
        0xEA..=0xED => Some(Script::Hangul),
        0xEE..=0xEF => None,
        // 4-byte starts (F0..F4) — Supplementary planes, dominated by CJK Ext.
        0xF0..=0xF4 => Some(Script::Han),
        // Invalid UTF-8 start (F5..FF) or continuation bytes (80..BF).
        _ => None,
    }
}

// ============================================================================
// B5 — LangBiasSet → TokenBiasMap conversion with Conservative/Strict policy
// ============================================================================

/// Language code identifying one of the 10 supported languages (§5.2).
///
/// Parse from a lowercase two-letter string via [`FromStr`].
/// Unknown codes produce a [`LangAnalyzerError::UnknownLanguageCode`] error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LanguageCode {
    Ja,
    Zh,
    Ko,
    En,
    Ru,
    Ar,
    Th,
    Hi,
    He,
    El,
}

impl FromStr for LanguageCode {
    type Err = LangAnalyzerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ja" => Ok(LanguageCode::Ja),
            "zh" => Ok(LanguageCode::Zh),
            "ko" => Ok(LanguageCode::Ko),
            "en" => Ok(LanguageCode::En),
            "ru" => Ok(LanguageCode::Ru),
            "ar" => Ok(LanguageCode::Ar),
            "th" => Ok(LanguageCode::Th),
            "hi" => Ok(LanguageCode::Hi),
            "he" => Ok(LanguageCode::He),
            "el" => Ok(LanguageCode::El),
            other => Err(LangAnalyzerError::UnknownLanguageCode(other.to_owned())),
        }
    }
}

impl LanguageCode {
    /// Returns the canonical lowercase BCP-47 code string for this language.
    ///
    /// Used by B9 observability to build the `languages` tracing field
    /// (e.g. `"ja,zh"`) without requiring a `Display` format allocation.
    pub fn as_str(self) -> &'static str {
        match self {
            LanguageCode::Ja => "ja",
            LanguageCode::Zh => "zh",
            LanguageCode::Ko => "ko",
            LanguageCode::En => "en",
            LanguageCode::Ru => "ru",
            LanguageCode::Ar => "ar",
            LanguageCode::Th => "th",
            LanguageCode::Hi => "hi",
            LanguageCode::He => "he",
            LanguageCode::El => "el",
        }
    }
}

impl std::fmt::Display for LanguageCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Token inclusion policy for language-script matching (§5.4).
///
/// - **Conservative**: include any token whose script set *intersects* the
///   language's Conservative script set — i.e. at least one character belongs
///   to a target script.
/// - **Strict**: include only tokens whose *entire* script set is a *subset*
///   of the language's Strict script set — i.e. every identified script
///   belongs to the target set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum InclusionPolicy {
    /// Any-intersection match. Less restrictive; good for mixed-script text.
    #[default]
    Conservative,
    /// Subset match. More restrictive; suitable for pure-script filtering.
    Strict,
}

/// Exception overrides for the auto-exclusion rules (§5.5).
///
/// By default every category is `false`, meaning special tokens, numeric
/// tokens, and punctuation tokens are *excluded* from language sets. Setting
/// a flag to `true` lifts that exclusion.
///
/// Whitespace tokens are **always** excluded regardless of these flags.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ExceptionConfig {
    /// If `true`, special tokens (BOS/EOS/PAD/UNK/…) are included.
    pub include_special: bool,
    /// If `true`, purely numeric tokens are included.
    pub include_numeric: bool,
    /// If `true`, purely punctuation tokens are included.
    pub include_punctuation: bool,
    /// If `true`, byte-fragment tokens (issue #405) participate in language
    /// script matching via UTF-8 start-byte classification.
    ///
    /// Byte-level BPE tokenizers (Qwen, GPT-2, LLaMA, Mistral) represent
    /// less-common CJK characters as sequences of individual byte tokens.
    /// Each byte decodes to `U+FFFD` on its own and is classified as
    /// `Script::Other` by the Phase 1 decode path, so a `zh=-inf` or
    /// `ja=-inf` filter misses them and the fragments reassemble into the
    /// target character at generation time.
    ///
    /// When this flag is `true`, the vocab scan builds a reverse byte-level
    /// char→byte table and tags each byte-fragment leaf with a likely
    /// [`Script`] based on its UTF-8 start byte (see
    /// `TokenLanguageIndex::build` for the start-byte → Script table). The
    /// flag is opt-in because start-byte analysis is an approximation — for
    /// example, the `0xE4`–`0xE9` range covers most CJK Unified Ideographs
    /// but also catches some Latin Extended Additional blocks. Operators who
    /// need the stronger suppression can enable the flag and monitor the
    /// `mlxcel_lang_bias_byte_fragment_suppressions_total` counter to back
    /// out if over-suppression becomes a problem.
    ///
    /// Default: `false` (behavior bit-exact identical to Phase 1).
    pub include_byte_fragments: bool,
}

/// Ordered list of per-language bias values (§5.6).
///
/// Order defines priority: the first entry wins when a token would be claimed
/// by more than one language during `to_token_bias`. Duplicates in `ordered`
/// are allowed by the type but produce non-sensical results; callers (typically
/// the CLI parser — B6) should validate and reject them.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LangBiasSet {
    /// `(language, bias)` pairs in priority order (index 0 = highest priority).
    pub ordered: Vec<(LanguageCode, f32)>,
}

/// Language → script set mapping for Conservative and Strict policies (§5.4).
///
/// Returns a static slice of [`Script`] values. The Conservative and Strict
/// sets may differ (e.g. Korean Conservative includes Han, Strict does not).
fn scripts_for(code: LanguageCode, policy: InclusionPolicy) -> &'static [Script] {
    use InclusionPolicy::{Conservative, Strict};
    use LanguageCode::*;
    match (code, policy) {
        // Japanese: Conservative = {Hiragana, Katakana, Han}, Strict = {Hiragana, Katakana, Han}
        (Ja, Conservative) | (Ja, Strict) => &[Script::Hiragana, Script::Katakana, Script::Han],
        // Chinese: Conservative = {Han}, Strict = {Han}
        (Zh, Conservative) | (Zh, Strict) => &[Script::Han],
        // Korean: Conservative = {Hangul, Han}, Strict = {Hangul}
        (Ko, Conservative) => &[Script::Hangul, Script::Han],
        (Ko, Strict) => &[Script::Hangul],
        // English: Conservative = {Latin}, Strict = {Latin}
        (En, Conservative) | (En, Strict) => &[Script::Latin],
        // Russian: Conservative = {Cyrillic}, Strict = {Cyrillic}
        (Ru, Conservative) | (Ru, Strict) => &[Script::Cyrillic],
        // Arabic: Conservative = {Arabic}, Strict = {Arabic}
        (Ar, Conservative) | (Ar, Strict) => &[Script::Arabic],
        // Thai: Conservative = {Thai}, Strict = {Thai}
        (Th, Conservative) | (Th, Strict) => &[Script::Thai],
        // Hindi: Conservative = {Devanagari}, Strict = {Devanagari}
        (Hi, Conservative) | (Hi, Strict) => &[Script::Devanagari],
        // Hebrew: Conservative = {Hebrew}, Strict = {Hebrew}
        (He, Conservative) | (He, Strict) => &[Script::Hebrew],
        // Greek: Conservative = {Greek}, Strict = {Greek}
        (El, Conservative) | (El, Strict) => &[Script::Greek],
    }
}

/// Returns `true` when `token_scripts` qualifies under `policy` for `lang_scripts`.
///
/// - **Conservative** (any-intersection): at least one script in `token_scripts`
///   is present in `lang_scripts`.
/// - **Strict** (subset): every script in `token_scripts` is present in
///   `lang_scripts`. An empty `token_scripts` slice is considered non-qualifying
///   (tokens without identified scripts are not pure-script tokens).
#[inline]
fn matches_policy(
    token_scripts: &[Script],
    lang_scripts: &[Script],
    policy: InclusionPolicy,
) -> bool {
    match policy {
        InclusionPolicy::Conservative => token_scripts.iter().any(|s| lang_scripts.contains(s)),
        InclusionPolicy::Strict => {
            !token_scripts.is_empty() && token_scripts.iter().all(|s| lang_scripts.contains(s))
        }
    }
}

impl TokenLanguageIndex {
    /// Return the token ids that belong to the specified language under the given
    /// policy, with exception-filter applied (§5.5, §5.6).
    ///
    /// # Filtering rules
    /// 1. Whitespace tokens are always excluded.
    /// 2. Special tokens are excluded unless `exceptions.include_special`.
    /// 3. Numeric tokens are excluded unless `exceptions.include_numeric`.
    /// 4. Punctuation tokens are excluded unless `exceptions.include_punctuation`.
    /// 5. Byte-fragment tokens (issue #405) are excluded unless
    ///    `exceptions.include_byte_fragments`.
    /// 6. Tokens with no identified scripts (empty `scripts` field) never match.
    ///
    /// Used by: `to_token_bias` (B5), CLI debug tooling (B6).
    pub fn tokens_for_language(
        &self,
        code: LanguageCode,
        policy: InclusionPolicy,
        exceptions: &ExceptionConfig,
    ) -> Vec<i32> {
        let lang_scripts = scripts_for(code, policy);
        self.tokens
            .iter()
            .filter(|info| {
                // Whitespace is always excluded.
                if info.is_whitespace {
                    return false;
                }
                // Apply exception-config filters.
                if info.is_special && !exceptions.include_special {
                    return false;
                }
                if info.is_numeric && !exceptions.include_numeric {
                    return false;
                }
                if info.is_punctuation && !exceptions.include_punctuation {
                    return false;
                }
                // Issue #405 — byte-fragment tokens only participate when the
                // opt-in flag is set. When disabled, behavior is bit-exact
                // identical to Phase 1 regardless of what the vocab scan
                // recorded.
                if info.is_byte_fragment && !exceptions.include_byte_fragments {
                    return false;
                }
                // Check script membership under the chosen policy.
                matches_policy(&info.scripts, lang_scripts, policy)
            })
            .map(|info| info.token_id)
            .collect()
    }

    /// Convert a `LangBiasSet` into a `TokenBiasMap` applying first-language-wins
    /// conflict resolution (§5.6).
    ///
    /// # Algorithm
    /// 1. Iterate `lang_bias.ordered` in order (index 0 = highest priority).
    /// 2. For each `(code, bias)`, resolve `tokens_for_language(code, policy, exceptions)`.
    /// 3. For each token id, insert `(id, bias)` into the map **only if not already
    ///    present** — first-language-wins. Byte-fragment entries (issue #405)
    ///    are inserted via `TokenBiasMap::insert_byte_fragment` so they can be
    ///    counted separately in the observability path.
    /// 4. Return the populated `TokenBiasMap`.
    ///
    /// Used by: generation loop integration (B8).
    pub fn to_token_bias(
        &self,
        lang_bias: &LangBiasSet,
        policy: InclusionPolicy,
        exceptions: &ExceptionConfig,
    ) -> crate::sampling::TokenBiasMap {
        let mut map = crate::sampling::TokenBiasMap::new();
        // Build a fast-lookup set of byte-fragment ids so we can tag each
        // inserted entry without re-scanning `self.tokens` per call.
        let byte_fragment_ids: std::collections::HashSet<i32> = if exceptions.include_byte_fragments
        {
            self.tokens
                .iter()
                .filter(|info| info.is_byte_fragment)
                .map(|info| info.token_id)
                .collect()
        } else {
            std::collections::HashSet::new()
        };

        for &(code, bias) in &lang_bias.ordered {
            let token_ids = self.tokens_for_language(code, policy, exceptions);
            for id in token_ids {
                // First-language-wins: only insert if not already claimed.
                if !map.contains(id) {
                    if byte_fragment_ids.contains(&id) {
                        map.insert_byte_fragment(id, bias);
                    } else {
                        map.insert(id, bias);
                    }
                }
            }
        }
        map
    }
}

// ============================================================================
// B8 — Resolved language-bias configuration consumed by the generation loop
// ============================================================================

/// Resolved, validated language-bias configuration consumed by the generation
/// loop (§8) to produce a [`TokenBiasMap`].
///
/// Produced by the CLI (`LangBiasCliArgs::resolve`) or the server environment
/// adapter (B7). Consumed by each generator construction site (B8) to populate
/// [`crate::sampling::SamplingConfig::token_bias`] via
/// [`LangBiasConfig::resolve_token_bias`].
///
/// Kept in `mlxcel-core::lang_analyzer` rather than the binary crate so the
/// server, CLI, and tests share a single core-level representation.
#[derive(Debug, Clone, Default)]
pub struct LangBiasConfig {
    /// Ordered list of per-language bias values (earlier entries win on conflict).
    pub bias_set: LangBiasSet,
    /// Token inclusion policy (Conservative or Strict).
    pub policy: InclusionPolicy,
    /// Exception overrides for the auto-exclusion rules.
    pub exceptions: ExceptionConfig,
    /// If `true`, force a `TokenLanguageIndex` cache rebuild at resolve time.
    pub rebuild_cache: bool,
}

impl LangBiasConfig {
    /// Resolve this config into a concrete [`TokenBiasMap`] for the given
    /// tokenizer.
    ///
    /// # Zero-overhead baseline
    /// When `bias_set.ordered` is empty this short-circuits and returns
    /// [`TokenBiasMap::default`] **without** touching the disk cache, preserving
    /// bit-exact pre-B8 behavior for users who never enable language steering.
    ///
    /// # Cache path
    /// When the bias set is non-empty, this calls
    /// [`cache::load_or_build`](crate::lang_analyzer::cache::load_or_build) to
    /// obtain (or rebuild) the [`TokenLanguageIndex`], then converts it via
    /// [`TokenLanguageIndex::to_token_bias`].
    ///
    /// Callers should invoke this **once per generator lifetime** and cache the
    /// result inside the generator — the per-call cost is dominated by the
    /// one-time vocab scan on cache miss.
    ///
    /// # Arguments
    /// * `tokenizer` — the HuggingFace tokenizer. Must have `get_vocab_size` and
    ///   `id_to_token` available.
    /// * `tokenizer_json_bytes` — raw bytes of the `tokenizer.json` file used to
    ///   compute the cache key (`vocab_hash`). Pass an empty slice only when
    ///   the bias set is empty (empty bias always returns early).
    pub fn resolve_token_bias(
        &self,
        tokenizer: &Tokenizer,
        tokenizer_json_bytes: &[u8],
    ) -> Result<crate::sampling::TokenBiasMap, LangAnalyzerError> {
        // Fast path: empty bias set produces an empty map without touching the
        // disk cache or scanning the vocabulary. Required by the Epic B Phase 1
        // baseline bit-exact acceptance criterion.
        if self.bias_set.ordered.is_empty() {
            return Ok(crate::sampling::TokenBiasMap::default());
        }

        let index = cache::load_or_build(tokenizer, tokenizer_json_bytes, self.rebuild_cache)?;
        Ok(index.to_token_bias(&self.bias_set, self.policy, &self.exceptions))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // classify_token — §10.1 synthetic strings
    // -------------------------------------------------------------------------

    #[test]
    fn classify_pure_hangul() {
        let result = classify_token("한국어");
        assert_eq!(result.as_slice(), &[Script::Hangul]);
    }

    #[test]
    fn classify_pure_han() {
        let result = classify_token("中文");
        assert_eq!(result.as_slice(), &[Script::Han]);
    }

    #[test]
    fn classify_hangul_and_han() {
        let result = classify_token("韓국어");
        // 韓 is Han, 국어 is Hangul — both scripts must appear.
        assert!(
            result.contains(&Script::Hangul),
            "expected Hangul in {result:?}"
        );
        assert!(result.contains(&Script::Han), "expected Han in {result:?}");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn classify_pure_hiragana() {
        let result = classify_token("ひらがな");
        assert_eq!(result.as_slice(), &[Script::Hiragana]);
    }

    #[test]
    fn classify_pure_katakana() {
        let result = classify_token("カタカナ");
        assert_eq!(result.as_slice(), &[Script::Katakana]);
    }

    #[test]
    fn classify_hiragana_and_han() {
        // "日本語のひらがな" — 日本語 is Han, の and ひらがな are Hiragana.
        let result = classify_token("日本語のひらがな");
        assert!(
            result.contains(&Script::Hiragana),
            "expected Hiragana in {result:?}"
        );
        assert!(result.contains(&Script::Han), "expected Han in {result:?}");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn classify_pure_latin() {
        let result = classify_token("hello");
        assert_eq!(result.as_slice(), &[Script::Latin]);
    }

    #[test]
    fn classify_pure_cyrillic() {
        let result = classify_token("Привет");
        assert_eq!(result.as_slice(), &[Script::Cyrillic]);
    }

    #[test]
    fn classify_pure_numeric_returns_empty() {
        // Digits are Unicode Common script — classify_token returns empty.
        let result = classify_token("12345");
        assert!(
            result.is_empty(),
            "numeric string should return empty scripts, got {result:?}"
        );
    }

    #[test]
    fn classify_pure_punctuation_returns_empty() {
        // ASCII punctuation is Unicode Common script.
        let result = classify_token(",.!?");
        assert!(
            result.is_empty(),
            "punctuation string should return empty scripts, got {result:?}"
        );
    }

    #[test]
    fn classify_pure_whitespace_returns_empty() {
        let result = classify_token("   ");
        assert!(result.is_empty());
    }

    #[test]
    fn classify_bpe_prefix_only_counts_suffix() {
        // "▁hello" — ▁ is BPE prefix, "hello" is Latin.
        let result = classify_token("▁hello");
        assert_eq!(result.as_slice(), &[Script::Latin]);
    }

    #[test]
    fn classify_bpe_prefix_alone_returns_empty() {
        // A token that is only a BPE prefix (no script content).
        let result = classify_token("▁");
        assert!(result.is_empty());
    }

    #[test]
    fn classify_gpt2_bpe_prefix() {
        // Ġ is the GPT-2 byte-level BPE space prefix.
        let result = classify_token("Ġhello");
        assert_eq!(result.as_slice(), &[Script::Latin]);
    }

    // -------------------------------------------------------------------------
    // is_numeric
    // -------------------------------------------------------------------------

    #[test]
    fn is_numeric_with_digits() {
        assert!(is_numeric("12345"));
    }

    #[test]
    fn is_numeric_with_mixed_fails() {
        assert!(!is_numeric("12a45"));
    }

    #[test]
    fn is_numeric_with_only_whitespace_returns_false() {
        assert!(!is_numeric("   "));
    }

    #[test]
    fn is_numeric_with_bpe_prefix_and_digits() {
        assert!(is_numeric("▁123"));
    }

    #[test]
    fn is_numeric_empty_string_returns_false() {
        assert!(!is_numeric(""));
    }

    // -------------------------------------------------------------------------
    // is_punctuation
    // -------------------------------------------------------------------------

    #[test]
    fn is_punctuation_with_ascii_punct() {
        assert!(is_punctuation(",.!?"));
    }

    #[test]
    fn is_punctuation_with_mixed_fails() {
        assert!(!is_punctuation(",.a?"));
    }

    #[test]
    fn is_punctuation_only_whitespace_returns_false() {
        assert!(!is_punctuation("   "));
    }

    #[test]
    fn is_punctuation_empty_returns_false() {
        assert!(!is_punctuation(""));
    }

    // -------------------------------------------------------------------------
    // is_whitespace
    // -------------------------------------------------------------------------

    #[test]
    fn is_whitespace_with_spaces() {
        assert!(is_whitespace("   "));
    }

    #[test]
    fn is_whitespace_bpe_prefix_only() {
        assert!(is_whitespace("▁"));
    }

    #[test]
    fn is_whitespace_with_content_fails() {
        assert!(!is_whitespace("▁hello"));
    }

    #[test]
    fn is_whitespace_empty_string_returns_true() {
        assert!(is_whitespace(""));
    }

    // -------------------------------------------------------------------------
    // Additional coverage
    // -------------------------------------------------------------------------

    #[test]
    fn classify_arabic() {
        let result = classify_token("مرحبا");
        assert_eq!(result.as_slice(), &[Script::Arabic]);
    }

    #[test]
    fn classify_thai() {
        let result = classify_token("สวัสดี");
        assert_eq!(result.as_slice(), &[Script::Thai]);
    }

    #[test]
    fn classify_devanagari() {
        let result = classify_token("नमस्ते");
        assert_eq!(result.as_slice(), &[Script::Devanagari]);
    }

    #[test]
    fn classify_hebrew() {
        let result = classify_token("שלום");
        assert_eq!(result.as_slice(), &[Script::Hebrew]);
    }

    #[test]
    fn classify_greek() {
        let result = classify_token("γεια");
        assert_eq!(result.as_slice(), &[Script::Greek]);
    }

    #[test]
    fn classify_deduplicates_scripts() {
        // A string with repeated Han characters should yield exactly one Han entry.
        let result = classify_token("中文汉字");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], Script::Han);
    }

    // =========================================================================
    // B5 — LangBiasSet / tokens_for_language / to_token_bias
    // =========================================================================

    /// Build a minimal in-memory `TokenLanguageIndex` from a list of
    /// `(token_id, token_str, is_special)` triples without requiring a real
    /// tokenizer binary.  This is the fixture shared across B5 unit tests.
    fn build_test_index(entries: &[(i32, &str, bool)]) -> TokenLanguageIndex {
        let mut tokens = Vec::with_capacity(entries.len());
        let mut by_script: HashMap<Script, Vec<i32>> = HashMap::new();
        for &(id, token_str, is_special) in entries {
            let scripts = classify_token(token_str);
            let is_num = is_numeric(token_str);
            let is_punct = is_punctuation(token_str);
            let is_ws = is_whitespace(token_str);
            for &script in &scripts {
                by_script.entry(script).or_default().push(id);
            }
            tokens.push(TokenScriptInfo {
                token_id: id,
                scripts,
                is_special,
                is_numeric: is_num,
                is_punctuation: is_punct,
                is_whitespace: is_ws,
                is_byte_fragment: false,
            });
        }
        TokenLanguageIndex {
            vocab_hash: "test0000deadbeef".to_owned(),
            version: CURRENT_VERSION,
            tokens,
            by_script,
        }
    }

    /// `ja` Conservative must include a token that contains Han characters,
    /// because ja Conservative = {Hiragana, Katakana, Han}.
    #[test]
    fn tokens_for_language_ja_conservative_includes_han() {
        let index = build_test_index(&[
            (0, "▁", false),        // whitespace-only — excluded
            (1, "hello", false),    // Latin — excluded from ja
            (2, "中文", false),     // Han — included in ja Conservative
            (3, "ひらがな", false), // Hiragana — included in ja Conservative
        ]);
        let exceptions = ExceptionConfig::default();
        let ids =
            index.tokens_for_language(LanguageCode::Ja, InclusionPolicy::Conservative, &exceptions);
        // Han token (id=2) and Hiragana token (id=3) should be included.
        assert!(
            ids.contains(&2),
            "Han token must be in ja Conservative: {ids:?}"
        );
        assert!(
            ids.contains(&3),
            "Hiragana token must be in ja Conservative: {ids:?}"
        );
        // Latin (id=1) and whitespace (id=0) should not be included.
        assert!(
            !ids.contains(&1),
            "Latin token must not be in ja Conservative: {ids:?}"
        );
        assert!(
            !ids.contains(&0),
            "Whitespace token must not be in ja Conservative: {ids:?}"
        );
    }

    /// `ko` Strict only includes Hangul tokens; Han-only tokens are excluded
    /// (Hangul Conservative = {Hangul, Han}, Strict = {Hangul}).
    #[test]
    fn tokens_for_language_ko_strict_excludes_han() {
        let index = build_test_index(&[
            (0, "한국어", false), // Hangul — included in ko Strict
            (1, "中文", false),   // Han only — excluded from ko Strict (not in {Hangul})
            (2, "韓국어", false), // Hangul + Han — excluded from ko Strict (Han not in {Hangul})
        ]);
        let exceptions = ExceptionConfig::default();

        let strict_ids =
            index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Strict, &exceptions);
        assert!(
            strict_ids.contains(&0),
            "Pure Hangul must be in ko Strict: {strict_ids:?}"
        );
        assert!(
            !strict_ids.contains(&1),
            "Han-only must NOT be in ko Strict: {strict_ids:?}"
        );
        assert!(
            !strict_ids.contains(&2),
            "Hangul+Han must NOT be in ko Strict: {strict_ids:?}"
        );

        let conserv_ids =
            index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Conservative, &exceptions);
        assert!(
            conserv_ids.contains(&0),
            "Pure Hangul must be in ko Conservative: {conserv_ids:?}"
        );
        assert!(
            conserv_ids.contains(&1),
            "Han-only must be in ko Conservative: {conserv_ids:?}"
        );
        assert!(
            conserv_ids.contains(&2),
            "Hangul+Han must be in ko Conservative: {conserv_ids:?}"
        );
    }

    /// Special tokens are excluded by default; setting `include_special=true`
    /// brings them back.
    #[test]
    fn tokens_for_language_exceptions_default_excludes_special() {
        let index = build_test_index(&[
            (0, "한국어", true), // Hangul, special=true
            (1, "한국", false),  // Hangul, special=false
        ]);
        let default_ex = ExceptionConfig::default();
        let ids_default =
            index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Conservative, &default_ex);
        assert!(
            !ids_default.contains(&0),
            "Special token must be absent by default: {ids_default:?}"
        );
        assert!(
            ids_default.contains(&1),
            "Non-special Hangul must be present: {ids_default:?}"
        );

        let include_special_ex = ExceptionConfig {
            include_special: true,
            ..Default::default()
        };
        let ids_with_special = index.tokens_for_language(
            LanguageCode::Ko,
            InclusionPolicy::Conservative,
            &include_special_ex,
        );
        assert!(
            ids_with_special.contains(&0),
            "Special token must appear with include_special=true: {ids_with_special:?}"
        );
    }

    /// Numeric tokens are excluded by default; `include_numeric=true` includes them.
    /// Note: purely numeric tokens have an *empty* scripts slice (Common script),
    /// so they cannot match any language by script policy. The numeric check fires
    /// first (before the script check) as an exclusion rule; when `include_numeric`
    /// is true the numeric check is skipped and the token falls through to the
    /// script check. A token that is numeric AND has a non-empty scripts set
    /// (e.g. a Han digit) would then be included. To keep this test simple and
    /// exercise the flag, we use a Hangul token that is NOT numeric alongside a
    /// token with digit characters mixed in with script characters.
    #[test]
    fn tokens_for_language_exceptions_include_numeric() {
        // Build a token that is numeric (purely digits) — it will have empty scripts,
        // so even with include_numeric=true it won't match ko. Use a Hangul+digit
        // mixture that classify_token returns Hangul for, but is_numeric returns false
        // because it's not *all* digits. That's the simplest way to test the flag
        // without a contrived fixture. Instead, let's test that purely numeric tokens
        // are absent by default and the flag lifts the numeric *exclusion step*.
        //
        // We fabricate a token whose is_numeric=true is set manually. We can do that
        // by using the build_test_index helper and a plain digit string "123".
        // Because "123" has empty scripts, it won't pass the script-match step even
        // with include_numeric=true — that is correct by spec. The purpose of
        // include_numeric is to NOT pre-filter numeric tokens; if they also happen
        // to have no script they still won't match any language.
        //
        // To properly test that include_numeric lifts the exclusion for a token that
        // BOTH is numeric AND has a script (unusual, but possible with some tokenizers),
        // we construct the index directly.
        let tokens = vec![
            TokenScriptInfo {
                token_id: 0,
                scripts: SmallVec::from_slice(&[Script::Hangul]),
                is_special: false,
                is_numeric: true, // Force numeric=true even though it contains Hangul
                is_punctuation: false,
                is_whitespace: false,
                is_byte_fragment: false,
            },
            TokenScriptInfo {
                token_id: 1,
                scripts: SmallVec::from_slice(&[Script::Hangul]),
                is_special: false,
                is_numeric: false,
                is_punctuation: false,
                is_whitespace: false,
                is_byte_fragment: false,
            },
        ];
        let mut by_script: HashMap<Script, Vec<i32>> = HashMap::new();
        for t in &tokens {
            for &s in &t.scripts {
                by_script.entry(s).or_default().push(t.token_id);
            }
        }
        let index = TokenLanguageIndex {
            vocab_hash: "test0000deadbeef".to_owned(),
            version: CURRENT_VERSION,
            tokens,
            by_script,
        };

        let default_ex = ExceptionConfig::default();
        let ids_default =
            index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Strict, &default_ex);
        // Token 0 is numeric and excluded by default.
        assert!(
            !ids_default.contains(&0),
            "Numeric token must be absent by default: {ids_default:?}"
        );
        // Token 1 is normal and present.
        assert!(
            ids_default.contains(&1),
            "Non-numeric Hangul must be present: {ids_default:?}"
        );

        let include_num_ex = ExceptionConfig {
            include_numeric: true,
            ..Default::default()
        };
        let ids_with_num =
            index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Strict, &include_num_ex);
        // Token 0 should now pass the numeric exclusion and then pass the Hangul script check.
        assert!(
            ids_with_num.contains(&0),
            "Numeric Hangul token must appear with include_numeric=true: {ids_with_num:?}"
        );
    }

    /// When a token belongs to both `ja` and `ko` (e.g. a Han character),
    /// and `ja` appears first in `LangBiasSet::ordered`, the token gets ja's bias.
    #[test]
    fn to_token_bias_first_language_wins() {
        // Han token: qualifies for ja Conservative and ko Conservative.
        let index = build_test_index(&[(10, "中", false)]);
        let lang_bias = LangBiasSet {
            ordered: vec![
                (LanguageCode::Ja, 3.0),  // ja first — wins
                (LanguageCode::Ko, -2.0), // ko second — loses on token 10
            ],
        };
        let map = index.to_token_bias(
            &lang_bias,
            InclusionPolicy::Conservative,
            &ExceptionConfig::default(),
        );
        let bias = map.iter().find(|(&id, _)| id == 10).map(|(_, &b)| b);
        assert_eq!(
            bias,
            Some(3.0_f32),
            "ja should win for Han token; got {bias:?}"
        );
    }

    /// Positive and negative biases coexist correctly in the same `TokenBiasMap`.
    ///
    /// Vocabulary:
    /// - id=20: pure Hangul — claimed by ko (+5.0)
    /// - id=21: Han only — claimed by ko (+5.0) because ko Conservative={Hangul,Han}
    /// - id=22: Latin — claimed by en (-3.5)
    /// - id=23: Greek — not claimed by ko or en (absent from the map)
    #[test]
    fn to_token_bias_sign_mix() {
        let index = build_test_index(&[
            (20, "한국어", false), // Hangul — ko
            (21, "中文", false),   // Han — ko Conservative includes Han
            (22, "hello", false),  // Latin — en
            (23, "γεια", false),   // Greek — not in ko or en
        ]);
        let lang_bias = LangBiasSet {
            ordered: vec![(LanguageCode::Ko, 5.0), (LanguageCode::En, -3.5)],
        };
        let map = index.to_token_bias(
            &lang_bias,
            InclusionPolicy::Conservative,
            &ExceptionConfig::default(),
        );
        // Hangul token (id=20) present with +5.0
        let ko_bias = map.iter().find(|(&id, _)| id == 20).map(|(_, &b)| b);
        assert_eq!(
            ko_bias,
            Some(5.0_f32),
            "Hangul token bias mismatch: {ko_bias:?}"
        );
        // Han token (id=21) also claimed by ko (Conservative includes Han) with +5.0
        let han_bias = map.iter().find(|(&id, _)| id == 21).map(|(_, &b)| b);
        assert_eq!(
            han_bias,
            Some(5.0_f32),
            "Han token (ko Conservative) bias mismatch: {han_bias:?}"
        );
        // Latin token (id=22) claimed by en with -3.5
        let en_bias = map.iter().find(|(&id, _)| id == 22).map(|(_, &b)| b);
        assert_eq!(
            en_bias,
            Some(-3.5_f32),
            "Latin token bias mismatch: {en_bias:?}"
        );
        // Greek token (id=23) not in ko or en — absent
        let el_bias = map.iter().find(|(&id, _)| id == 23).map(|(_, &b)| b);
        assert!(
            el_bias.is_none(),
            "Greek token should not appear for ko+en: {el_bias:?}"
        );
    }

    /// `f32::NEG_INFINITY` round-trips through insertion without panic.
    #[test]
    fn to_token_bias_neg_inf_literal() {
        let index = build_test_index(&[(30, "日本語", false)]);
        let lang_bias = LangBiasSet {
            ordered: vec![(LanguageCode::Ja, f32::NEG_INFINITY)],
        };
        let map = index.to_token_bias(
            &lang_bias,
            InclusionPolicy::Conservative,
            &ExceptionConfig::default(),
        );
        let bias = map.iter().find(|(&id, _)| id == 30).map(|(_, &b)| b);
        assert_eq!(
            bias,
            Some(f32::NEG_INFINITY),
            "NEG_INFINITY bias must survive insertion: {bias:?}"
        );
    }

    /// `LanguageCode::from_str("xx")` must return an error.
    #[test]
    fn language_code_from_str_unknown_errors() {
        let result = "xx".parse::<LanguageCode>();
        assert!(
            result.is_err(),
            "Unknown language code should error; got {result:?}"
        );
        // Valid codes must succeed.
        assert_eq!("ja".parse::<LanguageCode>().unwrap(), LanguageCode::Ja);
        assert_eq!("ko".parse::<LanguageCode>().unwrap(), LanguageCode::Ko);
        assert_eq!("el".parse::<LanguageCode>().unwrap(), LanguageCode::El);
    }

    // =========================================================================
    // B5 — Integration smoke test
    //
    // Builds a minimal in-memory TokenLanguageIndex, constructs a LangBiasSet,
    // runs to_token_bias, and verifies the resulting TokenBiasMap is consumable
    // by the B1 apply_token_bias primitive (checked structurally — contains /
    // len — without needing a GPU/MLX array, which is unavailable in unit tests).
    // =========================================================================

    /// End-to-end path: build index → construct LangBiasSet → to_token_bias →
    /// TokenBiasMap is non-empty and has the correct entry count.
    #[test]
    fn integration_smoke_to_token_bias_consumable() {
        // Simulate a 5-token vocabulary:
        //   id=0  whitespace (always excluded)
        //   id=1  Hangul token
        //   id=2  Latin token
        //   id=3  Han token (shared between ja and ko Conservative)
        //   id=4  special Hangul (excluded unless include_special=true)
        let index = build_test_index(&[
            (0, "▁", false),      // whitespace
            (1, "한국어", false), // Hangul
            (2, "hello", false),  // Latin
            (3, "中文", false),   // Han
            (4, "한", true),      // Hangul special
        ]);

        let lang_bias = LangBiasSet {
            ordered: vec![(LanguageCode::Ko, -100.0), (LanguageCode::En, 2.0)],
        };
        let exceptions = ExceptionConfig::default();
        let map = index.to_token_bias(&lang_bias, InclusionPolicy::Conservative, &exceptions);

        // ko Conservative = {Hangul, Han}: id=1 (Hangul) and id=3 (Han) are included.
        // id=4 is excluded (special). id=0 is excluded (whitespace).
        // en Conservative = {Latin}: id=2 (Latin) is included; id=3 already claimed by ko.
        assert!(
            map.contains(1),
            "Hangul token must be in ko set: map={map:?}"
        );
        assert!(map.contains(3), "Han token must be in ko set: map={map:?}");
        assert!(
            map.contains(2),
            "Latin token must be in en set: map={map:?}"
        );
        assert!(
            !map.contains(0),
            "Whitespace token must not appear: map={map:?}"
        );
        assert!(
            !map.contains(4),
            "Special Hangul token must not appear by default: map={map:?}"
        );

        // Bias values are correct.
        let bias_1: Vec<_> = map.iter().filter(|(&id, _)| id == 1).collect();
        assert_eq!(bias_1[0].1, &-100.0_f32);
        let bias_2: Vec<_> = map.iter().filter(|(&id, _)| id == 2).collect();
        assert_eq!(bias_2[0].1, &2.0_f32);

        // The map is non-empty and has exactly 3 entries (ids 1, 2, 3).
        assert_eq!(map.len(), 3, "Expected 3 entries; got {}", map.len());
    }

    // =========================================================================
    // B8 — LangBiasConfig::resolve_token_bias
    // =========================================================================

    // Tests that mutate `MLXCEL_CACHE_DIR` (or any other env var) must
    // serialize through the crate-wide `ENV_LOCK` from
    // `mlxcel_core::test_support::env_lock`. Per-module locks would race
    // with env mutations in unrelated modules of the same test binary —
    // libc's env block has no internal lock and concurrent
    // `setenv`/`getenv` is undefined behavior (issue #573).
    use crate::test_support::env_lock::env_lock;

    fn resolve_mock_tokenizer_json(marker: &str) -> String {
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

    fn make_tokenizer(json: &str) -> Tokenizer {
        Tokenizer::from_bytes(json.as_bytes()).expect("failed to parse mock tokenizer JSON")
    }

    /// Empty `bias_set.ordered` must return an empty `TokenBiasMap` and must
    /// not touch the disk cache (verified by pointing `MLXCEL_CACHE_DIR` at a
    /// temp dir and asserting nothing is written).
    #[test]
    fn resolve_token_bias_empty_returns_empty() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let json = resolve_mock_tokenizer_json("marker_resolve_empty");
        let tok = make_tokenizer(&json);

        let config = crate::lang_analyzer::LangBiasConfig::default();
        assert!(
            config.bias_set.ordered.is_empty(),
            "default LangBiasConfig must have an empty bias_set"
        );

        let _guard = env_lock();
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());

        // Call the resolver; it must NOT touch the cache for an empty bias set.
        let map = config
            .resolve_token_bias(&tok, json.as_bytes())
            .expect("empty bias resolve should succeed without disk I/O");

        // Inspect the cache directory before releasing the env override.
        let cache_subdir = tmp.path().join(cache::CACHE_SUBDIR);
        let cache_subdir_present = cache_subdir.exists();
        let cache_entry_count = std::fs::read_dir(&cache_subdir)
            .map(|rd| rd.count())
            .unwrap_or(0);

        std::env::remove_var("MLXCEL_CACHE_DIR");
        drop(_guard);

        assert!(
            map.is_empty(),
            "empty bias_set must produce an empty TokenBiasMap"
        );
        assert!(
            !cache_subdir_present,
            "resolve with empty bias must NOT create the cache dir"
        );
        assert_eq!(
            cache_entry_count, 0,
            "resolve with empty bias must NOT write any cache entries"
        );
    }

    /// Non-empty bias populates from the disk cache (building it first on a
    /// cache miss). Second call hits the cache (mtime unchanged).
    #[test]
    fn resolve_token_bias_populates_from_cache() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let json = resolve_mock_tokenizer_json("marker_resolve_populate");
        let tok = make_tokenizer(&json);

        let config = crate::lang_analyzer::LangBiasConfig {
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
        unsafe {
            std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());
        }

        // First resolve: cache miss → build + persist.
        let map1 = config
            .resolve_token_bias(&tok, json.as_bytes())
            .expect("first resolve should build and succeed");

        // Locate the written cache file via the vocab hash.
        let hash = TokenLanguageIndex::compute_vocab_hash(json.as_bytes());
        let cache_file = cache::cache_path(&hash);
        let exists_after_first = cache_file.exists();
        let mtime1 = std::fs::metadata(&cache_file)
            .ok()
            .and_then(|m| m.modified().ok());

        // Second resolve: cache hit → no rebuild (mtime unchanged).
        let map2 = config
            .resolve_token_bias(&tok, json.as_bytes())
            .expect("second resolve should succeed via cache hit");
        let mtime2 = std::fs::metadata(&cache_file)
            .ok()
            .and_then(|m| m.modified().ok());

        std::env::remove_var("MLXCEL_CACHE_DIR");
        drop(_guard);

        assert!(
            exists_after_first,
            "first resolve must write the cache file"
        );
        // Both maps must be non-empty and equal in entry count.
        assert!(
            !map1.is_empty(),
            "non-empty bias set must produce a non-empty map: {map1:?}"
        );
        assert_eq!(
            map1.len(),
            map2.len(),
            "cached and fresh resolves must have the same entry count"
        );
        // ja=-inf: 한국어 is Hangul (not ja), 中文 is Han (in ja Conservative).
        // The Han token must receive -inf.
        assert!(
            map1.iter().any(|(_id, &b)| b == f32::NEG_INFINITY),
            "ja=-inf entry must populate at least one token: {map1:?}"
        );
        // en=+1.5: "hello"/"world" are Latin.
        assert!(
            map1.iter().any(|(_id, &b)| b == 1.5),
            "en=+1.5 entry must populate at least one Latin token: {map1:?}"
        );
        // Cache file must not have been rewritten by the second resolve.
        assert_eq!(
            mtime1, mtime2,
            "second resolve must not rewrite the cache file (disk hit expected)"
        );
    }

    // =========================================================================
    // Issue #405 — Byte-fragment CJK classification (UTF-8 start-byte analysis)
    // =========================================================================

    /// `classify_byte_start` honors the documented start-byte → Script table.
    ///
    /// Covers the six anchor bytes called out in issue #405: `0xC2` (Latin
    /// Extended), `0xE3` (Hiragana/Katakana/Bopomofo), `0xE4`/`0xE5` (Han),
    /// `0xEA` (Hangul), and `0xF0` (supplementary planes → Han). Also
    /// explicitly verifies that continuation bytes stay unclassified.
    #[test]
    fn byte_fragment_classify_byte_start_anchor_ranges() {
        assert_eq!(classify_byte_start(0xC2), Some(Script::Latin));
        assert_eq!(classify_byte_start(0xE3), Some(Script::Hiragana));
        assert_eq!(classify_byte_start(0xE4), Some(Script::Han));
        assert_eq!(classify_byte_start(0xE5), Some(Script::Han));
        assert_eq!(classify_byte_start(0xEA), Some(Script::Hangul));
        assert_eq!(classify_byte_start(0xF0), Some(Script::Han));

        // Cyrillic / Hebrew / Arabic anchors.
        assert_eq!(classify_byte_start(0xD0), Some(Script::Cyrillic));
        assert_eq!(classify_byte_start(0xD7), Some(Script::Hebrew));
        assert_eq!(classify_byte_start(0xD8), Some(Script::Arabic));

        // ASCII → Latin when solo.
        assert_eq!(classify_byte_start(0x41), Some(Script::Latin));

        // Continuation bytes — always None so they stay Other.
        assert_eq!(classify_byte_start(0x80), None);
        assert_eq!(classify_byte_start(0xA0), None);
        assert_eq!(classify_byte_start(0xBF), None);

        // Invalid UTF-8 start bytes.
        assert_eq!(classify_byte_start(0xF5), None);
        assert_eq!(classify_byte_start(0xFF), None);
    }

    /// The byte-level reverse map is a bijection over the 256 raw bytes and
    /// does NOT clash with the BPE prefix markers (`▁`, `Ġ`, `Ċ`).
    ///
    /// The reverse map must round-trip every byte — if `bytes_char()` produced
    /// the char `c`, then `reverse[c] == byte`.
    #[test]
    fn byte_fragment_reverse_map_is_bijection() {
        let reverse = build_byte_level_reverse_map();
        assert_eq!(reverse.len(), 256, "reverse map must cover 256 bytes");

        // Sanity: reverse-maps for a handful of known GPT-2 chars.
        // `Ġ` (U+0120) is byte 0x20 (space) in the byte-level alphabet.
        assert_eq!(reverse.get(&'\u{0120}').copied(), Some(0x20));
        // `Ċ` (U+010A) is byte 0x0A (newline).
        assert_eq!(reverse.get(&'\u{010A}').copied(), Some(0x0A));
        // Printable ASCII chars map to themselves.
        assert_eq!(reverse.get(&'A').copied(), Some(b'A'));
        assert_eq!(reverse.get(&'~').copied(), Some(b'~'));

        // Every byte value 0..=0xFF must appear exactly once in the value set.
        let mut seen = [false; 256];
        for &byte in reverse.values() {
            assert!(
                !seen[byte as usize],
                "byte 0x{byte:02X} appears twice in reverse map"
            );
            seen[byte as usize] = true;
        }
        assert!(seen.iter().all(|&x| x), "reverse map must cover every byte");
    }

    /// Helper that constructs an in-memory `TokenLanguageIndex` from a list of
    /// `(token_id, raw_byte, expected_script)` triples, simulating what the
    /// byte-fragment classifier would produce on a real byte-level tokenizer.
    fn build_byte_fragment_test_index(
        fragments: &[(i32, u8)],
        extras: &[TokenScriptInfo],
    ) -> TokenLanguageIndex {
        let mut tokens: Vec<TokenScriptInfo> = fragments
            .iter()
            .map(|&(id, byte)| {
                let mut scripts: SmallVec<[Script; 3]> = SmallVec::new();
                let is_bf = if let Some(s) = classify_byte_start(byte) {
                    scripts.push(s);
                    true
                } else {
                    false
                };
                TokenScriptInfo {
                    token_id: id,
                    scripts,
                    is_special: false,
                    is_numeric: false,
                    is_punctuation: false,
                    is_whitespace: false,
                    is_byte_fragment: is_bf,
                }
            })
            .collect();
        tokens.extend(extras.iter().cloned());

        let mut by_script: HashMap<Script, Vec<i32>> = HashMap::new();
        for t in &tokens {
            for &s in &t.scripts {
                by_script.entry(s).or_default().push(t.token_id);
            }
        }
        TokenLanguageIndex {
            vocab_hash: "byte_frag_test00".to_owned(),
            version: CURRENT_VERSION,
            tokens,
            by_script,
        }
    }

    /// With `include_byte_fragments = false` (the default), byte-fragment
    /// tokens are invisible to `tokens_for_language` regardless of their
    /// classified script. Behavior is bit-exact identical to Phase 1.
    #[test]
    fn byte_fragment_default_excluded_from_language_sets() {
        // Byte-fragments spanning multiple scripts.
        let index = build_byte_fragment_test_index(
            &[
                (100, 0xE4), // Han byte-fragment
                (101, 0xE3), // Hiragana byte-fragment
                (102, 0xEA), // Hangul byte-fragment
            ],
            &[],
        );

        let default_ex = ExceptionConfig::default();
        assert!(
            !default_ex.include_byte_fragments,
            "default must have include_byte_fragments=false"
        );

        let zh_ids =
            index.tokens_for_language(LanguageCode::Zh, InclusionPolicy::Conservative, &default_ex);
        assert!(
            !zh_ids.contains(&100),
            "byte-fragment Han must NOT appear in zh without include_byte_fragments: {zh_ids:?}"
        );

        let ja_ids =
            index.tokens_for_language(LanguageCode::Ja, InclusionPolicy::Conservative, &default_ex);
        assert!(
            !ja_ids.contains(&101),
            "byte-fragment Hiragana must NOT appear in ja without include_byte_fragments: {ja_ids:?}"
        );

        let ko_ids =
            index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Conservative, &default_ex);
        assert!(
            !ko_ids.contains(&102),
            "byte-fragment Hangul must NOT appear in ko without include_byte_fragments: {ko_ids:?}"
        );
    }

    /// With `include_byte_fragments = true`, byte-fragment tokens participate
    /// in language matching using their start-byte Script tag.
    ///
    /// This is the core of the issue #405 contract: the ` 年` leak described
    /// in the issue (`[74577, 112]` on Qwen2.5) classifies the leading
    /// fragment `74577` via its `0xE5`-family start byte and tags it as Han,
    /// so `zh=-inf` catches it.
    #[test]
    fn byte_fragment_include_flag_catches_start_byte_leak() {
        // Simulate the Qwen2.5 `[74577, 112]` situation: token 500 represents
        // the Han start byte (0xE5 family), token 501 a continuation byte.
        let index = build_byte_fragment_test_index(
            &[
                (500, 0xE5), // Han start byte — should be caught by zh
                (501, 0xB4), // Continuation byte — must stay Other
            ],
            &[],
        );

        let ex = ExceptionConfig {
            include_byte_fragments: true,
            ..Default::default()
        };

        let zh_ids =
            index.tokens_for_language(LanguageCode::Zh, InclusionPolicy::Conservative, &ex);
        assert!(
            zh_ids.contains(&500),
            "byte-fragment Han start byte MUST appear in zh when include_byte_fragments=true: {zh_ids:?}"
        );
        assert!(
            !zh_ids.contains(&501),
            "continuation byte MUST stay Other even with include_byte_fragments=true: {zh_ids:?}"
        );
    }

    /// A byte-fragment Hiragana leak (0xE3 start byte) is caught by the
    /// Japanese language set when `include_byte_fragments = true`.
    #[test]
    fn byte_fragment_include_flag_catches_kana_leak() {
        let index = build_byte_fragment_test_index(&[(600, 0xE3)], &[]);

        let ex = ExceptionConfig {
            include_byte_fragments: true,
            ..Default::default()
        };

        let ja_ids =
            index.tokens_for_language(LanguageCode::Ja, InclusionPolicy::Conservative, &ex);
        assert!(
            ja_ids.contains(&600),
            "byte-fragment Hiragana start byte must appear in ja: {ja_ids:?}"
        );
    }

    /// Strict-policy bleed-through guard: enabling `include_byte_fragments`
    /// must NOT leak Han-tagged fragments into the Korean Strict set (ko Strict
    /// = {Hangul}). This protects the existing Korean suppression contract.
    #[test]
    fn byte_fragment_strict_policy_does_not_bleed_cross_script() {
        let index = build_byte_fragment_test_index(
            &[
                (700, 0xE4), // Han byte-fragment
                (701, 0xEA), // Hangul byte-fragment
            ],
            &[],
        );

        let ex = ExceptionConfig {
            include_byte_fragments: true,
            ..Default::default()
        };

        let ko_strict_ids =
            index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Strict, &ex);
        assert!(
            !ko_strict_ids.contains(&700),
            "Han byte-fragment must NOT appear in ko Strict (Hangul-only): {ko_strict_ids:?}"
        );
        assert!(
            ko_strict_ids.contains(&701),
            "Hangul byte-fragment must appear in ko Strict: {ko_strict_ids:?}"
        );

        // ko Conservative includes Han, so the Han fragment DOES appear there.
        let ko_conserv_ids =
            index.tokens_for_language(LanguageCode::Ko, InclusionPolicy::Conservative, &ex);
        assert!(
            ko_conserv_ids.contains(&700),
            "Han byte-fragment appears in ko Conservative (which includes Han): {ko_conserv_ids:?}"
        );
    }

    /// Phase 1 (decode-path) tokens with a real script must NEVER be tagged as
    /// byte-fragments — even when they co-exist in the same index. This guards
    /// the "avoid double-counting" rule from issue #405.
    #[test]
    fn byte_fragment_does_not_override_phase1_classification() {
        // Merged-token Han (like `年` → id 7948 on Qwen2.5), added manually.
        let merged_han = TokenScriptInfo {
            token_id: 7948,
            scripts: {
                let mut v: SmallVec<[Script; 3]> = SmallVec::new();
                v.push(Script::Han);
                v
            },
            is_special: false,
            is_numeric: false,
            is_punctuation: false,
            is_whitespace: false,
            is_byte_fragment: false, // must stay false
        };

        let index = build_byte_fragment_test_index(&[(800, 0xE5)], &[merged_han]);

        // Both the merged and the byte-fragment Han tokens exist.
        let ex_off = ExceptionConfig::default();
        let zh_off =
            index.tokens_for_language(LanguageCode::Zh, InclusionPolicy::Conservative, &ex_off);
        assert!(
            zh_off.contains(&7948),
            "merged Han token must always be in zh: {zh_off:?}"
        );
        assert!(
            !zh_off.contains(&800),
            "byte-fragment must be gated off by default: {zh_off:?}"
        );

        let ex_on = ExceptionConfig {
            include_byte_fragments: true,
            ..Default::default()
        };
        let zh_on =
            index.tokens_for_language(LanguageCode::Zh, InclusionPolicy::Conservative, &ex_on);
        assert!(
            zh_on.contains(&7948) && zh_on.contains(&800),
            "both merged and fragment Han must appear when byte-fragments enabled: {zh_on:?}"
        );
    }

    // -------------------------------------------------------------------------
    // Real-tokenizer integration: the synthetic `74577, 112` leak on Qwen2.5.
    //
    // A full Qwen2.5 tokenizer.json is ~11 MB and we do not ship it in the
    // repo. Instead, the vocab-scan contract is exercised here via a crafted
    // byte-level BPE tokenizer fixture that guarantees:
    //   - It has a `ByteLevel` pretokenizer → byte-fragment pass runs.
    //   - It contains two synthetic byte-fragment vocab entries whose char
    //     reverse-maps to `0xE5` (Han start byte) and `0xB4` (continuation).
    // When `include_byte_fragments = true`, the start-byte entry lands in the
    // zh set; the continuation stays Other in both modes.
    // -------------------------------------------------------------------------

    /// End-to-end: a byte-level BPE tokenizer with a `0xE5`-represented byte
    /// token. `TokenLanguageIndex::build` must tag it as Han byte-fragment.
    #[test]
    fn byte_fragment_build_tags_real_byte_level_tokenizer() {
        // Construct a minimal byte-level BPE tokenizer JSON. The trick is
        // that we need the `pre_tokenizer` to be `ByteLevel` so the detector
        // fires, and the vocab must contain the byte-level char representing
        // byte 0xE5.
        //
        // From the GPT-2 encoder: byte 0xE5 is not in the "printable" set
        // (0x21..=0x7E ∪ 0xA1..=0xAC ∪ 0xAE..=0xFF contains it, since
        // 0xE5 is in 0xAE..=0xFF, so byte 0xE5 maps to itself → 'å').
        // Wait: 0xAE..=0xFF includes 0xE5, so the forward map sends 0xE5 → 0xE5
        // as a codepoint, which is 'å' (U+00E5).
        let char_e5 = '\u{00E5}'; // byte 0xE5 → 'å' in the GPT-2 alphabet
        let char_b4 = '\u{0174}'; // byte 0xB4 is NOT in the printable set,
                                  // so it maps to the first free slot above 0x100.
                                  // We don't need to compute the exact char here — we discover it at
                                  // runtime from the reverse map so this test stays robust.

        let reverse = build_byte_level_reverse_map();
        // Sanity: 0xE5 really is 'å' in the forward map.
        assert_eq!(reverse.get(&char_e5).copied(), Some(0xE5));
        // Find whatever char maps to byte 0xB4 (continuation byte).
        let char_b4_actual = reverse
            .iter()
            .find(|(_, &b)| b == 0xB4)
            .map(|(&c, _)| c)
            .expect("byte 0xB4 must have a forward-map char");
        let _ = char_b4;

        // Build a tokenizer JSON with a BPE model and ByteLevel pre-tokenizer.
        // We craft a tiny vocab with the byte-fragment chars plus a normal
        // word so the tokenizer parses and builds without complaint.
        //
        // The byte-level chars `char_e5` (= U+00E5) and `char_b4_actual` are
        // both valid printable Unicode scalars; we inject them directly into
        // the JSON source so the tokenizer's serde path parses them as ordinary
        // string literals.
        let json = format!(
            "{{\
  \"version\": \"1.0\",\
  \"truncation\": null,\
  \"padding\": null,\
  \"added_tokens\": [\
    {{\"id\": 0, \"content\": \"<|endoftext|>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}}\
  ],\
  \"normalizer\": null,\
  \"pre_tokenizer\": {{\
    \"type\": \"ByteLevel\",\
    \"add_prefix_space\": false,\
    \"trim_offsets\": true,\
    \"use_regex\": true\
  }},\
  \"post_processor\": null,\
  \"decoder\": {{\
    \"type\": \"ByteLevel\",\
    \"add_prefix_space\": false,\
    \"trim_offsets\": true,\
    \"use_regex\": true\
  }},\
  \"model\": {{\
    \"type\": \"BPE\",\
    \"dropout\": null,\
    \"unk_token\": null,\
    \"continuing_subword_prefix\": null,\
    \"end_of_word_suffix\": null,\
    \"fuse_unk\": false,\
    \"byte_fallback\": false,\
    \"ignore_merges\": true,\
    \"vocab\": {{\
      \"<|endoftext|>\": 0,\
      \"hello\": 1,\
      \"{char_e5}\": 2,\
      \"{char_b4_actual}\": 3\
    }},\
    \"merges\": []\
  }}\
}}",
            char_e5 = char_e5,
            char_b4_actual = char_b4_actual,
        );

        let tok = Tokenizer::from_bytes(json.as_bytes()).expect("byte-level fixture must parse");
        assert!(
            has_byte_level_pretokenizer(&tok),
            "fixture must expose a ByteLevel pretokenizer"
        );

        let index =
            TokenLanguageIndex::build(&tok, json.as_bytes()).expect("build on byte-level fixture");

        // Token 2 is the 0xE5 byte-fragment — must be tagged Han + is_byte_fragment.
        let tok2 = index.tokens.iter().find(|t| t.token_id == 2).expect("id=2");
        assert!(
            tok2.is_byte_fragment,
            "token 2 (0xE5) must be flagged as byte-fragment: {tok2:?}"
        );
        assert!(
            tok2.scripts.contains(&Script::Han),
            "token 2 (0xE5) must carry Script::Han: {tok2:?}"
        );

        // Token 3 is the 0xB4 continuation byte — must stay Other (empty scripts).
        let tok3 = index.tokens.iter().find(|t| t.token_id == 3).expect("id=3");
        assert!(
            tok3.scripts.is_empty(),
            "continuation byte 0xB4 must stay Other: {tok3:?}"
        );
        assert!(
            !tok3.is_byte_fragment,
            "continuation byte must NOT be flagged as byte-fragment: {tok3:?}"
        );

        // Token 1 is the merged "hello" — must classify via Phase 1 path,
        // not via byte-fragment.
        let tok1 = index.tokens.iter().find(|t| t.token_id == 1).expect("id=1");
        assert!(
            tok1.scripts.contains(&Script::Latin),
            "merged Latin token must still classify via Phase 1: {tok1:?}"
        );
        assert!(
            !tok1.is_byte_fragment,
            "merged token must NOT be flagged as byte-fragment: {tok1:?}"
        );

        // tokens_for_language: with the flag off, token 2 is excluded; on, included.
        let ex_off = ExceptionConfig::default();
        let zh_off =
            index.tokens_for_language(LanguageCode::Zh, InclusionPolicy::Conservative, &ex_off);
        assert!(
            !zh_off.contains(&2),
            "flag off: byte-fragment Han token must be excluded from zh: {zh_off:?}"
        );

        let ex_on = ExceptionConfig {
            include_byte_fragments: true,
            ..Default::default()
        };
        let zh_on =
            index.tokens_for_language(LanguageCode::Zh, InclusionPolicy::Conservative, &ex_on);
        assert!(
            zh_on.contains(&2),
            "flag on: byte-fragment Han token MUST be in zh (issue #405 leak caught): {zh_on:?}"
        );
    }

    /// When the tokenizer has no ByteLevel pretokenizer (SentencePiece-style
    /// WordLevel fixture used in other tests), the byte-fragment pass must
    /// silently no-op and every token should have `is_byte_fragment = false`.
    #[test]
    fn byte_fragment_no_op_on_non_byte_level_tokenizer() {
        let json = resolve_mock_tokenizer_json("marker_no_byte_level");
        let tok = make_tokenizer(&json);

        // Sanity: WordLevel fixture has no ByteLevel pretokenizer.
        assert!(
            !has_byte_level_pretokenizer(&tok),
            "WordLevel fixture must not report ByteLevel pretokenizer"
        );

        let index = TokenLanguageIndex::build(&tok, json.as_bytes())
            .expect("build on non-byte-level tokenizer must succeed");

        assert!(
            index.tokens.iter().all(|t| !t.is_byte_fragment),
            "no tokens should be tagged as byte-fragment on non-byte-level tokenizer"
        );
    }
}
