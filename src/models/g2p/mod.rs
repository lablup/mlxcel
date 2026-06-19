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

//! Grapheme-to-phoneme (g2p) front-end for the Kokoro TTS path.
//!
//! Kokoro consumes IPA phonemes, not raw text, so this module is the host-side
//! analogue of an STT mel front-end. The approach is a **self-contained
//! American-English phonemizer**: text is normalized into words and punctuation,
//! each word is looked up in a bundled lexicon ([`lexicon`]), and
//! out-of-vocabulary words fall back to deterministic letter-to-sound rules
//! ([`rules`]). No external binary or network dependency is required.
//!
//! Language scope: **American English** (the default voice `af_heart` and the
//! other `a*` voices). Non-English voices in the checkpoint still load and
//! synthesize, but the phonemes are produced by the English front-end, so their
//! pronunciation quality is limited. Extending to other languages is future
//! work (it would need per-language lexicons / rules, the analogue of upstream
//! Kokoro's `misaki[xx]` packages).
//!
//! Output: a phoneme string built from Kokoro vocab symbols, with words
//! separated by the space token and punctuation passed through. The acoustic
//! model maps each symbol to a token id via the checkpoint's `vocab`.

mod lexicon;
mod normalize;
mod rules;

use normalize::Token;

/// Convert raw English text into a Kokoro IPA phoneme string.
///
/// Words are looked up in the bundled lexicon, falling back to letter-to-sound
/// rules; punctuation and word boundaries are preserved as the corresponding
/// vocab symbols. The result is suitable for `KokoroModel::synthesize`.
pub fn text_to_phonemes(text: &str) -> String {
    let tokens = normalize::normalize(text);
    let mut out = String::new();
    let mut pending_space = false;

    for token in tokens {
        match token {
            Token::Word(word) => {
                if pending_space && !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(&phonemize_word(&word));
                pending_space = true;
            }
            Token::Punct(p) => {
                // Punctuation attaches to the preceding token (no leading space).
                out.push(p);
                pending_space = true;
            }
        }
    }
    out
}

/// Phonemize a single normalized word: lexicon first, then rules.
fn phonemize_word(word: &str) -> String {
    if let Some(ipa) = lexicon::lookup(word) {
        return ipa.to_string();
    }
    // Strip a trailing possessive/plural clitic and retry the lexicon before
    // resorting to letter-to-sound, so "cats" -> "cat" + s.
    if let Some(stripped) = word.strip_suffix("'s")
        && let Some(base) = lexicon::lookup(stripped)
    {
        return format!("{base}z");
    }
    rules::word_to_ipa(word)
}

/// Test-only access to the Kokoro vocab symbol set, used by the lexicon symbol
/// audit. Kept here so the table lives next to the lexicon it validates.
#[cfg(test)]
pub(crate) mod tests_support {
    use std::collections::HashSet;

    /// American-English-relevant Kokoro vocab characters plus stress/length
    /// marks and the space/punctuation tokens. Mirrors `config.json`'s `vocab`
    /// keys for the symbols the English front-end can emit.
    pub(crate) fn vocab_chars() -> HashSet<char> {
        let symbols = [
            // punctuation / space
            ';', ':', ',', '.', '!', '?', ' ', '\'', // ASCII letters used as phonemes
            'b', 'd', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'u', 'v', 'w',
            'z', 'a', 'e', 'o', // IPA vowels
            'ɑ', 'ɐ', 'ɒ', 'æ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɪ', 'ʊ', 'ʌ', 'ɝ',
            // IPA consonants
            'ð', 'ŋ', 'ɹ', 'ʃ', 'ʒ', 'ʤ', 'ʧ', 'θ', 'ɡ', 'ɾ', // stress / length
            'ˈ', 'ˌ', 'ː',
        ];
        symbols.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phonemizes_known_words() {
        let ps = text_to_phonemes("hello world");
        assert!(ps.contains("həˈloʊ"), "got {ps}");
        assert!(ps.contains("wɜːld"), "got {ps}");
        // space token separates words
        assert!(ps.contains(' '), "expected a word boundary in {ps}");
    }

    #[test]
    fn passes_punctuation_through() {
        let ps = text_to_phonemes("hello, world.");
        assert!(ps.contains(','), "comma preserved in {ps}");
        assert!(ps.ends_with('.'), "trailing period preserved in {ps}");
    }

    #[test]
    fn oov_word_uses_rules() {
        // A nonsense word not in the lexicon still yields phonemes.
        let ps = text_to_phonemes("zorptang");
        assert!(!ps.is_empty(), "OOV word should phonemize via rules");
        assert!(ps.contains('ˈ'), "rule output carries stress: {ps}");
    }

    #[test]
    fn empty_input_is_empty() {
        assert_eq!(text_to_phonemes(""), "");
        assert_eq!(text_to_phonemes("   "), "");
    }

    #[test]
    fn numbers_are_spoken() {
        let ps = text_to_phonemes("3 cats");
        // "three" -> θɹi must appear
        assert!(ps.contains("θɹi"), "number expanded in {ps}");
    }

    #[test]
    fn possessive_clitic_strips_and_retries_lexicon() {
        // "cat's" should produce the phonemes for "cat" plus a trailing /z/ for
        // the 's clitic, rather than treating "cat's" as an OOV word. The
        // lexicon entry for "cat" is `kæt` (no stress mark); the clitic appends z.
        let ps = text_to_phonemes("cat's");
        assert!(ps.contains("kæt"), "base form phonemized in {ps}");
        assert!(ps.ends_with('z'), "possessive /z/ appended in {ps}");
    }

    #[test]
    fn whitespace_only_input_is_empty() {
        // Tabs and newlines are not special-cased separately; they should all
        // produce an empty output the same way spaces do.
        assert_eq!(text_to_phonemes("\t\n "), "");
    }
}
