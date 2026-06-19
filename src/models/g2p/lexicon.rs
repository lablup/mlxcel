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

//! Bundled American-English pronunciation lexicon.
//!
//! A curated `word \t IPA` table compiled into the binary, covering the highest
//! frequency English words plus a handful of TTS-relevant terms. The IPA uses
//! only symbols present in the Kokoro vocab. Words absent from this table fall
//! back to the rule-based converter in [`super::rules`]. This keeps the g2p
//! front-end self-contained (no external binary or download).

use std::collections::HashMap;
use std::sync::OnceLock;

/// The embedded lexicon source (`word \t ipa` per line).
const LEXICON_TSV: &str = include_str!("data/lexicon_en.tsv");

/// Parsed lexicon, built once on first use.
fn lexicon() -> &'static HashMap<&'static str, &'static str> {
    static MAP: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = HashMap::new();
        for line in LEXICON_TSV.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            if let Some((word, ipa)) = line.split_once('\t') {
                let word = word.trim();
                let ipa = ipa.trim();
                if !word.is_empty() && !ipa.is_empty() {
                    map.insert(word, ipa);
                }
            }
        }
        map
    })
}

/// Look up a lower-cased word's IPA in the bundled lexicon.
pub(crate) fn lookup(word: &str) -> Option<&'static str> {
    lexicon().get(word).copied()
}

/// Number of entries in the bundled lexicon (used by the coverage tests).
#[cfg(test)]
pub(crate) fn len() -> usize {
    lexicon().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_common_words() {
        assert_eq!(lookup("the"), Some("ðə"));
        assert_eq!(lookup("hello"), Some("həˈloʊ"));
        assert_eq!(lookup("world"), Some("wɜːld"));
        assert!(lookup("zzzznotaword").is_none());
    }

    #[test]
    fn lexicon_is_nontrivial() {
        assert!(len() > 200, "expected a few hundred entries, got {}", len());
    }

    /// Every IPA value must use only characters Kokoro's vocab can represent,
    /// otherwise the phoneme would be silently dropped at tokenization. This
    /// guards against typos when extending the lexicon.
    #[test]
    fn all_entries_use_known_symbols() {
        // The Kokoro vocab keys (subset relevant to American English) plus the
        // stress/length marks. Mirrors config.json's `vocab`.
        let allowed: std::collections::HashSet<char> = super::super::tests_support::vocab_chars();
        for line in LEXICON_TSV.lines() {
            if let Some((word, ipa)) = line.split_once('\t') {
                for c in ipa.trim().chars() {
                    assert!(
                        allowed.contains(&c),
                        "lexicon entry '{}' has symbol '{}' (U+{:04X}) not in Kokoro vocab",
                        word.trim(),
                        c,
                        c as u32
                    );
                }
            }
        }
    }
}
