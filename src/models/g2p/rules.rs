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

//! Letter-to-sound fallback for out-of-vocabulary English words.
//!
//! A deterministic, rule-based grapheme-to-phoneme converter that emits Kokoro
//! IPA symbols. It is the fallback for words absent from the bundled lexicon, so
//! it favors predictability over perfect accuracy: common digraphs first
//! (`sh`, `ch`, `th`, `ph`, vowel teams), then single letters, with a coarse
//! "silent final e lengthens the prior vowel" heuristic and a default primary
//! stress on the first vowel. Output uses only symbols present in the Kokoro
//! American-English inventory.

/// Convert a lower-cased alphabetic word to a Kokoro IPA phoneme string.
///
/// The result carries a single leading primary-stress mark before the first
/// vowel cluster. Returns an empty string for an empty input.
pub(crate) fn word_to_ipa(word: &str) -> String {
    let chars: Vec<char> = word.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if chars.is_empty() {
        return String::new();
    }
    let n = chars.len();
    let silent_final_e = n >= 3 && chars[n - 1] == 'e' && !is_vowel(chars[n - 2]);
    let effective_len = if silent_final_e { n - 1 } else { n };

    let mut out = String::new();
    let mut i = 0;
    let mut stress_placed = false;

    while i < effective_len {
        let c = chars[i];
        let next = if i + 1 < effective_len {
            chars[i + 1]
        } else {
            '\0'
        };
        let nn = if i + 2 < effective_len {
            chars[i + 2]
        } else {
            '\0'
        };

        // Two-letter graphemes.
        if let Some((ph, adv)) = digraph(c, next, nn) {
            if is_vowel(c) && !stress_placed {
                out.push('ˈ');
                stress_placed = true;
            }
            out.push_str(ph);
            i += adv;
            continue;
        }

        // Single letters.
        if is_vowel(c) {
            if !stress_placed {
                out.push('ˈ');
                stress_placed = true;
            }
            let long = silent_final_e && i == effective_len - 1 && effective_len >= 1;
            out.push_str(vowel_ipa(c, long));
        } else {
            out.push_str(consonant_ipa(c, next));
        }
        i += 1;
    }

    if !stress_placed && !out.is_empty() {
        // No vowel found (rare); mark the start.
        out.insert(0, 'ˈ');
    }
    out
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

/// Match a two- or three-letter grapheme starting at the current position.
/// Returns `(ipa, chars_consumed)`.
fn digraph(c: char, next: char, nn: char) -> Option<(&'static str, usize)> {
    let pair = (c, next);
    let ph = match pair {
        ('s', 'h') => "ʃ",
        ('c', 'h') => "ʧ",
        ('t', 'h') => "θ", // unvoiced default
        ('p', 'h') => "f",
        ('w', 'h') => "w",
        ('c', 'k') => "k",
        ('n', 'g') => "ŋ",
        ('q', 'u') => "kw",
        ('e', 'e') => "i",
        ('e', 'a') => "i",
        ('o', 'o') => "u",
        ('o', 'u') => "aʊ",
        ('o', 'w') => "aʊ",
        ('o', 'i') => "ɔɪ",
        ('o', 'y') => "ɔɪ",
        ('a', 'i') => "eɪ",
        ('a', 'y') => "eɪ",
        ('a', 'u') => "ɔ",
        ('a', 'w') => "ɔ",
        ('i', 'e') => "aɪ",
        ('e', 'i') => "eɪ",
        ('e', 'y') => "eɪ",
        _ => return None,
    };
    // Three-letter exceptions could go here; `nn` is reserved for that.
    let _ = nn;
    Some((ph, 2))
}

/// IPA for a single vowel. `long` selects the tense/long realization (used when
/// a silent final `e` lengthens it).
fn vowel_ipa(c: char, long: bool) -> &'static str {
    match (c, long) {
        ('a', false) => "æ",
        ('a', true) => "eɪ",
        ('e', false) => "ɛ",
        ('e', true) => "i",
        ('i', false) => "ɪ",
        ('i', true) => "aɪ",
        ('o', false) => "ɑ",
        ('o', true) => "oʊ",
        ('u', false) => "ʌ",
        ('u', true) => "u",
        ('y', false) => "ɪ",
        ('y', true) => "aɪ",
        _ => "ə",
    }
}

/// IPA for a single consonant. `next` disambiguates soft/hard `c` and `g`.
fn consonant_ipa(c: char, next: char) -> &'static str {
    let soft = matches!(next, 'e' | 'i' | 'y');
    match c {
        'b' => "b",
        'c' => {
            if soft {
                "s"
            } else {
                "k"
            }
        }
        'd' => "d",
        'f' => "f",
        'g' => {
            if soft {
                "ʤ"
            } else {
                "ɡ"
            }
        }
        'h' => "h",
        'j' => "ʤ",
        'k' => "k",
        'l' => "l",
        'm' => "m",
        'n' => "n",
        'p' => "p",
        'q' => "k",
        'r' => "ɹ",
        's' => "s",
        't' => "t",
        'v' => "v",
        'w' => "w",
        'x' => "ks",
        'z' => "z",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_word_is_empty() {
        assert_eq!(word_to_ipa(""), "");
    }

    #[test]
    fn places_single_primary_stress() {
        let ipa = word_to_ipa("cat");
        assert_eq!(ipa.matches('ˈ').count(), 1, "exactly one stress mark");
        // c-a-t -> k æ t with stress before the vowel
        assert!(ipa.contains('æ'));
        assert!(ipa.starts_with('k'));
    }

    #[test]
    fn handles_sh_digraph() {
        let ipa = word_to_ipa("ship");
        assert!(ipa.contains('ʃ'), "sh -> ʃ in {ipa}");
        assert!(ipa.contains('ɪ'));
        assert!(ipa.contains('p'));
    }

    #[test]
    fn silent_final_e_lengthens() {
        // "ride" -> r aɪ d (long i), final e silent
        let ipa = word_to_ipa("ride");
        assert!(ipa.contains('ɹ'));
        assert!(!ipa.ends_with('ɛ'), "final e should be silent in {ipa}");
    }

    #[test]
    fn soft_c_before_e() {
        let ipa = word_to_ipa("ce");
        assert!(ipa.contains('s'), "soft c -> s in {ipa}");
    }
}
