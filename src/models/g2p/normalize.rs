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

//! Text normalization for the English g2p front-end.
//!
//! Splits raw input text into a sequence of tokens (words, numbers, and
//! punctuation) and expands a small set of numeric and symbolic forms into
//! words so they can be phonemized. The scope is intentionally narrow:
//! lower-casing, integer expansion, and common punctuation. Anything beyond the
//! ASCII letter / digit / supported-punctuation set is dropped.

/// A normalized token: a word to be phonemized, or a literal punctuation mark
/// that maps directly to a Kokoro vocab symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Token {
    /// A lower-cased alphabetic word.
    Word(String),
    /// A punctuation mark that Kokoro's vocab represents directly.
    Punct(char),
}

/// Punctuation marks Kokoro's vocab carries (kept in the phoneme stream).
const KEPT_PUNCT: &[char] = &[';', ':', ',', '.', '!', '?'];

/// Normalize raw text into a token stream.
pub(crate) fn normalize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    let mut number = String::new();

    let flush_word = |word: &mut String, tokens: &mut Vec<Token>| {
        if !word.is_empty() {
            tokens.push(Token::Word(std::mem::take(word)));
        }
    };
    let flush_number = |number: &mut String, tokens: &mut Vec<Token>| {
        if !number.is_empty() {
            for w in expand_integer(number) {
                tokens.push(Token::Word(w));
            }
            number.clear();
        }
    };

    for ch in text.chars() {
        if ch.is_ascii_alphabetic() {
            flush_number(&mut number, &mut tokens);
            word.push(ch.to_ascii_lowercase());
        } else if ch.is_ascii_digit() {
            flush_word(&mut word, &mut tokens);
            number.push(ch);
        } else if ch == '\'' {
            // Keep apostrophes inside words (contractions) but drop leading ones.
            if !word.is_empty() {
                word.push('\'');
            }
        } else {
            flush_word(&mut word, &mut tokens);
            flush_number(&mut number, &mut tokens);
            if KEPT_PUNCT.contains(&ch) {
                tokens.push(Token::Punct(ch));
            }
            // Whitespace and other symbols act only as separators.
        }
    }
    flush_word(&mut word, &mut tokens);
    flush_number(&mut number, &mut tokens);
    tokens
}

/// Expand a non-negative integer string into English number words.
///
/// Handles values up to the billions; longer inputs fall back to digit-by-digit
/// reading. Returns the words as separate tokens (e.g. `["one", "hundred",
/// "twenty", "three"]`).
pub(crate) fn expand_integer(digits: &str) -> Vec<String> {
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() {
        return vec!["zero".to_string()];
    }
    match trimmed.parse::<u64>() {
        Ok(n) if n < 1_000_000_000_000 => say_number(n).split(' ').map(str::to_string).collect(),
        _ => digits.chars().map(|d| digit_word(d).to_string()).collect(),
    }
}

fn digit_word(d: char) -> &'static str {
    match d {
        '0' => "zero",
        '1' => "one",
        '2' => "two",
        '3' => "three",
        '4' => "four",
        '5' => "five",
        '6' => "six",
        '7' => "seven",
        '8' => "eight",
        _ => "nine",
    }
}

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// Render `n` (< 1e12) as space-separated English words.
fn say_number(n: u64) -> String {
    if n < 20 {
        return ONES[n as usize].to_string();
    }
    if n < 100 {
        let t = TENS[(n / 10) as usize];
        let o = n % 10;
        return if o == 0 {
            t.to_string()
        } else {
            format!("{t} {}", ONES[o as usize])
        };
    }
    if n < 1_000 {
        let h = n / 100;
        let rest = n % 100;
        return if rest == 0 {
            format!("{} hundred", ONES[h as usize])
        } else {
            format!("{} hundred {}", ONES[h as usize], say_number(rest))
        };
    }
    for (div, name) in [
        (1_000_000_000u64, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ] {
        if n >= div {
            let high = n / div;
            let rest = n % div;
            let mut s = format!("{} {name}", say_number(high));
            if rest != 0 {
                s.push(' ');
                s.push_str(&say_number(rest));
            }
            return s;
        }
    }
    ONES[0].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_words_and_punct() {
        let toks = normalize("Hello, world!");
        assert_eq!(
            toks,
            vec![
                Token::Word("hello".into()),
                Token::Punct(','),
                Token::Word("world".into()),
                Token::Punct('!'),
            ]
        );
    }

    #[test]
    fn keeps_contractions() {
        let toks = normalize("don't");
        assert_eq!(toks, vec![Token::Word("don't".into())]);
    }

    #[test]
    fn expands_integers() {
        assert_eq!(expand_integer("0"), vec!["zero"]);
        assert_eq!(expand_integer("7"), vec!["seven"]);
        assert_eq!(expand_integer("23"), vec!["twenty", "three"]);
        assert_eq!(expand_integer("105"), vec!["one", "hundred", "five"]);
        assert_eq!(
            expand_integer("2026"),
            vec!["two", "thousand", "twenty", "six"]
        );
    }

    #[test]
    fn number_token_in_text() {
        let toks = normalize("I have 3 cats");
        assert_eq!(
            toks,
            vec![
                Token::Word("i".into()),
                Token::Word("have".into()),
                Token::Word("three".into()),
                Token::Word("cats".into()),
            ]
        );
    }
}
