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

//! Assemble transcribed blocks into page Markdown.
//!
//! Port of `reconstruct.py` (`idp_reconstruct.py`): de-hyphenation across
//! line breaks, math repair for Indic script runs and display equations, and
//! reading-ordered assembly that skips untranscribed blocks.

use std::sync::LazyLock;

use fancy_regex::Regex;

use super::blocks::Block;
use super::labels::is_drop_type;

static HYPHEN_BREAK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("(\\w)[-\u{2010}-\u{2014}\u{2212}]\\n[ \\t]*(\\w)").unwrap());
static MATH_SPAN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)\$\$(.+?)\$\$|(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)").unwrap());
static NON_LATIN_RUN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        "[\\u{0600}-\\u{06ff}\\u{0900}-\\u{0d7f}\\u{1c50}-\\u{1c7f}](?:[\\u{0600}-\\u{06ff}\\u{0900}-\\u{0d7f}\\u{1c50}-\\u{1c7f} \\u{200c}\\u{200d}]*[\\u{0600}-\\u{06ff}\\u{0900}-\\u{0d7f}\\u{1c50}-\\u{1c7f}])?",
    )
    .unwrap()
});
static TEXT_COMMAND: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\text\{[^{}]*\}").unwrap());
static PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| Regex::new("\u{0}(\\d+)\u{0}").unwrap());
static NEWLINES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s*\n\s*").unwrap());

const MATH_DELIMITERS: [&str; 3] = ["$", "\\[", "\\("];

/// Rejoin words split by a hyphen at a line break, to a fixpoint.
pub fn dehyphenate(text: &str) -> String {
    let mut current = text.to_string();
    loop {
        let next = HYPHEN_BREAK.replace_all(&current, "$1$2").into_owned();
        if next == current {
            return current;
        }
        current = next;
    }
}

fn replace_captures(
    text: &str,
    regex: &Regex,
    mut replacement: impl FnMut(&fancy_regex::Captures<'_, str>) -> String,
) -> String {
    let mut output = String::with_capacity(text.len());
    let mut last = 0usize;
    for captures in regex.captures_iter(text).flatten() {
        let Some(whole) = captures.get(0) else {
            continue;
        };
        output.push_str(&text[last..whole.start()]);
        output.push_str(&replacement(&captures));
        last = whole.end();
    }
    output.push_str(&text[last..]);
    output
}

/// Make one transcribed expression valid LaTeX.
fn repair_expression(tex: &str, display: bool) -> String {
    // Protect existing \text{...} runs from the Indic-script wrapper.
    let mut protected: Vec<String> = Vec::new();
    let stashed = replace_captures(tex, &TEXT_COMMAND, |captures| {
        protected.push(captures.get(0).expect("whole match").as_str().to_string());
        format!("\u{0}{}\u{0}", protected.len() - 1)
    });
    let wrapped = replace_captures(&stashed, &NON_LATIN_RUN, |captures| {
        format!(
            "\\text{{{}}}",
            captures.get(0).expect("whole match").as_str()
        )
    });
    let restored = replace_captures(&wrapped, &PLACEHOLDER, |captures| {
        let index: usize = captures
            .get(1)
            .expect("index group")
            .as_str()
            .parse()
            .expect("placeholder index");
        protected[index].clone()
    });
    if display {
        let trimmed = restored.trim();
        replace_captures(trimmed, &NEWLINES, |_| " \\\\ ".to_string())
    } else {
        restored
    }
}

/// Repair every math span in a Markdown string.
pub fn repair_math(text: &str) -> String {
    replace_captures(text, &MATH_SPAN, |captures| {
        let display = captures.get(1).is_some();
        let inner = captures
            .get(1)
            .or_else(|| captures.get(2))
            .expect("one math group")
            .as_str();
        let repaired = repair_expression(inner, display);
        if display {
            format!("$${repaired}$$")
        } else {
            format!("${repaired}$")
        }
    })
}

/// Reading-ordered Markdown. Blocks with no text contribute nothing.
pub fn reconstruct(blocks: &[Block], repair: bool) -> String {
    let mut kept: Vec<&Block> = blocks
        .iter()
        .filter(|block| !is_drop_type(&block.block_type))
        .collect();
    kept.sort_by_key(|block| block.order);

    let mut parts: Vec<String> = Vec::new();
    for block in kept {
        let mut text = block.text.as_deref().unwrap_or_default().trim().to_string();
        if text.is_empty() {
            continue;
        }
        if block.block_type == "Equation"
            && !text.contains('$')
            && !MATH_DELIMITERS
                .iter()
                .any(|delimiter| text.starts_with(delimiter))
        {
            text = format!("$${text}$$");
        }
        parts.push(text);
    }

    let markdown = dehyphenate(&parts.join("\n\n"));
    if repair {
        repair_math(&markdown)
    } else {
        markdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(order: usize, block_type: &str, text: &str) -> Block {
        Block {
            order,
            label: block_type.to_string(),
            block_type: block_type.to_string(),
            bbox: [0.0, 0.0, 1.0, 1.0],
            conf: 1.0,
            text: Some(text.to_string()),
        }
    }

    #[test]
    fn dehyphenation_joins_across_line_breaks_to_a_fixpoint() {
        assert_eq!(dehyphenate("exam-\nple"), "example");
        assert_eq!(dehyphenate("co-\nop-\nerate"), "cooperate");
        assert_eq!(dehyphenate("keep - this"), "keep - this");
    }

    #[test]
    fn repair_wraps_indic_script_runs_in_text() {
        let repaired = repair_math("$हिन्दी$");
        assert_eq!(repaired, "$\\text{हिन्दी}$");
        // An existing \text{} run is not wrapped twice.
        let already = repair_math("$\\text{हिन्दी}$");
        assert_eq!(already, "$\\text{हिन्दी}$");
    }

    #[test]
    fn display_math_newlines_become_line_breaks() {
        // Two literal backslashes in the output.
        assert_eq!(repair_math("$$a\nb$$"), "$$a \\\\ b$$");
        assert_eq!(repair_math("$$a+b$$"), "$$a+b$$");
        assert_eq!(repair_math("$x$ and $y$"), "$x$ and $y$");
    }

    #[test]
    fn reconstruction_is_reading_ordered_and_skips_empty_and_dropped_blocks() {
        let blocks = vec![
            block(2, "Table", "<table></table>"),
            block(0, "Title", "किताब"),
            block(1, "Text", "exam-\nple text"),
            block(3, "Figure", "never"),
            block(4, "Text", "   "),
        ];
        let markdown = reconstruct(&blocks, true);
        assert_eq!(markdown, "किताब\n\nexample text\n\n<table></table>");
    }

    #[test]
    fn an_equation_without_delimiters_is_wrapped() {
        let blocks = vec![block(0, "Equation", "x^2 + y^2")];
        assert_eq!(reconstruct(&blocks, true), "$$x^2 + y^2$$");
        let blocks = vec![block(0, "Equation", "$$x^2$$")];
        assert_eq!(reconstruct(&blocks, true), "$$x^2$$");
        let blocks = vec![block(0, "Equation", "\\[x^2\\]")];
        assert_eq!(reconstruct(&blocks, false), "\\[x^2\\]");
    }

    #[test]
    fn reconstruction_without_repair_keeps_display_newlines() {
        let blocks = vec![block(0, "Equation", "$$a\nb$$")];
        assert_eq!(reconstruct(&blocks, false), "$$a\nb$$");
    }
}
