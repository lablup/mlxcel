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

//! Ideogram 4 prompt normalization.
//!
//! A port of `normalize_prompt`, `prepare_prompt`, `_minimal_caption` and the
//! caption checks in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ideogram4/prompting.py>
//! (`#L127-L223`). Ideogram 4 is trained on JSON captions: a prompt that
//! starts with `{` passes through unchanged (schema problems are warnings,
//! not errors), and plain text is wrapped into a minimal caption unless
//! `auto_json_caption` is false. Local prompt expansion through a second
//! model (`prompt_expansion_model`, mlx-vlm#1276) is not supported; a request
//! naming one is rejected by [`super::config::Ideogram4Options::from_extra`].

use std::fmt::Write as _;

use anyhow::{Result, bail};
use serde_json::{Map, Value, json};

/// A prompt after normalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedPrompt {
    /// The text that is tokenized.
    pub text: String,
    pub is_json_caption: bool,
    pub is_structured_caption: bool,
    /// Plain text wrapped by [`minimal_caption`]; `text` is then the
    /// `revised_prompt`.
    pub was_wrapped: bool,
    /// Schema warnings, at most [`MAX_WARNINGS`] of them.
    pub warnings: Vec<String>,
    /// Further warnings found past [`MAX_WARNINGS`] and not kept (a lower
    /// bound: the walk stops building strings once the cap is reached).
    pub warnings_dropped: usize,
}

/// Most warnings kept (and so allocated) for one caption.
pub const MAX_WARNINGS: usize = 64;
/// Most warnings written to the log for one caption.
const MAX_LOGGED_WARNINGS: usize = 8;

/// Bounded warning collector.
struct Sink {
    issues: Vec<String>,
    dropped: usize,
}

impl Sink {
    fn new() -> Self {
        Self {
            issues: Vec::new(),
            dropped: 0,
        }
    }

    fn full(&self) -> bool {
        self.issues.len() >= MAX_WARNINGS
    }

    /// Record a warning, building its text only when there is room.
    fn push_with(&mut self, make: impl FnOnce() -> String) {
        if self.full() {
            self.dropped += 1;
        } else {
            self.issues.push(make());
        }
    }
}

impl NormalizedPrompt {
    /// The prompt to report as `revised_prompt`: the wrapped caption, or
    /// `None` when the caller's text was used as is.
    #[must_use]
    pub fn revised_prompt(&self) -> Option<String> {
        self.was_wrapped.then(|| self.text.clone())
    }
}

/// Upstream `format_caption`: compact JSON (`(",", ":")` separators), keys in
/// insertion order, non-ASCII kept.
#[must_use]
pub fn format_caption(caption: &Value) -> String {
    caption.to_string()
}

/// Upstream `_minimal_caption`.
#[must_use]
pub fn minimal_caption(prompt: &str) -> Value {
    json!({
        "high_level_description": prompt,
        "compositional_deconstruction": {
            "background": "The setting, environment, and surrounding context implied by the prompt.",
            "elements": [{"type": "obj", "desc": prompt}],
        },
    })
}

fn loads_json_caption(prompt: &str) -> Result<Map<String, Value>> {
    match serde_json::from_str::<Value>(prompt) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => bail!("Ideogram 4 JSON caption must be an object"),
        Err(_) => bail!("Invalid Ideogram 4 JSON caption"),
    }
}

/// Upstream `normalize_prompt`. A prompt starting with `{` must parse as a
/// JSON object (the error is the caller's); its schema problems are returned
/// as warnings and logged.
pub fn normalize_prompt(prompt: &str, auto_json_caption: bool) -> Result<NormalizedPrompt> {
    let stripped = prompt.trim();
    if stripped.starts_with('{') {
        let caption = loads_json_caption(stripped)?;
        let (warnings, warnings_dropped) = collect_warnings(&caption);
        log_warnings(&warnings, warnings_dropped);
        return Ok(NormalizedPrompt {
            text: prompt.to_string(),
            is_json_caption: true,
            is_structured_caption: warnings.is_empty() && warnings_dropped == 0,
            was_wrapped: false,
            warnings,
            warnings_dropped,
        });
    }
    if !auto_json_caption {
        return Ok(NormalizedPrompt {
            text: prompt.to_string(),
            is_json_caption: false,
            is_structured_caption: false,
            was_wrapped: false,
            warnings: Vec::new(),
            warnings_dropped: 0,
        });
    }
    Ok(NormalizedPrompt {
        text: format_caption(&minimal_caption(stripped)),
        is_json_caption: true,
        is_structured_caption: true,
        was_wrapped: true,
        warnings: Vec::new(),
        warnings_dropped: 0,
    })
}

/// Log the first few warnings with control characters replaced (warnings
/// embed request-controlled JSON keys), then one line for the rest.
fn log_warnings(warnings: &[String], dropped: usize) {
    for warning in warnings.iter().take(MAX_LOGGED_WARNINGS) {
        let clean: String = warning
            .chars()
            .map(|c| if c.is_control() { '?' } else { c })
            .collect();
        tracing::warn!("{clean}");
    }
    let more = warnings.len().saturating_sub(MAX_LOGGED_WARNINGS) + dropped;
    if more > 0 {
        tracing::warn!("... and {more} more Ideogram 4 caption warnings");
    }
}

fn is_non_empty_string(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_str)
        .is_some_and(|s| !s.trim().is_empty())
}

/// Upstream `_caption_warnings`, in its order, at most [`MAX_WARNINGS`].
#[must_use]
pub fn caption_warnings(caption: &Map<String, Value>) -> Vec<String> {
    collect_warnings(caption).0
}

/// [`caption_warnings`] and the number of warnings dropped past the cap.
fn collect_warnings(caption: &Map<String, Value>) -> (Vec<String>, usize) {
    const P: &str = "Ideogram 4 JSON caption";
    let mut sink = Sink::new();
    match caption.get("compositional_deconstruction") {
        Some(Value::Object(comp)) => {
            if !is_non_empty_string(comp.get("background")) {
                sink.push_with(|| format!(
                    "{P} should include 'compositional_deconstruction.background' as a non-empty string."
                ));
            }
            match comp.get("elements") {
                Some(Value::Array(elements)) => {
                    for (idx, element) in elements.iter().enumerate() {
                        element_warnings(element, idx, &mut sink);
                    }
                }
                _ => sink.push_with(|| {
                    format!("{P} should include 'compositional_deconstruction.elements' as a list.")
                }),
            }
        }
        _ => sink
            .push_with(|| format!("{P} should include a 'compositional_deconstruction' object.")),
    }
    match caption.get("style_description") {
        Some(Value::Object(style)) => {
            let has_photo = style.contains_key("photo");
            let has_art = style.contains_key("art_style");
            if has_photo == has_art {
                sink.push_with(|| format!(
                    "{P} 'style_description' should include exactly one of 'photo' or 'art_style'."
                ));
            }
            for key in ["aesthetics", "lighting", "medium"] {
                if !is_non_empty_string(style.get(key)) {
                    sink.push_with(|| {
                        format!(
                            "{P} 'style_description' should include '{key}' as a non-empty string."
                        )
                    });
                }
            }
            if has_photo && !is_non_empty_string(style.get("photo")) {
                sink.push_with(|| {
                    format!("{P} 'style_description.photo' should be a non-empty string.")
                });
            }
            if has_art && !is_non_empty_string(style.get("art_style")) {
                sink.push_with(|| {
                    format!("{P} 'style_description.art_style' should be a non-empty string.")
                });
            }
        }
        // Upstream treats a JSON null like a missing key (`style is not None`).
        None | Some(Value::Null) => {}
        Some(_) => sink.push_with(|| format!("{P} 'style_description' should be an object.")),
    }
    let mut path = String::from("$");
    walk_object(caption, &mut path, &mut sink);
    (sink.issues, sink.dropped)
}

fn element_warnings(value: &Value, idx: usize, sink: &mut Sink) {
    const P: &str = "Ideogram 4 JSON caption";
    if sink.full() {
        sink.dropped += 1;
        return;
    }
    let path = format!("compositional_deconstruction.elements[{idx}]");
    let Value::Object(element) = value else {
        sink.push_with(|| format!("{P} '{path}' should be an object."));
        return;
    };
    let element_type = element.get("type").and_then(Value::as_str);
    if !matches!(element_type, Some("obj" | "text")) {
        sink.push_with(|| format!("{P} '{path}.type' should be 'obj' or 'text'."));
    }
    if !is_non_empty_string(element.get("desc")) {
        sink.push_with(|| format!("{P} '{path}.desc' should be a non-empty string."));
    }
    if element_type == Some("text") && !element.get("text").is_some_and(Value::is_string) {
        sink.push_with(|| format!("{P} '{path}.text' should be a string."));
    }
    if let Some(bbox) = element.get("bbox")
        && let Some(issue) = bbox_warning(bbox, &format!("{path}.bbox"))
    {
        sink.push_with(|| issue);
    }
}

fn bbox_warning(value: &Value, path: &str) -> Option<String> {
    const P: &str = "Ideogram 4 JSON caption";
    let items = match value {
        Value::Array(items) if items.len() == 4 => items,
        _ => {
            return Some(format!(
                "{P} '{path}' should contain four integer coordinates."
            ));
        }
    };
    // Python's `isinstance(item, int)` excludes floats (and bools, which JSON
    // keeps distinct anyway).
    let coords: Option<Vec<i64>> = items
        .iter()
        .map(|v| v.as_i64().filter(|n| (0..=1000).contains(n)))
        .collect();
    let Some(c) = coords else {
        return Some(format!(
            "{P} '{path}' coordinates should be integers between 0 and 1000."
        ));
    };
    (c[0] >= c[2] || c[1] >= c[3])
        .then(|| format!("{P} '{path}' should satisfy y_min < y_max and x_min < x_max."))
}

/// Walk a map's entries with one shared path buffer: each segment is pushed,
/// recursed into and truncated away, so no node allocates its own path.
fn walk_object(map: &Map<String, Value>, path: &mut String, sink: &mut Sink) {
    for (key, nested) in map {
        let mark = path.len();
        path.push('.');
        path.push_str(key);
        if key == "color_palette" {
            validate_color_palette(nested, path, sink);
        } else {
            color_palette_warnings(nested, path, sink);
        }
        path.truncate(mark);
    }
}

fn color_palette_warnings(value: &Value, path: &mut String, sink: &mut Sink) {
    match value {
        Value::Object(map) => walk_object(map, path, sink),
        Value::Array(items) => {
            for (idx, nested) in items.iter().enumerate() {
                let mark = path.len();
                let _ = write!(path, "[{idx}]");
                color_palette_warnings(nested, path, sink);
                path.truncate(mark);
            }
        }
        _ => {}
    }
}

fn is_hex_color(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 7
        && b[0] == b'#'
        && b[1..]
            .iter()
            .all(|c| c.is_ascii_digit() || (b'A'..=b'F').contains(c))
}

fn validate_color_palette(value: &Value, path: &str, sink: &mut Sink) {
    const P: &str = "Ideogram 4 JSON caption";
    let Value::Array(colors) = value else {
        sink.push_with(|| format!("{P} '{path}' should be a list of hex colors."));
        return;
    };
    for (idx, color) in colors.iter().enumerate() {
        if !color.as_str().is_some_and(is_hex_color) {
            sink.push_with(|| {
                format!("{P} '{path}[{idx}]' should be an uppercase #RRGGBB hex color.")
            });
        }
    }
}

#[cfg(test)]
#[path = "prompting_tests.rs"]
mod tests;
