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

//! Expected strings are upstream's (`mlx_vlm.models.ideogram4.prompting`)
//! output, printed with `repr`.

use super::*;

#[test]
fn plain_text_is_wrapped_like_upstream() {
    let p = normalize_prompt("A cat", true).unwrap();
    assert!(p.was_wrapped && p.is_json_caption && p.is_structured_caption);
    assert_eq!(
        p.text,
        r#"{"high_level_description":"A cat","compositional_deconstruction":{"background":"The setting, environment, and surrounding context implied by the prompt.","elements":[{"type":"obj","desc":"A cat"}]}}"#
    );
    assert_eq!(p.revised_prompt().as_deref(), Some(p.text.as_str()));
}

#[test]
fn wrapping_escapes_like_python_json_dumps() {
    // Leading/trailing whitespace is stripped; quotes, newline, tab and C0
    // controls are escaped; non-ASCII (and U+2028) is kept raw.
    let p = normalize_prompt("  He said \"hi\"\n\tok \u{e9} \u{2028} \u{1} ", true).unwrap();
    assert_eq!(
        p.text,
        "{\"high_level_description\":\"He said \\\"hi\\\"\\n\\tok \u{e9} \u{2028} \\u0001\",\"compositional_deconstruction\":{\"background\":\"The setting, environment, and surrounding context implied by the prompt.\",\"elements\":[{\"type\":\"obj\",\"desc\":\"He said \\\"hi\\\"\\n\\tok \u{e9} \u{2028} \\u0001\"}]}}"
    );
}

#[test]
fn auto_json_caption_false_passes_text_through() {
    let p = normalize_prompt("plain", false).unwrap();
    assert_eq!(p.text, "plain");
    assert!(!p.was_wrapped && !p.is_json_caption && !p.is_structured_caption);
    assert_eq!(p.revised_prompt(), None);
}

#[test]
fn a_json_caption_passes_through_unchanged() {
    let prompt = r#" {"high_level_description":"A red square icon.","compositional_deconstruction":{"background":"A white background.","elements":[{"type":"obj","desc":"A centered red square."}]}}"#;
    let p = normalize_prompt(prompt, true).unwrap();
    assert_eq!(p.text, prompt);
    assert!(p.is_json_caption && p.is_structured_caption && !p.was_wrapped);
    assert!(p.warnings.is_empty());
    assert_eq!(p.revised_prompt(), None);
}

#[test]
fn malformed_json_captions_are_errors() {
    assert_eq!(
        normalize_prompt("{not json", true).unwrap_err().to_string(),
        "Invalid Ideogram 4 JSON caption"
    );
    // `[...]` does not start with `{`, so it is plain text and gets wrapped.
    assert!(normalize_prompt("[1, 2]", true).unwrap().was_wrapped);
}

#[test]
fn caption_warnings_match_upstream() {
    let caption = r##"{"compositional_deconstruction": {"background": " ", "elements": [{"type": "img", "desc": "", "bbox": [0, 0, 10]}, {"type": "text", "desc": "x", "bbox": [5, 5, 1, 9], "color_palette": ["#ff0000", "#00FF00"]}, 3]}, "style_description": {"photo": "p", "art_style": "a", "lighting": "l", "medium": "m"}}"##;
    let p = normalize_prompt(caption, true).unwrap();
    assert!(p.is_json_caption && !p.is_structured_caption);
    let want = [
        "Ideogram 4 JSON caption should include 'compositional_deconstruction.background' as a non-empty string.",
        "Ideogram 4 JSON caption 'compositional_deconstruction.elements[0].type' should be 'obj' or 'text'.",
        "Ideogram 4 JSON caption 'compositional_deconstruction.elements[0].desc' should be a non-empty string.",
        "Ideogram 4 JSON caption 'compositional_deconstruction.elements[0].bbox' should contain four integer coordinates.",
        "Ideogram 4 JSON caption 'compositional_deconstruction.elements[1].text' should be a string.",
        "Ideogram 4 JSON caption 'compositional_deconstruction.elements[1].bbox' should satisfy y_min < y_max and x_min < x_max.",
        "Ideogram 4 JSON caption 'compositional_deconstruction.elements[2]' should be an object.",
        "Ideogram 4 JSON caption 'style_description' should include exactly one of 'photo' or 'art_style'.",
        "Ideogram 4 JSON caption 'style_description' should include 'aesthetics' as a non-empty string.",
        "Ideogram 4 JSON caption '$.compositional_deconstruction.elements[1].color_palette[0]' should be an uppercase #RRGGBB hex color.",
    ];
    assert_eq!(p.warnings, want);
}

#[test]
fn warnings_are_capped_and_a_large_palette_is_fast() {
    // One long key holding 10,000 bad palette entries and 10,000 bad
    // elements: the list stays at the cap, the rest is counted, and the walk
    // does not rebuild the key path per node.
    let key = "k".repeat(10_000);
    let colors = vec!["red"; 10_000].join("\",\"");
    let elements = vec!["3"; 10_000].join(",");
    let caption = format!(
        r#"{{"compositional_deconstruction":{{"background":"b","elements":[{elements}]}},"{key}":[{{"color_palette":["{colors}"]}}]}}"#
    );
    let start = std::time::Instant::now();
    let p = normalize_prompt(&caption, true).unwrap();
    assert!(start.elapsed() < std::time::Duration::from_secs(5));
    assert_eq!(p.warnings.len(), MAX_WARNINGS);
    assert!(p.warnings_dropped >= 10_000, "{}", p.warnings_dropped);
    assert!(p.is_json_caption && !p.is_structured_caption);
}

#[test]
fn a_dropped_warning_still_means_not_structured() {
    let elements = vec!["3"; MAX_WARNINGS + 5].join(",");
    let caption = format!(
        r#"{{"compositional_deconstruction":{{"background":"b","elements":[{elements}]}}}}"#
    );
    let p = normalize_prompt(&caption, true).unwrap();
    assert_eq!(p.warnings.len(), MAX_WARNINGS);
    assert_eq!(p.warnings_dropped, 5);
    assert!(!p.is_structured_caption);
}
