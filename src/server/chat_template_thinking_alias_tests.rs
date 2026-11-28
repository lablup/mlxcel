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

//! The bare `thinking` kwarg alias against mlx-lm#1974 (#3209 item a): a
//! template that tests `thinking is defined` and never reads
//! `enable_thinking` gets the resolved thinking switch as `thinking`.

use super::{ChatMessage, ChatTemplateKwargs, ChatTemplateProcessor};

const BARE_THINKING: &str = "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}{% if thinking is defined and thinking %}<think>{% endif %}{% endif %}";

fn hi() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: "hi".to_string(),
    }]
}

fn render(processor: &ChatTemplateProcessor, kwargs: &str) -> String {
    let kwargs = ChatTemplateKwargs::from_json_str(kwargs).unwrap();
    processor.apply_with_kwargs(&hi(), None, &kwargs).unwrap()
}

#[test]
fn bare_thinking_is_defined_gets_alias() {
    let mut processor = ChatTemplateProcessor::with_template(BARE_THINKING.to_string());
    assert!(processor.wants_bare_thinking_alias());

    // The thinking default on (a think-marker tokenizer) reaches `thinking`.
    processor.set_default_enable_thinking(true);
    assert_eq!(render(&processor, "{}"), "hi<think>");
    // A caller's `enable_thinking=false` turns it off through the alias.
    assert_eq!(render(&processor, r#"{"enable_thinking": false}"#), "hi");
    // An explicit `thinking` kwarg still wins over the alias.
    assert_eq!(
        render(
            &processor,
            r#"{"enable_thinking": false, "thinking": true}"#
        ),
        "hi<think>"
    );

    // With the default off and no kwarg the off branch renders.
    let off = ChatTemplateProcessor::with_template(BARE_THINKING.to_string());
    assert_eq!(render(&off, "{}"), "hi");
    assert_eq!(render(&off, r#"{"enable_thinking": true}"#), "hi<think>");
}

#[test]
fn enable_thinking_template_gets_no_alias() {
    // A template that reads `enable_thinking` as well as testing
    // `thinking is defined` keeps the conventional kwarg alone.
    let template = "{% if enable_thinking is defined and enable_thinking %}E{% endif %}{% if thinking is defined and thinking %}T{% endif %}";
    let mut processor = ChatTemplateProcessor::with_template(template.to_string());
    assert!(!processor.wants_bare_thinking_alias());
    processor.set_default_enable_thinking(true);
    assert_eq!(render(&processor, "{}"), "E");

    // The DeepSeek-V3.2 idiom keeps matching, and GLM-5.2's
    // `enable_thinking is not defined` form keeps not matching.
    let dsv32 = ChatTemplateProcessor::with_template(
        "{% if not thinking is defined %}{% set thinking = false %}{% endif %}".to_string(),
    );
    assert!(dsv32.wants_bare_thinking_alias());
    let glm5 = ChatTemplateProcessor::with_template(
        "{%- if (enable_thinking is not defined or enable_thinking) -%}THINK{%- endif -%}"
            .to_string(),
    );
    assert!(!glm5.wants_bare_thinking_alias());
    // `enable_thinking is defined` alone is not a bare `thinking` test.
    let qwen = ChatTemplateProcessor::with_template(
        "{% if enable_thinking is defined and enable_thinking %}<think>{% endif %}".to_string(),
    );
    assert!(!qwen.wants_bare_thinking_alias());
}
