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

use super::*;
use serde_json::{Value, json};

fn checkpoint(family: &str, format: &str, template: &str) -> ChatTemplateProcessor {
    let dir = tempfile::tempdir().unwrap();
    for (file, data) in [
        ("config.json", json!({"model_type":family})),
        ("tokenizer_config.json", json!({"chat_template":template})),
        ("processor_config.json", json!({"sft_format":format})),
    ] {
        std::fs::write(dir.path().join(file), data.to_string()).unwrap();
    }
    ChatTemplateProcessor::from_model_path(dir.path())
        .unwrap()
        .unwrap()
}

fn vl2(format: &str) -> ChatTemplateProcessor {
    checkpoint("deepseek_vl_v2", format, "### Instruction: coding")
}

fn user(content: Value) -> Value {
    json!([{"role":"user", "content":content}])
}

#[test]
fn deepseek_vl2_image_turns_follow_processor_format() {
    for (format, prefix, suffix) in [
        ("deepseek", "<|User|>: ", "\n\n<|Assistant|>:"),
        ("deepseekv2", "", "<｜end▁of▁sentence｜>"),
        ("plain", "", ""),
    ] {
        let processor = vl2(format);
        for count in [0, 1, 2, 3] {
            let mut parts = vec![json!({"type":"image"}); count];
            parts.push(json!({"type":"text","text":"A"}));
            let rendered = processor.apply_raw(&user(json!(parts)), None).unwrap();
            assert_eq!(
                rendered,
                format!("{prefix}{}A{suffix}", "<image>\n".repeat(count))
            );
        }
    }
}

#[test]
fn deepseek_vl2_explicit_markers_and_interleaved_images() {
    let processor = vl2("deepseek");
    let render = |parts| processor.apply_raw(&user(parts), None);
    let image = json!({"type":"image"});
    let text = |text: &str| json!({"type":"text","text":text});
    assert_eq!(
        render(json!([image, text("look <image> here")])).unwrap(),
        "<|User|>: look <image> here\n\n<|Assistant|>:"
    );
    assert_eq!(
        render(json!([text("A"), image, text("B"), image, text("C")])).unwrap(),
        "<|User|>: A<image>\nB<image>\nC\n\n<|Assistant|>:"
    );
    for invalid in [json!([image, text("<image><image>")]), json!("<image>")] {
        let error = render(invalid).unwrap_err();
        assert!(template_rejection_message(&error).is_some());
    }
}

#[test]
fn deepseek_vl2_preserves_system_and_assistant_turn_boundaries() {
    let turns = json!([
        {"role":"system","content":"  S  "},
        {"role":"user","content":"U"},
        {"role":"assistant","content":" A "},
        {"role":"user","content":" V "}
    ]);
    assert_eq!(
        vl2("deepseek").apply_raw(&turns, None).unwrap(),
        "S<|User|>: U\n\n<|Assistant|>:A<｜end▁of▁sentence｜><|User|>: V\n\n<|Assistant|>:"
    );
    assert_eq!(
        vl2("plain").apply_raw(&turns, None).unwrap(),
        "SUA<｜end▁of▁sentence｜>V"
    );
    assert_eq!(
        vl2("deepseekv2").apply_raw(&turns, None).unwrap(),
        "SU<｜end▁of▁sentence｜>A<｜end▁of▁sentence｜>V<｜end▁of▁sentence｜>"
    );
}

#[test]
fn deepseek_vl2_keeps_ocr_sibling_templates() {
    let shipped = "{% for message in messages %}{{message['content']}} {% endfor %}";
    for (family, expected) in [
        ("deepseekocr", "A "),
        ("deepseekocr_2", "A "),
        ("unlimited-ocr", "A"),
        ("GOT", "A "),
    ] {
        let p = checkpoint(family, "deepseek", shipped);
        assert_eq!(
            p.apply_raw(&user(json!("A")), None).unwrap(),
            expected,
            "{family}"
        );
    }
}

// Subset of tiny's vocabulary and merges, including the real ":A" merge.
// A joined string would encode a previous assistant answer as 48389 instead
// of preserving the question's colon (28) and answer (35).
fn tiny_tokenizer() -> crate::tokenizer::MlxcelTokenizer {
    let special = [
        (0, "<｜begin▁of▁sentence｜>"),
        (1, "<｜end▁of▁sentence｜>"),
        (128815, "<image>"),
        (128821, "<|User|>"),
        (128822, "<|Assistant|>"),
    ];
    let mut vocab =
        json!({":":28, "S":53, "A":35, " ":223, "\n":201, "\n\n":271, " A":334, ":A":48389});
    let mut added = Vec::new();
    for (id, token) in special {
        vocab[token] = json!(id);
        added.push(
            json!({"id":id,"content":token,"normalized":false,"special":true,
            "single_word":false,"lstrip":false,"rstrip":false}),
        );
    }
    let spec = json!({"version":"1.0","added_tokens":added,
        "model":{"type":"BPE","vocab":vocab,"merges":[["\n","\n"],[" ","A"],[":","A"]]}});
    crate::tokenizer::MlxcelTokenizer::HuggingFace(
        tokenizers::Tokenizer::from_bytes(spec.to_string()).unwrap(),
    )
}

#[tokio::test]
async fn deepseek_vl2_server_ids_keep_question_and_answer_encodings_separate() {
    use crate::server::{chat_request::prepare_chat_request, state::attach_native_chat_renderer};
    let request = serde_json::from_value(json!({"model":"tiny","messages":[
        {"role":"system","content":" S "},
        {"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,YQ=="}},{"type":"text","text":"A"}]},
        {"role":"assistant","content":"A"},
        {"role":"user","content":"A"}
    ]})).unwrap();
    for (format, expected) in [
        (
            "deepseek",
            vec![
                0, 53, 128821, 28, 223, 128815, 201, 35, 271, 128822, 28, 35, 1, 128821, 28, 334,
                271, 128822, 28,
            ],
        ),
        ("deepseekv2", vec![0, 53, 128815, 201, 35, 1, 35, 1, 35, 1]),
        ("plain", vec![0, 53, 128815, 201, 35, 35, 1, 35]),
    ] {
        let (_, processor) = attach_native_chat_renderer(tiny_tokenizer(), vl2(format));
        let prepared = prepare_chat_request(&processor, &request, None)
            .await
            .unwrap();
        assert_eq!(
            prepared.prompt_token_ids.as_deref(),
            Some(expected.as_slice()),
            "{format}"
        );
    }
}
