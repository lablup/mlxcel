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

//! Shape-2 serialization and request parsing, against the dict shapes
//! upstream GLiNER2.5 `extract_entities` / `classify_text` produce.

use serde_json::json;

use super::*;

#[test]
fn entity_items_serialize_like_upstream() {
    let plain = ExtractOptions::default();
    let both = ExtractOptions {
        include_confidence: true,
        include_spans: true,
    };
    let spans = ExtractOptions {
        include_spans: true,
        ..plain
    };
    let confidence = ExtractOptions {
        include_confidence: true,
        ..plain
    };
    let item = |options| serde_json::to_value(EntityItem::new("Ada", 0.5, 4, 7, options)).unwrap();
    assert_eq!(item(plain), json!("Ada"));
    assert_eq!(
        item(both),
        json!({"text": "Ada", "confidence": 0.5, "start": 4, "end": 7})
    );
    assert_eq!(item(spans), json!({"text": "Ada", "start": 4, "end": 7}));
    assert_eq!(item(confidence), json!({"text": "Ada", "confidence": 0.5}));
}

#[test]
fn entity_result_keeps_label_order_and_empty_labels() {
    let result = EntityResult {
        entities: vec![
            ("person".to_string(), vec![EntityItem::Text("Ada".to_string())]),
            ("company".to_string(), Vec::new()),
            ("city".to_string(), vec![EntityItem::Text("Paris".to_string())]),
        ],
    };
    let text = serde_json::to_string(&result).unwrap();
    assert_eq!(
        text,
        r#"{"entities":{"person":["Ada"],"company":[],"city":["Paris"]}}"#
    );
    assert_eq!(
        serde_json::to_string(&EntityResult::default()).unwrap(),
        r#"{"entities":{}}"#
    );
}

#[test]
fn label_flags_parse() {
    assert_eq!(
        parse_label_list(" person, company ,,").unwrap(),
        [("person".to_string(), None), ("company".to_string(), None)]
    );
    assert!(parse_label_list(" , ").is_err());

    let described = parse_labels_json(r#"{"person": "a human", "company": null}"#).unwrap();
    assert_eq!(
        described,
        [
            ("person".to_string(), Some("a human".to_string())),
            ("company".to_string(), None)
        ]
    );
    assert_eq!(
        parse_labels_json(r#"["b", "a"]"#).unwrap(),
        [("b".to_string(), None), ("a".to_string(), None)]
    );
    assert!(parse_labels_json("{}").is_err());
    assert!(parse_labels_json("3").is_err());
    assert!(parse_labels_json("{not json").is_err());
}

#[test]
fn classify_tasks_parse_both_spellings_in_order() {
    let tasks = parse_classify_tasks(
        r#"{
            "sentiment": ["positive", "negative"],
            "topics": {"labels": ["tech", "sports"], "multi_label": true, "threshold": 0.3,
                       "prompt": "Which topics?", "label_descriptions": {"tech": "technology"}}
        }"#,
    )
    .unwrap();
    assert_eq!(tasks.len(), 2);
    let (name, sentiment) = &tasks[0];
    assert_eq!(name, "sentiment");
    assert_eq!(sentiment.labels(), ["positive", "negative"]);
    assert!(!sentiment.multi_label());
    assert_eq!(sentiment.threshold(0.5), 0.5);
    assert_eq!(sentiment.prompt(), None);

    let (name, topics) = &tasks[1];
    assert_eq!(name, "topics");
    assert!(topics.multi_label());
    assert_eq!(topics.threshold(0.5), 0.3);
    assert_eq!(topics.prompt(), Some("Which topics?"));
    assert_eq!(topics.label_description("tech"), Some("technology"));
    assert_eq!(topics.label_description("sports"), None);

    assert!(parse_classify_tasks(r#"["a"]"#).is_err());
    assert!(parse_classify_tasks("{}").is_err());
    assert!(parse_classify_tasks(r#"{"t": []}"#).is_err());
    assert!(parse_classify_tasks(r#"{"t": {"labels": []}}"#).is_err());
    assert!(parse_classify_tasks(r#"{"t": {"prompt": "x"}}"#).is_err());
}
