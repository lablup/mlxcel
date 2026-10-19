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

//! `mlxcel tag` / `mlxcel extract` through the real `Cli` parser: flag
//! parsing, request validation, and dispatch to a stub span extractor.

use clap::{CommandFactory, Parser};
use serde_json::json;

use mlxcel::token_classification::{EntityItem, EntityResult};

use super::*;
use crate::{Cli, Commands};

fn parse(args: &[&str]) -> Commands {
    let args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || Cli::try_parse_from(args).map(|cli| cli.command))
        .expect("spawn CLI parser thread")
        .join()
        .expect("CLI parser thread panicked")
        .expect("CLI args must parse")
}

fn parse_err(args: &[&str]) -> String {
    let args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || Cli::try_parse_from(args).err().map(|e| e.to_string()))
        .expect("spawn CLI parser thread")
        .join()
        .expect("CLI parser thread panicked")
        .expect("CLI args must be rejected")
}

fn extract(args: &[&str]) -> ExtractArgs {
    let mut full = vec!["mlxcel", "extract", "-m", "models/x"];
    full.extend_from_slice(args);
    match parse(&full) {
        Commands::Extract(args) => args,
        other => panic!("expected extract, got {other:?}"),
    }
}

/// Records what it was asked and answers with a fixed shape.
#[derive(Default)]
struct StubExtractor {
    entity_calls: Vec<(String, Vec<(String, Option<String>)>, f32, ExtractOptions)>,
    classify_calls: Vec<(String, Vec<String>, f32)>,
}

impl SpanExtractorModel for StubExtractor {
    fn extract_entities(
        &mut self,
        text: &str,
        labels: &[(String, Option<String>)],
        threshold: f32,
        options: ExtractOptions,
    ) -> anyhow::Result<EntityResult> {
        self.entity_calls
            .push((text.to_string(), labels.to_vec(), threshold, options));
        Ok(EntityResult {
            entities: labels
                .iter()
                .map(|(label, _)| {
                    (
                        label.clone(),
                        vec![EntityItem::new("Ada", 0.9, 0, 3, options)],
                    )
                })
                .collect(),
        })
    }

    fn classify_text(
        &mut self,
        text: &str,
        tasks: &[(String, ClassifyTask)],
        threshold: f32,
    ) -> anyhow::Result<Value> {
        self.classify_calls.push((
            text.to_string(),
            tasks.iter().map(|(name, _)| name.clone()).collect(),
            threshold,
        ));
        let mut out = serde_json::Map::new();
        for (name, task) in tasks {
            let value = if task.multi_label() {
                json!(task.labels())
            } else {
                json!(task.labels()[0])
            };
            out.insert(name.clone(), value);
        }
        Ok(Value::Object(out))
    }
}

#[test]
fn tag_flags_parse_like_upstream_run_cli() {
    let Commands::Tag(args) = parse(&[
        "mlxcel",
        "tag",
        "-m",
        "nativ-community/rampart-mlx-fp16",
        "--keep",
        "CITY,STATE,,ZIP_CODE",
        "--argmax",
        "--replacement",
        "[x]",
        "I'm Sarah Connor",
    ]) else {
        panic!("expected tag");
    };
    assert_eq!(args.text, "I'm Sarah Connor");
    let options = args.options();
    assert_eq!(options.keep_labels, ["CITY", "STATE", "ZIP_CODE"]);
    assert_eq!(
        options.decode,
        Some(mlxcel::token_classification::DecodeMode::Argmax)
    );
    assert_eq!(options.replacement.as_deref(), Some("[x]"));
    assert!(options.trim_whitespace);

    let Commands::Tag(bare) = parse(&["mlxcel", "tag", "-m", "m", "text"]) else {
        panic!("expected tag");
    };
    let options = bare.options();
    assert_eq!(options.decode, None, "the model's default decode mode");
    assert!(options.keep_labels.is_empty());
    assert!(parse_err(&["mlxcel", "tag", "-m", "m"]).contains("TEXT"));
}

#[test]
fn extract_entities_dispatches_to_the_model() {
    let args = extract(&[
        "--labels",
        "person, company",
        "--threshold",
        "0.3",
        "--include-spans",
        "Ada",
    ]);
    let request = args.request().unwrap();
    let mut model = StubExtractor::default();
    let output = execute(&mut model, &args.text, args.threshold, &request).unwrap();
    assert_eq!(
        output,
        json!({"entities": {"person": [{"text": "Ada", "start": 0, "end": 3}],
                            "company": [{"text": "Ada", "start": 0, "end": 3}]}})
    );
    let (text, labels, threshold, options) = &model.entity_calls[0];
    assert_eq!(text, "Ada");
    assert_eq!(
        labels,
        &[("person".to_string(), None), ("company".to_string(), None)]
    );
    assert_eq!(*threshold, 0.3);
    assert!(options.include_spans && !options.include_confidence);
    assert!(model.classify_calls.is_empty());

    let args = extract(&["--labels-json", r#"{"person": "a human"}"#, "Ada"]);
    let ExtractRequest::Entities { labels, options } = args.request().unwrap() else {
        panic!("entities request");
    };
    assert_eq!(
        labels,
        [("person".to_string(), Some("a human".to_string()))]
    );
    assert_eq!(options, ExtractOptions::default());
    assert_eq!(args.threshold, 0.5, "upstream default threshold");
}

#[test]
fn extract_classify_dispatches_to_the_model() {
    let args = extract(&[
        "--classify",
        r#"{"sentiment": ["positive", "negative"], "topics": {"labels": ["a", "b"], "multi_label": true}}"#,
        "I love it.",
    ]);
    let request = args.request().unwrap();
    let mut model = StubExtractor::default();
    let output = execute(&mut model, &args.text, args.threshold, &request).unwrap();
    assert_eq!(
        output,
        json!({"sentiment": "positive", "topics": ["a", "b"]})
    );
    assert_eq!(model.classify_calls[0].1, ["sentiment", "topics"]);
    assert!(model.entity_calls.is_empty());
}

#[test]
fn extract_rejects_bad_flag_combinations() {
    assert!(parse_err(&["mlxcel", "extract", "-m", "m", "text"]).contains("--labels"));
    let both = parse_err(&[
        "mlxcel",
        "extract",
        "-m",
        "m",
        "--labels",
        "a",
        "--classify",
        "{}",
        "t",
    ]);
    assert!(both.contains("cannot be used with"), "{both}");

    let confidence = extract(&["--classify", r#"{"t": ["a"]}"#, "--include-confidence", "t"]);
    assert!(confidence.request().is_err());
    let threshold = extract(&["--labels", "a", "--threshold", "1.5", "t"]);
    assert!(threshold.request().is_err());
    let empty = extract(&["--labels", " , ", "t"]);
    assert!(empty.request().is_err());
}

#[test]
fn tag_and_extract_are_discoverable_in_help() {
    let top_level = Cli::command().render_long_help().to_string();
    for expected in [
        "tag",
        "Tag and redact entities with a token-classification checkpoint",
        "extract",
        "Extract spans for labels chosen per request",
    ] {
        assert!(top_level.contains(expected), "missing {expected:?}");
    }
    let mut command = Cli::command();
    let tag = command
        .find_subcommand_mut("tag")
        .unwrap()
        .render_long_help()
        .to_string();
    for option in [
        "--model",
        "--argmax",
        "--replacement",
        "--keep",
        "--operating-point",
        "--models-dir",
    ] {
        assert!(
            tag.contains(option),
            "`mlxcel tag --help` is missing {option}"
        );
    }
    let mut command = Cli::command();
    let extract = command
        .find_subcommand_mut("extract")
        .unwrap()
        .render_long_help()
        .to_string();
    for option in [
        "--labels",
        "--labels-json",
        "--classify",
        "--threshold",
        "--include-confidence",
        "--include-spans",
    ] {
        assert!(
            extract.contains(option),
            "`mlxcel extract --help` is missing {option}"
        );
    }
}
