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

//! Shape 2: span extraction with labels chosen per request.
//!
//! The types mirror the GLiNER2.5 Python API in mlx-vlm
//! (`extract_entities` / `classify_text` in
//! `mlx_vlm/models/gliner2_5/gliner2_5.py`): [`EntityResult`] serializes
//! exactly like upstream's `{"entities": {...}}` dict and [`ClassifyTask`]
//! accepts the same task specs. Only the trait, the types and the CLI live
//! here; a family implements [`SpanExtractorModel`] in its own module.

use anyhow::{Context, Result, bail};
use serde::de::{self, Deserializer};
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// What each extracted entity carries besides its text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExtractOptions {
    /// Add the span's `confidence`.
    pub include_confidence: bool,
    /// Add the span's character `start` / `end`.
    pub include_spans: bool,
}

/// One extracted entity: the matched string, or an object when
/// [`ExtractOptions`] asks for more than the text.
#[derive(Debug, Clone, PartialEq)]
pub enum EntityItem {
    /// Serializes as the bare matched string.
    Text(String),
    /// Serializes as `{"text", "confidence"?, "start"?, "end"?}`.
    Detailed {
        text: String,
        confidence: Option<f32>,
        start: Option<usize>,
        end: Option<usize>,
    },
}

impl EntityItem {
    /// Format one match the way upstream `extract_entities` does: a bare
    /// string unless confidence or spans were requested. `start` / `end` are
    /// Unicode scalar offsets into the input text.
    pub fn new(
        text: impl Into<String>,
        confidence: f32,
        start: usize,
        end: usize,
        options: ExtractOptions,
    ) -> Self {
        let text = text.into();
        if !options.include_confidence && !options.include_spans {
            return Self::Text(text);
        }
        Self::Detailed {
            text,
            confidence: options.include_confidence.then_some(confidence),
            start: options.include_spans.then_some(start),
            end: options.include_spans.then_some(end),
        }
    }

    /// The matched string.
    pub fn text(&self) -> &str {
        match self {
            Self::Text(text) | Self::Detailed { text, .. } => text,
        }
    }
}

impl Serialize for EntityItem {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Text(text) => serializer.serialize_str(text),
            Self::Detailed {
                text,
                confidence,
                start,
                end,
            } => {
                let fields = 1
                    + usize::from(confidence.is_some())
                    + usize::from(start.is_some())
                    + usize::from(end.is_some());
                let mut map = serializer.serialize_map(Some(fields))?;
                map.serialize_entry("text", text)?;
                if let Some(confidence) = confidence {
                    map.serialize_entry("confidence", confidence)?;
                }
                if let Some(start) = start {
                    map.serialize_entry("start", start)?;
                }
                if let Some(end) = end {
                    map.serialize_entry("end", end)?;
                }
                map.end()
            }
        }
    }
}

/// `{"entities": {"<label>": [...]}}`, labels in request order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EntityResult {
    pub entities: Vec<(String, Vec<EntityItem>)>,
}

impl Serialize for EntityResult {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        struct Entities<'a>(&'a [(String, Vec<EntityItem>)]);
        impl Serialize for Entities<'_> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut map = serializer.serialize_map(Some(self.0.len()))?;
                for (label, items) in self.0 {
                    map.serialize_entry(label, items)?;
                }
                map.end()
            }
        }
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry("entities", &Entities(&self.entities))?;
        map.end()
    }
}

/// A `{"label": "description", ...}` JSON object read in key order.
fn ordered_descriptions<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<(String, String)>>, D::Error> {
    let Some(map) = Option::<serde_json::Map<String, Value>>::deserialize(deserializer)? else {
        return Ok(None);
    };
    map.into_iter()
        .map(|(label, value)| match value {
            Value::String(description) => Ok((label, description)),
            other => Err(de::Error::custom(format!(
                "label_descriptions['{label}'] must be a string, got {other}"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// One `classify_text` task: a label list, or the full spec object.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum ClassifyTask {
    /// `["positive", "negative"]`: single-label, the shared threshold.
    Labels(Vec<String>),
    /// `{"labels", "prompt"?, "label_descriptions"?, "multi_label"?, "threshold"?}`.
    Spec {
        labels: Vec<String>,
        #[serde(default)]
        prompt: Option<String>,
        #[serde(default, deserialize_with = "ordered_descriptions")]
        label_descriptions: Option<Vec<(String, String)>>,
        #[serde(default)]
        multi_label: bool,
        #[serde(default)]
        threshold: Option<f32>,
    },
}

impl ClassifyTask {
    pub fn labels(&self) -> &[String] {
        match self {
            Self::Labels(labels) | Self::Spec { labels, .. } => labels,
        }
    }

    pub fn prompt(&self) -> Option<&str> {
        match self {
            Self::Labels(_) => None,
            Self::Spec { prompt, .. } => prompt.as_deref(),
        }
    }

    /// The description of `label`, if the task gives one.
    pub fn label_description(&self, label: &str) -> Option<&str> {
        match self {
            Self::Labels(_) => None,
            Self::Spec {
                label_descriptions, ..
            } => label_descriptions
                .as_deref()?
                .iter()
                .find(|(name, _)| name == label)
                .map(|(_, description)| description.as_str()),
        }
    }

    /// `true` returns every label at or above the threshold, `false` the one
    /// best label.
    pub fn multi_label(&self) -> bool {
        matches!(self, Self::Spec { multi_label: true, .. })
    }

    /// The task's own threshold, falling back to the request's.
    pub fn threshold(&self, default: f32) -> f32 {
        match self {
            Self::Spec {
                threshold: Some(threshold),
                ..
            } => *threshold,
            _ => default,
        }
    }
}

/// A span extractor: labels and tasks arrive with every request.
pub trait SpanExtractorModel {
    /// Extract spans for `(label, description)` pairs. Returns
    /// `{"entities": {"<label>": [...]}}` with one key per requested label,
    /// in request order, even when a label has no match.
    fn extract_entities(
        &mut self,
        text: &str,
        labels: &[(String, Option<String>)],
        threshold: f32,
        options: ExtractOptions,
    ) -> Result<EntityResult>;

    /// Classify `text` once per task. Returns `{"<task>": "<label>"}`, or
    /// `{"<task>": ["<label>", ...]}` for a multi-label task.
    fn classify_text(
        &mut self,
        text: &str,
        tasks: &[(String, ClassifyTask)],
        threshold: f32,
    ) -> Result<Value>;
}

impl<T: SpanExtractorModel + ?Sized> SpanExtractorModel for Box<T> {
    fn extract_entities(
        &mut self,
        text: &str,
        labels: &[(String, Option<String>)],
        threshold: f32,
        options: ExtractOptions,
    ) -> Result<EntityResult> {
        (**self).extract_entities(text, labels, threshold, options)
    }

    fn classify_text(
        &mut self,
        text: &str,
        tasks: &[(String, ClassifyTask)],
        threshold: f32,
    ) -> Result<Value> {
        (**self).classify_text(text, tasks, threshold)
    }
}

/// `--labels L1,L2`: trimmed, empty entries dropped, no descriptions.
pub fn parse_label_list(raw: &str) -> Result<Vec<(String, Option<String>)>> {
    let labels: Vec<(String, Option<String>)> = raw
        .split(',')
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(|label| (label.to_string(), None))
        .collect();
    if labels.is_empty() {
        bail!("--labels names no label");
    }
    Ok(labels)
}

/// `--labels-json`: `{"label": "description", ...}` in key order, or a JSON
/// array of label strings. A non-string description is dropped, as upstream
/// keeps only `str` values.
pub fn parse_labels_json(raw: &str) -> Result<Vec<(String, Option<String>)>> {
    let value: Value = serde_json::from_str(raw).context("--labels-json is not valid JSON")?;
    let labels: Vec<(String, Option<String>)> = match value {
        Value::Object(map) => map
            .into_iter()
            .map(|(label, description)| (label, description.as_str().map(str::to_string)))
            .collect(),
        Value::Array(items) => items
            .into_iter()
            .map(|item| match item {
                Value::String(label) => Ok((label, None)),
                other => bail!("--labels-json array entries must be strings, got {other}"),
            })
            .collect::<Result<_>>()?,
        other => bail!("--labels-json must be an object or an array, got {other}"),
    };
    if labels.is_empty() {
        bail!("--labels-json names no label");
    }
    Ok(labels)
}

/// `--classify '<tasks JSON>'`: `{"<task>": [labels] | {spec}}` in key order.
pub fn parse_classify_tasks(raw: &str) -> Result<Vec<(String, ClassifyTask)>> {
    let value: Value = serde_json::from_str(raw).context("--classify is not valid JSON")?;
    let Value::Object(map) = value else {
        bail!("--classify must be a JSON object mapping each task name to its labels");
    };
    if map.is_empty() {
        bail!("--classify names no task");
    }
    map.into_iter()
        .map(|(task, spec)| {
            let spec: ClassifyTask = serde_json::from_value(spec).with_context(|| {
                format!(
                    "task '{task}' must be a label list or {{\"labels\", \"prompt\"?, \
                     \"label_descriptions\"?, \"multi_label\"?, \"threshold\"?}}"
                )
            })?;
            if spec.labels().is_empty() {
                bail!("task '{task}' has no labels");
            }
            Ok((task, spec))
        })
        .collect()
}

#[cfg(test)]
#[path = "extract_tests.rs"]
mod extract_tests;
