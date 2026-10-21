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

//! Per-compaction image cache for `POST /v1/responses/compact` (issue #2397).
//!
//! One compaction renders its history several times (the full input, each
//! selection candidate, the summary prompt, the compacted result), and every
//! render resolves the image URLs it carries. [`MediaCache`] reads each
//! non-`data:` image URL once and hands later renders an equivalent `data:`
//! URI, so a compaction fetches each URL at most once. Only the copies that
//! are rendered are rewritten; the compact `output` and the capsule keep the
//! URLs the client sent.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;

use crate::server::types::responses_request::{
    ResponseInputContent, ResponseInputItem, ResponseInputPart, ResponseToolOutput,
};

/// Image URL to `data:` URI, for the lifetime of one compaction. A URL that
/// could not be read maps to `None` and is left as sent, so the render reports
/// it exactly as `/v1/responses` would.
#[derive(Debug, Default)]
pub(crate) struct MediaCache {
    entries: Mutex<HashMap<String, Option<String>>>,
}

impl MediaCache {
    /// Copies of `items` with every cached-or-fetched image URL inlined.
    pub(crate) async fn localize(&self, items: &[ResponseInputItem]) -> Vec<ResponseInputItem> {
        let mut items = items.to_vec();
        for url in remote_image_urls(&items) {
            self.resolve(&url).await;
        }
        let entries = match self.entries.lock() {
            Ok(entries) => entries,
            Err(poisoned) => poisoned.into_inner(),
        };
        for item in &mut items {
            for url in image_urls_mut(item) {
                if let Some(Some(inline)) = entries.get(url.as_str()) {
                    *url = inline.clone();
                }
            }
        }
        items
    }

    async fn resolve(&self, url: &str) {
        let known = self
            .entries
            .lock()
            .map(|entries| entries.contains_key(url))
            .unwrap_or(false);
        if known {
            return;
        }
        let limits = crate::server::media::current_image_input_limits();
        let inline = match crate::server::media::try_read_image_url_with_limits(url, limits).await {
            Ok(Some(bytes)) => Some(data_uri(&bytes)),
            Ok(None) => None,
            Err(err) => {
                tracing::debug!(error = %err, "compaction image prefetch failed; render reports it");
                None
            }
        };
        if let Ok(mut entries) = self.entries.lock() {
            entries.insert(url.to_string(), inline);
        }
    }
}

fn data_uri(bytes: &[u8]) -> String {
    let mime = image::guess_format(bytes)
        .map(|format| format.to_mime_type())
        .unwrap_or("image/png");
    format!("data:{mime};base64,{}", STANDARD.encode(bytes))
}

/// Distinct image URLs in `items` that a render would have to read.
fn remote_image_urls(items: &[ResponseInputItem]) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    for part in items.iter().flat_map(parts_of) {
        let url = match part {
            ResponseInputPart::InputImage {
                image_url: Some(url),
                ..
            } => url,
            ResponseInputPart::ImageUrl { image_url } => &image_url.url,
            _ => continue,
        };
        if !url.starts_with("data:") && seen.insert(url.as_str()) {
            urls.push(url.clone());
        }
    }
    urls
}

fn parts_of(item: &ResponseInputItem) -> &[ResponseInputPart] {
    match item {
        ResponseInputItem::Message {
            content: ResponseInputContent::Parts(parts),
            ..
        }
        | ResponseInputItem::FunctionCallOutput {
            output: ResponseToolOutput::Parts(parts),
            ..
        } => parts,
        _ => &[],
    }
}

/// Every image URL slot in a message or a function output.
fn image_urls_mut(item: &mut ResponseInputItem) -> Vec<&mut String> {
    let parts = match item {
        ResponseInputItem::Message {
            content: ResponseInputContent::Parts(parts),
            ..
        }
        | ResponseInputItem::FunctionCallOutput {
            output: ResponseToolOutput::Parts(parts),
            ..
        } => parts,
        _ => return Vec::new(),
    };
    parts
        .iter_mut()
        .filter_map(|part| match part {
            ResponseInputPart::InputImage {
                image_url: Some(url),
                ..
            } => Some(url),
            ResponseInputPart::ImageUrl { image_url } => Some(&mut image_url.url),
            _ => None,
        })
        .collect()
}
