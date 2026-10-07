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

//! Rows and output of `mlxcel-engine-parity`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mlxcel::server::engine_probe::{Divergence, first_divergence};

/// One decode path the harness runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// (a) `CxxGenerator` through the `mlxcel generate` text path.
    Cli,
    /// (b) server scheduler, B=1, dense decode storage, prompt cache off.
    Dense,
    /// (c) server scheduler, B=1, paged decode storage, prompt cache off.
    Paged,
    /// (b) with the prompt cache on, first (cold) request.
    CacheMiss,
    /// (b) with the prompt cache on, the same request after a priming request
    /// stored its history prefix.
    CacheHit,
}

impl Side {
    fn label(self) -> &'static str {
        match self {
            Self::Cli => "a:cli",
            Self::Dense => "b:dense",
            Self::Paged => "c:paged",
            Self::CacheMiss => "b:dense+pc:miss",
            Self::CacheHit => "b:dense+pc:hit",
        }
    }
}

/// The pairs compared per case, in print order.
const PAIRS: [(Side, Side); 5] = [
    (Side::Cli, Side::Dense),
    (Side::Cli, Side::Paged),
    (Side::Dense, Side::Paged),
    (Side::Dense, Side::CacheMiss),
    (Side::CacheMiss, Side::CacheHit),
];

/// The token stream one path produced, or why it did not run.
#[derive(Debug, Clone)]
pub struct PathStream {
    tokens: Option<Vec<i32>>,
    detail: String,
}

impl PathStream {
    pub fn ran(tokens: Vec<i32>, detail: &str) -> Self {
        Self {
            tokens: Some(tokens),
            detail: detail.to_string(),
        }
    }

    pub fn not_applicable(reason: String) -> Self {
        Self {
            tokens: None,
            detail: reason,
        }
    }
}

/// One compared pair.
#[derive(Debug, Clone)]
pub struct PairRow {
    case: &'static str,
    left: Side,
    right: Side,
    /// `None` when either side did not run (`n/a`).
    result: Option<Divergence>,
}

impl PairRow {
    /// Whether this pair ran on both sides and differed.
    pub fn diverged(&self) -> bool {
        self.result.is_some_and(|d| !d.is_identical())
    }

    fn result_text(&self) -> String {
        self.result
            .map_or_else(|| "n/a".to_string(), |d| d.to_string())
    }
}

/// Everything one harness run recorded.
pub struct Report {
    model: PathBuf,
    prompt_tokens: usize,
    sdpa_deterministic: bool,
    streams: Vec<(&'static str, Side, PathStream)>,
}

impl Report {
    pub fn new(model: &Path, prompt_tokens: usize, sdpa_deterministic: bool) -> Self {
        Self {
            model: model.to_path_buf(),
            prompt_tokens,
            sdpa_deterministic,
            streams: Vec::new(),
        }
    }

    pub fn push_stream(&mut self, case: &'static str, side: Side, stream: PathStream) {
        self.streams.push((case, side, stream));
    }

    fn stream(&self, case: &str, side: Side) -> Option<&PathStream> {
        self.streams
            .iter()
            .find(|(c, s, _)| *c == case && *s == side)
            .map(|(_, _, stream)| stream)
    }

    fn cases(&self) -> Vec<&'static str> {
        let mut cases: Vec<&'static str> = Vec::new();
        for (case, _, _) in &self.streams {
            if !cases.contains(case) {
                cases.push(case);
            }
        }
        cases
    }

    /// Compare every pair whose two sides were recorded.
    pub fn rows(&self) -> Vec<PairRow> {
        let mut rows = Vec::new();
        for case in self.cases() {
            for (left, right) in PAIRS {
                let (Some(l), Some(r)) = (self.stream(case, left), self.stream(case, right)) else {
                    continue;
                };
                let result = match (l.tokens.as_deref(), r.tokens.as_deref()) {
                    (Some(a), Some(b)) => Some(first_divergence(a, b)),
                    _ => None,
                };
                rows.push(PairRow {
                    case,
                    left,
                    right,
                    result,
                });
            }
        }
        rows
    }

    pub fn print(&self, rows: &[PairRow]) {
        println!("[engine-parity] model: {}", self.model.display());
        println!(
            "[engine-parity] prompt tokens: {}  MLXCEL_SDPA_DETERMINISTIC={}",
            self.prompt_tokens,
            if self.sdpa_deterministic {
                "1"
            } else {
                "unset (results may vary run to run)"
            }
        );
        println!("[engine-parity] paths:");
        for (case, side, stream) in &self.streams {
            let len = stream
                .tokens
                .as_ref()
                .map_or_else(|| "n/a".to_string(), |t| format!("{} tokens", t.len()));
            println!(
                "  {case:<22} {:<16} {len:<11} {}",
                side.label(),
                stream.detail
            );
        }
        println!("[engine-parity] pairs:");
        for row in rows {
            println!(
                "  {:<22} {:<16} vs {:<16} {}",
                row.case,
                row.left.label(),
                row.right.label(),
                row.result_text()
            );
        }
        let diverged = rows.iter().filter(|r| r.diverged()).count();
        let na = rows.iter().filter(|r| r.result.is_none()).count();
        println!(
            "[engine-parity] summary: {} pair(s), {} identical, {diverged} diverged, {na} n/a",
            rows.len(),
            rows.len() - diverged - na
        );
    }

    pub fn write_json(&self, path: &Path, rows: &[PairRow]) -> Result<()> {
        let streams: Vec<serde_json::Value> = self
            .streams
            .iter()
            .map(|(case, side, stream)| {
                serde_json::json!({
                    "case": case,
                    "path": side.label(),
                    "detail": stream.detail,
                    "tokens": stream.tokens,
                })
            })
            .collect();
        let pairs: Vec<serde_json::Value> = rows
            .iter()
            .map(|row| {
                let (index, left, right) = match row.result {
                    Some(Divergence::At { index, left, right }) => (Some(index), left, right),
                    _ => (None, None, None),
                };
                serde_json::json!({
                    "case": row.case,
                    "left": row.left.label(),
                    "right": row.right.label(),
                    "result": row.result_text(),
                    "first_divergent_index": index,
                    "left_token": left,
                    "right_token": right,
                })
            })
            .collect();
        let doc = serde_json::json!({
            "model": self.model.display().to_string(),
            "prompt_tokens": self.prompt_tokens,
            "sdpa_deterministic": self.sdpa_deterministic,
            "streams": streams,
            "pairs": pairs,
        });
        let text = serde_json::to_string_pretty(&doc)?;
        std::fs::write(path, text).with_context(|| format!("failed to write {}", path.display()))
    }
}
