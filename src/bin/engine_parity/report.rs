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

    /// The text report, one line per element.
    fn render(&self, rows: &[PairRow]) -> Vec<String> {
        let mut lines = vec![
            format!("[engine-parity] model: {}", self.model.display()),
            format!(
                "[engine-parity] prompt tokens: {}  MLXCEL_SDPA_DETERMINISTIC={}",
                self.prompt_tokens,
                if self.sdpa_deterministic {
                    "1"
                } else {
                    "unset (results may vary run to run)"
                }
            ),
            "[engine-parity] paths:".to_string(),
        ];
        for (case, side, stream) in &self.streams {
            let len = stream
                .tokens
                .as_ref()
                .map_or_else(|| "n/a".to_string(), |t| format!("{} tokens", t.len()));
            lines.push(format!(
                "  {case:<22} {:<16} {len:<11} {}",
                side.label(),
                stream.detail
            ));
        }
        lines.push("[engine-parity] pairs:".to_string());
        for row in rows {
            lines.push(format!(
                "  {:<22} {:<16} vs {:<16} {}",
                row.case,
                row.left.label(),
                row.right.label(),
                row.result_text()
            ));
        }
        let diverged = rows.iter().filter(|r| r.diverged()).count();
        let na = rows.iter().filter(|r| r.result.is_none()).count();
        lines.push(format!(
            "[engine-parity] summary: {} pair(s), {} identical, {diverged} diverged, {na} n/a",
            rows.len(),
            rows.len() - diverged - na
        ));
        lines
    }

    pub fn print(&self, rows: &[PairRow]) {
        for line in self.render(rows) {
            println!("{line}");
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> Report {
        Report::new(Path::new("models/mlx/tiny"), 42, true)
    }

    #[test]
    fn identical_streams_make_an_identical_row() {
        let mut r = report();
        r.push_stream("greedy", Side::Cli, PathStream::ran(vec![1, 2, 3], "cli"));
        r.push_stream("greedy", Side::Dense, PathStream::ran(vec![1, 2, 3], "d"));
        let rows = r.rows();
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].diverged());
        assert_eq!(rows[0].result_text(), "identical");
    }

    #[test]
    fn divergent_streams_report_first_index_and_ids() {
        let mut r = report();
        r.push_stream("greedy", Side::Cli, PathStream::ran(vec![1, 2, 3], "cli"));
        r.push_stream("greedy", Side::Dense, PathStream::ran(vec![1, 9, 3], "d"));
        let rows = r.rows();
        assert!(rows[0].diverged());
        assert_eq!(rows[0].result_text(), "diverge@1 left=2 right=9");
    }

    #[test]
    fn a_path_that_did_not_run_gives_an_na_row_that_is_not_a_divergence() {
        let mut r = report();
        r.push_stream("greedy", Side::Cli, PathStream::ran(vec![1], "cli"));
        r.push_stream("greedy", Side::Dense, PathStream::ran(vec![1], "d"));
        r.push_stream(
            "greedy",
            Side::Paged,
            PathStream::not_applicable("paged decode unsupported".to_string()),
        );
        let rows = r.rows();
        // cli/dense, cli/paged, dense/paged
        assert_eq!(rows.len(), 3);
        let na: Vec<_> = rows.iter().filter(|row| row.result.is_none()).collect();
        assert_eq!(na.len(), 2);
        assert!(na.iter().all(|row| !row.diverged()));
        assert!(na.iter().all(|row| row.result_text() == "n/a"));
    }

    #[test]
    fn pairs_are_only_formed_between_recorded_sides_and_per_case() {
        let mut r = report();
        r.push_stream("greedy", Side::Cli, PathStream::ran(vec![1], "cli"));
        r.push_stream("seeded", Side::Cli, PathStream::ran(vec![1], "cli"));
        r.push_stream("seeded", Side::CacheMiss, PathStream::ran(vec![1], "m"));
        r.push_stream("seeded", Side::CacheHit, PathStream::ran(vec![2], "h"));
        let rows = r.rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].case, "seeded");
        assert_eq!(
            (rows[0].left, rows[0].right),
            (Side::CacheMiss, Side::CacheHit)
        );
        assert!(rows[0].diverged());
        assert_eq!(
            rows[0].result_text(),
            "diverge@0 (first token) left=1 right=2"
        );
    }

    #[test]
    fn rendered_report_lists_paths_pairs_and_summary() {
        let mut r = report();
        r.push_stream("greedy", Side::Cli, PathStream::ran(vec![1, 2], "cli"));
        r.push_stream(
            "greedy",
            Side::Dense,
            PathStream::ran(vec![1, 3], "Dense pc=off"),
        );
        r.push_stream(
            "greedy",
            Side::Paged,
            PathStream::not_applicable("no paged".to_string()),
        );
        let rows = r.rows();
        let text = r.render(&rows).join("\n");
        assert!(
            text.contains("prompt tokens: 42  MLXCEL_SDPA_DETERMINISTIC=1"),
            "{text}"
        );
        assert!(
            text.contains("a:cli") && text.contains("2 tokens"),
            "{text}"
        );
        assert!(text.contains("c:paged") && text.contains("n/a"), "{text}");
        assert!(text.contains("diverge@1 left=2 right=3"), "{text}");
        assert!(
            text.ends_with("summary: 3 pair(s), 0 identical, 1 diverged, 2 n/a"),
            "{text}"
        );
    }

    #[test]
    fn rendered_report_flags_an_unpinned_sdpa() {
        let r = Report::new(Path::new("m"), 1, false);
        let text = r.render(&[]).join("\n");
        assert!(
            text.contains("unset (results may vary run to run)"),
            "{text}"
        );
        assert!(text.ends_with("summary: 0 pair(s), 0 identical, 0 diverged, 0 n/a"));
    }

    #[test]
    fn json_output_carries_streams_and_pair_results() {
        let mut r = report();
        r.push_stream("greedy", Side::Cli, PathStream::ran(vec![1, 2], "cli"));
        r.push_stream("greedy", Side::Dense, PathStream::ran(vec![1, 3], "d"));
        r.push_stream("greedy", Side::Paged, PathStream::ran(vec![1, 2], "p"));
        let rows = r.rows();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.json");
        r.write_json(&path, &rows).unwrap();
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(doc["prompt_tokens"], 42);
        assert_eq!(doc["sdpa_deterministic"], true);
        assert_eq!(doc["streams"].as_array().unwrap().len(), 3);
        assert_eq!(doc["streams"][0]["path"], "a:cli");
        assert_eq!(doc["streams"][0]["tokens"], serde_json::json!([1, 2]));
        let pairs = doc["pairs"].as_array().unwrap();
        let cli_dense = &pairs[0];
        assert_eq!(cli_dense["left"], "a:cli");
        assert_eq!(cli_dense["right"], "b:dense");
        assert_eq!(cli_dense["first_divergent_index"], 1);
        assert_eq!(cli_dense["left_token"], 2);
        assert_eq!(cli_dense["right_token"], 3);
        let cli_paged = &pairs[1];
        assert_eq!(cli_paged["result"], "identical");
        assert!(cli_paged["first_divergent_index"].is_null());
    }

    #[test]
    fn json_output_marks_a_path_that_did_not_run() {
        let mut r = report();
        r.push_stream("greedy", Side::Cli, PathStream::ran(vec![1], "cli"));
        r.push_stream(
            "greedy",
            Side::Paged,
            PathStream::not_applicable("no paged".to_string()),
        );
        let rows = r.rows();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.json");
        r.write_json(&path, &rows).unwrap();
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(doc["streams"][1]["tokens"].is_null());
        assert_eq!(doc["pairs"][0]["result"], "n/a");
    }
}
