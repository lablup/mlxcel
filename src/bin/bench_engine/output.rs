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

//! Measurement records and output formats of `mlxcel-bench-engine`.

use std::io::Write as _;
use std::path::Path;

use anyhow::{Context, Result};
use mlxcel::GenerationStats;
use mlxcel::server::DecodeStorageBackend;
use mlxcel::server::engine_probe::ServerEngineRun;

/// CSV header. The first eight columns follow `scripts/bench_decode.sh`'s
/// (`prefill_ms` there is the TTFT here, the time to the first token).
const CSV_HEADER: &str = "model,model_path,prompt_tokens,generated_tokens,ttft_ms,prefill_tok_s,\
decode_ms,decode_tok_s,path,prompt_target_len,prefill_chunk,decode_storage,max_tokens,label";

/// One measured pass of one decode path.
#[derive(Debug, Clone)]
pub struct Measurement {
    path: &'static str,
    model: String,
    model_path: String,
    prompt_target: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    ttft_ms: f64,
    decode_ms: f64,
    decode_tok_s: f64,
    prefill_chunk: usize,
    decode_storage: &'static str,
    /// The scheduler's own millisecond-resolution figures, server path only.
    server_ms: Option<(u64, u64)>,
    /// Pooled paged-attention launches during the measured request, server
    /// path only. Zero on a `paged` row means the model decoded a lone
    /// sequence through dense caches (model-owned KV families).
    paged_decode_launches: Option<u64>,
}

fn model_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

fn storage_name(storage: DecodeStorageBackend) -> &'static str {
    match storage {
        DecodeStorageBackend::Auto => "auto",
        DecodeStorageBackend::Dense => "dense",
        DecodeStorageBackend::Paged => "paged",
    }
}

impl Measurement {
    pub fn from_cli(
        model: &Path,
        target: usize,
        stats: &GenerationStats,
        generated: usize,
        prefill_chunk: usize,
    ) -> Self {
        Self {
            path: "cli",
            model: model_name(model),
            model_path: model.display().to_string(),
            prompt_target: target,
            prompt_tokens: stats.prompt_tokens,
            generated_tokens: generated,
            ttft_ms: stats.prefill_time_ms,
            decode_ms: stats.decode_time_ms,
            decode_tok_s: stats.decode_tok_per_sec,
            prefill_chunk,
            decode_storage: "dense",
            server_ms: None,
            paged_decode_launches: None,
        }
    }

    pub fn from_server(
        model: &Path,
        target: usize,
        prompt_tokens: usize,
        run: &ServerEngineRun,
        prefill_chunk: usize,
        storage: DecodeStorageBackend,
    ) -> Self {
        Self {
            path: "server",
            model: model_name(model),
            model_path: model.display().to_string(),
            prompt_target: target,
            prompt_tokens,
            generated_tokens: run.tokens.len(),
            ttft_ms: run.ttft_ms,
            decode_ms: run.decode_ms,
            decode_tok_s: run.decode_tok_per_sec(),
            prefill_chunk,
            decode_storage: storage_name(storage),
            server_ms: Some((run.server_prompt_eval_ms, run.server_generation_ms)),
            paged_decode_launches: Some(run.paged_decode_launches),
        }
    }

    fn prefill_tok_s(&self) -> f64 {
        if self.ttft_ms > 0.0 {
            self.prompt_tokens as f64 / (self.ttft_ms / 1000.0)
        } else {
            0.0
        }
    }

    /// Print the human-readable line and the machine-readable JSON line.
    pub fn emit(&self, label: &str, max_tokens: usize) {
        let server = self.server_ms.map_or_else(String::new, |(p, g)| {
            format!(
                "  (scheduler: prompt_eval {p} ms, generation {g} ms; paged kernel launches {})",
                self.paged_decode_launches.unwrap_or(0)
            )
        });
        println!(
            "[{:<6}] prompt {:>5} tok  TTFT {:>9.2} ms  decode {:>5} tok in {:>9.2} ms = {:>8.2} tok/s  \
             chunk {} storage {}{server}",
            self.path,
            self.prompt_tokens,
            self.ttft_ms,
            self.generated_tokens,
            self.decode_ms,
            self.decode_tok_s,
            self.prefill_chunk,
            self.decode_storage,
        );
        if self.generated_tokens < max_tokens {
            eprintln!(
                "[engine-bench] warning: {} stopped at {} of {max_tokens} tokens (EOS); \
                 pass --ignore-eos for a fixed budget",
                self.path, self.generated_tokens
            );
        }
        if self.decode_storage == "paged" && self.paged_decode_launches == Some(0) {
            eprintln!(
                "[engine-bench] warning: storage resolved to paged but no paged attention kernel \
                 ran; this model keeps model-owned KV and decoded a lone sequence through dense \
                 caches, so this row does not measure paged decode"
            );
        }
        let json = serde_json::json!({
            "path": self.path,
            "model": self.model,
            "prompt_target_len": self.prompt_target,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "ttft_ms": self.ttft_ms,
            "prefill_tok_s": self.prefill_tok_s(),
            "decode_ms": self.decode_ms,
            "decode_tok_s": self.decode_tok_s,
            "prefill_chunk": self.prefill_chunk,
            "decode_storage": self.decode_storage,
            "max_tokens": max_tokens,
            "label": label,
            "server_prompt_eval_ms": self.server_ms.map(|(p, _)| p),
            "server_generation_ms": self.server_ms.map(|(_, g)| g),
            "paged_decode_launches": self.paged_decode_launches,
        });
        println!("[engine-bench] {json}");
    }
}

/// The two paths next to each other per prompt length, one line per element.
fn side_by_side_lines(measurements: &[Measurement]) -> Vec<String> {
    let mut lines =
        vec!["[engine-bench] summary (server vs cli at the same prompt length):".to_string()];
    let mut targets: Vec<usize> = measurements.iter().map(|m| m.prompt_target).collect();
    targets.sort_unstable();
    targets.dedup();
    for target in targets {
        let find = |path| {
            measurements
                .iter()
                .find(|m| m.prompt_target == target && m.path == path)
        };
        let cell = |m: Option<&Measurement>| {
            m.map_or_else(
                || "-".to_string(),
                |m| format!("{:.2} ms TTFT, {:.2} tok/s", m.ttft_ms, m.decode_tok_s),
            )
        };
        let (cli, server) = (find("cli"), find("server"));
        let ratio = match (cli, server) {
            (Some(c), Some(s)) if c.decode_tok_s > 0.0 => {
                format!("  server/cli decode {:.3}", s.decode_tok_s / c.decode_tok_s)
            }
            _ => String::new(),
        };
        lines.push(format!(
            "  prompt {target:>5}: cli {}  |  server {}{ratio}",
            cell(cli),
            cell(server)
        ));
    }
    lines
}

/// Print the two paths next to each other per prompt length.
pub fn print_side_by_side(measurements: &[Measurement]) {
    for line in side_by_side_lines(measurements) {
        println!("{line}");
    }
}

/// Append one CSV row per measurement, writing the header to a new file.
pub fn append_csv(
    path: &Path,
    measurements: &[Measurement],
    label: &str,
    max_tokens: usize,
) -> Result<()> {
    let fresh = std::fs::metadata(path).map_or(true, |m| m.len() == 0);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    if fresh {
        writeln!(file, "{CSV_HEADER}")?;
    }
    // Labels and paths land unquoted; keep commas out of them.
    let clean = |s: &str| s.replace(',', ";");
    for m in measurements {
        writeln!(
            file,
            "{},{},{},{},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{},{}",
            clean(&m.model),
            clean(&m.model_path),
            m.prompt_tokens,
            m.generated_tokens,
            m.ttft_ms,
            m.prefill_tok_s(),
            m.decode_ms,
            m.decode_tok_s,
            m.path,
            m.prompt_target,
            m.prefill_chunk,
            m.decode_storage,
            max_tokens,
            clean(label),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli(target: usize, ttft_ms: f64, decode_tok_s: f64) -> Measurement {
        let stats = GenerationStats {
            prompt_tokens: target,
            generated_tokens: 8,
            prefill_time_ms: ttft_ms,
            decode_time_ms: 100.0,
            prefill_tok_per_sec: 0.0,
            decode_tok_per_sec: decode_tok_s,
        };
        Measurement::from_cli(Path::new("models/mlx/tiny"), target, &stats, 8, 2048)
    }

    fn server(target: usize, ttft_ms: f64, tokens: usize, decode_ms: f64) -> Measurement {
        let run = ServerEngineRun {
            tokens: vec![0; tokens],
            prompt_tokens: target,
            cached_tokens: 0,
            forwarded_prefill_tokens: target as u64,
            ttft_ms,
            decode_ms,
            server_prompt_eval_ms: 3,
            server_generation_ms: 90,
            finish_reason: "length".to_string(),
            prompt_cache_inserts: 0,
            prompt_cache_reject: None,
            paged_decode_launches: 0,
        };
        Measurement::from_server(
            Path::new("models/mlx/tiny"),
            target,
            target,
            &run,
            512,
            DecodeStorageBackend::Paged,
        )
    }

    #[test]
    fn prefill_rate_is_prompt_tokens_over_ttft() {
        assert!((cli(1000, 500.0, 1.0).prefill_tok_s() - 2000.0).abs() < 1e-9);
        assert_eq!(cli(1000, 0.0, 1.0).prefill_tok_s(), 0.0);
    }

    #[test]
    fn server_measurement_derives_decode_rate_from_its_own_timing() {
        let m = server(256, 12.0, 100, 1000.0);
        assert!((m.decode_tok_s - 100.0).abs() < 1e-9);
        assert_eq!(m.decode_storage, "paged");
        assert_eq!(m.prefill_chunk, 512);
        assert_eq!(m.server_ms, Some((3, 90)));
        assert_eq!(m.paged_decode_launches, Some(0));
        // No decode time recorded: the rate is 0, not infinity.
        assert_eq!(server(256, 12.0, 100, 0.0).decode_tok_s, 0.0);
    }

    #[test]
    fn cli_measurement_is_dense_without_server_figures() {
        let m = cli(256, 10.0, 50.0);
        assert_eq!(m.path, "cli");
        assert_eq!(m.model, "tiny");
        assert_eq!(m.decode_storage, "dense");
        assert_eq!(m.prefill_chunk, 2048);
        assert!(m.server_ms.is_none() && m.paged_decode_launches.is_none());
    }

    #[test]
    fn side_by_side_pairs_paths_per_prompt_length_with_a_ratio() {
        let lines = side_by_side_lines(&[
            cli(8192, 900.0, 100.0),
            cli(256, 20.0, 120.0),
            server(256, 25.0, 100, 1000.0),
        ]);
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("server vs cli"));
        // Sorted by prompt length: 256 has both paths, 8192 only the CLI.
        assert!(lines[1].contains("prompt   256"), "{}", lines[1]);
        assert!(
            lines[1].contains("20.00 ms TTFT, 120.00 tok/s"),
            "{}",
            lines[1]
        );
        assert!(
            lines[1].contains("25.00 ms TTFT, 100.00 tok/s"),
            "{}",
            lines[1]
        );
        assert!(
            lines[1].ends_with("server/cli decode 0.833"),
            "{}",
            lines[1]
        );
        assert!(lines[2].contains("prompt  8192"), "{}", lines[2]);
        assert!(lines[2].contains("server -"), "{}", lines[2]);
        assert!(!lines[2].contains("server/cli"), "{}", lines[2]);
    }

    #[test]
    fn side_by_side_skips_the_ratio_when_the_cli_rate_is_zero() {
        let lines = side_by_side_lines(&[cli(256, 20.0, 0.0), server(256, 25.0, 100, 1000.0)]);
        assert!(!lines[1].contains("server/cli"), "{}", lines[1]);
    }

    #[test]
    fn csv_gets_a_header_once_and_one_row_per_measurement() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bench.csv");
        let ms = [cli(256, 20.0, 120.0), server(256, 25.0, 100, 1000.0)];
        append_csv(&path, &ms, "arm,one", 128).unwrap();
        append_csv(&path, &ms[..1], "arm-two", 128).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "{text}");
        assert_eq!(lines[0], CSV_HEADER);
        assert_eq!(text.matches("model,model_path").count(), 1);
        let columns = CSV_HEADER.split(',').count();
        assert_eq!(columns, 14);
        for row in &lines[1..] {
            assert_eq!(row.split(',').count(), columns, "{row}");
        }
        // A comma in the label cannot split a column.
        assert!(lines[1].ends_with(",arm;one"), "{}", lines[1]);
        assert!(
            lines[1].starts_with("tiny,models/mlx/tiny,256,8,20.000,"),
            "{}",
            lines[1]
        );
        assert!(
            lines[2].contains(",server,256,512,paged,128,"),
            "{}",
            lines[2]
        );
    }
}
