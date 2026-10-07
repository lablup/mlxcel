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
            format!("  (scheduler: prompt_eval {p} ms, generation {g} ms)")
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
        });
        println!("[engine-bench] {json}");
    }
}

/// Print the two paths next to each other per prompt length.
pub fn print_side_by_side(measurements: &[Measurement]) {
    println!("[engine-bench] summary (server vs cli at the same prompt length):");
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
        println!(
            "  prompt {target:>5}: cli {}  |  server {}{ratio}",
            cell(cli),
            cell(server)
        );
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
