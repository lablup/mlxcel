// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Whole-model Mamba-family execution check (issues #3338 and #3348).
//!
//! Loads a Mamba1, Falcon-Mamba, or Mamba2 checkpoint with the production
//! config reader and weight loader, compiles the state-threaded token prefill
//! for `--device`, and runs
//! `--tokens` twice: once in a single call, and once split at `--split` into
//! two calls that hand the returned state to the second call, then feeds the
//! same tokens one at a time through the decode step (state in module globals).
//! Writes `logits.npy` (single call), `logits_split.npy`, `logits_decode.npy`
//! (after the last decode step) and `run.json` to `--out` for
//! `spike/openxla/ssm_model_check.py`. `--compile-times DIR` instead compiles
//! the prefill and decode graphs once each, uncached, and prints the times.
//!
//! ```text
//! cargo run --release --features xla-iree --example xla_ssm_model_check -- \
//!     --model DIR --device hip --tokens 1,2,3 --out DIR [--capacity 128] \
//!     [--precision f32|f16|bf16] [--layout inout|outin]
//! ```

use std::path::{Path, PathBuf};

use mlxcel_xla::{SsmDecodeSession, SsmModelSession, compile_ssm_model_graphs, npy_f32_bytes};

struct Args {
    model: PathBuf,
    device: String,
    capacity: usize,
    precision: String,
    layout: String,
    tokens: Vec<i32>,
    split: Option<usize>,
    out: Option<PathBuf>,
    compile_times: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        model: PathBuf::new(),
        device: "local-task".to_string(),
        capacity: 128,
        precision: "f32".to_string(),
        layout: "inout".to_string(),
        tokens: Vec::new(),
        split: None,
        out: None,
        compile_times: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => args.model = PathBuf::from(value()?),
            "--device" => args.device = value()?,
            "--precision" => args.precision = value()?,
            "--layout" => args.layout = value()?,
            "--capacity" => {
                let text = value()?;
                args.capacity = text.parse().map_err(|_| format!("bad --capacity {text}"))?;
            }
            "--split" => {
                let text = value()?;
                args.split = Some(text.parse().map_err(|_| format!("bad --split {text}"))?);
            }
            "--tokens" => {
                args.tokens = value()?
                    .split(',')
                    .map(|t| t.trim().parse().map_err(|_| format!("bad token {t}")))
                    .collect::<Result<_, _>>()?;
            }
            "--out" => args.out = Some(PathBuf::from(value()?)),
            "--compile-times" => args.compile_times = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.model.as_os_str().is_empty() {
        return Err("--model DIR is required".to_string());
    }
    if args.out.is_none() && args.compile_times.is_none() {
        return Err("--out DIR (or --compile-times DIR) is required".to_string());
    }
    Ok(args)
}

fn write_npy(path: &Path, values: &[f32]) -> Result<(), String> {
    std::fs::write(path, npy_f32_bytes(values, &[values.len()])?)
        .map_err(|e| format!("write {}: {e}", path.display()))
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    if let Some(dir) = &args.compile_times {
        let times = compile_ssm_model_graphs(
            &args.device,
            &args.model,
            args.capacity,
            &args.precision,
            &args.layout,
            dir,
        )?;
        println!(
            "{}",
            serde_json::json!({
                "device": args.device,
                "capacity": args.capacity,
                "precision": args.precision,
                "layout": args.layout,
                "prefill_compile_s": times.prefill_seconds,
                "decode_compile_s": times.decode_seconds,
            })
        );
        return Ok(());
    }
    let out = args.out.clone().expect("checked in parse_args");
    std::fs::create_dir_all(&out).map_err(|e| format!("create {}: {e}", out.display()))?;
    let mut session = SsmModelSession::load(
        &args.device,
        &args.model,
        args.capacity,
        &args.precision,
        &args.layout,
    )?;
    let zero = session.zero_state();
    let whole = session.prefill(&args.tokens, &zero)?;
    write_npy(&out.join("logits.npy"), &whole.logits)?;

    let split = args.split.unwrap_or(args.tokens.len() / 2);
    if split == 0 || split >= args.tokens.len() {
        return Err(format!(
            "--split {split} must leave tokens on both sides of {}",
            args.tokens.len()
        ));
    }
    let first = session.prefill(&args.tokens[..split], &zero)?;
    let second = session.prefill(&args.tokens[split..], &first.states)?;
    write_npy(&out.join("logits_split.npy"), &second.logits)?;
    let mut decode = SsmDecodeSession::load(
        &args.device,
        &args.model,
        args.capacity,
        &args.precision,
        &args.layout,
    )?;
    let mut decoded = Vec::new();
    for &token in &args.tokens {
        decoded = decode.step(token)?;
    }
    write_npy(&out.join("logits_decode.npy"), &decoded)?;
    let state_max_abs = second
        .states
        .iter()
        .zip(&whole.states)
        .flat_map(|(a, b)| a.values.iter().zip(&b.values).map(|(x, y)| (x - y).abs()))
        .fold(0.0f32, f32::max);
    let run = serde_json::json!({
        "device": args.device,
        "capacity": args.capacity,
        "precision": args.precision,
        "layout": args.layout,
        "tokens": args.tokens.len(),
        "split": split,
        "compile_s": session.compile_seconds,
        "prefill_run_s": whole.run_seconds,
        "split_state_max_abs": state_max_abs,
    });
    std::fs::write(out.join("run.json"), run.to_string())
        .map_err(|e| format!("write run.json: {e}"))?;
    println!("{run}");
    Ok(())
}
