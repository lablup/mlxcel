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

//! Synchronous, teacher-forced model benchmark paired with
//! `scripts/qwen4_exp_benchmark.py`. Run MODEL CORPUS PREFILL DECODE [REPEATS].
//! Run zero is a complete warmup; each run resets all model-owned state.

use anyhow::{Result, ensure};
use mlxcel::LanguageModel;
use mlxcel_core::{argmax, array_to_raw_bytes, astype, dtype, eval, from_slice_i32};
use std::{path::Path, time::Instant};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() >= 5,
        "usage: MODEL CORPUS PREFILL DECODE [REPEATS]"
    );
    let prefill: usize = args[3].parse()?;
    let decode: usize = args[4].parse()?;
    let repeats: usize = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(3);
    ensure!(
        prefill > 0 && decode > 0 && repeats > 0,
        "counts must be positive"
    );
    mlxcel::initialize_runtime_checked().map_err(|err| anyhow::anyhow!("{err}"))?;
    let (model, tokenizer) = mlxcel::load_model(Path::new(&args[1]))?;
    let text = std::fs::read_to_string(&args[2])?;
    let ids: Vec<i32> = tokenizer
        .encode(&text, false)?
        .into_iter()
        .map(|t| t as i32)
        .collect();
    ensure!(ids.len() >= prefill + decode, "corpus is too short");
    for run in 0..=repeats {
        model.reset_runtime_state();
        let mut caches = model.make_caches();
        mlxcel_core::synchronize_default();
        mlxcel_core::reset_peak_memory();
        let prompt = from_slice_i32(&ids[..prefill], &[1, prefill as i32]);
        let start = Instant::now();
        let logits = model.forward(&prompt, &mut caches, None);
        eval(&logits);
        let prefill_seconds = start.elapsed().as_secs_f64();
        drop(logits);
        let mut top_ids = Vec::with_capacity(decode);
        let start = Instant::now();
        for &token in &ids[prefill..prefill + decode] {
            let input = from_slice_i32(&[token], &[1, 1]);
            let logits = model.forward(&input, &mut caches, None);
            let top = astype(&argmax(&logits, -1, false), dtype::INT32);
            eval(&top);
            let bytes = array_to_raw_bytes(&top);
            top_ids.push(i32::from_le_bytes(bytes[..4].try_into()?));
        }
        let decode_seconds = start.elapsed().as_secs_f64();
        println!(
            "{}",
            serde_json::json!({
                "runtime": "mlxcel", "run": run, "warmup": run == 0,
                "prefill_tokens": prefill, "decode_tokens": decode,
                "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
                "prefill_tps": prefill as f64 / prefill_seconds,
                "decode_tps": decode as f64 / decode_seconds,
                "peak_bytes": mlxcel_core::get_peak_memory(), "top_ids": top_ids,
            })
        );
    }
    Ok(())
}
