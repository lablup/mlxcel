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

//! Reference-equivalence + throughput harness for the OpenXLA/IREE continuous
//! batching engine (issue #449 M3 Stage 2b). Proves [`mlxcel_xla::XlaBatchEngine`]
//! without the server.
//!
//! `B_max` slots serve `N > B_max` requests of DIFFERENT prompt lengths AND token
//! budgets, so requests finish at staggered times and queued requests are admitted
//! mid-stream into freed slots (slot recycling). The gate (the Stage 2a gate, now
//! over the productized engine + the device-side slot write): every request's
//! output stream must equal its INDEPENDENT single-sequence reference, regardless
//! of when it was admitted or which peers shared its batch.
//!
//! Build needs the `xla-iree` feature (real IREE execution); see
//! `src/lib/mlxcel-xla/README.md`.
//!
//! ```bash
//! # CPU (prebuilt dist):
//! IREE_DIST=/path/to/iree-dist cargo run --release --features xla-iree \
//!   --example xla_batch_bench -- --batch 4 --requests 8 --maxcap 24
//! # CUDA (GB10): source-built runtime + cuda iree-compile, then run with cuda.
//! IREE_CUDA_HOME=... MLXCEL_XLA_IREE_COMPILE=... cargo run --release \
//!   --features xla-iree --example xla_batch_bench -- \
//!   --device cuda --batch 8 --requests 16 --maxcap 48
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use mlxcel_xla::{EngineEvent, FinishReason, SampleParams, XlaBatchEngine, XlaReferenceEngine};

fn arg(flag: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

/// Read an `[int, ...]` array under `key` from a JSON file.
fn read_int_array(path: &Path, key: &str) -> Vec<i32> {
    let s =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let v: serde_json::Value =
        serde_json::from_str(&s).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} is not an array in {}", path.display()))
        .iter()
        .map(|x| x.as_i64().expect("int element") as i32)
        .collect()
}

/// EOS ids from `generation_config.json` (a single int or a list), matching the
/// engine's own reader so the reference stop test agrees with the engine's.
fn read_eos(model_dir: &Path) -> Vec<i32> {
    let p = model_dir.join("generation_config.json");
    let Ok(s) = std::fs::read_to_string(p) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) else {
        return Vec::new();
    };
    match v.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => {
            n.as_i64().map(|x| vec![x as i32]).unwrap_or_default()
        }
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(serde_json::Value::as_i64)
            .map(|x| x as i32)
            .collect(),
        _ => Vec::new(),
    }
}

fn main() {
    let here = env!("CARGO_MANIFEST_DIR");
    let model = PathBuf::from(arg(
        "--model",
        &PathBuf::from(here)
            .join("spike/openxla/models/Llama-3.2-1B-Instruct")
            .to_string_lossy(),
    ));
    let prompts_json = PathBuf::from(arg(
        "--prompts",
        &PathBuf::from(here)
            .join("spike/openxla/artifacts/prompt_ids.json")
            .to_string_lossy(),
    ));
    // --device wins, else MLXCEL_XLA_DEVICE, else local-task (matches the engine).
    let device = {
        let a = arg("--device", "");
        if !a.is_empty() {
            a
        } else {
            std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".to_string())
        }
    };
    let b_max: usize = arg("--batch", "4").parse().expect("--batch int");
    let nreq: usize = arg("--requests", &(2 * b_max).to_string())
        .parse()
        .expect("--requests int");
    // Clamp per-request budgets so the (sequential) reference pass stays short on
    // CPU; the staggered-eviction behaviour holds for any spread of budgets.
    let maxcap: usize = arg("--maxcap", "24").parse().expect("--maxcap int");
    assert!(b_max >= 1 && nreq >= 1 && maxcap >= 2);

    let prompt_ids = read_int_array(&prompts_json, "prompt_ids");
    let full = prompt_ids.len();
    let eos = read_eos(&model);

    // N requests with varied lengths AND budgets (same spread as the spike
    // scheduler), so they evict at staggered steps and later ones admit mid-stream.
    let specs: Vec<(usize, usize)> = (0..nreq)
        .map(|i| {
            let len = (20 + (i * 13) % full.saturating_sub(20).max(1)).min(full);
            let cap = (12 + (i * 11) % 28).min(maxcap);
            (len, cap)
        })
        .collect();

    println!(
        "model = {}\ndevice = {device}, slots B_max = {b_max}, requests N = {nreq}, \
         full prompt = {full}, eos = {eos:?}",
        model.display()
    );
    for (i, (len, cap)) in specs.iter().enumerate() {
        print!(
            "  req {i}: len {len} cap {cap}{}",
            if i % 4 == 3 { "\n" } else { "" }
        );
    }
    println!();

    // === Phase 1: capture an independent single-seq reference per request ===
    // The reference engine loads the weights once and reuses them; the KV is
    // overwritten per request. It is dropped before the batched engine loads, so
    // peak memory stays at one resident weight set.
    let t_ref = Instant::now();
    let references: Vec<Vec<i32>> = {
        let mut refeng = XlaReferenceEngine::load(&model, &device).expect("load reference engine");
        specs
            .iter()
            .map(|&(len, cap)| {
                refeng
                    .generate(&prompt_ids[..len], cap, &eos)
                    .expect("reference generate")
            })
            .collect()
    };
    let ref_tokens: usize = references.iter().map(Vec::len).sum();
    let ref_secs = t_ref.elapsed().as_secs_f64();
    println!(
        "phase 1: captured {nreq} single-seq references ({ref_tokens} tokens, \
         {:.2} tok/s sequential baseline)",
        ref_tokens as f64 / ref_secs
    );

    // === Phase 2: drive all requests through the continuous-batching engine ===
    let mut engine = XlaBatchEngine::load(&model, b_max, &device).expect("load batch engine");
    let mut streams: Vec<Vec<i32>> = vec![Vec::new(); nreq];
    let mut finished: Vec<Option<FinishReason>> = vec![None; nreq];
    for &(len, cap) in &specs {
        // Greedy keeps the reference-equivalence gate exact: host argmax of the
        // logits graph == the single-seq argmax reference, token for token.
        engine
            .submit(&prompt_ids[..len], cap, SampleParams::greedy())
            .expect("submit request");
    }

    let mut steps = 0usize;
    let safety = nreq * (maxcap + 2) + 64;
    let t_run = Instant::now();
    while !engine.is_idle() {
        steps += 1;
        assert!(steps < safety, "engine exceeded the safety step cap");
        for ev in engine.pump().expect("pump") {
            match ev {
                EngineEvent::Token { req_id, token } => streams[req_id as usize].push(token),
                EngineEvent::Finished { req_id, reason } => {
                    finished[req_id as usize] = Some(reason)
                }
            }
        }
    }
    let run_secs = t_run.elapsed().as_secs_f64();
    let gen_tokens: usize = streams.iter().map(Vec::len).sum();

    // === compare each request's collected stream to its reference ===
    let mut all_ok = true;
    for (i, (stream, reference)) in streams.iter().zip(&references).enumerate() {
        let ok = stream == reference;
        all_ok &= ok;
        if !ok {
            let m = stream.len().min(reference.len());
            let div = (0..m).find(|&j| stream[j] != reference[j]);
            println!(
                "  req {i} (len {} cap {}): MISMATCH (got {} ref {}, first div {div:?})",
                specs[i].0,
                specs[i].1,
                stream.len(),
                reference.len(),
            );
        }
    }
    let unfinished = finished.iter().filter(|f| f.is_none()).count();
    assert_eq!(
        unfinished, 0,
        "{unfinished} requests never emitted Finished"
    );

    println!("\nphase 2: {steps} pump steps, {nreq} requests served, {gen_tokens} tokens",);
    println!(
        "throughput: {:.2} tok/s batched (vs {:.2} tok/s sequential baseline, {:.1}x)",
        gen_tokens as f64 / run_secs,
        ref_tokens as f64 / ref_secs,
        (gen_tokens as f64 / run_secs) / (ref_tokens as f64 / ref_secs),
    );
    println!(
        "RESULT: {}",
        if all_ok {
            "REFERENCE-EXACT PASS (every request matches its single-seq reference)"
        } else {
            "MISMATCH"
        }
    );
    println!(
        "BENCH\tdevice={device}\tB={b_max}\tN={nreq}\ttok_s={:.3}\tpass={all_ok}",
        gen_tokens as f64 / run_secs
    );
    std::process::exit(if all_ok { 0 } else { 1 });
}
