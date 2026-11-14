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
//! # Modes (issue #2923)
//!
//! - Default: greedy requests through one `XlaBatchEngine`. Unchanged.
//! - `--buckets <c1,c2,...>`: build an `XlaBucketSet` with those capacities and
//!   add one prompt sized to route into EACH bucket, so bucket routing and every
//!   compiled shape are reference-checked, not just the largest.
//! - `--seed <n>` (with `--temperature` and `--top-p`): run the same seeded
//!   non-greedy workload twice and require identical streams; a `top_k = 1`
//!   request must still equal the greedy reference.
//! - `--cancel <count>`: cancel that many admitted requests mid-stream, require
//!   no later events for them, and require every non-cancelled stream to stay
//!   reference-exact after their slots are reused by later requests.
//!
//! Each mode prints its own pass line and any mismatch exits non-zero.
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
//! # All extended modes together (ROCm):
//! MLXCEL_XLA_DEVICE=hip cargo run --release --features xla-iree \
//!   --example xla_batch_bench -- --batch 8 --requests 16 --maxcap 48 \
//!   --buckets 256,2048 --seed 7 --cancel 2
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use mlxcel_xla::{
    EngineEvent, FinishReason, SampleParams, XlaAdmissionError, XlaBatchEngine, XlaBucketSet,
    XlaReferenceEngine,
};

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

/// One request: a prompt of cycling source tokens plus its generation budget.
type Spec = (Vec<i32>, usize);

/// The subset of both engine types this harness drives, so the greedy, seeded
/// and cancel passes share one loop.
trait GateEngine {
    fn submit(
        &mut self,
        prompt: &[i32],
        max_new_tokens: usize,
        params: SampleParams,
    ) -> Result<u64, XlaAdmissionError>;
    fn pump(&mut self) -> Result<Vec<EngineEvent>, String>;
    fn is_idle(&self) -> bool;
    fn cancel(&mut self, req_id: u64) -> bool;
}

impl GateEngine for XlaBatchEngine {
    fn submit(
        &mut self,
        prompt: &[i32],
        max_new_tokens: usize,
        params: SampleParams,
    ) -> Result<u64, XlaAdmissionError> {
        XlaBatchEngine::submit(self, prompt, max_new_tokens, params)
    }

    fn pump(&mut self) -> Result<Vec<EngineEvent>, String> {
        XlaBatchEngine::pump(self)
    }

    fn is_idle(&self) -> bool {
        XlaBatchEngine::is_idle(self)
    }

    fn cancel(&mut self, req_id: u64) -> bool {
        XlaBatchEngine::cancel(self, req_id)
    }
}

impl GateEngine for XlaBucketSet {
    fn submit(
        &mut self,
        prompt: &[i32],
        max_new_tokens: usize,
        params: SampleParams,
    ) -> Result<u64, XlaAdmissionError> {
        XlaBucketSet::submit(self, prompt, max_new_tokens, params)
    }

    fn pump(&mut self) -> Result<Vec<EngineEvent>, String> {
        XlaBucketSet::pump(self)
    }

    fn is_idle(&self) -> bool {
        XlaBucketSet::is_idle(self)
    }

    fn cancel(&mut self, req_id: u64) -> bool {
        XlaBucketSet::cancel(self, req_id)
    }
}

struct Outcome {
    streams: Vec<Vec<i32>>,
    finished: Vec<Option<FinishReason>>,
    cancelled: Vec<bool>,
    steps: usize,
}

/// Submit every request, pump to idle, and optionally cancel admitted requests
/// after their first token. A cancelled request must never emit again.
fn drive(
    engine: &mut dyn GateEngine,
    specs: &[Spec],
    maxcap: usize,
    params_of: &dyn Fn(usize) -> SampleParams,
    cancel_count: usize,
) -> Outcome {
    let n = specs.len();
    let mut streams: Vec<Vec<i32>> = vec![Vec::new(); n];
    let mut finished: Vec<Option<FinishReason>> = vec![None; n];
    let mut cancelled = vec![false; n];
    let mut seen = vec![0usize; n];
    let mut to_cancel = cancel_count;
    // A bucket set namespaces its request ids per bucket, so an id is not the
    // submission index. Map it back explicitly instead of assuming they match.
    let mut req_ids = Vec::with_capacity(n);
    let mut row_of = std::collections::HashMap::with_capacity(n);
    for (index, (prompt, cap)) in specs.iter().enumerate() {
        let req_id = engine
            .submit(prompt, *cap, params_of(index))
            .unwrap_or_else(|error| panic!("submit request {index}: {error}"));
        assert!(
            row_of.insert(req_id, index).is_none(),
            "duplicate request id {req_id}"
        );
        req_ids.push(req_id);
    }
    let mut steps = 0usize;
    let safety = n * (maxcap + 2) + 64;
    while !engine.is_idle() {
        steps += 1;
        assert!(steps < safety, "engine exceeded the safety step cap");
        let events = engine.pump().expect("pump");
        for event in events {
            match event {
                EngineEvent::Token { req_id, token } => {
                    let row = *row_of
                        .get(&req_id)
                        .unwrap_or_else(|| panic!("event for unknown request id {req_id}"));
                    assert!(
                        !cancelled[row],
                        "cancelled request {req_id} emitted a later token"
                    );
                    streams[row].push(token);
                    seen[row] += 1;
                }
                EngineEvent::Finished { req_id, reason } => {
                    let row = *row_of
                        .get(&req_id)
                        .unwrap_or_else(|| panic!("event for unknown request id {req_id}"));
                    assert!(
                        !cancelled[row],
                        "cancelled request {req_id} emitted Finished"
                    );
                    finished[row] = Some(reason);
                }
            }
        }
        // Cancel after the whole step, on requests that emitted their first
        // token in it. The step already ran to completion, so this is the first
        // point a cancel can take effect; any later event for these rows is a
        // real violation and the asserts above catch it.
        while to_cancel > 0 {
            let candidate =
                (0..n).find(|&row| seen[row] > 0 && !cancelled[row] && finished[row].is_none());
            let Some(row) = candidate else {
                break;
            };
            assert!(engine.cancel(req_ids[row]), "cancel admitted request {row}");
            cancelled[row] = true;
            to_cancel -= 1;
        }
    }
    assert_eq!(
        to_cancel, 0,
        "could not cancel {to_cancel} admitted requests"
    );
    Outcome {
        streams,
        finished,
        cancelled,
        steps,
    }
}

/// Compare each non-cancelled stream to its reference, printing the first
/// divergence. Returns true when every compared stream matched.
fn compare(label: &str, outcome: &Outcome, references: &[Vec<i32>], specs: &[Spec]) -> bool {
    let mut all_ok = true;
    for (index, (stream, reference)) in outcome.streams.iter().zip(references).enumerate() {
        if outcome.cancelled[index] {
            continue;
        }
        if stream != reference {
            all_ok = false;
            let common = stream.len().min(reference.len());
            let divergence = (0..common).find(|&j| stream[j] != reference[j]);
            println!(
                "  {label} req {index} (len {} cap {}): MISMATCH (got {} ref {}, first div {divergence:?})",
                specs[index].0.len(),
                specs[index].1,
                stream.len(),
                reference.len(),
            );
        }
    }
    let unfinished = outcome
        .finished
        .iter()
        .zip(&outcome.cancelled)
        .filter(|(finished, cancelled)| finished.is_none() && !**cancelled)
        .count();
    if unfinished > 0 {
        all_ok = false;
        println!("  {label}: {unfinished} non-cancelled requests never emitted Finished");
    }
    all_ok
}

fn load_gate(
    model: &Path,
    device: &str,
    b_max: usize,
    buckets: Option<&[usize]>,
    reference_capacity: usize,
) -> Box<dyn GateEngine> {
    match buckets {
        Some(capacities) => {
            let (set, dropped) =
                XlaBucketSet::load(model, b_max, device, capacities).expect("load bucket set");
            for reason in dropped {
                println!("  bucket dropped: {reason}");
            }
            Box::new(set)
        }
        None => Box::new(
            XlaBatchEngine::load_with_context_capacity(model, b_max, device, reference_capacity)
                .expect("load batch engine"),
        ),
    }
}

/// The concrete engine phase 2 drives. It is loaded outside the timed region
/// (issue #2961): weight upload and module load are one-time costs, and folding
/// them into the decode window made the reported tok/s a function of the load
/// path rather than of the decode kernel.
enum LoadedGate {
    Buckets(XlaBucketSet),
    Engine(XlaBatchEngine),
}

impl LoadedGate {
    fn as_gate(&mut self) -> &mut dyn GateEngine {
        match self {
            LoadedGate::Buckets(set) => set,
            LoadedGate::Engine(engine) => engine,
        }
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

    let buckets: Option<Vec<usize>> = {
        let raw = arg("--buckets", "");
        if raw.is_empty() {
            None
        } else {
            let values: Vec<usize> = raw
                .split(',')
                .map(|value| {
                    value
                        .trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("--buckets element {value:?} is not an int"))
                })
                .collect();
            assert!(!values.is_empty(), "--buckets needs at least one capacity");
            Some(values)
        }
    };
    let sorted_buckets: Option<Vec<usize>> = buckets.as_ref().map(|values| {
        let mut sorted = values.clone();
        sorted.sort_unstable();
        sorted.dedup();
        sorted
    });
    let seed: Option<u64> = {
        let raw = arg("--seed", "");
        if raw.is_empty() {
            None
        } else {
            Some(raw.parse().expect("--seed int"))
        }
    };
    let temperature: f32 = arg("--temperature", "0.8")
        .parse()
        .expect("--temperature float");
    let top_p: f32 = arg("--top-p", "0.95").parse().expect("--top-p float");
    let cancel_count: usize = arg("--cancel", "0").parse().expect("--cancel int");

    let prompt_ids = read_int_array(&prompts_json, "prompt_ids");
    let full = prompt_ids.len();
    assert!(full > 0, "the prompts artifact has no prompt ids");
    let eos = read_eos(&model);

    // N requests with varied lengths AND budgets (same spread as the spike
    // scheduler), so they evict at staggered steps and later ones admit mid-stream.
    let mut specs: Vec<Spec> = (0..nreq)
        .map(|i| {
            let len = (20 + (i * 13) % full.saturating_sub(20).max(1)).min(full);
            let cap = (12 + (i * 11) % 28).min(maxcap);
            (prompt_ids[..len].to_vec(), cap)
        })
        .collect();

    // One extra request per capacity bucket, sized so it routes there: the
    // prompt fills capacity - 1 tokens and the budget is 1, so prompt + budget
    // lands exactly on that capacity and above the previous one. Prompts are
    // cycled when the source artifact is shorter than the target shape.
    let mut coverage: Vec<(usize, usize)> = Vec::new();
    if let Some(capacities) = &sorted_buckets {
        for (index, &capacity) in capacities.iter().enumerate() {
            let len = capacity.saturating_sub(1).max(1);
            let prompt = (0..len).map(|i| prompt_ids[i % full]).collect();
            specs.push((prompt, 1));
            coverage.push((specs.len() - 1, index));
        }
    }

    let reference_capacity = match &sorted_buckets {
        Some(capacities) => *capacities.last().expect("non-empty --buckets"),
        None => mlxcel_xla::context_capacity_from_env().expect("context capacity"),
    };
    assert!(
        cancel_count == 0 || specs.len() > b_max,
        "--cancel needs --requests > --batch"
    );

    println!(
        "model = {}\ndevice = {device}, slots B_max = {b_max}, requests N = {}, \
         full prompt = {full}, buckets = {:?}, eos = {eos:?}",
        model.display(),
        specs.len(),
        sorted_buckets,
    );
    for (i, (prompt, cap)) in specs.iter().enumerate() {
        print!(
            "  req {i}: len {} cap {cap}{}",
            prompt.len(),
            if i % 4 == 3 { "\n" } else { "" }
        );
    }
    println!();

    // === Phase 1: capture an independent single-seq reference per request ===
    // The reference engine loads the weights once and reuses them; the KV is
    // overwritten per request. It is dropped before the batched engine loads, so
    // peak memory stays at one resident weight set.
    let t_ref_load = Instant::now();
    let mut refeng =
        XlaReferenceEngine::load_with_context_capacity(&model, &device, reference_capacity)
            .expect("load reference engine");
    println!(
        "phase 1 load: reference engine (B_max=4, capacity {reference_capacity}) {:.2} s",
        t_ref_load.elapsed().as_secs_f64()
    );
    let t_ref = Instant::now();
    let references: Vec<Vec<i32>> = specs
        .iter()
        .map(|(prompt, cap)| {
            refeng
                .generate(prompt, *cap, &eos)
                .expect("reference generate")
        })
        .collect();
    let ref_tokens: usize = references.iter().map(Vec::len).sum();
    let ref_secs = t_ref.elapsed().as_secs_f64();
    drop(refeng);
    println!(
        "phase 1: captured {} single-seq references ({ref_tokens} tokens, \
         {:.2} tok/s sequential baseline, generation only)",
        specs.len(),
        ref_tokens as f64 / ref_secs
    );

    // === Phase 2: drive all requests through the continuous-batching engine ===
    // Load first, then start the clock: the #2933 targets are decode throughput,
    // so the one-time weight upload must not sit in the denominator (#2961).
    let mut all_ok = true;
    let t_load = Instant::now();
    let mut loaded = match &sorted_buckets {
        Some(capacities) => {
            let (set, dropped) =
                XlaBucketSet::load(&model, b_max, &device, capacities).expect("load bucket set");
            for reason in &dropped {
                println!("  bucket dropped: {reason}");
            }
            // Prove the added requests really land in every requested bucket.
            for &(spec_index, bucket_index) in &coverage {
                let (prompt, cap) = &specs[spec_index];
                assert_eq!(
                    set.route(prompt.len(), *cap),
                    Some(bucket_index),
                    "coverage request {spec_index} must route into bucket {bucket_index} \
                     ({:?})",
                    set.capacities(),
                );
            }
            LoadedGate::Buckets(set)
        }
        None => LoadedGate::Engine(
            XlaBatchEngine::load_with_context_capacity(&model, b_max, &device, reference_capacity)
                .expect("load batch engine"),
        ),
    };
    println!(
        "phase 2 load: {} (B_max={b_max}, capacity {reference_capacity}) {:.2} s",
        if sorted_buckets.is_some() {
            "bucket set"
        } else {
            "batch engine"
        },
        t_load.elapsed().as_secs_f64()
    );
    let t_run = Instant::now();
    let outcome = drive(
        loaded.as_gate(),
        &specs,
        maxcap,
        &|_| SampleParams::greedy(),
        0,
    );
    drop(loaded);
    let run_secs = t_run.elapsed().as_secs_f64();
    let gen_tokens: usize = outcome.streams.iter().map(Vec::len).sum();
    all_ok &= compare("greedy", &outcome, &references, &specs);
    println!(
        "\nphase 2: {} pump steps, {} requests served, {gen_tokens} tokens",
        outcome.steps,
        specs.len(),
    );
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
        "BENCH\tdevice={device}\tB={b_max}\tN={}\ttok_s={:.3}\tpass={all_ok}",
        specs.len(),
        gen_tokens as f64 / run_secs
    );
    if sorted_buckets.is_some() {
        println!(
            "BUCKETS {} (every bucket routed and reference-exact)",
            if all_ok {
                "REFERENCE-EXACT PASS"
            } else {
                "MISMATCH"
            }
        );
    }

    // === Seeded sampling: determinism plus a top_k = 1 greedy equivalence ===
    if let Some(seed) = seed {
        let seeded_params = |index: usize| SampleParams {
            temperature,
            top_p,
            seed: Some(seed.wrapping_add(index as u64)),
            ..SampleParams::greedy()
        };
        let first = {
            let mut engine = load_gate(
                &model,
                &device,
                b_max,
                sorted_buckets.as_deref(),
                reference_capacity,
            );
            drive(&mut *engine, &specs, maxcap, &seeded_params, 0)
        };
        let second = {
            let mut engine = load_gate(
                &model,
                &device,
                b_max,
                sorted_buckets.as_deref(),
                reference_capacity,
            );
            drive(&mut *engine, &specs, maxcap, &seeded_params, 0)
        };
        let deterministic = first.streams == second.streams;
        let top1_params = |index: usize| SampleParams {
            temperature,
            top_p: 1.0,
            top_k: 1,
            seed: Some(seed.wrapping_add(index as u64)),
            ..SampleParams::greedy()
        };
        let top1 = {
            let mut engine = load_gate(
                &model,
                &device,
                b_max,
                sorted_buckets.as_deref(),
                reference_capacity,
            );
            drive(&mut *engine, &specs, maxcap, &top1_params, 0)
        };
        let top1_exact = compare("top_k=1", &top1, &references, &specs);
        let passed = deterministic && top1_exact;
        all_ok &= passed;
        println!(
            "SEEDED SAMPLING {} (deterministic={deterministic}, top_k=1==greedy={top1_exact})",
            if passed { "PASS" } else { "MISMATCH" }
        );
    }

    // === Cancellation: silence after cancel, slot reuse, exact survivors ===
    if cancel_count > 0 {
        let mut engine = load_gate(
            &model,
            &device,
            b_max,
            sorted_buckets.as_deref(),
            reference_capacity,
        );
        let outcome = drive(
            &mut *engine,
            &specs,
            maxcap,
            &|_| SampleParams::greedy(),
            cancel_count,
        );
        let cancelled = outcome.cancelled.iter().filter(|flag| **flag).count();
        let survivors_exact = compare("cancel", &outcome, &references, &specs);
        let passed = cancelled == cancel_count && survivors_exact;
        all_ok &= passed;
        println!(
            "CANCEL {} ({cancelled}/{cancel_count} cancelled, non-cancelled requests reference-exact={survivors_exact})",
            if passed { "PASS" } else { "MISMATCH" }
        );
    }

    std::process::exit(if all_ok { 0 } else { 1 });
}
