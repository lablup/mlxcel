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

//! Token-exactness gate for the OpenXLA/IREE engine against an EXTERNAL greedy
//! oracle (issue #449 M3 Stage 2d, Stage B). Where `xla_batch_bench` proves the
//! batched engine matches its own single-sequence path (internal consistency),
//! this proves the single-sequence path matches a reference captured from a
//! trusted implementation (HF transformers), so the emitted StableHLO graph
//! computes the architecture itself correctly. This is the gate that catches a
//! wrong-but-self-consistent graph, e.g. a misplaced Qwen2 QKV bias or a wrong
//! RoPE table.
//!
//! The oracle JSON (see `spike/openxla/qwen_oracle.py`) holds `prompt_ids` and
//! `ref_token_ids`, the pure next-token-argmax trajectory for N steps with NO
//! EOS stop. This driver runs the engine's greedy generation for exactly N
//! tokens (empty EOS set, so it never stops early) and asserts an exact match.
//!
//! Build needs the `xla-iree` feature (real IREE execution).
//!
//! ```bash
//! # 1) capture the oracle (HF transformers, fp32):
//! python spike/openxla/qwen_oracle.py /models/qwen2.5-0.5b-bf16 /tmp/qwen.json \
//!   "The capital of France is" 40
//! # 2) check the OpenXLA engine against it on CUDA (GB10):
//! IREE_CUDA_HOME=... IREE_CUDA_COMPILE=... cargo run --release --features xla-iree \
//!   --example xla_oracle_check -- --model /models/qwen2.5-0.5b-bf16 \
//!   --oracle /tmp/qwen.json --device cuda
//! ```

use std::path::PathBuf;

use mlxcel_xla::XlaReferenceEngine;

fn arg(flag: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

/// Read an `[int, ...]` array under `key` from a JSON file.
fn read_int_array(path: &std::path::Path, key: &str) -> Vec<i32> {
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

fn main() {
    let model = PathBuf::from(arg("--model", "/home/inureyes/models/qwen2.5-0.5b-bf16"));
    let oracle = PathBuf::from(arg("--oracle", "/tmp/qwen_oracle.json"));
    let device = {
        let a = arg("--device", "");
        if !a.is_empty() {
            a
        } else {
            std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".to_string())
        }
    };

    let prompt_ids = read_int_array(&oracle, "prompt_ids");
    let reference = read_int_array(&oracle, "ref_token_ids");
    assert!(!prompt_ids.is_empty(), "oracle prompt_ids is empty");
    assert!(!reference.is_empty(), "oracle ref_token_ids is empty");

    println!(
        "model = {}\ndevice = {device}, prompt = {} tokens, reference = {} tokens",
        model.display(),
        prompt_ids.len(),
        reference.len(),
    );

    let mut eng = XlaReferenceEngine::load(&model, &device).expect("load reference engine");
    // Empty EOS set: generate exactly reference.len() tokens with no early stop,
    // matching the oracle's no-EOS-stop argmax trajectory so the streams are
    // directly comparable position for position.
    let got = eng
        .generate(&prompt_ids, reference.len(), &[])
        .expect("greedy generate");

    let ok = got == reference;
    if !ok {
        let m = got.len().min(reference.len());
        let div = (0..m).find(|&j| got[j] != reference[j]);
        println!(
            "MISMATCH: got {} tokens, ref {} tokens, first divergence at {div:?}",
            got.len(),
            reference.len(),
        );
        if let Some(j) = div {
            let lo = j.saturating_sub(2);
            let hi = (j + 3).min(m);
            println!("  ref[{lo}..{hi}] = {:?}", &reference[lo..hi]);
            println!("  got[{lo}..{hi}] = {:?}", &got[lo..hi]);
        }
    }
    println!(
        "RESULT: {}",
        if ok {
            "TOKEN-EXACT PASS (OpenXLA greedy == HF oracle)"
        } else {
            "MISMATCH"
        }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
