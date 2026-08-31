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

//! DeepSeek-V4 real-checkpoint smoke: load parity plus greedy generation.
//!
//! Loading alone is half the validation: `DeepSeekV4Model::from_weights`
//! runs the strict weight-coverage check, so a successful load proves every
//! tensor in the checkpoint index mapped onto a module path and every module
//! parameter found its tensor, with no silent fallbacks. Generation then
//! exercises all three attention kinds (the real `compress_ratios` mixes
//! local, compressed, and sparse-compressed layers), the hash-routed and
//! bias-routed MoE gates, the HyperConnection Sinkhorn gates, and the
//! rotating-window plus pooling caches across prefill and decode.
//!
//! ## Invocation
//!
//! ```bash
//! cargo test --test deepseek_v4_real_model --release --features metal,accelerate -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` is REQUIRED, not tidiness. Every gate here loads its own
//! copy of the checkpoint, so the default thread count runs as many ~96 GB
//! resident models as there are gates. Three fit on a 512 GB host; the fourth
//! pushes the long-context prefill's activations past the limit and Metal
//! aborts the process with `Insufficient Memory
//! (kIOGPUCommandBufferCallbackErrorOutOfMemory)`. That surfaces as a SIGABRT
//! from C++, not a test failure, so it reads like a port bug rather than a
//! host-capacity one. Adding a gate to this file without the flag is enough to
//! trip it.
//!
//! `#[ignore]`-gated as real-model heavy (~151 GB checkpoint on disk; needs
//! a high-memory Apple Silicon host). Skips silently when the checkpoint
//! directory is absent.

mod common;

use common::repo_model_dir;

use mlxcel::{CxxGenerator, LanguageModel, SamplingConfig, initialize_runtime, load_model};

const MODEL_DIR: &str = "deepseek-v4-flash-4bit";

#[test]
#[ignore]
fn deepseek_v4_real_model_loads_and_generates_coherently() {
    let model_dir = repo_model_dir(MODEL_DIR);
    if !model_dir.join("config.json").exists() {
        eprintln!(
            "Skipping: DeepSeek-V4 checkpoint not found at {}.\n\
             Fetch with: ./target/release/mlxcel download mlx-community/DeepSeek-V4-Flash-4bit",
            model_dir.display()
        );
        return;
    }

    let _runtime = initialize_runtime();
    let (model, tokenizer) = load_model(&model_dir).expect(
        "DeepSeek-V4 checkpoint must load: a failure here is a weight-coverage or \
         quantization-layout mismatch, not an environment problem",
    );

    let prompt = "The capital of France is";
    let prompt_ids: Vec<i32> = tokenizer
        .encode(prompt, true)
        .expect("tokenize prompt")
        .iter()
        .map(|&id| id as i32)
        .collect();
    assert!(!prompt_ids.is_empty());

    let mut generator = CxxGenerator::new(model.num_layers());
    let tokens = generator.generate(&model, &prompt_ids, 24, &SamplingConfig::greedy());
    assert!(
        !tokens.is_empty(),
        "greedy decode must produce at least one token before EOS"
    );

    let gen_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, true).expect("decode generation");
    eprintln!("[deepseek-v4] prompt: {prompt:?}");
    eprintln!(
        "[deepseek-v4] greedy continuation ({} tokens): {text:?}",
        tokens.len()
    );

    assert!(
        text.chars().any(|c| c.is_ascii_alphabetic()),
        "greedy continuation should contain text, got {text:?}"
    );
    assert!(
        text.contains("Paris"),
        "greedy continuation of {prompt:?} should mention Paris; got {text:?}. \
         A fluent-but-wrong answer here usually means a silently-misweighted \
         component (Sinkhorn order, overlap compressor, selection-vs-weighting \
         gate contract), not a load failure"
    );
}

#[test]
#[ignore]
fn deepseek_v4_real_model_decode_crosses_pooling_windows() {
    // A longer decode than one compress window (ratio 4) plus one full
    // simple window boundary region, so decode-mode pooling emission and the
    // HiSA decode fast path both run against real weights.
    let model_dir = repo_model_dir(MODEL_DIR);
    if !model_dir.join("config.json").exists() {
        eprintln!("Skipping: DeepSeek-V4 checkpoint not found");
        return;
    }

    let _runtime = initialize_runtime();
    let (model, tokenizer) = load_model(&model_dir).expect("load DeepSeek-V4 checkpoint");

    let prompt = "Write one short sentence about the ocean.";
    let prompt_ids: Vec<i32> = tokenizer
        .encode(prompt, true)
        .expect("tokenize prompt")
        .iter()
        .map(|&id| id as i32)
        .collect();

    let mut generator = CxxGenerator::new(model.num_layers());
    let tokens = generator.generate(&model, &prompt_ids, 48, &SamplingConfig::greedy());
    let gen_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, true).expect("decode generation");
    eprintln!("[deepseek-v4] 48-token decode: {text:?}");
    assert!(
        text.split_whitespace().count() >= 3,
        "a 48-token greedy decode should produce multiple words, got {text:?}"
    );
}

#[test]
#[ignore]
fn deepseek_v4_real_model_long_context_hits_sparse_and_compressed_paths() {
    // The sparse split-softmax and the batched/decode HiSA selection only
    // run once a ratio-4 layer's pooled count exceeds index_topk (512), i.e.
    // past ~2048 prompt tokens; the same prompt pushes the ratio-128 layers
    // past their first pooled window and rolls the 128-token local window
    // several times. Short smokes cannot reach any of that.
    let model_dir = repo_model_dir(MODEL_DIR);
    if !model_dir.join("config.json").exists() {
        eprintln!("Skipping: DeepSeek-V4 checkpoint not found");
        return;
    }

    let _runtime = initialize_runtime();
    let (model, tokenizer) = load_model(&model_dir).expect("load DeepSeek-V4 checkpoint");

    let mut prompt = String::new();
    let facts = [
        "The Nile is the longest river in Africa. ",
        "Mount Everest rises above the Himalayas. ",
        "Photosynthesis converts light into chemical energy. ",
        "The Pacific Ocean is the largest ocean on Earth. ",
        "Copper conducts electricity better than iron. ",
        "Honey never spoils when stored properly. ",
    ];
    let mut i = 0usize;
    loop {
        prompt.push_str(facts[i % facts.len()]);
        i += 1;
        if i.is_multiple_of(8) {
            let ids = tokenizer.encode(&prompt, true).expect("tokenize");
            if ids.len() > 2200 {
                break;
            }
        }
    }
    prompt.push_str("\n\nQuestion: Which ocean is the largest on Earth?\nAnswer:");

    let prompt_ids: Vec<i32> = tokenizer
        .encode(&prompt, true)
        .expect("tokenize long prompt")
        .iter()
        .map(|&id| id as i32)
        .collect();
    assert!(
        prompt_ids.len() > 2100,
        "long-context prompt must exceed index_topk * ratio tokens, got {}",
        prompt_ids.len()
    );
    eprintln!(
        "[deepseek-v4] long-context prompt: {} tokens",
        prompt_ids.len()
    );

    let mut generator = CxxGenerator::new(model.num_layers());
    let tokens = generator.generate(&model, &prompt_ids, 12, &SamplingConfig::greedy());
    assert!(
        !tokens.is_empty(),
        "long-context decode must produce tokens"
    );
    let gen_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, true).expect("decode generation");
    eprintln!("[deepseek-v4] long-context answer: {text:?}");
    assert!(
        text.to_lowercase().contains("pacific"),
        "retrieval across the sparse pooled path should answer Pacific; got {text:?}"
    );
}

/// The HiSA hierarchy itself, on real weights.
///
/// `deepseek_v4_real_model_long_context_hits_sparse_and_compressed_paths` runs
/// at ~2234 tokens, which at ratio 4 is 558 pooled rows: past `index_topk`
/// (512), so the sparse split-softmax combine engages, but short of
/// `index_block * index_keep` (1024), so selection takes the FLAT scan. That
/// test therefore says nothing about `hisa_select_decode` or
/// `hisa_select_batched`, which until this gate existed had only ever run on
/// synthetic unit fixtures.
///
/// This prompt clears `ratio * index_block * index_keep` tokens, so prefill
/// takes the batched hierarchical path and every decode step takes the fast
/// path. The thresholds are read from the real config rather than hardcoded,
/// so a config change that quietly moves the boundary fails here instead of
/// silently downgrading the test to the flat scan again.
///
/// What it proves: the hierarchy runs on real weights, end to end, and still
/// retrieves a fact planted before the sparse region. What it does not prove:
/// that retrieval REQUIRED the pooled path, since the planted fact is also
/// world knowledge. Making the answer purely synthetic would test retrieval
/// harder and be far less stable on a base checkpoint at this length.
#[test]
#[ignore]
fn deepseek_v4_real_model_long_context_engages_hisa_hierarchy() {
    let model_dir = repo_model_dir(MODEL_DIR);
    if !model_dir.join("config.json").exists() {
        eprintln!("Skipping: DeepSeek-V4 checkpoint not found");
        return;
    }

    let config_str = std::fs::read_to_string(model_dir.join("config.json")).expect("read config");
    let args: mlxcel::models::deepseek_v4::ModelArgs =
        serde_json::from_str(&config_str).expect("parse config");
    let args = args.normalized().expect("validate config");
    let ratio = *args
        .compress_ratios
        .iter()
        .filter(|r| **r > 0)
        .min()
        .expect("the real checkpoint has sparse layers") as usize;
    let block = args.index_block;
    let keep = args.index_keep;
    // Pooled rows needed before `use_hierarchy` opens, and the prompt length
    // that produces them at the sparse layers' ratio.
    let pooled_needed = block * keep;
    let tokens_needed = ratio * pooled_needed;

    let _runtime = initialize_runtime();
    let (model, tokenizer) = load_model(&model_dir).expect("load DeepSeek-V4 checkpoint");

    // The answer fact leads, then filler pushes it deep into the pooled prefix.
    let mut prompt = String::from("The Pacific Ocean is the largest ocean on Earth. ");
    let filler = [
        "The Nile is the longest river in Africa. ",
        "Mount Everest rises above the Himalayas. ",
        "Photosynthesis converts light into chemical energy. ",
        "Copper conducts electricity better than iron. ",
        "Honey never spoils when stored properly. ",
        "Basalt forms when lava cools quickly. ",
    ];
    let mut i = 0usize;
    loop {
        prompt.push_str(filler[i % filler.len()]);
        i += 1;
        if i.is_multiple_of(16) {
            let ids = tokenizer.encode(&prompt, true).expect("tokenize");
            if ids.len() > tokens_needed + 300 {
                break;
            }
        }
    }
    prompt.push_str("\n\nQuestion: Which ocean is the largest on Earth?\nAnswer:");

    let prompt_ids: Vec<i32> = tokenizer
        .encode(&prompt, true)
        .expect("tokenize")
        .iter()
        .map(|&id| id as i32)
        .collect();
    let pooled_rows = prompt_ids.len() / ratio;
    eprintln!(
        "[deepseek-v4] hisa prompt: {} tokens -> ~{} pooled rows at ratio {} \
         (hierarchy needs {})",
        prompt_ids.len(),
        pooled_rows,
        ratio,
        pooled_needed
    );
    assert!(
        pooled_rows >= pooled_needed,
        "prompt must clear index_block * index_keep = {pooled_needed} pooled rows \
         to reach the hierarchy, got {pooled_rows}"
    );
    assert!(
        pooled_rows > args.index_topk,
        "prompt must also clear index_topk = {} so selection is consumed at all",
        args.index_topk
    );

    let mut generator = CxxGenerator::new(model.num_layers());
    let tokens = generator.generate(&model, &prompt_ids, 12, &SamplingConfig::greedy());
    assert!(
        !tokens.is_empty(),
        "hierarchical decode must produce tokens"
    );
    let gen_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, true).expect("decode");
    eprintln!("[deepseek-v4] hisa answer: {text:?}");
    assert!(
        text.to_lowercase().contains("pacific"),
        "retrieval through the HiSA hierarchy should answer Pacific; got {text:?}"
    );
}

/// Measures how decode cost scales with the pooled prefix (#549 criterion 4).
///
/// **This REPORTS; it does not gate.** Two attempts to turn it into a gate both
/// measured the wrong thing, and the numbers it prints should be read as a
/// smoke signal, not as evidence about decode scaling.
///
/// Attempt 1 timed one whole `generate()` and divided by tokens generated. That
/// includes prefill, which at 8900 tokens is roughly 37 seconds against about 4
/// seconds of decode, so the figure was prefill cost wearing decode's clothes:
/// it tracked the prefix because prefill tracks prompt length, and the
/// `MLXCEL_V4_FLAT_INDEX` arm looked identical because prefill dominated both.
///
/// Attempt 2 (what the code below still does) times two decode lengths and
/// subtracts so prefill cancels. It does not cancel: `t_short` and `t_long` are
/// separate `generate()` calls each paying their own prefill, so any difference
/// between the two prefills lands whole in the estimate. Prefill for one
/// identical prompt was measured between 16 and 54 seconds across runs, and the
/// 4513-token decode figure came out 98.3, 369.5, then 186.4 ms/step on
/// consecutive runs, giving growth ratios of x1.01, x1.39 and x0.49 (decode
/// apparently getting faster as context doubled, which is not a thing). A gate
/// on that fails at random and teaches people to ignore it.
///
/// The right tool already exists: `src/bin/bench_decode.rs`
/// (`mlxcel-bench-decode`) loads the model once, warms up, and measures in a
/// single process, and its header documents exactly the cold-prefill artifact
/// attempt 1 fell into. Settling whether the HiSA hierarchy pays for itself
/// means driving that harness across context lengths with and without
/// `MLXCEL_V4_FLAT_INDEX` on a quiet machine. See #549.
///
/// What this test still earns its place doing: driving real decode at three
/// context lengths spanning the dense, flat-scan and hierarchy bands, so a
/// crash or a degenerate output in any band shows up.
///
#[test]
#[ignore]
fn deepseek_v4_real_model_hisa_decode_cost_scaling() {
    let model_dir = repo_model_dir(MODEL_DIR);
    if !model_dir.join("config.json").exists() {
        eprintln!("Skipping: DeepSeek-V4 checkpoint not found");
        return;
    }

    let config_str = std::fs::read_to_string(model_dir.join("config.json")).expect("read config");
    let args: mlxcel::models::deepseek_v4::ModelArgs =
        serde_json::from_str(&config_str).expect("parse config");
    let args = args.normalized().expect("validate config");
    let ratio = *args
        .compress_ratios
        .iter()
        .filter(|r| **r > 0)
        .min()
        .expect("sparse layers exist") as usize;
    let hierarchy_rows = args.index_block * args.index_keep;

    let _runtime = initialize_runtime();
    let (model, tokenizer) = load_model(&model_dir).expect("load DeepSeek-V4 checkpoint");

    let filler = [
        "The Nile is the longest river in Africa. ",
        "Mount Everest rises above the Himalayas. ",
        "Photosynthesis converts light into chemical energy. ",
        "Copper conducts electricity better than iron. ",
        "Honey never spoils when stored properly. ",
        "Basalt forms when lava cools quickly. ",
    ];
    let build = |target: usize| -> Vec<i32> {
        let mut prompt = String::new();
        let mut i = 0usize;
        loop {
            prompt.push_str(filler[i % filler.len()]);
            i += 1;
            if i.is_multiple_of(16) {
                let ids = tokenizer.encode(&prompt, true).expect("tokenize");
                if ids.len() >= target {
                    break;
                }
            }
        }
        tokenizer
            .encode(&prompt, true)
            .expect("tokenize")
            .iter()
            .map(|&id| id as i32)
            .collect()
    };

    // Two decode lengths per context so prefill cancels: t(long) - t(short)
    // over the step delta is the per-step decode cost, while a single timed
    // generate() would be dominated by prefill and would report prompt-length
    // scaling dressed up as decode scaling.
    const SHORT_STEPS: usize = 8;
    const LONG_STEPS: usize = 40;
    // One untimed warmup: the first generate() of a process pays kernel and
    // allocator init, which lands entirely in whichever run happens to go
    // first and can make t(long) - t(short) negative.
    {
        let warm = build(512);
        let mut g = CxxGenerator::new(model.num_layers());
        let _ = g.generate(&model, &warm, 4, &SamplingConfig::greedy());
    }

    let mut measured = Vec::new();
    // Two lengths inside the hierarchy regime (the second doubles the pooled
    // prefix), plus one below it for context on where the bands sit.
    for target in [2200usize, 4400, 8800] {
        let ids = build(target);
        let pooled = ids.len() / ratio;
        let run = |steps: usize| -> (f64, usize) {
            let mut generator = CxxGenerator::new(model.num_layers());
            let start = std::time::Instant::now();
            let tokens = generator.generate(&model, &ids, steps, &SamplingConfig::greedy());
            (start.elapsed().as_secs_f64() * 1000.0, tokens.len())
        };
        let (t_short, n_short) = run(SHORT_STEPS);
        let (t_long, n_long) = run(LONG_STEPS);
        assert!(
            n_long > n_short,
            "the long run must decode more tokens than the short one at {target} \
             (got {n_long} vs {n_short}); an early EOS breaks the subtraction"
        );
        let per_token_ms = (t_long - t_short) / (n_long - n_short) as f64;
        let prefill_ms = t_short - per_token_ms * n_short as f64;
        let path = if pooled >= hierarchy_rows {
            "hierarchy"
        } else if pooled > args.index_topk {
            "flat scan"
        } else {
            "dense"
        };
        eprintln!(
            "[deepseek-v4] {:>5} tokens -> {:>5} pooled rows ({:<9}) | prefill ~{:>8.0} ms | decode {:>6.1} ms/step",
            ids.len(),
            pooled,
            path,
            prefill_ms,
            per_token_ms
        );
        measured.push((ids.len(), pooled, path, per_token_ms));
    }

    let hier: Vec<_> = measured.iter().filter(|m| m.2 == "hierarchy").collect();
    assert!(
        hier.len() >= 2,
        "need two hierarchy-band points to compare; got {measured:?}"
    );
    let (small, large) = (hier[0], hier[hier.len() - 1]);
    let growth = large.3 / small.3;
    let pooled_growth = large.1 as f64 / small.1 as f64;
    eprintln!(
        "[deepseek-v4] pooled rows x{pooled_growth:.2} -> decode ms/token x{growth:.2} \
         (O(Np) selection would track the former)"
    );
    // Sanity only. Asserting anything about `growth` would be asserting on a
    // quantity measured at 4x spread between runs (see the doc comment), so the
    // only claims made here are ones the noise cannot flip: the arithmetic
    // produced usable numbers at every band.
    for (tokens, pooled, path, ms) in &measured {
        assert!(
            ms.is_finite(),
            "{tokens} tokens ({pooled} pooled, {path}): non-finite decode estimate"
        );
    }
}

/// Not `#[ignore]`: parsing the real config costs nothing and catches serde
/// shape drift (e.g. the checkpoint ships BOTH `quantization` and
/// `quantization_config`, and 44 `compress_ratios` for 43 layers) without
/// touching the shards. Skips when the checkpoint directory is absent.
#[test]
fn deepseek_v4_real_config_parses_and_validates() {
    let config_path = repo_model_dir(MODEL_DIR).join("config.json");
    let Ok(config_str) = std::fs::read_to_string(&config_path) else {
        eprintln!("Skipping: {} not present", config_path.display());
        return;
    };
    let args: mlxcel::models::deepseek_v4::ModelArgs =
        serde_json::from_str(&config_str).expect("real config.json must parse");
    let args = args.normalized().expect("real config.json must validate");
    assert_eq!(args.num_hidden_layers, 43);
    assert_eq!(args.compress_ratios.len(), 43);
    assert_eq!(args.compress_ratios[0], 0);
    assert_eq!(args.compress_ratios[42], 4);
    assert_eq!(args.num_hash_layers, 3);
    let (gs, bits, mode) = {
        let o = args
            .quantization_override("model.layers.0.ffn.switch_mlp.gate_proj")
            .expect("real config declares the expert override");
        (o.group_size, o.bits, o.mode.unwrap_or_default())
    };
    assert_eq!((gs, bits, mode.as_str()), (32, 4, "mxfp4"));
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
    assert_eq!(args.eos_token_ids(), vec![1]);
}
