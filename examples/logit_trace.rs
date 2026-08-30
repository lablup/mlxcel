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

// Teacher-forced per-position logit trace, for judging a change that moves
// the numbers.
//
// ## Why this exists
//
// Two tools already answer neighbouring questions and neither answers this
// one. Byte-identity (`models::speculative_exactness`) asks whether an
// arithmetic path is bit-equal to its reference: a yes/no that carries no
// information once the answer is no, and on Apple GPU generation 15 and
// newer it is no for reasons the caller did not choose. `examples/perplexity`
// asks whether a model's predictive distribution is worse on a corpus: a
// single scalar that a kernel reordering can leave unmoved while it flips
// percents of the greedy tokens a user would actually see.
//
// The question in between, and the one that keeps coming up, is: this change
// moves the numbers, so **what does it move, and does it matter**. That needs
// three things the scalar does not have.
//
// - Per position, not per corpus, so a disagreement can be counted rather
//   than averaged away.
// - The reference's own **decidedness** at each position. A model that was
//   near-indifferent between its top two has no right answer to disagree
//   with; a model that was decided does. Measured on the MTP drafter, the
//   target's top-two gap has median 9.25 where a prediction was right and
//   1.88 where it was wrong, so pooling the two hides the only distinction
//   that matters (issue #1204).
// - A rank, not just a match. Disagreeing by picking the reference's second
//   choice and disagreeing by picking its four-thousandth are different
//   failures, and only one of them is noise.
//
// ## Why it writes a trace instead of comparing in-process
//
// The changes this is built to judge include process-global runtime
// switches, `MLXCEL_QMV_WIDE` among them, which cannot coexist in one
// process. So each configuration writes its own trace and the comparison is
// offline (`scripts/compare_logit_traces.py`). That also makes the two arms
// separable in time and place: different binaries, different machines,
// different days, same corpus.
//
// ## What it does not do
//
// It does not generate. Both arms are teacher-forced over the same token
// stream, which is the point: a free-running comparison loses all its
// statistical power at the first flipped token, because everything after it
// is conditioned on different text. Measured that way on a real pairing, a
// 250-token generation yielded 16 comparable positions before the arms
// parted; teacher-forced, the same run yields all of them.
//
// Usage:
//   cargo run --release --features metal,accelerate --example logit_trace -- \
//       MODEL_DIR TEXT_FILE [CHUNK_TOKENS=256] [MAX_CHUNKS=4] [TOPK=8] [PREFILL=0] > trace.tsv
//
// ## Getting the shape right, which decides the answer
//
// A forward over `CHUNK_TOKENS` positions runs the quantized projections at
// `M = CHUNK_TOKENS`, and MLX selects a different kernel per `M`. So the
// chunk width is not a sampling knob, it selects what is being measured.
// Comparing `MLXCEL_QMV_WIDE` on against off on gemma-4-12b-it-4bit:
//
// | chunk width | top-1 disagreement |
// |---|---|
// | 8 | 20.6% |
// | 16 | 19.5% |
// | 32 | 0.0% |
// | 256 | 0.0% |
//
// The same change reads as catastrophic or as absent depending only on the
// width chosen, because `use_qmv_wide` splits at `M >= 2` and the batch limit
// sends larger `M` to a matrix-matrix kernel both arms share. Trace at the
// width the code under test actually runs at: an MTP verify block is the
// block size, a decode step is 1, a prefill is the prompt length.
//
// `PREFILL` separates context length from forward width. With `PREFILL=N` the
// first `N` tokens of a chunk are consumed in one forward whose rows are not
// traced, and the remaining rows are traced from a forward at the chunk's own
// width. That is the shape a real verify runs at, long context and narrow
// forward, which a bare small chunk does not reproduce.

use anyhow::{Context, Result};
use mlxcel::LanguageModel;
use mlxcel_core::{
    argpartition, array_to_raw_bytes, astype, eval, from_slice_i32, log_softmax, take_along_axis,
};

/// Read an evaluated array as `f32`, whatever it was stored as.
fn to_f32_vec(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    let a = astype(arr, mlxcel_core::dtype::FLOAT32);
    eval(&a);
    array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Read an evaluated array as `i32`.
fn to_i32_vec(arr: &mlxcel_core::MlxArray) -> Vec<i32> {
    let a = astype(arr, mlxcel_core::dtype::INT32);
    eval(&a);
    array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args
        .get(1)
        .context("usage: logit_trace MODEL_DIR TEXT_FILE [CHUNK_TOKENS] [MAX_CHUNKS] [TOPK]")?;
    let text_file = args
        .get(2)
        .context("usage: logit_trace MODEL_DIR TEXT_FILE [CHUNK_TOKENS] [MAX_CHUNKS] [TOPK]")?;
    let chunk_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let max_chunks: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    // Eight is enough to separate "picked the runner-up" from "picked
    // something unrelated" without making the trace large; anything past the
    // cut is reported as `>k`, which is already the interesting verdict.
    let topk: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
    // Context to establish before the traced forward, so forward width and
    // context length can be varied independently.
    let prefill: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0);

    // The b10621 RoPE override is process-wide (see
    // `mlxcel::models::rope_overrides`), so comparing two rotations means two
    // processes, exactly as this harness already handles `MLXCEL_QMV_WIDE`.
    // Installing it from `LLAMA_ARG_ROPE_*` here means the arms are selected the
    // same way the server selects them, through one set of flag definitions.
    let rope_override = mlxcel::cli::rope_args::install_from_env()
        .map_err(|message| anyhow::anyhow!("{message}"))?;

    let text = std::fs::read_to_string(text_file)
        .with_context(|| format!("reading corpus {text_file}"))?;
    // LoRA arms (#1439): `MLXCEL_TRACE_LORA=PATH:SCALE[,PATH:SCALE...]`
    // loads adapters, and `MLXCEL_TRACE_LORA_MODE` selects `fused`
    // (baked into the weights at load) or `runtime` (unfused terms, the
    // serving default). The runtime arm at scale 0.0 must be byte-identical
    // to a bare run, which is the control the unfused path gates on.
    let lora_env = std::env::var("MLXCEL_TRACE_LORA").ok();
    let (model, tokenizer) = if let Some(spec_str) = lora_env.as_deref() {
        let specs = mlxcel::lora::multi::parse_lora_flags(None, Some(spec_str), false)
            .map_err(|e| anyhow::anyhow!("MLXCEL_TRACE_LORA: {e}"))?;
        let mode = std::env::var("MLXCEL_TRACE_LORA_MODE").unwrap_or_else(|_| "runtime".into());
        println!("# lora\t{spec_str}\tmode\t{mode}");
        match mode.as_str() {
            "fused" => {
                mlxcel::load_model_with_adapter_specs(std::path::Path::new(model_dir), &specs, None)
                    .with_context(|| format!("loading model {model_dir} with fused adapters"))?
            }
            "runtime" => {
                let set = mlxcel::lora::RuntimeLoraSet::from_specs(&specs)?;
                mlxcel::load_model_with_adapter_specs(
                    std::path::Path::new(model_dir),
                    &specs,
                    Some(&set),
                )
                .with_context(|| format!("loading model {model_dir} with runtime adapters"))?
            }
            // The adapter-route base: same load path as the two arms above
            // (raw weights, no sanitize differences), zero adapters. This is
            // the reference the runtime scale-0 arm must be byte-identical
            // to; the plain `load_model` route differs from the adapter
            // route in weight dtype handling, which is load-route variance,
            // not an adapter effect.
            "none" => {
                mlxcel::load_model_with_adapter_specs(std::path::Path::new(model_dir), &[], None)
                    .with_context(|| format!("loading model {model_dir} on the adapter route"))?
            }
            other => {
                anyhow::bail!("MLXCEL_TRACE_LORA_MODE must be fused|runtime|none, found {other}")
            }
        }
    } else {
        mlxcel::load_model(std::path::Path::new(model_dir))
            .with_context(|| format!("loading model {model_dir}"))?
    };

    let ids: Vec<i32> = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .iter()
        .map(|&t| t as i32)
        .collect();
    let n_chunks = ((ids.len() - 1) / chunk_tokens).min(max_chunks);
    anyhow::ensure!(n_chunks > 0, "text too short: {} tokens", ids.len());

    // Same BOS anchoring as `examples/perplexity`, and for the same reason
    // (#686): the Gemma family collapses on a BOS-less window, so a
    // mid-corpus chunk scored raw measures the missing anchor rather than
    // the change under test.
    let bos_prefix: Vec<i32> = tokenizer
        .encode("", true)
        .map(|ids| ids.into_iter().take(1).map(|t| t as i32).collect())
        .unwrap_or_default();

    println!("# model\t{model_dir}");
    if let Some(over) = rope_override.as_ref() {
        println!("# rope_override\t{}", over.describe());
    }
    println!("# corpus\t{text_file}\ttokens\t{}", ids.len());
    println!(
        "# chunks\t{n_chunks}\tchunk_tokens\t{chunk_tokens}\ttopk\t{topk}\tprefill\t{prefill}"
    );
    println!("# columns\tchunk\tpos\ttarget\tnll\ttop_ids\ttop_logits");

    for c in 0..n_chunks {
        let seg = &ids[c * chunk_tokens..(c + 1) * chunk_tokens + 1];
        let l = seg.len() as i32;

        let (input_ids, target_offset) = if prefill > 0 {
            // The context pass already carried the anchor and the history, so
            // the traced forward is the continuation itself.
            (seg[..(l as usize - 1)].to_vec(), 0i32)
        } else if c == 0 || bos_prefix.is_empty() {
            (seg[..(l as usize - 1)].to_vec(), 0i32)
        } else {
            let mut with_bos = bos_prefix.clone();
            with_bos.extend_from_slice(&seg[..(l as usize - 1)]);
            (with_bos, bos_prefix.len() as i32)
        };
        let input_len = input_ids.len() as i32;
        let input = from_slice_i32(&input_ids, &[1, input_len]);

        // Fresh sequence per chunk. Models that key KV state on a model-owned
        // slot ignore the external caches, so without this reset every chunk
        // after the first silently continues the previous one (#686).
        model.reset_runtime_state();
        let mut caches = model.make_caches();
        // Optional context pass. Its rows are discarded; it exists so the
        // traced forward runs at the chunk's width against a realistic
        // context rather than against nothing.
        if prefill > 0 {
            let start = c * chunk_tokens;
            let ctx_from = start.saturating_sub(prefill);
            if ctx_from < start {
                let mut ctx: Vec<i32> = bos_prefix.clone();
                ctx.extend_from_slice(&ids[ctx_from..start]);
                let ctx_len = ctx.len() as i32;
                let ctx_arr = from_slice_i32(&ctx, &[1, ctx_len]);
                let warm = model.forward(&ctx_arr, &mut caches, None);
                eval(&warm);
            }
        }
        let logits = model.forward(&input, &mut caches, None);
        let shape = mlxcel_core::array_shape(&logits);
        anyhow::ensure!(
            shape.len() == 3 && shape[1] == input_len,
            "model.forward returned {shape:?}, not per-position logits"
        );
        let vocab = shape[2];
        anyhow::ensure!(
            vocab as usize > topk,
            "vocabulary {vocab} is smaller than topk {topk}"
        );

        // Row j predicts input[j + 1]; the rows that score `seg[1..]` start
        // at `target_offset`.
        let rows = mlxcel_core::slice(
            &logits,
            &[0, target_offset, 0],
            &[1, target_offset + l - 1, vocab],
        );
        let rows = astype(&rows, mlxcel_core::dtype::FLOAT32);

        // Per-position NLL of the corpus token, so a trace also carries
        // everything `examples/perplexity` reports and the two never
        // disagree about the same run.
        let lp = log_softmax(&rows, -1);
        let targets = from_slice_i32(&seg[1..], &[1, l - 1, 1]);
        let nll = to_f32_vec(&take_along_axis(&lp, &targets, -1));

        // Top-k by partition rather than sort: the vocabulary is large and
        // only the head is wanted. The k entries come back unordered and are
        // sorted host-side, where k is single digits.
        let part = argpartition(&rows, vocab - topk as i32, -1);
        let head = mlxcel_core::slice(&part, &[0, 0, vocab - topk as i32], &[1, l - 1, vocab]);
        let head_ids = to_i32_vec(&head);
        let head_logits = to_f32_vec(&take_along_axis(&rows, &head, -1));

        for j in 0..(l as usize - 1) {
            let mut pairs: Vec<(i32, f32)> = (0..topk)
                .map(|t| (head_ids[j * topk + t], head_logits[j * topk + t]))
                .collect();
            pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
            let ids_s = pairs
                .iter()
                .map(|(i, _)| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let lg_s = pairs
                .iter()
                .map(|(_, v)| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(",");
            let n = -nll[j];
            anyhow::ensure!(
                n.is_finite(),
                "chunk {c} position {j} produced a non-finite NLL; refusing to \
                 write a trace that cannot be compared"
            );
            println!("{c}\t{j}\t{}\t{n:.6}\t{ids_s}\t{lg_s}", seg[j + 1]);
        }

        drop(caches);
        mlxcel_core::clear_memory_cache();
        eprintln!("  chunk {c} traced ({} positions)", l - 1);
    }
    Ok(())
}
