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
//
// Every chunk, chunk 0 included, is anchored by the tokenizer's BOS: without
// `PREFILL` it is prepended to the traced forward, with `PREFILL` it leads the
// context pass (alone, for a chunk with no history). The `# bos_anchor`
// metadata line records it; traces without that line left chunk 0 unanchored
// and are not comparable with newer ones at chunk 0.
//
// `MLXCEL_TRACE_KV_CACHE_MODE` and `MLXCEL_TRACE_DECODE_STEP` select the KV
// cache storage and end the context pass with a single-token step (see
// `logit_trace/kv_mode.rs`); together they reach the quantized sliding-cache
// append a speculative verify block runs after decode (#1793). Warnings from
// the runtime go to stderr (`RUST_LOG` overrides the `warn` default), so an
// arm that reaches a fallback branch says so.
//
// `MLXCEL_TRACE_BATCH_ROWS=N` runs every traced position as a width-one
// `forward_batched_with_ids` decode over N independently prefixed rows with
// fresh `SequenceId`s. Only row zero is emitted, so traces from N=1 and N=4
// compare directly. With `MLXCEL_TRACE_IMAGE`, row zero is the real image
// prompt and the companion rows are text-only.
//
// `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1` requires
// `MLXCEL_TRACE_BATCH_ROWS=1` and routes each decode step through
// `forward_with_sequence_id`. Use it to build a scalar reference trace for a
// model whose baseline predates its `forward_batched_with_ids` override; the
// trait default drops sequence IDs, so its nominal B=1 batched trace can read
// the fallback state instead of the ID-keyed state established by prefill.
//
// `MLXCEL_TRACE_TP_SIZE=N` loads the model through the production tensor-
// parallel loader. Combine it with `MLXCEL_TRACE_BATCH_ROWS=1` and `=4` to
// compare the same teacher-forced row through TP native batched decode. It is
// restricted to the standard text arm and cannot be combined with LoRA.
//
// `MLXCEL_TRACE_AUDIO` and `MLXCEL_TRACE_AUDIO_PROMPT` trace a Gemma 4
// transcript behind an audio clip instead of the chunked corpus (see
// `logit_trace/audio.rs`); the corpus argument is then the transcript.
//
// `MLXCEL_TRACE_IMAGE=PATH` traces an image-prompted continuation instead of
// corpus chunks: one image and `MLXCEL_TRACE_IMAGE_PROMPT` through the chat
// template, then the first `CHUNK_TOKENS` tokens of `TEXT_FILE` teacher-forced
// as the reply (see `logit_trace/image.rs`). It is how a vision-tower change
// is judged, since only image-conditioned text can move (#2388).

#[path = "logit_trace/audio.rs"]
mod audio;
#[path = "logit_trace/gemma4.rs"]
mod gemma4;
#[path = "logit_trace/image.rs"]
mod image_arm;
#[path = "logit_trace/kv_mode.rs"]
mod kv_mode;

use anyhow::{Context, Result};
use mlxcel::{LanguageModel, distributed::ShardConfig};
use mlxcel_core::{
    argpartition, array_to_raw_bytes, astype, eval, from_slice_i32, log_softmax, take_along_axis,
};
use std::io::Write;

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

fn parse_trace_tp_size(raw: Option<&str>) -> Result<Option<usize>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let size = raw
        .parse::<usize>()
        .context("MLXCEL_TRACE_TP_SIZE must be a positive integer")?;
    anyhow::ensure!(size > 0, "MLXCEL_TRACE_TP_SIZE must be a positive integer");
    Ok(Some(size))
}

fn validate_scalar_reference(
    enabled: bool,
    batch_rows: Option<usize>,
    has_image_arm: bool,
) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    anyhow::ensure!(
        batch_rows == Some(1),
        "MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 requires MLXCEL_TRACE_BATCH_ROWS=1"
    );
    anyhow::ensure!(
        !has_image_arm,
        "MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE supports the standard text arm only"
    );
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();
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
    let batch_rows: Option<usize> = std::env::var("MLXCEL_TRACE_BATCH_ROWS")
        .ok()
        .map(|value| value.parse())
        .transpose()
        .context("MLXCEL_TRACE_BATCH_ROWS must be a positive integer")?;
    if let Some(rows) = batch_rows {
        anyhow::ensure!(rows > 0, "MLXCEL_TRACE_BATCH_ROWS must be positive");
    }
    let scalar_reference =
        std::env::var("MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE").as_deref() == Ok("1");
    validate_scalar_reference(
        scalar_reference,
        batch_rows,
        std::env::var_os("MLXCEL_TRACE_IMAGE").is_some(),
    )?;
    let tp_size_env = std::env::var("MLXCEL_TRACE_TP_SIZE").ok();
    let tp_size = parse_trace_tp_size(tp_size_env.as_deref())?;
    // Start after a fixed context instead of spending early chunks growing it.
    let start_token: usize = std::env::var("MLXCEL_TRACE_START_TOKEN")
        .ok()
        .map(|value| value.parse())
        .transpose()
        .context("MLXCEL_TRACE_START_TOKEN must be a non-negative integer")?
        .unwrap_or(0);
    let gemma4_mode = gemma4::Mode::from_env()?;
    let audio_arm = audio::AudioArm::from_env()?;
    let kv_arm = kv_mode::KvArm::from_env()?;
    anyhow::ensure!(
        !kv_arm.decode_step || (prefill > 0 && gemma4_mode.is_none()),
        "MLXCEL_TRACE_DECODE_STEP needs a nonzero prefill and no Gemma 4 mode"
    );
    let contiguous = std::env::var("MLXCEL_TRACE_CONTIGUOUS").as_deref() == Ok("1");
    anyhow::ensure!(
        !contiguous || (gemma4_mode.is_some() && prefill > 0 && start_token > 0),
        "contiguous trace requires a Gemma 4 mode and nonzero prefill/start token"
    );
    if batch_rows.is_some() {
        anyhow::ensure!(
            gemma4_mode.is_none() && audio_arm.is_none() && !contiguous,
            "MLXCEL_TRACE_BATCH_ROWS supports the text and image arms only"
        );
    }
    let lora_env = std::env::var("MLXCEL_TRACE_LORA").ok();
    if tp_size.is_some() {
        anyhow::ensure!(
            gemma4_mode.is_none()
                && audio_arm.is_none()
                && std::env::var_os("MLXCEL_TRACE_IMAGE").is_none()
                && !contiguous,
            "MLXCEL_TRACE_TP_SIZE supports the standard text arm only"
        );
        anyhow::ensure!(
            lora_env.is_none(),
            "MLXCEL_TRACE_TP_SIZE cannot be combined with MLXCEL_TRACE_LORA"
        );
    }

    // The b10621 RoPE override is process-wide (see
    // `mlxcel::models::rope_overrides`), so comparing two rotations means two
    // processes, exactly as this harness already handles `MLXCEL_QMV_WIDE`.
    // Installing it from `LLAMA_ARG_ROPE_*` here means the arms are selected the
    // same way the server selects them, through one set of flag definitions.
    let rope_override = mlxcel::cli::rope_args::install_from_env()
        .map_err(|message| anyhow::anyhow!("{message}"))?;

    // Resolve the device the same way the CLI and server do, so
    // `MLXCEL_DEVICE=cpu` selects the CPU here too. Without this call the
    // override is never read and a "CPU reference" trace silently runs on the
    // GPU (issue #1807, where the CPU device stands in for a Metal baseline).
    let runtime = mlxcel::initialize_runtime_checked().map_err(|err| anyhow::anyhow!("{err}"))?;
    if let Some(invalid) = runtime.invalid_device_override.as_deref() {
        eprintln!("ignoring invalid MLXCEL_DEVICE override {invalid:?}; using gpu");
    }

    let text = std::fs::read_to_string(text_file)
        .with_context(|| format!("reading corpus {text_file}"))?;
    // LoRA arms (#1439): `MLXCEL_TRACE_LORA=PATH:SCALE[,PATH:SCALE...]`
    // loads adapters, and `MLXCEL_TRACE_LORA_MODE` selects `fused`
    // (baked into the weights at load) or `runtime` (unfused terms, the
    // serving default). The runtime arm at scale 0.0 must be byte-identical
    // to a bare run, which is the control the unfused path gates on.
    let (model, tokenizer) = if let Some(tp_size) = tp_size {
        mlxcel::load_model_with_tensor_parallel(
            std::path::Path::new(model_dir),
            None,
            &ShardConfig::with_tp_size(tp_size),
        )
        .with_context(|| format!("loading model {model_dir} with tensor parallel size {tp_size}"))?
    } else if let Some(spec_str) = lora_env.as_deref() {
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

    kv_arm.install(&model)?;

    // The audio arm traces one transcript behind one clip, so the corpus is
    // the transcript and there is no chunking.
    if let Some(arm) = audio_arm {
        println!("# model\t{model_dir}");
        println!("# device\t{}", runtime.device);
        println!(
            "# audio\t{}",
            std::env::var("MLXCEL_TRACE_AUDIO").unwrap_or_default()
        );
        println!("# transcript\t{text_file}\ttopk\t{topk}");
        println!("# columns\tchunk\tpos\ttarget\tnll\ttop_ids\ttop_logits");
        let (row_offset, target, logits) = arm.forward(&model, &tokenizer, &text)?;
        return emit_rows(0, &logits, row_offset as i32, &target, topk);
    }

    if let Some(arm) = image_arm::ImageArm::from_env() {
        anyhow::ensure!(
            gemma4_mode.is_none() && prefill == 0 && !contiguous && start_token == 0,
            "MLXCEL_TRACE_IMAGE runs alone: no Gemma 4 mode, prefill, or start token"
        );
        let traced = image_arm::trace(
            &arm,
            &model,
            &tokenizer,
            model_dir,
            &text,
            chunk_tokens,
            batch_rows,
        )?;
        println!("# model\t{model_dir}");
        println!("# device\t{}", runtime.device);
        println!("# image\t{}\tprompt\t{}", arm.image_path, arm.prompt);
        println!("# prompt_tokens\t{}", traced.prompt_tokens);
        if let Some(rows) = batch_rows {
            println!("# batch_rows\t{rows}");
        }
        println!("# corpus\t{text_file}\ttokens\t{}", traced.targets.len());
        println!(
            "# chunks\t1\tchunk_tokens\t{}\ttopk\t{topk}\tprefill\t0",
            traced.targets.len()
        );
        println!("# columns\tchunk\tpos\ttarget\tnll\ttop_ids\ttop_logits");
        emit_rows(0, &traced.logits, 0, &traced.targets, topk)?;
        eprintln!("  image arm traced ({} positions)", traced.targets.len());
        return Ok(());
    }

    let ids: Vec<i32> = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .iter()
        .map(|&t| t as i32)
        .collect();
    let trace_start = if batch_rows.is_some() {
        start_token.saturating_add(prefill)
    } else {
        start_token
    };
    anyhow::ensure!(
        chunk_tokens > 0 && trace_start < ids.len(),
        "invalid chunk width or trace start"
    );
    let n_chunks = ((ids.len() - trace_start - 1) / chunk_tokens).min(max_chunks);
    anyhow::ensure!(n_chunks > 0, "text too short: {} tokens", ids.len());

    // Same BOS anchoring as `examples/perplexity`, and for the same reason
    // (#686): the Gemma family collapses on a BOS-less window, so a
    // mid-corpus chunk scored raw measures the missing anchor rather than
    // the change under test. Every chunk carries it, chunk 0 included
    // (#1785): the corpus is encoded without special tokens, so chunk 0
    // starts unanchored too, and on Laguna XS 2.1 at width 256 that alone
    // read as perplexity ~1000 against ~17.8 for an anchored chunk. The
    // corpus stays encoded without special tokens because adding them would
    // shift every chunk boundary, not just chunk 0's.
    let bos_prefix: Vec<i32> = tokenizer
        .encode("", true)
        .map(|ids| ids.into_iter().take(1).map(|t| t as i32).collect())
        .unwrap_or_default();

    println!("# model\t{model_dir}");
    println!("# device\t{}", runtime.device);
    if let Some(tp_size) = tp_size {
        println!("# tp_size\t{tp_size}");
    }
    if let Some(over) = rope_override.as_ref() {
        println!("# rope_override\t{}", over.describe());
    }
    println!("# corpus\t{text_file}\ttokens\t{}", ids.len());
    println!(
        "# chunks\t{n_chunks}\tchunk_tokens\t{chunk_tokens}\ttopk\t{topk}\tprefill\t{prefill}"
    );
    println!("# columns\tchunk\tpos\ttarget\tnll\ttop_ids\ttop_logits");
    // Traces written before #1785 left chunk 0 unanchored, so their chunk-0
    // rows are not comparable with a trace that carries this line.
    let (bos_anchor, anchored) = if bos_prefix.is_empty() {
        ("none".to_string(), "none")
    } else {
        let ids = bos_prefix
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        (ids, "all")
    };
    println!("# bos_anchor\t{bos_anchor}\tchunks\t{anchored}");

    println!("# start_token\t{start_token}");
    if let Ok(mode) = std::env::var("MLXCEL_TRACE_GEMMA4") {
        println!("# gemma4_mode\t{mode}");
    }
    println!("# contiguous\t{contiguous}");
    if let Some(rows) = batch_rows {
        println!("# batch_rows\t{rows}");
    }
    if scalar_reference {
        println!("# scalar_reference\ttrue");
    }
    if let Some(mode) = kv_arm.mode {
        println!("# kv_cache_mode\t{mode}");
    }
    if kv_arm.decode_step {
        println!("# decode_step\ttrue");
    }
    if let Some(rows) = batch_rows {
        return trace_batched_chunks(
            &model,
            &ids,
            &bos_prefix,
            trace_start,
            chunk_tokens,
            n_chunks,
            prefill,
            rows,
            topk,
            &kv_arm,
            scalar_reference,
        );
    }
    for c in 0..n_chunks {
        let start = trace_start + c * chunk_tokens;
        let seg = &ids[start..start + chunk_tokens + 1];
        let l = seg.len() as i32;

        let (input_ids, target_offset) = if prefill > 0 {
            // The context pass already carried the anchor and the history, so
            // the traced forward is the continuation itself.
            (seg[..(l as usize - 1)].to_vec(), 0i32)
        } else if bos_prefix.is_empty() {
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
        if !contiguous || c == 0 {
            model.reset_runtime_state();
        }
        let mut caches = model.make_caches();
        // Optional context pass. Its rows are discarded; it exists so the
        // traced forward runs at the chunk's width against a realistic
        // context rather than against nothing. A chunk with no history (chunk
        // 0 at the corpus head) still gets the anchor alone, so the traced
        // forward keeps the chunk's width and `target_offset` stays 0.
        if prefill > 0 && (!contiguous || c == 0) {
            let ctx_from = start.saturating_sub(prefill);
            let mut ctx: Vec<i32> = bos_prefix.clone();
            ctx.extend_from_slice(&ids[ctx_from..start]);
            if !ctx.is_empty() {
                let ctx_len = ctx.len() as i32;
                if let Some(mode) = gemma4_mode {
                    mode.prefill(&model, &ctx, chunk_tokens)?;
                } else {
                    let split = kv_arm.context_split(ctx.len());
                    let ctx_arr = from_slice_i32(&ctx[..split], &[1, split as i32]);
                    let warm = model.forward(&ctx_arr, &mut caches, None);
                    eval(&warm);
                    if split < ctx.len() {
                        let step = from_slice_i32(&ctx[split..], &[1, ctx_len - split as i32]);
                        let warm = model.forward(&step, &mut caches, None);
                        eval(&warm);
                    }
                }
            }
        }
        let logits = if let Some(mode) = gemma4_mode {
            mode.forward(&model, &input)?
        } else {
            model.forward(&input, &mut caches, None)
        };
        anyhow::ensure!(
            mlxcel_core::array_shape(&logits).get(1) == Some(&input_len),
            "model.forward returned {:?}, not per-position logits",
            mlxcel_core::array_shape(&logits)
        );
        // Row j predicts input[j + 1]; the rows that score `seg[1..]` start
        // at `target_offset`.
        emit_rows(c, &logits, target_offset, &seg[1..], topk)?;

        drop(caches);
        mlxcel_core::clear_memory_cache();
        eprintln!("  chunk {c} traced ({} positions)", l - 1);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn trace_batched_chunks(
    model: &mlxcel::LoadedModel,
    ids: &[i32],
    bos_prefix: &[i32],
    trace_start: usize,
    chunk_tokens: usize,
    n_chunks: usize,
    prefill: usize,
    batch_rows: usize,
    topk: usize,
    kv_arm: &kv_mode::KvArm,
    scalar_reference: bool,
) -> Result<()> {
    for chunk in 0..n_chunks {
        let traced_start = trace_start + chunk * chunk_tokens;
        let rows = batched_trace_rows(ids.len(), traced_start, chunk_tokens, prefill, batch_rows)?;

        model.reset_runtime_state();
        let seq_ids: Vec<_> = (0..batch_rows)
            .map(|row| {
                mlxcel_core::cache::SequenceId::from_raw(
                    0x2562_0000 + (chunk * batch_rows + row) as u64,
                )
            })
            .collect();
        let mut caches: Vec<Vec<mlxcel_core::layers::KVCache>> = Vec::with_capacity(batch_rows);
        for (row_idx, row) in rows.iter().enumerate() {
            let mut prefix = bos_prefix.to_vec();
            prefix.extend_from_slice(&ids[row.prefix_from..row.start]);
            let mut row_caches = model.make_caches();
            model.prepare_sequence_state(seq_ids[row_idx]);
            model.bind_qwen_vl_mrope_state_to_sequence(seq_ids[row_idx]);
            if !prefix.is_empty() {
                let split = kv_arm.context_split(prefix.len());
                let input = from_slice_i32(&prefix[..split], &[1, split as i32]);
                let warm = model.forward_with_sequence_id(
                    &input,
                    Some(seq_ids[row_idx]),
                    &mut row_caches,
                    None,
                );
                eval(&warm);
                if split < prefix.len() {
                    let step = from_slice_i32(&prefix[split..], &[1, 1]);
                    let warm = model.forward_with_sequence_id(
                        &step,
                        Some(seq_ids[row_idx]),
                        &mut row_caches,
                        None,
                    );
                    eval(&warm);
                }
            }
            caches.push(row_caches);
        }

        for position in 0..chunk_tokens {
            let tokens: Vec<i32> = rows.iter().map(|row| ids[row.start + position]).collect();
            let input = from_slice_i32(&tokens, &[batch_rows as i32, 1]);
            let logits = if scalar_reference {
                model.forward_with_sequence_id(&input, Some(seq_ids[0]), &mut caches[0], None)
            } else {
                let mut rows: Vec<&mut [mlxcel_core::layers::KVCache]> =
                    caches.iter_mut().map(Vec::as_mut_slice).collect();
                model.forward_batched_with_ids(&input, Some(&seq_ids), &mut rows, None)
            };
            eval(&logits);
            let shape = mlxcel_core::array_shape(&logits);
            anyhow::ensure!(
                shape.len() == 3 && shape[0] == batch_rows as i32 && shape[1] == 1,
                "batched decode returned {shape:?}, expected [{batch_rows}, 1, vocab]"
            );
            let row = mlxcel_core::slice(&logits, &[0, 0, 0], &[1, 1, shape[2]]);
            eval(&row);
            // Emit before advancing again so only one full-vocabulary decode
            // result remains live at a time.
            let target = ids[traced_start + position + 1];
            emit_rows_at(
                chunk,
                position,
                &row,
                0,
                std::slice::from_ref(&target),
                topk,
            )?;
        }
        for seq_id in seq_ids {
            model.release_sequence_state_by_id(seq_id);
        }
        mlxcel_core::clear_memory_cache();
        eprintln!("  chunk {chunk} traced ({chunk_tokens} batched decode positions)");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchedTraceRow {
    start: usize,
    prefix_from: usize,
}

fn batched_trace_rows(
    corpus_len: usize,
    traced_start: usize,
    chunk_tokens: usize,
    prefill: usize,
    batch_rows: usize,
) -> Result<Vec<BatchedTraceRow>> {
    (0..batch_rows)
        .map(|row| {
            let start = traced_start + row * (chunk_tokens + 1);
            anyhow::ensure!(
                start + chunk_tokens < corpus_len,
                "corpus is too short for batch row {row}: need token {}, have {corpus_len}",
                start + chunk_tokens
            );
            let prefix_len = prefill + row;
            anyhow::ensure!(
                start >= prefix_len,
                "corpus is too short to give batch row {row} its distinct prefix"
            );
            Ok(BatchedTraceRow {
                start,
                prefix_from: start - prefix_len,
            })
        })
        .collect()
}

/// Print one trace row per target: row `row_offset + j` of `logits` scores
/// `targets[j]`.
fn emit_rows(
    c: usize,
    logits: &mlxcel_core::MlxArray,
    row_offset: i32,
    targets: &[i32],
    topk: usize,
) -> Result<()> {
    emit_rows_at(c, 0, logits, row_offset, targets, topk)
}

fn emit_rows_at(
    c: usize,
    position_offset: usize,
    logits: &mlxcel_core::MlxArray,
    row_offset: i32,
    targets: &[i32],
    topk: usize,
) -> Result<()> {
    let stdout = std::io::stdout();
    emit_rows_to(
        &mut stdout.lock(),
        c,
        position_offset,
        logits,
        row_offset,
        targets,
        topk,
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_rows_to<W: Write>(
    writer: &mut W,
    c: usize,
    position_offset: usize,
    logits: &mlxcel_core::MlxArray,
    row_offset: i32,
    targets: &[i32],
    topk: usize,
) -> Result<()> {
    let shape = mlxcel_core::array_shape(logits);
    let n = targets.len() as i32;
    anyhow::ensure!(
        shape.len() == 3 && shape[1] >= row_offset + n,
        "model.forward returned {shape:?}, not per-position logits"
    );
    let vocab = shape[2];
    anyhow::ensure!(
        vocab as usize > topk,
        "vocabulary {vocab} is smaller than topk {topk}"
    );

    let rows = mlxcel_core::slice(logits, &[0, row_offset, 0], &[1, row_offset + n, vocab]);
    let rows = astype(&rows, mlxcel_core::dtype::FLOAT32);

    // Per-position NLL of the corpus token, so a trace also carries
    // everything `examples/perplexity` reports and the two never
    // disagree about the same run.
    let lp = log_softmax(&rows, -1);
    let target_arr = from_slice_i32(targets, &[1, n, 1]);
    let nll = to_f32_vec(&take_along_axis(&lp, &target_arr, -1));

    // Top-k by partition rather than sort: the vocabulary is large and
    // only the head is wanted. The k entries come back unordered and are
    // sorted host-side, where k is single digits.
    let part = argpartition(&rows, vocab - topk as i32, -1);
    let head = mlxcel_core::slice(&part, &[0, 0, vocab - topk as i32], &[1, n, vocab]);
    let head_ids = to_i32_vec(&head);
    let head_logits = to_f32_vec(&take_along_axis(&rows, &head, -1));

    for (j, target) in targets.iter().enumerate() {
        let position = position_offset + j;
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
            "chunk {c} position {position} produced a non-finite NLL; refusing to \
             write a trace that cannot be compared"
        );
        writeln!(writer, "{c}\t{position}\t{target}\t{n:.6}\t{ids_s}\t{lg_s}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BatchedTraceRow, batched_trace_rows, emit_rows_to, parse_trace_tp_size,
        validate_scalar_reference,
    };

    #[test]
    fn trace_tp_size_defaults_to_the_existing_loader() {
        assert_eq!(parse_trace_tp_size(None).unwrap(), None);
    }

    #[test]
    fn trace_tp_size_accepts_a_positive_rank_count() {
        assert_eq!(parse_trace_tp_size(Some("2")).unwrap(), Some(2));
    }

    #[test]
    fn trace_tp_size_rejects_zero_and_non_numeric_values() {
        for value in ["0", "two"] {
            let err = parse_trace_tp_size(Some(value)).unwrap_err();
            assert!(
                err.to_string()
                    .contains("MLXCEL_TRACE_TP_SIZE must be a positive integer")
            );
        }
    }

    #[test]
    fn scalar_reference_requires_one_text_batch_row() {
        validate_scalar_reference(true, Some(1), false).unwrap();

        for (rows, has_image_arm) in [(None, false), (Some(4), false), (Some(1), true)] {
            assert!(validate_scalar_reference(true, rows, has_image_arm).is_err());
        }
        validate_scalar_reference(false, Some(4), true).unwrap();
    }

    #[test]
    fn streamed_trace_rows_preserve_position_indices() {
        let first = mlxcel_core::from_slice_f32(&[0.0, 2.0, 1.0, -1.0], &[1, 1, 4]);
        let second = mlxcel_core::from_slice_f32(&[1.0, -1.0, 3.0, 0.0], &[1, 1, 4]);
        let mut output = Vec::new();

        emit_rows_to(&mut output, 3, 0, &first, 0, &[1], 2).unwrap();
        emit_rows_to(&mut output, 3, 1, &second, 0, &[2], 2).unwrap();

        let text = String::from_utf8(output).unwrap();
        let positions: Vec<&str> = text
            .lines()
            .map(|line| line.split('\t').nth(1).unwrap())
            .collect();
        assert_eq!(positions, ["0", "1"]);
    }

    #[test]
    fn batch_trace_row_zero_is_identical_for_one_and_four_rows() {
        let one = batched_trace_rows(100, 20, 4, 7, 1).unwrap();
        let four = batched_trace_rows(100, 20, 4, 7, 4).unwrap();
        assert_eq!(one[0], four[0]);
        assert_eq!(
            one[0],
            BatchedTraceRow {
                start: 20,
                prefix_from: 13,
            }
        );
    }

    #[test]
    fn batch_trace_rows_use_distinct_stream_offsets_and_prefixes() {
        let rows = batched_trace_rows(100, 20, 4, 7, 4).unwrap();
        assert_eq!(
            rows,
            vec![
                BatchedTraceRow {
                    start: 20,
                    prefix_from: 13,
                },
                BatchedTraceRow {
                    start: 25,
                    prefix_from: 17,
                },
                BatchedTraceRow {
                    start: 30,
                    prefix_from: 21,
                },
                BatchedTraceRow {
                    start: 35,
                    prefix_from: 25,
                },
            ]
        );
    }

    #[test]
    fn batch_trace_rows_reject_a_short_corpus() {
        let err = batched_trace_rows(24, 20, 4, 7, 1).unwrap_err();
        assert!(err.to_string().contains("need token 24, have 24"));
    }
}
