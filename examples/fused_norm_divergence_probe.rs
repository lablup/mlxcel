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

//! Where the graph path and the fused residual-add + RMSNorm part ways on a
//! Llama checkpoint (issue #3501).
//!
//! On GB10, `llama-3.2-1b-instruct-bf16` at B=1 on the 128-token synthesized
//! prompt generated different greedy text with `MLXCEL_FUSED_ADD_RMSNORM=1`
//! than without it, until #3510 made the CUDA kernel byte-identical to the
//! graph; since then the two arms' dumps compare equal and the op diff
//! reports no differing element, which makes this probe the end-to-end check
//! of that identity. The switch is process-global, so each arm is one process
//! of this probe, and the arms are compared offline:
//!
//! - **Steps.** The prompt (`synthesize_prompt_tokens`, as the checkpoint
//!   bench builds it) is prefilled in one forward that scores only its last
//!   position, as `mlxcel generate` does, and then width-one decode steps run
//!   over the caches `make_caches` builds, as the engine's pool does. With
//!   `--tokens FILE` every step is teacher-forced on that stream, so both arms
//!   stay on the same text after the flip; without it the probe decodes
//!   greedily and stops on EOS, which reproduces the CLI's token digest and so
//!   shows the probe runs the same computation. Per step it prints and writes
//!   to `OUT/steps.csv` the argmax, the top two ids and logits, and the
//!   top-1 minus top-2 margin.
//! - **Layers.** `--dump-layers STEP` walks the decoder layer by layer at that
//!   step (`0` is the prefill, whose tensors hold every prompt row) and writes each layer's input `x`,
//!   `attn_out`, the joined residual `h`, the MLP input `normed` and the block
//!   output, then the final norm and the logits, to `OUT/stepSTEP/`.
//! - **Offline.** `--compare A B` diffs two dumps in forward order and names the
//!   first tensor that is not byte-identical. `--op-diff DUMP` feeds one arm's
//!   dumped `x` and `attn_out` of every layer to both `graph_add_rms_norm` and
//!   the fused launcher in one process and diffs the outputs in ulps; with
//!   `--out` it also writes both sides' outputs for offline analysis.
//!
//! ```text
//! env -u MLXCEL_FUSED_ADD_RMSNORM gpu-lock run --tag issue-3501 -- \
//!   target/release/examples/fused_norm_divergence_probe \
//!   --model models/llama-3.2-1b-instruct-bf16 --tokens graph-tokens.txt \
//!   --dump-layers 11 --expect-route graph --out /tmp/probe/graph
//! MLXCEL_FUSED_ADD_RMSNORM=1 gpu-lock run ... --expect-route fused --out /tmp/probe/fused
//! target/release/examples/fused_norm_divergence_probe \
//!   --compare /tmp/probe/graph/step11 /tmp/probe/fused/step11
//! ```
//!
//! The route check reads the cuda-oxide adapter's counters, which exist only
//! in a build made with `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1`. Elsewhere the
//! probe names the route from `fused_add_rmsnorm_enabled()` and says so.

#[path = "fused_norm_divergence_probe/op_diff.rs"]
mod op_diff;
#[path = "fused_norm_divergence_probe/tensors.rs"]
mod tensors;

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use mlxcel::LoadedModel;
use mlxcel::models::Llama3Model;
use mlxcel::server::engine_probe::prompt::synthesize_prompt_tokens;
use mlxcel_core::fused_norm_route::fused_norm_route_snapshot;
use mlxcel_core::layers::KVCache;
use mlxcel_core::row_invariant::linear_whole_block;
use mlxcel_core::{MlxArray, UniquePtr};
use sha2::{Digest, Sha256};

use tensors::DumpWriter;

/// Per-step margins and per-layer dumps for the fused-norm divergence.
#[derive(Parser, Debug)]
#[command(name = "fused_norm_divergence_probe")]
struct Args {
    /// Path to the model directory (a Llama-family checkpoint).
    #[arg(short = 'm', long)]
    model: Option<PathBuf>,

    /// Synthesized prompt length in tokens.
    #[arg(long, default_value_t = 128, value_parser = parse_nonzero)]
    prompt_tokens: usize,

    /// Teacher-force the decode on these whitespace-separated token ids.
    /// Without it the probe decodes greedily and stops on EOS.
    #[arg(long, value_name = "FILE")]
    tokens: Option<PathBuf>,

    /// Generated tokens (steps) to run.
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: usize,

    /// Dump every layer's tensors at this generated index (0 is the prefill).
    #[arg(long, value_name = "STEP")]
    dump_layers: Option<usize>,

    /// Output directory for `steps.csv` and dumps; keep it outside the checkout.
    #[arg(long, value_name = "DIR")]
    out: Option<PathBuf>,

    /// The route this process must have taken: `graph`, `fused` or `oxide`.
    #[arg(long, value_parser = ["graph", "fused", "oxide"])]
    expect_route: Option<String>,

    /// Diff two dump directories and exit.
    #[arg(long, num_args = 2, value_names = ["A", "B"])]
    compare: Option<Vec<PathBuf>>,

    /// Feed this dump's per-layer inputs to both join implementations and exit.
    #[arg(long, value_name = "DUMP")]
    op_diff: Option<PathBuf>,
}

/// A prompt length of at least one token.
fn parse_nonzero(raw: &str) -> Result<usize, String> {
    match raw.parse::<usize>() {
        Ok(0) => Err("an empty prompt has no last position to score".to_string()),
        Ok(n) => Ok(n),
        Err(e) => Err(e.to_string()),
    }
}

/// Top two of a logits row, by value, ties to the lower id (MLX's argmax).
struct Top2 {
    first: (usize, f32),
    second: (usize, f32),
}

fn top2(row: &[f32]) -> Top2 {
    let mut first = (0usize, f32::NEG_INFINITY);
    let mut second = (0usize, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > first.1 {
            second = first;
            first = (i, v);
        } else if v > second.1 {
            second = (i, v);
        }
    }
    Top2 { first, second }
}

fn logits_row(logits: &MlxArray) -> (Vec<f32>, String) {
    let flat = mlxcel_core::astype(logits, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&flat);
    let raw = mlxcel_core::array_to_raw_bytes(logits);
    let digest = Sha256::digest(&raw);
    let sha = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    let vals = mlxcel_core::array_to_raw_bytes(&flat)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (vals, sha)
}

/// The same digest `oxide_fused_norm_checkpoint_bench` prints, so a free run
/// can be matched against its `tokens_sha256_16`.
fn tokens_digest(row: &[i32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((row.len() as u64).to_le_bytes());
    for id in row {
        hasher.update(id.to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The route this process took, and where that answer came from.
fn observed_route() -> (&'static str, &'static str) {
    match fused_norm_route_snapshot() {
        Some(s) => (s.route(), "adapter counters"),
        None if mlxcel_core::layers::fused_add_rmsnorm_enabled() => ("fused", "switch only"),
        None => ("graph", "switch only"),
    }
}

/// One forward through the layers by hand, writing each tensor. The sequence
/// of calls is `TransformerBlock::forward`'s, so the logits are the ones
/// `Llama3Model::forward` returns for the same input (or, with `last_pos`, the
/// row `Llama3Model::last_logits` returns, as the prefill scores it).
fn walk_and_dump(
    model: &Llama3Model,
    input: &MlxArray,
    caches: &mut [KVCache],
    last_pos: Option<usize>,
    dir: &Path,
) -> Result<UniquePtr<MlxArray>> {
    let mut dump = DumpWriter::create(dir)?;
    let mut h = model.embed_tokens.forward(input);
    for (i, layer) in model.layers.iter().enumerate() {
        let pre = layer.input_layernorm.forward(&h);
        let attn_out = layer.self_attn.forward(&pre, &mut caches[i], None);
        let (normed, joined) =
            mlxcel_core::layers::fused_add_rms_norm(&layer.post_attention_layernorm, &attn_out, &h);
        let ff = layer.mlp.forward(&normed);
        let out = mlxcel_core::add(&joined, &ff);
        dump.write(&format!("L{i:02}_x"), &h)?;
        dump.write(&format!("L{i:02}_attn_out"), &attn_out)?;
        dump.write(&format!("L{i:02}_h"), &joined)?;
        dump.write(&format!("L{i:02}_normed"), &normed)?;
        dump.write(&format!("L{i:02}_out"), &out)?;
        h = out;
    }
    let final_norm = model.norm.forward(&h);
    let logits = match last_pos {
        Some(pos) => {
            let row = mlxcel_core::generate::logits_at_position(&final_norm, pos);
            linear_whole_block(&model.lm_head, &row)
        }
        None => linear_whole_block(&model.lm_head, &final_norm),
    };
    dump.write("final_norm", &final_norm)?;
    dump.write("logits", &logits)?;
    Ok(logits)
}

fn read_tokens(path: &Path) -> Result<Vec<i32>> {
    std::fs::read_to_string(path)
        .with_context(|| format!("cannot read {}", path.display()))?
        .split_whitespace()
        .map(|t| {
            t.parse::<i32>()
                .with_context(|| format!("bad token id {t:?}"))
        })
        .collect()
}

fn load_llama(model_dir: &Path) -> Result<Llama3Model> {
    let (model, _) = mlxcel::load_model(model_dir).context("failed to load model")?;
    match model {
        LoadedModel::Llama(m) => Ok(m),
        _ => bail!("this probe walks Llama3Model layers; the checkpoint loaded as another family"),
    }
}

fn run_steps(args: &Args, model_dir: &Path, out: &Path) -> Result<()> {
    let tokenizer = mlxcel::tokenizer::load_tokenizer(model_dir)
        .with_context(|| format!("failed to load tokenizer from {}", model_dir.display()))?;
    let prompt = synthesize_prompt_tokens(&tokenizer, args.prompt_tokens)?;
    let forced = args.tokens.as_deref().map(read_tokens).transpose()?;
    // Refuse before loading when the switch already contradicts the arm.
    let switch_on = mlxcel_core::layers::fused_add_rmsnorm_enabled();
    if let Some(expected) = args.expect_route.as_deref()
        && (expected == "graph") == switch_on
    {
        bail!("--expect-route {expected} but MLXCEL_FUSED_ADD_RMSNORM resolves to {switch_on}");
    }
    let model = load_llama(model_dir)?;
    let eos = mlxcel::read_eos_token_ids(model_dir);
    std::fs::create_dir_all(out)?;
    let mut csv = std::fs::File::create(out.join("steps.csv"))?;
    writeln!(
        csv,
        "step,position,input_token,argmax,top1_id,top1_logit,top2_id,top2_logit,margin,next_token,logits_sha16"
    )?;

    let before = fused_norm_route_snapshot();
    let mut caches = model.make_caches();
    let ids = mlxcel_core::from_slice_i32(&prompt, &[1, prompt.len() as i32]);
    let last_pos = prompt.len() - 1;
    let mut logits = if args.dump_layers == Some(0) {
        walk_and_dump(
            &model,
            &ids,
            &mut caches,
            Some(last_pos),
            &out.join("step0"),
        )?
    } else {
        model.last_logits(&ids, None, &mut caches, None, last_pos)
    };
    let mut generated = Vec::new();
    let mut input_token = *prompt.last().context("empty prompt")?;
    for step in 0..args.max_tokens {
        let (row, sha) = logits_row(&logits);
        if step == 0
            && let Some(stream) = &forced
            && let Some(bad) = stream
                .iter()
                .find(|&&id| id < 0 || id as usize >= row.len())
        {
            bail!(
                "--tokens holds id {bad}, outside the vocabulary of {} logits",
                row.len()
            );
        }
        let t = top2(&row);
        let argmax = t.first.0 as i32;
        let next = match &forced {
            Some(stream) => match stream.get(step) {
                Some(&id) => id,
                None => break,
            },
            None => argmax,
        };
        let position = prompt.len() + step - 1;
        let margin = t.first.1 - t.second.1;
        println!(
            "[probe] step {step:3} pos {position} input {input_token} argmax {argmax} top1 {}:{:.4} top2 {}:{:.4} margin {margin:.4} next {next}",
            t.first.0, t.first.1, t.second.0, t.second.1
        );
        writeln!(
            csv,
            "{step},{position},{input_token},{argmax},{},{},{},{},{margin},{next},{sha}",
            t.first.0, t.first.1, t.second.0, t.second.1
        )?;
        if forced.is_none() && eos.contains(&next) {
            break;
        }
        generated.push(next);
        if step + 1 == args.max_tokens {
            break;
        }
        let input = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
        logits = if args.dump_layers == Some(step + 1) {
            let dir = out.join(format!("step{}", step + 1));
            walk_and_dump(&model, &input, &mut caches, None, &dir)?
        } else {
            model.forward(&input, &mut caches, None)
        };
        input_token = next;
    }
    let (route, source) = observed_route();
    let routed = before.zip(fused_norm_route_snapshot()).map_or_else(
        || "n/a".to_string(),
        |(b, a)| {
            format!(
                "native {} oxide {}",
                a.native_routes - b.native_routes,
                a.oxide_routes - b.oxide_routes
            )
        },
    );
    println!(
        "[probe] mode {} generated {} tokens_sha256_16 {} route {route} ({source}; joins {routed})",
        if forced.is_some() {
            "teacher-forced"
        } else {
            "free-run"
        },
        generated.len(),
        tokens_digest(&generated)
    );
    if let Some(expected) = &args.expect_route
        && expected != route
    {
        bail!("expected route {expected}, this process took {route} ({source})");
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(dirs) = &args.compare {
        if args.expect_route.is_some() {
            bail!("--expect-route does not apply to --compare, which runs no model");
        }
        return tensors::compare_dumps(&dirs[0], &dirs[1]);
    }
    let model_dir = args.model.clone().context("--model is required")?;
    mlxcel_core::hardware::apply_cuda_graph_cache_default();
    mlxcel_core::hardware::apply_cuda_sdpa_cache_default();
    mlxcel_core::hardware::apply_cuda_graph_budget_default(Some(&model_dir));
    let _runtime = mlxcel::initialize_runtime();
    if let Some(dump) = &args.op_diff {
        let model = load_llama(&model_dir)?;
        let result = op_diff::run(&model, dump, args.out.as_deref());
        // The op diff calls both joins directly, so this names the process's
        // switch (or counters), which the diff itself does not depend on.
        let (route, source) = observed_route();
        println!("[op-diff] route {route} ({source})");
        result?;
        if let Some(expected) = &args.expect_route
            && expected != route
        {
            bail!("expected route {expected}, this process took {route} ({source})");
        }
        return Ok(());
    }
    let out = args
        .out
        .clone()
        .context("--out is required for a step run")?;
    run_steps(&args, &model_dir, &out)
}
