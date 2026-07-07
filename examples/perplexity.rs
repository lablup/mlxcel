// Teacher-forced perplexity harness for quantization quality gates.
//
// Motivated by issue #683: requantizing gemma-4-12b's 8-bit MLP modules to
// 4-bit measured 1.6x decode throughput, but the shipped checkpoint keeps
// the MLP at 8-bit presumably for quality headroom, so variants need a
// quality gate beyond greedy smoke prompts. This harness computes exact
// next-token negative log-likelihood over a fixed text file, chunk by
// chunk with fresh KV caches, so different quantization variants of the
// same model are directly comparable (lower perplexity is better; the
// ABSOLUTE value depends on the corpus and is not comparable across
// tokenizers).
//
// Scoring mode: each chunk is scored as an INDEPENDENT fresh sequence
// (BOS anchor + fresh caches + reset model-owned state). This is not a
// continuous-context corpus perplexity; a mid-corpus window carries no real
// preceding context, so its absolute NLL is pessimistic. That penalty is
// applied equally to every variant, so cross-variant comparison stays fair,
// but the single cleanest signal is `MAX_CHUNKS=1`: it scores exactly one
// BOS-anchored forward over the first `CHUNK_TOKENS` positions, which is the
// reference-clean measurement used to characterize the gemma-4 long-position
// forward behavior in issue #686. Sweep `CHUNK_TOKENS` in {128,256,512,1024}
// at `MAX_CHUNKS=1` to read the NLL-vs-length curve directly.
//
// Usage: cargo run --release --features cuda --example perplexity -- \
//          MODEL_DIR TEXT_FILE [CHUNK_TOKENS=512] [MAX_CHUNKS=32]

use anyhow::{Context, Result};
use mlxcel::LanguageModel;
use mlxcel::tokenizer::load_tokenizer;
use mlxcel_core::{astype, eval, from_slice_i32, item_f32, log_softmax, sum_all, take_along_axis};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args
        .get(1)
        .context("usage: perplexity MODEL_DIR TEXT_FILE")?;
    let text_file = args
        .get(2)
        .context("usage: perplexity MODEL_DIR TEXT_FILE")?;
    let chunk_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
    let max_chunks: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(32);

    let text = std::fs::read_to_string(text_file).context("failed to read text file")?;

    let model_path = std::path::Path::new(model_dir);
    let (model, loaded_tokenizer) =
        mlxcel::load_model(model_path).context("failed to load model")?;
    let tokenizer = load_tokenizer(model_path).unwrap_or(loaded_tokenizer);

    // Encode once WITH special tokens so the first chunk starts from BOS,
    // then split into fixed-size windows. Every chunk after the first is
    // scored as a fresh sequence, which slightly pessimizes absolute
    // perplexity equally for every variant under comparison.
    let ids: Vec<i32> = tokenizer
        .encode(&text, true)
        .context("tokenize failed")?
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let n_chunks = ((ids.len() - 1) / chunk_tokens).min(max_chunks);
    anyhow::ensure!(n_chunks > 0, "text too short: {} tokens", ids.len());
    println!(
        "model={model_dir} corpus={} tokens, scoring {n_chunks} chunks x {chunk_tokens}",
        ids.len()
    );

    // BOS anchor for chunks after the first (issue #686). Gemma-family
    // quality collapses on BOS-less windows (measured on this harness's
    // corpus: gemma-3-4b ~+3.6 nats/token, gemma-4-12b ~+6.6 nats/token),
    // so mid-corpus chunks are scored behind the same single BOS the model
    // saw in training. Tokenizers that add no BOS produce an empty probe
    // and keep the historical raw-window behavior. The BOS position's own
    // logit row is excluded from scoring, so per-chunk token counts are
    // unchanged.
    let bos_prefix: Vec<i32> = tokenizer
        .encode("", true)
        .map(|ids| ids.into_iter().take(1).map(|t| t as i32).collect())
        .unwrap_or_default();

    let mut total_nll = 0.0f64;
    let mut total_tok = 0usize;

    for c in 0..n_chunks {
        let seg = &ids[c * chunk_tokens..(c + 1) * chunk_tokens + 1];
        let l = seg.len() as i32; // chunk_tokens + 1

        let (input_ids, target_offset) = if c == 0 || bos_prefix.is_empty() {
            (seg[..(l as usize - 1)].to_vec(), 0i32)
        } else {
            let mut with_bos = bos_prefix.clone();
            with_bos.extend_from_slice(&seg[..(l as usize - 1)]);
            (with_bos, bos_prefix.len() as i32)
        };
        let input_len = input_ids.len() as i32;
        let input = from_slice_i32(&input_ids, &[1, input_len]);

        // Fresh sequence per chunk: models that key their KV state on a
        // model-owned fallback slot (gemma3/gemma4/llama4/qwen3.5, issue
        // #686) ignore the external `caches` argument, so `make_caches()`
        // alone silently CONTINUED the previous chunk's context and made
        // every chunk after the first score with leaked history. Reset the
        // internal slot explicitly; external-cache models are a no-op.
        model.reset_runtime_state();
        let mut caches = model.make_caches();
        let logits = model.forward(&input, &mut caches, None); // [1, input_len, V]
        let shape = mlxcel_core::array_shape(&logits);
        if c == 0 {
            println!("  logits shape {:?} (expect [1, {}, vocab])", shape, l - 1);
        }
        anyhow::ensure!(
            shape.len() == 3 && shape[1] == input_len,
            "model.forward returned {:?}, not per-position logits; cannot score",
            shape
        );
        // Row j predicts input[j + 1]; targets seg[1..] live at rows
        // [target_offset, target_offset + l - 1).
        let logits = mlxcel_core::slice(
            &logits,
            &[0, target_offset, 0],
            &[1, target_offset + l - 1, shape[2]],
        );
        let logits = astype(&logits, mlxcel_core::dtype::FLOAT32);
        let lp = log_softmax(&logits, -1);

        let targets = from_slice_i32(&seg[1..], &[1, l - 1, 1]);
        let tok_lp = take_along_axis(&lp, &targets, -1); // [1, L-1, 1]
        let nll = sum_all(&tok_lp);
        eval(&nll);
        let chunk_nll = -f64::from(item_f32(&nll));
        // A non-finite chunk NLL (an all-masked softmax row, an overflow in a
        // degenerate window, or GPU-pool exhaustion after many fresh-cache
        // iterations) must abort loudly rather than silently poison the mean.
        anyhow::ensure!(
            chunk_nll.is_finite(),
            "chunk {c} produced a non-finite NLL ({chunk_nll}); refusing to \
             average it into the perplexity. Re-run with MAX_CHUNKS=1 for the \
             reference-clean single-forward measurement.",
        );
        total_nll += chunk_nll;
        total_tok += l as usize - 1;

        // Free the chunk's transients before the next fresh-cache pass.
        drop(caches);
        mlxcel_core::clear_memory_cache();

        println!(
            "  chunk {:>3}: nll/tok {:.4}  (running ppl {:.3})",
            c,
            chunk_nll / (l as f64 - 1.0),
            (total_nll / total_tok as f64).exp()
        );
    }

    println!();
    println!(
        "tokens={} mean_nll={:.5} perplexity={:.4}",
        total_tok,
        total_nll / total_tok as f64,
        (total_nll / total_tok as f64).exp()
    );
    Ok(())
}
