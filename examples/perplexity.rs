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
    // scored as a fresh sequence (fresh caches), which slightly pessimizes
    // absolute perplexity equally for every variant under comparison.
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

    let mut total_nll = 0.0f64;
    let mut total_tok = 0usize;

    for c in 0..n_chunks {
        let seg = &ids[c * chunk_tokens..(c + 1) * chunk_tokens + 1];
        let l = seg.len() as i32; // chunk_tokens + 1

        let input = from_slice_i32(&seg[..(l as usize - 1)], &[1, l - 1]);
        let mut caches = model.make_caches();
        let logits = model.forward(&input, &mut caches, None); // [1, L-1, V]
        let logits = astype(&logits, mlxcel_core::dtype::FLOAT32);
        let lp = log_softmax(&logits, -1);

        let targets = from_slice_i32(&seg[1..], &[1, l - 1, 1]);
        let tok_lp = take_along_axis(&lp, &targets, -1); // [1, L-1, 1]
        let nll = sum_all(&tok_lp);
        eval(&nll);
        let chunk_nll = -f64::from(item_f32(&nll));
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
