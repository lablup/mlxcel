//! Dump the OpenXLA/IREE engine's greedy argmax trajectory (no EOS stop) as an
//! oracle JSON `{"prompt_ids":[...], "ref_token_ids":[...]}`, at the ambient
//! `MLXCEL_XLA_PRECISION`.
//!
//! This lets the #515 token-exactness gate (`xla_oracle_check`) use the XLA
//! **f32** path itself as the reference, instead of the HF fp32 oracle from
//! `spike/openxla/oracle_continuation.py` (which needs a torch / transformers
//! venv). It is how issue #575 checks that the f16 path on Metal is token-exact
//! "versus the f32 path" on a host without the Python oracle:
//!
//! ```bash
//! eval "$(scripts/iree/setup-macos.sh --env)"
//! # 1) f32 reference trajectory (explicit f32 overrides the metal f16 default):
//! MLXCEL_XLA_PRECISION=f32 MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=metal \
//!   cargo run --release --features xla-iree --example xla_traj_dump -- \
//!   --model models/llama-3.2-1b-4bit --max-new 40 --out /tmp/f32_oracle.json
//! # 2) check the f16 path against it (token-exact gate reports first divergence):
//! MLXCEL_XLA_PRECISION=f16 MLXCEL_XLA_DEVICE=metal \
//!   cargo run --release --features xla-iree --example xla_oracle_check -- \
//!   --model models/llama-3.2-1b-4bit --oracle /tmp/f32_oracle.json --device metal
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

fn main() {
    let model = PathBuf::from(arg("--model", "models/llama-3.2-1b-4bit"));
    let device = {
        let a = arg("--device", "");
        if a.is_empty() {
            std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".to_string())
        } else {
            a
        }
    };
    let prompt = arg("--prompt", "The capital of France is");
    let n: usize = arg("--max-new", "40")
        .parse()
        .expect("--max-new is an integer");
    let out = PathBuf::from(arg("--out", "/tmp/xla_oracle.json"));

    // Raw prompt tokenization with special tokens (BOS), matching the HF oracle
    // (`oracle_continuation.py`: `tok(prompt).input_ids`), NOT the chat template.
    let tokenizer = tokenizers::Tokenizer::from_file(model.join("tokenizer.json"))
        .expect("load tokenizer.json");
    let enc = tokenizer
        .encode(prompt.as_str(), true)
        .expect("encode prompt");
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    assert!(!prompt_ids.is_empty(), "prompt tokenized to zero tokens");

    let mut eng = XlaReferenceEngine::load(&model, &device).expect("load reference engine");
    // Empty EOS set: exactly `n` argmax steps, no early stop, so the trajectory is
    // directly comparable position-for-position with another precision's run.
    let traj = eng.generate(&prompt_ids, n, &[]).expect("greedy generate");

    let json = serde_json::json!({
        "prompt_text": prompt,
        "prompt_ids": prompt_ids,
        "ref_token_ids": traj,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&json).unwrap()).expect("write oracle json");
    println!(
        "wrote {}: {} prompt tokens, {} ref tokens (device={device})",
        out.display(),
        prompt_ids.len(),
        traj.len(),
    );
}
