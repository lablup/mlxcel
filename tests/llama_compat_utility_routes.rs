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

//! Real-tokenizer coverage for the b10621 utility routes (issue #1442).
//!
//! The unit tests behind `/tokenize` and `/infill` build synthetic
//! vocabularies, which is what makes their assertions exact but also what makes
//! them blind to how a shipped checkpoint is actually spelled. Three of the
//! four families here would have passed those tests while failing in
//! production: Llama 3 has 256 reserved special tokens and a ByteLevel BPE
//! whose tokens straddle characters, Qwen carries the FIM markers on an
//! ordinary instruct checkpoint, and DeepSeek-Coder writes its markers with the
//! FULLWIDTH vertical line U+FF5C rather than ASCII `|`.
//!
//! Only the tokenizer is loaded, never the weights, so this runs in under a
//! second and needs no GPU. Each case skips with a message when its checkpoint
//! is absent, so CI (which has no checkpoints) stays green.
//!
//! ```bash
//! cargo test --test llama_compat_utility_routes --profile test-fast --features metal,accelerate -- --nocapture
//! ```

mod common;

use common::repo_model_dir;

use mlxcel::tokenizer::{MlxcelTokenizer, load_tokenizer, pieces};

const LLAMA: &str = "mlx/llama-3.2-1b-instruct";
const QWEN: &str = "mlx/qwen2.5-1.5b-instruct-4bit";
const QWEN3: &str = "mlx/qwen3-0.6b-4bit";
const DEEPSEEK_CODER: &str = "mlx/deepseek-coder-1.3b-4bit";

/// Load a checkpoint's tokenizer, or `None` with a skip message.
fn tokenizer_for(name: &str) -> Option<MlxcelTokenizer> {
    let dir = repo_model_dir(name);
    if !dir.join("config.json").exists() {
        eprintln!("Skipping {name}: checkpoint not found at {}", dir.display());
        return None;
    }
    Some(load_tokenizer(&dir).unwrap_or_else(|e| panic!("{name}: tokenizer must load: {e:#}")))
}

/// The `with_pieces` reassembly property, on a real vocabulary: concatenating
/// every token's piece bytes must reproduce the input exactly, whether an
/// individual piece was valid UTF-8 or a fragment of a character.
fn assert_pieces_reassemble(name: &str, tokenizer: &MlxcelTokenizer) {
    for content in [
        "Hello, world!",
        "\u{c548}\u{b155}\u{d558}\u{c138}\u{c694}",
        "\u{4E2D}\u{6587}\u{6D4B}\u{8BD5}",
        "caf\u{e9} na\u{ef}ve \u{1F600}\u{1F680}",
        "fn main() {\n    println!(\"hi\");\n}\n",
        "",
    ] {
        let ids = tokenizer
            .encode_with_special(content, false, true)
            .unwrap_or_else(|e| panic!("{name}: {content:?} must tokenize: {e:#}"));
        let mut bytes = Vec::new();
        for id in &ids {
            let piece = tokenizer
                .token_piece_bytes(*id)
                .unwrap_or_else(|| panic!("{name}: token {id} has no piece"));
            // The wire shape is derived from the same bytes, so exercise it.
            match pieces::piece_json(piece.clone()) {
                serde_json::Value::String(text) => assert_eq!(text.as_bytes(), piece.as_slice()),
                serde_json::Value::Array(values) => assert_eq!(values.len(), piece.len()),
                other => panic!("{name}: unexpected piece shape {other}"),
            }
            bytes.extend(piece);
        }
        let rebuilt = String::from_utf8(bytes)
            .unwrap_or_else(|e| panic!("{name}: pieces of {content:?} are not UTF-8: {e}"));
        assert_eq!(
            rebuilt, content,
            "{name}: pieces of {content:?} must reassemble to the input"
        );
    }
}

/// `/detokenize`'s round trip on a real vocabulary.
fn assert_detokenize_round_trips(name: &str, tokenizer: &MlxcelTokenizer) {
    for content in [
        "Hello, world!",
        "\u{c548}\u{b155}\u{d558}\u{c138}\u{c694}",
        "caf\u{e9} \u{1F600}",
        "",
    ] {
        let ids = tokenizer
            .encode_with_special(content, false, true)
            .expect("tokenizes");
        let decoded = tokenizer.decode(&ids, false).expect("decodes");
        assert_eq!(decoded, content, "{name}: {content:?} must round trip");
    }
}

#[test]
fn llama_tokenizer_serves_the_tokenize_schema() {
    let Some(tokenizer) = tokenizer_for(LLAMA) else {
        return;
    };
    assert_pieces_reassemble(LLAMA, &tokenizer);
    assert_detokenize_round_trips(LLAMA, &tokenizer);

    // `parse_special` is the switch the schema exposes, and Llama 3's chat
    // markers are exactly the tokens it governs.
    let parsed = tokenizer
        .encode_with_special("<|begin_of_text|>hi", false, true)
        .expect("tokenizes");
    let plain = tokenizer
        .encode_with_special("<|begin_of_text|>hi", false, false)
        .expect("tokenizes");
    let bos = tokenizer
        .token_to_id("<|begin_of_text|>")
        .expect("Llama 3 declares <|begin_of_text|>");
    assert!(
        parsed.contains(&bos),
        "parse_special:true must resolve the marker"
    );
    assert!(
        !plain.contains(&bos),
        "parse_special:false must leave the marker as ordinary text"
    );
    assert!(plain.len() > parsed.len(), "{plain:?} vs {parsed:?}");

    // A special token's piece is its own spelling, which is what `/detokenize`
    // renders and what `with_pieces` reports.
    assert_eq!(
        tokenizer.token_piece_bytes(bos).as_deref(),
        Some("<|begin_of_text|>".as_bytes())
    );

    // Llama 3 is a chat model with no FIM markers, so `/infill` must refuse it.
    let fim = tokenizer.fim_tokens();
    assert!(!fim.supports_infill(), "Llama 3.2 declares no FIM markers");
    assert_eq!(
        fim.require_triple().expect_err("no FIM triple"),
        "Infill is not supported by this model: prefix token is missing. suffix token is \
         missing. middle token is missing. "
    );
}

#[test]
fn qwen_tokenizer_serves_the_tokenize_schema_and_declares_fim() {
    for name in [QWEN, QWEN3] {
        let Some(tokenizer) = tokenizer_for(name) else {
            continue;
        };
        assert_pieces_reassemble(name, &tokenizer);
        assert_detokenize_round_trips(name, &tokenizer);

        let parsed = tokenizer
            .encode_with_special("<|im_start|>user", false, true)
            .expect("tokenizes");
        let plain = tokenizer
            .encode_with_special("<|im_start|>user", false, false)
            .expect("tokenizes");
        let im_start = tokenizer
            .token_to_id("<|im_start|>")
            .expect("Qwen declares <|im_start|>");
        assert!(parsed.contains(&im_start), "{name}");
        assert!(!plain.contains(&im_start), "{name}");

        // Qwen ships the FIM markers on its ordinary instruct checkpoints, so
        // `/infill` is served there and resolves the `<|fim_*|>` spellings.
        let fim = tokenizer.fim_tokens();
        let triple = fim
            .require_triple()
            .unwrap_or_else(|e| panic!("{name}: Qwen declares FIM markers, got {e}"));
        assert_eq!(triple.pre.spelling, "<|fim_prefix|>", "{name}");
        assert_eq!(triple.suf.spelling, "<|fim_suffix|>", "{name}");
        assert_eq!(triple.mid.spelling, "<|fim_middle|>", "{name}");
        assert_eq!(
            fim.rep.as_ref().map(|t| t.spelling),
            Some("<|repo_name|>"),
            "{name}"
        );
        assert_eq!(
            fim.sep.as_ref().map(|t| t.spelling),
            Some("<|file_sep|>"),
            "{name}"
        );

        // The assembled prompt must re-tokenize with the markers as their own
        // single ids, which is the assumption the string-based assembly rests
        // on. Ordering is asserted by id, not by substring.
        let inputs = mlxcel::server::infill::parse_infill_inputs(&serde_json::json!({
            "input_prefix": "def add(a, b):\n    return ",
            "input_suffix": "\n\nprint(add(1, 2))\n"
        }))
        .expect("inputs parse");
        for spm in [false, true] {
            let prompt = mlxcel::server::infill::format_infill_prompt(&fim, &triple, &inputs, spm);
            let ids = tokenizer
                .encode_with_special(&prompt, false, true)
                .expect("prompt tokenizes");
            assert_eq!(
                *ids.last().expect("a non-empty prompt"),
                triple.mid.id,
                "{name}: the middle marker must be last (spm={spm})"
            );
            let pre_at = ids
                .iter()
                .position(|id| *id == triple.pre.id)
                .expect("prefix marker");
            let suf_at = ids
                .iter()
                .position(|id| *id == triple.suf.id)
                .expect("suffix marker");
            if spm {
                assert!(
                    suf_at < pre_at,
                    "{name}: --spm-infill puts the suffix first"
                );
            } else {
                assert!(pre_at < suf_at, "{name}: the default puts the prefix first");
            }
        }
    }
}

#[test]
fn deepseek_coder_resolves_its_fullwidth_fim_markers() {
    let Some(tokenizer) = tokenizer_for(DEEPSEEK_CODER) else {
        return;
    };
    assert_pieces_reassemble(DEEPSEEK_CODER, &tokenizer);
    assert_detokenize_round_trips(DEEPSEEK_CODER, &tokenizer);

    // The whole point of this case: the published checkpoint spells its markers
    // with U+FF5C, so an implementation that scanned for the ASCII pipe would
    // refuse `/infill` on a model that supports it.
    let fim = tokenizer.fim_tokens();
    let triple = fim
        .require_triple()
        .unwrap_or_else(|e| panic!("DeepSeek-Coder declares FIM markers, got {e}"));
    assert_eq!(triple.pre.spelling, "<\u{FF5C}fim\u{2581}begin\u{FF5C}>");
    assert_eq!(triple.suf.spelling, "<\u{FF5C}fim\u{2581}hole\u{FF5C}>");
    assert_eq!(triple.mid.spelling, "<\u{FF5C}fim\u{2581}end\u{FF5C}>");

    let inputs = mlxcel::server::infill::parse_infill_inputs(&serde_json::json!({
        "input_prefix": "def add(a, b):\n    return ",
        "input_suffix": "\n"
    }))
    .expect("inputs parse");
    let prompt = mlxcel::server::infill::format_infill_prompt(&fim, &triple, &inputs, false);
    let ids = tokenizer
        .encode_with_special(&prompt, false, true)
        .expect("prompt tokenizes");
    assert_eq!(ids.first().copied(), Some(triple.pre.id));
    assert_eq!(ids.last().copied(), Some(triple.mid.id));
    assert!(ids.contains(&triple.suf.id));
}
