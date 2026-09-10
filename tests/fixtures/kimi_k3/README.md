# Kimi K3 tokenizer and XTML chat fixtures

Every file here was produced by the checkpoint's own reference implementation, not by mlxcel's Rust port, which is what makes them an oracle rather than a snapshot of the port's current behavior. `generate_fixtures.py` drives `transformers.AutoTokenizer.from_pretrained(..., trust_remote_code=True)`, which resolves `tokenization_kimi.TikTokenTokenizer` and `encoding_k3.build_chat_segments` out of the checkpoint directory itself.

## Files

| File | What it holds |
|------|---------------|
| `corpus.txt` | 1000 lines mixing English, Korean, Chinese, Japanese, code, emoji (ZWJ sequences and skin-tone modifiers included), long digit runs, CamelCase identifiers, contractions, full-width punctuation, URLs, literal `<\|open\|>` / `<\|sep\|>` / `[BOS]` spellings, `&quot;`, an embedded `\r`, leading and trailing spaces, and a few very long lines. |
| `corpus_ids.jsonl` | One compact JSON array per corpus line, from `tok.encode(line, allow_special_tokens=False)`. This is byte-identical to what `mlxcel inspect --tokenize` prints. |
| `corpus_whole_ids.json` | The whole file encoded as one document, which is the only way the pattern's newline alternatives get exercised. |
| `pretokenize_pins.json` | Per-string pre-tokenization pins: the exact piece split (from the `regex` module run over `tok.model._pat_str`), the ids of each piece, and the ids of the whole string. Pins the pattern's boundaries rather than only its final BPE output. |
| `control_tokens.json` | `vocab_size`, the control-block `base`, the BOS / EOS / PAD / UNK ids, and the name of all 256 control tokens. |
| `xtml/*.json` | One reference chat rendering each: `name`, `messages`, `tools`, `kwargs`, the `text` from `apply_chat_template(tokenize=False)` and the `ids` from `apply_chat_template(tokenize=True)`. A `kwargs` key that is absent takes the reference default, so `thinking_effort` absent means `max` while `thinking_effort: null` means "render no thinking-effort message". |

## Regenerating

The tokenizer directory is the tokenizer half of [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3): `config.json`, `generation_config.json`, `tokenizer_config.json`, `tiktoken.model`, and the `trust_remote_code` modules `tokenization_kimi.py` and `encoding_k3.py`. No weights are needed, and nothing here loads a model.

```bash
python3 -m venv /tmp/k3venv
/tmp/k3venv/bin/pip install tiktoken blobfile transformers regex
/tmp/k3venv/bin/python tests/fixtures/kimi_k3/generate_fixtures.py models/kimi-k3-tokenizer
```

The script is offline and deterministic: the corpus is built from `tests/fixtures/wikitext2_excerpt.txt`, `tests/fixtures/lang_bias_prompts_ko.txt` and sentence lists written into the script itself, so a rerun on any machine reproduces every file byte for byte. Verified against the checked-in set on 2026-09-10.

## Caveats

Float formatting is the one place the Rust port is allowed to differ. Python's `json.dumps` writes `1e+100` where serde_json writes `1e100`, and no fixture contains a float that reaches that form. Do not add one without deciding what the port should do first; the port keeps serde_json's output rather than emulating Python's `repr`.

Tests that read these files are gated on `models/kimi-k3-tokenizer` being present and skip loudly when it is not. Set `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` on a machine that owns the checkpoint to turn that skip into a failure.
