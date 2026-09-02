# Technical Report: PR #1594 - feat(models): add the IQuest-Coder (iquestcoder) route

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle, plus a five-prompt greedy comparison against an mlx-lm 0.31.3 oracle on the same weights
**Status**: Completed (unit coverage green; the real-checkpoint numbers in Appendix A and B were measured on `models/mlx/iquest-coder-v1-7b-instruct-8bit` and are reproducible from the commands there)
**Languages**: Rust
**Risk Level**: Low for the model route (purely additive: one previously rejected `model_type` gains a route, no existing arm is touched). Medium for the tokenizer change, which is shared: it is scoped to checkpoints that write `"add_prefix_space": false` in `tokenizer_config.json`, and that scoping is what the risk rests on.

---

## Executive Summary

IQuest-Coder V1 declares `model_type: "iquestcoder"` and is the shared `llama3` decoder under a different label, so mlxcel rejected it at detection for no architectural reason. PR #1594 adds `ModelType::IQuestCoder` on top of the existing Llama loader with no new architecture code, and refuses the two config switches (`clip_qkv`, sliding-window attention) that would stop the equivalence holding.

The more useful half of this report is the tokenizer. The measured divergence that motivated the issue was not in the decoder at all: it was that mlxcel's SentencePiece fallback ignored `"add_prefix_space": false`, so every prompt was tokenized as if it began with a space. Fixing that required rewriting one field of the SentencePiece `ModelProto` before load, because the `sentencepiece` crate has no normalizer setter. Along the way the investigation found that the reference tokenizer this checkpoint's oracle was built from is itself lossy for BPE SentencePiece models, which changes what "matches the oracle" can even mean for this family.

---

## 1. Problem Statement

### 1.1 Background

`mlxcel generate -m models/mlx/iquest-coder-v1-7b-instruct-8bit` failed with `Error: Unsupported model type: iquestcoder`. `src/models/detection.rs` mapped `"llama" | "mistral"` to `ModelType::Llama` and everything unrecognized fell through to the catch-all error arm.

The family (7B / 14B / 40B, Base / Instruct / Thinking) is a plain Llama decoder: RMSNorm, GQA with an explicit `head_dim`, SwiGLU, an untied `lm_head`, RoPE at base 500000 with no scaling, and no attention or MLP bias. The 7B Instruct config is `hidden_size` 5120, `head_dim` 128, 40 attention heads over 8 KV heads, `intermediate_size` 27648, 14 layers, `vocab_size` 76800. `llama3::ModelArgs` already parses all of it, including the optional `head_dim`.

### 1.2 The keys that are not Llama's

The config carries three keys Llama's does not: `clip_qkv`, `use_sliding_window` with `sliding_window`, and `max_window_layers`. All are inert in every published checkpoint, and the equivalence with the shared decoder holds only while they stay inert. The vendor decoder applies the QKV clamp whenever `clip_qkv` is not null, and windows a layer when `use_sliding_window` is set, `sliding_window` is non-null, and the layer index has reached `max_window_layers`. Neither behavior exists in mlxcel's Llama attention, and neither would produce a load error or an obviously broken output: a clamped model decodes fluently with wrong attention scores, and a windowed model decodes fluently with the wrong receptive field.

### 1.3 The tokenizer, which is where the real defect was

This family ships a SentencePiece `tokenizer.model` with no `tokenizer.json`, so mlxcel takes the SentencePiece fallback path in `src/tokenizer/mod.rs`. Its `tokenizer_config.json` sets `"add_prefix_space": false`. mlxcel never read that key.

A SentencePiece model normalizes with `add_dummy_prefix` on by default, which prepends a space to the input before escaping, so `encode("The Fibonacci ...")` produces `▁The` rather than `The`. On a raw prompt that is one wrong token at position zero. After a chat template it is worse: mlxcel splits the rendered text at special-token boundaries and encodes each segment separately, so every segment between `<|im_start|>` and `<|im_end|>` acquired a phantom leading space, and the model never saw that during training.

### 1.4 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Checkpoint stays unloadable | Medium (family is unreachable, workaround is to hand-edit `config.json`) | Certain before this change |
| A future checkpoint enables `clip_qkv` or the sliding window and loads anyway | High (fluent output, silently wrong attention) | Low, but undetectable without the guard |
| Phantom leading space on every SentencePiece checkpoint that disables it | Medium (measurably worse likelihood, divergent greedy decode) | Certain for any such checkpoint |
| Scoping the tokenizer fix too widely and moving unrelated checkpoints | High (a shared tokenizer change touches every SentencePiece family) | Mitigated by requiring an explicit `false` |

---

## 2. Technical Review

### 2.1 Security

The tokenizer change adds a protobuf reader that runs over `tokenizer.model`, a file that comes from whatever checkpoint the operator downloaded, so it is attacker-influenced input. `src/tokenizer/spm_proto.rs` is written accordingly: every length and every advance is checked (`checked_add` plus a bound against the buffer length), a varint wider than 64 bits is rejected rather than shifted past the register, field number zero and the deprecated group wire types are rejected rather than walked, and the module never indexes without a preceding bounds check. A malformed model returns an `Err`, and the caller logs one warning and loads the file unmodified rather than failing the whole model. Two tests cover the malformed cases (a truncated message and a length prefix that runs past the end of the buffer).

**Issues Found:** none open.

### 2.2 Performance

The rewrite happens once per model load and copies a `tokenizer.model` (1.28 MB here) twice: once to read, once to emit. Nothing in the decode path changed. The tokenization itself is unchanged in kind; the model is loaded through `from_serialized_proto` instead of `open`, which is the same C++ loader over a buffer instead of a path.

Worth recording for a different reason: the override lowers the model's own teacher-forced negative log-likelihood on every text tried, which is the evidence that it is correct rather than merely different.

| Text | Before (`add_dummy_prefix` on) | After (off) | Reference fast tokenizer |
|------|-------------------------------|-------------|--------------------------|
| Prose, 203 chars | 75.44 nats | **73.44** | 79.54 |
| Technical prose, 171 chars | 52.69 nats | **51.96** | 51.96 |
| Python source, 123 chars | 31.45 nats | **27.25** | 27.25 |

### 2.3 Compatibility and Dependencies

- **Breaking changes**: none. No existing `model_type` arm changed, and the tokenizer override fires only on an explicit `"add_prefix_space": false`.
- **New dependencies**: none. `SentencePieceProcessor::from_serialized_proto` is already in the pinned `sentencepiece` 0.13.2.
- **Compatibility**: `ModelType` gains a variant, which is a source-breaking change for an exhaustive external match, but `ModelType` is not part of a stability contract and the compiler located every internal site.

### 2.4 Code Quality

- **Test coverage**: 16 new tests. Eight cover the protobuf rewrite on synthetic messages, two read the real `tokenizer.model` (and skip through the shared pinned-checkpoint gate when it is absent), six cover detection including the two refusals and the two inert-key cases, one covers `add_prefix_space` parsing, and one asserts the family appears in `mlxcel arch`.
- **Complexity**: one new 237-line module with a single public function.
- **Technical debt**: `parse_special_tokens` returning a growing tuple was replaced by a named struct, which is a small reduction.

---

## 3. Technical Decisions

### 3.1 A dedicated `ModelType` rather than a third spelling on the `llama` arm

**Context:** The issue's own plan was `"llama" | "mistral" | "iquestcoder" => Ok(ModelType::Llama)`, which is the minimal change and is what several other aliases do.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Alias onto `ModelType::Llama` | One line; inherits every Llama capability automatically | Invisible in `mlxcel arch`, which renders `ModelType` and not the alias strings; the config refusals would have to live in the shared Llama loader, where they would be dead weight for every other Llama checkpoint |
| **Chosen: `ModelType::IQuestCoder` against the same loader** | Appears in `mlxcel arch`; the refusals are scoped to this family; the compiler locates every registration site | Four extra registration edits; each distributed capability has to be granted deliberately |

**Rationale:** `mlxcel arch` is the authoritative answer to "is this checkpoint supported?", and an alias cannot answer it. The precedent is one day old: PR #1593 added `ModelType::YoutuLLM` on top of an existing decoder for exactly this reason.

**Trade-offs:** The new variant does not inherit capabilities; it has to be given them. The rule applied here is that it reproduces exactly the capability set the alias would have had, no more and no less. Pipeline parallelism goes to `StageFamily::Llama`, which is what the alias would have resolved to. Tensor parallelism stays refused, which is also what the alias would have done, because `runtime_kind_for` gates the TP Llama runtime on the architecture string and that string is `iquestcoder` under either design. Granting TP would have been a capability claim with no multi-rank validation behind it.

### 3.2 Rewriting the SentencePiece proto rather than post-processing the pieces

**Context:** The `sentencepiece` crate exposes `encode`, `open` and `from_serialized_proto`, but no way to change the normalizer of a loaded model.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Strip the leading `▁` from the first piece after encoding | No new code paths | Wrong. The dummy prefix changes the whole segmentation, not just the first piece: `▁Fibonacci` splits as `▁Fi`+`bon`+`acci`, `Fibonacci` as different pieces entirely. Post-processing cannot recover that |
| Prepend a sentinel character and drop its pieces | No protobuf handling | Relies on no vocabulary piece spanning the sentinel boundary, which is a property of the vocabulary and not of the algorithm; on a code tokenizer full of multi-character whitespace pieces this is not safe |
| Rebuild the vocabulary as a `tokenizers` crate model, as `build_plamo_tokenizer` does | Reuses an existing in-tree pattern | Needs the piece scores, so it needs a protobuf reader anyway, and then reimplements the segmentation. Strictly more surface for strictly less fidelity |
| **Chosen: rewrite `normalizer_spec.add_dummy_prefix` and reload** | The C++ implementation still does the segmentation; pieces, scores, merge order and byte fallback are copied byte for byte | A hand-written protobuf editor, which has to be written defensively and tested against malformed input |

**Rationale:** The edit is one boolean in a message whose layout is fixed and public. Everything that decides tokenization stays untouched, which is the property that matters: the change must alter the phantom prefix and nothing else.

**Trade-offs:** About 90 lines of protobuf handling that the project did not previously have. Mitigated by keeping it in its own module with a single public function, rejecting anything it does not understand, and falling back to the unmodified load with a warning rather than failing the model.

### 3.3 Keeping the checkpoint's own BPE instead of matching the reference tokenizer exactly

**Context:** The reference oracle for this checkpoint was produced by relabelling the config as `llama` so that transformers would build a fast tokenizer from `tokenizer.model`. On one of the two validation prompts, that tokenizer and the checkpoint's own SentencePiece model split "Fibonacci" differently.

The reason is that this `tokenizer.model` is a SentencePiece **BPE** model (`trainer_spec.model_type == 2`), not Unigram. A BPE model's merge order lives in the trainer's history, not in the serialized model, so the transformers converter reconstructs merges from piece scores in `SentencePieceExtractor`: for each piece it records every split into two in-vocabulary pieces, then sorts by score. That is lossy, and the recovered merge order is not always the trained one.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Reimplement the transformers merge reconstruction | Byte-identical with the recorded oracle transcript on both prompts | Enshrines a conversion artifact in mlxcel; measurably worse (79.54 nats versus 73.44 on the prose sample) |
| **Chosen: keep the checkpoint's own SentencePiece BPE** | The model's own tokenizer, and the better one by the model's own likelihood | The recorded oracle transcript for one prompt no longer matches, and the gate has to be restated against a re-run oracle |

**Rationale:** The tokenizer that ships with the weights is the tokenizer the weights were trained with. A transcript produced through a lossy conversion is evidence about the conversion, not about mlxcel.

**Trade-offs:** Anyone comparing mlxcel against an mlx-lm run of this family has to be aware that mlx-lm cannot load this checkpoint without either `trust_remote_code` or a relabel, and that the relabel silently changes the tokenizer. Appendix B records what to compare instead.

---

## 4. Implementation Details

### 4.1 Where the change sits

```
[Before]
config.json model_type "iquestcoder"
  -> get_model_type()  ->  Err("Unsupported model type: iquestcoder")

tokenizer.model  ->  SentencePieceProcessor::open()  ->  add_dummy_prefix on

[After]
config.json model_type "iquestcoder"
  -> get_model_type() -> iquest_coder_model_type()
       clip_qkv non-null?            -> Err (named reason)
       sliding window reaches a layer? -> Err (named reason)
       otherwise                     -> ModelType::IQuestCoder
  -> model_metadata registry -> Llama3Model::load -> LoadedModel::Llama

tokenizer_config.json add_prefix_space == false
  -> spm_proto::disable_add_dummy_prefix(tokenizer.model bytes)
  -> SentencePieceProcessor::from_serialized_proto()
```

### 4.2 Key Code Changes

**File: `src/tokenizer/mod.rs`**

```rust
// Before
let processor = SentencePieceProcessor::open(&tokenizer_model_path)
    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer.model: {}", e))?;
let (special_tokens, added_token_contents, add_bos) = parse_special_tokens(model_path);

// After
let config = parse_special_tokens(model_path);
let processor = open_sentencepiece_processor(&tokenizer_model_path, config.add_prefix_space)?;
```

`add_prefix_space` is an `Option<bool>` rather than a `bool`, because "absent" and "explicitly false" are different instructions: absent leaves the SentencePiece model's own setting alone, and only an explicit `false` overrides it. A non-boolean spelling reads as absent. This is what keeps the change from moving any other checkpoint.

**File: `src/models/detection.rs`**

The refusals reproduce the vendor's activation conditions rather than being stricter than them. A `sliding_window` size with the switch off still routes, and a `max_window_layers` past the last layer still routes, because in both cases the vendor never windows anything either.

### 4.3 Data Model Changes

None. `llama3::ModelArgs` parses this family's config unchanged.

---

## 5. Learning Points

### 5.1 `add_dummy_prefix` and `add_prefix_space` are the same switch in two files

**Concept:** A SentencePiece model carries `normalizer_spec.add_dummy_prefix` (default true), which prepends a space before escaping so the first word gets the same `U+2581` marker as every later word. HuggingFace exposes the same decision as `add_prefix_space` in `tokenizer_config.json`, and its converter honors it by emitting the `Prepend` normalizer only when the key is true.

**Application in this PR:** A checkpoint that ships both files can disagree with itself, and this family does. The file that reflects the publisher's intent is `tokenizer_config.json`, because that is what every HuggingFace-based serving stack reads. mlxcel now reads it too.

**Common Use Cases:**
- Any checkpoint with a `tokenizer.model` and no `tokenizer.json`. Run `grep add_prefix_space tokenizer_config.json` before trusting a greedy comparison against a reference.
- Chat-templated prompts, where the effect multiplies: one phantom space per segment between special tokens, not one per prompt.

### 5.2 A SentencePiece BPE model does not round-trip through the HuggingFace converter

**Concept:** `trainer_spec.model_type` is 1 for Unigram and 2 for BPE. A Unigram model serializes everything the algorithm needs (pieces and scores). A BPE model does not: the merge order is training state, and the serialized model only has piece scores. The converter therefore guesses the merges.

**Application in this PR:** The recorded oracle for one validation prompt was produced through that guess, and split a word differently from the model's own tokenizer. Teacher-forced likelihood settled which one the model prefers.

**Example check:**

```python
from sentencepiece import sentencepiece_model_pb2 as pb2
m = pb2.ModelProto(); m.ParseFromString(open("tokenizer.model", "rb").read())
print(m.trainer_spec.model_type)   # 2 means BPE: a converted fast tokenizer may not agree with this file
```

### 5.3 A greedy transcript is not a parity gate when the model is undecided

**Concept:** Greedy decoding is only reproducible across implementations where the top-1 to top-2 logit gap exceeds the numerical noise between them. At bf16, adjacent representable values near a logit of 64 are 0.25 apart, so a 0.25 gap is one representable step.

**Application in this PR:** On the raw Raft prompt, mlx-lm's own `generate_step` and its own eager forward on the same model object pick opposite tokens at position zero, where the gap is exactly 0.25. Any comparison against a single recorded transcript would have called mlxcel wrong there. The measurement that actually discriminates is in Appendix B: compare the reference against itself first, then read mlxcel's agreement relative to that.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| `add_dummy_prefix` | SentencePiece normalizer flag, prepends a space before escaping | The field this PR rewrites |
| `add_prefix_space` | The HuggingFace `tokenizer_config.json` spelling of the same decision | The key that was being ignored |
| `SentencePieceExtractor` | The transformers helper that reconstructs BPE merges from piece scores | Why the reference tokenizer can disagree with the model's own |
| `clip_qkv` | Per-tensor clamp on Q, K and V before attention | One of the two refused switches |
| `max_window_layers` | First layer index at which sliding-window attention applies | Why a windowed config can still be inert |
| `StageFamily` | Pipeline-parallel stage loader identifier | Where the new variant was granted PP |

### Related Technologies

- **SentencePiece**: model format and normalizer spec, https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
- **transformers slow-to-fast conversion**: https://github.com/huggingface/transformers/blob/main/src/transformers/convert_slow_tokenizer.py
- **The checkpoint's vendor decoder**: https://huggingface.co/mlx-community/IQuest-Coder-V1-7B-Instruct-8bit/blob/main/modeling_iquestcoder.py

### Related PRs and Issues

- Issue #1357: the request this PR closes.
- PR #1593: added `ModelType::YoutuLLM` on top of an existing decoder one day earlier; the precedent for a dedicated variant, and the source of the "compare the oracle against itself first" method used here.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 11 |
| Lines added | +950 |
| Lines deleted | -11 |
| Tests added | 16 |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Model routing | 5 files | `ModelType::IQuestCoder`, its registry entry, the detection arm with two refusals, and the two distributed dispatch sites |
| Tokenizer correctness | 3 files | `add_prefix_space` is read and applied through a `ModelProto` rewrite |
| Tests | 3 files | 16 tests, two of which read the real checkpoint behind the pinned gate |
| Documentation | 1 file | The family entry in `docs/supported-models.md`, including both refusals and the distributed caveat |

---

## 8. Follow-up Actions

### Required

- [ ] None blocking.

### Monitoring Required

- If another SentencePiece checkpoint is added that sets `"add_prefix_space": false`, re-run its greedy comparison; it will now tokenize differently than it did before this PR. No such checkpoint exists in the local model set today.
- The warning emitted when the `ModelProto` rewrite fails names the model path. Its appearance in a log means a checkpoint's `tokenizer.model` did not parse, which is worth investigating rather than ignoring.

### Future Improvements

- The SentencePiece encode path splits only on tokens marked `special: true`. This family also ships non-special added tokens (`<think>`, `<tool_call>`, `<tool_response>` and their closers), which HuggingFace treats as atomic on encode and mlxcel currently re-segments. It affects re-encoding an assistant turn that already contains those markers, not first-turn generation, and it predates this PR for every SentencePiece checkpoint. Worth a separate issue rather than a change smuggled into a model route.
- Tensor parallelism for this family is refused rather than validated. Enabling it means adding `iquestcoder` to `is_llama_style_architecture` and measuring on a multi-rank host.

---

## Appendix

### A. Test Results

```
cargo test --profile test-fast --features metal,accelerate --lib -- \
  tokenizer:: models::detection_tests models::metadata_tests model_metadata_tests \
  loading::tests distributed::pipeline::stage_executor
  -> 261 passed, 0 failed, 5 ignored

cargo test --profile test-fast --features metal,accelerate --bin mlxcel -- \
  family_order all_model_types supported_models arch
  -> 11 passed, 0 failed

cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings
cargo fmt --all -- --check
python3 scripts/ci/check_cross_repo_refs.py
  -> all clean
```

The two real-checkpoint tests read `models/mlx/iquest-coder-v1-7b-instruct-8bit` and route through `crate::test_support::pinned_checkpoint::skip_or_fail_pinned_checkpoint`, so they skip where the checkpoint is absent and fail under `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1`.

### B. Parity measurement

mlx-lm 0.31.3, MLX 0.32.2, the same 8-bit weights relabelled `llama` in a symlink directory so mlx-lm can load them, greedy at temperature 0, 32 tokens, identical prompt ids on both sides. `gs` is `mlx_lm.generate.generate_step`; `eager` is a per-step `model(prefix)` argmax loop on the same model object. Cells are the length of the common token prefix.

| prompt | `gs` vs `eager` | mlxcel vs `gs` | mlxcel vs `eager` | median top-2 gap | positions within 0.5 |
|---|---|---|---|---|---|
| raw: `The Fibonacci sequence begins with` | 32/32 | **32/32** | **32/32** | 3.00 | 19% |
| raw: `In distributed systems, ... such as Raft` | 0/32 | 0/32 | 13/32 | 1.00 | 44% |
| chat: palindrome function | 32/32 | **32/32** | **32/32** | 13.00 | 6% |
| chat: explain a hash map | 10/32 | 10/32 | 22/32 | 3.50 | 19% |
| chat: binary search complexity | 25/32 | **32/32** | 25/32 | 3.25 | 9% |

Read the first column first. On both prompts where mlx-lm agrees with itself for all 32 tokens, mlxcel is token-exact with both of its paths. Where mlx-lm's own two evaluation paths disagree, mlxcel agrees with one of them at least as far as they agree with each other. At the first generated position of the raw Raft prompt, the eager forward gives `▁or` 63.75 and `▁and` 63.50, and `generate_step` picks `▁and`.

### C. Reproducing the measurements

```bash
# Detection and generation, native route
./target/release/mlxcel arch | head -8
./target/release/mlxcel generate -m models/mlx/iquest-coder-v1-7b-instruct-8bit \
  -p "The Fibonacci sequence begins with" -n 32 --no-chat-template --temp 0

# The tokenizer claim, without loading any weights
python3 - <<'PY'
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb2
raw = open("models/mlx/iquest-coder-v1-7b-instruct-8bit/tokenizer.model", "rb").read()
plain = spm.SentencePieceProcessor(model_proto=raw)
m = pb2.ModelProto(); m.ParseFromString(raw); m.normalizer_spec.add_dummy_prefix = False
patched = spm.SentencePieceProcessor(model_proto=m.SerializeToString())
t = "The Fibonacci sequence begins with"
print(plain.encode(t, out_type=str))    # ['..The', ...] with the leading word boundary
print(patched.encode(t, out_type=str))  # ['The', ...] which is what mlxcel now produces
PY
```

The oracle side needs an mlx-lm environment and a copy of the checkpoint whose `config.json` says `model_type: "llama"` with `auto_map` removed, because mlx-lm cannot load the `iquestcoder` label. Note that the relabel also changes which tokenizer transformers builds; Appendix B's runs feed both sides the same ids explicitly rather than letting each tokenize its own prompt.
