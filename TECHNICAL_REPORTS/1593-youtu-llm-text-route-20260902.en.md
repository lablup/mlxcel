# Technical Report: PR #1593 - feat(models): route text-only Youtu-LLM to its MLA decoder

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle, plus a stage-by-stage numerical diff against an mlx-lm oracle
**Status**: Completed (unit coverage green; the real-checkpoint parity numbers in Appendix C were measured during review and are reproducible from the commands there)
**Languages**: Rust
**Risk Level**: Low (purely additive: two previously rejected `model_type` labels gain a route; every existing route is untouched)

---

## Executive Summary

Tencent's text-only Youtu-LLM is the decoder mlxcel already runs as Youtu-VL's text tower, but no text-only route existed: `model_type: "youtu"` and `"youtu_llm"` were both rejected at detection. PR #1593 adds `ModelType::YoutuLLM` on top of the existing `YoutuLanguageModel`, with no new architecture code, and honors three config properties the Youtu-VL sibling checkpoint never exercises.

The more useful half of this report is what the review cycle found. A first version of this PR also split the `deepseek_v2` label on `architectures[0]`, on the belief that the third published conversion (`mlx-community/Youtu-LLM-2B-4bit`, which relabels itself `deepseek_v2` for mlx-lm compatibility) was being mishandled by mlxcel's DeepSeek-V2 decoder. That belief came from a greedy-decode comparison that looked decisive and was not. Measuring it directly showed the DeepSeek-V2 route tracks an mlx-lm oracle on the same weights, and that the apparent failure was two separate artifacts: the CLI was applying the model's chat template while the oracle was not, and on raw completion prompts this checkpoint is close to undecided at many positions, so greedy output is not reproducible across implementations at all. The split was removed. What remains is the additive route the issue asked for.

---

## 1. Problem Statement

### 1.1 Background

`src/models/youtu_vl_lm.rs` already implements the whole Youtu decoder: Multi-head Latent Attention in the DeepSeek-V2 layout (a LoRA-compressed query through `q_a_proj` into `q_a_layernorm` into `q_b_proj`, split into 128 non-positional and 64 rotary dimensions; a `kv_a_proj_with_mqa` yielding a 512-dimension latent plus one 64-dimension rotary key shared across heads; a `kv_b_proj` decomposed at load into the per-head `embed_q` / `unembed_out` pair), a dense SwiGLU MLP, and tied word embeddings. A standalone `YoutuLanguageModel::load` returning `(Self, YoutuTextConfig)` also existed, reachable only from tests.

What was missing was registry work: `ModelType`, `LoadedModel`, the metadata registry, a directory route, and the detection arms. Before this change `grep YoutuLanguageModel src/` hit only `src/loading/vlm_youtu_vl.rs` and `src/vision/youtu_vl.rs`.

### 1.2 Three labels, two of them missing

| checkpoint | `model_type` | before | after |
|---|---|---|---|
| `tencent/Youtu-LLM-2B` | `youtu` | rejected: "Unsupported model type" | `ModelType::YoutuLLM` |
| `mlx-community/Youtu-LLM-2B-mlx-4bit` | `youtu_llm` | rejected | `ModelType::YoutuLLM` |
| `mlx-community/Youtu-LLM-2B-4bit` | `deepseek_v2` | `ModelType::DeepSeekV2` | unchanged |

### 1.3 What the Youtu-VL sibling does not pin

`models/mlx/youtu-vl-4b-instruct` has no `rope_scaling` block, sets `rope_interleave: true`, and sets `q_lora_rank: 1536`. Three config properties therefore reach mlxcel for the first time with the text-only checkpoint, and each was a silent-wrong-output class rather than a load error:

- `rope_interleave` was parsed into `YoutuTextConfig` and never read. The shared `DeepSeekV3Attention` passed a literal `true` to both `fast_rope` calls, so a checkpoint declaring the half-split layout would have been rotated the wrong way with no error.
- `rope_scaling` reached the carrier `DeepSeekV3Config`, where `get_attention_scale` reads the mscale keys, but the frequency table itself is plain `rope_theta`. A YaRN block that actually interpolates would be silently dropped.
- `q_lora_rank` was a required `usize`, so a `null` would fail the whole load with a serde error rather than selecting the direct `q_proj` branch the shared attention already implements.

### 1.4 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| An existing route changes for some class of checkpoint | High if it happened | Eliminated: the final change adds two `model_type` arms and touches no existing arm. `deepseek_v2_label_keeps_the_deepseek_v2_route` pins all three architecture spellings |
| The `rope_traditional` field changes DeepSeek-V3 / V3.2 / V4 / Kimi-VL behavior | High if it happened | Eliminated: `from_weights` sets it to `true`, which is the literal the two `fast_rope` calls used before; only the Youtu decoder layer overrides it |
| Youtu-VL regresses because it now shares the validated config type | Medium | Avoided: `validate_rope_scaling` is called from the text-only `load` only; the VLM loader is untouched, and no published Youtu-VL checkpoint carries the block |
| The new route is numerically wrong on a real checkpoint | High | Measured, not assumed: Appendix C. Prefill matches an mlx-lm oracle at every position; on a chat-templated prompt greedy decode is token-exact for 32 tokens |
| The rope-scaling refusal blocks a legitimate long-context export | Low | Accepted deliberately: a load error naming the unsupported scheme beats fluent wrong output; the message says which key and value triggered it |

---

## 2. Technical Review

**Security.** No new parsing surface reaches untrusted input beyond a checkpoint `config.json`, which every route already parses. `validate_rope_scaling` reads two optional keys and formats them into an error string; `factor` is read as `f64` with a default of 1.0, so a non-numeric or absent value cannot make the guard misfire. The new `parse_eos_token_ids` bounds each id through `i32::try_from` rather than an `as` cast, so an out-of-range `eos_token_id` yields no id instead of a wrapped one.

**Performance.** `DeepSeekV3Attention` grows one `bool`, read twice per layer per forward against a call that already builds a frequency table. No allocation is added on the decode path. `validate_rope_scaling` and `parse_eos_token_ids` run once at load.

**Correctness.** The load-bearing claim is that `rope_traditional: true` in `from_weights` is exactly what the two call sites did before, so no existing family moves: the literal `true` moved from the argument position into the struct initializer, and `models::deepseek` coverage is unchanged. The second claim, that the flag is not merely parsed, is pinned by a differential test rather than by inspection. The third, that the route is numerically sound, is pinned by the measurements in Appendix C rather than by argument.

---

## 3. Technical Decisions

### 3.1 The `deepseek_v2` label keeps its route, and the reasoning that first said otherwise

The first version of this PR added `deepseek_v2_or_youtu`, sending any `deepseek_v2` checkpoint whose `architectures[0]` was `YoutuForCausalLM` to the Youtu decoder. The justification was a greedy-decode comparison: mlxcel continued "The Fibonacci sequence begins with" as "1 and 2, timing with the time of the day as a factor" while an mlx-lm `deepseek_v2` oracle on the same weights produced "1,1, and each subsequent number is the sum of the two preceding numbers".

Running both decoders directly against the oracle, rather than trusting that comparison, dissolved it. On the raw five-token prompt, mlxcel's DeepSeek-V2 route emits `[220, 16, 11, 16, 11, 323, 1981, 21056, 1692, 371, 290, 3304, 324, 290, 1552, 52512, 6578, 13]`, which is the oracle's own sequence for 18 tokens: " 1,1, and each subsequent number is the sum of the two preceding numbers." On a chat-templated prompt it matches the oracle for all 32. There was nothing to fix.

The split was removed. It would have moved a class of working checkpoints onto a different decoder in exchange for nothing, which is a strictly worse trade than leaving them alone, and the guard test now records why so it is not re-added on the same reasoning.

### 3.2 A field with a preserving default, not a new `DeepSeekV3Config` key

Honoring `rope_interleave` required the flag to reach `fast_rope`. Adding it to `DeepSeekV3Config` would have been tidy, but that struct is built as a literal in the Youtu carrier and in test fixtures, so a new field is a mechanical edit at every literal for a value only one caller varies, and it puts a Youtu-specific concept into the DeepSeek config type. Threading it as a `from_weights` parameter changes a signature four families call.

The chosen shape is a `rope_traditional` field on the attention struct, set to `true` by `from_weights` and overridden by `with_rope_traditional` in the Youtu decoder layer. It is the smallest change whose default provably preserves existing behavior: the literal `true` that the two rope calls used moved into the initializer. The builder is preferred over poking a public field because it reads as intent at the call site.

### 3.3 `rope_is_interleaved()` folds two spellings of one switch

`YoutuTextConfig` carries both `rope_traditional` (the mlx-vlm port's name) and `rope_interleave` (the vendor's name), both defaulting to true. Reading only one would make the decoder depend on which converter produced the checkpoint. The accessor returns the conjunction, so either key turned off selects the half-split form and a checkpoint that sets neither behaves as it always did. No published checkpoint sets them to opposite values, so the conjunction is not resolving a real conflict; it refuses to guess which key is authoritative.

### 3.4 Refuse a YaRN factor above 1 rather than ignore it

The published block is `{"type": "yarn", "factor": 1.0, "mscale_all_dim": 0}`, the identity twice over: at factor 1 YaRN's extrapolation and interpolation frequencies coincide, and the attention mscale is 1. The vendor agrees, applying its mscale only when `mscale_all_dim` is truthy and returning 1 from `yarn_get_mscale` at factor 1. mlxcel's `get_attention_scale` already guarded on `mscale_all_dim > 0 && factor > 1`, so the scale was already correct; a test now pins that equivalence instead of leaving it to be re-derived.

What was unhandled is a factor above 1, which asks for a frequency interpolation the shared MLA attention does not implement. Ignoring it produces fluent output that is positionally wrong only past the original context length, the least detectable failure mode there is. The guard turns it into a load error naming the scheme and the factor. It is called from the text route only: the VLM loader is a shipping path with users, and adding a refusal there with no evidence of a checkpoint that needs it is a behavior change without a reason.

### 3.5 `Nonstandard` directory route with no adapter weight route

`YoutuLanguageModel::load` returns `(Self, YoutuTextConfig)`, exactly the shape `loading::nonstandard::load_pair_from_dir` consumes, so the directory route is one arm alongside `KimiLinear` and `Qwen35`. The registry entry declares `weight: None, adapter: Some(...)`, matching `DiffusionGemma` and `Llada2Moe`. That field is `adapter_weight_route`: it drives LoRA adapter loading only, so declaring it absent states honestly that adapters are unsupported rather than wiring a `SpecialWeightLoaderKind` arm nothing exercises.

### 3.6 Parse `eos_token_id` in the loader even though callers also read it from disk

Both the CLI (`crate::read_eos_token_ids`) and the server (`model_worker.rs`) merge stop ids from `generation_config.json` and `tokenizer_config.json`, so the model's own `eos_token_ids()` is not the only source. Populating it is still worth four lines: it is the contract every other family honors, and a model constructed outside those two paths otherwise reports no stop id at all.

---

## 4. Implementation Details

### 4.1 Detection

The `"youtu" | "youtu_llm"` arm sits next to `"youtu_vl"`. The `"deepseek_v2"` arm is unchanged apart from a comment recording the measurement behind leaving it alone.

### 4.2 The rope flag's path from config to kernel

| Stage | Code |
|---|---|
| Parse | `YoutuTextConfig::rope_interleave` / `rope_traditional`, both `#[serde(default = "default_true")]` |
| Fold | `YoutuTextConfig::rope_is_interleaved()` |
| Apply | `YoutuDecoderLayer::from_weights` calls `.with_rope_traditional(config.rope_is_interleaved())` |
| Use | `DeepSeekV3Attention::forward` passes `self.rope_traditional` to both `fast_rope` calls |

The vendor's own branch is `YoutuMLAttention.forward` in https://huggingface.co/tencent/Youtu-LLM-2B/blob/main/modeling_youtu.py, selecting `apply_rotary_pos_emb_interleave` (a `view(..., d // 2, 2).transpose(4, 3)` before the usual `rotate_half`) over `apply_rotary_pos_emb`. That reshape is what MLX spells `traditional=True`. The two forms are related by one shared permutation of the rotated dimensions, which the query-key dot product is invariant under, so the interleaved HF form and MLX's traditional form give the same attention scores.

### 4.3 Config type changes

`q_lora_rank: Option<usize>` with `#[serde(default)]`, passed straight into the carrier `DeepSeekV3Config`, where `from_weights` already branches on `is_none()` to select `q_proj` over the LoRA chain. `validate_rope_scaling()` returns `Result<(), String>` and is called from `load` before any weight is touched, so a refused checkpoint costs one config read rather than a full weight load.

### 4.4 Registration

`ModelType::YoutuLLM` in the enum, `ALL_MODEL_TYPES`, `metadata()` (family `Specialized`, already in `FAMILY_ORDER`), and the `all_variants!` exhaustiveness list; `LoadedModel::YoutuLLM` plus its `delegate_language_model!` arm; `model_metadata.rs` as `kind: Text, directory: Nonstandard`; `loading/nonstandard.rs` as a `load_pair_from_dir` arm; and an arch-string arm in `src/distributed/tensor_parallel/inference.rs` so the dispatch table stays total.

---

## 5. Learning Points

### 5.1 A greedy-decode comparison is not evidence unless the reference is precision-stable

This checkpoint is close to undecided on raw completion prompts. Over a 126-token passage the median top-1 to top-2 logit gap is 0.625, and one position in eight is an exact tie in bf16. At that margin greedy output is not a property of the implementation; it is a property of the last bit. mlx-lm's own bf16 and float32 runs of the same weights agree on only 91.3 percent of those positions, and on the five-token Fibonacci prompt they produce entirely different text from the sixth token on.

So "mlxcel says X, the reference says Y, therefore mlxcel is broken" is only an argument when the reference agrees with itself at higher precision. The cheap check is to run the reference twice, once at its native precision and once upcast, and compare. If they differ, the comparison cannot carry the weight being put on it, and the prompt needs replacing before any conclusion is drawn.

### 5.2 Check which prompt each side actually received before comparing outputs

The reported mlxcel output began "Okay, so the question is about whether the Fibonacci sequence can be generalized", which reads as a model ignoring its prompt. It is not. Rendering the same prompt through the checkpoint's own chat template and running the oracle produces `<think>\nOkay, the user is asking about the Fibonacci sequence beginning...`. The CLI was applying the chat template and the oracle was not, so the two sides were answering different questions. The tell was available without any numerics: a reasoning model's `<think>` opener appearing in output that was supposed to be a raw completion.

### 5.3 A stage-by-stage diff answers "where", and a teacher-forced diff answers "whether"

The stage diff located the first numerical divergence precisely (bit-exact through the query path, the latent, and both rope calls; first difference at the per-head key and value materialization). But a stage diff cannot say whether a divergence matters, because every stage past the first inherits the previous one's error. The teacher-forced comparison answers that: feed both implementations the same fixed token sequence, compare the argmax at every position, and the feedback loop is removed. Here it gave 118 of 126 positions agreeing, with every disagreement at a position whose top-2 gap was 0 to 0.75 logits. That is the number that settles the question; the free-running greedy trajectory never could.

### 5.4 A config key that is parsed but never read is worse than one that is missing

`rope_interleave` had a field, a default, and a doc comment, and no consumer. Every review that grepped for the key found it. This is the same shape as `rope_scaling` being parsed and dropped across four families (#1355): the presence of the field is what stops anyone from asking whether it is applied. Grepping for consumers rather than declarations is the check that catches it, and the test that has teeth is differential (flip the flag, require the logits to change) rather than an assertion on the parsed value.

### 5.5 Verify a premise on the machine before encoding it in a routing rule

The detection split was written from a reported comparison rather than a measured one, and it survived a full implementation, review, and report cycle before anyone ran both decoders on the same input. Routing rules are exactly where an unverified premise is most expensive: they are invisible in normal use and they change behavior for checkpoints nobody in the loop owns.

---

## 6. Further Learning

### Key Terms

- **MLA (Multi-head Latent Attention)**: attention whose keys and values are reconstructed from a low-rank latent, so the cache stores the latent rather than full K/V. DeepSeek-V2's layout, reused verbatim by Youtu.
- **Absorbed MLA**: folding `kv_b_proj` into the query and output projections (`embed_q` / `unembed_out`) so decode never materializes full keys. What `DeepSeekV3Attention` always does and `deepseek_v2.rs` does only under a flag.
- **Interleaved (traditional) RoPE**: rotating adjacent dimension pairs rather than half-split pairs, spelled `traditional=True` in MLX and produced in PyTorch by a `view(d // 2, 2).transpose` before `rotate_half`.
- **YaRN identity block**: a `rope_scaling` entry whose `factor` is 1, where the interpolated and extrapolated frequency tables coincide and the attention mscale is 1, so it is equivalent to no block.
- **Teacher forcing**: running a model over a fixed token sequence and reading the prediction at every position, instead of feeding its own output back. Removes trajectory divergence from a comparison.

### Related PRs/Issues

- #1371: this issue.
- #1355: `rope_scaling` parsed but never applied across four families. The same defect class as 5.4.
- #958, #1026: the shared hardening of the MLA `kv_b_proj` sanitizers this route reuses.

---

## 7. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 13 |
| New `ModelType` variants | 1 |
| New tests | 8 |
| Existing detection arms modified | 0 |

### Changes by Category

- **Detection**: `src/models/detection.rs` (two new `model_type` arms), `src/models/detection_tests.rs`.
- **Model and config**: `src/models/youtu_vl_lm.rs`, `src/models/youtu_vl_lm_config.rs`, `src/models/youtu_vl_lm_sanitize.rs` (comment only), `src/models/youtu_vl_lm_tests.rs`, `src/models/deepseek_v3.rs`.
- **Registry and loading**: `src/models/mod.rs`, `src/loaded_model.rs`, `src/model_metadata.rs`, `src/loading/nonstandard.rs`, `src/distributed/tensor_parallel/inference.rs`.
- **Docs**: `docs/supported-models.md`.

---

## 8. Follow-up Actions

### Monitoring Required

- Parity comparisons for this family must use the chat template. A raw-prompt greedy diff against any reference will produce spurious failures, and the docs entry says so; a future contributor who does not read it will rediscover 5.1.

### Future Improvements

- Extend `validate_rope_scaling` to the Youtu-VL loader once someone confirms no shipping VLM checkpoint carries a scaling block. It is the same silent-wrong-output class on that route.
- Implement YaRN frequency interpolation in the shared MLA attention, turning the refusal into support. Worth doing only if a Youtu or Kimi-VL export appears with a factor above 1.
- Validate `tencent/Youtu-LLM-2B` (bf16, `model_type: youtu`) and `mlx-community/Youtu-LLM-2B-mlx-4bit` (`model_type: youtu_llm`) end to end. Neither is on the validation host; both take the plain label arms, so this closes the last two labels rather than testing new code.
- The absorbed MLA and the materialized form used by `deepseek_v2.rs` are algebraically the same and numerically different, and this checkpoint is sensitive enough to show it on raw prompts. If a family ever needs bit-level agreement with mlx-lm, the materialized form is the one that shares its arithmetic order.

---

## Appendix

### A. Test Results

| Suite | Result |
|---|---|
| `--lib -- models::youtu_vl_lm models::detection_tests loading::nonstandard` | 60 passed |
| `--lib -- models::deepseek models::metadata_tests model_metadata_tests loading::tests` | 197 passed |
| `--bin mlxcel -- family_order all_model_types supported_models arch` | 10 passed |
| `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |

### B. What the new tests pin

- `youtu_llm_model_type_is_detected_for_both_vendor_labels`: `youtu` and `youtu_llm` both reach `ModelType::YoutuLLM`.
- `deepseek_v2_label_keeps_the_deepseek_v2_route`: the Youtu relabel, a genuine `DeepseekV2ForCausalLM`, and a config with no `architectures` array all stay on `DeepSeekV2`. The guard against re-adding the split described in 3.1.
- `text_only_config_with_identity_yarn_parses`: the real field set parses, and the attention scale with the identity YaRN block equals both the no-block scale and the literal `(qk_nope + qk_rope) ** -0.5`.
- `rope_scaling_that_interpolates_is_refused`: a factor of 4 produces an error naming the factor and the scheme.
- `null_q_lora_rank_parses_and_selects_the_direct_q_projection`: null, absent, and numeric ranks all parse with the right branch selector.
- `rope_interleave_and_rope_traditional_are_the_same_switch`: either key turned off selects the half-split form.
- `tied_embeddings_produce_logits_without_an_lm_head`: a synthetic tied build has `lm_head == None` and produces finite `[1, 4, vocab]` logits through the whole decoder.
- `rope_interleave_reaches_the_rope_call`: two models built from identical synthetic weights with the flag flipped produce different logits. The assertion that the flag is not parsed and dropped.
- `prefill_and_decode_paths_agree_position_by_position`: the absorbed `l == 1` decode arm and the multi-token prefill arm agree at every position. Nothing else in the suite exercises the decode arm.

### C. Real-checkpoint measurements

All against `mlx-lm` 0.31.3 loading the same `mlx-community/Youtu-LLM-2B-4bit` weights, prompt ids identical on both sides, greedy at temperature 0.

**Prefill, five-token raw prompt.** Stage-by-stage, mlxcel against the oracle: embedding output bit-identical; layer 0 attention output 0.33 percent mean relative difference; per-layer hidden state drifting from 1.0 percent at layer 0 to 5.8 percent at layer 31; final norm 7.3 percent; logits 5.4 percent. The argmax at all five positions is identical. Within layer 0 the query path (`q_a_proj`, `q_a_layernorm`, `q_b_proj`, the nope and rope split), the compressed KV, the latent, and both rope calls are all bit-identical; the first difference is the per-head key and value materialization, where mlxcel contracts a dequantized weight and the oracle a still-quantized one.

**Teacher forcing, 126-token passage.** mlxcel agrees with the bf16 oracle on 118 of 126 argmax positions (93.7 percent). Every disagreement is at a position whose top-1 to top-2 gap is between 0.0 and 0.75 logits, against a median gap of 0.625 over the passage; three of the eight are exact ties. For scale, the bf16 oracle agrees with its own float32 self on only 115 of 126 (91.3 percent) of the same positions.

**Greedy decode, raw prompt.** On "The Fibonacci sequence begins with" the bf16 oracle and the float32 oracle diverge at step 5 (top-2 gap 0.250) and produce entirely different text: " 1,1, and each subsequent number is the sum" against " 1,1,1,4,34,". mlxcel reproduces the float32 sequence exactly for 12 tokens. The prompt is not usable as a parity gate.

**Greedy decode, chat-templated prompt.** Rendering the same request through the checkpoint's chat template raises the median top-2 gap from 0.625 to 3.875 and the bf16 and float32 oracles agree on all 32 tokens. mlxcel is token-exact against both for all 32:

```
prompt ids  [128000, 128236, 837, 91949, 12082, 13328, 458, 128237]
expected    [128227, 198, 37317, 11, 290, 1483, 371, 11935, 913, 290, 91949, 12082, 7963, 13, 6846, 611, 1311, 603, 125377, 1165, 290, 91949, 12082, 371, 13, 5902, 261, 4326, 324, 6578, 1551, 1981]
text        "<think>\nOkay, the user is asking about the Fibonacci sequence beginning. Let me start by recalling what the Fibonacci sequence is. It's a series of numbers where each"
```

A second chat prompt ("In distributed systems, consensus protocols such as Raft") is also token-exact for 32.

**The DeepSeek-V2 route on the same weights.** Chat-templated, 32 of 32 identical to the oracle. Raw prompt, 18 of the oracle's first tokens identical. This is the measurement that removed the detection split described in 3.1.

**Reproducing.** The route is exercised through `YoutuLanguageModel::load`, which is what `loading::nonstandard` calls. Because the local checkpoint declares `deepseek_v2` it reaches the DeepSeek-V2 route by design; to drive the new route from the CLI, copy the directory and rewrite `model_type` to `youtu`:

```bash
cargo build --release --features metal,accelerate
./target/release/mlxcel arch | grep -i youtu

DIR=models/mlx/youtu-llm-2b-4bit
COPY=/tmp/youtu-llm-2b-4bit-youtu
mkdir -p "$COPY" && for f in "$DIR"/*; do ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$COPY/"; done
rm -f "$COPY/config.json"
python3 -c "import json;c=json.load(open('$DIR/config.json'));c['model_type']='youtu';json.dump(c,open('$COPY/config.json','w'))"

./target/release/mlxcel generate -m "$COPY" -p "The Fibonacci sequence begins with" -n 32 --temp 0
```

`mlxcel arch` must list `Youtu-LLM (DeepSeek-V2-style MLA decoder, text-only)` under `Specialized:` next to the existing Youtu-VL entry under `Other VLM:`. The generation must reproduce the chat-templated text above, because the CLI applies the checkpoint's chat template by default; pass `--no-chat-template` only alongside a raw-prompt reference, and see 5.1 before drawing a conclusion from one.
