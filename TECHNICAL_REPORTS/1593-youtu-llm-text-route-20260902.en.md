# Technical Report: PR #1593 - feat(models): route text-only Youtu-LLM to its MLA decoder

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (unit coverage green; the real-checkpoint greedy comparison against the mlx-lm oracle is run by the merge orchestrator, see Appendix C)
**Languages**: Rust
**Risk Level**: Medium (one existing route changes for a specific class of checkpoint; the shared MLA attention gains a field whose default preserves every current caller)

---

## Executive Summary

Tencent's text-only Youtu-LLM is the decoder mlxcel already runs as Youtu-VL's text tower, but no text-only route existed. Two of its three published `model_type` labels were rejected at detection, and the third reached the wrong decoder. PR #1593 adds `ModelType::YoutuLLM` on top of the existing `YoutuLanguageModel` with no new architecture code, and makes detection decide the `deepseek_v2` label by `architectures[0]` rather than by the label alone.

That last part is a behavior change, not an addition, and it exists because the issue's premise was measurably wrong. `mlx-community/Youtu-LLM-2B-4bit` relabels itself `deepseek_v2` so mlx-lm can load it. The issue assumed that meant it already worked through mlxcel's DeepSeek-V2 route and only needed verifying. On the local checkpoint that route loads and generates but produces incoherent text, while mlx-lm's `deepseek_v2` route on the same weights is coherent. The export still declares `architectures: ["YoutuForCausalLM"]` and an `auto_map` naming the vendor's own modules, so the architecture string is the discriminator that survives the relabelling.

Three config properties that the Youtu-VL sibling checkpoint never exercises are now honored on the text route: `rope_interleave` reaches the rope call instead of being assumed, a `rope_scaling` block is validated at load rather than silently ignored, and `q_lora_rank` is optional.

---

## 1. Problem Statement

### 1.1 Background

`src/models/youtu_vl_lm.rs` already implements the whole Youtu decoder: Multi-head Latent Attention in the DeepSeek-V2 layout (a LoRA-compressed query through `q_a_proj` into `q_a_layernorm` into `q_b_proj`, split into 128 non-positional and 64 rotary dimensions; a `kv_a_proj_with_mqa` yielding a 512-dimension latent plus one 64-dimension rotary key shared across heads; a `kv_b_proj` decomposed at load into the per-head `embed_q` / `unembed_out` pair), a dense SwiGLU MLP, and tied word embeddings. It also already had a standalone `YoutuLanguageModel::load` returning `(Self, YoutuTextConfig)`, reachable only from tests.

What was missing was purely registry work: `ModelType`, `LoadedModel`, the metadata registry, a directory route, and the detection arms. `grep YoutuLanguageModel src/` hit only `src/loading/vlm_youtu_vl.rs` and `src/vision/youtu_vl.rs` before this change.

### 1.2 Three labels, and one of them lies

| checkpoint | `model_type` | `architectures[0]` | before |
|---|---|---|---|
| `tencent/Youtu-LLM-2B` | `youtu` | `YoutuForCausalLM` | rejected: "Unsupported model type" |
| `mlx-community/Youtu-LLM-2B-mlx-4bit` | `youtu_llm` | `YoutuForCausalLM` | rejected |
| `mlx-community/Youtu-LLM-2B-4bit` | `deepseek_v2` | `YoutuForCausalLM` | loaded through `DeepSeekV2`, incoherent output |

The first two are a missing-arm problem. The third is the interesting one, and its own model card admits the relabel: "Converted using deepseek_v2 architecture mapping (compatible MLA implementation)."

### 1.3 The measured evidence that the third row is a misroute

On `models/mlx/youtu-llm-2b-4bit`, greedy decode at temperature 0 with `--no-chat-template`:

| prompt | mlxcel on `main` (DeepSeek-V2 route) | mlx-lm `deepseek_v2` on the same weights |
|---|---|---|
| "The Fibonacci sequence begins with" | "1 and 2, timing with the time of the day as a factor. Given the time of day in minutes since midnight" | "1,1, and each subsequent number is the sum of the two preceding numbers. The sequence is d..." |
| "In distributed systems, consensus protocols such as Raft" | "Okay, so the user" | ", Paxos, and Raft again, are used to achieve agreement among multiple nodes. In a distribu..." |

Both mlxcel outputs are finite and fluent English, which is the signature of a wrong-but-loading numerical path rather than a crash or a weight-lookup failure. The issue's acceptance criterion "verified to load through the existing DeepSeek-V2 route" would have passed on a load check and on an eyeball of the first few tokens, which is exactly why it is written as a cross-check in the gate rather than as a load assertion.

### 1.4 What the VLM sibling does not pin

`models/mlx/youtu-vl-4b-instruct` has no `rope_scaling` block at all, sets `rope_interleave: true`, and sets `q_lora_rank: 1536`. So three config properties reach mlxcel for the first time with the text-only checkpoint, and each was a silent-wrong-output class rather than a load error:

- `rope_interleave` was parsed into `YoutuTextConfig` and never read. The shared `DeepSeekV3Attention` passed a literal `true` to both `fast_rope` calls.
- `rope_scaling` was passed into the carrier `DeepSeekV3Config` where `get_attention_scale` reads the mscale keys, but the frequency table itself is plain `rope_theta`. A YaRN block that actually interpolates would be silently dropped.
- `q_lora_rank` was a required `usize`, so a `null` would fail the entire load with a serde error rather than selecting the direct `q_proj` branch the shared attention already implements.

### 1.5 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| A genuine DeepSeek-V2 checkpoint is diverted to the Youtu decoder | High if it happened | Eliminated by construction: the split keys on `architectures[0] == "YoutuForCausalLM"` exactly; `DeepseekV2ForCausalLM` and a missing `architectures` array are both pinned by a test |
| The `rope_traditional` field changes DeepSeek-V3 / V3.2 / V4 / Kimi-VL behavior | High if it happened | Eliminated: `from_weights` sets it to `true`, which is the literal the two `fast_rope` calls used before; only the Youtu decoder layer overrides it |
| Youtu-VL regresses because it now shares the validated config type | Medium | Avoided: `validate_rope_scaling` is called from the text-only `load` only; the VLM loader is untouched, and no published Youtu-VL checkpoint carries the block |
| The new route inherits whatever makes the DeepSeek-V2 route wrong here | High | Not eliminated by unit tests: the shared MLA attention is a different implementation (always-absorbed, plain rope) than `deepseek_v2.rs` (kv_b_proj retained for prefill, YaRN rope), and Youtu-VL runs it, but only the orchestrator's real-checkpoint run proves it. That is Appendix C |
| The rope-scaling refusal blocks a legitimate long-context export | Low | Accepted deliberately: a load error naming the unsupported scheme beats fluent wrong output; the message says exactly which key and value triggered it |

---

## 2. Technical Review

**Security.** No new parsing surface reaches untrusted input beyond a checkpoint `config.json`, which every route already parses. The one new read is `architectures[0]`, through the existing `first_architecture` helper that four other detection rules already use. `validate_rope_scaling` reads two optional keys and formats them into an error string; the `factor` is read as `f64` with a default of 1.0, so a non-numeric or absent value cannot make the guard misfire. The new `parse_eos_token_ids` bounds each id through `i32::try_from` rather than an `as` cast, so an out-of-range `eos_token_id` yields no id instead of a wrapped one.

**Performance.** `DeepSeekV3Attention` grows one `bool`. It is read once per rope call, which is two reads per layer per forward, against a call that already builds a frequency table. No allocation is added anywhere on the decode path. `validate_rope_scaling` and `parse_eos_token_ids` run once at load.

**Correctness.** The load-bearing claim is that `rope_traditional: true` in `from_weights` is exactly what the two call sites did before, so no existing family moves. That is checkable by reading the diff: the literal `true` moved from the argument position into the struct initializer, and `models::deepseek` unit coverage (197 tests with the neighboring registry suites) is unchanged. The second claim, that the flag is not merely parsed, is pinned by a differential test rather than by inspection: two models built from the same synthetic weights with `rope_interleave` flipped must produce different logits. A test that only asserted the field's value would pass against an implementation that reads it and then ignores it.

---

## 3. Technical Decisions

### 3.1 The architecture string decides the `deepseek_v2` label, scoped to that one arm

The alternatives were a structural discriminator (MLA geometry, `rope_interleave` presence, `rope_theta` magnitude), a pre-match check ahead of the whole `model_type` dispatch, or asking users to rewrite the label.

A structural discriminator is the wrong shape here because Youtu's MLA geometry is close to DeepSeek-V2-Lite's by design; that closeness is why the relabel works for mlx-lm at all. Any threshold on head dims or `rope_theta` would be a guess that a future DeepSeek variant could cross.

A pre-match check would apply the rule to every label, which is broader than the evidence. Only `deepseek_v2` is known to be borrowed by a Youtu export. Scoping the rule to that arm through a named helper, `deepseek_v2_or_youtu`, keeps the blast radius to one label and puts the rationale where the next reader of that arm will find it.

Rewriting the label is a documentation answer to a detection problem: it makes the published checkpoint unusable as published, which is the thing this issue exists to fix.

The negative cases matter as much as the positive one, so `genuine_deepseek_v2_keeps_its_route` covers both `DeepseekV2ForCausalLM` and a config with no `architectures` array at all. Many older MLX conversions omit the array, and a rule that treated "absent" as "not DeepSeek-V2" would have been a silent regression for them.

### 3.2 A field with a preserving default, not a new `DeepSeekV3Config` key

Honoring `rope_interleave` required the flag to reach `fast_rope`. Three shapes were available.

Adding `rope_interleave` to `DeepSeekV3Config` would have been the tidy option, but that struct is built as a literal in the Youtu carrier and in test fixtures, so a new field is a mechanical edit at every literal for a value only one caller varies. It also puts a Youtu-specific concept into the DeepSeek config type.

Threading it as a parameter on `from_weights` changes a signature four families call.

The chosen shape is a `rope_traditional` field on the attention struct, set to `true` by `from_weights` and overridden by `with_rope_traditional` in the Youtu decoder layer. It is the smallest change whose default provably preserves the existing behavior: the literal `true` that the two rope calls used simply moved into the initializer. The builder is preferred over a bare public field poke because it reads as intent at the call site and keeps the override to one line in `YoutuDecoderLayer::from_weights`.

### 3.3 `rope_is_interleaved()` folds two spellings of one switch

`YoutuTextConfig` carries both `rope_traditional` (the mlx-vlm port's name) and `rope_interleave` (the vendor's name), both defaulting to true. Reading only one would make the decoder depend on which converter produced the checkpoint. The accessor returns `self.rope_interleave && self.rope_traditional`, so either key turned off selects the half-split form, and a checkpoint that sets neither behaves as it always did. No published checkpoint sets the two to opposite values, so the conjunction is not resolving a real conflict; it is refusing to guess which key is authoritative.

### 3.4 Refuse a YaRN factor above 1 rather than ignore it

The published block is `{"type": "yarn", "factor": 1.0, "mscale_all_dim": 0}`, which is the identity twice over: at factor 1 YaRN's extrapolation and interpolation frequencies coincide, and the attention mscale is 1. The vendor agrees, applying its mscale only when `mscale_all_dim` is truthy and returning 1 from `yarn_get_mscale` at factor 1. mlxcel's `get_attention_scale` already guards on `mscale_all_dim > 0 && factor > 1`, so the scale was already correct for this block. A test now pins that equivalence instead of leaving it to be re-derived.

What was not handled is a factor above 1, which asks for a frequency interpolation the shared MLA attention does not implement. Ignoring it produces fluent output that is positionally wrong only past the original context length, which is the least detectable failure mode there is. The guard turns it into a load error naming the scheme and the factor.

The guard is deliberately called from `YoutuLanguageModel::load` (the text route) and not from the VLM loader. The VLM loader is a shipping path with existing users, and while no published Youtu-VL checkpoint carries a `rope_scaling` block, adding a new refusal there is a behavior change with no evidence behind it. Extending the guard to the VLM route is listed as a follow-up rather than done silently here.

### 3.5 `Nonstandard` directory route with no adapter weight route

`YoutuLanguageModel::load` returns `(Self, YoutuTextConfig)`, which is the exact shape `loading::nonstandard::load_pair_from_dir` consumes, so the directory route is one arm alongside `KimiLinear` and `Qwen35`. The registry entry declares `weight: None, adapter: Some(...)`, matching `DiffusionGemma` and `Llada2Moe`. That field is `adapter_weight_route`: it drives LoRA adapter loading only, not the ordinary load, so declaring it absent states honestly that adapters are not supported rather than wiring a `SpecialWeightLoaderKind` arm that nothing exercises.

### 3.6 Parse `eos_token_id` in the loader even though callers also read it from disk

Both the CLI (`crate::read_eos_token_ids` in `commands/generate.rs`) and the server (`model_worker.rs`) merge stop ids read from `generation_config.json` and `tokenizer_config.json`, so the model's own `eos_token_ids()` is not the only source. It was still worth populating: `LanguageModel::eos_token_ids` is the contract every other family honors, and a model constructed outside those two paths otherwise reports no stop id at all. It is four lines and removes a way for a future caller to be surprised.

### 3.7 Not chasing the DeepSeek-V2 decoder

Why mlxcel's DeepSeek-V2 route mishandles a Youtu export is a real question and is not answered here. The two implementations differ in more than one place: `deepseek_v2.rs` keeps `kv_b_proj` loaded and up-projects for prefill with an optional absorbed decode path, and drives rope through a `YarnRoPE` with precomputed frequencies, whereas the shared `DeepSeekV3Attention` is always absorbed and calls `fast_rope` with a plain base. Either could be the divergence, and narrowing it needs a layer-by-layer trace against an mlx-lm oracle, which is its own piece of work with its own blast radius on four DeepSeek families. Sending Youtu exports to the decoder written for them is the correct fix for this issue either way, and it is independent of that investigation.

---

## 4. Implementation Details

### 4.1 Detection

`deepseek_v2_or_youtu(&v)` sits beside `first_architecture` in `src/models/detection.rs` and is called from the `"deepseek_v2"` arm; the `"youtu" | "youtu_llm"` arm sits next to `"youtu_vl"`. The constant `YOUTU_CAUSAL_LM_ARCHITECTURE` names the discriminator once.

### 4.2 The rope flag's path from config to kernel

| Stage | Code |
|---|---|
| Parse | `YoutuTextConfig::rope_interleave` / `rope_traditional`, both `#[serde(default = "default_true")]` |
| Fold | `YoutuTextConfig::rope_is_interleaved()` |
| Apply | `YoutuDecoderLayer::from_weights` calls `.with_rope_traditional(config.rope_is_interleaved())` |
| Use | `DeepSeekV3Attention::forward` passes `self.rope_traditional` to both `fast_rope` calls |

The vendor's own branch is `YoutuMLAttention.forward` in https://huggingface.co/tencent/Youtu-LLM-2B/blob/main/modeling_youtu.py, which selects `apply_rotary_pos_emb_interleave` (a `view(..., d // 2, 2).transpose(4, 3)` before the usual `rotate_half`) over `apply_rotary_pos_emb`. That reshape is what MLX spells `traditional=True`.

### 4.3 Config type changes

`q_lora_rank: Option<usize>` with `#[serde(default)]`, passed straight into the carrier `DeepSeekV3Config` where `from_weights` already branches on `is_none()` to select `q_proj` over the LoRA chain. `validate_rope_scaling()` returns `Result<(), String>` and is called from `load` before any weight is touched, so a refused checkpoint costs one config read rather than a full weight load.

### 4.4 Registration

`ModelType::YoutuLLM` in the enum, `ALL_MODEL_TYPES`, `metadata()` (family `Specialized`, already present in `FAMILY_ORDER`), and the `all_variants!` exhaustiveness list; `LoadedModel::YoutuLLM` plus its `delegate_language_model!` arm; `model_metadata.rs` as `kind: Text, directory: Nonstandard`; `loading/nonstandard.rs` as a `load_pair_from_dir` arm; and an arch-string arm in `src/distributed/tensor_parallel/inference.rs` so the dispatch table stays total (the family is not TP-enabled, and the planner's supported-architecture validation rejects the string before any TP load).

---

## 5. Learning Points

### 5.1 A compatibility relabel is a claim about the loader, not about the weights

`model_type` is not a description of an architecture; it is a request to be routed somewhere. When a converter relabels a checkpoint to be loadable by one runtime, the label becomes a statement about that runtime's decoder, and any other runtime that trusts it inherits a claim it never verified. `architectures[0]` is the field that keeps saying what the weights are, which is why it survives the relabel and makes a usable discriminator.

### 5.2 A config key that is parsed but never read is worse than one that is missing

`rope_interleave` had a field, a default, and a doc comment, and no consumer. Every review that grepped for the key found it. This is the same shape as `rope_scaling` being parsed and dropped across four families (#1355): the presence of the field is what stops anyone from asking whether it is applied. Grepping for consumers rather than for declarations is the check that catches it.

### 5.3 A flag test that asserts a value is not a test that the flag works

`assert!(config.rope_is_interleaved())` passes against a decoder that reads the flag and discards it. The assertion that has teeth is differential: build the model twice with the flag flipped and require the logits to differ. It costs a synthetic weight map and buys the one property the change is actually about.

### 5.4 An unimplementable config value should fail loudly at load

The decoder has no YaRN frequency interpolation. Accepting a `factor: 4.0` and decoding on the plain table would be correct for short prompts and wrong only past the original context length, so nothing short of a long-context parity run would catch it. Refusing it at load converts an invisible class into a message naming the key. This is the same lesson the maskless-prefill and `rope_scaling` defect classes taught: a short prompt cannot see a positional bug, and fluent output is not evidence.

---

## 6. Further Learning

### Key Terms

- **MLA (Multi-head Latent Attention)**: attention whose keys and values are reconstructed from a low-rank latent, so the cache stores the latent rather than full K/V. DeepSeek-V2's layout, reused verbatim by Youtu.
- **Absorbed MLA**: folding `kv_b_proj` into the query and output projections (`embed_q` / `unembed_out`) so decode never materializes full keys. What `DeepSeekV3Attention` always does and `deepseek_v2.rs` does only under a flag.
- **Interleaved (traditional) RoPE**: rotating adjacent dimension pairs rather than the half-split pairs, spelled `traditional=True` in MLX and produced in PyTorch by a `view(d // 2, 2).transpose` before `rotate_half`.
- **YaRN identity block**: a `rope_scaling` entry whose `factor` is 1, where the interpolated and extrapolated frequency tables coincide and the attention mscale is 1, so it is equivalent to no block at all.

### Related PRs/Issues

- #1371: this issue.
- #1355: `rope_scaling` parsed but never applied across four families. The same defect class as 5.2.
- #958, #1026: the shared hardening of the MLA `kv_b_proj` sanitizers this route reuses.

---

## 7. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 13 |
| Lines added | 617 |
| Lines removed | 20 |
| New `ModelType` variants | 1 |
| New tests | 7 |

### Changes by Category

- **Detection**: `src/models/detection.rs`, `src/models/detection_tests.rs`.
- **Model and config**: `src/models/youtu_vl_lm.rs`, `src/models/youtu_vl_lm_config.rs`, `src/models/youtu_vl_lm_sanitize.rs` (comment only), `src/models/youtu_vl_lm_tests.rs`, `src/models/deepseek_v3.rs`.
- **Registry and loading**: `src/models/mod.rs`, `src/loaded_model.rs`, `src/model_metadata.rs`, `src/loading/nonstandard.rs`, `src/distributed/tensor_parallel/inference.rs`.
- **Docs**: `docs/supported-models.md`.

---

## 8. Follow-up Actions

### Monitoring Required

- The claim that the new route is numerically correct on a real checkpoint rests entirely on the orchestrator's greedy comparison in Appendix C. No unit test can establish it, because the synthetic weights that unit tests use have no reference to compare against.

### Future Improvements

- Extend `validate_rope_scaling` to the Youtu-VL loader once someone confirms no shipping VLM checkpoint carries a scaling block. It is the same silent-wrong-output class on that route.
- Implement YaRN frequency interpolation in the shared MLA attention, which would turn the refusal into support. Worth doing only if a Youtu or Kimi-VL export appears with a factor above 1.
- Root-cause why mlxcel's DeepSeek-V2 route mishandles a Youtu-shaped export (3.7). Independent of this change, but the answer may be a defect that affects genuine DeepSeek-V2 checkpoints too.
- Validate `tencent/Youtu-LLM-2B` (bf16, `model_type: youtu`) and `mlx-community/Youtu-LLM-2B-mlx-4bit` (`model_type: youtu_llm`) end to end. Neither is on the validation host; both take the plain label arms that unit tests cover, so this closes the last of the three labels rather than testing new code.

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
- `deepseek_v2_labelled_youtu_export_routes_to_the_youtu_decoder`: the config shape of the local 4-bit checkpoint, built from its fields rather than read from disk, reaches the Youtu decoder.
- `genuine_deepseek_v2_keeps_its_route`: `DeepseekV2ForCausalLM` and a config with no `architectures` array both stay on `DeepSeekV2`. This is the regression guard for every existing DeepSeek-V2 user.
- `text_only_config_with_identity_yarn_parses`: the real field set parses, and the attention scale with the identity YaRN block equals both the no-block scale and the literal `(qk_nope + qk_rope) ** -0.5`.
- `rope_scaling_that_interpolates_is_refused`: a factor of 4 produces an error naming the factor and the scheme.
- `null_q_lora_rank_parses_and_selects_the_direct_q_projection`: null, absent, and numeric ranks all parse, and the carrier config carries the right branch selector for each.
- `rope_interleave_and_rope_traditional_are_the_same_switch`: either key turned off selects the half-split form.
- `tied_embeddings_produce_logits_without_an_lm_head`: a synthetic tied build has `lm_head == None` and produces finite `[1, 4, vocab]` logits through the whole decoder.
- `rope_interleave_reaches_the_rope_call`: two models built from identical synthetic weights with the flag flipped produce different logits. This is the assertion that the flag is not being parsed and dropped.

### C. Post-merge validation (orchestrator)

The local `models/mlx/youtu-llm-2b-4bit` checkout exercises the new route directly: with this change its `architectures[0]` sends it to `YoutuLLM`, and on `main` the same directory goes to `DeepSeekV2`, which makes this a differential gate rather than a smoke test.

```bash
cargo build --release --features metal,accelerate
./target/release/mlxcel arch | grep -i youtu
./target/release/mlxcel generate -m models/mlx/youtu-llm-2b-4bit --no-chat-template -n 32 --temp 0 \
  -p "The Fibonacci sequence begins with"
./target/release/mlxcel generate -m models/mlx/youtu-llm-2b-4bit --no-chat-template -n 32 --temp 0 \
  -p "In distributed systems, consensus protocols such as Raft"
```

Expected: `mlxcel arch` lists `Youtu-LLM (DeepSeek-V2-style MLA decoder, text-only)` under `Specialized:` alongside the existing Youtu-VL entry under `Other VLM:`. Both generations should match the mlx-lm `deepseek_v2` oracle continuations quoted in 1.3 over their leading tokens, with quantization and reduction-order noise allowed in the tail. The failure signal is the current `main` behavior, also quoted in 1.3.
