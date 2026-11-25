# PR #3370: OpenXLA Mamba2 config, mixer schedule and state-threaded graphs

**Date**: 2026-11-24
**Status**: PR open (closes #3338, epic #564 Phase 1 unit U2). Validated on ROCm gfx1151 with IREE `local-task` and `hip`. Not reachable from `generate`: the shim state buffers and the runtime are #3339, ragged decode with per-slot state is #3340.

## Executive Summary

The OpenXLA/IREE emitter could express one Mamba2 mixer layer after #3360, but no whole model: `Config::from_json` rejected `mamba2`, and every graph assumed every layer is attention with a KV cache. This PR adds a per-layer mixer schedule (`MixerKind`), a generic threaded-state ABI (`StateTensorSpec`), the `backbone.*` weight specs in lock-step with a new `take_ssm_layer_weights`, and prefill and single-sequence decode graphs that take and return `ssm_h [L,H,P,N]` and `ssm_conv [L,K-1,conv_dim]` with no KV cache, RoPE table or mask. Dense and MoE graphs stay byte-identical.

Against HF `Mamba2ForCausalLM` in fp32, `AntonV/mamba2-130m-hf` (200 tokens, two scan chunks) matches to 2.0e-4 max-abs logits on `local-task` and 3.9e-4 on `hip`, with the same argmax, across three paths: one prefill, two state-carrying prefill calls, and token-by-token decode. Prefill and decode compile in 1.6 to 2.6 s on `llvm-cpu` and `rocm` gfx1151 at capacities 256 and 4096. At f16 the logits drift 0.42 to 0.45 from fp32 with argmax and top 5 unchanged; this is measured and traced to projection rounding, not gated.

Because `config.json` and the safetensors headers are now untrusted inputs that size graphs and state, the PR cross-checks every SSM dim against the checkpoint, requires each header entry to be backed by its exact byte span in the file, bounds the per-layer state by the `in_proj` element count, and caps `num_hidden_layers` at 1024 for every config.

## 1. Problem Statement

#3360 (#3337) delivered `SsmConfig`, the conv and scan primitives, and `mamba2_mixer_prefill` / `mamba2_mixer_step`, checked one layer at a time through a probe graph. Turning that into a model needed four things the emitter did not have:

- **A schedule.** `Config` described a uniform attention stack. Mamba2 layers are `norm -> mixer -> residual` with no FFN, and later hybrids in epic #564 (Nemotron-H, Jamba, Granite 4) mix attention and recurrent layers, so the choice has to be per layer.
- **Attention-free configs.** Shared code read `n_q / n_kv` (`Config::group`), built RoPE tables from `head_dim`, built causal masks, and allocated `[L, S, n_kv, d]` caches unconditionally. With zero heads these divide by zero or emit zero-width tensors.
- **A state contract.** Prefill and decode had a fixed KV ABI. Recurrent state needs its own tensors with their own shapes, carried through prefill results into decode, and the ABI has to stay generic enough for later families (the orchestrator decision recorded in #3338: a list of threaded state tensors, KV optional, no dummy KV).
- **Untrusted dims.** #3360's `SsmConfig::validate` only rules out overflow and non-finite values. Once the dims come from a model directory, `config.json` could ask for state and graph sizes far beyond the weights it ships with. The carry-over comment on #3338 asked for a cross-check against the checkpoint tensor shapes and a host-side `real_len` bound.

The carry-over also listed two #3360 limitations to settle here: the f16 prefill conv rounded differently from the decode step, and the chunked-SSD carry costs `H*(T/Q)^3` adds, which matters only for very long context buckets.

## 2. Change Summary

| File | Change |
|---|---|
| `emitter/recurrent_config.rs` (new) | `MixerKind { Attention, Mamba2 }`, `StateElem`, `StateTensorSpec`; `Config::{layer_mixer, layer_has_ffn, recurrent_layer_count, state_layer_index, has_kv_cache, state_tensors, has_recurrent_state, reject_recurrent}`; `mamba2_from_json`; `quote_non_finite_json` for HF's bare `Infinity` |
| `emitter/recurrent_checkpoint.rs` (new) | `Config::validate_recurrent_checkpoint`: safetensors header reader with exact byte-span checks, per-tensor shape cross-check, state bound |
| `emitter/model_recurrent.rs` (new, child of `model`) | `RecurrentLayers` (split stacked state into per-layer rows and join it back), `take_ssm_layer_weights`, `emit_mamba2_block`, `take_state_args`, `load_state_globals`, `decode_state_entries` (`@ssm_h` / `@ssm_conv` globals and `@state_import`) |
| `emitter/config.rs` | `ssm` and `layer_mixers` fields, `WeightScheme::Mamba2`, `MAX_CONFIG_LAYERS = 1024` and `check_layer_count` (dense parser too), `from_json` runs the checkpoint cross-check, `group()` returns 0 for zero KV heads, `supports_packed_quant` false for SSM |
| `emitter/model.rs` | `emit_transformer_layer` branches to the Mamba2 block; RoPE tables become `Option`; decode drops `@kcache` / `@vcache` / `@kv_import` when there is no attention; prefill threads state inputs after `real_len` and returns them after the logits; `emit_decode_buffer_abi`; batched, ragged and every `prefill_embeddings*` entry return `Result` and refuse recurrent configs |
| `emitter/ssm.rs` | prefill conv rewritten as `K` shifted f32 multiply-adds |
| `weights.rs`, `weight_names.rs` | `push_ssm_layer_specs` in `take_ssm_layer_weights` order; Mamba2 scheme names |
| `iree.rs`, `shard_path.rs` (new) | `whole_graph_shape` (MLX `[C,K,1]` conv uploaded as `[C,1,K]`), `load_host_weights` through the production loader; `index_shard_path` moved to `shard_path.rs` so the emitter builds without `iree`; callers propagate the new `Result`s |
| `ssm_model_check.rs` (new), `examples/xla_ssm_model_check.rs`, `Cargo.toml`, `auxiliary.rs` | whole-model runner over the auxiliary IREE ABI: prefill, split prefill, decode session, compile timing |
| `validation.rs`, `emitter/mod.rs`, `emitter/context_tests.rs` | `GraphKind::emit` returns `Result`; call sites `.unwrap()` the now-fallible emitters |
| `emitter/recurrent_tests.rs` (new) | 16 tests (parse, ABI shape, weight order, refusals, cross-check hardening) |
| `spike/openxla/ssm_model_check.py`, `FINDINGS_ssm.md`, `README.md`, `CHANGELOG.md` | HF fp32 oracle with a seeded synthetic model, results, compile times, carry-over resolutions |

23 files, +3082 / -176 (before the merge of `origin/main`).

### Schedule and ABI

Every existing parser sets `ssm = None` and `layer_mixers = [Attention; n_layers]`, so `state_tensors()` is empty, `has_kv_cache()` is true and `layer_has_ffn(li)` is true everywhere. All new branches key on those queries, which is why the dense goldens do not move. `mamba2_from_json` builds the config from HF keys (`num_heads`, `head_dim`, `state_size` or `ssm_state_size`, `n_groups`, `conv_kernel`, `expand`, `use_conv_bias`, `use_bias`, `time_step_limit`, `layer_norm_epsilon`), sets attention fields to zero, `norm_group_size = d_inner`, `chunk_size = 128` from #3360, and refuses `rms_norm = false`, a non-silu `hidden_act`, and `num_heads * head_dim != expand * hidden_size`.

For Mamba2, `state_tensors()` is `ssm_h [L_ssm, H, P, N]` then `ssm_conv [L_ssm, K-1, conv_dim]`, both f32, where `L_ssm` counts recurrent layers and `state_layer_index(li)` gives a layer's row.

| Graph | State contract |
|---|---|
| Token prefill (`prefill.main`) | Inputs: existing inputs, then `ssm_h`, `ssm_conv` after `real_len`. Results: logits, then the KV caches only if `has_kv_cache()`, then the state tensors. Zeros start a fresh sequence. |
| Single-sequence decode (`@decode_step`) | `util.global private mutable @ssm_h` / `@ssm_conv`, seeded by `@state_import(%s0, %s1)`. `@main` keeps `token`, `pos`, `cache_len` but reads only the token. `@kcache`, `@vcache`, `@kv_import` absent. |
| Uniform batched, ragged, embeddings prefill (incl. DeepStack) | `Err` naming the model type, until #3340 |

Inside a graph, `RecurrentLayers::split` slices each layer's row out of the stacked tensors (unbatched for prefill, batch of one for decode), `emit_mamba2_block` runs `x + mixer(norm(x))` and writes the advanced row back, and `join` restacks the rows for the results or the global stores.

### Weight path

`push_ssm_layer_specs` lists per layer: `norm.weight`, `mixer.in_proj.weight` (+ bias), `conv1d.weight` (+ bias), `dt_bias`, `A_log`, `D`, `norm.weight`, `out_proj.weight` (+ bias). `take_ssm_layer_weights` consumes them in the same order into #3360's `Mamba2MixerW`, and a test compares the two orders. Projections go through `push_proj`, so the f16-resident path lines up with `take_weight`. Two loader details: an MLX `[C,K,1]` conv weight is declared and uploaded as torch's `[C,1,K]` (same row-major bytes), and packed / native quant is disabled for SSM configs because the mixer's projections use the sequence helpers, which have no single-row native path.

### Checkpoint cross-check

`Config::from_json` now calls `validate_recurrent_checkpoint(model_dir)`, a no-op for attention-only configs. For Mamba2 it reads the safetensors headers (single file or index) and checks, before anything is emitted:

- every layer's `in_proj [in_proj_dim, hidden]`, `out_proj`, `conv1d` (`[conv_dim, 1, K]` or MLX `[conv_dim, K, 1]`), `dt_bias` / `A_log` / `D [H]`, gated-norm weight `[d_inner]`, the embeddings and final norm, and `lm_head.weight` when untied, with quantized projections checked at their packed width with checked arithmetic;
- each header entry spans exactly `product(shape) * dtype` bytes inside the file's data section, so a small file cannot vouch for large dims;
- the per-layer state `H*P*N` does not exceed that layer's `in_proj` element count (`H*P` and `G*N` were bounded separately, their product was not);
- index shard names are plain file names (`index_shard_path`), deduplicated with a set.

## 3. Technical Decisions

**Generic state list, KV optional, no dummy KV.** This was decided at the epic level and spelled out in #3338. The alternatives were a Mamba2-specific ABI (cheaper now, a second ABI for every later family) or keeping a zero-size KV cache for attention-free models (one ABI, but the shim would allocate and thread buffers no layer reads). The list form means Nemotron-H or Jamba add tensors to `state_tensors()` without changing the prefill and decode plumbing; the cost is that the shim (#3339) must handle `has_kv = 0` and `n_state > 0` as distinct cases.

**Stacked `[L_ssm, ...]` state tensors, not one tensor per layer.** Two arguments per graph regardless of depth keep the graph signature, and the `XLA_MAX_STATE = 8` bound #3338 specifies for the shim, independent of `n_layers`. The graph pays a slice and an update per layer, which is negligible against the mixer.

**Branch on queries, not on the family.** Shared code asks `has_kv_cache()`, `layer_mixer(li).is_recurrent()` and `state_tensors()` instead of testing `weight_scheme == Mamba2`. RoPE tables became `Option<Val>` with accessors that panic only on an attention path, so an attention-free config cannot silently read a missing table. A hybrid then needs no new branches in prefill or decode, only a schedule with both kinds.

**Refuse graphs without a state contract.** The uniform batched decode, the ragged decode and every embeddings prefill entry now return `Result` and refuse recurrent configs with the model type in the message, instead of emitting a graph that silently drops the state. This changed the signatures of public emit functions and every test call site; the alternative (a panic) would have taken down `compile_vmfbs` for a config that only needs the entry skipped. #3339 must skip `prefill_embeddings` for recurrent configs, as the implementation review noted.

**Decode keeps `token`, `pos`, `cache_len`.** Mamba2 reads only the token. Keeping the three scalars keeps one `@main` calling convention (scalar ABI from #3060 or buffer ABI) for the shim, at the cost of two unread arguments.

**f32 prefill conv as `K` shifted multiply-adds.** #3360 recorded that the f16 prefill conv rounded its operands to f16 while the decode conv accumulated in f32, and that a mixed-type depthwise convolution does not compile in IREE. With `K = 4` the shifted form is four slices, multiplies and adds, on every precision path, so prefill and decode now do the same arithmetic. This changes the SSM probe graphs from #3360 but no dense golden.

**HF tie default.** `tie_word_embeddings` defaults to false when absent, as HF `Mamba2Config` does. The first revision defaulted to true, which would have silently used the embeddings as the head for a checkpoint that ships its own `lm_head.weight` (found in the implementation review, MEDIUM). The cross-check then requires `lm_head.weight` for an untied config.

**Model check through the production loader and the `InOut` layout.** The check loads weights with `load_host_weights`, which wraps the runtime's `upload_weights_with_sink`, so the MLX conv reinterpretation and the f16-resident path are exercised as #3339 will run them. It defaults to `InOut` because `IreeLlama::load` always uses that layout; the first revision ran the config's `OutIn`. Both give the same numbers.

**Bounded chunk carry deferred.** The carry costs `H * nc^3` adds against within-chunk work of order `H * T * Q * (P + N)`. At the largest ladder capacity, 32768 (`nc = 256`), that is about 2% for mamba2-130m, with the crossover near T = 185k to 230k. FINDINGS records "revisit if a context bucket beyond about 64k is enabled".

**f16 measured, not gated.** The 3e-2 gate fails at f16 by an order of magnitude (0.42 to 0.45 on logits spanning -3.4 to 16.7) while argmax and top 5 match. Rounding every `nn.Linear` input and weight to f16 inside HF's own fp32 forward reproduces 0.44 (0.39 from `in_proj` alone, 0.40 with the `dt` columns kept f32), so the error enters through `x`, `B`, `C` and `z`, and no graph-side change would remove it. In that simulation 40 greedy tokens on five prompts match fp32. Whether #3339 pins f32 for recurrent configs (as `effective_precision` does for MLA) is left to its token-level gate.

## 4. Review Outcomes

**Implementation review** (e17b0879):

| Finding | Severity | Status |
|---|---|---|
| `tie_word_embeddings` defaulted to true when omitted; HF defaults to false | MEDIUM | Fixed, regression test |
| The model check ran `OutIn`; `IreeLlama` always uses `InOut` | MEDIUM | Fixed: `--layout`, default `inout`; numbers unchanged |
| A header dim multiplied by 32 unchecked in the packed-width test; `2^59 + 2` wrapped to the expected width | LOW | Fixed: checked multiply, test |
| HF torch path and the MLX loader normalize the gated output over all of `d_inner`; `mamba_ssm` normalizes per group when `n_groups > 1` | MEDIUM | Open: verify before enabling an `n_groups > 1` checkpoint such as Codestral Mamba |
| `@state_import` is compiled but never executed here | note | Open: #3339 gate 1 is its first execution |
| `rms_norm = false` refused although HF ignores the flag | LOW | Open, conservative |

**Security and performance review** (d9cb7b80). Trust boundary: an untrusted model directory.

| Finding | Severity | Status |
|---|---|---|
| `num_hidden_layers` sized `layer_mixers` before any check | MEDIUM | Fixed: capped at 1024 at parse time, Mamba2 and dense |
| Header shapes trusted without checking them against the data section; `compile_ssm_model_graphs` never loads weights, so it compiled at the claimed dims | MEDIUM | Fixed: exact `product(shape) * dtype` span required |
| Per-layer state `H*P*N` unbounded | MEDIUM | Fixed: bounded by the layer's `in_proj` elements |
| Local shard-name check let `..`, `/` and NUL through; quadratic dedup | LOW | Fixed: shared `index_shard_path`, set |
| Python check loaded checkpoints and `.npy` permissively | LOW | Fixed: `use_safetensors=True`, `allow_pickle=False` |

Checked without change: the conv reinterpretation keeps the element count, `SafeTensors::deserialize` enforces byte length, token ids and prompt length are checked on the host, `--compile-times` writes only to the caller's directory, the Python runner has no `trust_remote_code` and passes subprocess arguments as a list. Each fix has a unit test that fails without it; the finalize commit (08ed5b36) confirmed that reverting the header-span check and the layer cap each fails its test.

**Carry-over from #3360**, as resolved in FINDINGS: dims are cross-checked (above); `real_len` is already refused outside `[0, capacity]` by the runtime prefill and the shim, and the check runner refuses empty, over-capacity and out-of-vocabulary prompts; the bounded carry is deferred (section 3); the f16 conv gap is closed (section 3).

## 5. Tests and Verification

`spike/openxla/ssm_model_check.py` with `examples/xla_ssm_model_check.rs`, last-token logits against HF `Mamba2ForCausalLM` fp32 (transformers slow path), tolerance 3e-2 max-abs plus equal argmax:

| model | device | precision | whole | split | decode | argmax |
|-------|--------|-----------|------:|------:|-------:|--------|
| synthetic (2 layers, H=8, G=2, biased, untied), 40 tokens, capacity 128 | local-task | f32 | 1.7e-6 | 1.3e-6 | 1.7e-6 | same |
| same | hip | f32 | 1.8e-6 | 1.2e-6 | 1.0e-6 | same |
| `AntonV/mamba2-130m-hf` @ `05e8773f`, 200 tokens (two chunks), capacity 256 | local-task | f32 | 2.0e-4 | 1.7e-4 | 1.1e-4 | same |
| same | hip | f32 | 3.5e-4 | 3.9e-4 | 3.3e-4 | same |
| same | local-task | f16 | 4.3e-1 | 4.5e-1 | 3.7e-1 | same (top 5 same) |
| same | hip | f16 | 4.2e-1 | 4.4e-1 | 4.3e-1 | same (top 5 same) |

Within f16 the three paths agree more closely than f16 agrees with fp32: whole against decode 6.6e-2 (local-task) and 4.2e-2 (hip), whole against split 3.8e-2 and 5.8e-2. The f16 hip run fails the gate as expected, which also shows the runner fails on a bad result.

Compile time of one uncached `iree-compile` for mamba2-130m (24 layers), f32:

| target | capacity | prefill | decode |
|--------|---------:|--------:|-------:|
| llvm-cpu (local-task) | 256 | 2.5 s | 1.7 s |
| llvm-cpu (local-task) | 4096 | 2.6 s | 1.7 s |
| rocm gfx1151 (hip) | 256 | 2.5 s | 1.6 s |
| rocm gfx1151 (hip) | 4096 | 2.6 s | 1.6 s |

Unit tests: 16 in `emitter/recurrent_tests.rs` (config parse and state ABI, attention configs unchanged, inconsistent dims, every `Infinity` spelling, the JSON quoting helper, `weight_specs_q` order equal to `take_ssm_layer_weights` order, prefill and decode signatures, refusal by every graph without a state contract, emission at every precision, the layer cap, the header span, index shard names, the state bound, every SSM dim, the tie default), 2 in `recurrent_checkpoint.rs` and 1 in `shard_path.rs`. `cargo test --profile test-fast -p mlxcel-xla --lib`: 506 passed on the PR head before the merge of `origin/main`. Also `scripts/xla/validate_arch.sh --structural-only` (dense goldens byte-identical), `cargo fmt --all -- --check`, and clippy `-D warnings` with no features, with `iree`, and on the example. After merging `origin/main` (no conflicts), `cargo fmt -p mlxcel-xla -- --check` passes.

## 6. Pending Work and Limitations

- **Not integrated.** No runtime path calls these graphs. #3339 adds the shim state buffers (`has_kv`, `n_state`, `state_desc`), `RuntimeConfig` state fields, the vmfb cache key and fingerprint over the state layout, and the generate path; #3340 adds per-slot state to the ragged decode. Until then the CHANGELOG states the feature is not reachable from `generate`.
- **`@state_import` has never executed.** The model check uses a single-entry auxiliary module that cannot call it, so the decode check rewrites the state globals with zero initializers (`zero_initialize_state_globals`) and runs the buffer-ABI decode (`emit_decode_buffer_abi`). The prefill-to-decode handoff and the scalar-ABI decode are compiled but first run in #3339.
- **`compile_vmfbs` now gets `Err` from `prefill_embeddings` for Mamba2** and must skip that entry for recurrent configs in #3339.
- **Per-group gated norm for `n_groups > 1`** (implementation review, MEDIUM, open). The emitter follows HF's torch path and MLX (`norm_group_size = d_inner`). The synthetic model has `G = 2` but is checked against the same HF path, so it does not discriminate. Verify against `mamba_ssm` before enabling a checkpoint such as Codestral Mamba.
- **f16 drift.** `hip` defaults to f16, which is what #3339's token-exact gate will run at. The simulation suggests low risk; pinning f32 for recurrent configs is the fallback.
- **Bounded chunk carry deferred** until a context bucket beyond about 64k is enabled.
- **`rms_norm = false` is refused** although HF ignores the flag.
- **The #3360 probe gate is not reported re-run** after the prefill conv rewrite; the PR's test plan covers the conv through the whole-model check (f32, 1.0e-6 to 1.8e-6 on the synthetic model) and the unit tests.
- **Only `local-task` and `hip` (gfx1151) were run.** CUDA was not measured. Only one real checkpoint (mamba2-130m, `n_groups = 1`) was checked.

## 7. Learning Points

- Make a schedule the single source of layer structure and let shared code ask questions of it (`has_kv_cache`, `is_recurrent`, `state_tensors`). With every existing config defaulting to the old answers, byte-identical goldens fall out, and hybrids need data rather than new branches.
- A fallible emitter is a better refusal than a panic or a degraded graph: the caller decides whether to skip the entry or fail, and the message names the model type.
- A safetensors header is a claim, not evidence. Shapes from a header only bound a config when each entry is backed by its exact byte span in the file; otherwise a few-KB file can request arbitrary graph sizes.
- Bound products of untrusted dims, not just the factors. `H*P` and `G*N` were each tied to a tensor shape, but the state `H*P*N` was not until it was tied to the `in_proj` element count.
- When a precision gate fails, reproduce the error inside the reference before changing the graph. Rounding HF's own `nn.Linear` inputs to f16 reproduced the drift, located it in `in_proj`, and showed no graph-side fix exists.
- Check the configuration the runtime will actually use. The model check first ran the config's projection layout, not the `InOut` layout `IreeLlama` emits; the numbers happened to match, but the check would not have covered #3339's path.
