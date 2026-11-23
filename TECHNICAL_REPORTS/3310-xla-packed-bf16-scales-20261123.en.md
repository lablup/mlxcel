# PR #3310: Accept bf16 scales and biases on the OpenXLA packed quant path

**Date**: 2026-11-23
**Status**: PR open (closes #603). Validated on ROCm gfx1151 with IREE `hip`. The CUDA run on GB10 (`MLXCEL_XLA_DEVICE=cuda`, qwen3-0.6b-4bit packed against dequant-at-load) is pending.

## Executive Summary

`MLXCEL_XLA_QUANT=packed` keeps MLX affine-quantized weights packed on the device and dequantizes them inside the StableHLO graph. Until this PR the graph declared the `scales` / `biases` arguments as f16 and the loader required f16 bytes, so every mlx-lm checkpoint that stores its affine metadata as bf16 (Qwen3, Gemma3-27B, Qwen3-MoE) failed at load: `quant part model.layers.0.mlp.down_proj.biases dtype BF16, expected F16` on qwen3-0.6b-4bit.

The loader now reads the dtype of every packed `scales` / `biases` tensor from the safetensors headers, stores it in a new `Config::quant_scale_dtype`, and that one field drives both the declared graph argument type and the dtype code of the uploaded buffer. The C shim gained code 3 (`IREE_HAL_ELEMENT_TYPE_BFLOAT_16`), and `dequant_affine` widens bf16 to f32 in the graph the same way it widens f16. A checkpoint that mixes f16 and bf16 metadata warns and dequantizes at load. `MLXCEL_XLA_QUANT=native`, whose HIP and CUDA kernels read `half`, now warns and falls back to the f16 weight path on a bf16 checkpoint instead of failing.

The security review found that the new header probe reached a pre-existing path traversal: shard names from `model.safetensors.index.json` were joined onto the model directory unchecked. The PR fixes it for the IREE loader and the Phi-4 audio loader.

## 1. Problem Statement

The packed path landed in #568 with f16 metadata only. #569 broadened the dequant-at-load path to bf16 scales, but the packed path still hard-coded `"f16"` in `take_weight` and `take_lm_head` (`emitter/model.rs`) and mapped `QuantPart::Scales | QuantPart::Biases` to `(WDT_F16, Dtype::F16)` in the `QuantRaw` upload (`iree.rs`). The issue was split out of #574 as low-priority breadth work: the #573 spike showed that IREE does not fuse the dequant into the matmul on CUDA, so packed is off by default and slower than f16-resident weights there. The point of #603 is to have the packed representation ready for more checkpoints when an upstream fused quantized matmul lands.

The core difficulty is that the emitter builds the graph from `config.json`, which does not record the metadata dtype. The type has to come from the weights, and the graph argument and the uploaded buffer must agree, or IREE receives bf16 bytes under an f16 declaration (or the reverse) and computes garbage without an error.

The native GEMV path (`MLXCEL_XLA_QUANT=native`) uses the same packed buffers but its kernels (`csrc/qmv4.hip`, `csrc/qmv4.cu`) read the scales as `half`. Passing bf16 bytes to them would be misinterpreted rather than rejected, so that path needed an explicit guard once bf16 became a legal packed input.

## 2. Change Summary

| File | Change |
|---|---|
| `emitter/quant_scale.rs` (new) | `QuantScaleDtype { F16, Bf16 }` with `elt()` (StableHLO type) and `weight_dtype_code()` (1 or 3); `resolve_quant_scale_dtype` reduces the per-tensor dtypes to one value or an error |
| `emitter/config.rs` | `Config::quant_scale_dtype`, default `F16` in both constructors |
| `emitter/model.rs` | `take_weight` and `take_lm_head` declare `scales` / `biases` with `c.quant_scale_dtype.elt()` |
| `emitter/builder.rs` | `dequant_affine` asserts a matching f16 or bf16 pair; doc and comments updated; two tests |
| `iree.rs` | `packed_scale_dtype` header probe, `native_scale_dtype_check`, `RuntimeConfig::{set_,}quant_scale_dtype`, the `QuantRaw` upload reads the config field, packed and native fallbacks, `index_shard_path` |
| `csrc/xla_iree.c` | `case 3` maps to `IREE_HAL_ELEMENT_TYPE_BFLOAT_16`, esize 2; the test byte counter counts code 3 as 2 bytes |
| `phi4_audio.rs` | `resolve_weight_shards` goes through `index_shard_path` |
| `docs/environment-variables.md`, `CHANGELOG.md` | `MLXCEL_XLA_QUANT` row describes bf16 metadata, the mixed fallback and the native fallback; Changed and Fixed entries |

### Load-time flow

On the single-sequence `IreeLlama` load, when `quant_in_graph()` is set the loader turns on `packed_args`, then calls `packed_scale_dtype(model_dir, &cfg)`. The probe takes the `QuantRaw` specs with `part: Scales | Biases` from `cfg.weight_specs()`, so it inspects exactly the tensors the loader will upload. It opens `model.safetensors` or the indexed shards, mmaps each one, parses only the header through `SafeTensors::deserialize`, and looks each name up directly and under its `language_model.` wrapped form. Each dtype maps to `F16`, `Bf16` or `None`; `resolve_quant_scale_dtype` returns the single type, the f16 default for an empty list, or an error naming the tensors for mixed or unsupported dtypes.

- `Ok(dtype)`: `cfg.set_quant_scale_dtype(dtype)`. The emitter declares that type and the upload expects `Dtype::F16` or `Dtype::BF16` and passes code 1 or 3.
- `Err`: one warning (`MLXCEL_XLA_QUANT=packed cannot keep this checkpoint packed (...); dequantizing at load instead`) and `set_packed_args(false)`. The dequant-at-load path, broadened to bf16 in #569, serves the checkpoint instead; the mixed-dtype run in section 5 shows it loading and matching.

The upload path does not re-derive anything: the `QuantRaw` branch reads `cfg.quant_scale_dtype()` for both the expected safetensors dtype and the C code, so the graph argument and the buffer come from the same field. Gemma3n returns `F16` because its weight list never carries raw quantized parts.

### In-graph dequant

`dequant_affine` already converted the metadata to f32 before `q * scale + bias`. Both f16 and bf16 widen to f32 exactly, so the bf16 form is the f16 graph with only the metadata type changed, and it matches the host bf16 dequant bit for bit. The new assertion rejects a mismatched pair (bf16 scales with f16 biases) or any other element type at emit time.

### Native fallback

Both native entry points (single-sequence and ragged) call `native_scale_dtype_check` after setting the packed flags. Anything other than uniform f16 logs `...; falling back to the f16 weight path` and clears `packed_args` (and `native_head` on the single-sequence graph). The native branch leaves `quant_scale_dtype` at its f16 default, so the upload still requires F16 bytes whenever native is active, and bf16 bytes cannot reach the kernels.

### Index shard names

`checkpoint_candidate_shards` (used by the new probe, the native head probe and the weight loader) and `resolve_weight_shards` in `phi4_audio.rs` joined each `weight_map` value onto the model directory, so `../x`, an absolute path or `nested/x` made the loader open a file outside the checkpoint. `index_shard_path` accepts a name only when it has no NUL and its components are exactly one `Component::Normal`, the rule `mlxcel-core`'s `validate_index_shards` already applies. A hostile entry now fails the load with an error naming the index and the shard; it does not fall through to the directory-scan fallback that a stale index still gets.

## 3. Technical Decisions

**Read the dtype from the safetensors header, not from `config.json` or a probe of the data.** The issue suggested threading the type the way precision is threaded through `resolve_precision`, but precision comes from the environment and the device, while the metadata dtype is a property of the weights. The header is the only authoritative source, it is already parsed by the loader, and reading it touches only header pages. The review measured 100 to 270 KB of headers on Qwen3-4B, Gemma3-12B and Qwen3.6-35B-A3B (1 to 4 shards), a few ms per load.

**One config field for both the declaration and the upload.** The pre-#603 code kept the emitter and the loader in step by both hard-coding f16. Replacing two constants with two independent derivations would have reintroduced the drift risk. Storing the result once in `Config` and reading it in `take_weight`, `take_lm_head` and the `QuantRaw` branch makes a mismatch impossible by construction.

**One scale dtype per checkpoint, mixed falls back.**

| Option | Pros | Cons |
|---|---|---|
| Per-tensor declared type | Mixed checkpoints stay packed | Graph shape depends on a per-tensor map; more MLIR variants and cache entries for a layout no known converter writes |
| Reject mixed metadata with a load error | Simple | A loadable checkpoint stops loading under a performance flag |
| **Chosen: one dtype, mixed dequantizes at load with a warning** | Graph stays a pure function of `Config`; the checkpoint still loads | Mixed checkpoints lose the packed representation |

The fallback was exercised by rewriting layer 3 `up_proj` of qwen3-0.6b-4bit to F16: the warning fired and the ids matched.

**Guard native rather than add a bf16 kernel.** Extending `qmv4.hip` / `qmv4.cu` to bf16 would be a separate kernel and performance task. The PR keeps the kernels unchanged and makes the unsupported case a warning plus the f16 weight path, which was previously a load error.

**Lock the Rust codes to the C `switch` with a source-text test.** `weight_dtype_codes_match_the_c_shim` reads `csrc/xla_iree.c` with `include_str!` and asserts the exact `case N: elt = ...; esize = 2;` text for both codes. The C shim is the only consumer of the code, and a drift would upload bf16 bytes as another element type with no runtime error. The test is sensitive to formatting changes in that `switch`, which is acceptable because it fails loudly.

**Fail closed on a hostile index.** A traversal name is treated as evidence of a hostile checkpoint rather than a stale index, so the load fails instead of scanning the directory. All 49 local checkpoint indexes use plain names, so no real checkpoint changes behavior.

## 4. Review Outcomes

**Security and performance review** (PR comment): one LOW finding, fixed in bf08e0a2. It is pre-existing but sits on the new probe's path. The new test `candidate_shards_reject_index_names_outside_the_model_dir` covers `../outside.safetensors`, an absolute path, `nested/inner.safetensors` and `./../outside.safetensors`, and fails on the previous code.

Checked with no change needed:

- Header parsing uses `SafeTensors::deserialize` on an mmap: the header is capped at 100 MB, shape and offset arithmetic is checked, and each tensor's byte span must equal its shape times the element size, BF16 included.
- `QuantRaw` upload: the dtype must equal the config's scale dtype and the rank must be 2, so the C copy length (`dims * 2` for code 3) equals the safetensors span. `xla_aux.c` rejects code 3 as unknown, and the aux path never carries packed parts.
- VMFB cache keys and the bundle fingerprint hash the MLIR text, so an f16 artifact cannot serve a bf16 graph.

**Correctness review**: no findings that required a change. qwen3-0.6b-4bit was re-run on a rebuilt binary; packed and native (fallback) both matched dequant-at-load over 32 tokens.

## 5. Tests and Verification

New tests:

- `quant_scale::tests` (6): uniform bf16 resolves bf16; empty and f16 lists keep f16; mixed metadata is rejected naming both tensors; an F32 scale is rejected; `packed_graphs_declare_the_config_scale_dtype` emits decode and prefill for both projection layouts and checks that the bf16 graph equals the f16 graph with `xf16>` replaced by `xbf16>`; `weight_dtype_codes_match_the_c_shim`.
- `builder` tests (2): bf16 metadata widens to f32 in the graph and the f16 form differs only in the metadata type; a bf16 / f16 pair panics.
- `checkpoint_name_tests::candidate_shards_reject_index_names_outside_the_model_dir`.

On main, the shim-code test and the emitter bf16-declaration test fail.

Real-model runs on gfx1151, IREE `hip`, `--temp 0 -n 32`:

| Checkpoint | Run | Result |
|---|---|---|
| qwen3-0.6b-4bit (BF16) | main, packed | load error, `expected F16` |
| qwen3-0.6b-4bit | packed vs dequant-at-load | 32/32 ids identical |
| qwen3-1.7b-4bit (BF16) | packed vs dequant-at-load | 32/32 ids identical |
| qwen3-0.6b-4bit | native | was a load error, now warns and falls back, ids identical |
| qwen3-0.6b-4bit, layer 3 `up_proj` rewritten to F16 | packed | mixed-dtype warning, dequant at load, ids identical |
| llama-3.2-1b-instruct-4bit (F16) | packed, main vs branch | ids identical; all three graphs reused main's cache entries |

The #515 gate is a token-exact greedy trajectory, and these runs are token-exact. The llama row is the evidence that f16 checkpoints emit byte-identical MLIR: the VMFB cache key hashes the MLIR text, so a cache hit against main's entries means the text did not change. `registered_fixtures_are_byte_exact` also passes unchanged.

Gates: `cargo test --profile test-fast -p mlxcel-xla --lib` 455 passed (480 with the `iree` feature, linked by hand with the ROCm IREE arguments); clippy `-D warnings` on mlxcel-xla lib and tests with and without `--features iree`; `cargo fmt`.

No throughput was measured. The PR does not claim a speedup: on gfx1151 the docs already note that host-dequantized f16 decodes faster than packed, and the benefit of packed there is about 3.5 times fewer upload bytes.

## 6. Pending Work and Limitations

- **CUDA (GB10) parity is pending.** The last acceptance item of #603 (packed against dequant-at-load on `MLXCEL_XLA_DEVICE=cuda` with qwen3-0.6b-4bit) has not been run. Only the `hip` target is verified.
- **Native kernels stay f16-only.** bf16 checkpoints lose the native GEMV speedup and run the f16 weight path. A bf16 variant of `qmv4.hip` / `qmv4.cu` would be a separate change.
- **Mixed-metadata checkpoints are not packed.** No known converter writes them; they dequantize at load.
- **The ragged batch graph has no plain packed path.** Only its native branch got the guard; this PR does not add packed support there.
- **Packed is still off by default** and remains bound by #574 (upstream fused quantized matmul in IREE).
- **The source-text shim test** must be updated if the `switch` in `xla_iree.c` is reformatted.

## 7. Learning Points

- When a graph argument type and a host buffer type must agree, derive the type once and store it where both sides read it. Two correct derivations can still drift; one stored value cannot.
- bf16 to f32 is an exact widening (bf16 is the upper half of an f32), so moving the widen from the host into the graph keeps results bit-identical. This is why the token-exact gate holds without tolerance.
- A kernel that reinterprets bytes (`half` reads of bf16 data) fails silently, so the guard has to sit before the bytes reach it, and the default of the shared field (f16) has to be what the restricted consumer accepts.
- A cache keyed on generated source text gives a free regression check: if an unrelated configuration reuses the old cache entries, its generated source is unchanged.
- A new reader of untrusted input can make an old unchecked path reachable from a new place. Reviewing the helpers a new probe calls, not only the probe, is what found the index traversal.
