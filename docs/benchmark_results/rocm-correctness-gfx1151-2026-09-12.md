# ROCm correctness matrix: Radeon 8060S (gfx1151), 2026-09-12

Validation run for issue #1809. It answers one question: does mlxcel on the experimental ROCm backend decode the same text as mlxcel on Metal, for the same checkpoint and the same input?

The answer for the twelve model x width pairs measured here is yes. The two backends disagree on the top-1 token at 114 of 5120 positions, and the largest reference gap at which they ever disagree is 1.125 logits: everywhere the reference model preferred its own choice by more than that, ROCm made the same choice. 100 of the 114 disagreements pick the reference's rank-2 token.

## Environment

| Field | ROCm side | Metal reference |
|---|---|---|
| Hardware | AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5, 40 CUs), 96 GiB VRAM carve-out, 31 GiB host RAM | Apple M1 Ultra, 128 GB unified memory |
| OS | Debian GNU/Linux 13 (trixie), kernel 6.18.12+deb13-amd64 | macOS 27.0 (26A428) |
| Backend | ROCm 10.0.0 (HIP 7.15.26333, AMD clang 23), `--features rocm` | Metal, `--features metal,accelerate` |
| mlxcel | `bec64748` | `bec64748` |
| MLX pin | `81ba1c6a` | `81ba1c6a` |
| Traces | `benchmarks/logit_traces/rocm_gfx1151_bec64748/` | `benchmarks/logit_traces/metal_m1u_bec64748/` |

The Mixtral ROCm traces were produced by a `logit_trace` built from `bec64748` plus one change: the f16 `gather_qmm` dispatch from #1822, which lives entirely in `src/lib/mlx-cpp/patches-rocm/` and touches no mlxcel Rust or bridge code. Without it this f16 checkpoint spends about 205 ms per MoE `gather_qmm` call and the traces would take hours instead of minutes. The numerics recorded are those of the kernel a user of the merged fix runs. Every other ROCm trace and the whole Metal side are plain `bec64748`.

## Method

`examples/logit_trace` teacher-forces a fixed corpus through the model and writes, per position, the top-k token ids and their logits. `scripts/compare_logit_traces.py` compares two such traces.

- Corpus: `tests/fixtures/wikitext2_excerpt.txt` (sha256 `16721fc1...`), 299190 tokens.
- Widths, as `CHUNK_TOKENS MAX_CHUNKS TOPK PREFILL`: `w1` = `1 128 8 0` (pure decode), `w8` = `8 80 8 512` (batched decode after a prefill), `w256` = `256 2 8 0` (prefill-shaped).
- Checkpoints: the same four affine 4-bit checkpoints at the same Hugging Face revisions on both backends.
- `qwen3-30b-a3b` and `mixtral-8x7b-instruct` ran with `MLXCEL_FUSED_MOE=0` on both sides, because the fused MoE kernel has no ROCm port (#1803). Their Metal counterparts are the `fused0` traces, not the default ones.

### What has to match

Four things must be identical for a comparison to mean anything, and they matter more than the threshold discussed below: the same mlxcel commit, the same corpus file, the same trace arguments, and the same checkpoint revision. `METADATA.txt` on both sides records all four so a reader can check rather than trust. A pair that differs in any of them is measuring two models, not two backends.

### Why byte-identity is not the criterion here

Byte-identical logits remain the strongest available signal wherever they are achievable, and they are used that way: the Metal regression gate on #1818 and #1819 confirmed the Metal path was unchanged by byte-identity on one M1 Ultra. This section narrows where that criterion applies, it does not discard it.

It does not apply across backends. Two accelerators with different FMA ordering, different reduction trees and different fast-math contractions do not produce identical floats for the same matmul. It does not even hold within one backend across hardware generations: `examples/logit_trace` documents that byte-identity breaks on Apple GPU families 15 and later for reasons the caller does not choose. What is left to measure is whether the difference ever changes the token the model emits.

The criterion used here is the decided-position disagreement count. A position is decided when the reference's top-1 logit leads its top-2 by at least `--decided`, that is, when the reference model is not itself choosing between near-ties. At a decided position a backend that picks a different token is wrong. At an undecided position the two backends are splitting a coin flip that the checkpoint left unresolved, and either answer is as defensible as the other.

### The threshold, and what it does and does not decide

`--decided 2.0` means the reference preferred its top-1 by a factor of about 7.4. That is a chosen line, so here is the same matrix at three thresholds, as `disagreements / decided positions`:

| Model | Variant | Width | gap >= 0.5 | gap >= 1.0 | gap >= 2.0 |
|---|---|---|---|---|---|
| qwen3-0.6b | default | w1 | 0 / 43 | 0 / 16 | 0 / 0 |
| qwen3-0.6b | default | w8 | 0 / 468 | 0 / 333 | 0 / 199 |
| qwen3-0.6b | default | w256 | 0 / 357 | 0 / 254 | 0 / 138 |
| llama-3.1-8b-instruct | default | w1 | 0 / 40 | 0 / 8 | 0 / 0 |
| llama-3.1-8b-instruct | default | w8 | 0 / 484 | 0 / 354 | 0 / 230 |
| llama-3.1-8b-instruct | default | w256 | 0 / 371 | 0 / 277 | 0 / 176 |
| qwen3-30b-a3b | fused0 | w1 | 1 / 110 | 0 / 92 | 0 / 70 |
| qwen3-30b-a3b | fused0 | w8 | 2 / 519 | 1 / 435 | 0 / 319 |
| qwen3-30b-a3b | fused0 | w256 | 1 / 413 | 0 / 350 | 0 / 248 |
| mixtral-8x7b-instruct | fused0 | w1 | 0 / 82 | 0 / 49 | 0 / 17 |
| mixtral-8x7b-instruct | fused0 | w8 | 0 / 516 | 0 / 434 | 0 / 345 |
| mixtral-8x7b-instruct | fused0 | w256 | 0 / 397 | 0 / 332 | 0 / 245 |

For eleven of the twelve pairs the threshold changes nothing: zero at 0.5, zero at 1.0, zero at 2.0. For `qwen3-30b-a3b` it does. That model has four disagreements above a 0.5 gap and one above 1.0, so the zero at 2.0 is a statement about where its line was drawn, not an unconditional zero.

The honest way to state the bound is without a threshold at all: across all twelve pairs, the largest reference gap at which the two backends disagree is 1.125 logits, in `qwen3-30b-a3b` `w8`, where the Metal token is ROCm's rank 2. The next-largest is 0.75 in `qwen3-30b-a3b` `w1`, the one rank-8 case. Every other disagreement sits under a 0.5 gap. So 2.0 clears the observed maximum by a comfortable margin, and any threshold at or above 1.2 gives the same all-zero result.

`qwen3-30b-a3b` being the only model to show this is consistent with it being the 128-expert MoE in the set: expert routing is a discrete decision taken on small score differences, so a numerical difference there moves more than the same difference in a dense model.

A same-backend control says the same thing. The Metal trace set includes both `default` (fused MoE) and `fused0` (`gather_qmm`) runs for the two MoE models, which is a kernel swap with no hardware change. Five of those six pairs agree at every position; the one that moves is `qwen3-30b-a3b` at `w1`, with one top-1 disagreement out of 128 and zero on its 74 decided positions. The model that reacts to a kernel change within Metal is the same model that reacts to a backend change, which is what you would expect if routing, not arithmetic, is the sensitive part. That one disagreement sits at a reference gap of 0.062, against 1.125 across backends, so a same-hardware kernel swap moves the decision only at gaps an order of magnitude smaller than a backend change does.

## Results

Ran with `python3 scripts/compare_logit_traces.py <metal.tsv> <rocm.tsv> --decided 2.0`.

| Model | Variant | Width | Top-1 disagreements | Decided positions disagreeing | Worst rank of the Metal choice on ROCm | Perplexity delta |
|---|---|---|---|---|---|---|
| qwen3-0.6b | default | w1 | 22 / 128 | 0 / 0 | rank 3 | -4.258% |
| qwen3-0.6b | default | w8 | 20 / 640 | 0 / 199 | rank 3 | -0.725% |
| qwen3-0.6b | default | w256 | 9 / 512 | 0 / 138 | rank 4 | +0.498% |
| llama-3.1-8b-instruct | default | w1 | 3 / 128 | 0 / 0 | rank 2 | +0.029% |
| llama-3.1-8b-instruct | default | w8 | 0 / 640 | 0 / 230 | all agree | -0.028% |
| llama-3.1-8b-instruct | default | w256 | 1 / 512 | 0 / 176 | rank 2 | +0.043% |
| qwen3-30b-a3b | fused0 | w1 | 5 / 128 | 0 / 70 | rank 8 | +2.428% |
| qwen3-30b-a3b | fused0 | w8 | 24 / 640 | 0 / 319 | rank 3 | +0.298% |
| qwen3-30b-a3b | fused0 | w256 | 26 / 512 | 0 / 248 | beyond top-8 | -0.463% |
| mixtral-8x7b-instruct | fused0 | w1 | 0 / 128 | 0 / 17 | all agree | -0.087% |
| mixtral-8x7b-instruct | fused0 | w8 | 2 / 640 | 0 / 345 | rank 2 | +0.101% |
| mixtral-8x7b-instruct | fused0 | w256 | 2 / 512 | 0 / 245 | rank 2 | +0.067% |

The "decided positions disagreeing" column reads as `disagreements / decided positions`, so `0 / 0` and `0 / 319` are different statements. The two `0 / 0` rows are the short `w1` runs at 128 positions, where no position in the reference had a 2.0 gap; they carry no evidence either way, which is why the comparison script now reports `inconclusive` for them instead of a pass. The `w8` and `w256` rows on the same models cover the same checkpoints with 138 to 345 decided positions each, so no model is left unmeasured.

### Where the disagreements land

Across all twelve pairs there are 114 top-1 disagreements, distributed by the rank the Metal choice holds in the ROCm top-8:

| Rank of the Metal token on ROCm | Count |
|---|---|
| 2 | 100 |
| 3 | 11 |
| 4 | 1 |
| 8 | 1 |
| beyond top-8 | 1 |

87.7% are a straight swap of two adjacent candidates. The single beyond-top-8 case is in `qwen3-30b-a3b` `w256`, an undecided prefill position in a 128-expert MoE where routing noise moves more than the logit does.

Perplexity deltas stay inside +/- 0.5% except for the two shortest runs (`qwen3-0.6b` `w1` at -4.3% over 128 positions, `qwen3-30b-a3b` `w1` at +2.4% over 128), where the sample is too small for the figure to mean anything. The sign is not consistent across pairs, which is what an unbiased numerical difference looks like.

### No noise floor was measured

An earlier plan was to run the same trace twice on one backend to establish a same-device noise floor, then require the cross-backend difference to sit inside it. That is unnecessary here. The threshold sweep above already shows where the disagreements stop, and it stops at a specific measured value (a 1.125 gap) rather than at a line borrowed from same-device noise. A floor would answer a question this data already answers directly.

## Serving on ROCm

`mlxcel-server` was built from the merged `1206f863` (`make release-rocm`) and run on this host for one dense and one MoE checkpoint, one at a time so neither competed for the GPU. `scripts/server_chat_smoke.sh` is the check, so the result is reproducible rather than a transcript of hand-typed curl calls.

| Check | `Meta-Llama-3.1-8B-Instruct-4bit` | `Qwen3-30B-A3B-4bit`, `MLXCEL_FUSED_MOE=0` |
|---|---|---|
| `GET /health` | 200, `{"status":"ok"}` | 200, `{"status":"ok"}` |
| `GET /v1/models` | 200, the loaded id | 200, the loaded id |
| `POST /v1/chat/completions`, non-streaming | 200, `finish_reason: stop`, content `Paris.` (0.46 s) | 200, `finish_reason: stop`, content `Paris.` after 205 tokens (3.51 s) |
| `POST /v1/chat/completions`, streaming | 200, 12 SSE frames closed by `[DONE]`, assembled `One, two, three, four, five.` | 200, 221 SSE frames closed by `[DONE]`, `finish_reason: stop` |
| Script verdict | `OK` | `OK` |

Both models answered coherently, the streaming and non-streaming paths agreed, and neither server log carried an error, a panic or a terminate. The MoE model runs through `gather_qmm` because the fused MoE kernel has no ROCm port (#1803).

Two things about the check are worth recording, because both look like backend faults and are not. `model` is a required field on the request body, and omitting it returns 422 with a deserialization message. And a thinking model fills `reasoning_content` before `content`, so a budget too small to close the thinking block yields `finish_reason: length` with empty `content`: `Qwen3-0.6B-4bit` does this even at 2048 tokens. The script reads both channels and names which one carried the text, the same answer `scripts/ab_output_equality.sh` reaches with `--show-reasoning`.

## The test gate alongside the matrix

`make verify-test-rocm` runs the workspace suite against a ROCm build. It is the broader of the two checks and it is not green: the missing primitives in #1825 throw through the cxx bridge, which ends in `std::terminate`, so a binary that reaches one aborts rather than reporting a failure. Run at the same commit as the traces, with aborting tests skipped so the rest of each binary can finish:

| Target | Result | What is left |
|---|---|---|
| 112 other test binaries | 1129 passed, 1 failed | The failure is `sampling_rejection_kill_switch`, which asserts the decline message names a GPU backend and does not know about ROCm (#1805) |
| `mlxcel` lib | 8200 passed, 1 failed, 3 aborted | `FFT` (#1825) and one `[cuda_kernel] No CUDA back-end` path (#1803) |
| `mlxcel-core` lib | 1434 passed, 2 failed, 46 aborted | `Hadamard` and `SearchSorted` (#1825), `[cuda_kernel]` routing (#1803), compute-capability reporting (#1805), and two nvfp4 group-16 aborts (#1806) |
| `mlxcel-surgery`, `mlxcel-xla` libs | 143 and 280 passed, 0 failed | Green after the rpath fix below |

Nothing in this list is a wrong-numbers defect. Every remaining item is a missing implementation or a backend-identification gap with an issue against it.

## Defects this matrix did not find

Three real defects were found during this work by the test gate, not by the traces, and all are fixed:

- #1823: the tiled qmv path was entered for any activation dtype but launched a kernel only for bf16 and f16, so f32 activations returned uninitialized memory. The traces run in bf16 and f16 and never touched it.
- #1824: `Scatter` passed `upd_post_idx_size` as `int32_t` to a kernel parameter declared `int64_t`, so every update landed on index 0, breaking `eye`, `identity`, `tri`, `diag` and `.at[].add`. None of these is on the decode path the traces exercise. This bug is present in the upstream fork as well and is a candidate to send back through #1813.
- The ROCm rpath was emitted only by the root `mlxcel` crate's build script, so the test binaries of `mlxcel-surgery` and `mlxcel-xla` built and then failed to start with "libamdhip64.so.7: cannot open shared object file". A dependency's `cargo:rustc-link-arg` does not reach the crate that links it, so each crate has to emit its own; the emission now lives in `src/lib/mlxcel-core/build_support/rocm_rpath.rs` and all three call it. The traces only ever ran the `mlxcel` binary, which had the rpath.

The conclusion is that a logit matrix is a decode-path check, not a backend check. `make verify-test-rocm` is the gate that covers the rest.

## Reproducing

```bash
make release-rocm
# Metal side, on an Apple machine: same commands with a Metal build.
MLXCEL_FUSED_MOE=0 ./target/release/examples/logit_trace models/mlx/Qwen3-30B-A3B-4bit tests/fixtures/wikitext2_excerpt.txt 8 80 8 512 > rocm_w8.tsv
python3 scripts/compare_logit_traces.py benchmarks/logit_traces/metal_m1u_bec64748/metal_m1u_bec64748_qwen3-30b-a3b_fused0_w8.tsv rocm_w8.tsv --decided 2.0
```

The checkpoint directory name is host-local: this host holds `models/mlx/Qwen3-30B-A3B-4bit` and the Metal host `models/mlx/qwen3-30b-a3b-4bit`, which is why the two trace headers disagree on it. What has to match is the Hugging Face revision, which `METADATA.txt` records on both sides.

`METADATA.txt` in each trace directory records the host, versions, checkpoint revisions and binary hashes; `RUNS.txt` records the exit status and row count of every run; `SHA256SUMS` covers every trace file.

`compare_logit_traces.py` prints the largest reference gap at a disagreement on every run, so the 1.125 figure above is reproducible from any pair without a separate script.

The serving check:

```bash
MLXCEL_FUSED_MOE=0 ./target/release/mlxcel-server -m models/mlx/Qwen3-30B-A3B-4bit --port 8080 &
./scripts/server_chat_smoke.sh --port 8080
```

## Known gaps at the time of this run

- Fused MoE has no ROCm kernel, so MoE models run through `gather_qmm` (#1803).
- `FFT`, `Hadamard` and `SearchSorted` are `NO_GPU` stubs, so audio, TurboQuant KV cache and stochastic speculative decoding abort (#1825).
- Only affine 4-bit was traced. mxfp4, mxfp8 and nvfp4 coverage is #1806, #1807 and #1808.
- Prefill through the sorted large-row `gather_qmm` path is slow on both dtypes (#1814).
- The model matrix in #1809 also names a sliding-window model, an SSM hybrid pair and a VLM. Those rows are not in this run; they wait on #1803 and #1805.
