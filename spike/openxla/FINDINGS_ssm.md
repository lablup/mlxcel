# SSM scan findings: prefill lowering of the selective scan (issue #3337)

Epic #564, Phase 1, unit U1. Scope: a causal depthwise conv1d and a selective scan for the OpenXLA/IREE emitter (`src/lib/mlxcel-xla/src/emitter/ssm.rs`, `ssm_scan.rs`), one full Mamba2 mixer layer proven numerically against HF `Mamba2Mixer.torch_forward` in fp32, and a measured decision on how the whole-prompt (prefill) scan lowers. Design record: `SSM_HYBRID_DESIGN.md`.

## Verdict

Chunked SSD lowers, compiles in under a second on `llvm-cpu` and `rocm` (gfx1151) at both 256 and 4096 rows, has a graph size independent of the capacity (157 StableHLO ops, 31 to 33 dispatches), and matches HF fp32 to 7e-6 on `local-task` and `hip`. It is the chosen prefill form, with `chunk_size = 128`. No fallback is needed. The associative scan is correct but materializes `[T, H, P, N]` per level (9.2 GiB transient at 4096 for the 130m mixer). The sequential unroll grows linearly in ops and dispatches (783 dispatches at 256) and at 4096 (102476 ops) neither `llvm-cpu` nor `rocm` finished compiling within the 1200 s budget.

The scan stays f32 on the f16 path: every scan contraction goes through `Builder::dot_general_keep_f32`, and only in_proj, out_proj and the conv follow the graph precision.

## Forms measured

All three forms share the probe interface and the same mixer around the scan; only `selective_scan_prefill_form` differs.

- Chunked SSD (`SsmPrefillForm::ChunkedSsd`). Rows are grouped into chunks of `Q`. Within a chunk the scan is `(C B^T o L) (x dt)` as two batched `dot_general`s, where `L = exp(segsum(dt A))` is built as HF's `segment_sum` does: a masked prefix sum (originally `reduce_window_prefix_add`, a log-depth shifted-add chain since #3343, see the IREE-Metal section), then a `-inf` mask above the diagonal, so near-diagonal decays carry no cancellation error. Each chunk's own end state is a third `dot_general`. The carry across chunks is one more `dot_general` against the chunk-level `exp(segsum)` matrix `[H, nc+1, nc+1]` (HF's form) instead of the unrolled sequence the issue suggested: the same math with no unroll, so the op count stays flat in `T`. The fifth contraction reads `C` against the state entering each chunk.
- Associative (`SsmPrefillForm::Associative`). Hillis-Steele inclusive scan over `(exp(dt A), dt x outer B)` pairs: `log2(T)` levels of pad/slice shifts and elementwise combines on `[T, H, P, N]`.
- Sequential (`SsmPrefillForm::Sequential`). `T` unrolled decode steps (`selective_scan_step`), outputs placed with `dynamic_update_slice`.

None uses scatter or `while`.

## Setup

- Host: Strix Halo gfx1151, 31 GiB unified memory, ROCm IREE toolchain `iree-rocm-3.12.0rc20260721` (`scripts/iree/setup-rocm.sh --env`).
- Mixer: the `state-spaces/mamba2-130m` mixer dimensions (`hidden 768`, `d_inner 1536`, `H 24`, `P 64`, `N 128`, `G 1`, `K 4`, `conv_dim 1792`) with seeded random weights (HF init for `A_log` and `dt_bias`, perturbed), fp32.
- Prefill input: `real_len = capacity - capacity/8` (224 of 256, 3584 of 4096). The padded rows are random, not zero, so the check also proves padding does not advance the state.
- Reference: HF `Mamba2Mixer.torch_forward` (transformers 5.12.1, slow path, `chunk_size=32` so the 4096-row reference fits in memory; the HF chunk size does not change the math).
- Compile: the emitted graph compiled directly by `iree-compile`; wall time and peak RSS of that process, dispatch count and transient size from `--iree-scheduling-dump-statistics-format=json`.
- Error: max-abs over the live `y` rows, the final `h`, and the conv tail, after running the graph through the IREE shim (`examples/xla_ssm_probe.rs`).
- Run time: one invocation through the auxiliary ABI, first call included. Indicative only (no warmup, no repeats).
- Reproduce: `spike/openxla/ssm_lowering_measure.py` (see README).

## Measurements

Compile columns are `seconds / peak RSS MiB`. Error columns are max-abs `y`; the `h` and conv-tail errors are below 3e-7 and 2.2e-6 in every measured row. Transient is the device scratch IREE plans for one call.

### context_capacity 256

| form | ops | dispatches | llvm-cpu compile | rocm compile | transient MiB | err local-task | err hip | run local-task / hip (s) |
|------|----:|-----------:|-----------------:|-------------:|--------------:|---------------:|--------:|-------------------------:|
| chunked Q=64 | 157 | 33 | 0.6 / 174 | 0.8 / 210 | 21.8 | 5.2e-6 | 6.0e-6 | 0.012 / 0.008 |
| chunked Q=128 | 157 | 33 | 0.6 / 174 | 0.8 / 211 | 15.9 | 5.2e-6 | 6.0e-6 | 0.012 / 0.009 |
| chunked Q=256 | 157 | 31 | 0.6 / 175 | 0.8 / 214 | 26.4 | 5.2e-6 | 6.0e-6 | 0.015 / 0.009 |
| associative | 173 | 29 | 0.6 / 167 | 0.6 / 201 | 587.4 | 5.0e-6 | 6.0e-6 | 0.126 / 0.053 |
| sequential | 6476 | 783 | 11.8 / 248 | 12.4 / 307 | 11.6 | 5.2e-6 | 6.0e-6 | 0.026 / 0.010 |

### context_capacity 4096

| form | ops | dispatches | llvm-cpu compile | rocm compile | transient MiB | err local-task | err hip | run local-task / hip (s) |
|------|----:|-----------:|-----------------:|-------------:|--------------:|---------------:|--------:|-------------------------:|
| chunked Q=64 | 157 | 33 | 0.8 / 176 | 0.8 / 208 | 315.1 | 6.7e-6 | 6.0e-6 | 0.129 / 0.085 |
| chunked Q=128 | 157 | 33 | 0.6 / 174 | 0.8 / 209 | 241.9 | 6.7e-6 | 6.0e-6 | 0.125 / 0.088 |
| chunked Q=256 | 157 | 33 | 0.6 / 173 | 0.8 / 212 | 362.3 | 6.7e-6 | 6.0e-6 | 0.151 / 0.097 |
| associative | 213 | 37 | 0.6 / 167 | 0.6 / 201 | 9385.9 | 6.7e-6 | 6.0e-6 | 3.292 / 1.146 |
| sequential | 102476 | n/a | timeout > 1200 s / ~0.8 GiB at kill | timeout > 1200 s / ~0.8 GiB at kill | n/a | not run | not run | n/a |

## Decision

- Prefill form: chunked SSD. Its op and dispatch counts do not depend on `context_capacity`, it compiles in under a second on both targets at 4096, and its error sits at the fp32 noise floor of the HF slow path.
- `chunk_size = 128` (`SSM_DEFAULT_CHUNK_SIZE`). All three chunk sizes give the same error and the same op count. Q=128 has the smallest transient at 4096 (242 MiB, against 315 for Q=64, where the `[H, nc+1, nc+1]` carry matrix and the per-chunk states grow, and 362 for Q=256, where the `[nc, H, Q, Q]` decay matrices grow) and the best or near-best run time.
- Associative scan: rejected. Correct, but its `[T, H, P, N]` levels need 9.2 GiB of transient for one 130m layer at 4096, which does not fit a real model in 31 GiB of unified memory, and it is the slowest at run time.
- Sequential unroll: rejected. 41x the ops and 24x the dispatches of chunked at 256, and at 4096 it does not compile within 20 minutes on either target, against under a second for chunked SSD.
- f32 scan on the f16 path: kept. With `MLXCEL_XLA_PRECISION=f16` (in_proj, out_proj and the conv demoted to f16 inputs, the scan contractions kept f32 by `dot_general_keep_f32`), chunked Q=128 errors against HF fp32 are `y` 2.0e-3 / 1.9e-3 at 256 and 2.2e-3 / 2.1e-3 at 4096 (local-task / hip), `h` at most 6.0e-5, conv tail at most 7.5e-4. The conv-tail error is the f16 in_proj rounding alone, and the `y` error does not grow from 256 to 4096 rows, so the f32 recurrence adds no accumulated drift. A demoted-scan variant was not measured: the design calls for an f32 recurrence, and the measured f16-path error is already dominated by the projections, so there is no evidence that would justify dropping it.

## Acceptance results (tiny mixer, `spike/openxla/ssm_gate.sh`)

The gate runs every acceptance case on `local-task` and `hip` at capacity 256 with a small mixer that has `G = 2 < H = 4`, so the group broadcast is exercised. Max-abs error against HF fp32, tolerance 5e-5:

| case | y local-task | y hip | h (worst) | conv tail (worst) |
|------|-------------:|------:|----------:|------------------:|
| decode, B=1 | 3.0e-7 | 4.8e-7 | 3.0e-8 | 3.0e-7 |
| decode, B=3, linear biases | 4.8e-7 | 4.8e-7 | 1.2e-7 | 2.4e-7 |
| prefill, real_len 100 < 256 | 7.2e-7 | 7.2e-7 | 3.3e-9 | 1.8e-7 |
| prefill, real_len = 256 | 7.2e-7 | 8.3e-7 | 4.2e-9 | 1.8e-7 |
| prefill, real_len 2 < K-1 | 3.0e-7 | 6.0e-7 | 4.1e-10 | 3.0e-7 |
| prefill, real_len 0 | 0 | 0 | 0 | 0 |
| prefill, real_len 0 from a carried state | 0 | 0 | 0 | 0 |
| prefill, real_len 77 from a carried state, biases | 9.5e-7 | 8.3e-7 | 1.2e-6 | 3.6e-7 |
| prefill, real_len 200 from a carried state, chunk 64 (4 chunks) | 1.3e-6 | 1.1e-6 | 6.9e-7 | 3.6e-7 |
| prefill, real_len 100, dt clamped to [0.01, 0.05] | 6.3e-7 | 7.7e-7 | 2.9e-8 | 1.8e-7 |
| decode, B=2, dt clamped to [0.01, 0.05] | 4.8e-7 | 6.0e-7 | 6.0e-8 | 2.4e-7 |

The multi-chunk carried case is the one where the inter-chunk decay reaches `y`: with a zero `h0` and at most two chunks the state entering every chunk needs no decay, so only `h_T` would catch a wrong chunk-level decay matrix (zeroing it leaves `y` within 1e-6 on the other prefill cases but fails this one with `y` off by 1.05). The dt-clamp cases fail by 1.7e-2 / 1.9e-2 in `y` with the clamp removed; HF's default `time_step_limit` is `(0, inf)`, so no other case exercises it.

The carried-state cases have no HF prefill equivalent (HF prefill always starts from zero), so their reference is HF's own decode branch run token by token from the same `h0` and conv tail.

## Notes for #3338 and later units

- The probe signature is `emit_ssm_probe(c, hidden, t, mode)`: `SsmConfig` has no model width, so the probe takes `hidden` explicitly.
- `mamba2_mixer_prefill` uses `conv0` as the left context of both the conv and the tail slice, so a prefill that continues from a carried state is exact and `real_len = 0` returns `h0` and `conv0` unchanged. `conv1d_causal_prefill` and `conv1d_tail` keep their zero-padded fresh-sequence contract.
- On the f16 path the prefill conv uses `Builder::convolution`, so its operands and result are f16 before the f32 cast, while the decode step's conv is an f32 multiply-reduce. Switching the prefill to `convolution_f32_accumulate` (f16 operands, f32 result) does not compile: iree-compile rejects the depthwise (`feature_group_count = conv_dim`) mixed-type convolution with a `tensor.collapse_shape` type mismatch. If the prefill/decode conv rounding gap matters for #3338's f16 graphs, the fix is a K-tap shifted multiply-add in f32 rather than the convolution op.
- HF's cache keeps the last `K` pre-conv inputs (`[B, C, K]`); the emitter state keeps the last `K-1` (`[B, K-1, C]`), which is all the next step reads.

## Whole-model graphs (issue #3338)

`spike/openxla/ssm_model_check.py` runs the state-threaded token prefill of a whole checkpoint, loaded with the production config reader and weight loader, through the IREE shim (`examples/xla_ssm_model_check.rs`) and compares last-token logits with HF `Mamba2ForCausalLM` fp32. Each run checks three paths: one prefill of the whole prompt, the prompt split in two prefill calls that hand the returned state on, and the prompt fed token by token through the decode step (state in module globals). Tolerance 3e-2 max-abs and the same argmax.

| model | device | precision | whole | split | decode | argmax |
|-------|--------|-----------|------:|------:|-------:|--------|
| synthetic (2 layers, H=8, G=2, biased, untied), 40 tokens, capacity 128 | local-task | f32 | 1.7e-6 | 1.3e-6 | 1.7e-6 | same |
| same | hip | f32 | 1.8e-6 | 1.2e-6 | 1.0e-6 | same |
| `AntonV/mamba2-130m-hf` @ `05e8773f`, 200 tokens (two chunks), capacity 256 | local-task | f32 | 2.0e-4 | 1.7e-4 | 1.1e-4 | same |
| same | hip | f32 | 3.5e-4 | 3.9e-4 | 3.3e-4 | same |
| same | local-task | f16 | 4.3e-1 | 4.5e-1 | 3.7e-1 | same (top-5 same) |
| same | hip | f16 | 4.2e-1 | 4.4e-1 | 4.3e-1 | same (top-5 same) |

The f32 rows pass. The f16 rows (`MLXCEL_XLA_PRECISION=f16`: f16-resident projections and f16 contraction inputs, the scan and conv kept f32) miss the 3e-2 bar against HF fp32 by an order of magnitude, on logits spanning -3.4 to 16.7, while the argmax and top 5 match. Within f16, the paths agree far more closely than f16 agrees with fp32: whole prefill against decode differs by 6.6e-2 (local-task) and 4.2e-2 (hip), and whole against split by 3.8e-2 and 5.8e-2. The gap is projection rounding accumulated over 24 layers, not a prefill/decode mismatch. f16 is measured here, not gated; whether #3339 defaults Mamba2 to f16 on GPU should be decided on a token-level comparison.

The f16 drift is fully explained by projection rounding. Rounding every `nn.Linear` input and weight to f16 (f32 accumulate) inside HF's own fp32 forward reproduces it on the same 200 tokens: 0.44 max-abs with all projections, 0.39 with only `in_proj`, 0.026 with only `out_proj`, 0.010 with only the head; keeping the `dt` columns of `in_proj` in f32 still leaves 0.40, so the error enters through `x`, `B`, `C` and `z`, not the step size. Norms, softplus, the D skip and the residual stream are f32 in the graph on every precision path, so `residual_in_fp32` changes nothing there. In that simulation, 40 greedy tokens from five prompts (including "The capital of France is") match fp32 exactly. `hip` defaults to f16 (`default_precision`), which is the precision #3339's token-exact gate will run at unless it pins f32 the way MLA configs do in `effective_precision`.

The check runs the `InOut` projection layout by default (`--layout inout`), the layout `IreeLlama` always emits and uploads; `OutIn` gives the same numbers on both devices.

Compile time of one uncached `iree-compile` for mamba2-130m (24 layers), f32:

| target | capacity | prefill | decode |
|--------|---------:|--------:|-------:|
| llvm-cpu (local-task) | 256 | 2.5 s | 1.7 s |
| llvm-cpu (local-task) | 4096 | 2.6 s | 1.7 s |
| rocm gfx1151 (hip) | 256 | 2.5 s | 1.6 s |
| rocm gfx1151 (hip) | 4096 | 2.6 s | 1.6 s |

### Carry-over from #3337

- Dim cross-check: `Config::from_json` checks every SSM tensor of every layer (`in_proj` `[in_proj_dim, hidden]`, `out_proj`, `conv1d` `[conv_dim, 1, K]` or MLX `[conv_dim, K, 1]`, `dt_bias` / `A_log` / `D` `[H]`, the gated-norm weight `[d_inner]`, the embeddings and final norm, and `lm_head.weight` when the head is untied) against the safetensors headers before any graph is emitted, so a config cannot ask for state or graph shapes the checkpoint does not have. `tie_word_embeddings` follows the HF `Mamba2Config` default (false), so a config that omits it reads `lm_head.weight` and the check refuses a checkpoint without one.
- The check does not trust a header alone: every entry must occupy exactly `product(shape) * dtype` bytes inside the file's data section (as the loader's `SafeTensors::deserialize` requires), index shard names go through the loader's plain-file-name rule (`shard_path.rs`), and the per-layer state `H*P*N` may not exceed the layer's `in_proj` element count, since `H*P` and `G*N` are bounded separately but their product is not. `num_hidden_layers` is capped at 1024 at parse time for every config, dense included, because the mixer schedule is sized by it before any checkpoint is read.
- `real_len`: the runtime prefill already refuses a prompt over the capacity on the host (`prefill_padded`, and the shim rejects `real_len` outside `[0, capacity]`); the model-check runner refuses an empty or over-capacity prompt and out-of-vocabulary tokens before the call.
- Bounded carry: not needed yet. The chunk carry costs `H * nc^3` adds against `H * T * Q * (P + N)`-order within-chunk work; at the largest ladder capacity, 32768 (`nc = 256`), that is about 2% of the within-chunk work for mamba2-130m (P = 64, N = 128), and the crossover is near T = 185k to 230k depending on how the within-chunk ops are counted. Revisit if a context bucket beyond about 64k is enabled.
- f16 conv: the prefill conv is now K shifted f32 multiply-adds (the same arithmetic as the decode step) instead of `Builder::convolution`, so prefill and decode round the same way on the f16 path.


## Single-sequence runtime (issue #3339)

The shim now carries the state from `prefill.main` into `decode_step` (`@state_import`, once per request) and `MLXCEL_BACKEND=xla mlxcel generate` runs `AntonV/mamba2-130m-hf` @ `05e8773f`. Greedy ids for "The capital of France is", 40 tokens, compared with the HF fp32 oracle (`oracle_continuation.py`, which needed no change for Mamba2):

| device | precision | CLI (prefix prefill + decode) | `xla_oracle_check` (full prefill + decode) |
|--------|-----------|------------------------------:|-------------------------------------------:|
| hip | f16 (default) | 40/40 | 40/40 |
| hip | f32 | 40/40 | not run |
| local-task | f32 (default) | 40/40 | 40/40 |

A one-token prompt ("The", 10 tokens), which takes the state-reset path instead of a `real_len = 0` prefill, also matches the oracle on hip. The f16 drift measured above (0.42 max-abs on logits) did not move a greedy token here, so recurrent configs kept the per-device default precision rather than being pinned to f32 like MLA (superseded in #3340, below).

Refusals: a request refused in Rust or failed in the shim (including a failed state reset) drops the stored state, so a decode refuses until a prefill or reset succeeds. The KV-only entry points (ragged, slot, embeddings) refuse any recurrent state until #3340, and pure `mamba2` with `n_groups > 1` is refused. The CLI omits the `KV cache mode` banner for Mamba2 on xla.

## Per-slot state carry and serving (issue #3340)

The continuous-batching engine now carries each slot's Mamba2 state: the ragged decode takes and returns `[B, ...]` state, the shim keeps `sscache_b[]` per slot under the KV's `rg_valid` flag, and `mlxcel-server` serves `AntonV/mamba2-130m-hf` @ `05e8773f` on `hip`.

First measurement, before the precision pin:

| check | precision | result |
|-------|-----------|--------|
| `validate_arch.sh --batch 4 --requests 8 --maxcap 24` (structural, oracle, serve) | f16 (hip default then) | PASS |
| `xla_batch_bench`, oracle prompt, `--cancel 2` | f16 | PASS (2/2 cancelled) |
| `xla_batch_bench`, oracle prompt, `--buckets 256,512` | f16 | PASS |
| `xla_batch_bench`, B=8, 16 requests, 64-token prompt | f16 | PASS |
| `xla_batch_bench`, B=4, 8 requests, 64-token prompt | f16 | 7/8; request 6 differs from token 17 |
| same | f32 | PASS |
| server, 4 concurrent `/v1/completions` at temperature 0 vs `mlxcel generate` | f16 | 3/4 |
| same | f32 | 4/4 |

The two f16 differences are rounding between the batched (`[B, hidden]`) and single-sequence projections, not state handling: the same workloads are exact at f32. On the server prompt that differs ("In computer science, a linked list is"), the batched f16 stream equals the HF fp32 oracle and the f32 streams, and it is the single-sequence f16 decode that departs from the oracle after 16 tokens. The single-prompt check in #3339 did not catch this.

Decision: recurrent configs are now pinned to f32 on every device, like MLA, for both the single-sequence and the ragged bundles (`Config::pins_f32_precision`, `effective_precision` in `iree.rs`). An explicit `MLXCEL_XLA_PRECISION` still overrides the pin; f16 is not token-exact for recurrent models and is for measurement only. Re-run at the default precision (now f32) on `hip`:

| check | result |
|-------|--------|
| `mlxcel generate` 40 tokens vs HF fp32 oracle: "The capital of France is", "In computer science, a linked list is", "Once upon a time, in a small village by the sea,", "The three primary colors are", "Water boils at a temperature of" | 5/5 token-exact (and `xla_oracle_check` 5/5) |
| `validate_arch.sh --batch 4 --requests 8 --maxcap 24` | PASS (structural, token-exact, serve reference-exact; 123.7 tok/s batched vs 75.0 sequential) |
| `xla_batch_bench`, B=4, 8 requests, 64-token prompt | PASS (122.8 tok/s) |
| `xla_batch_bench`, oracle prompt, `--cancel 2` | PASS (2/2 cancelled; 122.5 tok/s) |
| `xla_batch_bench`, oracle prompt, `--buckets 256,512` | PASS (98.3 tok/s) |
| `xla_batch_bench`, B=8, 16 requests, 64-token prompt | PASS (108.7 tok/s) |
| server, 4 concurrent `/v1/completions` at temperature 0 vs `mlxcel generate` | 4/4 |

The pin costs throughput on this 130M model: the serve gate ran at 285.6 tok/s batched at f16 and 123.7 tok/s at f32.

## IREE-CUDA validation on GB10 (issue #3342)

Validation ran on `NVIDIA GB10` (`sm_121`, driver 580.178.04) from main at `fc10c852`, with IREE `3.12.0rc20260721` at `dc9601f` and `AntonV/mamba2-130m-hf` at `05e8773f`. Every GPU command ran through `gpu-lock run --tag xla-ssm`. Recurrent precision resolved to f32, matching the ROCm decision above.

The single-layer HF fp32 probe passed all four required boundaries at tolerance `5e-5`.

| case | rows / real length | compile s | run s | max-abs y | max-abs h | max-abs conv |
|------|-------------------:|----------:|------:|----------:|----------:|-------------:|
| decode step | 1 / 1 | 1.22 | 0.009 | 2.623e-6 | 2.384e-7 | 6.557e-7 |
| padded prefill | 256 / 224 | 1.82 | 0.009 | 2.503e-6 | 1.583e-7 | 5.960e-7 |
| full prefill | 256 / 256 | cached | 0.006 | 2.503e-6 | 1.118e-7 | 7.153e-7 |
| shorter than `K-1` | 256 / 2 | cached | 0.009 | 2.027e-6 | 1.118e-8 | 6.557e-7 |

The exact production StableHLO modules emitted at capacity 256 were recompiled uncached with `--iree-hal-dump-executable-sources-to`. The single-sequence decode and ragged decode compiles retained the production aggressive-fusion flag. None of the three dispatch dumps contains `iree_linalg_ext.scatter`.

| module | compile s | peak compiler RSS MiB | scatter count |
|--------|----------:|----------------------:|--------------:|
| prefill | 4.57 | 213.2 | 0 |
| decode | 3.18 | 214.9 | 0 |
| ragged decode B=4 | 4.72 | 185.1 | 0 |

`validate_arch.sh --batch 4 --requests 8 --maxcap 24` passed its structural, HF token-exact, and serve reference-exact gates. The CLI generated the required 40-token continuation at 140.70 tok/s and matched the HF fp32 oracle 40/40. The production continuous-batching runs remained reference-exact across slot recycling, cancellation, capacity routing, and both bundled batch widths.

| execution gate | result | decode throughput |
|----------------|--------|------------------:|
| CLI B=1, 40 tokens | HF token-exact 40/40 | 140.70 tok/s |
| B=4, 8 requests | reference-exact | 88.71 tok/s |
| B=4, cancel 2 | 2/2 cancelled and remaining streams reference-exact | 89.22 tok/s before cancellation pass |
| B=4, buckets 256 and 512 | both buckets routed and reference-exact | 77.00 tok/s |
| B=8, 16 requests | reference-exact | 81.36 tok/s |
| server, 4 concurrent `/v1/completions` | four 24-token streams equal their independent CLI references | 4/4 exact |

The server gate used four distinct prompts of 5, 8, 13, and 5 tokens with `--ctx-size 2048 --max-batch-size 4`. It disabled recurrent prefix caching as designed, compiled the 256 and 4096 capacity buckets, and loaded in 10.43 s. All responses finished by the requested length.

The existing chunked SSD lowering therefore needs no CUDA-specific change. This PR changes only the findings record, so it does not require another ROCm gate beyond the already recorded #3339 and #3340 runs.

## Mamba1 diagonal scan (issue #3348)

Mamba1 and Falcon-Mamba use the same f32 diagonal recurrence but do not use
Mamba2's SSD contraction. Prefill is partitioned at `chunk_size = 128` in
production. Each partition materializes at most `[128,D,N]` `log_a` and input
terms and carries only `[D,N]` between partitions, so the 4096-row capacity
never creates a `[4096,D,N]` value. Within a partition, a Hillis-Steele prefix
scan composes affine pairs `(a, u)` as `(a_r*a_l, u_r + a_r*u_l)` in
`ceil(log2(Q))` levels. This replaced an initial per-row unroll that emitted
62,156 lines / 5.35 MiB and did not compile inside a 180 s CPU budget at 4096.

The production-dimension (`D=1536`, `N=16`, `dt_rank=48`) capacity-4096 graph
is 3,116 lines / 320 KiB. Source and emitted-shape inspection find the chunk
shape `[128,1536,16]` and no `[4096,1536,16]` tensor. The structural regression
also caps the graph below 4,000 lines and 80 concatenations, so a per-row scan
or output assembly cannot silently return.

The independent reference is Transformers 5.12.1 on CPU (`torch==2.12.1+cpu`),
calling `MambaMixer.slow_forward` or `FalconMambaMixer.slow_forward` directly
with seeded fp32 weights. The XLA side ran through the real auxiliary ABI and
IREE `local-task` runtime. Tolerance is `5e-5`.

| case | rows / real length | compile s | run s | max-abs y | max-abs h | max-abs conv |
|------|-------------------:|----------:|------:|----------:|----------:|-------------:|
| Mamba1 padded prefill | 9 / 7 | 1.61 | <0.001 | 2.235e-8 | 9.313e-10 | 0 |
| Mamba1 full prefill (same compiled shape) | 9 / 9 | cached | 0.002 | 2.235e-8 | 1.863e-9 | 0 |
| Mamba1 decode, B=3 | 3 / 1 | 0.87 | <0.001 | 2.980e-8 | 5.960e-8 | 0 |
| Falcon-Mamba decode, B=3 | 3 / 1 | 1.01 | <0.001 | 5.960e-8 | 5.960e-8 | 0 |

At the real 130M mixer dimensions and capacity 4096 (`real_len=3584`), direct
`llvm-cpu` compilation completed in 10.64 s at 228.2 MiB peak compiler RSS.
The same graph through the IREE shim compiled in 9.96 s, ran on `local-task` in
0.862 s, and the whole process peaked at 331.1 MiB RSS. Against Transformers'
independent sequential Mamba slow path, max-abs errors were `y=3.427e-7`,
`h=1.246e-8`, and conv tail `0` (tolerance `5e-5`).

Whole-model CPU checks use the official
`state-spaces/mamba-130m-hf` revision
`1e76775f628fbf1350fbe4dbb3d971ba64af25a1` and a seeded two-layer
`FalconMambaForCausalLM` checkpoint. Each check compares whole prefill, split
prefill with carried state, and token-by-token decode against Transformers
5.12.1 fp32. All argmaxes match; the Mamba checkpoint's worst max-abs error is
`2.043e-3`, and Falcon-Mamba's is `4.768e-7` (whole-model tolerance `3e-2`).

Reproduce the padded Mamba1 row (after exporting the pinned IREE environment):

```bash
PY=/home/inureyes/Development/mlxcel/spike/openxla/.venv/bin/python
$PY spike/openxla/ssm_oracle.py --kind mamba1 --dims tiny \
  --emit-inputs /tmp/mamba1-prefill --mode prefill --capacity 9 \
  --real-len 7 --chunk-size 4
cargo run --features xla-iree --example xla_ssm_probe -- \
  --inputs /tmp/mamba1-prefill --device local-task --out /tmp/mamba1-out
$PY spike/openxla/ssm_oracle.py --check /tmp/mamba1-out
```

These are CPU execution results only. The required `hip` probe and whole-model
rows, CLI, server, cancellation, and slot-recycling evidence remain pending.

## IREE-Metal validation on macOS (issue #3343)

Validation ran on an Apple M1 Ultra (macOS 27.2) from main at `66bf600a` plus this change, with IREE `3.12.0rc20260721` at `dc9601f` and `AntonV/mamba2-130m-hf` at `05e8773f`. The oracle venv used plain `torch==2.12.1` with transformers 5.12.1, because the `+cpu` wheel pinned in `requirements.lock` does not exist for macOS arm64. `metal-spirv` cannot lower bf16, so every gate ran at f32 and at f16 (`MLXCEL_XLA_PRECISION=f16`).

### Root cause: `iree_linalg_ext.scan` on IREE-Metal

Before this change the production chunked prefill failed on `metal` at f32 while `local-task` passed on the same inputs: `y` was NaN and `h_T` was off by 0.74 against a reference maximum of 0.10 on the padded case. The associative and sequential forms passed on `metal`, and so did the decode step.

Running the probe's StableHLO through `iree-run-module` on both devices and comparing intermediates puts the first divergence at the first `stablehlo.reduce_window`, the per-chunk prefix sum of `dt A`. Its input matches `local-task` to 2.5e-7 and its output is off by 1.4, and every value downstream inherits that. IREE rewrites a left-padded prefix-sum `reduce_window` into `iree_linalg_ext.scan`, and the Metal target returns values unrelated to the input for that op on any tensor of rank 2 or more. A standalone module with one such `reduce_window` reproduces it without mlxcel: rank 1 is correct at widths 8 and 128, and `[3,3]`, `[128,4]`, `[4,128]`, `[2,128,4]`, `[2,4,128]` and `[4,3,3]` are wrong along every axis (max-abs error 1.5 to 28 on unit-normal input). The SPIR-V IREE emits for the `[3,3]` dispatch looks correct; it binds the input at 0 and the output at 2 and leaves the accumulator binding 1 unused. Which later stage breaks it (SPIRV-Cross, the argument-buffer binding, or the Metal compiler) was not determined.

The `-inf` mask in `exp_segsum` was the first suspect and is not the cause. Replacing `-inf` with a large finite value, or taking `exp` before the mask, left the Metal error unchanged to four digits, and the fixed graph still carries the `-inf` select.

### Fix

`ssm_scan.rs` no longer emits `reduce_window`. Both prefix sums of the chunked scan, the per-chunk `acum` and the masked segment sum inside `exp_segsum`, are built by `prefix_add`: `ceil(log2(len))` levels of `slice`, `pad` and `add` in f32 (7 levels at `Q = 128`). The masked-rows structure of `exp_segsum` is unchanged, so near-diagonal decays still carry no cancellation error. `Builder::reduce_window_prefix_add` had no other caller and was removed.

The form is device-agnostic, so the emitted StableHLO changes on every backend, not only on Metal. At the real mixer dimensions and capacity 256 the prefill probe goes from 191 to 237 lines: the 3 `reduce_window` ops (7, 7 and 2 levels) become 16 levels of `slice`, `pad` and `add`; nothing else moves, and the decode graphs are byte-identical. The summation is pairwise instead of a sequential fold, so prefill results move in the last bits. On `local-task` at f32 the output changes by at most 7.2e-7 (`y`) and 4.3e-7 (`h`) across the four cases below, the decode and `K-1` cases are bitwise identical, and the error against HF stays in the same order (3.3e-6 before, 3.6e-6 after on `y`). The `nemotron_h` and `falcon_h1` prefill goldens were regenerated because those hybrids share the scan. ROCm and CUDA were not re-run on this host.

### Single-layer probe

Real `mamba2-130m` mixer dimensions against the HF fp32 oracle, tolerance `5e-5`. Compile times are uncached; the three prefill cases share one compiled graph.

| case | rows / real length | device | precision | compile s | run s | max-abs y | max-abs h | max-abs conv |
|------|-------------------:|--------|-----------|----------:|------:|----------:|----------:|-------------:|
| decode step | 1 / 1 | metal | f32 | 0.89 | 0.013 | 1.431e-6 | 2.027e-6 | 3.576e-7 |
| padded prefill, before the fix | 256 / 224 | metal | f32 | 4.83 | 0.227 | NaN | 7.422e-1 | 0 |
| padded prefill | 256 / 224 | metal | f32 | 5.17 | 0.192 | 1.431e-6 | 5.662e-7 | 0 |
| full prefill, before the fix | 256 / 256 | metal | f32 | cached | 0.220 | NaN | 8.108e-1 | 0 |
| full prefill | 256 / 256 | metal | f32 | cached | 0.195 | 1.431e-6 | 3.464e-7 | 0 |
| shorter than `K-1`, before the fix | 256 / 2 | metal | f32 | cached | 0.236 | 7.26e-3 | 1.145e-2 | 0 |
| shorter than `K-1` | 256 / 2 | metal | f32 | cached | 0.196 | 8.34e-7 | 1.68e-8 | 0 |
| long prefill | 4096 / 3584 | metal | f32 | 5.38 | 0.336 | 1.490e-6 | 2.123e-7 | 0 |
| decode step | 1 / 1 | metal | f16 | 0.93 | 0.013 | 8.34e-4 | 1.559e-4 | 6.096e-4 |
| padded prefill, before the fix | 256 / 224 | metal | f16 | cached | 0.235 | 3.31 | 8.674e28 | 6.557e-4 |
| padded prefill | 256 / 224 | metal | f16 | 5.34 | 0.194 | 1.51e-3 | 2.68e-5 | 6.56e-4 |
| full prefill | 256 / 256 | metal | f16 | cached | 0.198 | 1.51e-3 | 2.35e-5 | 6.32e-4 |
| shorter than `K-1` | 256 / 2 | metal | f16 | cached | 0.203 | 7.69e-4 | 9.15e-6 | 6.51e-4 |
| long prefill | 4096 / 3584 | metal | f16 | 5.56 | 0.341 | 1.593e-3 | 3.846e-5 | 7.537e-4 |
| padded prefill, before the fix | 256 / 224 | local-task | f32 | cached | 0.020 | 3.34e-6 | 1.60e-7 | 0 |
| padded prefill | 256 / 224 | local-task | f32 | 1.44 | 0.023 | 3.58e-6 | 1.70e-7 | 0 |
| padded prefill | 256 / 224 | local-task | f16 | 1.13 | 0.019 | 1.51e-3 | 2.68e-5 | 6.56e-4 |

All f32 rows on `metal` pass after the fix. The f16 rows are above the f32 tolerance by design (f16 projections around an f32 scan); the fixed `metal` numbers equal the `local-task` f16 numbers to three digits and carry no NaN.

`spike/openxla/ssm_gate.sh OUT local-task metal` (tiny mixer, 11 cases) at f32: `local-task` 11/11 before and after, `metal` 6/11 before (five of the six prefill cases with real rows failed, up to 1.97 on `y`) and 11/11 after, worst `y` 2.80e-6 and worst `h` 1.43e-6. At f16 the script reports FAIL on both devices because it applies the f32 tolerance; after the fix the worst `metal` errors are `y` 1.85e-3 and `h` 2.52e-5, matching `local-task` (1.85e-3 and 2.49e-5), against `h` errors up to 1.1e6 before.

### f32 scan on the f16 path

The capacity-256 probe emitted with `MLXCEL_XLA_PRECISION=f16` has `xf16` operands on exactly two contractions, `in_proj` (`256x768xf16` by `3352x768xf16`) and `out_proj` (`256x1536xf16` by `768x1536xf16`). The five scan contractions are f32 on both operands: `C B^T` (`2x128x1x128xf32`), the intra-chunk output (`2x24x128x128xf32` by `2x128x24x64xf32`), the chunk end states (`2x128x24x64xf32` by `2x128x24x128xf32`), the chunk carry (`24x3x3xf32` by `3x24x8192xf32`) and the off-diagonal output (`2x128x24x128xf32` by `2x24x64x128xf32`). All 16 prefix-sum adds are on `2x128x24xf32`, `2x24x128x128xf32` or `24x3x3xf32`. The production prefill graph that the f16 model gates compiled has 49 contractions with f16 operands (two projections per layer and the head), 120 f32-only contractions (five per layer), 384 f32 prefix-sum adds, and no `reduce_window`.

Recurrent configs pin f32 (`Config::pins_f32_precision`), and `metal` follows it: with `MLXCEL_XLA_PRECISION` unset the emitted prefill and decode graphs contain no `xf16`. Setting it to `f16` takes effect, as counted above.

### Model gates

Uncached `iree-compile` of the production graphs for `metal` (24 layers):

| module | f32 compile s | f16 compile s | peak compiler RSS MiB |
|--------|--------------:|--------------:|----------------------:|
| prefill, capacity 256 | 9.41 | 9.24 | 214 |
| prefill, capacity 4096 | 9.69 | 9.69 | 225 |
| decode (aggressive fusion) | 3.46 | 3.52 | 179 |
| ragged decode B=4 | 7.04 | 7.15 | 173 |

At f16 the server loaded both capacity buckets (256 and 4096) uncached in 18.6 s.

| gate | f32 | f16 |
|------|-----|-----|
| `validate_arch.sh --batch 4 --requests 8 --maxcap 24` | exit 0: structural, HF token-exact, serve reference-exact | exit 0 with `--skip-structural` (the goldens are f32 graphs): HF token-exact, serve reference-exact |
| B=4 batch bench inside it, 8 requests | 16.60 tok/s batched, 14.93 tok/s sequential baseline | 17.01 tok/s batched, 11.16 tok/s sequential baseline |
| `xla_oracle_check`, 40 tokens, five prompts | 5/5 token-exact | 4/5 token-exact, 1 near-tie |
| `mlxcel generate -n 40`, five prompts, B=1 | 5/5 token-exact, 49.3 to 53.9 tok/s | 4/5 token-exact, 1 near-tie, 53.9 to 56.6 tok/s |
| server, 4 concurrent `/v1/completions`, 24 tokens, temperature 0 | 4/4 equal their CLI streams and the oracle | 3/4 equal their CLI streams; 4/4 token-exact against the oracle |

The five prompts are the #3340 set: "The capital of France is", "In computer science, a linked list is", "Once upon a time, in a small village by the sea,", "The three primary colors are", "Water boils at a temperature of". The server ran with `--ctx-size 2048 --max-batch-size 4` on the first four (5, 8, 13 and 5 prompt tokens) and every response finished by length.

There is one f16 divergence from the oracle, and it is a near-tie. On "In computer science, a linked list is" the single-sequence f16 stream (`mlxcel generate` and `xla_oracle_check`) leaves the oracle at token 14: the oracle picks id 187 (a newline) and the stream picks the runner-up id 380, where the oracle's top-2 logit gap is 0.004593, below the 0.01 tolerance. The batched f16 stream for the same prompt (server) follows the oracle for all 24 tokens, which is why that one server stream differs from its CLI stream. This is the same prompt and the same batched versus single-sequence f16 rounding that #3340 recorded on `hip`; at f32 both paths are exact. The smallest top-2 gaps of the other four oracles over 40 tokens are 0.047, 0.057, 0.100 and 0.034, and no f16 stream diverged on them.

The gap comes from `oracle_continuation.py --top2`, which writes `ref_top2_ids` and `ref_top2_gaps` into the oracle JSON. `near_tie_check.py` applies the rule to a stream (only the first divergence counts), and `xla_oracle_check` prints the gap at its first divergence while keeping its strict exit status.

Reproduce (after `eval "$(scripts/iree/setup-macos.sh --env)"`):

```bash
PY=spike/openxla/.venv/bin/python
M=models/hf/mamba2-130m
cargo build --release --features xla-iree --bin mlxcel --bin mlxcel-server \
  --example xla_ssm_probe --example xla_oracle_check --example xla_batch_bench --example xla_traj_dump

# probe, one case (set MLXCEL_XLA_PRECISION=f16 for the f16 rows)
$PY spike/openxla/ssm_oracle.py --emit-inputs out/padded/inputs --dims mamba2-130m \
  --mode prefill --capacity 256 --real-len 224
target/release/examples/xla_ssm_probe --inputs out/padded/inputs --device metal --out out/padded/metal
$PY spike/openxla/ssm_oracle.py --check out/padded/metal
spike/openxla/ssm_gate.sh out/gate local-task metal

# model gates
MLXCEL_XLA_PRECISION=f32 scripts/xla/validate_arch.sh --model $M --device metal --batch 4 --requests 8 --maxcap 24
MLXCEL_XLA_PRECISION=f16 scripts/xla/validate_arch.sh --model $M --device metal --batch 4 --requests 8 --maxcap 24 --skip-structural

# near-tie rule on one f16 stream
$PY spike/openxla/oracle_continuation.py --model $M --out out/oracle.json --top2 \
  --prompt "In computer science, a linked list is" --max-new 40
MLXCEL_XLA_PRECISION=f16 target/release/examples/xla_oracle_check --model $M --oracle out/oracle.json --device metal
MLXCEL_XLA_PRECISION=f16 MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=metal target/release/mlxcel generate \
  -m $M -p "In computer science, a linked list is" -n 40 > out/cli.txt
$PY spike/openxla/near_tie_check.py --oracle out/oracle.json --model $M --text-file out/cli.txt --mlxcel-cli
```
