# SSM scan findings: prefill lowering of the selective scan (issue #3337)

Epic #564, Phase 1, unit U1. Scope: a causal depthwise conv1d and a selective scan for the OpenXLA/IREE emitter (`src/lib/mlxcel-xla/src/emitter/ssm.rs`, `ssm_scan.rs`), one full Mamba2 mixer layer proven numerically against HF `Mamba2Mixer.torch_forward` in fp32, and a measured decision on how the whole-prompt (prefill) scan lowers. Design record: `SSM_HYBRID_DESIGN.md`.

## Verdict

Chunked SSD lowers, compiles in under a second on `llvm-cpu` and `rocm` (gfx1151) at both 256 and 4096 rows, has a graph size independent of the capacity (157 StableHLO ops, 31 to 33 dispatches), and matches HF fp32 to 7e-6 on `local-task` and `hip`. It is the chosen prefill form, with `chunk_size = 128`. No fallback is needed. The associative scan is correct but materializes `[T, H, P, N]` per level (9.2 GiB transient at 4096 for the 130m mixer). The sequential unroll grows linearly in ops and dispatches (783 dispatches at 256) and at 4096 (102476 ops) neither `llvm-cpu` nor `rocm` finished compiling within the 1200 s budget.

The scan stays f32 on the f16 path: every scan contraction goes through `Builder::dot_general_keep_f32`, and only in_proj, out_proj and the conv follow the graph precision.

## Forms measured

All three forms share the probe interface and the same mixer around the scan; only `selective_scan_prefill_form` differs.

- Chunked SSD (`SsmPrefillForm::ChunkedSsd`). Rows are grouped into chunks of `Q`. Within a chunk the scan is `(C B^T o L) (x dt)` as two batched `dot_general`s, where `L = exp(segsum(dt A))` is built as HF's `segment_sum` does: a masked prefix sum with `reduce_window_prefix_add`, then a `-inf` mask above the diagonal, so near-diagonal decays carry no cancellation error. Each chunk's own end state is a third `dot_general`. The carry across chunks is one more `dot_general` against the chunk-level `exp(segsum)` matrix `[H, nc+1, nc+1]` (HF's form) instead of the unrolled sequence the issue suggested: the same math with no unroll, so the op count stays flat in `T`. The fifth contraction reads `C` against the state entering each chunk.
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

A one-token prompt ("The", 10 tokens), which takes the state-reset path instead of a `real_len = 0` prefill, also matches the oracle on hip. The f16 drift measured above (0.42 max-abs on logits) did not move a greedy token here, so recurrent configs keep the per-device default precision rather than being pinned to f32 like MLA.
