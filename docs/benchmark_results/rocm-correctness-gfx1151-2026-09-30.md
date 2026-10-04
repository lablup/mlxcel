# ROCm correctness matrix, remaining rows: Radeon 8060S (gfx1151), 2026-09-30

Second run for issue #1809, extending [the 2026-09-12 matrix](rocm-correctness-gfx1151-2026-09-12.md). That run covered four checkpoints against a Metal reference. This one adds the five rows it left open: a second dense model, a sliding-window model, two SSM hybrids and a VLM.

What this run establishes, and what it does not:

- **Established on ROCm:** all five checkpoints load and generate, and all sixteen teacher-forced traces, plus the two `w1ctx512` traces, complete with no NaN. Getting there found and fixed one ROCm backend defect, a strided-scan launch that wrote past the end of its array and faulted the GPU on both SSM hybrids.
- **Established against Metal:** the Metal half was traced on an Apple M5 Max and compared pair by pair. Fifteen of the sixteen pairs pass the first run's criterion, zero disagreements on decided positions at `--decided 2.0`. The sixteenth, `nemotron-3-nano-30b-a3b` at `w1`, is inconclusive: its reference has no position with a 2.0 gap, the same `0 / 0` case the first run recorded for two of its `w1` rows. No pair fails. Across all sixteen the backends disagree on the top-1 token at 155 of 6720 positions and at none of the 2696 decided ones, and the largest reference gap at any disagreement is 1.250 logits (`nemotron-3-nano-30b-a3b` `w256`, the Metal token at ROCm's rank 2).
- **Not like-for-like:** the Metal reference is an M5 (Apple GPU generation 17, NAX kernels), not the M1 Ultra of the first run; every Nemotron-H pair compares Metal's `fused_moe_forward` with ROCm's `forward_nonfused`; and the VLM rows trace only the language model. The hybrid `w1` rows are graph against graph: without a prefill no single-token step has SSM state, so Metal does not take its fused SSM update kernel there. [What is not like-for-like](#what-is-not-like-for-like) says what each of these changes.
- **The fused SSM kernel, added:** two more rows, `w1ctx512` for `granite-4.0-h-tiny` and `nemotron-3-nano-30b-a3b`, prefill 512 tokens before each single-token forward, so Metal runs its fused SSM update kernel and ROCm runs the SSD graph with state. Both pass the same criterion: 0 / 60 and 0 / 71 decided mismatches, largest gap at a disagreement 0.125 and 0.250. Both halves of these two rows are at `3c9edea0`, a later commit than the other sixteen pairs. [The fused SSM kernel against the graph with state](#the-fused-ssm-kernel-against-the-graph-with-state) has the numbers.

## Environment

| Field | ROCm side | Metal reference |
|---|---|---|
| Hardware | AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5, 40 CUs), 96 GiB VRAM carve-out, 31 GiB host RAM | Apple M5 Max (Mac17,7), 40-core GPU, 128 GiB unified memory; Apple GPU generation 17, NAX path available |
| OS | Debian GNU/Linux 13 (trixie), kernel 6.18.12+deb13-amd64 | macOS 27.0 (26A428) |
| Backend | ROCm 10.0.0 packages (HIP 7.15.26333, AMD clang 23), `--features rocm`, release profile | Metal, `--features metal,accelerate`, release profile |
| mlxcel | `c5fe9a16`: `origin/main` `4595b06f` plus the code commits of PR #2059 (see below) | `d1128266`, the merge commit of PR #2059 |
| MLX pin | `81ba1c6a` | `81ba1c6a`; metallib sha256 `dca1bb42...`, identical to the M1 Ultra run's |
| ROCm overlay | fork `NripeshN/mlx` branch `rocm-support` at `75915908` (`src/lib/mlx-cpp/patches-rocm/UPSTREAM`), plus the local fixes in `LOCAL_FIXES.md`, now 22 items | not used |
| Traces | `benchmarks/logit_traces/rocm_gfx1151_c5fe9a16/` | `benchmarks/logit_traces/metal_m5_d1128266/` |

The first run's document implied the fork commit through `UPSTREAM` but did not state it; it is `75915908` for both runs.

The two sides are at different commits. `git diff c5fe9a16 d1128266` touches no file under `src/models/`, not `examples/logit_trace.rs` and not `scripts/compare_logit_traces.py`; its source changes are the paged-attention port predicates, their tests, and the port check in the sparse paged decode, none of which `logit_trace` runs. The other three things that must match do: the corpus (sha256 `16721fc1...` on both), the arguments, and the five checkpoint revisions, which the two `METADATA.txt` files record as the same strings. The Metal session also checked every file of each checkpoint against the pinned Hugging Face revision (sha256 for LFS files, git blob sha1 otherwise), and `sha256sum -c SHA256SUMS` passes for all sixteen Metal traces.

## Method

The same as the first run: `examples/logit_trace` over `tests/fixtures/wikitext2_excerpt.txt` at `w1` = `1 128 8 0`, `w8` = `8 80 8 512` and `w256` = `256 2 8 0` (`CHUNK_TOKENS MAX_CHUNKS TOPK PREFILL`), compared by `python3 scripts/compare_logit_traces.py <metal.tsv> <rocm.tsv> --decided 2.0` with Metal as the reference and ROCm as the candidate. See the first run for why decided-position disagreement, not byte identity, is the criterion.

The pass criterion is the first run's: a pair passes when none of the reference's decided positions (top-two gap of at least 2.0) disagrees. A pair whose reference has no decided position (`0 / 0`) carries no evidence either way and is reported as inconclusive, not as a pass.

One shape is added. `gemma-3-4b-it` has a 1024-token sliding window, and the standard widths never give it more than 520 tokens of context, so they never exercise the window. `w8ctx1536` = `8 40 8 1536` with `MLXCEL_TRACE_START_TOKEN=1536` scores 320 positions with 1536 tokens of context each.

The VLM row traces the language model of a VLM loaded on the default path; `logit_trace` feeds text only, so the vision tower is covered by the generation check below and not by a trace, on either backend.

## Results against Metal

Ran with `python3 scripts/compare_logit_traces.py <metal.tsv> <rocm.tsv> --decided 2.0`, Metal as the reference. "Decided mismatches" reads as `disagreements / decided positions`, counted on the Metal side, so `0 / 0` and `0 / 287` are different statements. "Largest gap" is the largest Metal top-two gap at any top-1 disagreement. The logit delta is on the Metal top-1 token, between the two backends. The perplexity delta is ROCm against Metal, higher is worse. The verdict is the script's.

| Model | Width | Top-1 disagreement | Decided mismatches | Rate | Largest gap | Logit delta p50 / p90 / p99 / max | Perplexity delta | Verdict |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-7b-instruct | w1 | 3 / 128 (2.344%) | 0 / 19 | 0.000% | 0.008 | 0.0156 / 0.0469 / 0.0703 / 0.0781 | +0.219% | pass, rounding class |
| qwen2.5-7b-instruct | w8 | 4 / 640 (0.625%) | 0 / 274 | 0.000% | 0.047 | 0.0156 / 0.0312 / 0.1250 / 0.3438 | -0.030% | pass, rounding class |
| qwen2.5-7b-instruct | w256 | 2 / 512 (0.391%) | 0 / 204 | 0.000% | 0.008 | 0.0078 / 0.0156 / 0.0312 / 0.0859 | -0.010% | pass, rounding class |
| gemma-3-4b-it | w1 | 9 / 128 (7.031%) | 0 / 2 | 0.000% | 0.250 | 0.1250 / 0.2500 / 0.5000 / 0.5000 | -0.211% | pass, rounding class |
| gemma-3-4b-it | w8 | 10 / 640 (1.562%) | 0 / 320 | 0.000% | 0.250 | 0.1250 / 0.3750 / 1.2500 / 2.0000 | -0.326% | pass, rounding class |
| gemma-3-4b-it | w256 | 19 / 512 (3.711%) | 0 / 204 | 0.000% | 0.375 | 0.1250 / 0.2500 / 1.0000 / 4.5000 | +0.890% | pass, rounding class |
| gemma-3-4b-it | w8ctx1536 | 5 / 320 (1.562%) | 0 / 189 | 0.000% | 0.125 | 0.1250 / 0.3750 / 1.0000 / 1.5000 | -0.341% | pass, rounding class |
| granite-4.0-h-tiny | w1 | 0 / 128 (0.000%) | 0 / 112 | 0.000% | none, every position agrees | 0.1250 / 0.5000 / 1.0000 / 1.0000 | +4.345% | pass, byte-identical in effect |
| granite-4.0-h-tiny | w8 | 18 / 640 (2.812%) | 0 / 260 | 0.000% | 0.500 | 0.1250 / 0.2500 / 0.7500 / 2.0000 | +0.302% | pass, rounding class |
| granite-4.0-h-tiny | w256 | 18 / 512 (3.516%) | 0 / 208 | 0.000% | 0.500 | 0.1250 / 0.2500 / 0.8750 / 1.5000 | +0.349% | pass, rounding class |
| nemotron-3-nano-30b-a3b | w1 | 20 / 128 (15.625%) | 0 / 0 | n/a | 0.688 | 0.1250 / 0.5625 / 0.9375 / 1.5625 | -0.181% | inconclusive, no decided position |
| nemotron-3-nano-30b-a3b | w8 | 25 / 640 (3.906%) | 0 / 287 | 0.000% | 0.750 | 0.1250 / 0.2500 / 0.6250 / 1.5000 | +0.248% | pass, rounding class |
| nemotron-3-nano-30b-a3b | w256 | 18 / 512 (3.516%) | 0 / 216 | 0.000% | 1.250 | 0.1250 / 0.2500 / 0.6250 / 1.0625 | +0.396% | pass, rounding class |
| qwen2.5-vl-3b-instruct | w1 | 1 / 128 (0.781%) | 0 / 49 | 0.000% | 0.000 | 0.0156 / 0.0469 / 0.0625 / 0.0625 | -0.107% | pass, rounding class |
| qwen2.5-vl-3b-instruct | w8 | 3 / 640 (0.469%) | 0 / 198 | 0.000% | 0.016 | 0.0078 / 0.0156 / 0.0781 / 0.4062 | +0.066% | pass, rounding class |
| qwen2.5-vl-3b-instruct | w256 | 0 / 512 (0.000%) | 0 / 154 | 0.000% | none, every position agrees | 0.0000 / 0.0156 / 0.0312 / 0.0938 | +0.006% | pass, byte-identical in effect |

"Rounding class" is the script's verdict `the arms differ only where the reference was undecided, which is the rounding class rather than a behaviour change`; "byte-identical in effect" is `every position agrees` (top-1 everywhere, not equal logits); "inconclusive" is `the reference has no decided position at this threshold`.

The one inconclusive pair is `nemotron-3-nano-30b-a3b` at `w1`. Its 128 single-token chunks have no context and no BOS (#1785), and the Metal reference never leads its runner-up by 2.0 at any of them, so the gate has nothing to judge; the first run's `qwen3-0.6b` and `llama-3.1-8b-instruct` `w1` rows were `0 / 0` for the same reason. Its disagreement rate, 15.6%, is the highest in the set, all of it at gaps below 1.0 (17 of 92 positions under 0.5, 3 of 34 between 0.5 and 1.0), and one of the 20 puts the Metal token at ROCm's rank 8. The same checkpoint at `w8` and `w256` has 287 and 216 decided positions with no mismatch, so the model is measured, but not at a single-token shape.

### The threshold, and what it does and does not decide

The same sixteen pairs at three thresholds, as `disagreements / decided positions`:

| Model | Width | gap >= 0.5 | gap >= 1.0 | gap >= 2.0 |
|---|---|---|---|---|
| qwen2.5-7b-instruct | w1 | 0 / 75 | 0 / 41 | 0 / 19 |
| qwen2.5-7b-instruct | w8 | 0 / 495 | 0 / 394 | 0 / 274 |
| qwen2.5-7b-instruct | w256 | 0 / 381 | 0 / 289 | 0 / 204 |
| gemma-3-4b-it | w1 | 0 / 75 | 0 / 30 | 0 / 2 |
| gemma-3-4b-it | w8 | 0 / 526 | 0 / 437 | 0 / 320 |
| gemma-3-4b-it | w256 | 0 / 403 | 0 / 312 | 0 / 204 |
| gemma-3-4b-it | w8ctx1536 | 0 / 276 | 0 / 240 | 0 / 189 |
| granite-4.0-h-tiny | w1 | 0 / 128 | 0 / 128 | 0 / 112 |
| granite-4.0-h-tiny | w8 | 1 / 498 | 0 / 392 | 0 / 260 |
| granite-4.0-h-tiny | w256 | 1 / 401 | 0 / 299 | 0 / 208 |
| nemotron-3-nano-30b-a3b | w1 | 3 / 36 | 0 / 2 | 0 / 0 |
| nemotron-3-nano-30b-a3b | w8 | 2 / 515 | 0 / 414 | 0 / 287 |
| nemotron-3-nano-30b-a3b | w256 | 3 / 427 | 1 / 334 | 0 / 216 |
| qwen2.5-vl-3b-instruct | w1 | 0 / 117 | 0 / 89 | 0 / 49 |
| qwen2.5-vl-3b-instruct | w8 | 0 / 458 | 0 / 346 | 0 / 198 |
| qwen2.5-vl-3b-instruct | w256 | 0 / 348 | 0 / 260 | 0 / 154 |

The dense models, the sliding-window model and the VLM are zero at every threshold. The two hybrids are not: granite has two disagreements above a 0.5 gap and Nemotron-H has eight above 0.5 and one above 1.0, so for them the zero at 2.0 is a statement about where the line was drawn. Stated without a threshold, the largest reference gap at which the backends disagree in this run is 1.250 logits, in `nemotron-3-nano-30b-a3b` `w256`, where the Metal token is ROCm's rank 2. The next largest are 0.750 (`nemotron-3-nano-30b-a3b` `w8`) and 0.688 (`nemotron-3-nano-30b-a3b` `w1`). Any threshold at or above 1.3 gives the same all-zero result, and 2.0 clears the observed maximum. The first run's maximum was 1.125, in its 128-expert MoE; the 128-expert MoE here is again the one that moves the most, which is what routing on small score differences predicts.

### Where the disagreements land

Across the sixteen pairs there are 155 top-1 disagreements, by the rank the Metal choice holds in the ROCm top-8:

| Rank of the Metal token on ROCm | Count |
|---|---|
| 2 | 142 |
| 3 | 9 |
| 4 | 3 |
| 8 | 1 |
| beyond top-8 | 0 |

91.6% are a straight swap of two adjacent candidates. The rank-8 case is in `nemotron-3-nano-30b-a3b` `w1`, an undecided position (all of that pair's disagreements are at gaps under 1.0).

Perplexity deltas stay inside +/- 0.9% except `granite-4.0-h-tiny` `w1` at +4.345%. That is a 128-position no-context run, the kind the first run set aside as too small to mean anything, but it is the largest shift in this run, so it was checked per position. Per position, ROCm's NLL there is higher by 0.042 nats on average with a standard error of 0.017, and higher at 73 of 128 positions, so it is about 2.5 standard errors from zero: weak evidence of a small bias, not noise that can be dismissed outright. It does not reach the token: all 128 positions agree on top-1, 112 of them decided. Its Mamba2 layers run the same SSD graph code on both backends at this width (see below), so this is not a kernel-against-graph effect. At `w8` and `w256` the mean NLL shift is 0.003 and 0.004 nats. The sign of the perplexity delta is otherwise mixed across pairs, as in the first run.

## What is not like-for-like

### The reference is an M5, not an M1 Ultra

The first run's reference was an M1 Ultra, Apple GPU generation 13, which has no NAX path. This run's is an M5 Max, generation 17. At the MLX pin `81ba1c6a`, `metal::is_nax_available()` is true on generation 17 or later (18 or later for the `p` architecture suffix) under macOS 26.2 or later, and it gates NAX variants of the quantized matmul (`qmm`, `gather_qmm`, for `K` a multiple of 64 and non-f32 activations), the dense GEMM, `gather_mm_rhs`, and the full self-attention SDPA for the head dims 64, 96, 128 and 256. The single-token paths (`qmv`, `gather_qmv`, vector SDPA) have no NAX gate at the pin. The metallib hash is the same as the M1 Ultra run's, so the difference is kernel selection at run time, not a different build. Which kernel each op actually took was not logged.

What that changes:

- The multi-token widths (`w8`, `w256`, `w8ctx1536`) on the Metal side most likely ran NAX kernels for their matmuls and prefill attention, a different accumulation order from the kernels an M1 Ultra runs. The `w1` widths most likely ran the same kernel families as on an M1 Ultra.
- The two runs' bounds are against different references. The first run's 1.125 and this run's 1.250 are each a Metal-to-ROCm figure, but not against the same Metal arithmetic, and there are no M1 Ultra traces of these five checkpoints to separate the M5's own contribution. The comparison here is M5 Metal against gfx1151 ROCm, and nothing more.
- `examples/logit_trace` documents that byte identity does not hold on Apple GPU generation 15 and later for reasons the caller does not choose, so a rerun on another M5-class host is not expected to reproduce these Metal traces bit for bit either. The decided-position criterion does not depend on that.

### Two MoE implementations, and a kernel pairing that did not occur

Both sides ran `default`, with no MoE or kernel override. The trace directory's README named two pairings where `default` means a different computation on each backend. Checked against the code at `d1128266`, one of them holds and one does not:

- `nemotron-3-nano-30b-a3b` at every width: Metal takes `fused_moe_forward` (its C++ graph path, since `MLXCEL_FUSED_MOE_RELU2` was unset); ROCm takes `forward_nonfused`, because `use_fused` in `src/models/nemotron_h.rs` requires `custom_kernels_available()`. No environment variable selects `forward_nonfused` on Metal, so there is no Metal control for this pairing. All three Nemotron-H pairs measure two MoE implementations as well as two backends. Its `w8` and `w256` pairs pass with 287 and 216 decided positions, and the three Nemotron-H pairs hold the three largest disagreement gaps of the run (1.250, 0.750, 0.688).
- `granite-4.0-h-tiny` and `nemotron-3-nano-30b-a3b` at `w1`: the README expected Metal to run the fused SSM update kernel for these single-token steps and ROCm the SSD graph. That is not what these traces ran. The fused step is taken only when the cache already holds an SSM state (`ssm_step_kernel` in `src/models/granitemoehybrid.rs`; `forward_fused` in `src/models/nemotron_h.rs` also needs a conv state), and `logit_trace` builds fresh caches for every chunk. A `w1` chunk has `PREFILL` 0, so its one token always meets an empty cache, and Metal falls back to the same `ssm_step` graph ROCm runs. `w8` and `w256` chunks are 8 and 256 tokens, which never take the single-token branch. So the Mamba2 layers run the same SSD graph code on both backends at every width, and the hybrid `w1` pairs are graph against graph. The fused SSM update kernel on Metal is compared only by the two `w1ctx512` rows in [the next section](#the-fused-ssm-kernel-against-the-graph-with-state).

granite's MoE runs `SwitchGLU::forward` (`gather_qmm`) on both backends, so every granite pair, like the dense, sliding-window and VLM pairs, runs the same model code on both sides; what differs is the backend and, on Metal, the NAX kernel selection above.

### The VLM row is the language model only

`qwen2.5-vl-3b-instruct` is loaded on the default VLM path, but `logit_trace` feeds text, so all three pairs trace the language model and none touches the vision tower or the image-token merge. Its passing rows say nothing about image inputs; the image generation check under [Generation on ROCm](#generation-on-rocm) is the only coverage of that path, and it is ROCm-only.

### Nemotron-H traces carry loader lines on stdout

`src/models/nemotron_h.rs` prints five `[NemotronH] ...` loading messages with `println!`, so they land in the trace on stdout, on both backends, ahead of the `#` header. `compare_logit_traces.py` does not skip them and exits with `ValueError: not enough values to unpack`. The three Nemotron-H comparisons above were run on copies with those five lines removed (`grep -v '^\[NemotronH\] '`), which drops no data row; the committed traces are unchanged. The other four models print nothing to stdout.

## The fused SSM kernel against the graph with state

The sixteen pairs above never run Metal's fused SSM update kernel, so two rows were added that do. `w1ctx512` = `1 128 8 512` with `MLXCEL_TRACE_START_TOKEN=512`: every one of the 128 chunks prefills the 512 corpus tokens before it into a fresh cache (the prefill's rows are discarded) and then traces one token. That single-token forward meets an existing SSM state, which is the condition for the fused step: `seq_len == 1 && ssm_kernel_available()` with an `ssm_state` in the cache (`src/models/granitemoehybrid.rs`, where `ssm_step_kernel` is taken; `src/models/nemotron_h.rs`, where `forward_fused` also needs a `conv_state`). On Metal granite's Mamba2 layers therefore call `ssm_update_kernel` through `ssm_step_kernel`, and Nemotron-H's call `fused_mamba2_forward`, which fuses the single-token mixer (input projection, convolution, the same `ssm_update_kernel`, output projection); on ROCm `ssm_kernel_available()` is false, so the same forward runs `ssm_step`, the SSD graph, with the prefilled state. It is the only pairing in this document that compares that kernel with the graph.

Apple has no runtime switch that forces the graph path, so fused against graph could not be A/B tested on Metal itself. The routing above is read from the code, not observed.

### Traces and commits

- Metal: `benchmarks/logit_traces/metal_m5_3c9edea0/`, the same M5 Max, built at `3c9edea0` with `cargo build --release --features metal,accelerate --example logit_trace` (binary sha256 `decf1fa9...`, metallib `dca1bb42...` as before). The same two rows were first traced at `d1128266` (`metal_m5_d1128266/*_w1ctx512.tsv`); every data row of those is byte-identical to the `3c9edea0` traces, so the results below hold against either.
- ROCm: `benchmarks/logit_traces/rocm_gfx1151_3c9edea0/`, built at `3c9edea0` (`origin/main`, the #2082 merge) with `cargo build --release --features rocm --example logit_trace`, binary sha256 `00686315e5fbd9a4fcb9aad26a525b4d89ed60e37acf7a855edbcec2f1a66ae6`. The build ran on this PR's branch at `8e650514`, which differs from `3c9edea0` only under `benchmarks/` and `docs/`.

Both halves share a commit, but it is later than the `c5fe9a16` / `d1128266` pair of the other sixteen rows, and between them main merged ROCm fixes. Where they sit relative to the SSM-with-state path:

- #2070 (LOCAL_FIXES item 24): `SliceUpdate` and `DynamicSliceUpdate` no longer donate a source that is still referenced. Near the path, not on it: the Mamba2 conv state is carried by `concatenate` and the SSM state by `ssm_step`'s return value, neither through `SliceUpdate`; the attention layers' KV-cache writes after the prefill do go through `slice_update`, but they drop the source before evaluation, a pattern #2070 still donates and which it did not find affected (its live cases were DeepSeek-V4 pooling windows and rotating-cache snapshots).
- #2076 (LOCAL_FIXES item 26): HIP objects rebuild when an included header changes. Build-only, but it matters here: the SSD graph's `segsum` scan depends on the `get_2d_grid_dims` header fix (item 22), and before #2076 an incremental build could keep a HIP object compiled against the old header.
- #2079 (LOCAL_FIXES item 27): CPU-stream BLAS runs single-threaded over fine-grained memory. Only for ops on a CPU stream; `logit_trace` runs on the GPU stream (the trace header reads `# device GPU`).
- #2071 (LOCAL_FIXES item 23): `ScaledDotProductAttention::use_fallback` checks the stream device. Only for SDPA on a CPU stream. The same PR made `logit_trace` print a `# device` header line, so the `3c9edea0` traces on both sides have one more line than the `d1128266` ones; `compare_logit_traces.py` reads every `#` line as metadata, so it needs no filtering.
- #2073 (FFT cache size validation) and #2078 (overlay records, a comment in `rope.hip`) are not on this path.

### Results

`python3 scripts/compare_logit_traces.py <metal.tsv> <rocm.tsv> --decided 2.0`, Metal as the reference, both Nemotron-H files filtered with `grep -v '^\[NemotronH\] '` for the comparison only. Run against the `d1128266` Metal traces instead, every figure is the same. Columns as in [Results against Metal](#results-against-metal).

| Model | Width | Top-1 disagreement | Decided mismatches | Rate | Largest gap | Logit delta p50 / p90 / p99 / max | Perplexity Metal / ROCm | Perplexity delta | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| granite-4.0-h-tiny | w1ctx512 | 4 / 128 (3.125%) | 0 / 60 | 0.000% | 0.125 | 0.1250 / 0.3750 / 0.7500 / 1.0000 | 7.090 / 7.234 | +2.024% | pass, rounding class |
| nemotron-3-nano-30b-a3b | w1ctx512 | 7 / 128 (5.469%) | 0 / 71 | 0.000% | 0.250 | 0.1250 / 0.3750 / 0.6250 / 0.6250 | 7.489 / 7.489 | -0.003% | pass, rounding class |

At the lower thresholds, as `disagreements / decided positions`:

| Model | Width | gap >= 0.5 | gap >= 1.0 | gap >= 2.0 |
|---|---|---|---|---|
| granite-4.0-h-tiny | w1ctx512 | 0 / 104 | 0 / 84 | 0 / 60 |
| nemotron-3-nano-30b-a3b | w1ctx512 | 0 / 106 | 0 / 89 | 0 / 71 |

Every disagreement in both rows is at a reference gap under 0.5 and puts the Metal token at ROCm's rank 2, so the result does not depend on the threshold: both rows are zero at 0.5, 1.0 and 2.0. The perplexities, 7.09 and 7.49 on Metal, are those of a model with 512 tokens of context, unlike the no-context `w1` rows.

granite's +2.024% is the largest perplexity shift after its own `w1` row, so it was checked per position as that one was: ROCm's NLL is higher by 0.020 nats on average with a standard error of 0.020 (about 1.0 standard error), higher at 66 of 128 positions and lower at 58, with the largest single-position difference 2.25 nats. That is within noise for 128 positions. Nemotron-H's mean NLL shift is 0.000 nats (standard error 0.009).

What these rows establish: over 128 decode steps with a 512-token state, ROCm's SSD graph with state and Metal's fused SSM update kernel agree at every decided position, for both hybrids. For Nemotron-H the pair also differs in the MoE path (`fused_moe_forward` on Metal, `forward_nonfused` on ROCm), as every Nemotron-H pair does. What they do not establish: they compare the kernel with the graph across two backends, not the kernel with the graph on one backend; the kernel-against-kernel comparison needs the ROCm port in #1814.

## ROCm rows

ROCm side only, as first recorded. Perplexity and top-1 accuracy are against the corpus token, not against Metal; they are not the gate. "ROCm positions with gap >= 2.0" counts ROCm's own decided positions; the comparison above counts them on the Metal side, which is the reference, so the two counts differ by a few positions.

| Model | Width | Positions | NaN | Perplexity | Top-1 = corpus token | ROCm positions with gap >= 2.0 | Against Metal |
|---|---|---|---|---|---|---|---|
| qwen2.5-7b-instruct | w1 | 128 | 0 | 6292 | 2.3% | 19 | pass |
| qwen2.5-7b-instruct | w8 | 640 | 0 | 11.40 | 53.9% | 273 | pass |
| qwen2.5-7b-instruct | w256 | 512 | 0 | 13.50 | 52.1% | 203 | pass |
| gemma-3-4b-it | w1 | 128 | 0 | 775911 | 3.9% | 1 | pass |
| gemma-3-4b-it | w8 | 640 | 0 | 25.34 | 50.8% | 323 | pass |
| gemma-3-4b-it | w256 | 512 | 0 | 53.73 | 39.8% | 200 | pass |
| gemma-3-4b-it | w8ctx1536 | 320 | 0 | 9.34 | 60.3% | 185 | pass |
| granite-4.0-h-tiny | w1 | 128 | 0 | 133111 | 2.3% | 116 | pass |
| granite-4.0-h-tiny | w8 | 640 | 0 | 14.33 | 53.1% | 260 | pass |
| granite-4.0-h-tiny | w256 | 512 | 0 | 19.07 | 50.0% | 207 | pass |
| nemotron-3-nano-30b-a3b | w1 | 128 | 0 | 14868 | 1.6% | 0 | inconclusive |
| nemotron-3-nano-30b-a3b | w8 | 640 | 0 | 10.98 | 54.8% | 286 | pass |
| nemotron-3-nano-30b-a3b | w256 | 512 | 0 | 12.03 | 52.5% | 215 | pass |
| qwen2.5-vl-3b-instruct | w1 | 128 | 0 | 66669 | 0.8% | 47 | pass |
| qwen2.5-vl-3b-instruct | w8 | 640 | 0 | 21.02 | 47.7% | 197 | pass |
| qwen2.5-vl-3b-instruct | w256 | 512 | 0 | 23.67 | 45.7% | 154 | pass |

The `w1` perplexities are huge because each `w1` chunk is a single token scored with no context and no BOS (#1785); the first run's `w1` traces look the same (`llama-3.1-8b-instruct` `w1` on ROCm: 719878), and the Metal `w1` perplexities are of the same size (for example granite 127568, Nemotron-H 14895). It is a property of the shape, the same on both backends, and says nothing about the backend.

Which paths ROCm took, from the code at `c5fe9a16`: granite's MoE goes through `SwitchGLU::forward` (`gather_qmm`) on every backend; Nemotron-H's MoE takes `forward_nonfused` on ROCm because `custom_kernels_available()` is false there; the Mamba2 layers of both hybrids run the SSD graph path at every width because `ssm_kernel_available()` is false on ROCm. Metal runs the fused SSM update only for a single-token step with state; none of these sixteen rows has one, so Metal ran the same graph. The two `w1ctx512` rows, which do, are in [their own section](#the-fused-ssm-kernel-against-the-graph-with-state). [What is not like-for-like](#what-is-not-like-for-like) covers what this does to the comparison.

## Generation on ROCm

`mlxcel generate -t 0` with the release binary from `c5fe9a16`. Every checkpoint produced fluent, on-topic text with no NaN, error or abort. Free-running output is a smoke check only.

| Model | Prompt | Result |
|---|---|---|
| Qwen2.5-7B-Instruct-4bit | one sentence on the history of Paris | 39 tokens, "Paris has a rich history dating back to its Celtic origins in the 3rd century BC, ..." |
| gemma-3-4b-it-4bit | same | 49 tokens, "Emerging from a small Celtic settlement known as Lutetia, Paris has grown ..." |
| granite-4.0-h-tiny-4bit | same | 59 tokens, "Paris, the capital of France, was founded in the 3rd century BC by the Parisii, ..." |
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit | same, `--show-reasoning`, `-n 256` | 140 tokens: a reasoning block, then "Paris, originally settled as the Roman town of Lutetia in the 1st century BCE, ..." |
| Qwen2.5-VL-3B-Instruct-4bit | same | 33 tokens, "Paris, the capital of France, has a rich history dating back to Roman times ..." |
| Qwen2.5-VL-3B-Instruct-4bit | `--image tests/fixtures/test_image_shapes.png`, describe it | 17 tokens, "The image consists of a red square, a blue circle, and a green triangle." |
| gemma-3-4b-it-4bit | same image and prompt | 41 tokens, "An image shows a white square, with a red square on top of it, a blue circle on the right, and a green triangle on the left." |

Nemotron-3-Nano is a thinking model: at `-n 64` without `--show-reasoning` its whole budget stays in the reasoning channel and the visible answer is empty, which is the behavior the first run recorded for `Qwen3-0.6B-4bit`, not a backend fault.

## Defect found: strided scans launched too many blocks

The first attempt at the SSM-hybrid rows faulted the GPU. `granite-4.0-h-tiny` at `w8` and `nemotron-3-nano-30b-a3b` at `w8` and `w256` ended in `HSA_STATUS_ERROR_MEMORY_FAULT` inside `strided_scan<float, float, Sum, 4, 32, 32, true, false>`, on a build of `origin/main` `4595b06f`.

The scan was the SSD `segsum`, a `cumsum` along the second-to-last axis of `[1, heads, L, L]`. `Scan::eval_gpu` sizes that grid with `get_2d_grid_dims(shape, strides, axis_size * stride)`, and the ROCm overlay carried an older local body for that overload which reduced the divisor only by dimensions that divided it whole. For granite's 48 heads and a 24-token chunk it returned 576 blocks where 48 cover the array, and each extra block read and wrote past the end. Whether a shape was hit depended on how the head count and the chunk length share factors, which is why 8- and 16-token chunks on granite and every `w1` trace ran. The overload now delegates to the shared `get_2d_grid_dims_common`, as the CUDA backend does (`LOCAL_FIXES.md` item 22), and `tests/rocm_strided_scan.rs` compares GPU and CPU scans exactly; with the old header its `[64, 96, 32, 32]` case faults. Every trace in this document is from the fixed build.

This is the value of the SSM rows independent of the Metal comparison: before the fix, an SSM-hybrid model on ROCm could fault on a multi-token forward depending on its chunk length.

## The Nemotron-H `use_fused` guard: a fix for the opt-in path, defensive for the default

PR #2029 added `&& mlxcel_core::custom_kernels_available()` to `use_fused` in `src/models/nemotron_h.rs` without having run a Nemotron-H checkpoint on ROCm. It is now run. Built at `680eb064` (`d8d34e2b^`, the commit before the guard) and at `4595b06f`:

| Build | `MLXCEL_FUSED_MOE_RELU2` | Result |
|---|---|---|
| `680eb064`, before the guard | unset | generates (24 tokens, "Paris." ...) |
| `680eb064`, before the guard | `1` | aborts: `terminate called after throwing an instance of 'std::runtime_error'`, `what(): [metal_kernel] No Metal back-end.`, exit 134 |
| `4595b06f`, with the guard | unset | generates |
| `4595b06f`, with the guard | `1` | generates |

So the issue's trace was half right. Before the guard, `fused_moe_forward` reached a custom kernel only on its opt-in `MLXCEL_FUSED_MOE_RELU2` branch (a single-token step with 4- or 8-bit weights); by default it computed the MoE with MLX graph ops inside the bridge and never resolved a port, so the default path did not abort. With the opt-in set it launched the Metal-only `fc1_relu2` kernel and the throw crossed a `noexcept` extern into `std::terminate`. The guard is therefore a confirmed fix for the opt-in path and defensive for the default one, and it changes the default path on ROCm from the bridge's graph MoE to `forward_nonfused`. It does not change the port order in #1814: `moe_fc1_relu2_ports()` stays an opt-in, Metal-only entry.

## The test gate

`make verify-test-rocm` (part of `make verify-rocm`) failed on `origin/main` `4595b06f` with 37 tests: 34 tests of the fused paged-attention kernels, which have no ROCm port (#1814), the bf16 `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, and two tests from #2037 (`models::gemma3_backbone::gemma3_backbone_tests::gelu_approx_matches_mlx_nn_bit_for_bit` and `tests::family_order_is_exhaustive`). The NVFP4 abort that the issue recorded earlier is gone, so later-sorting `mlxcel-core` tests run again.

The 34 paged-attention tests no longer fail. Each paged-attention kernel now exports a support predicate that reads its own `KernelPorts` table through `has_kernel_port`, bridged to Rust as `paged_attention_decode_available` (v1 decode), `paged_attention_v2_available` (v2 partial and merge), `paged_attention_merge_available` (merge) and `paged_attention_kernels_available` (all three). The tests return early through one helper, `src/lib/mlxcel-core/src/test_support/kernel_ports.rs`, only when the backend is ROCm and the predicate for the kernels that test needs is false, and each skip prints `skipping <module>:<line>: ROCm has no <kernels> kernel port yet (lablup/mlxcel#1814) ...` to stderr so the gate log lists them. When #1814 fills a `.rocm` entry the predicates that depend on it turn true and those tests run again with no edit, even if the kernels are ported one at a time. On Metal and CUDA every predicate is true, so nothing is skipped there. Two more `ffi_tests` paged-decode tests, which used to return early off Metal and CUDA without saying so, now skip the same way on ROCm. The production paths in front of these kernels (batched paged decode, MLA split-KV, `paged_decode_backend`, and the sparse paged decode, which had no port check and on ROCm reported a refused launch as a rejected plan on every layer of every step) now ask the same predicates instead of the backend-wide `custom_kernels_available()`.

The remaining three failures belong to other work and are left as they are. The counts of the gate run for this change are in the PR that added this document (#1809's PR), measured on the rebased branch immediately before merge.

## Reproducing

```bash
cargo build --release --features rocm --example logit_trace --bin mlxcel
./target/release/examples/logit_trace models/mlx/granite-4.0-h-tiny-4bit tests/fixtures/wikitext2_excerpt.txt 8 80 8 512 > rocm_w8.tsv
MLXCEL_TRACE_START_TOKEN=1536 ./target/release/examples/logit_trace models/mlx/gemma-3-4b-it-4bit tests/fixtures/wikitext2_excerpt.txt 8 40 8 1536 > rocm_w8ctx1536.tsv
MLXCEL_TRACE_START_TOKEN=512 ./target/release/examples/logit_trace models/mlx/granite-4.0-h-tiny-4bit tests/fixtures/wikitext2_excerpt.txt 1 128 8 512 > rocm_w1ctx512.tsv
cargo test --features rocm --test rocm_strided_scan -- --test-threads=1
MLXCEL_FUSED_MOE_RELU2=1 ./target/release/mlxcel generate -m models/mlx/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit -p "The capital of France is" -n 24 -t 0 --no-chat-template
```

The comparison, from the committed traces (no GPU needed). The Nemotron-H traces need their five loader lines removed first, on both sides, as described above:

```bash
M=benchmarks/logit_traces/metal_m5_d1128266/metal_m5_d1128266
R=benchmarks/logit_traces/rocm_gfx1151_c5fe9a16/rocm_gfx1151_c5fe9a16
python3 scripts/compare_logit_traces.py ${M}_granite-4.0-h-tiny_default_w8.tsv ${R}_granite-4.0-h-tiny_default_w8.tsv --decided 2.0
python3 scripts/compare_logit_traces.py \
    <(grep -v '^\[NemotronH\] ' ${M}_nemotron-3-nano-30b-a3b_default_w256.tsv) \
    <(grep -v '^\[NemotronH\] ' ${R}_nemotron-3-nano-30b-a3b_default_w256.tsv) --decided 2.0
M2=benchmarks/logit_traces/metal_m5_3c9edea0/metal_m5_3c9edea0
R2=benchmarks/logit_traces/rocm_gfx1151_3c9edea0/rocm_gfx1151_3c9edea0
python3 scripts/compare_logit_traces.py ${M2}_granite-4.0-h-tiny_default_w1ctx512.tsv ${R2}_granite-4.0-h-tiny_default_w1ctx512.tsv --decided 2.0
```

The Metal traces were produced with the loop in `benchmarks/logit_traces/rocm_gfx1151_c5fe9a16/README.md`, built with `cargo build --release --features metal,accelerate --example logit_trace` at `d1128266`; `metal_m5_d1128266/METADATA.txt` and `RUNS.txt` record the host, hashes, arguments and the exit status and row count of every run.

## Open

- The second acceptance criterion of #1809 (decided-position mismatch reported per model and width against Metal) is now met for all nine models: fifteen of these sixteen pairs pass and `nemotron-3-nano-30b-a3b` `w1` is inconclusive, with the same checkpoint's `w8` and `w256` pairs passing. The two `w1ctx512` rows, which cover the fused SSM update kernel on Metal, also pass.
- Kernel against kernel for the SSM update: the `w1ctx512` rows here compare Metal's kernel with ROCm's graph. The HIP port (#2067) added kernel-against-kernel rows, with zero decided-position mismatches for both hybrids, in [rocm-ssm-update-kernel-gfx1151-2026-10-04.md](rocm-ssm-update-kernel-gfx1151-2026-10-04.md#model-logits-kernel-against-kernel).
- Two MoE implementations in the Nemotron-H pairs: since the HIP fused MoE ports (#2065) ROCm's gate reads `moe_down_kernel_available()` and takes `fused_moe_forward`, as Metal does. New Nemotron-H traces on that path pass against these Metal traces at `w8` and `w1ctx512` with zero decided-position mismatches, in [rocm-fused-moe-gfx1151-2026-10-05.md](rocm-fused-moe-gfx1151-2026-10-05.md#model-logits).
- `src/models/nemotron_h.rs` prints its loading messages to stdout, which puts five non-trace lines into every Nemotron-H trace and stops `compare_logit_traces.py`. Either the loader should log to stderr or the comparison script should skip them.
- No noise floor was measured; the first run's reasoning for not needing one still applies.
