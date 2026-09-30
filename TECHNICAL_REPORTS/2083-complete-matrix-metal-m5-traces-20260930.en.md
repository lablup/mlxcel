# Technical Report: PR #2083 - Complete the #1809 matrix rows with Metal M5 traces

**Date**: 2026-09-30

**Status**: Traces and comparison committed on `update/issue-1809-metal-m5-rows`; pending merge.

**Languages**: Markdown (correctness doc, trace README), TSV logit traces with their metadata

**Risk Level**: Low (trace data and documentation only; no model, kernel, script or runtime code changes)

## Executive Summary

PR #2059 traced five checkpoints on ROCm gfx1151 for issue #1809 (a second dense model, a sliding-window model, two SSM hybrids and a VLM), but no Metal host was reachable from the ROCm host, so every row of `docs/benchmark_results/rocm-correctness-gfx1151-2026-09-30.md` was left without its reference. This PR brings in the Metal half, traced on an Apple M5 Max (Apple GPU generation 17, NAX path available), and completes the comparison with the criterion of the first run (2026-09-12): a pair passes when none of the reference's decided positions (top-two gap of at least 2.0) disagrees.

- **The sixteen original pairs**: 0 decided mismatches in every row. Fifteen pass. `nemotron-3-nano-30b-a3b` `w1` is inconclusive: its Metal reference has no decided position (`0 / 0`). Across the sixteen, top-1 disagrees at 155 of 6720 positions and at 0 of the 2696 decided ones. The largest reference gap at any disagreement is 1.250 logits (Nemotron-H `w256`, the Metal token at ROCm's rank 2).
- **A corrected caveat**: the hybrid `w1` rows were expected to compare Metal's fused SSM update kernel with ROCm's SSD graph. They do not. `logit_trace` builds fresh caches per chunk, the fused step needs an existing `ssm_state`, and so `w1` is graph against graph on both backends. The ROCm orchestrator and the M5 session found this independently.
- **Two new `w1ctx512` rows** prefill 512 tokens before each single-token forward, which is the first real comparison of Metal's fused `ssm_update_kernel` against ROCm's SSD graph with state. Both halves are at code commit `3c9edea0`. granite: 0 / 60 decided mismatches, largest gap 0.125. Nemotron-H: 0 / 71, largest gap 0.250. Both are zero at 0.5 and 1.0 as well.

The Metal traces came from a Remote Control session on the user's M5, which the ROCm orchestrator asked for traces; the two sides exchanged data through the branch `data/issue-1809-metal-traces`.

## 1. Problem Statement

Issue #1809 asks for a ROCm correctness matrix against a Metal baseline, reported per model and width as decided-position mismatches. The first run (#1826, 2026-09-12) covered dense and MoE checkpoints against an M1 Ultra. The issue was reopened because the sliding-window, SSM-hybrid and VLM rows were missing. #2059 traced those rows on ROCm (`benchmarks/logit_traces/rocm_gfx1151_c5fe9a16/`) and found and fixed a ROCm defect along the way (a strided-scan launch that faulted the GPU on both hybrids), but the issue comment at its merge lists the Metal side as open: "No Metal host was reachable and no Metal trace for these checkpoints is in the repository."

Without the Metal half, the ROCm traces showed only that the checkpoints load, generate and trace with no NaN. They said nothing about whether ROCm computes the same model as Metal. That is the acceptance criterion this PR closes.

The README that #2059 left for the Metal host also predicted two pairings that would not be like-for-like. One of those predictions was wrong, and finding out why changed what the matrix could claim about the SSM kernels (section 4).

## 2. Change Summary

Five commits, all under `benchmarks/` and `docs/`:

- **`bd6c5863`** (cherry-picked from `9830bd60` on the data branch, authorship kept): `benchmarks/logit_traces/metal_m5_d1128266/`, sixteen Metal traces at `d1128266` (the #2059 merge commit), with `METADATA.txt`, `RUNS.txt` and `SHA256SUMS`.
- **`556e62b2`**: fills every row of the 2026-09-30 doc against those traces, adds the threshold sweep, the rank distribution and the "What is not like-for-like" section, and updates the `rocm_gfx1151_c5fe9a16/README.md` pointer and wording.
- **`8e650514`** (from `d2043c94`): Metal `w1ctx512` traces for granite and Nemotron-H at `d1128266`, added to `metal_m5_d1128266/`.
- **`ce6b2bd7`** (from `c5706667`): the same two rows retraced at `3c9edea0` in `benchmarks/logit_traces/metal_m5_3c9edea0/`.
- **`3ca19723`**: ROCm `w1ctx512` traces at `3c9edea0` in `benchmarks/logit_traces/rocm_gfx1151_3c9edea0/`, and the doc section "The fused SSM kernel against the graph with state".

Two one-line pointer updates in `docs/installation.md` and the 2026-09-12 doc now say the new rows were compared against an M5 Max reference.

## 3. Results

### 3.1 The sixteen pairs

`python3 scripts/compare_logit_traces.py <metal.tsv> <rocm.tsv> --decided 2.0`, Metal the reference. ROCm traces at `c5fe9a16`, Metal at `d1128266`; `git diff c5fe9a16 d1128266` touches no file under `src/models/`, `examples/logit_trace.rs` or `scripts/compare_logit_traces.py`.

| Model | Width | Top-1 disagreement | Decided mismatches | Largest gap | Perplexity delta | Verdict |
|---|---|---|---|---|---|---|
| qwen2.5-7b-instruct | w1 / w8 / w256 | 3/128, 4/640, 2/512 | 0/19, 0/274, 0/204 | 0.008, 0.047, 0.008 | +0.219%, -0.030%, -0.010% | pass |
| gemma-3-4b-it | w1 / w8 / w256 | 9/128, 10/640, 19/512 | 0/2, 0/320, 0/204 | 0.250, 0.250, 0.375 | -0.211%, -0.326%, +0.890% | pass |
| gemma-3-4b-it | w8ctx1536 | 5/320 | 0/189 | 0.125 | -0.341% | pass |
| granite-4.0-h-tiny | w1 / w8 / w256 | 0/128, 18/640, 18/512 | 0/112, 0/260, 0/208 | none, 0.500, 0.500 | +4.345%, +0.302%, +0.349% | pass |
| nemotron-3-nano-30b-a3b | w1 | 20/128 | 0/0 | 0.688 | -0.181% | inconclusive |
| nemotron-3-nano-30b-a3b | w8 / w256 | 25/640, 18/512 | 0/287, 0/216 | 0.750, 1.250 | +0.248%, +0.396% | pass |
| qwen2.5-vl-3b-instruct | w1 / w8 / w256 | 1/128, 3/640, 0/512 | 0/49, 0/198, 0/154 | 0.000, 0.016, none | -0.107%, +0.066%, +0.006% | pass |

Of the 155 disagreements, 142 put the Metal token at ROCm's rank 2, 9 at rank 3, 3 at rank 4 and 1 at rank 8 (in Nemotron-H `w1`, at an undecided position); none falls outside the ROCm top-8.

The inconclusive row is the same `0 / 0` case the first run recorded for two of its `w1` rows. Each `w1` chunk is a single token with no context and no BOS (#1785), and the Metal reference never leads its runner-up by 2.0. The same checkpoint at `w8` and `w256` has 287 and 216 decided positions with no mismatch, so the model is measured, but not at a single-token shape.

### 3.2 The threshold sweep for the hybrids

The dense models, the sliding-window model and the VLM are zero at 0.5, 1.0 and 2.0. The hybrids are not, so for them the zero at 2.0 depends on where the line was drawn:

| Model | Width | gap >= 0.5 | gap >= 1.0 | gap >= 2.0 |
|---|---|---|---|---|
| granite-4.0-h-tiny | w1 | 0 / 128 | 0 / 128 | 0 / 112 |
| granite-4.0-h-tiny | w8 | 1 / 498 | 0 / 392 | 0 / 260 |
| granite-4.0-h-tiny | w256 | 1 / 401 | 0 / 299 | 0 / 208 |
| nemotron-3-nano-30b-a3b | w1 | 3 / 36 | 0 / 2 | 0 / 0 |
| nemotron-3-nano-30b-a3b | w8 | 2 / 515 | 0 / 414 | 0 / 287 |
| nemotron-3-nano-30b-a3b | w256 | 3 / 427 | 1 / 334 | 0 / 216 |

granite has two disagreements above a 0.5 gap; Nemotron-H has eight above 0.5 and one above 1.0. The threshold-free statement is the largest gap at a disagreement, 1.250, so any `--decided` at or above 1.3 gives the same all-zero result. The first run's maximum was 1.125, also in its 128-expert MoE.

### 3.3 The granite `w1` perplexity shift

granite `w1` is +4.345% with every top-1 in agreement. Per position, ROCm's NLL is higher by 0.042 nats on average with a standard error of 0.017, higher at 73 of 128 positions: about 2.5 standard errors, weak evidence of a small bias. Its Mamba2 layers run the same SSD graph on both backends at this width (section 4), so this is not a kernel-against-graph effect. At `w8` and `w256` the mean NLL shift is 0.003 and 0.004 nats.

## 4. The Corrected Caveat and the `w1ctx512` Rows

### 4.1 Why `w1` is graph against graph

The #2059 README said the hybrid `w1` rows would compare Metal's fused SSM update kernel with ROCm's SSD graph. The code at `d1128266` shows otherwise. The fused single-token step requires `seq_len == 1`, `ssm_kernel_available()` and an `ssm_state` already in the cache (`ssm_step_kernel` in `src/models/granitemoehybrid.rs`; `forward_fused` in `src/models/nemotron_h.rs` also needs a `conv_state`). `logit_trace` builds fresh caches for every chunk, and a `w1` chunk has `PREFILL` 0, so its one token always meets an empty cache and Metal falls back to the same `ssm_step` graph ROCm runs. `w8` and `w256` chunks are multi-token and never take the single-token branch. So none of the sixteen pairs reaches the fused kernel on Metal.

The ROCm orchestrator recorded this in `556e62b2` while filling the rows, and the M5 session recorded it in `d2043c94` (cherry-picked as `8e650514`) while adding traces to cover it. The two sides reached it independently. The README and the doc now state the corrected pairing instead of the predicted one.

### 4.2 The `w1ctx512` shape

`w1ctx512` = `1 128 8 512` with `MLXCEL_TRACE_START_TOKEN=512`: each of the 128 chunks prefills the 512 corpus tokens before it into a fresh cache, discards those rows, and then traces one token. That forward meets an existing SSM state. On Metal, granite's Mamba2 layers call `ssm_update_kernel` through `ssm_step_kernel`, and Nemotron-H's call `fused_mamba2_forward`, which fuses input projection, convolution, the same `ssm_update_kernel` and output projection. On ROCm `ssm_kernel_available()` is false, so the same forward runs `ssm_step`, the SSD graph, with the prefilled state. These are the only rows in the doc that compare that kernel with the graph.

### 4.3 Commits

Both halves are at code commit `3c9edea0` (`origin/main`, the #2082 merge):

- Metal `metal_m5_3c9edea0/`: logit_trace sha256 `decf1fa9...`, metallib `dca1bb42...` (unchanged from the other Metal traces). The first Metal trace of these rows, at `d1128266`, is kept in `metal_m5_d1128266/`; every data row (all non-`#` lines) of it is byte-identical to the `3c9edea0` traces, for both models, so the results hold against either. The `3c9edea0` files have one more line because `logit_trace` gained a `# device` header in #2071.
- ROCm `rocm_gfx1151_3c9edea0/`: logit_trace sha256 `00686315e5fbd9a4fcb9aad26a525b4d89ed60e37acf7a855edbcec2f1a66ae6`, built on this branch at `8e650514`, which differs from `3c9edea0` only under `benchmarks/` and `docs/`.

`3c9edea0` is later than the `c5fe9a16` / `d1128266` pair of the other sixteen rows. The doc places each ROCm fix merged in between relative to the SSM-with-state path: #2070 (`SliceUpdate` donation, near the path but not on it), #2076 (HIP header rebuild, build-only but relevant to the `segsum` scan fix), #2079 and #2071 (CPU-stream only; `logit_trace` runs on the GPU stream), #2073 and #2078 (not on the path).

### 4.4 Results

| Model | Top-1 disagreement | Decided mismatches | Largest gap | Logit delta p50 / p90 / p99 / max | Perplexity Metal / ROCm | Delta |
|---|---|---|---|---|---|---|
| granite-4.0-h-tiny | 4 / 128 (3.125%) | 0 / 60 | 0.125 | 0.1250 / 0.3750 / 0.7500 / 1.0000 | 7.090 / 7.234 | +2.024% |
| nemotron-3-nano-30b-a3b | 7 / 128 (5.469%) | 0 / 71 | 0.250 | 0.1250 / 0.3750 / 0.6250 / 0.6250 | 7.489 / 7.489 | -0.003% |

At 0.5 and 1.0: granite 0 / 104 and 0 / 84, Nemotron-H 0 / 106 and 0 / 89. Every disagreement in both rows is at a gap under 0.5 and puts the Metal token at ROCm's rank 2, so the result does not depend on the threshold.

granite's +2.024% was checked per position: ROCm's NLL is higher by 0.020 nats on average with a standard error of 0.020 (about 1.0 standard error), higher at 66 of 128 positions and lower at 58, largest single difference 2.25 nats. That is within noise for 128 positions. Nemotron-H's mean shift is 0.000 nats (standard error 0.009).

What these rows establish: over 128 decode steps with a 512-token state, ROCm's SSD graph with state and Metal's fused SSM update kernel agree at every decided position, for both hybrids. What they do not: the comparison is kernel against graph across two backends, not kernel against graph on one backend. Apple has no runtime switch that forces the graph path, so the fused routing is read from the code, not observed, and it could not be A/B tested on Metal. The kernel-against-kernel comparison needs the ROCm port in #1814.

## 5. What Is Not Like-for-Like

- **M5 NAX kernels, not the M1 Ultra of the first run.** At the MLX pin `81ba1c6a`, `metal::is_nax_available()` is true on generation 17 or later under macOS 26.2 or later. It gates NAX variants of `qmm`, `gather_qmm`, the dense GEMM, `gather_mm_rhs` and full self-attention SDPA; the single-token paths (`qmv`, `gather_qmv`, vector SDPA) have no NAX gate. The metallib hash is the same as the M1 Ultra run's, so the difference is kernel selection at run time. The multi-token widths on Metal most likely ran NAX kernels; the `w1` widths most likely ran the M1 Ultra's kernel families. Which kernel each op took was not logged. The first run's 1.125 and this run's 1.250 are therefore bounds against different Metal arithmetic, and there are no M1 Ultra traces of these checkpoints to separate the M5's contribution.
- **Nemotron-H MoE.** Metal takes `fused_moe_forward` (its C++ graph path, since `MLXCEL_FUSED_MOE_RELU2` was unset); ROCm takes `forward_nonfused`, because `use_fused` requires `custom_kernels_available()`. No environment variable selects `forward_nonfused` on Metal, so there is no Metal control. Every Nemotron-H pair, including `w1ctx512`, measures two MoE implementations as well as two backends, and the three original Nemotron-H pairs hold the three largest disagreement gaps of the run (1.250, 0.750, 0.688). granite's MoE runs `SwitchGLU::forward` (`gather_qmm`) on both sides.
- **The VLM rows trace the language model only.** `logit_trace` feeds text, so the three `qwen2.5-vl-3b-instruct` pairs never touch the vision tower or the image-token merge. The image generation check in the doc is the only coverage of that path, and it is ROCm-only.

## 6. The Cross-Machine Workflow

The ROCm orchestrator runs on the gfx1151 host, which has no Metal. Instead of leaving the rows open until a Metal host was available, it requested the Metal traces from a Remote Control session on the user's M5 Max. The two sides did not share a filesystem; they exchanged data through git:

1. The M5 session built `logit_trace` at `d1128266` using the loop in `rocm_gfx1151_c5fe9a16/README.md`, verified every checkpoint file against the pinned Hugging Face revision (sha256 for LFS files, git blob sha1 otherwise), and pushed the sixteen traces with metadata to `data/issue-1809-metal-traces` (`9830bd60`).
2. The ROCm side cherry-picked that commit with its authorship onto the PR branch and ran all comparisons (`556e62b2`).
3. The M5 session pushed `w1ctx512` traces at `d1128266` (`d2043c94`). The ROCm side built its binary at `8e650514` (code equal to `3c9edea0`) and traced its half.
4. So that the pair shared a commit, the M5 session retraced at `3c9edea0` (`c5706667`), and the ROCm side committed its traces and the comparison (`3ca19723`).

The data branch carries only trace data and metadata; everything the comparison depends on (commit, corpus hash, arguments, checkpoint revisions, binary and metallib hashes, exit status and row counts) is written into each directory's `METADATA.txt` and `RUNS.txt`, so the ROCm side could check the Metal half without access to the Mac.

## 7. Technical Decisions

- **Keep the 2026-09-12 criterion unchanged.** Zero mismatches on positions decided at `--decided 2.0`, with `0 / 0` reported as inconclusive rather than as a pass. Changing the threshold after seeing the data would have made the result unfalsifiable; the sweep and the threshold-free maximum gap are reported next to it instead, so a reader can see how much margin 2.0 leaves.
- **Report Nemotron-H `w1` as inconclusive, not as a pass.** A pair with no decided position carries no evidence either way. The same model's `w8` and `w256` rows carry the evidence.
- **Add a new shape instead of relabeling `w1`.** Once `w1` was known to be graph against graph, the fused kernel had no coverage. `w1ctx512` reuses the existing `MLXCEL_TRACE_START_TOKEN` mechanism (as `w8ctx1536` does for the sliding window) and needs no code change.
- **Retrace Metal at the ROCm commit.** The ROCm binary for `w1ctx512` was built at current main, which includes several ROCm fixes since `c5fe9a16`. Retracing Metal at `3c9edea0` makes the pair share a commit; keeping the `d1128266` traces and recording that their data rows are byte-identical shows the retrace changed nothing on the Metal side.
- **Filter the Nemotron-H loader lines on copies, not in the committed traces.** The committed files stay exactly what each binary printed, and their checksums remain valid; the filter is part of the documented comparison command.

## 8. Validation

Per the PR body, on the gfx1151 host: `sha256sum -c SHA256SUMS` in the three new or updated trace directories; `cargo build --release --features rocm --example logit_trace` at `8e650514` and the two ROCm `w1ctx512` traces (exit 0, 128 data rows, no NaN), each started with no other process in `/sys/class/kfd/kfd/proc`; all 18 comparisons; `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-rocm-overlay verify-fmt verify-binary-assets` pass; `cargo test --features rocm --test dead_doc_pointers` pass (2 tests).

Rechecked from the committed traces while writing this report: the checksums pass in `metal_m5_d1128266/`, `metal_m5_3c9edea0/` and `rocm_gfx1151_3c9edea0/`; the two `w1ctx512` comparisons reproduce the table in 4.4; the Nemotron-H `w1` comparison reproduces 20 / 128, no decided position, largest gap 0.688, verdict inconclusive; the `d1128266` and `3c9edea0` Metal `w1ctx512` data rows are identical for both models; and an unfiltered Nemotron-H comparison raises `ValueError: not enough values to unpack (expected 6, got 1)`.

## 9. Learning Points

- **Check the routing condition, not the function name.** "Single-token step on Metal" sounded like the fused kernel, but the kernel also needs state, and the harness never provided it. A caveat written from the name was wrong for every hybrid `w1` row; one read of the guard in the model code corrected it.
- **A comparison harness can silently skip the path it is meant to cover.** Building fresh caches per chunk is right for isolation, and it also means no decode step in a zero-prefill trace ever has state. Coverage claims should name the shape that reaches the path.
- **Publish a threshold-free number alongside a threshold verdict.** For the hybrids, "0 at 2.0" and "1 at 1.0" are both true; the maximum gap at a disagreement (1.250) is the number that does not depend on the choice.
- **Make trace directories self-describing.** Because each directory records commit, hashes, arguments, checkpoint revisions and run status, a trace produced on another machine by another session could be checked and used without trusting the session that produced it.
- **Stdout is part of a trace's format.** A model loader's `println!` breaks every consumer that parses stdout.

## 10. Caveats, Not Verified, and Follow-Ups

- **Metal was not run on the ROCm host.** The Metal traces are taken on their `METADATA.txt`, `RUNS.txt` and checksums; the fused-kernel routing is read from the code.
- **Not bit-reproducible on another M5.** `examples/logit_trace` documents that byte identity does not hold on Apple GPU generation 15 and later; the decided-position criterion does not depend on it.
- **Kernel against kernel for the SSM update** needs the ROCm port in #1814.
- **No noise floor was measured**; the first run's reasoning for not needing one still applies.
- **Follow-up: Nemotron-H loader output on stdout.** `src/models/nemotron_h.rs` prints five `[NemotronH] ...` lines with `println!`, on both backends, ahead of the `#` header. `compare_logit_traces.py` does not skip them and exits with `ValueError`. Every Nemotron-H comparison here ran on copies filtered with `grep -v '^\[NemotronH\] '`. Either the loader should log to stderr or the script should skip non-trace lines. The `# device` header line of newer traces is read as metadata and needs no filter.
- **Follow-up: granite `w1` perplexity.** +4.345% at about 2.5 standard errors per position, with every top-1 in agreement and graph code on both sides. Small, but not dismissible as noise on this sample.

Refs: #1809, PR #2059, PR #1826, #1814, #1785, #2070, #2071, #2076, #2079.
