# Qwen3.8-Flash-Next on M3 Ultra — 2026-10-23

This run qualifies PR #2475 against the OptiQ 2-bit checkpoint and the updated mlx-vlm reference. Both runtimes keep the text backbone resident; SSD expert streaming is not used.

## Environment and checkpoint

- Apple M3 Ultra, 512 GiB unified memory, macOS 26.6.2 (25G83).
- mlxcel 0.8.0-beta.1, PR #2475 hardening tree based on `b68efbc9` (merged main `8116e266`).
- Bundled MLX: `bff10220c742907f0c68e55ff174aaea56f74e3d`.
- Python 3.12.14, MLX 0.32.4, mlx-vlm 0.7.9, mlx-lm 0.32.0, transformers 5.16.1.
- mlx-vlm reference: `abd3b2d7b728cb5372d4e092260da325e5ded209`; mlx-lm reference: `0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`. mlx-vlm provides this architecture; mlx-lm supplies shared operations.
- Checkpoint: `mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit`, revision `52a21f4a18963bffa64f5a356e10459f2513a900`, 80,764,996,024 bytes across 20 main shards and the bf16 vision sidecar. Per-module affine widths are preserved.

## OptiQ storage compatibility

OptiQ stores the non-gated RMSNorm scales with `1 + weight` already folded. Applying the native zero-centered norm directly adds one twice: both unadapted runtimes produced identical incoherent text and a corpus perplexity above 30,000. OptiQ's converter (`optiq/mlx_lm_patches/qwen4_exp.py`, package 0.7.0, `_FOLD_ONE` and `sanitize`) documents this storage convention.

The loader recognizes the `optiq_vision` metadata and subtracts one from only those text norm weights in float32. Gated output norms and vision weights remain unchanged. The reference trace loader performs the same storage conversion, prefix normalization and sidecar loading before strict loading into unchanged mlx-vlm forward functions. The corrected corpus perplexity is 6.5903. This adjustment is essential: matching an incorrectly loaded reference does not qualify the model.

## Numerical validation

The corpus concatenates `docs/architecture.md`, one newline, and `docs/adding-models.md`, encoded without special tokens. Its SHA-256 is `edeb4b52a2dd18bf875e0f3dead6e975479d27f9d8520b190ba91c08828a3ca9`. The tokenizer supplies no BOS anchor for this checkpoint.

| Arm, after 4,608 context tokens | Compared positions | Top-1 disagreements | Decided disagreements, gap ≥ 2 | Maximum selected-logit delta | Perplexity |
|---|---:|---:|---:|---:|---:|
| Sparse mlx-vlm vs mlxcel | 128 | 0 | 0 / 61 | 0 | 6.5903 |
| Forced dense mlx-vlm vs mlxcel | 128 | 0 | 0 / 60 | 0 | 6.6565 |
| Sparse vs forced dense negative control | 128 | 6 | 0 / 61 | 0.625 | 6.5903 vs 6.6565 |

The dense control raises `text_config.indexer_budget` to 8192 in a separate config overlay, keeping the same weight files. Its six changed greedy predictions demonstrate that the long-context test exercises QSA selection. The disagreements occur at low-margin positions; no decided negative-control disagreement is claimed.

Reproduce the sparse trace with:

```sh
python scripts/qwen4_exp_oracle_trace.py MODEL CORPUS 4608 128 > reference.tsv
MLXCEL_TRACE_START_TOKEN=4608 target/release/examples/logit_trace MODEL CORPUS 128 1 8 4608 > candidate.tsv
python scripts/compare_logit_traces.py reference.tsv candidate.tsv --decided 2
```

For the dense reference add `--indexer-budget 8192`; use the matching config overlay as the Rust model directory. The benchmark harness also records all 128 single-token decode predictions, separately from these 128-position prefill traces.

## Performance methodology

`examples/qwen4_exp_benchmark.rs` and `scripts/qwen4_exp_benchmark.py` run the same token IDs with batch size one and fresh cache state per repetition. One complete warmup precedes measured runs. Prefill is one forward call; decode performs 128 synchronous, teacher-forced singleton calls, including argmax evaluation and token readback. Unused vision allocations are excluded: mlxcel leaves the tower lazy, and the Python harness releases the wrapper while retaining its language model. Model loading, tokenization and HTTP overhead are excluded. GPU jobs run serially.

```sh
cargo build --release --features metal,accelerate --bin mlxcel --example qwen4_exp_benchmark --example logit_trace
python scripts/qwen4_exp_benchmark.py MODEL CORPUS 128 128 5
target/release/examples/qwen4_exp_benchmark MODEL CORPUS 128 128 5
python scripts/qwen4_exp_benchmark.py MODEL CORPUS 4608 128 5
target/release/examples/qwen4_exp_benchmark MODEL CORPUS 4608 128 5
```

The optimizations preserve bf16 expert reduction boundaries, use the native indexed QSA and MRoPE kernels, compile hyper-connection pointwise operations, fuse compatible GatedDeltaNet decode projections, and submit singleton layers asynchronously as the reference does. The generic fused MoE kernel is unsuitable here because it changes intermediate bf16 rounding.

## Measured results

Median of five measured runs; the full warmup row is retained in each raw JSONL file but excluded below. Peak memory is the maximum measured active allocation, in decimal GB. This is model execution throughput, not an HTTP or asynchronous generation benchmark.

| Prompt tokens | Runtime | Prefill tok/s | Decode tok/s | Decode range | Peak GB |
|---:|---|---:|---:|---:|---:|
| 128 | [reference](qwen4-exp-2026-10-23/reference-128.jsonl) | 418.00 | 40.84 | 40.78–40.88 | 81.29 |
| 128 | [mlxcel](qwen4-exp-2026-10-23/mlxcel-128.jsonl) | 434.02 | 41.39 | 40.60–41.41 | 81.27 |
| 4,608 | [reference](qwen4-exp-2026-10-23/reference-4608.jsonl) | 1006.52 | 32.00 | 31.02–32.21 | 88.08 |
| 4,608 | [mlxcel](qwen4-exp-2026-10-23/mlxcel-4608.jsonl) | 1056.91 | 37.77 | 36.89–37.79 | 87.92 |

All 128 decode predictions agree in every paired run, including warmup. Median prefill improves by 3.83% at 128 tokens and 5.01% at 4,608 tokens; median decode improves by 1.35% and 18.02%, respectively. The short-decode difference is small relative to the observed run-to-run spread and should be treated as near parity, not a broad speedup claim. These results qualify this checkpoint and M3 Ultra configuration; they do not establish performance on other models, quantizations or hardware.

The numerical trace files are [reference sparse](qwen4-exp-2026-10-23/reference-sparse.tsv), [mlxcel sparse](qwen4-exp-2026-10-23/mlxcel-sparse.tsv), [reference dense](qwen4-exp-2026-10-23/reference-dense.tsv), and [mlxcel dense](qwen4-exp-2026-10-23/mlxcel-dense.tsv).

## Binary fingerprints

The measurements use the full release profile (fat LTO), built from the implementation accompanying this report. SHA-256:

- `mlxcel`: `c6d5b4e9ca2d4748d8105b954c5750db99972d54b4a55243e7fc2e83b134ed4c`.
- `qwen4_exp_benchmark`: `e5d7c5f9bfdb5fa8c04850d974aec5cf913c903cc76b15c63084151405fda575`.
- `logit_trace`: `55bdefcd27c3ebdaf795502885f3b109548d2e29905d5ce672491d0759bed064`.

## CLI and server smoke tests

The release binary loaded the checkpoint in about 2.3 seconds and reported 75.19 GiB resident memory after projection fusion. `mlxcel generate -p Hello -n 64 --temp 0 --show-reasoning` produced `Hello! 👋 How can I help you today?` (35 generated tokens including reasoning). Separate CLI requests recognized the test image's colored vertical stripes, the video's changing test pattern and timer, and summarized the long architecture prompt coherently.

The image is the first frame of a 224×224, two-second ffmpeg `testsrc2` clip; video sampling uses 2 FPS. Image/video/long CLI requests use a 192-token cap with reasoning visible. These are execution and coherence checks, not a multimodal accuracy benchmark.

The loopback server ran with `--host 127.0.0.1 --port 18075 --alias qwen4-hardened --ctx-size 8192 --parallel 1`. Chat Completions used temperature zero, a 192-token cap, and `chat_template_kwargs: {"enable_thinking": false}`. All requests returned successfully:

| Request | Prompt / completion tokens | Observed output |
|---|---:|---|
| Text, “What is 2 + 2?” | 23 / 1 | `4` |
| Long document | 4,632 / 115 | Three coherent sentences describing Rust, the MLX C++ bridge, loading, the batch-native engine and backend overlays |
| PNG data URL | 71 / 192 | Describes colored vertical bars and foreground pixel shapes |
| MP4 data URL, four sampled frames | 118 / 192 | Identifies a color-bar test pattern, diagonal motion and the timer |
| SSE text | streamed | `Hello!`, followed by `[DONE]` |

Image and video answers reached the requested token cap; they are not presented as complete unconstrained answers. [Raw server responses](qwen4-exp-2026-10-23/server-smoke.jsonl) retain the finish reasons and usage. Model-owned state was reset between these independent requests; no cross-request cached prompt tokens were reported.

## Checks and scope

- `cargo test --workspace --profile test-fast --features metal,accelerate --no-fail-fast -- --test-threads=1`: 13,308 passed, 0 failed, 404 ignored.
- `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings`: passed.
- `cargo fmt --all --check`, license-header, kernel dtype-key and kernel port-dispatch checks: passed.
- Python 3.12 `unittest discover -s tests -p 'test_*.py' -v`: 139 tests, no failures, 28 platform skips on macOS. The script executable-mode CI failure was corrected and verified by this run.
- Added bf16 MoE and long-position rotary fixtures, float32 recurrent-state/chunking checks, anonymous-batch isolation, malformed PLE sidecar rejection, folded OptiQ norm tests, fused projection equivalence and sparse singleton attention coverage.

External PLE storage, FP8 checkpoints, MTP/DFlash, adapters, tensor/pipeline parallelism, padded or scheduled batched prefill, paged decoding, prompt-cache snapshots and KV-cache quantization remain outside this port's supported scope. Other model families retain their existing recurrent-kernel dtype behavior; Qwen4 opts into float32 gate/state preservation.
