# ROCm correctness matrix, remaining rows: Radeon 8060S (gfx1151), 2026-09-30

Second run for issue #1809, extending [the 2026-09-12 matrix](rocm-correctness-gfx1151-2026-09-12.md). That run covered four checkpoints against a Metal reference. This one adds the five rows it left open: a second dense model, a sliding-window model, two SSM hybrids and a VLM.

What this run establishes, and what it does not:

- **Established on ROCm:** all five checkpoints load and generate, and all sixteen teacher-forced traces complete with no NaN. Getting there found and fixed one ROCm backend defect, a strided-scan launch that wrote past the end of its array and faulted the GPU on both SSM hybrids.
- **Not established:** agreement with Metal. No Metal host was reachable from the ROCm host, and no Metal trace for these checkpoints exists in the repository, so every row's Metal side is pending. The ROCm traces are committed so a Metal host can complete each pair; the commands are below. No number in this document is a cross-backend comparison.

## Environment

| Field | Value |
|---|---|
| Hardware | AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5, 40 CUs), 96 GiB VRAM carve-out, 31 GiB host RAM |
| OS | Debian GNU/Linux 13 (trixie), kernel 6.18.12+deb13-amd64 |
| Backend | ROCm 10.0.0 packages (HIP 7.15.26333, AMD clang 23), `--features rocm`, release profile |
| mlxcel | `c5fe9a16`: `origin/main` `4595b06f` plus this PR's code commits (see below) |
| MLX pin | `81ba1c6a` |
| ROCm overlay | fork `NripeshN/mlx` branch `rocm-support` at `75915908` (`src/lib/mlx-cpp/patches-rocm/UPSTREAM`), plus the local fixes in `LOCAL_FIXES.md`, now 22 items |
| Traces | `benchmarks/logit_traces/rocm_gfx1151_c5fe9a16/` |
| Metal reference | pending |

The first run's document implied the fork commit through `UPSTREAM` but did not state it; it is `75915908` for both runs.

## Method

The same as the first run: `examples/logit_trace` over `tests/fixtures/wikitext2_excerpt.txt` at `w1` = `1 128 8 0`, `w8` = `8 80 8 512` and `w256` = `256 2 8 0` (`CHUNK_TOKENS MAX_CHUNKS TOPK PREFILL`), with the comparison to be made by `scripts/compare_logit_traces.py --decided 2.0` once the Metal traces exist. See the first run for why decided-position disagreement, not byte identity, is the criterion.

One shape is added. `gemma-3-4b-it` has a 1024-token sliding window, and the standard widths never give it more than 520 tokens of context, so they never exercise the window. `w8ctx1536` = `8 40 8 1536` with `MLXCEL_TRACE_START_TOKEN=1536` scores 320 positions with 1536 tokens of context each.

The VLM row traces the language model of a VLM loaded on the default path; `logit_trace` feeds text only, so the vision tower is covered by the generation check below and not by a trace.

## Rows

ROCm side only. Perplexity and top-1 accuracy are against the corpus token, not against Metal, and are listed so a Metal run has something to sanity-check against; they are not the gate. "Decided" counts the positions whose own top-two gap is at least 2.0, which is how many positions a comparison at `--decided 2.0` would judge, and it will be recounted on the Metal side, which is the reference.

| Model | Width | Positions | NaN | Perplexity | Top-1 = corpus token | ROCm positions with gap >= 2.0 | Metal |
|---|---|---|---|---|---|---|---|
| qwen2.5-7b-instruct | w1 | 128 | 0 | 6292 | 2.3% | 19 | pending |
| qwen2.5-7b-instruct | w8 | 640 | 0 | 11.40 | 53.9% | 273 | pending |
| qwen2.5-7b-instruct | w256 | 512 | 0 | 13.50 | 52.1% | 203 | pending |
| gemma-3-4b-it | w1 | 128 | 0 | 775911 | 3.9% | 1 | pending |
| gemma-3-4b-it | w8 | 640 | 0 | 25.34 | 50.8% | 323 | pending |
| gemma-3-4b-it | w256 | 512 | 0 | 53.73 | 39.8% | 200 | pending |
| gemma-3-4b-it | w8ctx1536 | 320 | 0 | 9.34 | 60.3% | 185 | pending |
| granite-4.0-h-tiny | w1 | 128 | 0 | 133111 | 2.3% | 116 | pending |
| granite-4.0-h-tiny | w8 | 640 | 0 | 14.33 | 53.1% | 260 | pending |
| granite-4.0-h-tiny | w256 | 512 | 0 | 19.07 | 50.0% | 207 | pending |
| nemotron-3-nano-30b-a3b | w1 | 128 | 0 | 14868 | 1.6% | 0 | pending |
| nemotron-3-nano-30b-a3b | w8 | 640 | 0 | 10.98 | 54.8% | 286 | pending |
| nemotron-3-nano-30b-a3b | w256 | 512 | 0 | 12.03 | 52.5% | 215 | pending |
| qwen2.5-vl-3b-instruct | w1 | 128 | 0 | 66669 | 0.8% | 47 | pending |
| qwen2.5-vl-3b-instruct | w8 | 640 | 0 | 21.02 | 47.7% | 197 | pending |
| qwen2.5-vl-3b-instruct | w256 | 512 | 0 | 23.67 | 45.7% | 154 | pending |

The `w1` perplexities are huge because each `w1` chunk is a single token scored with no context and no BOS (#1785); the first run's `w1` traces look the same (`llama-3.1-8b-instruct` `w1` on ROCm: 719878). It is a property of the shape, the same on both backends, and says nothing about the backend.

Which paths ROCm took, from the code at `c5fe9a16`: granite's MoE goes through `SwitchGLU::forward` (`gather_qmm`) on every backend; Nemotron-H's MoE takes `forward_nonfused` on ROCm because `custom_kernels_available()` is false there; the Mamba2 layers of both hybrids run the SSD graph path at every width because `ssm_kernel_available()` is false on ROCm, where Metal runs the fused SSM update for single-token steps. So the hybrids' `w1` pairs, and every Nemotron-H pair, will compare a kernel on Metal against a graph on ROCm, not only two backends. The trace directory's README lists these pairings.

### Completing a row on a Metal host

`benchmarks/logit_traces/rocm_gfx1151_c5fe9a16/README.md` has the full script. In short, at the commit this document was merged in, with the checkpoint revisions in that directory's `METADATA.txt`:

```bash
cargo build --release --features metal,accelerate --example logit_trace
./target/release/examples/logit_trace models/mlx/<checkpoint> tests/fixtures/wikitext2_excerpt.txt 8 80 8 512 > metal_<host>_<commit>_<tag>_default_w8.tsv
python3 scripts/compare_logit_traces.py metal_<host>_<commit>_<tag>_default_w8.tsv \
    benchmarks/logit_traces/rocm_gfx1151_c5fe9a16/rocm_gfx1151_c5fe9a16_<tag>_default_w8.tsv --decided 2.0
```

and likewise for `w1`, `w256`, and `gemma-3-4b-it` at `w8ctx1536` (`MLXCEL_TRACE_START_TOKEN=1536`, `8 40 8 1536`). If model code changed between `c5fe9a16` and that commit, retrace the ROCm side there too.

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

The 34 paged-attention tests no longer fail. Each paged-attention kernel now exports a support predicate that reads its own `KernelPorts` table through `has_kernel_port` (`paged_attention_decode_available`, `paged_attention_v2_partial_available`, `paged_attention_merge_states_available`, bridged as `paged_attention_kernels_available` and `paged_attention_merge_available`). The tests return early through one helper, `src/lib/mlxcel-core/src/test_support/kernel_ports.rs`, only when the backend is ROCm and that predicate is false, and each skip prints `skipping <test>: ROCm has no paged-attention kernel port yet (lablup/mlxcel#1814)` to stderr so the gate log lists them. When #1814 fills a `.rocm` entry the predicate turns true and the tests run again with no edit. On Metal and CUDA the predicate is true, so nothing is skipped there. The three production gates in front of these kernels (batched paged decode, MLA split-KV, `paged_decode_backend`) now ask the same predicates instead of the backend-wide `custom_kernels_available()`.

The remaining three failures belong to other work and are left as they are. The counts of the gate run for this change are in the PR that added this document (#1809's PR), measured on the rebased branch immediately before merge.

## Reproducing

```bash
cargo build --release --features rocm --example logit_trace --bin mlxcel
./target/release/examples/logit_trace models/mlx/granite-4.0-h-tiny-4bit tests/fixtures/wikitext2_excerpt.txt 8 80 8 512 > rocm_w8.tsv
MLXCEL_TRACE_START_TOKEN=1536 ./target/release/examples/logit_trace models/mlx/gemma-3-4b-it-4bit tests/fixtures/wikitext2_excerpt.txt 8 40 8 1536 > rocm_w8ctx1536.tsv
cargo test --features rocm --test rocm_strided_scan -- --test-threads=1
MLXCEL_FUSED_MOE_RELU2=1 ./target/release/mlxcel generate -m models/mlx/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit -p "The capital of France is" -n 24 -t 0 --no-chat-template
```

## Open

- The Metal side of all sixteen traces. Until it exists, the second acceptance criterion of #1809 (decided-position mismatch reported per model and width against Metal) is met for the first four models only.
- No noise floor was measured; the first run's reasoning for not needing one still applies once these pairs exist.
