# CUDA fused add+RMSNorm made byte-identical to the graph, GB10, 2026-11-28 (issue #3510)

After #3510 the CUDA fused residual-add + RMSNorm kernel (`MLXCEL_FUSED_ADD_RMSNORM=1`), the regenerated cuda-oxide artifact and the default `add` + `fast_rms_norm` pair produce the same bytes. This directory holds the rows behind that statement and the kernel time after the change. Host `spark-101` (NVIDIA GB10, sm_121, driver 580.178.04, kernel 7.0.0-1019-nvidia), `MLXCEL_CUDA_ARCHITECTURES=121`, MLX pin `fc344c986c4169d99f76b5d95ae6b4a58a359a63`, one release build made with `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1`, checkpoint `llama-3.2-1b-instruct-bf16`. Every GPU command ran under `gpu-lock`. The binaries were built from the sources of commit `2e9c078d7`. The host is also a self-hosted CI runner, and CI jobs ran during the correctness runs and during part of the kernel-time stage (see Kernel time).

Before the change the same commands gave two digests on two of the six cells and a one or two ulp `normed` difference on 2 of 2048 prefill rows: [`cuda-oxide-fused-norm-gb10-2026-11-27/`](../cuda-oxide-fused-norm-gb10-2026-11-27/) and [`issue-3501-fused-norm-divergence-gb10-2026-11-28/`](../issue-3501-fused-norm-divergence-gb10-2026-11-28/README.md). Those directories are left as measured.

| File | Measurement |
| --- | --- |
| `end-to-end-cells.csv` | One round of `scripts/oxide_fused_norm_rounds.py run`: 6 cells (CLI B=1, server B=1, server B=4, each at 128 and 2048 prompt tokens), arms `graph`, `fused`, `oxide`, `fused-null`; route counters and token ids per pass |
| `dump-compare-prefill.csv` | `fused_norm_divergence_probe --compare` of the graph arm's and the fused arm's prefill dumps, one row per tensor |
| `op-diff-prefill.csv` | `fused_norm_divergence_probe --op-diff` on the fused arm's and on the graph arm's prefill dump: both ops on the same inputs, one row per layer |
| `m4-nsys-kernels.csv`, `m4-nsys-cuda-api.csv` | Kernel durations and CUDA API calls from `nsys --trace=cuda`, standalone and through MLX with `MLX_USE_CUDA_GRAPHS=0`, three captures each |

`dump-compare-prefill.csv` and `op-diff-prefill.csv` are the probe's `[compare]` and `[op-diff]` lines with the fields split into columns; nothing is computed from them. Profiler captures, dumps and PTX caches were written to a scratch directory outside the checkout and are not kept.

## The geometry

The kernel takes the launch geometry of MLX's CUDA `rms_norm` (`mlx/backend/cuda/rms_norm.cu` at the pin), so each thread adds the same squares in the same order as the graph path does:

- Compute dtype `C = promote_types(activation, weight)`, the dtype `fast::rms_norm` casts its inputs to. With this tree's promotion table, equal dtypes stay, bf16 with f32 is bf16 (either way round), and f16 with f32 or with bf16 is f32.
- A row is read as vectors of `N = 16 / sizeof(C)` elements: 8 for f16 and bf16, 4 for f32. Thread `t` of a block of `B` threads adds the squares of vectors `t`, `t + B`, `t + 2B`, ... in ascending order into one fp32 register (one FMA per element), then the block is reduced by a warp butterfly and a second butterfly over the warp totals.
- `B` is `fused_norm_cuda_block(dim, N)`, which is `dispatch_num_chunks` and the threshold in `RMSNorm::eval_gpu`. With `nvec = ceil(dim / N)`:

| Row width | Block | MLX kernel |
| --- | --- | --- |
| `dim > N * 1024` (above 8192 at N = 8, above 4096 at N = 4) | 1024 | `rms_norm<T, 1024, N>`, looped |
| `dim <= N * 64` (up to 512 at N = 8, up to 256 at N = 4) | 64 | `rms_norm_small`, one chunk |
| `nvec % 128 == 0` | 128 | `rms_norm_small`, `nvec / 128` chunks |
| `nvec % 64 == 0` and `nvec <= 512` | 64 | `rms_norm_small`, `nvec / 64` chunks |
| anything else | 128 | `rms_norm_small`, `ceil(nvec / 128)` chunks, ragged tail |

At the measured shape (bf16, width 2048) that is 128 threads and two 8-wide vectors per thread, the block `m4-nsys-kernels.csv` shows for both `rms_norm_small` and the fused kernel. The first version of this table in the issue had bf16 under an f32 weight computing in f32; MLX's promotion in this tree gives bf16 there, and the kernel and the artifact follow the promotion.

## Byte identity

`fused_norm_parity_tests::cuda_byte_identity` (three tests) compares both outputs of the fused launcher with `graph_add_rms_norm` byte for byte, the graph's `normed` cast to the activation dtype for a mixed pair:

- 8 dtype pairs (activation / weight): f32/f32, f16/f16, bf16/bf16, f16/f32, bf16/f32, f32/f16, f32/bf16, f16/bf16.
- 21 widths at 1 and 33 rows: 1, 3, 100, 128, 512, 513, 768, 1000, 1280, 1536, 1792, 2048, 2049, 2560, 3072, 3584, 4096, 4097, 8192, 8193, 12288. Every row of the table above is reached at both vector widths.
- The #3501 inputs: `random_case(7000 + s, 4096, 2048, bf16)` for `s` in 0..6 (24,576 rows).
- The Gemma `weight_bias = 1` fold against `GemmaRMSNorm` and `eps = 1e-5`, each at six widths per pair; 1024 rows of widths 2048 and 4096 with per-row scales spread over `exp(2 * N(0, 1))`; signed zeros (underflowing elements, zero and negative-zero weights); and rows carrying a NaN, a +Inf and a -Inf, compared by class with the finite row between them bit-exact.

On the kernel before #3510 the first test reports 270 differing cases, among them 5 rows of the #3501 seeds, the count #3501 measured. On the new kernel all three pass, with `MLXCEL_FUSED_ADD_RMSNORM` unset, `=0` and `=1` (13 tests in `fused_norm_parity_tests` each time).

The cuda-oxide artifact was regenerated from the same contract: PTX SHA-256 `ed470a19c4848893a680223db9c6861363c4cb180a5c229a63307afc305357f4`, cubin `390fcc2833e532ef5bffb7d8699c2f43a0fd7dc53775079b9b2c6be938a558cd`. The standalone harness passes its 130 cases and the launch check, `scripts/compare.sh` reports `bitwise_identical_to_fused_cuda=130` with a normalized deviation of zero against the graph on every case, and the 33 `cuda_oxide` tests of `mlxcel-core` pass; the adapter matrix prints `cases=130 failed=0 bitwise_identical_to_fused_cuda=130 oxide_routes=130 launches=130 native_routes=0` and now requires equality with the graph on every row without a non-finite input.

## End to end

`end-to-end-cells.csv`: one `tokens_sha256_16` per cell across the arms, every run on the route its arm names.

| Cell | Prompt | `graph` | `fused` | `oxide` | Before #3510 (`graph` / `fused`, `oxide`) |
| --- | --- | --- | --- | --- | --- |
| CLI B=1 | 128 | `153e97b6c47aae0d` | `153e97b6c47aae0d` | `153e97b6c47aae0d` | `153e97b6c47aae0d` / `ca3dde687d118faf` |
| CLI B=1 | 2048 | `fc892689833f440e` | `fc892689833f440e` | `fc892689833f440e` | same on all arms |
| Server B=1 | 128 | `153e97b6c47aae0d` | `153e97b6c47aae0d` | `153e97b6c47aae0d` | `153e97b6c47aae0d` / `ca3dde687d118faf` |
| Server B=1 | 2048 | `fc892689833f440e` | `fc892689833f440e` | `fc892689833f440e` | same on all arms |
| Server B=4 | 128 | `f29f281eb40e23e3` | `f29f281eb40e23e3` | `f29f281eb40e23e3` | same on all arms |
| Server B=4 | 2048 | `8d216cd6fe8681b6` | `8d216cd6fe8681b6` | `8d216cd6fe8681b6` | same on all arms |

The fused arms now generate 128 tokens at B=1 / 128, where they stopped after 98. Route counters per pass: `graph` 0 native and 0 oxide joins; `fused` and `fused-null` 2048 native joins at B=1 (2080 and 2096 at B=4) and 0 oxide; `oxide` the same counts on the cuda-oxide route with as many launches and 0 native.

The probe, teacher-forced on the graph arm's tokens with `--dump-layers 0`: `--compare graph/step0 fused/step0` reports the dumps byte-identical on all 82 tensors (`dump-compare-prefill.csv`), and `--op-diff` reports 0 differing elements of `new_residual` and `normed` on all 16 layers, on the fused arm's dump and on the graph arm's (`op-diff-prefill.csv`).

## Kernel time

The #2834 M4 method, unchanged: `cuda-oxide-fused-norm-gb10-2026-11-27/run.sh m4`, that is `nsys --trace=cuda`, the standalone driver-API program, and `oxide_fused_norm_bench --mode iso` through MLX with `MLX_USE_CUDA_GRAPHS=0`, three captures each, 821 launches per capture and shape standalone and 1100 through MLX. bf16 activations and weight, width 2048. Median of the three capture medians in µs; "before" is `cuda-oxide-fused-norm-gb10-2026-11-27/m4-nsys-kernels.csv`, first set, same host and method. Graphs off is not the production mode, so these are kernel durations, not throughput.

| Rows | Model step | Fused kernel through MLX, before (block 512) | after (block 128) | Change | cuda-oxide through MLX, before | after | Change | Graph path after: `binary_vv` + `rms_norm_small` (block 128) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | decode, B=1 | 2.21 | 3.39 | +53 % | 2.14 | 4.06 | +90 % | 1.02 + 1.44 |
| 4 | decode, B=4 | 2.59 | 3.90 | +51 % | 2.21 | 4.06 | +84 % | 1.47 + 1.44 |
| 128 | prefill 128, B=1 | 6.53 | 12.74 | +95 % | 6.69 | 12.80 | +91 % | 1.66 + 1.92 |
| 512 | prefill 128, B=4 | 24.13 | 43.94 | +82 % | 24.26 | 43.71 | +80 % | 4.48 + 3.94 |
| 2048 | prefill 2048, B=1 | 139.68 | 185.50 | +33 % | 139.74 | 182.83 | +31 % | 114.62 + 31.58 |

Standalone, the cuda-oxide kernel reads 4.03, 4.06, 11.58, 41.89 and 179.90 µs at the five shapes, against 2.14, 2.18, 6.40, 24.61 and 120.77 µs before.

The fused kernel is slower after the change: 1.3 to 2 times the earlier kernel time at every shape. The cause is the geometry the identity needs. At width 2048 the kernel used to spread a row over 512 threads of four elements and now uses MLX's 128 threads of sixteen, and it still reads elements one at a time and reads `new_residual` back for the second pass, where `rms_norm_small` loads 16-byte vectors and keeps the row in registers. `rms_norm_small` itself takes 1.4 to 32 µs with the same 128-thread block, so the block width is not what costs the time; the scalar loads and the second read at that width are the likely cause, and that was not isolated. The graph path's two kernels now take less time than the fused kernel at every shape (2.5 against 3.4 µs at one row, 146 against 186 µs at 2048 rows).

Two controls bound the noise. The graph path's kernels, which this change does not touch, read the same as on 2026-11-27 (`binary_vv` 1.02, 1.47, 1.66, 4.48, 114.62 against 1.02, 1.47, 1.66, 4.48, 114.05; `rms_norm_small` 1.44, 1.44, 1.92, 3.94, 31.58 against 1.44, 1.44, 1.95, 4.03, 30.27), and the three captures of each fused row agree within 5 %. A CI job was running when the stage ended and other GPU jobs held `gpu-lock` between its captures, so the stage was not repeated on an idle host; the unchanged controls are the reason the changes are attributed to the kernel.

End to end the cost does not show at this model size: in the one round of `end-to-end-cells.csv` the fused arm decodes within 1.5 % of the graph arm in every cell (for example 91.6 against 92.0 tok/s at CLI B=1 / 128). One round under a CI job is not a throughput measurement and no claim is made from it. `MLXCEL_FUSED_ADD_RMSNORM` stays off by default on CUDA; with this kernel time there is no shape here at which opting in is faster than the graph path.

The spike's own harness (`spike/cuda-oxide-fused-norm/results/bench-sm_121.json`, CUDA events around 200 back-to-back launches) shows the same direction on wide single rows, for example bf16 1 x 8192 at 17.3 µs and 64 x 8192 at 28.6 µs. The earlier file read 6.2 and 18.5 µs, but it was recorded on another GB10 (`spark-d9e2`), so that pair is not a same-host comparison.

## Not covered

- One host and one architecture (GB10, sm_121). The identity rests on NVRTC and MLX's nvcc build contracting the same FMA and on the same `rsqrt`; another CUDA architecture or toolkit was not run. `cuda_byte_identity` fails there if it does not hold.
- Metal: the parity tests were not run on Metal. The Metal body, its threadgroup rule and its template arguments are unchanged in the diff.
- Compute Sanitizer was not rerun on the regenerated artifact.
- A faster fused kernel with the same summation order (vector loads, the row kept in registers, as `rms_norm_small` does) was not attempted.

## Reproduce

```bash
export MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 MLXCEL_CUDA_ARCHITECTURES=121
cargo test --release -p mlxcel-core --lib --features cuda --no-run
cargo build --release -p mlxcel-core --features cuda --example oxide_fused_norm_bench
cargo build --release --features cuda --example oxide_fused_norm_checkpoint_bench --example fused_norm_divergence_probe
T=target/release/deps/mlxcel_core-<hash>; P=target/release/examples/fused_norm_divergence_probe; M=models/llama-3.2-1b-instruct-bf16; O=/tmp/issue-3510
gpu-lock run --tag issue-3510 -- $T fused_norm_parity_tests --test-threads=1
gpu-lock run --tag issue-3510 -- $T cuda_oxide --test-threads=1 --nocapture
python3 scripts/oxide_fused_norm_rounds.py run --model $M --rounds 1 --tag issue-3510-e2e --out $O/end-to-end.csv
# graph-tokens.txt: first_row_tokens of the graph arm, CLI B=1 / 128
env -u MLXCEL_FUSED_ADD_RMSNORM gpu-lock run --tag issue-3510 -- $P --model $M --tokens graph-tokens.txt --dump-layers 0 --expect-route graph --out $O/graph
MLXCEL_FUSED_ADD_RMSNORM=1 gpu-lock run --tag issue-3510 -- $P --model $M --tokens graph-tokens.txt --dump-layers 0 --expect-route fused --out $O/fused
$P --compare $O/graph/step0 $O/fused/step0
gpu-lock run --tag issue-3510 -- $P --model $M --op-diff $O/fused/step0 --out $O/opdiff
WORK=$O/work SET=-issue3510 docs/benchmark_results/cuda-oxide-fused-norm-gb10-2026-11-27/run.sh m4
```

The last command writes `m4-nsys-*-issue3510.csv` and a `host-state.csv` row into the 2026-11-27 directory; the two CSVs were moved here and the row was not kept.
