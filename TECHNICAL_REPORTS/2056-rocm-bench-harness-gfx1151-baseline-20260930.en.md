# Technical Report: PR #2056 - ROCm Support in the Benchmark Harness and a gfx1151 Baseline

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: Bash (benchmark harness), Python (mlx-lm harness, CSV comparison, tests), Markdown, CSV

**Risk Level**: Low (benchmark scripts, docs, CSVs and Python tests only; no Rust, C++, kernel or build path changes. The Metal and CUDA code paths in `bench_decode.sh` were reordered, not changed in output, and were not exercised on those hosts)

## Executive Summary

Before this PR, `scripts/bench_decode.sh` on the AMD Strix Halo host would have written ROCm results as `benchmarks/metal_<first 20 characters of the CPU name>_<date>.csv`, budgeted its out-of-memory guard against the 31 GiB the host sees rather than the 96 GiB carve-out the GPU has, and recorded only the upstream MLX pin, which cannot tell upstream MLX from a build carrying the ROCm overlay. `scripts/compare_bench_csv.py` knew only the `m5max` and `m1ultra` hosts, so its superseded-baseline warning was off for every other host. Issue #1810 (Phase 3 of epic #1801) asked for correctly labeled, comparable ROCm results and a first published baseline.

The PR teaches both harnesses to recognize a ROCm host, writes ROCm runs as `benchmarks/rocm_strixhalo-gfx1151_<date>.csv`, budgets against device memory, adds two ROCm-only provenance columns (`mlx_rocm_overlay_commit`, `hip_version`), and makes the comparison script read runtime and host from any harness filename. It also publishes the first gfx1151 throughput baseline, five checkpoints at pp512/tg128, mlxcel against mlx-lm on the same host and day. From the committed CSVs, mlxcel's decode rate over mlx-lm's is 1.27x on Qwen3-0.6B-4bit (278.48 against 220.13 tok/s), 1.08x on Llama-3.1-8B-Instruct-4bit (35.41 against 32.81), 1.05x on Qwen3-30B-A3B-4bit (61.82 against 58.65) and 1.00x on gpt-oss-20b-MXFP4-Q4 (8.07 against 8.07). Mixtral-8x7B decode varies from about 8 to 11 tok/s run to run in both runtimes, so no ratio is claimed for it.

## 1. Problem Statement

### Mislabeling and a wrong memory budget

Issue #1810 re-verified each gap on `d8d34e2b`:

- `detect_backend` returned `cuda` only when `nvidia-smi` worked or `generate --help` mentioned cuda, otherwise `metal`. A ROCm sweep would have been filed under `metal_`.
- The hardware tag fell back to the first 20 characters of the `/proc/cpuinfo` model name when `nvidia-smi` found no GPU, which names the CPU, not the device that ran the kernels.
- The memory guard took 85% of `free -b`. On a UMA carve-out this is the wrong figure by a factor of about three: the host sees about 31 GiB, the GPU 96 GiB. A 26 GB Mixtral that the GPU holds with room to spare would have been skipped as `SKIP:oom_estimate`.
- `mlx_commit` holds the 8-character upstream pin. The ROCm build vendors NripeshN/mlx `rocm-support` retargeted onto that pin plus local fixes, and nothing in a row said so.
- `compare_bench_csv.py` hardcoded `m5max` and `m1ultra` and treated every runtime other than `pylm` as `metal`. For any other host (GB10, V100, gfx1151) `newer_readings_elsewhere` returned `{}`, so the check the script's own docstring calls the one that matters most was silently disabled.

### No way to name the device without loading a model

The issue's plan pointed at `gpu_backend_kind()` from #1805 (PR #1883) as the authoritative backend source, but the harness is a shell script and no CLI subcommand prints that value. `generate --help` is identical across backends. The only CLI output that names the backend, the `gfx` target and the device memory is a real `mlxcel generate` run: `[mlxcel] custom kernel backend: rocm` on stderr under `MLXCEL_DEBUG_KERNEL_BACKEND=1`, `HIP architecture gfx1151; compiled for [...]`, and `GPU: <name> (Amd), <N> GiB device memory.` on stdout. `scripts/ci/rocm_smoke.sh` already parsed these lines.

### No baseline

Neither `benchmarks/` nor `docs/benchmark_results/` had any ROCm throughput data. The issue carried spike numbers for orientation only; the ROCm performance work in #1814 needed a reproducible reference measured with the project's own harness.

## 2. Change Summary

- **`scripts/bench_decode.sh`** (283 lines changed).
  - *Runtime probe.* `probe_runtime` runs only on Linux when `nvidia-smi` fails. It picks the smallest checkpoint in `MODELS_DIR` (`smallest_checkpoint`, by safetensors size), falling back to the named model only when the store has none, and runs `MLXCEL_DEBUG_KERNEL_BACKEND=1 mlxcel generate -m <model> -p Hello -n 1` under `BENCH_PROBE_TIMEOUT` (default 300 s). Four `sed` parsers read the backend, `gfx` target, device name and device memory from the output.
  - *Fallbacks.* A probe that printed no backend line (the run failed before any kernel resolved) is not taken as evidence against ROCm: `rocminfo`, run under a 60 s timeout because it can block on a wedged KFD driver, supplies the first GPU agent's `gfx` name and marketing name. A probe that named another backend is trusted and `rocminfo` is not consulted. Device memory falls back to `/sys/class/drm/card*/device/mem_info_vram_total`. The ROCm release comes from `/opt/rocm/.info/version` or `core*/.info/version`, the HIP version from `hipconfig --version`. Each helper returns an empty field rather than failing, since they run inside command substitutions under `set -euo pipefail`.
  - *Labels.* `detect_backend` returns `rocm` after the NVIDIA check and before the `--help` check. `detect_hardware_full` builds `<device name>_<gfx>_ROCm<release>_<device memory>GB`, shaped like the CUDA string; `detect_hardware_short` maps `*_gfx1151_*` to `strixhalo-gfx1151` and any other `gfx` target to `amd-<gfx>`, never the 20-character truncation, which would have cut the target off.
  - *Budget.* `detect_memory_bytes` returns device memory when the backend is `rocm` and a device figure was found, host memory otherwise. The script prints `ROCm: <release> (HIP <v>), <gfx>` and `Memory budget: 85% of <N> GiB (<source>)`, naming whether the base came from the probe line, sysfs, or host memory.
  - *Ordering.* Backend, hardware and budget detection moved from script load into `resolve_platform`, called after argument parsing, because the probe needs `MODELS_DIR` and the model argument.
  - *Columns.* Every row's trailing commit fields now come from `COMMIT_FIELDS`. On ROCm it appends the 8-character overlay commit read from `src/lib/mlx-cpp/patches-rocm/UPSTREAM` and the HIP version, and the header gains `,mlx_rocm_overlay_commit,hip_version`. Metal and CUDA rows keep the previous schema. Device name and version strings have commas and quotes stripped before they reach unquoted CSV fields.
- **`scripts/bench_mlxlm.py`** (107 lines changed). `detect_rocm_gpu` defers to a working `nvidia-smi`, then parses `rocminfo`'s first GPU agent, the ROCm version file and sysfs VRAM. `rocm_hardware_names` builds the same short and full tags as the shell script. On ROCm the memory limit becomes 85% of device memory unless `PYLM_BENCH_MAX_GB` is set, and `baseline_version` appends `+mlx-<mx.__version__>`, since MLX there is a source build whose version string carries its commit.
- **`scripts/compare_bench_csv.py`** (25 lines changed). `runtime_and_host` takes the first two `_`-separated fields of a harness filename when the first is one of `metal`, `cuda`, `rocm`, `pylm`. `newer_readings_elsewhere` uses it in place of the hardcoded host list.
- **`tests/test_bench_rocm_detection.py`** (new, 351 lines). Extracts the shell functions and runs them in isolation: probe parsers, `rocminfo` agent selection, missing version tools, hardware tags (the new AMD tags and the existing Apple, GB10 and V100 ones), `probe_runtime` against a stub `mlxcel` (smallest-checkpoint choice, parsed device fields, a non-ROCm backend, the `rocminfo` fallback), backend and budget selection, the Python harness producing the same tag, and the compare scan on a ROCm host. The PR body states the tag and compare tests fail on the pre-change scripts.
- **`benchmarks/rocm_strixhalo-gfx1151_2026-09-30.csv`**, **`benchmarks/pylm_strixhalo-gfx1151_2026-09-30.csv`** (new, five rows each) and **`docs/benchmark_results/rocm-baseline-gfx1151-2026-09-30.md`** (new): the baseline and its write-up.
- **`docs/benchmarks.md`** documents the two new columns, the `baseline_version` suffix and a "ROCm hosts (issue #1810)" section with the commands. **`docs/installation.md`** links the baseline from the ROCm section.

Commit history: `9d412eef` adds detection, labels, budget and columns; `5fe7e13f` documents them; `fb1ee3e8` publishes the baseline; `a374a2d6` keeps the helpers from ending the script silently when a ROCm tool or file is missing; `c9c3b92c` probes with the smallest checkpoint, bounds `rocminfo` and sanitizes CSV fields; `d909b974` adds the `probe_runtime` tests.

## 3. Technical Decisions

### Probe with a one-token run instead of adding a CLI flag

The issue allowed either a probe run or a new flag that prints `gpu_backend_kind()` without loading a model, decided by whichever kept the probe under a few seconds. The probe needs no Rust change, reads exactly the lines `rocm_smoke.sh` already asserts, and reports what actually ran on the device rather than what the binary was compiled for. On the Strix Halo host it costs about 4 s with Qwen3-0.6B-4bit. The cost is that the probe needs a loadable checkpoint, which is why detection moved after argument parsing and why the fallbacks exist.

### Probe the smallest checkpoint, not the named one

The first version probed with the model named on the command line. Review found that the probe runs before `bench_one`'s `SKIP:oom_estimate` guard, so a single-model run on a checkpoint too large for the budget would load it anyway, and a large one would be loaded twice. The probe now uses the smallest checkpoint in the store and falls back to the named model only when the store is empty.

### Treat "no backend line" and "another backend" differently

A probe that crashed before printing anything says nothing about the host, so `rocminfo` decides. A probe that named `metal` or `cuda` is positive evidence and overrides an AMD agent that `rocminfo` might list. This keeps a broken checkpoint from mislabeling the sweep while never relabeling a non-ROCm binary as ROCm.

### Device memory as the budget base on ROCm only

On the UMA APU the carve-out is what the GPU can allocate, and `mlxcel generate`'s own preflight already reads the ROCm allocator's `memory_limit()` since PR #1883. Host memory remains the base everywhere else, and on ROCm when no device figure was found, in which case the script prints that the budget fell back to host memory rather than silently using it.

### New columns only on ROCm rows

Appending `mlx_rocm_overlay_commit` and `hip_version` to every backend would have changed the Metal and CUDA schema for columns that are always empty there. Since `compare_bench_csv.py` reads rows with `csv.DictReader` by header name, a trailing column on one runtime's files does not disturb pairing. `mlx_commit` keeps its meaning as the upstream pin, as the issue required.

### Parse runtime and host from the filename

Both harnesses already name files `<runtime>_<host>_...`. Reading the first two fields generalizes the superseded-baseline scan to every host without a list to maintain. One side effect the PR body records: the scan now also runs for `cuda_*` files, which the old code skipped.

### Publish the baseline under a contention guard

The host runs other GPU jobs. The page describes a wrapper that waited for 90 s with no KFD process and no compiler, sampled both once per second during the sweep, and rejected any sweep whose window saw either. Five sweeps were rejected and rerun. The published mlxcel sweep ran 02:31:32 to 02:40:28 KST and the mlx-lm sweep 03:21:29 to 03:29:42 KST.

## 4. Baseline Results

From the committed CSVs (pp512/tg128, batch 1, greedy, one measured pass after a 20-token warmup, one process per model):

| Model | mlxcel prefill tok/s | mlx-lm prefill tok/s | mlxcel decode tok/s | mlx-lm decode tok/s | Decode ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B-4bit | 4417.27 | 4316.74 | 278.48 | 220.13 | 1.27x |
| Meta-Llama-3.1-8B-Instruct-4bit | 1065.78 | 966.84 | 35.41 | 32.81 | 1.08x |
| Qwen3-30B-A3B-4bit | 275.65 | 280.40 | 61.82 | 58.65 | 1.05x |
| gpt-oss-20b-MXFP4-Q4 | 7.69 | 7.77 | 8.07 | 8.07 | 1.00x |
| Mixtral-8x7B-Instruct-v0.1-4bit | 25.85 | 26.43 | 7.65 | 9.98 | not claimed |

Hardware string in both files: `AMD_Radeon_8060S_Graphics_gfx1151_ROCm10.0.0_96GB`. mlxcel rows: `mlxcel_commit` `2fcdbe5f`, `mlx_commit` `81ba1c6a`, `mlx_rocm_overlay_commit` `75915908`, `hip_version` `7.15.26333`. mlx-lm rows: `baseline_version` `mlx-lm-0.31.3+mlx-0.32.3.dev20260912+a7d4c85a`. The page records that `2fcdbe5f` differs from the `784e35b7` build commit only in harness scripts, their test and docs, so the binary is the same.

Readings from the page:

- On the three dense or affine-MoE models mlxcel leads decode by 5 to 8% on the 8B and 30B-A3B and by 27% on the 0.6B, where per-token host overhead is the largest share of a step. Prefill is within 11% either way.
- The two MoE checkpoints whose experts go through the generic `gather_qmm` path are slow in both runtimes by the same amount: Mixtral prefills at about 26 tok/s and gpt-oss at under 8 tok/s (about 66 s for 512 tokens). With the two runtimes within 3% of each other, this is the overlay's shared kernel, not mlxcel's model code. The fused MoE kernels have no ROCm port yet (#1814).
- `compare_bench_csv.py --reference` pairs all five rows with nothing dropped.
- Every model fit under 85% of 96 GiB; nothing was skipped and no run failed.

## 5. Validation

PR author's runs (gfx1151, ROCm 10.0.0, HIP 7.15.26333, release binaries built with `--features rocm` from `784e35b7`): the unit tests (35 pass), the `make verify-*` gates, `insert_apache_header.py --check` and the `dead_doc_pointers` test pass; an end-to-end `bench_decode.sh all` detected `rocm`, `strixhalo-gfx1151` and the 96 GiB budget and measured all five models; `bench_mlxlm.py all` wrote the matching `pylm_strixhalo-gfx1151` file. After the review fixes, a single-model run still reported the same labels and budget, and a run whose probe checkpoint could not load took device memory from sysfs; those runs shared the GPU and their numbers are not published.

Orchestrator verification (gfx1151 host, branch rebased onto origin/main `81f13ecd`). The change is scripts, docs, CSVs and Python tests only, so the affected gates were run rather than the full test suite:

- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`: pass.
- `make verify-binary-assets`: fails on `tests/fixtures/nemotron_parse_page.png` (41.1 KB, over the 32 KB ceiling). It fails identically on main, where the fixture was added by #2034, and is unrelated to this PR.
- `bash -n scripts/bench_decode.sh` and `py_compile` of `scripts/bench_mlxlm.py` and `scripts/compare_bench_csv.py`: pass.
- `pytest tests/test_bench_rocm_detection.py tests/test_bench_decode_oom_classifier.py`: 35 passed.
- `cargo test --release --features rocm --test dead_doc_pointers`: 2 passed.

## 6. Learning Points

- **When the authoritative source is inside the binary, a short real run is a valid probe.** No flag exposes `gpu_backend_kind()`, but the lines a one-token `generate` prints report the resolved backend, target and device memory. Reusing the lines a CI smoke test already asserts keeps the harness and CI reading the same contract.
- **Detection that runs under `set -euo pipefail` inside command substitutions must never fail.** A missing ROCm version file or a failing `hipconfig` ended the first version of the script with no message. Every helper now returns an empty field, and a test runs them with the tools absent.
- **Two harnesses that must pair need the same tag from different sources.** The shell harness reads the device name from the probe's `GPU:` line, the Python one from `rocminfo`'s marketing name; memory comes from the probe or sysfs. They agree on this host (both produce `AMD_Radeon_8060S_Graphics_gfx1151_ROCm10.0.0_96GB`), and a test pins the Python tag against the shell one for the same inputs. A host where the probe name and the marketing name differ would split the pair.
- **A benchmark on a shared GPU needs a recorded guard, not an assurance.** The rejected sweeps, including one where another unit's GPU tests overlapped, show the guard was needed; recording its windows lets a reader judge the numbers.
- **A hardcoded host list is a silent off switch.** The superseded-baseline check had been disabled on every non-Apple host since it was written. Deriving the host from the filename convention removes the list.

## 7. Caveats and What Is Not Verified

- **The mlx-lm side is not byte-identical to mlxcel's MLX.** MLX has no ROCm wheel, so the Python baseline runs on a source build of the spike tree (MLX `a7d4c85a`). It is the same fork commit (`75915908`) on the same MLX pin (`81ba1c6a`), with trial commits equivalent to `LOCAL_FIXES.md` items 1 to 6 and 8 to 11 and part of item 7, but it lacks items 12 to 19 (the f16 `gather_qmm` fast path, f32 activations in tiled qmv, the scatter argument width, `SearchSorted`, `Hadamard`, narrow gather indices, FFT, and the `get_launch_args` removal). The page argues these do not affect bf16 text decode but did not verify it by tracing launched kernels. A baseline built from mlxcel's own overlay source would settle it.
- **Mixtral decode does not repeat.** Clean readings were 7.65, 8.22 and 11.16 tok/s for mlxcel and 9.98 and 8.11 for mlx-lm, while prefill stayed at 25.8 to 26.5. The sweep's 0.77x is noise, and no ratio is claimed. The cause (clock or power state, weight placement, or something else) is left to #1814.
- **MoE prefill is slow in both runtimes.** Mixtral and gpt-oss run their experts through the shared generic `gather_qmm`; this is the starting point for #1814, not an mlxcel regression.
- **One measured pass per model.** Apart from the Mixtral repeats, each number is a single run; run-to-run variance for the other four models was not measured.
- **Metal and CUDA hosts.** Not available here. The changes on those paths are the move of detection after argument parsing and the `${COMMIT_FIELDS}` substitution; unit tests pin the Apple, GB10 and V100 tags and the host-memory path, but no Metal or CUDA sweep was run. `compare_bench_csv.py` now scans `cuda_*` files for superseded readings, which it previously skipped, and may surface warnings on CUDA comparisons that were silent before.
- **Other `gfx` targets.** Only gfx1151 was run. The `amd-<gfx>` tag and the `rocminfo` fallback on discrete AMD GPUs are covered by unit tests only.

## 8. Remaining Work

- #1814 (ROCm performance) measures against this baseline. The page leaves the MoE prefill gap (no fused MoE port) and the cause of the Mixtral decode variance to that epic.
- Rebuilding the mlx-lm side from mlxcel's own overlay source would remove the build-difference caveat.

Refs: #1810 (closed by this PR), #1801, #1802, #1805, #1808, #1809, #1814, #2034, PR #1818, PR #1883.
