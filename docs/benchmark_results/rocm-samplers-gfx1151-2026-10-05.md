# HIP ports of the fused samplers on gfx1151 (2026-10-05)

lablup/mlxcel#2064, part of #1814. Before this change both fused samplers took the MLX graph on ROCm: `gumbel_ports()` and `rejection_ports()` had no `.rocm` entry, and `rejection_sample_supported()` answered with `custom_kernels_available()`, which is Metal-or-CUDA by definition, so a filled slot would have stayed unreachable. Sampled decode therefore ran `random::categorical` on the no-filter path and the `argpartition` / `argsort` / `cumsum` chain on the filtered path. The decode profile ([rocm-decode-profile-gfx1151-2026-09-30.md](rocm-decode-profile-gfx1151-2026-09-30.md)) put the whole sampler tail at 0% of greedy decode GPU time and 0.4 to 3.8% of sampled decode GPU time.

The ports (`sampling_gumbel_hip.h`, `sampling_rejection_hip.h` in `src/lib/mlx-cpp/turbo/`) are the CUDA bodies with the same inputs, outputs, grid, template arguments and Philox-4x32-10 counter and key layout, so a seed reproduces a ROCm stream as it does on CUDA and Metal. Neither kernel has a lane-level operation (every reduction and the rejection kernel's scan go through shared memory with a barrier per step), so neither carries a wave32 guard. `rejection_sample_supported()` now reads `has_kernel_port(rejection_ports())`.

The Gumbel-max port exposed a fault in the vendored `fast::hip_kernel`: it declared an input's `<name>_shape` as a pointer while the launch passes the shape by value, so the first kernel to read one (`logits_shape[1]`) faulted the queue. That is fixed in the overlay (LOCAL_FIXES item 30).

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), 96 GiB VRAM carve-out, Debian 13, ROCm 10.0.0. Before: `main` at `57d8ed29`. After: `85e39880` (the PR's second commit). Later commits add the benchmark script's `--temperature` / `--top-p` options (the runs used them, with each binary copied into place), docs, tests, a debug log string, two test-only bridge predicates, and the overlay's 0-d argument gating, none of which a sampler launch with 1-d and 2-d inputs reaches. MLX pin `81ba1c6a`, ROCm overlay `75915908`, `cargo build --release --features rocm`. Checkpoint `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (vocab 128256).

## Decode throughput

`scripts/bench_decode.sh` at its default pp512/tg128 shape with the `--temperature` / `--top-p` options this change adds, three runs per arm, before and after interleaved run by run (before, after, before, after). Every run went through `scripts/rocm_gpu_guard.sh` (the copy fixed in lablup/mlxcel#2098, run from the #2065 worktree) with `--idle-secs 75`: 75 consecutive 1 Hz samples with `/sys/class/kfd/kfd/proc` empty and no compiler, which took 112 to 116 s of wall time per run, then the run under the monitor. A parallel unit benchmarked with the guard's default 90 at the same time; at equal windows both guards started together and rejected each other, so the shorter sample count was used to break the tie. All 12 accepted runs were clean on their first attempt. Raw rows: `benchmarks/rocm_strixhalo-gfx1151_2026-10-05_samplers-{before,after}_{t0.8,t0.8p0.95}.csv`.

| Configuration | Path after | Before tok/s | After tok/s | Medians |
|---|---|---|---|---|
| `--temperature 0.8` | Gumbel-max kernel | 37.91 / 37.67 / 38.32, median 37.91 | 37.59 / 38.26 / 39.25, median 38.26 | 1.01x, inside the spread |
| `--temperature 0.8 --top-p 0.95` | rejection kernel | 36.26 / 33.98 / 32.81, median 33.98 | 37.82 / 37.56 / 37.06, median 37.56 | 1.11x |

`compare_bench_csv.py --before <before csv> --after <after csv>`, which keeps the last row per model and so compares the third runs, reports 1.024x and 1.130x.

The Gumbel-max row is not a measurable change: the arms overlap and the medians differ by 0.9%, against a spread of 1.7% before and 4.4% after. The top-p row is: every after run is faster than every before run, by 1.04x, 1.11x and 1.13x in the three interleaved pairs. The before arm also drifted down across its three runs (36.3 to 32.8), which the after arm did not. The cause was not isolated; the chain runs an `argsort` over all 128256 entries per token for top-p, which is what the rejection kernel replaces. Greedy decode reaches neither kernel and was not rerun.

The issue's second command (`--top-k 40 --top-p 0.95`) does not reach the rejection kernel at this vocabulary: the routing policy sends top-k with top-p to the kernel only up to vocab 32768 (`REJECTION_JOINT_VOCAB_MAX`, measured on M1 Ultra), so that configuration runs the stock chain before and after. Top-p alone is the routed configuration and is the one measured. Whether the joint cap is right on ROCm was not measured.

## Correctness

Sampling is distributional, so the ports are held to the graph in two ways.

**Fixed key.** Both kernels draw one Philox key per call from MLX's default key sequence, and given the key their output is deterministic. `sampling_fixed_key_tests` (`src/lib/mlxcel-core/src/sampling_fixed_key_tests.rs`) reseeds, reads the key the next launch will take through the new `random_bits` bridge function, reseeds again, launches, and recomputes the draw on the host from that key:

- Gumbel-max, 48 rows of 5003 logits in f32 (T 1.0 and 0.7), bf16 (T 1.3) and f16 (T 0.9): the kernel's index equals the host's `argmax(logits / T + g)` on every decided row, and so does the MLX graph's `argmax` with the same noise pushed through it. A row counts as decided when its winning score leads by at least 1e-4.
- Rejection under min-p 0.05, 40 rows of 3001 entries, T 1.0 and 0.7: min-p resolves its whole threshold before the first draw, so round 0 accepts and the token follows from the round-0 Philox word and the kernel's thread-major scan order; the kernel matches the host on every decided row (target at least 1e-5 of the mass from a cumulative boundary).

With the Philox counter (`row + 1`) or the drawn word (`c1` for `c0`) changed in the HIP sources, both tests fail on row 0.

**Statistical.** Two-sample chi-square tests (pooled to 20 draws per bin, critical value at p = 1e-6) between each kernel and `fused_sample_categorical`, the explicit graph arm, on the same input: 400,000 draws per arm for the Gumbel-max kernel against `random::categorical` at T 0.8, and for the rejection kernel against the stock chain at top-p 0.9, and at top-p 0.95 with min-p 0.02. Both pass. The existing suites, which test each kernel against the exact softmax or truncated distribution, now run on ROCm instead of returning early: `sampling_gumbel_tests` and `sampling_rejection_tests` pass on gfx1151.

**Speculative decoding.** The classic `SpeculativeGenerator` path (`mlxcel generate --draft-model`) verifies drafted tokens with the target's sampler, and `fused_sample_probs` follows the same routing, so with top-p the target's draws and reported distribution now come from the rejection kernel. Qwen3-30B-A3B-4bit with Qwen3-0.6B-4bit as the drafter, `--temp 0.8 --top-p 0.95 -n 128`, `MLXCEL_SPECULATIVE_ACCEPT_DIAG=1`: the dispatch log names the rejection kernel after and the stock chain before, the output is coherent, and the measured per-position acceptance sits near the closed form for its rule in both arms (after: sampler-match 0.643 against `sum_prod` 0.665, stochastic 0.732 against `sum_min` 0.676; before: 0.709 against 0.678 and 0.673 against 0.699, about 110 tested positions each). These are single runs and show the path works, not a distribution; the tests above carry that. No MTP checkpoint was available locally, so the MTP and DFlash round loops were not run; `docs/environment-variables.md` (`MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT`) records that they select the target token by argmax.

## Reproducing

```bash
cargo build --release --features rocm --bin mlxcel --bin mlxcel-bench-decode
cargo test --release --features rocm -p mlxcel-core --lib sampling_ -- --test-threads=1
cargo test --release --features rocm --test sampling_gumbel_kill_switch --test sampling_rejection_kill_switch
python3 scripts/compare_bench_csv.py --before benchmarks/rocm_strixhalo-gfx1151_2026-10-05_samplers-before_t0.8p0.95.csv --after benchmarks/rocm_strixhalo-gfx1151_2026-10-05_samplers-after_t0.8p0.95.csv
scripts/rocm_gpu_guard.sh -- scripts/bench_decode.sh models/mlx/Meta-Llama-3.1-8B-Instruct-4bit --temperature 0.8 --top-p 0.95
```
