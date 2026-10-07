# Issue #2191: Qwen 3.5 DFlash verify RoPE on CUDA, raw data (GB10, 2026-10-07)

Raw data for `../../qwen35-dflash-verify-rope-gb10-2026-10-07.md`. Checkpoints `models/mlx/qwen3.5-4b-4bit` and `models/mlx/qwen3.5-4b-dflash`; prompt and reference ids are the 158-token prompt and 200 classic ids in `../dflash-width-2-4-residual-gb10-2026-09-21/`.

## Layout

- `capture/op_level.log`: `op_level_fast_rope_block_versus_decode_rows` and `op_level_compiled_silu_block_versus_rows` (step 1.2 and the MLP cell).
- `capture/capture-w4-rr0.log`, `capture/capture-w4-rr1.log`: `post_rope_capture_bisect_on_the_real_transcript` at width 4 on the served width-4 accept sequence, block RoPE and per-row RoPE. Each line names the first differing sub-op per kept row (`faN` is the N-th full-attention layer, layer `4N + 3`).
- `matrix/`: the attribution matrix, `block_versus_chain_byte_bisect_on_the_real_transcript` per cell, `w{2,4}-rr{0,1}-ld{1,0}-r{1,2,3}.log` (width, per-row RoPE off/on, `MLXCEL_SDPA_VECTOR_LARGE_D`, repeat). `summary.txt` is one line per cell.
- `branch/`: step 1.1, a temporary `[2191] rope branch` log line per full-attention forward, from one classic server and one width-4 DFlash server (`MLXCEL_MTP_ALLOW_INEXACT=1`, since at the time the decline was still in place). The log line is not in the shipped code.
- `served/`: `served_identity.py` output on the final binary: three `/v1/completions` prompts at temperature 0, 200 tokens, classic and DFlash at widths 2, 4, 8, 16 and unset (the server resolves 4), no environment overrides.
- `throughput/`: `price_rope.py` output, three interleaved rounds of classic-a, DFlash at the resolved width, and classic-b (the null arm), three measured requests per arm after a discarded warm-up. Round 2 ran with `--ignore-ci-gate` (its records carry `ci_gate_ignored: true`), so their `ci_job_running: false` reflects the patched check, not an idle runner. `interrupted_*` is the first round-2 attempt, cut off by a harness timeout after one arm.

## Harness

`harness/matrix.sh` runs the matrix; it needs `BIN` set to the lib test binary (`cargo test --release --features cuda -p mlxcel --lib --no-run`) and runs every cell under `gpu-lock`. `MLXCEL_Q35_ROW_ROPE=0|1` is read by the test, not the runtime, and overrides the per-row verify RoPE through `Qwen35Model::set_verify_rope_rows_for_test`.

The in-process tests that run the exactness probe need `MLX_CUDA_GRAPH_CACHE_SIZE=2000`, the value `mlxcel-server` applies by default. The test binary does not apply it, and at MLX's own default of 400 the probe crosses MLX's lifetime "Cache thrashing" limit and aborts the process.

`harness/branch.sh` is the step 1.1 run. `harness/served_identity.py` and `harness/price_rope.py` are the served checks; `price_rope.py` reuses `../dflash-sdpav-options-gb10-2026-09-21/harness/price_options.py` for one arm (host gate, server lifecycle, warm-up, streaming request) and only replaces the arm table.
