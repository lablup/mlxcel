# mxfp8 and FP8 block checkpoints on ROCm: Radeon 8060S (gfx1151), 2026-09-30

Validation run for issue #1807. It answers three questions about the mxfp8 path on the experimental ROCm backend:

1. Do the two mxfp8 matmuls mlxcel runs, the MoE expert `gather_qmm` and the dense `quantized_matmul`, compute the right numbers?
2. Does a vendor FP8 block checkpoint, which `src/models/fp8_block.rs` requantizes to mxfp8 at load, decode the same tokens on ROCm as on a reference device?
3. Does it matter that ROCm's load-time `quantize` is not bit-identical to the CPU's?

Short answers: yes; yes against the CPU device (Metal was not available for this run), with 0 of 50 decided positions disagreeing; and no.

## Environment

| Field | Value |
|---|---|
| Hardware | AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), 96 GiB VRAM carve-out, 30 GiB host RAM |
| OS | Debian GNU/Linux 13 (trixie), kernel 6.18.12+deb13-amd64 |
| Backend | ROCm (HIP 7.15.26333), `--features rocm` |
| mlxcel | branch `feature/issue-1807-mxfp8-rocm-e2e` on `d1128266` |
| Checkpoint | [`Lazarus-Ai/ReAligned-Qwen3.5-0.8B-FP8`](https://huggingface.co/Lazarus-Ai/ReAligned-Qwen3.5-0.8B-FP8) at revision `db97e6a70bef791fb44a7e054fc7a9d8f50f090d` (dense Qwen3.5 0.8B fine-tune; `quant_method: fp8`, `weight_block_size: [128, 128]`, `*.weight_scale_inv` sidecars) |
| Traces | `benchmarks/logit_traces/fp8_block_gfx1151/` |

No official small Qwen3.5 FP8 release exists: `Qwen/Qwen3.5-27B-FP8` and `Qwen/Qwen3.5-35B-A3B-FP8` are 31 and 37.5 GB, and on a 30 GiB host no CPU reference would be possible for either. This community checkpoint uses exactly the vendor layout the loader converts, so it exercises the same load path. It is dense, so it covers the requantization and the dense mxfp8 `quantized_matmul`, not `gather_qmm`; the MoE path is covered by the op tests below.

## 1. Op tests

`src/models/switch_layers_mxfp_tests.rs` (in the `mlxcel` lib target) drives both call sites through the layers production code uses, `SwitchLinear::forward` and `UnifiedLinear::forward`, and compares against a reference computed on the host: packed codes and E8M0 scales decoded in Rust (E4M3 for mxfp8, E2M1 for mxfp4) and accumulated in f64. A separate test pins the host decoder bit for bit to MLX `dequantize`.

The gather cases cover every branch a block-float mode reaches in `GatherQMM::eval_gpu` (`patches-rocm/mlx/backend/rocm/quantized/qmm.hip`). Every specialised path there (grouped WMMA prefill, expert-batched, tiled, wide, idot and the warp-shared kernel) is gated on `mode_ == Affine`, so mxfp4 and mxfp8 reach exactly one launch, the `gather_qmv_kernel<T, uint8_t, BITS, 32, false>` block from `LOCAL_FIXES.md` item 10. The cases hit all three `T` arms (bf16, f16, f32), both `BITS` arms, the unsorted and the sorted-rhs schedule (the latter with the activation stride at `K` and at 0), `M = 1` and `M = 4`, and a partial second column block. The opt-in expert-batched path (item 9) is affine-only, so it cannot be reached from these modes. This is the default path.

Relative L2 error against the host reference, worst case over all gather and dense cases:

| Backend | bf16 | f16 | f32 |
|---|---|---|---|
| ROCm gfx1151 | 1.8e-3 | 2.2e-4 | 2.9e-7 |
| MLX CPU backend (same test binary, CPU default device, reduced matrix, see below) | 8.2e-3 | 1.0e-3 | not run |
| Test bound | 2e-2 | 4e-3 | 2e-3 |

The CPU row is the arm `mxfp_matmuls_match_host_reference_on_cpu_device` runs. MLX's CPU `fp_qmm_t` is scalar, so the full matrix took about 38 s and held the default-device lock for that time on every backend. The arm now runs both modes with the unsorted prefill, sorted shared-activation and multi-row gather cases and the dense `M = 1` and `M = 4` cases, in bf16 and f16, on a 64-wide output, in about 2.5 s. The first CPU measurement, over the full matrix (all five gather cases, dense `M = 1, 4, 64`, f32 included, 320-wide output), gave 7.9e-3 (bf16), 1.0e-3 (f16) and 1.2e-7 (f32).

The f32 bound is wider than these numbers need because on CUDA sm80 and later the sorted prefill case takes MLX's grouped GEMM, which runs f32 through TF32 tensor cores by default (`MLX_ENABLE_TF32`), rounding activations to a 10-bit mantissa.

With item 10 reverted in the overlay and the test binary rebuilt, both gather tests fail at their first case (`DecodeUnsorted bf16`) with non-finite output.

## 2. FP8 block checkpoint, ROCm against the CPU device

Metal (#1809's reference) is not available on this host, so the reference is the same binary on the CPU device, `MLXCEL_DEVICE=cpu`. Two fixes were needed before that reference could run at all:

- `examples/logit_trace` never read `MLXCEL_DEVICE`, so a "CPU" trace ran on the GPU. It now calls `initialize_runtime_checked` like the CLI and records the resolved device in a `# device` header line.
- On a ROCm build, `MLXCEL_DEVICE=cpu` aborted every attention model at its first token with `NYI`: the overlay's `ScaledDotProductAttention::use_fallback` never checked the stream device (`LOCAL_FIXES.md` item 23, guarded by `tests/cpu_device_sdpa.rs`, which aborts with the guard reverted).

The CPU backend in this build is slow for this checkpoint, about 2.7 minutes per traced token on one core; stack samples land in MLX's scalar `fp_qmm_t`. So the CPU arm is a 32-token width, `32 8 8 0` (256 positions, forward width `M = 32`), assembled from eight processes that each traced one chunk (`METADATA.txt` says how). A `w256` CPU arm was started and abandoned after 6.8 CPU-hours without finishing its first forward.

| Width | Positions | Top-1 disagreements | Decided disagreements (`--decided 2.0`) | Largest gap at a disagreement | Perplexity CPU / ROCm |
|---|---|---|---|---|---|
| `w32` (`32 8 8 0`) | 256 | 14 | 0 / 50 | 0.125 | 100.80 / 100.95 |

All 14 disagreements are at reference gaps under 0.5, and in every one the CPU's token is ROCm's second choice. That is inside the 2.0 threshold #1809 set for Metal against ROCm, where the largest gap at a disagreement across twelve pairs was 1.125. The CPU is a weaker reference than Metal would be: MLX's CPU `fp_qmm_t` accumulates in the activation dtype (bf16 here), which is also why it is the least accurate backend in the op-test table above.

## 3. GPU-quantized against CPU-quantized weights

ROCm `quantize` produces the same E8M0 scales as the CPU but rounds ties differently. To see whether that moves decoded tokens, the three widths were traced on the GPU twice: once as shipped, and once with `requantize_block_fp8_weights` temporarily quantizing on the CPU stream (a local experiment switch, not committed). Both arms compute on the GPU; only the packed weights differ. The table compares the shipped arm (`rocm_*`, candidate) against the CPU-quantized arm (`rocm_cpuquant_*`, reference), so "reference" below means the CPU-quantized weights. The ROCm traces are deterministic: a rerun of `w1` and `w256` was byte-identical, so every difference below comes from the weights.

| Width | Positions | Top-1 disagreements | Decided disagreements (`--decided 2.0`) | Largest gap at a disagreement |
|---|---|---|---|---|
| `w1` (`1 128 8 0`) | 128 | 17 | 0 / 2 | 0.375 |
| `w8` (`8 80 8 512`) | 640 | 9 | 0 / 192 | 0.250 |
| `w256` (`256 2 8 0`) | 512 | 16 | 0 / 152 | 0.375 |

Every disagreement is at a position where the reference was choosing between near-ties, and in 39 of the 42 the reference's token is the candidate's second choice (third in the other 3). Quantizing on the CPU stream would also cost load time: the `w1` trace took 250 s instead of 6 s, almost all of it in requantization. So load-time quantization stays on the default device, and the difference is documented in the `fp8_block.rs` module doc. The existing round-trip test `fp8_block_requantize_round_trip_stays_within_half_an_e4m3_step` is not backend-gated and passes on ROCm; no separate test was added because the quantize stream did not change.

## Generation

```
$ mlxcel generate -m models/fp8/ReAligned-Qwen3.5-0.8B-FP8 \
    -p "What is the capital of France? Answer in one sentence, then name two famous landmarks there." \
    -n 120 --temp 0 --show-reasoning
```

On ROCm the load log reports 132 tensors requantized to mxfp8 in 565 ms, and the model thinks through the answer ("The capital of France is Paris ... The Eiffel Tower comes to mind immediately. The Louvre Museum is another major landmark.") before the 120-token budget ends, at 56 tok/s.

## Not covered

- The official Qwen3.5 FP8 MoE release (`Qwen/Qwen3.5-35B-A3B-FP8`) and `Qwen/Qwen3.5-27B-FP8` were not downloaded or run. The MoE mxfp8 path is verified at op level only.
- Metal was not available; the reference device is the MLX CPU backend in the same ROCm build.
- The CPU arm covers one width, `w32`, with 50 decided positions. The `w1`, `w8` and `w256` widths were traced on ROCm only (for the quantization comparison in section 3). The MLX CPU backend runs this checkpoint's mxfp8 matmuls in a scalar single-threaded loop, and Qwen3-0.6B-4bit takes about two minutes per decode step on the CPU device as well, so a CPU arm at those widths would take many hours.

## Reproducing

```bash
make release-rocm   # or: cargo build --release --features rocm --bin mlxcel --example logit_trace
M=models/fp8/ReAligned-Qwen3.5-0.8B-FP8
./target/release/examples/logit_trace $M tests/fixtures/wikitext2_excerpt.txt 32 8 8 0 > rocm_w32.tsv
# hours on the CPU; METADATA.txt describes running the chunks in parallel
MLXCEL_DEVICE=cpu ./target/release/examples/logit_trace $M tests/fixtures/wikitext2_excerpt.txt 32 8 8 0 > cpu_w32.tsv
python3 scripts/compare_logit_traces.py cpu_w32.tsv rocm_w32.tsv --decided 2.0
# section 3, per width W in w1 w8 w256 (traces committed under benchmarks/logit_traces/fp8_block_gfx1151/)
python3 scripts/compare_logit_traces.py realigned-qwen3.5-0.8b-fp8_rocm_cpuquant_$W.tsv realigned-qwen3.5-0.8b-fp8_rocm_$W.tsv --decided 2.0
cargo test --features rocm --lib models::switch_layers::mxfp_tests -- --test-threads=1
cargo test --features rocm --test cpu_device_sdpa
```

`METADATA.txt` in the trace directory records the versions and revisions, `RUNS.txt` the exit status and row count of every run, and `SHA256SUMS` covers every trace file.
