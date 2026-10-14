# Raw data for mxfp4-dense-qmv-gfx1151-2026-10-10

One guarded `rocprofv3 --kernel-trace --hip-graph-trace --stats -f csv` decode profile of gpt-oss-20b-MXFP4-Q4 on gfx1151, taken for issue #2246's profile-first branch. The shape is `scripts/rocm_decode_profile.sh`'s default (pp512 / tg128, greedy, 20-token warmup); `guard.log` is the matching `scripts/rocm_gpu_guard.sh` log with local paths scrubbed.

Files: `gpt-oss-20b-MXFP4-Q4_greedy_kernel_stats.csv` is rocprofv3's whole-process per-kernel summary (warmup, prefill and decode together); `gpt-oss-20b-MXFP4-Q4_greedy_decode_kernels.csv` is the decode-window per-kernel table cut out by `scripts/rocm_decode_profile.py`; `gpt-oss-20b-MXFP4-Q4_greedy_summary.json` is the run summary; `*_bench.log` and `*_plain_bench.log` are the profiled and unprofiled bench outputs. The full kernel trace (hundreds of MB) is not committed.

## Finding

The dense mxfp4 qmv kernels named in issue #2246 (`qmv_warp_shared_kernel<*, unsigned char, 4, 32, false, *>`, `qmv_warp_shared_batched_kernel<..., 4, 32, false, ...>`, `qmv_warp_noshared_kernel<..., 4, 32, false, ...>`) receive zero dispatches, in the decode window and in the whole process. Their combined share of decode GPU time is 0.00%, under the issue's 5% threshold, so the issue closed without code changes.

Why: in this checkpoint only the MoE expert weights are mxfp4 group-size-32; the dense projections are group-size-64 4-bit affine quantization (q_proj scales are [4096, 45] over K = 2880, so 2880 / 45 = 64 groups), which dispatches to `qmv_wide_kernel<hip_bfloat16, hip_bfloat16, 64, 4>` (26.87% of decode GPU time, 12319 calls). The experts already run the #2235 word path: `gather_qmv_warp_shared_kernel<hip_bfloat16, unsigned char, 4, 32, false, 16>` at 50.32% of decode GPU time (9144 calls). The run measured 60.2 tok/s profiled, 62.88 tok/s plain, 136088 decode dispatches and 14.02 GPU ms/token.

Reproduce with `session.sh`.
