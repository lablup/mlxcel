# Phase 2b findings: GPU decode throughput on the GB10

Issue #449 Phase 2b (deferred from Phase 2a until a GPU host existed; CUDA turned
out to be available on this box). Goal: real decode throughput for the exported
StableHLO on the GB10, with the Phase 2a harness artifacts removed. CPU numbers
through the same harness for comparison.

## What changed from the Phase 2a GPU run

Phase 2a reported a misleading GPU rate because the functional-call harness
re-uploaded all weights as graph inputs every step and copied full logits to the
host per token. Phase 2b fixes all three:

- weights uploaded to the device ONCE as resident IREE device arrays,
- on-device argmax (decode returns the next token id, so 4 bytes cross per step
  instead of a 513 KB logits copy),
- KV cache kept resident on the device across steps (the returned device buffers
  are fed straight back in).

Same IREE path for both targets (`--iree-hal-target-device=cuda` for the GB10,
`--iree-hal-target-backends=llvm-cpu` for CPU); prefill runs once on CPU to seed
the cache. Reference model Llama-3.2-1B, single sequence, greedy.

## Numbers

| variant | device | tok/s | ms/tok |
|---|---|---|---|
| int4 | GB10 | 4.0 | 252 |
| int4 | CPU  | 0.8 | 1274 |
| fp32 | GB10 | 5.5 | 182 |
| fp32 | CPU  | 1.8 | 541 |

All four stay coherent (same continuation prefix). Two findings fall out.

### 1. With the harness fixed, the GPU beats the CPU

int4 is 5x faster on the GB10 than on CPU, fp32 is 3x. The Phase 2a result where
the GPU looked slower than CPU was entirely the weight-reupload and per-token
logit-copy artifact, not the GB10. Resident weights plus on-device sampling is
what a real backend does, and it is what makes the GPU win.

### 2. int4 is SLOWER than fp32 on the GPU, confirming the Phase 2a fusion finding

This is the important one. fp32 decode is 5.5 tok/s; int4 decode is 4.0 tok/s on
the same GPU. int4 loses. The reason is exactly what the Phase 2a dispatch
analysis predicted: the dequant is a separate kernel that materializes the full
fp32 weight, and the matmul then reads that fp32 weight. So per step int4 does
strictly more memory traffic than fp32 (read packed int4, write the fp32 weight,
read the fp32 weight) plus an extra kernel launch, while fp32 just reads its
weight once. Without dequant-matmul fusion, dequant-in-graph int4 buys an 8x
smaller weight on disk and in device memory, but costs decode latency rather than
saving it. The storage win is real; the compute and bandwidth win on the matmul
is not, and will not appear until the dequant fuses into the GEMM (a `custom_call`
to an int4 GEMM, or IREE's quantized-matmul fusion recognizing the pattern).

## On the absolute numbers

4 to 5.5 tok/s for a 1B model on a Blackwell-class GPU is low, and it is not a
ceiling. Single-sequence decode is latency-bound: 145 tiny `dot_general` calls per
token (batch 1, vec-by-matrix), each a separate kernel launch, with no CUDA-graph
capture, no fused attention, and IREE's default CUDA codegen which is not tuned
for batch-1 LLM decode. The point of Phase 2b is the relative result (GPU over
CPU, and fp32 over unfused int4), not a tuned tok/s. Real throughput work
(CUDA-graph capture, fused attention, batching when the session allows it) is
backend-optimization territory beyond this spike.

## Recommendation

Carry two facts into the backend design. First, the export-route runs on the GB10
through IREE-CUDA and clears the CPU comfortably once weights are resident and
sampling is on-device, so the GPU path is viable. Second, do not ship
dequant-in-graph int4 expecting a decode speedup: as measured it is slower than
fp32 because the dequant does not fuse. int4 earns its place through weight
storage and transfer (8x), and through a fused int4 GEMM (`custom_call` or a
recognized quantized-matmul pattern), which is the next thing to prototype if
int4 decode latency matters. Until then, fp32 (or bf16) resident weights are the
faster single-sequence decode path on this GPU.

## Files

`phase2b_gpu.py` (resident weights, on-device argmax, CUDA and CPU through IREE),
`artifacts/phase2b_perf.json` (the table), `artifacts/*_decode_argmax.*` (the
exported on-device-sampling graphs and vmfbs).
