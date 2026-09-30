# Jamba on GB10: attribution of the 300 s chat request and the CUDA Mamba1 scan (issue #1981)

Issue #1981 reported that `mlxcel-server` took 303944 ms (turn 1) and 376530 ms
(turn 2) for a ~3.3k-token Jamba chat request on GB10. That was measured on the
PR #1978 branch, before PR #2000 (merged 2026-09-27) made the Jamba prefill scan
linear. This record reproduces the old cost on GB10, measures what was left on
main, and measures the CUDA port of the fused Mamba1 scan that removes it.

- **Date:** 2026-09-30
- **Host:** NVIDIA GB10 (spark-101), kernel 7.0.0-1019-nvidia, driver
  580.178.04, CUDA 13.0 (V13.0.88), `MLX_CUDA_ARCHITECTURES` auto (sm_121)
- **Build:** `cargo build --release --features cuda`
- **Checkpoint:** `jamba-v0.1-4bit` (config `model_type` jamba, 1.7 GB on
  disk). Despite the name it is a reduced variant: 28 layers (26 Mamba, 2
  attention at layers 7 and 21), hidden 2560, Mamba intermediate 5120,
  d_state 16, dt_rank 160, `num_experts` 1 (so MoE routing is a single dense
  expert), all tensors bf16 or 4-bit affine.
- **Request:** chat, 3299 prompt tokens, `max_tokens` 64, temperature 0,
  default server launch (`mlxcel-server -m <ckpt> --port 8093`). Turn 2
  appends the answer and a follow-up (3380 to 3386 tokens).

## Attribution

| Code | What the scan does | Cost on GB10 |
|---|---|---|
| Before #2000 (the issue's binary) | per-step loop that also `concatenate`d every `[B, 1, 5120, 16]` state onto the growing result | prefill 8.1 s at 548 tokens, 26.6 s at 1059 (n=3 each): quadratic, extrapolates to about 260 to 300 s at 3299. At 3299 tokens the server run exhausted the host's 121 GB and was OOM-killed. |
| main `929c80ab` | per-step loop of about ten small MLX ops per timestep per Mamba layer, `y` rows stacked once | server request 4.3 s; CLI prefill 4.1 s of which about 2.3 s is the scan (a build with the scan replaced by a shape-correct trivial op prefilled in 1.33 s) |
| this change | one fused CUDA kernel per layer | server request 1.5 s |

So the 300 s was the quadratic `concatenate`, already removed by #2000. The
remaining dominant cost on CUDA was the per-timestep graph loop (about 86k
timestep iterations per 3.3k-token prefill across 26 layers), because the fused
scan kernel from #2005 was Metal-only. MoE dispatch is not a factor for this
checkpoint (one expert). Turn 2 was slower than turn 1 in the issue because it
re-prefilled the whole, longer history: the server log shows `cached=0` on every
turn for this model, on main and with this change alike (not addressed here).

## Results

Server, wall time per request, two server runs per arm in ABBA order, three
prompts per run (n=6 per cell):

| Arm | Turn 1 median (range) | Turn 2 median (range) |
|---|---|---|
| main | 4.33 s (4.23 to 4.49) | 4.17 s (4.08 to 4.28) |
| fused CUDA scan | 1.51 s (1.51 to 1.74) | 1.53 s (1.52 to 1.56) |

CLI (`mlxcel generate --profile`, 3299-token prompt, 64 tokens):

| Arm | Prefill (n) | Decode tok/s |
|---|---|---|
| main | 3.59 to 4.16 s (8, two sessions) | 68.7 to 76.7 |
| fused CUDA scan | 2.14 to 2.21 s (3) | 75.7 to 79.0 |
| fused, `MLXCEL_MAMBA1_SCAN_KERNEL=0` | 4.13 to 4.21 s (2) | 68.9 to 71.6 |

The CLI prefill includes the kernel's one-time NVRTC compile (MLX does not
disk-cache custom CUDA kernels), which the server pays during its startup
warmup instead (warmup 1.1 s to 1.8 s).

## Output identity

The CUDA kernel is written to reproduce the graph scan bit for bit: each step
rounds in the activation dtype with the graph's op order, uses MLX's own `Exp`
device op, and reduces `y` with the same `cg::reduce` the graph's gemv uses.
Multiplies and adds use the `_rn` intrinsics so nvrtc cannot contract them into
an fma (the first draft without this differed by one ulp in about half the
outputs).

- `mamba1_scan_parity_tests::cuda_kernel_is_bit_identical_to_the_graph_scan`:
  bit-identical output rows and final state against the graph scan for bf16,
  f16 and f32, d_state 8 and 16, 1, 7 and 33 steps, fresh and carried state.
- Server: all 12 requests of both fused runs match the first main run
  character for character (reasoning and content). The second main run
  differed from the first main run on one request (turn 2 of one prompt), so
  main itself is not fully deterministic run to run on this host; the fused
  runs matched the first main run on that request too.
- CLI greedy, 548, 1059 and 3299-token prompts, 64 tokens: identical to main.

## Not measured

Mamba / Falcon-Mamba on CUDA (`mamba.rs` keeps the graph scan there: its graph
path projects one timestep at a time, so switching would change output and
needs its own validation). ROCm has no port.
