# Gemma3n decode profile: is a compiled fusion justified? (#329)

Apple M1 Ultra (128 GB, macOS 26.5.0), MLX 0.31.2, mlx-vlm 0.4.4. Decode
throughput measured with `mlxcel-bench-decode` (raw continuation prompt, 80-100
tokens after a 20-token warmup, `caffeinate -i`, best-of-3 with the machine
cool). Pipeline, op-count, and bandwidth data collected with the
`mlxcel-gpu-profiling` hooks. This is the profile-first deliverable for #329;
it follows the same method as `moe-decode-gap-investigation.md` and the prior
Gemma3n bf16 decode-gap work.

## Verdict

No compiled fusion is justified for Gemma3n decode now. Three independent
measurements say a small-op fusion would not help on M1 Ultra, and the one
lever that does move decode is a scheduling knob, not a kernel:

1. Decode is ~92% GPU-bound. The Rust graph-build step (`forward`), where any
   FFI-crossing reduction would land, is ~7% of the token and is fully hidden
   behind GPU execution. A fusion that only collapses FFI crossings (the
   `MLXCEL_FUSED_QK_NORM` result on Qwen3, 1-3.4% slower on M1 Ultra) cannot win
   here for the same reason.
2. The decode-time overhead above pure weight streaming is dominated by
   command-buffer dispatch gaps, not small-kernel compute. Raising
   `MLX_MAX_OPS_PER_BUFFER` recovers +11-13% with zero code and zero risk; a
   compiled fusion targets the wrong cost.
3. The Gemma3n-specific fusions worth doing already shipped and run on M1 Ultra
   off NA hardware: the stacked AltUp path (#60) and the compiled gelu_topk /
   GeGLU activation kernels (landed before #60, reused by it). The #60 fused MLP
   bridge is bf16-only, so the 4-bit MLP here runs gate/up/down as separate
   `QuantizedMatmul` plus that compiled activation; extending the bridge to
   quantized weights would only collapse the FFI crossings that point 1 shows
   are hidden. The prior M5 investigation (the source of the
   `mlxcel-gpu-profiling` skill) found no remaining fusion lever above ~2.6%.

On top of that, mlxcel is already ahead of the only Python runtime that loads
these multimodal checkpoints (mlx-vlm; mlx-lm cannot load them), so there is no
parity gap pulling for a risky kernel.

## 1. Decode baseline (text and VLM)

Greedy, raw prompt, best-of-3, default `MLX_MAX_OPS_PER_BUFFER`.

| Checkpoint | layers / kv-shared | mlxcel text | mlxcel VLM (image) | mlx-vlm text | mlx-vlm VLM | mlxcel / mlx-vlm |
|------------|-------------------:|------------:|-------------------:|-------------:|------------:|-----------------:|
| gemma3n-e2b-4bit | 30 / 10 | 83.1 | - | 68.9 | - | 1.21x |
| gemma3n-e4b-4bit | 35 / 15 | 62.9 | 58.7 | 52.6 | 47.9 | 1.20x (text), 1.23x (VLM) |

mlx-vlm numbers are wall-clock `(tokens-1)/(t_last - t_first)` from
`mlx_vlm.stream_generate` (its `generation_tps` field is unreliable on a raw
prompt). mlx-lm cannot load these checkpoints (`KeyError('model')` on the
multimodal config), so mlx-vlm is the reference. mlxcel leads on both models and
both modalities.

## 2. Pipeline split: GPU-bound, not FFI-bound

`MLXCEL_PROFILE_PIPELINE_DETAIL`, per decode token:

| Checkpoint | reshape | forward (Rust graph build) | sample | async_eval (GPU) | item_wait | GPU share |
|------------|--------:|---------------------------:|-------:|-----------------:|----------:|----------:|
| e2b-4bit | 0.001 ms | 0.92 ms | 0.002 ms | 11.44 ms | 0.16 ms | ~92% |
| e4b-4bit | 0.002 ms | 1.09 ms | 0.002 ms | 14.47 ms | 0.19 ms | ~92% |

`forward` is the entire Rust-side graph construction, every `mlxcel_core::*` FFI
call for the token. It overlaps the previous token's GPU work via `async_eval`
lookahead, so its ~7% never appears on the critical path. This is the decisive
test against an FFI-collapsing fusion: there is no FFI cost to collapse.

## 3. Per-layer structure and where the GPU time goes

Each Gemma3n decoder layer runs far more than a dense transformer block: AltUp
predict/correct (4 parallel planes, a router norm + modality router matmul +
small coefficient matmuls), a LAUREL branch (two linears + a norm), per-layer
input gating (gate + projection + norm), four block norms plus q/k/v norms, and
KV sharing on the back half of the stack. The one-decode-token graph for
e4b-4bit is ~4209 nodes:

| Op | count | Op | count | Op | count |
|----|------:|----|------:|----|------:|
| Broadcast | 442 | QuantizedMatmul | 433 | RMSNorm | 357 |
| Add | 351 | Transpose | 320 | Slice | 315 |
| Squeeze | 254 | AsType | 249 | Reshape | 222 |
| Multiply | 205 | Full | 149 | ExpandDims | 144 |
| Matmul | 105 | Tanh | 71 | RoPE | 55 |
| SliceUpdate | 40 | Concatenate | 36 | ScaledDotProductAttention | 35 |

The long `Compiled*` nodes in the same graph are the compiled activation fusions
already in place (gelu_topk / GeGLU, predating #60 and reused by it):
`CompiledBroadcast...Tanh...` is gelu_topk / GeGLU, the `CompiledMaximum...Erf...`
chain is the gelu_topk cutoff, and `CompiledBroadcastBroadcastSubtractSquare` is
the magnitude reduction. The MLP gate/up/down are not bundled here because the
#60 fused MLP bridge is bf16-only; on these 4-bit checkpoints they are the
per-layer `QuantizedMatmul` nodes counted above. Most of
the high-count ops are views (Transpose, Slice, Squeeze, Reshape, ExpandDims,
Flatten) that carry no kernel, or cheap elementwise that MLX fuses at eval.

To separate the streaming work from the structure, a synthetic graph of just the
per-token quantized GEMVs (q/k/v/o, gate/up/down across all layers with the
KV-shared layers skipping k/v, plus the tied LM head), distinct weights per
layer, one `eval` per token:

- Pure-GEMV streaming floor (e4b): 6.17 ms/token, 2472 MB streamed, 401 GB/s.
- mlxcel e4b GPU time: 14.47 ms/token.

So pure weight streaming is ~43% of GPU time; the other ~57% is structure: the
norms, AltUp, LAUREL, per-layer gating, attention, dtype glue, and the dispatch
gaps between all of it. The streamed-bytes accounting also shows decode never
touches the heavy multimodal weights: of the 5.82 GB e4b file, 1.36 GB audio
tower + 0.59 GB vision tower + 1.32 GB per-layer lookup embedding are not
streamed per text token; the ~2.52 GB that is streamed matches the GEMV floor.

## 4. The real lever is command-buffer batching, not fusion

`MLX_MAX_OPS_PER_BUFFER` controls how many ops MLX batches before committing a
command buffer (runtime, no rebuild). Sweeping it on Gemma3n:

| MLX_MAX_OPS_PER_BUFFER | e2b-4bit | e4b-4bit |
|------------------------|---------:|---------:|
| default (unset) | 82.7 | 63.0 |
| 100 | - | 67.4 |
| 200 | - | 69.5 |
| 1000 | 93.2 | 70.0 |
| 2000 | 92.5 | 67.7 |

A clean monotonic climb to a plateau near 1000, +12.7% on e2b and +11% on e4b.
Per the profiling skill this response (GPU idle between kernels, fixed by
coarser dispatch) is the signature of dispatch-gap binding, whose fix is
batching or deeper command-buffer lookahead, not a fused kernel. It also at
least partly explains the 57% structure overhead in section 3: a meaningful
slice of it is GPU idle waiting on command-buffer commits.

This is the one place Gemma3n differs from the MoE decode gap (#268), where the
same sweep was flat. Gemma3n's higher op density per layer (AltUp 4-plane,
LAUREL, per-layer inputs, dual norms) makes the default buffer size too small,
so the GPU stalls at buffer boundaries. A compiled fusion would reduce the op
count and therefore shrink some of these gaps, but the env knob already captures
that benefit and more, for free, and a fusion of the small glue ops would not
beat it.

## 5. Text vs VLM: vision is a prefill cost only

| e4b-4bit | prefill tok/s | decode tok/s |
|----------|--------------:|-------------:|
| text prompt | 98 | 63.0 |
| image prompt | 483 | 58.7 |

The vision tower runs once during prefill (the high image-prompt prefill tok/s
reflects the image's many soft tokens processed in one pass). Decode is the same
per-layer structure either way; the small VLM decode delta is the longer KV
context after ingesting the image soft tokens (more K/V to read per step), which
is KV bandwidth, not a fusable small-op hotspot. Any decode fusion target is
identical with or without an image, and the vision path is not a decode
bottleneck.

## 6. Conclusion and when a fusion would become justified

Close #329 with no compiled fusion. The decode profile shows no fusable
small-kernel hotspot on M1 Ultra: decode is GPU-bound with FFI hidden, the
worthwhile Gemma3n fusions already shipped (#60 and earlier), mlxcel already
leads mlx-vlm,
and the residual overhead is dispatch-gap idle that a scheduling knob recovers
better than any kernel could.

A fusion would only become justified if a GPU trace (Xcode, the skill's Step 4)
showed the GPU busy (not idle) inside a specific small-kernel chain whose summed
compute time is large, after the dispatch-gap portion is removed. Today the
evidence points the other way.

Recommended follow-up, which is not a fusion and is out of scope for #329: make
`MLX_MAX_OPS_PER_BUFFER` a Gemma3n-aware default around 1000, gated on hardware.
It must be gated, because the prior M5 investigation found the default fastest
on M5-class hardware (larger buffers were slower there); blindly raising it
would regress M5. That needs a cross-hardware re-bench before it lands, so it
belongs in its own issue.

### Hardware-gated default (landed in #353)

`apply_metal_ops_per_buffer_default` (`src/lib/mlxcel-core/src/hardware.rs`) now sets `MLX_MAX_OPS_PER_BUFFER=1000` at process start on pre-M5 Apple Silicon (`AppleSiliconGen` M1 through M4, Metal GPU family 3) when the variable is unset. M5+ (Neural Accelerator) and non-Apple (CUDA) are left on MLX's default, and an operator-set `MLX_MAX_OPS_PER_BUFFER` always wins. The CLI, server, and `mlxcel-bench-decode` binaries apply it before any MLX op.

Cross-hardware basis for the gate:

| Class | MLX_MAX_OPS_PER_BUFFER default | Source |
|-------|--------------------------------|--------|
| M1-M4 (no Neural Accelerator) | 1000 (auto) | M1 Ultra sweep in section 4 + re-bench below |
| M5+ (Neural Accelerator) | MLX default (untouched) | gemma3n-decode-profile-m5max.md (#358): the sweep is flat on M5 Max, and the earlier M5 study recorded larger buffers as slower |
| Non-Apple (CUDA, e.g. GB10) | MLX default (untouched) | Metal command-buffer knob, irrelevant off Apple GPUs |

M1 Ultra re-bench with the gate active (2026-06-19, best-of-3, raw prompt, 80 tokens after a 20-token warmup, `caffeinate -i`):

| Checkpoint | gated default (1000) | recorded section-4 default | gain |
|------------|---------------------:|---------------------------:|-----:|
| gemma3n-e2b-4bit | 92.5 | 82.7 | +11.9% |
| gemma3n-e4b-4bit | 68.4 | 63.0 | +8.6% |

The gate fires on this M1 Ultra (detected as `M1`) and reproduces the 1000-row throughput from section 4. An explicit low override is respected and collapses throughput to 67.7 (e2b) and 51.8 (e4b) at `MLX_MAX_OPS_PER_BUFFER=10`, confirming the lever direction and that operator settings win.

Generalization: the gain concentrates in Gemma3n's high-op-density stack, but the cap is not Gemma3n-only. Dense qwen3-8b-4bit (86.1) and MoE qwen3-30b-a3b-4bit (89.0) also run faster at 1000 than at a starved cap (75.9 and 79.3 at ops=10), so the generation-wide pre-M5 gate does not regress them. The MoE decode-gap sweep (#268) was flat, consistent with the dispatch-gap binding being strongest where per-layer op density is highest. Section 4's sweep is monotonic up to the plateau near 1000 and regresses slightly at 2000, so 1000 sits at the knee.

Not yet benched: GB10 (no access in the landing environment), recorded here as a follow-up. The gate leaves GB10 on MLX's default (it is non-Apple, CUDA), so it is unaffected until measured.

### Reproduce

```bash
# decode baseline (best-of-3, cool)
for i in 1 2 3; do caffeinate -i ./target/release/mlxcel-bench-decode \
  -m models/gemma3n-e4b-4bit -p "Hello, how are you today?" -n 100 \
  --warmup-tokens 20 --no-chat-template 2>/dev/null | grep Decode:; done

# pipeline split
MLXCEL_PROFILE_PIPELINE=1 MLXCEL_PROFILE_PIPELINE_DETAIL=1 \
  ./target/release/mlxcel-bench-decode -m models/gemma3n-e4b-4bit -p Hi \
  -n 100 --warmup-tokens 20 --no-chat-template 2>&1 | grep PIPELINE

# command-buffer sweep
for OPS in 100 200 1000 2000; do MLX_MAX_OPS_PER_BUFFER=$OPS \
  ./target/release/mlxcel-bench-decode -m models/gemma3n-e4b-4bit \
  -p "Hello, how are you today?" -n 80 --warmup-tokens 20 \
  --no-chat-template 2>/dev/null | grep Decode:; done

# one-decode-token op histogram
MLXCEL_EXPORT_DECODE_DOT=/tmp/g3n.dot ./target/release/mlxcel-bench-decode \
  -m models/gemma3n-e4b-4bit -p Hi -n 3 --warmup-tokens 3 \
  --no-chat-template >/dev/null 2>&1
grep -oE 'label ="[A-Za-z0-9_]+' /tmp/g3n.dot | sed 's/label ="//' \
  | sort | uniq -c | sort -rn | head -30
```
