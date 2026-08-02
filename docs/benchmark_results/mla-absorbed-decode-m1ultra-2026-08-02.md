# MLA matrix-absorbed decode: Apple M1 Ultra, 2026-08-02

Validation run for issue #907 (epic #909).

**Outcome: the memory reduction is structural and large (8.89x), and decode speed
crosses over from a loss to a substantial win as context grows.** The path ships
opt-in (`MLXCEL_MLA_ABSORBED`, default off) because no MLA checkpoint was
available to satisfy the issue's pinned-oracle token-parity gate.

## Environment

| Field | Value |
|---|---|
| Hardware | Apple M1 Ultra, 20 cores (16P + 4E), 128 GB unified memory |
| OS | macOS 26.5.2 (Darwin 25.5.0) |
| Backend | Metal |
| mlxcel | 0.4.3, branch `feature/issue-907-mla-absorbed-decode`, base `649f0a52` |
| Harness | `examples/mla_absorbed_decode_bench.rs`, 64 steps, 16 warmup |
| Geometry | heads 16, kv_lora_rank 512, qk_nope 128, qk_rope 64, v_head 128, 27 layers |
| Load average | 7.7 during the sweep, from unrelated concurrent work |

**No CUDA was written for this issue at all**, deliberately, so it adds nothing
to the epic's stock of uncompiled CUDA.

## Every row proves its own path

Each measurement prints counters read from `mla::stats`, sampled and reset around
the timed region, and the harness warns on any row whose counters disagree with
its label. This exists because three separate measurements earlier in this epic
turned out to be comparing a path against itself. The counters below are quoted
from the run, not asserted.

- `decompressed` rows: `decompressed=64 absorbed_composed=0 absorbed_split_kv=0`
- `absorbed` rows: `decompressed=0 absorbed_composed=64 absorbed_split_kv=0`
- `split_kv` rows: `decompressed=0 absorbed_composed=0 absorbed_split_kv=64`

## Memory

| Quantity | Decompressed | Latent | Ratio |
|---|---|---|---|
| KV bytes / token / layer | 10240 | 1152 | **8.89x** |
| KV bytes / token, 27 layers | 276480 | 31104 | 8.89x |
| Max context under a 16 GiB KV budget | 62137 tokens | **552336 tokens** | 8.89x |

This is arithmetic on the cache layout rather than a timing, so it does not carry
the measurement caveats the latency numbers do. The fold costs 4.0 MiB per layer
in f16, 108 MiB over 27 layers, paid once at load.

## Decode latency

| batch | context | decompressed | absorbed | absorbed vs decompressed | split_kv |
|---|---|---|---|---|---|
| 1 | 4096 | 0.5449 ms | 0.6409 ms | **0.85x** | 1.1026 ms |
| 4 | 4096 | 1.0987 ms | 1.2782 ms | **0.86x** | 1.3369 ms |
| 1 | 16384 | 0.9993 ms | 0.8434 ms | **1.18x** | 1.2962 ms |
| 4 | 16384 | 3.5449 ms | 2.6618 ms | **1.33x** | 2.7411 ms |
| 1 | 32768 | 1.9613 ms | 1.0891 ms | **1.80x** | 1.5940 ms |
| 4 | 32768 | 5.8391 ms | 4.2186 ms | **1.38x** | 4.3312 ms |

### Reading

**There is a crossover, and it is where the mechanism says it should be.**
Absorption replaces reading a decompressed K and V per token with reading one
latent vector, at the cost of a larger per-step matmul against the folded
projection. At 4096 tokens the KV traffic is small enough that the extra matmul
dominates and absorption loses by 14-15%. By 16384 the traffic dominates and
absorption wins, and by 32768 at batch 1 it is 1.80x faster. Anyone enabling this
flag on short contexts should expect it to cost them; the flag is worth setting
for long-context serving, which is also where the 8.89x memory reduction matters.

**`split_kv` is slower than plain `absorbed` in every configuration measured.**
That is expected and was stated in advance: Stage 2's partial computation is
composed MLX ops rather than a fused kernel, so it pays the split without the
kernel that would make splitting pay off. It ships as a correct implementation
that exercises and validates reuse of the #898 merge kernel, not as a speed path.
The fused partial kernel the issue scopes for Stage 2 was not written.

## Reuse of the #898 merge kernel

The merge kernel from issue #898 was reused with no edit to
`paged_attention_v2_merge.cpp`, `paged_attention_v2.h`, the FFI signature, or the
C++ launcher, and **its contract held unchanged**. MLA maps onto it as
`H = num_heads`, `D = kv_lora_rank`, `M = batch`, `N = batch * chunks`, with a
request-major `o_indptr`.

One clause of that contract fails silently and is now pinned by a test: the LSE
inputs must be in log2 units, because a natural-log LSE still merges and returns
a plausible but wrong average rather than erroring.
`merge_rejects_natural_log_lse_units` asserts the natural-log variant does not
match a host f64 reference while the log2 variant does. Issue #903 reuses the same
kernel and should keep that test passing.

## What is verified, and what is not

**Verified**, on synthetic tensors with MLA geometry against a host f64
pre-absorption reference: the absorption identity in f32 and f16; causal-mask
survival through the rope score, with a positive control so the test cannot pass
on a no-op mask; the up-projection direction; `(ckv, kpe)` round-trip through a
real `KVCache` across the append boundary; trimmability for speculative decode;
split and merge at four chunk lengths and against unsplit Stage 1; and a 6-token
prefill plus 4 decode steps through `deepseek_v2`'s own `MLAAttention::forward`
matching step-for-step through `o_proj`.

**Not verified: anything on a real MLA checkpoint.** No DeepSeek-family model is
available on this host and a V3-class checkpoint does not fit in 128 GB. The
issue's pinned-oracle token-parity gate is therefore **not satisfied**, which is
the reason the flag defaults to off. The f16 tolerance is 6e-3, which is expected
drift given absorption reassociates a sum over 512 elements into one over 128, but
whether that clears a production sign-off bar is a judgement that needs a real
checkpoint.

## Premise correction

The issue states that the MLA families cache decompressed K/V. That is true for
`deepseek_v2`, `minicpm3` and `dots1`, but **not** for `deepseek_v3` and
`deepseek_v32`, which have shipped matrix absorption for some time, decomposing
`kv_b_proj` in `sanitize_weights` and caching `(kv_latent, k_pe)`. The actual gap
was the V2 family plus the transform existing twice with no shared home. This work
builds the shared module and moves V2 onto it; V3 and V3.2 are untouched.

## Open items

- Fused Stage 2 partial kernel, which is what would make split-KV a speed path.
- MiniCPM3 and dots1 wiring; converging V3/V3.2 onto the shared module.
- Real-checkpoint token parity, blocking any default-on change.
- `deepseek_v2` passes a null mask when `mask == None && l > 1`, i.e. non-causal
  prefill, mirroring a bug `deepseek_v3.rs` already carries a comment about. It
  was mirrored exactly here to preserve byte-parity with the baseline rather than
  fixed, and is worth its own issue.
- Fold memory is roughly 113 MiB for V2-Lite and about 2 GB for a V3-class model,
  reported by the harness rather than mitigated.
