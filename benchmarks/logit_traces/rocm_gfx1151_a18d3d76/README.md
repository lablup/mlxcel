# granite-4.0-h-tiny `w1` against an f32 reference (lablup/mlxcel#2154)

Traces and per-op numbers behind [the granite `w1` section of the 2026-09-30 correctness page](../../../docs/benchmark_results/rocm-correctness-gfx1151-2026-09-30.md#the-granite-w1-perplexity-gap). They answer one question: is the +4.345% `w1` perplexity of ROCm against Metal (`../rocm_gfx1151_c5fe9a16/` against `../metal_m5_d1128266/`) a ROCm defect. `METADATA.txt` has the host, commit, binary hash, checkpoint and the meaning of every tag; `RUNS.txt` has every run; `SHA256SUMS` covers every file here.

## What is here

| File (prefix `rocm_gfx1151_a18d3d76_granite-4.0-h-tiny_`) | What it is |
|---|---|
| `default_w1.tsv`, `default_w8.tsv`, `default_w256.tsv` | `logit_trace` at `a18d3d76`, the widths of `../rocm_gfx1151_c5fe9a16/`. The `w1` data rows are byte-identical to that directory's; `w8` and `w256` are not (see the page) |
| `ssmkernel0_w1.tsv` | `w1` with `MLXCEL_SSM_KERNEL=0`; byte-identical data rows to `default_w1.tsv`, because a `w1` chunk never has SSM state, so neither tag reaches the HIP SSM update kernel |
| `default_w1x1024.tsv` | `w1` over 1024 chunks; its first 128 rows are byte-identical to `default_w1.tsv` |
| `f32ref_w1x1024.tsv` | the reference: the same 1024 single-token forwards with f32 activations on the GPU (weights as stored, logits not rounded to bf16) |
| `f32refcpu_w1x3.tsv` | the f32 forward on the CPU stream, 3 chunks; it matches `f32ref` to 0.0007 nats per position, so the f32 reference does not depend on the device |
| `cpubf16_w1x1.tsv` | `MLXCEL_DEVICE=cpu logit_trace`, one chunk before a 600 s timeout. Not usable as a reference: its top-1 logit is 14.50 where the GPU gives 18.875 and f32 gives 18.63. The pre-#2248 accumulation bug: the CPU scalar quantized matmul kept its running sum in bf16 |
| `cpubf16_post2248_w1x1.tsv` | the same command rerun after #2248 (the scalar quantized matmul accumulating in f32, commit not in this ledger's `mlxcel_commit`): top-1 18.875000 in 125 s elapsed, no timeout, byte-identical top-1 logit to `default_w1.tsv` chunk 0. The remaining 0.24 gap to `f32refcpu` is the bf16-vs-f32 head error the GPU shows too, not a CPU defect |
| `f32mix-<family>_w1x1024.tsv` | the f32 forward with one op family in bf16: `resid` (the residual stream rounded to bf16 after the embedding and every residual add), `norm`, `mamba`, `attn`, `moe`, `shared`, `head`, and `all` |
| `ops_bf16_vs_f32.tsv` | per stage, first 128 chunks: relative L2 error of the GPU bf16 stage output against the f32 stage on the same inputs (mean and max over tokens and layers), and the largest absolute error. The last line counts MoE routing sets that differ between bf16 and f32 router logits |
| `granitemoehybrid_probe_2154.rs` | the probe that wrote the `f32*` and `ops` files |

## Reproducing

```bash
export OPENSSL_INCLUDE_DIR=/usr/include OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu
cargo build --release --features rocm --example logit_trace
T=./target/release/examples/logit_trace; M=models/mlx/granite-4.0-h-tiny-4bit; C=tests/fixtures/wikitext2_excerpt.txt
$T $M $C 1 128 8 0 > default_w1.tsv
MLXCEL_SSM_KERNEL=0 $T $M $C 1 128 8 0 > ssmkernel0_w1.tsv
$T $M $C 1 1024 8 0 > default_w1x1024.tsv
```

The probe reads private fields of `GraniteMoeHybridModel`, so it has to be compiled as a child module of `src/models/granitemoehybrid.rs`. Copy it to `src/models/granitemoehybrid_probe_2154.rs`, append

```rust
#[cfg(test)]
#[path = "granitemoehybrid_probe_2154.rs"]
mod probe_2154;
```

to `src/models/granitemoehybrid.rs`, and run it (one GPU process; take the host GPU lock on a shared host):

```bash
cargo test --profile test-fast --features rocm --lib --no-run
B=$(ls -t target/test-fast/deps/mlxcel-* | grep -v '\.d$' | head -1)
export PROBE_MODEL=$M PROBE_CORPUS=$C PROBE_OUT=out
PROBE_N=1024 PROBE_ARMS=f32gpu $B granitemoehybrid::probe_2154 --ignored --nocapture
PROBE_N=1024 PROBE_OPS_N=128 PROBE_ARMS=ops $B granitemoehybrid::probe_2154 --ignored --nocapture
PROBE_N=1024 PROBE_ARMS=none PROBE_MIX='all;resid;norm;mamba;attn;moe;shared;head' $B granitemoehybrid::probe_2154 --ignored --nocapture
```

The arms `f32cpu` and `bf16cpu` run on the CPU stream at roughly 400 s per chunk on this host, which is why only three `f32cpu` chunks exist.
