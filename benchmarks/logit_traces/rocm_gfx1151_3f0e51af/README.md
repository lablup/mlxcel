# ROCm logit traces with the HIP fused add-RMSNorm and RoPE + append ports (lablup/mlxcel#2063)

Teacher-forced traces from `examples/logit_trace` on the Radeon 8060S (`gfx1151`) after `fused_add_rms_norm` and `fused_rope_qk_append` gained HIP ports. Both fusions ship off, so each model is traced with them off (`fusedoff`) and on (`addrms`, `addrms-rope`); in every pair the data rows are byte-identical (`SHA256SUMS` shows it), so turning the fusions on changes no logit on ROCm. `METADATA.txt` records the host, versions, commit and binary hash; `RUNS.txt` the exit status and row count of every run. The comparisons are in `docs/benchmark_results/rocm-fused-norm-rope-gfx1151-2026-10-05.md`.

```bash
cargo build --release --features rocm --example logit_trace
T=./target/release/examples/logit_trace
C=tests/fixtures/wikitext2_excerpt.txt
P=rocm_gfx1151_3f0e51af
for w in "w1:0:1 128 8 0" "w8:0:8 80 8 512" "w1ctx512:512:1 128 8 512"; do
  tag=${w%%:*}; rest=${w#*:}; start=${rest%%:*}; args=${rest#*:}
  for m in "llama-3.1-8b-instruct:Meta-Llama-3.1-8B-Instruct-4bit:addrms:1:0" "qwen2.5-7b-instruct:Qwen2.5-7B-Instruct-4bit:addrms-rope:1:1"; do
    IFS=: read name dir on add rope <<<"$m"
    MLXCEL_TRACE_START_TOKEN=$start MLXCEL_FUSED_ADD_RMSNORM=0 MLXCEL_FUSED_ROPE_APPEND=0 $T models/mlx/$dir $C $args > ${P}_${name}_fusedoff_$tag.tsv
    MLXCEL_TRACE_START_TOKEN=$start MLXCEL_FUSED_ADD_RMSNORM=$add MLXCEL_FUSED_ROPE_APPEND=$rope $T models/mlx/$dir $C $args > ${P}_${name}_${on}_$tag.tsv
  done
done
```
