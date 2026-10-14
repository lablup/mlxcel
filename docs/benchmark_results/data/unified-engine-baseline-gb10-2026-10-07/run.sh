set -u
cd /home/inureyes/Development/backend.ai/mlxcel
D=docs/benchmark_results/data/unified-engine-baseline-gb10-2026-10-07
R="python3 scripts/engine_bench_rounds.py --bin target/release/mlxcel-bench-engine --hostgate"
{ echo "host: $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"; echo "commit: $(git rev-parse HEAD)"; echo "mlx pin: $(grep -E '^\s*GIT_TAG' src/lib/mlx-cpp/CMakeLists.txt | head -1 | tr -s ' ')"; echo "started: $(date -u +%FT%TZ)"; } > $D/environment.txt
for m in qwen3-1.7b-4bit llama-3.2-1b-instruct-4bit; do
  M=models/mlx/$m; t=${m%%-*}
  $R --model $M --rounds 5 --null-every-arm --arm cli="--path cli" --arm server="--path server" --out $D/baseline-$m.jsonl > $D/baseline-$m.log 2>&1
  $R --model $M --prompt-tokens 8192 --rounds 5 --arm c2048="--path cli --prefill-chunk 2048" --arm c512="--path cli --prefill-chunk 512" --out $D/chunk-cli-$m.jsonl > $D/chunk-cli-$m.log 2>&1
  $R --model $M --prompt-tokens 8192 --rounds 5 --arm c2048="--path server --prefill-chunk 2048" --arm c512="--path server --prefill-chunk 512" --out $D/chunk-server-$m.jsonl > $D/chunk-server-$m.log 2>&1
  $R --model $M --rounds 5 --arm dense="--path server --decode-storage dense" --arm paged="--path server --decode-storage paged" --out $D/storage-$m.jsonl > $D/storage-$m.log 2>&1
done
echo "finished: $(date -u +%FT%TZ)" >> $D/environment.txt
