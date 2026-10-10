set -u
D=$(dirname "$0")
R=${WT_PREFIX:-../wt-measure-}new/scripts
M=${MODELS:-models/mlx}
NEW=${WT_PREFIX:-../wt-measure-}new/target/release/mlxcel-server
echo "start $(date -u +%FT%TZ)"
{ hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; for w in base prev new; do echo "$w $(git -C ${WT_PREFIX:-../wt-measure-}$w log --oneline -1)"; done; } > $D/environment.txt
# A. #2229: penalty requests on the CLI path across three builds, plus the new build without a penalty
for m in qwen3-1.7b-4bit llama-3.2-1b-instruct-4bit; do
  python3 $R/engine_bench_rounds.py --bin $D/pick.sh --model $M/$m --prompt-tokens 256 8192 --rounds 5 --hostgate \
    --arm new-pen="--which new --path cli --repetition-penalty 1.1" \
    --arm prev-pen="--which prev --path cli --repetition-penalty 1.1" \
    --arm base-pen="--which base --path cli --repetition-penalty 1.1" \
    --arm new-plain="--which new --path cli" \
    --out $D/single-$m.jsonl > $D/single-$m.txt 2>&1
  echo "rc=$? :: single $m $(date -u +%T)"
done
wait_up() { for i in $(seq 1 240); do curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
# serve LABEL CMD SERVER-ARGS... : start the new server, warm it, run CMD, stop it
serve() {
  local label=$1 cmd=$2; shift 2
  $NEW -m "$@" --port 18080 --metrics > $D/server-$label.log 2>&1 &
  local pid=$!
  if wait_up; then
    python3 -I $D/m_client.py 18080 copy 1 1 32 > /dev/null 2>&1
    python3 -I $D/m_client.py 18080 chat 1 1 32 > /dev/null 2>&1
    eval "$cmd" > $D/$label.txt 2>&1; echo "rc=$? :: $label $(date -u +%T)"
  else echo "rc=9 :: $label server did not start"; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 5
}
Q=$M/qwen3-1.7b-4bit
# (a) chat B=1 and (b) copy B=1, server defaults, flag on vs off, 5 rounds, order alternating
for r in 1 2 3 4 5; do
  sides="off on"; [ $((r % 2)) -eq 0 ] && sides="on off"
  for s in $sides; do
    extra=""; [ $s = on ] && extra="--draft-kind prompt-lookup"
    serve ab-$s-r$r "python3 -I $D/m_client.py 18080 chat 1 4 256; python3 -I $D/m_client.py 18080 copy 1 4 256" $Q $extra
  done
done
# (c) concurrency, --parallel 8: off, on (default gate 2), on with the gate opened to 8; 3 rounds, rotating order
for r in 1 2 3; do
  case $r in 1) sides="off on mb8";; 2) sides="on mb8 off";; 3) sides="mb8 off on";; esac
  for s in $sides; do
    extra=""; [ $s = on ] && extra="--draft-kind prompt-lookup"; [ $s = mb8 ] && extra="--draft-kind prompt-lookup --prompt-lookup-max-batch 8"
    serve c-$s-r$r "python3 $R/bench_serving_concurrency.py --port 18080 --concurrency 2,4,8 --max-tokens 256 --prompt-style plain; python3 $R/bench_serving_concurrency.py --port 18080 --concurrency 2,4,8 --max-tokens 256 --prompt-style copy; for c in 2 4 8; do python3 -I $D/m_client.py 18080 chat \$c 2 256; python3 -I $D/m_client.py 18080 copy \$c 2 256; done" $Q --parallel 8 $extra
  done
done
# (d) 8192-token prompts, B=1: paged and dense storage, flag on vs off, 3 rounds
for r in 1 2 3; do
  for st in paged dense; do
    sides="off on"; [ $((r % 2)) -eq 0 ] && sides="on off"
    for s in $sides; do
      extra=""; [ $s = on ] && extra="--draft-kind prompt-lookup"
      serve d-$st-$s-r$r "python3 -I $D/m_client.py 18080 long-copy 1 3 256 8192; python3 -I $D/m_client.py 18080 long-chat 1 3 256 8192" $Q --decode-storage-backend $st $extra
    done
  done
done
echo "end $(date -u +%FT%TZ)"
echo ALLDONE
