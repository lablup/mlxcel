set -u
D=$(dirname "$0")
M=${MODELS:-models/mlx}
NEW=${WT_PREFIX:-../wt-measure-}new/target/release/mlxcel-server
echo "start $(date -u +%FT%TZ)"
wait_up() { for i in $(seq 1 240); do curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
serve() {
  local label=$1 cmd=$2; shift 2
  $NEW -m "$@" --port 18080 > $D/server-$label.log 2>&1 &
  local pid=$!
  if wait_up; then
    python3 -I $D/m_client.py 18080 copy 1 1 32 > /dev/null 2>&1
    python3 -I $D/m_client.py 18080 chat 1 1 32 > /dev/null 2>&1
    eval "$cmd" > $D/$label.txt 2>&1; echo "rc=$? :: $label $(date -u +%T)"
  else echo "rc=9 :: $label server did not start"; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 5
}
Q=$M/qwen3-1.7b-4bit
# (a2) realistic chat through the chat template, B=1, server defaults, 8 questions x 5 rounds, alternating order
for r in 1 2 3 4 5; do
  sides="off on"; [ $((r % 2)) -eq 0 ] && sides="on off"
  for s in $sides; do
    extra=""; [ $s = on ] && extra="--draft-kind prompt-lookup"
    serve a2-$s-r$r "python3 -I $D/m_chat.py 18080 256" $Q $extra
  done
done
# (d2) paged storage, B=1, prompts of about 1024, 2048 and 4096 tokens, flag on vs off, 3 rounds
for r in 1 2 3; do
  sides="off on"; [ $((r % 2)) -eq 0 ] && sides="on off"
  for s in $sides; do
    extra=""; [ $s = on ] && extra="--draft-kind prompt-lookup"
    serve d2-$s-r$r "for n in 1024 2048 4096; do python3 -I $D/m_client.py 18080 long-copy 1 3 256 \$n; python3 -I $D/m_client.py 18080 long-chat 1 3 256 \$n; done" $Q --decode-storage-backend paged $extra
  done
done
echo "end $(date -u +%FT%TZ)"
echo ALLDONE
