set -euo pipefail
D=$(dirname "$0")
R=${REPO_ROOT:?set REPO_ROOT}/scripts
M=${MODELS:-models/mlx}
NEW=${SERVER_BIN:?set SERVER_BIN to the same verified binary for both arms}
echo "start $(date -u +%FT%TZ)"
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_up() {
  for ((i=0; i<240; i++)); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}
serve() {
  local label=$1 cmd=$2; shift 2
  [[ ! -e "$D/$label.txt" && ! -e "$D/server-$label.log" ]] || { echo "Refusing to overwrite $label" >&2; exit 1; }
  "$NEW" -m "$@" --port 18080 --metrics > "$D/server-$label.log" 2>&1 &
  server_pid=$!
  wait_up || { echo "Server startup failed: $label" >&2; exit 1; }
  python3 -I -B "$D/record_arm.py" "$server_pid" > "$D/$label.arm.json"
  python3 -I -B "$D/m_client.py" 18080 copy 1 1 32 > /dev/null
  python3 -I -B "$D/m_client.py" 18080 chat 1 1 32 > /dev/null
  timeout --signal=TERM --kill-after=20s 1800s bash -e -c "$cmd" > "$D/$label.txt" 2>&1
  echo "rc=0 :: $label $(date -u +%T)"
  cleanup
  sleep 5
}
Q=$M/qwen3-1.7b-4bit
# (a) chat B=1 and (b) copy B=1, server defaults, flag on vs off, 5 rounds, order alternating
for r in 1 2 3 4 5; do
  sides="off on"; [ $((r % 2)) -eq 0 ] && sides="on off"
  for s in $sides; do
    extra=""; [ $s = on ] && extra="--draft-kind prompt-lookup"
    serve ab-$s-r$r "python3 -I -B $D/m_client.py 18080 chat 1 4 256; python3 -I -B $D/m_client.py 18080 copy 1 4 256" $Q $extra
  done
done
# (c) chat-template concurrency 2, default gate, three paired rounds.
for r in 1 2 3; do
  sides="off on"; [ $((r % 2)) -eq 0 ] && sides="on off"
  for s in $sides; do
    extra=""; [ "$s" = on ] && extra="--draft-kind prompt-lookup"
    serve c-$s-r$r "python3 $R/bench_serving_concurrency.py --port 18080 --concurrency 2 --max-tokens 256 --prompt-style plain; python3 $R/bench_serving_concurrency.py --port 18080 --concurrency 2 --max-tokens 256 --prompt-style copy" "$Q" --parallel 8 $extra
  done
done
# (d) 8192-token prompts, B=1: paged and dense storage, flag on vs off, 3 rounds
for r in 1 2 3; do
  for st in paged dense; do
    sides="off on"; [ $((r % 2)) -eq 0 ] && sides="on off"
    for s in $sides; do
      extra=""; [ $s = on ] && extra="--draft-kind prompt-lookup"
      serve d-$st-$s-r$r "python3 -I -B $D/m_client.py 18080 long-copy 1 3 256 8192; python3 -I -B $D/m_client.py 18080 long-chat 1 3 256 8192" $Q --decode-storage-backend $st $extra
    done
  done
done
echo "end $(date -u +%FT%TZ)"
echo ALLDONE
