#!/usr/bin/env bash
set -u
cd /home/devops/Development/wt-epic-2382-issue-2431 || exit 2
for kind in base seg; do
  log="/tmp/epic-2382-rfdetr-${kind}-reference-v2.log"
  status="/tmp/epic-2382-rfdetr-${kind}-reference-v2.child-status"
  if [[ -e "$log" || -e "$status" ]]; then
    echo "Refusing existing attempt evidence: $kind" >&2
    exit 2
  fi
  date -u +%FT%TZ
  bash /tmp/epic-2382-rfdetr-reference-v2.sh "$kind" > "$log" 2>&1
  child_rc=$?
  printf '%s\n' "$child_rc" > "$status"
  cat "$log"
  if (( child_rc != 0 )); then exit "$child_rc"; fi
done
