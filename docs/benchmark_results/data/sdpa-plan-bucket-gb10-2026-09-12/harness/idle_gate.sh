#!/usr/bin/env bash
# Block until the 1-minute load average is under 0.6 in two samples 20 s apart.
n=0
while :; do
  l=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$l" 'BEGIN{exit !(l < 0.6)}'; then n=$((n+1)); else n=0; fi
  [ "$n" -ge 2 ] && break
  echo "waiting for idle host: load1=$l"; sleep 20
done
echo "IDLE load=$(cat /proc/loadavg) $(date -Is)"
