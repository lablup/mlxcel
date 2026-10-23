#!/usr/bin/env bash
set -u
cd /home/devops/Development/wt-epic-2382-issue-2431 || exit 2
printf '%s  %s\n' '191c47c7c8e4dff2b88991f0004125705907ee444280be5ad6bfe51bfd434fa6' /tmp/epic-2382-rfdetr-reference-v2-launch.sha256 | sha256sum -c - || exit 2
printf '%s  %s\n' '8239f47d52e58a2f3bb16d8828b9e3d2432693b0c132eb6955d0e6cfa463def9' /tmp/epic-2382-rfdetr-reference-suite-v2.sh | sha256sum -c - || exit 2
sha256sum -c /tmp/epic-2382-rfdetr-reference-v2-launch.sha256 || exit 2
for path in /tmp/epic-2382-rfdetr-reference-v2.guard-status /tmp/epic-2382-rfdetr-reference-v2.wrapper-status; do
  if [[ -e "$path" ]]; then echo "Existing status $path" >&2; exit 2; fi
done
date -u +%FT%TZ
scripts/rocm_gpu_guard.sh --hold --max-wait 300 -- bash /tmp/epic-2382-rfdetr-reference-suite-v2.sh
guard_rc=$?
printf '%s\n' "$guard_rc" > /tmp/epic-2382-rfdetr-reference-v2.guard-status
printf '%s\n' "$guard_rc" > /tmp/epic-2382-rfdetr-reference-v2.wrapper-status
date -u +%FT%TZ
exit "$guard_rc"
