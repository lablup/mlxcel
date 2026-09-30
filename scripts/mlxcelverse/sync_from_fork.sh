#!/usr/bin/env bash
# Sync the mlxcelverse ROCm overlay to a NripeshN/mlx rocm-support commit
# (issue #1813).
#
# Usage:
#   scripts/mlxcelverse/sync_from_fork.sh [<fork-commit-or-branch>] [options]
#
# With no commit the head of the branch recorded in patches-rocm/UPSTREAM is
# used. Options are those of `rocm_overlay.py sync`, most usefully:
#   --check        sync into a scratch copy and require it to equal the
#                  committed overlay byte for byte (no change is written)
#   --out DIR      write the synced overlay to DIR instead of in place
#   --no-fetch     use only objects already in the git cache
#   --git-dir DIR  git cache (default ~/.cache/mlxcel/mlxcelverse/mlx.git)
#
# What it does: backend files are 3-way merged (old fork commit, overlay, new
# fork commit), so every local fix in LOCAL_FIXES.md is carried as the
# difference between the overlay and the old fork commit, and local-only files
# such as hadamard.hip and fft.hip are kept. Each MLX core file is 3-way merged
# from "pin + old fork delta" to "pin + new fork delta", which is the fork's
# change to that file against its own merge base, applied on the pin. UPSTREAM
# is updated; conflicts are left in the files with markers and make the exit
# status 1. The full procedure is docs/mlxcelverse/rocm-overlay.md.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  commit="$1"
  shift
  set -- --fork-commit "$commit" "$@"
fi
exec python3 "$here/rocm_overlay.py" sync "$@"
