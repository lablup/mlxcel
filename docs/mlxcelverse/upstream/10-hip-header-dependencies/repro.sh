#!/usr/bin/env bash
# Reproduction for "fix(rocm): track header dependencies of HIP objects".
#
# Usage: repro.sh <mlx-source-dir> <configured-build-dir>
#
# Builds once, appends a comment to mlx/backend/rocm/kernel_utils.hpp, builds
# again and counts the "Compiling HIP source" lines, then restores the header
# and builds a third time. Without the fix the edited build recompiles no HIP
# object; with it, it recompiles every object whose depfile names the header
# (all but a few), and the restore build recompiles the same set.
set -euo pipefail
src="$1"
build="$2"
hdr="$src/mlx/backend/rocm/kernel_utils.hpp"

cmake --build "$build" --target mlx >/dev/null
cp "$hdr" "$hdr.orig"
echo "// header-dependency probe" >>"$hdr"
edited="$(cmake --build "$build" --target mlx 2>&1 | grep -c 'Compiling HIP source' || true)"
mv "$hdr.orig" "$hdr"
restored="$(cmake --build "$build" --target mlx 2>&1 | grep -c 'Compiling HIP source' || true)"
unchanged="$(cmake --build "$build" --target mlx 2>&1 | grep -c 'Compiling HIP source' || true)"
echo "HIP objects recompiled after editing kernel_utils.hpp: $edited"
echo "after restoring it: $restored"
echo "with no change: $unchanged"
[ "$edited" -gt 0 ] && [ "$unchanged" -eq 0 ]
