#!/usr/bin/env bash
# Print the pinned MLX upstream commit on stdout.
#
# The pin has a single source of truth: the GIT_TAG argument of the
# FetchContent_Declare block naming the MLX repository in
# src/lib/mlx-cpp/CMakeLists.txt. src/lib/mlxcel-core/build.rs parses that same
# line in Rust (src/lib/mlxcel-core/build_support/mlx_pin.rs, unit-tested by
# src/lib/mlxcel-mlx-pin). This script exists so release.yml's "Validate MLX
# build cache" steps can compare the _deps/.mlx-build-commit markers against the
# pin without keeping a second copy of the literal. Issue #1047: the workflow's
# copy had already drifted a bump behind the CMake file, which silently turned
# every release build into a full MLX rebuild.
#
# Exits non-zero with a message on stderr unless the pin resolves to exactly one
# 40-character lowercase hex SHA. Callers must treat a failure as fatal and must
# never fall back to an empty value: an empty pin makes every marker comparison
# fail, which purges every build cache instead of validating it.
#
# MLX_PIN_CMAKE_FILE overrides the file to read; it exists for the negative-path
# checks that exercise this script against synthetic inputs.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cmake_file="${MLX_PIN_CMAKE_FILE:-$repo_root/src/lib/mlx-cpp/CMakeLists.txt}"

if [ ! -f "$cmake_file" ]; then
  echo "mlx_pinned_commit: $cmake_file not found" >&2
  exit 1
fi

# Comments are stripped first so a commented-out declaration or a GIT_TAG
# mentioned in prose cannot be read as the live pin, and the scan is scoped to
# the declaration whose GIT_REPOSITORY names the MLX repository so a second
# dependency's GIT_TAG can never be picked up. Intentionally free of regex
# interval syntax, which is not portable across the awk implementations on the
# GitHub-hosted Linux and self-hosted macOS runners; the SHA shape is validated
# in the shell below instead.
#
# Count the matching declarations before reading any tag out of them. Counting
# tags alone is not the same check and is not sufficient: with two blocks naming
# the MLX repository where only the second carries a GIT_TAG, the tag count is 1
# and this script would return that second block's SHA without complaint, while
# the Rust parser rejects the same file as ambiguous. Anything the two parsers
# disagree about is exactly the divergence #1047 exists to remove, so the shape
# they accept has to be the same one.
mlx_declarations="$(
  awk '
    { line = $0; sub(/#.*/, "", line); $0 = line }
    /FetchContent_Declare/ { in_block = 1; is_mlx = 0 }
    in_block && !is_mlx && index($0, "ml-explore/mlx") > 0 { is_mlx = 1; found++ }
    in_block && index($0, ")") > 0 { in_block = 0; is_mlx = 0 }
    END { print found + 0 }
  ' "$cmake_file"
)"

if [ "$mlx_declarations" -ne 1 ]; then
  echo "mlx_pinned_commit: expected exactly one FetchContent_Declare block naming ml-explore/mlx in $cmake_file, found $mlx_declarations. Leave exactly one; the pin is its GIT_TAG argument." >&2
  exit 1
fi

commits="$(
  awk '
    { line = $0; sub(/#.*/, "", line); $0 = line }
    /FetchContent_Declare/ { in_block = 1; is_mlx = 0 }
    in_block && index($0, "ml-explore/mlx") > 0 { is_mlx = 1 }
    in_block && is_mlx && $1 == "GIT_TAG" {
      value = $2
      gsub(/^"/, "", value)
      gsub(/[")]+$/, "", value)
      print value
    }
    in_block && index($0, ")") > 0 { in_block = 0; is_mlx = 0 }
  ' "$cmake_file"
)"

count="$(printf '%s' "$commits" | grep -c . || true)"
if [ "$count" -ne 1 ]; then
  echo "mlx_pinned_commit: expected exactly one GIT_TAG in the FetchContent_Declare block naming ml-explore/mlx in $cmake_file, found $count" >&2
  exit 1
fi

case "$commits" in
  *[!0-9a-f]*)
    echo "mlx_pinned_commit: GIT_TAG in $cmake_file is '$commits', which is not lowercase hex. Pin a full commit SHA, not a branch or tag name." >&2
    exit 1
    ;;
esac

if [ "${#commits}" -ne 40 ]; then
  echo "mlx_pinned_commit: GIT_TAG in $cmake_file is '$commits', which is ${#commits} characters. Pin a full 40-character commit SHA." >&2
  exit 1
fi

printf '%s\n' "$commits"
