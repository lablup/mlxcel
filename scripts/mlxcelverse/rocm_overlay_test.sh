#!/usr/bin/env bash
# Offline coverage for scripts/mlxcelverse/rocm_overlay.py (issue #1813).
#
# Builds a small synthetic world in a temporary directory: one git repository
# holding an "upstream" history (merge base B1, pins P1 and P2) and a "fork"
# history (F1 and variants), plus an overlay directory laid out like
# src/lib/mlx-cpp/patches-rocm/. Then it checks that each subcommand accepts
# the good cases and rejects the bad ones, including the review risk #1813 is
# about: an overlay core file that quietly drops part of an upstream change is
# reported even though its +/- line counts look plausible. Needs git, bash and
# python3; no network, no build. Runs in a few seconds.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tool="$here/rocm_overlay.py"
tmp="$(mktemp -d -t mlxcelverse-test.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT

pass=0
fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { pass=$((pass + 1)); }

# expect <status> <grep-pattern-or-empty> -- command...
expect() {
  local want="$1" pattern="$2"
  shift 3
  local out rc=0
  out="$("$@" 2>&1)" || rc=$?
  if [ "$rc" -ne "$want" ]; then
    echo "$out" >&2
    fail "expected exit $want, got $rc: $*"
  fi
  if [ -n "$pattern" ] && ! grep -q -- "$pattern" <<<"$out"; then
    echo "$out" >&2
    fail "output of '$*' lacks: $pattern"
  fi
  ok
}

# subst <file> <regex> <replacement>: in-place multiline substitution. Not
# `sed -i`, whose syntax differs between GNU and BSD sed (this runs in
# `make verify` on macOS too); fails when the pattern does not match.
subst() {
  python3 - "$@" <<'PY'
import re, sys
path, pattern, repl = sys.argv[1:4]
with open(path) as f:
    content = f.read()
if not re.search(pattern, content, flags=re.M):
    sys.exit(f"subst: no match for {pattern!r} in {path}")
with open(path, "w") as f:
    f.write(re.sub(pattern, repl, content, flags=re.M))
PY
}

export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@example.com GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@example.com
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null

repo="$tmp/mlx"
git init -q -b main "$repo"
g() { git -C "$repo" "$@"; }
lines() { for i in $(seq 1 "$1"); do echo "$2 line $i"; done; }
commit() { g add -A && g commit -q -m "$1" && g rev-parse HEAD; }

# ---- upstream: B1, P1, P2 --------------------------------------------------
mkdir -p "$repo/mlx" "$repo/python"
lines 30 core >"$repo/mlx/core.cpp"
lines 10 util >"$repo/mlx/util.cpp"
lines 5 cmake >"$repo/CMakeLists.txt"
echo "py" >"$repo/python/x.py"
B1="$(commit B1)"
subst "$repo/mlx/core.cpp" '^core line 25$' 'core line 25 upstream change 1'
P1="$(commit P1)"
subst "$repo/mlx/core.cpp" '^core line 28$' 'core line 28 upstream change 2'
P2="$(commit P2)"
g update-ref refs/remotes/upstream/main "$P2"

# ---- fork F1 from B1 ------------------------------------------------------
g checkout -q -b fork "$B1"
subst "$repo/mlx/core.cpp" '^core line 5$' 'core line 5\n#ifdef MLX_USE_ROCM\nrocm hook\n#endif'
subst "$repo/CMakeLists.txt" '^cmake line 2$' 'cmake line 2\noption(MLX_BUILD_ROCM)'
mkdir -p "$repo/mlx/backend/rocm" "$repo/mlx/backend/metal"
lines 20 a >"$repo/mlx/backend/rocm/a.hip"
lines 20 b >"$repo/mlx/backend/rocm/b.hip"
echo "metal custom kernel, fork variant" >"$repo/mlx/backend/metal/custom.cpp"
echo "rocm bindings" >"$repo/python/rocm.py"
F1="$(commit F1)"

# F2: fork changes that do not collide with the local fixes.
subst "$repo/mlx/backend/rocm/a.hip" '^a line 15$' 'a line 15 fork change'
subst "$repo/mlx/backend/rocm/b.hip" '^b line 10$' 'b line 10 fork change'
lines 5 d >"$repo/mlx/backend/rocm/d.hip"
subst "$repo/mlx/core.cpp" '^rocm hook$' 'rocm hook v2'
F2="$(commit F2)"

# F3: the fork rewrites the line a local fix changed.
g checkout -q -b fork3 "$F1"
subst "$repo/mlx/backend/rocm/a.hip" '^a line 3$' 'a line 3 fork rewrite'
F3="$(commit F3)"

# F4: the fork merges upstream past the pin (P2 while the pin is P1).
g checkout -q -b fork4 "$F1"
g merge -q --no-edit "$P2" >/dev/null
F4="$(g rev-parse HEAD)"
g checkout -q main

# F5: a hostile fork tree. Git stores and lists a `..` entry verbatim, so
# mlx/backend/rocm/../../../../escaped would land next to the sync output
# directory. Built with mktree because no index can hold such a path.
evil="$(echo pwned | g hash-object -w --stdin)"
t="$(printf '100644 blob %s\tescaped\n' "$evil" | g mktree)"
for _ in 1 2 3; do t="$(printf '040000 tree %s\t..\n' "$t" | g mktree)"; done
t="$( { g ls-tree "$F1:mlx/backend/rocm"; printf '040000 tree %s\t..\n' "$t"; } | g mktree)"
t="$( { g ls-tree "$F1:mlx/backend" | grep -v '	rocm$'; printf '040000 tree %s\trocm\n' "$t"; } | g mktree)"
t="$( { g ls-tree "$F1:mlx" | grep -v '	backend$'; printf '040000 tree %s\tbackend\n' "$t"; } | g mktree)"
t="$( { g ls-tree "$F1" | grep -v '	mlx$'; printf '040000 tree %s\tmlx\n' "$t"; } | g mktree)"
F5="$(echo F5 | g commit-tree "$t" -p "$F1")"

# ---- the overlay as committed at pin P1, fork F1 ----------------------------
ov="$tmp/overlay"
mkdir -p "$ov/mlx/backend/rocm"
# pin + fork delta, the same 3-way merge the tool reconstructs with.
merge_onto_pin() {
  g show "$P1:$1" >"$ov/$1"
  g show "$B1:$1" >"$tmp/base"
  g show "$F1:$1" >"$tmp/fork"
  git merge-file -q "$ov/$1" "$tmp/base" "$tmp/fork" || fail "merging $1"
}
merge_onto_pin CMakeLists.txt
merge_onto_pin mlx/core.cpp
subst "$ov/mlx/core.cpp" '^core line 15$' 'core line 15 local fix'
g show "$F1:mlx/backend/rocm/a.hip" >"$ov/mlx/backend/rocm/a.hip"
subst "$ov/mlx/backend/rocm/a.hip" '^a line 3$' 'a line 3 local fix'
g show "$F1:mlx/backend/rocm/b.hip" >"$ov/mlx/backend/rocm/b.hip"
lines 4 c >"$ov/mlx/backend/rocm/c.hip"
cat >"$ov/UPSTREAM" <<EOF
source: https://example.invalid/fork
branch: fork
commit: $F1
fork_upstream_merge_base: $B1
retargeted_to_mlx_pin: $P1
EOF
cat >"$ov/README.md" <<'EOF'
- `mlx/backend/rocm/**`: the ROCm backend (3 files).
- 2 MLX core files (`CMakeLists.txt`, `mlx/core.cpp`).
EOF
cat >"$ov/LOCAL_FIXES.md" <<'EOF'
1. **Local core fix.** `mlx/core.cpp` line 15.
2. **Local kernel fix.** `a.hip` line 3.
3. **Local-only kernel.** `c.hip`.

- `mlx/backend/metal/custom.cpp` from the fork is not carried.
EOF

cmake_file() {
  cat >"$1" <<EOF
FetchContent_Declare(
  mlx
  GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
  GIT_TAG $2)
EOF
}
cmake_file "$tmp/pin-p1.cmake" "$P1"
cmake_file "$tmp/pin-p2.cmake" "$P2"
export MLX_PIN_CMAKE_FILE="$tmp/pin-p1.cmake"

t() { python3 "$tool" --overlay-dir "$1" "${@:2}"; }
tg() { t "$1" "$2" --git-dir "$repo/.git" --no-fetch "${@:3}"; }
fresh() { cp -R "$ov" "$tmp/$1"; echo "$tmp/$1"; }

# ---- records and drift on the good overlay ----------------------------------
expect 1 "CORE_RESIDUAL.diff: missing" -- t "$ov" records
expect 0 "wrote" -- tg "$ov" drift --write
expect 1 "has a residual but no note" -- t "$ov" records
subst "$ov/CORE_RESIDUAL.diff" '^# note mlx/core.cpp: TODO.*' '# note mlx/core.cpp: LOCAL_FIXES.md item 1.'
grep -q 'core line 15 local fix' "$ov/CORE_RESIDUAL.diff" || fail "residual lacks the local fix"
if grep -q 'rocm hook' "$ov/CORE_RESIDUAL.diff"; then fail "the fork's own hook leaked into the residual"; fi
if grep -q 'upstream change 1' "$ov/CORE_RESIDUAL.diff"; then fail "the pin's own change leaked into the residual"; fi
expect 0 "records OK" -- t "$ov" records
expect 0 "drift OK" -- tg "$ov" drift

# ---- sync -----------------------------------------------------------------------
expect 0 "byte for byte" -- tg "$ov" sync --fork-commit "$F1" --check
o="$tmp/sync-f2"
expect 0 "synced" -- tg "$ov" sync --fork-commit "$F2" --out "$o"
grep -q '^a line 3 local fix$' "$o/mlx/backend/rocm/a.hip" || fail "sync lost the local fix"
grep -q '^a line 15 fork change$' "$o/mlx/backend/rocm/a.hip" || fail "sync lost the fork change"
grep -q '^b line 10 fork change$' "$o/mlx/backend/rocm/b.hip" || fail "sync missed a fork-only change"
[ -f "$o/mlx/backend/rocm/c.hip" ] || fail "sync dropped a local-only file"
[ -f "$o/mlx/backend/rocm/d.hip" ] || fail "sync missed a new fork file"
grep -q '^rocm hook v2$' "$o/mlx/core.cpp" || fail "sync missed the fork's core hook change"
grep -q '^core line 15 local fix$' "$o/mlx/core.cpp" || fail "sync lost the local core fix"
grep -q '^core line 25 upstream change 1$' "$o/mlx/core.cpp" || fail "sync lost the pin's change"
grep -q "^commit: $F2$" "$o/UPSTREAM" || fail "sync did not update UPSTREAM"
expect 1 "CONFLICT: mlx/backend/rocm/a.hip" -- tg "$ov" sync --fork-commit "$F3" --out "$tmp/sync-f3"
grep -q '^<<<<<<< overlay$' "$tmp/sync-f3/mlx/backend/rocm/a.hip" || fail "conflict markers missing"
expect 2 "does not contain" -- tg "$ov" sync --fork-commit "$F4" --out "$tmp/sync-f4"
expect 2 "refusing unsafe path" -- tg "$ov" sync --fork-commit "$F5" --out "$tmp/sync-f5"
expect 2 "refusing unsafe path" -- tg "$ov" export-tree --mlx-commit "$P1" --rocm-from "fork:$F5" --dest "$tmp/export-f5"
if [ -n "$(find "$tmp" -name escaped -print -quit)" ]; then fail "a hostile tree path escaped the output directory"; fi

# ---- a pin bump ---------------------------------------------------------------
MLX_PIN_CMAKE_FILE="$tmp/pin-p2.cmake" expect 1 "A pin bump has to retarget" -- t "$ov" records
r="$(fresh retarget)"
MLX_PIN_CMAKE_FILE="$tmp/pin-p2.cmake" expect 0 "retargeted" -- tg "$r" retarget
grep -q '^core line 28 upstream change 2$' "$r/mlx/core.cpp" || fail "retarget missed the new upstream change"
grep -q '^core line 15 local fix$' "$r/mlx/core.cpp" || fail "retarget lost the local fix"
MLX_PIN_CMAKE_FILE="$tmp/pin-p2.cmake" expect 1 "generated for pin" -- t "$r" records
MLX_PIN_CMAKE_FILE="$tmp/pin-p2.cmake" expect 1 "header is stale" -- tg "$r" drift
MLX_PIN_CMAKE_FILE="$tmp/pin-p2.cmake" expect 0 "drift OK" -- tg "$r" drift --write
MLX_PIN_CMAKE_FILE="$tmp/pin-p2.cmake" expect 0 "records OK" -- t "$r" records

# ---- an overlay that was already wrong: half of an upstream change missing ----
w="$(fresh wrong)"
subst "$w/mlx/core.cpp" '^core line 25 upstream change 1$' 'core line 25'
expect 1 "changed since the residual was generated" -- t "$w" records
expect 1 "core residual differs" -- tg "$w" drift
drift_out="$(tg "$w" drift 2>/dev/null || true)"
grep -q '^+-core line 25 upstream change 1$' <<<"$drift_out" || fail "drift did not show the dropped upstream line"
ok

# ---- a backend change LOCAL_FIXES.md does not name ------------------------------
u="$(fresh unnamed)"
subst "$u/mlx/backend/rocm/b.hip" '^b line 2$' 'b line 2 unrecorded'
expect 1 "mlx/backend/rocm/b.hip differs from the fork" -- tg "$u" drift
# A local-only backend file whose name only appears as the tail of another
# path (`mlx/core.cpp` in LOCAL_FIXES.md) is not named by it.
v="$(fresh tail-name)"
echo "local core helper" >"$v/mlx/backend/rocm/core.cpp"
subst "$v/README.md" '\(3 files\)' '(4 files)'
expect 1 "mlx/backend/rocm/core.cpp differs from the fork" -- tg "$v" drift
# Finder's .DS_Store is neither an overlay file nor a stray record.
d="$(fresh finder)"
echo x >"$d/.DS_Store"
echo x >"$d/mlx/.DS_Store"
expect 0 "records OK" -- t "$d" records

# ---- records negatives ----------------------------------------------------------
n="$(fresh numbering)"
subst "$n/LOCAL_FIXES.md" '^3\. \*\*' '4. **'
expect 1 "numbered" -- t "$n" records
c="$(fresh count)"
subst "$c/README.md" '\(3 files\)' '(106 files)'
expect 1 "says the backend has 106 files" -- t "$c" records
s="$(fresh stray)"
echo x >"$s/NOTES.txt"
expect 1 "not a record" -- t "$s" records

# ---- api-report -----------------------------------------------------------------
cat >"$tmp/build.log" <<EOF
[ 10%] Building CXX object foo.o
$tmp/src/mlx/backend/rocm/compiled.cpp:120:8: error: cannot decompose 4 elements into 3 names
$tmp/src/mlx/backend/rocm/compiled.cpp:120:8: error: cannot decompose 4 elements into 3 names
EOF
printf '0000000000000000 T mlx::core::Add::eval_gpu(std::vector<mlx::core::array> const&, mlx::core::array&)\n' >"$tmp/defined.txt"
printf '                 U mlx::core::Add::eval_gpu(std::vector<mlx::core::array> const&, mlx::core::array&)\n                 U mlx::core::SearchSorted::eval_gpu(std::vector<mlx::core::array> const&, mlx::core::array&)\n                 U hipMalloc\n' >"$tmp/undefined.txt"
expect 1 "compile errors: 1" -- t "$ov" api-report --build-log "$tmp/build.log" --defined "$tmp/defined.txt" --undefined "$tmp/undefined.txt" --source-dir "$tmp/src"
report_out="$(t "$ov" api-report --build-log "$tmp/build.log" --defined "$tmp/defined.txt" --undefined "$tmp/undefined.txt" --source-dir "$tmp/src" || true)"
grep -q '^  mlx::core::SearchSorted::eval_gpu' <<<"$report_out" || fail "api-report missed the undefined eval_gpu"
grep -q '^  mlx/backend/rocm/compiled.cpp:120:8: cannot decompose' <<<"$report_out" || fail "api-report did not relativize or dedupe the error"
if grep -qE 'Add::eval_gpu|hipMalloc' <<<"$report_out"; then fail "api-report listed a defined or non-MLX symbol"; fi
ok
: >"$tmp/clean.log"
expect 0 "undefined mlx::core symbols: 0" -- t "$ov" api-report --build-log "$tmp/clean.log" --defined "$tmp/defined.txt" --undefined "$tmp/defined.txt"

echo "rocm_overlay_test: OK ($pass checks)"
