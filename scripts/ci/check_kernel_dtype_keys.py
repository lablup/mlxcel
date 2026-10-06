#!/usr/bin/env python3
"""Require every CUDA or HIP JIT kernel launch to key its cache on the input dtypes.

Rationale
---------
MLX generates a custom kernel's buffer parameter types from the *runtime* dtypes
of its inputs, but the backends disagree on whether those dtypes belong in the
JIT cache key.

* Metal (``mlx/backend/common/metal_kernel.cpp``) appends one
  ``get_type_string(arr.dtype())`` per input to the kernel name, with the
  comment "The generated source depends on the dtypes of the inputs and outputs
  ... Include them in the kernel name so that a given name always maps to the
  same source."
* CUDA (``mlx/backend/cuda/custom_kernel.cpp``) builds its name as
  ``"custom_kernel_" + name + template_arguments_hash(template_args)``, which
  covers only the template args. Before upstream 9f5f7931
  (ml-explore/mlx#4273) ``cu::get_jit_module`` memoised the compiled module
  under exactly that name; since then, and at the current pin (81ba1c6a),
  ``CustomKernel::eval_gpu`` passes ``name_ + "_" + hex(hash(source_))`` as the
  module name, so the generated source, dtypes included, is in the key.
* ROCm (``fast::hip_kernel``, vendored at
  ``src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/custom_kernel.cpp``) builds
  the same name the same way and, since #2149 (``LOCAL_FIXES.md``), keys the
  module in ``rocm::get_jit_module`` by device index, that name and a hash of
  the generated source, as CUDA does. Before that it keyed by the name alone.

With a name-only module key, a launch whose ``template_args`` are all ints
hashes to one name for every input dtype. Whichever dtype compiles first wins
for the life of the process, and every later call at a different dtype reads
its buffers through the wrong pointer type and returns numbers unrelated to
its inputs. Nothing throws.

That produced issues #1053 (a sparse f16 decode off by a relative error of ~1.0)
and #1054 on the CUDA backend of an older pin, and it silently affected the
sampler, whose ``gumbel_max_sample_accepts`` admits float32, float16 and
bfloat16 at one ``NumSplits``.

``template_arguments_hash`` *does* hash a ``Dtype`` template arg, so naming the
input dtypes in ``template_args`` restores the discrimination under a name-only
key. This check enforces that.

Why the check stays now that every backend keys on the source: it is defense
in depth, not the fix. A re-vendored ROCm fork, an MLX pin bump or a new
backend can bring a name-only key back, and nothing else would notice until a
model returned wrong numbers. The explicit keys add no compiles (the source
already differs per dtype, so the source hash splits those modules anyway), so
no launch drops them: some are also referenced by their kernel bodies as type
aliases, and the rest are cheap insurance against exactly that regression.

The rule
--------
Every ``std::vector<std::pair<std::string, TemplateArg>>`` initialiser in a file
that also calls ``cuda_kernel(`` or ``hip_kernel(`` must name at least one input
dtype, either inline (``{"KVType", k_pool.dtype()}``) or through a local bound
from a ``.dtype()`` earlier in the same file (``auto T = x.inner.dtype();`` then
``{"T", T}``).

There is deliberately no allowlist. Metal-only launchers are out of scope
because Metal's key already carries the dtypes, and they are excluded by the
absence of a CUDA or HIP launch in the file rather than by a hand-maintained
list that could go stale. Adding a CUDA or HIP port to a Metal-only launcher
therefore brings it under the check automatically, which is the point: the
failure mode this guards against is a *new* call site repeating the omission.

Scope, and why it is pinned (#1875)
-----------------------------------
Scoping by a token in the file is correct but invisible: a refactor that moves
the launch somewhere else takes the file's ``template_args`` out of scope with
it, and a check over zero files still prints OK. So the scope is fail-closed in
three ways.

1. Every C, C++, CUDA, HIP and Objective-C++ source *and header* in the
   repository is scanned, not a fixed pair of directories and not ``*.cpp``
   only, so a launch that moves into a header or a new directory is still
   seen. In a git work tree the file list is ``git ls-files`` (tracked plus
   untracked, minus ``.gitignore``d), so build output and local reference
   checkouts such as ``references/mlx``, which has launchers of its own, stay
   out without a hand-kept prune list; any other tree is walked with only
   top-level build and dependency directories skipped (``PRUNED_DIRS``). A
   file counts as a launcher only when the token appears outside comments, and
   the vendored MLX definitions of the entry points themselves
   (``CustomKernelFunction hip_kernel(``) are not calls.
2. The success line reports the in-scope count next to the scanned count, so a
   drop in what is actually checked shows up in the CI log.
3. The in-scope set is pinned in ``EXPECTED_IN_SCOPE``. A pinned file that
   leaves it fails the check, and the message says whether it was deleted or
   merely stopped launching (a launch moved into a helper leaves its
   ``template_args`` behind, unchecked). A new launcher file that is not pinned
   fails too, so the pin never goes stale, and an empty scope always fails.
   Updating the pin is a one-line, reviewable change.

Limits: the rule reads only *named* ``std::vector<...TemplateArg>``
initialisers, so a launch that passes ``template_args`` inline (the #1804 ROCm
fault probe passes ``{}``, at a fixed float32; the #2149 JIT-key probe passes
``{}`` on purpose, so that only the backend's source-hash key separates its
dtypes) is in scope but has nothing to check. A launch reached without a
direct call in the file (a function pointer, or a macro defined elsewhere) does
not put that file in scope; the pin catches a pinned file that switches to such
a form, not a new file that starts with one.

Usage
-----
    scripts/ci/check_kernel_dtype_keys.py [--root DIR]

``--root`` points the check at another tree; the companion test
``check_kernel_dtype_keys_test.sh`` uses it on mutated copies. Exits non-zero
and names every offending initialiser and every scope change.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

# Every file that launches a CUDA or HIP JIT kernel today, relative to the
# repository root. Change it deliberately, in the same change that adds,
# removes or moves a launch.
EXPECTED_IN_SCOPE = frozenset(
    {
        "src/lib/mlx-cpp/turbo/fused_norm.cpp",
        "src/lib/mlx-cpp/turbo/fused_rope_append.cpp",
        "src/lib/mlx-cpp/turbo/paged_attention.cpp",
        "src/lib/mlx-cpp/turbo/paged_attention_v2.cpp",
        "src/lib/mlx-cpp/turbo/paged_attention_v2_merge.cpp",
        "src/lib/mlx-cpp/turbo/sampling.cpp",
        "src/lib/mlx-cpp/turbo/sampling_rejection.cpp",
        # The #1804 ROCm fault probe (`rocm_fault_probe_array`) and the #2149
        # JIT-key probe (`rocm_jit_key_probe`): `fast::hip_kernel` launches in
        # a file with no CUDA launch, so it was out of scope while only
        # `cuda_kernel(` counted.
        "src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp",
        "src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp",
    }
)

SOURCE_SUFFIXES = frozenset(
    {
        ".c", ".cc", ".cpp", ".cxx", ".cu",
        ".h", ".hh", ".hpp", ".hxx", ".cuh", ".inc", ".ipp",
        ".hip", ".m", ".mm",
    }
)
# Outside a git work tree only (the companion test's throwaway copies): the
# top-level directories not descended into, along with top-level hidden ones.
# Symlinked directories (`models/`) are never followed. Inside a work tree
# `.gitignore` decides instead.
PRUNED_DIRS = frozenset({"target", "node_modules", "build", "site", "__pycache__"})

# `fast::cuda_kernel(`, `mlx::core::fast::hip_kernel(`, `cuda_kernel (`. The
# leading `\b` keeps `precompiled_cuda_kernel(` out: it loads a prebuilt
# binary rather than JIT-compiling a source under a hashed name.
LAUNCH_RE = re.compile(r"\b(cuda|hip)_kernel\s*\(")
# The definition or declaration of the entry point itself, in the vendored MLX
# tree: `CustomKernelFunction hip_kernel(`, `MLX_API CustomKernelFunction ...`.
DEFINITION_PREFIX_RE = re.compile(r"CustomKernelFunction\s+(?:\w+::)*$")
# C/C++ string literals (raw ones included, since kernel sources are raw
# strings that contain `//`), character literals and comments. Only comments
# are blanked; literals are matched so a `//` inside one is not taken for a
# comment.
LEXEME_RE = re.compile(
    r'(?:u8|u|U|L)?R"(?P<delim>[^()\\\s"]{0,16})\(.*?\)(?P=delim)"'
    r'|"(?:\\.|[^"\\\n])*"'
    # `(?<!...)` keeps a digit separator (`1'000`, `0xFF'FF`) from opening a
    # character literal that would swallow the code after it.
    r"|(?<![0-9A-Fa-f])'(?:\\.|[^'\\\n])*'"
    r"|(?P<comment>//[^\n]*|/\*.*?\*/)",
    re.S,
)

TEMPLATE_ARGS_RE = re.compile(
    r"std::vector<std::pair<std::string,\s*(?:mlx::core::fast::)?TemplateArg>>"
    r"\s*(\w+)\s*=\s*\{(.*?)\n\s*\};",
    re.S,
)
# `auto T = x.inner.dtype();`, `const auto input_type = a.dtype();`
DTYPE_BINDING_RE = re.compile(r"\b(?:auto|Dtype)\s+(\w+)\s*=\s*[^;]*\.dtype\(\)")
# The value half of a `{"Name", value}` entry.
ENTRY_RE = re.compile(r'\{\s*"(\w+)"\s*,\s*([^}]+?)\s*\}')


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def without_comments(src: str) -> str:
    """Blank out comments, keeping newlines so line structure holds."""

    def blank(match: re.Match[str]) -> str:
        if match.group("comment") is None:
            return match.group(0)
        return re.sub(r"[^\n]", " ", match.group(0))

    return LEXEME_RE.sub(blank, src)


def launches_jit_kernel(src: str) -> bool:
    """True when the file calls `cuda_kernel(` or `hip_kernel(` in code."""
    code = without_comments(src)
    for match in LAUNCH_RE.finditer(code):
        line_start = code.rfind("\n", 0, match.start()) + 1
        if DEFINITION_PREFIX_RE.search(code[line_start : match.start()]):
            continue
        return True
    return False


def git_files(root: pathlib.Path) -> list[pathlib.Path] | None:
    """Tracked and untracked, non-ignored files when `root` is a work tree's top."""
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if pathlib.Path(top).resolve() != root:
            return None
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [root / rel for rel in sorted(set(listed.split("\0"))) if rel]


def walked_files(root: pathlib.Path) -> list[pathlib.Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if pathlib.Path(dirpath) == root:
            dirnames[:] = [
                d for d in dirnames if d not in PRUNED_DIRS and not d.startswith(".")
            ]
        dirnames.sort()
        found.extend(pathlib.Path(dirpath) / name for name in sorted(filenames))
    return found


def source_files(root: pathlib.Path) -> list[pathlib.Path]:
    candidates = git_files(root)
    if candidates is None:
        candidates = walked_files(root)
    # `is_file` drops tracked files deleted from the work tree, and symlinks
    # to directories.
    return [
        p for p in candidates if p.suffix in SOURCE_SUFFIXES and p.is_file()
    ]


def check_file(path: pathlib.Path, root: pathlib.Path, src: str) -> list[str]:
    dtype_locals = set(DTYPE_BINDING_RE.findall(src))
    failures = []
    for match in TEMPLATE_ARGS_RE.finditer(src):
        var, body = match.group(1), match.group(2)
        line = src[: match.start()].count("\n") + 1
        keys = []
        keyed_on_dtype = False
        for name, value in ENTRY_RE.findall(body):
            keys.append(name)
            if ".dtype()" in value or value.strip() in dtype_locals:
                keyed_on_dtype = True
        if not keyed_on_dtype:
            rel = path.relative_to(root).as_posix()
            failures.append(
                f"{rel}:{line}: `{var}` names no input dtype; keys are "
                f"{keys or '[]'}"
            )
    return failures


def scope_failures(root: pathlib.Path, in_scope: set[str]) -> list[str]:
    failures = []
    if not in_scope:
        failures.append(
            "no scanned file launches a CUDA or HIP JIT kernel, so the check "
            "would pass over nothing"
        )
    for rel in sorted(EXPECTED_IN_SCOPE - in_scope):
        if not (root / rel).exists():
            failures.append(
                f"{rel}: pinned in EXPECTED_IN_SCOPE but no longer exists. If it "
                "was deleted or renamed on purpose, update the pin in the same "
                "change."
            )
        else:
            failures.append(
                f"{rel}: pinned in EXPECTED_IN_SCOPE but no longer calls "
                "`cuda_kernel(` or `hip_kernel(`. If its launch moved into a "
                "helper, the `template_args` it still builds are no longer "
                "checked: keep the launch here, or move the `template_args` "
                "with it, then update the pin."
            )
    for rel in sorted(in_scope - EXPECTED_IN_SCOPE):
        failures.append(
            f"{rel}: launches a CUDA or HIP JIT kernel but is not pinned in "
            "EXPECTED_IN_SCOPE. Add it, so that a later move out of scope is "
            "caught."
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=repo_root(),
        help="tree to check (default: this repository)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"kernel-dtype-keys: FAIL\n  --root {root} is not a directory")
        return 1

    failures = []
    scanned = 0
    in_scope: set[str] = set()
    for path in source_files(root):
        scanned += 1
        src = path.read_text(errors="replace")
        if not launches_jit_kernel(src):
            continue  # Metal-only (or no) launcher: Metal's key carries dtypes.
        in_scope.add(path.relative_to(root).as_posix())
        failures.extend(check_file(path, root, src))
    scope = scope_failures(root, in_scope)
    counts = (
        f"{len(in_scope)} in scope (launching cuda_kernel or hip_kernel) of "
        f"{scanned} source files scanned"
    )

    if failures or scope:
        print("kernel-dtype-keys: FAIL")
        print(f"  {counts}.")
        for failure in failures:
            print(f"  {failure}")
        if failures:
            print()
            print(
                "Every CUDA or HIP JIT launch must key its cache on the input\n"
                "dtypes, or a second dtype at the same geometry silently reuses\n"
                "the first one's compiled module. Add the varying inputs' dtypes\n"
                'to `template_args`, e.g. `{"KVType", k_pool.dtype()}`. They may\n'
                "stay unreferenced by the kernel body; their job is the cache\n"
                "key. See issues #1053 and #1054."
            )
        if scope:
            print()
            print("kernel-dtype-keys: in-scope set changed (EXPECTED_IN_SCOPE, #1875):")
            for failure in scope:
                print(f"  {failure}")
        return 1

    print(f"kernel-dtype-keys: OK, {counts}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
