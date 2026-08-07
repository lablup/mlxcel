#!/usr/bin/env python3
"""Require every CUDA JIT kernel launch to key its cache on the input dtypes.

Rationale
---------
MLX generates a custom kernel's buffer parameter types from the *runtime* dtypes
of its inputs, but the two backends disagree on whether those dtypes belong in
the JIT cache key.

* Metal (``mlx/backend/common/metal_kernel.cpp``) appends one
  ``get_type_string(arr.dtype())`` per input to the kernel name, with the
  comment "The generated source depends on the dtypes of the inputs and outputs
  ... Include them in the kernel name so that a given name always maps to the
  same source."
* CUDA (``mlx/backend/cuda/custom_kernel.cpp``) builds its name as
  ``"custom_kernel_" + name + template_arguments_hash(template_args)`` and stops
  there. ``cu::get_jit_module`` then memoises the compiled module under exactly
  that name in a process-global map and invokes the source builder only on a
  cache miss.

So on CUDA a launch whose ``template_args`` are all ints hashes to one name for
every input dtype. Whichever dtype compiles first wins for the life of the
process, and every later call at a different dtype reads its buffers through the
wrong pointer type and returns numbers unrelated to its inputs. Nothing throws.

That produced issues #1053 (a sparse f16 decode off by a relative error of ~1.0)
and #1054, and it silently affected the sampler, whose
``gumbel_max_sample_accepts`` admits float32, float16 and bfloat16 at one
``NumSplits``.

``template_arguments_hash`` *does* hash a ``Dtype`` template arg, so naming the
input dtypes in ``template_args`` restores the discrimination. This check
enforces that.

The rule
--------
Every ``std::vector<std::pair<std::string, TemplateArg>>`` initialiser in a file
that also contains a ``cuda_kernel(`` call must name at least one input dtype,
either inline (``{"KVType", k_pool.dtype()}``) or through a local bound from a
``.dtype()`` earlier in the same file (``auto T = x.inner.dtype();`` then
``{"T", T}``).

There is deliberately no allowlist. Metal-only launchers are out of scope
because Metal's key already carries the dtypes, and they are excluded by the
absence of ``cuda_kernel(`` in the file rather than by a hand-maintained list
that could go stale. Adding a CUDA port to a Metal-only launcher therefore
brings it under the check automatically, which is the point: the failure mode
this guards against is a *new* call site repeating the omission.

Usage
-----
    scripts/ci/check_kernel_dtype_keys.py

Exits non-zero and names every offending initialiser.
"""
import pathlib
import re
import sys

# Directories holding the in-tree kernel launchers.
SEARCH_DIRS = ("src/lib/mlx-cpp/turbo", "src/lib/mlxcel-core/cpp")

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


def check_file(path: pathlib.Path) -> list[str]:
    src = path.read_text()
    if "cuda_kernel(" not in src:
        return []  # Metal-only launcher: Metal's cache key already carries dtypes.

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
            rel = path.relative_to(repo_root())
            failures.append(
                f"{rel}:{line}: `{var}` names no input dtype; keys are "
                f"{keys or '[]'}"
            )
    return failures


def main() -> int:
    root = repo_root()
    failures = []
    scanned = 0
    for directory in SEARCH_DIRS:
        for path in sorted((root / directory).glob("*.cpp")):
            scanned += 1
            failures.extend(check_file(path))

    if failures:
        print("kernel-dtype-keys: FAIL")
        for failure in failures:
            print(f"  {failure}")
        print()
        print(
            "Every CUDA JIT launch must key its cache on the input dtypes, or a\n"
            "second dtype at the same geometry silently reuses the first one's\n"
            "compiled module. Add the varying inputs' dtypes to `template_args`,\n"
            'e.g. `{"KVType", k_pool.dtype()}`. They may stay unreferenced by the\n'
            "kernel body; their job is the cache key. See issues #1053 and #1054."
        )
        return 1

    print(f"kernel-dtype-keys: OK — {scanned} source files scanned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
