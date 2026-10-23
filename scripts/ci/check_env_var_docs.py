#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Require every MLXCEL_* name in src/**/*.rs to be documented or allowlisted.

Rationale
---------
``docs/environment-variables.md`` is the reference page for the ``MLXCEL_*``
surface, and it is maintained by hand. Every new ``std::env::var("MLXCEL_...")``
call site is a chance for the page to drift, and before this check the drift
was found only when someone happened to look. Issue #1104 fixed six omissions
by hand; when this check was added (issue #1140), 110 names read under ``src``
were still missing from the page. This check stops the drift from coming back.

The rule
--------
Every ``MLXCEL_*`` name that appears as a complete string literal
(``"MLXCEL_FOO"``) in a ``*.rs`` file under ``src/`` must either

* appear somewhere on ``docs/environment-variables.md``, in a table row or in
  prose (a table-only parser would fail as the page's layout changes), or
* have an entry in ``ALLOWLIST`` below, with a one-line reason.

It fails in both directions of a break: a new call site with no row, and a
deleted row for a name that is still read. It also keeps the allowlist honest:
an entry for a name that ``src`` no longer contains fails, and so does an entry
for a name that the page documents, since the two records would contradict each
other.

Why there is an allowlist
-------------------------
``check_kernel_dtype_keys.py`` explains why it has no allowlist. This check
makes the opposite choice, for a reason worth recording in the same way.
Test-only, fixture-authoring and build-internal names are a real category here:
checkpoint paths for opt-in real-model tests, probe parameters, golden-freeze
switches, the handshake a test passes to the child process it re-executes, and
names the build script sets through ``cargo:rustc-env`` or ``bridge.define``.
None of these should be published as operator documentation. Excluding them by
path, or by ``#[cfg(test)]``, would hide them without recording that anyone
looked: ``MLXCEL_FREEZE_GOLDENS`` is read from ``#[cfg(test)]`` code in
``src/lib/mlxcel-xla/src/validation.rs``, a file whose name says nothing about
tests, and several ``*_tests.rs`` modules are also compiled into diagnostic
builds by a cargo feature. So the scan covers every ``*.rs`` file under
``src/``, and the allowlist holds the exceptions. Each entry is a decision that
a name is internal. Adding to it is a one-line change that a reviewer can see.

Limits
------
* Only complete string literals are seen. A name built at run time
  (``format!("MLXCEL_{x}")``, ``concat!``) is out of scope. When this check was
  written no such call site existed, but nothing enforces that.
* A literal counts whether or not the code reads it: a name in a comment
  written as a quoted string, or a value a test exports to a child process,
  counts too. That errs toward documenting too much, not too little.
* Only ``src/**/*.rs`` is scanned, as issue #1140 specifies. Names read only by
  the C++ and CUDA sources, by ``tests/``, by benches or by scripts are not
  checked. The page documents many of them anyway.
* There is no check from the page back to ``src``, which issue #1140 left
  optional. Most names on the page that ``src/**/*.rs`` does not contain are
  read by the C++ bridge, the CUDA overlays or ``tests/``, and the page also
  lists names it records as removed. A reverse check would have to know about
  all of those.

Usage
-----
    scripts/ci/check_env_var_docs.py [--root DIR]

``--root`` points the check at another tree. ``tests/test_check_env_var_docs.py``
uses it on synthetic trees. The script exits non-zero and names every offending
variable, with up to three files that contain it.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

DOC_PAGE = "docs/environment-variables.md"
SRC_DIR = "src"

# A whole string literal holding one name. The leading quote also matches the
# `"` of a raw string (`r"MLXCEL_X"`).
SRC_NAME_RE = re.compile(r'"(MLXCEL_[A-Z0-9_]+)"')
# A name anywhere on the page: in a code span, in a `NAME=value` example, or in
# prose. A name must end in a letter or digit, so a wildcard such as
# `MLXCEL_PROFILE_*` is read as `MLXCEL_PROFILE` and documents nothing that
# `src` reads.
DOC_NAME_RE = re.compile(r"(?<![A-Za-z0-9_])MLXCEL_[A-Z0-9_]*[A-Z0-9](?![A-Za-z0-9_])")

MAX_LOCATIONS = 3

_TEST_CHILD = (
    "test-only: handshake from a test to the child process it re-executes"
)
_RUSTC_ENV = (
    "build-internal: set by the mlxcel-core build script through "
    "cargo:rustc-env and read with env!/option_env!; never read from the "
    "operator's environment"
)
_BRIDGE_DEFINE = (
    "build-internal: a C++ preprocessor macro the mlxcel-core build script "
    "defines for the bridge (bridge.define), not an environment variable"
)

# Names under src that are deliberately left off docs/environment-variables.md.
# Grouped by reason. Every name must still be in src and must not be on the
# page; see the module docstring.
_ALLOWLIST_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (_BRIDGE_DEFINE, (
        "MLXCEL_BRIDGE_CUDA_BACKEND",
        "MLXCEL_BRIDGE_GPU_BACKEND",
        "MLXCEL_BRIDGE_METAL_BACKEND",
        "MLXCEL_BRIDGE_ROCM_BACKEND",
    )),
    (_RUSTC_ENV, (
        "MLXCEL_CUDA_ARCHITECTURES",
        "MLXCEL_MLX_COMMIT",
        "MLXCEL_ROCM_ARCHITECTURES",
    )),
    ("build-internal: set by the mlxcel-xla build script through "
     "cargo:rustc-env from the build-time IREE_DIST, which is the documented "
     "override", (
        "MLXCEL_XLA_IREE_DIST",
    )),
    ("fixture-authoring: makes a #[cfg(test)] test in "
     "mlxcel-xla/src/validation.rs rewrite the committed StableHLO goldens", (
        "MLXCEL_FREEZE_GOLDENS",
        "MLXCEL_FREEZE_MOE",
    )),
    ("fixture-authoring: input or output of the opt-in #[ignore] StableHLO "
     "graph-dump tests in mlxcel-xla/src/emitter/mod.rs", (
        "MLXCEL_DUMP_CONFIG",
        "MLXCEL_DUMP_DIR",
        "MLXCEL_DUMP_OUT",
        "MLXCEL_MOE_N",
    )),
    ("internal diagnostic: layer window of the xla-diagnostics #[ignore] "
     "Gemma3n all-layer trace test", (
        "MLXCEL_XLA_GEMMA3N_TRACE_LAYER_START",
    )),
    ("internal diagnostic: iree-run-module path for the diagnostics-feature "
     "CUDA Gemma3n QMV probes and their #[ignore] tests", (
        "MLXCEL_XLA_IREE_RUN_MODULE",
    )),
    ("test-only: compile target of an #[ignore] Gemma3n IREE compile test in "
     "mlxcel-xla/src/emitter/gemma3n_emit.rs", (
        "MLXCEL_XLA_IREE_TEST_TARGET",
    )),
    ("internal diagnostic: expected prompt ids of the reference-parity "
     "harness, read only by xla-diagnostics server builds", (
        "MLXCEL_XLA_REFERENCE_EXPECT_UNEXPANDED_IDS",
    )),
    (_TEST_CHILD, (
        "MLXCEL_CLI_ADAPTIVE_PREFILL_TEST_CHILD",
        "MLXCEL_CLI_PREFILL_CHUNK_TEST_CHILD",
        "MLXCEL_PREFILL_CHUNK_TEST_CHILD",
        "MLXCEL_RESTART_TEST_DIR",
        "MLXCEL_RESTART_TEST_PHASE",
    )),
    ("test-only: exported by router_webui_playwright_harness_tests.rs to the "
     "browser-test child it launches", (
        "MLXCEL_WEBUI_ROUTER_ARTIFACTS",
        "MLXCEL_WEBUI_ROUTER_CONTROL_DIR",
        "MLXCEL_WEBUI_ROUTER_KEY_FILE",
        "MLXCEL_WEBUI_ROUTER_MODELS_DIR",
        "MLXCEL_WEBUI_ROUTER_URL",
    )),
    ("test-only: sample count of a statistical sampler test", (
        "MLXCEL_SPEC_DIST_SAMPLES",
        "MLXCEL_TEST_GUMBEL_SAMPLES",
        "MLXCEL_TEST_REJECTION_SAMPLES",
    )),
    ("test-only: parameter of the Laguna DFlash developer probe in "
     "laguna_dflash_tests.rs", (
        "MLXCEL_LAGUNA_PROBE_BLOCK",
        "MLXCEL_LAGUNA_PROBE_CAPTURE",
        "MLXCEL_LAGUNA_PROBE_DRAFTER",
        "MLXCEL_LAGUNA_PROBE_MODE",
        "MLXCEL_LAGUNA_PROBE_PROMPT",
        "MLXCEL_LAGUNA_PROBE_REFERENCE",
        "MLXCEL_LAGUNA_PROBE_TARGET",
    )),
    ("test-only: parameter of the Qwen 3.5 DFlash and RoPE-attribution "
     "developer probes in qwen3_5_*_tests.rs", (
        "MLXCEL_Q35_PROBE_ACCEPTS",
        "MLXCEL_Q35_PROBE_BLOCK",
        "MLXCEL_Q35_PROBE_BLOCK_ARGMAX",
        "MLXCEL_Q35_PROBE_CAPTURE",
        "MLXCEL_Q35_PROBE_DRAFTER",
        "MLXCEL_Q35_PROBE_KEEP",
        "MLXCEL_Q35_PROBE_PROMPT",
        "MLXCEL_Q35_PROBE_REFERENCE",
        "MLXCEL_Q35_PROBE_ROUNDS",
        "MLXCEL_Q35_PROBE_ROWS",
        "MLXCEL_Q35_PROBE_STEPS",
        "MLXCEL_Q35_PROBE_TARGET",
        "MLXCEL_Q35_PROBE_WIDTHS",
        "MLXCEL_Q35_PROBE_WRONG",
        "MLXCEL_Q35_ROW_ROPE",
    )),
    ("test-only: parameter of the GLM-4.7-Flash MTP real-checkpoint gate in "
     "glm4_moe_lite_mtp_target_tests.rs", (
        "MLXCEL_TEST_GLM_MTP_BLOCK",
        "MLXCEL_TEST_GLM_MTP_GAP",
        "MLXCEL_TEST_GLM_MTP_PROMPT",
        "MLXCEL_TEST_GLM_MTP_STEPS",
        "MLXCEL_TEST_GLM_MTP_TARGET",
    )),
    ("test-only: parameter of the embedding-LoRA real-checkpoint test in "
     "adapter_embedding_tests.rs", (
        "MLXCEL_TEST_EMBED_LORA_ADAPTER",
        "MLXCEL_TEST_EMBED_LORA_DUMP",
        "MLXCEL_TEST_EMBED_LORA_MODEL",
        "MLXCEL_TEST_EMBED_LORA_PATHS",
        "MLXCEL_TEST_EMBED_LORA_PROMPT",
        "MLXCEL_TEST_EMBED_LORA_TOLERANCE",
    )),
    ("test-only: input of the Qwen3-VL vision stage-dump test in "
     "qwen3_vl_stage_dump_tests.rs", (
        "MLXCEL_QWEN3_VL_STAGE_DUMP_DIR",
        "MLXCEL_QWEN3_VL_STAGE_DUMP_IMAGE",
        "MLXCEL_QWEN3_VL_STAGE_DUMP_MODEL",
    )),
    ("test-only: checkpoint inputs of the opt-in SAM 3 BF16 parity and affine "
     "4-bit loader tests in real_tests.rs", (
        "MLXCEL_SAM3_4BIT_MODEL",
        "MLXCEL_SAM3_MODEL",
    )),
    ("fixture-authoring: pinned checkpoint, image, device or heartbeat of the "
     "Gemma3 and Molmo IREE reference gates (#869, #870), compiled for tests "
     "and xla-reference-diagnostics builds", (
        "MLXCEL_GEMMA3_FIXTURE",
        "MLXCEL_GEMMA3_IMAGE",
        "MLXCEL_GEMMA3_REVISION",
        "MLXCEL_MOLMO_FIXTURE",
        "MLXCEL_MOLMO_HEARTBEAT_SECS",
        "MLXCEL_MOLMO_IMAGE",
        "MLXCEL_MOLMO_IREE_DEVICE",
        "MLXCEL_MOLMO_REVISION",
    )),
    ("test-only: checkpoint, image and fresh output paths for the ignored "
     "SAM 3D Objects stage diagnostic in pipeline_diagnostics.rs", (
        "MLXCEL_SAM3D_OBJECTS_DIAGNOSTIC_DIR",
        "MLXCEL_SAM3D_OBJECTS_INPUT_DIR",
        "MLXCEL_SAM3D_OBJECTS_MODEL",
    )),
    ("test-only: checkpoint, image or device input of an opt-in real-model "
     "test in a *_tests.rs or tests.rs module", (
        "MLXCEL_FLOOR_MODEL",
        "MLXCEL_GEMMA3_MODEL",
        "MLXCEL_LLAMA3_MODEL",
        "MLXCEL_QWEN3_MODEL",
        "MLXCEL_TEST_GRANITE_VISION_DIR",
        "MLXCEL_TEST_JAGLE_VL_DIR",
        "MLXCEL_TEST_LLMJP_4VL_9B_DIR",
        "MLXCEL_TEST_MOLMO_IMAGE",
        "MLXCEL_TEST_MOLMO_IREE_DEVICE",
        "MLXCEL_TEST_MOLMO_MODEL",
        "MLXCEL_TEST_OLMOE_DIR",
        "MLXCEL_TEST_QWEN25VL_MLX_DIR",
        "MLXCEL_TEST_QWEN25VL_RAW_DIR",
        "MLXCEL_TEST_QWEN35_MTP_DRAFTER",
    )),
)


def build_allowlist(
    groups: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, str]:
    allowlist: dict[str, str] = {}
    for reason, names in groups:
        for name in names:
            if name in allowlist:
                raise ValueError(f"{name} is allowlisted twice")
            allowlist[name] = reason
    return allowlist


ALLOWLIST = build_allowlist(_ALLOWLIST_GROUPS)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def rust_files(src: pathlib.Path) -> list[pathlib.Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames.sort()
        found.extend(
            pathlib.Path(dirpath) / name
            for name in sorted(filenames)
            if name.endswith(".rs")
        )
    return found


def src_names(root: pathlib.Path) -> tuple[dict[str, list[str]], int]:
    """Each name found under src, with its `path:line` locations."""
    names: dict[str, list[str]] = {}
    files = rust_files(root / SRC_DIR)
    for path in files:
        rel = path.relative_to(root).as_posix()
        text = path.read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in SRC_NAME_RE.finditer(line):
                names.setdefault(match.group(1), []).append(f"{rel}:{lineno}")
    return names, len(files)


def doc_names(text: str) -> set[str]:
    return set(DOC_NAME_RE.findall(text))


def check(
    root: pathlib.Path, allowlist: dict[str, str]
) -> tuple[list[str], list[str], str]:
    """Return (undocumented failures, allowlist failures, counts line)."""
    read, scanned = src_names(root)
    documented = doc_names((root / DOC_PAGE).read_text(errors="replace"))

    undocumented = []
    for name in sorted(set(read) - documented - set(allowlist)):
        locations = read[name]
        shown = ", ".join(locations[:MAX_LOCATIONS])
        more = len(locations) - MAX_LOCATIONS
        if more > 0:
            shown += f" (+{more} more)"
        undocumented.append(f"{name}: in {shown}")

    stale = []
    if not read:
        stale.append(
            f"no MLXCEL_* string literal found in {scanned} *.rs files under "
            f"{SRC_DIR}/, so the check would pass over nothing"
        )
    for name in sorted(allowlist):
        if name not in read:
            stale.append(
                f"{name}: allowlisted but no longer in {SRC_DIR}/. Remove the "
                "ALLOWLIST entry."
            )
        elif name in documented:
            stale.append(
                f"{name}: allowlisted and also on {DOC_PAGE}. Keep one: drop "
                "the ALLOWLIST entry if the name is operator-facing, or the "
                "page's mention if it is internal."
            )

    counts = (
        f"{len(read)} MLXCEL_* names in {scanned} *.rs files under {SRC_DIR}/, "
        f"{len(set(read) & documented)} documented on "
        f"{DOC_PAGE}, {len(allowlist)} allowlisted"
    )
    return undocumented, stale, counts


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
    for required in (root / SRC_DIR, root / DOC_PAGE):
        if not required.exists():
            print(f"env-var-docs: FAIL\n  {required} does not exist")
            return 1

    undocumented, stale, counts = check(root, ALLOWLIST)
    if undocumented or stale:
        print("env-var-docs: FAIL")
        print(f"  {counts}.")
        if undocumented:
            print()
            print(f"Read in {SRC_DIR}/ but not on {DOC_PAGE}:")
            for failure in undocumented:
                print(f"  {failure}")
            print()
            print(
                "For each name, either:\n"
                f"  1. document it on {DOC_PAGE}, in the section that fits\n"
                "     (values, default, effect, which component reads it), or\n"
                "  2. if it is test-only, fixture-authoring, build-internal or an\n"
                "     internal diagnostic that operators should never set, add it\n"
                "     to ALLOWLIST in scripts/ci/check_env_var_docs.py with a\n"
                "     one-line reason. See issue #1140."
            )
        if stale:
            print()
            print("ALLOWLIST is out of date:")
            for failure in stale:
                print(f"  {failure}")
        return 1

    print(f"env-var-docs: OK, {counts}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
