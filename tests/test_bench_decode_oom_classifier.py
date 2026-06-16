# Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

"""Tests for the is_oom_failure() classifier in scripts/bench_decode.sh.

Run with:
    python3 -m unittest tests/test_bench_decode_oom_classifier.py
"""

import pathlib
import shlex
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_decode.sh"

# Awk program that extracts the is_oom_failure function body from bench_decode.sh.
# Matches the opening line, collects lines until the closing brace at column 0,
# then stops so unrelated functions are not included.
_AWK_EXTRACT = r"/^is_oom_failure\(\) \{/{f=1} f{print} f && /^\}$/{exit}"


def _is_oom(rc: int, err_text: str) -> bool:
    """Extract is_oom_failure from bench_decode.sh and call it with (rc, err_text).

    Returns True when the classifier reports OOM (exit 0), False otherwise.
    """
    cmd = (
        "eval \"$(awk '"
        + _AWK_EXTRACT
        + "' "
        + shlex.quote(str(SCRIPT))
        + ')\"; is_oom_failure '
        + str(rc)
        + " "
        + shlex.quote(err_text)
    )
    result = subprocess.run(["bash", "-c", cmd], capture_output=True)
    return result.returncode == 0


class IsOomFailureTests(unittest.TestCase):
    # --- positive cases: should be classified as OOM ---

    def test_sigkill_alone_is_oom(self) -> None:
        # rc=137 (SIGKILL) is the OS OOM-killer; exit code alone is sufficient.
        self.assertTrue(_is_oom(137, ""))

    def test_sigkill_with_oom_text_is_oom(self) -> None:
        # rc=137 should be OOM regardless of what text is present.
        self.assertTrue(
            _is_oom(
                137,
                "[metal::malloc] Attempting to allocate 8589934592 bytes "
                "which is greater than the maximum allowed buffer size of 4294967295 bytes.",
            )
        )

    def test_metal_malloc_pattern_is_oom(self) -> None:
        # rc=1 from a cxx-propagated MLX/Metal allocator exception.
        self.assertTrue(_is_oom(1, "[metal::malloc] Unable to allocate requested buffer"))

    def test_greater_than_maximum_allowed_buffer_size_is_oom(self) -> None:
        # Full Metal allocator message that appears in real OOM runs.
        self.assertTrue(
            _is_oom(
                1,
                "greater than the maximum allowed buffer size of 4294967295 bytes",
            )
        )

    def test_memory_allocation_of_n_bytes_failed_is_oom(self) -> None:
        # Rust std abort message: "memory allocation of N bytes failed".
        self.assertTrue(_is_oom(1, "memory allocation of 8192 bytes failed"))

    def test_bad_alloc_is_oom(self) -> None:
        self.assertTrue(_is_oom(134, "std::bad_alloc"))

    def test_out_of_memory_text_is_oom(self) -> None:
        self.assertTrue(_is_oom(1, "out of memory"))

    def test_unable_to_allocate_is_oom(self) -> None:
        self.assertTrue(_is_oom(1, "unable to allocate 2GB for weight tensor"))

    # --- negative cases: should NOT be classified as OOM ---

    def test_timeout_exit_is_not_oom(self) -> None:
        # rc=124 is timeout(1) expiry; a slow model is not out of memory.
        self.assertFalse(
            _is_oom(
                124,
                "[metal::malloc] greater than the maximum allowed buffer size",
            )
        )

    def test_non_oom_exit_code_with_oom_text_is_not_oom(self) -> None:
        # rc=1 without a signal-level OOM needs matching text; generic exit 1
        # with OOM text but rc not in (137, 134, ...) is still classified by
        # the text branch. This case uses an exit code of 1 with generic error.
        self.assertFalse(_is_oom(1, "benchmark failed with error code 42"))

    def test_sigkill_false_positive_guard_zoom_room(self) -> None:
        # rc=137 is always OOM regardless of text; this tests the text branch
        # with a bare "oom" substring (zoom, room) using a non-signal rc.
        self.assertFalse(_is_oom(1, "zoom error encountered in room 42"))

    def test_exceeds_limit_is_not_oom(self) -> None:
        # "exceeds limit" appears in KV-cache frame-check messages and must
        # not be mistaken for an allocator failure.
        self.assertFalse(_is_oom(1, "KV-cache block count exceeds limit for this context"))

    def test_greater_than_maximum_without_allowed_buffer_size_is_not_oom(self) -> None:
        # "greater than the maximum 4096" does not match the allocator pattern
        # "greater than the maximum allowed buffer size".
        self.assertFalse(_is_oom(1, "sequence length is greater than the maximum 4096"))


if __name__ == "__main__":
    unittest.main()
