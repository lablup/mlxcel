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

"""Tests for the VLM refusal check in scripts/bench_mlxlm.py (#2134).

A prompt-token count shows that an image reached the prompt, not that the
model used it. `vlm_refusal_reason` reads the generated text for the answers a
model gives when it did not, and `bench_one` records those runs as
`FAIL:image_ignored`.

Run with:
    python3 -m unittest tests/test_bench_mlxlm_refusal.py
"""

import contextlib
import io
import json
import pathlib
import subprocess
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_mlxlm  # noqa: E402

# The two answers granite-vision-3.2-2b-4bit gave in lablup/mlxcel#1683, verbatim.
REFUSAL_1683_A = (
    "I'm sorry, but I can't provide a description of the image as I can't see it."
)
REFUSAL_1683_B = (
    "I'm sorry, but I can't see any images. I can only provide text-based responses."
)
DESCRIPTION = "The image shows a solid orange square on a white background."


class VlmRefusalReasonTests(unittest.TestCase):
    def test_both_issue_1683_answers_match(self):
        self.assertEqual(bench_mlxlm.vlm_refusal_reason(REFUSAL_1683_A), "can't see")
        self.assertEqual(bench_mlxlm.vlm_refusal_reason(REFUSAL_1683_B), "can't see")

    def test_typographic_apostrophe_matches_like_a_plain_one(self):
        curly = REFUSAL_1683_A.replace("'", "’")
        self.assertNotIn("'", curly)
        self.assertEqual(bench_mlxlm.vlm_refusal_reason(curly), "can't see")

    def test_a_normal_description_does_not_match(self):
        self.assertIsNone(bench_mlxlm.vlm_refusal_reason(DESCRIPTION))

    def test_matching_ignores_case(self):
        self.assertEqual(
            bench_mlxlm.vlm_refusal_reason("I CANNOT SEE the picture."), "cannot see"
        )

    def test_the_refusal_is_found_before_trailing_filler(self):
        # EOS is suppressed in the measured pass, so the answer runs on.
        text = REFUSAL_1683_B + " The the the the." * 20
        self.assertEqual(bench_mlxlm.vlm_refusal_reason(text), "can't see")

    def test_every_listed_phrase_is_reachable(self):
        for phrase in bench_mlxlm.VLM_REFUSAL_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertEqual(phrase, phrase.lower())
                self.assertNotIn("’", phrase)
                self.assertIsNotNone(
                    bench_mlxlm.vlm_refusal_reason(f"Sorry, I {phrase} that.")
                )

    def test_the_first_listed_phrase_wins(self):
        # REFUSAL_1683_A contains both "can't provide a description" and
        # "can't see"; the tuple order decides which one is reported.
        first = bench_mlxlm.VLM_REFUSAL_PHRASES.index("can't see")
        second = bench_mlxlm.VLM_REFUSAL_PHRASES.index("can't provide a description")
        self.assertLess(first, second)

    def test_empty_text_does_not_match(self):
        self.assertIsNone(bench_mlxlm.vlm_refusal_reason(""))


def _child(result: dict) -> subprocess.CompletedProcess:
    """A finished benchmark child that printed one RESULT line."""
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout="RESULT " + json.dumps(result) + "\n", stderr=""
    )


def _bench(result: dict, *, vlm: bool):
    """Run bench_one against a canned child; return (status, fields, stderr)."""
    err = io.StringIO()
    with mock.patch.object(bench_mlxlm.subprocess, "run", return_value=_child(result)):
        with contextlib.redirect_stderr(err):
            status, fields = bench_mlxlm.bench_one(
                pathlib.Path("models/some-vlm-4bit"),
                vlm=vlm,
                vlm_image="tests/fixtures/image.png",
                max_tokens=128,
                warmup_tokens=10,
                timeout=60,
            )
    return status, fields, err.getvalue()


# A VLM row whose image expanded the prompt: 1545 measured against 60 text-only.
_EXPANDED = {
    "prompt_tokens": 1545,
    "prefill_tps": 900.0,
    "gen_tokens": 128,
    "decode_tps": 50.0,
    "text_only_prompt_tokens": 60,
}


class BenchOneImageIgnoredTests(unittest.TestCase):
    def test_a_refusal_is_recorded_as_image_ignored_with_the_phrase_on_stderr(self):
        status, fields, err = _bench(
            {**_EXPANDED, "generated_text": REFUSAL_1683_A}, vlm=True
        )
        self.assertEqual(status, "FAIL:image_ignored")
        self.assertIsNone(fields)
        self.assertIn("image_ignored", err)
        self.assertIn("can't see", err)

    def test_a_description_stays_ok(self):
        status, fields, err = _bench(
            {**_EXPANDED, "generated_text": DESCRIPTION}, vlm=True
        )
        self.assertEqual(status, "OK")
        self.assertEqual(fields["prompt_tokens"], 1545)
        self.assertEqual(err, "")

    def test_missing_or_empty_text_stays_ok(self):
        for result in (dict(_EXPANDED), {**_EXPANDED, "generated_text": ""},
                       {**_EXPANDED, "generated_text": None}):
            with self.subTest(result=result.get("generated_text", "<absent>")):
                status, _, _ = _bench(result, vlm=True)
                self.assertEqual(status, "OK")

    def test_image_not_applied_keeps_its_own_status(self):
        # The image never expanded the prompt. That is the older failure and it
        # is reported as such even when the text is also a refusal.
        status, _, err = _bench(
            {**_EXPANDED, "prompt_tokens": 60, "generated_text": REFUSAL_1683_B},
            vlm=True,
        )
        self.assertEqual(status, "FAIL:image_not_applied")
        self.assertNotIn("image_ignored", err)

    def test_text_runs_are_never_checked(self):
        status, _, err = _bench(
            {"prompt_tokens": 12, "prefill_tps": 900.0, "gen_tokens": 128,
             "decode_tps": 50.0, "generated_text": REFUSAL_1683_B},
            vlm=False,
        )
        self.assertEqual(status, "OK")
        self.assertEqual(err, "")


class VlmChildCodeTests(unittest.TestCase):
    def test_the_vlm_child_reports_the_generated_text(self):
        # The child runs under mlx-vlm, which the test host does not have, so
        # this checks the code string: it compiles, it accumulates each
        # result's text, and the RESULT dict carries the joined string.
        code = bench_mlxlm.SUBPROCESS_VLM_CODE
        compile(code, "<SUBPROCESS_VLM_CODE>", "exec")
        self.assertIn("_text_parts.append(_segment)", code)
        self.assertIn("'generated_text': ''.join(_text_parts)", code)

    def test_the_text_children_do_not_report_generated_text(self):
        self.assertNotIn("generated_text", bench_mlxlm.SUBPROCESS_TEXT_CODE)
        self.assertNotIn("generated_text", bench_mlxlm.SUBPROCESS_TEXT_VIA_VLM_CODE)


if __name__ == "__main__":
    unittest.main()
