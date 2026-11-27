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

"""Reduce the M2(c) test-binary capture to the compact CSV committed for #2834.

Usage: oxide_fused_norm_m2c_summarize.py <capture.log> <out.csv>

Rows are (record, name, value): `meta` for the header lines run.sh writes
(command, commit, binary digest, start time, exit status), `test` for each
test's outcome, `result` for each field of the `RESULT oxide-adapter-matrix`
line, and `summary` for the libtest totals.
"""

from __future__ import annotations

import csv
import re
import sys


def main(capture: str, out: str) -> int:
    rows: list[tuple[str, str, str]] = []
    result_seen = False
    # A test that prints under --nocapture puts its outcome on a later line.
    pending: str | None = None
    with open(capture, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            meta = re.match(r"^# (\w+): (.*)$", line)
            if meta:
                rows.append(("meta", meta.group(1), meta.group(2)))
            test = re.match(r"^test (\S+) \.\.\. ?(.*)$", line)
            if test:
                outcome = re.search(r"\b(ok|FAILED|ignored)$", test.group(2))
                if outcome:
                    rows.append(("test", test.group(1), outcome.group(1)))
                    pending = None
                else:
                    pending = test.group(1)
            elif pending and line in ("ok", "FAILED", "ignored"):
                rows.append(("test", pending, line))
                pending = None
            result = re.search(r"RESULT oxide-adapter-matrix (.*)$", line)
            if result:
                result_seen = True
                for field in result.group(1).split():
                    key, _, value = field.partition("=")
                    rows.append(("result", key, value))
            summary = re.match(
                r"^test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored; "
                r"(\d+) measured; (\d+) filtered out; finished in ([\d.]+)s$",
                line,
            )
            if summary:
                for key, value in zip(
                    ("outcome", "passed", "failed", "ignored", "measured", "filtered_out", "seconds"),
                    summary.groups(),
                ):
                    rows.append(("summary", key, value))
    if not result_seen:
        print(f"oxide_fused_norm_m2c_summarize: no RESULT oxide-adapter-matrix line in {capture}", file=sys.stderr)
        return 3
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(("record", "name", "value"))
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
