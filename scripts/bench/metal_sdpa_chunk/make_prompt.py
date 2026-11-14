#!/usr/bin/env python3
"""Print a deterministic long prompt of roughly N tokens (issue #3033 runs).

The body is a ledger of numbered records with distinct values, followed by a
question whose answer is one of them, so a greedy completion is short, checkable
and sensitive to attention over the whole prompt. Token counts are approximate
(about 27 tokens per record on the Qwen 3.5 tokenizer, digits split); `mlxcel generate`
prints the real prompt length.
"""

import sys


def main() -> None:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 28000
    records = max(8, target // 27)
    lines = [
        f"Record {i}: item {i} is stored in bin {(i * 7919) % 10007} with weight {(i * 104729) % 997}."
        for i in range(records)
    ]
    probe = records * 3 // 5
    lines.append("")
    lines.append(
        f"Question: in which bin is item {probe} stored, and what is its weight? "
        "Answer in one sentence, then list the bins of items "
        f"{probe - 1} and {probe + 1}."
    )
    sys.stdout.write("\n".join(lines))


if __name__ == "__main__":
    main()
