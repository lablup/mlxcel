#!/usr/bin/env python3
"""Judge a generated stream against an HF fp32 oracle with the near-tie rule
(issue #3343).

A reduced-precision run (f16 on IREE-Metal, for example) may leave the oracle's
greedy trajectory where the oracle's own top two logits are almost equal. Only
the FIRST divergence counts, because every later token is conditioned on a
different prefix. The stream passes when it is token-exact, or when the oracle's
top-1 minus top-2 logit gap at the first divergent position is below
``--tolerance`` (default 0.01).

The oracle JSON must come from ``oracle_continuation.py --top2``. The stream is
given as token ids (``--ids 1,2,3`` or ``--ids-json FILE [--ids-key KEY]``) or
as generated text (``--text`` / ``--text-file``, which needs ``--model`` for the
tokenizer and is matched against the decoded oracle prefix token by token).

Usage:
    spike/openxla/.venv/bin/python spike/openxla/oracle_continuation.py \\
        --model models/hf/mamba2-130m --out oracle.json --top2
    spike/openxla/.venv/bin/python spike/openxla/near_tie_check.py \\
        --oracle oracle.json --model models/hf/mamba2-130m --text-file out.txt

Exit status: 0 token-exact, 3 diverged at a near-tie only, 1 diverged elsewhere,
2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys

EXIT_EXACT = 0
EXIT_MISMATCH = 1
EXIT_USAGE = 2
EXIT_NEAR_TIE = 3


def first_divergence_ids(got: list[int], ref: list[int]) -> int | None:
    """Index of the first position where ``got`` leaves ``ref``, else ``None``.

    Only the common prefix is compared: a stream that stopped early (EOS or a
    shorter budget) is exact over what it produced.
    """
    for index, (a, b) in enumerate(zip(got, ref)):
        if a != b:
            return index
    return None


def first_divergence_text(text: str, ref: list[int], decode) -> tuple[int | None, int]:
    """First oracle token whose decoded prefix ``text`` does not follow.

    Returns ``(index, compared)``: ``index`` is ``None`` when ``text`` follows
    the oracle for all ``compared`` tokens it covers.
    """
    text = text.strip()
    for count in range(1, len(ref) + 1):
        prefix = decode(ref[:count]).strip()
        if text.startswith(prefix):
            continue
        if prefix.startswith(text):
            # The stream ended inside this token: nothing left to compare.
            return None, count - 1
        return count - 1, count - 1
    return None, len(ref)


def judge(oracle: dict, index: int | None, tolerance: float) -> tuple[int, str]:
    """Exit status and a one-line verdict for a first divergence at ``index``."""
    if index is None:
        return EXIT_EXACT, "TOKEN-EXACT"
    gaps = oracle.get("ref_top2_gaps")
    if gaps is None:
        return EXIT_USAGE, "oracle has no ref_top2_gaps; regenerate it with --top2"
    gap = float(gaps[index])
    runner = oracle.get("ref_top2_ids", [None] * len(gaps))[index]
    detail = (
        f"first divergence at token {index}: oracle id {oracle['ref_token_ids'][index]}, "
        f"runner-up id {runner}, top-2 logit gap {gap:.6f}"
    )
    if gap < tolerance:
        return EXIT_NEAR_TIE, f"NEAR-TIE ({detail} < {tolerance})"
    return EXIT_MISMATCH, f"MISMATCH ({detail} >= {tolerance})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", required=True, help="oracle JSON written with --top2")
    ap.add_argument("--ids", help="comma-separated generated token ids")
    ap.add_argument("--ids-json", help="JSON file holding the generated ids")
    ap.add_argument("--ids-key", default=None, help="key of the id array in --ids-json")
    ap.add_argument("--text", help="generated text (needs --model)")
    ap.add_argument("--text-file", help="file holding the generated text (needs --model)")
    ap.add_argument("--model", help="checkpoint directory, for the tokenizer")
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--label", default="stream")
    args = ap.parse_args()

    with open(args.oracle) as f:
        oracle = json.load(f)
    ref = [int(t) for t in oracle["ref_token_ids"]]

    if args.ids is not None or args.ids_json is not None:
        if args.ids is not None:
            got = [int(t) for t in args.ids.replace(" ", "").split(",") if t]
        else:
            with open(args.ids_json) as f:
                loaded = json.load(f)
            got = [int(t) for t in (loaded[args.ids_key] if args.ids_key else loaded)]
        index = first_divergence_ids(got, ref)
        compared = min(len(got), len(ref))
    elif args.text is not None or args.text_file is not None:
        if not args.model:
            ap.error("--text / --text-file need --model for the tokenizer")
        text = args.text
        if text is None:
            with open(args.text_file) as f:
                text = f.read()
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        index, compared = first_divergence_text(text, ref, tok.decode)
    else:
        ap.error("give the stream as --ids, --ids-json, --text or --text-file")

    status, verdict = judge(oracle, index, args.tolerance)
    print(f"[near-tie] {args.label}: {verdict} ({compared} tokens compared)")
    return status


if __name__ == "__main__":
    sys.exit(main())
