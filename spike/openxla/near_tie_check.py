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
Text cannot show token boundaries, so a text divergence that falls inside a
word (the stream may have emitted one longer token where the oracle emitted
two) is reported as ambiguous with exit 2; pass the ids instead.
``--mlxcel-cli`` says the text is the captured stdout of ``mlxcel generate``:
the banner lines, the echoed prompt and the trailing statistics are dropped.

Usage:
    spike/openxla/.venv/bin/python spike/openxla/oracle_continuation.py \\
        --model models/hf/mamba2-130m --out oracle.json --top2
    spike/openxla/.venv/bin/python spike/openxla/near_tie_check.py \\
        --oracle oracle.json --model models/hf/mamba2-130m --text-file out.txt

Exit status: 0 token-exact, 3 diverged at a near-tie only, 1 diverged elsewhere,
2 usage error, an empty stream, or an ambiguous text divergence.
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


def strip_mlxcel_cli(stdout: str, prompt: str) -> str:
    """The generated text inside captured ``mlxcel generate`` stdout.

    The CLI prints banner lines, echoes the prompt, may print an
    ``OpenXLA context capacity`` notice on the prompt's line, then streams the
    continuation and ends with a ``[Generated N tokens ...]`` line.
    """
    at = stdout.find(prompt)
    if at < 0:
        raise ValueError("the prompt is not echoed in the captured CLI output")
    rest = stdout[at + len(prompt) :]
    if rest.startswith("OpenXLA context capacity"):
        rest = rest.split("\n", 1)[1] if "\n" in rest else ""
    end = rest.rfind("\n[Generated ")
    return rest if end < 0 else rest[:end]


def first_divergence_text(text: str, ref: list[int], decode) -> tuple[int | None, int]:
    """First oracle token whose decoded prefix ``text`` does not follow.

    Returns ``(index, compared)``: ``index`` is ``None`` when ``text`` follows
    the oracle for all ``compared`` tokens it covers.

    Leading whitespace is ignored on both sides and trailing whitespace only on
    ``text`` (a capture may add or drop a final newline). The oracle prefix
    keeps its trailing whitespace, so a whitespace token such as a newline
    still counts as a position that can diverge.
    """
    text = text.strip()
    for count in range(1, len(ref) + 1):
        prefix = decode(ref[:count]).lstrip()
        if text.startswith(prefix):
            continue
        if prefix.startswith(text):
            # The stream ended inside this token: nothing left to compare.
            return None, count - 1
        index = count - 1
        if index > 0:
            # If the matched oracle prefix ends mid-word and the stream goes on
            # with more word characters, the stream token may have extended
            # the previous oracle token (oracle "the" + " cat", stream
            # "there"), so the real divergence can be earlier than `index`.
            before = decode(ref[:index]).lstrip()
            after = text[len(before) : len(before) + 1]
            if before and not before[-1].isspace() and after and not after.isspace():
                raise AmbiguousDivergence(index)
        return index, index
    return None, len(ref)


class AmbiguousDivergence(Exception):
    """A text divergence whose token position text alone cannot fix."""


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
    ap.add_argument(
        "--text-file", help="file holding the generated text (needs --model)"
    )
    ap.add_argument(
        "--mlxcel-cli",
        action="store_true",
        help="the text is captured `mlxcel generate` stdout; keep only the continuation",
    )
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
        if args.mlxcel_cli:
            try:
                text = strip_mlxcel_cli(text, oracle["prompt_text"])
            except ValueError as error:
                print(f"[near-tie] {args.label}: {error}", file=sys.stderr)
                return EXIT_USAGE
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        try:
            index, compared = first_divergence_text(text, ref, tok.decode)
        except AmbiguousDivergence as error:
            print(
                f"[near-tie] {args.label}: AMBIGUOUS (the text leaves the oracle inside a "
                f"word near token {error.args[0]}; give the stream as --ids)",
                file=sys.stderr,
            )
            return EXIT_USAGE
    else:
        ap.error("give the stream as --ids, --ids-json, --text or --text-file")

    if compared == 0 and index is None:
        print(f"[near-tie] {args.label}: no tokens to compare (empty stream)", file=sys.stderr)
        return EXIT_USAGE

    status, verdict = judge(oracle, index, args.tolerance)
    print(f"[near-tie] {args.label}: {verdict} ({compared} tokens compared)")
    return status


if __name__ == "__main__":
    sys.exit(main())
