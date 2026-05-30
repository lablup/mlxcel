#!/usr/bin/env python3
"""Flag bare ``#NNN`` issue/PR references (3+ digits) added in a diff.

Rationale
---------
A bare ``#NNN`` auto-links to *this* repository (``lablup/mlxcel``) on GitHub.
That is correct for same-repo references, but wrong for anything else:

* References to **mlxcel-internal** (a private repository) must never appear in
  the public tree — a public reader following them lands on an unrelated issue
  or a 404, and it leaks internal planning.
* References to upstream projects (``ml-explore/mlx-lm``, ``Blaizzy/mlx-vlm``,
  ``ml-explore/mlx``, HuggingFace ``transformers`` …) should be written as
  ``org/repo#NNN`` so they resolve as cross-repository links.

This check lists every bare 3+-digit ``#NNN`` introduced on added lines so the
author/reviewer can confirm each is a genuine ``lablup/mlxcel`` reference,
qualify an upstream reference as ``org/repo#NNN``, or drop an internal one.
``org/repo#NNN`` (already qualified) and corpus fixtures are ignored.

Usage
-----
    scripts/ci/check_cross_repo_refs.py [BASE_REF]

``BASE_REF`` defaults to ``$BASE_REF`` then ``origin/main``. Set ``STRICT=1`` to
exit non-zero when any reference needs review (default is advisory: exit 0).
"""
import os
import re
import subprocess
import sys

# Exclude a leading word char (a qualified `org/repo#NNN` and URL `…html#NNN`
# anchors both have a word char right before `#`) but NOT a leading `/`, so a
# bare ref written after a slash (a milestone label or an issue range, not an
# actual `org/repo`) is still caught.
BARE = re.compile(r"(?<!\w)#([0-9]{3,})(?![0-9a-fA-F])")
# Lines naming an upstream project: a bare ref here is almost certainly an
# unqualified upstream reference that should become org/repo#NNN.
UPSTREAM = re.compile(
    r"mlx[-_]lm|mlx[-_]vlm|ml-explore|Blaizzy|transformers|hugging\s*face|"
    r"\bupstream\b|sglang|llama\.cpp|pytorch",
    re.IGNORECASE,
)
IGNORE_PREFIXES = ("tests/fixtures/", ".github/PULL_REQUEST_TEMPLATE.md")


def diff_base() -> str:
    base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BASE_REF", "origin/main")
    mb = subprocess.run(
        ["git", "merge-base", base, "HEAD"], capture_output=True, text=True
    )
    return mb.stdout.strip() or base


def main() -> int:
    base = diff_base()
    diff = subprocess.run(
        ["git", "diff", "--unified=0", f"{base}...HEAD"],
        capture_output=True, text=True,
    ).stdout

    cur = None
    upstream_hits, samerepo_hits = [], []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:]
        elif line.startswith("+") and not line.startswith("+++") and cur:
            if cur.startswith(IGNORE_PREFIXES):
                continue
            content = line[1:]
            for m in BARE.finditer(content):
                num = int(m.group(1))
                hit = (cur, m.group(1), content.strip()[:110])
                # >=1000 cannot be a lablup/mlxcel number (this repo is far below
                # that), so it is always an upstream ref to qualify; below that,
                # an upstream-naming line is upstream and the rest is same-repo.
                if num >= 1000 or UPSTREAM.search(content):
                    upstream_hits.append(hit)
                else:
                    samerepo_hits.append(hit)

    if not upstream_hits and not samerepo_hits:
        print("cross-repo-refs: OK — no bare 3+-digit '#NNN' added.")
        return 0

    print("cross-repo-refs: bare 3+-digit '#NNN' references were added.\n")
    print(
        "Policy: same-repo (lablup/mlxcel) refs use bare '#N'. Any reference to another\n"
        "repository must be qualified as 'org/repo#NNN' (e.g. ml-explore/mlx-lm#1240,\n"
        "Blaizzy/mlx-vlm#1181, ml-explore/mlx#3475). mlxcel-internal numbers must not appear.\n"
    )
    if upstream_hits:
        print(f"Likely UPSTREAM (qualify as org/repo#NNN) — {len(upstream_hits)}:")
        for f, n, c in upstream_hits:
            print(f"  {f}: #{n}  | {c}")
        print()
    if samerepo_hits:
        print(f"Verify each is a real lablup/mlxcel #N (else qualify/remove) — {len(samerepo_hits)}:")
        for f, n, c in samerepo_hits:
            print(f"  {f}: #{n}  | {c}")
        print()

    return 1 if os.environ.get("STRICT") == "1" else 0


if __name__ == "__main__":
    raise SystemExit(main())
