#!/usr/bin/env python3
"""Find f32 constants that may promote an activation.

Two searches, described in docs/f32-promotion-audit.md. Neither is a gate: the
broad form has a high false-positive rate by construction, and the narrow form
is the one that found the only real defect the 2026-09 audit turned up.

  broad   an f32 array built without a dtype reaches an elementwise op
  narrow  the same, inside a function that names an intended dtype and does
          not pass it at this call

Every hit needs reading. A block that computes in f32 on purpose and casts at
the end is correct, and the cast is usually applied to a derived name several
lines later, so no purely local rule can tell the two apart.
"""

import argparse
import glob
import re

CTOR = re.compile(
    r"let\s+(?:mut\s+)?(\w+)\s*=\s*mlxcel_core::"
    r"(from_slice_f32|full_f32|zeros_f32|ones_f32|arange_f32)\s*\("
)
ELEM = re.compile(
    r"mlxcel_core::(multiply|add|subtract|divide|maximum|minimum)\(\s*&?([\w.]+)\s*,\s*&?([\w.]+)"
)
FN = re.compile(r"^\s*(pub\s+)?(async\s+)?fn\s+(\w+)")
DTVAR = re.compile(r"let\s+(\w+)\s*(?::\s*\w+)?\s*=\s*mlxcel_core::array_dtype\(")
DTLOCAL = re.compile(r"let\s+(\w*dtype\w*)\s*(?::\s*\w+)?\s*=")


def enclosing_fn(lines, i):
    for j in range(i, -1, -1):
        m = FN.match(lines[j])
        if not m:
            continue
        depth, seen = 0, False
        for k in range(j, len(lines)):
            depth += lines[k].count("{") - lines[k].count("}")
            seen = seen or lines[k].count("{") > 0
            if seen and depth <= 0:
                return m.group(3), j, k
        return m.group(3), j, len(lines) - 1
    return None


def sources(path):
    return [
        f
        for f in glob.glob(f"{path}/**/*.rs", recursive=True)
        if "_tests.rs" not in f and "/tests/" not in f and "/examples/" not in f
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["broad", "narrow"], default="narrow")
    ap.add_argument("--path", default="src")
    args = ap.parse_args()

    hits = []
    for f in sources(args.path):
        text = open(f, encoding="utf-8").read()
        lines = text.split("\n")
        dtvars = set(DTVAR.findall(text))
        for i, line in enumerate(lines):
            m = CTOR.search(line)
            if not m:
                continue
            name = m.group(1)
            call = "\n".join(lines[i : i + 6]).split(";")[0]
            # a constructor told a dtype is not a promotion source
            if "array_dtype(" in call:
                continue
            if any(re.search(rf"\b{re.escape(v)}\b", call) for v in dtvars):
                continue
            enc = enclosing_fn(lines, i)
            if not enc:
                continue
            fn_name, start, end = enc
            body = "\n".join(lines[start : end + 1])
            # A constructor immediately re-bound through a cast is handled,
            # whichever mode is running. Without this the search reports the
            # sites it has already caused to be fixed.
            window = lines[i + 1 : i + 12]
            if any(
                re.search(rf"let\s+{re.escape(name)}\s*=\s*mlxcel_core::astype\(", w)
                for w in window
            ):
                continue

            if args.mode == "narrow":
                intended = set(DTLOCAL.findall(body))
                if not intended:
                    continue
                if any(re.search(rf"\b{re.escape(v)}\b", call) for v in intended):
                    continue
                hits.append((f, i + 1, name, fn_name, ",".join(sorted(intended))))
            else:
                if any("array_dtype(" in w and name in w for w in window):
                    continue
                for w in window:
                    em = ELEM.search(w)
                    if em and name in (em.group(2), em.group(3)):
                        hits.append((f, i + 1, name, fn_name, ""))
                        break

    for f, ln, name, fn_name, extra in hits:
        note = f"  intended dtype in scope: {extra}" if extra else ""
        print(f"{f}:{ln}  {name}  (fn {fn_name}){note}")
    print(f"\n{len(hits)} candidate(s) in {args.mode} mode. Each needs reading.")


if __name__ == "__main__":
    main()
