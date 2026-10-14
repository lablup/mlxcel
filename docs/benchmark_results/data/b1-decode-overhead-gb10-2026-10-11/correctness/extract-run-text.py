#!/usr/bin/env python3
"""Extract the streamed completion from `mlxcel run` stdout byte-for-byte."""

from __future__ import annotations

import argparse
from pathlib import Path


START = b"Generating...\n"
END = b"\n\n[Generated "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stdout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    raw = args.stdout.read_bytes()
    start = raw.find(START)
    if start < 0:
        raise SystemExit(f"missing {START!r} in {args.stdout}")
    start += len(START)
    end = raw.rfind(END)
    if end < start:
        raise SystemExit(f"missing trailing {END!r} in {args.stdout}")
    generated = raw[start:end]
    if not generated:
        raise SystemExit(f"empty generated text in {args.stdout}")
    args.output.write_bytes(generated)


if __name__ == "__main__":
    main()
