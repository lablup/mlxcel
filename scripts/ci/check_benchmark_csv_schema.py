#!/usr/bin/env python3
"""Assert the benchmark CSV producers agree on one schema.

The harnesses write one CSV family: `bench_decode.sh` for generation,
`bench_longprompt.sh` for the prefill ladder, and the two Python server
ladders (`bench_embeddings.py`, `bench_imagegen.py`). A header/row width
mismatch is silent. `csv.writer` does not check the count and `csv.DictReader`
pads a short row with `None`, so a shifted column reads as a plausible number
until someone compares two files by name.

The check runs without loading a model or a GPU:

1. `bench_decode.sh` names every column once, with `backend` immediately before
   `decode_path`.
2. Every row template it emits has the header width, in both the CUDA/Metal
   configuration (two commit fields) and the ROCm one (four plus two header
   names).
3. `bench_longprompt.sh` mirrors the non-ROCm header and its synthesized skip
   rows match it.
4. The Python ladders put `backend` before `notes` and write LF line endings.
5. `MLXCEL_BENCH_BACKEND` reaches the Python `detect_backend()` unchanged.

Usage: `python3 scripts/ci/check_benchmark_csv_schema.py [--root DIR]`.
"""
import argparse
import csv
import importlib.util
import os
import re
import sys
from pathlib import Path

BASE_COLUMNS = [
    "model", "model_path", "prompt_tokens", "generated_tokens",
    "prefill_ms", "prefill_tok_s", "decode_ms", "decode_tok_s",
    "date", "hardware", "mlxcel_version", "build_type", "max_tokens",
    "prompt", "prompt_target_len", "mlxcel_commit", "mlx_commit",
]
ROCM_COLUMNS = ["mlx_rocm_overlay_commit", "hip_version"]
TAIL_COLUMNS = ["backend", "decode_path"]
ROW_START = re.compile(r'^\s*echo "((?:\\"|[^"])*)"(.*)$', re.M)
HEADER_DECL = re.compile(r'^CSV_HEADER="(.*)"\s*$', re.M)
VARIABLE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|\$[0-9]+")


def expand(template, rocm):
    """A shell row template as CSV text with distinctive placeholders.

    Variables that carry several CSV fields expand to several placeholders:
    `${fields}` is the six profiling numbers and `${COMMIT_FIELDS}` is two or
    four fields depending on the backend. Everything else is one field and
    becomes `V`, which keeps quoted prompts and the date from skewing the
    width.
    """
    text = template.replace('\\"', '"')
    text = text.replace("${fields}", "F1,F2,F3,F4,F5,F6")
    text = text.replace("${COMMIT_FIELDS}", "C1,C2,C3,C4" if rocm else "C1,C2")
    text = text.replace("${CSV_HEADER_EXTRA}",
                        ",mlx_rocm_overlay_commit,hip_version" if rocm else "")
    text = text.replace("${BACKEND_LABEL}", "BACKEND")
    text = text.replace("${decode_path}", "DECODE_PATH")
    text = text.replace("${classification}", "SKIP:test")
    return VARIABLE.sub("V", text)


def parse_one(text):
    rows = list(csv.reader([text]))
    if len(rows) != 1:
        raise ValueError(f"template parses as {len(rows)} rows: {text!r}")
    return rows[0]


def list_bodies(source):
    """The bracketed bodies of `writer.writerow([...])` and `row = [...]`.

    Scans with bracket depth and skips quoted strings, because a row list
    contains `meta["backend"]` and a naive non-greedy match stops at its `]`.
    """
    bodies = []
    for match in re.finditer(r"(?:writer\.writerow\(|row = )\[", source):
        index = match.end()
        depth = 1
        quote = None
        while index < len(source) and depth:
            char = source[index]
            if quote:
                if char == "\\":
                    index += 2
                    continue
                if char == quote:
                    quote = None
            elif char in "'\"":
                quote = char
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            index += 1
        bodies.append(source[match.end():index - 1])
    return bodies


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def check_shell(path, name, rocm, failures):
    text = path.read_text()
    match = HEADER_DECL.search(text)
    if not match:
        failures.append(f"{name}: no CSV_HEADER declaration")
        return
    columns = parse_one(expand(match.group(1), rocm))
    expected = BASE_COLUMNS + (ROCM_COLUMNS if rocm else []) + TAIL_COLUMNS
    require(columns == expected,
            f"{name}: header is {columns}, expected {expected}", failures)
    if columns != expected:
        return
    backend_at = columns.index("backend")
    decode_at = columns.index("decode_path")

    templates = [(body, rest) for body, rest in ROW_START.findall(text)
                 if body.startswith("${model_name}") and "OOM_RECORD_FILE" not in rest]
    require(bool(templates), f"{name}: no row templates found", failures)
    for template, _rest in templates:
        fields = parse_one(expand(template, rocm))
        head = template[:60].replace("\n", " ")
        if len(fields) == len(columns):
            status = ""
        elif len(fields) == len(columns) + 1:
            status = fields[-1]
            require(status.startswith(("SKIP:", "FAIL:")),
                    f"{name}: extra field in {head!r} is not a status token: {status!r}",
                    failures)
        else:
            failures.append(
                f"{name}: row has {len(fields)} fields, header has {len(columns)}: {head!r}")
            continue
        named = fields[:len(columns)]
        require(named[backend_at] == "BACKEND",
                f"{name}: backend column is {named[backend_at]!r} in {head!r}",
                failures)
        require(named[decode_at] in ("DECODE_PATH", ""),
                f"{name}: decode_path column is {named[decode_at]!r} in {head!r}",
                failures)
        if not status:
            require(named[decode_at] == "DECODE_PATH",
                    f"{name}: successful row has an empty decode_path: {head!r}",
                    failures)


def load_module(cwd, filename):
    spec = importlib.util.spec_from_file_location(
        Path(filename).stem, cwd / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_python(cwd, filename, failures):
    source = (cwd / "scripts" / filename).read_text()
    try:
        module = load_module(cwd, filename)
    except Exception as err:  # noqa: BLE001 - the message is the report
        failures.append(f"{filename}: import failed ({type(err).__name__}: {err})")
        return
    header = list(module.CSV_HEADER)
    require(header.count("backend") == 1,
            f"{filename}: header has {header.count('backend')} backend columns",
            failures)
    require(header and header[-1] == "notes",
            f"{filename}: last header column is {header[-1:]!r}, expected 'notes'",
            failures)
    if "backend" in header and header[-1] == "notes":
        require(header.index("backend") == len(header) - 2,
                f"{filename}: backend is not immediately before notes",
                failures)
    require('lineterminator="\\n"' in source,
            f"{filename}: csv.writer does not pin LF line endings", failures)

    row_lists = list_bodies(source)
    require(bool(row_lists), f"{filename}: no row lists found", failures)
    for index, body in enumerate(row_lists):
        require(re.search(r"(?<![A-Za-z0-9_])backend(?![A-Za-z0-9_])", body) is not None,
                f"{filename}: row list #{index + 1} omits the backend value",
                failures)


def check_override(cwd, filename, failures):
    previous = os.environ.get("MLXCEL_BENCH_BACKEND")
    os.environ["MLXCEL_BENCH_BACKEND"] = "schema-test-1.2"
    try:
        module = load_module(cwd, filename)
        got = module.detect_backend()
    finally:
        if previous is None:
            os.environ.pop("MLXCEL_BENCH_BACKEND", None)
        else:
            os.environ["MLXCEL_BENCH_BACKEND"] = previous
    require(got == "schema-test-1.2",
            f"{filename}: MLXCEL_BENCH_BACKEND did not reach detect_backend() ({got!r})",
            failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    cwd = Path(args.root)
    failures = []

    check_shell(cwd / "scripts" / "bench_decode.sh", "bench_decode.sh", False, failures)
    check_shell(cwd / "scripts" / "bench_decode.sh", "bench_decode.sh (ROCm)", True, failures)
    check_shell(cwd / "scripts" / "bench_longprompt.sh", "bench_longprompt.sh", False, failures)

    for filename in ("bench_embeddings.py", "bench_imagegen.py"):
        check_python(cwd, filename, failures)
        check_override(cwd, filename, failures)

    if failures:
        print("benchmark CSV schema check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("OK: benchmark CSV producers agree on the schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
