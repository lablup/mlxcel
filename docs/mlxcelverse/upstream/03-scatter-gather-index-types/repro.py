"""Reproduction for "fix(rocm): fix Scatter's size argument and widen narrow
index dtypes".

Scatter: every update landed on the first index, so x.at[idx].add(v), eye(),
tri() and friends were wrong on the GPU. Narrow index dtypes: a gather or
scatter with int8, uint8, int16, uint16 or bool indices read eight bytes per
index and faulted the GPU queue (the process then spins instead of failing), so
every case runs in a child process with a timeout. With the fix every case
matches the CPU stream exactly.

    python repro.py
"""

import argparse
import subprocess
import sys

import mlx.core as mx

INDEX_DTYPES = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]


def on(stream, fn):
    with mx.stream(stream):
        out = fn()
        mx.eval(out)
        return out


def scatter_cases():
    idx = mx.array([1, 4, 7])
    upd = mx.array([1.0, 2.0, 3.0])
    return {
        "scatter add": lambda: mx.zeros((10,)).at[idx].add(upd),
        "scatter set": lambda: _set(idx, upd),
        "eye": lambda: mx.eye(5),
        "tri": lambda: mx.tri(5, 7, k=1),
    }


def _set(idx, upd):
    a = mx.zeros((10,))
    a[idx] = upd
    return a


def index_case(dtype, kind):
    dt = getattr(mx, dtype)
    src = mx.arange(300, dtype=mx.float32).reshape(30, 10)
    idx = mx.array([0, 5, 17, 29, 5], dtype=dt)
    if kind == "take":
        return lambda: mx.take(src, idx, axis=0)
    if kind == "getitem":
        return lambda: src[idx]
    return lambda: mx.zeros((30, 10)).at[idx].add(1.0)


def run_case(name):
    if name.startswith("index:"):
        _, dtype, kind = name.split(":")
        fn = index_case(dtype, kind)
    else:
        fn = scatter_cases()[name]
    ok = mx.array_equal(on(mx.gpu, fn), on(mx.cpu, fn)).item()
    print(f"{'PASS' if ok else 'FAIL'} {name}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case")
    a = p.parse_args()
    if a.case:
        return run_case(a.case)
    names = list(scatter_cases())
    names += [f"index:{d}:{k}" for d in INDEX_DTYPES for k in ("take", "getitem", "scatter")]
    failed = 0
    for name in names:
        try:
            r = subprocess.run([sys.executable, __file__, "--case", name], timeout=60,
                               capture_output=True, text=True)
            sys.stdout.write(r.stdout or f"FAIL {name} exit={r.returncode} {r.stderr.strip()[-200:]}\n")
            failed += r.returncode != 0
        except subprocess.TimeoutExpired:
            print(f"FAIL {name} timed out (GPU fault or hang)")
            failed += 1
    print(f"{failed} failing case(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
