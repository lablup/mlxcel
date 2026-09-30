"""Reproduction for "fix(rocm): fix SliceUpdate grid-stride tail and source
donation".

Tail: x.at[a:b].add(u) and the other slice reductions left every element past
65535 * 256 * NWORK unapplied (the last 257 of 16,777,217 for a contiguous
int32 update with an odd innermost size). Donation: a source array the caller
still holds was overwritten by a GPU slice update when its buffer had a single
owner. Without the fix the tail cases report mismatching elements and the
donation cases report a modified source; with it everything matches the CPU.

    python repro.py
"""

import sys

import mlx.core as mx


def compare(name, fn):
    with mx.stream(mx.gpu):
        g = fn()
        mx.eval(g)
    with mx.stream(mx.cpu):
        c = fn()
        mx.eval(c)
    bad = mx.sum(g != c).item()
    print(f"{'PASS' if bad == 0 else 'FAIL'} {name}: {bad} mismatching element(s)")
    return bad == 0


def tail_cases():
    n = 16_777_217  # odd, so NWORK = 1 and the clamp covers 16,776,960
    ok = compare(
        "contiguous Sum, 16,777,217 elements",
        lambda: mx.zeros((n + 3,), dtype=mx.int32).at[1 : n + 1].add(mx.ones((n,), dtype=mx.int32)),
    )
    m = 4099
    ok &= compare(
        "strided Max, [4099, 4097] into [4099, 4099]",
        lambda: mx.zeros((m, m), dtype=mx.int32).at[:, 1 : m - 1].maximum(
            mx.arange(m * (m - 2), dtype=mx.int32).reshape(m, m - 2)
        ),
    )
    return ok


def held_source(name, update):
    with mx.stream(mx.gpu):
        src = mx.arange(8, dtype=mx.int32)
        mx.eval(src)
        out = update(src)
        mx.eval(out)
        ok = mx.array_equal(src, mx.arange(8, dtype=mx.int32)).item()
    print(f"{'PASS' if ok else 'FAIL'} {name}: held source {'unchanged' if ok else 'was overwritten'}")
    return ok


def donation_cases():
    ones = mx.ones((2,), dtype=mx.int32)
    ok = held_source("SliceUpdate Sum", lambda s: s.at[2:4].add(ones))
    ok &= held_source("SliceUpdate Max", lambda s: s.at[2:4].maximum(ones * 100))
    ok &= held_source(
        "DynamicSliceUpdate",
        lambda s: mx.slice_update(s, ones * 100, mx.array([2]), axes=[0]),
    )
    return ok


def main():
    ok = tail_cases()
    ok &= donation_cases()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
