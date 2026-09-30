"""Reproduction for "fix(rocm): keep ArgReduce's grid inside the per-dimension
limit".

An argmin over a short last axis with more than about 4.2M outputs launched one
1024-thread block per output on a 1-D grid, past AMD's 2^32 - 1 threads per
grid dimension, and failed with "invalid configuration argument". mxfp4
quantize of a 4096x4096 matrix reaches it through its per-group argmin.
Without the fix both cases raise; with it the GPU matches the CPU exactly.

    python repro.py
"""

import sys

import mlx.core as mx


def check(name, gpu, cpu):
    ok = mx.array_equal(gpu, cpu).item()
    print(f"{'PASS' if ok else 'FAIL'} {name}")
    return ok


def main():
    failed = 0
    mx.random.seed(0)
    x = mx.random.normal((5_000_000, 16))
    try:
        failed += not check(
            "argmin over 5M rows of 16",
            mx.argmin(x, axis=-1, stream=mx.gpu),
            mx.argmin(x, axis=-1, stream=mx.cpu),
        )
    except Exception as e:  # the unfixed launch raises
        print(f"FAIL argmin over 5M rows of 16: {e}")
        failed += 1

    w = mx.random.normal((4096, 4096)) * 0.02
    try:
        g = mx.quantize(w, group_size=32, bits=4, mode="mxfp4", stream=mx.gpu)
        c = mx.quantize(w, group_size=32, bits=4, mode="mxfp4", stream=mx.cpu)
        failed += not check("mxfp4 quantize 4096x4096 (weights)", g[0], c[0])
        failed += not check("mxfp4 quantize 4096x4096 (scales)", g[1], c[1])
    except Exception as e:
        print(f"FAIL mxfp4 quantize 4096x4096: {e}")
        failed += 1
    print(f"{failed} failing case(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
