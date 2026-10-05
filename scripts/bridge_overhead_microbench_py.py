#!/usr/bin/env python3
"""
Python mirror of `examples/bridge_overhead_microbench.rs`. Measures the
same sequence of single-op calls (add / multiply / reshape / expand_dims
/ from_slice / matmul) to quantify the per-call bridge overhead of Python
nanobind bindings for comparison against the Rust cxx bridge.
"""

from __future__ import annotations
import time
import mlx.core as mx

WARMUP = 200
ITERS = 10_000


def fmt_per_call(label: str, total: float, iters: int) -> None:
    per = total / iters * 1e9  # ns
    print(
        f"  {label:<26} total={total*1000:>8.2f}ms  "
        f"per_call={per/1000:>7.2f}us  ({iters} iters)"
    )


def bench_add(a, b):
    print("add(shape=[4]):")

    for _ in range(WARMUP):
        mx.add(a, b)
    start = time.perf_counter()
    for _ in range(ITERS):
        mx.add(a, b)
    t = time.perf_counter() - start
    fmt_per_call("no_eval", t, ITERS)

    for _ in range(WARMUP):
        c = mx.add(a, b)
        mx.eval(c)
    mx.synchronize()
    start = time.perf_counter()
    for _ in range(ITERS):
        c = mx.add(a, b)
        mx.eval(c)
    mx.synchronize()
    t = time.perf_counter() - start
    fmt_per_call("eval", t, ITERS)


def bench_multiply(a, b):
    print("multiply(shape=[4]):")

    for _ in range(WARMUP):
        mx.multiply(a, b)
    start = time.perf_counter()
    for _ in range(ITERS):
        mx.multiply(a, b)
    t = time.perf_counter() - start
    fmt_per_call("no_eval", t, ITERS)


def bench_reshape(a):
    print("reshape(shape=[4]->[2,2]):")

    for _ in range(WARMUP):
        mx.reshape(a, (2, 2))
    start = time.perf_counter()
    for _ in range(ITERS):
        mx.reshape(a, (2, 2))
    t = time.perf_counter() - start
    fmt_per_call("no_eval", t, ITERS)


def bench_expand_dims(a):
    print("expand_dims(axis=0):")

    for _ in range(WARMUP):
        mx.expand_dims(a, 0)
    start = time.perf_counter()
    for _ in range(ITERS):
        mx.expand_dims(a, 0)
    t = time.perf_counter() - start
    fmt_per_call("no_eval", t, ITERS)


def bench_from_slice():
    print("mx.array([42], dtype=int32):  # ≈ from_slice_i32 len=1")
    import numpy as np
    data = np.array([42], dtype=np.int32)

    for _ in range(WARMUP):
        mx.array(data)
    start = time.perf_counter()
    for _ in range(ITERS):
        mx.array(data)
    t = time.perf_counter() - start
    fmt_per_call("no_eval", t, ITERS)


def bench_matmul_512():
    print("matmul(512x512 @ 512x512):")
    import numpy as np
    a_np = (np.arange(512 * 512, dtype=np.float32) * 0.001).reshape(512, 512)
    b_np = (np.arange(512 * 512, dtype=np.float32) * 0.002).reshape(512, 512)
    a = mx.array(a_np)
    b = mx.array(b_np)
    mx.eval(a, b)
    mx.synchronize()

    sub_iters = ITERS // 10

    for _ in range(WARMUP):
        mx.matmul(a, b)
    start = time.perf_counter()
    for _ in range(sub_iters):
        mx.matmul(a, b)
    t = time.perf_counter() - start
    fmt_per_call("no_eval", t, sub_iters)

    for _ in range(WARMUP // 10):
        c = mx.matmul(a, b)
        mx.eval(c)
    mx.synchronize()
    start = time.perf_counter()
    for _ in range(sub_iters):
        c = mx.matmul(a, b)
        mx.eval(c)
    mx.synchronize()
    t = time.perf_counter() - start
    fmt_per_call("eval", t, sub_iters)


def main():
    print("=== Python nanobind microbench ===")
    print(f"warmup={WARMUP}, iters={ITERS}")
    print()
    import numpy as np
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    a = mx.array(data)
    b = mx.array(data)
    mx.eval(a, b)
    mx.synchronize()

    bench_add(a, b)
    print()
    bench_multiply(a, b)
    print()
    bench_reshape(a)
    print()
    bench_expand_dims(a)
    print()
    bench_from_slice()
    print()
    bench_matmul_512()


if __name__ == "__main__":
    main()
