"""Reproduction for "fix(rocm): use the shared get_2d_grid_dims and remove
get_launch_args".

cumsum along a non-last axis goes through strided_scan, whose grid is sized by
get_2d_grid_dims(shape, strides, divisor). The old local body launched 576
blocks where 48 cover a [1, 48, 24, 24] array scanned along axis 2, and the
extra blocks read and wrote past its end: an HSA_STATUS_ERROR_MEMORY_FAULT at
that shape (so each case runs in a child process with a timeout), silent
corruption where the overshoot stays in mapped memory. With the fix the GPU
matches the CPU exactly (small integers stored as float32 sum exactly).

The get_launch_args part has no runtime reproduction: nothing calls it, and a
call now fails to compile.

    python repro.py
"""

import argparse
import subprocess
import sys

import mlx.core as mx

SHAPES = [((1, 48, 24, 24), 2), ((1, 64, 16, 16), 2), ((64, 96, 32, 32), 1), ((4, 8, 16), 1), ((3, 5, 7), 0)]


def run_case(i):
    shape, axis = SHAPES[i]
    size = 1
    for d in shape:
        size *= d
    x = (mx.arange(size) % 7).astype(mx.float32).reshape(shape)
    g = mx.cumsum(x, axis=axis, stream=mx.gpu)
    c = mx.cumsum(x, axis=axis, stream=mx.cpu)
    ok = mx.array_equal(g, c).item()
    print(f"{'PASS' if ok else 'FAIL'} cumsum {shape} axis={axis}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int)
    a = p.parse_args()
    if a.case is not None:
        return run_case(a.case)
    failed = 0
    for i, (shape, axis) in enumerate(SHAPES):
        try:
            r = subprocess.run([sys.executable, __file__, "--case", str(i)], timeout=60,
                               capture_output=True, text=True)
            sys.stdout.write(r.stdout or f"FAIL cumsum {shape} axis={axis} exit={r.returncode}\n")
            failed += r.returncode != 0
        except subprocess.TimeoutExpired:
            print(f"FAIL cumsum {shape} axis={axis}: timed out (GPU fault or hang)")
            failed += 1
    print(f"{failed} failing case(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
