"""Reproduction for "feat(rocm): implement Hadamard for power-of-two sizes up
to 8192".

Without the change mx.hadamard_transform raises on the GPU (NO_GPU stub). With
it the GPU matches the CPU stream for power-of-two sizes up to 8192, the
default-scale transform is its own inverse, and unsupported sizes raise a
message naming the limit instead of returning a wrong answer.

    python repro.py
"""

import sys

import mlx.core as mx

TOL = {"float32": 1e-5, "float16": 2e-3, "bfloat16": 2e-2}


def rel_rms(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.sqrt(mx.mean((a - b) ** 2)) / mx.sqrt(mx.mean(b**2))).item()


def main():
    failed = 0
    mx.random.seed(0)
    for n in (64, 128, 256, 1024, 8192):
        for dtype in ("float32", "float16", "bfloat16"):
            x = mx.random.normal((3, 5, n)).astype(getattr(mx, dtype))
            try:
                g = mx.hadamard_transform(x, stream=mx.gpu)
                c = mx.hadamard_transform(x, stream=mx.cpu)
                back = mx.hadamard_transform(g, stream=mx.gpu)
                e1, e2 = rel_rms(g, c), rel_rms(back, x)
                ok = e1 <= TOL[dtype] and e2 <= 2 * TOL[dtype]
                print(f"{'PASS' if ok else 'FAIL'} n={n} {dtype}: vs CPU {e1:.2e}, round trip {e2:.2e}")
            except Exception as e:
                ok = False
                print(f"FAIL n={n} {dtype}: {e}")
            failed += not ok
    for n in (16384, 12 * 64):
        try:
            mx.eval(mx.hadamard_transform(mx.ones((1, n)), stream=mx.gpu))
            print(f"FAIL n={n}: expected an error naming the limit")
            failed += 1
        except Exception as e:
            print(f"PASS n={n} raises: {e}")
    print(f"{failed} failing case(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
