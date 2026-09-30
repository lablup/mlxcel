"""Reproduction for "feat(rocm): implement FFT through hipFFT".

Needs the blocking-sync fix ("fix(rocm): set blocking-sync before the first HIP
queue is created") applied first.

1. Correctness: rfft, irfft, round trips and complex fft against the CPU stream
   at several lengths and batches, within 1e-5 relative.
2. Plan-cache pressure: a child process with MLX_ROCM_FFT_CACHE_SIZE=16 runs 48
   round trips at distinct lengths, so plans are evicted. Without the
   blocking-sync fix this hangs inside hipfftDestroy at the first eviction.
3. MLX_ROCM_FFT_CACHE_SIZE validation: invalid values ("0", "-1", "abc",
   "8abc", "") print one warning and fall back to the default; valid ones are
   used silently. Before the change "0" was undefined behavior and "abc" left
   every FFT throwing "stoul".

    python repro.py
"""

import os
import subprocess
import sys

import mlx.core as mx


def rel(a, b):
    return (mx.abs(a - b).max() / mx.maximum(mx.abs(b).max(), 1e-12)).item()


def correctness():
    failed = 0
    mx.random.seed(0)
    for n in (2, 8, 400, 512, 1024, 1200, 2048):
        for batch in (1, 3, 512):
            x = mx.random.normal((batch, n))
            z = x + 1j * mx.random.normal((batch, n))
            checks = {
                "rfft": (mx.fft.rfft(x, stream=mx.gpu), mx.fft.rfft(x, stream=mx.cpu)),
                "irfft": (
                    mx.fft.irfft(mx.fft.rfft(x, stream=mx.cpu), n=n, stream=mx.gpu),
                    mx.fft.irfft(mx.fft.rfft(x, stream=mx.cpu), n=n, stream=mx.cpu),
                ),
                "round trip": (mx.fft.irfft(mx.fft.rfft(x, stream=mx.gpu), n=n, stream=mx.gpu), x),
                "fft": (mx.fft.fft(z, stream=mx.gpu), mx.fft.fft(z, stream=mx.cpu)),
            }
            for name, (g, c) in checks.items():
                e = rel(g, c)
                if not e <= 1e-5:
                    print(f"FAIL {name} n={n} batch={batch}: {e:.2e}")
                    failed += 1
    print(f"{'PASS' if not failed else 'FAIL'} correctness ({failed} failing)")
    return failed


CHILD_PRESSURE = """
import mlx.core as mx
for i, n in enumerate(range(100, 100 + 48 * 7, 7)):
    x = mx.random.normal((4, n))
    y = mx.fft.irfft(mx.fft.rfft(x, stream=mx.gpu), n=n, stream=mx.gpu)
    mx.eval(y)
    assert (mx.abs(y - x).max() < 1e-4).item(), n
print("ok")
"""

CHILD_ENV = """
import mlx.core as mx
x = mx.random.normal((2, 64))
g = mx.fft.rfft(x, stream=mx.gpu)
c = mx.fft.rfft(x, stream=mx.cpu)
assert (mx.abs(g - c).max() < 1e-4).item()
print("ok")
"""


def child(code, env_value, timeout):
    env = dict(os.environ)
    if env_value is None:
        env.pop("MLX_ROCM_FFT_CACHE_SIZE", None)
    else:
        env["MLX_ROCM_FFT_CACHE_SIZE"] = env_value
    return subprocess.run([sys.executable, "-c", code], env=env, timeout=timeout,
                          capture_output=True, text=True)


def pressure():
    try:
        r = child(CHILD_PRESSURE, "16", 120)
        ok = r.returncode == 0 and "ok" in r.stdout
        print(f"{'PASS' if ok else 'FAIL'} 48 distinct plans with a 16-entry cache {r.stderr.strip()[-200:]}")
    except subprocess.TimeoutExpired:
        ok = False
        print("FAIL 48 distinct plans with a 16-entry cache: timed out (hang at plan eviction)")
    return 0 if ok else 1


def env_validation():
    failed = 0
    for value, warns in (("0", True), ("-1", True), ("abc", True), ("8abc", True), ("", True),
                         ("16", False), ("1", False), (None, False)):
        r = child(CHILD_ENV, value, 60)
        warned = "MLX_ROCM_FFT_CACHE_SIZE" in r.stderr
        ok = r.returncode == 0 and "ok" in r.stdout and warned == warns
        print(f"{'PASS' if ok else 'FAIL'} MLX_ROCM_FFT_CACHE_SIZE={value!r}: exit {r.returncode}, warned={warned}")
        failed += not ok
    return failed


def main():
    failed = correctness() + pressure() + env_validation()
    print(f"{failed} failing check(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
