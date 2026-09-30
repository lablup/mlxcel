"""Reproduction for "fix(rocm): fix quantized matmul dispatch for mxfp4/mxfp8,
f32 and f16".

Compares GPU quantized_matmul and gather_qmm against a float32 reference built
from the CPU dequantization. Each case runs in a child process with a timeout,
because on rocm-support without the fix the mxfp4/mxfp8 quantized_matmul cases
can fault the GPU queue, which leaves the process spinning instead of failing.
Expected without the fix: mxfp4/mxfp8 quantized_matmul NaN or a fault,
mxfp4/mxfp8 gather_qmm relative error of 1 or more, float32 affine
quantized_matmul garbage. With the fix every case passes.

    python repro.py            # correctness
    python repro.py --bench    # also time float16 against bfloat16 gather_qmm
"""

import argparse
import subprocess
import sys
import time

import mlx.core as mx

TOL = {"float32": 1e-3, "float16": 1e-2, "bfloat16": 3e-2}
MODES = [("affine", 4, 64), ("mxfp4", 4, 32), ("mxfp8", 8, 32)]
DTYPES = ["float32", "float16", "bfloat16"]


def quantize(w, mode, bits, group_size, dtype=mx.float32):
    """Quantize on the CPU. Affine scales and biases are cast to the activation
    dtype, as a checkpoint stores them; otherwise MLX would promote the
    activations to float32 and the half-precision kernels would never run."""
    out = mx.quantize(w, group_size=group_size, bits=bits, mode=mode, stream=mx.cpu)
    wq, s, b = out[0], out[1], (out[2] if len(out) > 2 else None)
    if mode == "affine":
        s, b = s.astype(dtype), b.astype(dtype)
    return wq, s, b


def rel_err(y, ref):
    y = y.astype(mx.float32)
    return (mx.abs(y - ref).max() / mx.maximum(mx.abs(ref).max(), 1e-6)).item()


def case_qmm(mode, bits, gs, dtype, m):
    mx.random.seed(0)
    n, k = 1024, 1024
    w = mx.random.normal((n, k)) * 0.05
    wq, s, b = quantize(w, mode, bits, gs, getattr(mx, dtype))
    wd = mx.dequantize(wq, s, b, group_size=gs, bits=bits, mode=mode, dtype=mx.float32, stream=mx.cpu)
    x = mx.random.normal((m, k)).astype(getattr(mx, dtype))
    ref = mx.matmul(x.astype(mx.float32), wd.T, stream=mx.cpu)
    y = mx.quantized_matmul(
        x, wq, s, b, transpose=True, group_size=gs, bits=bits, mode=mode, stream=mx.gpu
    )
    return rel_err(y, ref)


def case_gather(mode, bits, gs, dtype, sorted_indices):
    mx.random.seed(1)
    e, n, k, tokens = 4, 512, 1024, 8
    w = mx.random.normal((e, n, k)) * 0.05
    wq, s, b = quantize(w, mode, bits, gs, getattr(mx, dtype))
    wd = mx.dequantize(wq, s, b, group_size=gs, bits=bits, mode=mode, dtype=mx.float32, stream=mx.cpu)
    x = mx.random.normal((tokens, 1, k)).astype(getattr(mx, dtype))
    idx = mx.array([3, 0, 2, 2, 1, 3, 0, 1], dtype=mx.uint32)
    if sorted_indices:
        idx = mx.sort(idx, stream=mx.cpu)
    ref = mx.stack(
        [mx.matmul(x[t].astype(mx.float32), wd[idx[t].item()].T, stream=mx.cpu) for t in range(tokens)]
    )
    y = mx.gather_qmm(
        x, wq, s, b, rhs_indices=idx, transpose=True, group_size=gs, bits=bits,
        mode=mode, sorted_indices=sorted_indices, stream=mx.gpu,
    )
    return rel_err(y, ref)


def cases():
    for mode, bits, gs in MODES:
        for dtype in DTYPES:
            for m in (1, 4):
                yield f"qmm:{mode}:{bits}:{gs}:{dtype}:{m}"
            for srt in (0, 1):
                yield f"gather:{mode}:{bits}:{gs}:{dtype}:{srt}"


def run_case(name):
    kind, mode, bits, gs, dtype, extra = name.split(":")
    bits, gs, extra = int(bits), int(gs), int(extra)
    if kind == "qmm":
        err = case_qmm(mode, bits, gs, dtype, extra)
    else:
        err = case_gather(mode, bits, gs, dtype, bool(extra))
    ok = err <= TOL[dtype]  # False for NaN
    print(f"{'PASS' if ok else 'FAIL'} {name} rel_err={err:.3g}")
    return 0 if ok else 1


def bench():
    # Mixtral-8x7B expert shape, 4-bit affine, two tokens routed to two experts.
    e, n, k = 8, 14336, 4096
    w = mx.random.normal((e, n, k)) * 0.02
    idx = mx.array([2, 5], dtype=mx.uint32)
    for dtype in ("bfloat16", "float16"):
        wq, s, b = quantize(w, "affine", 4, 64, getattr(mx, dtype))
        x = mx.random.normal((2, 1, k)).astype(getattr(mx, dtype))
        run = lambda: mx.eval(mx.gather_qmm(x, wq, s, b, rhs_indices=idx, group_size=64, bits=4, stream=mx.gpu))
        for _ in range(3):
            run()
        t0 = time.perf_counter()
        for _ in range(20):
            run()
        print(f"gather_qmm {dtype}: {(time.perf_counter() - t0) / 20 * 1e3:.2f} ms per call")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case")
    p.add_argument("--bench", action="store_true")
    a = p.parse_args()
    if a.case:
        return run_case(a.case)
    failed = 0
    for name in cases():
        try:
            r = subprocess.run([sys.executable, __file__, "--case", name], timeout=120,
                               capture_output=True, text=True)
            sys.stdout.write(r.stdout or f"FAIL {name} exit={r.returncode} {r.stderr.strip()[-200:]}\n")
            failed += r.returncode != 0
        except subprocess.TimeoutExpired:
            print(f"FAIL {name} timed out (GPU fault or hang)")
            failed += 1
    if a.bench:
        bench()
    print(f"{failed} failing case(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
