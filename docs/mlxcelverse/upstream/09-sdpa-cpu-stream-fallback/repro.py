"""Reproduction for "fix(rocm): fall back from fused SDPA on CPU streams".

On a ROCm build, scaled_dot_product_attention on a CPU stream with a
decode-shaped query built the fused primitive, whose eval_cpu raises "NYI".
With the fix the CPU stream takes the unfused fallback and matches a softmax
reference.

    python repro.py
"""

import sys

import mlx.core as mx


def main():
    mx.random.seed(0)
    b, h, d, kv = 1, 8, 64, 128
    q = mx.random.normal((b, h, 1, d))
    k = mx.random.normal((b, h, kv, d))
    v = mx.random.normal((b, h, kv, d))
    scale = d**-0.5
    ref = mx.softmax((q * scale) @ k.swapaxes(-1, -2), axis=-1, stream=mx.cpu) @ v
    try:
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, stream=mx.cpu)
        err = mx.abs(out - ref).max().item()
        ok = err < 1e-5
        print(f"{'PASS' if ok else 'FAIL'} decode-shaped SDPA on the CPU stream: max abs error {err:.2e}")
    except Exception as e:
        ok = False
        print(f"FAIL decode-shaped SDPA on the CPU stream: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
