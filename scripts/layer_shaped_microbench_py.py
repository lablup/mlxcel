#!/usr/bin/env python3
"""
Python mirror of `examples/layer_shaped_microbench.rs`. Runs the same
synthetic N × (rms_norm → linear → rms_norm → linear → add) block and
times it in two modes: one eval at the end of each step vs eval after
each layer.

Usage:
  python scripts/layer_shaped_microbench_py.py [HIDDEN] [N_LAYERS] [N_STEPS]
"""
from __future__ import annotations
import sys
import time
import numpy as np
import mlx.core as mx


def init_weight(shape):
    total = 1
    for s in shape:
        total *= s
    data = np.sin(np.arange(total, dtype=np.float32) * 0.0003) * 0.01
    return mx.array(data.reshape(shape))


def run_layers_no_sync(x0, w_norm1, w_norm2, w1, w2, n_layers):
    x = x0 + 0.0  # cheap copy
    for _ in range(n_layers):
        h = mx.fast.rms_norm(x, w_norm1, 1e-6)
        h = mx.matmul(h, w1)
        h = mx.fast.rms_norm(h, w_norm2, 1e-6)
        h = mx.matmul(h, w2)
        x = x + h
    return x


def run_layers_per_layer_sync(x0, w_norm1, w_norm2, w1, w2, n_layers):
    x = x0 + 0.0
    for _ in range(n_layers):
        h = mx.fast.rms_norm(x, w_norm1, 1e-6)
        h = mx.matmul(h, w1)
        h = mx.fast.rms_norm(h, w_norm2, 1e-6)
        h = mx.matmul(h, w2)
        x = x + h
        mx.eval(x)
    return x


def main():
    hidden = int(sys.argv[1]) if len(sys.argv) > 1 else 2816
    n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    print(f"=== Python layer-shaped microbench === "
          f"HIDDEN={hidden} N_LAYERS={n_layers} N_STEPS={n_steps}")
    print()

    x0 = init_weight((1, 1, hidden))
    w_norm1 = init_weight((hidden,))
    w_norm2 = init_weight((hidden,))
    w1 = init_weight((hidden, hidden))
    w2 = init_weight((hidden, hidden))
    mx.eval(x0, w_norm1, w_norm2, w1, w2)
    mx.synchronize()

    # warmup
    for _ in range(3):
        y = run_layers_no_sync(x0, w_norm1, w_norm2, w1, w2, n_layers)
        mx.eval(y)
    mx.synchronize()

    # no_sync
    start = time.perf_counter()
    for _ in range(n_steps):
        y = run_layers_no_sync(x0, w_norm1, w_norm2, w1, w2, n_layers)
        mx.eval(y)
    mx.synchronize()
    t_no_sync = time.perf_counter() - start
    print(
        f"no_sync          total={t_no_sync*1000:>8.2f}ms  "
        f"per_step={t_no_sync*1000/n_steps:>8.3f}ms  "
        f"({n_steps} steps × {n_layers} layers)"
    )

    # per_layer_sync
    for _ in range(3):
        _ = run_layers_per_layer_sync(x0, w_norm1, w_norm2, w1, w2, n_layers)
    mx.synchronize()

    start = time.perf_counter()
    for _ in range(n_steps):
        _ = run_layers_per_layer_sync(x0, w_norm1, w_norm2, w1, w2, n_layers)
    mx.synchronize()
    t_per_layer = time.perf_counter() - start
    print(
        f"per_layer_sync   total={t_per_layer*1000:>8.2f}ms  "
        f"per_step={t_per_layer*1000/n_steps:>8.3f}ms  "
        f"({n_steps} steps × {n_layers} layers)"
    )

    ratio = t_per_layer / t_no_sync
    print()
    print(f"per_layer_sync / no_sync = {ratio:.2f}× — "
          f"bigger ratio means graph fusion gives more here")


if __name__ == "__main__":
    main()
