"""Phase 2a core question: do XLA and IREE FUSE the int4 unpack/convert/scale
into the matmul, or materialize a full fp32 weight first?

Isolates one int4 linear (y = x @ dequant(packed,scale,wmin).T) so the optimized
HLO and the IREE dispatch IR are small and readable, then characterizes fusion.
"""
import collections
import re
import subprocess

import numpy as np
import jax
import jax.numpy as jnp
from jax import export as jexport

import model_int4 as Mi
import quant as Q

K, N, G = 2048, 2048, 64
ART = "artifacts"
IREE = ".venv/bin/iree-compile"


def f(x, packed, scale, wmin):
    W = Mi._dequant_ig((packed, scale, wmin), G)   # [N, K] dequantized in-graph
    return x @ W.T                                  # [1, N]


def main():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((N, K)).astype(np.float32)
    packed, scale, wmin = Q.quantize_affine_int4(W, G)
    x = rng.standard_normal((1, K)).astype(np.float32)
    args = (jnp.asarray(x), jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(wmin))

    # ---- XLA: optimized HLO ----
    compiled = jax.jit(f).lower(*args).compile()
    hlo = compiled.as_text()
    open(f"{ART}/qmatmul_xla_optimized.hlo", "w").write(hlo)

    # find fusion computations and whether dequant ops co-occur with the dot
    fused = False
    materialized_weight = False
    for comp in re.split(r"\n\}\n", hlo):
        has_dot = bool(re.search(r"\bdot\(", comp)) or "dot(" in comp
        has_dequant = any(t in comp for t in ("shift-right-logical", "convert", "multiply"))
        if has_dot and has_dequant:
            fused = True
    # a materialized f32 [N,K] weight buffer is the tell for non-fusion
    if re.search(rf"f32\[{N},{K}\].*(fusion|convert|multiply)", hlo):
        materialized_weight = True
    dots = hlo.count(" dot(")
    custom_matmul = bool(re.search(r"custom-call.*(matmul|dot|Eigen|gemm)", hlo, re.I))
    print("=== XLA optimized HLO (single int4 linear) ===")
    print(f"  dot instructions          : {dots}")
    print(f"  matmul lowered to custom-call: {custom_matmul}")
    print(f"  a fusion contains BOTH dequant ops AND the dot: {fused}")
    print(f"  a full f32[{N},{K}] dequant weight is materialized: {materialized_weight}")

    # ---- IREE: dispatch regions ----
    exp = jexport.export(jax.jit(f))(*[jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args])
    open(f"{ART}/qmatmul.stablehlo.mlir", "w").write(exp.mlir_module())
    proc = subprocess.run(
        [IREE, "--iree-input-type=stablehlo", "--iree-hal-target-backends=llvm-cpu",
         "--mlir-print-ir-after=iree-flow-form-dispatch-regions",
         "--mlir-disable-threading", f"{ART}/qmatmul.stablehlo.mlir", "-o", f"{ART}/qmatmul.vmfb"],
        capture_output=True, text=True)
    disp = proc.stderr
    open(f"{ART}/qmatmul_iree_dispatch.mlir", "w").write(disp)
    n_regions = disp.count("flow.dispatch.region")
    # within each dispatch region, does the matmul region also carry the unpack/convert?
    regions = re.split(r"flow\.dispatch\.region", disp)
    matmul_region_has_dequant = any(
        ("linalg.matmul" in r or "linalg.mmt4d" in r or "linalg.generic" in r and "arith.mulf" in r)
        and any(t in r for t in ("arith.shrui", "arith.uitofp", "arith.extui", "arith.sitofp"))
        for r in regions)
    has_matmul = any(t in disp for t in ("linalg.matmul", "linalg.mmt4d"))
    # a separate dispatch producing a full [N,K] f32 tensor = materialized dequant
    materialized_iree = bool(re.search(rf"tensor<{N}x{K}xf32>", disp))
    print("\n=== IREE dispatch regions (single int4 linear) ===")
    print(f"  flow.dispatch.region count : {n_regions}")
    print(f"  matmul present (matmul/mmt4d): {has_matmul}")
    print(f"  matmul dispatch also carries the int4 unpack/convert (fused): {matmul_region_has_dequant}")
    print(f"  a full tensor<{N}x{K}xf32> dequant weight is materialized: {materialized_iree}")
    print(f"\nwrote {ART}/qmatmul_xla_optimized.hlo, qmatmul_iree_dispatch.mlir, qmatmul.stablehlo.mlir")


if __name__ == "__main__":
    main()
