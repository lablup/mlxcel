#!/usr/bin/env python3
"""Generate PolarQuant optimal-centroid reference fixtures for Rust tests.

Runs the Lloyd-Max algorithm from the reference implementation and dumps
centroids to JSON for byte-for-byte comparison in Rust unit tests.

Usage:
    python3 generate_fixtures.py

Output:
    codebooks.json  — centroids for all (bit_width, head_dim) pairs

The JSON schema is:
    {
      "<b>_<d>": [c0, c1, ..., c_{2^b - 1}]  // f32 values as JSON numbers
    }
where b ∈ {2, 3, 4} and d ∈ {64, 80, 96, 128, 192, 256}.
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Reference implementation (verbatim from https://github.com/TheTom/turboquant_plus/blob/main/turboquant/codebook.py)
# ---------------------------------------------------------------------------

def optimal_centroids(bit_width: int, d: int) -> np.ndarray:
    """Compute optimal MSE centroids for the post-rotation coordinate distribution.

    Args:
        bit_width: Number of bits per coordinate (1, 2, 3, 4, ...).
        d: Vector dimension (affects centroid scale).

    Returns:
        Sorted array of 2^bit_width centroids.
    """
    n_centroids = 1 << bit_width

    if bit_width == 1:
        c = np.sqrt(2.0 / (np.pi * d))
        return np.array([-c, c])

    if bit_width == 2:
        return np.array([-1.51, -0.453, 0.453, 1.51]) / np.sqrt(d)

    # For b >= 3, use Lloyd's algorithm on N(0, 1/d)
    return _lloyds_gaussian(n_centroids, sigma=1.0 / np.sqrt(d))


def _lloyds_gaussian(n_centroids: int, sigma: float, n_iter: int = 100) -> np.ndarray:
    """Lloyd's algorithm (iterative k-means) for optimal scalar quantization of N(0, sigma^2).

    Args:
        n_centroids: Number of quantization levels (2^b).
        sigma: Standard deviation of the Gaussian.
        n_iter: Number of Lloyd iterations.

    Returns:
        Sorted array of optimal centroids.
    """
    # Initialize boundary positions from uniform quantiles
    boundaries = stats.norm.ppf(
        np.linspace(0, 1, n_centroids + 1)[1:-1], scale=sigma
    )
    centroids = np.zeros(n_centroids)

    # Initial centroids: conditional expectations within each region
    centroids[0] = _gaussian_conditional_expectation(sigma, -np.inf, boundaries[0])
    for i in range(1, n_centroids - 1):
        centroids[i] = _gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i])
    centroids[-1] = _gaussian_conditional_expectation(sigma, boundaries[-1], np.inf)

    for _ in range(n_iter):
        # Update boundaries (midpoints between consecutive centroids)
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Update centroids (conditional expectations within each region)
        centroids[0] = _gaussian_conditional_expectation(sigma, -np.inf, boundaries[0])
        for i in range(1, n_centroids - 1):
            centroids[i] = _gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i])
        centroids[-1] = _gaussian_conditional_expectation(sigma, boundaries[-1], np.inf)

    return np.sort(centroids)


def _gaussian_conditional_expectation(sigma: float, a: float, b: float) -> float:
    """E[X | a < X < b] where X ~ N(0, sigma^2).

    Uses the formula: E[X | a < X < b] = sigma^2 * (phi(a/sigma) - phi(b/sigma)) / (Phi(b/sigma) - Phi(a/sigma))
    where phi is the PDF and Phi is the CDF of standard normal.
    """
    a_std = a / sigma if np.isfinite(a) else a
    b_std = b / sigma if np.isfinite(b) else b

    # Use sf() for upper tail to avoid CDF cancellation at extreme values
    # prob = P(a < X/sigma < b) using the more numerically stable formulation
    if not np.isfinite(a_std):
        prob = stats.norm.cdf(b_std)
    elif not np.isfinite(b_std):
        prob = stats.norm.sf(a_std)
    else:
        prob = stats.norm.cdf(b_std) - stats.norm.cdf(a_std)

    if prob < 1e-15:
        # For semi-infinite intervals, use asymptotic approximation
        if np.isfinite(a) and not np.isfinite(b):
            return a + sigma  # E[X | X > a] ≈ a + sigma for extreme a
        elif not np.isfinite(a) and np.isfinite(b):
            return b - sigma
        elif np.isfinite(a) and np.isfinite(b):
            return (a + b) / 2.0
        else:  # pragma: no cover — both infinite always has prob=1
            return 0.0

    pdf_diff = stats.norm.pdf(a_std) - stats.norm.pdf(b_std)
    return sigma * pdf_diff / prob


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------

BIT_WIDTHS = [2, 3, 4]
HEAD_DIMS = [64, 80, 96, 128, 192, 256]


def f64_to_f32_hex(value: float) -> str:
    """Convert f64 value to f32 hex string for exact bit-level representation."""
    f32_bytes = struct.pack(">f", float(np.float32(value)))
    return f32_bytes.hex()


def main() -> None:
    output: dict = {}
    hex_output: dict = {}

    print("Generating PolarQuant codebook fixtures...")
    print(f"  bit_widths: {BIT_WIDTHS}")
    print(f"  head_dims: {HEAD_DIMS}")
    print()

    for b in BIT_WIDTHS:
        for d in HEAD_DIMS:
            key = f"{b}_{d}"
            centroids_f64 = optimal_centroids(b, d)
            # Downcast to f32 — this is what the Rust code will use
            centroids_f32 = centroids_f64.astype(np.float32)

            output[key] = centroids_f32.tolist()
            hex_output[key] = [f64_to_f32_hex(c) for c in centroids_f32]

            print(f"  b={b}, d={d:3d}: {centroids_f32.tolist()}")

    out_dir = Path(__file__).parent
    json_path = out_dir / "codebooks.json"
    hex_path = out_dir / "codebooks_hex.json"

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {json_path}")

    with open(hex_path, "w") as f:
        json.dump(hex_output, f, indent=2)
    print(f"Wrote {hex_path}")

    # Verify the fixtures can be re-read
    with open(json_path) as f:
        loaded = json.load(f)
    assert set(loaded.keys()) == {f"{b}_{d}" for b in BIT_WIDTHS for d in HEAD_DIMS}
    print("\nFixture verification: OK")

    # Print a summary table
    print("\n--- Centroid count summary ---")
    for b in BIT_WIDTHS:
        for d in HEAD_DIMS:
            key = f"{b}_{d}"
            print(f"  b={b}, d={d:3d}: {len(loaded[key])} centroids = {loaded[key]}")


if __name__ == "__main__":
    main()
