#!/usr/bin/env python3
"""Interleaved two-binary A/B for issue #1934, built on #1797's served sweep.

The question is whether compiling MLX for `121a-real;121` instead of `121`
costs decode throughput, because the architecture-specific cubin is the image
the driver loads and CUTLASS compiles the decode kernels differently under it
(`CUTLASS_ARCH_MMA_SM121A_ENABLED`, derived from `__CUDA_ARCH_SPECIFIC__`).

A sequential A/B answers that only as well as the host holds still between the
two arms. This driver removes that dependency by interleaving at round
granularity: one server start, one measured request, teardown, then the other
binary, and the arm that opens alternates every round so neither binary
systematically owns the warmer or cooler half of the session.

`run_arm` is imported from the #1797 harness rather than reimplemented, so the
host gate, the NVRM trip wire, the DFlash diagnostics parsing and the
"speculative actually ran" predicate are the same code that produced the
records this is compared against.

  ab_interleave.py --a-bin BIN_121 --b-bin BIN_COMBINED \\
      --target DIR --drafter DIR --rounds 6 --out ab.jsonl
"""
import argparse
import json
import os
import statistics
import sys

# The #1797 served harness, imported rather than copied. Point
# MLXCEL_SWEEP_HARNESS at it; there is no useful default, because this driver
# is meant to run from anywhere and guessing a repository root from __file__
# only produces a confusing failure two imports later.
HARNESS = os.environ.get("MLXCEL_SWEEP_HARNESS")
if not HARNESS or not os.path.isdir(HARNESS):
    raise SystemExit(
        "set MLXCEL_SWEEP_HARNESS to the #1797 harness directory "
        "(docs/benchmark_results/data/draft-block-width-gb10-2026-09-20/harness)"
    )
sys.path.insert(0, HARNESS)
import sweep_server_widths as sweep  # noqa: E402


class Args:
    """The attribute surface `sweep.run_arm` reads off its argparse namespace."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="one server binary to compare, repeatable; NAME labels it in the record",
    )
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument(
        "--widths",
        default="4,classic",
        help="arms measured per binary per round; `classic` drops the drafter",
    )
    ap.add_argument("--n", type=int, default=1, help="measured requests per server start")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--port", type=int, default=18934)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logdir", default=None)
    a = ap.parse_args()

    a.logdir = a.logdir or os.path.dirname(os.path.abspath(a.out))
    os.makedirs(a.logdir, exist_ok=True)
    prompt = open(os.path.join(HARNESS, "prompt_retry.txt")).read()
    widths = [w if w == "classic" else int(w) for w in a.widths.split(",")]

    arms = []
    for spec in a.arm:
        name, _, path = spec.partition("=")
        if not path or not os.path.exists(path):
            raise SystemExit(f"--arm {spec}: expected NAME=PATH with an existing binary")
        arms.append((chr(ord("A") + len(arms)), path, name))
    out = open(a.out, "w")
    records = []

    for rnd in range(1, a.rounds + 1):
        # Rotate which binary opens the round. Whatever the host does over the
        # session, each binary spends an equal share of its runs in the leading
        # slot, which is the slot that pays for anything the previous arm left
        # warm or cold.
        shift = (rnd - 1) % len(arms)
        order = arms[shift:] + arms[:shift]
        for width in widths:
            for side, binary, name in order:
                args = Args(
                    server=binary,
                    target=a.target,
                    drafter=a.drafter,
                    port=a.port,
                    n=a.n,
                    max_tokens=a.max_tokens,
                    logdir=a.logdir,
                )
                tag = f"r{rnd}.{side}.w{width}"
                # `run_arm` does `env.setdefault("MLX_CUDA_ARCHITECTURES", "121")`
                # and writes the environment into the arm's server log. The
                # variable is build-time only and inert here, but leaving the
                # default in place would put "121" in the combined arm's log and
                # make the record contradict itself. Set the arm's real value so
                # the setdefault is a no-op.
                os.environ["MLX_CUDA_ARCHITECTURES"] = name
                rec = sweep.run_arm(args, width, prompt, tag)
                rec["round"] = rnd
                rec["side"] = side
                rec["arch_list"] = name
                rec["server_bin"] = binary
                rec["opened_round"] = side == order[0][0]
                records.append(rec)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                runs = rec.get("runs") or []
                toks = [r["e2e_tok_s"] for r in runs]
                print(
                    f"[{tag}] {name}: "
                    + (", ".join(f"{t:.2f}" for t in toks) if toks else rec.get("error", "no runs"))
                    + f"  spec_ran={rec.get('speculative_ran')}"
                    + f"  nvrm_delta={rec.get('nvrm_delta')}"
                    + f"  ci={rec.get('ci_job_running')}",
                    flush=True,
                )
    out.close()

    print("\n=== summary ===")
    for width in widths:
        print(f"-- width {width} --")
        for _, _, name in arms:
            vals = [
                r["e2e_tok_s"]
                for rec in records
                if rec["arch_list"] == name and rec["width"] == width
                for r in (rec.get("runs") or [])
            ]
            if not vals:
                print(f"  {name}: no runs")
                continue
            print(
                f"  {name}: n={len(vals)} min={min(vals):.2f} max={max(vals):.2f} "
                f"mean={statistics.mean(vals):.2f} "
                f"median={statistics.median(vals):.2f}"
            )
        base_name = arms[0][2]
        base = [
            r["e2e_tok_s"]
            for rec in records
            if rec["arch_list"] == base_name and rec["width"] == width
            for r in (rec.get("runs") or [])
        ]
        for _, _, name in arms[1:]:
            vals = [
                r["e2e_tok_s"]
                for rec in records
                if rec["arch_list"] == name and rec["width"] == width
                for r in (rec.get("runs") or [])
            ]
            if not (base and vals):
                continue
            disjoint = max(vals) < min(base) or max(base) < min(vals)
            print(
                f"  {name} / {base_name}: "
                f"ratio of means {statistics.mean(vals) / statistics.mean(base):.4f}, "
                f"ranges disjoint {disjoint}"
            )


if __name__ == "__main__":
    main()
