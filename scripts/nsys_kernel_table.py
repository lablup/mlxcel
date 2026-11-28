#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-kernel and per-API-call duration tables from an Nsight Systems capture.

Input is the SQLite export of a capture taken with `nsys profile --trace=cuda`
(`nsys export --type sqlite -o run.sqlite run.nsys-rep`). Two CSVs come out:

  kernels  one row per (kernel name, grid, block): launch count and the median,
           mean, p90, min and max device duration in microseconds, from
           CUPTI_ACTIVITY_KIND_KERNEL
  api      one row per CUDA API name: call count and the same statistics of
           the host-side call duration, from CUPTI_ACTIVITY_KIND_RUNTIME

Kernels launched inside a CUDA graph do not appear in the kernel table on the
GB10 setup this was written for, so a capture of MLX work has to be taken with
MLX_USE_CUDA_GRAPHS=0 to attribute time to kernels (issue #2834). That is not
the production mode; read the result as kernel durations, not as throughput.

  nsys_kernel_table.py run.sqlite --kernels k.csv --api a.csv [--label TEXT]
      [--match SUBSTRING ...] [--skip-first N]

--match keeps kernels whose name contains any of the substrings. --skip-first
drops the first N launches of every (name, grid, block) group, which removes
warm-up launches when the profiled program runs them first.
"""
import argparse
import csv
import sqlite3
import statistics
import sys


def columns(db, table):
    return [row[1] for row in db.execute(f"PRAGMA table_info({table})")]


def stats(values_us):
    values = sorted(values_us)
    n = len(values)
    p90 = values[max(0, min(n - 1, -(-n * 9 // 10) - 1))]
    return {
        "count": n,
        "median_us": f"{statistics.median(values):.3f}",
        "mean_us": f"{statistics.fmean(values):.3f}",
        "p90_us": f"{p90:.3f}",
        "min_us": f"{values[0]:.3f}",
        "max_us": f"{values[-1]:.3f}",
    }


def kernel_rows(db, label, match, skip_first):
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    cols = columns(db, table)
    if not cols:
        return []
    name_col = "shortName" if "shortName" in cols else "demangledName"
    query = (f"SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, "
             f"k.start, k.end FROM {table} k JOIN StringIds s ON s.id = k.{name_col} "
             f"ORDER BY k.start")
    groups = {}
    for name, gx, gy, gz, bx, by, bz, start, end in db.execute(query):
        if match and not any(m in name for m in match):
            continue
        groups.setdefault((name, gx, gy, gz, bx, by, bz), []).append((end - start) / 1000.0)
    rows = []
    for (name, gx, gy, gz, bx, by, bz), durations in groups.items():
        kept = durations[skip_first:]
        if not kept:
            continue
        rows.append({"label": label, "kernel": name, "grid": f"{gx}x{gy}x{gz}",
                     "block": f"{bx}x{by}x{bz}", "launches_total": len(durations),
                     "skipped_first": skip_first, **stats(kept)})
    rows.sort(key=lambda r: (r["kernel"], r["grid"], r["block"]))
    return rows


def api_rows(db, label):
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    cols = columns(db, table)
    if not cols:
        return []
    query = (f"SELECT s.value, r.start, r.end FROM {table} r "
             f"JOIN StringIds s ON s.id = r.nameId")
    groups = {}
    for name, start, end in db.execute(query):
        groups.setdefault(name, []).append((end - start) / 1000.0)
    rows = [{"label": label, "api": name, **stats(durations)} for name, durations in groups.items()]
    rows.sort(key=lambda r: -r["count"])
    return rows


def write(path, rows):
    if not rows:
        sys.exit(f"nsys_kernel_table: nothing to write to {path} (empty table)")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite")
    ap.add_argument("--kernels", help="CSV for the per-kernel table")
    ap.add_argument("--api", help="CSV for the per-API-call table")
    ap.add_argument("--label", default="")
    ap.add_argument("--match", action="append", default=[])
    ap.add_argument("--skip-first", type=int, default=0)
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    kernels = kernel_rows(db, a.label, a.match, a.skip_first)
    for row in kernels:
        print(f"{row['kernel']:<48} grid {row['grid']:<10} block {row['block']:<9} "
              f"n={row['count']:<6} median {row['median_us']:>9} us  mean {row['mean_us']:>9}  "
              f"p90 {row['p90_us']:>9}")
    if a.kernels:
        write(a.kernels, kernels)
    if a.api:
        write(a.api, api_rows(db, a.label))


if __name__ == "__main__":
    main()
