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
"""Teacher-forced decode-step logit trace through a running mlxcel server.

`examples/logit_trace` is teacher-forced but decodes through a dense cache,
so it cannot observe the server's paged decode kernels. This driver asks a
running server the same question through `/completion` with `n_probs`: it
records, per generated position, the top-k token ids and log-probabilities
the server's decode step produced, in the TSV shape `logit_trace` writes, so
`scripts/compare_logit_traces.py` can count decided-position mismatches
between a dense-storage server and a paged-storage server (issue #2192).

Reference mode (no `--ref`): one greedy request of `--n` tokens; every row is
a decode step of the server (position 0 comes from the prefill's last row, as
it does in every free-running decode). The generated ids are the reference
stream.

Candidate mode (`--ref REF.tsv`): follows the reference stream. A greedy
request runs until its own token differs from the reference; the diverging
position is still a valid observation (both arms saw the same context), and
the run restarts from `prompt + reference[:pos+1]`, rebuilt as text through
`/detokenize` and checked through `/tokenize` to re-tokenize to exactly the
same ids. The first token after a restart comes from the prefill's last row,
not from a decode step, so those positions are listed in `# prefill_rows`,
and `<out>.decode_only.tsv` plus `<ref>.decode_only.tsv` hold both traces
with those positions removed.

Usage:
    paged_vs_dense_trace.py --url http://127.0.0.1:PORT --prompt-file F \
        --n 128 --n-probs 10 --out dense.tsv
    paged_vs_dense_trace.py --url http://127.0.0.1:PORT2 --prompt-file F \
        --n 128 --n-probs 10 --ref dense.tsv --out paged.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def post(url: str, path: str, body: dict, timeout: float = 3600.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def tokenize(url: str, text: str) -> list[int]:
    return [int(t) for t in post(url, "/tokenize", {"content": text, "add_special": False})["tokens"]]


def detokenize(url: str, ids: list[int]) -> str:
    return post(url, "/detokenize", {"tokens": ids})["content"]


def completion(url: str, prompt: str, n_predict: int, n_probs: int) -> dict:
    return post(
        url,
        "/completion",
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "n_probs": n_probs,
            "return_tokens": True,
            "cache_prompt": False,
            "ignore_eos": True,
        },
    )


def topk_of(entry: dict, n_probs: int) -> tuple[list[int], list[float]]:
    """Top ids and log-probs of one `completion_probabilities` entry, sorted."""
    scores: dict[int, float] = {}
    for alt in entry.get("top_logprobs", []):
        scores[int(alt["id"])] = float(alt["logprob"])
    scores[int(entry["id"])] = float(entry["logprob"])
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:n_probs]
    return [i for i, _ in ordered], [v for _, v in ordered]


def load_ref(path: str) -> tuple[dict, list[int]]:
    meta, stream = {}, []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                parts = line[1:].strip().split("\t")
                if len(parts) >= 2:
                    meta[parts[0]] = parts[1:]
                continue
            stream.append(int(line.split("\t")[2]))
    return meta, stream


def write_rows(path: str, header: list[str], rows: list[tuple]) -> None:
    with open(path, "w") as out:
        for line in header:
            out.write(line + "\n")
        for chunk, pos, target, nll, ids, lgs in rows:
            out.write(
                f"{chunk}\t{pos}\t{target}\t{nll:.6f}\t"
                f"{','.join(str(i) for i in ids)}\t{','.join(f'{v:.6f}' for v in lgs)}\n"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--n-probs", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", help="reference trace to teacher-force along")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    with open(args.prompt_file) as fh:
        prompt_text = fh.read()
    prompt_ids = tokenize(args.url, prompt_text)
    props = json.loads(urllib.request.urlopen(args.url.rstrip("/") + "/props", timeout=60).read())
    model = props.get("model_path") or props.get("default_generation_settings", {}).get("model", "?")

    header = [
        f"# model\t{model}",
        f"# label\t{args.label}",
        f"# prompt_tokens\t{len(prompt_ids)}",
        "# columns\tchunk\tpos\ttarget\tnll\ttop_ids\ttop_logits",
    ]
    rows: list[tuple] = []

    if not args.ref:
        resp = completion(args.url, prompt_text, args.n, args.n_probs)
        if int(resp.get("tokens_evaluated", -1)) != len(prompt_ids):
            print(
                f"prompt tokenized to {resp.get('tokens_evaluated')} tokens on the server, "
                f"{len(prompt_ids)} through /tokenize",
                file=sys.stderr,
            )
        probs = resp["completion_probabilities"]
        stream = [int(t) for t in resp["tokens"]]
        for pos, (tok, entry) in enumerate(zip(stream, probs)):
            ids, lgs = topk_of(entry, args.n_probs)
            rows.append((0, pos, tok, -float(entry["logprob"]), ids, lgs))
        header.append(f"# stream\t{','.join(str(t) for t in stream)}")
        write_rows(args.out, header, rows)
        print(f"reference: {len(rows)} positions, {len(prompt_ids)} prompt tokens", file=sys.stderr)
        return 0

    ref_meta, ref_stream = load_ref(args.ref)
    n = min(args.n, len(ref_stream))
    pos = 0
    restarts: list[int] = []
    prefill_rows: list[int] = []
    own_stream: list[int] = []
    first_divergence = None
    while pos < n:
        prefix_ids = prompt_ids + ref_stream[:pos]
        if pos == 0:
            prompt = prompt_text
        else:
            prompt = detokenize(args.url, prefix_ids)
            check = tokenize(args.url, prompt)
            if check != prefix_ids:
                print(
                    f"restart at {pos}: the rebuilt text re-tokenizes to {len(check)} ids, "
                    f"expected {len(prefix_ids)}; stopping here",
                    file=sys.stderr,
                )
                break
            restarts.append(pos)
            prefill_rows.append(pos)
        resp = completion(args.url, prompt, n - pos, args.n_probs)
        probs = resp["completion_probabilities"]
        got = [int(t) for t in resp["tokens"]]
        if pos == 0:
            own_stream = got
        advanced = False
        for j, (tok, entry) in enumerate(zip(got, probs)):
            p = pos + j
            if p >= n:
                break
            ids, lgs = topk_of(entry, args.n_probs)
            target = ref_stream[p]
            nll = -float(entry["logprob"]) if tok == target else float("nan")
            if target in ids:
                nll = -lgs[ids.index(target)]
            rows.append((0, p, target, nll, ids, lgs))
            if tok != target:
                if first_divergence is None:
                    first_divergence = (p, target, tok)
                pos = p + 1
                advanced = True
                break
        if not advanced:
            pos += len(got)
            if not got:
                print(f"server returned no tokens at position {pos}; stopping", file=sys.stderr)
                break

    header.append(f"# ref_stream\t{','.join(str(t) for t in ref_stream[:n])}")
    header.append(f"# own_stream\t{','.join(str(t) for t in own_stream)}")
    header.append(
        "# first_divergence\t"
        + ("identical" if first_divergence is None else f"{first_divergence[0]}\tref={first_divergence[1]}\tcand={first_divergence[2]}")
    )
    header.append(f"# restarts\t{','.join(str(r) for r in restarts)}")
    header.append(f"# prefill_rows\t{','.join(str(r) for r in prefill_rows)}")
    write_rows(args.out, header, rows)

    # Decode-only copies: both traces without the positions a restart turned
    # into a prefill row on the candidate side.
    drop = set(prefill_rows)
    keep = [r for r in rows if r[1] not in drop]
    kept_positions = {r[1] for r in keep}
    stem = args.out[:-4] if args.out.endswith(".tsv") else args.out
    write_rows(stem + ".decode_only.tsv", header, keep)
    ref_rows = []
    with open(args.ref) as fh:
        ref_header = []
        for line in fh:
            if line.startswith("#"):
                ref_header.append(line.rstrip("\n"))
                continue
            c, p, t, nll, ids, lgs = line.rstrip("\n").split("\t")
            if int(p) not in kept_positions:
                continue
            ref_rows.append((int(c), int(p), int(t), float(nll), [int(x) for x in ids.split(",")], [float(x) for x in lgs.split(",")]))
    ref_stem = args.ref[:-4] if args.ref.endswith(".tsv") else args.ref
    write_rows(ref_stem + ".decode_only.tsv", ref_header, ref_rows)
    print(
        f"candidate: {len(rows)} positions, first divergence "
        f"{'none' if first_divergence is None else first_divergence}, "
        f"{len(restarts)} restarts (prefill rows {prefill_rows})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
