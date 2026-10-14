#!/usr/bin/env python3
"""Drive concurrency levels and record per-request CLOCK_MONOTONIC stamps.

time.perf_counter() is CLOCK_MONOTONIC on Linux, the clock rocprofv3 stamps
dispatches with, so the stamps cut the kernel trace into windows:
all-decode window of a level = [max(first token), min(end)].
"""
import asyncio
import json
import sys
import time

sys.path.insert(0, "scripts")
import bench_serving_concurrency as b  # noqa: E402


def main() -> None:
    port, out = int(sys.argv[1]), sys.argv[2]
    levels = [tuple(map(int, c.split(":"))) for c in sys.argv[3:]]
    model = b.resolve_model("127.0.0.1", port, None)
    res = []
    for conc, pt in levels:
        prompt = b.build_prompt(pt)
        before = b.scrape_paths("127.0.0.1", port)
        loop_t0 = time.perf_counter()

        async def go():
            loop = asyncio.get_running_loop()
            return await asyncio.gather(*[
                loop.run_in_executor(None, b.stream_request, "127.0.0.1", port, model,
                                     prompt, 128, 600.0, "")
                for _ in range(conc)
            ])

        rs = asyncio.run(go())
        after = b.scrape_paths("127.0.0.1", port)
        d = b.batch_delta(before, after)
        res.append({
            "conc": conc, "prompt_tokens": pt, "t0_ns": int(loop_t0 * 1e9),
            "requests": [{
                "start_ns": int(r.start_s * 1e9),
                "first_ns": int((r.start_s + (r.ttft_s or 0)) * 1e9),
                "end_ns": int(r.end_s * 1e9),
                "tokens": r.completion_tokens, "decode_tok_s": r.decode_tok_s,
            } for r in rs],
            "delta": None if d is None else d.__dict__,
            "lines": b.format_batch_delta(d),
        })
        time.sleep(2)
    json.dump(res, open(out, "w"), indent=1)
    for r in res:
        print(r["conc"], r["prompt_tokens"], *r["lines"], sep="\n")


if __name__ == "__main__":
    main()
