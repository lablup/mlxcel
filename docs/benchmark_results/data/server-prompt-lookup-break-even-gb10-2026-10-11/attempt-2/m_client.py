"""Measurement client for #2255: sequential or concurrent /completion requests.

Usage: m_client.py PORT STYLE CONCURRENCY REPEATS MAX_TOKENS [PROMPT_TOKENS]
STYLE: chat (open-ended questions, little copying), copy (repeat a paragraph),
long-copy / long-chat (pad the prompt to about PROMPT_TOKENS with a document).
Prints one JSON line per batch: wall time, aggregate tok/s, per-request decode
tok/s (server timings), draft counts.
"""
import http.client
import json
import sys
import threading
import time

QUESTIONS = [
    "Explain in detail how a refrigerator moves heat out of its interior, covering the compressor, condenser, expansion valve and evaporator.",
    "Describe the main causes of the French Revolution and how they interacted, in a few paragraphs.",
    "Write a short story about a cartographer who discovers that one of her maps is changing on its own.",
    "Compare the strengths and weaknesses of trains and airplanes for travel between cities 500 km apart.",
    "Explain how vaccines train the immune system, including the roles of B cells and T cells.",
    "Give practical advice for someone learning to cook at home for the first time, with reasons for each tip.",
    "Describe how a bill becomes law in a parliamentary system, step by step.",
    "Explain what causes the seasons on Earth and why the southern hemisphere has summer in December.",
]
PARA = (
    "The lighthouse on the northern cape was built in 1871 from granite quarried two miles inland. "
    "Its keeper, a retired sailor named Elias Brandt, kept a logbook of every ship that passed, the weather "
    "at dawn and dusk, and the number of gulls nesting on the gallery rail. When the lamp was electrified in "
    "1936, Brandt refused to leave, and he stayed on as a caretaker until his death at the age of ninety-one. "
    "His logbooks, forty-two volumes bound in oilcloth, are now kept in the county archive, where historians "
    "use them to reconstruct shipping traffic and storm patterns along the coast in the late nineteenth century."
)
FILLER = (
    "Section {i}. The committee reviewed the quarterly figures for the harbour authority, noting that "
    "berth occupancy in district {i} rose while dredging costs for channel {j} remained within the budget "
    "approved in the previous session; the treasurer recorded item {k} for follow-up. "
)


def prompt_for(style, idx, prompt_tokens):
    if style == "chat":
        return QUESTIONS[idx % len(QUESTIONS)]
    if style == "copy":
        return "Repeat the following paragraph exactly, word for word.\n\nParagraph:\n" + PARA + "\n\nRepeated paragraph:\n"
    # Long styles: a non-repetitive document of about prompt_tokens tokens
    # (about 57 tokens per filler section on the Qwen3 tokenizer), then the instruction.
    n = max(1, prompt_tokens // 57)
    doc = "".join(FILLER.format(i=i + 1, j=(i * 7) % 97, k=(i * 13) % 101 + idx) for i in range(n))
    if style == "long-copy":
        return doc + "\n\nNow repeat the paragraph below exactly, word for word.\n\nParagraph:\n" + PARA + "\n\nRepeated paragraph:\n"
    return doc + "\n\n" + QUESTIONS[idx % len(QUESTIONS)]


def post(port, prompt, max_tokens):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1200)
    body = json.dumps({"prompt": prompt, "n_predict": max_tokens, "temperature": 0,
                       "cache_prompt": False, "ignore_eos": True})
    conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
    resp = json.loads(conn.getresponse().read())
    conn.close()
    return resp


def main():
    port, style, conc, repeats, max_tokens = (int(sys.argv[1]), sys.argv[2], int(sys.argv[3]),
                                              int(sys.argv[4]), int(sys.argv[5]))
    prompt_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    for rep in range(repeats):
        results = [None] * conc

        def run(i):
            results[i] = post(port, prompt_for(style, i + rep * conc, prompt_tokens), max_tokens)

        t0 = time.perf_counter()
        threads = [threading.Thread(target=run, args=(i,)) for i in range(conc)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        toks = sum(r.get("tokens_predicted", 0) for r in results)
        tm = [r.get("timings", {}) for r in results]
        print(json.dumps({
            "style": style, "conc": conc, "rep": rep, "wall_s": round(wall, 3),
            "tokens": toks, "agg_tok_s": round(toks / wall, 2),
            "decode_tok_s": [round(t.get("predicted_per_second", 0.0), 2) for t in tm],
            "prompt_n": [t.get("prompt_n") for t in tm],
            "draft_kind": tm[0].get("draft_kind"),
            "draft_n": sum((t.get("draft_n") or 0) for t in tm),
            "draft_n_accepted": sum((t.get("draft_n_accepted") or 0) for t in tm),
            "draft_rounds": sum((t.get("draft_rounds") or 0) for t in tm),
        }), flush=True)


if __name__ == "__main__":
    main()
