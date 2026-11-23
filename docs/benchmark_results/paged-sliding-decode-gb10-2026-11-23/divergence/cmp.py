import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for i, (x, y) in enumerate(zip(a, b)):
    lx, ly = x["choices"][0]["logprobs"], y["choices"][0]["logprobs"]
    tx, ty = lx["tokens"], ly["tokens"]
    n = next((j for j in range(min(len(tx), len(ty))) if tx[j] != ty[j]), None)
    print("row", i, "finish", x["choices"][0]["finish_reason"], y["choices"][0]["finish_reason"], "len", len(tx), len(ty), "first diff at", n)
    if n is not None:
        for j in range(max(0, n-1), n+1):
            print("  step", j, "A", repr(tx[j]), lx["token_logprobs"][j], lx["top_logprobs"][j], "\n         B", repr(ty[j]), ly["token_logprobs"][j], ly["top_logprobs"][j])
    # max logprob delta on common prefix
    m = n if n is not None else len(tx)
    d = max((abs(lx["token_logprobs"][j]-ly["token_logprobs"][j]) for j in range(m)), default=0)
    print("  max |dlogprob| over common prefix", round(d, 5))
