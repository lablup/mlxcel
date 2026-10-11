import json, glob, re, statistics as st, sys, os
D=sys.argv[1]
def load(f):
    out=[]
    for line in open(f):
        line=line.strip()
        if line.startswith('{'): out.append(json.loads(line))
    return out
med=st.median
print("== (a)/(b) B=1, server defaults, per prompt: decode tok/s median over 5 rounds (on vs off), paired delta median (range), accepted/proposed per request (on)")
for style in ("chat","copy"):
    for q in range(4):
        on=[];off=[];acc=[];prop=[];rounds=[]
        for r in range(1,6):
            a=[x for x in load(f"{D}/ab-on-r{r}.txt") if x["style"]==style and x["rep"]==q]
            b=[x for x in load(f"{D}/ab-off-r{r}.txt") if x["style"]==style and x["rep"]==q]
            if a and b:
                on.append(a[0]["decode_tok_s"][0]); off.append(b[0]["decode_tok_s"][0])
                acc.append(a[0]["draft_n_accepted"]); prop.append(a[0]["draft_n"]); rounds.append(a[0]["draft_rounds"])
        d=[(x/y-1)*100 for x,y in zip(on,off)]
        print(f"  {style} q{q}: on {med(on):.1f} off {med(off):.1f}  delta {med(d):+.2f}% ({min(d):+.2f}..{max(d):+.2f})  accepted {med(acc)}/{med(prop)} in {med(rounds)} rounds")
print("== (c) concurrency, --parallel 8, aggregate tok/s median over 3 rounds")
def bsc(f):
    txt=open(f).read(); res={}
    for block in txt.split("Server:")[1:]:
        style=re.search(r"style (\w+)",block).group(1)
        for m in re.finditer(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",block,re.M):
            res[(style,int(m.group(1)))]=float(m.group(7))
    return res
sides=("off","on","mb8")
for style in ("plain","copy"):
    for c in (2,4,8):
        vals={s:[bsc(f"{D}/c-{s}-r{r}.txt").get((style,c)) for r in (1,2,3)] for s in sides}
        print(f"  bench_serving {style:5} c={c}: "+"  ".join(f"{s} {med(v):.1f} ({min(v):.1f}..{max(v):.1f})" for s,v in vals.items()))
for style in ("chat","copy"):
    for c in (2,4,8):
        vals={};accs={}
        for s in sides:
            xs=[x for r in (1,2,3) for x in load(f"{D}/c-{s}-r{r}.txt") if x["style"]==style and x["conc"]==c]
            vals[s]=[x["agg_tok_s"] for x in xs]; accs[s]=sum(x["draft_n_accepted"] for x in xs)
        print(f"  client {style:5} c={c}: "+"  ".join(f"{s} {med(v):.1f} ({min(v):.1f}..{max(v):.1f}) acc={accs[s]}" for s,v in vals.items()))
print("== (d) ~8192-token prompts, B=1, decode tok/s median over 3 rounds x 3 requests")
for st_ in ("paged","dense"):
    for style in ("long-copy","long-chat"):
        res={}
        for s in ("on","off"):
            xs=[x for r in (1,2,3) for x in load(f"{D}/d-{st_}-{s}-r{r}.txt") if x["style"]==style]
            res[s]=([x["decode_tok_s"][0] for x in xs], sum(x["draft_n_accepted"] for x in xs), sum(x["draft_n"] for x in xs), [x["prompt_n"][0] for x in xs])
        on,off=res["on"][0],res["off"][0]
        print(f"  {st_:5} {style:9}: on {med(on):.1f} ({min(on):.1f}..{max(on):.1f}) off {med(off):.1f} ({min(off):.1f}..{max(off):.1f})  ratio {med(on)/med(off):.3f}  accepted {res['on'][1]}/{res['on'][2]}  prompt_n {res['on'][3][0]}")
