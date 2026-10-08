import sys, collections, pathlib
sys.path.insert(0,"scripts")
import rocm_decode_profile as rdp
ds=rdp.read_trace(pathlib.Path(sys.argv[1]))
batch=int(sys.argv[2]) if len(sys.argv)>2 else 4
# the batched decode follows the last Tensile GEMM (the batched pass prefill)
first=max(i for i,d in enumerate(ds) if "Cijk" in d.name)+1
dec=ds[first:]
lo,hi=dec[0].start,dec[-1].end
busy=rdp.busy_ns(dec,lo,hi)
nn=collections.Counter(); names=collections.Counter(); c=collections.Counter()
for d in dec:
    s=rdp.short_kernel(d.name); nn[s]+=1; names[s]+=d.dur; c[rdp.classify(d.name)]+=d.dur
sdpa=sum(v for k,v in nn.items() if "sdpav" in k)
steps=sdpa/(32*batch)
print("steps %.1f  per-step wall %.1f ms  GPU busy %.1f ms (%.1f%%)  host gap %.1f ms"%(steps,(hi-lo)/1e6/steps,busy/1e6/steps,100*busy/(hi-lo),(hi-lo-busy)/1e6/steps))
tot=sum(c.values())
for k,v in c.most_common(5): print("  class %-10s %.1f ms/step %.1f%%"%(k,v/1e6/steps,100*v/tot))
for k,v in names.most_common(6): print("  %.2f ms/step n/step=%.1f %s"%(v/1e6/steps,nn[k]/steps,k[:90]))
