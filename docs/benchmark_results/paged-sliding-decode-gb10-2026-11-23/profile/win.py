import sqlite3, sys, collections
N=400
for f in sys.argv[1:]:
    c=sqlite3.connect(f+".sqlite")
    ar=[r[0] for r in c.execute("select k.start from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where s.value like 'arg_reduce%' order by k.start")]
    t0,t1=ar[-N-1],ar[-1]
    ks=list(c.execute("select s.value,k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>=? and k.start<? order by k.start",(t0,t1)))
    busy=0; cur_s=cur_e=None
    for _,s,e in ks:
        if cur_e is None or s>cur_e:
            if cur_e is not None: busy+=cur_e-cur_s
            cur_s,cur_e=s,e
        else: cur_e=max(cur_e,e)
    busy+=cur_e-cur_s
    agg=collections.defaultdict(lambda:[0,0])
    for n,s,e in ks: agg[n][0]+=1; agg[n][1]+=e-s
    def api(name):
        r=c.execute("select count(*), sum(r.end-r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like ? and r.start>=? and r.start<?",(name,t0,t1)).fetchone()
        return f"{r[0]/N:.1f}/{(r[1] or 0)/N/1e3:.0f}us"
    wall=(t1-t0)/N/1e3
    print(f"{f}: wall/step={wall:.0f}us gpu_busy/step={busy/N/1e3:.0f}us kern_sum/step={sum(v[1] for v in agg.values())/N/1e3:.0f}us kernels/step={len(ks)/N:.0f} graphLaunch={api('cudaGraphLaunch%')} addNode={api('cudaGraphAddKernelNode%')} execUpdate={api('cudaGraphExecUpdate%')} evSync={api('cudaEventSynchronize')} malloc={api('cudaMallocAsync%')} memcpy={api('cudaMemcpy%')}")
    for n,(cnt,t) in sorted(agg.items(),key=lambda x:-x[1][1])[:12]:
        print(f"   {cnt/N:5.1f}/step {t/N/1e3:7.1f}us/step  {n[:70]}")
