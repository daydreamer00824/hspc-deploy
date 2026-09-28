import sqlite3, sys, json
c=sqlite3.connect(sys.argv[1])
names=("prep","patch_build","hsi_infer","pc_infer","assemble","match")
rng=[(t,s,e) for s,e,t in c.execute("select start,end,text from NVTX_EVENTS where end is not null and text is not null") if t in names or t.startswith("scene_")]
ker=[(a,b) for a,b in c.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL")]
mem=[(a,b) for a,b in c.execute("select start,end from CUPTI_ACTIVITY_KIND_MEMCPY")]
def busy(iv,s,e):
    iv=sorted((max(a,s),min(b,e)) for a,b in iv if b>s and a<e); tot=0; cs=ce=None
    for a,b in iv:
        if ce is None or a>ce:
            if ce is not None: tot+=ce-cs
            cs,ce=a,b
        else: ce=max(ce,b)
    if ce is not None: tot+=ce-cs
    return tot
out={}
print("range          wall_ms kernel_ms memcpy_ms gpu_busy_ms gpu_busy/wall n_kernels")
for n,s,e in sorted(rng,key=lambda x:x[1]):
    k=busy(ker,s,e)/1e6; m=busy(mem,s,e)/1e6; u=busy(ker+mem,s,e)/1e6; w=(e-s)/1e6
    nk=sum(1 for a,b in ker if a>=s and b<=e)
    out[n]={"wall_ms":w,"kernel_busy_ms":k,"memcpy_busy_ms":m,"gpu_busy_ms":u,"n_kernels":nk}
    print("%-13s %8.1f %9.1f %9.1f %11.1f %12.1f%% %9d"%(n,w,k,m,u,100*u/w,nk))
json.dump(out,open(sys.argv[2],"w"),indent=2)
