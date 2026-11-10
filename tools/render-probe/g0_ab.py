import json,statistics,sys
def load(p):
    out={}; meta=None; hashes={}
    for l in open(p,encoding="utf-8"):
        r=json.loads(l)
        if r["kind"]=="meta": meta=r
        elif r["kind"]=="page":
            base,_,rnd=r["id"].partition("#")
            hashes.setdefault(base,set()).add((r["page"],r.get("hash")))
            if r["phase"]=="warm" or r["scenario"] in ("seq","cold"):
                out.setdefault((base,rnd),{}).setdefault(r["page"],[]).append(r["render_cycles"]/meta["cycles_per_ns"]/1e6)
    # per round: sum over pages of the per-page minimum
    return {k:sum(min(v) for v in pages.values()) for k,pages in out.items()}, hashes
a,ha=load(sys.argv[1]); b,hb=load(sys.argv[2])
ids=sorted({k[0] for k in a})
for i in ids:
    ra=[a[k] for k in sorted(a) if k[0]==i]; rb=[b[k] for k in sorted(a) if k[0]==i and k in b]
    ratios=[y/x for x,y in zip(ra,rb)]
    same = ha.get(i)==hb.get(i)
    print(f"{i:22s} A med {statistics.median(ra):9.2f} ms  B med {statistics.median(rb):9.2f} ms  B/A med {statistics.median(ratios):.3f} [{min(ratios):.3f}..{max(ratios):.3f}]  B faster {sum(r<1 for r in ratios)}/{len(ratios)}  hashes {'same' if same else 'DIFFER'}")
