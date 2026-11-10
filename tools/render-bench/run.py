"""Rotated, isolated whole-document benchmark. See README.md for setup."""
from pathlib import Path
import argparse, hashlib, json, os, platform, subprocess, sys, time
import psutil
from winjob import measured

B=Path(__file__).resolve().parent
ROOT=B.parents[1]
OUT=ROOT/'docs/benchmarks/2026-11-09'
ENGINES=['exprim','pdfium','mupdf','hayro']

def commands():
    return {
        'exprim':[str(B/'exprim/target/release/bench-exprim.exe')],
        'hayro':[str(B/'hayro/target/release/bench-hayro.exe'),'hayro'],
        'pdfium':[sys.executable,str(B/'native_worker.py'),'pdfium'],
        'mupdf':[sys.executable,str(B/'native_worker.py'),'mupdf'],
    }

def environment(workers):
    env=os.environ.copy()
    env.update(RUST_BACKTRACE='0',PDF_RENDER_THREADS='1',RAYON_NUM_THREADS=str(workers),
               PDF_RENDERER_JPX_CONCURRENCY='serial',PDF_RENDERER_IMAGE_ROW_PAR='0')
    return env

def run(engine,doc,scale,workers,rep,mode='timing',pages=None,suffix=''):
    key=f'{engine}__{doc["id"]}__s{scale}__w{workers}__{mode}__r{rep}{suffix}'
    folder=OUT/('smoke' if suffix=='__smoke' else 'timing' if mode=='timing' else mode);folder.mkdir(exist_ok=True)
    dest=folder/(key+'.json')
    if dest.exists(): return json.loads(dest.read_text())
    raster=OUT/'rasters'/f's{scale}'/doc['id']/engine
    args=commands()[engine]+[doc['path'],str(scale),str(workers),mode,str(raster)]
    if pages is not None: args.append(','.join(map(str,pages)))
    # Prime the filesystem cache equally; this is outside the measured child.
    with open(doc['path'],'rb') as f:
        while f.read(8*1024*1024): pass
    before=dict(cpu_percent=psutil.cpu_percent(interval=1),available_bytes=psutil.virtual_memory().available)
    with dest.with_suffix('.stdout').open('w',encoding='utf-8') as stdout, dest.with_suffix('.stderr').open('w',encoding='utf-8') as stderr:
        memory=measured(args,environment(workers),stdout,stderr,str(B))
    lines=dest.with_suffix('.stdout').read_text(encoding='utf-8').splitlines()
    try: data=json.loads(lines[-1])
    except (IndexError,json.JSONDecodeError): data={'error':'no result JSON','ok':0}
    data.update(engine=engine,id=doc['id'],scale=scale,workers=workers,rep=rep,
                mode=mode,command=args,memory=memory,before=before)
    dest.write_text(json.dumps(data),encoding='utf-8')
    rate=1000*data.get('ok',0)/data.get('wall_ms',float('inf'))
    print(f'{key} ok={data.get("ok")}/{data.get("selected_count")} {rate:.2f} p/s '
          f'RSS={memory["peak_tree_rss_sampled_bytes"]/2**20:.1f} MiB '
          f'commit={memory["peak_job_commit_bytes"]/2**20:.1f} MiB '
          f'processes={memory["max_processes"]} exit={memory["returncode"]}',flush=True)
    if memory['returncode'] or 'error' in data: raise RuntimeError(f'Failed {key}; see {dest}')
    return data

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=3)
    ap.add_argument('--only',nargs='*');ap.add_argument('--engines',nargs='*',default=ENGINES)
    ap.add_argument('--mode',choices=['timing','dump','sample','verify'],default='timing')
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    docs=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    if args.only: docs=[d for d in docs if d['id'] in args.only]
    for rep in range(args.reps):
        order=args.engines[rep%len(args.engines):]+args.engines[:rep%len(args.engines)]
        for doc in docs:
            cells=[(2,1),(1,1),(2,8),(1,8)] if args.mode=='timing' else [(1,1),(2,1)]
            for scale,workers in cells:
                for engine in order:
                    pages=sorted({0,doc['pages']//2,doc['pages']-1}) if args.smoke or args.mode=='sample' else None
                    run(engine,doc,scale,workers,rep,args.mode,pages,'__smoke' if args.smoke else '')
    print('ALLDONE',flush=True)

if __name__=='__main__':main()
