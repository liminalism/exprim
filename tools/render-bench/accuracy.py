"""All-page four-engine agreement gate, with the 2026-11-06 thresholds.

PNG encoding, raster comparison and hashing run outside every timing series.
"""
from pathlib import Path
from itertools import combinations
import csv, ctypes as C, hashlib, json, shutil, subprocess
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from run import OUT, B, ENGINES, run

KERNEL=None
ARRAY_CACHE={}

def load_kernel():
    global KERNEL
    KERNEL=C.CDLL(str(B/'compare.dll'))
    KERNEL.headsup_compare.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_uint64)]
    KERNEL.headsup_compare.restype=None

def compare(a,b,scale):
    dd=max(abs(a.width-b.width),abs(a.height-b.height))
    if dd>1:return dict(dimension_delta=dd,agrees=False)
    w=max(a.width,b.width);h=max(a.height,b.height)
    def array(im):
        key=(id(im),w,h,scale)
        if key in ARRAY_CACHE:return ARRAY_CACHE[key]
        if im.size!=(w,h):
            padded=Image.new('RGB',(w,h),'white');padded.paste(im,(0,0));im=padded
        if scale==2:im=im.resize(((w+1)//2,(h+1)//2),Image.Resampling.BOX)
        result=np.asarray(im,dtype=np.uint8)
        ARRAY_CACHE[key]=result
        return result
    x=array(a);y=array(b);n=x.shape[0]*x.shape[1];metrics=(C.c_uint64*6)()
    KERNEL.headsup_compare(x.ctypes.data,y.ctypes.data,n,metrics)
    absolute,g,ai,bi,asum,bsum=metrics
    xi=ai/n;yi=bi/n;gross=g/n;ink=abs(xi-yi);continuous=abs(asum-bsum)/(n*3*255)
    blank=max(xi,yi)>.0005 and min(xi,yi)<max(xi,yi)*.05
    return dict(dimension_delta=dd,mean_abs=absolute/(n*3),gross=gross,ink_delta=ink,
                continuous_ink_delta=continuous,a_ink=xi,b_ink=yi,blank_mismatch=blank,
                agrees=not(blank or (ink>.01 and continuous>.003) or gross>.05))

def consensus(available,pairs):
    cliques=[]
    for n in range(3,len(available)+1):
        for subset in combinations(available,n):
            if all(pairs[tuple(sorted((a,b)))]['agrees'] for a,b in combinations(subset,2)):
                cliques.append(set(subset))
    if not cliques:return {e:'failed' if e not in available else 'unresolved' for e in ENGINES},[]
    largest=max(map(len,cliques));camps=[c for c in cliques if len(c)==largest]
    classes={}
    for e in ENGINES:
        if e not in available:classes[e]='failed';continue
        votes=[e in camp or all(pairs[tuple(sorted((e,o)))]['agrees'] for o in camp) for camp in camps]
        classes[e]='ok' if all(votes) else 'suspect' if not any(votes) else 'ambiguous'
    return classes,[sorted(c) for c in camps]

def raster(doc,scale,e,page): return OUT/'rasters'/f's{scale}'/doc['id']/e/f'{page:05}.png'

def review(doc,scale,page,label):
    folder=OUT/'review';folder.mkdir(exist_ok=True)
    images={e:Image.open(raster(doc,scale,e,page)).convert('RGB') for e in ENGINES if raster(doc,scale,e,page).exists()}
    canvas=Image.new('RGB',(1440,860),'#ddd');draw=ImageDraw.Draw(canvas)
    draw.text((8,6),f'{doc["id"]} | page {page+1} | {scale*72} DPI | {label}',fill='black')
    for i,e in enumerate(ENGINES):
        x=i*360;draw.text((x+5,30),e,fill='black')
        if e in images:
            im=images[e];thumb=ImageOps.contain(im,(350,790));canvas.paste(thumb,(x+5,55))
    canvas.save(folder/f'{doc["id"]}__s{scale}__p{page+1:04}.png')
    for im in images.values():im.close()

def score_page(job):
    doc,scale,page=job
    ARRAY_CACHE.clear();images={};hashes={};sizes={};pairrows=[]
    for e in ENGINES:
        p=raster(doc,scale,e,page)
        if p.exists():
            images[e]=Image.open(p).convert('RGB');im=images[e]
            hashes[e]=hashlib.sha256(im.tobytes()).hexdigest();sizes[e]=im.size
    pairs={}
    for a,b in combinations(images,2):
        d=compare(images[a],images[b],scale);pairs[tuple(sorted((a,b)))]=d
        pairrows.append(dict(id=doc['id'],scale=scale,page=page,a=a,b=b,**d))
    classes,camps=consensus(list(images),pairs)
    common=len(images)==4 and all(d['agrees'] for d in pairs.values())
    row=dict(id=doc['id'],scale=scale,page=page,**classes,camps=camps,all_ok=common,
             max_gross=max((d.get('gross',0) for d in pairs.values()),default=0),
             pixel_sha256=hashes,dimensions=sizes)
    for im in images.values():im.close()
    return row,pairrows

def score_doc(doc,scale):
    dest=OUT/'accuracy'/f'{doc["id"]}__s{scale}.json'
    if dest.exists():return json.loads(dest.read_text())
    for e in ENGINES:run(e,doc,scale,8,0,'dump')
    fields=['id','scale','page','a','b','dimension_delta','mean_abs','gross','ink_delta',
            'continuous_ink_delta','a_ink','b_ink','blank_mismatch','agrees']
    rows=[]
    with dest.with_suffix('.pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=load_kernel) as pool:
            for row,pairrows in pool.map(score_page,((doc,scale,p) for p in range(doc['pages'])),chunksize=1):
                writer.writerows(pairrows);rows.append(row)
                if row['page']%100==0:print('SCORE',doc['id'],scale,row['page'],flush=True)
    # Keep thumbnails for a representative page and the worst disagreements.
    selected={0}|{r['page'] for r in sorted((r for r in rows if not r['all_ok']),key=lambda r:-r['max_gross'])[:3]}
    for page in selected:review(doc,scale,page,'representative' if page==0 else 'disagreement')
    dest.write_text(json.dumps(rows),encoding='utf-8')
    # Only our generated raster subtree is removed; metrics and pixel hashes remain.
    target=(OUT/'rasters'/f's{scale}'/doc['id']).resolve()
    assert target.is_relative_to((OUT/'rasters').resolve()) and target!= (OUT/'rasters').resolve()
    shutil.rmtree(target)
    return rows

def main():
    docs=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    missing=[]
    for doc in docs:
        for scale in (1,2):
            for workers in (1,8):
                for e in ENGINES:
                    for rep in range(3):
                        path=OUT/'timing'/f'{e}__{doc["id"]}__s{scale}__w{workers}__timing__r{rep}.json'
                        if not path.exists():missing.append(path.name);continue
                        result=json.loads(path.read_text(encoding='utf-8'))
                        if result['memory']['returncode'] or result.get('error'):
                            missing.append(path.name+' (failed)')
    if missing:raise RuntimeError(f'Timing matrix incomplete: {len(missing)} jobs; first: {missing[0]}')
    import corpus
    corpus.write_mask_census()
    # Calibrate exporters, metric ABI, and child memory before the expensive pass.
    import audit
    audit.probe();audit.color_probe();audit.metric_probe()
    print('CALIBRATION PASSED: child memory, RGB export, native metrics',flush=True)
    load_kernel();(OUT/'accuracy').mkdir(exist_ok=True);rows=[]
    for doc in docs:
        for scale in (1,2):rows.extend(score_doc(doc,scale))
    summary={str(scale):{e:{status:sum(r[e]==status for r in rows if r['scale']==scale)
                           for status in ('ok','suspect','ambiguous','unresolved','failed')}
                        for e in ENGINES} for scale in (1,2)}
    (OUT/'accuracy/summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'accuracy/common-pages.json').write_text(json.dumps({d['id']:{str(s):[r['page'] for r in rows if r['id']==d['id'] and r['scale']==s and r['all_ok']]
                                         for s in (1,2)} for d in json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))}))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
