"""Validate the retained evidence, including the original missing-memory case."""
import argparse, json, math, os, sys, tempfile, subprocess, hashlib
from pathlib import Path
from run import OUT, ENGINES, commands, environment
from winjob import measured

def load(p):return json.loads(p.read_text(encoding='utf-8'))

def probe():
    # The allocated memory lives only in a descendant, not the launched parent.
    child="import time; a=bytearray(128*1024*1024); a[::4096]=b'x'*(len(a)//4096); time.sleep(.5)"
    parent=f"import subprocess,sys; subprocess.run([sys.executable,'-c',{child!r}],check=True)"
    with tempfile.TemporaryFile(mode='w+') as stdout,tempfile.TemporaryFile(mode='w+') as stderr:
        result=measured([sys.executable,'-c',parent],os.environ.copy(),stdout,stderr,str(OUT),timeout=30)
    assert result['returncode']==0 and result['max_processes']>=2
    assert result['peak_job_commit_bytes']>128*2**20
    assert result['peak_tree_rss_sampled_bytes']>128*2**20
    (OUT/'memory-probe.json').write_text(json.dumps(result,indent=2))
    return result

def color_probe():
    import pymupdf
    from PIL import Image
    pixels={}
    with tempfile.TemporaryDirectory(prefix='exprim-color-probe-') as tmp:
        folder=Path(tmp);path=folder/'red.pdf'
        doc=pymupdf.open();page=doc.new_page(width=72,height=72)
        page.draw_rect(pymupdf.Rect(8,8,64,64),color=(1,0,0),fill=(1,0,0))
        doc.save(path);doc.close()
        for e in ENGINES:
            out=folder/e
            result=subprocess.run(commands()[e]+[str(path),'1','1','dump',str(out),'0'],
                                  env=environment(1),capture_output=True,text=True,check=True,
                                  creationflags=0x08000000)
            with Image.open(out/'00000.png') as im:
                pixel=im.convert('RGB').getpixel((32,32));pixels[e]=pixel
                assert pixel==(255,0,0),(e,pixel,'channel-order calibration failed')
    (OUT/'color-probe.json').write_text(json.dumps(pixels,indent=2))

def metric_probe():
    import ctypes as C
    import numpy as np
    import accuracy
    accuracy.load_kernel()
    rng=np.random.default_rng(20261109)
    a=rng.integers(0,256,(1024,3),dtype=np.uint8);b=rng.integers(0,256,(1024,3),dtype=np.uint8)
    result=(C.c_uint64*6)();accuracy.KERNEL.headsup_compare(a.ctypes.data,b.ctypes.data,len(a),result)
    delta=np.abs(a.astype(np.int32)-b.astype(np.int32))
    expected=[int(delta.sum()),int((delta.max(axis=1)>48).sum()),
              int((a.astype(np.int32).sum(axis=1)<600).sum()),
              int((b.astype(np.int32).sum(axis=1)<600).sum()),int(a.sum()),int(b.sum())]
    assert list(result)==expected
    (OUT/'metric-probe.json').write_text(json.dumps(dict(native=list(result),reference=expected,pass_check=True),indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');args=ap.parse_args()
    docs=load(OUT/'manifest.json');n=sum(d['pages'] for d in docs)
    checks=[];issues=[];raw=[];pages=[];common_jobs=[]
    assert n==3871 and len(docs)==9
    for d in docs:
        with open(d['path'],'rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert digest==d['sha256'],f'Input changed: {d["path"]}'
    for d in docs:
        for scale in (1,2):
            quality=load(OUT/'accuracy'/f'{d["id"]}__s{scale}.json')
            assert len(quality)==d['pages'] and {r['page'] for r in quality}==set(range(d['pages']))
            pages.extend(quality)
            for row in quality:
                if set(row['pixel_sha256'])!=set(ENGINES):
                    issues.append(f'{d["id"]} scale={scale} page={row["page"]}: missing renderer raster')
                assert all(len(v)==64 for v in row['pixel_sha256'].values())
            for workers in (1,8):
                for e in ENGINES:
                    for rep in range(3):
                        p=OUT/'timing'/f'{e}__{d["id"]}__s{scale}__w{workers}__timing__r{rep}.json'
                        result=load(p);raw.append(result);m=result['memory']
                        assert m['returncode']==0 and not m['timed_out']
                        assert result['count']==d['pages'] and result['selected_count']==d['pages']
                        if result['ok']!=d['pages']:issues.append(f'{p.name}: {result["ok"]}/{d["pages"]} rendered')
                        assert result['wall_ms']>0 and m['peak_job_commit_bytes']>0 and m['peak_tree_rss_sampled_bytes']>0
                        assert m['sum_process_peak_wset_bytes']>=m['peak_tree_rss_sampled_bytes']
                        assert m['max_processes']>=9 if workers==8 and e in ('pdfium','mupdf') else m['max_processes']>=1
                        if workers==8 and e in ('pdfium','mupdf'):
                            assert sum(name.lower()=='python.exe' for name in m['process_names'].values())>=9
    assert len(raw)==432 and len(pages)==7742
    assert not load(OUT/'source-verification.json')['source_changed_during_run']
    common=load(OUT/'accuracy/common-pages.json')
    for d in docs:
        for s in (1,2):
            eligible=common[d['id']][str(s)]
            assert eligible==[r['page'] for r in pages if r['id']==d['id'] and r['scale']==s and r['all_ok']]
            if 0<len(eligible)<d['pages']:
                for w in (1,8):
                    for e in ENGINES:
                        for rep in range(3):
                            result=load(OUT/'timing'/f'{e}__{d["id"]}__s{s}__w{w}__timing__r{rep}__common.json')
                            common_jobs.append(result)
                            assert result['selected_count']==len(eligible) and result['ok']==len(eligible)
                            assert result['memory']['returncode']==0
                            assert list(map(int,result['command'][-1].split(',')))==eligible
                            if result['rows']:
                                assert {r['page'] for r in result['rows'] if r['ok']}==set(eligible)
    samples=list((OUT/'sample').glob('*.json'));assert len(samples)==72
    for p in samples:
        result=load(p)
        for page in {r['page'] for r in result['rows']}:
            rows=[r for r in result['rows'] if r['page']==page]
            assert len(rows)==26 and sum(r['phase']=='warm' for r in rows)==20
            if not all(r['ok'] for r in rows):issues.append(f'{p.name}: page {page} sample failure')
    if args.probe:
        probe();color_probe();metric_probe()
    calibration=load(OUT/'memory-probe.json')
    assert calibration['returncode']==0 and calibration['peak_job_commit_bytes']>128*2**20
    assert all(rgb==[255,0,0] for rgb in load(OUT/'color-probe.json').values())
    assert load(OUT/'metric-probe.json')['pass_check']
    result=dict(status='pass' if not issues else 'completed-with-render-failures',issues=issues,
                documents=len(docs),pages=n,raw_jobs=len(raw),common_rerun_jobs=len(common_jobs),
                physical_timing_jobs=len(raw)+len(common_jobs),accuracy_page_dpi_rows=len(pages),
                raster_hashes=sum(len(r['pixel_sha256']) for r in pages),sample_jobs=len(samples),
                pdfium_memory_present=True,mupdf_memory_present=True,source_stable=True,inputs_stable=True,
                max_sampler_gap_ms=max(r['memory']['max_sample_gap_ms'] for r in raw),
                pre_job_cpu_percent=dict(min=min(r['before']['cpu_percent'] for r in raw),
                                         max=max(r['before']['cpu_percent'] for r in raw)))
    (OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
