from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json, os, platform, subprocess, sys
import psutil, pymupdf

B=Path(__file__).resolve().parent; ROOT=B.parents[1]; OUT=ROOT/'docs/benchmarks/2026-11-09'
def sha(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def command(args): return subprocess.check_output(args,cwd=ROOT,text=True).strip()
def source_hashes():
    paths=list((ROOT/'src').rglob('*'))+list((ROOT/'fonts').glob('*'))
    paths += [ROOT/p for p in ('Cargo.toml','Cargo.lock','build.rs')]
    for codec in ('jp2lam','jbig2enc-rust'):
        paths += list((ROOT/'codecs'/codec/'src').rglob('*'))+[ROOT/'codecs'/codec/'Cargo.toml']
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(paths) if p.is_file()}

def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'start'
    sources=source_hashes()
    if mode=='finish':
        before=json.loads((OUT/'machine.json').read_text())['source_hashes']
        changed=[p for p in before.keys()|sources.keys() if before.get(p)!=sources.get(p)]
        result=dict(source_changed_during_run=changed,source_file_count=len(sources),
                    completed_at_bangkok=datetime.now(timezone(timedelta(hours=7))).isoformat())
        (OUT/'source-verification.json').write_text(json.dumps(result,indent=2))
        harness=[]
        for pattern in ('*.py','*.rs','*.ps1','*.txt','*.md'):
            harness.extend(B.glob(pattern))
        for adapter in ('exprim','hayro'):
            harness.extend((B/adapter/'src').glob('*.rs'))
            harness.extend([B/adapter/'Cargo.toml',B/adapter/'Cargo.lock'])
        (OUT/'harness-source-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(set(harness))},indent=2))
        print(json.dumps(result));return
    data=dict(date_bangkok=datetime.now(timezone(timedelta(hours=7))).isoformat(),
              platform=platform.platform(),cpu=platform.processor(),ram_bytes=psutil.virtual_memory().total,
              physical_cores=psutil.cpu_count(logical=False),logical_cores=psutil.cpu_count(),
              affinity=list(range(12)),affinity_basis='Same logical CPUs 0-11 as the 2026-11-06 run; not re-enumerated by efficiency class',
              power_scheme=command(['powercfg','/getactivescheme']),
              rustc=command(['rustc','-Vv']),cargo=command(['cargo','-V']),
              git_head=command(['git','rev-parse','HEAD']),
              git_status=command(['git','status','--short']),python=sys.version,python_executable=sys.executable,
              python_packages=command([sys.executable,'-m','pip','freeze']),
              versions=dict(exprim='0.4.0 (extracted working tree)',pdfium='158.0.8125 / chromium/8125',
                            mupdf=pymupdf.version,hayro='0.9.0'),
              build='Rust: release opt-level=3, fat LTO, one codegen unit, panic=unwind; no target-cpu override',
              env={k:v for k,v in os.environ.items() if k.startswith(('RUST','CARGO','PDF_RENDER','RAYON'))},
              source_hashes=sources)
    dll=Path(os.environ['BENCH_PDFIUM_DLL'])
    files=[B/'exprim/target/release/bench-exprim.exe',B/'hayro/target/release/bench-hayro.exe',dll,
           B/'exprim/Cargo.lock',B/'hayro/Cargo.lock']+list((B/'hayro/fonts').glob('*'))
    data['artifact_hashes']={str(p):sha(p) for p in files if p.is_file()}
    (OUT/'machine.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('Wrote machine.json and',len(sources),'source hashes')

if __name__=='__main__':main()
