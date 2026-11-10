"""Continue after a running timing driver, without overlapping timed jobs."""
import argparse, json, subprocess, sys
import psutil
from run import OUT, B, ENGINES, run

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--after-pid',type=int);args=ap.parse_args()
    if args.after_pid:
        try: psutil.Process(args.after_pid).wait()
        except psutil.NoSuchProcess: pass
    # Build the untimed metric kernel only once the timing driver has exited.
    subprocess.run(['rustc','--edition=2024','--crate-type','cdylib','-C','opt-level=3',
                    str(B/'compare_kernel.rs'),'-o',str(B/'compare.dll')],check=True)
    import accuracy
    accuracy.main()
    docs=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    common=json.loads((OUT/'accuracy/common-pages.json').read_text());reuse=[]
    for rep in range(3):
        order=ENGINES[rep%4:]+ENGINES[:rep%4]
        for doc in docs:
            for scale,workers in ((2,1),(1,1),(2,8),(1,8)):
                pages=common[doc['id']][str(scale)]
                if len(pages)==doc['pages']:
                    reuse.append(dict(id=doc['id'],scale=scale,workers=workers,rep=rep,
                                      reason='Every page passed; reuse the identical all-page job'))
                elif pages:
                    for e in order:run(e,doc,scale,workers,rep,'timing',pages,'__common')
    (OUT/'accuracy/gated-reuse.json').write_text(json.dumps(reuse,indent=2))
    for doc in docs:
        pages=sorted({0,doc['pages']//2,doc['pages']-1})
        for scale in (1,2):
            for e in ENGINES:run(e,doc,scale,1,0,'sample',pages)
    subprocess.run([sys.executable,str(B/'provenance.py'),'finish'],check=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
