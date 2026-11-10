"""Post-timing diagnostic: fresh-process JBIG2 prefixes, no comparative claims."""
import json
from run import OUT, B, commands, environment
from winjob import measured

def main():
    doc=next(d for d in json.loads((OUT/'manifest.json').read_text(encoding='utf-8')) if d['id']=='jbig2')
    folder=OUT/'memory-scaling';folder.mkdir(exist_ok=True);summary=[]
    for count in (32,64,128,256):
        dest=folder/f'exprim-jbig2-prefix-{count}.json'
        if dest.exists():result=json.loads(dest.read_text())
        else:
            args=commands()['exprim']+[doc['path'],'1','1','timing',str(folder),','.join(map(str,range(count)))]
            with dest.with_suffix('.stdout').open('w') as stdout,dest.with_suffix('.stderr').open('w') as stderr:
                memory=measured(args,environment(1),stdout,stderr,str(B),timeout=300)
            result=json.loads(dest.with_suffix('.stdout').read_text().splitlines()[-1])
            result.update(memory=memory,command=args,purpose='Memory-only fresh-process consecutive-prefix diagnostic')
            assert memory['returncode']==0 and result['selected_count']==count and result['ok']==count
            dest.write_text(json.dumps(result))
        row=dict(pages=count,peak_job_commit_mib=result['memory']['peak_job_commit_bytes']/2**20,
                 peak_tree_rss_sampled_mib=result['memory']['peak_tree_rss_sampled_bytes']/2**20)
        summary.append(row);print(json.dumps(row),flush=True)
    (OUT/'memory-scaling.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
