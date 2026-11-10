"""Generate reviewable tables from raw jobs; never fill missing cells."""
from pathlib import Path
from statistics import median, geometric_mean, quantiles
import csv, hashlib, json, math
from run import OUT, B, ENGINES

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def percentile(xs,q):
    xs=sorted(xs);x=(len(xs)-1)*q;i=int(x);f=x-i
    return xs[i]*(1-f)+xs[min(i+1,len(xs)-1)]*f
def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    docs=load(OUT/'manifest.json');accuracy=load(OUT/'accuracy/summary.json')
    common=load(OUT/'accuracy/common-pages.json');cells=[];flat=[];jobs=[]
    for doc in docs:
        for scale in (1,2):
            for workers in (1,8):
                for e in ENGINES:
                    for kind in ('raw','common'):
                        n=doc['pages'] if kind=='raw' else len(common[doc['id']][str(scale)])
                        if not n:continue
                        suffix='__common' if kind=='common' and n!=doc['pages'] else ''
                        reps=[]
                        for rep in range(3):
                            p=OUT/'timing'/f'{e}__{doc["id"]}__s{scale}__w{workers}__timing__r{rep}{suffix}.json'
                            if not p.exists():raise RuntimeError(f'Missing job: {p}')
                            d=load(p);m=d['memory']
                            assert m['returncode']==0 and d['selected_count']==n
                            reps.append(d)
                            row=dict(kind=kind,id=doc['id'],engine=e,scale=scale,dpi=scale*72,
                                     workers=workers,rep=rep,selected=n,ok=d['ok'],wall_ms=d['wall_ms'],
                                     pages_per_second=1000*d['ok']/d['wall_ms'],
                                     peak_rss_mib=m['peak_tree_rss_sampled_bytes']/2**20,
                                     peak_commit_mib=m['peak_job_commit_bytes']/2**20,
                                     peak_private_sampled_mib=m['peak_tree_private_sampled_bytes']/2**20,
                                     peak_wset_upper_bound_mib=m['sum_process_peak_wset_bytes']/2**20,
                                     processes=m['max_processes'],sampler_max_gap_ms=m['max_sample_gap_ms'],
                                     pre_job_cpu_percent=d['before']['cpu_percent'],
                                     pre_job_available_mib=d['before']['available_bytes']/2**20)
                            flat.append(row)
                        wall=median(d['wall_ms'] for d in reps)
                        cells.append(dict(kind=kind,id=doc['id'],engine=e,scale=scale,workers=workers,
                                          selected=n,ok_min=min(d['ok'] for d in reps),wall_ms=wall,
                                          pages_per_second=median(1000*d['ok']/d['wall_ms'] for d in reps),
                                          peak_rss_mib=median(d['memory']['peak_tree_rss_sampled_bytes']/2**20 for d in reps),
                                          peak_commit_mib=median(d['memory']['peak_job_commit_bytes']/2**20 for d in reps),
                                          spread_percent=100*(max(d['wall_ms'] for d in reps)-min(d['wall_ms'] for d in reps))/wall))
    write_csv(OUT/'measurements.csv',flat);write_csv(OUT/'medians.csv',cells)
    by={(c['kind'],c['id'],c['scale'],c['workers'],c['engine']):c for c in cells}
    def cell(kind,doc,s,w,e):return by.get((kind,doc['id'],s,w,e))
    def speed(kind,e,workers=None,scale=None,include_single_page=True):
        ratios=[]
        for d in docs:
            if d['pages']==1 and not include_single_page:continue
            for s in (1,2):
                if scale and s!=scale:continue
                for w in (1,8):
                    if workers and w!=workers:continue
                    a=cell(kind,d,s,w,'exprim');b=cell(kind,d,s,w,e)
                    if a and b and a['ok_min']==a['selected'] and b['ok_min']==b['selected']:
                        ratios.append(a['pages_per_second']/b['pages_per_second'])
        return (geometric_mean(ratios),min(ratios),max(ratios),len(ratios)) if ratios else None
    lines=['# exprim renderer benchmark — run started 2026-11-09','',
           'exprim 0.4.0 versus PDFium 158.0.8125, MuPDF 1.28.6 (PyMuPDF 2.1), and hayro 0.9.0. '
           'Nine preselected PDFs from a private local library, **3,871 pages**, at 72 and 144 DPI, with 1 and 8 workers. '
           'Three rotated repetitions per timing cell. Raw jobs, all-page agreement checks, and a rerunnable harness are retained here.','',
           '## Speed on the same accepted pages','',
           '**exprim speed multiple** is the geometric mean of exprim pages/s divided by the control pages/s, '
           'with equal weight per document × DPI × worker cell. Above 1 means exprim is faster. '
           'These figures use only pages on which all four engines agree, re-rendered as a separate timing series where necessary. '
           'Eligible cells from all nine documents are included; cells with no common accepted pages are omitted and their count is shown. '
           'A second aggregate excluding the one-page map is shown as a sensitivity check: '
           'the map tests dense path rendering but cannot exploit page parallelism.','',
           '| Control | All 9 documents | 8 multipage documents | 1 worker (all 9) | 8 workers (all 9) | 72 DPI | 144 DPI | Cell range (all 9) | Cells |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    raw_summary={};raw_lines=['## All-page render throughput (ungated)','',
            'This preserves the prior comparison’s full-document metric. Values are exprim pages/s divided by control pages/s, '
            'geometrically averaged with equal document × DPI × worker weight. These are measured throughput ratios, '
            '**not claims of equally correct output**; pairwise disagreement and coverage are reported below. '
            'Above 1 means greater measured exprim throughput.','',
            '| Control | All 9 documents | 8 multipage documents | 1 worker | 8 workers | 72 DPI | 144 DPI | Cell range | Cells |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for e in ENGINES[1:]:
        r=speed('raw',e);m=speed('raw',e,include_single_page=False)
        vals=[speed('raw',e,workers=1),speed('raw',e,workers=8),speed('raw',e,scale=1),speed('raw',e,scale=2)]
        raw_summary[e]=dict(geometric_mean_multiple=r[0],multipage_geometric_mean_multiple=m[0],minimum_cell_multiple=r[1],maximum_cell_multiple=r[2],cells=r[3])
        raw_lines.append(f'| {e} | **{r[0]:.2f}×** | {m[0]:.2f}× | '+' | '.join(f'{v[0]:.2f}×' for v in vals)+f' | {r[1]:.2f}–{r[2]:.2f}× | {r[3]} |')
    raw_lines.append('')
    lines[4:4]=raw_lines
    summary={}
    for e in ENGINES[1:]:
        overall=speed('common',e)
        multipage=speed('common',e,include_single_page=False)
        summary[e]=dict(geometric_mean_multiple=overall[0],multipage_geometric_mean_multiple=multipage[0] if multipage else None,minimum_cell_multiple=overall[1],maximum_cell_multiple=overall[2],cells=overall[3]) if overall else None
        vals=[speed('common',e,workers=1),speed('common',e,workers=8),speed('common',e,scale=1),speed('common',e,scale=2)]
        if overall:lines.append(f'| {e} | **{overall[0]:.2f}×** | '+(f'{multipage[0]:.2f}×' if multipage else '—')+' | '+ ' | '.join(f'{v[0]:.2f}×' if v else '—' for v in vals)+f' | {overall[1]:.2f}–{overall[2]:.2f}× | {overall[3]} |')
    lines+=['',f'**Coverage limits this conditional aggregate.** Only {len(common["text"]["1"])}/786 text pages at 72 DPI '
            f'and {len(common["text"]["2"])}/786 at 144 DPI pass the strict all-four gate. '
            'Reviewed disputed pages retain their text but differ in font weight; scan disputes also include resampling/contrast differences. '
            'Do not generalize the accepted text subset to full-book text performance. Raw full-document figures and all exclusions remain visible.']
    totals=[]
    lines+=['','### Corpus-total render throughput','',
            'This answers a different question: accepted corpus pages divided by the sum of per-document median render times. '
            'It gives large documents their actual time weight; the oversized JPEG book strongly influences it. '
            'Document open, process launch, and encoding remain excluded. The equal-document comparison above keeps class-level differences visible.','',
            '| DPI | Workers | Accepted pages | exprim pages/s | PDFium | MuPDF | hayro |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for s,w in ((2,1),(1,1),(2,8),(1,8)):
        eligible=[d for d in docs if cell('common',d,s,w,'exprim')]
        n=sum(cell('common',d,s,w,'exprim')['selected'] for d in eligible)
        rates={e:1000*n/sum(cell('common',d,s,w,e)['wall_ms'] for d in eligible) for e in ENGINES} if n else {}
        totals.append(dict(scale=s,workers=w,accepted_pages=n,pages_per_second=rates))
        lines.append(f'| {s*72} | {w} | {n} | '+' | '.join(f'{rates[e]:.2f}' if rates else '—' for e in ENGINES)+' |')
    lines+=['','## Corpus and agreement coverage','',
            'Selection was based on page/image dictionaries before performance results. SHA-256 hashes, absolute source paths, '
            'file sizes, and full image-filter counts are in [manifest.json](manifest.json); every page’s filters and dimensions '
            'are in [corpus-pages.json](corpus-pages.json). Resource images and referenced mask streams are counted separately; '
            'the latter are in [corpus-mask-census.json](corpus-mask-census.json). Source PDFs are private inputs and are not redistributed.','',
            '| ID / class | Pages | MiB | All-four agreement: 72 DPI | 144 DPI |',
            '|---|---:|---:|---:|---:|']
    for d in docs:lines.append(f'| {d["id"]}: {d["category"]} | {d["pages"]} | {d["bytes"]/2**20:.1f} | {len(common[d["id"]]["1"])}/{d["pages"]} | {len(common[d["id"]]["2"])}/{d["pages"]} |')
    dense=next(d for d in docs if d['id']=='vector-map')
    bilevel=next(d for d in docs if d['id']=='jbig2')
    dense_rates=', '.join(f'{e} {cell("raw",dense,2,1,e)["pages_per_second"]:.2f}' for e in ENGINES)
    bilevel_commit=', '.join(f'{e} {cell("raw",bilevel,2,1,e)["peak_commit_mib"]:.0f}' for e in ENGINES)
    lines+=['','### Tradeoffs visible in this corpus','',
            f'- Dense vector map, raw 144-DPI serial throughput (pages/s): **{dense_rates}**. '
            'This is a distinct path-heavy workload; scan-book performance does not describe it.',
            f'- JBIG2 book, median whole-job peak committed memory at 144 DPI / one worker (MiB): **{bilevel_commit}**. '
            'The high exprim footprint repeats across runs. This benchmark measures the peak but does not establish its allocation cause.',
            '- The headline ratios depend on the document mix and the agreement coverage. The full rates and the multi-page sensitivity check are retained to make that dependence visible.']
    scaling=load(OUT/'memory-scaling.json') if (OUT/'memory-scaling.json').exists() else []
    if scaling:
        lines+=['','#### JBIG2 memory prefix diagnostic','',
                'Four additional fresh-process exprim jobs render consecutive prefixes of the same JBIG2 book at 72 DPI / one worker. '
                'These memory-only diagnostics are separate from the 744 timing jobs and all speed aggregates. '
                'The peak grows strongly with prefix length in this case; the allocation cause is not established.','',
                '| Prefix pages | OS peak job commit, MiB | Sampled aggregate RSS, MiB |',
                '|---|---:|---:|']
        for row in scaling:lines.append(f'| {row["pages"]} | {row["peak_job_commit_mib"]:.0f} | {row["peak_tree_rss_sampled_mib"]:.0f} |')
        lines+=['','Raw diagnostic jobs and commands are under `memory-scaling/`; [memory-scaling.json](memory-scaling.json) summarizes them.']
    lines+=['','The JPX-only scan contains exactly 50 JPX image references and no masks. “Strict” describes the decoding path, '
            'not an independent standards-conformance certification. MRC contains 1,264 JPX image references and 632 one-bit JBIG2 soft-mask references. '
            'The CCITT document is CCITT-dominant (5,109 references), with two JBIG2 references. The dense map has no image XObjects.','',
            'Pages retain their declared PDF dimensions. The large JPEG book has oversized page boxes '
            '(for example, 2,758 × 3,561 points), giving about **39 million pixels at 144 DPI** on that page. '
            'These are intentionally large rasters; the DPI labels do not imply ordinary A4-sized output. '
            'Pixel dimensions are retained, and engines are compared on the same pages rather than normalizing rates across different page sizes.','',
            '## Raster agreement','',
            'Agreement is a consistency measure, **not ground-truth accuracy**. All six pairwise comparisons are retained. '
            'A page gets an engine-level `ok` only when that engine agrees with every maximum mutually agreeing clique of at least three engines. '
            'With fewer than three agreeing engines the page is unresolved. The performance gate is stricter: all four rasters must agree pairwise. '
            'exprim is partly derived from PDFium, so those two are not fully independent implementations; majority agreement can preserve shared defects.','',
            '| Engine | 72 DPI: ok / suspect / unresolved / ambiguous / failed | 144 DPI: same order |',
            '|---|---:|---:|']
    for e in ENGINES:
        cols=[' / '.join(str(accuracy[str(s)][e][k]) for k in ('ok','suspect','unresolved','ambiguous','failed')) for s in (1,2)]
        lines.append(f'| {e} | {cols[0]} | {cols[1]} |')
    lines+=['','Thresholds match the 2026-11-06 comparison: disagree if >5% of pixels differ by >48 in any RGB channel, '
            'or ink coverage differs by >1 percentage point **and** continuous darkness by >0.3 points, or blankness disagrees. '
            'An ink pixel has RGB sum <600; blankness means one raster’s ink fraction is <5% of the other’s when the latter is >0.05%. '
            'Extent differences of at most one pixel are white-padded at the same origin; larger differences fail the pair. '
            '144-DPI rasters are box-reduced 2× before comparison. RGB pixel SHA-256 hashes and actual dimensions are retained per engine/page. '
            'PDFium uses bitmap format BGRA with render flag `FPDF_REVERSE_BYTE_ORDER` (0x10), producing RGBA bytes; '
            'a generated red-page calibration verifies the exported RGB channel order in all four engines. '
            'Representative and worst disagreement sheets are under [review/](review/). Full transient PNGs were removed after scoring.','']
    lines+=['[Visual review notes](review-notes.md) describe the inspected representative and disagreement sheets. '
            'They preserve the automated labels and do not claim an independent ground truth.','']
    lines+=['The untimed pass uses up to four native PNG encoder threads and four independent scoring processes. '
            'No PDF API is called from an encoder thread, and this parallelism does not change any timed rendering budget.','']
    for kind,title in [('common','Same accepted pages'),('raw','All attempted pages (ungated)')]:
        lines += [f'## {title}: pages/s','']
        if kind=='raw':lines+=['These raw rates do not imply correct output; compare them with the agreement coverage above.','']
        for s,w in ((2,1),(1,1),(2,8),(1,8)):
            lines += [f'### {s*72} DPI, {w} worker'+('s' if w>1 else ''),'',
                      '| Document | Pages | exprim | PDFium | MuPDF | hayro |','|---|---:|---:|---:|---:|---:|']
            for d in docs:
                n=d['pages'] if kind=='raw' else len(common[d['id']][str(s)])
                values=[cell(kind,d,s,w,e) for e in ENGINES]
                lines.append(f'| {d["id"]} | {n} | '+' | '.join(f'{v["pages_per_second"]:.2f}' if v else '—' for v in values)+' |')
            lines.append('')
    lines+=['## Memory, including PDFium and MuPDF','',
            'Memory covers the **entire child job lifetime**, including document open, runtime, coordinator, workers, and console host. '
            'Tables show the median of three **all-page** jobs. Each entry is **sampled aggregate resident working set / OS peak job commit**, in MiB. '
            'The first figure is sampled every 10 ms and can miss short peaks; it also counts shared DLL and mapped-input pages once per process. '
            'The second is Windows’ continuously tracked job high-water mark for committed memory; it is not RSS. '
            'Both are retained separately, alongside sampled private memory and the observed sum of individual peak working sets '
            '(an upper bound on the sampled aggregate, not a simultaneous lifetime peak; final per-process peaks can be missed between polls).','',
            'The old sampler followed only the venv launcher. This harness uses the base interpreter with the venv packages and starts every job suspended, '
            'assigns it to a Windows Job Object before resuming it, then enumerates all job PIDs. This also captures launcher descendants if a venv executable is used. '
            'PDFium/MuPDF at eight workers have eight rendering processes plus an opened-document coordinator; Python overhead is included. '
            'Rust engines share one document in one process. A console host may add another small process to each job.','']
    for s,w in ((2,8),(1,8),(2,1),(1,1)):
        lines += [f'### {s*72} DPI, {w} worker'+('s' if w>1 else ''),'',
                  '| Document | exprim | PDFium | MuPDF | hayro |','|---|---:|---:|---:|---:|']
        for d in docs:
            values=[cell('raw',d,s,w,e) for e in ENGINES]
            lines.append(f'| {d["id"]} | '+' | '.join(f'{v["peak_rss_mib"]:.0f} / {v["peak_commit_mib"]:.0f}' for v in values)+' |')
        lines.append('')
    export=load(OUT/'dump/exprim__jpeg-large__s2__w8__dump__r0.json')
    lines += [f'**Separate export stress case:** the one untimed exprim PNG-export job for the oversized JPEG book at 144 DPI '
              f'reached {export["memory"]["peak_job_commit_bytes"]/2**30:.2f} GiB of job commit. '
              'That job includes PNG encoding and output buffering; it is excluded from the rendering-only table. '
              'Export implementations use different encoding parallelism, so their dump-memory figures are not a matched comparison. '
              'The retained dump logs expose this consumer/output workload separately.','']
    lines+=['PDFium uses a read-only, prefaulted memory map passed to `FPDF_LoadMemDocument64`, replacing the old Python adapter’s '
            'two full input copies per process. This matters for the 745 MiB input and avoids measuring gratuitous adapter copies. '
            'exprim and hayro use owned input bytes; MuPDF opens a filename. All documents are opened before timing and the OS file cache is primed equally. '
            'The mapping strategy and Python process overhead mean these are end-to-end adapter footprints, not isolated allocator totals for each library.','',
            'Windows accounting documentation: [job peak committed memory](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information), '
            '[job process enumeration](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-queryinformationjobobject).','',
            '## Timing protocol and limits','',
            '- Same basic protocol as the ledger’s 2026-11-06 comparison: one isolated child job per engine/document/DPI/workers/repetition; engine order rotates per repetition. '
            'No engines run concurrently. The timer includes page load, interpretation, decoding, rasterization, and native pixel materialization. '
            'File loading, document open, process launch, hashing, scoring, and PNG encoding are excluded. Pixels are opaque white RGB/RGBA8.',
            '- One worker means serial pages. Eight workers means exprim’s bounded scheduler (2 compile + 6 render), eight Rayon page workers for hayro, '
            'and eight processes for PDFium/MuPDF. Native process initialization is synchronized before the timer. These are controlled worker budgets, '
            'not claims that every engine reaches peak throughput with its own preferred defaults.',
            '- Nested work is controlled with `RAYON_NUM_THREADS=workers`, `PDF_RENDER_THREADS=1`, `PDF_RENDERER_JPX_CONCURRENCY=serial`, '
            '`PDF_RENDERER_IMAGE_ROW_PAR=0`. The old low-level adapter is ported to exprim’s extracted modules; it does not benchmark the public `RenderSession` page cache.',
            '- Intel Core i7-13700H, 32 GiB installed RAM, Windows 11 build 26300. Jobs use the same logical CPU affinity 0–11 as the prior run '
            '(the prior harness identified these as P-cores; efficiency classes were not re-enumerated). Machine, power plan, compiler, environment, '
            'dependency locks, DLL/binary hashes, and source hashes are recorded in [machine.json](machine.json).',
            '- The extracted engine sources were untracked at capture. The source-hash manifest and executable SHA-256 identify this tested working tree; '
            'the checkout’s codec-import Git HEAD alone does not identify the renderer sources.',
            '- Both Rust adapters use release optimization level 3, fat LTO, one codegen unit, panic=unwind, and no target-cpu override. '
            'Controls are pinned to the same engine versions as the prior comparison. This run uses Rust 1.99.1 versus 1.97.1 previously, new dependency resolutions, '
            'and a different corpus; it is **not** a controlled before/after measurement of the optimization commits.',
            '- This is a private, scan-heavy selection on one hybrid-core laptop. Small jobs, especially the one-page map, include substantial scheduling overhead. '
            'Background CPU and available RAM are sampled before every job and retained, so the report does not assume a perfectly idle host.','',
            '### Repetition spread','',
            'Per-cell spread = (slowest − fastest wall time) / median wall time, over the three all-page jobs.','',
            '| Engine | Median spread | Maximum spread |','|---|---:|---:|']
    for e in ENGINES:
        vals=[c['spread_percent'] for c in cells if c['kind']=='raw' and c['engine']==e]
        lines.append(f'| {e} | {median(vals):.1f}% | {max(vals):.1f}% |')
    samples=[]
    for p in sorted((OUT/'sample').glob('*.json')):
        d=load(p)
        for page in sorted({r['page'] for r in d['rows']}):
            rows=[r for r in d['rows'] if r['page']==page];warm=[r['ms'] for r in rows if r['phase']=='warm' and r['ok']]
            first=next(r for r in rows if r['phase']=='first')
            if warm:samples.append(dict(id=d['id'],engine=d['engine'],scale=d['scale'],page=page,
                                        first_ms=first['ms'],warm_samples=len(warm),warm_p50_ms=median(warm),
                                        warm_p90_ms=percentile(warm,.9),warm_p99_ms=percentile(warm,.99)))
    write_csv(OUT/'latency-samples.csv',samples)
    lines+=['','### Sampled page latency','',
            '[latency-samples.csv](latency-samples.csv) gives first-render and warm p50/p90/p99 for the first, middle, and last page of each document '
            '(one unique page for the map), separately at each DPI. Each page has one first render, five discarded warmups, and twenty warm samples. '
            'The document stays open across the three sampled pages, so “first” is not an OS-cold file measurement. Samples are separate from throughput jobs.','',
            '## Reproduction and retained evidence','',
            '- Harness and setup: [tools/render-bench](../../../tools/render-bench/README.md).',
            '- [measurements.csv](measurements.csv): every timing repetition, both populations, memory and pre-job load.',
            '- [medians.csv](medians.csv): per-cell medians and repetition spread.',
            '- [accuracy/summary.json](accuracy/summary.json), per-page JSON, pairwise CSVs, pixel hashes, and exact common page lists.',
            '- `timing/`, `dump/`, `sample/`: command lines, stdout/stderr, open times, successes/errors, and Windows job memory records.',
            '- [source-verification.json](source-verification.json): source stability check after the complete run.',
            '- [harness-source-hashes.json](harness-source-hashes.json): final harness source and dependency-lock identities.',
            '- [audit.json](audit.json): completeness and memory validation.','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(dict(raw_all_page_speed_multiples=raw_summary,common_page_speed_multiples=summary,corpus_total_throughput=totals,documents=len(docs),pages=sum(d['pages'] for d in docs)),indent=2))
    artifacts={}
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='artifact-integrity.json':
            with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            artifacts[str(p.relative_to(OUT)).replace('\\','/')]=dict(bytes=p.stat().st_size,sha256=digest)
    (OUT/'artifact-integrity.json').write_text(json.dumps(artifacts,indent=2))
    print(json.dumps(dict(raw_all_page=raw_summary,accepted_subset=summary),indent=2))

if __name__=='__main__':main()
