"""Select before timing; census every selected page without decoding pixels."""
from pathlib import Path
import collections, hashlib, json
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/benchmarks/2026-11-09'
# The benchmark documents are private local files and are not named here.
# List them in corpus.local.json beside this script (git-ignored):
#   [{"id": "vector-map", "label": "Vector map (dense paths)", "path": "..."}]
LOCAL = Path(__file__).resolve().parent/'corpus.local.json'
if not LOCAL.exists():
    raise SystemExit(f'{LOCAL} is missing: list the local benchmark documents there (id, label, path)')
CHOICES = [(e['id'], e['label'], Path(e['path'])) for e in json.loads(LOCAL.read_text(encoding='utf-8'))]

def main():
    manifest=[]; page_rows=[]
    for key,category,path in CHOICES:
        doc=pymupdf.open(path); filters=collections.Counter(); maskrefs=0; bits=collections.Counter()
        for i,page in enumerate(doc):
            images=page.get_images(full=True); local=collections.Counter()
            for im in images:
                filt=doc.xref_get_key(im[0], 'Filter')[1]
                local[filt]+=1; filters[filt]+=1; maskrefs+=bool(im[1]); bits[str(im[4])]+=1
            page_rows.append(dict(id=key,page=i,width=page.rect.width,height=page.rect.height,
                                  filters=dict(local),masks=sum(bool(im[1]) for im in images)))
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        entry=dict(id=key,category=category,path=str(path),name=path.name,bytes=path.stat().st_size,
                   pages=len(doc),sha256=digest,image_references=dict(filters),
                   soft_mask_references=maskrefs,image_bit_depth_references=dict(bits),
                   redistributable=False)
        manifest.append(entry); print(json.dumps(entry),flush=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'corpus-pages.json').write_text(json.dumps(page_rows),encoding='utf-8')

def write_mask_census():
    """Supplement the frozen manifest; never replace its original input hashes."""
    result={}
    for entry in json.loads((OUT/'manifest.json').read_text(encoding='utf-8')):
        doc=pymupdf.open(entry['path']);counts=collections.Counter();bits=collections.Counter()
        kinds=collections.Counter();embedded_alpha=collections.Counter()
        for page in doc:
            for im in page.get_images(full=True):
                for key in ('SMask','Mask'):
                    kind,value=doc.xref_get_key(im[0],key)
                    if kind=='xref':
                        xref=int(value.split()[0]);kinds[key]+=1
                        counts[key+':'+doc.xref_get_key(xref,'Filter')[1]]+=1
                        bits[key+':'+doc.xref_get_key(xref,'BitsPerComponent')[1]]+=1
                    elif key=='Mask' and kind=='array':kinds['Mask:color-key-array']+=1
                kind,value=doc.xref_get_key(im[0],'SMaskInData')
                if kind=='int' and value!='0':embedded_alpha[value]+=1
        result[entry['id']]=dict(mask_reference_kinds=dict(kinds),mask_filter_references=dict(counts),
                                mask_bit_depth_references=dict(bits),smask_in_data_references=dict(embedded_alpha))
    (OUT/'corpus-mask-census.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__': main()
