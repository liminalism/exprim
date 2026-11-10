"""Select before timing; census every selected page without decoding pixels."""
from pathlib import Path
import collections, hashlib, json
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/benchmarks/2026-11-09'
PDF = Path('{{secret:0cec3c4cdf5da2d1947072c580d8a8e9}}')
CHOICES = [
    ('jpeg-large', 'JPEG scan (large file)', PDF/'{{secret:85565cdf1a27ff0b324523252ad91553}}'),
    ('mrc', 'JPX image + mask MRC', next((PDF/'{{secret:3e7b1520b09025c2126636c01d79a130}}').glob('{{secret:0fd0fa87cc7800c2dc3a0862de48f623}}*'))),
    ('jp2', 'Full-page JPX (strict decoding)', PDF/'{{secret:a0c09e263074891fbb97ee5a473ede1f}}'),
    ('vector-tables', 'Vector tables (long document)', PDF/'{{secret:eacdef0f8fda69194d89bad0b068d750}}'),
    ('vector-map', 'Vector map (dense paths)', PDF/'{{secret:e32ef5391203ea1fd7a46224b3a3dd51}}'),
    ('jbig2', 'JBIG2 bilevel scan', PDF/'{{secret:07d812097a0dd9c466ee4e9835b25b35}}'),
    ('ccitt', 'CCITT bilevel scan', PDF/'{{secret:04cf4ec790db8420c2cdf50a5dae11d5}}'),
    ('text', 'Born-digital Latin text', PDF/'{{secret:bf9aacddb937f5d74aaaaac78df9f361}}'),
    ('mixed-cjk', 'CJK illustrated / soft masks', PDF/'{{secret:aa6a6a48aa819575fcaaad338afedc01}}'),
]

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
