"""Native controls; PDFium is the verified Blanchon DLL, MuPDF the PyMuPDF wheel.
No image encoding or disk I/O inside page timers. One independent document per
process; never call PDFium or MuPDF concurrently from Python threads.
"""
import ctypes as C
import json, math, os, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

B=Path(__file__).resolve().parent
class PyBuffer(C.Structure):
    _fields_=[('buf',C.c_void_p),('obj',C.c_void_p),('len',C.c_ssize_t),
              ('itemsize',C.c_ssize_t),('readonly',C.c_int),('ndim',C.c_int),
              ('format',C.c_char_p),('shape',C.c_void_p),('strides',C.c_void_p),
              ('suboffsets',C.c_void_p),('internal',C.c_void_p)]
class Matrix(C.Structure):
    _fields_=[(n,C.c_float) for n in ('a','b','c','d','e','f')]
class Rect(C.Structure):
    _fields_=[(n,C.c_float) for n in ('left','top','right','bottom')]
class PDFium:
    def __init__(self,path):
        self.lib=C.CDLL(str(Path(os.environ['BENCH_PDFIUM_DLL'])))
        sigs={
          'FPDF_InitLibrary':(None,[]),'FPDF_LoadMemDocument64':(C.c_void_p,[C.c_void_p,C.c_size_t,C.c_char_p]),
          'FPDF_GetPageCount':(C.c_int,[C.c_void_p]),'FPDF_LoadPage':(C.c_void_p,[C.c_void_p,C.c_int]),
          'FPDF_GetPageWidthF':(C.c_float,[C.c_void_p]),'FPDF_GetPageHeightF':(C.c_float,[C.c_void_p]),
          'FPDFBitmap_CreateEx':(C.c_void_p,[C.c_int,C.c_int,C.c_int,C.c_void_p,C.c_int]),
          'FPDFBitmap_FillRect':(C.c_int,[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_ulong]),
          'FPDF_RenderPageBitmap':(None,[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int]),
          'FPDF_RenderPageBitmapWithMatrix':(None,[C.c_void_p,C.c_void_p,C.POINTER(Matrix),C.POINTER(Rect),C.c_int]),
          'FPDFBitmap_Destroy':(None,[C.c_void_p]),'FPDF_ClosePage':(None,[C.c_void_p]),
          'FPDF_CloseDocument':(None,[C.c_void_p]),'FPDF_GetLastError':(C.c_ulong,[])}
        for name,(ret,args) in sigs.items():
            f=getattr(self.lib,name);f.restype=ret;f.argtypes=args
        self.lib.FPDF_InitLibrary()
        import mmap
        self.file=open(path,'rb');self.data=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
        # Fault input pages before the timer; retain a zero-copy C view.
        self.preload=sum(self.data[::4096]);self.view=PyBuffer()
        get_buffer=C.pythonapi.PyObject_GetBuffer
        get_buffer.argtypes=[C.py_object,C.POINTER(PyBuffer),C.c_int];get_buffer.restype=C.c_int
        if get_buffer(self.data,C.byref(self.view),0):raise RuntimeError('mmap buffer')
        self.buf=self.view.buf
        t=time.perf_counter();self.doc=self.lib.FPDF_LoadMemDocument64(self.buf,len(self.data),b'')
        if not self.doc:raise RuntimeError(f'PDFium open {self.lib.FPDF_GetLastError()}')
        self.count=self.lib.FPDF_GetPageCount(self.doc);self.open_ms=(time.perf_counter()-t)*1000
    def render(self,p,scale):
        l=self.lib;page=l.FPDF_LoadPage(self.doc,p)
        if not page:raise RuntimeError(f'PDFium page {l.FPDF_GetLastError()}')
        try:
            w=max(1,math.floor(l.FPDF_GetPageWidthF(page)*scale));h=max(1,math.floor(l.FPDF_GetPageHeightF(page)*scale))
            buf=C.create_string_buffer(w*h*4);bmp=l.FPDFBitmap_CreateEx(w,h,4,buf,w*4)
            if not bmp:raise RuntimeError('PDFium bitmap allocation')
            try:
                l.FPDFBitmap_FillRect(bmp,0,0,w,h,0xffffffff)
                if os.environ.get('HEADSUP_PDFIUM_LEGACY')=='1':l.FPDF_RenderPageBitmap(bmp,page,0,0,w,h,0,1|0x10)
                else:
                    matrix=Matrix(scale,0,0,scale,0,0);clip=Rect(0,0,w,h)
                    l.FPDF_RenderPageBitmapWithMatrix(bmp,page,C.byref(matrix),C.byref(clip),1|0x10)
                return w,h,buf,buf
            finally:l.FPDFBitmap_Destroy(bmp)
        finally:l.FPDF_ClosePage(page)

class MuPDF:
    def __init__(self,path):
        import pymupdf as f
        self.f=f;t=time.perf_counter();self.doc=f.open(path);self.count=len(self.doc);self.open_ms=(time.perf_counter()-t)*1000
    def render(self,p,scale):
        pix=self.doc[p].get_pixmap(matrix=self.f.Matrix(scale,scale),colorspace=self.f.csRGB,alpha=False,annots=True)
        # Retain the native owning Pixmap beside a zero-copy sample view.
        return pix.width,pix.height,pix,pix.samples_mv

ENGINE=None
def initialize(engine,file,scale,barrier=None):
    global ENGINE,SCALE,VERIFY
    ENGINE=PDFium(file) if engine=='pdfium' else MuPDF(file);SCALE=scale;VERIFY=os.environ.get('HEADSUP_VERIFY')=='1'
    if barrier is not None:barrier.wait(timeout=120)
def ready(p):return os.getpid()
def task_dump(p):
    t=time.perf_counter()
    try:
        w,h,owner,data=ENGINE.render(p,SCALE)
        return dict(page=p,ms=(time.perf_counter()-t)*1000,ok=True,width=w,height=h),bytes(data)
    except Exception as e:return dict(page=p,ok=False,error=str(e)),None
def bounded_dump(pool,pages,workers):
    # Python 3.13 Executor.map eagerly queues the whole book. A 144-DPI
    # Sahib page is ~75 MB RGBA: bound pending image payloads to two per worker.
    from collections import deque
    source=iter(pages);pending=deque()
    for _ in range(workers*2):
        try:pending.append(pool.submit(task_dump,next(source)))
        except StopIteration:break
    while pending:
        yield pending.popleft().result()
        try:pending.append(pool.submit(task_dump,next(source)))
        except StopIteration:pass
def task(p):
    t=time.perf_counter()
    try:
        w,h,owner,data=ENGINE.render(p,SCALE)
        row=dict(page=p,ms=(time.perf_counter()-t)*1000,ok=True,width=w,height=h,nbytes=len(data))
        if VERIFY:
            import hashlib
            row['hash']=hashlib.sha256(data).hexdigest()
        return row
    except Exception as e:return dict(page=p,ms=(time.perf_counter()-t)*1000,ok=False,error=str(e))
def main():
    engine,file,scale,workers,mode,out=sys.argv[1:7];scale=float(scale);workers=int(workers)
    if mode=='verify':os.environ['HEADSUP_VERIFY']='1'
    initialize(engine,file,scale);count=ENGINE.count;open_ms=ENGINE.open_ms
    selected=list(map(int,sys.argv[7].split(','))) if len(sys.argv)>7 else list(range(count))
    if mode=='sample':
        rows=[]
        indices=list(map(int,sys.argv[7].split(','))) if len(sys.argv)>7 else [0,count//2,count-1]
        for p in indices:
            for r in range(26):
                row=task(p);row.update(sample=r,phase='first' if r==0 else 'warmup' if r<6 else 'warm');rows.append(row)
        print(json.dumps(dict(count=count,open_ms=open_ms,rows=rows,transform_policy='exact-scale-matrix' if engine=='pdfium' else 'native-fixed-scale')));return
    rows=[]
    if mode=='dump':
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        Path(out).mkdir(parents=True,exist_ok=True)
        def encode(row,data):
            if row['ok']:
                size=(row['width'],row['height'])
                # Render flag 0x10 (FPDF_REVERSE_BYTE_ORDER) makes this RGBA.
                im=Image.frombytes('RGBA' if engine=='pdfium' else 'RGB',size,data)
                im.convert('RGB').save(Path(out)/f'{row["page"]:05}.png',compress_level=1)
            return row
        def consume(results):
            # Only PNG encoding is threaded; PDF API calls stay in processes.
            encoders=max(1,min(workers,4));pending=deque()
            with ThreadPoolExecutor(max_workers=encoders) as encoder:
                for row,data in results:
                    pending.append(encoder.submit(encode,row,data))
                    if len(pending)>=2*encoders:rows.append(pending.popleft().result())
                while pending:rows.append(pending.popleft().result())
        if workers==1:consume(map(task_dump,selected))
        else:
            import multiprocessing as mp
            context=mp.get_context('spawn');barrier=context.Barrier(workers)
            with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=initialize,initargs=(engine,file,scale,barrier)) as pool:
                list(pool.map(ready,range(workers),chunksize=1))
                consume(bounded_dump(pool,selected,workers))
        wall_ms=sum(r.get('ms',0) for r in rows)
    elif workers==1:
        t=time.perf_counter();rows=[task(p) for p in selected];wall_ms=(time.perf_counter()-t)*1000
    else:
        import multiprocessing as mp
        context=mp.get_context('spawn');barrier=context.Barrier(workers)
        with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=initialize,initargs=(engine,file,scale,barrier)) as pool:
            # Each process is ready and has opened a document before the wall clock.
            list(pool.map(ready,range(workers),chunksize=1))
            t=time.perf_counter();rows=list(pool.map(task,selected,chunksize=1));wall_ms=(time.perf_counter()-t)*1000
    print(json.dumps(dict(count=count,selected_count=len(selected),open_ms=open_ms,ok=sum(r['ok'] for r in rows),wall_ms=wall_ms,rows=rows,includes_png=mode=='dump',transform_policy='exact-scale-matrix' if engine=='pdfium' else 'native-fixed-scale')))
if __name__=='__main__':main()
