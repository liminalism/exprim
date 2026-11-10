//! Untimed quality scoring only. No renderer or timing adapter calls this code.
//! RGB8 metrics, with the same thresholds as the Python reference scorer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn headsup_compare(a:*const u8,b:*const u8,n:usize,out:*mut u64) {
    if a.is_null() || b.is_null() || out.is_null() || n>usize::MAX/3 {return;}
    // Caller supplies two contiguous RGB8 buffers of n pixels and six u64 outputs.
    let (a,b)=unsafe{(std::slice::from_raw_parts(a,n*3),std::slice::from_raw_parts(b,n*3))};
    let (mut absolute,mut gross,mut ai,mut bi,mut asum,mut bsum)=(0u64,0u64,0u64,0u64,0u64,0u64);
    for (x,y) in a.chunks_exact(3).zip(b.chunks_exact(3)) {
        let d0=x[0].abs_diff(y[0]);let d1=x[1].abs_diff(y[1]);let d2=x[2].abs_diff(y[2]);
        absolute+=d0 as u64+d1 as u64+d2 as u64;
        gross+=u64::from(d0.max(d1).max(d2)>48);
        let sx=x[0] as u64+x[1] as u64+x[2] as u64;
        let sy=y[0] as u64+y[1] as u64+y[2] as u64;
        ai+=u64::from(sx<600);bi+=u64::from(sy<600);asum+=sx;bsum+=sy;
    }
    unsafe{std::ptr::copy_nonoverlapping([absolute,gross,ai,bi,asum,bsum].as_ptr(),out,6);}
}
