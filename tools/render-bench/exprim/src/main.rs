use std::{sync::Arc, time::Instant, path::Path};
use exprim::document::{DocumentSnapshot, DocumentLimits, PageIndex, ParseContext};
use exprim::source::{PdfSource, OwnedBytesSource};
use exprim::render_api::*;
use exprim::render_cpu::CpuBackend;
use exprim::page_ir::{Matrix,DeviceSize};
use exprim::render_scheduler::{RenderScheduler,SchedulerOptions};
use serde_json::json;
use std::hash::{Hash,Hasher};
include!("helpers.rs");
fn fingerprint(data:&[u8])->String {let mut h=std::collections::hash_map::DefaultHasher::new();data.hash(&mut h);format!("{:016x}",h.finish())}

fn save(out:&str,p:u32, host:&HostPage) -> anyhow::Result<()> {
    std::fs::create_dir_all(out)?;
    let f=std::fs::File::create(Path::new(out).join(format!("{p:05}.png")))?;
    let mut e=png::Encoder::new(std::io::BufWriter::new(f),host.width,host.height);
    e.set_color(png::ColorType::Rgba); e.set_depth(png::BitDepth::Eight);
    e.set_compression(png::Compression::Fast);
    let mut w=e.write_header()?;
    if host.stride==host.width as usize*4 {w.write_image_data(&host.pixels)?;}
    else {let mut v=Vec::new(); for row in host.pixels.chunks(host.stride).take(host.height as usize){v.extend_from_slice(&row[..host.width as usize*4]);} w.write_image_data(&v)?;}
    Ok(())
}
fn main()->anyhow::Result<()> {
    let a:Vec<String>=std::env::args().collect();
    let file=&a[1]; let scale:f64=a[2].parse()?; let workers:usize=a[3].parse()?;
    let mode=&a[4]; let out=&a[5];
    let bytes=std::fs::read(file)?;
    let t=Instant::now();
    let source:Arc<dyn PdfSource>=Arc::new(OwnedBytesSource::new(bytes));
    let snap=DocumentSnapshot::open(source,DocumentLimits::default())?;
    let count=snap.page_count();
    let selected:Vec<u32>=a.get(6).map(|s|s.split(',').filter_map(|s|s.parse().ok()).collect()).unwrap_or_else(||(0..count).collect());
    let open_ms=t.elapsed().as_secs_f64()*1000.;
    let mut rows=Vec::new();
    if mode=="sample" {
        let backend=CpuBackend::default();
        let indices:Vec<u32>=a.get(6).map(|s|s.split(',').filter_map(|s|s.parse().ok()).collect()).unwrap_or(vec![0,count/2,count-1]);
        for p in indices {
            for r in 0..26 {
                let t=Instant::now();
                let result=build_request(&snap,PageIndex(p),scale,&mut ParseContext::new()).and_then(|req|backend.render_to_host(&req).map(|x|x.0));
                let ms=t.elapsed().as_secs_f64()*1000.;
                std::hint::black_box(&result);
                rows.push(json!({"page":p,"sample":r,"phase":if r==0 {"first"}else if r<6 {"warmup"}else{"warm"},"ms":ms,"ok":result.is_ok()}));
            }
        }
        println!("{}",json!({"open_ms":open_ms,"count":count,"rows":rows})); return Ok(());
    }
    let backend:Arc<dyn RenderBackend>=Arc::new(CpuBackend::default());
    let opts=SchedulerOptions{compile_workers:if workers==1 {1}else{(workers/3).max(1)},render_workers:if workers==1 {1}else{workers-(workers/3).max(1)},..Default::default()};
    let scheduler=RenderScheduler::new(backend,opts.clone());
    let t=Instant::now(); let mut ok=0; let mut degraded=0;
    if workers==1 {
        let backend=CpuBackend::default();
        for &p in &selected {
            let t=Instant::now();
            let result=build_request(&snap,PageIndex(p),scale,&mut ParseContext::new()).and_then(|req|backend.render_to_host(&req));
            let ms=t.elapsed().as_secs_f64()*1000.;
            match result {Ok((h,stats))=>{ok+=1;degraded+=stats.degraded_draws;let mut row=json!({"page":p,"ms":ms,"ok":true,"width":h.width,"height":h.height,"degraded":stats.degraded_draws});if mode=="verify"{row["hash"]=json!(fingerprint(&h.pixels));}rows.push(row);if mode=="dump" {save(out,p,&h)?;}std::hint::black_box(h);},Err(e)=>rows.push(json!({"page":p,"ms":ms,"ok":false,"error":e.to_string()}))}
        }
    }else{
        scheduler.render_range(&snap,0..selected.len() as u32,&|s,p|build_request(s,PageIndex(selected[p.0 as usize]),scale,&mut ParseContext::new()),None,&mut |result|{
            match result.result {Ok(page)=>{let p=selected[result.page.0 as usize];if mode=="dump" {if let Some(h)=page.as_host(){match save(out,p,h){Ok(())=>rows.push(json!({"page":p,"ok":true,"width":h.width,"height":h.height})),Err(e)=>{rows.push(json!({"page":p,"ok":false,"error":e.to_string()}));return;}}}}ok+=1;if mode=="verify" {if let Some(h)=page.as_host(){rows.push(json!({"page":p,"ok":true,"width":h.width,"height":h.height,"hash":fingerprint(&h.pixels)}));}}std::hint::black_box(page);},Err(e)=>rows.push(json!({"page":selected[result.page.0 as usize],"ok":false,"error":e.to_string()}))}
        });
    }
    let wall_ms=t.elapsed().as_secs_f64()*1000.;
    println!("{}",json!({"open_ms":open_ms,"count":count,"selected_count":selected.len(),"ok":ok,"degraded":degraded,"wall_ms":wall_ms,"compile_workers":opts.compile_workers,"render_workers":opts.render_workers,"rows":rows,"includes_png":mode=="dump"}));
    Ok(())
}
